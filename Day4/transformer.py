import torch
from torch import nn

# einops库提供更直观的张量操作（reshape/transpose等）
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


# 定义前馈神经网络模块
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, hidden_dim),  # 全连接层扩展维度
            nn.GELU(),  # GELU激活函数（比ReLU更平滑）
            nn.Dropout(dropout),  # 随机失活防止过拟合
            nn.Linear(hidden_dim, dim),  # 全连接层压缩回原始维度
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)  # 顺序通过各层


# 定义自注意力机制模块
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 计算多头注意力的总维度
        project_out = not (heads == 1 and dim_head == dim)  # 是否需要输出投影

        self.heads = heads  # 注意力头的数量
        self.scale = dim_head ** -0.5  # 缩放因子（1/√d_k）

        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.attend = nn.Softmax(dim=-1)  # 注意力分数归一化
        self.dropout = nn.Dropout(dropout)  # 注意力权重随机失活

        # 将输入投影到Q、K、V（使用单个线性层更高效）
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 输出投影层（如果需要）
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # 无操作层

    def forward(self, x):
        x = self.norm(x)  # 先归一化

        # 分割Q、K、V [batch, seq_len, inner_dim*3] -> 3个[batch, seq_len, inner_dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # 重排列为多头形式 [batch, seq_len, heads, dim_head] -> [batch, heads, seq_len, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 计算点积注意力分数 [batch, heads, q_len, k_len]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 注意力权重归一化
        attn = self.attend(dots)
        attn = self.dropout(attn)  # 随机失活

        # 加权求和 [batch, heads, seq_len, dim_head]
        out = torch.matmul(attn, v)

        # 合并多头 [batch, seq_len, heads*dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  # 输出投影


# 定义Transformer模块（包含多个Attention和FFN层）
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # 堆叠指定数量的Transformer层
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        # 残差连接 + 层归一化（已在各子模块内部实现）
        for attn, ff in self.layers:
            x = attn(x) + x  # 自注意力
            x = ff(x) + x  # 前馈网络
        return x


# 定义Vision Transformer模型
class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert (seq_len % patch_size) == 0  # 确保序列长度能被patch大小整除

        num_patches = seq_len // patch_size  # 计算patch数量
        patch_dim = channels * patch_size  # 每个patch的维度（展平后）

        # 将序列分割为patch并嵌入到指定维度
        self.to_patch_embedding = nn.Sequential(
            # [batch, channels, seq_len] -> [batch, num_patches, patch_dim]
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),  # 对每个patch归一化
            nn.Linear(patch_dim, dim),  # 线性投影到模型维度
            nn.LayerNorm(dim),  # 再次归一化
        )

        # 可学习的位置编码 [1, num_patches+1, dim]（+1是cls_token）
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # 可学习的分类token [dim]
        self.cls_token = nn.Parameter(torch.randn(dim))

        self.dropout = nn.Dropout(emb_dropout)  # 嵌入层随机失活

        # Transformer编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)  # 输出类别分数
        )

    def forward(self, series):
        # 1. 分割并嵌入patch
        x = self.to_patch_embedding(series)  # [batch, num_patches, dim]
        b, n, _ = x.shape  # batch_size, num_patches, dim

        # 2. 添加cls_token（复制batch份）
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)  # [batch, dim]

        # 3. 合并cls_token和patch嵌入 [batch, num_patches+1, dim]
        x, ps = pack([cls_tokens, x], 'b * d')

        # 4. 添加位置编码（截取到实际长度）
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # 5. 通过Transformer编码器
        x = self.transformer(x)

        # 6. 提取cls_token作为分类特征 [batch, dim]
        cls_tokens, _ = unpack(x, ps, 'b * d')

        # 7. 分类预测
        return self.mlp_head(cls_tokens)


if __name__ == '__main__':
    # 实例化ViT模型（用于时间序列分类）
    v = ViT(
        seq_len=256,  # 输入序列长度
        patch_size=16,  # 每个patch的长度
        num_classes=1000,  # 输出类别数
        dim=1024,  # 模型维度
        depth=6,  # Transformer层数
        heads=8,  # 注意力头数
        mlp_dim=2048,  # FFN隐藏层维度
        dropout=0.1,  # Transformer内部dropout
        emb_dropout=0.1  # 嵌入层dropout
    )

    # 测试输入 [batch, channels, seq_len]
    time_series = torch.randn(4, 3, 256)
    logits = v(time_series)  # 前向传播 [4, 1000]
    print(logits.shape)  # 输出预测结果的形状