import rasterio
import numpy as np
import matplotlib.pyplot as plt


def process_sentinel2(tif_file):
    """
    处理哨兵2号影像数据，生成RGB图像并显示

    参数:
        tif_file: 输入的TIFF文件路径

    返回:
        rgb_normalized: 处理后的RGB图像数组(0-255)
    """
    # 1. 打开TIFF文件并读取波段数据
    with rasterio.open(tif_file) as src:
        bands = src.read()  # 形状为 (5, height, width)

        # 验证波段数量
        if bands.shape[0] != 5:
            raise ValueError("The input TIFF file should contain 5 bands(B02,B03,B04,B08,B12)")

        # 获取图像元数据(用于后续保存)
        profile = src.profile

    # 2. 分配波段(假设顺序为B02,B03,B04,B08,B12)
    blue = bands[0].astype(float)  # B02 - 蓝(492nm)
    green = bands[1].astype(float)  # B03 - 绿(560nm)
    red = bands[2].astype(float)  # B04 - 红(665nm)
    nir = bands[3].astype(float)  # B08 - 近红外(842nm)
    swir = bands[4].astype(float)  # B12 - 短波红外(2190nm)

    # 3. 真彩色合成(RGB三通道)
    rgb_stack = np.dstack((red, green, blue))  # 形状(height, width, 3)

    # 4. 数据压缩(0-10000 → 0-255)
    # 方法1: 线性拉伸(基于每个通道的2%-98%百分位，避免异常值影响)
    def percentile_stretch(band, lower=2, upper=98):
        # 计算百分位值
        p_low, p_high = np.percentile(band, (lower, upper))
        # 线性拉伸
        stretched = np.clip((band - p_low) / (p_high - p_low) * 255, 0, 255)
        return stretched.astype(np.uint8)

    # 对每个通道分别进行拉伸
    rgb_normalized = np.zeros_like(rgb_stack, dtype=np.uint8)
    for i in range(3):
        rgb_normalized[:, :, i] = percentile_stretch(rgb_stack[:, :, i])

    # 方法2(备选): 简单线性归一化(原始代码使用的方法)
    # array_min, array_max = rgb_stack.min(), rgb_stack.max()
    # rgb_normalized = ((rgb_stack - array_min) / (array_max - array_min)) * 255
    # rgb_normalized = rgb_normalized.astype(np.uint8)

    # 5. 显示结果
    plt.figure(figsize=(12, 6))

    # 显示原始波段组合(未拉伸)
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_stack / 10000)  # 简单除以10000显示
    plt.title("Original image (not stretched)")
    plt.axis('off')

    # 显示处理后的RGB图像
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_normalized)
    plt.title("Processed RGB (0-255)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return rgb_normalized


# 使用示例
if __name__ == "__main__":
    # 替换为您的TIFF文件路径
    input_tif = "2019_1101_nofire_B2348_B12_10m_roi.tif"

    try:
        result = process_sentinel2(input_tif)
        print("Processing completed! Image size:", result.shape)
    except Exception as e:
        print(f"Processing failed: {e}")