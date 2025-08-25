import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from scipy.stats import skew, kurtosis
from tqdm import tqdm

def extract_features(image_path, mask_path):
    """
    手动从 NIfTI 文件中提取指定的影像组学特征。
    """
    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
    except Exception as e:
        print(f"警告: 无法读取 {image_path} 或 {mask_path}。跳过此文件。错误: {e}")
        return None

    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    lung_voxels = image_array[mask_array > 0]

    if len(lung_voxels) == 0:
        return None

    total_voxels = len(lung_voxels)

    # 获取体素的物理大小（mm^3）
    spacing = image.GetSpacing()
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    # 计算所有特征
    features = {}

    # 肺体积 (L)
    features['lung_volume_liters'] = (total_voxels * voxel_volume_mm3) / 1000000.0

    # LAA 百分比
    features['LAA910'] = np.sum(lung_voxels < -910) / total_voxels * 100
    features['LAA950'] = np.sum(lung_voxels < -950) / total_voxels * 100
    features['LAA980'] = np.sum(lung_voxels < -980) / total_voxels * 100

    # 一阶统计特征
    features['Mean'] = np.mean(lung_voxels)
    features['Median'] = np.median(lung_voxels)
    features['Skewness'] = skew(lung_voxels)
    features['Kurtosis'] = kurtosis(lung_voxels)
    features['Variance'] = np.var(lung_voxels)
    features['MeanAbsoluteDeviation'] = np.mean(np.abs(lung_voxels - features['Mean']))
    q1, q3 = np.percentile(lung_voxels, [25, 75])
    features['InterquartileRange'] = q3 - q1

    # 百分位数
    features['10Percentile'] = np.percentile(lung_voxels, 10)
    features['90Percentile'] = np.percentile(lung_voxels, 90)

    return features

# --------------------------
# 主程序
# --------------------------
data_dir = './datasets_corp/datasets'
output_csv = './datasets_corp/lung_features.csv'

all_features = []

file_list = sorted(os.listdir(data_dir))
file_pairs = {}

for filename in file_list:
    if filename.endswith('.nii.gz'):
        file_id = filename.split('.')[0].replace('-', '')
        if '_mask' not in filename:
            file_pairs[file_id] = {'image': os.path.join(data_dir, filename), 'mask': None}

for filename in file_list:
    if filename.endswith('_mask.nii.gz'):
        file_id = filename.split('_mask')[0].replace('-', '')
        if file_id in file_pairs:
            file_pairs[file_id]['mask'] = os.path.join(data_dir, filename)

print("开始遍历文件夹并提取特征...")

# 使用 tqdm 包装 file_pairs.items() 以显示进度条
for file_id, paths in tqdm(file_pairs.items(), desc="处理文件"):
    if paths['image'] and paths['mask']:
        features = extract_features(paths['image'], paths['mask'])
        if features:
            features['file_id'] = file_id
            all_features.append(features)
    else:
        # 仅在非交互式模式下打印警告，避免进度条混乱
        print(f"警告: 未找到 {file_id} 的完整文件对，跳过。")
        pass

if all_features:
    df = pd.DataFrame(all_features)
    cols = ['file_id'] + [col for col in df.columns if col != 'file_id']
    df = df[cols]
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n所有特征已成功保存到 {output_csv}")
else:
    print("\n没有成功提取到任何特征。请检查文件路径和命名格式。")