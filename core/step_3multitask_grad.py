import os
import glob
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from monai.data import DataLoader, Dataset, PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    ToTensord,
    Orientationd,
    Resized,
    MaskIntensityd,
    CropForegroundd,
    Spacingd
)
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import json
import random

# 导入您的多任务模型
import MultitaskModel as model

# --- 补丁代码 (保持不变) ---
orig_torch_load = torch.load
def torch_wrapper(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_torch_load(*args, **kwargs)
torch.load = torch_wrapper


# --- 配置 (与训练脚本保持一致) ---
task = '3multitask'  # 保持与多任务训练脚本一致
image_network_name = 'DenseNet'
DATA_ROOT_PATH = "./datasets_corp/datasets"
TEST_CSV_FILE_PATH = "./datasets_corp/测试集.csv"
MODEL_OUTPUT_DIR = f"./model/{task}/{image_network_name}"
GRAD_CAM_OUTPUT_DIR = f"./report/{task}/{image_network_name}/grad"

root_dir = "./cache"
persistent_cache = os.path.join(root_dir, "test_grad")

BATCH_SIZE = 1
NUM_WORKERS = 4
HU_LOWER_BOUND = -950
HU_UPPER_BOUND = -750
TARGET_SIZE = (128, 128, 128)
TARGET_SPACING = (1.0, 1.0, 1.0)
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# --- 定义特征和目标 (与训练脚本保持一致) ---
ONE_HOT_INPUT_FEATURES = ['性别', '吸烟史']
NORMAL_CSV_INPUT_FEATURES = ['年龄', '身高', '体重', 'lung_volume_liters', 'LAA910', 'LAA950',
                             'LAA980', 'Mean', 'Median', 'Skewness', 'Kurtosis', 'Variance',
                             'MeanAbsoluteDeviation', 'InterquartileRange', '10Percentile',
                             '90Percentile']
CLASSIFY_TARGET_COLUMN = '分类'
REGRESSION_TARGET_COLUMNS = ['FEV1', 'FVC'] # 确保与训练脚本一致

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# --- Grad-CAM 函数 ---
def get_grad_cam(model, image_tensor, csv_input_tensor, target_index=0, target_layer=None):
    model.eval()
    image_tensor.requires_grad_(True)
    feature_map = None
    gradient = None

    def forward_hook(module, input, output):
        nonlocal feature_map
        feature_map = output if not isinstance(output, tuple) else output[-1]

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradient
        gradient = grad_output[0]

    if target_layer is None:
        raise ValueError("必须指定用于 Grad-CAM 的目标层。")

    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_full_backward_hook(backward_hook)

    # 适配多任务模型的输出
    regression_outputs, classification_outputs, _ = model(image_tensor, csv_input_tensor)

    # 针对分类任务，我们对目标类别的logit进行反向传播
    target_output = classification_outputs[:, target_index]
    target_output.sum().backward(retain_graph=True)

    hook_forward.remove()
    hook_backward.remove()

    if feature_map is None or gradient is None:
        raise ValueError("无法获取 Grad-CAM 所需的特征图或梯度。请检查目标层是否正确。")

    weights = F.adaptive_avg_pool3d(gradient, 1)
    cam = torch.sum(weights * feature_map, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=image_tensor.shape[2:], mode='trilinear', align_corners=False)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().squeeze().numpy()

# --- 辅助绘图函数 (三视图) ---
def plot_and_save_cam_triple(original_image, cam_maps, class_names, output_path, sample_id):
    axial_slice_idx = original_image.shape[2] // 2
    coronal_slice_idx = original_image.shape[1] // 2
    sagittal_slice_idx = original_image.shape[0] // 2

    num_classes = len(cam_maps)
    fig, axes = plt.subplots(num_classes, 3, figsize=(18, 6 * num_classes))

    if num_classes == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_classes):
        cam = cam_maps[i]
        class_name = class_names[i]

        # Axial
        axes[i, 0].imshow(original_image[:, :, axial_slice_idx].T, cmap='gray', origin='lower')
        im0 = axes[i, 0].imshow(cam[:, :, axial_slice_idx].T, cmap='viridis', alpha=0.7, origin='lower')
        axes[i, 0].set_title(f"{class_name} Axial (z={axial_slice_idx})", fontsize=10)
        axes[i, 0].axis('off')
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Coronal
        axes[i, 1].imshow(original_image[:, coronal_slice_idx, :].T, cmap='gray', origin='lower')
        im1 = axes[i, 1].imshow(cam[:, coronal_slice_idx, :].T, cmap='viridis', alpha=0.7, origin='lower')
        axes[i, 1].set_title(f"{class_name} Coronal (y={coronal_slice_idx})", fontsize=10)
        axes[i, 1].axis('off')
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Sagittal
        axes[i, 2].imshow(original_image[sagittal_slice_idx, :, :].T, cmap='gray', origin='lower')
        im2 = axes[i, 2].imshow(cam[sagittal_slice_idx, :, :].T, cmap='viridis', alpha=0.7, origin='lower')
        axes[i, 2].set_title(f"{class_name} Sagittal (x={sagittal_slice_idx})", fontsize=10)
        axes[i, 2].axis('off')
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Sample {sample_id} - Multi-class Grad-CAM", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

# --- 主函数 ---
def main():
    os.makedirs(GRAD_CAM_OUTPUT_DIR, exist_ok=True)
    print(f"Grad-CAM 输出文件夹 '{GRAD_CAM_OUTPUT_DIR}' 已创建或已存在。")

    csv_input_scaler_path = os.path.join(MODEL_OUTPUT_DIR, "input_scaler.pkl")
    regression_target_scaler_path = os.path.join(MODEL_OUTPUT_DIR, "regression_target_scaler.pkl")
    label_encoder_path = os.path.join(MODEL_OUTPUT_DIR, "label_encoder.pkl")
    if not os.path.exists(csv_input_scaler_path) or not os.path.exists(label_encoder_path) or not os.path.exists(regression_target_scaler_path):
        print("错误: 找不到必要的标准化器或标签编码器文件。")
        return

    print("\n--- 加载模型、标准化器和标签编码器 ---")
    with open(csv_input_scaler_path, 'rb') as f:
        csv_input_scaler = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    num_csv_features = csv_input_scaler.n_features_in_
    num_regression_targets = len(REGRESSION_TARGET_COLUMNS)

    # 实例化在此文件中定义的多任务模型类
    model_to_load = model.CrossModalNet(
        image_network_name=image_network_name,
        num_image_channels=1,
        num_csv_input_features=num_csv_features,
        num_targets=num_regression_targets,
        num_classes=num_classes
    ).to(DEVICE)

    model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型权重文件 '{model_path}'。")
        return

    model_to_load.load_state_dict(torch.load(model_path))
    model_to_load.eval()

    # 动态确定 Grad-CAM 目标层
    target_layer = None
    if image_network_name == 'ResNet':
        target_layer = model_to_load.image_feature_network.layer4[-1].conv2
    elif image_network_name == 'DenseNet':
        target_layer = model_to_load.image_feature_network.features.denseblock4.denselayer16.layers.conv2
    elif image_network_name == 'SeResNet':
        target_layer = model_to_load.image_feature_network.layer4[-1].conv3

    if target_layer is None:
        print(f"警告: 无法为指定的图像网络 '{image_network_name}' 找到预设的 Grad-CAM 目标层。")
        return

    # 使用固定的病例ID列表
    fixed_sample_ids = [16, 87, 122, 170, 430, 435, 437, 1040, 1350, 1728]

    test_raw_df = pd.read_csv(TEST_CSV_FILE_PATH, index_col='序号')
    test_files = []
    search_pattern = os.path.join(DATA_ROOT_PATH, "*.nii.gz")
    all_nii_files = glob.glob(search_pattern)
    image_files = {int(os.path.basename(f).split('.')[0]): f for f in all_nii_files if '_mask' not in f}
    mask_files = {int(os.path.basename(f).split('_')[0]): f for f in all_nii_files if '_mask' in f}

    one_hot_cols = [col for col in csv_input_scaler.feature_names_in_ if any(f in col for f in ONE_HOT_INPUT_FEATURES)]

    # 在这里进行一次性的数据预处理，避免在循环中重复计算
    preprocessed_df = pd.get_dummies(test_raw_df, columns=ONE_HOT_INPUT_FEATURES, prefix=ONE_HOT_INPUT_FEATURES, dummy_na=False)
    missing_cols = set(one_hot_cols) - set(preprocessed_df.columns)
    for c in missing_cols:
        preprocessed_df[c] = 0
    preprocessed_df = preprocessed_df[one_hot_cols + list(preprocessed_df.columns.drop(one_hot_cols))]
    final_csv_input_features = NORMAL_CSV_INPUT_FEATURES + one_hot_cols
    preprocessed_df[final_csv_input_features] = csv_input_scaler.transform(preprocessed_df[final_csv_input_features])
    preprocessed_df[REGRESSION_TARGET_COLUMNS] = pd.read_csv(TEST_CSV_FILE_PATH, index_col='序号')[REGRESSION_TARGET_COLUMNS].values
    for idx in fixed_sample_ids:
        image_path = image_files.get(idx)
        mask_path = mask_files.get(idx)
        if image_path and mask_path:
            test_files.append({
                "image": image_path,
                "mask": mask_path,
                "csv_input_features": preprocessed_df.loc[idx, final_csv_input_features].values.astype(np.float32),
                "id": str(idx)
            })
        else:
            print(f"警告: 找不到序号为 {idx} 的图像或掩码文件，将跳过此样本。")

    if not test_files:
        print("错误: 找不到任何匹配的测试集数据。")
        return

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            ScaleIntensityRanged(keys=["image"], a_min=HU_LOWER_BOUND, a_max=HU_UPPER_BOUND, b_min=0.0, b_max=1.0, clip=True),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(keys=["image", "mask"], pixdim=TARGET_SPACING, mode=("bilinear", "nearest")),
            CropForegroundd(keys=["image", "mask"], source_key="mask"),
            Resized(keys=["image", "mask"], spatial_size=TARGET_SIZE, mode=("bilinear", "nearest")),
            ToTensord(keys=["image", "mask", "csv_input_features"]),
        ]
    )

    test_ds = PersistentDataset(data=test_files, transform=test_transforms, cache_dir=persistent_cache)
    print(f"\n--- 为 {len(test_ds)} 个固定样本生成 Grad-CAM 热力图（三视图） ---")

    for i in tqdm(range(len(test_ds)), desc="Generating Grad-CAM for each sample"):
        data = test_ds[i]
        sample_id = data['id']
        image_tensor = data['image'].unsqueeze(0).to(DEVICE)
        mask_tensor = data['mask'].unsqueeze(0).to(DEVICE)
        csv_input_tensor = data['csv_input_features'].unsqueeze(0).to(DEVICE)

        cam_maps = []
        for class_idx in range(num_classes):
            cam = get_grad_cam(model_to_load, image_tensor, csv_input_tensor, target_index=class_idx, target_layer=target_layer)
            mask_np = mask_tensor.cpu().squeeze().numpy()
            cam_masked = cam * (mask_np > 0)
            cam_masked = cam_masked / (cam_masked.max() + 1e-8)
            cam_maps.append(cam_masked)

        original_image_np = data['image'].squeeze().cpu().numpy()

        output_path = os.path.join(GRAD_CAM_OUTPUT_DIR, f"{sample_id}_Multi-class_Attention_Map.png")
        plot_and_save_cam_triple(
            original_image_np,
            cam_maps,
            class_names,
            output_path,
            sample_id
        )

    print("\n所有 Grad-CAM 热力图已生成并保存。")

if __name__ == "__main__":
    main()