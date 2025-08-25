import os
import glob
import torch
import numpy as np
import pandas as pd
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
    RandFlipd, RandGaussianNoised, RandAdjustContrastd, Spacingd,
    RandScaleIntensityd, RandShiftIntensityd,
    MaskIntensityd, CropForegroundd, RandGaussianSmoothd, RandZoomd
)
from monai.utils import set_determinism
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import json
import pickle
import random

# 导入您的多任务模型
import MultitaskModel as model

# --- 补丁代码 (保持不变) ---
orig_torch_load = torch.load
def torch_wrapper(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_torch_load(*args, **kwargs)
torch.load = torch_wrapper

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']
# --- 补丁代码结束 ---

# --- 配置 (Configuration) ---
task = '3multitask' # 任务名称更改，更具描述性
image_network_name = 'DenseNet'
DATA_ROOT_PATH = "./datasets_corp/datasets"
CSV_DIR = "./datasets_corp"
MODEL_OUTPUT_DIR = f"./model/{task}/{image_network_name}"
root_dir = "./cache"
persistent_cache = root_dir
os.makedirs(persistent_cache, exist_ok=True)

BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
MAX_EPOCHS = 50

HU_LOWER_BOUND = -950
HU_UPPER_BOUND = -750
RANDOM_STATE = 42
TARGET_SIZE = (128, 128, 128)
TARGET_SPACING = (1.0, 1.0, 1.0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# --- 定义特征处理方式 ---
ONE_HOT_INPUT_FEATURES = ['性别', '吸烟史']
BINNER_INPUT_FEATURES = []
NORMAL_CSV_INPUT_FEATURES = ['年龄', '身高', '体重', 'lung_volume_liters', 'LAA910', 'LAA950',
                             'LAA980', 'Mean', 'Median', 'Skewness', 'Kurtosis', 'Variance',
                             'MeanAbsoluteDeviation', 'InterquartileRange', '10Percentile',
                             '90Percentile']

# 您的分类目标列和新增的回归目标列
CLASSIFY_TARGET_COLUMN = '分类'
REGRESSION_TARGET_COLUMNS = ['FEV1', 'FVC']

# --- 数据加载与准备  ---
def load_and_prepare_data_multimodal(data_root_path, csv_dir):
    print(f"当前工作目录: {os.getcwd()}")
    print(f"尝试搜索NIfTI文件的根目录: {os.path.abspath(data_root_path)}")

    try:
        train_df_raw = pd.read_csv(os.path.join(csv_dir, '训练集.csv'))
        val_df_raw = pd.read_csv(os.path.join(csv_dir, '验证集.csv'))
        test_df_raw = pd.read_csv(os.path.join(csv_dir, '测试集.csv'))

        for df in [train_df_raw, val_df_raw, test_df_raw]:
            if '序号' not in df.columns:
                raise ValueError(f"CSV 文件中缺少 '序号' 列。")
            if CLASSIFY_TARGET_COLUMN not in df.columns:
                raise ValueError(f"CSV 文件中缺少分类目标列 '{CLASSIFY_TARGET_COLUMN}'。")
            for col in REGRESSION_TARGET_COLUMNS:
                if col not in df.columns:
                    raise ValueError(f"CSV 文件中缺少回归目标列 '{col}'。")
            df['序号'] = pd.to_numeric(df['序号'], errors='coerce')
            df.dropna(subset=['序号'], inplace=True)
            df['序号'] = df['序号'].astype(pd.Int64Dtype())
            df.set_index('序号', inplace=True)

        all_defined_features = ONE_HOT_INPUT_FEATURES + BINNER_INPUT_FEATURES + NORMAL_CSV_INPUT_FEATURES + [CLASSIFY_TARGET_COLUMN] + REGRESSION_TARGET_COLUMNS
        for df, name in zip([train_df_raw, val_df_raw, test_df_raw], ['训练', '验证', '测试']):
            missing_initial_cols = [col for col in all_defined_features if col not in df.columns]
            if missing_initial_cols:
                raise ValueError(f"{name}集 CSV 中缺少定义的特征列: {missing_initial_cols}")

    except FileNotFoundError:
        print(f"错误: 未找到训练/验证/测试 CSV 文件。请检查路径: {csv_dir}")
        return [], [], [], None, None, None, None, None
    except ValueError as e:
        print(f"CSV 数据错误: {e}")
        return [], [], [], None, None, None, None, None
    except Exception as e:
        print(f"加载或处理 CSV 时发生意外错误: {e}")
        return [], [], [], None, None, None, None, None

    # --- 1. 匹配影像文件和CSV数据 ---
    print("\n--- 匹配影像文件和CSV数据 ---")
    search_pattern = os.path.join(data_root_path, "*.nii.gz")
    all_nii_files = glob.glob(search_pattern)
    image_files = {int(os.path.basename(f).split('.')[0]): f for f in all_nii_files if '_mask' not in f}
    mask_files = {int(os.path.basename(f).split('_')[0]): f for f in all_nii_files if '_mask' in f}

    def match_data(df):
        matched_data = []
        for numeric_id in sorted(df.index.tolist()):
            image_path = image_files.get(numeric_id)
            mask_path = mask_files.get(numeric_id)
            if image_path and mask_path:
                matched_data.append({
                    "id": str(numeric_id),
                    "image": image_path,
                    "mask": mask_path,
                    "csv_data": df.loc[numeric_id]
                })
        return matched_data

    train_data_list_raw = match_data(train_df_raw)
    val_data_list_raw = match_data(val_df_raw)
    test_data_list_raw = match_data(test_df_raw)

    if not train_data_list_raw or not val_data_list_raw or not test_data_list_raw:
        print("错误: 找不到任何匹配的影像、掩码和CSV数据。")
        return [], [], [], None, None, None, None, None

    train_df = pd.DataFrame([d['csv_data'] for d in train_data_list_raw], index=[d['id'] for d in train_data_list_raw])
    val_df = pd.DataFrame([d['csv_data'] for d in val_data_list_raw], index=[d['id'] for d in val_data_list_raw])
    test_df = pd.DataFrame([d['csv_data'] for d in test_data_list_raw], index=[d['id'] for d in test_data_list_raw])

    print(f"\n训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(val_df)}")
    print(f"测试集样本数: {len(test_df)}")

    # --- 2. 数据预处理 (标准化与独热编码) ---
    def preprocess_df(df, is_training_set, csv_input_scaler=None, regression_target_scaler=None, label_encoder=None, one_hot_cols=None):
        preprocessed_df = df.copy()

        # 对分类目标进行标签编码
        if is_training_set:
            label_encoder = LabelEncoder()
            preprocessed_df[CLASSIFY_TARGET_COLUMN] = label_encoder.fit_transform(preprocessed_df[CLASSIFY_TARGET_COLUMN])
        else:
            preprocessed_df[CLASSIFY_TARGET_COLUMN] = label_encoder.transform(preprocessed_df[CLASSIFY_TARGET_COLUMN])

        # 对指定类别特征应用独热编码
        if is_training_set:
            preprocessed_df = pd.get_dummies(preprocessed_df, columns=ONE_HOT_INPUT_FEATURES, prefix=ONE_HOT_INPUT_FEATURES, dummy_na=False)
            one_hot_cols = [col for col in preprocessed_df.columns if any(p in col for p in ONE_HOT_INPUT_FEATURES)]
        else:
            preprocessed_df = pd.get_dummies(preprocessed_df, columns=ONE_HOT_INPUT_FEATURES, prefix=ONE_HOT_INPUT_FEATURES, dummy_na=False)
            missing_cols = set(one_hot_cols) - set(preprocessed_df.columns)
            for c in missing_cols:
                preprocessed_df[c] = 0
            preprocessed_df = preprocessed_df[one_hot_cols + list(preprocessed_df.columns.drop(one_hot_cols))]

        final_csv_input_features = NORMAL_CSV_INPUT_FEATURES + one_hot_cols
        final_csv_input_features = [col for col in final_csv_input_features if col in preprocessed_df.columns]

        # 标准化回归目标列
        if is_training_set:
            regression_target_scaler = StandardScaler()
            preprocessed_df[REGRESSION_TARGET_COLUMNS] = regression_target_scaler.fit_transform(preprocessed_df[REGRESSION_TARGET_COLUMNS])
        else:
            preprocessed_df[REGRESSION_TARGET_COLUMNS] = regression_target_scaler.transform(preprocessed_df[REGRESSION_TARGET_COLUMNS])

        # 标准化普通CSV输入特征
        if is_training_set:
            csv_input_scaler = StandardScaler()
            preprocessed_df[final_csv_input_features] = csv_input_scaler.fit_transform(preprocessed_df[final_csv_input_features])
        else:
            preprocessed_df[final_csv_input_features] = csv_input_scaler.transform(preprocessed_df[final_csv_input_features])

        return preprocessed_df, csv_input_scaler, regression_target_scaler, label_encoder, final_csv_input_features, one_hot_cols

    train_processed_df, csv_input_scaler, regression_target_scaler, label_encoder, final_csv_input_features, one_hot_cols = preprocess_df(train_df, True)
    val_processed_df, _, _, _, _, _ = preprocess_df(val_df, False, csv_input_scaler, regression_target_scaler, label_encoder, one_hot_cols)
    test_processed_df, _, _, _, _, _ = preprocess_df(test_df, False, csv_input_scaler, regression_target_scaler, label_encoder, one_hot_cols)

    scaler_output_dir = MODEL_OUTPUT_DIR
    os.makedirs(scaler_output_dir, exist_ok=True)

    csv_input_scaler_path = os.path.join(scaler_output_dir, "input_scaler.pkl")
    with open(csv_input_scaler_path, 'wb') as f:
        pickle.dump(csv_input_scaler, f)
    print(f"CSV 输入特征标准化器已保存到: {csv_input_scaler_path}")

    regression_target_scaler_path = os.path.join(scaler_output_dir, "regression_target_scaler.pkl")
    with open(regression_target_scaler_path, 'wb') as f:
        pickle.dump(regression_target_scaler, f)
    print(f"回归目标标准化器已保存到: {regression_target_scaler_path}")

    label_encoder_path = os.path.join(scaler_output_dir, "label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"分类目标标签编码器已保存到: {label_encoder_path}")

    train_files = []
    val_files = []
    test_files = []

    def prepare_file_list(data_list_raw, processed_df, features_list):
        file_list = []
        for d in data_list_raw:
            pid = d['id']
            file_list.append({
                "image": d['image'],
                "mask": d['mask'],
                "csv_input_features": processed_df.loc[pid, features_list].values.astype(np.float32),
                "classification_targets": processed_df.loc[pid, CLASSIFY_TARGET_COLUMN],
                "regression_targets": processed_df.loc[pid, REGRESSION_TARGET_COLUMNS].values.astype(np.float32),
                "id": pid
            })
        return file_list

    train_files = prepare_file_list(train_data_list_raw, train_processed_df, final_csv_input_features)
    val_files = prepare_file_list(val_data_list_raw, val_processed_df, final_csv_input_features)
    test_files = prepare_file_list(test_data_list_raw, test_processed_df, final_csv_input_features)

    return train_files, val_files, test_files, csv_input_scaler, regression_target_scaler, label_encoder, final_csv_input_features


# --- 数据预处理 Transforms (保持不变) ---
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_LOWER_BOUND,
            a_max=HU_UPPER_BOUND,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=TARGET_SPACING, mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "mask"], source_key="mask"),
        Resized(
            keys=["image", "mask"],
            spatial_size=TARGET_SIZE,
            mode=("trilinear", "nearest")
        ),
        RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
        RandGaussianNoised(keys="image",  std=0.01,prob=0.5),
        RandAdjustContrastd(keys="image", gamma=(0.7, 1.3),prob=0.5),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        MaskIntensityd(keys="image", mask_key="mask"),
        # 新增的ToTensord键
        ToTensord(keys=["image", "classification_targets", "regression_targets", "csv_input_features"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_LOWER_BOUND,
            a_max=HU_UPPER_BOUND,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=TARGET_SPACING, mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "mask"], source_key="mask"),
        Resized(
            keys=["image", "mask"],
            spatial_size=TARGET_SIZE,
            mode=("trilinear", "nearest")
        ),
        MaskIntensityd(keys="image", mask_key="mask"),
        # 新增的ToTensord键
        ToTensord(keys=["image", "classification_targets", "regression_targets", "csv_input_features"]),
    ]
)

test_transforms = val_transforms


# 修改后的评估函数
def validate_and_report(data_loader, model, classify_loss_function, regression_loss_function, set_name, label_encoder):
    model.eval()
    total_loss = 0
    step = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc=f"评估 {set_name} 周期"):
            step += 1
            inputs, classification_targets, regression_targets, csv_input_features = (
                batch_data["image"].to(DEVICE),
                batch_data["classification_targets"].to(DEVICE),
                batch_data["regression_targets"].to(DEVICE),
                batch_data["csv_input_features"].to(DEVICE)
            )

            # 模型返回回归和分类两个输出
            regression_outputs, classification_outputs, _ = model(inputs, csv_input_features)

            # 计算两个任务的损失
            classify_loss = classify_loss_function(classification_outputs, classification_targets.long())
            regression_loss = regression_loss_function(regression_outputs, regression_targets)

            # 加权求和作为总损失，您可以根据需要调整权重
            loss = classify_loss + regression_loss
            total_loss += loss.item()

            preds = torch.argmax(classification_outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(classification_targets.cpu().numpy())

    total_loss /= step

    all_preds_labels = label_encoder.inverse_transform(all_preds)
    all_targets_labels = label_encoder.inverse_transform(all_targets)

    # 返回准确的报告字典和报告字符串
    report_dict = classification_report(all_targets_labels, all_preds_labels, output_dict=True, zero_division=0)
    report_str = classification_report(all_targets_labels, all_preds_labels, zero_division=0)

    # 额外返回用于最终评估的预测和真实标签
    return total_loss, report_dict, report_str, all_preds, all_targets


def train_model(train_loader, val_loader, test_loader, model, classify_loss_function, regression_loss_function, optimizer,
                max_epochs, model_output_dir, label_encoder):

    best_val_loss = float('inf')
    best_epoch = -1
    best_val_preds = []
    best_val_targets = []
    best_test_preds = []
    best_test_targets = []

    metrics_history_file = os.path.join(model_output_dir, "training_metrics_history.json")
    history_data = {
        'train_loss_history': [],
        'val_loss_history': [],
        'test_loss_history': [],
        'val_report_history': [],
        'test_report_history': []
    }
    print(f"训练历史文件 '{metrics_history_file}' 将在每次训练时被覆盖。")

    for epoch in range(max_epochs):
        print(f"\n--- 周期 {epoch + 1}/{max_epochs} ---")
        model.train()
        epoch_train_loss = 0
        step = 0

        for batch_data in tqdm(train_loader, desc=f"训练周期 {epoch + 1}"):
            step += 1
            inputs, classification_targets, regression_targets, csv_input_features = (
                batch_data["image"].to(DEVICE),
                batch_data["classification_targets"].to(DEVICE),
                batch_data["regression_targets"].to(DEVICE),
                batch_data["csv_input_features"].to(DEVICE)
            )
            optimizer.zero_grad()

            # 模型返回两个输出
            regression_outputs, classification_outputs, _ = model(inputs, csv_input_features)

            # 计算两个任务的损失
            classify_loss = classify_loss_function(classification_outputs, classification_targets.long())
            regression_loss = regression_loss_function(regression_outputs, regression_targets)

            # 总损失
            total_loss = classify_loss + regression_loss
            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()

        epoch_train_loss /= step

        val_loss, val_report_dict, val_report_str, val_preds, val_targets = validate_and_report(val_loader, model, classify_loss_function, regression_loss_function, "验证集", label_encoder)
        test_loss, test_report_dict, test_report_str, test_preds, test_targets = validate_and_report(test_loader, model, classify_loss_function, regression_loss_function, "测试集", label_encoder)

        print(f"  train loss: {epoch_train_loss:.6f}, val loss: {val_loss:.6f}, test loss: {test_loss:.6f}")
        print("\n--- 验证集分类报告 ---")
        print(val_report_str)
        print("\n--- 测试集分类报告 ---")
        print(test_report_str)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_val_preds = val_preds
            best_val_targets = val_targets
            best_test_preds = test_preds
            best_test_targets = test_targets
            torch.save(model.state_dict(), os.path.join(model_output_dir, "best_model.pth"))
            print("  >>> 新的最佳多任务模型已保存！(基于最低验证损失) <<<")

        if best_epoch != -1:
            best_val_report_str = classification_report(
                label_encoder.inverse_transform(best_val_targets),
                label_encoder.inverse_transform(best_val_preds),
                zero_division=0
            )
            print(f"\n--- 历史最佳验证报告 (Epoch {best_epoch}) ---")
            print(best_val_report_str)

            best_test_report_str = classification_report(
                label_encoder.inverse_transform(best_test_targets),
                label_encoder.inverse_transform(best_test_preds),
                zero_division=0
            )
            print("\n--- 对应最佳验证指标时的测试集报告 ---")
            print(best_test_report_str)

        history_data['train_loss_history'].append(epoch_train_loss)
        history_data['val_loss_history'].append(val_loss)
        history_data['test_loss_history'].append(test_loss)
        history_data['val_report_history'].append(val_report_dict)
        history_data['test_report_history'].append(test_report_dict)

        with open(metrics_history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=4, ensure_ascii=False)
        print(f"  训练指标已保存到 {metrics_history_file}")
        print("-" * 30)

    print(f"\n训练完成！最低验证损失: {best_val_loss:.6f} (来自周期 {best_epoch})")

    return history_data, best_epoch


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"全局随机种子已设置为: {seed}")

if __name__ == "__main__":
    set_global_seeds(RANDOM_STATE)
    set_determinism(seed=RANDOM_STATE)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # 更改数据加载函数的返回值
    train_files, val_files, test_files, csv_input_scaler, regression_target_scaler, label_encoder, all_csv_features_list = \
        load_and_prepare_data_multimodal(DATA_ROOT_PATH, CSV_DIR)

    if not train_files or not val_files or not test_files:
        print("无法进行训练、验证或测试，请检查数据加载错误信息。")
        exit()

    num_classes = len(label_encoder.classes_)
    num_regression_targets = len(REGRESSION_TARGET_COLUMNS)
    print(f"\n检测到 {num_classes} 个类别: {label_encoder.classes_}")
    print(f"检测到 {num_regression_targets} 个回归目标: {REGRESSION_TARGET_COLUMNS}")

    print("\n--- 最终送入模型的 CSV 特征列表 ---")
    print(all_csv_features_list)
    print("---------------------------------")

    # 模型初始化时传入回归目标的数量
    model = model.CrossModalNet(
        image_network_name= image_network_name,
        num_image_channels=1,
        num_csv_input_features=len(all_csv_features_list),
        num_targets=num_regression_targets,  # 设置回归头的输出维度
        num_classes=num_classes              # 设置分类头的输出维度
    ).to(DEVICE)

    # 定义两个独立的损失函数
    classify_loss_function = torch.nn.CrossEntropyLoss()
    regression_loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=os.path.join(persistent_cache, "train"))
    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=os.path.join(persistent_cache, "val"))
    test_ds = PersistentDataset(data=test_files, transform=test_transforms, cache_dir=os.path.join(persistent_cache, "test"))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )

    print("--- PersistentDataset(缓存训练/验证/测试数据) 初始化完成。---")

    print("\n--- 开始多任务模型训练 ---")

    # 传递两个损失函数给训练函数
    history_data, best_epoch = train_model(
        train_loader, val_loader, test_loader, model, classify_loss_function, regression_loss_function, optimizer,
        MAX_EPOCHS, MODEL_OUTPUT_DIR, label_encoder
    )

    print("\n--- 训练过程结束 ---")
    print("\n--- 最终指标摘要 ---")

    # 重新加载最佳模型权重
    best_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"\n从 Epoch {best_epoch} 加载最佳模型权重进行最终评估...")
        model.load_state_dict(torch.load(best_model_path))

        # 重新评估以获得准确的最终报告
        _, final_val_report_dict, final_val_report_str, _, _ = validate_and_report(val_loader, model, classify_loss_function, regression_loss_function, "最终验证集", label_encoder)
        _, final_test_report_dict, final_test_report_str, _, _ = validate_and_report(test_loader, model, classify_loss_function, regression_loss_function, "最终测试集", label_encoder)

        print(f"\n--- 基于最低验证损失的最佳周期 (Epoch {best_epoch}) ---")

        print("\n--- 验证集最终分类报告 ---")
        print(final_val_report_str)

        print("\n--- 测试集最终分类报告 (与最佳验证周期对应) ---")
        print(final_test_report_str)

    else:
        print(f"警告: 未找到最佳模型文件 '{best_model_path}'。无法进行最终评估。")