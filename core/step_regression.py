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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
import json
import pickle
import random
from sklearn.model_selection import StratifiedShuffleSplit
import RegressionModel as model

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
task = 'regression'
image_network_name = 'ResNet'
DATA_ROOT_PATH = "./datasets_corp/datasets"
CSV_ENCODER_FILE_PATH = "./datasets_corp/临床信息_final.csv"
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
P_VALUE_THRESHOLD = 0.05
MAX_SPLIT_ATTEMPTS = 50
FORCE_SPLIT_ATTEMPTS = None

TARGET_COLUMNS = ['FVC', 'FEV1']
NUM_TARGETS = len(TARGET_COLUMNS)


# --- 数据加载与准备  ---
def load_and_prepare_data_multimodal(data_root_path, csv_file_path_encoder, train_val_split=0.8, val_test_split=0.5):
    print(f"当前工作目录: {os.getcwd()}")
    print(f"尝试搜索NIfTI文件的根目录: {os.path.abspath(data_root_path)}")

    try:
        current_df = pd.read_csv(csv_file_path_encoder)

        if '序号' not in current_df.columns:
            raise ValueError(f"CSV 文件 '{csv_file_path_encoder}' 中缺少 '序号' 列。")
        current_df['序号'] = pd.to_numeric(current_df['序号'], errors='coerce')
        current_df.dropna(subset=['序号'], inplace=True)
        current_df['序号'] = current_df['序号'].astype(pd.Int64Dtype())
        current_df.set_index('序号', inplace=True)

        all_defined_features = ONE_HOT_INPUT_FEATURES + BINNER_INPUT_FEATURES + NORMAL_CSV_INPUT_FEATURES + TARGET_COLUMNS
        missing_initial_cols = [col for col in all_defined_features if col not in current_df.columns]
        if missing_initial_cols:
            raise ValueError(f"CSV 中缺少定义的特征列: {missing_initial_cols}")

    except FileNotFoundError:
        print(f"错误: 未找到 CSV 文件 '{csv_file_path_encoder}'。请检查路径。")
        return [], [], [], None, None, None, None, None, None
    except ValueError as e:
        print(f"CSV 数据错误 ({csv_file_path_encoder}): {e}")
        return [], [], [], None, None, None, None, None, None
    except Exception as e:
        print(f"加载或处理 CSV '{csv_file_path_encoder}' 时发生意外错误: {e}")
        return [], [], [], None, None, None, None, None, None

    # --- 1. 匹配影像文件和CSV数据，并初步清理 ---
    print("\n--- 匹配影像文件和CSV数据，并初步清理 ---")
    data_list = []
    search_pattern = os.path.join(data_root_path, "*.nii.gz")
    all_nii_files = glob.glob(search_pattern)
    image_files = {int(os.path.basename(f).split('.')[0]): f for f in all_nii_files if '_mask' not in f}
    mask_files = {int(os.path.basename(f).split('_')[0]): f for f in all_nii_files if '_mask' in f}

    for numeric_id in sorted(current_df.index.tolist()):
        image_path = image_files.get(numeric_id)
        mask_path = mask_files.get(numeric_id)
        if image_path and mask_path:
            # 确保关键特征没有缺失值
            if not current_df.loc[numeric_id, BINNER_INPUT_FEATURES + ONE_HOT_INPUT_FEATURES + TARGET_COLUMNS].isnull().any():
                data_list.append({
                    "id": str(numeric_id),
                    "image": image_path,
                    "mask": mask_path,
                    "csv_data": current_df.loc[numeric_id]
                })

    if not data_list:
        print("错误: 找不到任何匹配的影像、掩码和CSV数据。")
        return [], [], [], None, None, None, None, None, None

    raw_data_df = pd.DataFrame([d['csv_data'] for d in data_list], index=[d['id'] for d in data_list])

    # 检查分层抽样所需的特征是否有足够的样本
    stratify_features = ONE_HOT_INPUT_FEATURES
    temp_df = raw_data_df[stratify_features].fillna("NaN_Value").astype(str).apply(lambda row: '_'.join(row), axis=1)

    value_counts = temp_df.value_counts()
    valid_categories = value_counts[value_counts > 1].index.tolist()
    valid_indices = temp_df[temp_df.isin(valid_categories)].index

    if len(valid_indices) < len(raw_data_df):
        print(f"警告：已从分层抽样中排除 {len(raw_data_df) - len(valid_indices)} 个样本，因为其性别/吸烟史组合类别样本数过少。")
        raw_data_df = raw_data_df.loc[valid_indices]
        data_list = [d for d in data_list if d['id'] in valid_indices]
        temp_df = temp_df.loc[valid_indices]

    if raw_data_df.empty:
        print("错误: 经过分层抽样预处理后，没有剩余有效数据。")
        return [], [], [], None, None, None, None, None, None

    # --- 2. 迭代进行分层抽样和统计学差异检查 (7:1.5:1.5) ---
    print(f"\n--- 尝试 7:1.5:1.5 分层抽样，并检查数值特征的统计学差异 (p-value < {P_VALUE_THRESHOLD}) ---")

    is_split_valid = False
    best_p_values = {}
    best_split = None

    num_attempts = MAX_SPLIT_ATTEMPTS if FORCE_SPLIT_ATTEMPTS is None else FORCE_SPLIT_ATTEMPTS

    features_for_p_check = NORMAL_CSV_INPUT_FEATURES + TARGET_COLUMNS

    for attempt in range(num_attempts):
        # 第一次分割: 总数据 -> 训练集(70%) 和 临时_测试_验证集(30%)
        splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random.randint(1, 10000))
        train_indices, temp_test_val_indices = next(splitter1.split(raw_data_df, temp_df))

        train_raw_df = raw_data_df.iloc[train_indices]
        temp_test_val_df = raw_data_df.iloc[temp_test_val_indices]
        temp_test_val_temp_df = temp_df.iloc[temp_test_val_indices]

        # 第二次分割: 临时_测试_验证集(30%) -> 验证集(15%) 和 测试集(15%)
        # test_size=0.5 表示从30%中取50%，即总数据的15%
        splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random.randint(1, 10000))
        val_indices, test_indices = next(splitter2.split(temp_test_val_df, temp_test_val_temp_df))

        val_raw_df = temp_test_val_df.iloc[val_indices]
        test_raw_df = temp_test_val_df.iloc[test_indices]

        p_values = {}
        all_p_values_ok = True

        # 检查训练集 vs 验证集
        for feature in features_for_p_check:
            stat, p_val = mannwhitneyu(train_raw_df[feature], val_raw_df[feature], alternative='two-sided')
            p_values[f'train_vs_val_{feature}'] = p_val
            if p_val < P_VALUE_THRESHOLD:
                all_p_values_ok = False

        # 检查训练集 vs 测试集
        for feature in features_for_p_check:
            stat, p_val = mannwhitneyu(train_raw_df[feature], test_raw_df[feature], alternative='two-sided')
            p_values[f'train_vs_test_{feature}'] = p_val
            if p_val < P_VALUE_THRESHOLD:
                all_p_values_ok = False

        if best_split is None or np.mean(list(p_values.values())) > np.mean(list(best_p_values.values())):
            best_split = (train_raw_df.index.tolist(), val_raw_df.index.tolist(), test_raw_df.index.tolist())
            best_p_values = p_values

        if FORCE_SPLIT_ATTEMPTS is None and all_p_values_ok:
            is_split_valid = True
            print(f"成功在第 {attempt + 1} 次尝试后找到符合统计学标准的划分！")
            break

    if FORCE_SPLIT_ATTEMPTS is None:
        if not is_split_valid:
            print(f"警告: 在 {MAX_SPLIT_ATTEMPTS} 次尝试后未能找到所有特征p > {P_VALUE_THRESHOLD} 的划分。将使用 p 值最高的划分。")
    else:
        print(f"已完成 {FORCE_SPLIT_ATTEMPTS} 次强制迭代，将使用 p 值最高的划分。")

    train_ids, val_ids, test_ids = best_split

    train_raw_df = raw_data_df.loc[train_ids]
    val_raw_df = raw_data_df.loc[val_ids]
    test_raw_df = raw_data_df.loc[test_ids]

    print(f"\n训练集样本数: {len(train_ids)}")
    print(f"验证集样本数: {len(val_ids)}")
    print(f"测试集样本数: {len(test_ids)}")

    output_dir = os.path.dirname(CSV_ENCODER_FILE_PATH)
    train_output_path = os.path.join(output_dir, "训练集.csv")
    val_output_path = os.path.join(output_dir, "验证集.csv")
    test_output_path = os.path.join(output_dir, "测试集.csv")

    train_raw_df.index.name = '序号'
    val_raw_df.index.name = '序号'
    test_raw_df.index.name = '序号'
    train_raw_df.to_csv(train_output_path, encoding='utf-8')
    val_raw_df.to_csv(val_output_path, encoding='utf-8')
    test_raw_df.to_csv(test_output_path, encoding='utf-8')

    print(f"训练集原始数据已保存至: {train_output_path}")
    print(f"验证集原始数据已保存至: {val_output_path}")
    print(f"测试集原始数据已保存至: {test_output_path}")

    print("\n--- 数值特征 p-values ---")
    for feature in features_for_p_check:
        try:
            _, p_val_tv = mannwhitneyu(train_raw_df[feature], val_raw_df[feature], alternative='two-sided')
            _, p_val_tt = mannwhitneyu(train_raw_df[feature], test_raw_df[feature], alternative='two-sided')
        except ValueError:
            _, p_val_tv = ttest_ind(train_raw_df[feature], val_raw_df[feature], equal_var=False)
            _, p_val_tt = ttest_ind(train_raw_df[feature], test_raw_df[feature], equal_var=False)
        print(f"  {feature}: Train vs Val p={p_val_tv:.6f}, Train vs Test p={p_val_tt:.6f}")
    print("-------------------------")

    print("\n--- 类别特征分布检查 ---")
    for feature in stratify_features:
        print(f"特征: {feature}")
        train_dist = train_raw_df[feature].value_counts(normalize=True) * 100
        val_dist = val_raw_df[feature].value_counts(normalize=True) * 100
        test_dist = test_raw_df[feature].value_counts(normalize=True) * 100
        print(f"  训练集 ({len(train_ids)} 样本):")
        print(train_dist.to_string(float_format="%.2f%%"))
        print(f"  验证集 ({len(val_ids)} 样本):")
        print(val_dist.to_string(float_format="%.2f%%"))
        print(f"  测试集 ({len(test_ids)} 样本):")
        print(test_dist.to_string(float_format="%.2f%%"))
        print("-" * 20)

    def preprocess_df(df, is_training_set, csv_input_scaler=None, target_scaler=None, one_hot_cols=None):
        preprocessed_df = df.copy()

        # 对指定类别特征应用独热编码
        if is_training_set:
            preprocessed_df = pd.get_dummies(preprocessed_df, columns=ONE_HOT_INPUT_FEATURES, prefix=ONE_HOT_INPUT_FEATURES, dummy_na=False)
            one_hot_cols = [col for col in preprocessed_df.columns if any(p in col for p in ONE_HOT_INPUT_FEATURES)]
        else:
            # 确保验证集与训练集有相同的独热编码列
            preprocessed_df = pd.get_dummies(preprocessed_df, columns=ONE_HOT_INPUT_FEATURES, prefix=ONE_HOT_INPUT_FEATURES, dummy_na=False)
            missing_cols = set(one_hot_cols) - set(preprocessed_df.columns)
            for c in missing_cols:
                preprocessed_df[c] = 0
            preprocessed_df = preprocessed_df[one_hot_cols + list(preprocessed_df.columns.drop(one_hot_cols))]

        final_csv_input_features = NORMAL_CSV_INPUT_FEATURES + one_hot_cols
        final_csv_input_features = [col for col in final_csv_input_features if col in preprocessed_df.columns]

        if is_training_set:
            csv_input_scaler = StandardScaler()
            preprocessed_df[final_csv_input_features] = csv_input_scaler.fit_transform(preprocessed_df[final_csv_input_features])

            target_scaler = StandardScaler()
            preprocessed_df[TARGET_COLUMNS] = target_scaler.fit_transform(preprocessed_df[TARGET_COLUMNS])
        else:
            preprocessed_df[final_csv_input_features] = csv_input_scaler.transform(preprocessed_df[final_csv_input_features])
            preprocessed_df[TARGET_COLUMNS] = target_scaler.transform(preprocessed_df[TARGET_COLUMNS])

        return preprocessed_df, csv_input_scaler, target_scaler, final_csv_input_features, one_hot_cols

    train_processed_df, csv_input_scaler, target_scaler, final_csv_input_features, one_hot_cols = preprocess_df(train_raw_df, True)
    val_processed_df, _, _, _, _ = preprocess_df(val_raw_df, False, csv_input_scaler, target_scaler, one_hot_cols)
    test_processed_df, _, _, _, _ = preprocess_df(test_raw_df, False, csv_input_scaler, target_scaler, one_hot_cols)

    scaler_output_dir = MODEL_OUTPUT_DIR
    os.makedirs(scaler_output_dir, exist_ok=True)

    csv_input_scaler_path = os.path.join(scaler_output_dir, "input_scaler.pkl")
    with open(csv_input_scaler_path, 'wb') as f:
        pickle.dump(csv_input_scaler, f)
    print(f"CSV 输入特征标准化器已保存到: {csv_input_scaler_path}")

    target_scaler_path = os.path.join(scaler_output_dir, "target_scaler.pkl")
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"目标变量标准化器已保存到: {target_scaler_path}")

    train_files = []
    val_files = []
    test_files = []
    id_to_data = {d['id']: d for d in data_list}

    for idx in train_ids:
        raw_info = id_to_data[idx]
        train_files.append({
            "image": raw_info['image'],
            "mask": raw_info['mask'],
            "csv_input_features": train_processed_df.loc[idx, final_csv_input_features].values.astype(np.float32),
            "targets": train_processed_df.loc[idx, TARGET_COLUMNS].values.astype(np.float32),
            "id": idx
        })

    for idx in val_ids:
        raw_info = id_to_data[idx]
        val_files.append({
            "image": raw_info['image'],
            "mask": raw_info['mask'],
            "csv_input_features": val_processed_df.loc[idx, final_csv_input_features].values.astype(np.float32),
            "targets": val_processed_df.loc[idx, TARGET_COLUMNS].values.astype(np.float32),
            "id": idx
        })

    for idx in test_ids:
        raw_info = id_to_data[idx]
        test_files.append({
            "image": raw_info['image'],
            "mask": raw_info['mask'],
            "csv_input_features": test_processed_df.loc[idx, final_csv_input_features].values.astype(np.float32),
            "targets": test_processed_df.loc[idx, TARGET_COLUMNS].values.astype(np.float32),
            "id": idx
        })

    return train_files, val_files, test_files, csv_input_scaler, target_scaler, final_csv_input_features, train_raw_df, val_raw_df, test_raw_df


# The rest of the script remains the same
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
        ToTensord(keys=["image", "targets", "csv_input_features"]),
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
        ToTensord(keys=["image", "targets", "csv_input_features"]),
    ]
)

# Test transforms are the same as validation transforms
test_transforms = val_transforms


def validate_model(data_loader, model, loss_function, scaler_targets, set_name):
    model.eval()
    val_loss = 0
    step = 0
    val_preds_np = []
    val_targets_np = []
    val_ids = []
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc=f"评估 {set_name} 周期"):
            step += 1
            inputs, targets, csv_input_features, ids = (
                batch_data["image"].to(DEVICE),
                batch_data["targets"].to(DEVICE),
                batch_data["csv_input_features"].to(DEVICE),
                batch_data["id"]
            )
            outputs,image_features = model(inputs, csv_input_features)
            loss = loss_function(outputs, targets)
            val_loss += loss.item()
            val_preds_np.append(outputs.detach().cpu().numpy())
            val_targets_np.append(targets.detach().cpu().numpy())
            val_ids.extend(ids)

    val_loss /= step
    val_preds_np = np.concatenate(val_preds_np, axis=0)
    val_targets_np = np.concatenate(val_targets_np, axis=0)
    val_metrics = calculate_regression_metrics(val_targets_np, val_preds_np, scaler_targets, set_name, val_ids)
    return val_loss, val_metrics

def train_model(train_loader, val_loader, test_loader, model, loss_function, optimizer,
                max_epochs, model_output_dir, scaler_targets):
    best_val_loss = float('inf')
    best_epoch = -1
    best_val_metrics = {}
    best_test_metrics_at_best_val_epoch = {}

    metrics_history_file = os.path.join(model_output_dir, "training_metrics_history.json")
    history_data = {
        'train_loss_history': [],
        'val_loss_history': [],
        'val_metrics_history': [],
        'test_loss_history': [],
        'test_metrics_history': []
    }
    print(f"训练历史文件 '{metrics_history_file}' 将在每次训练时被覆盖。")

    for epoch in range(max_epochs):
        print(f"\n--- 周期 {epoch + 1}/{max_epochs} ---")
        model.train()
        epoch_train_loss = 0
        step = 0

        for batch_data in tqdm(train_loader, desc=f"训练周期 {epoch + 1}"):
            step += 1
            inputs, targets, csv_input_features, _ = (
                batch_data["image"].to(DEVICE),
                batch_data["targets"].to(DEVICE),
                batch_data["csv_input_features"].to(DEVICE),
                batch_data["id"]
            )
            optimizer.zero_grad()
            outputs,image_features = model(inputs, csv_input_features)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= step

        val_loss, val_metrics = validate_model(val_loader, model, loss_function, scaler_targets, "验证集")
        test_loss, test_metrics = validate_model(test_loader, model, loss_function, scaler_targets, "测试集")


        print(f"  train loss: {epoch_train_loss:.6f}, val loss: {val_loss:.6f}, test loss: {test_loss:.6f}")
        for i, col in enumerate(TARGET_COLUMNS):
            val_m = val_metrics[col]
            test_m = test_metrics[col]
            print(f"    {col} - Val: ccc={val_m['ccc']:.4f}, r={val_m['r']:.4f}, r2={val_m['r2']:.4f}, rmse={val_m['rmse']:.6f}, mae={val_m['mae']:.6f}")
            print(f"    {col} - Test: ccc={test_m['ccc']:.4f}, r={test_m['r']:.4f}, r2={test_m['r2']:.4f}, rmse={test_m['rmse']:.6f}, mae={test_m['mae']:.6f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_val_metrics = val_metrics
            best_test_metrics_at_best_val_epoch = test_metrics
            torch.save(model.state_dict(), os.path.join(model_output_dir, "best_model.pth"))
            print("  >>> 新的最佳多模态回归模型已保存！(基于最低验证损失) <<<")

        print(f"  --- 历史最佳验证指标 (Epoch {best_epoch}) ---")
        print(f"  最佳验证损失: {best_val_loss:.6f}")
        for col in TARGET_COLUMNS:
            if col in best_val_metrics:
                metrics = best_val_metrics[col]
                print(f"    {col}: ccc={metrics['ccc']:.4f}, r={metrics['r']:.4f}, r2={metrics['r2']:.4f}, rmse={metrics['rmse']:.6f}, mae={metrics['mae']:.6f}")

        print(f"  --- 对应最佳验证指标时的测试集指标 ---")
        for col in TARGET_COLUMNS:
            if col in best_test_metrics_at_best_val_epoch:
                metrics = best_test_metrics_at_best_val_epoch[col]
                print(f"    {col}: ccc={metrics['ccc']:.4f}, r={metrics['r']:.4f}, r2={metrics['r2']:.4f}, rmse={metrics['rmse']:.6f}, mae={metrics['mae']:.6f}")

        print("-" * 30)

        history_data['train_loss_history'].append(epoch_train_loss)
        history_data['val_loss_history'].append(val_loss)
        history_data['val_metrics_history'].append(val_metrics)
        history_data['test_loss_history'].append(test_loss)
        history_data['test_metrics_history'].append(test_metrics)

        with open(metrics_history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=4, ensure_ascii=False)
        print(f"  训练指标已保存到 {metrics_history_file}")
        print("-" * 30)

    print(f"\n训练完成！最低验证损失: {best_val_loss:.6f} (来自周期 {best_epoch})")

    return history_data


def calculate_regression_metrics(y_true_scaled, y_pred_scaled, scaler_targets, set_name, sample_ids):
    y_true_unscaled = scaler_targets.inverse_transform(y_true_scaled)
    y_pred_unscaled = scaler_targets.inverse_transform(y_pred_scaled)

    nan_in_true = np.isnan(y_true_unscaled).any()
    nan_in_pred = np.isnan(y_pred_unscaled).any()

    if nan_in_true or nan_in_pred:
        print(f"\n警告: 在 '{set_name}' 数据集中，反标准化后的数据中检测到 NaN 值！")
        problem_rows_mask = np.isnan(y_true_unscaled).any(axis=1) | np.isnan(y_pred_unscaled).any(axis=1)
        problem_ids = [sample_ids[i] for i in np.where(problem_rows_mask)[0]]
        print(f"  受影响的样本 ID ({set_name}): {problem_ids}")

        valid_indices = ~problem_rows_mask
        if np.sum(valid_indices) == 0:
            print(f"错误: '{set_name}' 数据集中所有样本在反标准化后都包含 NaN。无法计算指标。")
            return {col: {'r2': np.nan, 'mse': np.nan, 'mae': np.nan, 'ccc': np.nan, 'r': np.nan, 'rmse': np.nan} for col in TARGET_COLUMNS}

        y_true_unscaled = y_true_unscaled[valid_indices]
        y_pred_unscaled = y_pred_unscaled[valid_indices]
        print(f"  已从 '{set_name}' 指标计算中排除 {np.sum(~valid_indices)} 个包含 NaN 的样本。")

    metrics = {}
    for i, col_name in enumerate(TARGET_COLUMNS):
        true_col = y_true_unscaled[:, i]
        pred_col = y_pred_unscaled[:, i]

        r2 = r2_score(true_col, pred_col)
        mse = mean_squared_error(true_col, pred_col)
        mae = mean_absolute_error(true_col, pred_col)

        mean_true, mean_pred = np.mean(true_col), np.mean(pred_col)
        var_true, var_pred = np.var(true_col), np.var(pred_col)
        cov_true_pred = np.mean((true_col - mean_true) * (pred_col - mean_pred))
        denominator_ccc = (var_true + var_pred + (mean_true - mean_pred)**2)
        ccc = (2 * cov_true_pred) / denominator_ccc if denominator_ccc != 0 else np.nan

        r, _ = pearsonr(true_col, pred_col) if np.std(true_col) != 0 and np.std(pred_col) != 0 else (np.nan, np.nan)

        metrics[col_name] = {
            'r2': float(r2),
            'mse': float(mse),
            'mae': float(mae),
            'ccc': float(ccc),
            'r': float(r),
            'rmse': float(np.sqrt(mse))
        }
    return metrics


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

    train_files, val_files, test_files, csv_input_scaler, target_scaler, all_csv_features_list, train_raw_df, val_raw_df, test_raw_df = \
        load_and_prepare_data_multimodal(DATA_ROOT_PATH, CSV_ENCODER_FILE_PATH, train_val_split=0.7, val_test_split=0.5)

    if not train_files or not val_files or not test_files:
        print("无法进行训练、验证或测试，请检查数据加载错误信息。")
        exit()

    print("\n--- 最终送入模型的 CSV 特征列表 ---")
    print(all_csv_features_list)
    print("---------------------------------")


    model = model.CrossModalNet(
        image_network_name= image_network_name,
        num_image_channels=1,
        num_csv_input_features=len(all_csv_features_list),
        num_targets=NUM_TARGETS
    ).to(DEVICE)
    loss_function = torch.nn.L1Loss()
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

    print("\n--- 开始多模态模型训练 ---")

    history_data = train_model(
        train_loader, val_loader, test_loader, model, loss_function, optimizer,
        MAX_EPOCHS, MODEL_OUTPUT_DIR, target_scaler
    )

    print("\n--- 训练过程结束 ---")

    print("\n--- 最终指标摘要 ---")
    if history_data:
        # 找到验证损失最低的周期索引
        val_loss_history = history_data['val_loss_history']
        best_epoch_index = np.argmin(val_loss_history)

        # 获取该周期对应的验证集和测试集指标
        final_val_metrics = history_data['val_metrics_history'][best_epoch_index]
        final_test_metrics = history_data['test_metrics_history'][best_epoch_index]

        print(f"\n--- 基于最低验证损失的最佳周期 (Epoch {best_epoch_index + 1}) ---")

        print("\n--- 验证集最终指标 ---")
        for col in TARGET_COLUMNS:
            metrics = final_val_metrics[col]
            print(f"  {col}: ccc={metrics['ccc']:.4f}, r={metrics['r']:.4f}, r2={metrics['r2']:.4f}, rmse={metrics['rmse']:.6f}, mae={metrics['mae']:.6f}")

        print("\n--- 测试集最终指标 (与最佳验证周期对应) ---")
        for col in TARGET_COLUMNS:
            metrics = final_test_metrics[col]
            print(f"  {col}: ccc={metrics['ccc']:.4f}, r={metrics['r']:.4f}, r2={metrics['r2']:.4f}, rmse={metrics['rmse']:.6f}, mae={metrics['mae']:.6f}")