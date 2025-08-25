import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
)
from scipy.stats import pearsonr
from monai.data import DataLoader, PersistentDataset
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
    Spacingd,
)
import json # 新增导入

# --- 导入您自定义的模型类 ---
import RegressionModel as model

# --- 补丁代码 (保持不变) ---
orig_torch_load = torch.load
def torch_wrapper(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_torch_load(*args, **kwargs)
torch.load = torch_wrapper

# --- 配置 (与训练脚本保持一致) ---
task = 'regression'
image_network_name = 'ResNet'
DATA_ROOT_PATH = "./datasets_corp/datasets"
TEST_CSV_FILE_PATH = "./datasets_corp/测试集.csv"
MODEL_OUTPUT_DIR = f"./model/{task}/{image_network_name}"
REPORT_OUTPUT_DIR = f"./report/{task}/{image_network_name}/plot"
root_dir = "./cache"
persistent_cache = root_dir

BATCH_SIZE = 8
NUM_WORKERS = 4

HU_LOWER_BOUND = -950
HU_UPPER_BOUND = -750
TARGET_SIZE = (128, 128, 128)
TARGET_SPACING = (1.0, 1.0, 1.0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# --- 定义特征和目标 (与训练脚本保持一致) ---
ONE_HOT_INPUT_FEATURES = ['性别', '吸烟史']
NORMAL_CSV_INPUT_FEATURES = ['年龄', '身高', '体重', 'lung_volume_liters', 'LAA910', 'LAA950',
                             'LAA980', 'Mean', 'Median', 'Skewness', 'Kurtosis', 'Variance',
                             'MeanAbsoluteDeviation', 'InterquartileRange', '10Percentile',
                             '90Percentile']
TARGET_COLUMNS = ['FVC', 'FEV1']


# --- 肺功能计算公式 ---
def calculate_fev1_pred_male(age, height):
    return -3.088 + 0.0487 * height - 0.0211 * age

def calculate_fvc_pred_male(age, height):
    return -4.634 + 0.0601 * height - 0.0216 * age

def calculate_fev1_pred_female(age, height):
    return -2.981 + 0.0384 * height - 0.0191 * age

def calculate_fvc_pred_female(age, height):
    return -3.090 + 0.0435 * height - 0.0189 * age

def predict_three_classes(predicted_fev1, predicted_fvc, age, height, sex):
    fev1_fvc_ratio = (predicted_fev1 / predicted_fvc) * 100

    if sex == '男':
        fev1_pred_calc = calculate_fev1_pred_male(age, height)
        fvc_pred_calc = calculate_fvc_pred_male(age, height)
    else: # sex == '女'
        fev1_pred_calc = calculate_fev1_pred_female(age, height)
        fvc_pred_calc = calculate_fvc_pred_female(age, height)

    fev1_percent = (predicted_fev1 / fev1_pred_calc) * 100
    fvc_percent = (predicted_fvc / fvc_pred_calc) * 100

    if fev1_fvc_ratio < 70:
        return 'COPD'
    elif fev1_fvc_ratio >= 70 and fev1_percent >= 80 and fvc_percent >= 80:
        return 'NORMAL'
    elif fev1_fvc_ratio >= 70 and (fev1_percent < 80 or fvc_percent < 80):
        return 'PRISm'
    else:
        return 'UNKNOWN' # 理论上不应该出现

# --- 指标计算与绘图函数 ---
def calculate_regression_metrics(y_true, y_pred, target_columns):
    """计算并返回多个回归指标。"""
    metrics = {}
    for i, col_name in enumerate(target_columns):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]

        nan_or_inf_mask = np.isnan(true_col) | np.isinf(true_col) | np.isnan(pred_col) | np.isinf(pred_col)
        valid_indices = ~nan_or_inf_mask

        if not np.any(valid_indices):
            metrics[col_name] = {'r': np.nan, 'r2': np.nan, 'ccc': np.nan, 'rmse': np.nan, 'mse': np.nan, 'mae': np.nan}
            continue

        true_col_valid = true_col[valid_indices]
        pred_col_valid = pred_col[valid_indices]

        try:
            r, _ = pearsonr(true_col_valid, pred_col_valid)
        except ValueError:
            r = np.nan
        r2 = r2_score(true_col_valid, pred_col_valid)

        mean_true = np.mean(true_col_valid)
        mean_pred = np.mean(pred_col_valid)
        var_true = np.var(true_col_valid)
        var_pred = np.var(pred_col_valid)
        cov_true_pred = np.mean((true_col_valid - mean_true) * (pred_col_valid - mean_pred))
        denominator_ccc = var_true + var_pred + (mean_true - mean_pred)**2
        ccc = (2 * cov_true_pred) / denominator_ccc if denominator_ccc != 0 else np.nan

        mse = mean_squared_error(true_col_valid, pred_col_valid)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_col_valid, pred_col_valid)

        metrics[col_name] = {
            'r': float(r), 'r2': float(r2), 'ccc': float(ccc),
            'rmse': float(rmse), 'mse': float(mse), 'mae': float(mae)
        }
    return metrics


def plot_roc_curve(y_true, scores, filename):
    """绘制带95%置信区间的ROC曲线并保存。"""
    plt.figure(figsize=(8, 8))
    tprs, aurocs = [], []
    base_fpr = np.linspace(0, 1, 101)

    # 使用bootstrap计算置信区间
    n_bootstrap = 2000
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        # 确保抽样数据包含至少两个类别，否则跳过
        if len(np.unique(y_true.astype(int)[indices])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true.astype(int)[indices], scores.astype(float)[indices])
        tpr_interpolated = np.interp(base_fpr, fpr, tpr)
        tprs.append(tpr_interpolated)
        aurocs.append(auc(fpr, tpr))

    # 计算均值和置信区间
    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
    mean_auroc = np.mean(aurocs)

    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Chance')
    plt.plot(base_fpr, mean_tpr, 'b', label=f'Mean ROC (AUC = {mean_auroc:.2f})')
    auroc_lower = np.percentile(aurocs, 2.5)
    auroc_upper = np.percentile(aurocs, 97.5)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='lightblue', alpha=0.5, label=f'95% CI ({auroc_lower:.2f}-{auroc_upper:.2f})')

    plt.title('ROC Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_roc_curve_multiclass(y_true, y_scores, n_classes, filename):
    """绘制带95%置信区间的多分类ROC曲线并保存。"""
    plt.figure(figsize=(10, 8))
    colors = ['b', 'r', 'g']
    class_names = ['COPD', 'NORMAL', 'PRISm']

    for i in range(n_classes):
        tprs, aurocs = [], []
        base_fpr = np.linspace(0, 1, 101)

        y_true_binary = (y_true == i).astype(int)

        # 使用bootstrap计算置信区间
        n_bootstrap = 2000
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true_binary), len(y_true_binary), replace=True)
            if len(np.unique(y_true_binary[indices])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true_binary[indices], y_scores[indices, i])
            tpr_interpolated = np.interp(base_fpr, fpr, tpr)
            tprs.append(tpr_interpolated)
            aurocs.append(auc(fpr, tpr))

        # 计算均值和置信区间
        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
        mean_auroc = np.mean(aurocs)

        plt.plot(base_fpr, mean_tpr, color=colors[i], label=f'{class_names[i]} ROC (AUC = {mean_auroc:.2f})')
        # 计算95% CI
        auroc_lower = np.percentile(aurocs, 2.5)
        auroc_upper = np.percentile(aurocs, 97.5)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.15, label=f'95% CI ({auroc_lower:.2f}-{auroc_upper:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Chance')
    plt.title('Multi-Class ROC Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_bland_altman(true_vals, pred_vals, title, filename):
    """绘制Bland-Altman图并保存。"""
    plt.figure(figsize=(10, 8))

    diff = pred_vals - true_vals
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    mean_vals = (true_vals + pred_vals) / 2

    plt.scatter(mean_vals, diff, alpha=0.5, s=20)
    plt.axhline(mean_diff, color='gray', linestyle='--', label=f'Mean Difference: {mean_diff:.2f}')
    plt.axhline(loa_upper, color='red', linestyle='--', label=f'95% LoA: {loa_upper:.2f}')
    plt.axhline(loa_lower, color='red', linestyle='--')

    plt.title(title, fontsize=16)
    plt.xlabel('Mean of Measurements', fontsize=12)
    plt.ylabel('Difference (Prediction - True)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# --- 数据加载与准备 ---
def load_and_prepare_test_data(data_root_path, csv_file_path, csv_input_scaler, one_hot_cols):
    """加载测试集的影像和CSV数据。"""
    try:
        current_df = pd.read_csv(csv_file_path, index_col='序号')
    except FileNotFoundError:
        print(f"错误: 未找到测试集 CSV 文件 '{csv_file_path}'。请检查路径。")
        return [], None, None

    data_list = []
    search_pattern = os.path.join(data_root_path, "*.nii.gz")
    all_nii_files = glob.glob(search_pattern)
    image_files = {int(os.path.basename(f).split('.')[0]): f for f in all_nii_files if '_mask' not in f}
    mask_files = {int(os.path.basename(f).split('_')[0]): f for f in all_nii_files if '_mask' in f}

    for numeric_id in sorted(current_df.index.tolist()):
        image_path = image_files.get(numeric_id)
        mask_path = mask_files.get(numeric_id)
        if image_path and mask_path:
            if not current_df.loc[numeric_id, TARGET_COLUMNS].isnull().any():
                data_list.append({
                    "id": str(numeric_id),
                    "image": image_path,
                    "mask": mask_path,
                    "csv_data": current_df.loc[numeric_id]
                })

    if not data_list:
        print("错误: 找不到任何匹配的影像、掩码和CSV数据。")
        return [], None, None

    raw_test_df = pd.DataFrame([d['csv_data'] for d in data_list], index=[d['id'] for d in data_list])
    preprocessed_df = pd.get_dummies(raw_test_df, columns=ONE_HOT_INPUT_FEATURES, prefix=ONE_HOT_INPUT_FEATURES, dummy_na=False)
    missing_cols = set(one_hot_cols) - set(preprocessed_df.columns)
    for c in missing_cols:
        preprocessed_df[c] = 0
    preprocessed_df = preprocessed_df[one_hot_cols + list(preprocessed_df.columns.drop(one_hot_cols))]
    final_csv_input_features = NORMAL_CSV_INPUT_FEATURES + one_hot_cols
    final_csv_input_features = [col for col in final_csv_input_features if col in preprocessed_df.columns]
    preprocessed_df[final_csv_input_features] = csv_input_scaler.transform(preprocessed_df[final_csv_input_features])

    test_files = []
    id_to_data = {d['id']: d for d in data_list}
    for idx in raw_test_df.index:
        raw_info = id_to_data[str(idx)]
        test_files.append({
            "image": raw_info['image'],
            "mask": raw_info['mask'],
            "csv_input_features": preprocessed_df.loc[str(idx), final_csv_input_features].values.astype(np.float32),
            "targets": raw_test_df.loc[idx, TARGET_COLUMNS].values.astype(np.float32),
            "original_type": raw_test_df.loc[idx, '类型'],
            "sex": raw_test_df.loc[idx, '性别'],
            "age": raw_test_df.loc[idx, '年龄'],
            "height": raw_test_df.loc[idx, '身高'],
            "id": str(idx)
        })

    return test_files, final_csv_input_features, raw_test_df


# --- 主执行函数 ---
def predict_and_evaluate():
    if not os.path.exists(MODEL_OUTPUT_DIR):
        print(f"错误: 找不到模型目录 '{MODEL_OUTPUT_DIR}'。")
        return

    if not os.path.exists(REPORT_OUTPUT_DIR):
        os.makedirs(REPORT_OUTPUT_DIR)
        print(f"已创建报告目录: {REPORT_OUTPUT_DIR}")

    # --- 1. 加载标准化器 ---
    try:
        with open(os.path.join(MODEL_OUTPUT_DIR, "input_scaler.pkl"), 'rb') as f:
            csv_input_scaler = pickle.load(f)
        with open(os.path.join(MODEL_OUTPUT_DIR, "target_scaler.pkl"), 'rb') as f:
            target_scaler = pickle.load(f)
        print("标准化器加载成功。")
        one_hot_cols = [col for col in csv_input_scaler.feature_names_in_ if any(f in col for f in ONE_HOT_INPUT_FEATURES)]
    except FileNotFoundError:
        print("错误: 找不到必要的标准化器文件。")
        return

    # --- 2. 加载和准备测试数据 ---
    test_files, all_csv_features_list, raw_test_df = load_and_prepare_test_data(DATA_ROOT_PATH, TEST_CSV_FILE_PATH, csv_input_scaler, one_hot_cols)
    if not test_files:
        print("无法进行预测，请检查数据加载错误信息。")
        return

    num_csv_features = len(all_csv_features_list)
    model_to_load = model.CrossModalNet(
        image_network_name=image_network_name,
        num_image_channels=1,
        num_csv_input_features=num_csv_features,
        num_targets=len(TARGET_COLUMNS)
    ).to(DEVICE)

    model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型权重文件 '{model_path}'。")
        return

    model_to_load.load_state_dict(torch.load(model_path))
    model_to_load.eval()
    print("模型加载成功。")

    transforms = Compose(
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

    # --- 3. 创建 DataLoader ---
    test_ds = PersistentDataset(data=test_files, transform=transforms, cache_dir=os.path.join(persistent_cache, "test_predict"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    # --- 4. 进行预测 ---
    print("\n--- 开始在测试集上进行预测 ---")
    all_preds_unscaled = []
    all_targets_unscaled = []
    all_fev1_fvc_ratios = []
    # 确保使用原始的raw_test_df来获取非处理过的分类信息
    all_raw_data_df = raw_test_df.loc[raw_test_df.index.astype(str).isin([d['id'] for d in test_files])].reset_index(drop=True)

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="处理测试集"):
            inputs, targets_unscaled_batch, csv_input_features = (
                batch_data["image"].to(DEVICE),
                batch_data["targets"].to(DEVICE),
                batch_data["csv_input_features"].to(DEVICE)
            )
            outputs_scaled,image_features = model_to_load(inputs, csv_input_features)

            y_pred_unscaled = target_scaler.inverse_transform(outputs_scaled.detach().cpu().numpy())
            y_true_unscaled = targets_unscaled_batch.detach().cpu().numpy()

            all_preds_unscaled.append(y_pred_unscaled)
            all_targets_unscaled.append(y_true_unscaled)

            # 计算模型预测的 FEV1/FVC 比值
            fev1_pred = y_pred_unscaled[:, 1]
            fvc_pred = y_pred_unscaled[:, 0]
            fev1_fvc_ratio = (fev1_pred / fvc_pred) * 100
            all_fev1_fvc_ratios.extend(fev1_fvc_ratio)

    all_preds_unscaled = np.concatenate(all_preds_unscaled, axis=0)
    all_targets_unscaled = np.concatenate(all_targets_unscaled, axis=0)
    all_fev1_fvc_ratios = np.array(all_fev1_fvc_ratios)

    # --- 5. 保存回归指标到txt ---
    print("\n--- 保存回归指标到文件 ---")
    metrics = calculate_regression_metrics(all_targets_unscaled, all_preds_unscaled, TARGET_COLUMNS)
    report_path = os.path.join(REPORT_OUTPUT_DIR, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Regression Metrics on Test Set ---\n\n")
        for col in TARGET_COLUMNS:
            if col in metrics:
                m = metrics[col]
                f.write(f"  {col}:\n")
                f.write(f"    r (Pearson Correlation): {m['r']:.4f}\n")
                f.write(f"    r2 (Coefficient of Determination): {m['r2']:.4f}\n")
                f.write(f"    ccc (Concordance Correlation Coefficient): {m['ccc']:.4f}\n")
                f.write(f"    rmse (Root Mean Squared Error): {m['rmse']:.4f}\n")
                f.write(f"    mse (Mean Squared Error): {m['mse']:.4f}\n")
                f.write(f"    mae (Mean Absolute Error): {m['mae']:.4f}\n")
                f.write("\n")
    print(f"回归指标已保存至: {report_path}")

    # --- 6. COPD二分类和报告 ---
    print("\n--- 执行COPD二分类和报告 ---")
    y_true_types_2class = all_raw_data_df['类型'].fillna('UNKNOWN').values
    y_true_binary = np.where(y_true_types_2class == 'COPD', 1, 0)
    y_pred_binary = np.where(all_fev1_fvc_ratios < 70, 1, 0)

    # 分类报告
    class_report_2class = classification_report(y_true_binary, y_pred_binary, target_names=['Non-COPD', 'COPD'], zero_division=0)
    report_path_2class = os.path.join(REPORT_OUTPUT_DIR, "2classify_classification_report.txt")
    with open(report_path_2class, "w") as f:
        f.write("--- 2-Class Classification Report ---\n\n")
        f.write(class_report_2class)
    print(f"二分类报告已保存至: {report_path_2class}")

    # 绘制二分类混淆矩阵
    unique_numerical_labels_2class = np.unique(y_true_binary)
    label_names_map_2class = {0: 'Non-COPD', 1: 'COPD'}
    existing_labels_2class = [label_names_map_2class[i] for i in unique_numerical_labels_2class]

    cm_2class = confusion_matrix(y_true_binary, y_pred_binary, labels=unique_numerical_labels_2class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_2class, annot=True, fmt='d', cmap='Blues', xticklabels=existing_labels_2class, yticklabels=existing_labels_2class)
    plt.title('2-Class Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "2classify_confusion_matrix.png"))
    plt.close()

    # 绘制二分类ROC曲线
    reversed_scores = 100 - all_fev1_fvc_ratios
    plot_roc_curve(y_true_binary, reversed_scores, os.path.join(REPORT_OUTPUT_DIR, "2classify_roc_curve.png"))

    print("二分类图表已保存。")


    # --- 7. 三分类分析和报告 ---
    print("\n--- 执行三分类分析和报告 ---")
    # 🚨 修改点: 确保使用 '分类' 列作为真实标签 🚨
    y_true_3class = all_raw_data_df['分类'].fillna('UNKNOWN').values
    y_pred_3class = np.array([
        predict_three_classes(
            all_preds_unscaled[i, 1],
            all_preds_unscaled[i, 0],
            all_raw_data_df.loc[i, '年龄'],
            all_raw_data_df.loc[i, '身高'],
            all_raw_data_df.loc[i, '性别']
        )
        for i in range(len(all_preds_unscaled))
    ])

    # 剔除无法分类的样本
    valid_indices = (y_true_3class != 'UNKNOWN') & (y_pred_3class != 'UNKNOWN')
    y_true_3class_filtered = y_true_3class[valid_indices]
    y_pred_3class_filtered = y_pred_3class[valid_indices]

    # 确保在计算前，有效样本不为空
    if len(y_true_3class_filtered) == 0:
        print("警告: 三分类分析的有效样本数量为0，无法生成报告和图表。")
        return

    # 分类报告
    class_names_3class = ['COPD', 'NORMAL', 'PRISm']
    class_report_3class = classification_report(
        y_true_3class_filtered,
        y_pred_3class_filtered,
        labels=class_names_3class,
        zero_division=0
    )
    report_path_3class = os.path.join(REPORT_OUTPUT_DIR, "3classify_classification_report.txt")
    with open(report_path_3class, "w") as f:
        f.write("--- 3-Class Classification Report ---\n\n")
        f.write(class_report_3class)
    print(f"三分类报告已保存至: {report_path_3class}")

    # 绘制三分类混淆矩阵
    cm_3class = confusion_matrix(y_true_3class_filtered, y_pred_3class_filtered, labels=class_names_3class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_3class, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_3class, yticklabels=class_names_3class)
    plt.title('3-Class Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "3classify_confusion_matrix.png"))
    plt.close()

    # 绘制三分类ROC曲线 (需要概率预测，这里用预测比值作为近似)
    # 制作三分类的数值标签
    y_true_3class_numerical = np.array([class_names_3class.index(c) for c in y_true_3class_filtered])

    # 构建伪概率矩阵，用于ROC曲线绘制
    filtered_fev1_fvc_ratios = all_fev1_fvc_ratios[valid_indices]

    y_scores_3class = np.zeros((len(filtered_fev1_fvc_ratios), 3))

    # COPD: FEV1/FVC < 70, score is inverse of ratio
    y_scores_3class[:, 0] = 100 - filtered_fev1_fvc_ratios

    # NORMAL & PRISm: 需要重新计算，因为它们的指标依赖于百分比
    fev1_percents_filtered = np.zeros_like(filtered_fev1_fvc_ratios)
    fvc_percents_filtered = np.zeros_like(filtered_fev1_fvc_ratios)

    raw_data_filtered = all_raw_data_df.loc[valid_indices].reset_index(drop=True)
    preds_unscaled_filtered = all_preds_unscaled[valid_indices]

    for i in range(len(filtered_fev1_fvc_ratios)):
        raw_data = raw_data_filtered.loc[i]
        predicted_fev1 = preds_unscaled_filtered[i, 1]
        predicted_fvc = preds_unscaled_filtered[i, 0]

        if raw_data['性别'] == '男':
            fev1_pred_calc = calculate_fev1_pred_male(raw_data['年龄'], raw_data['身高'])
            fvc_pred_calc = calculate_fvc_pred_male(raw_data['年龄'], raw_data['身高'])
        else:
            fev1_pred_calc = calculate_fev1_pred_female(raw_data['年龄'], raw_data['身高'])
            fvc_pred_calc = calculate_fvc_pred_female(raw_data['年龄'], raw_data['身高'])

        fev1_percents_filtered[i] = (predicted_fev1 / fev1_pred_calc) * 100
        fvc_percents_filtered[i] = (predicted_fvc / fvc_pred_calc) * 100

    y_scores_3class[:, 1] = fev1_percents_filtered + fvc_percents_filtered
    y_scores_3class[:, 2] = 200 - (fev1_percents_filtered + fvc_percents_filtered)

    plot_roc_curve_multiclass(y_true_3class_numerical, y_scores_3class, 3, os.path.join(REPORT_OUTPUT_DIR, "3classify_roc_curve.png"))

    print("三分类图表已保存。")

    # Bland-Altman图
    print("\n--- 绘制并保存回归图表 ---")
    # FVC
    plot_bland_altman(all_targets_unscaled[:, 0], all_preds_unscaled[:, 0],
                      'Bland-Altman Plot for FVC', os.path.join(REPORT_OUTPUT_DIR, "bland_altman_fvc.png"))
    # FEV1
    plot_bland_altman(all_targets_unscaled[:, 1], all_preds_unscaled[:, 1],
                      'Bland-Altman Plot for FEV1', os.path.join(REPORT_OUTPUT_DIR, "bland_altman_fev1.png"))

    print("所有图表已保存。")

# --- 修正后的绘图函数 ---
def plot_training_metrics():
    """根据 training_metrics_history.json 绘制训练和验证指标曲线图。"""
    # 修正点：直接在模型输出目录寻找JSON文件
    metrics_file_path = os.path.join(MODEL_OUTPUT_DIR, 'training_metrics_history.json')

    if not os.path.exists(metrics_file_path):
        print(f"\n警告: 找不到训练指标文件 '{metrics_file_path}'，跳过绘制训练曲线。")
        return

    try:
        with open(metrics_file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"\n错误: 文件 '{metrics_file_path}' 格式不正确，无法解析。")
        return

    train_loss_history = data.get('train_loss_history', [])
    val_metrics_history = data.get('val_metrics_history', [])

    if not train_loss_history and not val_metrics_history:
        print("\n警告: JSON 文件中没有训练或验证数据。")
        return

    # 修正点：从 FVC 和 FEV1 的 mse 中计算平均验证损失
    val_loss_history = [
        (d['FVC']['mse'] + d['FEV1']['mse']) / 2
        for d in val_metrics_history
    ]

    metrics_to_plot = ['r', 'r2', 'rmse', 'mse', 'mae', 'ccc']
    val_FVC_metrics = {metric: [d['FVC'][metric] for d in val_metrics_history] for metric in metrics_to_plot}
    val_FEV1_metrics = {metric: [d['FEV1'][metric] for d in val_metrics_history] for metric in metrics_to_plot}

    epochs_train = range(1, len(train_loss_history) + 1)
    epochs_val = range(1, len(val_loss_history) + 1)

    # 1. 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train, train_loss_history, label='Train Loss')
    if val_loss_history:
        plt.plot(epochs_val, val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "loss_history.png"))
    plt.close()
    print("\n损失曲线图已保存至: loss_history.png")

    # 2. 绘制 FVC 目标的六宫格指标曲线
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    fig.suptitle('FVC Validation Metrics History', fontsize=20)
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        axes[i].plot(epochs_val, val_FVC_metrics[metric], label='FVC ' + metric, )
        axes[i].set_title(metric, fontsize=14)
        axes[i].set_xlabel('Epoch', fontsize=10)
        axes[i].set_ylabel(metric, fontsize=10)
        axes[i].grid(True, linestyle='--')
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "fvc_metrics_history.png"))
    plt.close()
    print("FVC指标曲线图已保存至: fvc_metrics_history.png")

    # 3. 绘制 FEV1 目标的六宫格指标曲线
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    fig.suptitle('FEV1 Validation Metrics History', fontsize=20)
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        axes[i].plot(epochs_val, val_FEV1_metrics[metric], label='FEV1 ' + metric, )
        axes[i].set_title(metric, fontsize=14)
        axes[i].set_xlabel('Epoch', fontsize=10)
        axes[i].set_ylabel(metric, fontsize=10)
        axes[i].grid(True, linestyle='--')
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "fev1_metrics_history.png"))
    plt.close()
    print("FEV1指标曲线图已保存至: fev1_metrics_history.png")

if __name__ == "__main__":
    predict_and_evaluate()
    # 🚨 再次调用修正后的绘图函数 🚨
    plot_training_metrics()