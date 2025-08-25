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
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error
)
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
from monai.utils import set_determinism
import json
from sklearn.preprocessing import LabelEncoder
import math

# 导入您的多任务模型
import MultitaskModel as model

# --- 补丁代码 (保持不变) ---
orig_torch_load = torch.load
def torch_wrapper(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_torch_load(*args, **kwargs)
torch.load = torch_wrapper

# --- 配置 (与训练脚本保持一致) ---
task = '2multitask'
image_network_name = 'ResNet'
DATA_ROOT_PATH = "./datasets_corp/datasets"
TEST_CSV_FILE_PATH = "./datasets_corp/测试集.csv"
MODEL_OUTPUT_DIR = f"./model/{task}/{image_network_name}"
REPORT_OUTPUT_DIR = f"./report/{task}/{image_network_name}/plot"
root_dir = "./cache"
persistent_cache = root_dir
RANDOM_STATE = 42

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
CLASSIFY_TARGET_COLUMN = '类型'
REGRESSION_TARGET_COLUMNS = ['FEV1', 'FVC']

# 设置随机种子以保证可复现性
set_determinism(seed=RANDOM_STATE)

# --- 辅助函数 ---
def plot_roc_curve_multiclass(y_true, y_scores, n_classes, class_names, filename):
    """绘制带95%置信区间的多分类ROC曲线并保存。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    colors = plt.get_cmap('Set1', n_classes)
    le = LabelEncoder()
    le.fit(class_names)
    y_true_numerical = le.transform(y_true)

    for i in range(n_classes):
        tprs, aurocs = [], []
        base_fpr = np.linspace(0, 1, 101)
        y_true_binary = (y_true_numerical == i).astype(int)

        if len(np.unique(y_true_binary)) < 2:
            continue

        n_bootstrap = 2000
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true_binary), len(y_true_binary), replace=True)
            if len(np.unique(y_true_binary[indices])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true_binary[indices], y_scores[indices, i])
            tpr_interpolated = np.interp(base_fpr, fpr, tpr)
            tprs.append(tpr_interpolated)
            aurocs.append(auc(fpr, tpr))

        if not tprs:
            continue

        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        mean_auroc = np.mean(aurocs)
        auroc_lower = np.percentile(aurocs, 2.5)
        auroc_upper = np.percentile(aurocs, 97.5)

        plt.plot(base_fpr, mean_tpr, color=colors(i),
                 label=f'{class_names[i]} (AUC = {mean_auroc:.2f}, 95% CI: {auroc_lower:.2f}-{auroc_upper:.2f})')
        tprs_lower = np.percentile(tprs, 2.5, axis=0)
        tprs_upper = np.percentile(tprs, 97.5, axis=0)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors(i), alpha=0.15)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Chance')
    plt.title('Multi-Class ROC Curve (95% CI)', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, filename):
    """绘制混淆矩阵图并保存。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- 数据加载与准备 ---
def load_and_prepare_test_data(data_root_path, csv_file_path, csv_input_scaler, regression_target_scaler, one_hot_cols):
    """加载测试集的影像和CSV数据。"""
    try:
        current_df = pd.read_csv(csv_file_path, index_col='序号')
    except FileNotFoundError:
        print(f"错误: 未找到测试集 CSV 文件 '{csv_file_path}'。请检查路径。")
        return [], None, None, None

    data_list = []
    search_pattern = os.path.join(data_root_path, "*.nii.gz")
    all_nii_files = glob.glob(search_pattern)
    image_files = {int(os.path.basename(f).split('.')[0]): f for f in all_nii_files if '_mask' not in f}
    mask_files = {int(os.path.basename(f).split('_')[0]): f for f in all_nii_files if '_mask' in f}

    for numeric_id in sorted(current_df.index.tolist()):
        image_path = image_files.get(numeric_id)
        mask_path = mask_files.get(numeric_id)
        if image_path and mask_path:
            data_list.append({
                "id": str(numeric_id),
                "image": image_path,
                "mask": mask_path,
                "csv_data": current_df.loc[numeric_id]
            })

    if not data_list:
        print("错误: 找不到任何匹配的影像、掩码和CSV数据。")
        return [], None, None, None

    raw_test_df = pd.DataFrame([d['csv_data'] for d in data_list], index=[d['id'] for d in data_list])

    preprocessed_df = pd.get_dummies(raw_test_df, columns=ONE_HOT_INPUT_FEATURES, prefix=ONE_HOT_INPUT_FEATURES, dummy_na=False)
    missing_cols = set(one_hot_cols) - set(preprocessed_df.columns)
    for c in missing_cols:
        preprocessed_df[c] = 0
    preprocessed_df = preprocessed_df[one_hot_cols + list(preprocessed_df.columns.drop(one_hot_cols))]

    final_csv_input_features = NORMAL_CSV_INPUT_FEATURES + one_hot_cols
    final_csv_input_features = [col for col in final_csv_input_features if col in preprocessed_df.columns]
    preprocessed_df[final_csv_input_features] = csv_input_scaler.transform(preprocessed_df[final_csv_input_features])

    # 标准化回归目标列
    preprocessed_df[REGRESSION_TARGET_COLUMNS] = regression_target_scaler.transform(preprocessed_df[REGRESSION_TARGET_COLUMNS])

    test_files = []
    id_to_data = {d['id']: d for d in data_list}
    for idx in raw_test_df.index:
        raw_info = id_to_data[str(idx)]
        test_files.append({
            "image": raw_info['image'],
            "mask": raw_info['mask'],
            "csv_input_features": preprocessed_df.loc[str(idx), final_csv_input_features].values.astype(np.float32),
            "classification_targets": raw_test_df.loc[idx, CLASSIFY_TARGET_COLUMN],
            "regression_targets": preprocessed_df.loc[str(idx), REGRESSION_TARGET_COLUMNS].values.astype(np.float32),
            "id": str(idx)
        })

    return test_files, final_csv_input_features, raw_test_df, preprocessed_df


# --- 主执行函数 ---
def predict_and_evaluate():
    if not os.path.exists(MODEL_OUTPUT_DIR):
        print(f"错误: 找不到模型目录 '{MODEL_OUTPUT_DIR}'。")
        return

    if not os.path.exists(REPORT_OUTPUT_DIR):
        os.makedirs(REPORT_OUTPUT_DIR)
        print(f"已创建报告目录: {REPORT_OUTPUT_DIR}")

    # --- 1. 加载标准化器和标签编码器 ---
    try:
        with open(os.path.join(MODEL_OUTPUT_DIR, "input_scaler.pkl"), 'rb') as f:
            csv_input_scaler = pickle.load(f)
        with open(os.path.join(MODEL_OUTPUT_DIR, "regression_target_scaler.pkl"), 'rb') as f:
            regression_target_scaler = pickle.load(f)
        with open(os.path.join(MODEL_OUTPUT_DIR, "label_encoder.pkl"), 'rb') as f:
            label_encoder = pickle.load(f)
        print("标准化器和标签编码器加载成功。")
        one_hot_cols = [col for col in csv_input_scaler.feature_names_in_ if any(f in col for f in ONE_HOT_INPUT_FEATURES)]
    except FileNotFoundError:
        print("错误: 找不到必要的标准化器或标签编码器文件。")
        return

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    num_regression_targets = len(REGRESSION_TARGET_COLUMNS)

    # --- 2. 加载和准备测试数据 ---
    test_files, all_csv_features_list, raw_test_df, preprocessed_test_df = load_and_prepare_test_data(DATA_ROOT_PATH, TEST_CSV_FILE_PATH, csv_input_scaler, regression_target_scaler, one_hot_cols)
    if not test_files:
        print("无法进行预测，请检查数据加载错误信息。")
        return

    num_csv_features = len(all_csv_features_list)
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
            # 更新ToTensord的键以匹配新数据结构
            ToTensord(keys=["image", "classification_targets", "regression_targets", "csv_input_features"]),
        ]
    )

    test_ds = PersistentDataset(data=test_files, transform=transforms, cache_dir=os.path.join(persistent_cache, "test_predict"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    # --- 3. 进行预测 ---
    print("\n--- 开始在测试集上进行预测 ---")
    all_classification_preds_labels = []
    all_classification_targets_labels = []
    all_classification_preds_probs = []

    all_regression_preds = []
    all_regression_targets = []

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="处理测试集"):
            inputs, classification_targets_raw, regression_targets_raw, csv_input_features = (
                batch_data["image"].to(DEVICE),
                batch_data["classification_targets"],
                batch_data["regression_targets"].to(DEVICE),
                batch_data["csv_input_features"].to(DEVICE)
            )

            # 模型返回两个输出
            regression_outputs, classification_outputs, _ = model_to_load(inputs, csv_input_features)

            # 记录分类结果
            preds = torch.argmax(classification_outputs, dim=1).cpu().numpy()
            probs = torch.nn.functional.softmax(classification_outputs, dim=1).cpu().numpy()
            all_classification_preds_labels.extend(label_encoder.inverse_transform(preds))
            all_classification_targets_labels.extend(classification_targets_raw)
            all_classification_preds_probs.extend(probs)

            # 记录回归结果
            all_regression_preds.extend(regression_outputs.cpu().numpy())
            all_regression_targets.extend(regression_targets_raw.cpu().numpy())

    all_classification_preds_labels = np.array(all_classification_preds_labels)
    all_classification_targets_labels = np.array(all_classification_targets_labels)
    all_classification_preds_probs = np.array(all_classification_preds_probs)

    all_regression_preds = np.array(all_regression_preds)
    all_regression_targets = np.array(all_regression_targets)

    # --- 4. 生成和保存测试集报告 ---
    print("\n--- 生成并保存测试集分类和回归报告 ---")

    report_path = os.path.join(REPORT_OUTPUT_DIR, "test_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Test Set Classification Report ---\n\n")
        f.write(classification_report(all_classification_targets_labels, all_classification_preds_labels, labels=class_names, zero_division=0))

        f.write("\n\n--- Test Set Regression Metrics (on normalized values) ---\n")
        f.write(f"Mean Absolute Error (MAE): {mean_absolute_error(all_regression_targets, all_regression_preds):.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {math.sqrt(mean_squared_error(all_regression_targets, all_regression_preds)):.4f}\n")

        # 逆归一化并计算真实值的回归指标
        f.write("\n\n--- Test Set Regression Metrics (on original values) ---\n")
        original_regression_targets = regression_target_scaler.inverse_transform(all_regression_targets)
        original_regression_preds = regression_target_scaler.inverse_transform(all_regression_preds)

        for i, col in enumerate(REGRESSION_TARGET_COLUMNS):
            mae = mean_absolute_error(original_regression_targets[:, i], original_regression_preds[:, i])
            rmse = math.sqrt(mean_squared_error(original_regression_targets[:, i], original_regression_preds[:, i]))
            f.write(f"\n{col}:\n")
            f.write(f"  MAE: {mae:.2f}\n")
            f.write(f"  RMSE: {rmse:.2f}\n")

    print(f"报告已保存至: {report_path}")

    plot_confusion_matrix(all_classification_targets_labels, all_classification_preds_labels, class_names, os.path.join(REPORT_OUTPUT_DIR, "test_confusion_matrix.png"))
    print("混淆矩阵图已保存。")

    plot_roc_curve_multiclass(all_classification_targets_labels, all_classification_preds_probs, num_classes, class_names, os.path.join(REPORT_OUTPUT_DIR, "test_roc_curve.png"))
    print("ROC曲线图已保存。")

    # 新增的回归预测值与真实值散点图
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, len(REGRESSION_TARGET_COLUMNS), figsize=(6 * len(REGRESSION_TARGET_COLUMNS), 6), tight_layout=True)
    if len(REGRESSION_TARGET_COLUMNS) == 1:
        axes = [axes]

    for i, col in enumerate(REGRESSION_TARGET_COLUMNS):
        axes[i].scatter(original_regression_targets[:, i], original_regression_preds[:, i], alpha=0.6)
        axes[i].plot(axes[i].get_xlim(), axes[i].get_ylim(), ls="--", c=".3", label="Ideal")
        axes[i].set_title(f'Predicted vs True {col}', fontsize=14)
        axes[i].set_xlabel(f'True {col}', fontsize=12)
        axes[i].set_ylabel(f'Predicted {col}', fontsize=12)
        axes[i].legend()

    plt.suptitle('Regression Prediction Scatter Plots (on original values)', fontsize=16, y=1.02)
    plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "regression_scatter_plots.png"))
    plt.close(fig)
    print("回归预测散点图已保存。")


def plot_training_metrics():
    """根据 training_metrics_history.json 绘制训练和验证指标曲线图。"""
    plt.style.use('seaborn-v0_8-whitegrid')
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
    val_loss_history = data.get('val_loss_history', [])
    val_report_history = data.get('val_report_history', [])

    min_len = min(len(train_loss_history), len(val_loss_history), len(val_report_history))
    train_loss_history = train_loss_history[:min_len]
    val_loss_history = val_loss_history[:min_len]
    val_report_history = val_report_history[:min_len]
    epochs = range(1, min_len + 1)

    if not train_loss_history or not val_report_history:
        print("\n警告: JSON 文件中没有完整的训练或验证数据。")
        return

    val_accuracy_history = [d['accuracy'] for d in val_report_history]
    val_f1_macro_history = [d['macro avg']['f1-score'] for d in val_report_history]

    class_names = [k for k in val_report_history[0].keys() if k not in ['accuracy', 'macro avg', 'weighted avg', 'loss']]
    class_metrics = {name: {'precision': [], 'recall': [], 'f1-score': []} for name in class_names}
    for report in val_report_history:
        for name in class_names:
            if name in report:
                class_metrics[name]['precision'].append(report[name]['precision'])
                class_metrics[name]['recall'].append(report[name]['recall'])
                class_metrics[name]['f1-score'].append(report[name]['f1-score'])

    # 1. 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_history, label='Train Total Loss')
    plt.plot(epochs, val_loss_history, label='Validation Total Loss')
    plt.title('Training and Validation Total Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "loss_history.png"))
    plt.close()
    print("\n总损失曲线图已保存至: loss_history.png")

    # 2. 绘制整体准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accuracy_history, label='Validation Accuracy')
    plt.title('Overall Validation Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "overall_accuracy_history.png"))
    plt.close()
    print("整体验证准确率曲线图已保存至: overall_accuracy_history.png")

    # 3. 绘制整体F1-Score曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_f1_macro_history, label='Validation Macro F1-Score')
    plt.title('Overall Validation Macro F1-Score', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Macro F1-Score', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "overall_f1_history.png"))
    plt.close()
    print("整体验证宏观F1-Score曲线图已保存至: overall_f1_history.png")

    # 4. 绘制每个类别的精确度、召回率、F1-Score
    for metric in ['precision', 'recall', 'f1-score']:
        plt.figure(figsize=(10, 6))
        for class_name in class_names:
            if class_name in class_metrics:
                plt.plot(epochs, class_metrics[class_name][metric], label=class_name)
        plt.title(f'Validation {metric.capitalize()} per Class', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_OUTPUT_DIR, f"{metric}_history_per_class.png"))
        plt.close()
    print("每个类别的精确度、召回率和F1-Score曲线图已保存。")

if __name__ == "__main__":
    predict_and_evaluate()
    plot_training_metrics()