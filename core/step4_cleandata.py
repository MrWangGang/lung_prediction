import matplotlib
import pandas as pd
from sklearn.ensemble import IsolationForest
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("tkagg")

# --- 忽略 Scikit-learn 的未来版本警告 ---
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 假设您的 CSV 文件路径 ---
file_path = './datasets_corp/临床信息.csv'

try:
    df = pd.read_csv(file_path)
    print("文件读取成功。")
except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path}。请检查文件路径是否正确。")
    exit()

# --- 1. 数据预处理 ---
# Binary encode '性别' (Male=1, Female=0)
df['性别_编码'] = df['性别'].map({'男': 1, '女': 0}).fillna(0)

# Encode '吸烟史'
smoking_map = {'从不吸烟': 0, '偶尔吸烟': 1, '经常吸烟': 2}
df['吸烟史_编码'] = df['吸烟史'].map(smoking_map).fillna(0)

# --- 2. 异常值检测和标记 ---
print("\n--- Outlier Statistics and Marking ---")

# Set up Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)

# Case 1: Only ['FEV1', 'FVC']
data_case1 = df[['FEV1', 'FVC']]
df['outlier_flag_case1'] = model.fit_predict(data_case1)
count_case1 = len(df[df['outlier_flag_case1'] == -1])
print(f"Number of outliers found by (FEV1, FVC) model: {count_case1}")

# Case 2: Only ['年龄', '体重']
data_case2 = df[['年龄', '体重']]
df['outlier_flag_case2'] = model.fit_predict(data_case2)
count_case2 = len(df[df['outlier_flag_case2'] == -1])
print(f"Number of outliers found by (Age, Weight) model: {count_case2}")

# Case 3: Four-feature combined model
data_case3 = df[['FEV1', 'FVC', '年龄', '体重']]
df['outlier_flag_case3'] = model.fit_predict(data_case3)
count_case3 = len(df[df['outlier_flag_case3'] == -1])
print(f"Number of outliers found by the combined (4-feature) model: {count_case3}")

# Case 4: Five-feature combined model
data_case4 = df[['FEV1', 'FVC', '年龄', '体重', '性别_编码']]
df['outlier_flag_case4'] = model.fit_predict(data_case4)
count_case4 = len(df[df['outlier_flag_case4'] == -1])
print(f"Number of outliers found by the combined (5-feature) model: {count_case4}")

# Case 5: Six-feature combined model
data_case5 = df[['FEV1', 'FVC', '年龄', '体重', '性别_编码', '吸烟史_编码']]
df['outlier_flag_case5'] = model.fit_predict(data_case5)
count_case5 = len(df[df['outlier_flag_case5'] == -1])
print(f"Number of outliers found by the combined (6-feature) model: {count_case5}")

# Case 6: Logical error FEV1 > FVC
logical_error_condition = (df['FEV1'] > df['FVC'])
count_logical = len(df[logical_error_condition])
print(f"Number of outliers found by logical error (FEV1 > FVC): {count_logical}")

# --- 3. 移除异常值并保存新文件 ---
outlier_mask = (
        (df['outlier_flag_case1'] == -1) |
        (df['outlier_flag_case2'] == -1) |
        (df['outlier_flag_case3'] == -1) |
        (df['outlier_flag_case4'] == -1) |
        (df['outlier_flag_case5'] == -1) |
        (df['FEV1'] > df['FVC'])
)
union_outliers = len(df[outlier_mask])
print("\n--- Final Outlier Summary ---")
print(f"Total unique rows marked as outliers: {union_outliers} rows")

print("\n--- Deletion Statistics by '分级' ---")
total_counts = df['分级'].value_counts().sort_index()
deleted_counts = df[outlier_mask]['分级'].value_counts().sort_index()
remaining_counts = df[~outlier_mask]['分级'].value_counts().sort_index()
summary_df = pd.DataFrame({
    'Original Count': total_counts,
    'Deleted Count': deleted_counts.reindex(total_counts.index, fill_value=0),
    'Remaining Count': remaining_counts.reindex(total_counts.index, fill_value=0)
})
print(summary_df)

df_cleaned = df[~outlier_mask]

# --- 移除临时列 ---
columns_to_drop = [
    '性别_编码',
    '吸烟史_编码',
    'outlier_flag_case1',
    'outlier_flag_case2',
    'outlier_flag_case3',
    'outlier_flag_case4',
    'outlier_flag_case5'
]
df_cleaned = df_cleaned.drop(columns=columns_to_drop, errors='ignore')

print("\n--- Cleaned Data Info ---")
print(f"Original total rows: {len(df)}")
print(f"Remaining rows after removing outliers: {len(df_cleaned)}")
new_file_path = './datasets_corp/临床信息_cleaned.csv'
df_cleaned.to_csv(new_file_path, index=False)
print(f"\nCleaned data has been successfully saved to '{new_file_path}'")

# --- Outlier Visualization Scatter Plots ---
print("\n--- Generating outlier visualization plots ---")

# 重新加载数据以确保绘图的原始数据不受 drop 操作影响
df_for_plotting = pd.read_csv(file_path)
df_for_plotting['性别_编码'] = df_for_plotting['性别'].map({'男': 1, '女': 0}).fillna(0)
smoking_map = {'从不吸烟': 0, '偶尔吸烟': 1, '经常吸烟': 2}
df_for_plotting['吸烟史_编码'] = df_for_plotting['吸烟史'].map(smoking_map).fillna(0)
# 重新进行异常值检测
model = IsolationForest(contamination=0.05, random_state=42)
df_for_plotting['outlier_flag_case1'] = model.fit_predict(df_for_plotting[['FEV1', 'FVC']])
df_for_plotting['outlier_flag_case2'] = model.fit_predict(df_for_plotting[['年龄', '体重']])
df_for_plotting['outlier_flag_case3'] = model.fit_predict(df_for_plotting[['FEV1', 'FVC', '年龄', '体重']])
df_for_plotting['outlier_flag_case4'] = model.fit_predict(df_for_plotting[['FEV1', 'FVC', '年龄', '体重', '性别_编码']])
df_for_plotting['outlier_flag_case5'] = model.fit_predict(df_for_plotting[['FEV1', 'FVC', '年龄', '体重', '性别_编码', '吸烟史_编码']])


fig, axes = plt.subplots(3, 2, figsize=(8, 12))
fig.suptitle('Outlier Visualization per Model', fontsize=14, y=1.02)
fig.tight_layout(pad=4.0, h_pad=3.0, w_pad=2.0)

def plot_individual_outliers(ax, x_data, y_data, outlier_flag, title, x_label=None, y_label=None):
    outliers = df_for_plotting[outlier_flag == -1]

    sns.scatterplot(x=df_for_plotting[x_data], y=df_for_plotting[y_data], ax=ax, color='lightgray', s=15, label='Data Points')
    sns.scatterplot(x=outliers[x_data], y=outliers[y_data], ax=ax, color='red', marker='X', s=40, label='Outliers')

    ax.set_xlabel(x_label if x_label else x_data, fontsize=7)
    ax.set_ylabel(y_label if y_label else y_data, fontsize=7)
    ax.legend(fontsize=6)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=9, verticalalignment='top')

# Plotting each case individually
plot_individual_outliers(axes[0, 0], 'FEV1', 'FVC', df_for_plotting['outlier_flag_case1'], 'Model: FEV1, FVC')
plot_individual_outliers(axes[0, 1], '年龄', '体重', df_for_plotting['outlier_flag_case2'], 'Model: Age, Weight', x_label='Age', y_label='Weight')
plot_individual_outliers(axes[1, 0], 'FEV1', 'FVC', df_for_plotting['outlier_flag_case3'], 'Model: FEV1, FVC, Age, Weight')
plot_individual_outliers(axes[1, 1], 'FEV1', 'FVC', df_for_plotting['outlier_flag_case4'], 'Model: FEV1, FVC, Age, Weight, Gender')
plot_individual_outliers(axes[2, 0], 'FEV1', 'FVC', df_for_plotting['outlier_flag_case5'], 'Model: FEV1, FVC, Age, Weight, Gender, Smoking')

# Plotting the logical error separately
outliers_logical = df_for_plotting[df_for_plotting['FEV1'] > df_for_plotting['FVC']]
sns.scatterplot(x=df_for_plotting['FEV1'], y=df_for_plotting['FVC'], ax=axes[2, 1], color='lightgray', s=15, label='Data Points')
sns.scatterplot(x=outliers_logical['FEV1'], y=outliers_logical['FVC'], ax=axes[2, 1], color='red', marker='X', s=40, label='Outliers (FEV1>FVC)')
axes[2, 1].set_xlabel('FEV1', fontsize=7)
axes[2, 1].set_ylabel('FVC', fontsize=7)
axes[2, 1].legend(fontsize=6)
axes[2, 1].grid(True, linestyle='--', alpha=0.6)
axes[2, 1].text(0.05, 0.95, 'Logical Outliers: FEV1 > FVC', transform=axes[2, 1].transAxes, fontsize=9, verticalalignment='top')

plt.show()