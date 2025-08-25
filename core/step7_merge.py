import pandas as pd
import os

# 定义文件路径
file_path_clinical = './datasets_corp/临床信息_cleaned.csv'
file_path_lung_features = './datasets_corp/lung_features.csv'

# 检查文件是否存在
if not os.path.exists(file_path_clinical) or not os.path.exists(file_path_lung_features):
    print("错误：文件不存在，请检查路径是否正确。")
else:
    # 读取两个 CSV 文件
    df_clinical = pd.read_csv(file_path_clinical)
    df_lung_features = pd.read_csv(file_path_lung_features)

    # 1. 重命名 lung_features 中的 'file_id' 列，以便与 '临床信息_cleaned' 的 '序号' 列匹配
    # 这步是关键，它让两个文件有了相同的连接键
    df_lung_features.rename(columns={'file_id': '序号'}, inplace=True)

    # 2. 合并两个 DataFrame
    # 使用 '序号' 列作为连接键
    # 'how='left'' 表示保留所有 'df_clinical' 的行，并根据 '序号' 匹配 'df_lung_features' 中的数据
    merged_df = pd.merge(df_clinical, df_lung_features, on='序号', how='left')

    # 3. 将合并后的 DataFrame 保存为新的 CSV 文件
    output_file_path = './datasets_corp/临床信息_final.csv'
    merged_df.to_csv(output_file_path, index=False)

    print(f"数据合并完成！新文件已保存到：{output_file_path}")
    print("新文件的表头（列名）如下：")
    print(merged_df.columns)