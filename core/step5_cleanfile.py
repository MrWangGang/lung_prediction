import os
import pandas as pd
import re

# 定义文件路径
csv_path = './datasets_corp/临床信息.csv'
datasets_dir = './datasets_corp/datasets'

# --- 1. 读取CSV文件并处理'序号'列 ---
try:
    df = pd.read_csv(csv_path)
    
    # 将'序号'列转换为整数，处理 '1.0' 这样的值，同时处理无法转换的异常
    try:
        df['序号'] = df['序号'].astype(float).astype(int)
    except ValueError:
        print("警告: '序号'列包含非数字值，将使用字符串类型进行匹配。")
        df['序号'] = df['序号'].astype(str)
    
    csv_ids_set = set(df['序号'].tolist())
    
except FileNotFoundError:
    print(f"错误: 找不到文件 {csv_path}")
    exit()
except KeyError:
    print(f"错误: CSV文件中没有名为 '序号' 的列。")
    exit()

# --- 2. 收集所有nii.gz和mask文件的前缀，并处理为数字 ---
file_prefixes_set = set()
for filename in os.listdir(datasets_dir):
    # 使用正则表达式匹配文件名开头的所有数字
    match = re.match(r'(\d+)', filename)
    if match:
        try:
            # 提取匹配到的数字并转换为整数
            file_prefixes_set.add(int(match.group(1)))
        except ValueError:
            print(f"警告: 文件名 '{filename}' 的前缀无法转换为数字，将被忽略。")
            
# --- 3. 找出需要保留的共同前缀 ---
common_ids = csv_ids_set.intersection(file_prefixes_set)

# --- 4. 统计并告知删除信息 ---
total_csv_rows = len(csv_ids_set)
total_file_pairs = len(file_prefixes_set)
files_to_delete_count = total_file_pairs - len(common_ids)
csv_rows_to_delete = total_csv_rows - len(common_ids)

print(f"CSV文件中共有 {total_csv_rows} 条数据。")
print(f"文件目录中共有 {total_file_pairs} 对文件。")

if not common_ids:
    print("\nCSV数据和文件目录中没有匹配的共同前缀，将删除所有数据和文件。")
    
else:
    print(f"\n--- 删除前概览 ---")
    print(f"要删除的文件数量: {files_to_delete_count * 2} 个 ({files_to_delete_count} 对文件)")
    print(f"要删除的CSV数据行数: {csv_rows_to_delete} 条")
    
    print(f"\n--- 删除后结果 ---")
    print(f"最终会剩下 {len(common_ids)} 对文件 和 {len(common_ids)} 条数据。")

# --- 5. 获取用户确认，执行删除操作 ---
confirmation = input("\n确认删除吗？(y/n): ")

if confirmation.lower() == 'y':
    if files_to_delete_count > 0:
        prefixes_to_delete_files = file_prefixes_set - common_ids
        print("\n--- 正在删除文件 ---")
        for prefix in prefixes_to_delete_files:
            prefix_str = str(prefix)
            # 遍历所有文件，删除以该前缀开头的文件
            for filename in os.listdir(datasets_dir):
                if filename.startswith(prefix_str + '.') or filename.startswith(prefix_str + '_'):
                    file_path = os.path.join(datasets_dir, filename)
                    print(f"正在删除文件: {file_path}")
                    os.remove(file_path)

    if csv_rows_to_delete > 0:
        print("\n--- 正在保存过滤后的CSV文件 ---")
        # 使用isin()方法过滤出保留的行
        df_filtered = df[df['序号'].isin(common_ids)]
        df_filtered.to_csv(csv_path, index=False)
        print(f"已保存过滤后的CSV文件到: {csv_path}")
        
    print("\n所有删除操作已完成。")
else:
    print("\n操作已取消，未进行任何修改。")

print("\n脚本运行结束。")