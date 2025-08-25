import os
import shutil

def move_dcm_files(root_dir):
    """
    将指定目录下的所有Dicom文件从子文件夹移动到其父级文件夹。

    Args:
        root_dir (str): 包含数据集的根目录路径。
    """
    # 遍历根目录下的所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查当前文件夹是否包含.dcm文件
        dcm_files = [f for f in filenames if f.lower().endswith('.dcm')]
        
        if dcm_files:
            # 找到父级文件夹
            parent_dir = os.path.dirname(dirpath)
            
            # 遍历并移动每个Dicom文件
            for dcm_file in dcm_files:
                src_path = os.path.join(dirpath, dcm_file)
                dst_path = os.path.join(parent_dir, dcm_file)
                
                # 确保目标文件不存在，避免覆盖
                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
                    print(f"移动文件：{src_path} -> {dst_path}")
                else:
                    print(f"目标文件已存在，跳过：{dst_path}")

            # 移动完文件后，删除原来的子文件夹，如果它已变为空
            try:
                os.rmdir(dirpath)
                print(f"删除空文件夹：{dirpath}")
            except OSError as e:
                print(f"无法删除文件夹 {dirpath}：{e}")

# 设置你的数据集根目录
datasets_root = './datasets1' 

# 运行函数
move_dcm_files(datasets_root)