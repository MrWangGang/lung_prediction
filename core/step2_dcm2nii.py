import os
import dicom2nifti
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def convert_single_series_silent(args):
    """
    一个函数，用于处理单个DICOM系列到NIfTI的转换，
    只返回成功或失败状态，不打印详细信息。
    """
    dicom_folder_path, output_base_dir, parent_dir_name, dicom_series_folder_name = args

    try:
        # 构建输出NIfTI文件的类别目录
        output_category_dir = os.path.join(output_base_dir, parent_dir_name)
        os.makedirs(output_category_dir, exist_ok=True)

        # 构建输出NIfTI文件的完整路径
        output_nifti_filename = f"{dicom_series_folder_name}.nii.gz"
        output_nifti_path = os.path.join(output_category_dir, output_nifti_filename)

        dicom2nifti.dicom_series_to_nifti(dicom_folder_path, output_nifti_path, reorient_nifti=True)
        return True, dicom_folder_path # 返回成功状态和路径

    except Exception as e:
        # 内部捕获错误，但只返回失败状态，不在这里打印
        return False, dicom_folder_path # 返回失败状态和路径

def convert_dicom_to_nifti_multiprocess_concise(input_base_dir, output_base_dir, num_processes=None):
    """
    使用多进程将DICOM数据转换为NIfTI，保存到新目录，不删除源文件，
    只显示进度条和最终的成功/失败汇总。
    """
    if num_processes is None:
        num_processes = cpu_count() # 默认使用所有可用的CPU核心数
    print(f"将使用 {num_processes} 个进程进行转换。")

    # 确保输出根目录存在
    os.makedirs(output_base_dir, exist_ok=True)

    # 收集所有需要转换的DICOM系列文件夹及其相关信息
    tasks = []
    for root, dirs, files in os.walk(input_base_dir):
        if root == input_base_dir:
            continue
        if not dirs and files: # 确认是包含DICOM文件的最底层文件夹
            dicom_series_folder_name = os.path.basename(root)
            parent_dir_name = os.path.basename(os.path.dirname(root))
            tasks.append((root, output_base_dir, parent_dir_name, dicom_series_folder_name))

    if not tasks:
        print(f"在 {input_base_dir} 中未找到任何 DICOM 系列文件夹。请检查您的目录结构。")
        return

    print(f"找到 {len(tasks)} 个 DICOM 系列需要转换。")
    print("\n--- 开始并行转换 ---\n")

    # 创建一个进程池
    with Pool(processes=num_processes) as pool:
        # 使用 tqdm 包装 pool.imap_unordered，只显示进度条
        results = list(tqdm(pool.imap_unordered(convert_single_series_silent, tasks),
                            total=len(tasks),
                            desc="正在转换 DICOM 到 NIfTI"))

    # 汇总结果
    successful_conversions = [res[1] for res in results if res[0]]
    failed_conversions = [res[1] for res in results if not res[0]]

    print("\n--- 所有转换完成！ ---\n")
    print(f"总共处理了: {len(results)} 个系列")
    print(f"成功转换: {len(successful_conversions)} 个系列")
    if failed_conversions:
        print(f"转换失败: {len(failed_conversions)} 个系列")
        # 如果有失败，可以考虑在这里选择性地打印失败路径
        # print("失败详情:")
        # for path in failed_conversions:
        #     print(f"- {path}")

# --- 将执行代码放入 if __name__ == '__main__': 块中 ---
if __name__ == '__main__':
    # --- 定义输入和输出目录 ---
    input_data_directory = './datasets'
    output_data_directory = './datasets_nii'

    # --- 执行转换 ---
    convert_dicom_to_nifti_multiprocess_concise(input_data_directory, output_data_directory)