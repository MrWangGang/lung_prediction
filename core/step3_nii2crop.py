import os
import sys
from importlib import metadata
import ants # 用于图像处理和裁剪
import numpy as np
import SimpleITK as sitk # 用于读取和写入NIfTI文件
from lungmask import LMInferer, utils # lungmask库
from lungmask.logger import logger # lungmask的日志工具
from tqdm import tqdm # 进度条
from concurrent.futures import ThreadPoolExecutor, as_completed # 多线程处理
import shutil # 用于文件移动

# --- 1. 定义硬编码的全局参数 ---
# 请根据你的实际路径修改以下两个变量！
HARDCODED_INPUT_PATH = './datasets_nii'  # 你的默认输入目录或文件
HARDCODED_OUTPUT_DIR = './datasets_corp'  # 你的默认输出目录，裁剪后的文件将保存在这里
# lungmask模型相关参数
HARDCODED_MODEL_NAME = "R231"
HARDCODED_MODEL_PATH = "./model/unet_r231-d5d2fc3d.pth" # 确保此路径指向你的模型文件
HARDCODED_FORCE_CPU = False
HARDCODED_NO_POSTPROCESS = False
HARDCODED_BATCH_SIZE = 20
HARDCODED_NO_PROGRESS = True # 内部lungmask进度条，外部tqdm会覆盖
HARDCODED_REMOVE_METADATA = True
HARDCODED_SAVE_MASK = True   # 控制是否保存 裁剪后的掩膜 (image_mask.nii.gz)
HARDCODED_MAX_WORKERS = 8    # 多线程工作者数量
# --- 硬编码参数定义结束 ---

def path(string):
    """检查文件路径是否存在，不存在则退出。"""
    if os.path.exists(string):
        return string
    else:
        sys.exit(f"File not found: {string}")

def apply_mask_and_crop(input_image_path, temp_original_mask_path, output_cropped_image_path, temp_cropped_mask_path):
    """
    加载原始图像和mask，应用mask，然后裁剪图像和mask，并保存裁剪后的结果。
    裁剪操作会使得输出图像和掩码具有相同的最小包围盒维度。
    temp_original_mask_path: lungmask生成的原始尺寸掩膜的临时路径
    temp_cropped_mask_path: 裁剪后的掩膜的临时保存路径 (稍后根据 HARDCODED_SAVE_MASK 决定是否移动到最终位置)
    """
    # 加载原始图像和 mask (这里加载的是 lungmask 生成的原始mask，未裁剪)
    image = ants.image_read(input_image_path)
    mask = ants.image_read(temp_original_mask_path) # 从临时原始掩膜路径加载

    # 应用 mask 到原始图像 (这会保留原始图像的维度，只是将非mask区域置零)
    masked_image = image * mask

    # 裁剪图像和掩码，使用 mask 作为裁剪参考
    cropped_image = ants.crop_image(masked_image, mask)
    cropped_mask = ants.crop_image(mask, mask) # 对 mask 本身也进行相同的裁剪

    # 保存裁剪后的图像
    ants.image_write(cropped_image, output_cropped_image_path)
    logger.info(f"Cropped image saved to: {output_cropped_image_path}")

    # 保存裁剪后的掩码到临时路径
    ants.image_write(cropped_mask, temp_cropped_mask_path)
    logger.info(f"Cropped mask temporarily saved to: {temp_cropped_mask_path}")


def process_file(input_image_path, output_final_mask_path, output_cropped_image_path):
    """
    处理单个NIfTI图像文件：生成肺部掩码，应用并裁剪图像和掩码。
    output_final_mask_path: 裁剪后最终掩膜 (image_mask.nii.gz) 的目标路径
    """
    # 检查裁剪后的输出文件是否已存在，以避免重复处理
    # 注意：这里只检查最终输出的图像和掩膜
    if os.path.exists(output_cropped_image_path) and \
            (not HARDCODED_SAVE_MASK or os.path.exists(output_final_mask_path)):
        logger.info(f"Skipping {input_image_path}: Cropped output image and (if configured) final mask already exist.")
        return True

    logger.info(f"Processing {input_image_path}")
    temp_original_mask_for_cropping_path = None # 声明变量以确保其在 finally 块中可见
    temp_cropped_mask_path = None # 声明变量以确保其在 finally 块中可见

    try:
        input_image = sitk.ReadImage(input_image_path)

        logger.info(f"Infer lungmask for {os.path.basename(input_image_path)}")
        inferer = LMInferer(
            modelname=HARDCODED_MODEL_NAME,
            modelpath=HARDCODED_MODEL_PATH,
            force_cpu=HARDCODED_FORCE_CPU,
            batch_size=HARDCODED_BATCH_SIZE,
            volume_postprocessing=not (HARDCODED_NO_POSTPROCESS),
            tqdm_disable=True, # 禁用lungmask内部的tqdm，使用外部的tqdm
        )
        result_array = inferer.apply(sitk.GetArrayFromImage(input_image))
        result_array[result_array > 0] = 1 # 确保掩码是二值的 (前景为1，背景为0)

        original_mask_sitk = sitk.GetImageFromArray(result_array)
        original_mask_sitk.CopyInformation(input_image)

        # 总是将原始尺寸掩码保存到临时文件，仅用于裁剪，不作最终保存
        base_name = os.path.basename(input_image_path)
        file_base, file_ext = os.path.splitext(base_name)
        if file_ext == '.gz':
            file_base, nii_ext = os.path.splitext(file_base)
            file_ext = nii_ext + file_ext

        # 定义临时原始尺寸掩膜路径
        temp_original_mask_for_cropping_path = os.path.join(os.path.dirname(output_cropped_image_path), f"{file_base}_temp_original_mask{file_ext}")

        # 定义临时裁剪后掩膜路径
        temp_cropped_mask_path = os.path.join(os.path.dirname(output_cropped_image_path), f"{file_base}_temp_cropped_mask{file_ext}")

        writer = sitk.ImageFileWriter()
        writer.SetFileName(temp_original_mask_for_cropping_path)
        # 如果需要保留元数据 (仅针对 lungmask 原始输出，但这里是临时文件，通常不需要)
        if not HARDCODED_REMOVE_METADATA:
            writer.SetKeepOriginalImageUID(True)
            DICOM_tags_to_keep = utils.get_DICOM_tags_to_keep()
            for key in input_image.GetMetaDataKeys():
                if key in DICOM_tags_to_keep:
                    original_mask_sitk.SetMetaData(key, input_image.GetMetaData(key))
            original_mask_sitk.SetMetaData("0008|103e", "Created with lungmask")
            original_mask_sitk.SetMetaData("0028|1050", "1")
            original_mask_sitk.SetMetaData("0028|1051", "2")

        writer.Execute(original_mask_sitk)
        logger.info(f"Original size mask temporarily saved to: {temp_original_mask_for_cropping_path}")


        # 调用裁剪函数，它会保存裁剪后的图像和临时裁剪后的掩码
        apply_mask_and_crop(input_image_path,
                            temp_original_mask_for_cropping_path,
                            output_cropped_image_path,
                            temp_cropped_mask_path)

        # 根据 HARDCODED_SAVE_MASK 决定是否将临时裁剪后的掩码移动到最终位置
        if HARDCODED_SAVE_MASK:
            shutil.move(temp_cropped_mask_path, output_final_mask_path)
            logger.info(f"Final cropped mask moved to: {output_final_mask_path}")
        else:
            logger.info("Skipping saving final cropped mask as HARDCODED_SAVE_MASK is False.")


        logger.info(f"Successfully processed {os.path.basename(input_image_path)}.")
        return True
    except Exception as e:
        logger.error(f"Failed to process {input_image_path}: {e}", exc_info=True) # 打印详细错误信息
        return False
    finally:
        # 无论成功失败，删除所有临时文件
        if temp_original_mask_for_cropping_path and os.path.exists(temp_original_mask_for_cropping_path):
            os.remove(temp_original_mask_for_cropping_path)
            logger.info(f"Removed temporary original size mask file: {temp_original_mask_for_cropping_path}")
        if temp_cropped_mask_path and os.path.exists(temp_cropped_mask_path):
            # 如果 HARDCODED_SAVE_MASK 为 True，文件已经被移动了，这里会是 False
            # 如果 HARDCODED_SAVE_MASK 为 False，这里会删除文件
            os.remove(temp_cropped_mask_path)
            logger.info(f"Removed temporary cropped mask file: {temp_cropped_mask_path}")


def main():
    """主函数：处理单个文件或整个目录下的NIfTI文件。"""
    version = metadata.version("lungmask")
    print(f"Lungmask Version: {version}")

    success_count = 0
    failure_count = 0

    if os.path.isdir(HARDCODED_INPUT_PATH):
        # 处理整个目录
        all_files_to_process = []
        for root, dirs, files in os.walk(HARDCODED_INPUT_PATH):
            for file_name in files:
                if file_name.endswith(".nii.gz"):
                    all_files_to_process.append(os.path.join(root, file_name))

        if not all_files_to_process:
            print(f"No .nii.gz files found in {HARDCODED_INPUT_PATH}. Exiting.")
            return

        print(f"Found {len(all_files_to_process)} files to process.")

        with ThreadPoolExecutor(max_workers=HARDCODED_MAX_WORKERS) as executor:
            futures = []
            for input_image_path in all_files_to_process:
                relative_path = os.path.relpath(input_image_path, HARDCODED_INPUT_PATH)
                output_sub_dir = os.path.join(HARDCODED_OUTPUT_DIR, os.path.dirname(relative_path))
                os.makedirs(output_sub_dir, exist_ok=True) # 确保输出子目录存在

                base_name = os.path.basename(input_image_path)
                file_base, file_ext = os.path.splitext(base_name)
                if file_ext == '.gz': # 处理 .nii.gz 这种双扩展名
                    file_base, nii_ext = os.path.splitext(file_base)
                    file_ext = nii_ext + file_ext

                # 定义裁剪后的图像的输出路径 (文件名与原始图像相同)
                output_cropped_image_name = base_name
                output_cropped_image_path = os.path.join(output_sub_dir, output_cropped_image_name)

                # 定义最终裁剪后掩码的输出路径 (命名为 _mask)
                output_final_mask_name = f"{file_base}_mask{file_ext}"
                output_final_mask_path = os.path.join(output_sub_dir, output_final_mask_name)

                futures.append(executor.submit(process_file, input_image_path,
                                               output_final_mask_path,
                                               output_cropped_image_path))

            # 使用 tqdm 显示总体进度
            for future in tqdm(as_completed(futures), total=len(futures), desc="Overall Progress"):
                if future.result():
                    success_count += 1
                else:
                    failure_count += 1

        print(f"\n--- Processing Summary ---")
        print(f"Total files processed (attempted): {len(all_files_to_process)}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed to process: {failure_count}")

    else: # 处理单个文件
        os.makedirs(HARDCODED_OUTPUT_DIR, exist_ok=True)
        input_image_path = HARDCODED_INPUT_PATH
        if not os.path.exists(input_image_path):
            print(f"Error: Single input file not found: {input_image_path}. Exiting.")
            sys.exit(1)

        file_name = os.path.basename(input_image_path)
        file_base, file_ext = os.path.splitext(file_name)
        if file_ext == '.gz':
            file_base, nii_ext = os.path.splitext(file_base)
            file_ext = nii_ext + file_ext

        # 定义裁剪后的图像的输出路径
        output_cropped_image_name = file_name
        output_cropped_image_path = os.path.join(HARDCODED_OUTPUT_DIR, output_cropped_image_name)

        # 定义最终裁剪后掩码的输出路径
        output_final_mask_name = f"{file_base}_mask{file_ext}"
        output_final_mask_path = os.path.join(HARDCODED_OUTPUT_DIR, output_final_mask_name)

        print(f"Processing single file: {input_image_path}")
        if process_file(input_image_path,
                        output_final_mask_path,
                        output_cropped_image_path):
            success_count += 1
            print("Single file processed successfully.")
        else:
            failure_count += 1
            print("Single file processing failed.")

        print(f"\n--- Processing Summary ---")
        print(f"Successfully processed: {success_count}")
        print(f"Failed to process: {failure_count}")

if __name__ == "__main__":
    main()