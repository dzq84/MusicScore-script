import os
import shutil
import fitz  # PyMuPDF
import multiprocessing

def is_scanned_black_white(pdf_path):
    """
    检查PDF是否为扫描的黑白文档
    """
    is_bw = True  # 假设文档是黑白的
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                # 搜索页面上所有的图像
                images = page.get_images(full=True)
                for img_index, base_xref, xref, smask, width, height, bpc, colorspace, alt, img_name, _, _ in images:
                    if bpc > 1:  # 如果每个颜色通道的位数大于1，不是1-bit黑白图像
                        is_bw = False
                        break
                if not is_bw:
                    break
        return is_bw
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return False

def worker(pdf_path, target_folder):
    try:
        if is_scanned_black_white(pdf_path):
            # 复制文件到目标文件夹
            target_path = os.path.join(target_folder, os.path.basename(pdf_path))
            shutil.copy(pdf_path, target_path)
            print(f"Copied scanned black and white PDF to target folder: {os.path.basename(pdf_path)}")
        else:
            print(f"Skipped {os.path.basename(pdf_path)}: not scanned black and white")    
    except Exception as e:
        print(f"Error processing {os.path.basename(pdf_path)}: {e}")

if __name__ == "__main__":
    source_folder = "/datasets/score_data/data"
    target_folder = "/datasets/score_data/hd_data/hd_data_pdf"

    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    pdf_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.lower().endswith('.pdf')]

    # 创建进程池
    with multiprocessing.Pool() as pool:
        pool.starmap(worker, [(pdf_file, target_folder) for pdf_file in pdf_files])