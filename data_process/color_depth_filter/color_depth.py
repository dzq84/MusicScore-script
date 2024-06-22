import os
import shutil
import fitz  # PyMuPDF
import multiprocessing

def is_scanned_black_white(pdf_path):
    is_bw = True
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                images = page.get_images(full=True)
                for img_index, base_xref, xref, smask, width, height, bpc, colorspace, alt, img_name, _, _ in images:
                    if bpc > 1:
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

    os.makedirs(target_folder, exist_ok=True)

    pdf_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.lower().endswith('.pdf')]

    with multiprocessing.Pool() as pool:
        pool.starmap(worker, [(pdf_file, target_folder) for pdf_file in pdf_files])