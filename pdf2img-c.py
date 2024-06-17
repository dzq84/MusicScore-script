import torch
import os
from pdf2image import convert_from_path
from torchvision import transforms
from PIL import Image
from torchvision import models
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

# 加载模型
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 你的类别数为2，乐谱与封面
model.load_state_dict(torch.load('/datasets/score_data/src/2048-cover.pth'))
model.to(device)
model.eval()  # 设定模型为评估（预测）模式


# 创建用于预处理的转换
transform = transforms.Compose([transforms.Resize((2048, 2048)), 
                                transforms.ToTensor()])

def is_score(image):
    """
    使用模型预测指定图像是否为乐谱
    """
    with torch.no_grad():
        # 将Image对象转换为模型可以接受的形式
        image = transform(image).unsqueeze(0).to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item() == 1

def convert_pdf_to_jpg(pdf_file, source_folder, target_folder):
    # 获取不带扩展名的PDF文件名
    pdf_filename = os.path.splitext(pdf_file)[0]
    # PDF文件完整路径
    pdf_filepath = os.path.join(source_folder, pdf_file)
    
    try:
        # 将PDF转换为图像列表
        images = convert_from_path(pdf_filepath)

        # 保存每页为JPG
        for i, image in enumerate(images, start=1):
            if is_score(image):
                # 构建输出的JPG文件完整路径
                image_file = f"{pdf_filename}_{i}.jpg"
                image.save(os.path.join(target_folder, image_file), 'JPEG')
    except Exception as e:
        print(f"发生错误，无法转换{pdf_filepath}：{e}")
        # 在此处删除有问题的PDF文件
        os.remove(pdf_filepath)
        print(f"已删除有问题的PDF文件：{pdf_filepath}")

def main():
    source_folder = '/datasets/score_data/hd_data/hd_data_pdf'  # PDF文件所在文件夹
    target_folder = '/datasets/score_data/hd_data/hd_data_jpg'  # 图片保存的目标文件夹

    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 获取源文件夹中所有的pdf文件
    pdf_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.pdf')]

    # 创建任务列表（每个任务都是一组参数）
    tasks = [(pdf_file, source_folder, target_folder) for pdf_file in pdf_files]

    # 初始化进程池，设定所需的进程数量
    with Pool(processes=10) as pool:
        # 使用tqdm创建一个进度条
        for _ in tqdm(pool.starmap(convert_pdf_to_jpg, tasks), total=len(tasks)):
            pass

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()