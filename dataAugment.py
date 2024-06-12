import cv2
import os
import albumentations as A
import numpy as np
import glob
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 定義個別的增強轉換
augmentations = {
    'rotate270': A.Rotate(limit=(270, 270), p=1),
    'rotate180': A.Rotate(limit=(180, 180), p=1),
    'rotate90': A.Rotate(limit=(90, 90), p=1),
    'rotate40': A.Rotate(limit=(40, 40), p=1),
    'hue_saturation_value': A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
    'random_brightness_contrast': A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
    'median_blur': A.MedianBlur(blur_limit=9, p=1),
    'motion_blur': A.MotionBlur(blur_limit=9, p=1)
}

# 定義邊界框參數
bbox_params = A.BboxParams(format='pascal_voc', label_fields=['class_labels'])

# 從YOLO標籤文件中加載標籤
def load_yolo_label(file_path):
    with open(file_path, 'r') as file:
        labels = [list(map(float, line.strip().split())) for line in file.readlines()]
    return labels

# 保存標籤到YOLO格式的標籤文件中
def save_yolo_label(file_path, labels):
    with open(file_path, 'w') as file:
        for label in labels:
            file.write(' '.join(map(str, label)) + '\n')

# 圖片增強函數，並保存增強後的圖片和標籤
def augment_image(image_path, label_path, output_image_dir, output_label_dir, transform, suffix):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # 加載標籤並轉換邊界框坐標
    labels = load_yolo_label(label_path)
    bboxes, class_labels = [], []
    for label in labels:
        class_id, x_center, y_center, bbox_width, bbox_height = label
        x_min = (x_center - bbox_width / 2) * width
        y_min = (y_center - bbox_height / 2) * height
        x_max = (x_center + bbox_width / 2) * width
        y_max = (y_center + bbox_height / 2) * height
        bboxes.append([x_min, y_min, x_max, y_max])
        class_labels.append(class_id)

    # 應用增強轉換
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']

    # 轉換增強後的邊界框坐標回YOLO格式
    new_bboxes = [
        [class_id, (x_min + x_max) / 2 / width, (y_min + y_max) / 2 / height, (x_max - x_min) / width, (y_max - y_min) / height]
        for (x_min, y_min, x_max, y_max), class_id in zip(transformed_bboxes, transformed_class_labels)
    ]
    
    # 保存增強後的圖片和標籤
    augmented_image_path = os.path.join(output_image_dir, os.path.splitext(os.path.basename(image_path))[0] + suffix + '.jpg')
    augmented_label_path = os.path.join(output_label_dir, os.path.splitext(os.path.basename(label_path))[0] + suffix + '.txt')
    
    cv2.imwrite(augmented_image_path, transformed_image)
    save_yolo_label(augmented_label_path, new_bboxes)

# 主增強函數，用於處理所有圖片
def Augment(input_image_dir, input_label_dir, output_image_dir, output_label_dir):
    image_paths = glob.glob(os.path.join(input_image_dir, '*.jpg'))
    
    for image_path in tqdm(image_paths, desc="Processing Images", unit="image"):
        label_path = os.path.join(input_label_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
        if os.path.exists(label_path):
            shutil.copy(image_path, os.path.join(output_image_dir, os.path.basename(image_path)))
            shutil.copy(label_path, os.path.join(output_label_dir, os.path.basename(label_path)))
            
            for suffix, augmentation in augmentations.items():
                transform = A.Compose([augmentation], bbox_params=bbox_params)
                augment_image(image_path, label_path, output_image_dir, output_label_dir, transform, '_' + suffix)

# 計算目錄中文件的數量
def count_files_in_directory(directory_path):
    return len([name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

# 分割數據集成訓練、驗證和測試集
def split_dataset(image_dir, label_dir, output_base_dir, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42):
    image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    label_files = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])
    
    assert len(image_files) == len(label_files), "圖片和標籤的數量必須相同"
    
    # 分割數據集
    train_images, test_images, train_labels, test_labels = train_test_split(
        image_files, label_files, test_size=test_size, random_state=random_state
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=val_size_adjusted, random_state=random_state
    )
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, split, 'labels'), exist_ok=True)
    
    def copy_files(files, src_dir, dst_dir):
        for f in tqdm(files, desc=f"Copying to {dst_dir}"):
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    
    # 複製文件到相應的目錄
    copy_files(train_images, image_dir, os.path.join(output_base_dir, 'train', 'images'))
    copy_files(train_labels, label_dir, os.path.join(output_base_dir, 'train', 'labels'))
    copy_files(val_images, image_dir, os.path.join(output_base_dir, 'val', 'images'))
    copy_files(val_labels, label_dir, os.path.join(output_base_dir, 'val', 'labels'))
    copy_files(test_images, image_dir, os.path.join(output_base_dir, 'test', 'images'))
    copy_files(test_labels, label_dir, os.path.join(output_base_dir, 'test', 'labels'))
    
    print("數據集分割和複製完成。")
    print(f"訓練集: {len(train_images)} 張圖片")
    print(f"驗證集: {len(val_images)} 張圖片")
    print(f"測試集: {len(test_images)} 張圖片")

# 路徑設定
input_image_dir = 'Yolo/twelveObject-3_Origin/test/images'
input_label_dir = 'Yolo/twelveObject-3_Origin/test/labels'
output_image_dir = 'Yolo/twelveObject-3_Origin/ima'
output_label_dir = 'Yolo/twelveObject-3_Origin/lab'
output_final_dir = 'dataset_8x'

# 創建目錄
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(output_final_dir, exist_ok=True)

# 增強測試集圖片和標籤
Augment(input_image_dir, input_label_dir, output_image_dir, output_label_dir)

# 增強訓練集圖片和標籤
input_image_dir = 'Yolo/twelveObject-3_Origin/train/images'
input_label_dir = 'Yolo/twelveObject-3_Origin/train/labels'
Augment(input_image_dir, input_label_dir, output_image_dir, output_label_dir)

# 增強驗證集圖片和標籤
input_image_dir = 'Yolo/twelveObject-3_Origin/valid/images'
input_label_dir = 'Yolo/twelveObject-3_Origin/valid/labels'
Augment(input_image_dir, input_label_dir, output_image_dir, output_label_dir)

# 計算增強後的文件數量
image_count = count_files_in_directory(output_image_dir)
label_count = count_files_in_directory(output_label_dir)

print(f"{output_image_dir}中的文件數量: {image_count}")
print(f"{output_label_dir}中的文件數量: {label_count}")

# 分割數據集
split_dataset(output_image_dir, output_label_dir, output_final_dir)
