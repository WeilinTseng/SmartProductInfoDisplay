from ultralytics import YOLO

def main():
    # 載入一個YOLO模型
    model = YOLO("yolov8n.pt")  # 載入官方的YOLOv8n模型
    
    # 訓練模型
    results = model.train(data="dataset_8x/data.yaml", epochs=20)
    # 使用指定的資料集進行訓練，訓練過程持續20個epochs
    
    # 驗證模型
    results = model.val(data="dataset_8x/data.yaml")
    # 驗證模型，使用相同的資料集，無需額外參數，資料集和設定已經記錄下來

if __name__ == '__main__':
    main()
