import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from readData import rData  # 自定義模組，用於讀取產品數據
import threading
import time
import numpy as np
import requests
from io import BytesIO
import queue
import webbrowser

# 配置區域
# =====================
MODEL_PATH = "train3/weights/best.pt"  # 模型路徑
FRAME_RATE = 60  # 幀率，控制每秒的畫面更新次數
UI_WIDTH = 500  # 使用者介面的寬度 
UI_HEIGHT = 250  # 使用者介面的高度
SMOOTH_FACTOR = 0.1  # 平滑係數，用於平滑移動UI的位置
# =====================

# 從URL加載圖片
def load_image_from_url(url):
    """
    從指定的URL加載圖片並返回PIL圖像對象。
    如果加載失敗，則返回None。
    """
    response = requests.get(url)
    if response.status_code == 200:
        img_data = BytesIO(response.content)
        try:
            image = Image.open(img_data)
            return image
        except Exception as e:
            print("加載圖片錯誤:", e)
            return None
    else:
        print("錯誤:", response.status_code)
        return None

# 調整文字大小以適應標籤
def resize_text_to_fit(label, text, max_width, fontsize):
    """
    調整標籤的文字大小，以確保文字在指定的最大寬度內顯示。
    """
    font_size = fontsize
    while font_size > 10:
        font = ("Helvetica", font_size, "bold")
        label.config(font=font)
        if label.winfo_reqwidth() <= max_width:
            break
        font_size -= 1

# 調整折扣信息標籤大小以適應內容
def resize_discount_info_label(discount_info_label, text, max_width, max_height, fontsize):
    """
    調整折扣信息標籤的文字大小，以確保文字在指定的最大寬度和高度內顯示。
    """
    font_size = fontsize
    while font_size > 10:
        font = ("Helvetica", font_size, "bold")
        discount_info_label.config(font=font, wraplength=max_width)
        discount_info_label.update_idletasks()
        label_width = discount_info_label.winfo_width()
        label_height = discount_info_label.winfo_height()
        if label_width <= max_width and label_height <= max_height:
            break
        font_size -= 1

# 顯示產品信息的UI界面
def Product_info_UI(content_frame, product_info, store_logo_photo, manufacturer_logo_photo, info_icon_photo):
    """
    創建和配置顯示產品信息的UI界面，包括產品名稱、價格、折扣和其他信息。
    """
    for widget in content_frame.winfo_children():
        widget.destroy()  # 清空現有的所有子組件

    # 產品名稱標籤
    product_name_label = tk.Label(content_frame, text=product_info[1], font=("Helvetica", 42, "bold"), bg='white')
    product_name_label.grid(row=0, column=0, columnspan=3, pady=(10, 5), padx=(5, 5), sticky='we')
    content_frame.update_idletasks()
    resize_text_to_fit(product_name_label, product_info[1], max_width=300, fontsize=42)

    # 商店標誌標籤
    store_logo_label = tk.Label(content_frame, image=store_logo_photo, bg='white')
    store_logo_label.grid(row=0, column=2, padx=(0, 6), sticky='e')

    # 製造商標誌標籤
    manufacturer_logo_label = tk.Label(content_frame, image=manufacturer_logo_photo, bg='white')
    manufacturer_logo_label.grid(row=0, column=0, padx=(6, 0), sticky='w')

    # 配置列的權重和最小尺寸
    content_frame.grid_columnconfigure(0, weight=1, minsize=100)
    content_frame.grid_columnconfigure(1, weight=1, minsize=100)
    content_frame.grid_columnconfigure(2, weight=1, minsize=100)

    # 價格標題標籤
    price_title_label = tk.Label(content_frame, text="Price", font=("Helvetica", 21, "bold"), bg='orange', width=8)
    price_title_label.grid(row=1, column=0, pady=(5, 10), padx=5)

    # 折扣標題標籤
    discount_title_label = tk.Label(content_frame, text="Discount", font=("Helvetica", 21, "bold"), bg='orange', width=8)
    discount_title_label.grid(row=1, column=1, pady=(5, 10), padx=5)
    
    # 信息標題標籤
    info_title_label = tk.Label(content_frame, text="Info", font=("Helvetica", 21, "bold"), bg='orange', width=8)
    info_title_label.grid(row=1, column=2, pady=(5, 10), padx=5)

    # 價格信息標籤
    price_info_label = tk.Label(content_frame, text=product_info[3], font=("Helvetica", 30, "bold"), bg='white')
    price_info_label.grid(row=2, column=0, pady=10)

    # 折扣信息標籤
    discount_info_label = tk.Label(content_frame, text=product_info[4], font=("Helvetica", 24, "bold"), bg='white', wraplength=180, anchor='center', justify='center')
    discount_info_label.grid(row=2, column=1, pady=1)
    max_height = 100
    resize_discount_info_label(discount_info_label, product_info[4], max_width=180, max_height=max_height, fontsize=24)

    # 信息圖標標籤
    info_icon_label = tk.Label(content_frame, image=info_icon_photo, bg='white')
    info_icon_label.grid(row=2, column=2, pady=10)
    info_icon_label.bind("<Button-1>", lambda e: webbrowser.open(product_info[6]))  # 使其可點擊打開網頁

    # 配置框架邊界樣式
    content_frame.configure(highlightbackground="black", highlightcolor="black", highlightthickness=8)

# 加載YOLO模型
model = YOLO(MODEL_PATH)  # 初始化YOLO模型
cap = cv2.VideoCapture(0)  # 開啟攝像頭

if not cap.isOpened():
    print("錯誤: 無法從攝像頭打開視頻流")
    exit()

# 初始化Tkinter主窗口
root = tk.Tk()
root.title("產品信息顯示")
video_label = tk.Label(root)
video_label.pack()
content_frame = tk.Frame(root, bg='white', padx=10, pady=10)
content_frame.pack(expand=True, fill='both')

loaded_images = {}  # 已加載的圖片字典
image_queue = queue.Queue()  # 圖片加載隊列

current_position = np.array([0, 0])  # 當前UI位置
target_position = np.array([0, 0])  # 目標UI位置
previous_product_info = None  # 上一個產品信息

# 異步加載圖片
def load_images_asynchronously(product_info, class_name):
    """
    異步加載產品相關圖片，並將其存入佇列中。
    """
    store_logo_image_url = product_info[2]
    manufacturer_logo_image_url = product_info[0]
    info_icon_image_url = product_info[5]

    if class_name not in loaded_images:
        try:
            store_logo_image = load_image_from_url(store_logo_image_url)
            manufacturer_logo_image = load_image_from_url(manufacturer_logo_image_url)
            info_icon_image = load_image_from_url(info_icon_image_url)

            if store_logo_image is not None and manufacturer_logo_image is not None:
                store_logo_image = store_logo_image.resize((70, 70), Image.LANCZOS)
                manufacturer_logo_image = manufacturer_logo_image.resize((70, 70), Image.LANCZOS)
                info_icon_image = info_icon_image.resize((50, 50), Image.LANCZOS)
            else:
                print("錯誤: 無法加載一個或多個圖片。")

            images = {
                'store_logo': store_logo_image,
                'manufacturer_logo': manufacturer_logo_image,
                'info_icon': info_icon_image
            }
            image_queue.put((class_name, images))
        except Exception as e:
            print(f"錯誤: 無法加載圖片 {e}")

# 找到最接近中心的物體
def find_closest_to_center(detections, frame_center):
    """
    找到距離畫面中心最近的檢測框。
    """
    closest_distance = float('inf')
    closest_bbox = None

    for bbox in detections:
        x, y, w, h = bbox.xyxy[0].cpu().numpy()
        object_center = np.array([(x + w) / 2, (y + h) / 2])
        distance = np.linalg.norm(object_center - frame_center)

        if distance < closest_distance:
            closest_distance = distance
            closest_bbox = bbox

    return closest_bbox

# 繪製產品信息
def draw_product_info(frame, product_info, x, y, images):
    """
    在幀上繪製產品信息，並顯示相關圖片。
    """
    global current_position, target_position, previous_product_info

    if product_info:
        target_position = np.array([x + 300, y + 80])
        target_position[0] = max(0, min(target_position[0], frame.shape[1] - UI_WIDTH))
        target_position[1] = max(0, min(target_position[1], frame.shape[0] - UI_HEIGHT))

        current_position = (1 - SMOOTH_FACTOR) * current_position + SMOOTH_FACTOR * target_position
        ui_x, ui_y = int(current_position[0]), int(current_position[1])

        content_frame.place(x=ui_x, y=ui_y, width=UI_WIDTH, height=UI_HEIGHT)

        if product_info != previous_product_info:
            Product_info_UI(content_frame, product_info, images['store_logo'], images['manufacturer_logo'], images['info_icon'])
            previous_product_info = product_info

    return frame

# 隱藏產品信息
def hide_product_info():
    """
    隱藏顯示產品信息的UI界面。
    通過將content_frame從界面中移除來實現。
    """
    content_frame.place_forget()

# 初始化幀相關的全局變數
last_frame = None  # 保存最後一幀畫面
frame_lock = threading.Lock()  # 幀鎖，用於同步訪問幀數據
frame_ready_event = threading.Event()  # 幀準備事件，用於通知新幀已準備好

# 物體檢測函數
def detect_objects():
    """
    從攝像頭讀取畫面，檢測物體並顯示相應的產品信息。
    包括以下步驟：
    1. 從攝像頭讀取畫面。
    2. 使用YOLO模型檢測畫面中的物體。
    3. 找到距離畫面中心最近的檢測框。
    4. 根據檢測結果顯示對應的產品信息。
    5. 更新畫面並控制幀率。
    """
    global last_frame

    while True:
        start_time = time.time()  # 計時開始
        ret, frame = cap.read()  # 從攝像頭讀取一幀畫面
        if not ret or frame is None or not frame.any():
            continue  # 如果幀為空或無效，跳過這次迴圈

        results = model.predict(frame)  # 使用YOLO模型進行物體檢測
        frame_center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])  # 計算畫面中心點
        detections = []

        # 收集所有檢測到的邊界框
        for result in results:
            for bbox in result.boxes:
                detections.append(bbox)

        if detections:
            closest_bbox = find_closest_to_center(detections, frame_center)  # 找到距離中心最近的邊界框
            if closest_bbox:
                label = int(closest_bbox.cls)  # 獲取邊界框的類別標籤
                class_name = model.names[label]  # 根據標籤獲取類別名稱
                product_info = rData(class_name)  # 獲取對應的產品信息
                
                if product_info:
                    # 異步加載產品圖片
                    load_thread = threading.Thread(target=load_images_asynchronously, args=(product_info, class_name))
                    load_thread.start()
                    
                    if class_name in loaded_images:
                        x, y, w, h = closest_bbox.xyxy[0]  # 獲取邊界框的座標
                        with frame_lock:
                            # 在畫面上繪製產品信息
                            frame = draw_product_info(frame, product_info, int(x), int(y), loaded_images[class_name])
                        # 繪製邊界框
                        cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 2)
        else:
            hide_product_info()  # 如果沒有檢測到物體，隱藏產品信息

        with frame_lock:
            if frame is not None and frame.any():
                last_frame = frame.copy()  # 保存當前幀
                frame_ready_event.set()  # 通知新幀已準備好

        # 控制幀率
        elapsed_time = time.time() - start_time
        time_to_sleep = max(0, (1.0 / FRAME_RATE) - elapsed_time)
        time.sleep(time_to_sleep)  # 根據計算出的時間間隔進行睡眠


# 更新視頻標籤
def update_video_label():
    """
    從攝像頭幀中更新視頻顯示標籤。
    """
    global last_frame

    if frame_ready_event.is_set():
        with frame_lock:
            if last_frame is not None:
                cv2image = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)
                frame_ready_event.clear()

    root.after(15, update_video_label)

# 預加載圖片
def preload_images():
    """
    預加載預期產品的相關圖片。
    """
    for product in expected_products:
        product_info = rData(product)
        load_images_asynchronously(product_info, product)

# 處理圖片佇列
def process_queue():
    """
    從佇列中取出圖片並加載到程序中。
    """
    while not image_queue.empty():
        class_name, images = image_queue.get()
        images['store_logo'] = ImageTk.PhotoImage(images['store_logo'])
        images['manufacturer_logo'] = ImageTk.PhotoImage(images['manufacturer_logo'])
        images['info_icon'] = ImageTk.PhotoImage(images['info_icon'])
        loaded_images[class_name] = images

    root.after(100, process_queue)

print("Loading images, please wait...")

expected_products = ['Coca-Cola', 'Doritos', 'Fanta', 'Good-Luck', 'Lays-original', 'Liquor', 'Mountain-Dew', 'Oreo', 'Pocky', 'Pringles', 'Sprite', 'Water']
preload_thread = threading.Thread(target=preload_images)
preload_thread.start()
preload_thread.join()

threading.Thread(target=detect_objects, daemon=True).start()
root.after(100, process_queue)
root.after(15, update_video_label)
root.mainloop()
