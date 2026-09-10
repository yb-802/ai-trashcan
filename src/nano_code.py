# nano_code.py
'''
python
1. 接收arduino信號001後開啟攝影機（UI顯示："攝影機開啟"）並傳送影像
2. 影像分類（UI顯示："分類中"）並回傳結果002（UI顯示："分類結果：002"）
3. 接收arduino回傳數值c

arduino
1. 綠燈恆亮
2. 按鈕按下後回傳訊號001至python（以開啟攝影機），同時暗綠燈改紅燈恆亮，同時記錄重量感測器當下數值A
3. 接收到python回傳結果002後，旋轉馬達（若結果為'trash'則向右轉120度；反之則向左轉120度），等待3秒後回正
4. 再次記錄重量感測器數值B，並與A做相減（B-A=c）再回傳c

ui（僅程序介面）
1. 顯示各處理程序
（設計兩個介面：程序介面、分類總重量圓餅圖）
'''

import serial
import time
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from tensorflow.keras.models import load_model
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT/"models"/"resnet50_model.h5"
model = load_model(MODEL_PATH)     # 載入模型
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']     # 類別標籤（根據你的模型標籤進行修改）

## 開啟攝影機並儲存影像 ============================================================
def capture_image(save_path):
    cap = cv2.VideoCapture(0)     # 開啟攝影機

    # 設定解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)     
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 捕獲單張影像
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(save_path, frame)
    else:
        print("無法讀取影像")
    cap.release()
    cv2.destroyAllWindows()
    exit()


## 影像分類並回傳結果 =============================================================
def classify_image(image_path):
    img = Image.open(image_path)     # 使用 PIL 打開圖片
    
    # 影像預處理
    img = img.resize((224, 224))  # 調整大小
    img_array = np.array(img) / 255.0  # 正規化
    img_array = np.expand_dims(img_array, axis=0)  # 加入批次維度

    # 預測
    pred = model.predict(img_array)
    pred_label = np.argmax(pred, axis=1)[0]  # 取得預測的索引
    result = labels[pred_label]  # 轉換為對應的標籤

    return result

# print(f"Model prediction for the captured image: {result}")


## Arduino資料傳送(main_code) ==========================================
try:
    ser = serial.Serial("COM3", 9600, timeout=1)
    print("成功連接到 Arduino")
    time.sleep(2)
except serial.SerialException as e:
    print(f"無法連接到 Arduino：{e}")
    ser = None

save_path = ROOT/"picture.jpg"

while True:     # 保持迴圈運行，避免錯過arduino回傳資訊
    if ser is None:
        print("未成功連接到 Arduino，請檢查 COM 埠")
    elif ser.in_waiting > 0:
        command = ser.readline().decode().strip()
        print(f"[收到 Arduino 訊號]：{command}")
        
        # if command == weightA:
            # print("重量感測器開啟")
        if command == "buttonPressed":
            print("攝影機開啟")
            capture_image()
            print("分類中...")
            result = classify_image(save_path)
            print(f"分類結果：{result}")
            ser.write((result + "\n").encode())     # 回傳結果給arduino

            # 結果不是trash時，再接收重量變化量
            if result != "trash":
                start_time = time.time()
                timeout = 10  # 最多等待 10 秒
                while True:
                    if ser.in_waiting:
                        diff = ser.readline().decode().strip()
                        print(f"重量變化量：{diff} g")
                        break
                    elif time.time() - start_time > timeout:
                        print("超時未收到重量資料")
                        break
        if command == "abort":
            print("流程被使用者強制中止")
            continue  # 跳過這輪循環
    else:
        print("未接收到訊號")
        
        


