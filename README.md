# AI Smart Trash Can

AI Smart Trash Can 是一個結合 **AI 垃圾分類、Arduino 控制、攝影機影像辨識與重量感測** 的智慧垃圾桶系統。

系統使用 Python 執行 AI 影像分類，Arduino 負責按鈕、重量感測器與馬達控制，兩者透過 Serial 通訊互相傳遞資訊。

---

## 1. Project Structure

```text
ai_trashcan/
├── src/
│   ├── nano_code.py
│   ├── classification.py
│   └── classify.py
│
├── arduino/
│   └── controller.ino
│
├── models/
│   └── .gitkeep
│
├── dataset/
│   └── .gitkeep
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Python 程式

#### `src/nano_code.py`

整個系統的主程式。

主要負責：

- 與 Arduino 建立 Serial 通訊
- 接收 Arduino 的啟動訊號
- 控制攝影機
- 擷取垃圾影像
- 呼叫 `classification.py` 進行垃圾分類
- 將分類結果傳送回 Arduino
- 接收重量資料
- 控制整個垃圾分類流程

這是正式執行智慧垃圾桶時的主要入口。

---

#### `src/classification.py`

負責 **AI 模型推論**。

主要流程：

```text
Camera Image
     ↓
Image Preprocessing
     ↓
Trained Model
     ↓
Classification
     ↓
Garbage Class
```

目前分類類別：

```text
cardboard
glass
metal
paper
plastic
trash
```

此程式會從 `models/` 載入訓練完成的模型。

---

#### `src/classify.py`

負責 **AI 模型訓練**。

主要工作：

1. 讀取 `dataset/`
2. 建立訓練資料與驗證資料
3. 進行影像前處理
4. 使用 ResNet50 進行 Transfer Learning
5. 訓練垃圾分類模型
6. 評估模型
7. 將訓練完成的模型輸出到 `models/`

因此：

```text
classify.py
    ↓
Training Dataset
    ↓
ResNet50
    ↓
Trained Model
    ↓
models/
```

---

### Arduino

#### `arduino/controller.ino`

Arduino 端控制程式。

主要負責：

- 按鈕輸入
- HX711 重量感測器
- 馬達 / Servo 控制
- 垃圾分類後的分流
- 與 Python 透過 Serial 通訊

Arduino 負責硬體控制，Python 負責 AI 與主要流程。

---

## 2. System Flow

完整系統流程：

```text
User
 │
 ▼
Press Button
 │
 ▼
Arduino
 │
 │ buttonPressed
 ▼
Python / nano_code.py
 │
 ▼
Camera Capture
 │
 ▼
classification.py
 │
 ▼
ResNet50 Model
 │
 ▼
Garbage Classification
 │
 ├── cardboard
 ├── glass
 ├── metal
 ├── paper
 ├── plastic
 └── trash
 │
 ▼
Python sends result
 │
 ▼
Arduino
 │
 ▼
Motor / Servo
 │
 ▼
Garbage Sorting
 │
 ▼
Weight Measurement
 │
 ▼
Python receives weight
```

---

# 3. Requirements

建議使用：

- Python 3.10+
- Arduino IDE
- USB Serial connection
- Webcam / USB Camera
- Arduino board
- HX711
- Load Cell
- Servo / Motor

Python 套件可以透過以下指令安裝：

```bash
pip install -r requirements.txt
```

---

# 4. Dataset

本專案的 Dataset **不放在 GitHub repository 中**。

原因：

- Dataset 通常包含大量圖片
- Repository 容量會快速增加
- GitHub 不適合儲存大量訓練圖片
- Dataset 與程式碼應分開管理

將 Dataset 下載後放到：

```text
dataset/
```

---

## 4.1 Dataset Directory Structure

Dataset 必須按照分類建立資料夾。

例如：

```text
dataset/
├── cardboard/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
│
├── glass/
│   ├── image001.jpg
│   └── ...
│
├── metal/
│   ├── image001.jpg
│   └── ...
│
├── paper/
│   ├── image001.jpg
│   └── ...
│
├── plastic/
│   ├── image001.jpg
│   └── ...
│
└── trash/
    ├── image001.jpg
    └── ...
```

也就是：

```text
dataset/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

資料夾名稱必須與程式設定的 class name 一致。

---

## 4.2 Dataset Source

本專案使用的資料集可以從公開的垃圾分類資料集取得。

推薦使用 Kaggle 的 **Garbage Classification** 類型資料集。

下載 Dataset 後，將對應的六個分類資料夾放到：

```text
dataset/
```

最終確認：

```text
dataset/cardboard/
dataset/glass/
dataset/metal/
dataset/paper/
dataset/plastic/
dataset/trash/
```

如果下載的 Dataset 原本還有其他資料夾，請依照 `classify.py` 的資料讀取方式整理成上述結構。

---

# 5. Train the AI Model

第一次使用本專案時，需要先訓練模型。

確認 Dataset 已放置：

```text
dataset/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

然後執行：

```bash
python src/classify.py
```

訓練流程：

```text
Dataset
   ↓
Image Preprocessing
   ↓
Training / Validation Split
   ↓
ResNet50
   ↓
Transfer Learning
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Model
```

訓練完成後，模型會輸出到：

```text
models/
```

例如：

```text
models/
└── model.h5
```

實際模型檔名以 `classify.py` 的設定為準。

---

# 6. Use the Trained Model

完成模型訓練後，確認：

```text
models/
└── <trained-model>
```

然後連接：

- Arduino
- Webcam
- HX711
- Load Cell
- Servo / Motor

確認 Arduino 已經燒錄：

```text
arduino/controller.ino
```

接著執行：

```bash
python src/nano_code.py
```

此時 Python 主程式會：

1. 建立 Arduino Serial connection
2. 等待 Arduino 發送啟動訊號
3. 啟動攝影機
4. 擷取垃圾影像
5. 使用訓練好的模型進行分類
6. 將分類結果傳給 Arduino
7. Arduino 控制垃圾分流
8. 取得垃圾重量
9. 完成一次分類流程

---

# 7. Model and Dataset Are Not Included

本 GitHub repository **不包含 Dataset 與訓練完成的模型**。

Repository 只保留：

```text
dataset/
└── .gitkeep

models/
└── .gitkeep
```

這兩個 `.gitkeep` 的用途只是讓 GitHub 保留資料夾結構。

使用者 clone repository 後，需要自行：

### Step 1 — 取得 Dataset

將 Dataset 放入：

```text
dataset/
```

### Step 2 — 訓練模型

```bash
python src/classify.py
```

### Step 3 — 確認模型

```text
models/
└── <trained-model>
```

### Step 4 — 執行智慧垃圾桶

```bash
python src/nano_code.py
```

---

# 8. Why Model Is Not Included

訓練完成的模型可能具有數百 MB 的大小，因此不直接放入一般 Git repository。

如果需要分享模型，可以使用：

- Google Drive
- Hugging Face
- GitHub Releases
- Git LFS

README 可以另外提供模型下載位置。

---

# 9. Why Dataset Is Not Included

Dataset 通常包含大量圖片，直接放入 GitHub 會使 repository 非常龐大。

因此：

```text
Source Code → GitHub
Dataset     → Dataset hosting
Model       → Model hosting
```

這樣可以讓 GitHub repository 保持乾淨，也方便日後更新程式。

---

# 10. Recommended Workflow

第一次使用：

```bash
# 1. Clone repository
git clone <repository-url>

# 2. Enter project
cd ai_trashcan

# 3. Install dependencies
pip install -r requirements.txt

# 4. Put dataset into dataset/
# dataset/
# ├── cardboard/
# ├── glass/
# ├── metal/
# ├── paper/
# ├── plastic/
# └── trash/

# 5. Train model
python src/classify.py

# 6. Connect Arduino and camera

# 7. Start the system
python src/nano_code.py
```

---

# 11. Important Notes

- `nano_code.py` 是正式系統的主要入口。
- `classification.py` 負責模型推論。
- `classify.py` 負責模型訓練。
- `controller.ino` 負責 Arduino 硬體控制。
- Dataset 不需要放 GitHub。
- 訓練完成的模型不需要放 GitHub。
- 不需要 `yolo.py`。
- 不需要舊版測試程式。
- 不需要將 Python 的測試圖片或暫存檔加入 repository。

---

# 12. Project Architecture

```text
                    ┌─────────────────┐
                    │    Arduino      │
                    │ controller.ino  │
                    └────────┬────────┘
                             │
                         Serial
                             │
                             ▼
                    ┌─────────────────┐
                    │  nano_code.py   │
                    │   Main Program   │
                    └────────┬────────┘
                             │
                     Camera Image
                             │
                             ▼
                    ┌─────────────────┐
                    │classification.py│
                    │    Inference     │
                    └────────┬────────┘
                             │
                      Trained Model
                             │
                             ▼
                    ┌─────────────────┐
                    │     models/     │
                    └─────────────────┘

                    Training Pipeline

                    ┌─────────────────┐
                    │    dataset/     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   classify.py   │
                    │     Training    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     models/     │
                    └─────────────────┘
```