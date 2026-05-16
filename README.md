# 😷 Face Mask Detection Project

---

## 📁 Project Structure

```
face_mask_detection/
│
├── model.py          # CNN architecture definition
├── dataset.py        # Data loading & preprocessing
├── train.py          # Training loop (fit the model)
├── detect.py         # Real-time detection (webcam / image / video)
├── evaluate.py       # Metrics + confusion matrix
├── requirements.txt  # Python dependencies
│
└── data/             # ← You create this
    ├── train/
    │   ├── with_mask/       (images of people WITH masks)
    │   └── without_mask/    (images of people WITHOUT masks)
    └── val/
        ├── with_mask/
        └── without_mask/
```

---

## 🔧 Setup

```bash
pip install -r requirements.txt

not worked? and you have anaconda?? if yes:
  Set-ExecutionPolicy Unrestricted -Scope Process
  conda "shell.powershell" hook | Out-String | Invoke-Expression
```

---

## 📦 Dataset

Download from one of these sources:
- **Kaggle**: search "Face Mask Detection Dataset" (Prajna Bhandary)
  - Direct link: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
- **GitHub**: https://github.com/prajnasb/observations

Place images in `data/train/` and `data/val/` as shown above.

Typical split: **80% train / 20% val**

---

## 🚀 How to Run

### Step 1 — Train the model
```bash
python train.py
```
Trains for 20 epochs. Saves the best model as `mask_cnn.pth`.

### Step 2 — Evaluate
```bash
python evaluate.py
```
Prints accuracy, precision, recall, F1. Saves `confusion_matrix.png`.

### Step 3 — Real-time Detection
```bash
python detect.py                      # webcam
python detect.py --source photo.jpg   # image
python detect.py --source clip.mp4    # video
```

---

## 🧠 How It Works (Explain in Exam)

```
                    ┌──────────────────────────────────────┐
                    │         INPUT: Video Frame            │
                    └─────────────────┬────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────┐
                    │   STEP 1: Face Detection              │
                    │   OpenCV Haar Cascade                  │
                    │   → finds face bounding boxes (x,y,w,h)│
                    └─────────────────┬────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────┐
                    │   STEP 2: Preprocessing               │
                    │   Crop face → Resize 64×64            │
                    │   → Normalize with ImageNet stats     │
                    └─────────────────┬────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────┐
                    │   STEP 3: CNN Classification          │
                    │   3 Conv blocks → FC layers           │
                    │   → Softmax → P(Mask), P(No Mask)     │
                    └─────────────────┬────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────┐
                    │   STEP 4: Draw Result on Frame        │
                    │   ✅ Green box = Wearing Mask          │
                    │   ❌ Red box   = Not Wearing Mask      │
                    └──────────────────────────────────────┘
```

---

## 🏗️ CNN Architecture

| Layer             | Output Shape       | Purpose                           |
|-------------------|--------------------|-----------------------------------|
| Input             | (B, 3, 64, 64)     | RGB image, batch size B           |
| Conv1 + BN + Pool | (B, 32, 31, 31)    | Extract low-level features (edges)|
| Conv2 + BN + Pool | (B, 64, 14, 14)    | Extract mid-level features        |
| Conv3 + BN + Pool | (B, 128, 6, 6)     | Extract high-level features       |
| Flatten           | (B, 4608)          | Convert to vector                 |
| FC + Dropout      | (B, 512)           | Learn decision boundary           |
| Output FC         | (B, 2)             | 2 class scores (logits)           |

**Loss function**: CrossEntropyLoss  
**Optimizer**: Adam (lr=0.001)  
**Regularization**: BatchNorm + Dropout(0.5)  

---

## 📊 Expected Results

| Metric    | Expected Value|
|-----------|---------------|
| Accuracy  | 95–99%        |
| Precision | ~97%          |
| Recall    | ~96%          |
| F1 Score  | ~97%          |


*Project built with PyTorch + OpenCV*
