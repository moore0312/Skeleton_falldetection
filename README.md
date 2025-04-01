# Human-Falling-Detect-Tracks

本專案旨在偵測影片中是否發生人體跌倒事件。透過結合 AlphaPose 骨架估測與 ST-GCN（Temporal Spatial Graph Convolutional Network）進行姿態分類，實現動作辨識與跌倒判斷。

## 🎯 專案目標

- 使用 AlphaPose 擷取人體全身骨架。
- 將骨架關鍵點資料輸入 ST-GCN（TSSTG）模型，判斷是否為跌倒。
- 最終以「每支影片是否為跌倒」作為分類任務，並計算整體準確率（accuracy）。

---

## 🧩 系統架構

1. **偵測與姿態估測**：使用 YOLOv3-Tiny 和 AlphaPose 取得人體骨架。
2. **追蹤**：整合 DeepSort 或其他 tracker 確保骨架追蹤一致性。
3. **動作辨識**：將骨架序列輸入 ST-GCN 模型辨識動作。
4. **跌倒判斷**：若預測結果為 fall，則視為跌倒事件。

---

## 📂 資料夾結構（精簡）

```
Human-Falling-Detect-Tracks/
├── main.py                     # 主推論腳本
├── evaluation.py               # 評估準確率用
├── Models/                     # 儲存模型權重（不含在 GitHub）
├── Data/                       # 自製資料與標註檔
├── Actionsrecognition/         # ST-GCN 相關模型與訓練程式
├── Visualizer.py               # 顯示分類信心值
├── output/                     # 推論與評估結果（已忽略上傳）
└── README.md                   # 專案說明文件（本檔案）
```

---

## 🚀 功能與進度

- ✅ 完成整合推論流程（main.py）
- ✅ 優化 Visualizer：僅顯示動作變換時的信心值前 3 名
- ✅ 支援即時推論（webcam 模式）
- ✅ 重構 `evaluation.py`：計算整體 accuracy 並寫入報表



---

## 💡 使用方式

### 推論整個資料夾影片
```bash
python main.py --video_dir ./your_folder --model_type full
```


