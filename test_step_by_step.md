# 逐步測試指南

## 目標
逐步確認推理功能正常，然後再進行Unity整合。

## 步驟1: 測試基本推理功能

### 運行測試腳本
```bash
python simple_test_inference.py
```

### 預期輸出
- ✓ 模型加載成功
- ✓ 輸入數據準備完成
- ✓ 推理執行成功
- ✓ 輸出結果格式正確

### 如果出現錯誤
1. **模型文件不存在**
   - 檢查 `./results/checkpoints/led_new.p` 是否存在
   - 檢查 `./results/checkpoints/base_diffusion_model.p` 是否存在

2. **依賴項缺失**
   ```bash
   pip install torch numpy
   ```

3. **GPU/CUDA錯誤**
   - 腳本會自動回退到CPU模式
   - 如果還是失敗，請檢查錯誤信息

## 步驟2: 驗證輸入輸出格式

### 輸入格式要求
- **形狀**: `[11, 10, 2]`
  - 11個agents（1進攻球員 + 10防守球員）
  - 每個agent有10個歷史位置點
  - 每個位置是[x, y]座標

### 輸出格式
- **trajectories**: `[11, 20, 20, 2]` - 20個多樣性樣本，每個預測20幀
- **mean**: `[11, 20, 2]` - 平均預測軌跡（推薦使用）
- **variance**: `[11, 20]` - 方差估計

## 步驟3: 測試自定義輸入

運行測試腳本後，可以測試自定義輸入：

```python
import numpy as np
from inference_unity import LEDInference

# 加載模型
inference = LEDInference(
    model_path='./results/checkpoints/led_new.p',
    core_model_path='./results/checkpoints/base_diffusion_model.p',
    config_path='led_augment'
)

# 準備你的輸入數據（11個agents，每個10個歷史位置）
your_trajectories = np.array([
    # Agent 0: 進攻球員的10個歷史位置
    [[x0, y0], [x1, y1], ..., [x9, y9]],
    # Agent 1: 防守球員1的10個歷史位置
    [[x0, y0], [x1, y1], ..., [x9, y9]],
    # ... 共11個agents
], dtype=np.float32)

# 推理
predictions = inference.predict(your_trajectories)

# 獲取預測結果
mean_trajectories = predictions['mean']  # [11, 20, 2]
# mean_trajectories[0] 是第一個agent的未來20幀軌跡
```

## 步驟4: 確認輸出合理性

檢查輸出是否合理：
1. **數值範圍**: 預測軌跡應該在合理的籃球場座標範圍內
2. **連續性**: 軌跡應該連續，不會有突然的大跳躍
3. **多樣性**: 不同的樣本應該有不同的軌跡（展示不確定性）

## 步驟5: 準備進行Unity整合

**只有當步驟1-4都成功後，才進行Unity整合！**

Unity整合步驟將在下一步提供。

## 常見問題

### Q1: 模型加載很慢
**A**: 第一次加載模型需要時間，後續推理會很快。建議在Unity中保持模型常駐內存。

### Q2: 預測結果不合理
**A**: 
- 檢查輸入數據是否正確
- 確保輸入座標單位正確（可能需要縮放）
- 確保有足夠的歷史數據（10幀）

### Q3: 推理速度慢
**A**:
- 使用GPU可以加速
- 如果GPU不可用，CPU也能工作但會慢一些
- 實際應用中不需要每次都推理（可以每N幀推理一次）

## 下一步

完成這些測試後，告訴我：
1. ✓ 基本推理是否成功
2. ✓ 輸入輸出格式是否正確
3. ✓ 是否有任何錯誤或問題

然後我們再進行Unity整合！

