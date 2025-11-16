# 關於 11 個 Agents 的說明

## 問題
**Q: 為什麼模型需要11個agents？籃球場上不是只有10個人嗎？**

## 答案

### 數據集格式
這個LED模型是在NBA數據集上訓練的，數據集格式是：
- **數據形狀**: `(N, 15, 11, 2)` - N個場景，15幀，**11個agents**，2個座標

### 為什麼是11個？
可能有以下幾種解釋：

1. **1個持球者 + 10個其他球員**
   - 進攻方的持球者（焦點）
   - 其他10個球員（5防守 + 4其他進攻）

2. **5對5 + 1個焦點球員**
   - 標準5對5籃球
   - 第11個可能是持球者（被重點關注的球員）

3. **數據集特定格式**
   - NBA數據集的特定格式定義
   - 可能是為了保持數據一致性

### 模型限制
由於模型是訓練在11個agents上的，**模型架構固定要求11個agents作為輸入**。

## 解決方案

### 方案1: 使用10個agents（推薦）
推理腳本已支持自動處理：
- 如果輸入**10個agents**，會自動填充第11個
- 第11個agent使用第1個agent（進攻球員）的軌跡
- 輸出時可以忽略第11個agent的預測，或將其視為進攻球員的備份預測

**使用方法：**
```python
# 輸入10個agents（5對5）
trajectories_10 = np.array([
    # Agent 0: 進攻球員1
    [[x1, y1], [x2, y2], ..., [x10, y10]],
    # Agent 1: 進攻球員2
    [[x1, y1], ..., [x10, y10]],
    # ... 共10個agents
])

predictions = inference.predict(trajectories_10)
# predictions['mean'] 是 [11, 20, 2]
# 你可以使用前10個：predictions['mean'][:10]  # [10, 20, 2]
```

### 方案2: 使用11個agents
如果你有11個agents的數據（例如包含持球者焦點），可以直接使用：
```python
trajectories_11 = np.array([...])  # [11, 10, 2]
predictions = inference.predict(trajectories_11)
```

## 輸出說明

### 輸出格式
```python
{
    'trajectories': [11, 20, 20, 2],  # 11個agents，20個樣本，20幀
    'mean': [11, 20, 2],              # 11個agents，20幀平均預測
    'variance': [11, 20],             # 11個agents的方差
    'agent_mapping': {                # agents映射說明
        'num_agents_original': 10,    # 原始輸入agents數量
        'num_agents_output': 11,      # 輸出agents數量（固定11）
        'note': '...'
    }
}
```

### 如果輸入是10個agents
- **第0-9個agents**: 對應你的原始10個球員
- **第10個agent**: 複製自第0個agent（進攻球員）
- **建議**: 使用前10個agents的預測：`predictions['mean'][:10]`

### Unity使用建議
```csharp
// 在Unity中，如果你有10個球員
float[][][] predictions = LoadPredictions();  // [11][20][2]

// 只使用前10個agents
for (int i = 0; i < 10; i++)  // 跳過第11個agent
{
    float[] nextPos = predictions[i][frameIndex];
    UpdatePlayerPosition(i, nextPos);
}
```

## 常見問題

### Q1: 我只有10個球員怎麼辦？
**A**: 直接輸入10個agents，推理腳本會自動處理。輸出時使用前10個agents的預測即可。

### Q2: 第11個agent的預測有意義嗎？
**A**: 
- 如果輸入是10個agents，第11個是複製的，可以忽略
- 如果輸入是11個agents，第11個是有效的預測
- 建議：**總是使用前10個agents的預測**

### Q3: 可以修改模型支持10個agents嗎？
**A**: 理論上可以，但需要：
1. 重新訓練模型（耗時）
2. 修改模型架構
3. **不推薦**：使用現有的自動填充功能更簡單

### Q4: 為什麼不直接修改模型？
**A**: 
- 現有模型已經訓練好且有效
- 自動填充不會影響預測質量
- 保持與論文和預訓練模型的一致性

## 測試建議

使用測試腳本時，可以測試兩種情況：

```python
# 測試10個agents
trajectories_10 = np.random.randn(10, 10, 2)  # 10個agents
predictions = inference.predict(trajectories_10)
assert predictions['mean'].shape[0] == 11  # 輸出仍是11個

# 使用前10個
mean_10 = predictions['mean'][:10]  # [10, 20, 2]
```

## 總結

- **模型要求**: 11個agents（訓練格式）
- **實際使用**: 可以輸入10個agents（會自動填充）
- **輸出使用**: 使用前10個agents的預測即可
- **Unity集成**: 忽略第11個agent的預測

