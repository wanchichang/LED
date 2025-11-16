# Unity 集成快速開始指南

## 概述

這個腳本允許你直接使用訓練好的 LED 模型進行推理，為 Unity 籃球系統生成防守者軌跡。

## 文件說明

1. **`inference_unity.py`** - 主要的推理腳本
2. **`UNITY_INTEGRATION.md`** - 詳細的集成文檔
3. **`test_inference.py`** - 測試腳本
4. **`example_input.json`** - 示例輸入文件

## 快速開始

### 1. 測試模型是否正常工作

```bash
# 測試推理功能
python test_inference.py

# 使用示例文件測試
python inference_unity.py --input example_input.json --output example_output.json
```

### 2. 基本使用

#### 從文件讀取輸入

```bash
python inference_unity.py --input input.json --output output.json
```

#### 從標準輸入讀取（適合Unity調用）

```bash
# 在Python中
python inference_unity.py < input.json > output.json

# 在Unity C#中（見下方代碼）
```

#### 交互模式

```bash
python inference_unity.py --interactive
```

### 3. Unity 集成代碼示例

#### C# 代碼（最簡單的方式）

```csharp
using System.Diagnostics;
using System.IO;
using Newtonsoft.Json;

public class LEDPredictor
{
    private string pythonPath = @"python";  // 或完整路徑
    private string scriptPath = @"C:\path\to\LED\inference_unity.py";
    
    public float[][][] Predict(float[][][] pastTrajectories)
    {
        // 準備輸入
        var input = new { trajectories = pastTrajectories };
        string jsonInput = JsonConvert.SerializeObject(input);
        
        // 創建臨時文件
        string tempInput = Path.GetTempFileName();
        string tempOutput = Path.GetTempFileName();
        
        File.WriteAllText(tempInput, jsonInput);
        
        // 調用Python
        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = $"{scriptPath} --input {tempInput} --output {tempOutput}",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true
        };
        
        using (Process p = Process.Start(psi))
        {
            p.WaitForExit();
        }
        
        // 讀取結果
        string jsonOutput = File.ReadAllText(tempOutput);
        var result = JsonConvert.DeserializeObject<PredictionResult>(jsonOutput);
        
        // 清理
        File.Delete(tempInput);
        File.Delete(tempOutput);
        
        return result.mean_trajectories;  // 返回平均預測軌跡
    }
}

[System.Serializable]
public class PredictionResult
{
    public float[][][] mean_trajectories;  // [11][20][2]
    public float[][] initial_positions;    // [11][2]
    public int num_agents;
    public int future_frames;
}
```

#### Unity 使用示例

```csharp
public class DefenderController : MonoBehaviour
{
    public Transform[] defenders;  // 10個防守球員
    public Transform attacker;     // 1個進攻球員
    
    private LEDPredictor predictor;
    private Queue<Vector3>[] history = new Queue<Vector3>[11];
    
    void Start()
    {
        predictor = new LEDPredictor();
        
        // 初始化歷史記錄
        for (int i = 0; i < 11; i++)
        {
            history[i] = new Queue<Vector3>();
        }
    }
    
    void Update()
    {
        // 收集位置歷史
        CollectHistory();
        
        // 每N幀預測一次
        if (Time.frameCount % 10 == 0)  // 每10幀預測一次（約0.17秒）
        {
            PredictAndUpdate();
        }
    }
    
    void CollectHistory()
    {
        // 記錄進攻球員
        history[0].Enqueue(attacker.position);
        if (history[0].Count > 10) history[0].Dequeue();
        
        // 記錄防守球員
        for (int i = 0; i < defenders.Length && i < 10; i++)
        {
            history[i + 1].Enqueue(defenders[i].position);
            if (history[i + 1].Count > 10) history[i + 1].Dequeue();
        }
    }
    
    void PredictAndUpdate()
    {
        // 檢查是否有足夠的歷史數據
        if (history[0].Count < 10) return;
        
        // 準備輸入數據
        float[][][] trajectories = new float[11][][];
        for (int i = 0; i < 11; i++)
        {
            trajectories[i] = new float[10][];
            Vector3[] hist = history[i].ToArray();
            for (int j = 0; j < 10; j++)
            {
                // 轉換Unity座標到模型座標（根據你的場景調整）
                trajectories[i][j] = new float[] { hist[j].x, hist[j].z };
            }
        }
        
        // 預測
        float[][][] predictions = predictor.Predict(trajectories);
        
        // 更新防守者位置（使用預測的第3幀，約0.5秒後）
        for (int i = 0; i < defenders.Length && i < 10; i++)
        {
            float[] nextPos = predictions[i + 1][3];  // Agent i+1, frame 3
            Vector3 targetPos = new Vector3(nextPos[0], defenders[i].position.y, nextPos[1]);
            
            // 平滑移動
            defenders[i].position = Vector3.Lerp(defenders[i].position, targetPos, Time.deltaTime * 2f);
        }
    }
}
```

## 輸入格式

```json
{
  "trajectories": [
    [[x1, y1], [x2, y2], ..., [x10, y10]],  // Agent 0 (進攻球員)
    [[x1, y1], [x2, y2], ..., [x10, y10]],  // Agent 1 (防守球員 1)
    ...
    [[x1, y1], [x2, y2], ..., [x10, y10]]   // Agent 10 (防守球員 10)
  ]
}
```

**注意：**
- 必須有 **11 個 agents**（1進攻 + 10防守）
- 每個 agent 必須有 **10 個歷史位置點**
- 座標單位：根據你的Unity場景調整

## 輸出格式

```json
{
  "mean_trajectories": [
    [[x1, y1], ..., [x20, y20]],  // Agent 0 的平均預測（未來20幀）
    [[x1, y1], ..., [x20, y20]],  // Agent 1 的平均預測
    ...
  ],
  "initial_positions": [[x, y], ...],
  "num_agents": 11,
  "future_frames": 20,
  "samples": [...]  // 20個多樣性樣本（可選）
}
```

**使用建議：**
- **`mean_trajectories`**: 平均預測，最穩定，推薦使用
- **`samples`**: 多樣性樣本，用於展示不確定性或隨機選擇

## 性能優化

### 1. 減少預測頻率
```csharp
if (Time.frameCount % 10 == 0)  // 每10幀預測一次，而不是每幀
{
    PredictAndUpdate();
}
```

### 2. 使用CPU模式（如果GPU不可用）
```bash
python inference_unity.py --cpu --input input.json --output output.json
```

### 3. 只使用平均預測
使用 `mean_trajectories` 而不是遍歷所有 `samples`，可以節省時間。

### 4. 座標轉換優化
在Unity中直接使用預測結果，避免重複轉換。

## 座標系統轉換

模型使用的是籃球場座標系統。如果你的Unity場景使用不同的座標系統，需要調整：

```csharp
// 在Unity中轉換座標
Vector3 ConvertToUnityCoords(float[] modelPos)
{
    // 根據你的場景調整這些參數
    float scale = 1.0f;  // 縮放比例
    float offsetX = 0f;  // X偏移
    float offsetZ = 0f;  // Z偏移
    
    return new Vector3(
        modelPos[0] * scale + offsetX,
        0,  // Y軸（高度）根據需要調整
        modelPos[1] * scale + offsetZ
    );
}
```

## 故障排除

### 問題1: 模型加載失敗
- 檢查模型文件路徑
- 確保已安裝所有Python依賴

### 問題2: 輸入格式錯誤
- 確保有11個agents
- 確保每個agent有10個歷史幀
- 檢查JSON格式是否正確

### 問題3: Unity無法調用Python
- 檢查Python路徑
- 確保Python腳本有執行權限
- 檢查臨時文件權限

### 問題4: 推理速度慢
- 減少預測頻率（每N幀預測一次）
- 使用CPU模式（如果GPU很慢）
- 只使用平均預測，不使用所有樣本

## 下一步

1. 根據你的Unity場景調整座標轉換
2. 調整預測頻率以平衡性能和流暢度
3. 實現平滑插值避免軌跡跳躍
4. 添加碰撞檢測避免球員重疊
5. 優化性能（減少樣本數、批處理等）

## 參考

- 詳細文檔：`UNITY_INTEGRATION.md`
- 完整示例：`example_input.json`
- 測試腳本：`test_inference.py`

