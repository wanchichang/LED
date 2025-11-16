# LED 模型 Unity 集成指南

本指南說明如何將 LED 軌跡預測模型集成到 Unity 籃球系統中，實現實時防守者軌跡生成。

## 目錄
1. [快速開始](#快速開始)
2. [輸入輸出格式](#輸入輸出格式)
3. [Unity 調用方式](#unity-調用方式)
4. [性能優化](#性能優化)
5. [示例代碼](#示例代碼)

## 快速開始

### 1. 準備模型文件

確保以下文件存在：
- `results/checkpoints/led_new.p` - 初始化模型
- `results/checkpoints/base_diffusion_model.p` - 核心去噪模型

### 2. 測試推理腳本

```bash
# 使用示例數據測試
python inference_unity.py --input example_input.json --output example_output.json

# 交互模式測試
python inference_unity.py --interactive
```

## 輸入輸出格式

### 輸入格式（JSON）

模型需要過去 10 幀、11 個球員的軌跡數據：

```json
{
  "trajectories": [
    [[x1, y1], [x2, y2], ..., [x10, y10]],  // Agent 0 (進攻球員)
    [[x1, y1], [x2, y2], ..., [x10, y10]],  // Agent 1 (防守球員 1)
    [[x1, y1], [x2, y2], ..., [x10, y10]],  // Agent 2 (防守球員 2)
    ...
    [[x1, y1], [x2, y2], ..., [x10, y10]]   // Agent 10 (防守球員 10)
  ]
}
```

**注意事項：**
- 必須有 11 個 agents（1 個進攻球員 + 10 個防守球員）
- 每個 agent 必須有 10 個歷史位置點
- 座標單位：Unity 世界座標（需要根據實際場景調整縮放）

### 輸出格式（JSON）

模型輸出未來 20 幀的預測軌跡：

```json
{
  "mean_trajectories": [
    [[x1, y1], [x2, y2], ..., [x20, y20]],  // Agent 0 的平均預測
    [[x1, y1], [x2, y2], ..., [x20, y20]],  // Agent 1 的平均預測
    ...
  ],
  "initial_positions": [
    [x, y],  // Agent 0 的初始位置
    [x, y],  // Agent 1 的初始位置
    ...
  ],
  "num_agents": 11,
  "future_frames": 20,
  "samples": [
    {
      "sample_id": 0,
      "trajectories": [
        [[x1, y1], ..., [x20, y20]],  // Agent 0, Sample 0
        [[x1, y1], ..., [x20, y20]],  // Agent 1, Sample 0
        ...
      ]
    },
    ...
    {
      "sample_id": 19,
      "trajectories": [...]
    }
  ]
}
```

**解釋：**
- `mean_trajectories`: 所有樣本的平均軌跡（最穩定，推薦使用）
- `samples`: 20 個多樣性樣本（用於展示不確定性或隨機選擇）
- 每個樣本都是 [11 agents × 20 future frames × 2 (x, y)]

## Unity 調用方式

### 方法 1: 命令行調用（推薦用於批量處理）

Unity C# 代碼示例：

```csharp
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using UnityEngine;
using Newtonsoft.Json;

public class LEDTrajectoryPredictor
{
    private string pythonScriptPath = @"C:\path\to\LED\inference_unity.py";
    private string pythonExecutable = @"C:\Python39\python.exe";  // 或使用 conda env
    
    public class TrajectoryInput
    {
        public float[][] trajectories;  // [11][10][2]
    }
    
    public class TrajectoryOutput
    {
        public float[][][] mean_trajectories;  // [11][20][2]
        public float[][] initial_positions;    // [11][2]
        public int num_agents;
        public int future_frames;
        public Sample[] samples;
        
        public class Sample
        {
            public int sample_id;
            public float[][][] trajectories;  // [11][20][2]
        }
    }
    
    // 方法1: 通過臨時文件調用
    public TrajectoryOutput PredictFromFile(float[][][] pastTrajectories)
    {
        // 準備輸入數據
        var input = new TrajectoryInput
        {
            trajectories = ConvertTo2DArray(pastTrajectories)
        };
        
        // 創建臨時文件
        string inputPath = Path.Combine(Application.temporaryCachePath, "led_input.json");
        string outputPath = Path.Combine(Application.temporaryCachePath, "led_output.json");
        
        // 保存輸入
        string jsonInput = JsonConvert.SerializeObject(input);
        File.WriteAllText(inputPath, jsonInput);
        
        // 調用 Python 腳本
        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = pythonExecutable,
            Arguments = $"\"{pythonScriptPath}\" --input \"{inputPath}\" --output \"{outputPath}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        
        using (Process process = Process.Start(psi))
        {
            process.WaitForExit();
            
            if (process.ExitCode != 0)
            {
                string error = process.StandardError.ReadToEnd();
                UnityEngine.Debug.LogError($"LED Prediction Error: {error}");
                return null;
            }
        }
        
        // 讀取輸出
        if (File.Exists(outputPath))
        {
            string jsonOutput = File.ReadAllText(outputPath);
            return JsonConvert.DeserializeObject<TrajectoryOutput>(jsonOutput);
        }
        
        return null;
    }
    
    // 方法2: 通過標準輸入輸出調用（更快速）
    public TrajectoryOutput PredictFromStdin(float[][][] pastTrajectories)
    {
        var input = new TrajectoryInput
        {
            trajectories = ConvertTo2DArray(pastTrajectories)
        };
        
        string jsonInput = JsonConvert.SerializeObject(input);
        
        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = pythonExecutable,
            Arguments = $"\"{pythonScriptPath}\"",
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        
        using (Process process = Process.Start(psi))
        {
            // 發送輸入
            process.StandardInput.Write(jsonInput);
            process.StandardInput.Close();
            
            // 讀取輸出
            string jsonOutput = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            
            if (process.ExitCode == 0)
            {
                return JsonConvert.DeserializeObject<TrajectoryOutput>(jsonOutput);
            }
            else
            {
                string error = process.StandardError.ReadToEnd();
                UnityEngine.Debug.LogError($"LED Prediction Error: {error}");
                return null;
            }
        }
    }
    
    // 輔助函數：轉換 Unity 座標到模型座標
    private float[][] ConvertTo2DArray(float[][][] trajectories)
    {
        // 實現座標轉換（根據你的Unity座標系統）
        // 例如：縮放、翻轉等
        return trajectories.Select(agent => 
            agent.Select(frame => new float[] { frame[0], frame[1] })
                 .ToArray()
        ).ToArray();
    }
}
```

### 方法 2: HTTP API（適合實時系統）

創建一個簡單的 Flask API 服務器：

```python
# server_unity_api.py
from flask import Flask, request, jsonify
from inference_unity import LEDInference
import numpy as np

app = Flask(__name__)
inference = LEDInference()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        trajectories = np.array(data['trajectories'], dtype=np.float32)
        
        predictions = inference.predict(trajectories)
        
        return jsonify({
            'mean_trajectories': predictions['mean'].tolist(),
            'initial_positions': predictions['initial_positions'].reshape(-1, 2).tolist(),
            'num_agents': int(predictions['mean'].shape[0]),
            'future_frames': int(predictions['mean'].shape[1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True)
```

Unity 調用 HTTP API：

```csharp
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;

public class LEDAPIClient : MonoBehaviour
{
    private string apiUrl = "http://127.0.0.1:5000/predict";
    
    public IEnumerator PredictTrajectories(float[][][] pastTrajectories, System.Action<TrajectoryOutput> callback)
    {
        var input = new { trajectories = pastTrajectories };
        string jsonInput = JsonConvert.SerializeObject(input);
        
        using (UnityWebRequest request = new UnityWebRequest(apiUrl, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonInput);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                var output = JsonConvert.DeserializeObject<TrajectoryOutput>(request.downloadHandler.text);
                callback(output);
            }
            else
            {
                Debug.LogError($"API Error: {request.error}");
            }
        }
    }
}
```

## 性能優化

### 1. 減少樣本數量（加速推理）

修改 `inference_unity.py` 的 `predict` 方法，只生成需要的樣本數：

```python
# 在 predict 方法中，只使用前幾個樣本
loc = loc[:, :5]  # 只使用5個樣本而不是20個
```

### 2. 使用 CPU 推理（如果 GPU 不可用）

```bash
python inference_unity.py --cpu --input input.json --output output.json
```

### 3. 批處理模式

如果 Unity 需要預測多個場景，可以一次性傳入多個批次：

```python
# 修改 inference_unity.py 支持批量輸入
predictions_batch = []
for traj in batch_trajectories:
    pred = inference.predict(traj)
    predictions_batch.append(pred)
```

### 4. 模型量化（進階）

使用 PyTorch 的量化功能減少內存使用和加速：

```python
# 在初始化模型後
model_initializer = torch.quantization.quantize_dynamic(
    model_initializer, {torch.nn.Linear}, dtype=torch.qint8
)
```

## 示例代碼

### Unity 實時應用示例

```csharp
using UnityEngine;
using System.Collections.Generic;

public class BasketballDefenderController : MonoBehaviour
{
    private LEDTrajectoryPredictor predictor;
    private Queue<Vector3>[] agentHistory = new Queue<Vector3>[11];
    private const int HISTORY_LENGTH = 10;
    
    public Transform[] agents;  // 11個球員的Transform
    
    void Start()
    {
        predictor = new LEDTrajectoryPredictor();
        
        // 初始化歷史記錄
        for (int i = 0; i < 11; i++)
        {
            agentHistory[i] = new Queue<Vector3>();
        }
    }
    
    void Update()
    {
        // 收集當前位置
        UpdateHistory();
        
        // 每N幀預測一次
        if (Time.frameCount % 5 == 0)  // 每5幀預測一次
        {
            PredictAndUpdateDefenders();
        }
    }
    
    void UpdateHistory()
    {
        for (int i = 0; i < agents.Length && i < 11; i++)
        {
            Vector3 pos = agents[i].position;
            
            agentHistory[i].Enqueue(pos);
            if (agentHistory[i].Count > HISTORY_LENGTH)
            {
                agentHistory[i].Dequeue();
            }
        }
    }
    
    void PredictAndUpdateDefenders()
    {
        // 檢查是否有足夠的歷史數據
        if (agentHistory[0].Count < HISTORY_LENGTH)
            return;
        
        // 準備輸入數據
        float[][][] trajectories = new float[11][][];
        for (int i = 0; i < 11; i++)
        {
            trajectories[i] = new float[HISTORY_LENGTH][];
            Vector3[] history = agentHistory[i].ToArray();
            for (int j = 0; j < HISTORY_LENGTH; j++)
            {
                // 轉換到模型座標系統（根據需要調整）
                trajectories[i][j] = new float[] { history[j].x, history[j].z };
            }
        }
        
        // 異步預測（避免阻塞主線程）
        StartCoroutine(PredictAsync(trajectories));
    }
    
    System.Collections.IEnumerator PredictAsync(float[][][] trajectories)
    {
        var output = predictor.PredictFromFile(trajectories);
        
        if (output != null)
        {
            // 更新防守者位置（使用平均預測）
            for (int i = 0; i < agents.Length && i < output.mean_trajectories.Length; i++)
            {
                if (i > 0)  // 跳過進攻球員
                {
                    // 獲取下一個預測位置（例如第3幀，約0.5秒後）
                    float[] nextPos = output.mean_trajectories[i][3];
                    Vector3 targetPos = new Vector3(nextPos[0], agents[i].position.y, nextPos[1]);
                    
                    // 平滑移動到目標位置
                    agents[i].position = Vector3.Lerp(agents[i].position, targetPos, Time.deltaTime * 2f);
                }
            }
        }
        
        yield return null;
    }
}
```

## 座標系統轉換

Unity 和模型可能使用不同的座標系統。需要調整：

```python
# 在 inference_unity.py 中添加座標轉換
def unity_to_model_coords(unity_traj):
    """
    Unity座標轉換到模型座標
    Unity: (0,0) 在中心，X向右，Z向前
    模型: 可能需要調整縮放和原點
    """
    model_traj = unity_traj.copy()
    # 例如：縮放、翻轉、平移
    model_traj[:, :, 0] = (unity_traj[:, :, 0] + 47) / 3.36  # X軸
    model_traj[:, :, 1] = (unity_traj[:, :, 2] + 25) / 3.36  # Z軸→Y軸
    return model_traj
```

## 故障排除

### 問題1: 模型加載失敗
- 檢查模型文件路徑是否正確
- 確保已安裝所有依賴項

### 問題2: 輸入格式錯誤
- 確保輸入有11個agents
- 確保每個agent有10個歷史幀
- 檢查座標是否為浮點數

### 問題3: 推理速度慢
- 嘗試使用 `--cpu` 如果GPU太慢
- 減少樣本數量
- 使用HTTP API避免重複加載模型

### 問題4: Unity無法調用Python
- 檢查Python路徑是否正確
- 確保已安裝所有Python依賴
- 檢查Unity的臨時文件權限

## 下一步

1. 根據你的Unity場景調整座標轉換
2. 優化預測頻率（不需要每幀都預測）
3. 添加平滑插值避免軌跡跳躍
4. 實現多樣性樣本選擇（用於展示不確定性）

