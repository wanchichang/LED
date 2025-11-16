from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import torch
from inference_unity import LEDInference

app = FastAPI(
    title="LED Inference API",
    version="1.1",
    description="使用 LED 模型進行多球員未來軌跡預測，適用於 Unity VR 戰術系統"
)

# ======== 定義輸入模型（含 Swagger 預設範例：11 agents） ========

class TrajectoryInput(BaseModel):
    trajectories: List[List[List[float]]] = Field(
        ...,
        example=[
            # ===== 攻方 5 人 =====
            [[47,25],[47.2,25],[47.4,25],[47.6,25],[47.8,25],[48,25],[48.2,25],[48.4,25],[48.6,25],[48.8,25]],
            [[45,23],[45.2,23],[45.4,23],[45.6,23],[45.8,23],[46,23],[46.2,23],[46.4,23],[46.6,23],[46.8,23]],
            [[49,27],[49.2,27],[49.4,27],[49.6,27],[49.8,27],[50,27],[50.2,27],[50.4,27],[50.6,27],[50.8,27]],
            [[50,22],[50.2,22],[50.4,22],[50.6,22],[50.8,22],[51,22],[51.2,22],[51.4,22],[51.6,22],[51.8,22]],
            [[52,26],[52.2,26],[52.4,26],[52.6,26],[52.8,26],[53,26],[53.2,26],[53.4,26],[53.6,26],[53.8,26]],

            # ===== 防守方 5 人 =====
            [[47,28],[47.1,28.1],[47.2,28.2],[47.3,28.3],[47.4,28.4],[47.5,28.5],[47.6,28.6],[47.7,28.7],[47.8,28.8],[47.9,28.9]],
            [[44,24],[44.1,24.1],[44.2,24.2],[44.3,24.3],[44.4,24.4],[44.5,24.5],[44.6,24.6],[44.7,24.7],[44.8,24.8],[44.9,24.9]],
            [[51,29],[51.1,29.1],[51.2,29.2],[51.3,29.3],[51.4,29.4],[51.5,29.5],[51.6,29.6],[51.7,29.7],[51.8,29.8],[51.9,29.9]],
            [[53,23],[53.1,23.1],[53.2,23.2],[53.3,23.3],[53.4,23.4],[53.5,23.5],[53.6,23.6],[53.7,23.7],[53.8,23.8],[53.9,23.9]],
            [[49,21],[49.1,21.1],[49.2,21.2],[49.3,21.3],[49.4,21.4],[49.5,21.5],[49.6,21.6],[49.7,21.7],[49.8,21.8],[49.9,21.9]],

            # ===== 球（Agent 10） =====
            [[47,25.1],[47.2,25.2],[47.4,25.3],[47.6,25.4],[47.8,25.5],[48,25.6],[48.2,25.7],[48.4,25.8],[48.6,25.9],[48.8,26.0]]
        ]
    )

# ======== 啟動時載入模型 ========

print("🚀 Loading LED model ...")

try:
    led_model = LEDInference(
        model_path='./results/checkpoints/led_new.p',
        core_model_path='./results/checkpoints/base_diffusion_model.p',
        config_path='led_augment',
        device='cuda',
        gpu_id=0
    )
    print("✅ Model loaded on GPU")
except:
    print("⚠️ GPU 加載失敗，改用 CPU ...")
    led_model = LEDInference(
        model_path='./results/checkpoints/led_new.p',
        core_model_path='./results/checkpoints/base_diffusion_model.p',
        config_path='led_augment',
        device='cpu',
        gpu_id=0
    )
    print("✅ Model loaded on CPU")

# ======== API 路徑（你指定的 /led/predict） ========

@app.post("/led/predict", summary="LED 未來軌跡預測 API")
async def predict_trajectory(data: TrajectoryInput):
    """使用 LED 模型進行軌跡預測。"""
    input_traj = np.array(data.trajectories, dtype=np.float32)
    with torch.no_grad():
        predictions = led_model.predict(input_traj)

    return {
        "mean_trajectories": predictions["mean"].tolist(),
        "variance": predictions["variance"].tolist(),
        "trajectories": predictions["trajectories"].tolist()
    }

# ======== 健康檢查 ========
@app.get("/", summary="Health Check")
async def root():
    return {"status": "ok", "usage": "請使用 /docs 測試 API 或 POST /led/predict"}

# ======== 本地啟動 ========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
