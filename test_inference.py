"""
快速測試推理腳本

Usage:
    python test_inference.py
"""

import numpy as np
from inference_unity import LEDInference
import json

def test_inference():
    """測試推理功能"""
    print("Loading model...")
    try:
        inference = LEDInference(
            model_path='./results/checkpoints/led_new.p',
            core_model_path='./results/checkpoints/base_diffusion_model.p',
            config_path='led_augment',
            device='cuda',
            gpu_id=0
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying CPU mode...")
        inference = LEDInference(
            model_path='./results/checkpoints/led_new.p',
            core_model_path='./results/checkpoints/base_diffusion_model.p',
            config_path='led_augment',
            device='cpu',
            gpu_id=0
        )
    
    print("\n" + "="*50)
    print("Testing with example data...")
    print("="*50)
    
    # 創建測試數據（11個agents，每個有10個歷史幀）
    test_trajectories = np.random.randn(11, 10, 2).astype(np.float32)
    # 縮放到合理的範圍（模擬籃球場座標）
    test_trajectories = test_trajectories * 5 + np.array([47, 25]).reshape(1, 1, 2)
    
    print(f"Input shape: {test_trajectories.shape}")
    print(f"Input range: [{test_trajectories.min():.2f}, {test_trajectories.max():.2f}]")
    
    try:
        print("\nRunning prediction...")
        predictions = inference.predict(test_trajectories)
        
        print("\n" + "="*50)
        print("Prediction Results:")
        print("="*50)
        print(f"Trajectories shape: {predictions['trajectories'].shape}")
        print(f"Mean shape: {predictions['mean'].shape}")
        print(f"Variance shape: {predictions['variance'].shape}")
        print(f"Initial positions shape: {predictions['initial_positions'].shape}")
        
        print("\nSample predictions (Agent 0, first 5 frames):")
        print("Mean trajectory:")
        print(predictions['mean'][0, :5, :])
        
        print("\nSample 0 trajectory:")
        print(predictions['trajectories'][0, 0, :5, :])
        
        print("\nSample 1 trajectory:")
        print(predictions['trajectories'][0, 1, :5, :])
        
        print("\n" + "="*50)
        print("Test completed successfully!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_inference()
    exit(0 if success else 1)

