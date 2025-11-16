"""
簡單測試推理功能
逐步驗證：輸入座標 → 推理 → 輸出預測軌跡
"""

import numpy as np
import torch
from inference_unity import LEDInference

def test_basic_inference():
    """步驟1: 測試基本推理功能"""
    print("="*60)
    print("步驟1: 加載模型")
    print("="*60)
    
    try:
        # 嘗試加載模型
        inference = LEDInference(
            model_path='./results/checkpoints/led_new.p',
            core_model_path='./results/checkpoints/base_diffusion_model.p',
            config_path='led_augment',
            device='cuda',
            gpu_id=0
        )
        print("✓ 模型加載成功（使用GPU）")
    except Exception as e:
        print(f"✗ GPU加載失敗: {e}")
        print("嘗試使用CPU...")
        try:
            inference = LEDInference(
                model_path='./results/checkpoints/led_new.p',
                core_model_path='./results/checkpoints/base_diffusion_model.p',
                config_path='led_augment',
                device='cpu',
                gpu_id=0
            )
            print("✓ 模型加載成功（使用CPU）")
        except Exception as e2:
            print(f"✗ CPU加載也失敗: {e2}")
            print("\n請檢查:")
            print("1. 模型文件是否存在")
            print("2. 依賴項是否已安裝")
            return None
    
    print("\n" + "="*60)
    print("步驟2: 準備測試輸入數據")
    print("="*60)
    print("\n注意: 模型訓練時使用11個agents，但實際籃球場是10個人（5對5）")
    print("推理腳本會自動處理：如果輸入10個agents，會自動填充到11個")
    print("\n測試1: 使用10個agents（5對5籃球）")
    
    # 創建測試數據 - 10個球員（5對5）
    test_trajectories_10 = np.random.randn(10, 10, 2).astype(np.float32)
    
    # 縮放到合理的籃球場座標範圍 (約 0-100, 0-50)
    test_trajectories_10 = test_trajectories_10 * 5 + np.array([47, 25]).reshape(1, 1, 2)
    
    print(f"\n輸入數據形狀: {test_trajectories_10.shape}")
    print(f"  - Agents數量: {test_trajectories_10.shape[0]} (10個球員，5對5)")
    print(f"  - 歷史幀數: {test_trajectories_10.shape[1]} (需要10幀)")
    print(f"  - 座標維度: {test_trajectories_10.shape[2]} (x, y)")
    
    test_trajectories = test_trajectories_10
    print(f"\n輸入數據範圍:")
    print(f"  X: [{test_trajectories[:, :, 0].min():.2f}, {test_trajectories[:, :, 0].max():.2f}]")
    print(f"  Y: [{test_trajectories[:, :, 1].min():.2f}, {test_trajectories[:, :, 1].max():.2f}]")
    
    # 顯示第一個agent的前5個位置
    print(f"\n第一個agent的前5個位置:")
    for i in range(min(5, test_trajectories.shape[1])):
        print(f"  幀 {i}: ({test_trajectories[0, i, 0]:.2f}, {test_trajectories[0, i, 1]:.2f})")
    
    print("\n" + "="*60)
    print("步驟3: 執行推理")
    print("="*60)
    
    try:
        print("正在推理...")
        predictions = inference.predict(test_trajectories)
        print("✓ 推理完成！")
    except Exception as e:
        print(f"✗ 推理失敗: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n" + "="*60)
    print("步驟4: 檢查輸出結果")
    print("="*60)
    
    print(f"預測結果形狀:")
    print(f"  - trajectories: {predictions['trajectories'].shape}")
    print(f"    → [agents={predictions['trajectories'].shape[0]}, samples={predictions['trajectories'].shape[1]}, frames={predictions['trajectories'].shape[2]}, coords=2]")
    print(f"  - mean: {predictions['mean'].shape}")
    print(f"    → [agents={predictions['mean'].shape[0]}, frames={predictions['mean'].shape[1]}, coords=2]")
    print(f"  - variance: {predictions['variance'].shape}")
    print(f"    → [agents={predictions['variance'].shape[0]}, frames={predictions['variance'].shape[1]}]")
    
    print(f"\n輸出數據範圍:")
    print(f"  Mean X: [{predictions['mean'][:, :, 0].min():.2f}, {predictions['mean'][:, :, 0].max():.2f}]")
    print(f"  Mean Y: [{predictions['mean'][:, :, 1].min():.2f}, {predictions['mean'][:, :, 1].max():.2f}]")
    
    # 顯示agent映射信息
    if 'agent_mapping' in predictions:
        mapping = predictions['agent_mapping']
        print(f"\nAgent映射信息:")
        print(f"  - 原始輸入: {mapping['num_agents_original']}個agents")
        print(f"  - 模型輸出: {mapping['num_agents_output']}個agents")
        print(f"  - 說明: {mapping['note']}")
        
        # 如果輸入是10個agents，只顯示前10個的預測
        if mapping['num_agents_original'] == 10:
            print(f"\n  ✓ 建議: 使用前10個agents的預測 (predictions['mean'][:10])")
    
    print(f"\n第一個agent的預測軌跡 (平均):")
    print("  未來20幀的位置:")
    for i in range(min(10, predictions['mean'].shape[1])):
        print(f"    幀 {i}: ({predictions['mean'][0, i, 0]:.2f}, {predictions['mean'][0, i, 1]:.2f})")
    if predictions['mean'].shape[1] > 10:
        print("    ...")
        print(f"    幀 {predictions['mean'].shape[1]-1}: ({predictions['mean'][0, -1, 0]:.2f}, {predictions['mean'][0, -1, 1]:.2f})")
    
    print(f"\n第一個agent的多樣性樣本 (Sample 0 vs Sample 1):")
    sample0 = predictions['trajectories'][0, 0, :5, :]  # 前5幀
    sample1 = predictions['trajectories'][0, 1, :5, :]
    for i in range(5):
        print(f"  幀 {i}: Sample0=({sample0[i,0]:.2f}, {sample0[i,1]:.2f}), Sample1=({sample1[i,0]:.2f}, {sample1[i,1]:.2f})")
    
    print("\n" + "="*60)
    print("✓ 測試成功！基本推理功能正常")
    print("="*60)
    
    return inference, predictions


def test_with_custom_input():
    """步驟5: 使用自定義輸入測試"""
    print("\n" + "="*60)
    print("步驟5: 測試自定義輸入")
    print("="*60)
    
    # 加載模型（如果還沒加載）
    try:
        inference = LEDInference(
            model_path='./results/checkpoints/led_new.p',
            core_model_path='./results/checkpoints/base_diffusion_model.p',
            config_path='led_augment',
            device='cuda',
            gpu_id=0
        )
    except:
        inference = LEDInference(
            model_path='./results/checkpoints/led_new.p',
            core_model_path='./results/checkpoints/base_diffusion_model.p',
            config_path='led_augment',
            device='cpu',
            gpu_id=0
        )
    
    print("請輸入11個球員的歷史軌跡（每個球員10個位置點）")
    print("格式示例: 球員從左向右移動")
    
    # 創建一個示例：球員從左向右移動
    custom_trajectories = np.zeros((11, 10, 2), dtype=np.float32)
    
    # Agent 0 (進攻球員): 從左向右移動
    for i in range(10):
        custom_trajectories[0, i, 0] = 40 + i * 2  # X: 40 -> 58
        custom_trajectories[0, i, 1] = 25  # Y: 保持在25
    
    # Agent 1-10 (防守球員): 跟隨在進攻球員周圍
    for agent in range(1, 11):
        angle = (agent - 1) * 2 * np.pi / 10  # 均勻分布
        for i in range(10):
            custom_trajectories[agent, i, 0] = 45 + i * 2 + 3 * np.cos(angle)
            custom_trajectories[agent, i, 1] = 25 + 3 * np.sin(angle)
    
    print(f"\n輸入軌跡 (Agent 0 進攻球員):")
    for i in range(10):
        print(f"  幀 {i}: ({custom_trajectories[0, i, 0]:.2f}, {custom_trajectories[0, i, 1]:.2f})")
    
    print("\n正在推理...")
    predictions = inference.predict(custom_trajectories)
    
    print(f"\n預測結果 (Agent 0 未來軌跡):")
    for i in range(min(10, predictions['mean'].shape[1])):
        print(f"  幀 {i}: ({predictions['mean'][0, i, 0]:.2f}, {predictions['mean'][0, i, 1]:.2f})")
    
    print("\n" + "="*60)
    print("✓ 自定義輸入測試成功！")
    print("="*60)
    
    return predictions


if __name__ == '__main__':
    print("\n" + "="*60)
    print("LED 推理功能測試")
    print("="*60)
    print("\n這個腳本將逐步測試推理功能")
    print("請確保模型文件存在:")
    print("  - ./results/checkpoints/led_new.p")
    print("  - ./results/checkpoints/base_diffusion_model.p")
    print("\n")
    
    # 步驟1-4: 基本測試
    result = test_basic_inference()
    
    if result is not None:
        inference, predictions = result
        
        # 步驟5: 自定義輸入測試
        print("\n進行自定義輸入測試嗎？(y/n): ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y':
                test_with_custom_input()
        except:
            print("跳過自定義輸入測試")
        
        print("\n" + "="*60)
        print("✓ 所有測試完成！")
        print("="*60)
        print("\n下一步:")
        print("1. 確認推理功能正常後，可以使用 inference_unity.py 進行實際推理")
        print("2. 然後再進行Unity集成")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ 測試失敗，請檢查錯誤信息")
        print("="*60)

