"""
LED Model Inference for Unity Integration
實時推理腳本，用於 Unity 籃球系統生成防守者軌跡

Usage:
    python inference_unity.py --input input.json --output output.json
    python inference_unity.py --interactive  # 交互模式
    python inference_unity.py  # 從標準輸入讀取，輸出到標準輸出（適合Unity調用）
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel
from utils.config import Config


NUM_Tau = 5  # 擴散步數


class LEDInference:
    """LED 模型推理類，用於實時預測"""
    
    def __init__(self, 
                 model_path: str = './results/checkpoints/led_new.p',
                 core_model_path: str = './results/checkpoints/base_diffusion_model.p',
                 config_path: str = 'led_augment',
                 device: str = 'cuda',
                 gpu_id: int = 0):
        """
        初始化推理模型
        
        Args:
            model_path: 初始化模型路徑
            core_model_path: 核心去噪模型路徑
            config_path: 配置文件名稱
            device: 'cuda' 或 'cpu'
            gpu_id: GPU ID
        """
        self.device = torch.device(f'{device}:{gpu_id}' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加載配置
        self.cfg = Config(config_path, '')
        
        # 數據歸一化參數
        self.traj_mean = torch.FloatTensor(self.cfg.traj_mean).to(self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.traj_scale = self.cfg.traj_scale
        
        # 加載核心去噪模型
        self.core_model = CoreDenoisingModel().to(self.device)
        core_model_cp = torch.load(core_model_path, map_location=self.device)
        self.core_model.load_state_dict(core_model_cp['model_dict'])
        self.core_model.eval()
        
        # 加載初始化模型
        self.model_initializer = InitializationModel(
            t_h=self.cfg.past_frames, 
            d_h=6, 
            t_f=self.cfg.future_frames, 
            d_f=2, 
            k_pred=20
        ).to(self.device)
        
        model_dict = torch.load(model_path, map_location=self.device)
        self.model_initializer.load_state_dict(model_dict['model_initializer_dict'])
        self.model_initializer.eval()
        
        # 準備擴散參數
        self._setup_diffusion_params()
        
        print("Model loaded successfully!")
    
    def _setup_diffusion_params(self):
        """設置擴散過程的參數"""
        self.n_steps = self.cfg.diffusion.steps
        
        # 計算 beta schedule
        if self.cfg.diffusion.beta_schedule == 'linear':
            betas = torch.linspace(
                self.cfg.diffusion.beta_start, 
                self.cfg.diffusion.beta_end, 
                self.n_steps
            ).to(self.device)
        
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
    
    def extract(self, input_tensor, t, x):
        """從擴散參數中提取特定時間步的值"""
        shape = x.shape
        out = torch.gather(input_tensor, 0, t.to(input_tensor.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)
    
    def p_sample_accelerate(self, x, mask, cur_y, t):
        """加速採樣步驟"""
        if t == 0:
            z = torch.zeros_like(cur_y).to(self.device)
        else:
            z = torch.randn_like(cur_y).to(self.device)
        
        t_tensor = torch.tensor([t]).to(self.device)
        
        # 計算 eps_factor
        eps_factor = ((1 - self.extract(self.alphas, t_tensor, cur_y)) / 
                     self.extract(self.one_minus_alphas_bar_sqrt, t_tensor, cur_y))
        
        # 模型輸出
        beta = self.extract(self.betas, t_tensor.repeat(x.shape[0]), cur_y)
        eps_theta = self.core_model.generate_accelerate(cur_y, beta, x, mask)
        
        # 計算均值
        mean = (1 / self.extract(self.alphas, t_tensor, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        
        # 生成噪聲
        z = torch.randn_like(cur_y).to(self.device)
        sigma_t = self.extract(self.betas, t_tensor, cur_y).sqrt()
        sample = mean + sigma_t * z * 0.00001
        
        return sample
    
    def p_sample_loop_accelerate(self, x, mask, loc):
        """
        加速的採樣循環
        
        Args:
            x: [num_agents, past_frames, 6] - 過去軌跡特徵
            mask: [num_agents, num_agents] - 軌跡遮罩
            loc: [num_agents, num_samples, future_frames, 2] - 初始化位置
            
        Returns:
            prediction_total: [num_agents, num_samples, future_frames, 2] - 預測軌跡
        """
        cur_y = loc[:, :10]  # 前10個樣本
        for i in reversed(range(NUM_Tau)):
            cur_y = self.p_sample_accelerate(x, mask, cur_y, i)
        
        cur_y_ = loc[:, 10:]  # 後10個樣本
        for i in reversed(range(NUM_Tau)):
            cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i)
        
        prediction_total = torch.cat((cur_y_, cur_y), dim=1)
        return prediction_total
    
    def preprocess_trajectories(self, past_traj_3d: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        預處理輸入軌跡
        
        Args:
            past_traj_3d: [num_agents, past_frames (10), 2] - 過去軌跡
                         支持的agents數量：10（5對5）或11（模型訓練格式）
            
        Returns:
            past_traj: [11, past_frames, 6] - 處理後的過去軌跡特徵（固定11個agents）
            traj_mask: [11, 11] - 軌跡遮罩
            initial_pos: [11, 1, 2] - 初始位置
        """
        # 轉換為tensor
        past_traj_3d = torch.FloatTensor(past_traj_3d).to(self.device)
        
        # 確保維度正確
        if past_traj_3d.dim() == 2:
            # 如果是 [10, 2] 或 [11, 2]，需要添加時間維度
            past_traj_3d = past_traj_3d.unsqueeze(1)  # [agents, 1, 2]
            # 擴展到10幀（用最後位置填充）
            if past_traj_3d.shape[1] == 1:
                last_pos = past_traj_3d[:, -1:, :]
                past_traj_3d = last_pos.repeat(1, 10, 1)
        
        num_agents = past_traj_3d.shape[0]
        
        # 處理agents數量：模型訓練時使用11個agents
        # 如果是10個agents（5對5），自動填充第11個agent
        if num_agents == 10:
            # 使用進攻球員（第一個agent）的軌跡作為第11個agent（可能是持球者焦點）
            # 或者使用所有球員的平均位置
            if past_traj_3d.shape[1] >= 10:
                # 使用第一個agent的軌跡
                agent_11 = past_traj_3d[0:1, :10, :].clone()  # [1, 10, 2]
                past_traj_3d = torch.cat([past_traj_3d[:, :10, :], agent_11], dim=0)
                print(f"提示: 檢測到10個agents，已自動填充到11個agents（使用進攻球員軌跡）")
            else:
                raise ValueError(f"每個agent至少需要10個歷史幀，但只有{past_traj_3d.shape[1]}幀")
        elif num_agents == 11:
            # 正好11個agents，直接使用
            past_traj_3d = past_traj_3d[:, :10, :]
        elif num_agents < 10:
            # 少於10個agents，用最後一個agent的位置填充
            last_pos = past_traj_3d[:, -1:, :]  # [num_agents, 1, 2]
            num_missing = 11 - num_agents
            padding = last_pos[:, -1:, :].repeat(1, past_traj_3d.shape[1], 1)
            padding = padding.repeat(num_missing, 1, 1)
            past_traj_3d = torch.cat([past_traj_3d[:, :10, :], padding], dim=0)
            print(f"警告: 只有{num_agents}個agents，已填充到11個agents")
        else:
            # 多於11個agents，只取前11個
            past_traj_3d = past_traj_3d[:11, :10, :]
            print(f"警告: 有{num_agents}個agents，只使用前11個")
        
        # 確保最終形狀是 [11, 10, 2]
        assert past_traj_3d.shape == (11, 10, 2), f"最終形狀應該是(11, 10, 2)，但得到{past_traj_3d.shape}"
        
        # 計算初始位置（最後一幀）
        initial_pos = past_traj_3d[:, -1:, :]  # [11, 1, 2]
        
        # 創建軌跡遮罩（11x11，對角線為1）
        traj_mask = torch.eye(11).to(self.device)
        
        # 數據歸一化和特徵提取
        # absolute position
        past_traj_abs = ((past_traj_3d - self.traj_mean) / self.traj_scale).view(11, 10, 2)
        # relative position
        past_traj_rel = ((past_traj_3d - initial_pos) / self.traj_scale).view(11, 10, 2)
        # velocity
        past_traj_vel = torch.cat(
            (past_traj_rel[:, 1:] - past_traj_rel[:, :-1], 
             torch.zeros_like(past_traj_rel[:, -1:])), 
            dim=1
        )
        
        # 拼接特徵: [11, 10, 6]
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
        
        return past_traj, traj_mask, initial_pos
    
    def predict(self, past_traj_3d: np.ndarray, return_samples: bool = True) -> Dict:
        """
        預測未來軌跡
        
        Args:
            past_traj_3d: [num_agents, past_frames, 2] - 過去軌跡
                         - 支持10個agents（5對5）：會自動填充到11個
                         - 支持11個agents（模型訓練格式）：直接使用
                         - 每個agent需要10個歷史位置點
            return_samples: 是否返回多個樣本（用於多樣性）
            
        Returns:
            predictions: 預測結果字典
                - trajectories: [11, num_samples, future_frames, 2] - 預測軌跡（固定11個agents）
                - mean: [11, future_frames, 2] - 平均預測（固定11個agents）
                - variance: [11, future_frames] - 方差估計
                - agent_mapping: dict - 說明如何映射回原始agents數量
        """
        with torch.no_grad():
            # 預處理
            past_traj, traj_mask, initial_pos = self.preprocess_trajectories(past_traj_3d)
            
            # 模型前向傳播
            sample_prediction, mean_estimation, variance_estimation = self.model_initializer(
                past_traj.view(11, 10, 6), 
                traj_mask
            )
            
            # 標準化和初始化位置
            sample_prediction = (torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / 
                               sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None])
            loc = sample_prediction + mean_estimation[:, None]
            
            # 去噪過程
            pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
            
            # 反歸一化（轉換回原始座標系統）
            # pred_traj: [11, 20, 20, 2]
            pred_traj_denorm = pred_traj * self.traj_scale + initial_pos.view(11, 1, 1, 2)
            
            # 計算均值預測
            mean_pred = pred_traj_denorm.mean(dim=1)  # [11, 20, 2]
            
            # 轉換為numpy
            original_num_agents = past_traj_3d.shape[0] if isinstance(past_traj_3d, np.ndarray) else len(past_traj_3d)
            predictions = {
                'trajectories': pred_traj_denorm.cpu().numpy(),  # [11, 20, 20, 2]
                'mean': mean_pred.cpu().numpy(),  # [11, 20, 2]
                'variance': variance_estimation.cpu().numpy(),  # [11, 20]
                'initial_positions': initial_pos.cpu().numpy(),  # [11, 1, 2]
                'agent_mapping': {
                    'num_agents_original': original_num_agents,
                    'num_agents_output': 11,
                    'note': '如果輸入是10個agents，第11個agent是複製自第1個agent（進攻球員）'
                }
            }
            
            return predictions


def load_trajectories_from_json(json_path: str) -> np.ndarray:
    """
    從JSON文件加載軌跡數據
    
    JSON格式:
    {
        "trajectories": [
            [[x1, y1], [x2, y2], ..., [x10, y10]],  # agent 0
            [[x1, y1], [x2, y2], ..., [x10, y10]],  # agent 1
            ...
        ]
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    trajectories = np.array(data['trajectories'], dtype=np.float32)
    return trajectories


def save_predictions_to_json(predictions: Dict, output_path: str, format_for_unity: bool = True):
    """
    保存預測結果到JSON文件
    
    Args:
        predictions: 預測結果字典
        output_path: 輸出文件路徑
        format_for_unity: 是否格式化為Unity易讀格式
    """
    if format_for_unity:
        # Unity友好的格式
        output = {
            'mean_trajectories': predictions['mean'].tolist(),  # [11, 20, 2]
            'num_agents': int(predictions['mean'].shape[0]),
            'future_frames': int(predictions['mean'].shape[1]),
            'samples': []
        }
        
        # 添加每個樣本
        num_samples = predictions['trajectories'].shape[1]
        for sample_idx in range(num_samples):
            output['samples'].append({
                'sample_id': sample_idx,
                'trajectories': predictions['trajectories'][:, sample_idx, :, :].tolist()  # [11, 20, 2]
            })
        
        # 添加初始位置
        output['initial_positions'] = predictions['initial_positions'].reshape(-1, 2).tolist()  # [11, 2]
        
    else:
        output = {
            k: v.tolist() if isinstance(v, np.ndarray) else v 
            for k, v in predictions.items()
        }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='LED Inference for Unity')
    parser.add_argument('--input', type=str, help='Input JSON file path')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--model', type=str, default='./results/checkpoints/led_new.p', 
                       help='Path to model checkpoint')
    parser.add_argument('--core_model', type=str, default='./results/checkpoints/base_diffusion_model.p',
                       help='Path to core denoising model')
    parser.add_argument('--config', type=str, default='led_augment', help='Config name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # 初始化推理模型
    device = 'cpu' if args.cpu else 'cuda'
    inference = LEDInference(
        model_path=args.model,
        core_model_path=args.core_model,
        config_path=args.config,
        device=device,
        gpu_id=args.gpu
    )
    
    # 交互模式
    if args.interactive:
        print("Interactive mode. Enter trajectory data (press Ctrl+C to exit):")
        try:
            while True:
                line = input("\nEnter JSON trajectory data (or 'file:path.json'): ")
                if line.startswith('file:'):
                    json_path = line[5:].strip()
                    past_traj = load_trajectories_from_json(json_path)
                else:
                    data = json.loads(line)
                    past_traj = np.array(data['trajectories'], dtype=np.float32)
                
                print("Predicting...")
                predictions = inference.predict(past_traj)
                
                print(f"Mean trajectory shape: {predictions['mean'].shape}")
                print(f"Sample trajectories shape: {predictions['trajectories'].shape}")
                
                if args.output:
                    save_predictions_to_json(predictions, args.output)
                    print(f"Saved to {args.output}")
                else:
                    # 打印簡要結果
                    print("\nMean predictions (first agent, first 5 frames):")
                    print(predictions['mean'][0, :5, :])
        
        except KeyboardInterrupt:
            print("\nExiting...")
        return
    
    # 從標準輸入讀取（適合Unity調用）
    if not args.input:
        print("Reading from stdin...")
        try:
            stdin_data = sys.stdin.read()
            if stdin_data.strip():
                data = json.loads(stdin_data)
                past_traj = np.array(data['trajectories'], dtype=np.float32)
            else:
                print("No input data provided")
                return
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from stdin: {e}")
            return
    else:
        # 從文件讀取
        past_traj = load_trajectories_from_json(args.input)
    
    # 進行預測
    print("Predicting trajectories...")
    predictions = inference.predict(past_traj)
    
    # 輸出結果
    if args.output:
        save_predictions_to_json(predictions, args.output)
        print(f"Predictions saved to {args.output}")
    else:
        # 輸出到標準輸出（JSON格式，適合Unity讀取）
        output_json = json.dumps({
            'mean_trajectories': predictions['mean'].tolist(),
            'trajectories': predictions['trajectories'].tolist(),
            'variance': predictions['variance'].tolist()
        }, indent=2)
        print(output_json)


if __name__ == '__main__':
    main()

