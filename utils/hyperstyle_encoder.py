#!/usr/bin/env python3
"""
HyperStyle编码器
将FFHQ对齐的人脸图像编码到StyleGAN2潜在空间
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import os
import sys
from typing import Union, Optional

class HyperStyleEncoder:
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto'):
        """
        初始化HyperStyle编码器
        
        Args:
            model_path: HyperStyle模型文件路径
            device: 计算设备 ('cuda', 'cpu', 'auto')
        """
        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"🖥️  HyperStyle使用设备: {self.device}")
        
        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"HyperStyle模型文件不存在: {model_path}")
        
        # w_encoder固定路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        self.w_encoder_path = os.path.join(project_root, 'models', 'encoders', 'faces_w_encoder.pt')
        
        if not os.path.exists(self.w_encoder_path):
            raise FileNotFoundError(f"W编码器文件不存在: {self.w_encoder_path}")
        
        # 设置HyperStyle环境
        self._setup_hyperstyle_env()
        
        # 加载模型
        self.model, self.opts = self.load_model_official(model_path)
        
        # FFHQ人脸预处理（官方标准）
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print("✅ HyperStyle编码器初始化完成")
    
    def _setup_hyperstyle_env(self):
        """设置HyperStyle环境"""
        # 添加hyperstyle路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        hyperstyle_path = os.path.join(script_dir, 'hyperstyle')
        
        if not os.path.exists(hyperstyle_path):
            raise FileNotFoundError(f"HyperStyle目录不存在: {hyperstyle_path}")
        
        if hyperstyle_path not in sys.path:
            sys.path.insert(0, hyperstyle_path)
    
    def load_model_official(self, model_path: str):
        """使用官方方法加载模型"""
        print(f"📥 加载HyperStyle模型: {os.path.basename(model_path)}")
        
        # 导入官方的load_model函数
        from utils.model_utils import load_model
        
        # 使用官方加载方式
        update_opts = {"w_encoder_checkpoint_path": self.w_encoder_path}
        net, opts = load_model(model_path, update_opts=update_opts)
        
        # 移动到设备
        net = net.to(self.device)
        net.eval()
        
        print("✅ HyperStyle模型加载完成")
        return net, opts
    
    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """预处理图像为HyperStyle标准输入"""
        # 转换为PIL图像
        if isinstance(image_input, str):
            pil_image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert('RGB')
        else:
            raise TypeError(f"不支持的输入类型: {type(image_input)}")
        
        # 应用变换
        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def encode_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """编码图像到StyleGAN2潜在空间"""
        try:
            # 预处理图像
            input_tensor = self.preprocess_image(image_input)
            print(f"🔍 输入张量形状: {input_tensor.shape}")
            
            # 使用官方推理
            from utils.inference_utils import run_inversion
            
            with torch.no_grad():
                # 设置推理参数
                self.opts.resize_outputs = False
                if not hasattr(self.opts, 'n_iters_per_batch'):
                    self.opts.n_iters_per_batch = 5
                
                print(f"🔄 开始推理，迭代次数: {self.opts.n_iters_per_batch}")
                
                result_batch, result_latents, _ = run_inversion(
                    input_tensor, 
                    self.model, 
                    self.opts,
                    return_intermediate_results=True
                )
                
                print(f"🔍 推理结果类型: result_batch={type(result_batch)}, result_latents={type(result_latents)}")
                
                # 按照官方代码处理
                if isinstance(result_batch, dict) and isinstance(result_latents, dict):
                    print(f"🔍 result_batch键: {list(result_batch.keys())}")
                    print(f"🔍 result_latents键: {list(result_latents.keys())}")
                    
                    # result_batch[0] 是重建的图像张量（用于可视化）
                    # result_latents[0] 是潜在向量（用于编辑）
                    if 0 in result_latents:
                        latent_codes = result_latents[0]  # 潜在向量
                        print(f"🔍 latent_codes类型: {type(latent_codes)}")
                        
                        if isinstance(latent_codes, list) and len(latent_codes) > 0:
                            # 获取最终的潜在向量（最后一次迭代）
                            final_latent = latent_codes[-1]
                            print(f"✅ 获得潜在向量，形状: {final_latent.shape}")
                            
                            # 确保返回torch张量
                            if isinstance(final_latent, np.ndarray):
                                final_latent = torch.from_numpy(final_latent).to(self.device)
                                print(f"🔄 转换numpy到torch张量: {final_latent.shape}")
                            elif not isinstance(final_latent, torch.Tensor):
                                raise RuntimeError(f"未知的潜在向量类型: {type(final_latent)}")
                            
                            return final_latent
                        else:
                            raise RuntimeError(f"latent_codes格式错误: {type(latent_codes)}")
                    else:
                        raise RuntimeError("result_latents中没有找到潜在向量")
                else:
                    raise RuntimeError(f"结果格式错误: result_batch={type(result_batch)}, result_latents={type(result_latents)}")
            
        except Exception as e:
            print(f"❌ 图像编码失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def encode_batch(self, image_list: list, batch_size: int = 4) -> torch.Tensor:
        """批量编码图像"""
        all_latents = []
        
        for i in range(0, len(image_list), batch_size):
            batch_images = image_list[i:i + batch_size]
            print(f"🔄 编码批次 {i//batch_size + 1}/{(len(image_list)-1)//batch_size + 1}")
            
            for image in batch_images:
                latent = self.encode_image(image)
                all_latents.append(latent)
        
        return torch.cat(all_latents, dim=0)
    
    def save_latent(self, latent: torch.Tensor, save_path: str):
        """保存潜在向量到文件"""
        # 获取目录路径
        save_dir = os.path.dirname(save_path)
        
        # 如果目录不为空，创建目录
        if save_dir and save_dir != '':
            os.makedirs(save_dir, exist_ok=True)
        
        save_data = {
            'latent': latent.cpu(),
            'shape': list(latent.shape),
            'device': str(latent.device),
            'dtype': str(latent.dtype)
        }
        
        torch.save(save_data, save_path)
    
    def load_latent(self, load_path: str) -> torch.Tensor:
        """从文件加载潜在向量"""
        save_data = torch.load(load_path, map_location=self.device, weights_only=False)
        
        if isinstance(save_data, dict) and 'latent' in save_data:
            latent = save_data['latent'].to(self.device)
        else:
            latent = save_data.to(self.device)
        
        return latent
    
    def get_latent_stats(self, latent: torch.Tensor) -> dict:
        """分析潜在向量统计信息"""
        latent_cpu = latent.cpu()
        
        return {
            'shape': list(latent.shape),
            'mean': float(torch.mean(latent_cpu)),
            'std': float(torch.std(latent_cpu)),
            'min': float(torch.min(latent_cpu)),
            'max': float(torch.max(latent_cpu)),
            'norm': float(torch.norm(latent_cpu)),
            'device': str(latent.device),
            'dtype': str(latent.dtype)
        }
    
    def encode_and_save(self, image_path: str, output_path: str) -> bool:
        """编码图像并直接保存"""
        try:
            # 检查输出路径
            if not output_path or output_path.strip() == '':
                raise ValueError("输出路径不能为空")
            
            latent = self.encode_image(image_path)
            self.save_latent(latent, output_path)
            
            return True
        except Exception as e:
            print(f"❌ 编码保存失败: {e}")
            return False
    
    def batch_encode_directory(self, input_dir: str, output_dir: str) -> list:
        """批量编码目录中的图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in extensions:
            image_files.extend([
                f for f in os.listdir(input_dir) 
                if f.lower().endswith(ext.lower())
            ])
        
        success_paths = []
        
        for i, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_latent.pt")
            
            if self.encode_and_save(input_path, output_path):
                success_paths.append(output_path)
        
        return success_paths

def main():
    """测试函数"""
    if len(sys.argv) < 4:
        print("用法:")
        print("  单张编码: python hyperstyle_encoder.py hyperstyle_ffhq.pt input.jpg output.pt")
        print("  批量编码: python hyperstyle_encoder.py hyperstyle_ffhq.pt input_dir/ output_dir/ --batch")
        return
    
    model_path = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    batch_mode = len(sys.argv) > 4 and sys.argv[4] == '--batch'
    
    try:
        encoder = HyperStyleEncoder(model_path)
        
        if batch_mode:
            success_paths = encoder.batch_encode_directory(input_path, output_path)
        else:
            if encoder.encode_and_save(input_path, output_path):
                print("✅ 编码完成!")
            else:
                print("❌ 编码失败!")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
