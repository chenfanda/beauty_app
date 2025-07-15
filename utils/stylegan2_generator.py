#!/usr/bin/env python3
"""
StyleGAN2图像生成器 - 简化版本
专门处理 HyperStyle 兼容的 rosinality 格式模型
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import pickle
import os
import sys
from typing import Union, Optional

class StyleGAN2Generator:
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto',
                 truncation_psi: float = 0.7):
        """
        初始化StyleGAN2生成器
        
        Args:
            model_path: StyleGAN2模型文件路径 (.pkl 或 .pt)
            device: 计算设备
            truncation_psi: 截断参数，控制生成多样性 (0.0-1.0)
        """
        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"🖥️  StyleGAN2使用设备: {self.device}")
        
        self.truncation_psi = truncation_psi
        
        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"StyleGAN2模型文件不存在: {model_path}")
        
        # 加载生成器
        self.generator, self.w_avg = self.load_generator(model_path)
        
        print("✅ StyleGAN2生成器初始化完成")
    
    def _setup_stylegan2_dependencies(self):
        """设置StyleGAN2-ADA-PyTorch依赖"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 可能的StyleGAN2路径
        possible_paths = [
            os.path.join(current_dir, 'stylegan2-ada-pytorch'),
            os.path.join(os.path.dirname(current_dir), 'stylegan2-ada-pytorch'),
            '/utils/stylegan2-ada-pytorch',
            './stylegan2-ada-pytorch',
        ]
        
        stylegan2_root = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'dnnlib')):
                stylegan2_root = path
                break
        
        if stylegan2_root is None:
            raise ImportError("找不到StyleGAN2-ADA-PyTorch依赖")
        
        if stylegan2_root not in sys.path:
            sys.path.insert(0, stylegan2_root)
            print(f"✅ 已添加StyleGAN2路径: {stylegan2_root}")
        
        # 验证依赖
        try:
            import dnnlib
            import torch_utils
            print("✅ StyleGAN2依赖验证成功")
        except ImportError as e:
            raise ImportError(f"StyleGAN2依赖导入失败: {e}")
    
    def load_generator(self, model_path: str) -> tuple:
        """
        加载StyleGAN2生成器 - 专门处理 rosinality 格式
        """
        print(f"📥 加载StyleGAN2模型: {os.path.basename(model_path)}")
        
        file_ext = os.path.splitext(model_path)[1].lower()
        
        if file_ext == '.pt':
            # 加载 rosinality 格式的 .pt 文件
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if not isinstance(checkpoint, dict):
                raise ValueError("不支持的模型格式")
            
            print(f"🔍 模型文件包含键: {list(checkpoint.keys())}")
            
            # 获取生成器权重
            if 'g_ema' in checkpoint:
                generator_weights = checkpoint['g_ema']
                print("✅ 找到g_ema权重")
            else:
                raise ValueError("找不到生成器权重")
            
            # 获取预计算的W平均值
            w_avg = checkpoint.get('latent_avg', None)
            if w_avg is not None:
                w_avg = w_avg.to(self.device)
                print("🎯 使用预计算的W空间平均值")
            
            # 确定模型尺寸（通过权重键判断）
            if 'convs.14.conv.weight' in generator_weights:
                size = 1024  # 1024x1024
            elif 'convs.12.conv.weight' in generator_weights:
                size = 512   # 512x512
            else:
                size = 1024  # 默认
            
            print(f"🔍 检测到模型尺寸: {size}x{size}")
            
            # 导入并构建 rosinality 模型
            rosinality_path = os.path.join(os.path.dirname(__file__), 'stylegan2-pytorch')
            if rosinality_path not in sys.path:
                sys.path.insert(0, rosinality_path)
            
            try:
                from model import Generator
                
                # 构建生成器（使用正确的尺寸）
                generator = Generator(
                    size,        # 图像尺寸
                    512,         # style_dim
                    8,           # n_mlp
                    channel_multiplier=2
                )
                
                # 加载权重
                generator.load_state_dict(generator_weights)
                generator = generator.to(self.device)
                generator.eval()
                
                print("✅ rosinality StyleGAN2模型加载完成")
                return generator, w_avg
                
            except Exception as e:
                print(f"❌ rosinality模型构建失败: {e}")
                raise RuntimeError("无法构建rosinality StyleGAN2模型")
                
        elif file_ext == '.pkl':
            # StyleGAN2-ADA-PyTorch 格式 - 需要设置依赖
            self._setup_stylegan2_dependencies()
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                if 'G_ema' in model_data:
                    generator = model_data['G_ema']
                elif 'G' in model_data:
                    generator = model_data['G']
                else:
                    raise ValueError("找不到生成器")
            else:
                generator = model_data
            
            generator = generator.to(self.device)
            generator.eval()
            
            print("✅ StyleGAN2-ADA-PyTorch模型加载完成")
            return generator, None
        
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
    
    def apply_truncation(self, w: torch.Tensor, psi: Optional[float] = None) -> torch.Tensor:
        """应用截断技巧"""
        if psi is None:
            psi = self.truncation_psi
        
        if psi >= 1.0 or self.w_avg is None:
            return w
        
        return self.w_avg + psi * (w - self.w_avg)
    
    def generate_from_latent(self, 
                           latent: torch.Tensor,
                           truncation_psi: Optional[float] = None,
                           return_tensor: bool = False) -> Union[Image.Image, torch.Tensor]:
        """
        从潜在向量生成图像
        """
        try:
            latent = latent.to(self.device)
            
            with torch.no_grad():
                # 应用截断
                if truncation_psi is not None or self.truncation_psi < 1.0:
                    latent = self.apply_truncation(latent, truncation_psi)
                
                # 使用对应的生成方式
                if hasattr(self.generator, 'synthesis'):
                    # StyleGAN2-ADA-PyTorch 格式
                    print("🔄 使用 StyleGAN2-ADA-PyTorch synthesis")
                    image = self.generator.synthesis(latent)
                else:
                    # rosinality 格式
                    print("🔄 使用 rosinality 格式生成")
                    image, _ = self.generator([latent],
                                            input_is_latent=True,
                                            randomize_noise=False,
                                            return_latents=False)
                
                if return_tensor:
                    return image
                else:
                    return self.tensor_to_pil(image)
                    
        except Exception as e:
            print(f"❌ 图像生成失败: {e}")
            raise
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """将生成的张量转换为PIL图像"""
        # 确保是单张图像
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        # 移动到CPU
        tensor = tensor.cpu().detach()
        
        # 转换范围 [-1, 1] -> [0, 1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为numpy [C, H, W] -> [H, W, C]
        image_np = tensor.permute(1, 2, 0).numpy()
        
        # 转换为0-255范围
        image_np = (image_np * 255).astype(np.uint8)
        
        return Image.fromarray(image_np)
    
    def generate_and_save(self, 
                         latent_path: str, 
                         output_path: str,
                         truncation_psi: Optional[float] = None) -> bool:
        """从潜在向量文件生成并保存图像"""
        try:
            # 加载潜在向量
            latent_data = torch.load(latent_path, map_location=self.device)
            
            # 提取潜在向量
            if isinstance(latent_data, dict) and 'latent' in latent_data:
                latent = latent_data['latent']
            else:
                latent = latent_data
            
            latent = latent.to(self.device)
            
            # 确保有批次维度
            if len(latent.shape) == 2:
                latent = latent.unsqueeze(0)
            
            # 生成图像
            image = self.generate_from_latent(latent, truncation_psi)
            
            # 保存图像
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            image.save(output_path, quality=95)
            print(f"💾 图像保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 生成保存失败: {e}")
            return False
    
    def generate_random_image(self, 
                            seed: Optional[int] = None,
                            truncation_psi: Optional[float] = None) -> Image.Image:
        """生成随机图像"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 生成随机Z向量并映射到W空间
        z = torch.randn(1, 512).to(self.device)
        
        with torch.no_grad():
            # 简单的Z到W映射（实际应该用mapping网络）
            w = z.unsqueeze(1).repeat(1, 18, 1)  # [1, 18, 512]
        
        return self.generate_from_latent(w, truncation_psi)

def main():
    """测试函数"""
    import sys
    
    if len(sys.argv) < 4:
        print("用法:")
        print("  从潜在向量生成: python stylegan2_generator.py model.pt latent.pt output.jpg")
        print("  生成随机图像: python stylegan2_generator.py model.pt --random output.jpg [seed]")
        return
    
    model_path = sys.argv[1]
    
    try:
        # 创建生成器
        generator = StyleGAN2Generator(model_path)
        
        if sys.argv[2] == '--random':
            # 随机生成模式
            output_path = sys.argv[3]
            seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
            
            print(f"🎲 生成随机图像 (种子: {seed})")
            image = generator.generate_random_image(seed)
            image.save(output_path, quality=95)
            print(f"✅ 随机图像保存到: {output_path}")
            
        else:
            # 单个潜在向量生成
            latent_path = sys.argv[2]
            output_path = sys.argv[3]
            
            print(f"🎨 从潜在向量生成: {latent_path}")
            if generator.generate_and_save(latent_path, output_path):
                print("✅ 生成完成!")
            else:
                print("❌ 生成失败!")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
