#!/usr/bin/env python3
"""
StyleGAN2图像生成器
从潜在向量生成高质量人脸图像
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import pickle
import os
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
        self.generator = self.load_generator(model_path)
        
        # W空间平均值（用于截断）
        self.w_avg = None
        if truncation_psi < 1.0:
            self.compute_w_avg()
        
        print("✅ StyleGAN2生成器初始化完成")
    
    def load_generator(self, model_path: str) -> nn.Module:
        """
        加载StyleGAN2生成器
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            generator: StyleGAN2生成器
        """
        try:
            print(f"📥 加载StyleGAN2模型: {os.path.basename(model_path)}")
            
            file_ext = os.path.splitext(model_path)[1].lower()
            
            if file_ext == '.pkl':
                # 官方pickle格式 - 需要特殊处理
                try:
                    # 方法1: 标准pickle加载（适用于官方StyleGAN2-ADA-PyTorch模型）
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # 提取生成器
                    if isinstance(model_data, dict):
                        if 'G_ema' in model_data:
                            generator = model_data['G_ema']
                        elif 'G' in model_data:
                            generator = model_data['G']
                        else:
                            raise ValueError("无法从pickle文件中找到生成器键")
                    else:
                        generator = model_data
                        
                except Exception as pickle_error:
                    print(f"⚠️ 标准pickle加载失败: {pickle_error}")
                    print("🔄 尝试使用legacy方法加载...")
                    
                    # 方法2: 使用legacy加载（适用于旧版本或特殊格式）
                    try:
                        # 这需要安装官方StyleGAN2-ADA-PyTorch的dnnlib和torch_utils
                        import dnnlib
                        import torch_utils
                        
                        # 使用官方的legacy加载方法
                        with open(model_path, 'rb') as f:
                            legacy_data = dnnlib.util.load_pkl(f)
                        
                        if 'G_ema' in legacy_data:
                            generator = legacy_data['G_ema']
                        elif 'G' in legacy_data:
                            generator = legacy_data['G']
                        else:
                            raise ValueError("无法从legacy pickle中找到生成器")
                            
                    except ImportError:
                        print("❌ 缺少dnnlib/torch_utils依赖")
                        print("💡 解决方案:")
                        print("   1. 下载官方StyleGAN2-ADA-PyTorch仓库")
                        print("   2. 将dnnlib和torch_utils目录复制到项目中")
                        print("   3. 或者转换pkl为pt格式使用")
                        raise RuntimeError("无法加载StyleGAN2 pickle文件")
                    
            elif file_ext == '.pt':
                # PyTorch格式
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'generator' in checkpoint:
                        generator = checkpoint['generator']
                    elif 'G_ema' in checkpoint:
                        generator = checkpoint['G_ema']
                    elif 'G' in checkpoint:
                        generator = checkpoint['G']
                    elif 'state_dict' in checkpoint:
                        raise NotImplementedError(
                            "需要StyleGAN2模型架构定义来加载state_dict。"
                            "请提供完整的模型文件而非仅权重文件。"
                        )
                    else:
                        # 假设整个checkpoint就是模型
                        generator = checkpoint
                else:
                    generator = checkpoint
            else:
                raise ValueError(f"不支持的模型文件格式: {file_ext}")
            
            # 移动到设备并设为评估模式
            generator = generator.to(self.device)
            generator.eval()
            
            # 验证生成器
            if not hasattr(generator, '__call__'):
                raise ValueError("加载的对象不是有效的生成器模型")
            
            print("✅ StyleGAN2模型加载完成")
            return generator
            
        except Exception as e:
            print(f"❌ StyleGAN2模型加载失败: {e}")
            print("💡 常见解决方案:")
            print("   1. 确保使用官方StyleGAN2-ADA-PyTorch的模型文件")
            print("   2. 检查是否需要安装dnnlib和torch_utils依赖")
            print("   3. 尝试使用.pt格式的模型文件")
            raise
    
    def compute_w_avg(self, num_samples: int = 10000):
        """
        计算W空间的平均值（用于截断）
        
        Args:
            num_samples: 用于计算平均值的样本数
        """
        try:
            print(f"📊 计算W空间平均值 (样本数: {num_samples})")
            
            w_samples = []
            batch_size = 100
            
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    current_batch = min(batch_size, num_samples - i)
                    
                    # 生成随机Z向量
                    z = torch.randn(current_batch, 512).to(self.device)
                    
                    # 映射到W空间
                    if hasattr(self.generator, 'mapping'):
                        w = self.generator.mapping(z)
                    else:
                        # 尝试直接调用生成器的mapping部分
                        try:
                            w = self.generator.style(z)
                        except:
                            # 如果没有单独的mapping，跳过截断
                            print("⚠️ 无法访问mapping网络，跳过截断计算")
                            return
                    
                    w_samples.append(w.cpu())
            
            # 计算平均值
            all_w = torch.cat(w_samples, dim=0)
            self.w_avg = torch.mean(all_w, dim=0, keepdim=True).to(self.device)
            
            print("✅ W空间平均值计算完成")
            
        except Exception as e:
            print(f"⚠️ W平均值计算失败，跳过截断: {e}")
            self.w_avg = None
    
    def apply_truncation(self, w: torch.Tensor, psi: Optional[float] = None) -> torch.Tensor:
        """
        应用截断技巧
        
        Args:
            w: W空间向量
            psi: 截断参数，None则使用默认值
            
        Returns:
            truncated_w: 截断后的W向量
        """
        if psi is None:
            psi = self.truncation_psi
        
        if psi >= 1.0 or self.w_avg is None:
            return w
        
        # 截断公式: w_truncated = w_avg + psi * (w - w_avg)
        return self.w_avg + psi * (w - self.w_avg)
    
    def generate_from_latent(self, 
                           latent: torch.Tensor,
                           truncation_psi: Optional[float] = None,
                           return_tensor: bool = False) -> Union[Image.Image, torch.Tensor]:
        """
        从潜在向量生成图像
        
        Args:
            latent: 潜在向量
            truncation_psi: 截断参数
            return_tensor: 是否返回张量格式
            
        Returns:
            image: 生成的图像
        """
        try:
            latent = latent.to(self.device)
            
            with torch.no_grad():
                # 应用截断
                if truncation_psi is not None or self.truncation_psi < 1.0:
                    latent = self.apply_truncation(latent, truncation_psi)
                
                # 生成图像
                if hasattr(self.generator, 'synthesis'):
                    # 标准StyleGAN2架构
                    image = self.generator.synthesis(latent)
                else:
                    # 完整生成器调用
                    image = self.generator(latent, None)
                
                if return_tensor:
                    return image
                else:
                    return self.tensor_to_pil(image)
                    
        except Exception as e:
            print(f"❌ 图像生成失败: {e}")
            raise
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        将生成的张量转换为PIL图像
        
        Args:
            tensor: 图像张量 [B, C, H, W] 或 [C, H, W]
            
        Returns:
            image: PIL图像
        """
        try:
            # 确保是单张图像
            if len(tensor.shape) == 4:
                tensor = tensor[0]  # 取第一张图像
            
            # 移动到CPU
            tensor = tensor.cpu().detach()
            
            # StyleGAN2输出范围通常是[-1, 1]，转换为[0, 1]
            tensor = (tensor + 1) / 2
            tensor = torch.clamp(tensor, 0, 1)
            
            # 转换为numpy [C, H, W] -> [H, W, C]
            image_np = tensor.permute(1, 2, 0).numpy()
            
            # 转换为0-255范围
            image_np = (image_np * 255).astype(np.uint8)
            
            # 转换为PIL图像
            return Image.fromarray(image_np)
            
        except Exception as e:
            print(f"❌ 张量转PIL失败: {e}")
            raise
    
    def tensor_to_cv2(self, tensor: torch.Tensor) -> np.ndarray:
        """
        将生成的张量转换为OpenCV图像
        
        Args:
            tensor: 图像张量
            
        Returns:
            image: OpenCV图像 (BGR格式)
        """
        # 先转换为PIL，再转换为OpenCV
        pil_image = self.tensor_to_pil(tensor)
        
        # PIL (RGB) -> OpenCV (BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    
    def generate_and_save(self, 
                         latent_path: str, 
                         output_path: str,
                         truncation_psi: Optional[float] = None) -> bool:
        """
        从潜在向量文件生成并保存图像
        
        Args:
            latent_path: 潜在向量文件路径
            output_path: 输出图像路径
            truncation_psi: 截断参数
            
        Returns:
            success: 是否成功
        """
        try:
            # 加载潜在向量
            latent_data = torch.load(latent_path, map_location=self.device)
            
            # 提取潜在向量
            if isinstance(latent_data, dict) and 'latent' in latent_data:
                latent = latent_data['latent']
            else:
                latent = latent_data
            
            latent = latent.to(self.device)
            
            # 生成图像
            image = self.generate_from_latent(latent, truncation_psi)
            
            # 保存图像
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 根据文件扩展名选择质量
            file_ext = os.path.splitext(output_path)[1].lower()
            if file_ext in ['.jpg', '.jpeg']:
                image.save(output_path, 'JPEG', quality=95, optimize=True)
            else:
                image.save(output_path, 'PNG', optimize=True)
            
            print(f"💾 图像保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 生成保存失败: {e}")
            return False
    
    def batch_generate(self, latent_dir: str, output_dir: str) -> list:
        """
        批量从潜在向量生成图像
        
        Args:
            latent_dir: 潜在向量目录
            output_dir: 输出图像目录
            
        Returns:
            success_paths: 成功生成的图像路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找所有潜在向量文件
        latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.pt')]
        success_paths = []
        
        for i, filename in enumerate(latent_files):
            latent_path = os.path.join(latent_dir, filename)
            name = os.path.splitext(filename)[0]
            
            # 移除可能的"_latent"后缀
            if name.endswith('_latent'):
                name = name[:-7]
            
            output_path = os.path.join(output_dir, f"{name}_generated.jpg")
            
            print(f"🎨 生成图像 {i+1}/{len(latent_files)}: {filename}")
            
            if self.generate_and_save(latent_path, output_path):
                success_paths.append(output_path)
        
        print(f"🎉 批量生成完成! 成功: {len(success_paths)}/{len(latent_files)}")
        return success_paths
    
    def generate_random_image(self, 
                            seed: Optional[int] = None,
                            truncation_psi: Optional[float] = None) -> Image.Image:
        """
        生成随机图像
        
        Args:
            seed: 随机种子
            truncation_psi: 截断参数
            
        Returns:
            image: 生成的随机图像
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 生成随机Z向量
        z = torch.randn(1, 512).to(self.device)
        
        # 映射到W空间
        with torch.no_grad():
            if hasattr(self.generator, 'mapping'):
                w = self.generator.mapping(z)
            else:
                w = z  # 如果没有mapping，直接使用z
        
        # 生成图像
        return self.generate_from_latent(w, truncation_psi)
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            info: 模型信息字典
        """
        info = {
            'device': str(self.device),
            'truncation_psi': self.truncation_psi,
            'has_w_avg': self.w_avg is not None,
            'model_type': type(self.generator).__name__
        }
        
        # 尝试获取更多信息
        try:
            if hasattr(self.generator, 'img_resolution'):
                info['resolution'] = self.generator.img_resolution
            if hasattr(self.generator, 'img_channels'):
                info['channels'] = self.generator.img_channels
            if hasattr(self.generator, 'z_dim'):
                info['z_dim'] = self.generator.z_dim
            if hasattr(self.generator, 'w_dim'):
                info['w_dim'] = self.generator.w_dim
        except:
            pass
        
        return info

def main():
    """测试函数"""
    import sys
    
    if len(sys.argv) < 4:
        print("用法:")
        print("  从潜在向量生成: python stylegan2_generator.py model.pkl latent.pt output.jpg")
        print("  生成随机图像: python stylegan2_generator.py model.pkl --random output.jpg [seed]")
        print("  批量生成: python stylegan2_generator.py model.pkl latent_dir/ output_dir/ --batch")
        return
    
    model_path = sys.argv[1]
    
    try:
        # 创建生成器
        generator = StyleGAN2Generator(model_path)
        
        # 打印模型信息
        info = generator.get_model_info()
        print(f"📊 模型信息: {info}")
        
        if sys.argv[2] == '--random':
            # 随机生成模式
            output_path = sys.argv[3]
            seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
            
            print(f"🎲 生成随机图像 (种子: {seed})")
            image = generator.generate_random_image(seed)
            image.save(output_path, quality=95)
            print(f"✅ 随机图像保存到: {output_path}")
            
        elif len(sys.argv) > 4 and sys.argv[4] == '--batch':
            # 批量生成模式
            latent_dir = sys.argv[2]
            output_dir = sys.argv[3]
            
            print(f"📁 批量生成: {latent_dir} -> {output_dir}")
            success_paths = generator.batch_generate(latent_dir, output_dir)
            print(f"✅ 批量生成完成，共生成 {len(success_paths)} 张图像")
            
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
