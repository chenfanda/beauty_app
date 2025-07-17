#!/usr/bin/env python3
"""
ReStyle编码器 - 简化版本
去除冗余代码，保留核心功能
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

class ReStyleEncoder:
    def __init__(self, 
                 model_path: str,
                 stylegan_path: str = None,
                 id_model_path: str = None,
                 device: str = 'auto',
                 n_iters: int = 5):
        """
        初始化ReStyle编码器
        """
        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"🖥️  ReStyle使用设备: {self.device}")
        
        # 设置默认路径
        if stylegan_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            stylegan_path = os.path.join(project_root, 'models', 'stylegan2', 'stylegan2-ffhq-config-f.pt')
        
        if id_model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            id_model_path = os.path.join(project_root, 'models', 'encoders', 'model_ir_se50.pth')
        
        # 检查文件存在性
        self._check_files(model_path, stylegan_path, id_model_path)
        
        self.n_iters = n_iters
        
        # 设置环境
        self._setup_environment()
        
        # 加载模型
        self.model, self.opts = self._load_restyle_model(model_path)
        self.generator, self.latent_avg = self._load_stylegan2(stylegan_path)
        self.id_loss = self._load_id_loss_model(id_model_path) if self.use_id_loss else None
        
        # 预处理变换
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"✅ ReStyle编码器初始化完成 (迭代次数: {n_iters}, ID loss: {'启用' if self.use_id_loss else '禁用'})")
    
    def _check_files(self, model_path: str, stylegan_path: str, id_model_path: str):
        """检查模型文件存在性"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ReStyle模型文件不存在: {model_path}")
        
        if not os.path.exists(stylegan_path):
            raise FileNotFoundError(f"StyleGAN2模型文件不存在: {stylegan_path}")
        
        self.use_id_loss = os.path.exists(id_model_path)
        if self.use_id_loss:
            self.id_model_path = id_model_path
            print(f"✅ 将使用ID loss模型: {os.path.basename(id_model_path)}")
        else:
            print(f"⚠️  ID loss模型不存在，将跳过: {id_model_path}")
    
    def _setup_environment(self):
        """设置环境路径"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        paths_to_add = [
            os.path.join(script_dir, 'restyle-encoder'),
            os.path.join(script_dir, 'stylegan2-pytorch')
        ]
        
        for path in paths_to_add:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
    
    def _load_restyle_model(self, model_path: str):
        """加载ReStyle模型"""
        print(f"📥 加载ReStyle模型: {os.path.basename(model_path)}")
        
        from models.e4e import e4e
        from argparse import Namespace
        
        # 加载检查点
        ckpt = torch.load(model_path, map_location=self.device)
        
        # 获取配置
        opts = ckpt.get('opts', {})
        if isinstance(opts, dict):
            # 设置必要的配置项
            opts.update({
                'n_iters_per_batch': self.n_iters,
                'resize_outputs': False,
                'start_from_latent_avg': True,
                'checkpoint_path': model_path
            })
            opts = Namespace(**opts)
        else:
            opts.checkpoint_path = model_path
        
        # 创建并加载模型
        net = e4e(opts)
        net.load_state_dict(ckpt['state_dict'])
        net = net.to(self.device)
        net.eval()
        
        print("✅ ReStyle模型加载完成")
        return net, opts
    
    def _load_stylegan2(self, stylegan_path: str):
        """加载StyleGAN2生成器"""
        print(f"📥 加载StyleGAN2生成器: {os.path.basename(stylegan_path)}")
        
        from model import Generator
        
        ckpt = torch.load(stylegan_path, map_location=self.device)
        
        # 获取生成器权重和latent平均值
        generator_weights = ckpt['g_ema']
        latent_avg = ckpt.get('latent_avg', None)
        
        if latent_avg is not None:
            latent_avg = latent_avg.to(self.device)
            print("🎯 找到预计算的latent平均值")
        
        # 创建并加载生成器
        generator = Generator(1024, 512, 8, channel_multiplier=2)
        generator.load_state_dict(generator_weights)
        generator = generator.to(self.device)
        generator.eval()
        
        print("✅ StyleGAN2生成器加载完成")
        return generator, latent_avg
    
    def _load_id_loss_model(self, id_model_path: str):
        """加载ID loss模型"""
        print(f"📥 加载ID loss模型: {os.path.basename(id_model_path)}")
        
        try:
            from models.encoders.model_irse import Backbone
            
            class IDLoss(nn.Module):
                def __init__(self, model_path, device):
                    super(IDLoss, self).__init__()
                    self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
                    state_dict = torch.load(model_path, map_location=device)
                    self.facenet.load_state_dict(state_dict)
                    self.facenet.eval()
                    self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
                    self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
                
                def extract_feats(self, x):
                    if x.shape[2] != 256:
                        x = self.pool(x)
                    x = x[:, :, 35:223, 32:220]  # 裁剪感兴趣区域
                    x = self.face_pool(x)
                    return self.facenet(x)
                
                def forward(self, y_hat, y):
                    n_samples = y.shape[0]
                    y_feats = self.extract_feats(y).detach()
                    y_hat_feats = self.extract_feats(y_hat)
                    
                    loss = 0
                    for i in range(n_samples):
                        cos_sim = torch.dot(y_hat_feats[i], y_feats[i]) / (torch.norm(y_feats[i]) * torch.norm(y_hat_feats[i]))
                        loss += 1 - cos_sim
                    
                    return loss / n_samples
            
            id_loss_model = IDLoss(id_model_path, self.device).to(self.device)
            print("✅ ID loss模型加载完成")
            return id_loss_model
            
        except Exception as e:
            print(f"❌ ID loss模型加载失败: {e}")
            return None
    
    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """预处理图像"""
        if isinstance(image_input, str):
            pil_image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert('RGB')
        else:
            raise TypeError(f"不支持的输入类型: {type(image_input)}")
        
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)
    
    def encode_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """编码图像到StyleGAN2潜在空间 - 使用简化版本"""
        # 预处理图像
        inputs = self.preprocess_image(image_input)
        print(f"🔍 输入张量形状: {inputs.shape}")
        
        with torch.no_grad():
            batch_size = inputs.size(0)
            
            # 初始化latent codes
            if self.latent_avg is not None:
                latent_codes = self.latent_avg.unsqueeze(0).repeat(batch_size, 18, 1)
                y_hat = self.generator([latent_codes], input_is_latent=True, randomize_noise=False)[0]
            else:
                latent_codes = torch.zeros(batch_size, 18, 512).to(self.device)
                y_hat = self.generator([latent_codes], input_is_latent=True, randomize_noise=False)[0]
            
            # 调整图像尺寸到256x256
            if y_hat.size(2) != 256:
                y_hat = torch.nn.functional.interpolate(y_hat, size=(256, 256), mode='bilinear', align_corners=False)
            
            # 迭代优化
            for iter_idx in range(self.n_iters):
                # 准备输入：[原始图像, 当前重建图像]
                inputs_iter = torch.cat([inputs, y_hat], dim=1)
                
                # 通过编码器获取latent偏移
                delta_latents = self.model.encoder(inputs_iter)
                delta_latents = delta_latents.view(batch_size, 18, 512)
                
                # 更新latent codes
                latent_codes = latent_codes + delta_latents
                
                # 生成新图像
                y_hat = self.generator([latent_codes], input_is_latent=True, randomize_noise=False)[0]
                
                # 调整图像尺寸
                if y_hat.size(2) != 256:
                    y_hat = torch.nn.functional.interpolate(y_hat, size=(256, 256), mode='bilinear', align_corners=False)
                
                print(f"🔄 完成迭代 {iter_idx + 1}/{self.n_iters}")
            
            # 计算ID loss（如果可用）
            if self.id_loss is not None:
                try:
                    id_loss_value = self.id_loss(y_hat, inputs)
                    print(f"📊 ID loss: {float(id_loss_value):.4f}")
                except Exception as e:
                    print(f"⚠️  ID loss计算失败: {e}")
            
            # 提取最终的latent codes
            final_latent = latent_codes[0] if batch_size == 1 else latent_codes
            
        print(f"✅ 获得潜在向量，形状: {final_latent.shape}")
        return final_latent
    
    def save_latent(self, latent: torch.Tensor, save_path: str):
        """保存潜在向量"""
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        if len(latent.shape) == 2:
            latent = latent.unsqueeze(0)
            
        save_data = {
            'latent': latent.cpu(),
            'shape': list(latent.shape),
            'encoder': 'restyle',
            'n_iters': self.n_iters
        }
        
        torch.save(save_data, save_path)
        print(f"💾 潜在向量保存到: {save_path}")
    
    def encode_and_save(self, image_path: str, output_path: str) -> bool:
        """编码图像并保存"""
        try:
            latent = self.encode_image(image_path)
            self.save_latent(latent, output_path)
            return True
        except Exception as e:
            print(f"❌ 编码保存失败: {e}")
            return False

def main():
    """测试函数"""
    if len(sys.argv) < 4:
        print("用法: python restyle_encoder.py model.pt input.jpg output.pt [--iters N]")
        return
    
    model_path = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # 解析迭代次数
    n_iters = 5
    if '--iters' in sys.argv:
        iters_idx = sys.argv.index('--iters')
        if iters_idx + 1 < len(sys.argv):
            n_iters = int(sys.argv[iters_idx + 1])
    
    try:
        encoder = ReStyleEncoder(model_path, n_iters=n_iters)
        if encoder.encode_and_save(input_path, output_path):
            print("✅ 编码完成!")
        else:
            print("❌ 编码失败!")
    except Exception as e:
        print(f"❌ 处理失败: {e}")

if __name__ == "__main__":
    main()
