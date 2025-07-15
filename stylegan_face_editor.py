#!/usr/bin/env python3
"""
StyleGAN人脸编辑主脚本
集成MediaPipe对齐、HyperStyle编码、方向向量编辑、StyleGAN2生成
"""

import cv2
import numpy as np
import torch
import argparse
import os
import sys
from PIL import Image

# 导入ControlNet脚本用于生成参考图片
try:
    from controlnet_face_reshape import ControlNetFaceReshaper
except ImportError:
    print("⚠️ 无法导入ControlNet模块，参考图片生成功能将不可用")
    ControlNetFaceReshaper = None

# 导入工具模块
from utils.face_aligner import FFHQFaceAligner
from utils.hyperstyle_encoder import HyperStyleEncoder
from utils.stylegan2_generator import StyleGAN2Generator
from utils.direction_manager import DirectionManager

class StyleGANFaceEditor:
    def __init__(self,
                 hyperstyle_path: str = "./models/encoders/hyperstyle_ffhq.pt",
                 stylegan2_path: str = "./models/stylegan2/stylegan2-ffhq-config-f.pkl",
                 dlib_predictor_path: str = "./models/shape_predictor_68_face_landmarks.dat",
                 device: str = 'auto'):
        """
        初始化StyleGAN人脸编辑器
        
        Args:
            hyperstyle_path: HyperStyle模型路径
            stylegan2_path: StyleGAN2模型路径
            dlib_predictor_path: dlib关键点检测器路径
            device: 计算设备
        """
        print("🚀 初始化StyleGAN人脸编辑器...")
        
        self.device = device
        
        # 检查必要文件
        self._check_model_files(hyperstyle_path, stylegan2_path, dlib_predictor_path)
        
        # 初始化组件
        print("📐 初始化人脸对齐器...")
        self.face_aligner = FFHQFaceAligner(dlib_predictor_path)
        
        print("🔄 初始化编码器...")
        self.encoder = HyperStyleEncoder(hyperstyle_path, device)
        
        print("🎨 初始化生成器...")
        self.generator = StyleGAN2Generator(stylegan2_path, device)
        
        print("📂 初始化方向管理器...")
        self.direction_manager = DirectionManager(device=device)
        
        # 可选的ControlNet组件
        self.controlnet_reshaper = None
        if ControlNetFaceReshaper is not None:
            try:
                print("🎛️ 初始化ControlNet组件...")
                self.controlnet_reshaper = ControlNetFaceReshaper()
                print("✅ ControlNet组件可用")
            except Exception as e:
                print(f"⚠️ ControlNet初始化失败: {e}")
        
        print("🎉 StyleGAN人脸编辑器初始化完成!")
    
    def _check_model_files(self, hyperstyle_path: str, stylegan2_path: str, dlib_predictor_path: str):
        """检查必要的模型文件"""
        missing_files = []
        
        if not os.path.exists(hyperstyle_path):
            missing_files.append(f"HyperStyle模型: {hyperstyle_path}")
        
        if not os.path.exists(stylegan2_path):
            missing_files.append(f"StyleGAN2模型: {stylegan2_path}")
        
        if not os.path.exists(dlib_predictor_path):
            missing_files.append(f"dlib预测器: {dlib_predictor_path}")
        
        if missing_files:
            print("❌ 缺少必要的模型文件:")
            for file in missing_files:
                print(f"   - {file}")
            raise FileNotFoundError("请下载缺少的模型文件")
    
    def generate_reference_pairs(self, effect_types: list = None) -> dict:
        """
        使用ControlNet生成参考图片对
        
        Args:
            effect_types: 要生成的效果类型列表
            
        Returns:
            reference_pairs: 参考图片对字典
        """
        if self.controlnet_reshaper is None:
            raise RuntimeError("ControlNet组件不可用，无法生成参考图片")
        
        if effect_types is None:
            effect_types = ['face_slim', 'eye_enlarge', 'nose_lift', 'mouth_adjust']
        
        print(f"🎨 使用ControlNet生成 {len(effect_types)} 组参考图片...")
        
        reference_pairs = {}
        
        # 生成标准基础脸
        print("🎭 生成标准基础人脸...")
        base_image = self._generate_standard_base_face()
        
        for effect_type in effect_types:
            try:
                print(f"   生成 {effect_type} 参考对...")
                ref_a, ref_b = self._generate_single_reference_pair(base_image, effect_type)
                reference_pairs[effect_type] = (ref_a, ref_b)
                
            except Exception as e:
                print(f"❌ 生成 {effect_type} 参考对失败: {e}")
                continue
        
        print(f"✅ 成功生成 {len(reference_pairs)} 组参考图片对")
        return reference_pairs
    
    def _generate_standard_base_face(self) -> np.ndarray:
        """生成标准基础人脸"""
        # 使用ControlNet生成标准人脸
        base_params = {
            'face_slim': 0.0,
            'eye_enlarge': 0.0,
            'nose_lift': 0.0,
            'mouth_adjust': 0.0
        }
        
        # 生成1024x1024的标准人脸
        base_image = np.ones((1024, 1024, 3), dtype=np.uint8) * 128  # 占位符
        
        # 这里应该调用ControlNet生成，但需要根据实际的ControlNet接口调整
        # base_image = self.controlnet_reshaper.generate_standard_face()
        
        return base_image
    
    def _generate_single_reference_pair(self, base_image: np.ndarray, effect_type: str) -> tuple:
        """生成单个效果的参考图片对"""
        # 这里应该根据实际的ControlNet接口实现
        # 目前返回占位符
        ref_a = base_image.copy()  # 原始状态
        ref_b = base_image.copy()  # 调整后状态
        
        return ref_a, ref_b
    
    def setup_directions(self, 
                        reference_pairs: dict = None,
                        use_controlnet: bool = True,
                        force_regenerate: bool = False) -> dict:
        """
        设置方向向量（生成或加载）
        
        Args:
            reference_pairs: 自定义参考图片对
            use_controlnet: 是否使用ControlNet生成参考图片
            force_regenerate: 是否强制重新生成
            
        Returns:
            directions: 方向向量字典
        """
        print("📐 设置方向向量...")
        
        # 检查是否已有缓存的方向向量
        available_directions = self.direction_manager.list_available_directions()
        
        if not force_regenerate and available_directions:
            print(f"📁 发现已缓存的方向向量: {available_directions}")
            directions = self.direction_manager.load_all_directions()
            return directions
        
        # 需要生成新的方向向量
        if reference_pairs is None:
            if use_controlnet:
                reference_pairs = self.generate_reference_pairs()
            else:
                raise ValueError("未提供参考图片对，且ControlNet不可用")
        
        # 计算方向向量
        directions = self.direction_manager.compute_directions_from_pairs(
            reference_pairs, self.encoder
        )
        
        # 保存方向向量
        metadata = {
            'generated_by': 'StyleGAN人脸编辑器',
            'method': 'ControlNet' if use_controlnet else '用户提供',
            'num_directions': len(directions)
        }
        
        self.direction_manager.save_directions_batch(directions, metadata)
        
        return directions
    
    def edit_face(self, 
                  image_input: str,
                  params: dict,
                  output_path: str = None,
                  use_cached_directions: bool = True) -> np.ndarray:
        """
        编辑人脸主函数
        
        Args:
            image_input: 输入图像路径
            params: 编辑参数字典 {effect_name: strength}
            output_path: 输出路径（可选）
            use_cached_directions: 是否使用缓存的方向向量
            
        Returns:
            result_image: 编辑后的图像 (BGR格式)
        """
        try:
            print(f"🎭 开始编辑人脸: {os.path.basename(image_input)}")
            
            # 1. 人脸对齐
            print("📐 FFHQ标准对齐...")
            aligned_image = self.face_aligner.align_face(image_input)
            if aligned_image is None:
                raise ValueError("人脸对齐失败")
            
            # 验证对齐质量
            alignment_quality = self.face_aligner.verify_alignment(aligned_image)
            if alignment_quality['valid'] and not alignment_quality['ffhq_compliant']:
                print("⚠️ 对齐质量可能影响编辑效果")
            
            # 2. 编码到潜在空间
            print("🔄 HyperStyle编码...")
            target_latent = self.encoder.encode_image(aligned_image)
            
            # 显示编码统计
            latent_stats = self.encoder.get_latent_stats(target_latent)
            print(f"📊 潜在向量: 形状{latent_stats['shape']}, 范数{latent_stats['norm']:.3f}")
            
            # 3. 加载方向向量
            if use_cached_directions:
                directions = self.direction_manager.load_all_directions()
                if not directions:
                    print("⚠️ 未找到缓存的方向向量，尝试生成...")
                    directions = self.setup_directions()
            else:
                directions = self.setup_directions(force_regenerate=True)
            
            # 4. 应用编辑
            print("✨ 应用编辑效果...")
            edited_latent = self.direction_manager.apply_multiple_directions(
                target_latent, directions, params
            )
            
            # 5. 生成最终图像
            print("🎨 StyleGAN2生成...")
            result_pil = self.generator.generate_from_latent(edited_latent)
            
            # 转换为OpenCV格式
            result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
            
            # 6. 保存结果（可选）
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, result_bgr)
                print(f"💾 结果保存到: {output_path}")
            
            print("✅ 人脸编辑完成!")
            return result_bgr
            
        except Exception as e:
            print(f"❌ 人脸编辑失败: {e}")
            raise
    
    def batch_edit(self, 
                   input_dir: str,
                   output_dir: str,
                   params: dict) -> list:
        """
        批量编辑人脸
        
        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录
            params: 编辑参数
            
        Returns:
            success_paths: 成功处理的输出路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找图像文件
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend([
                f for f in os.listdir(input_dir) 
                if f.lower().endswith(ext.lower())
            ])
        
        print(f"📁 批量编辑 {len(image_files)} 张图像...")
        
        success_paths = []
        
        for i, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_edited{ext}")
            
            print(f"🎭 处理 {i+1}/{len(image_files)}: {filename}")
            
            try:
                self.edit_face(input_path, params, output_path)
                success_paths.append(output_path)
                
            except Exception as e:
                print(f"❌ 处理失败 {filename}: {e}")
                continue
        
        print(f"🎉 批量编辑完成! 成功: {len(success_paths)}/{len(image_files)}")
        return success_paths
    
    def get_system_info(self) -> dict:
        """获取系统信息"""
        info = {
            'device': str(self.encoder.device),
            'available_directions': self.direction_manager.list_available_directions(),
            'controlnet_available': self.controlnet_reshaper is not None,
            'generator_info': self.generator.get_model_info()
        }
        return info

def validate_params(params: dict) -> dict:
    """验证和约束编辑参数"""
    param_ranges = {
        'face_slim': (0, 0.5),
        'eye_enlarge': (0, 0.3),
        'nose_lift': (0, 0.3),
        'mouth_adjust': (0, 0.2),
        'skin_whiten': (0, 0.8)
    }
    
    validated = {}
    for param_name, value in params.items():
        if param_name in param_ranges:
            min_val, max_val = param_ranges[param_name]
            validated[param_name] = max(min_val, min(max_val, value))
            
            if validated[param_name] != value:
                print(f"⚠️ {param_name} 从 {value:.2f} 调整为 {validated[param_name]:.2f}")
        else:
            validated[param_name] = value
    
    return validated

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='StyleGAN人脸编辑工具')
    
    # 基本参数
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    
    # 编辑参数
    parser.add_argument('--face_slim', type=float, default=0.0, help='瘦脸强度 (0-0.5)')
    parser.add_argument('--eye_enlarge', type=float, default=0.0, help='眼部放大 (0-0.3)')
    parser.add_argument('--nose_lift', type=float, default=0.0, help='鼻梁增高 (0-0.3)')
    parser.add_argument('--mouth_adjust', type=float, default=0.0, help='嘴型调整 (0-0.2)')
    parser.add_argument('--skin_whiten', type=float, default=0.0, help='美白强度 (0-0.8)')
    
    # 系统参数
    parser.add_argument('--device', default='auto', help='计算设备')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    parser.add_argument('--setup', action='store_true', help='设置模式（生成方向向量）')
    parser.add_argument('--info', action='store_true', help='显示系统信息')
    parser.add_argument('--force_regenerate', action='store_true', help='强制重新生成方向向量')
    
    args = parser.parse_args()
    
    try:
        # 创建编辑器
        editor = StyleGANFaceEditor(device=args.device)
        
        if args.info:
            # 显示系统信息
            info = editor.get_system_info()
            print("📊 系统信息:")
            for key, value in info.items():
                print(f"   {key}: {value}")
            return
        
        if args.setup:
            # 设置模式
            print("⚙️ 设置方向向量...")
            directions = editor.setup_directions(force_regenerate=args.force_regenerate)
            print(f"✅ 设置完成，共 {len(directions)} 个方向向量")
            return
        
        # 准备编辑参数
        params = {
            'face_slim': args.face_slim,
            'eye_enlarge': args.eye_enlarge,
            'nose_lift': args.nose_lift,
            'mouth_adjust': args.mouth_adjust,
            'skin_whiten': args.skin_whiten
        }
        
        # 验证参数
        params = validate_params(params)
        
        # 过滤零值参数
        active_params = {k: v for k, v in params.items() if v > 0}
        if not active_params:
            print("⚠️ 所有编辑参数为零，将输出原图")
        else:
            print(f"🎯 编辑参数: {active_params}")
        
        if args.batch:
            # 批量处理模式
            print(f"📁 批量处理: {args.input} -> {args.output}")
            editor.batch_edit(args.input, args.output, params)
        else:
            # 单张图片处理
            editor.edit_face(args.input, params, args.output)
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
