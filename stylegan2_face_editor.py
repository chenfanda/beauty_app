#!/usr/bin/env python3
"""
StyleGAN2人脸编辑器 - 完整版
使用ReStyle编码器 + StyleGAN2生成器 + 语义方向向量
"""

import torch
import numpy as np
from PIL import Image
import os
import sys
import argparse
from typing import Dict, List, Union, Optional

# 添加utils路径
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

# 导入工具类
from face_aligner import FFHQFaceAligner
from restyle_encoder import ReStyleEncoder
from stylegan2_generator import StyleGAN2Generator

class StyleGAN2FaceEditor:
    def __init__(self, 
                 encoder_path: str = "models/encoders/restyle_e4e_ffhq_encode.pt",
                 generator_path: str = "models/stylegan2/stylegan2-ffhq-config-f.pkl",
                 directions_dir: str = "utils/generators-with-stylegan2/latent_directions",
                 predictor_path: str = "models/shape_predictor_68_face_landmarks.dat"):
        """
        初始化StyleGAN2人脸编辑器
        
        Args:
            encoder_path: ReStyle编码器路径
            generator_path: StyleGAN2生成器路径  
            directions_dir: 方向向量目录
            predictor_path: dlib人脸关键点检测器路径
        """
        print("🚀 初始化StyleGAN2人脸编辑器...")
        
        # 检查文件存在性
        self._check_files(encoder_path, generator_path, directions_dir, predictor_path)
        
        # 初始化各个组件
        print("📐 初始化人脸对齐器...")
        self.aligner = FFHQFaceAligner(predictor_path=predictor_path)
        
        print("🔍 初始化ReStyle编码器...")
        self.encoder = ReStyleEncoder(
            model_path=encoder_path,
            stylegan_path=generator_path
        )
        
        print("🎨 初始化StyleGAN2生成器...")
        self.generator = StyleGAN2Generator(model_path=generator_path)
        
        # 加载方向向量
        print("📂 加载语义方向向量...")
        self.directions = self._load_directions(directions_dir)
        
        print("✅ StyleGAN2人脸编辑器初始化完成！")
        print(f"   可用的编辑属性: {list(self.directions.keys())}")
    
    def _check_files(self, encoder_path, generator_path, directions_dir, predictor_path):
        """检查必要文件是否存在"""
        files_to_check = [
            (encoder_path, "ReStyle编码器"),
            (generator_path, "StyleGAN2生成器"), 
            (directions_dir, "方向向量目录"),
            (predictor_path, "dlib关键点检测器")
        ]
        
        for file_path, description in files_to_check:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{description}不存在: {file_path}")
    
    def _load_directions(self, directions_dir: str) -> Dict[str, torch.Tensor]:
        """加载所有可用的方向向量"""
        directions = {}
        
        if not os.path.exists(directions_dir):
            print(f"⚠️ 方向向量目录不存在: {directions_dir}")
            return directions
        
        # 扫描所有.npy文件
        for filename in os.listdir(directions_dir):
            if filename.endswith('.npy'):
                try:
                    file_path = os.path.join(directions_dir, filename)
                    direction_vector = np.load(file_path)
                    
                    # 转换为torch tensor
                    direction_tensor = torch.from_numpy(direction_vector).float()
                    
                    # 确保形状正确 (应该是 [18*512] 或 [18, 512])
                    if direction_tensor.numel() == 18 * 512:
                        if len(direction_tensor.shape) == 1:
                            direction_tensor = direction_tensor.view(18, 512)
                    
                    # 提取属性名（去掉.npy后缀）
                    attr_name = os.path.splitext(filename)[0]
                    directions[attr_name] = direction_tensor
                    
                    print(f"   ✅ {attr_name}: {direction_tensor.shape}")
                    
                except Exception as e:
                    print(f"   ❌ 加载失败 {filename}: {e}")
        
        print(f"📊 总共加载了 {len(directions)} 个方向向量")
        return directions
    
    def list_available_attributes(self) -> List[str]:
        """获取所有可用的编辑属性"""
        return list(self.directions.keys())
    
    def edit_face(self, 
                  image_input: Union[str, Image.Image],
                  edits: Dict[str, float],
                  output_path: Optional[str] = None,
                  return_intermediate: bool = False) -> Union[Image.Image, Dict]:
        """
        编辑人脸图像
        
        Args:
            image_input: 输入图像路径或PIL图像
            edits: 编辑字典，如 {'age': 0.5, 'smile': 1.0}
            output_path: 输出路径（可选）
            return_intermediate: 是否返回中间结果
            
        Returns:
            编辑后的图像或包含中间结果的字典
        """
        print(f"🎭 开始人脸编辑...")
        print(f"   编辑参数: {edits}")
        
        results = {}
        
        # 步骤1: 人脸对齐
        print("📐 步骤1: 人脸对齐")
        if isinstance(image_input, str):
            aligned_image = self.aligner.align_face(image_input)
            original_path = image_input
        else:
            # 如果是PIL图像，临时保存后对齐
            temp_path = "/tmp/temp_input.jpg"
            image_input.save(temp_path)
            aligned_image = self.aligner.align_face(temp_path)
            original_path = "PIL_Image"
            os.remove(temp_path)
        
        if aligned_image is None:
            raise RuntimeError("人脸对齐失败，请检查输入图像是否包含清晰的人脸")
        
        if return_intermediate:
            results['aligned'] = aligned_image
        
        # 步骤2: 编码到潜在空间
        print("🔍 步骤2: 编码到W+空间")
        latent_code = self.encoder.encode_image(aligned_image)  # [18, 512]
        print(f"   获得潜在向量: {latent_code.shape}")
        
        if return_intermediate:
            results['original_latent'] = latent_code.clone()
        
        # 步骤3: 应用方向向量编辑
        print("✏️ 步骤3: 应用语义编辑")
        edited_latent = latent_code.clone()
        
        applied_edits = []
        for attribute, strength in edits.items():
            if attribute in self.directions:
                direction = self.directions[attribute].to(latent_code.device)
                
                # 确保形状匹配
                if direction.shape != latent_code.shape:
                    print(f"   ⚠️ 调整方向向量形状: {direction.shape} -> {latent_code.shape}")
                    if direction.numel() == latent_code.numel():
                        direction = direction.view_as(latent_code)
                    else:
                        print(f"   ❌ 方向向量维度不匹配: {direction.shape} vs {latent_code.shape}")
                        continue
                
                # 应用编辑
                edited_latent += strength * direction
                applied_edits.append(f"{attribute}({strength})")
                print(f"   ✅ 应用 {attribute}: 强度 {strength}")
            else:
                print(f"   ⚠️ 未找到方向向量: {attribute}")
                print(f"   可用属性: {list(self.directions.keys())}")
        
        if not applied_edits:
            print("⚠️ 没有应用任何编辑，返回重建的原始图像")
        
        if return_intermediate:
            results['edited_latent'] = edited_latent.clone()
        
        # 步骤4: 生成编辑后的图像
        print("🎨 步骤4: 生成编辑后的图像")
        edited_image = self.generator.generate_from_latent(edited_latent.unsqueeze(0))
        
        # 保存结果
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            edited_image.save(output_path, quality=95)
            print(f"💾 编辑结果保存到: {output_path}")
        
        print(f"✅ 人脸编辑完成! 应用的编辑: {', '.join(applied_edits)}")
        
        if return_intermediate:
            results['edited_image'] = edited_image
            return results
        else:
            return edited_image
    
    def batch_edit_single_attribute(self, 
                                   image_input: Union[str, Image.Image],
                                   attribute: str,
                                   strength_range: List[float],
                                   output_dir: str) -> List[str]:
        """
        批量测试单个属性的不同强度
        
        Args:
            image_input: 输入图像
            attribute: 要编辑的属性
            strength_range: 强度范围列表，如[-2, -1, 0, 1, 2]
            output_dir: 输出目录
            
        Returns:
            生成的文件路径列表
        """
        if attribute not in self.directions:
            raise ValueError(f"属性 '{attribute}' 不存在，可用属性: {list(self.directions.keys())}")
        
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []
        
        for strength in strength_range:
            edits = {attribute: strength}
            output_filename = f"{attribute}_{strength:+.1f}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                self.edit_face(image_input, edits, output_path)
                generated_files.append(output_path)
            except Exception as e:
                print(f"❌ 生成失败 {output_filename}: {e}")
        
        return generated_files
    
    def create_attribute_grid(self, 
                             image_input: Union[str, Image.Image],
                             attributes: List[str],
                             strength_range: List[float],
                             output_dir: str) -> Dict[str, List[str]]:
        """
        创建属性编辑网格
        
        Args:
            image_input: 输入图像
            attributes: 要测试的属性列表
            strength_range: 强度范围
            output_dir: 输出目录
            
        Returns:
            每个属性生成的文件路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for attribute in attributes:
            if attribute in self.directions:
                print(f"\n🎯 测试属性: {attribute}")
                attr_dir = os.path.join(output_dir, attribute)
                files = self.batch_edit_single_attribute(
                    image_input, attribute, strength_range, attr_dir
                )
                results[attribute] = files
            else:
                print(f"⚠️ 跳过未知属性: {attribute}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="StyleGAN2人脸编辑器")
    parser.add_argument("input", help="输入图像路径")
    parser.add_argument("--output", "-o", help="输出图像路径")
    parser.add_argument("--encoder", default="models/encoders/restyle_e4e_ffhq_encode.pt",
                       help="ReStyle编码器路径")
    parser.add_argument("--generator", default="models/stylegan2/stylegan2-ffhq-config-f.pt",
                       help="StyleGAN2生成器路径")
    parser.add_argument("--directions", default="utils/generators-with-stylegan2/latent_directions",
                       help="方向向量目录")
    parser.add_argument("--predictor", default="models/shape_predictor_68_face_landmarks.dat",
                       help="dlib关键点检测器路径")
    
    # 编辑参数
    parser.add_argument("--age", type=float, default=0, help="年龄编辑强度")
    parser.add_argument("--gender", type=float, default=0, help="性别编辑强度")
    parser.add_argument("--smile", type=float, default=0, help="微笑编辑强度")
    parser.add_argument("--emotion_happy", type=float, default=0, help="快乐表情强度")
    parser.add_argument("--glasses", type=float, default=0, help="眼镜编辑强度")
    
    # 批量测试选项
    parser.add_argument("--test-all", action="store_true", help="测试所有可用属性")
    parser.add_argument("--test-range", nargs=3, metavar=("ATTR", "MIN", "MAX"),
                       help="测试单个属性的范围，如: --test-range age -2 2")
    
    args = parser.parse_args()
    
    try:
        # 初始化编辑器
        editor = StyleGAN2FaceEditor(
            encoder_path=args.encoder,
            generator_path=args.generator,
            directions_dir=args.directions,
            predictor_path=args.predictor
        )
        
        if args.test_all:
            # 测试所有属性
            print("🧪 测试模式: 所有属性")
            output_dir = os.path.splitext(args.input)[0] + "_test_all"
            
            attributes = editor.list_available_attributes()[:10]  # 限制前10个属性
            strength_range = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
            
            results = editor.create_attribute_grid(args.input, attributes, strength_range, output_dir)
            print(f"🎉 测试完成，结果保存在: {output_dir}")
            
        elif args.test_range:
            # 测试单个属性范围
            attr, min_val, max_val = args.test_range
            min_val, max_val = float(min_val), float(max_val)
            
            print(f"🧪 测试模式: {attr} 范围 [{min_val}, {max_val}]")
            output_dir = os.path.splitext(args.input)[0] + f"_test_{attr}"
            
            strength_range = np.linspace(min_val, max_val, 9).tolist()
            files = editor.batch_edit_single_attribute(args.input, attr, strength_range, output_dir)
            print(f"🎉 生成了 {len(files)} 个测试图像")
            
        else:
            # 单次编辑
            edits = {}
            for attr in ['age', 'gender', 'smile', 'emotion_happy', 'glasses']:
                value = getattr(args, attr)
                if value != 0:
                    edits[attr] = value
            
            if not edits:
                print("⚠️ 没有指定编辑参数，执行重建测试")
                edits = {}  # 空编辑，仅重建
            
            output_path = args.output or os.path.splitext(args.input)[0] + "_edited.jpg"
            
            edited_image = editor.edit_face(args.input, edits, output_path)
            print("🎉 编辑完成!")
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
