#!/usr/bin/env python3
"""
StyleGAN2人脸编辑器 - 改造版
支持动态热加载编码器、独立生成器和方向向量
解决导入路径冲突问题
"""

import torch
import numpy as np
from PIL import Image
import os
import sys
import argparse
import json
import hashlib
from typing import Dict, List, Union, Optional

# 添加utils路径
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

# 导入工具类
from face_aligner import FFHQFaceAligner

class StyleGAN2FaceEditor:
    def __init__(self, 
                 encoder_type: str = "restyle",
                 encoder_path: str = "models/encoders/restyle_e4e_ffhq_encode.pt",
                 independent_generator_path: str = None,
                 directions_dir: str = "utils/generators-with-stylegan2/latent_directions",
                 predictor_path: str = "models/shape_predictor_68_face_landmarks.dat",
                 encoder_config: Optional[Dict] = None):
        """
        初始化StyleGAN2人脸编辑器 - 改造版
        
        Args:
            encoder_type: 编码器类型 ("restyle", "restyle_s", "hyperstyle")
            encoder_path: 编码器路径
            independent_generator_path: 独立生成器路径(.pkl文件，用于最终生成)
            directions_dir: 方向向量目录
            predictor_path: dlib人脸关键点检测器路径
            encoder_config: 编码器配置字典
        """
        print("🚀 初始化StyleGAN2人脸编辑器 - 改造版...")
        
        # 初始化状态变量
        self.current_encoder = None
        self.current_encoder_type = None
        self.current_encoder_path = None
        self.current_encoder_config = {}
        
        # 记录编码器环境变化
        self.current_encoder_paths = []  # 记录当前编码器添加的路径
        self.current_encoder_modules = []  # 记录当前编码器导入的模块
        
        # 独立生成器缓存（支持多个.pkl生成器）
        self.generator_cache = {}
        self.current_independent_generator = None
        self.current_generator_path = None
        
        # 方向向量缓存
        self.directions = {}
        self.current_directions_dir = None
        
        # latent编码缓存（避免重复编码）
        self.latent_cache = {}
        
        # 检查基础文件
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"dlib关键点检测器不存在: {predictor_path}")
        
        # 初始化人脸对齐器
        print("📐 初始化人脸对齐器...")
        self.aligner = FFHQFaceAligner(predictor_path=predictor_path)
        
        # 初始化编码器
        if encoder_path and os.path.exists(encoder_path):
            print(f"🔍 初始化默认编码器: {encoder_type}")
            self.switch_encoder(encoder_type, encoder_path, encoder_config or {})
        
        # 初始化独立生成器
        if independent_generator_path and os.path.exists(independent_generator_path):
            print("🎨 初始化独立生成器...")
            self.switch_independent_generator(independent_generator_path)
        
        # 加载方向向量
        if directions_dir and os.path.exists(directions_dir):
            print("📂 加载方向向量...")
            self.switch_directions(directions_dir)
        
        print("✅ StyleGAN2人脸编辑器初始化完成！")
        if self.directions:
            print(f"   可用的编辑属性: {list(self.directions.keys())}")
    
    def _clean_previous_encoder_environment(self):
        """精确清理上一个编码器的环境（不清理PyTorch组件）"""
        print("🧹 清理上一个编码器环境...")
        
        # 清理上一个编码器添加的路径（只清理编码器相关路径）
        encoder_related_paths = []
        for path in self.current_encoder_paths:
            if any(pattern in path for pattern in ['hyperstyle', 'restyle', 'stylegan']):
                encoder_related_paths.append(path)
        
        for path in encoder_related_paths:
            if path in sys.path:
                sys.path.remove(path)
                print(f"  🗑️ 移除路径: {os.path.basename(path)}")
        
        # 清理上一个编码器导入的模块（只清理编码器特定模块）
        encoder_specific_modules = []
        for module in self.current_encoder_modules:
            if any(pattern in module for pattern in [
                'hyperstyle_encoder', 'restyle_encoder', 
                'utils.model_utils', 'utils.inference_utils',
                'models.e4e'
            ]):
                encoder_specific_modules.append(module)
        
        for module in encoder_specific_modules:
            if module in sys.modules:
                del sys.modules[module]
                print(f"  🗑️ 清理模块: {module}")
        
        # 重置记录
        self.current_encoder_paths = []
        self.current_encoder_modules = []
        
        # GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ 上一个编码器环境已清理")
    
    def _create_encoder_and_track_env(self, encoder_type: str, encoder_path: str, encoder_config: Dict):
        """创建编码器并跟踪其环境变化"""
        print(f"🔄 创建{encoder_type.upper()}编码器并跟踪环境变化...")
        
        # 记录创建前的状态
        initial_paths = sys.path.copy()
        initial_modules = set(sys.modules.keys())
        
        print(f"  📊 创建前状态: paths={len(initial_paths)}, modules={len(initial_modules)}")
        
        # 创建编码器（编码器会自己添加路径和导入模块）
        encoder = None
        if encoder_type.lower() == "restyle":
            from restyle_encoder import ReStyleEncoder
            encoder = ReStyleEncoder(
                model_path=encoder_path,
                stylegan_path=None,  # 编码器内部管理自己的生成器
                id_model_path=encoder_config.get('id_model_path', None),
                device='auto',
                n_iters=encoder_config.get('n_iters', 5)
            )
        
        elif encoder_type.lower() == "restyle_s":
            from restyle_encoder_s import ReStyleEncoder
            encoder = ReStyleEncoder(
                model_path=encoder_path,
                stylegan_path=None,  # 编码器内部管理自己的生成器
                device='auto',
                n_iters=encoder_config.get('n_iters', 5),
                refinement_steps=encoder_config.get('refinement_steps', 15),
                use_lpips=encoder_config.get('use_lpips', True),
                use_face_align=encoder_config.get('use_face_align', True)
            )
        
        elif encoder_type.lower() == "hyperstyle":
            from hyperstyle_encoder import HyperStyleEncoder
            encoder = HyperStyleEncoder(
                model_path=encoder_path,
                device='auto'
            )
        
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        
        # 记录编码器添加了什么（只记录编码器相关的变化）
        encoder_related_paths = [p for p in sys.path if p not in initial_paths 
                                and any(pattern in p for pattern in ['hyperstyle', 'restyle', 'stylegan'])]
        
        encoder_specific_modules = [k for k in sys.modules.keys() if k not in initial_modules
                                   and any(pattern in k for pattern in [
                                       'hyperstyle_encoder', 'restyle_encoder', 
                                       'utils.model_utils', 'utils.inference_utils',
                                       'models.e4e'
                                   ])]
        
        self.current_encoder_paths = encoder_related_paths
        self.current_encoder_modules = encoder_specific_modules
        
        print(f"  📊 编码器添加: paths={len(encoder_related_paths)}, modules={len(encoder_specific_modules)}")
        if encoder_related_paths:
            print(f"  📁 添加的路径: {[os.path.basename(p) for p in encoder_related_paths]}")
        if encoder_specific_modules:
            print(f"  📦 添加的模块: {encoder_specific_modules}")
        
        return encoder
    
    def _create_independent_generator(self, generator_path: str):
        """创建独立生成器实例(.pkl文件)"""
        print(f"🎨 创建独立生成器: {os.path.basename(generator_path)}")
        
        from stylegan2_generator import StyleGAN2Generator
        return StyleGAN2Generator(model_path=generator_path)
    
    def _load_directions(self, directions_dir: str) -> Dict[str, torch.Tensor]:
        """加载方向向量"""
        print(f"📂 加载方向向量: {directions_dir}")
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
    
    def _generate_encoder_cache_key(self, encoder_type: str, encoder_path: str, encoder_config: Dict) -> str:
        """生成编码器缓存键"""
        config_str = str(sorted(encoder_config.items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        path_hash = hashlib.md5(encoder_path.encode()).hexdigest()[:8]
        return f"{encoder_type}_{path_hash}_{config_hash}"
    
    def _need_encoder_reload(self, encoder_type: str, encoder_path: str, encoder_config: Dict) -> bool:
        """检查是否需要重新加载编码器"""
        if self.current_encoder is None:
            return True
            
        if (self.current_encoder_type != encoder_type or 
            self.current_encoder_path != encoder_path or
            self.current_encoder_config != encoder_config):
            return True
            
        return False
    
    def switch_encoder(self, encoder_type: str, encoder_path: str, encoder_config: Dict = None):
        """动态切换编码器 - 热加载"""
        if encoder_config is None:
            encoder_config = {}
        
        print(f"\n🔄 切换编码器: {encoder_type}")
        print(f"   路径: {encoder_path}")
        print(f"   配置: {encoder_config}")
        
        # 检查是否需要切换
        if not self._need_encoder_reload(encoder_type, encoder_path, encoder_config):
            print("✅ 编码器无需重新加载")
            return
        
        # 检查文件存在性
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"编码器文件不存在: {encoder_path}")
        
        try:
            # 1. 精确清理上一个编码器的环境
            if self.current_encoder is not None:
                self._clean_previous_encoder_environment()
            
            # 2. 创建新编码器并跟踪其环境变化
            self.current_encoder = self._create_encoder_and_track_env(encoder_type, encoder_path, encoder_config)
            
            # 3. 更新状态
            self.current_encoder_type = encoder_type
            self.current_encoder_path = encoder_path
            self.current_encoder_config = encoder_config.copy()
            
            # 4. 清理latent缓存（编码器变了）
            self.latent_cache.clear()
            
            print(f"✅ 编码器切换完成: {encoder_type}")
            
        except Exception as e:
            print(f"❌ 编码器切换失败: {e}")
            # 重置状态
            self.current_encoder = None
            self.current_encoder_type = None
            self.current_encoder_path = None
            self.current_encoder_config = {}
            self.current_encoder_paths = []
            self.current_encoder_modules = []
            raise
    
    def switch_independent_generator(self, generator_path: str):
        """动态切换独立生成器 - 轻量级缓存"""
        print(f"\n🎨 切换独立生成器: {os.path.basename(generator_path)}")
        
        # 检查文件存在性
        if not os.path.exists(generator_path):
            raise FileNotFoundError(f"生成器文件不存在: {generator_path}")
        
        # 检查是否已经是当前生成器
        if self.current_generator_path == generator_path:
            print("✅ 生成器无需切换")
            return
        
        try:
            # 检查缓存
            if generator_path not in self.generator_cache:
                # 缓存管理：最多保留3个生成器
                if len(self.generator_cache) >= 3:
                    # 删除最老的（简单FIFO策略）
                    oldest_path = list(self.generator_cache.keys())[0]
                    print(f"🗑️ 清理旧生成器缓存: {os.path.basename(oldest_path)}")
                    del self.generator_cache[oldest_path]
                    
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 加载新生成器
                self.generator_cache[generator_path] = self._create_independent_generator(generator_path)
            
            # 切换到新生成器
            self.current_independent_generator = self.generator_cache[generator_path]
            self.current_generator_path = generator_path
            
            print(f"✅ 独立生成器切换完成: {os.path.basename(generator_path)}")
            
        except Exception as e:
            print(f"❌ 独立生成器切换失败: {e}")
            raise
    
    def switch_directions(self, directions_dir: str):
        """动态切换方向向量"""
        print(f"\n📂 切换方向向量目录: {directions_dir}")
        
        # 检查是否需要切换
        if self.current_directions_dir == directions_dir:
            print("✅ 方向向量无需重新加载")
            return
        
        try:
            # 加载新方向向量
            new_directions = self._load_directions(directions_dir)
            
            # 更新状态
            self.directions = new_directions
            self.current_directions_dir = directions_dir
            
            print(f"✅ 方向向量切换完成，加载了 {len(self.directions)} 个属性")
            
        except Exception as e:
            print(f"❌ 方向向量切换失败: {e}")
            raise
    
    def get_encoder_info(self) -> Dict:
        """获取当前编码器信息"""
        return {
            'type': self.current_encoder_type,
            'path': self.current_encoder_path,
            'config': self.current_encoder_config,
            'loaded': self.current_encoder is not None
        }
    
    def get_generator_info(self) -> Dict:
        """获取当前独立生成器信息"""
        return {
            'path': self.current_generator_path,
            'loaded': self.current_independent_generator is not None,
            'cached_generators': list(self.generator_cache.keys())
        }
    
    def get_directions_info(self) -> Dict:
        """获取当前方向向量信息"""
        return {
            'directory': self.current_directions_dir,
            'count': len(self.directions),
            'attributes': list(self.directions.keys())
        }
    
    def list_available_attributes(self) -> List[str]:
        """获取所有可用的编辑属性"""
        return list(self.directions.keys())
    
    def _get_image_hash(self, image: Union[str, Image.Image]) -> str:
        """计算图像哈希值"""
        if isinstance(image, str):
            # 文件路径 + 修改时间
            stat = os.stat(image)
            hash_input = f"{image}_{stat.st_mtime}_{stat.st_size}"
        else:
            # PIL图像内容
            hash_input = str(image.tobytes())
        
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _get_latent_from_cache(self, image_input: Union[str, Image.Image]) -> Optional[torch.Tensor]:
        """从缓存获取latent编码"""
        if self.current_encoder is None:
            return None
        
        # 生成缓存键
        image_hash = self._get_image_hash(image_input)
        encoder_key = self._generate_encoder_cache_key(
            self.current_encoder_type, 
            self.current_encoder_path, 
            self.current_encoder_config
        )
        cache_key = f"{encoder_key}_{image_hash}"
        
        return self.latent_cache.get(cache_key, None)
    
    def _save_latent_to_cache(self, image_input: Union[str, Image.Image], latent: torch.Tensor):
        """保存latent编码到缓存"""
        if self.current_encoder is None:
            return
        
        # 生成缓存键
        image_hash = self._get_image_hash(image_input)
        encoder_key = self._generate_encoder_cache_key(
            self.current_encoder_type, 
            self.current_encoder_path, 
            self.current_encoder_config
        )
        cache_key = f"{encoder_key}_{image_hash}"
        
        # 缓存管理：最多保留10个latent
        if len(self.latent_cache) >= 10:
            # 删除最老的
            oldest_key = list(self.latent_cache.keys())[0]
            del self.latent_cache[oldest_key]
        
        self.latent_cache[cache_key] = latent.clone()
    
    def edit_face(self, 
                  image_input: Union[str, Image.Image],
                  edits: Dict[str, float],
                  output_path: Optional[str] = None,
                  return_intermediate: bool = False) -> Union[Image.Image, Dict]:
        """
        编辑人脸图像 - 改造版
        
        Args:
            image_input: 输入图像路径或PIL图像
            edits: 编辑字典，如 {'age': 0.5, 'smile': 1.0}
            output_path: 输出路径（可选）
            return_intermediate: 是否返回中间结果
            
        Returns:
            编辑后的图像或包含中间结果的字典
        """
        print(f"\n🎭 开始人脸编辑...")
        print(f"   编码器: {self.current_encoder_type or 'None'}")
        print(f"   独立生成器: {os.path.basename(self.current_generator_path) if self.current_generator_path else 'None'}")
        print(f"   编辑参数: {edits}")
        
        # 检查必要组件
        if self.current_encoder is None:
            raise RuntimeError("编码器未加载，请先调用 switch_encoder()")
        
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
        
        # 步骤2: 编码到潜在空间（带缓存）
        print("🔍 步骤2: 编码到W+空间")
        
        # 尝试从缓存获取
        latent_code = self._get_latent_from_cache(image_input)
        
        if latent_code is not None:
            print("⚡ 使用缓存的latent编码")
        else:
            print("🔄 执行新的编码...")
            latent_code = self.current_encoder.encode_image(aligned_image)
            self._save_latent_to_cache(image_input, latent_code)
        
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
        
        # 选择生成器：优先使用独立生成器，否则使用编码器内置生成器
        if self.current_independent_generator is not None:
            print("   使用独立生成器")
            # 确保latent有批次维度
            if len(edited_latent.shape) == 2:
                generator_input = edited_latent.unsqueeze(0)
            else:
                generator_input = edited_latent
            edited_image = self.current_independent_generator.generate_from_latent(generator_input)
        else:
            print("   ⚠️ 未加载独立生成器，编码器可能需要实现generate_from_latent方法")
            # 这里可能需要根据具体编码器实现调整
            raise RuntimeError("未加载独立生成器，无法生成最终图像")
        
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
    
    def clear_all_caches(self):
        """清理所有缓存"""
        print("🧹 清理所有缓存...")
        
        # 清理编码器环境
        if self.current_encoder is not None:
            self._clean_previous_encoder_environment()
        
        # 清理其他缓存
        self.latent_cache.clear()
        self.generator_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ 缓存清理完成")

# 保持原有的解析和main函数，但使用新的接口
def parse_edits(edits_str: str) -> Dict[str, float]:
    """解析编辑字符串为字典"""
    if not edits_str:
        return {}
    
    edits = {}
    try:
        pairs = edits_str.split(',')
        for pair in pairs:
            attr, value = pair.split(':')
            edits[attr.strip()] = float(value.strip())
    except Exception as e:
        raise ValueError(f"编辑参数格式错误: {edits_str}。正确格式: attr1:val1,attr2:val2")
    
    return edits

def main():
    parser = argparse.ArgumentParser(description="StyleGAN2人脸编辑器 - 改造版")
    parser.add_argument("input", help="输入图像路径")
    parser.add_argument("--output", "-o", help="输出图像路径")
    
    # 核心配置
    parser.add_argument("--encoder-type", choices=['restyle', 'restyle_s', 'hyperstyle'], 
                       default='restyle', help="编码器类型 (默认: restyle)")
    parser.add_argument("--encoder-path", help="自定义编码器路径")
    parser.add_argument("--generator", help="独立生成器路径(.pkl文件)")
    parser.add_argument("--directions", default="utils/generators-with-stylegan2/latent_directions",
                       help="方向向量目录")
    parser.add_argument("--predictor", default="models/shape_predictor_68_face_landmarks.dat",
                       help="dlib关键点检测器路径")
    
    # 编码器配置
    parser.add_argument("--encoder-config", help="编码器配置JSON字符串")
    
    # 编辑参数
    parser.add_argument("--edits", help="多属性编辑，格式: attr1:val1,attr2:val2")
    
    # 向后兼容的快捷参数
    parser.add_argument("--age", type=float, default=0, help="年龄编辑强度")
    parser.add_argument("--gender", type=float, default=0, help="性别编辑强度")
    parser.add_argument("--smile", type=float, default=0, help="微笑编辑强度")
    parser.add_argument("--emotion_happy", type=float, default=0, help="快乐表情强度")
    parser.add_argument("--glasses", type=float, default=0, help="眼镜编辑强度")
    
    # 便捷功能
    parser.add_argument("--list-attributes", action="store_true", help="列出所有可用属性")
    
    args = parser.parse_args()
    
    try:
        # 处理编码器配置
        encoder_config = None
        if args.encoder_config:
            try:
                encoder_config = json.loads(args.encoder_config)
            except json.JSONDecodeError as e:
                print(f"❌ 编码器配置JSON格式错误: {e}")
                return
        
        # 设置默认编码器路径
        encoder_path = args.encoder_path
        if encoder_path is None:
            if args.encoder_type == "restyle":
                encoder_path = "models/encoders/restyle_e4e_ffhq_encode.pt"
            elif args.encoder_type == "restyle_s":
                encoder_path = "models/encoders/restyle_e4e_ffhq_encode.pt"
            elif args.encoder_type == "hyperstyle":
                encoder_path = "models/encoders/hyperstyle_ffhq.pt"
        
        # 初始化编辑器
        editor = StyleGAN2FaceEditor(
            encoder_type=args.encoder_type,
            encoder_path=encoder_path,
            independent_generator_path=args.generator,
            directions_dir=args.directions,
            predictor_path=args.predictor,
            encoder_config=encoder_config
        )
        
        # 列出可用属性
        if args.list_attributes:
            print("\n📋 可用的编辑属性:")
            attributes = editor.list_available_attributes()
            for i, attr in enumerate(sorted(attributes), 1):
                print(f"   {i:2d}. {attr}")
            print(f"\n总共 {len(attributes)} 个属性")
            return
        
        # 构建编辑字典
        edits = {}
        
        # 从多属性字符串获取
        if args.edits:
            parsed_edits = parse_edits(args.edits)
            edits.update(parsed_edits)
        
        # 从快捷参数获取（向后兼容）
        for attr in ['age', 'gender', 'smile', 'emotion_happy', 'glasses']:
            value = getattr(args, attr)
            if value != 0:
                edits[attr] = value
        
        # 单次编辑
        if not edits:
            print("⚠️ 没有指定编辑参数，执行重建测试")
            print("💡 使用示例:")
            print("   python stylegan2_face_editor.py input.jpg --edits 'age:1.0,smile:0.5'")
            print("   python stylegan2_face_editor.py input.jpg --smile 0.5 --age 1.0")
            print("   python stylegan2_face_editor.py input.jpg --encoder-type restyle_s --encoder-config '{\"n_iters\": 8}'")
            edits = {}  # 空编辑，仅重建
        
        output_path = args.output or os.path.splitext(args.input)[0] + "_edited.jpg"
        
        # 显示当前配置信息
        print("\n📊 当前配置:")
        print(f"   编码器: {editor.get_encoder_info()}")
        print(f"   独立生成器: {editor.get_generator_info()}")
        print(f"   方向向量: {editor.get_directions_info()}")
        
        edited_image = editor.edit_face(args.input, edits, output_path)
        print("🎉 编辑完成!")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()