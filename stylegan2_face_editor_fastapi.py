#!/usr/bin/env python3
"""
StyleGAN2人脸编辑器FastAPI后端服务
基于StyleGAN2FaceEditor的REST API服务，支持React前端
预加载版本：启动时加载默认配置，运行时支持灵活切换
"""

import os
import sys
import glob
import base64
import io
import time
import gc
import hashlib
import json
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image
import torch

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入StyleGAN2编辑器
try:
    from stylegan2_face_editor import StyleGAN2FaceEditor
except ImportError as e:
    print(f"❌ 无法导入StyleGAN2FaceEditor: {e}")
    raise

# 全局配置
MODELS_DIR = "models"
ENCODERS_DIR = os.path.join(MODELS_DIR, "encoders")
GENERATORS_DIR = os.path.join(MODELS_DIR, "stylegan2")
PREDICTOR_PATH = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
DEFAULT_DIRECTIONS_DIR = "utils/generators-with-stylegan2/latent_directions"
LATENT_CACHE_DIR = "latent_cache"  # latent缓存目录

# 默认编码器配置
DEFAULT_ENCODER_CONFIG = {
    "n_iters": 5,
    "refinement_steps": 200,
    "use_lpips": True,
    "use_face_align": True
}

# 属性中文描述映射
ATTRIBUTE_DESCRIPTIONS = {
    "age": "年龄",
    "angle_horizontal": "水平角度",
    "angle_pitch": "俯仰角度", 
    "beauty": "颜值",
    "emotion_angry": "愤怒情绪",
    "emotion_disgust": "厌恶情绪",
    "emotion_easy": "平静情绪",
    "emotion_fear": "恐惧情绪",
    "emotion_happy": "快乐情绪",
    "emotion_sad": "悲伤情绪",
    "emotion_surprise": "惊讶情绪",
    "eyes_open": "眼睛睁开",
    "face_shape": "脸型",
    "gender": "性别",
    "glasses": "眼镜",
    "height": "脸部高度",
    "race_black": "黑人特征",
    "race_white": "白人特征", 
    "race_yellow": "亚洲人特征",
    "smile": "微笑",
    "width": "脸部宽度"
}

class StyleGAN2Processor:
    """StyleGAN2人脸编辑处理器"""
    
    def __init__(self):
        self.editor = None
        self.available_encoders = {}
        self.available_generators = {}
        self.current_generator = None
        self.current_directions_dir = None
        self.predictor_path = PREDICTOR_PATH
        self._init_latent_cache_dir()
        self._scan_models()
        self._init_and_preload()
    
    def _init_latent_cache_dir(self):
        """初始化latent缓存目录"""
        os.makedirs(LATENT_CACHE_DIR, exist_ok=True)
        print(f"📂 latent缓存目录: {LATENT_CACHE_DIR}")
    
    def _calculate_md5(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """通用MD5计算方法"""
        hash_md5 = hashlib.md5()
        
        if isinstance(data, str):
            # 文件路径
            with open(data, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        elif isinstance(data, bytes):
            # 字节数据
            hash_md5.update(data)
        elif isinstance(data, dict):
            # 字典数据
            data_str = json.dumps(data, sort_keys=True)
            hash_md5.update(data_str.encode())
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
        
        return hash_md5.hexdigest()
    
    def _check_editor_state(self, operation_name: str = "操作") -> tuple:
        """通用的编辑器状态检查"""
        if not self.editor:
            return False, f"编辑器未初始化，无法执行{operation_name}"
        return True, ""
    
    def _conditional_update(self, check_func, update_func, item_name: str) -> bool:
        """通用的条件更新方法"""
        try:
            if check_func():
                return False  # 无需更新
            
            success, message = update_func()
            if success:
                print(f"✅ {item_name}已更新: {message}")
                return True
            else:
                print(f"⚠️ {item_name}更新失败: {message}")
                return False
        except Exception as e:
            print(f"⚠️ {item_name}更新异常: {e}")
            return False
    
    def _get_current_encoder_config(self) -> Dict[str, Any]:
        """获取当前编码器配置"""
        if not self.editor:
            return {}
        encoder_info = self.editor.get_encoder_info()
        return encoder_info.get('config', {})
    
    def _scan_models(self):
        """扫描可用的模型文件"""
        # 扫描restyle编码器
        if os.path.exists(ENCODERS_DIR):
            restyle_files = glob.glob(os.path.join(ENCODERS_DIR, "*restyle*.pt"))
            for file in restyle_files:
                name = os.path.splitext(os.path.basename(file))[0]
                self.available_encoders[f"ReStyle_S - {name}"] = ("restyle_s", file)
        
        # 扫描StyleGAN2生成器
        if os.path.exists(GENERATORS_DIR):
            generator_files = glob.glob(os.path.join(GENERATORS_DIR, "*.pkl"))
            for file in generator_files:
                name = os.path.splitext(os.path.basename(file))[0]
                display_name = self._get_generator_display_name(name)
                self.available_generators[display_name] = file
        
        print(f"✅ 扫描完成: {len(self.available_encoders)} 个编码器, {len(self.available_generators)} 个生成器")
    
    def _get_generator_display_name(self, name: str) -> str:
        """获取生成器的显示名称"""
        name_lower = name.lower()
        if "ffhq" in name_lower:
            return "官方FFHQ模型"
        elif "baby" in name_lower:
            return "婴儿风格模型"
        elif "model" in name_lower:
            return "模特风格模型"
        elif "star" in name_lower:
            return "明星风格模型"
        elif "wanghong" in name_lower:
            return "网红风格模型"
        elif "yellow" in name_lower:
            return "亚洲人脸模型"
        else:
            return f"自定义模型 - {name}"
    
    def _init_and_preload(self):
        """初始化编辑器并预加载默认配置"""
        print("🚀 初始化StyleGAN2编辑器...")
        
        # 1. 初始化编辑器
        try:
            self.editor = StyleGAN2FaceEditor(
                encoder_type=None,
                encoder_path=None,
                independent_generator_path=None,
                directions_dir=None,
                predictor_path=self.predictor_path,
                encoder_config=None
            )
            print("✅ StyleGAN2编辑器初始化完成")
        except Exception as e:
            print(f"❌ 编辑器初始化失败: {e}")
            raise
        
        # 2. 预加载restyle_s编码器
        print("📤 预加载restyle_s编码器...")
        if self._preload_encoder():
            print("✅ restyle_s编码器预加载成功")
        else:
            raise Exception("restyle_s编码器预加载失败")
        
        # 3. 预加载默认FFHQ生成器
        print("📤 预加载默认FFHQ生成器...")
        if self._preload_generator():
            print("✅ 默认生成器预加载成功")
        else:
            raise Exception("默认生成器预加载失败")
        
        # 4. 预加载方向向量
        print("📤 预加载方向向量...")
        if self._preload_directions():
            print("✅ 方向向量预加载成功")
        else:
            raise Exception("方向向量预加载失败")
        
        print("🎉 所有组件预加载完成，服务已就绪")
    
    def _preload_encoder(self) -> bool:
        """预加载编码器"""
        if not self.available_encoders:
            print("❌ 未找到可用的编码器")
            return False
        
        try:
            # 选择第一个可用的restyle编码器
            encoder_name = list(self.available_encoders.keys())[0]
            encoder_type, encoder_path = self.available_encoders[encoder_name]
            
            self.editor.switch_encoder(encoder_type, encoder_path, DEFAULT_ENCODER_CONFIG)
            print(f"✅ 编码器已加载: {encoder_name}")
            return True
        except Exception as e:
            print(f"❌ 编码器预加载失败: {e}")
            return False
    
    def _preload_generator(self) -> bool:
        """预加载生成器"""
        if not self.available_generators:
            print("❌ 未找到可用的生成器")
            return False
        
        try:
            # 优先选择FFHQ模型
            default_generator = None
            for name, path in self.available_generators.items():
                if "ffhq" in name.lower() or "官方" in name:
                    default_generator = name
                    break
            
            # 如果没有FFHQ，选择第一个可用的
            if not default_generator:
                default_generator = list(self.available_generators.keys())[0]
            
            self.editor.switch_independent_generator(self.available_generators[default_generator])
            self.current_generator = default_generator
            print(f"✅ 生成器已加载: {default_generator}")
            return True
        except Exception as e:
            print(f"❌ 生成器预加载失败: {e}")
            return False
    
    def _preload_directions(self) -> bool:
        """预加载方向向量"""
        if not os.path.exists(DEFAULT_DIRECTIONS_DIR):
            print(f"❌ 默认方向向量目录不存在: {DEFAULT_DIRECTIONS_DIR}")
            return False
        
        try:
            self.editor.switch_directions(DEFAULT_DIRECTIONS_DIR)
            self.current_directions_dir = DEFAULT_DIRECTIONS_DIR
            directions_info = self.editor.get_directions_info()
            print(f"✅ 方向向量已加载: {directions_info['count']} 个属性")
            return True
        except Exception as e:
            print(f"❌ 方向向量预加载失败: {e}")
            return False
    
    def _cleanup_old_latent_cache(self):
        """清理旧的latent缓存，保持最新5000个"""
        cache_files = glob.glob(os.path.join(LATENT_CACHE_DIR, "*.pt"))
        if len(cache_files) > 5000:
            # 按修改时间排序，删除最旧的
            cache_files.sort(key=os.path.getmtime)
            removed_count = 0
            for old_file in cache_files[:-5000]:
                try:
                    os.remove(old_file)
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ 删除缓存文件失败: {e}")
            print(f"🧹 清理了 {removed_count} 个旧的latent缓存文件")
    
    def get_available_encoders(self) -> Dict[str, str]:
        """获取可用编码器列表"""
        return {name: info[1] for name, info in self.available_encoders.items()}
    
    def get_available_generators(self) -> Dict[str, str]:
        """获取可用生成器列表"""
        return self.available_generators.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if not self.editor:
            return {"editor_initialized": False}
        
        encoder_info = self.editor.get_encoder_info()
        generator_info = self.editor.get_generator_info()
        directions_info = self.editor.get_directions_info()
        
        return {
            "editor_initialized": True,
            "encoder_loaded": encoder_info['loaded'],
            "encoder_type": encoder_info.get('type'),
            "encoder_config": encoder_info.get('config', {}),
            "generator_loaded": generator_info['loaded'],
            "generator_path": generator_info.get('path'),
            "current_generator": self.current_generator,
            "directions_loaded": directions_info['count'] > 0,
            "directions_count": directions_info['count'],
            "current_directions_dir": self.current_directions_dir,
            "available_attributes": self.editor.list_available_attributes()
        }
    
    def update_encoder_config_if_needed(self, new_config: Dict[str, Any]) -> bool:
        """检查编码器配置是否需要更新"""
        def check_same():
            current_config = self._get_current_encoder_config()
            return current_config == new_config
        
        def update():
            return self._update_encoder_config(new_config)
        
        return self._conditional_update(check_same, update, "编码器配置")
    
    def _update_encoder_config(self, config: Dict[str, Any]) -> tuple:
        """动态更新编码器配置（不重新加载模型）"""
        state_ok, msg = self._check_editor_state("编码器配置更新")
        if not state_ok:
            return False, msg
        
        encoder_info = self.editor.get_encoder_info()
        if not encoder_info['loaded']:
            return False, "编码器未加载"
        
        try:
            # 直接更新编码器实例的配置参数
            if hasattr(self.editor, 'current_encoder') and self.editor.current_encoder:
                if 'n_iters' in config:
                    self.editor.current_encoder.n_iters = config['n_iters']
                if 'refinement_steps' in config:
                    self.editor.current_encoder.refinement_steps = config['refinement_steps']
                if 'use_lpips' in config:
                    self.editor.current_encoder.use_lpips = config['use_lpips']
                if 'use_face_align' in config:
                    self.editor.current_encoder.use_face_align = config['use_face_align']
                
                return True, f"编码器配置已更新: {config}"
            else:
                return False, "编码器实例不可访问"
        except Exception as e:
            return False, f"配置更新失败: {str(e)}"
    
    def switch_generator_if_needed(self, generator_name: str) -> bool:
        """检查生成器是否需要切换"""
        def check_same():
            return self.current_generator == generator_name
        
        def update():
            return self.switch_generator(generator_name)
        
        return self._conditional_update(check_same, update, "生成器")
    
    def switch_generator(self, generator_name: str) -> tuple:
        """切换生成器"""
        state_ok, msg = self._check_editor_state("生成器切换")
        if not state_ok:
            return False, msg
        
        if generator_name not in self.available_generators:
            return False, f"生成器不存在: {generator_name}"
        
        try:
            generator_path = self.available_generators[generator_name]
            self.editor.switch_independent_generator(generator_path)
            self.current_generator = generator_name
            return True, f"生成器已切换: {generator_name}"
        except Exception as e:
            return False, f"生成器切换失败: {str(e)}"
    
    def switch_directions_if_needed(self, directions_dir: str) -> bool:
        """检查方向向量目录是否需要切换"""
        def check_same():
            return self.current_directions_dir == directions_dir
        
        def update():
            return self.switch_directions(directions_dir)
        
        return self._conditional_update(check_same, update, "方向向量")
    
    def switch_directions(self, directions_dir: str) -> tuple:
        """切换方向向量目录"""
        state_ok, msg = self._check_editor_state("方向向量切换")
        if not state_ok:
            return False, msg
        
        if not os.path.exists(directions_dir):
            return False, f"方向向量目录不存在: {directions_dir}"
        
        try:
            self.editor.switch_directions(directions_dir)
            self.current_directions_dir = directions_dir
            return True, f"方向向量已切换: {directions_dir}"
        except Exception as e:
            return False, f"方向向量切换失败: {str(e)}"
    
    def edit_face_with_latent_cache(self, image_data: bytes, encoder_config: Dict[str, Any], edit_attributes: Dict[str, float]) -> tuple:
        """
        带latent缓存的完整人脸编辑流程
        
        参数:
        - image_data: 原始图片字节数据
        - encoder_config: 编码器配置字典
        - edit_attributes: 编辑属性字典
        
        返回:
        - (edited_image, success, message)
        """
        state_ok, msg = self._check_editor_state("人脸编辑")
        if not state_ok:
            return None, False, msg
        
        try:
            print("🔍 检查latent缓存...")
            
            # 计算缓存key
            image_md5 = self._calculate_md5(image_data)
            config_md5 = self._calculate_md5(encoder_config)
            cache_filename = f"{image_md5}_{config_md5}.pt"
            cache_path = os.path.join(LATENT_CACHE_DIR, cache_filename)
            
            latent = None
            from_cache = False
            
            # 检查缓存是否存在
            if os.path.exists(cache_path):
                try:
                    latent = torch.load(cache_path, map_location='cpu')
                    print(f"⚡ 使用缓存的latent: {cache_filename}")
                    from_cache = True
                except Exception as e:
                    print(f"⚠️ 加载缓存失败，重新编码: {e}")
                    # 删除损坏的缓存文件
                    try:
                        os.remove(cache_path)
                    except:
                        pass
            
            # 如果没有缓存，需要编码
            if latent is None:
                print(f"🔄 重新编码并缓存: {cache_filename}")
                
                # 将图片数据转换为PIL Image
                image = Image.open(io.BytesIO(image_data))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 人脸对齐（参考原编辑器的正确实现）
                temp_path = "/tmp/temp_input.jpg"
                image.save(temp_path)
                aligned_image = self.editor.aligner.align_face(temp_path)
                os.remove(temp_path)
                
                if aligned_image is None:
                    return None, False, "人脸对齐失败，请检查输入图像是否包含清晰的人脸"
                
                # 直接编码获取latent
                latent = self.editor.current_encoder.encode_image(aligned_image)
                
                # 保存到缓存
                torch.save(latent, cache_path)
                print(f"💾 latent已缓存: {cache_filename}")
                # 清理旧缓存
                self._cleanup_old_latent_cache()
            
            # 检查是否需要编辑latent
            if edit_attributes:
                print("🎨 进行语义编辑...")
                edited_latent = latent.clone()
                
                applied_edits = []
                for attribute, strength in edit_attributes.items():
                    if attribute in self.editor.directions:
                        direction = self.editor.directions[attribute].to(latent.device)
                        
                        # 确保形状匹配
                        if direction.shape != latent.shape:
                            print(f"   ⚠️ 调整方向向量形状: {direction.shape} -> {latent.shape}")
                            if direction.numel() == latent.numel():
                                direction = direction.view_as(latent)
                            else:
                                print(f"   ❌ 方向向量维度不匹配: {direction.shape} vs {latent.shape}")
                                continue
                        
                        # 应用编辑
                        edited_latent += strength * direction
                        applied_edits.append(f"{attribute}({strength})")
                        print(f"   ✅ 应用 {attribute}: 强度 {strength}")
                    else:
                        print(f"   ⚠️ 未找到方向向量: {attribute}")
            else:
                print("📸 直接重建原图...")
                edited_latent = latent
                applied_edits = ["重建"]
            
            # 使用独立生成器生成最终图片
            print("🎨 生成最终图片...")
            
            # 确保latent有批次维度
            if len(edited_latent.shape) == 2:
                generator_input = edited_latent.unsqueeze(0)
            else:
                generator_input = edited_latent
            
            edited_image = self.editor.current_independent_generator.generate_from_latent(generator_input)
            
            success_msg = f"编辑成功"
            if from_cache:
                success_msg += " - 使用了缓存latent"
            else:
                success_msg += " - 新编码已缓存"
            
            if applied_edits:
                success_msg += f" - 应用编辑: {', '.join(applied_edits)}"
            
            return edited_image, True, success_msg
            
        except Exception as e:
            error_msg = f"人脸编辑失败: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return None, False, error_msg
    def edit_face(self, image: Image.Image, edit_attributes: Dict[str, float]) -> tuple:
        """编辑人脸（原有方法保持不变）"""
        if not self.editor:
            return None, False, "编辑器未初始化"
        
        try:
            edited_image = self.editor.edit_face(image, edit_attributes)
            return edited_image, True, "编辑成功"
        except Exception as e:
            error_msg = f"人脸编辑失败: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return None, False, error_msg
    
    def get_available_attributes(self) -> List[str]:
        """获取可用属性列表"""
        if not self.editor:
            return []
        return self.editor.list_available_attributes()
    
    def clear_caches(self):
        """清理编辑器缓存"""
        if self.editor:
            try:
                self.editor.clear_all_caches()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("🧹 编辑器缓存清理完成")
            except Exception as e:
                print(f"⚠️ 编辑器缓存清理异常: {e}")
    
    def clear_latent_cache(self):
        """清理latent缓存目录"""
        try:
            cache_files = glob.glob(os.path.join(LATENT_CACHE_DIR, "*.pt"))
            removed_count = 0
            for cache_file in cache_files:
                try:
                    os.remove(cache_file)
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ 删除缓存文件失败: {e}")
            print(f"🧹 清理了 {removed_count} 个latent缓存文件")
        except Exception as e:
            print(f"⚠️ latent缓存清理异常: {e}")
    
    def get_latent_cache_info(self) -> Dict[str, Any]:
        """获取latent缓存信息"""
        try:
            cache_files = glob.glob(os.path.join(LATENT_CACHE_DIR, "*.pt"))
            total_size = sum(os.path.getsize(f) for f in cache_files)
            return {
                "cache_count": len(cache_files),
                "total_size_mb": round(total_size / (1024*1024), 2),
                "cache_dir": LATENT_CACHE_DIR
            }
        except Exception as e:
            return {"error": str(e)}

# 创建全局处理器实例
print("🔧 正在启动StyleGAN2人脸编辑服务...")
try:
    processor = StyleGAN2Processor()
except Exception as e:
    print(f"❌ 服务启动失败: {e}")
    sys.exit(1)

# FastAPI应用实例
app = FastAPI(
    title="StyleGAN2人脸编辑API",
    description="基于StyleGAN2的人脸编辑API服务（预加载+灵活切换版本）",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 响应模型
class StyleGAN2Response(BaseModel):
    success: bool
    message: str
    image_base64: Optional[str] = None
    processing_time: float
    edit_attributes: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    editor_status: Dict[str, Any]
    available_models: Dict[str, Any]

# 工具函数
def numpy_to_base64(image: Image.Image) -> str:
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64

def file_to_pil(file: UploadFile) -> Image.Image:
    """将上传文件转换为PIL图像"""
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def validate_edit_attributes(attributes: Dict[str, float]) -> Dict[str, float]:
    """验证编辑属性参数"""
    validated = {}
    for attr, strength in attributes.items():
        if isinstance(strength, (int, float)):
            validated[attr] = max(-5.0, min(5.0, float(strength)))
        else:
            validated[attr] = 0.0
    return validated

def get_attribute_info(attr_name: str) -> Dict[str, str]:
    """获取属性信息"""
    return {
        "name": attr_name,
        "display_name": ATTRIBUTE_DESCRIPTIONS.get(attr_name, attr_name),
        "description": f"{ATTRIBUTE_DESCRIPTIONS.get(attr_name, attr_name)} ({attr_name})"
    }

# API端点
@app.get("/", summary="根端点")
async def root():
    """API根端点"""
    return {
        "service": "StyleGAN2人脸编辑API服务",
        "version": "1.0.0",
        "description": "基于StyleGAN2的人脸编辑服务（预加载+灵活切换版本）",
        "status": "ready",
        "preloaded_components": ["restyle_s编码器", "默认生成器", "方向向量"],
        "endpoints": {
            "health": "/health",
            "edit": "/edit",
            "attributes": "/attributes",
            "system_status": "/system-status",
            "cleanup": "/cleanup",
            "cleanup_latent": "/cleanup-latent",
            "latent_cache_info": "/latent-cache-info"
        }
    }

@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """健康检查端点"""
    editor_status = processor.get_status()
    available_models = {
        "encoders": processor.get_available_encoders(),
        "generators": processor.get_available_generators()
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        editor_status=editor_status,
        available_models=available_models
    )

@app.get("/attributes", summary="获取可用属性")
async def get_available_attributes():
    """获取可用的编辑属性"""
    attributes = processor.get_available_attributes()
    
    attribute_info = []
    for attr in attributes:
        info = get_attribute_info(attr)
        attribute_info.append(info)
    
    return {
        "attributes": attributes,
        "attribute_info": attribute_info,
        "total_count": len(attributes)
    }

@app.post("/edit", response_model=StyleGAN2Response, summary="人脸编辑")
async def edit_face(
    file: UploadFile = File(..., description="要编辑的人脸图片"),
    generator_name: str = Form("官方FFHQ模型", description="生成器名称"),
    directions_dir: str = Form("default", description="方向向量目录"),
    encoder_config: str = Form("{}", description="编码器配置JSON字符串"),
    edit_attributes: str = Form("{}", description="编辑属性JSON字符串"),
    save_temp: bool = Form(False, description="是否临时保存结果图片")
):
    """
    人脸编辑API（统一参数管理版本）
    
    参数说明：
    - file: 图片文件
    - generator_name: 生成器名称（如：官方FFHQ模型）
    - directions_dir: 方向向量目录（default使用默认目录）
    - encoder_config: 编码器配置，如：{"n_iters": 5, "refinement_steps": 200}
    - edit_attributes: 编辑属性，如：{"age": 2.0, "smile": 3.0}
    
    处理流程：
    1. 动态配置模型组件
    2. latent缓存检查（基于图片+编码器配置）
    3. 语义编辑和图像生成
    """
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="文件必须是图片格式")
    
    start_time = time.time()
    
    try:
        # Step 1: 参数解析和验证
        print("📋 Step 1: 参数解析和验证")
        
        # 解析编码器配置
        try:
            encoder_config_dict = json.loads(encoder_config) if encoder_config.strip() else {}
            # 填充默认配置
            default_config = {
                "n_iters": 5,
                "refinement_steps": 200,
                "use_lpips": True,
                "use_face_align": True
            }
            default_config.update(encoder_config_dict)
            encoder_config_dict = default_config
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="编码器配置JSON格式错误")
        
        # 解析编辑属性
        try:
            edit_attributes_dict = json.loads(edit_attributes) if edit_attributes.strip() else {}
            if not isinstance(edit_attributes_dict, dict):
                raise ValueError("编辑属性必须是字典格式")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="编辑属性JSON格式错误")
        
        # 验证属性参数
        validated_attributes = validate_edit_attributes(edit_attributes_dict)
        
        # 处理directions_dir参数
        actual_directions_dir = DEFAULT_DIRECTIONS_DIR if directions_dir == "default" else directions_dir
        
        print(f"  📝 编码器配置: {encoder_config_dict}")
        print(f"  🎨 生成器: {generator_name}")
        print(f"  📂 方向向量: {actual_directions_dir}")
        print(f"  ✏️ 编辑属性: {validated_attributes}")
        
        # Step 2: 动态配置模型组件
        print("🔧 Step 2: 动态配置模型组件")
        
        config_changed = processor.update_encoder_config_if_needed(encoder_config_dict)
        generator_changed = processor.switch_generator_if_needed(generator_name)  
        directions_changed = processor.switch_directions_if_needed(actual_directions_dir)
        
        # Step 3: 读取图片数据
        print("📸 Step 3: 读取图片数据")
        image_data = file.file.read()
        
        # Step 4: latent缓存检查和编辑
        print("⚡ Step 4: latent缓存检查和编辑")
        edited_image, success, message = processor.edit_face_with_latent_cache(
            image_data, encoder_config_dict, validated_attributes
        )
        
        if not success:
            return StyleGAN2Response(
                success=False,
                message=message,
                processing_time=time.time() - start_time
            )
        
        # Step 5: 返回结果
        print("📤 Step 5: 返回结果")
        image_base64 = numpy_to_base64(edited_image)

        # 可选保存到临时目录
        if save_temp:
            temp_dir = "temp_results"
            os.makedirs(temp_dir, exist_ok=True)
            temp_filename = f"result_{int(time.time())}_{hash(image_base64)[:8]}.png"
            temp_path = os.path.join(temp_dir, temp_filename)
            edited_image.save(temp_path)
            result_data["temp_file"] = temp_filename
            result_data["temp_url"] = f"/temp-results/{temp_filename}"
        
        return StyleGAN2Response(
            success=True,
            message=message,
            image_base64=image_base64,
            processing_time=time.time() - start_time,
            edit_attributes=validated_attributes
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"编辑失败: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return StyleGAN2Response(
            success=False,
            message=error_msg,
            processing_time=time.time() - start_time
        )

@app.get("/system-status", summary="系统状态")
async def system_status():
    """获取详细系统状态"""
    status = processor.get_status()
    latent_cache_info = processor.get_latent_cache_info()
    
    return {
        "editor_status": status,
        "model_counts": {
            "available_encoders": len(processor.get_available_encoders()),
            "available_generators": len(processor.get_available_generators())
        },
        "current_models": {
            "generator": processor.current_generator,
            "directions_dir": processor.current_directions_dir
        },
        "preloaded_config": {
            "encoder_config": DEFAULT_ENCODER_CONFIG,
            "default_directions": DEFAULT_DIRECTIONS_DIR
        },
        "latent_cache": latent_cache_info,
        "gpu_info": {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        },
        "attribute_descriptions": ATTRIBUTE_DESCRIPTIONS
    }

@app.post("/cleanup", summary="清理缓存")
async def cleanup_caches():
    """清理系统缓存"""
    try:
        processor.clear_caches()
        return {
            "message": "缓存清理完成",
            "timestamp": datetime.now().isoformat(),
            "status": processor.get_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"缓存清理失败: {str(e)}")

@app.post("/cleanup-latent", summary="清理latent缓存")
async def cleanup_latent_caches():
    """清理latent缓存"""
    try:
        cache_info_before = processor.get_latent_cache_info()
        processor.clear_latent_cache()
        cache_info_after = processor.get_latent_cache_info()
        
        return {
            "message": "latent缓存清理完成",
            "timestamp": datetime.now().isoformat(),
            "before": cache_info_before,
            "after": cache_info_after
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"latent缓存清理失败: {str(e)}")

@app.get("/latent-cache-info", summary="获取latent缓存信息")
async def get_latent_cache_info():
    """获取latent缓存详细信息"""
    return processor.get_latent_cache_info()

if __name__ == "__main__":
    print("🌐 启动StyleGAN2人脸编辑API服务...")
    print(f"📊 系统状态: {processor.get_status()}")
    print(f"🎨 可用生成器: {len(processor.get_available_generators())}")
    print(f"🔧 可用编码器: {len(processor.get_available_encoders())}")
    print(f"🌐 服务地址: http://0.0.0.0:8005")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        reload=False
    )
