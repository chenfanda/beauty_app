#!/usr/bin/env python3
"""
面部精细调整FastAPI后端服务
整合七大调整功能的REST API服务，支持React前端
"""

import cv2
import numpy as np
import gc
from PIL import Image
import warnings
import os
import base64
import io
import sys
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import mediapipe as mp

warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FacialAdjustmentProcessor:
    """面部调整处理器"""
    
    def __init__(self):
        self.modules = {}
        self.load_status = {}
        self.shared_face_mesh = None
        self._init_mediapipe()
        self._load_all_modules()
    
    def _init_mediapipe(self):
        """初始化共享MediaPipe实例"""
        try:
            mp_face_mesh = mp.solutions.face_mesh
            self.shared_face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            )
            print("✅ 共享MediaPipe实例创建成功")
        except Exception as e:
            print(f"❌ 创建MediaPipe实例失败: {e}")
            self.shared_face_mesh = None
    
    def _load_all_modules(self):
        """加载所有面部调整模块"""
        if not self.shared_face_mesh:
            print("❌ MediaPipe实例未创建，无法加载模块")
            return
        
        # 定义所有模块
        modules_to_load = [
            ('chin', 'chin_adjust', 'OptimizedChin3DAdjustment', '下巴调整'),
            ('eye_adjust', 'eye_adjust', 'SafeEyeAdjustment', '眼睛调整'),
            ('face_slim', 'face_slim', 'SimpleFaceSlimming', '瘦脸功能'),
            ('forehead', 'forehead_adjust', 'ForeheadAdjustment', '额头调整'),
            ('mouth', 'mouth_adjust', 'Mouth3DAdjustment', '嘴巴调整'),
            ('nose', 'nose_3d', 'Nose3DGeometry', '鼻子调整'),
            ('whitening', 'face_whitening_advanced', 'AdvancedFaceWhitening', '美白功能')
        ]
        
        for module_key, module_file, class_name, display_name in modules_to_load:
            try:
                module = __import__(module_file, fromlist=[class_name])
                class_obj = getattr(module, class_name)
                
                # 传入共享MediaPipe实例
                self.modules[module_key] = class_obj(face_mesh=self.shared_face_mesh)
                
                self.load_status[module_key] = {
                    'status': 'success', 
                    'message': f'{display_name}模块加载成功'
                }
                print(f"✅ {display_name}模块加载成功")
                
            except ImportError as e:
                self.modules[module_key] = None
                self.load_status[module_key] = {
                    'status': 'error', 
                    'message': f'{display_name}模块加载失败: {e}'
                }
                print(f"❌ {display_name}模块加载失败: {e}")
            except Exception as e:
                self.modules[module_key] = None
                self.load_status[module_key] = {
                    'status': 'error', 
                    'message': f'{display_name}模块初始化失败: {e}'
                }
                print(f"❌ {display_name}模块初始化失败: {e}")
    
    def get_available_modules(self) -> Dict[str, bool]:
        """获取可用模块状态"""
        return {
            key: module is not None 
            for key, module in self.modules.items()
        }
    
    def process_image(self, image: np.ndarray, params: Dict[str, Any]) -> tuple:
        """
        处理图片
        返回: (处理后的图片, 成功标志, 处理信息)
        """
        if image is None:
            return None, False, "输入图片为空"
        
        start_time = time.time()
        result = image.copy()
        processing_steps = []
        
        try:
            # 处理顺序：瘦脸 -> 下巴 -> 鼻子 -> 眼睛 -> 嘴巴 -> 额头 -> 美白
            process_order = [
                ('face_slim', 'face_slimming', ['intensity', 'region']),
                ('chin', 'process_image', ['type', 'intensity']),
                ('nose', 'process_image', ['type', 'intensity']),
                ('eye_adjust', 'process_eye_adjustment', ['mode', 'intensity']),
                ('mouth', 'process_image', ['type', 'intensity']),
                ('forehead', 'process_image', ['type', 'intensity']),
                ('whitening', 'face_whitening', ['intensity', 'region'])
            ]
            
            for module_key, method_name, param_keys in process_order:
                if not params.get(f'{module_key}_enabled', False):
                    continue
                    
                if self.modules[module_key] is None:
                    continue
                
                try:
                    # 构建参数
                    kwargs = self._build_module_params(module_key, params, param_keys)
                    
                    # 特殊处理眼睛调整模块
                    if module_key == 'eye_adjust':
                        mode = kwargs.get('mode', 'tps')
                        intensity = kwargs.get('intensity', 0.2)
                        
                        if mode == 'simple':
                            result = self.modules[module_key].test_simple_transform(result, intensity)
                        else:  # tps模式
                            result = self.modules[module_key].process_image(result, intensity)
                    else:
                        # 调用处理方法
                        method = getattr(self.modules[module_key], method_name)
                        result = method(result, **kwargs)
                    
                    # 记录处理步骤
                    intensity = kwargs.get('intensity', 0)
                    type_or_region = (kwargs.get('region') or 
                                    kwargs.get('transform_type') or 
                                    kwargs.get('mode'))
                    processing_steps.append(f"{module_key}({type_or_region}, {intensity:.2f})")
                    
                except Exception as e:
                    error_msg = f"{module_key}处理出错: {e}"
                    print(f"⚠️ {error_msg}")
                    processing_steps.append(f"{module_key}(错误)")
                    # 继续处理其他模块，不中断整个流程
        
        except Exception as e:
            error_msg = f"图片处理异常: {str(e)}"
            print(f"❌ {error_msg}")
            return image, False, error_msg
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 生成处理信息
        if processing_steps:
            info = f"处理完成，用时{processing_time:.2f}s，步骤: {' → '.join(processing_steps)}"
        else:
            info = "未启用任何调整功能"
        
        return result, True, info
    
    def _build_module_params(self, module_key: str, params: Dict[str, Any], param_keys: List[str]) -> Dict[str, Any]:
        """构建模块参数"""
        kwargs = {}
        
        for param_key in param_keys:
            if param_key == 'type':
                # 处理不同模块的type参数名称
                if module_key == 'face_slim':
                    kwargs['region'] = params.get(f'{module_key}_region', 'both')
                elif module_key == 'chin':
                    kwargs['transform_type'] = params.get(f'{module_key}_type', 'length')
                else:
                    kwargs['transform_type'] = params.get(f'{module_key}_type', 'wings')
            elif param_key == 'mode':
                # 眼睛调整的模式参数
                kwargs['mode'] = params.get(f'{module_key}_mode', 'tps')
            elif param_key == 'region':
                kwargs['region'] = params.get(f'{module_key}_region', 'both')
            elif param_key == 'intensity':
                kwargs['intensity'] = params.get(f'{module_key}_intensity', 0.3)
        
        return kwargs
    
    def cleanup_resources(self):
        """清理资源"""
        try:
            if self.shared_face_mesh:
                self.shared_face_mesh.close()
                self.shared_face_mesh = None
            
            # 清理模块
            for module_key in list(self.modules.keys()):
                if self.modules[module_key]:
                    del self.modules[module_key]
            
            gc.collect()
            print("🧹 资源清理完成")
        except Exception as e:
            print(f"⚠️ 资源清理异常: {e}")

# 创建全局处理器实例
processor = FacialAdjustmentProcessor()

# FastAPI应用实例
app = FastAPI(
    title="面部精细调整API",
    description="基于MediaPipe和计算机视觉的面部精细调整API服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 响应模型
class FacialAdjustmentResponse(BaseModel):
    success: bool
    message: str
    image_base64: Optional[str] = None
    processing_time: float
    processing_steps: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    mediapipe_status: str
    available_modules: Dict[str, bool]
    module_load_status: Dict[str, Dict[str, str]]

class AdjustmentParams(BaseModel):
    """调整参数模型"""
    # 下巴调整
    chin_enabled: bool = False
    chin_type: str = "length"  # length, width, sharp, round
    chin_intensity: float = 0.3
    
    # 眼睛调整
    eye_adjust_enabled: bool = False
    eye_adjust_mode: str = "tps"  # tps, simple
    eye_adjust_intensity: float = 0.2
    
    # 瘦脸功能
    face_slim_enabled: bool = False
    face_slim_region: str = "both"  # both, left, right
    face_slim_intensity: float = 0.3
    
    # 额头调整
    forehead_enabled: bool = False
    forehead_type: str = "shrink"  # shrink, expand
    forehead_intensity: float = 0.3
    
    # 嘴巴调整
    mouth_enabled: bool = False
    mouth_type: str = "thickness"  # thickness, peak, corners, resize, enlarge
    mouth_intensity: float = 0.3
    
    # 鼻子调整
    nose_enabled: bool = False
    nose_type: str = "wings"  # wings, tip, resize, bridge
    nose_intensity: float = 0.3
    
    # 美白功能
    whitening_enabled: bool = False
    whitening_region: str = "both"  # both, core, outer, left, right
    whitening_intensity: float = 0.5

# 工具函数
def numpy_to_base64(image_array: np.ndarray) -> str:
    """将numpy数组转换为base64字符串"""
    if len(image_array.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(image_array)
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64

def file_to_numpy(file: UploadFile) -> np.ndarray:
    """将上传文件转换为numpy数组"""
    contents = file.file.read()
    pil_image = Image.open(io.BytesIO(contents))
    
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # 转换为OpenCV格式 (BGR)
    image_np = np.array(pil_image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 限制图片大小
    height, width = image_cv.shape[:2]
    max_size = 1200
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_cv = cv2.resize(image_cv, (new_width, new_height))
    
    return image_cv

def validate_adjustment_params(params: dict) -> dict:
    """验证和规范化调整参数"""
    validated = {}
    
    # 验证类型选项
    valid_options = {
        'chin_type': ['length', 'width', 'sharp', 'round'],
        'eye_adjust_mode': ['tps', 'simple'],
        'face_slim_region': ['both', 'left', 'right'],
        'forehead_type': ['shrink', 'expand'],
        'mouth_type': ['thickness', 'peak', 'corners', 'resize', 'enlarge'],
        'nose_type': ['wings', 'tip', 'resize', 'bridge'],
        'whitening_region': ['both', 'core', 'outer', 'left', 'right']
    }
    
    # 验证强度范围
    intensity_ranges = {
        'chin_intensity': (0.1, 0.7),
        'eye_adjust_intensity': (0.05, 0.4),
        'face_slim_intensity': (0.1, 0.7),
        'forehead_intensity': (0.1, 0.7),
        'mouth_intensity': (0.1, 0.7),
        'nose_intensity': (0.1, 0.7),
        'whitening_intensity': (0.1, 0.8)
    }
    
    # 复制所有参数
    for key, value in params.items():
        validated[key] = value
    
    # 验证类型参数
    for param, options in valid_options.items():
        if param in validated and validated[param] not in options:
            validated[param] = options[0]  # 使用默认值
    
    # 验证强度参数
    for param, (min_val, max_val) in intensity_ranges.items():
        if param in validated:
            val = validated[param]
            if not isinstance(val, (int, float)) or val < min_val or val > max_val:
                validated[param] = (min_val + max_val) / 2  # 使用中间值
    
    return validated

# API端点
@app.get("/", summary="根端点")
async def root():
    """API根端点"""
    return {
        "service": "面部精细调整API服务",
        "version": "1.0.0",
        "description": "提供7种面部调整功能的REST API服务",
        "endpoints": {
            "health": "/health",
            "process": "/process",
            "modules": "/modules",
            "system_status": "/system-status",
            "cleanup": "/cleanup"
        },
        "supported_adjustments": [
            "下巴调整", "眼睛调整", "瘦脸功能", 
            "额头调整", "嘴巴调整", "鼻子调整", "美白功能"
        ]
    }

@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """健康检查端点"""
    mediapipe_status = "正常" if processor.shared_face_mesh else "异常"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        mediapipe_status=mediapipe_status,
        available_modules=processor.get_available_modules(),
        module_load_status=processor.load_status
    )

@app.get("/modules", summary="获取模块状态")
async def get_modules_status():
    """获取所有模块的详细状态"""
    return {
        "available_modules": processor.get_available_modules(),
        "load_status": processor.load_status,
        "mediapipe_instance": processor.shared_face_mesh is not None,
        "total_modules": len(processor.modules),
        "loaded_modules": sum(1 for m in processor.modules.values() if m is not None)
    }

@app.post("/process", response_model=FacialAdjustmentResponse, summary="面部调整处理")
async def process_facial_adjustment(
    file: UploadFile = File(..., description="要处理的图片文件"),
    # 下巴调整参数
    chin_enabled: bool = Form(False),
    chin_type: str = Form("length"),
    chin_intensity: float = Form(0.3),
    
    # 眼睛调整参数
    eye_adjust_enabled: bool = Form(False),
    eye_adjust_mode: str = Form("tps"),
    eye_adjust_intensity: float = Form(0.2),
    
    # 瘦脸功能参数
    face_slim_enabled: bool = Form(False),
    face_slim_region: str = Form("both"),
    face_slim_intensity: float = Form(0.3),
    
    # 额头调整参数
    forehead_enabled: bool = Form(False),
    forehead_type: str = Form("shrink"),
    forehead_intensity: float = Form(0.3),
    
    # 嘴巴调整参数
    mouth_enabled: bool = Form(False),
    mouth_type: str = Form("thickness"),
    mouth_intensity: float = Form(0.3),
    
    # 鼻子调整参数
    nose_enabled: bool = Form(False),
    nose_type: str = Form("wings"),
    nose_intensity: float = Form(0.3),
    
    # 美白功能参数
    whitening_enabled: bool = Form(False),
    whitening_region: str = Form("both"),
    whitening_intensity: float = Form(0.5)
):
    """
    面部调整处理API
    
    支持7种调整功能：
    - 下巴调整：拉长/收窄/尖锐/圆润
    - 眼睛调整：TPS模式/简单模式
    - 瘦脸功能：双侧/左脸/右脸
    - 额头调整：饱满/平整效果
    - 嘴巴调整：增厚/唇峰/嘴角/缩放
    - 鼻子调整：鼻翼/鼻尖/整体/鼻梁
    - 美白功能：分层/核心/外围等区域
    """
    
    # 验证文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="文件必须是图片格式")
    
    start_time = time.time()
    
    try:
        # 转换图片格式
        image_array = file_to_numpy(file)
        
        # 构建参数字典
        params = {
            'chin_enabled': chin_enabled,
            'chin_type': chin_type,
            'chin_intensity': chin_intensity,
            'eye_adjust_enabled': eye_adjust_enabled,
            'eye_adjust_mode': eye_adjust_mode,
            'eye_adjust_intensity': eye_adjust_intensity,
            'face_slim_enabled': face_slim_enabled,
            'face_slim_region': face_slim_region,
            'face_slim_intensity': face_slim_intensity,
            'forehead_enabled': forehead_enabled,
            'forehead_type': forehead_type,
            'forehead_intensity': forehead_intensity,
            'mouth_enabled': mouth_enabled,
            'mouth_type': mouth_type,
            'mouth_intensity': mouth_intensity,
            'nose_enabled': nose_enabled,
            'nose_type': nose_type,
            'nose_intensity': nose_intensity,
            'whitening_enabled': whitening_enabled,
            'whitening_region': whitening_region,
            'whitening_intensity': whitening_intensity
        }
        
        # 验证参数
        validated_params = validate_adjustment_params(params)
        
        # 处理图片
        result, success, message = processor.process_image(image_array, validated_params)
        
        if not success:
            return FacialAdjustmentResponse(
                success=False,
                message=message,
                processing_time=time.time() - start_time
            )
        
        # 转换为base64
        image_base64 = numpy_to_base64(result)
        
        # 获取处理步骤
        processing_steps = []
        for key, value in validated_params.items():
            if key.endswith('_enabled') and value:
                module_name = key.replace('_enabled', '')
                processing_steps.append(module_name)
        
        return FacialAdjustmentResponse(
            success=True,
            message=message,
            image_base64=image_base64,
            processing_time=time.time() - start_time,
            processing_steps=processing_steps
        )
    
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return FacialAdjustmentResponse(
            success=False,
            message=error_msg,
            processing_time=time.time() - start_time
        )

@app.get("/system-status", summary="系统状态")
async def system_status():
    """获取系统详细状态"""
    return {
        "mediapipe_status": "正常" if processor.shared_face_mesh else "异常",
        "available_modules": processor.get_available_modules(),
        "module_count": {
            "total": len(processor.modules),
            "loaded": sum(1 for m in processor.modules.values() if m is not None),
            "failed": sum(1 for m in processor.modules.values() if m is None)
        },
        "load_details": processor.load_status,
        "supported_features": {
            "chin_types": ["length", "width", "sharp", "round"],
            "eye_modes": ["tps", "simple"],
            "slim_regions": ["both", "left", "right"],
            "forehead_types": ["shrink", "expand"],
            "mouth_types": ["thickness", "peak", "corners", "resize", "enlarge"],
            "nose_types": ["wings", "tip", "resize", "bridge"],
            "whitening_regions": ["both", "core", "outer", "left", "right"]
        }
    }

@app.post("/cleanup", summary="清理资源")
async def cleanup_resources():
    """清理系统资源"""
    try:
        processor.cleanup_resources()
        return {"message": "资源清理完成", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"资源清理失败: {str(e)}")

if __name__ == "__main__":
    print("🚀 启动面部精细调整API服务...")
    print(f"🔧 MediaPipe状态: {'正常' if processor.shared_face_mesh else '异常'}")
    print(f"📊 模块加载状态: {sum(1 for m in processor.modules.values() if m is not None)}/{len(processor.modules)}")
    print(f"🌐 服务地址: http://0.0.0.0:8004")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        reload=False
    )
