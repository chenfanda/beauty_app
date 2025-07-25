#!/usr/bin/env python3
"""
面部精细调整前端应用 - 基于Streamlit
整合下巴、眼角、眉毛、瘦脸、额头、嘴巴、鼻子七大调整功能
修复了session state修改问题
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import sys
import os
from typing import Dict, Any, Optional, Tuple

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FacialAdjustmentApp:
    def __init__(self):
        # 初始化各个调整模块
        self.modules = {}
        self.load_status = {}
        self._load_all_modules()
        
        # 初始化session state
        self._init_session_state()
    
    def _load_all_modules(self):
        """加载所有面部调整模块"""
        
        # 1. 下巴调整模块
        try:
            from chin_adjust import OptimizedChin3DAdjustment
            self.modules['chin'] = OptimizedChin3DAdjustment()
            self.load_status['chin'] = {'status': 'success', 'message': '下巴调整模块加载成功'}
            st.success("✅ 下巴调整模块加载成功")
        except ImportError as e:
            self.modules['chin'] = None
            self.load_status['chin'] = {'status': 'error', 'message': f'下巴调整模块加载失败: {e}'}
            st.error(f"❌ 下巴调整模块加载失败: {e}")
        
        # 2. 眼角调整模块
        try:
            from eye_corner_3d_complete import EyeCornerAdjuster
            self.modules['eye_corner'] = EyeCornerAdjuster()
            self.load_status['eye_corner'] = {'status': 'success', 'message': '眼角调整模块加载成功'}
            st.success("✅ 眼角调整模块加载成功")
        except ImportError as e:
            self.modules['eye_corner'] = None
            self.load_status['eye_corner'] = {'status': 'error', 'message': f'眼角调整模块加载失败: {e}'}
            st.error(f"❌ 眼角调整模块加载失败: {e}")
        
        # 3. 眉毛调整模块
        try:
            from eyebrow_adjust import CorrectEyebrowAdjustment
            self.modules['eyebrow'] = CorrectEyebrowAdjustment()
            self.load_status['eyebrow'] = {'status': 'success', 'message': '眉毛调整模块加载成功'}
            st.success("✅ 眉毛调整模块加载成功")
        except ImportError as e:
            self.modules['eyebrow'] = None
            self.load_status['eyebrow'] = {'status': 'error', 'message': f'眉毛调整模块加载失败: {e}'}
            st.error(f"❌ 眉毛调整模块加载失败: {e}")
        
        # 4. 瘦脸模块
        try:
            from face_slim import SimpleFaceSlimming
            self.modules['face_slim'] = SimpleFaceSlimming()
            self.load_status['face_slim'] = {'status': 'success', 'message': '瘦脸模块加载成功'}
            st.success("✅ 瘦脸模块加载成功")
        except ImportError as e:
            self.modules['face_slim'] = None
            self.load_status['face_slim'] = {'status': 'error', 'message': f'瘦脸模块加载失败: {e}'}
            st.error(f"❌ 瘦脸模块加载失败: {e}")
        
        # 5. 额头调整模块
        try:
            from forehead_adjust import ForeheadAdjustment
            self.modules['forehead'] = ForeheadAdjustment()
            self.load_status['forehead'] = {'status': 'success', 'message': '额头调整模块加载成功'}
            st.success("✅ 额头调整模块加载成功")
        except ImportError as e:
            self.modules['forehead'] = None
            self.load_status['forehead'] = {'status': 'error', 'message': f'额头调整模块加载失败: {e}'}
            st.error(f"❌ 额头调整模块加载失败: {e}")
        
        # 6. 嘴巴调整模块
        try:
            from mouth_adjust import Mouth3DAdjustment
            self.modules['mouth'] = Mouth3DAdjustment()
            self.load_status['mouth'] = {'status': 'success', 'message': '嘴巴调整模块加载成功'}
            st.success("✅ 嘴巴调整模块加载成功")
        except ImportError as e:
            self.modules['mouth'] = None
            self.load_status['mouth'] = {'status': 'error', 'message': f'嘴巴调整模块加载失败: {e}'}
            st.error(f"❌ 嘴巴调整模块加载失败: {e}")
        
        # 7. 鼻子调整模块
        try:
            from nose_3d import Nose3DGeometry
            self.modules['nose'] = Nose3DGeometry()
            self.load_status['nose'] = {'status': 'success', 'message': '鼻子调整模块加载成功'}
            st.success("✅ 鼻子调整模块加载成功")
        except ImportError as e:
            self.modules['nose'] = None
            self.load_status['nose'] = {'status': 'error', 'message': f'鼻子调整模块加载失败: {e}'}
            st.error(f"❌ 鼻子调整模块加载失败: {e}")
    
    def _init_session_state(self):
        """初始化session state"""
        if 'original_image' not in st.session_state:
            st.session_state.original_image = None
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = None
        if 'processing_time' not in st.session_state:
            st.session_state.processing_time = 0
        if 'last_params' not in st.session_state:
            st.session_state.last_params = None
        if 'adjustment_history' not in st.session_state:
            st.session_state.adjustment_history = []
        
        # 添加预设状态管理
        if 'preset_applied' not in st.session_state:
            st.session_state.preset_applied = None
        if 'preset_values' not in st.session_state:
            st.session_state.preset_values = {}
    
    def load_image(self, uploaded_file):
        """加载上传的图片"""
        if uploaded_file is not None:
            try:
                # 读取图片
                bytes_data = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(bytes_data))
                
                # 转换为OpenCV格式
                image_np = np.array(image)
                if len(image_np.shape) == 3:
                    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
                
                # 限制图片大小以提高处理速度
                height, width = image_cv.shape[:2]
                max_size = 1200
                if max(height, width) > max_size:
                    scale = max_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image_cv = cv2.resize(image_cv, (new_width, new_height))
                
                return image_cv
            except Exception as e:
                st.error(f"图片加载失败: {e}")
                return None
        return None
    
    def cv2_to_pil(self, cv_image):
        """将OpenCV图片转换为PIL图片"""
        try:
            if len(cv_image.shape) == 3:
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv_image
            return Image.fromarray(rgb_image)
        except Exception as e:
            st.error(f"图片格式转换失败: {e}")
            return None
    
    def get_preset_values(self, preset_type):
        """获取预设值"""
        presets = {
            'natural': {
                'chin_enabled': True, 'chin_type': 'round', 'chin_intensity': 0.2,
                'eye_enabled': True, 'eye_type': 'lift', 'eye_intensity': 0.25,
                'eyebrow_enabled': True, 'eyebrow_type': 'lift', 'eyebrow_intensity': 0.2,
                'slim_enabled': True, 'slim_region': 'both', 'slim_intensity': 0.4,
                'nose_enabled': True, 'nose_type': 'wings', 'nose_intensity': 0.2,
                'mouth_enabled': False, 'mouth_type': 'thickness', 'mouth_intensity': 0.0,
                'forehead_enabled': False, 'forehead_type': 'shrink', 'forehead_intensity': 0.0,
                'eye_auto_optimize': True
            },
            'glamour': {
                'chin_enabled': True, 'chin_type': 'sharp', 'chin_intensity': 0.35,
                'eye_enabled': True, 'eye_type': 'shape', 'eye_intensity': 0.4,
                'eyebrow_enabled': True, 'eyebrow_type': 'peak', 'eyebrow_intensity': 0.3,
                'slim_enabled': True, 'slim_region': 'both', 'slim_intensity': 0.6,
                'nose_enabled': True, 'nose_type': 'tip', 'nose_intensity': 0.3,
                'mouth_enabled': True, 'mouth_type': 'corners', 'mouth_intensity': 0.25,
                'forehead_enabled': True, 'forehead_type': 'shrink', 'forehead_intensity': 0.3,
                'eye_auto_optimize': True
            },
            'reset': {
                'chin_enabled': False, 'chin_type': 'length', 'chin_intensity': 0.0,
                'eye_enabled': False, 'eye_type': 'stretch', 'eye_intensity': 0.0,
                'eyebrow_enabled': False, 'eyebrow_type': 'lift', 'eyebrow_intensity': 0.0,
                'slim_enabled': False, 'slim_region': 'both', 'slim_intensity': 0.0,
                'nose_enabled': False, 'nose_type': 'wings', 'nose_intensity': 0.0,
                'mouth_enabled': False, 'mouth_type': 'thickness', 'mouth_intensity': 0.0,
                'forehead_enabled': False, 'forehead_type': 'shrink', 'forehead_intensity': 0.0,
                'eye_auto_optimize': True
            }
        }
        return presets.get(preset_type, {})
    
    def create_sidebar_controls(self):
        """创建侧边栏控制面板"""
        st.sidebar.title("🎨 面部精细调整控制")
        
        # 检查是否需要应用预设
        current_preset = st.session_state.get('preset_applied', None)
        preset_values = st.session_state.get('preset_values', {})
        
        params = {}
        
        # 下巴调整
        with st.sidebar.expander("🎯 下巴调整", expanded=False):
            if self.modules['chin'] is not None:
                # 使用预设值或默认值
                default_enabled = preset_values.get('chin_enabled', False)
                default_type = preset_values.get('chin_type', 'length')
                default_intensity = preset_values.get('chin_intensity', 0.3)
                
                params['chin_enabled'] = st.checkbox("启用下巴调整", value=default_enabled, key="chin_enabled")
                if params['chin_enabled']:
                    params['chin_type'] = st.selectbox(
                        "调整类型", 
                        ['length', 'width', 'sharp', 'round'],
                        format_func=lambda x: {'length': '拉长', 'width': '收窄', 'sharp': '尖锐', 'round': '圆润'}[x],
                        index=['length', 'width', 'sharp', 'round'].index(default_type),
                        key="chin_type"
                    )
                    params['chin_intensity'] = st.slider(
                        "调整强度", 0.1, 0.5, default_intensity, 0.05, key="chin_intensity",
                        help="下巴调整强度，0.3-0.4为推荐值"
                    )
                else:
                    params['chin_type'] = default_type
                    params['chin_intensity'] = 0.0
            else:
                st.error("❌ 下巴调整模块不可用")
                params['chin_enabled'] = False
                params['chin_type'] = 'length'
                params['chin_intensity'] = 0.0
        
        # 眼角调整
        with st.sidebar.expander("👁️ 眼角调整", expanded=False):
            if self.modules['eye_corner'] is not None:
                default_enabled = preset_values.get('eye_enabled', False)
                default_type = preset_values.get('eye_type', 'stretch')
                default_intensity = preset_values.get('eye_intensity', 0.3)
                default_optimize = preset_values.get('eye_auto_optimize', True)
                
                params['eye_enabled'] = st.checkbox("启用眼角调整", value=default_enabled, key="eye_enabled")
                if params['eye_enabled']:
                    params['eye_type'] = st.selectbox(
                        "调整类型",
                        ['stretch', 'lift', 'shape'],
                        format_func=lambda x: {'stretch': '拉长', 'lift': '上扬', 'shape': '综合'}[x],
                        index=['stretch', 'lift', 'shape'].index(default_type),
                        key="eye_type"
                    )
                    params['eye_intensity'] = st.slider(
                        "调整强度", 0.1, 0.5, default_intensity, 0.05, key="eye_intensity",
                        help="眼角调整强度，0.2-0.4为推荐值"
                    )
                    params['eye_auto_optimize'] = st.checkbox(
                        "智能优化", value=default_optimize, key="eye_auto_optimize",
                        help="根据眼形特征自动优化参数"
                    )
                else:
                    params['eye_type'] = default_type
                    params['eye_intensity'] = 0.0
                    params['eye_auto_optimize'] = default_optimize
            else:
                st.error("❌ 眼角调整模块不可用")
                params['eye_enabled'] = False
        
        # 眉毛调整
        with st.sidebar.expander("📐 眉毛调整", expanded=False):
            if self.modules['eyebrow'] is not None:
                default_enabled = preset_values.get('eyebrow_enabled', False)
                default_type = preset_values.get('eyebrow_type', 'lift')
                default_intensity = preset_values.get('eyebrow_intensity', 0.3)
                
                params['eyebrow_enabled'] = st.checkbox("启用眉毛调整", value=default_enabled, key="eyebrow_enabled")
                if params['eyebrow_enabled']:
                    params['eyebrow_type'] = st.selectbox(
                        "调整类型",
                        ['lift', 'peak', 'shape'],
                        format_func=lambda x: {'lift': '整体提升', 'peak': '眉峰抬高', 'shape': '眉形调整'}[x],
                        index=['lift', 'peak', 'shape'].index(default_type),
                        key="eyebrow_type"
                    )
                    params['eyebrow_intensity'] = st.slider(
                        "调整强度", 0.1, 0.4, default_intensity, 0.05, key="eyebrow_intensity",
                        help="眉毛调整强度，0.2-0.35为推荐值"
                    )
                else:
                    params['eyebrow_type'] = default_type
                    params['eyebrow_intensity'] = 0.0
            else:
                st.error("❌ 眉毛调整模块不可用")
                params['eyebrow_enabled'] = False
        
        # 瘦脸调整
        with st.sidebar.expander("🔥 瘦脸调整", expanded=False):
            if self.modules['face_slim'] is not None:
                default_enabled = preset_values.get('slim_enabled', False)
                default_region = preset_values.get('slim_region', 'both')
                default_intensity = preset_values.get('slim_intensity', 0.5)
                
                params['slim_enabled'] = st.checkbox("启用瘦脸", value=default_enabled, key="slim_enabled")
                if params['slim_enabled']:
                    params['slim_region'] = st.selectbox(
                        "调整区域",
                        ['both', 'left', 'right'],
                        format_func=lambda x: {'both': '双侧', 'left': '左脸', 'right': '右脸'}[x],
                        index=['both', 'left', 'right'].index(default_region),
                        key="slim_region"
                    )
                    params['slim_intensity'] = st.slider(
                        "调整强度", 0.3, 0.7, default_intensity, 0.05, key="slim_intensity",
                        help="瘦脸强度，0.4-0.6为推荐值"
                    )
                else:
                    params['slim_region'] = default_region
                    params['slim_intensity'] = 0.0
            else:
                st.error("❌ 瘦脸模块不可用")
                params['slim_enabled'] = False
        
        # 额头调整
        with st.sidebar.expander("🌟 额头调整", expanded=False):
            if self.modules['forehead'] is not None:
                default_enabled = preset_values.get('forehead_enabled', False)
                default_type = preset_values.get('forehead_type', 'shrink')
                default_intensity = preset_values.get('forehead_intensity', 0.3)
                
                params['forehead_enabled'] = st.checkbox("启用额头调整", value=default_enabled, key="forehead_enabled")
                if params['forehead_enabled']:
                    params['forehead_type'] = st.selectbox(
                        "调整类型",
                        ['shrink', 'expand'],
                        format_func=lambda x: {'shrink': '饱满效果', 'expand': '平整效果'}[x],
                        index=['shrink', 'expand'].index(default_type),
                        key="forehead_type"
                    )
                    params['forehead_intensity'] = st.slider(
                        "调整强度", 0.1, 0.8, default_intensity, 0.05, key="forehead_intensity",
                        help="额头调整强度，0.2-0.4为推荐值"
                    )
                else:
                    params['forehead_type'] = default_type
                    params['forehead_intensity'] = 0.0
            else:
                st.error("❌ 额头调整模块不可用")
                params['forehead_enabled'] = False
        
        # 嘴巴调整
        with st.sidebar.expander("💋 嘴巴调整", expanded=False):
            if self.modules['mouth'] is not None:
                default_enabled = preset_values.get('mouth_enabled', False)
                default_type = preset_values.get('mouth_type', 'thickness')
                default_intensity = preset_values.get('mouth_intensity', 0.3)
                
                params['mouth_enabled'] = st.checkbox("启用嘴巴调整", value=default_enabled, key="mouth_enabled")
                if params['mouth_enabled']:
                    params['mouth_type'] = st.selectbox(
                        "调整类型",
                        ['thickness', 'peak', 'corners', 'resize', 'enlarge'],
                        format_func=lambda x: {
                            'thickness': '唇部增厚', 'peak': '唇峰抬高', 
                            'corners': '嘴角上扬', 'resize': '嘴巴缩小', 'enlarge': '嘴巴放大'
                        }[x],
                        index=['thickness', 'peak', 'corners', 'resize', 'enlarge'].index(default_type),
                        key="mouth_type"
                    )
                    params['mouth_intensity'] = st.slider(
                        "调整强度", 0.1, 0.5, default_intensity, 0.05, key="mouth_intensity",
                        help="嘴巴调整强度，0.2-0.4为推荐值"
                    )
                else:
                    params['mouth_type'] = default_type
                    params['mouth_intensity'] = 0.0
            else:
                st.error("❌ 嘴巴调整模块不可用")
                params['mouth_enabled'] = False
        
        # 鼻子调整
        with st.sidebar.expander("👃 鼻子调整", expanded=False):
            if self.modules['nose'] is not None:
                default_enabled = preset_values.get('nose_enabled', False)
                default_type = preset_values.get('nose_type', 'wings')
                default_intensity = preset_values.get('nose_intensity', 0.3)
                
                params['nose_enabled'] = st.checkbox("启用鼻子调整", value=default_enabled, key="nose_enabled")
                if params['nose_enabled']:
                    params['nose_type'] = st.selectbox(
                        "调整类型",
                        ['wings', 'tip', 'resize', 'bridge'],
                        format_func=lambda x: {
                            'wings': '鼻翼收缩', 'tip': '鼻尖抬高', 
                            'resize': '整体缩小', 'bridge': '鼻梁变挺'
                        }[x],
                        index=['wings', 'tip', 'resize', 'bridge'].index(default_type),
                        key="nose_type"
                    )
                    params['nose_intensity'] = st.slider(
                        "调整强度", 0.1, 0.5, default_intensity, 0.05, key="nose_intensity",
                        help="鼻子调整强度，0.2-0.4为推荐值"
                    )
                else:
                    params['nose_type'] = default_type
                    params['nose_intensity'] = 0.0
            else:
                st.error("❌ 鼻子调整模块不可用")
                params['nose_enabled'] = False
        
        # 全局设置
        st.sidebar.markdown("---")
        st.sidebar.subheader("🛠️ 全局设置")
        
        # 预设效果
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("自然美颜", key="preset_natural", help="温和的自然美颜效果"):
                st.session_state.preset_applied = 'natural'
                st.session_state.preset_values = self.get_preset_values('natural')
                st.rerun()
        
        with col2:
            if st.button("精致美颜", key="preset_glamour", help="明显的精致美颜效果"):
                st.session_state.preset_applied = 'glamour'
                st.session_state.preset_values = self.get_preset_values('glamour')
                st.rerun()
        
        # 重置按钮
        if st.sidebar.button("🔄 重置所有参数", key="reset_all"):
            st.session_state.preset_applied = 'reset'
            st.session_state.preset_values = self.get_preset_values('reset')
            st.rerun()
        
        # 如果应用了预设，清除标记以避免重复应用
        if current_preset:
            st.session_state.preset_applied = None
            st.session_state.preset_values = {}
        
        return params
    
    def process_image(self, image, params):
        """处理图片"""
        if image is None:
            return None
        
        start_time = time.time()
        result = image.copy()
        processing_steps = []
        
        try:
            # 1. 瘦脸处理（通常最先进行）
            if params.get('slim_enabled', False) and self.modules['face_slim'] is not None:
                try:
                    result = self.modules['face_slim'].face_slimming(
                        result, 
                        intensity=params['slim_intensity'],
                        region=params['slim_region']
                    )
                    processing_steps.append(f"瘦脸({params['slim_region']}, {params['slim_intensity']:.2f})")
                except Exception as e:
                    st.warning(f"瘦脸处理出错: {e}")
            
            # 2. 下巴调整
            if params.get('chin_enabled', False) and self.modules['chin'] is not None:
                try:
                    result = self.modules['chin'].process_image(
                        result,
                        transform_type=params['chin_type'],
                        intensity=params['chin_intensity']
                    )
                    processing_steps.append(f"下巴({params['chin_type']}, {params['chin_intensity']:.2f})")
                except Exception as e:
                    st.warning(f"下巴调整出错: {e}")
            
            # 3. 鼻子调整
            if params.get('nose_enabled', False) and self.modules['nose'] is not None:
                try:
                    result = self.modules['nose'].process_image(
                        result,
                        transform_type=params['nose_type'],
                        intensity=params['nose_intensity']
                    )
                    processing_steps.append(f"鼻子({params['nose_type']}, {params['nose_intensity']:.2f})")
                except Exception as e:
                    st.warning(f"鼻子调整出错: {e}")
            
            # 4. 眼角调整
            if params.get('eye_enabled', False) and self.modules['eye_corner'] is not None:
                try:
                    result = self.modules['eye_corner'].process_image(
                        result,
                        adjustment_type=params['eye_type'],
                        intensity=params['eye_intensity'],
                        auto_optimize=params.get('eye_auto_optimize', True)
                    )
                    processing_steps.append(f"眼角({params['eye_type']}, {params['eye_intensity']:.2f})")
                except Exception as e:
                    st.warning(f"眼角调整出错: {e}")
            
            # 5. 眉毛调整
            if params.get('eyebrow_enabled', False) and self.modules['eyebrow'] is not None:
                try:
                    result = self.modules['eyebrow'].process_image(
                        result,
                        adjustment_type=params['eyebrow_type'],
                        intensity=params['eyebrow_intensity']
                    )
                    processing_steps.append(f"眉毛({params['eyebrow_type']}, {params['eyebrow_intensity']:.2f})")
                except Exception as e:
                    st.warning(f"眉毛调整出错: {e}")
            
            # 6. 嘴巴调整
            if params.get('mouth_enabled', False) and self.modules['mouth'] is not None:
                try:
                    result = self.modules['mouth'].process_image(
                        result,
                        transform_type=params['mouth_type'],
                        intensity=params['mouth_intensity']
                    )
                    processing_steps.append(f"嘴巴({params['mouth_type']}, {params['mouth_intensity']:.2f})")
                except Exception as e:
                    st.warning(f"嘴巴调整出错: {e}")
            
            # 7. 额头调整（通常最后进行）
            if params.get('forehead_enabled', False) and self.modules['forehead'] is not None:
                try:
                    result = self.modules['forehead'].process_image(
                        result,
                        transform_type=params['forehead_type'],
                        intensity=params['forehead_intensity']
                    )
                    processing_steps.append(f"额头({params['forehead_type']}, {params['forehead_intensity']:.2f})")
                except Exception as e:
                    st.warning(f"额头调整出错: {e}")
        
        except Exception as e:
            st.error(f"处理图片时出错: {str(e)}")
            return image
        
        end_time = time.time()
        st.session_state.processing_time = end_time - start_time
        st.session_state.adjustment_history = processing_steps
        
        return result
    
    def create_main_interface(self):
        """创建主界面"""
        st.title("🎨 面部精细调整系统")
        st.markdown("**专业级面部调整工具** - 整合7大核心调整功能")
        
        # 显示模块状态
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            chin_status = "✅ 已加载" if self.modules['chin'] else "❌ 未加载"
            eye_status = "✅ 已加载" if self.modules['eye_corner'] else "❌ 未加载"
            st.metric("下巴调整", chin_status)
            st.metric("眼角调整", eye_status)
        
        with col2:
            eyebrow_status = "✅ 已加载" if self.modules['eyebrow'] else "❌ 未加载"
            slim_status = "✅ 已加载" if self.modules['face_slim'] else "❌ 未加载"
            st.metric("眉毛调整", eyebrow_status)
            st.metric("瘦脸功能", slim_status)
        
        with col3:
            forehead_status = "✅ 已加载" if self.modules['forehead'] else "❌ 未加载"
            mouth_status = "✅ 已加载" if self.modules['mouth'] else "❌ 未加载"
            st.metric("额头调整", forehead_status)
            st.metric("嘴巴调整", mouth_status)
        
        with col4:
            nose_status = "✅ 已加载" if self.modules['nose'] else "❌ 未加载"
            loaded_count = sum(1 for module in self.modules.values() if module is not None)
            st.metric("鼻子调整", nose_status)
            st.metric("已加载模块", f"{loaded_count}/7")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择图片文件", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="支持 JPG, JPEG, PNG, BMP 格式，建议分辨率1200x1200以下"
        )
        
        if uploaded_file is not None:
            # 加载图片
            if st.session_state.original_image is None:
                with st.spinner("正在加载图片..."):
                    st.session_state.original_image = self.load_image(uploaded_file)
            
            # 获取参数
            params = self.create_sidebar_controls()
            
            # 检查参数是否有变化
            params_changed = st.session_state.last_params != params
            if params_changed:
                st.session_state.last_params = params.copy()
            
            # 实时处理
            if st.session_state.original_image is not None and params_changed:
                with st.spinner("正在处理图片..."):
                    processed_image = self.process_image(st.session_state.original_image, params)
                    st.session_state.processed_image = processed_image
            
            # 显示结果
            if st.session_state.processed_image is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 原始图片")
                    pil_original = self.cv2_to_pil(st.session_state.original_image)
                    if pil_original is not None:
                        st.image(pil_original, use_column_width=True)
                
                with col2:
                    st.subheader("✨ 处理结果")
                    pil_processed = self.cv2_to_pil(st.session_state.processed_image)
                    if pil_processed is not None:
                        st.image(pil_processed, use_column_width=True)
                
                # 显示处理时间和统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ 处理时间", f"{st.session_state.processing_time:.2f}s")
                with col2:
                    active_adjustments = len(st.session_state.adjustment_history)
                    st.metric("🎛️ 调整项目", active_adjustments)
                with col3:
                    total_changes = sum(1 for k, v in params.items() 
                                      if k.endswith('_enabled') and v)
                    st.metric("📊 启用功能", total_changes)
                
                # 显示调整历史
                if st.session_state.adjustment_history:
                    st.subheader("📋 调整历史")
                    history_text = " → ".join(st.session_state.adjustment_history)
                    st.text(history_text)
                
                # 下载按钮
                if st.session_state.processed_image is not None:
                    result_pil = self.cv2_to_pil(st.session_state.processed_image)
                    if result_pil is not None:
                        buf = io.BytesIO()
                        result_pil.save(buf, format='JPEG', quality=95)
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="📥 下载处理结果",
                            data=byte_im,
                            file_name=f"facial_adjustment_result_{int(time.time())}.jpg",
                            mime="image/jpeg"
                        )
                
                # 效果对比分析
                if st.button("📊 生成效果分析"):
                    self.show_effect_analysis(st.session_state.original_image, 
                                            st.session_state.processed_image, params)
            
            else:
                # 首次加载时显示原图
                if st.session_state.original_image is not None:
                    st.subheader("📷 上传的图片")
                    pil_original = self.cv2_to_pil(st.session_state.original_image)
                    if pil_original is not None:
                        st.image(pil_original, use_column_width=True)
                    st.info("👆 请在左侧调整参数开始精细调整")
        
        else:
            st.info("👆 请先上传一张图片开始面部精细调整")
            
            # 显示功能介绍
            self._show_feature_introduction()
    
    def show_effect_analysis(self, original, processed, params):
        """显示效果分析"""
        try:
            # 计算图像差异
            diff = cv2.absdiff(original, processed)
            total_diff = np.sum(diff)
            changed_pixels = np.sum(diff > 0)
            total_pixels = original.shape[0] * original.shape[1]
            change_ratio = changed_pixels / total_pixels
            
            st.subheader("📊 效果分析报告")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总像素变化", f"{total_diff:,}")
            with col2:
                st.metric("变化像素数", f"{changed_pixels:,}")
            with col3:
                st.metric("变化比例", f"{change_ratio:.2%}")
            with col4:
                avg_intensity = total_diff / (total_pixels * 255 * 3)
                st.metric("平均强度", f"{avg_intensity:.4f}")
            
            # 显示差异图
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔍 变化热力图")
                diff_pil = self.cv2_to_pil(diff_colored)
                if diff_pil:
                    st.image(diff_pil, use_column_width=True)
            
            with col2:
                st.subheader("📈 调整强度分布")
                enabled_params = {k: v for k, v in params.items() 
                                if k.endswith('_intensity') and v > 0}
                if enabled_params:
                    try:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 6))
                        names = [k.replace('_intensity', '').title() for k in enabled_params.keys()]
                        values = list(enabled_params.values())
                        bars = ax.bar(names, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'])
                        ax.set_ylabel('调整强度')
                        ax.set_title('各部位调整强度对比')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    except ImportError:
                        st.info("📊 需要安装matplotlib库才能显示强度分布图")
            
            # 建议和提示
            st.subheader("💡 优化建议")
            if change_ratio < 0.01:
                st.info("🔵 变化较小，可以考虑适当增加调整强度以获得更明显的效果")
            elif change_ratio > 0.05:
                st.warning("🟡 变化较大，建议适当降低调整强度以保持自然效果")
            else:
                st.success("🟢 调整效果适中，在理想范围内")
            
        except Exception as e:
            st.error(f"效果分析失败: {e}")
    
    def _show_feature_introduction(self):
        """显示功能介绍"""
        st.markdown("""
        ### 🌟 面部精细调整系统特色
        
        **🎯 七大核心调整功能**
        
        **1. 🎯 下巴调整**
        - 拉长/收窄/尖锐/圆润四种模式
        - 5层架构3D变形算法
        - 自适应3D姿态识别
        
        **2. 👁️ 眼角调整**
        - 拉长/上扬/综合塑形
        - 智能眼形特征分析
        - 自适应参数优化
        
        **3. 📐 眉毛调整**
        - 整体提升/眉峰抬高/眉形调整
        - 基于医学美学的移动策略
        - 对称性保护机制
        
        **4. 🔥 瘦脸功能**
        - 智能Jawline梯度收缩
        - 姿态自适应左右非对称处理
        - 可选择单侧或双侧调整
        
        **5. 🌟 额头调整**
        - 饱满/平整两种3D效果
        - 径向+垂直双重变形
        - 发际线精确控制
        
        **6. 💋 嘴巴调整**
        - 增厚/抬高/上扬/缩放多种模式
        - 3D几何感知变形
        - 精确唇部轮廓控制
        
        **7. 👃 鼻子调整**
        - 收缩/抬高/缩小/变挺
        - MediaPipe + 3D几何算法
        - PnP算法3D姿态估计
        
        ---
        
        ### 🛠️ 技术特色
        
        **🔬 先进算法**
        - MediaPipe 468点精确面部识别
        - 3D PnP算法姿态估计
        - TPS薄板样条变形
        - 智能mask生成和融合
        
        **🎨 用户体验**
        - 实时参数调整
        - 智能预设效果
        - 效果分析报告
        - 一键下载结果
        
        **🔒 安全保障**
        - 多层锚点保护
        - 结构完整性验证
        - 自适应强度控制
        - 边界安全检查
        
        ---
        
        ### 💡 使用建议
        
        **📸 图片要求**
        - 建议使用正脸照片
        - 分辨率不超过1200x1200
        - 光线充足，面部清晰
        - 避免过度修图的照片
        
        **⚙️ 参数调节**
        - 建议从小参数开始
        - 观察实时效果反馈
        - 使用预设效果快速入门
        - 可单独或组合使用功能
        
        **🎯 最佳实践**
        - 瘦脸通常最先进行
        - 五官调整按从大到小的顺序
        - 注意保持整体协调性
        - 适度调整保持自然感
        
        ---
        
        ### ⚠️ 注意事项
        
        - 处理时间取决于启用的功能数量
        - 建议在性能较好的设备上使用
        - 过高的参数可能导致不自然效果
        - 建议保存原图备份
        """)
    
    def run(self):
        """运行应用"""
        # 设置页面配置
        st.set_page_config(
            page_title="面部精细调整系统",
            page_icon="🎨",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 添加自定义CSS
        st.markdown("""
        <style>
        .stApp > div:first-child > div:first-child > div:first-child {
            padding-top: 1rem;
        }
        .stSelectbox > div > div > select {
            background-color: #f0f2f6;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .stButton > button {
            width: 100%;
        }
        .adjustment-history {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 0.25rem;
            border-left: 4px solid #007bff;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 创建主界面
        self.create_main_interface()
        
        # 在侧边栏底部显示系统信息
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 系统状态")
        
        loaded_modules = [name for name, module in self.modules.items() if module is not None]
        failed_modules = [name for name, module in self.modules.items() if module is None]
        
        st.sidebar.success(f"✅ 已加载: {', '.join(loaded_modules)}")
        if failed_modules:
            st.sidebar.error(f"❌ 未加载: {', '.join(failed_modules)}")
        
        st.sidebar.markdown(f"""
        **版本信息:**
        - 核心功能: {len(loaded_modules)}/{len(self.modules)}
        - 技术栈: MediaPipe + OpenCV + TPS
        - 3D算法: PnP姿态估计
        - 界面框架: Streamlit
        """)

def main():
    """主函数"""
    try:
        app = FacialAdjustmentApp()
        app.run()
    except Exception as e:
        st.error(f"应用启动失败: {e}")
        st.info("请检查所有模块文件是否存在并且依赖库已正确安装")
        
        # 显示调试信息
        with st.expander("🔧 调试信息"):
            st.code(f"""
错误类型: {type(e).__name__}
错误信息: {str(e)}

所需文件:
- chin_adjust.py (下巴调整)
- eye_corner_3d_complete.py (眼角调整)
- eyebrow_adjust.py (眉毛调整)
- face_slim.py (瘦脸功能)
- forehead_adjust.py (额头调整)
- mouth_adjust.py (嘴巴调整)
- nose_3d.py (鼻子调整)

所需依赖:
- streamlit
- opencv-python
- mediapipe
- numpy
- pillow
- matplotlib (可选，用于效果分析)

启动命令:
streamlit run facial_adjustment_app.py
            """)

if __name__ == "__main__":
    main()
