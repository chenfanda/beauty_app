#!/usr/bin/env python3
"""
美颜应用前端 - 基于Streamlit的实时美颜编辑器
整合美形、美颜、美妆、美体四大功能模块
修复版本：解决API弃用和导入问题
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入我们创建的模块 - 添加更好的错误处理
def import_modules():
    """安全导入模块"""
    modules = {}
    
    try:
        # 尝试导入ControlNet版本
        from controlnet_face_reshape import ControlNetFaceReshaper
        modules['face_reshaper'] = ControlNetFaceReshaper
        modules['controlnet_available'] = True
        st.success("✅ ControlNet美形模块加载成功")
    except ImportError as e:
        st.warning(f"⚠️ ControlNet模块未找到: {e}")
        try:
            # 降级到传统美形
            from face_reshape import FaceReshaper
            modules['face_reshaper'] = FaceReshaper
            modules['controlnet_available'] = False
            st.info("📋 使用传统美形模块")
        except ImportError as e2:
            st.error(f"❌ 美形模块加载失败: {e2}")
            modules['face_reshaper'] = None
            modules['controlnet_available'] = False
    
    try:
        from face_beauty import FaceBeauty
        modules['face_beauty'] = FaceBeauty
        st.success("✅ 美颜模块加载成功")
    except ImportError as e:
        st.error(f"❌ 美颜模块加载失败: {e}")
        modules['face_beauty'] = None
    
    try:
        from face_makeup import FaceMakeup
        modules['face_makeup'] = FaceMakeup
        st.success("✅ 美妆模块加载成功")
    except ImportError as e:
        st.error(f"❌ 美妆模块加载失败: {e}")
        modules['face_makeup'] = None
    
    try:
        from body_reshape import BodyReshaper
        modules['body_reshaper'] = BodyReshaper
        st.success("✅ 美体模块加载成功")
    except ImportError as e:
        st.error(f"❌ 美体模块加载失败: {e}")
        modules['body_reshaper'] = None
    
    return modules

class BeautyApp:
    def __init__(self):
        # 导入模块
        self.modules = import_modules()
        
        # 初始化模块实例
        self.controlnet_available = self.modules.get('controlnet_available', False)
        
        # 初始化美形器
        if self.modules['face_reshaper'] is not None:
            try:
                if self.controlnet_available:
                    # ControlNet版本需要特殊初始化
                    self.face_reshaper = self.modules['face_reshaper']()
                else:
                    # 传统版本
                    self.face_reshaper = self.modules['face_reshaper']()
            except Exception as e:
                st.error(f"美形器初始化失败: {e}")
                self.face_reshaper = None
                self.controlnet_available = False
        else:
            self.face_reshaper = None
        
        # 初始化其他模块
        self.face_beauty = None
        self.face_makeup = None
        self.body_reshaper = None
        
        if self.modules['face_beauty'] is not None:
            try:
                self.face_beauty = self.modules['face_beauty']()
            except Exception as e:
                st.warning(f"美颜模块初始化失败: {e}")
        
        if self.modules['face_makeup'] is not None:
            try:
                self.face_makeup = self.modules['face_makeup']()
            except Exception as e:
                st.warning(f"美妆模块初始化失败: {e}")
        
        if self.modules['body_reshaper'] is not None:
            try:
                self.body_reshaper = self.modules['body_reshaper']()
            except Exception as e:
                st.warning(f"美体模块初始化失败: {e}")
        
        # 初始化session state
        self._init_session_state()
    
    def _init_session_state(self):
        """初始化session state"""
        if 'original_image' not in st.session_state:
            st.session_state.original_image = None
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = None
        if 'processing_time' not in st.session_state:
            st.session_state.processing_time = 0
        if 'landmarks_preview' not in st.session_state:
            st.session_state.landmarks_preview = None
        if 'last_params' not in st.session_state:
            st.session_state.last_params = None
    
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
                max_size = 800
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
    
    def process_image(self, image, params):
        """处理图片"""
        if image is None:
            return None
        
        start_time = time.time()
        result = image.copy()
        
        try:
            # 1. 美形处理
            if self.face_reshaper is not None:
                reshape_needed = any([
                    params['reshape']['face_slim'] > 0,
                    params['reshape']['nose_lift'] > 0,
                    params['reshape']['eye_enlarge'] > 0,
                    params['reshape'].get('mouth_adjust', 0) > 0  # 兼容传统版本
                ])
                
                if reshape_needed:
                    reshape_params = {}
                    if params['reshape']['face_slim'] > 0:
                        reshape_params['face_slim'] = params['reshape']['face_slim']
                    if params['reshape']['nose_lift'] > 0:
                        reshape_params['nose_lift'] = params['reshape']['nose_lift']
                    if params['reshape']['eye_enlarge'] > 0:
                        reshape_params['eye_enlarge'] = params['reshape']['eye_enlarge']
                    
                    # ControlNet版本特有参数
                    if self.controlnet_available and params['reshape'].get('mouth_adjust', 0) > 0:
                        reshape_params['mouth_adjust'] = params['reshape']['mouth_adjust']
                    
                    if reshape_params:
                        try:
                            result = self.face_reshaper.reshape_face(result, reshape_params)
                            st.success(f"✨ 美形处理完成，调整参数: {list(reshape_params.keys())}")
                        except Exception as e:
                            st.warning(f"美形处理出错: {e}")
            
            # 2. 美颜处理
            if self.face_beauty is not None:
                beauty_threshold = 0.05
                beauty_params = {}
                
                for key, value in params['beauty'].items():
                    if isinstance(value, (int, float)) and abs(value) > beauty_threshold:
                        beauty_params[key] = value
                
                if beauty_params:
                    beauty_params['smooth_method'] = params['beauty']['smooth_method']
                    try:
                        result = self.face_beauty.apply_beauty_filter(result, beauty_params)
                    except Exception as e:
                        st.warning(f"美颜处理出错: {e}")
            
            # 3. 美妆处理
            if self.face_makeup is not None:
                makeup_threshold = 0.05
                makeup_params = {}
                
                for key, value in params['makeup'].items():
                    if isinstance(value, (int, float)) and value > makeup_threshold:
                        makeup_params[key] = value
                    elif isinstance(value, str):  # 颜色选择
                        makeup_params[key] = value
                
                if any(isinstance(v, (int, float)) and v > makeup_threshold for v in makeup_params.values()):
                    try:
                        result = self.face_makeup.apply_makeup(result, makeup_params)
                    except Exception as e:
                        st.warning(f"美妆处理出错: {e}")
            
            # 4. 美体处理
            if self.body_reshaper is not None:
                body_threshold = 0.05
                body_params = {k: v for k, v in params['body'].items() if abs(v) > body_threshold}
                
                if body_params:
                    try:
                        result = self.body_reshaper.reshape_body(result, body_params)
                    except Exception as e:
                        st.warning(f"美体处理出错: {e}")
        
        except Exception as e:
            st.error(f"处理图片时出错: {str(e)}")
            return image
        
        end_time = time.time()
        st.session_state.processing_time = end_time - start_time
        
        return result
    
    def generate_landmarks_preview(self, image, params):
        """生成关键点预览（仅ControlNet版本支持）"""
        if not self.controlnet_available or self.face_reshaper is None:
            return None
        
        try:
            # 检查是否有extract_landmarks方法
            if not hasattr(self.face_reshaper, 'extract_landmarks'):
                return None
            
            # 提取原始关键点
            original_landmarks = self.face_reshaper.extract_landmarks(image)
            if original_landmarks is None:
                return None
            
            # 应用调整
            adjusted_landmarks = original_landmarks.copy()
            
            if params['reshape']['face_slim'] > 0:
                if hasattr(self.face_reshaper, 'apply_face_slim_adjustment'):
                    adjusted_landmarks = self.face_reshaper.apply_face_slim_adjustment(
                        adjusted_landmarks, params['reshape']['face_slim']
                    )
            
            if params['reshape']['eye_enlarge'] > 0:
                if hasattr(self.face_reshaper, 'apply_eye_enlargement'):
                    adjusted_landmarks = self.face_reshaper.apply_eye_enlargement(
                        adjusted_landmarks, params['reshape']['eye_enlarge']
                    )
            
            if params['reshape']['nose_lift'] > 0:
                if hasattr(self.face_reshaper, 'apply_nose_adjustment'):
                    adjusted_landmarks = self.face_reshaper.apply_nose_adjustment(
                        adjusted_landmarks, params['reshape']['nose_lift']
                    )
            
            if params['reshape']['mouth_adjust'] > 0:
                if hasattr(self.face_reshaper, 'apply_mouth_adjustment'):
                    adjusted_landmarks = self.face_reshaper.apply_mouth_adjustment(
                        adjusted_landmarks, params['reshape']['mouth_adjust']
                    )
            
            # 创建关键点可视化
            if hasattr(self.face_reshaper, 'create_landmark_image'):
                height, width = image.shape[:2]
                landmarks_image = self.face_reshaper.create_landmark_image(adjusted_landmarks,  height, width)
                return landmarks_image
            
        except Exception as e:
            st.error(f"关键点预览生成失败: {e}")
        
        return None
    
    def create_sidebar_controls(self):
        """创建侧边栏控制面板"""
        st.sidebar.title("🎨 美颜参数控制")
        
        params = {
            'reshape': {},
            'beauty': {},
            'makeup': {},
            'body': {}
        }
        
        # 美形参数
        with st.sidebar.expander("✨ 美形调整", expanded=True):
            if not self.controlnet_available:
                if self.face_reshaper is not None:
                    st.sidebar.warning("📋 传统美形算法")
                else:
                    st.sidebar.error("❌ 美形功能不可用")
            else:
                st.sidebar.success("🚀 使用ControlNet技术")
            
            params['reshape']['face_slim'] = st.slider(
                "瘦脸", 0.0, 1.0, 0.0, 0.1, key="face_slim",
                help="瘦脸效果，推荐0.3-0.5"
            )
            
            # 实时效果提示
            if params['reshape']['face_slim'] > 0:
                if params['reshape']['face_slim'] < 0.3:
                    st.sidebar.success("✨ 微调效果")
                elif params['reshape']['face_slim'] < 0.7:
                    st.sidebar.success("🔥 明显效果")
                else:
                    st.sidebar.warning("⚠️ 强烈效果")
            
            params['reshape']['nose_lift'] = st.slider(
                "鼻梁增高", 0.0, 1.0, 0.0, 0.1, key="nose_lift",
                help="让鼻梁更挺拔，推荐0.2-0.4"
            )
            params['reshape']['eye_enlarge'] = st.slider(
                "眼部放大", 0.0, 0.5, 0.0, 0.05, key="eye_enlarge",
                help="让眼睛更有神，推荐0.1-0.2"
            )
            
            # ControlNet特有功能
            if self.controlnet_available:
                params['reshape']['mouth_adjust'] = st.slider(
                    "嘴型调整", 0.0, 0.5, 0.0, 0.05, key="mouth_adjust",
                    help="微调嘴部形状，推荐0.1-0.2"
                )
                
                # 关键点预览选项
                preview_landmarks = st.checkbox(
                    "预览关键点", key="preview_landmarks",
                    help="显示调整后的面部关键点"
                )
                
                if preview_landmarks and st.session_state.original_image is not None:
                    landmarks_preview = self.generate_landmarks_preview(
                        st.session_state.original_image, params
                    )
                    if landmarks_preview is not None:
                        st.sidebar.image(
                            self.cv2_to_pil(landmarks_preview),
                            caption="关键点预览",
                            use_column_width=True  # 修复：使用正确的API
                        )
            else:
                params['reshape']['mouth_adjust'] = 0.0  # 传统版本不支持
        
        # 美颜参数
        with st.sidebar.expander("💫 美颜效果", expanded=True):
            params['beauty']['smooth_skin'] = st.slider(
                "磨皮强度", 0.0, 1.0, 0.0, 0.1, key="smooth_skin",
                help="皮肤平滑程度"
            )
            params['beauty']['smooth_method'] = st.selectbox(
                "磨皮方法", ['bilateral', 'frequency'], 
                index=0, key="smooth_method",
                help="bilateral: 快速模式, frequency: 高质量模式"
            )
            params['beauty']['skin_whitening'] = st.slider(
                "美白强度", 0.0, 1.0, 0.0, 0.1, key="skin_whitening",
                help="皮肤美白程度"
            )
            params['beauty']['skin_warmth'] = st.slider(
                "肤色暖度", -1.0, 1.0, 0.0, 0.1, key="skin_warmth",
                help="负值偏冷色调，正值偏暖色调"
            )
            params['beauty']['skin_saturation'] = st.slider(
                "肤色饱和度", -1.0, 1.0, 0.0, 0.1, key="skin_saturation",
                help="调整肤色饱和度"
            )
            params['beauty']['remove_blemishes'] = st.slider(
                "去瑕疵", 0.0, 1.0, 0.0, 0.1, key="remove_blemishes",
                help="去除皮肤瑕疵"
            )
            params['beauty']['enhance_texture'] = st.slider(
                "质感增强", 0.0, 1.0, 0.0, 0.1, key="enhance_texture",
                help="增强皮肤质感"
            )
        
        # 美妆参数
        with st.sidebar.expander("💄 美妆效果", expanded=False):
            # 唇彩
            st.subheader("💋 唇彩")
            params['makeup']['lipstick_intensity'] = st.slider(
                "唇彩强度", 0.0, 1.0, 0.0, 0.1, key="lipstick_intensity"
            )
            params['makeup']['lipstick_color'] = st.selectbox(
                "唇彩颜色", ['red', 'pink', 'coral', 'berry', 'nude'],
                index=0, key="lipstick_color"
            )
            
            # 眼影
            st.subheader("👁️ 眼影")
            params['makeup']['eyeshadow_intensity'] = st.slider(
                "眼影强度", 0.0, 1.0, 0.0, 0.1, key="eyeshadow_intensity"
            )
            params['makeup']['eyeshadow_color'] = st.selectbox(
                "眼影颜色", ['brown', 'pink', 'blue', 'gold', 'purple'],
                index=0, key="eyeshadow_color"
            )
            
            # 腮红
            st.subheader("🌸 腮红")
            params['makeup']['blush_intensity'] = st.slider(
                "腮红强度", 0.0, 1.0, 0.0, 0.1, key="blush_intensity"
            )
            params['makeup']['blush_color'] = st.selectbox(
                "腮红颜色", ['pink', 'peach', 'coral', 'rose'],
                index=0, key="blush_color"
            )
            
            # 其他
            st.subheader("✏️ 其他")
            params['makeup']['eyebrow_intensity'] = st.slider(
                "眉毛增强", 0.0, 1.0, 0.0, 0.1, key="eyebrow_intensity"
            )
            params['makeup']['eyeliner_intensity'] = st.slider(
                "眼线强度", 0.0, 1.0, 0.0, 0.1, key="eyeliner_intensity"
            )
        
        # 美体参数
        with st.sidebar.expander("🏃‍♀️ 美体调整", expanded=False):
            params['body']['waist_slim'] = st.slider(
                "收腰", 0.0, 1.0, 0.0, 0.1, key="waist_slim",
                help="收缩腰部，打造小蛮腰"
            )
            params['body']['leg_lengthen'] = st.slider(
                "拉腿", 0.0, 1.0, 0.0, 0.1, key="leg_lengthen",
                help="拉长腿部比例"
            )
            params['body']['arm_slim'] = st.slider(
                "瘦手臂", 0.0, 1.0, 0.0, 0.1, key="arm_slim",
                help="收缩手臂轮廓"
            )
            params['body']['shoulder_adjust'] = st.slider(
                "肩膀调整", -1.0, 1.0, 0.0, 0.1, key="shoulder_adjust",
                help="负值变宽，正值变窄"
            )
            params['body']['overall_slim'] = st.slider(
                "整体瘦身", 0.0, 1.0, 0.0, 0.1, key="overall_slim",
                help="整体向中心收缩"
            )
        
        # 预设效果
        st.sidebar.subheader("🎭 预设效果")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("清新自然", key="preset_natural"):
                self.apply_preset('natural')
                st.rerun()  # 修复：使用新的API
        
        with col2:
            if st.button("浓妆艳抹", key="preset_glamour"):
                self.apply_preset('glamour')
                st.rerun()  # 修复：使用新的API
        
        # 重置按钮
        if st.sidebar.button("🔄 重置所有参数", key="reset_all"):
            self.reset_all_params()
            st.rerun()  # 修复：使用新的API
        
        return params
    
    def apply_preset(self, preset_type):
        """应用预设效果"""
        if preset_type == 'natural':
            # 清新自然预设
            presets = {
                'face_slim': 0.3,
                'eye_enlarge': 0.1,
                'mouth_adjust': 0.1,
                'smooth_skin': 0.3,
                'skin_whitening': 0.2,
                'lipstick_intensity': 0.3,
                'lipstick_color': 'coral',
                'blush_intensity': 0.2,
                'blush_color': 'pink'
            }
        elif preset_type == 'glamour':
            # 浓妆艳抹预设
            presets = {
                'face_slim': 0.5,
                'nose_lift': 0.3,
                'eye_enlarge': 0.2,
                'mouth_adjust': 0.2,
                'smooth_skin': 0.5,
                'skin_whitening': 0.4,
                'lipstick_intensity': 0.7,
                'lipstick_color': 'red',
                'eyeshadow_intensity': 0.5,
                'eyeshadow_color': 'brown',
                'blush_intensity': 0.4,
                'blush_color': 'rose',
                'eyeliner_intensity': 0.6
            }
        
        # 更新session state
        for key, value in presets.items():
            if key in st.session_state:
                st.session_state[key] = value
    
    def reset_all_params(self):
        """重置所有参数"""
        reset_keys = [
            'face_slim', 'nose_lift', 'eye_enlarge', 'mouth_adjust',
            'smooth_skin', 'skin_whitening', 'skin_warmth', 'skin_saturation',
            'remove_blemishes', 'enhance_texture',
            'lipstick_intensity', 'eyeshadow_intensity', 'blush_intensity',
            'eyebrow_intensity', 'eyeliner_intensity',
            'waist_slim', 'leg_lengthen', 'arm_slim', 'shoulder_adjust', 'overall_slim'
        ]
        
        for key in reset_keys:
            if key in st.session_state:
                st.session_state[key] = 0.0
        
        # 重置选择框
        reset_selects = {
            'smooth_method': 'bilateral',
            'lipstick_color': 'red',
            'eyeshadow_color': 'brown',
            'blush_color': 'pink'
        }
        
        for key, default_value in reset_selects.items():
            if key in st.session_state:
                st.session_state[key] = default_value
    
    def create_main_interface(self):
        """创建主界面"""
        st.title("🎨 智能美颜编辑器")
        
        # 显示版本信息
        if self.controlnet_available:
            st.markdown("**ControlNet版** - 使用最新AI技术实现自然美形效果！")
        else:
            st.markdown("**传统版** - 基于几何变形的美颜处理")
        
        # 显示模块状态
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if self.face_reshaper is not None:
                if self.controlnet_available:
                    st.success("🚀 ControlNet: 已加载")
                else:
                    st.success("📋 传统美形: 已加载")
            else:
                st.error("❌ 美形: 未加载")
        
        with col2:
            if self.face_beauty is not None:
                st.success("✨ 美颜: 已加载")
            else:
                st.warning("⚠️ 美颜: 未加载")
        
        with col3:
            if self.face_makeup is not None:
                st.success("💄 美妆: 已加载")
            else:
                st.warning("⚠️ 美妆: 未加载")
        
        with col4:
            if self.body_reshaper is not None:
                st.success("🏃‍♀️ 美体: 已加载")
            else:
                st.warning("⚠️ 美体: 未加载")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择图片文件", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="支持 JPG, JPEG, PNG, BMP 格式，建议分辨率800x800以下"
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
                        st.image(pil_original, use_column_width=True)  # 修复：使用正确的API
                
                with col2:
                    st.subheader("✨ 处理结果")
                    pil_processed = self.cv2_to_pil(st.session_state.processed_image)
                    if pil_processed is not None:
                        st.image(pil_processed, use_column_width=True)  # 修复：使用正确的API
                
                # 显示处理时间和统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ 处理时间", f"{st.session_state.processing_time:.2f}s")
                with col2:
                    active_params = sum(1 for category in params.values() 
                                      for v in category.values() 
                                      if isinstance(v, (int, float)) and v > 0)
                    st.metric("🎛️ 活跃参数", active_params)
                with col3:
                    if self.controlnet_available:
                        st.metric("🚀 美形引擎", "ControlNet")
                    elif self.face_reshaper is not None:
                        st.metric("📋 美形引擎", "传统算法")
                    else:
                        st.metric("⚠️ 美形引擎", "不可用")
                
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
                            file_name=f"beauty_result_{int(time.time())}.jpg",
                            mime="image/jpeg"
                        )
            else:
                # 首次加载时显示原图
                if st.session_state.original_image is not None:
                    st.subheader("📷 上传的图片")
                    pil_original = self.cv2_to_pil(st.session_state.original_image)
                    if pil_original is not None:
                        st.image(pil_original, use_column_width=True)
                    st.info("👆 请在左侧调整参数开始美颜处理")
        
        else:
            st.info("👆 请先上传一张图片开始美颜编辑")
            
            # 显示功能说明
            self._show_feature_description()
    
    def _show_feature_description(self):
        """显示功能说明"""
        if self.controlnet_available:
            st.markdown("""
            ### 🚀 ControlNet版本特色
            
            **✨ 美形调整 (ControlNet技术)**
            - 🆕 MediaPipe关键点检测: 468个精确面部关键点
            - 🆕 ControlNet重建: 基于语义理解的自然变形
            - 🆕 关键点预览: 实时查看调整效果
            - 🆕 几何约束: 确保面部结构合理性
            - 🆕 质量验证: 防止过度变形和失真
            
            **💫 美颜效果**
            - 磨皮: 平滑皮肤纹理
            - 美白: 提亮肤色
            - 肤色调整: 调整暖度和饱和度
            
            **💄 美妆效果**
            - 唇彩: 多种颜色可选
            - 眼影: 丰富的色彩选择
            - 腮红: 增添红润气色
            - 眉毛和眼线: 精致五官
            
            **🏃‍♀️ 美体调整**
            - 收腰: 打造小蛮腰
            - 拉腿: 让腿部更修长
            - 瘦手臂: 告别拜拜肉
            
            ---
            
            **🔧 技术架构:**
            - ✅ MediaPipe Face Mesh (468关键点)
            - ✅ ControlNet MediaPipe Face模型
            - ✅ Stable Diffusion 1.5
            - ✅ 几何约束验证
            - ✅ 实时关键点预览
            
            **📋 系统要求:**
            - Python 3.8+
            - CUDA GPU (推荐16GB显存)
            - 依赖: torch, diffusers, mediapipe, opencv-python
            """)
        else:
            st.markdown("""
            ### 📋 传统版本功能
            
            **✨ 美形调整 (几何变形)**
            - 📐 Moving Least Squares变形
            - 🎯 精确关键点控制
            - ⚖️ 自适应强度调整
            - 🛡️ 多层约束保护
            
            **💫 美颜效果**
            - 磨皮: 双边滤波/频率分离
            - 美白: LAB色彩空间调整
            - 肤色调整: HSV色调控制
            - 去瑕疵: 形态学处理
            
            **💄 美妆效果**
            - 唇彩: 颜色混合算法
            - 眼影: 区域蒙版处理
            - 腮红: 自然晕染效果
            - 眉毛眼线: 精确绘制
            
            **🏃‍♀️ 美体调整**
            - 网格变形: 基于关键点的身体调整
            - 比例优化: 智能拉伸算法
            - 轮廓修饰: 自然过渡处理
            
            ---
            
            **📋 系统要求:**
            - Python 3.7+
            - 依赖: opencv-python, mediapipe, scipy
            """)
        
        # 显示使用提示
        st.markdown("""
        ### 💡 使用提示
        
        1. **图片要求**: 建议使用正脸照片，分辨率不超过800x800
        2. **参数调节**: 建议从小参数开始，逐步增加强度
        3. **效果预览**: 每次调整参数都会实时更新结果
        4. **预设效果**: 可以使用"清新自然"或"浓妆艳抹"快速应用
        5. **参数重置**: 使用"重置所有参数"按钮清空所有设置
        
        ### ⚠️ 注意事项
        
        - 过高的参数可能导致不自然的效果
        - ControlNet版本需要较多显存，建议使用GPU
        - 处理大图片可能需要较长时间
        - 建议在良好光照条件下拍摄的照片效果更佳
        """)
    
    def run(self):
        """运行应用"""
        # 设置页面配置
        st.set_page_config(
            page_title="智能美颜编辑器",
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
        </style>
        """, unsafe_allow_html=True)
        
        # 创建主界面
        self.create_main_interface()
        
        # 在侧边栏底部显示版本信息
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"""
        **版本信息:**
        - 美形引擎: {'ControlNet' if self.controlnet_available else '传统算法' if self.face_reshaper else '不可用'}
        - 美颜模块: {'✅' if self.face_beauty else '❌'}
        - 美妆模块: {'✅' if self.face_makeup else '❌'}
        - 美体模块: {'✅' if self.body_reshaper else '❌'}
        """)

def main():
    """主函数"""
    try:
        app = BeautyApp()
        app.run()
    except Exception as e:
        st.error(f"应用启动失败: {e}")
        st.info("请检查模块文件是否存在并且依赖库已正确安装")
        
        # 显示调试信息
        with st.expander("调试信息"):
            st.code(f"""
错误类型: {type(e).__name__}
错误信息: {str(e)}

所需文件:
- face_reshape.py (传统美形)
- controlnet_face_reshape.py (ControlNet美形，可选)
- face_beauty.py (美颜功能)
- face_makeup.py (美妆功能)  
- body_reshape.py (美体功能)

所需依赖:
- streamlit
- opencv-python
- mediapipe
- numpy
- pillow
- scipy
- torch (ControlNet版本)
- diffusers (ControlNet版本)
            """)

if __name__ == "__main__":
    main()
