#!/usr/bin/env python3
"""
面部精细调整前端应用 - 基于Streamlit
整合六大调整功能，修复MediaPipe资源重复初始化问题
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import sys
import os
import mediapipe as mp

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FacialAdjustmentApp:
    def __init__(self):
        self.modules = {}
        self.load_status = {}
        self.shared_face_mesh = None
        self._load_all_modules()
        self._init_session_state()
    
    def _load_all_modules(self):
        """加载所有面部调整模块 - 使用共享MediaPipe实例"""
        
        # 创建共享MediaPipe实例
        mp_face_mesh = mp.solutions.face_mesh
        try:
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
            return
        
        # 加载各个模块并替换MediaPipe实例
        modules_to_load = [
            ('chin', 'chin_adjust', 'OptimizedChin3DAdjustment', '下巴调整'),
            ('eye_corner', 'eye_corner_3d_complete', 'EyeCornerAdjuster', '眼角调整'),
            ('face_slim', 'face_slim', 'SimpleFaceSlimming', '瘦脸功能'),
            ('forehead', 'forehead_adjust', 'ForeheadAdjustment', '额头调整'),
            ('mouth', 'mouth_adjust', 'Mouth3DAdjustment', '嘴巴调整'),
            ('nose', 'nose_3d', 'Nose3DGeometry', '鼻子调整')
        ]
        
        for module_key, module_file, class_name, display_name in modules_to_load:
            try:
                module = __import__(module_file, fromlist=[class_name])
                class_obj = getattr(module, class_name)
                
                # 直接传入共享MediaPipe实例
                self.modules[module_key] = class_obj(face_mesh=self.shared_face_mesh)
                
                self.load_status[module_key] = {'status': 'success', 'message': f'{display_name}模块加载成功'}
                st.success(f"✅ {display_name}模块加载成功")
            except ImportError as e:
                self.modules[module_key] = None
                self.load_status[module_key] = {'status': 'error', 'message': f'{display_name}模块加载失败: {e}'}
                st.error(f"❌ {display_name}模块加载失败: {e}")
    
    def _init_session_state(self):
        """初始化session state"""
        defaults = {
            'original_image': None,
            'processed_image': None,
            'processing_time': 0,
            'last_params': None,
            'adjustment_history': [],
            'reset_applied': None,
            'reset_values': {}
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def load_image(self, uploaded_file):
        """加载上传的图片"""
        if uploaded_file is None:
            return None
        
        try:
            bytes_data = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(bytes_data))
            image_np = np.array(image)
            
            if len(image_np.shape) == 3:
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            
            # 限制图片大小
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
    
    def get_reset_values(self):
        """获取重置值"""
        return {
            'chin_enabled': False, 'chin_type': 'length', 'chin_intensity': 0.0,
            'eye_enabled': False, 'eye_type': 'stretch', 'eye_intensity': 0.0,
            'slim_enabled': False, 'slim_region': 'both', 'slim_intensity': 0.0,
            'nose_enabled': False, 'nose_type': 'wings', 'nose_intensity': 0.0,
            'mouth_enabled': False, 'mouth_type': 'thickness', 'mouth_intensity': 0.0,
            'forehead_enabled': False, 'forehead_type': 'shrink', 'forehead_intensity': 0.0,
            'eye_auto_optimize': True
        }
    
    def create_sidebar_controls(self):
        """创建侧边栏控制面板"""
        st.sidebar.title("🎨 面部调整控制")
        
        # 检查重置应用
        current_reset = st.session_state.get('reset_applied', None)
        reset_values = st.session_state.get('reset_values', {})
        
        params = {}
        
        # 各功能模块控制
        controls = [
            ('chin', '🎯 下巴调整', {
                'types': ['length', 'width', 'sharp', 'round'],
                'type_names': {'length': '拉长', 'width': '收窄', 'sharp': '尖锐', 'round': '圆润'}
            }),
            ('eye_corner', '👁️ 眼角调整', {
                'types': ['stretch', 'lift', 'shape'],
                'type_names': {'stretch': '拉长', 'lift': '上扬', 'shape': '综合'},
                'has_auto_optimize': True
            }),
            ('face_slim', '🔥 瘦脸调整', {
                'types': ['both', 'left', 'right'],
                'type_names': {'both': '双侧', 'left': '左脸', 'right': '右脸'},
                'param_name': 'region'
            }),
            ('forehead', '🌟 额头调整', {
                'types': ['shrink', 'expand'],
                'type_names': {'shrink': '饱满效果', 'expand': '平整效果'}
            }),
            ('mouth', '💋 嘴巴调整', {
                'types': ['thickness', 'peak', 'corners', 'resize', 'enlarge'],
                'type_names': {
                    'thickness': '唇部增厚', 'peak': '唇峰抬高', 
                    'corners': '嘴角上扬', 'resize': '嘴巴缩小', 'enlarge': '嘴巴放大'
                }
            }),
            ('nose', '👃 鼻子调整', {
                'types': ['wings', 'tip', 'resize', 'bridge'],
                'type_names': {
                    'wings': '鼻翼收缩', 'tip': '鼻尖抬高', 
                    'resize': '整体缩小', 'bridge': '鼻梁变挺'
                }
            })
        ]
        
        for module_key, title, config in controls:
            with st.sidebar.expander(title, expanded=False):
                if self.modules[module_key] is not None:
                    # 获取重置值
                    default_enabled = reset_values.get(f'{module_key}_enabled', False)
                    default_type = reset_values.get(f'{module_key}_type', config['types'][0])
                    default_intensity = reset_values.get(f'{module_key}_intensity', 0.3)
                    
                    # 启用开关
                    params[f'{module_key}_enabled'] = st.checkbox(
                        f"启用{title[2:]}", 
                        value=default_enabled, 
                        key=f"{module_key}_enabled"
                    )
                    
                    if params[f'{module_key}_enabled']:
                        # 类型选择
                        param_name = config.get('param_name', 'type')
                        params[f'{module_key}_{param_name}'] = st.selectbox(
                            "调整类型",
                            config['types'],
                            format_func=lambda x: config['type_names'][x],
                            index=config['types'].index(default_type),
                            key=f"{module_key}_{param_name}"
                        )
                        
                        # 强度调整
                        params[f'{module_key}_intensity'] = st.slider(
                            "调整强度", 
                            0.1, 0.7, 
                            default_intensity, 
                            0.05, 
                            key=f"{module_key}_intensity"
                        )
                        
                        # 眼角特殊参数
                        if config.get('has_auto_optimize'):
                            default_optimize = reset_values.get('eye_auto_optimize', True)
                            params['eye_auto_optimize'] = st.checkbox(
                                "智能优化", 
                                value=default_optimize, 
                                key="eye_auto_optimize"
                            )
                    else:
                        # 未启用时设置默认值
                        param_name = config.get('param_name', 'type')
                        params[f'{module_key}_{param_name}'] = default_type
                        params[f'{module_key}_intensity'] = 0.0
                        if config.get('has_auto_optimize'):
                            params['eye_auto_optimize'] = reset_values.get('eye_auto_optimize', True)
                else:
                    st.error(f"❌ {title[2:]}模块不可用")
                    params[f'{module_key}_enabled'] = False
        
        # 全局设置
        st.sidebar.markdown("---")
        st.sidebar.subheader("🛠️ 全局设置")
        
        # 重置和清理
        if st.sidebar.button("🔄 重置参数", key="reset_all"):
            st.session_state.reset_applied = True
            st.session_state.reset_values = self.get_reset_values()
            st.rerun()
        
        if st.sidebar.button("🧹 清理资源", key="cleanup_resources"):
            self.cleanup_resources()
            st.sidebar.success("资源清理完成")
        
        # 清除重置标记
        if current_reset:
            st.session_state.reset_applied = None
            st.session_state.reset_values = {}
        
        return params
    
    def cleanup_resources(self):
        """清理所有资源"""
        try:
            if self.shared_face_mesh:
                self.shared_face_mesh.close()
                self.shared_face_mesh = None
            
            st.session_state.original_image = None
            st.session_state.processed_image = None
            
            import gc
            gc.collect()
            
            print("🧹 资源清理完成")
        except Exception as e:
            print(f"⚠️ 资源清理异常: {e}")
    
    def process_image(self, image, params):
        """处理图片"""
        if image is None:
            return None
        
        start_time = time.time()
        result = image.copy()
        processing_steps = []
        
        try:
            # 处理顺序：瘦脸 -> 下巴 -> 鼻子 -> 眼角 -> 嘴巴 -> 额头
            process_order = [
                ('face_slim', 'face_slimming', ['intensity', 'region']),
                ('chin', 'process_image', ['type', 'intensity']),
                ('nose', 'process_image', ['type', 'intensity']),
                ('eye_corner', 'process_image', ['type', 'intensity', 'auto_optimize']),
                ('mouth', 'process_image', ['type', 'intensity']),
                ('forehead', 'process_image', ['type', 'intensity'])
            ]
            
            for module_key, method_name, param_keys in process_order:
                if params.get(f'{module_key}_enabled', False) and self.modules[module_key] is not None:
                    try:
                        # 构建参数
                        kwargs = {}
                        for param_key in param_keys:
                            if param_key == 'type':
                                # 处理不同模块的type参数名称
                                if module_key == 'face_slim':
                                    kwargs['region'] = params.get(f'{module_key}_region', 'both')
                                elif module_key == 'chin':
                                    kwargs['transform_type'] = params.get(f'{module_key}_type', 'length')
                                elif module_key == 'eye_corner':
                                    kwargs['adjustment_type'] = params.get(f'{module_key}_type', 'stretch')
                                else:
                                    kwargs['transform_type'] = params.get(f'{module_key}_type', 'wings')
                            elif param_key == 'region':
                                kwargs['region'] = params.get(f'{module_key}_region', 'both')
                            elif param_key == 'intensity':
                                kwargs['intensity'] = params.get(f'{module_key}_intensity', 0.3)
                            elif param_key == 'auto_optimize':
                                kwargs['auto_optimize'] = params.get('eye_auto_optimize', True)
                        
                        # 调用处理方法
                        method = getattr(self.modules[module_key], method_name)
                        result = method(result, **kwargs)
                        
                        # 记录处理步骤
                        intensity = kwargs.get('intensity', 0)
                        type_or_region = kwargs.get('region') or kwargs.get('transform_type') or kwargs.get('adjustment_type')
                        processing_steps.append(f"{module_key}({type_or_region}, {intensity:.2f})")
                        
                    except Exception as e:
                        st.warning(f"{module_key}处理出错: {e}")
        
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
        st.markdown("**专业级面部调整工具** - 整合6大核心调整功能")
        
        # 显示模块状态
        col1, col2, col3 = st.columns(3)
        with col1:
            chin_status = "✅" if self.modules['chin'] else "❌"
            eye_status = "✅" if self.modules['eye_corner'] else "❌"
            st.metric("下巴调整", chin_status)
            st.metric("眼角调整", eye_status)
        
        with col2:
            slim_status = "✅" if self.modules['face_slim'] else "❌"
            forehead_status = "✅" if self.modules['forehead'] else "❌"
            st.metric("瘦脸功能", slim_status)
            st.metric("额头调整", forehead_status)
        
        with col3:
            mouth_status = "✅" if self.modules['mouth'] else "❌"
            nose_status = "✅" if self.modules['nose'] else "❌"
            st.metric("嘴巴调整", mouth_status)
            st.metric("鼻子调整", nose_status)
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择图片文件", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="支持 JPG, JPEG, PNG, BMP 格式"
        )
        
        if uploaded_file is not None:
            # 加载图片
            if st.session_state.original_image is None:
                with st.spinner("正在加载图片..."):
                    st.session_state.original_image = self.load_image(uploaded_file)
            
            # 获取参数
            params = self.create_sidebar_controls()
            
            # 检查参数变化
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
                
                # 显示统计信息
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
            
            else:
                # 首次加载显示原图
                if st.session_state.original_image is not None:
                    st.subheader("📷 上传的图片")
                    pil_original = self.cv2_to_pil(st.session_state.original_image)
                    if pil_original is not None:
                        st.image(pil_original, use_column_width=True)
                    st.info("👆 请在左侧调整参数开始精细调整")
        
        else:
            st.info("👆 请先上传一张图片开始面部精细调整")
    
    def run(self):
        """运行应用"""
        st.set_page_config(
            page_title="面部精细调整系统",
            page_icon="🎨",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.create_main_interface()
        
        # 侧边栏系统信息
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 系统状态")
        
        loaded_count = sum(1 for module in self.modules.values() if module is not None)
        st.sidebar.success(f"✅ 已加载: {loaded_count}/6")
        
        mediapipe_status = "✅ 正常" if self.shared_face_mesh else "❌ 异常"
        st.sidebar.info(f"🔧 MediaPipe: {mediapipe_status}")

def main():
    """主函数"""
    try:
        app = FacialAdjustmentApp()
        app.run()
    except Exception as e:
        st.error(f"应用启动失败: {e}")
        st.info("请检查所有模块文件是否存在并且依赖库已正确安装")

if __name__ == "__main__":
    main()
