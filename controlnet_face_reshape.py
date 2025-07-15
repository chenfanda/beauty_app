#!/usr/bin/env python3
"""
ControlNet 人脸美形脚本 - 基于标准脸型分类的智能提示词系统
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import argparse
import os

class ControlNetFaceReshaper:
    def __init__(self, model_path="./models/controlnet_mediapipe_face", sd_model_path="./models/stable-diffusion"):
        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        
        # 面部关键点
        self.face_regions = {
            'face_contour': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 292, 284],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324],
            'left_cheek': [116, 117, 118, 119, 120, 121, 234],
            'right_cheek': [345, 346, 347, 348, 349, 350, 454]
        }
        
        # 黄金比例标准
        self.golden_ratios = {
            'face_width_height': 0.75,
            'eye_distance_face_width': 0.46,
            'nose_width_mouth_width': 0.67,
            'lower_face_total_face': 0.55
        }
        
        # 加载模型
        self.load_models(model_path, sd_model_path)
        
        # 初始化CodeFormer
        self.load_codeformer()
    
    def load_codeformer(self):
        """加载CodeFormer模型"""
        try:
            import onnxruntime as ort
            
            codeformer_path = "./models/CodeFormer/codeformer.onnx"
            
            if not os.path.exists(codeformer_path):
                print(f"⚠️ CodeFormer模型文件不存在: {codeformer_path}")
                self.codeformer_session = None
                return
            
            print("📥 加载 CodeFormer 模型...")
            self.codeformer_session = ort.InferenceSession(
                codeformer_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            print("✅ CodeFormer 加载完成")
            
        except ImportError:
            print("⚠️ onnxruntime 未安装，CodeFormer功能不可用")
            self.codeformer_session = None
        except Exception as e:
            print(f"⚠️ CodeFormer 加载失败: {e}")
            self.codeformer_session = None
    
    def load_models(self, model_path, sd_model_path):
        """加载ControlNet和Stable Diffusion模型"""
        try:
            print("📥 加载 ControlNet 模型...")
            self.controlnet = ControlNetModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=True
            )
            print("✅ ControlNet 加载完成")
            
            print("📥 加载 Stable Diffusion Pipeline...")
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                os.path.join(sd_model_path, 'v2-1_768-ema-pruned.safetensors'),
                controlnet=self.controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,
            )
            print("✅ Pipeline 加载完成")
            
            print("⚙️ 配置调度器...")
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            print(f"📱 移动到 {self.device}...")
            self.pipe = self.pipe.to(self.device)
            
            # GPU内存优化
            if self.device == "cuda":
                print("🚀 启用GPU优化...")
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    self.pipe.enable_model_cpu_offload()
                    print("✅ GPU优化完成")
                except Exception as e:
                    print(f"⚠️ 优化设置失败: {e}")
            
            print("🎉 模型初始化完成!")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.pipe = None
            self.controlnet = None
    
    def extract_landmarks(self, image):
        """提取关键点 - 输入必须是numpy数组(BGR格式)"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = []
            h, w = image.shape[:2]
            for landmark in results.multi_face_landmarks[0].landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            
            return np.array(landmarks)
            
        except Exception as e:
            print(f"⚠️ 关键点提取失败: {e}")
            return None
    
    def calculate_face_ratios(self, landmarks):
        """计算当前面部比例"""
        try:
            # 脸宽高比
            face_left = np.min(landmarks[:, 0])
            face_right = np.max(landmarks[:, 0])
            face_top = np.min(landmarks[:, 1])
            face_bottom = np.max(landmarks[:, 1])
            
            face_width = face_right - face_left
            face_height = face_bottom - face_top
            face_width_height = face_width / face_height if face_height > 0 else 0.75
            
            # 双眼间距/脸宽
            left_eye_center = np.mean(landmarks[self.face_regions['left_eye']], axis=0)
            right_eye_center = np.mean(landmarks[self.face_regions['right_eye']], axis=0)
            eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
            eye_distance_face_width = eye_distance / face_width if face_width > 0 else 0.46
            
            # 鼻宽/嘴宽
            nose_width = abs(landmarks[234][0] - landmarks[454][0]) * 0.3
            mouth_width = abs(landmarks[61][0] - landmarks[291][0]) if 291 < len(landmarks) else nose_width / 0.67
            nose_width_mouth_width = nose_width / mouth_width if mouth_width > 0 else 0.67
            
            # 下半脸/全脸
            nose_tip_y = landmarks[1][1]
            lower_face_total_face = (face_bottom - nose_tip_y) / face_height if face_height > 0 else 0.55
            
            return {
                'face_width_height': face_width_height,
                'eye_distance_face_width': eye_distance_face_width,
                'nose_width_mouth_width': nose_width_mouth_width,
                'lower_face_total_face': lower_face_total_face
            }
        
        except Exception as e:
            print(f"⚠️ 比例计算失败: {e}")
            return self.golden_ratios
    
    def apply_golden_ratio_constraints(self, landmarks, params):
        """应用黄金比例约束"""
        current_ratios = self.calculate_face_ratios(landmarks)
        constrained_params = {}
        
        for param_name, value in params.items():
            if param_name == 'face_slim' and value > 0:
                current_width_height = current_ratios['face_width_height']
                reduction_factor = 1 - (value * 0.1)
                predicted_ratio = current_width_height * reduction_factor
                min_allowed_ratio = self.golden_ratios['face_width_height'] * 0.85
                
                if predicted_ratio < min_allowed_ratio:
                    max_reduction = 1 - (min_allowed_ratio / current_width_height)
                    max_allowed_slim = max_reduction / 0.1
                    constrained_params[param_name] = min(value, max(0, max_allowed_slim))
                    print(f"⚠️ 瘦脸强度受黄金比例约束，从{value:.2f}调整为{constrained_params[param_name]:.2f}")
                else:
                    constrained_params[param_name] = value
            else:
                constrained_params[param_name] = value
        
        return constrained_params
    
    def apply_face_slim_adjustment(self, landmarks, intensity=0.3):
        """瘦脸调整"""
        adjusted_landmarks = landmarks.copy()
        face_center_x = landmarks[1][0]
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        max_shrink = min(face_width * 0.08, 12)
        
        key_points = [234, 454, 172, 397]
        
        for idx in key_points:
            if idx >= len(landmarks):
                continue
            
            point = landmarks[idx]
            dist_to_center = abs(point[0] - face_center_x)
            
            if dist_to_center > 15:
                direction = 1 if point[0] > face_center_x else -1
                distance_factor = min(dist_to_center / (face_width * 0.3), 1.0)
                shrink_amount = intensity * max_shrink * distance_factor * 0.6
                new_x = point[0] - direction * shrink_amount
                adjusted_landmarks[idx][0] = int(max(0, min(new_x, landmarks.shape[0] - 1)))
        
        return adjusted_landmarks
    
    def apply_eye_enlargement(self, landmarks, intensity=0.2):
        """眼部放大"""
        adjusted_landmarks = landmarks.copy()
        
        for eye_region in ['left_eye', 'right_eye']:
            eye_indices = self.face_regions[eye_region]
            eye_points = landmarks[eye_indices]
            eye_center = np.mean(eye_points, axis=0)
            
            for idx in eye_indices:
                if idx >= len(landmarks):
                    continue
                point = landmarks[idx]
                direction = point - eye_center
                
                if abs(direction[0]) > abs(direction[1]):
                    scale_factor = 1 + intensity * 0.5
                else:
                    scale_factor = 1 + intensity * 0.3
                
                enlarged_point = eye_center + direction * scale_factor
                adjusted_landmarks[idx] = enlarged_point.astype(int)
        
        return adjusted_landmarks
    
    def apply_nose_adjustment(self, landmarks, intensity=0.3):
        """鼻部调整"""
        adjusted_landmarks = landmarks.copy()
        nose_bridge_indices = [6, 19, 20, 94, 125, 141, 235, 236]
        
        for idx in nose_bridge_indices:
            if idx >= len(landmarks):
                continue
            
            point = landmarks[idx]
            nose_center_x = landmarks[1][0]
            distance_from_center = abs(point[0] - nose_center_x)
            lift_factor = max(0, 1 - distance_from_center / 8)
            lift_amount = intensity * 3 * lift_factor
            adjusted_landmarks[idx][1] -= int(lift_amount)
        
        return adjusted_landmarks
    
    def apply_mouth_adjustment(self, landmarks, intensity=0.2):
        """嘴部调整"""
        adjusted_landmarks = landmarks.copy()
        mouth_indices = self.face_regions['mouth']
        mouth_points = landmarks[mouth_indices]
        mouth_center = np.mean(mouth_points, axis=0)
        
        for idx in mouth_indices:
            if idx >= len(landmarks):
                continue
            point = landmarks[idx]
            direction = point - mouth_center
            adjusted_point = mouth_center + direction * (1 + intensity * 0.3)
            adjusted_landmarks[idx] = adjusted_point.astype(int)
        
        return adjusted_landmarks
    
    def create_landmark_image(self, landmarks, height, width):
        """创建关键点图像"""
        landmark_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 绘制面部轮廓
        contour_indices = self.face_regions['face_contour']
        contour_points = landmarks[contour_indices]
        cv2.polylines(landmark_image, [contour_points], True, (255, 255, 255), 2)
        
        # 绘制眼部轮廓
        for eye_region in ['left_eye', 'right_eye']:
            eye_indices = self.face_regions[eye_region]
            eye_points = landmarks[eye_indices]
            cv2.polylines(landmark_image, [eye_points], True, (192, 192, 192), 1)
        
        # 绘制鼻部和嘴部轮廓
        for region in ['nose', 'mouth']:
            region_indices = self.face_regions[region]
            region_points = landmarks[region_indices]
            hull = cv2.convexHull(region_points)
            cv2.polylines(landmark_image, [hull], True, (128, 128, 128), 1)
        
        # 绘制关键点
        for point in landmarks:
            cv2.circle(landmark_image, tuple(point), 1, (64, 64, 64), -1)
        
        return landmark_image
    
    def generate_advanced_prompt(self, params):
        """基于标准脸型分类的高级提示词系统"""
        
        # 基础质量提示词
        base_quality = [
            "professional photography",
            "high resolution",
            "sharp focus",
            "natural lighting",
            "photorealistic",
            "detailed skin texture",
            "film grain"
        ]
        
        # 根据调整参数生成具体的脸型描述
        facial_features = []
        
        # 标准脸型分类
        if params.get('face_slim', 0) > 0:
            intensity = params['face_slim']
            if intensity >= 0.7:
                facial_features.append("pointed chin face")
            elif intensity >= 0.5:
                facial_features.append("oval face shape")
            elif intensity >= 0.3:
                facial_features.append("oblong face shape")
            elif intensity >= 0.1:
                facial_features.append("square face shape")
            else:
                facial_features.append("round face shape")
        
        # 标准眼型分类
        if params.get('eye_enlarge', 0) > 0:
            intensity = params['eye_enlarge']
            if intensity >= 0.4:
                facial_features.append("wide-set eyes")
            elif intensity >= 0.3:
                facial_features.append("almond eyes")
            elif intensity >= 0.2:
                facial_features.append("round eyes")
            elif intensity >= 0.1:
                facial_features.append("monolid eyes")
            else:
                facial_features.append("double eyelid eyes")
        
        # 标准鼻型分类
        if params.get('nose_lift', 0) > 0:
            intensity = params['nose_lift']
            if intensity >= 0.6:
                facial_features.append("aquiline nose")
            elif intensity >= 0.4:
                facial_features.append("straight nose")
            elif intensity >= 0.3:
                facial_features.append("Roman nose")
            elif intensity >= 0.2:
                facial_features.append("snub nose")
            else:
                facial_features.append("button nose")
        
        # 标准唇型分类
        if params.get('mouth_adjust', 0) > 0:
            intensity = params['mouth_adjust']
            if intensity >= 0.4:
                facial_features.append("full lips")
            elif intensity >= 0.3:
                facial_features.append("Cupid's bow lips")
            elif intensity >= 0.2:
                facial_features.append("wide mouth")
            elif intensity >= 0.1:
                facial_features.append("thin lips")
            else:
                facial_features.append("small mouth")
        
        # 面部轮廓细节描述
        contour_details = []
        total_face_adjustment = params.get('face_slim', 0)
        if total_face_adjustment > 0.5:
            contour_details.extend([
                "prominent cheekbones",
                "defined jawline",
                "angular facial structure"
            ])
        elif total_face_adjustment > 0.3:
            contour_details.extend([
                "visible cheekbones",
                "tapered jaw",
                "structured facial contours"
            ])
        elif total_face_adjustment > 0.1:
            contour_details.extend([
                "soft cheekbones",
                "gentle jaw curve",
                "balanced facial structure"
            ])
        
        # 美学比例描述
        aesthetic_terms = [
            "golden ratio facial proportions",
            "symmetrical features",
            "classical facial aesthetics",
            "harmonious facial geometry"
        ]
        
        # 皮肤和质感
        texture_terms = [
            "natural skin texture",
            "healthy skin tone",
            "realistic skin details",
            "natural skin luminosity"
        ]
        
        # 摄影技术描述
        photography_terms = [
            "professional portrait lighting",
            "natural shadow placement",
            "authentic light reflection",
            "studio quality headshot"
        ]
        
        # 组合所有描述
        all_prompts = []
        all_prompts.extend(base_quality)
        all_prompts.extend(facial_features)
        all_prompts.extend(contour_details[:2])
        all_prompts.extend(aesthetic_terms[:2])
        all_prompts.extend(texture_terms[:2])
        all_prompts.extend(photography_terms[:2])
        
        return ", ".join(all_prompts)
    
    def generate_advanced_negative_prompt(self, params):
        """生成高级负面提示词"""
        
        # 基础负面词
        base_negative = [
            "blurry", "out of focus", "low quality", "low resolution",
            "pixelated", "jpeg artifacts", "compression artifacts"
        ]
        
        # 面部缺陷
        facial_defects = [
            "asymmetrical face", "crooked features", "unnatural proportions",
            "distorted facial features", "malformed face", "weird facial structure",
            "abnormal face shape", "deformed eyes", "uneven eyes"
        ]
        
        # AI生成痕迹
        ai_artifacts = [
            "artificial looking", "computer generated appearance",
            "digital art style", "cartoon style", "anime style",
            "painting style", "illustration style", "rendered look",
            "synthetic appearance", "fake looking"
        ]
        
        # 皮肤问题
        skin_issues = [
            "plastic skin", "waxy skin", "unrealistic skin",
            "overly smooth skin", "artificial skin texture",
            "doll-like skin", "overprocessed skin"
        ]
        
        # 身份变化相关
        identity_issues = [
            "different person", "changed identity", "face swap",
            "different facial structure", "unrecognizable"
        ]
        
        # 根据调整强度动态添加
        dynamic_negative = []
        total_intensity = sum([
            params.get('face_slim', 0),
            params.get('eye_enlarge', 0),
            params.get('nose_lift', 0),
            params.get('mouth_adjust', 0)
        ])
        
        if total_intensity > 1.0:
            dynamic_negative.extend([
                "over-edited", "heavily modified", "unnatural enhancement",
                "excessive editing", "overdone effects"
            ])
        
        # 组合负面提示词
        all_negative = []
        all_negative.extend(base_negative)
        all_negative.extend(facial_defects[:6])
        all_negative.extend(ai_artifacts[:5])
        all_negative.extend(skin_issues[:4])
        all_negative.extend(identity_issues)
        all_negative.extend(dynamic_negative)
        
        return ", ".join(all_negative)
    
    def enhance_face_realism_with_codeformer(self, image):
        """使用CodeFormer增强人脸真实感"""
        if self.codeformer_session is None:
            return image
        
        try:
            # 预处理
            h, w = image.shape[:2]
            
            # CodeFormer通常需要512x512输入
            resized = cv2.resize(image, (512, 512))
            
            # 转换为RGB并归一化
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_tensor = rgb_image.astype(np.float32) / 255.0
            
            # 调整维度顺序：HWC -> CHW
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
            
            # 添加batch维度
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # ONNX推理
            w_value = np.array(0.95 , dtype=np.float64)  # 修复质量权重，0.5是平衡值
            outputs = self.codeformer_session.run(None, {
                'x': input_tensor,  # 输入图像
                'w': w_value        # 修复权重
            })
            
            # 后处理
            output_tensor = outputs[0][0]  # 移除batch维度
            output_tensor = np.transpose(output_tensor, (1, 2, 0))  # CHW -> HWC
            output_tensor = np.clip(output_tensor * 255, 0, 255).astype(np.uint8)
            
            # 转换回BGR
            result_bgr = cv2.cvtColor(output_tensor, cv2.COLOR_RGB2BGR)
            
            # 调整回原尺寸
            result_resized = cv2.resize(result_bgr, (w, h))
            
            return result_resized
            
        except Exception as e:
            print(f"⚠️ CodeFormer处理失败: {e}")
            return image
    
    def blend_with_original_for_realism(self, original, enhanced, blend_ratio=0.8):
        """与原图混合以保持真实感"""
        try:
            # 确保图像尺寸一致
            if original.shape != enhanced.shape:
                enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
            
            # 加权混合
            result = cv2.addWeighted(enhanced, blend_ratio, original, 1 - blend_ratio, 0)
            
            return result
            
        except Exception as e:
            print(f"⚠️ 图像混合失败: {e}")
            return enhanced
    
    def reshape_face(self, image, params):
        """美形处理主函数 - 输入输出都是numpy数组(BGR格式)"""
        if self.pipe is None:
            return image
        
        try:
            # 前端传入的是numpy数组(BGR格式)
            height, width = image.shape[:2]
            
            # 提取关键点
            landmarks = self.extract_landmarks(image)
            if landmarks is None:
                print("⚠️ 未检测到人脸关键点")
                return image
            
            # 应用黄金比例约束
            constrained_params = self.apply_golden_ratio_constraints(landmarks, params)
            
            # 调整关键点
            adjusted_landmarks = landmarks.copy()
            
            if constrained_params.get('face_slim', 0) > 0:
                adjusted_landmarks = self.apply_face_slim_adjustment(adjusted_landmarks, constrained_params['face_slim'])
            
            if constrained_params.get('eye_enlarge', 0) > 0:
                adjusted_landmarks = self.apply_eye_enlargement(adjusted_landmarks, constrained_params['eye_enlarge'])
            
            if constrained_params.get('nose_lift', 0) > 0:
                adjusted_landmarks = self.apply_nose_adjustment(adjusted_landmarks, constrained_params['nose_lift'])
            
            if constrained_params.get('mouth_adjust', 0) > 0:
                adjusted_landmarks = self.apply_mouth_adjustment(adjusted_landmarks, constrained_params['mouth_adjust'])
            
            # 创建控制图像
            control_image = self.create_landmark_image(adjusted_landmarks, height, width)
            control_pil = Image.fromarray(control_image)
            
            # 转换为PIL用于ControlNet
            input_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 生成高级提示词
            positive_prompt = self.generate_advanced_prompt(constrained_params)
            negative_prompt = self.generate_advanced_negative_prompt(constrained_params)
            
            # ControlNet生成
            result = self.pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=input_pil,
                control_image=control_pil,
                strength=0.25,
                num_inference_steps=20,
                guidance_scale=8.0,
                controlnet_conditioning_scale=0.6,
                height=768,
                width=768,
                eta=0.0,
                generator=torch.manual_seed(42)
            ).images[0]
            
            # 转换回numpy数组(BGR格式)返回给前端
            result_np = np.array(result)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            
            # 调整回原图尺寸
            result_resized = cv2.resize(result_bgr, (width, height))
            
            # 使用CodeFormer增强真实感
            # if self.codeformer_session is not None:
            #     print("🎨 使用CodeFormer增强真实感...")
            #     realistic_result = self.enhance_face_realism_with_codeformer(result_resized)
                
            #     # 与原图适度混合，保持一些原始特征
            #     final_result = self.blend_with_original_for_realism(
            #         original=image, 
            #         enhanced=realistic_result, 
            #         blend_ratio=0.75  # 75%增强结果，25%原图
            #     )
                
            #     print("✅ 真实感增强完成")
            #     return final_result
            # else:
            #     print("⚠️ CodeFormer不可用，返回原始ControlNet结果")
            return result_resized
            
        except Exception as e:
            print(f"❌ ControlNet生成失败: {e}")
            import traceback
            traceback.print_exc()
            return image

def main():
    parser = argparse.ArgumentParser(description='ControlNet人脸美形工具')
    parser.add_argument('--input', '-i', required=True, help='输入图片')
    parser.add_argument('--output', '-o', required=True, help='输出图片')
    parser.add_argument('--face_slim', type=float, default=0.2, help='瘦脸强度 (0-0.5)')
    parser.add_argument('--eye_enlarge', type=float, default=0.1, help='眼部放大 (0-0.3)')
    parser.add_argument('--nose_lift', type=float, default=0.1, help='鼻梁增高 (0-0.3)')
    parser.add_argument('--mouth_adjust', type=float, default=0.05, help='嘴型调整 (0-0.2)')
    parser.add_argument('--disable_codeformer', action='store_true', help='禁用CodeFormer真实感增强')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"❌ 无法加载图片: {args.input}")
        return
    
    print(f"📷 处理图片: {image.shape}")
    
    # 创建美形器
    reshaper = ControlNetFaceReshaper()
    
    # 如果用户禁用CodeFormer
    if args.disable_codeformer:
        reshaper.codeformer_session = None
        print("⚠️ 用户禁用了CodeFormer真实感增强")
    
    # 设置参数
    params = {
        'face_slim': max(0, min(0.5, args.face_slim)),
        'eye_enlarge': max(0, min(0.3, args.eye_enlarge)),
        'nose_lift': max(0, min(0.3, args.nose_lift)),
        'mouth_adjust': max(0, min(0.2, args.mouth_adjust))
    }
    
    print(f"🎯 美形参数: {params}")
    
    # 执行美形
    print("🚀 开始美形处理...")
    result = reshaper.reshape_face(image, params)
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"✅ 美形完成! 结果保存到: {args.output}")

if __name__ == "__main__":
    main()
