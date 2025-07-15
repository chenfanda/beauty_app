#!/usr/bin/env python3
"""
美妆脚本 - 基于传统图像处理的实时美妆效果
包含唇彩、腮红、眉毛、眼影、眼线等功能
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import ConvexHull
import argparse

class FaceMakeup:
    def __init__(self):
        # 初始化MediaPipe人脸检测
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 面部关键区域索引定义
        self.region_indices = {
            # 嘴唇区域
            'upper_lip': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'lower_lip': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415],
            'lip_contour': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 
                           308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82],
            
            # 眼部区域
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_eyebrow': [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305, 292, 284, 283, 282],
            'right_eyebrow': [276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
            
            # 脸颊区域
            'left_cheek': [116, 117, 118, 119, 120, 121, 128, 126, 142, 36, 205, 206, 207, 213, 192],
            'right_cheek': [345, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410, 454, 323, 361],
            
            # 鼻影区域
            'nose_bridge': [6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134],
            'nose_tip': [1, 2, 5, 4, 6, 19, 20]
        }
        
        # 预定义颜色
        self.colors = {
            'lipstick': {
                'red': (0, 0, 255),
                'pink': (180, 105, 255),
                'coral': (80, 127, 255),
                'berry': (75, 0, 130),
                'nude': (173, 158, 144)
            },
            'eyeshadow': {
                'brown': (42, 42, 165),
                'pink': (147, 20, 255),
                'blue': (255, 0, 0),
                'gold': (0, 215, 255),
                'purple': (128, 0, 128)
            },
            'blush': {
                'pink': (180, 105, 255),
                'peach': (153, 102, 255),
                'coral': (80, 127, 255),
                'rose': (143, 143, 188)
            }
        }
    
    def detect_landmarks(self, image):
        """检测人脸关键点"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # 提取关键点坐标
        landmarks = []
        h, w = image.shape[:2]
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
        
        return np.array(landmarks)
    
    def create_region_mask(self, image_shape, points, blur_radius=5):
        """创建区域掩码"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(points) > 2:
            # 使用凸包创建区域
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], 255)
            
            # 高斯模糊实现羽化效果
            if blur_radius > 0:
                mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), 0)
        
        return mask
    
    def apply_color_blend(self, image, mask, color, intensity=0.5, blend_mode='normal'):
        """应用颜色混合"""
        # 创建颜色层
        color_layer = np.full_like(image, color, dtype=np.uint8)
        
        # 归一化掩码
        mask_normalized = mask.astype(np.float32) / 255.0
        intensity_mask = mask_normalized * intensity
        
        # 扩展掩码到3通道
        mask_3d = np.stack([intensity_mask, intensity_mask, intensity_mask], axis=2)
        
        if blend_mode == 'normal':
            # 普通混合
            result = image * (1 - mask_3d) + color_layer * mask_3d
        elif blend_mode == 'multiply':
            # 正片叠底
            normalized_img = image.astype(np.float32) / 255.0
            normalized_color = color_layer.astype(np.float32) / 255.0
            multiplied = normalized_img * normalized_color
            result = image * (1 - mask_3d) + (multiplied * 255) * mask_3d
        elif blend_mode == 'overlay':
            # 叠加模式
            normalized_img = image.astype(np.float32) / 255.0
            normalized_color = color_layer.astype(np.float32) / 255.0
            
            overlay = np.where(normalized_img < 0.5,
                              2 * normalized_img * normalized_color,
                              1 - 2 * (1 - normalized_img) * (1 - normalized_color))
            result = image * (1 - mask_3d) + (overlay * 255) * mask_3d
        else:
            result = image
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_lipstick(self, image, landmarks, color_name='red', intensity=0.6):
        """应用唇彩"""
        if color_name not in self.colors['lipstick']:
            color_name = 'red'
        
        color = self.colors['lipstick'][color_name]
        
        # 获取嘴唇区域点
        upper_lip_points = landmarks[self.region_indices['upper_lip']]
        lower_lip_points = landmarks[self.region_indices['lower_lip']]
        
        # 创建上下唇掩码
        upper_mask = self.create_region_mask(image.shape, upper_lip_points, blur_radius=3)
        lower_mask = self.create_region_mask(image.shape, lower_lip_points, blur_radius=3)
        
        # 合并嘴唇掩码
        lip_mask = cv2.bitwise_or(upper_mask, lower_mask)
        
        # 应用唇彩效果
        result = self.apply_color_blend(image, lip_mask, color, intensity, 'multiply')
        
        return result
    
    def apply_eyeshadow(self, image, landmarks, color_name='brown', intensity=0.4):
        """应用眼影"""
        if color_name not in self.colors['eyeshadow']:
            color_name = 'brown'
        
        color = self.colors['eyeshadow'][color_name]
        
        # 获取眼部区域
        left_eye_points = landmarks[self.region_indices['left_eye']]
        right_eye_points = landmarks[self.region_indices['right_eye']]
        
        # 扩展眼影区域（向上扩展）
        def expand_eye_region(eye_points):
            center = np.mean(eye_points, axis=0)
            expanded_points = []
            
            for point in eye_points:
                # 向上和向外扩展
                direction = point - center
                expanded_point = point + direction * 0.3
                expanded_point[1] -= 15  # 向上扩展
                expanded_points.append(expanded_point)
            
            return np.array(expanded_points, dtype=np.int32)
        
        left_shadow_points = expand_eye_region(left_eye_points)
        right_shadow_points = expand_eye_region(right_eye_points)
        
        # 创建眼影掩码
        left_shadow_mask = self.create_region_mask(image.shape, left_shadow_points, blur_radius=8)
        right_shadow_mask = self.create_region_mask(image.shape, right_shadow_points, blur_radius=8)
        
        # 应用眼影
        result = image.copy()
        result = self.apply_color_blend(result, left_shadow_mask, color, intensity, 'multiply')
        result = self.apply_color_blend(result, right_shadow_mask, color, intensity, 'multiply')
        
        return result
    
    def apply_blush(self, image, landmarks, color_name='pink', intensity=0.3):
        """应用腮红"""
        if color_name not in self.colors['blush']:
            color_name = 'pink'
        
        color = self.colors['blush'][color_name]
        
        # 获取脸颊区域
        left_cheek_points = landmarks[self.region_indices['left_cheek']]
        right_cheek_points = landmarks[self.region_indices['right_cheek']]
        
        # 创建腮红掩码（使用更大的羽化半径）
        left_blush_mask = self.create_region_mask(image.shape, left_cheek_points, blur_radius=15)
        right_blush_mask = self.create_region_mask(image.shape, right_cheek_points, blur_radius=15)
        
        # 应用腮红
        result = image.copy()
        result = self.apply_color_blend(result, left_blush_mask, color, intensity, 'overlay')
        result = self.apply_color_blend(result, right_blush_mask, color, intensity, 'overlay')
        
        return result
    
    def apply_eyebrow_enhancement(self, image, landmarks, intensity=0.4):
        """眉毛增强"""
        # 获取眉毛区域
        left_eyebrow_points = landmarks[self.region_indices['left_eyebrow']]
        right_eyebrow_points = landmarks[self.region_indices['right_eyebrow']]
        
        # 深棕色
        brown_color = (42, 42, 88)
        
        # 创建眉毛掩码
        left_eyebrow_mask = self.create_region_mask(image.shape, left_eyebrow_points, blur_radius=2)
        right_eyebrow_mask = self.create_region_mask(image.shape, right_eyebrow_points, blur_radius=2)
        
        # 应用眉毛增强
        result = image.copy()
        result = self.apply_color_blend(result, left_eyebrow_mask, brown_color, intensity, 'multiply')
        result = self.apply_color_blend(result, right_eyebrow_mask, brown_color, intensity, 'multiply')
        
        return result
    
    def apply_eyeliner(self, image, landmarks, intensity=0.6):
        """应用眼线"""
        # 获取眼部轮廓点
        left_eye_points = landmarks[self.region_indices['left_eye']]
        right_eye_points = landmarks[self.region_indices['right_eye']]
        
        black_color = (0, 0, 0)
        
        # 创建眼线掩码（更细的线条）
        def create_eyeliner_mask(eye_points):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # 上眼线
            upper_points = eye_points[:8]  # 上眼睑点
            for i in range(len(upper_points) - 1):
                cv2.line(mask, tuple(upper_points[i]), tuple(upper_points[i+1]), 255, 2)
            
            # 羽化
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            return mask
        
        left_eyeliner_mask = create_eyeliner_mask(left_eye_points)
        right_eyeliner_mask = create_eyeliner_mask(right_eye_points)
        
        # 应用眼线
        result = image.copy()
        result = self.apply_color_blend(result, left_eyeliner_mask, black_color, intensity, 'multiply')
        result = self.apply_color_blend(result, right_eyeliner_mask, black_color, intensity, 'multiply')
        
        return result
    
    def apply_makeup(self, image, params):
        """主要美妆函数"""
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            return image
        
        result = image.copy()
        
        # 应用各种美妆效果
        if params.get('lipstick_intensity', 0) > 0:
            lipstick_color = params.get('lipstick_color', 'red')
            result = self.apply_lipstick(result, landmarks, lipstick_color, 
                                       params['lipstick_intensity'])
        
        if params.get('eyeshadow_intensity', 0) > 0:
            eyeshadow_color = params.get('eyeshadow_color', 'brown')
            result = self.apply_eyeshadow(result, landmarks, eyeshadow_color,
                                        params['eyeshadow_intensity'])
        
        if params.get('blush_intensity', 0) > 0:
            blush_color = params.get('blush_color', 'pink')
            result = self.apply_blush(result, landmarks, blush_color,
                                    params['blush_intensity'])
        
        if params.get('eyebrow_intensity', 0) > 0:
            result = self.apply_eyebrow_enhancement(result, landmarks,
                                                  params['eyebrow_intensity'])
        
        if params.get('eyeliner_intensity', 0) > 0:
            result = self.apply_eyeliner(result, landmarks,
                                       params['eyeliner_intensity'])
        
        return result

def main():
    parser = argparse.ArgumentParser(description='人脸美妆工具')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    
    # 唇彩参数
    parser.add_argument('--lipstick_intensity', type=float, default=0.6, help='唇彩强度 (0-1)')
    parser.add_argument('--lipstick_color', choices=['red', 'pink', 'coral', 'berry', 'nude'],
                       default='red', help='唇彩颜色')
    
    # 眼影参数
    parser.add_argument('--eyeshadow_intensity', type=float, default=0.4, help='眼影强度 (0-1)')
    parser.add_argument('--eyeshadow_color', choices=['brown', 'pink', 'blue', 'gold', 'purple'],
                       default='brown', help='眼影颜色')
    
    # 腮红参数
    parser.add_argument('--blush_intensity', type=float, default=0.3, help='腮红强度 (0-1)')
    parser.add_argument('--blush_color', choices=['pink', 'peach', 'coral', 'rose'],
                       default='pink', help='腮红颜色')
    
    # 其他参数
    parser.add_argument('--eyebrow_intensity', type=float, default=0.4, help='眉毛增强强度 (0-1)')
    parser.add_argument('--eyeliner_intensity', type=float, default=0.6, help='眼线强度 (0-1)')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"无法加载图片: {args.input}")
        return
    
    # 创建美妆处理器
    makeup = FaceMakeup()
    
    # 设置参数
    params = {
        'lipstick_intensity': args.lipstick_intensity,
        'lipstick_color': args.lipstick_color,
        'eyeshadow_intensity': args.eyeshadow_intensity,
        'eyeshadow_color': args.eyeshadow_color,
        'blush_intensity': args.blush_intensity,
        'blush_color': args.blush_color,
        'eyebrow_intensity': args.eyebrow_intensity,
        'eyeliner_intensity': args.eyeliner_intensity
    }
    
    # 执行美妆
    result = makeup.apply_makeup(image, params)
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"美妆完成! 结果保存到: {args.output}")
    
    # 显示可用颜色
    print("\n可用颜色选项:")
    print("唇彩颜色:", list(makeup.colors['lipstick'].keys()))
    print("眼影颜色:", list(makeup.colors['eyeshadow'].keys()))
    print("腮红颜色:", list(makeup.colors['blush'].keys()))

if __name__ == "__main__":
    main()
