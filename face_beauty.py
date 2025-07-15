#!/usr/bin/env python3
"""
美颜脚本 - 基于传统图像处理的实时人脸美颜
包含磨皮、美白、肤色调整、去瑕疵等功能
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.ndimage import gaussian_filter
import argparse

class FaceBeauty:
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
        
        # 面部区域关键点索引
        self.face_region_indices = [
            # 面部轮廓
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
            # 额头区域
            9, 10, 151, 337, 299, 333, 298, 301, 368, 264, 447, 366,
            401, 435, 410, 454, 103, 67, 109, 10, 151, 9, 10, 151,
            # 脸颊区域
            116, 117, 118, 119, 120, 121, 128, 126, 142, 36, 205, 206,
            207, 213, 192, 147, 187, 207, 213, 192, 147, 187, 207, 213,
            345, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410,
            # 鼻子区域
            1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48,
            115, 131, 134, 102, 49, 220, 305, 292, 284, 281, 275, 266,
            # 嘴唇周围
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            12, 15, 16, 17, 18, 200, 199, 175, 0, 13, 269, 270, 267,
            271, 272, 12, 15, 16, 17, 18, 200, 199, 175, 0, 13, 269
        ]
    
    def detect_face_region(self, image):
        """检测人脸区域并创建掩码"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # 提取关键点坐标
        landmarks = []
        h, w = image.shape[:2]
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
        
        landmarks = np.array(landmarks)
        
        # 创建面部掩码
        face_points = landmarks[self.face_region_indices]
        
        # 使用凸包创建面部区域
        hull = cv2.convexHull(face_points)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        
        # 膨胀掩码以确保覆盖完整
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 羽化边缘
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return landmarks, mask
    
    def bilateral_smooth(self, image, mask, intensity=0.5):
        """双边滤波磨皮"""
        # 根据强度调整参数
        d = int(9 + intensity * 6)  # 邻域直径
        sigma_color = 20 + intensity * 60  # 颜色标准差
        sigma_space = 20 + intensity * 60  # 空间标准差
        
        # 多次双边滤波
        smooth = image.copy()
        iterations = max(1, int(intensity * 3))
        
        for _ in range(iterations):
            smooth = cv2.bilateralFilter(smooth, d, sigma_color, sigma_space)
        
        # 使用掩码混合
        mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
        result = image * (1 - mask_3d) + smooth * mask_3d
        
        return result.astype(np.uint8)
    
    def frequency_separation_smooth(self, image, mask, intensity=0.5):
        """频率分离磨皮（高级磨皮）"""
        # 转换为浮点数
        img_float = image.astype(np.float32) / 255.0
        
        # 高斯模糊获得低频信息
        blur_radius = max(3, int(intensity * 15))
        low_freq = cv2.GaussianBlur(img_float, (blur_radius, blur_radius), 0)
        
        # 高频信息 = 原图 - 低频
        high_freq = img_float - low_freq
        
        # 对高频信息进行轻微模糊（减少瑕疵）
        high_freq_smooth = cv2.GaussianBlur(high_freq, (3, 3), 0)
        
        # 重新组合
        result = low_freq + high_freq_smooth * (1 - intensity * 0.7)
        
        # 使用掩码混合
        mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
        final_result = image.astype(np.float32) / 255.0 * (1 - mask_3d) + result * mask_3d
        
        return (np.clip(final_result, 0, 1) * 255).astype(np.uint8)
    
    def skin_whitening(self, image, mask, intensity=0.3):
        """肌肤美白"""
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 提亮L通道
        l_float = l.astype(np.float32)
        l_enhanced = l_float + intensity * 30
        l_enhanced = np.clip(l_enhanced, 0, 255)
        
        # 合并通道
        lab_enhanced = cv2.merge([l_enhanced.astype(np.uint8), a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 使用掩码混合
        mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
        final_result = image * (1 - mask_3d) + result * mask_3d
        
        return final_result.astype(np.uint8)
    
    def skin_tone_adjustment(self, image, mask, warmth=0.0, saturation=0.0):
        """肤色调整"""
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 调整色调（暖色调）
        h_float = h.astype(np.float32)
        h_adjusted = h_float + warmth * 10
        h_adjusted = np.clip(h_adjusted, 0, 179)
        
        # 调整饱和度
        s_float = s.astype(np.float32)
        s_adjusted = s_float * (1 + saturation * 0.5)
        s_adjusted = np.clip(s_adjusted, 0, 255)
        
        # 合并通道
        hsv_adjusted = cv2.merge([h_adjusted.astype(np.uint8), 
                                 s_adjusted.astype(np.uint8), v])
        result = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
        
        # 使用掩码混合
        mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
        final_result = image * (1 - mask_3d) + result * mask_3d
        
        return final_result.astype(np.uint8)
    
    def remove_blemishes(self, image, mask, intensity=0.5):
        """去瑕疵"""
        # 使用开运算去除小的瑕疵
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 转换到LAB空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 对L通道进行形态学操作
        l_smooth = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        
        # 混合原始和处理后的L通道
        l_result = l * (1 - intensity) + l_smooth * intensity
        
        # 合并通道
        lab_result = cv2.merge([l_result.astype(np.uint8), a, b])
        result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
        
        # 使用掩码混合
        mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
        final_result = image * (1 - mask_3d) + result * mask_3d
        
        return final_result.astype(np.uint8)
    
    def enhance_skin_texture(self, image, mask, intensity=0.3):
        """肌肤质感增强"""
        # 使用非锐化掩蔽增强细节
        gaussian = cv2.GaussianBlur(image, (5, 5), 0)
        unsharp_mask = cv2.addWeighted(image, 1 + intensity, gaussian, -intensity, 0)
        
        # 使用掩码混合
        mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
        result = image * (1 - mask_3d) + unsharp_mask * mask_3d
        
        return result.astype(np.uint8)
    
    def apply_beauty_filter(self, image, params):
        """主要美颜函数"""
        # 检测人脸区域
        landmarks, mask = self.detect_face_region(image)
        if landmarks is None or mask is None:
            return image
        
        result = image.copy()
        
        # 磨皮
        if params.get('smooth_skin', 0) > 0:
            smooth_method = params.get('smooth_method', 'bilateral')
            if smooth_method == 'bilateral':
                result = self.bilateral_smooth(result, mask, params['smooth_skin'])
            else:
                result = self.frequency_separation_smooth(result, mask, params['smooth_skin'])
        
        # 美白
        if params.get('skin_whitening', 0) > 0:
            result = self.skin_whitening(result, mask, params['skin_whitening'])
        
        # 肤色调整
        warmth = params.get('skin_warmth', 0)
        saturation = params.get('skin_saturation', 0)
        if warmth != 0 or saturation != 0:
            result = self.skin_tone_adjustment(result, mask, warmth, saturation)
        
        # 去瑕疵
        if params.get('remove_blemishes', 0) > 0:
            result = self.remove_blemishes(result, mask, params['remove_blemishes'])
        
        # 质感增强
        if params.get('enhance_texture', 0) > 0:
            result = self.enhance_skin_texture(result, mask, params['enhance_texture'])
        
        return result

def main():
    parser = argparse.ArgumentParser(description='人脸美颜工具')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--smooth_skin', type=float, default=0.5, help='磨皮强度 (0-1)')
    parser.add_argument('--smooth_method', choices=['bilateral', 'frequency'], 
                       default='bilateral', help='磨皮方法')
    parser.add_argument('--skin_whitening', type=float, default=0.3, help='美白强度 (0-1)')
    parser.add_argument('--skin_warmth', type=float, default=0.0, help='肤色暖度 (-1到1)')
    parser.add_argument('--skin_saturation', type=float, default=0.0, help='肤色饱和度 (-1到1)')
    parser.add_argument('--remove_blemishes', type=float, default=0.3, help='去瑕疵强度 (0-1)')
    parser.add_argument('--enhance_texture', type=float, default=0.2, help='质感增强强度 (0-1)')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"无法加载图片: {args.input}")
        return
    
    # 创建美颜处理器
    beauty = FaceBeauty()
    
    # 设置参数
    params = {
        'smooth_skin': args.smooth_skin,
        'smooth_method': args.smooth_method,
        'skin_whitening': args.skin_whitening,
        'skin_warmth': args.skin_warmth,
        'skin_saturation': args.skin_saturation,
        'remove_blemishes': args.remove_blemishes,
        'enhance_texture': args.enhance_texture
    }
    
    # 执行美颜
    result = beauty.apply_beauty_filter(image, params)
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"美颜完成! 结果保存到: {args.output}")

if __name__ == "__main__":
    main()
