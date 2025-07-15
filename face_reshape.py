#!/usr/bin/env python3
"""
美形脚本 - 基于传统几何变形的实时人脸美形
采用Thin Plate Spline (TPS) 变形算法
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import cdist
import argparse

class FaceReshaper:
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
        
        # 关键点索引 (MediaPipe 468点模型)
        self.key_points = {
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 187, 207, 213, 192, 147],
            'right_cheek': [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427, 436, 416, 376, 411, 427, 436, 416, 376],
            'nose_bridge': [6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 292, 284],
            'nose_tip': [1, 2, 5, 4, 6, 19, 20, 94, 125],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'eyes': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 
                    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
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
    
    def get_region_points(self, landmarks, region):
        """获取特定区域的关键点"""
        if region not in self.key_points:
            return None
        indices = self.key_points[region]
        return landmarks[indices]
    
    def thin_plate_spline_warp(self, image, src_points, dst_points):
        """Thin Plate Spline变形"""
        h, w = image.shape[:2]
        
        # 计算TPS变换矩阵
        def tps_kernel(r):
            return np.where(r == 0, 0, r * r * np.log(r))
        
        n = len(src_points)
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = np.linalg.norm(src_points[i] - src_points[j])
                    K[i, j] = tps_kernel(r)
        
        # 构建线性系统
        P = np.hstack([np.ones((n, 1)), src_points])
        A = np.vstack([
            np.hstack([K, P]),
            np.hstack([P.T, np.zeros((3, 3))])
        ])
        
        # 解线性系统
        b_x = np.hstack([dst_points[:, 0], np.zeros(3)])
        b_y = np.hstack([dst_points[:, 1], np.zeros(3)])
        
        try:
            weights_x = np.linalg.solve(A, b_x)
            weights_y = np.linalg.solve(A, b_y)
        except np.linalg.LinAlgError:
            return image
        
        # 应用变换
        result = np.zeros_like(image)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        # 计算新坐标
        def apply_tps(points, weights):
            new_coords = np.zeros(len(points))
            for i, point in enumerate(points):
                val = weights[-3] + weights[-2] * point[0] + weights[-1] * point[1]
                for j in range(n):
                    r = np.linalg.norm(point - src_points[j])
                    if r > 0:
                        val += weights[j] * tps_kernel(r)
                new_coords[i] = val
            return new_coords
        
        new_x = apply_tps(coords, weights_x).reshape(h, w)
        new_y = apply_tps(coords, weights_y).reshape(h, w)
        
        # 使用双线性插值
        new_x = np.clip(new_x, 0, w-1)
        new_y = np.clip(new_y, 0, h-1)
        
        result = cv2.remap(image, new_x.astype(np.float32), new_y.astype(np.float32), 
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    def apply_face_slim(self, image, landmarks, intensity=0.3):
        """瘦脸效果 - 快速局部收缩算法"""
        if intensity <= 0:
            return image
        
        h, w = image.shape[:2]
        result = image.copy()
        
        # 获取关键点
        try:
            # 脸颊最外侧点
            left_cheek = landmarks[234]   # 左脸颊
            right_cheek = landmarks[454]  # 右脸颊
            nose_tip = landmarks[1]       # 鼻尖
            chin = landmarks[152]         # 下巴
            
            # 计算脸部中心
            face_center_x = nose_tip[0]
            
            # 定义收缩区域
            def apply_local_squeeze(center_point, is_left_side):
                cx, cy = center_point
                radius = int(80 * (1 + intensity))  # 影响半径
                squeeze_strength = intensity * 0.8
                
                # 创建局部变形
                y1, y2 = max(0, cy - radius), min(h, cy + radius)
                x1, x2 = max(0, cx - radius), min(w, cx + radius)
                
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        # 计算到中心的距离
                        dx = x - cx
                        dy = y - cy
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        if distance < radius and distance > 0:
                            # 计算收缩力度（距离越近力度越大）
                            force = (1 - distance / radius) * squeeze_strength
                            
                            # 水平方向收缩
                            if is_left_side and dx < 0:  # 左脸颊向右收缩
                                new_x = int(x + abs(dx) * force)
                            elif not is_left_side and dx > 0:  # 右脸颊向左收缩
                                new_x = int(x - abs(dx) * force)
                            else:
                                continue
                            
                            new_x = max(0, min(w-1, new_x))
                            
                            # 复制像素
                            if 0 <= new_x < w:
                                result[y, new_x] = image[y, x]
            
            # 应用左右脸颊收缩
            apply_local_squeeze(left_cheek, True)
            apply_local_squeeze(right_cheek, False)
            
            return result
            
        except (IndexError, TypeError) as e:
            print(f"瘦脸处理失败: {e}")
            return image
    
    def apply_nose_lift(self, image, landmarks, intensity=0.3):
        """鼻梁增高"""
        nose_points = self.get_region_points(landmarks, 'nose_bridge')
        if nose_points is None:
            return image
        
        src_points = nose_points
        dst_points = src_points.copy()
        
        # 鼻梁中线向上提升
        nose_center = np.mean(src_points, axis=0)
        for i, point in enumerate(src_points):
            # 向上提升
            lift_amount = intensity * 5
            dst_points[i][1] -= lift_amount
        
        return self.thin_plate_spline_warp(image, src_points, dst_points)
    
    def apply_eye_enlarge(self, image, landmarks, intensity=0.2):
        """眼部放大"""
        eye_points = self.get_region_points(landmarks, 'eyes')
        if eye_points is None:
            return image
        
        # 分别处理左右眼
        left_eye = eye_points[:16]
        right_eye = eye_points[16:]
        
        src_points = eye_points
        dst_points = src_points.copy()
        
        # 左眼放大
        left_center = np.mean(left_eye, axis=0)
        for i in range(16):
            direction = src_points[i] - left_center
            dst_points[i] = left_center + direction * (1 + intensity)
        
        # 右眼放大
        right_center = np.mean(right_eye, axis=0)
        for i in range(16, 32):
            direction = src_points[i] - right_center
            dst_points[i] = right_center + direction * (1 + intensity)
        
        return self.thin_plate_spline_warp(image, src_points, dst_points)
    
    def reshape_face(self, image, params):
        """主要美形函数"""
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            return image
        
        result = image.copy()
        
        # 应用各种美形效果
        if params.get('face_slim', 0) > 0:
            result = self.apply_face_slim(result, landmarks, params['face_slim'])
        
        if params.get('nose_lift', 0) > 0:
            result = self.apply_nose_lift(result, landmarks, params['nose_lift'])
        
        if params.get('eye_enlarge', 0) > 0:
            result = self.apply_eye_enlarge(result, landmarks, params['eye_enlarge'])
        
        return result

def main():
    parser = argparse.ArgumentParser(description='人脸美形工具')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--face_slim', type=float, default=0.3, help='瘦脸强度 (0-1)')
    parser.add_argument('--nose_lift', type=float, default=0.2, help='鼻梁增高强度 (0-1)')
    parser.add_argument('--eye_enlarge', type=float, default=0.1, help='眼部放大强度 (0-1)')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"无法加载图片: {args.input}")
        return
    
    # 创建美形处理器
    reshaper = FaceReshaper()
    
    # 设置参数
    params = {
        'face_slim': args.face_slim,
        'nose_lift': args.nose_lift,
        'eye_enlarge': args.eye_enlarge
    }
    
    # 执行美形
    result = reshaper.reshape_face(image, params)
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"美形完成! 结果保存到: {args.output}")

if __name__ == "__main__":
    main()
