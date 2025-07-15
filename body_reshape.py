#!/usr/bin/env python3
"""
美体脚本 - 基于传统图像处理的身材调整
包含瘦身、拉腿、收腰、丰胸等功能
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import argparse

class BodyReshaper:
    def __init__(self):
        # 初始化MediaPipe姿态检测
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 身体关键点索引定义（MediaPipe Pose 33个关键点）
        self.body_landmarks = {
            # 上半身
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            
            # 躯干
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            
            # 核心点
            'nose': 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10
        }
    
    def detect_body_landmarks(self, image):
        """检测身体关键点"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return None, None
        
        # 提取关键点坐标
        landmarks = []
        h, w = image.shape[:2]
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
        
        # 获取分割掩码
        segmentation_mask = None
        if results.segmentation_mask is not None:
            segmentation_mask = (results.segmentation_mask * 255).astype(np.uint8)
        
        return np.array(landmarks), segmentation_mask
    
    def create_mesh_grid(self, image_shape, grid_size=20):
        """创建网格用于变形"""
        h, w = image_shape[:2]
        
        # 创建规则网格
        x = np.linspace(0, w-1, grid_size)
        y = np.linspace(0, h-1, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        src_points = np.column_stack([xx.ravel(), yy.ravel()])
        dst_points = src_points.copy()
        
        return src_points, dst_points, (grid_size, grid_size)
    
    def apply_mesh_warp(self, image, src_points, dst_points):
        """应用网格变形"""
        h, w = image.shape[:2]
        
        # 创建映射
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        # 网格坐标
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        # 使用插值计算新坐标
        try:
            new_x = griddata(dst_points, src_points[:, 0], grid_points, method='cubic', fill_value=0)
            new_y = griddata(dst_points, src_points[:, 1], grid_points, method='cubic', fill_value=0)
            
            map_x = new_x.reshape(h, w)
            map_y = new_y.reshape(h, w)
            
            # 边界处理
            map_x = np.clip(map_x, 0, w-1)
            map_y = np.clip(map_y, 0, h-1)
            
            # 应用变形
            result = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            
        except Exception as e:
            print(f"网格变形失败: {e}")
            result = image
        
        return result
    
    def get_body_region_points(self, landmarks, region):
        """获取身体特定区域的关键点"""
        if region == 'waist':
            # 腰部区域
            left_hip = landmarks[self.body_landmarks['left_hip']]
            right_hip = landmarks[self.body_landmarks['right_hip']]
            left_shoulder = landmarks[self.body_landmarks['left_shoulder']]
            right_shoulder = landmarks[self.body_landmarks['right_shoulder']]
            
            # 计算腰部位置（肩膀和臀部中间）
            waist_y = int((left_hip[1] + right_hip[1] + left_shoulder[1] + right_shoulder[1]) / 4)
            waist_left = [int((left_hip[0] + left_shoulder[0]) / 2), waist_y]
            waist_right = [int((right_hip[0] + right_shoulder[0]) / 2), waist_y]
            
            return np.array([waist_left, waist_right])
        
        elif region == 'legs':
            # 腿部区域
            return np.array([
                landmarks[self.body_landmarks['left_hip']],
                landmarks[self.body_landmarks['right_hip']],
                landmarks[self.body_landmarks['left_knee']],
                landmarks[self.body_landmarks['right_knee']],
                landmarks[self.body_landmarks['left_ankle']],
                landmarks[self.body_landmarks['right_ankle']]
            ])
        
        elif region == 'arms':
            # 手臂区域
            return np.array([
                landmarks[self.body_landmarks['left_shoulder']],
                landmarks[self.body_landmarks['right_shoulder']],
                landmarks[self.body_landmarks['left_elbow']],
                landmarks[self.body_landmarks['right_elbow']],
                landmarks[self.body_landmarks['left_wrist']],
                landmarks[self.body_landmarks['right_wrist']]
            ])
        
        return None
    
    def apply_waist_slim(self, image, landmarks, intensity=0.3):
        """收腰效果"""
        try:
            waist_points = self.get_body_region_points(landmarks, 'waist')
            if waist_points is None or len(waist_points) < 2:
                return image
            
            src_points, dst_points, grid_shape = self.create_mesh_grid(image.shape)
            
            # 计算腰部中心
            waist_center = np.mean(waist_points, axis=0)
            waist_width = abs(waist_points[1][0] - waist_points[0][0])
            
            # 收腰变形
            for i, point in enumerate(src_points):
                # 计算距离腰部的距离
                dist_to_waist = abs(point[1] - waist_center[1])
                
                # 在腰部区域进行收缩
                if dist_to_waist < waist_width * 0.8:
                    # 水平方向向中心收缩
                    if point[0] < waist_center[0]:  # 左侧
                        shrink_amount = intensity * 20 * (1 - dist_to_waist / (waist_width * 0.8))
                        dst_points[i][0] = point[0] + shrink_amount
                    elif point[0] > waist_center[0]:  # 右侧
                        shrink_amount = intensity * 20 * (1 - dist_to_waist / (waist_width * 0.8))
                        dst_points[i][0] = point[0] - shrink_amount
            
            return self.apply_mesh_warp(image, src_points, dst_points)
        
        except Exception as e:
            print(f"收腰效果失败: {e}")
            return image
    
    def apply_leg_lengthen(self, image, landmarks, intensity=0.2):
        """拉腿效果"""
        try:
            leg_points = self.get_body_region_points(landmarks, 'legs')
            if leg_points is None:
                return image
            
            src_points, dst_points, grid_shape = self.create_mesh_grid(image.shape)
            
            # 获取腿部关键位置
            hip_y = max(leg_points[0][1], leg_points[1][1])  # 臀部Y坐标
            ankle_y = min(leg_points[4][1], leg_points[5][1])  # 脚踝Y坐标
            leg_length = ankle_y - hip_y
            
            if leg_length <= 0:
                return image
            
            # 拉腿变形
            stretch_amount = intensity * leg_length * 0.3
            
            for i, point in enumerate(src_points):
                # 只对腿部区域进行拉伸
                if point[1] > hip_y:
                    # 计算在腿部的相对位置
                    leg_ratio = (point[1] - hip_y) / leg_length
                    leg_ratio = min(leg_ratio, 1.0)
                    
                    # Y方向拉伸
                    dst_points[i][1] = point[1] + stretch_amount * leg_ratio
            
            return self.apply_mesh_warp(image, src_points, dst_points)
        
        except Exception as e:
            print(f"拉腿效果失败: {e}")
            return image
    
    def apply_arm_slim(self, image, landmarks, intensity=0.3):
        """瘦手臂效果"""
        try:
            arm_points = self.get_body_region_points(landmarks, 'arms')
            if arm_points is None:
                return image
            
            src_points, dst_points, grid_shape = self.create_mesh_grid(image.shape)
            
            # 计算手臂区域
            left_arm_line = [arm_points[0], arm_points[2], arm_points[4]]  # 左肩-左肘-左腕
            right_arm_line = [arm_points[1], arm_points[3], arm_points[5]]  # 右肩-右肘-右腕
            
            def slim_arm_region(arm_line, side_multiplier):
                for arm_segment in range(len(arm_line) - 1):
                    start_point = arm_line[arm_segment]
                    end_point = arm_line[arm_segment + 1]
                    
                    # 计算手臂方向向量
                    arm_vector = end_point - start_point
                    arm_length = np.linalg.norm(arm_vector)
                    
                    if arm_length == 0:
                        continue
                    
                    # 垂直于手臂的方向（用于收缩）
                    perp_vector = np.array([-arm_vector[1], arm_vector[0]])
                    perp_vector = perp_vector / np.linalg.norm(perp_vector)
                    
                    # 对该手臂段周围的网格点进行变形
                    for i, point in enumerate(src_points):
                        # 计算点到手臂线段的距离
                        t = np.dot(point - start_point, arm_vector) / (arm_length ** 2)
                        t = np.clip(t, 0, 1)
                        
                        closest_point = start_point + t * arm_vector
                        dist_to_arm = np.linalg.norm(point - closest_point)
                        
                        # 在手臂附近进行收缩
                        if dist_to_arm < 50:  # 50像素范围内
                            shrink_factor = intensity * (1 - dist_to_arm / 50) * 10
                            
                            # 判断在手臂的哪一侧
                            side_check = np.dot(point - closest_point, perp_vector)
                            
                            if side_check * side_multiplier > 0:  # 外侧
                                dst_points[i] = point - perp_vector * shrink_factor
            
            # 瘦左臂（外侧向内收缩）
            slim_arm_region(left_arm_line, 1)
            
            # 瘦右臂（外侧向内收缩）
            slim_arm_region(right_arm_line, -1)
            
            return self.apply_mesh_warp(image, src_points, dst_points)
        
        except Exception as e:
            print(f"瘦手臂效果失败: {e}")
            return image
    
    def apply_shoulder_adjust(self, image, landmarks, intensity=0.2):
        """肩膀调整（使肩膀更窄或更宽）"""
        try:
            left_shoulder = landmarks[self.body_landmarks['left_shoulder']]
            right_shoulder = landmarks[self.body_landmarks['right_shoulder']]
            
            src_points, dst_points, grid_shape = self.create_mesh_grid(image.shape)
            
            # 计算肩膀中心和宽度
            shoulder_center = (left_shoulder + right_shoulder) / 2
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            
            # 肩膀调整（intensity > 0 窄肩，< 0 宽肩）
            for i, point in enumerate(src_points):
                # 在肩膀区域
                if abs(point[1] - shoulder_center[1]) < 50:
                    if point[0] < shoulder_center[0]:  # 左侧
                        adjust_amount = intensity * 15
                        dst_points[i][0] = point[0] + adjust_amount
                    elif point[0] > shoulder_center[0]:  # 右侧
                        adjust_amount = intensity * 15
                        dst_points[i][0] = point[0] - adjust_amount
            
            return self.apply_mesh_warp(image, src_points, dst_points)
        
        except Exception as e:
            print(f"肩膀调整失败: {e}")
            return image
    
    def apply_overall_slim(self, image, landmarks, intensity=0.2):
        """整体瘦身效果"""
        try:
            # 找到身体的边界
            body_points = np.array([
                landmarks[self.body_landmarks['left_shoulder']],
                landmarks[self.body_landmarks['right_shoulder']],
                landmarks[self.body_landmarks['left_hip']],
                landmarks[self.body_landmarks['right_hip']]
            ])
            
            # 计算身体中心线
            body_center_x = np.mean(body_points[:, 0])
            
            src_points, dst_points, grid_shape = self.create_mesh_grid(image.shape)
            
            # 整体向中心收缩
            for i, point in enumerate(src_points):
                # 计算距离身体中心线的距离
                dist_to_center = abs(point[0] - body_center_x)
                
                # 根据距离进行收缩
                if dist_to_center > 10:  # 避免过度收缩中心
                    shrink_amount = intensity * 8
                    
                    if point[0] < body_center_x:  # 左侧
                        dst_points[i][0] = point[0] + shrink_amount
                    else:  # 右侧
                        dst_points[i][0] = point[0] - shrink_amount
            
            return self.apply_mesh_warp(image, src_points, dst_points)
        
        except Exception as e:
            print(f"整体瘦身失败: {e}")
            return image
    
    def reshape_body(self, image, params):
        """主要美体函数"""
        landmarks, segmentation_mask = self.detect_body_landmarks(image)
        if landmarks is None:
            return image
        
        result = image.copy()
        
        # 应用各种美体效果
        if params.get('waist_slim', 0) > 0:
            result = self.apply_waist_slim(result, landmarks, params['waist_slim'])
        
        if params.get('leg_lengthen', 0) > 0:
            result = self.apply_leg_lengthen(result, landmarks, params['leg_lengthen'])
        
        if params.get('arm_slim', 0) > 0:
            result = self.apply_arm_slim(result, landmarks, params['arm_slim'])
        
        if params.get('shoulder_adjust', 0) != 0:
            result = self.apply_shoulder_adjust(result, landmarks, params['shoulder_adjust'])
        
        if params.get('overall_slim', 0) > 0:
            result = self.apply_overall_slim(result, landmarks, params['overall_slim'])
        
        return result

def main():
    parser = argparse.ArgumentParser(description='身材美体工具')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--waist_slim', type=float, default=0.0, help='收腰强度 (0-1)')
    parser.add_argument('--leg_lengthen', type=float, default=0.0, help='拉腿强度 (0-1)')
    parser.add_argument('--arm_slim', type=float, default=0.0, help='瘦手臂强度 (0-1)')
    parser.add_argument('--shoulder_adjust', type=float, default=0.0, help='肩膀调整 (-1到1, 负值变宽，正值变窄)')
    parser.add_argument('--overall_slim', type=float, default=0.0, help='整体瘦身强度 (0-1)')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"无法加载图片: {args.input}")
        return
    
    # 创建美体处理器
    reshaper = BodyReshaper()
    
    # 设置参数
    params = {
        'waist_slim': args.waist_slim,
        'leg_lengthen': args.leg_lengthen,
        'arm_slim': args.arm_slim,
        'shoulder_adjust': args.shoulder_adjust,
        'overall_slim': args.overall_slim
    }
    
    # 执行美体
    result = reshaper.reshape_body(image, params)
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"美体完成! 结果保存到: {args.output}")

if __name__ == "__main__":
    main()
