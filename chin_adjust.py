#!/usr/bin/env python3
"""
优化的5层架构3D下巴调整脚本 - chin_adjust_optimized.py
解决边缘扭曲问题，实现更自然的变形效果
基于5层点位架构：硬锚点→弱锚点→动态锚点→过渡层→主控变形点
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
from typing import List, Tuple, Optional

class OptimizedChin3DAdjustment:
    def __init__(self):
        # MediaPipe设置
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # 相机参数
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 🎯 优化的5层架构点位系统
        # 第1层：硬锚点（绝对稳定）- 脸型基准框架
        self.HARD_ANCHORS = [10, 338, 297]  # 额头中点和两侧颧骨点
        
        # 第2层：弱锚点（微跟随稳定）- 中脸区域防护
        self.WEAK_ANCHORS = [117, 123, 147, 346, 352, 376]  # 脸颊区域稳定点
        
        # 第3层：动态锚点（任务自适应）- 根据变形类型和姿态动态启用
        self.DYNAMIC_ANCHORS = [234, 454, 93, 323]  # 下颌起始点
        
        # 第4层：过渡层（渐变跟随）- 锚点到变形点的平滑过渡
        self.TRANSITION_LAYER = [132, 58, 288, 361]  # 下颌线中段
        
        # 第5层：主控变形点（完全控制）- 核心变形区域
        self.CHIN_CENTER_SYSTEM = [152, 176, 400, 378]  # 152主导，其他支撑
        self.OTHER_CONTOUR = [172, 136, 150, 149, 148, 377, 365, 379, 397, 205, 200, 424, 427]
        
        # 标准3D面部模型关键点
        self.face_model_3d = np.array([
            [0.0, 0.0, 0.0],           # 鼻尖 (1)
            [0.0, -330.0, -65.0],      # 下巴 (152)  
            [-225.0, 170.0, -135.0],   # 左眼外角 (33)
            [225.0, 170.0, -135.0],    # 右眼外角 (263)
            [-150.0, -150.0, -125.0],  # 左嘴角 (61)
            [150.0, -150.0, -125.0],   # 右嘴角 (291)
        ], dtype=np.float32)
        
        # 对应的MediaPipe索引
        self.pnp_indices = [1, 152, 33, 263, 61, 291]
    
    def setup_camera(self, image_shape):
        """设置相机参数"""
        h, w = image_shape[:2]
        focal_length = w
        center = (w/2, h/2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4,1), dtype=np.float32)
    
    def detect_landmarks_3d(self, image):
        """检测面部关键点并计算3D姿态"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None, None
        
        landmarks_2d = []
        h, w = image.shape[:2]
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_2d.append([x, y])
        
        landmarks_2d = np.array(landmarks_2d, dtype=np.float32)
        
        if self.camera_matrix is None:
            self.setup_camera(image.shape)
        
        # 提取用于PnP的关键点
        image_points = []
        for idx in self.pnp_indices:
            if idx < len(landmarks_2d):
                image_points.append(landmarks_2d[idx])
        
        if len(image_points) < 6:
            print("PnP关键点不足")
            return landmarks_2d, None
        
        image_points = np.array(image_points, dtype=np.float32)
        
        # PnP求解3D姿态
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.face_model_3d, image_points, 
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                print("PnP求解失败")
                return landmarks_2d, None
            
            return landmarks_2d, (rotation_vector, translation_vector)
            
        except Exception as e:
            print(f"PnP求解异常: {e}")
            return landmarks_2d, None
    
    def estimate_face_pose_3d(self, rotation_vector):
        """基于3D旋转向量估计面部姿态"""
        if rotation_vector is None:
            return "frontal"
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        
        if sy > 1e-6:
            y = np.arctan2(-rotation_matrix[2,0], sy)
        else:
            y = np.arctan2(-rotation_matrix[2,0], sy)
        
        yaw = np.degrees(y)
        print(f"3D姿态检测: Yaw = {yaw:.1f}度")
        
        if abs(yaw) < 15:
            return "frontal"
        elif yaw > 15:
            return "right_profile"
        else:
            return "left_profile"
    
    def get_dynamic_anchors(self, transform_type, pose_3d):
        """根据变形类型和姿态动态选择锚点"""
        base_anchors = self.DYNAMIC_ANCHORS.copy()
        
        print(f"🔄 动态锚点策略: {transform_type} + {pose_3d}")
        
        if transform_type == "width":
            # 宽度调整时，减少可能冲突的侧面约束
            selected = [93, 323]  # 只保留上方的点，避免宽度冲突
            print(f"   宽度模式：选择上方锚点 {selected}")
            return selected
            
        elif pose_3d == "left_profile":
            # 头向左转，右侧可见，左侧被遮挡
            selected = [454, 323]  # 只保留右侧可见的点
            print(f"   左转头像：选择右侧可见点 {selected}")
            return selected
            
        elif pose_3d == "right_profile":
            # 头向右转，左侧可见，右侧被遮挡
            selected = [234, 93]   # 只保留左侧可见的点
            print(f"   右转头像：选择左侧可见点 {selected}")
            return selected
        else:
            # 正脸时全部启用
            print(f"   正脸模式：启用全部锚点 {base_anchors}")
            return base_anchors
    
    def create_optimized_chin_mask(self, image_shape, landmarks_2d, pose_3d):
        """基于变形点创建优化的下巴mask"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 基于所有变形相关的点创建mask
        mask_points = []
        
        # 包含主控变形点和轮廓点
        for idx in self.CHIN_CENTER_SYSTEM + self.OTHER_CONTOUR:
            if idx < len(landmarks_2d):
                mask_points.append(landmarks_2d[idx])
        
        if len(mask_points) >= 3:
            mask_points = np.array(mask_points, dtype=np.int32)
            hull = cv2.convexHull(mask_points)
            
            # 计算mask区域的实际尺寸
            mask_center = np.mean(hull.reshape(-1, 2), axis=0)
            mask_width = np.max(hull[:, :, 0]) - np.min(hull[:, :, 0])
            mask_height = np.max(hull[:, :, 1]) - np.min(hull[:, :, 1])
            
            # 优化扩展策略
            if pose_3d == "frontal":
                horizontal_expand = 1.4
                vertical_expand = 1.4
            else:  # 侧脸
                horizontal_expand = 1.2
                vertical_expand = 1.2
            
            # 扩展mask
            expanded_hull = []
            for point in hull.reshape(-1, 2):
                dx = point[0] - mask_center[0]
                dy = point[1] - mask_center[1]
                
                new_x = mask_center[0] + dx * horizontal_expand
                new_y = mask_center[1] + dy * vertical_expand
                
                expanded_hull.append([new_x, new_y])
            
            expanded_hull = np.array(expanded_hull, dtype=np.int32)
            cv2.fillPoly(mask, [expanded_hull], 255)
            
            # 边界保护
            mouth_y = max(landmarks_2d[61][1], landmarks_2d[291][1]) if 61 < len(landmarks_2d) and 291 < len(landmarks_2d) else h//2
            mask_top = np.min(expanded_hull[:, 1])
            
            if mask_top < mouth_y + 8:
                mask[:int(mouth_y + 8), :] = 0
                print("⚠️  Mask上边界保护")
            
            print(f"✅ 优化Mask: 姿态={pose_3d}, 扩展={horizontal_expand:.1f}x{vertical_expand:.1f}")
        
        return mask
    
    def apply_optimized_chin_transform(self, image, landmarks_2d, pose_info, transform_type="length", intensity=0.3):
        """基于5层架构的优化下巴变形 - 单一主导点+协调跟随方案"""
        if pose_info is None:
            print("3D姿态信息无效，使用2D模式")
            return image
        
        rotation_vector, translation_vector = pose_info
        pose_3d = self.estimate_face_pose_3d(rotation_vector)
        
        # 3D姿态强度调整
        if pose_3d != "frontal":
            intensity = intensity * 0.75
            print(f"3D侧脸检测，强度调整到: {intensity:.2f}")
        
        # 获取动态锚点
        active_dynamic_anchors = self.get_dynamic_anchors(transform_type, pose_3d)
        
        # 构建5层控制点系统
        src_points = []
        dst_points = []
        h, w = image.shape[:2]
        
        print(f"🎯 5层架构变形系统 - {transform_type} (统一主导模式):")
        
        # 获取152号点作为主导参考点
        chin_center_152 = landmarks_2d[152] if 152 < len(landmarks_2d) else np.array([w//2, h*0.8])
        chin_center_x = chin_center_152[0]
        chin_center_y = chin_center_152[1]
        
        # 🔧 第1层：硬锚点（绝对稳定，0%变形）
        hard_count = 0
        for idx in self.HARD_ANCHORS:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                dst_points.append(point)  # 绝对不变
                hard_count += 1
        print(f"   第1层-硬锚点: {hard_count}个 (0%变形)")
        
        # 🔧 第2层：弱锚点（微跟随，5-10%变形）
        weak_count = 0
        weak_follow_ratio = 0.08  # 8%跟随变形
        
        # 计算主变形方向（用于弱锚点跟随）
        main_transform_vector = [0, 0]
        if transform_type == "length":
            main_transform_vector = [0, intensity * 12]  # 主要是垂直变形
        elif transform_type == "width":
            main_transform_vector = [intensity * 6, 0]   # 主要是水平变形
        elif transform_type == "sharp":
            main_transform_vector = [0, intensity * 8]   # 向下收缩
        elif transform_type == "round":
            main_transform_vector = [0, -intensity * 6]  # 向上扩展
        
        for idx in self.WEAK_ANCHORS:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                # 弱跟随变形
                follow_x = main_transform_vector[0] * weak_follow_ratio
                follow_y = main_transform_vector[1] * weak_follow_ratio
                
                # 根据距离调整跟随强度
                distance_to_center = abs(point[0] - chin_center_x)
                distance_factor = min(distance_to_center / (w * 0.3), 1.0)
                follow_x *= (1 - distance_factor * 0.5)
                follow_y *= (1 - distance_factor * 0.5)
                
                new_point = [point[0] + follow_x, point[1] + follow_y]
                dst_points.append(new_point)
                weak_count += 1
        print(f"   第2层-弱锚点: {weak_count}个 ({weak_follow_ratio*100:.0f}%跟随)")
        
        # 🔧 第3层：动态锚点（任务自适应）
        dynamic_count = 0
        for idx in active_dynamic_anchors:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                # 根据变形类型决定是否跟随
                if transform_type == "width" and abs(intensity) > 0.2:
                    # 宽度调整时，动态锚点可以有微小跟随
                    is_left_side = point[0] < chin_center_x
                    micro_follow = 1.5 if (intensity > 0) == is_left_side else -1.5
                    new_point = [point[0] + micro_follow, point[1]]
                    dst_points.append(new_point)
                else:
                    # 其他情况保持稳定
                    dst_points.append(point)
                dynamic_count += 1
        print(f"   第3层-动态锚点: {dynamic_count}个 (自适应)")
        
        # 🔧 第4层：过渡层（渐变跟随，30-50%变形）
        transition_count = 0
        transition_ratio = 0.4  # 40%跟随主变形
        
        for idx in self.TRANSITION_LAYER:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                # 计算相对于152的位置关系
                relative_vector = point - chin_center_152
                distance_from_152 = np.linalg.norm(relative_vector)
                max_distance = w * 0.15
                distance_ratio = min(distance_from_152 / max_distance, 1.0)
                adjusted_ratio = transition_ratio * (1 - distance_ratio * 0.3)
                
                # 3D姿态调整
                is_left_side = point[0] < chin_center_x
                if pose_3d == "left_profile" and is_left_side:
                    adjusted_ratio *= 0.7  # 不可见侧降低
                elif pose_3d == "right_profile" and not is_left_side:
                    adjusted_ratio *= 0.7  # 不可见侧降低
                
                # 🎯 统一主导模式：都基于152的主导变形进行协调跟随
                if transform_type == "length":
                    # 152主导向下，过渡层跟随+纵向拉伸
                    follow_y = intensity * 12 * adjusted_ratio
                    stretch_effect = intensity * 3 * adjusted_ratio  # 增加拉伸效果
                    new_point = [point[0], point[1] + follow_y + stretch_effect]
                elif transform_type == "width":
                    # 152微调水平，过渡层径向收缩/扩展
                    if distance_from_152 > 5:  # 避免除零
                        radial_direction = relative_vector / distance_from_152
                        radial_move = radial_direction * intensity * 8 * adjusted_ratio
                        new_point = point + radial_move
                    else:
                        new_point = point
                elif transform_type == "sharp":
                    # 152主导向下，过渡层向152收缩+跟随向下
                    shrink_ratio = 0.3 * adjusted_ratio
                    follow_down_ratio = 0.5 * adjusted_ratio
                    if distance_from_152 > 5:
                        shrink_vector = -relative_vector * shrink_ratio
                        follow_down = [0, intensity * 10 * follow_down_ratio]
                        new_point = point + shrink_vector + follow_down
                    else:
                        new_point = [point[0], point[1] + intensity * 10 * follow_down_ratio]
                elif transform_type == "round":
                    # 152主导向上，过渡层远离152+跟随向上
                    expand_ratio = 0.2 * adjusted_ratio
                    follow_up_ratio = 0.3 * adjusted_ratio
                    if distance_from_152 > 5:
                        expand_vector = relative_vector * expand_ratio
                        follow_up = [0, -intensity * 8 * follow_up_ratio]
                        new_point = point + expand_vector + follow_up
                    else:
                        new_point = [point[0], point[1] - intensity * 8 * follow_up_ratio]
                else:
                    new_point = point
                
                dst_points.append(new_point)
                transition_count += 1
        print(f"   第4层-过渡层: {transition_count}个 ({transition_ratio*100:.0f}%协调跟随)")
        
        # 🔧 第5层：主控变形点（完全控制，100%变形）- 🎯统一主导模式
        main_count = 0
        
        # 🎯 统一主导算法：每种模式都有明确的主导点和协调规则
        if transform_type == "length":
            # 152主导向下 + 增强纵向拉伸场效应
            if 152 < len(landmarks_2d):
                point = landmarks_2d[152]
                src_points.append(point)
                main_move = intensity * 18  # 主导移动增强
                new_point = [point[0], point[1] + main_move]
                dst_points.append(new_point)
                main_count += 1
                print(f"     152主导向下: {main_move:.1f}像素")
            
            # 中轴系统协调跟随 + 拉伸场效应
            for idx in [176, 400, 378]:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    relative_vector = point - chin_center_152
                    distance_from_152 = np.linalg.norm(relative_vector)
                    follow_ratio = 0.7 * max(0.3, 1 - distance_from_152 / (w * 0.1))
                    
                    # 跟随向下 + 纵向拉伸效应
                    follow_down = intensity * 18 * follow_ratio
                    stretch_effect = intensity * 5 * follow_ratio  # 拉伸场
                    new_point = [point[0], point[1] + follow_down + stretch_effect]
                    dst_points.append(new_point)
                    main_count += 1
        
        elif transform_type == "width":
            # 152微调水平位置作为宽度视觉锚点
            if 152 < len(landmarks_2d):
                point = landmarks_2d[152]
                src_points.append(point)
                # 轻微水平偏移作为视觉引导
                horizontal_shift = intensity * 4
                new_point = [point[0] + horizontal_shift, point[1]]
                dst_points.append(new_point)
                main_count += 1
                print(f"     152水平引导: {horizontal_shift:.1f}像素")
            
            # 中轴系统统一径向移动
            for idx in [176, 400, 378]:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    relative_vector = point - chin_center_152
                    distance_from_152 = np.linalg.norm(relative_vector)
                    
                    if distance_from_152 > 5:
                        # 统一径向移动：收缩或扩展
                        radial_direction = relative_vector / distance_from_152
                        radial_move = radial_direction * intensity * 12
                        new_point = point + radial_move
                    else:
                        new_point = point
                    
                    dst_points.append(new_point)
                    main_count += 1
        
        elif transform_type == "sharp":
            # 152主导向下尖锐
            if 152 < len(landmarks_2d):
                point = landmarks_2d[152]
                src_points.append(point)
                sharp_down = intensity * 20  # 增强主导效果
                new_point = [point[0], point[1] + sharp_down]
                dst_points.append(new_point)
                main_count += 1
                print(f"     152尖锐主导: {sharp_down:.1f}像素")
            
            # 中轴系统统一向152收缩+跟随向下
            for idx in [176, 400, 378]:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    relative_vector = point - chin_center_152
                    distance_from_152 = np.linalg.norm(relative_vector)
                    
                    # 收缩比例和跟随比例
                    shrink_ratio = 0.4
                    follow_ratio = 0.6
                    
                    if distance_from_152 > 5:
                        # 向152收缩
                        shrink_vector = -relative_vector * shrink_ratio
                        # 跟随向下
                        follow_down = [0, intensity * 15 * follow_ratio]
                        new_point = point + shrink_vector + follow_down
                    else:
                        new_point = [point[0], point[1] + intensity * 15 * follow_ratio]
                    
                    dst_points.append(new_point)
                    main_count += 1
        
        elif transform_type == "round":
            # 152主导向上收缩
            if 152 < len(landmarks_2d):
                point = landmarks_2d[152]
                src_points.append(point)
                round_up = intensity * 15  # 主导向上
                new_point = [point[0], point[1] - round_up]
                dst_points.append(new_point)
                main_count += 1
                print(f"     152圆润主导: -{round_up:.1f}像素")
            
            # 中轴系统统一远离152+微跟随向上
            for idx in [176, 400, 378]:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    relative_vector = point - chin_center_152
                    distance_from_152 = np.linalg.norm(relative_vector)
                    
                    # 扩展比例和跟随比例
                    expand_ratio = 0.25
                    follow_ratio = 0.4
                    
                    if distance_from_152 > 5:
                        # 远离152扩展
                        expand_vector = relative_vector * expand_ratio
                        # 轻微跟随向上
                        follow_up = [0, -intensity * 8 * follow_ratio]
                        new_point = point + expand_vector + follow_up
                    else:
                        new_point = [point[0], point[1] - intensity * 8 * follow_ratio]
                    
                    dst_points.append(new_point)
                    main_count += 1
        
        # 🎯 其他轮廓点：统一采用协调跟随模式
        for idx in self.OTHER_CONTOUR:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                # 计算相对于152的关系
                relative_vector = point - chin_center_152
                distance_from_152 = np.linalg.norm(relative_vector)
                max_distance = w * 0.15
                distance_ratio = min(distance_from_152 / max_distance, 1.0)
                is_left_side = point[0] < chin_center_x
                
                # 3D姿态调整
                multiplier = 1.0
                if pose_3d == "left_profile" and is_left_side:
                    multiplier = 0.8
                elif pose_3d == "right_profile" and not is_left_side:
                    multiplier = 0.8
                
                # 🎯 统一协调跟随算法
                if transform_type == "length":
                    # 跟随152向下 + 拉伸场效应
                    follow_ratio = 0.5 * (1 - distance_ratio * 0.3) * multiplier
                    follow_down = intensity * 18 * follow_ratio
                    stretch_effect = intensity * 4 * follow_ratio
                    new_point = [point[0], point[1] + follow_down + stretch_effect]
                    
                elif transform_type == "width":
                    # 径向移动跟随
                    if distance_from_152 > 5:
                        radial_direction = relative_vector / distance_from_152
                        radial_intensity = intensity * 10 * (1 - distance_ratio * 0.4) * multiplier
                        radial_move = radial_direction * radial_intensity
                        new_point = point + radial_move
                    else:
                        new_point = point
                        
                elif transform_type == "sharp":
                    # 向152收缩 + 跟随向下
                    shrink_ratio = 0.25 * (1 - distance_ratio * 0.3) * multiplier
                    follow_ratio = 0.4 * (1 - distance_ratio * 0.2) * multiplier
                    
                    if distance_from_152 > 5:
                        shrink_vector = -relative_vector * shrink_ratio
                        follow_down = [0, intensity * 15 * follow_ratio]
                        new_point = point + shrink_vector + follow_down
                    else:
                        new_point = [point[0], point[1] + intensity * 15 * follow_ratio]
                        
                elif transform_type == "round":
                    # 远离152扩展 + 轻微向上
                    expand_ratio = 0.2 * (1 - distance_ratio * 0.3) * multiplier
                    follow_ratio = 0.3 * (1 - distance_ratio * 0.2) * multiplier
                    
                    if distance_from_152 > 5:
                        expand_vector = relative_vector * expand_ratio
                        follow_up = [0, -intensity * 10 * follow_ratio]
                        new_point = point + expand_vector + follow_up
                    else:
                        new_point = [point[0], point[1] - intensity * 10 * follow_ratio]
                else:
                    new_point = point
                
                dst_points.append(new_point)
                main_count += 1
        
        print(f"   第5层-主控变形: {main_count}个 (统一主导模式)")
        
        # 添加其他稳定区域（眼部、鼻子、嘴部）
        other_stable = [33, 133, 362, 263, 1, 2, 5, 61, 291, 9, 151]
        other_count = 0
        for idx in other_stable:
            if idx < len(landmarks_2d):
                src_points.append(landmarks_2d[idx])
                dst_points.append(landmarks_2d[idx])
                other_count += 1
        print(f"   其他稳定区域: {other_count}个")
        
        # 添加边界锚点
        boundary_margin = 25
        boundary_points = [
            [boundary_margin, boundary_margin], 
            [w//2, boundary_margin], 
            [w-boundary_margin, boundary_margin],
            [boundary_margin, h//2], 
            [w-boundary_margin, h//2],
            [boundary_margin, h-boundary_margin], 
            [w//2, h-boundary_margin], 
            [w-boundary_margin, h-boundary_margin]
        ]
        
        for bp in boundary_points:
            src_points.append(bp)
            dst_points.append(bp)
        
        total_points = len(src_points)
        print(f"   边界保护: 8个")
        print(f"   总计控制点: {total_points}个")
        
        if total_points < 15:
            print("❌ 控制点不足")
            return image
        
        # 执行TPS变换
        try:
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            
            tps = cv2.createThinPlateSplineShapeTransformer()
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
            
            tps.estimateTransformation(
                dst_points.reshape(1, -1, 2), 
                src_points.reshape(1, -1, 2), 
                matches
            )
            
            # TPS变形
            transformed = tps.warpImage(image)
            
            if transformed is None:
                print("❌ TPS变换失败")
                return image
            
            # 创建优化mask
            chin_mask = self.create_optimized_chin_mask(image.shape, landmarks_2d, pose_3d)
            
            # 融合
            mask_3d = cv2.merge([chin_mask, chin_mask, chin_mask]).astype(np.float32) / 255.0
            kernel_size = max(17, min(w, h) // 45)
            if kernel_size % 2 == 0:
                kernel_size += 1
            mask_blurred = cv2.GaussianBlur(mask_3d, (kernel_size, kernel_size), kernel_size//3)
            
            result = image.astype(np.float32) * (1 - mask_blurred) + transformed.astype(np.float32) * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print("✅ 统一主导模式变形成功")
            return result
            
        except Exception as e:
            print(f"❌ 统一主导变形异常: {e}")
            return image
    
    def debug_optimized_detection(self, image, save_dir=None):
        """调试5层架构的关键点检测"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("未检测到面部关键点")
            return image
        
        debug_img = image.copy()
        
        # 🎯 按5层架构绘制关键点
        layer_configs = [
            (self.HARD_ANCHORS, (255, 255, 0), "HARD", 10),           # 硬锚点-黄色-大圆
            (self.WEAK_ANCHORS, (0, 255, 255), "WEAK", 8),            # 弱锚点-青色-中圆  
            (self.DYNAMIC_ANCHORS, (255, 0, 255), "DYN", 7),          # 动态锚点-紫色-中圆
            (self.TRANSITION_LAYER, (0, 255, 0), "TRANS", 6),         # 过渡层-绿色-小圆
            (self.CHIN_CENTER_SYSTEM, (0, 0, 255), "CENTER", 8),      # 中轴系统-红色-中圆
            (self.OTHER_CONTOUR, (255, 0, 0), "CONT", 5),             # 其他轮廓-蓝色-小圆
        ]
        
        for indices, color, prefix, radius in layer_configs:
            for i, idx in enumerate(indices):
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx].astype(int)
                    
                    # 152用特大圆突出显示
                    if idx == 152:
                        radius = 12
                    
                    cv2.circle(debug_img, tuple(point), radius, color, -1)
                    cv2.circle(debug_img, tuple(point), radius + 2, color, 2)
                    cv2.putText(debug_img, f"{prefix}{idx}", tuple(point + 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 显示PnP关键点（用于3D姿态求解的6个点）
        pnp_colors = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 255, 128), (255, 128, 128)]
        for i, idx in enumerate(self.pnp_indices):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 9, pnp_colors[i], 3)
                cv2.putText(debug_img, f"PnP{idx}", tuple(point + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, pnp_colors[i], 2)
        
        # 显示3D姿态信息和动态锚点选择
        if pose_info is not None:
            rotation_vector, translation_vector = pose_info
            pose_3d = self.estimate_face_pose_3d(rotation_vector)
            cv2.putText(debug_img, f"3D Pose: {pose_3d}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示动态锚点选择结果
            active_anchors = self.get_dynamic_anchors("length", pose_3d)  # 示例用length
            cv2.putText(debug_img, f"Active Dynamic: {active_anchors}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # 显示5层架构统计
        stats_text = f"5-Layer: HARD({len(self.HARD_ANCHORS)}) WEAK({len(self.WEAK_ANCHORS)}) DYN({len(self.DYNAMIC_ANCHORS)}) TRANS({len(self.TRANSITION_LAYER)}) MAIN({len(self.CHIN_CENTER_SYSTEM)+len(self.OTHER_CONTOUR)})"
        cv2.putText(debug_img, stats_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 显示优化mask
        if pose_info is not None:
            pose_3d = self.estimate_face_pose_3d(pose_info[0])
            chin_mask = self.create_optimized_chin_mask(image.shape, landmarks_2d, pose_3d)
            mask_overlay = debug_img.copy()
            mask_overlay[chin_mask > 0] = [128, 0, 128]  # 深紫色mask
            debug_img = cv2.addWeighted(debug_img, 0.75, mask_overlay, 0.25, 0)
        
        # 更新图例说明（5层架构）
        legend_y = 120
        cv2.putText(debug_img, "YELLOW: Hard Anchors (Absolute Stable)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(debug_img, "CYAN: Weak Anchors (8% Follow)", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(debug_img, "MAGENTA: Dynamic Anchors (Task Adaptive)", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(debug_img, "GREEN: Transition Layer (40% Follow)", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug_img, "RED: Main Control (100% Transform)", (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(debug_img, "BLUE: Other Contour Points", (10, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_img, "WHITE: PnP Points for 3D Pose", (10, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_img, "PURPLE: Optimized Transform Mask", (10, legend_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1)
        
        # 绘制层级连接线（显示5层关系）
        if len(self.CHIN_CENTER_SYSTEM) >= 2:
            # 连接中轴系统的点
            center_points = []
            for idx in self.CHIN_CENTER_SYSTEM:
                if idx < len(landmarks_2d):
                    center_points.append(landmarks_2d[idx].astype(int))
            
            if len(center_points) >= 2:
                for i in range(len(center_points) - 1):
                    cv2.line(debug_img, tuple(center_points[i]), tuple(center_points[i+1]), (0, 0, 255), 2)
        
        # 连接过渡层点（显示渐变关系）
        if len(self.TRANSITION_LAYER) >= 2:
            trans_points = []
            for idx in self.TRANSITION_LAYER:
                if idx < len(landmarks_2d):
                    trans_points.append(landmarks_2d[idx].astype(int))
            
            if len(trans_points) >= 2:
                for i in range(len(trans_points) - 1):
                    cv2.line(debug_img, tuple(trans_points[i]), tuple(trans_points[i+1]), (0, 255, 0), 1)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "debug_chin_5layer_optimized.png"), debug_img)
            print(f"5层架构调试文件保存到: {save_dir}")
        
        return debug_img
    
    def process_image(self, image, transform_type="length", intensity=0.3, save_debug=False, output_path=None):
        """处理图像主函数"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("未检测到面部关键点")
            return image
        
        print(f"✅ 检测到 {len(landmarks_2d)} 个2D关键点")
        
        if save_debug and output_path:
            debug_dir = os.path.splitext(output_path)[0] + "_debug"
            self.debug_optimized_detection(image, debug_dir)
        
        return self.apply_optimized_chin_transform(image, landmarks_2d, pose_info, transform_type, intensity)

def main():
    parser = argparse.ArgumentParser(description='优化的5层架构3D下巴调整工具 - 解决边缘扭曲问题')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--type', '-t', choices=['length', 'width', 'sharp', 'round'], 
                       default='length', help='调整类型：length=长度, width=宽度, sharp=尖锐, round=圆润')
    parser.add_argument('--intensity', type=float, default=0.3, help='调整强度 (0.1-0.5)')
    parser.add_argument('--debug', action='store_true', help='5层架构调试模式')
    parser.add_argument('--save-debug', action='store_true', help='保存调试文件')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"无法加载图片: {args.input}")
        return
    
    print(f"图片尺寸: {image.shape}")
    
    # 创建优化处理器
    processor = OptimizedChin3DAdjustment()
    
    # 调试模式
    if args.debug:
        debug_result = processor.debug_optimized_detection(image)
        cv2.imwrite(args.output, debug_result)
        print(f"5层架构调试图保存到: {args.output}")
        return
    
    # 强度安全检查
    if args.intensity > 0.5:
        print(f"警告：强度 {args.intensity} 可能造成扭曲，自动限制到0.5")
        args.intensity = 0.5
    elif args.intensity < -0.5:
        print(f"警告：强度 {args.intensity} 可能造成扭曲，自动限制到-0.5")
        args.intensity = -0.5
    
    # 处理图片
    print(f"开始5层架构优化处理 - 类型: {args.type}, 强度: {args.intensity}")
    result = processor.process_image(image, args.type, args.intensity, args.save_debug, args.output)
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"处理完成，保存到: {args.output}")
    
    # 检查变化
    diff = cv2.absdiff(image, result)
    total_diff = np.sum(diff)
    changed_pixels = np.sum(diff > 0)
    max_diff = np.max(diff)
    print(f"📊 像素变化统计:")
    print(f"   总差异: {total_diff}")
    print(f"   最大差异: {max_diff}")
    print(f"   变化像素数: {changed_pixels}")
    
    if total_diff < 1000:
        print("⚠️  变化很小，可能需要：")
        print("   1. 提高 --intensity 参数")
        print("   2. 检查关键点是否正确检测")
        print("   3. 尝试不同的变形类型")
    
    # 生成对比图
    if args.comparison:
        comparison_path = args.output.replace('.', '_5layer_comparison.')
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
        
        # 添加标签
        cv2.putText(comparison, 'Original', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(comparison, f'5Layer-{args.type.upper()} {args.intensity}', (image.shape[1] + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imwrite(comparison_path, comparison)
        print(f"5层架构对比图保存到: {comparison_path}")
    
    print("\n✅ 5层架构优化特性:")
    print("🔧 第1层: 硬锚点 - 绝对稳定脸型基准 [10,338,297]")
    print("🔧 第2层: 弱锚点 - 8%跟随防护中脸 [117,123,147,346,352,376]")
    print("🔧 第3层: 动态锚点 - 任务自适应选择 [234,454,93,323]")
    print("🔧 第4层: 过渡层 - 40%渐变平滑过渡 [132,58,288,361]")
    print("🔧 第5层: 主控变形 - 100%精确控制变形")
    print("\n🎯 解决的问题:")
    print("• ✅ 消除边缘扭曲：弱锚点+过渡层提供平滑梯度")
    print("• ✅ 侧脸自适应：动态锚点根据姿态智能选择")
    print("• ✅ 变形自然度：5层精细强度分级控制")
    print("• ✅ 脸型稳定性：硬锚点确保整体不漂移")
    print("\n📖 使用示例:")
    print("• 下巴拉长: python chin_adjust_optimized.py -i input.jpg -o output.png -t length --intensity 0.3 --comparison")
    print("• 下巴收窄: python chin_adjust_optimized.py -i input.jpg -o output.png -t width --intensity -0.2 --comparison")
    print("• 尖下巴:   python chin_adjust_optimized.py -i input.jpg -o output.png -t sharp --intensity 0.25 --comparison")
    print("• 圆下巴:   python chin_adjust_optimized.py -i input.jpg -o output.png -t round --intensity 0.2 --comparison")
    print("• 调试模式: python chin_adjust_optimized.py -i input.jpg -o debug.png --debug")
    print("\n⚠️  重要参数:")
    print("• 强度范围: -0.5 到 0.5")
    print("• 5层架构自动防扭曲")
    print("• 3D姿态自适应锚点")

if __name__ == "__main__":
    main()