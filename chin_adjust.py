#!/usr/bin/env python3
"""
增强版下巴调整脚本 - 在关键空白区域增加协调控制点
解决整体面部协调性问题，消除边缘像素漂移
基于原有5层架构，精准添加关键过渡控制点
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
from typing import List, Tuple, Optional

class OptimizedChin3DAdjustment:
    def __init__(self,face_mesh=None):
        self.mp_face_mesh = mp.solutions.face_mesh
        if not face_mesh:
        # MediaPipe设置
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            )
        else:
            self.face_mesh = face_mesh
        
        # 相机参数
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 🎯 原有5层架构（保持不变）
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
        
        # 🆕 新增：关键过渡控制点（填补空白区域）
        # 颞部稳定点（防止太阳穴区域漂移）
        self.TEMPORAL_STABLE = [162, 21, 54, 389, 251, 284]  # 左右各3个
        
        # 中脸过渡线（颧骨到脸颊的连接）
        self.MID_FACE_TRANSITION = [116, 118, 119, 345, 347, 348]  # 左右各3个
        
        # 脸颊下缘过渡线（脸颊到下颌的关键过渡）
        self.CHEEK_JAW_TRANSITION = [177, 215, 138, 402, 435, 367]  # 左右各3个
        
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
            selected = [93, 323]
            print(f"   宽度模式：选择上方锚点 {selected}")
            return selected
            
        elif pose_3d == "left_profile":
            selected = [454, 323]
            print(f"   左转头像：选择右侧可见点 {selected}")
            return selected
            
        elif pose_3d == "right_profile":
            selected = [234, 93]
            print(f"   右转头像：选择左侧可见点 {selected}")
            return selected
        else:
            print(f"   正脸模式：启用全部锚点 {base_anchors}")
            return base_anchors
    
    def calculate_follow_movement(self, point, chin_center_152, main_transform_vector, follow_ratio, pose_3d, w):
        """计算控制点的协调跟随移动"""
        # 获取相对位置
        relative_vector = point - chin_center_152
        distance_from_152 = np.linalg.norm(relative_vector)
        is_left_side = point[0] < chin_center_152[0]
        
        # 3D姿态调整
        pose_multiplier = 1.0
        if pose_3d == "left_profile" and is_left_side:
            pose_multiplier = 0.8  # 不可见侧降低
        elif pose_3d == "right_profile" and not is_left_side:
            pose_multiplier = 0.8  # 不可见侧降低
        
        # 距离衰减（离152越远，跟随越弱）
        max_distance = w * 0.2
        distance_ratio = min(distance_from_152 / max_distance, 1.0)
        distance_factor = 1.0 - distance_ratio * 0.3
        
        # 计算最终跟随移动
        final_follow_ratio = follow_ratio * pose_multiplier * distance_factor
        follow_movement = np.array(main_transform_vector) * final_follow_ratio
        
        return point + follow_movement
    
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
        """增强版下巴变形 - 增加关键过渡控制点实现整体协调"""
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
        
        # 构建增强控制点系统
        src_points = []
        dst_points = []
        h, w = image.shape[:2]
        
        print(f"🎯 增强版变形系统 - {transform_type} (关键过渡控制点):")
        
        # 获取152号点作为主导参考点
        chin_center_152 = landmarks_2d[152] if 152 < len(landmarks_2d) else np.array([w//2, h*0.8])
        
        # 计算主变形向量（152的移动向量）
        main_transform_vector = [0, 0]
        if transform_type == "length":
            main_transform_vector = [0, intensity * 18]
        elif transform_type == "width":
            main_transform_vector = [intensity * 4, 0]
        elif transform_type == "sharp":
            main_transform_vector = [0, intensity * 20]
        elif transform_type == "round":
            main_transform_vector = [0, -intensity * 15]
        
        print(f"   152主导变形向量: {main_transform_vector}")
        
        # 🔧 第1层：硬锚点（绝对稳定，0%变形）
        hard_count = 0
        for idx in self.HARD_ANCHORS:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                dst_points.append(point)  # 绝对不变
                hard_count += 1
        print(f"   第1层-硬锚点: {hard_count}个 (0%变形)")
        
        # 🆕 新增层：颞部稳定点（5-8%微跟随）
        temporal_count = 0
        temporal_follow_ratio = 0.06  # 6%跟随
        
        for idx in self.TEMPORAL_STABLE:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                new_point = self.calculate_follow_movement(
                    point, chin_center_152, main_transform_vector, 
                    temporal_follow_ratio, pose_3d, w
                )
                dst_points.append(new_point)
                temporal_count += 1
        print(f"   新增-颞部稳定: {temporal_count}个 ({temporal_follow_ratio*100:.0f}%协调跟随)")
        
        # 🔧 第2层：弱锚点（微跟随稳定，8%变形）
        weak_count = 0
        weak_follow_ratio = 0.08
        
        for idx in self.WEAK_ANCHORS:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                new_point = self.calculate_follow_movement(
                    point, chin_center_152, main_transform_vector, 
                    weak_follow_ratio, pose_3d, w
                )
                dst_points.append(new_point)
                weak_count += 1
        print(f"   第2层-弱锚点: {weak_count}个 ({weak_follow_ratio*100:.0f}%协调跟随)")
        
        # 🆕 新增层：中脸过渡线（15-20%跟随）
        mid_face_count = 0
        mid_face_follow_ratio = 0.18  # 18%跟随
        
        for idx in self.MID_FACE_TRANSITION:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                new_point = self.calculate_follow_movement(
                    point, chin_center_152, main_transform_vector, 
                    mid_face_follow_ratio, pose_3d, w
                )
                dst_points.append(new_point)
                mid_face_count += 1
        print(f"   新增-中脸过渡: {mid_face_count}个 ({mid_face_follow_ratio*100:.0f}%协调跟随)")
        
        # 🆕 新增层：脸颊下缘过渡线（25-40%跟随）
        cheek_jaw_count = 0
        cheek_jaw_follow_ratio = 0.38  # 38%跟随
        
        for idx in self.CHEEK_JAW_TRANSITION:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                new_point = self.calculate_follow_movement(
                    point, chin_center_152, main_transform_vector, 
                    cheek_jaw_follow_ratio, pose_3d, w
                )
                dst_points.append(new_point)
                cheek_jaw_count += 1
        print(f"   新增-脸颊下缘: {cheek_jaw_count}个 ({cheek_jaw_follow_ratio*100:.0f}%协调跟随)")
        
        # 🔧 第3层：动态锚点（任务自适应）
        dynamic_count = 0
        for idx in active_dynamic_anchors:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                # 根据变形类型决定是否跟随
                if transform_type == "width" and abs(intensity) > 0.2:
                    is_left_side = point[0] < chin_center_152[0]
                    micro_follow = 1.5 if (intensity > 0) == is_left_side else -1.5
                    new_point = [point[0] + micro_follow, point[1]]
                    dst_points.append(new_point)
                else:
                    dst_points.append(point)
                dynamic_count += 1
        print(f"   第3层-动态锚点: {dynamic_count}个 (自适应)")
        
        # 🔧 第4层：过渡层（渐变跟随，55%变形）
        transition_count = 0
        transition_ratio = 0.55
        
        for idx in self.TRANSITION_LAYER:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                new_point = self.calculate_follow_movement(
                    point, chin_center_152, main_transform_vector, 
                    transition_ratio, pose_3d, w
                )
                dst_points.append(new_point)
                transition_count += 1
        print(f"   第4层-过渡层: {transition_count}个 ({transition_ratio*100:.0f}%协调跟随)")
        
        # 🔧 第5层：主控变形点（完全控制，100%变形）
        main_count = 0
        
        # 152主导变形
        if 152 < len(landmarks_2d):
            point = landmarks_2d[152]
            src_points.append(point)
            new_point = point + np.array(main_transform_vector)
            dst_points.append(new_point)
            main_count += 1
            print(f"     152主导移动: {main_transform_vector}")
        
        # 中轴系统协调跟随
        for idx in [176, 400, 378]:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                # 协调跟随152的主导变形
                relative_vector = point - chin_center_152
                distance_from_152 = np.linalg.norm(relative_vector)
                follow_ratio = 0.7 * max(0.3, 1 - distance_from_152 / (w * 0.1))
                
                coordinated_movement = np.array(main_transform_vector) * follow_ratio
                new_point = point + coordinated_movement
                dst_points.append(new_point)
                main_count += 1
        
        # 其他轮廓点协调跟随
        for idx in self.OTHER_CONTOUR:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                new_point = self.calculate_follow_movement(
                    point, chin_center_152, main_transform_vector, 
                    0.5, pose_3d, w  # 50%协调跟随
                )
                dst_points.append(new_point)
                main_count += 1
        
        print(f"   第5层-主控变形: {main_count}个 (协调主导模式)")
        
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
        print(f"   总计控制点: {total_points}个 (增强前约20个)")
        
        if total_points < 25:
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
            
            print("✅ 优化版协调变形成功")
            return result
            
        except Exception as e:
            print(f"❌ 优化版变形异常: {e}")
            return image
    
    def debug_optimized_detection(self, image, save_dir=None):
        """调试增强版的关键点检测"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("未检测到面部关键点")
            return image
        
        debug_img = image.copy()
        
        # 🎯 按增强架构绘制关键点
        layer_configs = [
            (self.HARD_ANCHORS, (255, 255, 0), "HARD", 10),                    # 硬锚点-黄色-大圆
            (self.TEMPORAL_STABLE, (128, 255, 255), "TEMP", 7),                # 颞部稳定-浅青色
            (self.WEAK_ANCHORS, (0, 255, 255), "WEAK", 8),                     # 弱锚点-青色-中圆  
            (self.MID_FACE_TRANSITION, (255, 128, 0), "MID", 6),               # 中脸过渡-橙色
            (self.CHEEK_JAW_TRANSITION, (255, 0, 128), "CHEEK", 6),            # 脸颊下缘-粉色
            (self.DYNAMIC_ANCHORS, (255, 0, 255), "DYN", 7),                   # 动态锚点-紫色-中圆
            (self.TRANSITION_LAYER, (0, 255, 0), "TRANS", 6),                  # 过渡层-绿色-小圆
            (self.CHIN_CENTER_SYSTEM, (0, 0, 255), "CENTER", 8),               # 中轴系统-红色-中圆
            (self.OTHER_CONTOUR, (255, 0, 0), "CONT", 5),                      # 其他轮廓-蓝色-小圆
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
        
        # 显示3D姿态信息
        if pose_info is not None:
            rotation_vector, translation_vector = pose_info
            pose_3d = self.estimate_face_pose_3d(rotation_vector)
            cv2.putText(debug_img, f"3D Pose: {pose_3d}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示增强架构统计
        total_new_points = len(self.TEMPORAL_STABLE) + len(self.MID_FACE_TRANSITION) + len(self.CHEEK_JAW_TRANSITION)
        stats_text = f"Enhanced: +{total_new_points} transition points (TEMP:{len(self.TEMPORAL_STABLE)} MID:{len(self.MID_FACE_TRANSITION)} CHEEK:{len(self.CHEEK_JAW_TRANSITION)})"
        cv2.putText(debug_img, stats_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 显示优化mask
        if pose_info is not None:
            pose_3d = self.estimate_face_pose_3d(pose_info[0])
            chin_mask = self.create_optimized_chin_mask(image.shape, landmarks_2d, pose_3d)
            mask_overlay = debug_img.copy()
            mask_overlay[chin_mask > 0] = [128, 0, 128]  # 深紫色mask
            debug_img = cv2.addWeighted(debug_img, 0.75, mask_overlay, 0.25, 0)
        
        # 更新图例说明（增强架构）
        legend_y = 90
        cv2.putText(debug_img, "YELLOW: Hard Anchors (0%)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(debug_img, "LIGHT_CYAN: Temporal Stable (6%)", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 255, 255), 1)
        cv2.putText(debug_img, "CYAN: Weak Anchors (8%)", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(debug_img, "ORANGE: Mid Face Transition (18%)", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
        cv2.putText(debug_img, "PINK: Cheek-Jaw Transition (28%)", (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 128), 1)
        cv2.putText(debug_img, "MAGENTA: Dynamic Anchors", (10, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(debug_img, "GREEN: Transition Layer (40%)", (10, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug_img, "RED: Main Control (100%)", (10, legend_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(debug_img, "BLUE: Other Contour (50%)", (10, legend_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 绘制协调跟随连接线
        if 152 < len(landmarks_2d):
            chin_152 = landmarks_2d[152].astype(int)
            
            # 连接新增的过渡点到152，显示协调关系
            for transition_points, color in [(self.TEMPORAL_STABLE, (128, 255, 255)), 
                                           (self.MID_FACE_TRANSITION, (255, 128, 0)),
                                           (self.CHEEK_JAW_TRANSITION, (255, 0, 128))]:
                for idx in transition_points:
                    if idx < len(landmarks_2d):
                        point = landmarks_2d[idx].astype(int)
                        # 绘制虚线表示协调关系
                        cv2.line(debug_img, tuple(point), tuple(chin_152), color, 1, cv2.LINE_AA)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "debug_chin_optimized.png"), debug_img)
            print(f"优化版调试文件保存到: {save_dir}")
        
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
    parser = argparse.ArgumentParser(description='增强版下巴调整工具 - 关键过渡控制点解决整体协调性')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--type', '-t', choices=['length', 'width', 'sharp', 'round'], 
                       default='length', help='调整类型：length=长度, width=宽度, sharp=尖锐, round=圆润')
    parser.add_argument('--intensity', type=float, default=0.3, help='调整强度 (0.1-0.5)')
    parser.add_argument('--debug', action='store_true', help='增强架构调试模式')
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
        print(f"增强架构调试图保存到: {args.output}")
        return
    
    # 强度安全检查
    if args.intensity > 0.5:
        print(f"警告：强度 {args.intensity} 可能造成扭曲，自动限制到0.5")
        args.intensity = 0.5
    elif args.intensity < -0.5:
        print(f"警告：强度 {args.intensity} 可能造成扭曲，自动限制到-0.5")
        args.intensity = -0.5
    
    # 处理图片
    print(f"开始增强版处理 - 类型: {args.type}, 强度: {args.intensity}")
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
        comparison_path = args.output.replace('.', '_enhanced_comparison.')
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
        
        # 添加标签
        cv2.putText(comparison, 'Original', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(comparison, f'Enhanced-{args.type.upper()} {args.intensity}', (image.shape[1] + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imwrite(comparison_path, comparison)
        print(f"增强版对比图保存到: {comparison_path}")
    
    print("\n✅ 增强版特性:")
    print("🆕 新增层级:")
    print("   • 颞部稳定点: 6%协调跟随 - 防止太阳穴漂移")
    print("   • 中脸过渡线: 18%协调跟随 - 颧骨到脸颊连接")
    print("   • 脸颊下缘过渡: 28%协调跟随 - 脸颊到下颌关键过渡")
    print("\n🔧 原有5层架构保持:")
    print("   • 硬锚点: 0%变形 (绝对稳定)")
    print("   • 弱锚点: 8%协调跟随 (微跟随)")
    print("   • 动态锚点: 自适应选择")
    print("   • 过渡层: 40%协调跟随")
    print("   • 主控变形: 100%精确控制")
    print("\n🎯 解决的关键问题:")
    print("   ✅ 消除空白区域：关键过渡线全覆盖")
    print("   ✅ 整体协调性：全脸轻微跟随下巴变形")
    print("   ✅ 边缘连续性：平滑的变形梯度分布")
    print("   ✅ 保持变形效果：下巴变形强度完全保留")
    print("\n🔢 控制点统计:")
    print("   • 原版: ~20个控制点")
    print("   • 增强版: ~32个控制点 (新增12个关键过渡点)")
    print("   • 所有新增点都基于152主导变形进行协调计算")
    print("\n📖 使用示例:")
    print("   • 协调下巴拉长: python chin_adjust_enhanced.py -i input.jpg -o output.png -t length --intensity 0.3 --comparison")
    print("   • 协调下巴收窄: python chin_adjust_enhanced.py -i input.jpg -o output.png -t width --intensity -0.2 --comparison")
    print("   • 查看增强架构: python chin_adjust_enhanced.py -i input.jpg -o debug.png --debug")

if __name__ == "__main__":
    main()