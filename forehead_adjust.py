#!/usr/bin/env python3
"""
额头调整脚本 - forehead_adjust.py
精确版本：只使用确认的额头点，采用更有效的变形策略
借鉴瘦脸成功经验，专注发际线和额头高度调整
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os

class ForeheadAdjustment:
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
        
        # 精确的额头控制点（只使用确认的MediaPipe额头点）
        self.FOREHEAD_CENTER = [10, 9, 151]                    # 中央脊椎（固定）
        self.FOREHEAD_LEFT = [66, 107]                         # 左侧额头主点
        self.FOREHEAD_RIGHT = [336, 296]                       # 右侧额头主点
        
        # 发际线相关点（额头上边缘）
        self.HAIRLINE_POINTS = [
            67, 69, 70, 71,                # 左侧发际线
            299, 333, 298, 301              # 右侧发际线
        ]
        
        # 眉毛上方点（额头下边缘）
        self.EYEBROW_UPPER = [
            55, 65, 52, 53,                # 左眉上方
            285, 295, 282, 283              # 右眉上方
        ]
        
        # 太阳穴区域点
        self.TEMPLE_POINTS = [234, 454]    # 左右太阳穴
        
        # 稳定锚点（远离额头的点）
        self.STABLE_ANCHORS = [
            # 眼部
            33, 133, 362, 263, 7, 163, 144, 145, 153, 154, 155, 173,
            # 鼻子
            1, 2, 5, 4, 6, 19, 20,
            # 嘴部
            61, 291, 39, 269, 270, 0, 17, 18,
            # 下脸轮廓
            200, 199, 175, 164, 136, 150, 149, 176, 148, 152
        ]
        
        # 3D面部模型
        self.face_model_3d = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0],
        ], dtype=np.float32)
        
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
        
        # 获取2D关键点
        landmarks_2d = []
        h, w = image.shape[:2]
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_2d.append([x, y])
        
        landmarks_2d = np.array(landmarks_2d, dtype=np.float32)
        
        # 设置相机参数
        if self.camera_matrix is None:
            self.setup_camera(image.shape)
        
        # 提取用于PnP的关键点
        image_points = []
        for idx in self.pnp_indices:
            if idx < len(landmarks_2d):
                image_points.append(landmarks_2d[idx])
        
        if len(image_points) < 6:
            print("⚠️  PnP关键点不足")
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
                print("⚠️  PnP求解失败")
                return landmarks_2d, None
            
            return landmarks_2d, (rotation_vector, translation_vector)
            
        except Exception as e:
            print(f"⚠️  PnP求解异常: {e}")
            return landmarks_2d, None
    
    def estimate_face_pose_3d(self, rotation_vector):
        """估计面部姿态"""
        if rotation_vector is None:
            return "frontal"
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        
        if sy > 1e-6:
            y = np.arctan2(-rotation_matrix[2,0], sy)
        else:
            y = np.arctan2(-rotation_matrix[2,0], sy)
        
        yaw = np.degrees(y)
        print(f"🔍 3D姿态检测: Yaw = {yaw:.1f}度")
        
        if abs(yaw) < 15:
            return "frontal"
        elif yaw > 15:
            return "right_profile"
        else:
            return "left_profile"
    
    def create_forehead_mask(self, image_shape, landmarks_2d):
        """创建精确的额头mask"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 收集确认的额头点
        forehead_points = []
        all_points = (self.FOREHEAD_CENTER + self.FOREHEAD_LEFT + self.FOREHEAD_RIGHT + 
                     self.HAIRLINE_POINTS + self.EYEBROW_UPPER + self.TEMPLE_POINTS)
        
        for idx in all_points:
            if idx < len(landmarks_2d):
                forehead_points.append(landmarks_2d[idx])
        
        if len(forehead_points) >= 4:
            forehead_points = np.array(forehead_points, dtype=np.int32)
            hull = cv2.convexHull(forehead_points)
            
            # 保守扩展
            forehead_center = np.mean(hull.reshape(-1, 2), axis=0)
            expanded_hull = []
            
            for point in hull.reshape(-1, 2):
                dx = point[0] - forehead_center[0]
                dy = point[1] - forehead_center[1]
                
                new_x = forehead_center[0] + dx * 1.1
                new_y = forehead_center[1] + dy * 1.1
                
                expanded_hull.append([new_x, new_y])
            
            expanded_hull = np.array(expanded_hull, dtype=np.int32)
            
            # 边界保护
            if 33 < len(landmarks_2d) and 263 < len(landmarks_2d):
                eye_y = min(landmarks_2d[33][1], landmarks_2d[263][1])
                for i, point in enumerate(expanded_hull):
                    if point[1] > eye_y - 20:
                        expanded_hull[i][1] = eye_y - 20
                    expanded_hull[i][0] = np.clip(point[0], 5, w-5)
                    expanded_hull[i][1] = np.clip(point[1], 5, h-5)
            
            cv2.fillPoly(mask, [expanded_hull], 255)
        
        return mask

    def calculate_forehead_center(self, landmarks_2d):
        """计算额头3D变形中心点"""
        # 收集额头关键点
        forehead_points = []
        for idx in self.FOREHEAD_CENTER + self.FOREHEAD_LEFT + self.FOREHEAD_RIGHT:
            if idx < len(landmarks_2d):
                forehead_points.append(landmarks_2d[idx])
        
        if len(forehead_points) < 3:
            return None
        
        # 计算几何中心
        forehead_points = np.array(forehead_points)
        center = np.mean(forehead_points, axis=0)
        return center

    def apply_radial_transform(self, point, center, intensity, mode="expand"):
        """径向变形算法 - 模拟3D效果"""
        if center is None:
            return point
        
        # 计算从中心到点的向量
        direction = point - center
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:  # 避免除零
            return point
        
        # 归一化方向向量
        direction_normalized = direction / distance
        
        if mode == "expand":
            # 径向向外扩展（饱满效果）
            expansion = intensity * 8.0 * (1 - distance / 200.0)  # 距离越近扩展越大
            expansion = max(0, expansion)  # 确保不为负
            new_point = point + direction_normalized * expansion
        else:  # "shrink"
            # 径向向内收缩（平整效果）
            contraction = intensity * 6.0 * (1 - distance / 100.0)
            contraction = max(0, contraction)
            new_point = point - direction_normalized * contraction
        
        return new_point

    def apply_vertical_lift(self, point, center, intensity, lift_type="up"):
        """垂直提升算法 - 模拟3D隆起"""
        if center is None:
            return point
        
        # 计算距离中心的距离
        distance_from_center = np.linalg.norm(point - center)
        max_distance = 300.0  # 最大影响距离
        
        if distance_from_center > max_distance:
            return point
        
        # 距离越近，提升越大（高斯分布效果）
        distance_ratio = distance_from_center / max_distance
        lift_factor = np.exp(-distance_ratio * distance_ratio * 0.5)  # 高斯衰减
        
        if lift_type == "up":
            # 向上提升（饱满效果）
            lift_amount = intensity * 35.0 * lift_factor
            new_point = [point[0], point[1] - lift_amount]  # Y减小是向上
        else:  # "down"
            # 向下压平（平整效果）
            lift_amount = intensity * 20.0 * lift_factor
            new_point = [point[0], point[1] + lift_amount]  # Y增大是向下
        
        return new_point
        
    def apply_effective_forehead_transform(self, image, landmarks_2d, pose_info, transform_type="shrink", intensity=0.3):
        """有效的额头3D变形 - 饱满vs平整效果"""
        if pose_info is None:
            pose_3d = "frontal"
        else:
            rotation_vector, translation_vector = pose_info
            pose_3d = self.estimate_face_pose_3d(rotation_vector)
        
        # 3D姿态自适应
        if pose_3d != "frontal":
            intensity = intensity * 0.8
            print(f"🔍 3D侧脸检测，强度调整到: {intensity:.2f}")
        
        # 构建控制点
        src_points = []
        dst_points = []
        
        # 检查关键点
        core_points = self.FOREHEAD_CENTER + self.FOREHEAD_LEFT + self.FOREHEAD_RIGHT
        if any(idx >= len(landmarks_2d) for idx in core_points):
            print("❌ 找不到核心额头控制点")
            return image
        
        # 固定中央脊椎
        for idx in self.FOREHEAD_CENTER:
            point = landmarks_2d[idx]
            src_points.append(point)
            dst_points.append(point)
        
        # 计算额头3D变形中心
        forehead_center = self.calculate_forehead_center(landmarks_2d)
        if forehead_center is None:
            print("❌ 无法计算额头中心")
            return image
        
        # 3D姿态自适应系数
        if pose_3d == "left_profile":
            left_multiplier = 1.2
            right_multiplier = 0.8
        elif pose_3d == "right_profile":
            left_multiplier = 0.8
            right_multiplier = 1.2
        else:
            left_multiplier = right_multiplier = 1.0
        
        print(f"🎯 额头3D变形 - 借鉴鼻子3D成功经验")
        
        if transform_type == "shrink":
            print(f"🔵 额头饱满效果 - 中央隆起 + 径向扩展")
            
            # 发际线点：向上提升 + 径向扩展（双重3D效果）
            for idx in self.HAIRLINE_POINTS:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    # 先垂直提升
                    lifted_point = self.apply_vertical_lift(point, forehead_center, intensity, "up")
                    # 再径向扩展
                    final_point = self.apply_radial_transform(lifted_point, forehead_center, intensity, "expand")
                    
                    dst_points.append(final_point)
            
            # 左侧额头点：主要径向扩展
            for idx in self.FOREHEAD_LEFT:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    # 径向扩展 + 3D姿态调整
                    radial_point = self.apply_radial_transform(point, forehead_center, intensity * left_multiplier, "expand")
                    # 轻微上提
                    final_point = self.apply_vertical_lift(radial_point, forehead_center, intensity * 0.5, "up")
                    
                    dst_points.append(final_point)
            
            # 右侧额头点：主要径向扩展
            for idx in self.FOREHEAD_RIGHT:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    # 径向扩展 + 3D姿态调整
                    radial_point = self.apply_radial_transform(point, forehead_center, intensity * right_multiplier, "expand")
                    # 轻微上提
                    final_point = self.apply_vertical_lift(radial_point, forehead_center, intensity * 0.5, "up")
                    
                    dst_points.append(final_point)
        
        elif transform_type == "expand":
            print(f"🔶 额头平整效果 - 中央压平 + 径向收缩")
            
            # 发际线点：向下压平 + 径向收缩
            for idx in self.HAIRLINE_POINTS:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    # 先垂直压平
                    flattened_point = self.apply_vertical_lift(point, forehead_center, intensity, "down")
                    # 再径向收缩
                    final_point = self.apply_radial_transform(flattened_point, forehead_center, intensity, "shrink")
                    
                    dst_points.append(final_point)
            
            # 左侧额头点：主要径向收缩
            for idx in self.FOREHEAD_LEFT:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    # 径向收缩 + 3D姿态调整
                    radial_point = self.apply_radial_transform(point, forehead_center, intensity * left_multiplier, "shrink")
                    # 轻微下压
                    final_point = self.apply_vertical_lift(radial_point, forehead_center, intensity * 0.3, "down")
                    
                    dst_points.append(final_point)
            
            # 右侧额头点：主要径向收缩
            for idx in self.FOREHEAD_RIGHT:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    # 径向收缩 + 3D姿态调整
                    radial_point = self.apply_radial_transform(point, forehead_center, intensity * right_multiplier, "shrink")
                    # 轻微下压
                    final_point = self.apply_vertical_lift(radial_point, forehead_center, intensity * 0.3, "down")
                    
                    dst_points.append(final_point)
        
        # 眉毛上方点轻微跟随（保持自然过渡）
        for idx in self.EYEBROW_UPPER:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                if transform_type == "shrink":  # 饱满时轻微上提
                    follow_point = self.apply_vertical_lift(point, forehead_center, intensity * 0.3, "up")
                else:  # 平整时轻微下压
                    follow_point = self.apply_vertical_lift(point, forehead_center, intensity * 0.2, "down")
                
                dst_points.append(follow_point)
        
        # 太阳穴轻微跟随
        for idx in self.TEMPLE_POINTS:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx]
                src_points.append(point)
                
                if transform_type == "shrink":  # 饱满时轻微外扩
                    temple_point = self.apply_radial_transform(point, forehead_center, intensity * 0.3, "expand")
                else:  # 平整时轻微内收
                    temple_point = self.apply_radial_transform(point, forehead_center, intensity * 0.2, "shrink")
                
                dst_points.append(temple_point)
        
        # 添加稳定锚点
        anchor_count = 0
        for idx in self.STABLE_ANCHORS:
            if idx < len(landmarks_2d):
                src_points.append(landmarks_2d[idx])
                dst_points.append(landmarks_2d[idx])
                anchor_count += 1
        
        # 边界锚点
        h, w = image.shape[:2]
        boundary_points = [
            [20, 20], [w//2, 20], [w-20, 20],
            [20, h//2], [w-20, h//2],
            [20, h-20], [w//2, h-20], [w-20, h-20]
        ]
        
        for bp in boundary_points:
            src_points.append(bp)
            dst_points.append(bp)
        
        forehead_control_count = len(self.HAIRLINE_POINTS + self.FOREHEAD_LEFT + self.FOREHEAD_RIGHT + 
                                    self.EYEBROW_UPPER + self.TEMPLE_POINTS)
        
        print(f"🔧 3D变形控制点统计:")
        print(f"   中央脊椎: 3个(固定)")
        print(f"   额头控制点: {forehead_control_count}个(3D变形)")
        print(f"   稳定锚点: {anchor_count}个")
        print(f"   边界锚点: 8个")
        print(f"   总计: {len(src_points)}个")
        
        if len(src_points) < 15:
            print("❌ 控制点不足")
            return image
        
        # 执行TPS变换
        try:
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            
            # 检查移动距离
            distances = np.linalg.norm(dst_points - src_points, axis=1)
            max_distance = np.max(distances)
            moving_count = np.sum(distances > 0.5)
            
            print(f"📊 3D变形统计:")
            print(f"   最大移动: {max_distance:.2f}像素")
            print(f"   移动点数: {moving_count}")
            
            # TPS变换
            tps = cv2.createThinPlateSplineShapeTransformer()
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
            
            print("🔄 开始3D TPS变换...")
            
            # 正确的参数顺序
            tps.estimateTransformation(
                dst_points.reshape(1, -1, 2),
                src_points.reshape(1, -1, 2),
                matches
            )
            
            transformed = tps.warpImage(image)
            
            if transformed is None:
                print("❌ TPS变换返回空结果")
                return image
            
            # 异常像素修复
            green_mask = ((transformed[:, :, 0] == 0) & 
                        (transformed[:, :, 1] == 255) & 
                        (transformed[:, :, 2] == 0))
            if np.sum(green_mask) > 0:
                transformed[green_mask] = image[green_mask]
            
            gray_transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
            gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dark_mask = (gray_transformed < 10) & (gray_original > 50)
            if np.sum(dark_mask) > 0:
                transformed[dark_mask] = image[dark_mask]
            
            # 创建mask并融合
            forehead_mask = self.create_forehead_mask(image.shape, landmarks_2d)
            mask_3d = cv2.merge([forehead_mask, forehead_mask, forehead_mask]).astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_3d, (15, 15), 0)
            
            result = image.astype(np.float32) * (1 - mask_blurred) + transformed.astype(np.float32) * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print("✅ 3D额头变形成功")
            return result
            
        except Exception as e:
            print(f"❌ TPS变形异常: {e}")
            return image
        
    def debug_forehead_detection(self, image):
        """调试模式 - 显示精确的额头点"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        debug_img = image.copy()
        
        # 绘制中央脊椎 - 红色
        for i, idx in enumerate(self.FOREHEAD_CENTER):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 10, (0, 0, 255), -1)
                cv2.putText(debug_img, f"C{idx}", tuple(point + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 绘制发际线点 - 绿色
        for i, idx in enumerate(self.HAIRLINE_POINTS):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 8, (0, 255, 0), -1)
                cv2.putText(debug_img, f"H{idx}", tuple(point + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制左右额头主点 - 蓝色
        for i, idx in enumerate(self.FOREHEAD_LEFT + self.FOREHEAD_RIGHT):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 8, (255, 0, 0), -1)
                cv2.putText(debug_img, f"F{idx}", tuple(point + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 绘制眉毛上方点 - 黄色
        for i, idx in enumerate(self.EYEBROW_UPPER):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 6, (0, 255, 255), -1)
                cv2.putText(debug_img, f"E{idx}", tuple(point + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 绘制太阳穴点 - 紫色
        for i, idx in enumerate(self.TEMPLE_POINTS):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 8, (255, 0, 255), -1)
                cv2.putText(debug_img, f"T{idx}", tuple(point + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # 显示3D姿态
        if pose_info is not None:
            rotation_vector, translation_vector = pose_info
            pose_3d = self.estimate_face_pose_3d(rotation_vector)
            cv2.putText(debug_img, f"3D Pose: {pose_3d}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示mask
        mask = self.create_forehead_mask(image.shape, landmarks_2d)
        mask_overlay = debug_img.copy()
        mask_overlay[mask > 0] = [255, 255, 255]
        debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)
        
        # 图例
        legend_y = 70
        cv2.putText(debug_img, "RED: Center Spine (Fixed)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(debug_img, "GREEN: Hairline (Main Effect)", (10, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug_img, "BLUE: Forehead Main Points", (10, legend_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_img, "YELLOW: Eyebrow Upper", (10, legend_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(debug_img, "PURPLE: Temple Points", (10, legend_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        return debug_img
    
    def process_image(self, image, transform_type="shrink", intensity=0.3):
        """处理图像"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        print(f"✅ 检测到 {len(landmarks_2d)} 个2D关键点")
        
        return self.apply_effective_forehead_transform(image, landmarks_2d, pose_info, transform_type, intensity)
def main():
   parser = argparse.ArgumentParser(description='额头调整工具 - 精确有效版本')
   parser.add_argument('--input', '-i', required=True, help='输入图片路径')
   parser.add_argument('--output', '-o', required=True, help='输出图片路径')
   parser.add_argument('--type', '-t', choices=['shrink', 'expand'], 
                   default='shrink', help='调整类型：shrink=饱满效果, expand=平整效果')
   parser.add_argument('--intensity', type=float, default=0.3, 
                      help='调整强度 (0.1-0.8)')
   parser.add_argument('--debug', action='store_true', help='调试模式')
   parser.add_argument('--comparison', action='store_true', help='生成对比图')
   
   args = parser.parse_args()
   
   # 加载图片
   image = cv2.imread(args.input)
   if image is None:
       print(f"❌ 无法加载图片: {args.input}")
       return
   
   print(f"📸 图片尺寸: {image.shape}")
   
   # 强度检查
   if args.intensity > 0.8:
       print(f"⚠️  强度过大，限制到0.8")
       args.intensity = 0.8
   elif args.intensity < 0.1:
       print(f"⚠️  强度过小，提升到0.1")
       args.intensity = 0.1
   
   # 创建处理器
   processor = ForeheadAdjustment()
   
   # 调试模式
   if args.debug:
       debug_result = processor.debug_forehead_detection(image)
       cv2.imwrite(args.output, debug_result)
       print(f"🔍 调试图保存到: {args.output}")
       return
   
   # 处理图片
   print(f"🔄 开始精确额头处理 - 类型: {args.type}, 强度: {args.intensity}")
   result = processor.process_image(image, args.type, args.intensity)
   
   # 保存结果
   cv2.imwrite(args.output, result)
   print(f"✅ 处理完成，保存到: {args.output}")
   
   # 统计变化
   diff = cv2.absdiff(image, result)
   total_diff = np.sum(diff)
   changed_pixels = np.sum(diff > 0)
   print(f"📊 变化统计: 总差异={total_diff}, 变化像素={changed_pixels}")
   
   # 生成对比图
   if args.comparison:
       comparison_path = args.output.replace('.', '_comparison.')
       comparison = np.hstack([image, result])
       cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
       
       cv2.putText(comparison, 'Original', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       cv2.putText(comparison, f'Hairline-{args.type}-{args.intensity}', (image.shape[1] + 20, 40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       
       cv2.imwrite(comparison_path, comparison)
       print(f"📊 对比图保存到: {comparison_path}")
   
print("\n🎯 3D变形策略:")
print("• 饱满(shrink)：中央隆起 + 径向扩展 (模拟3D凸出)")
print("• 平整(expand)：中央压平 + 径向收缩 (模拟3D扁平)")
print("• 借鉴鼻子3D脚本的立体变形技术")
print("• 垂直提升 + 径向变形 = 双重3D效果")

print("\n📖 使用示例:")
print("• 额头饱满: python forehead_adjust.py -i input.jpg -o output.png -t shrink --intensity 0.4 --comparison")
print("• 额头平整: python forehead_adjust.py -i input.jpg -o output.png -t expand --intensity 0.3 --comparison")
print("• 3D调试: python forehead_adjust.py -i input.jpg -o debug.png --debug")

if __name__ == "__main__":
   main()
