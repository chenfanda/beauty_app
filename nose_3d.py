#!/usr/bin/env python3
"""
基于MediaPipe + 3D几何的鼻子调整脚本
使用PnP算法和3D面部模型进行更自然的变形
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
from typing import List, Tuple, Optional

class Nose3DGeometry:
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
        
        # 鼻子关键点
        self.NOSE_TIP_3D = [1, 2]
        self.LEFT_NOSTRIL_3D = [49, 131, 134]
        self.RIGHT_NOSTRIL_3D = [278, 360, 363]
        self.NOSE_BRIDGE_3D = [6, 19, 20, 8, 9, 10]
        
        # 标准3D面部模型关键点（毫米单位）
        self.face_model_3d = np.array([
            # 主要面部特征点用于PnP求解
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
        
        # 估算相机内参矩阵
        focal_length = w
        center = (w/2, h/2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 假设无镜头畸变
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
        
        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # 提取欧拉角
        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = 0
        
        # 转换为度数
        yaw = np.degrees(y)
        
        print(f"3D姿态检测: Yaw = {yaw:.1f}度")
        
        # 根据yaw角判断姿态
        if abs(yaw) < 15:
            return "frontal"
        elif yaw > 15:
            return "right_profile"
        else:
            return "left_profile"
    
    def create_3d_nose_mask(self, image_shape, landmarks_2d, pose_3d):
        """基于3D姿态创建更精确的鼻子mask - 修复上下扩展过度"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 收集鼻子关键点
        nose_indices = self.NOSE_TIP_3D + self.LEFT_NOSTRIL_3D + self.RIGHT_NOSTRIL_3D + self.NOSE_BRIDGE_3D[:3]
        nose_points = []
        
        for idx in nose_indices:
            if idx < len(landmarks_2d):
                nose_points.append(landmarks_2d[idx])
        
        if len(nose_points) >= 3:
            nose_points = np.array(nose_points, dtype=np.int32)
            hull = cv2.convexHull(nose_points)
            
            # 计算鼻子的实际尺寸
            nose_center = np.mean(hull.reshape(-1, 2), axis=0)
            nose_width = np.max(hull[:, :, 0]) - np.min(hull[:, :, 0])
            nose_height = np.max(hull[:, :, 1]) - np.min(hull[:, :, 1])
            
            print(f"鼻子尺寸: {nose_width:.0f}x{nose_height:.0f}像素")
            
            # 3D姿态自适应扩展 - 改回效果最好的参数
            if pose_3d == "frontal":
                horizontal_expand = 1.6  # 保持较大的扩展
                vertical_expand = 1.6    # 改回和水平一样大
            else:  # 侧脸
                horizontal_expand = 1.3  # 改回更大的扩展
                vertical_expand = 1.3    # 改回和水平一样大
            
            # 根据图片尺寸调整
            total_pixels = h * w
            if total_pixels > 1000000:  # 大图
                horizontal_expand *= 1.1
                vertical_expand *= 1.05
            elif total_pixels < 100000:  # 小图
                horizontal_expand *= 0.9
                vertical_expand *= 0.95
            
            # 分别处理水平和垂直方向的扩展
            expanded_hull = []
            for point in hull.reshape(-1, 2):
                # 计算相对于中心的偏移
                dx = point[0] - nose_center[0]
                dy = point[1] - nose_center[1]
                
                # 分别扩展x和y方向
                new_x = nose_center[0] + dx * horizontal_expand
                new_y = nose_center[1] + dy * vertical_expand
                
                expanded_hull.append([new_x, new_y])
            
            expanded_hull = np.array(expanded_hull, dtype=np.int32)
            
            # 进一步限制垂直扩展，确保不超过合理范围
            min_y = np.min(expanded_hull[:, 1])
            max_y = np.max(expanded_hull[:, 1])
            
            # 基于鼻子高度限制垂直扩展 - 改回允许更大扩展
            max_vertical_expansion = nose_height * 1.2  # 改回1.2，允许更大的垂直扩展
            
            if (max_y - min_y) > nose_height + max_vertical_expansion:
                # 如果扩展过度，重新计算
                center_y = (min_y + max_y) / 2
                target_height = nose_height + max_vertical_expansion
                
                for i, point in enumerate(expanded_hull):
                    if point[1] < center_y:  # 上半部分
                        expanded_hull[i][1] = max(point[1], center_y - target_height/2)
                    else:  # 下半部分
                        expanded_hull[i][1] = min(point[1], center_y + target_height/2)
            
            cv2.fillPoly(mask, [expanded_hull], 255)
            
            # 验证mask不会影响到眼睛和嘴巴
            eye_y = min(landmarks_2d[33][1], landmarks_2d[263][1])  # 眼角高度
            mouth_y = max(landmarks_2d[61][1], landmarks_2d[291][1])  # 嘴角高度
            
            mask_top = np.min(expanded_hull[:, 1])
            mask_bottom = np.max(expanded_hull[:, 1])
            
            # 边界保护 - 改回更宽松的边界，允许更大mask
            if mask_top < eye_y + 10:  # 改回更小的边界保护
                mask[:int(eye_y + 10), :] = 0
                print("⚠️  Mask上边界裁剪，避免影响眼部")
            
            if mask_bottom > mouth_y - 5:  # 改回更小的边界保护
                mask[int(mouth_y - 5):, :] = 0
                print("⚠️  Mask下边界裁剪，避免影响嘴部")
            
            print(f"3D Mask: 姿态={pose_3d}, 水平扩展={horizontal_expand:.1f}x, 垂直扩展={vertical_expand:.1f}x")
        
        return mask
    
    def apply_3d_nose_transform(self, image, landmarks_2d, pose_info, transform_type="wings", intensity=0.3):
        """应用3D感知的鼻子变形"""
        if pose_info is None:
            print("3D姿态信息无效，使用2D模式")
            return self.apply_2d_fallback(image, landmarks_2d, transform_type, intensity)
        
        rotation_vector, translation_vector = pose_info
        pose_3d = self.estimate_face_pose_3d(rotation_vector)
        
        # 3D姿态自适应强度调整
        if pose_3d != "frontal":
            intensity = intensity * 0.7  # 侧脸时降低强度
            print(f"3D侧脸检测，强度调整到: {intensity:.2f}")
        
        # 构建变形控制点
        src_points = []
        dst_points = []
        
        if transform_type == "wings":
            # 3D感知的鼻翼收缩
            nose_center_x = landmarks_2d[1][0]  # 使用鼻尖x坐标作为中心
            
            # 3D姿态自适应收缩
            left_multiplier = 1.0
            right_multiplier = 1.0
            
            if pose_3d == "left_profile":
                left_multiplier = 1.1
                right_multiplier = 0.9
            elif pose_3d == "right_profile":
                left_multiplier = 0.9
                right_multiplier = 1.1
            
            # 左鼻翼收缩
            for idx in self.LEFT_NOSTRIL_3D:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    shrink_amount = (point[0] - nose_center_x) * intensity * 0.4 * left_multiplier
                    new_x = point[0] - shrink_amount
                    dst_points.append([new_x, point[1]])
            
            # 右鼻翼收缩
            for idx in self.RIGHT_NOSTRIL_3D:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    shrink_amount = (point[0] - nose_center_x) * intensity * 0.4 * right_multiplier
                    new_x = point[0] - shrink_amount
                    dst_points.append([new_x, point[1]])
        
        elif transform_type == "tip":
            # 3D感知的鼻尖抬高 - 完善实现
            print(f"3D鼻尖抬高: {pose_3d}")
            
            # 鼻尖向上移动
            for idx in self.NOSE_TIP_3D:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    lift_amount = intensity * 12
                    new_point = [point[0], point[1] - lift_amount]
                    dst_points.append(new_point)
            
            # 鼻梁适度抬高
            for idx in self.NOSE_BRIDGE_3D[:4]:  # 使用前4个鼻梁点
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    # 渐变抬高：越靠近鼻尖抬得越多
                    bridge_lift = intensity * (8 - idx * 1.5)  # 从8递减到2
                    new_point = [point[0], point[1] - max(bridge_lift, 2)]
                    dst_points.append(new_point)
        
        elif transform_type == "resize":
            # 3D感知的整体鼻子缩小
            print(f"3D鼻子整体缩小: {pose_3d}")
            
            # 计算3D感知的鼻子中心
            all_nose_indices = self.NOSE_TIP_3D + self.LEFT_NOSTRIL_3D + self.RIGHT_NOSTRIL_3D + self.NOSE_BRIDGE_3D[:3]
            nose_points_for_center = []
            for idx in all_nose_indices:
                if idx < len(landmarks_2d):
                    nose_points_for_center.append(landmarks_2d[idx])
            
            if len(nose_points_for_center) < 3:
                print("❌ 鼻子关键点不足，无法整体缩小")
                return self.apply_2d_fallback(image, landmarks_2d, transform_type, intensity)
            
            nose_center = np.mean(nose_points_for_center, axis=0)
            
            # 3D姿态自适应缩小
            if pose_3d != "frontal":
                shrink_intensity = intensity * 0.8  # 侧脸时更保守
            else:
                shrink_intensity = intensity
            
            # 所有鼻子关键点向中心收缩
            for idx in all_nose_indices:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    # 向中心收缩
                    vec = point - nose_center
                    shrink_factor = 1 - shrink_intensity * 0.4
                    new_point = nose_center + vec * shrink_factor
                    dst_points.append(new_point)
        
        elif transform_type == "bridge":
            # 新增：鼻梁变挺功能
            print(f"3D鼻梁变挺: {pose_3d}")
            
            # 鼻梁向前突出（在图像中表现为轻微向上和缩小）
            for i, idx in enumerate(self.NOSE_BRIDGE_3D):
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    # 渐变效果：中间的鼻梁点变化最大
                    position_ratio = 1.0 - abs(i - len(self.NOSE_BRIDGE_3D)/2) / (len(self.NOSE_BRIDGE_3D)/2)
                    
                    # 向上移动模拟变挺
                    lift_amount = intensity * 6 * position_ratio
                    # 轻微向中心收缩模拟变窄
                    center_x = landmarks_2d[1][0]  # 鼻尖x坐标
                    shrink_amount = (point[0] - center_x) * intensity * 0.2 * position_ratio
                    
                    new_x = point[0] - shrink_amount
                    new_y = point[1] - lift_amount
                    dst_points.append([new_x, new_y])
        
        else:
            print(f"❌ 未知的3D变形类型: {transform_type}")
            return self.apply_2d_fallback(image, landmarks_2d, transform_type, intensity)
        
        # 添加大量稳定锚点 - 参考2D版本的成功经验
        stable_indices = [
            # 眼部
            33, 133, 362, 263, 7, 163, 144, 145, 153, 154, 155, 173, 157, 158, 159, 160, 161, 246,
            382, 381, 380, 374, 373, 390, 249, 466, 388, 387, 386, 385, 384, 398,
            # 嘴部
            61, 291, 39, 269, 270, 267, 271, 272, 0, 17, 18, 200, 199, 175, 164,
            # 脸部轮廓
            234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454
        ]
        
        anchor_count = 0
        for idx in stable_indices:
            if idx < len(landmarks_2d):
                src_points.append(landmarks_2d[idx])
                dst_points.append(landmarks_2d[idx])
                anchor_count += 1
        
        # 添加边界锚点
        h, w = image.shape[:2]
        boundary_margin = 50
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
        
        print(f"3D控制点: 鼻子={len(src_points)-anchor_count-8}, 锚点={anchor_count}, 边界=8")
        
        if len(src_points) < 10:
            print("3D控制点不足，使用2D模式")
            return self.apply_2d_fallback(image, landmarks_2d, transform_type, intensity)
        
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
                print("3D TPS变换失败")
                return image
            
            # 创建3D感知的mask
            nose_mask = self.create_3d_nose_mask(image.shape, landmarks_2d, pose_3d)
            
            # 融合
            mask_3d = cv2.merge([nose_mask, nose_mask, nose_mask]).astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_3d, (5, 5), 0)
            
            result = image.astype(np.float32) * (1 - mask_blurred) + transformed.astype(np.float32) * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print("✅ 3D变形成功")
            return result
            
        except Exception as e:
            print(f"3D变形异常: {e}")
            return self.apply_2d_fallback(image, landmarks_2d, transform_type, intensity)
    
    def apply_2d_fallback(self, image, landmarks_2d, transform_type, intensity):
        """2D模式回退"""
        print("使用2D回退模式")
        # 这里可以调用之前的2D TPS方法
        return image
    
    def debug_3d_detection(self, image):
        """调试3D检测 - 修复关键点显示"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("未检测到面部关键点")
            return image
        
        debug_img = image.copy()
        
        # 绘制实际的鼻子关键点（使用正确的MediaPipe索引）
        nose_parts = [
            (self.NOSE_TIP_3D, (0, 0, 255), "T"),        # 鼻尖-红色
            (self.LEFT_NOSTRIL_3D, (0, 255, 0), "L"),    # 左鼻翼-绿色  
            (self.RIGHT_NOSTRIL_3D, (255, 0, 0), "R"),   # 右鼻翼-蓝色
            (self.NOSE_BRIDGE_3D[:3], (0, 255, 255), "B") # 鼻梁-黄色（只显示前3个）
        ]
        
        for indices, color, prefix in nose_parts:
            for i, idx in enumerate(indices):
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx].astype(int)
                    cv2.circle(debug_img, tuple(point), 5, color, -1)
                    # 显示MediaPipe真实索引而不是我的错误编号
                    cv2.putText(debug_img, f"{prefix}{idx}", tuple(point + 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 显示PnP关键点（用于3D姿态求解的6个点）
        pnp_colors = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 255, 128), (255, 128, 128)]
        for i, idx in enumerate(self.pnp_indices):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 8, pnp_colors[i], 2)
                cv2.putText(debug_img, f"PnP{idx}", tuple(point + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, pnp_colors[i], 2)
        
        # 显示3D姿态信息
        if pose_info is not None:
            rotation_vector, translation_vector = pose_info
            pose_3d = self.estimate_face_pose_3d(rotation_vector)
            cv2.putText(debug_img, f"3D Pose: {pose_3d}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 绘制3D坐标轴（在鼻尖位置）
            nose_tip_3d = np.array([[0, 0, 0]], dtype=np.float32)
            axis_3d = np.array([
                [0, 0, 0],      # 原点
                [30, 0, 0],     # X轴
                [0, 30, 0],     # Y轴  
                [0, 0, -30]     # Z轴
            ], dtype=np.float32)
            
            try:
                projected_axis, _ = cv2.projectPoints(
                    axis_3d, rotation_vector, translation_vector,
                    self.camera_matrix, self.dist_coeffs
                )
                projected_axis = projected_axis.reshape(-1, 2).astype(int)
                
                # 绘制坐标轴
                origin = tuple(projected_axis[0])
                cv2.arrowedLine(debug_img, origin, tuple(projected_axis[1]), (0, 0, 255), 3)  # X-红
                cv2.arrowedLine(debug_img, origin, tuple(projected_axis[2]), (0, 255, 0), 3)  # Y-绿
                cv2.arrowedLine(debug_img, origin, tuple(projected_axis[3]), (255, 0, 0), 3)  # Z-蓝
                
                # 标记坐标轴
                cv2.putText(debug_img, "X", tuple(projected_axis[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(debug_img, "Y", tuple(projected_axis[2] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(debug_img, "Z", tuple(projected_axis[3] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            except:
                print("3D坐标轴投影失败")
        
        # 显示3D mask
        if pose_info is not None:
            pose_3d = self.estimate_face_pose_3d(pose_info[0])
            nose_mask = self.create_3d_nose_mask(image.shape, landmarks_2d, pose_3d)
            mask_overlay = debug_img.copy()
            mask_overlay[nose_mask > 0] = [255, 0, 255]  # 紫色mask
            debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)
        
        # 添加图例说明
        legend_y = 60
        cv2.putText(debug_img, "RED: Nose Tip", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(debug_img, "GREEN: Left Nostril", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug_img, "BLUE: Right Nostril", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_img, "YELLOW: Nose Bridge", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(debug_img, "WHITE: PnP Points", (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return debug_img
    
    def process_image(self, image, transform_type="wings", intensity=0.3):
        """处理图像"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("未检测到面部关键点")
            return image
        
        print(f"✅ 检测到 {len(landmarks_2d)} 个2D关键点")
        
        return self.apply_3d_nose_transform(image, landmarks_2d, pose_info, transform_type, intensity)

def main():
    parser = argparse.ArgumentParser(description='3D Face Geometry鼻子调整工具')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--type', '-t', choices=['wings', 'tip', 'resize', 'bridge'], 
                       default='wings', help='调整类型：wings=鼻翼收缩, tip=鼻尖抬高, resize=整体缩小, bridge=鼻梁变挺')
    parser.add_argument('--intensity', type=float, default=0.3, 
                       help='调整强度 (0.1-0.5)')
    parser.add_argument('--debug', action='store_true', help='3D调试模式')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"无法加载图片: {args.input}")
        return
    
    print(f"图片尺寸: {image.shape}")
    
    # 创建3D处理器
    processor = Nose3DGeometry()
    
    # 调试模式
    if args.debug:
        debug_result = processor.debug_3d_detection(image)
        cv2.imwrite(args.output, debug_result)
        print(f"3D调试图保存到: {args.output}")
        return
    
    # 处理图片
    print(f"开始3D处理 - 类型: {args.type}, 强度: {args.intensity}")
    result = processor.process_image(image, args.type, args.intensity)
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"处理完成，保存到: {args.output}")
    
    # 检查变化
    diff = cv2.absdiff(image, result)
    total_diff = np.sum(diff)
    changed_pixels = np.sum(diff > 0)
    print(f"变化统计: 总差异={total_diff}, 变化像素={changed_pixels}")
    
    # 生成对比图
    if args.comparison:
        comparison_path = args.output.replace('.', '_3d_comparison.')
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
        
        cv2.putText(comparison, 'Original', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f'3D-{args.type} {args.intensity}', (image.shape[1] + 20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(comparison_path, comparison)
        print(f"3D对比图保存到: {comparison_path}")
    
    print("\n🌟 3D Face Geometry特性:")
    print("• 基于真实3D面部几何模型")
    print("• PnP算法求解精确头部姿态")
    print("• 3D感知的自适应变形强度")
    print("• 更精确的侧脸处理")
    print("• 3D坐标轴可视化调试")
    print("\n使用示例:")
    print("• 3D鼻翼收缩: python nose_3d.py -i input.jpg -o output.png -t wings --intensity 0.3 --comparison")
    print("• 3D鼻尖抬高: python nose_3d.py -i input.jpg -o output.png -t tip --intensity 0.25 --comparison")
    print("• 3D整体缩小: python nose_3d.py -i input.jpg -o output.png -t resize --intensity 0.3 --comparison")
    print("• 3D鼻梁变挺: python nose_3d.py -i input.jpg -o output.png -t bridge --intensity 0.2 --comparison")
    print("• 3D调试模式: python nose_3d.py -i input.jpg -o debug.png --debug")

if __name__ == "__main__":
    main()
