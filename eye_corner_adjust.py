#!/usr/bin/env python3
"""
重构的眼角调整脚本 - eye_corner_adjust_fixed.py
基于正确原则：远程锚点 + 医学美学参考 + 去重机制 + 参考眼睛脚本成功模式
功能定位：眼型比例调整（与eye_adjust.py的整体放大区分）
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
from typing import List, Tuple, Optional

class FixedEyeCornerAdjustment:
    def __init__(self):
        # MediaPipe设置（参考眼睛脚本的稳定配置）
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
        
        # 🎯 修正：基于正确眼角解剖学的关键点设计
        
        # 医学美学参考点（基于眼球中心）
        self.LEFT_EYE_CENTER_REFS = [33, 133]   # 左眼内外眦
        self.RIGHT_EYE_CENTER_REFS = [362, 263] # 右眼内外眦
        
        # 🔧 修正：正确的眼角移动点 - 围绕眼角形成完整控制
        # 左眼角：内外眦 + 上下眼睑关键点
        self.LEFT_EYE_CORNER_POINTS = [
            133,     # 左眼内眦（最重要的眼角点）
            33,      # 左眼外眦（最重要的眼角点）
            157,     # 左眼上眼睑外侧
            159,     # 左眼上眼睑内侧
        ]
        
        self.LEFT_EYE_CORNER_AUX = [
            158,     # 左眼上眼睑中部
            160,     # 左眼下眼睑内侧  
            161,     # 左眼下眼睑中部
            163      # 左眼下眼睑外侧
        ]
        
        # 右眼角：对称设计
        self.RIGHT_EYE_CORNER_POINTS = [
            362,     # 右眼内眦（最重要的眼角点）
            263,     # 右眼外眦（最重要的眼角点）
            386,     # 右眼上眼睑外侧
            388,     # 右眼上眼睑内侧
        ]
        
        self.RIGHT_EYE_CORNER_AUX = [
            387,     # 右眼上眼睑中部
            385,     # 右眼下眼睑内侧
            384,     # 右眼下眼睑中部
            390      # 右眼下眼睑外侧
        ]
        
        # 🔒 远程稳定锚点（参考眼睛脚本的成功设计）
        self.STABLE_ANCHORS = [
            # 鼻子区域（远离眼部）
            1, 2, 5, 4, 6, 19, 20,
            # 额头和下巴（远离变形区）
            9, 10, 151, 175,
            # 嘴巴区域（稳定区域）
            61, 291, 39, 269, 270,
            # 脸颊边缘（远程稳定）
            234, 93, 132, 454, 323, 361,
            # 眉毛区域（距离眼角足够远）
            70, 296, 107, 276
        ]
        
        # 对称辅助点（确保左右平衡）
        self.SYMMETRY_HELPERS = [168, 6, 8, 9]  # 面部中线稳定点
        
        # 3D姿态用的标准点
        self.face_model_3d = np.array([
            [0.0, 0.0, 0.0],           # 鼻尖 (1)
            [0.0, -330.0, -65.0],      # 下巴 (152)  
            [-225.0, 170.0, -135.0],   # 左眼外角 (33)
            [225.0, 170.0, -135.0],    # 右眼外角 (263)
            [-150.0, -150.0, -125.0],  # 左嘴角 (61)
            [150.0, -150.0, -125.0],   # 右嘴角 (291)
        ], dtype=np.float32)
        
        self.pnp_indices = [1, 152, 33, 263, 61, 291]
        
        # 🛡️ 简化的安全参数（参考眼睛脚本）
        self.MAX_INTENSITY = 0.4
        self.MAX_DISPLACEMENT = 40        # 眼角调整相对保守
        self.MIN_EYE_CORNER_AREA = 400
    
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
            return landmarks_2d, None
        
        image_points = np.array(image_points, dtype=np.float32)
        
        # PnP求解3D姿态
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.face_model_3d, image_points, 
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                return landmarks_2d, (rotation_vector, translation_vector)
            else:
                return landmarks_2d, None
                
        except Exception as e:
            print(f"⚠️  PnP求解异常: {e}")
            return landmarks_2d, None
    
    def estimate_face_pose_3d(self, rotation_vector):
        """基于3D旋转向量估计面部姿态"""
        if rotation_vector is None:
            return "frontal", 0.0
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        
        if sy > 1e-6:
            y = np.arctan2(-rotation_matrix[2,0], sy)
        else:
            y = np.arctan2(-rotation_matrix[2,0], sy)
        
        yaw = np.degrees(y)
        print(f"🔍 3D姿态检测: Yaw = {yaw:.1f}度")
        
        if abs(yaw) < 15:
            return "frontal", yaw
        elif yaw > 15:
            return "right_profile", yaw
        else:
            return "left_profile", yaw
    
    def calculate_eye_centers(self, landmarks_2d):
        """计算眼球中心参考点（参考眼睛脚本）"""
        try:
            # 计算眼球几何中心
            left_eye_center = (landmarks_2d[self.LEFT_EYE_CENTER_REFS[0]] + 
                             landmarks_2d[self.LEFT_EYE_CENTER_REFS[1]]) / 2
            right_eye_center = (landmarks_2d[self.RIGHT_EYE_CENTER_REFS[0]] + 
                              landmarks_2d[self.RIGHT_EYE_CENTER_REFS[1]]) / 2
            
            print(f"✅ 眼球中心参考点: 左({left_eye_center[0]:.0f},{left_eye_center[1]:.0f}), 右({right_eye_center[0]:.0f},{right_eye_center[1]:.0f})")
            return left_eye_center, right_eye_center
            
        except Exception as e:
            print(f"❌ 眼球中心计算失败: {e}")
            return None, None
    
    def validate_eye_corner_points(self, landmarks_2d):
        """简化的眼角点验证（参考眼睛脚本模式）"""
        try:
            # 检查所有关键点是否在有效范围内
            all_points = (self.LEFT_EYE_CORNER_POINTS + self.RIGHT_EYE_CORNER_POINTS + 
                         self.LEFT_EYE_CORNER_AUX + self.RIGHT_EYE_CORNER_AUX)
            
            valid_count = sum(1 for idx in all_points if idx < len(landmarks_2d))
            total_count = len(all_points)
            
            if valid_count < total_count * 0.8:  # 至少80%的点有效
                print(f"⚠️  眼角关键点不足: {valid_count}/{total_count}")
                return False
            
            # 检查左右眼角点数是否对称
            left_count = sum(1 for idx in self.LEFT_EYE_CORNER_POINTS + self.LEFT_EYE_CORNER_AUX 
                           if idx < len(landmarks_2d))
            right_count = sum(1 for idx in self.RIGHT_EYE_CORNER_POINTS + self.RIGHT_EYE_CORNER_AUX 
                            if idx < len(landmarks_2d))
            
            if abs(left_count - right_count) > 1:
                print(f"⚠️  左右眼角点数不对称: 左{left_count}, 右{right_count}")
                # 参考眼睛脚本：警告但不阻止
            
            print(f"✅ 眼角关键点验证: 左{left_count}个, 右{right_count}个, 总计{valid_count}/{total_count}")
            return True
            
        except Exception as e:
            print(f"❌ 眼角验证失败: {e}")
            return False
    
    def create_safe_eye_corner_mask(self, image_shape, landmarks_2d):
        """创建安全的眼角mask（参考眼睛脚本模式）"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 收集所有眼角点
        all_eye_corner_points = []
        
        for idx in (self.LEFT_EYE_CORNER_POINTS + self.RIGHT_EYE_CORNER_POINTS + 
                   self.LEFT_EYE_CORNER_AUX + self.RIGHT_EYE_CORNER_AUX):
            if idx < len(landmarks_2d):
                all_eye_corner_points.append(landmarks_2d[idx])
        
        if len(all_eye_corner_points) < 8:
            print("⚠️  眼角点不足，创建保守mask")
            return mask
        
        all_eye_corner_points = np.array(all_eye_corner_points, dtype=np.int32)
        
        # 参考眼睛脚本：为每个点创建小圆形区域
        for point in all_eye_corner_points:
            cv2.circle(mask, tuple(point), 10, 255, -1)  # 10像素半径的圆形
        
        # 轻微连接（参考眼睛脚本的处理）
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # 边界保护
        # 眉毛保护
        eyebrow_points = [70, 296]
        if all(idx < len(landmarks_2d) for idx in eyebrow_points):
            eyebrow_bottom = max([landmarks_2d[idx][1] for idx in eyebrow_points])
            mask[:int(eyebrow_bottom - 5), :] = 0
        
        # 鼻子保护
        nose_points = [1, 2]
        if all(idx < len(landmarks_2d) for idx in nose_points):
            nose_left = min([landmarks_2d[idx][0] for idx in nose_points]) - 15
            nose_right = max([landmarks_2d[idx][0] for idx in nose_points]) + 15
            nose_top = min([landmarks_2d[idx][1] for idx in nose_points]) - 10
            nose_bottom = max([landmarks_2d[idx][1] for idx in nose_points]) + 10
            mask[int(nose_top):int(nose_bottom), int(nose_left):int(nose_right)] = 0
        
        mask_area = np.sum(mask > 0)
        print(f"✅ 安全眼角Mask创建: 区域={mask_area}像素")
        return mask
    
    def apply_eye_corner_transform(self, image, landmarks_2d, pose_info, adjustment_type="stretch", intensity=0.3):
        """重新设计的眼角变形 - 基于参考点的径向移动"""
        
        # 3D姿态分析
        if pose_info is None:
            pose_3d, pose_angle = "frontal", 0.0
        else:
            rotation_vector, translation_vector = pose_info
            pose_3d, pose_angle = self.estimate_face_pose_3d(rotation_vector)
        
        # 🎯 建立统一的参考点系统
        left_eye_center, right_eye_center = self.calculate_eye_centers(landmarks_2d)
        if left_eye_center is None:
            print("❌ 无法建立眼球中心参考点")
            return image
        
        print(f"✅ 参考点: 左眼中心({left_eye_center[0]:.0f},{left_eye_center[1]:.0f}), 右眼中心({right_eye_center[0]:.0f},{right_eye_center[1]:.0f})")
        
        # 根据3D姿态调整强度
        if pose_3d == "frontal":
            left_intensity = right_intensity = intensity
        elif pose_3d == "right_profile":
            right_intensity = intensity * 1.0
            left_intensity = intensity * 0.7
        else:  # left_profile  
            left_intensity = intensity * 1.0
            right_intensity = intensity * 0.7
        
        print(f"🔧 3D自适应强度: 左眼={left_intensity:.2f}, 右眼={right_intensity:.2f}")
        
        # 构建控制点
        src_points = []
        dst_points = []
        used_indices = set()
        
        if adjustment_type == "stretch":
            print(f"↔️ 眼型径向拉长（基于眼球中心参考点）")
            
            # 🎯 左眼径向拉长
            for idx in self.LEFT_EYE_CORNER_POINTS + self.LEFT_EYE_CORNER_AUX:
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    # 计算从眼球中心到当前点的方向向量
                    direction = point - left_eye_center
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 1e-6:  # 避免除零
                        direction = direction / direction_norm  # 标准化方向向量
                        
                        # 🔧 根据点的类型调整拉长强度
                        if idx in [33, 133]:  # 内外眦：主要拉长点
                            stretch_amount = left_intensity * 12
                        elif idx in [157, 159, 158]:  # 上眼睑：中等拉长
                            stretch_amount = left_intensity * 8
                        else:  # 下眼睑：轻微拉长
                            stretch_amount = left_intensity * 6
                        
                        # 沿径向方向拉长
                        new_point = point + direction * stretch_amount
                        print(f"   左眼点{idx}: 方向({direction[0]:.2f},{direction[1]:.2f}), 移动{stretch_amount:.1f}像素")
                    else:
                        # 如果点恰好在中心，不移动
                        new_point = point
                        print(f"   左眼点{idx}: 在中心点，不移动")
                    
                    dst_points.append(new_point)
            
            # 🎯 右眼径向拉长（对称处理）
            for idx in self.RIGHT_EYE_CORNER_POINTS + self.RIGHT_EYE_CORNER_AUX:
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    # 计算从眼球中心到当前点的方向向量
                    direction = point - right_eye_center
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                        
                        # 根据点的类型调整拉长强度
                        if idx in [263, 362]:  # 内外眦：主要拉长点
                            stretch_amount = right_intensity * 12
                        elif idx in [386, 388, 387]:  # 上眼睑：中等拉长
                            stretch_amount = right_intensity * 8
                        else:  # 下眼睑：轻微拉长
                            stretch_amount = right_intensity * 6
                        
                        # 沿径向方向拉长
                        new_point = point + direction * stretch_amount
                        print(f"   右眼点{idx}: 方向({direction[0]:.2f},{direction[1]:.2f}), 移动{stretch_amount:.1f}像素")
                    else:
                        new_point = point
                        print(f"   右眼点{idx}: 在中心点，不移动")
                    
                    dst_points.append(new_point)
        
        elif adjustment_type == "lift":
            print(f"↗️ 眼角上扬（基于眼球中心 + 向上偏移）")
            
            # 🎯 左眼上扬：主要调整外眼角区域
            for idx in self.LEFT_EYE_CORNER_POINTS + self.LEFT_EYE_CORNER_AUX:
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    # 计算从眼球中心到当前点的方向向量
                    direction = point - left_eye_center
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                        
                        # 🔧 区分内外眼角的上扬策略
                        if idx == 33:  # 左眼外眦：主要上扬点
                            # 在原方向基础上增加向上分量
                            lift_direction = direction + np.array([0, -1]) * 0.8  # 加强向上
                            lift_direction = lift_direction / np.linalg.norm(lift_direction)
                            lift_amount = left_intensity * 15
                            print(f"   左眼外眦{idx}: 主要上扬点，移动{lift_amount:.1f}像素")
                        elif idx == 133:  # 左眼内眦：保持相对稳定
                            # 轻微向上，主要保持位置
                            lift_direction = direction + np.array([0, -1]) * 0.2
                            lift_direction = lift_direction / np.linalg.norm(lift_direction)
                            lift_amount = left_intensity * 3
                            print(f"   左眼内眦{idx}: 轻微调整，移动{lift_amount:.1f}像素")
                        elif idx in [157, 159, 158]:  # 上眼睑：跟随外眦
                            # 根据距离外眦的远近决定上扬程度
                            distance_to_outer = np.linalg.norm(point - landmarks_2d[33])
                            outer_influence = max(0.3, 1.0 - distance_to_outer / 30)  # 距离越近影响越大
                            lift_direction = direction + np.array([0, -1]) * (0.6 * outer_influence)
                            lift_direction = lift_direction / np.linalg.norm(lift_direction)
                            lift_amount = left_intensity * (8 + 4 * outer_influence)
                            print(f"   左眼上睑{idx}: 外眦影响{outer_influence:.2f}，移动{lift_amount:.1f}像素")
                        else:  # 下眼睑：轻微跟随
                            lift_direction = direction + np.array([0, -1]) * 0.3
                            lift_direction = lift_direction / np.linalg.norm(lift_direction)
                            lift_amount = left_intensity * 5
                            print(f"   左眼下睑{idx}: 轻微跟随，移动{lift_amount:.1f}像素")
                        
                        new_point = point + lift_direction * lift_amount
                    else:
                        new_point = point
                    
                    dst_points.append(new_point)
            
            # 🎯 右眼上扬（镜像对称处理）
            for idx in self.RIGHT_EYE_CORNER_POINTS + self.RIGHT_EYE_CORNER_AUX:
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    direction = point - right_eye_center
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                        
                        if idx == 263:  # 右眼外眦：主要上扬点
                            lift_direction = direction + np.array([0, -1]) * 0.8
                            lift_direction = lift_direction / np.linalg.norm(lift_direction)
                            lift_amount = right_intensity * 15
                            print(f"   右眼外眦{idx}: 主要上扬点，移动{lift_amount:.1f}像素")
                        elif idx == 362:  # 右眼内眦：保持相对稳定
                            lift_direction = direction + np.array([0, -1]) * 0.2
                            lift_direction = lift_direction / np.linalg.norm(lift_direction)
                            lift_amount = right_intensity * 3
                            print(f"   右眼内眦{idx}: 轻微调整，移动{lift_amount:.1f}像素")
                        elif idx in [386, 388, 387]:  # 上眼睑：跟随外眦
                            distance_to_outer = np.linalg.norm(point - landmarks_2d[263])
                            outer_influence = max(0.3, 1.0 - distance_to_outer / 30)
                            lift_direction = direction + np.array([0, -1]) * (0.6 * outer_influence)
                            lift_direction = lift_direction / np.linalg.norm(lift_direction)
                            lift_amount = right_intensity * (8 + 4 * outer_influence)
                            print(f"   右眼上睑{idx}: 外眦影响{outer_influence:.2f}，移动{lift_amount:.1f}像素")
                        else:  # 下眼睑：轻微跟随
                            lift_direction = direction + np.array([0, -1]) * 0.3
                            lift_direction = lift_direction / np.linalg.norm(lift_direction)
                            lift_amount = right_intensity * 5
                            print(f"   右眼下睑{idx}: 轻微跟随，移动{lift_amount:.1f}像素")
                        
                        new_point = point + lift_direction * lift_amount
                    else:
                        new_point = point
                    
                    dst_points.append(new_point)
        
        elif adjustment_type == "shape":
            print(f"👁️ 综合眼型调整（径向拉长 + 上扬组合）")
            
            # 🎯 左眼综合调整：拉长 + 上扬
            for idx in self.LEFT_EYE_CORNER_POINTS + self.LEFT_EYE_CORNER_AUX:
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    direction = point - left_eye_center
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                        
                        # 🔧 综合变形：径向拉长 + 向上分量
                        if idx == 33:  # 左眼外眦：最大变形
                            # 径向拉长分量
                            stretch_component = direction * (left_intensity * 10)
                            # 向上分量
                            lift_component = np.array([0, -1]) * (left_intensity * 10)
                            total_transform = stretch_component + lift_component
                            print(f"   左眼外眦{idx}: 径向拉长+上扬，总移动{np.linalg.norm(total_transform):.1f}像素")
                        elif idx == 133:  # 左眼内眦：适度变形
                            stretch_component = direction * (left_intensity * 6)
                            lift_component = np.array([0, -1]) * (left_intensity * 2)
                            total_transform = stretch_component + lift_component
                            print(f"   左眼内眦{idx}: 适度变形，总移动{np.linalg.norm(total_transform):.1f}像素")
                        elif idx in [157, 159, 158]:  # 上眼睑：中等变形
                            stretch_component = direction * (left_intensity * 7)
                            lift_component = np.array([0, -1]) * (left_intensity * 6)
                            total_transform = stretch_component + lift_component
                            print(f"   左眼上睑{idx}: 中等变形，总移动{np.linalg.norm(total_transform):.1f}像素")
                        else:  # 下眼睑：轻微变形
                            stretch_component = direction * (left_intensity * 5)
                            lift_component = np.array([0, -1]) * (left_intensity * 3)
                            total_transform = stretch_component + lift_component
                            print(f"   左眼下睑{idx}: 轻微变形，总移动{np.linalg.norm(total_transform):.1f}像素")
                        
                        new_point = point + total_transform
                    else:
                        new_point = point
                    
                    dst_points.append(new_point)
            
            # 🎯 右眼综合调整（镜像对称）
            for idx in self.RIGHT_EYE_CORNER_POINTS + self.RIGHT_EYE_CORNER_AUX:
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    direction = point - right_eye_center
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                        
                        if idx == 263:  # 右眼外眦：最大变形
                            stretch_component = direction * (right_intensity * 10)
                            lift_component = np.array([0, -1]) * (right_intensity * 10)
                            total_transform = stretch_component + lift_component
                            print(f"   右眼外眦{idx}: 径向拉长+上扬，总移动{np.linalg.norm(total_transform):.1f}像素")
                        elif idx == 362:  # 右眼内眦：适度变形
                            stretch_component = direction * (right_intensity * 6)
                            lift_component = np.array([0, -1]) * (right_intensity * 2)
                            total_transform = stretch_component + lift_component
                            print(f"   右眼内眦{idx}: 适度变形，总移动{np.linalg.norm(total_transform):.1f}像素")
                        elif idx in [386, 388, 387]:  # 上眼睑：中等变形
                            stretch_component = direction * (right_intensity * 7)
                            lift_component = np.array([0, -1]) * (right_intensity * 6)
                            total_transform = stretch_component + lift_component
                            print(f"   右眼上睑{idx}: 中等变形，总移动{np.linalg.norm(total_transform):.1f}像素")
                        else:  # 下眼睑：轻微变形
                            stretch_component = direction * (right_intensity * 5)
                            lift_component = np.array([0, -1]) * (right_intensity * 3)
                            total_transform = stretch_component + lift_component
                            print(f"   右眼下睑{idx}: 轻微变形，总移动{np.linalg.norm(total_transform):.1f}像素")
                        
                        new_point = point + total_transform
                    else:
                        new_point = point
                    
                    dst_points.append(new_point)
        
        else:
            print(f"❌ 未知的调整类型: {adjustment_type}")
            return image
        
        # 添加锚点（检查重复）
        anchor_count = 0
        for idx in self.STABLE_ANCHORS:
            if idx < len(landmarks_2d) and idx not in used_indices:
                src_points.append(landmarks_2d[idx])
                dst_points.append(landmarks_2d[idx])
                used_indices.add(idx)
                anchor_count += 1
        
        # 添加对称辅助点（检查重复）
        symmetry_count = 0
        for idx in self.SYMMETRY_HELPERS:
            if idx < len(landmarks_2d) and idx not in used_indices:
                src_points.append(landmarks_2d[idx])
                dst_points.append(landmarks_2d[idx])
                used_indices.add(idx)
                symmetry_count += 1
        
        # 添加边界锚点
        h, w = image.shape[:2]
        boundary_points = [
            [30, 30], [w//2, 30], [w-30, 30],
            [30, h//2], [w-30, h//2],
            [30, h-30], [w//2, h-30], [w-30, h-30]
        ]
        
        for bp in boundary_points:
            src_points.append(bp)
            dst_points.append(bp)
        
        print(f"🔧 控制点统计:")
        print(f"   眼角移动点: {len(src_points) - anchor_count - symmetry_count - 8}")
        print(f"   稳定锚点: {anchor_count}")
        print(f"   对称辅助点: {symmetry_count}")
        print(f"   边界锚点: 8")
        print(f"   总计: {len(src_points)} 个控制点")
        
        # 验证移动方向的连续性
        self.validate_movement_directions(src_points, dst_points, left_eye_center, right_eye_center)
        
        # 简化的安全检查
        if len(src_points) < 15:
            print("❌ 控制点数量不足")
            return image
        
        # 执行TPS变形
        try:
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            
            tps = cv2.createThinPlateSplineShapeTransformer()
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
            
            print("🔄 开始基于参考点的TPS变换...")
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
            
            # 创建安全mask
            eye_corner_mask = self.create_safe_eye_corner_mask(image.shape, landmarks_2d)
            
            # 安全融合
            mask_3d = cv2.merge([eye_corner_mask, eye_corner_mask, eye_corner_mask]).astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_3d, (5, 5), 0)
            
            result = image.astype(np.float32) * (1 - mask_blurred) + transformed.astype(np.float32) * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print("✅ 基于参考点的眼角变形成功")
            return result
            
        except Exception as e:
            print(f"❌ TPS变形异常: {e}")
            return image

    def validate_movement_directions(self, src_points, dst_points, left_eye_center, right_eye_center):
        """验证移动方向的合理性"""
        print("🔍 验证移动方向连续性:")
        
        direction_conflicts = 0
        total_movements = 0
        
        for i, (src, dst) in enumerate(zip(src_points, dst_points)):
            src = np.array(src)
            dst = np.array(dst)
            movement = dst - src
            movement_norm = np.linalg.norm(movement)
            
            if movement_norm > 0.1:  # 只检查有明显移动的点
                total_movements += 1
                
                # 判断是左眼还是右眼区域的点
                dist_to_left = np.linalg.norm(src - left_eye_center)
                dist_to_right = np.linalg.norm(src - right_eye_center)
                
                if dist_to_left < dist_to_right:  # 左眼区域
                    reference = left_eye_center
                    eye_side = "左眼"
                else:  # 右眼区域
                    reference = right_eye_center
                    eye_side = "右眼"
                
                # 计算期望的径向方向
                expected_direction = src - reference
                expected_norm = np.linalg.norm(expected_direction)
                
                if expected_norm > 0:
                    expected_direction = expected_direction / expected_norm
                    actual_direction = movement / movement_norm
                    
                    # 计算方向夹角
                    cos_angle = np.dot(expected_direction, actual_direction)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.degrees(np.arccos(cos_angle))
                    
                    if angle > 90:  # 如果移动方向与径向方向夹角过大
                        direction_conflicts += 1
                        print(f"   ⚠️  点{i} ({eye_side}): 方向偏差{angle:.1f}度")
        
        if total_movements > 0:
            conflict_ratio = direction_conflicts / total_movements
            print(f"   移动点数: {total_movements}, 方向冲突: {direction_conflicts} ({conflict_ratio*100:.1f}%)")
            
            if conflict_ratio > 0.3:
                print("   ⚠️  方向冲突较多，可能影响变形效果")
            else:
                print("   ✅ 移动方向基本合理")
        else:
            print("   ℹ️  无明显移动点")
    
    def debug_eye_corner_detection(self, image, save_dir=None):
        """调试眼角检测（基于正确原则）"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        debug_img = image.copy()
        
        # 绘制移动点（形成"面"分布）
        # 左眼角主要移动点 - 绿色大圆
        for i, idx in enumerate(self.LEFT_EYE_CORNER_POINTS):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 8, (0, 255, 0), -1)
                cv2.putText(debug_img, f"LM{i}", tuple(point + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 左眼角辅助点 - 绿色小圆
        for i, idx in enumerate(self.LEFT_EYE_CORNER_AUX):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 5, (0, 200, 0), -1)
                cv2.putText(debug_img, f"LA{i}", tuple(point + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        
        # 右眼角主要移动点 - 红色大圆
        for i, idx in enumerate(self.RIGHT_EYE_CORNER_POINTS):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 8, (0, 0, 255), -1)
                cv2.putText(debug_img, f"RM{i}", tuple(point + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 右眼角辅助点 - 红色小圆
        for i, idx in enumerate(self.RIGHT_EYE_CORNER_AUX):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 5, (0, 0, 200), -1)
                cv2.putText(debug_img, f"RA{i}", tuple(point + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        
        # 绘制锚点（远离变形区）- 蓝色小点
        for i, idx in enumerate(self.STABLE_ANCHORS[:10]):  # 只显示前10个避免过密
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 3, (255, 0, 0), -1)
        
        # 绘制对称辅助点 - 黄色方块
        for idx in self.SYMMETRY_HELPERS:
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.rectangle(debug_img, tuple(point-3), tuple(point+3), (0, 255, 255), -1)
        
        # 绘制眼球中心参考点
        left_eye_center, right_eye_center = self.calculate_eye_centers(landmarks_2d)
        if left_eye_center is not None and right_eye_center is not None:
            cv2.circle(debug_img, tuple(left_eye_center.astype(int)), 6, (255, 255, 0), -1)
            cv2.circle(debug_img, tuple(right_eye_center.astype(int)), 6, (255, 255, 0), -1)
            cv2.putText(debug_img, 'L-CENTER', tuple(left_eye_center.astype(int) + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(debug_img, 'R-CENTER', tuple(right_eye_center.astype(int) + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 绘制眼型基准线
        if landmarks_2d is not None:
            # 左眼型线条（内眦到外眦）
            if all(idx < len(landmarks_2d) for idx in [133, 33]):
                cv2.line(debug_img, tuple(landmarks_2d[133].astype(int)), 
                        tuple(landmarks_2d[33].astype(int)), (0, 255, 0), 2)
            # 右眼型线条
            if all(idx < len(landmarks_2d) for idx in [362, 263]):
                cv2.line(debug_img, tuple(landmarks_2d[362].astype(int)), 
                        tuple(landmarks_2d[263].astype(int)), (0, 0, 255), 2)
        
        # 显示3D姿态信息
        if pose_info is not None:
            pose_3d, pose_angle = self.estimate_face_pose_3d(pose_info[0])
            cv2.putText(debug_img, f"3D Pose: {pose_3d} ({pose_angle:.1f}°)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示验证结果
        is_valid = self.validate_eye_corner_points(landmarks_2d)
        status_text = "VALID" if is_valid else "INVALID" 
        cv2.putText(debug_img, f"Eye Corner Points: {status_text}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_valid else (0, 0, 255), 2)
        
        # 显示mask预览
        eye_corner_mask = self.create_safe_eye_corner_mask(image.shape, landmarks_2d)
        mask_overlay = debug_img.copy()
        mask_overlay[eye_corner_mask > 0] = [255, 0, 255]  # 紫色mask
        debug_img = cv2.addWeighted(debug_img, 0.8, mask_overlay, 0.2, 0)
        
        # 详细图例
        legend_y = 90
        legends = [
            ("GREEN BIG: Left Main Move (LM0-3)", (0, 255, 0)),
            ("GREEN SMALL: Left Auxiliary (LA0-3)", (0, 200, 0)),
            ("RED BIG: Right Main Move (RM0-3)", (0, 0, 255)),
            ("RED SMALL: Right Auxiliary (RA0-3)", (0, 0, 200)),
            ("BLUE DOTS: Stable Anchors (Far from eyes)", (255, 0, 0)),
            ("YELLOW SQUARES: Symmetry Helpers", (0, 255, 255)),
            ("CYAN CIRCLES: Eye Centers (Medical Ref)", (255, 255, 0)),
            ("GREEN/RED LINES: Eye Shape Baseline", (255, 255, 255)),
            ("PURPLE OVERLAY: Transform Mask", (255, 0, 255))
        ]
        
        for i, (text, color) in enumerate(legends):
            cv2.putText(debug_img, text, (10, legend_y + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 显示点数统计
        left_main_count = sum(1 for idx in self.LEFT_EYE_CORNER_POINTS if idx < len(landmarks_2d))
        left_aux_count = sum(1 for idx in self.LEFT_EYE_CORNER_AUX if idx < len(landmarks_2d))
        right_main_count = sum(1 for idx in self.RIGHT_EYE_CORNER_POINTS if idx < len(landmarks_2d))
        right_aux_count = sum(1 for idx in self.RIGHT_EYE_CORNER_AUX if idx < len(landmarks_2d))
        anchor_count = sum(1 for idx in self.STABLE_ANCHORS if idx < len(landmarks_2d))
        
        stats_y = legend_y + len(legends) * 20 + 20
        cv2.putText(debug_img, f"Point Stats: L({left_main_count}+{left_aux_count}) R({right_main_count}+{right_aux_count}) A{anchor_count}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "debug_fixed_eye_corner.png"), debug_img)
            print(f"📁 调试文件保存到: {save_dir}")
        
        return debug_img
    
    def process_image(self, image, adjustment_type="stretch", intensity=0.3, save_debug=False, output_path=None):
        """处理图像（主要接口）"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        print(f"✅ 检测到 {len(landmarks_2d)} 个关键点")
        
        # 验证眼角关键点
        if not self.validate_eye_corner_points(landmarks_2d):
            print("⚠️  眼角关键点验证有问题，但继续处理")  # 参考眼睛脚本：警告不阻止
        
        if save_debug and output_path:
            debug_dir = os.path.splitext(output_path)[0] + "_debug"
            self.debug_eye_corner_detection(image, debug_dir)
        
        return self.apply_eye_corner_transform(image, landmarks_2d, pose_info, adjustment_type, intensity)

def main():
    parser = argparse.ArgumentParser(description='重构的眼角调整工具 - 基于正确原则')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--type', '-t', choices=['stretch', 'lift', 'shape'], 
                       default='stretch', help='调整类型: stretch=水平拉长, lift=眼角上扬, shape=综合调整')
    parser.add_argument('--intensity', type=float, default=0.3, 
                       help='调整强度 (0.1-0.4)')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--save-debug', action='store_true', help='保存调试文件')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"❌ 无法加载图片: {args.input}")
        return
    
    print(f"📸 图片尺寸: {image.shape}")
    
    # 强度安全检查（参考眼睛脚本的温和限制）
    if args.intensity > 0.4:
        print(f"⚠️  强度 {args.intensity} 超出建议范围，限制到0.4")
        args.intensity = 0.4
    elif args.intensity < 0.1:
        print(f"⚠️  强度 {args.intensity} 过小，提升到0.1")
        args.intensity = 0.1
    
    # 创建处理器
    processor = FixedEyeCornerAdjustment()
    
    # 调试模式
    if args.debug:
        debug_result = processor.debug_eye_corner_detection(image)
        cv2.imwrite(args.output, debug_result)
        print(f"🔍 调试图保存到: {args.output}")
        return
    
    # 处理图片
    print(f"🔄 开始眼角处理 - 类型: {args.type}, 强度: {args.intensity}")
    result = processor.process_image(image, args.type, args.intensity, args.save_debug, args.output)
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"✅ 处理完成，保存到: {args.output}")
    
    # 检查变化统计（参考眼睛脚本）
    diff = cv2.absdiff(image, result)
    total_diff = np.sum(diff)
    max_diff = np.max(diff)
    changed_pixels = np.sum(diff > 0)
    print(f"📊 像素变化统计:")
    print(f"   总差异: {total_diff}")
    print(f"   最大差异: {max_diff}")
    print(f"   变化像素数: {changed_pixels}")
    
    if total_diff < 1000:
        print("⚠️  变化很小，可能需要：")
        print("   1. 提高 --intensity 参数 (如0.4)")
        print("   2. 检查眼角检测是否正常 (--debug)")
        print("   3. 尝试不同的调整类型")
    
    # 生成对比图
    if args.comparison:
        comparison_path = args.output.replace('.', '_comparison.')
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
        
        # 添加标签
        cv2.putText(comparison, 'Original', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(comparison, f'Fixed-{args.type.upper()} {args.intensity:.2f}', 
                   (image.shape[1] + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.imwrite(comparison_path, comparison)
        print(f"📊 对比图保存到: {comparison_path}")
    
    print("\n🎯 重构后的设计优势:")
    print("✅ 远程锚点: 使用鼻子、额头、嘴巴等远离眼部的稳定锚点")
    print("✅ 医学美学参考: 基于眼球中心的径向移动方向")
    print("✅ 去重机制: 防止重复点位导致绿色遮挡")
    print("✅ 对称分布: 左右眼角各4+4个点，完全对称")
    print("✅ 温和验证: 参考眼睛脚本，警告不阻止")
    print("✅ 安全mask: 小圆形区域，避免过度扩展")
    print("✅ 三种模式区分明确: 拉长/上扬/综合调整")
    
    print("\n📍 三种模式详解:")
    print("• STRETCH（水平拉长）: 改变眼睛长宽比，从圆眼变长眼")
    print("  - 外眼角向外，内眼角向内，基于眼球中心径向拉长")
    print("• LIFT（眼角上扬）: 外眼角向上，内眼角保持，创造有神眼型")
    print("  - 主要调整外眼角区域，内眼角区域保持稳定")
    print("• SHAPE（综合调整）: 拉长+上扬的组合效果")
    print("  - 外眼角向外上，内眼角轻微调整，综合优化眼型")
    
    print(f"\n📍 使用的关键点:")
    print(f"• 左眼角移动: {processor.LEFT_EYE_CORNER_POINTS + processor.LEFT_EYE_CORNER_AUX}")
    print(f"• 右眼角移动: {processor.RIGHT_EYE_CORNER_POINTS + processor.RIGHT_EYE_CORNER_AUX}")
    print(f"• 稳定锚点: {processor.STABLE_ANCHORS[:7]}...")
    print(f"• 对称辅助: {processor.SYMMETRY_HELPERS}")
    
    print("\n📖 使用示例:")
    print("• 眼型水平拉长: python script.py -i input.jpg -o output.png -t stretch --intensity 0.3 --comparison")
    print("• 眼角上扬: python script.py -i input.jpg -o output.png -t lift --intensity 0.25 --comparison")
    print("• 综合眼型调整: python script.py -i input.jpg -o output.png -t shape --intensity 0.35 --comparison")
    print("• 调试检查点位: python script.py -i input.jpg -o debug.png --debug")
    
    print("\n🔄 修正的问题:")
    print("✅ 锚点距离: 移除眼部中心区域锚点，使用远程稳定点")
    print("✅ 重复点位: 实现去重机制，防止TPS插值失败")
    print("✅ 医学参考: 基于眼球中心建立正确的移动方向")
    print("✅ 验证逻辑: 改为温和警告模式，不阻止处理")
    print("✅ mask设计: 小圆形区域，避免绿色遮挡")
    print("✅ 模式区分: 三种模式效果明确，与eye_adjust.py区分")

if __name__ == "__main__":
    main()
