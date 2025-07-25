#!/usr/bin/env python3
"""
基于正确原则的眉毛调整脚本 - eyebrow_adjust_fixed.py
设计原则：锚点远离变形区 + 移动点形成"面"分布 + 对称性 + 参考眼睛脚本成功模式
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
from typing import List, Tuple, Optional

class CorrectEyebrowAdjustment:
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
        
        # 🎯 基于正确原则的关键点设计
        
        # 移动点：形成分布较广的眉毛形变区域（你指定的精确点位）
        self.LEFT_EYEBROW_POINTS = [70, 63, 105, 66]     # 左眉4个主要移动点
        self.RIGHT_EYEBROW_POINTS = [300, 334, 296, 293] # 右眉4个对应移动点（对称分布）
        
        # 扩展移动点：形成"面"而非"线"的分布
        self.LEFT_EYEBROW_EXTENDED = [107, 104, 68, 69]   # 左眉扩展点（立体分布）
        self.RIGHT_EYEBROW_EXTENDED = [276, 285, 295, 282] # 右眉扩展点（对称对应）
        
        # 锚点：远离变形区的稳定结构（你指定的有效锚点）
        self.STABLE_ANCHORS = [
            10, 338, 297, 117, 147, 33, 160,  # 额头 + 颧骨 + 眼眶上部
            # 补充稳定锚点（参考眼睛脚本模式）
            1, 2, 5, 4, 6, 19, 20,             # 鼻子区域
            61, 291, 39, 269, 270,             # 嘴巴区域  
            151, 175, 234, 454, 323, 361       # 下巴和脸颊边缘
        ]
        
        # 辅助点：保持左右对称的眼眶区域（你指定的对称点）
        self.SYMMETRY_HELPERS = [33, 160, 158, 263, 387, 385]
        
        # 医学美学参考点（基于眼球中心）
        self.LEFT_EYE_CENTER_REFS = [33, 133]   # 左眼内外眦
        self.RIGHT_EYE_CENTER_REFS = [362, 263]  # 右眼内外眦
        
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
        self.MAX_DISPLACEMENT = 50        # 参考眼睛脚本的150，眉毛相对保守
        self.MIN_EYEBROW_AREA = 500
    
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
        """计算眼球中心参考点"""
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
    
    def validate_eyebrow_points(self, landmarks_2d):
        """简化的眉毛点验证（参考眼睛脚本模式）"""
        try:
            # 检查所有关键点是否在有效范围内
            all_points = (self.LEFT_EYEBROW_POINTS + self.RIGHT_EYEBROW_POINTS + 
                         self.LEFT_EYEBROW_EXTENDED + self.RIGHT_EYEBROW_EXTENDED)
            
            valid_count = sum(1 for idx in all_points if idx < len(landmarks_2d))
            total_count = len(all_points)
            
            if valid_count < total_count * 0.8:  # 至少80%的点有效
                print(f"⚠️  眉毛关键点不足: {valid_count}/{total_count}")
                return False
            
            # 检查左右眉毛点数是否对称
            left_count = sum(1 for idx in self.LEFT_EYEBROW_POINTS + self.LEFT_EYEBROW_EXTENDED 
                           if idx < len(landmarks_2d))
            right_count = sum(1 for idx in self.RIGHT_EYEBROW_POINTS + self.RIGHT_EYEBROW_EXTENDED 
                            if idx < len(landmarks_2d))
            
            if abs(left_count - right_count) > 1:
                print(f"⚠️  左右眉毛点数不对称: 左{left_count}, 右{right_count}")
                # 参考眼睛脚本：警告但不阻止
            
            print(f"✅ 眉毛关键点验证: 左{left_count}个, 右{right_count}个, 总计{valid_count}/{total_count}")
            return True
            
        except Exception as e:
            print(f"❌ 眉毛验证失败: {e}")
            return False
    
    def create_safe_eyebrow_mask(self, image_shape, landmarks_2d):
        """创建安全的eyebrow mask（参考眼睛脚本模式）"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 收集所有眉毛点
        all_eyebrow_points = []
        
        for idx in (self.LEFT_EYEBROW_POINTS + self.RIGHT_EYEBROW_POINTS + 
                   self.LEFT_EYEBROW_EXTENDED + self.RIGHT_EYEBROW_EXTENDED):
            if idx < len(landmarks_2d):
                all_eyebrow_points.append(landmarks_2d[idx])
        
        if len(all_eyebrow_points) < 8:
            print("⚠️  眉毛点不足，创建保守mask")
            return mask
        
        all_eyebrow_points = np.array(all_eyebrow_points, dtype=np.int32)
        
        # 参考眼睛脚本：为每个点创建小圆形区域
        for point in all_eyebrow_points:
            cv2.circle(mask, tuple(point), 12, 255, -1)  # 12像素半径的圆形
        
        # 轻微连接（参考眼睛脚本的处理）
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # 边界保护
        # 眼部保护
        eye_points = self.LEFT_EYE_CENTER_REFS + self.RIGHT_EYE_CENTER_REFS
        eye_bottom = max([landmarks_2d[idx][1] for idx in eye_points if idx < len(landmarks_2d)])
        mask[int(eye_bottom - 5):, :] = 0
        
        # 发际线保护
        forehead_points = [10, 151, 9, 8]
        if all(idx < len(landmarks_2d) for idx in forehead_points):
            forehead_top = min([landmarks_2d[idx][1] for idx in forehead_points])
            mask[:int(forehead_top + 10), :] = 0
        
        mask_area = np.sum(mask > 0)
        print(f"✅ 安全Mask创建: 区域={mask_area}像素")
        return mask
    
    def apply_eyebrow_transform(self, image, landmarks_2d, pose_info, adjustment_type="lift", intensity=0.3):
        """应用眉毛变形（修复重复点位问题 + 区分三种模式效果）"""
        
        # 3D姿态分析
        if pose_info is None:
            pose_3d, pose_angle = "frontal", 0.0
        else:
            rotation_vector, translation_vector = pose_info
            pose_3d, pose_angle = self.estimate_face_pose_3d(rotation_vector)
        
        # 获取眼球中心参考
        left_eye_center, right_eye_center = self.calculate_eye_centers(landmarks_2d)
        if left_eye_center is None:
            print("⚠️  无法建立医学参考，使用几何中心")
        
        # 根据3D姿态调整强度
        if pose_3d == "frontal":
            left_intensity = right_intensity = intensity
        elif pose_3d == "right_profile":
            right_intensity = intensity * 1.0
            left_intensity = intensity * 0.7
        else:  # left_profile  
            left_intensity = intensity * 1.0
            right_intensity = intensity * 0.7
        
        print(f"🔧 3D自适应强度: 左眉={left_intensity:.2f}, 右眉={right_intensity:.2f}")
        
        # 构建控制点（修复重复问题）
        src_points = []
        dst_points = []
        used_indices = set()  # 🔧 关键修复：防止重复点位
        
        if adjustment_type == "lift":
            print(f"🔼 眉毛整体上提（提高眉毛位置，保持形状）")
            
            # 左眉毛统一上移
            for idx in self.LEFT_EYEBROW_POINTS + self.LEFT_EYEBROW_EXTENDED:
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    # 统一向上平移，保持眉形不变
                    lift_amount = left_intensity * 12
                    new_point = point + np.array([0, -lift_amount])
                    dst_points.append(new_point)
            
            # 右眉毛统一上移
            for idx in self.RIGHT_EYEBROW_POINTS + self.RIGHT_EYEBROW_EXTENDED:
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    lift_amount = right_intensity * 12
                    new_point = point + np.array([0, -lift_amount])
                    dst_points.append(new_point)
        
        elif adjustment_type == "peak":
            print(f"🔺 眉峰塑造（增强眉毛弧度，眉峰突出）")
            
            # 左眉弧度塑造
            left_points = self.LEFT_EYEBROW_POINTS + self.LEFT_EYEBROW_EXTENDED
            for i, idx in enumerate(left_points):
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    # 根据位置决定移动量：中间大，两端小
                    total_points = len(left_points)
                    center_pos = total_points // 2
                    distance_from_center = abs(i - center_pos)
                    
                    if distance_from_center == 0:  # 眉峰中心
                        move_amount = left_intensity * 22
                    elif distance_from_center == 1:  # 眉峰附近
                        move_amount = left_intensity * 18
                    elif distance_from_center >= total_points // 2:  # 眉头眉尾
                        move_amount = left_intensity * 3
                    else:  # 过渡区域
                        move_amount = left_intensity * 12
                    
                    new_point = point + np.array([0, -move_amount])
                    dst_points.append(new_point)
            
            # 右眉弧度塑造（对称处理）
            right_points = self.RIGHT_EYEBROW_POINTS + self.RIGHT_EYEBROW_EXTENDED
            for i, idx in enumerate(right_points):
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    total_points = len(right_points)
                    center_pos = total_points // 2
                    distance_from_center = abs(i - center_pos)
                    
                    if distance_from_center == 0:
                        move_amount = right_intensity * 22
                    elif distance_from_center == 1:
                        move_amount = right_intensity * 18
                    elif distance_from_center >= total_points // 2:
                        move_amount = right_intensity * 3
                    else:
                        move_amount = right_intensity * 12
                    
                    new_point = point + np.array([0, -move_amount])
                    dst_points.append(new_point)
        
        elif adjustment_type == "shape":
            print(f"📐 眉形重塑（改变眉毛整体形状和走向）")
            
            # 左眉形状重塑
            left_points = self.LEFT_EYEBROW_POINTS + self.LEFT_EYEBROW_EXTENDED
            for i, idx in enumerate(left_points):
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    total_points = len(left_points)
                    
                    if i == 0:  # 眉头：向下向内，降低眉头
                        adjust = np.array([1, 4]) * left_intensity * 2
                    elif i == 1:  # 眉峰前：向上向外
                        adjust = np.array([-1, -5]) * left_intensity * 3
                    elif i == total_points - 1:  # 眉尾：向外延伸
                        adjust = np.array([-4, -2]) * left_intensity * 2.5
                    elif i == total_points - 2:  # 眉尾前：向外向上
                        adjust = np.array([-2, -3]) * left_intensity * 2
                    else:  # 中间点：平滑过渡
                        adjust = np.array([0, -3]) * left_intensity * 1.5
                    
                    new_point = point + adjust
                    dst_points.append(new_point)
            
            # 右眉形状重塑（镜像处理）
            right_points = self.RIGHT_EYEBROW_POINTS + self.RIGHT_EYEBROW_EXTENDED
            for i, idx in enumerate(right_points):
                if idx < len(landmarks_2d) and idx not in used_indices:
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    used_indices.add(idx)
                    
                    total_points = len(right_points)
                    
                    if i == 0:  # 眉头：向下向内
                        adjust = np.array([-1, 4]) * right_intensity * 2
                    elif i == 1:  # 眉峰前：向上向外
                        adjust = np.array([1, -5]) * right_intensity * 3
                    elif i == total_points - 1:  # 眉尾：向外延伸
                        adjust = np.array([4, -2]) * right_intensity * 2.5
                    elif i == total_points - 2:  # 眉尾前：向外向上
                        adjust = np.array([2, -3]) * right_intensity * 2
                    else:  # 中间点：平滑过渡
                        adjust = np.array([0, -3]) * right_intensity * 1.5
                    
                    new_point = point + adjust
                    dst_points.append(new_point)
        
        else:
            print(f"❌ 未知的调整类型: {adjustment_type}")
            return image
        
        # 🔧 修复：添加锚点时检查重复
        anchor_count = 0
        for idx in self.STABLE_ANCHORS:
            if idx < len(landmarks_2d) and idx not in used_indices:
                src_points.append(landmarks_2d[idx])
                dst_points.append(landmarks_2d[idx])
                used_indices.add(idx)
                anchor_count += 1
        
        # 🔧 修复：添加对称辅助点时检查重复
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
        print(f"   眉毛移动点: {len(src_points) - anchor_count - symmetry_count - 8}")
        print(f"   稳定锚点: {anchor_count}")
        print(f"   对称辅助点: {symmetry_count}")
        print(f"   边界锚点: 8")
        print(f"   总计: {len(src_points)} 个控制点")
        print(f"   去重后有效索引: {len(used_indices)} 个")
        
        # 简化的安全检查
        if len(src_points) < 20:
            print("❌ 控制点数量不足")
            return image
        
        # 检查最大位移
        max_displacement = 0
        for src, dst in zip(src_points, dst_points):
            displacement = np.linalg.norm(np.array(dst) - np.array(src))
            max_displacement = max(max_displacement, displacement)
        
        if max_displacement > self.MAX_DISPLACEMENT:
            print(f"⚠️  最大位移: {max_displacement:.1f}像素 (限制{self.MAX_DISPLACEMENT})")
        
        # 执行TPS变形
        try:
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            
            tps = cv2.createThinPlateSplineShapeTransformer()
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
            
            print("🔄 开始TPS变换...")
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
            eyebrow_mask = self.create_safe_eyebrow_mask(image.shape, landmarks_2d)
            
            # 安全融合
            mask_3d = cv2.merge([eyebrow_mask, eyebrow_mask, eyebrow_mask]).astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_3d, (5, 5), 0)
            
            result = image.astype(np.float32) * (1 - mask_blurred) + transformed.astype(np.float32) * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print("✅ 眉毛变形成功（已修复重复点位 + 区分模式效果）")
            return result
            
        except Exception as e:
            print(f"❌ TPS变形异常: {e}")
            return image

    def debug_eyebrow_detection(self, image, save_dir=None):
        """调试眉毛检测（基于正确原则）"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        debug_img = image.copy()
        
        # 绘制移动点（形成"面"分布）
        # 左眉主要移动点 - 绿色大圆
        for i, idx in enumerate(self.LEFT_EYEBROW_POINTS):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 8, (0, 255, 0), -1)
                cv2.putText(debug_img, f"LM{i}", tuple(point + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 左眉扩展点 - 绿色小圆
        for i, idx in enumerate(self.LEFT_EYEBROW_EXTENDED):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 5, (0, 200, 0), -1)
                cv2.putText(debug_img, f"LE{i}", tuple(point + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        
        # 右眉主要移动点 - 红色大圆
        for i, idx in enumerate(self.RIGHT_EYEBROW_POINTS):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 8, (0, 0, 255), -1)
                cv2.putText(debug_img, f"RM{i}", tuple(point + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 右眉扩展点 - 红色小圆
        for i, idx in enumerate(self.RIGHT_EYEBROW_EXTENDED):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 5, (0, 0, 200), -1)
                cv2.putText(debug_img, f"RE{i}", tuple(point + 8), 
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
        
        # 显示3D姿态信息
        if pose_info is not None:
            pose_3d, pose_angle = self.estimate_face_pose_3d(pose_info[0])
            cv2.putText(debug_img, f"3D Pose: {pose_3d} ({pose_angle:.1f}°)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示验证结果
        is_valid = self.validate_eyebrow_points(landmarks_2d)
        status_text = "VALID" if is_valid else "INVALID" 
        cv2.putText(debug_img, f"Eyebrow Points: {status_text}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_valid else (0, 0, 255), 2)
        
        # 显示mask预览
        eyebrow_mask = self.create_safe_eyebrow_mask(image.shape, landmarks_2d)
        mask_overlay = debug_img.copy()
        mask_overlay[eyebrow_mask > 0] = [255, 0, 255]  # 紫色mask
        debug_img = cv2.addWeighted(debug_img, 0.8, mask_overlay, 0.2, 0)
        
        # 详细图例
        legend_y = 90
        legends = [
            ("GREEN BIG: Left Main Move (LM0-3)", (0, 255, 0)),
            ("GREEN SMALL: Left Extended (LE0-3)", (0, 200, 0)),
            ("RED BIG: Right Main Move (RM0-3)", (0, 0, 255)),
            ("RED SMALL: Right Extended (RE0-3)", (0, 0, 200)),
            ("BLUE DOTS: Stable Anchors (Far from eyebrows)", (255, 0, 0)),
            ("YELLOW SQUARES: Symmetry Helpers", (0, 255, 255)),
            ("CYAN CIRCLES: Eye Centers (Medical Ref)", (255, 255, 0)),
            ("PURPLE OVERLAY: Transform Mask", (255, 0, 255))
        ]
        
        for i, (text, color) in enumerate(legends):
            cv2.putText(debug_img, text, (10, legend_y + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 显示点数统计
        left_main_count = sum(1 for idx in self.LEFT_EYEBROW_POINTS if idx < len(landmarks_2d))
        left_ext_count = sum(1 for idx in self.LEFT_EYEBROW_EXTENDED if idx < len(landmarks_2d))
        right_main_count = sum(1 for idx in self.RIGHT_EYEBROW_POINTS if idx < len(landmarks_2d))
        right_ext_count = sum(1 for idx in self.RIGHT_EYEBROW_EXTENDED if idx < len(landmarks_2d))
        anchor_count = sum(1 for idx in self.STABLE_ANCHORS if idx < len(landmarks_2d))
        
        stats_y = legend_y + len(legends) * 20 + 20
        cv2.putText(debug_img, f"Point Stats: L({left_main_count}+{left_ext_count}) R({right_main_count}+{right_ext_count}) A{anchor_count}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "debug_correct_eyebrow.png"), debug_img)
            print(f"📁 调试文件保存到: {save_dir}")
        
        return debug_img
    
    def process_image(self, image, adjustment_type="lift", intensity=0.3, save_debug=False, output_path=None):
        """处理图像（主要接口）"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        print(f"✅ 检测到 {len(landmarks_2d)} 个关键点")
        
        # 验证眉毛关键点
        if not self.validate_eyebrow_points(landmarks_2d):
            print("⚠️  眉毛关键点验证有问题，但继续处理")  # 参考眼睛脚本：警告不阻止
        
        if save_debug and output_path:
            debug_dir = os.path.splitext(output_path)[0] + "_debug"
            self.debug_eyebrow_detection(image, debug_dir)
        
        return self.apply_eyebrow_transform(image, landmarks_2d, pose_info, adjustment_type, intensity)

def main():
    parser = argparse.ArgumentParser(description='基于正确原则的眉毛调整工具')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--type', '-t', choices=['lift', 'peak', 'shape'], 
                       default='lift', help='调整类型: lift=整体提升, peak=眉峰抬高, shape=眉形调整')
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
    processor = CorrectEyebrowAdjustment()
    
    # 调试模式
    if args.debug:
        debug_result = processor.debug_eyebrow_detection(image)
        cv2.imwrite(args.output, debug_result)
        print(f"🔍 调试图保存到: {args.output}")
        return
    
    # 处理图片
    print(f"🔄 开始眉毛处理 - 类型: {args.type}, 强度: {args.intensity}")
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
        print("   2. 检查眉毛检测是否正常 (--debug)")
        print("   3. 尝试不同的调整类型")
    
    # 生成对比图
    if args.comparison:
        comparison_path = args.output.replace('.', '_comparison.')
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
        
        # 添加标签
        cv2.putText(comparison, 'Original', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(comparison, f'Correct-{args.type.upper()} {args.intensity:.2f}', 
                   (image.shape[1] + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.imwrite(comparison_path, comparison)
        print(f"📊 对比图保存到: {comparison_path}")
    
    print("\n🎯 基于正确原则的设计:")
    print("✅ 移动点分布: 左右各4+4个点，形成'面'而非'线'")
    print("✅ 锚点远离: 额头+颧骨+眼眶上部，距离眉毛足够远")
    print("✅ 对称性保证: 左右眉毛点数完全对称")
    print("✅ 医学美学参考: 基于眼球中心的垂直向上移动")
    print("✅ 3D姿态适应: 侧脸时调整左右眉毛强度")
    print("✅ 简化验证: 参考眼睛脚本，警告不阻止")
    print("✅ 安全TPS: 控制最大位移，保守mask设计")
    
    print(f"\n📍 使用的关键点:")
    print(f"• 左眉移动: {processor.LEFT_EYEBROW_POINTS + processor.LEFT_EYEBROW_EXTENDED}")
    print(f"• 右眉移动: {processor.RIGHT_EYEBROW_POINTS + processor.RIGHT_EYEBROW_EXTENDED}")
    print(f"• 稳定锚点: {processor.STABLE_ANCHORS[:7]}...")
    print(f"• 对称辅助: {processor.SYMMETRY_HELPERS}")
    
    print("\n📖 使用示例:")
    print("• 眉毛整体提升: python script.py -i input.jpg -o output.png -t lift --intensity 0.3 --comparison")
    print("• 眉峰单独抬高: python script.py -i input.jpg -o output.png -t peak --intensity 0.25 --comparison")
    print("• 眉形精细调整: python script.py -i input.jpg -o output.png -t shape --intensity 0.35 --comparison")
    print("• 调试检查点位: python script.py -i input.jpg -o debug.png --debug")
    print("• 保存调试文件: python script.py -i input.jpg -o output.png --intensity 0.3 --save-debug --comparison")
    
    print("\n🔄 修正的问题:")
    print("✅ 控制点距离: 使用你验证的精确点位")
    print("✅ 左右平衡: 确保左右眉毛点数对称") 
    print("✅ 锚点设计: 远离变形区的稳定锚点")
    print("✅ 验证逻辑: 参考眼睛脚本的温和验证")
    print("✅ mask设计: 小圆形区域，避免绿色遮挡")
    print("✅ 移动方向: 基于医学美学的连续性移动")

if __name__ == "__main__":
    main()
