#!/usr/bin/env python3
"""
精简高效眼角调整器 - streamlined_eye_corner_adjuster.py
核心理念：基于美学原理的智能调整，删除过度工程化的功能

保留的核心功能：
1. 3D姿态感知 - PnP算法的真实头部姿态估计
2. 美学导向的移动策略 - 识别眼形特征，制定针对性移动策略
3. 结构完整性保护 - 确保眼周组织协调性
4. 智能强度控制 - 保证可见效果的同时确保自然

删除的冗余功能：
- 过于复杂的年龄估计
- 过度细分的眼形分类
- 多层次因子相乘导致的强度稀释
- 过于保守的安全机制
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
from typing import Tuple, Dict, List, Optional

class EyeCornerAdjuster:
    """精简高效的眼角调整器"""
    
    def __init__(self,face_mesh=None):
        # MediaPipe初始化
        self.mp_face_mesh = mp.solutions.face_mesh
        if not face_mesh:
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
        
        # 3D面部模型 (简化版，只保留关键点)
        self.face_model_3d = np.array([
            [0.0, 0.0, 0.0],           # 鼻尖
            [0.0, -330.0, -65.0],      # 下巴
            [-225.0, 170.0, -135.0],   # 左眼外角
            [225.0, 170.0, -135.0],    # 右眼外角
            [-150.0, -150.0, -125.0],  # 左嘴角
            [150.0, -150.0, -125.0],   # 右嘴角
        ], dtype=np.float32)
        
        self.pnp_indices = [1, 152, 33, 263, 61, 291]
        
        # 眼部移动区域定义 (精简版)
        self.EYE_REGIONS = {
            'left_eye': {
                'outer_corner': [33],                    # 主要外眦点
                'outer_support': [7, 163],               # 外眦支撑点
                'inner_corner': [133],                   # 内眦点
                'upper_lid': [157, 158, 159, 160, 161],  # 上眼睑
                'lower_lid': [144, 145, 153],            # 下眼睑
            },
            'right_eye': {
                'outer_corner': [263],                   # 主要外眦点
                'outer_support': [249, 390],             # 外眦支撑点
                'inner_corner': [362],                   # 内眦点
                'upper_lid': [386, 387, 388, 466],       # 上眼睑
                'lower_lid': [373, 374, 380],            # 下眼睑
            }
        }
        
        # 稳定锚点 (简化版)
        self.STABLE_ANCHORS = [
            1, 2, 5, 4, 6,              # 鼻部核心
            9, 10, 151, 175,            # 面部中轴
            61, 291, 39, 269, 270,      # 嘴部区域
            70, 296, 107, 276,          # 眉毛区域
            168, 6, 8                   # 对称轴
        ]
    
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
    
    def detect_landmarks_with_pose(self, image):
        """检测关键点并估计3D姿态"""
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
        
        # PnP求解3D姿态
        pose_info = None
        try:
            image_points = []
            for idx in self.pnp_indices:
                if idx < len(landmarks_2d):
                    image_points.append(landmarks_2d[idx])
            
            if len(image_points) >= 6:
                image_points = np.array(image_points, dtype=np.float32)
                
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.face_model_3d, image_points, 
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    pose_info = (rotation_vector, translation_vector)
        
        except Exception as e:
            print(f"⚠️ 3D姿态估计失败: {e}")
        
        return landmarks_2d, pose_info
    
    def analyze_eye_characteristics(self, landmarks_2d):
        """简化的眼形特征分析 - 只识别关键特征"""
        
        try:
            # 关键点获取
            left_inner, left_outer = landmarks_2d[133], landmarks_2d[33]
            left_top, left_bottom = landmarks_2d[159], landmarks_2d[145]
            right_inner, right_outer = landmarks_2d[362], landmarks_2d[263]
            right_top, right_bottom = landmarks_2d[386], landmarks_2d[374]
            
            # 计算关键特征
            left_tilt = np.degrees(np.arctan2(left_outer[1] - left_inner[1], left_outer[0] - left_inner[0]))
            right_tilt = np.degrees(np.arctan2(right_outer[1] - right_inner[1], right_outer[0] - right_inner[0]))
            avg_tilt = (left_tilt + right_tilt) / 2
            
            left_length = np.linalg.norm(left_outer - left_inner)
            left_height = np.linalg.norm(left_top - left_bottom)
            right_length = np.linalg.norm(right_outer - right_inner)
            right_height = np.linalg.norm(right_top - right_bottom)
            
            left_ratio = left_length / max(left_height, 1)
            right_ratio = right_length / max(right_height, 1)
            avg_ratio = (left_ratio + right_ratio) / 2
            
            # 识别关键特征 (简化为最重要的几个)
            features = {
                'upward_tilt': avg_tilt > 10,           # 明显上扬
                'downward_tilt': avg_tilt < -8,         # 明显下垂
                'very_round': avg_ratio < 2.4,          # 很圆
                'very_elongated': avg_ratio > 3.4,      # 很长
                'asymmetric': abs(left_tilt - right_tilt) > 8  # 不对称
            }
            
            return features, {
                'avg_tilt': avg_tilt,
                'avg_ratio': avg_ratio,
                'left_tilt': left_tilt,
                'right_tilt': right_tilt
            }
            
        except Exception as e:
            print(f"⚠️ 眼形分析失败: {e}")
            return {}, {}
    
    def get_pose_analysis(self, pose_info):
        """简化的3D姿态分析"""
        if pose_info is None:
            return {'type': 'frontal', 'yaw': 0, 'difficulty': 'easy', 'intensity_factor': 1.0}
        
        try:
            rotation_vector, translation_vector = pose_info
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
            yaw = np.degrees(np.arctan2(-rotation_matrix[2,0], sy))
            
            # 简化的姿态分类
            if abs(yaw) < 20:
                pose_type, difficulty = 'frontal', 'easy'
                intensity_factor = 1.0
            elif abs(yaw) < 35:
                pose_type, difficulty = 'slight_profile', 'medium'
                intensity_factor = 0.8
            else:
                pose_type, difficulty = 'profile', 'hard'
                intensity_factor = 0.6
            
            return {
                'type': pose_type,
                'yaw': yaw,
                'difficulty': difficulty,
                'intensity_factor': intensity_factor
            }
            
        except Exception as e:
            print(f"⚠️ 姿态分析失败: {e}")
            return {'type': 'frontal', 'yaw': 0, 'difficulty': 'easy', 'intensity_factor': 1.0}
    
    def calculate_adaptive_parameters(self, landmarks_2d, image_shape):
        """计算自适应参数 - 基于图像尺寸和眼部实际大小"""
        
        try:
            # 计算眼部特征尺寸
            left_eye_width = np.linalg.norm(landmarks_2d[33] - landmarks_2d[133])
            right_eye_width = np.linalg.norm(landmarks_2d[263] - landmarks_2d[362])
            avg_eye_width = (left_eye_width + right_eye_width) / 2
            
            # 计算眼部高度
            left_eye_height = np.linalg.norm(landmarks_2d[159] - landmarks_2d[145])
            right_eye_height = np.linalg.norm(landmarks_2d[386] - landmarks_2d[374])
            avg_eye_height = (left_eye_height + right_eye_height) / 2
            
            # 计算面部特征尺寸作为参考
            face_width = np.linalg.norm(landmarks_2d[234] - landmarks_2d[454])  # 脸宽
            inter_eye_distance = np.linalg.norm(landmarks_2d[33] - landmarks_2d[263])  # 双眼间距
            
            # 图像分辨率信息
            h, w = image_shape[:2]
            image_area = h * w
            image_diagonal = np.sqrt(h*h + w*w)
            
            # 计算眼部在图像中的相对大小
            eye_to_face_ratio = avg_eye_width / face_width if face_width > 0 else 0.3
            eye_to_image_ratio = avg_eye_width / w if w > 0 else 0.1
            
            # 自适应参数计算
            adaptive_params = {}
            
            # 1. 基础移动scale - 相对于眼部大小
            # 目标：移动距离应该是眼宽的8-15%才有明显效果
            base_movement_ratio = 0.12  # 眼宽的12%
            adaptive_params['base_scale'] = avg_eye_width * base_movement_ratio
            
            # 2. 分辨率补偿因子
            # 高分辨率图像需要更大的移动距离
            base_resolution = 1920 * 1080  # 基准分辨率
            resolution_factor = np.sqrt(image_area / base_resolution)
            resolution_factor = max(0.8, min(3.0, resolution_factor))  # 限制在0.8-3.0之间
            adaptive_params['resolution_factor'] = resolution_factor
            
            # 3. 眼部尺寸补偿
            # 小眼睛需要相对更大的调整，大眼睛可以相对保守
            base_eye_width = 120  # 基准眼宽(像素)
            eye_size_factor = base_eye_width / avg_eye_width if avg_eye_width > 0 else 1.0
            eye_size_factor = max(0.7, min(1.5, eye_size_factor))  # 限制范围
            adaptive_params['eye_size_factor'] = eye_size_factor
            
            # 4. 综合自适应scale
            final_scale = (adaptive_params['base_scale'] * 
                          adaptive_params['resolution_factor'] * 
                          adaptive_params['eye_size_factor'])
            
            # 确保最小有效scale
            min_effective_scale = max(15, avg_eye_width * 0.08)  # 至少眼宽的8%
            final_scale = max(final_scale, min_effective_scale)
            
            adaptive_params['final_scale'] = final_scale
            
            # 5. 自适应安全限制
            # 最大移动距离相对于眼部尺寸
            max_safe_ratio = 0.25  # 眼宽的25%为安全上限
            adaptive_params['max_displacement'] = avg_eye_width * max_safe_ratio
            
            # 6. mask尺寸调整
            # mask半径相对于眼部尺寸
            base_mask_ratio = 0.15  # 眼宽的15%
            adaptive_params['mask_radius'] = max(8, avg_eye_width * base_mask_ratio)
            
            # 调试信息
            print(f"📏 自适应参数计算:")
            print(f"   图像尺寸: {w}×{h} ({image_area/1000000:.1f}M像素)")
            print(f"   平均眼宽: {avg_eye_width:.1f}px, 眼高: {avg_eye_height:.1f}px")
            print(f"   眼部相对尺寸: {eye_to_image_ratio:.3f} (占图像宽度比例)")
            print(f"   分辨率因子: {resolution_factor:.2f}")
            print(f"   眼部尺寸因子: {eye_size_factor:.2f}")
            print(f"   最终scale: {final_scale:.1f} (目标移动距离)")
            print(f"   安全上限: {adaptive_params['max_displacement']:.1f}px")
            
            return adaptive_params
            
        except Exception as e:
            print(f"⚠️ 自适应参数计算失败: {e}")
            # 返回默认参数
            return {
                'base_scale': 20,
                'resolution_factor': 1.0,
                'eye_size_factor': 1.0,
                'final_scale': 20,
                'max_displacement': 35,
                'mask_radius': 12
            }
    
    def calculate_movement_strategy(self, features, pose_analysis, adjustment_type, adaptive_params):
        """基于美学原理的移动策略 (增强自适应版本)"""
        
        # 使用自适应scale替代固定scale
        base_scale = adaptive_params['final_scale']
        max_displacement = adaptive_params['max_displacement']
        
        # 基础策略 - 使用自适应scale
        strategy = {
            'outer_corner': {'intensity': 1.0, 'direction': 'radial', 'scale': base_scale},
            'outer_support': {'intensity': 0.8, 'direction': 'radial', 'scale': base_scale * 0.8},
            'inner_corner': {'intensity': 0.2, 'direction': 'minimal', 'scale': base_scale * 0.4},
            'upper_lid': {'intensity': 0.6, 'direction': 'arch', 'scale': base_scale * 0.7},
            'lower_lid': {'intensity': 0.4, 'direction': 'gentle', 'scale': base_scale * 0.6},
        }
        
        # 美学禁忌和增强规则
        warnings = []
        
        if features.get('upward_tilt') and adjustment_type in ['lift', 'shape']:
            # 上扬眼禁止进一步上扬
            strategy['outer_corner']['direction'] = 'horizontal_only'
            strategy['outer_support']['direction'] = 'horizontal_only'
            warnings.append("🚫 上扬眼特征：禁止进一步上扬")
        
        if features.get('very_elongated') and adjustment_type in ['stretch', 'shape']:
            # 细长眼禁止进一步拉长
            for region in strategy:
                strategy[region]['direction'] = 'vertical_priority'
                strategy[region]['scale'] *= 0.7
            warnings.append("🚫 细长眼特征：限制水平拉长")
        
        if features.get('downward_tilt'):
            # 下垂眼增强上扬 - 使用更大的scale
            strategy['outer_corner']['direction'] = 'upward_priority'
            strategy['outer_corner']['intensity'] = 1.4
            strategy['outer_corner']['scale'] = base_scale * 1.2  # 增强scale
            strategy['upper_lid']['intensity'] = 0.8
            warnings.append("✨ 下垂眼特征：启用上扬增强")
        
        if features.get('very_round'):
            # 圆眼增强拉长 - 使用更大的scale
            strategy['outer_corner']['direction'] = 'horizontal_priority'
            strategy['outer_corner']['intensity'] = 1.3
            strategy['outer_corner']['scale'] = base_scale * 1.15  # 增强scale
            strategy['outer_support']['intensity'] = 1.0
            warnings.append("✨ 圆眼特征：启用拉长增强")
        
        # 3D姿态调整
        pose_factor = pose_analysis['intensity_factor']
        for region in strategy:
            strategy[region]['intensity'] *= pose_factor
        
        if pose_analysis['difficulty'] != 'easy':
            warnings.append(f"📐 {pose_analysis['type']}姿态：强度调整为{pose_factor:.1f}x")
        
        # 不对称处理
        asymmetry_adjust = {'left': 1.0, 'right': 1.0}
        if features.get('asymmetric'):
            # 简化的不对称处理
            asymmetry_adjust = {'left': 1.1, 'right': 0.9}
            warnings.append("⚖️ 不对称特征：启用差异化调整")
        
        # 更新最大位移限制
        for region in strategy:
            strategy[region]['max_displacement'] = max_displacement
        
        for warning in warnings:
            print(warning)
        
        return strategy, asymmetry_adjust
    
    def calculate_movement_vectors(self, landmarks_2d, adjustment_type, intensity, image_shape):
        """计算移动向量 - 增强自适应版本"""
        
        # 1. 计算自适应参数
        adaptive_params = self.calculate_adaptive_parameters(landmarks_2d, image_shape)
        
        # 2. 分析眼形特征
        features, measurements = self.analyze_eye_characteristics(landmarks_2d)
        
        # 3. 计算眼球中心
        left_center = (landmarks_2d[33] + landmarks_2d[133]) / 2
        right_center = (landmarks_2d[263] + landmarks_2d[362]) / 2
        
        # 4. 获取姿态分析
        pose_analysis = getattr(self, '_current_pose_analysis', 
                               {'type': 'frontal', 'yaw': 0, 'difficulty': 'easy', 'intensity_factor': 1.0})
        
        # 5. 制定移动策略 (传入自适应参数)
        strategy, asymmetry_adjust = self.calculate_movement_strategy(features, pose_analysis, adjustment_type, adaptive_params)
        
        # 6. 计算具体移动向量
        movements = {}
        
        for eye_side in ['left_eye', 'right_eye']:
            eye_center = left_center if eye_side == 'left_eye' else right_center
            side_multiplier = asymmetry_adjust['left'] if eye_side == 'left_eye' else asymmetry_adjust['right']
            
            for region_name, point_indices in self.EYE_REGIONS[eye_side].items():
                if region_name in strategy:
                    region_strategy = strategy[region_name]
                    
                    for point_idx in point_indices:
                        if point_idx < len(landmarks_2d):
                            movement = self.calculate_point_movement(
                                landmarks_2d[point_idx], eye_center, region_strategy,
                                adjustment_type, intensity * side_multiplier, adaptive_params
                            )
                            movements[point_idx] = movement
        
        # 7. 结构保护 (传入自适应参数)
        movements = self.apply_structure_protection(movements, landmarks_2d, adaptive_params)
        
        return movements, adaptive_params
    
    def calculate_point_movement(self, point, eye_center, strategy, adjustment_type, intensity, adaptive_params):
        """计算单点移动向量 - 自适应版本"""
        
        direction_type = strategy['direction']
        region_intensity = strategy['intensity']
        scale = strategy['scale']  # 现在是自适应的scale
        max_displacement = strategy.get('max_displacement', adaptive_params['max_displacement'])
        
        # 基础方向向量
        to_point = point - eye_center
        to_point_norm = np.linalg.norm(to_point)
        if to_point_norm <= 1e-6:
            return np.array([0.0, 0.0])
        
        to_point_unit = to_point / to_point_norm
        
        # 根据方向类型计算移动向量
        if direction_type == 'minimal':
            movement_vector = to_point_unit * 0.2
        elif direction_type == 'horizontal_only':
            movement_vector = np.array([to_point_unit[0], 0])
        elif direction_type == 'vertical_priority':
            movement_vector = np.array([to_point_unit[0] * 0.3, to_point_unit[1]])
        elif direction_type == 'horizontal_priority':
            horizontal = np.array([to_point_unit[0], 0])
            vertical = np.array([0, to_point_unit[1]])
            movement_vector = horizontal * 0.8 + vertical * 0.2
        elif direction_type == 'upward_priority':
            upward = np.array([0, -1])
            movement_vector = to_point_unit * 0.4 + upward * 0.6
        else:  # 'radial', 'arch', 'gentle'
            movement_vector = to_point_unit
        
        # 归一化
        movement_norm = np.linalg.norm(movement_vector)
        if movement_norm > 1e-6:
            movement_vector = movement_vector / movement_norm
        
        # 应用调整类型的修正
        type_scales = {'stretch': 1.2, 'lift': 1.0, 'shape': 1.1}
        type_scale = type_scales.get(adjustment_type, 1.0)
        
        # 计算最终移动 - 现在scale是自适应的
        final_movement = movement_vector * intensity * region_intensity * scale * type_scale
        
        # 自适应安全限制
        movement_magnitude = np.linalg.norm(final_movement)
        if movement_magnitude > max_displacement:
            final_movement = final_movement * (max_displacement / movement_magnitude)
        
        return final_movement
    
    def apply_structure_protection(self, movements, landmarks_2d, adaptive_params):
        """简化的结构保护 - 自适应版本"""
        
        if not movements:
            return movements
        
        protected_movements = movements.copy()
        
        # 使用自适应的差异阈值
        base_scale = adaptive_params['final_scale']
        difference_threshold = base_scale * 2.0  # 相对于预期移动距离的阈值
        
        # 定义相邻点组 (简化版)
        adjacent_groups = [
            [33, 7, 163],               # 左眼外角
            [157, 158, 159, 160],       # 左眼上眼睑
            [263, 249, 390],            # 右眼外角
            [386, 387, 388, 466]        # 右眼上眼睑
        ]
        
        for group in adjacent_groups:
            valid_movements = []
            valid_indices = []
            
            for idx in group:
                if idx in movements:
                    valid_movements.append(movements[idx])
                    valid_indices.append(idx)
            
            if len(valid_movements) >= 2:
                # 自适应平滑：如果相邻点移动差异过大，进行平滑
                for i in range(len(valid_movements) - 1):
                    curr_movement = valid_movements[i]
                    next_movement = valid_movements[i + 1]
                    
                    curr_mag = np.linalg.norm(curr_movement)
                    next_mag = np.linalg.norm(next_movement)
                    
                    # 使用自适应阈值判断是否需要平滑
                    mag_diff = abs(curr_mag - next_mag)
                    if mag_diff > difference_threshold:
                        avg_mag = (curr_mag + next_mag) / 2
                        curr_dir = curr_movement / max(curr_mag, 1e-6)
                        next_dir = next_movement / max(next_mag, 1e-6)
                        avg_dir = (curr_dir + next_dir) / 2
                        avg_dir = avg_dir / max(np.linalg.norm(avg_dir), 1e-6)
                        
                        protected_movements[valid_indices[i]] = avg_dir * avg_mag * 0.9
                        protected_movements[valid_indices[i + 1]] = avg_dir * avg_mag * 0.9
        
        return protected_movements
    
    def create_eye_mask(self, image_shape, landmarks_2d, adaptive_params):
        """创建眼部mask - 自适应版本"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 收集所有眼部点
        eye_points = []
        for eye_regions in self.EYE_REGIONS.values():
            for point_list in eye_regions.values():
                for idx in point_list:
                    if idx < len(landmarks_2d):
                        eye_points.append(landmarks_2d[idx])
        
        if len(eye_points) < 8:
            return mask
        
        # 使用自适应的mask半径
        mask_radius = int(adaptive_params['mask_radius'])
        
        # 为每个点创建圆形区域
        eye_points = np.array(eye_points, dtype=np.int32)
        for point in eye_points:
            cv2.circle(mask, tuple(point), mask_radius, 255, -1)
        
        # 自适应的形态学处理
        kernel_size = max(3, int(mask_radius / 4))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def apply_eye_corner_adjustment(self, image, landmarks_2d, pose_info, adjustment_type="stretch", intensity=0.3):
        """应用眼角调整 - 主处理函数 (自适应版本)"""
        
        print(f"🎯 开始眼角调整:")
        print(f"   调整类型: {adjustment_type}")
        print(f"   调整强度: {intensity}")
        
        # 缓存姿态分析结果
        self._current_pose_analysis = self.get_pose_analysis(pose_info)
        pose_analysis = self._current_pose_analysis
        
        print(f"   3D姿态: {pose_analysis['type']} (yaw={pose_analysis['yaw']:.1f}°)")
        
        # 计算移动向量 (现在返回自适应参数)
        movement_vectors, adaptive_params = self.calculate_movement_vectors(landmarks_2d, adjustment_type, intensity, image.shape)
        
        if not movement_vectors:
            print("❌ 未生成有效移动向量")
            return image
        
        # 构建控制点
        src_points = []
        dst_points = []
        used_indices = set()
        
        # 添加移动点
        movement_count = 0
        total_movement = 0
        for point_idx, movement in movement_vectors.items():
            src_points.append(landmarks_2d[point_idx])
            dst_points.append(landmarks_2d[point_idx] + movement)
            used_indices.add(point_idx)
            movement_count += 1
            total_movement += np.linalg.norm(movement)
        
        avg_movement = total_movement / movement_count if movement_count > 0 else 0
        
        # 添加稳定锚点
        anchor_count = 0
        for idx in self.STABLE_ANCHORS:
            if idx < len(landmarks_2d) and idx not in used_indices:
                src_points.append(landmarks_2d[idx])
                dst_points.append(landmarks_2d[idx])
                used_indices.add(idx)
                anchor_count += 1
        
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
        
        print(f"🔧 控制点: 移动={movement_count}, 锚点={anchor_count}, 边界=8, 总计={len(src_points)}")
        print(f"   平均移动距离: {avg_movement:.1f}像素 (预期: {adaptive_params['final_scale']:.1f})")
        
        # 效果预期评估
        expected_range = adaptive_params['final_scale']
        if avg_movement < expected_range * 0.5:
            print("⚠️ 实际移动距离低于预期，效果可能不明显")
        elif avg_movement > expected_range * 1.5:
            print("⚠️ 实际移动距离超出预期，请检查参数")
        else:
            print("✅ 移动距离在预期范围内")
        
        # 验证控制点数量
        if len(src_points) < 15:
            print("❌ 控制点数量不足")
            return image
        
        # TPS变形
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
            
            transformed = tps.warpImage(image)
            
            if transformed is None:
                print("❌ TPS变换失败")
                return image
            
            # 创建自适应mask并融合
            eye_mask = self.create_eye_mask(image.shape, landmarks_2d, adaptive_params)
            mask_3d = cv2.merge([eye_mask, eye_mask, eye_mask]).astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_3d, (5, 5), 0)
            
            result = image.astype(np.float32) * (1 - mask_blurred) + transformed.astype(np.float32) * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print("✅ 眼角调整完成")
            
            # 效果评估
            diff = cv2.absdiff(image, result)
            total_diff = np.sum(diff)
            relative_change = total_diff / (image.shape[0] * image.shape[1] * 255 * 3)
            
            print(f"📊 变化评估:")
            print(f"   总差异: {total_diff:,}")
            print(f"   平均移动: {avg_movement:.1f}px")
            print(f"   相对变化: {relative_change:.4f}")
            print(f"   自适应scale: {adaptive_params['final_scale']:.1f}px")
            
            # 智能建议
            if relative_change < 0.001:
                print("💡 效果较小建议:")
                print("   • 当前图像分辨率较高，需要更大的调整强度")
                print("   • 建议强度范围: 0.4-0.6")
            elif relative_change > 0.01:
                print("💡 效果较大提醒:")
                print("   • 调整效果比较明显，如需更自然可降低强度")
                print("   • 建议强度范围: 0.2-0.3")
            else:
                print("✅ 调整效果适中，在理想范围内")
            
            return result
            
        except Exception as e:
            print(f"❌ TPS变形异常: {e}")
            return image
    
    def debug_eye_analysis(self, image, save_path=None):
        """调试眼形分析 - 增强自适应版本"""
        
        landmarks_2d, pose_info = self.detect_landmarks_with_pose(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        features, measurements = self.analyze_eye_characteristics(landmarks_2d)
        pose_analysis = self.get_pose_analysis(pose_info)
        
        # 计算自适应参数用于调试显示
        adaptive_params = self.calculate_adaptive_parameters(landmarks_2d, image.shape)
        
        debug_img = image.copy()
        
        # 绘制眼部区域
        colors = {
            'outer_corner': (0, 255, 0),    # 绿色
            'outer_support': (0, 200, 0),   # 深绿
            'inner_corner': (255, 0, 0),    # 红色
            'upper_lid': (0, 0, 255),       # 蓝色
            'lower_lid': (255, 255, 0),     # 黄色
        }
        
        for eye_side, regions in self.EYE_REGIONS.items():
            for region_name, point_indices in regions.items():
                color = colors.get(region_name, (128, 128, 128))
                for i, idx in enumerate(point_indices):
                    if idx < len(landmarks_2d):
                        point = landmarks_2d[idx].astype(int)
                        cv2.circle(debug_img, tuple(point), 4, color, -1)
                        cv2.putText(debug_img, f"{eye_side[0].upper()}{region_name[:3].upper()}{i}", 
                                   tuple(point + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        
        # 显示分析结果
        info_y = 30
        cv2.putText(debug_img, f"3D Pose: {pose_analysis['type']} (yaw={pose_analysis['yaw']:.1f}°)", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(debug_img, f"Eye Features:", (10, info_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        feature_y = info_y + 60
        for feature, value in features.items():
            if value:
                cv2.putText(debug_img, f"✓ {feature}", (10, feature_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                feature_y += 20
        
        cv2.putText(debug_img, f"Tilt: {measurements.get('avg_tilt', 0):.1f}°, Ratio: {measurements.get('avg_ratio', 0):.2f}", 
                   (10, feature_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # 显示自适应参数
        cv2.putText(debug_img, f"Adaptive Scale: {adaptive_params['final_scale']:.1f}px", 
                   (10, feature_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
        
        cv2.putText(debug_img, f"Resolution Factor: {adaptive_params['resolution_factor']:.2f}", 
                   (10, feature_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
        
        cv2.putText(debug_img, f"Max Displacement: {adaptive_params['max_displacement']:.1f}px", 
                   (10, feature_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
        
        # 显示眼球中心
        if len(landmarks_2d) > 362:
            left_center = (landmarks_2d[33] + landmarks_2d[133]) / 2
            right_center = (landmarks_2d[263] + landmarks_2d[362]) / 2
            cv2.circle(debug_img, tuple(left_center.astype(int)), 6, (255, 255, 0), -1)
            cv2.circle(debug_img, tuple(right_center.astype(int)), 6, (255, 255, 0), -1)
        
        # 显示自适应mask预览
        mask_preview = self.create_eye_mask(image.shape, landmarks_2d, adaptive_params)
        mask_overlay = debug_img.copy()
        mask_overlay[mask_preview > 0] = [255, 0, 255]
        debug_img = cv2.addWeighted(debug_img, 0.8, mask_overlay, 0.2, 0)
        
        if save_path:
            cv2.imwrite(save_path, debug_img)
            print(f"📁 调试图保存到: {save_path}")
            
            # 输出详细的自适应参数信息到控制台
            print(f"\n📊 详细自适应参数:")
            for key, value in adaptive_params.items():
                print(f"   {key}: {value}")
        
        return debug_img
    
    def process_image(self, image, adjustment_type="stretch", intensity=0.3, auto_optimize=True):
        """处理图像的主接口"""
        
        # 检测关键点和姿态
        landmarks_2d, pose_info = self.detect_landmarks_with_pose(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        print(f"✅ 检测到 {len(landmarks_2d)} 个关键点")
        
        # Auto-Optimize: 根据眼形特征优化参数
        if auto_optimize:
            features, measurements = self.analyze_eye_characteristics(landmarks_2d)
            pose_analysis = self.get_pose_analysis(pose_info)
            
            print("🧠 Auto-Optimize分析:")
            
            # 显示检测到的特征
            active_features = [f for f, v in features.items() if v]
            if active_features:
                print(f"   检测特征: {', '.join(active_features)}")
            else:
                print("   检测特征: 标准眼形")
            
            # 智能调整建议
            original_intensity = intensity
            
            # 基于眼形特征调整强度
            if features.get('very_round') and adjustment_type in ['stretch', 'shape']:
                intensity = max(intensity, 0.35)  # 圆眼需要更大强度
                print(f"   圆眼优化: 强度提升至 {intensity}")
            
            if features.get('downward_tilt') and adjustment_type in ['lift', 'shape']:
                intensity = max(intensity, 0.4)   # 下垂眼需要更大强度
                print(f"   下垂眼优化: 强度提升至 {intensity}")
            
            if features.get('upward_tilt') and adjustment_type == 'lift':
                adjustment_type = 'stretch'  # 自动切换调整类型
                print(f"   上扬眼优化: 调整类型切换为 {adjustment_type}")
            
            if features.get('very_elongated') and adjustment_type == 'stretch':
                adjustment_type = 'lift'     # 自动切换调整类型
                print(f"   细长眼优化: 调整类型切换为 {adjustment_type}")
            
            # 基于姿态调整强度
            if pose_analysis['difficulty'] != 'easy':
                intensity *= pose_analysis['intensity_factor']
                print(f"   姿态优化: 强度调整为 {intensity:.3f} (因子: {pose_analysis['intensity_factor']:.2f})")
            
            # 确保最小有效强度
            intensity = max(intensity, 0.2)
            
            if abs(intensity - original_intensity) > 0.05:
                print(f"   最终优化: {original_intensity:.2f} → {intensity:.2f}")
        
        # 应用调整
        result = self.apply_eye_corner_adjustment(image, landmarks_2d, pose_info, adjustment_type, intensity)
        
        return result

def main():
    parser = argparse.ArgumentParser(description='精简高效眼角调整器 - 基于美学原理的智能调整')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--type', '-t', choices=['stretch', 'lift', 'shape'], 
                       default='stretch', help='调整类型: stretch=拉长, lift=上扬, shape=综合')
    parser.add_argument('--intensity', type=float, default=0.3, 
                       help='调整强度 (0.1-0.5)')
    parser.add_argument('--auto-optimize', action='store_true', default=True,
                       help='启用智能优化 (默认开启)')
    parser.add_argument('--no-auto-optimize', dest='auto_optimize', action='store_false',
                       help='禁用智能优化')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"❌ 无法加载图片: {args.input}")
        return
    
    print(f"📸 图片信息: {image.shape[1]}×{image.shape[0]} 像素")
    
    # 强度安全检查
    if args.intensity > 0.5:
        print(f"⚠️ 强度 {args.intensity} 过大，限制到0.5")
        args.intensity = 0.5
    elif args.intensity < 0.1:
        print(f"⚠️ 强度 {args.intensity} 过小，提升到0.1")
        args.intensity = 0.1
    
    # 创建调整器
    adjuster = EyeCornerAdjuster()
    
    # 调试模式
    if args.debug:
        debug_result = adjuster.debug_eye_analysis(image, args.output)
        print(f"🔍 调试图保存到: {args.output}")
        return
    
    # 处理图片
    print(f"🚀 启动眼角调整...")
    print(f"   Auto-Optimize: {'开启' if args.auto_optimize else '关闭'}")
    
    result = adjuster.process_image(image, args.type, args.intensity, args.auto_optimize)
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"✅ 处理完成，保存到: {args.output}")
    
    # 生成对比图
    if args.comparison:
        comparison_path = args.output.replace('.', '_comparison.')
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
        
        # 添加标签
        cv2.putText(comparison, 'Original', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(comparison, f'{args.type.upper()}-{args.intensity:.2f}', 
                   (image.shape[1] + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.imwrite(comparison_path, comparison)
        print(f"📊 对比图保存到: {comparison_path}")
    
    # 最终统计
    diff = cv2.absdiff(image, result)
    total_diff = np.sum(diff)
    changed_pixels = np.sum(diff > 0)
    
    print(f"\n📊 处理统计:")
    print(f"   总像素变化: {total_diff:,}")
    print(f"   变化像素数: {changed_pixels:,}")
    print(f"   变化强度: {total_diff/(image.shape[0]*image.shape[1]*255*3):.4f}")
    
    if total_diff < 50000:
        print("💡 变化较小建议:")
        print("   • 尝试增加 --intensity 参数")
        print("   • 使用 --debug 查看眼形特征检测")
        print("   • 确保图片中人脸清晰且角度合适")
    
    print(f"\n🌟 精简眼角调整器特色:")
    print("✅ 3D姿态感知 - 真实头部姿态估计")
    print("✅ 美学导向策略 - 基于眼形特征的智能调整")
    print("✅ 结构完整性保护 - 确保自然协调的效果")
    print("✅ Auto-Optimize - 智能参数优化")
    print("✅ 精简高效 - 删除冗余功能，专注核心效果")
    
    print(f"\n📖 使用示例:")
    print("• 智能自动调整: python script.py -i input.jpg -o output.png --comparison")
    print("• 手动调整参数: python script.py -i input.jpg -o output.png -t lift --intensity 0.4")
    print("• 调试眼形分析: python script.py -i input.jpg -o debug.png --debug")
    print("• 禁用智能优化: python script.py -i input.jpg -o output.png --no-auto-optimize")

if __name__ == "__main__":
    main()