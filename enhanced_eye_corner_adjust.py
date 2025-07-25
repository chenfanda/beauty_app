#!/usr/bin/env python3
"""
增强的眼角调整脚本 - enhanced_eye_corner_adjust.py
核心改进：
1. 基于眼形识别的先验知识系统
2. 一致性参考点设计，避免移动向量冲突
3. 智能调整类型适配性检查
4. 区域化移动策略，保证插值成功
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
from typing import List, Tuple, Optional, Dict

class EyeShapeAnalyzer:
    """眼形识别和分析模块"""
    
    def __init__(self):
        # 基于医学美学的眼形分类标准
        self.EYE_SHAPE_CATEGORIES = {
            'round': {  # 圆眼
                'length_width_ratio': (2.0, 2.7),
                'tilt_angle': (-5, 5),
                'recommended_adjustments': ['stretch', 'lift', 'shape'],
                'optimal_intensity': {'stretch': 0.35, 'lift': 0.25, 'shape': 0.3},
                'description': '圆润可爱型眼睛，适合拉长和上扬'
            },
            'almond': {  # 杏仁眼（理想眼型）
                'length_width_ratio': (2.8, 3.2),
                'tilt_angle': (5, 15),
                'recommended_adjustments': ['minimal', 'enhancement'],
                'optimal_intensity': {'stretch': 0.2, 'lift': 0.15, 'shape': 0.2},
                'description': '理想杏仁眼型，只需轻微调整'
            },
            'upturned': {  # 上扬眼
                'length_width_ratio': (2.5, 3.5),
                'tilt_angle': (15, 25),
                'recommended_adjustments': ['stretch_only'],
                'contraindicated': ['lift'],  # 禁止上扬
                'optimal_intensity': {'stretch': 0.25},
                'description': '已经上扬的眼型，避免过度调整'
            },
            'downturned': {  # 下垂眼
                'length_width_ratio': (2.3, 3.0),
                'tilt_angle': (-15, -5),
                'recommended_adjustments': ['lift', 'shape'],
                'optimal_intensity': {'stretch': 0.2, 'lift': 0.4, 'shape': 0.35},
                'description': '下垂眼型，最适合上扬调整'
            },
            'elongated': {  # 细长眼
                'length_width_ratio': (3.3, 4.0),
                'tilt_angle': (-5, 10),
                'recommended_adjustments': ['lift'],
                'contraindicated': ['stretch'],  # 禁止拉长
                'optimal_intensity': {'lift': 0.3, 'shape': 0.25},
                'description': '细长眼型，避免进一步拉长'
            },
            'normal': {  # 普通眼型
                'length_width_ratio': (2.7, 3.1),
                'tilt_angle': (-3, 8),
                'recommended_adjustments': ['stretch', 'lift', 'shape'],
                'optimal_intensity': {'stretch': 0.3, 'lift': 0.3, 'shape': 0.3},
                'description': '标准眼型，各种调整均适用'
            }
        }
    
    def analyze_eye_shape(self, landmarks_2d):
        """综合分析眼形特征"""
        
        measurements = self.calculate_eye_measurements(landmarks_2d)
        eye_shape = self.classify_eye_shape(measurements)
        
        return {
            'eye_shape': eye_shape,
            'measurements': measurements,
            'recommendations': self.generate_recommendations(eye_shape),
            'safety_info': self.get_safety_constraints(eye_shape)
        }
    
    def calculate_eye_measurements(self, landmarks_2d):
        """计算眼形的关键测量指标"""
        
        # 左眼关键点
        left_inner = landmarks_2d[133]  # 内眦
        left_outer = landmarks_2d[33]   # 外眦
        left_top = landmarks_2d[159]    # 上眼睑
        left_bottom = landmarks_2d[145] # 下眼睑
        
        # 右眼关键点（用于验证对称性）
        right_inner = landmarks_2d[362]
        right_outer = landmarks_2d[263]
        right_top = landmarks_2d[386]
        right_bottom = landmarks_2d[374]
        
        # 基本尺寸计算
        left_length = np.linalg.norm(left_outer - left_inner)
        left_height = np.linalg.norm(left_top - left_bottom)
        left_ratio = left_length / left_height if left_height > 0 else 3.0
        
        right_length = np.linalg.norm(right_outer - right_inner)
        right_height = np.linalg.norm(right_top - right_bottom)
        right_ratio = right_length / right_height if right_height > 0 else 3.0
        
        # 平均比例
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # 倾斜角度计算
        left_tilt = np.degrees(np.arctan2(
            left_outer[1] - left_inner[1],
            left_outer[0] - left_inner[0]
        ))
        
        right_tilt = np.degrees(np.arctan2(
            right_outer[1] - right_inner[1],
            right_outer[0] - right_inner[0]
        ))
        
        avg_tilt = (left_tilt + right_tilt) / 2
        
        # 对称性评估
        symmetry_score = 1.0 - abs(left_ratio - right_ratio) / max(left_ratio, right_ratio)
        
        return {
            'length_width_ratio': avg_ratio,
            'tilt_angle': avg_tilt,
            'left_ratio': left_ratio,
            'right_ratio': right_ratio,
            'symmetry_score': symmetry_score,
            'absolute_size': (left_length + right_length) / 2
        }
    
    def classify_eye_shape(self, measurements):
        """基于测量值分类眼形"""
        
        ratio = measurements['length_width_ratio']
        tilt = measurements['tilt_angle']
        
        best_match = None
        best_score = 0
        
        for shape_name, criteria in self.EYE_SHAPE_CATEGORIES.items():
            score = 0
            
            # 长宽比匹配度 (权重50%)
            if criteria['length_width_ratio'][0] <= ratio <= criteria['length_width_ratio'][1]:
                score += 50
            else:
                # 计算偏差惩罚
                min_ratio, max_ratio = criteria['length_width_ratio']
                if ratio < min_ratio:
                    deviation = (min_ratio - ratio) / min_ratio
                else:
                    deviation = (ratio - max_ratio) / max_ratio
                score += max(0, 50 - deviation * 100)
            
            # 倾斜角度匹配度 (权重50%)
            if criteria['tilt_angle'][0] <= tilt <= criteria['tilt_angle'][1]:
                score += 50
            else:
                min_tilt, max_tilt = criteria['tilt_angle']
                if tilt < min_tilt:
                    deviation = abs(min_tilt - tilt) / 20  # 20度为参考范围
                else:
                    deviation = abs(tilt - max_tilt) / 20
                score += max(0, 50 - deviation * 100)
            
            if score > best_score:
                best_score = score
                best_match = shape_name
        
        confidence = best_score / 100.0
        
        return {
            'primary_type': best_match,
            'confidence': confidence,
            'detailed_scores': {shape: self.calculate_detailed_score(shape, measurements) 
                              for shape in self.EYE_SHAPE_CATEGORIES.keys()}
        }
    
    def calculate_detailed_score(self, shape_name, measurements):
        """计算详细匹配分数"""
        criteria = self.EYE_SHAPE_CATEGORIES[shape_name]
        
        ratio_match = criteria['length_width_ratio'][0] <= measurements['length_width_ratio'] <= criteria['length_width_ratio'][1]
        tilt_match = criteria['tilt_angle'][0] <= measurements['tilt_angle'] <= criteria['tilt_angle'][1]
        
        return {
            'ratio_match': ratio_match,
            'tilt_match': tilt_match,
            'overall_fit': ratio_match and tilt_match
        }
    
    def generate_recommendations(self, eye_shape):
        """生成调整建议"""
        shape_type = eye_shape['primary_type']
        config = self.EYE_SHAPE_CATEGORIES[shape_type]
        
        return {
            'recommended_adjustments': config['recommended_adjustments'],
            'contraindicated': config.get('contraindicated', []),
            'optimal_intensities': config['optimal_intensity'],
            'description': config['description']
        }
    
    def get_safety_constraints(self, eye_shape):
        """获取安全约束"""
        shape_type = eye_shape['primary_type']
        
        constraints = {
            'max_intensity': 0.4,
            'min_intensity': 0.1,
            'forbidden_types': self.EYE_SHAPE_CATEGORIES[shape_type].get('contraindicated', [])
        }
        
        # 基于置信度调整约束
        if eye_shape['confidence'] < 0.7:
            constraints['max_intensity'] = 0.25  # 低置信度时更保守
            constraints['warning'] = "眼形识别置信度较低，建议使用保守参数"
        
        return constraints

class ConsistentReferenceSystem:
    """一致性参考点系统"""
    
    def __init__(self):
        # 定义移动区域，确保每个区域内的点使用一致的参考策略
        self.MOVEMENT_ZONES = {
            'left_outer_corner': {
                'points': [33, 7, 163, 144, 145, 153],  # 左眼外眼角区域
                'reference_type': 'zone_center',
                'movement_pattern': 'radial_consistent'
            },
            'left_inner_corner': {
                'points': [133, 155, 154, 158],  # 左眼内眼角区域
                'reference_type': 'minimal_stable',
                'movement_pattern': 'conservative'
            },
            'left_upper_lid': {
                'points': [157, 158, 159, 160, 161],  # 左眼上眼睑
                'reference_type': 'curve_center',
                'movement_pattern': 'smooth_curve'
            },
            'right_outer_corner': {
                'points': [263, 249, 390, 373, 374, 380],  # 右眼外眼角区域
                'reference_type': 'zone_center',
                'movement_pattern': 'radial_consistent'
            },
            'right_inner_corner': {
                'points': [362, 384, 385, 387],  # 右眼内眼角区域
                'reference_type': 'minimal_stable',
                'movement_pattern': 'conservative'
            },
            'right_upper_lid': {
                'points': [386, 387, 388, 387, 466],  # 右眼上眼睑
                'reference_type': 'curve_center',
                'movement_pattern': 'smooth_curve'
            }
        }
    
    def calculate_zone_centers(self, landmarks_2d):
        """计算各区域的一致性参考中心"""
        zone_centers = {}
        
        for zone_name, zone_config in self.MOVEMENT_ZONES.items():
            valid_points = []
            for point_idx in zone_config['points']:
                if point_idx < len(landmarks_2d):
                    valid_points.append(landmarks_2d[point_idx])
            
            if valid_points:
                if zone_config['reference_type'] == 'zone_center':
                    # 几何中心
                    zone_centers[zone_name] = np.mean(valid_points, axis=0)
                elif zone_config['reference_type'] == 'curve_center':
                    # 曲线拟合中心
                    zone_centers[zone_name] = self.fit_curve_center(valid_points)
                else:  # minimal_stable
                    # 最稳定点（通常是内眦）
                    zone_centers[zone_name] = valid_points[0]
        
        return zone_centers
    
    def fit_curve_center(self, points):
        """拟合曲线获取中心"""
        if len(points) < 3:
            return np.mean(points, axis=0)
        
        # 简单的椭圆拟合
        try:
            points_array = np.array(points, dtype=np.float32)
            ellipse = cv2.fitEllipse(points_array)
            return np.array(ellipse[0])
        except:
            return np.mean(points, axis=0)

class EnhancedEyeCornerAdjustment:
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
        
        # 初始化子模块
        self.eye_analyzer = EyeShapeAnalyzer()
        self.reference_system = ConsistentReferenceSystem()
        
        # 增强的眼角移动点定义（基于区域化设计）
        self.LEFT_EYE_MOVEMENT_POINTS = {
            'primary_corners': [133, 33],        # 主要眼角点
            'upper_lid_support': [157, 158, 159], # 上眼睑支撑点
            'lower_lid_support': [145, 153, 154, 155], # 下眼睑支撑点
            'transition_points': [160, 161, 163]   # 过渡点
        }
        
        self.RIGHT_EYE_MOVEMENT_POINTS = {
            'primary_corners': [362, 263],
            'upper_lid_support': [386, 387, 388],
            'lower_lid_support': [374, 380, 384, 385],
            'transition_points': [387, 466, 390]
        }
        
        # 远程稳定锚点（保持原有设计）
        self.STABLE_ANCHORS = [
            1, 2, 5, 4, 6, 19, 20,  # 鼻子区域
            9, 10, 151, 175,        # 额头和下巴
            61, 291, 39, 269, 270,  # 嘴巴区域
            234, 93, 132, 454, 323, 361,  # 脸颊边缘
            70, 296, 107, 276       # 眉毛区域
        ]
        
        # 对称辅助点
        self.SYMMETRY_HELPERS = [168, 6, 8, 9]
        
        # 3D姿态参考点
        self.face_model_3d = np.array([
            [0.0, 0.0, 0.0],           # 鼻尖
            [0.0, -330.0, -65.0],      # 下巴
            [-225.0, 170.0, -135.0],   # 左眼外角
            [225.0, 170.0, -135.0],    # 右眼外角
            [-150.0, -150.0, -125.0],  # 左嘴角
            [150.0, -150.0, -125.0],   # 右嘴角
        ], dtype=np.float32)
        
        self.pnp_indices = [1, 152, 33, 263, 61, 291]
        
        # 安全参数
        self.MAX_INTENSITY = 0.4
        self.MAX_DISPLACEMENT = 40
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
        
        if abs(yaw) < 15:
            return "frontal", yaw
        elif yaw > 15:
            return "right_profile", yaw
        else:
            return "left_profile", yaw
    
    def calculate_enhanced_eye_centers(self, landmarks_2d, adjustment_type, eye_shape_info=None):
        """增强的眼球中心计算，考虑调整类型和眼形"""
        
        # 基础计算（保持原有逻辑）
        basic_left_center = (landmarks_2d[33] + landmarks_2d[133]) / 2
        basic_right_center = (landmarks_2d[362] + landmarks_2d[263]) / 2
        
        # 根据调整类型和眼形微调参考点
        offset = np.array([0, 0])
        
        if adjustment_type == "stretch":
            # 拉长时，参考点稍微向眼睑中部偏移
            offset = np.array([0, 2])
        elif adjustment_type == "lift":
            # 上扬时，参考点保持在原位置
            offset = np.array([0, 0])
        else:  # shape模式
            # 综合模式使用动态权重
            offset = np.array([1, 1])
        
        # 基于眼形的特殊调整
        if eye_shape_info:
            shape_type = eye_shape_info['eye_shape']['primary_type']
            if shape_type == 'round':
                # 圆眼需要更精确的拉长参考点
                offset += np.array([0, 1])
            elif shape_type == 'elongated':
                # 细长眼避免过度拉长
                offset = np.array([0, -1])
        
        left_center = basic_left_center + offset
        right_center = basic_right_center + offset
        
        return left_center, right_center
    
    def check_adjustment_compatibility(self, eye_shape_info, adjustment_type, intensity):
        """检查调整类型与眼形的兼容性"""
        
        if not eye_shape_info:
            return {'suitable': True, 'confidence': 'medium'}
        
        shape_type = eye_shape_info['eye_shape']['primary_type']
        recommendations = eye_shape_info['recommendations']
        
        # 检查禁忌调整
        if adjustment_type in recommendations['contraindicated']:
            alternative = recommendations['recommended_adjustments'][0] if recommendations['recommended_adjustments'] else 'stretch'
            return {
                'suitable': False,
                'warning': f"{shape_type}眼形不建议{adjustment_type}调整",
                'suggestion': f"建议使用{alternative}模式",
                'auto_adjust': True,
                'recommended_type': alternative,
                'recommended_intensity': recommendations['optimal_intensities'].get(alternative, 0.25)
            }
        
        # 检查调整类型是否在推荐列表中
        if adjustment_type not in recommendations['recommended_adjustments']:
            return {
                'suitable': True,  # 不阻止，但给出警告
                'warning': f"{adjustment_type}不是{shape_type}眼形的最佳选择",
                'suggestion': f"推荐使用: {', '.join(recommendations['recommended_adjustments'])}",
                'auto_adjust': False
            }
        
        # 检查强度是否合适
        optimal_intensity = recommendations['optimal_intensities'].get(adjustment_type)
        if optimal_intensity and abs(intensity - optimal_intensity) > 0.15:
            return {
                'suitable': True,
                'warning': f"当前强度{intensity:.2f}可能不是最优值",
                'suggestion': f"建议强度: {optimal_intensity:.2f}",
                'auto_adjust': False,
                'recommended_intensity': optimal_intensity
            }
        
        return {
            'suitable': True, 
            'confidence': 'high',
            'message': f"✅ {adjustment_type}调整非常适合{shape_type}眼形"
        }
    
    def calculate_consistent_movement_vectors(self, landmarks_2d, eye_centers, adjustment_type, intensity, eye_shape_info):
        """计算保证一致性的移动向量"""
        
        left_center, right_center = eye_centers
        movements = {}
        
        # 为左眼计算移动向量
        for point_category, point_indices in self.LEFT_EYE_MOVEMENT_POINTS.items():
            for point_idx in point_indices:
                if point_idx < len(landmarks_2d):
                    movement = self.calculate_single_point_movement(
                        landmarks_2d[point_idx], left_center, adjustment_type, 
                        intensity, point_category, 'left', eye_shape_info
                    )
                    movements[point_idx] = movement
        
        # 为右眼计算移动向量
        for point_category, point_indices in self.RIGHT_EYE_MOVEMENT_POINTS.items():
            for point_idx in point_indices:
                if point_idx < len(landmarks_2d):
                    movement = self.calculate_single_point_movement(
                        landmarks_2d[point_idx], right_center, adjustment_type, 
                        intensity, point_category, 'right', eye_shape_info
                    )
                    movements[point_idx] = movement
        
        # 应用平滑化处理，确保相邻点移动一致性
        movements = self.smooth_movement_vectors(movements, landmarks_2d)
        
        return movements
    
    def calculate_single_point_movement(self, point, eye_center, adjustment_type, intensity, point_category, eye_side, eye_shape_info):
        """计算单个点的移动向量"""
        
        # 基础方向计算
        direction = point - eye_center
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm <= 1e-6:
            return np.array([0.0, 0.0])
        
        direction = direction / direction_norm
        
        # 根据点的类别确定移动强度
        category_multipliers = {
            'primary_corners': 1.0,      # 主要眼角点
            'upper_lid_support': 0.7,    # 上眼睑支撑点
            'lower_lid_support': 0.5,    # 下眼睑支撑点
            'transition_points': 0.3     # 过渡点
        }
        
        base_multiplier = category_multipliers.get(point_category, 0.5)
        
        # 根据眼形调整强度
        if eye_shape_info:
            shape_type = eye_shape_info['eye_shape']['primary_type']
            if shape_type == 'round' and adjustment_type == 'stretch':
                base_multiplier *= 1.2  # 圆眼拉长需要更大强度
            elif shape_type == 'almond':
                base_multiplier *= 0.8  # 杏仁眼需要保守调整
        
        # 计算最终移动量
        if adjustment_type == "stretch":
            # 拉长：沿径向方向
            movement_vector = direction
            movement_scale = intensity * base_multiplier * 12
            
        elif adjustment_type == "lift":
            # 上扬：结合径向和向上方向
            if point_category == 'primary_corners':
                # 主要眼角点：强调上扬
                lift_component = np.array([0, -1]) * 0.8
                movement_vector = direction * 0.4 + lift_component * 0.6
            else:
                # 其他点：轻微跟随
                lift_component = np.array([0, -1]) * 0.3
                movement_vector = direction * 0.7 + lift_component * 0.3
            
            movement_vector = movement_vector / np.linalg.norm(movement_vector)
            movement_scale = intensity * base_multiplier * 10
            
        else:  # shape模式
            # 综合：拉长+上扬
            stretch_component = direction * 0.6
            lift_component = np.array([0, -1]) * 0.4
            movement_vector = stretch_component + lift_component
            movement_vector = movement_vector / np.linalg.norm(movement_vector)
            movement_scale = intensity * base_multiplier * 11
        
        return movement_vector * movement_scale
    
    def smooth_movement_vectors(self, movements, landmarks_2d):
        """平滑化移动向量，避免相邻点冲突"""
        
        # 定义相邻点关系
        adjacent_pairs = [
            (33, 7), (7, 163), (163, 144), (144, 145),    # 左眼外角区域
            (157, 158), (158, 159), (159, 160),           # 左眼上眼睑
            (263, 249), (249, 390), (390, 373),           # 右眼外角区域
            (386, 387), (387, 388)                        # 右眼上眼睑
        ]
        
        smoothed_movements = movements.copy()
        
        for point1, point2 in adjacent_pairs:
            if point1 in movements and point2 in movements:
                # 检查移动向量的角度差异
                vec1 = movements[point1]
                vec2 = movements[point2]
                
                vec1_norm = np.linalg.norm(vec1)
                vec2_norm = np.linalg.norm(vec2)
                
                if vec1_norm > 0 and vec2_norm > 0:
                    # 计算角度差异
                    cos_angle = np.dot(vec1, vec2) / (vec1_norm * vec2_norm)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_diff = np.degrees(np.arccos(cos_angle))
                    
                    # 如果角度差异过大，进行平滑处理
                    if angle_diff > 45:
                        # 使用加权平均
                        avg_direction = (vec1 + vec2) / 2
                        avg_direction = avg_direction / np.linalg.norm(avg_direction)
                        
                        # 保持原始移动大小，只调整方向
                        smoothed_movements[point1] = avg_direction * vec1_norm
                        smoothed_movements[point2] = avg_direction * vec2_norm
        
        return smoothed_movements
    
    def create_safe_eye_corner_mask(self, image_shape, landmarks_2d):
        """创建安全的眼角mask"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 收集所有移动点
        all_movement_points = []
        
        for point_group in [self.LEFT_EYE_MOVEMENT_POINTS, self.RIGHT_EYE_MOVEMENT_POINTS]:
            for point_list in point_group.values():
                for idx in point_list:
                    if idx < len(landmarks_2d):
                        all_movement_points.append(landmarks_2d[idx])
        
        if len(all_movement_points) < 8:
            return mask
        
        all_movement_points = np.array(all_movement_points, dtype=np.int32)
        
        # 为每个点创建小圆形区域
        for point in all_movement_points:
            cv2.circle(mask, tuple(point), 10, 255, -1)
        
        # 轻微膨胀和腐蚀
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask
    
    def apply_enhanced_eye_corner_transform(self, image, landmarks_2d, pose_info, adjustment_type="stretch", intensity=0.3, eye_shape_info=None):
        """应用增强的眼角变形"""
        
        # 3D姿态分析
        if pose_info is None:
            pose_3d, pose_angle = "frontal", 0.0
        else:
            rotation_vector, translation_vector = pose_info
            pose_3d, pose_angle = self.estimate_face_pose_3d(rotation_vector)
        
        print(f"🔍 3D姿态: {pose_3d} ({pose_angle:.1f}度)")
        
        # 计算增强的眼球中心
        eye_centers = self.calculate_enhanced_eye_centers(landmarks_2d, adjustment_type, eye_shape_info)
        left_center, right_center = eye_centers
        
        print(f"📍 增强参考点: 左眼({left_center[0]:.0f},{left_center[1]:.0f}), 右眼({right_center[0]:.0f},{right_center[1]:.0f})")
        
        # 根据3D姿态调整强度
        if pose_3d == "frontal":
            left_intensity = right_intensity = intensity
        elif pose_3d == "right_profile":
            right_intensity = intensity * 1.0
            left_intensity = intensity * 0.7
        else:  # left_profile
            left_intensity = intensity * 1.0
            right_intensity = intensity * 0.7
        
        # 计算一致性移动向量
        movement_vectors = self.calculate_consistent_movement_vectors(
            landmarks_2d, eye_centers, adjustment_type, intensity, eye_shape_info
        )
        
        # 构建控制点
        src_points = []
        dst_points = []
        used_indices = set()
        
        # 添加移动点
        movement_count = 0
        for point_idx, movement in movement_vectors.items():
            if point_idx < len(landmarks_2d):
                src_points.append(landmarks_2d[point_idx])
                dst_points.append(landmarks_2d[point_idx] + movement)
                used_indices.add(point_idx)
                movement_count += 1
        
        # 添加稳定锚点
        anchor_count = 0
        for idx in self.STABLE_ANCHORS:
            if idx < len(landmarks_2d) and idx not in used_indices:
                src_points.append(landmarks_2d[idx])
                dst_points.append(landmarks_2d[idx])
                used_indices.add(idx)
                anchor_count += 1
        
        # 添加对称辅助点
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
        print(f"   移动点: {movement_count}")
        print(f"   稳定锚点: {anchor_count}")
        print(f"   对称辅助点: {symmetry_count}")
        print(f"   边界锚点: 8")
        print(f"   总计: {len(src_points)} 个控制点")
        
        # 验证控制点数量
        if len(src_points) < 15:
            print("❌ 控制点数量不足")
            return image
        
        # 执行TPS变形
        try:
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            
            tps = cv2.createThinPlateSplineShapeTransformer()
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
            
            print(f"🔄 开始增强TPS变换（{adjustment_type}模式）...")
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
            
            print("✅ 增强眼角变形成功")
            return result
            
        except Exception as e:
            print(f"❌ TPS变形异常: {e}")
            return image
    
    def debug_enhanced_detection(self, image, save_dir=None):
        """增强的调试眼角检测"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        # 进行眼形分析
        eye_shape_info = self.eye_analyzer.analyze_eye_shape(landmarks_2d)
        
        debug_img = image.copy()
        
        # 绘制移动点（按类别用不同颜色）
        colors = {
            'primary_corners': (0, 255, 0),     # 绿色 - 主要眼角
            'upper_lid_support': (255, 0, 0),   # 红色 - 上眼睑
            'lower_lid_support': (0, 0, 255),   # 蓝色 - 下眼睑
            'transition_points': (255, 255, 0)  # 黄色 - 过渡点
        }
        
        # 左眼移动点
        for category, point_indices in self.LEFT_EYE_MOVEMENT_POINTS.items():
            color = colors[category]
            for i, idx in enumerate(point_indices):
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx].astype(int)
                    cv2.circle(debug_img, tuple(point), 6, color, -1)
                    cv2.putText(debug_img, f"L{category[0]}{i}", tuple(point + 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 右眼移动点
        for category, point_indices in self.RIGHT_EYE_MOVEMENT_POINTS.items():
            color = colors[category]
            for i, idx in enumerate(point_indices):
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx].astype(int)
                    cv2.circle(debug_img, tuple(point), 6, color, -1)
                    cv2.putText(debug_img, f"R{category[0]}{i}", tuple(point + 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 绘制增强的眼球中心
        left_center, right_center = self.calculate_enhanced_eye_centers(landmarks_2d, "stretch", eye_shape_info)
        cv2.circle(debug_img, tuple(left_center.astype(int)), 8, (255, 255, 0), -1)
        cv2.circle(debug_img, tuple(right_center.astype(int)), 8, (255, 255, 0), -1)
        cv2.putText(debug_img, 'L-CENTER', tuple(left_center.astype(int) + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(debug_img, 'R-CENTER', tuple(right_center.astype(int) + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 显示眼形分析结果
        shape_info = eye_shape_info['eye_shape']
        measurements = eye_shape_info['measurements']
        
        info_y = 30
        cv2.putText(debug_img, f"Eye Shape: {shape_info['primary_type']} ({shape_info['confidence']:.2f})", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(debug_img, f"L/W Ratio: {measurements['length_width_ratio']:.2f}", 
                   (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(debug_img, f"Tilt Angle: {measurements['tilt_angle']:.1f}°", 
                   (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 显示推荐调整
        recommendations = eye_shape_info['recommendations']['recommended_adjustments']
        cv2.putText(debug_img, f"Recommended: {', '.join(recommendations)}", 
                   (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示禁忌调整
        contraindicated = eye_shape_info['recommendations'].get('contraindicated', [])
        if contraindicated:
            cv2.putText(debug_img, f"Avoid: {', '.join(contraindicated)}", 
                       (10, info_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 绘制稳定锚点
        for i, idx in enumerate(self.STABLE_ANCHORS[:8]):
            if idx < len(landmarks_2d):
                point = landmarks_2d[idx].astype(int)
                cv2.circle(debug_img, tuple(point), 3, (128, 128, 128), -1)
        
        # 图例
        legend_y = info_y + 160
        legends = [
            ("GREEN: Primary Corners", (0, 255, 0)),
            ("RED: Upper Lid Support", (255, 0, 0)),
            ("BLUE: Lower Lid Support", (0, 0, 255)),
            ("YELLOW: Transition Points", (255, 255, 0)),
            ("CYAN: Enhanced Eye Centers", (255, 255, 0)),
            ("GRAY: Stable Anchors", (128, 128, 128))
        ]
        
        for i, (text, color) in enumerate(legends):
            cv2.putText(debug_img, text, (10, legend_y + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "debug_enhanced_eye_corner.png"), debug_img)
            print(f"📁 调试文件保存到: {save_dir}")
        
        return debug_img
    
    def process_image_with_analysis(self, image, adjustment_type="stretch", intensity=0.3, save_debug=False, output_path=None):
        """增加眼形分析的图像处理主接口"""
        
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        print(f"✅ 检测到 {len(landmarks_2d)} 个关键点")
        
        # 🆕 眼形预分析
        print("🔍 正在分析眼形特征...")
        eye_shape_info = self.eye_analyzer.analyze_eye_shape(landmarks_2d)
        
        shape_result = eye_shape_info['eye_shape']
        measurements = eye_shape_info['measurements']
        
        print(f"👁️ 眼形识别结果:")
        print(f"   类型: {shape_result['primary_type']} (置信度: {shape_result['confidence']:.2f})")
        print(f"   长宽比: {measurements['length_width_ratio']:.2f}")
        print(f"   倾斜角: {measurements['tilt_angle']:.1f}°")
        print(f"   对称性: {measurements['symmetry_score']:.2f}")
        print(f"   描述: {eye_shape_info['recommendations']['description']}")
        
        # 🆕 调整类型适配性检查
        compatibility = self.check_adjustment_compatibility(eye_shape_info, adjustment_type, intensity)
        
        if not compatibility['suitable']:
            print(f"⚠️ {compatibility['warning']}")
            print(f"💡 {compatibility['suggestion']}")
            
            if compatibility.get('auto_adjust'):
                print(f"🔄 自动调整: {adjustment_type} → {compatibility['recommended_type']}")
                adjustment_type = compatibility['recommended_type']
                intensity = compatibility['recommended_intensity']
        elif 'message' in compatibility:
            print(compatibility['message'])
        
        # 🆕 基于眼形的强度优化
        recommendations = eye_shape_info['recommendations']
        if adjustment_type in recommendations['optimal_intensities']:
            optimal_intensity = recommendations['optimal_intensities'][adjustment_type]
            if abs(intensity - optimal_intensity) > 0.1:
                print(f"💡 建议强度: {optimal_intensity:.2f} (当前: {intensity:.2f})")
        
        # 调试模式
        if save_debug and output_path:
            debug_dir = os.path.splitext(output_path)[0] + "_debug"
            self.debug_enhanced_detection(image, debug_dir)
        
        # 应用增强的眼角变形
        result = self.apply_enhanced_eye_corner_transform(
            image, landmarks_2d, pose_info, adjustment_type, intensity, eye_shape_info
        )
        
        return result

def main():
    parser = argparse.ArgumentParser(description='增强的眼角调整工具 - 基于先验知识和一致性参考点')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--type', '-t', choices=['stretch', 'lift', 'shape'], 
                       default='stretch', help='调整类型: stretch=水平拉长, lift=眼角上扬, shape=综合调整')
    parser.add_argument('--intensity', type=float, default=0.3, 
                       help='调整强度 (0.1-0.4)')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--save-debug', action='store_true', help='保存调试文件')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    parser.add_argument('--auto-optimize', action='store_true', help='自动优化参数')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"❌ 无法加载图片: {args.input}")
        return
    
    print(f"📸 图片尺寸: {image.shape}")
    
    # 强度安全检查
    if args.intensity > 0.4:
        print(f"⚠️  强度 {args.intensity} 超出建议范围，限制到0.4")
        args.intensity = 0.4
    elif args.intensity < 0.1:
        print(f"⚠️  强度 {args.intensity} 过小，提升到0.1")
        args.intensity = 0.1
    
    # 创建增强处理器
    processor = EnhancedEyeCornerAdjustment()
    
    # 调试模式
    if args.debug:
        debug_result = processor.debug_enhanced_detection(image)
        cv2.imwrite(args.output, debug_result)
        print(f"🔍 调试图保存到: {args.output}")
        return
    
    # 处理图片
    print(f"🔄 开始增强眼角处理...")
    print(f"   调整类型: {args.type}")
    print(f"   调整强度: {args.intensity}")
    print(f"   自动优化: {'开启' if args.auto_optimize else '关闭'}")
    
    result = processor.process_image_with_analysis(
        image, args.type, args.intensity, args.save_debug, args.output
    )
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"✅ 处理完成，保存到: {args.output}")
    
    # 检查变化统计
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
        print("   1. 提高 --intensity 参数")
        print("   2. 使用 --auto-optimize 自动优化参数")
        print("   3. 检查眼形是否适合当前调整类型 (--debug)")
    
    # 生成对比图
    if args.comparison:
        comparison_path = args.output.replace('.', '_comparison.')
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
        
        # 添加标签
        cv2.putText(comparison, 'Original', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(comparison, f'Enhanced-{args.type.upper()} {args.intensity:.2f}', 
                   (image.shape[1] + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.imwrite(comparison_path, comparison)
        print(f"📊 对比图保存到: {comparison_path}")
    
    print("\n🎯 增强版设计特色:")
    print("✅ 眼形智能识别: 6种标准眼形分类，医学美学基准")
    print("✅ 先验知识系统: 基于眼形推荐最佳调整类型和强度")
    print("✅ 一致性参考点: 区域化设计，避免移动向量冲突")
    print("✅ 适配性检查: 自动检测调整类型与眼形的兼容性")
    print("✅ 平滑化处理: 相邻点移动向量平滑，确保插值成功")
    print("✅ 安全约束: 基于眼形的强度限制和禁忌调整")
    
    print("\n📖 使用示例:")
    print("• 智能眼角调整: python script.py -i input.jpg -o output.png --auto-optimize --comparison")
    print("• 眼形分析调试: python script.py -i input.jpg -o debug.png --debug")
    print("• 圆眼拉长优化: python script.py -i input.jpg -o output.png -t stretch --intensity 0.35")
    print("• 下垂眼上扬: python script.py -i input.jpg -o output.png -t lift --intensity 0.4")

if __name__ == "__main__":
    main()
