#!/usr/bin/env python3
"""
3D姿态感知的眼角调整脚本 - eye_corner_3d_complete.py
核心特色：
1. 基于PnP算法的真实3D头部姿态估计
2. 3D感知的自适应眼角变形策略
3. 眼形识别 + 3D姿态的双重智能控制
4. 透视校正的参考点计算
5. 姿态自适应的移动向量和强度
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
from typing import List, Tuple, Optional, Dict

class EyeShape3DAnalyzer:
    """3D感知的眼形识别和分析模块"""
    
    def __init__(self):
        # 基于医学美学的眼形分类标准
        self.EYE_SHAPE_CATEGORIES = {
            'round': {
                'length_width_ratio': (2.0, 2.7),
                'tilt_angle': (-5, 5),
                'recommended_adjustments': ['stretch', 'lift', 'shape'],
                'pose_sensitivity': {'frontal': 1.0, 'profile': 0.8},
                'description': '圆润可爱型眼睛，适合拉长和上扬'
            },
            'almond': {
                'length_width_ratio': (2.8, 3.2),
                'tilt_angle': (5, 15),
                'recommended_adjustments': ['minimal', 'enhancement'],
                'pose_sensitivity': {'frontal': 0.7, 'profile': 0.5},
                'description': '理想杏仁眼型，只需轻微调整'
            },
            'upturned': {
                'length_width_ratio': (2.5, 3.5),
                'tilt_angle': (15, 25),
                'recommended_adjustments': ['stretch_only'],
                'contraindicated': ['lift'],
                'pose_sensitivity': {'frontal': 0.8, 'profile': 0.6},
                'description': '已经上扬的眼型，避免过度调整'
            },
            'downturned': {
                'length_width_ratio': (2.3, 3.0),
                'tilt_angle': (-15, -5),
                'recommended_adjustments': ['lift', 'shape'],
                'pose_sensitivity': {'frontal': 1.2, 'profile': 1.0},
                'description': '下垂眼型，最适合上扬调整'
            },
            'elongated': {
                'length_width_ratio': (3.3, 4.0),
                'tilt_angle': (-5, 10),
                'recommended_adjustments': ['lift'],
                'contraindicated': ['stretch'],
                'pose_sensitivity': {'frontal': 0.6, 'profile': 0.8},
                'description': '细长眼型，避免进一步拉长'
            },
            'normal': {
                'length_width_ratio': (2.7, 3.1),
                'tilt_angle': (-3, 8),
                'recommended_adjustments': ['stretch', 'lift', 'shape'],
                'pose_sensitivity': {'frontal': 1.0, 'profile': 0.9},
                'description': '标准眼型，各种调整均适用'
            }
        }
    
    def analyze_eye_shape_3d(self, landmarks_2d, pose_info):
        """基于3D姿态的眼形分析"""
        measurements = self.calculate_eye_measurements(landmarks_2d)
        
        # 3D姿态修正
        if pose_info is not None:
            measurements = self.correct_measurements_for_3d_pose(measurements, pose_info)
        
        eye_shape = self.classify_eye_shape(measurements)
        
        # 3D姿态修正分类置信度
        if pose_info is not None:
            eye_shape = self.adjust_classification_for_pose(eye_shape, pose_info)
        
        return {
            'eye_shape': eye_shape,
            'measurements': measurements,
            'recommendations': self.generate_3d_recommendations(eye_shape, pose_info)
        }
    
    def calculate_eye_measurements(self, landmarks_2d):
        """计算眼形的关键测量指标"""
        # 左眼关键点
        left_inner = landmarks_2d[133]
        left_outer = landmarks_2d[33]
        left_top = landmarks_2d[159]
        left_bottom = landmarks_2d[145]
        
        # 右眼关键点
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
        
        return {
            'length_width_ratio': avg_ratio,
            'tilt_angle': avg_tilt,
            'left_ratio': left_ratio,
            'right_ratio': right_ratio,
            'symmetry_score': 1.0 - abs(left_ratio - right_ratio) / max(left_ratio, right_ratio),
            'absolute_size': (left_length + right_length) / 2
        }
    
    def correct_measurements_for_3d_pose(self, measurements, pose_info):
        """3D姿态透视校正"""
        rotation_vector, translation_vector = pose_info
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # 计算yaw角度
        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        yaw = np.degrees(np.arctan2(-rotation_matrix[2,0], sy))
        
        # 透视校正因子
        perspective_factor = np.cos(np.radians(abs(yaw)))
        
        # 校正长宽比（侧脸时眼睛被压缩）
        corrected_ratio = measurements['length_width_ratio'] / max(perspective_factor, 0.7)
        
        # 校正倾斜角（侧脸时角度被扭曲）
        if abs(yaw) > 20:
            angle_correction = yaw * 0.3
            corrected_tilt = measurements['tilt_angle'] - angle_correction
        else:
            corrected_tilt = measurements['tilt_angle']
        
        measurements_corrected = measurements.copy()
        measurements_corrected['length_width_ratio'] = corrected_ratio
        measurements_corrected['tilt_angle'] = corrected_tilt
        measurements_corrected['pose_corrected'] = True
        measurements_corrected['yaw_angle'] = yaw
        
        print(f"🔄 3D透视校正: yaw={yaw:.1f}°, 比例{measurements['length_width_ratio']:.2f}→{corrected_ratio:.2f}")
        
        return measurements_corrected
    
    def adjust_classification_for_pose(self, eye_shape, pose_info):
        """基于3D姿态调整分类置信度"""
        rotation_vector, _ = pose_info
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        yaw = np.degrees(np.arctan2(-rotation_matrix[2,0], sy))
        
        # 侧脸时降低分类置信度
        if abs(yaw) > 30:
            eye_shape['confidence'] *= 0.8
            eye_shape['pose_warning'] = f"大角度侧脸(yaw={yaw:.1f}°)，分类精度降低"
        elif abs(yaw) > 15:
            eye_shape['confidence'] *= 0.9
        
        return eye_shape
    
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
                    deviation = abs(min_tilt - tilt) / 20
                else:
                    deviation = abs(tilt - max_tilt) / 20
                score += max(0, 50 - deviation * 100)
            
            if score > best_score:
                best_score = score
                best_match = shape_name
        
        confidence = best_score / 100.0
        
        return {
            'primary_type': best_match,
            'confidence': confidence
        }
    
    def generate_3d_recommendations(self, eye_shape, pose_info):
        """生成3D感知的调整建议"""
        shape_type = eye_shape['primary_type']
        config = self.EYE_SHAPE_CATEGORIES[shape_type]
        
        base_recommendations = {
            'recommended_adjustments': config['recommended_adjustments'],
            'contraindicated': config.get('contraindicated', []),
            'description': config['description']
        }
        
        # 3D姿态感知的强度调整
        if pose_info is not None:
            rotation_vector, _ = pose_info
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
            yaw = np.degrees(np.arctan2(-rotation_matrix[2,0], sy))
            
            # 根据姿态调整推荐强度
            if abs(yaw) < 15:
                pose_type = 'frontal'
                pose_multiplier = config['pose_sensitivity']['frontal']
            else:
                pose_type = 'profile'
                pose_multiplier = config['pose_sensitivity']['profile']
            
            base_recommendations['pose_type'] = pose_type
            base_recommendations['pose_multiplier'] = pose_multiplier
            base_recommendations['yaw_angle'] = yaw
        
        return base_recommendations

class EyeCorner3DProcessor:
    """3D感知的眼角处理器"""
    
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
        
        # 初始化眼形分析器
        self.eye_analyzer = EyeShape3DAnalyzer()
        
        # 3D面部模型（毫米单位）
        self.face_model_3d = np.array([
            [0.0, 0.0, 0.0],           # 鼻尖 (1)
            [0.0, -330.0, -65.0],      # 下巴 (152)
            [-225.0, 170.0, -135.0],   # 左眼外角 (33)
            [225.0, 170.0, -135.0],    # 右眼外角 (263)
            [-150.0, -150.0, -125.0],  # 左嘴角 (61)
            [150.0, -150.0, -125.0],   # 右嘴角 (291)
        ], dtype=np.float32)
        
        self.pnp_indices = [1, 152, 33, 263, 61, 291]
        
        # 3D感知的眼角移动点定义
        self.EYE_MOVEMENT_REGIONS = {
            'left_eye': {
                'outer_corner': [33],           # 左眼外眦
                'inner_corner': [133],          # 左眼内眦
                'upper_lid': [157, 158, 159],   # 上眼睑
                'lower_lid': [145, 153, 154],   # 下眼睑
                'transition': [160, 161, 163]   # 过渡区域
            },
            'right_eye': {
                'outer_corner': [263],          # 右眼外眦
                'inner_corner': [362],          # 右眼内眦
                'upper_lid': [386, 387, 388],   # 上眼睑
                'lower_lid': [374, 380, 384],   # 下眼睑
                'transition': [385, 390, 466]   # 过渡区域
            }
        }
        
        # 远程稳定锚点
        self.STABLE_ANCHORS = [
            1, 2, 5, 4, 6, 19, 20,      # 鼻子区域
            9, 10, 151, 175,            # 额头和下巴
            61, 291, 39, 269, 270,      # 嘴巴区域
            234, 93, 132, 454, 323, 361, # 脸颊边缘
            70, 296, 107, 276           # 眉毛区域
        ]
        
        # 对称辅助点
        self.SYMMETRY_HELPERS = [168, 6, 8, 9]
    
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
            print("⚠️ PnP关键点不足，3D姿态估计失败")
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
                print("⚠️ PnP求解失败")
                return landmarks_2d, None
                
        except Exception as e:
            print(f"⚠️ PnP求解异常: {e}")
            return landmarks_2d, None
    
    def estimate_face_pose_3d(self, rotation_vector):
        """基于3D旋转向量估计面部姿态"""
        if rotation_vector is None:
            return "frontal", 0.0, 0.0, 0.0
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # 计算欧拉角
        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        
        if sy > 1e-6:
            x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = 0
        
        # 转换为度数
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)
        
        # 判断主要姿态
        if abs(yaw) < 15 and abs(pitch) < 15:
            pose_type = "frontal"
        elif yaw > 15:
            pose_type = "right_profile"
        elif yaw < -15:
            pose_type = "left_profile"
        elif pitch > 15:
            pose_type = "looking_down"
        elif pitch < -15:
            pose_type = "looking_up"
        else:
            pose_type = "mixed"
        
        return pose_type, yaw, pitch, roll
    
    def calculate_3d_aware_eye_centers(self, landmarks_2d, pose_info, adjustment_type):
        """3D感知的眼球中心计算"""
        # 基础几何中心
        basic_left_center = (landmarks_2d[33] + landmarks_2d[133]) / 2
        basic_right_center = (landmarks_2d[362] + landmarks_2d[263]) / 2
        
        if pose_info is None:
            return basic_left_center, basic_right_center
        
        rotation_vector, translation_vector = pose_info
        pose_type, yaw, pitch, roll = self.estimate_face_pose_3d(rotation_vector)
        
        # 3D透视校正
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        
        # 透视变形校正矩阵
        perspective_factor_x = np.cos(yaw_rad)
        perspective_factor_y = np.cos(pitch_rad)
        
        # 根据调整类型和3D姿态调整参考点
        if adjustment_type == "stretch":
            if abs(yaw) > 20:
                left_offset = np.array([2 / perspective_factor_x, 0])
                right_offset = np.array([2 / perspective_factor_x, 0])
            else:
                left_offset = np.array([0, 2])
                right_offset = np.array([0, 2])
                
        elif adjustment_type == "lift":
            if abs(pitch) > 20:
                vertical_correction = pitch * 0.1
                left_offset = np.array([0, vertical_correction])
                right_offset = np.array([0, vertical_correction])
            else:
                left_offset = np.array([0, 0])
                right_offset = np.array([0, 0])
                
        else:  # shape模式
            left_offset = np.array([1 / perspective_factor_x, 1 / perspective_factor_y])
            right_offset = np.array([1 / perspective_factor_x, 1 / perspective_factor_y])
        
        # 应用3D校正
        left_center = basic_left_center + left_offset
        right_center = basic_right_center + right_offset
        
        print(f"📍 3D校正眼球中心: 姿态={pose_type}, yaw={yaw:.1f}°, pitch={pitch:.1f}°")
        
        return left_center, right_center
    
    def calculate_3d_adaptive_movements(self, landmarks_2d, eye_centers, pose_info, adjustment_type, intensity, eye_shape_info):
        """计算3D自适应的移动向量"""
        if pose_info is None:
            pose_type, yaw, pitch, roll = "frontal", 0.0, 0.0, 0.0
        else:
            rotation_vector, translation_vector = pose_info
            pose_type, yaw, pitch, roll = self.estimate_face_pose_3d(rotation_vector)
        
        left_center, right_center = eye_centers
        movements = {}
        
        # 3D姿态自适应强度
        pose_intensity_multiplier = self.get_pose_intensity_multiplier(pose_type, yaw, pitch, eye_shape_info)
        
        # 左右眼的不对称调整（基于yaw角）
        if abs(yaw) > 10:
            if yaw > 0:  # 右转，左眼更明显
                left_eye_multiplier = 1.0 + abs(yaw) * 0.01
                right_eye_multiplier = 1.0 - abs(yaw) * 0.008
            else:  # 左转，右眼更明显
                left_eye_multiplier = 1.0 - abs(yaw) * 0.008
                right_eye_multiplier = 1.0 + abs(yaw) * 0.01
        else:
            left_eye_multiplier = right_eye_multiplier = 1.0
        
        print(f"🔧 3D自适应强度: 基础={intensity:.2f} × 姿态={pose_intensity_multiplier:.2f}")
        
        # 为左眼计算3D感知移动
        left_movements = self.calculate_single_eye_3d_movement(
            'left_eye', landmarks_2d, left_center, pose_info, adjustment_type,
            intensity * pose_intensity_multiplier * left_eye_multiplier, eye_shape_info
        )
        movements.update(left_movements)
        
        # 为右眼计算3D感知移动
        right_movements = self.calculate_single_eye_3d_movement(
            'right_eye', landmarks_2d, right_center, pose_info, adjustment_type,
            intensity * pose_intensity_multiplier * right_eye_multiplier, eye_shape_info
        )
        movements.update(right_movements)
        
        return movements
    
    def get_pose_intensity_multiplier(self, pose_type, yaw, pitch, eye_shape_info):
        """根据3D姿态和眼形获取强度倍数"""
        base_multiplier = 1.0
        
        if pose_type == "frontal":
            base_multiplier = 1.0
        elif pose_type in ["right_profile", "left_profile"]:
            base_multiplier = max(0.6, 1.0 - abs(yaw) * 0.008)
        elif pose_type in ["looking_up", "looking_down"]:
            base_multiplier = max(0.7, 1.0 - abs(pitch) * 0.006)
        else:
            base_multiplier = 0.8
        
        # 基于眼形的姿态敏感度
        if eye_shape_info and 'recommendations' in eye_shape_info:
            recommendations = eye_shape_info['recommendations']
            if 'pose_multiplier' in recommendations:
                base_multiplier *= recommendations['pose_multiplier']
        
        return base_multiplier
    
    def calculate_single_eye_3d_movement(self, eye_side, landmarks_2d, eye_center, pose_info, adjustment_type, intensity, eye_shape_info):
        """计算单眼的3D感知移动"""
        movements = {}
        regions = self.EYE_MOVEMENT_REGIONS[eye_side]
        
        # 获取3D姿态信息
        if pose_info is None:
            pose_type, yaw, pitch, roll = "frontal", 0.0, 0.0, 0.0
        else:
            rotation_vector, translation_vector = pose_info
            pose_type, yaw, pitch, roll = self.estimate_face_pose_3d(rotation_vector)
        
        # 为每个区域计算3D感知移动
        for region_name, point_indices in regions.items():
            for point_idx in point_indices:
                if point_idx < len(landmarks_2d):
                    movement = self.calculate_3d_point_movement(
                        landmarks_2d[point_idx], eye_center, pose_type, yaw, pitch,
                        adjustment_type, intensity, region_name, eye_side, eye_shape_info
                    )
                    movements[point_idx] = movement
        
        return movements
    
    def calculate_3d_point_movement(self, point, eye_center, pose_type, yaw, pitch, adjustment_type, intensity, region_name, eye_side, eye_shape_info):
        """计算单个点的3D感知移动"""
        # 基础方向计算
        direction = point - eye_center
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm <= 1e-6:
            return np.array([0.0, 0.0])
        
        direction = direction / direction_norm
        
        # 区域权重
        region_weights = {
            'outer_corner': 1.0,
            'inner_corner': 0.3,
            'upper_lid': 0.7,
            'lower_lid': 0.5,
            'transition': 0.4
        }
        
        base_weight = region_weights.get(region_name, 0.5)
        
        # 3D姿态权重调整
        if pose_type != "frontal":
            if eye_side == 'left_eye' and yaw > 20:
                base_weight *= 0.8
            elif eye_side == 'right_eye' and yaw < -20:
                base_weight *= 0.8
        
        # 根据调整类型计算移动向量
        if adjustment_type == "stretch":
            movement_vector = self.calculate_3d_stretch_vector(direction, pose_type, yaw, pitch, region_name)
            movement_scale = intensity * base_weight * 12
            
        elif adjustment_type == "lift":
            movement_vector = self.calculate_3d_lift_vector(direction, pose_type, yaw, pitch, region_name, eye_side)
            movement_scale = intensity * base_weight * 10
            
        else:  # shape模式
            stretch_vec = self.calculate_3d_stretch_vector(direction, pose_type, yaw, pitch, region_name)
            lift_vec = self.calculate_3d_lift_vector(direction, pose_type, yaw, pitch, region_name, eye_side)
            movement_vector = stretch_vec * 0.6 + lift_vec * 0.4
            movement_vector = movement_vector / np.linalg.norm(movement_vector)
            movement_scale = intensity * base_weight * 11
        
        return movement_vector * movement_scale
    
    def calculate_3d_stretch_vector(self, direction, pose_type, yaw, pitch, region_name):
        """计算3D感知的拉长向量"""
        if pose_type == "frontal":
            return direction
        else:
            # 侧脸时需要透视校正
            yaw_rad = np.radians(yaw)
            perspective_correction = np.cos(yaw_rad)
            
            if region_name == 'outer_corner':
                # 外眦主要沿水平方向拉长，需要透视校正
                corrected_direction = np.array([direction[0] / max(perspective_correction, 0.5), direction[1]])
                return corrected_direction / np.linalg.norm(corrected_direction)
            else:
                return direction * 0.8
    
    def calculate_3d_lift_vector(self, direction, pose_type, yaw, pitch, region_name, eye_side):
        """计算3D感知的上扬向量"""
        base_lift = np.array([0, -1])
        
        if pose_type == "frontal":
            if region_name == 'outer_corner':
                return direction * 0.4 + base_lift * 0.6
            else:
                return direction * 0.7 + base_lift * 0.3
        else:
            # 3D姿态时的上扬校正
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
            
            pitch_factor = 1.0 - pitch * 0.008
            
            if abs(yaw) > 20:
                rotated_lift = np.array([
                    base_lift[0] * np.cos(yaw_rad) - base_lift[1] * np.sin(yaw_rad),
                    base_lift[0] * np.sin(yaw_rad) + base_lift[1] * np.cos(yaw_rad)
                ]) * pitch_factor
            else:
                rotated_lift = base_lift * pitch_factor
            
            if region_name == 'outer_corner':
                return direction * 0.3 + rotated_lift * 0.7
            else:
                return direction * 0.8 + rotated_lift * 0.2
    
    def create_3d_aware_eye_mask(self, image_shape, landmarks_2d, pose_info):
        """创建3D感知的眼角mask"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 收集所有眼部移动点
        all_eye_points = []
        for eye_regions in self.EYE_MOVEMENT_REGIONS.values():
            for point_list in eye_regions.values():
                for idx in point_list:
                    if idx < len(landmarks_2d):
                        all_eye_points.append(landmarks_2d[idx])
        
        if len(all_eye_points) < 8:
            return mask
        
        all_eye_points = np.array(all_eye_points, dtype=np.int32)
        
        # 3D姿态自适应扩展
        if pose_info is None:
            expansion_factor = 1.5
        else:
            rotation_vector, _ = pose_info
            pose_type, yaw, pitch, roll = self.estimate_face_pose_3d(rotation_vector)
            
            if pose_type == "frontal":
                expansion_factor = 1.6
            else:
                expansion_factor = 1.3 - abs(yaw) * 0.005
                expansion_factor = max(expansion_factor, 1.1)
        
        # 为每个点创建扩展圆形
        for point in all_eye_points:
            radius = int(10 * expansion_factor)
            cv2.circle(mask, tuple(point), radius, 255, -1)
        
        # 形态学处理
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask
    
    def apply_3d_eye_corner_transform(self, image, landmarks_2d, pose_info, adjustment_type="stretch", intensity=0.3, eye_shape_info=None):
        """应用3D感知的眼角变形"""
        # 3D姿态分析
        if pose_info is None:
            print("⚠️ 3D姿态信息无效，使用2D模式")
            pose_type, yaw, pitch, roll = "frontal", 0.0, 0.0, 0.0
        else:
            rotation_vector, translation_vector = pose_info
            pose_type, yaw, pitch, roll = self.estimate_face_pose_3d(rotation_vector)
            print(f"🔍 3D姿态: {pose_type} (yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°)")
        
        # 计算3D感知的眼球中心
        eye_centers = self.calculate_3d_aware_eye_centers(landmarks_2d, pose_info, adjustment_type)
        
        # 计算3D自适应移动向量
        movement_vectors = self.calculate_3d_adaptive_movements(
            landmarks_2d, eye_centers, pose_info, adjustment_type, intensity, eye_shape_info
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
        
        print(f"🔧 3D控制点: 移动={movement_count}, 锚点={anchor_count}, 对称={symmetry_count}, 边界=8")
        
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
            
            print(f"🔄 开始3D感知TPS变换（{adjustment_type}模式，姿态：{pose_type}）...")
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
            
            # 创建3D感知mask
            eye_mask = self.create_3d_aware_eye_mask(image.shape, landmarks_2d, pose_info)
            
            # 安全融合
            mask_3d = cv2.merge([eye_mask, eye_mask, eye_mask]).astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_3d, (5, 5), 0)
            
            result = image.astype(np.float32) * (1 - mask_blurred) + transformed.astype(np.float32) * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print("✅ 3D感知眼角变形成功")
            return result
            
        except Exception as e:
            print(f"❌ TPS变形异常: {e}")
            return image
    
    def debug_3d_eye_detection(self, image, save_dir=None):
        """3D感知的眼角调试"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        # 3D眼形分析
        eye_shape_info = self.eye_analyzer.analyze_eye_shape_3d(landmarks_2d, pose_info)
        
        debug_img = image.copy()
        
        # 绘制3D姿态信息
        if pose_info is not None:
            rotation_vector, translation_vector = pose_info
            pose_type, yaw, pitch, roll = self.estimate_face_pose_3d(rotation_vector)
            
            # 显示3D姿态
            cv2.putText(debug_img, f"3D Pose: {pose_type}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(debug_img, f"Yaw: {yaw:.1f}° Pitch: {pitch:.1f}° Roll: {roll:.1f}°", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 绘制3D坐标轴
            try:
                axis_3d = np.array([
                    [0, 0, 0], [40, 0, 0], [0, 40, 0], [0, 0, -40]
                ], dtype=np.float32)
                
                projected_axis, _ = cv2.projectPoints(
                    axis_3d, rotation_vector, translation_vector,
                    self.camera_matrix, self.dist_coeffs
                )
                projected_axis = projected_axis.reshape(-1, 2).astype(int)
                
                origin = tuple(projected_axis[0])
                cv2.arrowedLine(debug_img, origin, tuple(projected_axis[1]), (0, 0, 255), 3)  # X-红
                cv2.arrowedLine(debug_img, origin, tuple(projected_axis[2]), (0, 255, 0), 3)  # Y-绿
                cv2.arrowedLine(debug_img, origin, tuple(projected_axis[3]), (255, 0, 0), 3)  # Z-蓝
            except:
                pass
        
        # 绘制眼部移动区域
        region_colors = {
            'outer_corner': (0, 255, 0),    # 绿色
            'inner_corner': (255, 0, 0),    # 红色
            'upper_lid': (0, 0, 255),       # 蓝色
            'lower_lid': (255, 255, 0),     # 黄色
            'transition': (255, 0, 255)     # 紫色
        }
        
        for eye_side, regions in self.EYE_MOVEMENT_REGIONS.items():
            for region_name, point_indices in regions.items():
                color = region_colors[region_name]
                for i, idx in enumerate(point_indices):
                    if idx < len(landmarks_2d):
                        point = landmarks_2d[idx].astype(int)
                        cv2.circle(debug_img, tuple(point), 6, color, -1)
                        cv2.putText(debug_img, f"{eye_side[0].upper()}{region_name[0].upper()}{i}", 
                                   tuple(point + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 绘制3D感知的眼球中心
        left_center, right_center = self.calculate_3d_aware_eye_centers(landmarks_2d, pose_info, "stretch")
        cv2.circle(debug_img, tuple(left_center.astype(int)), 8, (255, 255, 0), -1)
        cv2.circle(debug_img, tuple(right_center.astype(int)), 8, (255, 255, 0), -1)
        cv2.putText(debug_img, '3D-L', tuple(left_center.astype(int) + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(debug_img, '3D-R', tuple(right_center.astype(int) + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 显示3D眼形分析结果
        shape_info = eye_shape_info['eye_shape']
        measurements = eye_shape_info['measurements']
        
        info_y = 90
        cv2.putText(debug_img, f"3D Eye: {shape_info['primary_type']} ({shape_info['confidence']:.2f})", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if 'pose_corrected' in measurements:
            cv2.putText(debug_img, f"3D L/W: {measurements['length_width_ratio']:.2f}", 
                       (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 显示3D感知mask
        eye_mask = self.create_3d_aware_eye_mask(image.shape, landmarks_2d, pose_info)
        mask_overlay = debug_img.copy()
        mask_overlay[eye_mask > 0] = [255, 0, 255]
        debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "debug_3d_eye_corner.png"), debug_img)
            print(f"📁 3D调试文件保存到: {save_dir}")
        
        return debug_img
    
    def process_image_3d(self, image, adjustment_type="stretch", intensity=0.3, save_debug=False, output_path=None):
        """3D感知的图像处理主接口"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("❌ 未检测到面部关键点")
            return image
        
        print(f"✅ 检测到 {len(landmarks_2d)} 个关键点")
        
        # 3D眼形分析
        print("🔍 正在进行3D感知眼形分析...")
        eye_shape_info = self.eye_analyzer.analyze_eye_shape_3d(landmarks_2d, pose_info)
        
        shape_result = eye_shape_info['eye_shape']
        measurements = eye_shape_info['measurements']
        
        print(f"👁️ 3D眼形识别结果:")
        print(f"   类型: {shape_result['primary_type']} (置信度: {shape_result['confidence']:.2f})")
        if 'pose_corrected' in measurements:
            print(f"   3D校正长宽比: {measurements['length_width_ratio']:.2f}")
            print(f"   头部yaw角: {measurements['yaw_angle']:.1f}°")
        
        # 3D感知的推荐
        recommendations = eye_shape_info['recommendations']
        if 'pose_type' in recommendations:
            print(f"   3D姿态类型: {recommendations['pose_type']}")
            print(f"   姿态强度倍数: {recommendations['pose_multiplier']:.2f}")
        
        # 调试模式
        if save_debug and output_path:
            debug_dir = os.path.splitext(output_path)[0] + "_3d_debug"
            self.debug_3d_eye_detection(image, debug_dir)
        
        # 应用3D感知的眼角变形
        result = self.apply_3d_eye_corner_transform(
            image, landmarks_2d, pose_info, adjustment_type, intensity, eye_shape_info
        )
        
        return result

def main():
    parser = argparse.ArgumentParser(description='3D姿态感知的眼角调整工具')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--type', '-t', choices=['stretch', 'lift', 'shape'], 
                       default='stretch', help='调整类型: stretch=水平拉长, lift=眼角上扬, shape=综合调整')
    parser.add_argument('--intensity', type=float, default=0.3, 
                       help='调整强度 (0.1-0.4)')
    parser.add_argument('--debug', action='store_true', help='3D调试模式')
    parser.add_argument('--save-debug', action='store_true', help='保存3D调试文件')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    
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
    
    # 创建3D感知处理器
    processor = EyeCorner3DProcessor()
    
    # 调试模式
    if args.debug:
        debug_result = processor.debug_3d_eye_detection(image)
        cv2.imwrite(args.output, debug_result)
        print(f"🔍 3D调试图保存到: {args.output}")
        return
    
    # 处理图片
    print(f"🔄 开始3D感知眼角处理...")
    print(f"   调整类型: {args.type}")
    print(f"   调整强度: {args.intensity}")
    
    result = processor.process_image_3d(
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
        print("   2. 检查3D姿态是否影响效果 (--debug)")
        print("   3. 尝试不同的调整类型")
    
    # 生成对比图
    if args.comparison:
        comparison_path = args.output.replace('.', '_3d_comparison.')
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
        
        cv2.putText(comparison, 'Original', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(comparison, f'3D-{args.type.upper()} {args.intensity:.2f}', 
                   (image.shape[1] + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.imwrite(comparison_path, comparison)
        print(f"📊 3D对比图保存到: {comparison_path}")
    
    print("\n🌟 3D姿态感知眼角调整特色:")
    print("✅ 真实3D头部姿态估计: 基于PnP算法和标准3D面部模型")
    print("✅ 透视校正的眼形分析: 自动校正侧脸时的测量误差")
    print("✅ 姿态自适应的移动策略: 正面/侧脸/俯仰不同的变形方式")
    print("✅ 3D感知的强度调整: 根据头部角度自动调整变形强度")
    print("✅ 智能的左右眼不对称处理: 侧脸时突出可见眼部的调整")
    print("✅ 3D边界保护: 基于真实姿态的mask边界计算")
    
    print("\n📖 使用示例:")
    print("• 3D智能眼角调整: python script.py -i input.jpg -o output.png --comparison")
    print("• 3D姿态调试: python script.py -i input.jpg -o debug.png --debug")
    print("• 侧脸眼角拉长: python script.py -i profile.jpg -o output.png -t stretch --intensity 0.25")
    print("• 俯视眼角上扬: python script.py -i looking_down.jpg -o output.png -t lift --intensity 0.3")

if __name__ == "__main__":
    main()
