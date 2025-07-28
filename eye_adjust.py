#!/usr/bin/env python3
"""
改进版眼部调整脚本 - eye_adjust.py
重点解决："带子"问题的根本原因
技术：关键点异常检测 + 简化TPS变形 + 分步调试
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
from typing import List, Tuple, Optional

class SafeEyeAdjustment:
    def __init__(self,face_mesh=None):
        # MediaPipe设置
        self.mp_face_mesh = mp.solutions.face_mesh
        if not face_mesh:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,  # 静态图片模式更稳定
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.8,  # 提高检测阈值
                min_tracking_confidence=0.8
            )
        else:
            self.face_mesh = face_mesh
        
        # 简化的眼部关键点（增加更多控制点以获得更好效果）
        self.LEFT_EYE_ENHANCED = [33, 133, 159, 145, 158, 157, 173, 163]   # 8个点：内角、外角、上下点+中间点
        self.RIGHT_EYE_ENHANCED = [362, 263, 386, 374, 387, 388, 466, 381] # 8个点：内角、外角、上下点+中间点
        
        # 完整眼部轮廓（用于mask）
        self.LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # 稳定锚点（防止全局变形）
        self.STABLE_ANCHORS = [
            1, 2, 5, 4, 6, 19, 20,        # 鼻子
            9, 10, 151, 175,              # 额头和下巴
            61, 291, 39, 269, 270,        # 嘴巴
            234, 93, 132, 454, 323, 361   # 脸颊
        ]
        
        # 修改：不同模式使用不同的强度上限
        self.MAX_INTENSITY_TPS = 0.4      # TPS模式上限
        self.MAX_INTENSITY_SIMPLE = 0.15  # 简单模式上限
        self.MIN_EYE_WIDTH = 10    # 最小眼部宽度（像素）
        self.MIN_EYE_HEIGHT = 3    # 最小眼部高度（像素）
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """检测面部关键点"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = []
        h, w = image.shape[:2]
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
        
        return np.array(landmarks)
    
    def get_adaptive_intensity(self, image_shape: Tuple[int, int], base_intensity: float) -> float:
        """修改：简化自适应调整，避免过度衰减"""
        h, w = image_shape[:2]
        total_pixels = h * w
        
        # 修改：更温和的自适应策略
        if total_pixels > 1000000:  # 超过1M像素
            scale_factor = 0.8  # 轻微降低
        elif total_pixels < 100000:  # 小于100K像素  
            scale_factor = 1.2  # 轻微提高
        else:
            scale_factor = 1.0  # 中等尺寸保持原样
        
        adaptive_intensity = base_intensity * scale_factor
        
        print(f"🔧 自适应强度调整:")
        print(f"   图片尺寸: {w}x{h} ({total_pixels}像素)")
        print(f"   缩放因子: {scale_factor:.2f}")
        print(f"   原始强度: {base_intensity:.3f} → 调整后: {adaptive_intensity:.3f}")
        
        return adaptive_intensity
    
    def validate_eye_landmarks(self, landmarks: np.ndarray) -> bool:
        """验证眼部关键点是否正常（使用增强关键点）"""
        try:
            # 检查左眼
            left_eye_points = [landmarks[i] for i in self.LEFT_EYE_ENHANCED if i < len(landmarks)]
            if len(left_eye_points) < 6:  # 至少需要6个点
                print("警告：左眼关键点不足")
                return False
            
            left_eye_points = np.array(left_eye_points)
            left_width = np.max(left_eye_points[:, 0]) - np.min(left_eye_points[:, 0])
            left_height = np.max(left_eye_points[:, 1]) - np.min(left_eye_points[:, 1])
            
            # 检查右眼
            right_eye_points = [landmarks[i] for i in self.RIGHT_EYE_ENHANCED if i < len(landmarks)]
            if len(right_eye_points) < 6:  # 至少需要6个点
                print("警告：右眼关键点不足")
                return False
            
            right_eye_points = np.array(right_eye_points)
            right_width = np.max(right_eye_points[:, 0]) - np.min(right_eye_points[:, 0])
            right_height = np.max(right_eye_points[:, 1]) - np.min(right_eye_points[:, 1])
            
            # 检查眼部尺寸是否合理
            if left_width < self.MIN_EYE_WIDTH or left_height < self.MIN_EYE_HEIGHT:
                print(f"警告：左眼尺寸异常 - 宽度:{left_width}, 高度:{left_height}")
                return False
            
            if right_width < self.MIN_EYE_WIDTH or right_height < self.MIN_EYE_HEIGHT:
                print(f"警告：右眼尺寸异常 - 宽度:{right_width}, 高度:{right_height}")
                return False
            
            # 修正标准差检查（眼部垂直分布本来就小）
            left_std_x = np.std(left_eye_points[:, 0])
            left_std_y = np.std(left_eye_points[:, 1])
            right_std_x = np.std(right_eye_points[:, 0])
            right_std_y = np.std(right_eye_points[:, 1])
            
            # 放宽检查条件：水平分布要求 > 2，垂直分布要求 > 1（眼睛本来就扁）
            if left_std_x < 2 or right_std_x < 2:
                print(f"警告：眼部水平分布异常 - 左眼std_x:{left_std_x:.1f}, 右眼std_x:{right_std_x:.1f}")
                return False
            
            if left_std_y < 1 or right_std_y < 1:
                print(f"警告：眼部垂直分布异常 - 左眼std_y:{left_std_y:.1f}, 右眼std_y:{right_std_y:.1f}")
                return False
            
            # 检查眼部之间的距离是否合理
            left_center = np.mean(left_eye_points, axis=0)
            right_center = np.mean(right_eye_points, axis=0)
            eye_distance = np.linalg.norm(left_center - right_center)
            
            if eye_distance < 30:  # 两眼距离至少30像素
                print(f"警告：双眼距离异常 - 距离:{eye_distance:.1f}")
                return False
            
            print(f"✅ 眼部关键点验证通过")
            print(f"   左眼: 尺寸({left_width:.0f}x{left_height:.0f}), std({left_std_x:.1f},{left_std_y:.1f})")
            print(f"   右眼: 尺寸({right_width:.0f}x{right_height:.0f}), std({right_std_x:.1f},{right_std_y:.1f})")
            print(f"   双眼距离: {eye_distance:.0f}像素")
            return True
            
        except Exception as e:
            print(f"眼部关键点验证失败: {e}")
            return False
    
    def create_safe_eye_mask(self, image_shape: Tuple[int, int], landmarks: np.ndarray) -> np.ndarray:
        """创建安全的眼部区域mask"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 分别为左右眼创建mask
        for eye_contour in [self.LEFT_EYE_CONTOUR, self.RIGHT_EYE_CONTOUR]:
            eye_points = []
            for idx in eye_contour:
                if idx < len(landmarks):
                    eye_points.append(landmarks[idx])
            
            if len(eye_points) >= 3:
                eye_points = np.array(eye_points, dtype=np.int32)
                # 创建保守的凸包
                hull = cv2.convexHull(eye_points)
                cv2.fillPoly(mask, [hull], 255)
        
        return mask
    
    def simple_eye_enlarge(self, image: np.ndarray, landmarks: np.ndarray, 
                          intensity: float = 0.1) -> np.ndarray:
        """增强版眼睛放大算法（更多控制点+自适应强度）"""
        # 验证关键点
        if not self.validate_eye_landmarks(landmarks):
            print("❌ 眼部关键点异常，跳过眼睛放大")
            return image
        
        # 自适应强度调整
        adaptive_intensity = self.get_adaptive_intensity(image.shape, intensity)
        adaptive_intensity = min(adaptive_intensity, self.MAX_INTENSITY_TPS)  # 使用TPS专用上限
        
        print(f"应用眼睛放大，自适应强度: {adaptive_intensity:.3f}")
        
        src_points = []
        dst_points = []
        
        # 处理左眼（使用增强关键点）
        left_eye_points = [landmarks[i] for i in self.LEFT_EYE_ENHANCED if i < len(landmarks)]
        if len(left_eye_points) >= 6:
            left_eye_points = np.array(left_eye_points)
            left_center = np.mean(left_eye_points, axis=0)
            
            print(f"左眼中心: ({left_center[0]:.1f}, {left_center[1]:.1f})")
            
            for i, point in enumerate(left_eye_points):
                src_points.append(point)
                # 从中心向外扩展
                vec = point - left_center
                new_point = left_center + vec * (1 + adaptive_intensity)
                dst_points.append(new_point)
                
                distance = np.linalg.norm(new_point - point)
                print(f"   左眼点{i}: 移动 {distance:.2f}像素")
        
        # 处理右眼（使用增强关键点）
        right_eye_points = [landmarks[i] for i in self.RIGHT_EYE_ENHANCED if i < len(landmarks)]
        if len(right_eye_points) >= 6:
            right_eye_points = np.array(right_eye_points)
            right_center = np.mean(right_eye_points, axis=0)
            
            print(f"右眼中心: ({right_center[0]:.1f}, {right_center[1]:.1f})")
            
            for i, point in enumerate(right_eye_points):
                src_points.append(point)
                # 从中心向外扩展
                vec = point - right_center
                new_point = right_center + vec * (1 + adaptive_intensity)
                dst_points.append(new_point)
                
                distance = np.linalg.norm(new_point - point)
                print(f"   右眼点{i}: 移动 {distance:.2f}像素")
        
        # 添加稳定锚点
        anchor_count = 0
        for anchor_idx in self.STABLE_ANCHORS:
            if anchor_idx < len(landmarks):
                src_points.append(landmarks[anchor_idx])
                dst_points.append(landmarks[anchor_idx])
                anchor_count += 1
        
        print(f"添加了 {anchor_count} 个稳定锚点")
        
        if len(src_points) < 6:
            print("❌ 控制点不足，跳过变形")
            return image
        
        # 创建眼部mask
        mask = self.create_safe_eye_mask(image.shape, landmarks)
        mask_area = np.sum(mask > 0)
        print(f"Mask区域像素数: {mask_area}")
        
        # 应用TPS变形
        return self.apply_safe_tps(image, src_points, dst_points, mask)
    
    def apply_safe_tps(self, image: np.ndarray, src_points: List, dst_points: List, 
                      mask: np.ndarray) -> np.ndarray:
        """安全的TPS变形（增强调试）"""
        try:
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            
            if len(src_points) != len(dst_points):
                print("❌ 源点和目标点数量不匹配")
                return image
            
            # 检查点的移动距离
            distances = np.linalg.norm(dst_points - src_points, axis=1)
            max_distance = np.max(distances)
            avg_distance = np.mean(distances)
            moving_points = np.sum(distances > 0.1)  # 移动超过0.1像素的点
            
            print(f"📊 TPS变形统计:")
            print(f"   控制点数: {len(src_points)}")
            print(f"   移动点数: {moving_points}")
            print(f"   最大移动: {max_distance:.2f}像素")
            print(f"   平均移动: {avg_distance:.2f}像素")
            
            # 修改：放宽移动距离限制
            if max_distance > 150:  # 从100提高到150
                print(f"❌ 点移动距离过大: {max_distance:.1f}, 跳过变形")
                return image
            
            if moving_points < 4:  # 至少要有4个点在移动
                print(f"❌ 移动点数太少: {moving_points}, 跳过变形")
                return image
            
            # 创建TPS变换器
            tps = cv2.createThinPlateSplineShapeTransformer()
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
            
            print("🔄 开始TPS变换...")
            
            # 估计变换
            ret = tps.estimateTransformation(
                dst_points.reshape(1, -1, 2),
                src_points.reshape(1, -1, 2),
                matches
            )
            
            print(f"   变换估计返回值: {ret}")
            
            # 应用变换
            print("🔄 应用TPS变换...")
            transformed = tps.warpImage(image)
            
            if transformed is None or transformed.size == 0:
                print("❌ TPS变换返回空结果")
                return image
            
            print(f"   变换后图像尺寸: {transformed.shape}")
            
            # 检查变换后的图像是否有明显变化
            diff = cv2.absdiff(image, transformed)
            diff_sum = np.sum(diff)
            print(f"   图像差异总和: {diff_sum}")
            
            if diff_sum < 1000:  # 如果变化太小
                print("⚠️  变换后图像变化很小，可能TPS变换无效")
                # 但仍然继续处理
            
            # 安全融合
            result = image.copy()
            
            # 创建3通道mask
            mask_3d = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
            
            # 修改：减少mask羽化
            mask_blurred = cv2.GaussianBlur(mask_3d, (5, 5), 0)  # 从(11,11)改为(5,5)
            
            print("🔄 融合图像...")
            
            # 融合
            result = result.astype(np.float32)
            transformed = transformed.astype(np.float32)
            result = result * (1 - mask_blurred) + transformed * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # 检查最终结果的变化
            final_diff = cv2.absdiff(image, result)
            final_diff_sum = np.sum(final_diff)
            print(f"   最终差异总和: {final_diff_sum}")
            
            print("✅ TPS变形完成")
            return result
            
        except Exception as e:
            print(f"❌ TPS变形异常: {e}")
            import traceback
            traceback.print_exc()
            return image
    
    def test_simple_transform(self, image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """测试简单的局部缩放变换（修复版本）"""
        landmarks = self.detect_landmarks(image)
        if landmarks is None or not self.validate_eye_landmarks(landmarks):
            print("❌ 关键点检测失败")
            return image
        
        print(f"🔄 开始简单变换，强度: {intensity}")
        result = image.copy()
        
        # 处理左右眼
        for eye_name, eye_indices in [("左眼", self.LEFT_EYE_ENHANCED), ("右眼", self.RIGHT_EYE_ENHANCED)]:
            eye_points = [landmarks[i] for i in eye_indices if i < len(landmarks)]
            if len(eye_points) < 6:
                print(f"❌ {eye_name}关键点不足: {len(eye_points)}")
                continue
            
            eye_points = np.array(eye_points)
            
            # 计算眼部边界框
            min_x = int(np.min(eye_points[:, 0]) - 15)
            max_x = int(np.max(eye_points[:, 0]) + 15)
            min_y = int(np.min(eye_points[:, 1]) - 10)
            max_y = int(np.max(eye_points[:, 1]) + 10)
            
            # 确保边界框在图像范围内
            h, w = image.shape[:2]
            min_x = max(0, min_x)
            max_x = min(w, max_x)
            min_y = max(0, min_y)
            max_y = min(h, max_y)
            
            # 检查边界框是否有效
            roi_width = max_x - min_x
            roi_height = max_y - min_y
            
            if roi_width <= 0 or roi_height <= 0:
                print(f"❌ {eye_name}边界框无效: {roi_width}x{roi_height}")
                continue
            
            print(f"🔄 处理{eye_name}: 区域({min_x},{min_y})-({max_x},{max_y}), 尺寸{roi_width}x{roi_height}")
            
            try:
                # 提取眼部区域
                eye_roi = image[min_y:max_y, min_x:max_x].copy()
                
                if eye_roi.size == 0:
                    print(f"❌ {eye_name}ROI为空")
                    continue
                
                # 计算缩放后的尺寸
                scale = 1 + intensity
                new_width = int(roi_width * scale)
                new_height = int(roi_height * scale)
                
                print(f"   缩放从 {roi_width}x{roi_height} 到 {new_width}x{new_height}")
                
                # 缩放眼部区域
                enlarged_roi = cv2.resize(eye_roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                # 计算放置位置（居中）
                offset_x = (new_width - roi_width) // 2
                offset_y = (new_height - roi_height) // 2
                
                start_x = min_x - offset_x
                start_y = min_y - offset_y
                end_x = start_x + new_width
                end_y = start_y + new_height
                
                # 处理边界超出
                crop_left = max(0, -start_x)
                crop_top = max(0, -start_y)
                crop_right = max(0, end_x - w)
                crop_bottom = max(0, end_y - h)
                
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(w, end_x)
                end_y = min(h, end_y)
                
                # 裁剪enlarged_roi
                if crop_left > 0 or crop_top > 0 or crop_right > 0 or crop_bottom > 0:
                    enlarged_roi = enlarged_roi[crop_top:new_height-crop_bottom, 
                                              crop_left:new_width-crop_right]
                
                final_width = end_x - start_x
                final_height = end_y - start_y
                
                if final_width <= 0 or final_height <= 0 or enlarged_roi.size == 0:
                    print(f"❌ {eye_name}最终区域无效")
                    continue
                
                # 确保尺寸匹配
                if enlarged_roi.shape[:2] != (final_height, final_width):
                    enlarged_roi = cv2.resize(enlarged_roi, (final_width, final_height))
                
                print(f"   最终放置: ({start_x},{start_y})-({end_x},{end_y})")
                
                # 修改：减少羽化程度
                mask = np.ones((final_height, final_width), dtype=np.float32)
                
                # 边缘羽化
                feather_size = min(3, final_width//6, final_height//6)  # 从5改为3，从//4改为//6
                if feather_size > 0:
                    for i in range(feather_size):
                        alpha = (i + 1) / feather_size
                        if i < final_height:
                            mask[i, :] *= alpha
                            mask[-(i+1), :] *= alpha
                        if i < final_width:
                            mask[:, i] *= alpha
                            mask[:, -(i+1)] *= alpha
                
                # 应用到结果图像
                for c in range(3):
                    result[start_y:end_y, start_x:end_x, c] = (
                        result[start_y:end_y, start_x:end_x, c] * (1 - mask) +
                        enlarged_roi[:, :, c] * mask
                    )
                
                print(f"✅ {eye_name}处理完成")
                
            except Exception as e:
                print(f"❌ {eye_name}处理出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("✅ 简单变换完成")
        return result
    
    def debug_eye_detection(self, image: np.ndarray) -> np.ndarray:
        """调试眼部检测（修复显示问题）"""
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            print("❌ 未检测到面部关键点")
            return image
        
        debug_img = image.copy()
        
        # 绘制增强版眼部关键点（左眼绿色）
        for i, idx in enumerate(self.LEFT_EYE_ENHANCED):
            if idx < len(landmarks):
                point = landmarks[idx]
                cv2.circle(debug_img, tuple(point), 6, (0, 255, 0), -1)
                cv2.circle(debug_img, tuple(point), 8, (0, 255, 0), 2)
                # 使用简单的ASCII字符避免编码问题
                label = f"{i}"  # 简化标签
                cv2.putText(debug_img, label, tuple(point + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制增强版眼部关键点（右眼红色）
        for i, idx in enumerate(self.RIGHT_EYE_ENHANCED):
            if idx < len(landmarks):
                point = landmarks[idx]
                cv2.circle(debug_img, tuple(point), 6, (0, 0, 255), -1)
                cv2.circle(debug_img, tuple(point), 8, (0, 0, 255), 2)
                # 使用简单的ASCII字符
                label = f"{i}"  # 简化标签
                cv2.putText(debug_img, label, tuple(point + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 绘制眼部轮廓（左眼青色小点）
        for idx in self.LEFT_EYE_CONTOUR:
            if idx < len(landmarks):
                cv2.circle(debug_img, tuple(landmarks[idx]), 3, (255, 255, 0), -1)
        
        # 绘制眼部轮廓（右眼紫色小点）
        for idx in self.RIGHT_EYE_CONTOUR:
            if idx < len(landmarks):
                cv2.circle(debug_img, tuple(landmarks[idx]), 3, (255, 0, 255), -1)
        
        # 显示验证结果（避免使用特殊字符）
        is_valid = self.validate_eye_landmarks(landmarks)
        status_text = "VALID" if is_valid else "INVALID"
        cv2.putText(debug_img, f"Eye landmarks: {status_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_valid else (0, 0, 255), 2)
        
        # 添加图例
        legend_y = 60
        cv2.putText(debug_img, "GREEN: Left Eye (0-7)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_img, "RED: Right Eye (0-7)", (10, legend_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(debug_img, "CYAN: Left contour", (10, legend_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(debug_img, "PURPLE: Right contour", (10, legend_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # 显示关键点索引信息
        cv2.putText(debug_img, f"Enhanced control points: 8 per eye", (10, legend_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return debug_img
    
    def process_image(self, image: np.ndarray, enlarge_intensity: float = 0.1) -> np.ndarray:
        """处理图片"""
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            print("❌ 未检测到面部关键点")
            return image
        
        print(f"✅ 检测到 {len(landmarks)} 个面部关键点")
        
        # 应用眼睛放大
        if enlarge_intensity > 0:
            result = self.simple_eye_enlarge(image, landmarks, enlarge_intensity)
        else:
            result = image
        
        return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='安全眼部调整工具 - 解决带子问题')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--enlarge', type=float, default=0.2, help='眼睛放大强度 (简单模式:0.01-0.15, TPS模式:0.01-0.4)')  # 修改：说明不同模式的范围
    parser.add_argument('--debug', action='store_true', help='调试模式：显示关键点')
    parser.add_argument('--simple', action='store_true', help='使用简单变换（绕过TPS）')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"❌ 无法加载图片: {args.input}")
        return
    
    print(f"原图尺寸: {image.shape}")
    
    # 创建处理器
    processor = SafeEyeAdjustment()
    
    # 调试模式
    if args.debug:
        debug_result = processor.debug_eye_detection(image)
        cv2.imwrite(args.output, debug_result)
        print(f"调试图保存到: {args.output}")
        return
    
    # 修改：根据模式设置不同的强度限制
    if args.simple:
        # 简单模式：限制到0.15避免像素扭曲
        max_allowed = 0.15
        enlarge_intensity = min(args.enlarge, max_allowed)
        if enlarge_intensity != args.enlarge:
            print(f"简单模式强度限制到: {enlarge_intensity} (建议≤0.15避免像素扭曲)")
    else:
        # TPS模式：允许更高强度0.4
        max_allowed = 0.4
        enlarge_intensity = min(args.enlarge, max_allowed)
        if enlarge_intensity != args.enlarge:
            print(f"TPS模式强度限制到: {enlarge_intensity} (建议≤0.4保持稳定性)")
    
    # 处理图片
    print("开始安全眼部调整...")
    if args.simple:
        result = processor.test_simple_transform(image, enlarge_intensity)
    else:
        result = processor.process_image(image, enlarge_intensity)
    
    # 修改：根据用户指定格式保存，但给出PNG建议
    output_path = args.output
    is_jpeg = output_path.lower().endswith(('.jpg', '.jpeg'))
    
    if is_jpeg:
        print("⚠️  注意：JPEG格式会压缩细微变化，建议使用PNG格式获得最佳效果")
        png_suggestion = output_path.rsplit('.', 1)[0] + '.png'
        print(f"   建议命令：python eye_adjust.py -i {args.input} -o {png_suggestion} --enlarge {args.enlarge} {'--comparison' if args.comparison else ''}")
    
    cv2.imwrite(output_path, result)
    print(f"✅ 处理完成！保存到: {output_path}")
    
    # 检查像素差异
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
        print("   1. 提高 --enlarge 参数 (如0.3)")
        print("   2. 尝试 --simple 模式")
        print("   3. 使用PNG格式避免压缩")
    
    # 生成对比图
    if args.comparison:
        # 对比图使用相同格式
        if is_jpeg:
            comparison_path = output_path.replace('.jpg', '_comparison.jpg').replace('.jpeg', '_comparison.jpeg')
            diff_path = output_path.rsplit('.', 1)[0] + '_diff.png'  # 差异图建议用PNG
        else:
            comparison_path = output_path.replace('.png', '_comparison.png')
            diff_path = output_path.replace('.png', '_diff.png')
        
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
        
        # 添加标签
        cv2.putText(comparison, 'Original', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(comparison, f'Enlarge {enlarge_intensity:.2f}', (image.shape[1] + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imwrite(comparison_path, comparison)
        print(f"对比图保存到: {comparison_path}")
        
        # 差异图用PNG保存（无损，便于观察细微变化）
        if changed_pixels > 0:
            diff_colored = cv2.applyColorMap((diff * 10).astype(np.uint8), cv2.COLORMAP_HOT)
            cv2.imwrite(diff_path, diff_colored)
            print(f"差异图保存到: {diff_path} (PNG格式便于观察)")
        else:
            print("无像素变化，跳过差异图生成")
    
    print("\n🛡️  修复内容:")
    print("✅ 双重强度限制: 简单模式≤0.15, TPS模式≤0.4")
    print("✅ 默认强度: 0.1 → 0.2") 
    print("✅ 自适应强度: 简化算法，避免过度衰减")
    print("✅ 移动距离限制: 100 → 150像素")
    print("✅ mask羽化: (11,11) → (5,5)")
    print("✅ 边缘羽化: 更保守的羽化策略")
    print("✅ 格式处理: 支持用户指定格式，JPEG时给出PNG建议")
    print("✅ 像素差异统计: 量化变化程度")
    print("\n📖 使用示例:")
    print("• TPS模式(高强度): python eye_adjust.py -i input.jpg -o result.png --enlarge 0.3 --comparison")
    print("• 简单模式(稳定):  python eye_adjust.py -i input.jpg -o result.png --enlarge 0.15 --simple --comparison") 
    print("• 对比测试:       python eye_adjust.py -i input.jpg -o result.png --enlarge 0.25 --comparison")
    print("• 调试关键点:     python eye_adjust.py -i input.jpg -o debug.png --debug")
    print("\n⚠️  重要:")
    print("• 简单模式: 建议强度0.05-0.15 (避免像素扭曲)")
    print("• TPS模式: 可用强度0.1-0.4 (平滑插值，承受更高强度)")
    print("• 差异图始终用PNG保存便于观察细微变化")

if __name__ == "__main__":
    main()
