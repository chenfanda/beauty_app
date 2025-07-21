#!/usr/bin/env python3
"""
优化版嘴巴调整脚本 - 完全复制鼻子脚本的成功模式
不再自己发明，直接复用成功的代码结构
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
from typing import List, Tuple, Optional

class Mouth3DAdjustment:
    def __init__(self):
        # MediaPipe设置（完全复制鼻子脚本）
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
        
        # 🎯 使用正确的MediaPipe嘴部点定义
        # 外嘴唇轮廓：用于mask绘制和主要变形
        self.MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
        # 内嘴唇轮廓：可用于闭合/开嘴判断和辅助变形
        self.MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
        # 嘴角点（最常用）
        self.MOUTH_CORNERS = [61, 291]
        # 选择关键的上下唇点
        self.MOUTH_KEY_POINTS = [84, 17, 314, 405]  # 上下唇中心区域
        
        # 标准3D面部模型关键点（完全复制鼻子脚本）
        self.face_model_3d = np.array([
            [0.0, 0.0, 0.0],           # 鼻尖 (1)
            [0.0, -330.0, -65.0],      # 下巴 (152)  
            [-225.0, 170.0, -135.0],   # 左眼外角 (33)
            [225.0, 170.0, -135.0],    # 右眼外角 (263)
            [-150.0, -150.0, -125.0],  # 左嘴角 (61)
            [150.0, -150.0, -125.0],   # 右嘴角 (291)
        ], dtype=np.float32)
        
        # 对应的MediaPipe索引（完全复制鼻子脚本）
        self.pnp_indices = [1, 152, 33, 263, 61, 291]
    
    def setup_camera(self, image_shape):
        """设置相机参数（完全复制鼻子脚本）"""
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
        """检测面部关键点并计算3D姿态（完全复制鼻子脚本）"""
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
        """基于3D旋转向量估计面部姿态（完全复制鼻子脚本）"""
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
    
    def create_3d_mouth_mask(self, image_shape, landmarks_2d, pose_3d):
        """基于3D姿态创建更精确的嘴巴mask（完全复制鼻子脚本的逻辑）"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 收集嘴巴关键点（使用正确的轮廓点）
        mouth_indices = self.MOUTH_OUTER  # 使用完整的外嘴唇轮廓
        mouth_points = []
        
        for idx in mouth_indices:
            if idx < len(landmarks_2d):
                mouth_points.append(landmarks_2d[idx])
        
        if len(mouth_points) >= 3:
            mouth_points = np.array(mouth_points, dtype=np.int32)
            hull = cv2.convexHull(mouth_points)
            
            # 计算嘴巴的实际尺寸
            mouth_center = np.mean(hull.reshape(-1, 2), axis=0)
            mouth_width = np.max(hull[:, :, 0]) - np.min(hull[:, :, 0])
            mouth_height = np.max(hull[:, :, 1]) - np.min(hull[:, :, 1])
            
            print(f"嘴巴尺寸: {mouth_width:.0f}x{mouth_height:.0f}像素")
            
            # 3D姿态自适应扩展（完全复制鼻子脚本）
            if pose_3d == "frontal":
                horizontal_expand = 1.6
                vertical_expand = 1.6
            else:  # 侧脸
                horizontal_expand = 1.3
                vertical_expand = 1.3
            
            # 根据图片尺寸调整（完全复制鼻子脚本）
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
                dx = point[0] - mouth_center[0]
                dy = point[1] - mouth_center[1]
                
                new_x = mouth_center[0] + dx * horizontal_expand
                new_y = mouth_center[1] + dy * vertical_expand
                
                expanded_hull.append([new_x, new_y])
            
            expanded_hull = np.array(expanded_hull, dtype=np.int32)
            
            # 进一步限制垂直扩展（完全复制鼻子脚本）
            min_y = np.min(expanded_hull[:, 1])
            max_y = np.max(expanded_hull[:, 1])
            
            max_vertical_expansion = mouth_height * 1.2
            
            if (max_y - min_y) > mouth_height + max_vertical_expansion:
                center_y = (min_y + max_y) / 2
                target_height = mouth_height + max_vertical_expansion
                
                for i, point in enumerate(expanded_hull):
                    if point[1] < center_y:
                        expanded_hull[i][1] = max(point[1], center_y - target_height/2)
                    else:
                        expanded_hull[i][1] = min(point[1], center_y + target_height/2)
            
            cv2.fillPoly(mask, [expanded_hull], 255)
            
            # 验证mask不会影响到眼睛和鼻子（完全复制鼻子脚本的边界保护逻辑）
            eye_y = min(landmarks_2d[33][1], landmarks_2d[263][1])
            nose_y = landmarks_2d[1][1] if 1 < len(landmarks_2d) else h//3
            chin_y = landmarks_2d[152][1] if 152 < len(landmarks_2d) else h*2//3
            
            mask_top = np.min(expanded_hull[:, 1])
            mask_bottom = np.max(expanded_hull[:, 1])
            
            # 边界保护（完全复制鼻子脚本）
            if mask_top < eye_y + 10:
                mask[:int(eye_y + 10), :] = 0
                print("⚠️  Mask上边界裁剪，避免影响眼部")
            
            if mask_bottom > chin_y - 5:
                mask[int(chin_y - 5):, :] = 0
                print("⚠️  Mask下边界裁剪，避免影响下巴")
            
            print(f"3D Mask: 姿态={pose_3d}, 水平扩展={horizontal_expand:.1f}x, 垂直扩展={vertical_expand:.1f}x")
        
        return mask
    
    def apply_3d_mouth_transform(self, image, landmarks_2d, pose_info, transform_type="thickness", intensity=0.3):
        """应用3D感知的嘴巴变形（完全复制鼻子脚本的结构）"""
        if pose_info is None:
            print("3D姿态信息无效，使用2D模式")
            return image
        
        rotation_vector, translation_vector = pose_info
        pose_3d = self.estimate_face_pose_3d(rotation_vector)
        
        # 3D姿态自适应强度调整（降低衰减）
        if pose_3d != "frontal":
            intensity = intensity * 0.9  # 降低衰减：从0.7改为0.9
            print(f"3D侧脸检测，强度调整到: {intensity:.2f}")
        else:
            print(f"3D正脸检测，强度保持: {intensity:.2f}")
        
        # 构建变形控制点
        src_points = []
        dst_points = []
        
        if transform_type == "thickness":
            # 嘴唇增厚：使用外嘴唇轮廓的关键点
            print(f"嘴唇增厚 - 正确外轮廓点: {pose_3d}")
            
            # 计算嘴巴中心（用嘴角）
            mouth_center = np.array([(landmarks_2d[61][0] + landmarks_2d[291][0]) / 2,
                                   (landmarks_2d[61][1] + landmarks_2d[291][1]) / 2])
            
            # 使用嘴角 + 关键的上下唇点
            for idx in self.MOUTH_CORNERS + self.MOUTH_KEY_POINTS:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    # 向外扩展
                    vec = point - mouth_center
                    expansion_factor = 1 + intensity * 0.4
                    new_point = mouth_center + vec * expansion_factor
                    dst_points.append(new_point)
        
        elif transform_type == "enlarge":
            # 嘴巴放大：使用更多外轮廓点
            print(f"嘴巴放大 - 外轮廓8点: {pose_3d}")
            
            # 选择外轮廓的8个关键点
            key_outer_points = [61, 146, 91, 84, 17, 314, 375, 291]
            
            # 计算中心
            all_points = []
            for idx in key_outer_points:
                if idx < len(landmarks_2d):
                    all_points.append(landmarks_2d[idx])
            mouth_center = np.mean(all_points, axis=0)
            
            # 所有点向外扩展
            for idx in key_outer_points:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    vec = point - mouth_center
                    scale_factor = 1 + intensity * 0.5
                    new_point = mouth_center + vec * scale_factor
                    dst_points.append(new_point)
        
        elif transform_type == "corners":
            # 嘴角抬升：只移动嘴角
            print(f"嘴角抬升 - 纯嘴角: {pose_3d}")
            
            # 移动嘴角
            for idx in self.MOUTH_CORNERS:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    lift_amount = intensity * 20
                    new_point = [point[0], point[1] - lift_amount]
                    dst_points.append(new_point)
            
            # 关键点保持稳定
            for idx in self.MOUTH_KEY_POINTS:
                if idx < len(landmarks_2d):
                    src_points.append(landmarks_2d[idx])
                    dst_points.append(landmarks_2d[idx])
        
        elif transform_type == "resize":
            # 嘴巴缩小：使用外轮廓关键点
            print(f"嘴巴缩小 - 外轮廓收缩: {pose_3d}")
            
            # 使用6个关键的外轮廓点
            key_points = self.MOUTH_CORNERS + self.MOUTH_KEY_POINTS
            
            # 计算中心
            all_points = []
            for idx in key_points:
                if idx < len(landmarks_2d):
                    all_points.append(landmarks_2d[idx])
            mouth_center = np.mean(all_points, axis=0)
            
            # 向中心收缩
            for idx in key_points:
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx]
                    src_points.append(point)
                    
                    vec = point - mouth_center
                    scale_factor = 1 - intensity * 0.3
                    new_point = mouth_center + vec * scale_factor
                    dst_points.append(new_point)
        
        elif transform_type == "peak":
            # 唇峰抬高：移动上唇中心点17
            print(f"唇峰抬高 - 上唇中心17: {pose_3d}")
            
            # 移动上唇中心点17（在外轮廓中）
            if 17 < len(landmarks_2d):
                point = landmarks_2d[17]
                src_points.append(point)
                lift_amount = intensity * 25
                new_point = [point[0], point[1] - lift_amount]
                dst_points.append(new_point)
            
            # 其他关键点保持稳定
            for idx in [61, 291, 84, 314, 405]:
                if idx < len(landmarks_2d):
                    src_points.append(landmarks_2d[idx])
                    dst_points.append(landmarks_2d[idx])
        
        else:
            print(f"❌ 未知的变形类型: {transform_type}")
            return image
        
        # 添加大量稳定锚点（完全复制鼻子脚本）
        stable_indices = [
            # 眼部
            33, 133, 362, 263, 7, 163, 144, 145, 153, 154, 155, 173, 157, 158, 159, 160, 161, 246,
            382, 381, 380, 374, 373, 390, 249, 466, 388, 387, 386, 385, 384, 398,
            # 鼻子区域
            1, 2, 6, 19, 20, 8, 9, 10, 49, 131, 134, 278, 360, 363,
            # 脸部轮廓
            234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454
        ]
        
        anchor_count = 0
        for idx in stable_indices:
            if idx < len(landmarks_2d):
                src_points.append(landmarks_2d[idx])
                dst_points.append(landmarks_2d[idx])
                anchor_count += 1
        
        # 添加边界锚点（完全复制鼻子脚本）
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
        
        print(f"3D控制点: 嘴巴={len(src_points)-anchor_count-8}, 锚点={anchor_count}, 边界=8")
        
        if len(src_points) < 10:
            print("3D控制点不足")
            return image
        
        # 执行TPS变换（完全复制鼻子脚本）
        try:
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            
            tps = cv2.createThinPlateSplineShapeTransformer()
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
            
            # 完全复制鼻子脚本的参数顺序
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
            mouth_mask = self.create_3d_mouth_mask(image.shape, landmarks_2d, pose_3d)
            
            # 融合（完全复制鼻子脚本）
            mask_3d = cv2.merge([mouth_mask, mouth_mask, mouth_mask]).astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_3d, (5, 5), 0)
            
            result = image.astype(np.float32) * (1 - mask_blurred) + transformed.astype(np.float32) * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print("✅ 3D变形成功")
            return result
            
        except Exception as e:
            print(f"3D变形异常: {e}")
            return image
    
    def debug_mouth_detection(self, image, save_dir=None):
        """调试嘴巴检测（复制鼻子脚本的调试逻辑）"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("未检测到面部关键点")
            return image
        
        debug_img = image.copy()
        
        # 绘制实际的嘴巴关键点
        mouth_parts = [
            (self.MOUTH_CORNERS, (0, 0, 255), "C"),     # 嘴角-红色
            (self.MOUTH_CENTER, (0, 255, 0), "M"),      # 中心-绿色  
            (self.MOUTH_OUTER, (255, 0, 0), "O"),       # 外围-蓝色
            (self.MOUTH_AUX, (0, 255, 255), "A")        # 辅助-黄色
        ]
        
        for indices, color, prefix in mouth_parts:
            for i, idx in enumerate(indices):
                if idx < len(landmarks_2d):
                    point = landmarks_2d[idx].astype(int)
                    cv2.circle(debug_img, tuple(point), 5, color, -1)
                    cv2.putText(debug_img, f"{prefix}{idx}", tuple(point + 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 显示3D姿态信息
        if pose_info is not None:
            rotation_vector, translation_vector = pose_info
            pose_3d = self.estimate_face_pose_3d(rotation_vector)
            cv2.putText(debug_img, f"3D Pose: {pose_3d}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示3D mask
        if pose_info is not None:
            pose_3d = self.estimate_face_pose_3d(pose_info[0])
            mouth_mask = self.create_3d_mouth_mask(image.shape, landmarks_2d, pose_3d)
            mask_overlay = debug_img.copy()
            mask_overlay[mouth_mask > 0] = [255, 0, 255]  # 紫色mask
            debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "debug_mouth_3d.png"), debug_img)
            print(f"调试文件保存到: {save_dir}")
        
        return debug_img
    
    def process_image(self, image, transform_type="thickness", intensity=0.3, save_debug=False, output_path=None):
        """处理图像（完全复制鼻子脚本的结构）"""
        landmarks_2d, pose_info = self.detect_landmarks_3d(image)
        if landmarks_2d is None:
            print("未检测到面部关键点")
            return image
        
        print(f"✅ 检测到 {len(landmarks_2d)} 个2D关键点")
        
        if save_debug and output_path:
            debug_dir = os.path.splitext(output_path)[0] + "_debug"
            self.debug_mouth_detection(image, debug_dir)
        
        return self.apply_3d_mouth_transform(image, landmarks_2d, pose_info, transform_type, intensity)

def main():
    parser = argparse.ArgumentParser(description='3D Face Geometry嘴巴调整工具 - 完全复制鼻子脚本成功模式')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--type', '-t', choices=['thickness', 'peak', 'corners', 'resize', 'enlarge'], 
                       default='thickness', help='调整类型')
    parser.add_argument('--intensity', type=float, default=0.3, help='调整强度')
    parser.add_argument('--debug', action='store_true', help='3D调试模式')
    parser.add_argument('--save-debug', action='store_true', help='保存调试文件')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    
    args = parser.parse_args()
    
    image = cv2.imread(args.input)
    if image is None:
        print(f"无法加载图片: {args.input}")
        return
    
    print(f"图片尺寸: {image.shape}")
    
    processor = Mouth3DAdjustment()
    
    if args.debug:
        debug_result = processor.debug_mouth_detection(image)
        cv2.imwrite(args.output, debug_result)
        print(f"3D调试图保存到: {args.output}")
        return
    
    print(f"开始3D处理 - 类型: {args.type}, 强度: {args.intensity}")
    result = processor.process_image(image, args.type, args.intensity, args.save_debug, args.output)
    
    cv2.imwrite(args.output, result)
    print(f"处理完成，保存到: {args.output}")
    
    if args.comparison:
        comparison_path = args.output.replace('.', '_comparison.')
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
        
        cv2.putText(comparison, 'Original', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f'3D-{args.type}', (image.shape[1] + 20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(comparison_path, comparison)
        print(f"3D对比图保存到: {comparison_path}")
    
    print("\n✅ 完全复制鼻子脚本成功模式:")
    print("• 相同的3D几何模型和PnP算法")
    print("• 相同的TPS参数顺序 dst->src")
    print("• 相同的控制点和锚点策略")
    print("• 相同的mask创建和融合逻辑")

if __name__ == "__main__":
    main()
