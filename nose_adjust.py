#!/usr/bin/env python3
"""
简化版鼻子调整 - 参考你的眼部和瘦脸代码的有效方法
去掉无用的复杂算法，只保留真正有效果的TPS
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse

class SimpleNoseAdjustment:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 简化关键点定义 - 只用真正有效的
        self.NOSE_TIP = [1, 2]
        self.LEFT_NOSTRIL = [49, 131, 134]   # 只用3个主要点
        self.RIGHT_NOSTRIL = [278, 360, 363] # 只用3个主要点
        self.NOSE_BRIDGE = [6, 19, 20]       # 只用3个主要点
        
        # 稳定锚点 - 参考你的瘦脸代码
        self.STABLE_ANCHORS = [
            33, 133, 362, 263,      # 眼角
            61, 291, 39, 269,       # 嘴角
            9, 10, 151              # 额头和下巴
        ]
    
    def detect_landmarks(self, image):
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
    
    def estimate_face_pose(self, landmarks):
        """姿态估计 - 复制瘦脸代码"""
        left_eye_x = landmarks[33][0]
        right_eye_x = landmarks[263][0] 
        nose_x = landmarks[1][0]
        
        eye_center_x = (left_eye_x + right_eye_x) / 2
        nose_offset = nose_x - eye_center_x
        eye_distance = abs(right_eye_x - left_eye_x)
        
        if eye_distance > 0:
            yaw_ratio = nose_offset / eye_distance
            if abs(yaw_ratio) < 0.1:
                return "frontal"
            elif yaw_ratio > 0.1:
                return "right_profile" 
            else:
                return "left_profile"
        return "frontal"

    def create_nose_mask(self, image_shape, landmarks):
        """创建自适应鼻子区域mask"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 姿态检测
        face_pose = self.estimate_face_pose(landmarks)
        
        # 收集所有鼻子关键点
        nose_points = []
        for idx in self.NOSE_TIP + self.LEFT_NOSTRIL + self.RIGHT_NOSTRIL + self.NOSE_BRIDGE:
            if idx < len(landmarks):
                nose_points.append(landmarks[idx])
        
        if len(nose_points) >= 3:
            nose_points = np.array(nose_points, dtype=np.int32)
            hull = cv2.convexHull(nose_points)
            
            # 计算扩展系数 - 图片尺寸自适应
            total_pixels = h * w
            if total_pixels > 1000000:  # 大图
                base_expand = 1.8
            elif total_pixels < 100000:  # 小图
                base_expand = 1.3
            else:  # 中等图
                base_expand = 1.5
            
            # 姿态自适应扩展 - 侧脸用更小mask避免扭曲
            if face_pose == "left_profile":
                expand_factor = base_expand * 0.9  # 侧脸用更小mask
            elif face_pose == "right_profile":
                expand_factor = base_expand * 0.9  # 侧脸用更小mask
            else:
                expand_factor = base_expand
            
            # 扩大mask区域
            center = np.mean(hull.reshape(-1, 2), axis=0)
            expanded_hull = []
            for point in hull.reshape(-1, 2):
                vec = point - center
                expanded_point = center + vec * expand_factor
                expanded_hull.append(expanded_point)
            
            expanded_hull = np.array(expanded_hull, dtype=np.int32)
            cv2.fillPoly(mask, [expanded_hull], 255)
            
            print(f"姿态: {face_pose}, 图片: {w}x{h}, mask扩展: {expand_factor:.1f}x")
        
        return mask

    def tps_nose_wings_shrink(self, image, landmarks, intensity=0.3):
        """鼻翼收缩 - 使用TPS变形 + 姿态自适应"""
        h, w = image.shape[:2]
        
        # 姿态检测
        face_pose = self.estimate_face_pose(landmarks)
        
        # 侧脸时降低强度避免扭曲
        if face_pose != "frontal":
            intensity = intensity * 0.6  # 侧脸时大幅降低强度
            print(f"侧脸检测到，强度降低到: {intensity:.2f}")
        
        # 姿态自适应强度
        left_multiplier = 1.0
        right_multiplier = 1.0
        
        if face_pose == "left_profile":
            left_multiplier = 1.2   # 降低从1.3到1.2
            right_multiplier = 0.8  # 调整从0.7到0.8
        elif face_pose == "right_profile":
            left_multiplier = 0.8   # 调整从0.7到0.8  
            right_multiplier = 1.2  # 降低从1.3到1.2
        
        src_points = []
        dst_points = []
        
        # 计算鼻子中心点
        nose_center_x = landmarks[1][0]  # 鼻尖x坐标作为中心
        
        # 左鼻翼收缩
        for idx in self.LEFT_NOSTRIL:
            if idx < len(landmarks):
                point = landmarks[idx]
                src_points.append(point)
                # 向中心收缩 - 应用左侧系数
                shrink_amount = (point[0] - nose_center_x) * intensity * 0.5 * left_multiplier
                new_x = point[0] - shrink_amount
                dst_points.append([new_x, point[1]])
        
        # 右鼻翼收缩
        for idx in self.RIGHT_NOSTRIL:
            if idx < len(landmarks):
                point = landmarks[idx]
                src_points.append(point)
                # 向中心收缩 - 应用右侧系数
                shrink_amount = (point[0] - nose_center_x) * intensity * 0.5 * right_multiplier
                new_x = point[0] - shrink_amount
                dst_points.append([new_x, point[1]])
        
        print(f"鼻翼收缩: {face_pose}, 左侧系数: {left_multiplier:.1f}, 右侧系数: {right_multiplier:.1f}")
        
        # 添加稳定锚点 - 现在有很多了
        anchor_points_added = 0
        for anchor_idx in self.STABLE_ANCHORS:
            if anchor_idx < len(landmarks):
                anchor_point = landmarks[anchor_idx]
                src_points.append(anchor_point)
                dst_points.append(anchor_point)
                anchor_points_added += 1
        
        print(f"添加了 {anchor_points_added} 个稳定锚点")
        
        # 添加边界锚点
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
            
            # 1. TPS变形整张图
            transformed = tps.warpImage(image)
            
            if transformed is None:
                print("TPS变换失败，返回原图")
                return image
            
            # 2. 创建自适应鼻子区域mask
            nose_mask = self.create_nose_mask(image.shape, landmarks)
            
            # 3. 只在鼻子区域融合变形效果
            mask_3d = cv2.merge([nose_mask, nose_mask, nose_mask]).astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_3d, (5, 5), 0)
            
            result = image.astype(np.float32) * (1 - mask_blurred) + transformed.astype(np.float32) * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"TPS变换异常: {e}")
            return image
    
    def tps_nose_tip_lift(self, image, landmarks, intensity=0.3):
        """鼻尖抬高 - 使用TPS变形 + mask融合"""
        h, w = image.shape[:2]
        
        src_points = []
        dst_points = []
        
        # 鼻尖向上移动
        for idx in self.NOSE_TIP:
            if idx < len(landmarks):
                point = landmarks[idx]
                src_points.append(point)
                # 向上移动
                lift_amount = intensity * 15  # 15像素的基础移动量
                new_y = point[1] - lift_amount
                dst_points.append([point[0], new_y])
        
        # 鼻梁轻微向上
        for idx in self.NOSE_BRIDGE:
            if idx < len(landmarks):
                point = landmarks[idx]
                src_points.append(point)
                # 轻微向上
                lift_amount = intensity * 5
                new_y = point[1] - lift_amount
                dst_points.append([point[0], new_y])
        
        # 添加稳定锚点
        for anchor_idx in self.STABLE_ANCHORS:
            if anchor_idx < len(landmarks):
                anchor_point = landmarks[anchor_idx]
                src_points.append(anchor_point)
                dst_points.append(anchor_point)
        
        # 添加边界锚点
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
            
            # 1. TPS变形整张图
            transformed = tps.warpImage(image)
            
            if transformed is None:
                print("TPS变换失败，返回原图")
                return image
            
            # 2. 创建鼻子区域mask
            nose_mask = self.create_nose_mask(image.shape, landmarks)
            
            # 3. 只在鼻子区域融合变形效果
            mask_3d = cv2.merge([nose_mask, nose_mask, nose_mask]).astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_3d, (5, 5), 0)
            
            result = image.astype(np.float32) * (1 - mask_blurred) + transformed.astype(np.float32) * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"TPS变换异常: {e}")
            return image
    
    def tps_nose_resize(self, image, landmarks, intensity=0.3):
        """整体鼻子缩小 - 使用TPS变形 + mask融合"""
        h, w = image.shape[:2]
        
        src_points = []
        dst_points = []
        
        # 计算鼻子中心
        all_nose_points = []
        for idx in self.NOSE_TIP + self.LEFT_NOSTRIL + self.RIGHT_NOSTRIL + self.NOSE_BRIDGE:
            if idx < len(landmarks):
                all_nose_points.append(landmarks[idx])
        
        if len(all_nose_points) < 3:
            return image
        
        nose_center = np.mean(all_nose_points, axis=0)
        
        # 所有鼻子关键点向中心收缩
        for idx in self.NOSE_TIP + self.LEFT_NOSTRIL + self.RIGHT_NOSTRIL + self.NOSE_BRIDGE:
            if idx < len(landmarks):
                point = landmarks[idx]
                src_points.append(point)
                # 向中心收缩
                vec = point - nose_center
                new_point = nose_center + vec * (1 - intensity * 0.3)
                dst_points.append(new_point)
        
        # 添加稳定锚点
        for anchor_idx in self.STABLE_ANCHORS:
            if anchor_idx < len(landmarks):
                anchor_point = landmarks[anchor_idx]
                src_points.append(anchor_point)
                dst_points.append(anchor_point)
        
        # 添加边界锚点
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
            
            # 1. TPS变形整张图
            transformed = tps.warpImage(image)
            
            if transformed is None:
                print("TPS变换失败，返回原图")
                return image
            
            # 2. 创建鼻子区域mask
            nose_mask = self.create_nose_mask(image.shape, landmarks)
            
            # 3. 只在鼻子区域融合变形效果
            mask_3d = cv2.merge([nose_mask, nose_mask, nose_mask]).astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_3d, (5, 5), 0)
            
            result = image.astype(np.float32) * (1 - mask_blurred) + transformed.astype(np.float32) * mask_blurred
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"TPS变换异常: {e}")
            return image
    
    def debug_nose_detection(self, image):
        """调试显示鼻子关键点和mask"""
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            print("未检测到面部关键点")
            return image
        
        debug_img = image.copy()
        
        # 绘制鼻尖（红色）
        for idx in self.NOSE_TIP:
            if idx < len(landmarks):
                cv2.circle(debug_img, tuple(landmarks[idx]), 6, (0, 0, 255), -1)
        
        # 绘制左鼻翼（绿色）
        for idx in self.LEFT_NOSTRIL:
            if idx < len(landmarks):
                cv2.circle(debug_img, tuple(landmarks[idx]), 5, (0, 255, 0), -1)
        
        # 绘制右鼻翼（蓝色）
        for idx in self.RIGHT_NOSTRIL:
            if idx < len(landmarks):
                cv2.circle(debug_img, tuple(landmarks[idx]), 5, (255, 0, 0), -1)
        
        # 绘制鼻梁（黄色）
        for idx in self.NOSE_BRIDGE:
            if idx < len(landmarks):
                cv2.circle(debug_img, tuple(landmarks[idx]), 4, (0, 255, 255), -1)
        
        # 显示mask区域
        nose_mask = self.create_nose_mask(image.shape, landmarks)
        mask_overlay = debug_img.copy()
        mask_overlay[nose_mask > 0] = [255, 0, 255]  # 紫色显示mask区域
        debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)
        
        # 保存单独的mask图像用于查看
        cv2.imwrite("debug_nose_mask.png", nose_mask)
        print("Mask保存到: debug_nose_mask.png")
        
        return debug_img
    
    def process_image(self, image, adjust_type="wings", intensity=0.3):
        """处理图像"""
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            print("未检测到面部关键点")
            return image
        
        print(f"检测到 {len(landmarks)} 个关键点")
        
        if adjust_type == "wings":
            return self.tps_nose_wings_shrink(image, landmarks, intensity)
        elif adjust_type == "tip":
            return self.tps_nose_tip_lift(image, landmarks, intensity)
        elif adjust_type == "resize":
            return self.tps_nose_resize(image, landmarks, intensity)
        else:
            print(f"未知调整类型: {adjust_type}")
            return image

def main():
    parser = argparse.ArgumentParser(description='简化版鼻子调整工具')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--type', '-t', choices=['wings', 'tip', 'resize'], 
                       default='wings', help='调整类型')
    parser.add_argument('--intensity', type=float, default=0.3, 
                       help='调整强度 (0.1-0.7)')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"无法加载图片: {args.input}")
        return
    
    print(f"图片尺寸: {image.shape}")
    
    # 创建处理器
    processor = SimpleNoseAdjustment()
    
    # 调试模式
    if args.debug:
        debug_result = processor.debug_nose_detection(image)
        cv2.imwrite(args.output, debug_result)
        print(f"调试图保存到: {args.output}")
        return
    
    # 处理图片
    print(f"开始处理 - 类型: {args.type}, 强度: {args.intensity}")
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
        comparison_path = args.output.replace('.', '_comparison.')
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
        
        cv2.putText(comparison, 'Original', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f'{args.type} {args.intensity}', (image.shape[1] + 20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(comparison_path, comparison)
        print(f"对比图保存到: {comparison_path}")
    
    print("\n使用示例:")
    print("• 鼻翼收缩: python nose_adjust.py -i input.jpg -o output.png -t wings --intensity 0.4 --comparison")
    print("• 鼻尖抬高: python nose_adjust.py -i input.jpg -o output.png -t tip --intensity 0.3 --comparison")
    print("• 整体缩小: python nose_adjust.py -i input.jpg -o output.png -t resize --intensity 0.5 --comparison")

if __name__ == "__main__":
    main()
