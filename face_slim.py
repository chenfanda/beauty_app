#!/usr/bin/env python3
"""
简化版专业瘦脸算法 - 只保留有效的TPS方法
去除无效的位移场方法，专注于真正有效果的算法
"""

import cv2
import numpy as np
import mediapipe as mp

class SimpleFaceSlimming:
    def __init__(self,face_mesh=None):
        self.mp_face_mesh = mp.solutions.face_mesh
        if not face_mesh:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = face_mesh
        
        # Jawline关键点（真正有效的控制点）
        self.JAWLINE_LEFT = [234, 93, 132, 58, 172, 136, 150, 149]
        self.JAWLINE_RIGHT = [454, 323, 361, 288, 397, 365, 379, 378]
        
        # 稳定锚点（保持面部特征不变）
        self.STABLE_ANCHORS = [
            1, 2, 5, 4, 6, 19, 20,  # 鼻子
            33, 133, 362, 263,      # 眼角
            9, 10, 151,             # 额头和下巴中心
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
        """简单姿态估计用于自适应瘦脸"""
        left_eye_x = landmarks[33][0]
        right_eye_x = landmarks[263][0] 
        nose_x = landmarks[1][0]
        
        eye_center_x = (left_eye_x + right_eye_x) / 2
        nose_offset = nose_x - eye_center_x
        eye_distance = abs(right_eye_x - left_eye_x)
        
        if eye_distance > 0:
            yaw_ratio = nose_offset / eye_distance
            # 简化判断：正值右偏，负值左偏
            if abs(yaw_ratio) < 0.1:
                return "frontal"
            elif yaw_ratio > 0.1:
                return "right_profile" 
            else:
                return "left_profile"
        return "frontal"

    def tps_face_slimming(self, image, landmarks, intensity=0.5, region="both"):
        """
        进阶TPS瘦脸算法
        - jawline梯度收缩（下颌强，腮部弱）
        - 姿态自适应（左右非对称）
        - 区域选择（left/right/both）
        """
        h, w = image.shape[:2]
        
        # 获取源控制点和目标控制点
        src_points = []
        dst_points = []
        
        # 获取jawline控制点
        left_jawline = [landmarks[i] for i in self.JAWLINE_LEFT]
        right_jawline = [landmarks[i] for i in self.JAWLINE_RIGHT]
        
        # 计算面部中心轴
        face_center_x = landmarks[1][0]  # 鼻尖x坐标
        
        # 计算收缩因子（限制范围避免过度扭曲）
        max_intensity = 0.7  # 限制最大强度
        safe_intensity = min(intensity, max_intensity)
        base_shrink = safe_intensity * 0.12  # 基础收缩幅度
        
        # 姿态自适应
        face_pose = self.estimate_face_pose(landmarks)
        left_multiplier = 1.0
        right_multiplier = 1.0
        
        if face_pose == "left_profile":
            left_multiplier = 1.3   # 可见侧加强
            right_multiplier = 0.7  # 隐藏侧减弱
        elif face_pose == "right_profile":
            left_multiplier = 0.7   # 隐藏侧减弱  
            right_multiplier = 1.3  # 可见侧加强
        
        print(f"姿态: {face_pose}, 左侧系数: {left_multiplier:.1f}, 右侧系数: {right_multiplier:.1f}")
        
        # 左侧jawline梯度收缩
        if region in ["both", "left"]:
            for i, point in enumerate(left_jawline):
                src_points.append(point)
                
                # 梯度收缩：下颌部分（前几个点）收缩强，腮部（后几个点）收缩弱
                gradient_ratio = 1.0 - (i / len(left_jawline)) * 0.6  # 从1.0递减到0.4
                local_shrink = base_shrink * gradient_ratio * left_multiplier
                
                # 向面部中心收缩
                new_x = point[0] + (face_center_x - point[0]) * local_shrink
                dst_points.append([new_x, point[1]])
        else:
            # 不变形，添加为稳定点
            for point in left_jawline:
                src_points.append(point)
                dst_points.append(point)
        
        # 右侧jawline梯度收缩  
        if region in ["both", "right"]:
            for i, point in enumerate(right_jawline):
                src_points.append(point)
                
                # 梯度收缩：下颌部分收缩强，腮部收缩弱
                gradient_ratio = 1.0 - (i / len(right_jawline)) * 0.6  # 从1.0递减到0.4
                local_shrink = base_shrink * gradient_ratio * right_multiplier
                
                # 向面部中心收缩
                new_x = point[0] + (face_center_x - point[0]) * local_shrink
                dst_points.append([new_x, point[1]])
        else:
            # 不变形，添加为稳定点
            for point in right_jawline:
                src_points.append(point)
                dst_points.append(point)
        
        # 添加稳定锚点（关键：保持面部结构）
        for anchor_idx in self.STABLE_ANCHORS:
            if anchor_idx < len(landmarks):
                anchor_point = landmarks[anchor_idx]
                src_points.append(anchor_point)
                dst_points.append(anchor_point)  # 锚点位置不变
        
        # 添加边界锚点（防止图像边缘变形）
        boundary_margin = 50  # 边界留白
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
        
        # 转换为numpy数组
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        print(f"TPS控制点: {len(self.JAWLINE_LEFT + self.JAWLINE_RIGHT)} jawline + {len(self.STABLE_ANCHORS)} anchor + 8 boundary = {len(src_points)} 总计")
        
        # 使用OpenCV的TPS变换
        try:
            tps = cv2.createThinPlateSplineShapeTransformer()
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
            
            # 估计变换
            tps.estimateTransformation(
                dst_points.reshape(1, -1, 2), 
                src_points.reshape(1, -1, 2), 
                matches
            )
            
            # 应用变换
            result = tps.warpImage(image)
            
            # 检查结果有效性
            if result is None or result.size == 0:
                print("TPS变换返回空结果，使用原图")
                return image
            
            # 抗锯齿后处理
            result = cv2.bilateralFilter(result, 5, 80, 80)
            
            return result
            
        except Exception as e:
            print(f"TPS变换失败: {e}")
            return image
    
    def face_slimming(self, image, intensity=0.5, region="both", debug=False):
        """主处理函数 - 进阶版"""
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            print("错误：未检测到面部关键点")
            return image
        
        print(f"检测到 {len(landmarks)} 个关键点, 区域: {region}")
        
        # 调试模式：显示控制点
        if debug:
            debug_img = image.copy()
            
            # 绘制jawline控制点（绿色，显示梯度）
            for i, idx in enumerate(self.JAWLINE_LEFT + self.JAWLINE_RIGHT):
                if idx < len(landmarks):
                    # 梯度颜色：前面点（下颌）更红，后面点（腮部）更绿
                    is_left = i < len(self.JAWLINE_LEFT)
                    local_i = i if is_left else i - len(self.JAWLINE_LEFT)
                    max_len = len(self.JAWLINE_LEFT) if is_left else len(self.JAWLINE_RIGHT)
                    gradient = local_i / max_len
                    
                    color = (0, int(255 * gradient), int(255 * (1 - gradient)))  # 绿到红梯度
                    cv2.circle(debug_img, tuple(landmarks[idx]), 5, color, -1)
                    cv2.putText(debug_img, f"{local_i}", tuple(landmarks[idx] + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # 绘制稳定锚点（蓝色）
            for idx in self.STABLE_ANCHORS:
                if idx < len(landmarks):
                    cv2.circle(debug_img, tuple(landmarks[idx]), 4, (255, 0, 0), -1)
            
            # 显示姿态信息
            pose = self.estimate_face_pose(landmarks)
            cv2.putText(debug_img, f"Pose: {pose}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            return debug_img
        
        # 应用进阶TPS瘦脸
        return self.tps_face_slimming(image, landmarks, intensity, region)

def main():
    """进阶版主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='进阶TPS瘦脸工具 - 梯度收缩+姿态自适应')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--intensity', type=float, default=0.5, help='瘦脸强度(0.3-0.7推荐)')
    parser.add_argument('--region', choices=['left', 'right', 'both'], default='both', 
                       help='变形区域：left=仅左脸, right=仅右脸, both=双侧')
    parser.add_argument('--debug', action='store_true', help='显示控制点和梯度')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"错误：无法加载图片 {args.input}")
        return
    
    print(f"原图尺寸: {image.shape}")
    
    # 强度安全检查
    if args.intensity > 0.7:
        print(f"警告：强度 {args.intensity} 可能造成扭曲，自动限制到0.7")
        args.intensity = 0.7
    
    # 创建处理器
    processor = SimpleFaceSlimming()
    
    print("开始进阶TPS瘦脸处理...")
    result = processor.face_slimming(image, args.intensity, args.region, args.debug)
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"处理完成！保存到: {args.output}")
    
    # 生成对比图
    if args.comparison and not args.debug:
        comparison = np.hstack([image, result])
        cv2.line(comparison, (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
        
        # 添加标签
        cv2.putText(comparison, 'Original', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(comparison, f'TPS-{args.region.upper()} {args.intensity}', (image.shape[1] + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        comparison_path = args.output.replace('.jpg', '_comparison.jpg')
        cv2.imwrite(comparison_path, comparison)
        print(f"对比图保存到: {comparison_path}")
    
    # 使用建议
    print("\n✅ 进阶功能已添加:")
    print("🔄 Jawline梯度收缩 - 下颌强，腮部弱")
    print("🎯 姿态自适应 - 侧脸时可见侧加强")
    print("📍 区域选择 - 可单独瘦左脸或右脸")
    print("🎨 双边滤波抗锯齿 - 视觉效果更平滑")
    print("\n📖 使用建议：")
    print("• 正常双侧：--region both --intensity 0.5")
    print("• 单侧瘦脸：--region left --intensity 0.6") 
    print("• 侧脸照片：自动检测姿态并调整")
    print("• 查看梯度：--debug （绿到红显示收缩强度）")
    print("• 生成对比：--comparison")
    print("⚠️  建议强度范围：0.3-0.7")

if __name__ == "__main__":
    main()
