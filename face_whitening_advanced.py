#!/usr/bin/env python3
"""
进阶版专业美白算法 - 分层处理+自适应增强
基于工业界标准的分层美白方案，支持姿态自适应和区域选择
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse

class AdvancedFaceWhitening:
    def __init__(self, face_mesh=None):
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
        
        # 核心美白区域（颧骨与脸颊中部 - 强度100%）
        self.CORE_WHITENING_REGION = [117, 118, 119, 121, 122, 123, 147, 213, 192, 352, 351, 350, 346]
        
        # 完整面部轮廓（外围区域 - 强度60-80%）
        self.FULL_FACE_CONTOUR = [10, 338, 297, 332, 284, 251, 389, 356, 454, 
                                  323, 361, 288, 397, 365, 379, 378, 400, 377, 
                                  152, 148, 176, 149, 150, 136, 172, 58, 132, 
                                  93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # 肤色保护区域（避免过度处理）
        self.SKIN_PROTECTION_ANCHORS = [
            # 眼部区域
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            # 鼻部区域  
            1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 31, 228, 229, 230, 231, 232, 233, 244, 245, 122, 6, 202, 214, 234,
            # 嘴部区域
            164, 0, 17, 18, 200, 199, 175, 178, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410, 454, 323, 366, 401, 361, 340, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410, 454, 323, 366, 401, 361, 340
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
        """
        改进的姿态估计 - 参考瘦脸脚本逻辑
        使用多个特征点组合判断，提高准确性
        """
        # 关键特征点
        left_eye_x = landmarks[33][0]
        right_eye_x = landmarks[263][0] 
        nose_tip_x = landmarks[1][0]
        nose_bridge_x = landmarks[6][0]  # 鼻梁中点
        
        # 面部中轴线估计
        eye_center_x = (left_eye_x + right_eye_x) / 2
        nose_center_x = (nose_tip_x + nose_bridge_x) / 2
        
        # 计算偏移比例
        nose_offset = nose_center_x - eye_center_x
        eye_distance = abs(right_eye_x - left_eye_x)
        
        if eye_distance > 0:
            yaw_ratio = nose_offset / eye_distance
            
            # 参考瘦脸脚本的阈值设置，更严格的判断
            if abs(yaw_ratio) < 0.05:  # 更严格的正脸判断
                return "frontal"
            elif yaw_ratio > 0.05:
                return "right_profile" 
            else:
                return "left_profile"
        return "frontal"
    
    def analyze_skin_brightness(self, image, landmarks):
        """分析面部区域亮度用于自适应调整"""
        # 提取核心区域的亮度信息
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        core_points = np.array([landmarks[i] for i in self.CORE_WHITENING_REGION if i < len(landmarks)], np.int32)
        
        if len(core_points) > 0:
            cv2.fillConvexPoly(mask, core_points, 255)
            
            # 计算LAB色彩空间的L通道平均值
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab_image[:, :, 0]
            mean_brightness = np.mean(l_channel[mask > 0])
            
            return mean_brightness
        return 128  # 默认中等亮度
    
    def create_layered_masks(self, image, landmarks):
        """创建分层遮罩：核心区域+外围区域"""
        h, w = image.shape[:2]
        
        # 核心遮罩（强度100%）
        core_mask = np.zeros((h, w), dtype=np.uint8)
        core_points = np.array([landmarks[i] for i in self.CORE_WHITENING_REGION if i < len(landmarks)], np.int32)
        if len(core_points) > 0:
            cv2.fillPoly(core_mask, [core_points], 255)
        
        # 完整面部遮罩
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_points = np.array([landmarks[i] for i in self.FULL_FACE_CONTOUR if i < len(landmarks)], np.int32)
        if len(full_points) > 0:
            cv2.fillPoly(full_mask, [full_points], 255)
        
        # 外围遮罩 = 完整面部 - 核心区域
        outer_mask = cv2.subtract(full_mask, core_mask)
        
        # 高斯模糊软化边界
        core_mask = cv2.GaussianBlur(core_mask, (15, 15), 0)
        outer_mask = cv2.GaussianBlur(outer_mask, (25, 25), 0)
        
        return core_mask, outer_mask
    
    def adaptive_whitening_pipeline(self, face_region, brightness_level, intensity=0.5, region_type="core"):
        """
        自适应美白管道处理
        - 基于亮度自动调整参数
        - 区域类型决定处理强度
        - LAB色彩空间保持肤色自然
        """
        if face_region.sum() == 0:
            return face_region
        
        # 转换到LAB色彩空间（更好的肤色保护）
        lab_image = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image)
        
        # 根据区域类型调整基础强度
        if region_type == "core":
            base_intensity = intensity
        else:  # outer
            base_intensity = intensity * 0.7  # 外围区域强度降低
        
        # 基于亮度自适应调整（暗部增强更多，亮部适度）
        if brightness_level < 100:  # 较暗
            brightness_multiplier = 1.3
        elif brightness_level > 160:  # 较亮
            brightness_multiplier = 0.8
        else:  # 中等
            brightness_multiplier = 1.0
        
        final_intensity = base_intensity * brightness_multiplier
        final_intensity = np.clip(final_intensity, 0.1, 0.9)  # 安全范围
        
        # 步骤1: L通道亮度增强（主要美白效果）
        l_enhanced = l.astype(np.float32)
        l_enhanced = l_enhanced * (1.0 + final_intensity * 0.15)  # 15%亮度提升
        l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)
        
        # 步骤2: 轻微调整a通道（减少红色偏向，更自然）
        a_adjusted = a.astype(np.float32)
        a_adjusted = a_adjusted * (1.0 - final_intensity * 0.05)  # 减少5%红色
        a_adjusted = np.clip(a_adjusted, 0, 255).astype(np.uint8)
        
        # 合并通道并转换回BGR
        enhanced_lab = cv2.merge([l_enhanced, a_adjusted, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 步骤3: 轻微高斯模糊（平滑肌理）
        if region_type == "core":
            result = cv2.bilateralFilter(result, 5, 40, 40)
        
        return result
    
    def pose_adaptive_processing(self, image, landmarks, intensity, face_pose):
        """
        姿态自适应处理 - 参考瘦脸脚本的更精确实现
        """
        # 基于姿态调整左右区域强度 - 参考瘦脸脚本的系数
        left_multiplier = 1.0  
        right_multiplier = 1.0
        
        if face_pose == "left_profile":
            left_multiplier = 1.3   # 可见侧加强 - 与瘦脸脚本一致
            right_multiplier = 0.7  # 隐藏侧减弱 - 与瘦脸脚本一致
        elif face_pose == "right_profile":
            left_multiplier = 0.7   # 隐藏侧减弱 - 与瘦脸脚本一致  
            right_multiplier = 1.3  # 可见侧加强 - 与瘦脸脚本一致
        
        # 创建左右分区遮罩 - 使用鼻尖作为分界线
        h, w = image.shape[:2]
        face_center_x = landmarks[1][0]  # 鼻尖x坐标
        
        left_region_mask = np.zeros((h, w), dtype=np.uint8)
        left_region_mask[:, :face_center_x] = 255
        
        right_region_mask = np.zeros((h, w), dtype=np.uint8) 
        right_region_mask[:, face_center_x:] = 255
        
        return left_multiplier, right_multiplier, left_region_mask, right_region_mask
    
    def face_whitening(self, image, intensity=0.5, region="both", debug=False):
        """主处理函数 - 进阶分层美白"""
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            print("错误：未检测到面部关键点")
            return image
        
        print(f"检测到 {len(landmarks)} 个关键点, 区域: {region}")
        
        # 调试模式：显示控制点和区域
        if debug:
            debug_img = image.copy()
            
            # 绘制核心美白区域（红色）
            for idx in self.CORE_WHITENING_REGION:
                if idx < len(landmarks):
                    cv2.circle(debug_img, tuple(landmarks[idx]), 4, (0, 0, 255), -1)
            
            # 绘制完整面部轮廓（绿色）
            for idx in self.FULL_FACE_CONTOUR:
                if idx < len(landmarks):
                    cv2.circle(debug_img, tuple(landmarks[idx]), 2, (0, 255, 0), -1)
            
            # 显示姿态和亮度信息
            pose = self.estimate_face_pose(landmarks)
            brightness = self.analyze_skin_brightness(image, landmarks)
            
            # 添加详细的姿态检测信息 - 参考瘦脸脚本的调试输出
            left_eye_x = landmarks[33][0]
            right_eye_x = landmarks[263][0] 
            nose_tip_x = landmarks[1][0]
            nose_bridge_x = landmarks[6][0]
            eye_center_x = (left_eye_x + right_eye_x) / 2
            nose_center_x = (nose_tip_x + nose_bridge_x) / 2
            eye_distance = abs(right_eye_x - left_eye_x)
            yaw_ratio = (nose_center_x - eye_center_x) / eye_distance if eye_distance > 0 else 0
            
            cv2.putText(debug_img, f"Pose: {pose} (yaw:{yaw_ratio:.3f})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(debug_img, f"Brightness: {brightness:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(debug_img, f"Core Region: {len(self.CORE_WHITENING_REGION)} points", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(debug_img, f"Full Contour: {len(self.FULL_FACE_CONTOUR)} points", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 绘制姿态检测关键点
            cv2.circle(debug_img, (int(eye_center_x), landmarks[33][1]), 6, (255, 0, 255), -1)  # 眼部中心
            cv2.circle(debug_img, (int(nose_center_x), landmarks[1][1]), 6, (0, 255, 255), -1)  # 鼻部中心
            cv2.line(debug_img, (int(eye_center_x), 0), (int(eye_center_x), debug_img.shape[0]), (255, 0, 255), 2)  # 眼部中轴线
            cv2.line(debug_img, (int(nose_center_x), 0), (int(nose_center_x), debug_img.shape[0]), (0, 255, 255), 2)  # 鼻部中轴线
            
            return debug_img
        
        # 分析面部姿态和亮度
        face_pose = self.estimate_face_pose(landmarks)
        brightness_level = self.analyze_skin_brightness(image, landmarks)
        
        print(f"姿态: {face_pose}, 平均亮度: {brightness_level:.1f}")
        
        # 创建分层遮罩
        core_mask, outer_mask = self.create_layered_masks(image, landmarks)
        
        if core_mask.sum() == 0:
            print("警告：未能创建有效的美白遮罩")
            return image
        
        # 姿态自适应处理
        left_mult, right_mult, left_mask, right_mask = self.pose_adaptive_processing(
            image, landmarks, intensity, face_pose
        )
        
        print(f"左侧强度系数: {left_mult:.1f}, 右侧强度系数: {right_mult:.1f}")
        
        # 开始分层美白处理
        result = image.astype(np.float32)
        
        # 处理核心区域（强度100%）
        if region in ["both", "core"]:
            core_region = cv2.bitwise_and(image, image, mask=core_mask)
            
            # 左右分区处理
            if region != "right":  # 处理左侧
                left_core_mask = cv2.bitwise_and(core_mask, left_mask)
                if left_core_mask.sum() > 0:
                    left_core_region = cv2.bitwise_and(image, image, mask=left_core_mask)
                    enhanced_left_core = self.adaptive_whitening_pipeline(
                        left_core_region, brightness_level, intensity * left_mult, "core"
                    )
                    left_core_mask_norm = left_core_mask.astype(np.float32) / 255.0
                    result = result * (1 - left_core_mask_norm[..., None]) + \
                            enhanced_left_core.astype(np.float32) * left_core_mask_norm[..., None]
            
            if region != "left":  # 处理右侧
                right_core_mask = cv2.bitwise_and(core_mask, right_mask)
                if right_core_mask.sum() > 0:
                    right_core_region = cv2.bitwise_and(image, image, mask=right_core_mask)
                    enhanced_right_core = self.adaptive_whitening_pipeline(
                        right_core_region, brightness_level, intensity * right_mult, "core"
                    )
                    right_core_mask_norm = right_core_mask.astype(np.float32) / 255.0
                    result = result * (1 - right_core_mask_norm[..., None]) + \
                            enhanced_right_core.astype(np.float32) * right_core_mask_norm[..., None]
        
        # 处理外围区域（强度60-80%）
        if region in ["both", "outer"]:
            # 左右分区处理外围
            if region != "right":  # 处理左侧
                left_outer_mask = cv2.bitwise_and(outer_mask, left_mask)
                if left_outer_mask.sum() > 0:
                    left_outer_region = cv2.bitwise_and(image, image, mask=left_outer_mask)
                    enhanced_left_outer = self.adaptive_whitening_pipeline(
                        left_outer_region, brightness_level, intensity * left_mult, "outer"
                    )
                    left_outer_mask_norm = left_outer_mask.astype(np.float32) / 255.0
                    result = result * (1 - left_outer_mask_norm[..., None]) + \
                            enhanced_left_outer.astype(np.float32) * left_outer_mask_norm[..., None]
            
            if region != "left":  # 处理右侧
                right_outer_mask = cv2.bitwise_and(outer_mask, right_mask)
                if right_outer_mask.sum() > 0:
                    right_outer_region = cv2.bitwise_and(image, image, mask=right_outer_mask)
                    enhanced_right_outer = self.adaptive_whitening_pipeline(
                        right_outer_region, brightness_level, intensity * right_mult, "outer"
                    )
                    right_outer_mask_norm = right_outer_mask.astype(np.float32) / 255.0
                    result = result * (1 - right_outer_mask_norm[..., None]) + \
                            enhanced_right_outer.astype(np.float32) * right_outer_mask_norm[..., None]
        
        # 转换回uint8并进行最终的边界平滑
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = cv2.bilateralFilter(result, 3, 20, 20)
        
        return result

def main():
    """进阶版主函数"""
    parser = argparse.ArgumentParser(description='进阶分层美白工具 - 自适应+姿态感知')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出图片路径')
    parser.add_argument('--intensity', type=float, default=0.5, help='美白强度(0.3-0.8推荐)')
    parser.add_argument('--region', choices=['left', 'right', 'both', 'core', 'outer'], default='both', 
                       help='美白区域：left=仅左脸, right=仅右脸, core=仅核心区域, outer=仅外围区域, both=分层处理')
    parser.add_argument('--debug', action='store_true', help='显示控制点和区域划分')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    
    args = parser.parse_args()
    
    # 加载图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"错误：无法加载图片 {args.input}")
        return
    
    print(f"原图尺寸: {image.shape}")
    
    # 强度安全检查
    if args.intensity > 0.8:
        print(f"警告：强度 {args.intensity} 可能造成过度美白，自动限制到0.8")
        args.intensity = 0.8
    elif args.intensity < 0.1:
        print(f"警告：强度 {args.intensity} 过低，自动调整到0.1")
        args.intensity = 0.1
    
    # 创建处理器
    processor = AdvancedFaceWhitening()
    
    print("开始进阶分层美白处理...")
    result = processor.face_whitening(image, args.intensity, args.region, args.debug)
    
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
        cv2.putText(comparison, f'Whitening-{args.region.upper()} {args.intensity}', (image.shape[1] + 20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        comparison_path = args.output.replace('.jpg', '_comparison.jpg')
        cv2.imwrite(comparison_path, comparison)
        print(f"对比图保存到: {comparison_path}")
    
    # 使用建议
    print("\n✅ 进阶功能已添加:")
    print("🎯 分层美白 - 核心区域100%，外围区域70%")
    print("🔄 自适应强度 - 基于面部亮度自动调整")
    print("🎨 LAB色彩空间 - 更好的肤色保护")
    print("📍 姿态感知 - 侧脸时可见侧适度加强")
    print("🌟 双边滤波 - 最终边界平滑处理")
    print("\n📖 使用建议：")
    print("• 标准分层美白：--region both --intensity 0.5")
    print("• 仅美白核心区域：--region core --intensity 0.6") 
    print("• 单侧美白：--region left --intensity 0.5")
    print("• 查看区域划分：--debug")
    print("• 生成对比图：--comparison")
    print("⚠️  推荐强度范围：0.3-0.8")
    print("💡 LAB色彩空间处理，肤色更自然！")

if __name__ == "__main__":
    main()
