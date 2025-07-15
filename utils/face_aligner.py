#!/usr/bin/env python3
"""
FFHQ标准人脸对齐器
基于官方FFHQ数据集预处理算法，使用dlib 68点关键点检测
专为HyperStyle + StyleGAN2优化
"""

import numpy as np
import PIL.Image
import scipy.ndimage
import dlib
import cv2
import os
from typing import Optional, Union

class FFHQFaceAligner:
    def __init__(self, 
                 predictor_path: str = "./models/shape_predictor_68_face_landmarks.dat",
                 output_size: int = 1024,
                 transform_size: int = 4096,
                 enable_padding: bool = True):
        """
        初始化FFHQ标准人脸对齐器
        
        Args:
            predictor_path: dlib 68点关键点检测器模型路径
            output_size: 输出图像尺寸
            transform_size: 中间变换尺寸
            enable_padding: 是否启用边界填充
        """
        self.output_size = output_size
        self.transform_size = transform_size
        self.enable_padding = enable_padding
        
        # 检查dlib模型文件
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"dlib模型文件不存在: {predictor_path}\n"
                f"请从以下链接下载:\n"
                f"http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        
        # 初始化dlib检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        print("✅ FFHQ标准人脸对齐器初始化完成")
    
    def get_landmarks(self, image_path: str) -> Optional[np.ndarray]:
        """
        使用dlib提取68个关键点
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            landmarks: shape=(68, 2) 的关键点数组 或 None
        """
        try:
            # 加载图像
            img = dlib.load_rgb_image(image_path)
            
            # 检测人脸
            dets = self.detector(img, 1)
            
            if len(dets) == 0:
                print(f"⚠️ 未检测到人脸: {os.path.basename(image_path)}")
                return None
            
            if len(dets) > 1:
                print(f"⚠️ 检测到多张人脸，使用第一张: {os.path.basename(image_path)}")
            
            # 提取第一张人脸的关键点
            shape = self.predictor(img, dets[0])
            
            # 转换为numpy数组
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            return landmarks
            
        except Exception as e:
            print(f"❌ 关键点提取失败 {image_path}: {e}")
            return None
    
    def align_face(self, image_path: str) -> Optional[PIL.Image.Image]:
        """
        使用官方FFHQ算法对齐人脸
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            aligned_image: 对齐后的PIL图像 或 None
        """
        try:
            # 提取关键点
            lm = self.get_landmarks(image_path)
            if lm is None:
                return None
            
            # 按照官方FFHQ算法分组关键点
            lm_chin = lm[0:17]      # 下巴轮廓
            lm_eyebrow_left = lm[17:22]   # 左眉毛
            lm_eyebrow_right = lm[22:27]  # 右眉毛
            lm_nose = lm[27:31]     # 鼻梁
            lm_nostrils = lm[31:36] # 鼻孔
            lm_eye_left = lm[36:42] # 左眼
            lm_eye_right = lm[42:48] # 右眼
            lm_mouth_outer = lm[48:60] # 嘴巴外轮廓
            lm_mouth_inner = lm[60:68] # 嘴巴内轮廓
            
            # 计算关键几何量（官方FFHQ算法）
            eye_left = np.mean(lm_eye_left, axis=0)
            eye_right = np.mean(lm_eye_right, axis=0)
            eye_avg = (eye_left + eye_right) * 0.5
            eye_to_eye = eye_right - eye_left
            
            mouth_left = lm_mouth_outer[0]
            mouth_right = lm_mouth_outer[6]
            mouth_avg = (mouth_left + mouth_right) * 0.5
            eye_to_mouth = mouth_avg - eye_avg
            
            # 选择定向裁剪矩形（官方FFHQ核心算法）
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.4, np.hypot(*eye_to_mouth) * 2.2)
            y = np.flipud(x) * [-1, 1]
            c = eye_avg + eye_to_mouth * 0.05
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
            qsize = np.hypot(*x) * 2
            
            # 读取原始图像
            img = PIL.Image.open(image_path)
            
            # 缩小处理（提升速度）
            shrink = int(np.floor(qsize / self.output_size * 0.5))
            if shrink > 1:
                rsize = (
                    int(np.rint(float(img.size[0]) / shrink)), 
                    int(np.rint(float(img.size[1]) / shrink))
                )
                img = img.resize(rsize, PIL.Image.LANCZOS)
                quad /= shrink
                qsize /= shrink
            
            # 裁剪
            border = max(int(np.rint(qsize * 0.15)), 5)
            crop = (
                int(np.floor(min(quad[:, 0]))), 
                int(np.floor(min(quad[:, 1]))), 
                int(np.ceil(max(quad[:, 0]))), 
                int(np.ceil(max(quad[:, 1])))
            )
            crop = (
                max(crop[0] - border, 0), 
                max(crop[1] - border, 0), 
                min(crop[2] + border, img.size[0]), 
                min(crop[3] + border, img.size[1])
            )
            
            if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
                img = img.crop(crop)
                quad -= crop[0:2]
            
            # 填充（官方FFHQ边界处理）
            pad = (
                int(np.floor(min(quad[:, 0]))), 
                int(np.floor(min(quad[:, 1]))), 
                int(np.ceil(max(quad[:, 0]))), 
                int(np.ceil(max(quad[:, 1])))
            )
            pad = (
                max(-pad[0] + border, 0), 
                max(-pad[1] + border, 0), 
                max(pad[2] - img.size[0] + border, 0), 
                max(pad[3] - img.size[1] + border, 0)
            )
            
            if self.enable_padding and max(pad) > border - 4:
                pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
                img_array = np.pad(
                    np.float32(img), 
                    ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 
                    'reflect'
                )
                
                h, w, _ = img_array.shape
                y, x, _ = np.ogrid[:h, :w, :1]
                
                # 羽化边界
                mask = np.maximum(
                    1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 
                    1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3])
                )
                
                blur = qsize * 0.02
                img_array += (scipy.ndimage.gaussian_filter(img_array, [blur, blur, 0]) - img_array) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                img_array += (np.median(img_array, axis=(0,1)) - img_array) * np.clip(mask, 0.0, 1.0)
                
                img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img_array), 0, 255)), 'RGB')
                quad += pad[:2]
            
            # 最终变换到标准尺寸
            img = img.transform(
                (self.transform_size, self.transform_size), 
                PIL.Image.QUAD, 
                (quad + 0.5).flatten(), 
                PIL.Image.BILINEAR
            )
            
            if self.output_size < self.transform_size:
                img = img.resize((self.output_size, self.output_size), PIL.Image.LANCZOS)
            
            return img
            
        except Exception as e:
            print(f"❌ 人脸对齐失败 {image_path}: {e}")
            return None
    
    def align_face_cv2(self, image_input: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        对齐人脸并返回OpenCV格式（BGR）
        
        Args:
            image_input: 图像路径或numpy数组
            
        Returns:
            aligned_image: BGR格式的numpy数组 或 None
        """
        if isinstance(image_input, np.ndarray):
            # 临时保存numpy数组为文件
            temp_path = "/tmp/temp_face_align.jpg"
            cv2.imwrite(temp_path, image_input)
            aligned_pil = self.align_face(temp_path)
            os.remove(temp_path)
        else:
            aligned_pil = self.align_face(image_input)
        
        if aligned_pil is None:
            return None
        
        # 转换为OpenCV格式
        aligned_rgb = np.array(aligned_pil)
        aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
        
        return aligned_bgr
    
    def batch_align(self, image_dir: str, output_dir: str) -> list:
        """
        批量对齐人脸
        
        Args:
            image_dir: 输入图像目录
            output_dir: 输出目录
            
        Returns:
            success_paths: 成功处理的输出路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 支持的图像格式
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend([
                f for f in os.listdir(image_dir) 
                if f.lower().endswith(ext.lower())
            ])
        
        success_paths = []
        
        for i, filename in enumerate(image_paths):
            input_path = os.path.join(image_dir, filename)
            print(f"📷 对齐图像 {i+1}/{len(image_paths)}: {filename}")
            
            try:
                aligned = self.align_face(input_path)
                if aligned is None:
                    continue
                
                # 保存对齐结果
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_aligned{ext}")
                aligned.save(output_path, quality=95)
                
                success_paths.append(output_path)
                
            except Exception as e:
                print(f"❌ 处理失败 {filename}: {e}")
                continue
        
        print(f"🎉 批量对齐完成! 成功: {len(success_paths)}/{len(image_paths)}")
        return success_paths
    
    def verify_alignment(self, aligned_image: Union[PIL.Image.Image, np.ndarray]) -> dict:
        """
        验证对齐质量（与FFHQ标准对比）
        
        Args:
            aligned_image: 对齐后的图像
            
        Returns:
            metrics: 质量指标字典
        """
        try:
            # 转换为PIL格式用于验证
            if isinstance(aligned_image, np.ndarray):
                if aligned_image.shape[2] == 3 and aligned_image.dtype == np.uint8:
                    # BGR -> RGB
                    rgb_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
                    pil_image = PIL.Image.fromarray(rgb_image)
                else:
                    pil_image = PIL.Image.fromarray(aligned_image)
            else:
                pil_image = aligned_image
            
            # 临时保存用于dlib分析
            temp_path = "/tmp/temp_verify.jpg"
            pil_image.save(temp_path)
            
            # 重新检测关键点
            landmarks = self.get_landmarks(temp_path)
            os.remove(temp_path)
            
            if landmarks is None:
                return {'valid': False, 'error': '无法重新检测关键点'}
            
            # 计算FFHQ标准指标
            eye_left = np.mean(landmarks[36:42], axis=0)
            eye_right = np.mean(landmarks[42:48], axis=0)
            mouth_left = landmarks[48]
            mouth_right = landmarks[54]
            
            # 双眼水平度
            eye_to_eye = eye_right - eye_left
            eye_angle = abs(np.degrees(np.arctan2(eye_to_eye[1], eye_to_eye[0])))
            
            # 双眼间距（应该符合FFHQ比例）
            eye_distance = np.linalg.norm(eye_to_eye)
            expected_distance = self.output_size * 0.25  # FFHQ标准比例
            distance_ratio = eye_distance / expected_distance
            
            # 眼嘴比例
            mouth_center = (mouth_left + mouth_right) * 0.5
            eye_center = (eye_left + eye_right) * 0.5
            eye_to_mouth_distance = np.linalg.norm(mouth_center - eye_center)
            eye_mouth_ratio = eye_to_mouth_distance / eye_distance
            
            return {
                'valid': True,
                'eye_angle': eye_angle,
                'distance_ratio': distance_ratio,
                'eye_mouth_ratio': eye_mouth_ratio,
                'ffhq_compliant': (
                    eye_angle < 3.0 and 
                    0.85 < distance_ratio < 1.15 and
                    1.1 < eye_mouth_ratio < 1.6
                )
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}

def main():
    """测试函数"""
    import sys
    
    if len(sys.argv) != 3:
        print("用法: python face_aligner.py input.jpg output.jpg")
        print("请确保已下载dlib模型文件:")
        print("wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        # 创建对齐器
        aligner = FFHQFaceAligner()
        
        # 对齐人脸
        print(f"📷 使用FFHQ标准算法对齐: {input_path}")
        aligned = aligner.align_face(input_path)
        
        if aligned is not None:
            # 保存结果
            aligned.save(output_path, quality=95)
            print(f"✅ 对齐完成: {output_path}")
            
            # 验证质量
            metrics = aligner.verify_alignment(aligned)
            if metrics['valid']:
                print(f"📊 FFHQ标准验证: {'✅ 合格' if metrics['ffhq_compliant'] else '⚠️  需要调整'}")
                print(f"   眼部角度: {metrics['eye_angle']:.2f}°")
                print(f"   距离比例: {metrics['distance_ratio']:.3f}")
                print(f"   眼嘴比例: {metrics['eye_mouth_ratio']:.3f}")
            else:
                print(f"⚠️ 质量验证失败: {metrics['error']}")
        else:
            print("❌ 对齐失败")
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")

if __name__ == "__main__":
    main()
