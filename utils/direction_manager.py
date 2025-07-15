#!/usr/bin/env python3
"""
方向向量管理器
计算、缓存和应用StyleGAN2潜在空间的编辑方向向量
"""

import torch
import numpy as np
import os
import json
from typing import Dict, Union, List, Optional, Tuple
from datetime import datetime

class DirectionManager:
    def __init__(self, 
                 cache_dir: str = "./data/cache/directions",
                 device: str = 'auto'):
        """
        初始化方向向量管理器
        
        Args:
            cache_dir: 方向向量缓存目录
            device: 计算设备
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 内存缓存
        self.direction_cache = {}
        
        print(f"🖥️  DirectionManager使用设备: {self.device}")
        print("✅ 方向向量管理器初始化完成")
    
    def compute_direction(self, 
                         latent_a: torch.Tensor, 
                         latent_b: torch.Tensor,
                         normalize: bool = False) -> torch.Tensor:
        """
        计算两个潜在向量之间的方向向量
        
        Args:
            latent_a: 起始潜在向量
            latent_b: 目标潜在向量  
            normalize: 是否标准化方向向量
            
        Returns:
            direction: 方向向量 (B -> A，即从B到A的方向)
        """
        try:
            # 确保在同一设备上
            latent_a = latent_a.to(self.device)
            latent_b = latent_b.to(self.device)
            
            # 计算方向向量: A - B (从B变化到A的方向)
            direction = latent_a - latent_b
            
            # 可选标准化
            if normalize:
                direction = self._normalize_direction(direction)
            
            return direction
            
        except Exception as e:
            print(f"❌ 计算方向向量失败: {e}")
            raise
    
    def _normalize_direction(self, direction: torch.Tensor) -> torch.Tensor:
        """
        标准化方向向量
        
        Args:
            direction: 原始方向向量
            
        Returns:
            normalized_direction: 标准化后的方向向量
        """
        # L2标准化
        norm = torch.norm(direction, dim=-1, keepdim=True)
        # 避免除零
        norm = torch.clamp(norm, min=1e-8)
        return direction / norm
    
    def compute_directions_from_pairs(self, 
                                    reference_pairs: Dict[str, Tuple],
                                    hyperstyle_encoder) -> Dict[str, torch.Tensor]:
        """
        从参考图片对批量计算方向向量
        
        Args:
            reference_pairs: 参考图片对字典 {effect_name: (img_a, img_b)}
            hyperstyle_encoder: HyperStyle编码器实例
            
        Returns:
            directions: 方向向量字典 {effect_name: direction_tensor}
        """
        directions = {}
        
        print(f"🔄 计算 {len(reference_pairs)} 个方向向量...")
        
        for effect_name, (img_a, img_b) in reference_pairs.items():
            try:
                print(f"   计算 {effect_name} 方向向量...")
                
                # 编码参考图片
                latent_a = hyperstyle_encoder.encode_image(img_a)
                latent_b = hyperstyle_encoder.encode_image(img_b)
                
                # 计算方向向量
                direction = self.compute_direction(latent_a, latent_b)
                directions[effect_name] = direction
                
                # 显示统计信息
                stats = self.analyze_direction(direction)
                print(f"     范数: {stats['norm']:.3f}, 均值: {stats['mean']:.3f}")
                
            except Exception as e:
                print(f"❌ 计算 {effect_name} 方向失败: {e}")
                continue
        
        print(f"✅ 完成 {len(directions)} 个方向向量计算")
        return directions
    
    def apply_direction(self, 
                       latent: torch.Tensor,
                       direction: torch.Tensor,
                       strength: float = 1.0) -> torch.Tensor:
        """
        应用单个方向向量到潜在向量
        
        Args:
            latent: 原始潜在向量
            direction: 方向向量
            strength: 应用强度
            
        Returns:
            edited_latent: 编辑后的潜在向量
        """
        try:
            latent = latent.to(self.device)
            direction = direction.to(self.device)
            
            # 应用方向: latent + strength * direction
            edited_latent = latent + strength * direction
            
            return edited_latent
            
        except Exception as e:
            print(f"❌ 应用方向向量失败: {e}")
            raise
    
    def apply_multiple_directions(self,
                                latent: torch.Tensor,
                                directions: Dict[str, torch.Tensor],
                                strengths: Dict[str, float]) -> torch.Tensor:
        """
        应用多个方向向量到潜在向量
        
        Args:
            latent: 原始潜在向量
            directions: 方向向量字典
            strengths: 强度字典
            
        Returns:
            edited_latent: 编辑后的潜在向量
        """
        try:
            edited_latent = latent.clone().to(self.device)
            
            print(f"✨ 应用 {len([k for k, v in strengths.items() if v > 0])} 个编辑效果:")
            
            for effect_name, strength in strengths.items():
                if strength > 0 and effect_name in directions:
                    direction = directions[effect_name].to(self.device)
                    edited_latent = self.apply_direction(edited_latent, direction, strength)
                    print(f"   {effect_name}: {strength:.3f}")
            
            return edited_latent
            
        except Exception as e:
            print(f"❌ 应用多个方向失败: {e}")
            raise
    
    def save_direction(self, 
                      direction: torch.Tensor, 
                      effect_name: str,
                      metadata: Optional[Dict] = None):
        """
        保存方向向量到文件
        
        Args:
            direction: 方向向量
            effect_name: 效果名称
            metadata: 元数据信息
        """
        try:
            # 准备保存数据
            save_data = {
                'direction': direction.cpu(),
                'effect_name': effect_name,
                'shape': list(direction.shape),
                'device': str(direction.device),
                'dtype': str(direction.dtype),
                'created_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            # 保存路径
            save_path = os.path.join(self.cache_dir, f"{effect_name}_direction.pt")
            
            torch.save(save_data, save_path)
            print(f"💾 方向向量保存: {effect_name} -> {save_path}")
            
            # 同时缓存到内存
            self.direction_cache[effect_name] = direction.to(self.device)
            
        except Exception as e:
            print(f"❌ 保存方向向量失败: {e}")
            raise
    
    def load_direction(self, effect_name: str) -> Optional[torch.Tensor]:
        """
        加载方向向量
        
        Args:
            effect_name: 效果名称
            
        Returns:
            direction: 方向向量 或 None
        """
        try:
            # 首先检查内存缓存
            if effect_name in self.direction_cache:
                return self.direction_cache[effect_name]
            
            # 从文件加载
            load_path = os.path.join(self.cache_dir, f"{effect_name}_direction.pt")
            
            if not os.path.exists(load_path):
                print(f"⚠️ 方向向量文件不存在: {effect_name}")
                return None
            
            save_data = torch.load(load_path, map_location=self.device)
            
            if isinstance(save_data, dict) and 'direction' in save_data:
                direction = save_data['direction'].to(self.device)
                print(f"📁 加载方向向量: {effect_name} (形状: {save_data['shape']})")
            else:
                # 兼容旧格式
                direction = save_data.to(self.device)
                print(f"📁 加载方向向量: {effect_name}")
            
            # 缓存到内存
            self.direction_cache[effect_name] = direction
            
            return direction
            
        except Exception as e:
            print(f"❌ 加载方向向量失败 {effect_name}: {e}")
            return None
    
    def load_all_directions(self) -> Dict[str, torch.Tensor]:
        """
        加载所有可用的方向向量
        
        Returns:
            directions: 所有方向向量字典
        """
        directions = {}
        
        # 查找所有方向向量文件
        direction_files = [f for f in os.listdir(self.cache_dir) if f.endswith('_direction.pt')]
        
        print(f"📁 发现 {len(direction_files)} 个方向向量文件")
        
        for filename in direction_files:
            effect_name = filename.replace('_direction.pt', '')
            direction = self.load_direction(effect_name)
            if direction is not None:
                directions[effect_name] = direction
        
        print(f"✅ 成功加载 {len(directions)} 个方向向量")
        return directions
    
    def save_directions_batch(self, 
                            directions: Dict[str, torch.Tensor],
                            metadata: Optional[Dict] = None):
        """
        批量保存方向向量
        
        Args:
            directions: 方向向量字典
            metadata: 通用元数据
        """
        print(f"💾 批量保存 {len(directions)} 个方向向量...")
        
        for effect_name, direction in directions.items():
            effect_metadata = metadata.copy() if metadata else {}
            effect_metadata['batch_saved'] = True
            
            self.save_direction(direction, effect_name, effect_metadata)
        
        print("✅ 批量保存完成")
    
    def analyze_direction(self, direction: torch.Tensor) -> Dict:
        """
        分析方向向量的统计信息
        
        Args:
            direction: 方向向量
            
        Returns:
            stats: 统计信息字典
        """
        direction_cpu = direction.cpu()
        
        stats = {
            'shape': list(direction.shape),
            'norm': float(torch.norm(direction_cpu)),
            'mean': float(torch.mean(direction_cpu)),
            'std': float(torch.std(direction_cpu)),
            'min': float(torch.min(direction_cpu)),
            'max': float(torch.max(direction_cpu)),
            'num_elements': direction_cpu.numel(),
            'sparsity': float(torch.sum(torch.abs(direction_cpu) < 1e-6)) / direction_cpu.numel()
        }
        
        return stats
    
    def list_available_directions(self) -> List[str]:
        """
        列出所有可用的方向向量
        
        Returns:
            effect_names: 效果名称列表
        """
        direction_files = [f for f in os.listdir(self.cache_dir) if f.endswith('_direction.pt')]
        effect_names = [f.replace('_direction.pt', '') for f in direction_files]
        
        # 同时检查内存缓存
        cached_names = list(self.direction_cache.keys())
        
        # 合并并去重
        all_names = list(set(effect_names + cached_names))
        all_names.sort()
        
        return all_names
    
    def get_direction_info(self, effect_name: str) -> Optional[Dict]:
        """
        获取方向向量的详细信息
        
        Args:
            effect_name: 效果名称
            
        Returns:
            info: 详细信息字典 或 None
        """
        try:
            load_path = os.path.join(self.cache_dir, f"{effect_name}_direction.pt")
            
            if not os.path.exists(load_path):
                return None
            
            save_data = torch.load(load_path, map_location='cpu')
            
            if isinstance(save_data, dict):
                info = {
                    'effect_name': save_data.get('effect_name', effect_name),
                    'shape': save_data.get('shape', []),
                    'created_at': save_data.get('created_at', 'Unknown'),
                    'metadata': save_data.get('metadata', {}),
                    'file_path': load_path
                }
                
                # 添加统计信息
                if 'direction' in save_data:
                    direction_stats = self.analyze_direction(save_data['direction'])
                    info.update(direction_stats)
                
                return info
            else:
                return {
                    'effect_name': effect_name,
                    'shape': list(save_data.shape),
                    'file_path': load_path
                }
                
        except Exception as e:
            print(f"❌ 获取方向信息失败 {effect_name}: {e}")
            return None
    
    def clear_cache(self, memory_only: bool = False):
        """
        清理缓存
        
        Args:
            memory_only: 是否只清理内存缓存
        """
        # 清理内存缓存
        self.direction_cache.clear()
        print("🗑️  内存缓存已清理")
        
        if not memory_only:
            # 清理文件缓存
            direction_files = [f for f in os.listdir(self.cache_dir) if f.endswith('_direction.pt')]
            for filename in direction_files:
                file_path = os.path.join(self.cache_dir, filename)
                os.remove(file_path)
            print(f"🗑️  文件缓存已清理 ({len(direction_files)} 个文件)")

def main():
    """测试函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  列出方向: python direction_manager.py list")
        print("  方向信息: python direction_manager.py info <effect_name>")
        print("  清理缓存: python direction_manager.py clear [--memory-only]")
        return
    
    try:
        manager = DirectionManager()
        
        command = sys.argv[1]
        
        if command == 'list':
            # 列出所有方向
            directions = manager.list_available_directions()
            print(f"📋 可用方向向量 ({len(directions)} 个):")
            for name in directions:
                print(f"   - {name}")
                
        elif command == 'info':
            # 显示方向信息
            if len(sys.argv) < 3:
                print("❌ 请指定效果名称")
                return
                
            effect_name = sys.argv[2]
            info = manager.get_direction_info(effect_name)
            
            if info:
                print(f"📊 方向向量信息: {effect_name}")
                for key, value in info.items():
                    print(f"   {key}: {value}")
            else:
                print(f"❌ 方向向量不存在: {effect_name}")
                
        elif command == 'clear':
            # 清理缓存
            memory_only = len(sys.argv) > 2 and sys.argv[2] == '--memory-only'
            manager.clear_cache(memory_only)
            
        else:
            print(f"❌ 未知命令: {command}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
