#!/usr/bin/env python3
"""
Smart GPU Management System for SAM Annotation Tool
"""

import torch
import contextlib
import threading
import time
from typing import Optional, Dict, Any
import logging

class GPUManager:
    """Intelligent GPU allocation and management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.strategy = self._analyze_and_allocate()
        self.device_locks = {i: threading.Lock() for i in range(self.gpu_count)}
        self.memory_tracking = {i: {'peak': 0, 'current': 0} for i in range(self.gpu_count)}
        
    def _analyze_and_allocate(self) -> Dict[str, Any]:
        """Analyze GPUs and determine optimal allocation strategy"""
        if self.gpu_count == 0:
            return {'type': 'cpu', 'sam_device': 'cpu', 'yolo_device': 'cpu'}
        
        if self.gpu_count == 1:
            return {
                'type': 'single_gpu', 
                'sam_device': 'cuda:0', 
                'yolo_device': 'cuda:0',
                'reasoning': 'Single GPU system - shared usage'
            }
        
        # Multi-GPU analysis
        gpu_info = []
        for i in range(self.gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            
            gpu_info.append({
                'id': i,
                'name': props.name,
                'memory_gb': memory_gb,
                'multiprocessors': props.multi_processor_count,
                'compute_capability': f"{props.major}.{props.minor}"
            })
        
        # Strategy: SAM needs more memory, YOLO needs more compute
        sam_gpu = max(gpu_info, key=lambda x: x['memory_gb'])
        yolo_candidates = [gpu for gpu in gpu_info if gpu['id'] != sam_gpu['id']]
        yolo_gpu = max(yolo_candidates, key=lambda x: x['multiprocessors']) if yolo_candidates else sam_gpu
        
        strategy = {
            'type': 'dual_gpu_optimized',
            'sam_device': f"cuda:{sam_gpu['id']}",
            'yolo_device': f"cuda:{yolo_gpu['id']}",
            'sam_gpu_info': sam_gpu,
            'yolo_gpu_info': yolo_gpu,
            'reasoning': f"SAM→GPU{sam_gpu['id']}({sam_gpu['memory_gb']:.1f}GB), YOLO→GPU{yolo_gpu['id']}({yolo_gpu['multiprocessors']}MPs)"
        }
        
        self.logger.info(f"GPU Strategy: {strategy['reasoning']}")
        return strategy
    
    @contextlib.contextmanager
    def sam_context(self):
        """Context manager for SAM operations"""
        device = self.strategy['sam_device']
        if device.startswith('cuda'):
            gpu_id = int(device.split(':')[1])
            with self.device_locks[gpu_id]:
                original_device = torch.cuda.current_device() if torch.cuda.is_available() else None
                torch.cuda.set_device(gpu_id)
                self._update_memory_tracking(gpu_id)
                try:
                    yield device
                finally:
                    if original_device is not None and original_device != gpu_id:
                        torch.cuda.set_device(original_device)
        else:
            yield device
    
    @contextlib.contextmanager  
    def yolo_context(self):
        """Context manager for YOLO operations"""
        device = self.strategy['yolo_device']
        if device.startswith('cuda'):
            gpu_id = int(device.split(':')[1])
            with self.device_locks[gpu_id]:
                original_device = torch.cuda.current_device() if torch.cuda.is_available() else None
                torch.cuda.set_device(gpu_id)
                self._update_memory_tracking(gpu_id)
                try:
                    yield device
                finally:
                    if original_device is not None and original_device != gpu_id:
                        torch.cuda.set_device(original_device) 
        else:
            yield device
    
    def _update_memory_tracking(self, gpu_id: int):
        """Update memory usage tracking for a GPU"""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated(gpu_id) / (1024**2)  # MB
            self.memory_tracking[gpu_id]['current'] = current
            if current > self.memory_tracking[gpu_id]['peak']:
                self.memory_tracking[gpu_id]['peak'] = current
    
    def get_memory_stats(self) -> Dict[int, Dict[str, float]]:
        """Get current memory statistics for all GPUs"""
        stats = {}
        for gpu_id in range(self.gpu_count):
            self._update_memory_tracking(gpu_id)
            stats[gpu_id] = self.memory_tracking[gpu_id].copy()
            if torch.cuda.is_available():
                stats[gpu_id]['total'] = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**2)
                stats[gpu_id]['utilization'] = (stats[gpu_id]['current'] / stats[gpu_id]['total']) * 100
        return stats
    
    def cleanup_gpu_memory(self, device: Optional[str] = None):
        """Clean up GPU memory for specified device or all devices"""
        if not torch.cuda.is_available():
            return
            
        if device:
            if device.startswith('cuda'):
                gpu_id = int(device.split(':')[1])
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
        else:
            for gpu_id in range(self.gpu_count):
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
    
    def get_optimal_batch_size(self, task_type: str, base_size: int = 8) -> int:
        """Get optimal batch size based on GPU memory"""
        if task_type == 'yolo_training':
            device = self.strategy['yolo_device']
        else:
            device = self.strategy['sam_device']
        
        if not device.startswith('cuda'):
            return base_size
        
        gpu_id = int(device.split(':')[1])
        memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        
        # Heuristic based on available memory
        if memory_gb >= 12:
            return base_size * 2
        elif memory_gb >= 8:
            return base_size
        else:
            return max(1, base_size // 2)
    
    def print_status(self):
        """Print current GPU status"""
        print(f"\n🎯 GPU MANAGER STATUS")
        print(f"Strategy: {self.strategy['type']}")
        print(f"SAM Device: {self.strategy['sam_device']}")
        print(f"YOLO Device: {self.strategy['yolo_device']}")
        
        if self.gpu_count > 0:
            stats = self.get_memory_stats()
            for gpu_id, stat in stats.items():
                print(f"GPU {gpu_id}: {stat['current']:.0f}MB/{stat['total']:.0f}MB ({stat['utilization']:.1f}%)")

# Global GPU manager instance
gpu_manager = GPUManager()

# Convenience functions
def sam_device_context():
    """Get SAM device context"""
    return gpu_manager.sam_context()

def yolo_device_context():
    """Get YOLO device context"""  
    return gpu_manager.yolo_context()

def get_sam_device():
    """Get SAM device string"""
    return gpu_manager.strategy['sam_device']

def get_yolo_device():
    """Get YOLO device string"""
    return gpu_manager.strategy['yolo_device']

def cleanup_all_gpu_memory():
    """Clean up all GPU memory"""
    gpu_manager.cleanup_gpu_memory()

def get_gpu_stats():
    """Get GPU memory statistics"""
    return gpu_manager.get_memory_stats()