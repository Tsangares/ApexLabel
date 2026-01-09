#!/usr/bin/env python3
"""
Optimized Rendering System for SAM Annotator
Implements composite rendering and level-of-detail approaches for massive performance improvement
"""

import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import time
from typing import List, Dict, Tuple, Optional, Any
import threading
import queue


class OptimizedRenderer:
    """High-performance rendering system for mask overlays"""
    
    def __init__(self, canvas: tk.Canvas, canvas_width: int, canvas_height: int):
        self.canvas = canvas
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Composite overlay management
        self.composite_overlay = None
        self.composite_tk_image = None
        self.composite_dirty = True
        
        # Level-of-detail settings
        self.recent_segments_count = 3  # Show full detail for 3 most recent
        self.bbox_only_threshold = 10   # After 10 segments, show bbox only
        
        # Caching system
        self.overlay_cache = {}
        self.scale_cache = {}
        
        # Performance tracking
        self.render_times = []
        
        # Async rendering
        self.render_queue = queue.Queue()
        self.render_thread = None
        self.stop_rendering = False
        
    def update_segments(self, segments: List[Dict], image_shape: Tuple[int, int], 
                       image_scale: float, pan_offset: Tuple[int, int]):
        """Update all segments with optimized composite rendering"""
        start_time = time.perf_counter()
        
        # Clear existing segment drawings
        self.canvas.delete("segment")
        self.canvas.delete("overlay")
        
        if not segments:
            return
            
        # Strategy selection based on segment count
        segment_count = len(segments)
        
        if segment_count <= 5:
            # Few segments: Use individual overlays (acceptable performance)
            self._render_individual_overlays(segments, image_shape, image_scale, pan_offset)
        elif segment_count <= 20:
            # Medium segments: Use composite rendering
            self._render_composite_overlays(segments, image_shape, image_scale, pan_offset)
        else:
            # Many segments: Use level-of-detail rendering
            self._render_lod_overlays(segments, image_shape, image_scale, pan_offset)
        
        # Always draw click points and bounding boxes (lightweight)
        self._draw_segment_annotations(segments, image_scale, pan_offset)
        
        # Track performance
        elapsed = time.perf_counter() - start_time
        self.render_times.append(elapsed)
        if len(self.render_times) > 50:
            self.render_times.pop(0)
            
        avg_time = sum(self.render_times) / len(self.render_times)
        print(f"Render time: {elapsed*1000:.1f}ms (avg: {avg_time*1000:.1f}ms)")
    
    def _render_individual_overlays(self, segments: List[Dict], image_shape: Tuple[int, int],
                                  image_scale: float, pan_offset: Tuple[int, int]):
        """Render each segment individually (for small counts)"""
        h, w = image_shape
        
        for i, segment in enumerate(segments):
            mask = segment['mask']
            
            # Create individual overlay (handles None masks internally)
            overlay = self._create_segment_overlay(mask, i, h, w, image_scale)
            
            if overlay is not None:
                # Draw with pan offset
                self.canvas.create_image(
                    pan_offset[0], pan_offset[1],
                    anchor=tk.NW, image=overlay,
                    tags=f"overlay_segment_{i}"
                )
    
    def _render_composite_overlays(self, segments: List[Dict], image_shape: Tuple[int, int],
                                 image_scale: float, pan_offset: Tuple[int, int]):
        """Render all segments into single composite overlay"""
        h, w = image_shape
        
        # Create composite overlay
        composite_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Add all segments to composite
        for i, segment in enumerate(segments):
            mask = segment['mask']
            
            # Skip segments without masks (loaded annotations)
            if mask is None:
                continue
                
            color = self._get_segment_color(i)
            
            # Ensure mask dimensions match composite dimensions
            mask_h, mask_w = mask.shape
            if mask_h != h or mask_w != w:
                # Resize mask to match image dimensions
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)
            
            # Add segment to composite with alpha blending
            alpha = 120  # Semi-transparent
            composite_overlay[mask] = [color[0], color[1], color[2], alpha]
        
        # Resize composite once
        scaled_h = int(h * image_scale)
        scaled_w = int(w * image_scale)
        composite_resized = cv2.resize(composite_overlay, (scaled_w, scaled_h))
        
        # Convert to Tkinter image once
        composite_pil = Image.fromarray(composite_resized, "RGBA")
        composite_tk = ImageTk.PhotoImage(composite_pil)
        
        # Store reference to prevent garbage collection
        self.composite_tk_image = composite_tk
        
        # Draw composite overlay
        self.canvas.create_image(
            pan_offset[0], pan_offset[1],
            anchor=tk.NW, image=composite_tk,
            tags="composite_overlay"
        )
    
    def _render_lod_overlays(self, segments: List[Dict], image_shape: Tuple[int, int],
                           image_scale: float, pan_offset: Tuple[int, int]):
        """Level-of-detail rendering: Full detail for recent, bbox for old"""
        total_segments = len(segments)
        
        # Render full overlays for recent segments only
        recent_segments = segments[-self.recent_segments_count:]
        h, w = image_shape
        
        if recent_segments:
            # Create composite for recent segments
            composite_overlay = np.zeros((h, w, 4), dtype=np.uint8)
            
            for i, segment in enumerate(recent_segments):
                mask = segment['mask']
                
                # Skip segments without masks (loaded annotations)
                if mask is None:
                    continue
                    
                segment_idx = total_segments - self.recent_segments_count + i
                color = self._get_segment_color(segment_idx)
                
                # Ensure mask dimensions match composite dimensions
                mask_h, mask_w = mask.shape
                if mask_h != h or mask_w != w:
                    # Resize mask to match image dimensions
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = mask.astype(bool)
                
                alpha = 120
                composite_overlay[mask] = [color[0], color[1], color[2], alpha]
            
            # Resize and convert composite
            scaled_h = int(h * image_scale)
            scaled_w = int(w * image_scale)
            composite_resized = cv2.resize(composite_overlay, (scaled_w, scaled_h))
            
            composite_pil = Image.fromarray(composite_resized, "RGBA")
            composite_tk = ImageTk.PhotoImage(composite_pil)
            
            self.composite_tk_image = composite_tk
            
            # Draw composite overlay for recent segments
            self.canvas.create_image(
                pan_offset[0], pan_offset[1],
                anchor=tk.NW, image=composite_tk,
                tags="lod_composite_overlay"
            )
        
        # For older segments, only show bounding boxes (already handled in _draw_segment_annotations)
    
    def _create_segment_overlay(self, mask: np.ndarray, segment_id: int, 
                              orig_h: int, orig_w: int, scale: float) -> Optional[ImageTk.PhotoImage]:
        """Create individual segment overlay with caching"""
        # Return None for segments without masks (loaded annotations)
        if mask is None:
            return None
            
        # Cache key includes mask hash for uniqueness
        mask_hash = hash(mask.tobytes())
        cache_key = (segment_id, mask_hash, scale, orig_h, orig_w)
        
        if cache_key in self.overlay_cache:
            return self.overlay_cache[cache_key]
        
        # Create new overlay
        color = self._get_segment_color(segment_id)
        
        # Ensure mask dimensions match the expected image dimensions
        mask_h, mask_w = mask.shape
        if mask_h != orig_h or mask_w != orig_w:
            # Resize mask to match image dimensions
            mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
        
        # Create RGBA overlay
        overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        overlay[mask] = [color[0], color[1], color[2], 120]
        
        # Resize to scale
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        overlay_resized = cv2.resize(overlay, (new_w, new_h))
        
        # Convert to PIL and Tkinter
        overlay_pil = Image.fromarray(overlay_resized, "RGBA")
        overlay_tk = ImageTk.PhotoImage(overlay_pil)
        
        # Cache result (with size limit)
        if len(self.overlay_cache) < 100:  # Limit cache size
            self.overlay_cache[cache_key] = overlay_tk
        
        return overlay_tk
    
    def _draw_segment_annotations(self, segments: List[Dict], image_scale: float, 
                                pan_offset: Tuple[int, int]):
        """Draw click points and bounding boxes for all segments"""
        total_segments = len(segments)
        
        for i, segment in enumerate(segments):
            point = segment['point']
            bbox = segment['bbox']
            
            # Transform coordinates
            canvas_x = int(point[0] * image_scale) + pan_offset[0]
            canvas_y = int(point[1] * image_scale) + pan_offset[1]
            
            # Determine style based on segment type and recency
            is_loaded = segment.get('loaded', False)
            
            if is_loaded:
                # Loaded annotation - distinctive yellow/orange colors
                point_color = '#ffaa00'  # Orange
                point_outline = '#cc8800'
                bbox_color = '#ffaa00'
                bbox_width = 2
            elif i >= total_segments - self.recent_segments_count:
                # Recent segment - bright colors
                point_color = 'red'
                point_outline = 'darkred'
                bbox_color = 'red'
                bbox_width = 2
            else:
                # Older segment - muted colors
                point_color = 'gray'
                point_outline = 'darkgray'
                bbox_color = 'gray'
                bbox_width = 1
            
            # Draw click point
            self.canvas.create_oval(
                canvas_x - 3, canvas_y - 3, canvas_x + 3, canvas_y + 3,
                fill=point_color, outline=point_outline, width=2,
                tags=f"segment_point_{i}"
            )
            
            # Draw bounding box
            x, y, w, h = bbox
            bbox_x1 = int(x * image_scale) + pan_offset[0]
            bbox_y1 = int(y * image_scale) + pan_offset[1]
            bbox_x2 = int((x + w) * image_scale) + pan_offset[0]
            bbox_y2 = int((y + h) * image_scale) + pan_offset[1]
            
            self.canvas.create_rectangle(
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                outline=bbox_color, width=bbox_width,
                tags=f"segment_bbox_{i}"
            )
    
    def _get_segment_color(self, segment_id: int) -> Tuple[int, int, int]:
        """Get color for segment based on ID"""
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
            (255, 150, 100),  # Orange
            (150, 100, 255),  # Purple
        ]
        return colors[segment_id % len(colors)]
    
    def clear_cache(self):
        """Clear rendering caches"""
        self.overlay_cache.clear()
        self.scale_cache.clear()
        self.composite_tk_image = None
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get rendering performance statistics"""
        if not self.render_times:
            return {'avg_ms': 0, 'min_ms': 0, 'max_ms': 0, 'count': 0}
        
        times_ms = [t * 1000 for t in self.render_times]
        return {
            'avg_ms': sum(times_ms) / len(times_ms),
            'min_ms': min(times_ms),
            'max_ms': max(times_ms),
            'count': len(times_ms)
        }