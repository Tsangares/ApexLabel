#!/usr/bin/env python3
"""
SAM Annotation Tool
Interactive GUI for annotating images with Segment Anything Model (SAM)
Features:
- Click to segment objects
- Scroll wheel to adjust threshold
- Real-time segmentation feedback
- Bounding box generation from segments
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import os
import json
from pathlib import Path
import time
import threading
from datetime import datetime
import shutil
from .gpu_manager import gpu_manager, sam_device_context, yolo_device_context
from .optimized_renderer import OptimizedRenderer
from .cyberpunk_theme import CyberpunkTheme
from .yolo_prediction_manager import YOLOPredictionManager
from .annotation_data_manager import AnnotationDataManager
import logging
import random
import sys
import argparse
from typing import Optional, Tuple, List, Dict, Any

# Import project configuration
try:
    from config import ProjectConfig
except ImportError:
    # Fallback for standalone usage
    ProjectConfig = None

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Error: segment_anything not installed. Please install with: pip install segment-anything")
    exit(1)

class SAMAnnotator:
    def __init__(self, root, initial_directory=None, config=None):
        self.root = root

        # Load configuration
        self.config = config
        if self.config is None and ProjectConfig is not None:
            # Try to load from default location
            try:
                self.config = ProjectConfig.from_yaml("config/default_config.yaml")
            except FileNotFoundError:
                self.config = ProjectConfig()

        # Validate configuration has class names
        if self.config and self.config.class_names:
            self.default_label = self.config.class_names[0]
        else:
            self.default_label = ""  # User must set via UI or config

        # Initial title (will be updated with HITS counter)
        self.update_window_title()
        self.root.geometry("1600x1000")

        # Initialize cyberpunk theme
        self.theme = CyberpunkTheme(root)
        self.colors = self.theme.get_cyber_colors()
        self.symbols = self.theme.get_cyber_symbols()

        # Configure root window
        self.root.configure(bg=self.colors['bg_primary'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', self.colors['neon_cyan'])

        # Directory selection - must happen before UI setup
        self.root_directory = self.select_annotation_directory(initial_directory)
        if not self.root_directory:
            self.root.quit()
            return

        # SAM model setup
        self.sam_model = None
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Image and annotation data
        self.current_image = None
        self.current_image_path = None
        self.original_image = None
        self.canvas_image = None
        self.image_scale = 1.0
        self.canvas_width = 900
        self.canvas_height = 700

        # Image management
        self.image_list = []
        self.current_image_index = 0
        self.input_directory = self.root_directory  # Images are in the selected root directory

        # Initialize managers with config
        self.yolo_manager = YOLOPredictionManager(config=self.config)
        self.data_manager = AnnotationDataManager()

        # Prediction toggle system
        self.prediction_toggle_enabled = False

        # Zoom and pan functionality
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.space_pressed = False

        # Annotation settings - use config class name or empty for user to set
        self.current_label = self.default_label
        self.annotation_data = {}  # Store all annotations by image path
        self.skip_annotated = False  # Option to skip already annotated images
        
        # Segmentation parameters
        self.threshold = 0.5
        self.current_masks = []
        self.current_segments = []
        self.annotations = []
        
        # UI state
        self.drawing_bbox = False
        self.bbox_start = None
        self.temp_bbox = None
        
        # Annotation mode toggle
        self.annotation_mode = "SAM"  # "SAM" or "MANUAL"
        
        # Prevent rapid segment removal (debounce)
        self.last_removal_time = 0
        self.removal_cooldown = 0.3  # 300ms cooldown between removals
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.setup_ui()
        self.setup_window_close_handler()  # Ensure proper cleanup on close
        self.create_backup_directory()  # Create backup system
        self.create_startup_backup()  # Backup existing annotations on startup
        self.load_sam_model()
        self.load_image_list()
        self.auto_load_first_image()
    
    def update_window_title(self):
        """Update window title with current HITS (annotation count)"""
        if hasattr(self, 'annotation_data') and self.annotation_data:
            total_hits = sum(len(annotations) for annotations in self.annotation_data.values())
            self.root.title(f"◢◣ SAM NEURAL INTERFACE ◤◥ - HITS: {total_hits}")
        else:
            self.root.title("◢◣ SAM NEURAL INTERFACE ◤◥ - HITS: 0")
    
    def setup_window_close_handler(self):
        """Ensure the on_closing method is called when window is closed"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_backup_directory(self):
        """Create .annotations backup directory if it doesn't exist"""
        self.backup_dir = Path(self.root_directory) / ".annotations"
        self.backup_dir.mkdir(exist_ok=True)
        self.log_status(f"◢ BACKUP SYSTEM: Ready at {self.backup_dir} ◣")
    
    def create_backup(self, reason="manual"):
        """Create a timestamped backup of the current annotation file"""
        try:
            # Path to the main annotation file
            main_file = Path(self.root_directory) / "all_annotations.json"
            
            if not main_file.exists():
                self.log_status("◢ BACKUP: No annotation file to backup yet ◣")
                return
            
            # Create timestamped backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"annotations_backup_{timestamp}_{reason}.json"
            backup_path = self.backup_dir / backup_filename
            
            # Copy the file
            shutil.copy2(main_file, backup_path)
            
            # Get file size for logging
            file_size = backup_path.stat().st_size
            self.log_status(f"◢ BACKUP: Created {backup_filename} ({file_size} bytes) ◣")
            
            # Clean up old backups (keep last 20)
            self.cleanup_old_backups()
            
        except Exception as e:
            self.log_status(f"◢ BACKUP ERROR: {str(e)} ◣")
    
    def cleanup_old_backups(self):
        """Keep only the last 20 backup files"""
        try:
            backup_files = list(self.backup_dir.glob("annotations_backup_*.json"))
            if len(backup_files) > 20:
                # Sort by creation time (oldest first)
                backup_files.sort(key=lambda x: x.stat().st_ctime)
                
                # Remove oldest files
                files_to_remove = backup_files[:-20]
                for old_file in files_to_remove:
                    old_file.unlink()
                    
                self.log_status(f"◢ BACKUP CLEANUP: Removed {len(files_to_remove)} old backups ◣")
                
        except Exception as e:
            self.log_status(f"◢ BACKUP CLEANUP ERROR: {str(e)} ◣")
    
    def create_startup_backup(self):
        """Create a backup when the program starts (if annotation file exists)"""
        try:
            main_file = Path(self.root_directory) / "all_annotations.json"
            if main_file.exists():
                self.create_backup("on_startup")
                self.log_status("◢ STARTUP BACKUP: Created backup of existing annotations ◣")
        except Exception as e:
            self.log_status(f"◢ STARTUP BACKUP ERROR: {str(e)} ◣")
    
    def select_annotation_directory(self, initial_directory=None):
        """Show directory picker for selecting annotation workspace or use provided directory"""
        
        # If directory provided via command line, use it directly
        if initial_directory:
            # Resolve path relative to the current working directory where user ran the command
            if os.path.isabs(initial_directory):
                abs_path = initial_directory
            else:
                abs_path = os.path.join(os.getcwd(), initial_directory)
            
            abs_path = os.path.normpath(abs_path)  # Clean up the path
            
            if os.path.isdir(abs_path):
                print(f"Using provided annotation directory: {abs_path}")
                return abs_path
            else:
                print(f"Provided directory does not exist: {abs_path}")
                print("Falling back to directory selection dialog...")
        
        # Hide the main window during directory selection
        self.root.withdraw()
        
        # Show directory selection dialog
        selected_dir = filedialog.askdirectory(
            title="Select Annotation Directory (contains images and will store annotations)",
            initialdir=os.getcwd()
        )
        
        # Show the main window again
        self.root.deiconify()
        
        if selected_dir:
            print(f"Selected annotation directory: {selected_dir}")
            return selected_dir
        else:
            print("No directory selected. Exiting.")
            return None
    
    def toggle_skip_annotated(self):
        """Toggle unprocessed-only navigation mode"""
        self.skip_annotated = self.skip_var.get()
        if self.skip_annotated:
            self.log_status("◢ FILTER ON: Only showing unprocessed images ◣")
        else:
            self.log_status("◢ FILTER OFF: Showing all images ◣")
            # Clear completion screen if it's showing
            self.canvas.delete("completion")
            # Reload current image to restore normal display
            if hasattr(self, 'current_image_path') and self.current_image_path:
                self.display_image()
    
    def is_image_annotated(self, image_path):
        """Check if an image has been processed (has annotations or has been reviewed)"""
        return str(image_path) in self.annotation_data
    
    def find_next_unannotated_image(self, start_index, direction=1):
        """Find the next unannotated image"""
        if not self.skip_annotated or not self.image_list:
            return None
            
        checked = 0
        current_index = start_index
        
        while checked < len(self.image_list):
            current_index = (current_index + direction) % len(self.image_list)
            
            if not self.is_image_annotated(self.image_list[current_index]):
                return current_index
                
            checked += 1
            
        return None  # All images are annotated
    
    def get_mode_button_text(self):
        """Get the text for the mode toggle button with clear selection indicators"""
        if self.annotation_mode == "SAM":
            return "▶ SAM ◀     ⭘ MANUAL"
        else:
            return "⭘ SAM     ▶ MANUAL ◀"
    
    def get_mode_status_text(self):
        """Get the status text for the current mode"""
        if self.annotation_mode == "SAM":
            return "Click = AI Segmentation | Scroll = Threshold"
        else:
            return "Click+Drag = Manual Bounding Box"
    
    def toggle_annotation_mode(self):
        """Toggle between SAM and Manual annotation modes"""
        if self.annotation_mode == "SAM":
            self.annotation_mode = "MANUAL"
            self.log_status("◢ SWITCHED TO MANUAL BOXING MODE ◣")
        else:
            self.annotation_mode = "SAM"
            self.log_status("◢ SWITCHED TO SAM AI MODE ◣")
        
        # Update button text and status
        self.mode_button.config(text=self.get_mode_button_text())
        self.mode_status_label.config(text=self.get_mode_status_text())
        
        # Change button color to indicate mode
        if self.annotation_mode == "SAM":
            self.mode_button.config(bg=self.colors['neon_cyan'])
        else:
            self.mode_button.config(bg=self.colors['neon_orange'])
    
    def update_model_dropdown(self):
        """Update the model selection dropdowns"""
        try:
            models = self.yolo_manager.get_available_models()
            model_names = [f"{model['name']} ({model['size_mb']:.1f}MB)" for model in models]
            
            # Update prediction tab dropdown
            self.model_dropdown['values'] = model_names
            
            # Update training tab dropdown if it exists
            if hasattr(self, 'model_dropdown_training'):
                self.model_dropdown_training['values'] = model_names
            
            # Set current model as selected in both dropdowns
            current_model = self.yolo_manager.model_path
            selected_index = 0
            for i, model in enumerate(models):
                if model['path'] == current_model:
                    selected_index = i
                    break
            
            self.model_dropdown.current(selected_index)
            if hasattr(self, 'model_dropdown_training'):
                self.model_dropdown_training.current(selected_index)
                    
        except Exception as e:
            self.log_status(f"◢ ERROR updating model dropdown: {str(e)} ◣")
    
    def on_model_selection_changed(self, event=None):
        """Handle model selection change"""
        try:
            selected_index = self.model_dropdown.current()
            if selected_index >= 0:
                models = self.yolo_manager.get_available_models()
                if selected_index < len(models):
                    selected_model = models[selected_index]
                    self.yolo_manager.set_model_path(selected_model['path'])
                    self.log_status(f"◢ SWITCHED MODEL: {selected_model['name']} ◣")
                    # Update dropdown display
                    self.update_model_dropdown()
        except Exception as e:
            self.log_status(f"◢ ERROR switching model: {str(e)} ◣")
    
    def update_dataset_status(self):
        """Update dataset status in training tab"""
        try:
            # Count available annotations
            annotation_count = len([path for path, annotations in self.annotation_data.items() 
                                  if annotations and len(annotations) > 0])
            
            total_annotations = sum(len(annotations) for annotations in self.annotation_data.values() 
                                  if annotations)
            
            status_text = f"◢ DATASET: {annotation_count} images, {total_annotations} annotations ◣"
            
            if hasattr(self, 'dataset_status_label'):
                self.dataset_status_label.config(text=status_text)
                
        except Exception as e:
            if hasattr(self, 'dataset_status_label'):
                self.dataset_status_label.config(text=f"◢ DATASET: Error calculating ◣")
    
    def toggle_prediction_mode(self):
        """Toggle YOLO prediction assist mode"""
        self.prediction_toggle_enabled = self.prediction_var.get()
        self.yolo_manager.set_enabled(self.prediction_toggle_enabled)
        
        if self.prediction_toggle_enabled:
            self.log_status("🤖 YOLO Prediction Assist: ENABLED")
            # Start background prediction for current and next few images
            self.start_background_prediction()
        else:
            self.log_status("🤖 YOLO Prediction Assist: DISABLED")
        
        # Update display if we have a current image
        if hasattr(self, 'current_image_path') and self.current_image_path:
            self.display_image()
    
    def train_yolo_from_annotations(self):
        """Train YOLO model using current annotations"""
        try:
            # Count available annotations
            annotation_count = len([path for path, annotations in self.annotation_data.items() 
                                  if annotations and len(annotations) > 0])
            
            if annotation_count < 10:
                messagebox.showwarning("Insufficient Data", 
                    f"Only {annotation_count} annotated images found. Need at least 10 for training.")
                return
            
            # Ask user for training parameters
            dialog = TrainingParametersDialog(self.root, annotation_count)
            if not dialog.result:
                return
            
            self.log_status(f"◢ STARTING YOLO TRAINING: {annotation_count} images ◣")
            
            # Export annotations to YOLO format first
            yolo_dataset_path = os.path.join(self.config.output_dir if self.config else "./output", "training_dataset")
            export_stats = self.yolo_manager.export_yolo_dataset(self.annotation_data, yolo_dataset_path)
            
            if not export_stats:
                self.log_status("◢ ERROR: Failed to export dataset ◣")
                return
            
            self.log_status(f"◢ DATASET EXPORTED: {export_stats['train']} train, {export_stats['val']} val ◣")
            
            # Start training in background
            dataset_yaml = f"{yolo_dataset_path}/dataset.yaml"
            
            def training_progress(message):
                self.log_status(f"◢ TRAINING: {message} ◣")
            
            def training_complete(message):
                self.log_status(f"◢ TRAINING COMPLETE: {message} ◣")
                # Update model dropdown to include new model
                self.update_model_dropdown()
                messagebox.showinfo("Training Complete", message)
            
            # Start training with parameters from dialog
            self.yolo_manager.train_model(
                dataset_yaml_path=dataset_yaml,
                epochs=dialog.epochs,
                image_size=dialog.image_size,
                batch_size=dialog.batch_size,
                progress_callback=training_progress
            )
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.log_status(f"◢ {error_msg} ◣")
            messagebox.showerror("Training Error", error_msg)
    
    def start_background_prediction(self):
        """Start background prediction for images around current index"""
        if not self.yolo_manager.is_model_available():
            self.log_status("⚠️ No YOLO model found. Train a model first!")
            return
        
        # Predict for current image and next 5 images
        start_idx = max(0, self.current_image_index)
        end_idx = min(len(self.image_list), start_idx + 6)
        
        images_to_predict = []
        for i in range(start_idx, end_idx):
            if i < len(self.image_list):
                images_to_predict.append(self.image_list[i])
        
        if images_to_predict:
            self.yolo_manager.cache_predictions_batch(
                images_to_predict,
                complete_callback=self.on_background_prediction_complete
            )
    
    def on_background_prediction_complete(self, message):
        """Handle completion of background prediction"""
        self.log_status(f"🤖 {message}")
        self.update_cache_status()
        # Refresh display if we have predicted annotations for current image
        if hasattr(self, 'current_image_path') and self.current_image_path:
            self.display_image()
    
    def predict_current_image(self):
        """Predict current image only"""
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            self.log_status("❌ No image loaded")
            return
        
        if not self.yolo_manager.is_model_available():
            self.log_status("⚠️ No YOLO model found. Train a model first!")
            return
        
        self.log_status("🎯 Predicting current image...")
        self.yolo_manager.predict_current_only(
            self.current_image_path,
            result_callback=self.on_current_prediction_complete
        )
    
    def on_current_prediction_complete(self, predictions):
        """Handle completion of current image prediction"""
        count = len(predictions)
        self.log_status(f"🎯 Current prediction completed: {count} detections found")
        self.update_cache_status()
        self.display_image()  # Refresh display to show predictions
    
    def predict_batch_images(self):
        """Predict a batch of images (next 20)"""
        if not self.yolo_manager.is_model_available():
            self.log_status("⚠️ No YOLO model found. Train a model first!")
            return
        
        # Get next 20 images starting from current
        start_idx = self.current_image_index
        end_idx = min(len(self.image_list), start_idx + 20)
        
        images_to_predict = self.image_list[start_idx:end_idx]
        
        if not images_to_predict:
            self.log_status("❌ No images to predict")
            return
        
        self.log_status(f"📊 Starting batch prediction for {len(images_to_predict)} images...")
        self.show_progress_bar()
        
        self.yolo_manager.cache_predictions_batch(
            images_to_predict,
            progress_callback=self.on_batch_progress,
            complete_callback=self.on_batch_prediction_complete
        )
    
    def predict_all_images(self):
        """Predict all images in dataset"""
        if not self.yolo_manager.is_model_available():
            self.log_status("⚠️ No YOLO model found. Train a model first!")
            return
        
        if not self.image_list:
            self.log_status("❌ No images loaded")
            return
        
        total_images = len(self.image_list)
        self.log_status(f"🚀 Starting prediction for ALL {total_images} images...")
        self.show_progress_bar()
        
        self.yolo_manager.predict_all_images(
            self.image_list,
            progress_callback=self.on_predict_all_progress,
            complete_callback=self.on_predict_all_complete
        )
    
    def clear_prediction_cache(self):
        """Clear all cached predictions"""
        self.yolo_manager.clear_cache()
        self.log_status("🗑️ Prediction cache cleared")
        self.update_cache_status()
        self.display_image()  # Refresh display
    
    def show_progress_bar(self):
        """Show the progress bar"""
        self.prediction_progress_frame.pack(fill=tk.X, pady=5)
        self.prediction_progress['value'] = 0
        self.progress_label.config(text="◢ INITIALIZING ◣")
    
    def hide_progress_bar(self):
        """Hide the progress bar"""
        self.prediction_progress_frame.pack_forget()
    
    def on_batch_progress(self, current, total):
        """Handle batch prediction progress"""
        progress_percent = (current / total) * 100
        self.prediction_progress['value'] = progress_percent
        self.progress_label.config(text=f"◢ PROCESSING: {current}/{total} ({progress_percent:.1f}%) ◣")
        self.root.update_idletasks()
    
    def on_batch_prediction_complete(self, message):
        """Handle batch prediction completion"""
        self.log_status(f"📊 {message}")
        self.hide_progress_bar()
        self.update_cache_status()
        self.display_image()
    
    def on_predict_all_progress(self, current, total, successful, failed):
        """Handle predict all progress"""
        progress_percent = (current / total) * 100
        self.prediction_progress['value'] = progress_percent
        self.progress_label.config(text=f"◢ ALL: {current}/{total} | ✓{successful} ✗{failed} ◣")
        self.root.update_idletasks()
    
    def on_predict_all_complete(self, message):
        """Handle predict all completion"""
        self.log_status(f"🚀 {message}")
        self.hide_progress_bar()
        self.update_cache_status()
        self.display_image()
    
    def get_cached_predictions(self, image_path):
        """Get cached predictions for an image"""
        return self.yolo_manager.get_cached_predictions(image_path)
    
    def update_cache_status(self):
        """Update the cache status label"""
        if hasattr(self, 'cache_status_label'):
            stats = self.yolo_manager.get_cache_stats()
            cache_count = stats['count']
            self.cache_status_label.config(text=f"◢ CACHE: {cache_count} predictions stored ◣")
        
    def setup_ui(self):
        """Setup the cyberpunk-themed user interface"""
        # Main container with cyberpunk styling
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left control panel with glowing border - increased width for better layout
        left_panel = self.theme.create_glowing_frame(main_frame, "NEURAL CONTROL MATRIX", width=500)
        left_panel.master.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.master.pack_propagate(False)
        
        # Create notebook for tabbed interface
        from tkinter import ttk as tkinter_ttk
        self.notebook = tkinter_ttk.Notebook(left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Tab 1: Annotation Controls
        annotation_tab = tk.Frame(self.notebook, bg=self.colors['bg_panel'])
        self.notebook.add(annotation_tab, text="📝 ANNOTATION")
        
        # Tab 2: Prediction Controls  
        prediction_tab = tk.Frame(self.notebook, bg=self.colors['bg_panel'])
        self.notebook.add(prediction_tab, text="🎯 PREDICTION")
        
        # Tab 3: Training Controls
        training_tab = tk.Frame(self.notebook, bg=self.colors['bg_panel'])
        self.notebook.add(training_tab, text="🏋️ TRAINING")
        
        # === ANNOTATION TAB CONTENTS ===
        
        # Target Classification Section
        label_section = tk.Frame(annotation_tab, bg=self.colors['bg_panel'])
        label_section.pack(fill=tk.X, pady=5)
        
        tk.Label(label_section, 
            text=f"{self.symbols['target']} TARGET CLASSIFICATION", 
            bg=self.colors['bg_panel'], 
            fg=self.colors['neon_orange'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        self.label_var = tk.StringVar(value=self.current_label)
        self.label_entry = tk.Entry(label_section,
            textvariable=self.label_var,
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['neon_cyan'],
            relief='solid',
            bd=1,
            font=('Consolas', 12)
        )
        self.label_entry.pack(fill=tk.X, pady=5)
        self.label_entry.bind('<Return>', self.on_label_change)
        
        # Annotation Mode Toggle Section
        mode_section = tk.Frame(annotation_tab, bg=self.colors['bg_panel'])
        mode_section.pack(fill=tk.X, pady=10)
        
        tk.Label(mode_section,
            text=f"{self.symbols['lightning']} ANNOTATION MODE",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_cyan'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        # Mode toggle button with visual indicator
        self.mode_button = tk.Button(mode_section,
            text=self.get_mode_button_text(),
            command=self.toggle_annotation_mode,
            bg=self.colors['neon_cyan'],
            fg=self.colors['bg_primary'],
            activebackground=self.colors['neon_orange'],
            activeforeground=self.colors['bg_primary'],
            font=('Consolas', 12, 'bold'),
            relief='solid',
            bd=2,
            cursor='hand2'
        )
        self.mode_button.pack(fill=tk.X, pady=5)
        
        # Mode status indicator
        self.mode_status_label = tk.Label(mode_section,
            text=self.get_mode_status_text(),
            bg=self.colors['bg_panel'],
            fg=self.colors['text_secondary'],
            font=('Consolas', 12)
        )
        self.mode_status_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Image Navigation Section
        nav_section = tk.Frame(annotation_tab, bg=self.colors['bg_panel'])
        nav_section.pack(fill=tk.X, pady=10)
        
        tk.Label(nav_section,
            text=f"{self.symbols['scanner']} DATA STREAM NAVIGATION",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_green'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        # Navigation buttons with cyber styling
        nav_buttons_frame = tk.Frame(nav_section, bg=self.colors['bg_panel'])
        nav_buttons_frame.pack(fill=tk.X, pady=5)
        
        self.prev_button = self.theme.create_cyber_button(nav_buttons_frame, 
            f"{self.symbols['arrow_left']} PREV", self.previous_image, 'neon_green')
        self.prev_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.next_button = self.theme.create_cyber_button(nav_buttons_frame,
            f"NEXT {self.symbols['arrow_right']}", self.next_image, 'neon_green')
        self.next_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Skip annotated images option
        skip_frame = tk.Frame(nav_section, bg=self.colors['bg_panel'])
        skip_frame.pack(fill=tk.X, pady=5)
        
        self.skip_var = tk.BooleanVar()
        self.skip_checkbox = tk.Checkbutton(skip_frame,
            text="⚡ Only show unprocessed images",
            variable=self.skip_var,
            command=self.toggle_skip_annotated,
            bg=self.colors['bg_panel'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['bg_secondary'],
            activebackground=self.colors['bg_panel'],
            activeforeground=self.colors['neon_cyan'],
            font=('Consolas', 11)
        )
        self.skip_checkbox.pack(anchor=tk.W)
        
        self.image_info_label = tk.Label(nav_section,
            text="◢ NO DATA STREAMS LOADED ◣",
            bg=self.colors['bg_panel'],
            fg=self.colors['text_secondary'],
            font=('Consolas', 11)
        )
        self.image_info_label.pack(fill=tk.X, pady=5)
        
        # Random and manual navigation
        random_frame = tk.Frame(nav_section, bg=self.colors['bg_panel'])
        random_frame.pack(fill=tk.X, pady=5)
        
        self.random_button = self.theme.create_cyber_button(random_frame,
            f"{self.symbols['diamond']} RANDOM", self.jump_to_random_image, 'neon_purple')
        self.random_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.load_button = self.theme.create_cyber_button(random_frame,
            f"{self.symbols['crosshair']} CUSTOM", self.load_custom_image, 'neon_purple')
        self.load_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Copy filename button
        copy_frame = tk.Frame(nav_section, bg=self.colors['bg_panel'])
        copy_frame.pack(fill=tk.X, pady=5)
        
        self.copy_filename_button = self.theme.create_cyber_button(copy_frame,
            f"{self.symbols['data']} COPY FILENAME", self.copy_current_filename, 'neon_pink')
        self.copy_filename_button.pack(fill=tk.X)
        
        # Neural Network Controls Section
        neural_section = tk.Frame(annotation_tab, bg=self.colors['bg_panel'])
        neural_section.pack(fill=tk.X, pady=10)
        
        tk.Label(neural_section,
            text=f"{self.symbols['neural']} NEURAL SEGMENTATION",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_cyan'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        # Threshold control with cyber styling
        threshold_frame = tk.Frame(neural_section, bg=self.colors['bg_panel'])
        threshold_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(threshold_frame,
            text=f"{self.symbols['lightning']} NEURAL THRESHOLD:",
            bg=self.colors['bg_panel'],
            fg=self.colors['text_primary'],
            font=('Consolas', 11)
        ).pack(anchor=tk.W)
        
        self.threshold_var = tk.DoubleVar(value=self.threshold)
        self.threshold_scale = tk.Scale(threshold_frame,
            from_=0.0, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self.on_threshold_change,
            bg=self.colors['bg_secondary'],
            fg=self.colors['neon_cyan'],
            activebackground=self.colors['neon_cyan'],
            highlightbackground=self.colors['bg_panel'],
            troughcolor=self.colors['bg_primary'],
            font=('Consolas', 12)
        )
        self.threshold_scale.pack(fill=tk.X, pady=2)
        
        self.threshold_label = tk.Label(threshold_frame,
            text=f"◢ PRECISION: {self.threshold:.3f} ◣",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_cyan'],
            font=('Consolas', 11, 'bold')
        )
        self.threshold_label.pack(anchor=tk.W)
        
        # Zoom controls
        zoom_frame = tk.Frame(neural_section, bg=self.colors['bg_panel'])
        zoom_frame.pack(fill=tk.X, pady=5)
        
        self.zoom_label = tk.Label(zoom_frame,
            text="◎ MAGNIFICATION: 1.0x",
            bg=self.colors['bg_panel'],
            fg=self.colors['text_secondary'],
            font=('Consolas', 11)
        )
        self.zoom_label.pack(side=tk.LEFT)
        
        fit_button = self.theme.create_cyber_button(zoom_frame, "FIT", self.reset_view_to_fit, 'neon_blue')
        fit_button.pack(side=tk.RIGHT)
        
        # Neural Operations
        ops_frame = tk.Frame(neural_section, bg=self.colors['bg_panel'])
        ops_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(ops_frame,
            text=f"{self.symbols['atom']} NEURAL OPERATIONS:",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_orange'],
            font=('Consolas', 11, 'bold')
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # Action buttons with cyber styling
        buttons_grid = tk.Frame(ops_frame, bg=self.colors['bg_panel'])
        buttons_grid.pack(fill=tk.X)
        
        # Row 1
        row1 = tk.Frame(buttons_grid, bg=self.colors['bg_panel'])
        row1.pack(fill=tk.X, pady=2)
        
        self.undo_button = self.theme.create_cyber_button(row1, "UNDO", self.undo_last_segment, 'neon_pink')
        self.undo_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.predict_button = self.theme.create_cyber_button(row1, "AUTO-SCAN", self.sam_auto_predict, 'neon_green')
        self.predict_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Row 2
        row2 = tk.Frame(buttons_grid, bg=self.colors['bg_panel'])
        row2.pack(fill=tk.X, pady=2)
        
        self.clear_button = self.theme.create_cyber_button(row2, f"{self.symbols['skull']} PURGE", self.clear_segments, 'error')
        self.clear_button.pack(fill=tk.X)
        
        # Instructions with cyber styling
        instructions_frame = tk.Frame(neural_section, bg=self.colors['bg_accent'], relief='solid', bd=1)
        instructions_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(instructions_frame,
            text="◢◣ NEURAL INTERFACE COMMANDS ◤◥",
            bg=self.colors['bg_accent'],
            fg=self.colors['neon_cyan'],
            font=('Consolas', 12, 'bold')
        ).pack()
        
        instructions_text = "◆ CLICK: Target Lock\n◆ SHIFT+CLICK: Eliminate\n◆ ENTER: Toggle SAM/Manual\n◆ WHEEL: Zoom Matrix\n◆ CTRL+WHEEL: Neural Threshold\n◆ SPACE+DRAG: Navigate\n◆ CTRL+F: Auto-Fit"
        tk.Label(instructions_frame,
            text=instructions_text,
            bg=self.colors['bg_accent'],
            fg=self.colors['text_secondary'],
            font=('Consolas', 12),
            justify=tk.LEFT
        ).pack(pady=5)
        
        # === PREDICTION TAB CONTENTS ===
        
        # Prediction Toggle Section
        prediction_section = tk.Frame(prediction_tab, bg=self.colors['bg_panel'])
        prediction_section.pack(fill=tk.X, pady=10)
        
        tk.Label(prediction_section,
            text=f"{self.symbols['neural']} YOLO PREDICTION ASSIST",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_orange'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        # Model selection
        model_frame = tk.Frame(prediction_section, bg=self.colors['bg_panel'])
        model_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(model_frame,
            text="📦 MODEL:",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_orange'],
            font=('Consolas', 10, 'bold')
        ).pack(side=tk.LEFT)
        
        self.model_var = tk.StringVar()
        from tkinter import ttk as tkinter_ttk
        self.model_dropdown = tkinter_ttk.Combobox(model_frame,
            textvariable=self.model_var,
            state="readonly",
            font=('Consolas', 9),
            width=35
        )
        self.model_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_selection_changed)
        
        # Populate model dropdown
        self.update_model_dropdown()
        
        # Prediction toggle checkbox
        pred_toggle_frame = tk.Frame(prediction_section, bg=self.colors['bg_panel'])
        pred_toggle_frame.pack(fill=tk.X, pady=5)
        
        self.prediction_var = tk.BooleanVar()
        self.prediction_checkbox = tk.Checkbutton(pred_toggle_frame,
            text="🤖 Auto-predict bounding boxes on image load",
            variable=self.prediction_var,
            command=self.toggle_prediction_mode,
            bg=self.colors['bg_panel'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['bg_secondary'],
            activebackground=self.colors['bg_panel'],
            activeforeground=self.colors['neon_orange'],
            font=('Consolas', 11)
        )
        self.prediction_checkbox.pack(anchor=tk.W)
        
        # Prediction control buttons
        pred_buttons_frame = tk.Frame(prediction_section, bg=self.colors['bg_panel'])
        pred_buttons_frame.pack(fill=tk.X, pady=5)
        
        # Row 1: Current and Batch prediction
        row1 = tk.Frame(pred_buttons_frame, bg=self.colors['bg_panel'])
        row1.pack(fill=tk.X, pady=2)
        
        self.predict_current_button = self.theme.create_cyber_button(row1, 
            "🎯 PREDICT CURRENT", self.predict_current_image, 'neon_cyan')
        self.predict_current_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.predict_batch_button = self.theme.create_cyber_button(row1, 
            "📊 PREDICT BATCH", self.predict_batch_images, 'neon_purple')
        self.predict_batch_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Row 2: Predict All and Clear Cache
        row2 = tk.Frame(pred_buttons_frame, bg=self.colors['bg_panel'])
        row2.pack(fill=tk.X, pady=2)
        
        self.predict_all_button = self.theme.create_cyber_button(row2, 
            "🚀 PREDICT ALL", self.predict_all_images, 'neon_green')
        self.predict_all_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.clear_cache_button = self.theme.create_cyber_button(row2, 
            "🗑️ CLEAR CACHE", self.clear_prediction_cache, 'error')
        self.clear_cache_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Training/Export buttons moved to Training tab for better organization
        
        # Progress bar for batch operations
        self.prediction_progress_frame = tk.Frame(prediction_section, bg=self.colors['bg_panel'])
        
        tk.Label(self.prediction_progress_frame,
            text="⚡ PREDICTION PROGRESS:",
            bg=self.colors['bg_panel'],
            fg=self.colors['text_primary'],
            font=('Consolas', 10)
        ).pack(anchor=tk.W)
        
        from tkinter import ttk
        self.prediction_progress = ttk.Progressbar(
            self.prediction_progress_frame,
            mode='determinate',
            length=300,
            style='TProgressbar'
        )
        self.prediction_progress.pack(fill=tk.X, pady=2)
        
        self.progress_label = tk.Label(self.prediction_progress_frame,
            text="◢ READY TO PREDICT ◣",
            bg=self.colors['bg_panel'],
            fg=self.colors['text_secondary'],
            font=('Consolas', 10)
        )
        self.progress_label.pack(anchor=tk.W)
        
        # Cache status label
        self.cache_status_label = tk.Label(prediction_section,
            text="◢ CACHE: 0 predictions stored ◣",
            bg=self.colors['bg_panel'],
            fg=self.colors['text_secondary'],
            font=('Consolas', 11)
        )
        self.cache_status_label.pack(anchor=tk.W, pady=(0, 5))
        
        # === TRAINING TAB CONTENTS ===
        
        # Training Section
        training_section = tk.Frame(training_tab, bg=self.colors['bg_panel'])
        training_section.pack(fill=tk.X, pady=10)
        
        tk.Label(training_section,
            text=f"{self.symbols['neural']} MODEL TRAINING & EXPORT",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_orange'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        # Dataset status
        self.dataset_status_label = tk.Label(training_section,
            text="◢ DATASET: Calculating... ◣",
            bg=self.colors['bg_panel'],
            fg=self.colors['text_secondary'],
            font=('Consolas', 10)
        )
        self.dataset_status_label.pack(anchor=tk.W, pady=5)
        
        # Training controls
        train_controls = tk.Frame(training_section, bg=self.colors['bg_panel'])
        train_controls.pack(fill=tk.X, pady=10)
        
        # Export and Train buttons (larger, more spacious)
        self.export_dataset_button_training = self.theme.create_cyber_button(train_controls, 
            "📦 EXPORT DATASET", self.export_yolo_dataset, 'neon_purple')
        self.export_dataset_button_training.pack(fill=tk.X, pady=2)
        
        self.train_model_button_training = self.theme.create_cyber_button(train_controls, 
            "🏋️ TRAIN NEW MODEL", self.train_yolo_from_annotations, 'warning')
        self.train_model_button_training.pack(fill=tk.X, pady=2)
        
        # Model management
        model_mgmt_frame = tk.Frame(training_section, bg=self.colors['bg_panel'])
        model_mgmt_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(model_mgmt_frame,
            text="🔧 MODEL MANAGEMENT",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_cyan'],
            font=('Consolas', 11, 'bold')
        ).pack(anchor=tk.W)
        
        # Model selection (duplicate for training tab)
        model_select_frame = tk.Frame(model_mgmt_frame, bg=self.colors['bg_panel'])
        model_select_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(model_select_frame,
            text="📦 ACTIVE MODEL:",
            bg=self.colors['bg_panel'],
            fg=self.colors['text_primary'],
            font=('Consolas', 10)
        ).pack(anchor=tk.W)
        
        # Duplicate model dropdown for training tab
        self.model_var_training = tk.StringVar()
        self.model_dropdown_training = tkinter_ttk.Combobox(model_select_frame,
            textvariable=self.model_var_training,
            state="readonly",
            font=('Consolas', 9),
            width=50
        )
        self.model_dropdown_training.pack(fill=tk.X, pady=2)
        self.model_dropdown_training.bind("<<ComboboxSelected>>", self.on_model_selection_changed)
        
        # Data Management Section moved to right panel for better space usage
        
        # Status display moved to right panel for better layout
        
        # Text selection code moved to after status_text creation
        
        # Main Display Area
        display_frame = tk.Frame(main_frame, bg=self.colors['bg_primary'])
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create the main visual interface container
        visual_container = self.theme.create_glowing_frame(display_frame, "VISUAL TARGETING MATRIX")
        visual_container.master.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Canvas for image display with cyberpunk styling
        self.canvas = tk.Canvas(visual_container,
            width=self.canvas_width, 
            height=self.canvas_height,
            bg=self.colors['canvas_bg'],
            highlightbackground=self.colors['border_glow'],
            highlightcolor=self.colors['neon_cyan'],
            highlightthickness=2,
            cursor='crosshair'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize optimized renderer
        self.renderer = OptimizedRenderer(self.canvas, self.canvas_width, self.canvas_height)
        
        # System Metrics Panel
        metrics_container = self.theme.create_glowing_frame(display_frame, "SYSTEM METRICS", width=380)
        metrics_container.master.pack(side=tk.RIGHT, fill=tk.Y)
        metrics_container.master.pack_propagate(False)  # Maintain fixed width
        
        self.create_cyberpunk_statistics_panel(metrics_container)
        
        # Data Management Section - moved here from left panel for better space usage
        data_section = tk.Frame(metrics_container, bg=self.colors['bg_panel'])
        data_section.pack(fill=tk.X, pady=10)
        
        tk.Label(data_section,
            text=f"{self.symbols['data']} DATA MANAGEMENT",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_purple'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        # Stats display - more vertical layout
        stats_frame = tk.Frame(data_section, bg=self.colors['bg_accent'], relief='solid', bd=1)
        stats_frame.pack(fill=tk.X, pady=5)
        
        # Current segments count
        self.current_stats_label = tk.Label(stats_frame,
            text="◢ CURRENT: 0 segments ◣",
            bg=self.colors['bg_accent'],
            fg=self.colors['neon_green'],
            font=('Consolas', 11, 'bold')
        )
        self.current_stats_label.pack(fill=tk.X, pady=1)
        
        # Total processed images
        self.processed_stats_label = tk.Label(stats_frame,
            text="◢ PROCESSED: 0/0 images ◣",
            bg=self.colors['bg_accent'],
            fg=self.colors['neon_cyan'],
            font=('Consolas', 11, 'bold')
        )
        self.processed_stats_label.pack(fill=tk.X, pady=1)
        
        # Images with segments
        self.with_segments_label = tk.Label(stats_frame,
            text="◢ WITH SEGMENTS: 0 ◣",
            bg=self.colors['bg_accent'],
            fg=self.colors['neon_purple'],
            font=('Consolas', 11, 'bold')
        )
        self.with_segments_label.pack(fill=tk.X, pady=1)
        
        # Total segments in dataset
        self.total_stats_label = tk.Label(stats_frame,
            text="◢ TOTAL SEGMENTS: 0 ◣",
            bg=self.colors['bg_accent'],
            fg=self.colors['neon_orange'],
            font=('Consolas', 11, 'bold')
        )
        self.total_stats_label.pack(fill=tk.X, pady=1)
        
        # Data operations
        data_ops = tk.Frame(data_section, bg=self.colors['bg_panel'])
        data_ops.pack(fill=tk.X, pady=5)
        
        self.save_button = self.theme.create_cyber_button(data_ops, "SECURE DATA", self.save_annotations, 'neon_green')
        self.save_button.pack(fill=tk.X, pady=2)
        
        self.load_ann_button = self.theme.create_cyber_button(data_ops, "LOAD ARCHIVE", self.load_annotations, 'neon_cyan')
        self.load_ann_button.pack(fill=tk.X, pady=2)
        
        # Add status terminal to right panel instead of left for better layout
        self.status_text, status_container = self.theme.create_status_display(metrics_container)
        status_container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Enable text selection and copying for status terminal
        def select_all(e):
            self.status_text.tag_add(tk.SEL, "1.0", tk.END)
            return "break"
        
        def copy_text(e):
            try:
                selection = self.status_text.get(tk.SEL_FIRST, tk.SEL_LAST)
                self.root.clipboard_clear()
                self.root.clipboard_append(selection)
            except tk.TclError:
                pass  # No selection
            return "break"
            
        self.status_text.bind("<Control-a>", select_all)
        self.status_text.bind("<Control-c>", copy_text)
        
        # Start the cyberpunk animations
        self.start_cyberpunk_animations()
        
        # Bind events
        self.bind_events()
        
        # Focus the root to receive key events
        self.root.focus_set()
        
    def bind_events(self):
        """Bind mouse and keyboard events"""
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux scroll down
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Pan functionality
        self.canvas.bind("<Motion>", self.on_mouse_motion)
        self.canvas.bind("<Button-2>", self.on_middle_click)  # Middle mouse button
        self.canvas.bind("<Button-3>", self.on_right_click)  # Right mouse button for panning
        self.canvas.bind("<B3-Motion>", self.on_right_drag)  # Right mouse drag
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)  # Right mouse release
        
        # Keyboard bindings
        self.root.bind("<Control-o>", lambda e: self.load_custom_image())
        self.root.bind("<Control-s>", lambda e: self.save_annotations())
        self.root.bind("<Control-b>", lambda e: self.create_backup("manual"))  # Manual backup
        self.root.bind("<Control-c>", lambda e: self.clear_segments())
        self.root.bind("<Control-z>", lambda e: self.undo_last_segment())
        self.root.bind("<Control-f>", lambda e: self.reset_view_to_fit())
        self.root.bind("<Left>", lambda e: self.previous_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Up>", lambda e: self.next_image(save_image=False)) #Skip the annotation of this image.
        self.root.bind("<Down>", lambda e: self.previous_image(save_image=False)) #Skip the annotation of this image backwards.
        self.root.bind("<Return>", lambda e: self.toggle_annotation_mode()) # Toggle between SAM and Manual modes
        self.root.bind("<Key>", self.on_key_press)
        
        # Pan and zoom bindings
        self.root.bind("<KeyPress-space>", self.on_space_press)
        self.root.bind("<KeyRelease-space>", self.on_space_release)
        self.root.bind("<space>", self.on_space_press)  # Alternative space binding
        
        # Enable focus for key events
        self.root.focus_set()
        self.canvas.focus_set()
        
        # Reset zoom binding
        self.root.bind("<Control-0>", lambda e: self.reset_zoom())
        
        # Random image binding
        self.root.bind("<Control-r>", lambda e: self.jump_to_random_image())
        
        # Make sure the root can receive focus
        self.root.bind("<Button-1>", lambda e: self.root.focus_set())
        self.canvas.bind("<Button-1>", lambda e: self.root.focus_set(), add="+")
        
    def load_sam_model(self):
        """Load the SAM model with GPU partitioning"""
        try:
            self.log_status("Loading SAM model...")
            
            # Try to find the SAM model file in various locations
            models_dir = self.config.models_dir if self.config else "./models"
            possible_paths = [
                os.path.join(models_dir, "sam_vit_h_4b8939.pth"),
                os.path.expanduser("~/.cache/sam/sam_vit_h_4b8939.pth"),
                "./models/sam_vit_h_4b8939.pth",
            ]
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            if not model_path:
                self.log_status(f"SAM model file not found. Run install_dependencies.sh to download.")
                return
                
            # Load SAM model with GPU context manager
            with sam_device_context() as device:
                self.device = device
                sam = sam_model_registry["vit_h"](checkpoint=model_path)
                sam.to(device=device)
                
                self.predictor = SamPredictor(sam)
                self.sam_model = sam
                
                self.log_status(f"SAM model loaded successfully on {device}")
                
                # Print GPU strategy for user visibility
                gpu_manager.print_status()
            
            # Update statistics panel
            if hasattr(self, 'update_statistics_panel'):
                self.update_statistics_panel()
            
        except Exception as e:
            error_msg = f"Error loading SAM model: {str(e)}"
            self.log_status(error_msg)
            messagebox.showerror("Model Error", error_msg)
    
    def load_image_list(self):
        """Load list of images from input directory"""
        try:
            input_path = Path(self.input_directory)
            if not input_path.exists():
                self.log_status(f"Input directory not found: {self.input_directory}")
                return
                
            # Get all image files
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            self.image_list = []
            
            for ext in image_extensions:
                self.image_list.extend(list(input_path.glob(f"*{ext}")))
                self.image_list.extend(list(input_path.glob(f"*{ext.upper()}")))
            
            # Sort for consistent ordering
            self.image_list.sort()
            
            self.log_status(f"Found {len(self.image_list)} images in directory")
            
            # Load existing annotations
            self.load_existing_annotations()
            
        except Exception as e:
            self.log_status(f"Error loading image list: {str(e)}")
    
    def load_existing_annotations(self):
        """Load existing annotations from file if they exist"""
        try:
            annotation_file = Path(self.root_directory) / "all_annotations.json"
            if annotation_file.exists():
                import json
                with open(annotation_file, 'r') as f:
                    saved_data = json.load(f)
                
                loaded_count = 0
                
                # Handle both dictionary format (correct) and array format (old/malformed)
                if isinstance(saved_data, dict):
                    # Correct dictionary format: {image_path: [annotations]}
                    for image_path, annotations in saved_data.items():
                        # Convert relative paths to absolute if needed
                        if not Path(image_path).is_absolute():
                            abs_path = Path(self.input_directory) / Path(image_path).name
                            if abs_path.exists():
                                image_path = str(abs_path)
                        
                        if Path(image_path).exists():
                            self.annotation_data[image_path] = annotations
                            loaded_count += len(annotations)
                            
                elif isinstance(saved_data, list):
                    # Legacy array format: [annotation_objects] - convert on the fly
                    self.log_status("Converting legacy annotation format...")
                    from collections import defaultdict
                    converted_data = defaultdict(list)
                    
                    for ann in saved_data:
                        if 'image_path' in ann:
                            image_path = ann['image_path']
                            correct_ann = {
                                'id': ann.get('id', 0),
                                'label': ann.get('label', self.default_label or 'object'),
                                'bbox': ann['bbox'],
                                'score': ann['score'],
                                'point': ann['point']
                            }
                            converted_data[image_path].append(correct_ann)
                    
                    # Store converted data
                    for image_path, annotations in converted_data.items():
                        if Path(image_path).exists():
                            self.annotation_data[image_path] = annotations
                            loaded_count += len(annotations)
                
                self.log_status(f"Loaded {loaded_count} existing annotations from {len(self.annotation_data)} images")
            else:
                self.log_status("No existing annotation file found - starting fresh")
                
        except Exception as e:
            self.log_status(f"Error loading existing annotations: {str(e)}")
    
    def auto_load_first_image(self):
        """Auto-load a random starting image"""
        if self.image_list:
            # Start at a random index instead of 0
            self.current_image_index = random.randint(0, len(self.image_list) - 1)
            self.load_current_image()
            
            # Reset zoom and center image for full view
            self.reset_view_to_fit()
            
            # Log the random start
            self.log_status(f"Started at random image {self.current_image_index + 1}/{len(self.image_list)}")
        else:
            self.log_status("No images found for auto-loading")
    
    def load_current_image(self, save_previous=True):
        """Load the current image by index"""
        if not self.image_list or self.current_image_index >= len(self.image_list):
            self.log_status("No valid image to load")
            return
            
        file_path = str(self.image_list[self.current_image_index])
        self.load_image_by_path(file_path, save_previous=save_previous)
        
        # Update navigation info
        self.update_image_info()
    
    def load_image_by_path(self, file_path: str, save_previous=True):
        """Load a specific image by file path"""
        try:
            # Save current image annotations before switching (only if save_previous=True)
            if save_previous and self.current_image_path:
                self.save_current_image_annotations()
            
            self.current_image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                self.log_status(f"Failed to load image: {file_path}")
                return
                
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Set image in SAM predictor
            if self.predictor:
                self.predictor.set_image(self.original_image)
            
            self.display_image()
            
            # Load existing annotations for this image
            self.load_current_image_annotations()
            
            h, w = self.original_image.shape[:2]
            filename = os.path.basename(file_path)
            self.log_status(f"Loaded image: {filename} ({w}x{h})")
            
        except Exception as e:
            error_msg = f"Error loading image: {str(e)}"
            self.log_status(error_msg)
            messagebox.showerror("Image Error", error_msg)
    
    def update_image_info(self):
        """Update the image info display"""
        if self.image_list and self.current_image_index < len(self.image_list):
            filename = self.image_list[self.current_image_index].name
            info_text = f"{filename}\n{self.current_image_index + 1}/{len(self.image_list)}"
            
            if self.original_image is not None:
                h, w = self.original_image.shape[:2]
                info_text += f"\n{w}x{h}"
                
            self.image_info_label.config(text=info_text)
        else:
            self.image_info_label.config(text="No images loaded")
    
    def next_image(self, save_image=True):
        """Go to next image"""
        if not self.image_list:
            return
            
        # Save current state before switching (only if save_image=True)
        if save_image:
            self.save_current_image_annotations()
        else:
            self.log_status("◢ SKIPPING IMAGE - NO ANNOTATIONS SAVED ◣")
        
        # Check if skip mode is enabled
        if self.skip_annotated:
            next_index = self.find_next_unannotated_image(self.current_image_index, direction=1)
            if next_index is not None:
                self.current_image_index = next_index
            else:
                self.log_status("◢ 🎉 ALL IMAGES PROCESSED! Turn off filter to review completed work ◣")
                self.show_completion_screen()
                return
        else:
            # Move to next image normally
            self.current_image_index = (self.current_image_index + 1) % len(self.image_list)
        
        # Clear any lingering drawings from previous image
        self.canvas.delete("segment")
        self.canvas.delete("yolo_prediction")
        if hasattr(self, 'overlay_images'):
            self.overlay_images.clear()
        if hasattr(self, 'overlay_cache'):
            self.overlay_cache.clear()
        
        # Load the new image WITHOUT auto-saving (pass save_image parameter)
        self.load_current_image(save_previous=save_image)
        
        # Reset zoom and center image for full view
        self.reset_view_to_fit()
        
        # Update label to reflect current state
        self.label_var.set(self.current_label)
    
    def previous_image(self, save_image=True):
        """Go to previous image"""
        if not self.image_list:
            return
            
        # Save current state before switching (only if save_image=True)
        if save_image:
            self.save_current_image_annotations()
        else:
            self.log_status("◢ SKIPPING IMAGE BACKWARDS - NO ANNOTATIONS SAVED ◣")
        
        # Check if skip mode is enabled
        if self.skip_annotated:
            prev_index = self.find_next_unannotated_image(self.current_image_index, direction=-1)
            if prev_index is not None:
                self.current_image_index = prev_index
            else:
                self.log_status("◢ 🎉 ALL IMAGES PROCESSED! Turn off filter to review completed work ◣")
                self.show_completion_screen()
                return
        else:
            # Move to previous image normally
            self.current_image_index = (self.current_image_index - 1) % len(self.image_list)
        
        # Clear any lingering drawings from previous image
        self.canvas.delete("segment")
        self.canvas.delete("yolo_prediction")
        if hasattr(self, 'overlay_images'):
            self.overlay_images.clear()
        if hasattr(self, 'overlay_cache'):
            self.overlay_cache.clear()
        
        # Load the new image WITHOUT auto-saving (pass save_image parameter)
        self.load_current_image(save_previous=save_image)
        
        # Reset zoom and center image for full view
        self.reset_view_to_fit()
        
        # Update label to reflect current state
        self.label_var.set(self.current_label)
    
    def show_completion_screen(self):
        """Display completion screen with statistics when all images are processed"""
        try:
            # Clear canvas
            self.canvas.delete("all")
            
            # Calculate comprehensive statistics
            total_images = len(self.image_list) if hasattr(self, 'image_list') else 0
            processed_images = len(self.annotation_data)
            total_annotations = sum(len(annotations) for annotations in self.annotation_data.values())
            images_with_annotations = len([k for k, v in self.annotation_data.items() if v])
            images_reviewed_empty = processed_images - images_with_annotations
            
            # Calculate average annotations per annotated image
            avg_annotations = total_annotations / images_with_annotations if images_with_annotations > 0 else 0
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            
            # Create completion screen
            # Main title
            self.canvas.create_text(
                center_x, center_y - 200,
                text="◢◣ ANNOTATION MISSION COMPLETE ◤◥",
                fill=self.colors['neon_cyan'],
                font=('Consolas', 24, 'bold'),
                tags="completion"
            )
            
            # Subtitle
            self.canvas.create_text(
                center_x, center_y - 160,
                text="🎯 ALL IMAGES PROCESSED SUCCESSFULLY 🎯",
                fill=self.colors['neon_green'],
                font=('Consolas', 16, 'bold'),
                tags="completion"
            )
            
            # Statistics section
            stats_y_start = center_y - 100
            line_height = 30
            
            stats = [
                f"📊 TOTAL IMAGES PROCESSED: {processed_images:,} / {total_images:,}",
                f"🎯 TOTAL ANNOTATIONS (HITS): {total_annotations:,}",
                f"📝 IMAGES WITH ANNOTATIONS: {images_with_annotations:,}",
                f"👁️ IMAGES REVIEWED (EMPTY): {images_reviewed_empty:,}",
                f"📈 AVERAGE ANNOTATIONS PER IMAGE: {avg_annotations:.1f}",
                "",
                f"⚡ ANNOTATION RATE: {(total_annotations/processed_images):.1f} hits/image",
                f"🏆 COMPLETION PERCENTAGE: 100.0%"
            ]
            
            for i, stat in enumerate(stats):
                if stat:  # Skip empty lines
                    color = self.colors['neon_orange'] if '🏆' in stat else self.colors['text_primary']
                    if 'HITS' in stat:
                        color = self.colors['neon_cyan']
                    elif 'COMPLETION' in stat:
                        color = self.colors['neon_green']
                    
                    self.canvas.create_text(
                        center_x, stats_y_start + (i * line_height),
                        text=stat,
                        fill=color,
                        font=('Consolas', 14, 'bold' if any(x in stat for x in ['HITS', 'COMPLETION']) else 'normal'),
                        tags="completion"
                    )
            
            # Instructions
            self.canvas.create_text(
                center_x, center_y + 120,
                text="📋 NEXT STEPS:",
                fill=self.colors['neon_purple'],
                font=('Consolas', 14, 'bold'),
                tags="completion"
            )
            
            instructions = [
                "• Uncheck '⚡ Only show unprocessed images' to review your work",
                "• Use Ctrl+S to save final annotations",
                "• Export YOLO dataset for training",
                "• Your work is automatically backed up!"
            ]
            
            for i, instruction in enumerate(instructions):
                self.canvas.create_text(
                    center_x, center_y + 150 + (i * 25),
                    text=instruction,
                    fill=self.colors['text_secondary'],
                    font=('Consolas', 12),
                    tags="completion"
                )
            
            # Decorative elements
            self.canvas.create_text(
                center_x, center_y + 280,
                text="◢◣◤◥ ⚡ NEURAL MATRIX ANNOTATION SYSTEM ⚡ ◢◣◤◥",
                fill=self.colors['border_glow'],
                font=('Consolas', 12, 'bold'),
                tags="completion"
            )
            
        except Exception as e:
            self.log_status(f"◢ ERROR creating completion screen: {str(e)} ◣")
    
    def jump_to_random_image(self):
        """Jump to a random image in the list"""
        if not self.image_list:
            return
            
        # Save current state before jumping
        if len(self.current_segments) > 0:
            self.save_current_image_annotations()
        
        # Jump to random image
        old_index = self.current_image_index
        self.current_image_index = random.randint(0, len(self.image_list) - 1)
        
        # Make sure we actually moved (avoid staying on same image)
        if len(self.image_list) > 1:
            while self.current_image_index == old_index:
                self.current_image_index = random.randint(0, len(self.image_list) - 1)
        
        # Clear any lingering drawings from previous image
        self.canvas.delete("segment")
        self.canvas.delete("yolo_prediction")
        if hasattr(self, 'overlay_images'):
            self.overlay_images.clear()
        if hasattr(self, 'overlay_cache'):
            self.overlay_cache.clear()
        
        self.load_current_image()
        
        # Reset zoom and center image for full view
        self.reset_view_to_fit()
        
        # Update label and log the jump
        self.label_var.set(self.current_label)
        self.log_status(f"Jumped to random image {self.current_image_index + 1}/{len(self.image_list)}")
    
    def copy_current_filename(self):
        """Copy the current image filename to clipboard"""
        if not self.current_image_path:
            self.log_status("No image loaded to copy filename")
            return
            
        filename = os.path.basename(self.current_image_path)
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(filename)
            self.log_status(f"◢ FILENAME COPIED: {filename} ◣")
        except Exception as e:
            self.log_status(f"Error copying filename: {str(e)}")
    
    def save_current_image_annotations(self):
        """Save annotations for the current image"""
        if not self.current_image_path:
            return
            
        # Prepare annotation data for current image
        annotations = []
        for i, segment in enumerate(self.current_segments):
            # Ensure all values are JSON-serializable Python types
            bbox = segment['bbox']
            point = segment['point']
            
            ann = {
                'id': int(i),
                'label': str(self.current_label),
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                'score': float(segment['score']),
                'point': [int(point[0]), int(point[1])]
            }
            annotations.append(ann)
        
        # Store in annotation data dictionary
        self.annotation_data[self.current_image_path] = annotations
        
        # Log the save action
        image_name = os.path.basename(self.current_image_path)
        if annotations:
            self.log_status(f"Auto-saved {len(annotations)} segments for {image_name}")
        else:
            self.log_status(f"Auto-saved 0 segments for {image_name} (empty image)")
        
        # Update stats
        self.update_annotation_stats()
    
    def load_current_image_annotations(self):
        """Load annotations for the current image"""
        if not self.current_image_path:
            return
            
        # Clear current segments
        self.current_segments.clear()
        
        # First load existing manual annotations
        manual_annotations_loaded = False
        if self.current_image_path in self.annotation_data:
            annotations = self.annotation_data[self.current_image_path]
            self.log_status(f"◢ LOADED: {len(annotations)} existing annotations ◣")
            manual_annotations_loaded = True
            
            # Convert loaded annotations back to segments format
            for annotation in annotations:
                try:
                    # Extract data from saved annotation
                    bbox = annotation.get('bbox', [0, 0, 0, 0])
                    point = annotation.get('point', [0, 0])
                    score = annotation.get('score', 1.0)
                    
                    # Create segment structure (no mask for loaded annotations)
                    segment = {
                        'bbox': bbox,
                        'point': tuple(point),
                        'score': score,
                        'mask': None,  # Loaded annotations don't have masks
                        'loaded': True  # Mark as loaded annotation for different rendering
                    }
                    
                    self.current_segments.append(segment)
                    
                except Exception as e:
                    self.log_status(f"◢ ERROR loading annotation: {str(e)} ◣")
        
        # If no manual annotations exist, check for cached predictions and transfer them
        if not manual_annotations_loaded:
            predictions = self.get_cached_predictions(self.current_image_path)
            if predictions:
                self.log_status(f"◢ TRANSFERRING: {len(predictions)} YOLO predictions to editable annotations ◣")
                
                for i, prediction in enumerate(predictions):
                    try:
                        # Extract prediction data - YOLO format is [x1, y1, x2, y2]
                        yolo_bbox = prediction.get('bbox', [0, 0, 0, 0])
                        confidence = prediction.get('confidence', 0.5)
                        
                        # Convert YOLO bbox [x1, y1, x2, y2] to SAM bbox [x, y, width, height]
                        x1, y1, x2, y2 = yolo_bbox
                        sam_bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, width, height]
                        
                        # Calculate center point from bbox for editing
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Create segment structure from prediction
                        segment = {
                            'bbox': sam_bbox,  # Use SAM format bbox
                            'point': (center_x, center_y),
                            'score': confidence,
                            'mask': None,  # Predictions don't have masks
                            'loaded': True,  # Mark as loaded for rendering
                            'from_prediction': True  # Mark as transferred from prediction
                        }
                        
                        self.current_segments.append(segment)
                        
                    except Exception as e:
                        self.log_status(f"◢ ERROR transferring prediction {i}: {str(e)} ◣")
                
                # Auto-save the transferred predictions as regular annotations
                if predictions:
                    self.save_current_image_annotations()
                    self.log_status(f"◢ AUTO-SAVED: {len(predictions)} transferred predictions ◣")
        
        # Refresh display to show loaded/transferred segments
        if self.current_segments:
            self.display_image()
            
        self.update_annotation_stats()
        self.update_dataset_status()
    
    def update_annotation_stats(self):
        """Update annotation statistics display"""
        current_count = len(self.current_segments)
        self.current_stats_label.config(text=f"◢ CURRENT: {current_count} segments ◣")
        
        # Count ALL processed images (including zero-annotation negative examples)
        total_processed = len(self.annotation_data)
        total_images = len(self.image_list) if hasattr(self, 'image_list') else 0
        self.processed_stats_label.config(text=f"◢ PROCESSED: {total_processed}/{total_images} images ◣")
        
        # Count images with segments
        total_with_segments = len([k for k, v in self.annotation_data.items() if v])
        self.with_segments_label.config(text=f"◢ WITH SEGMENTS: {total_with_segments} ◣")
        
        # Calculate total segments across all images
        total_segments = sum(len(annotations) for annotations in self.annotation_data.values())
        self.total_stats_label.config(text=f"◢ TOTAL SEGMENTS: {total_segments} ◣")
        
        # Update window title with HITS count
        self.update_window_title()
        
        # Update the statistics panel if it exists
        if hasattr(self, 'update_statistics_panel'):
            self.update_statistics_panel()
    
    def on_label_change(self, event=None):
        """Handle label change"""
        self.current_label = self.label_var.get().strip()
        if not self.current_label:
            self.current_label = self.default_label or "object"
            self.label_var.set(self.current_label)
        self.log_status(f"Changed label to: {self.current_label}")
    
    def on_key_press(self, event):
        """Handle key press events"""
        # This helps ensure arrow keys work
        if event.keysym in ['Left', 'Right']:
            self.root.focus_set()
    
    def on_space_press(self, event):
        """Handle space key press for pan mode"""
        if event.keysym == 'space':
            self.space_pressed = True
            self.canvas.config(cursor="fleur")
            self.log_status("Pan mode enabled (hold space and drag)")
        
    def on_space_release(self, event):
        """Handle space key release"""
        if event.keysym == 'space':
            self.space_pressed = False
            self.is_panning = False
            self.canvas.config(cursor="crosshair")
            self.log_status("Pan mode disabled")
    
    def on_ctrl_press(self, event):
        """Handle Ctrl key press"""
        pass  # Handled in mouse wheel event
        
    def on_ctrl_release(self, event):
        """Handle Ctrl key release"""
        pass
    
    def on_mouse_motion(self, event):
        """Handle mouse motion for panning"""
        if self.space_pressed and self.is_panning:
            # Calculate pan delta
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            
            self.display_image()
    
    def on_middle_click(self, event):
        """Handle middle mouse button for panning"""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
    def reset_zoom(self):
        """Reset zoom and pan to default"""
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.display_image()
        self.zoom_label.config(text=f"◎ MAGNIFICATION: {self.zoom_level:.1f}x")
        self.log_status("Reset zoom and pan")
    
    def reset_view_to_fit(self):
        """Reset zoom and pan to fit image to window based on orientation"""
        if self.original_image is None:
            return
            
        # Get image dimensions
        h, w = self.original_image.shape[:2]
        
        # Get canvas dimensions (use actual size if available)
        canvas_w = self.canvas.winfo_width() or self.canvas_width
        canvas_h = self.canvas.winfo_height() or self.canvas_height
        
        # Calculate scale factors
        scale_w = canvas_w / w
        scale_h = canvas_h / h
        
        # Choose scale based on image orientation:
        # If width < height: scale so height = window height (use scale_h)
        # If width > height: scale so width = window width (use scale_w)
        if w < h:
            # Portrait image - fit to window height
            fit_scale = scale_h
            self.log_status(f"Portrait image: fitting to height")
        else:
            # Landscape image (including square) - fit to window width  
            fit_scale = scale_w
            self.log_status(f"Landscape image: fitting to width")
        
        # Store the base scale (how image fits in window)
        self.base_scale = min(fit_scale, 1.0)
        
        # Reset user zoom to 1.0 (100%) on fit
        self.zoom_level = 1.0
        
        # Final image scale is base_scale * zoom_level
        self.image_scale = self.base_scale * self.zoom_level
        
        # Calculate image size at this image scale
        scaled_w = w * self.image_scale
        scaled_h = h * self.image_scale
        
        # Center the image in the canvas
        self.pan_offset_x = (canvas_w - scaled_w) / 2
        self.pan_offset_y = (canvas_h - scaled_h) / 2
        
        # Update display
        self.display_image()
        self.zoom_label.config(text=f"◎ MAGNIFICATION: {self.zoom_level:.1f}x")
        self.log_status(f"Fit image to view ({self.zoom_level:.1f}x) - {w}x{h}")
            
    def load_custom_image(self):
        """Load a custom image via file dialog"""
        if not hasattr(self, 'initial_dir'):
            self.initial_dir = self.root_directory
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            initialdir=self.initial_dir,
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image_by_path(file_path)
            # Update the image info since this is a custom load
            h, w = self.original_image.shape[:2] if self.original_image is not None else (0, 0)
            filename = os.path.basename(file_path)
            self.image_info_label.config(text=f"{filename} (custom)\n{w}x{h}")
            
    def load_image(self):
        """Load an image for annotation"""
        if not hasattr(self, 'initial_dir'):
            self.initial_dir = self.root_directory
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            initialdir=self.initial_dir,
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.original_image = cv2.imread(file_path)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                # Set image in SAM predictor
                if self.predictor:
                    self.predictor.set_image(self.original_image)
                
                self.display_image()
                self.clear_segments()
                
                # Update image info
                h, w = self.original_image.shape[:2]
                filename = os.path.basename(file_path)
                self.image_info_label.config(text=f"{filename}\\n{w}x{h}")
                
                self.log_status(f"Loaded image: {filename} ({w}x{h})")
                
            except Exception as e:
                error_msg = f"Error loading image: {str(e)}"
                self.log_status(error_msg)
                messagebox.showerror("Image Error", error_msg)
                
    def display_image(self):
        """Display the current image on canvas with zoom and pan support"""
        if self.original_image is None:
            return
            
        # Use the image_scale as set by reset_view_to_fit or zoom operations
        # image_scale should already be properly calculated
        h, w = self.original_image.shape[:2]
        
        # Only recalculate if not properly initialized
        if not hasattr(self, 'image_scale') or not hasattr(self, 'base_scale'):
            # This should trigger reset_view_to_fit instead
            self.reset_view_to_fit()
            return
        
        # Resize image for display
        new_w = int(w * self.image_scale)
        new_h = int(h * self.image_scale)
        
        self.current_image = cv2.resize(self.original_image, (new_w, new_h))
        
        # Convert to PIL and display
        pil_image = Image.fromarray(self.current_image)
        self.canvas_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and reset overlay references
        self.canvas.delete("all")
        if hasattr(self, 'overlay_images'):
            self.overlay_images.clear()
        
        # Clear temporary elements
        self.canvas.delete("temp_bbox")
        
        # Create image with pan offset
        self.canvas.create_image(
            self.pan_offset_x, self.pan_offset_y, 
            anchor=tk.NW, image=self.canvas_image, 
            tags="main_image"
        )
        
        # Draw existing segments (loaded annotations will already be in current_segments)
        self.draw_segments()
        
        # Draw predicted annotations if toggle is enabled and predictions exist
        # (but only if they haven't been transferred to regular annotations)
        if self.prediction_toggle_enabled and hasattr(self, 'current_image_path') and self.current_image_path:
            # Check if we have any transferred predictions in current segments
            has_transferred_predictions = any(seg.get('from_prediction', False) for seg in self.current_segments)
            if not has_transferred_predictions and self.current_image_path not in self.annotation_data:
                self.draw_predicted_annotations()
        
    def on_canvas_click(self, event):
        """Handle canvas click for segmentation, removal, or panning"""
        # Always set focus to ensure key events work
        self.root.focus_set()
        
        if self.space_pressed:
            # Start panning
            self.is_panning = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.log_status("Started panning...")
            return
            
        if self.original_image is None:
            return
        
        # Convert canvas coordinates to image coordinates accounting for zoom/pan
        canvas_x = event.x - self.pan_offset_x
        canvas_y = event.y - self.pan_offset_y
        
        # image_scale already includes zoom_level, so don't multiply again
        img_x = int(canvas_x / self.image_scale)
        img_y = int(canvas_y / self.image_scale)
        
        # Ensure coordinates are within image bounds
        h, w = self.original_image.shape[:2]
        if img_x < 0 or img_x >= w or img_y < 0 or img_y >= h:
            self.log_status(f"Click outside image bounds: ({img_x}, {img_y})")
            return  # Click outside image bounds
        
        # Check if Shift is pressed for removal
        shift_pressed = bool(event.state & 0x1)  # Shift modifier
        
        if shift_pressed:
            # Remove nearest annotation
            self.remove_nearest_segment(img_x, img_y)
        else:
            # Handle annotation based on current mode
            if self.annotation_mode == "SAM":
                # SAM mode - AI segmentation
                if self.predictor is None:
                    self.log_status("SAM model not loaded - cannot annotate")
                    return
                self.segment_at_point(img_x, img_y)
            else:
                # Manual mode - start drawing bounding box
                self.start_manual_bbox(img_x, img_y)
    
    def start_manual_bbox(self, x, y):
        """Start drawing a manual bounding box"""
        self.drawing_bbox = True
        self.bbox_start = (x, y)
        self.temp_bbox = None
        
        # Bind motion and release events for drawing
        self.canvas.bind("<B1-Motion>", self.on_bbox_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_bbox_release)
        
        self.log_status(f"◢ MANUAL BOX: Started at ({x}, {y}) ◣")
    
    def on_bbox_drag(self, event):
        """Handle mouse drag while drawing bounding box"""
        if not self.drawing_bbox or not self.bbox_start:
            return
            
        # Convert canvas coordinates to image coordinates
        canvas_x = event.x - self.pan_offset_x
        canvas_y = event.y - self.pan_offset_y
        img_x = int(canvas_x / self.image_scale)
        img_y = int(canvas_y / self.image_scale)
        
        # Update temp bbox
        self.temp_bbox = (self.bbox_start[0], self.bbox_start[1], img_x, img_y)
        
        # Redraw to show current bbox
        self.display_image()
    
    def on_bbox_release(self, event):
        """Handle mouse release to finish bounding box"""
        if not self.drawing_bbox or not self.bbox_start:
            return
            
        # Convert canvas coordinates to image coordinates
        canvas_x = event.x - self.pan_offset_x
        canvas_y = event.y - self.pan_offset_y
        img_x = int(canvas_x / self.image_scale)
        img_y = int(canvas_y / self.image_scale)
        
        # Ensure coordinates are within image bounds
        h, w = self.original_image.shape[:2]
        img_x = max(0, min(img_x, w - 1))
        img_y = max(0, min(img_y, h - 1))
        
        # Create final bounding box
        x1, y1 = self.bbox_start
        x2, y2 = img_x, img_y
        
        # Ensure proper ordering (x1 < x2, y1 < y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Check minimum box size
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            self.log_status("◢ BOX TOO SMALL: Minimum 10x10 pixels ◣")
        else:
            # Create manual segment
            self.create_manual_segment(x1, y1, x2, y2)
        
        # Reset drawing state
        self.drawing_bbox = False
        self.bbox_start = None
        self.temp_bbox = None
        
        # Unbind motion events
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        
        # Redraw without temp bbox
        self.display_image()
    
    def create_manual_segment(self, x1, y1, x2, y2):
        """Create a segment from manual bounding box with anti-SAM feature"""
        # Create a mask for the bounding box
        h, w = self.original_image.shape[:2]
        manual_mask = np.zeros((h, w), dtype=bool)
        manual_mask[y1:y2, x1:x2] = True
        
        # Anti-bounding box: Remove intersecting areas from existing SAM segments
        sam_segments_modified = 0
        segments_to_remove = []
        
        for i, segment in enumerate(self.current_segments):
            # Only process SAM segments (they have masks)
            if segment.get('mask') is not None:
                sam_mask = segment['mask']
                
                # Check if there's an intersection
                intersection = sam_mask & manual_mask
                if np.any(intersection):
                    # Remove the intersecting area from SAM mask
                    segment['mask'] = segment['mask'] & ~manual_mask
                    
                    # Check if the SAM segment still has any pixels left
                    if not np.any(segment['mask']):
                        # If the entire SAM segment was removed, mark it for deletion
                        segments_to_remove.append(i)
                        self.log_status(f"◢ ANTI-BOX: Completely removed SAM segment #{i} ◣")
                    else:
                        # Update the bounding box for the modified SAM segment
                        y_indices, x_indices = np.where(segment['mask'])
                        if len(x_indices) > 0 and len(y_indices) > 0:
                            x_min, x_max = int(x_indices.min()), int(x_indices.max())
                            y_min, y_max = int(y_indices.min()), int(y_indices.max())
                            segment['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
                            
                            # Update center point
                            segment['point'] = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
                        
                        sam_segments_modified += 1
                        self.log_status(f"◢ ANTI-BOX: Modified SAM segment #{i} ◣")
        
        # Remove segments that were completely erased (in reverse order to maintain indices)
        for i in reversed(segments_to_remove):
            self.current_segments.pop(i)
        
        # Create manual segment data structure
        segment = {
            'mask': manual_mask,
            'score': 1.0,  # Manual annotations get perfect score
            'point': ((x1 + x2) // 2, (y1 + y2) // 2),  # Center point
            'bbox': [x1, y1, x2 - x1, y2 - y1]  # [x, y, width, height]
        }
        
        # Add to current segments
        self.current_segments.append(segment)
        
        if sam_segments_modified > 0 or len(segments_to_remove) > 0:
            self.log_status(f"◢ ANTI-BOX: Manual box {x2-x1}x{y2-y1} removed parts from {sam_segments_modified} SAM segments, deleted {len(segments_to_remove)} ◣")
        else:
            self.log_status(f"◢ MANUAL BOX: Created {x2-x1}x{y2-y1} at ({x1},{y1}) ◣")
        
        # Update display
        self.display_image()
        
        # Update annotation stats
        self.update_annotation_stats()
        
        # Clear any old segments to ensure our new segment shows
        self.current_masks = []  # Clear SAM masks since we're not using them
        
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zoom or threshold adjustment"""
        # Check if Ctrl is pressed for threshold
        ctrl_pressed = False
        if hasattr(event, 'state'):
            ctrl_pressed = bool(event.state & 0x4)  # Standard Ctrl modifier
            
        if ctrl_pressed:
            self.handle_threshold_scroll(event)
        else:
            self.handle_zoom(event)
        
    def handle_threshold_scroll(self, event):
        """Handle scroll wheel for threshold adjustment"""
        delta = 0
        
        # Determine scroll direction
        if hasattr(event, 'delta') and event.delta != 0:
            # Windows/MacOS
            delta = 0.05 if event.delta > 0 else -0.05
        elif hasattr(event, 'num'):
            # Linux/X11
            if event.num == 4:
                delta = 0.05  # Scroll up
            elif event.num == 5:
                delta = -0.05  # Scroll down
        
        # Apply threshold change
        if delta != 0:
            new_threshold = max(0.0, min(1.0, self.threshold + delta))
            self.threshold_var.set(new_threshold)
            self.on_threshold_change(new_threshold)
            self.log_status(f"Threshold: {new_threshold:.2f}")

            
    def handle_zoom(self, event):
        """Handle Ctrl+scroll for zoom"""
        delta = 0
        
        # Determine zoom direction
        if hasattr(event, 'delta') and event.delta != 0:
            # Windows/MacOS
            delta = 0.1 if event.delta > 0 else -0.1
        elif hasattr(event, 'num'):
            # Linux/X11
            if event.num == 4:
                delta = 0.1  # Zoom in
            elif event.num == 5:
                delta = -0.1  # Zoom out
        
        # Apply zoom change
        if delta != 0:
            mouse_x = event.x
            mouse_y = event.y
            
            old_zoom = self.zoom_level
            self.zoom_level = max(0.1, min(5.0, self.zoom_level + delta))
            
            if self.zoom_level != old_zoom:
                # Calculate zoom ratio for smooth transitions
                zoom_ratio = self.zoom_level / old_zoom
                
                # Update image_scale using base_scale * zoom_level
                if hasattr(self, 'base_scale'):
                    self.image_scale = self.base_scale * self.zoom_level
                else:
                    # Fallback if base_scale not set
                    self.image_scale *= zoom_ratio
                
                # Zoom towards mouse position (translate so mouse point stays fixed)
                self.pan_offset_x = mouse_x - (mouse_x - self.pan_offset_x) * zoom_ratio
                self.pan_offset_y = mouse_y - (mouse_y - self.pan_offset_y) * zoom_ratio
                
                self.display_image()
                self.zoom_label.config(text=f"◎ MAGNIFICATION: {self.zoom_level:.1f}x")
                self.log_status(f"◎ ZOOM MATRIX: {self.zoom_level:.1f}x")

        
    def on_threshold_change(self, value):
        """Handle threshold change"""
        self.threshold = float(value)
        self.threshold_label.config(text=f"◢ PRECISION: {self.threshold:.3f} ◣")
        
        # Re-segment if we have existing segments
        if self.current_segments:
            self.refresh_segments()
            
    def on_canvas_drag(self, event):
        """Handle canvas drag for panning"""
        if self.space_pressed and self.is_panning:
            # Calculate pan delta
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            
            self.display_image()
        
    def on_canvas_release(self, event):
        """Handle canvas release"""
        if self.is_panning:
            self.is_panning = False
            self.log_status("Panning finished")
    
    def on_right_click(self, event):
        """Handle right mouse button press for panning"""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.log_status("Right-click panning started...")
        
    def on_right_drag(self, event):
        """Handle right mouse button drag for panning"""
        if self.is_panning:
            # Calculate pan delta
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            
            self.display_image()
    
    def on_right_release(self, event):
        """Handle right mouse button release"""
        if self.is_panning:
            self.is_panning = False
            self.log_status("Right-click panning finished")
        
    def is_spurious_mask(self, mask: np.ndarray, score: float) -> tuple[bool, str]:
        """Check if mask is likely spurious (background, too large, etc.)"""
        if not hasattr(self, 'original_image') or self.original_image is None:
            return False, ""
            
        h, w = self.original_image.shape[:2]
        total_pixels = h * w
        mask_area = np.sum(mask)
        coverage_ratio = mask_area / total_pixels
        
        # Filter 1: Reject masks covering more than 40% of image (likely background)
        if coverage_ratio > 0.4:
            return True, f"Large background mask ({coverage_ratio:.1%} coverage)"
            
        # Filter 2: Reject very low score large masks
        if coverage_ratio > 0.15 and score < 0.8:
            return True, f"Low quality large mask (score: {score:.3f}, coverage: {coverage_ratio:.1%})"
            
        # Filter 3: Reject masks with extreme aspect ratios (likely artifacts)
        bbox = self.mask_to_bbox(mask)
        if bbox[2] > 0 and bbox[3] > 0:
            aspect_ratio = max(bbox[2], bbox[3]) / min(bbox[2], bbox[3])
            if aspect_ratio > 10 and coverage_ratio > 0.05:
                return True, f"Extreme aspect ratio mask ({aspect_ratio:.1f}:1)"
        
        return False, ""

    def check_mask_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> tuple[bool, float]:
        """Check if two masks have overlapping areas and return overlap ratio"""
        intersection = np.logical_and(mask1, mask2)
        overlap_area = np.sum(intersection)
        
        if overlap_area == 0:
            return False, 0.0
            
        # Calculate overlap ratio relative to smaller mask
        area1 = np.sum(mask1)
        area2 = np.sum(mask2)
        smaller_area = min(area1, area2)
        overlap_ratio = overlap_area / smaller_area if smaller_area > 0 else 0.0
        
        # Smart overlap threshold: reject 100% overlaps (usually spurious)
        if overlap_ratio > 0.95:
            return False, overlap_ratio  # Don't split on near-complete overlaps
            
        return overlap_ratio > 0.15, overlap_ratio  # 15% meaningful overlap threshold
    
    def create_voronoi_boundary(self, mask: np.ndarray, point1: tuple, point2: tuple) -> np.ndarray:
        """Create Voronoi-like boundary between two points within a mask"""
        h, w = mask.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Calculate distance from each pixel to each point
        dist1 = np.sqrt((x_coords - point1[0])**2 + (y_coords - point1[1])**2)
        dist2 = np.sqrt((x_coords - point2[0])**2 + (y_coords - point2[1])**2)
        
        # Create boundary: pixels closer to point1 belong to region1
        voronoi_region1 = dist1 <= dist2
        
        # Only apply within the original mask
        region1_mask = mask & voronoi_region1
        region2_mask = mask & ~voronoi_region1
        
        return region1_mask, region2_mask
    
    def find_nearby_segments(self, point: tuple, max_distance: int = 100) -> list:
        """Find segments within proximity distance of a point"""
        nearby_segments = []
        px, py = point
        
        for i, segment in enumerate(self.current_segments):
            seg_point = segment['point']
            sx, sy = seg_point
            distance = ((px - sx)**2 + (py - sy)**2)**0.5
            
            if distance <= max_distance:
                nearby_segments.append((i, distance))
                
        return nearby_segments
    
    def proximity_split_segments(self, new_mask: np.ndarray, new_point: tuple, new_score: float, nearby_segments: list):
        """Split nearby segments using proximity-based Voronoi approach"""
        if len(nearby_segments) > 4:  # Limit to prevent chaos
            self.log_status(f"◢ TOO MANY NEARBY: {len(nearby_segments)} segments, adding normally ◣")
            return [{
                'mask': new_mask,
                'score': new_score,
                'point': new_point,
                'bbox': self.mask_to_bbox(new_mask)
            }]
        
        # Create a combined mask from all nearby segments plus the new one
        combined_mask = new_mask.copy()
        points_to_split = [new_point]
        scores_to_split = [new_score]
        
        # Get indices of nearby segments for removal
        removal_indices = []
        for seg_idx, distance in nearby_segments:
            segment = self.current_segments[seg_idx]
            combined_mask = np.logical_or(combined_mask, segment['mask'])
            points_to_split.append(segment['point'])
            scores_to_split.append(segment['score'])
            removal_indices.append(seg_idx)
        
        # Remove the original segments (mark for removal)
        for idx in sorted(removal_indices, reverse=True):
            self.current_segments[idx] = None
        
        # Clean up None entries
        self.current_segments = [seg for seg in self.current_segments if seg is not None]
        
        # Now split the combined mask using Voronoi boundaries for all points
        split_masks = self.multi_point_voronoi_split(combined_mask, points_to_split)
        
        # Create new segments from split masks
        new_segments = []
        for i, (split_mask, point, score) in enumerate(zip(split_masks, points_to_split, scores_to_split)):
            if np.any(split_mask):  # Only add non-empty masks
                new_segments.append({
                    'mask': split_mask,
                    'score': score,
                    'point': point,
                    'bbox': self.mask_to_bbox(split_mask)
                })
        
        self.log_status(f"◢ PROXIMITY SPLIT: Created {len(new_segments)} segments from {len(points_to_split)} points ◣")
        return new_segments
    
    def multi_point_voronoi_split(self, combined_mask: np.ndarray, points: list) -> list:
        """Split a mask into regions using Voronoi diagram for multiple points"""
        h, w = combined_mask.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # For each pixel, find the closest point
        region_masks = []
        
        for i, target_point in enumerate(points):
            tx, ty = target_point
            
            # Calculate distance from each pixel to this point
            dist_to_target = np.sqrt((x_coords - tx)**2 + (y_coords - ty)**2)
            
            # Check if this point is closest for each pixel
            is_closest = np.ones_like(dist_to_target, dtype=bool)
            
            for j, other_point in enumerate(points):
                if i == j:
                    continue
                ox, oy = other_point
                dist_to_other = np.sqrt((x_coords - ox)**2 + (y_coords - oy)**2)
                is_closest &= (dist_to_target <= dist_to_other)
            
            # Apply only within the original combined mask
            region_mask = combined_mask & is_closest
            region_masks.append(region_mask)
        
        return region_masks
    
    def split_overlapping_segments(self, new_mask: np.ndarray, new_point: tuple, new_score: float):
        """Check for overlaps and split segments using Voronoi approach"""
        overlapping_indices = []
        
        # Find all segments that overlap with the new mask
        for i, existing_segment in enumerate(self.current_segments):
            existing_mask = existing_segment['mask']
            is_overlap, overlap_ratio = self.check_mask_overlap(new_mask, existing_mask)
            
            if is_overlap:
                overlapping_indices.append((i, overlap_ratio))
                self.log_status(f"Found {overlap_ratio:.1%} overlap with segment {i}")
        
        # If no overlaps but nearby segments exist, check for proximity-based splitting
        if not overlapping_indices:
            nearby_segments = self.find_nearby_segments(new_point, max_distance=80)
            
            if len(nearby_segments) >= 2:  # At least 2 nearby segments to consider splitting
                self.log_status(f"◢ PROXIMITY SPLIT: Found {len(nearby_segments)} nearby segments ◣")
                return self.proximity_split_segments(new_mask, new_point, new_score, nearby_segments)
            
            # No overlaps or nearby segments, add segment normally
            return [{
                'mask': new_mask,
                'score': new_score,
                'point': new_point,
                'bbox': self.mask_to_bbox(new_mask)
            }]
        
        # Safety check: Limit number of splits to prevent chaos
        if len(overlapping_indices) > 3:
            self.log_status(f"◢ SPLIT LIMIT: Too many overlaps ({len(overlapping_indices)}), adding without split ◣")
            return [{
                'mask': new_mask,
                'score': new_score,
                'point': new_point,
                'bbox': self.mask_to_bbox(new_mask)
            }]
        
        # Handle overlaps with Voronoi splitting
        new_segments = []
        
        for overlap_idx, overlap_ratio in overlapping_indices:
            existing_segment = self.current_segments[overlap_idx]
            existing_mask = existing_segment['mask']
            existing_point = existing_segment['point']
            existing_score = existing_segment['score']
            
            self.log_status(f"◢ VORONOI SPLIT: Segments at {existing_point} and {new_point} ◣")
            
            # Create union of both masks for splitting
            union_mask = np.logical_or(new_mask, existing_mask)
            
            # Split using Voronoi boundary
            region1_mask, region2_mask = self.create_voronoi_boundary(
                union_mask, existing_point, new_point
            )
            
            # Create new segments with tight bounding boxes
            if np.any(region1_mask):
                split_segment1 = {
                    'mask': region1_mask,
                    'score': existing_score,
                    'point': existing_point,
                    'bbox': self.mask_to_bbox(region1_mask)
                }
                new_segments.append(split_segment1)
            
            if np.any(region2_mask):
                split_segment2 = {
                    'mask': region2_mask,
                    'score': new_score,
                    'point': new_point,
                    'bbox': self.mask_to_bbox(region2_mask)
                }
                new_segments.append(split_segment2)
            
            # Mark existing segment for removal
            self.current_segments[overlap_idx] = None
        
        # Remove marked segments
        self.current_segments = [seg for seg in self.current_segments if seg is not None]
        
        # If new mask doesn't overlap with anything, add it normally
        if not any(self.check_mask_overlap(new_mask, seg['mask'])[0] for seg in new_segments):
            new_segments.append({
                'mask': new_mask,
                'score': new_score,
                'point': new_point,
                'bbox': self.mask_to_bbox(new_mask)
            })
        
        return new_segments

    def segment_at_point(self, x: int, y: int):
        """Perform segmentation at the given point"""
        try:
            self.log_status(f"Segmenting at point ({x}, {y})")
            
            # Perform segmentation with SAM GPU context
            input_point = np.array([[x, y]])
            input_label = np.array([1])
            
            with sam_device_context():
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
            
            # Filter masks by threshold
            valid_masks = []
            self.log_status(f"SAM returned {len(masks)} masks with scores: {[f'{s:.3f}' for s in scores]}")
            for i, (mask, score) in enumerate(zip(masks, scores)):
                if score >= self.threshold:
                    valid_masks.append((mask, score, i))
                    
            if valid_masks:
                # Take the best mask
                best_mask, best_score, best_idx = max(valid_masks, key=lambda x: x[1])
                
                # Check if mask is spurious (background, too large, etc.)
                is_spurious, reason = self.is_spurious_mask(best_mask, best_score)
                if is_spurious:
                    self.log_status(f"◢ REJECTED SEGMENT: {reason} ◣")
                    return
                    
                # Create segment from the best mask (without Voronoi splitting for now)
                bbox = self.mask_to_bbox(best_mask)
                segment = {
                    'mask': best_mask,
                    'score': float(best_score),  # Convert numpy float to Python float
                    'point': (int(x), int(y)),  # Ensure Python ints
                    'bbox': bbox
                }
                
                # Add to current segments
                self.current_segments.append(segment)
                self.log_status(f"◢ SAM SEGMENT: Added with score {best_score:.3f} ◣")

                # Update stats
                self.update_annotation_stats()
                
                # Refresh display
                self.draw_segments()
                
            else:
                self.log_status(f"No segments found above threshold {self.threshold:.2f}")
                
        except Exception as e:
            error_msg = f"Error during segmentation: {str(e)}"
            self.log_status(error_msg)
            
    def mask_to_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert binary mask to bounding box (x, y, width, height)"""
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return (0, 0, 0, 0)
            
        x_min, x_max = int(x_indices.min()), int(x_indices.max())
        y_min, y_max = int(y_indices.min()), int(y_indices.max())
        
        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
        
    def draw_segments(self):
        """Draw current segments using optimized renderer"""
        if not hasattr(self, 'original_image') or self.original_image is None:
            return
            
        # Get image shape for renderer
        image_shape = self.original_image.shape[:2]  # (height, width)
        
        # Use optimized renderer for existing segments
        if self.current_segments:
            self.renderer.update_segments(
                segments=self.current_segments,
                image_shape=image_shape,
                image_scale=self.image_scale,
                pan_offset=(self.pan_offset_x, self.pan_offset_y)
            )
        
        # Draw temporary bounding box if in manual mode
        if self.temp_bbox and self.annotation_mode == "MANUAL":
            self.draw_temp_bbox()
    
    def draw_temp_bbox(self):
        """Draw temporary bounding box while dragging"""
        if not self.temp_bbox:
            return
            
        x1, y1, x2, y2 = self.temp_bbox
        
        # Convert to canvas coordinates
        canvas_x1 = x1 * self.image_scale + self.pan_offset_x
        canvas_y1 = y1 * self.image_scale + self.pan_offset_y
        canvas_x2 = x2 * self.image_scale + self.pan_offset_x
        canvas_y2 = y2 * self.image_scale + self.pan_offset_y
        
        # Draw temporary bounding box
        self.canvas.create_rectangle(
            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
            outline=self.colors['neon_orange'],
            width=2,
            dash=(5, 5),  # Dashed line to indicate temporary
            tags="temp_bbox"
        )
        
        # Add size indicator
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        center_x = (canvas_x1 + canvas_x2) / 2
        center_y = (canvas_y1 + canvas_y2) / 2
        
        self.canvas.create_text(
            center_x, center_y,
            text=f"{width}x{height}",
            fill=self.colors['neon_orange'],
            font=('Consolas', 12, 'bold'),
            tags="temp_bbox"
        )
            
    def draw_mask_overlay(self, mask: np.ndarray, segment_id: int):
        """Legacy method - now handled by optimized renderer"""
        # This method is kept for compatibility but rendering is now handled
        # by the OptimizedRenderer in draw_segments()
        pass
        
    def draw_bbox(self, bbox: Tuple[int, int, int, int], segment_id: int):
        """Legacy method - now handled by optimized renderer"""
        # This method is kept for compatibility but rendering is now handled
        # by the OptimizedRenderer in draw_segments()
        pass
        
    def refresh_segments(self):
        """Refresh all segments with current threshold"""
        # Clear canvas drawings and renderer cache
        self.canvas.delete("segment")
        self.canvas.delete("overlay")
        self.canvas.delete("composite_overlay")
        self.canvas.delete("lod_composite_overlay")
        
        if hasattr(self, 'renderer'):
            self.renderer.clear_cache()
            
        # Re-process segments with new threshold
        valid_segments = []
        for segment in self.current_segments:
            if segment['score'] >= self.threshold:
                valid_segments.append(segment)
                
        self.current_segments = valid_segments
        self.draw_segments()
        
        self.log_status(f"Refreshed segments: {len(valid_segments)} remaining")
        
    def clear_segments(self):
        """Clear all current segments"""
        self.current_segments.clear()
        
        # Clear renderer cache
        if hasattr(self, 'renderer'):
            self.renderer.clear_cache()
            
        # Clear canvas
        if self.canvas_image:
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)
            
        self.log_status("Cleared all segments")
        
        # Update stats
        self.update_annotation_stats()
    
    def draw_predicted_annotations(self):
        """Draw predicted YOLO bounding boxes as semi-transparent overlays"""
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            return
        
        predictions = self.get_cached_predictions(self.current_image_path)
        if not predictions:
            return
        
        # Draw each predicted bounding box
        for i, prediction in enumerate(predictions):
            if 'bbox' in prediction:
                bbox = prediction['bbox']
                confidence = prediction.get('confidence', 0.0)
                
                # Scale bbox to current display scale
                x1, y1, x2, y2 = bbox
                x1 = int(x1 * self.image_scale) + self.pan_offset_x
                y1 = int(y1 * self.image_scale) + self.pan_offset_y
                x2 = int(x2 * self.image_scale) + self.pan_offset_x
                y2 = int(y2 * self.image_scale) + self.pan_offset_y
                
                # Choose color based on confidence
                if confidence > 0.7:
                    color = '#00ff00'  # Green for high confidence
                elif confidence > 0.4:
                    color = '#ffff00'  # Yellow for medium confidence
                else:
                    color = '#ff8800'  # Orange for low confidence
                
                # Draw semi-transparent rectangle
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline=color,
                    width=2,
                    tags=f"predicted_bbox_{i}",
                    dash=(5, 5)  # Dashed line to distinguish from real annotations
                )
                
                # Draw confidence label
                self.canvas.create_text(
                    x1 + 2, y1 - 15,
                    text=f"PRED: {confidence:.2f}",
                    fill=color,
                    font=('Consolas', 10, 'bold'),
                    anchor=tk.W,
                    tags=f"predicted_label_{i}"
                )
    
    def undo_last_segment(self):
        """Undo the last segmentation"""
        if not self.current_segments:
            self.log_status("No segments to undo")
            return
            
        # Remove the last segment
        removed_segment = self.current_segments.pop()
        
        # Clear overlay cache to force regeneration
        if hasattr(self, 'overlay_cache'):
            self.overlay_cache.clear()
            
        # Refresh display
        self.display_image()
        
        # Update stats
        self.update_annotation_stats()
        
        self.log_status(f"Undid last segment (score: {removed_segment['score']:.3f})")
    
    def remove_nearest_segment(self, click_x, click_y):
        """Remove the nearest segment to the click point"""
        if not self.current_segments:
            self.log_status("No segments to remove")
            return
        
        # Find the nearest segment by click point distance
        min_distance = float('inf')
        nearest_index = -1
        distances = []  # For debugging
        
        for i, segment in enumerate(self.current_segments):
            seg_x, seg_y = segment['point']
            distance = ((click_x - seg_x) ** 2 + (click_y - seg_y) ** 2) ** 0.5
            distances.append(f"#{i}: {distance:.1f}px")
            
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        
        # Debug info
        self.log_status(f"◢ REMOVAL CHECK: Click at ({click_x}, {click_y})")
        self.log_status(f"◢ DISTANCES: {', '.join(distances)}")
        
        # Remove the nearest segment if within reasonable distance (30 pixels - more precise)
        if nearest_index >= 0 and min_distance < 30:
            removed_segment = self.current_segments.pop(nearest_index)
            
            # Clear overlay cache to force regeneration
            if hasattr(self, 'overlay_cache'):
                self.overlay_cache.clear()
            
            # Refresh display
            self.display_image()
            
            # Update stats
            self.update_annotation_stats()
            
            segment_type = "SAM" if removed_segment.get('mask') is not None else "Manual"
            self.log_status(f"◢ REMOVED: {segment_type} segment #{nearest_index} at ({removed_segment['point'][0]}, {removed_segment['point'][1]}) - distance: {min_distance:.1f}px ◣")
        else:
            self.log_status(f"◢ NO REMOVAL: Closest segment is {min_distance:.1f}px away (threshold: 30px) ◣")
    
    def sam_auto_predict(self):
        """Use SAM to automatically predict and annotate objects in the current image"""
        if self.original_image is None:
            self.log_status("No image loaded")
            return
            
        if self.predictor is None:
            self.log_status("SAM model not loaded")
            return
        
        try:
            self.log_status("Running SAM automatic prediction...")
            
            # Use SAM's automatic mask generation with GPU context
            from segment_anything import SamAutomaticMaskGenerator
            
            with sam_device_context():
                mask_generator = SamAutomaticMaskGenerator(
                    model=self.sam_model,
                    points_per_side=32,
                    pred_iou_thresh=0.8,
                    stability_score_thresh=0.85,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=500,  # Filter small regions
                )
                
                # Generate masks
                masks = mask_generator.generate(self.original_image)
            
            if not masks:
                self.log_status("No objects detected by SAM auto-prediction")
                return
            
            # Filter masks by stability score and area
            filtered_masks = []
            for mask_data in masks:
                stability_score = mask_data.get('stability_score', 0)
                area = mask_data.get('area', 0)
                
                # Only keep high-quality, reasonably-sized masks
                if stability_score >= self.threshold and area >= 500 and area <= 50000:
                    filtered_masks.append(mask_data)
            
            # Sort by stability score (best first) and limit to reasonable number
            filtered_masks.sort(key=lambda x: x.get('stability_score', 0), reverse=True)
            filtered_masks = filtered_masks[:10]  # Limit to top 10
            
            # Convert to our segment format
            new_segments_count = 0
            for mask_data in filtered_masks:
                mask = mask_data['segmentation']
                bbox_data = mask_data['bbox']  # [x, y, w, h]
                stability_score = mask_data.get('stability_score', 0)
                
                # Convert bbox format
                bbox = (int(bbox_data[0]), int(bbox_data[1]), int(bbox_data[2]), int(bbox_data[3]))
                
                # Find center point of mask for display
                y_indices, x_indices = np.where(mask)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    center_x = int(x_indices.mean())
                    center_y = int(y_indices.mean())
                else:
                    center_x, center_y = bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2
                
                # Create segment info
                segment_info = {
                    'mask': mask,
                    'score': float(stability_score),  # Convert numpy float to Python float
                    'point': (int(center_x), int(center_y)),  # Ensure Python ints
                    'bbox': bbox
                }
                
                self.current_segments.append(segment_info)
                new_segments_count += 1
            
            # Update display and stats
            self.display_image()
            self.update_annotation_stats()
            
            self.log_status(f"SAM auto-prediction completed: added {new_segments_count} segments")
            
        except Exception as e:
            error_msg = f"Error in SAM auto-prediction: {str(e)}"
            self.log_status(error_msg)
    
    def create_statistics_panel(self, parent):
        """Create the statistics and model feedback panel"""
        # Model Statistics
        model_frame = ttk.LabelFrame(parent, text="SAM Model Stats", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.sam_status_label = ttk.Label(model_frame, text="SAM: Loading...", foreground="orange")
        self.sam_status_label.pack(anchor=tk.W)
        
        self.sam_device_label = ttk.Label(model_frame, text="Device: Unknown")
        self.sam_device_label.pack(anchor=tk.W)
        
        self.sam_memory_label = ttk.Label(model_frame, text="Memory: Unknown")
        self.sam_memory_label.pack(anchor=tk.W)
        
        # Current Image Stats
        image_frame = ttk.LabelFrame(parent, text="Current Image", padding=10)
        image_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.current_segments_label = ttk.Label(image_frame, text="Segments: 0")
        self.current_segments_label.pack(anchor=tk.W)
        
        self.avg_confidence_label = ttk.Label(image_frame, text="Avg Confidence: N/A")
        self.avg_confidence_label.pack(anchor=tk.W)
        
        self.bbox_coverage_label = ttk.Label(image_frame, text="Coverage: 0%")
        self.bbox_coverage_label.pack(anchor=tk.W)
        
        # Dataset Stats
        dataset_frame = ttk.LabelFrame(parent, text="Dataset Stats", padding=10)
        dataset_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.total_images_label = ttk.Label(dataset_frame, text="Total Images: 0")
        self.total_images_label.pack(anchor=tk.W)
        
        self.annotated_images_label = ttk.Label(dataset_frame, text="Annotated: 0")
        self.annotated_images_label.pack(anchor=tk.W)
        
        self.total_segments_label = ttk.Label(dataset_frame, text="Total Segments: 0")
        self.total_segments_label.pack(anchor=tk.W)
        
        # YOLO Data Format Info
        yolo_frame = ttk.LabelFrame(parent, text="YOLO Export Info", padding=10)
        yolo_frame.pack(fill=tk.X, pady=(0, 10))
        
        class_info = self.current_label or "object"
        format_text = f"Format: YOLO v8\nClass: 0 ({class_info})\nCoords: Normalized\nBBoxes: x_center y_center w h"
        ttk.Label(yolo_frame, text=format_text, font=('TkDefaultFont', 8)).pack(anchor=tk.W)
        
        # Action Buttons
        actions_frame = ttk.LabelFrame(parent, text="YOLO Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.export_yolo_button = ttk.Button(actions_frame, text="Export YOLO Dataset", command=self.export_yolo_dataset)
        self.export_yolo_button.pack(fill=tk.X, pady=(0, 5))
        
        self.train_yolo_button = ttk.Button(actions_frame, text="Train YOLO Model", command=self.train_yolo_model)
        self.train_yolo_button.pack(fill=tk.X, pady=(0, 5))
        
        self.predict_yolo_button = ttk.Button(actions_frame, text="Test YOLO on Current", command=self.predict_yolo_current)
        self.predict_yolo_button.pack(fill=tk.X)
        
        # Update initial stats
        self.update_statistics_panel()
        
    def update_statistics_panel(self):
        """Update all statistics in the panel"""
        try:
            # SAM Model Status (using cyberpunk labels)
            if hasattr(self, 'predictor') and self.predictor is not None:
                if hasattr(self, 'model_status_label'):
                    self.model_status_label.config(text="◢ MODEL STATUS: ACTIVE ◣")
                if hasattr(self, 'sam_gpu_label'):
                    self.sam_gpu_label.config(text=f"◢ SAM CORE: {self.device.upper()} ◣")
            else:
                if hasattr(self, 'model_status_label'):
                    self.model_status_label.config(text="◢ MODEL STATUS: OFFLINE ◣")
                if hasattr(self, 'sam_gpu_label'):
                    self.sam_gpu_label.config(text="◢ SAM CORE: LOADING... ◣")
            
            # Current Image Stats and update prediction count
            num_segments = len(self.current_segments)
            if hasattr(self, 'current_segments_label'):
                self.current_segments_label.config(text=f"Segments: {num_segments}")
            if hasattr(self, 'prediction_count_label'):
                self.prediction_count_label.config(text=f"◢ PREDICTIONS: {num_segments} ◣")
            
            if num_segments > 0:
                avg_conf = sum(seg['score'] for seg in self.current_segments) / num_segments
                if hasattr(self, 'avg_confidence_label'):
                    self.avg_confidence_label.config(text=f"Avg Confidence: {avg_conf:.3f}")
                
                # Calculate coverage (rough estimate based on bbox areas)
                if hasattr(self, 'original_image') and self.original_image is not None:
                    h, w = self.original_image.shape[:2]
                    total_area = w * h
                    bbox_area = sum(bbox[2] * bbox[3] for bbox in [seg['bbox'] for seg in self.current_segments])
                    coverage = min(100, (bbox_area / total_area) * 100)
                    if hasattr(self, 'bbox_coverage_label'):
                        self.bbox_coverage_label.config(text=f"Coverage: {coverage:.1f}%")
                else:
                    if hasattr(self, 'bbox_coverage_label'):
                        self.bbox_coverage_label.config(text="Coverage: N/A")
            else:
                if hasattr(self, 'avg_confidence_label'):
                    self.avg_confidence_label.config(text="Avg Confidence: N/A")
                if hasattr(self, 'bbox_coverage_label'):
                    self.bbox_coverage_label.config(text="Coverage: 0%")
            
            # Dataset Stats - Enhanced with total progress information
            total_images = len(self.image_list) if hasattr(self, 'image_list') else 0
            total_processed = len(self.annotation_data)
            total_with_segments = len([k for k, v in self.annotation_data.items() if v])
            total_segments = sum(len(annotations) for annotations in self.annotation_data.values())
            
            if hasattr(self, 'total_images_label'):
                self.total_images_label.config(text=f"Total Images: {total_images}")
            if hasattr(self, 'annotated_images_label'):
                self.annotated_images_label.config(text=f"Processed: {total_processed} ({total_with_segments} with segments)")
            if hasattr(self, 'total_segments_label'):
                self.total_segments_label.config(text=f"Total Segments: {total_segments}")
            
        except Exception as e:
            self.log_status(f"Error updating statistics: {str(e)}")
        
    def export_yolo_dataset(self):
        """Export current annotations to YOLO format"""
        if not self.annotation_data:
            self.log_status("No annotations to export")
            return
            
        try:
            from pathlib import Path
            
            # Create export directory
            output_dir = self.config.output_dir if self.config else "./output"
            export_dir = Path(output_dir) / "yolo_export"
            export_dir.mkdir(exist_ok=True)
            
            images_dir = export_dir / "images"
            labels_dir = export_dir / "labels"
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
            
            exported_count = 0
            
            for image_path, annotations in self.annotation_data.items():
                if not annotations:  # Skip empty annotations
                    continue
                    
                # Copy image file
                image_name = Path(image_path).name
                target_image = images_dir / image_name
                
                import shutil
                shutil.copy2(image_path, target_image)
                
                # Create YOLO label file
                label_name = Path(image_path).stem + ".txt"
                label_file = labels_dir / label_name
                
                # Load image to get dimensions
                import cv2
                img = cv2.imread(image_path)
                h, w = img.shape[:2]
                
                with open(label_file, 'w') as f:
                    for ann in annotations:
                        bbox = ann['bbox']  # (x, y, w, h) tuple
                        # Convert to YOLO format (normalized center coordinates)
                        x_center = (bbox[0] + bbox[2] / 2) / w
                        y_center = (bbox[1] + bbox[3] / 2) / h
                        width = bbox[2] / w
                        height = bbox[3] / h
                        
                        # Class 0 for target object
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                exported_count += 1
            
            # Create proper train/val split (80/20)
            import random
            image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            random.shuffle(image_files)
            
            split_idx = int(0.8 * len(image_files))
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            # Create train/val directories
            train_images_dir = export_dir / "train" / "images"
            train_labels_dir = export_dir / "train" / "labels"
            val_images_dir = export_dir / "val" / "images"
            val_labels_dir = export_dir / "val" / "labels"
            
            for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Move files to train/val splits
            import shutil
            for img_file in train_files:
                label_file = labels_dir / (img_file.stem + ".txt")
                shutil.move(str(img_file), str(train_images_dir / img_file.name))
                if label_file.exists():
                    shutil.move(str(label_file), str(train_labels_dir / label_file.name))
            
            for img_file in val_files:
                label_file = labels_dir / (img_file.stem + ".txt")
                shutil.move(str(img_file), str(val_images_dir / img_file.name))
                if label_file.exists():
                    shutil.move(str(label_file), str(val_labels_dir / label_file.name))
            
            # Remove old directories
            shutil.rmtree(images_dir)
            shutil.rmtree(labels_dir)
            
            # Create dataset YAML with proper paths
            class_names = self.config.class_names if self.config and self.config.class_names else [self.current_label or "object"]
            yaml_content = f"""train: {train_images_dir.absolute()}
val: {val_images_dir.absolute()}

nc: {len(class_names)}
names: {class_names}
"""
            
            with open(export_dir / "dataset.yaml", 'w') as f:
                f.write(yaml_content)
            
            self.log_status(f"Exported {exported_count} annotated images to YOLO format: {export_dir}")
            
        except Exception as e:
            error_msg = f"Error exporting YOLO dataset: {str(e)}"
            self.log_status(error_msg)
    
    def train_yolo_model(self):
        """Train a YOLO model on the exported dataset"""
        try:
            from ultralytics import YOLO
            import threading
            
            def train_thread():
                try:
                    self.log_status("Starting YOLO training...")
                    
                    # Use YOLO GPU context for training
                    with yolo_device_context() as device:
                        # Load YOLO model
                        model = YOLO('yolov8n.pt')  # Start with nano model for speed
                        
                        # First export the dataset
                        self.export_yolo_dataset()
                        
                        # Generate unique model name with timestamp
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_name = f'yolo_sam_{timestamp}'
                        
                        # Get optimal batch size for this GPU
                        optimal_batch = gpu_manager.get_optimal_batch_size('yolo_training', base_size=4)
                        
                        # Get paths from config
                        output_dir = self.config.output_dir if self.config else "./output"
                        models_dir = self.config.models_dir if self.config else "./models"
                        yolo_export_path = Path(output_dir) / "yolo_export" / "dataset.yaml"
                        yolo_models_path = Path(models_dir) / "yolo_models"

                        # Train the model with GPU-optimized settings
                        results = model.train(
                            data=str(yolo_export_path),
                            epochs=100,  # More epochs for better training
                            imgsz=640,
                            batch=optimal_batch,  # GPU-optimized batch size
                            device=device,  # Use dynamically assigned GPU
                            project=str(yolo_models_path),
                            name=model_name,
                            patience=20,  # Early stopping
                            save_period=10  # Save checkpoints
                        )

                    # Copy best model to easily accessible location
                    import shutil
                    best_model_path = yolo_models_path / model_name / "weights" / "best.pt"
                    if best_model_path.exists():
                        current_best = Path(models_dir) / "current_best_yolo.pt"
                        shutil.copy2(str(best_model_path), str(current_best))
                        self.log_status(f"YOLO training completed! Best model saved as: {current_best}")
                        self.log_status(f"Full results in: {yolo_models_path}/{model_name}/")
                    else:
                        self.log_status("YOLO training completed but best model not found")
                    
                except Exception as e:
                    self.log_status(f"Training error: {str(e)}")
            
            # Run training in background thread
            train_thread_obj = threading.Thread(target=train_thread, daemon=True)
            train_thread_obj.start()
            
            self.log_status("YOLO training started in background...")
            
        except ImportError:
            self.log_status("Error: ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            self.log_status(f"Error starting YOLO training: {str(e)}")
    
    def predict_yolo_current(self):
        """Run YOLO prediction on current image"""
        if not hasattr(self, 'original_image') or self.original_image is None:
            self.log_status("No image loaded")
            return
            
        try:
            from ultralytics import YOLO
            from pathlib import Path
            
            # Look for current best model first
            models_dir = self.config.models_dir if self.config else "./models"
            model_path = Path(models_dir) / "current_best_yolo.pt"

            if not model_path.exists():
                # Try to find any trained model in yolo_models directory
                yolo_models_dir = Path(models_dir) / "yolo_models"
                if yolo_models_dir.exists():
                    model_files = list(yolo_models_dir.glob("*/weights/best.pt"))
                    if model_files:
                        # Use most recent model
                        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
                        self.log_status(f"Using model: {model_path}")
                    else:
                        self.log_status("No trained YOLO model found. Train first!")
                        return
                else:
                    self.log_status("No trained YOLO model found. Train first!")
                    return
            else:
                self.log_status("Using current best YOLO model")
            
            # Use YOLO GPU context for prediction
            with yolo_device_context() as device:
                # Load model and predict
                model = YOLO(str(model_path))
                
                # Save current image temporarily for prediction
                temp_image_path = "/tmp/current_image_for_yolo.jpg"
                cv2.imwrite(temp_image_path, self.original_image)
                
                # Run prediction with dynamic device assignment
                results = model(temp_image_path, device=device)
            
            # Process results
            detections = results[0].boxes
            
            if detections is not None and len(detections) > 0:
                self.log_status(f"YOLO found {len(detections)} detections!")
                
                # Draw YOLO detections on canvas
                self.draw_yolo_predictions(detections)
                
                # Log detection details
                for i, det in enumerate(detections):
                    conf = float(det.conf[0])
                    bbox = det.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                    self.log_status(f"Detection {i+1}: confidence={conf:.3f}, bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})")
            else:
                self.log_status("YOLO found no detections")
            
        except ImportError:
            self.log_status("Error: ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            self.log_status(f"Error running YOLO prediction: {str(e)}")
    
    def draw_yolo_predictions(self, detections):
        """Draw YOLO predictions on the canvas"""
        try:
            for i, det in enumerate(detections):
                conf = float(det.conf[0])
                bbox = det.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                
                # Convert to canvas coordinates
                x1, y1, x2, y2 = bbox
                canvas_x1 = int(x1 * self.image_scale) + self.pan_offset_x
                canvas_y1 = int(y1 * self.image_scale) + self.pan_offset_y
                canvas_x2 = int(x2 * self.image_scale) + self.pan_offset_x
                canvas_y2 = int(y2 * self.image_scale) + self.pan_offset_y
                
                # Draw YOLO detection in blue
                self.canvas.create_rectangle(
                    canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                    outline='blue', width=3,
                    tags="yolo_prediction"
                )
                
                # Add confidence label
                self.canvas.create_text(
                    canvas_x1, canvas_y1 - 10,
                    text=f"YOLO: {conf:.2f}",
                    fill='blue', anchor=tk.SW,
                    tags="yolo_prediction"
                )
                
        except Exception as e:
            self.log_status(f"Error drawing YOLO predictions: {str(e)}")
        
    def save_annotations(self):
        """Save all annotations to file with automatic backup"""
        # First save current image annotations
        self.save_current_image_annotations()
        
        if not self.annotation_data:
            messagebox.showwarning("Save Warning", "No annotations to save")
            return
            
        try:
            # Save in the same dictionary format that can be loaded back
            # This preserves the image_path -> annotations structure
            
            # Save to JSON file
            save_path = str(Path(self.root_directory) / "all_annotations.json")
            
            # Create backup before saving (if file exists)
            if Path(save_path).exists():
                self.create_backup("before_save")
            
            with open(save_path, 'w') as f:
                json.dump(self.annotation_data, f, indent=2)
                
            total_images = len(self.annotation_data)
            total_segments = sum(len(annotations) for annotations in self.annotation_data.values())
            self.log_status(f"Saved {total_segments} annotations from {total_images} images to {save_path}")
            messagebox.showinfo("Save Success", f"Saved {total_segments} annotations from {total_images} images")
            
            # Create a post-save backup as well
            self.create_backup("after_save")
            
        except Exception as e:
            error_msg = f"Error saving annotations: {str(e)}"
            self.log_status(error_msg)
            messagebox.showerror("Save Error", error_msg)
            
    def load_annotations(self):
        """Load annotations from file"""
        file_path = filedialog.askopenfilename(
            title="Load Annotations",
            initialdir=self.config.models_dir if self.config else os.getcwd(),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    annotations = json.load(f)
                    
                # TODO: Implement annotation loading logic
                self.log_status(f"Loaded {len(annotations)} annotations from {file_path}")
                
            except Exception as e:
                error_msg = f"Error loading annotations: {str(e)}"
                self.log_status(error_msg)
                messagebox.showerror("Load Error", error_msg)
                
    def log_status(self, message: str):
        """Log status message"""
        self.logger.info(message)
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def on_closing(self):
        """Handle application closing - auto-save current work with backup"""
        try:
            # Save current image annotations
            self.save_current_image_annotations()
            
            # Create backup before closing if any annotations exist
            if self.annotation_data:
                # First create backup of current state
                self.create_backup("on_exit")
                # Then save normally (which will also create backups)
                self.save_annotations()
                
            self.log_status("Auto-saved all work before closing")
            
        except Exception as e:
            self.log_status(f"Error during auto-save on close: {str(e)}")
        
        # Close the application
        self.root.destroy()
    
    def create_cyberpunk_statistics_panel(self, parent):
        """Create a cyberpunk-themed statistics panel"""
        
        # GPU Status Section
        gpu_section = tk.Frame(parent, bg=self.colors['bg_panel'])
        gpu_section.pack(fill=tk.X, pady=5)
        
        tk.Label(gpu_section,
            text=f"{self.symbols['lightning']} GPU NEURAL CORES",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_orange'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        gpu_frame = tk.Frame(gpu_section, bg=self.colors['bg_accent'], relief='solid', bd=1)
        gpu_frame.pack(fill=tk.X, pady=2)
        
        self.sam_gpu_label = tk.Label(gpu_frame,
            text="◢ SAM CORE: LOADING... ◣",
            bg=self.colors['bg_accent'],
            fg=self.colors['neon_cyan'],
            font=('Consolas', 12),
            width=30,  # Fixed width
            anchor='center'
        )
        self.sam_gpu_label.pack(fill=tk.X, pady=1)
        
        self.yolo_gpu_label = tk.Label(gpu_frame,
            text="◢ YOLO CORE: LOADING... ◣",
            bg=self.colors['bg_accent'],
            fg=self.colors['neon_green'],
            font=('Consolas', 12),
            width=30,  # Fixed width
            anchor='center'
        )
        self.yolo_gpu_label.pack(fill=tk.X, pady=1)
        
        # Neural Network Status section removed to save space
        
        # Performance Metrics
        perf_section = tk.Frame(parent, bg=self.colors['bg_panel'])
        perf_section.pack(fill=tk.X, pady=5)
        
        tk.Label(perf_section,
            text=f"{self.symbols['processing']} PERFORMANCE",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_pink'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        perf_frame = tk.Frame(perf_section, bg=self.colors['bg_accent'], relief='solid', bd=1)
        perf_frame.pack(fill=tk.X, pady=2)
        
        self.render_time_label = tk.Label(perf_frame,
            text="◢ RENDER TIME: 0.0ms ◣",
            bg=self.colors['bg_accent'],
            fg=self.colors['neon_orange'],
            font=('Consolas', 12),
            width=30,  # Fixed width
            anchor='center'
        )
        self.render_time_label.pack(fill=tk.X, pady=1)
        
        self.avg_render_label = tk.Label(perf_frame,
            text="◢ AVG RENDER: 0.0ms ◣",
            bg=self.colors['bg_accent'],
            fg=self.colors['neon_cyan'],
            font=('Consolas', 12),
            width=30,  # Fixed width
            anchor='center'
        )
        self.avg_render_label.pack(fill=tk.X, pady=1)
        
        # YOLO Training Section
        yolo_section = tk.Frame(parent, bg=self.colors['bg_panel'])
        yolo_section.pack(fill=tk.X, pady=5)
        
        tk.Label(yolo_section,
            text=f"{self.symbols['atom']} YOLO TRAINING",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_green'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        yolo_buttons = tk.Frame(yolo_section, bg=self.colors['bg_panel'])
        yolo_buttons.pack(fill=tk.X, pady=2)
        
        self.export_button = self.theme.create_cyber_button(yolo_buttons, "EXPORT", self.export_yolo_dataset, 'neon_orange')
        self.export_button.pack(fill=tk.X, pady=1)
        
        self.train_button = self.theme.create_cyber_button(yolo_buttons, "TRAIN NET", self.train_yolo_model, 'neon_green')
        self.train_button.pack(fill=tk.X, pady=1)
        
        self.predict_yolo_button = self.theme.create_cyber_button(yolo_buttons, "PREDICT", self.predict_yolo_current, 'neon_cyan')
        self.predict_yolo_button.pack(fill=tk.X, pady=1)
        
        # Memory Status
        memory_section = tk.Frame(parent, bg=self.colors['bg_panel'])
        memory_section.pack(fill=tk.X, pady=5)
        
        tk.Label(memory_section,
            text=f"{self.symbols['data']} MEMORY BANK",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_purple'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        memory_frame = tk.Frame(memory_section, bg=self.colors['bg_accent'], relief='solid', bd=1)
        memory_frame.pack(fill=tk.X, pady=2)
        
        self.gpu_memory_label = tk.Label(memory_frame,
            text="◢ GPU MEMORY: SCANNING... ◣",
            bg=self.colors['bg_accent'],
            fg=self.colors['neon_pink'],
            font=('Consolas', 12),
            width=30,  # Fixed width
            anchor='center'
        )
        self.gpu_memory_label.pack(fill=tk.X, pady=1)
        
        self.cache_status_label = tk.Label(memory_frame,
            text="◢ CACHE STATUS: EMPTY ◣",
            bg=self.colors['bg_accent'],
            fg=self.colors['neon_cyan'],
            font=('Consolas', 12),
            width=30,  # Fixed width
            anchor='center'
        )
        self.cache_status_label.pack(fill=tk.X, pady=1)
        
        # System Status
        system_section = tk.Frame(parent, bg=self.colors['bg_panel'])
        system_section.pack(fill=tk.X, pady=5)
        
        tk.Label(system_section,
            text=f"{self.symbols['target']} SYSTEM STATUS",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_orange'],
            font=('Consolas', 12, 'bold')
        ).pack(anchor=tk.W)
        
        # Animated status indicator with fixed width
        self.status_indicator = tk.Label(system_section,
            text="◢◣◤◥ NEURAL MATRIX ONLINE ◢◣◤◥",
            bg=self.colors['bg_panel'],
            fg=self.colors['neon_green'],
            font=('Consolas', 11, 'bold'),
            width=35,  # Fixed character width
            anchor='center'
        )
        self.status_indicator.pack(fill=tk.X, pady=2)
        
        # Cleanup button
        cleanup_frame = tk.Frame(parent, bg=self.colors['bg_panel'])
        cleanup_frame.pack(fill=tk.X, pady=10)
        
        cleanup_button = self.theme.create_cyber_button(cleanup_frame, 
            f"{self.symbols['skull']} MEMORY PURGE", self.cleanup_gpu_memory, 'error')
        cleanup_button.pack(fill=tk.X)
    
    def start_cyberpunk_animations(self):
        """Start cyberpunk animation effects"""
        self.animate_status_indicator()
        self.update_gpu_status_display()
    
    def animate_status_indicator(self):
        """Animate the status indicator"""
        def update_status():
            if not hasattr(self, 'status_indicator'):
                return
                
            frames = [
                "◢◣◤◥ NEURAL MATRIX ONLINE ◢◣◤◥",
                "▓▒░  NEURAL MATRIX ONLINE  ░▒▓",
                "◆◇◆  NEURAL MATRIX ONLINE  ◆◇◆",
                "▰▱▰  NEURAL MATRIX ONLINE  ▰▱▰"
            ]
            
            frame_idx = int(time.time() * 2) % len(frames)
            
            try:
                self.status_indicator.config(text=frames[frame_idx])
                self.root.after(500, update_status)
            except:
                pass  # Widget destroyed
        
        update_status()
    
    def update_gpu_status_display(self):
        """Update GPU status display"""
        def update_display():
            try:
                # Update GPU info
                if hasattr(self, 'sam_gpu_label'):
                    sam_device = gpu_manager.strategy.get('sam_device', 'N/A')
                    self.sam_gpu_label.config(text=f"◢ SAM CORE: {sam_device.upper()} ACTIVE ◣")
                
                if hasattr(self, 'yolo_gpu_label'):
                    yolo_device = gpu_manager.strategy.get('yolo_device', 'N/A')
                    self.yolo_gpu_label.config(text=f"◢ YOLO CORE: {yolo_device.upper()} ACTIVE ◣")
                
                # Update model status
                if hasattr(self, 'model_status_label'):
                    status = "ONLINE" if self.sam_model else "LOADING"
                    self.model_status_label.config(text=f"◢ MODEL STATUS: {status} ◣")
                
                # Update memory info
                self.update_gpu_memory_display()
                
                # Schedule next update
                self.root.after(2000, update_display)
                
            except Exception as e:
                # Silently continue if widgets are destroyed
                pass
        
        update_display()
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory with cyber feedback"""
        self.log_status("◢◣ INITIATING MEMORY PURGE SEQUENCE ◤◥")
        
        try:
            from gpu_manager import cleanup_all_gpu_memory
            cleanup_all_gpu_memory()
            self.log_status("✓ MEMORY PURGE COMPLETE - NEURAL CORES REFRESHED")
            
            # Update memory display
            self.update_gpu_memory_display()
            
        except Exception as e:
            self.log_status(f"⚠ PURGE SEQUENCE FAILED: {str(e)}")

    def update_gpu_memory_display(self):
        """Update GPU memory status display"""
        try:
            from gpu_manager import get_gpu_stats
            stats = get_gpu_stats()
            
            total_usage = sum(stat['current'] for stat in stats.values())
            if hasattr(self, 'gpu_memory_label'):
                self.gpu_memory_label.config(
                    text=f"◢ GPU MEMORY: {total_usage:.0f}MB ALLOCATED ◣"
                )
            
            if hasattr(self, 'renderer'):
                perf_stats = self.renderer.get_performance_stats()
                if perf_stats['count'] > 0 and hasattr(self, 'render_time_label'):
                    self.render_time_label.config(
                        text=f"◢ RENDER TIME: {perf_stats['avg_ms']:.1f}ms ◣"
                    )
                if hasattr(self, 'avg_render_label'):
                    self.avg_render_label.config(
                        text=f"◢ AVG RENDER: {perf_stats['avg_ms']:.1f}ms ◣"
                    )
            
        except Exception as e:
            if hasattr(self, 'gpu_memory_label'):
                self.gpu_memory_label.config(
                    text="◢ GPU MEMORY: SCAN ERROR ◣"
                )

def main():
    """Main function to run the SAM annotator"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="SAM Annotation Tool - Interactive image annotation with Segment Anything Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m sam_annotation                              # Start with directory picker
  python -m sam_annotation ./images                    # Start in ./images directory
  python -m sam_annotation --config my_config.yaml     # Start with custom config
  python -m sam_annotation --config cfg.yaml ./images  # Config + directory
        """
    )
    parser.add_argument(
        'directory',
        nargs='?',
        help='Directory containing images to annotate (optional - if not provided, directory picker will be shown)'
    )
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration YAML file (see config/default_config.yaml for template)'
    )

    args = parser.parse_args()

    # Load configuration if specified
    config = None
    if args.config:
        if ProjectConfig is not None:
            try:
                config = ProjectConfig.from_yaml(args.config)
                config.validate()
                print(f"Loaded configuration from: {args.config}")
                print(f"  Class names: {config.class_names}")
            except Exception as e:
                print(f"Error loading config: {e}")
                print("Starting with default configuration...")
        else:
            print("Warning: config module not available, ignoring --config")

    # Create and run the application
    root = tk.Tk()
    app = SAMAnnotator(root, initial_directory=args.directory, config=config)
    root.mainloop()

class TrainingParametersDialog:
    """Dialog for setting YOLO training parameters"""
    
    def __init__(self, parent, annotation_count):
        self.result = None
        self.epochs = 100
        self.image_size = 640
        self.batch_size = 8
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("YOLO Training Parameters")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (300 // 2)
        self.dialog.geometry(f"400x300+{x}+{y}")
        
        self.create_widgets(annotation_count)
        
    def create_widgets(self, annotation_count):
        """Create dialog widgets"""
        main_frame = tk.Frame(self.dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, 
            text="🏋️ YOLO Training Configuration",
            font=('Consolas', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Dataset info
        info_label = tk.Label(main_frame,
            text=f"📊 Dataset: {annotation_count} annotated images",
            font=('Consolas', 10))
        info_label.pack(pady=(0, 20))
        
        # Epochs
        epochs_frame = tk.Frame(main_frame)
        epochs_frame.pack(fill=tk.X, pady=5)
        tk.Label(epochs_frame, text="Epochs:", font=('Consolas', 10)).pack(side=tk.LEFT)
        self.epochs_var = tk.StringVar(value="100")
        epochs_entry = tk.Entry(epochs_frame, textvariable=self.epochs_var, width=10)
        epochs_entry.pack(side=tk.RIGHT)
        
        # Image size
        size_frame = tk.Frame(main_frame)
        size_frame.pack(fill=tk.X, pady=5)
        tk.Label(size_frame, text="Image Size:", font=('Consolas', 10)).pack(side=tk.LEFT)
        self.size_var = tk.StringVar(value="640")
        from tkinter import ttk as tkinter_ttk
        size_combo = tkinter_ttk.Combobox(size_frame, textvariable=self.size_var, 
                                values=["416", "512", "640", "800", "1024"], width=8, state="readonly")
        size_combo.pack(side=tk.RIGHT)
        
        # Batch size
        batch_frame = tk.Frame(main_frame)
        batch_frame.pack(fill=tk.X, pady=5)
        tk.Label(batch_frame, text="Batch Size:", font=('Consolas', 10)).pack(side=tk.LEFT)
        self.batch_var = tk.StringVar(value="8")
        batch_combo = tkinter_ttk.Combobox(batch_frame, textvariable=self.batch_var, 
                                 values=["1", "2", "4", "8", "16", "32"], width=8, state="readonly")
        batch_combo.pack(side=tk.RIGHT)
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=self.cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        start_btn = tk.Button(button_frame, text="Start Training", command=self.start_training)
        start_btn.pack(side=tk.RIGHT)
        
    def start_training(self):
        """Start training with selected parameters"""
        try:
            self.epochs = int(self.epochs_var.get())
            self.image_size = int(self.size_var.get())
            self.batch_size = int(self.batch_var.get())
            
            if self.epochs < 1 or self.epochs > 1000:
                messagebox.showerror("Invalid Parameters", "Epochs must be between 1 and 1000")
                return
                
            self.result = True
            self.dialog.destroy()
            
        except ValueError:
            messagebox.showerror("Invalid Parameters", "Please enter valid numeric values")
    
    def cancel(self):
        """Cancel dialog"""
        self.result = False
        self.dialog.destroy()


if __name__ == "__main__":
    main()