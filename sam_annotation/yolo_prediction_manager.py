"""
YOLO Prediction Manager for SAM Annotation Tool

Handles YOLO model prediction, caching, training, and dataset export.
This module encapsulates all YOLO-related functionality to keep the main
annotator class focused on UI and SAM operations.
"""

import os
import time
import json
import threading
import shutil
from pathlib import Path
import cv2
import numpy as np
from .gpu_manager import yolo_device_context


class YOLOPredictionManager:
    """Manages YOLO predictions, caching, and training operations."""

    def __init__(self, model_path=None, config=None):
        # Store configuration
        self.config = config
        self.models_dir = config.models_dir if config else "./models"
        self.output_dir = config.output_dir if config else "./output"

        # Default model selection
        if model_path is None:
            model_path = self._get_default_model_path()

        self.model_path = model_path
        self.predicted_annotations = {}  # Cache for predicted annotations
        self.model = None
        self.is_enabled = False

        # Cache file location based on config
        self.cache_file = os.path.join(self.output_dir, "yolo_predictions_cache.json")
        self._load_persistent_cache()

    def _get_default_model_path(self):
        """Get the best available model"""
        # Priority order: current model, models directory, base model
        candidates = [
            os.path.join(self.models_dir, "current_best_yolo.pt"),
            os.path.join(self.models_dir, "yolo_models", "*", "weights", "best.pt"),
            "yolo11m.pt"  # Base model as fallback
        ]

        for model_path in candidates:
            if '*' in model_path:
                import glob
                matches = glob.glob(model_path)
                if matches:
                    return matches[0]
            elif os.path.exists(model_path):
                return model_path

        return "yolo11m.pt"  # Final fallback

    def set_model_path(self, model_path):
        """Change the model being used"""
        if model_path != self.model_path:
            self.model_path = model_path
            self.model = None  # Force reload
            # Clear cache for different model
            self.predicted_annotations.clear()
            print(f"Switched to model: {model_path}")

    def get_available_models(self):
        """Get list of available YOLO models"""
        models = []
        import glob

        # Model locations based on config
        model_locations = [
            os.path.join(self.models_dir, "current_best_yolo.pt"),
            os.path.join(self.models_dir, "yolo_models", "*", "weights", "best.pt"),
        ]

        for pattern in model_locations:
            if '*' in pattern:
                for path in glob.glob(pattern):
                    if os.path.exists(path):
                        models.append({
                            'path': path,
                            'name': f"Trained Model {Path(path).parent.parent.name}",
                            'size_mb': os.path.getsize(path) / (1024*1024)
                        })
            else:
                if os.path.exists(pattern):
                    models.append({
                        'path': pattern,
                        'name': "Current Best Model",
                        'size_mb': os.path.getsize(pattern) / (1024*1024)
                    })

        # Add base models
        base_models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"]
        for base_model in base_models:
            models.append({
                'path': base_model,
                'name': f"Base {base_model.upper()}",
                'size_mb': 0  # Will be downloaded
            })

        return models
        
    def _load_persistent_cache(self):
        """Load predictions cache from JSON file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Validate cache data structure
                if isinstance(cache_data, dict) and 'model_path' in cache_data and 'predictions' in cache_data:
                    # Only load if cache is from the same model
                    if cache_data['model_path'] == self.model_path:
                        self.predicted_annotations = cache_data['predictions']
                        print(f"Loaded {len(self.predicted_annotations)} cached predictions from {self.cache_file}")
                    else:
                        print(f"Cache is for different model ({cache_data['model_path']}), starting fresh")
                else:
                    print("Invalid cache format, starting with empty cache")
        except Exception as e:
            print(f"Error loading cache: {e}")
            
    def _save_persistent_cache(self):
        """Save predictions cache to JSON file"""
        try:
            cache_data = {
                'model_path': self.model_path,
                'created_at': time.time(),
                'predictions': self.predicted_annotations
            }
            
            # Create backup of existing cache
            if os.path.exists(self.cache_file):
                backup_file = f"{self.cache_file}.backup"
                shutil.copy2(self.cache_file, backup_file)
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            print(f"Saved {len(self.predicted_annotations)} predictions to cache")
        except Exception as e:
            print(f"Error saving cache: {e}")
        
    def set_enabled(self, enabled):
        """Enable or disable prediction caching"""
        self.is_enabled = enabled
        
    def is_model_available(self):
        """Check if YOLO model is available"""
        return Path(self.model_path).exists()
        
    def load_model(self):
        """Load YOLO model for predictions"""
        if not self.is_model_available():
            return False
            
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False
            
    def predict_image(self, image_path, confidence_threshold=0.2):
        """Predict bounding boxes for a single image"""
        try:
            if self.model is None and not self.load_model():
                return []
                
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            
            # Get predictions with GPU context
            with yolo_device_context() as device:
                results = self.model(image, conf=confidence_threshold, device=device)
            
            # Convert to our format
            predictions = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    predictions.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else 0
                    })
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error for {image_path}: {e}")
            return []
    
    def cache_prediction(self, image_path, force=False):
        """Cache YOLO prediction for a specific image"""
        if not self.is_enabled and not force:
            return None
            
        image_name = os.path.basename(image_path)
        
        # Skip if already cached (unless forced)
        if not force and image_name in self.predicted_annotations:
            return self.predicted_annotations[image_name]['predictions']
        
        # Get predictions
        predictions = self.predict_image(image_path)
        
        # Cache the predictions
        self.predicted_annotations[image_name] = {
            'predictions': predictions,
            'timestamp': time.time()
        }
        
        # Save to persistent cache periodically (every 10 new predictions)
        if len(self.predicted_annotations) % 10 == 0:
            self._save_persistent_cache()
        
        return predictions
    
    def get_cached_predictions(self, image_path):
        """Get cached predictions for an image"""
        image_name = os.path.basename(image_path)
        if image_name in self.predicted_annotations:
            return self.predicted_annotations[image_name]['predictions']
        return []
    
    def cache_predictions_batch(self, image_paths, progress_callback=None, complete_callback=None):
        """Cache predictions for multiple images in background"""
        def cache_thread():
            try:
                for i, image_path in enumerate(image_paths):
                    self.cache_prediction(image_path, force=True)
                    
                    if progress_callback:
                        progress_callback(i + 1, len(image_paths))
                
                if complete_callback:
                    complete_callback(f"Batch prediction completed: {len(image_paths)} images processed")
                    
            except Exception as e:
                if complete_callback:
                    complete_callback(f"Batch prediction error: {str(e)}")
        
        thread = threading.Thread(target=cache_thread, daemon=True)
        thread.start()
        return thread
    
    def predict_all_images(self, image_paths, progress_callback=None, complete_callback=None):
        """Predict all images with detailed progress tracking"""
        def predict_all_thread():
            try:
                total_images = len(image_paths)
                successful_predictions = 0
                failed_predictions = 0
                
                for i, image_path in enumerate(image_paths):
                    try:
                        predictions = self.cache_prediction(image_path, force=True)
                        if predictions is not None and len(predictions) >= 0:  # Accept empty predictions as success
                            successful_predictions += 1
                        else:
                            failed_predictions += 1
                            
                    except Exception as e:
                        failed_predictions += 1
                        print(f"Failed to predict {image_path}: {e}")
                    
                    if progress_callback:
                        progress_callback(i + 1, total_images, successful_predictions, failed_predictions)
                
                # Save cache after completing all predictions
                self._save_persistent_cache()
                
                if complete_callback:
                    complete_callback(f"Prediction completed: {successful_predictions} successful, {failed_predictions} failed")
                    
            except Exception as e:
                if complete_callback:
                    complete_callback(f"Predict all error: {str(e)}")
        
        thread = threading.Thread(target=predict_all_thread, daemon=True)
        thread.start()
        return thread
    
    def predict_current_only(self, image_path, result_callback=None):
        """Predict current image only and return results immediately"""
        def predict_thread():
            try:
                predictions = self.predict_image(image_path)
                
                # Also cache it
                image_name = os.path.basename(image_path)
                self.predicted_annotations[image_name] = {
                    'predictions': predictions,
                    'timestamp': time.time()
                }
                
                if result_callback:
                    result_callback(predictions)
                    
            except Exception as e:
                if result_callback:
                    result_callback([])
                print(f"Current prediction error: {e}")
        
        thread = threading.Thread(target=predict_thread, daemon=True)
        thread.start()
        return thread
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.predicted_annotations.clear()
        # Also clear persistent cache
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("Cleared both memory and persistent prediction cache")
    
    def get_cache_stats(self):
        """Get cache statistics"""
        return {
            'count': len(self.predicted_annotations),
            'size_bytes': sum(len(str(v)) for v in self.predicted_annotations.values()),
            'oldest_timestamp': min((v['timestamp'] for v in self.predicted_annotations.values()), default=0),
            'newest_timestamp': max((v['timestamp'] for v in self.predicted_annotations.values()), default=0)
        }
    
    def export_yolo_dataset(self, annotations_data, output_dir, train_split=0.8):
        """Export annotations to YOLO format dataset"""
        try:
            from pathlib import Path
            import shutil
            import random
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create directory structure
            (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
            (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
            
            # Split data
            image_files = list(annotations_data.keys())
            random.shuffle(image_files)
            split_idx = int(len(image_files) * train_split)
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            stats = {'train': 0, 'val': 0, 'total_annotations': 0}
            
            # Process training files
            for image_file in train_files:
                self._process_yolo_file(image_file, annotations_data[image_file], 
                                      output_path, 'train', stats)
            
            # Process validation files
            for image_file in val_files:
                self._process_yolo_file(image_file, annotations_data[image_file], 
                                      output_path, 'val', stats)
            
            # Create dataset.yaml
            self._create_dataset_yaml(output_path)
            
            return stats
            
        except Exception as e:
            print(f"Dataset export error: {e}")
            return None
    
    def _process_yolo_file(self, image_file, annotations, output_path, split, stats):
        """Process a single file for YOLO dataset export"""
        try:
            import shutil
            
            # Copy image
            src_image = Path(image_file)
            if src_image.exists():
                dst_image = output_path / 'images' / split / src_image.name
                shutil.copy2(src_image, dst_image)
                stats[split] += 1
                
                # Create label file
                label_file = output_path / 'labels' / split / f"{src_image.stem}.txt"
                
                with open(label_file, 'w') as f:
                    for annotation in annotations:
                        if 'bbox' in annotation:
                            bbox = annotation['bbox']
                            # Convert to YOLO format (normalized center coordinates)
                            img = cv2.imread(str(src_image))
                            if img is not None:
                                h, w = img.shape[:2]
                                x1, y1, x2, y2 = bbox
                                
                                # Convert to center coordinates and normalize
                                x_center = ((x1 + x2) / 2) / w
                                y_center = ((y1 + y2) / 2) / h
                                width = (x2 - x1) / w
                                height = (y2 - y1) / h
                                
                                # Class ID (0 for single class)
                                class_id = annotation.get('class_id', 0)
                                
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                                stats['total_annotations'] += 1
                                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    def _create_dataset_yaml(self, output_path):
        """Create dataset.yaml file for YOLO training"""
        # Get class names from config or use default
        class_names = self.config.class_names if self.config and self.config.class_names else ['object']
        yaml_content = f"""# YOLO Dataset Configuration
path: {output_path}
train: images/train
val: images/val

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""
        
        with open(output_path / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
    
    def train_model(self, dataset_yaml_path, epochs=100, image_size=640, batch_size=None, 
                   project_path=None, model_name=None, progress_callback=None):
        """Train YOLO model on dataset"""
        try:
            from ultralytics import YOLO
            import datetime
            
            # Generate unique model name with timestamp
            if model_name is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f'yolo_sam_{timestamp}'
            
            if project_path is None:
                project_path = os.path.join(self.models_dir, 'yolo_models')
            
            # Get optimal batch size if not specified
            if batch_size is None:
                from .gpu_manager import gpu_manager
                batch_size = gpu_manager.get_optimal_batch_size('yolo_training', base_size=4)
            
            def train_thread():
                try:
                    with yolo_device_context() as device:
                        # Load YOLO model - use yolo11m for consistency with main project
                        model = YOLO('yolo11m.pt')  # Use medium model for better performance
                        
                        # Train the model
                        results = model.train(
                            data=str(dataset_yaml_path),
                            epochs=epochs,
                            imgsz=image_size,
                            batch=batch_size,
                            device=device,
                            project=str(project_path),
                            name=model_name,
                            patience=20,  # Early stopping
                            save_period=10  # Save checkpoints
                        )
                    
                    # Copy best model to easily accessible location
                    import shutil
                    best_model_path = Path(f'{project_path}/{model_name}/weights/best.pt')
                    if best_model_path.exists():
                        current_best = Path(self.models_dir) / 'current_best_yolo.pt'
                        shutil.copy2(str(best_model_path), str(current_best))
                        self.model_path = str(current_best)
                        self.model = None  # Force reload next time
                        
                        if progress_callback:
                            progress_callback(f"Training completed! Best model saved as: current_best_yolo.pt")
                        return True
                    else:
                        if progress_callback:
                            progress_callback("Training completed but best model not found")
                        return False
                        
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Training error: {str(e)}")
                    return False
            
            # Run training in background thread
            train_thread_obj = threading.Thread(target=train_thread, daemon=True)
            train_thread_obj.start()
            
            if progress_callback:
                progress_callback("YOLO training started in background...")
            
            return train_thread_obj
            
        except ImportError:
            if progress_callback:
                progress_callback("Error: ultralytics not installed. Run: pip install ultralytics")
            return None
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error starting YOLO training: {str(e)}")
            return None