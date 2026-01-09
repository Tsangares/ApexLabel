"""
Annotation Data Manager for SAM Annotation Tool

Handles loading, saving, statistics, and management of annotation data.
This module encapsulates all data persistence and statistics functionality
to keep the main annotator class focused on UI and interaction.
"""

import os
import json
import time
from pathlib import Path
from collections import defaultdict, Counter


class AnnotationDataManager:
    """Manages annotation data persistence and statistics."""
    
    def __init__(self, annotations_file="all_annotations.json"):
        self.annotations_file = annotations_file
        self.annotations_data = {}
        self.backup_interval = 300  # 5 minutes
        self.last_backup_time = 0
        
    def load_annotations(self, annotations_file=None):
        """Load annotations from JSON file"""
        if annotations_file:
            self.annotations_file = annotations_file
            
        try:
            if os.path.exists(self.annotations_file):
                with open(self.annotations_file, 'r') as f:
                    self.annotations_data = json.load(f)
                return True
            else:
                self.annotations_data = {}
                return False
        except Exception as e:
            print(f"Error loading annotations: {e}")
            self.annotations_data = {}
            return False
    
    def save_annotations(self, force_backup=False):
        """Save annotations to JSON file with automatic backup"""
        try:
            # Save main file
            with open(self.annotations_file, 'w') as f:
                json.dump(self.annotations_data, f, indent=2)
            
            # Create backup if needed
            current_time = time.time()
            if force_backup or (current_time - self.last_backup_time) > self.backup_interval:
                self.create_backup()
                self.last_backup_time = current_time
                
            return True
            
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False
    
    def create_backup(self):
        """Create a timestamped backup of annotations"""
        try:
            import shutil
            import datetime
            
            if os.path.exists(self.annotations_file):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{self.annotations_file}.backup_{timestamp}"
                shutil.copy2(self.annotations_file, backup_name)
                
                # Keep only last 10 backups
                self.cleanup_old_backups()
                
        except Exception as e:
            print(f"Error creating backup: {e}")
    
    def cleanup_old_backups(self, max_backups=10):
        """Clean up old backup files, keeping only the most recent ones"""
        try:
            backup_pattern = f"{self.annotations_file}.backup_"
            backup_files = []
            
            for file in os.listdir('.'):
                if file.startswith(backup_pattern):
                    backup_files.append((file, os.path.getmtime(file)))
            
            # Sort by modification time, newest first
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups
            for file, _ in backup_files[max_backups:]:
                try:
                    os.remove(file)
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"Error cleaning up backups: {e}")
    
    def get_image_annotations(self, image_path):
        """Get annotations for a specific image"""
        image_key = str(Path(image_path))
        return self.annotations_data.get(image_key, [])
    
    def set_image_annotations(self, image_path, annotations):
        """Set annotations for a specific image"""
        image_key = str(Path(image_path))
        self.annotations_data[image_key] = annotations
    
    def has_annotations(self, image_path):
        """Check if an image has any annotations"""
        return len(self.get_image_annotations(image_path)) > 0
    
    def get_annotated_images(self):
        """Get list of all images that have annotations"""
        return [path for path, annotations in self.annotations_data.items() 
                if annotations and len(annotations) > 0]
    
    def get_unannotated_images(self, all_image_paths):
        """Get list of images that don't have annotations"""
        annotated_set = set(self.get_annotated_images())
        return [path for path in all_image_paths if str(Path(path)) not in annotated_set]
    
    def get_statistics(self):
        """Get comprehensive statistics about annotations"""
        stats = {
            'total_images': len(self.annotations_data),
            'annotated_images': 0,
            'total_annotations': 0,
            'annotations_per_image': {},
            'label_distribution': Counter(),
            'bbox_size_stats': {'small': 0, 'medium': 0, 'large': 0},
            'recent_activity': self.get_recent_activity_stats()
        }
        
        for image_path, annotations in self.annotations_data.items():
            if annotations and len(annotations) > 0:
                stats['annotated_images'] += 1
                annotation_count = len(annotations)
                stats['total_annotations'] += annotation_count
                stats['annotations_per_image'][image_path] = annotation_count
                
                # Analyze individual annotations
                for annotation in annotations:
                    # Label distribution
                    label = annotation.get('label', 'unknown')
                    stats['label_distribution'][label] += 1
                    
                    # Bounding box size analysis
                    if 'bbox' in annotation:
                        bbox = annotation['bbox']
                        if len(bbox) >= 4:
                            width = abs(bbox[2] - bbox[0])
                            height = abs(bbox[3] - bbox[1])
                            area = width * height
                            
                            if area < 5000:
                                stats['bbox_size_stats']['small'] += 1
                            elif area < 20000:
                                stats['bbox_size_stats']['medium'] += 1
                            else:
                                stats['bbox_size_stats']['large'] += 1
        
        return stats
    
    def get_recent_activity_stats(self, hours=24):
        """Get statistics about recent annotation activity"""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        recent_stats = {
            'recent_annotations': 0,
            'recent_images': 0,
            'activity_timeline': defaultdict(int)
        }
        
        for image_path, annotations in self.annotations_data.items():
            image_has_recent = False
            
            for annotation in annotations:
                timestamp = annotation.get('timestamp', 0)
                if timestamp > cutoff_time:
                    recent_stats['recent_annotations'] += 1
                    image_has_recent = True
                    
                    # Group by hour for timeline
                    hour_key = int((timestamp - cutoff_time) // 3600)
                    recent_stats['activity_timeline'][hour_key] += 1
            
            if image_has_recent:
                recent_stats['recent_images'] += 1
        
        return recent_stats
    
    def export_statistics_report(self, output_file=None):
        """Export detailed statistics report"""
        if output_file is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"annotation_report_{timestamp}.json"
        
        try:
            stats = self.get_statistics()
            
            # Add metadata
            report = {
                'generated_at': time.time(),
                'annotations_file': self.annotations_file,
                'statistics': stats
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return output_file
            
        except Exception as e:
            print(f"Error exporting statistics: {e}")
            return None
    
    def merge_annotations(self, other_annotations_file):
        """Merge annotations from another file"""
        try:
            with open(other_annotations_file, 'r') as f:
                other_data = json.load(f)
            
            merged_count = 0
            conflicts = []
            
            for image_path, annotations in other_data.items():
                if image_path in self.annotations_data:
                    if self.annotations_data[image_path] != annotations:
                        conflicts.append(image_path)
                    # Keep existing annotations in case of conflict
                else:
                    self.annotations_data[image_path] = annotations
                    merged_count += 1
            
            return {
                'merged_count': merged_count,
                'conflicts': conflicts,
                'total_processed': len(other_data)
            }
            
        except Exception as e:
            print(f"Error merging annotations: {e}")
            return None
    
    def validate_annotations(self):
        """Validate annotation data integrity"""
        issues = {
            'missing_files': [],
            'invalid_bboxes': [],
            'missing_labels': [],
            'duplicate_entries': []
        }
        
        seen_images = set()
        
        for image_path, annotations in self.annotations_data.items():
            # Check for duplicate entries
            if image_path in seen_images:
                issues['duplicate_entries'].append(image_path)
            seen_images.add(image_path)
            
            # Check if file exists
            if not os.path.exists(image_path):
                issues['missing_files'].append(image_path)
            
            # Validate individual annotations
            for i, annotation in enumerate(annotations):
                annotation_id = f"{image_path}[{i}]"
                
                # Check for missing labels
                if 'label' not in annotation or not annotation['label']:
                    issues['missing_labels'].append(annotation_id)
                
                # Validate bounding boxes
                if 'bbox' in annotation:
                    bbox = annotation['bbox']
                    if (not isinstance(bbox, list) or len(bbox) != 4 or 
                        not all(isinstance(x, (int, float)) for x in bbox)):
                        issues['invalid_bboxes'].append(annotation_id)
                    else:
                        # Check bbox coordinates make sense
                        x1, y1, x2, y2 = bbox
                        if x1 >= x2 or y1 >= y2:
                            issues['invalid_bboxes'].append(annotation_id)
        
        return issues
    
    def cleanup_invalid_annotations(self, dry_run=True):
        """Clean up invalid annotations"""
        issues = self.validate_annotations()
        cleanup_report = {
            'removed_images': [],
            'fixed_annotations': [],
            'unfixable_issues': []
        }
        
        if dry_run:
            return issues  # Just return issues without making changes
        
        # Remove entries for missing files
        for image_path in issues['missing_files']:
            if image_path in self.annotations_data:
                del self.annotations_data[image_path]
                cleanup_report['removed_images'].append(image_path)
        
        # Try to fix invalid bboxes or remove them
        for annotation_id in issues['invalid_bboxes']:
            # This would require more complex logic to fix
            cleanup_report['unfixable_issues'].append(annotation_id)
        
        return cleanup_report