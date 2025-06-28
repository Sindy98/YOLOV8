import os
import zipfile
import yaml
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from backend.models import TrainingState
from backend.visualizer import draw_labels_on_image
from backend.dataio import load_yaml, save_yaml, list_images, list_labels, load_image

class DatasetManager:
    """Manages dataset operations"""

    def __init__(self, target_dir="datasets"):
        self.target_dir = target_dir
        os.makedirs(target_dir, exist_ok=True)
        self.current_dataset_info = None
        self.current_image_index = 0
        self.current_split = "train"  # "train" or "val"
        self.include_labels = True  # Whether to include labels in the image display

    def process_dataset(self, file_path):
        """Process dataset file (zip or yaml)"""
        if file_path.endswith('.zip'):
            return self._process_zip_dataset(file_path)
        elif file_path.endswith('.yaml'):
            return self._process_yaml_dataset(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .zip or .yaml file")

    def _process_zip_dataset(self, zip_path):
        """Process zip dataset"""
        zip_name = os.path.basename(zip_path).replace(".zip", "")
        data_folder = self._get_unique_folder_name(zip_name)

        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_folder)

        # Validate dataset
        if self._validate_dataset(data_folder):
            data_yaml_path = os.path.join(data_folder, "data.yaml")
            result = {
                "success": True,
                "data_path": data_yaml_path,
                "folder": data_folder
            }
            # Store dataset info for visualization
            self.current_dataset_info = self._get_dataset_info(data_folder)
            self.current_image_index = 0  # Reset to first image
            return result
        else:
            raise ValueError("Invalid dataset structure")

    def _process_yaml_dataset(self, yaml_path):
        """Process yaml dataset"""
        data_yaml = load_yaml(yaml_path)
        data_name = data_yaml['path']
        data_folder = os.path.join(self.target_dir, data_name)

        if self._validate_dataset(data_folder):
            result = {
                "success": True,
                "data_path": yaml_path,
                "folder": data_folder
            }
            # Store dataset info for visualization
            self.current_dataset_info = self._get_dataset_info(data_folder)
            self.current_image_index = 0  # Reset to first image
            return result
        else:
            raise ValueError("Invalid dataset structure")

    def _get_unique_folder_name(self, base_name):
        """Get unique folder name to avoid conflicts"""
        folder_path = os.path.join(self.target_dir, base_name)
        if os.path.exists(folder_path):
            i = 1
            while os.path.exists(os.path.join(self.target_dir, f"{base_name}_{i}")):
                i += 1
            folder_path = os.path.join(self.target_dir, f"{base_name}_{i}")

        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def _validate_dataset(self, data_folder):
        """Validate dataset structure"""
        data_yaml_path = os.path.join(data_folder, "data.yaml")
        if not os.path.exists(data_yaml_path):
            return False

        data_yaml = load_yaml(data_yaml_path)
        required_keys = ["train", "path"]
        if not all(key in data_yaml for key in required_keys):
            return False

        # Update path if needed
        if data_yaml["path"] != os.path.basename(data_folder):
            data_yaml["path"] = os.path.basename(data_folder)
            save_yaml(data_yaml, data_yaml_path)

        train_path = os.path.join(data_folder, data_yaml["train"])
        labels_path = os.path.join(data_folder, "labels")

        if not os.path.exists(train_path) or not os.path.exists(labels_path):
            return False

        image_count = len(list_images(train_path))
        label_count = len(list_labels(labels_path))

        if image_count == 0 or label_count == 0:
            return False

        if image_count != label_count:
            return False

        return True

    def _get_dataset_info(self, data_folder):
        """Get comprehensive dataset information"""
        data_yaml_path = os.path.join(data_folder, "data.yaml")
        data_yaml = load_yaml(data_yaml_path)
        train_path = os.path.join(data_folder, data_yaml["train"])
        val_path = os.path.join(data_folder, data_yaml.get("val", ""))
        labels_path = os.path.join(data_folder, "labels")
        train_images = list_images(train_path)
        val_images = list_images(val_path) if val_path and os.path.exists(val_path) else []
        label_files = list_labels(labels_path)
        class_names = data_yaml.get("names", [])
        return {
            "folder": data_folder,
            "data_yaml": data_yaml,
            "train_images": train_images,
            "val_images": val_images,
            "label_files": label_files,
            "class_names": class_names,
            "train_path": train_path,
            "val_path": val_path,
            "labels_path": labels_path
        }

    def get_dataset_summary(self):
        """Get dataset summary for display"""
        if not self.current_dataset_info:
            return None

        info = self.current_dataset_info
        # Convert class_names dict to a comma-separated string of values
        class_names_str = ", ".join(str(v) for v in info["class_names"].values()) if isinstance(info["class_names"], dict) else str(info["class_names"])
        return {
            "dataset_name": os.path.basename(info["folder"]),
            "total_images": len(info["train_images"]) + len(info["val_images"]),
            "train_images": len(info["train_images"]),
            "val_images": len(info["val_images"]),
            "total_labels": len(info["label_files"]),
            "classes": len(info["class_names"]),
            "class_names": class_names_str
        }

    def get_current_image_info(self):
        """Get information about current image position"""
        if not self.current_dataset_info:
            return None

        info = self.current_dataset_info
        current_images = info["train_images"] if self.current_split == "train" else info["val_images"]

        if not current_images:
            return None

        return {
            "current_index": self.current_image_index,
            "total_images": len(current_images),
            "current_split": self.current_split,
            "current_filename": current_images[self.current_image_index] if self.current_image_index < len(current_images) else None
        }

    def get_current_image(self):
        """Get current image with optional labels"""
        if not self.current_dataset_info:
            return None

        info = self.current_dataset_info
        current_images = info["train_images"] if self.current_split == "train" else info["val_images"]
        current_path = info["train_path"] if self.current_split == "train" else info["val_path"]

        if not current_images or self.current_image_index >= len(current_images):
            return None

        img_name = current_images[self.current_image_index]
        img_path = os.path.join(current_path, img_name)
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(info["labels_path"], label_name)

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add labels if requested
        if self.include_labels and os.path.exists(label_path):
            img = draw_labels_on_image(img, label_path, info["class_names"])

        return {
            "image": img,
            "filename": img_name,
            "path": img_path,
            "index": self.current_image_index,
            "total": len(current_images)
        }

    def navigate_image(self, direction):
        """Navigate to next/previous image"""
        if not self.current_dataset_info:
            return None

        info = self.current_dataset_info
        current_images = info["train_images"] if self.current_split == "train" else info["val_images"]

        if not current_images:
            return None

        if direction == "next":
            self.current_image_index = min(self.current_image_index + 1, len(current_images) - 1)
        elif direction == "prev":
            self.current_image_index = max(self.current_image_index - 1, 0)
        elif direction == "first":
            self.current_image_index = 0
        elif direction == "last":
            self.current_image_index = len(current_images) - 1

        return self.get_current_image()

    def switch_split(self, split):
        """Switch between train and validation splits"""
        if split not in ["train", "val"]:
            return None

        self.current_split = split
        self.current_image_index = 0  # Reset to first image of new split
        return self.get_current_image()

    def clear_current_dataset(self):
        """Clear current dataset info"""
        self.current_dataset_info = None
        self.current_image_index = 0
        self.current_split = "train"