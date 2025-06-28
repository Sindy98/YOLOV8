import os
import yaml
import cv2
import shutil

def load_yaml(file_path):
    """Load YAML file and return as dict."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    """Save dict as YAML file."""
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

def list_images(folder, exts=('.jpg', '.jpeg', '.png')):
    """List image files in a folder, sorted."""
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

def list_labels(folder, ext='.txt'):
    """List label files in a folder, sorted."""
    return sorted([f for f in os.listdir(folder) if f.endswith(ext)])

def load_image(path):
    """Load an image as RGB numpy array."""
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_image(path, img):
    """Save an RGB numpy array as an image."""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

def list_datasets(base_dir="datasets"):
    """Return a list of dataset folder names that contain a data.yaml."""
    datasets = []
    for name in os.listdir(base_dir):
        folder = os.path.join(base_dir, name)
        if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "data.yaml")):
            datasets.append(name)
    return sorted(datasets)

def delete_dataset(folder):
    """Delete the entire dataset folder."""
    if os.path.exists(folder) and os.path.isdir(folder):
        shutil.rmtree(folder)
        return True
    return False