from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class TrainingState:
    """Training state model"""
    is_running: bool = False
    progress: int = 0
    logs: List[str] = None
    model_path: Optional[str] = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []

@dataclass
class TrainingConfig:
    """Training configuration model"""
    data_path: str = "datasets/dataset2024070401/data.yaml"
    model: str = "yolov8n-seg"
    epochs: int = 2
    imgsz: int = 640
    batch: int = 8
    project: str = "runs"

    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)