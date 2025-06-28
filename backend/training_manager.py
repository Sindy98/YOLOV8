import threading
import os
from ultralytics import YOLO
from backend.models import TrainingState, TrainingConfig
from backend.dataset_manager import DatasetManager

class TrainingManager:
    """Manages YOLO training operations"""

    def __init__(self):
        self.state = TrainingState()
        self.config = TrainingConfig()
        self.dataset_manager = DatasetManager()
        self.training_thread = None

    def get_status(self):
        """Get current training status"""
        return {
            "is_running": self.state.is_running,
            "progress": self.state.progress,
            "logs": self.state.logs[-10:],  # Last 10 log entries
            "model_path": self.state.model_path
        }

    def get_config(self):
        """Get current training configuration"""
        return self.config.to_dict()

    def update_config(self, config_data):
        """Update training configuration"""
        if self.state.is_running:
            raise ValueError("Cannot update config while training is running")

        for key, value in config_data.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def is_running(self):
        """Check if training is currently running"""
        return self.state.is_running

    def start_training(self):
        """Start the training process in a separate thread"""
        if self.state.is_running:
            raise ValueError("Training already in progress")

        self.training_thread = threading.Thread(target=self._train_yolo)
        self.training_thread.daemon = True
        self.training_thread.start()

    def stop_training(self):
        """Stop the training process"""
        self.state.is_running = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)

    def get_model_path(self):
        """Get the path to the trained model"""
        return self.state.model_path

    def clear_model(self):
        """Clear the trained model"""
        if self.state.model_path and os.path.exists(self.state.model_path):
            os.remove(self.state.model_path)
        self.state.model_path = None

    def _train_yolo(self):
        """Internal training method"""
        self.state.is_running = True
        self.state.progress = 0
        self.state.logs = ["Training started..."]

        def on_train_epoch_end(data):
            epoch = data.epoch
            total_epochs = data.epochs
            if epoch is not None and total_epochs:
                percent = int((epoch + 1) / total_epochs * 100)
                self.state.progress = percent

        try:
            model = YOLO(f"{self.config.model}.pt")
            self.state.logs.append(f"Model {self.config.model} loaded successfully.")

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            model.train(
                data=self.config.data_path,
                epochs=self.config.epochs,
                imgsz=self.config.imgsz,
                batch=self.config.batch,
                project=self.config.project,
            )

            model.export(format='onnx')
            self.state.model_path = str(model.trainer.best)[:-2] + "onnx"
            self.state.logs.append(f"Training completed! Model saved to {self.state.model_path}.")

        except Exception as e:
            self.state.logs.append(f"Error: {str(e)}")
        finally:
            self.state.is_running = False
            self.state.progress = 100