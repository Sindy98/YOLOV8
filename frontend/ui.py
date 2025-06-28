import gradio as gr
import os
import shutil
import numpy as np
from backend.training_manager import TrainingManager
from backend.dataio import list_datasets
from backend.dataset_manager import DatasetManager

class YOLOTrainingUI:
    """Gradio UI for YOLO training"""

    def __init__(self):
        self.training_manager = TrainingManager()
        self.dataset_manager = DatasetManager()
        self.selected_dataset = None

    def create_ui(self):
        """Create the Gradio UI"""

        with gr.Blocks(css="""
        #row1 {height: 10vh;}
        #data_folder_input { height: 15vh !important; }
        #file_output { height: 15vh !important; }
        .dataset-info { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        .navigation-controls { display: flex; justify-content: center; align-items: center; gap: 10px; margin: 10px 0; }
        .image-info { text-align: center; margin: 10px 0; font-weight: bold; }
        """) as app:

            gr.Markdown("## YOLOv8 Training Monitor")

            # Configuration row
            with gr.Row(equal_height=True, elem_id="row1"):
                epoch_input = gr.Number(label="Epochs", value=2, precision=0, scale=1)
                status_label = gr.Textbox(label="Status", value="Ready", interactive=False, scale=1)
                progress_label = gr.Textbox(label="Progress", value="0%", interactive=False, scale=1)

            # File upload and controls row
            with gr.Row(equal_height=True, elem_id="row2", height="15vh"):
                data_folder_input = gr.File(
                    label="Upload .zip or data.yaml under dataset",
                    file_count="single",
                    scale=1,
                    elem_id="data_folder_input"
                )
                start_btn = gr.Button("Start Training", variant="primary", scale=1, interactive=False)
                file_output = gr.File(label="Download Model", interactive=False, scale=1, elem_id="file_output")

            # Dataset Visualizer Section
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Information")
                    dataset_info_display = gr.HTML(value="<div class='dataset-info'>No dataset loaded</div>")

                    # Split selector
                    split_selector = gr.Radio(
                        choices=["train", "val"],
                        value="train",
                        label="Dataset Split",
                        interactive=True
                    )

                    # Image navigation info
                    image_info_display = gr.HTML(value="<div class='image-info'>No images loaded</div>")

                    # Dataset selection dropdown
                    dataset_selector = gr.Dropdown(
                        choices=list_datasets(),
                        label="Select Dataset",
                        interactive=True
                    )

                    # Delete button
                    delete_btn = gr.Button("Delete Selected Dataset", variant="stop")

                with gr.Column(scale=2):
                    gr.Markdown("### Image Viewer")

                    # Current image display
                    current_image_display = gr.Image(
                        label="Current Image",
                        show_label=True,
                        height=400
                    )

                    # Navigation controls
                    with gr.Row():
                        first_btn = gr.Button("⏮️ First", size="sm")
                        prev_btn = gr.Button("⬅️ Previous", size="sm")
                        next_btn = gr.Button("Next ➡️", size="sm")
                        last_btn = gr.Button("Last ⏭️", size="sm")

                    # Label toggle
                    show_labels_btn = gr.Button("Show/Hide Labels", variant="secondary")

            # Console and controls
            console = gr.Textbox(label="Training Logs", lines=20, interactive=False)
            refresh_btn = gr.Button("Refresh Status")

            # Event handlers
            def update_epochs(val):
                """Update epochs configuration"""
                try:
                    if val < 1:
                        val = 1
                    self.training_manager.update_config({"epochs": int(val)})
                except Exception as e:
                    console.update(value=f"Error updating epochs: {str(e)}")

            def handle_dataset_upload(file):
                """Handle dataset upload"""
                try:
                    if file is None:
                        return self._get_empty_upload_outputs()

                    # Save uploaded file temporarily
                    temp_path = file.name
                    result = self.training_manager.dataset_manager.process_dataset(temp_path)

                    if result.get("success"):
                        # Update config with new data path
                        self.training_manager.update_config({"data_path": result["data_path"]})

                        # Update dataset info and current image
                        dataset_info = self._get_dataset_info_html()
                        current_image = self._get_current_image()
                        image_info = self._get_image_info_html()

                        return (
                            gr.update(interactive=True),
                            dataset_info,
                            current_image,
                            image_info
                        )
                    else:
                        error_msg = f"Dataset upload failed: {result.get('error', 'Unknown error')}"
                        console.update(value=error_msg)
                        return self._get_empty_upload_outputs()
                except Exception as e:
                    error_msg = f"Error uploading dataset: {str(e)}"
                    console.update(value=error_msg)
                    return self._get_empty_upload_outputs()

            def handle_dataset_clear():
                """Handle dataset clear"""
                return self._get_empty_upload_outputs()

            def handle_model_clear():
                """Handle model clear"""
                try:
                    self.training_manager.clear_model()
                    return "Model cleared", gr.update(interactive=False, value=None)
                except Exception as e:
                    return f"Error clearing model: {str(e)}", gr.update(interactive=False, value=None)

            def start_training():
                """Start training process"""
                try:
                    self.training_manager.start_training()
                    return "Started", "0%", "Initializing..."
                except Exception as e:
                    return "Error", "0%", f"Failed to start training: {str(e)}"

            def refresh_status():
                """Refresh training status"""
                try:
                    status = self.training_manager.get_status()
                    progress_text = f"{status['progress']}%"
                    logs = "\n".join(status['logs'])

                    # Update file output if model is ready
                    file_output_state = gr.update(interactive=False, value=None)
                    if status['progress'] == 100 and status.get('model_path'):
                        file_output_state = gr.update(interactive=True, value=status['model_path'])

                    return status['is_running'] and "Running" or "Ready", progress_text, logs, file_output_state
                except Exception as e:
                    return "Error", "0%", f"Failed to refresh status: {str(e)}", gr.update(interactive=False, value=None)

            # Navigation event handlers
            def navigate_image(direction):
                """Navigate to next/previous image"""
                try:
                    result = self.training_manager.dataset_manager.navigate_image(direction)
                    if result:
                        return result["image"], self._get_image_info_html()
                    else:
                        return None, self._get_image_info_html()
                except Exception as e:
                    console.update(value=f"Error navigating: {str(e)}")
                    return None, self._get_image_info_html()

            def switch_split(split):
                """Switch between train and validation splits"""
                try:
                    result = self.training_manager.dataset_manager.switch_split(split)
                    if result:
                        return result["image"], self._get_image_info_html()
                    else:
                        return None, self._get_image_info_html()
                except Exception as e:
                    console.update(value=f"Error switching split: {str(e)}")
                    return None, self._get_image_info_html()

            def toggle_labels():
                """Toggle label display"""

                self.training_manager.dataset_manager.include_labels = not self.training_manager.dataset_manager.include_labels
                return self._get_current_image()

            def on_select_dataset(dataset_name):
                if not dataset_name:
                    return "No dataset selected"
                folder = os.path.join("datasets", dataset_name)
                # Load dataset info (simulate upload)
                self.dataset_manager.process_dataset(os.path.join(folder, "data.yaml"))
                self.selected_dataset = folder
                # Return info for UI update (e.g., dataset info, images, etc.)
                return self._get_dataset_info_html(), self._get_current_image(), self._get_image_info_html()

            def on_delete_dataset():
                if self.selected_dataset:
                    from backend.dataio import delete_dataset
                    delete_dataset(self.selected_dataset)
                    self.selected_dataset = None
                    # Update dataset list and clear info
                    return gr.update(choices=list_datasets()), "<div>No dataset selected</div>", None, "<div>No image</div>"
                return gr.update(), None, None, None

            # Bind events
            epoch_input.change(fn=update_epochs, inputs=epoch_input, outputs=[])

            data_folder_input.upload(
                fn=handle_dataset_upload,
                inputs=data_folder_input,
                outputs=[start_btn, dataset_info_display, current_image_display, image_info_display]
            )

            data_folder_input.clear(
                fn=handle_dataset_clear,
                inputs=[],
                outputs=[start_btn, dataset_info_display, current_image_display, image_info_display]
            )

            file_output.clear(
                fn=handle_model_clear,
                outputs=[console, file_output]
            )

            start_btn.click(
                fn=start_training,
                outputs=[status_label, progress_label, console]
            )

            refresh_btn.click(
                fn=refresh_status,
                outputs=[status_label, progress_label, console, file_output]
            )

            # Navigation events
            first_btn.click(
                fn=lambda: navigate_image("first"),
                outputs=[current_image_display, image_info_display]
            )

            prev_btn.click(
                fn=lambda: navigate_image("prev"),
                outputs=[current_image_display, image_info_display]
            )

            next_btn.click(
                fn=lambda: navigate_image("next"),
                outputs=[current_image_display, image_info_display]
            )

            last_btn.click(
                fn=lambda: navigate_image("last"),
                outputs=[current_image_display, image_info_display]
            )

            split_selector.change(
                fn=switch_split,
                inputs=[split_selector],
                outputs=[current_image_display, image_info_display]
            )

            show_labels_btn.click(
                fn=toggle_labels,
                outputs=[current_image_display]
            )

            # Auto-refresh timer
            timer = gr.Timer(30, active=True)
            timer.tick(
                fn=refresh_status,
                outputs=[status_label, progress_label, console, file_output]
            )

            # Dataset selection dropdown
            dataset_selector.change(
                fn=on_select_dataset,
                inputs=[dataset_selector],
                outputs=[dataset_info_display, current_image_display, image_info_display]
            )

            # Delete button
            delete_btn.click(
                fn=on_delete_dataset,
                outputs=[dataset_selector, dataset_info_display, current_image_display, image_info_display]
            )

        return app

    def _get_dataset_info_html(self):
        """Get dataset information as HTML"""
        try:
            summary = self.training_manager.dataset_manager.get_dataset_summary()
            if not summary:
                return "<div class='dataset-info'>No dataset loaded</div>"

            html = f"""
            <div class='dataset-info'>
                <h4>Dataset: {summary['dataset_name']}</h4>
                <p><strong>Total Images:</strong> {summary['total_images']}</p>
                <p><strong>Training Images:</strong> {summary['train_images']}</p>
                <p><strong>Validation Images:</strong> {summary['val_images']}</p>
                <p><strong>Total Labels:</strong> {summary['total_labels']}</p>
                <p><strong>Classes:</strong> {summary['classes']}</p>
                <p><strong>Class Names:</strong> {summary['class_names'] if summary['class_names'] else 'Not specified'}</p>
            </div>
            """
            return html
        except Exception as e:
            return f"<div class='dataset-info'>Error loading dataset info: {str(e)}</div>"

    def _get_current_image(self):
        """Get current image for display"""
        try:
            result = self.training_manager.dataset_manager.get_current_image()
            if result:
                return result["image"]
            else:
                return None
        except Exception as e:
            print(f"Error getting current image: {e}")
            return None

    def _get_image_info_html(self):
        """Get current image information as HTML"""
        try:
            info = self.training_manager.dataset_manager.get_current_image_info()
            if not info:
                return "<div class='image-info'>No images loaded</div>"

            html = f"""
            <div class='image-info'>
                Image {info['current_index'] + 1} of {info['total_images']} ({info['current_split']} split)
                <br>
                <small>{info['current_filename']}</small>
            </div>
            """
            return html
        except Exception as e:
            return f"<div class='image-info'>Error loading image info: {str(e)}</div>"

    def _get_empty_upload_outputs(self):
        """Get empty outputs for upload handlers"""
        return (
            gr.update(interactive=False),
            self._get_empty_dataset_info(),
            None,
            self._get_empty_image_info()
        )

    def _get_empty_dataset_info(self):
        """Get empty dataset info HTML"""
        return "<div class='dataset-info'>No dataset loaded</div>"

    def _get_empty_image_info(self):
        """Get empty image info HTML"""
        return "<div class='image-info'>No images loaded</div>"