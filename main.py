#!/usr/bin/env python3
"""Main application entry point"""
from frontend.ui import YOLOTrainingUI

def main():
    """Main application function"""
    ui = YOLOTrainingUI()
    app = ui.create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()