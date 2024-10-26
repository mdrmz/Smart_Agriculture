
from ultralytics import YOLO
import cv2
import os
import time
import threading
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QFrame, QMenuBar, QFileDialog
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon
from PySide6.QtCore import Qt, QTimer, QTime, QPropertyAnimation
from PySide6.QtGui import QAction
import sys

# Load the YOLO model
model_obj = YOLO('26102024Fisitk2.pt')

# Directory paths
output_directory = 'output/'
os.makedirs(output_directory, exist_ok=True)  # Create output directory if it doesn't exist

# Video output settings
video_output_path = 'output/video_output.mp4'
fps = 60  # Frames per second

# Initialize video writer (we'll create it after we get the frame size)
video_writer = None
object_counter = 0  # Tracks total detected objects

# Set up GUI
class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Object Detection")
        self.setGeometry(100, 100, 900, 650)
        self.setWindowIcon(QIcon('logo.png'))  # Logo added

        # Styling
        self.setStyleSheet("""
            background-color: #2C2C2C; 
            color: #FFFFFF; 
            font-family: 'Verdana';
            border-radius: 10px;
        """)
        self.init_ui()

        # Timer and Video Control Flags
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.is_paused = False
        self.elapsed_time = 0

        # Initialize threading and multithreading flags
        self.processing_thread = None
        self.capture_flag = False
        self.video_input_path = None  # To hold the selected video path

    def init_ui(self):
        # Menu bar
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu("File")

        # Menu actions
        open_video_action = QAction("Open Video", self)
        open_video_action.triggered.connect(self.open_video)
        file_menu.addAction(open_video_action)

        # Layout
        layout = QVBoxLayout()

        # Logo Display
        logo_label = QLabel()
        logo_pixmap = QPixmap("logo.png").scaled(100, 100, Qt.KeepAspectRatio)
        logo_label.setPixmap(logo_pixmap)
        layout.addWidget(logo_label, alignment=Qt.AlignCenter)

        # Title
        title = QLabel("Object Detection from Video")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Verdana", 18, QFont.Bold))
        layout.addWidget(menubar)
        layout.addWidget(title)

        # Timer display
        self.time_display = QLabel()
        self.update_clock()
        self.time_display.setAlignment(Qt.AlignCenter)
        self.time_display.setFont(QFont("Verdana", 14, QFont.Bold))
        layout.addWidget(self.time_display)

        # Timer Label
        self.timer_label = QLabel("Elapsed Time: 0 sec")
        self.timer_label.setFont(QFont("Verdana", 12))
        layout.addWidget(self.timer_label)

        # Image display frame
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.Box)
        image_frame.setStyleSheet("border: 2px solid #6D6D6D; border-radius: 10px;")
        image_layout = QVBoxLayout(image_frame)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label)
        layout.addWidget(image_frame)

        # Object count label
        self.object_count_label = QLabel("Detected objects will be displayed here")
        self.object_count_label.setFont(QFont("Verdana", 12))
        layout.addWidget(self.object_count_label)

        # Control buttons
        button_layout = QHBoxLayout()

        # Start button with animation
        self.process_button = self.create_button("Start Processing Video", "#5B8CC2", self.start_processing)
        button_layout.addWidget(self.process_button)

        # Pause button with animation
        self.pause_button = self.create_button("Pause", "#FFD700", self.pause_processing)
        button_layout.addWidget(self.pause_button)

        # Save button
        self.save_button = self.create_button("Save Video", "#228B22", self.save_video)
        button_layout.addWidget(self.save_button)

        # Capture snapshot button
        self.snapshot_button = self.create_button("Capture Snapshot", "#C71585", self.capture_snapshot)
        button_layout.addWidget(self.snapshot_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Real-time clock display
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

    def create_button(self, text, color, on_click):
        button = QPushButton(text)
        button.setFont(QFont("Verdana", 14))
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                padding: 10px 20px;
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background-color: #777777;
            }}
        """)
        button.clicked.connect(on_click)

        # Animation for click effect
        animation = QPropertyAnimation(button, b"geometry")
        animation.setDuration(100)
        button.pressed.connect(lambda: animation.start())
        return button

    def open_video(self):
        # Opens a dialog to select a video file
        self.video_input_path, _ = QFileDialog.getOpenFileName(self, "Select a Video File", "", "Video Files (*.mp4 *.avi)")
        if self.video_input_path:
            print(f"Selected video: {self.video_input_path}")

    def start_processing(self):
        if self.video_input_path and (not self.processing_thread or not self.processing_thread.is_alive()):
            self.is_paused = False
            self.elapsed_time = 0
            self.timer.start(1000)  # Start timer to update every second
            self.processing_thread = threading.Thread(target=self.process_video)
            self.processing_thread.start()

    def pause_processing(self):
        self.is_paused = not self.is_paused
        self.pause_button.setText("Resume" if self.is_paused else "Pause")

    def save_video(self):
        print(f"Video saved at: {video_output_path}")

    def capture_snapshot(self):
        self.capture_flag = True

    def update_time(self):
        self.elapsed_time += 1
        self.timer_label.setText(f"Elapsed Time: {self.elapsed_time} sec")

    def update_clock(self):
        current_time = QTime.currentTime().toString("hh:mm:ss")
        self.time_display.setText(f"Current Time: {current_time}")

    def process_video(self):
        global video_writer, object_counter
        cap = cv2.VideoCapture(self.video_input_path)

        while cap.isOpened():
            if self.is_paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # Keep the frame in RGB format and pass it to the model
            model_result = model_obj(frame, verbose=False)
            result_frame = model_result[0].plot()
            object_count = len(model_result[0].boxes)
            object_counter += object_count

            # Initialize video writer with frame size if it's not set
            if video_writer is None:
                frame_height, frame_width = result_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

            # Save snapshot if requested
            if self.capture_flag:
                snapshot_path = os.path.join(output_directory, f'snapshot_{int(time.time())}.png')
                cv2.imwrite(snapshot_path, result_frame)
                print(f"Snapshot saved at: {snapshot_path}")
                self.capture_flag = False

            # Write processed frame to video
            video_writer.write(result_frame)

            # Update GUI with current frame
            self.update_gui(result_frame, object_count)
            time.sleep(1 / fps)

        cap.release()
        if video_writer:
            video_writer.release()
        self.timer.stop()

    def update_gui(self, frame, object_count):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap)
        self.object_count_label.setText(f"Total Detected Objects: {object_counter} | Current Frame: {object_count}")
        QApplication.processEvents()

# Main function to run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec())
