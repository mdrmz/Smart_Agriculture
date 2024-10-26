from ultralytics import YOLO
import cv2
import os
import time
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import sys

# Load the YOLO model
model_obj = YOLO('26102024Fistik.pt')

# Directory paths
image_directory = 'Data/'
output_directory = 'output/'
os.makedirs(output_directory, exist_ok=True)  # Create output directory if it doesn't exist

# Video settings
video_output_path = 'output/video_output.mp4'
fps = 1  # Frames per second

# Initialize video writer (we'll create it after we get the first image size)
video_writer = None


# Set up GUI
class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection GUI")

        # Layout and labels
        self.layout = QVBoxLayout()
        self.image_label = QLabel("Image will be displayed here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.object_count_label = QLabel("Detected objects will be displayed here")

        # Process images button
        self.process_button = QPushButton("Start Processing Images")
        self.process_button.clicked.connect(self.process_images)

        # Add widgets to layout
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.object_count_label)
        self.layout.addWidget(self.process_button)
        self.setLayout(self.layout)

    def process_images(self):
        global video_writer

        # Process each image
        image_paths = [os.path.join(image_directory, item) for item in os.listdir(image_directory)]

        for image_path in image_paths:
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image_colored = cv2.cvtColor(gray_image,
                                              cv2.COLOR_GRAY2BGR)  # Convert to 3 channels for YOLO compatibility

            # Run the grayscale image through the model
            model_result = model_obj(gray_image_colored, verbose=False)

            # Get the result image with boxes, masks, etc.
            result_image = model_result[0].plot()

            # Count the number of detected objects
            object_count = len(model_result[0].boxes)

            # Update GUI with image and object count
            self.update_gui(result_image, object_count)

            # Define frame size for the video writer if not already set
            if video_writer is None:
                frame_height, frame_width = result_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

            # Write the frame to the video
            video_writer.write(result_image)

            # Short delay to view each processed frame
            time.sleep(0.5)

            # Release video writer when done
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved at: {video_output_path}")

    def update_gui(self, image, object_count):
        # Convert OpenCV image (BGR) to Qt format (RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Update labels with pixmap and object count
        self.image_label.setPixmap(pixmap)
        self.object_count_label.setText(f"Detected Objects: {object_count}")
        QApplication.processEvents()  # Ensure the GUI updates immediately


# Main function to run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
