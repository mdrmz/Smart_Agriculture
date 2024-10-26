# AI-Based Crop Yield Estimation and Prediction 🌾

This project, developed by **Mehmet Durmaz** and **Ber Egeli**, focuses on enhancing agricultural productivity through **AI-driven crop yield estimation and prediction**. Using **YOLOv11** for real-time object detection, we built a model that analyzes crop density, providing valuable insights into crop health and yield potential.

## 📹 Demo Video
Check out the project demo on YouTube [here](https://youtu.be/pKKtFfQxmUI?si=BsYRspBAgtIpVegn)!

## 🔍 Project Overview
The goal of this project is to assist farmers and agricultural stakeholders in optimizing crop yield using advanced object detection and machine learning techniques. The system not only detects crops in real-time but also provides analysis based on crop density, enabling better decision-making for sustainable farming.

## 🚀 Key Features
- **Real-time Crop Detection**: Detects and tracks crops in real-time using YOLOv11.
- **Crop Density Analysis**: Estimates crop health and density, aiding yield prediction.
- **User Interface**: Built with **PySide6**, providing a user-friendly and interactive experience.
- **Real-Time Object Counter**: Displays live object count on the GUI, enhancing real-time monitoring capabilities.

## 🔧 Technologies Used
- **YOLOv11**: For efficient and accurate real-time object detection.
- **PySide6**: To create a responsive and intuitive graphical user interface.
- **Python & OpenCV**: For image processing and video handling.

## 💻 Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/mdrmz/Smart_Agriculture.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Smart_Agriculture
    ```
3. Install the required dependencies:
    ```bash
    !pip install opencv-python
    !pip install ultralytics
    !pip install PySide6
    ```

## 📁 Directory Structure
```plaintext
├── Video_Input.py              # Main script to run the application
├── README.md                   # Project documentation
├── output/                     # Folder to save video output and snapshots
├── Data/                       # Contains images, icons, and other resources


## Requirements
## Gereksinimler
- Python 3.8+
- PySide6
- OpenCV
- YOLOv11 model weights

To install the dependencies, run:
Gereksinimleri yüklemek için:

```bash
gh repo clone mdrmz/Smart_Agriculture
cd Smart_Agriculture
```

```bash
python Video_Input.py
```

```bash
python Fistik_Object.py

```












