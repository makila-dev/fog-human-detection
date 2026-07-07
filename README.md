# AI-Based Human Detection in Foggy and Hazy Conditions Using YOLOv8

## Overview

The **AI-Based Human Detection in Foggy and Hazy Conditions Using YOLOv8** is a deep learning-based web application developed using **Python**, **Flask**, **YOLOv8**, and **OpenCV**. The system is designed to accurately detect human beings in images and live webcam streams under low-visibility conditions such as fog and haze.

To improve detection performance, the application first enhances the image using OpenCV's image enhancement techniques before performing object detection with a custom-trained YOLOv8 model. The detected persons are highlighted with bounding boxes, confidence scores, and the total number of detected persons is displayed.

---

## Features

- Human detection in foggy and hazy environments
- Image enhancement using OpenCV
- Custom-trained YOLOv8 model
- Upload image for detection
- Live webcam human detection
- Automatic person counting
- Bounding boxes with confidence scores
- Responsive Flask web interface
- Fast real-time detection
- Supports JPG, JPEG, and PNG images

---

## Technologies Used

- Python 3.x
- Flask
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- NumPy
- HTML5
- CSS3
- 

---

## Project Structure

```
fog/
│
├── app.py
├── requirements.txt
├── README.md
│
├── static/
│   ├── uploads/
│   ├── results/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── home.html
│   ├── detect.html
│   ├── live.html
│   ├── about.html
│   └── contact.html
│
└── runs/
    └── detect/
        └── train/
            └── weights/
                └── best.pt
```

---

## Workflow

1. User opens the Flask web application.
2. Upload an image or open the live camera page.
3. The uploaded image is enhanced using OpenCV.
4. The enhanced image is passed to the custom YOLOv8 model.
5. The model detects all humans in the scene.
6. Bounding boxes and confidence scores are displayed.
7. Total detected persons are counted.
8. The processed image is saved in the results folder.

---

## Image Detection

The application allows users to upload an image containing people in foggy or hazy conditions.

The system performs:

- Image upload
- Image enhancement
- Human detection
- Person counting
- Result visualization

Output includes:

- Uploaded Image
- Detected Image
- Total Person Count

---

## Live Camera Detection

The live detection module uses the webcam to detect humans in real time.

Features include:

- Live video streaming
- Continuous frame processing
- Image enhancement
- Real-time detection
- Automatic person counting
- Bounding boxes with confidence scores

---

## Image Enhancement

Before detection, each frame is enhanced using OpenCV.

Enhancement includes:

- Detail Enhancement
- Contrast Improvement
- Noise Reduction
- Better visibility in foggy conditions

---

## YOLOv8 Detection

The project uses a custom-trained YOLOv8 model stored at:

```
runs/detect/train/weights/best.pt
```

Detection Parameters:

- Confidence Threshold: 0.4
- Target Class: Person (Class ID = 0)

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YourUsername/AI-Based-Human-Detection.git
```

### Move into Project

```bash
cd AI-Based-Human-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

---

## Application URLs

Home Page

```
http://127.0.0.1:5000/
```

Image Detection

```
http://127.0.0.1:5000/detect
```

Live Detection

```
http://127.0.0.1:5000/live
```

About

```
http://127.0.0.1:5000/about
```

Contact

```
http://127.0.0.1:5000/contact
```

---

## Requirements

```
Flask
ultralytics
opencv-python
numpy
Pillow
```

---

## Applications

- Smart Surveillance
- Border Security
- Traffic Monitoring
- Railway Platforms
- Airports
- Industrial Safety
- Military Surveillance
- Autonomous Vehicles
- Search and Rescue Operations
- Smart Cities

---

## Future Enhancements

- Video file detection
- Email alert system
- Intrusion detection
- Cloud deployment
- Mobile application
- Multi-person tracking
- Face recognition integration
- Performance analytics dashboard

---

## Output

The application displays:

- Enhanced Image
- Human Detection
- Bounding Boxes
- Confidence Score
- Total Person Count
- Live Webcam Detection

---

## Author

**M. Akila**

**M.Sc. Computer Science**

Python | Flask | YOLOv8 | OpenCV | Deep Learning | Computer Vision

---

## License

This project is developed for educational, academic, and research purposes.