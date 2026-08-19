Voedger — Autonomous Crack Detection System

An AI-powered crack detection system designed to identify structural cracks from images using deep learning and computer vision.

Overview

Voedger is a prototype for automated structural crack inspection. The system uses trained machine-learning models to analyze images and determine whether visible cracks are present.

The project is intended as a foundation for an autonomous crack inspection robot that can eventually be deployed for inspecting:

Bridges
Flyovers
Dams
Tunnels
Other concrete infrastructure

The goal is to reduce dependence on manual inspection while making structural monitoring faster, safer, and more scalable.

Features
AI-based crack detection from images
Positive/Negative crack classification
Trained deep-learning models
Test dataset containing cracked and non-cracked images
Image-based prediction scripts
Separate training and testing implementations
Sample images for testing the model
Machine Learning

The repository contains trained models and experimentation code under algo2/.

Dataset Structure
test_data/
├── Negative/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
│
└── Positive/
    ├── 00001.jpg
    ├── 00002.jpg
    └── ...
Positive — images containing structural cracks
Negative — images without detectable cracks
Project Structure
Voedger/
│
├── algo2/
│   ├── Sample.jpg
│   ├── crack_model.keras
│   ├── test.py
│   ├── train.py
│   └── test_data/
│       ├── Negative/
│       └── Positive/
│
├── Onlinetest/
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpeg
│   └── img4.jpg
│
├── Predict
├── Predict.zip
├── crack_model.pt
├── predict.py
├── train.py
└── .gitignore
System Workflow
Input Image
     ↓
Image Preprocessing
     ↓
Trained ML Model
     ↓
Crack Detection
     ↓
Positive / Negative Result

The eventual robotic system can extend this pipeline by combining the detection model with cameras, sensors, autonomous navigation, and structural mapping.

Installation

Clone the repository:

git clone https://github.com/KiRiT095/Voedger.git
cd Voedger

Install the required Python dependencies:

pip install -r requirements.txt

A requirements.txt file should be added once the project's final dependencies are confirmed.

Usage
Run Prediction
python predict.py
Train the Model
python train.py

The exact command-line arguments may vary depending on the current implementation of the scripts.

Current Prototype

The current repository represents the AI crack-detection component of the larger autonomous inspection concept.

Future versions can integrate:

High-resolution cameras
Autonomous robotic movement
Navigation and localization
Crack location mapping
Crack severity estimation
Structural condition reports
Remote monitoring dashboards
Automated alerts
Improved crack segmentation and classification
Future Vision

The long-term objective is to develop an autonomous inspection robot capable of navigating infrastructure, capturing surface images, detecting structural defects, and generating a digital inspection report.

Robot Navigation
       ↓
Surface Scanning
       ↓
Image Capture
       ↓
AI Crack Detection
       ↓
Crack Localization
       ↓
Severity Analysis
       ↓
Inspection Report
Technologies

The current project primarily uses:

Python
Machine Learning
Deep Learning
Computer Vision
Keras
PyTorch
Image Classification

Additional robotics and navigation technologies will be integrated as the project develops.

Project

Voedger
Autonomous Crack Inspection & Structural Monitoring

Developed as a prototype for an autonomous infrastructure inspection system.

License

This project currently does not specify a license.
