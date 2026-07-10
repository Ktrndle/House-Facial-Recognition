Smart Home — Object Detection & Voice Control 🏠

A home automation system that combines real-time object detection with voice-command control, integrating computer vision and embedded hardware for hands-free home management.

Overview

This project uses YOLOv8 for real-time object detection to identify people, objects, or activity within the home environment, paired with voice-command recognition to control connected devices. Arduino handles the hardware-side actions (e.g. lights, appliances, locks) based on commands and detection events.

Features


🎯 Real-time object detection using YOLOv8
🎙️ Voice-command control for home devices
🔌 Arduino-based hardware control (lights, appliances, etc.)
🤖 Combines computer vision with embedded systems for automation


Built With


Python — YOLOv8 object detection and voice command processing
YOLOv8 — real-time object detection model
Arduino (C/C++) — hardware control and device automation


How It Works


A camera feed is processed in real time using YOLOv8 to detect objects/people within the home
Voice commands are captured and processed to trigger specific actions
Detected events and recognized commands are sent to Arduino via [serial/Wi-Fi/Bluetooth — specify]
Arduino executes the corresponding hardware action (e.g. turning on lights, unlocking a door)


Setup / Installation

1. Clone this repository
2. Install Python dependencies: pip install -r requirements.txt
3. Connect Arduino via [USB/serial port]
4. Upload the Arduino sketch (smart_home.ino) to your board
5. Run the main Python script: python main.py
6. Speak a command or allow the camera to detect objects to trigger automation

Author

Kathryn Deille C. Abasolo
LinkedIn

License

This project was developed as a personal/academic project.
