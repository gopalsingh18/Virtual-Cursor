# Virtual Cursor using Hand and Eye Detection

**Aim:**  
Develop a computer vision-based gesture recognition system that enables users to control a virtual cursor using hand and eye movements.

**Summary:**
This project lets users control the mouse, media playback, and volume using hand gestures and face tracking. It leverages OpenCV, MediaPipe, and pynput to detect hand movements and facial landmarks.

**Features:**
- Move the mouse using the index finger.
- Left-click when index & middle fingers are close together.
- Double-click on repeated pinch.
- Right-click when only the thumb is up.
- Scroll up/down using pinky or ring fingers.
- Adjust volume based on head movement (nose Y-coordinate).
- Pause media if eyes are not detected.
- Lock screen if eyes are not detected for too long.

**Requirements:**
- Python 3.10 recommended
- Webcam

**Run the main script:**
python virtualmouse.py 

**To stop the program:** 
press `Ctrl+C` in the terminal.

**Applications:**
- Gaming and Virtual Reality (VR)
- Augmented Reality and Smart Displays
- Assistive Technology
- Security and Authentication




