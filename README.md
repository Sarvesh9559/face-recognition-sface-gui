🧠 Face Recognition GUI (SFace + YuNet)

A powerful desktop application for comparing faces & facial regions using OpenCV SFace (ArcFace-based embeddings) and YuNet Face Detector.
This tool allows you to:

✔ Compare full faces automatically
✔ Select any region manually (eyes, nose, lips, forehead, chin, etc.)
✔ Get accurate similarity percentage
✔ Supports drag-to-select, high-accuracy detection, and robust embeddings
✔ Fully offline (no cloud required)
✔ Works on Windows with Python 3.10

🚀 Demo (Screenshots)
Full GUI

(Insert screenshot)

Manual Region Selection

(Insert screenshot)

✨ Features
🔍 Face Detection — YuNet

Ultra-fast, accurate face detector

Detects bounding box + 5 landmarks

Auto-selects the best face (highest score)

🧠 Face Recognition — SFace

ArcFace-based embeddings

High-accuracy comparison

Cosine-similarity scoring

Configurable threshold (default: 50%)

🖼 Manual Region Matching

Draw rectangles on both images

Compare only selected facial parts

Useful for partial-face comparisons

🎨 GUI Features

Built using Tkinter

Loads images with Unicode-safe loader

Zoom-scaled display

Easy drag & draw boxes

Color-coded results (green/red)

📂 Repository Structure
face-recognition-sface-gui/
│
├── face_match_gui.py              # Main GUI Application
├── sface.py                        # SFace model wrapper (OpenCV Zoo)
├── sface_2021dec.onnx             # SFace model (36MB)
├── face_detection_yunet_2023mar.onnx  # YuNet model
└── README.md                      # Documentation

🛠 Installation
1️⃣ Install Python (Recommended: 3.10)

Download Python 3.10 from:
https://www.python.org/downloads/release/python-3100/

Ensure “Add to PATH” is checked.

2️⃣ Install Required Packages

Run this in PowerShell:

pip install numpy==1.26.4
pip install opencv-contrib-python==4.8.1.78
pip install pillow

3️⃣ Download SFace + YuNet Models

Already included in repo.
But if you want to re-download:

SFace (36 MB)
curl.exe -L "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx" -o sface_2021dec.onnx

YuNet (5 MB)
curl.exe -L "https://raw.githubusercontent.com/opencv/opencv_zoo/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -o face_detection_yunet_2023mar.onnx

▶️ Running the Application
python face_match_gui.py

🎯 Usage
✔ Auto Full-Face Match

Load two images

Click Auto Detect & Compare FULL Faces

YuNet detects faces

SFace computes embeddings

Similarity shown as percentage

✔ Manual Region Comparison

Load Image 1

Draw rectangle on area (drag the mouse)

Repeat for Image 2

Click Compare MANUAL Regions

Useful for checking:

Eyes

Nose

Lips

Forehead

Specific marks or regions

📊 Similarity Rules

Output is from 0% to 100%

Default threshold: 50%

50% = Match

<50% = Not Match

You can adjust this easily in code:

THRESHOLD = 50.0

🔮 Future Improvements

Add zoom-in/out feature

Export results to PDF

Add 3-image comparison mode

Add batch face comparison

Mobile app (Android/Flutter)

📝 License

MIT License — free for personal and commercial use.

🤝 Contributing

Pull requests are welcome!
If you want help writing CONTRIBUTING.md, just say “write contributing doc”.

⭐ Support

If you like the project, please give it a star ⭐ on GitHub:
👉 https://github.com/Sarvesh9559/face-recognition-sface-gui
