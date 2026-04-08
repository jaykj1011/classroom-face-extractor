# Classroom Face Extractor

A simple Python project to detect and extract faces from a classroom image using MTCNN and OpenCV.

## Features

* Detects multiple faces in an image
* Draws bounding boxes around detected faces
* Saves each face as a separate image file

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/classroom-face-extractor.git
cd classroom-face-extractor
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Update the image path in `main.py` before running.

## Output

Extracted face images are saved in:

```
extracted_faces/
```

## Technologies Used

* MTCNN (face detection)
* OpenCV (image processing)
* Matplotlib (visualization)
