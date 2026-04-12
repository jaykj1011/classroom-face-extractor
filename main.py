import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import os

def extract_classroom_faces(image_path, output_dir="extracted_faces"):
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Load the image using OpenCV
    # OpenCV loads images in BGR format, so we convert it to RGB for MTCNN and Matplotlib
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return
        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 3. Initialize the MTCNN detector
    detector = MTCNN()

    # 4. Detect faces
    # This returns a list of dictionaries, each containing the bounding box and confidence score
    faces = detector.detect_faces(image_rgb)
    print(f"Detected {len(faces)} faces in the classroom.")

    # Copy the image to draw bounding boxes on
    image_with_boxes = image_rgb.copy()

    # 5. Loop through all detected faces, crop, and save them
    for i, face in enumerate(faces):
        # Extract the bounding box coordinates
        x, y, width, height = face['box']
        
        # MTCNN can sometimes return negative coordinates if a face is cut off at the edge
        x, y = max(0, x), max(0, y) 

        # Draw a rectangle on our visualization image
        cv2.rectangle(image_with_boxes, (x, y), (x + width, y + height), (0, 255, 0), 3)

        # Crop the face from the ORIGINAL RGB image
        cropped_face = image_rgb[y:y+height, x:x+width]

        # Convert back to BGR to save properly with OpenCV
        cropped_face_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

        # Save the cropped face
        face_filename = os.path.join(output_dir, f"student_face_{i+1}.jpg")
        cv2.imwrite(face_filename, cropped_face_bgr)

    # 6. Display the overall result
    plt.figure(figsize=(12, 8))
    plt.imshow(image_with_boxes)
    plt.title(f"Found {len(faces)} Students")
    plt.axis('off')
    plt.show()

    print(f"All cropped faces have been saved to the '{output_dir}' folder.")

extract_classroom_faces('IMG_5852.jpeg', output_dir='dataset/unknown_crops1')
