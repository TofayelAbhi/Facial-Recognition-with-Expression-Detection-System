import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
import time

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the face detector
detector = FaceDetector()
# Set parameters
offset = 20
imgSize = 224  # Set the desired output image size to 224x224
folder = r"C:\Users\USER\Downloads\Spring-2025\CSE445\Facial Expression\surprised"  # Adjust path as needed
counter = 0
save_images = False  # Flag to control saving of images

while True:
    success, img = cap.read()
    if not success:
        break  # Break the loop if the video frame is not successfully read

    # Create a copy of the image for detection drawing
    imgDisplay = img.copy()
    imgDisplay, bboxs = detector.findFaces(imgDisplay, draw=True)

    if bboxs:
        # Assuming bboxs[0] as the primary face
        x, y, w, h = bboxs[0]["bbox"]

        # Add offsets to the detected bounding box
        x1, y1 = x - offset, y - offset
        x2, y2 = x + w + offset, y + h + offset

        # Ensure bounding box is within image boundaries
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])

        # Crop the image from the original image
        imgCrop = img[y1:y2, x1:x2]
        # Resize the cropped image directly to 224x224, ignoring aspect ratio
        imgResize = cv2.resize(imgCrop, (imgSize, imgSize))

        # Show the cropped and resized image
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageResize", imgResize)  # Display the resized image

    # Show the original image with detected faces
    cv2.imshow("Image", imgDisplay)

    # Handle key presses
    key = cv2.waitKey(1)
    if key == ord('s'):
        save_images = True
        print("Saving images: ON")  # Debug print
    elif key == ord('f'):
        save_images = False
        print("Saving images: OFF")  # Debug print
    elif key == 27:  # ESC key
        break  # Exit the loop

    # Save images if flag is set
    if save_images and bboxs:
        img_path = f'{folder}/surprised_{time.time()}.jpg'  # Define the img_path variable
        cv2.imwrite(img_path, imgResize)  # Save the 224x224 image
        print(f"Image saved at: {img_path}")  # Print the path where the image is saved
        counter += 1

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
