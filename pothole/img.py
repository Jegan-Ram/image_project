import cv2
import numpy as np
import os

def detect_potholes(image_path):
   
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    # Read the image
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    # Convert the image to grayscale
    image = cv2.resize(image, (400, 300))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and help edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through the contours
    for contour in contours:
        # Ignore small contours (adjust the threshold as needed)
        if cv2.contourArea(contour) > 300:
            # Draw a rectangle around the detected pothole
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Pothole Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the path to the image you want to analyze
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_directory, r"C:\Users\Jegan Ramamurthy\Downloads\New folder\pothole\img1.png")
    
    # Call the function to detect potholes
    detect_potholes(image_path)
