import cv2
import numpy as np
import imutils
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
    image = cv2.resize(image, (800, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and help edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 50, 100)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through the contours
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detecting all the regions in the 
    # Image that has a pedestrians inside it
    (regions, _) = hog.detectMultiScale(image, 
                                        winStride=(4, 4),
                                        padding=(4, 4),
                                        scale=1.05)

    # Drawing the regions in the Image
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), 
                    (x + w, y + h), 
                    (0, 0, 255), 1)
        
    for contour in contours:
        if cv2.contourArea(contour) > 200:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Pothole Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the path to the image you want to analyze
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_directory, r"C:\Users\Jegan Ramamurthy\Downloads\New folder\total\img1.png")
    
    # Call the function to detect potholes
    detect_potholes(image_path)
