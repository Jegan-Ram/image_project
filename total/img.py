import cv2

# Load the pre-trained pedestrian detection model
pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Load the pre-trained pothole detection model (custom-trained model or use a pre-trained model)
# For pothole detection, you may need a more advanced approach like deep learning or machine learning
# Here, I'll use a simple example to demonstrate the concept
pothole_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_bottomplate.xml')

# Open the video capture (you can replace 'your_video_file.mp4' with 0 for webcam)
cap = cv2.VideoCapture(r'C:\Users\Jegan Ramamurthy\Downloads\New folder\pedestrian\vid.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame is not read properly, break the loop
    if not ret:
        break

    # Convert the frame to grayscale for pedestrian detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect pedestrians in the frame
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Detect potholes in the frame
    potholes = pothole_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected potholes
    for (x, y, w, h) in potholes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Pedestrian and Pothole Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
