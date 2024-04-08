# Import necessary libraries
from retinaface import RetinaFace  # Importing RetinaFace for face detection
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt  # Matplotlib for plotting images



# Path to the image file
img_path = "target.jpg"
# Load the image using OpenCV
img = cv2.imread(img_path)

# Detect faces in the image using RetinaFace
# `detect_faces` returns a dictionary where each key represents a detected face
resp = RetinaFace.detect_faces(img_path)
# Count the number of detected faces by counting the keys in the response dictionary
num_faces = len(resp.keys())


# Iterate over each detected face
for key in resp.keys():
    # Get the information about the detected face using its key
    identity = resp[key]
    # Extract the coordinates of the bounding box surrounding the face
    facial_area = identity['facial_area']
    # Draw a rectangle around the detected face on the image
    cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)


# Plot the image with detected faces using Matplotlib
plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.show()


# Parameters Explanation:
# - `img_path`: Path to the input image file.
# - `img`: The loaded image using OpenCV.
# - `resp`: The response from the face detection model. 
#           It's a dictionary where each key corresponds to a detected face. 
#           Each value contains information about the detected face, such as its bounding box coordinates.
# - `num_faces`: Number of detected faces in the image.
# - `identity`: Information about each detected face, extracted from the response dictionary.
# - `facial_area`: Coordinates of the bounding box around each detected face.
# - `plt.figure(figsize=(20, 20))`: Configuring the size of the plot to display the image with detected faces.
# - `plt.imshow(img)`: Displaying the image with detected faces.
# - `plt.show()`: Showing the plot with the image.
