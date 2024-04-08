# Import necessary libraries
from deepface import DeepFace  # Importing DeepFace for face verification

# Define image paths
img1_path = "target.jpg"
img2_path = "target.jpg"

# Perform face verification using DeepFace
# `verify` method verifies whether two faces belong to the same person or not
# Parameters:
#   - `img1_path`: Path to the first image file.
#   - `img2_path`: Path to the second image file.
#   - `model_name`: Name of the face recognition model to be used (here, ArcFace).
#   - `detector_backend`: Backend detector for detecting faces (here, RetinaFace).
obj = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name='ArcFace', detector_backend='retinaface')

# Print the verification result
print(obj)
