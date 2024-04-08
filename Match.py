# Import necessary libraries
from deepface import DeepFace  # Importing DeepFace for face recognition
from deepface.basemodels import VGGFace  # Importing VGGFace model for face recognition
import pandas as pd  # Pandas for data manipulation

# Define image paths
target_img_path = "target.jpg"
db_path = "C:\\workspace\\my_db"  # Note: Double backslashes are required to escape the backslash character

# Load the VGGFace model for face recognition
model = VGGFace.load_model()

# Find similar faces in a database using DeepFace
# `find` method searches for similar faces to the target image in the specified database
# Parameters:
#   - `img_path`: Path to the target image file.
#   - `db_path`: Path to the directory containing the database of images.
#   - `model_name`: Name of the face recognition model to be used (here, VGGFace).
#   - `model`: Pre-trained face recognition model (here, loaded VGGFace model).
#   - `distance_metric`: Distance metric used for comparing faces (e.g., 'cosine' for cosine similarity).
df = DeepFace.find(img_path=target_img_path, db_path=db_path, model_name=VGGFace, model=model, distance_metric='cosine')

# Display the first few rows of the DataFrame containing the results
df.head()

# Check if any matches were found
if df.shape[0] > 0:
    # Get the identity of the first matched face
    matched = df.iloc[0].identity
    # Print the identity of the matched face
    print(matched)
