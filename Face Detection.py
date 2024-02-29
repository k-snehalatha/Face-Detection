from google.colab import drive
drive.mount('/content/drive')
import cv2
from google.colab.patches import cv2_imshow # Special function to display images in Colab
# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
from google.colab import files
uploaded = files.upload()

# Assuming you upload only one image, get the filename
filename = next(iter(uploaded))

# Read the image
image = cv2.imread(filename)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

# Draw rectangles around each face
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the image with detected faces
cv2_imshow(image)