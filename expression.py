import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print(cv2.__version__)

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 emotions

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
device = torch.device('cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('fer2013_model.pth', map_location=device))
model.eval()

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 48x48
    gray = cv2.resize(gray, (48, 48))
    # Normalize
    gray = gray / 255.0
    # Add channel and batch dimensions
    gray = np.expand_dims(gray, axis=(0, 1))  # (1, 1, 48, 48)
    return torch.tensor(gray, dtype=torch.float32).to(device)

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera, or replace with a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    image = preprocess_image(frame)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)  # No need for unsqueeze(0) as preprocess_image adds batch dimension
        _, predicted = torch.max(outputs, 1)
        emotion = predicted.item()

    # Display the resulting frame
    emotion_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    cv2.putText(frame, f'Emotion: {emotion_map[emotion]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Recognition', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
