Real-time Emotion Detection System:

This project implements a real-time emotion detection system using a Convolutional Neural Network (CNN) built with PyTorch. The model is trained on the FER2013 dataset for emotion classification, and features the following:

Data Preprocessing: 

Used Pandas and NumPy for data manipulation, and TorchVision for data augmentation.
Model Training:

Built a CNN architecture with convolutional layers, ReLU activations, max pooling, and fully connected layers. Trained with cross-entropy loss and the Adam optimizer.
Regularization: 

Integrated dropout, L2 regularization (weight decay), and early stopping to prevent overfitting.
Performance Evaluation: 

Evaluated the model using Scikit-learn metrics.
Live Classification: 

Used OpenCV for real-time emotion classification via webcam.
