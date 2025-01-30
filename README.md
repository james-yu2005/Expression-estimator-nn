## Real-time Emotion Detection System

This project implements a **real-time emotion detection system** using a **Convolutional Neural Network (CNN)** built with **PyTorch**. The model classifies emotions from images in real-time, trained on the **FER2013 dataset**.

### Key Features:
- **Data Preprocessing**: 
  - Used **Pandas** and **NumPy** for data manipulation.
  - Applied **TorchVision** for data augmentation to improve model robustness.
  
- **Model Architecture**:
  - Built a **CNN** with convolutional layers, **ReLU** activation, **max pooling**, and **fully connected layers** for classification.
  - Trained the model with **cross-entropy loss** and **Adam optimizer**.

- **Regularization**: 
  - Integrated **dropout**, **L2 regularization (weight decay)**, and **early stopping** to prevent overfitting during training.

- **Performance Evaluation**: 
  - Evaluated model performance using **Scikit-learn** metrics such as accuracy and confusion matrix.

- **Live Emotion Classification**: 
  - Used **OpenCV** to implement real-time emotion detection through a webcam, providing live feedback on detected emotions.

### How It Works:
1. The model is trained on the **FER2013 dataset**, which contains facial expressions categorized into different emotions.
2. The real-time classification process uses **OpenCV** to capture live video feed and predict emotions based on facial expressions detected in each frame.
3. The CNN is optimized using **Adam optimizer** and regularized with techniques like **dropout** and **early stopping** for better generalization.

### Tools & Technologies:
- **PyTorch**: For building and training the CNN model.
- **Pandas** & **NumPy**: For data manipulation and preprocessing.
- **TorchVision**: For image augmentations.
- **OpenCV**: For real-time webcam capture and emotion detection.
- **Scikit-learn**: For evaluating model performance using various metrics.
