# Handwritten Digit Recognition System

## Project Overview
This **deep learning project** focuses on classifying handwritten digits (0-9) using the MNIST dataset, a benchmark dataset in computer vision. The primary objective is to build and train a neural network to accurately predict the digit represented in a given grayscale image. The project involves preprocessing the data, designing and training a feedforward neural network, and visualizing results to demonstrate performance. This project serves as a showcase of deep learning expertise, including proper documentation, effective data visualization, and a clear workflow.


## Motivation
Handwritten digit recognition is a fundamental problem in computer vision and a gateway to understanding neural networks. This project aims to:

- Apply **Deep learning concepts** to a real-world dataset.
- Demonstrate skills in building, training, and evaluating **Neural networks**.


## Technologies Used
- **Programming Language**: Python
- **Platform**: Jupyter Notebook
- **Libraries**:
  - Pandas, NumPy: Data manipulation and analysis
  - Seaborn, Matplotlib: Data visualization
  - Scikit-learn: Metrics for evaluation
  - TensorFlow/Keras: Neural network modeling and training
    

## Dataset Details
- **Source**: [MNIST Handwritten Digit Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- **Size**: 60,000 training samples, 10,000 test samples (28x28 grayscale images)
- **Classes**: 10 digits (0-9)


## Project Workflow

- **Data Preprocessing**:
   - Loaded the MNIST dataset using Keras utilities.
   - Normalized pixel values to the range [0, 1] to optimize model training.
   - One-hot encoded labels to ensure compatibility with the categorical cross-entropy loss function.
     
- **Data Visualization**:
  - Visualized sample images from the dataset to confirm preprocessing steps.
  - Used histograms and heatmaps to analyze class distributions and model performance.
 
- **Neural Network Model**:
  - **Architecture**:
     - **Input layer**: Flatten (28x28 pixels → 784 features)
     - **Hidden layers**: Fully connected layers with ReLU activation (128 and 64 units) and Dropout regularization.
     - **Output layer**: Fully connected layer with 10 units and softmax activation.
  - **Optimization**: Adam optimizer with categorical cross-entropy loss.
  - **Training**:
     - Trained the model over 20 epochs using a batch size of 128.
     - Evaluated performance on validation data after each epoch.
    
- **Evaluation**:
  - Visualized training and validation accuracy/loss trends.
  - Computed metrics like accuracy, precision, recall, and F1-score.
  - Generated a confusion matrix for detailed insights into model predictions.

- **Results**:
   The final model achieved the following metrics on the test dataset:

   - Accuracy: `0.97`
   - Precision: `0.96`
   - Recall: `0.95`
   - F1-Score:`0.95`
  

## Future Work
- Experiment with Convolutional Neural Networks (CNNs) to improve accuracy further.
- Use data augmentation techniques to enhance generalization.
- Explore hyperparameter tuning and other optimizers for better performance.
- Integrate the model into an application for real-time handwritten digit recognition.
  
## License
This project is open-source and available under the **MIT** License.

## Authors

- [M Tareq Rahman](https://github.com/Tareq553)

