# Developing-a-Chessboard-Diagram-classifier
It will take a printed chessboard diagram and report which piece is on each square.

# Overview

This Python project is a chess board classifier that takes images of chess boards as input and returns a representation of the board. The classifier is developed using only the used NumPy and scipy.stats libraries, and the data preprocessing steps and classification are implemented from scratch.

# Features

- **Input:** The classifier takes images of chessboards as input.
- **Output:** It returns a representation of the chessboard.

# Dependencies

The project relies on the following libraries:

- NumPy
- scipy

# Training

The training process involves implementing data preprocessing steps and the classification algorithm from scratch using NumPy and scipy.stats. The classifier was trained on 6400 chess board images.

# Testing

the testing process was staged using 1600 chess board images which the model has never encountered, during testing, it achieved an accuracy of 95.1%.

# How to Use

 1. Clone the repository to your local machine:

    ```bash
    git clone git@github.com:youngaryan/Developing-a-Chessboard-Diagram-classifier.git
    ```

2. Install the required dependencies:

    ```bash
    pip install numpy
    pip install scipy
    ```

3. Train the model:
The training process involves loading board metadata, reading corresponding image data, and processing them to generate feature vectors. The trained model's parameters are then stored in a dictionary and saved to a JSON file.

    ```bash
    python train.py
    ```

5. Test the model:
The evaluation process involves loading the model data and test metadata, selecting a subsets of the availble fetures (choose 10 features out of 2500 features) classifying (KNN) the test data using the trained model, and comparing the predicted labels with the true labels. 
    ```bash
    python evaluate.py
    ```

## Contributors

- Aryan Golbaghi
