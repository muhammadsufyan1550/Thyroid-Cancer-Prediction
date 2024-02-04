# Thyroid Cancer Recurrence Prediction

This repository contains code for a machine learning project that predicts thyroid cancer recurrence using two different models: Support Vector Machine (SVM) and Artificial Neural Network (ANN). The project aims to classify patients into two classes: those with recurrent thyroid cancer (label: 'YES') and those without recurrence (label: 'NO').

## Dataset
The dataset used for this project contains information about various features related to thyroid cancer patients, with the 'Recurred' column indicating whether the cancer recurred ('YES') or not ('NO'). The dataset consists of 383 rows and 17 columns, where 16 columns are features and 1 column is the target label.

## Code Structure
- `thyroid_cancer_prediction.ipynb`: Jupyter Notebook containing the code for data loading, preprocessing, model training, evaluation, and visualization.
- `thyroid.csv`: CSV file containing the dataset used in the project.

## Models
1. **Support Vector Machine (SVM)**:
   - Achieved an accuracy of 98% on the test set.
   - Utilized the `SVC` class from scikit-learn for model training and prediction.
   - Applied feature scaling using `StandardScaler`.
   - Evaluated model performance using accuracy, confusion matrix, and F1 score.

2. **Artificial Neural Network (ANN)**:
   - Achieved an accuracy of 97% on the test set.
   - Implemented using Keras with TensorFlow backend.
   - Built a neural network with three layers: 128, 64, and 1 neuron(s) respectively, with 'elu' activation for hidden layers and 'sigmoid' activation for the output layer.
   - Compiled the model with 'adam' optimizer and 'binary_crossentropy' loss function.
   - Evaluated model performance using accuracy, confusion matrix, and F1 score.

## Usage
1. **Clone the repository**:
     ```
     https://github.com/muhammadsufyan1550/Thyroid-Cancer-Recurrence-Prediction.git
     ```

2. **Install the required dependencies**:
     ```
     pip install -r requirements.txt
     ```

3. **Run the Jupyter Notebook**:
   - Open the Jupyter Notebook `thyroid_cancer_prediction.ipynb` in your preferred Python environment (e.g., Jupyter Notebook or JupyterLab).
   - Execute the code cells in the notebook to reproduce the results of the thyroid cancer recurrence prediction project.

## Acknowledgments
- The dataset used in this project was sourced from [https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence].

Feel free to explore the code, experiment with different models or techniques, and contribute to further improvements!

If you have any questions or suggestions, please feel free to reach out.
