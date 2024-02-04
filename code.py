import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from keras.models import Sequential
from keras.layers import Dense

# Loading the dataset from the CSV file
data = pd.read_csv("/content/thyroid.csv")

# Splitting the data into features (X) and the target variable (y)
# The "Recurred" column is popped from the DataFrame and assigned to 'y'
y = data.pop("Recurred")
X = data

# Using 'get_dummies' function to perform One-Hot Encoding to the categorical features in 'X'
X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Classifier
# Scaling the features using StandardScalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC()                                       # Here SVC() initializes an SVM classifier
svm_model.fit(X_train_scaled, y_train)

svm_predictions = svm_model.predict(X_test_scaled)      # Predictions on the scaled test set (X_test_scaled).

# Computing accuracy, confusion amtrix and F1 score
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_conf_matrix = confusion_matrix(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions, pos_label='Yes')

print("SVM Accuracy:", svm_accuracy)
print("SVM Confusion Matrix:\n", svm_conf_matrix)
print("SVM F1 Score:", svm_f1)

# ANN Classifier using Keras
# Encoding categorical labels to numerical format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Building ANN model with three layers: 128, 64, and 1 neuron(s) respectively,
# using 'elu' activation for hidden layers and 'sigmoid' activation for the output layer.
ann_model = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], activation='elu'),
    Dense(64, activation='elu'),
    Dense(1, activation='sigmoid')
])

# Compiling the ANN model with 'RMSprop' optimizer, 'binary_crossentropy' loss function,
# and accuracy as the evaluation metric.
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the neural network model with 15 epochs, a batch size of 64, and 20% of the training data used for validation.
ann_model.fit(X_train_scaled, y_train_encoded, epochs=15, batch_size=64, validation_split=0.2)

# Thresholding the NN predictions at 0.5 to obtain binary classification results (0 or 1).
ann_predictions = (ann_model.predict(X_test_scaled) > 0.5).astype("int32")

# Converting ANN predictions to categorical format for consistency
ann_predictions_categorical = label_encoder.inverse_transform(ann_predictions.flatten())

# Evaluating accuracy, confusion matrix, and F1 score
ann_accuracy = accuracy_score(y_test, ann_predictions_categorical)
ann_conf_matrix = confusion_matrix(y_test, ann_predictions_categorical)
ann_f1 = f1_score(y_test, ann_predictions_categorical, pos_label='Yes')

# Showcasing the result of ANN
print("\nANN Accuracy:", ann_accuracy)
print("ANN Confusion Matrix:\n", ann_conf_matrix)
print("ANN F1 Score:", ann_f1)