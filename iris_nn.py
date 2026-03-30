import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras import layers, models 
import numpy as np

df = pd.read_csv("iris.csv")

print(df.head(10)) # first 10 rows
print(df.info())   # Dataset info
print(df.describe()) #summary stats

null_count = df.isnull().sum().sum()
if null_count > 0:
    print(f"Found {null_count} missing values. Cleaning now.....")
    df.dropna()
    print("Cleaned")
else:
    print("No missing values found.Dataset is already clean.")

le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data Preprocessing Complete")

model = models.Sequential([layers.Dense(16, activation = 'relu', input_shape = (X_train.shape[1],)),layers.Dense(8, activation = 'softmax')])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

print("\nStarting Training...")
model.fit(X_train, y_train, epochs = 50, batch_size = 8, verbose = 1)

print("\n--Evaluating on Test Data--")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:2f}%")

predictions = model.predict(X_test[:5])
predicted_classes = np.argmax(predictions, axis=1)

print("\n--Sample Predictions--")
print(f"Predicted labels: {predicted_classes}")
print(f"Actual labels: {y_test[:5].values}")

print("\nSaving the model....")
model.save('iris_model.h5')
print("Model saved as iris_model.h5")
