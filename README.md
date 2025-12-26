# logistic-regression-from-scratch
A from-scratch implementation of Logistic Regression using NumPy. Covers mathematical foundations such as sigmoid activation, binary cross-entropy loss, and gradient descent optimization, built without scikit-learn or deep learning frameworks.


🚀 Logistic Regression From Scratch (NumPy)
A clean and minimal implementation of Logistic Regression built from scratch using NumPy.
This project focuses on understanding the mathematics and optimization behind Logistic Regression, without using machine learning libraries like scikit-learn or TensorFlow.

📌 Designed for learning, interviews, and showcasing strong ML fundamentals.

✨ Highlights
✔ Built entirely from scratch
✔ Uses Sigmoid activation
✔ Binary Cross-Entropy loss
✔ Manual Gradient Descent
✔ Simple, readable NumPy code
✔ Perfect for ML internships & interviews

📁 Repository Structure
📦 logistic-regression-from-scratch
 ┣ 📜 logistic_regression.py
 ┗ 📜 README.md

🧠 Model Workflow
Initialize weights and bias
Compute linear output
Apply Sigmoid activation
Calculate Binary Cross-Entropy loss
Compute gradients
Update parameters using Gradient Descent
Repeat for multiple epochs

📐 Mathematical Formulation

Sigmoid Function
sigmoid(z) = 1 / (1 + exp(-z))

Linear Model
z = XW + b

Binary Cross-Entropy Loss
loss = -mean( y * log(y_pred) + (1 - y) * log(1 - y_pred) )

Gradient Descent Update
W = W - learning_rate * dW
b = b - learning_rate * db

🛠 Requirements
Python 3.x
NumPy

Install NumPy:
pip install numpy


▶️ Usage Example
Python
import numpy as np
from logistic_regression import Logisic_Regression

# Training data
X = np.random.randn(5, 4)
y = np.array([[1], [0], [0], [1], [0]])

# Initialize model
model = Logisic_Regression(X, y)

# Train
model.train(epoch=1200, lr=0.01, show_loss=True)

# Test
X_test = np.random.randn(2, 4)
y_pred = model.predict(X_test)

print("Predicted probabilities:", y_pred)

📊 Output
Outputs probabilities in the range [0, 1]
Can be converted to class labels using a threshold (e.g., 0.5)

⚠️ Notes
Educational implementation
No regularization or numerical stability tricks
Not intended for production deployment

🎯 What You Learn From This
Logistic Regression internals
Gradient Descent optimization
Binary Cross-Entropy loss
Vectorized NumPy operations
Building ML models without black-box libraries

👤 Author
Priyam Khare
Machine Learning & AI Enthusiast

⭐ Why This Project Matters
Implementing Logistic Regression from scratch demonstrates strong ML fundamentals, which are highly valued in:
ML / AI internships
Technical interviews
