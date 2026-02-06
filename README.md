# 🧬 Breast Cancer Classification System using Neural Networks

An end-to-end **Machine Learning + Deep Learning** project for **Breast Cancer Diagnosis Classification** using structured medical data and a **Neural Network model** built with **TensorFlow / Keras**.

---

## 📌 Project Overview

This project aims to classify tumors as:

- 🔴 **Malignant (M)**  
- 🟢 **Benign (B)**  

Using a fully connected **Neural Network** trained on medical diagnostic features.

---

## 🎯 Objective

Build an intelligent predictive system that:
- Analyzes tumor measurements
- Learns patterns from historical medical data
- Classifies tumors accurately
- Provides real-time predictions for new patient data

---

## 📂 Dataset Description

**Source:** CSV medical dataset (`data.csv`)  
**Features:** 30 numerical medical features  
**Target column:** `diagnosis`

### Original Labels:
| Diagnosis | Meaning |
|------|--------|
| B | Benign |
| M | Malignant |

### Encoded Labels:
| Class | Value |
|------|-------|
| Benign | 0 |
| Malignant | 1 |

---

## 🧠 Machine Learning Pipeline

1. Data loading  
2. Data cleaning  
3. Null value handling  
4. Column filtering  
5. Feature selection  
6. Label encoding  
7. Statistical analysis  
8. Data visualization  
9. Feature-target separation  
10. Train-test splitting  
11. Feature scaling (Standardization)  
12. Neural network modeling  
13. Model training  
14. Model validation  
15. Model evaluation  
16. Performance visualization  
17. Prediction system  
18. Real-time inference  

---

## 🧹 Data Cleaning

- Dropped `Unnamed: 32` column (contains only NaN values)
- Dropped `id` column (non-informative identifier)
- Encoded `diagnosis` column
- Created new target column: `target`

---

## 📊 Data Visualization

- Class distribution using `seaborn.countplot`
- Statistical analysis using `describe()`
- Group analysis using `groupby()`

---

## ⚙️ Dependencies

```text
Python 3.x
numpy
pandas
scikit-learn
tensorflow
keras
seaborn
matplotlib
Install:

pip install numpy pandas scikit-learn tensorflow keras seaborn matplotlib
🧪 Data Preparation
Features & Target:
X → All feature columns
Y → target column
Split:
Training set: 80%
Testing set: 20%
Validation set: 10% (from training)
📏 Feature Scaling
Method: StandardScaler
Technique: Z-score normalization
(mean = 0, std = 1)
🧠 Neural Network Architecture
Input Layer:
Flatten (30 features)

Hidden Layer:
Dense(20 neurons, ReLU activation)

Output Layer:
Dense(2 neurons, Sigmoid activation)
⚡ Model Configuration
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metric: Accuracy
🏋️ Training Configuration
Epochs: 15
Validation Split: 0.1
Random Seed: 3 (for reproducibility)
📈 Training Performance
Training Accuracy: ~97%

Validation Accuracy: ~97.8%

Test Accuracy: ~95.6%

📊 Model Evaluation
Test Accuracy ≈ 95.6%
📉 Performance Visualization
Accuracy vs Epoch

Validation Accuracy vs Epoch

Loss vs Epoch

Validation Loss vs Epoch

🔮 Prediction System
Input:
input_data = (11.76, 21.6, 74.72, 427.9, 0.08637, ...)
Pipeline:
Convert to NumPy array

Reshape to model format

Standardize input

Predict probabilities

Apply argmax()

Final classification

🧾 Output Classes
0 → Malignant
1 → Benign
🩺 Sample Prediction Output
Prediction Probabilities: [[0.8421648  0.02204067]]
Predicted Class: 0
Result: The tumor is Malignant
🚀 How To Run
python breast_cancer_nn.py
🏥 Real-World Applications
Medical diagnosis support systems

Clinical decision support tools

AI-assisted oncology systems

Hospital AI platforms

HealthTech solutions

Medical research

Smart healthcare systems

Predictive diagnostics

🔮 Future Enhancements
Multi-layer deep neural networks

Dropout regularization

Batch normalization

Hyperparameter tuning

Ensemble models

Explainable AI (XAI)

Model interpretability (SHAP, LIME)

API deployment (FastAPI)

Web interface

Cloud deployment

Mobile integration

Medical-grade AI compliance

👩‍💻 Author
Shereen Alaa
Machine Learning Engineer

GitHub: https://github.com/shreenalaa

LinkedIn: https://www.linkedin.com/in/shreen-alaa/

✨ This project demonstrates a complete medical AI pipeline:
Data Processing → Machine Learning → Deep Learning → Evaluation → Prediction → Deployment Logic
