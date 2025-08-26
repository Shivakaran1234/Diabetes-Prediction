Diabetes Prediction using Machine Learning
📌 Project Overview

This project predicts whether a patient is diabetic or not based on their medical attributes using the Pima Indians Diabetes Dataset.
We build and compare different ML models (Logistic Regression vs Random Forest) and analyze their performance using metrics like confusion matrix, ROC curve, and AUC.

Hyperparameter tuning is performed using GridSearchCV to optimize the Random Forest model.

📂 Dataset

Source: Kaggle - Pima Indians Diabetes Database

Features (8):

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Target (Outcome):

0 → Non-diabetic

1 → Diabetic

⚙️ Project Workflow

Data Preprocessing

Handle missing values (replace 0 with median values for medical attributes).

Normalize data using StandardScaler.

Train-test split (80:20).

Modeling

Logistic Regression (baseline model).

Random Forest Classifier (ensemble learning).

Hyperparameter Tuning

Use GridSearchCV for Random Forest optimization.

Evaluation

Accuracy, Precision, Recall, F1-score.

Confusion Matrix.

ROC Curve & AUC.

Feature Importance.

📊 Results

Logistic Regression: Good baseline, interpretable model.

Random Forest: Better performance, handles non-linear relationships.

GridSearchCV optimized Random Forest for maximum accuracy and AUC.

📦 Installation

Clone the repository and install dependencies:

git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt

▶️ Usage

Run the Jupyter Notebook:

jupyter notebook Diabetes_Prediction.ipynb


Or run Python script:

python diabetes_prediction.py

📈 Visualizations

Correlation Heatmap

Confusion Matrix

ROC Curve

Feature Importance

🛠️ Tech Stack

Python 🐍

Pandas, NumPy (data processing)

Matplotlib, Seaborn (visualization)

Scikit-learn (ML models & evaluation)

🚀 Future Improvements

Try other ML models (XGBoost, SVM, Neural Networks).

Deploy model with Flask/Django or Streamlit.

Build an API for healthcare integration.
