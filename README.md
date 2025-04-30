# Breast-Cancer-Survival-Prediction
A machine learning project to predict the survival of breast cancer patients using a Random Forest Classifier trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset**. This project demonstrates how ML can be applied in **healthcare** to drive impactful decision-making.

---

## 📌 Overview

Breast cancer is a leading cause of mortality among women. Early detection and prediction of survival rates can assist doctors in planning treatment strategies. This project builds a classification model to predict whether a tumor is malignant or benign using clinical and imaging features.

---

## 📊 Dataset

- Source: `sklearn.datasets.load_breast_cancer()`
- Samples: `569`
- Features: `30` numerical features (mean, standard error, worst values for radius, texture, perimeter, etc.)
- Target:
  - `0` = Malignant
  - `1` = Benign

---

## 🛠️ Tech Stack

- **Python** (v3.8+)
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**

🎯 Results
✅ Accuracy: ~95-98%
🔍 Key Influential Features: worst radius, mean concave points, worst perimeter, etc.
📌 Low false positive rate for malignant class — important in medical predictions

