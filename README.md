%%writefile README.md
# Customer Churn Prediction using Artificial Neural Networks (ANN)

An end-to-end deep learning project that predicts bank customer churn using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras**. The project features automated model training with early stopping, experiment tracking with TensorBoard, and an interactive web interface powered by **Streamlit** to facilitate live classification requests.

---

## 🚀 Project Overview

Customer churn (attrition) is a major business challenge in the banking sector. Retaining existing customers is significantly more cost-effective than acquiring new ones. This repository implements a robust deep learning pipeline to identify high-risk customers, allowing business stakeholders to take proactive retention measures.

### Key Features
* **Advanced Preprocessing:** Handles numerical feature scaling, binary encoding, and One-Hot Encoding (OHE) for multiclass categorical variables.
* **ANN Architecture:** A highly optimized feedforward Neural Network utilizing modern best practices (ReLU activation, Sigmoid output layer).
* **Robust Callbacks:** Integrated `EarlyStopping` to prevent model overfitting and `TensorBoard` for real-time visualization of loss and accuracy metrics.
* **Production Deployment:** Lightweight interactive web app built with **Streamlit** for instant customer churn predictions.

---

## 🛠️ Tech Stack & Dependencies

* **Frameworks:** TensorFlow (Keras), Streamlit
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Serialization:** Pickle
* **Experiment Tracking:** TensorBoard
* **Hosting/Exposure:** Localtunnel (for local prototyping)

---

## 📊 Dataset

The model is trained on the classic **Churn Modelling Dataset** containing demographic and financial profiles of 10,000 bank customers.

### Key Features Used:
* **Demographics:** Geography, Gender, Age
* **Financial Profile:** Credit Score, Balance, Estimated Salary
* **Activity Metrics:** Tenure, Number of Products, Credit Card ownership (`HasCrCard`), Active membership status (`IsActiveMember`)
* **Target Variable:** `Exited` (1 if the customer churned, 0 otherwise)

---

## 📈 Performance & Results

During training, early stopping automatically halt the epoch execution to secure the model weights with the lowest validation loss.
* **Achieved Accuracy:** ~86%
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam (Learning Rate: `0.01`)

---

## 🔌 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🐳 Streamlit UI Preview

The web dashboard allows non-technical users to input customer attributes manually and get immediate results:
* **Interactive Controls:** Sliders for `Age`, `Tenure`, and `Number of Products`; dropdowns for `Geography` and `Gender`.
* **Real-time Inference:** Automatically scales and prepares raw inputs to match the ANN feature template, yielding a precise churn probability score.

---

## 📁 Directory Structure

```text
├── app.py                     # Streamlit application entry point
├── Churn_Modelling.csv        # Raw dataset source file
├── model.h5                   # Trained serialized Keras ANN model
├── label_encoder_gender.pkl          # Serialized Gender encoder
├── onehot_encoder_geo.pkl         # Serialized Geography encoder
├── scaler.pkl                 # Serialized StandardScaler template
├── logs/                      # TensorBoard logging directory
└── experiments.ipynb
|__requirements.txt                # Project documentation
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
