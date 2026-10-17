# ❤️ Heart Disease Prediction Using Machine Learning

A machine learning web application that predicts the likelihood of heart disease using **Logistic Regression** and patient health parameters.

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. This project uses machine learning to estimate the likelihood of heart disease based on input health measurements.

The application is built with Python and Streamlit, with a trained Logistic Regression model used to generate predictions.

## ✨ Features

- Predicts the likelihood of heart disease from patient inputs.
- Uses a trained Logistic Regression model.
- Applies feature scaling using a saved scaler.
- Maintains the expected input feature order.
- Provides an interactive web interface using Streamlit.

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Model Storage:** Joblib / Pickle

## 📂 Project Structure

```text
Heart-Diseases-Prediction-Model/
├── app.py
├── columns_heart.pkl
├── LR_heart.pkl
├── scaler_heart.pkl
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/alkashbaig/Heart-Diseases-Prediction-Model.git
cd Heart-Diseases-Prediction-Model
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### 4. Run the application

```bash
streamlit run app.py
```

## 🧠 Model Information

- **Algorithm:** Logistic Regression
- **Preprocessing:** Feature scaling
- **Saved artifacts:** Trained model, scaler, and input column information

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. Its predictions are not a substitute for professional medical advice, diagnosis, or treatment.

## 👨‍💻 Author

**Alkash Baig**

- GitHub: [@alkashbaig](https://github.com/alkashbaig)

---

If you found this project interesting, consider giving the repository a ⭐.g