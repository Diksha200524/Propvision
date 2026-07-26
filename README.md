# 🏠 PropVision - AI Powered Real Estate Price Prediction

PropVision is an end-to-end Machine Learning web application that predicts the prices of **Flats**, **Independent Houses**, and **Residential Plots** in Gurugram using trained ML models.

The application also provides:

- 📈 Investment Analysis
- 🏦 EMI Calculator
- 🏘 Similar Property Recommendations
- 📊 Interactive Streamlit Dashboard

---

## 🚀 Features

### 🏢 Flat Price Prediction
- Predict flat prices
- EMI Calculator
- Investment Analysis
- Similar Flats

### 🏡 House Price Prediction
- Predict independent house prices
- EMI Calculator
- Investment Analysis
- Similar Houses

### 🌍 Plot Price Prediction
- Predict residential plot prices
- EMI Calculator
- Investment Analysis
- Similar Plots

---

## 🛠 Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Random Forest Regressor
- Pipeline
- ColumnTransformer
- StandardScaler
- OneHotEncoder

### Data Processing
- Pandas
- NumPy

### Model Serialization
- Joblib

### Frontend
- Streamlit

### Version Control
- Git
- GitHub

---

## 📂 Project Structure

```
propvision/
│
├── app.py
├── README.md
├── requirements.txt
│
├── datasets/
│   ├── flat_final.csv
│   ├── cleaned_house_final.csv
│   └── cleaned_plot_final.csv
│
├── models/
│   ├── flat_model.pkl
│   ├── house_model.pkl
│   └── plot_model.pkl
│
├── modules/
│   ├── predict.py
│   ├── features.py
│   ├── emi.py
│   └── investment.py
│
├── pages/
│   ├── Flats.py
│   ├── Houses.py
│   └── Plots.py
│
└── training/
    ├── train_flat.py
    ├── train_house.py
    └── train_plot.py
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/propvision.git
```

Go inside the project

```bash
cd propvision
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Data Preprocessing
5. Model Training
6. Model Evaluation
7. Model Serialization
8. Streamlit Deployment

---

## 📈 Model Used

Random Forest Regressor

Evaluation Metrics

- R² Score
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

---

## 📸 Screenshots

### Home Page

(Add Screenshot)

### Flat Prediction

(Add Screenshot)

### House Prediction

(Add Screenshot)

### Plot Prediction

(Add Screenshot)

---

## 🌟 Future Improvements

- Property Recommendation System
- Interactive Price Trends
- Price Prediction Confidence Interval
- Rental Price Prediction
- Map Integration
- User Authentication
- Database Integration
- Cloud Deployment

---

## 👨‍💻 Author

**Diksha**

AI & Machine Learning Enthusiast


## ⭐ If you like this project

Please give it a ⭐ on GitHub.
