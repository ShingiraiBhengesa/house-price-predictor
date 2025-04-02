# 🏡 House Price Prediction Web App

A Machine Learning project that predicts house prices based on input features such as square footage, number of bedrooms, location, and more. Built using Python, Scikit-learn, Pandas, and deployed with Flask. 

## 🚀 Demo
🔗 [Live Demo](#) — *(Link to be updated once deployed)*

---

## 📌 Project Overview

This project is a complete machine learning pipeline for predicting house prices using regression techniques. It includes:
- **Exploratory Data Analysis (EDA)**
- **Feature engineering & preprocessing**
- **Model training and evaluation**
- **Interactive web application** built with Flask
- **Deployment to the web**

---

## 📂 Dataset

**Primary Dataset**: California Housing Dataset from Scikit-learn  
Alternative: [Kaggle House Prices Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

Features include:
- Median income
- Housing median age
- Number of rooms
- Population
- Location (latitude/longitude)
- Median house value (target variable)

---

## 🧠 Machine Learning Approach

- **Type**: Supervised Learning – Regression
- **Algorithms Used**:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor (Best performing)

- **Evaluation Metrics**:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Python` | Core programming language |
| `Pandas` | Data manipulation and analysis |
| `NumPy` | Numerical operations |
| `Matplotlib & Seaborn` | Data visualization |
| `Scikit-learn` | ML modeling and preprocessing |
| `Flask` | Web framework for deployment |
| `HTML/CSS` | Frontend for the app |
| `Git & GitHub` | Version control and source code hosting |

---

## 🖼️ Visualizations

- Correlation heatmaps
- Feature importance plot
- Actual vs. predicted price scatter plot

*All available in the Jupyter Notebook under `/notebooks/EDA_and_Modeling.ipynb`*

---

## 🌐 Web Application

### 🎯 Features
- User-friendly input form
- Predicts house price instantly
- Simple and clean UI using HTML & Flask

### 🧪 Input Example
| Feature | Example Value |
|---------|---------------|
| Median Income | 8.3252 |
| Housing Age | 41 |
| Total Rooms | 880 |
| Bedrooms | 129 |
| Population | 322 |
| Latitude | 37.88 |
| Longitude | -122.23 |

---

## 💾 Installation (Run Locally)

1. Clone the repo:
```bash
git clone https://github.com/yourusername/house-price-predictor.git
cd house-price-predictor
