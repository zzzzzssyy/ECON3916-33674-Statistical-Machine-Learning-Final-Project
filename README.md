# ECON3916-33674-Statistical-Machine-Learning-Final-Project
# 📊 ML Prediction Project – Netflix User Behavior

## 📌 Overview

This project explores whether user watch time can be predicted using basic demographic and preference-related features. The goal is to apply a simple machine learning workflow, including data exploration and baseline modeling.

---

## 🎯 Prediction Question

Can we predict how many hours a user spends watching content based on features such as age, country, subscription type, and favorite genre?

---

## 📂 Dataset

* Source: Kaggle – Netflix Users Database
* Observations (N): 25,000
* Features: 6 variables (excluding ID)
* Target: `Watch_Time_Hours`

---

## 🔍 Exploratory Data Analysis (EDA)

### Key Findings:

* The target variable (watch time) is continuous and spread across a wide range.
* The dataset contains no missing values and no detected outliers.
* Correlation analysis shows almost no linear relationship between age and watch time.
* No strong relationships were found between individual features and the target variable.

These results suggest that predicting watch time may be challenging using the available features.

---

## 🤖 Models Implemented

### Model 1: Linear Regression (Baseline)

A simple linear regression model was used as a baseline.

* RMSE: ~285
* MAE: ~245
* R²: ≈ 0

The model predicts values close to the mean, indicating that it does not capture meaningful patterns in the data.

---

### Model 2: Random Forest Regressor

A Random Forest model was applied to explore whether a more flexible model could improve performance.

Compared to linear regression, this model introduces more variation in predictions, but overall predictive performance remains limited.

---

## 📈 Model Insights

* The linear model fails to capture any meaningful relationship between features and watch time.
* The Random Forest model shows slight improvement but does not significantly enhance prediction accuracy.
* Visualization of predicted vs actual values confirms that the models struggle to fit the data.

---

## 💡 Key Takeaway

The results suggest that watch time is difficult to predict using the current set of features. Both EDA and model performance indicate weak relationships between inputs and the target variable. This implies that additional or more informative features would likely be needed to improve prediction accuracy.

---

## 🛠️ Tools Used

* Python
* Pandas
* Scikit-learn
* Matplotlib / Seaborn

---

## 📌 Note

This project focuses on demonstrating the modeling workflow and interpreting results, rather than achieving high predictive performance.
