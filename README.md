# ECON3916-33674-Statistical-Machine-Learning-Final-Project  
# 📊 ML Prediction Project – Netflix User Behavior  

## 📌 Overview  

This project explores whether user watch time can be predicted using basic demographic and preference-related features. The goal is to apply a simple machine learning workflow, including data exploration and baseline modeling.

More importantly, this project also evaluates what happens when the available features contain little to no predictive signal.

---

## 🎯 Prediction Question  

Can we predict how many hours a user spends watching content based on features such as age, country, subscription type, and favorite genre?

---

## 📂 Dataset  

- Source: Kaggle – Netflix Users Database  
- Observations (N): 25,000  
- Features used: Age, Country, Subscription_Type, Favorite_Genre  
- Dropped:
  - `User_ID` (identifier, no information)
  - `Last_Login` (timestamp not meaningful in this context)

---

## 🔍 Exploratory Data Analysis (EDA)  

### Key Findings  

- No missing values (0%)  
- No outliers detected (IQR method)  
- Target variable is continuous and spread across a wide range  
- Correlation analysis shows almost no linear relationship:
  - |Pearson r| < 0.02 for all features  
- Mutual information ≈ 0 → no nonlinear signal  

### Interpretation  

All diagnostics consistently indicate:

> The features contain almost **no predictive information** about watch time.

This suggests that the dataset may be **synthetically generated**, where the target variable is largely independent of the features.

---

## 🤖 Models Implemented  

### Model 1: Linear Regression (Baseline)  

- RMSE: ~285  
- MAE: ~245  
- R²: ≈ 0  

➡️ Interpretation:  
The model predicts values close to the mean, indicating that it does not capture meaningful patterns in the data.

---

### Model 2: Random Forest Regressor  

- Train R²: high (~0.8+)  
- Test R²: ≈ 0 or negative  
- CV R²: ≈ 0  

➡️ Interpretation:  
- Strong performance on training data  
- Poor performance on test data  

This is a classic case of **overfitting**, where the model memorizes noise rather than learning general patterns.

---

## 📈 Model Insights  

- The linear model fails to capture any meaningful relationship between features and watch time  
- The Random Forest introduces complexity but does not improve real predictive performance  
- Visualization of predicted vs actual values shows predictions clustering around a constant value  

➡️ Conclusion:  
> The limitation comes from the data, not the model choice.

---

## 💡 Key Takeaway  

The results suggest that watch time is difficult to predict using the current set of features.

> **Model performance is limited by data quality, not algorithm complexity.**

---

## 📊 Business Interpretation  

### ❌ Why Prediction Fails  

Using only:
- Demographics  
- Subscription type  
- Stated preferences  

➡️ Cannot explain real user behavior  

Watch time is influenced by:
- Dynamic behavioral patterns  
- Time-dependent usage  
- External factors not captured in the dataset  

---

### ✅ What Would Improve the Model  

**1. Add behavioral features**
- Session frequency  
- Viewing completion rate  
- Time-of-day activity  
- Clickstream data  

**2. Reframe the problem**
- Predict high vs low engagement users  
- Use ranking instead of exact prediction  

---

## ⚠️ Limitations  

- Dataset likely synthetic  
- Very weak or no feature-target relationship  
- Results may not generalize to real-world data  

---

## 🛠️ Tools Used  

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  

---

## 📌 Note  

This project focuses on demonstrating the modeling workflow and interpreting results, rather than achieving high predictive performance.
