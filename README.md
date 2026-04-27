# ECON3916-33674-Statistical-Machine-Learning-Final-Project  
# 📊 ML Prediction Project – Netflix User Behavior  

## 📌 Overview  

This project explores whether user watch time can be predicted using basic demographic and preference-related features. The goal is to apply a standard machine learning workflow, including exploratory data analysis, baseline modeling, and model evaluation.

More importantly, this project investigates a deeper question:  
> What happens when the data itself contains little to no predictive signal?

---

## 🎯 Prediction Question  

Can we predict how many hours a user spends watching content based on features such as:  
- Age  
- Country  
- Subscription type  
- Favorite genre  

Target variable: `Watch_Time_Hours` (continuous → regression problem)

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
- Target variable is continuous with a wide range  
- Correlation analysis:  
  - |Pearson r| < 0.02 for all features  
- Mutual information ≈ 0 → no nonlinear signal  

### Interpretation  

All diagnostics consistently indicate:

> The features contain almost **no predictive information** about watch time.

This is highly unusual for real-world behavioral data and suggests that the dataset may be **synthetically generated**, where the target variable is effectively independent of the features.

---

## 🤖 Models Implemented  

### Model 1: Linear Regression (Baseline)  

- RMSE: ~285  
- MAE: ~245  
- R²: ≈ 0  

➡️ Interpretation:  
The model predicts values very close to the mean. It explains **0% of the variance**, indicating no linear relationship between features and the target.

---

### Model 2: Random Forest Regressor  

- Train R²: high (~0.8+)  
- Test R²: ≈ 0 or negative  
- CV R²: ≈ 0  

➡️ Interpretation:  
- Strong performance on training data  
- Poor performance on test data  

This is a classic case of **overfitting**:  
> The model memorizes noise rather than learning a generalizable pattern.

---

## 📈 Model Insights  

- Both models perform nearly identically on unseen data  
- Predictions cluster around a constant value (the mean)  
- No model captures meaningful variation in watch time  

From visualization (actual vs predicted):  
- Points do **not align with the 45° line**  
- Instead, predictions form a horizontal band  

➡️ Conclusion:  
> The failure is not due to model choice, but due to lack of signal in the data.

---

## 💡 Key Takeaway  

This project highlights a fundamental principle in machine learning:

> **Model performance is limited by data quality, not algorithm complexity.**

Even flexible models like Random Forest cannot extract signal when:
- Features are weak or irrelevant  
- Target variable is independent of inputs  

---

## 📊 Business Interpretation  

### ❌ Why Prediction Fails  

Using only:
- Demographics  
- Subscription type  
- Stated preferences  

➡️ Cannot explain real user behavior  

Watch time is driven by:
- Dynamic behavior (daily usage patterns)  
- Contextual factors (time, mood, habits)  

---

### ✅ What Would Improve the Model  

To make this problem meaningful:

**1. Add behavioral features**
- Session frequency  
- Viewing completion rate  
- Time-of-day activity  
- Clickstream / interaction logs  

**2. Reframe the problem**
Instead of predicting exact hours:

➡️ Use classification or ranking  
- High vs low engagement users  
- Retention targeting  

---

## ⚠️ Limitations  

- Dataset likely synthetic  
- No real behavioral signal  
- Results may not generalize to real platforms  

---

## 🛠️ Tools Used  

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  

---

## 📌 Final Note  

This project focuses not on achieving high accuracy, but on correctly diagnosing **why models fail**.

> Understanding when a model *should not work* is just as important as making one work.
