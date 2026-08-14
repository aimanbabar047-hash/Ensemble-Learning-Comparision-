# Ensemble Learning: Random Forest vs XGBoost

## Project Overview

This project compares ensemble learning algorithms with traditional single machine learning models for **Telco Customer Churn Prediction**.

The main objective is to understand whether ensemble methods such as **Random Forest** and **XGBoost** can provide better predictive performance than individual models such as **Logistic Regression** and **Decision Tree**.

The project includes data preprocessing, model training, performance evaluation, feature importance analysis, and comparison of multiple classification models.

---

## Dataset

The project uses the **Telco Customer Churn dataset**.

The dataset contains customer information such as:

* Gender
* Senior Citizen status
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

### Target Variable

**Churn**

* `Yes` → 1
* `No` → 0

The `customerID` column was removed because it does not provide useful predictive information.

---

## Models Used

Four classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

The first two models were used as baseline/single-model approaches, while Random Forest and XGBoost were used as ensemble learning approaches.

---

## Technologies and Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* KaggleHub
* Jupyter Notebook

---

## Installation

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost kagglehub
```

Or install everything using the provided requirements file:

```bash
pip install -r requirements.txt
```

---

## Project Workflow

```text
Telco Customer Churn Dataset
            ↓
      Data Loading
            ↓
      Data Cleaning
            ↓
  Missing Value Handling
            ↓
    Feature Preparation
            ↓
Categorical Encoding + Scaling
            ↓
      Train/Test Split
            ↓
 ┌──────────┬───────────┬──────────────┬─────────┐
 ↓          ↓           ↓              ↓
Logistic   Decision    Random        XGBoost
Regression Tree        Forest
 ↓          ↓           ↓              ↓
 └──────────┴───────────┴──────────────┴─────────┘
            ↓
    Model Evaluation
            ↓
 Performance Comparison
            ↓
   Feature Importance
```

---

## Data Preprocessing

The following preprocessing steps were performed:

### 1. Total Charges Conversion

`TotalCharges` was converted into a numerical data type.

### 2. Missing Values

Missing values in `TotalCharges` were handled using the median.

### 3. Customer ID Removal

The `customerID` column was removed because it is an identifier rather than a predictive feature.

### 4. Target Encoding

The target variable was converted from:

```text
Yes → 1
No → 0
```

### 5. Categorical Encoding

Categorical features were transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

### 6. Numerical Feature Scaling

Numerical features were standardized using:

```python
StandardScaler()
```

### 7. Train-Test Split

The dataset was divided into:

* 80% Training Data
* 20% Testing Data

A fixed `random_state=42` and stratification were used for reproducibility.

---

## Model Performance Comparison

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Results

| Model               |   Accuracy |  Precision | Recall | F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | -----: | -------: | ---------: |
| Logistic Regression |     80.55% |     65.72% | 55.88% |   60.40% |     84.19% |
| Decision Tree       |     72.89% |     48.96% | 50.53% |   49.74% |     65.73% |
| Random Forest       |     77.50% |     59.79% | 46.52% |   52.33% |     81.87% |
| **XGBoost**         | **80.70%** | **67.35%** | 52.94% |   59.28% | **84.44%** |

---

## Results Analysis

### Logistic Regression

Logistic Regression achieved an accuracy of approximately **80.55%** and an ROC-AUC of **84.19%**.

It also achieved the highest Recall and F1 Score among the four models.

### Decision Tree

Decision Tree produced the lowest overall performance, with an accuracy of approximately **72.89%** and ROC-AUC of **65.73%**.

This shows that a single decision tree was less effective for this dataset compared with the other approaches.

### Random Forest

Random Forest improved upon the single Decision Tree in several metrics.

It achieved:

* 77.50% Accuracy
* 59.79% Precision
* 46.52% Recall
* 52.33% F1 Score
* 81.87% ROC-AUC

Random Forest combines multiple decision trees to reduce the limitations of a single tree.

### XGBoost

XGBoost achieved the best overall performance in this experiment.

It achieved:

* **80.70% Accuracy**
* **67.35% Precision**
* 52.94% Recall
* 59.28% F1 Score
* **84.44% ROC-AUC**

XGBoost therefore performed slightly better than Logistic Regression in Accuracy and ROC-AUC and achieved the highest Precision.

---

## Random Forest vs XGBoost

Random Forest builds many decision trees independently using randomly selected samples and features, then combines their predictions, typically through voting for classification. This approach mainly reduces variance and helps prevent overfitting compared with a single decision tree.

XGBoost builds trees sequentially, where each new tree focuses on correcting the errors made by the previous trees. It uses gradient boosting to minimize the prediction error step by step. Therefore, Random Forest mainly relies on **parallel independent trees**, while XGBoost relies on **sequential error-correcting trees**.

---

## Feature Importance

Feature importance was analyzed for both ensemble models to identify which customer characteristics contributed most to churn prediction.

### Random Forest

Random Forest calculates feature importance based on the contribution of features across its collection of decision trees.

### XGBoost

XGBoost calculates feature importance based on how features contribute to the boosted decision trees.

Comparing the feature importance plots helps identify which customer characteristics are considered important by both ensemble methods and where their rankings differ.

---

## Key Findings

* XGBoost achieved the highest overall accuracy.
* XGBoost achieved the highest precision.
* XGBoost achieved the highest ROC-AUC.
* Logistic Regression achieved the highest recall.
* Logistic Regression achieved the highest F1 Score.
* Decision Tree performed worst overall.
* Random Forest performed better than the single Decision Tree.
* Ensemble methods provide a useful comparison against traditional single models.
* Feature importance analysis helps understand which customer characteristics influence churn prediction.


## Conclusion

This project demonstrates how ensemble learning can be applied to customer churn prediction and compared with traditional machine learning models.

Among the tested models, **XGBoost provided the strongest overall performance**, achieving the highest Accuracy, Precision, and ROC-AUC. Random Forest also demonstrated the advantages of combining multiple decision trees compared with a single Decision Tree.

The comparison shows that model selection should not depend on accuracy alone. Metrics such as Precision, Recall, F1 Score, and ROC-AUC should also be considered depending on the business objective.

---

## Author

**Aiman Babar**

Machine Learning | Data Analytics | Bioinformatics

---

## Acknowledgement

This project was completed as part of the **Neurofive Solutions** learning/task program.
