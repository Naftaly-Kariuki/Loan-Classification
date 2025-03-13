# Loan Classification Project

I built this repository to predict whether a loan application will be approved or rejected using multiple machine learning models i.e Logistic Regression, Decision Tree, Random Forest, and XGBoost.

## Overview
- **Goal**: Predict **loan_status** (Approved or Rejected).
- **Data**: A dataset containing borrower details such as credit score, income, asset values etc.
- **Models**: Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- **Result**: Achieved up to ~98% accuracy using XGBoost
## Project Steps

### 1. Data Loading
I began by importing the dataset into a pandas DataFrame and exploring its structure.  
![image](https://github.com/user-attachments/assets/4929e8d1-6a7a-4b50-ac19-15d16bb59c18)

### 2. Exploratory Data Analysis (EDA)
- **Missing Values**: Checked for any nulls and decided whether to drop or impute.  
- **Target Distribution**: Analyzed the balance of Approved vs. Rejected loans.

 ![image](https://github.com/user-attachments/assets/2af6f2b9-e576-41a6-95b8-5122b67eb0b4)

### 3. Data Preprocessing
1. **Dropping Irrelevant Columns**: Removed any unnecessary identifiers (e.g., `loan_id`).  
2. **Encoding the Target**: Mapped `loan_status` from `"Approved"/"Rejected"` to `1/0`.  
3. **One-Hot Encoding**: For categorical features like `education` and `self_employed`.  
4. **Feature Scaling**: Applied `StandardScaler`) to numerical features including `income_annum` , `residential_assets_value`, `commercial_assets_value`and `loan_amount`.  

### 4. Model Training with Cross-Validation
- **Cross-Validation**: Used 5-Fold StratifiedKFold to preserve class distribution in each fold.  
- **Models**: Trained Logistic Regression, Decision Tree, Random Forest, and XGBoost.  
- **Evaluation**: Measured accuracy for each fold, then averaged the scores to find the best-performing model.

![image](https://github.com/user-attachments/assets/c4bb4119-64bf-4677-9eea-edb7582f3395)


### 5. Model Evaluation
After XGBoost emerging as the best model, I performed a final train-test split:
- **Accuracy Score**: Provided an overall correctness measure.  
- **Classification Report**: Showed precision, recall, and F1-score for each class (Approved/Rejected).

![image](https://github.com/user-attachments/assets/ffc1dcb1-cfa2-4e7c-9c20-2885b34062b9)

- **Confusion Matrix**: Visualized false positives and false negatives.
![image](https://github.com/user-attachments/assets/37602b6b-db00-4803-aba0-a717af82c76c)

## Results
- **XGBoost** provided the highest accuracy (~98%) with a good balance of precision and recall.  
- **Random Forest** and **Decision Tree** also performed well but benefited from regularization to reduce overfitting.
![image](https://github.com/user-attachments/assets/42a51b25-b6d0-4f84-9e3a-9318937ecde9)
 
- **Logistic Regression** served as a strong baseline with an accuracy of ~90% accuracy.
![image](https://github.com/user-attachments/assets/8bd1b980-94ff-49ed-a71c-34adb7af4bf9)
