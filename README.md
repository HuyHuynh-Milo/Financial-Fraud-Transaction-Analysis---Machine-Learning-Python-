# 🚫 Financial-Fraud-Transaction-Analysis-Machine-Learning-Python-
<img width="1440" height="754" alt="Fraudulent-transactions" src="https://github.com/user-attachments/assets/af177de6-c0cf-45fa-9d66-c539d7929dd6" />

- Author: Huy Huynh
- Date: Sept 2025
- Tool: Python

---
## 📃 Background & Overview:

✅ **Objective:**

This project is designed to analyze and predict fraudulent financial transactions with the goal of enhancing fraud detection and prevention. The objectives include:
- **Feature engineering & analysis**: Develop meaningful features from raw transaction data to improve model accuracy and interpretability.
- **Identify fraud patterns**: Explore large-scale transaction data to uncover hidden trends and behavioral patterns associated with fraudulent activity.
- **Support investigation & prevention**: Provide insights that can assist organizations in detecting suspicious activity earlier, reducing financial losses, and strengthening risk management strategies.
- **Build predictive models**: Apply machine learning algorithms to classify transactions as fraudulent or non-fraudulent, and evaluate their performance.
- **Scalability & adaptability**: Ensure the solution can be applied to different datasets and adapted to evolving fraud techniques.

🔭 **Scope:**

The scope of this project defines the boundaries and focus areas for fraud detection and prediction:
- Data exploration & preprocessing: Cleaning, transforming, and preparing raw financial transaction data for analysis.
- Exploratory data analysis (EDA): Visualizing and understanding transaction patterns, fraud distribution, and correlations among features.
- Model development & evaluation: Building and testing machine learning models (e.g., Logistic Regression, Random Forest, XGBoost) to detect fraud.
- Performance measurement: Using metrics such as precision, recall, F1-score, and ROC-AUC to assess model effectiveness.
- Practical application: Providing insights and predictive tools that can support fraud investigators and risk management teams.
Out of scope: This project does not cover deployment into production systems, integration with real-time transaction monitoring platforms, or regulatory/legal aspects of fraud management. 

---
## 📂 Dataset description & structure
📎 **Data Source:**
- The dataset for this project is in the csv file name "mini-project2" attached to this repository
- Size: 1 dataframe contains 22 features and nearly 100,000 observations

📌 **Data Description:**

- The dataset contains detailed information about financial transactions, including customer demographics, merchant details, and fraud labels. Each row represents a single transaction with the following attributes:

| **Feature**                  | **Description**                                                |
| ---------------------------- | -------------------------------------------------------------- |
| **trans\_date\_trans\_time** | Date and time of the transaction                               |
| **cc\_num**                  | Credit card number (anonymized)                                |
| **merchant**                 | Merchant receiving the payment                                 |
| **category**                 | Business category of the merchant (e.g., retail, food, travel) |
| **amt**                      | Transaction amount in U.S. dollars                             |
| **first**                    | First name of the cardholder                                   |
| **last**                     | Last name of the cardholder                                    |
| **gender**                   | Gender of the cardholder (`Male` / `Female`)                   |
| **street**                   | Street address of the cardholder’s residence                   |
| **city**                     | City of the cardholder’s residence                             |
| **state**                    | State of the cardholder’s residence                            |
| **zip**                      | ZIP code of the cardholder’s residence                         |
| **lat**                      | Latitude of the cardholder’s residence                         |
| **long**                     | Longitude of the cardholder’s residence                        |
| **city\_pop**                | Population of the cardholder’s city                            |
| **job**                      | Occupation of the cardholder                                   |
| **dob**                      | Date of birth of the cardholder                                |
| **trans\_num**               | Unique transaction ID                                          |
| **unix\_time**               | Unix timestamp (seconds since Jan 1, 1970)                     |
| **merch\_lat**               | Latitude of the merchant’s location                            |
| **merch\_long**              | Longitude of the merchant’s location                           |
| **is\_fraud**                | Fraud label (`1 = Fraudulent`, `0 = Legitimate`)               |

---

## 🖥️ Computational Thinking Process:
To approach the fraud detection problem, I followed a computational thinking framework consisting of four main stages:

<img width="1769" height="817" alt="Computationnal thinking" src="https://github.com/user-attachments/assets/9d388ed1-575c-4938-8a5f-65c5b8121a70" />

---

## 📖 Main Process:
