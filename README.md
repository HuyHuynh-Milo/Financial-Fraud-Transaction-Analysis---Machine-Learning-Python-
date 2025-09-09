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

| **Feature**                  | **Description**                                                | **Dtype**
| ---------------------------- | -------------------------------------------------------------- |-----------
| **trans\_date\_trans\_time** | Date and time of the transaction                               | datetime64[ns]
| **cc\_num**                  | Credit card number (anonymized)                                | int64
| **merchant**                 | Merchant receiving the payment                                 | object
| **category**                 | Business category of the merchant (e.g., retail, food, travel) | object
| **amt**                      | Transaction amount in U.S. dollars                             | float64
| **first**                    | First name of the cardholder                                   | object
| **last**                     | Last name of the cardholder                                    | object
| **gender**                   | Gender of the cardholder (`Male` / `Female`)                   | object
| **street**                   | Street address of the cardholder’s residence                   | object
| **city**                     | City of the cardholder’s residence                             | object
| **state**                    | State of the cardholder’s residence                            | object
| **zip**                      | ZIP code of the cardholder’s residence                         | int64 
| **lat**                      | Latitude of the cardholder’s residence                         | float64
| **long**                     | Longitude of the cardholder’s residence                        | float64
| **city\_pop**                | Population of the cardholder’s city                            | int64
| **job**                      | Occupation of the cardholder                                   | object
| **dob**                      | Date of birth of the cardholder                                | datetime64[ns]
| **trans\_num**               | Unique transaction ID                                          | object
| **unix\_time**               | Unix timestamp (seconds since Jan 1, 1970)                     | int64
| **merch\_lat**               | Latitude of the merchant’s location                            | float64 
| **merch\_long**              | Longitude of the merchant’s location                           | float64
| **is\_fraud**                | Fraud label (`1 = Fraudulent`, `0 = Legitimate`)               | int64

---

## 🖥️ Computational Thinking Process:
To approach the fraud detection problem, I followed a computational thinking framework consisting of four main stages:

<img width="1769" height="817" alt="Computationnal thinking" src="https://github.com/user-attachments/assets/9d388ed1-575c-4938-8a5f-65c5b8121a70" />

---

## 📖 Main Process:
### 🔍 1. Data Observation:
- There are 24 features and 97748 observations.
- No null value
- No duplicate observation
- For unique values:
  - The dataset includes 693 merchants,51 states, 894 cities, and 14 different categories.
  
### ⚙️ 2. Features Engineering:
- Create hour_trans & weekday_trans columns ```
```python
fraud_trans['trans_date_trans_time'] = pd.to_datetime(fraud_trans['trans_date_trans_time'], format = "%Y-%m-%d %H:%M:%S")
fraud_trans['hour_trans'] = fraud_trans['trans_date_trans_time'].apply(lambda x: x.hour)
fraud_trans['weekday_trans'] = fraud_trans['trans_date_trans_time'].dt.dayofweek
```
- Create cardholder age when they made transaction column
```python
fraud_trans['dob'] = pd.to_datetime(fraud_trans['dob'], format = "%Y-%m-%d")
fraud_trans['year_old_dur'] = fraud_trans['trans_date_trans_time'] - fraud_trans['dob'] # delete this col
fraud_trans['year_old'] = round(fraud_trans['year_old_dur'].dt.total_seconds()/(365.25 *24 *60*60),0)
```
- Create Distance from merchants to cardholders column
```python
# Haversine formula to calculate distance (in kilometers)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    distance = R * c
    return distance

# Apply the function row-wise
fraud_trans['distance_km'] = haversine(
    fraud_trans['lat'], fraud_trans['long'],
    fraud_trans['merch_lat'], fraud_trans['merch_long']
```
### 📊 3. Exploratory Data Analysis
**📀 1. Fraud Ratio**
is_fraud  | Ratio
----------|--------
0         | 0.923211
1         | 0.076789

- Fraud percentage in this dataset is over 7.67%, an mid-imbalance ratio dataset
- For EDA, must separate fraud and non-fraud observations so the analysis won't be biased.
- For ML model, Accuracy is not a good metric for model validation, F1-score or recall should be considered.

**📈 2. Fraud over time**
- Its suspicious that the fraud happens usually sometimes of the year. The fraud over time must be considered.

```python
# Fraud growth over time
fraud_df = fraud_trans.copy()
fraud_df['year_month'] = fraud_df['trans_date_trans_time'].dt.to_period('M')

# Calculate fraud and amount by month
fraud_count_month = fraud_df[fraud_df['is_fraud'] == 1].groupby('year_month')['is_fraud'].count()
fraud_amt_month = fraud_df[fraud_df['is_fraud'] == 1].groupby('year_month')['amt'].sum()

# Create subplot 2 row, 1 columns, sharing x-axis
fig, (ax1, ax2) = plt.subplots(2, 1,                 # plt.subplots(2, 1, ..): tạo hai trục con (axes), xếp theo 2 hàng, 1 cột (2,1,..)
                               figsize=(10,8),       # fig (Figure): là toàn bộ khung của biểu đồ (hộp chứa mọi thứ)
                               sharex=True)          # ax (Axes): là vùng tọa độ chính để vẽ dữ liệu (trục X, trục Y, đường, nhãn..)
                                                                            
                                                                            
# Biểu đồ số lượng giao dịch fraud
ax1.plot(fraud_count_month.index.astype(str),       # vẽ trục x,index của series hiện đang ở dạng period nên phải chuyẻn qua string plt mới vẽ đc                       
         fraud_count_month.values,                  # vẽ trục y 
         color='red', 
         marker='o')
ax1.set_title('Fraud Count Over Time')
ax1.set_ylabel('Count')

# Biểu đồ số tiền fraud
ax2.plot(fraud_amt_month.index.astype(str), 
         fraud_amt_month.values, x`
         color='orange', 
         marker='s')
ax2.set_title('Fraud Amount Over Time')
ax2.set_ylabel('Amount ($)')

# Dùng chung trục x
plt.xlabel('Month')
plt.xticks(rotation=45)                               # rotate x-ticks by 45

plt.tight_layout()
plt.show()
```

<img width="989" height="790" alt="Fraud growth" src="https://github.com/user-attachments/assets/ad7c5a29-b665-45bb-9076-00a1897a313a" />

- The Fraud growth is fluctuating with many peaks in some month.
- Fraud happened a lot in December, nearly 600 cases of fraud and cost $330,000
  - It might be because of the holiday season which transactions happen a lot.
- May and October should be the next to be considered.

**⏱️3. Fraud difference at hour**
```python
# Create fraud_only and non_fruad table
fraud_only = fraud_trans[fraud_trans['is_fraud']columns
plt.title('Fraud rate (%) by Category')
plt.ylabel('Category')
plt.xlabel('Fraud Rate %')

# Money amount by category graph
plt.subplot(1,2,2)
sns.barplot(y = top_segment_amt.index,
            x = top_segment_amt.values,
            palette = 'Blues_r')
plt.title('Fraud Money by Category')
plt.ylabel('Category')
plt.xlabel('Money')

plt.show()
```
<img width="1699" height="699" alt="Fraud by category" src="https://github.com/user-attachments/assets/2716d75e-6ec9-43f1-a0c3-c8d693f40a69" />

- The left plot, showing fraud rate (%) by category, indicates that "shopping_net" has the highest fraud rate at around 20%, followed by "misc_net" and "grocery_pos" with rates slightly below 15%. Other categories like "gas_transport" and "misc_pos" have moderate rates around 5-7.5%.
  - This suggests online shopping and miscellaneous transactions are the most fraud-prone categories.
- The right plot, depicting fraud money amount by category, highlights "shopping_net" again as the leader, with the highest monetary loss, followed by "shopping_pos" and "misc_net".
  - This indicates that financial impact is most significant in online shopping and miscellaneous transactions.

**Merchant**
- Find out any relationship between fraud rate and transaction/transaction amount
```python
# Relationship of fraud rate to number of transactions & Money amount for merchant aspect

# Calculate fraud_rate
fraud_merchant = fraud_trans[fraud_trans['is_fraud'] == 1]['merchant'].value_counts()
non_fraud_merchant = fraud_trans[fraud_trans['is_fraud'] == 0]['merchant'].value_counts()
fraud_rate_merchant = (fraud_merchant/non_fraud_merchant).fillna(0)

# Turn fraud_rate into dataframe
fraud_rate_merchant_df = fraud_rate_merchant.reset_index()
fraud_rate_merchant_df.columns = ['merchant','fraud_rate']

# Calculate total transactions each merchants
total_trans_merchant = fraud_trans['merchant'].value_counts()
# print(total_trans_merchant)

# Calculate total money amount each merchant
total_amt_merchant = fraud_trans.groupby('merchant')['amt'].sum()
# print(total_amt_merchant)

# Convert total transactions and total amount to dataframe
total_trans_merchant_df = total_trans_merchant.reset_index()
total_trans_merchant_df.columns = ['merchant','total_trans']

total_amt_merchant_df = total_amt_merchant.reset_index()
total_amt_merchant_df.columns = ['merchant','total_amt']

# Merge everything vào fraud_rate_merchant_df
final_merchant_df = fraud_rate_merchant_df.merge(total_trans_merchant_df, on='merchant', how='left') \
                                  .merge(total_amt_merchant_df, on='merchant', how='left')
```
```python
# Draw plot for relationship with total trans, total amt to fraud rate
plt.figure(figsize = (14,7))

# Plot for fraud vs population
plt.subplot(1,2,1)
sns.scatterplot(                  # Scatter plot
    data = final_merchant_df,
    x = 'fraud_rate',
    y = 'total_trans'
)
sns.regplot(                      # Regression line plot
    data = final_merchant_df, 
    x = "fraud_rate",
    y = "total_trans",
    line_kws={"color":"red"},
    ci = None                    # remove confidence interval 
)
plt.title("Fraud Rate vs Total transactions for merchant")

# Plot for Fraud_rate vs Population
plt.subplot(1,2,2)
sns.scatterplot(                
    data = final_merchant_df,
    x = 'fraud_rate',
    y = 'total_amt'
)
sns.regplot(                      # Regression line plot
    data = final_merchant_df,
    x = "fraud_rate",
    y = "total_amt",
    line_kws={"color":"red"},
    ci= None                     # remove confidence interval
)
plt.title("Fraud Rate vs Total amount money for merchant")

plt.show()
```
<img width="1166" height="622" alt="fraud rate merchant" src="https://github.com/user-attachments/assets/4ca56a31-bb57-46c6-a67f-e902c1964487" />


- Most merchants have a fraud rate below 5%, and a few merchants show higher fraud rates (15–30%) but with low transaction counts. This suggests that a high number of transactions doesn't mean a higher fraud rate.
- On the other hand, merchants with higher fraud rates tend to be associated with higher transaction amounts. This indicates that the financial risk is more strongly related to the fraud rate than to the number of transactions.

Therefore, merchants with both high fraud rates and large transaction amounts should be prioritized for monitoring.

**Top 10 most fraud merchants**
- The limit anti-fraud resources make us focus on the top fraud merchants first

<img width="1110" height="678" alt="Top 10 fraud merchant" src="https://github.com/user-attachments/assets/910437d9-5e83-48ea-a383-5fca469f71c4" />

- The top 10 fraud by merchants show 10 name that be targeted the most, each one have more than 45 fraud transactions in just 18 months
  - Cormier LLC, Kozey-Boehm, Padberg-Welch, and Terry-Huel have very high fraud and fraud amount value.

**🗺️4. Fraud in state and city aspect**

**Top 10 most fraud state**
```python
# Top 10 state have most fraud by card holder
fraud_state = fraud_trans[fraud_trans['is_fraud'] == 1]['state'].value_counts().head(10)

# Create top10 fraud by state dataframe for map graph
fraud_by_state = { 
        'state' : fraud_state.index,
        'fraud' : fraud_state.values
}
fraud_state_df = pd.DataFrame(fraud_by_state)

# Draw Top 10 state with most fraud
import plotly.express as px

# Create state coordinates dictionary
state_coords = {
    'NY': {'lat': 43.2994, 'lon': -74.2179},
    'TX': {'lat': 31.9686, 'lon': -99.9018},
    'PA': {'lat': 41.2033, 'lon': -77.1945},
    'CA': {'lat': 36.7783, 'lon': -119.4179},
    'OH': {'lat': 40.4173, 'lon': -82.9071},
    'FL': {'lat': 27.6648, 'lon': -81.5158},
    'IL': {'lat': 40.6331, 'lon': -89.3985},
    'MI': {'lat': 44.3148, 'lon': -85.6024},
    'AL': {'lat': 32.3182, 'lon': -86.9023},
    'MN': {'lat': 46.7296, 'lon': -94.6859}
}

# Adding lat and lon columns into fraud_state_df
# Add latitude column to the dataframe by mapping from state code
fraud_state_df['lat'] = fraud_state_df['state'].map(lambda x: state_coords[x]['lat'])

# Add longitude column to the dataframe by mapping from state code
fraud_state_df['lon'] = fraud_state_df['state'].map(lambda x: state_coords[x]['lon'])

# Create scatter plot on US map
fig = px.scatter_geo(
    fraud_state_df,
    lat='lat',                   # latitude values
    lon='lon',                   # longitude values
    size='fraud',                # bubble size is proportional to fraud count
    color='fraud',               # bubble color also based on fraud count
    text='state',                # display state on bubble
    hover_name='state',          # show state name on hover tooltip
    projection="albers usa",     # projection type for USA map (phép chiếu)
    scope="usa"                  # focus on USA map only
)

# Update text style inside bubbles
fig.update_traces(
    textfont=dict(color="white", size=12),  # set text color to white, font size = 12
    textposition="middle center"            # place text at the center of each bubble
)

# Update layout of the map
fig.update_layout(
    title="Top 10 Fraud cases by US State",        # title of the chart
    geo=dict(                               # "geo" is a dictionary in plotly that include the option for the geometry map like scope, projection, landcolor,..
        scope='usa',                        # focus map on USA
        projection=dict(type='albers usa'), # use "albers usa" projection
        showland=True                       # show land area shading
    )
)

fig.show()  
```

<img width="1477" height="604" alt="Top 10 fraud state" src="https://github.com/user-attachments/assets/0da859eb-2a34-48ea-8e0a-6253e41a1442" />

- Most of the top fraud cases are concentrated in large and economically strong states such as CA, NY, TX, and FL. This may suggest a correlation between fraud and population or transaction volume.


 
