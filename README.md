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
- For EDA, some anaalysis must separate fraud and non-fraud observations so the analysis won't be biased.
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

**⏱️3. Fraud difference at hours**
- Fraud transactions and normal transactions may have different active hours during a day
```python
# Create fraud_only and non_fruad table
fraud_only = fraud_trans[fraud_trans['is_fraud'] == 1]
non_fraud = fraud_trans[fraud_trans['is_fraud'] == 0]

# Create countplot for those
plt.figure(figsize = (12,5))

# Countplot for fraud
plt.subplot(1,2,1)
sns.countplot(data = fraud_only, x = 'hour_trans', color = 'salmon')
plt.title('Fraud transaction by hour')
plt.xticks(ticks=range(0, 25, 2)) 

# Countplot for non-fraud
plt.subplot(1,2,2)
sns.countplot(data = non_fraud, x = 'hour_trans', color = 'skyblue')
plt.title('Non-Fraud transaction by hour')
plt.xticks(ticks=range(0, 25, 2)) 

plt.show()
```
<img width="1014" height="468" alt="image" src="https://github.com/user-attachments/assets/cce59b1b-2a65-49ea-89d8-4f7ea5dd1b85" />

- Fraud transactions usually occured at the late night time, from 22h-3h, especially at 22-23h 
- Normal transaction usually occured from 12h-23h

-> Check for cardholders who normally do not make late-hour transactions but suddenly make a purchase from 22h-3h, and those who always purchase at these times.

**💵4. Amount difference between fraud and normal transactions**
- Fraud transactions may focus on some value segment, compare fraud and normal transactions to find some trends:
```python
# Draw box plot to check how the value amounts distribute
plt.figure(figsize = (8,6))

sns.boxplot(data = fraud_trans,
            x = 'is_fraud',            # Check for both normal and fraud trans
            y = 'amt')

plt.xticks([0,1],['Non-fraud','Fraud'])     # Change the name for x-sticks
plt.title('Amount of Money by Fraud')
plt.xlabel('Is Fraud')
plt.ylabel('Amount of Money')
plt.ylim(-100,2000)                   # Set the limit for y_axis to maximum 2000

plt.show()
```
<img width="704" height="545" alt="image" src="https://github.com/user-attachments/assets/b8bad9c8-e6f4-4756-b389-215af44d3f59" />

- Normal transactions are very diverse, ranging from 1$ to more than 14000$ with many large transactions indicated by outliers (**I only show the plot to $2000 to focus on the 2 boxex itself**).
  - Focuses on low transactions (below $200) with the median at about $70
- Fraud transactions have a wider distribution, with a maximum transaction at $1300, with no outliers
  - Median value higher than normal trans (at about $400)

-> Fraud transaction focuses on a certain range of $ amount with higher value than normal transactions.

- Use KDE plot for a deeper understanding:

```python
plt.figure(figsize=(8,6))
sns.kdeplot(
    fraud_trans[fraud_trans.is_fraud==0]["amt"], 
    label="Non-fraud", 
    fill=True, alpha=0.4)
sns.kdeplot(
    fraud_trans[fraud_trans.is_fraud==1]["amt"], 
    label="Fraud", 
    fill=True, alpha=0.4
)
plt.legend()
plt.title("KDE Plot: Transaction Amount Distribution")
plt.xlim(-100,2000)

plt.show()
```
<img width="726" height="545" alt="image" src="https://github.com/user-attachments/assets/8e9a9329-2f55-4947-8c75-565f35c3f8cf" />

- Normal transactions distribution center around 1$ to approximately 200$, which is understandable because most normal trade activities fall within that range.
- For the Fraud Transaction things is more fluctual. There are 3 peaks with 3 different meanings:
    - Peak 1 from 1$ to 100$: Micro-Transactions, the villain make small transaction to check if the card is still active without drawing attention from bank or card holder
    - Peak 2 from 250$ to 400$: Medium-transactions, the villain make transaction amount that banks wont set under alarm. Usually they buy mid-value products and resell them for profit.
    - Peak 3 from 750$ to 1100$: High-value-transactions, the villain maximize profit after confirm that the card is clean. It might buy some jewelry, electronic or plane tickets

**🏢5. Fraud Merchant & Category**

**Category:**

- Check for fraud rate and fraud loss $ to find the most risky category
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
fraud_rate_merchant = ((fraud_merchant/non_fraud_merchant)*100).fillna(0)

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
<img width="1161" height="622" alt="fraud rate merchant" src="https://github.com/user-attachments/assets/ffd0389b-2922-4bf0-931e-6fb5735ee89f" />


- Most merchants have a fraud rate below 5%, and a few merchants show higher fraud rates (20-27%) but with low transaction counts. This suggests that a high number of transactions doesn't mean a higher fraud rate.
- On the other hand, merchants with higher fraud rates tend to be associated with higher transaction amounts. This indicates that the financial is more strongly related to the fraud rate than to the number of transactions.

Therefore, merchants with both high fraud rates and large transaction amounts should be prioritized for monitoring.

**Top 10 most fraud rate merchants**
- The limit anti-fraud resources and the relation with fraud rate and transaction amount $ -> focus on the top fraud rate merchants and their fraud loss $

```python
# Top 10 most fraud merchants 
fraud_merchant = fraud_trans[fraud_trans['is_fraud'] == 1]['merchant'].value_counts()
all_merchant = fraud_trans['merchant'].value_counts()
fraud_rate_merchant = ((fraud_merchant/all_merchant)*100).fillna(0).sort_values(ascending = False).head(10)

# Fraud money amt $ for top10 fraud merchant
amt_fraud_merchant = fraud_trans[fraud_trans['is_fraud'] == 1].groupby('merchant')['amt'].sum()
amtfraud_merchant = amt_fraud_merchant[amt_fraud_merchant.index.isin(fraud_rate_merchant.index)]

# merge these series into dataframe 
top_merchant = pd.concat(
                    [fraud_rate_merchant, amtfraud_merchant],
                    axis = 1).reset_index()
top_merchant.columns = ['merchant', 'fraud_rate_%', 'amt']

# Draw chart for top 10 fraud merchant
fig, ax1= plt.subplots(figsize = (12,6))

# Draw bar plot for fraud count
ax1.bar(top_merchant['merchant'],
        top_merchant['fraud_rate_%'],
        color = 'skyblue',
        label = 'Fraud Rate %')
ax1.set_ylabel('Fraud count', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Draw line plot for fraud amount
ax2 = ax1.twinx()
ax2.plot(top_merchant['merchant'],
         top_merchant['amt'],
         color = 'red',
         marker = 'o',
         label = 'Fraud Amount $')
ax2.set_ylabel('Fraud amount $', color = 'red')
ax2.tick_params(axis='y', labelcolor='red')
# Make the right column start with 0
# ax2.set_ylim(0, max(top_city['amt']) * 1.1)

# Make legend for ax1 and ax2
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles1+handles2, labels1+labels2,
           loc='upper right')   #  Same legend in the upper right

# Add title
plt.title('Top 10 merchants with most fraud (count vs amount $)')
ax1.tick_params(axis='x', rotation=45)

plt.show()
```

<img width="1110" height="766" alt="image" src="https://github.com/user-attachments/assets/9f0e1b47-2dfe-4ab0-981e-a4b76d5d10ae" />

- The top 10 highest fraud rates suggest that not every high fraud rate leads to high fraud loss. So we can just focus on some high-risk merchants like:
  - Kozey-Boehm, last_ltd, Bins and bfeffer, and Terry Huel
  - Those merchants have >20% fraud rate and > $40000 fraud loss

**🗺️6. Fraud in state and city aspects**

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

**City population aspect**

```python
# Calculate fraud_rate
fraud_city = fraud_trans[fraud_trans['is_fraud'] == 1]['city'].value_counts()
non_fraud_city = fraud_trans['city'].value_counts()
fraud_rate_city = ((fraud_city/non_fraud_city)*100).fillna(0)

# Turn fraud_city into dataframe
fraud_city_df = fraud_city.reset_index()
fraud_city_df.columns = ['city','fraud']

# Turn fraud_rate_city into DataFrame
fraud_rate_city_df = fraud_rate_city.reset_index()
fraud_rate_city_df.columns = ['city','fraud_rate']

# Filter the city max population (some city have many population data)
top_city_pop = fraud_trans[['city','city_pop']]
city_max_pop = top_city_pop.groupby('city', as_index=False)['city_pop'].max()

# Merge fraud table into city population table
pop_fraud = fraud_city_df.merge(city_max_pop, on = 'city', how = 'left') 

# Merge fraud rate table into city population table
pop_fraudrate = fraud_rate_city_df.merge(city_max_pop, on = 'city', how = 'left')
# print(pop_fraudrate)

# Draw scatter plot for fraud rate, fraud & population
plt.figure(figsize = (14,7))

# Plot for fraud vs population
plt.subplot(1,2,1)
sns.scatterplot(                  # Scatter plot
    data = pop_fraud,
    x = 'city_pop',
    y = 'fraud'
)
sns.regplot(                      # Regression line plot
    data = pop_fraud, 
    x="city_pop", 
    y="fraud", 
    line_kws={"color":"red"},
    ci = None                    # remove confidence interval 
)
plt.title("City Population vs Fraud")

# Plot for Fraud_rate vs Population
plt.subplot(1,2,2)
sns.scatterplot(                
    data = pop_fraudrate,
    x = 'city_pop',
    y = 'fraud_rate'
)
sns.regplot(                      # Regression line plot
    data = pop_fraudrate, 
    x="city_pop", 
    y="fraud_rate", 
    line_kws={"color":"red"},
    ci= None                     # remove confidence interval
)
plt.title("City Population vs Fraud Rate %")

plt.show()
```

 <img width="1155" height="622" alt="image" src="https://github.com/user-attachments/assets/e5bc8e28-6433-40b8-b955-e7daa98a46d0" />

- The graph indicate that larger city population tend to have more fraud. Its show that city population have a positive relation to fraud cases.
- However the fraud rate seems to have no relationship with city population
    -> Absolute volume scale with size, but rate does not.

**Relationship between fraud rate and number/amount$ of transactions**

```python
# Calculate fraud_rate
fraud_city = fraud_trans[fraud_trans['is_fraud'] == 1]['city'].value_counts()
non_fraud_city = fraud_trans['city'].value_counts()
fraud_rate_city = ((fraud_city/non_fraud_city)*100).fillna(0)

# Turn fraud_rate into dataframe
fraud_rate_city_df = fraud_rate_city.reset_index()
fraud_rate_city_df.columns = ['city','fraud_rate']

# Calculate total transactions each city
total_trans_city = fraud_trans['city'].value_counts()
# print(total_trans_city)

# Calculate total money amount each city
total_amt_city = fraud_trans.groupby('city')['amt'].sum()
# print(total_amt_city)

# Convert total transactions and total amount to dataframe
total_trans_city_df = total_trans_city.reset_index()
total_trans_city_df.columns = ['city','total_trans']

total_amt_city_df = total_amt_city.reset_index()
total_amt_city_df.columns = ['city','total_amt']

# Merge everything vào fraud_rate_city_df
final_city_df = fraud_rate_city_df.merge(total_trans_city_df, on='city', how='left') \
                                  .merge(total_amt_city_df, on='city', how='left')
# print(final_city_df)

# Draw plot for relationship with total trans, total amt to fraud rate
plt.figure(figsize = (14,7))

# Plot for fraud vs population
plt.subplot(1,2,1)
sns.scatterplot(                  # Scatter plot
    data = final_city_df,
    x = 'fraud_rate',
    y = 'total_trans'
)
sns.regplot(                      # Regression line plot
    data = final_city_df, 
    x = "fraud_rate",
    y = "total_trans",
    line_kws={"color":"red"},
    ci = None                    # remove confidence interval 
)
plt.title("Fraud Rate % vs Total transactions for city")

# Plot for Fraud_rate vs Population
plt.subplot(1,2,2)
sns.scatterplot(                
    data = final_city_df,
    x = 'fraud_rate',
    y = 'total_amt'
)
sns.regplot(                      # Regression line plot
    data = final_city_df,
    x = "fraud_rate",
    y = "total_amt",
    line_kws={"color":"red"},
    ci= None                     # remove confidence interval
)
plt.title("Fraud Rate % vs Total amount money for city")

plt.show()
```
<img width="1161" height="622" alt="image" src="https://github.com/user-attachments/assets/97c74696-20d6-4dc6-bbd7-26b51123a67d" />

- There is a negative relationship between Fraud rate and total transactions
  - Most city have fraud rate <40%, but almost every city have fraud rate > 20% have less than 100 transactions 
  - Some city have 100% fraud rate with small transaction volume, which is very fishy and need some attention
- Similarly, there is a downward trend: cities with higher fraud rates often correspond to lower total transaction amounts $.
  - Outliers exist: some cities with low fraud rates still handle very large total transaction amounts (up to $50,000).

-> Fraud is more concentrated in smaller-volume cities: places with fewer transactions and lower transaction amounts often show high fraud rates (even reaching 100%). High-transaction cities tend to have lower fraud rates, which makes sense because fraudsters are less able to dominate the transaction volume in larger markets.

**Top 10 cities with the most fraud volume**
```python
# Calculate Top 10 city have most fraud
top_fraud_city = fraud_trans[fraud_trans['is_fraud'] == 1]['city'].value_counts().head(10)
top_fraud_money_city = fraud_trans[fraud_trans['is_fraud'] == 1].groupby('city')['amt'].sum()      # Calculate fraud money each city
top_money = top_fraud_money_city[top_fraud_money_city.index.isin(top_fraud_city.index)]            # Filter top10 fraud city

# Connect top fraud and top money into 1 DataFrame
top_city = pd.concat(
    [top_fraud_city,top_money],
    axis = 1
).reset_index()

# Change the name of those column
top_city.columns = ['city', 'fraud', 'amt']

# Draw column and line plot for top 10 most fraud city
fig, ax1 = plt.subplots(figsize=(10,6))               # use subplots() 'cause it make axes easier

# Draw bar plot for fraud count (using the left side y-axis)
ax1.bar(top_city['city'], 
        top_city['fraud'], 
        color='skyblue', 
        label='Fraud count')
ax1.set_ylabel('Fraud count', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Draw line plot for fraud amount (using the right side y-axis)
ax2 = ax1.twinx()                                 # Create duplicate axes use same x-axis with ax1 
ax2.plot(top_city['city'], 
         top_city['amt'], 
         color='red', 
         marker='o', 
         label='Fraud amount $')
ax2.set_ylabel('Fraud amount $', color = 'red')
ax2.tick_params(axis='y', labelcolor='red')
# Ép trục bên phải bắt đầu từ 0
ax2.set_ylim(0, max(top_city['amt']) * 1.1)       # set limit min = 0, max = (max of the amt)*10% so the plot wont touch the ceiling

# Create legend for ax1 and ax2
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles1+handles2, labels1+labels2,
           loc='upper right')   # Legend chung, góc phải trên

# Add title
plt.title('Top 10 cities with most fraud (count vs amount)')
plt.xticks(rotation=45)

plt.show()
```
<img width="935" height="570" alt="image" src="https://github.com/user-attachments/assets/a9b7a797-050e-4792-8a6b-dd3a7ca0aa1f" />

- Houston seemed to have both highest fraud count at nearly 40 and most costly with nearly $22000
- Dallas has higher money value for each fraud transaction, only 28 fraud but costed $20000
- Warren, Naples and Tulsa are both have high fraud count and fraud money amount, Need to be taken care of.

--
### 🤖 4. Machine Learning predicts fraudulent transactions
