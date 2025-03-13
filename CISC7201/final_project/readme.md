# GRP_BE

# Beijing Second-hand Property Price Prediction-Based on Lian Jia estate Platform

## 1. Introduction
This project aims to provide accurate price predictions to assist buyers, sellers, and investors in making informed decisions. We have utilized data from the Lianjia estate platform and applied various data processing and modeling techniques to achieve our goals.

## 2. Dataset Collection
The code for data collection is in the **lianjia_crawler.ipynb**.
We defined a class named LianjiaCrawler aimed at scraping second-hand housing market information from Lianjia.com for Beijing.

**Initialization Method**

- Sets up basic parameters such as total pages to scrape, batch size, and a mapping of Beijing's administrative districts.

**Browser Setup**

- Initializes a Chrome browser instance with Selenium, employing random user agents to mimic human behavior and evade detection. It also sets login cookies for accessing information that requires authentication.

**Scraping Logic**

- Page Loading and Interaction: Implements more natural delays via the function and simulates human behaviors like scrolling through a function, reducing the risk of being flagged as automated traffic.

- Parsing Housing Information ：Extracts and parses detailed information about each property listing from the webpage, including number of bedrooms, area, orientation, decoration status, floor level, construction year, etc., organizing this data into structured storage.

**Scraping a Specified District**

- Based on given district codes and names, it scrapes listings in batches for properties with and without elevators within that district. For each page, it attempts to gather as many valid data entries as possible. If there are consecutive errors, it stops automatically to prevent IP blocking due to excessive requests.

**Data Processing**

- Saving Batch Data: Stores each batch of data into CSV files, ensuring that already scraped data is preserved even if the program terminates unexpectedly.

- Merging CSV Files: After all batches have been scraped, it combines all CSV files into one comprehensive dataset and deletes the temporary individual batch files **(lianjia_houses_merged.csv)**.

**Exception Handling**

- Exception handling mechanisms are integrated at critical steps to ensure that the program can continue executing subsequent tasks or exit safely in case of errors, minimizing the potential loss of data.

**Dataset**
![alt Lianjia_Dataset](Picture/Dataset_picture/Lianjia_Daataset.png)

## 3.Data Preprocessing

The code for data preprocessing is in the **process_lianjia_data.ipynb**.

**Data Cleaning and Handling**

- Handling Missing Values: It removes records where the number of bedrooms or living rooms is null; for missing total floor numbers, it fills them with the median value rounded up.

- Categorical Feature Transformation:
  - Floor information is converted into English labels based on its description (e.g., '底层' to 'ground', '高楼层' to 'high'), with a default setting of 'middle'.
  - The construction year is categorized into segments and transformed into categorical data (such as 'before_2000', '2000_2010').
  - For building types, it fills missing values with 'plate_building' as the default.

**Feature Engineering**

- Average Price Calculation per Community: It calculates the average unit price grouped by community names and divides each community into five price levels accordingly.

- One-Hot Encoding: It applies one-hot encoding to the number of bedrooms, living rooms, and all categorical variables (such as orientation, decoration status, floor level, building type, construction year, district, community price level) to create new binary feature columns.

- Binary Variable Processing: Binary variables like elevator availability, proximity to subway, and ownership over 5 years are mapped to integer values of 0 or 1.

**Output Processed Data**

- Finally, it saves the processed DataFrame to a new CSV file named **processed_lianjia_data.csv**.
![alt Lianjia_Dataset](Picture/Dataset_picture/Preprocess.png)
## 4. Training

The code for training is in the **train.ipynb**.

**Model Selection**

- We experimented with several machine learning models, including Linear Regression, Decision Tree, Random Forest, and Gradient Boosting, to identify the best-performing model for our dataset.

**Training Methods**

- We employed three predictive analysis methods: Random Forest, MLP (Multi-Layer Perceptron), and Simple Decision Tree. Additionally, we performed PCA (Principal Component Analysis) on the source data and re-executed these three predictive analysis methods. Furthermore, we conducted enhanced training for the MLP method.


**Training Results**
Simple Decision Tree Regressor MSE with increased max_depth: 3285492.4020566302 
Simple Decision Tree Regressor MSE: 12704063.710866436
MSE of the pure Decision Tree regression model after PCA: 121836559.96824104
Random Forest Regressor MSE: 6758089.96849267
Random Forest Regressor MSE with PCA:210775278.4338572
Pure MLP Regressor MSE: 433872352.0
MLP Regressor MSE with PCA: 1535190144.0
Gradual MLP Regressor MSE: 629248064.0