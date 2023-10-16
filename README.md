# Machine-Learning-Models

****MachineLearning_Project#1.jpynb****

**Problem**: Build ML models to predict stock close price and classify it as 'buy' or 'sell' depending on it's price

**Task Performed**

1) Loaded stock data of amazon using yfinance API
2) Performed Exploratory Data Analysis (EDA)
3) Built following ML models
   * K-Nearest Neighbors (KNN)
   * Random Forest Classifier (RF)
   * Gradient Boosting Classifier (GB)
   * Support Vector Machines (SVMs)
   * XGBoost Classifier
4) Strategy-1:
   If the next trading day's close price is greater than today's close price then, the signal is ‘buy’, otherwise ‘sell’.
   Strategy-2:
   Utilize the 50-day moving average vs the 200-day moving average
5) Evaluated Model performance based on Accuracy, Precision and Recall


****MachineLearning_Project#2.jpynb****

**Problem**: Develop machine learning models to detect anomalies in temperature data collected by a temperature measuring device, with the objective of identifying the specific time of day when the device malfunctions.

**Task Performed**

1) Loaded temperature device failure dataset
2) Performed Exploratory Data Analysis (EDA)
3) Performed Feature Engineering on the dataset such that new features to be added.
Specifically, you need to create a feature that will indicate the day of the week and time of the day. Namely, there should be four (4) categories (clusters?) for the feature, name it 'dtcat' (date-time-category):

*Weekday Day
*Weekday Night
*Weekend Day
*Weekend Night
Note: Some features such as ‘dayofweek’, ‘hours’, ‘day’, etc. may remain in the dataset.
We define the duration of ‘Day’ and Night’ as follows:
Duration of 'Day' should be defined: 7:00am - 7:00pm
Duration of 'Night' should be defined: 7:01pm - 6:59am
Ultimately, we would like to figure out when (weekday, weekend, day or night) the device fails!

4) Looked for Outliers using distance between centroid and data points
5) Applied the Gaussian distribution (EllipticEnvelope) algorithm
6) Applied the Isolation Forest algorithm at each category


**MachineLearning_Project#3.jpynb**

**Problem:** Create ML models to identify features in the dataset linked to heart failure-related deaths. Use LIME and eli5 with XGBoost to explain why the model predicts both positive (death) and negative (no death) outcomes. This helps us understand what factors influence these predictions.

**Task Performed**

1) Loaded Data into Dataframe using Python Pandas
2) Performed Exploratory Data Analysis (EDA)
3) Built following ML models with hyperparameters tuned
   * Logistic Regression (LR)
   * Random Forest Classifier (RF)
   * Gradient Boosting Classifier (GB)
   * XGBoost Classifier
4) Performed Machine Learning Interpretability/Explanability tasks as follows:
  * Use the **'eli5'** and **'LIME'** to interpret the "white box" model of Logistic Regression. Apply 'eli5' to visualize the weights associated to each feature. Use 'eli5' to explain specific predictions, pick a row in the test data with negative label and one with positive.
