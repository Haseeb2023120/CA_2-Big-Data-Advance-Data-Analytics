# CA_2- MSC_DA_BD_ADAv5

Identify and Carry out an analysis of a large a Dataset gleaned from X ( Foremerly Twitter) api
#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install pymongo')


# In[3]:


import pymongo
import pandas as pd
import json


# In[4]:


client=pymongo.MongoClient("mongodb://localhost:27017")





# Load the dataset
file_path = 'ProjectTweets.csv'
df = pd.read_csv(file_path, sep=',', names=['ids', 'date', 'flag', 'user', 'text'], skiprows=1)  # Skip the first row if it's a header row


# In[12]:


df.head()


# In[13]:





# In[14]:


data=df.to_dict(orient="records")


# In[15]:


data


# In[16]:


db=client["tweet_db"]


# In[17]:


print(db)


# In[18]:


db.tweet.insert_many(data)

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pymongo')


# In[1]:


import pymongo
import pandas as pd
import json


# In[2]:


client=pymongo.MongoClient("mongodb://localhost:27017")


# In[3]:


# Load the dataset
file_path = 'ProjectTweets.csv'
df = pd.read_csv(file_path, sep=',', names=['ids', 'date', 'flag', 'user', 'text'], skiprows=1)  # Skip the first row if it's a header row


# In[4]:


df.head()


# In[5]:


data=df.to_dict(orient="records")


# In[ ]:


data


# In[6]:


db=client["tweet_db"]


# In[7]:


print(db)


# In[8]:


db.tweet.insert_many(data)


# # Step 2: Data Preprocessing and Cleaning

# Load Data from MongoDB

# In[9]:


db = client["tweet_db"]


# In[10]:


print(db.list_collection_names())


# In[11]:


collection = db["tweet"]  # Replace with the actual collection name


# Inspect Data

# In[12]:


data = pd.DataFrame(list(collection.find()))
data.head()  # Display the first few rows


# ### Data Cleaning

# Handle Missing Values

# In[13]:


data.isnull().sum()
data.fillna("Missing", inplace=True)  # Example of filling missing values


# Data Formatting

# In[14]:


data['date'] = pd.to_datetime(data['date'])
data['ids'] = data['ids'].astype(int)


# Remove Unnecessary Columns

# In[15]:


data.drop(['_id','flag'], axis=1, inplace=True)


# #### Data Transformation

# Feature Engineering

# In[16]:


data['day_of_week'] = data['date'].dt.day_name()


# Text Preprocessing

# In[17]:


import re
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

data['text_clean'] = data['text'].apply(preprocess_text)


# Store Cleaned Data

# In[18]:


db.cleaned_tweets.insert_many(data.to_dict('records'))


# ### Step 3: Distributed Data Processing with Apache Spark

# Load Data from MongoDB

# In[46]:


get_ipython().system('pip install pyspark')


# In[51]:


get_ipython().system('pip install findspark')


# In[ ]:


import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SimpleTest").getOrCreate()
df = spark.createDataFrame([                                  (1, "foo"), (2, "bar")], ["id", "label"])
df.show()

spark.stop()


# ## Virtual

# In[54]:


get_ipython().system('pip install virtualenv')


# In[ ]:


from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/mydb.myCollection") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/mydb.myCollection") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()


# In[ ]:


from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("SimpleTest").getOrCreate()

# Create a simple DataFrame
data = [("John Doe", "2021-01-01"), ("Jane Doe", "2021-07-01")]
columns = ["Name", "Date"]
df = spark.createDataFrame(data, columns)

# Show DataFrame
df.show()

# Stop Spark Session
spark.stop()


# In[62]:


from pyspark.sql import SparkSession


# In[ ]:


spark=SparkSession.\
builder.\
appName("sparksql").\
getOrCreate()


# #### Step 4: Sentiment Analysis

# In[21]:


get_ipython().system('pip install TextBlob')


# In[ ]:


from textblob import TextBlob

# Assuming 'text_clean' is the column containing preprocessed tweet text
data['sentiment'] = data['text_clean'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# You can adjust the polarity threshold based on your analysis needs
data['sentiment_label'] = data['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Assuming 'db' is your MongoDB connection
db.cleaned_tweets_sentiment.insert_many(data.to_dict('records'))


# #### Step 5: Time Series Forecasting

# In[7]:


import pymongo
import pandas as pd

# Connect to the local MongoDB instance
client = pymongo.MongoClient("mongodb://localhost:27017")

# Replace 'your_database_name' with the name of your database
db = client['tweet_db']
collection = db['cleaned_tweets_sentiment']

# Fetching data from the collection
data = list(collection.find({}))


# In[9]:


data = pd.DataFrame(data)


# In[10]:


data = data.head(4000)


# In[2]:


get_ipython().system('pip install pandas numpy statsmodels')


# In[18]:


from statsmodels.tsa.api import SimpleExpSmoothing
import numpy as np

# Simple Moving Average Function
def simple_moving_average(series, periods):
    return series.rolling(window=periods).mean().iloc[-1]

# Simple Exponential Smoothing Function
def simple_exponential_smoothing(series, periods):
    try:
        model = SimpleExpSmoothing(series)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        return forecast
    except Exception as e:
        print(f"Error in Simple Exponential Smoothing forecast: {e}")
        return [None] * periods

# Performing forecasting
forecasts = {}
for period_name, period_length in forecast_periods.items():
    sma_result = simple_moving_average(daily_sentiment, period_length)
    ses_result = simple_exponential_smoothing(daily_sentiment, period_length)
    
    forecasts[period_name] = {
        'Simple_Moving_Average': sma_result,
        'Simple_Exponential_Smoothing': ses_result.tolist() if ses_result is not None else [None] * period_length
    }

# Display the forecasts
for period_name, forecast_data in forecasts.items():
    print(f"\nForecast for {period_name}:")
    print(f"Simple Moving Average: {forecast_data['Simple_Moving_Average']}")
    print(f"Simple Exponential Smoothing: {forecast_data['Simple_Exponential_Smoothing']}")


# In[25]:


import matplotlib.pyplot as plt
import pandas as pd

# [Your existing code to compute forecasts]

# Plotting the forecasts
plt.figure(figsize=(12, 6))

# Plotting the original daily sentiment data
plt.plot(daily_sentiment.index, daily_sentiment['sentiment'], label='Actual Sentiment', color='blue')

# Plotting the forecasts
for period_name, period_length in forecast_periods.items():
    # Generating future dates for forecasts
    last_date = daily_sentiment.index[-1]
    future_dates = pd.date_range(start=last_date, periods=period_length + 1, freq='D')[1:]

    # Plotting the SMA forecast
    sma_forecast = [forecasts[period_name]['Simple_Moving_Average']] * period_length
    plt.plot(future_dates, sma_forecast, label=f'SMA Forecast for {period_name}', linestyle='--')

    # Plotting the SES forecast
    ses_forecast = forecasts[period_name]['Simple_Exponential_Smoothing']
    plt.plot(future_dates, ses_forecast, label=f'SES Forecast for {period_name}', linestyle=':')

plt.title('Sentiment Forecast using SMA and SES')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.legend()
plt.show()


# #### LSTM

# In[31]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# Assuming 'data' is your DataFrame with 'sentiment' as one of the columns
# and 'date' as the index
# Normalize the sentiment scores
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['sentiment'].values.reshape(-1, 1))

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

# Create sequences
sequence_length = 5  # Adjust as needed
X, y = create_sequences(data_scaled, sequence_length)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]



# In[32]:


# Reshape input for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)


# In[33]:


# Forecasting function
def forecast(model, data, sequence_length, periods):
    forecast = data[-sequence_length:].tolist()
    for _ in range(periods):
        x = np.array(forecast[-sequence_length:]).reshape(1, sequence_length, 1)
        out = model.predict(x)[0][0]
        forecast.append([out])
    return scaler.inverse_transform(forecast[sequence_length:])

