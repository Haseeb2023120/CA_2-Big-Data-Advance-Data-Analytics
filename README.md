"""

!python -V

!pip install "pymongo[snappy,gssapi,srv,tls]"==3.9

import pandas as pd
import json
import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017')
display(client)

client.list_database_names()

Load the dataset

file_path = 'ProjectTweets.csv'
df = pd.read_csv(file_path, sep=',', names=['ids', 'date', 'flag', 'user', 'text'], skiprows=1)
# Skip the first row if it's a header row
display(df)

data = df.to_dict(orient="records")

# Step 2: Data Preprocessing and Cleaning

Load Data from MongoDB

db = client["tweet_db"]
display(db)

db.tweet.insert_many(data)

collection = db["tweet"]  # Replace with the actual collection name

Inspect Data

data = pd.DataFrame(list(collection.find()))
data.head()  # Display the first few rows

### Data Cleaning

Handle Missing Values

data.isnull().sum()
data.fillna("Missing", inplace=True)  # Example of filling missing values

Data Formatting

data['date'] = pd.to_datetime(data['date'])
data['ids'] = data['ids'].astype(int)

Remove Unnecessary Columns

data.drop(['_id','flag'], axis=1, inplace=True)

#### Data Transformation

Feature Engineering

data['day_of_week'] = data['date'].dt.day_name()

Text Preprocessing

import re
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

data['text_clean'] = data['text'].apply(preprocess_text)

Store Cleaned Data

db.cleaned_tweets.insert_many(data.to_dict('records'))

### Step 3: Distributed Data Processing with Apache Spark

Load Data from MongoDB

!pip install pyspark findspark

import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SimpleTest").getOrCreate()
df = spark.createDataFrame([(1, "foo"), (2, "bar")], ["id", "label"])
df.show()

spark.stop()

## Virtual

!pip install virtualenv

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/mydb.myCollection") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/mydb.myCollection") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()

Initialize Spark Session

spark = SparkSession.builder.appName("SimpleTest").getOrCreate()

Create a simple DataFrame

data = [("John Doe", "2021-01-01"), ("Jane Doe", "2021-07-01")]
columns = ["Name", "Date"]
df = spark.createDataFrame(data, columns)

Show DataFrame

df.show()

Stop Spark Session

spark.stop()

from pyspark.sql import SparkSession

# spark = SparkSession.builder.appName("sparksql").getOrCreate()

spark = SparkSession.builder \
    .appName("sparksql") \
    .getOrCreate()

display(spark)

#### Step 4: Sentiment Analysis

!pip install TextBlob

from textblob import TextBlob

Assuming 'text_clean' is the column containing preprocessed tweet text



# data['sentiment'] = data['text_clean'].apply(lambda x: TextBlob(int(x)).sentiment.polarity)

# data['sentiment'] = data['text_clean'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

def calculate_sentiment(x):
    return TextBlob(str(x)).sentiment.

# Assuming 'data' is a DataFrame with a 'text_clean' column
data['sentiment'] = data['text_clean'].apply(calculate_sentiment)


You can adjust the polarity threshold based on your analysis needs

data['sentiment_label'] = data['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

Assuming 'db' is your MongoDB connection

db.cleaned_tweets_sentiment.insert_many(data.to_dict('records'))

#### Step 5: Time Series Forecasting

import pymongo
import pandas as pd

Connect to the local MongoDB instance

client = pymongo.MongoClient("mongodb://localhost:27017")

Replace 'your_database_name' with the name of your database

db = client['tweet_db']
collection = db['cleaned_tweets_sentiment']

Fetching data from the collection

data = list(collection.find({}))

data = pd.DataFrame(data)

data = data.head(4000)

get_ipython().system('pip install pandas numpy statsmodels')

from statsmodels.tsa.api import SimpleExpSmoothing
import numpy as np

Simple Moving Average Function

def simple_moving_average(series, periods):
    return series.rolling(window=periods).mean().iloc[-1]

Simple Exponential Smoothing Function

def simple_exponential_smoothing(series, periods):
    try:
        model = SimpleExpSmoothing(series)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        return forecast
    except Exception as e:
        print(f"Error in Simple Exponential Smoothing forecast: {e}")
        return [None] * periods

Performing forecasting

forecasts = {}
for period_name, period_length in forecast_periods.items():
    sma_result = simple_moving_average(daily_sentiment, period_length)
    ses_result = simple_exponential_smoothing(daily_sentiment, period_length)

    forecasts[period_name] = {
        'Simple_Moving_Average': sma_result,
        'Simple_Exponential_Smoothing': ses_result.tolist() if ses_result is not None else [None] * period_length
    }

Display the forecasts

for period_name, forecast_data in forecasts.items():
    print(f"\nForecast for {period_name}:")
    print(f"Simple Moving Average: {forecast_data['Simple_Moving_Average']}")
    print(f"Simple Exponential Smoothing: {forecast_data['Simple_Exponential_Smoothing']}")

import matplotlib.pyplot as plt
import pandas as pd

[Your existing code to compute forecasts]

Plotting the forecasts

plt.figure(figsize=(12, 6))

Plotting the original daily sentiment data

plt.plot(daily_sentiment.index, daily_sentiment['sentiment'], label='Actual Sentiment', color='blue')

Plotting the forecasts

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

#### LSTM

In[31]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

Assuming 'data' is your DataFrame with 'sentiment' as one of the columns<br>
and 'date' as the index<br>
Normalize the sentiment scores

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['sentiment'].values.reshape(-1, 1))

Function to create sequences

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

Create sequences

sequence_length = 5  # Adjust as needed
X, y = create_sequences(data_scaled, sequence_length)

Split data into training and testing sets

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

In[32]:

Reshape input for LSTM

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

Build LSTM model

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

Train the model

model.fit(X_train, y_train, epochs=50, batch_size=32)

In[33]:

Forecasting function

def forecast(model, data, sequence_length, periods):
    forecast = data[-sequence_length:].tolist()
    for _ in range(periods):
        x = np.array(forecast[-sequence_length:]).reshape(1, sequence_length, 1)
        out = model.predict(x)[0][0]
        forecast.append([out])
    return scaler.inverse_transform(forecast[sequence_length:])

Define periods for 1 week, 1 month, and 3 months<br>
Assuming daily data: 7 days, 30 days, and 90 days

periods = {'1_week': 7, '1_month': 30, '3_months': 90}

Perform forecasting

forecasts = {}
for period_name, period_length in periods.items():
    forecasts[period_name] = forecast(model, data_scaled, sequence_length, period_length)

Display the forecasts

for period_name, forecast_data in forecasts.items():
    print(f"\nForecast for {period_name}:")
    print(forecast_data)

In[35]:

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

[Previous code for data preparation, model training, and forecasting]

Define the starting point for forecasts

last_date = data.index[-1]

Function to plot each forecast period

def plot_forecast(period_name, forecast_data):
    future_dates = pd.date_range(start=last_date, periods=len(forecast_data) + 1, freq='D')[1:]
    plt.figure(figsize=(10, 4))
    plt.plot(data.index, data['sentiment'], label='Actual Sentiment', color='blue')
    plt.plot(future_dates, forecast_data.flatten(), label=f'Forecast {period_name}', linestyle='--')
    plt.title(f'Sentiment Forecast for {period_name} using LSTM')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

Plot forecasts for 1 week, 1 month, 3 months

plot_forecast('1_week', forecasts['1_week'])
plot_forecast('1_month', forecasts['1_month'])
plot_forecast('3_months', forecasts['3_months'])

In[57]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

Step 1: Prepare and Scale the Data

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['sentiment'].values.reshape(-1, 1))

Function to Create Sequences

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

Step 2: Create Sequences

sequence_length = 5  # Adjust as needed
X, y = create_sequences(data_scaled, sequence_length)

Step 3: Split Data into Training and Test Sets

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

Reshape Input for LSTM

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

Step 4: Build and Train LSTM Model

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

Forecasting Function

def forecast(model, data, sequence_length, periods):
    forecast = data[-sequence_length:].tolist()
    for _ in range(periods):
        x = np.array(forecast[-sequence_length:]).reshape(1, sequence_length, 1)
        out = model.predict(x)[0][0]
        forecast.append([out])
    return scaler.inverse_transform(forecast[sequence_length:])

Step 5: Generate Forecasts

periods = {'1_week': 7, '1_month': 30, '3_months': 90}
forecasts = {}
for period_name, period_length in periods.items():
    forecasts[period_name] = forecast(model, data_scaled, sequence_length, period_length).flatten()

Step 6: Print Numeric Forecast Values

for period_name, forecast_data in forecasts.items():
    print(f"\nForecast for {period_name}:")
    print(forecast_data)

In[51]:

print("Forecasts dictionary:", forecasts)

#### Step 6 : building a dynamic and interactive dashboard

In[42]:

get_ipython().system('pip install dash dash-bootstrap-components plotly')

In[63]:

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

Initialize the Dash app

app = dash.Dash()

Define the layout of the app

app.layout = html.Div([
    dcc.Graph(id='time-series-chart'),
    dcc.Dropdown(
        id='forecast-period-dropdown',
        options=[
            {'label': '1 Week', 'value': '1_week'},
            {'label': '1 Month', 'value': '1_month'},
            {'label': '3 Months', 'value': '3_months'}
        ],
        value='1_week'
    )
])

Assuming 'data' is your DataFrame with 'sentiment' as one of the columns<br>
and 'date' as the index. Replace 'forecasts' with your actual forecast data.

forecasts = {
    '1_week': [0.03044248, 0.02936256, 0.02980471, 0.02974749, 0.02946949, 0.02944112, 0.02944124],
    '1_month': [0.03044248, 0.02936256, 0.02980471, 0.02974749, 0.02946949, 0.02944112, 0.02944124, 0.02944124, 0.02944088, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076],
    '3_months': [0.03044248, 0.02936256, 0.02980471, 0.02974749, 0.02946949, 0.02944112, 0.02944124, 0.02944124, 0.02944088, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076, 0.02944076]
}

Callback for updating the graph based on dropdown selection

@app.callback(
    Output('time-series-chart', 'figure'),
    [Input('forecast-period-dropdown', 'value')]
)
def update_graph(selected_period):
    # Get the selected forecast data
    selected_forecast = forecasts[selected_period]
    future_dates = pd.date_range(start=data.index[-1], periods=len(selected_forecast) + 1, freq='D')[1:]

    # Create a Plotly graph
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data.index, y=data['sentiment'], mode='lines', name='Actual Sentiment'))
    figure.add_trace(go.Scatter(x=future_dates, y=selected_forecast, mode='lines', name=f'Forecast {selected_period}', line=dict(color='red')))
    figure.update_layout(title=f'Sentiment Forecast - {selected_period}', xaxis_title='Date', yaxis_title='Sentiment Score')
    return figure

Run the app

if __name__ == '__main__':
    app.run_server(debug=True)

#### Step 7 : Implementation and Integration

import pymongo
import time

MongoDB connection settings

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["tweet_db"]
mongo_collection = mongo_db["cleaned_tweets_sentiment"]

Function to measure MongoDB query performance

def measure_mongodb_performance():
    start_time = time.time()
    result = mongo_collection.find({"sentiment_label": "Positive"})
    for _ in result:
        pass
    end_time = time.time()
    return end_time - start_time

Perform performance measurements

mongo_execution_time = measure_mongodb_performance()

print("MongoDB Query Execution Time:", mongo_execution_time)

Close connections

mongo_client.close()

"""
