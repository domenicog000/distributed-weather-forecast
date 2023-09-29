from pymongo import MongoClient
import pandas as pd

client = MongoClient("localhost", 27017)
db_name = "weather_data"
database = client[db_name]
collection = database["measurements"]
collection = database.create_collection("measurements")
data = pd.read_csv('data.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.sort_values(by='Timestamp', inplace=True)
data_json = data.to_dict(orient='records')
collection.insert_many(data_json)

    