from pymongo import MongoClient

# Connect to the local server
client = MongoClient("mongodb://localhost:27017/")
db =  client ["measurements_espaciosoa"]

rooms = db["rooms"]
measurements = db["roommeasurements"]
sessions = db["measurementsessions"]
jobs = db["jobs"]