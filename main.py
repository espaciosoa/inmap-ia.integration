# This file is the starting point of the app
from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

import numpy as np
# from db import db, measurements,rooms
# The other project


from routers import rooms, measurements, signal_estimator


load_dotenv()
port = os.getenv("PORT", 8001)

app = FastAPI()

app.include_router(rooms.router)
app.include_router(measurements.router)
#app.include_router(inference.router)
app.include_router(signal_estimator.router)


origins = [
    "https://measurements.espaciosoa.com",
    "https://localhost",
    "https://localhost:8443",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow any domain
    allow_credentials=False,  # must be False with "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return JSONResponse({"message": "Hello, world!"})



# @app.get("/boundingBox/{name}")
# def getBoundingBoxOfRoom(name:str):

    
#     theRoom = rooms.find_one({"name": name})
#     # print(resultsAsList)
#     # cleanResults = [clean_document(doc) for doc in resultsAsList]

#     associatedMeasurements = list(measurements.find({"roomId":theRoom["_id"]}))
#     cleanResults = [clean_document(doc) for doc in associatedMeasurements]


    
#     cleanCoords = [getJustCoords(measurement) for measurement in cleanResults]
    
#     bbox = bounding_box_2d(cleanCoords)
#     print (bbox)

#     return JSONResponse({"theRoom":name, "bbox": bbox })














#Todo : 
# 1)expose an endpoint that provides access to the AI generated heatmap for a given room (if exists)
# 2) Access the database and monitor how many different rooms exist, then train a different model for each of them.



