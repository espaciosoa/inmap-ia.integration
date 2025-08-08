from fastapi import APIRouter
from fastapi.responses import JSONResponse
from db import measurements
from utils.listComprehensions import serialize_doc

router = APIRouter(
    prefix="/measurements",
    )

@router.get("/")
def getMeasurements():
    resultsAsList = list(measurements.find({}))
    print("Accediente a /measurements/")
    for r in resultsAsList:
        print(f"hehe {r}")
    cleanResults = [serialize_doc(doc) for doc in resultsAsList]
    return JSONResponse(cleanResults)

@router.get("?roomName={roomName}")
def getMeasurements(roomName:str):
    resultsAsList = list(measurements.find({"roomName":roomName}))
    for r in resultsAsList:
        print(f"room {r}")
    cleanResults = [serialize_doc(doc) for doc in resultsAsList]
    return JSONResponse(cleanResults)
