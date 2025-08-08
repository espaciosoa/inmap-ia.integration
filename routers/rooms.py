from fastapi import APIRouter
from fastapi.responses import JSONResponse
from db import rooms
from utils.listComprehensions import serialize_doc


router = APIRouter(
    prefix="/rooms",
    )


@router.get("/")
def rgetRooms():
    roomsCollection = rooms
    resultsAsList = list(roomsCollection.find({}))
    for r in resultsAsList:
        print(f"hehe {r}")
    cleanResults = [serialize_doc(doc) for doc in resultsAsList]
    return JSONResponse(cleanResults)
