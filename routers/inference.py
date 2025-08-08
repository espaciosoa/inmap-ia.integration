# Basics
import json
import math
import base64
import io
import logging
from pydantic import BaseModel
from PIL import Image, ImageDraw
from datetime import datetime, timezone
import time
import math



from fastapi import APIRouter,BackgroundTasks
from fastapi.responses import JSONResponse, Response
import uuid


from db import rooms, measurements, jobs


from utils.ImageUtils import bounding_box_2d , numpy_array_to_base64_image
from utils.listComprehensions import serialize_doc, getJustCoords,getValueAndCoords, normalize_dbm
from math import ceil
# I think all of this file is deprecated, but I leave it here for now


import sys
sys.path.append("../backend.ai.signal_cleaner")
# Importando el módulo de IA de la empresa EspacioSOA.sl
from Integration import prepararModelo, ejecutarModelo
from EntrenarAutoencoder import entrenarAutoencoder

logging.basicConfig(filename='background_task.log', level=logging.INFO)


router = APIRouter(
    prefix="/ia",
)


@router.get("/")
def rootIA():
    return "Base endpoint"



@router.get("/testReturnImage")
def test():
    print("TEST")
    image = Image.open("image.png")
    # image.show()  # Optional: display the image

    # Save to bytes
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    return JSONResponse({"base64":base64_img})




def train_model(job_id, params:dict):
    '''
    Función que se ejecuta en segundo plano para entrenar el autoencoder
    '''

    print(f"Training model with job id {job_id}")
    # print(f"Parameters: {params}")
    try:
        # Aquí iría el código para entrenar el modelo
        # * Llamamos a la función que se encarga de ejecutar los pasos estáticos del modelo
        # * (generarDataset + entrenarAutoencoder)
        prepararModelo( params["roomName"], params["imageSize"], params["x_z_val_tuple"] ,  params["epochs"] )
        print("Training finished")
        #jobs.find({"job_id":job_id}).update_one({"$set": {"status": "completed"}})
        result = jobs.update_one(
            {"job_id": job_id}, {"$set": {"status": "completed"}}
        )

        # Verificamos resultado
        if result.modified_count == 1:
            print("Documento actualizado")
        else:
            print("Documento no encontrado o no modificado")
    except (ValueError, TypeError):
        # Aquí finalmente habría que guardar en la base de datos que el entrenamiento ha terminado
        print("Error during training")
        #jobs.find({"job_id":job_id}).update_one({"$set": {"status": "error"}})
        result = jobs.update_one(
            {"job_id": job_id}, {"$set": {"status": "error"}}
        )

        # Verificamos resultado
        if result.modified_count == 1:
            print("Documento actualizado")
        else:
            print("Documento no encontrado o no modificado")

class RoomTrainingParams(BaseModel):
    # roomName: str Not using it because I get it from the URL and the library does not support duplicated names
    signalParameter: str
    epochs: int

@router.post("/runTraining/{roomName}")
def runTrainingSingle( 
    roomName: str ,
    params: RoomTrainingParams, # signalParameter (str) and epochs (int)
    background_tasks: BackgroundTasks, # from fastapi
    response: Response
    ):
    '''
    Ejecuta el entrenamiento de un modelo de IA 
    (quizá tenga sentido que le pueda pasar en en body a con qué parámetro entrenarlo -e.g., dDm-)
    (enviar épocas también, que es un parámetro importante)
    Devuelve un identificador de trabajo al front, ya que es algo que tarda un montón:
    jobId
    room
    status
    '''

    print(f"/runTraining/{roomName} Training model for room {roomName}")
    # * 1. Extraemos el parámetro que quiero utilizar para medir (ej: dbm) y el número de epochs 
    # (Falta parsear parámetros de entrada)
    rq_signalParameter=params.signalParameter
    rq_epochs=params.epochs


    # * 2. Recuperamos la room de la DB y convertimos los Ids a string
    room = rooms.find_one({"name":roomName})    
    room = serialize_doc(room) # Convierto Ids a string

    # * 3. Obtenemos TODAS las medidas de esa room ordenadas por su timestamp
    thisRoomMeasurements = list(
        measurements.find({
                "roomId":room["_id"],
                "fullCellSignalStrength": {"$exists": True, "$ne": None}
        })
    )
    # Si filtramos...
    #allThisRoomMeasurements = list(
    #    measurements.find({
    #        "roomId": room["_id"],
    #        "measurementSession": "e0f064f5-1b1d-4929-b5fa-08bb15d67259"
    #    }).sort("timestamp", 1)
    #)
    # * 3.1 Seleccionamos 200 de manera sistemática para aplicar el muestreo
    #M = len(allThisRoomMeasurements)
    #if M <= 200:
    #    thisRoomMeasurements = allThisRoomMeasurements  # Si ya hay ≤200, las tomamos todas
    #else:
    #    k = math.floor(M / 200)  # Espaciado entre muestras
    #    thisRoomMeasurements = allThisRoomMeasurements[::k][:200]  # [::k] toma cada k-ésima muestra, [:200] asegura ≤200

    print(f"Cantidad de medidas tomadas para {roomName}: {len(thisRoomMeasurements)}")
    # *     3.2 Obtenemos el valor de la medida (dbm)
    # *     3.3 Obtenemos y normalizamos las coordenadas
    # Get just the coordinates 
    thisRoomcleanCoords = [getJustCoords(measurement) for measurement in thisRoomMeasurements]
    thisRoomBbox = bounding_box_2d(thisRoomcleanCoords)
    x_z_val_tuple = [getValueAndCoords(measurement,"dbm") for measurement in thisRoomMeasurements]
    # Normalizamos dbm entre 0 y 1
    x_z_val_tuple = normalize_dbm(x_z_val_tuple)
    
    epocas = rq_epochs      
    metrosX = thisRoomBbox["max"]["x"] - thisRoomBbox["min"]["x"] 
    metrosY = thisRoomBbox["max"]["z"] - thisRoomBbox["min"]["z"]

    l = max(metrosX, metrosY) # As the IA only works for square boxes 

    #pixelsPerMeter =  4
    #pixelsPerMeter =  15 # 12 pixeles por metro
    #imageSize = ceil (l*pixelsPerMeter)
    # Ponemos como tamaño de imagen 512px directamente
    imageSize = 236
    #imageSize = 512


    job_id = str(uuid.uuid4())
    # Get the current timestamp in UTC with microsecond precision
    timestamp = datetime.now(timezone.utc).isoformat(timespec='microseconds')

    # ? Hasta aquí todo bien, llama a trainModel pasándole todos los parámetros que necesita
    # ? el módulo de signal_cleaner
    params = { 
        "roomId": room["_id"],
        "roomName": room["name"],
        "trainedOnParameter": rq_signalParameter,
        "imageSize": imageSize,
        "x_z_val_tuple": x_z_val_tuple,
        "epochs": rq_epochs,
        "others" : ""
    }
    #params = { 
    #    "roomId": room["_id"],
    #    "roomName": "officeJC01_4",
    #    "trainedOnParameter": rq_signalParameter,
    #    "imageSize": imageSize,
    #    "x_z_val_tuple": x_z_val_tuple,
    #    "epochs": rq_epochs,
    #    "others" : ""
    #}


    # * 4. Añadimos la tarea en segundo plano de "train_model"
    background_tasks.add_task(train_model, job_id, params )
    # RUN PREPARE MODEL
    #prepararModelo( room["name"], imageSize, x_z_val_tuple ,  epocas )



    # * 5. Creamos objeto que representa trabajo en curso
    # No sé qué hay mal de sintaxis aquí
    job ={
        "job_id" : job_id,
        "created_t" : timestamp,
        "type": "MODEL TRAINING",
        "status": "Ongoing",
        "parameters" : params
    }
    jobs.insert_one(job)
    ## Guardar este objeto en la base de datos
    # Missing
    job = serialize_doc(job)

    
    return JSONResponse({
        "success": True,
        "data" : job
    }   )


# --- ENTRENAR AUTOENCODER ---

def entrenar_autoencoder_background(jobId: str):
    try:
        # * 1. Buscamos el job que es donde están los datos necesarios para entrenar el autoencoder
        job = jobs.find_one({"job_id": jobId})
        job = serialize_doc(job)
        if not job:
            print(f"Job {jobId} no encontrado")
            return

        # * 2. Obtener parámetros
        nombreRoom = job["parameters"]["roomName"]
        mediciones = job["parameters"]["x_z_val_tuple"]
        imageSize = job["parameters"]["imageSize"]
        #epocas = job["parameters"]["epochs"]
        epocas = 50

        datasetPath = f"./generated/{nombreRoom}"
        modelGenPath = f"./generated/{nombreRoom}/model"
        print(f"Entrenado con {imageSize} imageSize")

        # * 3. Entrenamos el modelo (esto es bloqueante)
        entrenarAutoencoder(epocas, imageSize, datasetPath, modelGenPath, nombreRoom)

        # * 4. Actualizamos estado
        jobs.update_one(
            {"job_id": jobId}, 
            {"$set": {"status": "completed"}}
        )
        print(f"Entrenamiento completado para Job {jobId}")

    except Exception as e:
        print(f"Error en el entrenamiento del autoencoder: {e}")

@router.post("/trainAutoencoder/{jobId}")
async def trainAutoencoder(
    jobId: str,
    background_tasks: BackgroundTasks
):
    """
    Endpoint que inicia el entrenamiento en segundo plano.
    """
    # Verificamos si el job existe
    job = jobs.find_one({"job_id": jobId})
    if not job:
        return JSONResponse(
            {"success": False, "message": "Job not found"},
            status_code=404
        )

    # Añadir tarea en segundo plano
    background_tasks.add_task(entrenar_autoencoder_background, jobId)

    return JSONResponse({
        "success": True,
        "message": "Entrenamiento de autoencoder iniciado en segundo plano",
        "job_id": jobId
    })

# --- ENTRENAR AUTOENCODER ---

@router.get("/jobStatus/{jobId}")
def runTrainingSingle(jobId, response: Response):
    '''
        Yo diría que aquí lo que hago es de vez en cuando preguntar desde el front cómo va y eso lo que hace es comprobar si ya se ha terminado el entrenamiento porque se han generado todos los archivos. No?
        Es importante ver que se pueda ejecutar la generación del modelo de forma asíncrona.
    '''

    ## 1) Buscar en base de datos o bien directamente en el sistema de ficheros
    ## ** Si hago lo de mirar en el sistema de ficheros, directamente actualizar aquí también la base de datos con el nuevo estado
    job = jobs.find_one({"_id":jobId})
    job = serialize_doc(job) # Convierto Ids a string
    if not job:
            return JSONResponse({"success":False, "message":"Job not found"})
    ## 3) devlolver el estado actual 


    return JSONResponse({
        "success": True,
        "data" : job
    }   )




@router.post("/runModel/{jobId}")
def runModel(jobId, response: Response):
    """
    Endpoint que ejecuta un modelo si previamente se ha completado el entrenamiento de esa room
    """
    #print(jobId)
    job = jobs.find_one({"job_id":jobId})
    job = serialize_doc(job) # Convierto Ids a string
    if not job:
            return JSONResponse({"success":False, "message":"Job not found"})

    # * 2. Para ejecutar el modelo necesitamos pasarle el nombre de la room, las mediciones y el tamaño de la imagen
    nombreRoom = job["parameters"]["roomName"]
    #nombreRoom = "officeJC01_4"
    mediciones = job["parameters"]["x_z_val_tuple"]
    imageSize = job["parameters"]["imageSize"]

    # Convertimos cada lista interna a tupla
    mediciones_transformadas =  [tuple(item) for item in mediciones]
    

    print("Ejecutando el modelo para la room: ", nombreRoom)
    start = time.time()
    # Ejecutamos el modelo
    #resultingHeatmap = ejecutarModelo(nombreRoom, mediciones, imageSize)
    resultingHeatmap = ejecutarModelo(nombreRoom, mediciones_transformadas, imageSize)
    end = time.time()

    print(f"Tiempo de ejecucion del modelo: {end - start:.4f} segundos")

    print(resultingHeatmap)

    # * 3. Se genera la imagen y la abrimos
    image = Image.open("image.png")
    # image.show()  # Optional: display the image


    # base64Output= numpy_array_to_base64_image(image)

    # Save to bytes
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
  
    
    # buf = io.BytesIO()
    # resultingHeatmap.save(buf, format="PNG")
    # img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")


    return JSONResponse({
        # "imgAsBase64":base64Output
        "imgAsBase642":base64_img,
        "result":resultingHeatmap.tolist()
    })












# Runs training for all models
@router.get("/runTraining")
def runTraining():
    '''
    Ejecuta el entrenamiento de un modelo de IA para cada una de las "rooms" de la aplicación
    '''
    resultsAsList = list(rooms.find({"name": "testOfi01"}))

    allRooms = [serialize_doc(doc) for doc in resultsAsList]


    for r in allRooms:
        print(f"Running training for room {r["name"]}")
        thisRoomMeasurements = list(measurements.find({"roomId":r["_id"]}))

        thisRoomcleanCoords = [getJustCoords(measurement) for measurement in thisRoomMeasurements]

        thisRoomBbox = bounding_box_2d(thisRoomcleanCoords)

        x_z_val_tuple = [getValueAndCoords(measurement,"dbm") for measurement in thisRoomMeasurements]

        epocas = 10

            
        metrosX = thisRoomBbox["max"]["x"] - thisRoomBbox["min"]["x"] 
        metrosY = thisRoomBbox["max"]["z"] - thisRoomBbox["min"]["z"]

        l = max(metrosX, metrosY) # As the IA only works for square boxes 

        pixelsPerMeter =  4
        imageSize = ceil (l*pixelsPerMeter)

        # RUN PREPARE MODEL

        prepararModelo( r["name"],  imageSize, x_z_val_tuple ,  epocas
                        )

    return {"State": "runningTraining"}















@router.get("/inference/{roomName}")
def runInference(roomName, response: Response):
    """
    Endpoint que ejecuta una inferencia completa sin guardar jobs
    """

    room = rooms.find_one({"name":roomName})    
    if(not room):
        return JSONResponse({"result":"Not found"})
    thisRoomMeasurements = list(measurements.find({"roomId":room["_id"]}))

    #Get bounding box of coordinates
    thisRoomcleanCoords = [getJustCoords(measurement) for measurement in thisRoomMeasurements]
    thisRoomBbox = bounding_box_2d(thisRoomcleanCoords)

    x_z_val_tuple = [getValueAndCoords(measurement,"dbm") for measurement in thisRoomMeasurements]

    x_z_val_tuple = normalize_dbm(x_z_val_tuple)

    metrosX = thisRoomBbox["max"]["x"] - thisRoomBbox["min"]["x"] 
    metrosY = thisRoomBbox["max"]["z"] - thisRoomBbox["min"]["z"]

    l = max(metrosX, metrosY) # As the IA only works for square boxes 

    pixelsPerMeter =  4
    imageSize = ceil (l*pixelsPerMeter)

    print("Performing inference for room: ", roomName)
    start = time.time()
    # Ejecutar modelo
    resultingHeatmap = ejecutarModelo(roomName,x_z_val_tuple,imageSize)
    end = time.time()

    print(f"Inference time: {end - start:.4f} seconds")

    print(resultingHeatmap)

    image = Image.open("image.png")
    # image.show()  # Optional: display the image


    # base64Output= numpy_array_to_base64_image(image)

    # Save to bytes
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
  
    
    # buf = io.BytesIO()
    # resultingHeatmap.save(buf, format="PNG")
    # img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")


    return JSONResponse({
        # "imgAsBase64":base64Output
        "imgAsBase642":base64_img,
        "result":resultingHeatmap.tolist()
    })
