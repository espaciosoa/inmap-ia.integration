from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from utils.listComprehensions import serialize_doc, getJustCoords,getValueAndCoords, normalize_dbm
from utils.jobUtils import crear_job

from db import rooms, measurements, jobs
from bson import ObjectId

import sys
sys.path.append("../backend.ai.signal_estimator")
# Importamos funciones del módulo
from PrepararDataset import preparar_dataset, generar_dataset # Antiguo
from generar_dataset import calcular_limit, generar_matriz, guardar_matriz_como_npy
from entrenar_autoencoder import cargar_datos, train_test_split, entrenar_autoencoder
from estimar_mapa import estimar_mapa

import numpy as np

import os

router = APIRouter(
    prefix="/ia_estimator",
)


@router.get("/")
def rootIA():
 return JSONResponse({
                "success":True,
                "data" : f"El servidor de estimaciones con IA está desplegado"
            }   )

# Clase que define el modelo de datos de lo que recogeremos en el cuerpo de esta solicitud
class RoomTrainingParams(BaseModel):
    limit: int # Límite para el grid, se genera de -limit a +limit en ambos ejes

@router.post("/runTraining/{roomName}")
def runTrainingSingle( 
    roomName: str,
    params: RoomTrainingParams # width (int) and height (int)
):
    try:

        print("Preparando dataset...")

        rq_limit = params.limit

        # * 1. Recuperamos la room de la DB y convertimos los Ids a string
        room = rooms.find_one({"name":roomName})    
        room = serialize_doc(room) # Convierto Ids a string

        # * 2. Obtenemos TODAS las medidas de esa room ordenadas por su timestamp
        room_measurements = list(
            measurements.find({
                    "roomId":room["_id"],
                    "fullCellSignalStrength": {"$exists": True, "$ne": None}
            })
        )

        if not room_measurements:
            return JSONResponse({
                "success": False,
                "data" : f"La room {roomName} no tiene ninguna medida asociada"
            }   )
        
        # * 3. El primer paso del módulo para generar el dataset necesita las muestras
        # *    como una lista de tuplas con la coordenada x, coordenada z y dbm
        formatted_measurements = [
            (
                measurement['position']['x'],
                measurement['position']['z'],
                measurement['fullCellSignalStrength']['dbm']
            )
            for measurement in room_measurements
        ]

        prueba = [
            (1.2, -0.5, -65),
            (3.4, 2.1, -78),
            (-4.3, 6.2, -89),
        ]
        
        #result = preparar_dataset(prueba, rq_limit)
        X_train, y_train, X_grid, known_coords, mapping_info, X_predecir = generar_dataset(prueba, rq_limit)

        print("Resultado de generar_dataset:")
        print("Coordenadas normalizadas de puntos conocidos (X_train):", X_train)
        print("Coordenadas normalizadas del grid completo (X_grid):", X_grid)
        print("X_predecir:")
        print(X_predecir)

        return JSONResponse({
            "success": True,
            "message" : f"Entrenamiento finalizado con éxito para la room {roomName}",
            "data": formatted_measurements
        }   )
    except Exception as e:
        print(f"Error during training: {e}")
        return JSONResponse({
            "success": False,
            "message" : f"Ejecución de entrenamiento fallado para la room {roomName}",
            "error": e            
        }   )

# --- NUEVO ---
# --- GENERACIÓN DATASET INDIVIDUAL --- #
def run_individual_training_task(job_id: str, room_name: str):
    try:
        print(f"==> Ejecutando tarea individual para room {room_name} y job {job_id}")

        # Actualizar estado del job a "in_progress"
        jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {
                "individualDataset_status": "Ongoing"
            }}
        )

        # * 1. Recuperamos la room de la DB y convertimos los Ids a string
        room = rooms.find_one({"name": room_name})
        if not room:
            raise Exception(f"No se encontró la room '{room_name}'")

        room = serialize_doc(room)

        # * 2. Obtenemos TODAS las medidas de esa room (ordenadas por su timestamp)
        room_measurements = list(
            measurements.find({
                "roomId": room["_id"],
                "fullCellSignalStrength.dbm": {"$exists": True, "$ne": None}
            })
        )

        if not room_measurements:
            raise Exception(f"La room '{room_name}' no tiene ninguna medición con señal.")

        # * 3. El primer paso del módulo para generar el dataset necesita las muestras
        # *    como una lista de tuplas con la coordenada x, coordenada z y dbm
        formatted_measurements = [
            (
                m["position"]["x"],
                m["position"]["z"],
                m["fullCellSignalStrength"]["dbm"]
            ) for m in room_measurements
        ]

        # * 4. Recuperaramos limit del job, ya que deberá ser el mismo para el entrenamiento individual que para el global
        job = jobs.find_one({"_id": ObjectId(job_id)})
        if not job or "parameters" not in job or "limit" not in job["parameters"]:
            raise Exception("No se encontró el límite (limit) dentro del job")

        limit = job["parameters"]["limit"]

        # * 5. Generamos la matriz de la sala concreta y la guardamos
        dataset_path = f"./generated/{room_name}/dataset"
        matriz = generar_matriz(formatted_measurements, limit=limit, step=0.5)
        guardar_matriz_como_npy(matriz, room_name, carpeta_destino=dataset_path)

        # Actualizamos el job con el roomId, roomName y estado final
        jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {
                "individualDataset_status": "Completed",
                "parameters.roomId": room["_id"],
                "parameters.roomName": room_name
            }}
        )

        print(f"==> Dataset individual generado correctamente para {room_name}")

    except Exception as e:
        print(f"Error durante generación del dataset individual: {e}")
        jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {
                "individualDataset_status": "Failed"
            }}
        )

# Clase que define el modelo de datos de lo que recogeremos en el cuerpo de esta solicitud
class RoomTrainingParams(BaseModel):
    job_id: str # ID del job asociado al entrenamiento
    room_name: str # Nombre de la sala que se quiere estimar

@router.post("/runIndividualTraining")
def run_training_single(
    background_tasks: BackgroundTasks,
    params: RoomTrainingParams
):
    try:
        job_id = params.job_id
        room_name = params.room_name

        if not job_id or not room_name:
            return JSONResponse({
                "success": False,
                "message": "Required parameters are missing: job_id or room_name"
            })

        # Añadir la tarea en segundo plano
        background_tasks.add_task(run_individual_training_task, job_id, room_name)

        return JSONResponse({
            "success": True,
            "message": f"Individual training launched for {room_name}"
        })

    except Exception as e:
        print(f"Error al iniciar tarea individual: {e}")
        return JSONResponse({
            "success": False,
            "message": "Failure to start individual task",
            "error": str(e)
        })

# --- GENERACIÓN DATASET GLOBAL --- #
def run_global_training_task(job_id):
    try:
        print("Iniciando generación de dataset global...")

        all_rooms = list(rooms.find({}))
        if not all_rooms:
            jobs.update_one({"_id": job_id}, {"$set": {"globalDataset_status": "Failed"}})
            return

        all_measurements = []
        room_blocks = []
        bloques_por_sala = 6

        # Paso 1: Recolectamoss todas las mediciones válidas
        for room in all_rooms:
            print(f"Procesando sala {room['name']}...")
            room_measurements = list(measurements.find({
                "roomId": room["_id"],
                "fullCellSignalStrength.dbm": {"$exists": True, "$ne": None}
            }))

            if len(room_measurements) < bloques_por_sala * 30:
                print(f"Saltando sala {room['name']} por falta de datos suficientes.")
                continue

            formatted = [
                (
                    m['position']['x'],
                    m['position']['z'],
                    m['fullCellSignalStrength']['dbm']
                )
                for m in room_measurements
            ][:bloques_por_sala * 30]

            all_measurements.extend(formatted)
            room_blocks.append(formatted)

        if not all_measurements:
            jobs.update_one({"_id": job_id}, {"$set": {"globalDataset_status": "Failed"}})
            return

        # Paso 2: Calculamos un limit_global a partir de todas las mediciones (este será el que se utilice en el entrenamiento de la sala individual)
        limit_global = calcular_limit(all_measurements)
        print(f"Usando limit_global = {limit_global}")

        # Paso 3: Generaramos matrices usando limit_global
        matrices_por_sala = []

        for formatted in room_blocks:
            for i in range(bloques_por_sala):
                bloque = formatted[i*30:(i+1)*30]
                matriz = generar_matriz(bloque, limit=limit_global, step=0.5)
                if matriz.shape == (81, 81):
                    matrices_por_sala.append(matriz)

        if not matrices_por_sala:
            jobs.update_one({"_id": job_id}, {"$set": {"globalDataset_status": "Failed"}})
            return

        # Paso 4: guardamos el array final
        array_final = np.stack(matrices_por_sala, axis=0)
        array_final = array_final.reshape(-1, 81, 81, 1)
        os.makedirs("./generated/global/dataset", exist_ok=True)
        np.save("./generated/global/dataset/global_dataset.npy", array_final)

        # Actualizamos job con resultado
        jobs.update_one({"_id": job_id}, {
            "$set": {
                "globalDataset_status": "Completed",
                "parameters.limit": limit_global
            }
        })

    except Exception as e:
        print(f"Error durante generación del dataset global: {e}")
        jobs.update_one({"_id": job_id}, {
            "$set": {"globalDataset_status": "Failed", "error": str(e)}
        })

@router.post("/runGlobalTraining")
def run_training_global(background_tasks: BackgroundTasks):
    try:
        # --- Paso 1: Creamos el job en MongoDB ---
        parametros = {
            "roomId": None,
            "roomName": "GLOBAL",
            "trainedOnParameter": "dbm",
            "limit": None,
            "others": ""
        }

        job = crear_job("MODEL TRAINING", parametros)
        job_id = jobs.insert_one(job).inserted_id  # Guardamos el job

        # --- Paso 2: Agregamos tarea en segundo plano ---
        background_tasks.add_task(run_global_training_task, job_id)

        return JSONResponse({
            "success": True,
            "message": "Global dataset generation task launched in the background.",
            "job_id": str(job_id)
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": "Error when starting the task.",
            "error": str(e)
        })

# --- GENERACIÓN AUTOENCODER --- #
# TODO: lanzar automáticamente cada cierto tiempo o mediante trigger
class AutoencoderTrainingParams(BaseModel):
    job_id: str
    ruta_dataset: str = "generated/global/dataset/global_dataset.npy"  # Ruta por defecto (siempre la misma por ahora)
def entrenar_autoencoder_task(job_id: str, ruta_dataset: str):
    try:
        print("Cargando datos...")
        datos = cargar_datos(ruta_dataset)
        print("Forma original del dataset:", datos.shape)

        print("Dividiendo en entrenamiento y validación...")
        x_train, x_val = train_test_split(datos, test_size=0.2, random_state=42)

        print("Entrenando autoencoder...")
        modelo = entrenar_autoencoder(x_train, x_val, input_shape=datos.shape[1:])

        print("Entrenamiento completado correctamente y guardado como modelo_autoencoder.h5")
    
        # Actualizamos el estado del job asociado
        jobs.update_one({"_id": job_id}, {"$set": {"autoencoder_status": "completed"}})
    except Exception as e:
        print(f"Error durante entrenamiento del autoencoder: {e}")
        jobs.update_one({"_id": job_id}, {
            "$set": {"autoencoder_status": "Failed", "error": str(e)}
        })
        
@router.post("/entrenarAutoencoder")
def lanzar_entrenamiento_autoencoder(params: AutoencoderTrainingParams, background_tasks: BackgroundTasks):
    """
    Lanza el entrenamiento del autoencoder en segundo plano y actualiza el estado del job asociado.
    """
    background_tasks.add_task(entrenar_autoencoder_task, params.job_id, params.ruta_dataset)
    return JSONResponse(content={"mensaje": "Training started in the background"})

# --- ESTIMACIÓN MAPA FINAL --- #
class MapEstimationParams(BaseModel):
    room_name: str # Nombre sala
    # mapa_path a partir del room_name
    # model_path siempre el mismo
    # mapa_salida_path a partir del room_name
@router.post("/estimarMapa")
def estimar_mapa_endpoint(params: MapEstimationParams):
    try:
        room_name = params.room_name
        mapa_path = f"generated/{room_name}/dataset/{room_name}.npy" # Ruta por defecto del dataset individual de cada sala
        model_path = "modelo_autoencoder.h5"  # Ruta por defecto del autoencoder (ahora mismo para simplicar siempre el mismo)
        mapa_salida_path = f"generated/{room_name}/{room_name}_estimado.npy" # Ruta por defecto de salida
       # Estimamos el mapa
        resultado = estimar_mapa(
            mapa_path=mapa_path,
            model_path=model_path,
            mapa_salida_path=mapa_salida_path,
        )
        return {"message": "✅ Map estimated correctly", **resultado}
    except Exception as e:
        print(f"Error durante la estimación del mapa: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mapa-json/{room_name}")
def get_mapa(room_name: str):
    try:
        file_path = f"generated/{room_name}/{room_name}_estimado.npy"
        array = np.load(file_path)
        return {"map": array.tolist()}
    except FileNotFoundError:
        return {"error": f"No se encontró el mapa estimado para '{room_name}'"}

def extraer_dbm_de_mapa(mapa_path: str, limit=10.0, step=0.5):
    mapa = np.load(mapa_path)

    puntos = []
    alto, ancho = mapa.shape

    # x, z -> originales
    # i, j -> transformados
    for j in range(alto):      # j → fila → z
        for i in range(ancho): # i → columna → x
            dbm = float(mapa[j, i])
            # Revertimos la transformación de coordenadas
            x = i * step - limit
            z = j * step - limit
            puntos.append({"x": x, "z": z, "dbm": dbm})

    return puntos

class MapExtractionParams(BaseModel):
    room_name: str # Nombre sala
    job_id: str # ID del job asociado para extraer el limit
@router.post("/mapa/extraer")
def extraer_puntos_dbm(params: MapExtractionParams):
    try:
        mapa_path = f"generated/{params.room_name}/{params.room_name}_estimado.npy"
        if not os.path.exists(mapa_path):
            return []

        # * Recuperamos el job para obtener el limit utilizado
        job = jobs.find_one({"_id": ObjectId(params.job_id)})
        if not job or "parameters" not in job or "limit" not in job["parameters"]:
            raise Exception("The limit was not found within the job.")

        limit = job["parameters"]["limit"]

        return extraer_dbm_de_mapa(mapa_path, limit)
    except Exception as e:
        print(f"Error durante la extracción del mapa: {e}")
        raise HTTPException(status_code=500, detail=str(e))



from bson import ObjectId

@router.get("/jobStatus/{jobId}")
def runTrainingSingle(jobId: str, response: Response):
    try:
        job = jobs.find_one({"_id": ObjectId(jobId)})
    except Exception:
        return JSONResponse({"success": False, "message": "Invalid job ID format"})

    if not job:
        return JSONResponse({"success": False, "message": "Job not found"})

    job = serialize_doc(job)
    
    return JSONResponse({
        "success": True,
        "data": job
    })

