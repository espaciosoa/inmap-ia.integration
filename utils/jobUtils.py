from datetime import datetime, timezone
from bson import ObjectId

def crear_job(tipo, parametros):
    return {
        "_id": ObjectId(),  # MongoDB ID
        "created_at": datetime.now(timezone.utc).isoformat(timespec='microseconds'), # Timestamp en UTF con precisión en microsegundos
        "type": tipo,
        "parameters": parametros,
        "individualDataset_status": None,
        "globalDataset_status": "Ongoing", # Ya que es el primer paso del entrenamiento
        "autoencoder_status": None
    }
