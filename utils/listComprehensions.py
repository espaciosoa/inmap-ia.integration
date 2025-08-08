import numpy as np

def serialize_doc(doc):
    doc["_id"] = str(doc["_id"])
    return doc

def getJustCoords(measurement):
    # print(measurement["position"]["y"])
    del measurement["position"]["y"]

    return (measurement["position"]["x"],measurement["position"]["z"])

def getValueAndCoords(measurement, key):
    return (measurement["position"]["x"],measurement["position"]["z"], measurement["fullCellSignalStrength"][key])

def dbm_to_mw(dbm_value):
    return 10 ** (dbm_value / 10)

def normalize_dbm(data):    
    # Extraemos la tercera columna (que es la que contiene los dbm)
    columna_dbm = [fila[2] for fila in data]  # Ej: [-70, -90, -100]
    
    # ? Pasamos a escala lineal???
    #columna_mw = [dbm_to_mw(dbm) for dbm in columna_dbm]

    min_val = min(columna_dbm)
    max_val = max(columna_dbm)
    
    # Normalizamos entre 0 y 1 y reemplazamos
    # Creamos nuevas tuplas normalizadas (asume que max_val != min_val)
    return [
        (*fila[:2], (fila[2] - min_val) / (max_val - min_val))
        for fila in data
    ]
