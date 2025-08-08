# This file is the starting point of the app
from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import io
import numpy as np
# from db import db, measurements,rooms
from PIL import Image, ImageDraw
# The other project




def bounding_box_2d(points):
    """
    Given a list of (x, z) points, return the bounding box as
    ((min_x, min_x), (max_z, max_z))
    """
    if not points:
        return None

    xs, zs = zip(*points)
    print(xs,zs)
    return {"min": {"x": min(xs),
                    "z": min(zs)},
            "max": { "x":max(xs),
                     "z":max(zs)}}


def numpy_array_to_base64_image(array_2d: np.ndarray) -> str:
    # Normalize if float (0.0–1.0) → 0–255
    if array_2d.dtype != np.uint8:
        array_2d = (255 * (array_2d - np.min(array_2d)) / (np.ptp(array_2d) + 1e-8)).astype(np.uint8)

    # Convert to grayscale image
    image = Image.fromarray(array_2d, mode='L')  # 'L' for 8-bit grayscale

    # Save to bytes
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)

    # Encode as base64
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    return base64_img