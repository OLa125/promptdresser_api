from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
from typing import Optional

app = FastAPI()

model_app = None

def load_model():
    global model_app
    if model_app is None:
        from insightface.app import FaceAnalysis
        model_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        model_app.prepare(ctx_id=0)

def read_image(uploaded_file: UploadFile):
    contents = uploaded_file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def get_embedding(img):
    faces = model_app.get(img)
    if not faces:
        return None
    return faces[0].embedding

@app.post("/verify/")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...), threshold: Optional[float] = 0.6):
    load_model()

    img1 = read_image(file1)
    img2 = read_image(file2)

    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    if emb1 is None or emb2 is None:
        return JSONResponse({"verified": False, "detail": "Face not detected in one or both images"}, status_code=400)

    dist = np.linalg.norm(emb1 - emb2)
    verified = dist < threshold

    return {"verified": verified, "distance": float(dist), "threshold": threshold}
