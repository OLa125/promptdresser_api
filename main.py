from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
from numpy.linalg import norm
import insightface

app = FastAPI()

# تحميل نموذج التعرف على الوجه
face_app = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

@app.post("/verify/")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...), threshold: float = 0.6):
    # قراءة الصور
    img_bytes1 = await file1.read()
    img_bytes2 = await file2.read()

    img_array1 = np.frombuffer(img_bytes1, np.uint8)
    img_array2 = np.frombuffer(img_bytes2, np.uint8)

    img1 = cv2.imdecode(img_array1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(img_array2, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        return JSONResponse(content={"error": "تعذر قراءة الصور."}, status_code=400)

    # استخراج الوجوه
    faces1 = face_app.get(img1)
    faces2 = face_app.get(img2)

    if not faces1 or not faces2:
        return JSONResponse(content={"error": "لم يتم الكشف عن وجه في إحدى الصور."}, status_code=400)

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    similarity = cosine_similarity(emb1, emb2)
    is_same = similarity > threshold

    return {
        "similarity": round(float(similarity), 4),
        "is_same_person": is_same
    }

# لو حابة تجربي محليًا
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)