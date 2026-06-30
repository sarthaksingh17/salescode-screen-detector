"""
FastAPI backend for the live camera demo.
Run: python app.py or uvicorn app:app --reload --port 5000
Then open http://localhost:5000 in your browser.
"""

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import base64
import io
import os
from PIL import Image
from predict import predict as run_predict
import uvicorn

app = FastAPI()

@app.get("/")
def index():
    return FileResponse("demo.html")

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        if not data or "image" not in data:
            return JSONResponse(status_code=400, content={"error": "No image provided"})

        img_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        tmp_path = "tmp_frame.jpg"
        img.save(tmp_path, "JPEG", quality=95)
        
        score = run_predict(tmp_path)
        os.remove(tmp_path)

        return {
            "score": round(score, 4),
            "label": "SCREEN" if score >= 0.5 else "REAL",
            "confidence": round(max(score, 1 - score) * 100, 1)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
