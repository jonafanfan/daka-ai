from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid, os, tempfile
from scene_analysis import analyze_scene, _load_clip

@asynccontextmanager
async def lifespan(_):
    _load_clip()  # warm up CLIP before the first request
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)
    try:
        result = analyze_scene(tmp_path)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)