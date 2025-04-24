from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv5 model (assumes best.pt is in the same folder)
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)

@app.get("/")
def home():
    return {"message": "Gym Machine Detection API running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file sent by the frontend
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Perform inference with YOLOv5 model
    results = model(image)

    # Get the labels and predicted classes
    pred_labels = results.names
    pred_classes = results.pred[0][:, -1].tolist()

    # If no machine detected, return an appropriate message
    if not pred_classes:
        return {"detected_machine": "No machine detected"}

    # Get the first predicted class
    first_class_id = int(pred_classes[0])
    detected_label = pred_labels[first_class_id]

    return {"detected_machine": detected_label}
