from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load Keras .h5 model
model = tf.keras.models.load_model("artifacts/model.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    image = image.resize((128,128))
    x = np.array(image) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    y = model.predict(x)[0]

    # Convert output → image
    y = (y * 255).astype("uint8")
    out_image = Image.fromarray(y)

    # Return image bytes
    buf = io.BytesIO()
    out_image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
