from fastapi import FastAPI, UploadFile, File
from io import BytesIO
import numpy as np
import tensorflow as tf

app = FastAPI(
    title="FastAPI Keras Model Service",
    description="FastAPI service for EfficientDFU diabetic foot ulcer classification",
    version="1.0.0"
)

model = None

# These must match the directory ordering from the notebook.
class_names = [
    "Abnormal(Ulcer)",
    "Normal(Healthy skin)"
]


@app.on_event("startup")
async def startup_event():
    global model

    model = tf.keras.models.load_model(
        "app/models/efficientdfu-v1.keras"
    )

    print("Model loaded successfully")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)


@app.get("/")
async def root():
    return {
        "message": "EfficientDFU Model API"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image bytes
    image_bytes = await file.read()

    # Convert bytes to PIL image
    image = tf.keras.utils.load_img(
        BytesIO(image_bytes),
        target_size=(224, 224),
        color_mode="rgb"
    )

    # Convert PIL image to NumPy array
    image = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    # (224, 224, 3) -> (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)

    # Run inference
    predictions = model.predict(
        image,
        verbose=0
    )

    # Model output:
    # [probability_class_0, probability_class_1]
    predicted_index = int(np.argmax(predictions[0]))

    predicted_class = class_names[predicted_index]

    confidence = float(predictions[0][predicted_index])

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            class_names[0]: float(predictions[0][0]),
            class_names[1]: float(predictions[0][1])
        }
    }