import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import tensorflow as tf
import uvicorn

# Constants
MODEL_PATH = 'models/cat_dog_classifier.h5'  # Updated path to H5 model file
IMG_SIZE = 224  # Same as in training

app = FastAPI(title="Cat vs Dog Classifier API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
print(f"Loading model from {MODEL_PATH}...")
try:
    # Try to load the model directly
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Creating a new model instance...")
    
    # Create a new model with the same architecture
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    
    # Create the base model from MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Load weights from the saved model if possible
    try:
        model.load_weights(MODEL_PATH)
        print("Model weights loaded successfully!")
    except Exception as weight_error:
        print(f"Error loading weights: {weight_error}")
        print("Using model with ImageNet weights only.")

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("Static files mounted successfully")
except Exception as e:
    print(f"Error mounting static files: {e}")

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    # Resize image to match the input size expected by the model
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0  # Normalize to [0,1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)[0][0]
        
        # Convert prediction to class and confidence
        is_dog = prediction > 0.5
        confidence = float(prediction) if is_dog else float(1 - prediction)
        
        # Return result in the format expected by the frontend
        return {
            "prediction": "dog" if is_dog else "cat",
            "confidence": confidence,
            "filename": file.filename,
            "content_type": file.content_type
        }
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)