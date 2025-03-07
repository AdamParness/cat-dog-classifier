import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import shutil
import random
from PIL import Image
import io

# Set paths
DATASET_PATH = 'kagglecatsanddogs_5340/PetImages'  # This contains 'Cat' and 'Dog' folders
MODEL_SAVE_PATH = 'models/cat_dog_classifier.h5'
os.makedirs('models', exist_ok=True)

# Create temporary train/validation split
TEMP_TRAIN_DIR = 'data_temp/train'
TEMP_VAL_DIR = 'data_temp/validation'

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 1
VALIDATION_SPLIT = 0.2  # 20% for validation

def is_valid_image(file_path):
    """Check if the image is valid and can be opened by PIL"""
    try:
        img = Image.open(file_path)
        img.verify()  # Verify it's an image
        return True
    except (IOError, SyntaxError, OSError):
        return False

def setup_train_val_dirs():
    """Create temporary train/validation directory structure"""
    # Remove temp directories if they exist
    if os.path.exists('data_temp'):
        shutil.rmtree('data_temp')
    
    # Create temp directories
    for dir_path in [
        TEMP_TRAIN_DIR + '/cats', 
        TEMP_TRAIN_DIR + '/dogs',
        TEMP_VAL_DIR + '/cats',
        TEMP_VAL_DIR + '/dogs'
    ]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Split and copy cat images
    cat_images = os.listdir(os.path.join(DATASET_PATH, 'Cat'))
    valid_cat_images = []
    
    print("Filtering cat images...")
    for img in cat_images:
        img_path = os.path.join(DATASET_PATH, 'Cat', img)
        if os.path.isfile(img_path) and is_valid_image(img_path):
            valid_cat_images.append(img)
        
    print(f"Found {len(valid_cat_images)} valid cat images out of {len(cat_images)}")
    
    random.shuffle(valid_cat_images)
    split_idx = int(len(valid_cat_images) * (1 - VALIDATION_SPLIT))
    
    for i, img in enumerate(valid_cat_images):
        src = os.path.join(DATASET_PATH, 'Cat', img)
        if i < split_idx:
            dst = os.path.join(TEMP_TRAIN_DIR, 'cats', img)
        else:
            dst = os.path.join(TEMP_VAL_DIR, 'cats', img)
        shutil.copy(src, dst)
    
    # Split and copy dog images
    dog_images = os.listdir(os.path.join(DATASET_PATH, 'Dog'))
    valid_dog_images = []
    
    print("Filtering dog images...")
    for img in dog_images:
        img_path = os.path.join(DATASET_PATH, 'Dog', img)
        if os.path.isfile(img_path) and is_valid_image(img_path):
            valid_dog_images.append(img)
            
    print(f"Found {len(valid_dog_images)} valid dog images out of {len(dog_images)}")
    
    random.shuffle(valid_dog_images)
    split_idx = int(len(valid_dog_images) * (1 - VALIDATION_SPLIT))
    
    for i, img in enumerate(valid_dog_images):
        src = os.path.join(DATASET_PATH, 'Dog', img)
        if i < split_idx:
            dst = os.path.join(TEMP_TRAIN_DIR, 'dogs', img)
        else:
            dst = os.path.join(TEMP_VAL_DIR, 'dogs', img)
        shutil.copy(src, dst)
    
    print(f"Created temporary train/validation split:")
    print(f"Training: {len(os.listdir(os.path.join(TEMP_TRAIN_DIR, 'cats')))} cats, {len(os.listdir(os.path.join(TEMP_TRAIN_DIR, 'dogs')))} dogs")
    print(f"Validation: {len(os.listdir(os.path.join(TEMP_VAL_DIR, 'cats')))} cats, {len(os.listdir(os.path.join(TEMP_VAL_DIR, 'dogs')))} dogs")

# Create the train/validation split
setup_train_val_dirs()

# Data generators with augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TEMP_TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    TEMP_VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load the MobileNetV2 model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# Save the model in multiple formats
print("Saving the model...")

# 1. Save in H5 format (legacy but widely compatible)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# 2. Save in SavedModel format (recommended for TensorFlow serving)
try:
    saved_model_path = 'models/cat_dog_model'
    tf.saved_model.save(model, saved_model_path)
    print(f"SavedModel format saved to {saved_model_path}")
except Exception as e:
    print(f"Error saving in SavedModel format: {e}")

# Skip TFLite conversion as it's causing errors
print("TensorFlow Lite conversion skipped due to compatibility issues.")
print("You can use the saved H5 or SavedModel formats for inference.")

# Print a summary of the model architecture
model.summary()