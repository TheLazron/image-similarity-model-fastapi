import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial import distance

app = FastAPI()

model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
IMAGE_SHAPE = (224, 224)
layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])

@app.post("/compare_images")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    image1 = Image.open(file1.file).convert('L').resize(IMAGE_SHAPE)
    image2 = Image.open(file2.file).convert('L').resize(IMAGE_SHAPE)

    image1 = np.stack((image1,)*3, axis=-1)
    image2 = np.stack((image2,)*3, axis=-1)

    image1 = np.array(image1)/255.0
    image2 = np.array(image2)/255.0

    feature_vector1 = np.array(model.predict(image1[np.newaxis, ...])).flatten()
    feature_vector2 = np.array(model.predict(image2[np.newaxis, ...])).flatten()

    metric = 'cosine'
    distance_metric = distance.cdist([feature_vector1], [feature_vector2], metric)[0]

    return {"distance": distance_metric.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
