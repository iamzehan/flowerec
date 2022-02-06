import gradio as gr
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
json_file = open('./model/flower_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./model/flower_model.h5")

labels = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

def classify_image(inp):
  image_array= np.expand_dims(inp, axis=0)
  predictions=model.predict(image_array)
  score = tf.nn.softmax(predictions[0])
  return {labels[i]: float(score[i]) for i in range(len(labels))}

image = gr.inputs.Image(shape=(180, 180))
label = gr.outputs.Label(num_top_classes=6)

gr.Interface(fn=classify_image, title='Flower Recognizer', inputs=image, theme='dark-grass', outputs=label).launch()