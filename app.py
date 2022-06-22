from __future__ import division, print_function
# coding=utf-8
import sys
import os, shutil
import glob
import re
import numpy as np
from IPython.display import Image as im1, display
from tensorflow.keras.preprocessing import image as im
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from werkzeug.security import safe_join
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Define a flask app
app = Flask(__name__, static_url_path='')


app.config['HEATMAP_FOLDER'] = 'heatmap'
app.config['UPLOAD_FOLDER'] = 'uploads'
# Model saved with Keras model.save()
MODEL_PATH = 'models/model_v1.h5'


#Load your trained model
model = load_model(MODEL_PATH)
        # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')


@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)




def model_predict(img_path, model):
    img = Image.open(img_path).resize((224,224)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
   
    preds = model.predict(img)
    pred = np.argmax(preds,axis = 1)
    print(preds)
    return pred,img


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        file_name=os.path.basename(file_path)
        # Make prediction
        pred,img = model_predict(file_path, model)
        #os.remove(file_path)#removes file from the server after prediction has been returned

        # Arrange the correct return according to the model. 
		# In this model 1 is Pneumonia and 0 is Normal.
        str1 = 'Basal Cel Carcinoma'
        str2 = 'Melanoma'
        str3 = 'Nevus'
        
        last_conv_layer_name = "block_16_depthwise"
        heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)
        fname=save_and_display_gradcam(file_path, heatmap)
        if pred[0] == 0:
            res=str1
            
        elif pred[0] == 1:
            res= str2
        else:
            res= str3
        
        
    return render_template('predict.html',file_name=file_name, heatmap_file=fname,result=res)


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    # fname=img_path.split('.')[-1]
    file_name=os.path.basename(img_path)
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    # img = im.open(img_path).resize((224,224)) #target_size must agree with what the trained model expects!!
    
    # # Preprocessing the image
    # img = im.img_to_array(img)
    # img = np.expand_dims(img, axis=0)
    # img = img.astype('float32')/255

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    file_name=file_name+"_"+cam_path
    basepath = os.path.dirname(__file__)
    cam_path = os.path.join(
            basepath, 'uploads', secure_filename(file_name))
    # Save the superimposed image
    
    
    superimposed_img.save(cam_path)
    
    return file_name

    # Display Grad CAM
   # display(Image(cam_path))

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
