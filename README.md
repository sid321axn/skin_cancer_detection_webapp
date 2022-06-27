# Flask web app for Skin Cancer Detection

[![](https://img.shields.io/badge/python-2.7%2C%203.5%2B-green.svg)]()



------------------
## About the app
> This repository was created to help clarify how to utilise flask and gunicorn to easily deploy a python/keras deep learning model as a web app on Azure. This example features code for online deployment of a multi-class medical image classification model, based on convolutional neural network architecture. The CNN has three hidden layers and has been trained on the following  Skin Cancer dataset  <a href="https://challenge.isic-archive.com/data/"> ISIC Skin Cancer Challenge 2019</a>. The trained model achieved accuracy of more than 97% on the test set and its weights have been saved in the Models folder (see file: my_model_v1.h5) in the very useful HDF5 format. You may use your own saved trained model! Just make sure you put it in the Models folder and name it appropriately so that the flask app may call it.



> This web application has been created and the changes to whatManu Siddhartha  had already prepared were the following:
<ul>

<li>Inclusion of gunicorn to the requirements.txt file</li>
<li>Changed the app.py (flask app) file to adapt it to the required functionality according to the trained binary image classification model. The program was also modified to delete every uploaded image after providing the prediction. This will prevent exceeding capacity limits on Heroku servers. The last lines of the file have been modified to work with gunicorn. </li>
<li>The Index.html and base.html files have been modified accordingly to include references and information about the model</li>
<li>This README file has been adapted to provide instructions about heroku deployment. The part for customisation has been modified also</li>
</ul>

> To create a web app that runs locally follow instructions in the README file, to run the model locally. Gevent or gunicorn may be used for local deployment too.

## Customization options

### Use your own model

Place your trained keras deep learning model to the models directory.


### Use other pre-trained model

See [Keras applications](https://keras.io/applications/) for more available models such as DenseNet, MobilNet, NASNet, etc.


### UI Modification

Modify files in `templates` and `static` directory.

`index.html`, `predict.html` for the UI and `app.py` for all the behaviors

## Deployment

Update soon


