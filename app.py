#importing flask libraries
from flask import Flask, render_template, request, redirect, url_for,  send_file
from flask import send_from_directory

#importing libraries
import numpy as np
import os
import random
import string
import matplotlib
from keras.models import load_model
from skimage.transform import resize
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img

#Defining paths
UPLOAD_FOLDER = 'uploads_image/images'
PREDICTED_FOLDER = 'images'
path = 'uploads_image/'
img_path = 'images/'


#Loading the model
json_file = open('model/model_num.json', 'r') #Loading the model json file
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model/model_weights.h5") #Loading the model weights
loaded_model._make_predict_function()

#Flask app
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTED_FOLDER'] = PREDICTED_FOLDER

@app.route('/')
def home():
   return render_template('home.html')

#Predict loaded image
def pred_image(X):
   pred = loaded_model.predict(X, verbose=1)
   pred_b = (pred > 0.5).astype(np.uint8)
   imgs = X.reshape(128,128)
   pred = pred.reshape(128,128)
   pred_b = pred_b.reshape(128,128)
   imgs_save(imgs, pred, pred_b)

#Deleteting previous loaded images
def del_pred_image():
   mdir = app.config['PREDICTED_FOLDER']
   filelist = [ f for f in os.listdir(mdir) if f.endswith(".png") ]
   for f in filelist:
   	os.remove(os.path.join(mdir, f))

def imgs_save(imgs, pred, pred_b):
   import matplotlib
   del_pred_image()
   
   #Random names for predicted names
   n = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
   na = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
   nam = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
   
   #Saving the predicted images
   matplotlib.image.imsave(img_path+'a'+n+'.png', imgs)
   matplotlib.image.imsave(img_path+'b'+na+'.png', pred)
   matplotlib.image.imsave(img_path+'c'+nam+'.png', pred_b)

#Main method
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':
      #Saving image     
      f = request.files['file']
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],"test_image.png"))
   
   #Loading the image
   ids = 'test_image.png'
   for n in range(1):
      X = np.zeros((1, 128, 128, 1),dtype=np.float32)
      img  = load_img(path + '/images/' + ids, grayscale=True)
      x_img = img_to_array(img)
      x_img = resize(x_img, (128,128,1), mode='constant', preserve_range=True)
      X[n,...,0] = x_img.squeeze()/255  
      
   
   #Predicting the image
   pred_image(X)
      
   #Return    
   return render_template("home.html", msg="completed")

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory("images",filename)

@app.route('/gallery')
def gallery():
    image_names = os.listdir('./images')
    return render_template("gallery.html", image_names=image_names)

@app.route('/back', methods = ['POST'])
def back():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
