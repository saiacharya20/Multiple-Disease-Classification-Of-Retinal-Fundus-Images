from flask import Flask, render_template, request,redirect,url_for
import keras
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import expand_dims
from skimage import data, img_as_float,io
from skimage import exposure,filters,feature
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.io import imread
from skimage.transform import resize
from skimage import data, io, img_as_ubyte
from keras.applications.vgg16 import VGG16
from glob import glob
from PIL import ImageOps,ImageFilter,Image
from keras.models import load_model
from skimage import exposure,filters,feature
from skimage.filters import unsharp_mask,gaussian


app = Flask(__name__)

vgg = tf.keras.models.load_model(r'C:\Users\saiac\Documents\Sai\Projects\Multiple-Retinal-Disease-Detection-main\Flask_App\VGG-16.h5')
#vgg=load_model('VGG-16.h5')

# routes
# @app.route("/")
# def main():
# 	return render_template("prediction.html")

@app.route("/", methods = ['GET']) #index.html
def main():
	return render_template("index.html")

@app.route("/prediction", methods = ['GET'])
def load():
	return render_template("prediction.html")

@app.route("/about_us", methods = ['GET'])
def about():
	return render_template("about_us.html")


@app.route("/prediction", methods = ['POST'])
def prediction():
	imagefile = request.files['img']
	image_path = "static/image/" + imagefile.filename
	imagefile.save(image_path)
	image_pat = "image/" + imagefile.filename
	var = "modal"

	#crop image code
	# img= cv2.imread('static/image/' + imagefile.filename)
	# original_image= img
	# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# edges= cv2.Canny(gray, 50,200)
	# contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
	# for (i,c) in enumerate(sorted_contours):
	# 	x,y,w,h= cv2.boundingRect(c)
	# 	cropped_contour= original_image[y:y+h, x:x+w]
	# img = cropped_contour
	# down_width = 224
	# down_height = 224
	# down_points = (down_width, down_height)
	# img = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)

	SIZE=224
	img = image.load_img(image_path, target_size=(SIZE,SIZE,3))
	img = img.filter(ImageFilter.SHARPEN)
	# kernel = np.array([[0, -1, 0],
    #                [-1, 5,-1],
    #                [0, -1, 0]])
	# img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
	img = image.img_to_array(img)
	img = img/255.
	p2, p98 = np.percentile(img , (2,98))
	p20, p90 = np.percentile(img , (20,90))
	p20, p95 = np.percentile(img , (2,95))
	img = exposure.rescale_intensity(img, in_range=(p2,p98))
	img = exposure.rescale_intensity(img, in_range=(p20,p95))
	img = exposure.rescale_intensity(img, in_range=(p20,p90))
	img = gaussian(img, sigma = 0.5)
	img = unsharp_mask(img, radius = 50, amount=2)
	img = np.expand_dims(img, axis=0)



	columns=['ID','Diabetic Retinopathy','Media Haze','Optic Disc Cupping']
	classes = np.array(columns[1:]) #Get array of all classes
	proba = vgg.predict(img)*100  #Get probabilities for each class
	sorted_categories = np.argsort(proba[0][:-11:-1]) #Get class names for top 10 categories
	if(proba[0][sorted_categories[0]]) >= 50:
		disease = classes[sorted_categories[0]]  
		accu = str(proba[0][sorted_categories[0]])
	disease1 = classes[sorted_categories[0]]
	accu1 = str(proba[0][sorted_categories[0]])
		# return("{}".format(classes[sorted_categories[0]])+" ({:.3})".format(proba[0][sorted_categories[0]]))
	if(proba[0][sorted_categories[1]]) >= 50:
		disease = classes[sorted_categories[1]]
		accu = str(proba[0][sorted_categories[1]])
	disease2 = classes[sorted_categories[1]]
	accu2 = str(proba[0][sorted_categories[1]])
		# return("{}".format(classes[sorted_categories[1]])+" ({:.3})".format(proba[0][sorted_categories[1]]))
	if(proba[0][sorted_categories[2]]) >= 50:
		disease = classes[sorted_categories[2]]
		accu = str(proba[0][sorted_categories[2]])
	disease3 = classes[sorted_categories[2]]
	accu3 = str(proba[0][sorted_categories[2]])
		# return("{}".format(classes[sorted_categories[2]])+" ({:.3})".format(proba[0][sorted_categories[2]])) 
	accu = round(float(accu), 2)
	accu1 = round(float(accu1), 2)
	accu2 = round(float(accu2), 2)
	accu3 = round(float(accu3), 2)
	return render_template("prediction.html",dis=disease,accuracy=accu,dis1=disease1,dis2=disease2,dis3=disease3,ac1=accu1,ac2=accu2,ac3=accu3,filename=image_pat,var=var)

		

if __name__ =='__main__':
	app.debug = True
	app.run(debug = True)