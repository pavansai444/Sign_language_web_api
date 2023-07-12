# Flask utils
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
import numpy
import cv2
from waitress import serve

#custom model deployment
model= load_model(r'new_model20.h5')
list = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
# Defining a flask app
app = Flask(__name__)

@app.route("/")
def index(): 
    return render_template("index.html")


@app.route('/uploader', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static','uploads', secure_filename(f.filename))
        f.save(file_path)
        image = cv2.imread(file_path)
        image = cv2.resize(image,(96,96))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = image.reshape(-1,96,96,3)
        preds = model.predict(image)
        j = numpy.argmax(preds[0])
        return render_template("upload.html", predictions=("The Sign You Uploaded is "+str(list[j])+" confidence: "+str(preds[0][j])), display_image=f.filename) 


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method =='POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static','uploads', secure_filename(f.filename))
        f.save(file_path)
        image = cv2.imread(file_path)
        image = cv2.resize(image,(96,96))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = image.reshape(-1,96,96,3)
        preds = model.predict(image)
        j = numpy.argmax(preds[0])
        return jsonify({"result":str(list[j])})
        

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app,host="0.0.0.0",port=port)
    #print("WEBSITE LIVE")
    #app.run(host="0.0.0.0",debug=True,port="4100")
