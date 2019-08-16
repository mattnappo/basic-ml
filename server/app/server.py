from flask import Flask, render_template, flash, request, redirect, url_for, session, send_from_directory, logging
from wtforms import Form, StringField, TextAreaField, PasswordField, validators, FileField
from werkzeug.utils import secure_filename
from net import NeuralNet, FashionNetPredictor
import random, os
import hashlib

def md5(s):
    hash_object = hashlib.md5(s.encode())
    return hash_object.hexdigest()
    
net = NeuralNet()

fashion_net = FashionNetPredictor(net)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg'])

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class EncryptForm(Form):
    text = TextAreaField("", [validators.DataRequired()])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    session['filename'] = ""

    if request.method == 'POST':

        if 'file' not in request.files :
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            PATH = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(PATH)
            session['filename'] = filename
            
            prediction = fashion_net.predict('./static/uploads/' + secure_filename(file.filename))

            print('' + str(prediction) + '\n##########')
            return render_template("predict.html", prediction=prediction)

    else:
        session['filename'] = ""

    return render_template("predict.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(host= '0.0.0.0', debug=True)
