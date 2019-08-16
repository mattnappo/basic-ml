from flask import Flask, render_template, flash, request, redirect, url_for, session, send_from_directory, logging
from wtforms import Form, StringField, TextAreaField, PasswordField, validators, FileField
from werkzeug.utils import secure_filename
from net import NeuralNet, FashionNetPredictor
import random, os

net = NeuralNet()

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app = Flask(__name__, static_url_path='/static')

class EncryptForm(Form):
    text = TextAreaField("", [validators.DataRequired()])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/decrypt', methods=['GET', 'POST'])
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
            
            fashion_net = FashionNetPredictor(net, './data/custom_sneaker.jpg')
            prediction = fashion_net.prediction

            print(prediction)
            return render_template("decrypt.html", prediction=prediction)

    else:
        session['filename'] = ""

    return render_template("decrypt.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(host= '0.0.0.0', debug=True)
