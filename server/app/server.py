from flask import Flask, render_template, flash, request, redirect, url_for, session, send_from_directory, logging
from wtforms import Form, StringField, TextAreaField, PasswordField, validators, FileField
from werkzeug.utils import secure_filename
from labels import labels
import tensorflow as tf
from tensorflow import keras as kr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random, os
import hashlib

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

# tf_config = some_custom_config
# sess = tf.Session(config=tf_config)

sess = tf.Session()

graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)

mnist = kr.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255
test_images  = test_images  / 255

model = kr.Sequential([
    kr.layers.Flatten(
        input_shape = (28, 28)
    ),
    kr.layers.Dense(128, activation=tf.nn.relu),
    kr.layers.Dense(len(labels), activation=tf.nn.softmax)
])

model.compile(
    optimizer = 'adam',
    loss      = 'sparse_categorical_crossentropy',
    metrics   = ['accuracy']
)

# model.fit(train_images, train_labels, epochs=5)
model.fit(train_images, train_labels, epochs=1)

predictions_ = model.predict(test_images)
print(predictions_)
print('\n\n\n')



# graph = tf.get_default_graph()

def flatten_image(path):
    img = Image.open(path).convert('F')
    WIDTH, HEIGHT = img.size

    if WIDTH != 28 or HEIGHT != 28:
        img = img.resize((28, 28))
        
    WIDTH, HEIGHT = img.size

    print(img.size)

    data = list(img.getdata())
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = int(255 - data[i][j])
    
    return data

def predict_image(path):
    image = flatten_image(path)

    with graph.as_default():
        prediction = model.predict([[image]])

    plot(prediction, image)

def plot_image_predict(prediction_vectors, image):
    prediction_vector = prediction_vectors[0]
    
    plt.grid(False)
    
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(image, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(prediction_vector)
        
    plt.xlabel(
        'Guess: {} ({:2.0f} % certain)'.format(
            labels[predicted_label],
            100 * np.max(prediction_vector)
        )
    )

def plot_value_array_predict(prediction_vectors):
    prediction_vector = prediction_vectors[0]
    
    plt.grid(False)
    
    plot = plt.bar(
        range(len(labels)),
        prediction_vector,
        color = '#fa34ab'
    )
    
    plt.xlabel('Clothing type')
    plt.ylabel('Certainty')
    
    plt.xticks(
        range(len(labels)),
        labels,
        size='small',
        rotation=90
    )
    
    plt.ylim([0, 1])

    nums = []
    for num in np.arange(0, 120, 20):
        nums.append(str(num) + "%")

    plt.yticks(np.arange(0, 1.2, .2), nums, size='small')

    predicted_label = np.argmax(prediction_vector)
    
    plot[predicted_label].set_color('red')
    
def plot(prediction, image):
    plt.figure(figsize = (8, 4))
    plt.subplot(1, 2, 1)

    plot_image_predict(prediction, image)
    plt.subplot(1, 2, 2)

    plot_value_array_predict(prediction)

    plt.tight_layout()
    
    # plt.show()
    plt.savefig('somefile.png')

def md5(s):
    hash_object = hashlib.md5(s.encode())
    return hash_object.hexdigest()

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
def predict():
    session['filename'] = ""

    if request.method == 'POST':

        if 'file' not in request.files:
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
            
            abs_path = os.path.abspath(UPLOAD_FOLDER + secure_filename(file.filename))

            # prediction = predict_image(abs_path)

            image = flatten_image(abs_path)

            # print(tf.global_variables())
            # predictor = tf.predict(image, name="Predict")
            # with tf.Session() as sess:
            #     print(sess.run(predictor))
            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                prediction = model.predict([[image]])
                print('\n# # # # # # # # # # # # # # #' + str(prediction) + '\n# # # # # # # # # # # # # # #')

            return render_template("predict.html", prediction='nothing')

    else:
        session['filename'] = ""

    return render_template("predict.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(host= '0.0.0.0', debug=True, threaded=False)
