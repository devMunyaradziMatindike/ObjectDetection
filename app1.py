# from warnings import filterwarnings
from flask import Flask, flash, request, redirect, render_template, jsonify
# from werkzeug.utils import secure_filename
#from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
#import joblib
import base64
from PIL import Image
from io import BytesIO
import tensorflow as tf
import cv2
#import json
import numpy as np
import math
import os
import pandas as pd
# import pandas as pd

UPLOAD_FOLDER = './static/images/'
ALLOWED_EXTENSIONS = {'mp4'}
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'super secret key'
# Set the maximum file size to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

items = ['heel', 'sdfg', 'dfeer']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# error handler for 413 status code


@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('upload.html', message='File is too large. File limit is 10mb.'), 413


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':

        # check if the post request has the file part

        file = request.files['file']

        # if file.content_length > app.config['MAX_CONTENT_LENGTH']:
        #     return 'File size exceeds the maximum allowed size'

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if 'file' in request.files:
                save_video_frames(file)
                make_df('static/images/train')
                items = get_classes()
                # model = load_model('models/inception.h5')
                # items = ['heel','sdfg','dfeer']

                # print('tapinda')
                # # model = joblib.load('models/VGG16_model.pkl')
                # kmeans = joblib.load('models/image_clustering_model.pkl')
                # model = load_model("models/VGG16_model.h5")
                # # img = Image.open(request.files['file']).convert('L')
                # test_image = Image.open(request.files['file']).convert('L')
                # test_image = test_image.resize((224, 224))
                # test_image = test_image.convert('L')
                # test_image = np.array(test_image) / 255.0
                # test_image = np.expand_dims(test_image, axis=-1)
                # test_image = np.repeat(test_image, 3, axis=-1)
                # test_feature = model.predict(
                #     np.expand_dims(test_image, axis=0)).flatten()
                # test_label = kmeans.predict(
                #     np.array([test_feature.astype(float)]))[0]

                # print(test_label)

                return render_template('main.html')
            else:
                print(items)
    return render_template('main.html',)


@app.route('/search', methods=['POST'])
def search():
    search_input = request.form['text-input']
    if len(search_input) > 0:
        cluster = generate_cluster_image(search_input)
        # buffered = BytesIO()
        # cluster.save(buffered, format="JPEG")
        img_str = base64.b64encode(cluster.getvalue())
        image_data = base64.b64encode(
            cluster.getvalue()).decode('utf-8')
    # Do something with the search input data
    print(search_input)
    return render_template('main.html', items=image_data)


def generate_cluster_image(search_text):
    dataframe = pd.read_csv('preds.csv')
    dataframe = dataframe[['Class', 'Source']]
    image_filenames = os.listdir('static/images/train')
    images = []
    for filename in image_filenames:
        image_path = os.path.join('static/images/train', filename)
        image = Image.open(image_path)
        images.append(image)

    needed_images = dataframe[dataframe['Class'] ==
                              search_text]['Source'].unique().tolist()

    # Step 6: Plot the similar images on one figure with subplots
    fig = plt.figure(figsize=(30, 30), frameon=True)
    fig.suptitle(f'Items with {search_text}', fontsize=18)
    columns = 6
    rows = 5
    for i in range(1, len(needed_images)+1):
        if i == 30:
            break
        # img = plt.imread(similar_images[i])
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(images[i-1])
        plt.axis('off')
        # plt.title(f"k = {i}")
    plt.show()
    # plt.show()
    fig.savefig('static/images/result.png')
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png')

    # im = Image.open(img_buf)
    # im.show(title="My Image")

    # img_buf.close()
    return img_buf


def _predict(frame):
    # Load the pre-trained Inception v3 model
    model = tf.keras.applications.InceptionV3(weights='imagenet')
    # pred_df = pd.DataFrame(columns=['Class','Source'])
    # Load the image
    img = cv2.imread(frame)

    # Preprocess the image
    img = cv2.resize(img, (299, 299))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Run the image through the model to get predictions
    predictions = model.spredict(img)

    # Decode the predictions
    decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(
        predictions)

    preds = []
    # Print the top 5 predictions
    for i in range(5):
        # print(decoded_predictions[0][i][1])
        preds.append({'Class': decoded_predictions[0][i][1], 'Source': frame})
        print(decoded_predictions[0][i][1])
        # print((decoded_predictions[0][i][1],frame))

    return preds


def get_classes():
    dataframe = pd.read_csv('preds.csv')
    dataframe = dataframe[['Class', 'Source']]
    return dataframe['Class'].unique().tolist()


def make_df(file_path):
    predictions_df = pd.DataFrame(columns=['Class', 'Source'])
    predictions = []
    for filename in os.listdir(file_path):
        pred = []
        print(filename)

        pred = _predict(os.path.join('./train', filename))
        print(type(pred))

        for prediction in pred:
            predictions_df = predictions_df.append(
                prediction, ignore_index=True)
    predictions_df.to_csv('preds.csv')


def save_video_frames(video_path, fname):
    count = 0
    # capturing the video from the given path
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)  # frame rate
    x = 1
    names = []
    while (cap.isOpened()):

        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()

        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a new folder named train_1

            filename = 'static/images/train' + fname + "_frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()


if __name__ == '__main__':
    app.run(debug=True)
