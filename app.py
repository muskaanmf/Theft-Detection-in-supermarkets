from flask import Flask, request, render_template, redirect, url_for, Response
import subprocess
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/prelabel_page')
# def prelabel_page():
#     with open('prelabel.pickle', 'rb') as f:
#         prelabel = pickle.load(f)
#     return render_template('prelabel.html', prelabel=prelabel)


@app.route('/predict_static', methods=['POST'])
def predict_static():
    file = request.files['file']
    file_path = file.filename
    file.save(file_path)
    command = f'python predict_video.py -m model.hdf5 -l label_bin -i {file_path} -o new.mp4'
    # Execute command
    subprocess.run(command, shell=True)
    with open('prelabel.pickle', 'rb') as f:
        prelabel = pickle.load(f)
    return render_template('prelabel.html', prelabel=prelabel)

@app.route('/predict_realTime', methods=['POST'])
def predict_realTime():
    command = f'python predict_video_realtime.py -m model.hdf5 -l label_bin -o output'
    subprocess.run(command, shell=True)
    with open('prelabelreal.pickle', 'rb') as f:
        prelabel = pickle.load(f)
    return render_template('prelabel.html', prelabel=prelabel)

if __name__ == '__main__':
    app.run(debug=True)