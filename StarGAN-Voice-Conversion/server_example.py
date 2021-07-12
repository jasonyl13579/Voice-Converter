# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:17:22 2018

@author: JasonHuang
"""
from flask import Flask, send_file, request, abort, g
import sys, getopt
import time
from collections import Counter
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
import scipy.io as sio
from werkzeug import secure_filename
#from keras.backend.tensorflow_backend import set_session
#from flask_script import Manager

## StarGAN model
from model import StarGANVC
from preprocess import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utility import *

import pickle
from scipy.io.wavfile import read
from featureextraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time
app = Flask(__name__)
#manager = Manager(app)

#%% parameter to modify

port = 8000
model_dir = './online/model/'
test_dir =  './online/oursounds'
output_dir = './online/convert'
receive_dir= './online/receive'
app.config['UPLOAD_FOLDER'] = './online/receive'
all_speaker = get_speakers('./online/oursounds')
label_enc = LabelEncoder()
label_enc.fit(all_speaker)
#%% server
#app = Flask(__name__)
def load_starGAN_model():
    # load pre-trained 好的 Keras model，這邊使用 ResNet50 和 ImageNet 資料集（你也可以使用自己的 model）
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
   

    global model
    global graph
    global sess
    global normlizer
    sess = tf.Session(config=config)
    model = StarGANVC(num_features=FEATURE_DIM, mode='test')
    model.load(filepath=os.path.join(model_dir, MODEL_NAME))
    # 初始化 tensorflow graph
    normlizer = Normalizer(input_dir=test_dir)
    graph = tf.get_default_graph()
def get_speaker(input_dir):


    modelpath = "Speakers_models/"
    gmm_files = [os.path.join(modelpath,fname) for fname in 
                  os.listdir(modelpath) if fname.endswith('.gmm')]
    models    = [pickle.load(open(fname,'rb'), encoding='latin1') for fname in gmm_files]
    speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
                  in gmm_files]
    
    sr,audio = read(input_dir)
    vector   = extract_features(audio,sr)

    log_likelihood = np.zeros(len(models)) 

    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    
    return speakers[winner]
@app.route('/', methods=['POST'])
def post_data_from_router():
    
    print(request.form['target'],file=sys.stderr)
    if not request.form['target']:
        return 'NO TARGET'
    target = request.form['target']
    f = request.files['file']
    f.save(os.path.join(receive_dir, secure_filename(f.filename)))
    source = get_speaker(receive_dir + '/'+ f.filename)
    print (source,file=sys.stderr)
    conversion(receive_dir + '/'+ f.filename, source, target)
    return 'POST_OK'
@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
def conversion(filepath, source, target):
    
    #tempfiles = glob.glob(p.replace('\\','/'))
    speaker = source
    _, __, name = filepath.rsplit('/', maxsplit=2)
    print (name)
    wav_, fs = librosa.load(filepath, sr=SAMPLE_RATE, mono=True, dtype=np.float64)
    wav, pad_length = pad_wav_to_get_fixed_frames(wav_, frames=FRAMES)

    f0, timeaxis = pyworld.harvest(wav, fs, f0_floor=71.0, f0_ceil=500.0)

    #CheapTrick harmonic spectral envelope estimation algorithm.
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs, fft_size=FFTSIZE)

    #D4C aperiodicity estimation algorithm.
    ap = pyworld.d4c(wav, f0, timeaxis, fs, fft_size=FFTSIZE)
    #feature reduction
    coded_sp = pyworld.code_spectral_envelope(sp, fs, FEATURE_DIM)

    coded_sps_mean = np.mean(coded_sp, axis=0, dtype=np.float64, keepdims=True)
    coded_sps_std = np.std(coded_sp, axis=0, dtype=np.float64, keepdims=True)
    #normalize
    # coded_sp = (coded_sp - coded_sps_mean) / coded_sps_std
    # print(coded_sp.shape, f0.shape, ap.shape)

    #one audio file to multiple slices(that's one_test_sample),every slice is an input
    one_test_sample = []
    csp_transpose = coded_sp.T  #36x512 36x128...
    for i in range(0, csp_transpose.shape[1] - FRAMES + 1, FRAMES):
        t = csp_transpose[:, i:i + FRAMES]
        #normalize t
        t = normlizer.forward_process(t, speaker)
        t = np.reshape(t, [t.shape[0], t.shape[1], 1])
        one_test_sample.append(t)
    # print(f'{len(one_test_sample)} slices appended!')

    #generate target label (one-hot vector)
    one_test_sample_label = np.zeros([len(one_test_sample), len(all_speaker)])
    temp_index = label_enc.transform([target])[0]
    one_test_sample_label[:, temp_index] = 1

    generated_results = model.test(one_test_sample, one_test_sample_label)

    reshpaped_res = []
    for one in generated_results:
        t = np.reshape(one, [one.shape[0], one.shape[1]])

        t = normlizer.backward_process(t, target)
        reshpaped_res.append(t)
    #collect the generated slices, and concate the array to be a whole representation of the whole audio
    c = []
    for one_slice in reshpaped_res:
        one_slice = np.ascontiguousarray(one_slice.T, dtype=np.float64)
        # one_slice = one_slice * coded_sps_std + coded_sps_mean

        # print(f'one_slice : {one_slice.shape}')
        decoded_sp = pyworld.decode_spectral_envelope(one_slice, SAMPLE_RATE, fft_size=FFTSIZE)
        # print(f'decoded_sp shape: {decoded_sp.shape}')
        c.append(decoded_sp)

    concated = np.concatenate((c), axis=0)
    # print(f'concated shape: {concated.shape}')
    #f0 convert
    f0 = normlizer.pitch_conversion(f0, speaker, target)

    synwav = pyworld.synthesize(f0, concated, ap, fs)
    # print(f'origin wav:{len(wav_)} paded wav:{len(wav)} synthesize wav:{len(synwav)}')

    #remove synthesized wav paded length
    synwav = synwav[:-pad_length]

    #save synthesized wav to file
    wavname = f'{speaker}-{target}+{name}'
    wavpath = f'{output_dir}/wavs'
    if not os.path.exists(wavpath):
        os.makedirs(wavpath, exist_ok=True)
    librosa.output.write_wav(f'{wavpath}/{wavname}', synwav, sr=fs)
    print (f'Save:{wavname}')
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"p:",["port"])
    except getopt.GetoptError:
        print ('Usage: server_example.py\n [-p <port>] Port\n')
        sys.exit(2)
    for opt, arg in opts:  
        if opt == '-p':
            port = arg
   # model = StarGANVC(num_features=FEATURE_DIM, mode='test')
   # model.load(filepath=os.path.join(model_dir, MODEL_NAME))
    load_starGAN_model()
   
    app.run(host='0.0.0.0', port=port, debug=True)