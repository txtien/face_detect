from flask import Blueprint, render_template, request, flash, redirect, url_for, send_from_directory, make_response, \
    Response
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import os
import numpy as np
import pickle
import dlib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer

upload_api = Blueprint('upload_api', __name__)

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}


face_model = load_model('/home/tien/Desktop/flask_doc/app/blueprints/upload_api/facenet_keras.h5', compile=False)

with open('/home/tien/Desktop/flask_doc/app/blueprints/upload_api/SVM_final.sav', 'rb') as f:
    clf = pickle.load(f)  # Load model

with open('/home/tien/Desktop/flask_doc/app/blueprints/upload_api/decoder.pickle', 'rb') as f:
    decode = pickle.load(f)  # Load model

detector = dlib.get_frontal_face_detector()
def get_face_boundary(frame, upsample = 0):
    rects = detector(frame, upsample)
    rects_list = []
    if len(rects) > 0:   
        for rect in rects:
            rects_list.append(rect)
        return rects_list
    else:
        return

def draw_rectangle_around_faces(frame, rect_list):
    for rect in rect_list:
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    return 

def crop_face(frame, rect_list):
    faces = []
    for rect in rect_list:
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        faces.append(frame[top: bottom, left: right])
    return faces

def standardize(pixels):
    mean, std = pixels.mean(), pixels.std()
    pixels_std = (pixels - mean) / std
    return pixels_std

def get_features(face):
    array = np.asarray(face, dtype=np.float32)
    array = cv2.resize(array, (160, 160))
    array = standardize(array)
    array = np.expand_dims(array, axis=0)
    yhat = face_model.predict(array)
    yhat = Normalizer().transform(yhat)
    return yhat

def preprocess_frame(faces, rect_list):
    x_test = np.zeros(shape=(len(faces), 128))
    i = 0
    for face in faces:
        features = get_features(face)
        x_test[i] = features

    
    name = clf.predict(x_test)
    name = decode.inverse_transform(name)
    proba = clf.predict_proba(x_test)
    proba = [proba[0, clf.predict(x_test)[0]] * 100]
    return zip(name, proba, rect_list)

text = ''
proba_str = ''
def run_facedetector_camera():
    cap = cv2.VideoCapture(0)

    i = 0
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # fontScale 
    fontScale = 1.1

    # Red color in BGR 
    color = (0, 0, 255) 

    # Line thickness of 2 px 
    thickness = 2
   
    while (True):
        # Capture frame-by-frame
        _, frame = cap.read()
        gray = frame
        if get_face_boundary(frame, upsample=0):
            rect_list = get_face_boundary(frame, upsample=0)
            draw_rectangle_around_faces(frame, rect_list)
            
            faces = crop_face(frame, rect_list)
            if i % 20 == 0:
                try:
                    for name, proba, rect in preprocess_frame(faces, rect_list):
                        left = rect.left()
                        right = rect.right()
                        bottom = rect.bottom()
                        if int(proba) > 75:
                            text = name 
                            proba_str = str(round(proba, 2))
                            cv2.putText(frame, name + ' ' + str(round(proba,2 )) + '%', ((left + right) // 2, bottom + 20), font, fontScale,  color, thickness, cv2.LINE_AA, False)
                        else:
                            cv2.putText(frame, 'Unknown', ((left + right) // 2, bottom + 20), font, fontScale,  color, thickness, cv2.LINE_AA, False)
                except:
                    pass
               
            else:
                try:
                    for rect in rect_list:
                        left = rect.left()
                        right = rect.right()
                        bottom = rect.bottom()
                        cv2.putText(frame, text + ' ' + proba_str, ((left + right) // 2, bottom + 20), font, fontScale,  color, thickness, cv2.LINE_AA, False)
                except:
                    pass
                


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', gray)[1].tostring() + b'\r\n')
           
        # Display the resulting frame
        # cv2.imshow('Image', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        i += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    return


@upload_api.route('/video_feed')
def video_feed():
    return Response(run_facedetector_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

