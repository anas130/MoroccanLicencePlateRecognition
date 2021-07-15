#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
from werkzeug.utils import secure_filename
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
import cv2
import matplotlib.gridspec as gridspec
from os.path import splitext, basename
from keras.models import model_from_json
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from local_utils import detect_lp
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
K.set_learning_phase(0)
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ehtp = os.path.join('static', 'css')
app.config['EHTP_FOLDER'] = ehtp



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

#Fonction pour assurer le prÃ©-traitement des images
def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_path):
    Dmax = 608
    Dmin = 288
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

#Fonction pour extraire la postion de la lettre arabe
def arab_character_postion(distances):
  sorted_distances=[]
  sorted_distances=sorted(distances,reverse=True)
  max1=sorted_distances[0]
  max2=sorted_distances[1]
  for i in range(len(distances)):
    if distances[i]==max1:
      return i+1
    if distances[i]==max2:
      return i+1
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

#Dictionnaire des lettres arabes    
dictio_lettres={'a':'أ','b':'ب','d':'د','h':'ه','w':'و'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    ehtpim = os.path.join(app.config['EHTP_FOLDER'], 'LogoEHTP.jpg')
    return render_template('index.html', ehtp_image = ehtpim)

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)
        try:
            vehicle, LpImg, cor = get_plate(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
            # cv2.imwrite((os.path.join(app.config['UPLOAD_FOLDER'],filename)),plate_image)
        except:
            print("No plate was detected..")
            return render_template('index.html', filename=filename,prediction_text='No plate was detected..')
        
        if (len(LpImg)): #Vérifier si qu'une plaque au moins est bien détectée

            # Convertir le résultat sur un échelle de 8 bits
            plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
            
            # Convertir l'image en un gradient de couleur grise
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
            # Appliquer un lissage gaussien
            blur = cv2.GaussianBlur(gray,(7,7),0)
            
            # Appliquer un treshold de 180
            binary = cv2.threshold(blur, 180, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        
        cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Créer une copie de l'image pour dessiner les contours
        test_plat = plate_image.copy()
        
        # Initialiser une liste qui va contenir les contours des caractères
        crop_characters = []
        
        # définir la longeur et largeur standarisées des caractères et initialisation des variables
        digit_w, digit_h = 30, 60
        (x_old, y_old, w_old, h_old)=(0,0,0,0)
        iter_1=True
        
        #Liste des distances entre caractères
        distances_between=[]
        
        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
        
        
            # Ne selectionner que les contours avec le ratio défini
            if 1<=ratio<=5.5:
        
            # Ne selectionner que les contours dont la longeur occupe 20% de la largeur de l'image de la plaque
                if h/plate_image.shape[0]>=0.2:
        
                    if iter_1:
                        (x_old, y_old, w_old, h_old)=(x, y, w, h)
                        iter_1=False
                    else:
                        distances_between.append(abs(x_old+w_old-x))
                        (x_old, y_old, w_old, h_old)=(x, y, w, h)
        
        
                    # Dessiner les rectangles autour des caractères détectés
                    cv2.rectangle(test_plat, (x, y), (x + w, y + h), (0, 255,0), 2)
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)
                    
        json_file = open('MobileNets_character_recognition.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_chiffres = model_from_json(loaded_model_json)
        model_chiffres.load_weights("License_character_recognition.h5")
        labels = LabelEncoder()
        labels.classes_ = np.load('license_character_classes.npy')
        
        json_file_lettres = open('MobileNets_character_recognition_lettres.json', 'r')
        loaded_model_json_lettres = json_file_lettres.read()
        json_file_lettres.close()
        model_lettres = model_from_json(loaded_model_json_lettres)
        model_lettres.load_weights("License_character_recognition_lettres.h5")
        labels_lettres = LabelEncoder()
        labels_lettres.classes_ = np.load('license_character_classes_lettres.npy')
        
        pos=arab_character_postion(distances_between)
        fig = plt.figure(figsize=(15,3))
        cols = len(crop_characters)
        grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)
        det = []
        #Visualiser les résultats des prédictions
        for i,character in enumerate(crop_characters):
            fig.add_subplot(grid[i])
            if i==pos:
                caracter = np.array2string(predict_from_model(character,model_lettres,labels_lettres))
                title=dictio_lettres[caracter.strip("'[]")]
                print(title)
                det.append((i,caracter.strip("'[]")))
            else:
                title = np.array2string(predict_from_model(character,model_chiffres,labels)).strip("'[]")
                print(title)
                det.append((i,title))
            plt.title('{}'.format(title.strip("'[]"),fontsize=20))
            plt.axis(False)
            plt.imshow(character,cmap='gray')
        # im_rgb = cv2.cvtColor(LpImg[0]*255, cv2.COLOR_RGB2BGR)
        new_file = filename.split('.')[0]+'plt.'+filename.split('.')[1]
        cv2.imwrite((os.path.join(app.config['UPLOAD_FOLDER'],new_file)),LpImg[0]*255)
        # sorte = sorted(det, key=lambda tup: tup[0],reverse=True)
        prediction_text = "Le modèle donne la prédiction suivante: "+' '.join([i[1] for i in det])
        # flash('Image successfully uploaded and displayed below')
        ehtpim = os.path.join(app.config['EHTP_FOLDER'], 'LogoEHTP.jpg')
        return render_template('index.html', filename=new_file,prediction_text=prediction_text, ehtp_image=ehtpim)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()
