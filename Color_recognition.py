from PIL import Image
import numpy as np
import os
from os import walk
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation

# Costruisce le immagini rosse e blu

directory = "/foo/Temp/"
folders = [directory+"rosso", directory+"blu"]
for dir in folders:
    if not os.path.exists(dir):
        os.makedirs(dir)

files = []
t = 0  # contatore
size = 10 # dimensione k, in pixel, di immagini kxk
iterations = range(0, 60) # costruisce 60 immagini
Y = []

for u in iterations:

    k1 = int(np.random.uniform(0, 100)) # Valori aleatori per introdurre variabilità nei colori
    k2 = int(np.random.uniform(0, 100))
    k3 = int(np.random.uniform(0, 100))
    k4 = int(np.random.uniform(0, 100))

    col = [["rosso/", (255, k1, k2)], ["blu/", (k3, k4, 255)]] # paletta colori

    for n,i in col:
        im = Image.new("RGB", (size, size))
        pix = im.load()

        for x in range(size):
            for y in range(size):
                pix[x, y] = i # setta il colore dei pixel
                
        filename = directory + n + str(t) + ".png"
        im.save(filename, "PNG")
        files.append(filename) # salva lista dei filenames
        Y.append(n[0:-1]) # Vettore variabile dipendente
        t += 1

# --------------------------------------------------------------------------------------- #

# Estrae le misure (features)

print("Extracting features")
def extract_features(path):

    ''' Extract features from image of MetroBank Clapham Junction: colours distributions, sdev of the colours, number
    of objects and several measurements on those objects.
    Input: path - string with path of the picture
    Output: a vector '''


    # Resizing to 1000x1000, pixel distribution for each colour channel according to a fixed number of bins
    im = Image.open(path)
    im_arr = np.array(im)

    hist_R = np.histogram(im_arr[:, :, 0].flatten()) # Distibuzione Rosso
    hist_G = np.histogram(im_arr[:, :, 1].flatten()) # Distribuzione Verde
    hist_B = np.histogram(im_arr[:, :, 2].flatten()) # Distribuzione Blu

    features = np.concatenate((hist_R[0], hist_G[0], hist_B[0], hist_R[1], hist_G[1], hist_B[1]), axis=0)

    features = np.reshape(features, (1, len(features)))
    return (features)

X = np.zeros((len(files), 63)) # ncol da modificare a seconda del'entrata
i = 0
for im in files:
    feat = extract_features(im)
    X[i, 0:len(feat[0])] = feat[0]
    i += 1
    print(i/len(files)*100, '%', ' done')


# -------------------------------------------------------------------------------------------- #

# Stima del modello: Random Forest

print("Training")
# n_estimators is the number of decision trees
# max_features also known as m_try is set to the default value of the square root of the number of features
clf = RF(n_estimators=100, n_jobs=3)
scores = cross_validation.cross_val_score(clf, X, Y, cv=5, n_jobs=1)
print("Accuracy of all classes")
print(np.mean(scores))
