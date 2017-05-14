from PIL import Image
from os import walk
import shutil
from pylab import *
from scipy.ndimage import measurements,morphology
from skimage import measure
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation


path_failure = '/Users/StarShipIV/Google_Drive/Progetti/Machine_Vision/Devotion_pictures/Failure'
path_success = '/Users/StarShipIV/Google_Drive/Progetti/Machine_Vision/Devotion_pictures/Success'

# ------------------------------------------------------------------------------------------------------------------- #
# Preparing the dataset: augmenting through simple identity multiplication. The rationale is that pictures of the
# 'Devotion' logo will be always equal and with very low variation and noise.

for (dirpath, dirnames, filenames) in walk(path_failure):
    for f in filenames[1:len(filenames)]:
        for i in range(0, 2):
            save_name = dirpath + '/' + str(i) + '_' + f
            shutil.copy(dirpath + '/' + f, save_name)

for (dirpath, dirnames, filenames) in walk(path_success):
    for f in filenames[1:len(filenames)]:
        for i in range(0, 2):
            save_name = dirpath + '/' + str(i) + '_' + f
            shutil.copy(dirpath + '/' + f, save_name)

# ------------------------------------------------------------------------------------------------------------------- #
# Create list of filenames

files_fail = []
for (dirpath, dirnames, filenames) in walk(path_failure):
    for f in filenames[1:len(filenames)]:
        files_fail.append(dirpath + '/' + f)

files_success = []
for (dirpath, dirnames, filenames) in walk(path_success):
    for f in filenames[1:len(filenames)]:
        files_success.append(dirpath + '/' + f)

files = files_fail + files_success
# ------------------------------------------------------------------------------------------------------------------- #

def extract_features(path):

    ''' Extract features from image of DEVOTION: colours distributions, sdev of the colours, number of objects and
    several measurements on those objects.
    Input: path - string with path of the picture
    Output: a vector '''


    # Resizing to 1000x1000, pixel distribution for each colour channel according to a fixed number of bins
    im = Image.open(path)
    im_grey = im.convert('L')
    im = im.resize((1000, 1000))
    im_arr = np.array(im)

    hist_R = hist(im_arr[:, :, 0].flatten())
    hist_G = hist(im_arr[:, :, 1].flatten())
    hist_B = hist(im_arr[:, :, 2].flatten())

    # Std dev of colours of the whole image: very slow!
    std_dev_R = std(im_arr[:, :, 0])
    std_dev_G = std(im_arr[:, :, 1])
    std_dev_B = std(im_arr[:, :, 2])

    # Morphology: counting the objects
    im_bin = 1 * (np.array(im_grey) < 128)
    im_open = morphology.binary_opening(im_bin, ones((3, 3)), iterations=2)
    labels, nbr_objs = measurements.label(im_open)

    # Getting the measures of the 4 objects: filled area, axes lengths, ellipse, centroid, perimeter, Hu moments
    meas = measure.regionprops(labels)

    if nbr_objs >= 4:
        areas = np.array([meas[obj].filled_area for obj in range(0, 4)])
        axes_minor = np.array([meas[obj].minor_axis_length for obj in range(0, 4)])
        axes_maior = np.array([meas[obj].major_axis_length for obj in range(0, 4)])
        eccentricity = np.array([meas[obj].eccentricity for obj in range(0, 4)])
        centroids = np.array([meas[obj].centroid for obj in range(0, 4)]).flatten()
        perimeters = np.array([meas[obj].perimeter for obj in range(0, 4)])

    else:
        areas = array([meas[obj].filled_area for obj in range(0, nbr_objs)])
        axes_minor = array([meas[obj].minor_axis_length for obj in range(0, nbr_objs)])
        axes_maior = array([meas[obj].major_axis_length for obj in range(0, nbr_objs)])
        eccentricity = array([meas[obj].eccentricity for obj in range(0, nbr_objs)])
        centroids = array([meas[obj].centroid for obj in range(0, nbr_objs)]).flatten()
        perimeters = array([meas[obj].perimeter for obj in range(0, nbr_objs)])

    features = np.concatenate(
        (hist_R[0], hist_G[0], hist_B[0], np.array([std_dev_R, std_dev_G, std_dev_B, nbr_objs]), areas,
         axes_minor, axes_maior, eccentricity, centroids, perimeters), axis=0)

    features = reshape(features, (1, len(features)))
    return (features)


# Estrae le features per ogni immagine. Costrutto orrendo ma funziona.
X = np.zeros((len(files), 62))
i = 0
for im in files:
    feat = extract_features(im)
    X[i, 0:len(feat[0])] = feat[0]
    i += 1
    print((i / len(files)) * 100, "% done")

y = [0]*len(files_fail) + [1]*len(files_success)

print("Training")
# n_estimators is the number of decision trees
# max_features also known as m_try is set to the default value of the square root of the number of features
clf = RF(n_estimators=100, n_jobs=3)
scores = cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=1)
print("Accuracy of all classes")
print(np.mean(scores))


# Test su nuova foto Devotion
clf.fit(X,y)

x_test_path = '/Users/StarShipIV/Desktop/IMG_1354.jpg'
X_test = extract_features(x_test_path)
y_hat = clf.predict(X_test)