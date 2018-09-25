import numpy as np
from skimage.feature import hog
import  matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn import svm, metrics
import itertools
import tables
from sklearn.externals import joblib

TRAIN_FILES_DIR = 'data/train'
TEST_FILES_DIR = 'data/test'
DIGITS_DIR = 'digits'
DIGIT_PATCHES_DIR = './classes/class_'

MAIN_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = MAIN_PATH + '\\' + TRAIN_FILES_DIR
TARIN_H5_PATH = MAIN_PATH + '/data/train_set.h5'
PATCH_NAME = "Patches"
CROP_SIZE = 16

DIGIT_WIDTH = 10
DIGIT_HEIGHT = 20
IMG_HEIGHT = 28
IMG_WIDTH = 28
CLASS_N = 10 # 0-9

def pixels_to_hog_20(img_array):
    """
    Creates a feature vector for each image in the array, using the Histogram of Gradients method.
    :param img_array: Array of images to get their feature vector
    :return: The feature vector extracted
    """
    hog_featuresData = []
    for img in img_array:
        # Getting for each image the feature vector using the Histogram of Gradients method:
        fd = hog(img,
                 orientations=10,
                 pixels_per_cell=(5,5),
                 cells_per_block=(1,1),
                 visualise=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)




def load_data():
    """
    This method loads the data from the h5py file saved before.
    Gets the array and the corresponding label from the h5py file for all images in the "train" directory,
    and appends to the x_data and y_data accordingly.
    :return:
    """
    x_data = np.empty((0, CROP_SIZE, CROP_SIZE))
    y_data = np.empty((0))
    for i in range(CLASS_N):
        cur_h5_filename = DIGIT_PATCHES_DIR + str(i) + '.h5'
        cur_hf = tables.open_file(cur_h5_filename, 'r')
        cur_x_data = getattr(cur_hf.root, PATCH_NAME)[0:None]  # Extracting all patch names of sick samples
        cur_x_data = np.array(np.split(cur_x_data, cur_x_data.shape[0] / CROP_SIZE))
        num_cur_samples = cur_x_data.shape[0]
        cur_y_data = np.ones(num_cur_samples)*i
        x_data = np.append(x_data, cur_x_data, axis=0)
        y_data = np.append(y_data, cur_y_data, axis=0)

    return x_data, y_data




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')






def train_model(model_string):
    """
    This function runs the file- loads the data, splits the training data for test data (more a validation data here
    for the classifier), for evaluating the classifier before applying it to new test data (which will be taken from
    the "extra" directory).
    Also plots the confusion matrix for visualization.
    :param model_string: the classifier method- "svm" for SVM or "knn" for K-nearest neighbors
    """

    x_data, y_data = load_data()
    y_data = np.reshape(y_data, newshape=(y_data.shape[0], 1))
    digits, labels, fixed_weights = shuffle(x_data, y_data, random_state=256)
    train_digits_data = pixels_to_hog_20(digits)
    X_train, X_test, y_train, y_test = train_test_split(train_digits_data, labels, shuffle=True, test_size=0.1)

    # ------------------training and testing----------------------------------------

    if model_string == "svm":
        print('training SVM...')  # gets 80% in most user images
        # Create a classifier: a support vector classifier
        classifier = svm.SVC(gamma=0.001)
    elif model_string == "knn":
        print('training knn...')
        classifier = KNeighborsClassifier(n_neighbors=31, weights='distance')


    # fit to the trainin data
    classifier.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(classifier, model_string + '_svhn.pkl')
    # now to Now predict the value of the digit on the test data
    y_pred = classifier.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, y_pred)))

    # Compute confusion matrix and plot it:
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    np.savetxt("conf_mtx.csv", cnf_matrix, delimiter=",")

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[i for i in range(CLASS_N)],
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[i for i in range(CLASS_N)], normalize=True,
                          title='Normalized confusion matrix')
    plt.show()


# K nearest neighbors "won" against SVM on my experiments when I checked the performance of the confusion matrix,
#  so I chose KNN as the classifier
model_str = "knn"
train_model(model_str)