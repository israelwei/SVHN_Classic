#Predict for test images for Computer Vision task - SVHN dataset, format 1
import h5py
import numpy as np
from scipy.misc.pilutil import imresize
import cv2
from skimage.feature import hog
import sys
from PIL import Image
import  matplotlib.pyplot as plt

DIGIT_WIDTH = 10
DIGIT_HEIGHT = 20
IMG_HEIGHT = 28
IMG_WIDTH = 28
CLASS_N = 10 # 0-9

def get_digits(contours, hierarchy):
	hierarchy = hierarchy[0]
	bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]
	final_bounding_rectangles = []
	# find the most common heirarchy level - that is where our digits's bounding boxes are
	u, indices = np.unique(hierarchy[:, -1], return_inverse=True)
	most_common_heirarchy = u[np.argmax(np.bincount(indices))]

	for r, hr in zip(bounding_rectangles, hierarchy):
		x, y, w, h = r
		# this could vary depending on the image you are trying to predict
		# we are trying to extract ONLY the rectangles with images in it (this is a very simple way to do it)
		# we use heirarchy to extract only the boxes that are in the same global level - to avoid digits inside other digits
		# ex: there could be a bounding box inside every 6,9,8 because of the loops in the number's appearence - we don't want that.
		# read more about it here: https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
		if ((w * h) > 250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy:
			final_bounding_rectangles.append(r)

	return final_bounding_rectangles


def pixels_to_hog_20(img_array):
    hog_featuresData = []
    for img in img_array:
        fd = hog(img,
                 orientations=10,
                 pixels_per_cell=(5,5),
                 cells_per_block=(1,1),
                 visualise=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)

def proc_user_img(img_file, model):
	print('loading "%s for digit recognition" ...' % img_file)
	im = cv2.imread(img_file)
	blank_image = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
	blank_image.fill(255)

	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	plt.imshow(imgray)
	kernel = np.ones((5, 5), np.uint8)

	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	thresh = cv2.erode(thresh, kernel, iterations=1)
	thresh = cv2.dilate(thresh, kernel, iterations=1)
	thresh = cv2.erode(thresh, kernel, iterations=1)

	_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	digits_rectangles = get_digits(contours, hierarchy)  # rectangles of bounding the digits in user image

	for rect in digits_rectangles:
		x, y, w, h = rect
		cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
		im_digit = imgray[y:y + h, x:x + w]
		im_digit = (255 - im_digit)
		im_digit = imresize(im_digit, (IMG_WIDTH, IMG_HEIGHT))

		hog_img_data = pixels_to_hog_20([im_digit])
		pred = model.predict(hog_img_data)
		cv2.putText(im, str(int(pred[0])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
		cv2.putText(blank_image, str(int(pred[0])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

	plt.imshow(im)
	cv2.imwrite("original_overlay.png", im)
	cv2.imwrite("final_digits.png", blank_image)
	cv2.destroyAllWindows()
