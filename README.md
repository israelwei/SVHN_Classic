# SVHN_Classic

The challenge: implement a good classifier for the SVHN dataset without a neural network.
The solution: detect the digits in images using MSER features and Strokw Width variation,
		  classify the digits in each bounding box using the K-Nearest Neighbors method.
		  I achieved an average F1-score of 0.8 on images of the validation set. 

![20109](https://user-images.githubusercontent.com/23454156/46013156-156f1000-c0d4-11e8-8220-71a7240a3ab1.png)
![23475](https://user-images.githubusercontent.com/23454156/46013180-2881e000-c0d4-11e8-8b92-ce9d02cf194f.png)
![26148](https://user-images.githubusercontent.com/23454156/46013197-36376580-c0d4-11e8-8d36-007496ed26df.png)


Order of running files:
1) detect_text.m via Matlab: for extracting the bounding boxes for each digit in each image,
                             using image processing techniques such as MSER feature detection and
			     Stroke Width variance. The images that are used for this file are the images
			     extracted from the "extra" tar dataset (these are more suitable for my assumptions
			     for the algorithm).


2) convert_to_h5py.py via Python 3: for converting the data in the digitStruct.mat for the training set,
				    which is the data from the "train" tar.


3) train_svhn.py via Python 3: for training the classifier and saving the model. Best model was chosen according
				to the confusion matrix results on validation set 
				(I chose k-nearest neighbors with k=31). Best F1-score I got on the validation set
				was 80% percent on average for all digits. I attach the model I saved "knn_svhn.pkl".


4) predict_svhn.py via Python 3: for inference on the test set, which are the images from the "extra" tar dataset,
				 where we predict on each cropped digit. The cropping is done by using the bounding
				 boxes we extracted using the matlab code ( detect_text.m ).
