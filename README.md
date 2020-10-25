# Covid Chest Xray CNN Classifier

## Dataset Information:

Open source public dataset of COVID-19 and Normal patient chest X-ray dataset have been used to train the model.

#### COVID-19 positive patients dataset has been collected from below GitHub repository

- [COVID-19 Positive patient's chest X-ray Data](https://github.com/ieee8023/covid-chestxray-dataset)


This dataset has chest X-ray & CT scan images of patients which are positive or suspected of COVID-19 or other viral and bacterial pneumonias (MERS, SARS, and ARDS.). This data is further preprocessed and only the posteroanterior (PA) view X-ray images of COVID-19 are seleted which are around 206 images.


#### Normal patient’s chest X-rays dataset has been collected from Kaggle. 

- [Healthy patient's chest X-ray Data](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

It has Normal and Pneumonia patient chest X-rays data. From that only healthy patiet's 206 Normal chest X-rays image are collected to make sure dataset are balanced with COVID-19 data.


## Data Preprocessing

- [Data Preprocessing](https://github.com/parekhjigar/covid-chestxray-cnn/blob/master/Data_preprocessing.ipynb)

## CNN Model
- [Covid Classifier CNN](https://github.com/parekhjigar/covid-chestxray-cnn/blob/master/covid_classifier_cnn.ipynb)


## A) Dataset Collection

The dataset used for the following study is obtained from two different open source repositories. The first one is a collection of the chest X-ray and CT images of the patients which are suspected of positive Covid-19 or any other pneumonias including (ARDS, SARS and MERS) from a Github repository belonging to Dr. Joseph Cohen [(Link)](https://github.com/ieee8023/covid-chestxray-dataset). This data is further pre-processed to get X-ray images of only Covid-19 positive cases with Posterior-Anterior (PA), which is basically a front-view of an X-ray. This summed up to 206 images for positive cases. Another set of data for the negative case of Covid-19, i.e. Normal X-ray images are collected from a Kaggle’s repository belonging to Paul Mooney [(Link)](https://www.kaggle.com/paultimothymooney/chest-xray-). It basically consists of X-ray images of pneumonia and normal cases. As of now, there are very less datasets of positive Covid-19 cases available as compared to that of pneumonia and normal cases. So, to avoid overfitting and balance the data for both the classes only 206 images of normal X-ray are collected. The images are further down-sampled to a resolution of 224x224 pixels to make the dimensions of the dataset consistent.

## B) Data Normalization

It is very essential to perform data normalization while building an image classifier. It makes sure that each input pixel has a similar distribution of data. Moreover, it is ideally better to have all the images with same aspect ratio. As I want all the pixel numbers to be positive, so I will scale it between the range 0 to 255 by dividing every image data by 255. This will further make the convergence faster while training the network, as the images of X-ray are of higher resolution.

## C) Splitting the data

The dataset is then split into training, validation and testing set respectively. The training set is used to train the actual model. So, 75% of the data is used for training. The Validation set is used to give us biased evaluation of our trained model. So, 15% of data is used for validation. Meanwhile, the test set is unseen by the model while training, so it is used for the final evaluation of the model. So, 10% data is used for testing. This splitting of data is performed using train_test_split of sklearn library. Figure 2. shows the data exploration after split.

## D) Data Augmentation

As the dataset available for positive Covid-19 cases is limited, there is a need to augment this data to increase the number of samples in our dataset as well as keep the data balanced. This can be virtually achieved by adding various geometric transformations, changes in colour channels by adjusting the brightness and contrast. So here I applied various transformations to the images randomly by using the Keras package ImageDataGenerator while training the model. The transformation included horizontal flips, shearing the edges of images with a range of 0.2, scaling the image using a random scaling factor between .9 and 1.1, rotating the image using a random angle between -15 and 15. I did not prefer to do vertical flips and cropping as it might distract the features required to train the model. So, overall by augmenting the data it improved the generalization and helped to prevent overfitting by providing the model ample amount of data to train efficiently.

## E. Parameter Tuning

Firstly, the X-ray images are needed to be rescaled to 244x244 pixels and then the dataset is split into 75% training, 15% validation and 10% testing set randomly. The batch size can be set anywhere between 32 to 128 depending on the model’s performances. I would recommend a batch size of 128 so as to get the best results in this case. Here, we can train the model by having five blocks of convolutions each having a convolutional layer along with max-pooling and batch normalization and Relu as the activation function. ReLU is used as the activation function mainly because it clips the negative values and keeps the positive values unchanged increasing the non-linearity of the model. ReLUs usually train several times faster than their equivalent tanh units according to the statistical demonstration made by the authors of Alexnet. After this, Max pooling layer with stride size = 2 can be applied which helps to reduce the spatial size to reduce the computation and parameters in the network. Now, one more layer of each is added again and flatten output is passed on to two fully connected layers. Now, the last layer should have a Sigmoid function as there are only two classes to be identified, i.e. Covid or Normal making it a binary classification model. The model should be trained for 10 epochs with a small learning rate. The learning rate is the most essential hyperparameter while training a CNN. Large learning rates can result in an unstable training, so 0.01 is an ideal learning rate in this case. For optimization, Adam algorithm can be used, which is an adaptive learning rate optimizer, and cross- entropy for the loss. Further, a Model Checkpoint can be used in this case which will save the copy of the best performing model. Then early stopping should be implemented so as to avoid the model from overfitting whenever we observe that the gap between the training and validation loss is increasing.

## G. Performance Evaluation

In this binary classification problem, the confusion matrix would be the best to evaluate the accuracy along with the other evaluation metrics like recall, specificity, sensitivity, precision and F1-score using the statistical indices- true positive (TP), true negative (TN), false positive (FP) and false negative (FN).
Here, the TP are the ones which are affected Covid-19 and are predicted as Covid-19 itself. The TN are the ones which are normal and are predicted as normal. The FP are the ones which are affected with Covid-19 yet predicted as normal. The FN are the ones which are normal yet predicted as Covid-19 case.
