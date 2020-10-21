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
