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