# imbalanced-seismic-data-classification
- [data.rar](https://github.com/danielgy/imbalanced-seismic-data-classification/blob/master/data.rar): The data sets were downloaded from the data mining competion: [AAIA'16 Data Mining Challenge: Predicting Dangerous Seismic Events in Active Coal Mines](https://knowledgepit.fedcsis.org/contest/view.php?id=112) 
- [seismic data preprocessing.ipynb](https://github.com/danielgy/imbalanced-seismic-data-classification-/blob/master/seismic%20data%20preprocessing.ipynb) 
- [dara_pre.py](https://github.com/danielgy/imbalanced-seismic-data-classification-/blob/master/data_pre.py)
- [model.py](https://github.com/danielgy/imbalanced-seismic-data-classification-/blob/master/model.py)
- [feature combined.ipynb](https://github.com/danielgy/imbalanced-seismic-data-classification-/blob/master/feature%20combined.ipynb)
- [MLP.py](https://github.com/danielgy/imbalanced-seismic-data-classification-/blob/master/MLP.py): Multilayer Perceptron model
- [CNN_series.py](https://github.com/danielgy/imbalanced-seismic-data-classification/blob/master/CNN_series.py): Convolutional Neural Network as the classifer.
- [FCN.py](https://github.com/danielgy/imbalanced-seismic-data-classification/blob/master/FCN.py): Fully Convolutional Neural Network as the classifer.
- [res_net.py](https://github.com/danielgy/imbalanced-seismic-data-classification/blob/master/res_net.py): Residual Network as the classifer.
- [LSTM_FCN.py]()
- [classifier_train.py](https://github.com/danielgy/imbalanced-seismic-data-classification-/blob/master/classify_train.py): train and test the classifer with the processed and filed data sets. For rebalance the training samples, [SMOTE(Synthetic Minority Over-sampling Technique)](https://www.jair.org/media/953/live-953-2037-jair.pdf) was applied and 10-fold cross validation was used during the training 
period. Change the "NET", such as MLP, CNN_series or FCN, you can use different models to do the imbalanced data classification. The messures are confusion matrix, ROC AUC, G-mean, F1 score and so on. 
- [classify_train_without_features.py](https://github.com/danielgy/imbalanced-seismic-data-classification/blob/master/classify_train_without_features.py): Same with [classifier_train.py](https://github.com/danielgy/imbalanced-seismic-data-classification-/blob/master/classify_train.py), but the input data samples without features.
