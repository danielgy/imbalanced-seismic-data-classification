# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:52:57 2018
@author: GY
"""
import h5py
import os
import time
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import roc_auc_score, confusion_matrix, auc, roc_curve, precision_recall_curve, f1_score, \
    recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, KFold
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler,ClusterCentroids,InstanceHardnessThreshold,NearMiss,TomekLinks,\
EditedNearestNeighbours,RepeatedEditedNearestNeighbours,AllKNN,OneSidedSelection,CondensedNearestNeighbour,NeighbourhoodCleaningRule
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.ensemble import EasyEnsemble,BalancedBaggingClassifier,BalanceCascade
import CNN_series
import LSTM_FCN
import FCN
import res_net
import MLP
import data_preprocess
import data_pre
import data_vis
import classify_eval


# Hyper-parameter
NET = MLP
Th = 0.5
OUTPUT_NODE = 1
LENGTH = 1
BATCH_SIZE = 512
TRANING_STEPS = 20000

# Define saver
model_dir = "saver"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

MODEL_SAVE_PATH = model_dir
MODEL_NAME = "model.ckpt"

# GPU config
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True


def model_train(train, valid, pos_weight, WIDTH):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None,
                                        LENGTH,
                                        WIDTH,
                                        NET.NUM_CHANNELS],
                           name='x-input')

        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        phase = tf.placeholder(tf.bool, name='phase')
        logit, y = NET.model(x, phase, WIDTH)

        global_step = tf.Variable(0, trainable=False)
        # loss
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logit, targets=y_, pos_weight=pos_weight)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean
        # accuracy
        correct_prediction = tf.equal(tf.greater(y, Th), tf.greater(y_, Th))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # optimizer
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            for i in range(TRANING_STEPS):
                xs, ys = train.next_batch(BATCH_SIZE)
                reshaped_xs = np.reshape(xs, [BATCH_SIZE,
                                              LENGTH,
                                              WIDTH,
                                              NET.NUM_CHANNELS])
                reshaped_ys = np.reshape(ys, [BATCH_SIZE, OUTPUT_NODE])

                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: reshaped_xs, y_: reshaped_ys, phase: 1})
                if i % 2000 == 0:
                    train_accuracy, train_logit = sess.run([accuracy, y], feed_dict={x: reshaped_xs,
                                                                                     y_: reshaped_ys,
                                                                                     phase: 1})
                    print('batch samples: {}'.format(Counter(ys)))
                    print("After %d training steps, loss %g, training accuracy %g" % (step, loss_value, train_accuracy))

                    a = train_logit.reshape([1, -1]) > Th
                    a = a.astype("int")
                    train_predict = a.reshape(-1)
                    cm = np.array(confusion_matrix(ys, train_predict))
                    tn, fn, fp, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                    print('cm[[tn  fn] [fp tp]]:', [[tn, fn], [fp, tp]])

            test_xs = np.reshape(valid.images, [-1,
                                                LENGTH,
                                                WIDTH,
                                                NET.NUM_CHANNELS])
            test_ys = np.reshape(valid.labels, [-1, OUTPUT_NODE])

            test_accuracy, test_logit = sess.run([accuracy, y], feed_dict={x: test_xs, y_: test_ys, phase: 1})
            a = test_logit.reshape([1, -1]) > Th
            a.astype("int")
            test_predict = a.reshape(-1)
            try:
                roc_score = roc_auc_score(valid.labels, test_predict)
            except ValueError:
                roc_score = 0
            print("-" * 75)
            print("After %d training steps, test accuracy %g" % (step, test_accuracy))

            # TODO:plot the confusion matrix
            target_names = ['normal', 'warning']
            cnf_matrix1 = confusion_matrix(valid.labels, test_predict)
            data_vis.plot_confusion_matrix(cnf_matrix1, classes=target_names,
                                           title='Confusion matrix')
            # TODO: plot the ROC Curves and AUC Score
            fpr, tpr, _ = roc_curve(valid.labels, test_logit, pos_label=1)
            auc_score = auc(fpr, tpr)
            fs = f1_score(valid.labels, test_predict)
            G_mean = np.sqrt(np.mean(tpr) * np.mean(fpr))
            data_vis.plot_roc_curve(fpr, tpr, auc_score)
            # TODO: plot Precision and Recall Curves
            precision, recall, _ = precision_recall_curve(valid.labels, test_logit, pos_label=1)
            auc_score_1 = auc(recall, precision)
            r = recall_score(valid.labels, test_predict)
            p = precision_score(valid.labels, test_predict)
            data_vis.plot_precision_recall_curve(recall, precision, auc_score_1)

            print("ROC AUC : %.10f" % auc_score)
            print("G_mean : %.10f" % G_mean)
            print("Sensitivity(TPR) : %.10f" % np.mean(tpr))
            print("Recall : %.10f" % r)
            print("Precision: %.10f" % p)
            print("F1 Score : %.10f" % fs)
            print("Specificity(TNR) : %.10f" % (1 - np.mean(fpr)))
            print("PR AUC : %.10f" % (auc_score_1))

            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

            print("testing!")
            reshaped_test = np.reshape(X_test, [3860,
                                          LENGTH,
                                          WIDTH,
                                          NET.NUM_CHANNELS])
            test_label = Y_test.reshape([3860,1])

            t_accuracy, t_logit = sess.run([accuracy, y],feed_dict={x: reshaped_test, y_: test_label, phase: 1})
            aa = t_logit.reshape([1, -1]) > Th
            aa.astype("int")
            t_predict = aa.reshape(-1)
            try:
                roc_score = roc_auc_score(Y_test, t_predict)
            except ValueError:
                roc_score = 0
            print("-" * 75)
            print("test accuracy %g" % t_accuracy)

            # TODO:plot the confusion matrix
            target_names = ['normal', 'warning']
            cnf_matrix2 = confusion_matrix(Y_test, t_predict)
            data_vis.plot_confusion_matrix(cnf_matrix2, classes=target_names,
                                           title='TEST Confusion matrix')
            # TODO: plot the ROC Curves and AUC Score
            fpr1, tpr1, _ = roc_curve(Y_test, t_logit, pos_label=1)
            auc_score1 = auc(fpr1, tpr1)
            fs1 = f1_score(Y_test, t_predict)
            G_mean1 = np.sqrt(np.mean(tpr1) * np.mean(fpr1))
            data_vis.plot_roc_curve(fpr1, tpr1, auc_score1)
            # TODO: plot Precision and Recall Curves
            precision1, recall1, _ = precision_recall_curve(Y_test, t_logit, pos_label=1)
            auc_score_11 = auc(recall1, precision1)
            r1 = recall_score(Y_test, t_predict)
            p1 = precision_score(Y_test, t_predict)
            data_vis.plot_precision_recall_curve(recall1, precision1, auc_score_11)

            print("TEST ROC AUC : %.10f" % auc_score1)
            print("TEST G_mean : %.10f" % G_mean1)
            print("TEST Sensitivity(TPR) : %.10f" % np.mean(tpr1))
            print("TEST Recall : %.10f" % r1)
            print("TEST Precision: %.10f" % p1)
            print("TEST F1 Score : %.10f" % fs1)
            print("TEST Specificity(TNR) : %.10f" % (1 - np.mean(fpr1)))
            print("TEST PR AUC : %.10f" % (auc_score_11))


if __name__ == "__main__":
    st = time.time()

    file = h5py.File('./Pre_combined.h5', 'r')
    data = file['train_data'][:]
    label = file['train_label'][:]
    X_test = file['test_data'][:]
    Y_test = file['test_label'][:]
    file.close()
    _, WIDTH = data.shape
    nb_folds = 10

    kfolds = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=None)
    kfolds.get_n_splits(data, label)
    st = time.time()
    cv = 0
    for train, valid in kfolds.split(data, label):
        st1 = time.time()
        cv += 1
        print("{} cross validation!".format(cv))
        X, y = np.array(data[train]), np.array(label[train])
        ada = SMOTE(random_state=42)
        X_res, y_res = ada.fit_sample(X, y)
        print('Resampled dataset shape {}'.format(Counter(y_res)))
        train_input = data_preprocess.DataSet(np.array(X_res), np.array(y_res))
        pw = np.sum(y_res == 0) / np.sum(y_res == 1)
        print('dataset shape {}'.format(Counter(y)))

        valid_input = data_preprocess.DataSet(np.array(data[valid]), np.array(label[valid]))
        print('positive weight:', pw)

        model_train(train_input, valid_input, pos_weight=pw, WIDTH=WIDTH)
        end1 = time.time()
        print("{} cross validation time spend is: {}s".format(cv, (end1-st1)))
        print("*" * 75)

    print("Total time spend is: {}s".format((time.time()-st)))
