#!/usr/bin/env python

import numpy as np
import joblib
import network
import json
from sklearn.preprocessing import normalize
from get_12ECG_features import get_12ECG_features
MAX_LEN = 119808
def run_12ECG_classifier(data,header_data,classes,model):

    with open('net_classes.txt', 'r') as result_file:
        net_classes = result_file.read().splitlines()
    # net_classes = ['AF', 'AF,LBBB', 'AF,LBBB,STD', 'AF,PAC', 'AF,PVC', 'AF,RBBB', 'AF,STD', 'AF,STE', 'I-AVB', 'I-AVB,LBBB',
    #  'I-AVB,PAC', 'I-AVB,PVC', 'I-AVB,RBBB', 'I-AVB,STD', 'I-AVB,STE', 'LBBB', 'LBBB,PAC', 'LBBB,PVC', 'LBBB,STE',
    #  'Normal', 'PAC', 'PAC,PVC', 'PAC,STD', 'PAC,STE', 'PVC', 'PVC,STD', 'PVC,STE', 'RBBB', 'RBBB,PAC', 'RBBB,PAC,STE',
    #  'RBBB,PVC', 'RBBB,STD', 'RBBB,STE', 'STD', 'STD,STE', 'STE']

    num_classes = len(classes)
    class_to_int = dict(zip(classes, range(num_classes)))
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    # Use your classifier here to obtain a label and score for each class.
    input_length = MAX_LEN
    if data.T.shape[0]>input_length:
        # input_length = int(data.T.shape[0]/256+1)*256
        data = data[:,1:input_length]
    input_data = np.zeros([1,input_length,data.T.shape[1]])
    input_data[0,0:data.T.shape[0],:] = data.T
    score = model.predict(input_data)
    pred_scores = np.sum(score,axis=1)
    pred_scores = pred_scores[:, 0:-1]
    pred_scores = normalize(pred_scores)
    pred_label = pred_scores.argmax(axis=1)
    pred_c = net_classes[pred_label[0]]
    pred_c_split = pred_c.split(',');
    for i in range(len(pred_c_split)):
        current_label[class_to_int[pred_c_split[i]]] = 1
        current_score[class_to_int[pred_c_split[i]]] = np.max(pred_scores)
    pred_l = np.argwhere(pred_scores[0,:]>0.5)
    if pred_l.size != 0:
        for ii in range(len(pred_l)):
            pred_c = net_classes[pred_l[ii][0]]
            pred_c_split = pred_c.split(',')
            for j in range(len(pred_c_split)):
                current_label[class_to_int[pred_c_split[j]]] = 1
                current_score[class_to_int[pred_c_split[j]]] = pred_scores[0,pred_l[ii][0]]

    # for i in range(num_classes):
    #     current_score[class_to_int[pred_c_split[i]]] = np.max(pred_scores)

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    filename = 'net_weights.h5'
    config_file = 'config.json'
    params = json.load(open(config_file, 'r'))
    params.update({
        "input_shape": [MAX_LEN, 12]
    })
    loaded_model = network.build_network(**params)
    loaded_model.load_weights('net_weights.h5')

    return loaded_model
