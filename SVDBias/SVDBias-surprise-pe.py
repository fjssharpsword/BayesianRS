# coding:utf-8  
'''
@author: Jason.F
@data: 2019.07.15
@function: Implementing SVDBias with surprise lirbray
           Dataset: Pinterest-20
           Evaluating by hitradio,ndcg
'''
import numpy as np
import pandas as pd
import math
from collections import defaultdict
import heapq
import surprise as sp

#1. loading the dataset.
def load_dataset():
    train_data = pd.read_csv("./data/pinterest-20.train.rating", \
                             sep='\t', header=None, names=['user', 'item', 'rating'], \
                             usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2 :np.float})
    
    test_data = []
    with open("./data/pinterest-20.test.negative", 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1], 1])#one postive item
            for i in arr[1:]:
                test_data.append([u, int(i), 0]) #99 negative items
            line = fd.readline()
    return train_data, test_data

train_data,test_set = load_dataset()
#2. Transforming into data format of surprise and spliting the train-set and test-set
# The columns must correspond to user id, item id and ratings (in that order).
reader = sp.Reader(rating_scale=(0, 1))
spdata = sp.Dataset.load_from_df(train_data,reader)
trainset = spdata.build_full_trainset()
#testset = np.array(testset).tolist()

#3.training and evaluating 
def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
print ("%3s%20s%20s" % ('K','HR@10', 'NDCG@10'))
for K in [8,16,32,64]:#iterations epoches
    algo = sp.SVD(n_factors=K, n_epochs=20, lr_all=0.001, reg_all=0.01 )#NMF,SVDpp
    algo.fit(trainset)
    #print (algo.predict(str(1),str(1), r_ui=0, verbose=True)) 
    predictions = algo.test(test_set)#testset include one positive and 99 negtive sample of every user.
    user_iid_true_est = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_iid_true_est[uid].append((iid, true_r, est))
    hits = []
    ndcgs = []
    for uid, iid_ratings in user_iid_true_est.items():
        # Sort user ratings by estimated value
        #iid_ratings.sort(key=lambda x: x[2], reverse=True) #sorted by est
        scorelist = []
        positem = -1
        for iid, ture_r, est in iid_ratings:
            if positem == -1: positem=iid #one positive item in first
            scorelist.append([iid,est])
        map_item_score = {}
        for item, rate in scorelist: #turn dict
            map_item_score[item] = rate
        ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)#default Topn=10
        hr = getHitRatio(ranklist, positem)
        hits.append(hr)
        ndcg = getNDCG(ranklist, positem)
        ndcgs.append(ndcg)
    hitratio,ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print ("%3d%20.6f%20.6f" % (K, hitratio, ndcg))
'''
nohup python -u SVDBias-surprise-pe > svdbias-surprise-pe.log  &
 K               HR@10             NDCG@10
  8            0.496965            0.496965
 16            0.500299            0.500299
 32            0.495533            0.495533
 64            0.497599            0.497599
    
'''