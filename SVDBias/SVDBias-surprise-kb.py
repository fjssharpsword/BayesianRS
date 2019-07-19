# coding:utf-8  
'''
@author: Jason.F
@data: 2019.07.13
@function: Implementing SVDBias with surprise lirbray
           Dataset: KnowledgeBase-cc 
           Evaluating by hitradio,ndcg
'''
import numpy as np
import pandas as pd
import math
from collections import defaultdict
import heapq
import surprise as sp

#1. loading the KnowledgeBase dataset.
trainset = pd.read_csv("./data/kbcc_trainset.csv", sep='|', low_memory=False)
trainset['num']=trainset['num'].apply(lambda x: 1 if float(x)>0.0 else 0)
print ('Trainset shape is:%d rows and %d columns'%(trainset.shape[0],trainset.shape[1]))
#testset includes 100 items for every user, one item is positive and other 99 is negtive items.
testset = pd.read_csv("./data/ctKngBase/kbcc_testset.csv", sep='|', low_memory=False)
testset['num']=testset['num'].apply(lambda x: 1 if float(x)>0.0 else 0)
print ('Testset shape is:%d rows and %d columns'%(testset.shape[0],testset.shape[1]))
csrNum = trainset['csr'].max()+1
keNum = trainset['ke'].max()+1
print('Dataset Statistics: Interaction = %d, User = %d, Item = %d, Sparsity = %.4f' % 
      (trainset.shape[0], csrNum, keNum, trainset.shape[0]/(csrNum*keNum)) )

#2. Transforming into data format of surprise and spliting the train-set and test-set
# The columns must correspond to user id, item id and ratings (in that order).
reader = sp.Reader(rating_scale=(0, 1))
spdata = sp.Dataset.load_from_df(trainset,reader)
trainset = spdata.build_full_trainset()
testset = np.array(testset).tolist()

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
    predictions = algo.test(testset)#testset include one positive and 99 negtive sample of every user.
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
            '''
            if (ture_r+1)>0.0:#one positive item
                scorelist.append([iid,est])
                positem = iid 
            else:# 99 negative items
                scorelist.append([iid,est])
            '''
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
nohup python -u SVDBias-surprise-kb > svdbias-surprise-kb.log  &
Trainset shape is:2547452 rows and 3 columns
Testset shape is:1021600 rows and 3 columns
Dataset Statistics: Interaction = 2547452, User = 10216, Item = 96324, Sparsity = 0.0026
  K               HR@10             NDCG@10
  8            0.500098            0.500098
 16            0.490407            0.490407
 32            0.501077            0.501077
 64            0.498727            0.498727
    
'''