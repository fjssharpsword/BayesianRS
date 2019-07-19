# -*- Encoding:UTF-8 -*-
'''
@author: Jason.F
@data: 2019.07.17
@function: Implementing DMF with Tensorflow  
           Dataset: Pinterest-20 
           Evaluating: hitradio,ndcg
           https://www.ijcai.org/proceedings/2017/0447.pdf
           https://github.com/RuidongZ/Deep_Matrix_Factorization_Models
'''
import tensorflow as tf
import numpy as np
import argparse
import os
import heapq
import math
import sys

class DMF:
    def __init__(self, K, negNum=4, lr=0.001, maxEpochs=20, topK=10):
        #prepare data
        self.dataSet = DataSet()
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate

        self.train = self.dataSet.train
        self.testNeg = self.dataSet.getTestNeg()
        
        #initiate model
        self.negNum = negNum
        self.add_embedding_matrix()
        self.add_placeholders()

        self.userLayer = [512, K]
        self.itemLayer = [512, K]
        self.add_model()

        self.add_loss()

        self.lr = lr
        self.add_train_step()
        self.init_sess()

        self.maxEpochs = maxEpochs
        self.batchSize = 256
        self.topK = topK
        self.earlyStop = 5


    def add_placeholders(self):
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        self.matrix_init = tf.placeholder(tf.float32, shape=(self.shape[0], self.shape[1]))
        matrix = tf.Variable(self.matrix_init)
        self.user_item_embedding = tf.convert_to_tensor(matrix)
        #self.user_item_embedding = tf.convert_to_tensor(self.dataSet.getEmbedding())
        self.item_user_embedding = tf.transpose(self.user_item_embedding)

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer)-1):
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.itemLayer)-1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keepdims=False) / (norm_item_output* norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)

    def add_loss(self):
        regRate = self.rate / self.maxRate
        losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
        loss = -tf.reduce_sum(losses)
        # regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # self.loss = loss + self.reg * regLoss
        self.loss = loss

    def add_train_step(self):
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, global_step,
                                             self.decay_steps, self.decay_rate, staircase=True)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)

    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        #self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.global_variables_initializer(), feed_dict={self.matrix_init: self.dataSet.getEmbedding()})

    def run(self):
        best_hr = -1
        best_NDCG = -1
        best_epoch = -1
        print("Start Training!")
        for epoch in range(self.maxEpochs):
            print("="*20+"Epoch ", epoch, "="*20)
            self.run_epoch(self.sess)
            print('='*50)
            print("Start Evaluation!")
            hr, NDCG = self.evaluate(self.sess, self.topK)
            print("Epoch ", epoch, "HR: {}, NDCG: {}".format(hr, NDCG))
            if hr > best_hr or NDCG > best_NDCG:
                best_hr = hr
                best_NDCG = NDCG
                best_epoch = epoch
            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")
                break
            print("="*20+"Epoch ", epoch, "End"+"="*20)
        print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
        print("Training complete!")

    def run_epoch(self, sess, verbose=10):
        train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum)
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1

        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]
            
            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            losses.append(tmp_loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(i, num_batches, np.mean(losses[-verbose:])))
                sys.stdout.flush()
        loss = np.mean(losses)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss

    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,
                self.drop: drop}

    def evaluate(self, sess, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0


        hr =[]
        NDCG = []
        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        for i in range(len(testUser)):
            target = testItem[i][0]
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = sess.run(self.y_, feed_dict=feed_dict)

            item_score_dict = {}

            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(hr), np.mean(NDCG)

class DataSet(object):
    
    def __init__(self):
        self.train, self.shape = self.getTrainData()
        self.trainDict = self.getTrainDict()
        
    def getTrainData(self):
        data = []
        filePath = '/data/fjsdata/ctKngBase/ml/pinterest-20.train.rating'
        u = 0
        i = 0
        maxr = 0.0
        with open(filePath, 'r') as f:
            for line in f:
                if line:
                    lines = line[:-1].split("\t")
                    user = int(lines[0])
                    movie = int(lines[1])
                    score = float(lines[2])
                    data.append((user, movie, score))
                    if user > u:u = user
                    if movie > i:i = movie
                    if score > maxr:maxr = score
        self.maxRate = maxr
        print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}\n"
                  "\tItem Num: {}\n"
                  "\tData Size: {}".format(u, i, len(data)))
        return data, [u+1, i+1]

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getEmbedding(self):
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating
        return np.array(train_matrix)

    def getInstances(self, data, negNum):
        user = []
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self):
        #loading data
        testset = []
        filePath = '/data/fjsdata/ctKngBase/ml/pinterest-20.test.negative'
        with open(filePath, 'r') as fd:
            line = fd.readline()
            while line != None and line != '':
                arr = line.split('\t')
                u = eval(arr[0])[0]
                testset.append([u, eval(arr[0])[1]])#one postive item
                for i in arr[1:]:
                    testset.append([u, int(i)]) #99 negative items
                line = fd.readline()
        #format    
        user = []
        item = []
        u_prev = testset[0][0]
        tmp_user = []
        tmp_item = []
        for u, i in testset:
            if u_prev ==u:
                tmp_user.append(u)
                tmp_item.append(i)
            else:
                user.append(tmp_user)
                item.append(tmp_item)
                tmp_user = []
                tmp_item = []
                tmp_user.append(u)
                tmp_item.append(i)
            u_prev = u
        return [np.array(user), np.array(item)]
    
if __name__ == '__main__':
     for K in [64, 32, 16, 8]:
        dmf = DMF(K, negNum=4, lr=0.001, maxEpochs=2, topK=10)
        dmf.run()
        
'''
nohup python -u DMF-TF-pe.py > dmftf-pe.log  &
K = 8 
K = 16
K = 32
K = 64
'''