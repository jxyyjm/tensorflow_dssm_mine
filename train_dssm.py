#!/usr/bin/python
# -*- coding:utf-8 -*-

## @objection : try to implement the DSSM using tensorflow ##
## @reference : Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
## @date      : 2017-01-23
## @author    : yujianmin

from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

## data-batch get from this function ##
def get_query_doc_input(step): ## query_batch, doc_batch -> array ##
	query_batch = np.ndarray()
	doc_batch   = np.ndarray()
	


## model struct ##
## input   : 30k word-hash vector ##
## layer-1 : 300 hidden nodes     ##
## layer-2 : 300 hidden nodes     ##
## output  : 128 output nodes     ##

## input variables ##
input_node_num  = 30000
layer1_node_num = 300
layer2_node_num = 300
output_node_num = 128

## notice : 英文可以用word-hash来限制输入维度的数量26个字母的n-letter组合，但对中文较为勉强，还是用tf-idf算了 ##
## notice : paper-author用的是query-title做的试验 ## 我们这里也用query-title 对来做输入 ##

## 先用稀疏占位符 构造输入数据的接收变量 ##
## 稀疏矩阵 ##
query_input = tf.sparse_placeholder(shape = [None, input_node_num], dtype = tf.float32, name = 'query_batch')
doc_input   = tf.sparse_placeholder(shape = [None, input_node_num], dtype = tf.float32, name = 'doc_batch')

## middle variables ##
## layer-1 ## output ##
max_min_threshold = tf.sqrt(6*(input_node_num + output_node_num))
layer1_w = tf.Variable(tf.random_uniform(
										shape  = [input_node_num, layer1_node_num], 
										minval = -max_min_threshold, 
										maxval = +max_min_threshold),
						name  = 'layer1_w', 
						dtype = tf.float32)
layer1_b = tf.Variable(tf.random_uniform(
										shape  = [layer1_node_num],
										minval = -max_min_threshold,
										maxval = +max_min_threshold),
						name  = 'layer1_b',
						dtype = tf.float32)
query_1_output = tf.nn.relu(
						tf.sparse_tensor_dense_matmul(query_input, layer1_w) + layer1_b
						)
doc_1_output   = tf.nn.relu(
						tf.sparse_tensor_dense_matmul(doc_input, layer1_w) + layer1_b
						)
## layer-2 ## output ##
layer2_w = tf.Variable(tf.random_uniform(
										shape  = [layer1_node_num, layer2_node_num],
										minval = -max_min_threshold,
										maxval = +max_min_threshold),
						name  = 'layer2_w',
						dtype = tf.float32)
layer2_b = tf.Variable(tf.random_uniform(
										shape  = [layer2_node_num],
										minval = -max_min_threshold,
										maxval = +max_min_threshold),
						name  = 'layer2_b',
						dtype = tf.float32)
query_2_output = tf.nn.relu(
						tf.matmul(query_1_output, layer2_w) + layer2_b
							)
doc_2_output   = tf.nn.relu(
						tf.matmul(doc_1_output, layer2_w) + layer2_b
						)
## layer-3 and output ##
layer3_w = tf.Variable(tf.random_uniform(
										shape  = [layer2_node_num, output_node_num],
										minval = -max_min_threshold,
										maxval = +max_min_threshold),
						name  = 'layer3_w',
						dtype = tf.float32)
layer3_b = tf.Variable(tf.random_uniform(
										shape  = [output_node_num],
										minval = -max_min_threshold,
										maxval = +max_min_threshold),
						name  = 'layer3_b',
						dtype = tf.float32)
query_output = tf.nn.relu(
						tf.matmul(query_2_output, layer3_w) + layer3_b
						)
doc_output   = tf.nn.relu(
						tf.matmul(doc_2_output, layer3_w) + layer3_b
						)
## optimal-function ##
# loss   = - tf.sum( tf.log(P(D|Q)))  min 
# P(D|Q) = tf.nn.softmax(similar_matrix(Q&D))

## once a batch ：shape = m * input_node_num
# [       input          ]      [   output    ]
# [ sample-1 : query|doc ]      [ f(sample-1) ]
# [ sampel-2 : query|doc ] == > [ f(sample-2) ]
# [ sample-3 : query|doc ]      [ f(sampel-3) ]
# [ sample-4 : query|doc ]      [ f(sampel-4) ]
## current time ##
## Q&D 的相似度 矩阵，用于计算 softmax ## P(D|Q) = R(D, Q) / sum( R(Q, Di)) ## 所有文档中，当前分布所占概率 ##
## 故相似矩阵如下 ：shape = m * m
# [ R(D1, Q1), R(D2, Q1), R(D3, Q1), ... , R(Dm, Q1) ]
# [ R(D1, Q2), R(D2, Q2), R(D3, Q2), ... , R(Dm, Q2) ]
# [ R(D1, Q3), R(D2, Q3), R(D3, Q3), ... , R(Dm, Q3) ]
# [ R(D1, Q4), R(D2, Q4), R(D3, Q4), ... , R(Dm, Q4) ]
## 其中 R(D, Q) = (query_output * doc_output.T) / (|query_output| * |doc_output|)
# query_input.shape  = m * input_node_num
# query_output.shape = m * 128
# doc_input.shape    = m * input_node_num
# doc_output.shape   = m * 128
## 故计算相似矩阵 如下 ：
# 分子 = tf.matmul(query_output, doc_output, transpose_b=True)  shape = m * m 
# 分母 = tf.mul(query_norm.tile([1, m]), doc_norm.tile([1, m]).T)
#		query_norm = tf.sqrt( tf.reduce_sum(query_output, reduction_indices=0, keep_dim=True))
#		doc_norm   = tf.sqrt( tf.reduce_sum(doc_output,   reduceion_indices=0, keep_dim=True))

similar_fenzi = tf.matmul(query_output, doc_output, transpose_b = True)
query_norm    = tf.sqrt(tf.reduce_sum(query_output, reduction_indices = 0, keep_dim = True)
doc_norm      = tf.sqrt(tf.reduce_sum(doc_output  , reduction_indices = 0, keep_dim = True)
similar_fenmu = tf.mul(query_norm.tile([1, m]), tf.transpose(doc_norm.tile([1, m])))
similar_matrix= tf.truediv(similar_fenzi, similar_fenmu)


prob     = tf.nn.softmax(similar_matrix) ## P(D|Q) ##
## 取出对角线的概率来，这些是成对的query-doc的条件概率 ##
Batch_num= tf.shape(query_input)[0]
one_diag = tf.diag(tf.ones(shape = [Batch_num])) 
pair_pro = tf.mul(prob, one_diag) ## element-mul ## 这样就取出了query-doc对的条件概率 ## 非对角位置都是 0 ##
loss     = -tf.reduce_mean(tf.log(pair_pro), [0,1])     ## 对那些成对的query-doc, max{sum(log(P(D|Q)))}，即min{-sum}


learn_step = 0.001
train_step = tf.train.GradientDescentOptimizer(learn_step).minimize(loss)


## 训练效果评估 ##
accuracy_bool  = tf.equal(
						tf.argmax(prob, dimension=1),
						tf.constant(value = range(Batch_num), dtype = tf.int64)) ## 匹配y_real; y_predict ##
accuracy_ratio = tf.reduce_mean(tf.cast(accuracy_bool, dtype = tf.int64))
## train the model ##
train_max_num = 100
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for step in range(train_max_num):
		cur_query_input, cur_doc_input = get_query_doc_input(step)
		sess.run(train_step, feed_dict = {
											query_input : cur_query_input, 
											doc_input : cur_doc_input})
		train_accuracy = sess.run(accuracy_ratio, feed_dict = {
															query_input : cur_query_input, 
															doc_input : cur_doc_input})
		print ('step : ', step, ', its train accracy is : ', train_accuracy)

## 测试集效果评估 ##



sess.close()
