#!/usr/bin/python
# -*- coding:utf-8 -*-
# reference : http://liaha.github.io/models/2016/06/21/dssm-on-tensorflow.html
# reference : https://github.com/ShuaiyiLiu/sent_cnn_tf/blob/master/train_cnn.py
# reference : https://github.com/Microsoft/CNTK/wiki/Train-a-DSSM-(or-a-convolutional-DSSM)-model

from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

## 1) input batch and sparse ## 每个样本都是由 query-word-hash | doc-word-hash 组成
##	tensorflow 支持稀疏占位符 ##

TRIGRAM_D   = 500000 ## word-hash之后的向量长度 大小 ##
query_batch = tf.sparse_placeholder(dtype = tf.float32, 
									shape = [None, TRIGRAM_D], 
									name  = 'QueryBatch')
doc_batch   = tf.sparse_placeholder(dtype = tf.float32, 
									shape = [None, TRIGRAM_D], 
									name  = 'DocBatch')
## 2) initialize weight and bias
##	input num : TRIGRAM_D
##	Layer one : L_1_node = 300
##  Layer two : L_2_node = 300
##  outputnum : OUTPUT_D  = 128

L_1_node    = 300
L_2_node    = 300
OUTPUT_D    = 128
minval_init = -np.sqrt(TRIGRAM_D + OUTPUT_D)
maxval_init = -minal_init


## first layer ##
Weight_1 = tf.Variable(tf.random_uniform(shape  = [TRIGRAM_D, L_1_node],
										 minval = minval_init,
										 maxval = maxval_init))
Bias_1   = tf.variable(tf.random_uniform(shape  = [L_1_node],
										 minval = minval_init,
										 maxval = maxval_init))
## second layer ##
Weight_2 = tf.Variable(tf.random_uniform(shape  = [L_1_node, L_2_node],
										 minval = minval_init,
										 maxval = maxval_init))
Bias_2   = tf.variable(tf.random_uniform(shape  = [L_2_node],
										 minval = minval_init,
										 maxval = maxval_init))
## third layer ##
Weigth_3 = tf.Variable(tf.random_uniform(shape  = [L_2_node, OUTPUT_D], 
										 minval = minval_init, 
										 maxval = maxval_init))
Bias_3   = tf.Variable(tf.random_uniform(shape  = [OUTPUT_D],
										 minval = minval_init,
										 maxval = maxval_init))

## 3) define the full-connection layer operation
query_1_out = tf.nn.relu(
						tf.sparse_tensor_dense_matmul(query_batch, Weight_1) + Bias_1
						)
doc_1_out   = tf.nn.relu(
						tf.sparse_tensor_dense_matmul(doc_batch, Weight_1) + Bias_1
						)
query_2_out = tf.nn.relu(
						tf.sparse_tensor_dense_matmul(query_1_out_batch, Weight_2) + Bias_2
						)
doc_2_out   = tf.nn.relu(
						tf.sparse_tensor_dense_matmul(doc_1_out_batch, Weight_2) + Bias_2
						)
query_3_out = tf.nn.relu(
						tf.sparse_tensor_dense_matmul(query_2_out_batch, Weight_3) + Bias_3
						)
doc_3_out   = tf.nn.relu(
						tf.sparse_tensor_dense_matmul(doc_2_out_batch, Weight_3) + Bias_3
						)

## 4) 相似度计算 ##
NEG = ##  ????  ## 当次 负样本个数 ? ##
BS  = ## 每份batch的行数 ##
Gamma = 0.001 ##

## 计算 每个query与doc的相似度, ##
## 因为有多个doc对应一个query , 直接用复制 ##
##									  先计算 R(Q, D) ##
##									   R(Q, D)    = (y_q * y_d ) / (||y_q|| * ||y_d||)
##									   其中 ||y|| = sqrt( sum( y_i*y_i ))
## query 的映射空间向量，  sqrt( sum[ query_y)**2])
## 4.1) 下面的query_norm 和 doc_norm 是计算的 R(Q, D) 的分母的一半 ## ||y_q||和||y_d|| ##
query_norm = tf.tile(
					tf.sqrt(tf.reduce_sum(tf.square(query_3_out), 1, keep_dims=True)), ## 保留维度keep_dims ##
					[NEG + 1, 1]  ## 行复制为NEG+1倍 ## 列不复制 ##
					)
doc_norm   = tf.tile(
					tf.sqrt(tf.reduce_sum(tf.square(doc_3_out), 1, keep_dims=True)),
					[NEG + 1, 1]
					)
norm_prod  = tf.mul(query_norm, doc_norm) ## 分母 ##

## 4.2) P(D|Q) = exp(R(D|Q)) / sum( exp( R(D|Q))) ## 某个doc的概率 ##
prod       = tf.reduce_sum(tf.mul(query_3_out, doc_3_out)) ## 分子 ##
## notice : tf.mul 在元素上相乘 ## tf.matmul 是矩阵相乘 ##

cos_sim_raw= tf.truediv(prod, norm_prod) ## 元素级别的除法 ## R(Q, D) ##
cos_sim    = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG+1, BS])) * Gamma
		   # tf.transpose -> 反转矩阵 #

## 5) loss 函数 ## -min{ prod{ P(D|Q) }} ##
prob       = tf.nn.softmax(cos_sim)) ## P(D|Q) = softmax()
# For each batch `i` and class `j` we have 
# softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j])) 
hit_prob   = tf.slice(prob, [0, 0], [-1, 1]) ## 从二维输入的最开始，切 所有行，第1列 ##
loss       = -tf.reduce_sum(tf.log(hit_prob))/ BS ## 本次输入的条件概率之log和 ##
## tf.slice ## 切片取数据 ##
## tf.slice(input, begin=[], size=[])
## for example : 
## input.shape=[3,2,3] ## 三维 ##
## tf.slice(input, begin=[1,0,0], size=[1,1,3]) 
## 		begin[i] 表示切片在input的某个维度开始的位置
##		size[i]  表示切片在input的某个维度要切多少位
##		上面表示，从input的[1,0,0]开始，切1行，1列，3纵 #


## 6) Training ##
Learning_rate = 0.001
max_steps     = 1000

train_step = tf.train.GradientDescentOptimizer(Learning_rate).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for step in range(max_steps):
		sess.run(train_step, feed_dict = {query_batch: get_query_input(step)
										  doc_batch  : get_doc_input(step)})
# |      The optional `feed_dict` argument allows the caller to override
# |      the value of tensors in the graph


