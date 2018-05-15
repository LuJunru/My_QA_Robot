# -*- coding: utf-8 -*-
# @Author  : Junru_Lu
# @File    : CNN_SentenceSimilarity_Training.py
# @Software: PyCharm
# @Environment : Python 3.6+

# 基础包
import jieba
import tensorflow as tf
import numpy as np
import math
import os
from sklearn.model_selection import train_test_split
from gensim.models import keyedvectors

# 编码相关包
import importlib, sys
importlib.reload(sys)


'''
本配置文件用于利用CNN训练相似句子匹配模型

按如下方法查看Tensorboard：
-模型训练结束后，在终端输入：tensorboard --logdir='本次训练对应的日志的上级文件夹地址'
-'Starting TensorBoard 41 on port 6006'这句话出现后，将显示的网址复制到浏览器地址栏
-*如果没有出现网址，在地址栏输入'localhost:6006'即可
'''


# ------------------预加载------------------ #

cur_dir = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()  # 当前项目路径

# jieba分词的词典包括：基础词典+动态添加的项目核心词+动态添加的领域相关词
# 项目核心词表
all_keywords = set(list(open(cur_dir + '/basic_data_file/keywords.txt', 'r').read().strip().split('\n')))
for keyword in all_keywords:
    jieba.add_word(keyword, freq=None, tag=None)
# 领域相关词：推荐来源之一，搜狗细胞词库
# 附：搜狗细胞词库解析代码https://github.com/LuJunru/SMPCUP2017_ELP/blob/master/Sougou_dic_trans.py
field_words = set(list(open(cur_dir + '/basic_data_file/fieldwords.txt', 'r').read().strip().split('\n')))
for field_word in field_words:
    jieba.add_word(field_word, freq=None, tag=None)

# 基于10G中文维基训练的60维词向量
word_vectors = keyedvectors.KeyedVectors.load(cur_dir + '/Word Embedding/Word60.model')  # 加载预先训练好的词向量

# CNN相关参数
MAX_LENTH = 16  # 训练时保留的词的最大数量, 必须为4的倍数
OUT_SIZE1 = int(MAX_LENTH / 4)  # MAX_LENTH / 4 = 10
OUT_SIZE2 = int(MAX_LENTH / 2)  # MAX_LENTH / 2 = 20
CLASS_TYPE = 2
path = cur_dir + '/pre_trained_models/CNN_' + str(MAX_LENTH)
if os.path.exists(path) is False:
    os.makedirs(path)


# ------------------基础函数------------------ #


def sen_vector_gen(title_words):  # 生成句子的向量
    sen_vector = np.zeros(60, dtype=float)
    length = 0
    for word in title_words:
        try:
            sen_vector += word_vectors[word]
            length += 1
        except:
            try:  # 若当前词没有对应向量，尝试能否进一步切词获得向量(总比直接跳过好)
                for w in jieba.lcut(word):
                    sen_vector += word_vectors[w]
                    length += 1
            except:
                pass
    if length != 0:
        sen_vector = sen_vector / length  # 用词的词向量的平均值来表示句子向量
    return [sen_vector]


def get_vec_cosine(vec1, vec2):  # 计算两个向量的相似度
    tmp = np.vdot(vec1, vec1) * np.vdot(vec2, vec2)
    if tmp == 0.0:
        return 0.0
    return np.vdot(vec1, vec2) / math.sqrt(tmp)


def s1_s2_simipics(s1_list, s2_list, max_lenth):  # 生成feature map
    k = 0
    simi = []
    while k < max_lenth:
        try:
            sen_k = sen_vector_gen(s1_list[k])
            j = 0
            while j < max_lenth:
                try:
                    sen_j = sen_vector_gen(s2_list[j])
                    simi_pic = get_vec_cosine(sen_k, sen_j)
                except:
                    simi_pic = 0.0
                simi.append(simi_pic)
                j += 1
        except:
            simi_pic = 0.0
            simi.append(simi_pic)
        k += 1
    while len(simi) < MAX_LENTH**2:
        simi.append(0.0)
    return simi


# ------------------CNN------------------ #


def weight_variable(shape, var_name):  # 定义初始权重
    # 形状为shape的随机变量
    initial = tf.truncated_normal(shape, stddev=0.1, name=var_name)
    return tf.Variable(initial)


def bias_variable(shape, var_name):  # 定义初始偏置值
    initial = tf.constant(0.1, shape=shape, name=var_name)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride[1,x_movement,y_movement,1]
    # padding=same:不够的地方补0；padding=valid:会缩小
    # 2维卷积层,卷积步长为(x=1,y=1)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # maxpooling
    # ksize表示核函数大小
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # 预测的向量中1的位置和标签中1的位置是否一致
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# ------------------主函数------------------ #


if __name__ == '__main__':

    with tf.name_scope('inputs'):  # 'Inputs'层
        keep_prob = tf.placeholder(tf.float32)
        # None表示无论多少行例子都可以
        xs = tf.placeholder(tf.float32, [None, MAX_LENTH ** 2], 'x_input')
        ys = tf.placeholder(tf.float32, [None, CLASS_TYPE], 'y_input')
        # -1表示feature map个数,1表示Channel个数
        x_image = tf.reshape(xs, [-1, MAX_LENTH, MAX_LENTH, 1])

    # 第一层卷积+pooling
    # 核函数大小patch=2*2;通道数，即特征数为1所以in_size=1;新特征的厚度为OUT_SIZE1
    W_conv1 = weight_variable([5, 5, 1, OUT_SIZE1], 'w1')
    b_conv1 = bias_variable([OUT_SIZE1], 'b1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积+pooling
    # 核函数大小patch=2*2;in_size=4;新特征的厚度为OUT_SIZE2
    W_conv2 = weight_variable([5, 5, OUT_SIZE1, OUT_SIZE2], 'w2')
    b_conv2 = bias_variable([OUT_SIZE2], 'b2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 第一层全连接层func1 layer
    W_fc1 = weight_variable([OUT_SIZE1 * OUT_SIZE1 * OUT_SIZE2, MAX_LENTH], 'wf1')
    b_fc1 = bias_variable([MAX_LENTH], 'bf1')
    h_pool2_flat = tf.reshape(h_pool2, [-1, OUT_SIZE1 * OUT_SIZE1 * OUT_SIZE2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 第二层全连接层func2 layer
    W_fc2 = weight_variable([MAX_LENTH, CLASS_TYPE], 'wf2')
    b_fc2 = bias_variable([CLASS_TYPE], 'bf2')
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    with tf.name_scope('loss'):
        # 交叉熵损失函数
        loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        # 训练目标
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    init = tf.global_variables_initializer()  # 变量初始化
    saver = tf.train.Saver()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('CNN_logs/', sess.graph)
        sess.run(init)

        s = np.zeros(MAX_LENTH ** 2 + 2, float)  # 构建输入特征
        for line in open(cur_dir + '/basic_data_file/cnn_train_data.txt', 'r'):
            line_seg = line.strip().split('\t')
            w1 = jieba.lcut(line_seg[1])
            w2 = jieba.lcut(line_seg[3])
            label1 = line_seg[4]
            label2 = line_seg[5]
            line_simi = s1_s2_simipics(w1, w2, MAX_LENTH) + [float(label1)] + [float(label2)]
            s = np.vstack((s, line_simi))
        S = s[1:, :]
        np.random.shuffle(S)
        X_train, X_test, Y_train, Y_test = train_test_split(S[:, :-2], S[:, -2:], test_size=0.1)

        for i in range(1000):  # 训练
            # batch_xs, batch_xy = X_train[5 * i:5 * i + 400, :], Y_train[5 * i:5 * i + 400, :]
            batch_xs, batch_xy = X_train, Y_train
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_xy, keep_prob: 0.8})
            if i % 50 == 0:
                result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_xy, keep_prob: 1})
                writer.add_summary(result, i)
                print(compute_accuracy(X_test, Y_test))

        save_path = saver.save(sess, path + '/CNN_' + str(MAX_LENTH) + '.ckpt')
        print("模型保存到路径:" + save_path)
