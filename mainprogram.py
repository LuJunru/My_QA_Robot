# -*- coding: utf-8 -*-
# @Author  : Junru_Lu
# @File    : mainprogram.py
# @Software: PyCharm
# @Environment : Python 3.6+

# 网页和服务请求相关包
from bs4 import BeautifulSoup
from urllib.parse import quote
import requests

# ES相关包
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
import elasticsearch

# 基础包
import time
import jieba.posseg as pseg  # 用于词性标注
import jieba
import tensorflow as tf  # 启动CNN模型时使用
from snownlp import SnowNLP  # 用于句子情感打分
import os
import re
import numpy as np
import math
import json
import pickle
from gensim.models import keyedvectors
from pyltp import NamedEntityRecognizer, Postagger, Segmentor, Parser  # 用于抽取句子NER和主谓宾主干成分

# 编码相关包
import importlib, sys
importlib.reload(sys)


'''
本程序设计：对于预处理后的新问题，先用ES进行初选，ES中使用添加了top1近义词并对主谓宾、NER和重点词加权的分词结果生成query，查询时保证逐词非逐字查询
          对于获取到的初选集，一方面使用CNN计算句对语义相似度，另一方面使用问答质量匹配子模块计算问答匹配度，
          该模块使用10W对百度知道问答训练获得，模型准确率74%(预计更新词向量后重新训练)。完成两部分得分后，将总得分最高的初选问答作为精选返回给用户。
          若本地精选完成仍没有合适问答返回时，启动在线搜索模块先后到百度主页面和百度知道进行查询，并将百度知道中的优质问答添加到ES缓存中。
          
本程序启动前，必须先在cmd启动ES服务：
进入ES地址：cd your-es-address/elasticsearch-6.2.2
启动ES服务：bin/elasticsearch
ES工作原理：主要是倒排索引，还用了各种跳表、term dictionary等技巧。详见:http://blog.csdn.net/cyony/article/details/65437708?locationNum=9&fps=1
'''


# ------------------预加载------------------ #

cur_dir = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()  # 当前项目路径

stopwords = set(list(open(cur_dir + '/basic_data_file/stopwords.txt', 'r').read().strip().split('\n')))  # 停用词表

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

photo_content = {}  # 百度知道上图片-文字对应表，目前共99个文字
for content_line in open(cur_dir + '/basic_data_file/photo_content.txt', 'r'):
    content = content_line.strip().split('\t')[1]
    photo_content[content] = content_line.strip().split('\t')[0]

# 添加你自己的词向量或者下载我提供的词向量，注意：如果换用其它维度的词向量，需要重新训练CNN和问答质量两个子模型
# 提供一个基于10G中文维基训练的60维词向量，链接: https://pan.baidu.com/s/19f5FkvJi7IKgU-UAkSWPXg 密码: fxjg
word_vectors = keyedvectors.KeyedVectors.load(cur_dir + '/Word Embedding/Word60.model')  # 加载预先训练好的词向量

# 预测问答对属于"好"的概率的问答质量评价子模块（XGB）
xgb = pickle.load(open(cur_dir + '/pre_trained_models/xgboost_qaquality_21_60dz_s0.745.pkl', 'rb'))

# NER识别和主谓宾抽取服务，调用哈工大LTP的PyLTP包:http://pyltp.readthedocs.io/zh_CN/latest/api.html#id13
# 加载ltp模型，请先到该链接下载模型http://pyltp.readthedocs.io/zh_CN/latest/api.html#id2
# 注意模型版本与pyltp版本的对应关系
LTP_DATA_DIR = cur_dir + '/ltp_data_v3.4.0'
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径
pos_model_path = os.path.join(LTP_DATA_DIR, "pos.model")  # 词性模型路径
cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")  # 分词模型路径
par_model_path = os.path.join(LTP_DATA_DIR, "parser.model")  # parser模型路径
recognizer = NamedEntityRecognizer()  # 初始化实例
recognizer.load(ner_model_path)  # 加载模型
postagger = Postagger()
postagger.load(pos_model_path)
segmentor = Segmentor()
segmentor.load(cws_model_path)
parser = Parser()
parser.load(par_model_path)

# 建立ES实例
es = Elasticsearch()

# 预设参数
THRESHOLD = 1.0  # 控制是否返回精选结果，[0，2]
ESMAX = 1  # 控制ES初选集的大小，该参数越大，ES初选集遗漏正确问答对的概率就越低，但程序运行速度就越慢
SIMI = 1.0  # 计算精选得分时新问题-候选问题语义相似度similarity的权重参数
QUA = 1.0  # 计算精选得分时新问题-候选答案匹配度quality的权重参数
ZHIDAOMAX = 3  # 控制在线搜索模块中从百度知道请求返回的问答数量。该值越大，返回的相似问答数量越多，但响应速度也就越慢
SAVESCORE = 0.8  # 控制增量存储模块的百度知道问答对的阈值, [0, 1]

# 调用CNN模型时需用到的参数
MAX_LENTH = 16  # 训练时保留的词的最大数量, 必须为4的倍数
OUT_SIZE1 = int(MAX_LENTH / 4)  # MAX_LENTH / 4 = 10
OUT_SIZE2 = int(MAX_LENTH / 2)  # MAX_LENTH / 2 = 20
CLASS_TYPE = 2
path = cur_dir + '/pre_trained_models/CNN_' + str(MAX_LENTH)
saverpath = path + '/CNN_' + str(MAX_LENTH) + '.ckpt'


# ------------------ES相关函数------------------ #

def query_generation(inputs):
    '''
    ES DSL语法:https://es.xiaoleilu.com/054_Query_DSL/70_Important_clauses.html
    全文搜索: https://es.xiaoleilu.com/100_Full_Text_Search/00_Intro.html
    '''
    _query = \
        {
            "query": {
                "bool": {
                    "should": inputs
                }
            }, 'size': ESMAX
        }

    '''# multi-match多级查询例子
    {
        "multi_match": {
            "query": {
                "bool": {
                    "should": inputs
                }
            }
        }
    }
    '''

    '''# bool布尔查询例子
    {
        "bool": {
            "must":     { "match": { "title": "how to make millions" }},
            "must_not": { "match": { "tag":   "spam" }},
            "should": [
                { "match": { "tag": "starred" }},
                { "range": { "date": { "gte": "2014-01-01" }}}
            ]
        }
    }
    '''

    return _query


class ElasticSearchClient(object):  # 启动ES
    @staticmethod
    def get_es_servers():
        es_servers = [{
            "host": "localhost",
            "port": "9200"
        }]
        es_client = elasticsearch.Elasticsearch(hosts=es_servers)
        return es_client


class LoadElasticSearch(object):  # 在ES中加载、批量插入数据
    def __init__(self):
        self.index = "my-index"
        self.doc_type = "test-type"
        self.es_client = ElasticSearchClient.get_es_servers()
        self.set_mapping()

    def set_mapping(self):
        """
        设置mapping
        IK插件: https://github.com/medcl/elasticsearch-analysis-ik
        """
        mapping = {
            self.doc_type: {
                    "topic": {
                        "type": "string"
                    },
                    "question": {
                        "type": "string"
                    },
                    "answer": {
                        "type": "string"
                    }
                }
            }

        if not self.es_client.indices.exists(index=self.index):
            # 创建Index和mapping
            self.es_client.indices.create(index=self.index, body=mapping, ignore=400)
            self.es_client.indices.put_mapping(index=self.index, doc_type=self.doc_type, body=mapping)

    def add_date(self, row_obj):
        """
        单条插入ES
        """
        _id = row_obj.get("_id", 1)
        row_obj.pop("_id")
        self.es_client.index(index=self.index, doc_type=self.doc_type, body=row_obj, id=_id)

    def add_date_bulk(self, row_obj_list):
        """
        批量插入ES
        """
        load_data = []
        i = 1
        bulk_num = 100000  # 100000条为一批
        for row_obj in row_obj_list:
            action = {
                "_index": self.index,
                "_type": self.doc_type,
                "_source": {
                    'topic': row_obj.get('topic', None),
                    'question': row_obj.get('question', None),
                    'answer': row_obj.get('answer', None),
                }
            }
            load_data.append(action)
            i += 1
            # 批量处理
            if len(load_data) == bulk_num:
                print('插入', int(i / bulk_num), '批数据')
                success, failed = bulk(self.es_client, load_data, index=self.index, raise_on_error=True)
                del load_data[0:len(load_data)]
                print(success, failed)

        if len(load_data) > 0:
            success, failed = bulk(self.es_client, load_data, index=self.index, raise_on_error=True)
            del load_data[0:len(load_data)]
            print(success, failed)


# ------------------基础函数------------------- #

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


# 调用CNN模型时需要的函数
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


def ner(words):  # 命名实体识别
    postags = postagger.postag(words)
    netags = recognizer.recognize(words, postags)
    return [words[list(netags).index(netag)] for netag in list(netags) if netag != "O"]


# 以下三个函数实现了主谓宾三元组抽取
def build_parse_child_dict(words, arcs):  # 依存句法分析函数
    """
    为句子中的每个词语维护一个保存句法依存儿子节点的字典
    Args:
        words: 分词列表
        postags: 词性列表
        arcs: 句法依存列表
    """
    child_dict_list = []
    for index in range(len(words)):
        child_dict = dict()
        for arc_index in range(len(arcs)):
            if arcs[arc_index].head == index + 1:
                if arcs[arc_index].relation in child_dict:
                    child_dict[arcs[arc_index].relation].append(arc_index)
                else:
                    child_dict[arcs[arc_index].relation] = []
                    child_dict[arcs[arc_index].relation].append(arc_index)
        child_dict_list.append(child_dict)
    return child_dict_list


def complete_e(words, postags, child_dict_list, word_index):  # 完善识别的部分实体
    child_dict = child_dict_list[word_index]
    prefix = ''
    if 'ATT' in child_dict:
        for i in range(len(child_dict['ATT'])):
            prefix += complete_e(words, postags, child_dict_list, child_dict['ATT'][i])

    postfix = ''
    if postags[word_index] == 'v':
        if 'VOB' in child_dict:
            postfix += complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
        if 'SBV' in child_dict:
            prefix = complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

    return prefix + words[word_index] + postfix


def fact_triple_extract(sentence):  # 对于给定的句子进行事实三元组抽取
    words = segmentor.segment(sentence)
    postags = postagger.postag(words)
    arcs = parser.parse(words, postags)

    child_dict_list = build_parse_child_dict(words, arcs)
    for index in range(len(postags)):
        # 抽取以谓词为中心的事实三元组
        if postags[index] == 'v':
            child_dict = child_dict_list[index]
            # 主谓宾
            if 'SBV' in child_dict and 'VOB' in child_dict:
                e1 = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                r = words[index]
                e2 = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                return [e1, r, e2]
    return []


def word_weight_exchange(original_list, original_text):  # 修改列表中各个词的词频权重

    weighted_list = original_list
    key_tempt_list = []
    for word in original_list:  # 提高重点词的权重
        if word in all_keywords:
            key_tempt_list.append(word)
    weighted_list += key_tempt_list
    synon_tempt_list = []
    for synon in original_list:  # 添加部分近义词
        if synon in word_vectors:
            synon_tempt_list.append(word_vectors.most_similar(synon)[0][0])
    weighted_list += synon_tempt_list
    weighted_list += ner(original_list)  # 提高NER权重
    weighted_list += fact_triple_extract(original_text)  # 提高主干部分权重
    final_list = [fw for fw in weighted_list if fw not in stopwords]

    return final_list


# ------------------问答质量评价子模块------------------ #

class QAQuality:
    def __init__(self, Q, A):
        # 基础运算变量
        self.answer = A  # 候选答案
        self.question = Q  # 候选问题
        self.alenth = len(self.answer)  # 答案长度
        self.qlenth = len(self.question)  # 问题长度
        self.q_pos_list = pseg.lcut(self.question)  # 问题分词结果
        self.a_pos_list = pseg.lcut(self.answer)  # 答案分词结果
        self.inter_list = [val for val in self.q_pos_list if val in self.a_pos_list]  # 分词交集
        self.union_list = list(set(self.q_pos_list + self.a_pos_list))  # 分词并集
        self.q_noun = [qnoun.word for qnoun in self.q_pos_list if 'n' in str(qnoun)]  # 问题名词
        self.a_noun = [anoun.word for anoun in self.a_pos_list if 'n' in str(anoun)]  # 答案名词
        self.qa_noun = self.q_noun + self.a_noun  # 问答对中的名词
        self.q_verb = [qverb.word for qverb in self.q_pos_list if 'v' in str(qverb)]  # 问题动词
        self.a_verb = [averb.word for averb in self.a_pos_list if 'v' in str(averb)]  # 答案动词
        self.qa_verb = self.q_verb + self.a_verb  # 问答对中的动词
        self.a_stop = [stopword.word for stopword in self.a_pos_list if stopword.word in stopwords]  # 答案停用词
        self.q_nonpunc = "".join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', self.question))  # 问题中英文及数字部分
        self.a_nonpunc = "".join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', self.answer))  # 答案中英文及数字部分
        self.qa_nonpunc = "".join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', self.question + self.answer))  # 问答对文字部分
        self.a_singleword = ['@&@' for a in self.a_pos_list if len(a.word) == 1].count('@&@')  # 答案中单字个数
        self.n_sentence = (len(self.answer.strip().split('。')) - 1 if '。' in self.answer else 1)  # 答案中句子数

        # 各类判断指标
        self.lenth_ratio = float(len(self.answer)) / float(len(self.question))  # 答案-问题长度比
        self.resemblance = float(len(self.inter_list)) / float(len(self.union_list))  # 问题-答案相似度
        self.contain = float(len(self.inter_list)) / float(len(self.a_pos_list))  # 答案覆盖度
        self.overlap = float(len(self.a_pos_list)) / float(len(self.q_pos_list))  # 问题-答案重合度
        self.cosine = float(len(self.inter_list)) / math.sqrt((len(self.q_pos_list)*len(self.a_pos_list)))  # 长度余弦值
        self.qa_noun_per = float(len(self.qa_noun)) / float(len(self.q_pos_list) + len(self.a_pos_list))  # 问答对中名词比重
        self.a_noun_per = float(len(self.a_noun)) / float(len(self.a_pos_list))  # 答案名词比重
        self.qa_verb_per = float(len(self.qa_verb)) / float(len(self.q_pos_list) + len(self.a_pos_list))  # 问答对中动词比重
        self.a_verb_per = float(len(self.a_verb)) / float(len(self.a_pos_list))  # 问题动词比重
        self.a_stop_per = float(len(self.a_stop)) / float(len(self.a_pos_list))  # 答案停用词比重
        self.a_punc_dens = 1.0 - float(len(self.a_nonpunc)) / float(len(self.answer))  # 答案非文字部分比重
        self.qa_punc_dens = 1.0 - float(len(self.qa_nonpunc)) / float(len(self.question + self.answer))  # 问答对非文字比重
        self.a_sentiment = SnowNLP(self.answer).sentiments  # 答案情感得分
        self.a_singleword_per = float(self.a_singleword) / float(len(self.a_pos_list))  # 答案中单字比重
        self.q_punc_dens = 1.0 - float(len(self.q_nonpunc)) / float(len(self.question))  # 问题非文字部分比重
        self.fog_score = 0.4 * (float(len(self.a_pos_list)) / float(self.n_sentence) + 100 * (
            float(len(self.a_pos_list) - self.a_singleword)) / (float(len(self.a_pos_list))))  # fog指数
        self.flesch_score = 206.835 * (1.015 * float(len(self.a_pos_list)) / float(self.n_sentence)) * (
            84.6 * float(sum(len(a.word) for a in self.a_pos_list)) / len(self.a_pos_list))  # flesch指数
        self.flesch_kincaid_score = 0.39 * (float(len(self.a_pos_list)) / float(self.n_sentence)) + 11.8 * (
            float(sum(len(a.word) for a in self.a_pos_list)) / len(self.a_pos_list)) - 15.59  # kincaid指数

    def a_wentroy(self):  # 答案词级别文本熵
        num_dict = {}
        for item in [l.word for l in self.a_pos_list]:
            num_dict[item] = num_dict.get(item, 0.0) + 1.0
        a_dic = {k: v / float(len(self.a_pos_list)) for k, v in num_dict.items()}
        entroy = 0.0
        for key in a_dic:
            entroy += a_dic[key]*(math.log(float(len(self.a_pos_list)), 10)-math.log(a_dic[key], 10))
        return entroy / float(len(self.a_pos_list))

    def a_centroy(self):  # 答案字级别文本熵
        num_dict = {}
        for item in list(self.answer):
            num_dict[item] = num_dict.get(item, 0.0) + 1.0
        a_dic = {k: v / float(self.alenth) for k, v in num_dict.items()}
        entroy = 0.0
        for key in a_dic:
            entroy += a_dic[key]*(math.log(float(self.alenth), 10)-math.log(a_dic[key], 10))
        return entroy / float(self.alenth)

    def a_singles(self):  # 答案最大单字散串长度
        maxflag = 0
        countflag = 0
        for apw in [l.word for l in self.a_pos_list]:
            if len(apw) == 1:
                countflag += 1
                maxflag = countflag
            else:
                countflag = 0
        return maxflag

    def answer_judge_pro(self):  # 计算问答对属于"好问答对"的概率
        features = [float(self.a_singles()), self.lenth_ratio, self.resemblance, self.contain, self.overlap,
                    self.cosine, self.qa_noun_per, self.a_noun_per, self.qa_verb_per, self.a_verb_per, self.a_stop_per,
                    self.a_punc_dens, self.qa_punc_dens, self.a_sentiment, self.a_singleword_per, self.a_wentroy(),
                    self.a_centroy(), self.q_punc_dens, self.fog_score, self.flesch_score, self.flesch_kincaid_score]
        return xgb.predict_proba(features)[:, -1].tolist()


# 调用CNN模型时需要的函数及网络结构描述
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

# CNN网络结构
keep_prob = tf.placeholder(tf.float32)
# None表示无论多少行例子都可以
xs = tf.placeholder(tf.float32, [MAX_LENTH ** 2], 'x_input')
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


# ------------------在线搜索模块------------------ #

badanswers = ['同问', '我也想知道', '。。。。。', '不知道', '', '百度一下', '问百度', '我也不知道', 'ggggggggg', '滚']  # bad answer词表
sensitivewords = ['天气']  # 实时词表


def get_html(url):  # 模拟代理请求和下载html
    headers = {'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686)Gecko/20071127 Firefox/2.0.0.11'}
    soup = BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")
    return soup


class OnlineSearch:  # 在线搜索：先在百度主页面上搜索，搜不到再到百度知道上搜索
    def __init__(self, text):
        self.content = text
        self.content1 = re.sub('天气', 'tianqi', self.content)
        self.main_url = 'https://www.baidu.com/s?wd=' + quote(self.content1)  # 百度知道上搜索的url请求
        self.main_soup = get_html(self.main_url)  # 百度主页面搜索结果返回的html

    def Baidu_Search(self):

        answer = {}
        try:
            result = self.main_soup.find(id=1)  # 获取html中第一个搜索结果
            d = result.attrs  # 检验搜索结果是否为空
        except:
            answer["安全回答"] = "我还在学习中，暂时无法回答您的问题"  # 无搜索结果时返回安全回答
            return answer

        # 给出知乎、搜狐或爱问知识人链接
        if result.find(class_='f13') and ("zhihu" in result.find(class_='f13').get_text()
                                          or "sohu" in result.find(class_='f13').get_text()
                                          or "iask" in result.find(class_='f13').get_text()):
                sublink = re.findall(r'[a-zA-z]+://[^\s]*', str(result))[0][:-1]
                answer[result.find(class_='t').get_text()] = sublink
                return answer

        if result.attrs.get('mu'):  # 检测搜索结果是否为百度知识图谱
            r = result.find(class_='op_exactqa_s_answer')
            if r is not None:
                answer['online answer'] = r.get_text().strip().replace("\n", "").replace(" ", "")
                return answer

        if result.attrs.get('mu') and \
                result.attrs['mu'].__contains__('http://open.baidu.com/static/calculator/calculator.html'):
            # 检测搜索结果是否为百度计算器
            r = result.find(class_='op_new_val_screen_result')
            if r is not None:
                answer['online answer'] = r.get_text().strip().replace("\xa0", "").replace(" ", "").replace("\n", "")
                return answer

        if result.attrs.get('tpl') and result.attrs['tpl'].__contains__('calendar'):
            # 检测搜索结果是否为日期(万年历)
            r = result.find(class_='op-calendar-content')
            if r is not None:
                answer['online answer'] = ''.join(re.compile('[\u4e00-\u9fa50-9\t]').
                    findall(re.sub('\s', '', str(r)).replace('</span>', '\t').replace('<span>', ''))[:-1])
                return answer

        try:
            result2 = self.main_soup.find(id=2)  # 获取html中第二个搜索结果
            d2 = result2.attrs  # 检验搜索结果是否为空
        except:
            result2 = '不存在'

        if result2 != '不存在':
            if result2.attrs.get('tpl') and result2.attrs['tpl'].__contains__('calendar'):
                # 检测搜索结果是否为日期(万年历)
                r = result2.find(class_='op-calendar-content')
                if r is not None:
                    answer['online answer'] = ''.join(re.compile('[\u4e00-\u9fa50-9\t]').\
                        findall(re.sub('\s', '', str(r)).replace('</span>', '\t').replace('<span>', ''))[:-1])
                    return answer

        if result.attrs.get('tpl') and "time" in result.attrs['tpl'] and "weather" not in result.attrs['tpl'] \
                and "news" not in result.attrs['tpl'] and "realtime" not in result.attrs['tpl']:
            # 检测搜索结果是否为日期或时间
            sublink = result.attrs['mu']
            if sublink == 'http://time.tianqi.com/':
                sublink = 'http://time.tianqi.com/beijing'
            r = get_html(sublink).find(class_='time').get_text()
            if r is not None:
                answer['online answer'] = r
                return answer

        if result.attrs.get('mu'):  # 检测搜索结果是否为百度天气
            r = result.find(class_='op_weather4_twoicon_today OP_LOG_LINK')
            if r is not None:
                answer['online answer'] = r.get_text().strip().replace("\n", "").replace(" ", "").replace('\xa0', '\n')
                return answer

        if result.attrs.get('tpl') and 'sp_fanyi' in result.attrs['tpl']:  # 检测搜索结果是否为百度翻译
            r = result.find(class_='op_sp_fanyi_line_two')
            if r is not None:
                answer['online answer'] = r.get_text().strip()
                return answer

        if result.find("h3") is not None and result.find("h3").find("a").get_text().__contains__(u"百度百科"):
            # 检测搜索结果是否为百度百科
            url = result.find("h3").find("a")['href']
            if url is not None:
                baike_soup = get_html(url)  # 获取百度百科链接，进入百科，获取百科标题、摘要和基本信息
                r = baike_soup.find(class_='lemmaWgt-lemmaTitle lemmaWgt-lemmaTitle-')
                r1 = baike_soup.find(class_='lemma-summary')
                basicinfo = baike_soup.find_all("div", class_="basic-info cmn-clearfix")
                basicinfo2 = []  # 建立一个list存放最后的basicinfo
                for line in basicinfo:
                    i = 0
                    basicinfo_names = line.find_all("dt", class_="basicInfo-item name")  # 在basicinfo中获取全部basicinfo的项目名称
                    basicinfo_value = line.find_all("dd", class_="basicInfo-item value")  # 在basicinfo中获取全部basicinfo的项目内容
                    while i < len(basicinfo_names):  # 将basicinfo信息串联成一个字符串
                        basicinfo_value1 = re.sub('\r', '',
                                                  re.sub('\t', '', re.sub('\[\d\]', '', basicinfo_value[i].getText())))
                        basicinfo1 = basicinfo_names[i].getText() + ":" + basicinfo_value1
                        basicinfo2.append(basicinfo1)
                        i = i + 1
                if r1 is not None and r is not None:
                    r1 = r.get_text().strip().split('\n')[0] + ":" + r1.get_text().replace("\n", "").strip()
                    answer['百度百科'] = r1 + "\n" + "\n".join([re.subn('\n|\xa0', '', b)[0] for b in basicinfo2])
                    return answer

        if len(answer) == 0:  # 当百度主页面的第一条检索结果未能被上述条件捕获时，请求百度知道检索

            zhidao_url = "https://zhidao.baidu.com/search?word=" + quote(self.content)  # 百度主页面搜索的url请求
            zhidao_soup = get_html(zhidao_url)

            try:  # 一种情况是百度知道返回的搜索结果链接到了百度百科
                subsoup = get_html(zhidao_soup.find(class_='wgt-baike mb-20').find('a', href=True)['href'])
                r = subsoup.find(class_='lemmaWgt-lemmaTitle lemmaWgt-lemmaTitle-')
                r1 = subsoup.find(class_='lemma-summary')
                basicinfo = subsoup.find_all("div", class_="basic-info cmn-clearfix")
                basicinfo2 = []  # 建立一个list存放最后的basicinfo
                for line in basicinfo:
                    i = 0
                    basicinfo_names = line.find_all("dt", class_="basicInfo-item name")  # 在basicinfo中获取全部basicinfo的项目名称
                    basicinfo_value = line.find_all("dd",
                                                    class_="basicInfo-item value")  # 在basicinfo中获取全部basicinfo的项目内容
                    while i < len(basicinfo_names):  # 将basicinfo信息串联成一个字符串
                        basicinfo_value1 = re.sub('\r', '',
                                                  re.sub('\t', '', re.sub('\[\d\]', '', basicinfo_value[i].getText())))
                        basicinfo1 = basicinfo_names[i].getText() + ":" + basicinfo_value1
                        basicinfo2.append(basicinfo1)
                        i = i + 1
                if r1 is not None and r is not None:
                    r1 = r.get_text().strip().split('\n')[0] + ":" + r1.get_text().replace("\n", "").strip()
                    answer['百度百科'] = r1 + "\n" + "\n".join([re.subn('\n|\xa0', '', b)[0] for b in basicinfo2])
                    return answer
            except:
                try:  # 另一种情况是获取百度知道搜索结果中前三条带有最佳回答/推荐回答的问答
                    subsoups = [get_html(subsoup.find('a', href=True)['href']) for subsoup in zhidao_soup.find_all(class_='dt mb-8')]
                    if len(subsoups) == 0:
                        subsoups = [get_html(subsoup.find('a', href=True)['href']) for subsoup in zhidao_soup.find_all(class_='dt mb-4 line')[0:ZHIDAOMAX]]
                    p = 0
                    for subsoup in subsoups:
                        try:  # 在问答页面中，获取最佳回答，并解决文字以图片形式呈现和答案换行消失的问题
                            qtitle = re.subn('\?|\？|\！|\!|\.|\。|\·', '', subsoup.find(class_='ask-title').get_text().strip())[0] + "?"
                            ans = subsoup.find(class_='bd answer').find('pre')
                            if ans is None:
                                ans = subsoup.find(class_='bd answer').find('ul')
                            if ans is None:
                                ans = subsoup.find(class_='bd answer').find('ol')
                            anss = re.sub("<br/>", "\n", str(ans))
                            anss = re.sub("<br>", "\n", anss)
                            anss = re.sub("<p/>", "\n", anss)
                            anss = re.sub("<p>", "\n", anss)
                            anss = re.sub("<li>", "·", anss)
                            anss = re.sub("·\n", "·", anss)
                            anss = re.sub("</li>", "\n", anss)
                            anss = re.sub("\n+", "\n", anss)
                            anss_list = [el.strip().split('"')[0] for el in re.findall(r'[a-zA-z]+://[^\s]*', anss)]
                            for eln in anss_list:
                                if photo_content.get(eln) is not None:
                                    ansss = anss.replace(eln, photo_content[eln])
                                    anss = ansss
                            anss = re.sub('<img class="word-replace" src="', '', anss).replace('"/>', '').replace('查看原帖>>', '')
                            bdans = BeautifulSoup(anss, 'lxml').get_text().replace('\u3000', '')
                            if bdans != 'None' and (bdans not in badanswers) or (bdans not in qtitle):
                                answer[qtitle] = bdans

                                try:
                                    ac1 = QAQuality(qtitle, bdans).answer_judge_pro()[0]
                                except:
                                    ac1 = 0.0

                                # 直接向ES中加入已经搜索过的百度知道的问答，提高效率
                                if ac1 > SAVESCORE and len([senw for senw in sensitivewords if senw in qtitle]) == 0:

                                    # 首先确保库中没有相关数据，不重复存储
                                    _query_name_contains = query_generation([{"match_phrase": {"question": phrase}} for
                                                                             phrase in jieba.lcut(qtitle)])
                                    _searched = es.search(index='my-index', doc_type='test-type',
                                                          body=_query_name_contains)

                                    if _searched['hits']['hits'][0]['_source']['q'] != qtitle and \
                                                    _searched['hits']['hits'][0]['_source']['a'] != bdans:

                                        load_es = LoadElasticSearch()  # 创建插入新数据的实例
                                        load_es.add_date({'question': qtitle, 'answer': bdans})  # 插入新数据

                                        print("第" + str(p + 1) + "条数据已经加入本地库")
                                p += 1

                        except:
                            pass
                except:
                    pass

        if len(answer) == 0:  # 实时搜索未能获取答案，返回安全回答
            answer["安全回答"] = "我还在学习中，暂时无法回答您的问题"

        return answer


# -----------------问答函数----------------- #

def question_answer(new_question):  # 用户输入一句新问题

    return_dic = {}  # 本函数最终返回的精选问答对

    # 问题实时性检测
    for senword in sensitivewords:  # 如果文本中含有实时词，则直接启动实时搜索

        if senword in new_question:

            OS = OnlineSearch(new_question)  # 启动在线搜索
            ans = OS.Baidu_Search()  # 在线搜索的结果

            if len(ans) > 0:  # 若在线搜索能给出回答
                return json.dumps(return_dic)

            return_dic["安全回答"] = "我还在学习中，暂时无法回答您的问题"
            return json.dumps(return_dic)

    # 经加权后的分词、去停用词后的用户新问题主要内容
    text_main_content = word_weight_exchange(jieba.lcut(new_question), new_question)

    # 封装成es query需要的样式，match_phrase可保证分词后的词组不被再拆散成字
    input_string = [{"match_phrase": {"question": phrase}} for phrase in text_main_content]
    _query_name_contains = query_generation(input_string)  # 生成ES查询
    _searched = es.search(index='my-index', doc_type='test-type', body=_query_name_contains)  # 执行查询

    # 无查询结果，直接返回安全回答
    if len(_searched['hits']['hits']) == 0:
        return_dic["安全回答"] = "我还在学习中，暂时无法回答您的问题"
        return json.dumps(return_dic)

    _selected = {}  # 存放精选后的候选问题-答案的集合
    for l1 in _searched['hits']['hits']:  # 问题精选

        # 如果新问题与候选问题完全一致，则直接返回
        if l1['_source']['question'] == new_question:
            return_dic[l1['_source']['question']] = l1['_source']['answer']
            return json.dumps(return_dic)

        # 生成与新问题、候选问题相关的基础运算变量
        qa_pair = (l1['_source']['question'], l1['_source']['answer'])
        # 经加权的分词、去停用词后的候选问题主要内容
        question_main_content = word_weight_exchange(jieba.lcut(l1['_source']['question']), l1['_source']['question'])

        # 新问题与候选问题句子相似度similarity
        with tf.Session() as sess:
              saver = tf.train.Saver()
              saver.restore(sess, saverpath)
              prediction_result = \
                  sess.run(prediction, 
                           feed_dict={xs: s1_s2_simipics(text_main_content, question_main_content, MAX_LENTH), 
                                      keep_prob: 1.0})
        similarity = prediction_result.tolist()[0][-1]

        try:  # 答案排序模块对新问题和候选答案的组合给出的分数
            qa_quality_score = float(QAQuality(new_question, l1['_source']['answer']).answer_judge_pro()[0])
        except:
            # 有时由于答案过短，答案排序模块无法完成计算，则直接置0
            qa_quality_score = 0.0

        # 最终精选得分
        _selected[qa_pair] = SIMI * similarity + QUA * qa_quality_score

    # 精选top1
    selected = sorted(_selected.items(), key=lambda item: item[1], reverse=True)[0]
    if selected[1] > THRESHOLD:
        return_dic[selected[0][0]] = selected[0][1]
        return json.dumps(return_dic)

    if len(return_dic) == 0:  # 若经过精选，所有粗选结果均不符合要求

        OS = OnlineSearch(new_question)  # 启动在线搜索
        ans = OS.Baidu_Search()  # 在线搜索的结果

        if len(ans) > 0:  # 若在线搜索能给出回答
            return json.dumps(return_dic)

        return_dic["安全回答"] = "我还在学习中，暂时无法回答您的问题"
        return json.dumps(return_dic)

    return json.dumps(return_dic)


# ------------------主函数------------------ #

if __name__ == '__main__':  # 供测试时使用

    '''
    # 往ES缓存中插入全量数据，使用时只需取消注释，并将主函数内其它部分注释即可
    # 重复插入数据时，需先到ES包内将整个data文件直接删除；若需要缩小空间，ES包内的logs文件下带日期的log可以删除

    # 使用时间数据随机初始化索引
    from datetime import datetime
    es.index(index="my-index", doc_type="test-type", body={"any": "data", "timestamp": datetime.now()})

    load_es = LoadElasticSearch()

    i = 0
    row_obj_list = []
    for record in open(cur_dir + '/your-qadata-path', 'r').read().strip().split('\n====================\n'):
        record_list = record.strip().split('分隔符')
        try:
            questions = record_list[1]
            answers = record_list[2]
            qa_pairs = [(x, y) for x in questions for y in answers]
            for qa in qa_pairs:
                row_obj_list.append({'question': qa[1], 'answer': qa[2]})
            i += 1
        except:
            pass
            
    load_es.add_date_bulk(row_obj_list)  # 批量加载
    '''

    # 执行整个问答系统
    t = 0
    k = 0
    while True:

        s_input = input('请输入：')  # 输入文本
        start = time.time()  # 开始查找
        result = json.loads(question_answer(s_input))
        for key in result:
            print(key, result[key])
        end = time.time()  # 结束查找及输出
        print("=" * 40)
        print("耗时：" + str(round((end - start), 3)) + "s")

    # '''