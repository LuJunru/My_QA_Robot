# -*- coding: utf-8 -*-
# @Author  : Junru_Lu
# @File    : QA_Quality_Module.py
# @Software: PyCharm
# @Environment : Python 3.6+

# 基础包
import jieba.posseg as pseg  # 用于词性标注
import jieba
from snownlp import SnowNLP  # 用于句子情感打分
import re
import math
import pickle
import os
from gensim.models import keyedvectors

# 编码相关包
import importlib, sys
importlib.reload(sys)


'''
本配置文件用于测试问答质量评价子模型
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

# 基于10G中文维基训练的60维词向量
word_vectors = keyedvectors.KeyedVectors.load(cur_dir + '/Word Embedding/Word60.model')  # 加载预先训练好的词向量

# 预测问答对属于"好"的概率的问答质量评价子模块（XGB）
xgb = pickle.load(open(cur_dir + '/pre_trained_models/xgboost_qaquality_21_60dz_s0.745.pkl', 'rb'))


# ------------------候选答案重排序模块------------------ #

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


# ------------------主函数------------------ #

if __name__ == '__main__':  # 供测试时使用

    while True:
        q_input = input('请输入问题：')
        a_input = input('请输入回答：')
        print(float(QAQuality(q_input, a_input).answer_judge_pro()[0]))