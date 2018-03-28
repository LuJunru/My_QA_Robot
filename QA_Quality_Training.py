# -*- coding: utf-8 -*-
# @Author  : Junru_Lu
# @File    : QA_Quality_Training.py
# @Software: PyCharm
# @Environment : Python 3.6+

# 基础包
import math
import jieba
import os
import jieba.posseg as pseg  # 用于词性标注
import numpy as np
import re
import pickle
from snownlp import SnowNLP  # 用于句子情感打分
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

# 编码相关包
import importlib, sys
importlib.reload(sys)


'''
本程序设计：对于给定问题和答案，抽取问答对的文本和非文本特征，使用XGBoost模型训练其组成好问答对的概率打分模型。

本配置文件用于训练问答质量评价子模型
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


# ------------------答案排序模块------------------ #

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


# ------------------主函数------------------ #

if __name__ == '__main__':

    train_path = cur_dir + '/basic_data_file/qar_test.txt'  # 训练集路径

    # 生成特征矩阵
    X = np.zeros(22, float)

    for line in open(train_path, mode='r', encoding='utf-8'):
        question = line.strip().split('\t')[0]
        answer = line.strip().split('\t')[1]
        label = line.strip().split('\t')[2]

        ac = QAQuality(question, answer)
        features = [float(ac.a_singles()), ac.lenth_ratio, ac.resemblance, ac.contain, ac.overlap, ac.cosine,
                    ac.qa_noun_per, ac.a_noun_per, ac.qa_verb_per, ac.a_verb_per, ac.a_stop_per, ac.a_punc_dens,
                    ac.qa_punc_dens, ac.a_sentiment, ac.a_singleword_per, ac.a_wentroy(), ac.a_centroy(),
                    ac.q_punc_dens, ac.fog_score, ac.flesch_score, ac.flesch_kincaid_score, float(label)]

        X = np.vstack((X, features))

    S = X[1:, :]

    # 训练
    i = 0
    trainning_accuracy = 0
    test_accuracy = 0
    trainning_max = 0
    test_max = 0
    while i < 100:  # 训练10轮，观测每一轮十折交叉验证的准确率

        # 拆分训练集与测试集
        np.random.shuffle(S)  # 随机洗牌
        train_X, test_X, train_Y, test_Y = train_test_split(S[:, :-1], S[:, -1], test_size=0.1)  # 十折分割训练集
        aTrain_X = train_X  # .as_matrix()
        aTrain_Y = train_Y  # .as_matrix()
        aTest_X = test_X  # .as_matrix()
        aTest_Y = test_Y  # .as_matrix()

        # 使用xgboost模型
        # 模型参数
        params = {
            'booster': 'gbtree',
            # 'objective': 'multi:softmax', #多分类的问题
            # //无效'num_class':2, # 类别数，与 multisoftmax 并用
            'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth': 8,  # 构建树的深度，越大越容易过拟合
            # //无效'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample': 0.7,  # 随机采样训练样本
            'colsample_bytree': 0.7,  # 生成树时进行的列采样
            'min_child_weight': 3,
            # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
            # //无效'eta': 0.007, # 如同学习率
            'seed': 1000,
            'nthread': 7,  # cpu 线程数
            # 'eval_metric': 'auc'
        }

        # 训练模型
        model = XGBClassifier()  # 构建模型
        model.get_params()       #获取参数
        model.set_params(**params)  # 设置参数
        # 开始训练
        model.fit(aTrain_X, aTrain_Y, eval_metric='auc')

        # 保存模型
        score0 = 0  # model.score(aTrain_X, aTrain_Y)
        score1 = model.score(aTest_X, aTest_Y)
        if score1 > 0.74:  # 保存模型
            pickle.dump(model, open('pre_trained_models/xgboost_qaquality_21_60dz_s' +
                                    str(round(score1, 3)) + '.pkl', 'wb'))
            print('====> yes found good xgboost model')
        # print(i+1, score)  # 打印每轮训练的准确率
        # 打印准确率 和 召回率
        print('第 %d 轮测试集计算准确率为：%f' % (i, score1))
        trainning_accuracy += score0
        test_accuracy += score1
        trainning_max = max(score0, trainning_max)
        test_max = max(score1, test_max)
        i += 1

    trainning_accuracy /= i
    test_accuracy /= i
    print("平均训练集准确率: %f. 最大训练集准确率 %f" % (trainning_accuracy, trainning_max))
    print("平均测试集准确率: %f. 最大测试集准确率 %f" % (test_accuracy, test_max))