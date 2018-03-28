# Introduction
  * 本项目为基于社区问答的单轮检索式通用知识问答工程实现，原理及设计参考本项目附带的PDF文件

# Usage

  * Environment and requirements<br>
    1. 环境要求: python3.6+<br>
    2. 安装包要求: elasticsearch, jieba, tensorflow, snownlp, numpy, json, pickle, gensim, pyltp, importlib等<br>

  * Files<br>
    1. basic_data_files: 所有相关数据文件，其中cnn_train_data指cnn模型训练语料，keywords指项目核心词表，qadata指本地知识数据<br>
                         qar_test指用于问答质量子模块训练用的10W百度知道问答数据，stopwords指停用词表，fieldwords指领域词<br>
                         photo_content指的是在爬取百度知道内容时，为应对文字图片化反爬机制收集的文字-图片对照表<br>
    2. CNN_logs: 存放训练基于CNN的问题相似度模型时产生的Tensorboard日志<br>
                 按如下方法查看Tensorboard:<br>
                   1. 模型训练结束后，在终端输入：tensorboard --logdir='本次训练对应的日志地址'<br>
                   2. 'Starting TensorBoard 41 on port 6006'这句话出现后，将显示的网址复制到浏览器地址栏<br>
                   3. 如果没有出现网址，在地址栏输入'localhost:6006'即可<br>
    3. elasticsearch-6.2.2: 存放ES中插入的数据及对应索引<br>
    4. ltp_data_v3.4.0: 存放使用pyltp提取ner和主谓宾时的预加载模型，请参照mainprogram.py第79-80行注释下载<br>
    5. pre_trained_models: 预训练好的模型，CNN_N指Maxlength为N的CNN模型，xgboost_qaquality_21_60dz_s0.745.pkl<br>
                           指使用21维特征和60维词向量训练的十折交叉准确率为0.745的问答质量评价子模型<br>
    6. Word Embedding: 词向量，请参照mainprogram.py第72-73行注释下载<br>

  * Programs<br>
    1. QA_Quality_Training.py: 编写、训练问答质量评价子模型<br>
    2. QA_Quality_Module.py: 测试问答质量评价子模型<br>
    3. CNN_SentenceSimilarity_Training.py: 编写、训练问题相似度模型<br>
    4. CNN_SentenceSimilarity_Module.py: 测试、调用问题相似度模型<br>
    5. Online_Search_Module.py: 编写、测试和调用在线搜索模型，缓存机制请参考mainprogram.py第704-722行<br>
    6. mainprogram.py: 主程序<br>

# More: junru.lu@enactusuir.org<br>