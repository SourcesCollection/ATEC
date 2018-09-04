[金融大脑-金融智能NLP服务](https://dc.cloud.alipay.com/index#/topic/intro?id=3)
===
## 赛题任务描述

问题相似度计算，即给定客服里用户描述的两句话，用算法来判断是否表示了相同的语义。

示例：

“花呗如何还款” --“花呗怎么还款”：同义问句
“花呗如何还款” -- “我怎么还我的花被呢”：同义问句
“花呗分期后逾期了如何还款”-- “花呗分期后逾期了哪里还款”：非同义问句
对于例子a，比较简单的方法就可以判定同义；对于例子b，包含了错别字、同义词、词序变换等问题，两个句子乍一看并不类似，想正确判断比较有挑战；对于例子c，两句话很类似，仅仅有一处细微的差别 “如何”和“哪里”，就导致语义不一致。


## cdssm模型

### Setup
* python2.7
* pytorch 0.4.0
* jieba
* sklearn
* torchtext 0.2.3(比赛评测系统没有装)


### 主要文件

* train_cdssm.py : 包括train/eval/predict等主要函数
* model_cdssm.py : CDSSM模型
* main_cdssm.py : 用于训练模型
* main_cdssm_stack.py : 得到cdssm模型的预测结果


## xgboost

### Setup
* python2.7
* xgboost
* jieba
* pandas
* gensim
* sklearn

### 主要文件

#### 词典/词向量文件
* userdict.txt : 根据训练语料构建的自定义用户词典
* word2vec_vectors.txt : 用word2vec工具预训练的word embedding
* golve_vectors.txt : 用glove工具包预训练的词向量

#### 提取各种feature的py文件
* cut_utils.py : 对数据进行分词处理
* string_diff.py : 字符串长度比较
* string_distance.py : 编辑距离、jaccard距离、jaro_winkler相似度等
* word2vec_utils.py: 首先得到每个词的词向量（glove/word2vec）,然后根据tfidf/bow加权取平均得到句子向量，最后用scipy.spatial.distance计算各种向量距离
* doc2vec_model.py : 训练doc2vec模型
* doc2vec_infer.py : 用训练好的doc2vec模型得到句子向量，然后计算各种向量距离
* n-grams.py :  得到句子的n-gram，计算各种集合距离
* train_xgb.py : 训练xgb模型
* main_xgb_stack.py : 得到xgb模型的预测结果










