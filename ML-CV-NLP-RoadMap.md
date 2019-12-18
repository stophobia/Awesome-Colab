
# CV课程大纲Syllabus

## 1深度学习基础篇

### 实战案例
```
使用Tensorflow和PyTorch构建自己的神经网络

Python与Numpy摸底考试
```
### 核心知识点
```
逻辑回归与梯度下降法

凸函数与凸优化

BP算法的讲解

Tensorflow、Kersa、PyTorch的使用教程

GPU配置，安装，训练模型及评估

Conv2D，Conv2DTranspos详解

Dropout, Batch Normalization详解

如何自定义网络层，损失函数

深度学习中的调参技术. 解决过拟合与欠拟合

激活函数详解：Sigmoid, Softmax, tanh, softplus, ReLU, hard_sigmoid, linear, exponential, LeakyReLU, PReLU, ELU.

优化器详解：GD，SGD，MiniBatch GD，Nesterov，RMSprop，Adagrad，Adadelta，Adam.
```
## 2多模态模型

### 实战案例
```
利用Pytorch搭建VGG16卷积神经网络

搭建基于注意力机制的LSTM网络

基于Bert的Image Captioning项目实战
```
### 核心知识点
```
什么是多模态学习？

VGG16以及基于Pytorch的实现

迁移学习详解

RNN以及BPTT，梯度消失问题

LSTM, GRU详解

注意力机制

SkipGram

Elmo, Bert, XLNet

Beam Search, Greedy Decoding

BLEU评价指标

搭建系统过程中用到的工程技巧
```
## 3物体识别

### 实战案例
```
各类卷积神经网络结构剖析以及实现

不同环境下交通指示牌的识别

定制自己的神经网络
```
### 核心知识点
```
CNN卷积层工作原理剖析

卷积核尺寸，卷积步长，边界填充，输出通道，输出特征图，视场计算

LeNet-5

AlexNet

ZFNet

GoogleNet/Inception

VGGNet

ResNet

Fully-Convolutional Network

DenseNet

图像增强技术

图像增加噪声与降噪
```
## 4目标检测技术

### 实战案例
```
利用SSD模型完成自动驾驶中的车辆，行人，交通灯的定位

人脸关键点定位

图像分割任务
```
### 核心知识点
```
R-CNN，Fast R-CNN, Faster RCNN

Region Proposal，Region Proposal Network

One-Stage物体检测网络模型

SSD模型

Anchor的内涵与工作原理

IoU (Intersection Over Union)

Hard Negative Mining

Non-Max Suppression

OpenCV Haar小波滤波器

OpenCV Adaboost

图像分割 Dense Prediction

Unet，Up-Conv

Transpose Convolution/Deconvolution
```
## 5自动驾驶

### 实战案例
```
自动驾驶方向盘转向预测

自动驾驶中行车道的检测
```
### 核心知识点
```
自动驾驶技术介绍

如何使用多个摄像头

DataGenerator技术

图像的空间域

频率域滤波

图像色彩变换

边缘检测

Hough Transform用于检测图像中的几何形状物体
```
## 6图像生成

### 实战案例
```
Python编写GAN生成手写数字图像

图像风格迁移：将自拍照转换为毕加索油画
```
### 核心知识点
```
GAN 生成对抗网络

Generator，Discriminator网络结构

GAN的优化以及实现

GAN与其他生成模型的比较

图像风格化迁移的实现

Gram Matrix图像风格表达
```
## 7低能耗神经网络

### 实战案例
```
二值化神经网络识别交通指示牌

低能耗网络完成自动驾驶方向盘转向预测
```
### 核心知识点
```
如何降低神经网络的耗能

Binarized Neural Network

MobileNet

ShuffleNet

EffNet

神经网络的节能原理

Depth-wise Separable Convolution

Spatial Separable Convolution

Grouped Convolution

Channel Shuffle
```
## 8新颖网络结果

### 实战案例
```
双子网络完成人脸识别项目

使用胶囊网络进行手写数字的识别和重建
```
### 核心知识点
```
One-Shot Learning

Siamese Network 双子网络

人脸识别关键技术

CapsuleNet 胶囊网络

胶囊替代神经元旦原理
```
## 9Capstone 开放式项目(Optional)
```
往期学员项目展示
利用图像识别对英雄联盟游戏中的英雄走位进行定位，从而分析战队战术

通过图像技术对二手车市场的车辆进行伤损评定

探索财务系统与OCR技术的结合

利用图像识别技术自动批改数学卷子

项目展示
什么是Capstone项目？

项目介绍
开放式项目又称为课程的capstone项目。作为 课程中的很重要的一部分，可以选择work on 一个具有挑战性的项目。通过此项目，可以深 入去理解某一个特定领域，快速成为这个领域 内的专家，并且让项目成果成为简历中的一个 亮点。

项目流程
Step 1: 组队

Step 2: 立项以及提交proposal

Step 3: Short Survey Paper

Step 4: 中期项目Review Step

5: 最终项目PPT以及代码提交

Step 6: 最终presentation

Step 7: Technical Report/博客

结果输出
完整PPT、代码和Conference-Style Technical Report 最为项目的最后阶段，我们 将组织学员的presentation分享大会。借此我 们会邀请一些同行业的专家、从业者、企业招 聘方、优质猎头资源等共同参与分享大会。
```



# ML课程大纲Syllabus

## 机器学习基础与凸优化

### 实战案例
```
基于QP的股票投资组合策略设计

基于LP的短文本相似度计算

基于KNN的图像识别
```
### 核心知识点
```
KNN算法，Weighted KNN算法

Approximated KNN算法

KD树，近似KD树

Locality Sensitivity Hashing

线性回归模型

Bias-Variance Trade-off

正则的使用：L1, L2, L-inifity Norm

LASSO， Coordinate Descent，ElasticNet

逻辑回归与最大似然

随机梯度下降法与小批量梯度下降法

多元逻辑回归模型

凸集，凸函数

凸函数与判定凸函数

Linear/Quadratic/Integer Programming

对偶理论，Duality Gap，KKT条件

Projected Gradient Descentg

迭代式算法的收敛分析
```

## SVM与集成模型

### 实战案例
```
基于XGBoost的金融风控模型

基于PCA和Kernel SVM的人脸识别

基于Kernal PCA和Linear SVM的人脸识别
```
### 核心知识点
```
Max-Margin的方法核心思想

线性SVM的一步步构建

Slack Variable以及条件的松弛

SVM的Dual Formulation

Kernelized SVM

不同核函数的详解以及使用

核函数设计以及Mercer's Theorem

Kernelized Linear Regression

Kernelized PCA, Kernelized K-means

集成模型的优势

Bagging, Boosting, Stacking

决策树以及信息论回顾

随机森林，完全随机森林

基于残差的提升树训练思想

GBDT与XGBoost

集成不同类型的模型

VC理论
```
## 无监督学习与序列模型

### 实战案例
```
基于HMM和GMM的语音识别

基于聚类分析的用户群体分析

基于CRF的命名实体识别
```
### 核心知识点
```
K-means算法， K-means++

EM算法以及收敛性

高斯混合模型以及K-means

层次聚类算法

Spectral Clustering

DCSCAN

隐变量与隐变量模型

HMM的应用以及参数

条件独立、D-separation

基于Viterbi的Decoding

Forward/Backward算法

基于EM算法的参数估计

有向图与无向图模型区别

Log-Linear Model

Feature Function的设计

Linear CRF以及参数估计
```

## 深度学习

### 实战案例
```
基于Seq2Seq和注意力机制的机器翻译

基于TransE和GCN的知识图谱推理

基于CNN的人脸关键点检测
```
### 核心知识点
```
神经网络与激活函数

BP算法

卷积层、Pooling层、全连接层

卷积神经网络

常用的CNN结构

Dropout与Bath Normalization

SGD、Adam、Adagrad算法

RNN与梯度消失

LSTM与GRU

Seq2Seq模型与注意力机制

Word2Vec, Elmo, Bert, XLNet

深度学习中的调参技术

深度学习与图嵌入(Graph Embedding)

Translating Embedding (TransE)

Node2Vec

Graph Convolutional Network

Structured Deep Network Embedding

Dynamic Graph Embedding
```

## 推荐系统与在线学习

### 实战案例
```
使用Gradient Boosting Tree做基于 interaction 与 content的广告推荐

使用深度神经网络做基于interaction 与 content的推荐

LinUCB做新闻推荐, 最大化rewards
```
### 核心知识点
```
基于内容的推荐算法

基于协同过滤的推荐算法

矩阵分解

基于内容的Gradient Tree

基于深度学习的推荐算法

冷启动问题的处理

Exploration vs Exploitation

Multi-armed Bandit

UCB1 algorithm，EXP3 algorithm

Adversarial Bandit model

Contexulalized Bandit

LinUCB
```
### 贝叶斯模型

### 实战案例
```
基于Bayesian LSTM的文本分析

使用无参主题模型做文本分类

基于贝叶斯模型实现小数量的图像识别
```
### 核心知识点
```
主题模型（LDA) 以及生成过程

Dirichlet Distribution, Multinomial Distribution

蒙特卡洛与MCMC

Metropolis Hasting与Gibbs Sampling

使用Collapsed Gibbs Sampler求解LDA

Mean-field variational Inference

使用VI求解LDA

Stochastic Optimization与Bayesian Inference

利用SLGD和SVI求解LDA

基于分布式计算的贝叶斯模型求解

随机过程与无参模型（non-parametric)

Chinese Retarant Process

Stick Breaking Process

Stochastic Block Model与MMSB

基于SGLD与SVI的MMSB求解

Bayesian Deep Learning模型

Deep Generative Model
```

## 增强学习与其他前沿主题

### 实战案例
```
基于GAN的图像生成

基于VAE的文本Style Transfer

可视化机器翻译系统
```
### 核心知识点
```
Policy Learning

Deep RL

Variational Autoencoder(VAE)与求解

隐变量的Disentangling

图像的生成以及Disentangling

文本的生成以及Disentangling

Generative Adversial Network(GAN)

CycleGan

深度学习的可解释性

Deconvolution与图像特征的解释

Layer-wise Propagation

Adversial Machine Learning

Purturbation Analysis

Fair Learning
```


# NLP课程大纲Syllabus

## 1算法与机器学习基础

### 实战案例
```
基于Sparse Quadratic Programming的股票投资组合优化策略编写

基于Earth Mover's Distance的短文本相似度计算

基于Projected Gradient Descent和非负矩阵分解的词向量学习

基于Linear Programming的机票定价系统

基于DTW的文本相似度分析
```
### 核心知识点
```
时间复杂度，空间复杂度分析

Master's Theorem，递归复杂度分析

动态规划以及Dynamic Time Warpping

Earth Mover's Distance

维特比算法

LR、决策树、随机森林、XGBoost

梯度下降法、随机梯度下降法、牛顿法

Projected Gradient Descent

L0, L1, L2, L-Infinity Norm

Grid Search, Bayesian Optimization

凸函数、凸集、Duality、KKT条件

Linear SVM、Dual of SVM

Kernel Tick, Mercer's Theorem

Kernelized Linear Regression、Kernelized KNN

Linear/Quadratic Programming

Integer/Semi-definite Programming

NP-completeness/NP-hard/P/NP

Constrained Relaxation、Approximate Algorithm

Convergence Analysis of Iterative Algorithm
```
## 2语言模型与序列标注

### 实战案例
```
基于无监督学习方法的问答系统搭建

基于监督学习的Aspect-Based 情感分析系统搭建

基于CRF、LSTM-CRF、BERT-CRF 的命名实体识别应用

基于语言模型和Noisy Channel Model的拼写纠错
```
### 核心知识点
```
文本预处理技术（tf-idf，Stemming等）

文本领域的特征工程

倒排表、信息检索技术

Noisy Channel Model

N-gram模型，词向量介绍

常见的Smoothing Techniques

Learning to Rank

Latent Variable Model

EM算法与Local Optimality

Convergence of EM

EM与K-Means, GMM

Variational Autoencoder与Text Disentangling

有向图与无向图模型

Conditional Indepence、D-separation、Markov Blanket

HMM模型以及参数估计

Viterbi、Baum Welch

Log-Linear Model与参数估计

CRF模型与Linear-CRF

CRF的Viterbi Decoding与参数估计
```
## 3信息抽取、词向量与知识图谱

### 实战案例
```
利用非结构化数据和信息抽取技术构建知识图谱

任务导向型聊天机器人的搭建

包含Intent与Entity Extraction的NLU模块实现

基于SkipGram的推荐系统实现（参考Airbnb论文）
```
### 核心知识点
```
命名实体识别技术

信息抽取技术

Snowball, KnowitAll, RunnerText

Distant Supervision, 无监督学习方法

实体统一、实体消歧义、指代消解

知识图谱、实体与关系

词向量、Skip-Gram、Negative Sampling

矩阵分解、CBOW与Glove向量

Contexualized Embedding与ELMo

KL Divergence与Gaussian Embedding

非欧式空间与Pointcare Embedding

黎曼空间中的梯度下降法

知识图谱嵌入技术

TransE, NTN 的详解

Node2Vec详解

Adversial Learning与KBGAN
```
## 4深度学习与NLP

### 实战案例
```
利用纯Python实现BP算法

基于Seq2Seq+注意力机制、基于Transformer的机器翻译系统

基于Transformer的闲聊型聊天机器人

基于BI-LSTM-CRF和BERT-BiLSTM-CRF在命名实体中的比较

利用Laywer-wise RP可视化端到端的机器翻译系统
```
### 核心知识点
```
Pytorch与Tensorflow详解. 表示学习，分布式表示技术

文本领域中的Disentangling

深度神经网络与BP算法详解

RNN与Vanishing/Exploding Gradient

LSTM与GRU

Seq2Seq与注意力机制

Greedy Decoding与Beam Search

BI-LSTM-CRF模型

Neural Turing Machine

Memory Network

Self Attention，Transformer以及Transformer-XL.

Bert的详解

BERT-BiLSTM-CRF

GPT，MASS, XLNet

Low-resource learning

深度学习的可视化

Laywer-wise Relevance Propagation
```
## 5贝叶斯模型与NLP

### 实战案例
```
利用Collapsed Gibbs Sampler和SGLD对主题模型做Inference

基于Bayesian-LSTM的命名实体识别

利用主题模型做文本分类在

LDA的基础上修改并搭建无监督情感分析模型
```
### 核心知识点
```
概率图模型与条件独立

Markov Blanket

Dirichlet分布、Multinomial分布

Beta分布、Conjugate Prior回顾

Detail Balance

主题模型详解

MCMC与吉布斯采样

主题模型与Collapsed Gibbs Sampling

Metropolis Hasting, Rejection Sampling

Langevin Dyamics与SGLD

分布式SGLD与主题模型

Dynamic Topic Model

Supervised Topic Model

KL Divergence与ELBO

Variantional Inference, Stochastic VI

主题模型与变分法

Nonparametric Models

Dirichlet Process

Chinese Restarant Process

Bayesian Deep Neural Network

VAE与Reparametrization trick

Bayesian RNN/LSTM

Bayesian Word2Vec

MMSB
```
## 6Capstone 开放式项目(Optional)
```
往期学员项目展示
搭建辅助医疗诊断的智能问答系统

LDA主题模型的平滑处理方法研究

基于知识驱动的对话聊天机器人

基于深度学习的命名实体识别研究

项目展示
什么是Capstone项目？

项目介绍
开放式项目又称为课程的capstone项目。作为 课程中的很重要的一部分，可以选择work on 一个具有挑战性的项目。通过此项目，可以深 入去理解某一个特定领域，快速成为这个领域 内的专家，并且让项目成果成为简历中的一个 亮点。

项目流程
Step 1: 组队

Step 2: 立项以及提交proposal

Step 3: Short Survey Paper

Step 4: 中期项目Review Step

5: 最终项目PPT以及代码提交

Step 6: 最终presentation

Step 7: Technical Report/博客

结果输出
完整PPT、代码和Conference-Style Technical Report 最为项目的最后阶段，我们 将组织学员的presentation分享大会。借此我 们会邀请一些同行业的专家、从业者、企业招 聘方、优质猎头资源等共同参与分享大会。

Capstone项目选题方向有哪些？
学员可以选择自己感兴趣的项目来做，可以是 自己在公司中遇到的问题，也可以纯粹中自己 的兴趣出发，也可以是偏学术性的。主要分成 四个方向：应用型、工具型、论文复现性/总 结型的、研究性质的。

应用型项目
选定一个特定领域（比如医疗，汽车行业， 法律等等），并构建此领域的知识图谱， 然后基于知识图谱搭建问答系统。 此项目 的难点在于数据获取这一端。

工具型项目
课程里涉及到了很多NLP核心的技术，比 如拼写纠错，分词，NER识别，关系抽 取，POS识别等等，而且市面上也有一些 开源的工具比如HanNLP, 哈工大NLP, 结 巴等等。 有没有可能自己写出在某些问题 上更好的NLP相关的API呢，然后再开源？

论文复现性/总结型项目
我们可以选择一些比较前沿的技术而且 “重要”的论文来做复现，可以偏向于是系 统实现，也可以是对某一个技术的总结。 例如：利用深度增强学习的方式来搭建 聊天机器人(参考https://arxiv.org/pdf/ 1709.02349.pdf)。

研究型项目
研究是具有挑战性的，其中很重要的问题 是选题。基于个人的兴趣，narrow down 到一个特定的问题。我们将在研究的过程 中给一些思路上的指导，最终达到发表一 篇论文的目的。
```

# 机器学习高阶训练营
```
第1章: 课程介绍
任务1： MLcamp_course_info
19:33 
第2章: 凸优化基础 （11.3）
 任务2： 课程介绍
16:58 
 任务3： 凸集、凸函数、判定凸函数
01:09:38 
任务4： Transportation Problem
23:21 
任务5： Portfolio Optimization
33:15 
任务6： Set Cover Problem
27:20 
任务7： Duality
01:28:20 
任务8： 答疑部分
35:16 
第3章: 20191109 Paper 从词嵌入到文档距离
任务9： 从词嵌入到文档距离01
47:14 
任务10： 从词嵌入到文档距离02
44:52 
第4章: 20191110 SVM
任务11： KKT Condition
15:42 
任务12： SVM 的直观理解
13:46 
任务13： SVM 的数学模型
20:44 
任务14： 带松弛变量的SVM
26:46 
任务15： 带Kernel的SVM
34:39 
任务16： SVM的SMO的解法
22:53 
任务17： 使用SVM支持多个类别
07:35 
任务18： Kernel Linear Regression
12:29 
任务19： Kernel PCA
24:28 
任务20： 交叉验证
06:28 
任务21： VC维
05:03 
任务22： 直播答疑01
33:06 
任务23： 直播答疑02
44:39 
第5章: 20191110 Review
任务24： LP实战01
31:29 
任务25： LP实战02
22:23 
任务26： LP实战03
21:53 
任务27： Hard，NP Hard-01
19:44 
任务28： Hard，NP Hard-02
24:32 
任务29： Hard，NP Hard-03
25:52 
第6章: 20191117 简单机器学习算法与正则
任务30： 引言
04:53 
任务31： 线性回归
33:19 
任务32： Basis Expansion
13:28 
任务33： Bias 与 Variance
20:25 
任务34： 正则化
33:29 
任务35： Ridge, Lasso, ElasticNet
08:22 
任务36： 逻辑回归
46:46 
任务37： Softmax 多元逻辑回归
10:09 
任务38： 梯度下降法
13:23 
第7章: 20191117 Review
任务39： SVM人脸识别结合Cross-validation交叉验证01
20:49 
任务40： SVM人脸识别结合Cross-validation交叉验证02
21:18 
任务41： SVM人脸识别结合Cross-validation交叉验证03
27:24 
任务42： SVM人脸识别结合Cross-validation交叉验证04
34:21 
任务43： 模型评估方法和SVM做人脸识别01
25:52 
任务44： 模型评估方法和SVM做人脸识别02
19:17 
任务45： 模型评估方法和SVM做人脸识别03
32:25 
第8章: 20191124 Review
任务46： PCA和LDA的原理和实战01
28:28 
任务47： PCA和LDA的原理和实战02
18:09 
任务48： PCA和LDA的原理和实战03
30:20 
任务49： Softmax with Cross Entropy01
31:57 
任务50： Softmax with Cross Entropy02
40:01 
任务51： Softmax with Cross Entropy03
22:21 
第9章: 20191124 Paper
任务52： Kernel Logistic Regression and the Import Vec01
33:21 
任务53： Kernel Logistic Regression and the Import Vec02
36:38 
第10章: 20191124 LDA.EnsembleMethod
任务54： LDA 作为分类器
43:14 
任务55： LDA 作为分类器答疑
42:22 
任务56： LDA 作为降维工具
14:26 
任务57： Kernel LDA 5 Kernel LDA答疑
02:51 
任务58： Ensemble Majority Voting
17:37 
任务59： Ensemble Bagging
12:27 
任务60： Ensemble Boosting
28:44 
任务61： Ensemble Random Forests
05:46 
任务62： Ensemble Stacking
10:38 
任务63： 答疑
54:49 
第11章: 20191201 集成模型
任务64： 决策树的应用
29:17 
任务65： 集成模型
26:10 
任务66： 提升树
21:21 
任务67： 目标函数的构建
18:42 
任务68： Additive Training
15:07 
任务69： 使用泰勒级数近似目标函数
18:25 
任务70： 重新定义一棵树
40:36 
任务71： 如何寻找树的形状
37:54 
第12章: 20191130 paper XGBoost
任务72： XGBoost-01
23:43 
任务73： XGBoost-02
24:51 
任务74： XGBoost-03
28:15 
第13章: 20191130 Review
任务75： XGBoost的代码解读 工程实战-01
33:11 
任务76： XGBoost的代码解读 工程实战-02
24:59 
任务77： XGBoost的代码解读 工程实战-03
26:22 
任务78： 理解和比较XGBoost GBDT LightGBM-01
33:59 
任务79： 理解和比较XGBoost GBDT LightGBM-02
32:22 
任务80： 理解和比较XGBoost GBDT LightGBM-03
49:02 
第14章: 20191207 Paper LightGBM
任务81： LightGBM-01
25:21 
任务82： LightGBM-02
28:57 
任务83： LightGBM-03
32:22 
第15章: 20191208 k-MEANS.EM.DBSCAN v2
任务84： 聚类算法介绍 K-Means 算法描述
17:32 
任务85： K-Means 的特性 K-Means++
39:43 
任务86： EM 算法思路
19:16 
任务87： EM 算法推演
18:41 
任务88： EM 算法的收敛性证明
11:40 
任务89： EM 与高斯混合模型
40:24 
任务90： EM 与 KMeans 的关系
05:21 
任务91： DBSCAN聚类算法
29:23 
任务92： 课后答疑
21:51 
第16章: 20191208 Review
任务93： kaggle广告点击欺诈识别实战-01
28:25 
任务94： kaggle广告点击欺诈识别实战-02
36:58 
任务95： kaggle广告点击欺诈识别实战-03
27:44 
任务96： kaggle广告点击欺诈识别实战-04
29:18 
任务97： KLDA实例+homework1讲评-01
33:30 
任务98： KLDA实例+homework1讲评-02
30:10 
任务99： KLDA实例+homework1讲评-03
41:52 
任务100： KLDA实例+homework1讲评-04
39:48 
第17章: 深度学习
第1节: 神经网络与激活函数
第2节: BP算法
第3节: 卷积层、Pooling层、全连接层
第4节: 卷积神经网络
第5节: 常用的CNN结构
第6节: Dropout与Bath Normalization
第7节: SGD、Adam、Adagrad算法
第8节: RNN与梯度消失
第9节: LSTM与GRU
第10节: Seq2Seq模型与注意力机制
第11节: Word2Vec, Elmo, Bert, XLNet
第12节: 深度学习中的调参技术
第13节: 深度学习与图嵌入（Graph Embedding）
第14节: Translating Embedding (TransE)
第15节: Node2Vec
第16节: Graph Convolutional Network
第17节: Structured Deep Network Embedding
第18节: Dynamic Graph Embedding
第19节: 【项目实战】基于Seq2Seq和注意力机制的机器翻译
第20节: 【项目实战】基于TransE和GCN的知识图谱推理
第21节: 【项目实战】基于CNN的人脸关键点检测
第18章: 推荐系统与在线学习
第1节: 基于内容的推荐算法
第2节: 基于协同过滤的推荐算法
第3节: 矩阵分解
第4节: 基于内容的Gradient Tree
第5节: 基于深度学习的推荐算法
第6节: 冷启动问题的处理
第7节: Exploration vs Exploitation
第8节: Multi-armed Bandit
第9节: UCB1 algorithm，EXP3 algorithm
第10节: Adversarial Bandit model
第11节: Contexulalized Bandit
第12节: LinUCB
第13节: 【项目实战】使用Gradient Boosting Tree做基于 interaction 与 content的广告推荐
第14节: 【项目实战】使用深度神经网络做基于interaction 与 content的推荐
第15节: 【项目实战】LinUCB做新闻推荐, 最大化rewards
第19章: 贝叶斯模型
第1节: 主题模型（LDA) 以及生成过程
第2节: Dirichlet Distribution, Multinomial Distribution
第3节: 蒙特卡洛与MCMC
第4节: Metropolis Hasting与Gibbs Sampling
第5节: 使用Collapsed Gibbs Sampler求解LDA
第6节: Mean-field variational Inference
第7节: 使用VI求解LDA
第8节: Stochastic Optimization与Bayesian Inference
第9节: 利用SLGD和SVI求解LDA
第10节: 基于分布式计算的贝叶斯模型求解
第11节: 随机过程与无参模型（non-parametric)
第12节: Chinese Retarant Process
第13节: Stick Breaking Process
第14节: Stochastic Block Model与MMSB
第15节: 基于SGLD与SVI的MMSB求解
第16节: Bayesian Deep Learning模型
第17节: Deep Generative Model
第18节: 【项目实战】基于Bayesian LSTM的文本分析
第19节: 【项目实战】使用无参主题模型做文本分类
第20节: 【项目实战】基于贝叶斯模型实现小数量的图像识别
第20章: 增强学习与其他前沿主题
第1节: Policy Learning
第2节: Deep RL
第3节: Variational Autoencoder(VAE)与求解
第4节: 隐变量的Disentangling
第5节: 图像的生成以及Disentangling
第6节: 文本的生成以及Disentangling
第7节: Generative Adversial Network(GAN)
第8节: CycleGan
第9节: 深度学习的可解释性
第10节: Deconvolution与图像特征的解释
第11节: Layer-wise Propagation
第12节: Adversial Machine Learning
第13节: Purturbation Analysis
第14节: Fair Learning
第15节: 【项目实战】基于GAN的图像生成
第16节: 【项目实战】基于VAE的文本Style Transfer
第17节: 【项目实战】可视化机器翻译系统
```

# 论文社区
```
第1章: BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding(2018)
免费 任务1： 2019.1.13 paper-01
29:45 
任务2： 2019.1.13 paper-02
25:57 
任务3： 2019.1.13 paper-03
25:58 
第2章: Deep Contextualized Word Representations
免费 任务4： Deep Contextualized Word Representations(2018)-01
20:11 
任务5： Deep Contextualized Word Representations(2018)-02
11:22 
任务6： Deep Contextualized Word Representations(2018)-03
16:38 
任务7： Deep Contextualized Word Representations(2018)-04
21:04 
第3章: Real-time Personalization using Embeddings for Search Ranking at Airbnb
免费 任务8： airbnb-01
21:12 
任务9： airbnb-02
25:28 
任务10： airbnb-03
27:14 
第4章: Know What You Don’t Know Unanswerable Questions for SQuAD
任务11： 2019.2.17Paper-01
19:41 
任务12： 2019.2.17Paper-02
20:59 
任务13： 2019.2.17Paper-03
17:28 
第5章: Paper-Enricingwordvectors
任务14： 22019.2.25 Paper-Enricingwordvectors-01
10:19 
任务15： 2019.2.25 Paper-Enricingwordvectors-02
22:08 
任务16： 2019.2.25 Paper-Enricingwordvectors-03
19:19 
第6章: 20190303 Batch Renormalization
任务17： Batch Renormalization-01
40:58 
任务18： Batch Renormalization-02
22:24 
第7章: 20190310 Poisson Image Editing（2003）
任务19： Poisson Image Editing（2003）
35:31 
第8章: 20190317 Modeling Relational Data with Graph Convolutional Networks
任务20： ModelingRelationalData GraphConvolutionalNetworks
48:08 
第9章: 20190324RippleNet Propagating User Preferences on the Knowledge Graph for Recommender Systems
任务21： RippleNet Propagating User Preferences on the
42:49 
第10章: 20190331《Glyce：Glyph-vectors for Chinese Character Representations》
任务22： Glyce：Glyph-vectors for Chinese Character
49:45 
第11章: 20190414 Deep Short Text Classification with Knowledge Powered Attention
任务23： Deep Short Text Classification with Knowledge-01
44:56 
任务24： Deep Short Text Classification with Knowledge-02
33:07 
第12章: 20190421《Pointer Networks》
任务25： 《Pointer Networks》-01
18:22 
任务26： 《Pointer Networks》-02
22:52 
第13章: How Does Batch Normalization Help Optimization
任务27： How Does Batch Normalization Help Optimization-01
18:28 
任务28： How Does Batch Normalization Help Optimization-02
34:40 
任务29： How Does Batch Normalization Help Optimization-03
41:29 
第14章: 20190519 项目展示
任务30： NLP落地项目分享展示
01:43:10 
第15章: 20190526 模型解析类论文和产品安利类论文
任务31： 模型解析类论文和产品安利类论文
30:58 
第16章: TCN模型
任务32： TCN模型-01
20:08 
任务33： TCN模型-02
19:20 
任务34： TCN模型-03
20:50 
第17章: GloVe Global Vectors for Word Representation
任务35： GloVe Global Vectors for Word Representation-01
26:26 
任务36： GloVe Global Vectors for Word Representation-02
24:18 
第18章: Multimodal Machine Learning A Survey and Taxonomy
任务37： Multimodal Machine Learning A Survey and Taxonomy
49:19 
第19章: Tutorial-What is a variational autoencoder
任务38： Tutorial-What is a variational autoencoder-01
18:35 
任务39： Tutorial-What is a variational autoencoder-02
27:31 
任务40： Tutorial-What is a variational autoencoder-03
23:28 
第20章: Reading Wikipedia to Answer Open-Domain Question
任务41： Reading Wikipedia to Answer Open-Domain Question-1
40:20 
任务42： Reading Wikipedia to Answer Open-Domain Question-2
41:36 
第21章: 《Mining and Summarizing Customer Reviews 》
任务43： 《Mining and Summarizing Customer Reviews 》1
26:44 
任务44： 《Mining and Summarizing Customer Reviews 》2
31:05 
第22章: 论文 GLOVE
任务45： GLOVE-01
28:42 
任务46： GLOVE-02
38:01 
第23章: Real-time Personalization using Embeddings for Search Ranking at Airbnb
任务47： Real-time Personalization using Embeddings for -01
26:22 
任务48： Real-time Personalization using Embeddings for-02
35:45 
任务49： Real-time Personalization using Embeddings for-03
31:52 
第24章: Representation Learning- A Review and New Perspectives
任务50： A Review and New Perspectives01
26:33 
任务51： A Review and New Perspectives02
27:00 
任务52： Visualizing and Understanding Neural Models-01
20:23 
任务53： Visualizing and Understanding Neural Models-02
24:16 
第25章: Bidirectional LSTM-CRF Models for Sequence Tagging
任务54： Bidirectional LSTM-CRF Models for Sequence Tagging
42:36 
第26章: 20191020 LSTM-A search Space Odyssey01
任务55： LSTM-A search Space Odyssey01
30:41 
任务56： LSTM-A search Space Odyssey02
21:32 
第27章: 20191020 Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping
任务57： Searching and Mining Trillions of 01
25:19 
任务58： Searching and Mining Trillions of 02
31:54 
任务59： Searching and Mining Trillions of 03
14:11 
第28章: 20191027From Word Embeddings To Document Distances
任务60： From Word Embeddings To Document Distances
41:24 
第29章: 20191027 论文BERT
任务61： BERT
39:19 
第30章: 20191109 Adam
任务62： Adam01
26:26 
任务63： Adam02
39:12 
第31章: 20191109 XGBoost
任务64： XGBoost01
31:25 
任务65： XGBoost02
33:16 
第32章: 20191109 从词嵌入到文档距离
任务66： 从词嵌入到文档距离01
47:14 
任务67： 从词嵌入到文档距离02
44:52 
第33章: 20191109 Xlnet
任务68： Xlnet
19:44 
第34章: 20191109 LMNN
任务69： LMNN
46:48 
第35章: 20191117 ALbert
任务70： ALbert
44:31 
第36章: 20191119 Mining and Summarizing Customer Reviews
任务71： Mining and Summarizing Customer Reviews01
20:57 
任务72： Mining and Summarizing Customer Reviews02
21:49 
任务73： Mining and Summarizing Customer Reviews03
17:23 
第37章: Kernel Logistic Regression and the Import Vector Machine
任务74： Kernel Logistic Regression and the Import VM01
33:21 
任务75： Kernel Logistic Regression and the Import VM02
36:38 
第38章: 20191124 Reading Wikipedia to Answer Open-Domain Questions
任务76： Reading Wikipedia to Answer Open-Domain Q01
23:40 
任务77： Reading Wikipedia to Answer Open-Domain Q02
31:18 
第39章: 20191130 XGBoost
任务78： XGBoost-01
23:43 
任务79： XGBoost-02
24:51 
任务80： XGBoost-03
28:15 
第40章: 20191207 LightGBM
任务81： LightGBM-01
25:21 
任务82： LightGBM-02
28:57 
任务83： LightGBM-03
32:22 
第41章: 20191208 Bidirectional LSTM-CRF Models for Sequence Tagging
任务84： Bidirectional LSTM-CRF Models for Sequence Tagging
```
