# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Educational Outcome with Machine Learning: Modeling Friendship Formation, Measuring Peer Effect and Optimizing Class Assignment](https://arxiv.org/abs/2404.02497) | 该论文利用机器学习解决学校班级分配问题，通过友谊预测、同侪影响估计和班级分配优化，发现将学生分成有性别特征的班级能够提高平均同侪影响，并且极端混合的班级分配方法可以改善底部四分之一学生的同侪影响。 |
| [^2] | [Identifying Causal Effects in Information Provision Experiments.](http://arxiv.org/abs/2309.11387) | 信息提供实验用于确定信念如何因果地影响决策和行为。通过应用贝叶斯估计器，可以准确识别出（非加权的）平均部分效应。 |
| [^3] | [Torch-Choice: A PyTorch Package for Large-Scale Choice Modelling with Python.](http://arxiv.org/abs/2304.01906) | 本文介绍了一款名为 Torch-Choice 的 PyTorch 软件包，用于管理数据库、构建多项式Logit和嵌套Logit模型，并支持GPU加速，具有灵活性和高效性。 |

# 详细

[^1]: 用机器学习提升教育成果：建模友谊形成，衡量同侪影响和优化班级分配

    Enhancing Educational Outcome with Machine Learning: Modeling Friendship Formation, Measuring Peer Effect and Optimizing Class Assignment

    [https://arxiv.org/abs/2404.02497](https://arxiv.org/abs/2404.02497)

    该论文利用机器学习解决学校班级分配问题，通过友谊预测、同侪影响估计和班级分配优化，发现将学生分成有性别特征的班级能够提高平均同侪影响，并且极端混合的班级分配方法可以改善底部四分之一学生的同侪影响。

    

    在这篇论文中，我们研究了学校校长的班级分配问题。我们将问题分为三个阶段：友谊预测、同侪影响评估和班级分配优化。我们建立了一个微观基础模型来模拟友谊形成，并将该模型逼近为一个神经网络。利用预测的友谊概率邻接矩阵，我们改进了传统的线性均值模型并估计了同侪影响。我们提出了一种新的工具以解决友谊选择的内生性问题。估计的同侪影响略大于线性均值模型的估计。利用友谊预测和同侪影响估计结果，我们模拟了所有学生的反事实同侪影响。我们发现将学生分成有性别特征的班级可以将平均同侪影响提高0.02分（在5分制中）。我们还发现极端混合的班级分配方法可以提高底部四分之一学生的同侪影响。

    arXiv:2404.02497v1 Announce Type: new  Abstract: In this paper, we look at a school principal's class assignment problem. We break the problem into three stages (1) friendship prediction (2) peer effect estimation (3) class assignment optimization. We build a micro-founded model for friendship formation and approximate the model as a neural network. Leveraging on the predicted friendship probability adjacent matrix, we improve the traditional linear-in-means model and estimate peer effect. We propose a new instrument to address the friendship selection endogeneity. The estimated peer effect is slightly larger than the linear-in-means model estimate. Using the friendship prediction and peer effect estimation results, we simulate counterfactual peer effects for all students. We find that dividing students into gendered classrooms increases average peer effect by 0.02 point on a scale of 5. We also find that extreme mixing class assignment method improves bottom quartile students' peer ef
    
[^2]: 信息提供实验中的因果效应识别

    Identifying Causal Effects in Information Provision Experiments. (arXiv:2309.11387v1 [econ.EM])

    [http://arxiv.org/abs/2309.11387](http://arxiv.org/abs/2309.11387)

    信息提供实验用于确定信念如何因果地影响决策和行为。通过应用贝叶斯估计器，可以准确识别出（非加权的）平均部分效应。

    

    信息提供实验是一种越来越流行的工具，用于确定信念如何因果地影响决策和行为。在基于负担信息获取的简单贝叶斯信念形成模型中，当这些信念对他们的决策至关重要时，人们形成精确的信念。先前信念的精确度控制着当他们接受新信息时他们的信念变化程度（即第一阶段的强度）。由于两阶段最小二乘法（TSLS）以权重与第一阶段的强度成比例的加权平均为目标，TSLS会过度加权具有较小因果效应的个体，并低估具有较大效应的个体，从而低估了信念对行为的平均部分效应。在所有参与者都接受新信息的实验设计中，贝叶斯更新意味着可以使用控制函数来确定（非加权的）平均部分效应。我将这个估计器应用于最近一项关于效应的研究。

    Information provision experiments are an increasingly popular tool to identify how beliefs causally affect decision-making and behavior. In a simple Bayesian model of belief formation via costly information acquisition, people form precise beliefs when these beliefs are important for their decision-making. The precision of prior beliefs controls how much their beliefs shift when they are shown new information (i.e., the strength of the first stage). Since two-stage least squares (TSLS) targets a weighted average with weights proportional to the strength of the first stage, TSLS will overweight individuals with smaller causal effects and underweight those with larger effects, thus understating the average partial effect of beliefs on behavior. In experimental designs where all participants are exposed to new information, Bayesian updating implies that a control function can be used to identify the (unweighted) average partial effect. I apply this estimator to a recent study of the effec
    
[^3]: Torch-Choice: 用Python实现大规模选择建模的PyTorch包

    Torch-Choice: A PyTorch Package for Large-Scale Choice Modelling with Python. (arXiv:2304.01906v1 [cs.LG])

    [http://arxiv.org/abs/2304.01906](http://arxiv.org/abs/2304.01906)

    本文介绍了一款名为 Torch-Choice 的 PyTorch 软件包，用于管理数据库、构建多项式Logit和嵌套Logit模型，并支持GPU加速，具有灵活性和高效性。

    

    $\texttt{torch-choice}$ 是一款开源软件包，使用Python和PyTorch实现灵活、快速的选择建模。它提供了 $\texttt{ChoiceDataset}$ 数据结构，以便灵活而高效地管理数据库。本文演示了如何从各种格式的数据库中构建 $\texttt{ChoiceDataset}$，并展示了 $\texttt{ChoiceDataset}$ 的各种功能。该软件包实现了两种常用的模型: 多项式Logit和嵌套Logit模型，并支持模型估计期间的正则化。该软件包还支持使用GPU进行估计，使其可以扩展到大规模数据集而且在计算上更高效。模型可以使用R风格的公式字符串或Python字典进行初始化。最后，我们比较了 $\texttt{torch-choice}$ 和 R中的 $\texttt{mlogit}$ 在以下几个方面的计算效率: (1) 观测数增加时，(2) 协变量个数增加时， (3) 测试数升高时。

    The $\texttt{torch-choice}$ is an open-source library for flexible, fast choice modeling with Python and PyTorch. $\texttt{torch-choice}$ provides a $\texttt{ChoiceDataset}$ data structure to manage databases flexibly and memory-efficiently. The paper demonstrates constructing a $\texttt{ChoiceDataset}$ from databases of various formats and functionalities of $\texttt{ChoiceDataset}$. The package implements two widely used models, namely the multinomial logit and nested logit models, and supports regularization during model estimation. The package incorporates the option to take advantage of GPUs for estimation, allowing it to scale to massive datasets while being computationally efficient. Models can be initialized using either R-style formula strings or Python dictionaries. We conclude with a comparison of the computational efficiencies of $\texttt{torch-choice}$ and $\texttt{mlogit}$ in R as (1) the number of observations increases, (2) the number of covariates increases, and (3) th
    

