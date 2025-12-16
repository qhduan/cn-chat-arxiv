# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | ["All of Me": Mining Users' Attributes from their Public Spotify Playlists.](http://arxiv.org/abs/2401.14296) | 本研究调查了Spotify用户属性与他们公开播放列表之间的关系，特别关注识别与用户个人属性相关的音乐特征。 |
| [^2] | [An Anytime Algorithm for Good Arm Identification.](http://arxiv.org/abs/2310.10359) | 提出了一个适用于随机贝叶斯臂机的随时和无参数采样规则APGAI，通过自适应策略提高了好臂识别效率，在固定置信度和固定预算的情况下有良好实验表现。 |
| [^3] | [Sharp Generalization of Transductive Learning: A Transductive Local Rademacher Complexity Approach.](http://arxiv.org/abs/2309.16858) | 我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们利用变量的方差信息构建了TLRC，并将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。 |
| [^4] | [Behavioral Machine Learning? Computer Predictions of Corporate Earnings also Overreact.](http://arxiv.org/abs/2303.16158) | 本文研究发现，机器学习算法可以更准确地预测公司盈利，但同样存在过度反应的问题，而传统培训的股市分析师和经过机器学习方法培训的分析师相比会产生较少的过度反应。 |
| [^5] | [Vertical Semi-Federated Learning for Efficient Online Advertising.](http://arxiv.org/abs/2209.15635) | 垂直半联合学习为在线广告领域提供了高效的解决方案，通过学习一个联合感知的局部模型以应对传统垂直联合学习的限制。 |
| [^6] | [The prediction of the quality of results in Logic Synthesis using Transformer and Graph Neural Networks.](http://arxiv.org/abs/2207.11437) | 该论文介绍了一种使用Transformer和图神经网络预测逻辑综合结果质量的深度学习方法，通过将结构转换表示为向量并提取优化序列的特征，以及利用图神经网络学习电路的图表示和预测QoR。 |

# 详细

[^1]: "All of Me": 从公开的Spotify播放列表中挖掘用户属性

    "All of Me": Mining Users' Attributes from their Public Spotify Playlists. (arXiv:2401.14296v1 [cs.CR])

    [http://arxiv.org/abs/2401.14296](http://arxiv.org/abs/2401.14296)

    本研究调查了Spotify用户属性与他们公开播放列表之间的关系，特别关注识别与用户个人属性相关的音乐特征。

    

    在数字音乐流媒体时代，像Spotify这样的平台上的播放列表已经成为个人音乐体验的重要组成部分。人们创建并公开分享自己的播放列表，以表达他们的音乐品味，推广他们最喜爱的艺术家的发现，并促进社交联系。这些可以公开访问的播放列表超越了仅仅音乐偏好的界限：它们是丰富洞察用户属性和身份的来源。例如，老年人的音乐偏好可能更偏向于弗兰克·辛纳屈，而比莉·艾利什仍然是十几岁青少年的首选。因此，这些播放列表成为了一扇了解音乐身份多样而不断演变的窗口。在这项工作中，我们研究了Spotify用户属性和他们的公开播放列表之间的关系。我们特别关注识别与用户个人属性相关的经常出现的音乐特征，例如人口统计信息，习惯或个性等。

    In the age of digital music streaming, playlists on platforms like Spotify have become an integral part of individuals' musical experiences. People create and publicly share their own playlists to express their musical tastes, promote the discovery of their favorite artists, and foster social connections. These publicly accessible playlists transcend the boundaries of mere musical preferences: they serve as sources of rich insights into users' attributes and identities. For example, the musical preferences of elderly individuals may lean more towards Frank Sinatra, while Billie Eilish remains a favored choice among teenagers. These playlists thus become windows into the diverse and evolving facets of one's musical identity.  In this work, we investigate the relationship between Spotify users' attributes and their public playlists. In particular, we focus on identifying recurring musical characteristics associated with users' individual attributes, such as demographics, habits, or perso
    
[^2]: 一个适用于好臂识别的随时算法

    An Anytime Algorithm for Good Arm Identification. (arXiv:2310.10359v1 [stat.ML])

    [http://arxiv.org/abs/2310.10359](http://arxiv.org/abs/2310.10359)

    提出了一个适用于随机贝叶斯臂机的随时和无参数采样规则APGAI，通过自适应策略提高了好臂识别效率，在固定置信度和固定预算的情况下有良好实验表现。

    

    在好臂识别（GAI）中，目标是识别其中一个平均性能超过给定阈值的臂，称为好臂（如果存在）。目前很少有研究在固定预算的情况下进行GAI，即在先确定好预算之后，或者在任何时刻都可以要求推荐的随时设置下进行GAI。我们提出了一种名为APGAI的随时和无参数采样规则，用于随机贝叶斯臂机。APGAI可以直接用于固定置信度和固定预算的设定中。首先，我们得出其任何时刻的误差概率的上界。这些上界表明，自适应策略在检测没有好臂的时候比均匀采样更高效。其次，当APGAI与一个停止规则结合时，我们证明了在任何置信水平下的预期采样复杂性的上界。最后，我们展示了APGAI在合成数据和真实世界数据上的良好实验性能。我们的工作为所有设置中的GAI问题提供了一个广泛的概述。

    In good arm identification (GAI), the goal is to identify one arm whose average performance exceeds a given threshold, referred to as good arm, if it exists. Few works have studied GAI in the fixed-budget setting, when the sampling budget is fixed beforehand, or the anytime setting, when a recommendation can be asked at any time. We propose APGAI, an anytime and parameter-free sampling rule for GAI in stochastic bandits. APGAI can be straightforwardly used in fixed-confidence and fixed-budget settings. First, we derive upper bounds on its probability of error at any time. They show that adaptive strategies are more efficient in detecting the absence of good arms than uniform sampling. Second, when APGAI is combined with a stopping rule, we prove upper bounds on the expected sampling complexity, holding at any confidence level. Finally, we show good empirical performance of APGAI on synthetic and real-world data. Our work offers an extensive overview of the GAI problem in all settings.
    
[^3]: Transductive Learning的尖锐泛化：一种Transductive Local Rademacher Complexity方法

    Sharp Generalization of Transductive Learning: A Transductive Local Rademacher Complexity Approach. (arXiv:2309.16858v1 [stat.ML])

    [http://arxiv.org/abs/2309.16858](http://arxiv.org/abs/2309.16858)

    我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们利用变量的方差信息构建了TLRC，并将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。

    

    我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们的工作将传统的local rademacher complexity (LRC)的思想扩展到了transductive设置中，相对于典型的LRC方法在归纳设置中的分析有了相当大的变化。我们提出了一种基于Rademacher complex的局部化工具，可以应用于各种transductive learning问题，并在适当条件下得到了尖锐的界限。与LRC的发展类似，我们通过从独立变量的方差信息开始构建TLRC，将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。经过精心设计的...

    We introduce a new tool, Transductive Local Rademacher Complexity (TLRC), to analyze the generalization performance of transductive learning methods and motivate new transductive learning algorithms. Our work extends the idea of the popular Local Rademacher Complexity (LRC) to the transductive setting with considerable changes compared to the analysis of typical LRC methods in the inductive setting. We present a localized version of Rademacher complexity based tool wihch can be applied to various transductive learning problems and gain sharp bounds under proper conditions. Similar to the development of LRC, we build TLRC by starting from a sharp concentration inequality for independent variables with variance information. The prediction function class of a transductive learning model is then divided into pieces with a sub-root function being the upper bound for the Rademacher complexity of each piece, and the variance of all the functions in each piece is limited. A carefully designed 
    
[^4]: 机器学习准确预测财报，但同样存在过度反应

    Behavioral Machine Learning? Computer Predictions of Corporate Earnings also Overreact. (arXiv:2303.16158v1 [q-fin.ST])

    [http://arxiv.org/abs/2303.16158](http://arxiv.org/abs/2303.16158)

    本文研究发现，机器学习算法可以更准确地预测公司盈利，但同样存在过度反应的问题，而传统培训的股市分析师和经过机器学习方法培训的分析师相比会产生较少的过度反应。

    

    大量证据表明，在金融领域中，机器学习算法的预测能力比人类更为准确。但是，文献并未测试算法预测是否更为理性。本文研究了几个算法（包括线性回归和一种名为Gradient Boosted Regression Trees的流行算法）对于公司盈利的预测结果。结果发现，GBRT平均胜过线性回归和人类股市分析师，但仍存在过度反应且无法满足理性预期标准。通过降低学习率，可最小程度上减少过度反应程度，但这会牺牲预测准确性。通过机器学习方法培训过的股市分析师比传统训练的分析师产生的过度反应较少。此外，股市分析师的预测反映出机器算法没有捕捉到的信息。

    There is considerable evidence that machine learning algorithms have better predictive abilities than humans in various financial settings. But, the literature has not tested whether these algorithmic predictions are more rational than human predictions. We study the predictions of corporate earnings from several algorithms, notably linear regressions and a popular algorithm called Gradient Boosted Regression Trees (GBRT). On average, GBRT outperformed both linear regressions and human stock analysts, but it still overreacted to news and did not satisfy rational expectation as normally defined. By reducing the learning rate, the magnitude of overreaction can be minimized, but it comes with the cost of poorer out-of-sample prediction accuracy. Human stock analysts who have been trained in machine learning methods overreact less than traditionally trained analysts. Additionally, stock analyst predictions reflect information not otherwise available to machine algorithms.
    
[^5]: 垂直半联合学习用于高效在线广告

    Vertical Semi-Federated Learning for Efficient Online Advertising. (arXiv:2209.15635v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.15635](http://arxiv.org/abs/2209.15635)

    垂直半联合学习为在线广告领域提供了高效的解决方案，通过学习一个联合感知的局部模型以应对传统垂直联合学习的限制。

    

    传统的垂直联合学习架构存在两个主要问题：1）适用范围受限于重叠样本；2）实时联合服务的系统挑战较高，这限制了其在广告系统中的应用。为解决这些问题，我们提出了一种新的学习设置——半垂直联合学习(Semi-VFL)，以应对这些挑战。半垂直联合学习旨在实现垂直联合学习的实际工业应用方式，通过学习一个联合感知的局部模型，该模型表现优于单方模型，同时保持了局部服务的便利性。为此，我们提出了精心设计的联合特权学习框架(JPL)，来解决被动方特征缺失和适应整个样本空间这两个问题。具体而言，我们构建了一个推理高效的适用于整个样本空间的单方学生模型，同时保持了联合特征扩展的优势。新的表示蒸馏

    The traditional vertical federated learning schema suffers from two main issues: 1) restricted applicable scope to overlapped samples and 2) high system challenge of real-time federated serving, which limits its application to advertising systems. To this end, we advocate a new learning setting Semi-VFL (Vertical Semi-Federated Learning) to tackle these challenge. Semi-VFL is proposed to achieve a practical industry application fashion for VFL, by learning a federation-aware local model which performs better than single-party models and meanwhile maintain the convenience of local-serving. For this purpose, we propose the carefully designed Joint Privileged Learning framework (JPL) to i) alleviate the absence of the passive party's feature and ii) adapt to the whole sample space. Specifically, we build an inference-efficient single-party student model applicable to the whole sample space and meanwhile maintain the advantage of the federated feature extension. New representation distilla
    
[^6]: 使用Transformer和图神经网络预测逻辑综合结果的质量

    The prediction of the quality of results in Logic Synthesis using Transformer and Graph Neural Networks. (arXiv:2207.11437v2 [cs.AR] UPDATED)

    [http://arxiv.org/abs/2207.11437](http://arxiv.org/abs/2207.11437)

    该论文介绍了一种使用Transformer和图神经网络预测逻辑综合结果质量的深度学习方法，通过将结构转换表示为向量并提取优化序列的特征，以及利用图神经网络学习电路的图表示和预测QoR。

    

    在逻辑综合阶段，综合工具中的结构转换需要与优化序列结合，并作用于电路，以满足指定的电路面积和延迟。然而，逻辑综合优化序列的运行时间较长，为电路对综合优化序列的结果质量（QoR）进行预测可以帮助工程师更快地找到更好的优化序列。在这项工作中，我们提出了一种深度学习方法，用于预测未见过的电路-优化序列对的QoR。具体而言，通过嵌入方法将结构转换转化为向量，并利用先进的自然语言处理（NLP）技术（Transformer）提取优化序列的特征。此外，为了使模型的预测过程能够从电路泛化到电路，电路的图表示被表示为邻接矩阵和特征矩阵。图神经网络被用于学习电路的图表示和预测QoR。

    In the logic synthesis stage, structure transformations in the synthesis tool need to be combined into optimization sequences and act on the circuit to meet the specified circuit area and delay. However, logic synthesis optimization sequences are time-consuming to run, and predicting the quality of the results (QoR) against the synthesis optimization sequence for a circuit can help engineers find a better optimization sequence faster. In this work, we propose a deep learning method to predict the QoR of unseen circuit-optimization sequences pairs. Specifically, the structure transformations are translated into vectors by embedding methods and advanced natural language processing (NLP) technology (Transformer) is used to extract the features of the optimization sequences. In addition, to enable the prediction process of the model to be generalized from circuit to circuit, the graph representation of the circuit is represented as an adjacency matrix and a feature matrix. Graph neural net
    

