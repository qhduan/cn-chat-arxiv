# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedShift: Tackling Dual Heterogeneity Problem of Federated Learning via Weight Shift Aggregation](https://rss.arxiv.org/abs/2402.01070) | 本文介绍了一种名为FedShift的算法，通过权重迁移聚合来解决联邦学习中的异质性问题，并提高训练速度和模型准确性。 |
| [^2] | [Learning General Policies for Classical Planning Domains: Getting Beyond C$_2$](https://arxiv.org/abs/2403.11734) | 该研究提出了一种参数化版本的关系GNNs，通过在$t$为无穷大时仅使用二次空间的嵌入来近似$3$-GNNs，对于较低的$t$值，通过交换较少的消息实现弱的近似，同时通常产生了几个规划领域中所需的$C_3$特征。 |
| [^3] | [Second-Order Fine-Tuning without Pain for LLMs:A Hessian Informed Zeroth-Order Optimizer](https://arxiv.org/abs/2402.15173) | 提出了HiZOO，一种对角Hessian信息的零阶优化器，以增强LLMs微调过程中的模型收敛速度和准确性 |
| [^4] | [AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts](https://arxiv.org/abs/2402.07625) | 本论文介绍了一种自主数据选择策略，利用语言模型进行数学文本的自动评估和选择，并通过连续预训练显著提高了数学推理能力。主要创新包括利用元提示语言模型作为验证器，发布了高质量的AutoMathText数据集，并实现了预训练令牌效率的提升。 |
| [^5] | [Fast and Efficient Matching Algorithm with Deadline Instances](https://arxiv.org/abs/2305.08353) | 本文介绍了一种带有截止期限实例的快速高效匹配算法，通过引入带有截止期限的市场模型，提出了两种优化算法（FastGreedy和FastPostponedGreedy）。该算法在处理机器学习中的在线加权匹配问题时具有较快的速度和准确性。 |
| [^6] | [Domain Adaptation based Interpretable Image Emotion Recognition using Facial Expression Recognition](https://arxiv.org/abs/2011.08388) | 本论文提出了一种基于领域自适应的图像情绪识别方法，通过提出面部情绪识别系统并将其适应为图像情绪识别系统，解决了预训练模型和数据集不足的挑战。同时提出了一种新颖的解释性方法，用于解释情绪识别中关键的视觉特征。 |
| [^7] | [Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach.](http://arxiv.org/abs/2401.10747) | 本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。 |
| [^8] | [$G$-Mapper: Learning a Cover in the Mapper Construction.](http://arxiv.org/abs/2309.06634) | 本论文介绍了一种基于统计检验和聚类算法的优化Mapper图覆盖的方法，通过分割覆盖选择生成了保留数据集本质的Mapper图。 |
| [^9] | [Synthetic Control Methods by Density Matching under Implicit Endogeneitiy.](http://arxiv.org/abs/2307.11127) | 本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。 |
| [^10] | [Do humans and machines have the same eyes? Human-machine perceptual differences on image classification.](http://arxiv.org/abs/2304.08733) | 本文研究通过图像分类探究了人机感知差异，发现即使准确率相似，人类和机器的答案分布也可能不同，并提出了一种后期人机合作来提高任务表现。 |
| [^11] | [repliclust: Synthetic Data for Cluster Analysis.](http://arxiv.org/abs/2303.14301) | repliclust 是一个 Python 包，用于生成具有聚类的合成数据集，基于数据集的原型，提供了放置集群中心、采样集群形状、选择每个集群的数据点数量以及为集群分配概率分布的算法。 |
| [^12] | [Granular-ball Optimization Algorithm.](http://arxiv.org/abs/2303.12807) | 粒球优化算法(GBO)是一种新的多粒度优化算法，可以通过引入粒球计算来提高全局搜索能力和收敛速度，实验结果表明，在这些方面它比现有的最先进的算法表现更优。 |
| [^13] | [Statistical Inference of Constrained Stochastic Optimization via Sketched Sequential Quadratic Programming.](http://arxiv.org/abs/2205.13687) | 本篇论文提出了一种用于等式约束的随机非线性优化问题的统计推断方法，通过基于草图的顺序二次规划（StoSQP）进行求解，并且允许自适应选择随机步长和使用高效随机迭代求解器来降低计算成本。 |

# 详细

[^1]: FedShift: 通过权重迁移聚合解决联邦学习的双重异质性问题

    FedShift: Tackling Dual Heterogeneity Problem of Federated Learning via Weight Shift Aggregation

    [https://rss.arxiv.org/abs/2402.01070](https://rss.arxiv.org/abs/2402.01070)

    本文介绍了一种名为FedShift的算法，通过权重迁移聚合来解决联邦学习中的异质性问题，并提高训练速度和模型准确性。

    

    联邦学习（FL）提供了一种训练机器学习模型并注重保护数据隐私的有力方法。FL中存在的系统异质性和统计异质性问题源于客户端硬件、网络和数据集分布的多样性。这种多样性可能会严重影响训练速度和模型性能。虽然许多研究通过引入通信效率或稳定收敛算法来解决系统或统计异质性问题，但单独解决这些挑战往往会导致妥协，因为异质性问题未得到解决。为此，本文介绍了一种名为FedShift的新算法，旨在在双重异质性场景中提高训练速度和模型的准确性。我们的解决方案通过量化改善客户参与度，并通过应用迁移技术来缓解量化通常导致的性能不良影响。

    Federated Learning (FL) offers a compelling method for training machine learning models with a focus on preserving data privacy. The presence of system heterogeneity and statistical heterogeneity, recognized challenges in FL, arises from the diversity of client hardware, network, and dataset distribution. This diversity can critically affect the training pace and the performance of models. While many studies address either system or statistical heterogeneity by introducing communication-efficient or stable convergence algorithms, addressing these challenges in isolation often leads to compromises due to unaddressed heterogeneity. In response, this paper introduces FedShift, a novel algorithm designed to enhance both the training speed and the models' accuracy in a dual heterogeneity scenario. Our solution can improve client engagement through quantization and mitigate the adverse effects on performance typically associated with quantization by employing a shifting technique. This techn
    
[^2]: 学习古典规划领域的通用策略：超越$C_2$

    Learning General Policies for Classical Planning Domains: Getting Beyond C$_2$

    [https://arxiv.org/abs/2403.11734](https://arxiv.org/abs/2403.11734)

    该研究提出了一种参数化版本的关系GNNs，通过在$t$为无穷大时仅使用二次空间的嵌入来近似$3$-GNNs，对于较低的$t$值，通过交换较少的消息实现弱的近似，同时通常产生了几个规划领域中所需的$C_3$特征。

    

    基于GNN的方法用于学习跨规划领域的通用策略受到$C_2$表达能力的限制，即一阶逻辑只能包含两个变量和计数。这种限制可以通过转向$k$-GNNs，其中$k=3$，其中物体嵌入被三元组嵌入所替换，来克服。然而，尽管$3$-GNNs具有$C_3$的表达能力，但不同于受限于$C_2$的$1$-和$2$-GNNs，它们需要四次时间进行消息交换和三次空间进行嵌入，使它们变得不切实际。在这项工作中，我们引入了一个参数化版本的关系GNNs。当$t$为无穷大时，R-GNN[$t$]仅使用二次空间的嵌入来近似$3$-GNNs。对于较低的$t$值，例如$t=1$和$t=2$，R-GNN[$t$]通过交换较少的消息实现了更弱的近似，但有趣的是，通常产生了在几个规划领域中所需的$C_3$特征。此外，新的R-GNN[$t$] ar

    arXiv:2403.11734v1 Announce Type: new  Abstract: GNN-based approaches for learning general policies across planning domains are limited by the expressive power of $C_2$, namely; first-order logic with two variables and counting. This limitation can be overcomed by transitioning to $k$-GNNs, for $k=3$, wherein object embeddings are substituted with triplet embeddings. Yet, while $3$-GNNs have the expressive power of $C_3$, unlike $1$- and $2$-GNNs that are confined to $C_2$, they require quartic time for message exchange and cubic space for embeddings, rendering them impractical. In this work, we introduce a parameterized version of relational GNNs. When $t$ is infinity, R-GNN[$t$] approximates $3$-GNNs using only quadratic space for embeddings. For lower values of $t$, such as $t=1$ and $t=2$, R-GNN[$t$] achieves a weaker approximation by exchanging fewer messages, yet interestingly, often yield the $C_3$ features required in several planning domains. Furthermore, the new R-GNN[$t$] ar
    
[^3]: 无痛人工大语言模型的二阶微调：一种基于Hessian信息的零阶优化器

    Second-Order Fine-Tuning without Pain for LLMs:A Hessian Informed Zeroth-Order Optimizer

    [https://arxiv.org/abs/2402.15173](https://arxiv.org/abs/2402.15173)

    提出了HiZOO，一种对角Hessian信息的零阶优化器，以增强LLMs微调过程中的模型收敛速度和准确性

    

    通过背向传播过程对大型语言模型（LLMs）进行微调，通常需要昂贵的GPU内存。最近的研究转向使用零阶优化器进行微调，通过两次前向传递显著节省内存。然而，这些优化器受不同维度之间参数曲率的异质性困扰。在这项工作中，我们提出了HiZOO，一种对角Hessian信息的零阶优化器，这是第一项利用对角Hessian增强零阶优化器进行LLMs微调的工作。HiZOO避免了昂贵的内存成本，并且每步只增加了一个前向传递。对各种模型（350M〜66B参数）进行的大量实验表明，HiZOO提高了模型收敛速度，显著减少了训练步骤，并有效提高了模型准确性。此外，我们可视化了HiZOO在测试函数上的优化轨迹，

    arXiv:2402.15173v1 Announce Type: new  Abstract: Fine-tuning large language models (LLMs) with classic first-order optimizers entails prohibitive GPU memory due to the backpropagation process. Recent works have turned to zeroth-order optimizers for fine-tuning, which save substantial memory by using two forward passes. However, these optimizers are plagued by the heterogeneity of parameter curvatures across different dimensions. In this work, we propose HiZOO, a diagonal Hessian informed zeroth-order optimizer which is the first work to leverage the diagonal Hessian to enhance zeroth-order optimizer for fine-tuning LLMs. What's more, HiZOO avoids the expensive memory cost and only increases one forward pass per step. Extensive experiments on various models (350M~66B parameters) indicate that HiZOO improves model convergence, significantly reducing training steps and effectively enhancing model accuracy. Moreover, we visualize the optimization trajectories of HiZOO on test functions, il
    
[^4]: AutoMathText：使用语言模型进行数学文本的自主数据选择

    AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts

    [https://arxiv.org/abs/2402.07625](https://arxiv.org/abs/2402.07625)

    本论文介绍了一种自主数据选择策略，利用语言模型进行数学文本的自动评估和选择，并通过连续预训练显著提高了数学推理能力。主要创新包括利用元提示语言模型作为验证器，发布了高质量的AutoMathText数据集，并实现了预训练令牌效率的提升。

    

    为了通过持续的预训练改善语言模型在数学推理方面的能力，我们引入了一种新颖的策略，利用基础语言模型进行自主数据选择。与传统的有人工标注数据的监督微调或训练过的分类器不同，我们的方法利用元提示语言模型作为零样本验证器，自主评估和选择高质量的数学内容，并发布了经过策划的开源AutoMathText数据集，其中包含超过200GB的数据。为了证明我们方法的有效性，我们对AutoMathText数据集进行了连续预训练，使得7B参数的Mistral语言模型在MATH数据集上的下游性能大幅提升，而令牌数量比之前的连续预训练工作减少了几个数量级。我们的方法展示了基准的预训练令牌效率提高了2倍，突显了我们方法在增强中的潜力。

    To improve language models' proficiency in mathematical reasoning via continual pretraining, we introduce a novel strategy that leverages base language models for autonomous data selection. Departing from conventional supervised fine-tuning or trained classifiers with human-annotated data, our approach utilizes meta-prompted language models as zero-shot verifiers to autonomously evaluate and select high-quality mathematical content, and we release the curated open-source AutoMathText dataset encompassing over 200GB of data. To demonstrate the efficacy of our method, we continuously pretrained a 7B-parameter Mistral language model on the AutoMathText dataset, achieving substantial improvements in downstream performance on the MATH dataset with a token amount reduced by orders of magnitude compared to previous continuous pretraining works. Our method showcases a 2 times increase in pretraining token efficiency compared to baselines, underscoring the potential of our approach in enhancing
    
[^5]: 快速高效的带有截止期限实例的匹配算法

    Fast and Efficient Matching Algorithm with Deadline Instances

    [https://arxiv.org/abs/2305.08353](https://arxiv.org/abs/2305.08353)

    本文介绍了一种带有截止期限实例的快速高效匹配算法，通过引入带有截止期限的市场模型，提出了两种优化算法（FastGreedy和FastPostponedGreedy）。该算法在处理机器学习中的在线加权匹配问题时具有较快的速度和准确性。

    

    在机器学习中，在线加权匹配问题由于其众多应用而成为一个基本问题。尽管在这个领域已经做了很多努力，但现有的算法要么速度太慢，要么没有考虑到截止期限（节点可以匹配的最长时间）。在本文中，我们首先引入了一个带有截止期限的市场模型。接下来，我们提出了两个优化算法（FastGreedy和FastPostponedGreedy），并给出了关于算法时间复杂度和正确性的理论证明。在FastGreedy算法中，我们已经知道一个节点是买家还是卖家。但在FastPostponedGreedy算法中，一开始我们不知道每个节点的状态。然后，我们推广了一个草图矩阵，以在真实数据集和合成数据集上运行原始算法和我们的算法。设 ε ∈（0,0.1）表示每条边的真实权重的相对误差。原始的Greedy和Po算法的竞争比率是多少。

    The online weighted matching problem is a fundamental problem in machine learning due to its numerous applications. Despite many efforts in this area, existing algorithms are either too slow or don't take $\mathrm{deadline}$ (the longest time a node can be matched) into account. In this paper, we introduce a market model with $\mathrm{deadline}$ first. Next, we present our two optimized algorithms (\textsc{FastGreedy} and \textsc{FastPostponedGreedy}) and offer theoretical proof of the time complexity and correctness of our algorithms. In \textsc{FastGreedy} algorithm, we have already known if a node is a buyer or a seller. But in \textsc{FastPostponedGreedy} algorithm, the status of each node is unknown at first. Then, we generalize a sketching matrix to run the original and our algorithms on both real data sets and synthetic data sets. Let $\epsilon \in (0,0.1)$ denote the relative error of the real weight of each edge. The competitive ratio of original \textsc{Greedy} and \textsc{Po
    
[^6]: 基于领域自适应的可解释图像情绪识别，并利用面部表情识别

    Domain Adaptation based Interpretable Image Emotion Recognition using Facial Expression Recognition

    [https://arxiv.org/abs/2011.08388](https://arxiv.org/abs/2011.08388)

    本论文提出了一种基于领域自适应的图像情绪识别方法，通过提出面部情绪识别系统并将其适应为图像情绪识别系统，解决了预训练模型和数据集不足的挑战。同时提出了一种新颖的解释性方法，用于解释情绪识别中关键的视觉特征。

    

    本文提出了一种领域自适应技术，用于识别包含面部和非面部物体以及非人类组件的通用图像中的情绪。它解决了图像情绪识别（IER）中预训练模型和良好注释数据集的不足挑战。首先，提出了一种基于深度学习的面部情绪识别（FER）系统，将给定的面部图像分类为离散情绪类别。然后，提出了一种图像识别系统，将提出的FER系统适应于利用领域自适应识别图像所传达的情绪。它将通用图像分类为“快乐”，“悲伤”，“仇恨”和“愤怒”类别。还提出了一种新颖的解释性方法，称为分而治之的Shap（DnCShap），用于解释情绪识别中高度相关的视觉特征。

    A domain adaptation technique has been proposed in this paper to identify the emotions in generic images containing facial & non-facial objects and non-human components. It addresses the challenge of the insufficient availability of pre-trained models and well-annotated datasets for image emotion recognition (IER). It starts with proposing a facial emotion recognition (FER) system and then moves on to adapting it for image emotion recognition. First, a deep-learning-based FER system has been proposed that classifies a given facial image into discrete emotion classes. Further, an image recognition system has been proposed that adapts the proposed FER system to recognize the emotions portrayed by images using domain adaptation. It classifies the generic images into 'happy,' 'sad,' 'hate,' and 'anger' classes. A novel interpretability approach, Divide and Conquer based Shap (DnCShap), has also been proposed to interpret the highly relevant visual features for emotion recognition. The prop
    
[^7]: 缺失模态下的多模态情感分析:一种知识迁移方法

    Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach. (arXiv:2401.10747v1 [cs.SD])

    [http://arxiv.org/abs/2401.10747](http://arxiv.org/abs/2401.10747)

    本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。

    

    多模态情感分析旨在通过视觉、语言和声音线索来识别个体表达的情绪。然而，现有研究大多假设在训练和测试过程中所有模态都是可用的，这使得它们的算法容易受到缺失模态的影响。在本文中，我们提出了一种新颖的知识迁移网络，用于在不同模态之间进行翻译，以重构缺失的音频模态。此外，我们还开发了一种跨模态注意机制，以保留重构和观察到的模态的最大信息，用于情感预测。在三个公开数据集上进行的大量实验证明了相对于基线算法的显著改进，并实现了与具有完整多模态监督的先前方法相媲美的结果。

    Multimodal sentiment analysis aims to identify the emotions expressed by individuals through visual, language, and acoustic cues. However, most of the existing research efforts assume that all modalities are available during both training and testing, making their algorithms susceptible to the missing modality scenario. In this paper, we propose a novel knowledge-transfer network to translate between different modalities to reconstruct the missing audio modalities. Moreover, we develop a cross-modality attention mechanism to retain the maximal information of the reconstructed and observed modalities for sentiment prediction. Extensive experiments on three publicly available datasets demonstrate significant improvements over baselines and achieve comparable results to the previous methods with complete multi-modality supervision.
    
[^8]: $G$-Mapper：学习Mapper构造中的覆盖

    $G$-Mapper: Learning a Cover in the Mapper Construction. (arXiv:2309.06634v1 [cs.LG])

    [http://arxiv.org/abs/2309.06634](http://arxiv.org/abs/2309.06634)

    本论文介绍了一种基于统计检验和聚类算法的优化Mapper图覆盖的方法，通过分割覆盖选择生成了保留数据集本质的Mapper图。

    

    Mapper算法是拓扑数据分析(TDA)中一种反映给定数据集结构的可视化技术。Mapper算法需要调整多个参数以生成一个"好看的"Mapper图。该论文关注于选择覆盖参数。我们提出了一种通过根据正态性的统计检验反复分割覆盖来优化Mapper图的算法。我们的算法基于$G$-means聚类，通过迭代地进行Anderson-Darling检验来寻找$k$-means中最佳的簇数。我们的分割过程利用高斯混合模型，根据给定数据的分布精心选择覆盖。对于合成和真实数据集的实验表明，我们的算法生成的覆盖使Mapper图保留了数据集的本质。

    The Mapper algorithm is a visualization technique in topological data analysis (TDA) that outputs a graph reflecting the structure of a given dataset. The Mapper algorithm requires tuning several parameters in order to generate a "nice" Mapper graph. The paper focuses on selecting the cover parameter. We present an algorithm that optimizes the cover of a Mapper graph by splitting a cover repeatedly according to a statistical test for normality. Our algorithm is based on $G$-means clustering which searches for the optimal number of clusters in $k$-means by conducting iteratively the Anderson-Darling test. Our splitting procedure employs a Gaussian mixture model in order to choose carefully the cover based on the distribution of a given data. Experiments for synthetic and real-world datasets demonstrate that our algorithm generates covers so that the Mapper graphs retain the essence of the datasets.
    
[^9]: 通过密度匹配实现的合成对照方法下的隐式内生性问题

    Synthetic Control Methods by Density Matching under Implicit Endogeneitiy. (arXiv:2307.11127v1 [econ.EM])

    [http://arxiv.org/abs/2307.11127](http://arxiv.org/abs/2307.11127)

    本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。

    

    合成对照方法（SCMs）已成为比较案例研究中因果推断的重要工具。SCMs的基本思想是通过使用来自未处理单元的观测结果的加权和来估计经过处理单元的反事实结果。合成对照（SC）的准确性对于估计因果效应至关重要，因此，SC权重的估计成为了研究的焦点。在本文中，我们首先指出现有的SCMs存在一个隐式内生性问题，即未处理单元的结果与反事实结果模型中的误差项之间的相关性。我们展示了这个问题会对因果效应估计器产生偏差。然后，我们提出了一种基于密度匹配的新型SCM，假设经过处理单元的结果密度可以用未处理单元的密度的加权平均来近似（即混合模型）。基于这一假设，我们通过匹配来估计SC权重。

    Synthetic control methods (SCMs) have become a crucial tool for causal inference in comparative case studies. The fundamental idea of SCMs is to estimate counterfactual outcomes for a treated unit by using a weighted sum of observed outcomes from untreated units. The accuracy of the synthetic control (SC) is critical for estimating the causal effect, and hence, the estimation of SC weights has been the focus of much research. In this paper, we first point out that existing SCMs suffer from an implicit endogeneity problem, which is the correlation between the outcomes of untreated units and the error term in the model of a counterfactual outcome. We show that this problem yields a bias in the causal effect estimator. We then propose a novel SCM based on density matching, assuming that the density of outcomes of the treated unit can be approximated by a weighted average of the densities of untreated units (i.e., a mixture model). Based on this assumption, we estimate SC weights by matchi
    
[^10]: 人类和机器有相同的眼睛吗？基于图像分类的人机感知差异研究

    Do humans and machines have the same eyes? Human-machine perceptual differences on image classification. (arXiv:2304.08733v1 [cs.CV])

    [http://arxiv.org/abs/2304.08733](http://arxiv.org/abs/2304.08733)

    本文研究通过图像分类探究了人机感知差异，发现即使准确率相似，人类和机器的答案分布也可能不同，并提出了一种后期人机合作来提高任务表现。

    

    训练良好的计算机视觉模型通常通过模仿从训练标签中学到的人类行为来解决视觉任务。近期视觉研究的大部分努力集中在使用标准化基准来测量模型任务性能。然而，了解人与机器之间的感知差异方面的工作还很有限。为了填补这一空白，我们的研究首先量化并分析了两种来源错误的统计分布。然后我们通过难度级别对任务进行排序，探讨人类与机器专业知识的差异。即使人类和机器的整体准确性相似，答案的分布也可能会有所不同。利用人类和机器之间的感知差异，我们通过实证研究表明了一种后期人机合作，其表现比单独的人或机器更好。

    Trained computer vision models are assumed to solve vision tasks by imitating human behavior learned from training labels. Most efforts in recent vision research focus on measuring the model task performance using standardized benchmarks. Limited work has been done to understand the perceptual difference between humans and machines. To fill this gap, our study first quantifies and analyzes the statistical distributions of mistakes from the two sources. We then explore human vs. machine expertise after ranking tasks by difficulty levels. Even when humans and machines have similar overall accuracies, the distribution of answers may vary. Leveraging the perceptual difference between humans and machines, we empirically demonstrate a post-hoc human-machine collaboration that outperforms humans or machines alone.
    
[^11]: repliclust：聚类分析的合成数据

    repliclust: Synthetic Data for Cluster Analysis. (arXiv:2303.14301v1 [cs.LG])

    [http://arxiv.org/abs/2303.14301](http://arxiv.org/abs/2303.14301)

    repliclust 是一个 Python 包，用于生成具有聚类的合成数据集，基于数据集的原型，提供了放置集群中心、采样集群形状、选择每个集群的数据点数量以及为集群分配概率分布的算法。

    

    我们介绍了 repliclust（来自于 repli-cate 和 clust-er），这是一个用于生成具有聚类的合成数据集的 Python 包。我们的方法基于数据集的原型，即高级几何描述，用户可以从中创建许多不同的数据集，并具有所需的几何特性。我们软件的架构是模块化和面向对象的，将数据生成分解成放置集群中心的算法、采样集群形状的算法、选择每个集群的数据点数量的算法以及为集群分配概率分布的算法。repliclust.org 项目网页提供了简明的用户指南和全面的文档。

    We present repliclust (from repli-cate and clust-er), a Python package for generating synthetic data sets with clusters. Our approach is based on data set archetypes, high-level geometric descriptions from which the user can create many different data sets, each possessing the desired geometric characteristics. The architecture of our software is modular and object-oriented, decomposing data generation into algorithms for placing cluster centers, sampling cluster shapes, selecting the number of data points for each cluster, and assigning probability distributions to clusters. The project webpage, repliclust.org, provides a concise user guide and thorough documentation.
    
[^12]: 粒球优化算法

    Granular-ball Optimization Algorithm. (arXiv:2303.12807v1 [cs.LG])

    [http://arxiv.org/abs/2303.12807](http://arxiv.org/abs/2303.12807)

    粒球优化算法(GBO)是一种新的多粒度优化算法，可以通过引入粒球计算来提高全局搜索能力和收敛速度，实验结果表明，在这些方面它比现有的最先进的算法表现更优。

    

    现有的智能优化算法都是基于最小粒度即点的设计，导致全局搜索能力较弱且效率低下。为了解决这个问题，我们提出了一种新的多粒度优化算法，即粒球优化算法(GBO)，通过引入粒球计算来实现。GBO使用多个粒球来覆盖解空间，使用许多细小的细粒度粒球来描述重要部分，使用少量的大粗粒度粒球来描述不重要的部分，精细的多粒度数据描述能力提高了全局搜索能力和收敛速度。针对二十个基准函数的实验结果表明，与最流行的最先进的算法相比，GBO具有更好的性能和更快的速度，更接近最优解，没有超参数，设计更简单。

    The existing intelligent optimization algorithms are designed based on the finest granularity, i.e., a point. This leads to weak global search ability and inefficiency. To address this problem, we proposed a novel multi-granularity optimization algorithm, namely granular-ball optimization algorithm (GBO), by introducing granular-ball computing. GBO uses many granular-balls to cover the solution space. Quite a lot of small and fine-grained granular-balls are used to depict the important parts, and a little number of large and coarse-grained granular-balls are used to depict the inessential parts. Fine multi-granularity data description ability results in a higher global search capability and faster convergence speed. In comparison with the most popular and state-of-the-art algorithms, the experiments on twenty benchmark functions demonstrate its better performance. The faster speed, higher approximation ability of optimal solution, no hyper-parameters, and simpler design of GBO make it 
    
[^13]: 通过基于草图的顺序二次规划对约束的随机优化进行统计推断

    Statistical Inference of Constrained Stochastic Optimization via Sketched Sequential Quadratic Programming. (arXiv:2205.13687v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2205.13687](http://arxiv.org/abs/2205.13687)

    本篇论文提出了一种用于等式约束的随机非线性优化问题的统计推断方法，通过基于草图的顺序二次规划（StoSQP）进行求解，并且允许自适应选择随机步长和使用高效随机迭代求解器来降低计算成本。

    

    我们考虑对等式约束的随机非线性优化问题进行统计推断。我们开发了一种全在线随机顺序二次规划（StoSQP）方法来解决这些问题，可以将其视为将牛顿法应用于一阶最优性条件（即KKT条件）。受最近数值二阶方法设计的启发，我们允许StoSQP自适应地选择任意随机步长$ \bar {\ alpha} _t $，只要$ \ beta _t \ leq \ bar {\ alpha} _t \ leq \ beta _t + \ chi _t $，其中 $ \ beta_t $ 和 $ \ chi_t = o(\beta_t) $ 是某些控制序列。为了降低二阶方法的主要计算成本，我们还允许StoSQP通过使用草图技术的高效随机迭代求解器来不精确地解决二次规划问题。值得注意的是，我们不要求逼近误差随着迭代的进行而减小。对于开发的方法，我们证明在温和的假设（i）下，它的计算复杂度最多为$ O(1 / \ ep）$。

    We consider statistical inference of equality-constrained stochastic nonlinear optimization problems. We develop a fully online stochastic sequential quadratic programming (StoSQP) method to solve the problems, which can be regarded as applying Newton's method to the first-order optimality conditions (i.e., the KKT conditions). Motivated by recent designs of numerical second-order methods, we allow StoSQP to adaptively select any random stepsize $\bar{\alpha}_t$, as long as $\beta_t\leq \bar{\alpha}_t \leq \beta_t+\chi_t$, for some control sequences $\beta_t$ and $\chi_t=o(\beta_t)$. To reduce the dominant computational cost of second-order methods, we additionally allow StoSQP to inexactly solve quadratic programs via efficient randomized iterative solvers that utilize sketching techniques. Notably, we do not require the approximation error to diminish as iteration proceeds. For the developed method, we show that under mild assumptions (i) computationally, it can take at most $O(1/\ep
    

