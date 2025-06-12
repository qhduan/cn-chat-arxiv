# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knowing Your Nonlinearities: Shapley Interactions Reveal the Underlying Structure of Data](https://arxiv.org/abs/2403.13106) | 该论文使用Shapley Taylor互动指数（STII）分析了底层数据结构对各种模态、任务和架构中模型表征的影响，发现了语言模型和语音模型中的新颖现象，并展示了特征交互如何直观反映对象边界。 |
| [^2] | [A Comparative Study of Conventional and Tripolar EEG for High-Performance Reach-to-Grasp BCI Systems](https://arxiv.org/abs/2402.09448) | 比较传统EEG与三极EEG在高性能到颤抓握BCI系统中的有效性，包括信噪比、空间分辨率、ERPs和小波时频分析。 |
| [^3] | [Forecasting high-impact research topics via machine learning on evolving knowledge graphs](https://arxiv.org/abs/2402.08640) | 通过机器学习预测未发布研究想法的影响力，我们使用一个由超过2100万篇科学论文构建的演化知识图谱，结合论文内容和历史引用的信息，高准确度预测未来的演化网络动态和新的研究方向的影响力。 |
| [^4] | [Plug-and-Play image restoration with Stochastic deNOising REgularization](https://arxiv.org/abs/2402.01779) | 本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。 |
| [^5] | [Learning Concepts Definable in First-Order Logic with Counting](https://arxiv.org/abs/1909.03820) | 该研究将一阶逻辑与计数符号相结合，证明了可以在多对数度结构下以次线性时间一致学习可定义的分类器，为包含数值方面的机器学习扩展学习框架迈出了第一步。 |
| [^6] | [Generating Likely Counterfactuals Using Sum-Product Networks.](http://arxiv.org/abs/2401.14086) | 由于用户需求和最近的法规要求，需要对AI系统所做出的决策进行解释。本论文提出了一种使用Sum-Product Networks模拟寻找高可能性反事实推理的系统，该系统能够提供满足多个常见要求的最佳解释。 |
| [^7] | [Byzantine-Resilient Decentralized Multi-Armed Bandits.](http://arxiv.org/abs/2310.07320) | 这篇论文介绍了一种拜占庭容错的分散式多臂赌博机算法，通过信息混合和值的截断实现了对拜占庭代理的恢复和鲁棒性。 |
| [^8] | [Feature Normalization Prevents Collapse of Non-contrastive Learning Dynamics.](http://arxiv.org/abs/2309.16109) | 本论文研究了非对比学习中的动力崩溃问题，发现特征归一化可以防止此问题的出现，为解决自监督表示学习的计算效率提供了新的思路。 |
| [^9] | [Simulation-based inference using surjective sequential neural likelihood estimation.](http://arxiv.org/abs/2308.01054) | 我们提出了一种使用全射序列神经似然估计（SSNL）进行基于仿真的推断的新方法，在模型中无法计算似然函数并且只能使用模拟器生成数据的情况下，SSNL通过拟合降维的全射归一化流模型，并将其作为替代似然函数，解决了先前基于似然方法在高维数据集中遇到的问题，并在各种实验中展示了其优越性能。 |
| [^10] | [Optimal Estimation in Mixed-Membership Stochastic Block Models.](http://arxiv.org/abs/2307.14530) | 本论文研究了重叠社区检测问题，在混合成员随机块模型的基础上提出了一个新的估计器，并建立了估计误差的极小下界。 |

# 详细

[^1]: 认识你的非线性：Shapley互动揭示数据的潜在结构

    Knowing Your Nonlinearities: Shapley Interactions Reveal the Underlying Structure of Data

    [https://arxiv.org/abs/2403.13106](https://arxiv.org/abs/2403.13106)

    该论文使用Shapley Taylor互动指数（STII）分析了底层数据结构对各种模态、任务和架构中模型表征的影响，发现了语言模型和语音模型中的新颖现象，并展示了特征交互如何直观反映对象边界。

    

    测量非线性特征交互是理解许多模型中复杂归因模式的一种已建立的方法。本文使用Shapley Taylor互动指数（STII）来分析底层数据结构对多种模态、任务和架构中模型表征的影响。在考虑掩码和自回归语言模型（MLMs和ALMs）中的语言结构时，我们发现STII在惯用表达中增加，MLMs随句法距离扩展STII，更多地依赖语法在其非线性结构中相比ALMs。我们的语音模型研究反映了口腔张开程度决定音素根据上下文变化的数量的原则。最后，我们研究图像分类器并说明特征交互直观反映对象边界。我们广泛的结果展示了跨学科工作和领域之间的益处。

    arXiv:2403.13106v1 Announce Type: cross  Abstract: Measuring nonlinear feature interaction is an established approach to understanding complex patterns of attribution in many models. In this paper, we use Shapley Taylor interaction indices (STII) to analyze the impact of underlying data structure on model representations in a variety of modalities, tasks, and architectures. Considering linguistic structure in masked and auto-regressive language models (MLMs and ALMs), we find that STII increases within idiomatic expressions and that MLMs scale STII with syntactic distance, relying more on syntax in their nonlinear structure than ALMs do. Our speech model findings reflect the phonetic principal that the openness of the oral cavity determines how much a phoneme varies based on its context. Finally, we study image classifiers and illustrate that feature interactions intuitively reflect object boundaries. Our wide range of results illustrates the benefits of interdisciplinary work and doma
    
[^2]: 普通EEG与三极EEG在高性能到颤抓握BCI系统中的比较研究

    A Comparative Study of Conventional and Tripolar EEG for High-Performance Reach-to-Grasp BCI Systems

    [https://arxiv.org/abs/2402.09448](https://arxiv.org/abs/2402.09448)

    比较传统EEG与三极EEG在高性能到颤抓握BCI系统中的有效性，包括信噪比、空间分辨率、ERPs和小波时频分析。

    

    本研究旨在比较传统EEG与三极EEG在提升运动障碍个体的BCI应用方面的有效性。重点是解读和解码各种抓握动作，如力握和精确握持。目标是确定哪种EEG技术在处理和翻译与抓握相关的脑电信号方面更为有效。研究涉及对十名健康参与者进行实验，参与者进行了两种不同的握持运动：力握和精确握持，无运动条件作为基线。我们的研究在解码抓握动作方面对EEG和三极EEG进行了全面比较。该比较涵盖了几个关键参数，包括信噪比（SNR）、通过功能连接的空间分辨率、ERPs和小波时频分析。此外，我们的研究还涉及从...

    arXiv:2402.09448v1 Announce Type: cross  Abstract: This study aims to enhance BCI applications for individuals with motor impairments by comparing the effectiveness of tripolar EEG (tEEG) with conventional EEG. The focus is on interpreting and decoding various grasping movements, such as power grasp and precision grasp. The goal is to determine which EEG technology is more effective in processing and translating grasp related neural signals. The approach involved experimenting on ten healthy participants who performed two distinct grasp movements: power grasp and precision grasp, with a no movement condition serving as the baseline. Our research presents a thorough comparison between EEG and tEEG in decoding grasping movements. This comparison spans several key parameters, including signal to noise ratio (SNR), spatial resolution via functional connectivity, ERPs, and wavelet time frequency analysis. Additionally, our study involved extracting and analyzing statistical features from th
    
[^3]: 通过机器学习在不断演化的知识图谱上预测高影响力的研究主题

    Forecasting high-impact research topics via machine learning on evolving knowledge graphs

    [https://arxiv.org/abs/2402.08640](https://arxiv.org/abs/2402.08640)

    通过机器学习预测未发布研究想法的影响力，我们使用一个由超过2100万篇科学论文构建的演化知识图谱，结合论文内容和历史引用的信息，高准确度预测未来的演化网络动态和新的研究方向的影响力。

    

    科学出版物的指数增长对人类研究者构成了严峻挑战。它迫使研究者将注意力集中在更狭窄的子领域上，使得发现其他领域的新颖且有影响力的研究想法和合作变得困难。虽然有办法预测科学论文未来的引用次数，但通常需要等到研究完成并且论文写成后才能进行评估，这样就错过了想法构思的早期阶段。在本文中，我们展示了如何预测从未被研究者发布的想法的影响力。为此，我们开发了一个大型的演化知识图谱，其中包含超过2100万篇科学论文。它结合了从论文内容中创建的语义网络和从历史引用中创建的影响网络。利用机器学习，我们可以高准确度地预测演化网络的动态情况，从而预测新的研究方向的影响力。我们预期这种能力将有助于研究者发现具有高影响力的研究主题。

    The exponential growth in scientific publications poses a severe challenge for human researchers. It forces attention to more narrow sub-fields, which makes it challenging to discover new impactful research ideas and collaborations outside one's own field. While there are ways to predict a scientific paper's future citation counts, they need the research to be finished and the paper written, usually assessing impact long after the idea was conceived. Here we show how to predict the impact of onsets of ideas that have never been published by researchers. For that, we developed a large evolving knowledge graph built from more than 21 million scientific papers. It combines a semantic network created from the content of the papers and an impact network created from the historic citations of papers. Using machine learning, we can predict the dynamic of the evolving network into the future with high accuracy, and thereby the impact of new research directions. We envision that the ability to 
    
[^4]: 带有随机去噪正则化的即插即用图像恢复

    Plug-and-Play image restoration with Stochastic deNOising REgularization

    [https://arxiv.org/abs/2402.01779](https://arxiv.org/abs/2402.01779)

    本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。

    

    即插即用（PnP）算法是一类迭代算法，通过结合物理模型和深度神经网络进行正则化来解决图像反演问题。尽管这些算法能够产生令人印象深刻的图像恢复结果，但它们依赖于在迭代过程中越来越少噪音的图像上的一种非标准的去噪器使用方法，这与基于扩散模型（DM）的最新算法相矛盾，在这些算法中，去噪器仅应用于重新加噪的图像上。我们提出了一种新的PnP框架，称为随机去噪正则化（SNORE），它仅在噪声水平适当的图像上应用去噪器。它基于显式的随机正则化，从而导致了一种解决病态逆问题的随机梯度下降算法。我们提供了该算法及其退火扩展的收敛分析。在实验上，我们证明SNORE在去模糊和修复任务上与最先进的方法相竞争。

    Plug-and-Play (PnP) algorithms are a class of iterative algorithms that address image inverse problems by combining a physical model and a deep neural network for regularization. Even if they produce impressive image restoration results, these algorithms rely on a non-standard use of a denoiser on images that are less and less noisy along the iterations, which contrasts with recent algorithms based on Diffusion Models (DM), where the denoiser is applied only on re-noised images. We propose a new PnP framework, called Stochastic deNOising REgularization (SNORE), which applies the denoiser only on images with noise of the adequate level. It is based on an explicit stochastic regularization, which leads to a stochastic gradient descent algorithm to solve ill-posed inverse problems. A convergence analysis of this algorithm and its annealing extension is provided. Experimentally, we prove that SNORE is competitive with respect to state-of-the-art methods on deblurring and inpainting tasks, 
    
[^5]: 用计数符号的一阶逻辑定义的概念的学习

    Learning Concepts Definable in First-Order Logic with Counting

    [https://arxiv.org/abs/1909.03820](https://arxiv.org/abs/1909.03820)

    该研究将一阶逻辑与计数符号相结合，证明了可以在多对数度结构下以次线性时间一致学习可定义的分类器，为包含数值方面的机器学习扩展学习框架迈出了第一步。

    

    我们研究了在Grohe和Tur\'an引入的逻辑框架下的关系背景结构上的布尔分类问题。众所周知(Grohe和Ritzert, LICS 2017)，在多对数度结构上的一阶逻辑可定义的分类器可以在次线性时间内学习，其中结构的度和运行时间是以结构的大小为单位来衡量的。我们将结果推广到了由Kuske和Schweikardt(LICS 2017)引入的带计数的一阶逻辑FOCN，它作为一个广泛推广各种计数逻辑的表现逻辑。具体来说，我们证明了可以在多对数度结构类上定义的FOCN中的分类器可以在次线性时间内一致地学习。这可以看作是将学习框架扩展以包含机器学习的数值方面的第一步。我们将这一结果扩展到了无视的概率

    arXiv:1909.03820v2 Announce Type: replace-cross  Abstract: We study Boolean classification problems over relational background structures in the logical framework introduced by Grohe and Tur\'an (TOCS 2004). It is known (Grohe and Ritzert, LICS 2017) that classifiers definable in first-order logic over structures of polylogarithmic degree can be learned in sublinear time, where the degree of the structure and the running time are measured in terms of the size of the structure. We generalise the results to the first-order logic with counting FOCN, which was introduced by Kuske and Schweikardt (LICS 2017) as an expressive logic generalising various other counting logics. Specifically, we prove that classifiers definable in FOCN over classes of structures of polylogarithmic degree can be consistently learned in sublinear time. This can be seen as a first step towards extending the learning framework to include numerical aspects of machine learning. We extend the result to agnostic probabl
    
[^6]: 使用Sum-Product Networks生成可能的反事实推理

    Generating Likely Counterfactuals Using Sum-Product Networks. (arXiv:2401.14086v1 [cs.AI])

    [http://arxiv.org/abs/2401.14086](http://arxiv.org/abs/2401.14086)

    由于用户需求和最近的法规要求，需要对AI系统所做出的决策进行解释。本论文提出了一种使用Sum-Product Networks模拟寻找高可能性反事实推理的系统，该系统能够提供满足多个常见要求的最佳解释。

    

    由于用户需求和最近的法规（GDPR、AI法案），需要解释AI系统所做出的决策。这些决策往往只能在事后解释，反事实推理成为常见的解释方式。什么构成了最佳的反事实解释必须考虑多个方面，其中“样本距离”是最常见的。我们认为，这一要求经常会导致不太可能且因此价值有限的解释。在这里，我们提出了一个能够提供高可能性解释的系统。我们展示了使用混合整数优化（MIO）模拟寻找满足反事实推理的许多常见要求的最有可能解释。在此过程中，我们提出了Sum-Product Network（SPN）的MIO表达，并使用SPN估计反事实的可能性，这对独立的兴趣也有用。与生成反事实解释的几种方法进行数值比较。

    Due to user demand and recent regulation (GDPR, AI Act), decisions made by AI systems need to be explained. These decisions are often explainable only post hoc, where counterfactual explanations are popular. The question of what constitutes the best counterfactual explanation must consider multiple aspects, where "distance from the sample" is the most common. We argue that this requirement frequently leads to explanations that are unlikely and, therefore, of limited value. Here, we present a system that provides high-likelihood explanations. We show that the search for the most likely explanations satisfying many common desiderata for counterfactual explanations can be modeled using mixed-integer optimization (MIO). In the process, we propose an MIO formulation of a Sum-Product Network (SPN) and use the SPN to estimate the likelihood of a counterfactual, which can be of independent interest. A numerical comparison against several methods for generating counterfactual explanations is pr
    
[^7]: 拜占庭容错的分散式多臂赌博机算法

    Byzantine-Resilient Decentralized Multi-Armed Bandits. (arXiv:2310.07320v1 [cs.LG])

    [http://arxiv.org/abs/2310.07320](http://arxiv.org/abs/2310.07320)

    这篇论文介绍了一种拜占庭容错的分散式多臂赌博机算法，通过信息混合和值的截断实现了对拜占庭代理的恢复和鲁棒性。

    

    在分散式合作的多臂赌博机中，每个代理观察到不同的奖励流，试图与其他代理交换信息以选择一系列手臂以最小化遗憾。与独立运行上界置信度（UCB）等多臂赌博机方法相比，协作场景中的代理可以表现得更好。本文研究了如何在未知比例的代理可能是拜占庭（即，以奖励均值估计或置信度集的形式传递任意错误信息）时恢复此类突出行为。该框架可用于模拟计算机网络中的攻击者，向推荐系统中插入攻击性内容的策划者，或者金融市场的操纵者。我们的主要贡献是开发了一种完全分散的具有容错上界置信度（UCB）算法，该算法将代理间的信息混合步骤与不一致和极端值的截断相结合。这个截断步骤使我们能够建立

    In decentralized cooperative multi-armed bandits (MAB), each agent observes a distinct stream of rewards, and seeks to exchange information with others to select a sequence of arms so as to minimize its regret. Agents in the cooperative setting can outperform a single agent running a MAB method such as Upper-Confidence Bound (UCB) independently. In this work, we study how to recover such salient behavior when an unknown fraction of the agents can be Byzantine, that is, communicate arbitrarily wrong information in the form of reward mean-estimates or confidence sets. This framework can be used to model attackers in computer networks, instigators of offensive content into recommender systems, or manipulators of financial markets. Our key contribution is the development of a fully decentralized resilient upper confidence bound (UCB) algorithm that fuses an information mixing step among agents with a truncation of inconsistent and extreme values. This truncation step enables us to establis
    
[^8]: 特征归一化防止非对比学习动力的崩溃

    Feature Normalization Prevents Collapse of Non-contrastive Learning Dynamics. (arXiv:2309.16109v1 [cs.LG])

    [http://arxiv.org/abs/2309.16109](http://arxiv.org/abs/2309.16109)

    本论文研究了非对比学习中的动力崩溃问题，发现特征归一化可以防止此问题的出现，为解决自监督表示学习的计算效率提供了新的思路。

    

    对比学习是一种自监督表示学习框架，通过数据增强生成的两个正视图在数据表示空间中通过吸引力使它们相似，而通过排斥力使它们远离负样本。非对比学习通过BYOL和SimSiam等手段去除了负样本，并提高了计算效率。虽然由于缺乏排斥力，学到的表示可能会崩溃成一个单点，但田等人（2021）通过学习动力分析揭示，如果数据增强足够强于正则化，则表示可以避免崩溃。然而，他们的分析没有考虑常用的特征归一化，即在衡量表示相似性之前进行的归一化操作，因此过强的正则化可能会导致动力崩溃，这在特征归一化存在的情况下是不自然的行为。

    Contrastive learning is a self-supervised representation learning framework, where two positive views generated through data augmentation are made similar by an attraction force in a data representation space, while a repulsive force makes them far from negative examples. Non-contrastive learning, represented by BYOL and SimSiam, further gets rid of negative examples and improves computational efficiency. While learned representations may collapse into a single point due to the lack of the repulsive force at first sight, Tian et al. (2021) revealed through the learning dynamics analysis that the representations can avoid collapse if data augmentation is sufficiently stronger than regularization. However, their analysis does not take into account commonly-used feature normalization, a normalizer before measuring the similarity of representations, and hence excessively strong regularization may collapse the dynamics, which is an unnatural behavior under the presence of feature normalizat
    
[^9]: 使用全射序列神经似然估计进行基于仿真的推断

    Simulation-based inference using surjective sequential neural likelihood estimation. (arXiv:2308.01054v1 [stat.ML])

    [http://arxiv.org/abs/2308.01054](http://arxiv.org/abs/2308.01054)

    我们提出了一种使用全射序列神经似然估计（SSNL）进行基于仿真的推断的新方法，在模型中无法计算似然函数并且只能使用模拟器生成数据的情况下，SSNL通过拟合降维的全射归一化流模型，并将其作为替代似然函数，解决了先前基于似然方法在高维数据集中遇到的问题，并在各种实验中展示了其优越性能。

    

    我们提出了全射序列神经似然（SSNL）估计方法，这是一种在模型中无法计算似然函数并且只能使用可以生成合成数据的模拟器时进行基于仿真的推断的新方法。SSNL拟合一个降维的全射归一化流模型，并将其用作替代似然函数，从而可以使用传统的贝叶斯推断方法，包括马尔科夫链蒙特卡罗方法或变分推断。通过将数据嵌入到低维空间中，SSNL解决了先前基于似然方法在应用于高维数据集时遇到的几个问题，例如包含无信息数据维度或位于较低维流形上的数据。我们对SSNL在各种实验中进行了评估，并表明它通常优于在基于仿真推断中使用的现代方法，例如在一项来自天体物理学的具有挑战性的真实世界例子上对磁场模型的建模。

    We present Surjective Sequential Neural Likelihood (SSNL) estimation, a novel method for simulation-based inference in models where the evaluation of the likelihood function is not tractable and only a simulator that can generate synthetic data is available. SSNL fits a dimensionality-reducing surjective normalizing flow model and uses it as a surrogate likelihood function which allows for conventional Bayesian inference using either Markov chain Monte Carlo methods or variational inference. By embedding the data in a low-dimensional space, SSNL solves several issues previous likelihood-based methods had when applied to high-dimensional data sets that, for instance, contain non-informative data dimensions or lie along a lower-dimensional manifold. We evaluate SSNL on a wide variety of experiments and show that it generally outperforms contemporary methods used in simulation-based inference, for instance, on a challenging real-world example from astrophysics which models the magnetic fi
    
[^10]: 混合成员随机块模型中的最优估计

    Optimal Estimation in Mixed-Membership Stochastic Block Models. (arXiv:2307.14530v1 [stat.ML])

    [http://arxiv.org/abs/2307.14530](http://arxiv.org/abs/2307.14530)

    本论文研究了重叠社区检测问题，在混合成员随机块模型的基础上提出了一个新的估计器，并建立了估计误差的极小下界。

    

    社区检测是现代网络科学中最关键的问题之一。其应用可以在各个领域找到，从蛋白质建模到社交网络分析。最近，出现了许多论文研究重叠社区检测问题，即网络中的每个节点可能属于多个社区。在本文中，我们考虑了由Airoldi等人（2008）首次提出的混合成员随机块模型（MMSB）。MMSB在图中对重叠社区结构提供了相当一般的设置。本文的核心问题是在观察到的网络中重建社区之间的关系。我们比较了不同的方法，并建立了估计误差的极小下界。然后，我们提出了一个与这个下界匹配的新估计器。理论结果在对所考虑的模型的相当普遍条件下得到证明。最后，我们通过一系列实验来说明这个理论。

    Community detection is one of the most critical problems in modern network science. Its applications can be found in various fields, from protein modeling to social network analysis. Recently, many papers appeared studying the problem of overlapping community detection, where each node of a network may belong to several communities. In this work, we consider Mixed-Membership Stochastic Block Model (MMSB) first proposed by Airoldi et al. (2008). MMSB provides quite a general setting for modeling overlapping community structure in graphs. The central question of this paper is to reconstruct relations between communities given an observed network. We compare different approaches and establish the minimax lower bound on the estimation error. Then, we propose a new estimator that matches this lower bound. Theoretical results are proved under fairly general conditions on the considered model. Finally, we illustrate the theory in a series of experiments.
    

