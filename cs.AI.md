# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From Similarity to Superiority: Channel Clustering for Time Series Forecasting](https://arxiv.org/abs/2404.01340) | 通过对通道进行聚类，实现了一种新颖的通道策略，有效平衡了个体通道处理和通道之间必要交互作用，从而提高了时间序列预测性能。 |
| [^2] | [Initialisation and Topology Effects in Decentralised Federated Learning](https://arxiv.org/abs/2403.15855) | 分散式联邦学习的有效性受到连接设备网络拓扑结构的显著影响，我们提出了基于底层网络节点特征向量中心性分布的改进神经网络初始化策略，大大提高了训练效率。 |
| [^3] | [Policy Mirror Descent with Lookahead](https://arxiv.org/abs/2403.14156) | 提出了一种新类别的策略镜像下降算法$h$-PMD，它通过在PMD更新规则中结合多步贪心策略改进和前瞻深度$h，以解决折扣无限时间视角下的马尔可夫决策过程。 |
| [^4] | [A Learnable Prior Improves Inverse Tumor Growth Modeling](https://arxiv.org/abs/2403.04500) | 提出了一种新颖的框架，以结合深度学习集成进行初始参数估计，促进了有效的下游演化抽样，有效地估计了从磁共振图像中脑肿瘤细胞浓度，DL-Prior在其中起着关键作用 |
| [^5] | [Ensemble-Based Unsupervised Discontinuous Constituency Parsing by Tree Averaging](https://arxiv.org/abs/2403.00143) | 通过树平均法构建集成解析器，稳定并提升无监督不连续成分句法分析性能，实验结果表明该方法在所有指标上均优于基准线 |
| [^6] | [Deep Learning Based Amharic Chatbot for FAQs in Universities](https://arxiv.org/abs/2402.01720) | 本文提出了一个基于深度学习的阿姆哈拉语常见问题解答聊天机器人模型，可以帮助大学生解答常见问题，通过使用自然语言处理和深度学习技术，采用多种机器学习模型算法进行分析和分类，取得了最好的成绩。 |
| [^7] | [A dynamical clipping approach with task feedback for Proximal Policy Optimization](https://arxiv.org/abs/2312.07624) | 提出了一种基于任务反馈的动态剪裁方法，通过增加最大累积回报来优化近端策略优化的性能。 |
| [^8] | [Incorporating Riemannian Geometric Features for Learning Coefficient of Pressure Distributions on Airplane Wings.](http://arxiv.org/abs/2401.09452) | 该论文提出了一种将黎曼几何特征应用于学习翼面压力系数分布的方法，以提高气动系数的预测准确性。 |
| [^9] | [Teach Large Language Models to Forget Privacy.](http://arxiv.org/abs/2401.00870) | 这项研究提出了Prompt2Forget（P2F）框架，通过教导大型语言模型（LLM）忘记隐私信息，解决了LLM本地隐私挑战。P2F方法将问题分解为片段并生成虚构答案，模糊化模型对原始输入的记忆。实验证明，P2F具有很强的模糊化能力，并且可以在各种应用场景下自适应使用，无需手动设置。 |
| [^10] | [D3: Data Diversity Design for Systematic Generalization in Visual Question Answering.](http://arxiv.org/abs/2309.08798) | 本论文研究了视觉问答中系统一般化的关键因素，发现简单任务的多样性在实现系统一般化中起到了重要的作用，这意味着不必收集大量和多样的复杂任务。 |
| [^11] | [Breaking the Metric Voting Distortion Barrier.](http://arxiv.org/abs/2306.17838) | 这篇论文研究了社会选择中的度量扭曲问题，提出了一个新的投票规则，该规则可以选择距离选民平均距离较小的候选人。之前的研究发现确定性投票规则的度量扭曲限制为3，但在无限制的情况下，我们对该问题仍然了解有限。 |
| [^12] | [Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics.](http://arxiv.org/abs/2306.10656) | 本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。 |
| [^13] | [CatE: Embedding $\mathcal{ALC}$ ontologies using category-theoretical semantics.](http://arxiv.org/abs/2305.07163) | 本文介绍了一种使用范畴论语义的方法CatE，该方法可以高效地将本体公理投影到图形中，并生成本体论的嵌入，从而在知识图谱的链接预测任务上取得更好的表现。 |

# 详细

[^1]: 从相似到优越：时间序列预测的通道聚类

    From Similarity to Superiority: Channel Clustering for Time Series Forecasting

    [https://arxiv.org/abs/2404.01340](https://arxiv.org/abs/2404.01340)

    通过对通道进行聚类，实现了一种新颖的通道策略，有效平衡了个体通道处理和通道之间必要交互作用，从而提高了时间序列预测性能。

    

    时间序列预测在最近几十年吸引了很多关注。先前的研究表明，独立通道策略通过单独处理不同通道来提高预测性能，但在未知实例上导致了差劲的泛化，并忽略了通道之间潜在的必要交互作用。相反，依赖通道策略将所有通道混合在一起，甚至包含无关紧要和随意的信息，然而这会导致过度平滑的问题并限制了预测的准确性。目前缺乏一种能够有效平衡个体通道处理以提高预测性能而又不忽视通道之间必要交互作用的通道策略。受到我们观察到的时间序列模型练习提高对混合通道的结果与一对通道之间本质相似性之间的关联的启发，我们开发了一种新颖且适应性的方法

    arXiv:2404.01340v1 Announce Type: cross  Abstract: Time series forecasting has attracted significant attention in recent decades. Previous studies have demonstrated that the Channel-Independent (CI) strategy improves forecasting performance by treating different channels individually, while it leads to poor generalization on unseen instances and ignores potentially necessary interactions between channels. Conversely, the Channel-Dependent (CD) strategy mixes all channels with even irrelevant and indiscriminate information, which, however, results in oversmoothing issues and limits forecasting accuracy. There is a lack of channel strategy that effectively balances individual channel treatment for improved forecasting performance without overlooking essential interactions between channels. Motivated by our observation of a correlation between the time series model's performance boost against channel mixing and the intrinsic similarity on a pair of channels, we developed a novel and adapt
    
[^2]: 初始值和拓扑结构在分散式联邦学习中的影响

    Initialisation and Topology Effects in Decentralised Federated Learning

    [https://arxiv.org/abs/2403.15855](https://arxiv.org/abs/2403.15855)

    分散式联邦学习的有效性受到连接设备网络拓扑结构的显著影响，我们提出了基于底层网络节点特征向量中心性分布的改进神经网络初始化策略，大大提高了训练效率。

    

    具有完全分散式特征的联邦学习使得在网络上分布式设备上对个体机器学习模型进行协作训练，同时保持训练数据本地化。这种方法增强了数据隐私性，消除了单点故障和中央协调的必要性。我们的研究强调了分散式联邦学习的有效性受到连接设备的网络拓扑结构的显著影响。一个简化的数值模型用于研究这些系统的早期行为，使我们得出了一个利用底层网络节点的特征向量中心性分布的改进人工神经网络初始值策略，从而大大提高了训练效率。此外，我们的研究探讨了在我们提出的初始化策略下的比例行为和环境参数的选择。这项工作为更多研究打开了道路。

    arXiv:2403.15855v1 Announce Type: cross  Abstract: Fully decentralised federated learning enables collaborative training of individual machine learning models on distributed devices on a network while keeping the training data localised. This approach enhances data privacy and eliminates both the single point of failure and the necessity for central coordination. Our research highlights that the effectiveness of decentralised federated learning is significantly influenced by the network topology of connected devices. A simplified numerical model for studying the early behaviour of these systems leads us to an improved artificial neural network initialisation strategy, which leverages the distribution of eigenvector centralities of the nodes of the underlying network, leading to a radically improved training efficiency. Additionally, our study explores the scaling behaviour and choice of environmental parameters under our proposed initialisation strategy. This work paves the way for mor
    
[^3]: 具有前瞻特性的策略镜像下降算法

    Policy Mirror Descent with Lookahead

    [https://arxiv.org/abs/2403.14156](https://arxiv.org/abs/2403.14156)

    提出了一种新类别的策略镜像下降算法$h$-PMD，它通过在PMD更新规则中结合多步贪心策略改进和前瞻深度$h，以解决折扣无限时间视角下的马尔可夫决策过程。

    

    策略镜像下降（PMD）作为一种多功能算法框架，包括几种重要的策略梯度算法，如自然策略梯度，并与最先进的强化学习（RL）算法（如TRPO和PPO）相联系。PMD可以看作是实现正则化1步贪心策略改进的软策略迭代算法。然而，1步贪心策略可能不是最佳选择，最近在RL领域取得了显着的实证成功，如AlphaGo和AlphaZero已经证明，相对于多步骤，贪心方法可以超越它们的1步骤对应物。在这项工作中，我们提出了一种新类别的PMD算法，称为$h$-PMD，它将具有前瞻深度$h$的多步贪心策略改进结合到PMD更新规则中。为了解决折扣无限时间视角下的马尔可夫决策过程，其中折扣因子为$\gamma$，我们展示了$h$-PMD可以推广标准的PMD。

    arXiv:2403.14156v1 Announce Type: cross  Abstract: Policy Mirror Descent (PMD) stands as a versatile algorithmic framework encompassing several seminal policy gradient algorithms such as natural policy gradient, with connections with state-of-the-art reinforcement learning (RL) algorithms such as TRPO and PPO. PMD can be seen as a soft Policy Iteration algorithm implementing regularized 1-step greedy policy improvement. However, 1-step greedy policies might not be the best choice and recent remarkable empirical successes in RL such as AlphaGo and AlphaZero have demonstrated that greedy approaches with respect to multiple steps outperform their 1-step counterpart. In this work, we propose a new class of PMD algorithms called $h$-PMD which incorporates multi-step greedy policy improvement with lookahead depth $h$ to the PMD update rule. To solve discounted infinite horizon Markov Decision Processes with discount factor $\gamma$, we show that $h$-PMD which generalizes the standard PMD enj
    
[^4]: 一种可学习的先验改进了逆肿瘤生长建模

    A Learnable Prior Improves Inverse Tumor Growth Modeling

    [https://arxiv.org/abs/2403.04500](https://arxiv.org/abs/2403.04500)

    提出了一种新颖的框架，以结合深度学习集成进行初始参数估计，促进了有效的下游演化抽样，有效地估计了从磁共振图像中脑肿瘤细胞浓度，DL-Prior在其中起着关键作用

    

    生物物理建模，特别是涉及偏微分方程（PDEs）的建模，为个体化疾病治疗方案提供了巨大潜力。然而，这些模型的逆问题求解方面提出了巨大挑战，要么是由于基于模型的方法的高计算要求，要么是深度学习（DL）方法的有限鲁棒性。我们提出了一个新颖的框架，以一种协同的方式利用两种方法的独特优势。我们的方法结合了DL集成进行初始参数估计，从而促进了初始化为基于DL的先验的有效下游进化抽样。我们展示了将快速深度学习算法与高精度演化策略结合起来在从磁共振图像中估计脑肿瘤细胞浓度方面的有效性。DL-Prior起着关键作用，显著约束了效果。

    arXiv:2403.04500v1 Announce Type: cross  Abstract: Biophysical modeling, particularly involving partial differential equations (PDEs), offers significant potential for tailoring disease treatment protocols to individual patients. However, the inverse problem-solving aspect of these models presents a substantial challenge, either due to the high computational requirements of model-based approaches or the limited robustness of deep learning (DL) methods. We propose a novel framework that leverages the unique strengths of both approaches in a synergistic manner. Our method incorporates a DL ensemble for initial parameter estimation, facilitating efficient downstream evolutionary sampling initialized with this DL-based prior. We showcase the effectiveness of integrating a rapid deep-learning algorithm with a high-precision evolution strategy in estimating brain tumor cell concentrations from magnetic resonance images. The DL-Prior plays a pivotal role, significantly constraining the effect
    
[^5]: 基于集成的无监督不连续成分句法分析：树平均法

    Ensemble-Based Unsupervised Discontinuous Constituency Parsing by Tree Averaging

    [https://arxiv.org/abs/2403.00143](https://arxiv.org/abs/2403.00143)

    通过树平均法构建集成解析器，稳定并提升无监督不连续成分句法分析性能，实验结果表明该方法在所有指标上均优于基准线

    

    我们解决了无监督不连续成分句法分析的问题，在这个问题中我们观察到先前唯一模型的性能存在高方差。我们提出通过对现有不连续解析器的不同运行构建一个集成，并通过平均预测树来稳定和提升性能。首先，我们针对不同的二元性和连续性设置提供了全面的树平均计算复杂度分析（以P和NP完全为单位）。然后，我们开发了一种高效的精确算法来处理这一任务，在我们的实验中对所有样本运行时间均合理。在三个数据集上的结果显示我们的方法在所有指标上均优于所有基准线，我们还对我们的方法进行了深入分析。

    arXiv:2403.00143v1 Announce Type: cross  Abstract: We address unsupervised discontinuous constituency parsing, where we observe a high variance in the performance of the only previous model. We propose to build an ensemble of different runs of the existing discontinuous parser by averaging the predicted trees, to stabilize and boost performance. To begin with, we provide comprehensive computational complexity analysis (in terms of P and NP-complete) for tree averaging under different setups of binarity and continuity. We then develop an efficient exact algorithm to tackle the task, which runs in a reasonable time for all samples in our experiments. Results on three datasets show our method outperforms all baselines in all metrics; we also provide in-depth analyses of our approach.
    
[^6]: 基于深度学习的阿姆哈拉语常见问题解答聊天机器人

    Deep Learning Based Amharic Chatbot for FAQs in Universities

    [https://arxiv.org/abs/2402.01720](https://arxiv.org/abs/2402.01720)

    本文提出了一个基于深度学习的阿姆哈拉语常见问题解答聊天机器人模型，可以帮助大学生解答常见问题，通过使用自然语言处理和深度学习技术，采用多种机器学习模型算法进行分析和分类，取得了最好的成绩。

    

    大学生常常花费大量时间向管理员或教师寻求常见问题的答案。这对双方来说都很繁琐，需要找到一个解决方案。为此，本文提出了一个聊天机器人模型，利用自然语言处理和深度学习技术，在阿姆哈拉语中回答常见问题。聊天机器人是通过人工智能模拟人类对话的计算机程序，作为虚拟助手处理问题和其他任务。所提出的聊天机器人程序使用标记化、规范化、去除停用词和词干提取对阿姆哈拉语输入句子进行分析和分类。采用了三种机器学习模型算法来分类标记和检索合适的回答：支持向量机（SVM）、多项式朴素贝叶斯和通过TensorFlow、Keras和NLTK实现的深度神经网络。深度学习模型取得了最好的成绩。

    University students often spend a considerable amount of time seeking answers to common questions from administrators or teachers. This can become tedious for both parties, leading to a need for a solution. In response, this paper proposes a chatbot model that utilizes natural language processing and deep learning techniques to answer frequently asked questions (FAQs) in the Amharic language. Chatbots are computer programs that simulate human conversation through the use of artificial intelligence (AI), acting as a virtual assistant to handle questions and other tasks. The proposed chatbot program employs tokenization, normalization, stop word removal, and stemming to analyze and categorize Amharic input sentences. Three machine learning model algorithms were used to classify tokens and retrieve appropriate responses: Support Vector Machine (SVM), Multinomial Na\"ive Bayes, and deep neural networks implemented through TensorFlow, Keras, and NLTK. The deep learning model achieved the be
    
[^7]: 一种基于任务反馈的动态剪裁方法用于近端策略优化

    A dynamical clipping approach with task feedback for Proximal Policy Optimization

    [https://arxiv.org/abs/2312.07624](https://arxiv.org/abs/2312.07624)

    提出了一种基于任务反馈的动态剪裁方法，通过增加最大累积回报来优化近端策略优化的性能。

    

    近端策略优化（PPO）已被广泛应用于各个领域，包括大型语言模型（LLM）优化和机器人学习等。然而，PPO受到固定剪裁边界的限制。具体而言，目前没有理论证明最佳剪裁边界在整个训练过程中始终保持一致。通过用一个独特的剪裁边界截断新旧策略的比率，可以确保稳定的训练并实现最佳的训练性能。此外，先前的研究表明，固定的剪裁边界限制了agent的探索。因此，研究一种动态剪裁边界以增强PPO的性能是非常有益的。与以往的剪裁方法不同，我们考虑将在强化学习（RL）任务中增加最大累积回报视作RL任务的偏好，并提出了一个双层近端策略优化范式。

    arXiv:2312.07624v2 Announce Type: replace-cross  Abstract: Proximal Policy Optimization (PPO) has been broadly applied to various domains, including Large Language Model (LLM) optimization and Robotics learning, etc. However, PPO is limited by a fixed setting for the clipping bound. Specifically, there is no theoretical proof that the optimal clipping bound remains consistent throughout the entire training process. Truncating the ratio of the new and old policies with a unique clipping bound ensures stable training and can achieve the best training performance. Additionally, previous research suggests that a fixed clipping bound limits the agent's exploration. Therefore, researching a dynamical clipping bound to enhance PPO's performance can be highly beneficial. Different from previous clipping approaches, we consider increasing the maximum cumulative Return in reinforcement learning (RL) tasks as the preference of the RL task, and propose a bi-level proximal policy optimization parad
    
[^8]: 使用黎曼几何特征学习飞机机翼上的压力系数

    Incorporating Riemannian Geometric Features for Learning Coefficient of Pressure Distributions on Airplane Wings. (arXiv:2401.09452v1 [cs.LG])

    [http://arxiv.org/abs/2401.09452](http://arxiv.org/abs/2401.09452)

    该论文提出了一种将黎曼几何特征应用于学习翼面压力系数分布的方法，以提高气动系数的预测准确性。

    

    飞机的气动系数受其几何形状的显著影响，尤其是当攻角较大时。在空气动力学领域，传统的基于多项式的参数化方法使用尽可能少的参数来描述翼型的几何形状。然而，由于翼的三维几何形状比二维翼型复杂，基于多项式的参数化方法难以准确表示翼在三维空间中的整体形状。现有的基于深度学习的方法可以提取用于描述二维翼型或翼截面形状的大量潜在神经表示。最近的研究表明，直接将几何特征作为神经网络的输入可以提高预测的气动系数的准确性。受几何理论的启发，我们提出了将黎曼几何特征纳入学习翼面压力系数分布的方法。我们的方法计算几何特征（黎曼）。

    The aerodynamic coefficients of aircrafts are significantly impacted by its geometry, especially when the angle of attack (AoA) is large. In the field of aerodynamics, traditional polynomial-based parameterization uses as few parameters as possible to describe the geometry of an airfoil. However, because the 3D geometry of a wing is more complicated than the 2D airfoil, polynomial-based parameterizations have difficulty in accurately representing the entire shape of a wing in 3D space. Existing deep learning-based methods can extract massive latent neural representations for the shape of 2D airfoils or 2D slices of wings. Recent studies highlight that directly taking geometric features as inputs to the neural networks can improve the accuracy of predicted aerodynamic coefficients. Motivated by geometry theory, we propose to incorporate Riemannian geometric features for learning Coefficient of Pressure (CP) distributions on wing surfaces. Our method calculates geometric features (Rieman
    
[^9]: 教导大型语言模型忘记隐私

    Teach Large Language Models to Forget Privacy. (arXiv:2401.00870v1 [cs.CR])

    [http://arxiv.org/abs/2401.00870](http://arxiv.org/abs/2401.00870)

    这项研究提出了Prompt2Forget（P2F）框架，通过教导大型语言模型（LLM）忘记隐私信息，解决了LLM本地隐私挑战。P2F方法将问题分解为片段并生成虚构答案，模糊化模型对原始输入的记忆。实验证明，P2F具有很强的模糊化能力，并且可以在各种应用场景下自适应使用，无需手动设置。

    

    大型语言模型（LLM）已被证明具有强大的能力，但隐私泄露的风险仍然是一个重要问题。传统的保护隐私方法，如差分隐私和同态加密，在只有黑盒API的环境下是不足够的，要求模型透明性或大量计算资源。我们提出了Prompt2Forget（P2F），这是第一个设计用于解决LLM本地隐私挑战的框架，通过教导LLM忘记来实现。该方法涉及将完整问题分解为较小的片段，生成虚构的答案，并使模型对原始输入的记忆模糊化。我们根据不同领域的包含隐私敏感信息的问题创建了基准数据集。P2F实现了零-shot泛化，可以在多种应用场景下自适应，无需手动调整。实验结果表明，P2F具有很强的模糊化LLM记忆的能力，而不会损失任何实用性。

    Large Language Models (LLMs) have proven powerful, but the risk of privacy leakage remains a significant concern. Traditional privacy-preserving methods, such as Differential Privacy and Homomorphic Encryption, are inadequate for black-box API-only settings, demanding either model transparency or heavy computational resources. We propose Prompt2Forget (P2F), the first framework designed to tackle the LLM local privacy challenge by teaching LLM to forget. The method involves decomposing full questions into smaller segments, generating fabricated answers, and obfuscating the model's memory of the original input. A benchmark dataset was crafted with questions containing privacy-sensitive information from diverse fields. P2F achieves zero-shot generalization, allowing adaptability across a wide range of use cases without manual adjustments. Experimental results indicate P2F's robust capability to obfuscate LLM's memory, attaining a forgetfulness score of around 90\% without any utility los
    
[^10]: D3: 数据多样性设计为系统一般化在视觉问答中。

    D3: Data Diversity Design for Systematic Generalization in Visual Question Answering. (arXiv:2309.08798v1 [cs.AI])

    [http://arxiv.org/abs/2309.08798](http://arxiv.org/abs/2309.08798)

    本论文研究了视觉问答中系统一般化的关键因素，发现简单任务的多样性在实现系统一般化中起到了重要的作用，这意味着不必收集大量和多样的复杂任务。

    

    系统一般化是智能的关键方面，它指的是通过结合已知的子任务和概念来推广到新任务的能力。已经显示影响系统一般化的一个关键因素是训练数据的多样性。然而，多样性可以以多种方式定义，因为数据具有许多变化因素。对于不同方面的数据多样性如何影响系统一般化的更细致的理解尚缺乏。我们在视觉问答（VQA）问题中提供了新的证据，揭示了简单任务的多样性（即由几个子任务和概念组成的任务）在实现系统一般化中的关键作用。这意味着收集大量和多样化的复杂任务可能并非必要，这可能成本高昂。我们证明了这个结果与训练和测试数据之间的相似性无关，并适用于众所周知的神经网络家族。

    Systematic generalization is a crucial aspect of intelligence, which refers to the ability to generalize to novel tasks by combining known subtasks and concepts. One critical factor that has been shown to influence systematic generalization is the diversity of training data. However, diversity can be defined in various ways, as data have many factors of variation. A more granular understanding of how different aspects of data diversity affect systematic generalization is lacking. We present new evidence in the problem of Visual Question Answering (VQA) that reveals that the diversity of simple tasks (i.e. tasks formed by a few subtasks and concepts) plays a key role in achieving systematic generalization. This implies that it may not be essential to gather a large and varied number of complex tasks, which could be costly to obtain. We demonstrate that this result is independent of the similarity between the training and testing data and applies to well-known families of neural network 
    
[^11]: 打破度量投票扭曲的障碍

    Breaking the Metric Voting Distortion Barrier. (arXiv:2306.17838v1 [cs.GT])

    [http://arxiv.org/abs/2306.17838](http://arxiv.org/abs/2306.17838)

    这篇论文研究了社会选择中的度量扭曲问题，提出了一个新的投票规则，该规则可以选择距离选民平均距离较小的候选人。之前的研究发现确定性投票规则的度量扭曲限制为3，但在无限制的情况下，我们对该问题仍然了解有限。

    

    我们考虑社会选择中度量扭曲的经典问题。假设我们有一个选举，有n名选民和m名候选人，他们位于一个共享的度量空间中。我们希望设计一个投票规则，选择一个平均距离选民较小的候选人。然而，我们不能直接获得度量空间中的距离信息，每个选民只能给出候选人的排序列表。我们能否设计一条规则，无论选举实例和底层度量空间如何，都能选择出一个与真正最优解的代价只相差一个小因子（称为扭曲度）的候选人？许多研究的成果将确定性投票规则的度量扭曲限制为3，这是确定性规则和许多其他投票规则类别的最佳结果。然而，在没有任何限制的情况下，我们对该问题仍然了解有限：尽管最佳下界已经降低到2.112，但现有规则的扭曲度仍然相对较高。

    We consider the following well studied problem of metric distortion in social choice. Suppose we have an election with $n$ voters and $m$ candidates who lie in a shared metric space. We would like to design a voting rule that chooses a candidate whose average distance to the voters is small. However, instead of having direct access to the distances in the metric space, each voter gives us a ranked list of the candidates in order of distance. Can we design a rule that regardless of the election instance and underlying metric space, chooses a candidate whose cost differs from the true optimum by only a small factor (known as the distortion)?  A long line of work culminated in finding deterministic voting rules with metric distortion $3$, which is the best possible for deterministic rules and many other classes of voting rules. However, without any restrictions, there is still a significant gap in our understanding: Even though the best lower bound is substantially lower at $2.112$, the b
    
[^12]: 虚拟人类生成模型：基于掩码建模的方法来学习人类特征

    Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics. (arXiv:2306.10656v1 [cs.LG])

    [http://arxiv.org/abs/2306.10656](http://arxiv.org/abs/2306.10656)

    本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。

    

    识别医疗属性、生活方式和人格之间的关系对于理解和改善身体和精神状况至关重要。本文提出了一种名为虚拟人类生成模型（VHGM）的机器学习模型，用于估计有关医疗保健、生活方式和个性的属性。VHGM是一个深度生成模型，使用掩码建模训练，在已知属性的条件下学习属性的联合分布。利用异构表格数据集，VHGM高效地学习了超过1,800个属性。我们数值评估了VHGM及其训练技术的性能。作为VHGM的概念验证，我们提出了几个应用程序，演示了用户情境，例如医疗属性的虚拟测量和生活方式的假设验证。

    Identifying the relationship between healthcare attributes, lifestyles, and personality is vital for understanding and improving physical and mental conditions. Machine learning approaches are promising for modeling their relationships and offering actionable suggestions. In this paper, we propose Virtual Human Generative Model (VHGM), a machine learning model for estimating attributes about healthcare, lifestyles, and personalities. VHGM is a deep generative model trained with masked modeling to learn the joint distribution of attributes conditioned on known ones. Using heterogeneous tabular datasets, VHGM learns more than 1,800 attributes efficiently. We numerically evaluate the performance of VHGM and its training techniques. As a proof-of-concept of VHGM, we present several applications demonstrating user scenarios, such as virtual measurements of healthcare attributes and hypothesis verifications of lifestyles.
    
[^13]: CatE：使用范畴论语义嵌入$\mathcal{ALC}$本体论

    CatE: Embedding $\mathcal{ALC}$ ontologies using category-theoretical semantics. (arXiv:2305.07163v1 [cs.LO])

    [http://arxiv.org/abs/2305.07163](http://arxiv.org/abs/2305.07163)

    本文介绍了一种使用范畴论语义的方法CatE，该方法可以高效地将本体公理投影到图形中，并生成本体论的嵌入，从而在知识图谱的链接预测任务上取得更好的表现。

    

    与语义Web本体一起进行机器学习遵循几个策略之一，其中包括将本体投影到图形结构中，并将图形嵌入或基于图形的机器学习方法应用于生成的图形。已经开发出了几种将本体公理投影到图形中的方法。然而，这些方法在它们可以投影的公理类型（全面性）、它们是否可逆（单射性）以及它们如何利用语义信息方面存在局限性。这些限制限制了它们可以应用的任务类型。逻辑语言的范畴论语义以范畴而不是集合的形式形式化解释，并且范畴具有类似于图形的结构。我们开发了CatE，它使用$\mathcal{ALC}$描述逻辑的范畴论公式来生成本体公理的图形表示。CatE投影具有全面性和单射性，因此克服了其他基于图形本体论的方法的限制。CatE还通过将$\mathcal{ALC}$的语法编码为一种范畴来利用语义信息，这使我们能够为本体论生成嵌入。我们在知识图谱的链接预测任务上评估了CatE，并表明它优于现有方法。

    Machine learning with Semantic Web ontologies follows several strategies, one of which involves projecting ontologies into graph structures and applying graph embeddings or graph-based machine learning methods to the resulting graphs. Several methods have been developed that project ontology axioms into graphs. However, these methods are limited in the type of axioms they can project (totality), whether they are invertible (injectivity), and how they exploit semantic information. These limitations restrict the kind of tasks to which they can be applied. Category-theoretical semantics of logic languages formalizes interpretations using categories instead of sets, and categories have a graph-like structure. We developed CatE, which uses the category-theoretical formulation of the semantics of the Description Logic $\mathcal{ALC}$ to generate a graph representation for ontology axioms. The CatE projection is total and injective, and therefore overcomes limitations of other graph-based ont
    

