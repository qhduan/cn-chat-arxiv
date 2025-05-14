# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonlinearity Enhanced Adaptive Activation Function](https://arxiv.org/abs/2403.19896) | 引入了一种带有甚至立方非线性的简单实现激活函数，通过引入可优化参数使得激活函数具有更大的自由度，可以提高神经网络的准确性，同时不需要太多额外的计算资源。 |
| [^2] | [Considerations in the use of ML interaction potentials for free energy calculations](https://arxiv.org/abs/2403.13952) | 该研究探讨了机器学习势在自由能计算中的应用，重点关注使用等变性图神经网络MLPs来准确预测自由能和过渡态，考虑到分子构型的能量和多样性。 |
| [^3] | [Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective](https://arxiv.org/abs/2402.10686) | 通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险 |
| [^4] | [GenSTL: General Sparse Trajectory Learning via Auto-regressive Generation of Feature Domains](https://arxiv.org/abs/2402.07232) | GenSTL是一个通用的稀疏轨迹学习框架，通过自回归生成特征域来实现稀疏轨迹与密集轨迹之间的连接，从而消除了对大规模密集轨迹数据的依赖。 |
| [^5] | [Learning Optimal Classification Trees Robust to Distribution Shifts.](http://arxiv.org/abs/2310.17772) | 本研究提出了一种学习对分布变化具有鲁棒性的最优分类树的方法，通过混合整数鲁棒优化技术将该问题转化为单阶段混合整数鲁棒优化问题，并设计了基于约束生成的解决过程。 |
| [^6] | [A stochastic optimization approach to train non-linear neural networks with regularization of higher-order total variation.](http://arxiv.org/abs/2308.02293) | 通过引入高阶总变差正则化的随机优化算法，可以高效地训练非线性神经网络，避免过拟合问题。 |

# 详细

[^1]: 非线性增强自适应激活函数

    Nonlinearity Enhanced Adaptive Activation Function

    [https://arxiv.org/abs/2403.19896](https://arxiv.org/abs/2403.19896)

    引入了一种带有甚至立方非线性的简单实现激活函数，通过引入可优化参数使得激活函数具有更大的自由度，可以提高神经网络的准确性，同时不需要太多额外的计算资源。

    

    引入一种简单实现的激活函数，具有甚至立方非线性，可以提高神经网络的准确性，而不需要太多额外的计算资源。通过一种明显的收敛与准确性之间的权衡来实现。该激活函数通过引入可优化参数来泛化标准RELU函数，从而增加了额外的自由度，使得非线性程度可以被调整。通过与标准技术进行比较，将在MNIST数字数据集的背景下量化相关准确性的提升。

    arXiv:2403.19896v1 Announce Type: new  Abstract: A simply implemented activation function with even cubic nonlinearity is introduced that increases the accuracy of neural networks without substantial additional computational resources. This is partially enabled through an apparent tradeoff between convergence and accuracy. The activation function generalizes the standard RELU function by introducing additional degrees of freedom through optimizable parameters that enable the degree of nonlinearity to be adjusted. The associated accuracy enhancement is quantified in the context of the MNIST digit data set through a comparison with standard techniques.
    
[^2]: 机器学习相互作用势在自由能计算中的应用考虑

    Considerations in the use of ML interaction potentials for free energy calculations

    [https://arxiv.org/abs/2403.13952](https://arxiv.org/abs/2403.13952)

    该研究探讨了机器学习势在自由能计算中的应用，重点关注使用等变性图神经网络MLPs来准确预测自由能和过渡态，考虑到分子构型的能量和多样性。

    

    机器学习势（MLPs）具有准确建模分子能量和自由能景观的潜力，该准确性可媲美量子力学，并具有类似经典模拟的效率。本研究侧重于使用等变性图神经网络MLPs，因为它们在建模平衡分子轨迹中已被证明有效。一个关键问题是MLPs能否准确预测自由能和过渡态，要考虑分子构型的能量和多样性。我们检查了训练数据中集体变量（CVs）的分布如何影响MLP在确定系统自由能面（FES）时的准确性，使用Metadynamics模拟对丁烷和丙氨酸二肽（ADP）进行实验。该研究涉及对四十三个MLP进行训练，其中一半基于经典分子动力学数据，其余的基于从头计算的能量。这些MLPs进行了训练

    arXiv:2403.13952v1 Announce Type: cross  Abstract: Machine learning potentials (MLPs) offer the potential to accurately model the energy and free energy landscapes of molecules with the precision of quantum mechanics and an efficiency similar to classical simulations. This research focuses on using equivariant graph neural networks MLPs due to their proven effectiveness in modeling equilibrium molecular trajectories. A key issue addressed is the capability of MLPs to accurately predict free energies and transition states by considering both the energy and the diversity of molecular configurations. We examined how the distribution of collective variables (CVs) in the training data affects MLP accuracy in determining the free energy surface (FES) of systems, using Metadynamics simulations for butane and alanine dipeptide (ADP). The study involved training forty-three MLPs, half based on classical molecular dynamics data and the rest on ab initio computed energies. The MLPs were trained u
    
[^3]: 不确定性、校准和成员推理攻击：信息论视角

    Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective

    [https://arxiv.org/abs/2402.10686](https://arxiv.org/abs/2402.10686)

    通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险

    

    在成员推理攻击（MIA）中，攻击者利用典型机器学习模型表现出的过度自信来确定特定数据点是否被用于训练目标模型。在本文中，我们在一个信息理论框架内分析了最先进的似然比攻击（LiRA）的性能，这个框架可以允许研究真实数据生成过程中的不确定性的影响，由有限训练数据集引起的认知不确定性以及目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型接收到的信息逐渐减少：置信向量（CV）披露，其中输出概率向量被发布；真实标签置信度（TLC）披露，其中只有模型分配给真实标签的概率是可用的；以及决策集（DS）披露。

    arXiv:2402.10686v1 Announce Type: cross  Abstract: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the state-of-the-art likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in 
    
[^4]: GenSTL: 通过特征域的自回归生成实现通用稀疏轨迹学习

    GenSTL: General Sparse Trajectory Learning via Auto-regressive Generation of Feature Domains

    [https://arxiv.org/abs/2402.07232](https://arxiv.org/abs/2402.07232)

    GenSTL是一个通用的稀疏轨迹学习框架，通过自回归生成特征域来实现稀疏轨迹与密集轨迹之间的连接，从而消除了对大规模密集轨迹数据的依赖。

    

    轨迹是时间戳位置样本的序列。在稀疏轨迹中，位置样本的采样是不频繁的；尽管这种轨迹在现实世界中很常见，但要使用它们来实现高质量的与交通相关的应用程序是具有挑战性的。当前的方法要么假设轨迹是密集采样的并且经过准确的地图匹配，要么依赖于两阶段方案，从而产生次优的应用程序。为了扩展稀疏轨迹的效用，我们提出了一种新颖的稀疏轨迹学习框架GenSTL。该框架经过预训练以使用特征域的自回归生成形成稀疏轨迹与密集轨迹之间的连接。GenSTL可以直接应用于下游任务，或者可以先进行微调。通过这种方式，GenSTL消除了对大规模密集和地图匹配轨迹数据的依赖。其中包括精心设计的特征域编码层和分层的...

    Trajectories are sequences of timestamped location samples. In sparse trajectories, the locations are sampled infrequently; and while such trajectories are prevalent in real-world settings, they are challenging to use to enable high-quality transportation-related applications. Current methodologies either assume densely sampled and accurately map-matched trajectories, or they rely on two-stage schemes, yielding sub-optimal applications.   To extend the utility of sparse trajectories, we propose a novel sparse trajectory learning framework, GenSTL. The framework is pre-trained to form connections between sparse trajectories and dense counterparts using auto-regressive generation of feature domains. GenSTL can subsequently be applied directly in downstream tasks, or it can be fine-tuned first. This way, GenSTL eliminates the reliance on the availability of large-scale dense and map-matched trajectory data. The inclusion of a well-crafted feature domain encoding layer and a hierarchical m
    
[^5]: 学习对分布变化具有鲁棒性的最优分类树

    Learning Optimal Classification Trees Robust to Distribution Shifts. (arXiv:2310.17772v1 [cs.LG])

    [http://arxiv.org/abs/2310.17772](http://arxiv.org/abs/2310.17772)

    本研究提出了一种学习对分布变化具有鲁棒性的最优分类树的方法，通过混合整数鲁棒优化技术将该问题转化为单阶段混合整数鲁棒优化问题，并设计了基于约束生成的解决过程。

    

    我们考虑学习对训练和测试/部署数据之间的分布变化具有鲁棒性的分类树的问题。这个问题经常在高风险环境中出现，例如公共卫生和社会工作，其中数据通常是通过自我报告的调查收集的，这些调查对问题的表述方式、调查进行的时间和地点、以及受访者与调查员分享信息的舒适程度非常敏感。我们提出了一种基于混合整数鲁棒优化技术的学习最优鲁棒分类树的方法。特别地，我们证明学习最优鲁棒树的问题可以等价地表达为一个具有高度非线性和不连续目标的单阶段混合整数鲁棒优化问题。我们将这个问题等价地重新表述为一个两阶段线性鲁棒优化问题，为此我们设计了一个基于约束生成的定制解决过程。

    We consider the problem of learning classification trees that are robust to distribution shifts between training and testing/deployment data. This problem arises frequently in high stakes settings such as public health and social work where data is often collected using self-reported surveys which are highly sensitive to e.g., the framing of the questions, the time when and place where the survey is conducted, and the level of comfort the interviewee has in sharing information with the interviewer. We propose a method for learning optimal robust classification trees based on mixed-integer robust optimization technology. In particular, we demonstrate that the problem of learning an optimal robust tree can be cast as a single-stage mixed-integer robust optimization problem with a highly nonlinear and discontinuous objective. We reformulate this problem equivalently as a two-stage linear robust optimization problem for which we devise a tailored solution procedure based on constraint gene
    
[^6]: 用正则化高阶总变差的随机优化方法训练非线性神经网络

    A stochastic optimization approach to train non-linear neural networks with regularization of higher-order total variation. (arXiv:2308.02293v1 [stat.ME])

    [http://arxiv.org/abs/2308.02293](http://arxiv.org/abs/2308.02293)

    通过引入高阶总变差正则化的随机优化算法，可以高效地训练非线性神经网络，避免过拟合问题。

    

    尽管包括深度神经网络在内的高度表达的参数模型可以更好地建模复杂概念，但训练这种高度非线性模型已知会导致严重的过拟合风险。针对这个问题，本研究考虑了一种k阶总变差（k-TV）正则化，它被定义为要训练的参数模型的k阶导数的平方积分，通过惩罚k-TV来产生一个更平滑的函数，从而避免过拟合。尽管将k-TV项应用于一般的参数模型由于积分而导致计算复杂，本研究提供了一种随机优化算法，可以高效地训练带有k-TV正则化的一般模型，而无需进行显式的数值积分。这种方法可以应用于结构任意的深度神经网络的训练，因为它只需要进行简单的随机梯度优化即可实现。

    While highly expressive parametric models including deep neural networks have an advantage to model complicated concepts, training such highly non-linear models is known to yield a high risk of notorious overfitting. To address this issue, this study considers a $k$th order total variation ($k$-TV) regularization, which is defined as the squared integral of the $k$th order derivative of the parametric models to be trained; penalizing the $k$-TV is expected to yield a smoother function, which is expected to avoid overfitting. While the $k$-TV terms applied to general parametric models are computationally intractable due to the integration, this study provides a stochastic optimization algorithm, that can efficiently train general models with the $k$-TV regularization without conducting explicit numerical integration. The proposed approach can be applied to the training of even deep neural networks whose structure is arbitrary, as it can be implemented by only a simple stochastic gradien
    

