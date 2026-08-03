# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Extended Neighbourhood Rule $k$ Nearest Neighbours Ensemble](https://arxiv.org/abs/2211.11278) | 提出了一种基于最优扩展邻域规则的集成方法，通过新规则确定邻居和模型选择策略来解决传统$k$最近邻方法的局限性和提升集成性能。 |
| [^2] | [Nonparametric Partial Disentanglement via Mechanism Sparsity: Sparse Actions, Interventions and Sparse Temporal Dependencies.](http://arxiv.org/abs/2401.04890) | 本研究引入了一种称为机制稀疏性正则化的解缠原则，通过同时学习潜在因素和解释它们的稀疏因果图模型来实现解缠。这项工作通过非参数化可辨识性理论证明了这一原则，并提供了一种图形准则来保证完全解缠。 |
| [^3] | [Information Processing by Neuron Populations in the Central Nervous System: Mathematical Structure of Data and Operations.](http://arxiv.org/abs/2309.02332) | 神经群体在中枢神经系统中使用数学结构精确地表示和操作信息，实现了特化、泛化、新奇检测等多种功能。 |
| [^4] | [Tipping Point Forecasting in Non-Stationary Dynamics on Function Spaces.](http://arxiv.org/abs/2308.08794) | 本文提出了一种利用循环神经算子学习非平稳动力系统演化的方法，并且通过基于不确定性的方法检测未来的翻车点。同时，我们还提出了一种符合预测框架，通过监测与物理约束的偏离来预测翻车点，从而使得预测结果具有严格的不确定性度量。 |
| [^5] | [Beyond Black-Box Advice: Learning-Augmented Algorithms for MDPs with Q-Value Predictions.](http://arxiv.org/abs/2307.10524) | 该论文研究了在具有不可信的机器学习建议的单轨迹时间变化的MDP中一致性和鲁棒性之间的权衡，并证明了利用Q值建议可以获得接近最优的性能保证，并改进了仅使用黑盒建议的情况。 |

# 详细

[^1]: 最优扩展邻域规则$k$最近邻集成

    Optimal Extended Neighbourhood Rule $k$ Nearest Neighbours Ensemble

    [https://arxiv.org/abs/2211.11278](https://arxiv.org/abs/2211.11278)

    提出了一种基于最优扩展邻域规则的集成方法，通过新规则确定邻居和模型选择策略来解决传统$k$最近邻方法的局限性和提升集成性能。

    

    传统的$k$最近邻($k$NN)方法使用一个球形区域内的距离公式来确定训练观测中与测试样本点最接近的$k$个观测。然而，当测试点位于该区域之外时，这种方法可能不起作用。此外，聚合许多基础$k$NN学习器可能会导致由于高分类误差而表现不佳的集成性能。为解决这些问题，本文提出了一种新的基于最优扩展邻域规则的集成方法。该规则从距离未见观测最近的样本点开始，经过$k$步确定邻居，并选择直到达到所需数量的观测数据点。每个基础模型都是在一个随机特征子集上的自举样本上构建的，并且在构建足够数量的模型后基于袋外表现选择最优模型。提出的集成方法与st进行了比较

    arXiv:2211.11278v2 Announce Type: replace-cross  Abstract: The traditional k nearest neighbor (kNN) approach uses a distance formula within a spherical region to determine the k closest training observations to a test sample point. However, this approach may not work well when test point is located outside this region. Moreover, aggregating many base kNN learners can result in poor ensemble performance due to high classification errors. To address these issues, a new optimal extended neighborhood rule based ensemble method is proposed in this paper. This rule determines neighbors in k steps starting from the closest sample point to the unseen observation and selecting subsequent nearest data points until the required number of observations is reached. Each base model is constructed on a bootstrap sample with a random subset of features, and optimal models are selected based on out-of-bag performance after building a sufficient number of models. The proposed ensemble is compared with st
    
[^2]: 通过机制稀疏性进行非参数化部分解缠: 稀疏动作, 干预和稀疏时间依赖性

    Nonparametric Partial Disentanglement via Mechanism Sparsity: Sparse Actions, Interventions and Sparse Temporal Dependencies. (arXiv:2401.04890v1 [stat.ML])

    [http://arxiv.org/abs/2401.04890](http://arxiv.org/abs/2401.04890)

    本研究引入了一种称为机制稀疏性正则化的解缠原则，通过同时学习潜在因素和解释它们的稀疏因果图模型来实现解缠。这项工作通过非参数化可辨识性理论证明了这一原则，并提供了一种图形准则来保证完全解缠。

    

    这项工作引入一种新的解缠原则，即机制稀疏规则，该规则适用于感兴趣的潜在因素在观察辅助变量和/或过去潜在因素上稀疏依赖的情况。我们提出了一种表示学习方法，通过同时学习潜在因素和解释它们的稀疏因果图模型来引导解缠。我们开发了一个非参数化可辨识性理论来形式化这一原则，并证明通过将学习到的因果图稀疏化，可以恢复潜在因素。更确切地说，我们展示了一种新的等价关系"一致性"来描述能够保持一些潜在因素纠缠的部分解缠过程。为了描述纠缠的结构，我们引入了纠缠图和图保持函数的概念。我们还提供了一个图形准则，用于保证完全解缠。

    This work introduces a novel principle for disentanglement we call mechanism sparsity regularization, which applies when the latent factors of interest depend sparsely on observed auxiliary variables and/or past latent factors. We propose a representation learning method that induces disentanglement by simultaneously learning the latent factors and the sparse causal graphical model that explains them. We develop a nonparametric identifiability theory that formalizes this principle and shows that the latent factors can be recovered by regularizing the learned causal graph to be sparse. More precisely, we show identifiablity up to a novel equivalence relation we call "consistency", which allows some latent factors to remain entangled (hence the term partial disentanglement). To describe the structure of this entanglement, we introduce the notions of entanglement graphs and graph preserving functions. We further provide a graphical criterion which guarantees complete disentanglement, that
    
[^3]: 神经群体在中枢神经系统中的信息处理：数据和操作的数学结构

    Information Processing by Neuron Populations in the Central Nervous System: Mathematical Structure of Data and Operations. (arXiv:2309.02332v1 [q-bio.NC] CROSS LISTED)

    [http://arxiv.org/abs/2309.02332](http://arxiv.org/abs/2309.02332)

    神经群体在中枢神经系统中使用数学结构精确地表示和操作信息，实现了特化、泛化、新奇检测等多种功能。

    

    在哺乳动物中枢神经系统的复杂结构中，神经元形成群体。轴索束通过脉冲列作为媒介在这些群集之间进行通信。然而，这些神经群体的精确编码和操作还有待发现。在我们的分析中，出发点是一个具有可塑性的通用神经元的先进的机械模型。从这个简单的框架中出现了一个深刻的数学构造：通过有限凸锥的代数可以准确地描述信息的表示和操作。此外，这些神经群体不仅仅是被动传输者。它们在这个代数结构中扮演着运算符的角色，反映了低级编程语言的功能。当这些群体互连时，它们具有简洁而强大的代数表达式。这些网络使它们能够实现许多操作，如特化、泛化、新奇检测、维度降低等。

    In the intricate architecture of the mammalian central nervous system, neurons form populations. Axonal bundles communicate between these clusters using spike trains as their medium. However, these neuron populations' precise encoding and operations have yet to be discovered. In our analysis, the starting point is a state-of-the-art mechanistic model of a generic neuron endowed with plasticity. From this simple framework emerges a profound mathematical construct: The representation and manipulation of information can be precisely characterized by an algebra of finite convex cones. Furthermore, these neuron populations are not merely passive transmitters. They act as operators within this algebraic structure, mirroring the functionality of a low-level programming language. When these populations interconnect, they embody succinct yet potent algebraic expressions. These networks allow them to implement many operations, such as specialization, generalization, novelty detection, dimensiona
    
[^4]: 功能空间中非平稳动力学中的翻车点预测

    Tipping Point Forecasting in Non-Stationary Dynamics on Function Spaces. (arXiv:2308.08794v1 [cs.LG])

    [http://arxiv.org/abs/2308.08794](http://arxiv.org/abs/2308.08794)

    本文提出了一种利用循环神经算子学习非平稳动力系统演化的方法，并且通过基于不确定性的方法检测未来的翻车点。同时，我们还提出了一种符合预测框架，通过监测与物理约束的偏离来预测翻车点，从而使得预测结果具有严格的不确定性度量。

    

    翻车点是非平稳和混沌动力系统演化中的突变、剧烈且常常不可逆的变化。例如，预计温室气体浓度的增加会导致低云覆盖的急剧减少，被称为气候学的翻车点。在本文中，我们利用一种新颖的循环神经算子（RNO）学习这种非平稳动力系统的演化，RNO可以学习函数空间之间的映射关系。在仅训练RNO在翻车点之前的动力学数据之后，我们采用基于不确定性的方法来检测未来的翻车点。具体而言，我们提出了一个符合预测框架，通过监测与物理约束（如守恒量和偏微分方程）偏离来预测翻车点，从而使得对这些突变的预测伴随着一种严格的不确定性度量。我们将我们提出的方法应用于非平稳常微分方程和偏微分方程的案例。

    Tipping points are abrupt, drastic, and often irreversible changes in the evolution of non-stationary and chaotic dynamical systems. For instance, increased greenhouse gas concentrations are predicted to lead to drastic decreases in low cloud cover, referred to as a climatological tipping point. In this paper, we learn the evolution of such non-stationary dynamical systems using a novel recurrent neural operator (RNO), which learns mappings between function spaces. After training RNO on only the pre-tipping dynamics, we employ it to detect future tipping points using an uncertainty-based approach. In particular, we propose a conformal prediction framework to forecast tipping points by monitoring deviations from physics constraints (such as conserved quantities and partial differential equations), enabling forecasting of these abrupt changes along with a rigorous measure of uncertainty. We illustrate our proposed methodology on non-stationary ordinary and partial differential equations,
    
[^5]: 超越黑盒建议: 基于学习的增强算法用于具有Q值预测的MDPs

    Beyond Black-Box Advice: Learning-Augmented Algorithms for MDPs with Q-Value Predictions. (arXiv:2307.10524v1 [cs.LG])

    [http://arxiv.org/abs/2307.10524](http://arxiv.org/abs/2307.10524)

    该论文研究了在具有不可信的机器学习建议的单轨迹时间变化的MDP中一致性和鲁棒性之间的权衡，并证明了利用Q值建议可以获得接近最优的性能保证，并改进了仅使用黑盒建议的情况。

    

    我们研究了在单轨迹时间变化的马尔科夫决策过程(MDP)中一致性和鲁棒性之间的权衡，该过程具有不可信的机器学习建议。我们的工作不同于常规方法，不再将建议视为来自黑盒来源，而是考虑到有关如何生成建议的其他信息。我们证明了在包括连续和离散状态/动作空间的一般MDP模型下给出的Q值建议的一种新型一致性和鲁棒性权衡。我们的结果表明，利用Q值建议可以动态追求机器学习建议和稳健基线中较优的那个，从而产生接近最优的性能保证，并且改进了仅使用黑盒建议所能获得的结果。

    We study the tradeoff between consistency and robustness in the context of a single-trajectory time-varying Markov Decision Process (MDP) with untrusted machine-learned advice. Our work departs from the typical approach of treating advice as coming from black-box sources by instead considering a setting where additional information about how the advice is generated is available. We prove a first-of-its-kind consistency and robustness tradeoff given Q-value advice under a general MDP model that includes both continuous and discrete state/action spaces. Our results highlight that utilizing Q-value advice enables dynamic pursuit of the better of machine-learned advice and a robust baseline, thus result in near-optimal performance guarantees, which provably improves what can be obtained solely with black-box advice.
    

