# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SVGCraft: Beyond Single Object Text-to-SVG Synthesis with Comprehensive Canvas Layout](https://arxiv.org/abs/2404.00412) | SVGCraft引入了一种端到端框架，可以从文本描述中生成描绘整个场景的矢量图，其中包括利用预训练的LLM进行布局生成、产生遮罩潜变量以进行准确对象放置、融合注意力图以及使用扩散U-Net进行合成，同时通过预训练的编码器和LPIPS损失进行优化。 |
| [^2] | [SINDy-RL: Interpretable and Efficient Model-Based Reinforcement Learning](https://arxiv.org/abs/2403.09110) | 将稀疏识别的非线性动力学（SINDy）与深度强化学习（DRL）结合，提出了SINDy-RL框架，用于创建高效解释性还有在低数据制度下创建高效且可解释的数据驱动模型。 |
| [^3] | [Don't Forget What I did?: Assessing Client Contributions in Federated Learning](https://arxiv.org/abs/2403.07151) | 提出了一个历史感知的博弈理论框架FLContrib，用来评估联邦学习中的客户贡献。 |
| [^4] | [How Does Selection Leak Privacy: Revisiting Private Selection and Improved Results for Hyper-parameter Tuning](https://arxiv.org/abs/2402.13087) | 本论文探讨了超参数调整中的隐私性问题，发现当前的隐私分析在一般情况下是紧密的，但在特定的超参数调整问题上则不再成立，并通过隐私审计揭示了当前理论隐私界与实证之间的显著差距。 |
| [^5] | [Optimizing the Design of an Artificial Pancreas to Improve Diabetes Management](https://arxiv.org/abs/2402.07949) | 通过神经进化算法优化人工胰腺治疗策略，减少糖尿病患者的血糖偏差，并且降低注射次数。 |
| [^6] | [Neural Rank Collapse: Weight Decay and Small Within-Class Variability Yield Low-Rank Bias](https://arxiv.org/abs/2402.03991) | 神经网络中的权重衰减和小的类内变化与低秩偏差现象有关 |
| [^7] | [Does provable absence of barren plateaus imply classical simulability? Or, why we need to rethink variational quantum computing](https://arxiv.org/abs/2312.09121) | 常用的具有无荒原证明的模型也可以在进行初始数据采集阶段从量子设备中收集一些经典数据的情况下经典模拟 |
| [^8] | [Simulation Based Bayesian Optimization.](http://arxiv.org/abs/2401.10811) | 本文介绍了基于仿真的贝叶斯优化（SBBO）作为一种新方法，用于通过仅需基于采样的访问来优化获取函数。 |
| [^9] | [Intelligent Condition Monitoring of Industrial Plants: An Overview of Methodologies and Uncertainty Management Strategies.](http://arxiv.org/abs/2401.10266) | 本论文综述了工业厂房智能状态监测和故障检测和诊断方法，重点关注了Tennessee Eastman Process。调研总结了最流行和最先进的深度学习和机器学习算法，并探讨了算法的优劣势。还讨论了不平衡数据和无标记样本等挑战，以及深度学习模型如何应对。比较了不同算法在Tennessee Eastman Process上的准确性和规格。 |
| [^10] | [Towards Identifiable Unsupervised Domain Translation: A Diversified Distribution Matching Approach.](http://arxiv.org/abs/2401.09671) | 本研究旨在解决无监督领域转换中的可识别性问题，引入了一个MPA消除理论，解决了CycleGAN及其变体产生内容不对齐的限制。 |
| [^11] | [On the Foundation of Distributionally Robust Reinforcement Learning.](http://arxiv.org/abs/2311.09018) | 该论文为分布鲁棒强化学习的理论基础做出了贡献，通过一个综合的建模框架，决策者在最坏情况下的分布转变下选择最优策略，并考虑了各种建模属性和对手引起的转变的灵活性。 |
| [^12] | [Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings.](http://arxiv.org/abs/2308.11804) | 该论文研究了多模态嵌入中的对抗幻觉问题。对手可以扰动输入的任意模态，使其嵌入与其他模态的任意输入接近，从而实现任意图像与任意文本、任意文本与任意声音的对齐。该问题与下游任务无关，对生成和分类任务会产生误导。 |
| [^13] | [Robust Sparse Mean Estimation via Incremental Learning.](http://arxiv.org/abs/2305.15276) | 本文提出了一个简单的增量学习方法，仅需要较少的样本即可在近线性时间内估计稀疏均值，克服了现有估计器的限制。 |
| [^14] | [Automatic learning algorithm selection for classification via convolutional neural networks.](http://arxiv.org/abs/2305.09101) | 本文提出了一种直接使用表格数据自动训练卷积神经网络的方法，以学习数据固有的结构，无需识别元特征。在模拟和真实数据集上，该方法均取得了竞争性能。 |
| [^15] | [Robust Mode Connectivity-Oriented Adversarial Defense: Enhancing Neural Network Robustness Against Diversified $\ell_p$ Attacks.](http://arxiv.org/abs/2303.10225) | 本文提出一种新颖的鲁棒模态连接导向的对抗性防御，实现神经网络对多样化$\ell_p$攻击的鲁棒性，其中包括两个基于种群学习的学习阶段。 |
| [^16] | [Hyperbolic Graph Neural Networks: A Review of Methods and Applications.](http://arxiv.org/abs/2202.13852) | 本文综述了当前超边界图神经网络的技术细节，提出了一个通用框架，并总结了每个组件的变体。此外，还介绍了各种与HGNN相关的应用和当前面临的挑战。 |

# 详细

[^1]: SVGCraft:超越单个目标文字到SVG综合画布布局

    SVGCraft: Beyond Single Object Text-to-SVG Synthesis with Comprehensive Canvas Layout

    [https://arxiv.org/abs/2404.00412](https://arxiv.org/abs/2404.00412)

    SVGCraft引入了一种端到端框架，可以从文本描述中生成描绘整个场景的矢量图，其中包括利用预训练的LLM进行布局生成、产生遮罩潜变量以进行准确对象放置、融合注意力图以及使用扩散U-Net进行合成，同时通过预训练的编码器和LPIPS损失进行优化。

    

    生成从文本提示到矢量图的VectorArt是一项具有挑战性的视觉任务，需要对已知和未知实体进行多样化而真实的描述。然而，现有研究主要局限于生成单个对象，而不是由多个元素组成的场景。为此，本文介绍了SVGCraft，这是一个新颖的端到端框架，用于从文本描述中生成描绘整个场景的矢量图。该框架利用预训练的LLM从文本提示生成布局，并引入了一种技术，通过生产特定边界框中的掩膜潜变量实现准确的对象放置。它引入了一个融合机制，用于集成注意力图，并使用扩散U-Net进行连贯的合成，加快绘图过程。生成的SVG使用预训练的编码器和LPIPS损失进行优化，通过透明度调制来最大程度地增加相似性。

    arXiv:2404.00412v1 Announce Type: cross  Abstract: Generating VectorArt from text prompts is a challenging vision task, requiring diverse yet realistic depictions of the seen as well as unseen entities. However, existing research has been mostly limited to the generation of single objects, rather than comprehensive scenes comprising multiple elements. In response, this work introduces SVGCraft, a novel end-to-end framework for the creation of vector graphics depicting entire scenes from textual descriptions. Utilizing a pre-trained LLM for layout generation from text prompts, this framework introduces a technique for producing masked latents in specified bounding boxes for accurate object placement. It introduces a fusion mechanism for integrating attention maps and employs a diffusion U-Net for coherent composition, speeding up the drawing process. The resulting SVG is optimized using a pre-trained encoder and LPIPS loss with opacity modulation to maximize similarity. Additionally, th
    
[^2]: SINDy-RL: 可解释和高效的基于模型的强化学习

    SINDy-RL: Interpretable and Efficient Model-Based Reinforcement Learning

    [https://arxiv.org/abs/2403.09110](https://arxiv.org/abs/2403.09110)

    将稀疏识别的非线性动力学（SINDy）与深度强化学习（DRL）结合，提出了SINDy-RL框架，用于创建高效解释性还有在低数据制度下创建高效且可解释的数据驱动模型。

    

    深度强化学习（DRL）已显示出在与复杂动态环境中相互作用的复杂控制策略中具有显著潜力，例如稳定托卡马克聚变反应堆的磁流体动力学或使物体在流体流动中受到的阻力最小化。然而，这些算法需要大量的训练样本，对许多应用而言成本可能过高。另外，依赖深度神经网络往往会导致难以解释的黑盒策略，可能在某些嵌入式系统中使用时计算成本过高。最近的稀疏字典学习方法的进展，如稀疏非线性动力学的稀疏识别（SINDy），显示出在低数据制度下创建高效且可解释的数据驱动模型的前景。在这项工作中，我们介绍SINDy-RL，这是一个结合SINDy和DRL的统一框架，用于创建高效的可解释的模型。

    arXiv:2403.09110v1 Announce Type: new  Abstract: Deep reinforcement learning (DRL) has shown significant promise for uncovering sophisticated control policies that interact in environments with complicated dynamics, such as stabilizing the magnetohydrodynamics of a tokamak fusion reactor or minimizing the drag force exerted on an object in a fluid flow. However, these algorithms require an abundance of training examples and may become prohibitively expensive for many applications. In addition, the reliance on deep neural networks often results in an uninterpretable, black-box policy that may be too computationally expensive to use with certain embedded systems. Recent advances in sparse dictionary learning, such as the sparse identification of nonlinear dynamics (SINDy), have shown promise for creating efficient and interpretable data-driven models in the low-data regime. In this work we introduce SINDy-RL, a unifying framework for combining SINDy and DRL to create efficient, interpret
    
[^3]: 不要忘记我做的事：评估联邦学习中的客户贡献

    Don't Forget What I did?: Assessing Client Contributions in Federated Learning

    [https://arxiv.org/abs/2403.07151](https://arxiv.org/abs/2403.07151)

    提出了一个历史感知的博弈理论框架FLContrib，用来评估联邦学习中的客户贡献。

    

    联邦学习（FL）是一种协作机器学习（ML）方法，多个客户参与训练ML模型，而不暴露私人数据。公平准确评估客户贡献在FL中是一个重要问题，以促进激励分配并鼓励多样化客户参与统一模型训练。本文提出了一个历史感知的博弈理论框架FLContrib，用于评估在每个FL训练时期中的（潜在非独立同分布）客户参与。

    arXiv:2403.07151v1 Announce Type: cross  Abstract: Federated Learning (FL) is a collaborative machine learning (ML) approach, where multiple clients participate in training an ML model without exposing the private data. Fair and accurate assessment of client contributions is an important problem in FL to facilitate incentive allocation and encouraging diverse clients to participate in a unified model training. Existing methods for assessing client contribution adopts co-operative game-theoretic concepts, such as Shapley values, but under simplified assumptions. In this paper, we propose a history-aware game-theoretic framework, called FLContrib, to assess client contributions when a subset of (potentially non-i.i.d.) clients participate in each epoch of FL training. By exploiting the FL training process and linearity of Shapley value, we develop FLContrib that yields a historical timeline of client contributions as FL training progresses over epochs. Additionally, to assess client cont
    
[^4]: 选择如何泄漏隐私：重新审视私有选择及超参数调整的改进结果

    How Does Selection Leak Privacy: Revisiting Private Selection and Improved Results for Hyper-parameter Tuning

    [https://arxiv.org/abs/2402.13087](https://arxiv.org/abs/2402.13087)

    本论文探讨了超参数调整中的隐私性问题，发现当前的隐私分析在一般情况下是紧密的，但在特定的超参数调整问题上则不再成立，并通过隐私审计揭示了当前理论隐私界与实证之间的显著差距。

    

    我们研究了在超参数调整中保证差分隐私(DP)的问题，这是机器学习中一个关键的过程，涉及从几个运行中选择最佳的过程。与许多私有算法（包括普遍存在的DP-SGD）不同，调整的隐私影响仍然不够了解。最近的研究提出了一个通用的私有解决方案用于调整过程，然而一个根本的问题仍然存在：当前解决方案的隐私界是否紧密？本文对这个问题提出了积极和消极的答案。最初，我们提供的研究证实了当前的隐私分析在一般意义上确实是紧密的。然而，当我们专门研究超参数调整问题时，这种紧密性则不再成立。首先，通过对调整过程进行隐私审计来证明了这一点。我们的研究结果突显了当前理论隐私界与实证之间存在重大差距。

    arXiv:2402.13087v1 Announce Type: new  Abstract: We study the problem of guaranteeing Differential Privacy (DP) in hyper-parameter tuning, a crucial process in machine learning involving the selection of the best run from several. Unlike many private algorithms, including the prevalent DP-SGD, the privacy implications of tuning remain insufficiently understood. Recent works propose a generic private solution for the tuning process, yet a fundamental question still persists: is the current privacy bound for this solution tight?   This paper contributes both positive and negative answers to this question. Initially, we provide studies affirming the current privacy analysis is indeed tight in a general sense. However, when we specifically study the hyper-parameter tuning problem, such tightness no longer holds. This is first demonstrated by applying privacy audit on the tuning process. Our findings underscore a substantial gap between the current theoretical privacy bound and the empirica
    
[^5]: 优化人工胰腺设计以改善糖尿病管理

    Optimizing the Design of an Artificial Pancreas to Improve Diabetes Management

    [https://arxiv.org/abs/2402.07949](https://arxiv.org/abs/2402.07949)

    通过神经进化算法优化人工胰腺治疗策略，减少糖尿病患者的血糖偏差，并且降低注射次数。

    

    糖尿病是一种慢性疾病，影响美国境内有3800万人，它会影响身体将食物转化为能量（即血糖）的能力。标准的治疗方法是通过使用人工胰腺，即持续胰岛素泵（基础注射），以及定期注射胰岛素（突发注射）来补充碳水化合物摄入量。治疗目标是将血糖保持在可接受范围的中心位置，通过持续血糖测量来进行衡量。次要目标是减少注射次数，因为对某些患者来说注射是不愉快且难以实施的。本研究使用神经进化来发现治疗的最佳策略。基于30天的治疗和单个患者的测量数据集，首先训练了随机森林来预测未来的血糖水平。然后通过进化了一个神经网络来指定碳水化合物摄入量、基础注射水平和突发注射。进化发现了一个帕累托前沿，减少了与目标值的偏差。

    Diabetes, a chronic condition that impairs how the body turns food into energy, i.e. blood glucose, affects 38 million people in the US alone. The standard treatment is to supplement carbohydrate intake with an artificial pancreas, i.e. a continuous insulin pump (basal shots), as well as occasional insulin injections (bolus shots). The goal of the treatment is to keep blood glucose at the center of an acceptable range, as measured through a continuous glucose meter. A secondary goal is to minimize injections, which are unpleasant and difficult for some patients to implement. In this study, neuroevolution was used to discover an optimal strategy for the treatment. Based on a dataset of 30 days of treatment and measurements of a single patient, a random forest was first trained to predict future glucose levels. A neural network was then evolved to prescribe carbohydrates, basal pumping levels, and bolus injections. Evolution discovered a Pareto front that reduced deviation from the targe
    
[^6]: 神经网络的权重衰减和类内变化小会导致低秩偏差

    Neural Rank Collapse: Weight Decay and Small Within-Class Variability Yield Low-Rank Bias

    [https://arxiv.org/abs/2402.03991](https://arxiv.org/abs/2402.03991)

    神经网络中的权重衰减和小的类内变化与低秩偏差现象有关

    

    近期在深度学习领域的研究显示了一个隐含的低秩偏差现象：深度网络中的权重矩阵往往近似为低秩，在训练过程中或从已经训练好的模型中去除相对较小的奇异值可以显著减小模型大小，同时保持甚至提升模型性能。然而，大多数关于神经网络低秩偏差的理论研究都涉及到简化的线性深度网络。在本文中，我们考虑了带有非线性激活函数和权重衰减参数的通用网络，并展示了一个有趣的神经秩崩溃现象，它将训练好的网络的低秩偏差与网络的神经崩溃特性联系起来：随着权重衰减参数的增加，网络中每一层的秩呈比例递减，与前面层的隐藏空间嵌入的类内变化成反比。我们的理论发现得到了支持。

    Recent work in deep learning has shown strong empirical and theoretical evidence of an implicit low-rank bias: weight matrices in deep networks tend to be approximately low-rank and removing relatively small singular values during training or from available trained models may significantly reduce model size while maintaining or even improving model performance. However, the majority of the theoretical investigations around low-rank bias in neural networks deal with oversimplified deep linear networks. In this work, we consider general networks with nonlinear activations and the weight decay parameter, and we show the presence of an intriguing neural rank collapse phenomenon, connecting the low-rank bias of trained networks with networks' neural collapse properties: as the weight decay parameter grows, the rank of each layer in the network decreases proportionally to the within-class variability of the hidden-space embeddings of the previous layers. Our theoretical findings are supporte
    
[^7]: 证实无荒原存在是否意味着经典模拟？或者，为什么我们需要重新思考变分量子计算

    Does provable absence of barren plateaus imply classical simulability? Or, why we need to rethink variational quantum computing

    [https://arxiv.org/abs/2312.09121](https://arxiv.org/abs/2312.09121)

    常用的具有无荒原证明的模型也可以在进行初始数据采集阶段从量子设备中收集一些经典数据的情况下经典模拟

    

    最近，人们对荒原现象进行了大量研究。 在这篇观点文章中，我们面对了越来越明显的问题，并提出了一个许多人暗示但尚未明确解决的问题：允许避免荒原的结构是否也可以被利用来有效地经典模拟损失？ 我们提供了强有力的证据，表明常用的具有无荒原证明的模型也可以在进行初始数据采集阶段从量子设备中收集一些经典数据的情况下经典模拟。 这是因为荒原现象是由维度的诅咒导致的，而目前解决问题的方法最终将问题编码到一些小的、经典可模拟的子空间中。 因此，尽管强调量子计算可以是收集数据的必要条件，我们的分析引起了严重的思考。

    arXiv:2312.09121v2 Announce Type: replace-cross  Abstract: A large amount of effort has recently been put into understanding the barren plateau phenomenon. In this perspective article, we face the increasingly loud elephant in the room and ask a question that has been hinted at by many but not explicitly addressed: Can the structure that allows one to avoid barren plateaus also be leveraged to efficiently simulate the loss classically? We present strong evidence that commonly used models with provable absence of barren plateaus are also classically simulable, provided that one can collect some classical data from quantum devices during an initial data acquisition phase. This follows from the observation that barren plateaus result from a curse of dimensionality, and that current approaches for solving them end up encoding the problem into some small, classically simulable, subspaces. Thus, while stressing quantum computers can be essential for collecting data, our analysis sheds seriou
    
[^8]: 基于仿真的贝叶斯优化

    Simulation Based Bayesian Optimization. (arXiv:2401.10811v1 [stat.ML])

    [http://arxiv.org/abs/2401.10811](http://arxiv.org/abs/2401.10811)

    本文介绍了基于仿真的贝叶斯优化（SBBO）作为一种新方法，用于通过仅需基于采样的访问来优化获取函数。

    

    贝叶斯优化是一种将先验知识与持续函数评估相结合的强大方法，用于优化黑盒函数。贝叶斯优化通过构建与协变量相关的目标函数的概率代理模型来指导未来评估点的选择。对于平滑连续的搜索空间，高斯过程经常被用作代理模型，因为它们提供对后验预测分布的解析访问，从而便于计算和优化获取函数。然而，在涉及对分类或混合协变量空间进行优化的复杂情况下，高斯过程可能不是理想的选择。本文介绍了一种名为基于仿真的贝叶斯优化（SBBO）的新方法，该方法仅需要对后验预测分布进行基于采样的访问，以优化获取函数。

    Bayesian Optimization (BO) is a powerful method for optimizing black-box functions by combining prior knowledge with ongoing function evaluations. BO constructs a probabilistic surrogate model of the objective function given the covariates, which is in turn used to inform the selection of future evaluation points through an acquisition function. For smooth continuous search spaces, Gaussian Processes (GPs) are commonly used as the surrogate model as they offer analytical access to posterior predictive distributions, thus facilitating the computation and optimization of acquisition functions. However, in complex scenarios involving optimizations over categorical or mixed covariate spaces, GPs may not be ideal.  This paper introduces Simulation Based Bayesian Optimization (SBBO) as a novel approach to optimizing acquisition functions that only requires \emph{sampling-based} access to posterior predictive distributions. SBBO allows the use of surrogate probabilistic models tailored for co
    
[^9]: 工业厂房智能状态监测: 方法论和不确定性管理策略综述

    Intelligent Condition Monitoring of Industrial Plants: An Overview of Methodologies and Uncertainty Management Strategies. (arXiv:2401.10266v1 [cs.LG])

    [http://arxiv.org/abs/2401.10266](http://arxiv.org/abs/2401.10266)

    本论文综述了工业厂房智能状态监测和故障检测和诊断方法，重点关注了Tennessee Eastman Process。调研总结了最流行和最先进的深度学习和机器学习算法，并探讨了算法的优劣势。还讨论了不平衡数据和无标记样本等挑战，以及深度学习模型如何应对。比较了不同算法在Tennessee Eastman Process上的准确性和规格。

    

    状态监测在现代工业系统的安全性和可靠性中起着重要作用。人工智能（AI）方法作为一种在工业应用中日益受到学术界和行业关注的增长主题和一种强大的故障识别方式。本文概述了工业厂房智能状态监测和故障检测和诊断方法，重点关注开源基准Tennessee Eastman Process（TEP）。在这项调查中，总结了用于工业厂房状态监测、故障检测和诊断的最流行和最先进的深度学习（DL）和机器学习（ML）算法，并研究了每种算法的优点和缺点。还涵盖了不平衡数据、无标记样本以及深度学习模型如何处理这些挑战。最后，比较了利用Tennessee Eastman Process的不同算法的准确性和规格。

    Condition monitoring plays a significant role in the safety and reliability of modern industrial systems. Artificial intelligence (AI) approaches are gaining attention from academia and industry as a growing subject in industrial applications and as a powerful way of identifying faults. This paper provides an overview of intelligent condition monitoring and fault detection and diagnosis methods for industrial plants with a focus on the open-source benchmark Tennessee Eastman Process (TEP). In this survey, the most popular and state-of-the-art deep learning (DL) and machine learning (ML) algorithms for industrial plant condition monitoring, fault detection, and diagnosis are summarized and the advantages and disadvantages of each algorithm are studied. Challenges like imbalanced data, unlabelled samples and how deep learning models can handle them are also covered. Finally, a comparison of the accuracies and specifications of different algorithms utilizing the Tennessee Eastman Process 
    
[^10]: 迈向可识别的无监督领域转换：一种多样化分布匹配的方法

    Towards Identifiable Unsupervised Domain Translation: A Diversified Distribution Matching Approach. (arXiv:2401.09671v1 [cs.LG])

    [http://arxiv.org/abs/2401.09671](http://arxiv.org/abs/2401.09671)

    本研究旨在解决无监督领域转换中的可识别性问题，引入了一个MPA消除理论，解决了CycleGAN及其变体产生内容不对齐的限制。

    

    无监督领域转换（UDT）旨在找到将一个领域的样本（例如素描）转换为另一个领域（例如照片）的函数，同时不改变高层语义意义（也称为“内容”）。这些转换函数通常通过转换源领域和目标领域的概率分布来寻找。CycleGAN可以说是这一领域中最具代表性的方法。然而，文献中指出CycleGAN及其变体可能无法识别所需的转换函数，并产生内容不对齐的转换。这种局限性源于学习准则解空间中存在多个转换函数，称为“保度自同构（MPA）”。尽管意识到了这种可识别性问题，但解决方案仍然难以找到。本研究深入探究了核心的可识别性问题，并引入了MPA消除理论。我们的分析表明...

    Unsupervised domain translation (UDT) aims to find functions that convert samples from one domain (e.g., sketches) to another domain (e.g., photos) without changing the high-level semantic meaning (also referred to as ``content''). The translation functions are often sought by probability distribution matching of the transformed source domain and target domain. CycleGAN stands as arguably the most representative approach among this line of work. However, it was noticed in the literature that CycleGAN and variants could fail to identify the desired translation functions and produce content-misaligned translations. This limitation arises due to the presence of multiple translation functions -- referred to as ``measure-preserving automorphism" (MPA) -- in the solution space of the learning criteria. Despite awareness of such identifiability issues, solutions have remained elusive. This study delves into the core identifiability inquiry and introduces an MPA elimination theory. Our analysi
    
[^11]: 关于分布鲁棒强化学习的基础

    On the Foundation of Distributionally Robust Reinforcement Learning. (arXiv:2311.09018v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.09018](http://arxiv.org/abs/2311.09018)

    该论文为分布鲁棒强化学习的理论基础做出了贡献，通过一个综合的建模框架，决策者在最坏情况下的分布转变下选择最优策略，并考虑了各种建模属性和对手引起的转变的灵活性。

    

    出于对在训练和部署之间环境变化时鲁棒策略的需求，我们为分布鲁棒强化学习的理论基础做出了贡献。通过一个以分布鲁棒马尔科夫决策过程（DRMDPs）为中心的综合建模框架，我们使决策者在一个由对手操纵的最坏情况分布转变下选择最优策略。通过统一和扩展现有的表述，我们严格构建了适用于决策者和对手的各种建模属性的DRMDPs，包括适应性粒度、探索历史依赖性、马尔科夫和马尔科夫时间齐次的决策者和对手动态。此外，我们深入研究了对手引起的转变的灵活性，研究了SA和S-矩形性。在这个DRMDP框架下，我们研究了实现鲁棒性所需的条件。

    Motivated by the need for a robust policy in the face of environment shifts between training and the deployment, we contribute to the theoretical foundation of distributionally robust reinforcement learning (DRRL). This is accomplished through a comprehensive modeling framework centered around distributionally robust Markov decision processes (DRMDPs). This framework obliges the decision maker to choose an optimal policy under the worst-case distributional shift orchestrated by an adversary. By unifying and extending existing formulations, we rigorously construct DRMDPs that embraces various modeling attributes for both the decision maker and the adversary. These attributes include adaptability granularity, exploring history-dependent, Markov, and Markov time-homogeneous decision maker and adversary dynamics. Additionally, we delve into the flexibility of shifts induced by the adversary, examining SA and S-rectangularity. Within this DRMDP framework, we investigate conditions for the e
    
[^12]: 这不是一个苹果：多模态嵌入中的对抗幻觉

    Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings. (arXiv:2308.11804v1 [cs.CR])

    [http://arxiv.org/abs/2308.11804](http://arxiv.org/abs/2308.11804)

    该论文研究了多模态嵌入中的对抗幻觉问题。对手可以扰动输入的任意模态，使其嵌入与其他模态的任意输入接近，从而实现任意图像与任意文本、任意文本与任意声音的对齐。该问题与下游任务无关，对生成和分类任务会产生误导。

    

    多模态编码器将图像、声音、文本、视频等映射到一个单一的嵌入空间中，通过对齐不同模态的表示（例如将一张狗的图像与一种叫声相关联）。我们展示了多模态嵌入可以受到一种我们称之为“对抗幻觉”的攻击。给定任意模态的输入，对手可以扰动它，使其嵌入接近于另一模态中任意对手选择的输入的嵌入。幻觉使对手能够将任意图像与任意文本、任意文本与任意声音等进行对齐。对抗幻觉利用了嵌入空间中的接近性，因此与下游任务无关。使用ImageBind嵌入，我们演示了在没有具体下游任务知识的情况下，通过对抗性对齐的输入如何误导图像生成、文本生成和零样例分类。

    Multi-modal encoders map images, sounds, texts, videos, etc. into a single embedding space, aligning representations across modalities (e.g., associate an image of a dog with a barking sound). We show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an input in any modality, an adversary can perturb it so as to make its embedding close to that of an arbitrary, adversary-chosen input in another modality. Illusions thus enable the adversary to align any image with any text, any text with any sound, etc.  Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks. Using ImageBind embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, and zero-shot classification.
    
[^13]: 增量学习下的稀疏均值鲁棒性估计

    Robust Sparse Mean Estimation via Incremental Learning. (arXiv:2305.15276v1 [cs.LG])

    [http://arxiv.org/abs/2305.15276](http://arxiv.org/abs/2305.15276)

    本文提出了一个简单的增量学习方法，仅需要较少的样本即可在近线性时间内估计稀疏均值，克服了现有估计器的限制。

    

    本文研究了稀疏均值的鲁棒性估计问题，旨在估计从重尾分布中抽取的部分损坏样本的$k$-稀疏均值。现有估计器在这种情况下面临两个关键挑战：首先，它们受到一个被推测的计算统计权衡的限制，这意味着任何计算效率高的算法需要$\tilde\Omega(k^2)$个样本，而其在统计上最优的对应物只需要$\tilde O(k)$个样本。其次，现有的估计器规模随着环境的维度增加而急剧上升，难以在实践中使用。本文提出了一个简单的均值估计器，在适度的条件下克服了这两个挑战：它在几乎线性的时间和内存中运行（相对于环境维度），同时只需要$\tilde O(k)$个样本来恢复真实的均值。我们方法的核心是增量学习现象，我们引入了一个简单的非凸框架，它可以将均值估计问题转化为线性回归问题，并利用基于增量学习的算法大大提高了效率。

    In this paper, we study the problem of robust sparse mean estimation, where the goal is to estimate a $k$-sparse mean from a collection of partially corrupted samples drawn from a heavy-tailed distribution. Existing estimators face two critical challenges in this setting. First, they are limited by a conjectured computational-statistical tradeoff, implying that any computationally efficient algorithm needs $\tilde\Omega(k^2)$ samples, while its statistically-optimal counterpart only requires $\tilde O(k)$ samples. Second, the existing estimators fall short of practical use as they scale poorly with the ambient dimension. This paper presents a simple mean estimator that overcomes both challenges under moderate conditions: it runs in near-linear time and memory (both with respect to the ambient dimension) while requiring only $\tilde O(k)$ samples to recover the true mean. At the core of our method lies an incremental learning phenomenon: we introduce a simple nonconvex framework that ca
    
[^14]: 卷积神经网络的分类自动学习算法选择

    Automatic learning algorithm selection for classification via convolutional neural networks. (arXiv:2305.09101v1 [cs.LG])

    [http://arxiv.org/abs/2305.09101](http://arxiv.org/abs/2305.09101)

    本文提出了一种直接使用表格数据自动训练卷积神经网络的方法，以学习数据固有的结构，无需识别元特征。在模拟和真实数据集上，该方法均取得了竞争性能。

    

    与其他任务一样，构建机器学习模型的过程可以受益于先前的经验。基于元学习的分类器选择通过比较不同数据集的特征和机器学习技术的性能，以提高当前建模过程中的决策。然而，本文提出了一种自动学习方案，直接使用表格数据为二进制分类训练卷积神经网络，以学习数据固有的结构而不识别元特征。在模拟数据集上进行的实验显示，所提出的方法在识别线性和非线性模式方面达到了几乎完美的性能，优于基于元特征的传统两步方法。文中所提出的方法随后在真实数据集上进行评估，证明了与现有最先进方法相比的竞争性能。

    As in any other task, the process of building machine learning models can benefit from prior experience. Meta-learning for classifier selection gains knowledge from characteristics of different datasets and/or previous performance of machine learning techniques to make better decisions for the current modeling process. Meta-learning approaches first collect meta-data that describe this prior experience and then use it as input for an algorithm selection model. In this paper, however, we propose an automatic learning scheme in which we train convolutional networks directly with the information of tabular datasets for binary classification. The goal of this study is to learn the inherent structure of the data without identifying meta-features. Experiments with simulated datasets show that the proposed approach achieves nearly perfect performance in identifying linear and nonlinear patterns, outperforming the traditional two-step method based on meta-features. The proposed method is then 
    
[^15]: 增强神经网络对多样化$\ell_p$攻击的鲁棒性:鲁棒模态连接导向的对抗性防御

    Robust Mode Connectivity-Oriented Adversarial Defense: Enhancing Neural Network Robustness Against Diversified $\ell_p$ Attacks. (arXiv:2303.10225v1 [cs.AI])

    [http://arxiv.org/abs/2303.10225](http://arxiv.org/abs/2303.10225)

    本文提出一种新颖的鲁棒模态连接导向的对抗性防御，实现神经网络对多样化$\ell_p$攻击的鲁棒性，其中包括两个基于种群学习的学习阶段。

    

    对抗性鲁棒性是衡量神经网络在推理阶段抵御对抗性攻击能力的关键概念。最近的研究表明，尽管使用的强化鲁棒性训练技术能够提高对一种类型的攻击的鲁棒性，但模型仍然容易受到多样化的$\ell_p$攻击。为了实现多样化的$\ell_p$鲁棒性，我们提出了一种新颖的鲁棒模态连接 (RMC) 导向的对抗性防御，它包含两个基于种群学习的学习阶段。第一个阶段，RMC，能够搜索两个预先训练模型之间的模型参数空间，并找到包含高鲁棒性点的路径以抵御多样化的$\ell_p$攻击。基于RMC的有效性，我们开发了第二个阶段，基于RMC的优化，其中RMC作为神经网络多样化$\ell_p$鲁棒性进一步增强的基本单元。为了提高计算效率，我们将学习与仅选择子集的对抗性示例相结合，这导致了一组较小的代表性对抗性示例，可用于增强神经网络对多样化$\ell_p$攻击的鲁棒性。

    Adversarial robustness is a key concept in measuring the ability of neural networks to defend against adversarial attacks during the inference phase. Recent studies have shown that despite the success of improving adversarial robustness against a single type of attack using robust training techniques, models are still vulnerable to diversified $\ell_p$ attacks. To achieve diversified $\ell_p$ robustness, we propose a novel robust mode connectivity (RMC)-oriented adversarial defense that contains two population-based learning phases. The first phase, RMC, is able to search the model parameter space between two pre-trained models and find a path containing points with high robustness against diversified $\ell_p$ attacks. In light of the effectiveness of RMC, we develop a second phase, RMC-based optimization, with RMC serving as the basic unit for further enhancement of neural network diversified $\ell_p$ robustness. To increase computational efficiency, we incorporate learning with a sel
    
[^16]: 超边界图神经网络：方法和应用综述

    Hyperbolic Graph Neural Networks: A Review of Methods and Applications. (arXiv:2202.13852v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.13852](http://arxiv.org/abs/2202.13852)

    本文综述了当前超边界图神经网络的技术细节，提出了一个通用框架，并总结了每个组件的变体。此外，还介绍了各种与HGNN相关的应用和当前面临的挑战。

    

    图神经网络将传统的神经网络推广到了图结构数据，并因其出色的表征能力而受到广泛关注。尽管取得了显著的成就，但欧几里得模型在与图相关的学习中的性能仍然受到欧几里得几何的表征能力的限制，特别是对于具有高度非欧几里得潜在解剖的数据集。最近，超边界空间在处理具有树状结构和幂律分布的图数据方面越来越受欢迎，这归功于其指数级的增长特性。在本综述中，我们全面回顾了当前超边界图神经网络的技术细节，将它们统一为一个通用框架，并总结了每个组件的变体。更重要的是，我们介绍了各种与HGNN相关的应用。最后，我们还确定了一些挑战，这些挑战可能成为进一步发展图神经网络成就的指导方针。

    Graph neural networks generalize conventional neural networks to graph-structured data and have received widespread attention due to their impressive representation ability. In spite of the remarkable achievements, the performance of Euclidean models in graph-related learning is still bounded and limited by the representation ability of Euclidean geometry, especially for datasets with highly non-Euclidean latent anatomy. Recently, hyperbolic space has gained increasing popularity in processing graph data with tree-like structure and power-law distribution, owing to its exponential growth property. In this survey, we comprehensively revisit the technical details of the current hyperbolic graph neural networks, unifying them into a general framework and summarizing the variants of each component. More importantly, we present various HGNN-related applications. Last, we also identify several challenges, which potentially serve as guidelines for further flourishing the achievements of graph
    

