# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Gradient-free online learning of subgrid-scale dynamics with neural emulators.](http://arxiv.org/abs/2310.19385) | 本文提出了一种利用神经仿真器在线训练亚网格参数化的算法，通过后验损失函数适应非可微分数值求解器，并通过时间积分步骤允许梯度传播。实验证明，将神经仿真器和参数化组件分别用相应的损失量进行训练是必要的，以最小化某些近似偏差的传播。 |
| [^2] | [Graph Neural Networks with polynomial activations have limited expressivity.](http://arxiv.org/abs/2310.13139) | 本文证明了具有多项式激活函数的图神经网络无法表达GC2查询，与常用的非多项式激活函数存在分离，这回答了一个开放问题。 |
| [^3] | [Quality-Diversity through AI Feedback.](http://arxiv.org/abs/2310.13032) | 基于AI反馈的质量-多样性（QDAIF）算法利用语言模型来生成和评估创造性写作，比传统算法更广泛地覆盖高质量样本的搜索空间。 |
| [^4] | [Improving SCGAN's Similarity Constraint and Learning a Better Disentangled Representation.](http://arxiv.org/abs/2310.12262) | 本论文改进了SCGAN模型中的相似性约束，使用SSIM度量图像相似性并应用对比损失原则，提高了模型的性能和泛化能力。 |
| [^5] | [Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory.](http://arxiv.org/abs/2310.04935) | 这项工作利用PAC-Bayesian理论为变分自动编码器提供了统计保证，包括对后验分布、重构损失和输入与生成分布之间距离的上界。 |
| [^6] | [Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization.](http://arxiv.org/abs/2310.03234) | 本文研究了一种新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)，通过扩展已有的研究，我们研究了非光滑弱凸FCCO的问题，并提出了一种单循环算法来找到Moreau环的ε-稳定点。 |
| [^7] | [SmartPlay : A Benchmark for LLMs as Intelligent Agents.](http://arxiv.org/abs/2310.01557) | SmartPlay是一个用于评估LLMs作为智能Agent能力的基准，包括6个具有不同挑战的游戏，并测试了智能LLM Agent的多种关键能力。这不仅是一个评估LLM Agent整体性能的严格测试场地，还可以分析每个能力的表现。 |
| [^8] | [Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit.](http://arxiv.org/abs/2309.16620) | 这项研究通过残差分支尺度和$\mu$P参数化的残差网络，实现了深度学习中超参数的跨宽度和深度的转移。 |
| [^9] | [Temporal graph models fail to capture global temporal dynamics.](http://arxiv.org/abs/2309.15730) | 时间图模型无法捕捉全局时间动态，我们提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了两个基于Wasserstein距离的度量来量化全局动态。我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用，我们还展示了简单的负采样方法可能导致模型退化。我们提出了改进的负采样方案，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。 |
| [^10] | [Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM.](http://arxiv.org/abs/2309.14348) | 本文提出了一种稳健对齐的LLM（RA-LLM），用于防御可能发生的对齐破坏攻击。RA-LLM可以直接在现有的对齐LLM上构建，并通过稳健的对齐检查函数来确保其有效性。 |
| [^11] | [A Study of Forward-Forward Algorithm for Self-Supervised Learning.](http://arxiv.org/abs/2309.11955) | 本文首次研究了自监督表示学习中的向前-向前算法和反向传播的性能，发现在自监督表示学习中，向前-向前算法与反向传播表现相当。 |
| [^12] | [A Function Interpretation Benchmark for Evaluating Interpretability Methods.](http://arxiv.org/abs/2309.03886) | 本文介绍了一个用于评估自动解释性方法的基准套件，该套件包括了类似于传统系统组件的函数。 |
| [^13] | [Temporal Inductive Path Neural Network for Temporal Knowledge Graph Reasoning.](http://arxiv.org/abs/2309.03251) | 本论文提出了一种临时归纳路径神经网络（TiPNN）用于时间知识图的推理，采用实体独立的角度建模历史信息，并通过临时归纳路径提取结构和时间信息。 |
| [^14] | [Microscopy Image Segmentation via Point and Shape Regularized Data Synthesis.](http://arxiv.org/abs/2308.09835) | 本文提出了一种采用合成训练数据的统一流程，通过点注释和形状先验进行显微图像分割。该方法克服了标注成本高的问题，且仍能提供关键信息用于分割。该流程包括三个阶段：获取点注释并生成伪密集分割掩码，将伪掩码转化为真实显微图像，并通过对象级一致性进行正则化。 |
| [^15] | [On the Trustworthiness Landscape of State-of-the-art Generative Models: A Comprehensive Survey.](http://arxiv.org/abs/2307.16680) | 本文综合调查了大规模生成模型的可信度问题，涵盖了隐私、安全、公平性和责任等多个维度，并提出了实际建议和未来发展方向。 |
| [^16] | [BayesDAG: Gradient-Based Posterior Sampling for Causal Discovery.](http://arxiv.org/abs/2307.13917) | 这项研究引入了一种基于梯度的后验采样方法，用于解决Bayesian causal discovery中的计算挑战，能够高效地推断因果模型，并且不依赖于DAG正则化。 |
| [^17] | [Gradient-Based Spectral Embeddings of Random Dot Product Graphs.](http://arxiv.org/abs/2307.13818) | 本文介绍了基于梯度的随机点积图谱嵌入方法，并通过利用非凸优化技术改进了在观察图中估计节点潜在向量的任务。同时，作者还提出了一阶梯度下降方法来更好地解决嵌入问题，并适应更广泛的实用网络嵌入应用。 |
| [^18] | [ECSIC: Epipolar Cross Attention for Stereo Image Compression.](http://arxiv.org/abs/2307.10284) | ECSIC是一种用于立体图像压缩的新方法，通过利用左右图像之间的相互信息进行联合压缩，并使用新颖的立体交叉注意力模块和立体上下文模块实现。与现有方法相比，ECSIC在两个流行的立体图像数据集上取得了最先进的性能，并且具有快速编码和解码的特性。 |
| [^19] | [FreeDrag: Point Tracking is Not What You Need for Interactive Point-based Image Editing.](http://arxiv.org/abs/2307.04684) | FreeDrag提出了一种基于特征的方法来解决DragGAN在点追踪方面的困难，通过自适应模板特征、线性搜索和模糊定位技术，实现了稳定和高效的基于点的图像编辑。 |
| [^20] | [Margin Maximization in Attention Mechanism.](http://arxiv.org/abs/2306.13596) | 这篇论文证明了，在softmax-attention模型中，通过在p或等价的W上运行梯度下降，可以收敛到一个最大边缘解，这将局部最优的标记与非最优的标记分隔开。这明确地将注意力机制形式化为标记分离机制。 |
| [^21] | [Causal normalizing flows: from theory to practice.](http://arxiv.org/abs/2306.05415) | 本文研究了使用因果归一化流进行因果推论的方法，证明了在给定因果排序情况下，利用自回归归一化流可以恢复因果模型。通过实验和比较研究，证明了因果归一化流可用于解决实际问题。 |
| [^22] | [G$^2$uardFL: Safeguarding Federated Learning Against Backdoor Attacks through Attributed Client Graph Clustering.](http://arxiv.org/abs/2306.04984) | 本论文提出了G$^2$uardFL，这是一个基于属性化客户端图聚类的联邦学习保护框架，能够有效识别恶意客户端，即使恶意客户端数量高达50％。 |
| [^23] | [Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities.](http://arxiv.org/abs/2306.04829) | 本研究提出了一种新方法，利用预训练的自监督特征和时间特征相似性损失，实现了对真实世界视频的物体中心学习，在合成MOVi数据集上取得了最先进的性能。同时，本模型是首个能够扩展到无约束视频数据集的物体中心视频模型。 |
| [^24] | [Spike-based computation using classical recurrent neural networks.](http://arxiv.org/abs/2306.03623) | 本文提出了一种新的脉冲神经网络方法，通过修改一种易于训练的循环神经网络的动态特性，使其产生基于脉冲的计算，并在进行了脉冲网络的训练后，在多个数据集上取得了最先进的性能。 |
| [^25] | [Cycle Consistency Driven Object Discovery.](http://arxiv.org/abs/2306.02204) | 该方法通过循环一致性目标的引入，明确优化场景中每个物体应映射到不同槽位的约束，从而实现了在完全无监督的情况下有效地学习发现物体。在实验中表现出了优于现有方法的性能。 |
| [^26] | [Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection.](http://arxiv.org/abs/2306.00006) | 本文针对图形异常监测数据集中存在的一类同型现象，提出了一种新的无监督异常评分度量——当前节点亲和力，并通过学习量身定制的节点表示，实现了截断亲和力最大化（TAM）方法，优化在原始图形结构上进行，能够有效进行双重One-Class的GAD。 |
| [^27] | [Off-By-One Implementation Error in J-UNIWARD.](http://arxiv.org/abs/2305.19776) | J-UNIWARD 是一种将秘密信息隐藏在JPEG图像中的隐写方法，本文发现了其实现中存在的一个 off-by-one 错误，使一些图像块被高估，另一些被低估，同时提供了一个概念验证用于检测此种错误。 |
| [^28] | [When Does Optimizing a Proper Loss Yield Calibration?.](http://arxiv.org/abs/2305.18764) | 研究优化适当的损失函数是否能在受限的预测器族中得到校准的模型，使用局部最优条件取代全局最优性条件并在此基础上进行了严格的证明。 |
| [^29] | [Variational Classification.](http://arxiv.org/abs/2305.10406) | 提出一种新的变分分类方法，通过引入潜变量建模来优化训练，允许灵活的设计选择以改善校准和对抗鲁棒性，实验结果表明其对于域外数据的分类准确性得到了保持。 |
| [^30] | [Loss minimization yields multicalibration for large neural networks.](http://arxiv.org/abs/2304.09424) | 本文展示了对于大型神经网络大小，最优地最小化损失会导致多校准，以提供公平的预测结果。 |
| [^31] | [Coupled Multiwavelet Neural Operator Learning for Coupled Partial Differential Equations.](http://arxiv.org/abs/2303.02304) | 本论文提出一种耦合多小波神经算子学习的方案，解决了处理耦合多变量映射问题的难点，能够显著提高解决耦合偏微分方程的准确性，并在实验中得到了验证。 |
| [^32] | [Fixing Overconfidence in Dynamic Neural Networks.](http://arxiv.org/abs/2302.06359) | 该论文提出了一种修复动态神经网络中过度自信问题的方法，通过对最后几层进行概率化处理，量化和纳入不确定性并有助于决定计算预算的确定。 |
| [^33] | [Construction of Hierarchical Neural Architecture Search Spaces based on Context-free Grammars.](http://arxiv.org/abs/2211.01842) | 本研究基于无上下文文法提出了一个统一的搜索空间设计框架，可以生成表达力强大的分层搜索空间，实现了对整个体系结构的搜索并促进结构的规律性。 |
| [^34] | [Neural Eigenfunctions Are Structured Representation Learners.](http://arxiv.org/abs/2210.12637) | 本文提出了一种称为神经特征映射的结构化自适应深度表示方法，它通过神经网络对特征值函数进行参数化建模。应用神经特征映射可以得到类似于流行的自监督学习方法的目标函数，并具有打破对称性的属性，从而产生结构化表示，其中特征按重要性进行排序。在图像检索系统中，通过根据特征的重要性进行截断，我们的方法所需的表示长度比领先的自监督学习方法短16倍，同时具有相似的检索性能。 |
| [^35] | [BAFFLE: Backdoor Attack in Offline Reinforcement Learning.](http://arxiv.org/abs/2210.04688) | 本文研究离线增强学习中的后门攻击，通过向数据中添加扰动，使得智能体在注入触发器的观测值上采取低奖励动作，从而提出了BAFFLE方法。 |
| [^36] | [Collaborative causal inference on distributed data.](http://arxiv.org/abs/2208.07898) | 提出了一种数据协作准实验（DC-QE）方法，可以在保护隐私的前提下对分布式数据进行因果推断。通过共享中间表示而不是私有数据，估计倾向分数和处理效应，能够减少随机误差和偏差，相比现有方法有更好的估计结果。 |
| [^37] | [Multi-Frequency Joint Community Detection and Phase Synchronization.](http://arxiv.org/abs/2206.12276) | 本文提出了两种简单而高效的算法，利用MLE公式并从多个频率的信息中受益，用于解决具有相对相位的随机块模型上的联合社区检测和相位同步问题。 |
| [^38] | [Predicting Census Survey Response Rates With Parsimonious Additive Models and Structured Interactions.](http://arxiv.org/abs/2108.11328) | 本文提出了一种可解释的非参数加性模型，使用少量主要和成对交互效应预测调查反应率。该模型可以生成易于可视化和解释的预测面，并取得了 ROAM 数据集上的最先进性能，可以提供改进美国人口普查局和其他调查的反应率议论。 |

# 详细

[^1]: 基于神经仿真器的无梯度在线学习亚网格尺度动力学

    Gradient-free online learning of subgrid-scale dynamics with neural emulators. (arXiv:2310.19385v2 [physics.comp-ph] UPDATED)

    [http://arxiv.org/abs/2310.19385](http://arxiv.org/abs/2310.19385)

    本文提出了一种利用神经仿真器在线训练亚网格参数化的算法，通过后验损失函数适应非可微分数值求解器，并通过时间积分步骤允许梯度传播。实验证明，将神经仿真器和参数化组件分别用相应的损失量进行训练是必要的，以最小化某些近似偏差的传播。

    

    本文提出了一种通用算法，用于在线训练基于机器学习的亚网格参数化，并通过后验损失函数适应非可微分数值求解器。所提出的方法利用神经仿真器训练简化状态空间求解器的近似值，然后通过时间积分步骤允许梯度传播。该算法能够在不计算原始求解器梯度的情况下恢复大部分在线策略的好处。实验证明，将神经仿真器和参数化组件分别用相应的损失量进行训练是必要的，以最小化某些近似偏差的传播。

    In this paper, we propose a generic algorithm to train machine learning-based subgrid parametrizations online, i.e., with $\textit{a posteriori}$ loss functions for non-differentiable numerical solvers. The proposed approach leverage neural emulators to train an approximation of the reduced state-space solver, which is then used to allows gradient propagation through temporal integration steps. The algorithm is able to recover most of the benefit of online strategies without having to compute the gradient of the original solver. It is demonstrated that training the neural emulator and parametrization components separately with respective loss quantities is necessary in order to minimize the propagation of some approximation bias.
    
[^2]: 具有多项式激活函数的图神经网络具有有限的表达能力

    Graph Neural Networks with polynomial activations have limited expressivity. (arXiv:2310.13139v1 [cs.LG])

    [http://arxiv.org/abs/2310.13139](http://arxiv.org/abs/2310.13139)

    本文证明了具有多项式激活函数的图神经网络无法表达GC2查询，与常用的非多项式激活函数存在分离，这回答了一个开放问题。

    

    图神经网络（GNNs）的表达能力可以完全由适当的一阶逻辑片段来描述。换句话说，任何在标记图上解释的关于二元逻辑片段（GC2）的查询都可以使用一个大小仅取决于查询深度的GNN来表示。正如[Barcelo＆Al。，2020，Grohe，2021]指出的那样，这个描述适用于一组激活函数的家族，这表明GNN可以通过不同的激活函数选择来表达不同的逻辑层次结构。在本文中，我们证明了这样的层次结构的存在，证明了具有多项式激活函数的GNN无法表示GC2查询。这意味着多项式和常用的非多项式激活函数（如ReLU、sigmoid、双曲正切等）之间存在一个分离，并回答了[Grohe，2021]提出的一个悬而未决的问题。

    The expressivity of Graph Neural Networks (GNNs) can be entirely characterized by appropriate fragments of the first order logic. Namely, any query of the two variable fragment of graded modal logic (GC2) interpreted over labelled graphs can be expressed using a GNN whose size depends only on the depth of the query. As pointed out by [Barcelo & Al., 2020, Grohe, 2021 ], this description holds for a family of activation functions, leaving the possibibility for a hierarchy of logics expressible by GNNs depending on the chosen activation function. In this article, we show that such hierarchy indeed exists by proving that GC2 queries cannot be expressed by GNNs with polynomial activation functions. This implies a separation between polynomial and popular non polynomial activations (such as ReLUs, sigmoid and hyperbolic tan and others) and answers an open question formulated by [Grohe, 2021].
    
[^3]: AI反馈促进的质量-多样性算法

    Quality-Diversity through AI Feedback. (arXiv:2310.13032v1 [cs.CL])

    [http://arxiv.org/abs/2310.13032](http://arxiv.org/abs/2310.13032)

    基于AI反馈的质量-多样性（QDAIF）算法利用语言模型来生成和评估创造性写作，比传统算法更广泛地覆盖高质量样本的搜索空间。

    

    在许多文本生成问题中，用户可能不仅偏好单一回复，而是希望得到多样性的高质量输出以供选择。质量-多样性（QD）搜索算法旨在通过不断改进和多样化候选人群来实现这一目标。然而，QD在创作性写作等质性领域的应用受到算法指定质量和多样性度量的困难的限制。有趣的是，最近语言模型（LMs）的发展使得通过AI反馈指导搜索成为可能，其中LMs在自然语言中被提示来评估文本的质性方面。借助这一进展，我们引入了通过AI反馈实现的质量-多样性算法（QDAIF），其中进化算法应用LMs来生成变异并评估候选文本的质量和多样性。在创作性写作领域的评估中，与非QDAIF算法相比，QDAIF更广泛地覆盖高质量样本的指定搜索空间。

    In many text-generation problems, users may prefer not only a single response, but a diverse range of high-quality outputs from which to choose. Quality-diversity (QD) search algorithms aim at such outcomes, by continually improving and diversifying a population of candidates. However, the applicability of QD to qualitative domains, like creative writing, has been limited by the difficulty of algorithmically specifying measures of quality and diversity. Interestingly, recent developments in language models (LMs) have enabled guiding search through AI feedback, wherein LMs are prompted in natural language to evaluate qualitative aspects of text. Leveraging this development, we introduce Quality-Diversity through AI Feedback (QDAIF), wherein an evolutionary algorithm applies LMs to both generate variation and evaluate the quality and diversity of candidate text. When assessed on creative writing domains, QDAIF covers more of a specified search space with high-quality samples than do non-
    
[^4]: 改进SCGAN的相似性约束并学习更好的解耦表示

    Improving SCGAN's Similarity Constraint and Learning a Better Disentangled Representation. (arXiv:2310.12262v1 [cs.CV])

    [http://arxiv.org/abs/2310.12262](http://arxiv.org/abs/2310.12262)

    本论文改进了SCGAN模型中的相似性约束，使用SSIM度量图像相似性并应用对比损失原则，提高了模型的性能和泛化能力。

    

    SCGAN在生成对抗网络中添加了一个相似性约束，将生成的图像与条件之间的相似性作为正则化项。相似性约束作为导师，指导生成器网络理解基于条件的表示差异。我们深入理解了SCGAN的工作原理。这种理解使我们意识到相似性约束的功能类似于对比损失函数。我们相信，具有高度理解和智能的模型可以根据图像的结构和高级特征来度量它们之间的相似性，就像人类一样。我们对SCGAN进行了两个主要改变，以创建一个改进的模型：使用SSIM来度量图像之间的相似性，并将对比损失原则应用于相似性约束。改进的模型在FID和FactorVAE指标下表现更好。与其他模型相比，改进的模型还具有更好的泛化能力。

    SCGAN adds a similarity constraint between generated images and conditions as a regularization term on generative adversarial networks. Similarity constraint works as a tutor to instruct the generator network to comprehend the difference of representations based on conditions. We understand how SCGAN works on a deeper level. This understanding makes us realize that the similarity constraint functions like the contrastive loss function. We believe that a model with high understanding and intelligence measures the similarity between images based on their structure and high level features, just like humans do. Two major changes we applied to SCGAN in order to make a modified model are using SSIM to measure similarity between images and applying contrastive loss principles to the similarity constraint. The modified model performs better using FID and FactorVAE metrics. The modified model also has better generalisability compared to other models. Keywords Generative Adversarial Nets, Unsupe
    
[^5]: 使用PAC-Bayesian理论给变分自动编码器提供统计保证

    Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory. (arXiv:2310.04935v1 [cs.LG])

    [http://arxiv.org/abs/2310.04935](http://arxiv.org/abs/2310.04935)

    这项工作利用PAC-Bayesian理论为变分自动编码器提供了统计保证，包括对后验分布、重构损失和输入与生成分布之间距离的上界。

    

    自从它们的问世以来，变分自动编码器（VAEs）在机器学习中变得非常重要。尽管它们被广泛使用，关于它们的理论性质仍存在许多问题。本文利用PAC-Bayesian理论为VAEs提供统计保证。首先，我们推导出了基于独立样本的后验分布的首个PAC-Bayesian界限。然后，利用这一结果为VAE的重构损失提供了泛化保证，同时提供了输入分布与VAE生成模型定义的分布之间距离的上界。更重要的是，我们提供了输入分布与VAE生成模型定义的分布之间Wasserstein距离的上界。

    Since their inception, Variational Autoencoders (VAEs) have become central in machine learning. Despite their widespread use, numerous questions regarding their theoretical properties remain open. Using PAC-Bayesian theory, this work develops statistical guarantees for VAEs. First, we derive the first PAC-Bayesian bound for posterior distributions conditioned on individual samples from the data-generating distribution. Then, we utilize this result to develop generalization guarantees for the VAE's reconstruction loss, as well as upper bounds on the distance between the input and the regenerated distributions. More importantly, we provide upper bounds on the Wasserstein distance between the input distribution and the distribution defined by the VAE's generative model.
    
[^6]: 非光滑弱凸有限和耦合组合优化

    Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization. (arXiv:2310.03234v1 [math.OC])

    [http://arxiv.org/abs/2310.03234](http://arxiv.org/abs/2310.03234)

    本文研究了一种新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)，通过扩展已有的研究，我们研究了非光滑弱凸FCCO的问题，并提出了一种单循环算法来找到Moreau环的ε-稳定点。

    

    本文研究了一类新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)。由于其在机器学习和人工智能领域的广泛应用以及其解决基于经验风险最小化的随机算法的局限性，FCCO引起了越来越多的关注。然而，目前对于FCCO的研究假设内外函数都是光滑的，限制了其能够解决更多种类的问题的潜力。我们的研究从非光滑弱凸FCCO的角度进行了扩展，其中外函数是弱凸且非递减的，内函数是弱凸的。我们分析了一种单循环算法，并确定其在找到Moreau环的ε-稳定点的复杂度。

    This paper investigates new families of compositional optimization problems, called $\underline{\bf n}$on-$\underline{\bf s}$mooth $\underline{\bf w}$eakly-$\underline{\bf c}$onvex $\underline{\bf f}$inite-sum $\underline{\bf c}$oupled $\underline{\bf c}$ompositional $\underline{\bf o}$ptimization (NSWC FCCO). There has been a growing interest in FCCO due to its wide-ranging applications in machine learning and AI, as well as its ability to address the shortcomings of stochastic algorithms based on empirical risk minimization. However, current research on FCCO presumes that both the inner and outer functions are smooth, limiting their potential to tackle a more diverse set of problems. Our research expands on this area by examining non-smooth weakly-convex FCCO, where the outer function is weakly convex and non-decreasing, and the inner function is weakly-convex. We analyze a single-loop algorithm and establish its complexity for finding an $\epsilon$-stationary point of the Moreau env
    
[^7]: SmartPlay: 一种用于评估LLMs作为智能Agent能力的基准

    SmartPlay : A Benchmark for LLMs as Intelligent Agents. (arXiv:2310.01557v1 [cs.LG])

    [http://arxiv.org/abs/2310.01557](http://arxiv.org/abs/2310.01557)

    SmartPlay是一个用于评估LLMs作为智能Agent能力的基准，包括6个具有不同挑战的游戏，并测试了智能LLM Agent的多种关键能力。这不仅是一个评估LLM Agent整体性能的严格测试场地，还可以分析每个能力的表现。

    

    最近的大型语言模型(LLMs)在智能Agent和下一代自动化方面展示了巨大的潜力，但目前缺乏一个系统化的基准来评估LLMs作为Agent的能力。我们介绍了SmartPlay：一个具有挑战性的基准和评估LLMs作为Agent的方法论。SmartPlay包括6个不同的游戏，包括剪刀石头布、汉诺塔、Minecraft等。每个游戏都具有独特的设置，提供最多20个评估设置和无限的环境变化。SmartPlay中的每个游戏都独特地挑战了智能LLM Agent的9个重要能力的子集，包括对对象依赖的推理、提前规划、空间推理、从历史中学习和理解随机性。每个游戏测试的能力集的区别使我们能够单独分析每个能力。SmartPlay不仅是评估LLM Agent整体性能的严格测试场地，而且也是评估Agent在不同能力方面的性能的一个重要工具。

    Recent large language models (LLMs) have demonstrated great potential toward intelligent agents and next-gen automation, but there currently lacks a systematic benchmark for evaluating LLMs' abilities as agents. We introduce SmartPlay: both a challenging benchmark and a methodology for evaluating LLMs as agents. SmartPlay consists of 6 different games, including Rock-Paper-Scissors, Tower of Hanoi, Minecraft. Each game features a unique setting, providing up to 20 evaluation settings and infinite environment variations. Each game in SmartPlay uniquely challenges a subset of 9 important capabilities of an intelligent LLM agent, including reasoning with object dependencies, planning ahead, spatial reasoning, learning from history, and understanding randomness. The distinction between the set of capabilities each game test allows us to analyze each capability separately. SmartPlay serves not only as a rigorous testing ground for evaluating the overall performance of LLM agents but also as
    
[^8]: 残差网络中的深度超参数转移：动态和缩放限制

    Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit. (arXiv:2309.16620v1 [stat.ML])

    [http://arxiv.org/abs/2309.16620](http://arxiv.org/abs/2309.16620)

    这项研究通过残差分支尺度和$\mu$P参数化的残差网络，实现了深度学习中超参数的跨宽度和深度的转移。

    

    随着模型大小的增加，深度学习中超参数调整的成本不断上升，促使从业者寻找使用较小网络的代理方法进行调整。其中一个建议使用$\mu$P参数化网络，其中小宽度网络的最佳超参数转移到任意宽度的网络中。然而，在这个方案中，超参数不会在不同深度之间转移。为了解决这个问题，我们研究了具有$1/\sqrt{\text{depth}}$的残差分支尺度和$\mu$P参数化的残差网络。我们通过实验证明，使用这种参数化训练的残差结构，包括卷积ResNet和Vision Transformer，在CIFAR-10和ImageNet上展示了跨宽度和深度的最佳超参数转移。此外，我们的经验发现得到了理论的支持和动机。利用神经网络学习动力学的动态均场理论（DMFT）描述的最新进展，我们展示了

    The cost of hyperparameter tuning in deep learning has been rising with model sizes, prompting practitioners to find new tuning methods using a proxy of smaller networks. One such proposal uses $\mu$P parameterized networks, where the optimal hyperparameters for small width networks transfer to networks with arbitrarily large width. However, in this scheme, hyperparameters do not transfer across depths. As a remedy, we study residual networks with a residual branch scale of $1/\sqrt{\text{depth}}$ in combination with the $\mu$P parameterization. We provide experiments demonstrating that residual architectures including convolutional ResNets and Vision Transformers trained with this parameterization exhibit transfer of optimal hyperparameters across width and depth on CIFAR-10 and ImageNet. Furthermore, our empirical findings are supported and motivated by theory. Using recent developments in the dynamical mean field theory (DMFT) description of neural network learning dynamics, we show
    
[^9]: 时间图模型无法捕捉全局时间动态

    Temporal graph models fail to capture global temporal dynamics. (arXiv:2309.15730v1 [cs.IR])

    [http://arxiv.org/abs/2309.15730](http://arxiv.org/abs/2309.15730)

    时间图模型无法捕捉全局时间动态，我们提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了两个基于Wasserstein距离的度量来量化全局动态。我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用，我们还展示了简单的负采样方法可能导致模型退化。我们提出了改进的负采样方案，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。

    

    在动态链接属性预测的背景下，我们分析了最近发布的时间图基准，并提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了基于Wasserstein距离的两个度量，可以量化数据集的短期和长期全局动态的强度。通过分析我们出乎意料的强大基线，我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用。我们还展示了简单的负采样方法在训练过程中可能导致模型退化，导致无法对时间图网络进行排序的预测完全饱和。我们提出了改进的负采样方案用于训练和评估，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。我们的结果表明...

    A recently released Temporal Graph Benchmark is analyzed in the context of Dynamic Link Property Prediction. We outline our observations and propose a trivial optimization-free baseline of "recently popular nodes" outperforming other methods on all medium and large-size datasets in the Temporal Graph Benchmark. We propose two measures based on Wasserstein distance which can quantify the strength of short-term and long-term global dynamics of datasets. By analyzing our unexpectedly strong baseline, we show how standard negative sampling evaluation can be unsuitable for datasets with strong temporal dynamics. We also show how simple negative-sampling can lead to model degeneration during training, resulting in impossible to rank, fully saturated predictions of temporal graph networks. We propose improved negative sampling schemes for both training and evaluation and prove their usefulness. We conduct a comparison with a model trained non-contrastively without negative sampling. Our resul
    
[^10]: 通过稳健对齐的LLM抵御对齐破坏攻击

    Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM. (arXiv:2309.14348v1 [cs.CL])

    [http://arxiv.org/abs/2309.14348](http://arxiv.org/abs/2309.14348)

    本文提出了一种稳健对齐的LLM（RA-LLM），用于防御可能发生的对齐破坏攻击。RA-LLM可以直接在现有的对齐LLM上构建，并通过稳健的对齐检查函数来确保其有效性。

    

    最近，大型语言模型（LLMs）取得了显著的进展，并在各个领域得到广泛应用。不幸的是，人们越来越担心LLMs可能被滥用来生成有害或恶意内容。尽管有一系列的研究专注于对齐LLMs与人类价值观，并防止它们生成不适当的内容，但这些对齐通常是脆弱的，并且可以通过对抗优化或手工构建的越狱提示来绕过。在这项工作中，我们介绍了一种稳健对齐的LLM（RA-LLM），以防范潜在的对齐破坏攻击。RA-LLM可以直接构建在现有的对齐LLM上，通过具有稳健对齐检查功能的方法，而无需对原始LLM进行任何昂贵的重新训练或微调。此外，我们还通过理论分析验证了RA-LLM在防御对齐破坏攻击方面的有效性。通过现实世界的实验，

    Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments
    
[^11]: 自监督学习的向前-向前算法研究

    A Study of Forward-Forward Algorithm for Self-Supervised Learning. (arXiv:2309.11955v1 [cs.CV])

    [http://arxiv.org/abs/2309.11955](http://arxiv.org/abs/2309.11955)

    本文首次研究了自监督表示学习中的向前-向前算法和反向传播的性能，发现在自监督表示学习中，向前-向前算法与反向传播表现相当。

    

    在过去的几年中，自监督表示学习取得了显著的进展，其中一些最新方法能够在没有标签的情况下学习出有用的图像表示。这些方法使用了反向传播作为训练的事实标准。最近，Geoffrey Hinton提出了向前-向前算法作为一种替代的训练方法。它利用了两次向前传递和每层都有一个单独的损失函数来训练网络，从而避免了反向传播。在这项研究中，我们首次研究了向前-向前算法与反向传播在自监督表示学习中的性能，并对学习到的表示空间提供了一些见解。我们的基准测试使用了四个标准数据集，分别是MNIST、F-MNIST、SVHN和CIFAR-10，以及三种常用的自监督表示学习技术，即旋转、翻转和拼图。我们的主要发现是，在自监督表示学习中，向前-向前算法与反向传播表现相当。

    Self-supervised representation learning has seen remarkable progress in the last few years, with some of the recent methods being able to learn useful image representations without labels. These methods are trained using backpropagation, the de facto standard. Recently, Geoffrey Hinton proposed the forward-forward algorithm as an alternative training method. It utilizes two forward passes and a separate loss function for each layer to train the network without backpropagation.  In this study, for the first time, we study the performance of forward-forward vs. backpropagation for self-supervised representation learning and provide insights into the learned representation spaces. Our benchmark employs four standard datasets, namely MNIST, F-MNIST, SVHN and CIFAR-10, and three commonly used self-supervised representation learning techniques, namely rotation, flip and jigsaw.  Our main finding is that while the forward-forward algorithm performs comparably to backpropagation during (self-)
    
[^12]: 一个用于评估解释性方法的功能解释基准

    A Function Interpretation Benchmark for Evaluating Interpretability Methods. (arXiv:2309.03886v1 [cs.CL])

    [http://arxiv.org/abs/2309.03886](http://arxiv.org/abs/2309.03886)

    本文介绍了一个用于评估自动解释性方法的基准套件，该套件包括了类似于传统系统组件的函数。

    

    使用人类可读的描述标记神经网络子模块对于许多下游任务非常有用：这些描述可以暴露失败、引导干预，甚至可以解释重要的模型行为。到目前为止，大多数基于机械原理的已训练网络描述都涉及到小模型、狭义现象，并且需要大量人力。在不断增加的模型大小和复杂性中标记出所有人可解释的子计算几乎肯定需要能够自动生成和验证描述的工具。最近，利用学习模型进行标记的技术开始受到关注，但评估其有效性的方法有限且临时。我们应该如何验证和比较开放式标记工具？本文介绍了FIND（函数解释和描述），一个用于评估自动解释方法构建模块的基准套件。FIND包含了类似于传统系统的组件的函数。

    Labeling neural network submodules with human-legible descriptions is useful for many downstream tasks: such descriptions can surface failures, guide interventions, and perhaps even explain important model behaviors. To date, most mechanistic descriptions of trained networks have involved small models, narrowly delimited phenomena, and large amounts of human labor. Labeling all human-interpretable sub-computations in models of increasing size and complexity will almost certainly require tools that can generate and validate descriptions automatically. Recently, techniques that use learned models in-the-loop for labeling have begun to gain traction, but methods for evaluating their efficacy are limited and ad-hoc. How should we validate and compare open-ended labeling tools? This paper introduces FIND (Function INterpretation and Description), a benchmark suite for evaluating the building blocks of automated interpretability methods. FIND contains functions that resemble components of tr
    
[^13]: 临时归纳路径神经网络用于时间知识图推理

    Temporal Inductive Path Neural Network for Temporal Knowledge Graph Reasoning. (arXiv:2309.03251v1 [cs.AI])

    [http://arxiv.org/abs/2309.03251](http://arxiv.org/abs/2309.03251)

    本论文提出了一种临时归纳路径神经网络（TiPNN）用于时间知识图的推理，采用实体独立的角度建模历史信息，并通过临时归纳路径提取结构和时间信息。

    

    时间知识图（TKG）是传统知识图（KG）的扩展，融入了时间维度。在TKGs上进行推理是一个关键任务，旨在基于历史事件预测未来事实。关键挑战在于揭示历史子图和时间模式中的结构依赖关系。大多数现有方法依靠实体建模来模拟TKGs，因为图中的节点在知识表示中起着至关重要的作用。然而，现实场景通常涉及大量实体，并且随着时间的推移会出现新实体。这使得依赖于实体的方法很难应对大量实体，并且有效处理新出现的实体也成为一个重要的挑战。因此，我们提出了一种临时归纳路径神经网络（TiPNN），它以实体独立的角度对历史信息进行建模。具体而言，TiPNN采用了一个统一的图，名为历史时间图，来建模历史信息，并通过临时归纳路径提取结构和时间信息。

    Temporal Knowledge Graph (TKG) is an extension of traditional Knowledge Graph (KG) that incorporates the dimension of time. Reasoning on TKGs is a crucial task that aims to predict future facts based on historical occurrences. The key challenge lies in uncovering structural dependencies within historical subgraphs and temporal patterns. Most existing approaches model TKGs relying on entity modeling, as nodes in the graph play a crucial role in knowledge representation. However, the real-world scenario often involves an extensive number of entities, with new entities emerging over time. This makes it challenging for entity-dependent methods to cope with extensive volumes of entities, and effectively handling newly emerging entities also becomes a significant challenge. Therefore, we propose Temporal Inductive Path Neural Network (TiPNN), which models historical information in an entity-independent perspective. Specifically, TiPNN adopts a unified graph, namely history temporal graph, to
    
[^14]: 通过点和形状正则化的数据合成进行显微图像分割

    Microscopy Image Segmentation via Point and Shape Regularized Data Synthesis. (arXiv:2308.09835v1 [cs.CV])

    [http://arxiv.org/abs/2308.09835](http://arxiv.org/abs/2308.09835)

    本文提出了一种采用合成训练数据的统一流程，通过点注释和形状先验进行显微图像分割。该方法克服了标注成本高的问题，且仍能提供关键信息用于分割。该流程包括三个阶段：获取点注释并生成伪密集分割掩码，将伪掩码转化为真实显微图像，并通过对象级一致性进行正则化。

    

    当前基于深度学习的显微图像分割方法严重依赖于大量需要密集注释的训练数据，在实践中成本高且劳动密集。与完整标注所描述的对象的完整轮廓相比，点注释，特别是对象质心，更容易获取，并且仍然为后续分割提供关键信息。本文假设仅在训练期间有点注释，并开发了一个使用合成训练数据的统一流程进行显微图像分割的框架。我们的框架包括三个阶段：（1）获取点注释并使用形状先验约束采样一个伪密集分割掩码；（2）通过以非配对的方式训练的图像生成模型，将伪掩码转化为真实显微镜图像，并通过对象级一致性进行正则化；（3）伪掩码和合成图像共同构成了训练集。

    Current deep learning-based approaches for the segmentation of microscopy images heavily rely on large amount of training data with dense annotation, which is highly costly and laborious in practice. Compared to full annotation where the complete contour of objects is depicted, point annotations, specifically object centroids, are much easier to acquire and still provide crucial information about the objects for subsequent segmentation. In this paper, we assume access to point annotations only during training and develop a unified pipeline for microscopy image segmentation using synthetically generated training data. Our framework includes three stages: (1) it takes point annotations and samples a pseudo dense segmentation mask constrained with shape priors; (2) with an image generative model trained in an unpaired manner, it translates the mask to a realistic microscopy image regularized by object level consistency; (3) the pseudo masks along with the synthetic images then constitute 
    
[^15]: 关于最先进生成模型的可信度景观：一项综合调查

    On the Trustworthiness Landscape of State-of-the-art Generative Models: A Comprehensive Survey. (arXiv:2307.16680v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.16680](http://arxiv.org/abs/2307.16680)

    本文综合调查了大规模生成模型的可信度问题，涵盖了隐私、安全、公平性和责任等多个维度，并提出了实际建议和未来发展方向。

    

    扩散模型和大规模语言模型已经成为领先的生成模型，并对人类生活的各个方面产生了革命性的影响。然而，这些模型的实际应用也暴露出固有的风险，突显了它们的双重性质，并引发了对它们可信度的担忧。尽管有大量关于这个主题的文献，但针对大规模生成模型及其可信度的综合调查仍然很少见。为了弥补这一空白，本文调查了涉及这些模型的长期和新兴威胁，涵盖了隐私、安全、公平和责任这四个基本维度。通过这种方式，我们构建了一张详尽的地图，概述了这些模型的可信度，并提供了实际建议和未来的发展方向。这些努力对于促进这些模型的可信度部署至关重要。

    Diffusion models and large language models have emerged as leading-edge generative models and have sparked a revolutionary impact on various aspects of human life. However, the practical implementation of these models has also exposed inherent risks, highlighting their dual nature and raising concerns regarding their trustworthiness. Despite the abundance of literature on this subject, a comprehensive survey specifically delving into the intersection of large-scale generative models and their trustworthiness remains largely absent. To bridge this gap, This paper investigates both the long-standing and emerging threats associated with these models across four fundamental dimensions: privacy, security, fairness, and responsibility. In this way, we construct an extensive map outlining the trustworthiness of these models, while also providing practical recommendations and identifying future directions. These efforts are crucial for promoting the trustworthy deployment of these models, ulti
    
[^16]: BayesDAG：基于梯度的因果发现的后验采样

    BayesDAG: Gradient-Based Posterior Sampling for Causal Discovery. (arXiv:2307.13917v1 [cs.LG])

    [http://arxiv.org/abs/2307.13917](http://arxiv.org/abs/2307.13917)

    这项研究引入了一种基于梯度的后验采样方法，用于解决Bayesian causal discovery中的计算挑战，能够高效地推断因果模型，并且不依赖于DAG正则化。

    

    贝叶斯因果发现旨在从观测数据中推断出因果模型的后验分布，量化认知不确定性，从而有助于下游任务。然而，由于有向无环图（DAG）和非线性函数的组合空间的联合推理而带来了计算挑战。尽管近年来在DAG上的高效后验推断方面取得了进展，但现有方法要么仅限于对线性因果模型的节点排列矩阵进行变分推断，导致推断准确性受损，要么是在受DAG正则化约束的邻接矩阵上进行连续松弛，而不能确保得到的图是DAGs。在这项工作中，我们介绍了一种基于随机梯度马尔科夫链蒙特卡罗（SG-MCMC）的可扩展贝叶斯因果发现框架，克服了这些局限性。我们的方法直接从后验中采样DAG，并且不需要任何DAG正则化，同时还绘制函数参数样本和…

    Bayesian causal discovery aims to infer the posterior distribution over causal models from observed data, quantifying epistemic uncertainty and benefiting downstream tasks. However, computational challenges arise due to joint inference over combinatorial space of Directed Acyclic Graphs (DAGs) and nonlinear functions. Despite recent progress towards efficient posterior inference over DAGs, existing methods are either limited to variational inference on node permutation matrices for linear causal models, leading to compromised inference accuracy, or continuous relaxation of adjacency matrices constrained by a DAG regularizer, which cannot ensure resulting graphs are DAGs. In this work, we introduce a scalable Bayesian causal discovery framework based on stochastic gradient Markov Chain Monte Carlo (SG-MCMC) that overcomes these limitations. Our approach directly samples DAGs from the posterior without requiring any DAG regularization, simultaneously draws function parameter samples and 
    
[^17]: 基于梯度的随机点积图谱嵌入

    Gradient-Based Spectral Embeddings of Random Dot Product Graphs. (arXiv:2307.13818v1 [cs.LG])

    [http://arxiv.org/abs/2307.13818](http://arxiv.org/abs/2307.13818)

    本文介绍了基于梯度的随机点积图谱嵌入方法，并通过利用非凸优化技术改进了在观察图中估计节点潜在向量的任务。同时，作者还提出了一阶梯度下降方法来更好地解决嵌入问题，并适应更广泛的实用网络嵌入应用。

    

    随机点积图谱（RDPG）是一个关系数据的生成模型，其中节点通过在低维欧氏空间中的潜在向量表示。RDPG关键地假设边的形成概率由相应的潜在位置的点积给出。因此，从观察到的图中估计这些向量的嵌入任务通常被设定为一个低秩矩阵分解问题。经典的邻接谱嵌入（ASE）具有可靠的统计性质，但它在形式上解决的是一个代理问题，并且计算复杂度较高。在本文中，我们利用非凸优化的最新进展，并展示它们对RDPG推断的影响。我们提倡使用一阶梯度下降方法来更好地解决嵌入问题，并自然地适应更广泛的实用网络嵌入应用。值得注意的是，我们认为RDPG嵌入有向图失去了可解释性，除非...

    The Random Dot Product Graph (RDPG) is a generative model for relational data, where nodes are represented via latent vectors in low-dimensional Euclidean space. RDPGs crucially postulate that edge formation probabilities are given by the dot product of the corresponding latent positions. Accordingly, the embedding task of estimating these vectors from an observed graph is typically posed as a low-rank matrix factorization problem. The workhorse Adjacency Spectral Embedding (ASE) enjoys solid statistical properties, but it is formally solving a surrogate problem and can be computationally intensive. In this paper, we bring to bear recent advances in non-convex optimization and demonstrate their impact to RDPG inference. We advocate first-order gradient descent methods to better solve the embedding problem, and to organically accommodate broader network embedding applications of practical relevance. Notably, we argue that RDPG embeddings of directed graphs loose interpretability unless 
    
[^18]: ECSIC: 用于立体图像压缩的极线交叉注意力技术

    ECSIC: Epipolar Cross Attention for Stereo Image Compression. (arXiv:2307.10284v1 [eess.IV])

    [http://arxiv.org/abs/2307.10284](http://arxiv.org/abs/2307.10284)

    ECSIC是一种用于立体图像压缩的新方法，通过利用左右图像之间的相互信息进行联合压缩，并使用新颖的立体交叉注意力模块和立体上下文模块实现。与现有方法相比，ECSIC在两个流行的立体图像数据集上取得了最先进的性能，并且具有快速编码和解码的特性。

    

    在本文中，我们提出了一种新颖的学习方法ECSIC，用于立体图像压缩。我们的方法通过利用立体图像对左右图像之间的相互信息，采用一种新颖的立体交叉注意力（SCA）模块和两个立体上下文模块，以联合方式压缩左右图像。SCA模块在两个图像的对应极线范围内进行交叉注意力处理，并且并行处理它们。立体上下文模块通过使用第一个图像作为上下文来改善对第二个编码图像的熵估计。我们进行了大量的剔除实验，证明了所提出模块的有效性，并与现有方法进行了全面的定量和定性比较。ECSIC在两个常用的立体图像数据集Cityscapes和InStereo2k上达到了立体图像压缩模型中的最先进性能，同时允许快速编码和解码，非常适用于实时应用。

    In this paper, we present ECSIC, a novel learned method for stereo image compression. Our proposed method compresses the left and right images in a joint manner by exploiting the mutual information between the images of the stereo image pair using a novel stereo cross attention (SCA) module and two stereo context modules. The SCA module performs cross-attention restricted to the corresponding epipolar lines of the two images and processes them in parallel. The stereo context modules improve the entropy estimation of the second encoded image by using the first image as a context. We conduct an extensive ablation study demonstrating the effectiveness of the proposed modules and a comprehensive quantitative and qualitative comparison with existing methods. ECSIC achieves state-of-the-art performance among stereo image compression models on the two popular stereo image datasets Cityscapes and InStereo2k while allowing for fast encoding and decoding, making it highly practical for real-time
    
[^19]: FreeDrag: 点追踪并不适用于交互式的基于点的图像编辑

    FreeDrag: Point Tracking is Not What You Need for Interactive Point-based Image Editing. (arXiv:2307.04684v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2307.04684](http://arxiv.org/abs/2307.04684)

    FreeDrag提出了一种基于特征的方法来解决DragGAN在点追踪方面的困难，通过自适应模板特征、线性搜索和模糊定位技术，实现了稳定和高效的基于点的图像编辑。

    

    为了满足图像编辑的复杂和多样化需求，对图像内容的精确和灵活的操纵是不可或缺的。最近，DragGAN通过基于点的操纵实现了令人印象深刻的编辑结果。然而，我们观察到DragGAN在点的追踪上存在困难，包括错误追踪和模糊追踪。为了解决这些问题，我们提出了FreeDrag，它采用了基于特征的方法来减轻DragGAN中点追踪的负担。FreeDrag结合了自适应模板特征、线性搜索和模糊定位技术，实现了稳定和高效的基于点的图像编辑。大量实验表明，我们的方法优于DragGAN，并能在具有相似特征的困难情景下实现稳定的基于点的编辑。

    To serve the intricate and varied demands of image editing, precise and flexible manipulation of image content is indispensable. Recently, DragGAN has achieved impressive editing results through point-based manipulation. However, we have observed that DragGAN struggles with miss tracking, where DragGAN encounters difficulty in effectively tracking the desired handle points, and ambiguous tracking, where the tracked points are situated within other regions that bear resemblance to the handle points. To deal with the above issues, we propose FreeDrag, which adopts a feature-oriented approach to free the burden on point tracking within the point-oriented methodology of DragGAN. The FreeDrag incorporates adaptive template features, line search, and fuzzy localization techniques to perform stable and efficient point-based image editing. Extensive experiments demonstrate that our method is superior to the DragGAN and enables stable point-based editing in challenging scenarios with similar st
    
[^20]: 注意力机制中的边缘最大化

    Margin Maximization in Attention Mechanism. (arXiv:2306.13596v1 [cs.LG])

    [http://arxiv.org/abs/2306.13596](http://arxiv.org/abs/2306.13596)

    这篇论文证明了，在softmax-attention模型中，通过在p或等价的W上运行梯度下降，可以收敛到一个最大边缘解，这将局部最优的标记与非最优的标记分隔开。这明确地将注意力机制形式化为标记分离机制。

    

    注意力机制是Transformer架构的核心组件，也是大型语言模型取得惊人成功的原因之一。然而，注意力机制背后的理论原则尚不清楚，特别是它的非凸优化动力学。本文探讨了开创性的softmax-attention模型$f(\boldsymbol{X})=\langle \boldsymbol{Xv}, \texttt{softmax}(\boldsymbol{XWp})\rangle$，其中$\boldsymbol{X}$是标记序列，$(\boldsymbol{v},\boldsymbol{W},\boldsymbol{p})$是可调参数。我们证明了在$\boldsymbol{p}$或等价的$\boldsymbol{W}$上运行梯度下降会沿着方向收敛到分隔“局部最优”标记和“非最优”标记的最大边缘解。这明确地形式化了注意力作为一种标记分离机制。值得注意的是，我们的结果适用于一般数据，并使用嵌入$\boldsymbol{Xv}$和$\texttt{softmax}(\boldsymbol{XWp})$精细地表征标记的“最优性”。

    Attention mechanism is a central component of the transformer architecture which led to the phenomenal success of large language models. However, the theoretical principles underlying the attention mechanism are poorly understood, especially its nonconvex optimization dynamics. In this work, we explore the seminal softmax-attention model $f(\boldsymbol{X})=\langle \boldsymbol{Xv}, \texttt{softmax}(\boldsymbol{XWp})\rangle$, where, $\boldsymbol{X}$ is the token sequence and $(\boldsymbol{v},\boldsymbol{W},\boldsymbol{p})$ are tunable parameters. We prove that running gradient descent on $\boldsymbol{p}$, or equivalently $\boldsymbol{W}$, converges in direction to a max-margin solution that separates $\textit{locally-optimal}$ tokens from non-optimal ones. This clearly formalizes attention as a token separation mechanism. Remarkably, our results are applicable to general data and precisely characterize $\textit{optimality}$ of tokens in terms of the value embeddings $\boldsymbol{Xv}$ and
    
[^21]: 因果归一化流：从理论到实践

    Causal normalizing flows: from theory to practice. (arXiv:2306.05415v1 [cs.LG])

    [http://arxiv.org/abs/2306.05415](http://arxiv.org/abs/2306.05415)

    本文研究了使用因果归一化流进行因果推论的方法，证明了在给定因果排序情况下，利用自回归归一化流可以恢复因果模型。通过实验和比较研究，证明了因果归一化流可用于解决实际问题。

    

    本文深入探讨了利用归一化流进行因果推论的应用。具体来说，我们首先利用非线性ICA的最新结果，显示出在给定因果排序的情况下，因果模型可以从观测数据中鉴别出来，并且可以使用自回归归一化流进行恢复。其次，我们分析了用于捕捉潜在因果数据生成过程的不同设计和学习选择的因果归一化流。第三，我们描述了如何在因果归一化流中实现do-operator，从而回答干预和反事实问题。最后，在我们的实验中，我们通过综合对比研究验证了我们的设计和训练选择；将因果归一化流与其他逼近因果模型的方法进行比较；并通过实证研究证明因果归一化流可用于解决现实世界中存在混合离散连续数据和因果图部分知识的问题。本文的代码可以进行访问。

    In this work, we deepen on the use of normalizing flows for causal reasoning. Specifically, we first leverage recent results on non-linear ICA to show that causal models are identifiable from observational data given a causal ordering, and thus can be recovered using autoregressive normalizing flows (NFs). Second, we analyze different design and learning choices for causal normalizing flows to capture the underlying causal data-generating process. Third, we describe how to implement the do-operator in causal NFs, and thus, how to answer interventional and counterfactual questions. Finally, in our experiments, we validate our design and training choices through a comprehensive ablation study; compare causal NFs to other approaches for approximating causal models; and empirically demonstrate that causal NFs can be used to address real-world problems, where the presence of mixed discrete-continuous data and partial knowledge on the causal graph is the norm. The code for this work can be f
    
[^22]: G$^2$uardFL: 通过属性化客户端图聚类来防御后门攻击的联邦学习保护框架

    G$^2$uardFL: Safeguarding Federated Learning Against Backdoor Attacks through Attributed Client Graph Clustering. (arXiv:2306.04984v1 [cs.CR])

    [http://arxiv.org/abs/2306.04984](http://arxiv.org/abs/2306.04984)

    本论文提出了G$^2$uardFL，这是一个基于属性化客户端图聚类的联邦学习保护框架，能够有效识别恶意客户端，即使恶意客户端数量高达50％。

    

    作为协同范式，联邦学习（FL）使客户端能够进行集体模型训练而不交换各自的本地数据。然而，FL仍然容易受到后门攻击的影响，攻击者会通过篡改模型权重注入有毒数据，从而得到针对特定样本的攻击者选择的预测结果。现有的对策主要基于异常检测，但由于量化客户模型相似性的不足，这些对策可能会错误地拒绝合法权重，同时接受恶意权重。其他防御机制仅在面对少量恶意客户端，例如少于10％的恶意客户端时才有效。为了解决这些漏洞，我们提出了G$^2$uardFL，这是一个保护框架，它将检测恶意客户端视为一个属性图聚类问题，从而保护FL系统。该框架采用客户端图聚类技术，根据模型权重的相似性将客户端分类为正常或恶意。通过采用对客户端固有属性进行编码的属性标签，G$^2$uardFL在识别受损客户端方面优于现有的防御机制，而不排除合法客户端。实验结果表明，即使有50％的客户端是恶意的，G$^2$uardFL也能显著降低后门攻击成功率。

    As a collaborative paradigm, Federated Learning (FL) empowers clients to engage in collective model training without exchanging their respective local data. Nevertheless, FL remains vulnerable to backdoor attacks in which an attacker compromises malicious clients, and injects poisoned model weights into the aggregation process to yield attacker-chosen predictions for particular samples. Existing countermeasures, mainly based on anomaly detection, may erroneously reject legitimate weights while accepting malicious ones, which is due to inadequacies in quantifying client model similarities. Other defense mechanisms prove effective exclusively when confronted with a restricted number of malicious clients, e.g., less than 10%. To address these vulnerabilities, we present G$^2$uardFL, a protective framework that reframes the detection of malicious clients as an attributed graph clustering problem, thereby safeguarding FL systems. This framework employs a client graph clustering technique to
    
[^23]: 利用预测时间特征相似性的物体中心学习实现对真实世界视频的分析

    Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities. (arXiv:2306.04829v1 [cs.CV])

    [http://arxiv.org/abs/2306.04829](http://arxiv.org/abs/2306.04829)

    本研究提出了一种新方法，利用预训练的自监督特征和时间特征相似性损失，实现了对真实世界视频的物体中心学习，在合成MOVi数据集上取得了最先进的性能。同时，本模型是首个能够扩展到无约束视频数据集的物体中心视频模型。

    

    无监督的基于视频的物体中心学习是从大规模无标签视频集合中学习结构化表示的有前途的途径。然而，以前的方法只能在受限领域内缩放到真实世界的数据集。最近的研究表明，预训练的自监督特征的重建会导致在不受约束的真实世界图像数据集上的物体中心表示。基于这种方法，我们提出了一种利用这些预训练特征的新方法，形式为时间特征相似性损失。该损失编码图像块之间的时间相关性，并自然地引入运动偏差来发现物体。我们证明，这种损失导致了在具有挑战性的合成MOVi数据集上的最先进性能。当与特征重建损失结合使用时，我们的模型是首个能够扩展到无约束视频数据集（如YouTube-VIS）的物体中心视频模型。

    Unsupervised video-based object-centric learning is a promising avenue to learn structured representations from large, unlabeled video collections, but previous approaches have only managed to scale to real-world datasets in restricted domains. Recently, it was shown that the reconstruction of pre-trained self-supervised features leads to object-centric representations on unconstrained real-world image datasets. Building on this approach, we propose a novel way to use such pre-trained features in the form of a temporal feature similarity loss. This loss encodes temporal correlations between image patches and is a natural way to introduce a motion bias for object discovery. We demonstrate that this loss leads to state-of-the-art performance on the challenging synthetic MOVi datasets. When used in combination with the feature reconstruction loss, our model is the first object-centric video model that scales to unconstrained video datasets such as YouTube-VIS.
    
[^24]: 经典循环神经网络的脉冲计算

    Spike-based computation using classical recurrent neural networks. (arXiv:2306.03623v1 [cs.NE])

    [http://arxiv.org/abs/2306.03623](http://arxiv.org/abs/2306.03623)

    本文提出了一种新的脉冲神经网络方法，通过修改一种易于训练的循环神经网络的动态特性，使其产生基于脉冲的计算，并在进行了脉冲网络的训练后，在多个数据集上取得了最先进的性能。

    

    脉冲神经网络是一种人工神经网络，其中神经元之间的通信仅由事件或所谓的脉冲组成。这种特性使得神经网络能够进行异步和稀疏计算，并因此在专用硬件上运行时大幅减少能源消耗。本文中，我们尝试采用一种对称的方法：修改一种已知的、易于训练的循环神经网络的动态特性，使其产生基于脉冲的计算。通过明确引入脉冲阈值和重置机制，我们使网络能够仅使用脉冲来执行前向和循环计算。然后，我们展示了这种修改后的构架既可以实现，同时在两个基准数据集上实现了最先进的性能，包括具有挑战性的ImageNet数据集。

    Spiking neural networks are a type of artificial neural networks in which communication between neurons is only made of events, also called spikes. This property allows neural networks to make asynchronous and sparse computations and therefore to drastically decrease energy consumption when run on specialized hardware. However, training such networks is known to be difficult, mainly due to the non-differentiability of the spike activation, which prevents the use of classical backpropagation. This is because state-of-the-art spiking neural networks are usually derived from biologically-inspired neuron models, to which are applied machine learning methods for training. Nowadays, research about spiking neural networks focuses on the design of training algorithms whose goal is to obtain networks that compete with their non-spiking version on specific tasks. In this paper, we attempt the symmetrical approach: we modify the dynamics of a well-known, easily trainable type of recurrent neural 
    
[^25]: 循环一致性驱动的物体发现方法

    Cycle Consistency Driven Object Discovery. (arXiv:2306.02204v1 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2306.02204](http://arxiv.org/abs/2306.02204)

    该方法通过循环一致性目标的引入，明确优化场景中每个物体应映射到不同槽位的约束，从而实现了在完全无监督的情况下有效地学习发现物体。在实验中表现出了优于现有方法的性能。

    

    开发能够有效学习类似于人类认知的以物体为中心的表示的深度学习模型仍然是一项具有挑战性的任务。现有的方法利用架构先验或辅助信息（例如深度图或流场图）来探索基于槽位的方法，以表示对象为称为“槽位”或“对象文件”的固定大小的向量，从而促进物体发现。 然而，依赖于架构先验会引入不可靠性，并需要精心设计才能识别正确的对象。 同样，依赖辅助信息的方法也不够优越，因为这种信息通常在大多数自然情况下不可用。为了解决这些限制，我们提出了一种明确优化场景中每个对象应映射到一个不同槽位的方法。我们通过引入循环一致性目标来形式化这个约束，称之为循环一致性目标。通过应用这些限制，我们的方法可以在完全无监督的情况下有效地学习发现物体。 在实验中，我们展示了我们的方法在无监督物体发现和少样本物体分类基准测试中均优于现有的最先进方法。

    Developing deep learning models that effectively learn object-centric representations, akin to human cognition, remains a challenging task. Existing approaches have explored slot-based methods utilizing architectural priors or auxiliary information such as depth maps or flow maps to facilitate object discovery by representing objects as fixed-size vectors, called ``slots'' or ``object files''. However, reliance on architectural priors introduces unreliability and requires meticulous engineering to identify the correct objects. Likewise, methods relying on auxiliary information are suboptimal as such information is often unavailable for most natural scenes. To address these limitations, we propose a method that explicitly optimizes the constraint that each object in a scene should be mapped to a distinct slot. We formalize this constraint by introducing consistency objectives which are cyclic in nature. We refer to them as the \textit{cycle-consistency} objectives. By applying these con
    
[^26]: 截断亲和力最大化：用于图形异常监测的单类同型建模

    Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection. (arXiv:2306.00006v1 [cs.SI])

    [http://arxiv.org/abs/2306.00006](http://arxiv.org/abs/2306.00006)

    本文针对图形异常监测数据集中存在的一类同型现象，提出了一种新的无监督异常评分度量——当前节点亲和力，并通过学习量身定制的节点表示，实现了截断亲和力最大化（TAM）方法，优化在原始图形结构上进行，能够有效进行双重One-Class的GAD。

    

    我们在现实世界的图形异常监测（GAD）数据集中经常发现一种普遍的属性......本文提出了一种新的无监督异常评分度量 - 当前节点亲和力......我们进一步提出了截断亲和力最大化 (TAM)，该方法通过最大化与_neighbors的本地亲和力来学习量身定制的节点表示。本文所提方法在原始图形结构上进行优化，可以进行双重One-Class的GAD。

    One prevalent property we find empirically in real-world graph anomaly detection (GAD) datasets is a one-class homophily, i.e., normal nodes tend to have strong connection/affinity with each other, while the homophily in abnormal nodes is significantly weaker than normal nodes. However, this anomaly-discriminative property is ignored by existing GAD methods that are typically built using a conventional anomaly detection objective, such as data reconstruction. In this work, we explore this property to introduce a novel unsupervised anomaly scoring measure for GAD -- local node affinity -- that assigns a larger anomaly score to nodes that are less affiliated with their neighbors, with the affinity defined as similarity on node attributes/representations. We further propose Truncated Affinity Maximization (TAM) that learns tailored node representations for our anomaly measure by maximizing the local affinity of nodes to their neighbors. Optimizing on the original graph structure can be bi
    
[^27]: J-UNIWARD中的一个实现错误

    Off-By-One Implementation Error in J-UNIWARD. (arXiv:2305.19776v1 [cs.CR])

    [http://arxiv.org/abs/2305.19776](http://arxiv.org/abs/2305.19776)

    J-UNIWARD 是一种将秘密信息隐藏在JPEG图像中的隐写方法，本文发现了其实现中存在的一个 off-by-one 错误，使一些图像块被高估，另一些被低估，同时提供了一个概念验证用于检测此种错误。

    

    J-UNIWARD是一种将秘密信息隐藏在JPEG盖板图像中的流行隐写术方法。作为一种内容自适应方法，J-UNIWARD旨在嵌入到纹理图像区域，这些区域的变化难以检测。为此，J-UNIWARD首先为每个DCT系数分配一个嵌入成本，该成本基于图像的小波残差计算，然后使用一种编码方法，该方法在嵌入所需的有效载荷的同时，最小化成本。更改一个DCT系数会影响23x23个小波系数窗口。为了加速成本图的计算，原始实现预先计算小波残差，然后对于每个更改的DCT系数，考虑一个23x23的小波残差窗口。然而，该实现错误地将窗口偏移了一个像素到右下方。在这份报告中，我们评估了这个off-by-one错误对生成的成本图的影响。一些图像块被高估，而其他图像块则被低估，但差异相对较小。我们提供了一个概念验证，说明如何检测使用带有偏移错误的J-UNIWARD隐藏的隐写术信息。

    J-UNIWARD is a popular steganography method for hiding secret messages in JPEG cover images. As a content-adaptive method, J-UNIWARD aims to embed into textured image regions where changes are difficult to detect. To this end, J-UNIWARD first assigns to each DCT coefficient an embedding cost calculated based on the image's Wavelet residual, and then uses a coding method that minimizes the cost while embedding the desired payload. Changing one DCT coefficient affects a 23x23 window of Wavelet coefficients. To speed up the costmap computation, the original implementation pre-computes the Wavelet residual and then considers per changed DCT coefficient a 23x23 window of the Wavelet residual. However, the implementation accesses a window accidentally shifted by one pixel to the bottom right. In this report, we evaluate the effect of this off-by-one error on the resulting costmaps. Some image blocks are over-priced while other image blocks are under-priced, but the difference is relatively s
    
[^28]: 优化适当的损失函数是否能得到校准的预测器？

    When Does Optimizing a Proper Loss Yield Calibration?. (arXiv:2305.18764v1 [cs.LG])

    [http://arxiv.org/abs/2305.18764](http://arxiv.org/abs/2305.18764)

    研究优化适当的损失函数是否能在受限的预测器族中得到校准的模型，使用局部最优条件取代全局最优性条件并在此基础上进行了严格的证明。

    

    优化适当的损失函数被广泛认为会得到具有良好校准特性的预测器，这是因为对于这样的损失，全局最优解是预测真实概率，这确实是校准的。但是，典型的机器学习模型是训练来近似地最小化在受限制的预测器族中的损失，这些预测器族不太可能包含真实的概率。在什么情况下，优化受限制的预测器族中适当的损失可以得到校准的模型？它提供了什么精确的校准保证？在这项工作中，我们提供了这些问题的严格答案。我们用局部最优条件替换全局最优性条件，该条件规定了预测器（适当的）损失不能通过使用一定族群的Lipschitz函数后处理其预测而降低太多。我们证明了具有这种局部最优性质的任何预测器都满足Kakade-Foster(2008)、Błasiok等人(2023)中定义的平稳校准。

    Optimizing proper loss functions is popularly believed to yield predictors with good calibration properties; the intuition being that for such losses, the global optimum is to predict the ground-truth probabilities, which is indeed calibrated. However, typical machine learning models are trained to approximately minimize loss over restricted families of predictors, that are unlikely to contain the ground truth. Under what circumstances does optimizing proper loss over a restricted family yield calibrated models? What precise calibration guarantees does it give? In this work, we provide a rigorous answer to these questions. We replace the global optimality with a local optimality condition stipulating that the (proper) loss of the predictor cannot be reduced much by post-processing its predictions with a certain family of Lipschitz functions. We show that any predictor with this local optimality satisfies smooth calibration as defined in Kakade-Foster (2008), B{\l}asiok et al. (2023). L
    
[^29]: 变分分类

    Variational Classification. (arXiv:2305.10406v1 [cs.LG])

    [http://arxiv.org/abs/2305.10406](http://arxiv.org/abs/2305.10406)

    提出一种新的变分分类方法，通过引入潜变量建模来优化训练，允许灵活的设计选择以改善校准和对抗鲁棒性，实验结果表明其对于域外数据的分类准确性得到了保持。

    

    我们提出了一种传统神经网络方法的新型扩展，称为变分分类 (VC)。通过引入潜变量建模，类似于变分自编码器和传统自编码器之间的关系，我们得到了一个基于证据下界 (ELBO) 的训练目标，采用对抗性方法优化。我们的VC模型允许在设计选择方面更加灵活，特别是类条件潜先验，而不是在现成的softmax分类器中做出的隐式假设。在图像和文本分类数据集上的实证评估表明，我们的方法在保持预测准确性的同时，改善了其他良好特性，如校准和对抗鲁棒性，即使应用于域外数据。

    We present a novel extension of the traditional neural network approach to classification tasks, referred to as variational classification (VC). By incorporating latent variable modeling, akin to the relationship between variational autoencoders and traditional autoencoders, we derive a training objective based on the evidence lower bound (ELBO), optimized using an adversarial approach. Our VC model allows for more flexibility in design choices, in particular class-conditional latent priors, in place of the implicit assumptions made in off-the-shelf softmax classifiers. Empirical evaluation on image and text classification datasets demonstrates the effectiveness of our approach in terms of maintaining prediction accuracy while improving other desirable properties such as calibration and adversarial robustness, even when applied to out-of-domain data.
    
[^30]: 大型神经网络的多校准可最小化损失

    Loss minimization yields multicalibration for large neural networks. (arXiv:2304.09424v1 [cs.LG])

    [http://arxiv.org/abs/2304.09424](http://arxiv.org/abs/2304.09424)

    本文展示了对于大型神经网络大小，最优地最小化损失会导致多校准，以提供公平的预测结果。

    

    多校准是一种公平性概念，旨在提供跨大量团体的准确预测。即使对于简单的预测器，如线性函数，多校准也被认为是与最小化损失不同的目标。在本文中，我们展示了对于（几乎所有的）大型神经网络大小，最优地最小化平方误差会导致多校准。我们的结果关于神经网络的表征方面，而不是关于算法或样本复杂性考虑。以前的这样的结果仅适用于几乎贝叶斯最优的预测器，因此是表征无关的。我们强调，我们的结果不适用于优化神经网络的特定算法，如 SGD，并且不应解释为“公平性从优化神经网络中获得免费的好处”。

    Multicalibration is a notion of fairness that aims to provide accurate predictions across a large set of groups. Multicalibration is known to be a different goal than loss minimization, even for simple predictors such as linear functions. In this note, we show that for (almost all) large neural network sizes, optimally minimizing squared error leads to multicalibration. Our results are about representational aspects of neural networks, and not about algorithmic or sample complexity considerations. Previous such results were known only for predictors that were nearly Bayes-optimal and were therefore representation independent. We emphasize that our results do not apply to specific algorithms for optimizing neural networks, such as SGD, and they should not be interpreted as "fairness comes for free from optimizing neural networks".
    
[^31]: 针对耦合偏微分方程的耦合多小波神经算子学习

    Coupled Multiwavelet Neural Operator Learning for Coupled Partial Differential Equations. (arXiv:2303.02304v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.02304](http://arxiv.org/abs/2303.02304)

    本论文提出一种耦合多小波神经算子学习的方案，解决了处理耦合多变量映射问题的难点，能够显著提高解决耦合偏微分方程的准确性，并在实验中得到了验证。

    

    耦合偏微分方程是描述许多物理过程复杂动态的关键任务。最近，神经算子已经展示出通过在傅里叶/小波空间直接学习积分核来解决PDE的能力。对于耦合PDE的解决方法，难点在于处理函数之间的耦合映射。为此，我们提出了一种耦合多小波神经算子（CMWNO）学习方案，通过在小波空间中进行多小波分解和重构过程中解耦合积分核。在解决Gray-Scott（GS）方程和非局部均场博弈（MFG）问题等耦合PDE方面，所提出的模型相对于先前基于学习的求解器实现了显著提高的准确性。根据我们的实验结果，所提出的模型相对于最先进模型的$L^2$误差表现出了$2\times \sim 4\times$的改进。

    Coupled partial differential equations (PDEs) are key tasks in modeling the complex dynamics of many physical processes. Recently, neural operators have shown the ability to solve PDEs by learning the integral kernel directly in Fourier/Wavelet space, so the difficulty for solving the coupled PDEs depends on dealing with the coupled mappings between the functions. Towards this end, we propose a \textit{coupled multiwavelets neural operator} (CMWNO) learning scheme by decoupling the coupled integral kernels during the multiwavelet decomposition and reconstruction procedures in the Wavelet space. The proposed model achieves significantly higher accuracy compared to previous learning-based solvers in solving the coupled PDEs including Gray-Scott (GS) equations and the non-local mean field game (MFG) problem. According to our experimental results, the proposed model exhibits a $2\times \sim 4\times$ improvement relative $L$2 error compared to the best results from the state-of-the-art mode
    
[^32]: 修复动态神经网络中的过度自信问题

    Fixing Overconfidence in Dynamic Neural Networks. (arXiv:2302.06359v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.06359](http://arxiv.org/abs/2302.06359)

    该论文提出了一种修复动态神经网络中过度自信问题的方法，通过对最后几层进行概率化处理，量化和纳入不确定性并有助于决定计算预算的确定。

    

    动态神经网络是一种最近的技术，通过根据输入难度动态调整计算代价，承诺缓解现代深度学习模型越来越大的问题。然而，深度学习模型中uncertainty estimates的质量较差，很难区分hard和easy的样本。为了解决这个挑战，我们提出了一种在动态神经网络中进行后处理不确定性量化的计算有效方法。我们展示了通过对最后几层进行概率化处理，充分量化和纳入aleatoric和epistemic uncertainty，可以提高预测性能，并在确定计算预算时有助于决策。在实验中，我们在CIFAR-100、ImageNet和Caltech-256方面展示了准确性、捕获不确定性和校准误差的改进。

    Dynamic neural networks are a recent technique that promises a remedy for the increasing size of modern deep learning models by dynamically adapting their computational cost to the difficulty of the inputs. In this way, the model can adjust to a limited computational budget. However, the poor quality of uncertainty estimates in deep learning models makes it difficult to distinguish between hard and easy samples. To address this challenge, we present a computationally efficient approach for post-hoc uncertainty quantification in dynamic neural networks. We show that adequately quantifying and accounting for both aleatoric and epistemic uncertainty through a probabilistic treatment of the last layers improves the predictive performance and aids decision-making when determining the computational budget. In the experiments, we show improvements on CIFAR-100, ImageNet, and Caltech-256 in terms of accuracy, capturing uncertainty, and calibration error.
    
[^33]: 基于无上下文文法的分层神经架构搜索空间构建

    Construction of Hierarchical Neural Architecture Search Spaces based on Context-free Grammars. (arXiv:2211.01842v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.01842](http://arxiv.org/abs/2211.01842)

    本研究基于无上下文文法提出了一个统一的搜索空间设计框架，可以生成表达力强大的分层搜索空间，实现了对整个体系结构的搜索并促进结构的规律性。

    

    从简单的构建块中发现神经结构是神经架构搜索(NAS)的一个长期目标。分层搜索空间是实现这一目标的一个有前途的步骤，但缺乏统一的搜索空间设计框架，并且通常仅搜索一些限定方面的架构。在本研究中，我们介绍了一个基于无上下文文法的统一搜索空间设计框架，它可以自然而紧凑地生成表达力强大的分层搜索空间，比文献中常见的空间大几个数量级。通过增强和利用它们的属性，我们有效地实现了对整个体系结构的搜索，并促进了结构的规律性。此外，我们提出了一种高效的分层核设计用于贝叶斯优化搜索策略，以高效搜索如此庞大的空间。我们展示了我们搜索空间设计框架的多样性，并表明我们的搜索策略可以优于现有的NAS方法。

    The discovery of neural architectures from simple building blocks is a long-standing goal of Neural Architecture Search (NAS). Hierarchical search spaces are a promising step towards this goal but lack a unifying search space design framework and typically only search over some limited aspect of architectures. In this work, we introduce a unifying search space design framework based on context-free grammars that can naturally and compactly generate expressive hierarchical search spaces that are 100s of orders of magnitude larger than common spaces from the literature. By enhancing and using their properties, we effectively enable search over the complete architecture and can foster regularity. Further, we propose an efficient hierarchical kernel design for a Bayesian Optimization search strategy to efficiently search over such huge spaces. We demonstrate the versatility of our search space design framework and show that our search strategy can be superior to existing NAS approaches. Co
    
[^34]: 神经特征向量是结构化表示学习器

    Neural Eigenfunctions Are Structured Representation Learners. (arXiv:2210.12637v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.12637](http://arxiv.org/abs/2210.12637)

    本文提出了一种称为神经特征映射的结构化自适应深度表示方法，它通过神经网络对特征值函数进行参数化建模。应用神经特征映射可以得到类似于流行的自监督学习方法的目标函数，并具有打破对称性的属性，从而产生结构化表示，其中特征按重要性进行排序。在图像检索系统中，通过根据特征的重要性进行截断，我们的方法所需的表示长度比领先的自监督学习方法短16倍，同时具有相似的检索性能。

    

    本文介绍了一种称为神经特征映射的结构化自适应深度表示。与先前的谱方法（如拉普拉斯特征映射）以非参数化方式进行操作不同，神经特征映射利用神经网络对特征值函数进行参数化建模。我们展示了当特征值函数来自于数据扩增设置中的正相关关系时，应用神经特征映射会产生类似于流行的自监督学习方法的目标函数，同时还具有打破对称性的属性，从而导致结构化表示，其中特征按重要性进行排序。我们在图像检索系统中演示了使用这样的自适应长度编码来表示。通过根据特征的重要性进行截断，我们的方法所需的表示长度比领先的自监督学习方法短16倍，同时达到相似的检索性能。我们进一步将我们的方法应用于图形数据，并报告了强大的结果。

    This paper introduces a structured, adaptive-length deep representation called Neural Eigenmap. Unlike prior spectral methods such as Laplacian Eigenmap that operate in a nonparametric manner, Neural Eigenmap leverages NeuralEF to parametrically model eigenfunctions using a neural network. We show that, when the eigenfunction is derived from positive relations in a data augmentation setup, applying NeuralEF results in an objective function that resembles those of popular self-supervised learning methods, with an additional symmetry-breaking property that leads to structured representations where features are ordered by importance. We demonstrate using such representations as adaptive-length codes in image retrieval systems. By truncation according to feature importance, our method requires up to $16\times$ shorter representation length than leading self-supervised learning ones to achieve similar retrieval performance. We further apply our method to graph data and report strong results
    
[^35]: BAFFLE: 离线增强学习中的后门攻击

    BAFFLE: Backdoor Attack in Offline Reinforcement Learning. (arXiv:2210.04688v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.04688](http://arxiv.org/abs/2210.04688)

    本文研究离线增强学习中的后门攻击，通过向数据中添加扰动，使得智能体在注入触发器的观测值上采取低奖励动作，从而提出了BAFFLE方法。

    

    越来越多的研究关注于强化学习（RL）方法，允许智能体通过与环境的交互中收集的试错经验进行学习。最近，离线RL成为一种流行的RL范例，因为它节省了与环境的交互。在离线RL中，数据提供者共享大规模的预先收集的数据集，其他人可以在不与环境交互的情况下训练高质量的智能体。这种范例在机器人控制、自动驾驶等关键任务中表现出有效性。然而，较少关注研究离线RL系统的安全威胁。本文关注后门攻击，其中一些扰动被添加到数据（观测值）中，使得在给定正常观测值的情况下，智能体采取高奖励的动作，在注入触发器的观测值上采取低奖励的动作。在本文中，我们提出了BAFFLE（离线增强学习中的后门攻击），这是一种方法。

    A growing body of research has focused on the Reinforcement Learning (RL) methods which allow the agent to learn from trial-and-error experiences gathered during the interaction with the environment. Recently, offline RL becomes a popular RL paradigm because it saves the interactions with environments. In offline RL, data providers share large pre-collected datasets, and others can train high-quality agents without interacting with the environments. This paradigm has demonstrated effectiveness in critical tasks like robot control, autonomous driving, etc. However, less attention is paid to investigating the security threats to the offline RL system. This paper focuses on backdoor attacks, where some perturbations are added to the data (observations) such that given normal observations, the agent takes high-rewards actions, and low-reward actions on observations injected with triggers. In this paper, we propose Baffle (Backdoor Attack for Offline Reinforcement Learning), an approach tha
    
[^36]: 分布式数据上的协同因果推断

    Collaborative causal inference on distributed data. (arXiv:2208.07898v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.07898](http://arxiv.org/abs/2208.07898)

    提出了一种数据协作准实验（DC-QE）方法，可以在保护隐私的前提下对分布式数据进行因果推断。通过共享中间表示而不是私有数据，估计倾向分数和处理效应，能够减少随机误差和偏差，相比现有方法有更好的估计结果。

    

    近年来，基于隐私保护的分布式数据因果推断技术的发展引起了广泛关注。为了解决这个问题，我们提出了一种数据协作准实验（DC-QE）方法，可以在保护隐私的前提下对分布式数据进行因果推断。在我们的方法中，首先，本地各方从私有数据中构建降维的中间表示。其次，他们共享中间表示，而不是私有数据，以保护隐私。然后，从共享的中间表示中估计倾向分数。最后，从倾向分数中估计处理效应。我们的方法能够减少随机误差和偏差，而现有方法只能减少处理效应估计中的随机误差。通过在人工数据和实际数据上进行数值实验，我们确认我们的方法可以得到比单独分析更好的估计结果。

    The development of technologies for causal inference with the privacy preservation of distributed data has attracted considerable attention in recent years. To address this issue, we propose a data collaboration quasi-experiment (DC-QE) that enables causal inference from distributed data with privacy preservation. In our method, first, local parties construct dimensionality-reduced intermediate representations from the private data. Second, they share intermediate representations, instead of private data for privacy preservation. Third, propensity scores were estimated from the shared intermediate representations. Finally, the treatment effects were estimated from propensity scores. Our method can reduce both random errors and biases, whereas existing methods can only reduce random errors in the estimation of treatment effects. Through numerical experiments on both artificial and real-world data, we confirmed that our method can lead to better estimation results than individual analyse
    
[^37]: 多频联合社区检测和相位同步

    Multi-Frequency Joint Community Detection and Phase Synchronization. (arXiv:2206.12276v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2206.12276](http://arxiv.org/abs/2206.12276)

    本文提出了两种简单而高效的算法，利用MLE公式并从多个频率的信息中受益，用于解决具有相对相位的随机块模型上的联合社区检测和相位同步问题。

    This paper proposes two simple and efficient algorithms that leverage the MLE formulation and benefit from the information across multiple frequencies to solve the joint community detection and phase synchronization problem on the stochastic block model with relative phase.

    本文研究了具有相对相位的随机块模型上的联合社区检测和相位同步问题，其中每个节点都与一个未知的相位角相关联。这个问题具有多种实际应用，旨在同时恢复簇结构和相关的相位角。我们通过仔细研究其最大似然估计（MLE）公式，展示了这个问题呈现出“多频”结构，而现有方法并非源于这个角度。为此，提出了两种简单而高效的算法，利用MLE公式并从多个频率的信息中受益。前者是基于新颖的多频列主元QR分解的谱方法。应用于观测矩阵的前几个特征向量的分解提供了有关簇结构和相关相位角的关键信息。第二种方法是迭代的多频率方法。

    This paper studies the joint community detection and phase synchronization problem on the stochastic block model with relative phase, where each node is associated with an unknown phase angle. This problem, with a variety of real-world applications, aims to recover the cluster structure and associated phase angles simultaneously. We show this problem exhibits a ``multi-frequency'' structure by closely examining its maximum likelihood estimation (MLE) formulation, whereas existing methods are not originated from this perspective. To this end, two simple yet efficient algorithms that leverage the MLE formulation and benefit from the information across multiple frequencies are proposed. The former is a spectral method based on the novel multi-frequency column-pivoted QR factorization. The factorization applied to the top eigenvectors of the observation matrix provides key information about the cluster structure and associated phase angles. The second approach is an iterative multi-frequen
    
[^38]: 用简洁可解释的加性模型和结构交互预测人口普查调查反应率

    Predicting Census Survey Response Rates With Parsimonious Additive Models and Structured Interactions. (arXiv:2108.11328v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2108.11328](http://arxiv.org/abs/2108.11328)

    本文提出了一种可解释的非参数加性模型，使用少量主要和成对交互效应预测调查反应率。该模型可以生成易于可视化和解释的预测面，并取得了 ROAM 数据集上的最先进性能，可以提供改进美国人口普查局和其他调查的反应率议论。

    

    本文考虑使用一系列灵活且可解释的非参数模型预测调查反应率。本研究受到美国人口普查局著名的 ROAM 应用的启发，该应用使用在美国人口普查规划数据库数据上训练的线性回归模型来识别难以调查的区域。十年前组织的一场众包竞赛表明，基于回归树集成的机器学习方法在预测调查反应率方面表现最佳；然而，由于它们的黑盒特性，相应的模型不能用于拟定的应用。我们考虑使用 $\ell_0$-based 惩罚的非参数加性模型，它具有少数主要和成对交互效应。从方法论的角度来看，我们研究了我们估计器的计算和统计方面，并讨论了将强层次交互合并的变体。我们的算法（在Github 上开源）允许我们生成易于可视化和解释的预测面，从而获得有关调查反应率的可行见解。我们提出的模型在 ROAM 数据集上实现了最先进的性能，并可以提供有关美国人口普查局和其他调查的改进调查反应率的见解。

    In this paper we consider the problem of predicting survey response rates using a family of flexible and interpretable nonparametric models. The study is motivated by the US Census Bureau's well-known ROAM application which uses a linear regression model trained on the US Census Planning Database data to identify hard-to-survey areas. A crowdsourcing competition organized around ten years ago revealed that machine learning methods based on ensembles of regression trees led to the best performance in predicting survey response rates; however, the corresponding models could not be adopted for the intended application due to their black-box nature. We consider nonparametric additive models with small number of main and pairwise interaction effects using $\ell_0$-based penalization. From a methodological viewpoint, we study both computational and statistical aspects of our estimator; and discuss variants that incorporate strong hierarchical interactions. Our algorithms (opensourced on gith
    

