# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RealKIE: Five Novel Datasets for Enterprise Key Information Extraction](https://arxiv.org/abs/2403.20101) | RealKIE提供了五个具有挑战性的企业关键信息提取数据集，为投资分析和法律数据处理等任务提供了现实的测试基地，并为NLP模型的发展做出了贡献。 |
| [^2] | [PolyNet: Learning Diverse Solution Strategies for Neural Combinatorial Optimization](https://arxiv.org/abs/2402.14048) | PolyNet通过学习互补解决策略来改善解空间探索，避免了人为规则导致解决方案质量下降的问题。 |
| [^3] | [Understanding Practical Membership Privacy of Deep Learning](https://arxiv.org/abs/2402.06674) | 该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。 |
| [^4] | [Tabular Data: Is Attention All You Need?](https://arxiv.org/abs/2402.03970) | 本文引入了一项大规模实证研究，比较了神经网络和梯度提升决策树在表格数据上的表现，还比较了基于Transformer的架构和传统的多层感知器（MLP）与残差连接的架构。实证结果显示，神经网络在决策树方面具有竞争力，而基于Transformer的架构在表格数据集上并没有超过传统MLP架构的简化变体。 |
| [^5] | [Position Paper: Assessing Robustness, Privacy, and Fairness in Federated Learning Integrated with Foundation Models](https://arxiv.org/abs/2402.01857) | 本文评估了基于Foundation模型集成联邦学习中鲁棒性、隐私和公平性的挑战和问题，并提出了应对策略和研究方向。 |
| [^6] | [Generalization of LiNGAM that allows confounding.](http://arxiv.org/abs/2401.16661) | 本文提出了一种名为LiNGAM-MMI的方法，可以增强LiNGAM模型以处理混淆问题。该方法使用KL散度量化混淆程度，并通过最短路径问题解决方案高效地确定变量顺序，不论是否存在混淆情况。实验证明，LiNGAM-MMI可以更准确地识别正确的变量顺序。 |
| [^7] | [Sum-of-Parts Models: Faithful Attributions for Groups of Features.](http://arxiv.org/abs/2310.16316) | Sum-of-Parts模型通过构造保证特征组归因的忠实性，将预测分解为可解释的分数之和，帮助天体物理学家发现了关于星系形成的新知识。 |
| [^8] | [AgentBench: Evaluating LLMs as Agents.](http://arxiv.org/abs/2308.03688) | AgentBench是一个用于评估LLMs作为代理人的多维度基准，发现在复杂环境中，商业LLMs在充当代理人方面表现强劲，但与开源竞争对手相比，存在显著性能差距。该研究揭示了LLMs在长期推理、决策和指令遵循能力上的瓶颈。 |
| [^9] | [Fitted Value Iteration Methods for Bicausal Optimal Transport.](http://arxiv.org/abs/2306.12658) | 本文提出了一种适用于双因果最优传输问题的拟合值迭代方法，能够在保证精度的同时具有良好的可扩展性，数值实验结果也证明了该方法的优越性。 |
| [^10] | [Quick Adaptive Ternary Segmentation: An Efficient Decoding Procedure For Hidden Markov Models.](http://arxiv.org/abs/2305.18578) | 提出了一种名为QATS的新方法，用于高效解码隐藏马尔可夫模型序列。它的计算复杂性为多对数和立方，特别适用于具有相对较少状态的大型HMM。 |
| [^11] | [Tutorial on amortized optimization.](http://arxiv.org/abs/2202.00665) | 该教程介绍了分摊优化的基础，并总结了其在变分推断、稀疏编码、元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。 |

# 详细

[^1]: RealKIE: 五个新颖的企业关键信息提取数据集

    RealKIE: Five Novel Datasets for Enterprise Key Information Extraction

    [https://arxiv.org/abs/2403.20101](https://arxiv.org/abs/2403.20101)

    RealKIE提供了五个具有挑战性的企业关键信息提取数据集，为投资分析和法律数据处理等任务提供了现实的测试基地，并为NLP模型的发展做出了贡献。

    

    我们介绍了RealKIE，这是一个旨在推动关键信息提取方法发展的五个具有挑战性的数据集基准，重点是企业应用。这些数据集包括美国SEC S1文件、美国保密协议、英国慈善报告、FCC发票和资源合同等各种类型的文档。每个数据集都具有独特的挑战：文本序列化不佳、长文档中稀疏的注释和复杂的表格布局。这些数据集为关键信息提取任务（如投资分析和法律数据处理）提供了一个现实的测试基地。除了介绍这些数据集外，我们还提供了对注释过程、文档处理技术和基线建模方法的深入描述。这一贡献促进了能够处理实际挑战的NLP模型的发展，并支持进一步研究可应用于工业的信息提取技术。

    arXiv:2403.20101v1 Announce Type: new  Abstract: We introduce RealKIE, a benchmark of five challenging datasets aimed at advancing key information extraction methods, with an emphasis on enterprise applications. The datasets include a diverse range of documents including SEC S1 Filings, US Non-disclosure Agreements, UK Charity Reports, FCC Invoices, and Resource Contracts. Each presents unique challenges: poor text serialization, sparse annotations in long documents, and complex tabular layouts. These datasets provide a realistic testing ground for key information extraction tasks like investment analysis and legal data processing.   In addition to presenting these datasets, we offer an in-depth description of the annotation process, document processing techniques, and baseline modeling approaches. This contribution facilitates the development of NLP models capable of handling practical challenges and supports further research into information extraction technologies applicable to indu
    
[^2]: PolyNet：学习神经组合优化的多样化解决策略

    PolyNet: Learning Diverse Solution Strategies for Neural Combinatorial Optimization

    [https://arxiv.org/abs/2402.14048](https://arxiv.org/abs/2402.14048)

    PolyNet通过学习互补解决策略来改善解空间探索，避免了人为规则导致解决方案质量下降的问题。

    

    强化学习方法用于构建组合优化问题解决方案，迅速接近人类设计的算法性能。为了进一步缩小差距，基于学习的方法在搜索过程中必须高效地探索解空间。最近的方法通过强制实施多样化解生成来人为增加探索，然而，这些规则可能损害解决方案质量，并且难以为更复杂的问题设计。本文介绍了PolyNet，一种通过学习互补解决策略来改善解空间探索的方法。与其他作品不同，PolyNet仅使用单个解码器，并且训练图式不通过人为规则强制实施多样化解生成。我们在四个组合优化问题上评估PolyNet，并观察到隐式多样性机制允许P

    arXiv:2402.14048v1 Announce Type: cross  Abstract: Reinforcement learning-based methods for constructing solutions to combinatorial optimization problems are rapidly approaching the performance of human-designed algorithms. To further narrow the gap, learning-based approaches must efficiently explore the solution space during the search process. Recent approaches artificially increase exploration by enforcing diverse solution generation through handcrafted rules, however, these rules can impair solution quality and are difficult to design for more complex problems. In this paper, we introduce PolyNet, an approach for improving exploration of the solution space by learning complementary solution strategies. In contrast to other works, PolyNet uses only a single-decoder and a training schema that does not enforce diverse solution generation through handcrafted rules. We evaluate PolyNet on four combinatorial optimization problems and observe that the implicit diversity mechanism allows P
    
[^3]: 理解深度学习的实际成员隐私

    Understanding Practical Membership Privacy of Deep Learning

    [https://arxiv.org/abs/2402.06674](https://arxiv.org/abs/2402.06674)

    该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。

    

    我们应用最先进的成员推理攻击（MIA）来系统地测试细调大型图像分类模型的实际隐私漏洞。我们的重点是理解使数据集和样本容易受到成员推理攻击的特性。在数据集特性方面，我们发现数据中每个类别的示例数量与成员推理攻击的漏洞之间存在强烈的幂律依赖关系，这是以攻击的真阳性率（在低假阳性率下测量）来衡量的。对于个别样本而言，在训练结束时产生的大梯度与成员推理攻击的漏洞之间存在很强的相关性。

    We apply a state-of-the-art membership inference attack (MIA) to systematically test the practical privacy vulnerability of fine-tuning large image classification models.We focus on understanding the properties of data sets and samples that make them vulnerable to membership inference. In terms of data set properties, we find a strong power law dependence between the number of examples per class in the data and the MIA vulnerability, as measured by true positive rate of the attack at a low false positive rate. For an individual sample, large gradients at the end of training are strongly correlated with MIA vulnerability.
    
[^4]: 表格数据：注意力是唯一需要的吗？

    Tabular Data: Is Attention All You Need?

    [https://arxiv.org/abs/2402.03970](https://arxiv.org/abs/2402.03970)

    本文引入了一项大规模实证研究，比较了神经网络和梯度提升决策树在表格数据上的表现，还比较了基于Transformer的架构和传统的多层感知器（MLP）与残差连接的架构。实证结果显示，神经网络在决策树方面具有竞争力，而基于Transformer的架构在表格数据集上并没有超过传统MLP架构的简化变体。

    

    深度学习彻底改变了人工智能领域，并在涉及图像和文本数据的应用中取得了令人瞩目的成就。遗憾的是，关于神经网络在结构化表格数据上的优势存在着不一致的证据。本文引入了一项大规模实证研究，比较了神经网络和梯度提升决策树在表格数据上的表现，还比较了基于Transformer的架构和传统的多层感知器（MLP）与残差连接的架构。与之前的研究相比，我们的实证发现表明神经网络在决策树方面具有竞争力。此外，我们还评估了基于Transformer的架构在表格数据集上并没有超过传统MLP架构的简化变体。因此，本文帮助研究和实践社区在未来的表格数据应用中做出明智的选择。

    Deep Learning has revolutionized the field of AI and led to remarkable achievements in applications involving image and text data. Unfortunately, there is inconclusive evidence on the merits of neural networks for structured tabular data. In this paper, we introduce a large-scale empirical study comparing neural networks against gradient-boosted decision trees on tabular data, but also transformer-based architectures against traditional multi-layer perceptrons (MLP) with residual connections. In contrast to prior work, our empirical findings indicate that neural networks are competitive against decision trees. Furthermore, we assess that transformer-based architectures do not outperform simpler variants of traditional MLP architectures on tabular datasets. As a result, this paper helps the research and practitioner communities make informed choices on deploying neural networks on future tabular data applications.
    
[^5]: 评估基于Foundation模型集成联邦学习的鲁棒性、隐私和公平性的立场论文

    Position Paper: Assessing Robustness, Privacy, and Fairness in Federated Learning Integrated with Foundation Models

    [https://arxiv.org/abs/2402.01857](https://arxiv.org/abs/2402.01857)

    本文评估了基于Foundation模型集成联邦学习中鲁棒性、隐私和公平性的挑战和问题，并提出了应对策略和研究方向。

    

    联邦学习（FL）是分散式机器学习的重大突破，但面临诸多挑战，如数据可用性有限和计算资源的变化性，这可能会限制模型的性能和可伸缩性。将Foundation模型（FM）集成到FL中，可以解决这些问题，通过预训练和数据增强增加数据丰富性并减少计算需求。然而，这种集成引入了鲁棒性、隐私和公平性方面的新问题，在现有研究中尚未得到充分解决。我们通过系统评估FM-FL集成对这些方面的影响，进行了初步调查。我们分析了其中的权衡取舍，揭示了该集成引入的威胁和问题，并提出了一套用于应对这些挑战的标准和策略。此外，我们还鉴定了可能解决这些问题的一些前景方向和研究方向。

    Federated Learning (FL), while a breakthrough in decentralized machine learning, contends with significant challenges such as limited data availability and the variability of computational resources, which can stifle the performance and scalability of the models. The integration of Foundation Models (FMs) into FL presents a compelling solution to these issues, with the potential to enhance data richness and reduce computational demands through pre-training and data augmentation. However, this incorporation introduces novel issues in terms of robustness, privacy, and fairness, which have not been sufficiently addressed in the existing research. We make a preliminary investigation into this field by systematically evaluating the implications of FM-FL integration across these dimensions. We analyze the trade-offs involved, uncover the threats and issues introduced by this integration, and propose a set of criteria and strategies for navigating these challenges. Furthermore, we identify po
    
[^6]: 允许混淆的LiNGAM的泛化

    Generalization of LiNGAM that allows confounding. (arXiv:2401.16661v1 [cs.LG])

    [http://arxiv.org/abs/2401.16661](http://arxiv.org/abs/2401.16661)

    本文提出了一种名为LiNGAM-MMI的方法，可以增强LiNGAM模型以处理混淆问题。该方法使用KL散度量化混淆程度，并通过最短路径问题解决方案高效地确定变量顺序，不论是否存在混淆情况。实验证明，LiNGAM-MMI可以更准确地识别正确的变量顺序。

    

    LiNGAM使用加性噪声模型来确定因果关系的变量顺序，但在混淆方面面临挑战。先前的方法在保持LiNGAM的基本结构的同时，试图识别和处理受混淆影响的变量。结果是，不论是否存在混淆，这些方法都需要大量的计算资源，并且不能确保检测到所有的混淆类型。相比之下，本文通过引入LiNGAM-MMI对LiNGAM进行了增强，该方法使用KL散度量化混淆程度，并安排变量以最小化其影响。该方法通过最短路径问题的形式高效地实现全局最优的变量顺序。在无混淆的情况下，LiNGAM-MMI的处理数据效率与传统LiNGAM相当，同时有效处理混淆情况。我们的实验结果表明，LiNGAM-MMI更准确地确定了正确的变量顺序...

    LiNGAM determines the variable order from cause to effect using additive noise models, but it faces challenges with confounding. Previous methods maintained LiNGAM's fundamental structure while trying to identify and address variables affected by confounding. As a result, these methods required significant computational resources regardless of the presence of confounding, and they did not ensure the detection of all confounding types. In contrast, this paper enhances LiNGAM by introducing LiNGAM-MMI, a method that quantifies the magnitude of confounding using KL divergence and arranges the variables to minimize its impact. This method efficiently achieves a globally optimal variable order through the shortest path problem formulation. LiNGAM-MMI processes data as efficiently as traditional LiNGAM in scenarios without confounding while effectively addressing confounding situations. Our experimental results suggest that LiNGAM-MMI more accurately determines the correct variable order, bo
    
[^7]: Sum-of-Parts模型：对特征组的忠实归因

    Sum-of-Parts Models: Faithful Attributions for Groups of Features. (arXiv:2310.16316v1 [cs.LG])

    [http://arxiv.org/abs/2310.16316](http://arxiv.org/abs/2310.16316)

    Sum-of-Parts模型通过构造保证特征组归因的忠实性，将预测分解为可解释的分数之和，帮助天体物理学家发现了关于星系形成的新知识。

    

    如果机器学习模型的解释准确反映了其决策过程，则被认为是“忠实”的解释。然而，例如深度学习的特征归因等解释并不能保证忠实，有可能产生具有误导性的解释。在这项工作中，我们开发了Sum-of-Parts（SOP）模型，它是一类模型，其预测具有通过构造保证忠实的特征组归因。该模型将预测分解为可解释的分数之和，每个分数直接归因于一组稀疏特征。我们使用标准可解释性指标对SOP进行评估，并在一个案例研究中，利用SOP提供的忠实解释帮助天体物理学家发现了关于星系形成的新知识。

    An explanation of a machine learning model is considered "faithful" if it accurately reflects the model's decision-making process. However, explanations such as feature attributions for deep learning are not guaranteed to be faithful, and can produce potentially misleading interpretations. In this work, we develop Sum-of-Parts (SOP), a class of models whose predictions come with grouped feature attributions that are faithful-by-construction. This model decomposes a prediction into an interpretable sum of scores, each of which is directly attributable to a sparse group of features. We evaluate SOP on benchmarks with standard interpretability metrics, and in a case study, we use the faithful explanations from SOP to help astrophysicists discover new knowledge about galaxy formation.
    
[^8]: AgentBench: 评估LLMs作为代理人

    AgentBench: Evaluating LLMs as Agents. (arXiv:2308.03688v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2308.03688](http://arxiv.org/abs/2308.03688)

    AgentBench是一个用于评估LLMs作为代理人的多维度基准，发现在复杂环境中，商业LLMs在充当代理人方面表现强劲，但与开源竞争对手相比，存在显著性能差距。该研究揭示了LLMs在长期推理、决策和指令遵循能力上的瓶颈。

    

    大型语言模型(LLMs)变得越来越智能和自主，针对传统的NLP任务之外的现实世界实际任务。因此，迫切需要在互动环境中评估LLMs作为代理人在具有挑战性的任务上的推理和决策能力。我们提出了AgentBench，一个多维度演变的基准，目前包括8个不同的环境，以评估LLM作为代理人在多轮开放式生成设置中的推理和决策能力。我们在27个基于API和开源的LLM上进行了广泛的测试，结果表明，虽然顶级商业LLM在复杂环境中表现出良好的代理人能力，但它们与开源竞争对手之间的性能差距很大。我们找出了环境和LLM中失败的典型原因，表明长期推理、决策和遵循指示能力不佳是开发可用LLM代理人的主要障碍。通过对代码和高质量进行训练

    Large Language Models (LLMs) are becoming increasingly smart and autonomous, targeting real-world pragmatic missions beyond traditional NLP tasks. As a result, there has been an urgent need to evaluate LLMs as agents on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional evolving benchmark that currently consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities in a multi-turn open-ended generation setting. Our extensive test over 27 API-based and open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and OSS competitors. We identify the typical reasons of failures in environments and LLMs, showing that poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents. Training on code and high quality 
    
[^9]: 拟合值迭代方法求解适应结构双因果最优传输问题

    Fitted Value Iteration Methods for Bicausal Optimal Transport. (arXiv:2306.12658v1 [stat.ML])

    [http://arxiv.org/abs/2306.12658](http://arxiv.org/abs/2306.12658)

    本文提出了一种适用于双因果最优传输问题的拟合值迭代方法，能够在保证精度的同时具有良好的可扩展性，数值实验结果也证明了该方法的优越性。

    

    本文提出一种拟合值迭代方法(FVI)用于计算具有适应结构的双因果最优传输(OT)。基于动态规划的形式化表述，FVI采用函数类用于近似双因果OT中的值函数。在可集中条件和近似完备性假设下，我们使用（局部）Rademacher复杂度证明了样本复杂度。此外，我们证明了深度多层神经网络具有适当结构，满足样本复杂度证明所需的关键假设条件。数值实验表明，FVI在时间跨度增加时优于线性规划和适应性Sinkhorn方法，在保持可接受精度的同时具有很好的可扩展性。

    We develop a fitted value iteration (FVI) method to compute bicausal optimal transport (OT) where couplings have an adapted structure. Based on the dynamic programming formulation, FVI adopts a function class to approximate the value functions in bicausal OT. Under the concentrability condition and approximate completeness assumption, we prove the sample complexity using (local) Rademacher complexity. Furthermore, we demonstrate that multilayer neural networks with appropriate structures satisfy the crucial assumptions required in sample complexity proofs. Numerical experiments reveal that FVI outperforms linear programming and adapted Sinkhorn methods in scalability as the time horizon increases, while still maintaining acceptable accuracy.
    
[^10]: 快速自适应三元分割：隐马尔可夫模型的有效解码程序。

    Quick Adaptive Ternary Segmentation: An Efficient Decoding Procedure For Hidden Markov Models. (arXiv:2305.18578v1 [stat.ME])

    [http://arxiv.org/abs/2305.18578](http://arxiv.org/abs/2305.18578)

    提出了一种名为QATS的新方法，用于高效解码隐藏马尔可夫模型序列。它的计算复杂性为多对数和立方，特别适用于具有相对较少状态的大型HMM。

    

    隐马尔可夫模型（HMM）以不可观察的（隐藏的）马尔可夫链和可观测的过程为特征，后者是隐藏链的噪声版本。从嘈杂的观测中解码原始信号（即隐藏链）是几乎所有基于HMM的数据分析的主要目标。现有的解码算法，如维特比算法，在观测序列长度最多线性的情况下具有计算复杂度，并且在马尔可夫链状态空间的大小中具有次二次计算复杂度。我们提出了快速自适应三元分割（QATS），这是一种分而治之的过程，可在序列长度的多对数计算复杂度和马尔可夫链状态空间的三次计算复杂度下解码隐藏的序列，因此特别适用于具有相对较少状态的大规模HMM。该程序还建议一种有效的数据存储方式，即特定的累积总和。实质上，估计的状态序列按顺序最大化局部似然。

    Hidden Markov models (HMMs) are characterized by an unobservable (hidden) Markov chain and an observable process, which is a noisy version of the hidden chain. Decoding the original signal (i.e., hidden chain) from the noisy observations is one of the main goals in nearly all HMM based data analyses. Existing decoding algorithms such as the Viterbi algorithm have computational complexity at best linear in the length of the observed sequence, and sub-quadratic in the size of the state space of the Markov chain. We present Quick Adaptive Ternary Segmentation (QATS), a divide-and-conquer procedure which decodes the hidden sequence in polylogarithmic computational complexity in the length of the sequence, and cubic in the size of the state space, hence particularly suited for large scale HMMs with relatively few states. The procedure also suggests an effective way of data storage as specific cumulative sums. In essence, the estimated sequence of states sequentially maximizes local likeliho
    
[^11]: 关于分摊优化的教程

    Tutorial on amortized optimization. (arXiv:2202.00665v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.00665](http://arxiv.org/abs/2202.00665)

    该教程介绍了分摊优化的基础，并总结了其在变分推断、稀疏编码、元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。

    

    优化是一种普遍的建模工具，经常在反复解决相同问题的情况下使用。分摊优化方法使用学习来预测这些设置中问题的解决方案，利用相似问题实例之间的共享结构。这些方法在变分推断和强化学习中至关重要，能够比不使用分摊的传统优化方法快几个数量级地解决优化问题。本次教程介绍了这些进步背后的分摊优化基础，并概述了它们在变分推断、稀疏编码、基于梯度的元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。本教程的源代码可在https://github.com/facebookresearch/amortized-optimization-tutorial上获得。

    Optimization is a ubiquitous modeling tool and is often deployed in settings which repeatedly solve similar instances of the same problem. Amortized optimization methods use learning to predict the solutions to problems in these settings, exploiting the shared structure between similar problem instances. These methods have been crucial in variational inference and reinforcement learning and are capable of solving optimization problems many orders of magnitudes times faster than traditional optimization methods that do not use amortization. This tutorial presents an introduction to the amortized optimization foundations behind these advancements and overviews their applications in variational inference, sparse coding, gradient-based meta-learning, control, reinforcement learning, convex optimization, optimal transport, and deep equilibrium networks. The source code for this tutorial is available at https://github.com/facebookresearch/amortized-optimization-tutorial.
    

