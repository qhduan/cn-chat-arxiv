# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TrustAgent: Towards Safe and Trustworthy LLM-based Agents through Agent Constitution](https://rss.arxiv.org/abs/2402.01586) | 本文介绍了一种基于代理构成的代理框架TrustAgent，该框架通过预先规划、规划过程中和计划后检查三种策略来提高LLM代理的安全性。实验结果表明，这些方法可以有效识别和预防潜在危险。此外，还研究了安全性与使用者满意度以及模型推理能力与效率之间的关系。 |
| [^2] | [Data-centric Prediction Explanation via Kernelized Stein Discrepancy](https://arxiv.org/abs/2403.15576) | 该论文提出了一种基于内核化斯坦不相容性的数据中心预测解释方法，通过利用内核函数识别提供最佳预测支持给测试点的训练样本，取得了优异性能。 |
| [^3] | [Scattered Mixture-of-Experts Implementation](https://arxiv.org/abs/2403.08245) | ScatterMoE是一种在GPU上实现的稀疏专家混合模型，通过避免填充和过多复制输入，提高了推理和训练速度，并减少了内存占用。 |
| [^4] | [Efficient and Guaranteed-Safe Non-Convex Trajectory Optimization with Constrained Diffusion Model](https://arxiv.org/abs/2403.05571) | 本文提出了一种具有约束扩散模型的高效和保证安全的非凸轨迹优化框架，通过结合扩散模型和数值求解器，保证了计算效率和约束满足。 |
| [^5] | [Aligners: Decoupling LLMs and Alignment](https://arxiv.org/abs/2403.04224) | 提出了一种通过训练对齐器模型来解耦大型语言模型（LLMs）和对齐，以减少对齐对性能的潜在负面影响。 |
| [^6] | [GreenLLaMA: A Framework for Detoxification with Explanations](https://arxiv.org/abs/2402.15951) | GreenLLaMA是一种全面的端到端解毒框架，通过跨平台语料库训练出的模型优于当前最先进的模型。 |
| [^7] | [Divide-or-Conquer? Which Part Should You Distill Your LLM?](https://arxiv.org/abs/2402.15000) | 本文提出了一种将推理任务分解为问题分解阶段和问题解决阶段的策略，发现问题分解阶段相比问题解决更容易提炼为较小模型，并证实该策略胜过单阶段解决方案。 |
| [^8] | [LexC-Gen: Generating Data for Extremely Low-Resource Languages with Large Language Models and Bilingual Lexicons](https://arxiv.org/abs/2402.14086) | LexC-Gen提出了一种词典条件数据生成方法，可以以大规模生成低资源语言分类任务数据，取得了较好的效果。 |
| [^9] | [Average gradient outer product as a mechanism for deep neural collapse](https://arxiv.org/abs/2402.13728) | 本文通过提供证据表明，深度神经网络中的神经坍塌主要是通过平均梯度外积进行深度特征学习的，权重的奇异结构与AGOP高度相关，导致类内变异坍塌。 |
| [^10] | [Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A](https://arxiv.org/abs/2402.13213) | 多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。 |
| [^11] | [Align Your Intents: Offline Imitation Learning via Optimal Transport](https://arxiv.org/abs/2402.13037) | 通过最优传输的离线模仿学习方法AILOT，可以在缺乏明确奖励的情况下，仅通过观察专家学习所需的行为。 |
| [^12] | [Identifying Factual Inconsistency in Summaries: Towards Effective Utilization of Large Language Model](https://arxiv.org/abs/2402.12821) | 该研究提出了针对摘要中事实不一致性的解决方案：通过大型语言模型在正确的范式设计下无需训练即可解决任务，并提出了训练策略以精炼更小型的高准确性的语言模型。 |
| [^13] | [Statistical Test for Generated Hypotheses by Diffusion Models](https://arxiv.org/abs/2402.11789) | 本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。 |
| [^14] | [Avoiding Catastrophe in Continuous Spaces by Asking for Help](https://arxiv.org/abs/2402.08062) | 在连续空间中，通过寻求帮助来避免灾难。引入了一种上下文多臂赌博问题的变体，目标是最小化灾难发生的概率。提出了一种算法，在连续1D状态空间和相对简单的回报函数下，遗憾和向导师查询率都趋近于0。 |
| [^15] | [Foundational Inference Models for Dynamical Systems](https://arxiv.org/abs/2402.07594) | 本研究提出了一种基于监督学习的框架，用于从噪声数据中零样本推理动态系统的普通微分方程（ODE）。通过生成大型ODE数据集，并利用神经网络将噪声观察和初始条件以及向量场进行映射，得到称为基础推理模型（FIM）的结果模型。这些模型可以复制、匹配和组合，用于构建任何维度的推理模型。 |
| [^16] | [Graph Cuts with Arbitrary Size Constraints Through Optimal Transport](https://arxiv.org/abs/2402.04732) | 本论文提出了一种新的图割算法，通过将图割问题制定为正则化的Gromov-Wasserstein问题，并使用加速的近端GD算法解决，实现在任意大小约束下对图进行分割，在非平衡数据集聚类等应用中具有更高的效率。 |
| [^17] | [Densely Multiplied Physics Informed Neural Network](https://arxiv.org/abs/2402.04390) | 该论文通过改进神经网络架构，提出了一种密集乘法物理信息神经网络（DM-PINN）架构，它有效利用隐藏层的输出，显著提高了PINN的准确性和性能。 |
| [^18] | [MetaOptimize: A Framework for Optimizing Step Sizes and Other Meta-parameters](https://arxiv.org/abs/2402.02342) | MetaOptimize是一个框架，通过动态调整学习率来优化机器学习算法中的元参数，以提高训练效率和模型性能。 |
| [^19] | [LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law](https://arxiv.org/abs/2402.00795) | 本文研究了预训练语言模型LLMs对动力系统行为的外推能力，发现LLaMA 2能够准确预测动力系统的时间序列。此外，输入上下文窗口的长度越长，学习到的物理规则的准确性越高，揭示了一种上下文中的神经比例定律。 |
| [^20] | [Extension of Recurrent Kernels to different Reservoir Computing topologies.](http://arxiv.org/abs/2401.14557) | 该研究通过提供特定RC体系结构与相应循环内核形式等价性的经验分析，填补了Leaky RC、Sparse RC和Deep RC等已建立的RC范例尚未进行分析的空白。此外，研究还揭示了稀疏连接在RC体系结构中的作用，并提出了一种依赖储备大小的最佳稀疏性水平。最后，该研究的系统分析表明，在Deep RC模型中，通过减小尺寸的连续储备可以更好地实现收敛。 |
| [^21] | [MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks.](http://arxiv.org/abs/2401.09624) | MITS-GAN是一种用于保护医学影像免受篡改的新方法，通过引入适当的高斯噪声作为防护措施，打乱攻击者的生成对抗网络架构的输出。实验结果表明MITS-GAN能够生成耐篡改图像，具有优越的性能。 |
| [^22] | [Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning.](http://arxiv.org/abs/2401.08632) | 将质量多样性优化与描述符条件加强学习相结合，以克服进化算法的局限性，并在生成既多样又高性能的解决方案集合方面取得成功。 |
| [^23] | [Data Augmentation for Code Translation with Comparable Corpora and Multiple References.](http://arxiv.org/abs/2311.00317) | 该论文介绍了两种数据增强方法来改善编程语言之间的代码翻译。通过构建可比较的语料库和增加多个参考翻译，实验结果表明这些方法显著提高了CodeT5在Java、Python和C++之间的翻译准确性。 |
| [^24] | [Automated Bug Generation in the era of Large Language Models.](http://arxiv.org/abs/2310.02407) | 本论文探讨了在大型语言模型时代的自动缺陷生成问题，针对难以检测和难以修复的缺陷提出了解决方案，并分析了基于学习的技术中这两个目标的冲突。 |
| [^25] | [Imitation Learning from Observation through Optimal Transport.](http://arxiv.org/abs/2310.01632) | 本文提出了一种通过最优输运进行从观察中的模仿学习的方法，该方法不需要学习模型或对抗学习，可以与任何强化学习算法集成，并在各种连续控制任务上超过了现有最先进方法，在ILfO设置下实现了专家级的性能。 |
| [^26] | [Accelerating Non-IID Federated Learning via Heterogeneity-Guided Client Sampling.](http://arxiv.org/abs/2310.00198) | 本文提出了一种名为HiCS-FL的方法，通过利用客户端的网络输出层更新来估计数据的统计异质性，并利用此信息进行客户端的聚类选择，以加速非独立同分布联邦学习。 |
| [^27] | [MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation.](http://arxiv.org/abs/2309.13042) | MosaicFusion是一种用于大词汇实例分割的数据增强方法，通过将扩散模型作为数据集生成器，能够生成大量合成标记数据。在实验中，我们的方法在准确率和泛化能力方面取得了显著的提升。 |
| [^28] | [Efficient Finite Initialization for Tensorized Neural Networks.](http://arxiv.org/abs/2309.06577) | 这种方法提出了一种高效有限初始化张量化神经网络层的方法，避免了参数爆炸问题，并通过使用弗罗贝尼乌斯范数的迭代部分形式来计算范数，使其具有有限范围。应用于不同层的实验表明其性能良好。 |
| [^29] | [Exploring Link Prediction over Hyper-Relational Temporal Knowledge Graphs Enhanced with Time-Invariant Relational Knowledge.](http://arxiv.org/abs/2307.10219) | 这项研究填补了时间KG和超关系KG推理之间的差距，并开发了两个新的基准超关系TKG数据集。 |
| [^30] | [Benchmarks and Custom Package for Electrical Load Forecasting.](http://arxiv.org/abs/2307.07191) | 本文提供了一个全面的电力负荷预测存档，包括负荷领域特定的特征工程，帮助模型更好地模拟负荷数据，并提供了一种新的损失函数来最小化后续任务的成本。 |
| [^31] | [Scattering Spectra Models for Physics.](http://arxiv.org/abs/2306.17210) | 本文介绍了物理学中的散射谱模型，用于描述各种场的统计特性。这些模型基于散射系数的协方差，结合了场的小波分解和点位模，能够准确且稳健地重现标准统计量，捕捉了关键特性。 |
| [^32] | [A Survey on Time-Series Pre-Trained Models.](http://arxiv.org/abs/2305.10716) | 本综述全面回顾了时间序列预训练模型，其中监督、无监督和自监督是主要类别。通过使用这些模型，可以克服构建大规模标记数据集的困难，提高时间序列挖掘的性能和效率。 |
| [^33] | [Federated Ensemble-Directed Offline Reinforcement Learning.](http://arxiv.org/abs/2305.03097) | 本文开发了一种名为FEDORA的联邦集成指导的离线强化学习算法，通过集成学习方法提炼客户群体的集体智慧，显著优于其他方法，包括在合并的数据汇总中进行离线强化学习，在各种复杂的连续控制环境和真实世界数据集中进行了实验。 |
| [^34] | [Global and Preference-based Optimization with Mixed Variables using Piecewise Affine Surrogates.](http://arxiv.org/abs/2302.04686) | 本文提出了一种基于分段仿射代理构建的全局和基于偏好优化算法，可解决线性约束的混合变量问题，算法通过两种探索函数可有效搜索可行域 |
| [^35] | [E-MCTS: Deep Exploration in Model-Based Reinforcement Learning by Planning with Epistemic Uncertainty.](http://arxiv.org/abs/2210.13455) | 本文提出了一种新的方法E-MCTS，通过在MCTS预测中应用表观不确定性估计，实现了模型基强化学习中的深度探索，以及规划探索策略。通过实验证明这种方法在成功的表观不确定性估计和深度探索方面表现优异。 |

# 详细

[^1]: TrustAgent: 通过代理构成实现安全可信赖的LLM代理

    TrustAgent: Towards Safe and Trustworthy LLM-based Agents through Agent Constitution

    [https://rss.arxiv.org/abs/2402.01586](https://rss.arxiv.org/abs/2402.01586)

    本文介绍了一种基于代理构成的代理框架TrustAgent，该框架通过预先规划、规划过程中和计划后检查三种策略来提高LLM代理的安全性。实验结果表明，这些方法可以有效识别和预防潜在危险。此外，还研究了安全性与使用者满意度以及模型推理能力与效率之间的关系。

    

    近年来，基于LLM的代理引起了广泛关注，但其可信度仍未得到深入探索。由于代理可以直接与物理环境交互，其可靠性和安全性至关重要。本文提出了一种基于代理构成的代理框架TrustAgent，对LLM代理的安全性维度进行了初步研究。该框架包括三种策略：预先规划策略，在生成计划之前向模型注入安全知识；规划过程中策略，在生成计划时增强安全性；计划后检查策略，通过计划后检查确保安全性。通过实验分析，我们展示了这些方法如何通过识别和预防潜在危险有效提高LLM代理的安全性。此外，我们还探讨了安全性与使用者满意度之间的复杂关系，以及模型的推理能力与其效率之间的关联。

    The emergence of LLM-based agents has garnered considerable attention, yet their trustworthiness remains an under-explored area. As agents can directly interact with the physical environment, their reliability and safety is critical. This paper presents an Agent-Constitution-based agent framework, TrustAgent, an initial investigation into improving the safety dimension of trustworthiness in LLM-based agents. This framework consists of threefold strategies: pre-planning strategy which injects safety knowledge to the model prior to plan generation, in-planning strategy which bolsters safety during plan generation, and post-planning strategy which ensures safety by post-planning inspection. Through experimental analysis, we demonstrate how these approaches can effectively elevate an LLM agent's safety by identifying and preventing potential dangers. Furthermore, we explore the intricate relationships between safety and helpfulness, and between the model's reasoning ability and its efficac
    
[^2]: 基于内核化斯坦不相容性的数据中心预测解释

    Data-centric Prediction Explanation via Kernelized Stein Discrepancy

    [https://arxiv.org/abs/2403.15576](https://arxiv.org/abs/2403.15576)

    该论文提出了一种基于内核化斯坦不相容性的数据中心预测解释方法，通过利用内核函数识别提供最佳预测支持给测试点的训练样本，取得了优异性能。

    

    现有的基于示例的预测解释方法通常通过模型的参数或潜在表示来连接测试和训练数据点。尽管这些方法提供了有关模型预测原因的线索，但它们经常表现出固有的缺陷，比如产生显着的计算开销或生成粗粒度的解释。本文提出了一种高精度和数据中心的解释（HD-Explain），这是一种利用内核化斯坦不相容性（KSD）属性的简单预测解释方法。具体来说，KSD唯一地为经过训练的模型定义了一个参数化的内核函数，用于编码与模型相关的数据相关性。通过利用内核函数，可以有效地识别提供最佳预测支持给测试点的训练样本。我们在多个分类领域进行了彻底的分析和实验，结果表明HD-Explain取得了优异的性能。

    arXiv:2403.15576v1 Announce Type: new  Abstract: Existing example-based prediction explanation methods often bridge test and training data points through the model's parameters or latent representations. While these methods offer clues to the causes of model predictions, they often exhibit innate shortcomings, such as incurring significant computational overhead or producing coarse-grained explanations. This paper presents a Highly-precise and Data-centric Explanation (HD-Explain), a straightforward prediction explanation method exploiting properties of Kernelized Stein Discrepancy (KSD). Specifically, the KSD uniquely defines a parameterized kernel function for a trained model that encodes model-dependent data correlation. By leveraging the kernel function, one can identify training samples that provide the best predictive support to a test point efficiently. We conducted thorough analyses and experiments across multiple classification domains, where we show that HD-Explain outperform
    
[^3]: 分散式专家混合模型的实现

    Scattered Mixture-of-Experts Implementation

    [https://arxiv.org/abs/2403.08245](https://arxiv.org/abs/2403.08245)

    ScatterMoE是一种在GPU上实现的稀疏专家混合模型，通过避免填充和过多复制输入，提高了推理和训练速度，并减少了内存占用。

    

    我们提出了ScatterMoE，这是一种在GPU上实现的稀疏专家混合模型（SMoE）。ScatterMoE在现有实现的基础上构建，克服了一些限制以提高推理和训练速度，并减少内存占用。该实现通过避免填充和过多复制输入来实现这一目标。我们介绍了ParallelLinear，这是我们用来构建实现的主要组件，以及用于加速操作的各种内核。我们对我们的实现进行了与Megablocks的基准测试，并展示它可以实现更高的吞吐量和更低的内存占用。我们还展示了ParallelLinear如何通过展示Mixture of Attention的实现来扩展专家混合模型的概念。

    arXiv:2403.08245v1 Announce Type: new  Abstract: We present ScatterMoE, an implementation of Sparse Mixture-of-Experts (SMoE) on GPUs. ScatterMoE builds upon existing implementations, and overcoming some of the limitations to improve inference and training speed, and memory footprint. This implementation achieves this by avoiding padding and making excessive copies of the input.   We introduce ParallelLinear, the main component we use to build our implementation and the various kernels used to speed up the operation. We benchmark our implementation against Megablocks, and show that it enables a higher throughput and lower memory footprint. We also show how ParallelLinear enables extension of the Mixture-of-Experts concept by demonstrating with an implementation of Mixture of Attention.
    
[^4]: 具有约束扩散模型的高效和保证安全的非凸轨迹优化

    Efficient and Guaranteed-Safe Non-Convex Trajectory Optimization with Constrained Diffusion Model

    [https://arxiv.org/abs/2403.05571](https://arxiv.org/abs/2403.05571)

    本文提出了一种具有约束扩散模型的高效和保证安全的非凸轨迹优化框架，通过结合扩散模型和数值求解器，保证了计算效率和约束满足。

    

    机器人轨迹优化面临一个具有挑战性的非凸问题，这是由于复杂的动力学和环境设置造成的。本文引入了一个通用且完全可并行化的框架，将扩散模型和数值求解器结合起来，用于非凸轨迹优化，确保计算效率和约束满足。提出了一种新颖的带有额外约束违反损失的约束扩散模型进行训练。它旨在在采样过程中近似局部最优解的分布，同时最小化约束违反。然后用样本作为数值求解器的初始猜测，来优化并得出最终解，并验证可行性和最优性。

    arXiv:2403.05571v1 Announce Type: cross  Abstract: Trajectory optimization in robotics poses a challenging non-convex problem due to complex dynamics and environmental settings. Traditional numerical optimization methods are time-consuming in finding feasible solutions, whereas data-driven approaches lack safety guarantees for the output trajectories. In this paper, we introduce a general and fully parallelizable framework that combines diffusion models and numerical solvers for non-convex trajectory optimization, ensuring both computational efficiency and constraint satisfaction. A novel constrained diffusion model is proposed with an additional constraint violation loss for training. It aims to approximate the distribution of locally optimal solutions while minimizing constraint violations during sampling. The samples are then used as initial guesses for a numerical solver to refine and derive final solutions with formal verification of feasibility and optimality. Experimental evalua
    
[^5]: Aligners: 解耦LLMs和对齐

    Aligners: Decoupling LLMs and Alignment

    [https://arxiv.org/abs/2403.04224](https://arxiv.org/abs/2403.04224)

    提出了一种通过训练对齐器模型来解耦大型语言模型（LLMs）和对齐，以减少对齐对性能的潜在负面影响。

    

    大型语言模型（LLMs）需要与人类期望对齐，以确保它们在大多数应用中的安全性和实用性。对齐具有挑战性，成本高昂，并且需要为每个LLM和对齐标准重复进行。我们建议通过训练可以根据需要用于对齐给定标准的任何LLM的对齐模型来解耦LLMs和对齐，从而在一定程度上减少对性能的潜在负面影响。我们提出的对齐模型训练配方仅依赖于使用（提示的）LLM 生成的合成数据，并且可以轻松调整以适应各种对齐标准。我们通过训练一个“道德”对齐器并在实验上验证其有效性来阐明我们的方法。

    arXiv:2403.04224v1 Announce Type: cross  Abstract: Large Language Models (LLMs) need to be aligned with human expectations to ensure their safety and utility in most applications. Alignment is challenging, costly, and needs to be repeated for every LLM and alignment criterion. We propose to decouple LLMs and alignment by training aligner models that can be used to align any LLM for a given criteria on an as-needed basis, thus also reducing the potential negative impacts of alignment on performance. Our recipe for training the aligner models solely relies on synthetic data generated with a (prompted) LLM and can be easily adjusted for a variety of alignment criteria. We illustrate our method by training an "ethical" aligner and verify its efficacy empirically.
    
[^6]: GreenLLaMA: 一种带有解释的解毒框架

    GreenLLaMA: A Framework for Detoxification with Explanations

    [https://arxiv.org/abs/2402.15951](https://arxiv.org/abs/2402.15951)

    GreenLLaMA是一种全面的端到端解毒框架，通过跨平台语料库训练出的模型优于当前最先进的模型。

    

    先前关于解毒的研究工作分散在某种程度上，因为它们并没有涵盖到真实场景中所需的所有解毒方面。值得注意的是，先前的研究将开发解毒模型的任务局限在仅见过的平台子集上，没有探讨模型在未知平台上的表现如何。此外，这些工作没有解决不可解毒性这一现象，即毒性文本无法在不改变含义的情况下进行解毒。我们提出了GreenLLaMA，这是第一个全面的端到端解毒框架，旨在减轻上述限制。我们首先介绍了一个跨平台伪并行语料库，应用多步数据处理和生成策略利用ChatGPT。然后，我们使用跨平台语料库训练一套解毒模型。我们展示了我们的解毒模型优于使用人工注释的最先进模型的表现。

    arXiv:2402.15951v1 Announce Type: cross  Abstract: Prior works on detoxification are scattered in the sense that they do not cover all aspects of detoxification needed in a real-world scenario. Notably, prior works restrict the task of developing detoxification models to only a seen subset of platforms, leaving the question of how the models would perform on unseen platforms unexplored. Additionally, these works do not address non-detoxifiability, a phenomenon whereby the toxic text cannot be detoxified without altering the meaning. We propose GreenLLaMA, the first comprehensive end-to-end detoxification framework, which attempts to alleviate the aforementioned limitations. We first introduce a cross-platform pseudo-parallel corpus applying multi-step data processing and generation strategies leveraging ChatGPT. We then train a suite of detoxification models with our cross-platform corpus. We show that our detoxification models outperform the SoTA model trained with human-annotated par
    
[^7]: 划分还是征服？你应该提炼LLM的哪一部分？

    Divide-or-Conquer? Which Part Should You Distill Your LLM?

    [https://arxiv.org/abs/2402.15000](https://arxiv.org/abs/2402.15000)

    本文提出了一种将推理任务分解为问题分解阶段和问题解决阶段的策略，发现问题分解阶段相比问题解决更容易提炼为较小模型，并证实该策略胜过单阶段解决方案。

    

    最近的研究表明，大型语言模型（LLMs）在被鼓励先解决主要任务的子任务时可以更好地解决推理任务。本文设计了一种类似的策略，将推理任务分解为问题分解阶段和问题解决阶段，并展示该策略能够胜过单阶段解决方案。此外，我们假设与解决问题相比，分解阶段更容易被提炼为较小的模型，因为后者需要大量的领域知识，而前者只需要学习一般的问题解决策略。我们提出了提炼这两种能力的方法，并评估了它们对推理结果和推理成本的影响。我们发现我们可以提炼问题分解阶段，并同时在任务、数据集和模型之间实现良好的泛化。然而，要提炼问题解决阶段就更困难了。

    arXiv:2402.15000v1 Announce Type: new  Abstract: Recent methods have demonstrated that Large Language Models (LLMs) can solve reasoning tasks better when they are encouraged to solve subtasks of the main task first. In this paper we devise a similar strategy that breaks down reasoning tasks into a problem decomposition phase and a problem solving phase and show that the strategy is able to outperform a single stage solution. Further, we hypothesize that the decomposition should be easier to distill into a smaller model compared to the problem solving because the latter requires large amounts of domain knowledge while the former only requires learning general problem solving strategies. We propose methods to distill these two capabilities and evaluate their impact on reasoning outcomes and inference cost. We find that we can distill the problem decomposition phase and at the same time achieve good generalization across tasks, datasets, and models. However, it is harder to distill the pr
    
[^8]: LexC-Gen: 利用大型语言模型和双语词汇表为极低资源语言生成数据

    LexC-Gen: Generating Data for Extremely Low-Resource Languages with Large Language Models and Bilingual Lexicons

    [https://arxiv.org/abs/2402.14086](https://arxiv.org/abs/2402.14086)

    LexC-Gen提出了一种词典条件数据生成方法，可以以大规模生成低资源语言分类任务数据，取得了较好的效果。

    

    低资源语言的数据匮乏可以通过利用双语词典中从高资源语言的标记任务数据进行逐字翻译来解决，然而，双语词典通常与任务数据有限的词汇重叠，导致翻译覆盖和词典利用不佳。我们提出了一种称为LexC-Gen的词典条件数据生成方法，该方法可以大规模生成低资源语言分类任务数据。具体而言，LexC-Gen首先使用双语词典中的高资源语言单词生成与词典兼容的任务数据，然后通过单词翻译将其翻译成低资源语言。在17种极低资源语言中，LexC-Gen生成的数据在性能上与专家翻译的黄金数据竞争力相当，并且在情感分析和主题分类上平均比现有的基于词典的单词翻译方法提高了5.6和8.9个分数。

    arXiv:2402.14086v1 Announce Type: cross  Abstract: Data scarcity in low-resource languages can be addressed with word-to-word translations from labeled task data in high-resource languages using bilingual lexicons. However, bilingual lexicons often have limited lexical overlap with task data, which results in poor translation coverage and lexicon utilization. We propose lexicon-conditioned data generation (LexC-Gen), a method that generates low-resource-language classification task data at scale. Specifically, LexC-Gen first uses high-resource-language words from bilingual lexicons to generate lexicon-compatible task data, and then it translates them into low-resource languages with bilingual lexicons via word translation. Across 17 extremely low-resource languages, LexC-Gen generated data is competitive with expert-translated gold data, and yields on average 5.6 and 8.9 points improvement over existing lexicon-based word translation methods on sentiment analysis and topic classificati
    
[^9]: 平均梯度外积作为深度神经坍塌机制的研究

    Average gradient outer product as a mechanism for deep neural collapse

    [https://arxiv.org/abs/2402.13728](https://arxiv.org/abs/2402.13728)

    本文通过提供证据表明，深度神经网络中的神经坍塌主要是通过平均梯度外积进行深度特征学习的，权重的奇异结构与AGOP高度相关，导致类内变异坍塌。

    

    Deep Neural Collapse (DNC)指的是深度神经网络(DNNs)最后几层数据表示的惊人刚性结构。尽管这种现象在各种情境中都得到了测量，但其出现只有部分被理解。本文提供了充分证据，表明DNC主要是通过平均梯度外积(AGOP)进行深度特征学习而发生的。相比于解释神经坍塌的特征不可知方法，如无约束特征模型，这一进展更进一步。我们继续提供证据表明，权重的右奇异向量和奇异值是DNN中类内变异坍塌的主要因素。正如最近的研究所示，这种奇异结构与AGOP的高度相关。然后我们在实验和理论上证明了AGOP在随机初始化的神经网络中引发神经坍塌。

    arXiv:2402.13728v1 Announce Type: new  Abstract: Deep Neural Collapse (DNC) refers to the surprisingly rigid structure of the data representations in the final layers of Deep Neural Networks (DNNs). Though the phenomenon has been measured in a wide variety of settings, its emergence is only partially understood. In this work, we provide substantial evidence that DNC formation occurs primarily through deep feature learning with the average gradient outer product (AGOP). This takes a step further compared to efforts that explain neural collapse via feature-agnostic approaches, such as the unconstrained features model. We proceed by providing evidence that the right singular vectors and values of the weights are responsible for the majority of within-class variability collapse in DNNs. As shown in recent work, this singular structure is highly correlated with that of the AGOP. We then establish experimentally and theoretically that AGOP induces neural collapse in a randomly initialized ne
    
[^10]: 软最大概率（大部分时候）在多项选择问答任务中预测大型语言模型的正确性

    Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A

    [https://arxiv.org/abs/2402.13213](https://arxiv.org/abs/2402.13213)

    多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。

    

    尽管大型语言模型（LLMs）在许多任务上表现出色，但过度自信仍然是一个问题。我们假设在多项选择问答任务中，错误答案将与最大softmax概率（MSPs）较小相关，相比之下正确答案较大。我们在十个开源LLMs和五个数据集上全面评估了这一假设，在表现良好的原始问答任务中发现了对我们假设的强有力证据。对于表现最佳的六个LLMs，从MSP导出的AUROC在59/60个实例中都优于随机机会，p < 10^{-4}。在这六个LLMs中，平均AUROC范围在60%至69%之间。利用这些发现，我们提出了一个带有弃权选项的多项选择问答任务，并展示通过根据初始模型响应的MSP有选择地弃权可以提高性能。我们还用预softmax logits而不是softmax进行了相同的实验。

    arXiv:2402.13213v1 Announce Type: cross  Abstract: Although large language models (LLMs) perform impressively on many tasks, overconfidence remains a problem. We hypothesized that on multiple-choice Q&A tasks, wrong answers would be associated with smaller maximum softmax probabilities (MSPs) compared to correct answers. We comprehensively evaluate this hypothesis on ten open-source LLMs and five datasets, and find strong evidence for our hypothesis among models which perform well on the original Q&A task. For the six LLMs with the best Q&A performance, the AUROC derived from the MSP was better than random chance with p < 10^{-4} in 59/60 instances. Among those six LLMs, the average AUROC ranged from 60% to 69%. Leveraging these findings, we propose a multiple-choice Q&A task with an option to abstain and show that performance can be improved by selectively abstaining based on the MSP of the initial model response. We also run the same experiments with pre-softmax logits instead of sof
    
[^11]: 对齐您的意图：通过最优传输的离线模仿学习

    Align Your Intents: Offline Imitation Learning via Optimal Transport

    [https://arxiv.org/abs/2402.13037](https://arxiv.org/abs/2402.13037)

    通过最优传输的离线模仿学习方法AILOT，可以在缺乏明确奖励的情况下，仅通过观察专家学习所需的行为。

    

    离线强化学习（RL）通过学习预先收集的数据来解决顺序决策问题，而无需与环境进行交互。我们展示出，即使缺乏明确的奖励或动作标签，模仿代理也可以仅通过观察专家来学习所需的行为。在我们的方法AILOT（通过最优传输对齐模仿学习）中，我们使用意图的特殊状态表示形式，其中包含数据内的两两空间距离。在给定这种表示形式的情况下，我们通过专家和代理轨迹之间的最优传输距离定义内在奖励函数。我们报告称AILOT在D4RL基准测试上优于最先进的离线模仿学习算法。

    arXiv:2402.13037v1 Announce Type: cross  Abstract: Offline reinforcement learning (RL) addresses the problem of sequential decision-making by learning optimal policy through pre-collected data, without interacting with the environment. As yet, it has remained somewhat impractical, because one rarely knows the reward explicitly and it is hard to distill it retrospectively. Here, we show that an imitating agent can still learn the desired behavior merely from observing the expert, despite the absence of explicit rewards or action labels. In our method, AILOT (Aligned Imitation Learning via Optimal Transport), we involve special representation of states in a form of intents that incorporate pairwise spatial distances within the data. Given such representations, we define intrinsic reward function via optimal transport distance between the expert's and the agent's trajectories. We report that AILOT outperforms state-of-the art offline imitation learning algorithms on D4RL benchmarks and im
    
[^12]: 在摘要中识别事实不一致性：朝向大型语言模型的有效利用

    Identifying Factual Inconsistency in Summaries: Towards Effective Utilization of Large Language Model

    [https://arxiv.org/abs/2402.12821](https://arxiv.org/abs/2402.12821)

    该研究提出了针对摘要中事实不一致性的解决方案：通过大型语言模型在正确的范式设计下无需训练即可解决任务，并提出了训练策略以精炼更小型的高准确性的语言模型。

    

    事实上的不一致性对抽象性摘要生成器的商业部署构成重要障碍。本研究围绕两个重要问题展开：如何最好地利用大型语言模型来检测事实不一致性，以及如何精炼一个同时具有高效性和功效性的更小型语言模型？首先提出并评估了三种零样本范式，跨越五个不同数据集：直接推理整个摘要或每个摘要窗口；通过问题生成和回答进行实体验证。实验表明，在适当的范式设计下，语言模型本身能够在无需训练的情况下解决这一任务，平均超过强大的训练基线2.8%。为进一步促进实用性，我们提出针对精炼更小的开源语言模型的训练策略，该模型可以一次性高准确地评分整个摘要，胜过零

    arXiv:2402.12821v1 Announce Type: new  Abstract: Factual inconsistency poses a significant hurdle for the commercial deployment of abstractive summarizers. Under this Large Language Model (LLM) era, this work focuses around two important questions: what is the best way to leverage LLM for factual inconsistency detection, and how could we distill a smaller LLM with both high efficiency and efficacy? Three zero-shot paradigms are firstly proposed and evaluated across five diverse datasets: direct inference on the entire summary or each summary window; entity verification through question generation and answering. Experiments suggest that LLM itself is capable to resolve this task train-free under the proper paradigm design, surpassing strong trained baselines by 2.8% on average. To further promote practical utility, we then propose training strategies aimed at distilling smaller open-source LLM that learns to score the entire summary at once with high accuracy, which outperforms the zero
    
[^13]: 通过扩散模型生成的假设的统计检验

    Statistical Test for Generated Hypotheses by Diffusion Models

    [https://arxiv.org/abs/2402.11789](https://arxiv.org/abs/2402.11789)

    本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。

    

    AI的增强性能加速了其融入科学研究。特别是，利用生成式AI创建科学假设是很有前途的，并且正在越来越多地应用于各个领域。然而，当使用AI生成的假设进行关键决策（如医学诊断）时，验证它们的可靠性至关重要。在本研究中，我们考虑使用扩散模型生成的图像进行医学诊断任务，并提出了一种统计检验来量化其可靠性。所提出的统计检验的基本思想是使用选择性推断框架，我们考虑在生成的图像是由经过训练的扩散模型产生的这一事实条件下的统计检验。利用所提出的方法，医学图像诊断结果的统计可靠性可以以p值的形式量化，从而实现在控制错误率的情况下进行决策。

    arXiv:2402.11789v1 Announce Type: cross  Abstract: The enhanced performance of AI has accelerated its integration into scientific research. In particular, the use of generative AI to create scientific hypotheses is promising and is increasingly being applied across various fields. However, when employing AI-generated hypotheses for critical decisions, such as medical diagnoses, verifying their reliability is crucial. In this study, we consider a medical diagnostic task using generated images by diffusion models, and propose a statistical test to quantify its reliability. The basic idea behind the proposed statistical test is to employ a selective inference framework, where we consider a statistical test conditional on the fact that the generated images are produced by a trained diffusion model. Using the proposed method, the statistical reliability of medical image diagnostic results can be quantified in the form of a p-value, allowing for decision-making with a controlled error rate. 
    
[^14]: 避免连续空间中的灾难：通过寻求帮助

    Avoiding Catastrophe in Continuous Spaces by Asking for Help

    [https://arxiv.org/abs/2402.08062](https://arxiv.org/abs/2402.08062)

    在连续空间中，通过寻求帮助来避免灾难。引入了一种上下文多臂赌博问题的变体，目标是最小化灾难发生的概率。提出了一种算法，在连续1D状态空间和相对简单的回报函数下，遗憾和向导师查询率都趋近于0。

    

    大多数具有正式遗憾保证的强化学习算法假设所有错误都是可逆的，并依赖于尝试所有可能的选项。当一些错误是无法修复甚至是灾难性的时，这种方法会导致糟糕的结果。我们提出了一种上下文多臂赌博问题的变体，在这个问题中，目标是最小化发生灾难的概率。具体而言，我们假设每轮的回报代表了在该轮避免灾难的概率，并尝试最大化回报的乘积（总体避免灾难的概率）。为了给 agent 一些成功的机会，我们允许有限次向导师提问，并假设回报函数为 Lipschitz 连续的。我们提出了一种算法，当时间跨度增长时，它的遗憾和向导师查询率都趋近于 0，假设是一个连续的 1D 状态空间和相对"简单"的回报函数。我们还提供了一个匹配的下界：在没有简单性假设的情况下，任何算法要么不断查询异常的行为，要么每次查询完全相同的行为。

    Most reinforcement learning algorithms with formal regret guarantees assume all mistakes are reversible and rely on essentially trying all possible options. This approach leads to poor outcomes when some mistakes are irreparable or even catastrophic. We propose a variant of the contextual bandit problem where the goal is to minimize the chance of catastrophe. Specifically, we assume that the payoff each round represents the chance of avoiding catastrophe that round, and try to maximize the product of payoffs (the overall chance of avoiding catastrophe). To give the agent some chance of success, we allow a limited number of queries to a mentor and assume a Lipschitz continuous payoff function. We present an algorithm whose regret and rate of querying the mentor both approach 0 as the time horizon grows, assuming a continuous 1D state space and a relatively "simple" payoff function. We also provide a matching lower bound: without the simplicity assumption: any algorithm either constantly
    
[^15]: 动态系统的基础推理模型

    Foundational Inference Models for Dynamical Systems

    [https://arxiv.org/abs/2402.07594](https://arxiv.org/abs/2402.07594)

    本研究提出了一种基于监督学习的框架，用于从噪声数据中零样本推理动态系统的普通微分方程（ODE）。通过生成大型ODE数据集，并利用神经网络将噪声观察和初始条件以及向量场进行映射，得到称为基础推理模型（FIM）的结果模型。这些模型可以复制、匹配和组合，用于构建任何维度的推理模型。

    

    普通微分方程（ODE）构成了作为自然和社会现象模型的动态系统的基础。然而，推断出最佳描述给定现象的一组噪声观察的ODE可能非常具有挑战性，现有的模型往往也非常专业化和复杂。在这项工作中，我们提出了一种新颖的监督式学习框架，用于从噪声数据中零样本推理ODE。我们首先通过对初始条件空间和定义它们的向量场空间的分布进行采样，生成大型一维ODE数据集。然后，我们学习将这些方程的解的噪声观察与其相应的初始条件和向量场之间的神经映射。我们将结果模型称为基础推理模型（FIM），它们可以（i）沿时间维复制和匹配以增加分辨率；（ii）复制和组合以构建任何维度的推理模型。

    Ordinary differential equations (ODEs) underlie dynamical systems which serve as models for a vast number of natural and social phenomena. Yet inferring the ODE that best describes a set of noisy observations on one such phenomenon can be remarkably challenging, and the models available to achieve it tend to be highly specialized and complex too. In this work we propose a novel supervised learning framework for zero-shot inference of ODEs from noisy data. We first generate large datasets of one-dimensional ODEs, by sampling distributions over the space of initial conditions, and the space of vector fields defining them. We then learn neural maps between noisy observations on the solutions of these equations, and their corresponding initial condition and vector fields. The resulting models, which we call foundational inference models (FIM), can be (i) copied and matched along the time dimension to increase their resolution; and (ii) copied and composed to build inference models of any d
    
[^16]: 通过最优输运实现具有任意大小约束的图割

    Graph Cuts with Arbitrary Size Constraints Through Optimal Transport

    [https://arxiv.org/abs/2402.04732](https://arxiv.org/abs/2402.04732)

    本论文提出了一种新的图割算法，通过将图割问题制定为正则化的Gromov-Wasserstein问题，并使用加速的近端GD算法解决，实现在任意大小约束下对图进行分割，在非平衡数据集聚类等应用中具有更高的效率。

    

    图的常见分割方法是最小割。经典最小割方法的一个缺点是它们倾向于生成小的分组，这就是为什么更平衡的变体，如归一化割和比例割取得了更多的成功。然而，我们认为对于某些应用，如非平衡数据集的聚类，这些变体的平衡约束可能过于限制，而对于寻找完美平衡分区来说不够限制。在这里，我们提出了一种新的图割算法，用于在任意大小约束下对图进行分割。我们将图割问题制定为正则化的Gromov-Wasserstein问题。然后，我们提出使用加速的近端GD算法来解决它，该算法具有全局收敛性保证，产生稀疏解，并且只比经典谱聚类算法多消耗$\mathcal{O}(\log(n))$的附加比率，但效率更高。

    A common way of partitioning graphs is through minimum cuts. One drawback of classical minimum cut methods is that they tend to produce small groups, which is why more balanced variants such as normalized and ratio cuts have seen more success. However, we believe that with these variants, the balance constraints can be too restrictive for some applications like for clustering of imbalanced datasets, while not being restrictive enough for when searching for perfectly balanced partitions. Here, we propose a new graph cut algorithm for partitioning graphs under arbitrary size constraints. We formulate the graph cut problem as a regularized Gromov-Wasserstein problem. We then propose to solve it using accelerated proximal GD algorithm which has global convergence guarantees, results in sparse solutions and only incurs an additional ratio of $\mathcal{O}(\log(n))$ compared to the classical spectral clustering algorithm but was seen to be more efficient.
    
[^17]: 密集乘法物理信息神经网络

    Densely Multiplied Physics Informed Neural Network

    [https://arxiv.org/abs/2402.04390](https://arxiv.org/abs/2402.04390)

    该论文通过改进神经网络架构，提出了一种密集乘法物理信息神经网络（DM-PINN）架构，它有效利用隐藏层的输出，显著提高了PINN的准确性和性能。

    

    尽管物理信息神经网络（Physics-Informed Neural Networks, PINNs）在处理非线性偏微分方程（PDEs）方面显示出巨大潜力，但常常会出现精度不足或获取不正确结果的问题。与大多数现有的解决方案不同，该论文改进了神经网络架构以提高PINN的性能。我们提出了一种密集乘法PINN（DM-PINN）架构，它将隐藏层的输出与所有后面的隐藏层的输出相乘。在不引入更多可训练参数的情况下，该有效机制可以显著提高PINN的准确性。所提出的架构在四个基准示例（Allan-Cahn方程，Helmholtz方程，Burgers方程和1D对流方程）上进行了评估。将所提出的架构与不同的PINN结构进行比较，证明了其卓越的性能。

    Although physics-informed neural networks (PINNs) have shown great potential in dealing with nonlinear partial differential equations (PDEs), it is common that PINNs will suffer from the problem of insufficient precision or obtaining incorrect outcomes. Unlike most of the existing solutions trying to enhance the ability of PINN by optimizing the training process, this paper improved the neural network architecture to improve the performance of PINN. We propose a densely multiply PINN (DM-PINN) architecture, which multiplies the output of a hidden layer with the outputs of all the behind hidden layers. Without introducing more trainable parameters, this effective mechanism can significantly improve the accuracy of PINNs. The proposed architecture is evaluated on four benchmark examples (Allan-Cahn equation, Helmholtz equation, Burgers equation and 1D convection equation). Comparisons between the proposed architecture and different PINN structures demonstrate the superior performance of 
    
[^18]: MetaOptimize：一个优化步长和其他元参数的框架

    MetaOptimize: A Framework for Optimizing Step Sizes and Other Meta-parameters

    [https://arxiv.org/abs/2402.02342](https://arxiv.org/abs/2402.02342)

    MetaOptimize是一个框架，通过动态调整学习率来优化机器学习算法中的元参数，以提高训练效率和模型性能。

    

    本文解决了机器学习算法中优化元参数（即超参数）的挑战，这是影响训练效率和模型性能的关键因素。我们引入了MetaOptimize框架，摆脱了计算昂贵的传统元参数搜索方法，通过动态调整元参数，特别是步长（也称为学习率），来训练模型。具体而言，MetaOptimize可以适用于任何一阶优化算法，在训练过程中实时调整步长，通过未来损失的折现总和来最小化一种特定形式的遗憾。我们还介绍了MetaOptimize的低复杂度变体，结合其适应多个优化算法的能力，展示了在各种机器学习应用中与手工设计的学习率计划相媲美的性能。

    This paper addresses the challenge of optimizing meta-parameters (i.e., hyperparameters) in machine learning algorithms, a critical factor influencing training efficiency and model performance. Moving away from the computationally expensive traditional meta-parameter search methods, we introduce MetaOptimize framework that dynamically adjusts meta-parameters, particularly step sizes (also known as learning rates), during training. More specifically, MetaOptimize can wrap around any first-order optimization algorithm, tuning step sizes on the fly to minimize a specific form of regret that accounts for long-term effect of step sizes on training, through a discounted sum of future losses. We also introduce low complexity variants of MetaOptimize that, in conjunction with its adaptability to multiple optimization algorithms, demonstrate performance competitive to those of best hand-crafted learning rate schedules across various machine learning applications.
    
[^19]: LLMs学习动力系统的控制原理，揭示了上下文中的神经比例定律

    LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law

    [https://arxiv.org/abs/2402.00795](https://arxiv.org/abs/2402.00795)

    本文研究了预训练语言模型LLMs对动力系统行为的外推能力，发现LLaMA 2能够准确预测动力系统的时间序列。此外，输入上下文窗口的长度越长，学习到的物理规则的准确性越高，揭示了一种上下文中的神经比例定律。

    

    预训练的大型语言模型（LLMs）在零-shot任务，包括时间序列预测方面表现出惊人的有效性。然而，由于模型的复杂性，理解其背后的机制仍然极具挑战性。本文研究了LLMs对受物理原理控制的动力系统行为的外推能力。我们的结果表明，主要在文本上进行训练的语言模型LLaMA 2在没有微调或提示工程的情况下，能够准确预测动力系统的时间序列。此外，学习到的物理规则的准确性随着输入上下文窗口的长度增加而增加，揭示了一种上下文中的神经比例定律。同时，我们还提出了一种灵活高效的算法，用于直接从LLMs中提取多位数的概率密度函数。

    Pretrained large language models (LLMs) are surprisingly effective at performing zero-shot tasks, including time-series forecasting. However, understanding the mechanisms behind such capabilities remains highly challenging due to the complexity of the models. In this paper, we study LLMs' ability to extrapolate the behavior of dynamical systems whose evolution is governed by principles of physical interest. Our results show that LLaMA 2, a language model trained primarily on texts, achieves accurate predictions of dynamical system time series without fine-tuning or prompt engineering. Moreover, the accuracy of the learned physical rules increases with the length of the input context window, revealing an in-context version of neural scaling law. Along the way, we present a flexible and efficient algorithm for extracting probability density functions of multi-digit numbers directly from LLMs.
    
[^20]: 不同的循环内核拓展到不同的储备计算拓扑的研究

    Extension of Recurrent Kernels to different Reservoir Computing topologies. (arXiv:2401.14557v1 [cs.LG])

    [http://arxiv.org/abs/2401.14557](http://arxiv.org/abs/2401.14557)

    该研究通过提供特定RC体系结构与相应循环内核形式等价性的经验分析，填补了Leaky RC、Sparse RC和Deep RC等已建立的RC范例尚未进行分析的空白。此外，研究还揭示了稀疏连接在RC体系结构中的作用，并提出了一种依赖储备大小的最佳稀疏性水平。最后，该研究的系统分析表明，在Deep RC模型中，通过减小尺寸的连续储备可以更好地实现收敛。

    

    近年来，由于其快速高效的计算能力，储备计算（RC）变得越来越受欢迎。标准的RC在渐近极限下已被证明与循环内核等效，这有助于分析其表达能力。然而，许多已建立的RC范例，如Leaky RC、Sparse RC和Deep RC，尚未以这种方式进行分析。本研究旨在通过提供特定RC体系结构与相应循环内核形式等价性的经验分析来填补这一空白。我们通过改变每个体系结构中实施的激活函数进行收敛研究。我们的研究还揭示了稀疏连接在RC体系结构中的作用，并提出了一种依赖储备大小的最佳稀疏性水平。此外，我们的系统分析表明，在Deep RC模型中，通过减小尺寸的连续储备可以更好地实现收敛。

    Reservoir Computing (RC) has become popular in recent years due to its fast and efficient computational capabilities. Standard RC has been shown to be equivalent in the asymptotic limit to Recurrent Kernels, which helps in analyzing its expressive power. However, many well-established RC paradigms, such as Leaky RC, Sparse RC, and Deep RC, are yet to be analyzed in such a way. This study aims to fill this gap by providing an empirical analysis of the equivalence of specific RC architectures with their corresponding Recurrent Kernel formulation. We conduct a convergence study by varying the activation function implemented in each architecture. Our study also sheds light on the role of sparse connections in RC architectures and propose an optimal sparsity level that depends on the reservoir size. Furthermore, our systematic analysis shows that in Deep RC models, convergence is better achieved with successive reservoirs of decreasing sizes.
    
[^21]: MITS-GAN: 用生成对抗网络保护医学影像免受篡改

    MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks. (arXiv:2401.09624v1 [eess.IV])

    [http://arxiv.org/abs/2401.09624](http://arxiv.org/abs/2401.09624)

    MITS-GAN是一种用于保护医学影像免受篡改的新方法，通过引入适当的高斯噪声作为防护措施，打乱攻击者的生成对抗网络架构的输出。实验结果表明MITS-GAN能够生成耐篡改图像，具有优越的性能。

    

    生成模型，特别是生成对抗网络（GANs），在图像生成方面取得了进展，但也引发了潜在的恶意使用的担忧，尤其是在医学影像等敏感领域。本研究提出了一种新颖的方法MITS-GAN，用于防止医学影像中的篡改，特别关注CT扫描。该方法通过引入不可察觉但精确的扰动来打乱攻击者的CT-GAN架构的输出。具体而言，所提出的方法涉及将适当的高斯噪声引入到输入中作为对各种攻击的保护措施。我们的方法旨在提高防篡改能力，与现有技术相比具有优势。对CT扫描数据集的实验结果表明MITS-GAN具有卓越的性能，强调了其能够生成具有可忽略伪影的耐篡改图像的能力。由于医学领域中的图像篡改带来了危及生命的风险，我们的主动防护方法具有重要意义。

    The progress in generative models, particularly Generative Adversarial Networks (GANs), opened new possibilities for image generation but raised concerns about potential malicious uses, especially in sensitive areas like medical imaging. This study introduces MITS-GAN, a novel approach to prevent tampering in medical images, with a specific focus on CT scans. The approach disrupts the output of the attacker's CT-GAN architecture by introducing imperceptible but yet precise perturbations. Specifically, the proposed approach involves the introduction of appropriate Gaussian noise to the input as a protective measure against various attacks. Our method aims to enhance tamper resistance, comparing favorably to existing techniques. Experimental results on a CT scan dataset demonstrate MITS-GAN's superior performance, emphasizing its ability to generate tamper-resistant images with negligible artifacts. As image tampering in medical domains poses life-threatening risks, our proactive approac
    
[^22]: 将质量多样性与描述符条件加强学习相结合

    Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning. (arXiv:2401.08632v1 [cs.NE])

    [http://arxiv.org/abs/2401.08632](http://arxiv.org/abs/2401.08632)

    将质量多样性优化与描述符条件加强学习相结合，以克服进化算法的局限性，并在生成既多样又高性能的解决方案集合方面取得成功。

    

    智能的基本特征之一是找到新颖和有创造性的解决方案来解决给定的挑战或适应未预料到的情况。质量多样性优化是一类进化算法，可以生成既多样又高性能的解决方案集合。其中，MAP-Elites是一个著名的例子，已成功应用于各种领域，包括进化机器人学。然而，MAP-Elites通过遗传算法的随机突变进行发散搜索，因此仅限于进化低维解决方案的种群。PGA-MAP-Elites通过受深度强化学习启发的基于梯度的变异算子克服了这一限制，从而实现了大型神经网络的进化。尽管在许多环境中性能优秀，但PGA-MAP-Elites在一些任务中失败，其中基于梯度的变异算子的收敛搜索阻碍了多样性。在这项工作中，我们...

    A fundamental trait of intelligence involves finding novel and creative solutions to address a given challenge or to adapt to unforeseen situations. Reflecting this, Quality-Diversity optimization is a family of Evolutionary Algorithms, that generates collections of both diverse and high-performing solutions. Among these, MAP-Elites is a prominent example, that has been successfully applied to a variety of domains, including evolutionary robotics. However, MAP-Elites performs a divergent search with random mutations originating from Genetic Algorithms, and thus, is limited to evolving populations of low-dimensional solutions. PGA-MAP-Elites overcomes this limitation using a gradient-based variation operator inspired by deep reinforcement learning which enables the evolution of large neural networks. Although high-performing in many environments, PGA-MAP-Elites fails on several tasks where the convergent search of the gradient-based variation operator hinders diversity. In this work, we
    
[^23]: 用可比较的语料和多个参考文献进行代码翻译的数据增强

    Data Augmentation for Code Translation with Comparable Corpora and Multiple References. (arXiv:2311.00317v1 [cs.CL])

    [http://arxiv.org/abs/2311.00317](http://arxiv.org/abs/2311.00317)

    该论文介绍了两种数据增强方法来改善编程语言之间的代码翻译。通过构建可比较的语料库和增加多个参考翻译，实验结果表明这些方法显著提高了CodeT5在Java、Python和C++之间的翻译准确性。

    

    在编程语言之间进行代码翻译的一个主要挑战是平行训练数据通常有限。为了克服这个挑战，我们提出了两种数据增强技术，一种是构建可比较的语料库（即具有类似功能的代码对），另一种是用多个参考翻译来增强现有的平行数据。具体而言，我们构建并分析了多种类型的可比较的语料库，包括使用代码生成模型从自然语言文档中生成的程序。此外，为了减少对单个参考翻译的过拟合，我们自动生成了可用平行数据的额外翻译参考，并通过单元测试对翻译进行筛选，从而增加了目标翻译的变化。实验证明，我们的数据增强技术显著提高了CodeT5在Java、Python和C++之间的翻译准确性（平均提升了7.5%的计算准确性（CA@1））。

    One major challenge of translating code between programming languages is that parallel training data is often limited. To overcome this challenge, we present two data augmentation techniques, one that builds comparable corpora (i.e., code pairs with similar functionality), and another that augments existing parallel data with multiple reference translations. Specifically, we build and analyze multiple types of comparable corpora, including programs generated from natural language documentation using a code generation model. Furthermore, to reduce overfitting to a single reference translation, we automatically generate additional translation references for available parallel data and filter the translations by unit tests, which increases variation in target translations. Experiments show that our data augmentation techniques significantly improve CodeT5 for translation between Java, Python, and C++ by an average of 7.5% Computational Accuracy (CA@1), which verifies the correctness of tr
    
[^24]: 在大型语言模型时代的自动缺陷生成

    Automated Bug Generation in the era of Large Language Models. (arXiv:2310.02407v1 [cs.SE])

    [http://arxiv.org/abs/2310.02407](http://arxiv.org/abs/2310.02407)

    本论文探讨了在大型语言模型时代的自动缺陷生成问题，针对难以检测和难以修复的缺陷提出了解决方案，并分析了基于学习的技术中这两个目标的冲突。

    

    缺陷在软件工程中是至关重要的；过去几十年的许多研究已经提出了检测、定位和修复软件系统中的缺陷的方法。评估这些技术的有效性需要复杂的缺陷，即那些很难通过测试和调试来检测和修复的缺陷。从传统软件工程的角度来看，难以修复的缺陷与正确的代码在多个位置上有所差异，这使得它们难以定位和修复。而难以检测的缺陷则在特定的测试输入和可达条件下展现出来。这两个目标，即生成难以检测和难以修复的缺陷，大多数是一致的；缺陷生成技术可以将多个语句更改为仅在特定输入集合下被覆盖。然而，对于基于学习的技术来说，这两个目标是相互冲突的：一个缺陷应该有与训练数据中的正确代码相似的代码表示，以挑战缺陷预测。

    Bugs are essential in software engineering; many research studies in the past decades have been proposed to detect, localize, and repair bugs in software systems. Effectiveness evaluation of such techniques requires complex bugs, i.e., those that are hard to detect through testing and hard to repair through debugging. From the classic software engineering point of view, a hard-to-repair bug differs from the correct code in multiple locations, making it hard to localize and repair. Hard-to-detect bugs, on the other hand, manifest themselves under specific test inputs and reachability conditions. These two objectives, i.e., generating hard-to-detect and hard-to-repair bugs, are mostly aligned; a bug generation technique can change multiple statements to be covered only under a specific set of inputs. However, these two objectives are conflicting for learning-based techniques: A bug should have a similar code representation to the correct code in the training data to challenge a bug predi
    
[^25]: 通过最优输运进行从观察中的模仿学习

    Imitation Learning from Observation through Optimal Transport. (arXiv:2310.01632v1 [cs.RO])

    [http://arxiv.org/abs/2310.01632](http://arxiv.org/abs/2310.01632)

    本文提出了一种通过最优输运进行从观察中的模仿学习的方法，该方法不需要学习模型或对抗学习，可以与任何强化学习算法集成，并在各种连续控制任务上超过了现有最先进方法，在ILfO设置下实现了专家级的性能。

    

    从观察中的模仿学习（ILfO）是一种学习者试图在没有直接指导的情况下，使用观测数据模仿专家行为的设置。在本文中，我们重新审视了最优输运在IL中的应用，其中根据学习者和专家的状态轨迹之间的Wasserstein距离生成奖励。我们证明了现有方法可以简化为生成无需学习模型或对抗学习的奖励函数。与许多其他最先进的方法不同，我们的方法可以与任何强化学习算法集成，并适用于ILfO。我们在各种连续控制任务上展示了这种简单方法的有效性，并发现即使只观察单个专家轨迹而没有动作，它在ILfO设置中超过了现有最先进方法，在一系列评估领域中实现了专家级的性能。

    Imitation Learning from Observation (ILfO) is a setting in which a learner tries to imitate the behavior of an expert, using only observational data and without the direct guidance of demonstrated actions. In this paper, we re-examine the use of optimal transport for IL, in which a reward is generated based on the Wasserstein distance between the state trajectories of the learner and expert. We show that existing methods can be simplified to generate a reward function without requiring learned models or adversarial learning. Unlike many other state-of-the-art methods, our approach can be integrated with any RL algorithm, and is amenable to ILfO. We demonstrate the effectiveness of this simple approach on a variety of continuous control tasks and find that it surpasses the state of the art in the IlfO setting, achieving expert-level performance across a range of evaluation domains even when observing only a single expert trajectory without actions.
    
[^26]: 利用异构引导的客户端采样加速非独立同分布联邦学习

    Accelerating Non-IID Federated Learning via Heterogeneity-Guided Client Sampling. (arXiv:2310.00198v1 [cs.LG])

    [http://arxiv.org/abs/2310.00198](http://arxiv.org/abs/2310.00198)

    本文提出了一种名为HiCS-FL的方法，通过利用客户端的网络输出层更新来估计数据的统计异质性，并利用此信息进行客户端的聚类选择，以加速非独立同分布联邦学习。

    

    客户端设备中存在的数据的统计异质性使得在联邦学习（FL）系统中训练全局模型变得困难。尤其具有挑战性的是，在由于资源限制只有一小部分客户端能参与任何一轮FL的设置中。最近的一些方法致力于训练非独立同分布数据的FL系统中的全局模型，它们着重于开发采样更具信息更新的客户端选择方法。然而，现有的客户端选择技术要么引入了显著的计算开销，要么只在客户端具有类似异质性配置文件的情况下表现良好。本文提出了HiCS-FL（通过分层聚类采样进行联邦学习），一种新的客户端选择方法，其中服务器使用客户端网络输出层的更新来估计客户端数据的统计异质性，并依赖此信息来进行聚类。

    Statistical heterogeneity of data present at client devices in a federated learning (FL) system renders the training of a global model in such systems difficult. Particularly challenging are the settings where due to resource constraints only a small fraction of clients can participate in any given round of FL. Recent approaches to training a global model in FL systems with non-IID data have focused on developing client selection methods that aim to sample clients with more informative updates of the model. However, existing client selection techniques either introduce significant computation overhead or perform well only in the scenarios where clients have data with similar heterogeneity profiles. In this paper, we propose HiCS-FL (Federated Learning via Hierarchical Clustered Sampling), a novel client selection method in which the server estimates statistical heterogeneity of a client's data using the client's update of the network's output layer and relies on this information to clu
    
[^27]: MosaicFusion: 将扩散模型作为大词汇实例分割的数据增强器

    MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation. (arXiv:2309.13042v1 [cs.CV])

    [http://arxiv.org/abs/2309.13042](http://arxiv.org/abs/2309.13042)

    MosaicFusion是一种用于大词汇实例分割的数据增强方法，通过将扩散模型作为数据集生成器，能够生成大量合成标记数据。在实验中，我们的方法在准确率和泛化能力方面取得了显著的提升。

    

    我们提出了MosaicFusion，一种简单而有效的基于扩散的数据增强方法，用于大词汇实例分割。我们的方法无需训练，也不依赖于任何标签监督。两个关键设计使我们能够将现成的文本到图像扩散模型作为有用的数据集生成器，用于对象实例和蒙版注释。首先，我们将图像画布分为几个区域，并执行一轮扩散过程，同时基于不同的文本提示生成多个实例。其次，我们通过聚合与对象提示相关联的跨注意力图在层和扩散时间步上，然后进行简单的阈值处理和边缘感知的细化处理，得到相应的实例蒙版。我们的MosaicFusion可以为稀缺和新颖类别产生大量的合成标记数据，而无需复杂的处理。在具有挑战性的LVIS长尾和开放词汇基准上进行的实验结果表明，我们的方法在准确率和泛化能力方面均取得了显著的提升。

    We present MosaicFusion, a simple yet effective diffusion-based data augmentation approach for large vocabulary instance segmentation. Our method is training-free and does not rely on any label supervision. Two key designs enable us to employ an off-the-shelf text-to-image diffusion model as a useful dataset generator for object instances and mask annotations. First, we divide an image canvas into several regions and perform a single round of diffusion process to generate multiple instances simultaneously, conditioning on different text prompts. Second, we obtain corresponding instance masks by aggregating cross-attention maps associated with object prompts across layers and diffusion time steps, followed by simple thresholding and edge-aware refinement processing. Without bells and whistles, our MosaicFusion can produce a significant amount of synthetic labeled data for both rare and novel categories. Experimental results on the challenging LVIS long-tailed and open-vocabulary benchma
    
[^28]: 高效有限初始化张量化神经网络的方法

    Efficient Finite Initialization for Tensorized Neural Networks. (arXiv:2309.06577v1 [cs.LG])

    [http://arxiv.org/abs/2309.06577](http://arxiv.org/abs/2309.06577)

    这种方法提出了一种高效有限初始化张量化神经网络层的方法，避免了参数爆炸问题，并通过使用弗罗贝尼乌斯范数的迭代部分形式来计算范数，使其具有有限范围。应用于不同层的实验表明其性能良好。

    

    我们提出了一种新的方法，用于初始化张量化神经网络的层，以避免参数爆炸。该方法适用于具有大量节点的层，其中所有或大多数节点与输入或输出有连接。该方法的核心是使用该层的弗罗贝尼乌斯范数的迭代部分形式，使其具有有限的范围。这个范数的计算是高效的，对于大多数情况都可以完全或部分计算。我们将这个方法应用于不同的层，并检查其性能。我们创建了一个Python函数，在i3BQuantum存储库的Jupyter Notebook中可以运行它：https://github.com/i3BQuantumTeam/Q4Real/blob/e07c827651ef16bcf74590ab965ea3985143f891/Quantum-Inspired%20Variational%20Methods/Normalization_process.ipynb

    We present a novel method for initializing layers of tensorized neural networks in a way that avoids the explosion of the parameters of the matrix it emulates. The method is intended for layers with a high number of nodes in which there is a connection to the input or output of all or most of the nodes. The core of this method is the use of the Frobenius norm of this layer in an iterative partial form, so that it has to be finite and within a certain range. This norm is efficient to compute, fully or partially for most cases of interest. We apply the method to different layers and check its performance. We create a Python function to run it on an arbitrary layer, available in a Jupyter Notebook in the i3BQuantum repository: https://github.com/i3BQuantumTeam/Q4Real/blob/e07c827651ef16bcf74590ab965ea3985143f891/Quantum-Inspired%20Variational%20Methods/Normalization_process.ipynb
    
[^29]: 在增强的不变关系知识上探索超关系时间知识图的链接预测

    Exploring Link Prediction over Hyper-Relational Temporal Knowledge Graphs Enhanced with Time-Invariant Relational Knowledge. (arXiv:2307.10219v1 [cs.AI])

    [http://arxiv.org/abs/2307.10219](http://arxiv.org/abs/2307.10219)

    这项研究填补了时间KG和超关系KG推理之间的差距，并开发了两个新的基准超关系TKG数据集。

    

    超关系知识图(HKGs)是传统知识图(KGs)的延伸，为每个KG事实提供额外的键值对(即限定词)，以更好地限制事实的有效性。近年来，研究在HKGs上进行图推理越来越受关注。与此同时，由于世界知识的不断演变，大量平行工作集中在对时间KGs(TKGs)进行推理，其中每个TKG事实可以被视为带有时间戳(或时间段)的KG事实，指定其时间有效性。现有的HKG推理方法不考虑时间信息，因为在之前的基准数据集中没有显式地指定。此外，所有以前的TKG推理方法只重视时间推理，并没有办法从限定词中学习。因此，我们的目标是填补TKG推理和HKG推理之间的差距。我们开发了两个新的基准超关系TKG(HTKG)数据集，即Wiki-hy和...

    Stemming from traditional knowledge graphs (KGs), hyper-relational KGs (HKGs) provide additional key-value pairs (i.e., qualifiers) for each KG fact that help to better restrict the fact validity. In recent years, there has been an increasing interest in studying graph reasoning over HKGs. In the meantime, due to the ever-evolving nature of world knowledge, extensive parallel works have been focusing on reasoning over temporal KGs (TKGs), where each TKG fact can be viewed as a KG fact coupled with a timestamp (or time period) specifying its time validity. The existing HKG reasoning approaches do not consider temporal information because it is not explicitly specified in previous benchmark datasets. Besides, all the previous TKG reasoning methods only lay emphasis on temporal reasoning and have no way to learn from qualifiers. To this end, we aim to fill the gap between TKG reasoning and HKG reasoning. We develop two new benchmark hyper-relational TKG (HTKG) datasets, i.e., Wiki-hy and 
    
[^30]: 用于电力负荷预测的基准和自定义包

    Benchmarks and Custom Package for Electrical Load Forecasting. (arXiv:2307.07191v1 [cs.LG])

    [http://arxiv.org/abs/2307.07191](http://arxiv.org/abs/2307.07191)

    本文提供了一个全面的电力负荷预测存档，包括负荷领域特定的特征工程，帮助模型更好地模拟负荷数据，并提供了一种新的损失函数来最小化后续任务的成本。

    

    负荷预测在电力行业中具有重要意义，可以为后续任务如电网调度提供参考，从而带来巨大的经济效益。然而，负荷预测与传统的时间序列预测之间存在许多差异。一方面，负荷预测的目标是最小化后续任务（如电网调度）的成本，而不仅仅追求预测准确性。另一方面，负荷受到许多外部因素的影响，如温度或日历变量。此外，预测的规模（如建筑级负荷和聚合级负荷）也会对预测结果产生重大影响。在本文中，我们提供了一个全面的负荷预测存档，其中包括负荷领域特定的特征工程，以帮助预测模型更好地模拟负荷数据。此外，与传统的损失函数仅追求准确性不同，我们还提供了一种方法来...

    Load forecasting is of great significance in the power industry as it can provide a reference for subsequent tasks such as power grid dispatch, thus bringing huge economic benefits. However, there are many differences between load forecasting and traditional time series forecasting. On the one hand, load forecasting aims to minimize the cost of subsequent tasks such as power grid dispatch, rather than simply pursuing prediction accuracy. On the other hand, the load is largely influenced by many external factors, such as temperature or calendar variables. In addition, the scale of predictions (such as building-level loads and aggregated-level loads) can also significantly impact the predicted results. In this paper, we provide a comprehensive load forecasting archive, which includes load domain-specific feature engineering to help forecasting models better model load data. In addition, different from the traditional loss function which only aims for accuracy, we also provide a method to
    
[^31]: 物理学的散射谱模型

    Scattering Spectra Models for Physics. (arXiv:2306.17210v1 [physics.data-an])

    [http://arxiv.org/abs/2306.17210](http://arxiv.org/abs/2306.17210)

    本文介绍了物理学中的散射谱模型，用于描述各种场的统计特性。这些模型基于散射系数的协方差，结合了场的小波分解和点位模，能够准确且稳健地重现标准统计量，捕捉了关键特性。

    

    物理学家常常需要概率模型来进行参数推断或生成一个场的新实现。针对高度非高斯场的建立这样的模型是一项挑战，特别是当样本数量有限时。在本文中，我们介绍了散射谱模型用于平稳场，并展示了它们在物理学中遇到的各种场的准确且稳健的统计描述。这些模型基于散射系数的协方差，即场的小波分解和点位模。在介绍利用旋转和缩放下场的规律性进行有用的维度约简后，我们验证了这些模型在不同多尺度物理场上的效果，并证明它们能够重现标准统计量，包括四阶空间矩。这些散射谱为我们提供了一种低维结构化表示，捕捉了关键特性。

    Physicists routinely need probabilistic models for a number of tasks such as parameter inference or the generation of new realizations of a field. Establishing such models for highly non-Gaussian fields is a challenge, especially when the number of samples is limited. In this paper, we introduce scattering spectra models for stationary fields and we show that they provide accurate and robust statistical descriptions of a wide range of fields encountered in physics. These models are based on covariances of scattering coefficients, i.e. wavelet decomposition of a field coupled with a point-wise modulus. After introducing useful dimension reductions taking advantage of the regularity of a field under rotation and scaling, we validate these models on various multi-scale physical fields and demonstrate that they reproduce standard statistics, including spatial moments up to 4th order. These scattering spectra provide us with a low-dimensional structured representation that captures key prop
    
[^32]: 时间序列预训练模型综述

    A Survey on Time-Series Pre-Trained Models. (arXiv:2305.10716v1 [cs.LG])

    [http://arxiv.org/abs/2305.10716](http://arxiv.org/abs/2305.10716)

    本综述全面回顾了时间序列预训练模型，其中监督、无监督和自监督是主要类别。通过使用这些模型，可以克服构建大规模标记数据集的困难，提高时间序列挖掘的性能和效率。

    

    时间序列挖掘是一个重要的研究领域，因为它在实际应用中显示出巨大的潜力。依赖于大量标记数据的深度学习模型已经成功地用于时间序列挖掘。然而，由于数据注释成本的原因，构建大规模、良好标记的数据集是困难的。最近，预训练模型在时间序列领域逐渐引起关注，因为它们在计算机视觉和自然语言处理方面表现出色。在本综述中，我们全面回顾了时间序列预训练模型（TS-PTMs），旨在指导了解、应用和研究TS-PTMs。具体而言，我们先简要介绍了TSM中使用的典型深度学习模型。然后，我们根据预训练技术概述了TS-PTMs。我们探讨的主要类别包括监督、无监督和自监督TS-PTMs。此外，进行了广泛的实验来分析它们的优缺点。

    Time-Series Mining (TSM) is an important research area since it shows great potential in practical applications. Deep learning models that rely on massive labeled data have been utilized for TSM successfully. However, constructing a large-scale well-labeled dataset is difficult due to data annotation costs. Recently, Pre-Trained Models have gradually attracted attention in the time series domain due to their remarkable performance in computer vision and natural language processing. In this survey, we provide a comprehensive review of Time-Series Pre-Trained Models (TS-PTMs), aiming to guide the understanding, applying, and studying TS-PTMs. Specifically, we first briefly introduce the typical deep learning models employed in TSM. Then, we give an overview of TS-PTMs according to the pre-training techniques. The main categories we explore include supervised, unsupervised, and self-supervised TS-PTMs. Further, extensive experiments are conducted to analyze the advantages and disadvantage
    
[^33]: 联邦集成指导的离线强化学习算法

    Federated Ensemble-Directed Offline Reinforcement Learning. (arXiv:2305.03097v1 [cs.LG])

    [http://arxiv.org/abs/2305.03097](http://arxiv.org/abs/2305.03097)

    本文开发了一种名为FEDORA的联邦集成指导的离线强化学习算法，通过集成学习方法提炼客户群体的集体智慧，显著优于其他方法，包括在合并的数据汇总中进行离线强化学习，在各种复杂的连续控制环境和真实世界数据集中进行了实验。

    

    本文考虑了联邦离线强化学习问题。在这一场景下，分布式的学习代理必须仅使用由不同的未知的行为策略生成的小型预先收集的数据集协作学习出高质量的控制策略。笨拙地将标准离线强化学习方法与标准联邦学习方法组合来解决这个问题可能会导致表现不佳的策略。我们因此设计了Federated Ensemble-Directed Offline Reinforcement Learning Algorithm (FEDORA)，通过集成学习方法提炼客户群体的集体智慧。我们开发了FEDORA代码库，利用联邦学习平台上的分布式计算资源。我们证明了FEDORA在各种复杂的连续控制环境和真实世界数据集中均显著优于其他方法，包括在合并的数据汇总中进行离线强化学习。最后，我们展示了FEDORA在真实世界中的表现。

    We consider the problem of federated offline reinforcement learning (RL), a scenario under which distributed learning agents must collaboratively learn a high-quality control policy only using small pre-collected datasets generated according to different unknown behavior policies. Naively combining a standard offline RL approach with a standard federated learning approach to solve this problem can lead to poorly performing policies. In response, we develop the Federated Ensemble-Directed Offline Reinforcement Learning Algorithm (FEDORA), which distills the collective wisdom of the clients using an ensemble learning approach. We develop the FEDORA codebase to utilize distributed compute resources on a federated learning platform. We show that FEDORA significantly outperforms other approaches, including offline RL over the combined data pool, in various complex continuous control environments and real world datasets. Finally, we demonstrate the performance of FEDORA in the real-world on 
    
[^34]: 利用分段仿射代理实现混合变量的全局和优先级优化

    Global and Preference-based Optimization with Mixed Variables using Piecewise Affine Surrogates. (arXiv:2302.04686v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2302.04686](http://arxiv.org/abs/2302.04686)

    本文提出了一种基于分段仿射代理构建的全局和基于偏好优化算法，可解决线性约束的混合变量问题，算法通过两种探索函数可有效搜索可行域

    

    在存在复杂限制条件的情况下，涉及混合变量（即数值和分类性的变量）的优化问题可能难以解决。此外，当目标函数是复杂模拟或实验的结果时，评估代价可能很高。本文提出了一种新颖的代理全局优化算法，基于对可行样本上目标函数的分段仿射代理构建来解决线性约束的混合变量问题，可解决中到大规模问题（编码后约100个变量和20个约束）。我们介绍了两种探索函数来通过混合整数线性规划求解器有效地搜索可行域。我们还提供了一种基于偏好的算法版本，当只能获得样本间的成对比较而未量化底层要最小化的目标函数时，可使用该算法。这两种算法进行了测试。

    Optimization problems involving mixed variables, i.e., variables of numerical and categorical nature, can be challenging to solve, especially in the presence of complex constraints. Moreover, when the objective function is the result of a complicated simulation or experiment, it may be expensive to evaluate. This paper proposes a novel surrogate-based global optimization algorithm to solve linearly constrained mixed-variable problems up to medium-large size (around 100 variables after encoding and 20 constraints) based on constructing a piecewise affine surrogate of the objective function over feasible samples. We introduce two types of exploration functions to efficiently search the feasible domain via mixed-integer linear programming solvers. We also provide a preference-based version of the algorithm, which can be used when only pairwise comparisons between samples can be acquired while the underlying objective function to minimize remains unquantified. The two algorithms are tested
    
[^35]: E-MCTS：通过规划表观不确定性进行深度探索的模型基强化学习

    E-MCTS: Deep Exploration in Model-Based Reinforcement Learning by Planning with Epistemic Uncertainty. (arXiv:2210.13455v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13455](http://arxiv.org/abs/2210.13455)

    本文提出了一种新的方法E-MCTS，通过在MCTS预测中应用表观不确定性估计，实现了模型基强化学习中的深度探索，以及规划探索策略。通过实验证明这种方法在成功的表观不确定性估计和深度探索方面表现优异。

    

    模拟退火树搜索（MCTS）是模型基强化学习中应用最广泛、性能最优秀的规划方法之一。MCTS的关键挑战在于深度探索和面对未知时的可靠性，这两个挑战可以通过在MCTS预测中使用原则性的表观不确定性估计来缓解。本文提出了两个主要贡献：首先，我们开发了一种在MCTS中传播表观不确定性的方法，使智能体能够估计其预测的表观不确定性。其次，我们利用传播的不确定性提出了一种新的深度探索算法，通过明确规划探索策略。我们将这种方法应用于基于MCTS的模型基强化学习方法中，包括使用学习和提供的模型，通过实验证明了我们的方法实现了成功的表观不确定性估计并进行了深度探索。我们将其与基于非规划的深度探索基线进行了比较，并表明...

    One of the most well-studied and highly performing planning approaches used in Model-Based Reinforcement Learning (MBRL) is Monte-Carlo Tree Search (MCTS). Key challenges of MCTS-based MBRL methods remain dedicated deep exploration and reliability in the face of the unknown, and both challenges can be alleviated through principled epistemic uncertainty estimation in the predictions of MCTS. We present two main contributions: First, we develop methodology to propagate epistemic uncertainty in MCTS, enabling agents to estimate the epistemic uncertainty in their predictions. Second, we utilize the propagated uncertainty for a novel deep exploration algorithm by explicitly planning to explore. We incorporate our approach into variations of MCTS-based MBRL approaches with learned and provided models, and empirically show deep exploration through successful epistemic uncertainty estimation achieved by our approach. We compare to a non-planning-based deep-exploration baseline, and demonstrate
    

