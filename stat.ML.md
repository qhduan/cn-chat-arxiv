# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization](https://rss.arxiv.org/abs/2402.01401) | 通过Lipschitz正则化实现零样本机器遗忘，可以及时忘记私人或受版权保护的信息，同时保持模型性能。 |
| [^2] | [Linguistic Calibration of Language Models](https://arxiv.org/abs/2404.00474) | 该论文提出了一种通过语言模型的文本生成来实现语言校准，可以使用户做出校准概率预测的方法。 |
| [^3] | [Span-Based Optimal Sample Complexity for Weakly Communicating and General Average Reward MDPs](https://arxiv.org/abs/2403.11477) | 该研究提出了对于弱通信MDPs的样本复杂度界限为 $\tilde{O}(SA\frac{H}{\epsilon^2})$，改进了现有工作，是在所有参数上最小最优的。 |
| [^4] | [Convergence of Some Convex Message Passing Algorithms to a Fixed Point](https://arxiv.org/abs/2403.07004) | 这项研究证明了一些凸消息传递算法会收敛到固定点，并在一定迭代次数内达到特定精度。 |
| [^5] | [Scalable Online Exploration via Coverability](https://arxiv.org/abs/2403.06571) | 提出了探索目标框架，引入了$L_1$-覆盖度作为新的探索目标，支持内在复杂度控制、高效规划和灵活集成的优点。 |
| [^6] | [Transformers Provably Learn Feature-Position Correlations in Masked Image Modeling](https://arxiv.org/abs/2403.02233) | 本文提供了关于使用MIM自监督预训练学习transformers的首个端到端理论，揭示了transformers如何学习到在具有空间结构的数据分布上突显特征-位置相关性的本地和多样化注意模式 |
| [^7] | [Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models](https://arxiv.org/abs/2402.12336) | 通过无监督对抗微调，提出了一种强大的CLIP视觉编码器，用于增强各种视觉-语言模型的鲁棒性。恶意第三方提供操纵图像的用户隐形攻击得以杜绝。 |
| [^8] | [How to validate average calibration for machine learning regression tasks ?](https://arxiv.org/abs/2402.10043) | 本文提出了两种验证机器学习回归任务平均校准性的方法，将校准误差与平均绝对误差之间的差值和将平均平方z-分数与1进行比较。研究发现，前者对不确定性分布敏感，而后者在该方面提供了最可靠的方法。 |
| [^9] | [On the Universality of Coupling-based Normalizing Flows](https://arxiv.org/abs/2402.06578) | 我们提出了一个新颖的理论框架，用于理解基于耦合的标准化流的表达能力，并提出了一个新的分布普适性定理来克服以前工作的限制。这些结果支持耦合架构的表达能力，并弥补了实证结果和理论理解之间的差距。 |
| [^10] | [How do Transformers perform In-Context Autoregressive Learning?](https://arxiv.org/abs/2402.05787) | 本文研究了Transformers在上下文自回归学习中的表现，并通过训练模型发现了其预测下一个标记的过程。针对不同情况，我们证明了单层线性Transformer实现了梯度下降以及正交性之间的关系。 |
| [^11] | [Future Directions in Foundations of Graph Machine Learning](https://arxiv.org/abs/2402.02287) | 图机器学习领域的未来方向应该是发展一个更加均衡的理论，从更完整的角度探究图神经网络的表达能力、泛化和优化之间的相互关系。 |
| [^12] | [Uncertainty-Aware Partial-Label Learning](https://arxiv.org/abs/2402.00592) | 本文提出了一种基于最近邻的部分标签学习算法，利用Dempster-Shafer理论实现对模糊标记的数据的训练。实验结果表明，该算法能够提供良好的不确定性估计，并具有竞争力的预测性能。 |
| [^13] | [Efficient Exploration for LLMs](https://arxiv.org/abs/2402.00396) | 高效探索在改善大型语言模型方面具有显著优势，可以以较少的查询实现较高的性能水平。不确定性估计和探索方案的选择是关键因素。 |
| [^14] | [Harnessing Density Ratios for Online Reinforcement Learning.](http://arxiv.org/abs/2401.09681) | 本文通过提出一种新的算法（GLOW），结合密度比率模型和值函数模型，在线强化学习中解决了瓶颈问题，即如何在没有初始数据集的情况下收集具有良好覆盖度的探索数据集。 |
| [^15] | [Randomized Kaczmarz with geometrically smoothed momentum.](http://arxiv.org/abs/2401.09415) | 本文研究了向随机Kaczmarz算法中添加几何平滑动量的效果，并证明了关于最小二乘损失矩阵奇异向量方向上的期望误差。数值示例证明了结果的实用性，并提出了几个问题。 |
| [^16] | [Sampling in Unit Time with Kernel Fisher-Rao Flow.](http://arxiv.org/abs/2401.03892) | 本文提出了一种具有核Fisher-Rao流的新方法，在单位时间内从非归一化目标密度或贝叶斯后验中进行采样。方法使用了均场ODE和相互作用粒子系统，无需梯度，只需要能够从参考密度中采样并计算目标对参考密度的比率。该方法通过在几何混合的路径上沿速度场运输样本，径向输运样本。方法通过在再生核希尔伯特空间中求解泊松方程，使泊松方程的求解变得可行，并将其离散化为有限样本的均场ODE，作为实现简单的相互作用粒子系统。同时，这种方法也可以从离散时间的角度推导出均场ODE，作为蒙杰-安普尔方程连续线性化的极限。 |
| [^17] | [Isolated pulsar population synthesis with simulation-based inference.](http://arxiv.org/abs/2312.14848) | 本论文使用模拟推断方法结合脉冲星种群合成，来限制孤立银河射电脉冲星的磁旋转特性。 |
| [^18] | [CONFIDERAI: a novel CONFormal Interpretable-by-Design score function forExplainable and Reliable Artificial Intelligence.](http://arxiv.org/abs/2309.01778) | 提出了一种新的可解释机器学习评分函数CONFIDERAI，它将一致性预测与规则模型相结合，利用规则的预测能力和点的几何位置，在特征空间中定义满足一致性保证的区域。 |
| [^19] | [ARK: Robust Knockoffs Inference with Coupling.](http://arxiv.org/abs/2307.04400) | 本研究探讨了在特征分布被错误估计或估计的情况下，通过耦合模型-X Knockoffs过程与近似Knockoffs过程，实现了在目标水平上的鲁棒的FDR或FWER控制。 |
| [^20] | [Deep Stochastic Mechanics.](http://arxiv.org/abs/2305.19685) | 本文提出了一种基于深度学习的方法，用于数值模拟时间演化薛定谔方程，利用马尔可夫扩散采样来适应波函数的潜在低维结构，并提出了新的随机量子力学方程，具有线性的计算复杂度。数值模拟显示出显着的优势。 |
| [^21] | [Improved Convergence of Score-Based Diffusion Models via Prediction-Correction.](http://arxiv.org/abs/2305.14164) | 本文通过使用预测校正方案，提高了基于得分扩散模型的收敛性。 |
| [^22] | [Statistical Guarantees of Group-Invariant GANs.](http://arxiv.org/abs/2305.13517) | 本研究提出了群不变GAN的统计保证，发现当学习群不变分布时，群不变GAN所需样本数会按群体大小的幂比例减少。 |
| [^23] | [Posterior Inference on Infinitely Wide Bayesian Neural Networks under Weights with Unbounded Variance.](http://arxiv.org/abs/2305.10664) | 本文提出了一种新的方法进行关于具有无界方差权重的贝叶斯神经网络的后验推断，并表明后验分布集中在具有非标准超参数依赖性的稀疏促进和均值收缩先验周围。 |
| [^24] | [The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning.](http://arxiv.org/abs/2304.05366) | 本论文阐述了无免费午餐定理的监督学习中的限制，证明了归纳偏差可以提高学习算法的效果，并且展示了神经网络模型的偏好与现实世界的数据分布相关。 |
| [^25] | [Synergistic Graph Fusion via Encoder Embedding.](http://arxiv.org/abs/2303.18051) | 本文提出了一种协同图融合的新方法，该方法处理具有共同顶点集的多个图，有着非常理想的“协同效应”，即顶点分类准确度总是受益于额外的图，并在实验中证实了其卓越性能。 |
| [^26] | [Dynamic Ranking and Translation Synchronization.](http://arxiv.org/abs/2207.01455) | 本论文研究了动态排名和翻译同步问题，主要关注成对比较数据随时间变化的情况，并给出了相应的理论结果。 |

# 详细

[^1]: 通过Lipschitz正则化在规模上实现零样本机器遗忘

    Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization

    [https://rss.arxiv.org/abs/2402.01401](https://rss.arxiv.org/abs/2402.01401)

    通过Lipschitz正则化实现零样本机器遗忘，可以及时忘记私人或受版权保护的信息，同时保持模型性能。

    

    为了遵守人工智能和数据规定，从训练得到的机器学习模型中遗忘私人或受版权保护的信息的需求变得越来越重要。遗忘的关键挑战是及时忘记必要的数据，同时保持模型性能。在这项工作中，我们解决了零样本遗忘的场景，即只有一个经过训练的模型和要遗忘的数据，遗忘算法必须能够移除数据。根据这样定义，现有的最先进的方法是不够的。基于Lipschitz连续性的概念，我们提出了一种方法，通过对样本扰动的输出进行平滑处理来诱导遗忘。我们展示了这种平滑性成功地实现了遗忘，同时保持了总体模型性能。我们对我们的方法进行了广泛的经验评估，包括一系列当代基准测试，验证了我们的方法在严格的零样本约束下达到了最先进的性能。

    To comply with AI and data regulations, the need to forget private or copyrighted information from trained machine learning models is increasingly important. The key challenge in unlearning is forgetting the necessary data in a timely manner, while preserving model performance. In this work, we address the zero-shot unlearning scenario, whereby an unlearning algorithm must be able to remove data given only a trained model and the data to be forgotten. Under such a definition, existing state-of-the-art methods are insufficient. Building on the concepts of Lipschitz continuity, we present a method that induces smoothing of the forget sample's output, with respect to perturbations of that sample. We show this smoothing successfully results in forgetting while preserving general model performance. We perform extensive empirical evaluation of our method over a range of contemporary benchmarks, verifying that our method achieves state-of-the-art performance under the strict constraints of ze
    
[^2]: 语言模型的语言校准

    Linguistic Calibration of Language Models

    [https://arxiv.org/abs/2404.00474](https://arxiv.org/abs/2404.00474)

    该论文提出了一种通过语言模型的文本生成来实现语言校准，可以使用户做出校准概率预测的方法。

    

    语言模型可能会在自信幻觉时导致用户做出次优化的下游决策。通过语言模型口头传达其主张正确概率可以缓解这个问题，但现有模型无法生成具有校准置信度声明的文本。我们通过决策角度，为长篇生成形式的语言校准形式化定义：如果语言模型的生成使其用户能够做出校准概率预测，则该模型是语言上校准的。这个定义使得一个训练框架成为可能，其中一个监督微调步骤引导一个语言模型发出带有置信度声明的长篇生成，诸如“我估计有30%的机会…”或“我确信…”，然后是一个强化学习步骤，奖励使用户能够对相关问题提供校准答案的生成。我们对Llama 2 7B 进行语言校准，并发现在自动化和人类测试中...

    arXiv:2404.00474v1 Announce Type: cross  Abstract: Language models (LMs) may lead their users to make suboptimal downstream decisions when they confidently hallucinate. This issue can be mitigated by having the LM verbally convey the probability that its claims are correct, but existing models cannot produce text with calibrated confidence statements. Through the lens of decision-making, we formalize linguistic calibration for long-form generations: an LM is linguistically calibrated if its generations enable its users to make calibrated probabilistic predictions. This definition enables a training framework where a supervised finetuning step bootstraps an LM to emit long-form generations with confidence statements such as "I estimate a 30% chance of..." or "I am certain that...", followed by a reinforcement learning step which rewards generations that enable a user to provide calibrated answers to related questions. We linguistically calibrate Llama 2 7B and find in automated and huma
    
[^3]: 弱通信和一般平均奖赏MDPs的基于跨度的最佳样本复杂度

    Span-Based Optimal Sample Complexity for Weakly Communicating and General Average Reward MDPs

    [https://arxiv.org/abs/2403.11477](https://arxiv.org/abs/2403.11477)

    该研究提出了对于弱通信MDPs的样本复杂度界限为 $\tilde{O}(SA\frac{H}{\epsilon^2})$，改进了现有工作，是在所有参数上最小最优的。

    

    我们研究了在生成模型下学习平均奖赏马尔可夫决策过程（MDP）中$\epsilon$-最佳策略的样本复杂度。对于弱通信MDPs，我们建立了复杂度界限为$\tilde{O}(SA\frac{H}{\epsilon^2})$，其中$H$是最优策略的偏差函数的跨度，$SA$是状态-动作空间的基数。我们的结果是在所有参数$S,A,H$和$\epsilon$上（最多对数因子）最小最优的，改进了现有工作，现有工作要么假设所有策略的混合时间均匀有界，要么对参数有次优的依赖。我们进一步研究一般（非弱通信）平均奖赏MDPs中的样本复杂度。我们认为需要一个新的瞬态时间参数$B$，建立了一个$\tilde{O}(SA\frac{B+H}{\epsilon^2})$的复杂度界限，并证明了匹配的（最多对数因子）最小最优下界。这两个结果都是基于减少

    arXiv:2403.11477v1 Announce Type: new  Abstract: We study the sample complexity of learning an $\epsilon$-optimal policy in an average-reward Markov decision process (MDP) under a generative model. For weakly communicating MDPs, we establish the complexity bound $\tilde{O}(SA\frac{H}{\epsilon^2})$, where $H$ is the span of the bias function of the optimal policy and $SA$ is the cardinality of the state-action space. Our result is the first that is minimax optimal (up to log factors) in all parameters $S,A,H$ and $\epsilon$, improving on existing work that either assumes uniformly bounded mixing times for all policies or has suboptimal dependence on the parameters. We further investigate sample complexity in general (non-weakly-communicating) average-reward MDPs. We argue a new transient time parameter $B$ is necessary, establish an $\tilde{O}(SA\frac{B+H}{\epsilon^2})$ complexity bound, and prove a matching (up to log factors) minimax lower bound. Both results are based on reducing the
    
[^4]: 一些凸消息传递算法收敛到固定点

    Convergence of Some Convex Message Passing Algorithms to a Fixed Point

    [https://arxiv.org/abs/2403.07004](https://arxiv.org/abs/2403.07004)

    这项研究证明了一些凸消息传递算法会收敛到固定点，并在一定迭代次数内达到特定精度。

    

    在图模型中解决MAP推断问题的一种流行方法是通过（块状）坐标下降最小化从对偶线性规划或Lagrange松弛中获得的一个上界。这样的算法包括最大和扩散以及顺序树重新加权消息传递。这些方法的收敛性质目前尚未完全理解。它们已被证明会收敛到由活跃约束的局部一致性所表征的集合，但收敛速度未知；然而，尚不清楚迭代是否会收敛（到任何一个单一点）。我们证明了一个更强的结果（之前有猜想但从未证明过）：迭代会收敛到算法的一个固定点。此外，我们还展示它们在$\mathcal{O}(1/\varepsilon)$次迭代中达到了精度$\varepsilon>0$。

    arXiv:2403.07004v1 Announce Type: new  Abstract: A popular approach to the MAP inference problem in graphical models is to minimize an upper bound obtained from a dual linear programming or Lagrangian relaxation by (block-)coordinate descent. Examples of such algorithms are max-sum diffusion and sequential tree-reweighted message passing. Convergence properties of these methods are currently not fully understood. They have been proved to converge to the set characterized by local consistency of active constraints, with unknown convergence rate; however, it was not clear if the iterates converge at all (to any single point). We prove a stronger result (which was conjectured before but never proved): the iterates converge to a fixed point of the algorithm. Moreover, we show that they achieve precision $\varepsilon>0$ in $\mathcal{O}(1/\varepsilon)$ iterations.   We first prove this for a version of coordinate descent applied to a general piecewise-affine convex objective, using a novel p
    
[^5]: 可扩展的在线探索方法：通过Coverability

    Scalable Online Exploration via Coverability

    [https://arxiv.org/abs/2403.06571](https://arxiv.org/abs/2403.06571)

    提出了探索目标框架，引入了$L_1$-覆盖度作为新的探索目标，支持内在复杂度控制、高效规划和灵活集成的优点。

    

    在强化学习中，探索是一个主要挑战，尤其对于需要函数逼近的高维领域。我们提出了探索目标——作为一个概念框架，能够使任何奖励函数的下游最大化成为可能。在这个框架内，我们引入了一个新的目标，即$L_1$-覆盖度，它泛化了以往的探索方案，并支持三个基本愿望：1.内在复杂度控制。$L_1$-覆盖度与结构参数$L_1$-Coverability相关联，反映了潜在MDP的内在统计困难度，包含Block和Low-Rank MDPs。2.高效规划。对于已知的MDP，优化$L_1$-覆盖度能够有效地降低到标准的策略优化，允许与诸如策略梯度和Q-learning等现成方法灵活集成。3.高效的探索。$L_1$-覆盖度的优化等同于现有强化学习算法的操作，尤其在高维领域中具有很强的泛化性。

    arXiv:2403.06571v1 Announce Type: new  Abstract: Exploration is a major challenge in reinforcement learning, especially for high-dimensional domains that require function approximation. We propose exploration objectives -- policy optimization objectives that enable downstream maximization of any reward function -- as a conceptual framework to systematize the study of exploration. Within this framework, we introduce a new objective, $L_1$-Coverage, which generalizes previous exploration schemes and supports three fundamental desiderata:   1. Intrinsic complexity control. $L_1$-Coverage is associated with a structural parameter, $L_1$-Coverability, which reflects the intrinsic statistical difficulty of the underlying MDP, subsuming Block and Low-Rank MDPs.   2. Efficient planning. For a known MDP, optimizing $L_1$-Coverage efficiently reduces to standard policy optimization, allowing flexible integration with off-the-shelf methods such as policy gradient and Q-learning approaches.   3. E
    
[^6]: Transformers在Masked Image Modeling中能够证明学习特征-位置相关性

    Transformers Provably Learn Feature-Position Correlations in Masked Image Modeling

    [https://arxiv.org/abs/2403.02233](https://arxiv.org/abs/2403.02233)

    本文提供了关于使用MIM自监督预训练学习transformers的首个端到端理论，揭示了transformers如何学习到在具有空间结构的数据分布上突显特征-位置相关性的本地和多样化注意模式

    

    Masked image modeling (MIM)是一种新兴的自监督视觉预训练方法，它从未屏蔽的图像中预测随机屏蔽的补丁。然而，对于基于transformers的MIM的理论理解相当有限。本文提供了有关使用MIM自监督预训练学习一层transformers的首个端到端理论。我们提出了transformers如何学习到在具有空间结构的数据分布上突显特征-位置相关性的本地和多样化注意模式的理论机制。

    arXiv:2403.02233v1 Announce Type: new  Abstract: Masked image modeling (MIM), which predicts randomly masked patches from unmasked ones, has emerged as a promising approach in self-supervised vision pretraining. However, the theoretical understanding of MIM is rather limited, especially with the foundational architecture of transformers. In this paper, to the best of our knowledge, we provide the first end-to-end theory of learning one-layer transformers with softmax attention in MIM self-supervised pretraining. On the conceptual side, we posit a theoretical mechanism of how transformers, pretrained with MIM, produce empirically observed local and diverse attention patterns on data distributions with spatial structures that highlight feature-position correlations. On the technical side, our end-to-end analysis of the training dynamics of softmax-based transformers accommodates both input and position embeddings simultaneously, which is developed based on a novel approach to track the i
    
[^7]: Robust CLIP: 对视觉嵌入进行无监督对抗微调以获得强大的大规模视觉-语言模型

    Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models

    [https://arxiv.org/abs/2402.12336](https://arxiv.org/abs/2402.12336)

    通过无监督对抗微调，提出了一种强大的CLIP视觉编码器，用于增强各种视觉-语言模型的鲁棒性。恶意第三方提供操纵图像的用户隐形攻击得以杜绝。

    

    诸如OpenFlamingo、LLaVA和GPT-4之类的多模型基础模型越来越广泛地用于各种真实世界任务。先前的工作表明，这些模型在视觉模态上极易受到对抗性攻击的影响。这些攻击可以用来传播虚假信息或欺骗用户，因此构成了一个重大风险，这使得大型多模型基础模型的鲁棒性成为一项紧迫的问题。我们提出了一种无监督对抗微调方案，以获得强大的CLIP视觉编码器，在所有依赖于CLIP的视觉下游任务（VLMs、零样本分类）上具有鲁棒性。特别地，我们展示了一旦更换原始的CLIP模型，用户在使用VLMs时会受到恶意第三方提供的操纵图像的潜在攻击。

    arXiv:2402.12336v1 Announce Type: cross  Abstract: Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many vision-language models (VLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (VLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of VLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP mo
    
[^8]: 如何验证机器学习回归任务的平均校准性？

    How to validate average calibration for machine learning regression tasks ?

    [https://arxiv.org/abs/2402.10043](https://arxiv.org/abs/2402.10043)

    本文提出了两种验证机器学习回归任务平均校准性的方法，将校准误差与平均绝对误差之间的差值和将平均平方z-分数与1进行比较。研究发现，前者对不确定性分布敏感，而后者在该方面提供了最可靠的方法。

    

    机器学习回归任务的平均校准性可以通过两种方式进行测试。一种方式是将校准误差（CE）估计为平均绝对误差（MSE）与平均方差（MV）或平均平方不确定性之间的差值。另一种方式是将平均平方z-分数或缩放误差（ZMS）与1进行比较。两种方法可能得出不同的结论，正如来自最近的机器学习不确定性量化文献中的数据集集合所示。研究表明，CE对不确定性分布非常敏感，特别是对于离群不确定性的存在，因此无法可靠地用于校准测试。相比之下，ZMS统计量不具有这种敏感性问题，在这种情况下提供了最可靠的方法。文章还讨论了对条件校准验证的影响。

    arXiv:2402.10043v1 Announce Type: cross  Abstract: Average calibration of the uncertainties of machine learning regression tasks can be tested in two ways. One way is to estimate the calibration error (CE) as the difference between the mean absolute error (MSE) and the mean variance (MV) or mean squared uncertainty. The alternative is to compare the mean squared z-scores or scaled errors (ZMS) to 1. Both approaches might lead to different conclusion, as illustrated on an ensemble of datasets from the recent machine learning uncertainty quantification literature. It is shown here that the CE is very sensitive to the distribution of uncertainties, and notably to the presence of outlying uncertainties, and that it cannot be used reliably for calibration testing. By contrast, the ZMS statistic does not present this sensitivity issue and offers the most reliable approach in this context. Implications for the validation of conditional calibration are discussed.
    
[^9]: 关于基于耦合的标准化流的普适性

    On the Universality of Coupling-based Normalizing Flows

    [https://arxiv.org/abs/2402.06578](https://arxiv.org/abs/2402.06578)

    我们提出了一个新颖的理论框架，用于理解基于耦合的标准化流的表达能力，并提出了一个新的分布普适性定理来克服以前工作的限制。这些结果支持耦合架构的表达能力，并弥补了实证结果和理论理解之间的差距。

    

    我们提出了一个新颖的理论框架，用于理解基于耦合的标准化流（如RealNVP）的表达能力。尽管耦合流在科学应用中很普遍，但由于其受限的架构，对于耦合流的全面理解仍然困难。现有的定理在实际应用中存在限制，因为它们需要使用任意病态的神经网络。此外，我们还证明了这些结构本质上导致体积保持流，这是一个限制表达能力的基本约束。我们提出了一种新的基于分布的耦合标准化流普适性定理，克服了以前工作的几个限制。我们的结果支持耦合架构具有表达能力的普遍经验，并为选择耦合函数的表达能力提供了细致入微的观点，填补了实证结果和理论理解之间的差距。

    We present a novel theoretical framework for understanding the expressive power of coupling-based normalizing flows such as RealNVP. Despite their prevalence in scientific applications, a comprehensive understanding of coupling flows remains elusive due to their restricted architectures. Existing theorems fall short as they require the use of arbitrarily ill-conditioned neural networks, limiting practical applicability. Additionally, we demonstrate that these constructions inherently lead to volume-preserving flows, a property which we show to be a fundamental constraint for expressivity. We propose a new distributional universality theorem for coupling-based normalizing flows, which overcomes several limitations of prior work. Our results support the general wisdom that the coupling architecture is expressive and provide a nuanced view for choosing the expressivity of coupling functions, bridging a gap between empirical results and theoretical understanding.
    
[^10]: Transformers在上下文自回归学习中的表现如何？

    How do Transformers perform In-Context Autoregressive Learning?

    [https://arxiv.org/abs/2402.05787](https://arxiv.org/abs/2402.05787)

    本文研究了Transformers在上下文自回归学习中的表现，并通过训练模型发现了其预测下一个标记的过程。针对不同情况，我们证明了单层线性Transformer实现了梯度下降以及正交性之间的关系。

    

    Transformers在语言建模任务中取得了最先进的性能。然而，它们取得巨大成功的原因还不清楚。本文通过在简单的下一个标记预测任务上训练Transformer模型，为了更好地理解这一问题。我们展示了训练后的Transformer如何通过首先在上下文中学习W，然后应用预测映射来预测下一个标记。我们称这个结果为上下文自回归学习。具体来说，我们针对W是交换正交矩阵的情况，首先证明了一个训练后的单层线性Transformer在考虑扩展标记的情况下实现一步梯度下降来最小化内部目标函数。当标记没有扩展时，我们对于一个单层对角线线性多头Transformer的全局最小值进行了表征。重要的是，我们展示了头部之间的正交性。

    Transformers have achieved state-of-the-art performance in language modeling tasks. However, the reasons behind their tremendous success are still unclear. In this paper, towards a better understanding, we train a Transformer model on a simple next token prediction task, where sequences are generated as a first-order autoregressive process $s_{t+1} = W s_t$. We show how a trained Transformer predicts the next token by first learning $W$ in-context, then applying a prediction mapping. We call the resulting procedure in-context autoregressive learning. More precisely, focusing on commuting orthogonal matrices $W$, we first show that a trained one-layer linear Transformer implements one step of gradient descent for the minimization of an inner objective function, when considering augmented tokens. When the tokens are not augmented, we characterize the global minima of a one-layer diagonal linear multi-head Transformer. Importantly, we exhibit orthogonality between heads and show that posi
    
[^11]: 图机器学习基础的未来方向

    Future Directions in Foundations of Graph Machine Learning

    [https://arxiv.org/abs/2402.02287](https://arxiv.org/abs/2402.02287)

    图机器学习领域的未来方向应该是发展一个更加均衡的理论，从更完整的角度探究图神经网络的表达能力、泛化和优化之间的相互关系。

    

    随着图数据在不同学科（从生命科学到社会科学和工程科学）上的广泛应用，图机器学习，尤其是使用图神经网络（GNNs），引起了人们浓厚的兴趣。尽管在实际应用中取得了成功，但我们对GNNs性质的理论理解仍然非常不完整。最近的理论发展主要集中在阐明GNNs粗粒度表达能力方面，主要采用组合技巧。然而，这些研究与实践并不完全一致，特别是在使用随机一阶优化技术训练GNNs时，对GNNs的泛化行为的理解。在这篇定位论文中，我们认为图机器学习领域需要将注意力转移到发展一个更加均衡的图机器学习理论上来，重点关注表达能力、泛化和优化的相互关系的更全面的理解。

    Machine learning on graphs, especially using graph neural networks (GNNs), has seen a surge in interest due to the wide availability of graph data across a broad spectrum of disciplines, from life to social and engineering sciences. Despite their practical success, our theoretical understanding of the properties of GNNs remains highly incomplete. Recent theoretical advancements primarily focus on elucidating the coarse-grained expressive power of GNNs, predominantly employing combinatorial techniques. However, these studies do not perfectly align with practice, particularly in understanding the generalization behavior of GNNs when trained with stochastic first-order optimization techniques. In this position paper, we argue that the graph machine learning community needs to shift its attention to developing a more balanced theory of graph machine learning, focusing on a more thorough understanding of the interplay of expressive power, generalization, and optimization.
    
[^12]: 不确定性感知的部分标签学习

    Uncertainty-Aware Partial-Label Learning

    [https://arxiv.org/abs/2402.00592](https://arxiv.org/abs/2402.00592)

    本文提出了一种基于最近邻的部分标签学习算法，利用Dempster-Shafer理论实现对模糊标记的数据的训练。实验结果表明，该算法能够提供良好的不确定性估计，并具有竞争力的预测性能。

    

    在现实世界的应用中，人们经常遇到标记模糊的数据，即不同的标注者为相同样本分配了冲突的类别标签。部分标签学习允许在这种弱监督的情况下训练分类器。虽然最先进的方法已经具有良好的预测性能，但它们往往受到错误的不确定性估计的影响。然而，在医学和自动驾驶等安全关键领域，具有良好校准的不确定性估计尤为重要。在本文中，我们提出了一种基于最近邻的部分标签学习算法，该算法利用了Dempster-Shafer理论。对人工数据集和实际数据集进行的广泛实验表明，所提出的方法能够提供良好的不确定性估计，并具有竞争力的预测性能。此外，我们还证明了我们的算法具有风险一致性。

    In real-world applications, one often encounters ambiguously labeled data, where different annotators assign conflicting class labels. Partial-label learning allows training classifiers in this weakly supervised setting. While state-of-the-art methods already feature good predictive performance, they often suffer from miscalibrated uncertainty estimates. However, having well-calibrated uncertainty estimates is important, especially in safety-critical domains like medicine and autonomous driving. In this article, we propose a novel nearest-neighbor-based partial-label-learning algorithm that leverages Dempster-Shafer theory. Extensive experiments on artificial and real-world datasets show that the proposed method provides a well-calibrated uncertainty estimate and achieves competitive prediction performance. Additionally, we prove that our algorithm is risk-consistent.
    
[^13]: LLMs的高效探索

    Efficient Exploration for LLMs

    [https://arxiv.org/abs/2402.00396](https://arxiv.org/abs/2402.00396)

    高效探索在改善大型语言模型方面具有显著优势，可以以较少的查询实现较高的性能水平。不确定性估计和探索方案的选择是关键因素。

    

    我们提供了证据，表明高效探索在获取人类反馈以改善大型语言模型方面具有显著优势。在我们的实验中，一个代理程序在收到反馈时将奖励模型拟合到查询上。我们表现最佳的代理程序使用双Thompson采样生成查询，不确定性由认知神经网络表示。我们的结果表明，高效探索使得性能水平可以在较少的查询下达到较高水平。此外，不确定性估计和探索方案的选择起着关键作用。

    We present evidence of substantial benefit from efficient exploration in gathering human feedback to improve large language models. In our experiments, an agent sequentially generates queries while fitting a reward model to the feedback received. Our best-performing agent generates queries using double Thompson sampling, with uncertainty represented by an epistemic neural network. Our results demonstrate that efficient exploration enables high levels of performance with far fewer queries. Further, both uncertainty estimation and the choice of exploration scheme play critical roles.
    
[^14]: 利用密度比率进行在线强化学习

    Harnessing Density Ratios for Online Reinforcement Learning. (arXiv:2401.09681v1 [cs.LG])

    [http://arxiv.org/abs/2401.09681](http://arxiv.org/abs/2401.09681)

    本文通过提出一种新的算法（GLOW），结合密度比率模型和值函数模型，在线强化学习中解决了瓶颈问题，即如何在没有初始数据集的情况下收集具有良好覆盖度的探索数据集。

    

    尽管离线和在线强化学习的理论发展方向一直是平行的，但它们开始显示出可能统一的迹象，其中一个环境的算法和分析技术通常在另一个环境中具有自然的对应物。然而，密度比率建模的概念，这是离线强化学习中的新兴范式，在在线强化学习中很少出现，也许有充足的理由：密度比率的存在和有界性依赖于具有良好覆盖度的探索性数据集的访问性，但在线强化学习的核心挑战是在没有初始数据集的情况下收集这样的数据集。在这项工作中，我们表明 - 也许令人惊讶的是 - 基于密度比率的算法具有在线对应物。假定只存在具有良好覆盖度的探索性分布，即结构条件已知为coverability（Xie等，2023），我们提供了一种新的算法（GLOW），它利用密度比率可实现性和值函数可实现性来进行高效采样。

    The theories of offline and online reinforcement learning, despite having evolved in parallel, have begun to show signs of the possibility for a unification, with algorithms and analysis techniques for one setting often having natural counterparts in the other. However, the notion of density ratio modeling, an emerging paradigm in offline RL, has been largely absent from online RL, perhaps for good reason: the very existence and boundedness of density ratios relies on access to an exploratory dataset with good coverage, but the core challenge in online RL is to collect such a dataset without having one to start. In this work we show -- perhaps surprisingly -- that density ratio-based algorithms have online counterparts. Assuming only the existence of an exploratory distribution with good coverage, a structural condition known as coverability (Xie et al., 2023), we give a new algorithm (GLOW) that uses density ratio realizability and value function realizability to perform sample-effici
    
[^15]: 具有几何平滑动量的随机Kaczmarz方法

    Randomized Kaczmarz with geometrically smoothed momentum. (arXiv:2401.09415v1 [math.NA])

    [http://arxiv.org/abs/2401.09415](http://arxiv.org/abs/2401.09415)

    本文研究了向随机Kaczmarz算法中添加几何平滑动量的效果，并证明了关于最小二乘损失矩阵奇异向量方向上的期望误差。数值示例证明了结果的实用性，并提出了几个问题。

    

    本文研究了向随机Kaczmarz算法中添加几何平滑动量的效果，该算法是线性最小二乘损失函数上随机梯度下降的实例。我们证明了关于定义最小二乘损失的矩阵的奇异向量方向上期望误差的结果。我们给出了几个数值示例来说明我们结果的实用性，并提出了几个问题。

    This paper studies the effect of adding geometrically smoothed momentum to the randomized Kaczmarz algorithm, which is an instance of stochastic gradient descent on a linear least squares loss function. We prove a result about the expected error in the direction of singular vectors of the matrix defining the least squares loss. We present several numerical examples illustrating the utility of our result and pose several questions.
    
[^16]: 以核Fisher-Rao流进行单位时间采样

    Sampling in Unit Time with Kernel Fisher-Rao Flow. (arXiv:2401.03892v1 [stat.CO])

    [http://arxiv.org/abs/2401.03892](http://arxiv.org/abs/2401.03892)

    本文提出了一种具有核Fisher-Rao流的新方法，在单位时间内从非归一化目标密度或贝叶斯后验中进行采样。方法使用了均场ODE和相互作用粒子系统，无需梯度，只需要能够从参考密度中采样并计算目标对参考密度的比率。该方法通过在几何混合的路径上沿速度场运输样本，径向输运样本。方法通过在再生核希尔伯特空间中求解泊松方程，使泊松方程的求解变得可行，并将其离散化为有限样本的均场ODE，作为实现简单的相互作用粒子系统。同时，这种方法也可以从离散时间的角度推导出均场ODE，作为蒙杰-安普尔方程连续线性化的极限。

    

    我们引入了一种新的均场ODE和相应的相互作用粒子系统，用于从非归一化的目标密度或贝叶斯后验中进行采样。相互作用粒子系统无需梯度，可以闭合形式获得，并且只需要能够从参考密度中采样并计算（非归一化的）目标对参考密度的比率。通过求解运输样本沿两个密度的几何混合的速度场的泊松方程来获得均场ODE，这是一种特定的Fisher-Rao梯度流的路径。我们采用再生核希尔伯特空间方法来获得速度场的泊松方程，这使得泊松方程可处理，并使我们能够离散化有限样本的结果均场ODE，形成一个简单的相互作用粒子系统。均场ODE还可以通过离散时间视角从蒙杰-安普尔方程的连续线性化的极限中推导出来，这在一个已知的框架内进行。

    We introduce a new mean-field ODE and corresponding interacting particle systems for sampling from an unnormalized target density or Bayesian posterior. The interacting particle systems are gradient-free, available in closed form, and only require the ability to sample from the reference density and compute the (unnormalized) target-to-reference density ratio. The mean-field ODE is obtained by solving a Poisson equation for a velocity field that transports samples along the geometric mixture of the two densities, which is the path of a particular Fisher-Rao gradient flow. We employ a reproducing kernel Hilbert space ansatz for the velocity field, which makes the Poisson equation tractable and enables us to discretize the resulting mean-field ODE over finite samples, as a simple interacting particle system. The mean-field ODE can be additionally be derived from a discrete-time perspective as the limit of successive linearizations of the Monge-Amp\`ere equations within a framework known 
    
[^17]: 用基于模拟推断的孤立脉冲星种群合成

    Isolated pulsar population synthesis with simulation-based inference. (arXiv:2312.14848v1 [astro-ph.HE] CROSS LISTED)

    [http://arxiv.org/abs/2312.14848](http://arxiv.org/abs/2312.14848)

    本论文使用模拟推断方法结合脉冲星种群合成，来限制孤立银河射电脉冲星的磁旋转特性。

    

    我们将脉冲星种群合成与基于模拟推断相结合，以限制孤立银河射电脉冲星的磁旋转特性。我们首先构建了一个灵活的框架来模拟中子星的诞生特性和演化，重点是它们的动力学、旋转和磁性特征。特别是，我们从对数正态分布中采样初始磁场强度B和自转周期P，并用幂律来捕捉后期磁场的衰减。每个对数正态分布由均值μlogB，μlogP和标准差σlogB，σlogP描述，而幂律由指数a_late描述，共计五个自由参数。然后我们模拟了星体的射电发射和观测偏差，以模拟三个射电调查中的探测，并通过改变输入参数产生了一个大型的合成P-Ṗ图数据库。接着我们采用基于模拟推断的方法进行推断

    We combine pulsar population synthesis with simulation-based inference to constrain the magneto-rotational properties of isolated Galactic radio pulsars. We first develop a flexible framework to model neutron-star birth properties and evolution, focusing on their dynamical, rotational and magnetic characteristics. In particular, we sample initial magnetic-field strengths, $B$, and spin periods, $P$, from log-normal distributions and capture the late-time magnetic-field decay with a power law. Each log-normal is described by a mean, $\mu_{\log B}, \mu_{\log P}$, and standard deviation, $\sigma_{\log B}, \sigma_{\log P}$, while the power law is characterized by the index, $a_{\rm late}$, resulting in five free parameters. We subsequently model the stars' radio emission and observational biases to mimic detections with three radio surveys, and produce a large database of synthetic $P$-$\dot{P}$ diagrams by varying our input parameters. We then follow a simulation-based inference approach 
    
[^18]: CONFIDERAI：一种新颖的CONFIRMAL可解释设计评分函数，用于可解释和可靠的人工智能

    CONFIDERAI: a novel CONFormal Interpretable-by-Design score function forExplainable and Reliable Artificial Intelligence. (arXiv:2309.01778v1 [cs.LG])

    [http://arxiv.org/abs/2309.01778](http://arxiv.org/abs/2309.01778)

    提出了一种新的可解释机器学习评分函数CONFIDERAI，它将一致性预测与规则模型相结合，利用规则的预测能力和点的几何位置，在特征空间中定义满足一致性保证的区域。

    

    每天的生活越来越受人工智能的影响，毫无疑问，机器学习算法必须为所有人设计成可靠和值得信赖的。具体来说，如果人工智能系统满足解释性、健壮性、透明性、公平性和隐私性这五个方面，计算机科学家认为它是安全和可信赖的。除了这五个方面，我们提出了第六个基本方面：一致性，即机器学习者对系统行为的概率性保证。在本文中，我们提出了一种方法，通过定义CONFIDERAI，一种基于规则的模型的新评分函数，将一致性预测与可解释的机器学习相结合，利用规则的预测能力和点在规则边界内的几何位置。我们还通过利用控制非一致性的数量的技术来解决在特征空间中定义满足一致性保证的区域的问题。

    Everyday life is increasingly influenced by artificial intelligence, and there is no question that machine learning algorithms must be designed to be reliable and trustworthy for everyone. Specifically, computer scientists consider an artificial intelligence system safe and trustworthy if it fulfills five pillars: explainability, robustness, transparency, fairness, and privacy. In addition to these five, we propose a sixth fundamental aspect: conformity, that is, the probabilistic assurance that the system will behave as the machine learner expects. In this paper, we propose a methodology to link conformal prediction with explainable machine learning by defining CONFIDERAI, a new score function for rule-based models that leverages both rules predictive ability and points geometrical position within rules boundaries. We also address the problem of defining regions in the feature space where conformal guarantees are satisfied by exploiting techniques to control the number of non-conforma
    
[^19]: ARK: 鲁棒的耦合型Robust Knockoffs推理方法

    ARK: Robust Knockoffs Inference with Coupling. (arXiv:2307.04400v1 [stat.ME])

    [http://arxiv.org/abs/2307.04400](http://arxiv.org/abs/2307.04400)

    本研究探讨了在特征分布被错误估计或估计的情况下，通过耦合模型-X Knockoffs过程与近似Knockoffs过程，实现了在目标水平上的鲁棒的FDR或FWER控制。

    

    本研究通过在特征分布被错误估计或估计的情况下，理论上研究了实际实现的近似Knockoffs算法的特征选择性能，其中我们将该算法称为近似Knockoffs（ARK）过程。我们的理论分析关键技术是将近似Knockoffs过程与模型-X Knockoffs过程耦合，以使这两个过程中的随机变量在实现中接近。我们证明了如果存在这样的耦合模型-X Knockoffs过程，近似Knockoffs过程可以在目标水平上达到渐近的FDR或FWER控制。我们展示了三种具体的构建方法。

    We investigate the robustness of the model-X knockoffs framework with respect to the misspecified or estimated feature distribution. We achieve such a goal by theoretically studying the feature selection performance of a practically implemented knockoffs algorithm, which we name as the approximate knockoffs (ARK) procedure, under the measures of the false discovery rate (FDR) and family wise error rate (FWER). The approximate knockoffs procedure differs from the model-X knockoffs procedure only in that the former uses the misspecified or estimated feature distribution. A key technique in our theoretical analyses is to couple the approximate knockoffs procedure with the model-X knockoffs procedure so that random variables in these two procedures can be close in realizations. We prove that if such coupled model-X knockoffs procedure exists, the approximate knockoffs procedure can achieve the asymptotic FDR or FWER control at the target level. We showcase three specific constructions of s
    
[^20]: 深度随机力学

    Deep Stochastic Mechanics. (arXiv:2305.19685v1 [cs.LG])

    [http://arxiv.org/abs/2305.19685](http://arxiv.org/abs/2305.19685)

    本文提出了一种基于深度学习的方法，用于数值模拟时间演化薛定谔方程，利用马尔可夫扩散采样来适应波函数的潜在低维结构，并提出了新的随机量子力学方程，具有线性的计算复杂度。数值模拟显示出显着的优势。

    

    本文引入了一种基于深度学习的方法，用于数值模拟时间演化薛定谔方程，受随机力学和生成性扩散模型的启发。与现有方法不同的是，我们的方法允许我们通过从马尔可夫扩散中采样来适应波函数潜在的低维结构，因此可以在更高的维度上降低计算复杂度。此外，我们提出了新的随机量子力学方程，结果具有与维数数量线性的计算复杂度。数值模拟验证了我们的理论发现，并显示出我们的方法与其他用于量子力学的基于深度学习的方法相比具有显着优势。

    This paper introduces a novel deep-learning-based approach for numerical simulation of a time-evolving Schr\"odinger equation inspired by stochastic mechanics and generative diffusion models. Unlike existing approaches, which exhibit computational complexity that scales exponentially in the problem dimension, our method allows us to adapt to the latent low-dimensional structure of the wave function by sampling from the Markovian diffusion. Depending on the latent dimension, our method may have far lower computational complexity in higher dimensions. Moreover, we propose novel equations for stochastic quantum mechanics, resulting in linear computational complexity with respect to the number of dimensions. Numerical simulations verify our theoretical findings and show a significant advantage of our method compared to other deep-learning-based approaches used for quantum mechanics.
    
[^21]: 通过预测修正提高基于得分扩散模型的收敛性

    Improved Convergence of Score-Based Diffusion Models via Prediction-Correction. (arXiv:2305.14164v1 [cs.LG])

    [http://arxiv.org/abs/2305.14164](http://arxiv.org/abs/2305.14164)

    本文通过使用预测校正方案，提高了基于得分扩散模型的收敛性。

    

    基于得分的生成模型（SGM）是从复杂数据分布中进行采样的强大工具。其基本思想是（i）通过向数据添加噪声运行时间为$T_1$的正向过程，（ii）估计其得分函数，并（iii）使用此估计值运行反向过程。由于反向过程以正向过程的平稳分布作为初始值，因此现有的分析范式要求$T_1\to\infty$。然而，从理论角度来看，对于给定的分数逼近精度，当$T_1$发散时，收敛保证将失败；从实际角度来看，$T_1$越大，计算成本就越高，并且会导致误差传播。本文通过考虑流行的预测器校正方案的一个版本来解决这个问题：在运行正向过程之后，我们首先通过不精确的 Langevin 动力学估计最终分布，然后恢复该过程。我们的关键技术贡献是提供了收敛保证。

    Score-based generative models (SGMs) are powerful tools to sample from complex data distributions. Their underlying idea is to (i) run a forward process for time $T_1$ by adding noise to the data, (ii) estimate its score function, and (iii) use such estimate to run a reverse process. As the reverse process is initialized with the stationary distribution of the forward one, the existing analysis paradigm requires $T_1\to\infty$. This is however problematic: from a theoretical viewpoint, for a given precision of the score approximation, the convergence guarantee fails as $T_1$ diverges; from a practical viewpoint, a large $T_1$ increases computational costs and leads to error propagation. This paper addresses the issue by considering a version of the popular predictor-corrector scheme: after running the forward process, we first estimate the final distribution via an inexact Langevin dynamics and then revert the process. Our key technical contribution is to provide convergence guarantees
    
[^22]: Group-Invariant GAN的统计保证

    Statistical Guarantees of Group-Invariant GANs. (arXiv:2305.13517v1 [stat.ML])

    [http://arxiv.org/abs/2305.13517](http://arxiv.org/abs/2305.13517)

    本研究提出了群不变GAN的统计保证，发现当学习群不变分布时，群不变GAN所需样本数会按群体大小的幂比例减少。

    

    Group-Invariant生成对抗网络(GAN)是一种GAN，其中生成器和判别器具有硬性集团对称性。实证研究表明，这些网络能够学习具有显着改进数据效率的集团不变分布。在本研究中，我们旨在通过分析群不变GAN的样本复杂度减少来严格量化这种改进。我们的研究发现，在学习群不变分布时，群不变GAN所需样本数按照群体大小的幂比例减少，这个幂取决于分布支持的本质维度。据我们所知，这项工作是首个为群不变生成模型，特别是GAN提供统计估计的工作，并可以为其他群不变生成模型的研究提供借鉴。

    Group-invariant generative adversarial networks (GANs) are a type of GANs in which the generators and discriminators are hardwired with group symmetries. Empirical studies have shown that these networks are capable of learning group-invariant distributions with significantly improved data efficiency. In this study, we aim to rigorously quantify this improvement by analyzing the reduction in sample complexity for group-invariant GANs. Our findings indicate that when learning group-invariant distributions, the number of samples required for group-invariant GANs decreases proportionally with a power of the group size, and this power depends on the intrinsic dimension of the distribution's support. To our knowledge, this work presents the first statistical estimation for group-invariant generative models, specifically for GANs, and it may shed light on the study of other group-invariant generative models.
    
[^23]: 权重具有无界方差的无限宽贝叶斯神经网络后验推断

    Posterior Inference on Infinitely Wide Bayesian Neural Networks under Weights with Unbounded Variance. (arXiv:2305.10664v1 [stat.ML])

    [http://arxiv.org/abs/2305.10664](http://arxiv.org/abs/2305.10664)

    本文提出了一种新的方法进行关于具有无界方差权重的贝叶斯神经网络的后验推断，并表明后验分布集中在具有非标准超参数依赖性的稀疏促进和均值收缩先验周围。

    

    由Neal（1996）的经典而有影响力的作品已知，具有一层隐藏层的贝叶斯神经网络的无限宽度标度极限是一个高斯过程，当网络权重具有有界先验方差时。Neal的结果已扩展到具有多个隐藏层和卷积神经网络的网络，也具有高斯过程标度极限。高斯过程的易处理属性允许直接的后验推断和不确定性量化，相比有限宽度的网络，极大地简化了极限过程的研究。然而，具有无界方差的神经网络权重面临着独特的挑战。在这种情况下，经典的中心极限定理失效，据适当条件下的稳定$\alpha$过程的标度极限的文献较多的是前向模拟，而在这些权重下的后验推断问题仍然是一个未解决的问题。在本文中，我们提出了关于具有无界方差权重的贝叶斯神经网络后验推断的新理论洞察力。具体而言，我们建立了一种新的后验收缩速率结果，并表明后验分布集中在具有非标准超参数依赖性的稀疏促进和均值收缩先验周围。

    From the classical and influential works of Neal (1996), it is known that the infinite width scaling limit of a Bayesian neural network with one hidden layer is a Gaussian process, \emph{when the network weights have bounded prior variance}. Neal's result has been extended to networks with multiple hidden layers and to convolutional neural networks, also with Gaussian process scaling limits. The tractable properties of Gaussian processes then allow straightforward posterior inference and uncertainty quantification, considerably simplifying the study of the limit process compared to a network of finite width. Neural network weights with unbounded variance, however, pose unique challenges. In this case, the classical central limit theorem breaks down and it is well known that the scaling limit is an $\alpha$-stable process under suitable conditions. However, current literature is primarily limited to forward simulations under these processes and the problem of posterior inference under s
    
[^24]: 《无免费午餐定理、科尔莫戈洛夫复杂性及归纳偏差在机器学习中的作用》

    The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning. (arXiv:2304.05366v1 [cs.LG])

    [http://arxiv.org/abs/2304.05366](http://arxiv.org/abs/2304.05366)

    本论文阐述了无免费午餐定理的监督学习中的限制，证明了归纳偏差可以提高学习算法的效果，并且展示了神经网络模型的偏好与现实世界的数据分布相关。

    

    监督学习的无免费午餐定理指出，没有一个学习算法可以解决所有问题，或者所有学习算法在均匀分布的学习问题上平均精度达到完全相同。因此，这些定理经常被引用来支持个别问题需要特别定制的归纳偏差的概念。我们认为，尽管几乎所有均匀采样的数据集具有高复杂性，但现实世界中的问题不成比例地产生低复杂度的数据，并且我们认为神经网络模型也具有同样的偏好，这种偏好使用科尔莫戈洛夫复杂度进行了形式化。值得注意的是，我们展示了为特定领域设计的体系结构，例如计算机视觉，可以压缩各种看似不相关的领域的数据集。我们的实验表明，预先训练和即使是随机初始化的语言模型都更喜欢生成低复杂度的序列。尽管无免费午餐定理似乎表明各个问题需要专门的学习算法，但我们解释说，学习算法通常可以通过编码关于真实世界数据分布的先前知识的归纳偏差来改进。

    No free lunch theorems for supervised learning state that no learner can solve all problems or that all learners achieve exactly the same accuracy on average over a uniform distribution on learning problems. Accordingly, these theorems are often referenced in support of the notion that individual problems require specially tailored inductive biases. While virtually all uniformly sampled datasets have high complexity, real-world problems disproportionately generate low-complexity data, and we argue that neural network models share this same preference, formalized using Kolmogorov complexity. Notably, we show that architectures designed for a particular domain, such as computer vision, can compress datasets on a variety of seemingly unrelated domains. Our experiments show that pre-trained and even randomly initialized language models prefer to generate low-complexity sequences. Whereas no free lunch theorems seemingly indicate that individual problems require specialized learners, we exp
    
[^25]: 基于编码器嵌入的协同图融合

    Synergistic Graph Fusion via Encoder Embedding. (arXiv:2303.18051v1 [cs.SI])

    [http://arxiv.org/abs/2303.18051](http://arxiv.org/abs/2303.18051)

    本文提出了一种协同图融合的新方法，该方法处理具有共同顶点集的多个图，有着非常理想的“协同效应”，即顶点分类准确度总是受益于额外的图，并在实验中证实了其卓越性能。

    

    本文提出了一种称为图融合编码器嵌入的多图嵌入新方法，该方法旨在处理具有共同顶点集的多个图。在监督学习设置下，我们证明了该方法展现出了令人惊叹但非常理想的“协同效应”：对于足够大的顶点数，分类准确度总是受益于额外的图。我们在随机块模型下提供了这种效应的数学证明，并确定了渐近完美分类的必要条件和充分条件。模拟和真实数据实验证实了所提出的方法的卓越性能，该方法在分类中始终优于最近的基准方法。

    In this paper, we introduce a novel approach to multi-graph embedding called graph fusion encoder embedding. The method is designed to work with multiple graphs that share a common vertex set. Under the supervised learning setting, we show that the resulting embedding exhibits a surprising yet highly desirable "synergistic effect": for sufficiently large vertex size, the vertex classification accuracy always benefits from additional graphs. We provide a mathematical proof of this effect under the stochastic block model, and identify the necessary and sufficient condition for asymptotically perfect classification. The simulations and real data experiments confirm the superiority of the proposed method, which consistently outperforms recent benchmark methods in classification.
    
[^26]: 动态排名和翻译同步

    Dynamic Ranking and Translation Synchronization. (arXiv:2207.01455v3 [math.ST] UPDATED)

    [http://arxiv.org/abs/2207.01455](http://arxiv.org/abs/2207.01455)

    本论文研究了动态排名和翻译同步问题，主要关注成对比较数据随时间变化的情况，并给出了相应的理论结果。

    

    在许多应用中，如体育比赛或推荐系统，我们可以获得由一组$n$个项目（或选手）之间的成对比较组成的数据。目标是利用这些数据推断每个项目和/或它们的排名的潜在实力。现有结果主要关注单个比较图$G$的设置。然而，在某些情况下（如体育比赛），成对比较数据会随时间变化。对于这种动态设置，理论结果相对有限，是本文的重点。我们研究了翻译同步问题在动态设置下的扩展。在这个设置中，我们给定了一个比较图序列$(G_t)_{t\in \mathcal{T}}$，其中$\mathcal{T} \subset [0,1]$是表示时间域的格点，对于每个项目$i$和时间$t\in \mathcal{T}$，存在一个关联的未知实力参数$z^*_{t,i}\in \mathbb{R}$。

    In many applications, such as sport tournaments or recommendation systems, we have at our disposal data consisting of pairwise comparisons between a set of $n$ items (or players). The objective is to use this data to infer the latent strength of each item and/or their ranking. Existing results for this problem predominantly focus on the setting consisting of a single comparison graph $G$. However, there exist scenarios (e.g., sports tournaments) where the the pairwise comparison data evolves with time. Theoretical results for this dynamic setting are relatively limited and is the focus of this paper.  We study an extension of the \emph{translation synchronization} problem, to the dynamic setting. In this setup, we are given a sequence of comparison graphs $(G_t)_{t\in \mathcal{T}}$, where $\mathcal{T} \subset [0,1]$ is a grid representing the time domain, and for each item $i$ and time $t\in \mathcal{T}$ there is an associated unknown strength parameter $z^*_{t,i}\in \mathbb{R}$. We ai
    

