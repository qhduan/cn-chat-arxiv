# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Policy Mirror Descent with Lookahead](https://arxiv.org/abs/2403.14156) | 提出了一种新类别的策略镜像下降算法$h$-PMD，它通过在PMD更新规则中结合多步贪心策略改进和前瞻深度$h，以解决折扣无限时间视角下的马尔可夫决策过程。 |
| [^2] | [Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference](https://arxiv.org/abs/2403.04082) | 通过对比学习学到的时间序列数据表示遵循高斯马尔可夫链，从而启用规划和推断 |
| [^3] | [Epsilon-Greedy Thompson Sampling to Bayesian Optimization](https://arxiv.org/abs/2403.00540) | 将$\varepsilon$-greedy策略引入Thompson采样以改进贝叶斯优化中的开发功能，并实证表明其有效性。 |
| [^4] | [Implicit Bias of Next-Token Prediction](https://arxiv.org/abs/2402.18551) | 在基于梯度的优化器训练下的线性NTP模型中，确定了NTP可分离条件，并证明梯度下降能够实现其下界；同时证明了这些条件在过参数化时仍然成立。 |
| [^5] | [Achieving Near-Optimal Regret for Bandit Algorithms with Uniform Last-Iterate Guarantee](https://arxiv.org/abs/2402.12711) | 本文引入了统一最后迭代(ULI)保证这一更强的性能度量，同时证明接近最优的ULI保证直接导致了在各种性能度量上接近最优的累积性能。 |
| [^6] | [Self-Consistent Conformal Prediction](https://arxiv.org/abs/2402.07307) | 自洽的符合预测方法能够提供既符合校准的预测又符合以模型预测的动作为条件的预测区间，为决策者提供了严格的、针对具体动作的决策保证。 |
| [^7] | [Metrics on Markov Equivalence Classes for Evaluating Causal Discovery Algorithms](https://arxiv.org/abs/2402.04952) | 本文提出了三个新的距离度量指标（s/c距离、马尔科夫距离和忠实度距离），用于评估因果推断算法的输出图与真实情况的分离/连接程度。 |
| [^8] | [Learning Best-in-Class Policies for the Predict-then-Optimize Framework](https://arxiv.org/abs/2402.03256) | 我们提出了一种新颖的决策感知替代损失函数家族，用于predict-then-optimize框架，并且通过数值证据证实了其在误设置下的优越性。 |
| [^9] | [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://arxiv.org/abs/2312.02119) | 提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成只需要对目标大型语言模型进行黑盒访问的越狱方法，并通过思维树推理和修剪生成准确的越狱提示。 |
| [^10] | [Ensemble sampling for linear bandits: small ensembles suffice](https://arxiv.org/abs/2311.08376) | 该论文对随机线性赌臂环境中的集成抽样进行了首次实用和严格的分析，展示了在标准假设下，采用规模为$d \log T$的集成抽样可以获得接近$\sqrt{T}$阶的后悔，而不需要集成大小与$T$线性扩展。 |
| [^11] | [A powerful rank-based correction to multiple testing under positive dependency.](http://arxiv.org/abs/2311.10900) | 我们提出了一种基于秩的多重检验修正方法，能够有效利用正相关的统计假设检验之间的依赖关系，并在存在正相关依赖情况下优于Bonferroni修正。我们的方法尤其适用于并行置换检验，在保证FWER控制的同时保持高统计功效。 |
| [^12] | [Generative Fractional Diffusion Models.](http://arxiv.org/abs/2310.17638) | 该论文提出了一种基于分数阶扩散模型的生成模型，其中通过将分数布朗运动表示为奥恩斯坦-乌伦贝克过程的随机积分，实现了驱动噪声收敛到非马尔可夫过程的效果。这是第一个在具有无限二次变差的随机过程上构建生成模型的尝试。 |
| [^13] | [Controlling Continuous Relaxation for Combinatorial Optimization.](http://arxiv.org/abs/2309.16965) | 本文研究了在相对密集的图上组合优化问题中物理启发的图神经网络（PI-GNN）求解器的表现。通过数值实验，我们发现PI-GNN求解器在学习早期可能陷入所有变量为零的局部解。为了解决这个问题，我们通过控制连续性和离散性提出了一种改进方法。 |
| [^14] | [Infeasible Deterministic, Stochastic, and Variance-Reduction Algorithms for Optimization under Orthogonality Constraints.](http://arxiv.org/abs/2303.16510) | 本研究提出了一种简单迭代的Landing算法，可以在不强制执行正交约束的同时顺畅地吸引到正交约束流形上。我们扩展了这种算法以支持斯托菲尔（Stiefel）流形，并提供了随机和方差约减算法，这些方法与黎曼优化算法的收敛速度相同但需要更少的计算。 |
| [^15] | [On the tightness of information-theoretic bounds on generalization error of learning algorithms.](http://arxiv.org/abs/2303.14658) | 本文研究了学习算法泛化误差信息理论界限的紧密性。研究表明，通过适当的假设，可以在快速收敛速度下使用信息理论量$O(\lambda/n)$来上界估计泛化误差。 |

# 详细

[^1]: 具有前瞻特性的策略镜像下降算法

    Policy Mirror Descent with Lookahead

    [https://arxiv.org/abs/2403.14156](https://arxiv.org/abs/2403.14156)

    提出了一种新类别的策略镜像下降算法$h$-PMD，它通过在PMD更新规则中结合多步贪心策略改进和前瞻深度$h，以解决折扣无限时间视角下的马尔可夫决策过程。

    

    策略镜像下降（PMD）作为一种多功能算法框架，包括几种重要的策略梯度算法，如自然策略梯度，并与最先进的强化学习（RL）算法（如TRPO和PPO）相联系。PMD可以看作是实现正则化1步贪心策略改进的软策略迭代算法。然而，1步贪心策略可能不是最佳选择，最近在RL领域取得了显着的实证成功，如AlphaGo和AlphaZero已经证明，相对于多步骤，贪心方法可以超越它们的1步骤对应物。在这项工作中，我们提出了一种新类别的PMD算法，称为$h$-PMD，它将具有前瞻深度$h$的多步贪心策略改进结合到PMD更新规则中。为了解决折扣无限时间视角下的马尔可夫决策过程，其中折扣因子为$\gamma$，我们展示了$h$-PMD可以推广标准的PMD。

    arXiv:2403.14156v1 Announce Type: cross  Abstract: Policy Mirror Descent (PMD) stands as a versatile algorithmic framework encompassing several seminal policy gradient algorithms such as natural policy gradient, with connections with state-of-the-art reinforcement learning (RL) algorithms such as TRPO and PPO. PMD can be seen as a soft Policy Iteration algorithm implementing regularized 1-step greedy policy improvement. However, 1-step greedy policies might not be the best choice and recent remarkable empirical successes in RL such as AlphaGo and AlphaZero have demonstrated that greedy approaches with respect to multiple steps outperform their 1-step counterpart. In this work, we propose a new class of PMD algorithms called $h$-PMD which incorporates multi-step greedy policy improvement with lookahead depth $h$ to the PMD update rule. To solve discounted infinite horizon Markov Decision Processes with discount factor $\gamma$, we show that $h$-PMD which generalizes the standard PMD enj
    
[^2]: 通过插值进行推断：对比表示可证明启用规划和推断

    Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference

    [https://arxiv.org/abs/2403.04082](https://arxiv.org/abs/2403.04082)

    通过对比学习学到的时间序列数据表示遵循高斯马尔可夫链，从而启用规划和推断

    

    给定时间序列数据，我们如何回答诸如“未来会发生什么？”和“我们是如何到达这里的？”这类概率推断问题在观测值为高维时具有挑战性。本文展示了这些问题如何通过学习表示的紧凑闭式解决方案。关键思想是将对比学习的变体应用于时间序列数据。之前的工作已经表明，通过对比学习学到的表示编码了概率比。通过将之前的工作扩展以表明表示的边际分布是高斯分布，我们随后证明表示的联合分布也是高斯分布。这些结果共同表明，通过时间对比学习学到的表示遵循高斯马尔可夫链，一种图形模型，其中对表示进行的推断（例如预测、规划）对应于反演低维分布。

    arXiv:2403.04082v1 Announce Type: new  Abstract: Given time series data, how can we answer questions like "what will happen in the future?" and "how did we get here?" These sorts of probabilistic inference questions are challenging when observations are high-dimensional. In this paper, we show how these questions can have compact, closed form solutions in terms of learned representations. The key idea is to apply a variant of contrastive learning to time series data. Prior work already shows that the representations learned by contrastive learning encode a probability ratio. By extending prior work to show that the marginal distribution over representations is Gaussian, we can then prove that joint distribution of representations is also Gaussian. Taken together, these results show that representations learned via temporal contrastive learning follow a Gauss-Markov chain, a graphical model where inference (e.g., prediction, planning) over representations corresponds to inverting a low-
    
[^3]: Epsilon-Greedy Thompson Sampling用于贝叶斯优化

    Epsilon-Greedy Thompson Sampling to Bayesian Optimization

    [https://arxiv.org/abs/2403.00540](https://arxiv.org/abs/2403.00540)

    将$\varepsilon$-greedy策略引入Thompson采样以改进贝叶斯优化中的开发功能，并实证表明其有效性。

    

    Thompson采样（TS）被认为是解决贝叶斯优化中开发-探索困境的解决方案。 虽然它通过随机生成和最大化高斯过程（GP）后验的样本路径来优先进行探索，但TS在每次执行探索后通过收集关于真实目标函数的信息来弱化其开发功能。 本研究将在TS中引入$\varepsilon$-greedy策略，这是一种在强化学习中被广泛应用的选择策略，以改进其开发功能。 我们首先描述了TS应用于BO的两个极端，即通用TS和样本平均TS。前者和后者分别提倡探索和开发。 然后我们使用$\varepsilon$-greedy策略在两个极端之间随机切换。 $\varepsilon \in (0,1)$的小值优先考虑开发，反之亦然。 我们实证表明$\varepsilon$-greedy T

    arXiv:2403.00540v1 Announce Type: new  Abstract: Thompson sampling (TS) serves as a solution for addressing the exploitation-exploration dilemma in Bayesian optimization (BO). While it prioritizes exploration by randomly generating and maximizing sample paths of Gaussian process (GP) posteriors, TS weakly manages its exploitation by gathering information about the true objective function after each exploration is performed. In this study, we incorporate the epsilon-greedy ($\varepsilon$-greedy) policy, a well-established selection strategy in reinforcement learning, into TS to improve its exploitation. We first delineate two extremes of TS applied for BO, namely the generic TS and a sample-average TS. The former and latter promote exploration and exploitation, respectively. We then use $\varepsilon$-greedy policy to randomly switch between the two extremes. A small value of $\varepsilon \in (0,1)$ prioritizes exploitation, and vice versa. We empirically show that $\varepsilon$-greedy T
    
[^4]: 隐性偏见的下一个标记预测

    Implicit Bias of Next-Token Prediction

    [https://arxiv.org/abs/2402.18551](https://arxiv.org/abs/2402.18551)

    在基于梯度的优化器训练下的线性NTP模型中，确定了NTP可分离条件，并证明梯度下降能够实现其下界；同时证明了这些条件在过参数化时仍然成立。

    

    下一个标记预测（NTP）是训练大型语言模型的首选范式，它涉及预测序列中的下一个标记。与传统的独热分类不同，在NTP中，多个具有不同频率的标记在给定上下文后继。本文将NTP训练框架化为跨不同上下文的交叉熵最小化，每个上下文都与有限词汇表中的稀疏经验概率向量相关联。然后，它探讨了以下问题：当NTP训练损失达到其下界（熵）时，基于梯度的优化器是否会对具有特定结构的解决方案存在偏见？具体地，对于使用梯度下降（GD）训练的线性NTP模型，我们做出以下贡献：首先，我们确定了数据上的NTP可分离条件，在这些条件下，GD能够达到其下界。我们还证明了这些条件在过参数化时仍成立。

    arXiv:2402.18551v1 Announce Type: cross  Abstract: Next-token prediction (NTP), the go-to training paradigm in training large language models, involves predicting the next token in a sequence. Departing from traditional one-hot classification, in NTP, multiple tokens with varying frequencies follow each given context. This work frames NTP training as cross-entropy minimization over distinct contexts, each associated with a sparse empirical probability vector across a finite vocabulary. It then addresses the following question: do gradient-based optimizers exhibit a bias towards solutions with specific structure as the NTP training loss reaches its lower bound (entropy)? Specifically, for linear NTP models trained using gradient descent (GD), we make the following contributions: Firstly, we determine NTP-separability conditions on the data, under which GD can attain its lower bound. We also demonstrate that these conditions hold under overparameterization. Secondly, we establish that th
    
[^5]: 具有统一最后迭代保证的赌博算法实现接近最优遗憾

    Achieving Near-Optimal Regret for Bandit Algorithms with Uniform Last-Iterate Guarantee

    [https://arxiv.org/abs/2402.12711](https://arxiv.org/abs/2402.12711)

    本文引入了统一最后迭代(ULI)保证这一更强的性能度量，同时证明接近最优的ULI保证直接导致了在各种性能度量上接近最优的累积性能。

    

    现有的赌博算法性能度量，如遗憾、PAC界限或统一PAC(Dann等人，2017)，通常评估累积性能，同时允许在任意有限时间t内玩弱劣的臂。这种行为在高风险应用中可能造成严重损失。本文介绍了一种更强的性能度量，统一最后迭代(ULI)保证，捕捉赌博算法的累积和瞬时性能。具体来说，ULI表征了瞬时性能，因为它确保所玩弱劣臂的每轮遗憾受到一个函数的限制，该函数随着（大）轮次t单调递减，在有足够样本可用时防止重复访问劣质臂。我们证明，接近最优的ULI保证直接意味着在上述性能度量中实现接近最优的累积性能。为了研究ULI在有限臂集上的可达性

    arXiv:2402.12711v1 Announce Type: new  Abstract: Existing performance measures for bandit algorithms such as regret, PAC bounds, or uniform-PAC (Dann et al., 2017), typically evaluate the cumulative performance, while allowing the play of an arbitrarily bad arm at any finite time t. Such a behavior can be highly detrimental in high-stakes applications. This paper introduces a stronger performance measure, the uniform last-iterate (ULI) guarantee, capturing both cumulative and instantaneous performance of bandit algorithms. Specifically, ULI characterizes the instantaneous performance since it ensures that the per-round regret of the played arm is bounded by a function, monotonically decreasing w.r.t. (large) round t, preventing revisits to bad arms when sufficient samples are available. We demonstrate that a near-optimal ULI guarantee directly implies near-optimal cumulative performance across aforementioned performance measures. To examine the achievability of ULI in the finite arm se
    
[^6]: 自洽的符合预测

    Self-Consistent Conformal Prediction

    [https://arxiv.org/abs/2402.07307](https://arxiv.org/abs/2402.07307)

    自洽的符合预测方法能够提供既符合校准的预测又符合以模型预测的动作为条件的预测区间，为决策者提供了严格的、针对具体动作的决策保证。

    

    在机器学习指导下的决策中，决策者通常在具有相同预测结果的情境中采取相同的行动。符合预测帮助决策者量化动作的结果不确定性，从而实现更好的风险管理。受这种观点的启发，我们引入了自洽的符合预测，它产生了既符合Venn-Abers校准的预测，又符合以模型预测引发的动作为条件的符合预测区间。我们的方法可以后验地应用于任何黑盒预测器，提供严格的、针对具体动作的决策保证。数值实验表明，我们的方法在区间的效率和条件的有效性之间达到了平衡。

    In decision-making guided by machine learning, decision-makers often take identical actions in contexts with identical predicted outcomes. Conformal prediction helps decision-makers quantify outcome uncertainty for actions, allowing for better risk management. Inspired by this perspective, we introduce self-consistent conformal prediction, which yields both Venn-Abers calibrated predictions and conformal prediction intervals that are valid conditional on actions prompted by model predictions. Our procedure can be applied post-hoc to any black-box predictor to provide rigorous, action-specific decision-making guarantees. Numerical experiments show our approach strikes a balance between interval efficiency and conditional validity.
    
[^7]: 评估因果推断算法的马尔科夫等价类指标

    Metrics on Markov Equivalence Classes for Evaluating Causal Discovery Algorithms

    [https://arxiv.org/abs/2402.04952](https://arxiv.org/abs/2402.04952)

    本文提出了三个新的距离度量指标（s/c距离、马尔科夫距离和忠实度距离），用于评估因果推断算法的输出图与真实情况的分离/连接程度。

    

    许多最先进的因果推断方法旨在生成一个输出图，该图编码了生成数据过程的因果图的图形分离和连接陈述。在本文中，我们认为，对合成数据的因果推断方法进行评估应该包括分析该方法的输出与真实情况的分离/连接程度，以衡量这一明确目标的实现情况。我们证明现有的评估指标不能准确捕捉到两个因果图的分离/连接差异，并引入了三个新的距离度量指标，即s/c距离、马尔科夫距离和忠实度距离，以解决这个问题。我们通过玩具示例、实证实验和伪代码来补充我们的理论分析。

    Many state-of-the-art causal discovery methods aim to generate an output graph that encodes the graphical separation and connection statements of the causal graph that underlies the data-generating process. In this work, we argue that an evaluation of a causal discovery method against synthetic data should include an analysis of how well this explicit goal is achieved by measuring how closely the separations/connections of the method's output align with those of the ground truth. We show that established evaluation measures do not accurately capture the difference in separations/connections of two causal graphs, and we introduce three new measures of distance called s/c-distance, Markov distance and Faithfulness distance that address this shortcoming. We complement our theoretical analysis with toy examples, empirical experiments and pseudocode.
    
[^8]: 学习Predict-then-Optimize框架中的最优策略

    Learning Best-in-Class Policies for the Predict-then-Optimize Framework

    [https://arxiv.org/abs/2402.03256](https://arxiv.org/abs/2402.03256)

    我们提出了一种新颖的决策感知替代损失函数家族，用于predict-then-optimize框架，并且通过数值证据证实了其在误设置下的优越性。

    

    我们提出了一种新颖的决策感知替代损失函数家族，称为Perturbation Gradient（PG）损失，用于predict-then-optimize框架。这些损失直接近似了下游决策损失，并可以使用现成的基于梯度的方法进行优化。重要的是，与现有的替代损失不同，我们的PG损失的近似误差随着样本数量的增加而消失。这意味着优化我们的替代损失可以在渐近意义下得到最佳策略，即使在误设置下也是如此。这是第一个在误设置下的这样的结果，我们提供了数值证据证实了当基础模型误设置且噪声不是中心对称时，我们的PG损失在实践中显著优于现有的提案。鉴于在实践中误设置很常见--特别是当我们可能更喜欢一个更简单、更可解释的模型时--PG损失提供了一种新颖的、理论上有依据的、可计算的决策感知方法。

    We propose a novel family of decision-aware surrogate losses, called Perturbation Gradient (PG) losses, for the predict-then-optimize framework. These losses directly approximate the downstream decision loss and can be optimized using off-the-shelf gradient-based methods. Importantly, unlike existing surrogate losses, the approximation error of our PG losses vanishes as the number of samples grows. This implies that optimizing our surrogate loss yields a best-in-class policy asymptotically, even in misspecified settings. This is the first such result in misspecified settings and we provide numerical evidence confirming our PG losses substantively outperform existing proposals when the underlying model is misspecified and the noise is not centrally symmetric. Insofar as misspecification is commonplace in practice -- especially when we might prefer a simpler, more interpretable model -- PG losses offer a novel, theoretically justified, method for computationally tractable decision-aware 
    
[^9]: 攻击树：自动破解黑盒大型语言模型

    Tree of Attacks: Jailbreaking Black-Box LLMs Automatically

    [https://arxiv.org/abs/2312.02119](https://arxiv.org/abs/2312.02119)

    提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成只需要对目标大型语言模型进行黑盒访问的越狱方法，并通过思维树推理和修剪生成准确的越狱提示。

    

    大型语言模型(LLMs)展示了多功能性，但仍在生成有害、带偏见和有毒内容，这一点由人为设计的越狱行为的普遍存在得以证明。在这项工作中，我们提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成越狱，仅需要对目标LLM进行黑盒访问。TAP利用LLM来通过思维树推理迭代地优化候选（攻击）提示，直到生成的提示之一越狱目标。关键在于，在将提示发送给目标之前，TAP对其进行评估并移除可能不会导致越狱的提示。使用思维树推理使TAP能够在大量提示的搜索空间中导航，而修剪则减少了发送给目标的总查询数量。在实证评估中，我们观察到TAP生成的提示越狱了超过80%的最先进LLMs（包括GPT4和GPT4-Turbo）。

    arXiv:2312.02119v2 Announce Type: replace-cross  Abstract: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thought reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80%
    
[^10]: 线性赌臂的集成抽样：小集成足矣

    Ensemble sampling for linear bandits: small ensembles suffice

    [https://arxiv.org/abs/2311.08376](https://arxiv.org/abs/2311.08376)

    该论文对随机线性赌臂环境中的集成抽样进行了首次实用和严格的分析，展示了在标准假设下，采用规模为$d \log T$的集成抽样可以获得接近$\sqrt{T}$阶的后悔，而不需要集成大小与$T$线性扩展。

    

    我们首次对随机线性赌臂设定下的集成抽样进行了有用且严谨的分析。特别地，我们展示了在标准假设下，对于一个具有交互作用时间跨度$T$的$d$维随机线性赌臂，采用集成大小为$\smash{d \log T}$的集成抽样，遭受的后悔最多为$\smash{(d \log T)^{5/2} \sqrt{T}}$阶。我们的结果是在任何结构化环境中第一个不要求集成大小与$T$线性扩展的结果，这使得集成抽样失去意义，同时获得了接近$\smash{\sqrt{T}}$阶的后悔。我们的结果也是第一个允许无限动作集的结果。

    arXiv:2311.08376v2 Announce Type: replace-cross  Abstract: We provide the first useful and rigorous analysis of ensemble sampling for the stochastic linear bandit setting. In particular, we show that, under standard assumptions, for a $d$-dimensional stochastic linear bandit with an interaction horizon $T$, ensemble sampling with an ensemble of size of order $\smash{d \log T}$ incurs regret at most of the order $\smash{(d \log T)^{5/2} \sqrt{T}}$. Ours is the first result in any structured setting not to require the size of the ensemble to scale linearly with $T$ -- which defeats the purpose of ensemble sampling -- while obtaining near $\smash{\sqrt{T}}$ order regret. Ours is also the first result that allows infinite action sets.
    
[^11]: 一种基于秩的多重检验正相关依赖的强大修正方法

    A powerful rank-based correction to multiple testing under positive dependency. (arXiv:2311.10900v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2311.10900](http://arxiv.org/abs/2311.10900)

    我们提出了一种基于秩的多重检验修正方法，能够有效利用正相关的统计假设检验之间的依赖关系，并在存在正相关依赖情况下优于Bonferroni修正。我们的方法尤其适用于并行置换检验，在保证FWER控制的同时保持高统计功效。

    

    我们开发了一种能够高效利用可能相关的统计假设检验之间正相关性的家族误差率(FWER)控制的新型多重假设检验修正算法$\texttt{max-rank}$。我们的方法概念上很直观，依赖于在计算的统计检验的秩域使用$\max$算子。通过理论和经验的比较，我们证明了在存在正相关依赖的情况下，我们的方法优于经常使用的Bonferroni修正，而在不存在正相关依赖的情况下等效。我们的优势随着测试数量的增加而增加，同时在保证FWER控制的情况下保持高统计功效。我们特别将我们的算法应用于并行置换检验的背景中，这是在我们主要应用的一种复杂预测场景中产生的情况下。

    We develop a novel multiple hypothesis testing correction with family-wise error rate (FWER) control that efficiently exploits positive dependencies between potentially correlated statistical hypothesis tests. Our proposed algorithm $\texttt{max-rank}$ is conceptually straight-forward, relying on the use of a $\max$-operator in the rank domain of computed test statistics. We compare our approach to the frequently employed Bonferroni correction, theoretically and empirically demonstrating its superiority over Bonferroni in the case of existing positive dependency, and its equivalence otherwise. Our advantage over Bonferroni increases as the number of tests rises, and we maintain high statistical power whilst ensuring FWER control. We specifically frame our algorithm in the context of parallel permutation testing, a scenario that arises in our primary application of conformal prediction, a recently popularized approach for quantifying uncertainty in complex predictive settings.
    
[^12]: 生成分数阶扩散模型

    Generative Fractional Diffusion Models. (arXiv:2310.17638v1 [cs.LG])

    [http://arxiv.org/abs/2310.17638](http://arxiv.org/abs/2310.17638)

    该论文提出了一种基于分数阶扩散模型的生成模型，其中通过将分数布朗运动表示为奥恩斯坦-乌伦贝克过程的随机积分，实现了驱动噪声收敛到非马尔可夫过程的效果。这是第一个在具有无限二次变差的随机过程上构建生成模型的尝试。

    

    我们将基于分数布朗运动（FBM）的连续时间框架推广到基于分数布朗运动的近似形式。我们通过将FBM表示为家族奥恩斯坦-乌伦贝克过程的随机积分，推导出连续再参数化技巧和逆时模型，定义了具有驱动噪声收敛到无限二次变差的非马尔可夫过程的生成分数阶扩散模型（GFDM）。FBM的赫斯特指数$H \in (0,1)$ 可以控制路径变换分布的粗糙度。据我们所知，这是第一次尝试在具有无限二次变差的随机过程上建立生成模型。

    We generalize the continuous time framework for score-based generative models from an underlying Brownian motion (BM) to an approximation of fractional Brownian motion (FBM). We derive a continuous reparameterization trick and the reverse time model by representing FBM as a stochastic integral over a family of Ornstein-Uhlenbeck processes to define generative fractional diffusion models (GFDM) with driving noise converging to a non-Markovian process of infinite quadratic variation. The Hurst index $H\in(0,1)$ of FBM enables control of the roughness of the distribution transforming path. To the best of our knowledge, this is the first attempt to build a generative model upon a stochastic process with infinite quadratic variation.
    
[^13]: 控制组合优化的连续放松

    Controlling Continuous Relaxation for Combinatorial Optimization. (arXiv:2309.16965v1 [stat.ML])

    [http://arxiv.org/abs/2309.16965](http://arxiv.org/abs/2309.16965)

    本文研究了在相对密集的图上组合优化问题中物理启发的图神经网络（PI-GNN）求解器的表现。通过数值实验，我们发现PI-GNN求解器在学习早期可能陷入所有变量为零的局部解。为了解决这个问题，我们通过控制连续性和离散性提出了一种改进方法。

    

    最近在组合优化（CO）问题中，图神经网络（GNNs）显示出巨大潜力。通过无监督学习找到近似解的受物理启发的GNN（PI-GNN）求解器在大规模CO问题上引起了极大关注。然而，对于相对密集图上的CO问题，贪婪算法的性能恶化，但对于PI-GNN求解器的性能却没有太多讨论。此外，由于PI-GNN求解器采用了放松策略，学习后需要从连续空间人工转换回原始离散空间，可能会破坏解的鲁棒性。本文通过数值实验证明了PI-GNN求解器在密集图上的CO问题的学习早期可能陷入局部解的情况，其中所有变量都为零。然后，我们通过控制连续性和离散性来解决这些问题。

    Recent advancements in combinatorial optimization (CO) problems emphasize the potential of graph neural networks (GNNs). The physics-inspired GNN (PI-GNN) solver, which finds approximate solutions through unsupervised learning, has attracted significant attention for large-scale CO problems. Nevertheless, there has been limited discussion on the performance of the PI-GNN solver for CO problems on relatively dense graphs where the performance of greedy algorithms worsens. In addition, since the PI-GNN solver employs a relaxation strategy, an artificial transformation from the continuous space back to the original discrete space is necessary after learning, potentially undermining the robustness of the solutions. This paper numerically demonstrates that the PI-GNN solver can be trapped in a local solution, where all variables are zero, in the early stage of learning for CO problems on the dense graphs. Then, we address these problems by controlling the continuity and discreteness of rela
    
[^14]: 在正交约束下的优化问题中的不可行确定性、随机和方差约减算法

    Infeasible Deterministic, Stochastic, and Variance-Reduction Algorithms for Optimization under Orthogonality Constraints. (arXiv:2303.16510v1 [stat.ML])

    [http://arxiv.org/abs/2303.16510](http://arxiv.org/abs/2303.16510)

    本研究提出了一种简单迭代的Landing算法，可以在不强制执行正交约束的同时顺畅地吸引到正交约束流形上。我们扩展了这种算法以支持斯托菲尔（Stiefel）流形，并提供了随机和方差约减算法，这些方法与黎曼优化算法的收敛速度相同但需要更少的计算。

    

    正交约束在许多机器学习问题中都会自然地出现，从主成分分析到鲁棒性神经网络训练。传统上，这些问题需要使用黎曼优化算法来求解，该算法在强制执行约束时最耗费时间。最近，Ablin＆Peyr\'e（2022）提出了Landing算法，这是一种廉价迭代方法，它不强制执行正交约束，但会以平滑的方式吸引到流形上。在本文中，我们为Landing算法提供了新的实用和理论发展。首先，该方法被扩展到斯托菲尔流形，即矩形正交矩阵的集合。当成本函数是许多函数的平均值时，我们还考虑随机和方差约减算法。我们证明了所有这些方法的收敛速度与它们的黎曼优化算法相同，同时需要更少的计算。

    Orthogonality constraints naturally appear in many machine learning problems, from Principal Components Analysis to robust neural network training. They are usually solved using Riemannian optimization algorithms, which minimize the objective function while enforcing the constraint. However, enforcing the orthogonality constraint can be the most time-consuming operation in such algorithms. Recently, Ablin & Peyr\'e (2022) proposed the Landing algorithm, a method with cheap iterations that does not enforce the orthogonality constraint but is attracted towards the manifold in a smooth manner. In this article, we provide new practical and theoretical developments for the landing algorithm. First, the method is extended to the Stiefel manifold, the set of rectangular orthogonal matrices. We also consider stochastic and variance reduction algorithms when the cost function is an average of many functions. We demonstrate that all these methods have the same rate of convergence as their Rieman
    
[^15]: 关于学习算法泛化误差信息理论界限的紧密性研究

    On the tightness of information-theoretic bounds on generalization error of learning algorithms. (arXiv:2303.14658v1 [cs.IT])

    [http://arxiv.org/abs/2303.14658](http://arxiv.org/abs/2303.14658)

    本文研究了学习算法泛化误差信息理论界限的紧密性。研究表明，通过适当的假设，可以在快速收敛速度下使用信息理论量$O(\lambda/n)$来上界估计泛化误差。

    

    Russo和Xu提出了一种方法来证明学习算法的泛化误差可以通过信息度量进行上界估计。然而，这种收敛速度通常被认为是“慢”的，因为它的期望收敛速度的形式为$O(\sqrt{\lambda/n})$，其中$\lambda$是一些信息理论量。在本文中我们证明了根号并不一定意味着收敛速度慢，可以在适当的假设下使用这个界限来得到$O(\lambda/n)$的快速收敛速度。此外，我们确定了达到快速收敛速度的关键条件，即所谓的$(\eta,c)$-中心条件。在这个条件下，我们给出了学习算法泛化误差的信息理论界限。

    A recent line of works, initiated by Russo and Xu, has shown that the generalization error of a learning algorithm can be upper bounded by information measures. In most of the relevant works, the convergence rate of the expected generalization error is in the form of $O(\sqrt{\lambda/n})$ where $\lambda$ is some information-theoretic quantities such as the mutual information or conditional mutual information between the data and the learned hypothesis. However, such a learning rate is typically considered to be ``slow", compared to a ``fast rate" of $O(\lambda/n)$ in many learning scenarios. In this work, we first show that the square root does not necessarily imply a slow rate, and a fast rate result can still be obtained using this bound under appropriate assumptions. Furthermore, we identify the critical conditions needed for the fast rate generalization error, which we call the $(\eta,c)$-central condition. Under this condition, we give information-theoretic bounds on the generaliz
    

