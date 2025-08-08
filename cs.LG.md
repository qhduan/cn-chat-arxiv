# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustness of the data-driven approach in limited angle tomography](https://arxiv.org/abs/2403.11350) | 数据驱动方法在有限角度层析成像中相较于传统方法能更稳定地重构更多信息 |
| [^2] | [Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A](https://arxiv.org/abs/2402.13213) | 多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。 |
| [^3] | [Thompson Exploration with Best Challenger Rule in Best Arm Identification.](http://arxiv.org/abs/2310.00539) | 本文提出了一种新的策略，将Thompson采样与最佳候选规则相结合，用于解决最佳臂识别问题。该策略在渐近情况下是最优的，并在一般的多臂赌博机问题中达到接近最优的性能。 |
| [^4] | [Unsupervised Graph Deep Learning Reveals Emergent Flood Risk Profile of Urban Areas.](http://arxiv.org/abs/2309.14610) | 本研究基于无监督图深度学习模型提出了集成城市洪水风险评级模型，能够捕捉区域之间的空间依赖关系和洪水危险与城市要素之间的复杂相互作用，揭示了城市地区的突发洪水风险概况 |
| [^5] | [Eliciting Latent Predictions from Transformers with the Tuned Lens.](http://arxiv.org/abs/2303.08112) | 本文提出了一种改进版的“逻辑透镜”技术——“调谐透镜”，通过训练一个仿射探针，可以将每个隐藏状态解码成词汇分布。这个方法被应用于各种自回归语言模型上，比逻辑透镜更具有预测性、可靠性和无偏性，并且通过因果实验验证使用的特征与模型本身类似。同时，本文发现潜在预测的轨迹可以用于高精度地检测恶意输入。 |

# 详细

[^1]: 有限角度层析成像中数据驱动方法的鲁棒性

    Robustness of the data-driven approach in limited angle tomography

    [https://arxiv.org/abs/2403.11350](https://arxiv.org/abs/2403.11350)

    数据驱动方法在有限角度层析成像中相较于传统方法能更稳定地重构更多信息

    

    有限角度Radon变换由于其不逆问题而闻名于世。在这项工作中，我们给出了一个数学解释，即基于深度神经网络的数据驱动方法相较于传统方法可以以更稳定的方式重构更多信息。

    arXiv:2403.11350v1 Announce Type: cross  Abstract: The limited angle Radon transform is notoriously difficult to invert due to the ill-posedness. In this work, we give a mathematical explanation that the data-driven approach based on deep neural networks can reconstruct more information in a stable way compared to traditional methods.
    
[^2]: 软最大概率（大部分时候）在多项选择问答任务中预测大型语言模型的正确性

    Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A

    [https://arxiv.org/abs/2402.13213](https://arxiv.org/abs/2402.13213)

    多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。

    

    尽管大型语言模型（LLMs）在许多任务上表现出色，但过度自信仍然是一个问题。我们假设在多项选择问答任务中，错误答案将与最大softmax概率（MSPs）较小相关，相比之下正确答案较大。我们在十个开源LLMs和五个数据集上全面评估了这一假设，在表现良好的原始问答任务中发现了对我们假设的强有力证据。对于表现最佳的六个LLMs，从MSP导出的AUROC在59/60个实例中都优于随机机会，p < 10^{-4}。在这六个LLMs中，平均AUROC范围在60%至69%之间。利用这些发现，我们提出了一个带有弃权选项的多项选择问答任务，并展示通过根据初始模型响应的MSP有选择地弃权可以提高性能。我们还用预softmax logits而不是softmax进行了相同的实验。

    arXiv:2402.13213v1 Announce Type: cross  Abstract: Although large language models (LLMs) perform impressively on many tasks, overconfidence remains a problem. We hypothesized that on multiple-choice Q&A tasks, wrong answers would be associated with smaller maximum softmax probabilities (MSPs) compared to correct answers. We comprehensively evaluate this hypothesis on ten open-source LLMs and five datasets, and find strong evidence for our hypothesis among models which perform well on the original Q&A task. For the six LLMs with the best Q&A performance, the AUROC derived from the MSP was better than random chance with p < 10^{-4} in 59/60 instances. Among those six LLMs, the average AUROC ranged from 60% to 69%. Leveraging these findings, we propose a multiple-choice Q&A task with an option to abstain and show that performance can be improved by selectively abstaining based on the MSP of the initial model response. We also run the same experiments with pre-softmax logits instead of sof
    
[^3]: 最佳候选规则下的Thompson探索在最佳臂识别中的应用

    Thompson Exploration with Best Challenger Rule in Best Arm Identification. (arXiv:2310.00539v1 [stat.ML])

    [http://arxiv.org/abs/2310.00539](http://arxiv.org/abs/2310.00539)

    本文提出了一种新的策略，将Thompson采样与最佳候选规则相结合，用于解决最佳臂识别问题。该策略在渐近情况下是最优的，并在一般的多臂赌博机问题中达到接近最优的性能。

    

    本文研究了在经典单参数指数模型下，固定置信度下的最佳臂识别（BAI）问题。针对这个问题，目前已有很多策略被提出，但大多数需要在每一轮解决一个最优化问题和/或者需要探索一个臂至少一定次数，除非是针对高斯模型的限制。为了解决这些限制，我们提出了一种新的策略，将Thompson采样与一个计算效率高的方法——最佳候选规则相结合。虽然Thompson采样最初被考虑用于最大化累积奖励，但我们证明它也可以自然地用于在BAI中探索臂而不强迫最大化奖励。我们证明了我们的策略在任意两臂赌博机问题上是渐近最优的，并且在一般的$K$臂赌博机问题上（$K\geq 3$）达到接近最优的性能。然而，在数值实验中，我们的策略与现有方法相比表现出了竞争性的性能。

    This paper studies the fixed-confidence best arm identification (BAI) problem in the bandit framework in the canonical single-parameter exponential models. For this problem, many policies have been proposed, but most of them require solving an optimization problem at every round and/or are forced to explore an arm at least a certain number of times except those restricted to the Gaussian model. To address these limitations, we propose a novel policy that combines Thompson sampling with a computationally efficient approach known as the best challenger rule. While Thompson sampling was originally considered for maximizing the cumulative reward, we demonstrate that it can be used to naturally explore arms in BAI without forcing it. We show that our policy is asymptotically optimal for any two-armed bandit problems and achieves near optimality for general $K$-armed bandit problems for $K\geq 3$. Nevertheless, in numerical experiments, our policy shows competitive performance compared to as
    
[^4]: 无监督的图深度学习揭示了城市地区突发洪水风险概况

    Unsupervised Graph Deep Learning Reveals Emergent Flood Risk Profile of Urban Areas. (arXiv:2309.14610v1 [cs.LG])

    [http://arxiv.org/abs/2309.14610](http://arxiv.org/abs/2309.14610)

    本研究基于无监督图深度学习模型提出了集成城市洪水风险评级模型，能够捕捉区域之间的空间依赖关系和洪水危险与城市要素之间的复杂相互作用，揭示了城市地区的突发洪水风险概况

    

    城市洪水风险源于与洪水危险、洪水暴露以及社会和物理脆弱性相关的多个要素之间的复杂和非线性相互作用，以及复杂的空间洪水依赖关系。然而，现有的用于表征城市洪水风险的方法主要是基于洪水平原地图，侧重于有限数量的要素，主要是危险和暴露要素，没有考虑要素之间的相互作用或空间区域之间的依赖关系。为了填补这一空白，本研究提出了一种基于新颖的无监督图深度学习模型（称为FloodRisk-Net）的集成城市洪水风险评级模型。FloodRisk-Net能够捕捉区域之间的空间依赖关系以及洪水危险和城市要素之间的复杂和非线性相互作用，从而确定突发洪水风险。利用美国多个都市统计区（MSAs）的数据，该模型将它们的洪水风险特征化为

    Urban flood risk emerges from complex and nonlinear interactions among multiple features related to flood hazard, flood exposure, and social and physical vulnerabilities, along with the complex spatial flood dependence relationships. Existing approaches for characterizing urban flood risk, however, are primarily based on flood plain maps, focusing on a limited number of features, primarily hazard and exposure features, without consideration of feature interactions or the dependence relationships among spatial areas. To address this gap, this study presents an integrated urban flood-risk rating model based on a novel unsupervised graph deep learning model (called FloodRisk-Net). FloodRisk-Net is capable of capturing spatial dependence among areas and complex and nonlinear interactions among flood hazards and urban features for specifying emergent flood risk. Using data from multiple metropolitan statistical areas (MSAs) in the United States, the model characterizes their flood risk into
    
[^5]: 用调谐透镜从Transformer中获取潜在的预测能力

    Eliciting Latent Predictions from Transformers with the Tuned Lens. (arXiv:2303.08112v1 [cs.LG])

    [http://arxiv.org/abs/2303.08112](http://arxiv.org/abs/2303.08112)

    本文提出了一种改进版的“逻辑透镜”技术——“调谐透镜”，通过训练一个仿射探针，可以将每个隐藏状态解码成词汇分布。这个方法被应用于各种自回归语言模型上，比逻辑透镜更具有预测性、可靠性和无偏性，并且通过因果实验验证使用的特征与模型本身类似。同时，本文发现潜在预测的轨迹可以用于高精度地检测恶意输入。

    

    本文从迭代推理的角度分析了transformers模型，旨在了解模型预测是如何逐层进行精化的。为了实现这一目的，我们为冻结的预训练模型中的每个块训练一个仿射探针，使得可以将每个隐藏状态解码成词汇分布。我们的方法“调谐透镜”，是“逻辑透镜”技术的改进版本，前者给出了有用的见解，但常常易碎。我们将其应用于各种具有多达20B参数的自回归语言模型，表明其比逻辑透镜更具有预测性、可靠性和无偏性。通过因果实验显示，调谐透镜使用的特征与模型本身类似。我们还发现，潜在预测的轨迹可以用于高精度地检测恶意输入。我们的所有代码都可以在https://github.com/AlignmentResearch/tuned-lens 找到。

    We analyze transformers from the perspective of iterative inference, seeking to understand how model predictions are refined layer by layer. To do so, we train an affine probe for each block in a frozen pretrained model, making it possible to decode every hidden state into a distribution over the vocabulary. Our method, the \emph{tuned lens}, is a refinement of the earlier ``logit lens'' technique, which yielded useful insights but is often brittle.  We test our method on various autoregressive language models with up to 20B parameters, showing it to be more predictive, reliable and unbiased than the logit lens. With causal experiments, we show the tuned lens uses similar features to the model itself. We also find the trajectory of latent predictions can be used to detect malicious inputs with high accuracy. All code needed to reproduce our results can be found at https://github.com/AlignmentResearch/tuned-lens.
    

