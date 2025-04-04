# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [End-To-End Self-tuning Self-supervised Time Series Anomaly Detection](https://arxiv.org/abs/2404.02865) | 提出了TSAP方法来自动调整数据增强，为时间序列异常检测带来了端到端的自调节能力。 |
| [^2] | [Multi-Agent Reinforcement Learning with Control-Theoretic Safety Guarantees for Dynamic Network Bridging](https://arxiv.org/abs/2404.01551) | 将多智能体强化学习与控制理论相结合，提出了一种新的设定点更新算法，以确保安全条件并实现良好的任务目标性能。 |
| [^3] | [Do LLM Agents Have Regret? A Case Study in Online Learning and Games](https://arxiv.org/abs/2403.16843) | 通过研究在线学习和博弈论中的基准决策设置，评估LLM代理的交互行为和性能，以了解它们在多代理环境中的潜力和限制。 |
| [^4] | [IBCB: Efficient Inverse Batched Contextual Bandit for Behavioral Evolution History](https://arxiv.org/abs/2403.16075) | 提出了一种逆批处理上下文强化学习（IBCB）框架，可以高效地根据专家的行为演变历史对环境奖励参数和学习策略进行估计。 |
| [^5] | [A time-stepping deep gradient flow method for option pricing in (rough) diffusion models](https://arxiv.org/abs/2403.00746) | 提出了一种时间步进深度梯度流方法，用于处理（粗糙）扩散模型中的期权定价问题，保证了对大金额水平下期权价格的渐近行为和先验上下界。 |
| [^6] | [On Distributed Larger-Than-Memory Subset Selection With Pairwise Submodular Functions](https://arxiv.org/abs/2402.16442) | 本文提出了一种新颖的分布式约束算法，通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。 |
| [^7] | [Rethinking Optimization and Architecture for Tiny Language Models](https://arxiv.org/abs/2402.02791) | 本研究重新思考了微型语言模型的优化和架构，通过经验研究发现了在微型语言模型中特别有效的设计公式，并在多语种数据集上训练了高性能的微型语言模型。 |
| [^8] | [Mitigating Biases with Diverse Ensembles and Diffusion Models](https://arxiv.org/abs/2311.16176) | 通过利用扩散概率模型（DPMs）生成新特征组合的图像，可以在集成模型中增加模型多样性，并减轻捷径偏见，而无需额外监督信号。 |
| [^9] | [Prevalidated ridge regression is a highly-efficient drop-in replacement for logistic regression for high-dimensional data.](http://arxiv.org/abs/2401.15610) | 本论文提出了一种预验证的岭回归模型，该模型在高维数据中与逻辑回归非常接近，但具有更高的计算效率和几乎没有超参数。它通过利用在拟合过程中计算得到的数量来缩放模型系数，并最小化一组预验证预测的对数损失。 |
| [^10] | [Flexible Error Mitigation of Quantum Processes with Data Augmentation Empowered Neural Model.](http://arxiv.org/abs/2311.01727) | 提出了一种数据增强强化的神经模型，该模型可以灵活地缓解量子过程中的各种噪声，并展示了在不同类型量子过程中与先前方法相比的优越性能。 |
| [^11] | [LEACE: Perfect linear concept erasure in closed form.](http://arxiv.org/abs/2306.03819) | 本文介绍了一种闭合形式的方法LEACE，可在删除指定特征的同时尽可能少地改变表示，并可证明防止所有线性分类器检测到概念。作者用“概念擦除”这一新方法将其应用于大型语言模型，在测量语言模型对词性的依赖性和减少BERT嵌入中的性别偏差任务中得出良好表现。 |
| [^12] | [Utility Theory of Synthetic Data Generation.](http://arxiv.org/abs/2305.10015) | 本文从统计学角度建立效用理论，旨在基于一般性指标定量评估合成算法的效用，效用指标的分析界限揭示了指标收敛的关键条件，令人惊讶的是，只要下游学习任务中的模型规范是正确的，合成特征分布不一定与原始特征分布相同，效用指标会收敛。 |
| [^13] | [Convex optimization over a probability simplex.](http://arxiv.org/abs/2305.09046) | 这篇论文提出了一种新的迭代方案，用于求解概率单纯形上的凸优化问题。该方法具有收敛速度快且简单易行的特点。 |
| [^14] | [Efficient Training of Multi-task Neural Solver with Multi-armed Bandits.](http://arxiv.org/abs/2305.06361) | 本文提出了一种基于多臂赌博机的通用高效训练范式，用于多任务神经求解器的训练，通过任务影响矩阵进行更高效的训练，相比于标准计划，在有限的训练预算或相同的训练时长内实现了更高的整体性能。 |
| [^15] | [Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series.](http://arxiv.org/abs/2304.03069) | 本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。 |

# 详细

[^1]: 端到端自调节自监督时间序列异常检测

    End-To-End Self-tuning Self-supervised Time Series Anomaly Detection

    [https://arxiv.org/abs/2404.02865](https://arxiv.org/abs/2404.02865)

    提出了TSAP方法来自动调整数据增强，为时间序列异常检测带来了端到端的自调节能力。

    

    时间序列异常检测（TSAD）在监控环境传感器、行业KPI、患者生物标志物等方面有许多应用。TSAD的一个双重挑战是需要一种多功能且无监督模型，能够检测各种不同类型的时间序列异常（尖峰、不连续、趋势变化等），而不需要任何标记的数据。我们的工作旨在填补这一空白。我们引入了TSAP来执行TSA“自动驾驶”，可以端到端自动调整数据增强的超参数，自适应选择数据增强策略。

    arXiv:2404.02865v1 Announce Type: new  Abstract: Time series anomaly detection (TSAD) finds many applications such as monitoring environmental sensors, industry KPIs, patient biomarkers, etc. A two-fold challenge for TSAD is a versatile and unsupervised model that can detect various different types of time series anomalies (spikes, discontinuities, trend shifts, etc.) without any labeled data. Modern neural networks have outstanding ability in modeling complex time series. Self-supervised models in particular tackle unsupervised TSAD by transforming the input via various augmentations to create pseudo anomalies for training. However, their performance is sensitive to the choice of augmentation, which is hard to choose in practice, while there exists no effort in the literature on data augmentation tuning for TSAD without labels. Our work aims to fill this gap. We introduce TSAP for TSA "on autoPilot", which can (self-)tune augmentation hyperparameters end-to-end. It stands on two key c
    
[^2]: 具有控制理论安全保证的动态网络桥接的多智能体强化学习

    Multi-Agent Reinforcement Learning with Control-Theoretic Safety Guarantees for Dynamic Network Bridging

    [https://arxiv.org/abs/2404.01551](https://arxiv.org/abs/2404.01551)

    将多智能体强化学习与控制理论相结合，提出了一种新的设定点更新算法，以确保安全条件并实现良好的任务目标性能。

    

    在安全关键环境下解决复杂的合作任务对多智能体系统提出了重大挑战，尤其在部分可观测条件下。本文引入了一种混合方法，将多智能体强化学习与控制理论方法相结合，以确保安全和高效的分布式策略。我们的贡献包括一种新颖的设定点更新算法，动态调整智能体位置，以保持安全条件而不影响任务目标。通过实验验证，我们证明相比传统的多智能体强化学习策略，我们取得了显著优势，实现了与零安全违规相比可比的任务性能。研究结果表明，将安全控制与学习方法相结合不仅增强了安全合规性，还实现了良好的任务目标性能。

    arXiv:2404.01551v1 Announce Type: cross  Abstract: Addressing complex cooperative tasks in safety-critical environments poses significant challenges for Multi-Agent Systems, especially under conditions of partial observability. This work introduces a hybrid approach that integrates Multi-Agent Reinforcement Learning with control-theoretic methods to ensure safe and efficient distributed strategies. Our contributions include a novel setpoint update algorithm that dynamically adjusts agents' positions to preserve safety conditions without compromising the mission's objectives. Through experimental validation, we demonstrate significant advantages over conventional MARL strategies, achieving comparable task performance with zero safety violations. Our findings indicate that integrating safe control with learning approaches not only enhances safety compliance but also achieves good performance in mission objectives.
    
[^3]: LLM代理是否会感到后悔？在线学习和游戏案例研究

    Do LLM Agents Have Regret? A Case Study in Online Learning and Games

    [https://arxiv.org/abs/2403.16843](https://arxiv.org/abs/2403.16843)

    通过研究在线学习和博弈论中的基准决策设置，评估LLM代理的交互行为和性能，以了解它们在多代理环境中的潜力和限制。

    

    大型语言模型(LLMs)越来越多地被用于(交互式)决策制定，通过开发基于LLM的自主代理。尽管它们取得了不断的成功，但LLM代理在决策制定中的表现尚未通过定量指标进行充分调查，特别是在它们相互作用时的多代理设置中，这是实际应用中的典型场景。为了更好地理解LLM代理在这些交互环境中的限制，我们建议研究它们在在线学习和博弈论的基准决策设置中的相互作用，并通过\emph{后悔}性能指标进行评估。我们首先在经典(非平稳)在线学习问题中经验性地研究LLMs的无后悔行为，以及当LLM代理通过进行重复游戏进行交互时均衡的出现。然后我们对无后悔行为提供一些理论洞见。

    arXiv:2403.16843v1 Announce Type: cross  Abstract: Large language models (LLMs) have been increasingly employed for (interactive) decision-making, via the development of LLM-based autonomous agents. Despite their emerging successes, the performance of LLM agents in decision-making has not been fully investigated through quantitative metrics, especially in the multi-agent setting when they interact with each other, a typical scenario in real-world LLM-agent applications. To better understand the limits of LLM agents in these interactive environments, we propose to study their interactions in benchmark decision-making settings in online learning and game theory, through the performance metric of \emph{regret}. We first empirically study the {no-regret} behaviors of LLMs in canonical (non-stationary) online learning problems, as well as the emergence of equilibria when LLM agents interact through playing repeated games. We then provide some theoretical insights into the no-regret behavior
    
[^4]: IBCB: 高效的逆批处理上下文强化学习用于行为演变历史

    IBCB: Efficient Inverse Batched Contextual Bandit for Behavioral Evolution History

    [https://arxiv.org/abs/2403.16075](https://arxiv.org/abs/2403.16075)

    提出了一种逆批处理上下文强化学习（IBCB）框架，可以高效地根据专家的行为演变历史对环境奖励参数和学习策略进行估计。

    

    传统的模仿学习关注专家的行为机制建模，需要大量由某个固定专家生成的交互历史。然而，在许多流式应用中，如流式推荐系统，在线决策者通常在决策过程中进行在线学习，这意味着在线决策者生成的交互历史包括他们从新手专家到有经验专家的行为演变。这给现有的只能利用有经验专家数据的模仿学习方法带来了新挑战。为了解决这个问题，本文提出了一种逆批处理上下文强化学习（IBCB）框架，能够高效地进行基于专家行为演变历史的环境奖励参数和学习策略的估计。具体来说，IBCB将逆问题形式化为简单的二次规划。

    arXiv:2403.16075v1 Announce Type: new  Abstract: Traditional imitation learning focuses on modeling the behavioral mechanisms of experts, which requires a large amount of interaction history generated by some fixed expert. However, in many streaming applications, such as streaming recommender systems, online decision-makers typically engage in online learning during the decision-making process, meaning that the interaction history generated by online decision-makers includes their behavioral evolution from novice expert to experienced expert. This poses a new challenge for existing imitation learning approaches that can only utilize data from experienced experts. To address this issue, this paper proposes an inverse batched contextual bandit (IBCB) framework that can efficiently perform estimations of environment reward parameters and learned policy based on the expert's behavioral evolution history. Specifically, IBCB formulates the inverse problem into a simple quadratic programming 
    
[^5]: 一种针对（粗糙）扩散模型中期权定价的时间步进深度梯度流方法

    A time-stepping deep gradient flow method for option pricing in (rough) diffusion models

    [https://arxiv.org/abs/2403.00746](https://arxiv.org/abs/2403.00746)

    提出了一种时间步进深度梯度流方法，用于处理（粗糙）扩散模型中的期权定价问题，保证了对大金额水平下期权价格的渐近行为和先验上下界。

    

    我们开发了一种新颖的深度学习方法，用于在扩散模型中定价欧式期权，可以高效处理由于粗糙波动率模型的马尔可夫逼近而导致的高维问题。期权定价的偏微分方程被重新表述为能量最小化问题，该问题通过深度人工神经网络以时间步进的方式进行近似。所提出的方案符合期权价格在大金额水平上的渐近行为，并遵守期权价格的先验已知上下界。通过一系列数值示例评估了所提方法的准确性和效率，特别关注了提升Heston模型。

    arXiv:2403.00746v1 Announce Type: cross  Abstract: We develop a novel deep learning approach for pricing European options in diffusion models, that can efficiently handle high-dimensional problems resulting from Markovian approximations of rough volatility models. The option pricing partial differential equation is reformulated as an energy minimization problem, which is approximated in a time-stepping fashion by deep artificial neural networks. The proposed scheme respects the asymptotic behavior of option prices for large levels of moneyness, and adheres to a priori known bounds for option prices. The accuracy and efficiency of the proposed method is assessed in a series of numerical examples, with particular focus in the lifted Heston model.
    
[^6]: 在具有配对次模模函数的分布式大于内存的子集选择问题研究

    On Distributed Larger-Than-Memory Subset Selection With Pairwise Submodular Functions

    [https://arxiv.org/abs/2402.16442](https://arxiv.org/abs/2402.16442)

    本文提出了一种新颖的分布式约束算法，通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。

    

    许多学习问题取决于子集选择的基本问题，即确定一组重要和代表性的点。本文提出了一种具有可证估计近似保证的新颖分布式约束算法，它通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。

    arXiv:2402.16442v1 Announce Type: cross  Abstract: Many learning problems hinge on the fundamental problem of subset selection, i.e., identifying a subset of important and representative points. For example, selecting the most significant samples in ML training cannot only reduce training costs but also enhance model quality. Submodularity, a discrete analogue of convexity, is commonly used for solving subset selection problems. However, existing algorithms for optimizing submodular functions are sequential, and the prior distributed methods require at least one central machine to fit the target subset. In this paper, we relax the requirement of having a central machine for the target subset by proposing a novel distributed bounding algorithm with provable approximation guarantees. The algorithm iteratively bounds the minimum and maximum utility values to select high quality points and discard the unimportant ones. When bounding does not find the complete subset, we use a multi-round, 
    
[^7]: 重新思考微型语言模型的优化和架构

    Rethinking Optimization and Architecture for Tiny Language Models

    [https://arxiv.org/abs/2402.02791](https://arxiv.org/abs/2402.02791)

    本研究重新思考了微型语言模型的优化和架构，通过经验研究发现了在微型语言模型中特别有效的设计公式，并在多语种数据集上训练了高性能的微型语言模型。

    

    大型语言模型（LLMs）的威力通过大量的数据和计算资源得到了证明。然而，在移动设备上应用语言模型面临着计算和内存成本的巨大挑战，迫切需要高性能的微型语言模型。受复杂训练过程的限制，优化语言模型的许多细节很少得到仔细研究。在本研究中，基于一个具有10亿参数的微型语言模型，我们仔细设计了一系列经验研究来分析每个组件的影响。主要讨论了三个方面，即神经架构、参数初始化和优化策略。多个设计公式在微型语言模型中经验性地被证明特别有效，包括分词器压缩、架构调整、参数继承和多轮训练。然后，我们在1.6T多语种数据集上训练了PanGu-$\pi$-1B Pro和PanGu-$\pi$-1.5B Pro。

    The power of large language models (LLMs) has been demonstrated through numerous data and computing resources. However, the application of language models on mobile devices is facing huge challenge on the computation and memory costs, that is, tiny language models with high performance are urgently required. Limited by the highly complex training process, there are many details for optimizing language models that are seldom studied carefully. In this study, based on a tiny language model with 1B parameters, we carefully design a series of empirical study to analyze the effect of each component. Three perspectives are mainly discussed, i.e., neural architecture, parameter initialization, and optimization strategy. Several design formulas are empirically proved especially effective for tiny language models, including tokenizer compression, architecture tweaking, parameter inheritance and multiple-round training. Then we train PanGu-$\pi$-1B Pro and PanGu-$\pi$-1.5B Pro on 1.6T multilingu
    
[^8]: 通过多样化合成和扩散模型减轻偏见

    Mitigating Biases with Diverse Ensembles and Diffusion Models

    [https://arxiv.org/abs/2311.16176](https://arxiv.org/abs/2311.16176)

    通过利用扩散概率模型（DPMs）生成新特征组合的图像，可以在集成模型中增加模型多样性，并减轻捷径偏见，而无需额外监督信号。

    

    数据中的虚假相关性，即多个线索可以预测目标标签，常常导致一种称为捷径偏见的现象，即模型依赖于错误的、易学的线索，而忽略可靠的线索。在这项工作中，我们提出了一种利用扩散概率模型（DPMs）的集成多样化框架，用于减轻捷径偏见。我们展示了在特定的训练间隔中，DPMs可以生成具有新特征组合的图像，即使在显示相关输入特征的样本上进行训练。我们利用这一关键属性通过集成不一致性生成合成反事实来增加模型的多样性。我们展示了DPM引导的多样化足以消除对主要捷径线索的依赖，无需额外的监督信号。我们进一步在几个多样化目标上在实证上量化其有效性，并最终展示了改进的泛化性能。

    arXiv:2311.16176v2 Announce Type: replace-cross  Abstract: Spurious correlations in the data, where multiple cues are predictive of the target labels, often lead to a phenomenon known as shortcut bias, where a model relies on erroneous, easy-to-learn cues while ignoring reliable ones. In this work, we propose an ensemble diversification framework exploiting Diffusion Probabilistic Models (DPMs) for shortcut bias mitigation. We show that at particular training intervals, DPMs can generate images with novel feature combinations, even when trained on samples displaying correlated input features. We leverage this crucial property to generate synthetic counterfactuals to increase model diversity via ensemble disagreement. We show that DPM-guided diversification is sufficient to remove dependence on primary shortcut cues, without a need for additional supervised signals. We further empirically quantify its efficacy on several diversification objectives, and finally show improved generalizati
    
[^9]: 预验证的岭回归是高维数据中逻辑回归的高效替代方法

    Prevalidated ridge regression is a highly-efficient drop-in replacement for logistic regression for high-dimensional data. (arXiv:2401.15610v1 [cs.LG])

    [http://arxiv.org/abs/2401.15610](http://arxiv.org/abs/2401.15610)

    本论文提出了一种预验证的岭回归模型，该模型在高维数据中与逻辑回归非常接近，但具有更高的计算效率和几乎没有超参数。它通过利用在拟合过程中计算得到的数量来缩放模型系数，并最小化一组预验证预测的对数损失。

    

    逻辑回归是一种常见的概率分类方法。然而，逻辑回归的有效性取决于仔细且相对计算密集的调优，尤其是对于正则化超参数，并且尤其在高维数据的背景下。我们提出了一种预验证的岭回归模型，该模型在分类错误和对数损失方面与逻辑回归非常接近，特别适用于高维数据，同时在计算效率上明显更高，并且除了正则化之外没有超参数。我们通过缩放模型的系数来最小化由估计的留一交叉验证误差推导出的一组预验证预测的对数损失。这利用了在拟合岭回归模型过程中已经计算的数量，以找到具有名义附加计算开销的缩放参数。

    Logistic regression is a ubiquitous method for probabilistic classification. However, the effectiveness of logistic regression depends upon careful and relatively computationally expensive tuning, especially for the regularisation hyperparameter, and especially in the context of high-dimensional data. We present a prevalidated ridge regression model that closely matches logistic regression in terms of classification error and log-loss, particularly for high-dimensional data, while being significantly more computationally efficient and having effectively no hyperparameters beyond regularisation. We scale the coefficients of the model so as to minimise log-loss for a set of prevalidated predictions derived from the estimated leave-one-out cross-validation error. This exploits quantities already computed in the course of fitting the ridge regression model in order to find the scaling parameter with nominal additional computational expense.
    
[^10]: 使用数据增强强化的神经模型对量子过程进行灵活的误差缓解

    Flexible Error Mitigation of Quantum Processes with Data Augmentation Empowered Neural Model. (arXiv:2311.01727v1 [quant-ph])

    [http://arxiv.org/abs/2311.01727](http://arxiv.org/abs/2311.01727)

    提出了一种数据增强强化的神经模型，该模型可以灵活地缓解量子过程中的各种噪声，并展示了在不同类型量子过程中与先前方法相比的优越性能。

    

    神经网络在量子计算的各种任务中显示出了其有效性。然而，在量子误差缓解中的应用受到对无噪声统计的依赖限制，这是实现实际量子进展的关键步骤。为了解决这一关键挑战，我们提出了一种数据增强强化的神经模型用于误差缓解（DAEM）。我们的模型不需要任何关于特定噪声类型和测量设置的先验知识，并且可以仅根据目标量子过程的噪声测量结果估计无噪声统计值，使其非常适合实际实施。在数值实验中，我们展示了该模型在缓解各种类型的噪声（包括马尔可夫噪声和非马尔可夫噪声）方面与先前的误差缓解方法相比的优越性能。我们进一步通过利用该模型来缓解多种类型的量子过程中的错误来展示其多功能性。

    Neural networks have shown their effectiveness in various tasks in the realm of quantum computing. However, their application in quantum error mitigation, a crucial step towards realizing practical quantum advancements, has been restricted by reliance on noise-free statistics. To tackle this critical challenge, we propose a data augmentation empowered neural model for error mitigation (DAEM). Our model does not require any prior knowledge about the specific noise type and measurement settings and can estimate noise-free statistics solely from the noisy measurement results of the target quantum process, rendering it highly suitable for practical implementation. In numerical experiments, we show the model's superior performance in mitigating various types of noise, including Markovian noise and Non-Markovian noise, compared with previous error mitigation methods. We further demonstrate its versatility by employing the model to mitigate errors in diverse types of quantum processes, includ
    
[^11]: LEACE：闭合形式中的完美线性概念擦除

    LEACE: Perfect linear concept erasure in closed form. (arXiv:2306.03819v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.03819](http://arxiv.org/abs/2306.03819)

    本文介绍了一种闭合形式的方法LEACE，可在删除指定特征的同时尽可能少地改变表示，并可证明防止所有线性分类器检测到概念。作者用“概念擦除”这一新方法将其应用于大型语言模型，在测量语言模型对词性的依赖性和减少BERT嵌入中的性别偏差任务中得出良好表现。

    

    概念擦除旨在从表征中删除指定的特征。它可以提高公平性（例如，防止分类器使用性别或种族）和可解释性（例如，删除概念以观察模型行为的变化）。我们引入了LEAst-squares概念擦除（LEACE），这是一种闭合形式的方法，可证明防止所有线性分类器检测到概念，同时尽可能地改变表示，如广泛类别的范数所测量的那样。我们使用名为“概念擦除”的新方法将LEACE应用于大型语言模型，擦除每个层中的目标概念信息。我们在两个任务上展示了我们的方法：测量语言模型对词性信息的依赖性，以及减少BERT嵌入中的性别偏差。代码可在https://github.com/EleutherAI/concept-erasure上找到。

    Concept erasure aims to remove specified features from a representation. It can improve fairness (e.g. preventing a classifier from using gender or race) and interpretability (e.g. removing a concept to observe changes in model behavior). We introduce LEAst-squares Concept Erasure (LEACE), a closed-form method which provably prevents all linear classifiers from detecting a concept while changing the representation as little as possible, as measured by a broad class of norms. We apply LEACE to large language models with a novel procedure called "concept scrubbing," which erases target concept information from every layer in the network. We demonstrate our method on two tasks: measuring the reliance of language models on part-of-speech information, and reducing gender bias in BERT embeddings. Code is available at https://github.com/EleutherAI/concept-erasure.
    
[^12]: 合成数据生成的效用理论

    Utility Theory of Synthetic Data Generation. (arXiv:2305.10015v1 [stat.ML])

    [http://arxiv.org/abs/2305.10015](http://arxiv.org/abs/2305.10015)

    本文从统计学角度建立效用理论，旨在基于一般性指标定量评估合成算法的效用，效用指标的分析界限揭示了指标收敛的关键条件，令人惊讶的是，只要下游学习任务中的模型规范是正确的，合成特征分布不一定与原始特征分布相同，效用指标会收敛。

    

    评估合成数据的效用对于衡量合成算法的有效性和效率至关重要。现有的结果侧重于对合成数据效用的经验评估，而针对合成数据算法如何影响效用的理论理解仍然未被充分探索。本文从统计学角度建立效用理论，旨在基于一般性指标定量评估合成算法的效用。该指标定义为在合成和原始数据集上训练的模型之间泛化的绝对差异。我们建立了该效用指标的分析界限来研究指标收敛的关键条件。一个有趣的结果是，只要下游学习任务中的模型规范是正确的，合成特征分布不一定与原始特征分布相同，则该效用指标会收敛。另一个重要的效用指标基于合成和原始数据之间潜在的因果机制一致性。该理论使用几种合成算法进行说明，并分析了它们的效用属性。

    Evaluating the utility of synthetic data is critical for measuring the effectiveness and efficiency of synthetic algorithms. Existing results focus on empirical evaluations of the utility of synthetic data, whereas the theoretical understanding of how utility is affected by synthetic data algorithms remains largely unexplored. This paper establishes utility theory from a statistical perspective, aiming to quantitatively assess the utility of synthetic algorithms based on a general metric. The metric is defined as the absolute difference in generalization between models trained on synthetic and original datasets. We establish analytical bounds for this utility metric to investigate critical conditions for the metric to converge. An intriguing result is that the synthetic feature distribution is not necessarily identical to the original one for the convergence of the utility metric as long as the model specification in downstream learning tasks is correct. Another important utility metri
    
[^13]: 概率单纯形上的凸优化

    Convex optimization over a probability simplex. (arXiv:2305.09046v1 [math.OC])

    [http://arxiv.org/abs/2305.09046](http://arxiv.org/abs/2305.09046)

    这篇论文提出了一种新的迭代方案，用于求解概率单纯形上的凸优化问题。该方法具有收敛速度快且简单易行的特点。

    

    我们提出了一种新的迭代方案——柯西单纯形来优化凸问题，使其满足概率单纯形上的限制条件，即$w\in\mathbb{R}^n$中$\sum_i w_i=1$，$w_i\geq0$。我们将单纯形映射到单位球的正四面体，通过梯度下降获得隐变量的解，并将结果映射回原始变量。该方法适用于高维问题，每次迭代由简单的操作组成，且针对凸函数证明了收敛速度为${O}(1/T)$。同时本文关注了信息理论（如交叉熵和KL散度）的应用。

    We propose a new iteration scheme, the Cauchy-Simplex, to optimize convex problems over the probability simplex $\{w\in\mathbb{R}^n\ |\ \sum_i w_i=1\ \textrm{and}\ w_i\geq0\}$. Other works have taken steps to enforce positivity or unit normalization automatically but never simultaneously within a unified setting. This paper presents a natural framework for manifestly requiring the probability condition. Specifically, we map the simplex to the positive quadrant of a unit sphere, envisage gradient descent in latent variables, and map the result back in a way that only depends on the simplex variable. Moreover, proving rigorous convergence results in this formulation leads inherently to tools from information theory (e.g. cross entropy and KL divergence). Each iteration of the Cauchy-Simplex consists of simple operations, making it well-suited for high-dimensional problems. We prove that it has a convergence rate of ${O}(1/T)$ for convex functions, and numerical experiments of projection 
    
[^14]: 多臂赌博机用于多任务神经求解器的高效训练

    Efficient Training of Multi-task Neural Solver with Multi-armed Bandits. (arXiv:2305.06361v1 [cs.LG])

    [http://arxiv.org/abs/2305.06361](http://arxiv.org/abs/2305.06361)

    本文提出了一种基于多臂赌博机的通用高效训练范式，用于多任务神经求解器的训练，通过任务影响矩阵进行更高效的训练，相比于标准计划，在有限的训练预算或相同的训练时长内实现了更高的整体性能。

    

    针对如何高效地为各种组合优化问题 (COP) 训练多任务神经求解器，目前的研究相对较少。在本文中，我们提出了一种基于多臂赌博机的通用高效训练范式，以提供一个统一的多任务神经求解器。为此，我们利用编码器-解码器框架下的多任务理论损失分解，通过一个任务影响矩阵通过正确的赌博算法实现更高效的训练。相比标准的训练计划，我们的方法在有限的训练预算或相同的训练时段内实现了更高的整体性能，这可以为其他多任务大模型的高效训练提供指导，此外，影响矩阵可以提供学习优化领域中常见实践的经验证据，从而支持我们方法的可行性。

    Efficiently training a multi-task neural solver for various combinatorial optimization problems (COPs) has been less studied so far. In this paper, we propose a general and efficient training paradigm based on multi-armed bandits to deliver a unified multi-task neural solver. To this end, we resort to the theoretical loss decomposition for multiple tasks under an encoder-decoder framework, which enables more efficient training via proper bandit task-sampling algorithms through an intra-task influence matrix. Our method achieves much higher overall performance with either limited training budgets or the same training epochs, compared to standard training schedules, which can be promising for advising efficient training of other multi-task large models. Additionally, the influence matrix can provide empirical evidence of some common practices in the area of learning to optimize, which in turn supports the validity of our approach.
    
[^15]: 自适应学生t分布与方法矩移动估计器用于非平稳时间序列

    Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series. (arXiv:2304.03069v1 [stat.ME])

    [http://arxiv.org/abs/2304.03069](http://arxiv.org/abs/2304.03069)

    本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。

    

    真实的时间序列通常是非平稳的，这带来了模型适应的难题。传统方法如GARCH假定任意类型的依赖性。为了避免这种偏差，我们将着眼于最近提出的不可知的移动估计器哲学：在时间$t$找到优化$F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$移动对数似然的参数，随时间演化。例如，它允许使用廉价的指数移动平均值（EMA）来估计参数，例如绝对中心矩$E[|x-\mu|^p]$随$p\in\mathbb{R}^+$的变化而演化$m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$。这种基于方法的一般自适应矩的应用将呈现在学生t分布上，尤其是在经济应用中流行，这里应用于DJIA公司的对数收益率。

    The real life time series are usually nonstationary, bringing a difficult question of model adaptation. Classical approaches like GARCH assume arbitrary type of dependence. To prevent such bias, we will focus on recently proposed agnostic philosophy of moving estimator: in time $t$ finding parameters optimizing e.g. $F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$ moving log-likelihood, evolving in time. It allows for example to estimate parameters using inexpensive exponential moving averages (EMA), like absolute central moments $E[|x-\mu|^p]$ evolving with $m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$ for one or multiple powers $p\in\mathbb{R}^+$. Application of such general adaptive methods of moments will be presented on Student's t-distribution, popular especially in economical applications, here applied to log-returns of DJIA companies.
    

