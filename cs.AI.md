# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Agent Reinforcement Learning with Control-Theoretic Safety Guarantees for Dynamic Network Bridging](https://arxiv.org/abs/2404.01551) | 将多智能体强化学习与控制理论相结合，提出了一种新的设定点更新算法，以确保安全条件并实现良好的任务目标性能。 |
| [^2] | [Do LLM Agents Have Regret? A Case Study in Online Learning and Games](https://arxiv.org/abs/2403.16843) | 通过研究在线学习和博弈论中的基准决策设置，评估LLM代理的交互行为和性能，以了解它们在多代理环境中的潜力和限制。 |
| [^3] | [On Distributed Larger-Than-Memory Subset Selection With Pairwise Submodular Functions](https://arxiv.org/abs/2402.16442) | 本文提出了一种新颖的分布式约束算法，通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。 |
| [^4] | [Rethinking Optimization and Architecture for Tiny Language Models](https://arxiv.org/abs/2402.02791) | 本研究重新思考了微型语言模型的优化和架构，通过经验研究发现了在微型语言模型中特别有效的设计公式，并在多语种数据集上训练了高性能的微型语言模型。 |
| [^5] | [Mitigating Biases with Diverse Ensembles and Diffusion Models](https://arxiv.org/abs/2311.16176) | 通过利用扩散概率模型（DPMs）生成新特征组合的图像，可以在集成模型中增加模型多样性，并减轻捷径偏见，而无需额外监督信号。 |
| [^6] | [Flexible Error Mitigation of Quantum Processes with Data Augmentation Empowered Neural Model.](http://arxiv.org/abs/2311.01727) | 提出了一种数据增强强化的神经模型，该模型可以灵活地缓解量子过程中的各种噪声，并展示了在不同类型量子过程中与先前方法相比的优越性能。 |
| [^7] | [The Less Intelligent the Elements, the More Intelligent the Whole. Or, Possibly Not?.](http://arxiv.org/abs/2012.12689) | 我们探讨了个体智能是否对于集体智能的产生是必要的，以及怎样的个体智能有利于更大的集体智能。在Lotka-Volterra模型中，我们发现了一些个体行为，特别是掠食者的行为，有利于与其他种群共存，但如果猎物和掠食者都足够智能以推断彼此的行为，共存将伴随着两个种群的无限增长。 |

# 详细

[^1]: 具有控制理论安全保证的动态网络桥接的多智能体强化学习

    Multi-Agent Reinforcement Learning with Control-Theoretic Safety Guarantees for Dynamic Network Bridging

    [https://arxiv.org/abs/2404.01551](https://arxiv.org/abs/2404.01551)

    将多智能体强化学习与控制理论相结合，提出了一种新的设定点更新算法，以确保安全条件并实现良好的任务目标性能。

    

    在安全关键环境下解决复杂的合作任务对多智能体系统提出了重大挑战，尤其在部分可观测条件下。本文引入了一种混合方法，将多智能体强化学习与控制理论方法相结合，以确保安全和高效的分布式策略。我们的贡献包括一种新颖的设定点更新算法，动态调整智能体位置，以保持安全条件而不影响任务目标。通过实验验证，我们证明相比传统的多智能体强化学习策略，我们取得了显著优势，实现了与零安全违规相比可比的任务性能。研究结果表明，将安全控制与学习方法相结合不仅增强了安全合规性，还实现了良好的任务目标性能。

    arXiv:2404.01551v1 Announce Type: cross  Abstract: Addressing complex cooperative tasks in safety-critical environments poses significant challenges for Multi-Agent Systems, especially under conditions of partial observability. This work introduces a hybrid approach that integrates Multi-Agent Reinforcement Learning with control-theoretic methods to ensure safe and efficient distributed strategies. Our contributions include a novel setpoint update algorithm that dynamically adjusts agents' positions to preserve safety conditions without compromising the mission's objectives. Through experimental validation, we demonstrate significant advantages over conventional MARL strategies, achieving comparable task performance with zero safety violations. Our findings indicate that integrating safe control with learning approaches not only enhances safety compliance but also achieves good performance in mission objectives.
    
[^2]: LLM代理是否会感到后悔？在线学习和游戏案例研究

    Do LLM Agents Have Regret? A Case Study in Online Learning and Games

    [https://arxiv.org/abs/2403.16843](https://arxiv.org/abs/2403.16843)

    通过研究在线学习和博弈论中的基准决策设置，评估LLM代理的交互行为和性能，以了解它们在多代理环境中的潜力和限制。

    

    大型语言模型(LLMs)越来越多地被用于(交互式)决策制定，通过开发基于LLM的自主代理。尽管它们取得了不断的成功，但LLM代理在决策制定中的表现尚未通过定量指标进行充分调查，特别是在它们相互作用时的多代理设置中，这是实际应用中的典型场景。为了更好地理解LLM代理在这些交互环境中的限制，我们建议研究它们在在线学习和博弈论的基准决策设置中的相互作用，并通过\emph{后悔}性能指标进行评估。我们首先在经典(非平稳)在线学习问题中经验性地研究LLMs的无后悔行为，以及当LLM代理通过进行重复游戏进行交互时均衡的出现。然后我们对无后悔行为提供一些理论洞见。

    arXiv:2403.16843v1 Announce Type: cross  Abstract: Large language models (LLMs) have been increasingly employed for (interactive) decision-making, via the development of LLM-based autonomous agents. Despite their emerging successes, the performance of LLM agents in decision-making has not been fully investigated through quantitative metrics, especially in the multi-agent setting when they interact with each other, a typical scenario in real-world LLM-agent applications. To better understand the limits of LLM agents in these interactive environments, we propose to study their interactions in benchmark decision-making settings in online learning and game theory, through the performance metric of \emph{regret}. We first empirically study the {no-regret} behaviors of LLMs in canonical (non-stationary) online learning problems, as well as the emergence of equilibria when LLM agents interact through playing repeated games. We then provide some theoretical insights into the no-regret behavior
    
[^3]: 在具有配对次模模函数的分布式大于内存的子集选择问题研究

    On Distributed Larger-Than-Memory Subset Selection With Pairwise Submodular Functions

    [https://arxiv.org/abs/2402.16442](https://arxiv.org/abs/2402.16442)

    本文提出了一种新颖的分布式约束算法，通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。

    

    许多学习问题取决于子集选择的基本问题，即确定一组重要和代表性的点。本文提出了一种具有可证估计近似保证的新颖分布式约束算法，它通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。

    arXiv:2402.16442v1 Announce Type: cross  Abstract: Many learning problems hinge on the fundamental problem of subset selection, i.e., identifying a subset of important and representative points. For example, selecting the most significant samples in ML training cannot only reduce training costs but also enhance model quality. Submodularity, a discrete analogue of convexity, is commonly used for solving subset selection problems. However, existing algorithms for optimizing submodular functions are sequential, and the prior distributed methods require at least one central machine to fit the target subset. In this paper, we relax the requirement of having a central machine for the target subset by proposing a novel distributed bounding algorithm with provable approximation guarantees. The algorithm iteratively bounds the minimum and maximum utility values to select high quality points and discard the unimportant ones. When bounding does not find the complete subset, we use a multi-round, 
    
[^4]: 重新思考微型语言模型的优化和架构

    Rethinking Optimization and Architecture for Tiny Language Models

    [https://arxiv.org/abs/2402.02791](https://arxiv.org/abs/2402.02791)

    本研究重新思考了微型语言模型的优化和架构，通过经验研究发现了在微型语言模型中特别有效的设计公式，并在多语种数据集上训练了高性能的微型语言模型。

    

    大型语言模型（LLMs）的威力通过大量的数据和计算资源得到了证明。然而，在移动设备上应用语言模型面临着计算和内存成本的巨大挑战，迫切需要高性能的微型语言模型。受复杂训练过程的限制，优化语言模型的许多细节很少得到仔细研究。在本研究中，基于一个具有10亿参数的微型语言模型，我们仔细设计了一系列经验研究来分析每个组件的影响。主要讨论了三个方面，即神经架构、参数初始化和优化策略。多个设计公式在微型语言模型中经验性地被证明特别有效，包括分词器压缩、架构调整、参数继承和多轮训练。然后，我们在1.6T多语种数据集上训练了PanGu-$\pi$-1B Pro和PanGu-$\pi$-1.5B Pro。

    The power of large language models (LLMs) has been demonstrated through numerous data and computing resources. However, the application of language models on mobile devices is facing huge challenge on the computation and memory costs, that is, tiny language models with high performance are urgently required. Limited by the highly complex training process, there are many details for optimizing language models that are seldom studied carefully. In this study, based on a tiny language model with 1B parameters, we carefully design a series of empirical study to analyze the effect of each component. Three perspectives are mainly discussed, i.e., neural architecture, parameter initialization, and optimization strategy. Several design formulas are empirically proved especially effective for tiny language models, including tokenizer compression, architecture tweaking, parameter inheritance and multiple-round training. Then we train PanGu-$\pi$-1B Pro and PanGu-$\pi$-1.5B Pro on 1.6T multilingu
    
[^5]: 通过多样化合成和扩散模型减轻偏见

    Mitigating Biases with Diverse Ensembles and Diffusion Models

    [https://arxiv.org/abs/2311.16176](https://arxiv.org/abs/2311.16176)

    通过利用扩散概率模型（DPMs）生成新特征组合的图像，可以在集成模型中增加模型多样性，并减轻捷径偏见，而无需额外监督信号。

    

    数据中的虚假相关性，即多个线索可以预测目标标签，常常导致一种称为捷径偏见的现象，即模型依赖于错误的、易学的线索，而忽略可靠的线索。在这项工作中，我们提出了一种利用扩散概率模型（DPMs）的集成多样化框架，用于减轻捷径偏见。我们展示了在特定的训练间隔中，DPMs可以生成具有新特征组合的图像，即使在显示相关输入特征的样本上进行训练。我们利用这一关键属性通过集成不一致性生成合成反事实来增加模型的多样性。我们展示了DPM引导的多样化足以消除对主要捷径线索的依赖，无需额外的监督信号。我们进一步在几个多样化目标上在实证上量化其有效性，并最终展示了改进的泛化性能。

    arXiv:2311.16176v2 Announce Type: replace-cross  Abstract: Spurious correlations in the data, where multiple cues are predictive of the target labels, often lead to a phenomenon known as shortcut bias, where a model relies on erroneous, easy-to-learn cues while ignoring reliable ones. In this work, we propose an ensemble diversification framework exploiting Diffusion Probabilistic Models (DPMs) for shortcut bias mitigation. We show that at particular training intervals, DPMs can generate images with novel feature combinations, even when trained on samples displaying correlated input features. We leverage this crucial property to generate synthetic counterfactuals to increase model diversity via ensemble disagreement. We show that DPM-guided diversification is sufficient to remove dependence on primary shortcut cues, without a need for additional supervised signals. We further empirically quantify its efficacy on several diversification objectives, and finally show improved generalizati
    
[^6]: 使用数据增强强化的神经模型对量子过程进行灵活的误差缓解

    Flexible Error Mitigation of Quantum Processes with Data Augmentation Empowered Neural Model. (arXiv:2311.01727v1 [quant-ph])

    [http://arxiv.org/abs/2311.01727](http://arxiv.org/abs/2311.01727)

    提出了一种数据增强强化的神经模型，该模型可以灵活地缓解量子过程中的各种噪声，并展示了在不同类型量子过程中与先前方法相比的优越性能。

    

    神经网络在量子计算的各种任务中显示出了其有效性。然而，在量子误差缓解中的应用受到对无噪声统计的依赖限制，这是实现实际量子进展的关键步骤。为了解决这一关键挑战，我们提出了一种数据增强强化的神经模型用于误差缓解（DAEM）。我们的模型不需要任何关于特定噪声类型和测量设置的先验知识，并且可以仅根据目标量子过程的噪声测量结果估计无噪声统计值，使其非常适合实际实施。在数值实验中，我们展示了该模型在缓解各种类型的噪声（包括马尔可夫噪声和非马尔可夫噪声）方面与先前的误差缓解方法相比的优越性能。我们进一步通过利用该模型来缓解多种类型的量子过程中的错误来展示其多功能性。

    Neural networks have shown their effectiveness in various tasks in the realm of quantum computing. However, their application in quantum error mitigation, a crucial step towards realizing practical quantum advancements, has been restricted by reliance on noise-free statistics. To tackle this critical challenge, we propose a data augmentation empowered neural model for error mitigation (DAEM). Our model does not require any prior knowledge about the specific noise type and measurement settings and can estimate noise-free statistics solely from the noisy measurement results of the target quantum process, rendering it highly suitable for practical implementation. In numerical experiments, we show the model's superior performance in mitigating various types of noise, including Markovian noise and Non-Markovian noise, compared with previous error mitigation methods. We further demonstrate its versatility by employing the model to mitigate errors in diverse types of quantum processes, includ
    
[^7]: 元素越笨，整体越聪明。或者，可能并非如此？

    The Less Intelligent the Elements, the More Intelligent the Whole. Or, Possibly Not?. (arXiv:2012.12689v2 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2012.12689](http://arxiv.org/abs/2012.12689)

    我们探讨了个体智能是否对于集体智能的产生是必要的，以及怎样的个体智能有利于更大的集体智能。在Lotka-Volterra模型中，我们发现了一些个体行为，特别是掠食者的行为，有利于与其他种群共存，但如果猎物和掠食者都足够智能以推断彼此的行为，共存将伴随着两个种群的无限增长。

    

    我们探讨了大脑中的神经元与社会中的人类之间的利维坦类比，问自己是否个体智能对于集体智能的产生是必要的，更重要的是，怎样的个体智能有利于更大的集体智能。首先，我们回顾了连接主义认知科学、基于代理的建模、群体心理学、经济学和物理学的不同洞见。随后，我们将这些洞见应用于Lotka-Volterra模型中导致掠食者和猎物要么共存要么全球灭绝的智能类型和程度。我们发现几个个体行为 - 尤其是掠食者的行为 - 有利于共存，最终在一个平衡点周围产生震荡。然而，我们也发现，如果猎物和掠食者都足够智能以推断彼此的行为，共存就会伴随着两个种群的无限增长。由于Lotka-Volterra模型是不稳定的，我们提出了一些未来的研究方向来解决这个问题。

    We explore a Leviathan analogy between neurons in a brain and human beings in society, asking ourselves whether individual intelligence is necessary for collective intelligence to emerge and, most importantly, what sort of individual intelligence is conducive of greater collective intelligence. We first review disparate insights from connectionist cognitive science, agent-based modeling, group psychology, economics and physics. Subsequently, we apply these insights to the sort and degrees of intelligence that in the Lotka-Volterra model lead to either co-existence or global extinction of predators and preys.  We find several individual behaviors -- particularly of predators -- that are conducive to co-existence, eventually with oscillations around an equilibrium. However, we also find that if both preys and predators are sufficiently intelligent to extrapolate one other's behavior, co-existence comes along with indefinite growth of both populations. Since the Lotka-Volterra model is al
    

