# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tactical Decision Making for Autonomous Trucks by Deep Reinforcement Learning with Total Cost of Operation Based Reward](https://arxiv.org/abs/2403.06524) | 通过基于全面运营成本的奖励函数，采用深度强化学习框架优化自主卡车的战术决策，将高级决策与低级控制分离，并采用不同技巧提升性能。 |
| [^2] | [On the Convergence of Differentially-Private Fine-tuning: To Linearly Probe or to Fully Fine-tune?](https://arxiv.org/abs/2402.18905) | 本文分析了差分隐私线性探测（LP）和完全微调（FT）的训练动态，探索了从线性探测过渡到完全微调（LP-FT）的顺序微调现象及其对测试损失的影响，提供了关于在超参数化神经网络中差分隐私微调收敛性的理论洞见和隐私预算分配的效用曲线。 |
| [^3] | [The Less Intelligent the Elements, the More Intelligent the Whole. Or, Possibly Not?.](http://arxiv.org/abs/2012.12689) | 我们探讨了个体智能是否对于集体智能的产生是必要的，以及怎样的个体智能有利于更大的集体智能。在Lotka-Volterra模型中，我们发现了一些个体行为，特别是掠食者的行为，有利于与其他种群共存，但如果猎物和掠食者都足够智能以推断彼此的行为，共存将伴随着两个种群的无限增长。 |

# 详细

[^1]: 基于全面运营成本奖励的深度强化学习自主卡车战术决策

    Tactical Decision Making for Autonomous Trucks by Deep Reinforcement Learning with Total Cost of Operation Based Reward

    [https://arxiv.org/abs/2403.06524](https://arxiv.org/abs/2403.06524)

    通过基于全面运营成本的奖励函数，采用深度强化学习框架优化自主卡车的战术决策，将高级决策与低级控制分离，并采用不同技巧提升性能。

    

    我们为自主卡车的战术决策制定了一个基于深度强化学习的框架，特别是针对高速公路场景中的自适应巡航控制（ACC）和变道动作。我们的研究结果表明，在强化学习代理和基于物理模型的低级控制器之间分离高级决策过程和低级控制动作是有益的。接下来，我们研究了通过在卡车的全面运营成本（TCOP）为基础的多目标奖励函数优化性能的不同方法；通过为奖励分量添加权重，通过对奖励分量进行归一化，以及使用课程学习技术。

    arXiv:2403.06524v1 Announce Type: cross  Abstract: We develop a deep reinforcement learning framework for tactical decision making in an autonomous truck, specifically for Adaptive Cruise Control (ACC) and lane change maneuvers in a highway scenario. Our results demonstrate that it is beneficial to separate high-level decision-making processes and low-level control actions between the reinforcement learning agent and the low-level controllers based on physical models. In the following, we study optimizing the performance with a realistic and multi-objective reward function based on Total Cost of Operation (TCOP) of the truck using different approaches; by adding weights to reward components, by normalizing the reward components and by using curriculum learning techniques.
    
[^2]: 论差分隐私微调的收敛性：应线性探测还是完全微调？

    On the Convergence of Differentially-Private Fine-tuning: To Linearly Probe or to Fully Fine-tune?

    [https://arxiv.org/abs/2402.18905](https://arxiv.org/abs/2402.18905)

    本文分析了差分隐私线性探测（LP）和完全微调（FT）的训练动态，探索了从线性探测过渡到完全微调（LP-FT）的顺序微调现象及其对测试损失的影响，提供了关于在超参数化神经网络中差分隐私微调收敛性的理论洞见和隐私预算分配的效用曲线。

    

    差分隐私（DP）机器学习流水线通常包括两个阶段的过程：在公共数据集上进行非私有预训练，然后使用DP优化技术在私有数据上进行微调。在DP设置中，已经观察到完全微调有时候并不总是产生最佳的测试准确度，即使对于分布内数据也是如此。本文（1）分析了DP线性探测（LP）和完全微调（FT）的训练动态，以及（2）探索了顺序微调的现象，从线性探测开始，过渡到完全微调（LP-FT），以及它对测试损失的影响。我们提供了有关DP微调在超参数化神经网络中的收敛性的理论洞见，并建立了一个确定隐私预算在线性探测和完全微调之间分配的效用曲线。理论结果得到了对各种基准和模型的经验评估支持。

    arXiv:2402.18905v1 Announce Type: cross  Abstract: Differentially private (DP) machine learning pipelines typically involve a two-phase process: non-private pre-training on a public dataset, followed by fine-tuning on private data using DP optimization techniques. In the DP setting, it has been observed that full fine-tuning may not always yield the best test accuracy, even for in-distribution data. This paper (1) analyzes the training dynamics of DP linear probing (LP) and full fine-tuning (FT), and (2) explores the phenomenon of sequential fine-tuning, starting with linear probing and transitioning to full fine-tuning (LP-FT), and its impact on test loss. We provide theoretical insights into the convergence of DP fine-tuning within an overparameterized neural network and establish a utility curve that determines the allocation of privacy budget between linear probing and full fine-tuning. The theoretical results are supported by empirical evaluations on various benchmarks and models.
    
[^3]: 元素越笨，整体越聪明。或者，可能并非如此？

    The Less Intelligent the Elements, the More Intelligent the Whole. Or, Possibly Not?. (arXiv:2012.12689v2 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2012.12689](http://arxiv.org/abs/2012.12689)

    我们探讨了个体智能是否对于集体智能的产生是必要的，以及怎样的个体智能有利于更大的集体智能。在Lotka-Volterra模型中，我们发现了一些个体行为，特别是掠食者的行为，有利于与其他种群共存，但如果猎物和掠食者都足够智能以推断彼此的行为，共存将伴随着两个种群的无限增长。

    

    我们探讨了大脑中的神经元与社会中的人类之间的利维坦类比，问自己是否个体智能对于集体智能的产生是必要的，更重要的是，怎样的个体智能有利于更大的集体智能。首先，我们回顾了连接主义认知科学、基于代理的建模、群体心理学、经济学和物理学的不同洞见。随后，我们将这些洞见应用于Lotka-Volterra模型中导致掠食者和猎物要么共存要么全球灭绝的智能类型和程度。我们发现几个个体行为 - 尤其是掠食者的行为 - 有利于共存，最终在一个平衡点周围产生震荡。然而，我们也发现，如果猎物和掠食者都足够智能以推断彼此的行为，共存就会伴随着两个种群的无限增长。由于Lotka-Volterra模型是不稳定的，我们提出了一些未来的研究方向来解决这个问题。

    We explore a Leviathan analogy between neurons in a brain and human beings in society, asking ourselves whether individual intelligence is necessary for collective intelligence to emerge and, most importantly, what sort of individual intelligence is conducive of greater collective intelligence. We first review disparate insights from connectionist cognitive science, agent-based modeling, group psychology, economics and physics. Subsequently, we apply these insights to the sort and degrees of intelligence that in the Lotka-Volterra model lead to either co-existence or global extinction of predators and preys.  We find several individual behaviors -- particularly of predators -- that are conducive to co-existence, eventually with oscillations around an equilibrium. However, we also find that if both preys and predators are sufficiently intelligent to extrapolate one other's behavior, co-existence comes along with indefinite growth of both populations. Since the Lotka-Volterra model is al
    

