# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Climbing the Ladder of Interpretability with Counterfactual Concept Bottleneck Models](https://rss.arxiv.org/abs/2402.01408) | 本论文提出了一种新的模型 CF-CBMs，可以同时解决深度学习模型的预测、解释和想象能力的不足，为部署可靠的AI代理、校准人类信任和加深人机交互提供了一种有效的解决方法。 |
| [^2] | [Techniques for Measuring the Inferential Strength of Forgetting Policies](https://arxiv.org/abs/2404.02454) | 本文提出了一种衡量原理论推理强度变化的损失函数，并使用Problog工具计算损失度量，最终得出了关于不同遗忘策略强度的研究方法和实际应用示例。 |
| [^3] | [Reinforcement Learning-based Receding Horizon Control using Adaptive Control Barrier Functions for Safety-Critical Systems](https://arxiv.org/abs/2403.17338) | 提出了一种基于强化学习的滚动视野控制方法，利用自适应控制屏障函数，以解决安全关键系统中性能和可行性受影响的问题 |
| [^4] | [Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It](https://arxiv.org/abs/2403.14715) | LS方法在深度神经网络分类器训练中的标签平滑效果被发现会负面影响选择性分类，通过影响模型预测不确定性，此研究阐明了这一现象。 |
| [^5] | [Substrate Scope Contrastive Learning: Repurposing Human Bias to Learn Atomic Representations](https://arxiv.org/abs/2402.16882) | 提出了一种新颖的预训练策略，底物范围对比学习，以学习适合化学反应性的原子表示。 |
| [^6] | [Learning Planning Action Models from State Traces](https://arxiv.org/abs/2402.10726) | 该论文探讨了一种从状态轨迹中学习规划行动模型的方法，尤其关注在参数未提供的情况下进行学习，并提出了两种不同级别的跟踪质量以及相应的算法。 |
| [^7] | [On Memorization in Diffusion Models.](http://arxiv.org/abs/2310.02664) | 本论文研究了扩散模型的记忆化行为，发现记忆化倾向于在较小的数据集上发生。通过定义有效模型记忆化 (EMM) 这一指标，量化了数据分布和模型配置对记忆化行为的影响。 |
| [^8] | [Lifted Inference beyond First-Order Logic.](http://arxiv.org/abs/2308.11738) | 这项工作研究了超越一阶逻辑的提升推理问题，扩展了计数量词扩展的两个变量的一阶逻辑片段的域可提升性，并在限定了关系的情况下探索了不同属性的域可提升性。 |
| [^9] | [Testing GPT-4 with Wolfram Alpha and Code Interpreter plug-ins on math and science problems.](http://arxiv.org/abs/2308.05713) | 本研究测试了GPT-4在科学和数学问题上使用Wolfram Alpha和Code Interpreter插件的效果，结果表明插件显著提升了GPT的问题解决能力，但接口故障仍然是其可靠性的主要挑战。 |
| [^10] | [On the Effective Horizon of Inverse Reinforcement Learning.](http://arxiv.org/abs/2307.06541) | 本研究分析了逆强化学习中时间视野的重要性，发现短于实际值的有效时间视野可以更快且更准确地估计奖励函数，减轻过拟合问题。此外，研究还呼吁在IRL中同时学习奖励和有效时间视野。 |
| [^11] | [Robust Classification of High-Dimensional Data using Data-Adaptive Energy Distance.](http://arxiv.org/abs/2306.13985) | 该论文提出了一种用于高维低样本量数据分类的稳健的数据自适应能量距离分类器，该分类器无需调参且在一定条件下可以实现完美分类，已在模拟研究和实际数据分析中得到证明比其他方法表现更优。 |

# 详细

[^1]: 通过反事实概念瓶颈模型攀登解释性的阶梯

    Climbing the Ladder of Interpretability with Counterfactual Concept Bottleneck Models

    [https://rss.arxiv.org/abs/2402.01408](https://rss.arxiv.org/abs/2402.01408)

    本论文提出了一种新的模型 CF-CBMs，可以同时解决深度学习模型的预测、解释和想象能力的不足，为部署可靠的AI代理、校准人类信任和加深人机交互提供了一种有效的解决方法。

    

    当前的深度学习模型没有同时解决三个基本问题的设计：预测类别标签以解决给定的分类任务（“是什么？”），解释任务预测（“为什么？”），并想象可能导致不同预测的替代情景（“如果怎样？”）。无法回答这些问题代表了部署可靠的AI代理、校准人类信任和加深人机交互的关键差距。为了弥合这一差距，我们引入了反事实概念瓶颈模型（CF-CBMs），这是一类能够高效同时解决上述查询而无需进行事后搜索的模型。我们的结果表明，CF-CBMs能够产生准确的预测（“是什么？”），对任务预测提供简单的解释（“为什么？”），以及可解释的反事实情况（“如果怎样？”）。CF-CBMs还可以对概念干预的影响进行采样或估计最可能的反事实情况，以解释事件，并优化产生多样化的反事实。

    Current deep learning models are not designed to simultaneously address three fundamental questions: predict class labels to solve a given classification task (the "What?"), explain task predictions (the "Why?"), and imagine alternative scenarios that could result in different predictions (the "What if?"). The inability to answer these questions represents a crucial gap in deploying reliable AI agents, calibrating human trust, and deepening human-machine interaction. To bridge this gap, we introduce CounterFactual Concept Bottleneck Models (CF-CBMs), a class of models designed to efficiently address the above queries all at once without the need to run post-hoc searches. Our results show that CF-CBMs produce: accurate predictions (the "What?"), simple explanations for task predictions (the "Why?"), and interpretable counterfactuals (the "What if?"). CF-CBMs can also sample or estimate the most probable counterfactual to: (i) explain the effect of concept interventions on tasks, (ii) sh
    
[^2]: 衡量遗忘策略的推理强度技术

    Techniques for Measuring the Inferential Strength of Forgetting Policies

    [https://arxiv.org/abs/2404.02454](https://arxiv.org/abs/2404.02454)

    本文提出了一种衡量原理论推理强度变化的损失函数，并使用Problog工具计算损失度量，最终得出了关于不同遗忘策略强度的研究方法和实际应用示例。

    

    知识表示中的遗忘技术被证明是一种强大且有广泛应用的知识工程工具。然而，关于不同的遗忘策略或不同遗忘操作符的使用如何影响原理论的推理强度几乎没有研究。本文旨在根据模型计数和概率理论的直觉定义用于衡量推理强度变化的损失函数。研究了此类损失度量的性质，并提出了一种实用的知识工程工具，用于使用Problog计算损失度量。论文包括一个用于研究和确定不同遗忘策略强度的工作方法，以及展示如何利用Problog应用理论结果的具体示例。虽然重点是遗忘，但结果更为普遍，并且应具有更广泛的应用。

    arXiv:2404.02454v1 Announce Type: new  Abstract: The technique of forgetting in knowledge representation has been shown to be a powerful and useful knowledge engineering tool with widespread application. Yet, very little research has been done on how different policies of forgetting, or use of different forgetting operators, affects the inferential strength of the original theory. The goal of this paper is to define loss functions for measuring changes in inferential strength based on intuitions from model counting and probability theory. Properties of such loss measures are studied and a pragmatic knowledge engineering tool is proposed for computing loss measures using Problog. The paper includes a working methodology for studying and determining the strength of different forgetting policies, in addition to concrete examples showing how to apply the theoretical results using Problog. Although the focus is on forgetting, the results are much more general and should have wider applicati
    
[^3]: 使用自适应控制屏障函数的基于强化学习的滚动视野控制用于安全关键系统

    Reinforcement Learning-based Receding Horizon Control using Adaptive Control Barrier Functions for Safety-Critical Systems

    [https://arxiv.org/abs/2403.17338](https://arxiv.org/abs/2403.17338)

    提出了一种基于强化学习的滚动视野控制方法，利用自适应控制屏障函数，以解决安全关键系统中性能和可行性受影响的问题

    

    最优控制方法为安全关键问题提供解决方案，但很容易变得棘手。控制屏障函数(CBFs)作为一种流行技术出现，通过其前向不变性属性，有利于通过在损失一些性能的情况下，显式地保证安全。该方法涉及定义性能目标以及必须始终执行的基于CBF的安全约束。遗憾的是，两个关键因素可能会对性能和解决方案的可行性产生显著影响：(i)成本函数及其相关参数的选择，以及(ii)在CBF约束内进行参数校准，捕捉性能和保守性之间的折衷，以及不可行性。为了解决这些挑战，我们提出了一种利用模型预测控制(MPC)的强化学习(RL)滚动视野控制(RHC)方法。

    arXiv:2403.17338v1 Announce Type: cross  Abstract: Optimal control methods provide solutions to safety-critical problems but easily become intractable. Control Barrier Functions (CBFs) have emerged as a popular technique that facilitates their solution by provably guaranteeing safety, through their forward invariance property, at the expense of some performance loss. This approach involves defining a performance objective alongside CBF-based safety constraints that must always be enforced. Unfortunately, both performance and solution feasibility can be significantly impacted by two key factors: (i) the selection of the cost function and associated parameters, and (ii) the calibration of parameters within the CBF-based constraints, which capture the trade-off between performance and conservativeness. %as well as infeasibility. To address these challenges, we propose a Reinforcement Learning (RL)-based Receding Horizon Control (RHC) approach leveraging Model Predictive Control (MPC) with
    
[^4]: 理解为何标签平滑会降低选择性分类的效果以及如何解决这个问题

    Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It

    [https://arxiv.org/abs/2403.14715](https://arxiv.org/abs/2403.14715)

    LS方法在深度神经网络分类器训练中的标签平滑效果被发现会负面影响选择性分类，通过影响模型预测不确定性，此研究阐明了这一现象。

    

    标签平滑（LS）是一种流行的深度神经网络分类器训练的正则化方法，因为它在提高测试准确性方面效果显著，并且实现简单。"硬"的one-hot标签通过将概率质量均匀分配给其他类别来进行"平滑化"，从而减少过度拟合。在这项工作中，我们揭示了LS如何负面影响选择性分类（SC）- 其目标是利用模型的预测不确定性来拒绝错误分类。我们首先在一系列任务和架构中从经验上证明LS会导致SC的一致性降级。然后，我们通过分析logit级别的梯度来解释这一点，表明LS通过在错误概率低时更加正则化最大logit，而在错误概率高时更少正则化，加剧了过度自信和低自信。这阐明了以前报道的强分类器在SC中性能不佳的实验结果。

    arXiv:2403.14715v1 Announce Type: cross  Abstract: Label smoothing (LS) is a popular regularisation method for training deep neural network classifiers due to its effectiveness in improving test accuracy and its simplicity in implementation. "Hard" one-hot labels are "smoothed" by uniformly distributing probability mass to other classes, reducing overfitting. In this work, we reveal that LS negatively affects selective classification (SC) - where the aim is to reject misclassifications using a model's predictive uncertainty. We first demonstrate empirically across a range of tasks and architectures that LS leads to a consistent degradation in SC. We then explain this by analysing logit-level gradients, showing that LS exacerbates overconfidence and underconfidence by regularising the max logit more when the probability of error is low, and less when the probability of error is high. This elucidates previously reported experimental results where strong classifiers underperform in SC. We
    
[^5]: 底物范围对比学习：重新利用人类偏见学习原子表示

    Substrate Scope Contrastive Learning: Repurposing Human Bias to Learn Atomic Representations

    [https://arxiv.org/abs/2402.16882](https://arxiv.org/abs/2402.16882)

    提出了一种新颖的预训练策略，底物范围对比学习，以学习适合化学反应性的原子表示。

    

    学习分子表示是分子机器学习中的关键步骤，对建模成功产生显著影响，尤其在数据稀缺情况下。广义预训练神经网络的概念推动了计算机视觉、自然语言处理和蛋白质工程等领域的发展。然而，类似的方法在小有机分子方面并未取得类似的成功。在这项工作中，我们引入一种新颖的预训练策略，即底物范围对比学习，它学习适合化学反应性的原子表示。这种方法以已发表的底物范围表中底物的分组和产物收率作为化学反应性相似性或不相似性的衡量。我们关注 CAS Content Collection 中的 20,798 个芳香卤代烃，涵盖数千篇出版物，以学习芳香卤代烃的反应性表示。我们验证了我们的预训练方法。

    arXiv:2402.16882v1 Announce Type: cross  Abstract: Learning molecular representation is a critical step in molecular machine learning that significantly influences modeling success, particularly in data-scarce situations. The concept of broadly pre-training neural networks has advanced fields such as computer vision, natural language processing, and protein engineering. However, similar approaches for small organic molecules have not achieved comparable success. In this work, we introduce a novel pre-training strategy, substrate scope contrastive learning, which learns atomic representations tailored to chemical reactivity. This method considers the grouping of substrates and their yields in published substrate scope tables as a measure of their similarity or dissimilarity in terms of chemical reactivity. We focus on 20,798 aryl halides in the CAS Content Collection spanning thousands of publications to learn a representation of aryl halide reactivity. We validate our pre-training appr
    
[^6]: 从状态轨迹中学习规划行动模型

    Learning Planning Action Models from State Traces

    [https://arxiv.org/abs/2402.10726](https://arxiv.org/abs/2402.10726)

    该论文探讨了一种从状态轨迹中学习规划行动模型的方法，尤其关注在参数未提供的情况下进行学习，并提出了两种不同级别的跟踪质量以及相应的算法。

    

    先前从状态轨迹中学习STRIPS领域模型的方法通常从要学习的行动的名称和参数开始。因此，它们的唯一任务是推断给定行动的前提条件和效应。在这项工作中，我们探讨了在学习时未提供学习行动的参数的情况。我们根据提供的信息定义了两个级别的跟踪质量，并提出了相应的算法。在一个级别(L1)中，轨迹中的状态被标记为行动名称，因此我们可以推断出行动的数量和名称，但我们仍需要弄清参数的数量和类型。在另一个级别(L2)中，状态还额外标记有构成相应基于对象的行动的参数。在这里，我们仍然需要推断学习行动中参数的类型。我们对所提出的算法进行了实验评估，并将其与其他方法进行了比较。

    arXiv:2402.10726v1 Announce Type: new  Abstract: Previous STRIPS domain model acquisition approaches that learn from state traces start with the names and parameters of the actions to be learned. Therefore their only task is to deduce the preconditions and effects of the given actions. In this work, we explore learning in situations when the parameters of learned actions are not provided. We define two levels of trace quality based on which information is provided and present an algorithm for each. In one level (L1), the states in the traces are labeled with action names, so we can deduce the number and names of the actions, but we still need to work out the number and types of parameters. In the other level (L2), the states are additionally labeled with objects that constitute the parameters of the corresponding grounded actions. Here we still need to deduce the types of the parameters in the learned actions. We experimentally evaluate the proposed algorithms and compare them with the
    
[^7]: 关于扩散模型记忆化的研究

    On Memorization in Diffusion Models. (arXiv:2310.02664v1 [cs.LG])

    [http://arxiv.org/abs/2310.02664](http://arxiv.org/abs/2310.02664)

    本论文研究了扩散模型的记忆化行为，发现记忆化倾向于在较小的数据集上发生。通过定义有效模型记忆化 (EMM) 这一指标，量化了数据分布和模型配置对记忆化行为的影响。

    

    近年来，由于其生成新颖高质量样本的能力，扩散模型引起了广泛的研究兴趣。然而，通过典型的训练目标，即去噪得分匹配，扩散模型只能生成复制训练数据的样本，这表明在理论上会出现记忆化的行为，这与现有先进扩散模型的普遍泛化能力相矛盾，因此需要深入理解。我们观察到记忆化行为倾向于在较小的数据集上发生，我们提出了有效模型记忆化(EMM)的定义，这是一种衡量学习的扩散模型在最大数据集上近似其理论最优点的度量标准。然后，我们量化了影响这些记忆化行为的重要因素，重点关注数据分布和模型配置。

    Due to their capacity to generate novel and high-quality samples, diffusion models have attracted significant research interest in recent years. Notably, the typical training objective of diffusion models, i.e., denoising score matching, has a closed-form optimal solution that can only generate training data replicating samples. This indicates that a memorization behavior is theoretically expected, which contradicts the common generalization ability of state-of-the-art diffusion models, and thus calls for a deeper understanding. Looking into this, we first observe that memorization behaviors tend to occur on smaller-sized datasets, which motivates our definition of effective model memorization (EMM), a metric measuring the maximum size of training data at which a learned diffusion model approximates its theoretical optimum. Then, we quantify the impact of the influential factors on these memorization behaviors in terms of EMM, focusing primarily on data distribution, model configuratio
    
[^8]: 超出一阶逻辑的提升推理

    Lifted Inference beyond First-Order Logic. (arXiv:2308.11738v1 [cs.AI])

    [http://arxiv.org/abs/2308.11738](http://arxiv.org/abs/2308.11738)

    这项工作研究了超越一阶逻辑的提升推理问题，扩展了计数量词扩展的两个变量的一阶逻辑片段的域可提升性，并在限定了关系的情况下探索了不同属性的域可提升性。

    

    在统计关系学习模型中，加权一阶模型计数(WFOMC)是概率推理的基础。由于WFOMC在一般情况下是不可计算的（$\#$P完全），因此能够在多项式时间内进行WFOMC的逻辑碎片非常有意义。这样的碎片被称为域可提升。最近的研究表明，在计数量词（$\mathrm{C^2}$）扩展的两个变量的一阶逻辑片段中，可以进行域提升。然而，许多真实世界数据的属性，如引用网络中的非循环性和社交网络中的连通性，不能在$\mathrm{C^2}$或一阶逻辑中建模。在这项工作中，我们扩展了$\mathrm{C^2}$的域可提升性，包括多个这样的属性。我们证明了在将$\mathrm{C^2}$句子的一个关系限定为表示有向无环图、连通图、树（或有向树）或森林（或有向森林）时，它仍然保持了域可提升性。所有我们的结果都是...

    Weighted First Order Model Counting (WFOMC) is fundamental to probabilistic inference in statistical relational learning models. As WFOMC is known to be intractable in general ($\#$P-complete), logical fragments that admit polynomial time WFOMC are of significant interest. Such fragments are called domain liftable. Recent works have shown that the two-variable fragment of first order logic extended with counting quantifiers ($\mathrm{C^2}$) is domain-liftable. However, many properties of real-world data, like acyclicity in citation networks and connectivity in social networks, cannot be modeled in $\mathrm{C^2}$, or first order logic in general. In this work, we expand the domain liftability of $\mathrm{C^2}$ with multiple such properties. We show that any $\mathrm{C^2}$ sentence remains domain liftable when one of its relations is restricted to represent a directed acyclic graph, a connected graph, a tree (resp. a directed tree) or a forest (resp. a directed forest). All our results r
    
[^9]: 通过在数学和科学问题上使用Wolfram Alpha和Code Interpreter插件测试GPT-4

    Testing GPT-4 with Wolfram Alpha and Code Interpreter plug-ins on math and science problems. (arXiv:2308.05713v1 [cs.AI])

    [http://arxiv.org/abs/2308.05713](http://arxiv.org/abs/2308.05713)

    本研究测试了GPT-4在科学和数学问题上使用Wolfram Alpha和Code Interpreter插件的效果，结果表明插件显著提升了GPT的问题解决能力，但接口故障仍然是其可靠性的主要挑战。

    

    本报告描述了在2023年6月至8月期间对大型语言模型GPT-4在科学和数学领域进行的105个原创问题的测试，其中使用了Wolfram Alpha和Code Interpreter插件。我们的测试表明，这些插件显著增强了GPT解决这些问题的能力。然而，仍然经常出现“接口”故障；也就是说，GPT经常在问题的表述上遇到困难，无法从插件中得到有用的答案。解决这些接口故障似乎是使GPT成为可靠的大学级计算问题工具的关键挑战。

    This report describes a test of the large language model GPT-4 with the Wolfram Alpha and the Code Interpreter plug-ins on 105 original problems in science and math, at the high school and college levels, carried out in June-August 2023. Our tests suggest that the plug-ins significantly enhance GPT's ability to solve these problems. Having said that, there are still often "interface" failures; that is, GPT often has trouble formulating problems in a way that elicits useful answers from the plug-ins. Fixing these interface failures seems like a central challenge in making GPT a reliable tool for college-level calculation problems.
    
[^10]: 逆强化学习中的有效时间视野研究

    On the Effective Horizon of Inverse Reinforcement Learning. (arXiv:2307.06541v1 [cs.LG])

    [http://arxiv.org/abs/2307.06541](http://arxiv.org/abs/2307.06541)

    本研究分析了逆强化学习中时间视野的重要性，发现短于实际值的有效时间视野可以更快且更准确地估计奖励函数，减轻过拟合问题。此外，研究还呼吁在IRL中同时学习奖励和有效时间视野。

    

    逆强化学习（IRL）算法通常依赖于基于给定时间视野的（前向）强化学习或规划来计算一个近似最优策略，然后将该策略与专家演示匹配。时间视野在确定奖励估计的准确性和IRL算法的计算效率方面起着关键作用。有趣的是，比地面实际值更短的有效时间视野通常能更快地产生更好的结果。本文对此现象进行了正式分析并给出了解释：时间视野控制了引发策略类的复杂性，并在有限数据下减轻过拟合。这一分析为IRL的有效视野选择提供了原则性指导。它也促使我们重新审视经典的IRL公式：与仅具有给定视野的奖励相比，共同学习奖励和有效视野更加自然。我们的实验进一步验证了这一观点。

    Inverse reinforcement learning (IRL) algorithms often rely on (forward) reinforcement learning or planning over a given time horizon to compute an approximately optimal policy for a hypothesized reward function and then match this policy with expert demonstrations. The time horizon plays a critical role in determining both the accuracy of reward estimate and the computational efficiency of IRL algorithms. Interestingly, an effective time horizon shorter than the ground-truth value often produces better results faster. This work formally analyzes this phenomenon and provides an explanation: the time horizon controls the complexity of an induced policy class and mitigates overfitting with limited data. This analysis leads to a principled choice of the effective horizon for IRL. It also prompts us to reexamine the classic IRL formulation: it is more natural to learn jointly the reward and the effective horizon together rather than the reward alone with a given horizon. Our experimental re
    
[^11]: 使用数据自适应能量距离的高维数据稳健分类

    Robust Classification of High-Dimensional Data using Data-Adaptive Energy Distance. (arXiv:2306.13985v1 [stat.ML])

    [http://arxiv.org/abs/2306.13985](http://arxiv.org/abs/2306.13985)

    该论文提出了一种用于高维低样本量数据分类的稳健的数据自适应能量距离分类器，该分类器无需调参且在一定条件下可以实现完美分类，已在模拟研究和实际数据分析中得到证明比其他方法表现更优。

    

    在真实世界中，高维低样本量（HDLSS）数据的分类面临挑战，例如基因表达研究、癌症研究和医学成像等领域。本文提出了一些专门为HDLSS数据设计的分类器的开发和分析。这些分类器没有调节参数，并且是稳健的，因为它们不受底层数据分布的任何矩条件的影响。研究表明，在一些相当普遍的条件下，它们在HDLSS渐近区域内可以实现完美分类。还比较了所提出分类器的性能。我们的理论结果得到了广泛的模拟研究和实际数据分析的支持，证明了所提出分类技术优于几种广泛认可的方法的有希望优势。

    Classification of high-dimensional low sample size (HDLSS) data poses a challenge in a variety of real-world situations, such as gene expression studies, cancer research, and medical imaging. This article presents the development and analysis of some classifiers that are specifically designed for HDLSS data. These classifiers are free of tuning parameters and are robust, in the sense that they are devoid of any moment conditions of the underlying data distributions. It is shown that they yield perfect classification in the HDLSS asymptotic regime, under some fairly general conditions. The comparative performance of the proposed classifiers is also investigated. Our theoretical results are supported by extensive simulation studies and real data analysis, which demonstrate promising advantages of the proposed classification techniques over several widely recognized methods.
    

