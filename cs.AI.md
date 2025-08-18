# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey on Recent Advances in LLM-Based Multi-turn Dialogue Systems](https://arxiv.org/abs/2402.18013) | 这项调查综述了基于LLM的多轮对话系统的研究，并重点介绍了LLMs的应用和最新进展，对开放领域对话和任务导向对话系统进行了涵盖，并讨论了相关数据集和评估指标，以及未来研究方向和问题。 |
| [^2] | [JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example.](http://arxiv.org/abs/2401.01199) | JMA是一种通用算法，用于生成几乎最优的定向对抗样本。该算法通过最小化Jacobian引起的马氏距离，考虑了将输入样本的潜在空间表示在给定方向上移动所需的投入。该算法在解决对抗样本问题方面提供了最优解。 |
| [^3] | [Large-Scale Multi-Robot Assembly Planning for Autonomous Manufacturing.](http://arxiv.org/abs/2311.00192) | 本论文提出了一个大规模多机器人组装规划的算法堆栈，可以在短时间内合成复杂装配的建造计划，并解决了移动自主机器人在制造中面临的挑战。 |
| [^4] | [Planning to Learn: A Novel Algorithm for Active Learning during Model-Based Planning.](http://arxiv.org/abs/2308.08029) | 本论文提出了一种新颖的算法，称为SI SL，用于主动学习和模型优化过程中的规划。该算法通过与贝叶斯强化学习方案的比较证明了其性能的优越性。 |

# 详细

[^1]: 基于LLM的多轮对话系统最新进展综述

    A Survey on Recent Advances in LLM-Based Multi-turn Dialogue Systems

    [https://arxiv.org/abs/2402.18013](https://arxiv.org/abs/2402.18013)

    这项调查综述了基于LLM的多轮对话系统的研究，并重点介绍了LLMs的应用和最新进展，对开放领域对话和任务导向对话系统进行了涵盖，并讨论了相关数据集和评估指标，以及未来研究方向和问题。

    

    这项调查全面回顾了关于多轮对话系统的研究，特别侧重于基于大型语言模型（LLMs）的多轮对话系统。本文旨在（a）总结现有的LLMs和适应LLMs进行下游任务的方法；（b）详细阐述多轮对话系统的最新进展，涵盖基于LLM的开放领域对话（ODD）和任务导向对话（TOD）系统，以及数据集和评估指标；（c）讨论由于LLMs的发展和对多轮对话系统不断增加的需求而产生的未来重点和最新研究问题。

    arXiv:2402.18013v1 Announce Type: cross  Abstract: This survey provides a comprehensive review of research on multi-turn dialogue systems, with a particular focus on multi-turn dialogue systems based on large language models (LLMs). This paper aims to (a) give a summary of existing LLMs and approaches for adapting LLMs to downstream tasks; (b) elaborate recent advances in multi-turn dialogue systems, covering both LLM-based open-domain dialogue (ODD) and task-oriented dialogue (TOD) systems, along with datasets and evaluation metrics; (c) discuss some future emphasis and recent research problems arising from the development of LLMs and the increasing demands on multi-turn dialogue systems.
    
[^2]: JMA:一种快速生成几乎最优定向对抗样本的通用算法

    JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example. (arXiv:2401.01199v1 [cs.LG])

    [http://arxiv.org/abs/2401.01199](http://arxiv.org/abs/2401.01199)

    JMA是一种通用算法，用于生成几乎最优的定向对抗样本。该算法通过最小化Jacobian引起的马氏距离，考虑了将输入样本的潜在空间表示在给定方向上移动所需的投入。该算法在解决对抗样本问题方面提供了最优解。

    

    目前为止，大多数用于生成针对深度学习分类器的定向对抗样本的方法都是高度次优的，通常依赖于增加目标类别的可能性，因此隐含地专注于一热编码设置。在本文中，我们提出了一种更加通用的、理论上可靠的定向攻击方法，该方法利用最小化雅可比引起的马氏距离（JMA）项，考虑将输入样本的潜在空间表示在给定方向上移动所需的投入（在输入空间中）。通过利用沃尔夫二重性定理求解最小化问题，将问题简化为解非负最小二乘（NNLS）问题。所提出的算法为Szegedy等人最初引入的对抗样本问题的线性化版本提供了最优解。我们进行的实验证实了所提出的攻击的广泛性。

    Most of the approaches proposed so far to craft targeted adversarial examples against Deep Learning classifiers are highly suboptimal and typically rely on increasing the likelihood of the target class, thus implicitly focusing on one-hot encoding settings. In this paper, we propose a more general, theoretically sound, targeted attack that resorts to the minimization of a Jacobian-induced MAhalanobis distance (JMA) term, taking into account the effort (in the input space) required to move the latent space representation of the input sample in a given direction. The minimization is solved by exploiting the Wolfe duality theorem, reducing the problem to the solution of a Non-Negative Least Square (NNLS) problem. The proposed algorithm provides an optimal solution to a linearized version of the adversarial example problem originally introduced by Szegedy et al. \cite{szegedy2013intriguing}. The experiments we carried out confirm the generality of the proposed attack which is proven to be 
    
[^3]: 大规模多机器人组装规划用于自主制造

    Large-Scale Multi-Robot Assembly Planning for Autonomous Manufacturing. (arXiv:2311.00192v1 [cs.RO])

    [http://arxiv.org/abs/2311.00192](http://arxiv.org/abs/2311.00192)

    本论文提出了一个大规模多机器人组装规划的算法堆栈，可以在短时间内合成复杂装配的建造计划，并解决了移动自主机器人在制造中面临的挑战。

    

    移动自主机器人有潜力革新制造流程。然而，将大型机器人群体应用于制造领域需要解决一些挑战，包括在共享工作空间中实现无碰撞移动，有效的多机器人协作来操纵和运输大型负载，由于耦合的制造流程导致复杂的任务分配，以及嵌套子装配件的并行装配和运输的空间规划。我们提出了一个完整的算法堆栈，用于大规模多机器人组装规划，可以在几分钟内合成具有数千个部件的复杂装配的建造计划。我们的方法接受类似CAD的产品规范，并自动为一组机器人规划全栈装配过程来制造产品。我们提出了一个算法堆栈，包括：(i)迭代径向布局优化过程，定义制造过程的全局调度布局。

    Mobile autonomous robots have the potential to revolutionize manufacturing processes. However, employing large robot fleets in manufacturing requires addressing challenges including collision-free movement in a shared workspace, effective multi-robot collaboration to manipulate and transport large payloads, complex task allocation due to coupled manufacturing processes, and spatial planning for parallel assembly and transportation of nested subassemblies. We propose a full algorithmic stack for large-scale multi-robot assembly planning that addresses these challenges and can synthesize construction plans for complex assemblies with thousands of parts in a matter of minutes. Our approach takes in a CAD-like product specification and automatically plans a full-stack assembly procedure for a group of robots to manufacture the product. We propose an algorithmic stack that comprises: (i) an iterative radial layout optimization procedure to define a global staging layout for the manufacturin
    
[^4]: 规划学习：一种新颖的模型优化过程中的主动学习算法

    Planning to Learn: A Novel Algorithm for Active Learning during Model-Based Planning. (arXiv:2308.08029v1 [cs.AI])

    [http://arxiv.org/abs/2308.08029](http://arxiv.org/abs/2308.08029)

    本论文提出了一种新颖的算法，称为SI SL，用于主动学习和模型优化过程中的规划。该算法通过与贝叶斯强化学习方案的比较证明了其性能的优越性。

    

    主动推理是一种近期的对不确定性情境下规划建模的框架。现在人们已经开始评估这种方法的优缺点以及如何改进它。最近的一个拓展-复杂模型优化算法通过递归决策树搜索在多步规划问题上提高了性能。然而，迄今为止很少有工作对比SI与其他已建立的规划算法。SI算法也主要关注推理而不是学习。本文有两个目标。首先，我们比较SI与旨在解决相似问题的贝叶斯强化学习（RL）方案的性能。其次，我们提出了SI复杂学习（SL）的拓展，该拓展在规划过程中更加充分地引入了主动学习。SL维持对未来观测下每个策略下模型参数如何变化的信念。这允许了一种反事实的回顾性评估。

    Active Inference is a recent framework for modeling planning under uncertainty. Empirical and theoretical work have now begun to evaluate the strengths and weaknesses of this approach and how it might be improved. A recent extension - the sophisticated inference (SI) algorithm - improves performance on multi-step planning problems through recursive decision tree search. However, little work to date has been done to compare SI to other established planning algorithms. SI was also developed with a focus on inference as opposed to learning. The present paper has two aims. First, we compare performance of SI to Bayesian reinforcement learning (RL) schemes designed to solve similar problems. Second, we present an extension of SI sophisticated learning (SL) - that more fully incorporates active learning during planning. SL maintains beliefs about how model parameters would change under the future observations expected under each policy. This allows a form of counterfactual retrospective in
    

