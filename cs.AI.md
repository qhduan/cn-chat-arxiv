# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Do LLM Agents Have Regret? A Case Study in Online Learning and Games](https://arxiv.org/abs/2403.16843) | 通过研究在线学习和博弈论中的基准决策设置，评估LLM代理的交互行为和性能，以了解它们在多代理环境中的潜力和限制。 |
| [^2] | [Multi: Multimodal Understanding Leaderboard with Text and Images](https://arxiv.org/abs/2402.03173) | Multi是一个多模态理解的排行榜，提供了一个综合数据集，评估多模态大型语言模型对理解复杂图表和科学问题的能力。它兼具准确和开放式的回答形式，挑战MLLM的各种任务，并包含超过18,000个问题。 |
| [^3] | [A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models.](http://arxiv.org/abs/2310.00194) | 这个论文提出了一个受前额叶皮层启发的大型语言模型规划架构，利用多个基于LLM的模块实现规划的自主协调，从而在处理需要多步推理或目标导向规划的任务时取得了较好的效果。 |
| [^4] | [LLM-FuncMapper: Function Identification for Interpreting Complex Clauses in Building Codes via LLM.](http://arxiv.org/abs/2308.08728) | LLM-FuncMapper提出了一种通过LLM实现对建筑法规中复杂条款的函数识别的方法，通过定义原子函数和开发提示模板来解决传统逻辑表示的限制。 |
| [^5] | [Convergence and sample complexity of natural policy gradient primal-dual methods for constrained MDPs.](http://arxiv.org/abs/2206.02346) | 本文研究了约束马尔可夫决策过程中优化问题的自然策略梯度原始-对偶方法。通过自然策略梯度上升和投影次梯度下降更新变量，我们的方法在全局收敛中实现了次线性速率，而且不受状态-动作空间大小限制。 |

# 详细

[^1]: LLM代理是否会感到后悔？在线学习和游戏案例研究

    Do LLM Agents Have Regret? A Case Study in Online Learning and Games

    [https://arxiv.org/abs/2403.16843](https://arxiv.org/abs/2403.16843)

    通过研究在线学习和博弈论中的基准决策设置，评估LLM代理的交互行为和性能，以了解它们在多代理环境中的潜力和限制。

    

    大型语言模型(LLMs)越来越多地被用于(交互式)决策制定，通过开发基于LLM的自主代理。尽管它们取得了不断的成功，但LLM代理在决策制定中的表现尚未通过定量指标进行充分调查，特别是在它们相互作用时的多代理设置中，这是实际应用中的典型场景。为了更好地理解LLM代理在这些交互环境中的限制，我们建议研究它们在在线学习和博弈论的基准决策设置中的相互作用，并通过\emph{后悔}性能指标进行评估。我们首先在经典(非平稳)在线学习问题中经验性地研究LLMs的无后悔行为，以及当LLM代理通过进行重复游戏进行交互时均衡的出现。然后我们对无后悔行为提供一些理论洞见。

    arXiv:2403.16843v1 Announce Type: cross  Abstract: Large language models (LLMs) have been increasingly employed for (interactive) decision-making, via the development of LLM-based autonomous agents. Despite their emerging successes, the performance of LLM agents in decision-making has not been fully investigated through quantitative metrics, especially in the multi-agent setting when they interact with each other, a typical scenario in real-world LLM-agent applications. To better understand the limits of LLM agents in these interactive environments, we propose to study their interactions in benchmark decision-making settings in online learning and game theory, through the performance metric of \emph{regret}. We first empirically study the {no-regret} behaviors of LLMs in canonical (non-stationary) online learning problems, as well as the emergence of equilibria when LLM agents interact through playing repeated games. We then provide some theoretical insights into the no-regret behavior
    
[^2]: 多模态：文本和图像的多模态理解排行榜

    Multi: Multimodal Understanding Leaderboard with Text and Images

    [https://arxiv.org/abs/2402.03173](https://arxiv.org/abs/2402.03173)

    Multi是一个多模态理解的排行榜，提供了一个综合数据集，评估多模态大型语言模型对理解复杂图表和科学问题的能力。它兼具准确和开放式的回答形式，挑战MLLM的各种任务，并包含超过18,000个问题。

    

    多模态大型语言模型（MLLM）的快速进展强调了向学术界引入具有挑战性而又真实的基准的需求。现有的基准主要关注简单的自然图像理解，但Multi成为了MLLM的尖端基准，提供了一个综合性的数据集，用于评估MLLM对理解复杂图表和科学问题的能力。该基准反映了当前真实的考试风格，提供多模态的输入，并要求准确或开放式的回答，类似于现实中的学校考试。它通过各种任务挑战MLLM，从公式推导到图像细节分析，以及跨模态推理。Multi包括超过18,000个问题，重点关注不同格式的基于科学的问答。我们还引入了Multi-Elite，一个包含500个问题的子集，用于测试MLLM的极端情况，以及Multi-Extend，通过超过4..。

    Rapid progress in multimodal large language models (MLLMs) highlights the need to introduce challenging yet realistic benchmarks to the academic community. Existing benchmarks primarily focus on simple natural image understanding, but Multi emerges as a cutting-edge benchmark for MLLMs, offering a comprehensive dataset for evaluating MLLMs against understanding complex figures and tables, and scientific questions. This benchmark, reflecting current realistic examination styles, provides multimodal inputs and requires responses that are either precise or open-ended, similar to real-life school tests. It challenges MLLMs with a variety of tasks, ranging from formula derivation to image detail analysis, and cross-modality reasoning. Multi includes over 18,000 questions, with a focus on science-based QA in diverse formats. We also introduce Multi-Elite, a 500-question subset for testing the extremities of MLLMs, and Multi-Extend, which enhances In-Context Learning research with more than 4
    
[^3]: 受前额叶皮层启发的大型语言模型规划架构

    A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models. (arXiv:2310.00194v1 [cs.AI])

    [http://arxiv.org/abs/2310.00194](http://arxiv.org/abs/2310.00194)

    这个论文提出了一个受前额叶皮层启发的大型语言模型规划架构，利用多个基于LLM的模块实现规划的自主协调，从而在处理需要多步推理或目标导向规划的任务时取得了较好的效果。

    

    大型语言模型（LLM）在许多任务上展现出惊人的性能，但它们经常在需要多步推理或目标导向规划的任务中遇到困难。为了解决这个问题，我们从人脑中获取灵感，即通过前额叶皮层（PFC）中专门模块的重复交互来完成规划。这些模块执行冲突监测、状态预测、状态评估、任务分解和任务协调等功能。我们发现LLM有时能够单独执行这些功能，但在服务于一个目标时往往难以自主协调它们。因此，我们提出了一个带有多个基于LLM（GPT-4）模块的黑盒架构。该架构通过专门的PFC启发模块的交互将一个更大的问题分解为多个对LLM的简短自动调用，从而改善规划能力。我们在两个具有挑战性的规划任务上评估了组合架构。

    Large language models (LLMs) demonstrate impressive performance on a wide variety of tasks, but they often struggle with tasks that require multi-step reasoning or goal-directed planning. To address this, we take inspiration from the human brain, in which planning is accomplished via the recurrent interaction of specialized modules in the prefrontal cortex (PFC). These modules perform functions such as conflict monitoring, state prediction, state evaluation, task decomposition, and task coordination. We find that LLMs are sometimes capable of carrying out these functions in isolation, but struggle to autonomously coordinate them in the service of a goal. Therefore, we propose a black box architecture with multiple LLM-based (GPT-4) modules. The architecture improves planning through the interaction of specialized PFC-inspired modules that break down a larger problem into multiple brief automated calls to the LLM. We evaluate the combined architecture on two challenging planning tasks -
    
[^4]: LLM-FuncMapper:通过LLM解释建筑法规中的复杂条款的函数识别

    LLM-FuncMapper: Function Identification for Interpreting Complex Clauses in Building Codes via LLM. (arXiv:2308.08728v1 [cs.AI])

    [http://arxiv.org/abs/2308.08728](http://arxiv.org/abs/2308.08728)

    LLM-FuncMapper提出了一种通过LLM实现对建筑法规中复杂条款的函数识别的方法，通过定义原子函数和开发提示模板来解决传统逻辑表示的限制。

    

    作为自动化规则检查（ARC）的关键阶段，对监管性文本的规则解释需要相当大的努力。然而，由于缺乏领域知识和传统逻辑表示的表达能力有限，解释具有隐式属性或复杂计算逻辑的监管条款仍然具有挑战性。因此，提出了一种基于大型语言模型（LLM）的识别各种监管条款所需的预定义函数的方法LLM-FuncMapper。首先，通过对建筑法规进行系统分析，定义了一系列原子函数，以捕捉隐式属性和复杂约束的共享计算逻辑，创建了常见块的数据库，用于解释监管条款。然后，开发了一个具有思维链的提示模板，并通过基于分类的调优策略进一步增强，以实现有效的函数识别功能。最后，验证了所提出的方法。

    As a vital stage of automated rule checking (ARC), rule interpretation of regulatory texts requires considerable effort. However, interpreting regulatory clauses with implicit properties or complex computational logic is still challenging due to the lack of domain knowledge and limited expressibility of conventional logic representations. Thus, LLM-FuncMapper, an approach to identifying predefined functions needed to interpret various regulatory clauses based on the large language model (LLM), is proposed. First, by systematically analysis of building codes, a series of atomic functions are defined to capture shared computational logics of implicit properties and complex constraints, creating a database of common blocks for interpreting regulatory clauses. Then, a prompt template with the chain of thought is developed and further enhanced with a classification-based tuning strategy, to enable common LLMs for effective function identification. Finally, the proposed approach is validated
    
[^5]: 自然策略梯度原始-对偶方法在约束MDP中的收敛性和样本复杂度研究

    Convergence and sample complexity of natural policy gradient primal-dual methods for constrained MDPs. (arXiv:2206.02346v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2206.02346](http://arxiv.org/abs/2206.02346)

    本文研究了约束马尔可夫决策过程中优化问题的自然策略梯度原始-对偶方法。通过自然策略梯度上升和投影次梯度下降更新变量，我们的方法在全局收敛中实现了次线性速率，而且不受状态-动作空间大小限制。

    

    我们研究了顺序决策问题，旨在最大化预期总奖励，同时满足对预期总效用的约束。我们使用自然策略梯度方法来解决约束马尔可夫决策过程（约束MDP）的折扣无限时序优化控制问题。具体地，我们提出了一种新的自然策略梯度原始-对偶（NPG-PD）方法，该方法通过自然策略梯度上升更新原始变量，通过投影次梯度下降更新对偶变量。尽管底层最大化涉及非凸目标函数和非凸约束集，但在softmax策略参数化下，我们证明了我们的方法在优化间隙和约束违规方面实现全局收敛，并具有次线性速率。此类收敛与状态-动作空间的大小无关，即无维度限制。此外，对于对数线性和一般平滑策略参数化，我们确立了收敛性和样本复杂度界限。

    We study sequential decision making problems aimed at maximizing the expected total reward while satisfying a constraint on the expected total utility. We employ the natural policy gradient method to solve the discounted infinite-horizon optimal control problem for Constrained Markov Decision Processes (constrained MDPs). Specifically, we propose a new Natural Policy Gradient Primal-Dual (NPG-PD) method that updates the primal variable via natural policy gradient ascent and the dual variable via projected sub-gradient descent. Although the underlying maximization involves a nonconcave objective function and a nonconvex constraint set, under the softmax policy parametrization we prove that our method achieves global convergence with sublinear rates regarding both the optimality gap and the constraint violation. Such convergence is independent of the size of the state-action space, i.e., it is~dimension-free. Furthermore, for log-linear and general smooth policy parametrizations, we esta
    

