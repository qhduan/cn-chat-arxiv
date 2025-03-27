# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models](https://arxiv.org/abs/2403.17246) | 该论文将经典规划和大型语言模型相结合，通过近似人类直觉，以实现多智能体任务规划。 |
| [^2] | [Comparing Styles across Languages.](http://arxiv.org/abs/2310.07135) | 本研究通过引入解释框架，从多语言语言模型中提取风格差异并比较不同语言之间的风格，创建了全面的多语言礼貌数据集，探索了礼貌在四种语言中的变化，为评估语言类别对风格变化的贡献和了解世界各地人们的不同沟通方式提供了有效的方法和解释洞察力。 |
| [^3] | [A Geometric Notion of Causal Probing.](http://arxiv.org/abs/2307.15054) | 本文提出了一种几何观念的因果探测方法，通过在语言模型表示空间的子空间上进行反事实干预，优化了因果概念子空间，以实现概念控制生成。 |

# 详细

[^1]: TwoStep: 使用经典规划器和大型语言模型进行多智能体任务规划

    TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models

    [https://arxiv.org/abs/2403.17246](https://arxiv.org/abs/2403.17246)

    该论文将经典规划和大型语言模型相结合，通过近似人类直觉，以实现多智能体任务规划。

    

    类似规划领域定义语言（PDDL）之类的经典规划公式允许确定可实现目标状态的动作序列，只要存在任何可能的初始状态。然而，PDDL中定义的推理问题并未捕获行动进行的时间方面，例如领域中的两个智能体如果彼此的后况不干扰前提条件，则可以同时执行一个动作。人类专家可以将目标分解为大部分独立的组成部分，并将每个智能体分配给其中一个子目标，以利用同时进行动作来加快计划步骤的执行，每个部分仅使用单个智能体规划。相比之下，直接推断计划步骤的大型语言模型（LLMs）并不保证执行成功，但利用常识推理来组装动作序列。我们通过近似人类直觉，结合了经典规划和LLMs的优势

    arXiv:2403.17246v1 Announce Type: new  Abstract: Classical planning formulations like the Planning Domain Definition Language (PDDL) admit action sequences guaranteed to achieve a goal state given an initial state if any are possible. However, reasoning problems defined in PDDL do not capture temporal aspects of action taking, for example that two agents in the domain can execute an action simultaneously if postconditions of each do not interfere with preconditions of the other. A human expert can decompose a goal into largely independent constituent parts and assign each agent to one of these subgoals to take advantage of simultaneous actions for faster execution of plan steps, each using only single agent planning. By contrast, large language models (LLMs) used for directly inferring plan steps do not guarantee execution success, but do leverage commonsense reasoning to assemble action sequences. We combine the strengths of classical planning and LLMs by approximating human intuition
    
[^2]: 跨语言风格比较研究

    Comparing Styles across Languages. (arXiv:2310.07135v1 [cs.CL])

    [http://arxiv.org/abs/2310.07135](http://arxiv.org/abs/2310.07135)

    本研究通过引入解释框架，从多语言语言模型中提取风格差异并比较不同语言之间的风格，创建了全面的多语言礼貌数据集，探索了礼貌在四种语言中的变化，为评估语言类别对风格变化的贡献和了解世界各地人们的不同沟通方式提供了有效的方法和解释洞察力。

    

    理解跨语言风格的差异有助于训练人类和计算机生成符合文化背景的文本。我们引入了一个解释框架，可以从多语言语言模型中提取风格差异，并比较不同语言之间的风格。我们的框架(1)可以生成任何语言的全面风格词典，(2)将语言模型中的特征重要性统一为可比较的词汇类别。我们应用该框架比较了礼貌语言，创建了第一个全面的多语言礼貌数据集，并探索了礼貌在四种语言中的变化。我们的方法可以有效评估不同语言类别对风格变化的贡献，并提供可解释的洞察力，了解世界各地人们的不同沟通方式。

    Understanding how styles differ across languages is advantageous for training both humans and computers to generate culturally appropriate text. We introduce an explanation framework to extract stylistic differences from multilingual LMs and compare styles across languages. Our framework (1) generates comprehensive style lexica in any language and (2) consolidates feature importances from LMs into comparable lexical categories. We apply this framework to compare politeness, creating the first holistic multilingual politeness dataset and exploring how politeness varies across four languages. Our approach enables an effective evaluation of how distinct linguistic categories contribute to stylistic variations and provides interpretable insights into how people communicate differently around the world.
    
[^3]: 一种几何观念的因果探测

    A Geometric Notion of Causal Probing. (arXiv:2307.15054v1 [cs.CL])

    [http://arxiv.org/abs/2307.15054](http://arxiv.org/abs/2307.15054)

    本文提出了一种几何观念的因果探测方法，通过在语言模型表示空间的子空间上进行反事实干预，优化了因果概念子空间，以实现概念控制生成。

    

    大型语言模型依赖于文本的实值表示来进行预测。这些表示包含了模型在训练数据上学到的信息，包括语言属性和基于性别的人口偏见等。越来越多的研究关注通过在表示空间的子空间上进行正交投影来获得关于这些概念的信息。我们通过提出语言模型表示空间子空间的内在信息的形式定义，为这项研究贡献了新的内容。我们提出了一种反事实方法来避免虚假相关的失效模式，通过独立处理子空间中的分量和其正交补空间中的分量。我们展示了在子空间中的反事实信息概念是由一个因果概念子空间进行优化的。此外，这种干预使我们能够通过操作来尝试概念控制生成。

    Large language models rely on real-valued representations of text to make their predictions. These representations contain information learned from the data that the model has trained on, including knowledge of linguistic properties and forms of demographic bias, e.g., based on gender. A growing body of work has considered information about concepts such as these using orthogonal projections onto subspaces of the representation space. We contribute to this body of work by proposing a formal definition of intrinsic information in a subspace of a language model's representation space. We propose a counterfactual approach that avoids the failure mode of spurious correlations (Kumar et al., 2022) by treating components in the subspace and its orthogonal complement independently. We show that our counterfactual notion of information in a subspace is optimizing by an causal concept subspace. Furthermore, this intervention allows us to attempt concept controlled generation by manipulating the
    

