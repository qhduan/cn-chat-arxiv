# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Efficient Risk-Sensitive Policy Gradient: An Iteration Complexity Analysis](https://arxiv.org/abs/2403.08955) | 本文对风险敏感策略梯度方法进行了迭代复杂度分析，发现其能够通过使用指数效用函数达到较低的迭代复杂度。 |
| [^2] | [Adversarial Robustness Through Artifact Design](https://arxiv.org/abs/2402.04660) | 该研究提出了一种通过艺术设计实现对抗性鲁棒性的方法，通过微小更改现有规范来抵御对抗性示例的影响。 |
| [^3] | [Detecting LLM-Assisted Writing in Scientific Communication: Are We There Yet?.](http://arxiv.org/abs/2401.16807) | 这项研究评估了四种先进的文本检测器对LLM辅助写作的表现，发现它们的性能不如一个简单的检测器。研究认为需要开发专门用于LLM辅助写作的特定检测器，以解决当前承认实践中的挑战。 |
| [^4] | [Heterogeneous Directed Hypergraph Neural Network over abstract syntax tree (AST) for Code Classification.](http://arxiv.org/abs/2305.04228) | 本研究提出了使用异构有向超图表示AST，并使用异构有向超图神经网络处理图形进行代码分类，超过了现有方法。 |

# 详细

[^1]: 朝向高效的风险敏感策略梯度：一个迭代复杂度分析

    Towards Efficient Risk-Sensitive Policy Gradient: An Iteration Complexity Analysis

    [https://arxiv.org/abs/2403.08955](https://arxiv.org/abs/2403.08955)

    本文对风险敏感策略梯度方法进行了迭代复杂度分析，发现其能够通过使用指数效用函数达到较低的迭代复杂度。

    

    强化学习在各种应用中表现出色，使得自主智能体能够通过与环境的互动学习最佳策略。然而，传统的强化学习框架在迭代复杂度和鲁棒性方面经常面临挑战。风险敏感强化学习平衡了期望回报和风险，具有产生概率鲁棒策略的潜力，但其迭代复杂度分析尚未得到充分探讨。在本研究中，我们针对风险敏感策略梯度方法进行了彻底的迭代复杂度分析，重点关注REINFORCE算法并采用指数效用函数。我们获得了一个$\mathcal{O}(\epsilon^{-2})$的迭代复杂度，以达到$\epsilon$-近似的一阶稳定点（FOSP）。我们研究了风险敏感算法是否可以比风险中性算法实现更好的迭代复杂度。

    arXiv:2403.08955v1 Announce Type: cross  Abstract: Reinforcement Learning (RL) has shown exceptional performance across various applications, enabling autonomous agents to learn optimal policies through interaction with their environments. However, traditional RL frameworks often face challenges in terms of iteration complexity and robustness. Risk-sensitive RL, which balances expected return and risk, has been explored for its potential to yield probabilistically robust policies, yet its iteration complexity analysis remains underexplored. In this study, we conduct a thorough iteration complexity analysis for the risk-sensitive policy gradient method, focusing on the REINFORCE algorithm and employing the exponential utility function. We obtain an iteration complexity of $\mathcal{O}(\epsilon^{-2})$ to reach an $\epsilon$-approximate first-order stationary point (FOSP). We investigate whether risk-sensitive algorithms can achieve better iteration complexity compared to their risk-neutr
    
[^2]: 通过艺术设计提高对抗性鲁棒性

    Adversarial Robustness Through Artifact Design

    [https://arxiv.org/abs/2402.04660](https://arxiv.org/abs/2402.04660)

    该研究提出了一种通过艺术设计实现对抗性鲁棒性的方法，通过微小更改现有规范来抵御对抗性示例的影响。

    

    对抗性示例的出现给机器学习带来了挑战。为了阻碍对抗性示例，大多数防御方法都改变了模型的训练方式（如对抗性训练）或推理过程（如随机平滑）。尽管这些方法显著提高了模型的对抗性鲁棒性，但模型仍然极易受到对抗性示例的影响。在某些领域如交通标志识别中，我们发现对象是按照规范来设计（如标志规范）。为了改善对抗性鲁棒性，我们提出了一种新颖的方法。具体来说，我们提供了一种重新定义规范的方法，对现有规范进行微小的更改，以防御对抗性示例。我们将艺术设计问题建模为一个鲁棒优化问题，并提出了基于梯度和贪婪搜索的方法来解决它。我们在交通标志识别领域对我们的方法进行了评估，使其能够改变交通标志中的象形图标（即标志内的符号）。

    Adversarial examples arose as a challenge for machine learning. To hinder them, most defenses alter how models are trained (e.g., adversarial training) or inference is made (e.g., randomized smoothing). Still, while these approaches markedly improve models' adversarial robustness, models remain highly susceptible to adversarial examples. Identifying that, in certain domains such as traffic-sign recognition, objects are implemented per standards specifying how artifacts (e.g., signs) should be designed, we propose a novel approach for improving adversarial robustness. Specifically, we offer a method to redefine standards, making minor changes to existing ones, to defend against adversarial examples. We formulate the problem of artifact design as a robust optimization problem, and propose gradient-based and greedy search methods to solve it. We evaluated our approach in the domain of traffic-sign recognition, allowing it to alter traffic-sign pictograms (i.e., symbols within the signs) a
    
[^3]: 在科学交流中检测LLM辅助写作：我们已经到达了吗？

    Detecting LLM-Assisted Writing in Scientific Communication: Are We There Yet?. (arXiv:2401.16807v1 [cs.IR])

    [http://arxiv.org/abs/2401.16807](http://arxiv.org/abs/2401.16807)

    这项研究评估了四种先进的文本检测器对LLM辅助写作的表现，发现它们的性能不如一个简单的检测器。研究认为需要开发专门用于LLM辅助写作的特定检测器，以解决当前承认实践中的挑战。

    

    大型语言模型（LLMs），如ChatGPT，在文本生成方面产生了重大影响，尤其是在写作辅助领域。尽管伦理考虑强调了在科学交流中透明地承认LLM的使用的重要性，但真实的承认仍然很少见。鼓励准确承认LLM辅助写作的一个潜在途径涉及使用自动检测器。我们对四个前沿的LLM生成文本检测器进行了评估，发现它们的性能不如一个简单的临时检测器，该检测器设计用于识别在LLM大量出现时的突然写作风格变化。我们认为，开发专门用于LLM辅助写作检测的专用检测器是必要的。这样的检测器可以在促进对LLM参与科学交流的更真实认可、解决当前承认实践中的挑战方面发挥关键作用。

    Large Language Models (LLMs), exemplified by ChatGPT, have significantly reshaped text generation, particularly in the realm of writing assistance. While ethical considerations underscore the importance of transparently acknowledging LLM use, especially in scientific communication, genuine acknowledgment remains infrequent. A potential avenue to encourage accurate acknowledging of LLM-assisted writing involves employing automated detectors. Our evaluation of four cutting-edge LLM-generated text detectors reveals their suboptimal performance compared to a simple ad-hoc detector designed to identify abrupt writing style changes around the time of LLM proliferation. We contend that the development of specialized detectors exclusively dedicated to LLM-assisted writing detection is necessary. Such detectors could play a crucial role in fostering more authentic recognition of LLM involvement in scientific communication, addressing the current challenges in acknowledgment practices.
    
[^4]: 基于抽象语法树的异构有向超图神经网络用于代码分类

    Heterogeneous Directed Hypergraph Neural Network over abstract syntax tree (AST) for Code Classification. (arXiv:2305.04228v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2305.04228](http://arxiv.org/abs/2305.04228)

    本研究提出了使用异构有向超图表示AST，并使用异构有向超图神经网络处理图形进行代码分类，超过了现有方法。

    

    代码分类是程序理解和自动编码中的一个难题。由于程序的模糊语法和复杂语义，大多数现有研究使用基于抽象语法树（AST）和图神经网络（GNN）的技术创建代码表示用于代码分类。这些技术利用代码的结构和语义信息，但只考虑节点之间的成对关系，忽略了AST中节点之间已经存在的高阶相关性，可能导致代码结构信息的丢失。本研究提出使用异构有向超图（HDHG）表示AST，并使用异构有向超图神经网络（HDHGN）处理图形。HDHG保留了节点之间的高阶相关性，并更全面地编码了AST的语义和结构信息。HDHGN通过聚合不同节点的特征并使用不同的函数对其进行处理来对AST进行建模。在四个数据集上的实验表明，HDHG和HDHGN在代码分类任务中超越了现有方法。

    Code classification is a difficult issue in program understanding and automatic coding. Due to the elusive syntax and complicated semantics in programs, most existing studies use techniques based on abstract syntax tree (AST) and graph neural network (GNN) to create code representations for code classification. These techniques utilize the structure and semantic information of the code, but they only take into account pairwise associations and neglect the high-order correlations that already exist between nodes in the AST, which may result in the loss of code structural information. On the other hand, while a general hypergraph can encode high-order data correlations, it is homogeneous and undirected which will result in a lack of semantic and structural information such as node types, edge types, and directions between child nodes and parent nodes when modeling AST. In this study, we propose to represent AST as a heterogeneous directed hypergraph (HDHG) and process the graph by hetero
    

