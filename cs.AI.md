# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [KTbench: A Novel Data Leakage-Free Framework for Knowledge Tracing](https://arxiv.org/abs/2403.15304) | KTbench提出了一种无数据泄漏的知识追踪框架，解决了KT模型中KC之间的相关性学习可能导致的性能下降问题。 |
| [^2] | [Towards Efficient Risk-Sensitive Policy Gradient: An Iteration Complexity Analysis](https://arxiv.org/abs/2403.08955) | 本文对风险敏感策略梯度方法进行了迭代复杂度分析，发现其能够通过使用指数效用函数达到较低的迭代复杂度。 |
| [^3] | [Large Language Models are In-Context Molecule Learners](https://arxiv.org/abs/2403.04197) | 提出了上下文分子适应（ICMA）范式，允许LLMs通过上下文示例学习分子-文本对齐，解决了在分子-标题翻译任务中对LLMs的挑战。 |
| [^4] | [Purpose for Open-Ended Learning Robots: A Computational Taxonomy, Definition, and Operationalisation](https://arxiv.org/abs/2403.02514) | 提出了设定机器人目的的概念，以帮助机器人更加关注获取与目的相关的知识。 |
| [^5] | [Emojis Decoded: Leveraging ChatGPT for Enhanced Understanding in Social Media Communications](https://arxiv.org/abs/2402.01681) | 在表情符号研究中，我们评估了ChatGPT在处理注释和下游任务中的有效性。我们的研究结果表明ChatGPT可以作为一个可行的替代人类注释者的工具，有效地解释表情符号。 |
| [^6] | [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377) | 这项研究展示了n-gram语言模型的价值，并介绍了一个名为infini-gram的引擎，它可以以毫秒级的延迟计算任意n的n-gram概率，使得在神经大型语言模型中对文本进行更准确的分析成为可能。 |
| [^7] | [SoftMAC: Differentiable Soft Body Simulation with Forecast-based Contact Model and Two-way Coupling with Articulated Rigid Bodies and Clothes](https://arxiv.org/abs/2312.03297) | SoftMAC提出了一个不同于以往的可微仿真框架，能够将软体、关节刚体和衣物耦合在一起，并采用基于预测的接触模型和穿透追踪算法，有效地减少了穿透现象。 |
| [^8] | [Understanding deep neural networks through the lens of their non-linearity.](http://arxiv.org/abs/2310.11439) | 本文提出了一个理论上有效的解决方案，通过亲和度评分追踪深度神经网络中的非线性传播，尤其关注计算机视觉应用。实验证实了所提出方法的实用性和对广泛应用的潜力。 |
| [^9] | [Online POMDP Planning with Anytime Deterministic Guarantees.](http://arxiv.org/abs/2310.01791) | 本文中，我们推导出在线POMDP规划中一个简化解决方案与理论上最优解之间的确定性关系，以解决目前近似算法只能提供概率性和通常呈现渐进性保证的限制。 |
| [^10] | [Unveiling the frontiers of deep learning: innovations shaping diverse domains.](http://arxiv.org/abs/2309.02712) | 本文广泛研究了深度学习在各个主要研究领域中的潜在应用，揭示了其准确性和计算能力的优势，以及相关的挑战。 |
| [^11] | [The detection and rectification for identity-switch based on unfalsified control.](http://arxiv.org/abs/2307.14591) | 本文提出了一种基于未被伪造的控制的多目标跟踪方法，针对身份交换问题设计了检测和修正模块，以及解决外观信息模糊匹配的策略，并在实验中展示了其出色的效果和鲁棒性。 |
| [^12] | [InceptionNeXt: When Inception Meets ConvNeXt.](http://arxiv.org/abs/2303.16900) | 本论文提出了一种名为InceptionNeXt的新型神经网络，通过将大内核卷积沿通道维度分解为四个平行分支来提高模型效率，解决了保持性能的同时加快基于大内核的CNN模型的问题。 |

# 详细

[^1]: KTbench：一种全新的无数据泄漏的知识追踪框架

    KTbench: A Novel Data Leakage-Free Framework for Knowledge Tracing

    [https://arxiv.org/abs/2403.15304](https://arxiv.org/abs/2403.15304)

    KTbench提出了一种无数据泄漏的知识追踪框架，解决了KT模型中KC之间的相关性学习可能导致的性能下降问题。

    

    知识追踪（KT）涉及在智能辅导系统中预测学生对学习项目的未来表现。学习项目被标记为称为知识概念（KCs）的技能标签。许多KT模型通过用构成KC的学习项目取代学习项目来将学习项目-学生交互序列扩展为KC-学生交互序列，从而解决了稀疏的学习项目-学生交互问题并最小化了模型参数。然而，这种方法存在两个问题。第一个问题是模型学习同一项目内的KC之间的相关性的能力，这可能导致基本事实标签的泄漏并阻碍模型性能。第二个问题是现有的基准实现忽略了计数问题

    arXiv:2403.15304v1 Announce Type: cross  Abstract: Knowledge Tracing (KT) is concerned with predicting students' future performance on learning items in intelligent tutoring systems. Learning items are tagged with skill labels called knowledge concepts (KCs). Many KT models expand the sequence of item-student interactions into KC-student interactions by replacing learning items with their constituting KCs. This often results in a longer sequence length. This approach addresses the issue of sparse item-student interactions and minimises model parameters. However, two problems have been identified with such models.   The first problem is the model's ability to learn correlations between KCs belonging to the same item, which can result in the leakage of ground truth labels and hinder performance. This problem can lead to a significant decrease in performance on datasets with a higher number of KCs per item. The second problem is that the available benchmark implementations ignore accounti
    
[^2]: 朝向高效的风险敏感策略梯度：一个迭代复杂度分析

    Towards Efficient Risk-Sensitive Policy Gradient: An Iteration Complexity Analysis

    [https://arxiv.org/abs/2403.08955](https://arxiv.org/abs/2403.08955)

    本文对风险敏感策略梯度方法进行了迭代复杂度分析，发现其能够通过使用指数效用函数达到较低的迭代复杂度。

    

    强化学习在各种应用中表现出色，使得自主智能体能够通过与环境的互动学习最佳策略。然而，传统的强化学习框架在迭代复杂度和鲁棒性方面经常面临挑战。风险敏感强化学习平衡了期望回报和风险，具有产生概率鲁棒策略的潜力，但其迭代复杂度分析尚未得到充分探讨。在本研究中，我们针对风险敏感策略梯度方法进行了彻底的迭代复杂度分析，重点关注REINFORCE算法并采用指数效用函数。我们获得了一个$\mathcal{O}(\epsilon^{-2})$的迭代复杂度，以达到$\epsilon$-近似的一阶稳定点（FOSP）。我们研究了风险敏感算法是否可以比风险中性算法实现更好的迭代复杂度。

    arXiv:2403.08955v1 Announce Type: cross  Abstract: Reinforcement Learning (RL) has shown exceptional performance across various applications, enabling autonomous agents to learn optimal policies through interaction with their environments. However, traditional RL frameworks often face challenges in terms of iteration complexity and robustness. Risk-sensitive RL, which balances expected return and risk, has been explored for its potential to yield probabilistically robust policies, yet its iteration complexity analysis remains underexplored. In this study, we conduct a thorough iteration complexity analysis for the risk-sensitive policy gradient method, focusing on the REINFORCE algorithm and employing the exponential utility function. We obtain an iteration complexity of $\mathcal{O}(\epsilon^{-2})$ to reach an $\epsilon$-approximate first-order stationary point (FOSP). We investigate whether risk-sensitive algorithms can achieve better iteration complexity compared to their risk-neutr
    
[^3]: 大规模语言模型是上下文分子学习器

    Large Language Models are In-Context Molecule Learners

    [https://arxiv.org/abs/2403.04197](https://arxiv.org/abs/2403.04197)

    提出了上下文分子适应（ICMA）范式，允许LLMs通过上下文示例学习分子-文本对齐，解决了在分子-标题翻译任务中对LLMs的挑战。

    

    大型语言模型（LLMs）在生物化学任务中表现出色，尤其是分子标题翻译任务，旨在弥合分子和自然语言文本之间的差距。然而，先前在适应LLMs到分子-标题翻译任务中的方法需要额外的领域特定预训练阶段，存在分子和文本空间之间的弱对齐，或对LLMs的规模有严格要求。为了解决这些挑战，我们提出了上下文分子适应（ICMA），作为一种新的范例，允许LLMs通过上下文示例学习分子-文本对齐，通过上下文分子调整。具体而言，ICMA包括以下三个阶段：跨模态检索、检索后排序和上下文分子调整。

    arXiv:2403.04197v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated exceptional performance in biochemical tasks, especially the molecule caption translation task, which aims to bridge the gap between molecules and natural language texts. However, previous methods in adapting LLMs to the molecule-caption translation task required extra domain-specific pre-training stages, suffered weak alignment between molecular and textual spaces, or imposed stringent demands on the scale of LLMs. To resolve the challenges, we propose In-Context Molecule Adaptation (ICMA), as a new paradigm allowing LLMs to learn the molecule-text alignment from context examples via In-Context Molecule Tuning. Specifically, ICMA incorporates the following three stages: Cross-modal Retrieval, Post-retrieval Re-ranking, and In-context Molecule Tuning. Initially, Cross-modal Retrieval utilizes BM25 Caption Retrieval and Molecule Graph Retrieval to retrieve informative context examples. Addi
    
[^4]: 为开放式学习机器人设定目的：一个计算分类、定义和操作化

    Purpose for Open-Ended Learning Robots: A Computational Taxonomy, Definition, and Operationalisation

    [https://arxiv.org/abs/2403.02514](https://arxiv.org/abs/2403.02514)

    提出了设定机器人目的的概念，以帮助机器人更加关注获取与目的相关的知识。

    

    arXiv:2403.02514v1 公告类型: 跨领域 摘要: 自主开放式学习(OEL)机器人能够通过与环境的直接交互累积获取新技能和知识，例如依靠内在动机和自动生成的目标的指导。OEL机器人对应用具有很高的相关性，因为它们可以使用自主获取的知识来完成对人类用户有关的任务。然而，OEL机器人面临一个重要限制：这可能导致获取的知识对完成用户任务并不那么重要。本文分析了这个问题的一个可能解决方案，它围绕“目的”这一新概念展开。目的表示设计者和/或用户希望机器人从中获得什么。机器人应使用目的的内部表征，这里称为“愿望”，来将其开放式探索集中于获取与其完成目的相关的知识。这项工作有助于发展一个共同

    arXiv:2403.02514v1 Announce Type: cross  Abstract: Autonomous open-ended learning (OEL) robots are able to cumulatively acquire new skills and knowledge through direct interaction with the environment, for example relying on the guidance of intrinsic motivations and self-generated goals. OEL robots have a high relevance for applications as they can use the autonomously acquired knowledge to accomplish tasks relevant for their human users. OEL robots, however, encounter an important limitation: this may lead to the acquisition of knowledge that is not so much relevant to accomplish the users' tasks. This work analyses a possible solution to this problem that pivots on the novel concept of `purpose'. Purposes indicate what the designers and/or users want from the robot. The robot should use internal representations of purposes, called here `desires', to focus its open-ended exploration towards the acquisition of knowledge relevant to accomplish them. This work contributes to develop a co
    
[^5]: 表情符号解密：利用ChatGPT提升社交媒体沟通的理解能力

    Emojis Decoded: Leveraging ChatGPT for Enhanced Understanding in Social Media Communications

    [https://arxiv.org/abs/2402.01681](https://arxiv.org/abs/2402.01681)

    在表情符号研究中，我们评估了ChatGPT在处理注释和下游任务中的有效性。我们的研究结果表明ChatGPT可以作为一个可行的替代人类注释者的工具，有效地解释表情符号。

    

    表情符号在社交网络沟通中已经普遍存在，它们承载了超越文字或短语的语义，这引发了学术界对其属性和功能的越来越多的研究兴趣。然而，与表情符号相关的研究和应用面临两个主要挑战。首先，研究者通常依赖众包来注释表情符号，以了解其情感、使用意图和语义含义。其次，用户的主观解释往往会导致对表情符号的误解，并造成沟通障碍。大型语言模型（LLMs）在各种注释任务中取得了显著的成功，ChatGPT在多个领域展示了专业能力。在我们的研究中，我们评估了ChatGPT在处理以前注释和下游任务中的有效性。我们的目标是验证ChatGPT可以在表情符号研究中作为人类注释者的可行替代者，并验证其解释表情符号的能力。

    Emojis, which encapsulate semantics beyond mere words or phrases, have become prevalent in social network communications. This has spurred increasing scholarly interest in exploring their attributes and functionalities. However, emoji-related research and application face two primary challenges. First, researchers typically rely on crowd-sourcing to annotate emojis in order to understand their sentiments, usage intentions, and semantic meanings. Second, subjective interpretations by users can often lead to misunderstandings of emojis and cause the communication barrier. Large Language Models (LLMs) have achieved significant success in various annotation tasks, with ChatGPT demonstrating expertise across multiple domains. In our study, we assess ChatGPT's effectiveness in handling previously annotated and downstream tasks. Our objective is to validate the hypothesis that ChatGPT can serve as a viable alternative to human annotators in emoji research and that its ability to explain emoji
    
[^6]: 无限-gram：将无限n-gram语言模型扩展到万亿标记

    Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens

    [https://arxiv.org/abs/2401.17377](https://arxiv.org/abs/2401.17377)

    这项研究展示了n-gram语言模型的价值，并介绍了一个名为infini-gram的引擎，它可以以毫秒级的延迟计算任意n的n-gram概率，使得在神经大型语言模型中对文本进行更准确的分析成为可能。

    

    在神经大型语言模型（LLM）时代，n-gram语言模型还具有相关性吗？我们的答案是肯定的，并且我们展示了它们在文本分析和改进神经LLM方面的价值。然而，这需要在两个方面对n-gram模型进行现代化。首先，我们将它们与神经LLM相同的数据规模训练- 1.4万亿个标记。这是迄今为止构建的最大的n-gram模型。其次，现有的n-gram模型使用的n很小，这妨碍了它们的性能；相反，我们允许n可以是任意大的，通过引入一个新的无限-gram LM与回退。我们开发了一个名为infini-gram的引擎，它可以通过后缀数组计算无限-gram（以及任意n的n-gram）概率，并且具有毫秒级的延迟，而无需预先计算n-gram计数表（这将非常昂贵）。无限-gram框架和infini-gram引擎使我们能够对人类写作和机器生成的文本进行许多新颖和有意思的分析：我们发现无限-gram LM...

    Are n-gram language models still relevant in this era of neural large language models (LLMs)? Our answer is yes, and we show their values in both text analysis and improving neural LLMs. Yet this necessitates modernizing n-gram models in two aspects. First, we train them at the same data scale as neural LLMs -- 1.4 trillion tokens. This is the largest n-gram model ever built. Second, existing n-gram models use small n which hinders their performance; we instead allow n to be arbitrarily large, by introducing a new $\infty$-gram LM with backoff. Instead of pre-computing n-gram count tables (which would be very expensive), we develop an engine named infini-gram -- powered by suffix arrays -- that can compute $\infty$-gram (as well as n-gram with arbitrary n) probabilities with millisecond-level latency. The $\infty$-gram framework and infini-gram engine enable us to conduct many novel and interesting analyses of human-written and machine-generated text: we find that the $\infty$-gram LM 
    
[^7]: SoftMAC：基于预测接触模型和与关节刚体和衣物双向耦合的可微软体仿真

    SoftMAC: Differentiable Soft Body Simulation with Forecast-based Contact Model and Two-way Coupling with Articulated Rigid Bodies and Clothes

    [https://arxiv.org/abs/2312.03297](https://arxiv.org/abs/2312.03297)

    SoftMAC提出了一个不同于以往的可微仿真框架，能够将软体、关节刚体和衣物耦合在一起，并采用基于预测的接触模型和穿透追踪算法，有效地减少了穿透现象。

    

    可微物理仿真通过基于梯度的优化，显著提高了解决机器人相关问题的效率。为在各种机器人操纵场景中应用可微仿真，一个关键挑战是将各种材料集成到统一框架中。我们提出了SoftMAC，一个可微仿真框架，将软体与关节刚体和衣物耦合在一起。SoftMAC使用基于连续力学的材料点法来模拟软体。我们提出了一种新颖的基于预测的MPM接触模型，有效减少了穿透，而不会引入其他异常现象，如不自然的反弹。为了将MPM粒子与可变形和非体积衣物网格耦合，我们还提出了一种穿透追踪算法，重建局部区域的有符号距离场。

    arXiv:2312.03297v2 Announce Type: replace-cross  Abstract: Differentiable physics simulation provides an avenue to tackle previously intractable challenges through gradient-based optimization, thereby greatly improving the efficiency of solving robotics-related problems. To apply differentiable simulation in diverse robotic manipulation scenarios, a key challenge is to integrate various materials in a unified framework. We present SoftMAC, a differentiable simulation framework that couples soft bodies with articulated rigid bodies and clothes. SoftMAC simulates soft bodies with the continuum-mechanics-based Material Point Method (MPM). We provide a novel forecast-based contact model for MPM, which effectively reduces penetration without introducing other artifacts like unnatural rebound. To couple MPM particles with deformable and non-volumetric clothes meshes, we also propose a penetration tracing algorithm that reconstructs the signed distance field in local area. Diverging from prev
    
[^8]: 通过非线性研究深度神经网络的理解

    Understanding deep neural networks through the lens of their non-linearity. (arXiv:2310.11439v1 [cs.LG])

    [http://arxiv.org/abs/2310.11439](http://arxiv.org/abs/2310.11439)

    本文提出了一个理论上有效的解决方案，通过亲和度评分追踪深度神经网络中的非线性传播，尤其关注计算机视觉应用。实验证实了所提出方法的实用性和对广泛应用的潜力。

    

    深度神经网络(DNN)的显著成功常常归因于它们的高表达能力和近似任意复杂函数的能力。事实上，DNN是高度非线性的模型，其中引入的激活函数在其中起到了重要作用。然而，尽管许多研究通过近似能力的视角研究了DNN的表达能力，但量化DNN或个别激活函数的非线性仍然是一个开放性问题。在本文中，我们提出了第一个在具体关注计算机视觉应用中追踪非线性传播的理论有效解决方案。我们提出的亲和度评分允许我们深入了解各种不同体系结构和学习范式的内部工作原理。我们提供了大量的实验结果，突出了所提出的亲和度评分的实际效用和潜在应用的可能性。

    The remarkable success of deep neural networks (DNN) is often attributed to their high expressive power and their ability to approximate functions of arbitrary complexity. Indeed, DNNs are highly non-linear models, and activation functions introduced into them are largely responsible for this. While many works studied the expressive power of DNNs through the lens of their approximation capabilities, quantifying the non-linearity of DNNs or of individual activation functions remains an open problem. In this paper, we propose the first theoretically sound solution to track non-linearity propagation in deep neural networks with a specific focus on computer vision applications. Our proposed affinity score allows us to gain insights into the inner workings of a wide range of different architectures and learning paradigms. We provide extensive experimental results that highlight the practical utility of the proposed affinity score and its potential for long-reaching applications.
    
[^9]: 具有任意确定性保证的在线POMDP规划

    Online POMDP Planning with Anytime Deterministic Guarantees. (arXiv:2310.01791v1 [cs.AI])

    [http://arxiv.org/abs/2310.01791](http://arxiv.org/abs/2310.01791)

    本文中，我们推导出在线POMDP规划中一个简化解决方案与理论上最优解之间的确定性关系，以解决目前近似算法只能提供概率性和通常呈现渐进性保证的限制。

    

    在现实场景中，自主智能体经常遇到不确定性并基于不完整信息做出决策。在不确定性下的规划可以使用部分可观察的马尔科夫决策过程（POMDP）进行数学建模。然而，寻找POMDP的最优规划在计算上是昂贵的，只有在小规模任务中可行。近年来，近似算法（如树搜索和基于采样的方法）已经成为解决较大问题的先进POMDP求解器。尽管这些算法有效，但它们仅提供概率性和通常呈现渐进性保证，这是由于它们依赖于采样的缘故。为了解决这些限制，我们推导出一个简化解决方案与理论上最优解之间的确定性关系。首先，我们推导出选择一组观测以在计算每个后验节点时分支的边界。

    Autonomous agents operating in real-world scenarios frequently encounter uncertainty and make decisions based on incomplete information. Planning under uncertainty can be mathematically formalized using partially observable Markov decision processes (POMDPs). However, finding an optimal plan for POMDPs can be computationally expensive and is feasible only for small tasks. In recent years, approximate algorithms, such as tree search and sample-based methodologies, have emerged as state-of-the-art POMDP solvers for larger problems. Despite their effectiveness, these algorithms offer only probabilistic and often asymptotic guarantees toward the optimal solution due to their dependence on sampling. To address these limitations, we derive a deterministic relationship between a simplified solution that is easier to obtain and the theoretically optimal one. First, we derive bounds for selecting a subset of the observations to branch from while computing a complete belief at each posterior nod
    
[^10]: 揭示深度学习的前沿：塑造多个领域的创新

    Unveiling the frontiers of deep learning: innovations shaping diverse domains. (arXiv:2309.02712v1 [cs.LG])

    [http://arxiv.org/abs/2309.02712](http://arxiv.org/abs/2309.02712)

    本文广泛研究了深度学习在各个主要研究领域中的潜在应用，揭示了其准确性和计算能力的优势，以及相关的挑战。

    

    深度学习（DL）使得开发能够学习、可视化、优化、改进和预测数据的计算机模型成为可能。近年来，DL已经应用于多个领域，包括音频-视觉数据处理、农业、交通预测、自然语言、生物医学、灾害管理、生物信息学、药物设计、基因组学、人脸识别和生态学。为了探索深度学习的当前状态，有必要研究深度学习在这些学科中的最新发展和应用。然而，文献中缺乏对深度学习在所有潜在领域中的应用的探索。因此，本文广泛调查了深度学习在所有主要研究领域中的潜在应用，以及相关的优势和挑战。正如文献所证明的那样，DL在预测和分析方面表现出准确性，使其成为一种强大的计算工具，并具有表达能力。

    Deep learning (DL) enables the development of computer models that are capable of learning, visualizing, optimizing, refining, and predicting data. In recent years, DL has been applied in a range of fields, including audio-visual data processing, agriculture, transportation prediction, natural language, biomedicine, disaster management, bioinformatics, drug design, genomics, face recognition, and ecology. To explore the current state of deep learning, it is necessary to investigate the latest developments and applications of deep learning in these disciplines. However, the literature is lacking in exploring the applications of deep learning in all potential sectors. This paper thus extensively investigates the potential applications of deep learning across all major fields of study as well as the associated benefits and challenges. As evidenced in the literature, DL exhibits accuracy in prediction and analysis, makes it a powerful computational tool, and has the ability to articulate i
    
[^11]: 基于未被伪造的控制的身份交换检测与修正

    The detection and rectification for identity-switch based on unfalsified control. (arXiv:2307.14591v1 [cs.CV])

    [http://arxiv.org/abs/2307.14591](http://arxiv.org/abs/2307.14591)

    本文提出了一种基于未被伪造的控制的多目标跟踪方法，针对身份交换问题设计了检测和修正模块，以及解决外观信息模糊匹配的策略，并在实验中展示了其出色的效果和鲁棒性。

    

    多目标跟踪的目的是持续跟踪和识别视频中检测到的物体。目前，大多数多目标跟踪方法都是通过建模运动信息并将其与外观信息相结合，来确定和跟踪物体。本文采用了未被伪造的控制方法来解决多目标跟踪中的身份交换问题。我们在跟踪过程中建立了外观信息变化的序列，针对身份交换检测和恢复设计了一个检测和修正模块。我们还提出了一种简单而有效的策略来解决数据关联过程中外观信息模糊匹配的问题。公开可用的多目标跟踪数据集上的实验结果表明，该跟踪器在处理由遮挡和快速运动引起的跟踪错误方面具有出色的效果和鲁棒性。

    The purpose of multi-object tracking (MOT) is to continuously track and identify objects detected in videos. Currently, most methods for multi-object tracking model the motion information and combine it with appearance information to determine and track objects. In this paper, unfalsified control is employed to address the ID-switch problem in multi-object tracking. We establish sequences of appearance information variations for the trajectories during the tracking process and design a detection and rectification module specifically for ID-switch detection and recovery. We also propose a simple and effective strategy to address the issue of ambiguous matching of appearance information during the data association process. Experimental results on publicly available MOT datasets demonstrate that the tracker exhibits excellent effectiveness and robustness in handling tracking errors caused by occlusions and rapid movements.
    
[^12]: InceptionNeXt：当Inception遇到ConvNeXt

    InceptionNeXt: When Inception Meets ConvNeXt. (arXiv:2303.16900v1 [cs.CV])

    [http://arxiv.org/abs/2303.16900](http://arxiv.org/abs/2303.16900)

    本论文提出了一种名为InceptionNeXt的新型神经网络，通过将大内核卷积沿通道维度分解为四个平行分支来提高模型效率，解决了保持性能的同时加快基于大内核的CNN模型的问题。

    

    受ViTs长程建模能力的启发，近期广泛研究和采用了大内核卷积来扩大感受野和提高模型性能，例如ConvNeXt采用了7x7深度卷积。虽然这种深度操作仅消耗少量FLOPs，但由于高内存访问成本，这在功能强大的计算设备上大大损害了模型效率。尽管缩小ConvNeXt的内核大小能提高速度，但会导致性能显着下降。如何在保持性能的同时加快基于大内核的CNN模型仍不清楚。为了解决这个问题，受Inceptions的启发，我们提出将大内核深度卷积沿通道维度分解为四个平行分支，即小方内核、两个正交带内核和一个互补内核。

    Inspired by the long-range modeling ability of ViTs, large-kernel convolutions are widely studied and adopted recently to enlarge the receptive field and improve model performance, like the remarkable work ConvNeXt which employs 7x7 depthwise convolution. Although such depthwise operator only consumes a few FLOPs, it largely harms the model efficiency on powerful computing devices due to the high memory access costs. For example, ConvNeXt-T has similar FLOPs with ResNet-50 but only achieves 60% throughputs when trained on A100 GPUs with full precision. Although reducing the kernel size of ConvNeXt can improve speed, it results in significant performance degradation. It is still unclear how to speed up large-kernel-based CNN models while preserving their performance. To tackle this issue, inspired by Inceptions, we propose to decompose large-kernel depthwise convolution into four parallel branches along channel dimension, i.e. small square kernel, two orthogonal band kernels, and an ide
    

