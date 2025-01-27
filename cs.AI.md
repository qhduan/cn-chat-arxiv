# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NLP Verification: Towards a General Methodology for Certifying Robustness](https://arxiv.org/abs/2403.10144) | 本文尝试总结和评估由该领域迄今进展而形成的NLP验证流程的一般组成部分，贡献在于提出了将句子嵌入连续空间得到的可验证子空间的一般描述。 |
| [^2] | [What's in a Name? Auditing Large Language Models for Race and Gender Bias](https://arxiv.org/abs/2402.14875) | 调查发现，大型语言模型存在种族和性别偏见，尤其对与黑人女性相关的名字表现最不利。审计在模型部署和实施时的重要性得到强调。 |
| [^3] | [Exploring Neuron Interactions and Emergence in LLMs: From the Multifractal Analysis Perspective](https://arxiv.org/abs/2402.09099) | 该论文通过多重分形分析视角，深入研究了LLMs中神经元相互作用和出现现象。通过引入自组织和多重分形分析的概念，研究了神经元相互作用的动态演化过程，尤其关注训练中的复杂行为。通过提出基于神经元的多重分形分析方法，实现了对大型模型中神经元相互作用的定量分析。 |
| [^4] | [How VADER is your AI? Towards a definition of artificial intelligence systems appropriate for regulation](https://arxiv.org/abs/2402.05048) | 本研究提出了一个评分框架，用于衡量人工智能（AI）定义在监管方面的合适性。研究旨在排除通过AI监管提案所采用的不恰当的AI定义对ICT技术、方法和非AI作品的影响，并回归到以前在其他成功技术监管中观察到的原则。 |
| [^5] | [Toward Green and Human-Like Artificial Intelligence: A Complete Survey on Contemporary Few-Shot Learning Approaches](https://arxiv.org/abs/2402.03017) | 本文全面调查了少样本学习领域的最新进展，探讨了该方法在解决深度学习在实际应用中的限制方面的潜力和挑战。 |
| [^6] | [Learning Personalized Decision Support Policies.](http://arxiv.org/abs/2304.06701) | 本文提出了一种学习个性化决策支持策略的算法 $\texttt{THREAD}$，可以为决策者提供不同形式的支持。同时，引入了 $\texttt{Modiste}$ 工具来提供个性化的医学诊断决策支持，使用 $\texttt{THREAD}$ 学习个性化决策支持策略，有效提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。 |
| [^7] | [A Comparative Evaluation of Quantification Methods.](http://arxiv.org/abs/2103.03223) | 本研究通过对24种不同量化方法在超过40个数据集上进行全面实证比较，填补了量化方法比较研究的空白。我们发现在二分类设置中，基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法表现最佳；而在多分类设置中，Generaliz方法表现良好。 |

# 详细

[^1]: NLP验证：走向一种通用的用于认证鲁棒性的方法论

    NLP Verification: Towards a General Methodology for Certifying Robustness

    [https://arxiv.org/abs/2403.10144](https://arxiv.org/abs/2403.10144)

    本文尝试总结和评估由该领域迄今进展而形成的NLP验证流程的一般组成部分，贡献在于提出了将句子嵌入连续空间得到的可验证子空间的一般描述。

    

    深度神经网络在自然语言处理（NLP）领域取得了显著成功，确保它们的安全性和可靠性至关重要：在安全关键的情境中，这些模型必须对变化或攻击具有鲁棒性，并能对其输出给出保证。与计算机视觉不同，NLP缺乏一个统一的验证方法论，尽管近年来文献中取得了一些进展，但对于NLP验证的实用问题常常涉及不深。在本文中，我们尝试提炼和评估一个NLP验证流程的一般组成部分，该流程来源于迄今为止该领域的进展。我们的贡献有两方面：首先，我们给出了将句子嵌入连续空间得到的可验证子空间的一般描述。我们确定了可验证子空间的语义泛化技术挑战，并提出了一种有效处理的方法。

    arXiv:2403.10144v1 Announce Type: cross  Abstract: Deep neural networks have exhibited substantial success in the field of Natural Language Processing (NLP) and ensuring their safety and reliability is crucial: there are safety critical contexts where such models must be robust to variability or attack, and give guarantees over their output. Unlike Computer Vision, NLP lacks a unified verification methodology and, despite recent advancements in literature, they are often light on the pragmatical issues of NLP verification. In this paper, we make an attempt to distil and evaluate general components of an NLP verification pipeline, that emerges from the progress in the field to date. Our contributions are two-fold. Firstly, we give a general characterisation of verifiable subspaces that result from embedding sentences into continuous spaces. We identify, and give an effective method to deal with, the technical challenge of semantic generalisability of verified subspaces; and propose it a
    
[^2]: 名字的含义是什么？审计大型语言模型中的种族和性别偏见

    What's in a Name? Auditing Large Language Models for Race and Gender Bias

    [https://arxiv.org/abs/2402.14875](https://arxiv.org/abs/2402.14875)

    调查发现，大型语言模型存在种族和性别偏见，尤其对与黑人女性相关的名字表现最不利。审计在模型部署和实施时的重要性得到强调。

    

    我们采用审计设计来调查最先进的大型语言模型中的偏见，包括GPT-4。在我们的研究中，我们引发模型在各种情景下为个人提供建议，比如在购车谈判或选举结果预测过程中。我们发现该建议系统性地对与种族少数群体和女性常见相关的名字产生不利影响。与黑人女性相关的名字得到的结果最不利。这些偏见在42个提示模板和多个模型中都是一致的，表明这是一个系统性问题，而不是孤立事件。在提示中提供数值、与决策相关的锚点可以成功抵消偏见，而定性细节的影响并不一致，甚至可能会加剧差异。我们的研究结果强调了在语言模型部署和实施时进行审计的重要性，以减轻其潜在影响。

    arXiv:2402.14875v1 Announce Type: cross  Abstract: We employ an audit design to investigate biases in state-of-the-art large language models, including GPT-4. In our study, we elicit prompt the models for advice regarding an individual across a variety of scenarios, such as during car purchase negotiations or election outcome predictions. We find that the advice systematically disadvantages names that are commonly associated with racial minorities and women. Names associated with Black women receive the least advantageous outcomes. The biases are consistent across 42 prompt templates and several models, indicating a systemic issue rather than isolated incidents. While providing numerical, decision-relevant anchors in the prompt can successfully counteract the biases, qualitative details have inconsistent effects and may even increase disparities. Our findings underscore the importance of conducting audits at the point of LLM deployment and implementation to mitigate their potential for
    
[^3]: 通过多重分形分析视角探索LLMs中的神经元相互作用和出现现象

    Exploring Neuron Interactions and Emergence in LLMs: From the Multifractal Analysis Perspective

    [https://arxiv.org/abs/2402.09099](https://arxiv.org/abs/2402.09099)

    该论文通过多重分形分析视角，深入研究了LLMs中神经元相互作用和出现现象。通过引入自组织和多重分形分析的概念，研究了神经元相互作用的动态演化过程，尤其关注训练中的复杂行为。通过提出基于神经元的多重分形分析方法，实现了对大型模型中神经元相互作用的定量分析。

    

    在以往的大型模型中，关于出现现象的研究主要集中在大型语言模型（LLMs）的功能能力如何随模型规模的扩大而增加。然而，我们的研究超越了这一传统范式，旨在通过不仅仅依赖于模型规模，而更加关注训练过程中神经元相互作用的复杂行为，加深我们对LLMs内部出现现象的理解。通过引入“自组织”和“多重分形分析”概念，我们探索了神经元相互作用在训练过程中如何动态演化，从而导致“出现现象”，这种现象反映了自然系统中简单的微观相互作用如何导致复杂的宏观行为。为了定量分析训练过程中大型模型中神经元之间不断演化的相互作用，我们提出了基于神经元的多重分形分析（NeuroMFA）。利用NeuroMFA，我们进行了一系列的实验

    arXiv:2402.09099v1 Announce Type: new Abstract: Prior studies on the emergence in large models have primarily focused on how the functional capabilities of large language models (LLMs) scale with model size. Our research, however, transcends this traditional paradigm, aiming to deepen our understanding of the emergence within LLMs by placing a special emphasis not just on the model size but more significantly on the complex behavior of neuron interactions during the training process. By introducing the concepts of "self-organization" and "multifractal analysis," we explore how neuron interactions dynamically evolve during training, leading to "emergence," mirroring the phenomenon in natural systems where simple micro-level interactions give rise to complex macro-level behaviors. To quantitatively analyze the continuously evolving interactions among neurons in large models during training, we propose the Neuron-based Multifractal Analysis (NeuroMFA). Utilizing NeuroMFA, we conduct a com
    
[^4]: 你的AI有多么VADER？面向适用于监管的人工智能系统定义的探讨

    How VADER is your AI? Towards a definition of artificial intelligence systems appropriate for regulation

    [https://arxiv.org/abs/2402.05048](https://arxiv.org/abs/2402.05048)

    本研究提出了一个评分框架，用于衡量人工智能（AI）定义在监管方面的合适性。研究旨在排除通过AI监管提案所采用的不恰当的AI定义对ICT技术、方法和非AI作品的影响，并回归到以前在其他成功技术监管中观察到的原则。

    

    人工智能（AI）推动了许多信息和通信技术（ICT）突破。然而，自图灵测试提议以来，ICT系统的范围已远远超出了AI。关键是，最近的AI监管提案采用了影响ICT技术、方法和系统的AI定义，而这些并不是AI。在某些情况下，甚至包括数学、统计和工程领域的作品也会受到影响。令人担忧的是，从西方社会到全球南方，都发现了AI定义上的错误。在本文中，我们提出了一个评分框架，用于衡量AI定义在监管方面的合适性。我们的在线、公开可用的VADER框架评分了应该作为AI定义的前提的覆盖范围，这些定义旨在（i）重现其他成功技术监管中观察到的原则，以及（ii）包括所有的AI技术和方法，同时排除非AI的作品。关于后者，我们的评分基于一种...

    Artificial intelligence (AI) has driven many information and communication technology (ICT) breakthroughs. Nonetheless, the scope of ICT systems has expanded far beyond AI since the Turing test proposal. Critically, recent AI regulation proposals adopt AI definitions affecting ICT techniques, approaches, and systems that are not AI. In some cases, even works from mathematics, statistics, and engineering would be affected. Worryingly, AI misdefinitions are observed from Western societies to the Global South. In this paper, we propose a framework to score how \textit{validated as appropriately-defined for regulation} (VADER) an AI definition is. Our online, publicly-available VADER framework scores the coverage of premises that should underlie AI definitions for regulation, which aim to (i) reproduce principles observed in other successful technology regulations, and (ii) include all AI techniques and approaches while excluding non-AI works. Regarding the latter, our score is based on a 
    
[^5]: 向绿色且类人的人工智能迈进：当代少样本学习方法的全面调查

    Toward Green and Human-Like Artificial Intelligence: A Complete Survey on Contemporary Few-Shot Learning Approaches

    [https://arxiv.org/abs/2402.03017](https://arxiv.org/abs/2402.03017)

    本文全面调查了少样本学习领域的最新进展，探讨了该方法在解决深度学习在实际应用中的限制方面的潜力和挑战。

    

    尽管深度学习取得了广泛的成功，但其对数据的需求和计算的昂贵性使其在许多数据受限的真实应用中不实用。少样本学习（FSL）旨在通过实现对新学习任务的快速适应来解决这些限制，并在近年来取得了显著发展。本调查提供了该领域最新进展的全面概述。首先，正式定义了FSL，并介绍了它与不同学习领域的关系。引入了一种新的分类法，扩展了以前提出的方法，并对经典和新领域中的实际应用进行了描述。最后，讨论了塑造该领域的最新趋势、突出挑战和有前途的未来研究方向。

    Despite deep learning's widespread success, its data-hungry and computationally expensive nature makes it impractical for many data-constrained real-world applications. Few-Shot Learning (FSL) aims to address these limitations by enabling rapid adaptation to novel learning tasks, seeing significant growth in recent years. This survey provides a comprehensive overview of the field's latest advancements. Initially, FSL is formally defined, and its relationship with different learning fields is presented. A novel taxonomy is introduced, extending previously proposed ones, and real-world applications in classic and novel fields are described. Finally, recent trends shaping the field, outstanding challenges, and promising future research directions are discussed.
    
[^6]: 学习个性化决策支持策略

    Learning Personalized Decision Support Policies. (arXiv:2304.06701v1 [cs.LG])

    [http://arxiv.org/abs/2304.06701](http://arxiv.org/abs/2304.06701)

    本文提出了一种学习个性化决策支持策略的算法 $\texttt{THREAD}$，可以为决策者提供不同形式的支持。同时，引入了 $\texttt{Modiste}$ 工具来提供个性化的医学诊断决策支持，使用 $\texttt{THREAD}$ 学习个性化决策支持策略，有效提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。

    

    个体决策者可能需要不同形式的支持来提高决策结果，但重要的问题是，哪种形式的支持会在低成本下导致准确的决策。本文提出了学习决策支持策略的方法，它在给定输入时选择是否以及如何提供支持。我们考虑没有先验信息的决策者，并将学习各自的策略形式化为一个多目标优化问题，这个问题权衡了准确性和成本。使用随机环境的技术，我们提出了 $\texttt{THREAD}$，这是一种个性化决策支持策略的在线算法，并设计了一种超参数调整策略，以利用模拟人类行为来确定成本-性能权衡。我们提供计算实验来证明 $\texttt{THREAD}$ 相对于线下基线的优势。然后，我们推出了一个交互式工具 $\texttt{Modiste}$，它为现实中的医学诊断提供个性化决策支持。$\texttt{Modiste}$ 使用 $\texttt{THREAD}$ 为每位医生学习个性化的决策支持策略，并推荐个性化研究以优化患者的预期结果并将严重并发症的风险降至最低。使用电子健康记录数据，我们展示了 $\texttt{Modiste}$ 显著提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。

    Individual human decision-makers may benefit from different forms of support to improve decision outcomes. However, a key question is which form of support will lead to accurate decisions at a low cost. In this work, we propose learning a decision support policy that, for a given input, chooses which form of support, if any, to provide. We consider decision-makers for whom we have no prior information and formalize learning their respective policies as a multi-objective optimization problem that trades off accuracy and cost. Using techniques from stochastic contextual bandits, we propose $\texttt{THREAD}$, an online algorithm to personalize a decision support policy for each decision-maker, and devise a hyper-parameter tuning strategy to identify a cost-performance trade-off using simulated human behavior. We provide computational experiments to demonstrate the benefits of $\texttt{THREAD}$ compared to offline baselines. We then introduce $\texttt{Modiste}$, an interactive tool that pr
    
[^7]: 量化方法的比较评估

    A Comparative Evaluation of Quantification Methods. (arXiv:2103.03223v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2103.03223](http://arxiv.org/abs/2103.03223)

    本研究通过对24种不同量化方法在超过40个数据集上进行全面实证比较，填补了量化方法比较研究的空白。我们发现在二分类设置中，基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法表现最佳；而在多分类设置中，Generaliz方法表现良好。

    

    量化是指在数据集中预测类别分布的问题。它也代表着一个在监督式机器学习中不断发展的研究领域，近年来提出了大量不同的算法。然而，目前还没有一份全面的实证比较量化方法的研究，以支持算法选择。在本研究中，我们通过对超过40个数据集进行了24种不同量化方法的彻底实证性性能比较，包括二分类和多分类量化设置，填补了这一研究空白。我们观察到没有单一算法能够在所有竞争对手中始终表现最佳，但我们确定了一组在二分类设置中表现最佳的方法，包括基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法。对于多分类设置，我们观察到另一组算法表现良好，包括Generaliz方法。

    Quantification represents the problem of predicting class distributions in a dataset. It also represents a growing research field in supervised machine learning, for which a large variety of different algorithms has been proposed in recent years. However, a comprehensive empirical comparison of quantification methods that supports algorithm selection is not available yet. In this work, we close this research gap by conducting a thorough empirical performance comparison of 24 different quantification methods on overall more than 40 data sets, considering binary as well as multiclass quantification settings. We observe that no single algorithm generally outperforms all competitors, but identify a group of methods including the threshold selection-based Median Sweep and TSMax methods, the DyS framework, and Friedman's method that performs best in the binary setting. For the multiclass setting, we observe that a different group of algorithms yields good performance, including the Generaliz
    

