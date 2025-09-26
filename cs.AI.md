# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can ChatGPT predict article retraction based on Twitter mentions?](https://arxiv.org/abs/2403.16851) | 本研究探讨了ChatGPT是否能够基于Twitter提及来预测文章的撤回，研究发现在预测未来被撤回的有问题文章方面是具有一定潜力的。 |
| [^2] | [A Statistical Framework for Measuring AI Reliance.](http://arxiv.org/abs/2401.15356) | 该论文提出了一个基于统计决策理论的依赖的形式定义，用于衡量人工智能系统的适当依赖。该定义分离了依赖的概念和人类在形成准确信念时面临的挑战，为人类与人工智能互补性和依赖性的研究设计提供了指导。 |
| [^3] | [Levels of AGI: Operationalizing Progress on the Path to AGI.](http://arxiv.org/abs/2311.02462) | 本研究提出了一个框架来对人工通用智能（AGI）模型及其前驱进行分类。该框架引入了不同层次的AGI性能、广泛性和自主性，并提供了一个共同的语言用于比较模型、评估风险，并衡量在AGI路径上的进展。 |

# 详细

[^1]: ChatGPT是否能够基于Twitter提及来预测文章的撤回？

    Can ChatGPT predict article retraction based on Twitter mentions?

    [https://arxiv.org/abs/2403.16851](https://arxiv.org/abs/2403.16851)

    本研究探讨了ChatGPT是否能够基于Twitter提及来预测文章的撤回，研究发现在预测未来被撤回的有问题文章方面是具有一定潜力的。

    

    检测有问题的研究文章具有重要意义，本研究探讨了根据被撤回文章在Twitter上的提及是否能够在文章被撤回前发出信号，从而在预测未来被撤回的有问题文章方面发挥作用。分析了包括3,505篇已撤回文章及其相关Twitter提及在内的数据集，以及使用粗糙精确匹配方法获取的具有类似特征的3,505篇未撤回文章。通过四种预测方法评估了Twitter提及在预测文章撤回方面的有效性，包括手动标注、关键词识别、机器学习模型和ChatGPT。手动标注的结果表明，的确有被撤回的文章，其Twitter提及包含在撤回前发出信号的可识别证据，尽管它们只占所有被撤回文章的一小部分。

    arXiv:2403.16851v1 Announce Type: cross  Abstract: Detecting problematic research articles timely is a vital task. This study explores whether Twitter mentions of retracted articles can signal potential problems with the articles prior to retraction, thereby playing a role in predicting future retraction of problematic articles. A dataset comprising 3,505 retracted articles and their associated Twitter mentions is analyzed, alongside 3,505 non-retracted articles with similar characteristics obtained using the Coarsened Exact Matching method. The effectiveness of Twitter mentions in predicting article retraction is evaluated by four prediction methods, including manual labelling, keyword identification, machine learning models, and ChatGPT. Manual labelling results indicate that there are indeed retracted articles with their Twitter mentions containing recognizable evidence signaling problems before retraction, although they represent only a limited share of all retracted articles with 
    
[^2]: 一个用于衡量人工智能依赖的统计框架

    A Statistical Framework for Measuring AI Reliance. (arXiv:2401.15356v1 [cs.AI])

    [http://arxiv.org/abs/2401.15356](http://arxiv.org/abs/2401.15356)

    该论文提出了一个基于统计决策理论的依赖的形式定义，用于衡量人工智能系统的适当依赖。该定义分离了依赖的概念和人类在形成准确信念时面临的挑战，为人类与人工智能互补性和依赖性的研究设计提供了指导。

    

    人类经常在人工智能系统的帮助下做决策。一个常见模式是人工智能向人类推荐行动，而人类保留对最终决策的控制权。研究人员已经确认，确保人类对人工智能的适当依赖是实现互补性能的关键组成部分。我们认为，目前在这方面的研究中使用的适当依赖的定义缺乏形式化的统计基础，可能会导致矛盾。我们提出了一个基于统计决策理论的依赖的形式定义，它将依赖的概念与人类在区分信号并形成准确信念的挑战分开。我们的定义产生了一个框架，可以用来指导人类与人工智能互补性和依赖性的研究设计和解释。利用最近的人工智能辅助决策研究...

    Humans frequently make decisions with the aid of artificially intelligent (AI) systems. A common pattern is for the AI to recommend an action to the human who retains control over the final decision. Researchers have identified ensuring that a human has appropriate reliance on an AI as a critical component of achieving complementary performance. We argue that the current definition of appropriate reliance used in such research lacks formal statistical grounding and can lead to contradictions. We propose a formal definition of reliance, based on statistical decision theory, which separates the concepts of reliance as the probability the decision-maker follows the AI's prediction from challenges a human may face in differentiating the signals and forming accurate beliefs about the situation. Our definition gives rise to a framework that can be used to guide the design and interpretation of studies on human-AI complementarity and reliance. Using recent AI-advised decision making studies f
    
[^3]: AGI的层次：将AGI路径上的进展可操作化

    Levels of AGI: Operationalizing Progress on the Path to AGI. (arXiv:2311.02462v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2311.02462](http://arxiv.org/abs/2311.02462)

    本研究提出了一个框架来对人工通用智能（AGI）模型及其前驱进行分类。该框架引入了不同层次的AGI性能、广泛性和自主性，并提供了一个共同的语言用于比较模型、评估风险，并衡量在AGI路径上的进展。

    

    我们提出了一个框架，用于对人工通用智能（AGI）模型及其前驱的能力和行为进行分类。该框架引入了AGI性能、广泛性和自主性的层次。我们希望这个框架能够像自动驾驶的层次一样，通过提供一个共同的语言来比较模型、评估风险，并衡量在AGI路径上的进展。为了开发我们的框架，我们分析了现有的AGI定义，并提取出了一个有用的AGI本体论应满足的六个原则。这些原则包括关注能力而不是机制；分别评估广泛性和性能；定义AGI路径上的阶段，而不是专注于终点。基于这些原则，我们提出了“AGI的层次”，根据能力的深度（性能）和广度（广泛性），并思考当前系统如何符合这个本体论。我们讨论了对实现AGI所提出的具有挑战性的要求。

    We propose a framework for classifying the capabilities and behavior of Artificial General Intelligence (AGI) models and their precursors. This framework introduces levels of AGI performance, generality, and autonomy. It is our hope that this framework will be useful in an analogous way to the levels of autonomous driving, by providing a common language to compare models, assess risks, and measure progress along the path to AGI. To develop our framework, we analyze existing definitions of AGI, and distill six principles that a useful ontology for AGI should satisfy. These principles include focusing on capabilities rather than mechanisms; separately evaluating generality and performance; and defining stages along the path toward AGI, rather than focusing on the endpoint. With these principles in mind, we propose 'Levels of AGI' based on depth (performance) and breadth (generality) of capabilities, and reflect on how current systems fit into this ontology. We discuss the challenging req
    

