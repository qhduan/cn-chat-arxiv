# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CheckEval: Robust Evaluation Framework using Large Language Model via Checklist](https://arxiv.org/abs/2403.18771) | CheckEval是一种使用大型语言模型构建的评估框架，通过详细的子方面和布尔问题清单简化了评估过程，增强了评估结果的稳健性和可靠性，通过SummEval基准验证其有效性。 |
| [^2] | [MaiBaam Annotation Guidelines](https://arxiv.org/abs/2403.05902) | 该论文提供了MaiBaam语料库的注释准则，详细介绍了如何处理和标记巴伐利亚数据，说明了词性标记和依赖关系的使用，以及对德语等相关语言适用的注释决策和对巴伐利亚语法特定决策的介绍和推动。 |
| [^3] | [Complex QA and language models hybrid architectures, Survey.](http://arxiv.org/abs/2302.09051) | 本文综述了语言模型架构和策略的最新进展，并重点关注混合技术在复杂问题回答中的应用，讨论了该领域的挑战和未来研究方向。 |

# 详细

[^1]: CheckEval: 使用大型语言模型通过清单构建健壮评估框架

    CheckEval: Robust Evaluation Framework using Large Language Model via Checklist

    [https://arxiv.org/abs/2403.18771](https://arxiv.org/abs/2403.18771)

    CheckEval是一种使用大型语言模型构建的评估框架，通过详细的子方面和布尔问题清单简化了评估过程，增强了评估结果的稳健性和可靠性，通过SummEval基准验证其有效性。

    

    我们介绍了CheckEval，一种使用大型语言模型的新型评估框架，解决了当前评估方法中的歧义和不一致性挑战。CheckEval通过将评估标准分解为详细的子方面，并为每个构建布尔问题清单，简化了评估过程。这种方法不仅使过程更具可解释性，还通过专注于特定的评估维度显着增强了结果的稳健性和可靠性。通过使用SummEval基准进行的专注案例研究验证，CheckEval显示出与人类判断的强相关性。此外，它展示了高度一致的互注者一致性。这些发现突显了CheckEval在客观、灵活和精确评估方面的有效性。通过提供可定制和互动的框架，CheckEval为LL的使用设立了新的标准。

    arXiv:2403.18771v1 Announce Type: new  Abstract: We introduce CheckEval, a novel evaluation framework using Large Language Models, addressing the challenges of ambiguity and inconsistency in current evaluation methods. CheckEval addresses these challenges by dividing evaluation criteria into detailed sub-aspects and constructing a checklist of Boolean questions for each, simplifying the evaluation. This approach not only renders the process more interpretable but also significantly enhances the robustness and reliability of results by focusing on specific evaluation dimensions. Validated through a focused case study using the SummEval benchmark, CheckEval indicates a strong correlation with human judgments. Furthermore, it demonstrates a highly consistent Inter-Annotator Agreement. These findings highlight the effectiveness of CheckEval for objective, flexible, and precise evaluations. By offering a customizable and interactive framework, CheckEval sets a new standard for the use of LL
    
[^2]: MaiBaam注释准则

    MaiBaam Annotation Guidelines

    [https://arxiv.org/abs/2403.05902](https://arxiv.org/abs/2403.05902)

    该论文提供了MaiBaam语料库的注释准则，详细介绍了如何处理和标记巴伐利亚数据，说明了词性标记和依赖关系的使用，以及对德语等相关语言适用的注释决策和对巴伐利亚语法特定决策的介绍和推动。

    

    本文提供了MaiBaam的注释准则，这是一个注释了词性标记和句法依赖关系的巴伐利亚语语料库。MaiBaam属于通用依存关系项目（UD），我们的注释详细说明了一般和德国UD第2版指南。在本文中，我们详细介绍了如何预处理和标记巴伐利亚数据，概述了我们使用的词性标记和依赖关系，解释了也适用于德语等密切相关语言的注释决策，最后介绍并推动了适用于巴伐利亚语法的决策。

    arXiv:2403.05902v1 Announce Type: new  Abstract: This document provides the annotation guidelines for MaiBaam, a Bavarian corpus annotated with part-of-speech (POS) tags and syntactic dependencies. MaiBaam belongs to the Universal Dependencies (UD) project, and our annotations elaborate on the general and German UD version 2 guidelines. In this document, we detail how to preprocess and tokenize Bavarian data, provide an overview of the POS tags and dependencies we use, explain annotation decisions that would also apply to closely related languages like German, and lastly we introduce and motivate decisions that are specific to Bavarian grammar.
    
[^3]: 复杂问答和语言模型混合架构综述

    Complex QA and language models hybrid architectures, Survey. (arXiv:2302.09051v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.09051](http://arxiv.org/abs/2302.09051)

    本文综述了语言模型架构和策略的最新进展，并重点关注混合技术在复杂问题回答中的应用，讨论了该领域的挑战和未来研究方向。

    

    本文回顾了语言模型架构和策略的最新进展，重点关注混合技术在复杂问题回答中的应用。大型语言模型能够在标准问题上利用公共数据，但在解决更具体的复杂问题时（如在不同文化中个人自由概念的变化如何？什么是为减少气候变化而实现的最佳发电方法组合？），需要特定的架构、知识、技能、方法、敏感数据保护、可解释性、人类审批和多功能反馈。最近的项目如ChatGPT和GALACTICA允许非专业人员了解LLM在复杂QA中的巨大潜力以及同等强大的局限性。在本文中，我们首先审查所需的技能和评估技术。然后，我们综述了现有的混合架构，将LLM与基于规则的方法、信息检索、知识图谱和其他AI/ML技术相结合。最后，我们指出这些CQA系统的挑战，并提出未来研究的可能方向。

    This paper reviews the state-of-the-art of language models architectures and strategies for "complex" question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and an
    

