# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DeAL: Decoding-time Alignment for Large Language Models](https://arxiv.org/abs/2402.06147) | DeAL是一个允许用户自定义奖励函数并实现解码时对齐LLMs的框架。 |
| [^2] | [Survey of Natural Language Processing for Education: Taxonomy, Systematic Review, and Future Trends](https://arxiv.org/abs/2401.07518) | 这篇论文调查了教育领域自然语言处理的最新进展，提出了分类体系，并总结了挑战和未来研究方向。 |
| [^3] | [Turkish Native Language Identification.](http://arxiv.org/abs/2307.14850) | 这项研究首次将母语识别应用于土耳其语,通过分析作者不同语言的写作来预测作者的母语。研究使用了土耳其学习者语料库和三个句法特征来展示其有效性。 |
| [^4] | [SANTA: Separate Strategies for Inaccurate and Incomplete Annotation Noise in Distantly-Supervised Named Entity Recognition.](http://arxiv.org/abs/2305.04076) | 本文提出了一种处理Distantly-Supervised Named Entity Recognition中错误和不完整标注噪声的分离策略，使用不同的模型构建来应对两种类型的噪声。 |

# 详细

[^1]: DeAL：用于大型语言模型的解码时对齐

    DeAL: Decoding-time Alignment for Large Language Models

    [https://arxiv.org/abs/2402.06147](https://arxiv.org/abs/2402.06147)

    DeAL是一个允许用户自定义奖励函数并实现解码时对齐LLMs的框架。

    

    大型语言模型（LLMs）现在期望生成与人类偏好对齐的内容。目前的工作主要集中在模型训练时间对齐上，通过诸如强化学习与人类反馈（RLHF）等技术。然而，目前还不清楚这些方法是否有效地教导模型对齐目标。首先，无法整合多个自定义奖励和依赖模型开发者对通用和静态原则的理解是主要局限。其次，模型训练中的残留差距以及这些方法的可靠性也值得质疑（例如，即使在安全训练后仍然容易被越狱）。为了解决这些问题，我们提出了DeAL，一个允许用户自定义奖励函数并实现解码时对齐LLMs（DeAL）的框架。核心思想在于将解码视为一个启发式引导的搜索过程，并促使使用各种对齐目标。我们的实验以编程约束为例进行了验证。

    Large Language Models (LLMs) are nowadays expected to generate content aligned with human preferences. Current work focuses on alignment at model training time, through techniques such as Reinforcement Learning with Human Feedback (RLHF). However, it is unclear if such methods are an effective choice to teach alignment objectives to the model. First, the inability to incorporate multiple, custom rewards and reliance on a model developer's view of universal and static principles are key limitations. Second, the residual gaps in model training and the reliability of such approaches are also questionable (e.g. susceptibility to jail-breaking even after safety training). To address these, we propose DeAL, a framework that allows the user to customize reward functions and enables Decoding-time Alignment of LLMs (DeAL). At its core, we view decoding as a heuristic-guided search process and facilitate the use of a wide variety of alignment objectives. Our experiments with programmatic constra
    
[^2]: 教育领域自然语言处理的调查：分类体系、系统综述和未来趋势

    Survey of Natural Language Processing for Education: Taxonomy, Systematic Review, and Future Trends

    [https://arxiv.org/abs/2401.07518](https://arxiv.org/abs/2401.07518)

    这篇论文调查了教育领域自然语言处理的最新进展，提出了分类体系，并总结了挑战和未来研究方向。

    

    自然语言处理（NLP）旨在通过计算机科学领域的技术分析文本，应用于医疗保健、商业和教育领域。特别是，在教育领域，NLP已经被应用于教学和学习方面的帮助。本调查研究主要关注解决与教育领域相关的问题，并回顾了NLP的最新进展。具体来说，我们从介绍相关背景开始，然后提出教育领域NLP的分类系统。接着，我们根据上述分类系统说明任务定义、挑战和相应的技术。之后，我们展示了该领域中的一些现有演示，并总结了未来的研究方向。

    Natural Language Processing (NLP) aims to analyze the text via techniques in the computer science field. It serves the applications in healthcare, commerce, and education domains. Particularly, NLP has been applied to the education domain to help teaching and learning. In this survey, we review recent advances in NLP with a focus on solving problems related to the education domain. In detail, we begin with introducing the relevant background. Then, we present the taxonomy of NLP in the education domain. Next, we illustrate the task definition, challenges, and corresponding techniques based on the above taxonomy. After that, we showcase some off-the-shelf demonstrations in this domain and conclude with future directions.
    
[^3]: 在这篇论文中，我们介绍了首次将母语识别（Native Language Identification，NLI）应用于土耳其语的研究。

    Turkish Native Language Identification. (arXiv:2307.14850v1 [cs.CL])

    [http://arxiv.org/abs/2307.14850](http://arxiv.org/abs/2307.14850)

    这项研究首次将母语识别应用于土耳其语,通过分析作者不同语言的写作来预测作者的母语。研究使用了土耳其学习者语料库和三个句法特征来展示其有效性。

    

    在这篇论文中，我们首次将母语识别（NLI）应用于土耳其语。NLI 是通过分析作者不同语言的写作来预测作者的母语。尽管大多数NLI研究都侧重于英语，我们的研究将其范围扩展到土耳其语。我们使用了最近构建的土耳其学习者语料库，并结合了三个句法特征（CFG 产生规则，词性n-gram和函数词）与L2文本，以展示它们在该任务中的有效性。

    In this paper, we present the first application of Native Language Identification (NLI) for the Turkish language. NLI involves predicting the writer's first language by analysing their writing in different languages. While most NLI research has focused on English, our study extends its scope to Turkish. We used the recently constructed Turkish Learner Corpus and employed a combination of three syntactic features (CFG production rules, part-of-speech n-grams and function words) with L2 texts to demonstrate their effectiveness in this task.
    
[^4]: SANTA：Distantly-Supervised Named Entity Recognition中处理错误和不完整标注噪声的分离策略

    SANTA: Separate Strategies for Inaccurate and Incomplete Annotation Noise in Distantly-Supervised Named Entity Recognition. (arXiv:2305.04076v1 [cs.CL])

    [http://arxiv.org/abs/2305.04076](http://arxiv.org/abs/2305.04076)

    本文提出了一种处理Distantly-Supervised Named Entity Recognition中错误和不完整标注噪声的分离策略，使用不同的模型构建来应对两种类型的噪声。

    

    远程监督命名实体识别有效地减轻了监督设置中耗时且昂贵的注释负担，但是无上下文的匹配过程和知识库的有限覆盖引入了不准确和不完整的标注噪音。本研究提出了使用不同的策略来处理两种类型的噪声的SANTA，以解决由不准确和不完整标注带来的挑战。

    Distantly-Supervised Named Entity Recognition effectively alleviates the burden of time-consuming and expensive annotation in the supervised setting. But the context-free matching process and the limited coverage of knowledge bases introduce inaccurate and incomplete annotation noise respectively. Previous studies either considered only incomplete annotation noise or indiscriminately handle two types of noise with the same strategy. In this paper, we argue that the different causes of two types of noise bring up the requirement of different strategies in model architecture. Therefore, we propose the SANTA to handle these two types of noise separately with (1) Memory-smoothed Focal Loss and Entity-aware KNN to relieve the entity ambiguity problem caused by inaccurate annotation, and (2) Boundary Mixup to alleviate decision boundary shifting problem caused by incomplete annotation and a noise-tolerant loss to improve the robustness. Benefiting from our separate tailored strategies, we co
    

