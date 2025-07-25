# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DualView: Data Attribution from the Dual Perspective](https://arxiv.org/abs/2402.12118) | 提出了DualView，一种基于替代建模的后期数据归因方法，具有高效计算和优质评估结果。 |
| [^2] | [Learning Concepts Definable in First-Order Logic with Counting](https://arxiv.org/abs/1909.03820) | 该研究将一阶逻辑与计数符号相结合，证明了可以在多对数度结构下以次线性时间一致学习可定义的分类器，为包含数值方面的机器学习扩展学习框架迈出了第一步。 |
| [^3] | [Quantifying the Uniqueness of Donald Trump in Presidential Discourse.](http://arxiv.org/abs/2401.01405) | 这项研究使用了一种新的度量标准来量化唐纳德·特朗普在总统演讲中的独特性，并发现了他在使用具有分裂性和对抗性的语言、重复强调等方面与其他总统候选人存在显著差异。此外，特朗普比共和党同僚更具独特性。 |
| [^4] | [I-CEE: Tailoring Explanations of Image Classification Models to User Expertise.](http://arxiv.org/abs/2312.12102) | I-CEE是一个人为中心的框架，为用户专业知识定制了图像分类模型的解释，通过提供信息丰富的示例图像、局部解释和模型决策来帮助用户理解模型的决策。 |
| [^5] | [Eva-KELLM: A New Benchmark for Evaluating Knowledge Editing of LLMs.](http://arxiv.org/abs/2308.09954) | Eva-KELLM是一个新的用于评估LLMs知识编辑的基准，提供了一个评估框架和数据集。该基准通过使用原始文档进行知识编辑和多角度的评估来解决了现有研究中收集成本高、表达复杂事实困难、评估视角受限等问题。 |

# 详细

[^1]: DualView：双重视角下的数据归因

    DualView: Data Attribution from the Dual Perspective

    [https://arxiv.org/abs/2402.12118](https://arxiv.org/abs/2402.12118)

    提出了DualView，一种基于替代建模的后期数据归因方法，具有高效计算和优质评估结果。

    

    本文提出了DualView，这是一种基于替代建模的后期数据归因方法，展示了高计算效率和良好的评估结果。我们专注于神经网络，在与文献相关的适当定量评估策略下评估了我们提出的技术，比较了与相关主要本地数据归因方法的性能。

    arXiv:2402.12118v1 Announce Type: cross  Abstract: Local data attribution (or influence estimation) techniques aim at estimating the impact that individual data points seen during training have on particular predictions of an already trained Machine Learning model during test time. Previous methods either do not perform well consistently across different evaluation criteria from literature, are characterized by a high computational demand, or suffer from both. In this work we present DualView, a novel method for post-hoc data attribution based on surrogate modelling, demonstrating both high computational efficiency, as well as good evaluation results. With a focus on neural networks, we evaluate our proposed technique using suitable quantitative evaluation strategies from the literature against related principal local data attribution methods. We find that DualView requires considerably lower computational resources than other methods, while demonstrating comparable performance to comp
    
[^2]: 用计数符号的一阶逻辑定义的概念的学习

    Learning Concepts Definable in First-Order Logic with Counting

    [https://arxiv.org/abs/1909.03820](https://arxiv.org/abs/1909.03820)

    该研究将一阶逻辑与计数符号相结合，证明了可以在多对数度结构下以次线性时间一致学习可定义的分类器，为包含数值方面的机器学习扩展学习框架迈出了第一步。

    

    我们研究了在Grohe和Tur\'an引入的逻辑框架下的关系背景结构上的布尔分类问题。众所周知(Grohe和Ritzert, LICS 2017)，在多对数度结构上的一阶逻辑可定义的分类器可以在次线性时间内学习，其中结构的度和运行时间是以结构的大小为单位来衡量的。我们将结果推广到了由Kuske和Schweikardt(LICS 2017)引入的带计数的一阶逻辑FOCN，它作为一个广泛推广各种计数逻辑的表现逻辑。具体来说，我们证明了可以在多对数度结构类上定义的FOCN中的分类器可以在次线性时间内一致地学习。这可以看作是将学习框架扩展以包含机器学习的数值方面的第一步。我们将这一结果扩展到了无视的概率

    arXiv:1909.03820v2 Announce Type: replace-cross  Abstract: We study Boolean classification problems over relational background structures in the logical framework introduced by Grohe and Tur\'an (TOCS 2004). It is known (Grohe and Ritzert, LICS 2017) that classifiers definable in first-order logic over structures of polylogarithmic degree can be learned in sublinear time, where the degree of the structure and the running time are measured in terms of the size of the structure. We generalise the results to the first-order logic with counting FOCN, which was introduced by Kuske and Schweikardt (LICS 2017) as an expressive logic generalising various other counting logics. Specifically, we prove that classifiers definable in FOCN over classes of structures of polylogarithmic degree can be consistently learned in sublinear time. This can be seen as a first step towards extending the learning framework to include numerical aspects of machine learning. We extend the result to agnostic probabl
    
[^3]: 量化唐纳德·特朗普在总统演讲中的独特性

    Quantifying the Uniqueness of Donald Trump in Presidential Discourse. (arXiv:2401.01405v1 [cs.CL])

    [http://arxiv.org/abs/2401.01405](http://arxiv.org/abs/2401.01405)

    这项研究使用了一种新的度量标准来量化唐纳德·特朗普在总统演讲中的独特性，并发现了他在使用具有分裂性和对抗性的语言、重复强调等方面与其他总统候选人存在显著差异。此外，特朗普比共和党同僚更具独特性。

    

    唐纳德·特朗普与其他总统在演讲中是否表达出不同的风格？如果是，有哪些方面的不同？这些差异是否局限于任何单一的沟通媒介？为了调查这些问题，本文引入了一种基于大型语言模型的独特性度量标准，开发了一个新的分裂性演讲词库，并提出了比较政治对手词汇特征的框架。将这些工具应用于多种总统演讲语料库，我们发现有相当多的证据表明特朗普的讲话模式与近代历任主要总统候选人不同。一些显著的发现包括特朗普使用特别具有分裂性和对抗性的语言针对他的政治对手，并且他重复强调的模式。此外，特朗普比他的共和党同僚更加独特，而他们的独特性值与民主党相对较接近。这些差异在多种度量方法下保持一致。

    Does Donald Trump speak differently from other presidents? If so, in what ways? Are these differences confined to any single medium of communication? To investigate these questions, this paper introduces a novel metric of uniqueness based on large language models, develops a new lexicon for divisive speech, and presents a framework for comparing the lexical features of political opponents. Applying these tools to a variety of corpora of presidential speeches, we find considerable evidence that Trump's speech patterns diverge from those of all major party nominees for the presidency in recent history. Some notable findings include Trump's employment of particularly divisive and antagonistic language targeting of his political opponents and his patterns of repetition for emphasis. Furthermore, Trump is significantly more distinctive than his fellow Republicans, whose uniqueness values are comparably closer to those of the Democrats. These differences hold across a variety of measurement 
    
[^4]: I-CEE: 将图像分类模型的解释定制为用户专业知识

    I-CEE: Tailoring Explanations of Image Classification Models to User Expertise. (arXiv:2312.12102v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2312.12102](http://arxiv.org/abs/2312.12102)

    I-CEE是一个人为中心的框架，为用户专业知识定制了图像分类模型的解释，通过提供信息丰富的示例图像、局部解释和模型决策来帮助用户理解模型的决策。

    

    有效解释黑盒机器学习模型的决策对于依赖它们的人工智能系统的负责任部署至关重要。识别到其重要性，可以生成这些解释的可解释人工智能（XAI）领域提供了几种技术。然而，在这一不断发展的工作中，对用户（解释对象）的关注相对较少，大多数XAI技术产生的是“一刀切”的解释。为了弥合这一差距，实现更加以人为中心的XAI，我们提出了I-CEE，这是一个为用户专业知识定制图像分类解释的框架。受到现有工作的启发，I-CEE通过为用户提供信息丰富的训练数据子集（即示例图像）、相应的局部解释和模型决策来解释图像分类模型的决策。然而，与此前的工作不同的是，I-CEE模拟了示例图像的信息量依赖于用户专业知识的情况，从而为不同的用户提供不同的示例。

    Effectively explaining decisions of black-box machine learning models is critical to responsible deployment of AI systems that rely on them. Recognizing their importance, the field of explainable AI (XAI) provides several techniques to generate these explanations. Yet, there is relatively little emphasis on the user (the explainee) in this growing body of work and most XAI techniques generate "one-size-fits-all" explanations. To bridge this gap and achieve a step closer towards human-centered XAI, we present I-CEE, a framework that provides Image Classification Explanations tailored to User Expertise. Informed by existing work, I-CEE explains the decisions of image classification models by providing the user with an informative subset of training data (i.e., example images), corresponding local explanations, and model decisions. However, unlike prior work, I-CEE models the informativeness of the example images to depend on user expertise, resulting in different examples for different u
    
[^5]: Eva-KELLM：评估LLMs的知识编辑的新基准

    Eva-KELLM: A New Benchmark for Evaluating Knowledge Editing of LLMs. (arXiv:2308.09954v1 [cs.CL])

    [http://arxiv.org/abs/2308.09954](http://arxiv.org/abs/2308.09954)

    Eva-KELLM是一个新的用于评估LLMs知识编辑的基准，提供了一个评估框架和数据集。该基准通过使用原始文档进行知识编辑和多角度的评估来解决了现有研究中收集成本高、表达复杂事实困难、评估视角受限等问题。

    

    大型语言模型（LLMs）的参数中存储着丰富的知识。然而，这些知识随着时间的推移可能变得过时或不合适。因此，对LLMs进行知识编辑并评估其效果引起了越来越多的关注。现有研究主要集中在使用事实三元组进行知识编辑，这不仅在收集上产生高成本，而且在表达复杂事实时也存在困难。此外，这些研究在评估视角上往往受到限制。在本文中，我们提出了Eva-KELLM，用于评估LLMs的知识编辑的新基准。该基准包括一个评估框架和相应的数据集。在我们的框架下，我们首先要求LLM使用原始文档进行知识编辑，与使用事实三元组相比，这提供了一种更方便、更通用的方法。然后我们从多个角度评估更新后的LLM的表现。

    Large language models (LLMs) possess a wealth of knowledge encoded in their parameters. However, this knowledge may become outdated or unsuitable over time. As a result, there has been a growing interest in knowledge editing for LLMs and evaluating its effectiveness. Existing studies primarily focus on knowledge editing using factual triplets, which not only incur high costs for collection but also struggle to express complex facts. Furthermore, these studies are often limited in their evaluation perspectives. In this paper, we propose Eva-KELLM, a new benchmark for evaluating knowledge editing of LLMs. This benchmark includes an evaluation framework and a corresponding dataset. Under our framework, we first ask the LLM to perform knowledge editing using raw documents, which provides a more convenient and universal approach compared to using factual triplets. We then evaluate the updated LLM from multiple perspectives. In addition to assessing the effectiveness of knowledge editing and
    

