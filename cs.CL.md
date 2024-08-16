# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Much are LLMs Contaminated? A Comprehensive Survey and the LLMSanitize Library](https://arxiv.org/abs/2404.00699) | LLM受到污染可能导致其性能不可靠，挑战了自然语言处理领域的整体进展。 |
| [^2] | [A Unified Taxonomy-Guided Instruction Tuning Framework for Entity Set Expansion and Taxonomy Expansion](https://arxiv.org/abs/2402.13405) | 通过统一的基于分类学指导的指导调整框架，本文提出了一种利用现有分类学进行实体关系微调的方法，有效解决实体集扩展、分类学扩展和种子引导分类学构建三个任务。 |
| [^3] | [Direct Large Language Model Alignment Through Self-Rewarding Contrastive Prompt Distillation](https://arxiv.org/abs/2402.11907) | 通过对比提示自我奖励方法，提出了一种直接对齐大型语言模型的自动对齐方法，无需依赖人工注释的偏好数据，在实验中表现优于现有方法RLHF。 |
| [^4] | [Can Large Language Models Replace Economic Choice Prediction Labs?](https://arxiv.org/abs/2401.17435) | 该论文研究大型语言模型是否能够取代经济实验室进行选择预测，并通过相关实验证明了其可行性。 |
| [^5] | [CoTFormer: More Tokens With Attention Make Up For Less Depth.](http://arxiv.org/abs/2310.10845) | CoTFormer是一种transformer变体，通过使用隐含的链思考机制，实现了与更深模型相当的容量，并且在实证中显著优于更大的标准transformers。 |

# 详细

[^1]: LLM受到多少污染？一项全面调查和LLMSanitize库

    How Much are LLMs Contaminated? A Comprehensive Survey and the LLMSanitize Library

    [https://arxiv.org/abs/2404.00699](https://arxiv.org/abs/2404.00699)

    LLM受到污染可能导致其性能不可靠，挑战了自然语言处理领域的整体进展。

    

    随着近年来大型语言模型（LLMs）的崛起，新的机会正在出现，但也带来了新的挑战，污染问题迅速变得至关重要。企业应用和人工智能筹款已经达到一定规模，流行的问答基准提高几个百分点可能意味着数百万美元，对模型的完整性施加了巨大压力。同时，追踪LLMs见过的数据变得越来越困难；对于像GPT-4和Claude-3这样的闭源模型，他们不透露任何有关训练集的信息。因此，污染成为一个关键问题：LLMs的性能可能不再可靠，因为其高性能至少部分归因于其先前接触到的数据。这种局限性危及了自然语言处理领域的整体进展，然而，如何有效解决这一问题仍然缺乏方法。

    arXiv:2404.00699v1 Announce Type: new  Abstract: With the rise of Large Language Models (LLMs) in recent years, new opportunities are emerging, but also new challenges, and contamination is quickly becoming critical. Business applications and fundraising in AI have reached a scale at which a few percentage points gained on popular question-answering benchmarks could translate into dozens of millions of dollars, placing high pressure on model integrity. At the same time, it is becoming harder and harder to keep track of the data that LLMs have seen; if not impossible with closed-source models like GPT-4 and Claude-3 not divulging any information on the training set. As a result, contamination becomes a critical issue: LLMs' performance may not be reliable anymore, as the high performance may be at least partly due to their previous exposure to the data. This limitation jeopardizes the entire progress in the field of NLP, yet, there remains a lack of methods on how to efficiently address
    
[^2]: 一个统一的基于分类学指导的实体集扩展和分类学扩展的指导调整框架

    A Unified Taxonomy-Guided Instruction Tuning Framework for Entity Set Expansion and Taxonomy Expansion

    [https://arxiv.org/abs/2402.13405](https://arxiv.org/abs/2402.13405)

    通过统一的基于分类学指导的指导调整框架，本文提出了一种利用现有分类学进行实体关系微调的方法，有效解决实体集扩展、分类学扩展和种子引导分类学构建三个任务。

    

    实体集扩展、分类学扩展和种子引导分类学构建是三个代表性任务，可以用来自动向现有分类学填充新实体。然而，先前的方法通常使用异质技术分别解决这些任务，缺乏统一的视角。为了解决这个问题，在本文中，我们从分类学结构的视角确认了这些任务所需的共同关键技能——找到“兄弟”和找到“父母”，并提出了一个统一的基于分类学指导的指导调整框架来共同解决这三个任务。具体来说，通过利用现有分类学作为丰富的实体关系源，我们利用指导调整来微调大型语言模型，生成父母和兄弟实体。在多个基准数据集上的大量实验证明了TaxoInstruct的有效性，该方法在各项指标上均优于特定任务的基线方法。

    arXiv:2402.13405v1 Announce Type: new  Abstract: Entity Set Expansion, Taxonomy Expansion, and Seed-Guided Taxonomy Construction are three representative tasks that can be used to automatically populate an existing taxonomy with new entities. However, previous approaches often address these tasks separately with heterogeneous techniques, lacking a unified perspective. To tackle this issue, in this paper, we identify the common key skills needed for these tasks from the view of taxonomy structures -- finding 'siblings' and finding 'parents' -- and propose a unified taxonomy-guided instruction tuning framework to jointly solve the three tasks. To be specific, by leveraging the existing taxonomy as a rich source of entity relationships, we utilize instruction tuning to fine-tune a large language model to generate parent and sibling entities. Extensive experiments on multiple benchmark datasets demonstrate the effectiveness of TaxoInstruct, which outperforms task-specific baselines across 
    
[^3]: 通过自我奖励对比提示精炼直接对齐大型语言模型

    Direct Large Language Model Alignment Through Self-Rewarding Contrastive Prompt Distillation

    [https://arxiv.org/abs/2402.11907](https://arxiv.org/abs/2402.11907)

    通过对比提示自我奖励方法，提出了一种直接对齐大型语言模型的自动对齐方法，无需依赖人工注释的偏好数据，在实验中表现优于现有方法RLHF。

    

    在这篇论文中，我们提出了一种方法，通过使用对比提示对响应对的输出概率进行评估，从而在LLaMA2-7B和LLaMA2-13B上实现了比RLAIF更好的性能。基于此，我们提出了一种自动对齐方法，即直接大型模型对齐（DLMA）。首先，我们使用对比提示对自动生成的偏好数据。然后，我们继续使用对比提示对生成的偏好数据进行评估并计算自我奖励分数。最后，我们使用DPO算法通过结合这种自我奖励分数来有效地对齐LLMs。在实验阶段，我们的DLMA方法能够在不依赖人工注释的偏好数据的情况下超越RLHF方法。

    arXiv:2402.11907v1 Announce Type: new  Abstract: Aligning large language models (LLMs) with human expectations without human-annotated preference data is an important problem. In this paper, we propose a method to evaluate the response preference by using the output probabilities of response pairs under contrastive prompt pairs, which could achieve better performance on LLaMA2-7B and LLaMA2-13B compared to RLAIF. Based on this, we propose an automatic alignment method, Direct Large Model Alignment (DLMA). First, we use contrastive prompt pairs to automatically generate preference data. Then, we continue to evaluate the generated preference data using contrastive prompt pairs and calculate a self-rewarding score. Finally, we use the DPO algorithm to effectively align LLMs by combining this self-rewarding score. In the experimental stage, our DLMA method could surpass the \texttt{RLHF} method without relying on human-annotated preference data.
    
[^4]: 大型语言模型能否取代经济选择预测实验室？

    Can Large Language Models Replace Economic Choice Prediction Labs?

    [https://arxiv.org/abs/2401.17435](https://arxiv.org/abs/2401.17435)

    该论文研究大型语言模型是否能够取代经济实验室进行选择预测，并通过相关实验证明了其可行性。

    

    经济选择预测是一项具有挑战性的重要任务，往往受限于获取人类选择数据的困难。实验经济学研究在很大程度上专注于简单的选择环境。最近，人工智能界以两种方式为该努力做出了贡献：考虑大型语言模型是否可以代替人类在上述简单选择预测环境中，以及通过机器学习视角研究更复杂但仍严格的实验经济学环境，包括不完全信息、重复博弈和基于自然语言交流的说服游戏。这引发了一个重要的灵感：大型语言模型是否能够完全模拟经济环境，并生成用于高效人类选择预测的数据，替代复杂的经济实验室研究？我们在这个主题上开创了研究，并展示了其可行性。特别是，我们表明仅在大型语言模型生成的数据上训练的模型可以有效地进行预测。

    Economic choice prediction is an essential challenging task, often constrained by the difficulties in acquiring human choice data. Indeed, experimental economics studies had focused mostly on simple choice settings. The AI community has recently contributed to that effort in two ways: considering whether LLMs can substitute for humans in the above-mentioned simple choice prediction settings, and the study through ML lens of more elaborated but still rigorous experimental economics settings, employing incomplete information, repetitive play, and natural language communication, notably language-based persuasion games. This leaves us with a major inspiration: can LLMs be used to fully simulate the economic environment and generate data for efficient human choice prediction, substituting for the elaborated economic lab studies? We pioneer the study of this subject, demonstrating its feasibility. In particular, we show that a model trained solely on LLM-generated data can effectively predic
    
[^5]: CoTFormer：更多的关注令牌弥补了更少的深度

    CoTFormer: More Tokens With Attention Make Up For Less Depth. (arXiv:2310.10845v1 [cs.CL])

    [http://arxiv.org/abs/2310.10845](http://arxiv.org/abs/2310.10845)

    CoTFormer是一种transformer变体，通过使用隐含的链思考机制，实现了与更深模型相当的容量，并且在实证中显著优于更大的标准transformers。

    

    持续发展越来越大和更深的基础模型的竞赛正在进行中。然而，像链思考（CoT）方法这样的技术在实现最佳下游性能方面仍起着重要作用。在这项工作中，我们建立了使用链思考和使用更深的transformer之间的近似平行关系。基于这一洞见，我们引入了CoTFormer，一种使用隐含链思考机制来实现与更深模型相当容量的transformer变体。我们的实证发现证明了CoTFormer的有效性，因为它们明显优于更大的标准transformers。

    The race to continually develop ever larger and deeper foundational models is underway. However, techniques like the Chain-of-Thought (CoT) method continue to play a pivotal role in achieving optimal downstream performance. In this work, we establish an approximate parallel between using chain-of-thought and employing a deeper transformer. Building on this insight, we introduce CoTFormer, a transformer variant that employs an implicit CoT-like mechanism to achieve capacity comparable to a deeper model. Our empirical findings demonstrate the effectiveness of CoTFormers, as they significantly outperform larger standard transformers.
    

