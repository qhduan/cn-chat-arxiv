# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Are Structural Concepts Universal in Transformer Language Models? Towards Interpretable Cross-Lingual Generalization.](http://arxiv.org/abs/2310.12794) | 本文研究了在Transformer语言模型中明确对齐语言之间的概念对应关系的潜力，以强化跨语言泛化能力。研究发现，无论是仅有编码器还是仅有解码器的模型，各语言内的结构概念空间对齐度高。通过基于元学习的方法，可以学习对齐不同语言的概念空间，实现零样本和少样本泛化。 |
| [^2] | [Next Steps for Human-Centered Generative AI: A Technical Perspective.](http://arxiv.org/abs/2306.15774) | 这项研究从技术角度定义和提出了人类中心生成式人工智能(HGAI)的下一步工作，包括与人类价值观对齐、适应人类的意图表达和增强人类在协作工作流中的能力。这个工作的目标是吸引跨学科研究团队对HGAI的新兴想法进行讨论，并保持未来工作景观的整体连贯性。 |
| [^3] | [Probing in Context: Toward Building Robust Classifiers via Probing Large Language Models.](http://arxiv.org/abs/2305.14171) | 本文提出了一种在上下文中的探测方法，用于构建鲁棒的分类器。通过探测上下文化的表示来预测标签，这种方法对指令变化更加鲁棒，并且在多样化的分类任务上表现出竞争力或更好的性能。 |
| [^4] | [Is ChatGPT A Good Keyphrase Generator? A Preliminary Study.](http://arxiv.org/abs/2303.13001) | 本文对ChatGPT作为关键词生成器进行了初步研究，发现其在各个方面的性能表现良好，特别是在多领域关键词生成方面。ChatGPT仍面临生成缺失关键词的挑战。 |
| [^5] | [Dynamic Generation of Grounded Logical Explanations in a Neuro-Symbolic Expert System.](http://arxiv.org/abs/2209.07662) | 该研究提出一种新颖的方法，通过结合神经语言建模、引导生成和半参数密集检索，动态生成基于事实库的人类可解释的证明树，实现科学推理，并展现了强大的性能。 |

# 详细

[^1]: 结构概念在Transformer语言模型中是否具有普适性？走向可解释的跨语言泛化

    Are Structural Concepts Universal in Transformer Language Models? Towards Interpretable Cross-Lingual Generalization. (arXiv:2310.12794v1 [cs.CL])

    [http://arxiv.org/abs/2310.12794](http://arxiv.org/abs/2310.12794)

    本文研究了在Transformer语言模型中明确对齐语言之间的概念对应关系的潜力，以强化跨语言泛化能力。研究发现，无论是仅有编码器还是仅有解码器的模型，各语言内的结构概念空间对齐度高。通过基于元学习的方法，可以学习对齐不同语言的概念空间，实现零样本和少样本泛化。

    

    大型语言模型(LLMs)展示了显著的跨语言泛化能力，即它们通过隐式知识传输在不同语言之间进行转移。然而，这种转移对于所有语言而言并不均衡，特别是对于资源匮乏的语言，这是一个持续存在的挑战。目前尚不清楚我们是否已经达到了隐式跨语言泛化的极限，并且明确的知识传输是否可行。在本文中，我们调查了明确对齐语言之间概念对应关系的潜力，以增强跨语言泛化能力。通过将语法方面作为测试平台，我们对43种语言的分析显示，无论是仅有编码器还是仅有解码器的LLMs，各种语言内的结构概念空间之间存在高度的对准性。然后，我们提出了一种基于元学习的方法来学习对齐不同语言的概念空间，从而便于在概念分类和对齐上进行零样本和少样本泛化。

    Large language models (LLMs) have exhibited considerable cross-lingual generalization abilities, whereby they implicitly transfer knowledge across languages. However, the transfer is not equally successful for all languages, especially for low-resource ones, which poses an ongoing challenge. It is unclear whether we have reached the limits of implicit cross-lingual generalization and if explicit knowledge transfer is viable. In this paper, we investigate the potential for explicitly aligning conceptual correspondence between languages to enhance cross-lingual generalization. Using the syntactic aspect of language as a testbed, our analyses of 43 languages reveal a high degree of alignability among the spaces of structural concepts within each language for both encoder-only and decoder-only LLMs. We then propose a meta-learning-based method to learn to align conceptual spaces of different languages, which facilitates zero-shot and few-shot generalization in concept classification and al
    
[^2]: 人类中心生成式人工智能的下一步：技术视角

    Next Steps for Human-Centered Generative AI: A Technical Perspective. (arXiv:2306.15774v1 [cs.HC])

    [http://arxiv.org/abs/2306.15774](http://arxiv.org/abs/2306.15774)

    这项研究从技术角度定义和提出了人类中心生成式人工智能(HGAI)的下一步工作，包括与人类价值观对齐、适应人类的意图表达和增强人类在协作工作流中的能力。这个工作的目标是吸引跨学科研究团队对HGAI的新兴想法进行讨论，并保持未来工作景观的整体连贯性。

    

    通过反复跨学科讨论，我们从技术角度为人类中心生成式人工智能(HGAI)定义和提出了下一步的工作。我们贡献了一个路线图，概述了生成式人工智能在三个层面上的未来方向：与人类价值观对齐；适应人类的意图表达；增强人类在协作工作流中的能力。该路线图旨在吸引跨学科研究团队对HGAI的新兴想法进行全面的讨论，同时保持未来工作景观的整体连贯性。

    Through iterative, cross-disciplinary discussions, we define and propose next-steps for Human-centered Generative AI (HGAI) from a technical perspective. We contribute a roadmap that lays out future directions of Generative AI spanning three levels: Aligning with human values; Accommodating humans' expression of intents; and Augmenting humans' abilities in a collaborative workflow. This roadmap intends to draw interdisciplinary research teams to a comprehensive list of emergent ideas in HGAI, identifying their interested topics while maintaining a coherent big picture of the future work landscape.
    
[^3]: 在上下文中的探测：通过对大型语言模型的探测构建鲁棒的分类器

    Probing in Context: Toward Building Robust Classifiers via Probing Large Language Models. (arXiv:2305.14171v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14171](http://arxiv.org/abs/2305.14171)

    本文提出了一种在上下文中的探测方法，用于构建鲁棒的分类器。通过探测上下文化的表示来预测标签，这种方法对指令变化更加鲁棒，并且在多样化的分类任务上表现出竞争力或更好的性能。

    

    大型语言模型能够在上下文中学习新任务，在提供指令和少量注释示例的情况下。然而，上下文学习的有效性取决于提供的上下文，并且下游任务的性能可能会有很大变化，取决于指令。重要的是，这种对上下文的依赖可能以不可预测的方式表现，例如，一个看似更有信息量的指令可能导致性能更差。在本文中，我们提出了一种替代方法，称之为上下文中的探测。类似于上下文学习，我们用指令对输入的表示进行上下文化，但是我们不是解码输出预测，而是探测上下文化的表示来预测标签。通过一系列在多样化的分类任务上的实验，我们展示了上下文中的探测对指令变化更加鲁棒。我们进一步展示了探测的性能与其他方法相竞争或更胜一筹。

    Large language models are able to learn new tasks in context, where they are provided with instructions and a few annotated examples. However, the effectiveness of in-context learning is dependent on the provided context, and the performance on a downstream task can vary considerably, depending on the instruction. Importantly, such dependency on the context can surface in unpredictable ways, e.g., a seemingly more informative instruction might lead to a worse performance. In this paper, we propose an alternative approach, which we term in-context probing. Similar to in-context learning, we contextualize the representation of the input with an instruction, but instead of decoding the output prediction, we probe the contextualized representation to predict the label. Through a series of experiments on a diverse set of classification tasks, we show that in-context probing is significantly more robust to changes in instructions. We further show that probing performs competitive or superior
    
[^4]: ChatGPT是一款好的关键词生成器吗？初步研究。

    Is ChatGPT A Good Keyphrase Generator? A Preliminary Study. (arXiv:2303.13001v1 [cs.CL])

    [http://arxiv.org/abs/2303.13001](http://arxiv.org/abs/2303.13001)

    本文对ChatGPT作为关键词生成器进行了初步研究，发现其在各个方面的性能表现良好，特别是在多领域关键词生成方面。ChatGPT仍面临生成缺失关键词的挑战。

    

    ChatGPT的出现引起了计算语言学界的重视。为了展示其作为关键词生成器的能力，我们对ChatGPT进行了初步评估以用于关键词生成任务。我们评估了其在各个方面的性能，包括关键词生成提示，关键词生成多样性，多领域关键词生成和长文本理解。我们的评估基于六个基准数据集，并采用OpenAI建议的提示，并将其扩展为六个候选提示。我们发现ChatGPT在所有六个候选提示上表现出色，在不同数据集之间观察到了轻微的性能差异。基于我们的发现，我们得出结论，ChatGPT有很大的关键词生成潜力。此外，我们发现ChatGPT在生成缺失关键词方面仍面临挑战。最后，在最后一节中，我们还介绍了一些限制和未来的研究方向。

    The emergence of ChatGPT has recently garnered significant attention from the computational linguistics community. To demonstrate its capabilities as a keyphrase generator, we conduct a preliminary evaluation of ChatGPT for the keyphrase generation task. We evaluate its performance in various aspects, including keyphrase generation prompts, keyphrase generation diversity, multi-domain keyphrase generation, and long document understanding. Our evaluation is based on six benchmark datasets, and we adopt the prompt suggested by OpenAI while extending it to six candidate prompts. We find that ChatGPT performs exceptionally well on all six candidate prompts, with minor performance differences observed across the datasets. Based on our findings, we conclude that ChatGPT has great potential for keyphrase generation. Moreover, we discover that ChatGPT still faces challenges when it comes to generating absent keyphrases. Meanwhile, in the final section, we also present some limitations and futu
    
[^5]: 神经符号专家系统中基于事实库的逻辑推理的动态生成

    Dynamic Generation of Grounded Logical Explanations in a Neuro-Symbolic Expert System. (arXiv:2209.07662v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2209.07662](http://arxiv.org/abs/2209.07662)

    该研究提出一种新颖的方法，通过结合神经语言建模、引导生成和半参数密集检索，动态生成基于事实库的人类可解释的证明树，实现科学推理，并展现了强大的性能。

    

    我们提出了一个系统性推理的方法，可以产生基于事实库的人类可解释的证明树。我们的方法引发了经典的基于 Prolog 的推理引擎，其中我们通过结合神经语言建模、引导生成和半参数密集检索来替换手工制定的规则。我们通过一个新颖的系统 NELLIE 来演示这种方法，该系统动态地实例化可解释的推理规则，对自然语言语句的蕴含（去）组合进行捕捉和评分。这导致了强大的性能，在科学推理领域展示了如何逻辑地从经过人工验证的事实的组合中推导出答案的推理痕迹。

    We propose an approach for systematic reasoning that produces human interpretable proof trees grounded in a factbase. Our approach evokes classic Prolog-based inference engines, where we replace handcrafted rules by combining neural language modeling, guided generation, and semiparametric dense retrieval. We demonstrate this approach through a novel system, NELLIE, which dynamically instantiates interpretable inference rules that capture and score entailment (de)compositions over natural language statements. This leads to strong performance, as shown in the scientific reasoning domain, while also producing reasoning traces showing how answers derive logically from the composition of human-verified facts.
    

