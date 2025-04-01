# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Punctuation Restoration Improves Structure Understanding without Supervision](https://arxiv.org/abs/2402.08382) | 标点符号恢复是一个有效的学习目标，可以改善结构理解并提高模型性能。 |
| [^2] | [Rethinking Optimization and Architecture for Tiny Language Models](https://arxiv.org/abs/2402.02791) | 本研究重新思考了微型语言模型的优化和架构，通过经验研究发现了在微型语言模型中特别有效的设计公式，并在多语种数据集上训练了高性能的微型语言模型。 |
| [^3] | [Cultural and Linguistic Diversity Improves Visual Representations.](http://arxiv.org/abs/2310.14356) | 这项研究发现数据集和模型生成的图像描述在不同语言间存在显著的语义差异，多语言数据有更高的语义覆盖率，并且基于多语言训练的模型表现更好。 |
| [^4] | [Can Large Language Models Play Text Games Well? Current State-of-the-Art and Open Questions.](http://arxiv.org/abs/2304.02868) | 本文探究大型语言模型在玩文字游戏的能力，并发现其表现有竞争力，但仍然缺乏智能，有待提升。 |

# 详细

[^1]: 标点符号恢复在没有监督的情况下改善结构理解

    Punctuation Restoration Improves Structure Understanding without Supervision

    [https://arxiv.org/abs/2402.08382](https://arxiv.org/abs/2402.08382)

    标点符号恢复是一个有效的学习目标，可以改善结构理解并提高模型性能。

    

    无监督学习目标，如语言建模和去噪等，在生成预训练模型方面起着重要作用，这些预训练模型能够执行从自然语言理解到会话任务的各种下游应用。然而，尽管最近的大型语言模型具有令人印象深刻的对话能力，但它们在捕捉文本的句法或语义结构方面的能力仍然落后。我们假设，语言性能和机器能力之间的不匹配归因于当前流行的预训练目标未能充分传递语言结构知识给计算系统。我们展示了标点符号恢复对结构相关任务的内部和外部表现的改善，如命名实体识别、开放式信息提取、分块和词性标注。标点符号恢复是一个有效的学习目标，可以改善结构理解并产生更加鲁棒的模型。

    Unsupervised learning objectives like language modeling and de-noising constitute a significant part in producing pre-trained models that perform various downstream applications from natural language understanding to conversational tasks. However, despite impressive conversational capabilities of recent large language model, their abilities to capture syntactic or semantic structure within text lag behind. We hypothesize that the mismatch between linguistic performance and competence in machines is attributable to insufficient transfer of linguistic structure knowledge to computational systems with currently popular pre-training objectives. We show that punctuation restoration transfers to improvements in in- and out-of-distribution performance on structure-related tasks like named entity recognition, open information extraction, chunking, and part-of-speech tagging. Punctuation restoration is an effective learning objective that can improve structure understanding and yield a more rob
    
[^2]: 重新思考微型语言模型的优化和架构

    Rethinking Optimization and Architecture for Tiny Language Models

    [https://arxiv.org/abs/2402.02791](https://arxiv.org/abs/2402.02791)

    本研究重新思考了微型语言模型的优化和架构，通过经验研究发现了在微型语言模型中特别有效的设计公式，并在多语种数据集上训练了高性能的微型语言模型。

    

    大型语言模型（LLMs）的威力通过大量的数据和计算资源得到了证明。然而，在移动设备上应用语言模型面临着计算和内存成本的巨大挑战，迫切需要高性能的微型语言模型。受复杂训练过程的限制，优化语言模型的许多细节很少得到仔细研究。在本研究中，基于一个具有10亿参数的微型语言模型，我们仔细设计了一系列经验研究来分析每个组件的影响。主要讨论了三个方面，即神经架构、参数初始化和优化策略。多个设计公式在微型语言模型中经验性地被证明特别有效，包括分词器压缩、架构调整、参数继承和多轮训练。然后，我们在1.6T多语种数据集上训练了PanGu-$\pi$-1B Pro和PanGu-$\pi$-1.5B Pro。

    The power of large language models (LLMs) has been demonstrated through numerous data and computing resources. However, the application of language models on mobile devices is facing huge challenge on the computation and memory costs, that is, tiny language models with high performance are urgently required. Limited by the highly complex training process, there are many details for optimizing language models that are seldom studied carefully. In this study, based on a tiny language model with 1B parameters, we carefully design a series of empirical study to analyze the effect of each component. Three perspectives are mainly discussed, i.e., neural architecture, parameter initialization, and optimization strategy. Several design formulas are empirically proved especially effective for tiny language models, including tokenizer compression, architecture tweaking, parameter inheritance and multiple-round training. Then we train PanGu-$\pi$-1B Pro and PanGu-$\pi$-1.5B Pro on 1.6T multilingu
    
[^3]: 文化和语言多样性提高了视觉表示

    Cultural and Linguistic Diversity Improves Visual Representations. (arXiv:2310.14356v1 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2310.14356](http://arxiv.org/abs/2310.14356)

    这项研究发现数据集和模型生成的图像描述在不同语言间存在显著的语义差异，多语言数据有更高的语义覆盖率，并且基于多语言训练的模型表现更好。

    

    计算机视觉通常将感知视为客观的，并且这种假设在数据集收集和模型训练中得到反映。例如，不同语言的图像描述通常被假定为相同语义内容的翻译。然而，跨文化心理学和语言学的研究表明，个体的视觉感知因其文化背景和所说的语言而异。在本文中，我们展示了在数据集和模型生成的标题中，不同语言之间存在显著的语义内容差异。当数据是多语言而不是单语言时，标题的语义覆盖率平均更高，以场景图、嵌入和语言复杂性进行测量。例如，与一组单语标题相比，多语标题平均有21.8％更多的对象，24.5％更多的关系，以及27.1％更多的属性。此外，使用来自不同语言的内容训练的模型表现最好。

    Computer vision often treats perception as objective, and this assumption gets reflected in the way that datasets are collected and models are trained. For instance, image descriptions in different languages are typically assumed to be translations of the same semantic content. However, work in cross-cultural psychology and linguistics has shown that individuals differ in their visual perception depending on their cultural background and the language they speak. In this paper, we demonstrate significant differences in semantic content across languages in both dataset and model-produced captions. When data is multilingual as opposed to monolingual, captions have higher semantic coverage on average, as measured by scene graph, embedding, and linguistic complexity. For example, multilingual captions have on average 21.8% more objects, 24.5% more relations, and 27.1% more attributes than a set of monolingual captions. Moreover, models trained on content from different languages perform bes
    
[^4]: 大型语言模型能否能够很好地玩文字游戏？现状和未来问题研究

    Can Large Language Models Play Text Games Well? Current State-of-the-Art and Open Questions. (arXiv:2304.02868v1 [cs.CL])

    [http://arxiv.org/abs/2304.02868](http://arxiv.org/abs/2304.02868)

    本文探究大型语言模型在玩文字游戏的能力，并发现其表现有竞争力，但仍然缺乏智能，有待提升。

    

    最近，诸如ChatGPT和GPT-4之类的大型语言模型展示了它们与人类用户通信的卓越能力。本技术报告旨在调查它们在玩文字游戏方面的能力，这要求玩家通过与游戏世界的对话来理解环境并对情况做出反应。我们的实验表明，与所有现有系统相比，ChatGPT表现出有竞争力，但仍然表现出较低的智能水平。确切地说，ChatGPT无法通过玩游戏或阅读游戏手册来构建世界模型；它可能无法利用它已经拥有的世界知识；它无法推断出随着游戏进展的每一步的目标。我们的结果在人工智能、机器学习和自然语言处理交叉领域开启了新的研究问题。

    Large language models (LLMs) such as ChatGPT and GPT-4 have recently demonstrated their remarkable abilities of communicating with human users. In this technical report, we take an initiative to investigate their capacities of playing text games, in which a player has to understand the environment and respond to situations by having dialogues with the game world. Our experiments show that ChatGPT performs competitively compared to all the existing systems but still exhibits a low level of intelligence. Precisely, ChatGPT can not construct the world model by playing the game or even reading the game manual; it may fail to leverage the world knowledge that it already has; it cannot infer the goal of each step as the game progresses. Our results open up new research questions at the intersection of artificial intelligence, machine learning, and natural language processing.
    

