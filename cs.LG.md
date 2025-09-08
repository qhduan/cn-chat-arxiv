# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling](https://arxiv.org/abs/2402.12226) | AnyGPT是一个统一的多模态语言模型，通过离散表示实现各种模态的统一处理，能够在不改变大型语言模型架构或训练方式的情况下稳定训练，为新模态的无缝整合提供了可能。 |
| [^2] | [Human Curriculum Effects Emerge with In-Context Learning in Neural Networks](https://arxiv.org/abs/2402.08674) | 人类学习对示例课程和任务结构有敏感性。研究发现，在神经网络和语言模型中，通过上下文学习方法可以同时获得分组和交错训练的优势。 |
| [^3] | [Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts.](http://arxiv.org/abs/2401.14295) | 这篇论文探讨了结合结构的提示工程在提高大型语言模型推理性能方面的前景，通过思维链、思维树或思维图的设计来引导整体推理过程。通过大量实例，这种范式显著增强了模型在多个任务中的能力。总的来说，论文提供了一个通用蓝图，为未来的发展铺平道路。 |

# 详细

[^1]: AnyGPT：统一的多模式离散序列建模语言模型

    AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling

    [https://arxiv.org/abs/2402.12226](https://arxiv.org/abs/2402.12226)

    AnyGPT是一个统一的多模态语言模型，通过离散表示实现各种模态的统一处理，能够在不改变大型语言模型架构或训练方式的情况下稳定训练，为新模态的无缝整合提供了可能。

    

    我们介绍了 AnyGPT，这是一个任意多模式语言模型，利用离散表示统一处理各种模态，包括语音、文本、图像和音乐。AnyGPT 可以稳定训练，无需对当前大型语言模型（LLM）架构或训练范式进行任何改动。相反，它仅依赖于数据级预处理，促进了新模态的无缝集成到LLM中，类似于新语言的整合。我们构建了一个多模式文本中心的数据集，用于多模式对齐预训练。利用生成模型，我们合成了第一个大规模任意多模式指令数据集。它包括108k个多轮对话示例，精细地交织各种模态，从而使模型能够处理多模态输入和输出的任意组合。实验结果表明，AnyGPT能够促进...

    arXiv:2402.12226v1 Announce Type: cross  Abstract: We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages. We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs. Experimental results demonstrate that AnyGPT is capable of facilitat
    
[^2]: 使用上下文学习的神经网络中出现人类课程效应

    Human Curriculum Effects Emerge with In-Context Learning in Neural Networks

    [https://arxiv.org/abs/2402.08674](https://arxiv.org/abs/2402.08674)

    人类学习对示例课程和任务结构有敏感性。研究发现，在神经网络和语言模型中，通过上下文学习方法可以同时获得分组和交错训练的优势。

    

    人类学习对规则结构和训练中所使用的示例课程非常敏感。在由简洁规则控制的任务中，当相关示例在多次试验中被分组时，学习更加稳健；但在缺乏这样的规则的情况下，交错训练更加有效。迄今为止，没有神经模型能够同时捕捉到这些看似矛盾的效应。在本文中，我们展示了“上下文学习”（ICL）在使用元学习进行训练的神经网络和大型语言模型（LLMs）中自发产生了同样的权衡。ICL是通过内层循环算法在激活动力学中实现的一种“上下文内学习”（in-context learning）的能力，可以在没有权重更改的情况下学习新任务。对预训练的LLMs和元学习变压器进行的实验表明，ICL在涉及规则结构的任务中展示出了人类所示的分组优势，而同时进行权重学习则复制了人类在缺少这样结构的任务上所观察到的交错优势。

    Human learning is sensitive to rule-like structure and the curriculum of examples used for training. In tasks governed by succinct rules, learning is more robust when related examples are blocked across trials, but in the absence of such rules, interleaving is more effective. To date, no neural model has simultaneously captured these seemingly contradictory effects. Here we show that this same tradeoff spontaneously emerges with "in-context learning" (ICL) both in neural networks trained with metalearning and in large language models (LLMs). ICL is the ability to learn new tasks "in context" - without weight changes - via an inner-loop algorithm implemented in activation dynamics. Experiments with pretrained LLMs and metalearning transformers show that ICL exhibits the blocking advantage demonstrated in humans on a task involving rule-like structure, and conversely, that concurrent in-weight learning reproduces the interleaving advantage observed in humans on tasks lacking such structu
    
[^3]: 推理的拓扑学：揭秘思维链、树和图

    Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts. (arXiv:2401.14295v1 [cs.CL])

    [http://arxiv.org/abs/2401.14295](http://arxiv.org/abs/2401.14295)

    这篇论文探讨了结合结构的提示工程在提高大型语言模型推理性能方面的前景，通过思维链、思维树或思维图的设计来引导整体推理过程。通过大量实例，这种范式显著增强了模型在多个任务中的能力。总的来说，论文提供了一个通用蓝图，为未来的发展铺平道路。

    

    自然语言处理（NLP）领域近年来取得了显著进展，特别是在通过创新的提示技术提高大型语言模型（LLM）性能方面。其中，与结构相结合的提示工程被视为一种有前途的范式，其设计如思维链、思维树或思维图等，通过结构指导整体LLM推理过程。通过大量实例的说明，这种范式显著增强了LLM在逻辑或数学推理、规划或创造性写作等各种任务中的能力。为了方便理解这个不断发展的领域并为未来的发展铺平道路，我们设计了一个有效和高效的LLM推理方案的通用蓝图。为此，我们对提示执行流程进行了深入分析，澄清并明确定义了不同的概念。然后我们建立第一个分类系统

    The field of natural language processing (NLP) has witnessed significant progress in recent years, with a notable focus on improving large language models' (LLM) performance through innovative prompting techniques. Among these, prompt engineering coupled with structures has emerged as a promising paradigm, with designs such as Chain-of-Thought, Tree of Thoughts, or Graph of Thoughts, in which the overall LLM reasoning is guided by a structure such as a graph. As illustrated with numerous examples, this paradigm significantly enhances the LLM's capability to solve numerous tasks, ranging from logical or mathematical reasoning to planning or creative writing. To facilitate the understanding of this growing field and pave the way for future developments, we devise a general blueprint for effective and efficient LLM reasoning schemes. For this, we conduct an in-depth analysis of the prompt execution pipeline, clarifying and clearly defining different concepts. We then build the first taxon
    

