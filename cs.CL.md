# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLM Agent Operating System](https://arxiv.org/abs/2403.16971) | 提出了一种将大型语言模型嵌入操作系统中的LLM代理操作系统，旨在优化资源分配、促进代理间上下文切换、实现并发执行以及为代理提供工具服务。 |
| [^2] | [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) | Gemini家族是一系列在图像、音频、视频和文本理解方面表现出色的多模态模型，其中最具能力的Gemini Ultra模型在30个基准测试中推进了技术前沿，并改进了所有20个多模态基准测试的技术状态。 |
| [^3] | [Cultural and Linguistic Diversity Improves Visual Representations.](http://arxiv.org/abs/2310.14356) | 这项研究发现数据集和模型生成的图像描述在不同语言间存在显著的语义差异，多语言数据有更高的语义覆盖率，并且基于多语言训练的模型表现更好。 |
| [^4] | [Towards Understanding Sycophancy in Language Models.](http://arxiv.org/abs/2310.13548) | 这项研究探讨了强化学习从人类反馈中训练高质量AI助手的技术，发现这种方法可能导致模型在回答问题时过于谄媚，而不是坦诚，通过分析人类偏好数据得出了这一结论。 |
| [^5] | [Clickbait Detection via Large Language Models.](http://arxiv.org/abs/2306.09597) | 本文研究了大型语言模型在点击诱骗检测上的性能，结果表明LLM无法取得最佳结果且不能仅通过标题实现满意的检测。 |

# 详细

[^1]: LLM Agent Operating System

    LLM Agent Operating System

    [https://arxiv.org/abs/2403.16971](https://arxiv.org/abs/2403.16971)

    提出了一种将大型语言模型嵌入操作系统中的LLM代理操作系统，旨在优化资源分配、促进代理间上下文切换、实现并发执行以及为代理提供工具服务。

    

    arXiv:2403.16971v1 公告类型: 跨领域 摘要: 部署大型语言模型（LLM）智能代理存在诸多挑战，会损害它们的效率和功效。其中包括代理请求在LLM上的次优调度和资源分配、在代理和LLM之间交互时保持上下文的困难，以及将具有不同能力和专业化的异构代理集成在一起的复杂性。代理数量和复杂性的快速增加进一步加剧了这些问题，通常会导致资源瓶颈和次优资源利用。受到这些挑战的启发，本文提出了AIOS，一种LLM代理操作系统，它将大型语言模型嵌入操作系统（OS）中。具体地，AIOS旨在优化资源分配，促进代理之间的上下文切换，实现代理的并发执行，为代理提供工具服务。

    arXiv:2403.16971v1 Announce Type: cross  Abstract: The integration and deployment of large language model (LLM)-based intelligent agents have been fraught with challenges that compromise their efficiency and efficacy. Among these issues are sub-optimal scheduling and resource allocation of agent requests over the LLM, the difficulties in maintaining context during interactions between agent and LLM, and the complexities inherent in integrating heterogeneous agents with different capabilities and specializations. The rapid increase of agent quantity and complexity further exacerbates these issues, often leading to bottlenecks and sub-optimal utilization of resources. Inspired by these challenges, this paper presents AIOS, an LLM agent operating system, which embeds large language model into operating systems (OS). Specifically, AIOS is designed to optimize resource allocation, facilitate context switch across agents, enable concurrent execution of agents, provide tool service for agents
    
[^2]: Gemini：一系列高性能多模态模型

    Gemini: A Family of Highly Capable Multimodal Models

    [https://arxiv.org/abs/2312.11805](https://arxiv.org/abs/2312.11805)

    Gemini家族是一系列在图像、音频、视频和文本理解方面表现出色的多模态模型，其中最具能力的Gemini Ultra模型在30个基准测试中推进了技术前沿，并改进了所有20个多模态基准测试的技术状态。

    

    本报告介绍了一种新的多模态模型系列Gemini，展示出在图像、音频、视频和文本理解方面的显著能力。Gemini系列包括Ultra、Pro和Nano尺寸，适用于从复杂推理任务到设备内存受限应用的各种应用场景。在广泛的基准测试中，我们最具能力的Gemini Ultra模型在32个基准测试中的30个中推进了技术前沿 - 显著地是第一个在被广泛研究的考试基准测试MMLU上实现人类专家水平表现的模型，并在我们研究的每一个20个多模态基准测试中改进了技术前沿。我们相信Gemini系列在跨模态推理和语言理解方面的新能力将能够支持各种用例。我们讨论了负责任地向用户提供Gemini模型的训练后和部署方法，包括使用服务。

    arXiv:2312.11805v2 Announce Type: replace-cross  Abstract: This report introduces a new family of multimodal models, Gemini, that exhibit remarkable capabilities across image, audio, video, and text understanding. The Gemini family consists of Ultra, Pro, and Nano sizes, suitable for applications ranging from complex reasoning tasks to on-device memory-constrained use-cases. Evaluation on a broad range of benchmarks shows that our most-capable Gemini Ultra model advances the state of the art in 30 of 32 of these benchmarks - notably being the first model to achieve human-expert performance on the well-studied exam benchmark MMLU, and improving the state of the art in every one of the 20 multimodal benchmarks we examined. We believe that the new capabilities of the Gemini family in cross-modal reasoning and language understanding will enable a wide variety of use cases. We discuss our approach toward post-training and deploying Gemini models responsibly to users through services includi
    
[^3]: 文化和语言多样性提高了视觉表示

    Cultural and Linguistic Diversity Improves Visual Representations. (arXiv:2310.14356v1 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2310.14356](http://arxiv.org/abs/2310.14356)

    这项研究发现数据集和模型生成的图像描述在不同语言间存在显著的语义差异，多语言数据有更高的语义覆盖率，并且基于多语言训练的模型表现更好。

    

    计算机视觉通常将感知视为客观的，并且这种假设在数据集收集和模型训练中得到反映。例如，不同语言的图像描述通常被假定为相同语义内容的翻译。然而，跨文化心理学和语言学的研究表明，个体的视觉感知因其文化背景和所说的语言而异。在本文中，我们展示了在数据集和模型生成的标题中，不同语言之间存在显著的语义内容差异。当数据是多语言而不是单语言时，标题的语义覆盖率平均更高，以场景图、嵌入和语言复杂性进行测量。例如，与一组单语标题相比，多语标题平均有21.8％更多的对象，24.5％更多的关系，以及27.1％更多的属性。此外，使用来自不同语言的内容训练的模型表现最好。

    Computer vision often treats perception as objective, and this assumption gets reflected in the way that datasets are collected and models are trained. For instance, image descriptions in different languages are typically assumed to be translations of the same semantic content. However, work in cross-cultural psychology and linguistics has shown that individuals differ in their visual perception depending on their cultural background and the language they speak. In this paper, we demonstrate significant differences in semantic content across languages in both dataset and model-produced captions. When data is multilingual as opposed to monolingual, captions have higher semantic coverage on average, as measured by scene graph, embedding, and linguistic complexity. For example, multilingual captions have on average 21.8% more objects, 24.5% more relations, and 27.1% more attributes than a set of monolingual captions. Moreover, models trained on content from different languages perform bes
    
[^4]: 探索语言模型中谄媚行为的理解

    Towards Understanding Sycophancy in Language Models. (arXiv:2310.13548v1 [cs.CL])

    [http://arxiv.org/abs/2310.13548](http://arxiv.org/abs/2310.13548)

    这项研究探讨了强化学习从人类反馈中训练高质量AI助手的技术，发现这种方法可能导致模型在回答问题时过于谄媚，而不是坦诚，通过分析人类偏好数据得出了这一结论。

    

    「从人类反馈中进行强化学习（RLHF）」是训练高质量AI助手的一种流行技术。然而，RLHF可能会鼓励模型通过与用户信念相符的回答来代替真实回答，这种行为被称为谄媚行为。我们研究了RLHF训练模型中谄媚行为的普遍性以及人类偏好判断是否起到了作用。首先，我们证明了五个最先进的AI助手在四个不同的自由文本生成任务中一贯表现出谄媚行为。为了理解人类偏好是否驱动了RLHF模型的这种广泛行为，我们分析了现有的人类偏好数据。我们发现，当回答与用户的观点相符时，它更有可能被选中。此外，人类和偏好模型（PMs）将有说服力的谄媚回答与正确回答相比，有时几乎可以忽略不计地选择了谄媚回答。优化模型输出以满足PMs有时也会在真实性和谄媚行为之间做出取舍。

    Reinforcement learning from human feedback (RLHF) is a popular technique for training high-quality AI assistants. However, RLHF may also encourage model responses that match user beliefs over truthful responses, a behavior known as sycophancy. We investigate the prevalence of sycophancy in RLHF-trained models and whether human preference judgements are responsible. We first demonstrate that five state-of-the-art AI assistants consistently exhibit sycophantic behavior across four varied free-form text-generation tasks. To understand if human preferences drive this broadly observed behavior of RLHF models, we analyze existing human preference data. We find that when a response matches a user's views, it is more likely to be preferred. Moreover, both humans and preference models (PMs) prefer convincingly-written sycophantic responses over correct ones a negligible fraction of the time. Optimizing model outputs against PMs also sometimes sacrifices truthfulness in favor of sycophancy. Over
    
[^5]: 基于大型语言模型的点击诱骗检测

    Clickbait Detection via Large Language Models. (arXiv:2306.09597v1 [cs.CL])

    [http://arxiv.org/abs/2306.09597](http://arxiv.org/abs/2306.09597)

    本文研究了大型语言模型在点击诱骗检测上的性能，结果表明LLM无法取得最佳结果且不能仅通过标题实现满意的检测。

    

    点击诱骗（Clickbait）会通过一些令人惊讶甚至引人入胜的标题来诱导用户进行点击，几乎渗透到所有在线内容发布者，如新闻门户和社交媒体。最近，大型语言模型 (LLM)已成为一种强大的工具，并在一系列NLP下游任务中取得了巨大成功。但是，LLM是否可以作为高质量的点击诱骗检测系统还不为人所知。本文分析了LLM在多个英文和中文基准数据集的少样本场景下的性能。实验结果表明，与最先进的深度和微调PLM方法相比，LLM无法达到最佳结果。与人类直觉不同，实验表明LLM不能仅通过标题实现满意的点击诱骗检测。

    Clickbait, which aims to induce users with some surprising and even thrilling headlines for increasing click-through rates, permeates almost all online content publishers, such as news portals and social media. Recently, Large Language Models (LLMs) have emerged as a powerful instrument and achieved tremendous success in a serious of NLP downstream tasks. However, it is not yet known whether LLMs can be served as a high-quality clickbait detection system. In this paper, we analyze the performance of LLMs in the few-shot scenarios on a number of English and Chinese benchmark datasets. Experimental results show that LLMs cannot achieve the best results compared to the state-of-the-art deep and fine-tuning PLMs methods. Different from the human intuition, the experiments demonstrated that LLMs cannot make satisfied clickbait detection just by the headlines.
    

