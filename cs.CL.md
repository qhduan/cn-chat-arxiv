# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi: Multimodal Understanding Leaderboard with Text and Images](https://arxiv.org/abs/2402.03173) | Multi是一个多模态理解的排行榜，提供了一个综合数据集，评估多模态大型语言模型对理解复杂图表和科学问题的能力。它兼具准确和开放式的回答形式，挑战MLLM的各种任务，并包含超过18,000个问题。 |
| [^2] | [LLM-FuncMapper: Function Identification for Interpreting Complex Clauses in Building Codes via LLM.](http://arxiv.org/abs/2308.08728) | LLM-FuncMapper提出了一种通过LLM实现对建筑法规中复杂条款的函数识别的方法，通过定义原子函数和开发提示模板来解决传统逻辑表示的限制。 |

# 详细

[^1]: 多模态：文本和图像的多模态理解排行榜

    Multi: Multimodal Understanding Leaderboard with Text and Images

    [https://arxiv.org/abs/2402.03173](https://arxiv.org/abs/2402.03173)

    Multi是一个多模态理解的排行榜，提供了一个综合数据集，评估多模态大型语言模型对理解复杂图表和科学问题的能力。它兼具准确和开放式的回答形式，挑战MLLM的各种任务，并包含超过18,000个问题。

    

    多模态大型语言模型（MLLM）的快速进展强调了向学术界引入具有挑战性而又真实的基准的需求。现有的基准主要关注简单的自然图像理解，但Multi成为了MLLM的尖端基准，提供了一个综合性的数据集，用于评估MLLM对理解复杂图表和科学问题的能力。该基准反映了当前真实的考试风格，提供多模态的输入，并要求准确或开放式的回答，类似于现实中的学校考试。它通过各种任务挑战MLLM，从公式推导到图像细节分析，以及跨模态推理。Multi包括超过18,000个问题，重点关注不同格式的基于科学的问答。我们还引入了Multi-Elite，一个包含500个问题的子集，用于测试MLLM的极端情况，以及Multi-Extend，通过超过4..。

    Rapid progress in multimodal large language models (MLLMs) highlights the need to introduce challenging yet realistic benchmarks to the academic community. Existing benchmarks primarily focus on simple natural image understanding, but Multi emerges as a cutting-edge benchmark for MLLMs, offering a comprehensive dataset for evaluating MLLMs against understanding complex figures and tables, and scientific questions. This benchmark, reflecting current realistic examination styles, provides multimodal inputs and requires responses that are either precise or open-ended, similar to real-life school tests. It challenges MLLMs with a variety of tasks, ranging from formula derivation to image detail analysis, and cross-modality reasoning. Multi includes over 18,000 questions, with a focus on science-based QA in diverse formats. We also introduce Multi-Elite, a 500-question subset for testing the extremities of MLLMs, and Multi-Extend, which enhances In-Context Learning research with more than 4
    
[^2]: LLM-FuncMapper:通过LLM解释建筑法规中的复杂条款的函数识别

    LLM-FuncMapper: Function Identification for Interpreting Complex Clauses in Building Codes via LLM. (arXiv:2308.08728v1 [cs.AI])

    [http://arxiv.org/abs/2308.08728](http://arxiv.org/abs/2308.08728)

    LLM-FuncMapper提出了一种通过LLM实现对建筑法规中复杂条款的函数识别的方法，通过定义原子函数和开发提示模板来解决传统逻辑表示的限制。

    

    作为自动化规则检查（ARC）的关键阶段，对监管性文本的规则解释需要相当大的努力。然而，由于缺乏领域知识和传统逻辑表示的表达能力有限，解释具有隐式属性或复杂计算逻辑的监管条款仍然具有挑战性。因此，提出了一种基于大型语言模型（LLM）的识别各种监管条款所需的预定义函数的方法LLM-FuncMapper。首先，通过对建筑法规进行系统分析，定义了一系列原子函数，以捕捉隐式属性和复杂约束的共享计算逻辑，创建了常见块的数据库，用于解释监管条款。然后，开发了一个具有思维链的提示模板，并通过基于分类的调优策略进一步增强，以实现有效的函数识别功能。最后，验证了所提出的方法。

    As a vital stage of automated rule checking (ARC), rule interpretation of regulatory texts requires considerable effort. However, interpreting regulatory clauses with implicit properties or complex computational logic is still challenging due to the lack of domain knowledge and limited expressibility of conventional logic representations. Thus, LLM-FuncMapper, an approach to identifying predefined functions needed to interpret various regulatory clauses based on the large language model (LLM), is proposed. First, by systematically analysis of building codes, a series of atomic functions are defined to capture shared computational logics of implicit properties and complex constraints, creating a database of common blocks for interpreting regulatory clauses. Then, a prompt template with the chain of thought is developed and further enhanced with a classification-based tuning strategy, to enable common LLMs for effective function identification. Finally, the proposed approach is validated
    

