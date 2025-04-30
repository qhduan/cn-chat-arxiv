# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Era of Semantic Decoding](https://arxiv.org/abs/2403.14562) | 提出了一种名为语义解码的新观点，将LLM、人类输入和各种工具之间的协作过程构建为语义空间中的优化过程，促进了高效输出的构建。 |
| [^2] | [Evaluating the Symbol Binding Ability of Large Language Models for Multiple-Choice Questions in Vietnamese General Education.](http://arxiv.org/abs/2310.12059) | 本研究评估了大型语言模型在越南普通教育中对多项选择题符号绑定能力的能力，并创建了一个新颖且高质量的数据集来评估语言模型的符号绑定能力。 |
| [^3] | [Evaluating Multi-Agent Coordination Abilities in Large Language Models.](http://arxiv.org/abs/2310.03903) | 本研究构建了使用大型语言模型（LLMs）的智能体，并评估其在多智能体协调中的有效性。我们引入了LLM-Co框架，用于在三个游戏环境中评估LLMs的协调能力。评估结果显示LLMs具有推断伙伴意图和理解其行动的能力。 |
| [^4] | [Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering.](http://arxiv.org/abs/2303.01903) | 本研究提出了一个名为Prophet的框架，使用答案启发式方式促使GPT-3解决基于知识的视觉问答问题。在特定的知识型VQA数据集上训练一个纯VQA模型，并从中提取出答案启发式，可提高模型的性能。 |

# 详细

[^1]: 语义解码时代

    The Era of Semantic Decoding

    [https://arxiv.org/abs/2403.14562](https://arxiv.org/abs/2403.14562)

    提出了一种名为语义解码的新观点，将LLM、人类输入和各种工具之间的协作过程构建为语义空间中的优化过程，促进了高效输出的构建。

    

    最近的研究展现了在LLM（大型语言模型）、人类输入和各种工具之间编排协作以解决LLM固有局限性的想法具有巨大潜力。我们提出了一个名为语义解码的新观点，将这些协作过程构建为语义空间中的优化过程。具体来说，我们将LLM概念化为操纵我们称之为语义标记（已知思想）的有意义信息片段的语义处理器。LLM是众多其他语义处理器之一，包括人类和工具，比如搜索引擎或代码执行器。语义处理器集体参与语义标记的动态交流，逐步构建高效输出。我们称这些在语义空间中进行优化和搜索的协同作用，为语义解码算法。这个概念与已广为研究的语义解码问题直接平行。

    arXiv:2403.14562v1 Announce Type: cross  Abstract: Recent work demonstrated great promise in the idea of orchestrating collaborations between LLMs, human input, and various tools to address the inherent limitations of LLMs. We propose a novel perspective called semantic decoding, which frames these collaborative processes as optimization procedures in semantic space. Specifically, we conceptualize LLMs as semantic processors that manipulate meaningful pieces of information that we call semantic tokens (known thoughts). LLMs are among a large pool of other semantic processors, including humans and tools, such as search engines or code executors. Collectively, semantic processors engage in dynamic exchanges of semantic tokens to progressively construct high-utility outputs. We refer to these orchestrated interactions among semantic processors, optimizing and searching in semantic space, as semantic decoding algorithms. This concept draws a direct parallel to the well-studied problem of s
    
[^2]: 评估大型语言模型在越南普通教育中对多项选择题符号绑定能力的能力

    Evaluating the Symbol Binding Ability of Large Language Models for Multiple-Choice Questions in Vietnamese General Education. (arXiv:2310.12059v1 [cs.CL])

    [http://arxiv.org/abs/2310.12059](http://arxiv.org/abs/2310.12059)

    本研究评估了大型语言模型在越南普通教育中对多项选择题符号绑定能力的能力，并创建了一个新颖且高质量的数据集来评估语言模型的符号绑定能力。

    

    本文评估了大型语言模型（LLMs）在零样本、一次性和少样本设置下，执行多项选择符号绑定（MCSB）用于多项选择题回答（MCQA）任务的能力。我们将重点放在越南语上，因为越南语中的挑战性MCQA数据集较英语少。现有的两个数据集，ViMMRC 1.0和ViMMRC 2.0，专注于文学问题。越南自然语言处理（NLP）领域的最新研究侧重于评估ChatGPT在2019年至2023年的越南国家高中毕业考试（VNHSGE）中的解决方案。然而，这些研究主要关注ChatGPT如何逐步解决VNHSGE。我们的目标是通过为数学、物理、化学和生物的LaTeX公式输入提供结构化指南，创建一个新颖且高质量的数据集。该数据集可用于评估LLMs和较小的语言模型（LMs）的MCSB能力，因为数据集要求使用严格的LaTeX样式进行输入。我们重点预测字符（A、B、C或

    In this paper, we evaluate the ability of large language models (LLMs) to perform multiple choice symbol binding (MCSB) for multiple choice question answering (MCQA) tasks in zero-shot, one-shot, and few-shot settings. We focus on Vietnamese, with fewer challenging MCQA datasets than in English. The two existing datasets, ViMMRC 1.0 and ViMMRC 2.0, focus on literature. Recent research in Vietnamese natural language processing (NLP) has focused on the Vietnamese National High School Graduation Examination (VNHSGE) from 2019 to 2023 to evaluate ChatGPT. However, these studies have mainly focused on how ChatGPT solves the VNHSGE step by step. We aim to create a novel and high-quality dataset by providing structured guidelines for typing LaTeX formulas for mathematics, physics, chemistry, and biology. This dataset can be used to evaluate the MCSB ability of LLMs and smaller language models (LMs) because it is typed in a strict LaTeX style. We focus on predicting the character (A, B, C, or 
    
[^3]: 在大型语言模型中评估多智能体协调能力

    Evaluating Multi-Agent Coordination Abilities in Large Language Models. (arXiv:2310.03903v1 [cs.CL])

    [http://arxiv.org/abs/2310.03903](http://arxiv.org/abs/2310.03903)

    本研究构建了使用大型语言模型（LLMs）的智能体，并评估其在多智能体协调中的有效性。我们引入了LLM-Co框架，用于在三个游戏环境中评估LLMs的协调能力。评估结果显示LLMs具有推断伙伴意图和理解其行动的能力。

    

    当代人工智能研究的一个重要目标是开发能够熟练进行多智能体协调、有效与人类和其他系统合作的智能体。大型语言模型（LLM）以其显著的理解、生成和解释语言的能力成为开发这种智能体的有希望的候选模型。本研究中，我们构建了使用LLM构建的智能体，并评估其在各种协调场景中的有效性。我们引入了特别设计的LLM-Co框架，使LLM能够参与协调游戏。通过LLM-Co框架，我们在三个游戏环境中进行评估，并将评估分为五个方面：心智理论、情境推理、持续协调、对合作伙伴的稳健性和明确辅助。首先，心智理论和情境推理的评估揭示了LLM推断伙伴意图和理解其行动的能力。

    A pivotal aim in contemporary AI research is to develop agents proficient in multi-agent coordination, enabling effective collaboration with both humans and other systems. Large Language Models (LLMs), with their notable ability to understand, generate, and interpret language in a human-like manner, stand out as promising candidates for the development of such agents. In this study, we build and assess the effectiveness of agents crafted using LLMs in various coordination scenarios. We introduce the LLM-Coordination (LLM-Co) Framework, specifically designed to enable LLMs to play coordination games. With the LLM-Co framework, we conduct our evaluation with three game environments and organize the evaluation into five aspects: Theory of Mind, Situated Reasoning, Sustained Coordination, Robustness to Partners, and Explicit Assistance. First, the evaluation of the Theory of Mind and Situated Reasoning reveals the capabilities of LLM to infer the partner's intention and reason actions acco
    
[^4]: 用答案启发式方式促使大型语言模型解决基于知识的视觉问答问题

    Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering. (arXiv:2303.01903v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.01903](http://arxiv.org/abs/2303.01903)

    本研究提出了一个名为Prophet的框架，使用答案启发式方式促使GPT-3解决基于知识的视觉问答问题。在特定的知识型VQA数据集上训练一个纯VQA模型，并从中提取出答案启发式，可提高模型的性能。

    

    基于知识的视觉问答需要超出图像范围的外部知识来回答问题。早期的研究从显式知识库（KBs）检索所需的知识，这经常会引入与问题无关的信息，从而限制了模型的性能。最近的研究试图将大型语言模型（即GPT-3）作为隐含式知识引擎来获取回答所需的必要知识。尽管这些方法取得了令人鼓舞的结果，但我们认为它们还没有充分发挥GPT-3的能力，因为提供的输入信息仍然不足。在本文中，我们提出了Prophet——一个概念上简单的框架，旨在通过回答启发式方式，促使GPT-3解决基于知识的VQA问题。具体来说，我们首先在特定的基于知识的VQA数据集上训练一个纯VQA模型，而不使用外部知识。之后，我们从模型中提取了两种互补的答案启发式：答案候选项。

    Knowledge-based visual question answering (VQA) requires external knowledge beyond the image to answer the question. Early studies retrieve required knowledge from explicit knowledge bases (KBs), which often introduces irrelevant information to the question, hence restricting the performance of their models. Recent works have sought to use a large language model (i.e., GPT-3) as an implicit knowledge engine to acquire the necessary knowledge for answering. Despite the encouraging results achieved by these methods, we argue that they have not fully activated the capacity of GPT-3 as the provided input information is insufficient. In this paper, we present Prophet -- a conceptually simple framework designed to prompt GPT-3 with answer heuristics for knowledge-based VQA. Specifically, we first train a vanilla VQA model on a specific knowledge-based VQA dataset without external knowledge. After that, we extract two types of complementary answer heuristics from the model: answer candidates 
    

