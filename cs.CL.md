# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RealKIE: Five Novel Datasets for Enterprise Key Information Extraction](https://arxiv.org/abs/2403.20101) | RealKIE提供了五个具有挑战性的企业关键信息提取数据集，为投资分析和法律数据处理等任务提供了现实的测试基地，并为NLP模型的发展做出了贡献。 |
| [^2] | [Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models](https://arxiv.org/abs/2402.10612) | 本研究提出了一种新方法Rowen，通过选择性检索增强过程，采用多语义感知检测模块来平衡参数化知识和外部信息，以减轻大型语言模型中的幻觉问题。 |
| [^3] | [Jailbreaking Proprietary Large Language Models using Word Substitution Cipher](https://arxiv.org/abs/2402.10601) | 本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。 |
| [^4] | [Understanding Retrieval Augmentation for Long-Form Question Answering.](http://arxiv.org/abs/2310.12150) | 这项研究分析了长篇问答中的检索增强语言模型的影响，研究了生成答案的属性和归因模式，并找出了归因错误的主要原因。研究结果对长篇、知识丰富的文本生成提供了新的见解。 |
| [^5] | [AgentBench: Evaluating LLMs as Agents.](http://arxiv.org/abs/2308.03688) | AgentBench是一个用于评估LLMs作为代理人的多维度基准，发现在复杂环境中，商业LLMs在充当代理人方面表现强劲，但与开源竞争对手相比，存在显著性能差距。该研究揭示了LLMs在长期推理、决策和指令遵循能力上的瓶颈。 |

# 详细

[^1]: RealKIE: 五个新颖的企业关键信息提取数据集

    RealKIE: Five Novel Datasets for Enterprise Key Information Extraction

    [https://arxiv.org/abs/2403.20101](https://arxiv.org/abs/2403.20101)

    RealKIE提供了五个具有挑战性的企业关键信息提取数据集，为投资分析和法律数据处理等任务提供了现实的测试基地，并为NLP模型的发展做出了贡献。

    

    我们介绍了RealKIE，这是一个旨在推动关键信息提取方法发展的五个具有挑战性的数据集基准，重点是企业应用。这些数据集包括美国SEC S1文件、美国保密协议、英国慈善报告、FCC发票和资源合同等各种类型的文档。每个数据集都具有独特的挑战：文本序列化不佳、长文档中稀疏的注释和复杂的表格布局。这些数据集为关键信息提取任务（如投资分析和法律数据处理）提供了一个现实的测试基地。除了介绍这些数据集外，我们还提供了对注释过程、文档处理技术和基线建模方法的深入描述。这一贡献促进了能够处理实际挑战的NLP模型的发展，并支持进一步研究可应用于工业的信息提取技术。

    arXiv:2403.20101v1 Announce Type: new  Abstract: We introduce RealKIE, a benchmark of five challenging datasets aimed at advancing key information extraction methods, with an emphasis on enterprise applications. The datasets include a diverse range of documents including SEC S1 Filings, US Non-disclosure Agreements, UK Charity Reports, FCC Invoices, and Resource Contracts. Each presents unique challenges: poor text serialization, sparse annotations in long documents, and complex tabular layouts. These datasets provide a realistic testing ground for key information extraction tasks like investment analysis and legal data processing.   In addition to presenting these datasets, we offer an in-depth description of the annotation process, document processing techniques, and baseline modeling approaches. This contribution facilitates the development of NLP models capable of handling practical challenges and supports further research into information extraction technologies applicable to indu
    
[^2]: 仅在需要时检索：大型语言模型中的适应性检索增强以减轻幻觉

    Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models

    [https://arxiv.org/abs/2402.10612](https://arxiv.org/abs/2402.10612)

    本研究提出了一种新方法Rowen，通过选择性检索增强过程，采用多语义感知检测模块来平衡参数化知识和外部信息，以减轻大型语言模型中的幻觉问题。

    

    幻觉对于大型语言模型（LLMs）的实际实施构成了显著挑战。生成事实内容时利用参数化知识受到LLMs有限知识的限制，可能导致内部幻觉。虽然整合外部信息可以填补知识空白，但也会引入无关信息的风险，从而增加外部幻觉的可能性。在LLMs内部平衡地整合参数化知识和外部信息对缓解幻觉至关重要。本研究中，我们提出Rowen，一种增强LLMs的新方法，其中包括一种选择性检索增强过程，旨在解决幻觉输出。该过程由一个多语义感知检测模块管理，该模块评估了对相同查询在不同语言中的扰动响应的一致性。

    arXiv:2402.10612v1 Announce Type: new  Abstract: Hallucinations pose a significant challenge for the practical implementation of large language models (LLMs). The utilization of parametric knowledge in generating factual content is constrained by the limited knowledge of LLMs, potentially resulting in internal hallucinations. While incorporating external information can help fill knowledge gaps, it also introduces the risk of irrelevant information, thereby increasing the likelihood of external hallucinations. A careful and balanced integration of the parametric knowledge within LLMs with external information is crucial to alleviate hallucinations. In this study, we present Rowen, a novel approach that enhances LLMs with a selective retrieval augmentation process tailored to address hallucinated outputs. This process is governed by a multilingual semantic-aware detection module, which evaluates the consistency of the perturbed responses across various languages for the same queries. Up
    
[^3]: 使用单词替换密码来越狱专有的大型语言模型

    Jailbreaking Proprietary Large Language Models using Word Substitution Cipher

    [https://arxiv.org/abs/2402.10601](https://arxiv.org/abs/2402.10601)

    本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。

    

    大型语言模型（LLMs）遵循道德和伦理准则，但仍然容易受到名为Jailbreak的创意提示的影响，这些提示可以绕过对齐过程。然而，大多数越狱提示包含自然语言（主要是英语）中的有害问题，可以被LLMs自身检测到。本文提出了使用密码技术编码的越狱提示。我们首先在最先进的LLM，GPT-4上进行了一个试点研究，解码了使用各种密码技术加密的几个安全句子，发现简单的单词替换密码可以被最有效地解码。受此结果启发，我们使用这种编码技术来编写越狱提示。我们提供了将不安全单词映射到安全单词，并使用这些映射的单词提出不安全问题的映射。实验结果显示，我们提出的越狱攻击成功率（高达59.42%）。

    arXiv:2402.10601v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are aligned to moral and ethical guidelines but remain susceptible to creative prompts called Jailbreak that can bypass the alignment process. However, most jailbreaking prompts contain harmful questions in the natural language (mainly English), which can be detected by the LLM themselves. In this paper, we present jailbreaking prompts encoded using cryptographic techniques. We first present a pilot study on the state-of-the-art LLM, GPT-4, in decoding several safe sentences that have been encrypted using various cryptographic techniques and find that a straightforward word substitution cipher can be decoded most effectively. Motivated by this result, we use this encoding technique for writing jailbreaking prompts. We present a mapping of unsafe words with safe words and ask the unsafe question using these mapped words. Experimental results show an attack success rate (up to 59.42%) of our proposed jailbrea
    
[^4]: 理解用于长篇问答的检索增强

    Understanding Retrieval Augmentation for Long-Form Question Answering. (arXiv:2310.12150v1 [cs.CL])

    [http://arxiv.org/abs/2310.12150](http://arxiv.org/abs/2310.12150)

    这项研究分析了长篇问答中的检索增强语言模型的影响，研究了生成答案的属性和归因模式，并找出了归因错误的主要原因。研究结果对长篇、知识丰富的文本生成提供了新的见解。

    

    我们在长篇问答中提出了一项检索增强的语言模型（LMs）研究。我们通过比较使用相同证据文档的模型生成的答案，分析了检索增强对不同LMs的影响，以及检索文档集质量对相同LMs生成的答案的影响。我们研究了生成答案的各种属性（例如，流畅度、长度、变异性），重点在于将生成的长篇答案归因于文本中的证据文档。我们进行了答案归因的人工标注并评估了自动评判归因的方法。我们的研究为检索增强如何影响LMs生成长篇、知识丰富的文本提供了新的见解。我们进一步确定了长文本生成的归因模式并分析了归因错误的主要原因。综上所述，我们的分析揭示了检索增强对长篇、知识丰富的文本生成的影响，并提供了方向。

    We present a study of retrieval-augmented language models (LMs) on long-form question answering. We analyze how retrieval augmentation impacts different LMs, by comparing answers generated from models while using the same evidence documents, and how differing quality of retrieval document set impacts the answers generated from the same LM. We study various attributes of generated answers (e.g., fluency, length, variance) with an emphasis on the attribution of generated long-form answers to in-context evidence documents. We collect human annotations of answer attribution and evaluate methods for automatically judging attribution. Our study provides new insights on how retrieval augmentation impacts long, knowledge-rich text generation of LMs. We further identify attribution patterns for long text generation and analyze the main culprits of attribution errors. Together, our analysis reveals how retrieval augmentation impacts long knowledge-rich text generation and provide directions for 
    
[^5]: AgentBench: 评估LLMs作为代理人

    AgentBench: Evaluating LLMs as Agents. (arXiv:2308.03688v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2308.03688](http://arxiv.org/abs/2308.03688)

    AgentBench是一个用于评估LLMs作为代理人的多维度基准，发现在复杂环境中，商业LLMs在充当代理人方面表现强劲，但与开源竞争对手相比，存在显著性能差距。该研究揭示了LLMs在长期推理、决策和指令遵循能力上的瓶颈。

    

    大型语言模型(LLMs)变得越来越智能和自主，针对传统的NLP任务之外的现实世界实际任务。因此，迫切需要在互动环境中评估LLMs作为代理人在具有挑战性的任务上的推理和决策能力。我们提出了AgentBench，一个多维度演变的基准，目前包括8个不同的环境，以评估LLM作为代理人在多轮开放式生成设置中的推理和决策能力。我们在27个基于API和开源的LLM上进行了广泛的测试，结果表明，虽然顶级商业LLM在复杂环境中表现出良好的代理人能力，但它们与开源竞争对手之间的性能差距很大。我们找出了环境和LLM中失败的典型原因，表明长期推理、决策和遵循指示能力不佳是开发可用LLM代理人的主要障碍。通过对代码和高质量进行训练

    Large Language Models (LLMs) are becoming increasingly smart and autonomous, targeting real-world pragmatic missions beyond traditional NLP tasks. As a result, there has been an urgent need to evaluate LLMs as agents on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional evolving benchmark that currently consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities in a multi-turn open-ended generation setting. Our extensive test over 27 API-based and open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and OSS competitors. We identify the typical reasons of failures in environments and LLMs, showing that poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents. Training on code and high quality 
    

