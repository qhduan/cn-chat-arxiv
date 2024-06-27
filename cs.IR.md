# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generate then Retrieve: Conversational Response Retrieval Using LLMs as Answer and Query Generators](https://arxiv.org/abs/2403.19302) | 本文提出了一种利用大型语言模型生成多个查询以增强对话响应检索的方法，并基于不同LLMs实现评估，同时还提出了一个新的TREC iKAT基准。 |
| [^2] | [ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction.](http://arxiv.org/abs/2310.09234) | 这篇论文提出了一个新颖的模型，旨在同时模拟语义和协同知识，以实现准确的CTR估计，并解决推理效率问题。 |

# 详细

[^1]: 生成然后检索：使用LLM作为答案和查询生成器的对话响应检索

    Generate then Retrieve: Conversational Response Retrieval Using LLMs as Answer and Query Generators

    [https://arxiv.org/abs/2403.19302](https://arxiv.org/abs/2403.19302)

    本文提出了一种利用大型语言模型生成多个查询以增强对话响应检索的方法，并基于不同LLMs实现评估，同时还提出了一个新的TREC iKAT基准。

    

    CIS是信息检索中的一个重要领域，专注于开发交互式知识助手。这些系统必须能够熟练地理解用户在对话环境中的信息需求，并检索相关信息。为实现这一目标，现有方法使用一个称为重写查询的查询来建模用户的信息需求，并将此查询用于段落检索。在本文中，我们提出了三种用于生成多个查询以增强检索的不同方法。在这些方法中，我们利用大型语言模型（LLMs）的能力来理解用户的信息需求和生成适当的响应，以生成多个查询。我们实现并评估了提出的模型，利用包括GPT-4和Llama-2在零-shot和少-shot设置中的各种LLMs。此外，我们基于gpt 3.5的判断提出了一个针对TREC iKAT的新基准。我们的实验揭示了效果

    arXiv:2403.19302v1 Announce Type: new  Abstract: CIS is a prominent area in IR that focuses on developing interactive knowledge assistants. These systems must adeptly comprehend the user's information requirements within the conversational context and retrieve the relevant information. To this aim, the existing approaches model the user's information needs with one query called rewritten query and use this query for passage retrieval. In this paper, we propose three different methods for generating multiple queries to enhance the retrieval. In these methods, we leverage the capabilities of large language models (LLMs) in understanding the user's information need and generating an appropriate response, to generate multiple queries. We implement and evaluate the proposed models utilizing various LLMs including GPT-4 and Llama-2 chat in zero-shot and few-shot settings. In addition, we propose a new benchmark for TREC iKAT based on gpt 3.5 judgments. Our experiments reveal the effectivenes
    
[^2]: ClickPrompt: CTR模型是将语言模型适应为CTR预测的强大提示生成器

    ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction. (arXiv:2310.09234v1 [cs.IR])

    [http://arxiv.org/abs/2310.09234](http://arxiv.org/abs/2310.09234)

    这篇论文提出了一个新颖的模型，旨在同时模拟语义和协同知识，以实现准确的CTR估计，并解决推理效率问题。

    

    点击率（CTR）预测已经成为各种互联网应用程序中越来越不可或缺的。传统的CTR模型通过独热编码将多字段分类数据转换为ID特征，并提取特征之间的协同信号。这种范式的问题在于语义信息的丢失。另一方面的研究通过将输入数据转换为文本句子来探索预训练语言模型（PLM）在CTR预测中的潜力。虽然语义信号得到了保留，但它们通常无法捕捉到协同信息（如特征交互、纯ID特征），更不用说由庞大的模型大小带来的无法接受的推理开销了。在本文中，我们旨在为准确的CTR估计建立语义知识和协同知识，并解决推理效率问题。为了从两个领域中受益并弥合它们之间的差距，我们提出了一种新颖的模型-。

    Click-through rate (CTR) prediction has become increasingly indispensable for various Internet applications. Traditional CTR models convert the multi-field categorical data into ID features via one-hot encoding, and extract the collaborative signals among features. Such a paradigm suffers from the problem of semantic information loss. Another line of research explores the potential of pretrained language models (PLMs) for CTR prediction by converting input data into textual sentences through hard prompt templates. Although semantic signals are preserved, they generally fail to capture the collaborative information (e.g., feature interactions, pure ID features), not to mention the unacceptable inference overhead brought by the huge model size. In this paper, we aim to model both the semantic knowledge and collaborative knowledge for accurate CTR estimation, and meanwhile address the inference inefficiency issue. To benefit from both worlds and close their gaps, we propose a novel model-
    

