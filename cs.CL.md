# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MuseGraph: Graph-oriented Instruction Tuning of Large Language Models for Generic Graph Mining](https://arxiv.org/abs/2403.04780) | MuseGraph将GNNs和LLMs的优势结合起来，提出了一种更有效和通用的图挖掘方法，可以跨不同任务和数据集使用 |
| [^2] | [Backdoor Attacks on Dense Passage Retrievers for Disseminating Misinformation](https://arxiv.org/abs/2402.13532) | 本文介绍了一种后门攻击场景，攻击者通过利用密集通道检索的语法错误触发后门攻击，以秘密传播定向错误信息，如仇恨言论或广告，并通过实验证明了这种攻击方法的有效性和隐匿性。 |
| [^3] | [Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models.](http://arxiv.org/abs/2308.15022) | 递归总结在大型语言模型中实现长期对话记忆，可以提高对话系统在长对话中记忆重要信息的能力。 |

# 详细

[^1]: MuseGraph：面向大型语言模型的图导向指令调整用于通用图挖掘

    MuseGraph: Graph-oriented Instruction Tuning of Large Language Models for Generic Graph Mining

    [https://arxiv.org/abs/2403.04780](https://arxiv.org/abs/2403.04780)

    MuseGraph将GNNs和LLMs的优势结合起来，提出了一种更有效和通用的图挖掘方法，可以跨不同任务和数据集使用

    

    具有丰富属性的图在建模互联实体和改进各种实际应用中的预测方面至关重要。传统图神经网络（GNNs）通常用于建模带属性的图，但需要在应用于不同图任务和数据集时进行重新训练。尽管大型语言模型（LLMs）的出现在自然语言处理中引入了新的范例，但LLMs在图挖掘中的生成潜力仍未得到充分探索。为此，我们提出了一个新颖的框架 MuseGraph，它无缝整合了GNNs和LLMs的优势，并促进了一种更有效和通用的图挖掘方法，可跨不同任务和数据集使用。具体而言，我们首先通过提出的自适应输入生成引入一个紧凑的图描述，以在语言令牌限制的约束下封装来自图的关键信息。

    arXiv:2403.04780v1 Announce Type: cross  Abstract: Graphs with abundant attributes are essential in modeling interconnected entities and improving predictions in various real-world applications. Traditional Graph Neural Networks (GNNs), which are commonly used for modeling attributed graphs, need to be re-trained every time when applied to different graph tasks and datasets. Although the emergence of Large Language Models (LLMs) has introduced a new paradigm in natural language processing, the generative potential of LLMs in graph mining remains largely under-explored. To this end, we propose a novel framework MuseGraph, which seamlessly integrates the strengths of GNNs and LLMs and facilitates a more effective and generic approach for graph mining across different tasks and datasets. Specifically, we first introduce a compact graph description via the proposed adaptive input generation to encapsulate key information from the graph under the constraints of language token limitations. T
    
[^2]: 密集通道检索器用于传播信息错误的后门攻击

    Backdoor Attacks on Dense Passage Retrievers for Disseminating Misinformation

    [https://arxiv.org/abs/2402.13532](https://arxiv.org/abs/2402.13532)

    本文介绍了一种后门攻击场景，攻击者通过利用密集通道检索的语法错误触发后门攻击，以秘密传播定向错误信息，如仇恨言论或广告，并通过实验证明了这种攻击方法的有效性和隐匿性。

    

    密集检索器和检索增强语言模型已广泛用于各种NLP应用，尽管设计用于提供可靠和安全的结果，但检索器对潜在攻击的脆弱性仍不清楚，引发人们对其安全性的关注。本文介绍了一种新颖的情景，攻击者旨在通过检索系统隐蔽传播定向错误信息，如仇恨言论或广告。为实现这一目标，我们提出了一种在密集通道检索中由语法错误触发的危险后门攻击。我们的方法确保被攻击的模型在标准查询下可以正常运行，但在用户在查询中意外地犯语法错误时，被篡改以返回攻击者指定的段落。大量实验展示了我们提出的攻击方法的有效性和隐蔽性。

    arXiv:2402.13532v1 Announce Type: new  Abstract: Dense retrievers and retrieval-augmented language models have been widely used in various NLP applications. Despite being designed to deliver reliable and secure outcomes, the vulnerability of retrievers to potential attacks remains unclear, raising concerns about their security. In this paper, we introduce a novel scenario where the attackers aim to covertly disseminate targeted misinformation, such as hate speech or advertisement, through a retrieval system. To achieve this, we propose a perilous backdoor attack triggered by grammar errors in dense passage retrieval. Our approach ensures that attacked models can function normally for standard queries but are manipulated to return passages specified by the attacker when users unintentionally make grammatical mistakes in their queries. Extensive experiments demonstrate the effectiveness and stealthiness of our proposed attack method. When a user query is error-free, our model consistentl
    
[^3]: 递归总结在大型语言模型中实现长期对话记忆

    Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models. (arXiv:2308.15022v1 [cs.CL])

    [http://arxiv.org/abs/2308.15022](http://arxiv.org/abs/2308.15022)

    递归总结在大型语言模型中实现长期对话记忆，可以提高对话系统在长对话中记忆重要信息的能力。

    

    大多数开放领域的对话系统在长期对话中容易遗忘重要信息。现有方法通常训练特定的检索器或总结器从过去获取关键信息，这需要耗费时间且高度依赖标记数据的质量。为了缓解这个问题，我们提出使用大型语言模型（LLMs）递归生成总结/记忆，以增强长期记忆能力。具体而言，我们的方法首先刺激LLMs记住小对话上下文，然后递归地使用之前的记忆和随后的对话内容产生新的记忆。最后，LLM可以在最新记忆的帮助下轻松生成高度一致的响应。我们使用ChatGPT和text-davinci-003进行评估，对广泛使用的公共数据集进行的实验证明我们的方法在长对话中可以生成更一致的响应。值得注意的是，我们的方法是实现LLM建模的潜在解决方案。

    Most open-domain dialogue systems suffer from forgetting important information, especially in a long-term conversation. Existing works usually train the specific retriever or summarizer to obtain key information from the past, which is time-consuming and highly depends on the quality of labeled data. To alleviate this problem, we propose to recursively generate summaries/ memory using large language models (LLMs) to enhance long-term memory ability. Specifically, our method first stimulates LLMs to memorize small dialogue contexts and then recursively produce new memory using previous memory and following contexts. Finally, the LLM can easily generate a highly consistent response with the help of the latest memory. We evaluate our method using ChatGPT and text-davinci-003, and the experiments on the widely-used public dataset show that our method can generate more consistent responses in a long-context conversation. Notably, our method is a potential solution to enable the LLM to model
    

