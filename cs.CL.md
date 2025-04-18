# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Citation-Enhanced Generation for LLM-based Chatbot](https://arxiv.org/abs/2402.16063) | 提出一种基于引文增强的LLM聊天机器人生成方法，采用检索模块搜索支持文档来解决幻觉内容产生的问题。 |
| [^2] | [Why Lift so Heavy? Slimming Large Language Models by Cutting Off the Layers](https://arxiv.org/abs/2402.11700) | 减少大型语言模型的层数可在不损失性能的情况下减轻模型规模，甚至在某些情况下只有一个层的模型可以超越完全层式的对应项。 |
| [^3] | [Multi-Step Deductive Reasoning Over Natural Language: An Empirical Study on Out-of-Distribution Generalisation](https://arxiv.org/abs/2207.14000) | 提出了IMA-GloVe-GA，一个用于自然语言表达的多步推理的迭代神经推理网络，在超领域泛化方面具有更好的性能表现。 |
| [^4] | [Baichuan 2: Open Large-scale Language Models.](http://arxiv.org/abs/2309.10305) | Baichuan 2是一系列开放的大规模多语言模型，拥有70亿和130亿个参数，训练自26万亿个标记。Baichuan 2在公开基准测试中表现出色，并在垂直领域如医学和法律中具有优势。 |

# 详细

[^1]: 基于引文增强的LLM聊天机器人生成

    Citation-Enhanced Generation for LLM-based Chatbot

    [https://arxiv.org/abs/2402.16063](https://arxiv.org/abs/2402.16063)

    提出一种基于引文增强的LLM聊天机器人生成方法，采用检索模块搜索支持文档来解决幻觉内容产生的问题。

    

    大型语言模型（LLMs）在各种情景下展现出强大的通用智能，包括将它们集成到聊天机器人中。然而，基于LLM的聊天机器人面临的一个重要挑战是在回复中可能产生虚构内容，这严重限制了它们的适用性。本文提出了一种新颖的后续引用增强生成（CEG）方法，结合检索论证。与先前侧重于预防生成过程中幻觉的研究不同，我们的方法以后续方式解决了这个问题。它结合了一个检索模块来搜索与生成内容相关的支持文档，并采用基于自然语言推理的方法。

    arXiv:2402.16063v1 Announce Type: cross  Abstract: Large language models (LLMs) exhibit powerful general intelligence across diverse scenarios, including their integration into chatbots. However, a vital challenge of LLM-based chatbots is that they may produce hallucinated content in responses, which significantly limits their applicability. Various efforts have been made to alleviate hallucination, such as retrieval augmented generation and reinforcement learning with human feedback, but most of them require additional training and data annotation. In this paper, we propose a novel post-hoc \textbf{C}itation-\textbf{E}nhanced \textbf{G}eneration (\textbf{CEG}) approach combined with retrieval argumentation. Unlike previous studies that focus on preventing hallucinations during generation, our method addresses this issue in a post-hoc way. It incorporates a retrieval module to search for supporting documents relevant to the generated content, and employs a natural language inference-ba
    
[^2]: 为什么要举得那么沉？通过修剪层来减轻大型语言模型

    Why Lift so Heavy? Slimming Large Language Models by Cutting Off the Layers

    [https://arxiv.org/abs/2402.11700](https://arxiv.org/abs/2402.11700)

    减少大型语言模型的层数可在不损失性能的情况下减轻模型规模，甚至在某些情况下只有一个层的模型可以超越完全层式的对应项。

    

    大型语言模型(LLMs)在处理各种自然语言处理(NLP)任务方面具有出色的能力。然而，这些模型的巨大规模在存储、训练和推理方面带来挑战，因为它们通过层叠包含了数十亿个参数。尽管传统方法如模型修剪或蒸馏为减小模型大小提供了途径，但往往会以性能保留为代价。在我们的调查中，我们系统地探讨了通过减少LLMs中的层数来减少模型规模的方法。令人惊讶的是，我们观察到，即使层数较少，LLMs在特别是基于提示的文本分类任务的微调中也能保持类似或更好的性能水平。值得注意的是，在某些情况下，只有一个层的模型可以胜过完全层式的对应项。这些发现为未来旨在减轻LLMs大小约束的工作提供了宝贵的见解。

    arXiv:2402.11700v1 Announce Type: new  Abstract: Large Language Models (LLMs) possess outstanding capabilities in addressing various natural language processing (NLP) tasks. However, the sheer size of these models poses challenges in terms of storage, training and inference due to the inclusion of billions of parameters through layer stacking. While traditional approaches such as model pruning or distillation offer ways for reducing model size, they often come at the expense of performance retention. In our investigation, we systematically explore the approach of reducing the number of layers in LLMs. Surprisingly, we observe that even with fewer layers, LLMs maintain similar or better performance levels, particularly in prompt-based fine-tuning for text classification tasks. Remarkably, in certain cases, models with a single layer outperform their fully layered counterparts. These findings offer valuable insights for future work aimed at mitigating the size constraints of LLMs while p
    
[^3]: 自然语言上的多步演绎推理：基于超领域泛化的实证研究

    Multi-Step Deductive Reasoning Over Natural Language: An Empirical Study on Out-of-Distribution Generalisation

    [https://arxiv.org/abs/2207.14000](https://arxiv.org/abs/2207.14000)

    提出了IMA-GloVe-GA，一个用于自然语言表达的多步推理的迭代神经推理网络，在超领域泛化方面具有更好的性能表现。

    

    将深度学习与符号逻辑推理结合起来，旨在充分利用这两个领域的成功，并引起了越来越多的关注。受DeepLogic启发，该模型经过端到端训练，用于执行逻辑程序推理，我们介绍了IMA-GloVe-GA，这是一个用自然语言表达的多步推理的迭代神经推理网络。在我们的模型中，推理是使用基于RNN的迭代内存神经网络进行的，其中包含一个门关注机制。我们在PARARULES、CONCEPTRULES V1和CONCEPTRULES V2三个数据集上评估了IMA-GloVe-GA。实验结果表明，带有门关注机制的DeepLogic比DeepLogic和其他RNN基线模型能够实现更高的测试准确性。我们的模型在规则被打乱时比RoBERTa-Large实现了更好的超领域泛化性能。此外，为了解决当前多步推理数据集中推理深度不平衡的问题

    arXiv:2207.14000v2 Announce Type: replace-cross  Abstract: Combining deep learning with symbolic logic reasoning aims to capitalize on the success of both fields and is drawing increasing attention. Inspired by DeepLogic, an end-to-end model trained to perform inference on logic programs, we introduce IMA-GloVe-GA, an iterative neural inference network for multi-step reasoning expressed in natural language. In our model, reasoning is performed using an iterative memory neural network based on RNN with a gate attention mechanism. We evaluate IMA-GloVe-GA on three datasets: PARARULES, CONCEPTRULES V1 and CONCEPTRULES V2. Experimental results show DeepLogic with gate attention can achieve higher test accuracy than DeepLogic and other RNN baseline models. Our model achieves better out-of-distribution generalisation than RoBERTa-Large when the rules have been shuffled. Furthermore, to address the issue of unbalanced distribution of reasoning depths in the current multi-step reasoning datase
    
[^4]: Baichuan 2: 开放的大规模语言模型

    Baichuan 2: Open Large-scale Language Models. (arXiv:2309.10305v1 [cs.CL])

    [http://arxiv.org/abs/2309.10305](http://arxiv.org/abs/2309.10305)

    Baichuan 2是一系列开放的大规模多语言模型，拥有70亿和130亿个参数，训练自26万亿个标记。Baichuan 2在公开基准测试中表现出色，并在垂直领域如医学和法律中具有优势。

    

    大型语言模型（LLMs）在仅有少量自然语言指令示例的情况下，已经在各种自然语言任务中展示出了令人瞩目的性能，减少了对广泛特征工程的需求。然而，大多数强大的LLMs是封闭源代码的，或者在除了英语以外的其他语言方面能力有限。在这篇技术报告中，我们介绍了Baichuan 2系列，这是一系列从头开始进行训练的大规模多语言模型，包含70亿和130亿个参数，使用26万亿个标记进行训练。Baichuan 2在MMLU、CMMLU、GSM8K和HumanEval等公开基准测试中与其他相同规模的开源模型相匹配或胜过。此外，Baichuan 2在医学和法律等垂直领域表现出色。我们将发布所有预训练模型检查点，以使研究界更好地理解Baichuan 2的训练动态。

    Large language models (LLMs) have demonstrated remarkable performance on a variety of natural language tasks based on just a few examples of natural language instructions, reducing the need for extensive feature engineering. However, most powerful LLMs are closed-source or limited in their capability for languages other than English. In this technical report, we present Baichuan 2, a series of large-scale multilingual language models containing 7 billion and 13 billion parameters, trained from scratch, on 2.6 trillion tokens. Baichuan 2 matches or outperforms other open-source models of similar size on public benchmarks like MMLU, CMMLU, GSM8K, and HumanEval. Furthermore, Baichuan 2 excels in vertical domains such as medicine and law. We will release all pre-training model checkpoints to benefit the research community in better understanding the training dynamics of Baichuan 2.
    

