# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsupervised LLM Adaptation for Question Answering](https://arxiv.org/abs/2402.12170) | 提出了无监督LLM适应问答任务，通过利用预训练的LLM和目标领域的未标记文档，实现在新领域回答问题的目标。 |
| [^2] | [Exploring Value Biases: How LLMs Deviate Towards the Ideal](https://arxiv.org/abs/2402.11005) | 研究发现大型语言模型（LLMs）在给出响应时存在一个价值偏好的机制，倾向于偏向理想状态，这种偏差会对不同应用场景产生重要影响。 |

# 详细

[^1]: 无监督LLM适应问答任务

    Unsupervised LLM Adaptation for Question Answering

    [https://arxiv.org/abs/2402.12170](https://arxiv.org/abs/2402.12170)

    提出了无监督LLM适应问答任务，通过利用预训练的LLM和目标领域的未标记文档，实现在新领域回答问题的目标。

    

    大型语言模型（LLM）通过自监督训练学习大规模训练数据集中的多样化知识。接着通过指导微调，LLM能够返回多样问题的正确信息。然而，将这些预训练的LLM调整到新的目标领域，如不同组织或时期，用于问答任务会产生很高的注释成本。为解决这一挑战，我们提出了一个新颖的任务，即无监督LLM适应问答任务。在这个任务中，我们利用预训练的LLM、一个公开可用的问答数据集（源数据）和目标域的未标记文档。我们的目标是学习LLM，使其能够回答关于目标领域的问题。我们引入了一个合成数据集和两个真实数据集来评估在源数据和目标数据上微调的模型，并揭示了一些有趣的见解；（i）微调模型展示了提供正确答案的能力

    arXiv:2402.12170v1 Announce Type: cross  Abstract: Large language models (LLM) learn diverse knowledge present in the large-scale training dataset via self-supervised training. Followed by instruction-tuning, LLM acquires the ability to return correct information for diverse questions. However, adapting these pre-trained LLMs to new target domains, such as different organizations or periods, for the question-answering (QA) task incurs a substantial annotation cost. To tackle this challenge, we propose a novel task, unsupervised LLM adaptation for question answering. In this task, we leverage a pre-trained LLM, a publicly available QA dataset (source data), and unlabeled documents from the target domain. Our goal is to learn LLM that can answer questions about the target domain. We introduce one synthetic and two real datasets to evaluate models fine-tuned on the source and target data, and reveal intriguing insights; (i) fine-tuned models exhibit the ability to provide correct answers 
    
[^2]: 探究价值偏好：LLMs偏向理想状态的偏差

    Exploring Value Biases: How LLMs Deviate Towards the Ideal

    [https://arxiv.org/abs/2402.11005](https://arxiv.org/abs/2402.11005)

    研究发现大型语言模型（LLMs）在给出响应时存在一个价值偏好的机制，倾向于偏向理想状态，这种偏差会对不同应用场景产生重要影响。

    

    大型语言模型（LLMs）被部署在各种应用中，并且它们的响应对社会产生着越来越大的影响。理解LLMs在给出响应时的非故意机制对于解释它们的性能并辨别它们在现实世界应用中的偏差至关重要。这类似于人类研究中，这种无意识的响应被称为抽样。我们研究了LLMs的这种抽样现象，发现LLMs的抽样倾向于偏爱高价值选项。价值偏好对应于从最可能的响应向LLM中代表的理想价值的转变。实际上，即便是通过上下文提示学习到的新实体，这种效果也能够再现。我们表明这种偏差表现在意想不到的地方，并对选择典型实例等相关应用场景产生影响。结果显示，价值偏好在不同分类的LLMs中都很明显。

    arXiv:2402.11005v1 Announce Type: cross  Abstract: Large-Language-Models (LLMs) are deployed in a wide range of applications, and their response has an increasing social impact. Understanding the non-deliberate(ive) mechanism of LLMs in giving responses is essential in explaining their performance and discerning their biases in real-world applications. This is analogous to human studies, where such inadvertent responses are referred to as sampling. We study this sampling of LLMs in light of value bias and show that the sampling of LLMs tends to favour high-value options. Value bias corresponds to this shift of response from the most likely towards an ideal value represented in the LLM. In fact, this effect can be reproduced even with new entities learnt via in-context prompting. We show that this bias manifests in unexpected places and has implications on relevant application scenarios, like choosing exemplars. The results show that value bias is strong in LLMs across different categor
    

