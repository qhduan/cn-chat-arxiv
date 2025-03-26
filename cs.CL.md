# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Right for Right Reasons: Large Language Models for Verifiable Commonsense Knowledge Graph Question Answering](https://arxiv.org/abs/2403.01390) | LLM-based KGQA methods struggle with hallucination on commonsense reasoning questions, hindering their applicability in real-world applications. |
| [^2] | [Frustratingly Simple Memory Efficiency for Pre-trained Language Models via Dynamic Embedding Pruning.](http://arxiv.org/abs/2309.08708) | 该论文提出了一种简单但有效的方法，通过动态嵌入剪枝来减小预训练语言模型的内存占用。该方法在各种模型和任务中都能显著降低内存使用量，同时保持相当的下游任务性能，实现更高效地利用计算资源。 |
| [^3] | [Learning Evaluation Models from Large Language Models for Sequence Generation.](http://arxiv.org/abs/2308.04386) | 本文提出了一种评估能力转移方法（ECT），可以将大型语言模型的评估能力转移到相对轻量级的语言模型上，提高序列生成模型的性能。 |

# 详细

[^1]: 正当且充分：可验证的常识知识图问题回答中的大型语言模型

    Right for Right Reasons: Large Language Models for Verifiable Commonsense Knowledge Graph Question Answering

    [https://arxiv.org/abs/2403.01390](https://arxiv.org/abs/2403.01390)

    LLM-based KGQA methods struggle with hallucination on commonsense reasoning questions, hindering their applicability in real-world applications.

    

    知识图问题回答（KGQA）方法旨在利用知识图中存储的关系信息来回答自然语言问题。随着大型语言模型（LLMs）的最新进展及其出色的推理能力，利用它们进行KGQA的趋势日益增长。然而，现有方法仅专注于回答事实性问题，例如“Silvio Berlusconi的第一任妻子出生在哪座城市？”，而忽略了涉及常识推理的问题，这是现实世界用户可能更经常提出的，例如“我需要单独的签证才能看到威伦多夫的维纳斯并参加今年夏天的奥运会吗？”。在这项工作中，我们首先观察到，现有基于LLM的KGQA方法在处理这类问题时难以产生真实的答案，尤其是对针对长尾实体的查询（例如非主流和最近的实体），从而阻碍了它们在现实世界应用中的可应用性。

    arXiv:2403.01390v1 Announce Type: new  Abstract: Knowledge Graph Question Answering (KGQA) methods seek to answer Natural Language questions using the relational information stored in Knowledge Graphs (KGs). With the recent advancements of Large Language Models (LLMs) and their remarkable reasoning abilities, there is a growing trend to leverage them for KGQA. However, existing methodologies have only focused on answering factual questions, e.g., "In which city was Silvio Berlusconi's first wife born?", leaving questions involving commonsense reasoning that real-world users may pose more often, e.g., "Do I need separate visas to see the Venus of Willendorf and attend the Olympics this summer?" unaddressed. In this work, we first observe that existing LLM-based methods for KGQA struggle with hallucination on such questions, especially on queries targeting long-tail entities (e.g., non-mainstream and recent entities), thus hindering their applicability in real-world applications especial
    
[^2]: 经由动态嵌入剪枝实现的预训练语言模型的令人沮丧地简单的内存效率

    Frustratingly Simple Memory Efficiency for Pre-trained Language Models via Dynamic Embedding Pruning. (arXiv:2309.08708v1 [cs.CL])

    [http://arxiv.org/abs/2309.08708](http://arxiv.org/abs/2309.08708)

    该论文提出了一种简单但有效的方法，通过动态嵌入剪枝来减小预训练语言模型的内存占用。该方法在各种模型和任务中都能显著降低内存使用量，同时保持相当的下游任务性能，实现更高效地利用计算资源。

    

    预训练语言模型（PLMs）的广泛内存占用会阻碍其在内存受限环境（如云环境或设备上）的部署。 PLMs使用嵌入矩阵来表示广泛的词汇，构成了模型参数的大部分。尽管之前的工作已经考虑了在Transformer层内剪枝参数以提高参数效率，但在微调或推理过程中剪枝嵌入矩阵尚未被探索。我们首先证明了在这些情况下有一个显著比例的词汇未被使用。然后，我们提出了一个简单而有效的方法，利用这一发现来最小化嵌入矩阵的内存占用。我们展示了这种方法在各种模型和任务中都能显著降低内存使用量。值得注意的是，我们的方法在保持下游任务性能的同时允许更高效地使用计算资源。

    The extensive memory footprint of pre-trained language models (PLMs) can hinder deployment in memory-constrained settings, such as cloud environments or on-device. PLMs use embedding matrices to represent extensive vocabularies, forming a large proportion of the model parameters. While previous work towards parameter-efficient PLM development has considered pruning parameters within the transformer layers, pruning the embedding matrix as part of fine-tuning or inference has yet to be explored. We first demonstrate that a significant proportion of the vocabulary remains unused in these scenarios. We then propose a simple yet effective approach that leverages this finding to minimize the memory footprint of the embedding matrix. We show that this approach provides substantial reductions in memory usage across a wide range of models and tasks. Notably, our approach maintains equivalent downstream task performance while allowing a more efficient use of compute resources.
    
[^3]: 从大型语言模型中学习评估模型，用于序列生成

    Learning Evaluation Models from Large Language Models for Sequence Generation. (arXiv:2308.04386v1 [cs.CL])

    [http://arxiv.org/abs/2308.04386](http://arxiv.org/abs/2308.04386)

    本文提出了一种评估能力转移方法（ECT），可以将大型语言模型的评估能力转移到相对轻量级的语言模型上，提高序列生成模型的性能。

    

    大型语言模型在序列生成评估方面表现出最先进的性能，但通常具有大量的参数。这是一个计算挑战，因为在大规模应用它们的评估能力时会带来计算问题。为了克服这个挑战，本文提出了名为ECT的评估能力转移方法，将评估能力从大型语言模型转移到相对轻量级的语言模型上。基于所提出的ECT，我们从ChatGPT中学习了各种评估模型，并将它们作为奖励模型通过强化学习和重新排序方法来改进序列生成模型。在机器翻译、文本风格转换和摘要任务上的实验结果证明了我们的ECT的有效性。值得注意的是，将学习到的评估模型应用于序列生成模型会产生更好的生成序列，这是通过常用的度量和ChatGPT进行评估的。

    Large language models achieve state-of-the-art performance on sequence generation evaluation, but typically have a large number of parameters. This is a computational challenge as presented by applying their evaluation capability at scale. To overcome the challenge, in this paper, we propose \textbf{ECT}, an \textbf{e}valuation \textbf{c}apability \textbf{t}ransfer method, to transfer the evaluation capability from LLMs to relatively lightweight language models. Based on the proposed ECT, we learn various evaluation models from ChatGPT, and employ them as reward models to improve sequence generation models via reinforcement learning and reranking approaches. Experimental results on machine translation, text style transfer, and summarization tasks demonstrate the effectiveness of our ECT. Notably, applying the learned evaluation models to sequence generation models results in better generated sequences as evaluated by commonly used metrics and ChatGPT.
    

