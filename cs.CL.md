# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Evaluation Models from Large Language Models for Sequence Generation.](http://arxiv.org/abs/2308.04386) | 本文提出了一种评估能力转移方法（ECT），可以将大型语言模型的评估能力转移到相对轻量级的语言模型上，提高序列生成模型的性能。 |

# 详细

[^1]: 从大型语言模型中学习评估模型，用于序列生成

    Learning Evaluation Models from Large Language Models for Sequence Generation. (arXiv:2308.04386v1 [cs.CL])

    [http://arxiv.org/abs/2308.04386](http://arxiv.org/abs/2308.04386)

    本文提出了一种评估能力转移方法（ECT），可以将大型语言模型的评估能力转移到相对轻量级的语言模型上，提高序列生成模型的性能。

    

    大型语言模型在序列生成评估方面表现出最先进的性能，但通常具有大量的参数。这是一个计算挑战，因为在大规模应用它们的评估能力时会带来计算问题。为了克服这个挑战，本文提出了名为ECT的评估能力转移方法，将评估能力从大型语言模型转移到相对轻量级的语言模型上。基于所提出的ECT，我们从ChatGPT中学习了各种评估模型，并将它们作为奖励模型通过强化学习和重新排序方法来改进序列生成模型。在机器翻译、文本风格转换和摘要任务上的实验结果证明了我们的ECT的有效性。值得注意的是，将学习到的评估模型应用于序列生成模型会产生更好的生成序列，这是通过常用的度量和ChatGPT进行评估的。

    Large language models achieve state-of-the-art performance on sequence generation evaluation, but typically have a large number of parameters. This is a computational challenge as presented by applying their evaluation capability at scale. To overcome the challenge, in this paper, we propose \textbf{ECT}, an \textbf{e}valuation \textbf{c}apability \textbf{t}ransfer method, to transfer the evaluation capability from LLMs to relatively lightweight language models. Based on the proposed ECT, we learn various evaluation models from ChatGPT, and employ them as reward models to improve sequence generation models via reinforcement learning and reranking approaches. Experimental results on machine translation, text style transfer, and summarization tasks demonstrate the effectiveness of our ECT. Notably, applying the learned evaluation models to sequence generation models results in better generated sequences as evaluated by commonly used metrics and ChatGPT.
    

