# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Poro 34B and the Blessing of Multilinguality](https://arxiv.org/abs/2404.01856) | 多语种训练的Poro 34B模型在芬兰语等小语种上取得了显著进展，并具有比现有模型更出色的能力。 |
| [^2] | [Learning Transfers over Several Programming Languages.](http://arxiv.org/abs/2310.16937) | 这篇论文研究了使用跨语言迁移学习提高编程语言模型性能的问题，并进行了广泛实验验证。该研究表明，跨语言迁移学习在编程语言领域具有潜力，可以帮助低资源语言的用户受益于大规模语言模型。 |

# 详细

[^1]: Poro 34B和多语种的祝福

    Poro 34B and the Blessing of Multilinguality

    [https://arxiv.org/abs/2404.01856](https://arxiv.org/abs/2404.01856)

    多语种训练的Poro 34B模型在芬兰语等小语种上取得了显著进展，并具有比现有模型更出色的能力。

    

    最先进大型语言模型的预训练现在需要数万亿字的文本，这比绝大多数语言可获得的文本数量多几个数量级。尽管包含多种语言的文本是获取更多预训练数据的明显方法，但多语种往往被视为一种诅咒，大多数模型训练工作仍然主要集中在个别大语种上。我们相信多语种可以是一种祝福，并且应该有可能通过多语种训练显著提高小语种的模型能力。在这项研究中，我们介绍了Poro 34B，这是一个在1万亿个芬兰语、英语和编程语言标记上进行训练的拥有340亿参数的模型，并证明了多语种训练方法可以产生一个模型，不仅在芬兰语的现有模型能力上取得了显著进展，而且在表现方面表现出色。

    arXiv:2404.01856v1 Announce Type: new  Abstract: The pretraining of state-of-the-art large language models now requires trillions of words of text, which is orders of magnitude more than available for the vast majority of languages. While including text in more than one language is an obvious way to acquire more pretraining data, multilinguality is often seen as a curse, and most model training efforts continue to focus near-exclusively on individual large languages. We believe that multilinguality can be a blessing and that it should be possible to substantially improve over the capabilities of monolingual models for small languages through multilingual training. In this study, we introduce Poro 34B, a 34 billion parameter model trained for 1 trillion tokens of Finnish, English, and programming languages, and demonstrate that a multilingual training approach can produce a model that not only substantially advances over the capabilities of existing models for Finnish, but also excels i
    
[^2]: 跨多种编程语言的学习转移

    Learning Transfers over Several Programming Languages. (arXiv:2310.16937v1 [cs.CL])

    [http://arxiv.org/abs/2310.16937](http://arxiv.org/abs/2310.16937)

    这篇论文研究了使用跨语言迁移学习提高编程语言模型性能的问题，并进行了广泛实验验证。该研究表明，跨语言迁移学习在编程语言领域具有潜力，可以帮助低资源语言的用户受益于大规模语言模型。

    

    大规模语言模型（LLM）在提高高资源编程语言开发者生产力方面近年来取得了显著的进展。这些模型使用两种类型的数据：大量的无标签代码样本用于预训练，相对较少的带标签代码样本用于微调或上下文学习。然而，许多编程语言是低资源的，缺乏大多数任务的带标签样本，甚至缺乏无标签样本。因此，低资源语言（例如遗留或新语言）的用户无法享受到LLM的好处。跨语言迁移学习使用源语言的数据来提高模型在目标语言上的性能。它在自然语言领域已经得到了广泛研究，但在编程语言领域却受到了很少关注。本文使用基于Transformer的LLM和11到41种编程语言进行了广泛的实验，探讨了以下问题。

    Large language models (LLMs) have recently become remarkably good at improving developer productivity for high-resource programming languages. These models use two kinds of data: large amounts of unlabeled code samples for pretraining and relatively smaller amounts of labeled code samples for fine-tuning or in-context learning. Unfortunately, many programming languages are low-resource, lacking labeled samples for most tasks and often even lacking unlabeled samples. Therefore, users of low-resource languages (e.g., legacy or new languages) miss out on the benefits of LLMs. Cross-lingual transfer learning uses data from a source language to improve model performance on a target language. It has been well-studied for natural languages, but has received little attention for programming languages. This paper reports extensive experiments on four tasks using a transformer-based LLM and 11 to 41 programming languages to explore the following questions. First, how well cross-lingual transfer 
    

