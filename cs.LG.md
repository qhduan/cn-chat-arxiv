# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews](https://arxiv.org/abs/2403.07183) | 该研究提出了一种估计大语料库中被大语言模型大幅修改的文本比例的方法，并在AI会议的同行评审中进行了实证分析，发现6.5%至16.9%的文本可能被LLMs大幅修改，揭示了用户行为的一些见解。 |
| [^2] | [Federated Learning with Nonvacuous Generalisation Bounds.](http://arxiv.org/abs/2310.11203) | 这项研究提出了一种新的策略来在联邦学习中训练随机预测器，通过保护每个节点的隐私并且具有数值上非空的泛化界限，可以在保持预测性能的同时实现数据共享和保护隐私。 |

# 详细

[^1]: 在规模上监测AI修改的内容：AI会议同行评审中ChatGPT影响的案例研究

    Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews

    [https://arxiv.org/abs/2403.07183](https://arxiv.org/abs/2403.07183)

    该研究提出了一种估计大语料库中被大语言模型大幅修改的文本比例的方法，并在AI会议的同行评审中进行了实证分析，发现6.5%至16.9%的文本可能被LLMs大幅修改，揭示了用户行为的一些见解。

    

    我们提出了一种估计大语料库中文本可能被大语言模型（LLM）大幅修改或生成的部分比例的方法。我们的最大似然模型利用专家撰写和AI生成的参考文本，准确高效地检查语料库级别上真实世界LLM使用。我们将这种方法应用于AI会议上科学同行评审的案例研究，该研究发生在ChatGPT发布之后，包括ICLR 2024、NeurIPS 2023、CoRL 2023和EMNLP 2023。我们的研究结果表明，在这些会议提交的同行评审中，6.5%至16.9%的文本可能是由LLMs大幅修改的，即超出拼写检查或小幅更新的范围。生成文本出现的情况为用户行为提供了见解：在报告信心较低、在截止日期前提交的评论以及从评论公司

    arXiv:2403.07183v1 Announce Type: cross  Abstract: We present an approach for estimating the fraction of text in a large corpus which is likely to be substantially modified or produced by a large language model (LLM). Our maximum likelihood model leverages expert-written and AI-generated reference texts to accurately and efficiently examine real-world LLM-use at the corpus level. We apply this approach to a case study of scientific peer review in AI conferences that took place after the release of ChatGPT: ICLR 2024, NeurIPS 2023, CoRL 2023 and EMNLP 2023. Our results suggest that between 6.5% and 16.9% of text submitted as peer reviews to these conferences could have been substantially modified by LLMs, i.e. beyond spell-checking or minor writing updates. The circumstances in which generated text occurs offer insight into user behavior: the estimated fraction of LLM-generated text is higher in reviews which report lower confidence, were submitted close to the deadline, and from review
    
[^2]: 具有非空泛化界限的联邦学习

    Federated Learning with Nonvacuous Generalisation Bounds. (arXiv:2310.11203v1 [cs.LG])

    [http://arxiv.org/abs/2310.11203](http://arxiv.org/abs/2310.11203)

    这项研究提出了一种新的策略来在联邦学习中训练随机预测器，通过保护每个节点的隐私并且具有数值上非空的泛化界限，可以在保持预测性能的同时实现数据共享和保护隐私。

    

    我们引入了一种新的策略来训练联邦学习中的随机预测器，在这种策略中，网络的每个节点通过发布本地预测器但对其他节点保密其训练数据集的方式来保护其隐私。然后，我们构建一个全局的随机预测器，它在PAC-Bayesian泛化界限的意义上继承了本地私有预测器的属性。我们考虑了同步情况，即所有节点共享相同的训练目标（从泛化界限导出），以及异步情况，即每个节点可以有自己的个性化训练目标。通过一系列的数值实验，我们证明了我们的方法实现了与将所有数据集共享给所有节点的批处理方法相当的预测性能。此外，这些预测器支持着在保护每个节点隐私的同时具有数值上非空的泛化界限。我们明确地计算了预测性能的增量。

    We introduce a novel strategy to train randomised predictors in federated learning, where each node of the network aims at preserving its privacy by releasing a local predictor but keeping secret its training dataset with respect to the other nodes. We then build a global randomised predictor which inherits the properties of the local private predictors in the sense of a PAC-Bayesian generalisation bound. We consider the synchronous case where all nodes share the same training objective (derived from a generalisation bound), and the asynchronous case where each node may have its own personalised training objective. We show through a series of numerical experiments that our approach achieves a comparable predictive performance to that of the batch approach where all datasets are shared across nodes. Moreover the predictors are supported by numerically nonvacuous generalisation bounds while preserving privacy for each node. We explicitly compute the increment on predictive performance an
    

