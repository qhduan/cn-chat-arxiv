# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Privacy-Preserving Prompt Tuning for Large Language Model Services.](http://arxiv.org/abs/2305.06212) | RAPT是一个提供隐私保证的大语言模型服务的提示调整框架，采用本地差分隐私设置和新颖的隐私化标记重建任务，并在多种任务中取得有竞争力的性能和良好的隐私保护效果。 |

# 详细

[^1]: 大语言模型服务的隐私保护提示调整

    Privacy-Preserving Prompt Tuning for Large Language Model Services. (arXiv:2305.06212v1 [cs.CL])

    [http://arxiv.org/abs/2305.06212](http://arxiv.org/abs/2305.06212)

    RAPT是一个提供隐私保证的大语言模型服务的提示调整框架，采用本地差分隐私设置和新颖的隐私化标记重建任务，并在多种任务中取得有竞争力的性能和良好的隐私保护效果。

    

    提示调整为用户在新兴的大语言模型服务场景下使用其私有数据自定义大语言模型(LLM)的有效方式。但是，私有数据的敏感性需要在LLM服务定制中保护隐私。基于提示调整，我们提出了一种名为隐私保护提示调整(RAPT)的框架，为LLM服务提供隐私保证。RAPT采用本地隐私设置，允许用户使用本地差分隐私对其数据进行本地化隐私处理。由于在直接训练隐私化数据的情况下，提示调整表现不佳，因此我们引入了一种新颖的隐私化标记重建任务，与下游任务一起进行培训，使LLM学习更好的任务相关表示。尽管我们的框架简单，但实验表明，RAPT在各种任务中均具有竞争力的性能，并提供抵御对手的隐私保证。

    Prompt tuning provides an efficient way for users to customize Large Language Models (LLMs) with their private data in the emerging LLM service scenario. However, the sensitive nature of private data brings the need for privacy preservation in LLM service customization. Based on prompt tuning, we propose Privacy-Preserving Prompt Tuning (RAPT), a framework that provides privacy guarantees for LLM services. \textsc{rapt} adopts a local privacy setting, allowing users to privatize their data locally with local differential privacy. As prompt tuning performs poorly when directly trained on privatized data, we introduce a novel privatized token reconstruction task that is trained jointly with the downstream task, allowing LLMs to learn better task-dependent representations. Despite the simplicity of our framework, experiments show that RAPT achieves competitive performance across tasks while providing privacy guarantees against adversaries.
    

