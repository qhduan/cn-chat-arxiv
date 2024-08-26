# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Distillation Contrastive Decoding: Improving LLMs Reasoning with Contrastive Decoding and Distillation](https://arxiv.org/abs/2402.14874) | 该研究提出了一种叫做蒸馏对比解码（DCD）的方法，通过结合对比提示与蒸馏技术，有效提升了大型语言模型（LLM）在推理任务上的性能表现，超过了传统的对比解码方法，并在多个基准数据集上取得了显著成果。 |
| [^2] | [Tuning Language Models by Proxy](https://arxiv.org/abs/2401.08565) | 介绍了一种代理调整的轻量级解码时算法，可以通过对小型调整后的LM的预测与未调整LM的预测之间的差异来调整大型预训练LM的预测，从而实现资源节约和保留更大规模预训练的好处。 |
| [^3] | [Model Merging by Uncertainty-Based Gradient Matching.](http://arxiv.org/abs/2310.12808) | 本论文通过不确定性梯度匹配的方法，提出了一种新的模型合并方案，该方案能够减少梯度不匹配，从而提高了模型合并的性能并对超参数更具鲁棒性。 |
| [^4] | [SPICED: News Similarity Detection Dataset with Multiple Topics and Complexity Levels.](http://arxiv.org/abs/2309.13080) | 这个论文提出了一个名为SPICED的新闻相似性检测数据集，包括七个主题，并提供了四种不同的方法来生成新闻。 |

# 详细

[^1]: 蒸馏对比解码：利用对比解码和蒸馏提升LLM的推理能力

    Distillation Contrastive Decoding: Improving LLMs Reasoning with Contrastive Decoding and Distillation

    [https://arxiv.org/abs/2402.14874](https://arxiv.org/abs/2402.14874)

    该研究提出了一种叫做蒸馏对比解码（DCD）的方法，通过结合对比提示与蒸馏技术，有效提升了大型语言模型（LLM）在推理任务上的性能表现，超过了传统的对比解码方法，并在多个基准数据集上取得了显著成果。

    

    我们提出了一种称为蒸馏对比解码（DCD）的简单方法，以增强大型语言模型（LLMs）在推理过程中的推理能力。与先前依赖于较小的业余模型或隐藏状态差异分析的方法不同，DCD采用了对比式思维引导和先进的蒸馏技术，包括Dropout和量化。这种方法有效地解决了对比解码（CD）的局限性，后者通常需要专家和业余模型，从而增加计算资源需求。通过将对比提示与蒸馏相结合，DCD消除了对业余模型的需求并减少了内存使用。我们的评估表明，DCD显著增强了LLM在各种推理基准测试中的性能，在GSM8K和StrategyQA数据集中均超过了CD和现有方法。

    arXiv:2402.14874v1 Announce Type: cross  Abstract: We propose a straightforward approach called Distillation Contrastive Decoding (DCD) to enhance the reasoning capabilities of Large Language Models (LLMs) during inference. In contrast to previous approaches that relied on smaller amateur models or analysis of hidden state differences, DCD employs Contrastive Chain-of-thought Prompting and advanced distillation techniques, including Dropout and Quantization. This approach effectively addresses the limitations of Contrastive Decoding (CD), which typically requires both an expert and an amateur model, thus increasing computational resource demands. By integrating contrastive prompts with distillation, DCD obviates the need for an amateur model and reduces memory usage. Our evaluations demonstrate that DCD significantly enhances LLM performance across a range of reasoning benchmarks, surpassing both CD and existing methods in the GSM8K and StrategyQA datasets.
    
[^2]: 通过代理调整语言模型

    Tuning Language Models by Proxy

    [https://arxiv.org/abs/2401.08565](https://arxiv.org/abs/2401.08565)

    介绍了一种代理调整的轻量级解码时算法，可以通过对小型调整后的LM的预测与未调整LM的预测之间的差异来调整大型预训练LM的预测，从而实现资源节约和保留更大规模预训练的好处。

    

    尽管大型预训练语言模型具有一般的能力，但它们始终受益于进一步调整以更好地实现所需的行为。然而，调整这些模型变得越来越消耗资源，或者在模型权重是私有的情况下是不可能的。我们引入了代理调整，这是一种轻量级的解码时算法，它在黑盒语言模型的基础上运行，以实现与直接调整相同的目的，但只访问其在输出词汇上的预测，而不是其参数。我们的方法调整了一个较小的语言模型，然后将经过调整和未经调整的小模型的预测之间的差异应用于将更大的未调整模型的原始预测转移到调整方向，同时保留较大规模预训练的好处。在实验中，当我们使用仅为7B大小的代理对Llama2-70B应用代理调整时，我们可以关闭88% Llama2-70B 与其真正调整过的聊天版本之间的差距，

    arXiv:2401.08565v2 Announce Type: replace  Abstract: Despite the general capabilities of large pretrained language models, they consistently benefit from further adaptation to better achieve desired behaviors. However, tuning these models has become increasingly resource-intensive, or impossible when model weights are private. We introduce proxy-tuning, a lightweight decoding-time algorithm that operates on top of black-box LMs to achieve the same end as direct tuning, but by accessing only its predictions over the output vocabulary, not its parameters. Our method tunes a smaller LM, then applies the difference between the predictions of the small tuned and untuned LMs to shift the original predictions of the larger untuned model in the direction of tuning, while retaining the benefits of larger-scale pretraining. In experiments, when we apply proxy-tuning to Llama2-70B using proxies of only 7B size, we can close 88% of the gap between Llama2-70B and its truly-tuned chat version, when 
    
[^3]: 基于不确定性梯度匹配的模型合并

    Model Merging by Uncertainty-Based Gradient Matching. (arXiv:2310.12808v1 [cs.LG])

    [http://arxiv.org/abs/2310.12808](http://arxiv.org/abs/2310.12808)

    本论文通过不确定性梯度匹配的方法，提出了一种新的模型合并方案，该方案能够减少梯度不匹配，从而提高了模型合并的性能并对超参数更具鲁棒性。

    

    在不同数据集上训练的模型可以通过参数的加权平均来合并，但为什么会起作用，什么情况下会失败？在这里，我们将加权平均的不准确性与梯度不匹配联系起来，并提出了一种新的基于不确定性的方案，通过减少不匹配来提高性能。这种联系还揭示了其他方案（如平均值、任务算术和Fisher加权平均）中的隐含假设。我们的新方法在大型语言模型和视觉转换器方面都在性能和超参数鲁棒性方面得到了一致的改进。

    Models trained on different datasets can be merged by a weighted-averaging of their parameters, but why does it work and when can it fail? Here, we connect the inaccuracy of weighted-averaging to mismatches in the gradients and propose a new uncertainty-based scheme to improve the performance by reducing the mismatch. The connection also reveals implicit assumptions in other schemes such as averaging, task arithmetic, and Fisher-weighted averaging. Our new method gives consistent improvements for large language models and vision transformers, both in terms of performance and robustness to hyperparameters.
    
[^4]: SPICED: 具有多个主题和复杂程度的新闻相似性检测数据集

    SPICED: News Similarity Detection Dataset with Multiple Topics and Complexity Levels. (arXiv:2309.13080v1 [cs.CL])

    [http://arxiv.org/abs/2309.13080](http://arxiv.org/abs/2309.13080)

    这个论文提出了一个名为SPICED的新闻相似性检测数据集，包括七个主题，并提供了四种不同的方法来生成新闻。

    

    如今，使用智能系统来检测新闻文章中的冗余信息已经变得非常普遍，以增强用户体验，尤其是随着新闻媒体的蓬勃发展。然而，新闻的异质性可能导致这些系统中的虚假发现：简单的启发式算法，比如一对新闻是否都涉及政治问题，可以提供强大但具有误导性的下游性能。将新闻相似性数据集分割成主题可以通过强制模型学习如何在更狭窄的领域中区分显著特征来改进这些模型的训练。然而，这需要存在目前缺乏的专题特定数据集。在本文中，我们提出了一个新的相似新闻数据集SPICED，其中包括七个主题：犯罪与法律、文化与娱乐、灾难与事故、经济与商业、政治与冲突、科学与技术以及体育。此外，我们提供了四种不同的方法来生成新闻。

    Nowadays, the use of intelligent systems to detect redundant information in news articles has become especially prevalent with the proliferation of news media outlets in order to enhance user experience. However, the heterogeneous nature of news can lead to spurious findings in these systems: Simple heuristics such as whether a pair of news are both about politics can provide strong but deceptive downstream performance. Segmenting news similarity datasets into topics improves the training of these models by forcing them to learn how to distinguish salient characteristics under more narrow domains. However, this requires the existence of topic-specific datasets, which are currently lacking. In this article, we propose a new dataset of similar news, SPICED, which includes seven topics: Crime & Law, Culture & Entertainment, Disasters & Accidents, Economy & Business, Politics & Conflicts, Science & Technology, and Sports. Futhermore, we present four distinct approaches for generating news 
    

