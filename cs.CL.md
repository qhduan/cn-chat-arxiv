# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks](https://arxiv.org/abs/2403.04783) | 提出了一种基于响应过滤的多Agent防御框架AutoDefense，可以有效提高LLMs对抗越狱攻击的鲁棒性，同时保持正常用户请求的性能。 |
| [^2] | [Can LLMs Recognize Toxicity? Structured Toxicity Investigation Framework and Semantic-Based Metric](https://arxiv.org/abs/2402.06900) | 本研究提出了一种基于大型语言模型（LLMs）的自动度量方法，用于识别生成文本中的毒性。通过分析毒性因素和LLMs的内在毒性属性，该方法在测量毒性方面表现出众，比现有指标提升12个百分点。 |
| [^3] | [Language Models Understand Numbers, at Least Partially](https://arxiv.org/abs/2401.03735) | 本研究表明，大型语言模型在某种程度上理解数字，可以通过压缩和编码的方式执行算术计算。 |
| [^4] | [Unsupervised Summarization Re-ranking.](http://arxiv.org/abs/2212.09593) | 该论文提出了一种无监督的摘要再排序方法，可以将无监督模型的摘要表现提高，缩小其与有监督模型之间的性能差距。 |

# 详细

[^1]: AutoDefense: 多Agent LLM 防御对抗越狱攻击

    AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks

    [https://arxiv.org/abs/2403.04783](https://arxiv.org/abs/2403.04783)

    提出了一种基于响应过滤的多Agent防御框架AutoDefense，可以有效提高LLMs对抗越狱攻击的鲁棒性，同时保持正常用户请求的性能。

    

    尽管在道德对齐方面进行了广泛的预训练和微调以防止在用户请求时生成有害信息，但大型语言模型（LLMs）仍然容易受到越狱攻击。 本文提出了一种基于响应过滤的多Agent防御框架AutoDefense，用于从LLMs中过滤有害回复。 此框架为LLM代理分配不同角色，并利用它们共同完成防御任务。 任务的划分增强了LLMs的整体遵循指令能力，并使其他防御组件作为工具集成成为可能。 AutoDefense 可以适应各种规模和种类的开源LLMs作为代理。 通过对大量有害和安全提示进行广泛实验，我们验证了所提出的AutoDefense在提高对抗越狱攻击的鲁棒性的同时，保持了正常用户请求的性能。

    arXiv:2403.04783v1 Announce Type: cross  Abstract: Despite extensive pre-training and fine-tuning in moral alignment to prevent generating harmful information at user request, large language models (LLMs) remain vulnerable to jailbreak attacks. In this paper, we propose AutoDefense, a response-filtering based multi-agent defense framework that filters harmful responses from LLMs. This framework assigns different roles to LLM agents and employs them to complete the defense task collaboratively. The division in tasks enhances the overall instruction-following of LLMs and enables the integration of other defense components as tools. AutoDefense can adapt to various sizes and kinds of open-source LLMs that serve as agents. Through conducting extensive experiments on a large scale of harmful and safe prompts, we validate the effectiveness of the proposed AutoDefense in improving the robustness against jailbreak attacks, while maintaining the performance at normal user request. Our code and 
    
[^2]: LLM能够识别毒性吗？结构化毒性调查框架和基于语义的度量

    Can LLMs Recognize Toxicity? Structured Toxicity Investigation Framework and Semantic-Based Metric

    [https://arxiv.org/abs/2402.06900](https://arxiv.org/abs/2402.06900)

    本研究提出了一种基于大型语言模型（LLMs）的自动度量方法，用于识别生成文本中的毒性。通过分析毒性因素和LLMs的内在毒性属性，该方法在测量毒性方面表现出众，比现有指标提升12个百分点。

    

    在开发遵守社会标准的大型语言模型（LLMs）的过程中，识别生成文本中的毒性存在至关重要。现有的大多数毒性度量依赖于在特定毒性数据集上训练的编码模型。然而，这些编码器容易受到分布外的问题的影响，并且依赖于数据集中所假定的毒性定义。本文介绍了一种基于LLMs的自动鲁棒度量，用于区分模型回应是否具有毒性。我们首先分析了毒性因素，然后研究了LLMs的内在毒性属性，以确定它们作为评估器的适用性。随后，我们对评估数据集上的度量指标LLMs As ToxiciTy Evaluators（LATTE）进行了评估。实证结果表明，在不进行训练过程的情况下，我们的度量在测量毒性方面表现出色，F1得分比现有技术指标提高了12个百分点。我们还展示了上游毒性对度量结果的影响。

    In the pursuit of developing Large Language Models (LLMs) that adhere to societal standards, it is imperative to discern the existence of toxicity in the generated text. The majority of existing toxicity metrics rely on encoder models trained on specific toxicity datasets. However, these encoders are susceptible to out-of-distribution (OOD) problems and depend on the definition of toxicity assumed in a dataset. In this paper, we introduce an automatic robust metric grounded on LLMs to distinguish whether model responses are toxic. We start by analyzing the toxicity factors, followed by examining the intrinsic toxic attributes of LLMs to ascertain their suitability as evaluators. Subsequently, we evaluate our metric, LLMs As ToxiciTy Evaluators (LATTE), on evaluation datasets.The empirical results indicate outstanding performance in measuring toxicity, improving upon state-of-the-art metrics by 12 points in F1 score without training procedure. We also show that upstream toxicity has an 
    
[^3]: 语言模型在某种程度上理解数字

    Language Models Understand Numbers, at Least Partially

    [https://arxiv.org/abs/2401.03735](https://arxiv.org/abs/2401.03735)

    本研究表明，大型语言模型在某种程度上理解数字，可以通过压缩和编码的方式执行算术计算。

    

    大型语言模型(LLMs)在各种任务中展现出令人印象深刻的能力，但其不透明的内部机制限制了它们在数学问题中的应用。在本文中，我们研究了一个基本问题：语言模型是否理解数字，数学中的基本元素。基于一个假设，即LLMs应该能够在其隐藏状态中压缩数字以解决数学问题，我们构建了一个合成数据集，包括加法问题，并利用线性探测器从隐藏状态中读取输入数字。实验结果支持LLMs中存在压缩的数字。然而，精确重建原始数字是困难的，表明压缩过程可能不是无损的。进一步的实验证明，LLMs可以利用编码的数字来执行算术计算，并且计算能力随模型大小的增加而扩展。我们的初步研究表明，LLMs在数字上展现出部分理解。

    Large language models (LLMs) have exhibited impressive competence in various tasks, but their opaque internal mechanisms hinder their use in mathematical problems. In this paper, we study a fundamental question: whether language models understand numbers, a basic element in math. Based on an assumption that LLMs should be capable of compressing numbers in their hidden states to solve mathematical problems, we construct a synthetic dataset comprising addition problems and utilize linear probes to read out input numbers from the hidden states. Experimental results support the existence of compressed numbers in LLMs. However, it is difficult to precisely reconstruct the original numbers, indicating that the compression process may not be lossless. Further experiments show that LLMs can utilize encoded numbers to perform arithmetic computations, and the computational ability scales up with the model size. Our preliminary research suggests that LLMs exhibit a partial understanding of number
    
[^4]: 无监督摘要再排序

    Unsupervised Summarization Re-ranking. (arXiv:2212.09593v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09593](http://arxiv.org/abs/2212.09593)

    该论文提出了一种无监督的摘要再排序方法，可以将无监督模型的摘要表现提高，缩小其与有监督模型之间的性能差距。

    

    随着任务特定的预训练目标的兴起，像PEGASUS这样的抽象摘要模型在下游摘要任务中提供了令人满意的零样本性能。然而，这些无监督模型的性能仍然明显落后于它们的有监督对应物。本文提出了一种无监督的摘要再排序方法，旨在缩小无监督和有监督模型之间的性能差距。我们的方法在四个被广泛采用的摘要基准测试中，将PEGASUS的相对平均ROUGE提高了最多7.27％，ChatGPT提高了最多6.86％；并且在30种零样本转移设置（在一个数据集上微调，另一个数据集上评估）中，平均获得了7.51％的相对增益（从XSum到WikiHow最高可达23.73％）。

    With the rise of task-specific pre-training objectives, abstractive summarization models like PEGASUS offer appealing zero-shot performance on downstream summarization tasks. However, the performance of such unsupervised models still lags significantly behind their supervised counterparts. Similarly to the supervised setup, we notice a very high variance in quality among summary candidates from these models while only one candidate is kept as the summary output. In this paper, we propose to re-rank summary candidates in an unsupervised manner, aiming to close the performance gap between unsupervised and supervised models. Our approach improves the unsupervised PEGASUS by up to 7.27% and ChatGPT by up to 6.86% relative mean ROUGE across four widely-adopted summarization benchmarks ; and achieves relative gains of 7.51% (up to 23.73% from XSum to WikiHow) averaged over 30 zero-shot transfer setups (finetuning on a dataset, evaluating on another).
    

