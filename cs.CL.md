# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DeFT: Flash Tree-attention with IO-Awareness for Efficient Tree-search-based LLM Inference](https://arxiv.org/abs/2404.00242) | DeFT提出了一种带IO意识的树注意力算法，通过在QKV准备和注意力计算阶段实现内存高效的计算，降低内存印记，以解决当前树解码策略和推断系统不适配的问题。 |
| [^2] | [Privacy-Aware Semantic Cache for Large Language Models](https://arxiv.org/abs/2403.02694) | MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。 |
| [^3] | [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668) | 提出了一种简单的线性注意力语言模型架构，可以平衡召回和内存消耗之间的权衡。 |
| [^4] | [A match made in consistency heaven: when large language models meet evolutionary algorithms.](http://arxiv.org/abs/2401.10510) | 大型语言模型和进化算法的结合具有强大的一致性，包括标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化等多个核心特征。本文分析了现有的耦合研究，并为未来的研究提供了基本路线和关键挑战。 |
| [^5] | [CAMELL: Confidence-based Acquisition Model for Efficient Self-supervised Active Learning with Label Validation.](http://arxiv.org/abs/2310.08944) | CAMELL是一个适用于序列多输出问题的主动学习框架，通过仅需专家标注序列的一小部分、自监督和标签验证机制来解决监督神经方法对大规模标注数据集的依赖限制。 |
| [^6] | [Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes.](http://arxiv.org/abs/2304.09433) | 本文探讨了使用大型语言模型在不需要特定领域的训练和定制下实现生成可查询表格的简单系统，并给出了两种实现策略，在不同质量和成本中平衡。通过实验证明，该系统对不同类型文档生成的表格均高质量，且无需文档特定的定制。 |

# 详细

[^1]: DeFT：带IO意识的Flash Tree-attention用于高效的基于树搜索的LLM推断

    DeFT: Flash Tree-attention with IO-Awareness for Efficient Tree-search-based LLM Inference

    [https://arxiv.org/abs/2404.00242](https://arxiv.org/abs/2404.00242)

    DeFT提出了一种带IO意识的树注意力算法，通过在QKV准备和注意力计算阶段实现内存高效的计算，降低内存印记，以解决当前树解码策略和推断系统不适配的问题。

    

    使用树搜索进行解码可以极大地提高基于变压器的大型语言模型（LLMs）的推断质量。根据引导信号，它通过形成LLM输出从根到叶子的最佳路径来提高可控性、推理能力、对齐等。然而，由于计算冗余、内存占用和内存访问，当前的树解码策略及其推断系统互相不适配，导致推断效率低下。为解决这一问题，我们提出了DeFT，一种IO感知树注意力算法，它在两个阶段中保持内存高效的注意力计算，降低内存印记：（1）QKV准备：我们提出了一种KV引导树分裂策略，为GPU的高利用率和尽可能减少GPU全局内存和芯片上共享内存之间的KV缓存的内存读/写; （2）注意力计算...

    arXiv:2404.00242v1 Announce Type: cross  Abstract: Decoding using tree search can greatly enhance the inference quality for transformer-based Large Language Models (LLMs). Depending on the guidance signal, it searches for the best path from root to leaf in the tree by forming LLM outputs to improve controllability, reasoning ability, alignment, et cetera. However, current tree decoding strategies and their inference systems do not suit each other well due to redundancy in computation, memory footprints, and memory access, resulting in inefficient inference. To address this issue, we propose DeFT, an IO-aware tree attention algorithm that maintains memory-efficient attention calculation with low memory footprints in two stages: (1) QKV Preparation: we propose a KV-Guided Tree Split strategy to group QKV wisely for high utilization of GPUs and reduction of memory reads/writes for the KV cache between GPU global memory and on-chip shared memory as much as possible; (2) Attention Calculati
    
[^2]: 面向大型语言模型的隐私感知语义缓存

    Privacy-Aware Semantic Cache for Large Language Models

    [https://arxiv.org/abs/2403.02694](https://arxiv.org/abs/2403.02694)

    MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。

    

    大型语言模型（LLMs）如ChatGPT、Google Bard、Claude和Llama 2彻底改变了自然语言处理和搜索引擎动态。然而，这些模型造成了异常高的计算成本。本文介绍了MeanCache，一种用于LLMs的语义缓存，它能够识别语义上相似的查询以确定缓存命中或未命中。

    arXiv:2403.02694v1 Announce Type: cross  Abstract: Large Language Models (LLMs) like ChatGPT, Google Bard, Claude, and Llama 2 have revolutionized natural language processing and search engine dynamics. However, these models incur exceptionally high computational costs. For instance, GPT-3 consists of 175 billion parameters and inference on these models also demands billions of floating-point operations. Caching is a natural solution to reduce LLM inference costs on repeated queries. However, existing caching methods are incapable of finding semantic similarities among LLM queries, leading to unacceptable false hit-and-miss rates.   This paper introduces MeanCache, a semantic cache for LLMs that identifies semantically similar queries to determine cache hit or miss. Using MeanCache, the response to a user's semantically similar query can be retrieved from a local cache rather than re-querying the LLM, thus reducing costs, service provider load, and environmental impact. MeanCache lever
    
[^3]: 简单的线性注意力语言模型平衡了召回-吞吐量的折衷

    Simple linear attention language models balance the recall-throughput tradeoff

    [https://arxiv.org/abs/2402.18668](https://arxiv.org/abs/2402.18668)

    提出了一种简单的线性注意力语言模型架构，可以平衡召回和内存消耗之间的权衡。

    

    最近的研究表明，基于注意力的语言模型擅长召回，即在上下文中已经看到的标记。然而，在推断过程中，基于注意力的模型的效率受到KV-cache的内存消耗的瓶颈限制。在这项工作中，我们探讨了是否可以提高语言模型的效率（例如通过减少内存消耗）而不影响召回。通过将实验和理论应用于广泛的架构，我们确定了模型状态大小和召回能力之间的一个关键权衡。我们发现，与注意力的高效替代方法（例如H3、Mamba、RWKV）保持固定大小的循环状态，但在召回方面表现不佳。我们提出了BASED，这是一种结合了线性和滑动窗口注意力的简单架构。通过改变BASED窗口大小和线性注意力特征维度，我们可以调整状态大小，并遍历召回-内存折衷的帕累托前沿。

    arXiv:2402.18668v1 Announce Type: new  Abstract: Recent work has shown that attention-based language models excel at recall, the ability to ground generations in tokens previously seen in context. However, the efficiency of attention-based models is bottle-necked during inference by the KV-cache's aggressive memory consumption. In this work, we explore whether we can improve language model efficiency (e.g. by reducing memory consumption) without compromising on recall. By applying experiments and theory to a broad set of architectures, we identify a key tradeoff between a model's state size and recall ability. We show that efficient alternatives to attention (e.g. H3, Mamba, RWKV) maintain a fixed-size recurrent state, but struggle at recall. We propose BASED a simple architecture combining linear and sliding window attention. By varying BASED window size and linear attention feature dimension, we can dial the state size and traverse the pareto frontier of the recall-memory tradeoff cu
    
[^4]: 天作之合：大型语言模型与进化算法的结合

    A match made in consistency heaven: when large language models meet evolutionary algorithms. (arXiv:2401.10510v1 [cs.NE])

    [http://arxiv.org/abs/2401.10510](http://arxiv.org/abs/2401.10510)

    大型语言模型和进化算法的结合具有强大的一致性，包括标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化等多个核心特征。本文分析了现有的耦合研究，并为未来的研究提供了基本路线和关键挑战。

    

    预训练的大型语言模型（LLMs）在生成创造性的自然文本方面具有强大的能力。进化算法（EAs）可以发现复杂实际问题的多样解决方案。本文通过比较文本序列生成和进化的共同特点和方向性，阐述了LLMs与EAs之间的强大一致性，包括多个一对一的核心特征：标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化。在这种一致性视角下，分析了现有的耦合研究，包括进化微调和LLM增强型EAs。借助这些洞见，我们概述了未来在LLMs和EAs耦合方面的基本研究路线，并突出了其中的关键挑战。

    Pre-trained large language models (LLMs) have powerful capabilities for generating creative natural text. Evolutionary algorithms (EAs) can discover diverse solutions to complex real-world problems. Motivated by the common collective and directionality of text sequence generation and evolution, this paper illustrates the strong consistency of LLMs and EAs, which includes multiple one-to-one key characteristics: token embedding and genotype-phenotype mapping, position encoding and fitness shaping, position embedding and selection, attention and crossover, feed-forward neural network and mutation, model training and parameter update, and multi-task learning and multi-objective optimization. Based on this consistency perspective, existing coupling studies are analyzed, including evolutionary fine-tuning and LLM-enhanced EAs. Leveraging these insights, we outline a fundamental roadmap for future research in coupling LLMs and EAs, while highlighting key challenges along the way. The consist
    
[^5]: CAMELL：基于置信度的高效自监督主动学习与标签验证获取模型

    CAMELL: Confidence-based Acquisition Model for Efficient Self-supervised Active Learning with Label Validation. (arXiv:2310.08944v1 [cs.CL])

    [http://arxiv.org/abs/2310.08944](http://arxiv.org/abs/2310.08944)

    CAMELL是一个适用于序列多输出问题的主动学习框架，通过仅需专家标注序列的一小部分、自监督和标签验证机制来解决监督神经方法对大规模标注数据集的依赖限制。

    

    在序列任务中，受大规模且精确标注数据集的依赖限制，监督神经方法受到阻碍。标注质量随着从专家标注向众包标注的转变而逐渐恶化。为了解决这些挑战，我们提出了CAMELL（Confidence-based Acquisition Model for Efficient self-supervised active Learning with Label validation），这是一个针对序列多输出问题量身定制的基于池化的主动学习框架。CAMELL具有三个核心特点：(1)仅要求专家标注所选序列的一小部分，(2)为其余序列提供自监督，(3)采用标签验证机制，防止错误标签污染数据集并影响模型性能。我们在序列任务中对CAMELL进行了评估，特别强调对话信念跟踪，这是一个受限制的任务。

    Supervised neural approaches are hindered by their dependence on large, meticulously annotated datasets, a requirement that is particularly cumbersome for sequential tasks. The quality of annotations tends to deteriorate with the transition from expert-based to crowd-sourced labelling. To address these challenges, we present \textbf{CAMELL} (Confidence-based Acquisition Model for Efficient self-supervised active Learning with Label validation), a pool-based active learning framework tailored for sequential multi-output problems. CAMELL possesses three core features: (1) it requires expert annotators to label only a fraction of a chosen sequence, (2) it facilitates self-supervision for the remainder of the sequence, and (3) it employs a label validation mechanism to prevent erroneous labels from contaminating the dataset and harming model performance. We evaluate CAMELL on sequential tasks, with a special emphasis on dialogue belief tracking, a task plagued by the constraints of limited
    
[^6]: 语言模型实现异构数据湖结构化视图生成的简单系统

    Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes. (arXiv:2304.09433v1 [cs.CL])

    [http://arxiv.org/abs/2304.09433](http://arxiv.org/abs/2304.09433)

    本文探讨了使用大型语言模型在不需要特定领域的训练和定制下实现生成可查询表格的简单系统，并给出了两种实现策略，在不同质量和成本中平衡。通过实验证明，该系统对不同类型文档生成的表格均高质量，且无需文档特定的定制。

    

    数据管理界长期以来的目标是开发出通用自动化系统，可以在不需要人力或特定领域的定制情况下摄取半结构化文档并输出可查询的表格。鉴于潜在文档的多样性，现有的最先进系统进行简化的假设并使用特定领域的训练。本文中，我们询问是否可以通过使用大型语言模型（LLMs）来保持广泛性。在广泛数据上预训练的LLMs可仅限于基于自然语言任务描述执行各种下游任务。我们提出并评估了由LLMs驱动的简单原型系统EVAPORATE。我们确定了实现该系统的两种基本不同策略：提示LLM直接从文档中提取值或提示LLM合成执行提取的代码。我们的评估显示，这两种方法之间存在成本-质量权衡。代码合成便宜，但比直接抽取远不准确。我们对四种不同类型文档的实验表明，EVAPORATE可以在不需要任何文档特定的定制情况下为各种文档类型生成高质量的表格。

    A long standing goal of the data management community is to develop general, automated systems that ingest semi-structured documents and output queryable tables without human effort or domain specific customization. Given the sheer variety of potential documents, state-of-the art systems make simplifying assumptions and use domain specific training. In this work, we ask whether we can maintain generality by using large language models (LLMs). LLMs, which are pretrained on broad data, can perform diverse downstream tasks simply conditioned on natural language task descriptions.  We propose and evaluate EVAPORATE, a simple, prototype system powered by LLMs. We identify two fundamentally different strategies for implementing this system: prompt the LLM to directly extract values from documents or prompt the LLM to synthesize code that performs the extraction. Our evaluations show a cost-quality tradeoff between these two approaches. Code synthesis is cheap, but far less accurate than dire
    

