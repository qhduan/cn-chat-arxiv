# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [IISAN: Efficiently Adapting Multimodal Representation for Sequential Recommendation with Decoupled PEFT](https://arxiv.org/abs/2404.02059) | IISAN是一种简单的插拔架构，采用解耦PEFT结构，并利用内部和跨模态适应，与全微调和最先进的PEFT性能匹配，显著减少GPU内存使用量，并加速了训练时间。 |
| [^2] | [A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts](https://arxiv.org/abs/2402.09727) | ReadAgent是一个具有长期上下文概要记忆的阅读代理系统，通过实现一个简单的提示系统，它能够处理长输入并提高有效上下文长度。在评估中表现良好。 |
| [^3] | [Multimodal Pre-training Framework for Sequential Recommendation via Contrastive Learning.](http://arxiv.org/abs/2303.11879) | 通过对比学习的多模态预训练框架利用用户的序列行为和物品的多模态内容进行序列推荐，并提出了一种新的骨干网络进行特征融合，实验证明其优于现有最先进方法。 |

# 详细

[^1]: IISAN：使用解耦PEFT有效地调整多模态表示以顺序推荐

    IISAN: Efficiently Adapting Multimodal Representation for Sequential Recommendation with Decoupled PEFT

    [https://arxiv.org/abs/2404.02059](https://arxiv.org/abs/2404.02059)

    IISAN是一种简单的插拔架构，采用解耦PEFT结构，并利用内部和跨模态适应，与全微调和最先进的PEFT性能匹配，显著减少GPU内存使用量，并加速了训练时间。

    

    多模态基础模型在顺序推荐系统中具有转变性，利用强大的表示学习能力。虽然参数高效微调（PEFT）通常用于调整基础模型以进行推荐任务，但大多数研究优先考虑参数效率，通常忽略GPU内存效率和训练速度等关键因素。针对这一差距，本文引入了IISAN（多模态表示的内部和跨模态侧面适应网络），一个使用解耦PEFT结构并利用内部和跨模态适应的简单即插即用架构。IISAN与全微调（FFT）和最先进的PEFT的性能相匹配。更重要的是，它显著减少了GPU内存使用量 - 对于多模态顺序推荐任务，从47GB降低到仅3GB。此外，与FFT相比，它将每个时代的训练时间从443秒加速到22秒。

    arXiv:2404.02059v1 Announce Type: new  Abstract: Multimodal foundation models are transformative in sequential recommender systems, leveraging powerful representation learning capabilities. While Parameter-efficient Fine-tuning (PEFT) is commonly used to adapt foundation models for recommendation tasks, most research prioritizes parameter efficiency, often overlooking critical factors like GPU memory efficiency and training speed. Addressing this gap, our paper introduces IISAN (Intra- and Inter-modal Side Adapted Network for Multimodal Representation), a simple plug-and-play architecture using a Decoupled PEFT structure and exploiting both intra- and inter-modal adaptation.   IISAN matches the performance of full fine-tuning (FFT) and state-of-the-art PEFT. More importantly, it significantly reduces GPU memory usage - from 47GB to just 3GB for multimodal sequential recommendation tasks. Additionally, it accelerates training time per epoch from 443s to 22s compared to FFT. This is also
    
[^2]: 一种具有长期上下文概要记忆的人工智能阅读代理

    A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts

    [https://arxiv.org/abs/2402.09727](https://arxiv.org/abs/2402.09727)

    ReadAgent是一个具有长期上下文概要记忆的阅读代理系统，通过实现一个简单的提示系统，它能够处理长输入并提高有效上下文长度。在评估中表现良好。

    

    当前的大型语言模型不仅限制在某个最大上下文长度内，而且无法稳定地处理长输入。为了解决这些限制，我们提出了ReadAgent，一个增加了有效上下文长度的语言模型代理系统，在我们的实验中可以达到20倍。受到人类交互式阅读长文档的启发，我们将ReadAgent实现为一个简单的提示系统，利用LLM的高级语言能力来：（1）决定将哪些内容存储在一个记忆片段中，（2）将这些记忆片段压缩成为称为概要记忆的短时记忆，（3）在需要时通过原始文本查找段落来提醒自己相关细节以完成任务。我们使用检索方法、使用原始长上下文以及使用概要记忆来评估ReadAgent与基线的性能。这些评估是在三个长文档阅读理解任务上进行的。

    arXiv:2402.09727v1 Announce Type: cross  Abstract: Current Large Language Models (LLMs) are not only limited to some maximum context length, but also are not able to robustly consume long inputs. To address these limitations, we propose ReadAgent, an LLM agent system that increases effective context length up to 20x in our experiments. Inspired by how humans interactively read long documents, we implement ReadAgent as a simple prompting system that uses the advanced language capabilities of LLMs to (1) decide what content to store together in a memory episode, (2) compress those memory episodes into short episodic memories called gist memories, and (3) take actions to look up passages in the original text if ReadAgent needs to remind itself of relevant details to complete a task. We evaluate ReadAgent against baselines using retrieval methods, using the original long contexts, and using the gist memories. These evaluations are performed on three long-document reading comprehension task
    
[^3]: 通过对比学习的多模态预训练框架用于序列推荐

    Multimodal Pre-training Framework for Sequential Recommendation via Contrastive Learning. (arXiv:2303.11879v1 [cs.IR])

    [http://arxiv.org/abs/2303.11879](http://arxiv.org/abs/2303.11879)

    通过对比学习的多模态预训练框架利用用户的序列行为和物品的多模态内容进行序列推荐，并提出了一种新的骨干网络进行特征融合，实验证明其优于现有最先进方法。

    

    序列推荐系统利用用户与物品之间的序列交互作为主要的监督信号来学习用户的喜好。然而，由于用户行为数据的稀疏性，现有方法通常生成不尽如人意的结果。为了解决这个问题，我们提出了一个新颖的预训练框架，名为多模态序列混合（MSM4SR），它利用用户的序列行为和物品的多模态内容（即文本和图像）进行有效推荐。具体来说，MSM4SR将每个物品图像标记成多个文本关键词，并使用预训练的BERT模型获取物品的初始文本和视觉特征，以消除文本和图像模态之间的差异。提出了一种新的骨干网络，即多模态混合序列编码器（M $^2$ SE），它使用互补的序列混合策略来弥合物品多模态内容和用户行为之间的差距。此外，引入对比学习机制来强制学习到的表示变得更有区分度，进一步提高了序列推荐的性能。在两个真实世界数据集上的实验结果验证了我们提出的框架优于现有最先进方法。

    Sequential recommendation systems utilize the sequential interactions of users with items as their main supervision signals in learning users' preferences. However, existing methods usually generate unsatisfactory results due to the sparsity of user behavior data. To address this issue, we propose a novel pre-training framework, named Multimodal Sequence Mixup for Sequential Recommendation (MSM4SR), which leverages both users' sequential behaviors and items' multimodal content (\ie text and images) for effectively recommendation. Specifically, MSM4SR tokenizes each item image into multiple textual keywords and uses the pre-trained BERT model to obtain initial textual and visual features of items, for eliminating the discrepancy between the text and image modalities. A novel backbone network, \ie Multimodal Mixup Sequence Encoder (M$^2$SE), is proposed to bridge the gap between the item multimodal content and the user behavior, using a complementary sequence mixup strategy. In addition,
    

