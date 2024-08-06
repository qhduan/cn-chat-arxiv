# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Aligning Large Language Models for Controllable Recommendations](https://arxiv.org/abs/2403.05063) | 通过引入监督学习任务和强化学习对齐程序，研究人员提出了一种方法来改善大型语言模型适应推荐指令和减少格式错误的能力。 |
| [^2] | [Enhancing Conceptual Understanding in Multimodal Contrastive Learning through Hard Negative Samples](https://arxiv.org/abs/2403.02875) | 提出了一种通过硬负样本改进多模态对比学习中概念理解的方法，并引入了一个评估视觉-语言模型中颜色、对象和大小细粒度对齐的新数据集。 |
| [^3] | [eCeLLM: Generalizing Large Language Models for E-commerce from Large-scale, High-quality Instruction Data](https://arxiv.org/abs/2402.08831) | 本文利用开源的大规模高质量指导数据集ECInstruct，通过指导调优通用语言模型，开发了一系列电子商务LLMs（eCeLLM），在电子商务中表现出了显著的优势。 |
| [^4] | [Deep Evolutional Instant Interest Network for CTR Prediction in Trigger-Induced Recommendation.](http://arxiv.org/abs/2401.07769) | 这篇论文提出了一种基于深度进化的即时兴趣网络（DEI2N）来解决触发引导推荐（TIR）中的点击率预测问题。该方法考虑了用户行为的时间信息、即时兴趣的动态变化以及触发项和目标项之间的交互。 |
| [^5] | [From Retrieval to Generation: Efficient and Effective Entity Set Expansion.](http://arxiv.org/abs/2304.03531) | 本文提出了GenExpan，一种基于生成式预训练语言模型的实体集扩展框架，利用前缀树保证实体生成的有效性，采用自动生成的类名来引导模型生成同一类实体，从而提高了效率和可扩展性。 |

# 详细

[^1]: 调整大型语言模型以实现可控的推荐

    Aligning Large Language Models for Controllable Recommendations

    [https://arxiv.org/abs/2403.05063](https://arxiv.org/abs/2403.05063)

    通过引入监督学习任务和强化学习对齐程序，研究人员提出了一种方法来改善大型语言模型适应推荐指令和减少格式错误的能力。

    

    受到大型语言模型（LLMs）异常的智能启发，研究人员已开始探索将它们应用于开创下一代推荐系统 - 这些系统具有对话、可解释和可控的特性。然而，现有文献主要集中在将领域特定知识整合到LLMs中以提高准确性，通常忽略了遵循指令的能力。为填补这一空白，我们首先引入一组监督学习任务，标记来源于传统推荐模型的标签，旨在明确改善LLMs遵循特定推荐指令的熟练程度。随后，我们开发了一种基于强化学习的对齐程序，进一步加强了LLMs在响应用户意图和减少格式错误方面的能力。通过在两个真实世界数据集上进行广泛实验，我们的方法标记着

    arXiv:2403.05063v1 Announce Type: cross  Abstract: Inspired by the exceptional general intelligence of Large Language Models (LLMs), researchers have begun to explore their application in pioneering the next generation of recommender systems - systems that are conversational, explainable, and controllable. However, existing literature primarily concentrates on integrating domain-specific knowledge into LLMs to enhance accuracy, often neglecting the ability to follow instructions. To address this gap, we initially introduce a collection of supervised learning tasks, augmented with labels derived from a conventional recommender model, aimed at explicitly improving LLMs' proficiency in adhering to recommendation-specific instructions. Subsequently, we develop a reinforcement learning-based alignment procedure to further strengthen LLMs' aptitude in responding to users' intentions and mitigating formatting errors. Through extensive experiments on two real-world datasets, our method markedl
    
[^2]: 通过硬负样本增强多模态对比学习中的概念理解

    Enhancing Conceptual Understanding in Multimodal Contrastive Learning through Hard Negative Samples

    [https://arxiv.org/abs/2403.02875](https://arxiv.org/abs/2403.02875)

    提出了一种通过硬负样本改进多模态对比学习中概念理解的方法，并引入了一个评估视觉-语言模型中颜色、对象和大小细粒度对齐的新数据集。

    

    当前利用对比学习的多模态模型在发展精细的概念理解方面通常存在一些限制。在预训练过程中，由于随机负样本，导致几乎只有非常不同的概念进行损失函数比较。因此，模型在处理细粒度语义差异时遇到困难。为了解决这个问题，我们引入了一种新颖的预训练方法，结合了合成的硬负文本示例。这些硬负样本对应于视觉概念的排列，导致更精细的视觉和文本概念对齐。此外，我们引入了InpaintCOCO，一个用于评估视觉-语言模型中颜色、对象和大小细粒度对齐的新挑战性数据集。我们使用从COCO图像生成的信息填充来创建数据集，通过改变视觉概念，使图像不再与其原始标题匹配。我们的结果显示...

    arXiv:2403.02875v1 Announce Type: cross  Abstract: Current multimodal models leveraging contrastive learning often face limitations in developing fine-grained conceptual understanding. This is due to random negative samples during pretraining, causing almost exclusively very dissimilar concepts to be compared in the loss function. Consequently, the models struggle with fine-grained semantic differences. To address this problem, we introduce a novel pretraining method incorporating synthetic hard negative text examples. The hard negatives permute terms corresponding to visual concepts, leading to a more fine-grained visual and textual concept alignment. Further, we introduce InpaintCOCO, a new challenging dataset for assessing the fine-grained alignment of colors, objects, and sizes in vision-language models. We created the dataset using generative inpainting from COCO images by changing the visual concepts so that the images no longer match their original captions. Our results show sig
    
[^3]: eCeLLM：从大规模高质量指导数据中将大型语言模型推广到电子商务中

    eCeLLM: Generalizing Large Language Models for E-commerce from Large-scale, High-quality Instruction Data

    [https://arxiv.org/abs/2402.08831](https://arxiv.org/abs/2402.08831)

    本文利用开源的大规模高质量指导数据集ECInstruct，通过指导调优通用语言模型，开发了一系列电子商务LLMs（eCeLLM），在电子商务中表现出了显著的优势。

    

    通过在开发有效的电子商务模型方面做出巨大努力，传统的电子商务模型在通用电子商务建模上取得了有限的成功，并且在新用户和新产品上的表现不佳——这是一个典型的领域外泛化挑战。与此同时，大型语言模型(LLMs)在许多领域展示出了出色的通用建模和领域外泛化能力。为了充分发挥它们在电子商务中的作用，本文构建了ECInstruct，这是第一个面向电子商务的开源、大规模和高质量的指导数据集。利用ECInstruct，我们通过指导调优通用语言模型开发了一系列电子商务LLMs，称为eCeLLM。我们的综合实验和评估表明，eCeLLM模型在内部环境中明显优于基准模型，包括最先进的GPT-4和最先进的特定任务模型。

    arXiv:2402.08831v1 Announce Type: cross Abstract: With tremendous efforts on developing effective e-commerce models, conventional e-commerce models show limited success in generalist e-commerce modeling, and suffer from unsatisfactory performance on new users and new products - a typical out-of-domain generalization challenge. Meanwhile, large language models (LLMs) demonstrate outstanding performance in generalist modeling and out-of-domain generalizability in many fields. Toward fully unleashing their power for e-commerce, in this paper, we construct ECInstruct, the first open-sourced, large-scale, and high-quality benchmark instruction dataset for e-commerce. Leveraging ECInstruct, we develop eCeLLM, a series of e-commerce LLMs, by instruction-tuning general-purpose LLMs. Our comprehensive experiments and evaluation demonstrate that eCeLLM models substantially outperform baseline models, including the most advanced GPT-4, and the state-of-the-art task-specific models in in-domain ev
    
[^4]: 基于深度进化的即时兴趣网络用于触发引导推荐中的CTR预测

    Deep Evolutional Instant Interest Network for CTR Prediction in Trigger-Induced Recommendation. (arXiv:2401.07769v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2401.07769](http://arxiv.org/abs/2401.07769)

    这篇论文提出了一种基于深度进化的即时兴趣网络（DEI2N）来解决触发引导推荐（TIR）中的点击率预测问题。该方法考虑了用户行为的时间信息、即时兴趣的动态变化以及触发项和目标项之间的交互。

    

    推荐在许多行业中发挥着关键作用，例如电子商务、流媒体、社交媒体等。最近，出现了一种新的推荐场景，称为触发引导推荐（TIR），用户可以通过触发项明确表达他们的即时兴趣，在许多电子商务平台（如阿里巴巴和亚马逊）中起着重要作用。传统的推荐方法通常无法明确建模用户的即时兴趣，因此在TIR中获得次优结果。尽管有一些同时考虑触发项和目标项的方法来解决这个问题，但它们仍未考虑用户行为的时间信息、用户向下滚动时即时兴趣的动态变化以及触发项和目标项之间的交互。为了解决这些问题，我们提出了一种新的方法--深度进化的即时兴趣网络（DEI2N），用于TIR场景中的点击率预测。

    The recommendation has been playing a key role in many industries, e.g., e-commerce, streaming media, social media, etc. Recently, a new recommendation scenario, called Trigger-Induced Recommendation (TIR), where users are able to explicitly express their instant interests via trigger items, is emerging as an essential role in many e-commerce platforms, e.g., Alibaba.com and Amazon. Without explicitly modeling the user's instant interest, traditional recommendation methods usually obtain sub-optimal results in TIR. Even though there are a few methods considering the trigger and target items simultaneously to solve this problem, they still haven't taken into account temporal information of user behaviors, the dynamic change of user instant interest when the user scrolls down and the interactions between the trigger and target items. To tackle these problems, we propose a novel method -- Deep Evolutional Instant Interest Network (DEI2N), for click-through rate prediction in TIR scenarios
    
[^5]: 从检索到生成：高效且有效的实体集扩展方法

    From Retrieval to Generation: Efficient and Effective Entity Set Expansion. (arXiv:2304.03531v1 [cs.CL])

    [http://arxiv.org/abs/2304.03531](http://arxiv.org/abs/2304.03531)

    本文提出了GenExpan，一种基于生成式预训练语言模型的实体集扩展框架，利用前缀树保证实体生成的有效性，采用自动生成的类名来引导模型生成同一类实体，从而提高了效率和可扩展性。

    

    实体集扩展（ESE）是一项至关重要的任务，旨在扩展由小的种子实体集描述的目标语义类的实体。大多数现有的ESE方法是基于检索的框架，需要提取实体的上下文特征，并计算种子实体和候选实体之间的相似性。为了实现这两个目的，它们必须迭代地遍历语料库和数据集中提供的实体词汇，导致效率和可扩展性较差。实验结果表明，基于检索的ESE方法消耗的时间与实体词汇和语料库的大小成线性增长。本文首先提出了一种生成式ESE框架，Generative Entity Set Expansion (GenExpan)，它利用生成式预训练语言模型来完成ESE任务。具体而言，采用前缀树来保证实体生成的有效性，并采用自动生成的类名来引导模型生成同一类实体。

    Entity Set Expansion (ESE) is a critical task aiming to expand entities of the target semantic class described by a small seed entity set. Most existing ESE methods are retrieval-based frameworks that need to extract the contextual features of entities and calculate the similarity between seed entities and candidate entities. To achieve the two purposes, they should iteratively traverse the corpus and the entity vocabulary provided in the datasets, resulting in poor efficiency and scalability. The experimental results indicate that the time consumed by the retrieval-based ESE methods increases linearly with entity vocabulary and corpus size. In this paper, we firstly propose a generative ESE framework, Generative Entity Set Expansion (GenExpan), which utilizes a generative pre-trained language model to accomplish ESE task. Specifically, a prefix tree is employed to guarantee the validity of entity generation, and automatically generated class names are adopted to guide the model to gen
    

