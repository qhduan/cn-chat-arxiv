# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RAR: Retrieving And Ranking Augmented MLLMs for Visual Recognition](https://arxiv.org/abs/2403.13805) | 该论文介绍了RAR方法，通过结合CLIP和MLLMs的优势，增强了少样本/零样本识别能力，特别适用于具有广泛和细粒度词汇特征的数据集。 |
| [^2] | [Approximate and Weighted Data Reconstruction Attack in Federated Learning](https://arxiv.org/abs/2308.06822) | 提出了一种基于插值的近似方法和层次加权损失函数，用于攻击FedAvg场景中的数据重构攻击。 |

# 详细

[^1]: RAR：用于视觉识别的检索和排名增强MLLMs

    RAR: Retrieving And Ranking Augmented MLLMs for Visual Recognition

    [https://arxiv.org/abs/2403.13805](https://arxiv.org/abs/2403.13805)

    该论文介绍了RAR方法，通过结合CLIP和MLLMs的优势，增强了少样本/零样本识别能力，特别适用于具有广泛和细粒度词汇特征的数据集。

    

    arXiv:2403.13805v1 通告类型：交叉摘要：CLIP（对比语言-图像预训练）利用从噪声图像文本对中的对比学习，在识别各种候选项方面表现出色，但其对广泛关联的关注降低了在区分细粒度项目中微妙差异的精度。相反，多模态大型语言模型（MLLMs）在分类细粒度类别方面表现出色，这归功于它们在基于网络的语料库上的预训练所具有的大量知识。然而，由于类别数量的增加，MLLMs的性能下降，主要是由于不断增加的复杂性和有限上下文窗口大小的限制。为了协同两种方法的优势，并增强针对具有广泛和细粒度词汇特征的数据集的少样本/零样本识别能力，本文介绍了RAR，一种用于MLLMs的检索和排名增强方法。我们最初建立了一个基于CLIP的多模态检索器，用于创建和存储

    arXiv:2403.13805v1 Announce Type: cross  Abstract: CLIP (Contrastive Language-Image Pre-training) uses contrastive learning from noise image-text pairs to excel at recognizing a wide array of candidates, yet its focus on broad associations hinders the precision in distinguishing subtle differences among fine-grained items. Conversely, Multimodal Large Language Models (MLLMs) excel at classifying fine-grained categories, thanks to their substantial knowledge from pre-training on web-level corpora. However, the performance of MLLMs declines with an increase in category numbers, primarily due to growing complexity and constraints of limited context window size. To synergize the strengths of both approaches and enhance the few-shot/zero-shot recognition abilities for datasets characterized by extensive and fine-grained vocabularies, this paper introduces RAR, a Retrieving And Ranking augmented method for MLLMs. We initially establish a multi-modal retriever based on CLIP to create and stor
    
[^2]: 联邦学习中的近似和加权数据重构攻击

    Approximate and Weighted Data Reconstruction Attack in Federated Learning

    [https://arxiv.org/abs/2308.06822](https://arxiv.org/abs/2308.06822)

    提出了一种基于插值的近似方法和层次加权损失函数，用于攻击FedAvg场景中的数据重构攻击。

    

    联邦学习（FL）是一种分布式学习范例，使多个客户端能够在不共享私人数据的情况下合作构建机器学习模型。虽然FL被认为是通过设计保护隐私的，但最近的数据重构攻击表明，攻击者可以基于在FL中共享的参数恢复客户端的训练数据。然而，大多数现有方法未能攻击最广泛使用的水平联邦平均（FedAvg）场景，在此场景中，客户端在多个局部训练步骤之后共享模型参数。为了解决这个问题，我们提出了一种基于插值的近似方法，通过生成客户端局部训练过程的中间模型更新，使攻击FedAvg场景变得可行。然后，我们设计了一种层次加权损失函数来改善重构的数据质量。我们为不同层次的模型更新分配不同的权重

    arXiv:2308.06822v2 Announce Type: replace-cross  Abstract: Federated Learning (FL) is a distributed learning paradigm that enables multiple clients to collaborate on building a machine learning model without sharing their private data. Although FL is considered privacy-preserved by design, recent data reconstruction attacks demonstrate that an attacker can recover clients' training data based on the parameters shared in FL. However, most existing methods fail to attack the most widely used horizontal Federated Averaging (FedAvg) scenario, where clients share model parameters after multiple local training steps. To tackle this issue, we propose an interpolation-based approximation method, which makes attacking FedAvg scenarios feasible by generating the intermediate model updates of the clients' local training processes. Then, we design a layer-wise weighted loss function to improve the data quality of reconstruction. We assign different weights to model updates in different layers conc
    

