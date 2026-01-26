# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing](https://arxiv.org/abs/2402.03379) | 全链路上升建模方法ECUP旨在解决链路偏差和处理不适应问题，在线营销中有重要的应用价值。 |
| [^2] | [Vertical Semi-Federated Learning for Efficient Online Advertising.](http://arxiv.org/abs/2209.15635) | 垂直半联合学习为在线广告领域提供了高效的解决方案，通过学习一个联合感知的局部模型以应对传统垂直联合学习的限制。 |

# 详细

[^1]: 全链路上升建模与上下文增强学习用于智能营销

    Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing

    [https://arxiv.org/abs/2402.03379](https://arxiv.org/abs/2402.03379)

    全链路上升建模方法ECUP旨在解决链路偏差和处理不适应问题，在线营销中有重要的应用价值。

    

    上升建模在在线营销中非常重要，它旨在通过预测个体处理效果（ITE）来准确衡量不同策略（如优惠券或折扣）对不同用户的影响。在电子商务环境中，用户行为遵循确定的顺序链路，包括展示、点击和转化。营销策略在这个链路中的每个阶段都会产生不同的上升效应，影响着点击率和转化率等指标。尽管其实用性，现有研究忽视了特定处理中所有阶段的相互影响，并未充分利用处理信息，可能给后续的营销决策引入了重大偏差。本文将这两个问题称为链路偏差问题和处理不适应问题。本文介绍了一种用于解决这些问题的具有上下文增强学习的全链路上升方法（ECUP）。ECUP包括两个主要组成部分：

    Uplift modeling, vital in online marketing, seeks to accurately measure the impact of various strategies, such as coupons or discounts, on different users by predicting the Individual Treatment Effect (ITE). In an e-commerce setting, user behavior follows a defined sequential chain, including impression, click, and conversion. Marketing strategies exert varied uplift effects at each stage within this chain, impacting metrics like click-through and conversion rate. Despite its utility, existing research has neglected to consider the inter-task across all stages impacts within a specific treatment and has insufficiently utilized the treatment information, potentially introducing substantial bias into subsequent marketing decisions. We identify these two issues as the chain-bias problem and the treatment-unadaptive problem. This paper introduces the Entire Chain UPlift method with context-enhanced learning (ECUP), devised to tackle these issues. ECUP consists of two primary components: 1)
    
[^2]: 垂直半联合学习用于高效在线广告

    Vertical Semi-Federated Learning for Efficient Online Advertising. (arXiv:2209.15635v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.15635](http://arxiv.org/abs/2209.15635)

    垂直半联合学习为在线广告领域提供了高效的解决方案，通过学习一个联合感知的局部模型以应对传统垂直联合学习的限制。

    

    传统的垂直联合学习架构存在两个主要问题：1）适用范围受限于重叠样本；2）实时联合服务的系统挑战较高，这限制了其在广告系统中的应用。为解决这些问题，我们提出了一种新的学习设置——半垂直联合学习(Semi-VFL)，以应对这些挑战。半垂直联合学习旨在实现垂直联合学习的实际工业应用方式，通过学习一个联合感知的局部模型，该模型表现优于单方模型，同时保持了局部服务的便利性。为此，我们提出了精心设计的联合特权学习框架(JPL)，来解决被动方特征缺失和适应整个样本空间这两个问题。具体而言，我们构建了一个推理高效的适用于整个样本空间的单方学生模型，同时保持了联合特征扩展的优势。新的表示蒸馏

    The traditional vertical federated learning schema suffers from two main issues: 1) restricted applicable scope to overlapped samples and 2) high system challenge of real-time federated serving, which limits its application to advertising systems. To this end, we advocate a new learning setting Semi-VFL (Vertical Semi-Federated Learning) to tackle these challenge. Semi-VFL is proposed to achieve a practical industry application fashion for VFL, by learning a federation-aware local model which performs better than single-party models and meanwhile maintain the convenience of local-serving. For this purpose, we propose the carefully designed Joint Privileged Learning framework (JPL) to i) alleviate the absence of the passive party's feature and ii) adapt to the whole sample space. Specifically, we build an inference-efficient single-party student model applicable to the whole sample space and meanwhile maintain the advantage of the federated feature extension. New representation distilla
    

