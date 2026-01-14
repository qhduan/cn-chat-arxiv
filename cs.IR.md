# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TransFR: Transferable Federated Recommendation with Pre-trained Language Models](https://rss.arxiv.org/abs/2402.01124) | TransFR是一种具备通用文本表示的可迁移联邦推荐模型，它通过结合预训练语言模型和精调本地私有数据的能力，解决了联邦环境下的不可迁移性、冷启动环境下的不可用性和隐私泄露等问题。 |

# 详细

[^1]: TransFR：具备预训练语言模型的可迁移联邦推荐

    TransFR: Transferable Federated Recommendation with Pre-trained Language Models

    [https://rss.arxiv.org/abs/2402.01124](https://rss.arxiv.org/abs/2402.01124)

    TransFR是一种具备通用文本表示的可迁移联邦推荐模型，它通过结合预训练语言模型和精调本地私有数据的能力，解决了联邦环境下的不可迁移性、冷启动环境下的不可用性和隐私泄露等问题。

    

    联邦推荐 (FRs) 是一种促进多个本地客户端在不暴露用户私有数据的情况下共同学习全局模型的隐私保护推荐架构。在传统的FRs中，一种主导范式是利用离散的身份来表示用户/客户端和物品，然后将其映射到领域特定的嵌入中参与模型训练。尽管性能可观，我们揭示了在联邦环境中不能忽视的三个固有限制，即领域间的不可迁移性，在冷启动环境中的不可用性以及在联邦训练过程中潜在的隐私泄露。为此，我们提出了一种具备通用文本表示的可迁移联邦推荐模型TransFR，它巧妙地结合了预训练语言模型赋予的通用能力和通过精调本地私有数据赋予的个性化能力。具体地，它首先学习；...

    Federated recommendations (FRs), facilitating multiple local clients to collectively learn a global model without disclosing user private data, have emerged as a prevalent architecture for privacy-preserving recommendations. In conventional FRs, a dominant paradigm is to utilize discrete identities to represent users/clients and items, which are subsequently mapped to domain-specific embeddings to participate in model training. Despite considerable performance, we reveal three inherent limitations that can not be ignored in federated settings, i.e., non-transferability across domains, unavailability in cold-start settings, and potential privacy violations during federated training. To this end, we propose a transferable federated recommendation model with universal textual representations, TransFR, which delicately incorporates the general capabilities empowered by pre-trained language models and the personalized abilities by fine-tuning local private data. Specifically, it first learn
    

