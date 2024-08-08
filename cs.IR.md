# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LiRank: Industrial Large Scale Ranking Models at LinkedIn](https://arxiv.org/abs/2402.06859) | LiRank是领英的一个大规模排名框架，它应用了最先进的建模架构和优化方法，并提出了新的建模改进和技术，通过A/B测试取得了有效的结果。 |

# 详细

[^1]: LiRank: 领英的工业规模排名模型

    LiRank: Industrial Large Scale Ranking Models at LinkedIn

    [https://arxiv.org/abs/2402.06859](https://arxiv.org/abs/2402.06859)

    LiRank是领英的一个大规模排名框架，它应用了最先进的建模架构和优化方法，并提出了新的建模改进和技术，通过A/B测试取得了有效的结果。

    

    我们介绍了LiRank，这是领英的一个大规模排名框架，它将最先进的建模架构和优化方法应用于生产。我们揭示了几个建模改进，包括Residual DCN，它在著名的DCNv2架构中添加了注意力和残差连接。我们分享了将SOTA架构组合和调优以创建统一模型的见解，包括Dense Gating、Transformers和Residual DCN。我们还提出了用于校准的新技术，并描述了如何将基于深度学习的探索/利用方法应用于生产环境。为了实现大规模排名模型的有效、生产级服务，我们详细介绍了使用量化和词汇压缩训练和压缩模型的方法。我们提供了Feed排名、职位推荐和广告点击率（CTR）预测等大规模使用案例的部署设置细节。通过阐明最有效的技术方法，我们总结了各种A/B测试的经验教训。

    We present LiRank, a large-scale ranking framework at LinkedIn that brings to production state-of-the-art modeling architectures and optimization methods. We unveil several modeling improvements, including Residual DCN, which adds attention and residual connections to the famous DCNv2 architecture. We share insights into combining and tuning SOTA architectures to create a unified model, including Dense Gating, Transformers and Residual DCN. We also propose novel techniques for calibration and describe how we productionalized deep learning based explore/exploit methods. To enable effective, production-grade serving of large ranking models, we detail how to train and compress models using quantization and vocabulary compression. We provide details about the deployment setup for large-scale use cases of Feed ranking, Jobs Recommendations, and Ads click-through rate (CTR) prediction. We summarize our learnings from various A/B tests by elucidating the most effective technical approaches. T
    

