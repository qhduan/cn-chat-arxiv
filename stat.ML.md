# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Learning with Differential Privacy for End-to-End Speech Recognition.](http://arxiv.org/abs/2310.00098) | 本文提出了一种基于联邦学习和差分隐私的端到端语音识别方法，探索了大型Transformer模型的不同方面，并建立了基线结果。 |

# 详细

[^1]: 使用差分隐私的联邦学习进行端到端语音识别

    Federated Learning with Differential Privacy for End-to-End Speech Recognition. (arXiv:2310.00098v1 [cs.LG])

    [http://arxiv.org/abs/2310.00098](http://arxiv.org/abs/2310.00098)

    本文提出了一种基于联邦学习和差分隐私的端到端语音识别方法，探索了大型Transformer模型的不同方面，并建立了基线结果。

    

    联邦学习是一种有前景的训练机器学习模型的方法，但在自动语音识别领域仅限于初步探索。此外，联邦学习不能本质上保证用户隐私，并需要差分隐私来提供稳健的隐私保证。然而，我们还不清楚在自动语音识别中应用差分隐私的先前工作。本文旨在通过为联邦学习提供差分隐私的自动语音识别基准，并建立第一个基线来填补这一研究空白。我们扩展了现有的联邦学习自动语音识别研究，探索了最新的大型端到端Transformer模型的不同方面：架构设计，种子模型，数据异质性，领域转移，以及cohort大小的影响。通过合理的中央聚合数量，我们能够训练出即使在异构数据、来自另一个领域的种子模型或无预先训练的情况下仍然接近最优的联邦学习模型。

    While federated learning (FL) has recently emerged as a promising approach to train machine learning models, it is limited to only preliminary explorations in the domain of automatic speech recognition (ASR). Moreover, FL does not inherently guarantee user privacy and requires the use of differential privacy (DP) for robust privacy guarantees. However, we are not aware of prior work on applying DP to FL for ASR. In this paper, we aim to bridge this research gap by formulating an ASR benchmark for FL with DP and establishing the first baselines. First, we extend the existing research on FL for ASR by exploring different aspects of recent $\textit{large end-to-end transformer models}$: architecture design, seed models, data heterogeneity, domain shift, and impact of cohort size. With a $\textit{practical}$ number of central aggregations we are able to train $\textbf{FL models}$ that are \textbf{nearly optimal} even with heterogeneous data, a seed model from another domain, or no pre-trai
    

