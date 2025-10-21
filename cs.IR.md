# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification.](http://arxiv.org/abs/2310.11675) | 本论文提出了一种基于反馈的证据检索器(FER)，通过整合声明验证者的反馈来优化事实验证中的证据检索过程。实证研究表明FER优于现有的基准方法。 |

# 详细

[^1]: 从相关性到实用性: 基于反馈的证据检索在事实验证中的应用

    From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification. (arXiv:2310.11675v1 [cs.IR])

    [http://arxiv.org/abs/2310.11675](http://arxiv.org/abs/2310.11675)

    本论文提出了一种基于反馈的证据检索器(FER)，通过整合声明验证者的反馈来优化事实验证中的证据检索过程。实证研究表明FER优于现有的基准方法。

    

    在事实验证中，检索增强方法已成为主要的方法之一；它需要对多个检索到的证据进行推理，以验证声明的真实性。为了检索证据，现有的方法通常使用基于概率排序原则设计的现成检索模型。我们认为，在事实验证中，我们需要关注的是声明验证者从检索到的证据中获得的实用性，而不是相关性。我们引入了基于反馈的证据检索器（FER），通过整合声明验证者的反馈来优化证据检索过程。作为反馈信号，我们使用验证者有效利用检索到的证据和基准证据之间实用性的差异来产生最终的声明标签。实证研究证明FER优于现有的基准方法。

    Retrieval-enhanced methods have become a primary approach in fact verification (FV); it requires reasoning over multiple retrieved pieces of evidence to verify the integrity of a claim. To retrieve evidence, existing work often employs off-the-shelf retrieval models whose design is based on the probability ranking principle. We argue that, rather than relevance, for FV we need to focus on the utility that a claim verifier derives from the retrieved evidence. We introduce the feedback-based evidence retriever(FER) that optimizes the evidence retrieval process by incorporating feedback from the claim verifier. As a feedback signal we use the divergence in utility between how effectively the verifier utilizes the retrieved evidence and the ground-truth evidence to produce the final claim label. Empirical studies demonstrate the superiority of FER over prevailing baselines.
    

