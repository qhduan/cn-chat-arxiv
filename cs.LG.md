# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inferring Latent Temporal Sparse Coordination Graph for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2403.19253) | 提出了一种用于多智能体强化学习的潜在时间稀疏协调图，能够有效处理智能体之间的协作关系并利用历史观测来进行知识交换 |
| [^2] | [Authorship Verification based on the Likelihood Ratio of Grammar Models](https://arxiv.org/abs/2403.08462) | 提出了一种基于计算作者文件在候选作者语法模型与参考群体语法模型下的可能性比率的方法，用以解决作者身份验证中存在的科学解释不足和难以解释的问题 |

# 详细

[^1]: 推断多智能体强化学习的潜在时间稀疏协调图

    Inferring Latent Temporal Sparse Coordination Graph for Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2403.19253](https://arxiv.org/abs/2403.19253)

    提出了一种用于多智能体强化学习的潜在时间稀疏协调图，能够有效处理智能体之间的协作关系并利用历史观测来进行知识交换

    

    有效的智能体协调对于合作式多智能体强化学习(MARL)至关重要。当前MARL中的图学习方法局限性较大，仅仅依赖一步观察，忽略了重要的历史经验，导致生成的图存在缺陷，促进了冗余或有害信息交换。为了解决这些挑战，我们提出了推断用于MARL的潜在时间稀疏协调图（LTS-CG）。LTS-CG利用智能体的历史观测来计算智能体对概率矩阵，从中抽取稀疏图并用于智能体之间的知识交换，从而同时捕捉智能体的依赖关系和关系不确定性。该过程的计算复杂性仅与智能

    arXiv:2403.19253v1 Announce Type: new  Abstract: Effective agent coordination is crucial in cooperative Multi-Agent Reinforcement Learning (MARL). While agent cooperation can be represented by graph structures, prevailing graph learning methods in MARL are limited. They rely solely on one-step observations, neglecting crucial historical experiences, leading to deficient graphs that foster redundant or detrimental information exchanges. Additionally, high computational demands for action-pair calculations in dense graphs impede scalability. To address these challenges, we propose inferring a Latent Temporal Sparse Coordination Graph (LTS-CG) for MARL. The LTS-CG leverages agents' historical observations to calculate an agent-pair probability matrix, where a sparse graph is sampled from and used for knowledge exchange between agents, thereby simultaneously capturing agent dependencies and relation uncertainty. The computational complexity of this procedure is only related to the number o
    
[^2]: 基于语法模型似然比的作者身份验证

    Authorship Verification based on the Likelihood Ratio of Grammar Models

    [https://arxiv.org/abs/2403.08462](https://arxiv.org/abs/2403.08462)

    提出了一种基于计算作者文件在候选作者语法模型与参考群体语法模型下的可能性比率的方法，用以解决作者身份验证中存在的科学解释不足和难以解释的问题

    

    作者身份验证（AV）是分析一组文件以确定它们是否由特定作者撰写的过程。现有的最先进AV方法使用计算解决方案，对于其功能没有合理的科学解释，并且常常难以解释给分析人员。为解决这个问题，我们提出了一种方法，依赖于计算一个我们称之为 $\lambda_G$（LambdaG）的量：候选作者的上下文语法模型给出的文档的可能性与参考群体的上下文语法模型给出的相同文档的可能性之间的比率。这些语法模型是使用仅针对语法特征进行训练的 $n$-gram语言模型进行估计的。尽管不需要大量数据进行训练，LambdaG...

    arXiv:2403.08462v1 Announce Type: new  Abstract: Authorship Verification (AV) is the process of analyzing a set of documents to determine whether they were written by a specific author. This problem often arises in forensic scenarios, e.g., in cases where the documents in question constitute evidence for a crime. Existing state-of-the-art AV methods use computational solutions that are not supported by a plausible scientific explanation for their functioning and that are often difficult for analysts to interpret. To address this, we propose a method relying on calculating a quantity we call $\lambda_G$ (LambdaG): the ratio between the likelihood of a document given a model of the Grammar for the candidate author and the likelihood of the same document given a model of the Grammar for a reference population. These Grammar Models are estimated using $n$-gram language models that are trained solely on grammatical features. Despite not needing large amounts of data for training, LambdaG st
    

