# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Online Unlearning via Hessian-Free Recollection of Individual Data Statistics](https://arxiv.org/abs/2404.01712) | 通过提出的Hessian-free在线遗忘方法，实现了近乎瞬时的在线遗忘，仅需要进行矢量加法操作。 |
| [^2] | [MA4DIV: Multi-Agent Reinforcement Learning for Search Result Diversification](https://arxiv.org/abs/2403.17421) | 引入了基于多智能体强化学习的MA4DIV方法，将搜索结果多样化建模为多个智能体之间的合作任务，直接优化多样性指标，如$\alpha$-NDCG，以实现高训练效率。 |
| [^3] | [Collage Prompting: Budget-Friendly Visual Recognition with GPT-4V](https://arxiv.org/abs/2403.11468) | 通过引入Collage Prompting方法，我们实现了与GPT-4V合作的经济可行的视觉识别方法，通过优化图像排列顺序获得最大的识别准确性。 |
| [^4] | [Partitioned Neural Network Training via Synthetic Intermediate Labels](https://arxiv.org/abs/2403.11204) | 该研究提出了一种通过将模型分区到不同GPU上，并生成合成中间标签来训练各个部分的方法，以缓解大规模神经网络训练中的内存和计算压力。 |
| [^5] | [Robust Decision Aggregation with Adversarial Experts](https://arxiv.org/abs/2403.08222) | 论文考虑了在既有真实专家又有对抗性专家的情况下的二元决策聚合问题，提出了设计鲁棒聚合器以最小化遗憾的方法，并证明了当真实专家是对称的且对抗性专家不太多时，截尾均值是最优的。 |
| [^6] | [EBBS: An Ensemble with Bi-Level Beam Search for Zero-Shot Machine Translation](https://arxiv.org/abs/2403.00144) | 提出了一种集成方法EBBS，配合新颖的双层束搜索算法，能够优于直接和通过第三语言进行的翻译，并实现知识蒸馏来提高推理效率。 |
| [^7] | [A Survey on 3D Skeleton Based Person Re-Identification: Approaches, Designs, Challenges, and Future Directions.](http://arxiv.org/abs/2401.15296) | 本文通过对当前基于3D骨架的人员再识别方法、模型设计、挑战和未来方向的系统调研，填补了相关研究总结的空白。 |
| [^8] | [Expressive Modeling Is Insufficient for Offline RL: A Tractable Inference Perspective.](http://arxiv.org/abs/2311.00094) | 本文指出，在离线强化学习任务中，除了表达性强的序列模型，可处理性也起着重要的作用。由于离线数据收集策略和环境动态的随机性，需要精确且高效地回答各种概率查询，以找到有奖励的动作。基于此，本文提出了Trifle（离线强化学习的可处理推理）方法，利用现代可处理概率模型来解决这个问题。 |
| [^9] | [Layer-wise Feedback Propagation.](http://arxiv.org/abs/2308.12053) | 本文提出了一种名为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，通过利用可解释性细化与层级相关性传播（LRP）相结合，根据每个连接对任务的贡献分配奖励，该方法克服了传统梯度下降方法存在的问题。对于各种模型和数据集，LFP取得了与梯度下降相当的性能。 |
| [^10] | [Preserving Knowledge Invariance: Rethinking Robustness Evaluation of Open Information Extraction.](http://arxiv.org/abs/2305.13981) | 本文提出了第一个模拟评估开放式信息提取模型在真实世界中的基准测试，并通过判断模型在整个团体上的表现是否始终准确来评估模型的鲁棒性。 |

# 详细

[^1]: 通过免Hessian重新整合个体数据统计实现高效在线遗忘

    Efficient Online Unlearning via Hessian-Free Recollection of Individual Data Statistics

    [https://arxiv.org/abs/2404.01712](https://arxiv.org/abs/2404.01712)

    通过提出的Hessian-free在线遗忘方法，实现了近乎瞬时的在线遗忘，仅需要进行矢量加法操作。

    

    机器遗忘旨在通过使模型能够选择性地忘记特定数据来维护数据所有者的被遗忘权利。最近的方法表明，一种数据遗忘的方法是通过预先计算和存储携带二阶信息的统计数据，以改进计算和内存效率。然而，它们依赖于苛刻的假设，而且计算/存储受到模型参数维度的诅咒，这使得难以应用到大多数深度神经网络中。在本工作中，我们提出了一种免Hessian在线遗忘方法。我们建议为每个数据点维护一个统计向量，通过重新训练和学习模型之间的差异的仿射随机递归逼近来计算。我们提出的算法实现了近乎瞬时的在线遗忘，因为它只需要进行矢量加法操作。基于重新收集遗忘数据统计的策略，

    arXiv:2404.01712v1 Announce Type: cross  Abstract: Machine unlearning strives to uphold the data owners' right to be forgotten by enabling models to selectively forget specific data. Recent methods suggest that one approach of data forgetting is by precomputing and storing statistics carrying second-order information to improve computational and memory efficiency. However, they rely on restrictive assumptions and the computation/storage suffer from the curse of model parameter dimensionality, making it challenging to apply to most deep neural networks. In this work, we propose a Hessian-free online unlearning method. We propose to maintain a statistical vector for each data point, computed through affine stochastic recursion approximation of the difference between retrained and learned models. Our proposed algorithm achieves near-instantaneous online unlearning as it only requires a vector addition operation. Based on the strategy that recollecting statistics for forgetting data, the p
    
[^2]: MA4DIV：用于搜索结果多样化的多智能体强化学习

    MA4DIV: Multi-Agent Reinforcement Learning for Search Result Diversification

    [https://arxiv.org/abs/2403.17421](https://arxiv.org/abs/2403.17421)

    引入了基于多智能体强化学习的MA4DIV方法，将搜索结果多样化建模为多个智能体之间的合作任务，直接优化多样性指标，如$\alpha$-NDCG，以实现高训练效率。

    

    搜索结果多样化（SRD）的目标是确保所选文档涵盖尽可能多的不同子主题。现有方法主要利用“贪婪选择”范式，即一次选择一个具有最高多样性分数的文档。这些方法往往效率低下，容易陷入次优状态。此外，一些其他方法旨在近似优化多样性指标，如$\alpha$-NDCG，但结果仍然不尽如人意。为了解决这些挑战，我们引入了用于搜索结果多样性的多智能体强化学习（MARL）方法，称为MA4DIV。在这种方法中，每个文档都是一个智能体，搜索结果多样化被建模为多个智能体之间的合作任务。该方法允许直接优化多样性指标，如$\alpha$-NDCG，同时实现高训练效率。我们进行了初步实验。

    arXiv:2403.17421v1 Announce Type: cross  Abstract: The objective of search result diversification (SRD) is to ensure that selected documents cover as many different subtopics as possible. Existing methods primarily utilize a paradigm of "greedy selection", i.e., selecting one document with the highest diversity score at a time. These approaches tend to be inefficient and are easily trapped in a suboptimal state. In addition, some other methods aim to approximately optimize the diversity metric, such as $\alpha$-NDCG, but the results still remain suboptimal. To address these challenges, we introduce Multi-Agent reinforcement learning (MARL) for search result DIVersity, which called MA4DIV. In this approach, each document is an agent and the search result diversification is modeled as a cooperative task among multiple agents. This approach allows for directly optimizing the diversity metrics, such as $\alpha$-NDCG, while achieving high training efficiency. We conducted preliminary experi
    
[^3]: Collage Prompting: 与GPT-4V合作的经济可行的视觉识别

    Collage Prompting: Budget-Friendly Visual Recognition with GPT-4V

    [https://arxiv.org/abs/2403.11468](https://arxiv.org/abs/2403.11468)

    通过引入Collage Prompting方法，我们实现了与GPT-4V合作的经济可行的视觉识别方法，通过优化图像排列顺序获得最大的识别准确性。

    

    最近生成式人工智能的进展表明，通过采用视觉提示，GPT-4V可以在图像识别任务中展现出显著的熟练度。尽管其令人印象深刻的能力，但与GPT-4V的推断相关的财务成本构成了其广泛应用的重大障碍。为了解决这一挑战，我们的研究引入了Collage Prompting，这是一种经济实惠的提示方法，将多个图像连接成单个视觉输入。借助拼贴提示，GPT-4V可以同时在多幅图像上执行图像识别。基于GPT-4V的图像识别准确性与拼贴提示中图像顺序明显变化的观察，我们的方法进一步学习优化图像安排以获得最大的识别准确性。训练了一个图预测器来指示每个拼贴提示的准确性，然后我们提出了一种优化方法来导航搜索空间。

    arXiv:2403.11468v1 Announce Type: cross  Abstract: Recent advancements in generative AI have suggested that by taking visual prompt, GPT-4V can demonstrate significant proficiency in image recognition task. Despite its impressive capabilities, the financial cost associated with GPT-4V's inference presents a substantial barrier for its wide use. To address this challenge, our work introduces Collage Prompting, a budget-friendly prompting approach that concatenates multiple images into a single visual input. With collage prompt, GPT-4V is able to perform image recognition on several images simultaneously. Based on the observation that the accuracy of GPT-4V's image recognition varies significantly with the order of images within the collage prompt, our method further learns to optimize the arrangement of images for maximum recognition accuracy. A graph predictor is trained to indicate the accuracy of each collage prompt, then we propose an optimization method to navigate the search space
    
[^4]: 通过合成中间标签进行分区神经网络训练

    Partitioned Neural Network Training via Synthetic Intermediate Labels

    [https://arxiv.org/abs/2403.11204](https://arxiv.org/abs/2403.11204)

    该研究提出了一种通过将模型分区到不同GPU上，并生成合成中间标签来训练各个部分的方法，以缓解大规模神经网络训练中的内存和计算压力。

    

    大规模神经网络架构的普及，特别是深度学习模型，对资源密集型训练提出了挑战。 GPU 内存约束已经成为训练这些庞大模型的一个明显瓶颈。现有策略，包括数据并行、模型并行、流水线并行和完全分片数据并行，提供了部分解决方案。 特别是模型并行允许将整个模型分布在多个 GPU 上，但随后的这些分区之间的数据通信减慢了训练速度。此外，为在每个 GPU 上存储辅助参数所需的大量内存开销增加了计算需求。 本研究主张不使用整个模型进行训练，而是将模型分区到 GPU 上，并生成合成中间标签来训练各个部分。 通过随机过程生成的这些标签减缓了训练中的内存和计算压力。

    arXiv:2403.11204v1 Announce Type: cross  Abstract: The proliferation of extensive neural network architectures, particularly deep learning models, presents a challenge in terms of resource-intensive training. GPU memory constraints have become a notable bottleneck in training such sizable models. Existing strategies, including data parallelism, model parallelism, pipeline parallelism, and fully sharded data parallelism, offer partial solutions. Model parallelism, in particular, enables the distribution of the entire model across multiple GPUs, yet the ensuing data communication between these partitions slows down training. Additionally, the substantial memory overhead required to store auxiliary parameters on each GPU compounds computational demands. Instead of using the entire model for training, this study advocates partitioning the model across GPUs and generating synthetic intermediate labels to train individual segments. These labels, produced through a random process, mitigate me
    
[^5]: 具有对抗性专家的鲁棒决策聚合

    Robust Decision Aggregation with Adversarial Experts

    [https://arxiv.org/abs/2403.08222](https://arxiv.org/abs/2403.08222)

    论文考虑了在既有真实专家又有对抗性专家的情况下的二元决策聚合问题，提出了设计鲁棒聚合器以最小化遗憾的方法，并证明了当真实专家是对称的且对抗性专家不太多时，截尾均值是最优的。

    

    我们考虑了在既有真实专家又有对抗性专家的情况下的二元决策聚合问题。真实专家将会如实报告他们的私人信号，并获得适当的激励，而对抗性专家可以任意报告。决策者需要设计一个鲁棒的聚合器，根据专家的报告来预测世界的真实状态。决策者不了解具体的信息结构，即信号、状态以及对抗性专家的策略的联合分布。我们希望找到在最坏信息结构下最小化遗憾的最优聚合器。遗憾被定义为聚合器和一个基准之间的期望损失差，该基准根据联合分布和真实专家的报告做出最优决策。我们证明了当真实专家是对称的且对抗性专家不太多时，截尾均值是最优的。

    arXiv:2403.08222v1 Announce Type: cross  Abstract: We consider a binary decision aggregation problem in the presence of both truthful and adversarial experts. The truthful experts will report their private signals truthfully with proper incentive, while the adversarial experts can report arbitrarily. The decision maker needs to design a robust aggregator to forecast the true state of the world based on the reports of experts. The decision maker does not know the specific information structure, which is a joint distribution of signals, states, and strategies of adversarial experts. We want to find the optimal aggregator minimizing regret under the worst information structure. The regret is defined by the difference in expected loss between the aggregator and a benchmark who makes the optimal decision given the joint distribution and reports of truthful experts.   We prove that when the truthful experts are symmetric and adversarial experts are not too numerous, the truncated mean is opt
    
[^6]: EBBS: 一个具有双层束搜索的集成方法用于零翻译机器翻译

    EBBS: An Ensemble with Bi-Level Beam Search for Zero-Shot Machine Translation

    [https://arxiv.org/abs/2403.00144](https://arxiv.org/abs/2403.00144)

    提出了一种集成方法EBBS，配合新颖的双层束搜索算法，能够优于直接和通过第三语言进行的翻译，并实现知识蒸馏来提高推理效率。

    

    当我们用特定的翻译方向训练多语言模型时，零翻译的能力就会出现；模型可以直接在未见过的方向进行翻译。另外，零翻译也可以通过第三种语言（例如英语）来实现。在我们的工作中，我们发现直接和通过第三种语言进行的翻译都存在噪音，并且表现不尽如人意。我们提出了EBBS，一个具有新颖的双层束搜索算法的集成方法，其中每个集成组件在下层逐步探索自己的预测，但它们通过上层的“软投票”机制进行同步。在两个流行的多语言翻译数据集上的结果表明，EBBS始终优于直接和通过第三种语言进行的翻译，以及现有的集成技术。此外，我们可以将集成的知识传回到多语言模型中，以提高推理效率；值得注意的是，我们的E

    arXiv:2403.00144v1 Announce Type: cross  Abstract: The ability of zero-shot translation emerges when we train a multilingual model with certain translation directions; the model can then directly translate in unseen directions. Alternatively, zero-shot translation can be accomplished by pivoting through a third language (e.g., English). In our work, we observe that both direct and pivot translations are noisy and achieve less satisfactory performance. We propose EBBS, an ensemble method with a novel bi-level beam search algorithm, where each ensemble component explores its own prediction step by step at the lower level but they are synchronized by a "soft voting" mechanism at the upper level. Results on two popular multilingual translation datasets show that EBBS consistently outperforms direct and pivot translations as well as existing ensemble techniques. Further, we can distill the ensemble's knowledge back to the multilingual model to improve inference efficiency; profoundly, our E
    
[^7]: 基于3D骨架的人员再识别：方法、设计、挑战和未来方向的综述

    A Survey on 3D Skeleton Based Person Re-Identification: Approaches, Designs, Challenges, and Future Directions. (arXiv:2401.15296v1 [cs.CV])

    [http://arxiv.org/abs/2401.15296](http://arxiv.org/abs/2401.15296)

    本文通过对当前基于3D骨架的人员再识别方法、模型设计、挑战和未来方向的系统调研，填补了相关研究总结的空白。

    

    通过3D骨架进行人员再识别是一个重要的新兴研究领域，引起了模式识别社区的极大兴趣。近年来，针对骨架建模和特征学习中突出问题，已经提出了许多具有独特优势的基于3D骨架的人员再识别（SRID）方法。尽管近年来取得了一些进展，但据我们所知，目前还没有对这些研究及其挑战进行综合总结。因此，本文通过对当前SRID方法、模型设计、挑战和未来方向的系统调研，试图填补这一空白。具体而言，我们首先定义了SRID问题，并提出了一个SRID研究的分类体系，总结了常用的基准数据集、常用的模型架构，并对不同方法的特点进行了分析评价。然后，我们详细阐述了SRID模型的设计原则。

    Person re-identification via 3D skeletons is an important emerging research area that triggers great interest in the pattern recognition community. With distinctive advantages for many application scenarios, a great diversity of 3D skeleton based person re-identification (SRID) methods have been proposed in recent years, effectively addressing prominent problems in skeleton modeling and feature learning. Despite recent advances, to the best of our knowledge, little effort has been made to comprehensively summarize these studies and their challenges. In this paper, we attempt to fill this gap by providing a systematic survey on current SRID approaches, model designs, challenges, and future directions. Specifically, we first formulate the SRID problem, and propose a taxonomy of SRID research with a summary of benchmark datasets, commonly-used model architectures, and an analytical review of different methods' characteristics. Then, we elaborate on the design principles of SRID models fro
    
[^8]: 表达建模对于离线强化学习不足：可处理的推理角度

    Expressive Modeling Is Insufficient for Offline RL: A Tractable Inference Perspective. (arXiv:2311.00094v1 [cs.LG])

    [http://arxiv.org/abs/2311.00094](http://arxiv.org/abs/2311.00094)

    本文指出，在离线强化学习任务中，除了表达性强的序列模型，可处理性也起着重要的作用。由于离线数据收集策略和环境动态的随机性，需要精确且高效地回答各种概率查询，以找到有奖励的动作。基于此，本文提出了Trifle（离线强化学习的可处理推理）方法，利用现代可处理概率模型来解决这个问题。

    

    离线强化学习任务中，一种流行的范例是先将离线轨迹拟合到一个序列模型中，然后通过该模型提示高期望回报的动作。虽然普遍认为表达性更强的序列模型可以带来更好的性能，但本文强调了可处理性，即精确而高效地回答各种概率查询的能力，同样起着重要的作用。具体而言，由于离线数据收集策略和环境动态带来的基本随机性，需要进行高度非平凡的条件/约束生成，以引出有奖励的动作。虽然仍然可以近似处理这些查询，但我们观察到这种粗糙的估计显著削弱了表达性强的序列模型带来的好处。为了解决这个问题，本文提出了Trifle（离线强化学习的可处理推理），它利用了现代可处理概率模型（TPM）来弥合这个差距。

    A popular paradigm for offline Reinforcement Learning (RL) tasks is to first fit the offline trajectories to a sequence model, and then prompt the model for actions that lead to high expected return. While a common consensus is that more expressive sequence models imply better performance, this paper highlights that tractability, the ability to exactly and efficiently answer various probabilistic queries, plays an equally important role. Specifically, due to the fundamental stochasticity from the offline data-collection policies and the environment dynamics, highly non-trivial conditional/constrained generation is required to elicit rewarding actions. While it is still possible to approximate such queries, we observe that such crude estimates significantly undermine the benefits brought by expressive sequence models. To overcome this problem, this paper proposes Trifle (Tractable Inference for Offline RL), which leverages modern Tractable Probabilistic Models (TPMs) to bridge the gap b
    
[^9]: 层级反馈传播

    Layer-wise Feedback Propagation. (arXiv:2308.12053v1 [cs.LG])

    [http://arxiv.org/abs/2308.12053](http://arxiv.org/abs/2308.12053)

    本文提出了一种名为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，通过利用可解释性细化与层级相关性传播（LRP）相结合，根据每个连接对任务的贡献分配奖励，该方法克服了传统梯度下降方法存在的问题。对于各种模型和数据集，LFP取得了与梯度下降相当的性能。

    

    本文提出了一种称为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，该方法利用可解释性，具体而言是层级相关性传播（LRP），根据每个连接对解决给定任务的贡献独立分配奖励。这与传统的梯度下降方法不同，梯度下降方法是朝向估计的损失最小值更新参数。LFP在模型中传播奖励信号，而无需梯度计算。它增强接收到正反馈的结构，同时降低接收到负反馈的结构的影响。我们从理论和实证的角度证明了LFP的收敛性，并展示了它在各种模型和数据集上实现与梯度下降相当的性能。值得注意的是，LFP克服了梯度方法的某些局限性，例如对有意义的导数的依赖。我们进一步研究了LFP如何解决梯度方法相关问题的限制。

    In this paper, we present Layer-wise Feedback Propagation (LFP), a novel training approach for neural-network-like predictors that utilizes explainability, specifically Layer-wise Relevance Propagation(LRP), to assign rewards to individual connections based on their respective contributions to solving a given task. This differs from traditional gradient descent, which updates parameters towards anestimated loss minimum. LFP distributes a reward signal throughout the model without the need for gradient computations. It then strengthens structures that receive positive feedback while reducingthe influence of structures that receive negative feedback. We establish the convergence of LFP theoretically and empirically, and demonstrate its effectiveness in achieving comparable performance to gradient descent on various models and datasets. Notably, LFP overcomes certain limitations associated with gradient-based methods, such as reliance on meaningful derivatives. We further investigate how 
    
[^10]: 保持知识不变性：重新思考开放信息抽取的鲁棒性评估

    Preserving Knowledge Invariance: Rethinking Robustness Evaluation of Open Information Extraction. (arXiv:2305.13981v1 [cs.CL])

    [http://arxiv.org/abs/2305.13981](http://arxiv.org/abs/2305.13981)

    本文提出了第一个模拟评估开放式信息提取模型在真实世界中的基准测试，并通过判断模型在整个团体上的表现是否始终准确来评估模型的鲁棒性。

    

    鲁棒性是确保自然语言处理模型能够成功应用于现实世界中的关键因素，特别是对于信息抽取任务而言。然而，大多数先前的评估基准都专注于验证配对匹配的正确性，忽略了关键的鲁棒性测量。在本文中，我们提出了第一个基准测试，模拟在真实世界中评估开放式信息提取模型的情况，其中同一知识含义的句法和表达分布会各不相同。我们设计和注释了一个大规模的测试平台，其中每个示例都是一个知识不变的团体，由具有相同含义但结构不同的句子组成。通过进一步阐述鲁棒性指标，当模型在整个团体上的表现始终准确时，被判定为鲁棒性强。我们对过去十年中发表的几种典型模型进行了实验。

    The robustness to distribution changes ensures that NLP models can be successfully applied in the realistic world, especially for information extraction tasks. However, most prior evaluation benchmarks have been devoted to validating pairwise matching correctness, ignoring the crucial measurement of robustness. In this paper, we present the first benchmark that simulates the evaluation of open information extraction models in the real world, where the syntactic and expressive distributions under the same knowledge meaning may drift variously. We design and annotate a large-scale testbed in which each example is a knowledge-invariant clique that consists of sentences with structured knowledge of the same meaning but with different syntactic and expressive forms. By further elaborating the robustness metric, a model is judged to be robust if its performance is consistently accurate on the overall cliques. We perform experiments on typical models published in the last decade as well as a 
    

