# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedLP: Layer-wise Pruning Mechanism for Communication-Computation Efficient Federated Learning.](http://arxiv.org/abs/2303.06360) | 本文提出了一种显式的FL剪枝框架FedLP，采用局部训练和联邦更新中的层次剪枝，对不同类型的深度学习模型具有普适性，可以缓解通信和计算的系统瓶颈，并且性能下降较小。 |
| [^2] | [A Novel Tensor-Expert Hybrid Parallelism Approach to Scale Mixture-of-Experts Training.](http://arxiv.org/abs/2303.06318) | 本文提出了一种新的混合并行算法，结合了张量、专家和数据并行，以实现MoE模型的训练，其基本模型比当前最先进的DeepSpeed-MoE大4-8倍。 |
| [^3] | [Stabilizing and Improving Federated Learning with Non-IID Data and Client Dropout in IoT Systems.](http://arxiv.org/abs/2303.06314) | 本文提出了一个简单而有效的框架，通过引入一个先验校准的softmax函数来计算交叉熵损失和基于原型的特征提取来维护一个平衡的分类器头，以稳定和改进联邦学习。 |
| [^4] | [Papaya: Federated Learning, but Fully Decentralized.](http://arxiv.org/abs/2303.06189) | Papaya是一种点对点学习系统，节点在自己的数据上进行训练，并定期根据学习的信任矩阵将其参数与同伴的参数进行加权平均，从而实现联邦学习的去中心化，避免了集中式服务器的带宽和资源密集型限制和隐私问题。 |
| [^5] | [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference.](http://arxiv.org/abs/2303.06182) | 本文提出了三种优化技术来缓解混合专家（MoE）模型在推理时的低效率，包括动态门控、专家缓冲和专家负载平衡。这些技术可以显著提高执行时间和减少内存使用。 |
| [^6] | [Digital Twin-Assisted Knowledge Distillation Framework for Heterogeneous Federated Learning.](http://arxiv.org/abs/2303.06155) | 本文提出了一种数字孪生辅助的知识蒸馏框架，用于解决联邦学习系统中的异构性问题，用户可以选择自己的神经网络模型并从大型教师模型中蒸馏知识，同时利用数字孪生在服务器上训练大型教师模型，最终通过混合整数规划和Q-learning算法实现模型选择和资源分配。 |
| [^7] | [Exploiting Sparsity in Pruned Neural Networks to Optimize Large Model Training.](http://arxiv.org/abs/2302.05045) | 本文提出了一种新的方法，利用稀疏子网络来优化两种流行的深度学习并行算法 - 数据并行和层间并行的内存利用和通信。在512个NVIDIA V100 GPU上，我们的优化将27亿参数模型的内存消耗减少了74％，总通信时间减少了40％。 |
| [^8] | [Egeria: Efficient DNN Training with Knowledge-Guided Layer Freezing.](http://arxiv.org/abs/2201.06227) | Egeria是一种基于知识引导的DNN训练系统，通过跳过DNN层的计算和通信来实现高效训练，利用参考模型中的语义知识准确评估单个层的训练可塑性，并安全地冻结已收敛的层，从而节省相应的反向计算和通信。 |

# 详细

[^1]: FedLP: 一种用于通信计算高效的联邦学习的层次剪枝机制

    FedLP: Layer-wise Pruning Mechanism for Communication-Computation Efficient Federated Learning. (arXiv:2303.06360v1 [cs.LG])

    [http://arxiv.org/abs/2303.06360](http://arxiv.org/abs/2303.06360)

    本文提出了一种显式的FL剪枝框架FedLP，采用局部训练和联邦更新中的层次剪枝，对不同类型的深度学习模型具有普适性，可以缓解通信和计算的系统瓶颈，并且性能下降较小。

    This paper proposes an explicit FL pruning framework, FedLP, which adopts layer-wise pruning in local training and federated updating, and is model-agnostic and universal for different types of deep learning models. FedLP can relieve the system bottlenecks of communication and computation with marginal performance decay.

    联邦学习（FL）已经成为一种高效且隐私保护的分布式学习方案。本文主要关注FL中计算和通信的优化，采用局部训练和联邦更新中的层次剪枝，提出了一个显式的FL剪枝框架FedLP（Federated Layer-wise Pruning），该框架对不同类型的深度学习模型具有普适性。为具有同质本地模型和异质本地模型的场景设计了两种特定的FedLP方案。通过理论和实验评估，证明了FedLP可以缓解通信和计算的系统瓶颈，并且性能下降较小。据我们所知，FedLP是第一个正式将层次剪枝引入FL的框架。在联邦学习范围内，可以基于FedLP进一步设计更多的变体和组合。

    Federated learning (FL) has prevailed as an efficient and privacy-preserved scheme for distributed learning. In this work, we mainly focus on the optimization of computation and communication in FL from a view of pruning. By adopting layer-wise pruning in local training and federated updating, we formulate an explicit FL pruning framework, FedLP (Federated Layer-wise Pruning), which is model-agnostic and universal for different types of deep learning models. Two specific schemes of FedLP are designed for scenarios with homogeneous local models and heterogeneous ones. Both theoretical and experimental evaluations are developed to verify that FedLP relieves the system bottlenecks of communication and computation with marginal performance decay. To the best of our knowledge, FedLP is the first framework that formally introduces the layer-wise pruning into FL. Within the scope of federated learning, more variants and combinations can be further designed based on FedLP.
    
[^2]: 一种新的张量专家混合并行方法来扩展混合专家训练

    A Novel Tensor-Expert Hybrid Parallelism Approach to Scale Mixture-of-Experts Training. (arXiv:2303.06318v1 [cs.LG])

    [http://arxiv.org/abs/2303.06318](http://arxiv.org/abs/2303.06318)

    本文提出了一种新的混合并行算法，结合了张量、专家和数据并行，以实现MoE模型的训练，其基本模型比当前最先进的DeepSpeed-MoE大4-8倍。

    This paper proposes a novel hybrid parallel algorithm that combines tensor, expert, and data parallelism to enable the training of MoE models with 4-8x larger base models than the current state-of-the-art -- DeepSpeed-MoE.

    最近提出了一种名为Mixture-of-Experts（MoE）的新型神经网络架构，通过添加稀疏激活的专家块来增加神经网络（基本模型）的参数，而不改变训练或推理的总浮点操作数。理论上，这种架构允许我们训练任意大的模型，同时保持计算成本与基本模型相同。然而，在64到128个专家块之外，先前的工作观察到这些MoE模型的测试准确性递减。因此，训练高质量的MoE模型需要我们扩展基本模型的大小以及专家块的数量。在这项工作中，我们提出了一种新颖的三维混合并行算法，结合了张量、专家和数据并行，以实现MoE模型的训练，其基本模型比当前最先进的DeepSpeed-MoE大4-8倍。我们在优化器步骤中提出了内存优化。

    A new neural network architecture called Mixture-of-Experts (MoE) has been proposed recently that increases the parameters of a neural network (the base model) by adding sparsely activated expert blocks, without changing the total number of floating point operations for training or inference. In theory, this architecture allows us to train arbitrarily large models while keeping the computational costs same as that of the base model. However, beyond 64 to 128 experts blocks, prior work has observed diminishing returns in the test accuracies of these MoE models. Thus, training high quality MoE models requires us to scale the size of the base models, along with the number of expert blocks. In this work, we propose a novel, three-dimensional, hybrid parallel algorithm that combines tensor, expert, and data parallelism to enable the training of MoE models with 4-8x larger base models than the current state-of-the-art -- DeepSpeed-MoE. We propose memory optimizations in the optimizer step, a
    
[^3]: 在物联网系统中通过非独立同分布数据和客户端dropout来稳定和改进联邦学习

    Stabilizing and Improving Federated Learning with Non-IID Data and Client Dropout in IoT Systems. (arXiv:2303.06314v1 [cs.LG])

    [http://arxiv.org/abs/2303.06314](http://arxiv.org/abs/2303.06314)

    本文提出了一个简单而有效的框架，通过引入一个先验校准的softmax函数来计算交叉熵损失和基于原型的特征提取来维护一个平衡的分类器头，以稳定和改进联邦学习。

    This paper proposes a simple yet effective framework to stabilize and improve federated learning by introducing a prior-calibrated softmax function for computing the cross-entropy loss and a prototype-based feature extraction to maintain a balanced classifier head.

    联邦学习是一种新兴的技术，用于在不暴露私有数据的情况下在分散的客户端上训练深度模型，然而它受到标签分布偏斜的影响，通常导致收敛缓慢和模型性能下降。当参与的客户端处于不稳定的环境并经常掉线时，这个挑战可能更加严重。为了解决这个问题，我们提出了一个简单而有效的框架，通过引入一个先验校准的softmax函数来计算交叉熵损失和基于原型的特征提取来维护一个平衡的分类器头。

    Federated learning is an emerging technique for training deep models over decentralized clients without exposing private data, which however suffers from label distribution skew and usually results in slow convergence and degraded model performance. This challenge could be more serious when the participating clients are in unstable circumstances and dropout frequently. Previous work and our empirical observations demonstrate that the classifier head for classification task is more sensitive to label skew and the unstable performance of FedAvg mainly lies in the imbalanced training samples across different classes. The biased classifier head will also impact the learning of feature representations. Therefore, maintaining a balanced classifier head is of significant importance for building a better global model. To tackle this issue, we propose a simple yet effective framework by introducing a prior-calibrated softmax function for computing the cross-entropy loss and a prototype-based fe
    
[^4]: Papaya：联邦学习，但完全去中心化

    Papaya: Federated Learning, but Fully Decentralized. (arXiv:2303.06189v1 [cs.LG])

    [http://arxiv.org/abs/2303.06189](http://arxiv.org/abs/2303.06189)

    Papaya是一种点对点学习系统，节点在自己的数据上进行训练，并定期根据学习的信任矩阵将其参数与同伴的参数进行加权平均，从而实现联邦学习的去中心化，避免了集中式服务器的带宽和资源密集型限制和隐私问题。

    Papaya is a peer-to-peer learning system that allows nodes to train on their own data and periodically perform a weighted average of their parameters with that of their peers according to a learned trust matrix, achieving decentralized federated learning and avoiding the bandwidth and resource-heavy constraint and privacy concerns of a centralized server.

    联邦学习系统使用集中式服务器来聚合模型更新。这是一种带宽和资源密集型的限制，并暴露系统的隐私问题。相反，我们实现了一种点对点学习系统，其中节点在自己的数据上进行训练，并定期根据学习的信任矩阵将其参数与同伴的参数进行加权平均。到目前为止，我们已经创建了一个模型客户端框架，并使用多个虚拟节点在同一台计算机上运行实验来验证所提出的系统。我们使用了我们提案的第一轮中所述的策略来证明共享参数的点对点学习概念。我们现在希望运行更多实验，并构建一个更可部署的真实世界系统。

    Federated Learning systems use a centralized server to aggregate model updates. This is a bandwidth and resource-heavy constraint and exposes the system to privacy concerns. We instead implement a peer to peer learning system in which nodes train on their own data and periodically perform a weighted average of their parameters with that of their peers according to a learned trust matrix. So far, we have created a model client framework and have been using this to run experiments on the proposed system using multiple virtual nodes which in reality exist on the same computer. We used this strategy as stated in Iteration 1 of our proposal to prove the concept of peer to peer learning with shared parameters. We now hope to run more experiments and build a more deployable real world system for the same.
    
[^5]: 迈向MoE部署：缓解混合专家（MoE）推理中的低效率

    Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference. (arXiv:2303.06182v1 [cs.DC])

    [http://arxiv.org/abs/2303.06182](http://arxiv.org/abs/2303.06182)

    本文提出了三种优化技术来缓解混合专家（MoE）模型在推理时的低效率，包括动态门控、专家缓冲和专家负载平衡。这些技术可以显著提高执行时间和减少内存使用。

    This paper proposes three optimization techniques to mitigate inefficiencies in Mixture-of-Experts (MoE) models during inference, including dynamic gating, expert buffering, and expert load balancing. These techniques can significantly improve execution time and reduce memory usage.

    混合专家（MoE）模型最近在计算机视觉和自然语言处理的广泛任务中取得了最先进的性能。它们在训练期间有效地扩展了模型容量，同时增加的计算成本很小。然而，由于其庞大的模型大小和复杂的通信模式，部署这样的模型进行推理是困难的。在这项工作中，我们提供了两个MoE工作负载的特征化，即语言建模（LM）和机器翻译（MT），并确定了它们在部署时的低效率来源。我们提出了三种优化技术来缓解低效率的来源，即（1）动态门控，（2）专家缓冲和（3）专家负载平衡。我们展示了动态门控可以使LM的执行时间提高1.25-4倍，MT编码器提高2-5倍，MT解码器提高1.09-1.5倍。它还可以将LM的内存使用减少高达1.36倍，MT的内存使用减少高达1.1倍。

    Mixture-of-Experts (MoE) models have recently gained steam in achieving the state-of-the-art performance in a wide range of tasks in computer vision and natural language processing. They effectively expand the model capacity while incurring a minimal increase in computation cost during training. However, deploying such models for inference is difficult due to their large model size and complex communication pattern. In this work, we provide a characterization of two MoE workloads, namely Language Modeling (LM) and Machine Translation (MT) and identify their sources of inefficiencies at deployment.  We propose three optimization techniques to mitigate sources of inefficiencies, namely (1) Dynamic gating, (2) Expert Buffering, and (3) Expert load balancing. We show that dynamic gating improves execution time by 1.25-4$\times$ for LM, 2-5$\times$ for MT Encoder and 1.09-1.5$\times$ for MT Decoder. It also reduces memory usage by up to 1.36$\times$ for LM and up to 1.1$\times$ for MT. We f
    
[^6]: 数字孪生辅助异构联邦学习的知识蒸馏框架

    Digital Twin-Assisted Knowledge Distillation Framework for Heterogeneous Federated Learning. (arXiv:2303.06155v1 [cs.LG])

    [http://arxiv.org/abs/2303.06155](http://arxiv.org/abs/2303.06155)

    本文提出了一种数字孪生辅助的知识蒸馏框架，用于解决联邦学习系统中的异构性问题，用户可以选择自己的神经网络模型并从大型教师模型中蒸馏知识，同时利用数字孪生在服务器上训练大型教师模型，最终通过混合整数规划和Q-learning算法实现模型选择和资源分配。

    This paper proposes a digital twin-assisted knowledge distillation framework for heterogeneous federated learning, where users can select their own neural network models and distill knowledge from a big teacher model, and the teacher model can be trained on a digital twin located in the server. The joint problem of model selection and training offloading and resource allocation for users is formulated as a mixed integer programming problem and solved using Q-learning and optimization algorithms.

    本文提出了一种知识蒸馏驱动的联邦学习框架，以应对联邦学习系统中的异构性，其中每个用户可以根据需要选择其神经网络模型，并使用自己的私有数据集从大型教师模型中蒸馏知识。为了克服在资源有限的用户设备上训练大型教师模型的挑战，利用数字孪生的方式，教师模型可以在具有足够计算资源的服务器上的数字孪生中进行训练。然后，在模型蒸馏期间，每个用户可以在物理实体或数字代理处更新其模型的参数。为用户选择模型和训练卸载和资源分配制定了混合整数规划（MIP）问题。为了解决这个问题，联合使用Q-learning和优化，其中Q-learning为用户选择模型并确定是在本地还是在服务器上进行训练，而优化则用于资源分配。

    In this paper, to deal with the heterogeneity in federated learning (FL) systems, a knowledge distillation (KD) driven training framework for FL is proposed, where each user can select its neural network model on demand and distill knowledge from a big teacher model using its own private dataset. To overcome the challenge of train the big teacher model in resource limited user devices, the digital twin (DT) is exploit in the way that the teacher model can be trained at DT located in the server with enough computing resources. Then, during model distillation, each user can update the parameters of its model at either the physical entity or the digital agent. The joint problem of model selection and training offloading and resource allocation for users is formulated as a mixed integer programming (MIP) problem. To solve the problem, Q-learning and optimization are jointly used, where Q-learning selects models for users and determines whether to train locally or on the server, and optimiz
    
[^7]: 利用剪枝神经网络中的稀疏性来优化大型模型训练

    Exploiting Sparsity in Pruned Neural Networks to Optimize Large Model Training. (arXiv:2302.05045v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.05045](http://arxiv.org/abs/2302.05045)

    本文提出了一种新的方法，利用稀疏子网络来优化两种流行的深度学习并行算法 - 数据并行和层间并行的内存利用和通信。在512个NVIDIA V100 GPU上，我们的优化将27亿参数模型的内存消耗减少了74％，总通信时间减少了40％。

    This paper proposes a novel approach that exploits sparse subnetworks to optimize memory utilization and communication in two popular algorithms for parallel deep learning, and demonstrates significant reductions in memory consumption and communication time on a 2.7 billion parameter model using 512 NVIDIA V100 GPUs.

    由于通信开销的显著增加，规模化神经网络的并行训练具有挑战性。最近，深度学习研究人员开发了各种剪枝算法，能够剪枝（即将神经网络中的参数设置为零）80-90％的参数，以产生与未剪枝父网络相等的稀疏子网络。在本文中，我们提出了一种新的方法，利用这些稀疏子网络来优化两种流行的深度学习并行算法 - 数据并行和层间并行的内存利用和通信。我们将我们的方法集成到AxoNN中，这是一个高度可扩展的并行深度学习框架，依赖于数据和层间并行，并展示了通信时间和内存利用的减少。在512个NVIDIA V100 GPU上，我们的优化将27亿参数模型的内存消耗减少了74％，总通信时间减少了40％，从而提供了

    Parallel training of neural networks at scale is challenging due to significant overheads arising from communication. Recently, deep learning researchers have developed a variety of pruning algorithms that are capable of pruning (i.e. setting to zero) 80-90% of the parameters in a neural network to yield sparse subnetworks that equal the accuracy of the unpruned parent network. In this work, we propose a novel approach that exploits these sparse subnetworks to optimize the memory utilization and communication in two popular algorithms for parallel deep learning namely -- data and inter-layer parallelism. We integrate our approach into AxoNN, a highly scalable framework for parallel deep learning that relies on data and inter-layer parallelism, and demonstrate the reduction in communication time and memory utilization. On 512 NVIDIA V100 GPUs, our optimizations reduce the memory consumption of a 2.7 billion parameter model by 74%, and the total communication time by 40%, thus providing 
    
[^8]: Egeria: 基于知识引导的层冻结技术实现高效DNN训练

    Egeria: Efficient DNN Training with Knowledge-Guided Layer Freezing. (arXiv:2201.06227v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.06227](http://arxiv.org/abs/2201.06227)

    Egeria是一种基于知识引导的DNN训练系统，通过跳过DNN层的计算和通信来实现高效训练，利用参考模型中的语义知识准确评估单个层的训练可塑性，并安全地冻结已收敛的层，从而节省相应的反向计算和通信。

    Egeria is a knowledge-guided DNN training system that skips computing and communication through DNN layer freezing, accurately evaluates individual layers' training plasticity using semantic knowledge from a reference model, and safely freezes the converged ones, saving their corresponding backward computation and communication.

    训练深度神经网络（DNN）是一项耗时的任务。本文提出了一种基于知识引导的DNN训练系统Egeria，通过跳过DNN层的计算和通信来实现高效训练。我们的关键洞察是，内部DNN层的训练进度存在显著差异，前层通常比深层更早地得到很好的训练。为了探索这一点，我们首先引入了训练可塑性的概念，以量化内部DNN层的训练进度。然后，我们设计了Egeria，一种基于知识引导的DNN训练系统，利用参考模型中的语义知识准确评估单个层的训练可塑性，并安全地冻结已收敛的层，从而节省相应的反向计算和通信。

    Training deep neural networks (DNNs) is time-consuming. While most existing solutions try to overlap/schedule computation and communication for efficient training, this paper goes one step further by skipping computing and communication through DNN layer freezing. Our key insight is that the training progress of internal DNN layers differs significantly, and front layers often become well-trained much earlier than deep layers. To explore this, we first introduce the notion of training plasticity to quantify the training progress of internal DNN layers. Then we design Egeria, a knowledge-guided DNN training system that employs semantic knowledge from a reference model to accurately evaluate individual layers' training plasticity and safely freeze the converged ones, saving their corresponding backward computation and communication. Our reference model is generated on the fly using quantization techniques and runs forward operations asynchronously on available CPUs to minimize the overhe
    

