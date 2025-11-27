# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scaling Efficient LLMs](https://arxiv.org/abs/2402.14746) | 训练得到的LLM模型通常是稀疏的，为了提高效率，研究了在训练语料上达到所需准确度的参数最少的高效LLM模型，得出了参数数量与自然训练语料规模之间的关系，并指出扩展可以揭示新技能。 |
| [^2] | [Stochastic Approximation Approach to Federated Machine Learning](https://arxiv.org/abs/2402.12945) | 本文提出了一种基于随机逼近的联邦机器学习方法，通过使用近似样本梯度和缩小步长来定位成本函数的极小值，实现了在联邦学习中对神经网络模型进行协作训练的效果，并在数值模拟中与标准算法进行了比较。 |
| [^3] | [A Reinforcement Learning Approach for Dynamic Rebalancing in Bike-Sharing System](https://arxiv.org/abs/2402.03589) | 本研究介绍了一种针对自行车共享系统中动态再平衡问题的时空强化学习算法，通过多智能体马尔可夫决策过程实现独立和协作的车辆再平衡，解决了传统数学优化方法的不实际限制。 |
| [^4] | [Decentralized Bilevel Optimization over Graphs: Loopless Algorithmic Update and Transient Iteration Complexity](https://arxiv.org/abs/2402.03167) | 本文提出了一种单循环的去中心化双级优化算法（D-SOBA），首次阐明了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA在渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性方面达到了最先进水平。 |
| [^5] | [TinyFormer: Efficient Transformer Design and Deployment on Tiny Devices.](http://arxiv.org/abs/2311.01759) | TinyFormer是一个具有SuperNAS、SparseNAS和SparseEngine组成的框架，专门用于在MCUs上开发和部署资源高效的transformer模型。其创新之处在于提出了SparseEngine，这是第一个可以在MCUs上执行稀疏模型的transformer推理的部署框架。 |
| [^6] | [Federated Learning with Differential Privacy for End-to-End Speech Recognition.](http://arxiv.org/abs/2310.00098) | 本文提出了一种基于联邦学习和差分隐私的端到端语音识别方法，探索了大型Transformer模型的不同方面，并建立了基线结果。 |
| [^7] | [A Scale-Invariant Task Balancing Approach for Multi-Task Learning.](http://arxiv.org/abs/2308.12029) | 这篇论文提出了一种尺度不变的多任务学习方法（SI-MTL），通过对任务损失进行对数变换和对任务梯度进行归一化，解决了多任务学习中的任务平衡问题，并在多个基准数据集上取得了领先的性能。 |
| [^8] | [LTD: Low Temperature Distillation for Robust Adversarial Training.](http://arxiv.org/abs/2111.02331) | 本文提出了一种名为低温蒸馏（LTD）的新方法，通过使用修改的知识蒸馏框架生成软标签，解决了对抗训练中常用的独热向量标签带来的学习困难问题，提高了模型的稳健性。 |

# 详细

[^1]: 扩展高效的LLM模型

    Scaling Efficient LLMs

    [https://arxiv.org/abs/2402.14746](https://arxiv.org/abs/2402.14746)

    训练得到的LLM模型通常是稀疏的，为了提高效率，研究了在训练语料上达到所需准确度的参数最少的高效LLM模型，得出了参数数量与自然训练语料规模之间的关系，并指出扩展可以揭示新技能。

    

    训练得到的LLM模型通常是稀疏的，即大部分参数为零，这引发了关于效率的问题。为此，我们研究了高效的LLM模型，即那些在训练语料上达到所需准确度的参数最少。具体地，我们比较了当前规模下训练损失的理论和实证估计，以获得自然训练语料中独特序列数量上下界的数量。我们的结果暗示：(1)要在训练语料中表示的技能数量翻倍，需要将语料规模大约扩展三到五倍，(2)对于高效的LLM模型，参数数量$N$和自然训练语料规模$D$满足$N \sim D^{0.58}$的关系，(3)如果一个LLM模型的参数数量小于训练语料中的独特序列数量，扩展可以揭示出新的技能。

    arXiv:2402.14746v1 Announce Type: new  Abstract: Trained LLMs are typically sparse in that most of the parameters are zero, raising questions on efficiency. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, we compare theoretical and empirical estimates for training loss at current scale to obtain upper and lower bounds on the number of unique sequences in a natural training corpus as a function of its size. Our result implies (1) to double the number of skills represented in a training corpus, the corpus must scale roughly between three and five fold (2) for efficient LLMs, the number of parameters $N$ and the size $D$ of a natural training corpus scale as $N \sim D^{0.58}$ (3) if the number of parameters of an LLM is smaller than the number of unique sequences in the training corpus, scaling up can uncover emergent skills.
    
[^2]: 基于随机逼近的联邦机器学习方法

    Stochastic Approximation Approach to Federated Machine Learning

    [https://arxiv.org/abs/2402.12945](https://arxiv.org/abs/2402.12945)

    本文提出了一种基于随机逼近的联邦机器学习方法，通过使用近似样本梯度和缩小步长来定位成本函数的极小值，实现了在联邦学习中对神经网络模型进行协作训练的效果，并在数值模拟中与标准算法进行了比较。

    

    本文在随机逼近（SA）框架下研究了联邦学习（FL）。 FL是一种协作方式，用于跨不同参与方或客户端训练神经网络模型，而无需将它们的数据集中。 每个客户端将根据各自的数据训练一个模型，并定期将权重发送到服务器进行聚合。 服务器对这些权重进行聚合，然后客户端使用这些权重重新初始化其神经网络并继续训练。 SA是一种使用近似样本梯度和缩小步长来定位成本函数极小值的迭代算法。 本文中，客户端使用随机逼近迭代更新其神经网络的权重。 结果表明，聚合权重跟踪一个自治ODE。 进行了数值模拟，并将结果与FedAvg和FedProx等标准算法进行了比较。

    arXiv:2402.12945v1 Announce Type: new  Abstract: This paper examines Federated learning (FL) in a Stochastic Approximation (SA) framework. FL is a collaborative way to train neural network models across various participants or clients without centralizing their data. Each client will train a model on their respective data and send the weights across to a the server periodically for aggregation. The server aggregates these weights which are then used by the clients to re-initialize their neural network and continue the training. SA is an iterative algorithm that uses approximate sample gradients and tapering step size to locate a minimizer of a cost function. In this paper the clients use a stochastic approximation iterate to update the weights of its neural network. It is shown that the aggregated weights track an autonomous ODE. Numerical simulations are performed and the results are compared with standard algorithms like FedAvg and FedProx. It is observed that the proposed algorithm 
    
[^3]: 自行车共享系统中动态再平衡的强化学习方法

    A Reinforcement Learning Approach for Dynamic Rebalancing in Bike-Sharing System

    [https://arxiv.org/abs/2402.03589](https://arxiv.org/abs/2402.03589)

    本研究介绍了一种针对自行车共享系统中动态再平衡问题的时空强化学习算法，通过多智能体马尔可夫决策过程实现独立和协作的车辆再平衡，解决了传统数学优化方法的不实际限制。

    

    自行车共享系统提供环保的城市出行方式，有助于缓解交通拥堵，促进健康生活方式。由于行程需求的随机性，这些系统的有效运营和保持高客户满意度具有挑战性，常常出现满站或空站现象。为了解决这个问题，使用车辆重新分配自行车到不同站点的再平衡策略至关重要。本文引入了一种时空强化学习算法，用于解决带有多辆车辆的动态再平衡问题。首先，在连续时间框架中将问题建模为多智能体马尔可夫决策过程。这允许独立和协作的车辆再平衡，消除了基于时间离散化模型的不切实际的限制。

    Bike-Sharing Systems provide eco-friendly urban mobility, contributing to the alleviation of traffic congestion and to healthier lifestyles. Efficiently operating such systems and maintaining high customer satisfaction is challenging due to the stochastic nature of trip demand, leading to full or empty stations. Devising effective rebalancing strategies using vehicles to redistribute bikes among stations is therefore of uttermost importance for operators. As a promising alternative to classical mathematical optimization, reinforcement learning is gaining ground to solve sequential decision-making problems. This paper introduces a spatio-temporal reinforcement learning algorithm for the dynamic rebalancing problem with multiple vehicles. We first formulate the problem as a Multi-agent Markov Decision Process in a continuous time framework. This allows for independent and cooperative vehicle rebalancing, eliminating the impractical restriction of time-discretized models where vehicle dep
    
[^4]: 图上的去中心化双级优化: 无环算法更新和瞬态迭代复杂性

    Decentralized Bilevel Optimization over Graphs: Loopless Algorithmic Update and Transient Iteration Complexity

    [https://arxiv.org/abs/2402.03167](https://arxiv.org/abs/2402.03167)

    本文提出了一种单循环的去中心化双级优化算法（D-SOBA），首次阐明了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA在渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性方面达到了最先进水平。

    

    随机双级优化（SBO）在处理嵌套结构方面的多样性使其在机器学习中变得越来越重要。为了解决大规模SBO，去中心化方法作为有效的范例出现，其中节点与直接相邻节点进行通信，无需中央服务器，从而提高通信效率和增强算法的稳健性。然而，当前的去中心化SBO算法面临挑战，包括昂贵的内部循环更新和对网络拓扑、数据异构性和嵌套双级算法结构的影响不明确。在本文中，我们引入了一种单循环的去中心化SBO（D-SOBA）算法，并建立了其瞬态迭代复杂性，首次澄清了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA实现了最先进的渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性。

    Stochastic bilevel optimization (SBO) is becoming increasingly essential in machine learning due to its versatility in handling nested structures. To address large-scale SBO, decentralized approaches have emerged as effective paradigms in which nodes communicate with immediate neighbors without a central server, thereby improving communication efficiency and enhancing algorithmic robustness. However, current decentralized SBO algorithms face challenges, including expensive inner-loop updates and unclear understanding of the influence of network topology, data heterogeneity, and the nested bilevel algorithmic structures. In this paper, we introduce a single-loop decentralized SBO (D-SOBA) algorithm and establish its transient iteration complexity, which, for the first time, clarifies the joint influence of network topology and data heterogeneity on decentralized bilevel algorithms. D-SOBA achieves the state-of-the-art asymptotic rate, asymptotic gradient/Hessian complexity, and transien
    
[^5]: TinyFormer: 高效的Transformer设计和在小型设备上的部署

    TinyFormer: Efficient Transformer Design and Deployment on Tiny Devices. (arXiv:2311.01759v1 [cs.LG])

    [http://arxiv.org/abs/2311.01759](http://arxiv.org/abs/2311.01759)

    TinyFormer是一个具有SuperNAS、SparseNAS和SparseEngine组成的框架，专门用于在MCUs上开发和部署资源高效的transformer模型。其创新之处在于提出了SparseEngine，这是第一个可以在MCUs上执行稀疏模型的transformer推理的部署框架。

    

    在各种嵌入式物联网应用中，以微控制器单元（MCUs）为代表的小型设备上开发深度学习模型引起了广泛关注。然而，由于严重的硬件资源限制，如何高效地设计和部署最新的先进模型（如transformer）在小型设备上是一项挑战。在这项工作中，我们提出了TinyFormer，这是一个特别设计用于在MCUs上开发和部署资源高效的transformer的框架。TinyFormer主要由SuperNAS、SparseNAS和SparseEngine组成。其中，SuperNAS旨在从广大的搜索空间中寻找适当的超网络。SparseNAS评估最佳的稀疏单路径模型，包括从已识别的超网络中提取的transformer架构。最后，SparseEngine将搜索到的稀疏模型高效地部署到MCUs上。据我们所知，SparseEngine是第一个能够在MCUs上执行稀疏模型的transformer推理的部署框架。在CIFAR-10数据集上的评估结果表明，TinyFormer在保持推理精度的同时，相比于传统的transformer模型，减少了大约78％的推理计算量和53％的模型大小。

    Developing deep learning models on tiny devices (e.g. Microcontroller units, MCUs) has attracted much attention in various embedded IoT applications. However, it is challenging to efficiently design and deploy recent advanced models (e.g. transformers) on tiny devices due to their severe hardware resource constraints. In this work, we propose TinyFormer, a framework specifically designed to develop and deploy resource-efficient transformers on MCUs. TinyFormer mainly consists of SuperNAS, SparseNAS and SparseEngine. Separately, SuperNAS aims to search for an appropriate supernet from a vast search space. SparseNAS evaluates the best sparse single-path model including transformer architecture from the identified supernet. Finally, SparseEngine efficiently deploys the searched sparse models onto MCUs. To the best of our knowledge, SparseEngine is the first deployment framework capable of performing inference of sparse models with transformer on MCUs. Evaluation results on the CIFAR-10 da
    
[^6]: 使用差分隐私的联邦学习进行端到端语音识别

    Federated Learning with Differential Privacy for End-to-End Speech Recognition. (arXiv:2310.00098v1 [cs.LG])

    [http://arxiv.org/abs/2310.00098](http://arxiv.org/abs/2310.00098)

    本文提出了一种基于联邦学习和差分隐私的端到端语音识别方法，探索了大型Transformer模型的不同方面，并建立了基线结果。

    

    联邦学习是一种有前景的训练机器学习模型的方法，但在自动语音识别领域仅限于初步探索。此外，联邦学习不能本质上保证用户隐私，并需要差分隐私来提供稳健的隐私保证。然而，我们还不清楚在自动语音识别中应用差分隐私的先前工作。本文旨在通过为联邦学习提供差分隐私的自动语音识别基准，并建立第一个基线来填补这一研究空白。我们扩展了现有的联邦学习自动语音识别研究，探索了最新的大型端到端Transformer模型的不同方面：架构设计，种子模型，数据异质性，领域转移，以及cohort大小的影响。通过合理的中央聚合数量，我们能够训练出即使在异构数据、来自另一个领域的种子模型或无预先训练的情况下仍然接近最优的联邦学习模型。

    While federated learning (FL) has recently emerged as a promising approach to train machine learning models, it is limited to only preliminary explorations in the domain of automatic speech recognition (ASR). Moreover, FL does not inherently guarantee user privacy and requires the use of differential privacy (DP) for robust privacy guarantees. However, we are not aware of prior work on applying DP to FL for ASR. In this paper, we aim to bridge this research gap by formulating an ASR benchmark for FL with DP and establishing the first baselines. First, we extend the existing research on FL for ASR by exploring different aspects of recent $\textit{large end-to-end transformer models}$: architecture design, seed models, data heterogeneity, domain shift, and impact of cohort size. With a $\textit{practical}$ number of central aggregations we are able to train $\textbf{FL models}$ that are \textbf{nearly optimal} even with heterogeneous data, a seed model from another domain, or no pre-trai
    
[^7]: 一种针对多任务学习的尺度不变任务平衡方法

    A Scale-Invariant Task Balancing Approach for Multi-Task Learning. (arXiv:2308.12029v1 [cs.LG])

    [http://arxiv.org/abs/2308.12029](http://arxiv.org/abs/2308.12029)

    这篇论文提出了一种尺度不变的多任务学习方法（SI-MTL），通过对任务损失进行对数变换和对任务梯度进行归一化，解决了多任务学习中的任务平衡问题，并在多个基准数据集上取得了领先的性能。

    

    多任务学习（MTL）是一种同时学习多个相关任务的学习范式，在各个领域取得了巨大的成功。然而，任务平衡仍然是MTL中的一个重要挑战，损失/梯度尺度的不平衡经常导致性能折中。本文提出了一种尺度不变的多任务学习（SI-MTL）方法，从损失和梯度角度缓解了任务平衡问题。具体来说，SI-MTL包含对所有任务损失进行的对数变换，以确保在损失水平上具有尺度不变性，以及一种梯度平衡方法SI-G，它将所有任务的梯度归一化为与最大梯度范数相同的大小。在几个基准数据集上进行的大量实验一致证明了SI-G的有效性和SI-MTL的最先进性能。

    Multi-task learning (MTL), a learning paradigm to learn multiple related tasks simultaneously, has achieved great success in various fields. However, task-balancing remains a significant challenge in MTL, with the disparity in loss/gradient scales often leading to performance compromises. In this paper, we propose a Scale-Invariant Multi-Task Learning (SI-MTL) method to alleviate the task-balancing problem from both loss and gradient perspectives. Specifically, SI-MTL contains a logarithm transformation which is performed on all task losses to ensure scale-invariant at the loss level, and a gradient balancing method, SI-G, which normalizes all task gradients to the same magnitude as the maximum gradient norm. Extensive experiments conducted on several benchmark datasets consistently demonstrate the effectiveness of SI-G and the state-of-the-art performance of SI-MTL.
    
[^8]: 低温蒸馏：用于稳健对抗训练的方法

    LTD: Low Temperature Distillation for Robust Adversarial Training. (arXiv:2111.02331v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2111.02331](http://arxiv.org/abs/2111.02331)

    本文提出了一种名为低温蒸馏（LTD）的新方法，通过使用修改的知识蒸馏框架生成软标签，解决了对抗训练中常用的独热向量标签带来的学习困难问题，提高了模型的稳健性。

    

    对抗训练已经被广泛应用于增强神经网络模型对抗攻击的稳健性。尽管神经网络模型很受欢迎，但是这些模型的自然准确性和稳健准确性之间存在着显著差距。本文的主要贡献是发现了这个差距的一个主要原因是常用的独热向量作为标签，这阻碍了图像识别的学习过程。用独热向量表示模糊图像是不准确的，可能导致模型得到次优解。为了解决这个问题，我们提出了一种新颖的方法，称之为低温蒸馏（LTD），它使用修改的知识蒸馏框架生成软标签。与以前的方法不同，LTD在教师模型中使用相对较低的温度，而对教师和学生模型使用固定但不同的温度。这个修改可以提高模型的稳健性，而不会遇到已经在先前工作中解决的梯度掩码问题。

    Adversarial training has been widely used to enhance the robustness of neural network models against adversarial attacks. Despite the popularity of neural network models, a significant gap exists between the natural and robust accuracy of these models. In this paper, we identify one of the primary reasons for this gap is the common use of one-hot vectors as labels, which hinders the learning process for image recognition. Representing ambiguous images with one-hot vectors is imprecise and may lead the model to suboptimal solutions. To overcome this issue, we propose a novel method called Low Temperature Distillation (LTD) that generates soft labels using the modified knowledge distillation framework. Unlike previous approaches, LTD uses a relatively low temperature in the teacher model and fixed, but different temperatures for the teacher and student models. This modification boosts the model's robustness without encountering the gradient masking problem that has been addressed in defe
    

