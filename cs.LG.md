# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Heterophily-Aware Fair Recommendation using Graph Convolutional Networks](https://arxiv.org/abs/2402.03365) | 本文提出了一种利用图卷积网络的公平推荐系统，名为HetroFair，旨在提高项目侧的公平性。HetroFair使用公平注意力和异质性特征加权两个组件来生成具有公平性意识的嵌入。 |
| [^2] | [Discovering group dynamics in synchronous time series via hierarchical recurrent switching-state models.](http://arxiv.org/abs/2401.14973) | 通过分层的循环切换状态模型，我们可以无监督地同时解释系统级和个体级的动态，从而更好地建模同步时间序列中的群体动态。 |
| [^3] | [End-To-End Set-Based Training for Neural Network Verification.](http://arxiv.org/abs/2401.14961) | 本论文提出了一种端到端基于集合的训练方法，用于训练鲁棒性神经网络进行形式化验证，并证明该方法能够简化验证过程并有效训练出易于验证的神经网络。 |
| [^4] | [Food Classification using Joint Representation of Visual and Textual Data.](http://arxiv.org/abs/2308.02562) | 本研究提出了一种使用联合表示的多模态分类框架，通过修改版的EfficientNet和Mish激活函数实现图像分类，使用基于BERT的网络实现文本分类。实验结果表明，所提出的网络在图像和文本分类上表现优于其他方法，准确率提高了11.57%和6.34%。比较分析还证明了所提出方法的效率和鲁棒性。 |

# 详细

[^1]: 利用图卷积网络的異质友善推荐方法

    Heterophily-Aware Fair Recommendation using Graph Convolutional Networks

    [https://arxiv.org/abs/2402.03365](https://arxiv.org/abs/2402.03365)

    本文提出了一种利用图卷积网络的公平推荐系统，名为HetroFair，旨在提高项目侧的公平性。HetroFair使用公平注意力和异质性特征加权两个组件来生成具有公平性意识的嵌入。

    

    近年来，图神经网络（GNNs）已成为提高推荐系统准确性和性能的流行工具。现代推荐系统不仅设计为为最终用户服务，还要让其他参与者（如项目和项目供应商）从中受益。这些参与者可能具有不同或冲突的目标和利益，这引发了对公平性和流行度偏差考虑的需求。基于GNN的推荐方法也面临不公平性和流行度偏差的挑战，其归一化和聚合过程受到这些挑战的影响。在本文中，我们提出了一种公平的基于GNN的推荐系统，称为HetroFair，旨在提高项目侧的公平性。HetroFair使用两个独立的组件生成具有公平性意识的嵌入：i）公平注意力，它在GNN的归一化过程中结合了点积，以减少节点度数的影响；ii）异质性特征加权，为不同的特征分配不同的权重。

    In recent years, graph neural networks (GNNs) have become a popular tool to improve the accuracy and performance of recommender systems. Modern recommender systems are not only designed to serve the end users, but also to benefit other participants, such as items and items providers. These participants may have different or conflicting goals and interests, which raise the need for fairness and popularity bias considerations. GNN-based recommendation methods also face the challenges of unfairness and popularity bias and their normalization and aggregation processes suffer from these challenges. In this paper, we propose a fair GNN-based recommender system, called HetroFair, to improve items' side fairness. HetroFair uses two separate components to generate fairness-aware embeddings: i) fairness-aware attention which incorporates dot product in the normalization process of GNNs, to decrease the effect of nodes' degrees, and ii) heterophily feature weighting to assign distinct weights to 
    
[^2]: 通过分层循环切换状态模型发现同步时间序列中的群体动态

    Discovering group dynamics in synchronous time series via hierarchical recurrent switching-state models. (arXiv:2401.14973v1 [stat.ML])

    [http://arxiv.org/abs/2401.14973](http://arxiv.org/abs/2401.14973)

    通过分层的循环切换状态模型，我们可以无监督地同时解释系统级和个体级的动态，从而更好地建模同步时间序列中的群体动态。

    

    我们致力于对同一时间段内多个实体相互作用而产生的时间序列集合进行建模。最近的研究集中在建模个体时间序列方面对我们的预期应用是不足够的，其中集体系统级行为影响着个体实体的轨迹。为了解决这类问题，我们提出了一种新的分层切换状态模型，可以以无监督的方式训练，同时解释系统级和个体级的动态。我们采用了一个隐含的系统级离散状态马尔可夫链，驱动着隐含的实体级链，进而控制每个观测时间序列的动态。观测结果在实体和系统级的链之间进行反馈，通过依赖于上下文的状态转换来提高灵活性。我们的分层切换循环动力学模型可以通过封闭形式的变分坐标上升更新来学习，其在个体数量上呈线性扩展。

    We seek to model a collection of time series arising from multiple entities interacting over the same time period. Recent work focused on modeling individual time series is inadequate for our intended applications, where collective system-level behavior influences the trajectories of individual entities. To address such problems, we present a new hierarchical switching-state model that can be trained in an unsupervised fashion to simultaneously explain both system-level and individual-level dynamics. We employ a latent system-level discrete state Markov chain that drives latent entity-level chains which in turn govern the dynamics of each observed time series. Feedback from the observations to the chains at both the entity and system levels improves flexibility via context-dependent state transitions. Our hierarchical switching recurrent dynamical models can be learned via closed-form variational coordinate ascent updates to all latent chains that scale linearly in the number of indivi
    
[^3]: 神经网络验证的端到端基于集合的训练方法

    End-To-End Set-Based Training for Neural Network Verification. (arXiv:2401.14961v1 [cs.LG])

    [http://arxiv.org/abs/2401.14961](http://arxiv.org/abs/2401.14961)

    本论文提出了一种端到端基于集合的训练方法，用于训练鲁棒性神经网络进行形式化验证，并证明该方法能够简化验证过程并有效训练出易于验证的神经网络。

    

    神经网络容易受到对抗性攻击，即微小的输入扰动可能导致神经网络输出产生重大变化。安全关键环境需要对输入扰动具有鲁棒性的神经网络。然而，训练和形式化验证鲁棒性神经网络是具有挑战性的。我们首次采用端到端基于集合的训练方法来解决这个挑战，该训练方法能够训练出可进行形式化验证的鲁棒性神经网络。我们的训练方法能够大大简化已训练神经网络的后续形式化鲁棒性验证过程。相比于以往的研究主要关注增强神经网络训练的对抗性攻击，我们的方法利用基于集合的计算来训练整个扰动输入集合上的神经网络。此外，我们证明我们的基于集合的训练方法可以有效训练出易于验证的鲁棒性神经网络。

    Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can result in substantially different outputs of a neural network. Safety-critical environments require neural networks that are robust against input perturbations. However, training and formally verifying robust neural networks is challenging. We address this challenge by employing, for the first time, a end-to-end set-based training procedure that trains robust neural networks for formal verification. Our training procedure drastically simplifies the subsequent formal robustness verification of the trained neural network. While previous research has predominantly focused on augmenting neural network training with adversarial attacks, our approach leverages set-based computing to train neural networks with entire sets of perturbed inputs. Moreover, we demonstrate that our set-based training procedure effectively trains robust neural networks, which are easier to verify. In many cases, set-based trai
    
[^4]: 使用视觉和文本数据的联合表示进行食物分类

    Food Classification using Joint Representation of Visual and Textual Data. (arXiv:2308.02562v1 [cs.CV])

    [http://arxiv.org/abs/2308.02562](http://arxiv.org/abs/2308.02562)

    本研究提出了一种使用联合表示的多模态分类框架，通过修改版的EfficientNet和Mish激活函数实现图像分类，使用基于BERT的网络实现文本分类。实验结果表明，所提出的网络在图像和文本分类上表现优于其他方法，准确率提高了11.57%和6.34%。比较分析还证明了所提出方法的效率和鲁棒性。

    

    食物分类是健康保健中的重要任务。在这项工作中，我们提出了一个多模态分类框架，该框架使用了修改版的EfficientNet和Mish激活函数用于图像分类，同时使用传统的基于BERT的网络进行文本分类。我们在一个大型开源数据集UPMC Food-101上评估了所提出的网络和其他最先进的方法。实验结果显示，所提出的网络在图像和文本分类上的准确率分别比第二最好的方法提高了11.57%和6.34%。我们还比较了使用机器学习和深度学习模型进行文本分类的准确率、精确率和召回率。通过对图像和文本的预测结果进行比较分析，证明了所提出方法的效率和鲁棒性。

    Food classification is an important task in health care. In this work, we propose a multimodal classification framework that uses the modified version of EfficientNet with the Mish activation function for image classification, and the traditional BERT transformer-based network is used for text classification. The proposed network and the other state-of-the-art methods are evaluated on a large open-source dataset, UPMC Food-101. The experimental results show that the proposed network outperforms the other methods, a significant difference of 11.57% and 6.34% in accuracy is observed for image and text classification, respectively, when compared with the second-best performing method. We also compared the performance in terms of accuracy, precision, and recall for text classification using both machine learning and deep learning-based models. The comparative analysis from the prediction results of both images and text demonstrated the efficiency and robustness of the proposed approach.
    

