# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty estimation in spatial interpolation of satellite precipitation with ensemble learning](https://arxiv.org/abs/2403.10567) | 引入九种集成学习器并利用新颖特征工程策略，结合多种分位数回归算法，填补了空间插值中集成学习的不确定性估计领域的研究空白 |
| [^2] | [Node Duplication Improves Cold-start Link Prediction](https://arxiv.org/abs/2402.09711) | 本文研究了在链路预测中改进GNN在低度节点上的性能，提出了一种名为NodeDup的增强技术，通过复制低度节点并创建链接来提高性能。 |
| [^3] | [High-dimensional Linear Bandits with Knapsacks.](http://arxiv.org/abs/2311.01327) | 本文研究了具有背包约束的高维线性赌臂问题，利用稀疏结构实现改进遗憾。通过开发在线硬阈值算法和原始-对偶框架结合的方法，实现了对特征维度的对数改进的次线性遗憾。 |
| [^4] | [EqDrive: Efficient Equivariant Motion Forecasting with Multi-Modality for Autonomous Driving.](http://arxiv.org/abs/2310.17540) | 本研究发展了EqDrive模型，通过使用EqMotion等变粒子和人类预测模型以及多模式预测机制，在自动驾驶中实现了高效的车辆运动预测。该模型在模型容量较低、参数更少、训练时间显著缩短的情况下，取得了业界最先进的性能。 |
| [^5] | [Algorithmic Recourse for Anomaly Detection in Multivariate Time Series.](http://arxiv.org/abs/2309.16896) | 本论文提出了一种算法补救框架，RecAD，用于多变量时间序列异常检测。通过推荐以最小成本修复异常时间序列，该框架能够帮助领域专家理解如何修复异常行为，并在实验中证明了其有效性。 |
| [^6] | [Food Classification using Joint Representation of Visual and Textual Data.](http://arxiv.org/abs/2308.02562) | 本研究提出了一种使用联合表示的多模态分类框架，通过修改版的EfficientNet和Mish激活函数实现图像分类，使用基于BERT的网络实现文本分类。实验结果表明，所提出的网络在图像和文本分类上表现优于其他方法，准确率提高了11.57%和6.34%。比较分析还证明了所提出方法的效率和鲁棒性。 |
| [^7] | [You Can Generate It Again: Data-to-text Generation with Verification and Correction Prompting.](http://arxiv.org/abs/2306.15933) | 本文提出了一种多步骤生成、验证和纠正的数据生成文本方法，通过专门的错误指示提示来改善输出质量。 |
| [^8] | [Comparative Study of Coupling and Autoregressive Flows through Robust Statistical Tests.](http://arxiv.org/abs/2302.12024) | 本论文通过比较耦合流和自回归流的不同架构和多样目标分布，利用各种测试统计量进行性能比较，为正规化流的生成模型提供了深入的研究和实证评估。 |
| [^9] | [Impartial Games: A Challenge for Reinforcement Learning.](http://arxiv.org/abs/2205.12787) | AlphaZero-style reinforcement learning algorithms excel in various board games but face challenges with impartial games. The researchers present a concrete example of the game nim, and show that AlphaZero-style algorithms have difficulty learning these impartial games on larger board sizes. The difference between impartial games and partisan games can be explained by the vulnerability to adversarial attacks and perturbations. |

# 详细

[^1]: 集成学习中的卫星降水空间插值不确定性估计

    Uncertainty estimation in spatial interpolation of satellite precipitation with ensemble learning

    [https://arxiv.org/abs/2403.10567](https://arxiv.org/abs/2403.10567)

    引入九种集成学习器并利用新颖特征工程策略，结合多种分位数回归算法，填补了空间插值中集成学习的不确定性估计领域的研究空白

    

    arXiv:2403.10567v1 公告类型：新的 摘要：概率分布形式的预测对决策至关重要。分位数回归在空间插值设置中能够合并遥感和雨量数据，实现此目标。然而，在这种情境下，分位数回归算法的集成学习尚未被研究。本文通过引入九种基于分位数的集成学习器并将其应用于大型降水数据集来填补这一空白。我们采用了一种新颖的特征工程策略，将预测因子减少为相关位置的加权距离卫星降水，结合位置高程。我们的集成学习器包括六种堆叠方法和三种简单方法（均值、中位数、最佳组合器），结合了六种个体算法：分位数回归(QR)、分位数回归森林(QRF)、广义随机森林(GRF)、梯度提升机(GBM)、轻量级梯度提升机(LightGBM)和分位数回归神经网络

    arXiv:2403.10567v1 Announce Type: new  Abstract: Predictions in the form of probability distributions are crucial for decision-making. Quantile regression enables this within spatial interpolation settings for merging remote sensing and gauge precipitation data. However, ensemble learning of quantile regression algorithms remains unexplored in this context. Here, we address this gap by introducing nine quantile-based ensemble learners and applying them to large precipitation datasets. We employed a novel feature engineering strategy, reducing predictors to distance-weighted satellite precipitation at relevant locations, combined with location elevation. Our ensemble learners include six stacking and three simple methods (mean, median, best combiner), combining six individual algorithms: quantile regression (QR), quantile regression forests (QRF), generalized random forests (GRF), gradient boosting machines (GBM), light gradient boosting machines (LightGBM), and quantile regression neur
    
[^2]: 节点复制改善冷启动链路预测

    Node Duplication Improves Cold-start Link Prediction

    [https://arxiv.org/abs/2402.09711](https://arxiv.org/abs/2402.09711)

    本文研究了在链路预测中改进GNN在低度节点上的性能，提出了一种名为NodeDup的增强技术，通过复制低度节点并创建链接来提高性能。

    

    图神经网络（GNN）在图机器学习中非常突出，并在链路预测（LP）任务中展现了最先进的性能。然而，最近的研究表明，尽管整体上表现出色，GNN在低度节点上的表现却较差。在推荐系统等LP的实际应用中，改善低度节点的性能至关重要，因为这等同于解决冷启动问题，提高用户在少数观察的相互作用中的体验。本文研究了改进GNN在低度节点上的LP性能，同时保持其在高度节点上的性能，并提出了一种简单但非常有效的增强技术，称为NodeDup。具体而言，NodeDup在标准的监督LP训练方案中，在低度节点上复制节点并在节点和其副本之间创建链接。通过利用“多视图”视角，该方法可以显著提高LP的性能。

    arXiv:2402.09711v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) are prominent in graph machine learning and have shown state-of-the-art performance in Link Prediction (LP) tasks. Nonetheless, recent studies show that GNNs struggle to produce good results on low-degree nodes despite their overall strong performance. In practical applications of LP, like recommendation systems, improving performance on low-degree nodes is critical, as it amounts to tackling the cold-start problem of improving the experiences of users with few observed interactions. In this paper, we investigate improving GNNs' LP performance on low-degree nodes while preserving their performance on high-degree nodes and propose a simple yet surprisingly effective augmentation technique called NodeDup. Specifically, NodeDup duplicates low-degree nodes and creates links between nodes and their own duplicates before following the standard supervised LP training scheme. By leveraging a ''multi-view'' perspectiv
    
[^3]: 具有背包约束的高维线性赌臂问题研究

    High-dimensional Linear Bandits with Knapsacks. (arXiv:2311.01327v1 [cs.LG])

    [http://arxiv.org/abs/2311.01327](http://arxiv.org/abs/2311.01327)

    本文研究了具有背包约束的高维线性赌臂问题，利用稀疏结构实现改进遗憾。通过开发在线硬阈值算法和原始-对偶框架结合的方法，实现了对特征维度的对数改进的次线性遗憾。

    

    我们研究了在特征维度较大的高维设置下的具有背包约束的上下文赌臂问题。每个手臂拉动的奖励等于稀疏高维权重向量与当前到达的特征的乘积，加上额外的随机噪声。在本文中，我们研究如何利用这种稀疏结构来实现CBwK问题的改进遗憾。为此，我们首先开发了一种在线的硬阈值算法的变体，以在线方式进行稀疏估计。我们进一步将我们的在线估计器与原始-对偶框架结合起来，在每个背包约束上分配一个对偶变量，并利用在线学习算法来更新对偶变量，从而控制背包容量的消耗。我们证明了这种集成方法使我们能够实现对特征维度的对数改进的次线性遗憾，从而改进了多项式相关性。

    We study the contextual bandits with knapsack (CBwK) problem under the high-dimensional setting where the dimension of the feature is large. The reward of pulling each arm equals the multiplication of a sparse high-dimensional weight vector and the feature of the current arrival, with additional random noise. In this paper, we investigate how to exploit this sparsity structure to achieve improved regret for the CBwK problem. To this end, we first develop an online variant of the hard thresholding algorithm that performs the sparse estimation in an online manner. We further combine our online estimator with a primal-dual framework, where we assign a dual variable to each knapsack constraint and utilize an online learning algorithm to update the dual variable, thereby controlling the consumption of the knapsack capacity. We show that this integrated approach allows us to achieve a sublinear regret that depends logarithmically on the feature dimension, thus improving the polynomial depend
    
[^4]: EqDrive: 自动驾驶的高效等变运动预测与多模式处理

    EqDrive: Efficient Equivariant Motion Forecasting with Multi-Modality for Autonomous Driving. (arXiv:2310.17540v1 [cs.RO])

    [http://arxiv.org/abs/2310.17540](http://arxiv.org/abs/2310.17540)

    本研究发展了EqDrive模型，通过使用EqMotion等变粒子和人类预测模型以及多模式预测机制，在自动驾驶中实现了高效的车辆运动预测。该模型在模型容量较低、参数更少、训练时间显著缩短的情况下，取得了业界最先进的性能。

    

    在自动驾驶中预测车辆运动需要对车辆间的相互作用有深入的理解，并保持在欧几里得几何变换下的运动等变性。传统模型往往缺乏处理自动驾驶车辆中复杂动力学和场景中各主体之间交互关系所需的复杂性。因此，这些模型具有较低的模型容量，导致更高的预测误差和较低的训练效率。在我们的研究中，我们使用EqMotion，一个领先的等变粒子和人类预测模型，该模型还考虑到不变的主体间相互作用，用于多代理车辆运动预测任务。此外，我们使用多模式预测机制以概率化方式考虑多个可能的未来路径。通过利用EqMotion，我们的模型在参数更少（120万）和训练时间显著缩短（少于..）的情况下实现了业界最先进的性能。

    Forecasting vehicular motions in autonomous driving requires a deep understanding of agent interactions and the preservation of motion equivariance under Euclidean geometric transformations. Traditional models often lack the sophistication needed to handle the intricate dynamics inherent to autonomous vehicles and the interaction relationships among agents in the scene. As a result, these models have a lower model capacity, which then leads to higher prediction errors and lower training efficiency. In our research, we employ EqMotion, a leading equivariant particle, and human prediction model that also accounts for invariant agent interactions, for the task of multi-agent vehicle motion forecasting. In addition, we use a multi-modal prediction mechanism to account for multiple possible future paths in a probabilistic manner. By leveraging EqMotion, our model achieves state-of-the-art (SOTA) performance with fewer parameters (1.2 million) and a significantly reduced training time (less 
    
[^5]: 多变量时间序列异常检测的算法补救方法

    Algorithmic Recourse for Anomaly Detection in Multivariate Time Series. (arXiv:2309.16896v1 [cs.LG])

    [http://arxiv.org/abs/2309.16896](http://arxiv.org/abs/2309.16896)

    本论文提出了一种算法补救框架，RecAD，用于多变量时间序列异常检测。通过推荐以最小成本修复异常时间序列，该框架能够帮助领域专家理解如何修复异常行为，并在实验中证明了其有效性。

    

    由于广泛的应用领域，多变量时间序列的异常检测已经受到广泛研究。多变量时间序列中的异常通常表示临界事件，例如系统故障或外部攻击。因此，除了在异常检测方面有效之外，推荐异常缓解行动在实践中也很重要但研究不足。在这项工作中，我们专注于时间序列异常检测中的算法补救方法，即推荐以最小成本修复异常时间序列，以便领域专家可以理解如何修复异常行为。为此，我们提出了一种算法补救框架，称为RecAD，可以推荐翻转异常时间步骤的补救行动。对两个合成数据集和一个真实数据集的实验结果显示了我们框架的有效性。

    Anomaly detection in multivariate time series has received extensive study due to the wide spectrum of applications. An anomaly in multivariate time series usually indicates a critical event, such as a system fault or an external attack. Therefore, besides being effective in anomaly detection, recommending anomaly mitigation actions is also important in practice yet under-investigated. In this work, we focus on algorithmic recourse in time series anomaly detection, which is to recommend fixing actions on abnormal time series with a minimum cost so that domain experts can understand how to fix the abnormal behavior. To this end, we propose an algorithmic recourse framework, called RecAD, which can recommend recourse actions to flip the abnormal time steps. Experiments on two synthetic and one real-world datasets show the effectiveness of our framework.
    
[^6]: 使用视觉和文本数据的联合表示进行食物分类

    Food Classification using Joint Representation of Visual and Textual Data. (arXiv:2308.02562v1 [cs.CV])

    [http://arxiv.org/abs/2308.02562](http://arxiv.org/abs/2308.02562)

    本研究提出了一种使用联合表示的多模态分类框架，通过修改版的EfficientNet和Mish激活函数实现图像分类，使用基于BERT的网络实现文本分类。实验结果表明，所提出的网络在图像和文本分类上表现优于其他方法，准确率提高了11.57%和6.34%。比较分析还证明了所提出方法的效率和鲁棒性。

    

    食物分类是健康保健中的重要任务。在这项工作中，我们提出了一个多模态分类框架，该框架使用了修改版的EfficientNet和Mish激活函数用于图像分类，同时使用传统的基于BERT的网络进行文本分类。我们在一个大型开源数据集UPMC Food-101上评估了所提出的网络和其他最先进的方法。实验结果显示，所提出的网络在图像和文本分类上的准确率分别比第二最好的方法提高了11.57%和6.34%。我们还比较了使用机器学习和深度学习模型进行文本分类的准确率、精确率和召回率。通过对图像和文本的预测结果进行比较分析，证明了所提出方法的效率和鲁棒性。

    Food classification is an important task in health care. In this work, we propose a multimodal classification framework that uses the modified version of EfficientNet with the Mish activation function for image classification, and the traditional BERT transformer-based network is used for text classification. The proposed network and the other state-of-the-art methods are evaluated on a large open-source dataset, UPMC Food-101. The experimental results show that the proposed network outperforms the other methods, a significant difference of 11.57% and 6.34% in accuracy is observed for image and text classification, respectively, when compared with the second-best performing method. We also compared the performance in terms of accuracy, precision, and recall for text classification using both machine learning and deep learning-based models. The comparative analysis from the prediction results of both images and text demonstrated the efficiency and robustness of the proposed approach.
    
[^7]: 通过验证和纠正提示进行数据生成文本生成

    You Can Generate It Again: Data-to-text Generation with Verification and Correction Prompting. (arXiv:2306.15933v1 [cs.CL])

    [http://arxiv.org/abs/2306.15933](http://arxiv.org/abs/2306.15933)

    本文提出了一种多步骤生成、验证和纠正的数据生成文本方法，通过专门的错误指示提示来改善输出质量。

    

    尽管现有模型取得了显著进展，从结构化数据输入生成文本描述（称为数据生成文本）仍然是一个具有挑战性的任务。在本文中，我们提出了一种新的方法，通过引入包括生成、验证和纠正阶段的多步骤过程，超越了传统的一次性生成方法。我们的方法，VCP（验证和纠正提示），从模型生成初始输出开始。然后，我们继续验证所生成文本的不同方面的正确性。验证步骤的观察结果被转化为专门的错误指示提示，该提示指示模型在重新生成输出时考虑已识别的错误。为了增强模型的纠正能力，我们开发了一个经过精心设计的培训过程。该过程使模型能够融入错误指示提示的反馈，从而改善输出生成。

    Despite significant advancements in existing models, generating text descriptions from structured data input, known as data-to-text generation, remains a challenging task. In this paper, we propose a novel approach that goes beyond traditional one-shot generation methods by introducing a multi-step process consisting of generation, verification, and correction stages. Our approach, VCP(Verification and Correction Prompting), begins with the model generating an initial output. We then proceed to verify the correctness of different aspects of the generated text. The observations from the verification step are converted into a specialized error-indication prompt, which instructs the model to regenerate the output while considering the identified errors. To enhance the model's correction ability, we have developed a carefully designed training procedure. This procedure enables the model to incorporate feedback from the error-indication prompt, resulting in improved output generation. Throu
    
[^8]: 比较耦合流和自回归流的鲁棒统计检验研究

    Comparative Study of Coupling and Autoregressive Flows through Robust Statistical Tests. (arXiv:2302.12024v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.12024](http://arxiv.org/abs/2302.12024)

    本论文通过比较耦合流和自回归流的不同架构和多样目标分布，利用各种测试统计量进行性能比较，为正规化流的生成模型提供了深入的研究和实证评估。

    

    正规化流已经成为一种强大的生成模型，因为它们不仅能够有效地对复杂目标分布进行采样，而且还通过构造提供密度估计。我们在这里提出了对耦合流和自回归流进行深入比较的研究，包括仿射和有理二次样条类型的四种不同架构：实值非体积保持（RealNVP）、掩蔽自回归流（MAF）、耦合有理二次样条（C-RQS）和自回归有理二次样条（A-RQS）。我们关注一组从4维到400维递增的多模态目标分布。通过使用不同的两样本测试的测试统计量进行比较，我们建立了已知距离度量的测试统计量：切片Wasserstein距离、维度平均一维Kolmogorov-Smirnov检验和相关矩阵之差的Frobenius范数。另外，我们还包括了以下估计：

    Normalizing Flows have emerged as a powerful brand of generative models, as they not only allow for efficient sampling of complicated target distributions, but also deliver density estimation by construction. We propose here an in-depth comparison of coupling and autoregressive flows, both of the affine and rational quadratic spline type, considering four different architectures: Real-valued Non-Volume Preserving (RealNVP), Masked Autoregressive Flow (MAF), Coupling Rational Quadratic Spline (C-RQS), and Autoregressive Rational Quadratic Spline (A-RQS). We focus on a set of multimodal target distributions of increasing dimensionality ranging from 4 to 400. The performances are compared by means of different test-statistics for two-sample tests, built from known distance measures: the sliced Wasserstein distance, the dimension-averaged one-dimensional Kolmogorov-Smirnov test, and the Frobenius norm of the difference between correlation matrices. Furthermore, we include estimations of th
    
[^9]: 公正游戏：对强化学习的挑战

    Impartial Games: A Challenge for Reinforcement Learning. (arXiv:2205.12787v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.12787](http://arxiv.org/abs/2205.12787)

    AlphaZero-style reinforcement learning algorithms excel in various board games but face challenges with impartial games. The researchers present a concrete example of the game nim, and show that AlphaZero-style algorithms have difficulty learning these impartial games on larger board sizes. The difference between impartial games and partisan games can be explained by the vulnerability to adversarial attacks and perturbations.

    

    类似AlphaZero的强化学习算法在各种棋盘游戏中表现出色，但在公正游戏中却面临挑战，这些游戏中玩家共享棋子。我们提供了一个具体的游戏例子，即小孩们玩的尼姆游戏，以及其他一些公正游戏，这些游戏似乎成为AlphaZero和类似的强化学习算法的绊脚石。我们的发现与最近的研究一致，表明AlphaZero-style算法容易受到敌对攻击和敌对扰动的影响，显示了在所有合法状态下学习掌握这些游戏的困难。我们发现尼姆游戏在小型棋盘上可以学习，但当棋盘尺寸增大时，AlphaZero-style算法的学习速度显著减慢。直观上，尼姆等公正游戏与象棋和围棋等党派游戏之间的区别在于，如果系统中添加了微小的噪音（例如，棋盘的一小部分被覆盖），对于公正游戏来说，这是一种典型的情况。

    AlphaZero-style reinforcement learning (RL) algorithms excel in various board games but face challenges with impartial games, where players share pieces. We present a concrete example of a game - namely the children's game of nim - and other impartial games that seem to be a stumbling block for AlphaZero-style and similar reinforcement learning algorithms.  Our findings are consistent with recent studies showing that AlphaZero-style algorithms are vulnerable to adversarial attacks and adversarial perturbations, showing the difficulty of learning to master the games in all legal states.  We show that nim can be learned on small boards, but AlphaZero-style algorithms learning dramatically slows down when the board size increases. Intuitively, the difference between impartial games like nim and partisan games like Chess and Go can be explained by the fact that if a tiny amount of noise is added to the system (e.g. if a small part of the board is covered), for impartial games, it is typica
    

