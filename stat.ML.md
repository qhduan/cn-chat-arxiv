# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Test for Generated Hypotheses by Diffusion Models](https://arxiv.org/abs/2402.11789) | 本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。 |
| [^2] | [RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization.](http://arxiv.org/abs/2310.07983) | RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。 |
| [^3] | [MDI+: A Flexible Random Forest-Based Feature Importance Framework.](http://arxiv.org/abs/2307.01932) | MDI+是一种灵活的基于随机森林的特征重要性框架，通过替换线性回归模型和度量，利用正则化的广义线性模型和更适合数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。 |

# 详细

[^1]: 通过扩散模型生成的假设的统计检验

    Statistical Test for Generated Hypotheses by Diffusion Models

    [https://arxiv.org/abs/2402.11789](https://arxiv.org/abs/2402.11789)

    本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。

    

    AI的增强性能加速了其融入科学研究。特别是，利用生成式AI创建科学假设是很有前途的，并且正在越来越多地应用于各个领域。然而，当使用AI生成的假设进行关键决策（如医学诊断）时，验证它们的可靠性至关重要。在本研究中，我们考虑使用扩散模型生成的图像进行医学诊断任务，并提出了一种统计检验来量化其可靠性。所提出的统计检验的基本思想是使用选择性推断框架，我们考虑在生成的图像是由经过训练的扩散模型产生的这一事实条件下的统计检验。利用所提出的方法，医学图像诊断结果的统计可靠性可以以p值的形式量化，从而实现在控制错误率的情况下进行决策。

    arXiv:2402.11789v1 Announce Type: cross  Abstract: The enhanced performance of AI has accelerated its integration into scientific research. In particular, the use of generative AI to create scientific hypotheses is promising and is increasingly being applied across various fields. However, when employing AI-generated hypotheses for critical decisions, such as medical diagnoses, verifying their reliability is crucial. In this study, we consider a medical diagnostic task using generated images by diffusion models, and propose a statistical test to quantify its reliability. The basic idea behind the proposed statistical test is to employ a selective inference framework, where we consider a statistical test conditional on the fact that the generated images are produced by a trained diffusion model. Using the proposed method, the statistical reliability of medical image diagnostic results can be quantified in the form of a p-value, allowing for decision-making with a controlled error rate. 
    
[^2]: RandCom：去中心化随机通信跳跃方法用于分布式随机优化

    RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization. (arXiv:2310.07983v1 [cs.LG])

    [http://arxiv.org/abs/2310.07983](http://arxiv.org/abs/2310.07983)

    RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。

    

    具有随机通信跳过的分布式优化方法因其在加速通信复杂性方面具有的优势而受到越来越多的关注。然而，现有的研究主要集中在强凸确定性设置的集中式通信协议上。在本研究中，我们提出了一种名为RandCom的分布式优化方法，它采用了概率性的本地更新。我们分析了RandCom在随机非凸、凸和强凸设置中的性能，并证明了它能够通过通信概率来渐近地减少通信开销。此外，我们证明当节点数量增加时，RandCom能够实现线性加速。在随机强凸设置中，我们进一步证明了RandCom可以通过独立于网络的步长实现线性加速。此外，我们将RandCom应用于联邦学习，并提供了关于实现线性加速的潜力的积极结果。

    Distributed optimization methods with random communication skips are gaining increasing attention due to their proven benefits in accelerating communication complexity. Nevertheless, existing research mainly focuses on centralized communication protocols for strongly convex deterministic settings. In this work, we provide a decentralized optimization method called RandCom, which incorporates probabilistic local updates. We analyze the performance of RandCom in stochastic non-convex, convex, and strongly convex settings and demonstrate its ability to asymptotically reduce communication overhead by the probability of communication. Additionally, we prove that RandCom achieves linear speedup as the number of nodes increases. In stochastic strongly convex settings, we further prove that RandCom can achieve linear speedup with network-independent stepsizes. Moreover, we apply RandCom to federated learning and provide positive results concerning the potential for achieving linear speedup and
    
[^3]: MDI+:一种灵活的基于随机森林的特征重要性框架

    MDI+: A Flexible Random Forest-Based Feature Importance Framework. (arXiv:2307.01932v1 [stat.ME])

    [http://arxiv.org/abs/2307.01932](http://arxiv.org/abs/2307.01932)

    MDI+是一种灵活的基于随机森林的特征重要性框架，通过替换线性回归模型和度量，利用正则化的广义线性模型和更适合数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。

    

    以不纯度减少的平均值(MDI)是随机森林(RF)中一种流行的特征重要性评估方法。我们展示了在RF中每个树的特征$X_k$的MDI等价于响应变量在决策树集合上的线性回归的未归一化$R^2$值。我们利用这种解释提出了一种灵活的特征重要性框架MDI+，MDI+通过允许分析人员将线性回归模型和$R^2$度量替换为正则化的广义线性模型(GLM)和更适合给定数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。我们进一步提供了关于如何基于可预测性、可计算性和稳定性框架选择适当的GLM和度量的指导，以进行真实数据科学研究。大量基于数据的模拟结果显示，MDI+在性能上显著优于传统的MDI。

    Mean decrease in impurity (MDI) is a popular feature importance measure for random forests (RFs). We show that the MDI for a feature $X_k$ in each tree in an RF is equivalent to the unnormalized $R^2$ value in a linear regression of the response on the collection of decision stumps that split on $X_k$. We use this interpretation to propose a flexible feature importance framework called MDI+. Specifically, MDI+ generalizes MDI by allowing the analyst to replace the linear regression model and $R^2$ metric with regularized generalized linear models (GLMs) and metrics better suited for the given data structure. Moreover, MDI+ incorporates additional features to mitigate known biases of decision trees against additive or smooth models. We further provide guidance on how practitioners can choose an appropriate GLM and metric based upon the Predictability, Computability, Stability framework for veridical data science. Extensive data-inspired simulations show that MDI+ significantly outperfor
    

