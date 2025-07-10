# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Attribution Regularization for Multimodal Paradigms](https://arxiv.org/abs/2404.02359) | 提出一种新的正则化项，鼓励多模态模型有效利用所有模态信息，以解决多模态学习中单模态模型优于多模态模型的问题。 |
| [^2] | [Semantic Augmentation in Images using Language](https://arxiv.org/abs/2404.02353) | 深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。 |
| [^3] | [Proximal Oracles for Optimization and Sampling](https://arxiv.org/abs/2404.02239) | 论文研究了具有非光滑函数和对数凹抽样的凸优化问题，提出了在优化和抽样中应用近端框架的方法，并建立了近端映射的迭代复杂度。 |
| [^4] | [From Blurry to Brilliant Detection: YOLOv5-Based Aerial Object Detection with Super Resolution.](http://arxiv.org/abs/2401.14661) | 基于超分辨率和经过调整的轻量级YOLOv5架构，我们提出了一种创新的方法来解决航空影像中小而密集物体检测的挑战。我们的超分辨率YOLOv5模型采用Transformer编码器块，能够捕捉全局背景和上下文信息，从而在高密度、遮挡条件下提高检测结果。这种轻量级模型不仅准确性更高，而且资源利用效率高，非常适合实时应用。 |
| [^5] | [Provably Efficient Learning in Partially Observable Contextual Bandit.](http://arxiv.org/abs/2308.03572) | 本文研究了在部分可观察情境轮盘赌中的转移学习问题，提出了一种通过优化问题识别行为和奖励因果效应的方法，并利用因果约束来改进轮盘赌算法。 |
| [^6] | [PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series.](http://arxiv.org/abs/2305.18811) | PyPOTS是一个Python工具箱，用于对部分观测的时间序列数据进行数据挖掘和分析，包括插值、分类、聚类和预测等四个任务，算法种类繁多，适用于学术研究和工业应用。 |
| [^7] | [From Pseudorandomness to Multi-Group Fairness and Back.](http://arxiv.org/abs/2301.08837) | 本文探索了预测算法中多组公平性和伪随机性的联系，提供了新的多校准算法和实值函数核引理证明算法。 |

# 详细

[^1]: 多模态范式的归因正则化

    Attribution Regularization for Multimodal Paradigms

    [https://arxiv.org/abs/2404.02359](https://arxiv.org/abs/2404.02359)

    提出一种新的正则化项，鼓励多模态模型有效利用所有模态信息，以解决多模态学习中单模态模型优于多模态模型的问题。

    

    多模态机器学习近年来受到广泛关注，因为它能整合多个模态的信息以增强学习和决策过程。然而，通常观察到单模态模型优于多模态模型，尽管后者可以访问更丰富的信息。此外，单个模态的影响常常主导决策过程，导致性能不佳。这个研究项目旨在通过提出一种新颖的正则化项来解决这些挑战，该项鼓励多模态模型在做出决策时有效利用所有模态的信息。该项目的重点在于视频-音频领域，尽管所提出的正则化技术在涉及多个模态的体现AI研究中具有广泛应用前景。通过利用这种正则化项，提出的方法

    arXiv:2404.02359v1 Announce Type: new  Abstract: Multimodal machine learning has gained significant attention in recent years due to its potential for integrating information from multiple modalities to enhance learning and decision-making processes. However, it is commonly observed that unimodal models outperform multimodal models, despite the latter having access to richer information. Additionally, the influence of a single modality often dominates the decision-making process, resulting in suboptimal performance. This research project aims to address these challenges by proposing a novel regularization term that encourages multimodal models to effectively utilize information from all modalities when making decisions. The focus of this project lies in the video-audio domain, although the proposed regularization technique holds promise for broader applications in embodied AI research, where multiple modalities are involved. By leveraging this regularization term, the proposed approach
    
[^2]: 利用语言在图像中进行语义增强

    Semantic Augmentation in Images using Language

    [https://arxiv.org/abs/2404.02353](https://arxiv.org/abs/2404.02353)

    深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。

    

    深度学习模型需要非常庞大的标记数据集进行监督学习，缺乏这些数据集会导致过拟合并限制其泛化到现实世界示例的能力。最近扩散模型的进展使得能够基于文本输入生成逼真的图像。利用用于训练这些扩散模型的大规模数据集，我们提出一种利用生成的图像来增强现有数据集的技术。本文探讨了各种有效数据增强策略，以提高深度学习模型的跨领域泛化能力。

    arXiv:2404.02353v1 Announce Type: cross  Abstract: Deep Learning models are incredibly data-hungry and require very large labeled datasets for supervised learning. As a consequence, these models often suffer from overfitting, limiting their ability to generalize to real-world examples. Recent advancements in diffusion models have enabled the generation of photorealistic images based on textual inputs. Leveraging the substantial datasets used to train these diffusion models, we propose a technique to utilize generated images to augment existing datasets. This paper explores various strategies for effective data augmentation to improve the out-of-domain generalization capabilities of deep learning models.
    
[^3]: 优化和抽样的近端预言

    Proximal Oracles for Optimization and Sampling

    [https://arxiv.org/abs/2404.02239](https://arxiv.org/abs/2404.02239)

    论文研究了具有非光滑函数和对数凹抽样的凸优化问题，提出了在优化和抽样中应用近端框架的方法，并建立了近端映射的迭代复杂度。

    

    我们考虑具有非光滑目标函数和对数凹抽样（带非光滑潜势，即负对数密度）的凸优化。特别地，我们研究了两种具体设置，其中凸目标/潜势函数要么是半光滑的，要么是复合形式，作为半光滑分量的有限和。为了克服由于非光滑性而带来的挑战，我们的算法在优化和抽样中采用了两种强大的近端框架：优化中的近端点框架和替代抽样框架（ASF），该框架在增广分布上使用Gibbs抽样。优化和抽样算法的一个关键组件是通过正则化切平面方法高效实现近端映射。我们在半光滑和复合设置中建立了近端映射的迭代复杂度。我们进一步提出了一种用于非光滑优化的自适应近端捆绑方法。

    arXiv:2404.02239v1 Announce Type: cross  Abstract: We consider convex optimization with non-smooth objective function and log-concave sampling with non-smooth potential (negative log density). In particular, we study two specific settings where the convex objective/potential function is either semi-smooth or in composite form as the finite sum of semi-smooth components. To overcome the challenges caused by non-smoothness, our algorithms employ two powerful proximal frameworks in optimization and sampling: the proximal point framework for optimization and the alternating sampling framework (ASF) that uses Gibbs sampling on an augmented distribution. A key component of both optimization and sampling algorithms is the efficient implementation of the proximal map by the regularized cutting-plane method. We establish the iteration-complexity of the proximal map in both semi-smooth and composite settings. We further propose an adaptive proximal bundle method for non-smooth optimization. The 
    
[^4]: 从模糊到明亮的检测：基于YOLOv5的超分辨率航空物体检测

    From Blurry to Brilliant Detection: YOLOv5-Based Aerial Object Detection with Super Resolution. (arXiv:2401.14661v1 [cs.CV])

    [http://arxiv.org/abs/2401.14661](http://arxiv.org/abs/2401.14661)

    基于超分辨率和经过调整的轻量级YOLOv5架构，我们提出了一种创新的方法来解决航空影像中小而密集物体检测的挑战。我们的超分辨率YOLOv5模型采用Transformer编码器块，能够捕捉全局背景和上下文信息，从而在高密度、遮挡条件下提高检测结果。这种轻量级模型不仅准确性更高，而且资源利用效率高，非常适合实时应用。

    

    随着无人机和卫星技术的广泛应用，对航空影像中准确物体检测的需求大大增加。传统的物体检测模型在偏向大物体的数据集上训练，对于航空场景中普遍存在的小而密集的物体难以发挥最佳性能。为了解决这个挑战，我们提出了一种创新的方法，结合了超分辨率和经过调整的轻量级YOLOv5架构。我们使用多种数据集进行评估，包括VisDrone-2023、SeaDroneSee、VEDAI和NWPU VHR-10，以验证我们模型的性能。我们的超分辨率YOLOv5架构采用Transformer编码器块，使模型能够捕捉到全局背景和上下文信息，从而提高检测结果，特别是在高密度、遮挡条件下。这种轻量级模型不仅提供了更高的准确性，还确保了资源的有效利用，非常适合实时应用。我们的实验表明，我们的模型在航空物体检测任务中表现出色，特别是在复杂场景中。

    The demand for accurate object detection in aerial imagery has surged with the widespread use of drones and satellite technology. Traditional object detection models, trained on datasets biased towards large objects, struggle to perform optimally in aerial scenarios where small, densely clustered objects are prevalent. To address this challenge, we present an innovative approach that combines super-resolution and an adapted lightweight YOLOv5 architecture. We employ a range of datasets, including VisDrone-2023, SeaDroneSee, VEDAI, and NWPU VHR-10, to evaluate our model's performance. Our Super Resolved YOLOv5 architecture features Transformer encoder blocks, allowing the model to capture global context and context information, leading to improved detection results, especially in high-density, occluded conditions. This lightweight model not only delivers improved accuracy but also ensures efficient resource utilization, making it well-suited for real-time applications. Our experimental 
    
[^5]: 在部分可观察情境轮盘赌中的可证效率学习

    Provably Efficient Learning in Partially Observable Contextual Bandit. (arXiv:2308.03572v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2308.03572](http://arxiv.org/abs/2308.03572)

    本文研究了在部分可观察情境轮盘赌中的转移学习问题，提出了一种通过优化问题识别行为和奖励因果效应的方法，并利用因果约束来改进轮盘赌算法。

    

    本文研究了在部分可观察情境轮盘赌中的转移学习问题，其中代理人仅有来自其他代理人的有限知识，并且对隐藏的混淆因素只有部分信息。我们将该问题转化为通过优化问题来识别或部分识别行为和奖励之间的因果效应。为了解决这些优化问题，我们将未知分布的原始功能约束离散化为线性约束，并通过顺序解线性规划来采样兼容的因果模型，以考虑估计误差得到因果约束。我们的采样算法为适当的采样分布提供了理想的收敛结果。然后，我们展示了如何将因果约束应用于改进经典的轮盘赌算法，并以行动集和函数空间规模为参考改变了遗憾值。值得注意的是，在允许我们处理一般情境分布的函数逼近任务中

    In this paper, we investigate transfer learning in partially observable contextual bandits, where agents have limited knowledge from other agents and partial information about hidden confounders. We first convert the problem to identifying or partially identifying causal effects between actions and rewards through optimization problems. To solve these optimization problems, we discretize the original functional constraints of unknown distributions into linear constraints, and sample compatible causal models via sequentially solving linear programmings to obtain causal bounds with the consideration of estimation error. Our sampling algorithms provide desirable convergence results for suitable sampling distributions. We then show how causal bounds can be applied to improving classical bandit algorithms and affect the regrets with respect to the size of action sets and function spaces. Notably, in the task with function approximation which allows us to handle general context distributions
    
[^6]: PyPOTS：用于部分观测时间序列数据挖掘的Python工具箱

    PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series. (arXiv:2305.18811v1 [cs.LG])

    [http://arxiv.org/abs/2305.18811](http://arxiv.org/abs/2305.18811)

    PyPOTS是一个Python工具箱，用于对部分观测的时间序列数据进行数据挖掘和分析，包括插值、分类、聚类和预测等四个任务，算法种类繁多，适用于学术研究和工业应用。

    

    PyPOTS是一个开源的Python库，致力于在多元部分观测时间序列数据上进行数据挖掘和分析，即针对存在缺失值的不完整时间序列，也称为不规则采样时间序列。特别地，它提供了对四个任务分类的不同算法的易用性支持：插值、分类、聚类和预测。它包含了概率方法和神经网络方法，提供了设计良好、完整文档的编程接口，供学术研究人员和工业专业人员使用。该工具包的设计理念是鲁棒性和可伸缩性，开发过程中遵循了软件构建的最佳实践，例如单元测试、持续集成（CI）和持续交付（CD）、代码覆盖率、可维护性评估、交互式教程和并行化等原则。该工具箱可在Python包索引（PyPI）和Anaconda上使用。

    PyPOTS is an open-source Python library dedicated to data mining and analysis on multivariate partially-observed time series, i.e. incomplete time series with missing values, A.K.A. irregularlysampled time series. Particularly, it provides easy access to diverse algorithms categorized into four tasks: imputation, classification, clustering, and forecasting. The included models contain probabilistic approaches as well as neural-network methods, with a well-designed and fully-documented programming interface for both academic researchers and industrial professionals to use. With robustness and scalability in its design philosophy, best practices of software construction, for example, unit testing, continuous integration (CI) and continuous delivery (CD), code coverage, maintainability evaluation, interactive tutorials, and parallelization, are carried out as principles during the development of PyPOTS. The toolkit is available on both Python Package Index (PyPI) and Anaconda. PyPOTS is o
    
[^7]: 从伪随机性到多组公平性再到回来

    From Pseudorandomness to Multi-Group Fairness and Back. (arXiv:2301.08837v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.08837](http://arxiv.org/abs/2301.08837)

    本文探索了预测算法中多组公平性和伪随机性的联系，提供了新的多校准算法和实值函数核引理证明算法。

    

    本文探讨了预测算法中多组公平性和泄露-韧性和图形规则之间的联系，在一些参数范围内提供了新的多校准和实值函数核引理证明算法。

    We identify and explore connections between the recent literature on multi-group fairness for prediction algorithms and the pseudorandomness notions of leakage-resilience and graph regularity. We frame our investigation using new, statistical distance-based variants of multicalibration that are closely related to the concept of outcome indistinguishability. Adopting this perspective leads us naturally not only to our graph theoretic results, but also to new, more efficient algorithms for multicalibration in certain parameter regimes and a novel proof of a hardcore lemma for real-valued functions.
    

