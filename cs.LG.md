# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey](https://arxiv.org/abs/2402.02242) | 本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。 |
| [^2] | [Graph-Informed Neural Networks for Sparse Grid-Based Discontinuity Detectors.](http://arxiv.org/abs/2401.13652) | 本文提出了一种利用图信息神经网络和稀疏网格来检测不连续函数不连续界面的新方法，该方法在维度大于3的情况下表现出高效且准确的不连续性检测能力，在维度n = 2和n = 4的函数上进行的实验验证了其高效性和泛化能力，并具有可移植性和多功能性。 |
| [^3] | [Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors.](http://arxiv.org/abs/2401.02739) | 本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。 |
| [^4] | [Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization.](http://arxiv.org/abs/2310.00116) | 本文提出了一种基于动态边界最大化和改进的Lipschitz正则化的认证鲁棒性训练算法，通过增加输出空间中的边界和正则化模型的Lipschitz常数来提高深度分类器对抗性扰动的鲁棒性。 |
| [^5] | [Towards Real-World Test-Time Adaptation: Tri-Net Self-Training with Balanced Normalization.](http://arxiv.org/abs/2309.14949) | 本文研究了真实世界测试时间自适应问题，在全局类别不平衡的测试集上补充了现有的协议，并提出了一种平衡归一化层来适应不平衡的测试数据，以解决现有方法的失败。 |
| [^6] | [Synergies Between Federated Learning and O-RAN: Towards an Elastic Virtualized Architecture for Multiple Distributed Machine Learning Services.](http://arxiv.org/abs/2305.02109) | 本文研究了联邦学习在现代无线网络下的挑战，提出了一种方法称为动态多服务联邦学习（DMS-FL）来解决这个问题。同时，还提出了一种名为弹性虚拟化联邦学习（EV-FL）的分布式机器学习架构，来支持DMS-FL中的设计要求。 |
| [^7] | [Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models.](http://arxiv.org/abs/2304.03271) | 本论文揭示以及提出了解决人工智能模型巨大水足迹的方法，因为其淡水消耗已经引起国际社会的重视，并且AI模型应该承担社会责任，做出面对水危机的表率。 |
| [^8] | [Generalizable Memory-driven Transformer for Multivariate Long Sequence Time-series Forecasting.](http://arxiv.org/abs/2207.07827) | 本文提出了一种通用记忆驱动变压器，通过集成多个时间序列特征来驱动预测过程，逐步引入噪声以增强泛化能力，在多个数据集上实现了更优秀的预测性能。 |
| [^9] | [Sinkhorn Distributionally Robust Optimization.](http://arxiv.org/abs/2109.11926) | 本文通过使用Sinkhorn距离进行分布鲁棒优化，推导出更容易处理且在实际中更合理的最坏情况分布，提出了解决方案，并展示了其优越性能。 |

# 详细

[^1]: 面向预训练视觉模型的参数高效微调：一项综述

    Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey

    [https://arxiv.org/abs/2402.02242](https://arxiv.org/abs/2402.02242)

    本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。

    

    大规模预训练的视觉模型（PVMs）展示了在各种下游视觉任务中的适应能力潜力。然而，随着最先进的PVMs达到数十亿甚至数万亿个参数，标准的全面微调范式由于高计算和存储需求变得不可持续。作为响应，研究人员正在探索参数高效微调（PEFT），旨在以最小参数修改超越全面微调的性能。本综述提供了视觉PEFT的全面概述和未来方向，对最新进展进行了系统审查。首先，我们提供了PEFT的正式定义，并讨论了模型预训练方法。然后，我们将现有方法分为三类：基于添加的、基于部分的和基于统一的。最后，我们介绍了常用的数据集和应用，并提出了潜在的未来研究挑战。该综述还提供了丰富的资源收藏。

    Large-scale pre-trained vision models (PVMs) have shown great potential for adaptability across various downstream vision tasks. However, with state-of-the-art PVMs growing to billions or even trillions of parameters, the standard full fine-tuning paradigm is becoming unsustainable due to high computational and storage demands. In response, researchers are exploring parameter-efficient fine-tuning (PEFT), which seeks to exceed the performance of full fine-tuning with minimal parameter modifications. This survey provides a comprehensive overview and future directions for visual PEFT, offering a systematic review of the latest advancements. First, we provide a formal definition of PEFT and discuss model pre-training methods. We then categorize existing methods into three categories: addition-based, partial-based, and unified-based. Finally, we introduce the commonly used datasets and applications and suggest potential future research challenges. A comprehensive collection of resources is
    
[^2]: 基于稀疏网格的不连续性检测的图信息神经网络

    Graph-Informed Neural Networks for Sparse Grid-Based Discontinuity Detectors. (arXiv:2401.13652v1 [cs.LG])

    [http://arxiv.org/abs/2401.13652](http://arxiv.org/abs/2401.13652)

    本文提出了一种利用图信息神经网络和稀疏网格来检测不连续函数不连续界面的新方法，该方法在维度大于3的情况下表现出高效且准确的不连续性检测能力，在维度n = 2和n = 4的函数上进行的实验验证了其高效性和泛化能力，并具有可移植性和多功能性。

    

    本文提出了一种新颖的方法来检测不连续函数的不连续界面。该方法利用了基于图的神经网络（GINNs）和稀疏网格来解决维度大于3的情况下的不连续性检测。训练过的GINNs在稀疏网格上识别有问题的点，并利用构建在网格上的图结构实现高效准确的不连续性检测性能。我们还引入了一种递归算法用于一般的基于稀疏网格的检测器，具有收敛性和易于应用性。在维度n=2和n=4的函数上进行的数值实验证明了GINNs在检测不连续界面方面的高效性和鲁棒泛化能力。值得注意的是，经过训练的GINNs具有可移植性和多功能性，可以集成到各种算法中并共享给用户。

    In this paper, we present a novel approach for detecting the discontinuity interfaces of a discontinuous function. This approach leverages Graph-Informed Neural Networks (GINNs) and sparse grids to address discontinuity detection also in domains of dimension larger than 3. GINNs, trained to identify troubled points on sparse grids, exploit graph structures built on the grids to achieve efficient and accurate discontinuity detection performances. We also introduce a recursive algorithm for general sparse grid-based detectors, characterized by convergence properties and easy applicability. Numerical experiments on functions with dimensions n = 2 and n = 4 demonstrate the efficiency and robust generalization of GINNs in detecting discontinuity interfaces. Notably, the trained GINNs offer portability and versatility, allowing integration into various algorithms and sharing among users.
    
[^3]: 扩散变分推断：扩散模型作为表达性变分后验

    Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors. (arXiv:2401.02739v1 [cs.LG])

    [http://arxiv.org/abs/2401.02739](http://arxiv.org/abs/2401.02739)

    本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。

    

    我们提出了去噪扩散变分推断（DDVI），一种用扩散模型作为表达性变分后验的潜变量模型的近似推断算法。我们的方法通过辅助潜变量增加了变分后验，从而得到一个表达性的模型类，通过反转用户指定的加噪过程在潜空间中进行扩散。我们通过优化一个受到觉醒-睡眠算法启发的边际似然新下界来拟合这些模型。我们的方法易于实现（它适配了正则化的ELBO扩展），与黑盒变分推断兼容，并且表现优于基于归一化流或对抗网络的替代近似后验类别。将我们的方法应用于深度潜变量模型时，我们的方法得到了去噪扩散变分自动编码器（DD-VAE）算法。我们将该算法应用于生物学中的一个激励任务 -- 从人类基因组中推断潜在血统 -- 超过了强基线模型。

    We propose denoising diffusion variational inference (DDVI), an approximate inference algorithm for latent variable models which relies on diffusion models as expressive variational posteriors. Our method augments variational posteriors with auxiliary latents, which yields an expressive class of models that perform diffusion in latent space by reversing a user-specified noising process. We fit these models by optimizing a novel lower bound on the marginal likelihood inspired by the wake-sleep algorithm. Our method is easy to implement (it fits a regularized extension of the ELBO), is compatible with black-box variational inference, and outperforms alternative classes of approximate posteriors based on normalizing flows or adversarial networks. When applied to deep latent variable models, our method yields the denoising diffusion VAE (DD-VAE) algorithm. We use this algorithm on a motivating task in biology -- inferring latent ancestry from human genomes -- outperforming strong baselines
    
[^4]: 动态边界最大化和改进的Lipschitz正则化的认证鲁棒性

    Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization. (arXiv:2310.00116v1 [cs.LG])

    [http://arxiv.org/abs/2310.00116](http://arxiv.org/abs/2310.00116)

    本文提出了一种基于动态边界最大化和改进的Lipschitz正则化的认证鲁棒性训练算法，通过增加输出空间中的边界和正则化模型的Lipschitz常数来提高深度分类器对抗性扰动的鲁棒性。

    

    为了提高深度分类器对抗性扰动的鲁棒性，已经提出了许多方法，例如设计具有更好鲁棒性性质的新架构（例如，Lipschitz-capped网络）或修改训练过程本身（例如，最小-最大优化，约束学习或正则化）。然而，这些方法对于增加输入（特征）空间中的边界可能并不有效。因此，越来越多的人开始对开发能够直接操纵输入空间中的决策边界的训练过程感兴趣。在本文中，我们在该类别的最新发展基础上，开发了一种鲁棒训练算法，其目标是在输出（logit）空间中增加边界，并沿着脆弱方向正则化模型的Lipschitz常数。我们证明这两个目标可以直接促进输入空间中更大的边界。为此，我们开发了一种可扩展的方法来计算...

    To improve the robustness of deep classifiers against adversarial perturbations, many approaches have been proposed, such as designing new architectures with better robustness properties (e.g., Lipschitz-capped networks), or modifying the training process itself (e.g., min-max optimization, constrained learning, or regularization). These approaches, however, might not be effective at increasing the margin in the input (feature) space. As a result, there has been an increasing interest in developing training procedures that can directly manipulate the decision boundary in the input space. In this paper, we build upon recent developments in this category by developing a robust training algorithm whose objective is to increase the margin in the output (logit) space while regularizing the Lipschitz constant of the model along vulnerable directions. We show that these two objectives can directly promote larger margins in the input space. To this end, we develop a scalable method for calcula
    
[^5]: 面向真实世界测试时间自适应：具有平衡归一化的三网络自训练

    Towards Real-World Test-Time Adaptation: Tri-Net Self-Training with Balanced Normalization. (arXiv:2309.14949v1 [cs.LG])

    [http://arxiv.org/abs/2309.14949](http://arxiv.org/abs/2309.14949)

    本文研究了真实世界测试时间自适应问题，在全局类别不平衡的测试集上补充了现有的协议，并提出了一种平衡归一化层来适应不平衡的测试数据，以解决现有方法的失败。

    

    测试时间自适应旨在将源域模型适应到推断阶段的测试数据中，在适应到未见过的破损情况下取得了成功。然而，在更具挑战性的真实世界情境下，这些尝试可能会失败。现有的研究主要考虑非独立同分布的数据流和持续的领域转移下的真实世界测试时间自适应。在这项工作中，我们首先用全局类别不平衡的测试集来补充现有的真实世界TTA协议。我们证明把所有设置结合起来对现有方法提出了新的挑战。我们认为最先失败的现有方法是因为不加选择地将归一化层适应到不平衡的测试数据上所导致的。为了解决这个缺点，我们提出了一个平衡批量归一化层，在推断阶段替换原来的批量归一化。新的批量归一化层能够适应而不偏向多数类别。我们受到自学习（ST）在无标签学习中的成功启发。

    Test-Time Adaptation aims to adapt source domain model to testing data at inference stage with success demonstrated in adapting to unseen corruptions. However, these attempts may fail under more challenging real-world scenarios. Existing works mainly consider real-world test-time adaptation under non-i.i.d. data stream and continual domain shift. In this work, we first complement the existing real-world TTA protocol with a globally class imbalanced testing set. We demonstrate that combining all settings together poses new challenges to existing methods. We argue the failure of state-of-the-art methods is first caused by indiscriminately adapting normalization layers to imbalanced testing data. To remedy this shortcoming, we propose a balanced batchnorm layer to swap out the regular batchnorm at inference stage. The new batchnorm layer is capable of adapting without biasing towards majority classes. We are further inspired by the success of self-training~(ST) in learning from unlabeled 
    
[^6]: 联邦学习与O-RAN的协同：面向多个分布式机器学习服务的弹性虚拟化架构

    Synergies Between Federated Learning and O-RAN: Towards an Elastic Virtualized Architecture for Multiple Distributed Machine Learning Services. (arXiv:2305.02109v1 [cs.NI])

    [http://arxiv.org/abs/2305.02109](http://arxiv.org/abs/2305.02109)

    本文研究了联邦学习在现代无线网络下的挑战，提出了一种方法称为动态多服务联邦学习（DMS-FL）来解决这个问题。同时，还提出了一种名为弹性虚拟化联邦学习（EV-FL）的分布式机器学习架构，来支持DMS-FL中的设计要求。

    

    联邦学习是最流行的分布式机器学习技术，但是在现代无线网络中实现联邦学习面临着许多挑战，主要包括网络条件的动态性、系统中多个联邦学习服务/任务的并存以及联邦学习服务与其他网络服务的并行执行等。针对这些挑战，本文提出了一种名为动态多服务联邦学习（DMS-FL）的联邦学习泛型架构，并通过提出一种新的分布式机器学习架构——弹性虚拟化联邦学习（EV-FL）来解决DMS-FL中的三个未探索的设计问题。

    Federated learning (FL) is the most popular distributed machine learning technique. However, implementation of FL over modern wireless networks faces key challenges caused by (i) dynamics of the network conditions, (ii) coexistence of multiple FL services/tasks in the system, and (iii) concurrent execution of FL services with other network services, which are not jointly considered in prior works. Motivated by these challenges, we introduce a generic FL paradigm over next-generation (NextG) networks, called dynamic multi-service FL (DMS-FL). We identify three unexplored design considerations in DMS-FL: (i) FL service operator accumulation, (ii) wireless resource fragmentation, and (iii) signal strength fluctuations. We take the first steps towards addressing these design considerations through proposing a novel distributed ML architecture called elastic virtualized FL (EV-FL). EV-FL unleashes the full potential of Open RAN (O-RAN) systems and introduces an elastic resource provisioning
    
[^7]: 使AI“口渴”减少的方法：揭示和解决AI模型的秘密水消耗

    Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models. (arXiv:2304.03271v1 [cs.LG])

    [http://arxiv.org/abs/2304.03271](http://arxiv.org/abs/2304.03271)

    本论文揭示以及提出了解决人工智能模型巨大水足迹的方法，因为其淡水消耗已经引起国际社会的重视，并且AI模型应该承担社会责任，做出面对水危机的表率。

    

    人工智能（AI）模型的碳足迹不断增长，特别是像GPT-3和GPT-4这样的大型模型，已经受到公众的关注。然而，同等重要且巨大的AI模型水印尚未引起人们的注意。例如，在微软最先进的美国数据中心中训练GPT-3可以直接消耗70万升清洁淡水（相当于生产370辆宝马汽车或320辆特斯拉电动汽车），如果在微软的亚洲数据中心进行训练，这个水消耗量将增加三倍，但这样的信息一直被保密。这极其令人担忧，因为淡水短缺已成为在人口迅速增长、水资源减少和老化的水基础设施的背景下，我们所有人面临的最紧迫的挑战之一。为了应对全球水资源的挑战，人工智能模型可以，而且应该，承担社会责任，以身作则解决自己的问题。

    The growing carbon footprint of artificial intelligence (AI) models, especially large ones such as GPT-3 and GPT-4, has been undergoing public scrutiny. Unfortunately, however, the equally important and enormous water footprint of AI models has remained under the radar. For example, training GPT-3 in Microsoft's state-of-the-art U.S. data centers can directly consume 700,000 liters of clean freshwater (enough for producing 370 BMW cars or 320 Tesla electric vehicles) and the water consumption would have been tripled if training were done in Microsoft's Asian data centers, but such information has been kept as a secret. This is extremely concerning, as freshwater scarcity has become one of the most pressing challenges shared by all of us in the wake of the rapidly growing population, depleting water resources, and aging water infrastructures. To respond to the global water challenges, AI models can, and also should, take social responsibility and lead by example by addressing their own 
    
[^8]: 多元长序列时间序列预测的通用记忆驱动变压器

    Generalizable Memory-driven Transformer for Multivariate Long Sequence Time-series Forecasting. (arXiv:2207.07827v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.07827](http://arxiv.org/abs/2207.07827)

    本文提出了一种通用记忆驱动变压器，通过集成多个时间序列特征来驱动预测过程，逐步引入噪声以增强泛化能力，在多个数据集上实现了更优秀的预测性能。

    

    多元长序列时间序列预测(M-LSTF)是一个实际但具有挑战性的问题。与传统的时间序列预测任务不同，M-LSTF任务从两个方面更具挑战性：1) M-LSTF模型需要同时学习多个时间特征之间的时间序列模式；2)在滚动预测设置中，两个连续训练样本之间的相似度随着预测长度的增加而增加，这使得模型更易于过拟合。本文提出了一种通用记忆驱动变压器来解决M-LSTF问题。具体而言，我们首先提出了一个全局层面的记忆组件，通过集成多个时间序列特征来驱动预测过程。此外，我们采用渐进式的方式来训练我们的模型，以增强其泛化能力，逐步在训练样本中引入伯努利噪声。在多个领域的五个不同数据集上进行了大量实验。实验结果表明，我们提出的模型优于现有的方法，并在所有数据集上实现了更优异的预测性能。

    Multivariate long sequence time-series forecasting (M-LSTF) is a practical but challenging problem. Unlike traditional timer-series forecasting tasks, M-LSTF tasks are more challenging from two aspects: 1) M-LSTF models need to learn time-series patterns both within and between multiple time features; 2) Under the rolling forecasting setting, the similarity between two consecutive training samples increases with the increasing prediction length, which makes models more prone to overfitting. In this paper, we propose a generalizable memory-driven Transformer to target M-LSTF problems. Specifically, we first propose a global-level memory component to drive the forecasting procedure by integrating multiple time-series features. In addition, we adopt a progressive fashion to train our model to increase its generalizability, in which we gradually introduce Bernoulli noises to training samples. Extensive experiments have been performed on five different datasets across multiple fields. Exper
    
[^9]: Sinkhorn分布鲁棒优化

    Sinkhorn Distributionally Robust Optimization. (arXiv:2109.11926v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2109.11926](http://arxiv.org/abs/2109.11926)

    本文通过使用Sinkhorn距离进行分布鲁棒优化，推导出更容易处理且在实际中更合理的最坏情况分布，提出了解决方案，并展示了其优越性能。

    

    我们研究了使用Sinkhorn距离 -一种基于熵正则化的Wasserstein距离变体- 的分布鲁棒优化（DRO）。我们为一般名义分布推导了凸规划对偶重构。相比于Wasserstein DRO，对于更大类的损失函数，它在计算上更容易处理，它的最坏情况分布对实际应用更合理。为了解决对偶重构，我们开发了一种使用有偏梯度神经元的随机镜像下降算法，并分析了其收敛速度。最后，我们提供了使用合成和真实数据的数值实例，以证明其优越性能。

    We study distributionally robust optimization (DRO) with Sinkhorn distance -a variant of Wasserstein distance based on entropic regularization. We derive convex programming dual reformulation for a general nominal distribution. Compared with Wasserstein DRO, it is computationally tractable for a larger class of loss functions, and its worst-case distribution is more reasonable for practical applications. To solve the dual reformulation, we develop a stochastic mirror descent algorithm using biased gradient oracles and analyze its convergence rate. Finally, we provide numerical examples using synthetic and real data to demonstrate its superior performance.
    

