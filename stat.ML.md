# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Kalman Filters Can Filter.](http://arxiv.org/abs/2310.19603) | 本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。 |
| [^2] | [Boosting for Bounding the Worst-class Error.](http://arxiv.org/abs/2310.14890) | 该论文提出了一种基于Boosting的算法，可以保证最差类别训练误差的上界，并降低了最差类别的测试误差率。 |
| [^3] | [Don't be so negative! Score-based Generative Modeling with Oracle-assisted Guidance.](http://arxiv.org/abs/2307.16463) | 本文提出了一种基于得分的生成建模方法Gen-neG，它利用额外的辅助信息来指导生成过程。通过引导生成过程朝着正支持区域生成样本，该方法在自动驾驶模拟器中的避碰应用和安全防护人体动作生成中展现了实用性。 |

# 详细

[^1]: 深度卡尔曼滤波器可以进行滤波

    Deep Kalman Filters Can Filter. (arXiv:2310.19603v1 [cs.LG])

    [http://arxiv.org/abs/2310.19603](http://arxiv.org/abs/2310.19603)

    本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。

    

    深度卡尔曼滤波器（DKFs）是一类神经网络模型，可以从序列数据中生成高斯概率测度。虽然DKFs受卡尔曼滤波器的启发，但它们缺乏与随机滤波问题的具体理论关联，从而限制了它们在传统模型基础上的滤波问题的应用，例如数学金融中的债券和期权定价模型校准。我们通过展示一类连续时间DKFs，可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而解决了深度学习数学基础中的这个问题。我们的近似结果在路径的足够规则的紧致子集上一致成立，其中近似误差由在给定紧致路径集上均一地计算的最坏情况2-Wasserstein距离量化。

    Deep Kalman filters (DKFs) are a class of neural network models that generate Gaussian probability measures from sequential data. Though DKFs are inspired by the Kalman filter, they lack concrete theoretical ties to the stochastic filtering problem, thus limiting their applicability to areas where traditional model-based filters have been used, e.g.\ model calibration for bond and option prices in mathematical finance. We address this issue in the mathematical foundations of deep learning by exhibiting a class of continuous-time DKFs which can approximately implement the conditional law of a broad class of non-Markovian and conditionally Gaussian signal processes given noisy continuous-times measurements. Our approximation results hold uniformly over sufficiently regular compact subsets of paths, where the approximation error is quantified by the worst-case 2-Wasserstein distance computed uniformly over the given compact set of paths.
    
[^2]: Boosting用于界定最差分类误差

    Boosting for Bounding the Worst-class Error. (arXiv:2310.14890v1 [stat.ML])

    [http://arxiv.org/abs/2310.14890](http://arxiv.org/abs/2310.14890)

    该论文提出了一种基于Boosting的算法，可以保证最差类别训练误差的上界，并降低了最差类别的测试误差率。

    

    本文解决了最差类别误差率的问题，而不是针对所有类别的标准误差率的平均。例如，一个三类别分类任务，其中各类别的误差率分别为10％，10％和40％，其最差类别误差率为40％，而在类别平衡条件下的平均误差率为20％。最差类别错误在许多应用中很重要。例如，在医学图像分类任务中，对于恶性肿瘤类别具有40％的错误率而良性和健康类别具有10％的错误率是不能被接受的。我们提出了一种保证最差类别训练误差上界的提升算法，并推导出其泛化界。实验结果表明，该算法降低了最差类别的测试误差率，同时避免了对训练集的过拟合。

    This paper tackles the problem of the worst-class error rate, instead of the standard error rate averaged over all classes. For example, a three-class classification task with class-wise error rates of 10\%, 10\%, and 40\% has a worst-class error rate of 40\%, whereas the average is 20\% under the class-balanced condition. The worst-class error is important in many applications. For example, in a medical image classification task, it would not be acceptable for the malignant tumor class to have a 40\% error rate, while the benign and healthy classes have 10\% error rates.We propose a boosting algorithm that guarantees an upper bound of the worst-class training error and derive its generalization bound. Experimental results show that the algorithm lowers worst-class test error rates while avoiding overfitting to the training set.
    
[^3]: 不要那么消极！带有Oracle辅助指导的基于得分的生成建模方法

    Don't be so negative! Score-based Generative Modeling with Oracle-assisted Guidance. (arXiv:2307.16463v1 [cs.LG])

    [http://arxiv.org/abs/2307.16463](http://arxiv.org/abs/2307.16463)

    本文提出了一种基于得分的生成建模方法Gen-neG，它利用额外的辅助信息来指导生成过程。通过引导生成过程朝着正支持区域生成样本，该方法在自动驾驶模拟器中的避碰应用和安全防护人体动作生成中展现了实用性。

    

    最大似然原则提倡通过优化数据似然函数进行参数估计。以这种方式估计的模型可以展现出各种由架构、参数化和优化偏差等因素决定的泛化特性。本文解决了在存在额外辅助信息的情况下的模型学习问题，该辅助信息以Oracle的形式存在，可以标记样本是否处于真实数据生成分布的支持范围之外。具体而言，我们开发了一种新的去噪扩散概率建模（DDPM）方法，称为Gen-neG，它利用了这个额外的辅助信息。我们的方法基于生成对抗网络（GANs）和扩散模型中的鉴别器指导，以引导生成过程朝着Oracle所指示的正支持区域生成样本。我们通过在自动驾驶模拟器中的避碰应用和安全防护人体动作生成中的实证验证了Gen-neG的实用性。

    The maximum likelihood principle advocates parameter estimation via optimization of the data likelihood function. Models estimated in this way can exhibit a variety of generalization characteristics dictated by, e.g. architecture, parameterization, and optimization bias. This work addresses model learning in a setting where there further exists side-information in the form of an oracle that can label samples as being outside the support of the true data generating distribution. Specifically we develop a new denoising diffusion probabilistic modeling (DDPM) methodology, Gen-neG, that leverages this additional side-information. Our approach builds on generative adversarial networks (GANs) and discriminator guidance in diffusion models to guide the generation process towards the positive support region indicated by the oracle. We empirically establish the utility of Gen-neG in applications including collision avoidance in self-driving simulators and safety-guarded human motion generation.
    

