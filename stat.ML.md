# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Partial Rankings of Optimizers](https://arxiv.org/abs/2402.16565) | 该论文介绍了一种基于多个标准进行优化器基准测试的框架，通过利用次序信息并允许不可比性，避免了聚合的缺点，可以识别产生中心或离群排序的测试函数，并评估基准测试套件的质量。 |
| [^2] | [Invariant kernels on Riemannian symmetric spaces: a harmonic-analytic approach.](http://arxiv.org/abs/2310.19270) | 本文证明了在非欧几里德对称空间上定义的经典高斯核在任意参数选择下都不是正定的，通过发展新的几何和分析论证，并且给出了正定性的严格刻画以及L$^{\!\scriptscriptstyle p}$-$\hspace{0.02cm}$Godement定理的必要和充分条件。 |
| [^3] | [Seismic Data Interpolation based on Denoising Diffusion Implicit Models with Resampling.](http://arxiv.org/abs/2307.04226) | 本研究提出了一种基于去噪扩散隐式模型和重采样的地震数据插值方法，通过使用多头自注意力和余弦噪声计划，实现了稳定训练生成对抗网络，并提高了已知迹线信息的利用率。 |
| [^4] | [Probabilistic matching of real and generated data statistics in generative adversarial networks.](http://arxiv.org/abs/2306.10943) | 本文提出一种通过向生成器损失函数中添加KL散度项的方法，来保证生成数据统计分布与真实数据的相应分布重合，并在实验中展示了此方法的优越性能。 |
| [^5] | [A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System.](http://arxiv.org/abs/2211.03933) | 该论文提出了一种基于超图的机器学习集成网络入侵检测系统，使用超图捕捉端口扫描攻击的演化模式，并使用派生的度量来训练NIDS，从而允许在高精度、高准确率、高召回率性能下实时监测和检测端口扫描活动、其他类型的攻击和敌对入侵，解决了传统NIDS面临的挑战。 |

# 详细

[^1]: 优化器的部分排序

    Partial Rankings of Optimizers

    [https://arxiv.org/abs/2402.16565](https://arxiv.org/abs/2402.16565)

    该论文介绍了一种基于多个标准进行优化器基准测试的框架，通过利用次序信息并允许不可比性，避免了聚合的缺点，可以识别产生中心或离群排序的测试函数，并评估基准测试套件的质量。

    

    我们提出了一个根据多个标准在各种测试函数上对优化器进行基准测试的框架。基于最近引入的用于偏序/排序的无集合泛函深度函数，它充分利用了次序信息并允许不可比性。我们的方法描述了所有部分顺序/排序的分布，避免了聚合的臭名昭著的缺点。这允许识别产生优化器的中心或离群排序的测试函数，并评估基准测试套件的质量。

    arXiv:2402.16565v1 Announce Type: cross  Abstract: We introduce a framework for benchmarking optimizers according to multiple criteria over various test functions. Based on a recently introduced union-free generic depth function for partial orders/rankings, it fully exploits the ordinal information and allows for incomparability. Our method describes the distribution of all partial orders/rankings, avoiding the notorious shortcomings of aggregation. This permits to identify test functions that produce central or outlying rankings of optimizers and to assess the quality of benchmarking suites.
    
[^2]: 在黎曼对称空间上的不变核：一种谐波分析方法

    Invariant kernels on Riemannian symmetric spaces: a harmonic-analytic approach. (arXiv:2310.19270v1 [cs.LG])

    [http://arxiv.org/abs/2310.19270](http://arxiv.org/abs/2310.19270)

    本文证明了在非欧几里德对称空间上定义的经典高斯核在任意参数选择下都不是正定的，通过发展新的几何和分析论证，并且给出了正定性的严格刻画以及L$^{\!\scriptscriptstyle p}$-$\hspace{0.02cm}$Godement定理的必要和充分条件。

    

    本文旨在证明经典的高斯核，在非欧几里德对称空间上定义时，对于任意参数选择都不是正定的。为了实现这一目标，本文发展了新的几何和分析论证。这些论证提供了高斯核正定性的严格刻画，但仅限于在低维中通过数值计算处理的有限情况。其中最重要的结果是L$^{\!\scriptscriptstyle p}$-$\hspace{0.02cm}$Godement定理（其中$p = 1,2$），它提供了定义在非紧型对称空间上的核是正定的可验证的必要和充分条件。一种著名的定理，有时被称为Bochner-Godement定理，已经给出了这样的条件，并且在适用范围上更加广泛，但应用起来尤为困难。除了与高斯核的关联外，在本文中的新结果为s提供了一个蓝图。

    This work aims to prove that the classical Gaussian kernel, when defined on a non-Euclidean symmetric space, is never positive-definite for any choice of parameter. To achieve this goal, the paper develops new geometric and analytical arguments. These provide a rigorous characterization of the positive-definiteness of the Gaussian kernel, which is complete but for a limited number of scenarios in low dimensions that are treated by numerical computations. Chief among these results are the L$^{\!\scriptscriptstyle p}$-$\hspace{0.02cm}$Godement theorems (where $p = 1,2$), which provide verifiable necessary and sufficient conditions for a kernel defined on a symmetric space of non-compact type to be positive-definite. A celebrated theorem, sometimes called the Bochner-Godement theorem, already gives such conditions and is far more general in its scope, but is especially hard to apply. Beyond the connection with the Gaussian kernel, the new results in this work lay out a blueprint for the s
    
[^3]: 基于去噪扩散隐式模型和重采样的地震数据插值

    Seismic Data Interpolation based on Denoising Diffusion Implicit Models with Resampling. (arXiv:2307.04226v1 [physics.geo-ph])

    [http://arxiv.org/abs/2307.04226](http://arxiv.org/abs/2307.04226)

    本研究提出了一种基于去噪扩散隐式模型和重采样的地震数据插值方法，通过使用多头自注意力和余弦噪声计划，实现了稳定训练生成对抗网络，并提高了已知迹线信息的利用率。

    

    地震数据空间扩展上缺失剖面导致地震数据不完整是地震采集中普遍存在的问题，由于障碍物和经济限制，这严重影响了地下地质结构的成像质量。最近，基于深度学习的地震插值方法取得了令人期待的进展，但稳定训练生成对抗网络并不容易，如果测试和训练中的缺失模式不匹配，性能退化通常是显著的。在本文中，我们提出了一种新的地震去噪扩散隐式模型和重采样方法。模型训练建立在去噪扩散概率模型的基础上，其中U-Net配备了多头自注意力以匹配每个步骤中的噪声。余弦噪声计划作为全局噪声配置，通过加速过度信息的传递来促进已知迹线信息的高度利用。

    The incompleteness of the seismic data caused by missing traces along the spatial extension is a common issue in seismic acquisition due to the existence of obstacles and economic constraints, which severely impairs the imaging quality of subsurface geological structures. Recently, deep learning-based seismic interpolation methods have attained promising progress, while achieving stable training of generative adversarial networks is not easy, and performance degradation is usually notable if the missing patterns in the testing and training do not match. In this paper, we propose a novel seismic denoising diffusion implicit model with resampling. The model training is established on the denoising diffusion probabilistic model, where U-Net is equipped with the multi-head self-attention to match the noise in each step. The cosine noise schedule, serving as the global noise configuration, promotes the high utilization of known trace information by accelerating the passage of the excessive 
    
[^4]: 生成对抗网络中真实数据和生成数据统计的概率匹配

    Probabilistic matching of real and generated data statistics in generative adversarial networks. (arXiv:2306.10943v1 [stat.ML])

    [http://arxiv.org/abs/2306.10943](http://arxiv.org/abs/2306.10943)

    本文提出一种通过向生成器损失函数中添加KL散度项的方法，来保证生成数据统计分布与真实数据的相应分布重合，并在实验中展示了此方法的优越性能。

    

    生成对抗网络是一种强大的生成建模方法。虽然生成样本往往难以区分真实数据，但不能保证它们遵循真实数据分布。本文提出了一种方法，确保某些生成数据统计分布与真实数据的相应分布重合。为此，我们在生成器损失函数中添加了Kullback-Leibler项：KL散度是在每次迭代中从小批量值获得的相应生成分布和由条件能量模型表示的真实分布之间的差异。我们在一个合成数据集和两个实际数据集上评估了该方法，并展示了我们方法的优越性能。

    Generative adversarial networks constitute a powerful approach to generative modeling. While generated samples often are indistinguishable from real data, there is no guarantee that they will follow the true data distribution. In this work, we propose a method to ensure that the distributions of certain generated data statistics coincide with the respective distributions of the real data. In order to achieve this, we add a Kullback-Leibler term to the generator loss function: the KL divergence is taken between the true distributions as represented by a conditional energy-based model, and the corresponding generated distributions obtained from minibatch values at each iteration. We evaluate the method on a synthetic dataset and two real-world datasets and demonstrate improved performance of our method.
    
[^5]: 基于超图的机器学习集成网络入侵检测系统

    A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System. (arXiv:2211.03933v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2211.03933](http://arxiv.org/abs/2211.03933)

    该论文提出了一种基于超图的机器学习集成网络入侵检测系统，使用超图捕捉端口扫描攻击的演化模式，并使用派生的度量来训练NIDS，从而允许在高精度、高准确率、高召回率性能下实时监测和检测端口扫描活动、其他类型的攻击和敌对入侵，解决了传统NIDS面临的挑战。

    

    网络入侵检测系统(NIDS)在检测恶意攻击时仍然面临挑战。NIDS通常在离线状态下开发，但面对自动生成的端口扫描渗透尝试时，会导致从对手适应到NIDS响应的显着时间滞后。为了解决这些问题，我们使用以Internet协议地址和目标端口为重点的超图来捕捉端口扫描攻击的演化模式。然后使用派生的基于超图的度量来训练一个集成机器学习(ML)的NIDS，从而允许在高精度、高准确率、高召回率性能下实时调整，监测和检测端口扫描活动、其他类型的攻击和敌对入侵。这个ML自适应的NIDS是通过以下几个部分的组合开发出来的：(1)入侵示例，(2)NIDS更新规则，(3)触发NIDS重新训练请求的攻击阈值选择，以及(4)在没有先前网络性质知识的情况下的生产环境。

    Network intrusion detection systems (NIDS) to detect malicious attacks continue to meet challenges. NIDS are often developed offline while they face auto-generated port scan infiltration attempts, resulting in a significant time lag from adversarial adaption to NIDS response. To address these challenges, we use hypergraphs focused on internet protocol addresses and destination ports to capture evolving patterns of port scan attacks. The derived set of hypergraph-based metrics are then used to train an ensemble machine learning (ML) based NIDS that allows for real-time adaption in monitoring and detecting port scanning activities, other types of attacks, and adversarial intrusions at high accuracy, precision and recall performances. This ML adapting NIDS was developed through the combination of (1) intrusion examples, (2) NIDS update rules, (3) attack threshold choices to trigger NIDS retraining requests, and (4) a production environment with no prior knowledge of the nature of network 
    

