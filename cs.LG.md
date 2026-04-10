# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Privacy Funnel Model: From a Discriminative to a Generative Approach with an Application to Face Recognition](https://arxiv.org/abs/2404.02696) | 该研究将信息论隐私原则与表示学习相结合，提出了一种新的隐私保护表示学习方法，适用于人脸识别系统，并引入了生成式隐私漏斗模型。 |
| [^2] | [CODA: A COst-efficient Test-time Domain Adaptation Mechanism for HAR](https://arxiv.org/abs/2403.14922) | CODA 提出了一种节约成本的测试时域自适应机制，通过主动学习理论处理实时漂移，实现在设备上直接进行成本有效的自适应，并保留数据分布中的有意义结构。 |
| [^3] | [Time series generation for option pricing on quantum computers using tensor network](https://arxiv.org/abs/2402.17148) | 提出了一种使用矩阵乘积态作为时间序列生成的方法，可以有效生成多个时间点处基础资产价格的联合分布的态，并证实了该方法在Heston模型中的可行性。 |
| [^4] | [Reconstructing the Geometry of Random Geometric Graphs](https://arxiv.org/abs/2402.09591) | 该论文通过在底层空间中采样的图来有效地重构随机几何图的几何形状。该方法基于流形假设，即底层空间是低维流形，并且连接概率是嵌入在$\mathbb{R}^N$中的流形中点之间欧几里德距离的严格递减函数。 |
| [^5] | [A survey of Generative AI Applications.](http://arxiv.org/abs/2306.02781) | 本篇论文对350多个生成式人工智能应用进行了全面调查，总结了不同单模和多模生成式人工智能的应用。该调查为研究人员和从业者提供了宝贵的资源，帮助他们更好地了解生成式人工智能领域目前的最先进技术，并促进该领域的进一步创新。 |

# 详细

[^1]: 深度隐私漏斗模型：从判别式方法到生成式方法的转变，并其在人脸识别中的应用

    Deep Privacy Funnel Model: From a Discriminative to a Generative Approach with an Application to Face Recognition

    [https://arxiv.org/abs/2404.02696](https://arxiv.org/abs/2404.02696)

    该研究将信息论隐私原则与表示学习相结合，提出了一种新的隐私保护表示学习方法，适用于人脸识别系统，并引入了生成式隐私漏斗模型。

    

    在这项研究中，我们将信息论隐私漏斗（PF）模型应用于人脸识别领域，开发了一种新的隐私保护表示学习方法，以端到端训练框架来实现。我们的方法解决了数据保护中模糊化和效用之间的权衡，通过对数损失（也称为自信息损失）来量化。这项研究详细探讨了信息论隐私原则与表示学习的整合，在特定关注于人脸识别系统。我们特别强调了我们的框架与人脸识别网络的最新进展（如AdaFace和ArcFace）之间的适应性。此外，我们还介绍了生成式隐私漏斗（GenPF）模型，这是一种超出传统PF模型范围的范例，被称为判别式隐私漏斗（DisPF）。

    arXiv:2404.02696v1 Announce Type: new  Abstract: In this study, we apply the information-theoretic Privacy Funnel (PF) model to the domain of face recognition, developing a novel method for privacy-preserving representation learning within an end-to-end training framework. Our approach addresses the trade-off between obfuscation and utility in data protection, quantified through logarithmic loss, also known as self-information loss. This research provides a foundational exploration into the integration of information-theoretic privacy principles with representation learning, focusing specifically on the face recognition systems. We particularly highlight the adaptability of our framework with recent advancements in face recognition networks, such as AdaFace and ArcFace. In addition, we introduce the Generative Privacy Funnel ($\mathsf{GenPF}$) model, a paradigm that extends beyond the traditional scope of the PF model, referred to as the Discriminative Privacy Funnel ($\mathsf{DisPF}$)
    
[^2]: CODA：一种用于HAR的节约成本的测试时域自适应机制

    CODA: A COst-efficient Test-time Domain Adaptation Mechanism for HAR

    [https://arxiv.org/abs/2403.14922](https://arxiv.org/abs/2403.14922)

    CODA 提出了一种节约成本的测试时域自适应机制，通过主动学习理论处理实时漂移，实现在设备上直接进行成本有效的自适应，并保留数据分布中的有意义结构。

    

    近年来，移动感知的新兴研究导致了增强人类日常生活的新型场景，但是动态的使用条件经常导致系统在实际环境中部署时性能下降。现有的解决方案通常采用基于神经网络的一次性自适应方案，这些方案往往难以确保针对人类感知场景中不确定漂移条件的稳健性。本文提出了CODA，一种针对移动感知的节约成本域自适应机制，从数据分布角度利用主动学习理论解决实时漂移，以确保在设备上直接进行成本有效的自适应。通过结合聚类损失和重要性加权主动学习算法，CODA在成本效益的实例级更新过程中保留不同聚类之间的关系，保留数据分布中的有意义结构。

    arXiv:2403.14922v1 Announce Type: new  Abstract: In recent years, emerging research on mobile sensing has led to novel scenarios that enhance daily life for humans, but dynamic usage conditions often result in performance degradation when systems are deployed in real-world settings. Existing solutions typically employ one-off adaptation schemes based on neural networks, which struggle to ensure robustness against uncertain drifting conditions in human-centric sensing scenarios. In this paper, we propose CODA, a COst-efficient Domain Adaptation mechanism for mobile sensing that addresses real-time drifts from the data distribution perspective with active learning theory, ensuring cost-efficient adaptation directly on the device. By incorporating a clustering loss and importance-weighted active learning algorithm, CODA retains the relationship between different clusters during cost-effective instance-level updates, preserving meaningful structure within the data distribution. We also sho
    
[^3]: 使用张量网络在量子计算机上生成期权定价的时间序列

    Time series generation for option pricing on quantum computers using tensor network

    [https://arxiv.org/abs/2402.17148](https://arxiv.org/abs/2402.17148)

    提出了一种使用矩阵乘积态作为时间序列生成的方法，可以有效生成多个时间点处基础资产价格的联合分布的态，并证实了该方法在Heston模型中的可行性。

    

    金融，特别是期权定价，是一个有望从量子计算中受益的行业。尽管已经提出了用于期权定价的量子算法，但人们希望在算法中设计出更高效的实现方式，其中之一是准备编码基础资产价格概率分布的量子态。特别是在定价依赖路径的期权时，我们需要生成一个编码多个时间点处基础资产价格的联合分布的态，这更具挑战性。为解决这些问题，我们提出了一种使用矩阵乘积态（MPS）作为时间序列生成的生成模型的新方法。为了验证我们的方法，以Heston模型为目标，我们进行数值实验以在模型中生成时间序列。我们的研究结果表明MPS模型能够生成Heston模型中的路径，突显了...

    arXiv:2402.17148v1 Announce Type: cross  Abstract: Finance, especially option pricing, is a promising industrial field that might benefit from quantum computing. While quantum algorithms for option pricing have been proposed, it is desired to devise more efficient implementations of costly operations in the algorithms, one of which is preparing a quantum state that encodes a probability distribution of the underlying asset price. In particular, in pricing a path-dependent option, we need to generate a state encoding a joint distribution of the underlying asset price at multiple time points, which is more demanding. To address these issues, we propose a novel approach using Matrix Product State (MPS) as a generative model for time series generation. To validate our approach, taking the Heston model as a target, we conduct numerical experiments to generate time series in the model. Our findings demonstrate the capability of the MPS model to generate paths in the Heston model, highlightin
    
[^4]: 重构随机几何图的几何形状

    Reconstructing the Geometry of Random Geometric Graphs

    [https://arxiv.org/abs/2402.09591](https://arxiv.org/abs/2402.09591)

    该论文通过在底层空间中采样的图来有效地重构随机几何图的几何形状。该方法基于流形假设，即底层空间是低维流形，并且连接概率是嵌入在$\mathbb{R}^N$中的流形中点之间欧几里德距离的严格递减函数。

    

    随机几何图是在度量空间上定义的随机图模型。该模型首先从度量空间中采样点，然后以依赖于它们之间距离的概率独立地连接每对采样点。在本工作中，我们展示了如何在流形假设下有效地从采样的图中重构底层空间的几何形状，即假设底层空间是低维流形，并且连接概率是嵌入在$\mathbb{R}^N$中的流形中点之间欧几里德距离的严格递减函数。我们的工作补充了大量关于流形学习的工作，其目标是从在流形中采样的点及其（近似的）距离中恢复出流形。

    arXiv:2402.09591v1 Announce Type: new  Abstract: Random geometric graphs are random graph models defined on metric spaces. Such a model is defined by first sampling points from a metric space and then connecting each pair of sampled points with probability that depends on their distance, independently among pairs. In this work, we show how to efficiently reconstruct the geometry of the underlying space from the sampled graph under the manifold assumption, i.e., assuming that the underlying space is a low dimensional manifold and that the connection probability is a strictly decreasing function of the Euclidean distance between the points in a given embedding of the manifold in $\mathbb{R}^N$. Our work complements a large body of work on manifold learning, where the goal is to recover a manifold from sampled points sampled in the manifold along with their (approximate) distances.
    
[^5]: 生成式人工智能应用调查

    A survey of Generative AI Applications. (arXiv:2306.02781v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.02781](http://arxiv.org/abs/2306.02781)

    本篇论文对350多个生成式人工智能应用进行了全面调查，总结了不同单模和多模生成式人工智能的应用。该调查为研究人员和从业者提供了宝贵的资源，帮助他们更好地了解生成式人工智能领域目前的最先进技术，并促进该领域的进一步创新。

    

    近年来，生成式人工智能有了显著增长，并在各个领域展示了广泛的应用。本文对350多个生成式人工智能应用进行了全面调查，提供了分类结构和对不同单模和多模生成式人工智能的简洁描述。该调查分成多个部分，覆盖了文本、图像、视频、游戏和脑信息等单模生成式人工智能的广泛应用。我们的调研旨在为研究人员和从业者提供宝贵的资源，帮助他们更好地了解生成式人工智能领域目前的最先进技术，并促进该领域的进一步创新。

    Generative AI has experienced remarkable growth in recent years, leading to a wide array of applications across diverse domains. In this paper, we present a comprehensive survey of more than 350 generative AI applications, providing a structured taxonomy and concise descriptions of various unimodal and even multimodal generative AIs. The survey is organized into sections, covering a wide range of unimodal generative AI applications such as text, images, video, gaming and brain information. Our survey aims to serve as a valuable resource for researchers and practitioners to navigate the rapidly expanding landscape of generative AI, facilitating a better understanding of the current state-of-the-art and fostering further innovation in the field.
    

