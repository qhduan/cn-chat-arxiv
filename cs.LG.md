# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator](https://arxiv.org/abs/2402.17767) | 实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。 |
| [^2] | [TransAxx: Efficient Transformers with Approximate Computing](https://arxiv.org/abs/2402.07545) | TransAxx是一个基于PyTorch库的框架，可以支持近似计算，并通过对Vision Transformer模型进行近似感知微调，来提高在低功耗设备上的计算效率。 |
| [^3] | [Navigating Neural Space: Revisiting Concept Activation Vectors to Overcome Directional Divergence](https://arxiv.org/abs/2202.03482) | 本文重新审视了概念激活向量（CAVs）在建模人类可理解的概念中的应用，并引入了基于模式的CAVs来提供更准确的概念方向。 |
| [^4] | [Mori-Zwanzig latent space Koopman closure for nonlinear autoencoder.](http://arxiv.org/abs/2310.10745) | 本研究提出了一种名为Mori-Zwanzig自编码器（MZ-AE）的新方法，用于在低维空间中稳健地逼近Koopman算子，通过非线性自编码器和Mori-Zwanzig形式主义的集成实现对有限不变Koopman子空间的逼近，从而增强了精确性和准确预测复杂系统行为的能力。 |
| [^5] | [Diverse Neural Audio Embeddings -- Bringing Features back !.](http://arxiv.org/abs/2309.08751) | 本文通过在音频分类任务中学习多样化的特征表示，包括领域特定的音高、音色和神经表示，以及端到端架构，为学习稳健、多样化的表示铺平了道路，并显著提高了性能。 |
| [^6] | [Multivariate Time-Series Anomaly Detection with Contaminated Data: Application to Physiological Signals.](http://arxiv.org/abs/2308.12563) | 这个论文介绍了一种针对带有污染数据的多变量时间序列异常检测的新方法，通过去污和变量依赖建模实现了无监督的异常检测，对于实际场景中的异常检测具有重要意义。 |
| [^7] | [JEN-1: Text-Guided Universal Music Generation with Omnidirectional Diffusion Models.](http://arxiv.org/abs/2308.04729) | JEN-1是一个高保真度通用音乐生成模型，通过结合自回归和非自回归训练，实现了文本引导的音乐生成、音乐修补和延续等生成任务，在文本音乐对齐和音乐质量方面表现出优越性，同时保持计算效率。 |
| [^8] | [Optimal Low-Rank Matrix Completion: Semidefinite Relaxations and Eigenvector Disjunctions.](http://arxiv.org/abs/2305.12292) | 该论文通过重新表述低秩矩阵填补问题为投影矩阵的非凸问题，实现了能够确定最优解的分离分支定界方案，并且通过新颖和紧密的凸松弛方法，使得最优性差距相对于现有方法减少了两个数量级。 |
| [^9] | [A Machine Learning Approach to Forecasting Honey Production with Tree-Based Methods.](http://arxiv.org/abs/2304.01215) | 该论文使用树模型方法预测意大利蜂箱的蜜蜂生产量变化，帮助蜜蜂养殖者评估风险，保护蜜蜂活动。 |
| [^10] | [Auto.gov: Learning-based On-chain Governance for Decentralized Finance (DeFi).](http://arxiv.org/abs/2302.09551) | 这项研究提出了一个“Auto.gov”框架，可增强去中心化金融（DeFi）的安全性和降低受攻击的风险。该框架利用深度Q-网络（DQN）强化学习方法，提出了半自动的、直观的治理提案，并量化了其理由，使系统能够有效地应对恶意行为和意外的市场情况。 |

# 详细

[^1]: 在现实世界中使用商品移动操作器打开橱柜和抽屉

    Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator

    [https://arxiv.org/abs/2402.17767](https://arxiv.org/abs/2402.17767)

    实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。

    

    在这项工作中，我们构建了一个端到端系统，使商品移动操作器（Stretch RE2）能够在多样的以前未见的真实世界环境中拉开橱柜和抽屉。我们在31个不同的物体和13个不同真实世界环境中进行了4天的实际测试。我们的系统在零击打下，对在未知环境中新颖的橱柜和抽屉的打开率达到61%。对失败模式的分析表明，感知误差是我们系统面临的最重要挑战。

    arXiv:2402.17767v1 Announce Type: cross  Abstract: Pulling open cabinets and drawers presents many difficult technical challenges in perception (inferring articulation parameters for objects from onboard sensors), planning (producing motion plans that conform to tight task constraints), and control (making and maintaining contact while applying forces on the environment). In this work, we build an end-to-end system that enables a commodity mobile manipulator (Stretch RE2) to pull open cabinets and drawers in diverse previously unseen real world environments. We conduct 4 days of real world testing of this system spanning 31 different objects from across 13 different real world environments. Our system achieves a success rate of 61% on opening novel cabinets and drawers in unseen environments zero-shot. An analysis of the failure modes suggests that errors in perception are the most significant challenge for our system. We will open source code and models for others to replicate and bui
    
[^2]: TransAxx：具有近似计算能力的高效Transformer模型

    TransAxx: Efficient Transformers with Approximate Computing

    [https://arxiv.org/abs/2402.07545](https://arxiv.org/abs/2402.07545)

    TransAxx是一个基于PyTorch库的框架，可以支持近似计算，并通过对Vision Transformer模型进行近似感知微调，来提高在低功耗设备上的计算效率。

    

    最近，基于Transformer架构引入的Vision Transformer (ViT)模型已经展现出很大的竞争力，并且往往成为卷积神经网络(CNNs)的一种流行替代方案。然而，这些模型高计算需求限制了它们在低功耗设备上的实际应用。当前最先进的方法采用近似乘法器来解决DNN加速器高计算需求的问题，但之前的研究并没有探索其在ViT模型上的应用。在这项工作中，我们提出了TransAxx，这是一个基于流行的PyTorch库的框架，它能够快速支持近似算术，以无缝地评估近似计算对于DNN (如ViT模型)的影响。使用TransAxx，我们分析了Transformer模型在ImageNet数据集上对近似乘法的敏感性，并进行了近似感知的微调以恢复准确性。此外，我们提出了一种生成近似加法的方法。

    Vision Transformer (ViT) models which were recently introduced by the transformer architecture have shown to be very competitive and often become a popular alternative to Convolutional Neural Networks (CNNs). However, the high computational requirements of these models limit their practical applicability especially on low-power devices. Current state-of-the-art employs approximate multipliers to address the highly increased compute demands of DNN accelerators but no prior research has explored their use on ViT models. In this work we propose TransAxx, a framework based on the popular PyTorch library that enables fast inherent support for approximate arithmetic to seamlessly evaluate the impact of approximate computing on DNNs such as ViT models. Using TransAxx we analyze the sensitivity of transformer models on the ImageNet dataset to approximate multiplications and perform approximate-aware finetuning to regain accuracy. Furthermore, we propose a methodology to generate approximate ac
    
[^3]: 领航神经空间：重新审视概念激活向量以克服方向差异

    Navigating Neural Space: Revisiting Concept Activation Vectors to Overcome Directional Divergence

    [https://arxiv.org/abs/2202.03482](https://arxiv.org/abs/2202.03482)

    本文重新审视了概念激活向量（CAVs）在建模人类可理解的概念中的应用，并引入了基于模式的CAVs来提供更准确的概念方向。

    

    随着对于理解神经网络预测策略的兴趣日益增长，概念激活向量（CAVs）已成为一种流行的工具，用于在潜在空间中建模人类可理解的概念。通常，CAVs是通过利用线性分类器来计算的，该分类器优化具有给定概念和无给定概念的样本的潜在表示的可分离性。然而，在本文中我们展示了这种以可分离性为导向的计算方法会导致与精确建模概念方向的实际目标发散的解决方案。这种差异可以归因于分散方向的显著影响，即与概念无关的信号，这些信号被线性模型的滤波器（即权重）捕获以优化类别可分性。为了解决这个问题，我们引入基于模式的CAVs，仅关注概念信号，从而提供更准确的概念方向。我们评估了各种CAV方法与真实概念方向的对齐程度。

    With a growing interest in understanding neural network prediction strategies, Concept Activation Vectors (CAVs) have emerged as a popular tool for modeling human-understandable concepts in the latent space. Commonly, CAVs are computed by leveraging linear classifiers optimizing the separability of latent representations of samples with and without a given concept. However, in this paper we show that such a separability-oriented computation leads to solutions, which may diverge from the actual goal of precisely modeling the concept direction. This discrepancy can be attributed to the significant influence of distractor directions, i.e., signals unrelated to the concept, which are picked up by filters (i.e., weights) of linear models to optimize class-separability. To address this, we introduce pattern-based CAVs, solely focussing on concept signals, thereby providing more accurate concept directions. We evaluate various CAV methods in terms of their alignment with the true concept dire
    
[^4]: Mori-Zwanzig潜变空间Koopman闭包用于非线性自编码器

    Mori-Zwanzig latent space Koopman closure for nonlinear autoencoder. (arXiv:2310.10745v1 [cs.LG])

    [http://arxiv.org/abs/2310.10745](http://arxiv.org/abs/2310.10745)

    本研究提出了一种名为Mori-Zwanzig自编码器（MZ-AE）的新方法，用于在低维空间中稳健地逼近Koopman算子，通过非线性自编码器和Mori-Zwanzig形式主义的集成实现对有限不变Koopman子空间的逼近，从而增强了精确性和准确预测复杂系统行为的能力。

    

    Koopman算子提供了一种吸引人的方法来实现非线性系统的全局线性化，使其成为简化复杂动力学理解的宝贵方法。虽然数据驱动的方法在逼近有限Koopman算子方面表现出了潜力，但它们面临着各种挑战，例如选择合适的可观察量、降维和准确预测复杂系统行为的能力。本研究提出了一种名为Mori-Zwanzig自编码器（MZ-AE）的新方法，用于在低维空间中稳健地逼近Koopman算子。所提出的方法利用非线性自编码器提取关键可观察量来逼近有限不变Koopman子空间，并利用Mori-Zwanzig形式主义集成非马尔可夫校正机制。因此，该方法在非线性自编码器的潜变流形中产生了动力学的封闭表示，从而提高了精确性和...

    The Koopman operator presents an attractive approach to achieve global linearization of nonlinear systems, making it a valuable method for simplifying the understanding of complex dynamics. While data-driven methodologies have exhibited promise in approximating finite Koopman operators, they grapple with various challenges, such as the judicious selection of observables, dimensionality reduction, and the ability to predict complex system behaviours accurately. This study presents a novel approach termed Mori-Zwanzig autoencoder (MZ-AE) to robustly approximate the Koopman operator in low-dimensional spaces. The proposed method leverages a nonlinear autoencoder to extract key observables for approximating a finite invariant Koopman subspace and integrates a non-Markovian correction mechanism using the Mori-Zwanzig formalism. Consequently, this approach yields a closed representation of dynamics within the latent manifold of the nonlinear autoencoder, thereby enhancing the precision and s
    
[^5]: 多样的神经音频嵌入 - 恢复特征！

    Diverse Neural Audio Embeddings -- Bringing Features back !. (arXiv:2309.08751v1 [cs.SD])

    [http://arxiv.org/abs/2309.08751](http://arxiv.org/abs/2309.08751)

    本文通过在音频分类任务中学习多样化的特征表示，包括领域特定的音高、音色和神经表示，以及端到端架构，为学习稳健、多样化的表示铺平了道路，并显著提高了性能。

    

    随着现代人工智能架构的出现，从端到端的架构开始流行。这种转变导致了神经架构在没有领域特定偏见/知识的情况下进行训练，根据任务进行优化。本文中，我们通过多样的特征表示（在本例中是领域特定的）学习音频嵌入。对于涉及数百种声音分类的情况，我们学习分别针对音高、音色和神经表示等多样的音频属性建立稳健的嵌入，同时也通过端到端架构进行学习。我们观察到手工制作的嵌入，例如基于音高和音色的嵌入，虽然单独使用时无法击败完全端到端的表示，但将这些嵌入与端到端嵌入相结合可以显著提高性能。这项工作将为在端到端模型中引入一些领域专业知识来学习稳健、多样化的表示铺平道路，并超越仅训练端到端模型的性能。

    With the advent of modern AI architectures, a shift has happened towards end-to-end architectures. This pivot has led to neural architectures being trained without domain-specific biases/knowledge, optimized according to the task. We in this paper, learn audio embeddings via diverse feature representations, in this case, domain-specific. For the case of audio classification over hundreds of categories of sound, we learn robust separate embeddings for diverse audio properties such as pitch, timbre, and neural representation, along with also learning it via an end-to-end architecture. We observe handcrafted embeddings, e.g., pitch and timbre-based, although on their own, are not able to beat a fully end-to-end representation, yet adding these together with end-to-end embedding helps us, significantly improve performance. This work would pave the way to bring some domain expertise with end-to-end models to learn robust, diverse representations, surpassing the performance of just training 
    
[^6]: 针对带有污染数据的多变量时间序列异常检测：对生理信号的应用

    Multivariate Time-Series Anomaly Detection with Contaminated Data: Application to Physiological Signals. (arXiv:2308.12563v1 [cs.LG])

    [http://arxiv.org/abs/2308.12563](http://arxiv.org/abs/2308.12563)

    这个论文介绍了一种针对带有污染数据的多变量时间序列异常检测的新方法，通过去污和变量依赖建模实现了无监督的异常检测，对于实际场景中的异常检测具有重要意义。

    

    主流无监督异常检测算法通常在学术数据集中表现出色，但由于受到控制实验条件下的清洁训练数据的限制，它们在实际场景下的性能受到了限制。然而，在实际异常检测中，训练数据包含噪声的挑战经常被忽视。本研究在感知时间序列异常检测（TSAD）中深入研究了标签级噪声的领域。本文提出了一种新颖实用的端到端无监督TSAD方法，该方法处理训练数据中包含异常的情况下。该方法称为TSAD-C，其在训练阶段不需要访问异常标签。TSAD-C包括三个模块：一个去污器用于纠正训练数据中存在的异常（也称为噪声），一个变量依赖建模模块用于捕捉去污后数据中的长期内部和跨变量依赖关系，可以视为替代性的异常性度量方法。

    Mainstream unsupervised anomaly detection algorithms often excel in academic datasets, yet their real-world performance is restricted due to the controlled experimental conditions involving clean training data. Addressing the challenge of training with noise, a prevalent issue in practical anomaly detection, is frequently overlooked. In a pioneering endeavor, this study delves into the realm of label-level noise within sensory time-series anomaly detection (TSAD). This paper presents a novel and practical end-to-end unsupervised TSAD when the training data are contaminated with anomalies. The introduced approach, called TSAD-C, is devoid of access to abnormality labels during the training phase. TSAD-C encompasses three modules: a Decontaminator to rectify the abnormalities (aka noise) present in the training data, a Variable Dependency Modeling module to capture both long-term intra- and inter-variable dependencies within the decontaminated data that can be considered as a surrogate o
    
[^7]: JEN-1：具有全向扩散模型的文本引导通用音乐生成

    JEN-1: Text-Guided Universal Music Generation with Omnidirectional Diffusion Models. (arXiv:2308.04729v1 [cs.SD])

    [http://arxiv.org/abs/2308.04729](http://arxiv.org/abs/2308.04729)

    JEN-1是一个高保真度通用音乐生成模型，通过结合自回归和非自回归训练，实现了文本引导的音乐生成、音乐修补和延续等生成任务，在文本音乐对齐和音乐质量方面表现出优越性，同时保持计算效率。

    

    随着深度生成模型的进步，音乐生成引起了越来越多的关注。然而，基于文本描述生成音乐（即文本到音乐）仍然具有挑战性，原因是音乐结构的复杂性和高采样率的要求。尽管任务的重要性，当前的生成模型在音乐质量、计算效率和泛化能力方面存在局限性。本文介绍了JEN-1，这是一个用于文本到音乐生成的通用高保真模型。JEN-1是一个结合了自回归和非自回归训练的扩散模型。通过上下文学习，JEN-1可以执行各种生成任务，包括文本引导的音乐生成、音乐修补以及延续。评估结果表明，JEN-1在文本音乐对齐和音乐质量方面表现出优越性，同时保持计算效率。我们的演示可在此网址获取：http://URL

    Music generation has attracted growing interest with the advancement of deep generative models. However, generating music conditioned on textual descriptions, known as text-to-music, remains challenging due to the complexity of musical structures and high sampling rate requirements. Despite the task's significance, prevailing generative models exhibit limitations in music quality, computational efficiency, and generalization. This paper introduces JEN-1, a universal high-fidelity model for text-to-music generation. JEN-1 is a diffusion model incorporating both autoregressive and non-autoregressive training. Through in-context learning, JEN-1 performs various generation tasks including text-guided music generation, music inpainting, and continuation. Evaluations demonstrate JEN-1's superior performance over state-of-the-art methods in text-music alignment and music quality while maintaining computational efficiency. Our demos are available at this http URL
    
[^8]: 最优低秩矩阵填补：半定松弛和特征向量分离

    Optimal Low-Rank Matrix Completion: Semidefinite Relaxations and Eigenvector Disjunctions. (arXiv:2305.12292v1 [cs.LG])

    [http://arxiv.org/abs/2305.12292](http://arxiv.org/abs/2305.12292)

    该论文通过重新表述低秩矩阵填补问题为投影矩阵的非凸问题，实现了能够确定最优解的分离分支定界方案，并且通过新颖和紧密的凸松弛方法，使得最优性差距相对于现有方法减少了两个数量级。

    

    低秩矩阵填补的目的是计算一个复杂度最小的矩阵，以尽可能准确地恢复给定的一组观测数据，并且具有众多应用，如产品推荐。不幸的是，现有的解决低秩矩阵填补的方法是启发式的，虽然高度可扩展并且通常能够确定高质量的解决方案，但不具备任何最优性保证。我们通过将低秩问题重新表述为投影矩阵的非凸问题，并实现一种分离分支定界方案来重新审视矩阵填补问题，以实现最优性导向。此外，我们通过将低秩矩阵分解为一组秩一矩阵的和，并通过 Shor 松弛来激励每个秩一矩阵中的每个 2*2 小矩阵的行列式为零，从而推导出一种新颖且通常很紧的凸松弛类。在数值实验中，相对于最先进的启发式方法，我们的新凸松弛方法将最优性差距减少了两个数量级。

    Low-rank matrix completion consists of computing a matrix of minimal complexity that recovers a given set of observations as accurately as possible, and has numerous applications such as product recommendation. Unfortunately, existing methods for solving low-rank matrix completion are heuristics that, while highly scalable and often identifying high-quality solutions, do not possess any optimality guarantees. We reexamine matrix completion with an optimality-oriented eye, by reformulating low-rank problems as convex problems over the non-convex set of projection matrices and implementing a disjunctive branch-and-bound scheme that solves them to certifiable optimality. Further, we derive a novel and often tight class of convex relaxations by decomposing a low-rank matrix as a sum of rank-one matrices and incentivizing, via a Shor relaxation, that each two-by-two minor in each rank-one matrix has determinant zero. In numerical experiments, our new convex relaxations decrease the optimali
    
[^9]: 基于树模型的机器学习方法预测蜜蜂生产量

    A Machine Learning Approach to Forecasting Honey Production with Tree-Based Methods. (arXiv:2304.01215v1 [cs.LG])

    [http://arxiv.org/abs/2304.01215](http://arxiv.org/abs/2304.01215)

    该论文使用树模型方法预测意大利蜂箱的蜜蜂生产量变化，帮助蜜蜂养殖者评估风险，保护蜜蜂活动。

    

    蜜蜂养殖业在过去几年中经历了可观的生产变化，这是由于逐渐恶化的气候条件造成的。这些现象可能会对蜜蜂活动不利。我们使用树模型方法区分蜜蜂生产量的影响因素，并预测意大利蜂箱的蜜蜂生产量变化，意大利是欧洲最大的蜂蜜生产国之一。该数据库包含了从2019年到2022年收集的数百个蜂箱的数据，采用了先进的精度蜜蜂养殖技术。我们训练和解释机器学习模型，使其具有指导性而不仅仅是预测性。与标准线性技术相比，树模型方法具有更高的预测性能，可以更好地保护蜜蜂活动并评估风险管理中蜜蜂养殖者的潜在损失。

    The beekeeping sector has undergone considerable production variations over the past years due to adverse weather conditions, occurring more frequently as climate change progresses. These phenomena can be high-impact and cause the environment to be unfavorable to the bees' activity. We disentangle the honey production drivers with tree-based methods and predict honey production variations for hives in Italy, one of the largest honey producers in Europe. The database covers hundreds of beehive data from 2019-2022 gathered with advanced precision beekeeping techniques. We train and interpret the machine learning models making them prescriptive other than just predictive. Superior predictive performances of tree-based methods compared to standard linear techniques allow for better protection of bees' activity and assess potential losses for beekeepers for risk management.
    
[^10]: Auto.gov：面向DeFi的基于学习的链上治理

    Auto.gov: Learning-based On-chain Governance for Decentralized Finance (DeFi). (arXiv:2302.09551v2 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2302.09551](http://arxiv.org/abs/2302.09551)

    这项研究提出了一个“Auto.gov”框架，可增强去中心化金融（DeFi）的安全性和降低受攻击的风险。该框架利用深度Q-网络（DQN）强化学习方法，提出了半自动的、直观的治理提案，并量化了其理由，使系统能够有效地应对恶意行为和意外的市场情况。

    

    近年来，去中心化金融（DeFi）经历了显著增长，涌现出了各种协议，例如借贷协议和自动化做市商（AMM）。传统上，这些协议采用链下治理，其中代币持有者投票修改参数。然而，由协议核心团队进行的手动参数调整容易遭受勾结攻击，危及系统的完整性和安全性。此外，纯粹的确定性算法方法可能会使协议受到新的利用和攻击的威胁。本文提出了“Auto.gov”，这是一个面向DeFi的基于学习的链上治理框架，可增强安全性并降低受攻击的风险。我们的模型利用了深度Q-网络（DQN）强化学习方法，提出了半自动化的、直观的治理提案与量化的理由。这种方法使系统能够有效地适应和缓解恶意行为和意外的市场情况的负面影响。

    In recent years, decentralized finance (DeFi) has experienced remarkable growth, with various protocols such as lending protocols and automated market makers (AMMs) emerging. Traditionally, these protocols employ off-chain governance, where token holders vote to modify parameters. However, manual parameter adjustment, often conducted by the protocol's core team, is vulnerable to collusion, compromising the integrity and security of the system. Furthermore, purely deterministic, algorithm-based approaches may expose the protocol to novel exploits and attacks.  In this paper, we present "Auto.gov", a learning-based on-chain governance framework for DeFi that enhances security and reduces susceptibility to attacks. Our model leverages a deep Q- network (DQN) reinforcement learning approach to propose semi-automated, intuitive governance proposals with quantitative justifications. This methodology enables the system to efficiently adapt to and mitigate the negative impact of malicious beha
    

