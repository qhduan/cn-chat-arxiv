# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Privacy-Preserving Cooperative Visible Light Positioning for Nonstationary Environment: A Federated Learning Perspective.](http://arxiv.org/abs/2303.06361) | 本文提出了一种基于联邦学习的合作可见光定位方案，通过共同训练适应环境变化的全局模型，提高了在非静态环境下的定位精度和泛化能力。 |
| [^2] | [XG-BoT: An Explainable Deep Graph Neural Network for Botnet Detection and Forensics.](http://arxiv.org/abs/2207.09088) | 本文提出了一种名为XG-BoT的可解释的深度图神经网络模型，用于检测大规模网络中的恶意僵尸网络节点，并通过突出显示可疑的网络流和相关的僵尸网络节点来执行自动网络取证。该模型在关键评估指标方面优于现有的最先进方法。 |
| [^3] | [Developing a Trusted Human-AI Network for Humanitarian Benefit.](http://arxiv.org/abs/2112.11191) | 本文提出了一种可信的人工智能通信网络，将通信协议、区块链技术和信息融合与AI集成，以改善冲突通信，为人道主义利益提供可问责信息交换。 |
| [^4] | [PRONTO: Preamble Overhead Reduction with Neural Networks for Coarse Synchronization.](http://arxiv.org/abs/2112.10885) | 本文提出了一种名为PRONTO的基于神经网络的方案，用于在WiFi基础波形中执行粗略的时间和频率同步，以减少前导开销。该方案通过消除传统短训练场（L-STF）来缩短前导长度，并使用其他前导字段（特别是传统的长训练场（L-LTF））执行估计。 |

# 详细

[^1]: 面向非静态环境的隐私保护合作可见光定位：联邦学习视角

    Privacy-Preserving Cooperative Visible Light Positioning for Nonstationary Environment: A Federated Learning Perspective. (arXiv:2303.06361v1 [eess.SP])

    [http://arxiv.org/abs/2303.06361](http://arxiv.org/abs/2303.06361)

    本文提出了一种基于联邦学习的合作可见光定位方案，通过共同训练适应环境变化的全局模型，提高了在非静态环境下的定位精度和泛化能力。

    This paper proposes a cooperative visible light positioning scheme based on federated learning, which improves the positioning accuracy and generalization capability in nonstationary environments by jointly training a global model adaptive to environmental changes without sharing private data of users.

    可见光定位（VLP）作为一种有前途的室内定位技术，已经引起了足够的关注。然而，在非静态环境下，由于高度时变的信道，VLP的性能受到限制。为了提高非静态环境下的定位精度和泛化能力，本文提出了一种基于联邦学习（FL）的合作VLP方案。利用FL框架，用户可以共同训练适应环境变化的全局模型，而不共享用户的私有数据。此外，提出了一种合作可见光定位网络（CVPosNet），以加速收敛速度和提高定位精度。仿真结果表明，所提出的方案在非静态环境下优于基准方案。

    Visible light positioning (VLP) has drawn plenty of attention as a promising indoor positioning technique. However, in nonstationary environments, the performance of VLP is limited because of the highly time-varying channels. To improve the positioning accuracy and generalization capability in nonstationary environments, a cooperative VLP scheme based on federated learning (FL) is proposed in this paper. Exploiting the FL framework, a global model adaptive to environmental changes can be jointly trained by users without sharing private data of users. Moreover, a Cooperative Visible-light Positioning Network (CVPosNet) is proposed to accelerate the convergence rate and improve the positioning accuracy. Simulation results show that the proposed scheme outperforms the benchmark schemes, especially in nonstationary environments.
    
[^2]: XG-BoT：一种可解释的深度图神经网络用于僵尸网络检测和取证

    XG-BoT: An Explainable Deep Graph Neural Network for Botnet Detection and Forensics. (arXiv:2207.09088v5 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2207.09088](http://arxiv.org/abs/2207.09088)

    本文提出了一种名为XG-BoT的可解释的深度图神经网络模型，用于检测大规模网络中的恶意僵尸网络节点，并通过突出显示可疑的网络流和相关的僵尸网络节点来执行自动网络取证。该模型在关键评估指标方面优于现有的最先进方法。

    This paper proposes an explainable deep graph neural network model called XG-BoT for detecting malicious botnet nodes in large-scale networks and performing automatic network forensics by highlighting suspicious network flows and related botnet nodes. The model outperforms state-of-the-art approaches in terms of key evaluation metrics.

    本文提出了一种名为XG-BoT的可解释的深度图神经网络模型，用于检测僵尸网络节点。该模型包括一个僵尸网络检测器和一个自动取证的解释器。XG-BoT检测器可以有效地检测大规模网络中的恶意僵尸网络节点。具体而言，它利用分组可逆残差连接和图同构网络从僵尸网络通信图中学习表达性节点表示。基于GNNExplainer和XG-BoT中的显著性图，解释器可以通过突出显示可疑的网络流和相关的僵尸网络节点来执行自动网络取证。我们使用真实的大规模僵尸网络图数据集评估了XG-BoT。总体而言，XG-BoT在关键评估指标方面优于现有的最先进方法。此外，我们证明了XG-BoT解释器可以为自动网络取证生成有用的解释。

    In this paper, we propose XG-BoT, an explainable deep graph neural network model for botnet node detection. The proposed model comprises a botnet detector and an explainer for automatic forensics. The XG-BoT detector can effectively detect malicious botnet nodes in large-scale networks. Specifically, it utilizes a grouped reversible residual connection with a graph isomorphism network to learn expressive node representations from botnet communication graphs. The explainer, based on the GNNExplainer and saliency map in XG-BoT, can perform automatic network forensics by highlighting suspicious network flows and related botnet nodes. We evaluated XG-BoT using real-world, large-scale botnet network graph datasets. Overall, XG-BoT outperforms state-of-the-art approaches in terms of key evaluation metrics. Additionally, we demonstrate that the XG-BoT explainers can generate useful explanations for automatic network forensics.
    
[^3]: 为人道主义利益开发可信的人工智能网络

    Developing a Trusted Human-AI Network for Humanitarian Benefit. (arXiv:2112.11191v3 [cs.CY] UPDATED)

    [http://arxiv.org/abs/2112.11191](http://arxiv.org/abs/2112.11191)

    本文提出了一种可信的人工智能通信网络，将通信协议、区块链技术和信息融合与AI集成，以改善冲突通信，为人道主义利益提供可问责信息交换。

    This paper proposes a trusted human-AI communication network that integrates communication protocols, blockchain technology, and information fusion with AI to improve conflict communications for accountable information exchange regarding protected entities, critical infrastructure, and humanitarian signals and status updates for humans and machines in conflicts.

    人工智能（AI）将越来越多地在冲突中以数字和物理方式参与，但缺乏与人类进行人道主义目的的可信通信。本文考虑将通信协议（“白旗协议”）、分布式账本“区块链”技术和信息融合与AI集成，以改善冲突通信，称为“受保护的保证理解情况和实体”PAUSE。这样一个可信的人工智能通信网络可以提供关于受保护实体、关键基础设施、人道主义信号和人类和机器在冲突中的状态更新的可问责信息交换。我们研究了几个现实的潜在案例研究，将这些技术集成到一个可信的人工智能网络中，以实现人道主义利益，包括实时映射冲突区域的平民和战斗人员，为避免事故做准备，并使用网络管理错误信息。

    Artificial intelligences (AI) will increasingly participate digitally and physically in conflicts, yet there is a lack of trused communications with humans for humanitarian purposes. In this paper we consider the integration of a communications protocol (the 'whiteflag protocol'), distributed ledger 'blockchain' technology, and information fusion with AI, to improve conflict communications called 'protected assurance understanding situation and entitities' PAUSE. Such a trusted human-AI communication network could provide accountable information exchange regarding protected entities, critical infrastructure, humanitiarian signals and status updates for humans and machines in conflicts. We examine several realistic potential case studies for the integration of these technologies into a trusted human-AI network for humanitarian benefit including mapping a conflict zone with civilians and combatants in real time, preparation to avoid incidents and using the network to manage misinformatio
    
[^4]: PRONTO：基于神经网络的粗同步中的前导开销减少

    PRONTO: Preamble Overhead Reduction with Neural Networks for Coarse Synchronization. (arXiv:2112.10885v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2112.10885](http://arxiv.org/abs/2112.10885)

    本文提出了一种名为PRONTO的基于神经网络的方案，用于在WiFi基础波形中执行粗略的时间和频率同步，以减少前导开销。该方案通过消除传统短训练场（L-STF）来缩短前导长度，并使用其他前导字段（特别是传统的长训练场（L-LTF））执行估计。

    

    在IEEE 802.11 WiFi基础波形中，接收器使用前导的第一个字段（称为传统短训练场（L-STF））执行粗略的时间和频率同步。 L-STF占据了前导长度的高达40％，需要长达32微秒的空气时间。为了减少通信开销，我们提出了一种修改后的波形，其中通过消除L-STF来缩短前导长度。为了解码这种修改后的波形，我们提出了一种基于神经网络（NN）的方案，称为PRONTO，它使用其他前导字段（特别是传统的长训练场（L-LTF））执行粗略的时间和频率估计。我们的贡献有三个：（i）我们提出了PRONTO，其中包括用于数据检测和粗略载波频率偏移（CFO）估计的定制卷积神经网络（CNN），以及用于强化训练的数据增强步骤。 （ii）我们提出了一种广义决策流程，使PRONTO与包括th在内的传统波形兼容

    In IEEE 802.11 WiFi-based waveforms, the receiver performs coarse time and frequency synchronization using the first field of the preamble known as the legacy short training field (L-STF). The L-STF occupies upto 40% of the preamble length and takes upto 32 us of airtime. With the goal of reducing communication overhead, we propose a modified waveform, where the preamble length is reduced by eliminating the L-STF. To decode this modified waveform, we propose a neural network (NN)-based scheme called PRONTO that performs coarse time and frequency estimations using other preamble fields, specifically the legacy long training field (L-LTF). Our contributions are threefold: (i) We present PRONTO featuring customized convolutional neural networks (CNNs) for packet detection and coarse carrier frequency offset (CFO) estimation, along with data augmentation steps for robust training. (ii) We propose a generalized decision flow that makes PRONTO compatible with legacy waveforms that include th
    

