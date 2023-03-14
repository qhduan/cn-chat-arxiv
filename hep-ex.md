# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generative Adversarial Networks for Scintillation Signal Simulation in EXO-200.](http://arxiv.org/abs/2303.06311) | 本文介绍了一种基于生成对抗网络的新方法，用于从EXO-200实验的时间投影室中模拟光电探测器信号。该方法能够比传统的模拟方法快一个数量级地产生高质量的模拟波形，并且能够从训练样本中推广并识别数据的显著高级特征。 |

# 详细

[^1]: 生成对抗网络在EXO-200闪烁信号模拟中的应用

    Generative Adversarial Networks for Scintillation Signal Simulation in EXO-200. (arXiv:2303.06311v1 [hep-ex])

    [http://arxiv.org/abs/2303.06311](http://arxiv.org/abs/2303.06311)

    本文介绍了一种基于生成对抗网络的新方法，用于从EXO-200实验的时间投影室中模拟光电探测器信号。该方法能够比传统的模拟方法快一个数量级地产生高质量的模拟波形，并且能够从训练样本中推广并识别数据的显著高级特征。

    This paper introduces a novel approach using Generative Adversarial Networks to simulate photodetector signals from the time projection chamber of the EXO-200 experiment. The method is able to produce high-quality simulated waveforms an order of magnitude faster than traditional simulation methods and can generalize from the training sample and discern salient high-level features of the data.

    基于模拟或实际事件样本训练的生成对抗网络被提出作为一种以降低计算成本为代价生成大规模模拟数据集的方法。本文展示了一种新的方法，用于从EXO-200实验的时间投影室中模拟光电探测器信号。该方法基于Wasserstein生成对抗网络，这是一种深度学习技术，允许对给定对象集的总体分布进行隐式非参数估计。我们的网络使用原始闪烁波形作为输入，通过对真实校准数据进行训练。我们发现，它能够比传统的模拟方法快一个数量级地产生高质量的模拟波形，并且重要的是，能够从训练样本中推广并识别数据的显著高级特征。特别是，网络正确推断出探测器中闪烁光响应的位置依赖性和相关性。

    Generative Adversarial Networks trained on samples of simulated or actual events have been proposed as a way of generating large simulated datasets at a reduced computational cost. In this work, a novel approach to perform the simulation of photodetector signals from the time projection chamber of the EXO-200 experiment is demonstrated. The method is based on a Wasserstein Generative Adversarial Network - a deep learning technique allowing for implicit non-parametric estimation of the population distribution for a given set of objects. Our network is trained on real calibration data using raw scintillation waveforms as input. We find that it is able to produce high-quality simulated waveforms an order of magnitude faster than the traditional simulation approach and, importantly, generalize from the training sample and discern salient high-level features of the data. In particular, the network correctly deduces position dependency of scintillation light response in the detector and corr
    

