# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Kinematic Evidence of an Embedded Protoplanet in HD 142666 Identified by Machine Learning.](http://arxiv.org/abs/2301.05075) | 该论文使用机器学习模型在HD 142666的盘中识别出强烈的、局部的非开普勒运动，进而得出该盘中存在一个行星的结论，这是使用机器学习识别原行星盘中先前被忽视的非开普勒特征的第一步。 |
| [^2] | [Removing Radio Frequency Interference from Auroral Kilometric Radiation with Stacked Autoencoders.](http://arxiv.org/abs/2210.12931) | 本研究利用深度学习算法中的自编码器，提出了一种名为DAARE的去噪自编码器，用于消除极光千米辐射中的射频干扰。DAARE在合成AKR观测中实现了42.2的峰值信噪比(PSNR)和0.981的结构相似度(SSIM)，相比于最先进的滤波和去噪网络，PSNR提高了3.9，SSIM提高了0.064。定性比较表明，DAARE能够有效地消除RFI。 |

# 详细

[^1]: 机器学习识别出HD 142666中嵌入的原行星的运动学证据

    Kinematic Evidence of an Embedded Protoplanet in HD 142666 Identified by Machine Learning. (arXiv:2301.05075v2 [astro-ph.EP] UPDATED)

    [http://arxiv.org/abs/2301.05075](http://arxiv.org/abs/2301.05075)

    该论文使用机器学习模型在HD 142666的盘中识别出强烈的、局部的非开普勒运动，进而得出该盘中存在一个行星的结论，这是使用机器学习识别原行星盘中先前被忽视的非开普勒特征的第一步。

    This paper uses machine learning models to identify strong, localized non-Keplerian motion in the disk HD 142666, and concludes that there is a planet in the disk, which represents a first step towards using machine learning to identify previously overlooked non-Keplerian features in protoplanetary disks.

    原行星盘的观测表明，形成系外行星会在盘中的气体和尘埃上留下特征印记。在气体中，这些形成中的系外行星会引起开普勒运动的偏差，可以通过分子线观测来检测。我们之前的工作表明，机器学习可以正确确定这些盘中是否存在行星。使用我们的机器学习模型，我们在HD 142666的盘中识别出强烈的、局部的非开普勒运动。随后进行的一个系统中有一个5个木星质量的行星在75天文单位处的流体动力学模拟再现了这种运动学结构。根据该领域目前已经建立的标准，我们得出结论：HD 142666中存在一个行星。这项工作代表了使用机器学习识别原行星盘中先前被忽视的非开普勒特征的第一步。

    Observations of protoplanetary disks have shown that forming exoplanets leave characteristic imprints on the gas and dust of the disk. In the gas, these forming exoplanets cause deviations from Keplerian motion, which can be detected through molecular line observations. Our previous work has shown that machine learning can correctly determine if a planet is present in these disks. Using our machine learning models, we identify strong, localized non-Keplerian motion within the disk HD 142666. Subsequent hydrodynamics simulations of a system with a 5 Jupiter-mass planet at 75 au recreates the kinematic structure. By currently established standards in the field, we conclude that HD 142666 hosts a planet. This work represents a first step towards using machine learning to identify previously overlooked non-Keplerian features in protoplanetary disks.
    
[^2]: 利用堆叠自编码器消除极光千米辐射中的射频干扰

    Removing Radio Frequency Interference from Auroral Kilometric Radiation with Stacked Autoencoders. (arXiv:2210.12931v3 [astro-ph.IM] UPDATED)

    [http://arxiv.org/abs/2210.12931](http://arxiv.org/abs/2210.12931)

    本研究利用深度学习算法中的自编码器，提出了一种名为DAARE的去噪自编码器，用于消除极光千米辐射中的射频干扰。DAARE在合成AKR观测中实现了42.2的峰值信噪比(PSNR)和0.981的结构相似度(SSIM)，相比于最先进的滤波和去噪网络，PSNR提高了3.9，SSIM提高了0.064。定性比较表明，DAARE能够有效地消除RFI。

    This study proposes a denoising autoencoder named DAARE to remove radio frequency interference (RFI) from auroral kilometric radiation (AKR) signals collected at the South Pole Station. DAARE achieves 42.2 peak signal-to-noise ratio (PSNR) and 0.981 structural similarity (SSIM) on synthesized AKR observations, improving PSNR by 3.9 and SSIM by 0.064 compared to state-of-the-art filtering and denoising networks.

    天文射电频率数据可以帮助科学家分析天体物理现象。然而，这些数据可能会受到射频干扰(RFI)的影响，从而限制了对基础自然过程的观测。本研究将深度学习算法的最新进展扩展到天文数据中。我们从南极站收集的极光千米辐射(AKR)信号中，利用合成光谱图训练了一种名为DAARE的去噪自编码器，以消除RFI。DAARE在合成AKR观测中实现了42.2的峰值信噪比(PSNR)和0.981的结构相似度(SSIM)，相比于最先进的滤波和去噪网络，PSNR提高了3.9，SSIM提高了0.064。定性比较表明，DAARE能够有效地消除RFI。

    Radio frequency data in astronomy enable scientists to analyze astrophysical phenomena. However, these data can be corrupted by radio frequency interference (RFI) that limits the observation of underlying natural processes. In this study, we extend recent developments in deep learning algorithms to astronomy data. We remove RFI from time-frequency spectrograms containing auroral kilometric radiation (AKR), a coherent radio emission originating from the Earth's auroral zones that is used to study astrophysical plasmas. We propose a Denoising Autoencoder for Auroral Radio Emissions (DAARE) trained with synthetic spectrograms to denoise AKR signals collected at the South Pole Station. DAARE achieves 42.2 peak signal-to-noise ratio (PSNR) and 0.981 structural similarity (SSIM) on synthesized AKR observations, improving PSNR by 3.9 and SSIM by 0.064 compared to state-of-the-art filtering and denoising networks. Qualitative comparisons demonstrate DAARE's capability to effectively remove RFI
    

