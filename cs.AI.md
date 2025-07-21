# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SecurePose: Automated Face Blurring and Human Movement Kinematics Extraction from Videos Recorded in Clinical Settings](https://arxiv.org/abs/2402.14143) | SecurePose是一个开源软件，可以可靠地实现临床录制的患者视频中的人脸模糊和动力学特征提取，提高了视频评估和患者隐私的安全性。 |
| [^2] | [Improved DDIM Sampling with Moment Matching Gaussian Mixtures.](http://arxiv.org/abs/2311.04938) | 在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。 |

# 详细

[^1]: SecurePose：在临床环境中录制的视频中实现自动人脸模糊和人体运动动力学特征提取

    SecurePose: Automated Face Blurring and Human Movement Kinematics Extraction from Videos Recorded in Clinical Settings

    [https://arxiv.org/abs/2402.14143](https://arxiv.org/abs/2402.14143)

    SecurePose是一个开源软件，可以可靠地实现临床录制的患者视频中的人脸模糊和动力学特征提取，提高了视频评估和患者隐私的安全性。

    

    运动障碍通常通过专家对临床获取的患者视频进行共识评估来诊断。然而，这种广泛分享患者视频会对患者隐私构成风险。人脸模糊可以用来去标识化视频，但这个过程通常是手动且耗时的。现有的自动人脸模糊技术容易出现过度、不一致或不足的人脸模糊 - 这些都可能对视频评估和患者隐私造成灾难性影响。此外，在这些视频中评估运动障碍往往是主观的。提取可量化的动力学特征可以帮助了解这些视频中的运动障碍评估，但现有的方法在使用预模糊视频时容易出现错误。我们开发了一个名为SecurePose的开源软件，可以在临床录制的患者视频中实现可靠的人脸模糊和自动动力学特征提取。

    arXiv:2402.14143v1 Announce Type: cross  Abstract: Movement disorders are typically diagnosed by consensus-based expert evaluation of clinically acquired patient videos. However, such broad sharing of patient videos poses risks to patient privacy. Face blurring can be used to de-identify videos, but this process is often manual and time-consuming. Available automated face blurring techniques are subject to either excessive, inconsistent, or insufficient facial blurring - all of which can be disastrous for video assessment and patient privacy. Furthermore, assessing movement disorders in these videos is often subjective. The extraction of quantifiable kinematic features can help inform movement disorder assessment in these videos, but existing methods to do this are prone to errors if using pre-blurred videos. We have developed an open-source software called SecurePose that can both achieve reliable face blurring and automated kinematic extraction in patient videos recorded in a clinic 
    
[^2]: 使用矩匹配高斯混合模型改进了DDIM采样

    Improved DDIM Sampling with Moment Matching Gaussian Mixtures. (arXiv:2311.04938v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2311.04938](http://arxiv.org/abs/2311.04938)

    在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。

    

    我们提出在Denoising Diffusion Implicit Models (DDIM)框架中使用高斯混合模型（GMM）作为反向转移算子（内核），这是一种从预训练的Denoising Diffusion Probabilistic Models (DDPM)中加速采样的广泛应用方法之一。具体而言，我们通过约束GMM的参数，匹配DDPM前向边际的一阶和二阶中心矩。我们发现，通过矩匹配，可以获得与使用高斯核的原始DDIM相同或更好质量的样本。我们在CelebAHQ和FFHQ的无条件模型以及ImageNet数据集的类条件模型上提供了实验结果。我们的结果表明，在采样步骤较少的情况下，使用GMM内核可以显著改善生成样本的质量，这是通过FID和IS指标衡量的。例如，在ImageNet 256x256上，使用10个采样步骤，我们实现了一个FID值为...

    We propose using a Gaussian Mixture Model (GMM) as reverse transition operator (kernel) within the Denoising Diffusion Implicit Models (DDIM) framework, which is one of the most widely used approaches for accelerated sampling from pre-trained Denoising Diffusion Probabilistic Models (DDPM). Specifically we match the first and second order central moments of the DDPM forward marginals by constraining the parameters of the GMM. We see that moment matching is sufficient to obtain samples with equal or better quality than the original DDIM with Gaussian kernels. We provide experimental results with unconditional models trained on CelebAHQ and FFHQ and class-conditional models trained on ImageNet datasets respectively. Our results suggest that using the GMM kernel leads to significant improvements in the quality of the generated samples when the number of sampling steps is small, as measured by FID and IS metrics. For example on ImageNet 256x256, using 10 sampling steps, we achieve a FID of
    

