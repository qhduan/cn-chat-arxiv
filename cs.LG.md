# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improved DDIM Sampling with Moment Matching Gaussian Mixtures.](http://arxiv.org/abs/2311.04938) | 在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。 |
| [^2] | [TreeDQN: Learning to minimize Branch-and-Bound tree.](http://arxiv.org/abs/2306.05905) | TreeDQN提出了一种强化学习方法，可以学习到更加效率的分支启发式算法，减少了训练数据，并产生更小的子任务树。 |
| [^3] | [Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples.](http://arxiv.org/abs/2209.03358) | 这项研究主要关注于脉冲神经网络(SNNs)对抗性样本的鲁棒性和转移性。研究发现，成功的白盒对抗攻击SNNs在很大程度上依赖于替代梯度技术，并且非SNN架构创建的对抗样本往往不被SNNs误分类。 |

# 详细

[^1]: 使用矩匹配高斯混合模型改进了DDIM采样

    Improved DDIM Sampling with Moment Matching Gaussian Mixtures. (arXiv:2311.04938v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2311.04938](http://arxiv.org/abs/2311.04938)

    在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。

    

    我们提出在Denoising Diffusion Implicit Models (DDIM)框架中使用高斯混合模型（GMM）作为反向转移算子（内核），这是一种从预训练的Denoising Diffusion Probabilistic Models (DDPM)中加速采样的广泛应用方法之一。具体而言，我们通过约束GMM的参数，匹配DDPM前向边际的一阶和二阶中心矩。我们发现，通过矩匹配，可以获得与使用高斯核的原始DDIM相同或更好质量的样本。我们在CelebAHQ和FFHQ的无条件模型以及ImageNet数据集的类条件模型上提供了实验结果。我们的结果表明，在采样步骤较少的情况下，使用GMM内核可以显著改善生成样本的质量，这是通过FID和IS指标衡量的。例如，在ImageNet 256x256上，使用10个采样步骤，我们实现了一个FID值为...

    We propose using a Gaussian Mixture Model (GMM) as reverse transition operator (kernel) within the Denoising Diffusion Implicit Models (DDIM) framework, which is one of the most widely used approaches for accelerated sampling from pre-trained Denoising Diffusion Probabilistic Models (DDPM). Specifically we match the first and second order central moments of the DDPM forward marginals by constraining the parameters of the GMM. We see that moment matching is sufficient to obtain samples with equal or better quality than the original DDIM with Gaussian kernels. We provide experimental results with unconditional models trained on CelebAHQ and FFHQ and class-conditional models trained on ImageNet datasets respectively. Our results suggest that using the GMM kernel leads to significant improvements in the quality of the generated samples when the number of sampling steps is small, as measured by FID and IS metrics. For example on ImageNet 256x256, using 10 sampling steps, we achieve a FID of
    
[^2]: TreeDQN: 学习如何最小化分支定界树

    TreeDQN: Learning to minimize Branch-and-Bound tree. (arXiv:2306.05905v1 [cs.LG])

    [http://arxiv.org/abs/2306.05905](http://arxiv.org/abs/2306.05905)

    TreeDQN提出了一种强化学习方法，可以学习到更加效率的分支启发式算法，减少了训练数据，并产生更小的子任务树。

    

    组合优化问题需要通过全面搜索才能找到最优解。分支定界是解决混合整数线性规划问题的方便方法。分支定界求解器将任务分成两个部分，将整数变量的域分成两个部分，然后递归地解决它们，产生一个嵌套子任务树。求解器的效率取决于用于选择分裂变量的分支启发式算法。在本研究中，我们提出了一种强化学习方法，可以有效地学习分支启发式算法。我们将变量选择任务视为树形马尔科夫决策过程，并证明了适用于树形马尔科夫决策过程的贝尔曼算子平均收缩，并提出了修改后的强化学习代理的学习目标。与之前的强化学习方法相比，我们的代理需要更少的训练数据，并产生更小的子任务树。

    Combinatorial optimization problems require an exhaustive search to find the optimal solution. A convenient approach to solving combinatorial optimization tasks in the form of Mixed Integer Linear Programs is Branch-and-Bound. Branch-and-Bound solver splits a task into two parts dividing the domain of an integer variable, then it solves them recursively, producing a tree of nested sub-tasks. The efficiency of the solver depends on the branchning heuristic used to select a variable for splitting. In the present work, we propose a reinforcement learning method that can efficiently learn the branching heuristic. We view the variable selection task as a tree Markov Decision Process, prove that the Bellman operator adapted for the tree Markov Decision Process is contracting in mean, and propose a modified learning objective for the reinforcement learning agent. Our agent requires less training data and produces smaller trees compared to previous reinforcement learning methods.
    
[^3]: 攻击脉冲：关于脉冲神经网络对抗性样本的可转移性与安全性的研究

    Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples. (arXiv:2209.03358v3 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2209.03358](http://arxiv.org/abs/2209.03358)

    这项研究主要关注于脉冲神经网络(SNNs)对抗性样本的鲁棒性和转移性。研究发现，成功的白盒对抗攻击SNNs在很大程度上依赖于替代梯度技术，并且非SNN架构创建的对抗样本往往不被SNNs误分类。

    

    脉冲神经网络(SNNs)因其高能效和最近在分类性能上的进展而受到广泛关注。然而，与传统的深度学习方法不同，对SNNs对抗性样本的鲁棒性的分析和研究仍然相对不完善。在这项工作中，我们关注于推进SNNs的对抗攻击方面，并做出了三个主要贡献。首先，我们展示了成功的白盒对抗攻击SNNs在很大程度上依赖于底层的替代梯度技术，即使在对抗性训练SNNs的情况下也一样。其次，利用最佳的替代梯度技术，我们分析了对抗攻击在SNNs和其他最先进的架构如Vision Transformers(ViTs)和Big Transfer Convolutional Neural Networks(CNNs)之间的可转移性。我们证明了非SNN架构创建的对抗样本往往不被SNNs误分类。第三，由于缺乏一个共性

    Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and for recent advances in their classification performance. However, unlike traditional deep learning approaches, the analysis and study of the robustness of SNNs to adversarial examples remain relatively underdeveloped. In this work, we focus on advancing the adversarial attack side of SNNs and make three major contributions. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique, even in the case of adversarially trained SNNs. Second, using the best surrogate gradient technique, we analyze the transferability of adversarial attacks on SNNs and other state-of-the-art architectures like Vision Transformers (ViTs) and Big Transfer Convolutional Neural Networks (CNNs). We demonstrate that the adversarial examples created by non-SNN architectures are not misclassified often by SNNs. Third, due to the lack of an ubi
    

