# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Minimax Optimal and Computationally Efficient Algorithms for Distributionally Robust Offline Reinforcement Learning](https://arxiv.org/abs/2403.09621) | 研究提出了最小化最优和计算高效的算法，为鲁棒离线强化学习中的函数逼近带来新颖视角，并展示了其与标准离线强化学习中函数逼近的区别。 |
| [^2] | [Optimal Batched Best Arm Identification.](http://arxiv.org/abs/2310.14129) | 本论文研究了最佳批处理武器识别问题，在渐近和非渐近设置中提出了Tri-BBAI和Opt-BBAI算法，分别实现了最优和几乎最优的样本和批处理复杂度。 |
| [^3] | [Assessing Robustness via Score-Based Adversarial Image Generation.](http://arxiv.org/abs/2310.04285) | 本论文介绍了一种基于分数的对抗生成框架（ScoreAG），可以生成超过$\ell_p$-范数约束的对抗性示例，并通过图像转换或新图像合成的方法保持图像的核心语义，大大增强了分类器的鲁棒性。 |
| [^4] | [Interacting Particle Langevin Algorithm for Maximum Marginal Likelihood Estimation.](http://arxiv.org/abs/2303.13429) | 本文提出了一种相互作用粒子 Langevin 算法，用于最大边缘似然估计。使用此算法，估计器的优化误差具有非渐近浓度界限。 |

# 详细

[^1]: 最小化最优和计算高效的分布鲁棒离线强化学习算法

    Minimax Optimal and Computationally Efficient Algorithms for Distributionally Robust Offline Reinforcement Learning

    [https://arxiv.org/abs/2403.09621](https://arxiv.org/abs/2403.09621)

    研究提出了最小化最优和计算高效的算法，为鲁棒离线强化学习中的函数逼近带来新颖视角，并展示了其与标准离线强化学习中函数逼近的区别。

    

    分布鲁棒离线强化学习（RL）寻求针对环境扰动的鲁棒策略训练，通过建模动态不确定性来调用函数逼近，当面对庞大的状态-动作空间时，这种RL需要考虑到动态不确定性，引入了基本的非线性和计算负担，这给分析和实际应用函数逼近提出了独特挑战。在基本设置下，提议最小化最优和计算高效的算法，实现函数逼近，并在鲁棒离线RL的背景下启动对实例相关次优性分析的研究。我们的结果揭示了鲁棒离线RL中的函数逼近本质上与标准离线RL中的函数逼近有明显区别，可能更加困难。我们的算法和理论结果至关重要地依赖于

    arXiv:2403.09621v1 Announce Type: cross  Abstract: Distributionally robust offline reinforcement learning (RL), which seeks robust policy training against environment perturbation by modeling dynamics uncertainty, calls for function approximations when facing large state-action spaces. However, the consideration of dynamics uncertainty introduces essential nonlinearity and computational burden, posing unique challenges for analyzing and practically employing function approximation. Focusing on a basic setting where the nominal model and perturbed models are linearly parameterized, we propose minimax optimal and computationally efficient algorithms realizing function approximation and initiate the study on instance-dependent suboptimality analysis in the context of robust offline RL. Our results uncover that function approximation in robust offline RL is essentially distinct from and probably harder than that in standard offline RL. Our algorithms and theoretical results crucially depen
    
[^2]: 最佳武器识别的最佳批处理算法

    Optimal Batched Best Arm Identification. (arXiv:2310.14129v1 [cs.LG])

    [http://arxiv.org/abs/2310.14129](http://arxiv.org/abs/2310.14129)

    本论文研究了最佳批处理武器识别问题，在渐近和非渐近设置中提出了Tri-BBAI和Opt-BBAI算法，分别实现了最优和几乎最优的样本和批处理复杂度。

    

    我们研究了最佳批处理武器识别（BBAI）问题，其中学习者的目标是在尽量少地更换策略的同时识别出最佳武器。具体而言，我们的目标是以概率$1-\delta$找到最佳武器，其中$\delta>0$是一个小常数，同时最小化样本复杂度（武器拉取的总数）和批处理复杂度（批处理的总数）。我们提出了三批次最佳武器识别（Tri-BBAI）算法，这是第一个在渐近设置（即$\delta\rightarrow0$）中实现最优样本复杂度且仅在最多三个批次中运行的批处理算法。基于Tri-BBAI，我们进一步提出了几乎最优的批处理最佳武器识别（Opt-BBAI）算法，在非渐近设置（即$\delta>0$是任意固定的）中实现近似最优的样本和批处理复杂度，同时在$\delta$趋于零时享受与Tri-BBAI相同的批处理和样本复杂度。

    We study the batched best arm identification (BBAI) problem, where the learner's goal is to identify the best arm while switching the policy as less as possible. In particular, we aim to find the best arm with probability $1-\delta$ for some small constant $\delta>0$ while minimizing both the sample complexity (total number of arm pulls) and the batch complexity (total number of batches). We propose the three-batch best arm identification (Tri-BBAI) algorithm, which is the first batched algorithm that achieves the optimal sample complexity in the asymptotic setting (i.e., $\delta\rightarrow 0$) and runs only in at most $3$ batches. Based on Tri-BBAI, we further propose the almost optimal batched best arm identification (Opt-BBAI) algorithm, which is the first algorithm that achieves the near-optimal sample and batch complexity in the non-asymptotic setting (i.e., $\delta>0$ is arbitrarily fixed), while enjoying the same batch and sample complexity as Tri-BBAI when $\delta$ tends to zer
    
[^3]: 通过基于分数的对抗图像生成评估鲁棒性

    Assessing Robustness via Score-Based Adversarial Image Generation. (arXiv:2310.04285v1 [cs.CV])

    [http://arxiv.org/abs/2310.04285](http://arxiv.org/abs/2310.04285)

    本论文介绍了一种基于分数的对抗生成框架（ScoreAG），可以生成超过$\ell_p$-范数约束的对抗性示例，并通过图像转换或新图像合成的方法保持图像的核心语义，大大增强了分类器的鲁棒性。

    

    大多数对抗攻击和防御都集中在小的$\ell_p$-范数约束内的扰动上。然而，$\ell_p$威胁模型无法捕捉到所有相关的保留语义的扰动，因此，鲁棒性评估的范围是有限的。在这项工作中，我们引入了基于分数的对抗生成（ScoreAG），一种利用基于分数的生成模型的进展来生成超过$\ell_p$-范数约束的对抗性示例的新的框架，称为无限制的对抗性示例，克服了它们的局限性。与传统方法不同，ScoreAG在生成逼真的对抗性示例时保持图像的核心语义，可以通过转换现有图像或完全从零开始合成新图像的方式实现。我们进一步利用ScoreAG的生成能力来净化图像，从经验上增强分类器的鲁棒性。我们的大量实证评估表明，ScoreAG与现有最先进的对抗攻击方法的性能相当。

    Most adversarial attacks and defenses focus on perturbations within small $\ell_p$-norm constraints. However, $\ell_p$ threat models cannot capture all relevant semantic-preserving perturbations, and hence, the scope of robustness evaluations is limited. In this work, we introduce Score-Based Adversarial Generation (ScoreAG), a novel framework that leverages the advancements in score-based generative models to generate adversarial examples beyond $\ell_p$-norm constraints, so-called unrestricted adversarial examples, overcoming their limitations. Unlike traditional methods, ScoreAG maintains the core semantics of images while generating realistic adversarial examples, either by transforming existing images or synthesizing new ones entirely from scratch. We further exploit the generative capability of ScoreAG to purify images, empirically enhancing the robustness of classifiers. Our extensive empirical evaluation demonstrates that ScoreAG matches the performance of state-of-the-art atta
    
[^4]: 最大边缘似然估计的相互作用粒子 Langevin 算法

    Interacting Particle Langevin Algorithm for Maximum Marginal Likelihood Estimation. (arXiv:2303.13429v1 [stat.CO])

    [http://arxiv.org/abs/2303.13429](http://arxiv.org/abs/2303.13429)

    本文提出了一种相互作用粒子 Langevin 算法，用于最大边缘似然估计。使用此算法，估计器的优化误差具有非渐近浓度界限。

    

    本文研究了一类相互作用粒子系统，用于实现潜变量模型参数的最大边缘似然估计过程。为此，我们提出了一种连续时间相互作用粒子系统，它可以被看作是在扩展的状态空间上的 Langevin漂移，其中在经典的优化中，粒子数量作为相反温度参数。使用Langevin漂移，我们证明了最大边缘似然估计器的优化误差的非渐近浓度界限，这些界限与粒子系统中的粒子数量，算法的迭代次数以及时间离散化分析的步长参数有关。

    We study a class of interacting particle systems for implementing a marginal maximum likelihood estimation (MLE) procedure to optimize over the parameters of a latent variable model. To do so, we propose a continuous-time interacting particle system which can be seen as a Langevin diffusion over an extended state space, where the number of particles acts as the inverse temperature parameter in classical settings for optimisation. Using Langevin diffusions, we prove nonasymptotic concentration bounds for the optimisation error of the maximum marginal likelihood estimator in terms of the number of particles in the particle system, the number of iterations of the algorithm, and the step-size parameter for the time discretisation analysis.
    

