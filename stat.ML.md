# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rethinking Adversarial Inverse Reinforcement Learning: From the Angles of Policy Imitation and Transferable Reward Recovery](https://arxiv.org/abs/2403.14593) | 重新思考对抗逆强化学习中的策略模仿和可转移奖励恢复，提出了一个混合框架PPO-AIRL + SAC以解决SAC算法在AIRL训练中无法全面解开奖励函数的问题。 |
| [^2] | [Not all tickets are equal and we know it: Guiding pruning with domain-specific knowledge](https://arxiv.org/abs/2403.04805) | 使用领域特定结构信息来引导修剪的方法 DASH 在学习动态基因调控网络模型时表现出色，提供了更有意义的生物学见解 |
| [^3] | [Conformalized Selective Regression](https://arxiv.org/abs/2402.16300) | 通过利用一致性预测，提供基于模型特定偏差的置信度量，以解决选择性回归中不确定性测量的方法。 |
| [^4] | [Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning](https://arxiv.org/abs/2402.15734) | 该论文提出了一种通过无监督预训练和上下文学习方法实现PDE运算符学习的高效方式，以提高数据效率并改善模型的外域性能。 |
| [^5] | [On f-Divergence Principled Domain Adaptation: An Improved Framework](https://arxiv.org/abs/2402.01887) | 本文改进了基于f-散度的无监督领域自适应（UDA）框架，引入了f-领域差异度量指标，并通过去除绝对值函数和引入缩放参数，提出了新的目标误差和样本复杂度界限，从而使得我们能够恢复以前的KL结果，将算法和理论之间的差距缩小，并通过定位技术开发了快速率的泛化界限。实验结果证明了基于f-DD的领域学习算法在流行的UDA基准测试中表现出了卓越的性能。 |
| [^6] | [Multiscale Hodge Scattering Networks for Data Analysis](https://arxiv.org/abs/2311.10270) | 提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。 |
| [^7] | [Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?.](http://arxiv.org/abs/2401.13544) | 本文介绍了一种超越概念瓶颈模型的方法，可以使黑盒模型可干预。通过基于概念的干预来影响模型的输出，并利用这种方法对黑盒模型进行微调。实验证明，微调可以提高干预的效果，并产生更好校准的预测。 |
| [^8] | [Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors.](http://arxiv.org/abs/2401.02739) | 本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。 |
| [^9] | [Modular Learning of Deep Causal Generative Models for High-dimensional Causal Inference.](http://arxiv.org/abs/2401.01426) | 本文提出了一种用于高维因果推断的模块化深度生成模型学习算法，该算法利用预训练的模型来回答由高维数据引起的因果查询。 |
| [^10] | [Covariate Shift Adaptation Robust to Density-Ratio Estimation.](http://arxiv.org/abs/2310.16638) | 该论文研究了在协变量偏移下的密度比估计的罕见问题，提出了一种适应性方法来减轻密度比估计的偏差对模型的影响。 |
| [^11] | [qPOTS: Efficient batch multiobjective Bayesian optimization via Pareto optimal Thompson sampling.](http://arxiv.org/abs/2310.15788) | 提出了一种简单但有效的多目标贝叶斯优化方法，通过Thompson采样从GP Pareto前沿中选择新的候选者，避免了繁杂的获取函数优化步骤。 |
| [^12] | [Linear Convergence of Pre-Conditioned PI Consensus Algorithm under Restricted Strong Convexity.](http://arxiv.org/abs/2310.00419) | 本文在点对点多智能体网络中提出了一种使用比例积分（PI）控制策略的预条件PI共识算法，保证了其在受限强凸函数下的线性收敛性，无需个体局部代价函数的凸性，并且通过引入局部预条件进一步加速算法。 |
| [^13] | [Laplace-Approximated Neural Additive Models: Improving Interpretability with Bayesian Inference.](http://arxiv.org/abs/2305.16905) | 本文提出了拉普拉斯逼近神经加性模型，该模型从贝叶斯角度考虑加性结构，在恢复的特征交互中提供可信区间，提供可处理的边缘似然估计，可用于执行隐式特征选择并对特征对进行排名。 |
| [^14] | [Theoretical guarantees for neural control variates in MCMC.](http://arxiv.org/abs/2304.01111) | 本文提出了一种利用神经控制变量的方差缩减方法，推导并得出了在各种遍历性假设下渐近方差的最优收敛速率。 |
| [^15] | [A first-order augmented Lagrangian method for constrained minimax optimization.](http://arxiv.org/abs/2301.02060) | 本文提出了一种一阶增广拉格朗日方法来解决约束极小极大问题，其操作复杂度为 ${\cal O}(\varepsilon^{-4}\log\varepsilon^{-1})$。 |

# 详细

[^1]: 重新思考对抗逆强化学习：从策略模仿和可转移奖励恢复的角度

    Rethinking Adversarial Inverse Reinforcement Learning: From the Angles of Policy Imitation and Transferable Reward Recovery

    [https://arxiv.org/abs/2403.14593](https://arxiv.org/abs/2403.14593)

    重新思考对抗逆强化学习中的策略模仿和可转移奖励恢复，提出了一个混合框架PPO-AIRL + SAC以解决SAC算法在AIRL训练中无法全面解开奖励函数的问题。

    

    对抗逆强化学习（AIRL）作为模仿学习中的基石方法。本文重新思考了AIRL的两个不同角度：策略模仿和可转移奖励恢复。我们从用Soft Actor-Critic（SAC）替换AIRL中的内置算法开始，以增强样本效率，这要归功于SAC的离策略形式和相对于AIRL而言可识别的马尔可夫决策过程（MDP）模型。这确实在策略模仿方面表现出显著的改进，但不慎给可转移奖励恢复带来了缺点。为了解决这个问题，我们阐述了SAC算法本身在AIRL训练过程中无法全面解开奖励函数，提出了一个混合框架，PPO-AIRL + SAC，以获得令人满意的转移效果。此外，我们分析了环境提取解开的奖励的能力。

    arXiv:2403.14593v1 Announce Type: new  Abstract: Adversarial inverse reinforcement learning (AIRL) stands as a cornerstone approach in imitation learning. This paper rethinks the two different angles of AIRL: policy imitation and transferable reward recovery. We begin with substituting the built-in algorithm in AIRL with soft actor-critic (SAC) during the policy optimization process to enhance sample efficiency, thanks to the off-policy formulation of SAC and identifiable Markov decision process (MDP) models with respect to AIRL. It indeed exhibits a significant improvement in policy imitation but accidentally brings drawbacks to transferable reward recovery. To learn this issue, we illustrate that the SAC algorithm itself is not feasible to disentangle the reward function comprehensively during the AIRL training process, and propose a hybrid framework, PPO-AIRL + SAC, for satisfactory transfer effect. Additionally, we analyze the capability of environments to extract disentangled rewa
    
[^2]: 不是所有的票据都是平等的，而我们知道：用领域特定知识来引导修剪

    Not all tickets are equal and we know it: Guiding pruning with domain-specific knowledge

    [https://arxiv.org/abs/2403.04805](https://arxiv.org/abs/2403.04805)

    使用领域特定结构信息来引导修剪的方法 DASH 在学习动态基因调控网络模型时表现出色，提供了更有意义的生物学见解

    

    神经结构学习对于科学发现和可解释性至关重要。然而，当代侧重于计算资源效率的修剪算法在选择符合领域专业知识的有意义模型方面面临算法障碍。为了减轻这一挑战，我们提出了DASH，利用可用的领域特定结构信息来引导修剪。在学习动态基因调控网络模型的背景下，我们展示了DASH与现有一般知识相结合，提供了与生物学一致的数据特定见解。对于这一任务，我们展示了在具有地面真实信息的合成数据和两个真实世界应用中，DASH的有效性，其优于竞争方法很大，并提供了更有意义的生物学见解。我们的工作表明，领域特定的结构信息具有提高模型衍生科学洞见的潜力。

    arXiv:2403.04805v1 Announce Type: new  Abstract: Neural structure learning is of paramount importance for scientific discovery and interpretability. Yet, contemporary pruning algorithms that focus on computational resource efficiency face algorithmic barriers to select a meaningful model that aligns with domain expertise. To mitigate this challenge, we propose DASH, which guides pruning by available domain-specific structural information. In the context of learning dynamic gene regulatory network models, we show that DASH combined with existing general knowledge on interaction partners provides data-specific insights aligned with biology. For this task, we show on synthetic data with ground truth information and two real world applications the effectiveness of DASH, which outperforms competing methods by a large margin and provides more meaningful biological insights. Our work shows that domain specific structural information bears the potential to improve model-derived scientific insi
    
[^3]: Conformalized Selective Regression

    Conformalized Selective Regression

    [https://arxiv.org/abs/2402.16300](https://arxiv.org/abs/2402.16300)

    通过利用一致性预测，提供基于模型特定偏差的置信度量，以解决选择性回归中不确定性测量的方法。

    

    预测模型是否总是要提供预测？在追求最大预测性能的过程中，可靠性和公平性往往被忽视，尤其是关于不确定性的作用。选择性回归，也称为“拒绝选项”，允许模型在存在相当大的不确定性情况下放弃预测。尽管7十年前就最初提出了选择性回归的方法，但大多数方法主要集中在用于测量不确定性的基于分布的代理，尤其是条件方差。但这种关注忽视了模型特定偏差对模型性能的显著影响。本文提出了一种新的选择性回归方法，通过利用一致性预测，为基于模型特定偏差的个别预测提供有根据的置信度度量。此外，我们提出了一个标准化的评估框架，以便进行恰当的比较。

    arXiv:2402.16300v1 Announce Type: new  Abstract: Should prediction models always deliver a prediction? In the pursuit of maximum predictive performance, critical considerations of reliability and fairness are often overshadowed, particularly when it comes to the role of uncertainty. Selective regression, also known as the "reject option," allows models to abstain from predictions in cases of considerable uncertainty. Initially proposed seven decades ago, approaches to selective regression have mostly focused on distribution-based proxies for measuring uncertainty, particularly conditional variance. However, this focus neglects the significant influence of model-specific biases on a model's performance. In this paper, we propose a novel approach to selective regression by leveraging conformal prediction, which provides grounded confidence measures for individual predictions based on model-specific biases. In addition, we propose a standardized evaluation framework to allow proper compar
    
[^4]: 通过无监督预训练和上下文学习实现高效的运算符学习

    Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning

    [https://arxiv.org/abs/2402.15734](https://arxiv.org/abs/2402.15734)

    该论文提出了一种通过无监督预训练和上下文学习方法实现PDE运算符学习的高效方式，以提高数据效率并改善模型的外域性能。

    

    近年来，人们见证了将机器学习方法与物理领域特定洞察力相结合，以解决基于偏微分方程（PDEs）的科学问题的潜力。然而，由于数据密集，这些方法仍然需要大量PDE数据。 这重新引入了对昂贵的数值PDE解决方案的需求，部分削弱了避免这些昂贵模拟的原始目标。 在这项工作中，为了寻求数据效率，我们设计了用于PDE运算符学习的无监督预训练和上下文学习方法。 为了减少对带有模拟解的训练数据的需求，我们使用基于重构的代理任务在未标记的PDE数据上预训练神经运算符。 为了提高超出分布性能，我们进一步帮助神经运算符灵活地利用上下文学习方法，而无需额外的训练成本或设计。 在各种PD上进行了大量实证评估

    arXiv:2402.15734v1 Announce Type: new  Abstract: Recent years have witnessed the promise of coupling machine learning methods and physical domain-specific insight for solving scientific problems based on partial differential equations (PDEs). However, being data-intensive, these methods still require a large amount of PDE data. This reintroduces the need for expensive numerical PDE solutions, partially undermining the original goal of avoiding these expensive simulations. In this work, seeking data efficiency, we design unsupervised pretraining and in-context learning methods for PDE operator learning. To reduce the need for training data with simulated solutions, we pretrain neural operators on unlabeled PDE data using reconstruction-based proxy tasks. To improve out-of-distribution performance, we further assist neural operators in flexibly leveraging in-context learning methods, without incurring extra training costs or designs. Extensive empirical evaluations on a diverse set of PD
    
[^5]: 基于f-散度原理的领域自适应：一个改进的框架

    On f-Divergence Principled Domain Adaptation: An Improved Framework

    [https://arxiv.org/abs/2402.01887](https://arxiv.org/abs/2402.01887)

    本文改进了基于f-散度的无监督领域自适应（UDA）框架，引入了f-领域差异度量指标，并通过去除绝对值函数和引入缩放参数，提出了新的目标误差和样本复杂度界限，从而使得我们能够恢复以前的KL结果，将算法和理论之间的差距缩小，并通过定位技术开发了快速率的泛化界限。实验结果证明了基于f-DD的领域学习算法在流行的UDA基准测试中表现出了卓越的性能。

    

    无监督领域自适应（UDA）在解决机器学习中的分布偏移问题中起着至关重要的作用。在本文中，我们通过改进Acuna等人（2021年）提出的UDA的理论基础，对其基于f-散度的差异度进行了改进，并引入了一个新的度量指标，即f-领域差异（f-DD）。通过去除绝对值函数并引入一个缩放参数，f-DD产生了新的目标误差和样本复杂度界限，使我们能够恢复以前基于KL的结果，并弥合了Acuna等人（2021年）中提出的算法和理论之间的差距。利用定位技术，我们还开发了一种快速率的泛化界限。实证结果表明，在流行的UDA基准测试中，基于f-DD的领域学习算法表现出优越性能。

    Unsupervised domain adaptation (UDA) plays a crucial role in addressing distribution shifts in machine learning. In this work, we improve the theoretical foundations of UDA proposed by Acuna et al. (2021) by refining their f-divergence-based discrepancy and additionally introducing a new measure, f-domain discrepancy (f-DD). By removing the absolute value function and incorporating a scaling parameter, f-DD yields novel target error and sample complexity bounds, allowing us to recover previous KL-based results and bridging the gap between algorithms and theory presented in Acuna et al. (2021). Leveraging a localization technique, we also develop a fast-rate generalization bound. Empirical results demonstrate the superior performance of f-DD-based domain learning algorithms over previous works in popular UDA benchmarks.
    
[^6]: 用于数据分析的多尺度霍奇散射网络

    Multiscale Hodge Scattering Networks for Data Analysis

    [https://arxiv.org/abs/2311.10270](https://arxiv.org/abs/2311.10270)

    提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。

    

    我们提出了一种新的散射网络，用于在单纯复合仿射上测量的信号，称为\emph{多尺度霍奇散射网络}（MHSNs）。我们的构造基于单纯复合仿射上的多尺度基础词典，即$\kappa$-GHWT和$\kappa$-HGLET，我们最近为给定单纯复合仿射中的维度$\kappa \in \mathbb{N}$推广了基于节点的广义哈-沃什变换（GHWT）和分层图拉普拉斯特征变换（HGLET）。$\kappa$-GHWT和$\kappa$-HGLET都形成冗余集合（即词典）的多尺度基础向量和给定信号的相应扩展系数。我们的MHSNs使用类似于卷积神经网络（CNN）的分层结构来级联词典系数模的矩。所得特征对单纯复合仿射的重新排序不变（即节点排列的置换

    arXiv:2311.10270v2 Announce Type: replace  Abstract: We propose new scattering networks for signals measured on simplicial complexes, which we call \emph{Multiscale Hodge Scattering Networks} (MHSNs). Our construction is based on multiscale basis dictionaries on simplicial complexes, i.e., the $\kappa$-GHWT and $\kappa$-HGLET, which we recently developed for simplices of dimension $\kappa \in \mathbb{N}$ in a given simplicial complex by generalizing the node-based Generalized Haar-Walsh Transform (GHWT) and Hierarchical Graph Laplacian Eigen Transform (HGLET). The $\kappa$-GHWT and the $\kappa$-HGLET both form redundant sets (i.e., dictionaries) of multiscale basis vectors and the corresponding expansion coefficients of a given signal. Our MHSNs use a layered structure analogous to a convolutional neural network (CNN) to cascade the moments of the modulus of the dictionary coefficients. The resulting features are invariant to reordering of the simplices (i.e., node permutation of the u
    
[^7]: 超越概念瓶颈模型：如何使黑盒模型可干预？

    Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?. (arXiv:2401.13544v1 [cs.LG])

    [http://arxiv.org/abs/2401.13544](http://arxiv.org/abs/2401.13544)

    本文介绍了一种超越概念瓶颈模型的方法，可以使黑盒模型可干预。通过基于概念的干预来影响模型的输出，并利用这种方法对黑盒模型进行微调。实验证明，微调可以提高干预的效果，并产生更好校准的预测。

    

    最近，可解释的机器学习重新探索了概念瓶颈模型（CBM），包括从原始特征中逐步预测高级概念和从预测的概念中预测目标变量。这个模型类别的一个引人注目的优势是用户能够对预测的概念值进行干预，从而影响模型的下游输出。在这项工作中，我们介绍了一种方法，在已经训练好但本质上不可解释的神经网络上进行基于概念的干预，给定一个带有注释的验证集。此外，我们将模型的可干预性定义为基于概念干预的有效性的度量，并利用这个定义来对黑盒模型进行微调。实证上，我们探索了合成表格数据和自然图像基准上黑盒分类器的干预性。我们证明，微调提高了干预的效果，并经常产生更好校准的预测。

    Recently, interpretable machine learning has re-explored concept bottleneck models (CBM), comprising step-by-step prediction of the high-level concepts from the raw features and the target variable from the predicted concepts. A compelling advantage of this model class is the user's ability to intervene on the predicted concept values, affecting the model's downstream output. In this work, we introduce a method to perform such concept-based interventions on already-trained neural networks, which are not interpretable by design, given an annotated validation set. Furthermore, we formalise the model's intervenability as a measure of the effectiveness of concept-based interventions and leverage this definition to fine-tune black-box models. Empirically, we explore the intervenability of black-box classifiers on synthetic tabular and natural image benchmarks. We demonstrate that fine-tuning improves intervention effectiveness and often yields better-calibrated predictions. To showcase the 
    
[^8]: 扩散变分推断：扩散模型作为表达性变分后验

    Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors. (arXiv:2401.02739v1 [cs.LG])

    [http://arxiv.org/abs/2401.02739](http://arxiv.org/abs/2401.02739)

    本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。

    

    我们提出了去噪扩散变分推断（DDVI），一种用扩散模型作为表达性变分后验的潜变量模型的近似推断算法。我们的方法通过辅助潜变量增加了变分后验，从而得到一个表达性的模型类，通过反转用户指定的加噪过程在潜空间中进行扩散。我们通过优化一个受到觉醒-睡眠算法启发的边际似然新下界来拟合这些模型。我们的方法易于实现（它适配了正则化的ELBO扩展），与黑盒变分推断兼容，并且表现优于基于归一化流或对抗网络的替代近似后验类别。将我们的方法应用于深度潜变量模型时，我们的方法得到了去噪扩散变分自动编码器（DD-VAE）算法。我们将该算法应用于生物学中的一个激励任务 -- 从人类基因组中推断潜在血统 -- 超过了强基线模型。

    We propose denoising diffusion variational inference (DDVI), an approximate inference algorithm for latent variable models which relies on diffusion models as expressive variational posteriors. Our method augments variational posteriors with auxiliary latents, which yields an expressive class of models that perform diffusion in latent space by reversing a user-specified noising process. We fit these models by optimizing a novel lower bound on the marginal likelihood inspired by the wake-sleep algorithm. Our method is easy to implement (it fits a regularized extension of the ELBO), is compatible with black-box variational inference, and outperforms alternative classes of approximate posteriors based on normalizing flows or adversarial networks. When applied to deep latent variable models, our method yields the denoising diffusion VAE (DD-VAE) algorithm. We use this algorithm on a motivating task in biology -- inferring latent ancestry from human genomes -- outperforming strong baselines
    
[^9]: 高维因果推断的模块化深度生成模型学习

    Modular Learning of Deep Causal Generative Models for High-dimensional Causal Inference. (arXiv:2401.01426v1 [cs.LG])

    [http://arxiv.org/abs/2401.01426](http://arxiv.org/abs/2401.01426)

    本文提出了一种用于高维因果推断的模块化深度生成模型学习算法，该算法利用预训练的模型来回答由高维数据引起的因果查询。

    

    Pearl的因果层次结构在观测、干预和反事实问题之间建立了明确的分离。研究人员提出了计算可辨识因果查询的声音和完整算法，在给定层次的因果结构和数据的情况下使用较低层次的层次的数据。然而，大多数这些算法假设我们可以准确估计数据的概率分布，这对于如图像这样的高维变量是一个不切实际的假设。另一方面，现代生成式深度学习架构可以被训练来学习如何准确地从这样的高维分布中采样。特别是随着图像基模型的最近兴起，利用预训练模型来回答带有这样高维数据的因果查询是非常有吸引力的。为了解决这个问题，我们提出了一个顺序训练算法，给定因果结构和预训练的条件生成模型，可以训练一个模型来估计由高维数据引起的因果关系。

    Pearl's causal hierarchy establishes a clear separation between observational, interventional, and counterfactual questions. Researchers proposed sound and complete algorithms to compute identifiable causal queries at a given level of the hierarchy using the causal structure and data from the lower levels of the hierarchy. However, most of these algorithms assume that we can accurately estimate the probability distribution of the data, which is an impractical assumption for high-dimensional variables such as images. On the other hand, modern generative deep learning architectures can be trained to learn how to accurately sample from such high-dimensional distributions. Especially with the recent rise of foundation models for images, it is desirable to leverage pre-trained models to answer causal queries with such high-dimensional data. To address this, we propose a sequential training algorithm that, given the causal structure and a pre-trained conditional generative model, can train a
    
[^10]: 适应密度比估计的协变量偏移适应

    Covariate Shift Adaptation Robust to Density-Ratio Estimation. (arXiv:2310.16638v1 [stat.ME])

    [http://arxiv.org/abs/2310.16638](http://arxiv.org/abs/2310.16638)

    该论文研究了在协变量偏移下的密度比估计的罕见问题，提出了一种适应性方法来减轻密度比估计的偏差对模型的影响。

    

    在一种情况下，我们可以访问具有协变量和结果的训练数据，而测试数据只包含协变量。在这种情况下，我们的主要目标是预测测试数据中缺失的结果。为了实现这个目标，我们在协变量偏移下训练参数回归模型，其中训练数据和测试数据之间的协变量分布不同。对于这个问题，现有研究提出了通过使用密度比的重要性加权来进行协变量偏移适应的方法。该方法通过对训练数据损失进行加权平均，每个权重是训练数据和测试数据之间的协变量密度比的估计，以近似测试数据的风险。尽管它允许我们获得一个最小化测试数据风险的模型，但其性能严重依赖于密度比估计的准确性。此外，即使密度比可以一致地估计，密度比的估计误差也会导致回归模型的估计器产生偏差。

    Consider a scenario where we have access to train data with both covariates and outcomes while test data only contains covariates. In this scenario, our primary aim is to predict the missing outcomes of the test data. With this objective in mind, we train parametric regression models under a covariate shift, where covariate distributions are different between the train and test data. For this problem, existing studies have proposed covariate shift adaptation via importance weighting using the density ratio. This approach averages the train data losses, each weighted by an estimated ratio of the covariate densities between the train and test data, to approximate the test-data risk. Although it allows us to obtain a test-data risk minimizer, its performance heavily relies on the accuracy of the density ratio estimation. Moreover, even if the density ratio can be consistently estimated, the estimation errors of the density ratio also yield bias in the estimators of the regression model's 
    
[^11]: qPOTS: 高效的批量多目标贝叶斯优化算法

    qPOTS: Efficient batch multiobjective Bayesian optimization via Pareto optimal Thompson sampling. (arXiv:2310.15788v1 [math.OC])

    [http://arxiv.org/abs/2310.15788](http://arxiv.org/abs/2310.15788)

    提出了一种简单但有效的多目标贝叶斯优化方法，通过Thompson采样从GP Pareto前沿中选择新的候选者，避免了繁杂的获取函数优化步骤。

    

    传统进化方法在多目标优化中非常有效，但对目标进行大量查询可能不利于目标花费很多或者计算量很大的时候。用高斯过程（GP）替代物和贝叶斯优化（BO）来解决多目标优化是一种高效的方法。多目标贝叶斯优化(MOBO)涉及构建一个被优化用来获得新观察候选的获取函数。这个“内部”优化可能很困难，因为获取函数是非凸的，不可微的和/或者不出波，MOBO的成功在很大程度上依赖于这个内部优化。我们摒弃这个困难的获取函数优化步骤，提出一种简单但有效的基于Thompson采样的方法($q\texttt{POTS}$)，其中新的候选者是从通过求解一个更便宜的多个后验样本路径的GP Pareto前沿中选择的。

    Classical evolutionary approaches for multiobjective optimization are quite effective but incur a lot of queries to the objectives; this can be prohibitive when objectives are expensive oracles. A sample-efficient approach to solving multiobjective optimization is via Gaussian process (GP) surrogates and Bayesian optimization (BO). Multiobjective Bayesian optimization (MOBO) involves the construction of an acquisition function which is optimized to acquire new observation candidates. This ``inner'' optimization can be hard due to various reasons: acquisition functions being nonconvex, nondifferentiable and/or unavailable in analytical form; the success of MOBO heavily relies on this inner optimization. We do away with this hard acquisition function optimization step and propose a simple, but effective, Thompson sampling based approach ($q\texttt{POTS}$) where new candidate(s) are chosen from the Pareto frontier of random GP posterior sample paths obtained by solving a much cheaper mult
    
[^12]: 受限强凸性下的预条件PI共识算法的线性收敛性

    Linear Convergence of Pre-Conditioned PI Consensus Algorithm under Restricted Strong Convexity. (arXiv:2310.00419v1 [math.OC])

    [http://arxiv.org/abs/2310.00419](http://arxiv.org/abs/2310.00419)

    本文在点对点多智能体网络中提出了一种使用比例积分（PI）控制策略的预条件PI共识算法，保证了其在受限强凸函数下的线性收敛性，无需个体局部代价函数的凸性，并且通过引入局部预条件进一步加速算法。

    

    本文考虑在点对点多智能体网络中解决分布式凸优化问题。网络被假定为同步和连通的。采用比例积分（PI）控制策略，开发了多种具有固定步长的算法，其中最早的是PI共识算法。利用李雅普诺夫理论，我们首次保证了具有速率匹配离散化的受限强凸函数的PI共识算法的指数收敛性，而不需要个体局部代价函数的凸性。为了加速PI共识算法，我们采用了局部预条件的形式，即常数正定矩阵，并通过数值验证其相比于突出的分布式凸优化算法的效率。

    This paper considers solving distributed convex optimization problems in peer-to-peer multi-agent networks. The network is assumed to be synchronous and connected. By using the proportional-integral (PI) control strategy, various algorithms with fixed stepsize have been developed. The earliest among them is the PI consensus algorithm. Using Lyapunov theory, we guarantee exponential convergence of the PI consensus algorithm for restricted strongly convex functions with rate-matching discretization, without requiring convexity of individual local cost functions, for the first time. In order to accelerate the PI consensus algorithm, we incorporate local pre-conditioning in the form of constant positive definite matrices and numerically validate its efficiency compared to the prominent distributed convex optimization algorithms. Unlike classical pre-conditioning, where only the gradients are multiplied by a pre-conditioner, the proposed pre-conditioning modifies both the gradients and the 
    
[^13]: 拉普拉斯逼近神经加性模型：贝叶斯推理提高解释性

    Laplace-Approximated Neural Additive Models: Improving Interpretability with Bayesian Inference. (arXiv:2305.16905v1 [stat.ML])

    [http://arxiv.org/abs/2305.16905](http://arxiv.org/abs/2305.16905)

    本文提出了拉普拉斯逼近神经加性模型，该模型从贝叶斯角度考虑加性结构，在恢复的特征交互中提供可信区间，提供可处理的边缘似然估计，可用于执行隐式特征选择并对特征对进行排名。

    

    深度神经网络（DNN）在许多领域取得了成功应用，但它们的黑盒性质阻碍了解释性。神经加性模型（NAM）解决了这个问题，将网络分为加性子网络，从而使输入特征和预测之间的交互变得明显。在本文中，我们从贝叶斯角度考虑加性结构，并开发了一个实用的拉普拉斯逼近方法。这种方法在以下三个方面提高了可解释性：a）它通过估计子网络的函数空间不确定性为恢复的特征交互提供可信区间；b）它提供可处理的边缘似然估计，可用于通过经验贝叶斯过程执行特征的隐式选择；c）它可用于对特征对进行排名，作为精细调整的交互模型候选。我们在几个基准数据集上实证表明，我们提出的拉普拉斯逼近神经加性模型（LA-NAM）提高了NAM模型的可解释性，并进一步揭示了学习到的子网络的交互结构。

    Deep neural networks (DNNs) have found successful applications in many fields, but their black-box nature hinders interpretability. This is addressed by the neural additive model (NAM), in which the network is divided into additive sub-networks, thus making apparent the interaction between input features and predictions. In this paper, we approach the additive structure from a Bayesian perspective and develop a practical Laplace approximation. This enhances interpretability in three primary ways: a) It provides credible intervals for the recovered feature interactions by estimating function-space uncertainty of the sub-networks; b) it yields a tractable estimate of the marginal likelihood, which can be used to perform an implicit selection of features through an empirical Bayes procedure; and c) it can be used to rank feature pairs as candidates for second-order interactions in fine-tuned interaction models. We show empirically that our proposed Laplace-approximated NAM (LA-NAM) improv
    
[^14]: 神经控制变量在MCMC中的理论保证

    Theoretical guarantees for neural control variates in MCMC. (arXiv:2304.01111v1 [math.ST])

    [http://arxiv.org/abs/2304.01111](http://arxiv.org/abs/2304.01111)

    本文提出了一种利用神经控制变量的方差缩减方法，推导并得出了在各种遍历性假设下渐近方差的最优收敛速率。

    

    本文提出了一种基于加性控制变量和最小化渐近方差的马尔可夫链方差缩减方法。我们专注于控制变量表示为深度神经网络的特定情况。在基础马尔可夫链的各种遍历性假设下，推导了渐近方差的最优收敛速率。该方法依赖于方差缩减算法和函数逼近理论的随机误差的最新成果。

    In this paper, we propose a variance reduction approach for Markov chains based on additive control variates and the minimization of an appropriate estimate for the asymptotic variance. We focus on the particular case when control variates are represented as deep neural networks. We derive the optimal convergence rate of the asymptotic variance under various ergodicity assumptions on the underlying Markov chain. The proposed approach relies upon recent results on the stochastic errors of variance reduction algorithms and function approximation theory.
    
[^15]: 一种用于约束极小极大优化问题的一阶增广拉格朗日方法

    A first-order augmented Lagrangian method for constrained minimax optimization. (arXiv:2301.02060v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2301.02060](http://arxiv.org/abs/2301.02060)

    本文提出了一种一阶增广拉格朗日方法来解决约束极小极大问题，其操作复杂度为 ${\cal O}(\varepsilon^{-4}\log\varepsilon^{-1})$。

    

    本文研究了一类约束极小极大问题。特别地，我们提出了一种一阶增广拉格朗日方法来解决这些问题，其子问题被发现是一个更简单的结构化极小极大问题，并且可以通过作者在 [26] 中最近开发的一阶方法来适当地解决。在一些适当的假设下，为了找到约束极小极大问题的一个 $\varepsilon$-KKT 解，该方法的操作复杂度为 ${\cal O}(\varepsilon^{-4}\log\varepsilon^{-1})$，该复杂度是由基本操作测量得到的。

    In this paper we study a class of constrained minimax problems. In particular, we propose a first-order augmented Lagrangian method for solving them, whose subproblems turn out to be a much simpler structured minimax problem and are suitably solved by a first-order method recently developed in [26] by the authors. Under some suitable assumptions, an \emph{operation complexity} of ${\cal O}(\varepsilon^{-4}\log\varepsilon^{-1})$, measured by its fundamental operations, is established for the first-order augmented Lagrangian method for finding an $\varepsilon$-KKT solution of the constrained minimax problems.
    

