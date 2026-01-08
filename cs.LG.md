# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Anytime Algorithm for Good Arm Identification.](http://arxiv.org/abs/2310.10359) | 提出了一个适用于随机贝叶斯臂机的随时和无参数采样规则APGAI，通过自适应策略提高了好臂识别效率，在固定置信度和固定预算的情况下有良好实验表现。 |
| [^2] | [Tipping Point Forecasting in Non-Stationary Dynamics on Function Spaces.](http://arxiv.org/abs/2308.08794) | 本文提出了一种利用循环神经算子学习非平稳动力系统演化的方法，并且通过基于不确定性的方法检测未来的翻车点。同时，我们还提出了一种符合预测框架，通过监测与物理约束的偏离来预测翻车点，从而使得预测结果具有严格的不确定性度量。 |
| [^3] | [InstructBio: A Large-scale Semi-supervised Learning Paradigm for Biochemical Problems.](http://arxiv.org/abs/2304.03906) | InstructBio是一种针对生物化学问题的大规模半监督学习算法，引入教练模型提供有效的置信度比率来指导目标模型对不同数据点给予明显关注，避免依赖有限的标记数据和不正确的伪注释，提高了分子模型的泛化能力。 |
| [^4] | [Inference in conditioned dynamics through causality restoration.](http://arxiv.org/abs/2210.10179) | 本文提出了一种经由因果性恢复的方式来产生条件分布下的独立样本。 |
| [^5] | [Practitioner Motives to Select Hyperparameter Optimization Methods.](http://arxiv.org/abs/2203.01717) | 研究探讨了机器学习从业者选择超参数优化方法的动机，结果表明这基于个人目标和背景因素，调查还给出了优化模型的六个主要目标。 |

# 详细

[^1]: 一个适用于好臂识别的随时算法

    An Anytime Algorithm for Good Arm Identification. (arXiv:2310.10359v1 [stat.ML])

    [http://arxiv.org/abs/2310.10359](http://arxiv.org/abs/2310.10359)

    提出了一个适用于随机贝叶斯臂机的随时和无参数采样规则APGAI，通过自适应策略提高了好臂识别效率，在固定置信度和固定预算的情况下有良好实验表现。

    

    在好臂识别（GAI）中，目标是识别其中一个平均性能超过给定阈值的臂，称为好臂（如果存在）。目前很少有研究在固定预算的情况下进行GAI，即在先确定好预算之后，或者在任何时刻都可以要求推荐的随时设置下进行GAI。我们提出了一种名为APGAI的随时和无参数采样规则，用于随机贝叶斯臂机。APGAI可以直接用于固定置信度和固定预算的设定中。首先，我们得出其任何时刻的误差概率的上界。这些上界表明，自适应策略在检测没有好臂的时候比均匀采样更高效。其次，当APGAI与一个停止规则结合时，我们证明了在任何置信水平下的预期采样复杂性的上界。最后，我们展示了APGAI在合成数据和真实世界数据上的良好实验性能。我们的工作为所有设置中的GAI问题提供了一个广泛的概述。

    In good arm identification (GAI), the goal is to identify one arm whose average performance exceeds a given threshold, referred to as good arm, if it exists. Few works have studied GAI in the fixed-budget setting, when the sampling budget is fixed beforehand, or the anytime setting, when a recommendation can be asked at any time. We propose APGAI, an anytime and parameter-free sampling rule for GAI in stochastic bandits. APGAI can be straightforwardly used in fixed-confidence and fixed-budget settings. First, we derive upper bounds on its probability of error at any time. They show that adaptive strategies are more efficient in detecting the absence of good arms than uniform sampling. Second, when APGAI is combined with a stopping rule, we prove upper bounds on the expected sampling complexity, holding at any confidence level. Finally, we show good empirical performance of APGAI on synthetic and real-world data. Our work offers an extensive overview of the GAI problem in all settings.
    
[^2]: 功能空间中非平稳动力学中的翻车点预测

    Tipping Point Forecasting in Non-Stationary Dynamics on Function Spaces. (arXiv:2308.08794v1 [cs.LG])

    [http://arxiv.org/abs/2308.08794](http://arxiv.org/abs/2308.08794)

    本文提出了一种利用循环神经算子学习非平稳动力系统演化的方法，并且通过基于不确定性的方法检测未来的翻车点。同时，我们还提出了一种符合预测框架，通过监测与物理约束的偏离来预测翻车点，从而使得预测结果具有严格的不确定性度量。

    

    翻车点是非平稳和混沌动力系统演化中的突变、剧烈且常常不可逆的变化。例如，预计温室气体浓度的增加会导致低云覆盖的急剧减少，被称为气候学的翻车点。在本文中，我们利用一种新颖的循环神经算子（RNO）学习这种非平稳动力系统的演化，RNO可以学习函数空间之间的映射关系。在仅训练RNO在翻车点之前的动力学数据之后，我们采用基于不确定性的方法来检测未来的翻车点。具体而言，我们提出了一个符合预测框架，通过监测与物理约束（如守恒量和偏微分方程）偏离来预测翻车点，从而使得对这些突变的预测伴随着一种严格的不确定性度量。我们将我们提出的方法应用于非平稳常微分方程和偏微分方程的案例。

    Tipping points are abrupt, drastic, and often irreversible changes in the evolution of non-stationary and chaotic dynamical systems. For instance, increased greenhouse gas concentrations are predicted to lead to drastic decreases in low cloud cover, referred to as a climatological tipping point. In this paper, we learn the evolution of such non-stationary dynamical systems using a novel recurrent neural operator (RNO), which learns mappings between function spaces. After training RNO on only the pre-tipping dynamics, we employ it to detect future tipping points using an uncertainty-based approach. In particular, we propose a conformal prediction framework to forecast tipping points by monitoring deviations from physics constraints (such as conserved quantities and partial differential equations), enabling forecasting of these abrupt changes along with a rigorous measure of uncertainty. We illustrate our proposed methodology on non-stationary ordinary and partial differential equations,
    
[^3]: InstructBio：一种针对生物化学问题的大规模半监督学习范式。

    InstructBio: A Large-scale Semi-supervised Learning Paradigm for Biochemical Problems. (arXiv:2304.03906v1 [cs.LG])

    [http://arxiv.org/abs/2304.03906](http://arxiv.org/abs/2304.03906)

    InstructBio是一种针对生物化学问题的大规模半监督学习算法，引入教练模型提供有效的置信度比率来指导目标模型对不同数据点给予明显关注，避免依赖有限的标记数据和不正确的伪注释，提高了分子模型的泛化能力。

    

    在科学人工智能领域，面对真实世界问题中的有限标记数据始终是一个重要的挑战。目前的方法是在大型未标记语料库上预训练强力的任务无关模型，但在向下游任务转移知识方面可能存在困难。在本研究中，我们提出了InstructBio，一种半监督学习算法，更好地利用未标记的样例。它引入教练模型来提供伪标签可靠性的置信度比率。这些置信度分数然后指导目标模型对不同的数据点给予明显的关注，避免对标记数据的过度依赖以及不正确的伪注释的负面影响。全面的实验表明，InstructBio显著提高了分子模型的泛化能力，不仅在分子属性预测方面，在活性悬崖估计方面也表现出优越性。

    In the field of artificial intelligence for science, it is consistently an essential challenge to face a limited amount of labeled data for real-world problems. The prevailing approach is to pretrain a powerful task-agnostic model on a large unlabeled corpus but may struggle to transfer knowledge to downstream tasks. In this study, we propose InstructMol, a semi-supervised learning algorithm, to take better advantage of unlabeled examples. It introduces an instructor model to provide the confidence ratios as the measurement of pseudo-labels' reliability. These confidence scores then guide the target model to pay distinct attention to different data points, avoiding the over-reliance on labeled data and the negative influence of incorrect pseudo-annotations. Comprehensive experiments show that InstructBio substantially improves the generalization ability of molecular models, in not only molecular property predictions but also activity cliff estimations, demonstrating the superiority of 
    
[^4]: 经由因果性恢复在条件动力学中进行推断

    Inference in conditioned dynamics through causality restoration. (arXiv:2210.10179v2 [physics.data-an] UPDATED)

    [http://arxiv.org/abs/2210.10179](http://arxiv.org/abs/2210.10179)

    本文提出了一种经由因果性恢复的方式来产生条件分布下的独立样本。

    

    从有条件的动力学中计算可观测量通常是计算上困难的，因为虽然从非条件的动力学中高效地获取独立样本通常是可行的，但通常必须丢弃大部分样本(以一种重要性抽样的形式)因为它们不满足所施加的条件。直接从有条件的分布中抽样是不易的，因为条件打破了动力学的因果特性，最终使抽样过程变得低效。一种标准的方法是通过Metropolis Monte-Carlo过程实现，但这个过程通常很慢，需要大量的Monte-Carlo步骤来获得少量的统计独立样本。我们提出了一种替代方法，用于从有条件的分布中产生独立的样本。该方法学习一个广义动力学模型的参数，该模型最优地描述了条件分布的变分。

    Computing observables from conditioned dynamics is typically computationally hard, because, although obtaining independent samples efficiently from the unconditioned dynamics is usually feasible, generally most of the samples must be discarded (in a form of importance sampling) because they do not satisfy the imposed conditions. Sampling directly from the conditioned distribution is non-trivial, as conditioning breaks the causal properties of the dynamics which ultimately renders the sampling procedure efficient. One standard way of achieving it is through a Metropolis Monte-Carlo procedure, but this procedure is normally slow and a very large number of Monte-Carlo steps is needed to obtain a small number of statistically independent samples. In this work, we propose an alternative method to produce independent samples from a conditioned distribution. The method learns the parameters of a generalized dynamical model that optimally describe the conditioned distribution in a variational 
    
[^5]: 选择超参数优化方法的从业者动机

    Practitioner Motives to Select Hyperparameter Optimization Methods. (arXiv:2203.01717v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2203.01717](http://arxiv.org/abs/2203.01717)

    研究探讨了机器学习从业者选择超参数优化方法的动机，结果表明这基于个人目标和背景因素，调查还给出了优化模型的六个主要目标。

    

    先进的编程超参数优化方法，如贝叶斯优化，具有高样本效率，能够可靠地找到机器学习模型的最佳超参数值。然而，机器学习从业者经常应用样本效率较低的HPO方法，如网格搜索，这通常导致机器学习模型未经优化。我们怀疑，从业者选择HPO方法的原因基于个人动机，包括背景因素和个人目标。然而，从业者的动机仍然需要澄清，这妨碍了评估HPO方法以实现特定目标和以用户为中心的HPO工具的开发。为了了解从业者使用特定HPO方法的动机，我们采用混合方法，包括20个半结构化访谈和一项调查研究，共有71名机器学习专家参与，以收集访谈结果的外部有效性的证据。通过设置六个主要目标（例如，改进模型理解），

    Advanced programmatic hyperparameter optimization (HPO) methods, such as Bayesian optimization, have high sample efficiency in reproducibly finding optimal hyperparameter values of machine learning (ML) models. Yet, ML practitioners often apply less sample-efficient HPO methods, such as grid search, which often results in under-optimized ML models. As a reason for this behavior, we suspect practitioners choose HPO methods based on individual motives, consisting of contextual factors and individual goals. However, practitioners' motives still need to be clarified, hindering the evaluation of HPO methods for achieving specific goals and the user-centered development of HPO tools. To understand practitioners' motives for using specific HPO methods, we used a mixed-methods approach involving 20 semi-structured interviews and a survey study with 71 ML experts to gather evidence of the external validity of the interview results. By presenting six main goals (e.g., improving model understandi
    

