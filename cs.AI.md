# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration from Deliberation Allocation](https://arxiv.org/abs/2606.26502) | 论文发现，在相同问题上，人类答错时花更少时间（放弃），而大型推理模型答错时花更多令牌（坚持），揭示了人类与AI在失败时资源分配策略的根本差异。 |
| [^2] | [Unbiased Canonical Set-Valued Oracles Via Lattice Theory](https://arxiv.org/abs/2606.26418) | 本文通过Knaster–Tarski不动点定理，在完备格框架下提出了一种规范的非平凡credal集，解决了自指预言机在无偏和自洽约束下的唯一性问题。 |
| [^3] | [Explainable Machine Learning-Based Security and Privacy Protection Framework for Internet of Medical Things Systems](https://arxiv.org/abs/2403.09752) | 该论文提出了面向互联网医疗物联网系统的可解释机器学习安全与隐私保护框架，旨在解决IoMT系统面临的安全挑战，包括数据敏感性、恶意攻击和异常检测。 |
| [^4] | [Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization.](http://arxiv.org/abs/2307.10053) | 本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。 |

# 详细

[^1]: 人类放弃，推理模型坚持：区分难度登记与深思分配

    Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration from Deliberation Allocation

    [https://arxiv.org/abs/2606.26502](https://arxiv.org/abs/2606.26502)

    论文发现，在相同问题上，人类答错时花更少时间（放弃），而大型推理模型答错时花更多令牌（坚持），揭示了人类与AI在失败时资源分配策略的根本差异。

    

    arXiv:2606.26502v1 公告类型：新 摘要：大型推理模型（LRMs）在更困难的问题上花费更长时间，就像人类一样。这种表面上的相似性掩盖了项目内部的相反模式。当LRM答错一个问题时，它花费的令牌数比答对同一个问题时更多；而人类则相反，在答错的试验上花费的时间更少。我们将深思的两个层面分开：反应时间如何跨项目追踪难度（登记），以及在项目身份固定的情况下，智能体是在自己的失败还是成功上花费更多（分配）。在一个公开的人-LRM匹配语料库上，人类和所有五种思维LRM都再现了已知的跨项目一致性（登记），但在项目内部分配中出现了分歧：每个LRM都显示出显著的“答错 vs 答对”效应（在H-ARC上Cohen's d = 1.47-3.13），而人类则显示出相反的符号。比较是在每个智能体自身的尺度内进行的；我们从未将秒和令牌放在同一轴上。在项目固定的情况下，这种分离依然成立。

    arXiv:2606.26502v1 Announce Type: new  Abstract: Large reasoning models (LRMs) take longer on harder problems, just as humans do. This surface similarity hides an opposite pattern within items. When an LRM gets a problem wrong, it spends more tokens than when it gets the same problem right; humans do the reverse, spending less time on the trials they get wrong. We separate two levels of deliberation: how response time tracks difficulty across items (registration), and, with item identity held fixed, whether an agent spends more on its own failures or successes (allocation). On a public matched human-LRM corpus, humans and all five thinking LRMs reproduce the known cross-item alignment (registration) but diverge within items (allocation): every LRM shows a large wrong-vs-right effect (Cohen's d = 1.47-3.13 on H-ARC) while humans show the opposite sign. The comparison stays inside each agent's own scale; we never put seconds and tokens on one axis. The dissociation holds under item fixed
    
[^2]: 基于格理论的无偏规范集值预言机

    Unbiased Canonical Set-Valued Oracles Via Lattice Theory

    [https://arxiv.org/abs/2606.26418](https://arxiv.org/abs/2606.26418)

    本文通过Knaster–Tarski不动点定理，在完备格框架下提出了一种规范的非平凡credal集，解决了自指预言机在无偏和自洽约束下的唯一性问题。

    

    非智能体“预言机”AI在估计未来事件概率时面临自指问题：一旦其答案被学习并采取行动，就会改变它被要求报告的概率本身。针对科学家AI计划所倡导的一种回应是只询问反事实问题，并假设答案没有影响进行评估。我们观察到，这类答案一旦被学习就会变得无关紧要，恰恰是因为其前提随后变为假。因此，我们探索了一种自指替代方案：预言机报告的不是单一概率，而是一个同时无偏且与学习后果自洽的credal集。朴素的自洽性要求被太多集合满足（包括无用的答案[0,1]），因此问题在于挑选出一个规范的、非平凡的成员。我们通过闭包完备格上的Knaster–Tarski不动点定理实现了这一点。

    arXiv:2606.26418v1 Announce Type: new  Abstract: A non-agentic "oracle" AI that estimates probabilities of future events faces a self-reference problem: once its answer is learned and acted upon, it can change the very probability it was asked to report. One response, advocated for the Scientist AI programme, is to ask only counterfactual questions, evaluated as if the answer had no influence. We observe that such answers tend to become irrelevant the moment they are learned, precisely because their premise is then false. We therefore explore a self-referential alternative in which the oracle reports not a single probability but a credal set that is simultaneously unbiased and self-consistent with the consequences of being learned. The naive self-consistency requirement is satisfied by too many sets (including the useless answer $[0,1]$), so the problem is to single out a canonical, nontrivial member. We do so with the Knaster--Tarski fixed-point theorem on the complete lattice of clos
    
[^3]: 面向IoMT系统的可解释机器学习安全与隐私保护框架

    Explainable Machine Learning-Based Security and Privacy Protection Framework for Internet of Medical Things Systems

    [https://arxiv.org/abs/2403.09752](https://arxiv.org/abs/2403.09752)

    该论文提出了面向互联网医疗物联网系统的可解释机器学习安全与隐私保护框架，旨在解决IoMT系统面临的安全挑战，包括数据敏感性、恶意攻击和异常检测。

    

    互联网医疗物联网（IoMT）跨越了传统医疗边界，实现了从被动治疗向主动预防的过渡。这种创新方法通过实时健康数据收集实现早期疾病检测和个性化护理，特别在慢性病管理方面，IoMT可以自动化治疗。然而，由于处理数据的敏感性和价值，IoMT面临着严重的安全挑战，这会威胁到其用户的生命，因此吸引了恶意利益。此外，利用无线通信进行数据传输会使医疗数据暴露于被网络犯罪分子截获和篡改的风险之下。此外，由于人为错误、网络干扰或硬件故障，可能会出现异常。在这种背景下，基于机器学习（ML）的异常检测是一个有趣的解决方案，但它再次出现。

    arXiv:2403.09752v1 Announce Type: cross  Abstract: The Internet of Medical Things (IoMT) transcends traditional medical boundaries, enabling a transition from reactive treatment to proactive prevention. This innovative method revolutionizes healthcare by facilitating early disease detection and tailored care, particularly in chronic disease management, where IoMT automates treatments based on real-time health data collection. Nonetheless, its benefits are countered by significant security challenges that endanger the lives of its users due to the sensitivity and value of the processed data, thereby attracting malicious interests. Moreover, the utilization of wireless communication for data transmission exposes medical data to interception and tampering by cybercriminals. Additionally, anomalies may arise due to human errors, network interference, or hardware malfunctions. In this context, anomaly detection based on Machine Learning (ML) is an interesting solution, but it comes up again
    
[^4]: 非平滑非凸优化中随机次梯度方法的收敛性保证

    Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization. (arXiv:2307.10053v1 [math.OC])

    [http://arxiv.org/abs/2307.10053](http://arxiv.org/abs/2307.10053)

    本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。

    

    本文研究了随机梯度下降（SGD）方法及其变种在训练由非平滑激活函数构建的神经网络中的收敛性质。我们提出了一种新颖的框架，为更新动量项和变量的步长分配了不同的时间尺度。在一些温和的条件下，我们证明了我们提出的框架在单时间尺度和双时间尺度情况下的全局收敛性。我们还证明了我们提出的框架包含了很多已知的SGD类型方法，包括heavy-ball SGD、SignSGD、Lion、normalized SGD和clipped SGD。此外，当目标函数采用有限和形式时，我们基于我们提出的框架证明了这些SGD类型方法的收敛性质。特别地，在温和的假设下，我们证明了这些SGD类型方法在随机选择的步长和初始点上能够找到目标函数的Clarke稳定点。

    In this paper, we investigate the convergence properties of the stochastic gradient descent (SGD) method and its variants, especially in training neural networks built from nonsmooth activation functions. We develop a novel framework that assigns different timescales to stepsizes for updating the momentum terms and variables, respectively. Under mild conditions, we prove the global convergence of our proposed framework in both single-timescale and two-timescale cases. We show that our proposed framework encompasses a wide range of well-known SGD-type methods, including heavy-ball SGD, SignSGD, Lion, normalized SGD and clipped SGD. Furthermore, when the objective function adopts a finite-sum formulation, we prove the convergence properties for these SGD-type methods based on our proposed framework. In particular, we prove that these SGD-type methods find the Clarke stationary points of the objective function with randomly chosen stepsizes and initial points under mild assumptions. Preli
    

