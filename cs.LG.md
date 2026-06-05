# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Binary Gaussian Copula Synthesis: A Novel Data Augmentation Technique to Advance ML-based Clinical Decision Support Systems for Early Prediction of Dialysis Among CKD Patients](https://arxiv.org/abs/2403.00965) | 提出了一种新的数据增强技术 Binary Gaussian Copula Synthesis (BGCS)，用于解决基于机器学习的临床决策支持系统在早期预测慢性肾病患者透析需求中所面临的数据不平衡问题 |
| [^2] | [Exploration via linearly perturbed loss minimisation](https://arxiv.org/abs/2311.07565) | 提出了一种名为EVILL的随机探索方法，通过解决线性扰动的负对数似然函数的极小化问题来工作，提供了关于随机奖励扰动产生良好赌博算法的简洁解释，并在实践中展示了与汤普森抽样风格参数扰动方法性能相匹配的能力 |
| [^3] | [Semi-Offline Reinforcement Learning for Optimized Text Generation.](http://arxiv.org/abs/2306.09712) | 该研究提出了一种半离线强化学习范式，该范式平衡了探索能力和培训成本，提供了一个理论基础来比较不同的强化学习设置，并在优化成本、渐近误差和过度拟合误差界方面实现了最优的RL设置。实验结果表明，该方法高效且性能优异。 |

# 详细

[^1]: 二值高斯Copula合成：一种新的数据增强技术，用于推进基于机器学习的临床决策支持系统，旨在早期预测慢性肾病患者的透析需求

    Binary Gaussian Copula Synthesis: A Novel Data Augmentation Technique to Advance ML-based Clinical Decision Support Systems for Early Prediction of Dialysis Among CKD Patients

    [https://arxiv.org/abs/2403.00965](https://arxiv.org/abs/2403.00965)

    提出了一种新的数据增强技术 Binary Gaussian Copula Synthesis (BGCS)，用于解决基于机器学习的临床决策支持系统在早期预测慢性肾病患者透析需求中所面临的数据不平衡问题

    

    谷歌学术：2403.00965v1  公告类型：跨界  摘要：美国疾病控制中心估计，超过3700万成年美国人患有慢性肾病（CKD），然而其中的9成患者由于早期没有症状而不知道自己的状况。早期预测透析需求至关重要，因为这可以显著改善患者预后，并帮助医疗提供者及时做出知情决策。然而，开发有效的基于机器学习（ML）的早期透析预测临床决策支持系统（CDSS）面临关键挑战，即数据的不平衡性。为了解决这一挑战，本研究评估了各种数据增强技术，以了解它们在现实世界数据集上的有效性。我们提出了一种名为二值高斯Copula合成（BGCS）的新方法，该方法针对二进制数据进行了优化。

    arXiv:2403.00965v1 Announce Type: cross  Abstract: The Center for Disease Control estimates that over 37 million US adults suffer from chronic kidney disease (CKD), yet 9 out of 10 of these individuals are unaware of their condition due to the absence of symptoms in the early stages. It has a significant impact on patients' quality of life, particularly when it progresses to the need for dialysis. Early prediction of dialysis is crucial as it can significantly improve patient outcomes and assist healthcare providers in making timely and informed decisions. However, developing an effective machine learning (ML)-based Clinical Decision Support System (CDSS) for early dialysis prediction poses a key challenge due to the imbalanced nature of data. To address this challenge, this study evaluates various data augmentation techniques to understand their effectiveness on real-world datasets. We propose a new approach named Binary Gaussian Copula Synthesis (BGCS). BGCS is tailored for binary me
    
[^2]: 通过线性扰动的损失最小化来进行探索

    Exploration via linearly perturbed loss minimisation

    [https://arxiv.org/abs/2311.07565](https://arxiv.org/abs/2311.07565)

    提出了一种名为EVILL的随机探索方法，通过解决线性扰动的负对数似然函数的极小化问题来工作，提供了关于随机奖励扰动产生良好赌博算法的简洁解释，并在实践中展示了与汤普森抽样风格参数扰动方法性能相匹配的能力

    

    我们引入了一种称为通过线性损失扰动进行探索（EVILL）的随机探索方法，用于结构化随机赌博问题，该方法通过解决线性扰动正则化负对数似然函数的极小化问题来工作。我们展示，对于广义线性赌博问题，EVILL可以简化为扰动历史探索（PHE），一种通过在随机扰动奖励上进行训练来进行探索的方法。通过这样做，我们对随机奖励扰动何时以及为何产生良好的赌博算法提供了简单干净的解释。我们提出了先前PHE类型方法中不含的数据相关扰动，使EVILL能够在理论和实践中与汤普森抽样风格参数扰动方法的性能相匹配。此外，我们展示了一个超出广义线性赌博的例子，其中PHE导致不一致的估计，从而导致线性后悔，而EVILL则保持表现。

    arXiv:2311.07565v2 Announce Type: replace  Abstract: We introduce exploration via linear loss perturbations (EVILL), a randomised exploration method for structured stochastic bandit problems that works by solving for the minimiser of a linearly perturbed regularised negative log-likelihood function. We show that, for the case of generalised linear bandits, EVILL reduces to perturbed history exploration (PHE), a method where exploration is done by training on randomly perturbed rewards. In doing so, we provide a simple and clean explanation of when and why random reward perturbations give rise to good bandit algorithms. We propose data-dependent perturbations not present in previous PHE-type methods that allow EVILL to match the performance of Thompson-sampling-style parameter-perturbation methods, both in theory and in practice. Moreover, we show an example outside generalised linear bandits where PHE leads to inconsistent estimates, and thus linear regret, while EVILL remains performa
    
[^3]: 半离线强化学习用于优化文本生成

    Semi-Offline Reinforcement Learning for Optimized Text Generation. (arXiv:2306.09712v1 [cs.LG])

    [http://arxiv.org/abs/2306.09712](http://arxiv.org/abs/2306.09712)

    该研究提出了一种半离线强化学习范式，该范式平衡了探索能力和培训成本，提供了一个理论基础来比较不同的强化学习设置，并在优化成本、渐近误差和过度拟合误差界方面实现了最优的RL设置。实验结果表明，该方法高效且性能优异。

    

    在强化学习中，与环境交互有两种主要方式：在线和离线。在线方法探索环境所需时间较长，而离线方法通过牺牲探索能力有效地获得奖励信号。我们提出了半离线RL，一种新的范式，可以平滑地从离线转换到在线设置，平衡探索能力和培训成本，并为比较不同RL设置提供理论基础。基于半离线公式，我们提出了在优化成本、渐近误差和过度拟合误差界方面最优的RL设置。广泛的实验表明，我们的半离线方法效率高，与最先进的方法相比具有可比性或更好的性能。

    In reinforcement learning (RL), there are two major settings for interacting with the environment: online and offline. Online methods explore the environment at significant time cost, and offline methods efficiently obtain reward signals by sacrificing exploration capability. We propose semi-offline RL, a novel paradigm that smoothly transits from offline to online settings, balances exploration capability and training cost, and provides a theoretical foundation for comparing different RL settings. Based on the semi-offline formulation, we present the RL setting that is optimal in terms of optimization cost, asymptotic error, and overfitting error bound. Extensive experiments show that our semi-offline approach is efficient and yields comparable or often better performance compared with state-of-the-art methods.
    

