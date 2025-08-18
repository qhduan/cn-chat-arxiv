# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synthetic Data for Robust Stroke Segmentation](https://arxiv.org/abs/2404.01946) | 提出一种用于中风分割的合成框架，使用病变特定增强策略扩展了SynthSeg方法，通过训练深度学习模型实现对健康组织和病理病变的分割，无需特定序列的训练数据，在领域内和领域外数据集的评估中表现出鲁棒性能。 |
| [^2] | [Discovering Invariant Neighborhood Patterns for Heterophilic Graphs](https://arxiv.org/abs/2403.10572) | 本文提出了一种新颖的不变邻域模式学习方法，通过自适应邻域传播模块和不变非同源图学习模块，解决了非同源图上邻域模式分布偏移问题。 |
| [^3] | [JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example.](http://arxiv.org/abs/2401.01199) | JMA是一种通用算法，用于生成几乎最优的定向对抗样本。该算法通过最小化Jacobian引起的马氏距离，考虑了将输入样本的潜在空间表示在给定方向上移动所需的投入。该算法在解决对抗样本问题方面提供了最优解。 |
| [^4] | [Federated Learning with Differential Privacy for End-to-End Speech Recognition.](http://arxiv.org/abs/2310.00098) | 本文提出了一种基于联邦学习和差分隐私的端到端语音识别方法，探索了大型Transformer模型的不同方面，并建立了基线结果。 |
| [^5] | [Planning to Learn: A Novel Algorithm for Active Learning during Model-Based Planning.](http://arxiv.org/abs/2308.08029) | 本论文提出了一种新颖的算法，称为SI SL，用于主动学习和模型优化过程中的规划。该算法通过与贝叶斯强化学习方案的比较证明了其性能的优越性。 |

# 详细

[^1]: 用于鲁棒性中风分割的合成数据

    Synthetic Data for Robust Stroke Segmentation

    [https://arxiv.org/abs/2404.01946](https://arxiv.org/abs/2404.01946)

    提出一种用于中风分割的合成框架，使用病变特定增强策略扩展了SynthSeg方法，通过训练深度学习模型实现对健康组织和病理病变的分割，无需特定序列的训练数据，在领域内和领域外数据集的评估中表现出鲁棒性能。

    

    arXiv:2404.01946v1 公告类型：交叉 摘要：目前基于深度学习的神经影像语义分割需要高分辨率扫描和大量注释数据集，这给临床适用性带来了显著障碍。我们提出了一种新颖的合成框架，用于病变分割任务，扩展了已建立的SynthSeg方法的能力，以适应具有病变特定增强策略的大型异质病变。我们的方法使用从健康和中风数据集派生的标签映射训练深度学习模型，在这里演示了UNet架构，促进了健康组织和病理病变的分割，而无需特定于序列的训练数据。针对领域内和领域外（OOD）数据集进行评估，我们的框架表现出鲁棒性能，与训练领域内的当前方法相媲美，并在OOD数据上显着优于它们。这一贡献有望推动医学...

    arXiv:2404.01946v1 Announce Type: cross  Abstract: Deep learning-based semantic segmentation in neuroimaging currently requires high-resolution scans and extensive annotated datasets, posing significant barriers to clinical applicability. We present a novel synthetic framework for the task of lesion segmentation, extending the capabilities of the established SynthSeg approach to accommodate large heterogeneous pathologies with lesion-specific augmentation strategies. Our method trains deep learning models, demonstrated here with the UNet architecture, using label maps derived from healthy and stroke datasets, facilitating the segmentation of both healthy tissue and pathological lesions without sequence-specific training data. Evaluated against in-domain and out-of-domain (OOD) datasets, our framework demonstrates robust performance, rivaling current methods within the training domain and significantly outperforming them on OOD data. This contribution holds promise for advancing medical
    
[^2]: 发现异性图的不变邻域模式

    Discovering Invariant Neighborhood Patterns for Heterophilic Graphs

    [https://arxiv.org/abs/2403.10572](https://arxiv.org/abs/2403.10572)

    本文提出了一种新颖的不变邻域模式学习方法，通过自适应邻域传播模块和不变非同源图学习模块，解决了非同源图上邻域模式分布偏移问题。

    

    本文研究了非同源图上的分布偏移问题。大多数现有的图神经网络方法依赖于同源假设，即同一类别的节点更有可能被连接。然而，在现实世界的图中，这种同源性假设并不总是成立，这导致了在先前的方法中未能解释的更复杂的分布偏移。在非同源图上，邻域模式的分布偏移更加多样化。我们提出了一种新颖的不变邻域模式学习（INPL）方法，以缓解非同源图上的分布偏移问题。具体来说，我们提出了自适应邻域传播（ANP）模块来捕获自适应的邻域信息，这可以缓解非同源图上的邻域模式分布偏移问题。我们提出了不变非同源图学习（INHGL）模块，其约束了ANP并进行学习。

    arXiv:2403.10572v1 Announce Type: new  Abstract: This paper studies the problem of distribution shifts on non-homophilous graphs Mosting existing graph neural network methods rely on the homophilous assumption that nodes from the same class are more likely to be linked. However, such assumptions of homophily do not always hold in real-world graphs, which leads to more complex distribution shifts unaccounted for in previous methods. The distribution shifts of neighborhood patterns are much more diverse on non-homophilous graphs. We propose a novel Invariant Neighborhood Pattern Learning (INPL) to alleviate the distribution shifts problem on non-homophilous graphs. Specifically, we propose the Adaptive Neighborhood Propagation (ANP) module to capture the adaptive neighborhood information, which could alleviate the neighborhood pattern distribution shifts problem on non-homophilous graphs. We propose Invariant Non-Homophilous Graph Learning (INHGL) module to constrain the ANP and learn in
    
[^3]: JMA:一种快速生成几乎最优定向对抗样本的通用算法

    JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example. (arXiv:2401.01199v1 [cs.LG])

    [http://arxiv.org/abs/2401.01199](http://arxiv.org/abs/2401.01199)

    JMA是一种通用算法，用于生成几乎最优的定向对抗样本。该算法通过最小化Jacobian引起的马氏距离，考虑了将输入样本的潜在空间表示在给定方向上移动所需的投入。该算法在解决对抗样本问题方面提供了最优解。

    

    目前为止，大多数用于生成针对深度学习分类器的定向对抗样本的方法都是高度次优的，通常依赖于增加目标类别的可能性，因此隐含地专注于一热编码设置。在本文中，我们提出了一种更加通用的、理论上可靠的定向攻击方法，该方法利用最小化雅可比引起的马氏距离（JMA）项，考虑将输入样本的潜在空间表示在给定方向上移动所需的投入（在输入空间中）。通过利用沃尔夫二重性定理求解最小化问题，将问题简化为解非负最小二乘（NNLS）问题。所提出的算法为Szegedy等人最初引入的对抗样本问题的线性化版本提供了最优解。我们进行的实验证实了所提出的攻击的广泛性。

    Most of the approaches proposed so far to craft targeted adversarial examples against Deep Learning classifiers are highly suboptimal and typically rely on increasing the likelihood of the target class, thus implicitly focusing on one-hot encoding settings. In this paper, we propose a more general, theoretically sound, targeted attack that resorts to the minimization of a Jacobian-induced MAhalanobis distance (JMA) term, taking into account the effort (in the input space) required to move the latent space representation of the input sample in a given direction. The minimization is solved by exploiting the Wolfe duality theorem, reducing the problem to the solution of a Non-Negative Least Square (NNLS) problem. The proposed algorithm provides an optimal solution to a linearized version of the adversarial example problem originally introduced by Szegedy et al. \cite{szegedy2013intriguing}. The experiments we carried out confirm the generality of the proposed attack which is proven to be 
    
[^4]: 使用差分隐私的联邦学习进行端到端语音识别

    Federated Learning with Differential Privacy for End-to-End Speech Recognition. (arXiv:2310.00098v1 [cs.LG])

    [http://arxiv.org/abs/2310.00098](http://arxiv.org/abs/2310.00098)

    本文提出了一种基于联邦学习和差分隐私的端到端语音识别方法，探索了大型Transformer模型的不同方面，并建立了基线结果。

    

    联邦学习是一种有前景的训练机器学习模型的方法，但在自动语音识别领域仅限于初步探索。此外，联邦学习不能本质上保证用户隐私，并需要差分隐私来提供稳健的隐私保证。然而，我们还不清楚在自动语音识别中应用差分隐私的先前工作。本文旨在通过为联邦学习提供差分隐私的自动语音识别基准，并建立第一个基线来填补这一研究空白。我们扩展了现有的联邦学习自动语音识别研究，探索了最新的大型端到端Transformer模型的不同方面：架构设计，种子模型，数据异质性，领域转移，以及cohort大小的影响。通过合理的中央聚合数量，我们能够训练出即使在异构数据、来自另一个领域的种子模型或无预先训练的情况下仍然接近最优的联邦学习模型。

    While federated learning (FL) has recently emerged as a promising approach to train machine learning models, it is limited to only preliminary explorations in the domain of automatic speech recognition (ASR). Moreover, FL does not inherently guarantee user privacy and requires the use of differential privacy (DP) for robust privacy guarantees. However, we are not aware of prior work on applying DP to FL for ASR. In this paper, we aim to bridge this research gap by formulating an ASR benchmark for FL with DP and establishing the first baselines. First, we extend the existing research on FL for ASR by exploring different aspects of recent $\textit{large end-to-end transformer models}$: architecture design, seed models, data heterogeneity, domain shift, and impact of cohort size. With a $\textit{practical}$ number of central aggregations we are able to train $\textbf{FL models}$ that are \textbf{nearly optimal} even with heterogeneous data, a seed model from another domain, or no pre-trai
    
[^5]: 规划学习：一种新颖的模型优化过程中的主动学习算法

    Planning to Learn: A Novel Algorithm for Active Learning during Model-Based Planning. (arXiv:2308.08029v1 [cs.AI])

    [http://arxiv.org/abs/2308.08029](http://arxiv.org/abs/2308.08029)

    本论文提出了一种新颖的算法，称为SI SL，用于主动学习和模型优化过程中的规划。该算法通过与贝叶斯强化学习方案的比较证明了其性能的优越性。

    

    主动推理是一种近期的对不确定性情境下规划建模的框架。现在人们已经开始评估这种方法的优缺点以及如何改进它。最近的一个拓展-复杂模型优化算法通过递归决策树搜索在多步规划问题上提高了性能。然而，迄今为止很少有工作对比SI与其他已建立的规划算法。SI算法也主要关注推理而不是学习。本文有两个目标。首先，我们比较SI与旨在解决相似问题的贝叶斯强化学习（RL）方案的性能。其次，我们提出了SI复杂学习（SL）的拓展，该拓展在规划过程中更加充分地引入了主动学习。SL维持对未来观测下每个策略下模型参数如何变化的信念。这允许了一种反事实的回顾性评估。

    Active Inference is a recent framework for modeling planning under uncertainty. Empirical and theoretical work have now begun to evaluate the strengths and weaknesses of this approach and how it might be improved. A recent extension - the sophisticated inference (SI) algorithm - improves performance on multi-step planning problems through recursive decision tree search. However, little work to date has been done to compare SI to other established planning algorithms. SI was also developed with a focus on inference as opposed to learning. The present paper has two aims. First, we compare performance of SI to Bayesian reinforcement learning (RL) schemes designed to solve similar problems. Second, we present an extension of SI sophisticated learning (SL) - that more fully incorporates active learning during planning. SL maintains beliefs about how model parameters would change under the future observations expected under each policy. This allows a form of counterfactual retrospective in
    

