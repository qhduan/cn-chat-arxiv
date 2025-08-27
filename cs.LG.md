# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty-Calibrated Test-Time Model Adaptation without Forgetting](https://arxiv.org/abs/2403.11491) | 提出了一种高效的抗遗忘测试时间适应（EATA）方法，通过开发主动样本选择标准和引入Fisher正则化约束重要模型参数，实现了不会忘记的不确定性校准测试时间模型适应。 |
| [^2] | [Sharp Lower Bounds on Interpolation by Deep ReLU Neural Networks at Irregularly Spaced Data](https://arxiv.org/abs/2302.00834) | 该论文研究了深度ReLU神经网络在不规则间隔数据上的插值问题，证明了在数据点间距指数级小的情况下需要$\Omega(N)$个参数，同时指出现有的位提取技术无法应用于这种情况。 |
| [^3] | [Learning Optimal Classification Trees Robust to Distribution Shifts.](http://arxiv.org/abs/2310.17772) | 本研究提出了一种学习对分布变化具有鲁棒性的最优分类树的方法，通过混合整数鲁棒优化技术将该问题转化为单阶段混合整数鲁棒优化问题，并设计了基于约束生成的解决过程。 |
| [^4] | [Contraction Properties of the Global Workspace Primitive.](http://arxiv.org/abs/2310.01571) | 本文扩展了关于多区域递归神经网络的稳定性研究，并在全局工作空间模块化结构上证明了松散稳定性条件。实证结果显示全局工作空间稀疏组合网络在测试表现和韧性方面表现出较好的性能，强调了稳定性对于实现模块化RNN的重要性。 |
| [^5] | [On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets.](http://arxiv.org/abs/2307.05284) | 该论文通过对表格数据集中的自然偏移进行研究，发现$Y|X$-偏移最为普遍。为了推动研究人员开发描述数据分布偏移的精细语言，作者构建了WhyShift实验平台，并讨论了$Y|X$-偏移对算法的影响。 |

# 详细

[^1]: 不会忘却的不确定性校准测试时间模型适应

    Uncertainty-Calibrated Test-Time Model Adaptation without Forgetting

    [https://arxiv.org/abs/2403.11491](https://arxiv.org/abs/2403.11491)

    提出了一种高效的抗遗忘测试时间适应（EATA）方法，通过开发主动样本选择标准和引入Fisher正则化约束重要模型参数，实现了不会忘记的不确定性校准测试时间模型适应。

    

    测试时间适应（TTA）旨在通过根据任何测试样本调整给定模型，以应对训练和测试数据之间的潜在分布偏移。尽管最近的TTA表现出有希望的性能，但我们仍然面临两个关键挑战：1）先前的方法对每个测试样本执行反向传播，导致许多应用程序无法承受的优化成本；2）虽然现有的TTA可以显著改善在分布数据上的测试性能，但它们经常在TTA后在分布数据上遭受严重性能下降（即所谓的遗忘）。为此，我们提出了一种高效的抗遗忘测试时间适应（EATA）方法，该方法开发了一个主动样本选择标准，以识别可靠且非冗余的样本进行在测试时间的熵最小化。为了缓解遗忘，EATA引入了从测试样本中估计的Fisher正则化项，以约束重要的模型参数免于急剧变化。

    arXiv:2403.11491v1 Announce Type: new  Abstract: Test-time adaptation (TTA) seeks to tackle potential distribution shifts between training and test data by adapting a given model w.r.t. any test sample. Although recent TTA has shown promising performance, we still face two key challenges: 1) prior methods perform backpropagation for each test sample, resulting in unbearable optimization costs to many applications; 2) while existing TTA can significantly improve the test performance on out-of-distribution data, they often suffer from severe performance degradation on in-distribution data after TTA (known as forgetting). To this end, we have proposed an Efficient Anti-Forgetting Test-Time Adaptation (EATA) method which develops an active sample selection criterion to identify reliable and non-redundant samples for test-time entropy minimization. To alleviate forgetting, EATA introduces a Fisher regularizer estimated from test samples to constrain important model parameters from drastic c
    
[^2]: 用于不规则间隔数据的深度ReLU神经网络插值的尖锐下界

    Sharp Lower Bounds on Interpolation by Deep ReLU Neural Networks at Irregularly Spaced Data

    [https://arxiv.org/abs/2302.00834](https://arxiv.org/abs/2302.00834)

    该论文研究了深度ReLU神经网络在不规则间隔数据上的插值问题，证明了在数据点间距指数级小的情况下需要$\Omega(N)$个参数，同时指出现有的位提取技术无法应用于这种情况。

    

    我们研究了深度ReLU神经网络的插值能力。具体来说，我们考虑深度ReLU网络如何在单位球中的$N$个数据点上进行值的插值，这些点之间相距$\delta$。我们表明在$\delta$在$N$指数级小的区域中需要$\Omega(N)$个参数，这给出了该区域的尖锐结果，因为$O(N)$个参数总是足够的。 这也表明用于证明VC维度下界的位提取技术无法应用于不规则间隔的数据点。最后，作为应用，我们给出了深度ReLU神经网络在嵌入端点处为Sobolev空间实现的近似速率的下界。

    arXiv:2302.00834v2 Announce Type: replace  Abstract: We study the interpolation power of deep ReLU neural networks. Specifically, we consider the question of how efficiently, in terms of the number of parameters, deep ReLU networks can interpolate values at $N$ datapoints in the unit ball which are separated by a distance $\delta$. We show that $\Omega(N)$ parameters are required in the regime where $\delta$ is exponentially small in $N$, which gives the sharp result in this regime since $O(N)$ parameters are always sufficient. This also shows that the bit-extraction technique used to prove lower bounds on the VC dimension cannot be applied to irregularly spaced datapoints. Finally, as an application we give a lower bound on the approximation rates that deep ReLU neural networks can achieve for Sobolev spaces at the embedding endpoint.
    
[^3]: 学习对分布变化具有鲁棒性的最优分类树

    Learning Optimal Classification Trees Robust to Distribution Shifts. (arXiv:2310.17772v1 [cs.LG])

    [http://arxiv.org/abs/2310.17772](http://arxiv.org/abs/2310.17772)

    本研究提出了一种学习对分布变化具有鲁棒性的最优分类树的方法，通过混合整数鲁棒优化技术将该问题转化为单阶段混合整数鲁棒优化问题，并设计了基于约束生成的解决过程。

    

    我们考虑学习对训练和测试/部署数据之间的分布变化具有鲁棒性的分类树的问题。这个问题经常在高风险环境中出现，例如公共卫生和社会工作，其中数据通常是通过自我报告的调查收集的，这些调查对问题的表述方式、调查进行的时间和地点、以及受访者与调查员分享信息的舒适程度非常敏感。我们提出了一种基于混合整数鲁棒优化技术的学习最优鲁棒分类树的方法。特别地，我们证明学习最优鲁棒树的问题可以等价地表达为一个具有高度非线性和不连续目标的单阶段混合整数鲁棒优化问题。我们将这个问题等价地重新表述为一个两阶段线性鲁棒优化问题，为此我们设计了一个基于约束生成的定制解决过程。

    We consider the problem of learning classification trees that are robust to distribution shifts between training and testing/deployment data. This problem arises frequently in high stakes settings such as public health and social work where data is often collected using self-reported surveys which are highly sensitive to e.g., the framing of the questions, the time when and place where the survey is conducted, and the level of comfort the interviewee has in sharing information with the interviewer. We propose a method for learning optimal robust classification trees based on mixed-integer robust optimization technology. In particular, we demonstrate that the problem of learning an optimal robust tree can be cast as a single-stage mixed-integer robust optimization problem with a highly nonlinear and discontinuous objective. We reformulate this problem equivalently as a two-stage linear robust optimization problem for which we devise a tailored solution procedure based on constraint gene
    
[^4]: 全局工作空间基元的收缩性质

    Contraction Properties of the Global Workspace Primitive. (arXiv:2310.01571v1 [cs.LG])

    [http://arxiv.org/abs/2310.01571](http://arxiv.org/abs/2310.01571)

    本文扩展了关于多区域递归神经网络的稳定性研究，并在全局工作空间模块化结构上证明了松散稳定性条件。实证结果显示全局工作空间稀疏组合网络在测试表现和韧性方面表现出较好的性能，强调了稳定性对于实现模块化RNN的重要性。

    

    为了推动关于多区域递归神经网络(RNNs)的重要新兴研究领域，我们在Kozachkov等人的“递归构建稳定的递归神经网络组装品”的基础上在理论上和实证上进行了扩展。我们证明了该架构的显著特例的松散稳定性条件，特别是针对全局工作空间模块化结构。我们通过对具有少量可训练参数的全局工作空间稀疏组合网络进行实证成功，不仅在整体测试表现方面表现出强大的性能，还对单个子网络的移除具有更大的韧性。这些全局工作空间接触区拓扑的实证结果依赖于稳定性的保持，突出了我们的理论工作对于实现模块化RNN的成功的相关性。此外，通过更广泛地探索不同子网络模块之间的连接结构的稀疏性，

    To push forward the important emerging research field surrounding multi-area recurrent neural networks (RNNs), we expand theoretically and empirically on the provably stable RNNs of RNNs introduced by Kozachkov et al. in "RNNs of RNNs: Recursive Construction of Stable Assemblies of Recurrent Neural Networks". We prove relaxed stability conditions for salient special cases of this architecture, most notably for a global workspace modular structure. We then demonstrate empirical success for Global Workspace Sparse Combo Nets with a small number of trainable parameters, not only through strong overall test performance but also greater resilience to removal of individual subnetworks. These empirical results for the global workspace inter-area topology are contingent on stability preservation, highlighting the relevance of our theoretical work for enabling modular RNN success. Further, by exploring sparsity in the connectivity structure between different subnetwork modules more broadly, we 
    
[^5]: 关于需要描述分布偏移的语言：基于表格数据集的案例分析

    On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets. (arXiv:2307.05284v1 [cs.LG])

    [http://arxiv.org/abs/2307.05284](http://arxiv.org/abs/2307.05284)

    该论文通过对表格数据集中的自然偏移进行研究，发现$Y|X$-偏移最为普遍。为了推动研究人员开发描述数据分布偏移的精细语言，作者构建了WhyShift实验平台，并讨论了$Y|X$-偏移对算法的影响。

    

    不同的分布偏移需要不同的算法和操作干预。方法研究必须以其所涉及的具体偏移为基础。尽管新兴的基准数据为实证研究提供了有希望的基础，但它们隐含地关注协变量偏移，并且实证发现的有效性取决于偏移类型，例如，当$Y|X$分布发生变化时，之前关于算法性能的观察可能无效。我们对5个表格数据集中的自然偏移进行了深入研究，通过对86,000个模型配置进行实验，发现$Y|X$-偏移最为普遍。为了鼓励研究人员开发一种精细的描述数据分布偏移的语言，我们构建了WhyShift，一个由策划的真实世界偏移测试平台，在其中我们对我们基准性能的偏移类型进行了表征。由于$Y|X$-偏移在表格设置中很常见，我们确定了受到最大$Y|X$-偏移影响的协变量区域，并讨论了对算法的影响。

    Different distribution shifts require different algorithmic and operational interventions. Methodological research must be grounded by the specific shifts they address. Although nascent benchmarks provide a promising empirical foundation, they implicitly focus on covariate shifts, and the validity of empirical findings depends on the type of shift, e.g., previous observations on algorithmic performance can fail to be valid when the $Y|X$ distribution changes. We conduct a thorough investigation of natural shifts in 5 tabular datasets over 86,000 model configurations, and find that $Y|X$-shifts are most prevalent. To encourage researchers to develop a refined language for distribution shifts, we build WhyShift, an empirical testbed of curated real-world shifts where we characterize the type of shift we benchmark performance over. Since $Y|X$-shifts are prevalent in tabular settings, we identify covariate regions that suffer the biggest $Y|X$-shifts and discuss implications for algorithm
    

