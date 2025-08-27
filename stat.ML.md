# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sharp Lower Bounds on Interpolation by Deep ReLU Neural Networks at Irregularly Spaced Data](https://arxiv.org/abs/2302.00834) | 该论文研究了深度ReLU神经网络在不规则间隔数据上的插值问题，证明了在数据点间距指数级小的情况下需要$\Omega(N)$个参数，同时指出现有的位提取技术无法应用于这种情况。 |
| [^2] | [Learning Optimal Classification Trees Robust to Distribution Shifts.](http://arxiv.org/abs/2310.17772) | 本研究提出了一种学习对分布变化具有鲁棒性的最优分类树的方法，通过混合整数鲁棒优化技术将该问题转化为单阶段混合整数鲁棒优化问题，并设计了基于约束生成的解决过程。 |

# 详细

[^1]: 用于不规则间隔数据的深度ReLU神经网络插值的尖锐下界

    Sharp Lower Bounds on Interpolation by Deep ReLU Neural Networks at Irregularly Spaced Data

    [https://arxiv.org/abs/2302.00834](https://arxiv.org/abs/2302.00834)

    该论文研究了深度ReLU神经网络在不规则间隔数据上的插值问题，证明了在数据点间距指数级小的情况下需要$\Omega(N)$个参数，同时指出现有的位提取技术无法应用于这种情况。

    

    我们研究了深度ReLU神经网络的插值能力。具体来说，我们考虑深度ReLU网络如何在单位球中的$N$个数据点上进行值的插值，这些点之间相距$\delta$。我们表明在$\delta$在$N$指数级小的区域中需要$\Omega(N)$个参数，这给出了该区域的尖锐结果，因为$O(N)$个参数总是足够的。 这也表明用于证明VC维度下界的位提取技术无法应用于不规则间隔的数据点。最后，作为应用，我们给出了深度ReLU神经网络在嵌入端点处为Sobolev空间实现的近似速率的下界。

    arXiv:2302.00834v2 Announce Type: replace  Abstract: We study the interpolation power of deep ReLU neural networks. Specifically, we consider the question of how efficiently, in terms of the number of parameters, deep ReLU networks can interpolate values at $N$ datapoints in the unit ball which are separated by a distance $\delta$. We show that $\Omega(N)$ parameters are required in the regime where $\delta$ is exponentially small in $N$, which gives the sharp result in this regime since $O(N)$ parameters are always sufficient. This also shows that the bit-extraction technique used to prove lower bounds on the VC dimension cannot be applied to irregularly spaced datapoints. Finally, as an application we give a lower bound on the approximation rates that deep ReLU neural networks can achieve for Sobolev spaces at the embedding endpoint.
    
[^2]: 学习对分布变化具有鲁棒性的最优分类树

    Learning Optimal Classification Trees Robust to Distribution Shifts. (arXiv:2310.17772v1 [cs.LG])

    [http://arxiv.org/abs/2310.17772](http://arxiv.org/abs/2310.17772)

    本研究提出了一种学习对分布变化具有鲁棒性的最优分类树的方法，通过混合整数鲁棒优化技术将该问题转化为单阶段混合整数鲁棒优化问题，并设计了基于约束生成的解决过程。

    

    我们考虑学习对训练和测试/部署数据之间的分布变化具有鲁棒性的分类树的问题。这个问题经常在高风险环境中出现，例如公共卫生和社会工作，其中数据通常是通过自我报告的调查收集的，这些调查对问题的表述方式、调查进行的时间和地点、以及受访者与调查员分享信息的舒适程度非常敏感。我们提出了一种基于混合整数鲁棒优化技术的学习最优鲁棒分类树的方法。特别地，我们证明学习最优鲁棒树的问题可以等价地表达为一个具有高度非线性和不连续目标的单阶段混合整数鲁棒优化问题。我们将这个问题等价地重新表述为一个两阶段线性鲁棒优化问题，为此我们设计了一个基于约束生成的定制解决过程。

    We consider the problem of learning classification trees that are robust to distribution shifts between training and testing/deployment data. This problem arises frequently in high stakes settings such as public health and social work where data is often collected using self-reported surveys which are highly sensitive to e.g., the framing of the questions, the time when and place where the survey is conducted, and the level of comfort the interviewee has in sharing information with the interviewer. We propose a method for learning optimal robust classification trees based on mixed-integer robust optimization technology. In particular, we demonstrate that the problem of learning an optimal robust tree can be cast as a single-stage mixed-integer robust optimization problem with a highly nonlinear and discontinuous objective. We reformulate this problem equivalently as a two-stage linear robust optimization problem for which we devise a tailored solution procedure based on constraint gene
    

