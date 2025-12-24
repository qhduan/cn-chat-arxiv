# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Budget Aggregation with Single-Peaked Preferences](https://arxiv.org/abs/2402.15904) | 论文研究了具有单峰偏好的最优预算聚合问题，并在多维泛化的星形效用函数类别中探讨不同模型。对于两种备选方案，证明了统一幻影机制是唯一满足比例性的策略证明机制。然后，对于超过两种备选方案的情况，论文表明不存在同时满足效率、策略性和比例性的机制。 |
| [^2] | [Behavioral Machine Learning? Computer Predictions of Corporate Earnings also Overreact.](http://arxiv.org/abs/2303.16158) | 本文研究发现，机器学习算法可以更准确地预测公司盈利，但同样存在过度反应的问题，而传统培训的股市分析师和经过机器学习方法培训的分析师相比会产生较少的过度反应。 |

# 详细

[^1]: 具有单峰偏好的最优预算聚合

    Optimal Budget Aggregation with Single-Peaked Preferences

    [https://arxiv.org/abs/2402.15904](https://arxiv.org/abs/2402.15904)

    论文研究了具有单峰偏好的最优预算聚合问题，并在多维泛化的星形效用函数类别中探讨不同模型。对于两种备选方案，证明了统一幻影机制是唯一满足比例性的策略证明机制。然后，对于超过两种备选方案的情况，论文表明不存在同时满足效率、策略性和比例性的机制。

    

    我们研究了将分布（如预算提案）聚合成集体分布的问题。理想的聚合机制应该是帕累托有效、策略证明和公平的。大多数先前的工作假设代理根据其理想预算与$\ell_1$距离来评估预算。我们研究并比较了来自星形效用函数更大类别的不同模型，这是单峰偏好的多维泛化。对于两种备选方案的情况，我们通过证明在非常一般的假设下，统一幻影机制是唯一满足比例性的策略证明机制，从而扩展了现有结果。对于超过两种备选方案的情况，我们对$\ell_1$和$\ell_\infty$的不满意性建立了全面的不可能性：没有机制能够同时满足效率、策略性和比例性。

    arXiv:2402.15904v1 Announce Type: new  Abstract: We study the problem of aggregating distributions, such as budget proposals, into a collective distribution. An ideal aggregation mechanism would be Pareto efficient, strategyproof, and fair. Most previous work assumes that agents evaluate budgets according to the $\ell_1$ distance to their ideal budget. We investigate and compare different models from the larger class of star-shaped utility functions - a multi-dimensional generalization of single-peaked preferences. For the case of two alternatives, we extend existing results by proving that under very general assumptions, the uniform phantom mechanism is the only strategyproof mechanism that satisfies proportionality - a minimal notion of fairness introduced by Freeman et al. (2021). Moving to the case of more than two alternatives, we establish sweeping impossibilities for $\ell_1$ and $\ell_\infty$ disutilities: no mechanism satisfies efficiency, strategyproofness, and proportionalit
    
[^2]: 机器学习准确预测财报，但同样存在过度反应

    Behavioral Machine Learning? Computer Predictions of Corporate Earnings also Overreact. (arXiv:2303.16158v1 [q-fin.ST])

    [http://arxiv.org/abs/2303.16158](http://arxiv.org/abs/2303.16158)

    本文研究发现，机器学习算法可以更准确地预测公司盈利，但同样存在过度反应的问题，而传统培训的股市分析师和经过机器学习方法培训的分析师相比会产生较少的过度反应。

    

    大量证据表明，在金融领域中，机器学习算法的预测能力比人类更为准确。但是，文献并未测试算法预测是否更为理性。本文研究了几个算法（包括线性回归和一种名为Gradient Boosted Regression Trees的流行算法）对于公司盈利的预测结果。结果发现，GBRT平均胜过线性回归和人类股市分析师，但仍存在过度反应且无法满足理性预期标准。通过降低学习率，可最小程度上减少过度反应程度，但这会牺牲预测准确性。通过机器学习方法培训过的股市分析师比传统训练的分析师产生的过度反应较少。此外，股市分析师的预测反映出机器算法没有捕捉到的信息。

    There is considerable evidence that machine learning algorithms have better predictive abilities than humans in various financial settings. But, the literature has not tested whether these algorithmic predictions are more rational than human predictions. We study the predictions of corporate earnings from several algorithms, notably linear regressions and a popular algorithm called Gradient Boosted Regression Trees (GBRT). On average, GBRT outperformed both linear regressions and human stock analysts, but it still overreacted to news and did not satisfy rational expectation as normally defined. By reducing the learning rate, the magnitude of overreaction can be minimized, but it comes with the cost of poorer out-of-sample prediction accuracy. Human stock analysts who have been trained in machine learning methods overreact less than traditionally trained analysts. Additionally, stock analyst predictions reflect information not otherwise available to machine algorithms.
    

