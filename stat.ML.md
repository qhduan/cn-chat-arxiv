# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Anytime Algorithm for Good Arm Identification.](http://arxiv.org/abs/2310.10359) | 提出了一个适用于随机贝叶斯臂机的随时和无参数采样规则APGAI，通过自适应策略提高了好臂识别效率，在固定置信度和固定预算的情况下有良好实验表现。 |

# 详细

[^1]: 一个适用于好臂识别的随时算法

    An Anytime Algorithm for Good Arm Identification. (arXiv:2310.10359v1 [stat.ML])

    [http://arxiv.org/abs/2310.10359](http://arxiv.org/abs/2310.10359)

    提出了一个适用于随机贝叶斯臂机的随时和无参数采样规则APGAI，通过自适应策略提高了好臂识别效率，在固定置信度和固定预算的情况下有良好实验表现。

    

    在好臂识别（GAI）中，目标是识别其中一个平均性能超过给定阈值的臂，称为好臂（如果存在）。目前很少有研究在固定预算的情况下进行GAI，即在先确定好预算之后，或者在任何时刻都可以要求推荐的随时设置下进行GAI。我们提出了一种名为APGAI的随时和无参数采样规则，用于随机贝叶斯臂机。APGAI可以直接用于固定置信度和固定预算的设定中。首先，我们得出其任何时刻的误差概率的上界。这些上界表明，自适应策略在检测没有好臂的时候比均匀采样更高效。其次，当APGAI与一个停止规则结合时，我们证明了在任何置信水平下的预期采样复杂性的上界。最后，我们展示了APGAI在合成数据和真实世界数据上的良好实验性能。我们的工作为所有设置中的GAI问题提供了一个广泛的概述。

    In good arm identification (GAI), the goal is to identify one arm whose average performance exceeds a given threshold, referred to as good arm, if it exists. Few works have studied GAI in the fixed-budget setting, when the sampling budget is fixed beforehand, or the anytime setting, when a recommendation can be asked at any time. We propose APGAI, an anytime and parameter-free sampling rule for GAI in stochastic bandits. APGAI can be straightforwardly used in fixed-confidence and fixed-budget settings. First, we derive upper bounds on its probability of error at any time. They show that adaptive strategies are more efficient in detecting the absence of good arms than uniform sampling. Second, when APGAI is combined with a stopping rule, we prove upper bounds on the expected sampling complexity, holding at any confidence level. Finally, we show good empirical performance of APGAI on synthetic and real-world data. Our work offers an extensive overview of the GAI problem in all settings.
    

