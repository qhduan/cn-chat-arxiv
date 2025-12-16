# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical inference for pairwise comparison models.](http://arxiv.org/abs/2401.08463) | 本论文通过建立极大似然估计量的渐近正态性结果，填补了配对比较模型中统计推断的空白，为各种配对比较模型提供了统一的方法，超越了Bradley-Terry模型，为实践者提供了坚实的理论保证。 |
| [^2] | [An Anytime Algorithm for Good Arm Identification.](http://arxiv.org/abs/2310.10359) | 提出了一个适用于随机贝叶斯臂机的随时和无参数采样规则APGAI，通过自适应策略提高了好臂识别效率，在固定置信度和固定预算的情况下有良好实验表现。 |
| [^3] | [Sharp Generalization of Transductive Learning: A Transductive Local Rademacher Complexity Approach.](http://arxiv.org/abs/2309.16858) | 我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们利用变量的方差信息构建了TLRC，并将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。 |

# 详细

[^1]: 对于配对比较模型的统计推断

    Statistical inference for pairwise comparison models. (arXiv:2401.08463v1 [math.ST])

    [http://arxiv.org/abs/2401.08463](http://arxiv.org/abs/2401.08463)

    本论文通过建立极大似然估计量的渐近正态性结果，填补了配对比较模型中统计推断的空白，为各种配对比较模型提供了统一的方法，超越了Bradley-Terry模型，为实践者提供了坚实的理论保证。

    

    配对比较模型被用于各个领域的实用性和排名评估。现代问题规模的增加强调了对于当被比较对象数量无限增加时，对于这些模型中的统计推断的理解的需求。目前，文献中对于这些模型中的统计推断的理解还相当有限，除非只是在少数特殊实例中。本文通过在广泛的配对比较模型中建立极大似然估计量的渐近正态性结果来填补这一空白。关键思想在于将费舍尔信息矩阵识别为加权图拉普拉斯矩阵，通过一种细致入微的谱分析方法来进行研究。我们的发现为在各种配对比较模型中进行统计推断提供了第一个统一的方法，超越了Bradley-Terry模型，为实践者提供了坚实的理论保证。通过利用合成数据进行的模拟验证这一渐近正态性结果，然后进行了

    Pairwise comparison models are used for quantitatively evaluating utility and ranking in various fields. The increasing scale of modern problems underscores the need to understand statistical inference in these models when the number of subjects diverges, which is currently lacking in the literature except in a few special instances. This paper addresses this gap by establishing an asymptotic normality result for the maximum likelihood estimator in a broad class of pairwise comparison models. The key idea lies in identifying the Fisher information matrix as a weighted graph Laplacian matrix which can be studied via a meticulous spectral analysis. Our findings provide the first unified theory for performing statistical inference in a wide range of pairwise comparison models beyond the Bradley--Terry model, benefiting practitioners with a solid theoretical guarantee for their use. Simulations utilizing synthetic data are conducted to validate the asymptotic normality result, followed by 
    
[^2]: 一个适用于好臂识别的随时算法

    An Anytime Algorithm for Good Arm Identification. (arXiv:2310.10359v1 [stat.ML])

    [http://arxiv.org/abs/2310.10359](http://arxiv.org/abs/2310.10359)

    提出了一个适用于随机贝叶斯臂机的随时和无参数采样规则APGAI，通过自适应策略提高了好臂识别效率，在固定置信度和固定预算的情况下有良好实验表现。

    

    在好臂识别（GAI）中，目标是识别其中一个平均性能超过给定阈值的臂，称为好臂（如果存在）。目前很少有研究在固定预算的情况下进行GAI，即在先确定好预算之后，或者在任何时刻都可以要求推荐的随时设置下进行GAI。我们提出了一种名为APGAI的随时和无参数采样规则，用于随机贝叶斯臂机。APGAI可以直接用于固定置信度和固定预算的设定中。首先，我们得出其任何时刻的误差概率的上界。这些上界表明，自适应策略在检测没有好臂的时候比均匀采样更高效。其次，当APGAI与一个停止规则结合时，我们证明了在任何置信水平下的预期采样复杂性的上界。最后，我们展示了APGAI在合成数据和真实世界数据上的良好实验性能。我们的工作为所有设置中的GAI问题提供了一个广泛的概述。

    In good arm identification (GAI), the goal is to identify one arm whose average performance exceeds a given threshold, referred to as good arm, if it exists. Few works have studied GAI in the fixed-budget setting, when the sampling budget is fixed beforehand, or the anytime setting, when a recommendation can be asked at any time. We propose APGAI, an anytime and parameter-free sampling rule for GAI in stochastic bandits. APGAI can be straightforwardly used in fixed-confidence and fixed-budget settings. First, we derive upper bounds on its probability of error at any time. They show that adaptive strategies are more efficient in detecting the absence of good arms than uniform sampling. Second, when APGAI is combined with a stopping rule, we prove upper bounds on the expected sampling complexity, holding at any confidence level. Finally, we show good empirical performance of APGAI on synthetic and real-world data. Our work offers an extensive overview of the GAI problem in all settings.
    
[^3]: Transductive Learning的尖锐泛化：一种Transductive Local Rademacher Complexity方法

    Sharp Generalization of Transductive Learning: A Transductive Local Rademacher Complexity Approach. (arXiv:2309.16858v1 [stat.ML])

    [http://arxiv.org/abs/2309.16858](http://arxiv.org/abs/2309.16858)

    我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们利用变量的方差信息构建了TLRC，并将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。

    

    我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们的工作将传统的local rademacher complexity (LRC)的思想扩展到了transductive设置中，相对于典型的LRC方法在归纳设置中的分析有了相当大的变化。我们提出了一种基于Rademacher complex的局部化工具，可以应用于各种transductive learning问题，并在适当条件下得到了尖锐的界限。与LRC的发展类似，我们通过从独立变量的方差信息开始构建TLRC，将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。经过精心设计的...

    We introduce a new tool, Transductive Local Rademacher Complexity (TLRC), to analyze the generalization performance of transductive learning methods and motivate new transductive learning algorithms. Our work extends the idea of the popular Local Rademacher Complexity (LRC) to the transductive setting with considerable changes compared to the analysis of typical LRC methods in the inductive setting. We present a localized version of Rademacher complexity based tool wihch can be applied to various transductive learning problems and gain sharp bounds under proper conditions. Similar to the development of LRC, we build TLRC by starting from a sharp concentration inequality for independent variables with variance information. The prediction function class of a transductive learning model is then divided into pieces with a sub-root function being the upper bound for the Rademacher complexity of each piece, and the variance of all the functions in each piece is limited. A carefully designed 
    

