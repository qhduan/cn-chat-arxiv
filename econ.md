# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Identifying Causal Effects in Information Provision Experiments.](http://arxiv.org/abs/2309.11387) | 信息提供实验用于确定信念如何因果地影响决策和行为。通过应用贝叶斯估计器，可以准确识别出（非加权的）平均部分效应。 |
| [^2] | [Heterogeneity-robust granular instruments.](http://arxiv.org/abs/2304.01273) | 本论文提出了一种新的估计方法——鲁棒性粒度仪器变量（RGIV）——可以处理单位级异质性和未知的冲击方差，适用于各种经济环境，比现有方法更具广泛适用性。 |
| [^3] | [Bootstrap inference in the presence of bias.](http://arxiv.org/abs/2208.02028) | 本研究考虑了在有偏估计量情况下的Bootstrap推断，并展示了通过正确实施预估修正方法可以得到有效推断。研究提出了两种预估修正的实现方法，并给出了通用的条件确保Bootstrap推断的有效性。根据五个具体例子的讨论，研究证明了提出方法的实际相关性和实施方法。 |
| [^4] | [Stronger Monotone Signaling Equilibrium.](http://arxiv.org/abs/2109.03370) | 本文研究了更强的单调信号平衡，提出了针对稳定匹配的竞争性信号均衡概念，并给出了满足 D1 准则的条件。在具有准线性效用函数的单一匹配市场上建立了唯一的更强单调平衡。 |

# 详细

[^1]: 信息提供实验中的因果效应识别

    Identifying Causal Effects in Information Provision Experiments. (arXiv:2309.11387v1 [econ.EM])

    [http://arxiv.org/abs/2309.11387](http://arxiv.org/abs/2309.11387)

    信息提供实验用于确定信念如何因果地影响决策和行为。通过应用贝叶斯估计器，可以准确识别出（非加权的）平均部分效应。

    

    信息提供实验是一种越来越流行的工具，用于确定信念如何因果地影响决策和行为。在基于负担信息获取的简单贝叶斯信念形成模型中，当这些信念对他们的决策至关重要时，人们形成精确的信念。先前信念的精确度控制着当他们接受新信息时他们的信念变化程度（即第一阶段的强度）。由于两阶段最小二乘法（TSLS）以权重与第一阶段的强度成比例的加权平均为目标，TSLS会过度加权具有较小因果效应的个体，并低估具有较大效应的个体，从而低估了信念对行为的平均部分效应。在所有参与者都接受新信息的实验设计中，贝叶斯更新意味着可以使用控制函数来确定（非加权的）平均部分效应。我将这个估计器应用于最近一项关于效应的研究。

    Information provision experiments are an increasingly popular tool to identify how beliefs causally affect decision-making and behavior. In a simple Bayesian model of belief formation via costly information acquisition, people form precise beliefs when these beliefs are important for their decision-making. The precision of prior beliefs controls how much their beliefs shift when they are shown new information (i.e., the strength of the first stage). Since two-stage least squares (TSLS) targets a weighted average with weights proportional to the strength of the first stage, TSLS will overweight individuals with smaller causal effects and underweight those with larger effects, thus understating the average partial effect of beliefs on behavior. In experimental designs where all participants are exposed to new information, Bayesian updating implies that a control function can be used to identify the (unweighted) average partial effect. I apply this estimator to a recent study of the effec
    
[^2]: 异质性鲁棒性的粒度仪器

    Heterogeneity-robust granular instruments. (arXiv:2304.01273v1 [econ.EM])

    [http://arxiv.org/abs/2304.01273](http://arxiv.org/abs/2304.01273)

    本论文提出了一种新的估计方法——鲁棒性粒度仪器变量（RGIV）——可以处理单位级异质性和未知的冲击方差，适用于各种经济环境，比现有方法更具广泛适用性。

    

    粒度仪器变量在经验宏观金融方面得到了快速发展。它们的吸引力在于它们适用于各种经济环境，如需求系统和溢出的估计。我提出了一种新的估计方法——鲁棒性粒度仪器变量（RGIV）——它不像GIV那样只是允许单位对总量变量产生的响应不同，而且还可处理单位级异质性和未知的冲击方差。其广泛适用性使得研究人员可以考虑和研究单位层面的差异。我还开发了一个超识别检验，评估RGIV与数据的兼容性，以及一个参数限制检验，评估同质系数假设的适当性。在模拟中，我证明RGIV产生可靠且信息丰富的置信区间。

    Granular instrumental variables have experienced sharp growth in empirical macro-finance. Their attraction lies in their applicability to a wide set of economic environments like demand systems and the estimation of spillovers. I propose a new estimator$\unicode{x2014}$called robust granular instrumental variables (RGIV)$\unicode{x2014}$that, unlike GIV, allows for heterogeneous responses across units to the aggregate variable, unknown shock variances, and does not rely on skewness of the size distribution of units. Its generality allows researchers to account for and study unit-level heterogeneity. I also develop an overidentification test that evaluates the RGIV's compatibility with the data and a parameter restriction test that evaluates the appropriateness of the homogeneous coefficient assumption. In simulations, I show that RGIV produces reliable and informative confidence intervals.
    
[^3]: 在偏差存在的情况下的Bootstrap推断

    Bootstrap inference in the presence of bias. (arXiv:2208.02028v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2208.02028](http://arxiv.org/abs/2208.02028)

    本研究考虑了在有偏估计量情况下的Bootstrap推断，并展示了通过正确实施预估修正方法可以得到有效推断。研究提出了两种预估修正的实现方法，并给出了通用的条件确保Bootstrap推断的有效性。根据五个具体例子的讨论，研究证明了提出方法的实际相关性和实施方法。

    

    我们考虑对（渐近）有偏估计量的Bootstrap推断。我们展示了即使偏差项无法一致估计，通过正确实施Bootstrap，仍然可以得到有效的推断。具体而言，我们展示了Beran（1987, 1988）的预估修正方法，原本旨在提供更高阶的改进，通过将原始的Bootstrap p值转化为渐近均匀随机变量，恢复了Bootstrap的有效性。我们提出了两种预估修正的实现方法（插入法和双重Bootstrap），并提供了一般高级条件，这些条件意味着Bootstrap推断的有效性。为了说明我们结果的实际相关性和实施方法，我们讨论了五个例子：（i）基于模型平均的目标参数推断；（ii）岭型正则化估计量；（iii）非参数回归；（iv）无穷方差数据的位置模型；（v）动态面板数据模型。

    We consider bootstrap inference for estimators which are (asymptotically) biased. We show that, even when the bias term cannot be consistently estimated, valid inference can be obtained by proper implementations of the bootstrap. Specifically, we show that the prepivoting approach of Beran (1987, 1988), originally proposed to deliver higher-order refinements, restores bootstrap validity by transforming the original bootstrap p-value into an asymptotically uniform random variable. We propose two different implementations of prepivoting (plug-in and double bootstrap), and provide general high-level conditions that imply validity of bootstrap inference. To illustrate the practical relevance and implementation of our results, we discuss five examples: (i) inference on a target parameter based on model averaging; (ii) ridge-type regularized estimators; (iii) nonparametric regression; (iv) a location model for infinite variance data; and (v) dynamic panel data models.
    
[^4]: 更强的单调信号平衡

    Stronger Monotone Signaling Equilibrium. (arXiv:2109.03370v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2109.03370](http://arxiv.org/abs/2109.03370)

    本文研究了更强的单调信号平衡，提出了针对稳定匹配的竞争性信号均衡概念，并给出了满足 D1 准则的条件。在具有准线性效用函数的单一匹配市场上建立了唯一的更强单调平衡。

    

    本文研究了单调信号平衡，其中平衡结果——行动、反应、信念（匹配模型中的匹配）——均在更强的集合序列中是单调的。我们证明了如果发送者的效用是单调超模型的，则对于一对发送者和接收者的博弈，如果通过 D1 准则（Cho 和 Kreps（1987）、Banks 和 Sobel(1987)）则纯策略完美贝叶斯均衡更强单调。我们引入了一种基于稳定匹配的单一匹配市场上的竞争性信号均衡（CSE）概念，其中发送者和接收者都是异构的连续变量。我们证明了如果发送者效用是单调超模型的话，如果接收者效用是弱单调超模型，那么 CSE 更强单调，当且仅当它通过 D1 准则。最后，在具有准线性效用函数的单一匹配市场上，在接收者可以采取任何可行反应的区间内，建立了一个唯一的更强单调平衡。

    We study monotone signaling equilibrium where equilibrium outcomes - actions, reactions, beliefs (and matching in a matching model) - are all monotone in the stronger set order. We show that if the sender's utility is monotone-supermodular, a pure-strategy perfect Bayesian equilibrium of games with one sender and one receiver is stronger monotone if and only if it passes Criterion D1 (Cho and Kreps (1987), Banks and Sobel (1987)). We introduce a notion of competitive signaling equilibrium (CSE) in one-to-one matching markets with a continuum of heterogeneous senders and receivers, based on the notion of stable matching. We show that if the sender utility is monotone-supermodular and the receiver's utility is weakly monotone-supermodular, a CSE is stronger monotone if and only if it passes Criterion D1. Finally, in one-to-one matching markets with quasilinear utilities, a unique stronger monotone equilibrium is established given any interval of feasible reactions that receivers can take
    

