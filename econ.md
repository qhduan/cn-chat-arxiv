# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Comprehensive OOS Evaluation of Predictive Algorithms with Statistical Decision Theory](https://arxiv.org/abs/2403.11016) | 使用统计决策理论（SDT）进行全面的样本外（OOS）评估，取代当前机器学习（ML）研究中常见的K-fold和公共任务框架验证方法。 |
| [^2] | [Towards Generalizing Inferences from Trials to Target Populations](https://arxiv.org/abs/2402.17042) | 本研究在试图解决从试验结果推广到目标种群的外部有效性挑战方面取得了重要进展 |
| [^3] | [Characteristics and Predictive Modeling of Short-term Impacts of Hurricanes on the US Employment.](http://arxiv.org/abs/2307.13686) | 本研究研究了美国飓风影响后的短期就业变化，发现初始月份的县级就业变化平均较小，但在极端风暴之后可能出现大规模的就业损失。风暴后就业冲击与风暴危害指标和影响的地理空间细节呈负相关，并且非风暴因素也强烈影响短期就业变化。 |
| [^4] | [Long Story Short: Omitted Variable Bias in Causal Machine Learning.](http://arxiv.org/abs/2112.13398) | 在因果机器学习中，我们通过推导出遗漏变量偏差的尖锐上界，为广泛的线性泛函因果参数提供了一种简单而通用的方法。这种方法可以应用于许多传统的因果推断研究目标，并且仅取决于潜变量在结果和参数的Riesz表示器中所导致的额外变异。 |

# 详细

[^1]: 预测算法的全面OOS评估与统计决策理论

    Comprehensive OOS Evaluation of Predictive Algorithms with Statistical Decision Theory

    [https://arxiv.org/abs/2403.11016](https://arxiv.org/abs/2403.11016)

    使用统计决策理论（SDT）进行全面的样本外（OOS）评估，取代当前机器学习（ML）研究中常见的K-fold和公共任务框架验证方法。

    

    我们认为，应当使用统计决策理论（SDT）进行全面的样本外（OOS）评估，取代机器学习（ML）研究中当前的K-fold和公共任务框架验证的做法。SDT为在所有可能的（1）训练样本、（2）可能生成训练数据的人群和（3）预测利益人群之间进行全面的OOS评估提供了一个形式化框架。关于特征（3），我们强调SDT要求从业者直接面对未来可能不同于过去的可能性，并在构建预测算法时考虑从一个人群推断到另一个人群的可能需要。SDT在抽象上是简单的，但在实施时往往需要大量计算。我们讨论了当预测准确性通过均方误差或误分类率衡量时，SDT的可行实施进展。

    arXiv:2403.11016v1 Announce Type: new  Abstract: We argue that comprehensive out-of-sample (OOS) evaluation using statistical decision theory (SDT) should replace the current practice of K-fold and Common Task Framework validation in machine learning (ML) research. SDT provides a formal framework for performing comprehensive OOS evaluation across all possible (1) training samples, (2) populations that may generate training data, and (3) populations of prediction interest. Regarding feature (3), we emphasize that SDT requires the practitioner to directly confront the possibility that the future may not look like the past and to account for a possible need to extrapolate from one population to another when building a predictive algorithm. SDT is simple in abstraction, but it is often computationally demanding to implement. We discuss progress in tractable implementation of SDT when prediction accuracy is measured by mean square error or by misclassification rate. We summarize research st
    
[^2]: 通向从试验推广推理到目标种群的泛化

    Towards Generalizing Inferences from Trials to Target Populations

    [https://arxiv.org/abs/2402.17042](https://arxiv.org/abs/2402.17042)

    本研究在试图解决从试验结果推广到目标种群的外部有效性挑战方面取得了重要进展

    

    随机对照试验（RCTs）在产生内部有效估计方面起着至关重要的作用，而对扩展这些发现以获得外部有效估计至关重要，以促进更广泛的科学探究。本文探讨了应对这些外部有效性挑战的前沿，概括了2023年秋季在布朗大学计算与实验数学研究所（ICERM）举行的一次跨学科研讨会的精华。该研讨会汇集了来自社会科学、医学、公共卫生、统计学、计算机科学和教育等各个领域的专家，以解决每个学科在推断实验结果方面面临的独特障碍。我们的研究提出了三个关键贡献：我们整合正在进行的努力，突出了

    arXiv:2402.17042v1 Announce Type: cross  Abstract: Randomized Controlled Trials (RCTs) are pivotal in generating internally valid estimates with minimal assumptions, serving as a cornerstone for researchers dedicated to advancing causal inference methods. However, extending these findings beyond the experimental cohort to achieve externally valid estimates is crucial for broader scientific inquiry. This paper delves into the forefront of addressing these external validity challenges, encapsulating the essence of a multidisciplinary workshop held at the Institute for Computational and Experimental Research in Mathematics (ICERM), Brown University, in Fall 2023. The workshop congregated experts from diverse fields including social science, medicine, public health, statistics, computer science, and education, to tackle the unique obstacles each discipline faces in extrapolating experimental findings. Our study presents three key contributions: we integrate ongoing efforts, highlighting me
    
[^3]: 美国就业短期受飓风影响的特征和预测建模

    Characteristics and Predictive Modeling of Short-term Impacts of Hurricanes on the US Employment. (arXiv:2307.13686v1 [econ.EM])

    [http://arxiv.org/abs/2307.13686](http://arxiv.org/abs/2307.13686)

    本研究研究了美国飓风影响后的短期就业变化，发现初始月份的县级就业变化平均较小，但在极端风暴之后可能出现大规模的就业损失。风暴后就业冲击与风暴危害指标和影响的地理空间细节呈负相关，并且非风暴因素也强烈影响短期就业变化。

    

    本研究研究了美国飓风影响后的短期就业变化。对1990年至2021年的飓风事件进行分析发现，初始月份的县级就业变化平均较小，但在极端风暴之后可能出现大规模的就业损失（> 30％）。整体上的小幅变动部分是不同就业部门之间的补偿结果，例如建筑业和休闲酒店业。就业损失在服务业中相对明显。风暴后就业冲击与风暴危害指标（如极端风和降水量）以及影响的地理空间细节（如风暴实体距离）呈负相关。此外，县级特征等非风暴因素也强烈影响短期就业变化。研究结果对短期就业变化的预测建模具有启示作用，特别是在服务业和高—

    This study examines the short-term employment changes in the US after hurricane impacts. An analysis of hurricane events during 1990-2021 suggests that county-level employment changes in the initial month are small on average, though large employment losses (>30%) can occur after extreme storms. The overall small changes are partly a result of compensation among different employment sectors, such as the construction and leisure and hospitality sectors. Employment losses tend to be relatively pronounced in the service-providing industries. The post-storm employment shock is negatively correlated with the metrics of storm hazards (e.g., extreme wind and precipitation) and geospatial details of impacts (e.g., storm-entity distance). Additionally, non-storm factors such as county characteristics also strongly affect short-term employment changes. The findings inform predictive modeling of short-term employment changes, which shows promising skills for service-providing industries and high-
    
[^4]: 《长话短说：因果机器学习中的遗漏变量偏差》

    Long Story Short: Omitted Variable Bias in Causal Machine Learning. (arXiv:2112.13398v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2112.13398](http://arxiv.org/abs/2112.13398)

    在因果机器学习中，我们通过推导出遗漏变量偏差的尖锐上界，为广泛的线性泛函因果参数提供了一种简单而通用的方法。这种方法可以应用于许多传统的因果推断研究目标，并且仅取决于潜变量在结果和参数的Riesz表示器中所导致的额外变异。

    

    我们推导了一类广泛的因果参数的遗漏变量偏差的一般但简单的尖锐上界，这些参数可以被认定为结果的条件期望函数的线性泛函。这样的泛函包括许多因果推断研究中的传统调查目标，例如（加权）潜在结果的平均值、平均处理效应（包括子组效应，如对待处理对象的影响）、（加权）平均导数和来自协变量分布变化的策略效应 - 全部适用于一般的非参数因果模型。我们的构造依赖于目标泛函的Riesz-Fréchet表示。具体来说，我们展示了偏差上界仅取决于潜变量在结果和感兴趣参数的Riesz表示器中所创建的附加变化。此外，在许多重要情况下（例如平均处理效应和平均导数）

    We derive general, yet simple, sharp bounds on the size of the omitted variable bias for a broad class of causal parameters that can be identified as linear functionals of the conditional expectation function of the outcome. Such functionals encompass many of the traditional targets of investigation in causal inference studies, such as, for example, (weighted) average of potential outcomes, average treatment effects (including subgroup effects, such as the effect on the treated), (weighted) average derivatives, and policy effects from shifts in covariate distribution -- all for general, nonparametric causal models. Our construction relies on the Riesz-Frechet representation of the target functional. Specifically, we show how the bound on the bias depends only on the additional variation that the latent variables create both in the outcome and in the Riesz representer for the parameter of interest. Moreover, in many important cases (e.g, average treatment effects and avearage derivative
    

