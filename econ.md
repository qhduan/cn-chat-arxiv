# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Locally Optimal Best Arm Identification with a Fixed Budget.](http://arxiv.org/abs/2310.19788) | 该研究解决了识别具有最高预期效果的治疗方案的问题，并提出了具有固定预算的局部最优算法来降低错误识别的概率。 |
| [^2] | [Efficiency in Multiple-Type Housing Markets.](http://arxiv.org/abs/2308.14989) | 在多类型住房市场中，我们考虑了逐坐标效率和成对效率这两个较弱的效率属性，并展示了它们与个体合理性和策略无关性的兼容性。我们还提出了两种特定的机制：逐坐标顶级交易周期机制（cTTC）和捆绑顶级交易周期机制（bTTC）。 |
| [^3] | [Factor-augmented sparse MIDAS regression for nowcasting.](http://arxiv.org/abs/2306.13362) | 本文提出了一种因子增强的稀疏MIDAS回归模型的估计器，这种方法能够在COVID-19大流行期间提高对美国GDP增长率的预测精度，并提供了一种透明和可解释的方法来确定关键预测因子。 |
| [^4] | [Cross-Sectional Dynamics Under Network Structure: Theory and Macroeconomic Applications.](http://arxiv.org/abs/2211.13610) | 本文提出了一个网络结构下横向动力学的框架，可以用于估计动态网络效应和建模降维等应用。第一次应用该框架时，作者估计了部门生产力冲击如何沿着供应链传输并影响美国经济中部门价格的动态。分析表明，网络位置不仅可以解释经济变化的强度，还可以解释供应链的作用。 |
| [^5] | [Abadie's Kappa and Weighting Estimators of the Local Average Treatment Effect.](http://arxiv.org/abs/2204.07672) | 本文研究了基于 Abadie Kappa 定理的局部平均治疗效应的加权估计量的特性，并提出了一个归一化的 Abadie 估计值更为可取的观点。 |

# 详细

[^1]: 具有固定预算的局部最优最佳臂识别算法

    Locally Optimal Best Arm Identification with a Fixed Budget. (arXiv:2310.19788v1 [math.ST])

    [http://arxiv.org/abs/2310.19788](http://arxiv.org/abs/2310.19788)

    该研究解决了识别具有最高预期效果的治疗方案的问题，并提出了具有固定预算的局部最优算法来降低错误识别的概率。

    

    本研究探讨了识别最佳治疗方案的问题，即具有最高预期效果的治疗方案。我们旨在通过降低错误识别的概率来确定最佳治疗方案，这一问题在许多研究领域中已被探索，包括最佳臂识别（Best Arm Identification，BAI）和序列优化。在我们的实验中，治疗分配的轮数是固定的。在每一轮中，决策者将一种治疗方案分配给一个实验单元，并观察相应的结果，该结果遵循不同治疗方案之间方差不同的高斯分布。在实验结束时，我们根据观察结果推荐一种治疗方案作为最佳治疗方案的估计值。决策者的目标是设计一个实验，使错误识别最佳治疗方案的概率最小化。基于这一目标，我们开发了误识别概率的下界。

    This study investigates the problem of identifying the best treatment arm, a treatment arm with the highest expected outcome. We aim to identify the best treatment arm with a lower probability of misidentification, which has been explored under various names across numerous research fields, including \emph{best arm identification} (BAI) and ordinal optimization. In our experiments, the number of treatment-allocation rounds is fixed. In each round, a decision-maker allocates a treatment arm to an experimental unit and observes a corresponding outcome, which follows a Gaussian distribution with a variance different among treatment arms. At the end of the experiment, we recommend one of the treatment arms as an estimate of the best treatment arm based on the observations. The objective of the decision-maker is to design an experiment that minimizes the probability of misidentifying the best treatment arm. With this objective in mind, we develop lower bounds for the probability of misident
    
[^2]: 多类型住房市场的效率

    Efficiency in Multiple-Type Housing Markets. (arXiv:2308.14989v1 [econ.TH])

    [http://arxiv.org/abs/2308.14989](http://arxiv.org/abs/2308.14989)

    在多类型住房市场中，我们考虑了逐坐标效率和成对效率这两个较弱的效率属性，并展示了它们与个体合理性和策略无关性的兼容性。我们还提出了两种特定的机制：逐坐标顶级交易周期机制（cTTC）和捆绑顶级交易周期机制（bTTC）。

    

    我们考虑多类型住房市场（Moulin, 1995），将Shapley-Scarf住房市场（Shapley and Scarf, 1974）从一维扩展到高维。在这个模型中，帕累托效率与个体合理性和策略无关性不兼容（Konishi et al., 2001）。因此，我们考虑两个较弱的效率属性：逐坐标效率和成对效率。我们证明了这两个属性都（i）与个体合理性和策略无关性相兼容，并且（ii）帮助我们识别出两种特定机制。更明确地说，在各种偏好配置的定义域上，与其他已广泛研究的属性（个体合理性，策略无关性和非霸权性）合作，逐坐标效率和成对效率分别刻画了两种顶级交易周期机制（TTC）的扩展：逐坐标顶级交易周期机制（cTTC）和捆绑顶级交易周期机制（bTTC）。此外，我们提出se

    We consider multiple-type housing markets (Moulin, 1995), which extend Shapley-Scarf housing markets (Shapley and Scarf, 1974) from one dimension to higher dimensions. In this model, Pareto efficiency is incompatible with individual rationality and strategy-proofness (Konishi et al., 2001). Therefore, we consider two weaker efficiency properties: coordinatewise efficiency and pairwise efficiency. We show that these two properties both (i) are compatible with individual rationality and strategy-proofness, and (ii) help us to identify two specific mechanisms. To be more precise, on various domains of preference profiles, together with other well-studied properties (individual rationality, strategy-proofness, and non-bossiness), coordinatewise efficiency and pairwise efficiency respectively characterize two extensions of the top-trading-cycles mechanism (TTC): the coordinatewise top-trading-cycles mechanism (cTTC) and the bundle top-trading-cycles mechanism (bTTC). Moreover, we propose se
    
[^3]: 因子增强的稀疏MIDAS回归在现在预测方面的应用

    Factor-augmented sparse MIDAS regression for nowcasting. (arXiv:2306.13362v1 [econ.EM])

    [http://arxiv.org/abs/2306.13362](http://arxiv.org/abs/2306.13362)

    本文提出了一种因子增强的稀疏MIDAS回归模型的估计器，这种方法能够在COVID-19大流行期间提高对美国GDP增长率的预测精度，并提供了一种透明和可解释的方法来确定关键预测因子。

    

    本文旨在调查是否通过将（估计的）因子添加到稀疏回归中可以改善现在预测质量。我们提出了一种因子增强的稀疏MIDAS回归模型的估计器，并在时间序列情况下推导了估计器的收敛率，考虑了$\tau$混合过程和重尾分布。该新技术应用于预测美国国内生产总值增长率，揭示了一些重要发现。我们的新技术在2008年Q1到2022年Q2的特定时期内显著提高了现在预测的质量，相对于稀疏回归和无因子增强回归基准。这种提高在COVID流行期间尤其明显，表明该模型能够捕捉到疫情引入的特定动态。有趣的是，估计的稀疏因子提供了一种简洁而有效的方式来总结高维信息以进行现在预测，而稀疏系数提供了一种透明且可解释的方法来确定推动现在预测的关键预测因子。

    GDP nowcasting commonly employs either sparse regression or a dense approach based on factor models, which differ in the way they extract information from high-dimensional datasets. This paper aims to investigate whether augmenting sparse regression with (estimated) factors can improve nowcasts. We propose an estimator for a factor-augmented sparse MIDAS regression model. The rates of convergence of the estimator are derived in a time series context, accounting for $\tau$-mixing processes and fat-tailed distributions.  The application of this new technique to nowcast US GDP growth reveals several key findings. Firstly, our novel technique significantly improves the quality of nowcasts compared to both sparse regression and plain factor-augmented regression benchmarks over a period period from 2008 Q1 to 2022 Q2. This improvement is particularly pronounced during the COVID pandemic, indicating the model's ability to capture the specific dynamics introduced by the pandemic. Interestingly
    
[^4]: 网络结构下的横向动力学: 理论及其宏观经济学应用

    Cross-Sectional Dynamics Under Network Structure: Theory and Macroeconomic Applications. (arXiv:2211.13610v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.13610](http://arxiv.org/abs/2211.13610)

    本文提出了一个网络结构下横向动力学的框架，可以用于估计动态网络效应和建模降维等应用。第一次应用该框架时，作者估计了部门生产力冲击如何沿着供应链传输并影响美国经济中部门价格的动态。分析表明，网络位置不仅可以解释经济变化的强度，还可以解释供应链的作用。

    

    许多经济学领域都涉及通过双边联系联系在一起的单位横向切面。本文提出了一个研究利用这种网络结构研究横向变量动态的框架。它是一个矢量自回归，其中创新仅通过双边联系在横向传输，并且可以容纳网络效应的高阶如何随时间累积的丰富模式。该模型可用于估计动态网络效应，其中网络由数据中的动态交叉相关性确定或推断。它还提供了一种建模(new_word)的降维技术，因为网络能够通过相对较少的非零双边链接汇总单位之间的复杂关系。

    Many environments in economics feature a cross-section of units linked by bilateral ties. I develop a framework for studying dynamics of cross-sectional variables exploiting this network structure. It is a vector autoregression in which innovations transmit cross-sectionally only via bilateral links and which can accommodate rich patterns of how network effects of higher order accumulate over time. The model can be used to estimate dynamic network effects, with the network given or inferred from dynamic cross-correlations in the data. It also offers a dimensionality-reduction technique for modeling (cross-sectional) processes, owing to networks' ability to summarize complex relations among units by relatively few non-zero bilateral links. In a first application, I estimate how sectoral productivity shocks transmit along supply chain linkages and affect dynamics of sectoral prices in the US economy. The analysis suggests that network positions can rationalize not only the strength of a 
    
[^5]: Abadie's Kappa 和权重估计局部平均治疗效应

    Abadie's Kappa and Weighting Estimators of the Local Average Treatment Effect. (arXiv:2204.07672v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2204.07672](http://arxiv.org/abs/2204.07672)

    本文研究了基于 Abadie Kappa 定理的局部平均治疗效应的加权估计量的特性，并提出了一个归一化的 Abadie 估计值更为可取的观点。

    

    本文研究了各种基于 Abadie (2003) Kappa 定理的局部平均治疗效应（LATE）的加权估计量的有限样本和渐进性质。我们的框架假设二元治疗和二元工具，但这仅在附加协变量的条件下才有效。我们认为，其中一个归一化的 Abadie 估计值在许多情况下更为可取。其他几个未归一化的估计值通常不满足对数变换下的比例不变性和平移不变性的性质，因此在估计对数下的 LATE 和一般情况下的结果变量的中心化时，会对测量单位敏感。另一方面，当不遵守处理的一面时，某些未归一化的估计值具有以非零数为下限的分母的优势。为了调和这些发现，我们证明了当相关实验的非遵从性较小时，一些加权估计量在精确到等价意义时具有相同的渐进性质。

    In this paper we study the finite sample and asymptotic properties of various weighting estimators of the local average treatment effect (LATE), several of which are based on Abadie's (2003) kappa theorem. Our framework presumes a binary treatment and a binary instrument, which may only be valid after conditioning on additional covariates. We argue that one of the Abadie estimators, which is weight normalized, is preferable in many contexts. Several other estimators, which are unnormalized, do not generally satisfy the properties of scale invariance with respect to the natural logarithm and translation invariance, thereby exhibiting sensitivity to the units of measurement when estimating the LATE in logs and the centering of the outcome variable more generally. On the other hand, when noncompliance is one-sided, certain unnormalized estimators have the advantage of being based on a denominator that is bounded away from zero. To reconcile these findings, we demonstrate that when the ins
    

