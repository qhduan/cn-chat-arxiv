# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Speed, Accuracy, and Complexity](https://arxiv.org/abs/2403.11240) | 本文重新审视了使用反应时间来推断问题复杂性的有效性，提出了一个新方法来正确推断问题的复杂性，强调了在简单和复杂问题中决策速度都很快的特点，并探讨了反应时间与能力之间的非单调关系。 |
| [^2] | [LLM Voting: Human Choices and AI Collective Decision Making](https://arxiv.org/abs/2402.01766) | 本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并揭示了LLMs与人类在决策和偏见方面的差异。研究发现，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。 |
| [^3] | [Estimating Individual Responses when Tomorrow Matters.](http://arxiv.org/abs/2310.09105) | 本论文提出了一种基于回归的方法，用于估计个体对对立情况的反应。通过应用该方法于意大利的调查数据，研究发现考虑个体的信念对税收政策对消费决策的影响很重要。 |
| [^4] | [Synthetic Control Methods by Density Matching under Implicit Endogeneitiy.](http://arxiv.org/abs/2307.11127) | 本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。 |

# 详细

[^1]: 速度、准确性和复杂性

    Speed, Accuracy, and Complexity

    [https://arxiv.org/abs/2403.11240](https://arxiv.org/abs/2403.11240)

    本文重新审视了使用反应时间来推断问题复杂性的有效性，提出了一个新方法来正确推断问题的复杂性，强调了在简单和复杂问题中决策速度都很快的特点，并探讨了反应时间与能力之间的非单调关系。

    

    这篇论文重新审视了使用反应时间来推断问题复杂性的有效性。重新审视了一个经典的Wald模型，将信噪比作为问题复杂性的度量。虽然在问题复杂性上，选择质量是单调的，但期望的停止时间是倒U形的。事实上，决策在非常简单和非常复杂的问题中都很快：在简单问题中很快就能理解哪个选择是最好的，而在复杂问题中将会成本过高--这一洞察力也适用于一般的昂贵信息获取模型。这种非单调性也构成了反应时间与能力之间模糊关系的基础，即更高的能力意味着在非常复杂的问题中决策更慢，但在简单问题中决策更快。最后，本文提出了一种新方法，根据选择对激励变化的反应更多来正确推断问题的复杂性。

    arXiv:2403.11240v1 Announce Type: new  Abstract: This paper re-examines the validity of using response time to infer problem complexity. It revisits a canonical Wald model of optimal stopping, taking signal-to-noise ratio as a measure of problem complexity. While choice quality is monotone in problem complexity, expected stopping time is inverse $U$-shaped. Indeed decisions are fast in both very simple and very complex problems: in simple problems it is quick to understand which alternative is best, while in complex problems it would be too costly -- an insight which extends to general costly information acquisition models. This non-monotonicity also underlies an ambiguous relationship between response time and ability, whereby higher ability entails slower decisions in very complex problems, but faster decisions in simple problems. Finally, this paper proposes a new method to correctly infer problem complexity based on the finding that choices react more to changes in incentives in mo
    
[^2]: LLM投票：人类选择和AI集体决策

    LLM Voting: Human Choices and AI Collective Decision Making

    [https://arxiv.org/abs/2402.01766](https://arxiv.org/abs/2402.01766)

    本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并揭示了LLMs与人类在决策和偏见方面的差异。研究发现，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。

    

    本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并与人类投票模式进行了对比。我们的方法包括进行人类投票实验以建立人类偏好的基准，并与LLM代理进行平行实验。研究聚焦于集体结果和个体偏好，揭示了人类和LLMs之间在决策和固有偏见方面的差异。我们观察到LLMs在偏好多样性和一致性之间存在权衡，相比人类选民的多样偏好，LLMs有更趋向于一致选择的倾向。这一发现表明，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。

    This paper investigates the voting behaviors of Large Language Models (LLMs), particularly OpenAI's GPT4 and LLaMA2, and their alignment with human voting patterns. Our approach included a human voting experiment to establish a baseline for human preferences and a parallel experiment with LLM agents. The study focused on both collective outcomes and individual preferences, revealing differences in decision-making and inherent biases between humans and LLMs. We observed a trade-off between preference diversity and alignment in LLMs, with a tendency towards more uniform choices as compared to the diverse preferences of human voters. This finding indicates that LLMs could lead to more homogenized collective outcomes when used in voting assistance, underscoring the need for cautious integration of LLMs into democratic processes.
    
[^3]: 估计明天重要时的个体反应

    Estimating Individual Responses when Tomorrow Matters. (arXiv:2310.09105v1 [econ.EM])

    [http://arxiv.org/abs/2310.09105](http://arxiv.org/abs/2310.09105)

    本论文提出了一种基于回归的方法，用于估计个体对对立情况的反应。通过应用该方法于意大利的调查数据，研究发现考虑个体的信念对税收政策对消费决策的影响很重要。

    

    我们提出了一种基于回归的方法，用于估计个体的期望如何影响他们对对立情况的反应。我们提供了基于回归估计的平均偏效应恢复结构效应的条件。我们提出了一个依赖于主观信念数据的实用的三步估计方法。我们在一个消费和储蓄模型中说明了我们的方法，重点关注不仅改变当前收入而且影响对未来收入的信念的所得税的影响。通过将我们的方法应用于意大利的调查数据，我们发现考虑个体的信念对评估税收政策对消费决策的影响很重要。

    We propose a regression-based approach to estimate how individuals' expectations influence their responses to a counterfactual change. We provide conditions under which average partial effects based on regression estimates recover structural effects. We propose a practical three-step estimation method that relies on subjective beliefs data. We illustrate our approach in a model of consumption and saving, focusing on the impact of an income tax that not only changes current income but also affects beliefs about future income. By applying our approach to survey data from Italy, we find that considering individuals' beliefs matter for evaluating the impact of tax policies on consumption decisions.
    
[^4]: 通过密度匹配实现的合成对照方法下的隐式内生性问题

    Synthetic Control Methods by Density Matching under Implicit Endogeneitiy. (arXiv:2307.11127v1 [econ.EM])

    [http://arxiv.org/abs/2307.11127](http://arxiv.org/abs/2307.11127)

    本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。

    

    合成对照方法（SCMs）已成为比较案例研究中因果推断的重要工具。SCMs的基本思想是通过使用来自未处理单元的观测结果的加权和来估计经过处理单元的反事实结果。合成对照（SC）的准确性对于估计因果效应至关重要，因此，SC权重的估计成为了研究的焦点。在本文中，我们首先指出现有的SCMs存在一个隐式内生性问题，即未处理单元的结果与反事实结果模型中的误差项之间的相关性。我们展示了这个问题会对因果效应估计器产生偏差。然后，我们提出了一种基于密度匹配的新型SCM，假设经过处理单元的结果密度可以用未处理单元的密度的加权平均来近似（即混合模型）。基于这一假设，我们通过匹配来估计SC权重。

    Synthetic control methods (SCMs) have become a crucial tool for causal inference in comparative case studies. The fundamental idea of SCMs is to estimate counterfactual outcomes for a treated unit by using a weighted sum of observed outcomes from untreated units. The accuracy of the synthetic control (SC) is critical for estimating the causal effect, and hence, the estimation of SC weights has been the focus of much research. In this paper, we first point out that existing SCMs suffer from an implicit endogeneity problem, which is the correlation between the outcomes of untreated units and the error term in the model of a counterfactual outcome. We show that this problem yields a bias in the causal effect estimator. We then propose a novel SCM based on density matching, assuming that the density of outcomes of the treated unit can be approximated by a weighted average of the densities of untreated units (i.e., a mixture model). Based on this assumption, we estimate SC weights by matchi
    

