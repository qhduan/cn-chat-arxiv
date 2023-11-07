# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Evaluation of feature selection performance for identification of best effective technical indicators on stock market price prediction.](http://arxiv.org/abs/2310.09903) | 本研究评估了特征选择方法在股市价格预测中的性能，通过选择最佳的技术指标组合来实现最少误差的预测。研究结果表明，不同的包装器特征选择方法在不同的机器学习方法中具有不同的表现。 |
| [^2] | [Enhancing Financial Sentiment Analysis via Retrieval Augmented Large Language Models.](http://arxiv.org/abs/2310.04027) | 本文通过引入检索增强型的大型语言模型框架，提升金融情感分析的效果，并解决了传统模型在参数规模和训练数据范围方面的限制。 |
| [^3] | [The Emergence of Economic Rationality of GPT.](http://arxiv.org/abs/2305.12763) | 本文研究了GPT在经济理性方面的能力，通过指示其在4个领域中做出预算决策，发现GPT决策基本上是合理的，并且比人类更具有理性。 |
| [^4] | [Why Are Immigrants Always Accused of Stealing People's Jobs?.](http://arxiv.org/abs/2303.13319) | 移民在一个配给职位的匹配模型中可能降低本地工人的就业率，但移民对本地福利的总体影响取决于劳动力市场的状态，当劳动力市场繁荣时，移民的影响可能是正面的。 |
| [^5] | [Validation of machine learning based scenario generators.](http://arxiv.org/abs/2301.12719) | 本研究讨论了基于机器学习的场景生成器的验证，提出了两个额外的验证任务：检查风险因素之间的依赖关系和检测不希望出现的记忆效应。结论表明，在这个上下文中，这些措施能够产生合理的结果，并可以用于验证和优化模型。 |

# 详细

[^1]: 评估特征选择在股市价格预测中的性能，以确定最有效的技术指标

    Evaluation of feature selection performance for identification of best effective technical indicators on stock market price prediction. (arXiv:2310.09903v1 [q-fin.ST])

    [http://arxiv.org/abs/2310.09903](http://arxiv.org/abs/2310.09903)

    本研究评估了特征选择方法在股市价格预测中的性能，通过选择最佳的技术指标组合来实现最少误差的预测。研究结果表明，不同的包装器特征选择方法在不同的机器学习方法中具有不同的表现。

    

    鉴于技术指标对股市预测的影响，特征选择对选择最佳指标至关重要。一种考虑在特征选择过程中模型性能的特征选择方法是包装器特征选择方法。本研究旨在通过特征选择鉴定出最少误差的预测股市价格的最佳股市指标组合。为评估包装器特征选择技术对股市预测的影响，本文在过去10年苹果公司的数据上使用了10个评估器和123个技术指标进行了SFS和SBS的考察。此外，通过提出的方法，将由3天时间窗口创建的数据转化为适用于回归方法的输入。从观察结果可以得出：（1）每种包装器特征选择方法在不同的机器学习方法中具有不同的结果，每种方法在不同的预测准确性上也有所不同。

    Due to the influence of many factors, including technical indicators on stock market prediction, feature selection is important to choose the best indicators. One of the feature selection methods that consider the performance of models during feature selection is the wrapper feature selection method. The aim of this research is to identify a combination of the best stock market indicators through feature selection to predict the stock market price with the least error. In order to evaluate the impact of wrapper feature selection techniques on stock market prediction, in this paper SFS and SBS with 10 estimators and 123 technical indicators have been examined on the last 10 years of Apple Company. Also, by the proposed method, the data created by the 3-day time window were converted to the appropriate input for regression methods. Based on the results observed: (1) Each wrapper feature selection method has different results with different machine learning methods, and each method is mor
    
[^2]: 通过检索增强的大型语言模型提升金融情感分析

    Enhancing Financial Sentiment Analysis via Retrieval Augmented Large Language Models. (arXiv:2310.04027v1 [cs.CL])

    [http://arxiv.org/abs/2310.04027](http://arxiv.org/abs/2310.04027)

    本文通过引入检索增强型的大型语言模型框架，提升金融情感分析的效果，并解决了传统模型在参数规模和训练数据范围方面的限制。

    

    金融情感分析对于估值和投资决策至关重要。然而，传统的自然语言处理模型受其参数规模和训练数据集范围的限制，其泛化能力和在该领域的有效性受到了限制。最近，以广泛语料库进行预训练的大型语言模型（LLMs）由于其令人称赞的零样本能力，在各种自然语言处理任务中展示了优越的性能。然而，直接将LLMs应用于金融情感分析存在挑战：LLMs的预训练目标与情感标签预测之间的差异可能会 compromise其预测性能。此外，金融新闻的简洁性，常常缺乏足够的上下文，也可能会显著降低LLMs的情感分析可靠性。为了解决这些挑战，我们提出了一个用于金融情感分析的检索增强型LLMs框架。

    Financial sentiment analysis is critical for valuation and investment decision-making. Traditional NLP models, however, are limited by their parameter size and the scope of their training datasets, which hampers their generalization capabilities and effectiveness in this field. Recently, Large Language Models (LLMs) pre-trained on extensive corpora have demonstrated superior performance across various NLP tasks due to their commendable zero-shot abilities. Yet, directly applying LLMs to financial sentiment analysis presents challenges: The discrepancy between the pre-training objective of LLMs and predicting the sentiment label can compromise their predictive performance. Furthermore, the succinct nature of financial news, often devoid of sufficient context, can significantly diminish the reliability of LLMs' sentiment analysis. To address these challenges, we introduce a retrieval-augmented LLMs framework for financial sentiment analysis. This framework includes an instruction-tuned L
    
[^3]: GPT的经济理性出现

    The Emergence of Economic Rationality of GPT. (arXiv:2305.12763v1 [econ.GN])

    [http://arxiv.org/abs/2305.12763](http://arxiv.org/abs/2305.12763)

    本文研究了GPT在经济理性方面的能力，通过指示其在4个领域中做出预算决策，发现GPT决策基本上是合理的，并且比人类更具有理性。

    

    随着像GPT这样的大型语言模型越来越普遍，评估它们在语言处理之外的能力至关重要。本文通过指示GPT在风险、时间、社交和食品偏好的四个领域中进行预算决策来研究GPT的经济理性。我们通过评估GPT决策与古典揭示偏好理论中的效用最大化一致性来衡量经济理性。我们发现GPT在每个领域的决策基本上是合理的，并且表现出比文献报道的人类更高的理性得分。我们还发现，理性得分对于随机程度和人口统计学设置（如年龄和性别）是稳健的，但对基于选择情境的语言框架的上下文敏感。这些结果表明了LLM作出良好决策的潜力，以及需要进一步了解它们的能力、局限性和基本机制。

    As large language models (LLMs) like GPT become increasingly prevalent, it is essential that we assess their capabilities beyond language processing. This paper examines the economic rationality of GPT by instructing it to make budgetary decisions in four domains: risk, time, social, and food preferences. We measure economic rationality by assessing the consistency of GPT decisions with utility maximization in classic revealed preference theory. We find that GPT decisions are largely rational in each domain and demonstrate higher rationality scores than those of humans reported in the literature. We also find that the rationality scores are robust to the degree of randomness and demographic settings such as age and gender, but are sensitive to contexts based on the language frames of the choice situations. These results suggest the potential of LLMs to make good decisions and the need to further understand their capabilities, limitations, and underlying mechanisms.
    
[^4]: 为什么移民总被指责窃取人们的工作?

    Why Are Immigrants Always Accused of Stealing People's Jobs?. (arXiv:2303.13319v1 [econ.GN])

    [http://arxiv.org/abs/2303.13319](http://arxiv.org/abs/2303.13319)

    移民在一个配给职位的匹配模型中可能降低本地工人的就业率，但移民对本地福利的总体影响取决于劳动力市场的状态，当劳动力市场繁荣时，移民的影响可能是正面的。

    

    移民总是被指责窃取人们的工作。然而，在劳动力市场的新古典模型中，每个人都有工作可做，也没有工作可被窃取（没有失业，因此想工作的人都可以工作）。在标准匹配模型中，存在一些失业，但由于劳动力需求完全弹性，因此新进入劳动力市场的人被吸收时不会影响求职者的前景。再次说明，当移民到达时没有工作会被窃取。本文显示，在一个具有就业配给的匹配模型中，移民的进入会降低本地工人的就业率。此外，当劳动力市场不景气时，就业率的降幅更大，因为那时工作更加稀缺。因为移民降低了劳动力市场的紧张程度，使得公司更容易招聘，并改善公司利润。移民对本地福利的总体影响取决于劳动力市场的状态。当劳动力市场出现衰退时总体影响始终是负面的，并且当劳动力市场繁荣时可能是正面的。

    Immigrants are always accused of stealing people's jobs. Yet, in a neoclassical model of the labor market, there are jobs for everybody and no jobs to steal. (There is no unemployment, so anybody who wants to work can work.) In standard matching models, there is some unemployment, but labor demand is perfectly elastic so new entrants into the labor force are absorbed without affecting jobseekers' prospects. Once again, no jobs are stolen when immigrants arrive. This paper shows that in a matching model with job rationing, in contrast, the entry of immigrants reduces the employment rate of native workers. Moreover, the reduction in employment rate is sharper when the labor market is depressed -- because jobs are more scarce then. Because immigration reduces labor-market tightness, it makes it easier for firms to recruit and improves firm profits. The overall effect of immigration on native welfare depends on the state of the labor market. It is always negative when the labor market is i
    
[^5]: 机器学习场景生成器的验证

    Validation of machine learning based scenario generators. (arXiv:2301.12719v2 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2301.12719](http://arxiv.org/abs/2301.12719)

    本研究讨论了基于机器学习的场景生成器的验证，提出了两个额外的验证任务：检查风险因素之间的依赖关系和检测不希望出现的记忆效应。结论表明，在这个上下文中，这些措施能够产生合理的结果，并可以用于验证和优化模型。

    

    机器学习方法在使用场景生成器开发内部模型方面变得越来越重要。在 Solvency 2 下，内部模型需要经过验证，一个重要问题是这些数据驱动模型的验证与传统理论模型的区别在哪些方面。以市场风险为例，我们讨论了两个额外的验证任务的必要性：一个是检查所使用风险因素之间的依赖关系，一个是检测不希望出现的记忆效应。第一个任务是必要的，因为在这种新方法中，依赖关系不是从金融数学理论中推导出来的。后一个任务出现在机器学习模型仅重复经验数据而不生成新场景的情况下。然后，这些措施被应用于基于机器学习的经济场景生成器。结果表明，在这个上下文中，这些措施导致了合理的结果，并且可以用于验证和模型优化。

    Machine learning methods are getting more and more important in the development of internal models using scenario generation. As internal models under Solvency 2 have to be validated, an important question is in which aspects the validation of these data-driven models differs from a classical theory-based model. On the specific example of market risk, we discuss the necessity of two additional validation tasks: one to check the dependencies between the risk factors used and one to detect the unwanted memorizing effect. The first one is necessary because in this new method, the dependencies are not derived from a financial-mathematical theory. The latter one arises when the machine learning model only repeats empirical data instead of generating new scenarios. These measures are then applied for an machine learning based economic scenario generator. It is shown that those measures lead to reasonable results in this context and are able to be used for validation as well as for model opti
    

