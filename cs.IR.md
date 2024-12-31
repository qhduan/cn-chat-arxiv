# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models.](http://arxiv.org/abs/2401.10690) | 本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。 |
| [^2] | [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent.](http://arxiv.org/abs/2304.09542) | 本文研究了生成性LLMs，如ChatGPT和GPT-4在信息检索中的相关性排名能力，实验结果表明，这些模型经适当指导后表现优异，有时甚至优于传统监督学习方法。将ChatGPT的排名能力提炼为专门模型在BEIR上的效果更优。 |

# 详细

[^1]: 超越RMSE和MAE：引入EAUC来揭示偏见和不公平的迪亚德回归模型中的隐藏因素

    Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models. (arXiv:2401.10690v1 [cs.LG])

    [http://arxiv.org/abs/2401.10690](http://arxiv.org/abs/2401.10690)

    本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。

    

    迪亚德回归模型用于预测一对实体的实值结果，在许多领域中都是基础的（例如，在推荐系统中预测用户对产品的评分），在许多其他领域中也有许多潜力但尚未深入探索（例如，在个性化药理学中近似确定患者的适当剂量）。本研究中，我们证明个体实体观察值分布的非均匀性导致了最先进模型中的严重偏见预测，偏向于实体的观察过去值的平均值，并在另类但同样重要的情况下提供比随机预测更差的预测能力。我们表明，全局错误度量标准如均方根误差（RMSE）和平均绝对误差（MAE）不足以捕捉到这种现象，我们将其命名为另类偏见，并引入另类-曲线下面积（EAUC）作为一个新的补充度量，可以在所有研究的模型中量化它。

    Dyadic regression models, which predict real-valued outcomes for pairs of entities, are fundamental in many domains (e.g. predicting the rating of a user to a product in Recommender Systems) and promising and under exploration in many others (e.g. approximating the adequate dosage of a drug for a patient in personalized pharmacology). In this work, we demonstrate that non-uniformity in the observed value distributions of individual entities leads to severely biased predictions in state-of-the-art models, skewing predictions towards the average of observed past values for the entity and providing worse-than-random predictive power in eccentric yet equally important cases. We show that the usage of global error metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) is insufficient to capture this phenomenon, which we name eccentricity bias, and we introduce Eccentricity-Area Under the Curve (EAUC) as a new complementary metric that can quantify it in all studied models
    
[^2]: 大型语言模型在信息检索中的排名能力研究——以ChatGPT为例

    Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent. (arXiv:2304.09542v1 [cs.CL])

    [http://arxiv.org/abs/2304.09542](http://arxiv.org/abs/2304.09542)

    本文研究了生成性LLMs，如ChatGPT和GPT-4在信息检索中的相关性排名能力，实验结果表明，这些模型经适当指导后表现优异，有时甚至优于传统监督学习方法。将ChatGPT的排名能力提炼为专门模型在BEIR上的效果更优。

    

    大型语言模型（LLMs）已经证明具有remarkable能力，能够将一些零样本语言任务推广至其他领域。本文研究了ChatGPT和GPT-4等生成性LLMs的相关性排名在信息检索方面的能力。实验结果显示，经过适当的指导，ChatGPT和GPT-4可以在流行的信息检索基准上取得竞争优势，甚至有时优于监督学习方法。特别地，GPT-4在TREC数据集上的平均nDCG上表现优于完全微调的monoT5-3B，BEIR数据集上的平均nDCG上优于monoT5-3B 2.3个点，低资源语言Mr.TyDi上的平均nDCG上优于monoT5-3B 2.7个点。随后，我们探讨了将ChatGPT的排名能力提炼为专门的模型的潜力。我们训练的小型专门模型（训练于10K个ChatGPT生成的数据）在BEIR上的表现优于在400K个MS MARCO注释数据上训练的monoT5。代码可在www.github.com/sunnwe上复现。

    Large Language Models (LLMs) have demonstrated a remarkable ability to generalize zero-shot to various language-related tasks. This paper focuses on the study of exploring generative LLMs such as ChatGPT and GPT-4 for relevance ranking in Information Retrieval (IR). Surprisingly, our experiments reveal that properly instructed ChatGPT and GPT-4 can deliver competitive, even superior results than supervised methods on popular IR benchmarks. Notably, GPT-4 outperforms the fully fine-tuned monoT5-3B on MS MARCO by an average of 2.7 nDCG on TREC datasets, an average of 2.3 nDCG on eight BEIR datasets, and an average of 2.7 nDCG on ten low-resource languages Mr.TyDi. Subsequently, we delve into the potential for distilling the ranking capabilities of ChatGPT into a specialized model. Our small specialized model that trained on 10K ChatGPT generated data outperforms monoT5 trained on 400K annotated MS MARCO data on BEIR. The code to reproduce our results is available at www.github.com/sunnwe
    

