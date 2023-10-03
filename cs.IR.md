# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fairness for All: Investigating Harms to Within-Group Individuals in Producer Fairness Re-ranking Optimization -- A Reproducibility Study.](http://arxiv.org/abs/2309.09277) | 本研究复现了先前的生产者公平重新排名（PFR）方法，并发现它们对于冷门物品造成了显著的伤害，在优势和劣势组中存在公平差距。 |
| [^2] | [Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging.](http://arxiv.org/abs/2309.01026) | 该论文提出了一种利用生成型AI领域的新技术进行零样本推荐的方法，通过将多模态输入转化为文本描述，并利用预训练的语言模型计算语义嵌入，实现了对非平稳内容的推荐。在合成的多模态暗示环境中进行实验证明了该方法的有效性。 |
| [^3] | [NS4AR: A new, focused on sampling areas sampling method in graphical recommendation Systems.](http://arxiv.org/abs/2307.07321) | 本文提出了一种新的图形推荐系统采样方法，通过对样本区域进行划分并使用AdaSim对区域赋予不同的权重，形成正负样本集，并提出了一个子集选择模型来缩小核心负样本的数量。 |
| [^4] | [Data augmentation for recommender system: A semi-supervised approach using maximum margin matrix factorization.](http://arxiv.org/abs/2306.13050) | 本研究提出了一种基于最大边际矩阵分解的半监督方法来增广和细化协同过滤算法的评级预测。该方法利用自我训练来评估评分的置信度，并通过系统的数据增广策略来提高算法性能。 |
| [^5] | [Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models.](http://arxiv.org/abs/2306.08018) | Mol-Instructions是一个专门为生物分子领域设计的综合指令数据集，可以显著提高大语言模型在生物领域中的适应能力和认知敏锐度。 |

# 详细

[^1]: 所有人的公平性：探究生产者公平重新排名优化中对组内个体的伤害--一项可复现的研究

    Fairness for All: Investigating Harms to Within-Group Individuals in Producer Fairness Re-ranking Optimization -- A Reproducibility Study. (arXiv:2309.09277v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2309.09277](http://arxiv.org/abs/2309.09277)

    本研究复现了先前的生产者公平重新排名（PFR）方法，并发现它们对于冷门物品造成了显著的伤害，在优势和劣势组中存在公平差距。

    

    推荐系统被广泛用于为用户提供个性化推荐。最近的研究表明，推荐系统可能存在不同类型的偏见，如流行度偏见，导致生产者群体之间的推荐曝光分布不均衡。为了减轻这种情况，研究者提出了以生产者为中心的公平重新排名（PFR）方法，以确保各组之间的推荐效用公平。然而，这些方法忽视了它们对与冷门物品（即与用户的交互很少或没有交互的物品）相关的组内个体可能造成的伤害。本研究复现了先前的PFR方法，并显示它们在冷门物品上造成了显著的伤害，导致这些物品在优势和劣势组中出现了公平差距。令人惊讶的是，不公平的基准推荐模型给予这些冷门个体更多的曝光机会，尽管总体上看起来是不公平的。为了解决这个问题，

    Recommender systems are widely used to provide personalized recommendations to users. Recent research has shown that recommender systems may be subject to different types of biases, such as popularity bias, leading to an uneven distribution of recommendation exposure among producer groups. To mitigate this, producer-centered fairness re-ranking (PFR) approaches have been proposed to ensure equitable recommendation utility across groups. However, these approaches overlook the harm they may cause to within-group individuals associated with colder items, which are items with few or no interactions.  This study reproduces previous PFR approaches and shows that they significantly harm colder items, leading to a fairness gap for these items in both advantaged and disadvantaged groups. Surprisingly, the unfair base recommendation models were providing greater exposure opportunities to these individual cold items, even though at the group level, they appeared to be unfair. To address this issu
    
[^2]: 使用预训练的大型语言模型进行多模态暗示的零样本推荐

    Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging. (arXiv:2309.01026v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2309.01026](http://arxiv.org/abs/2309.01026)

    该论文提出了一种利用生成型AI领域的新技术进行零样本推荐的方法，通过将多模态输入转化为文本描述，并利用预训练的语言模型计算语义嵌入，实现了对非平稳内容的推荐。在合成的多模态暗示环境中进行实验证明了该方法的有效性。

    

    我们提出了一种利用生成型人工智能领域最新进展的方法，用于零样本推荐多模态非平稳内容。我们建议将不同模态的输入渲染为文本描述，并利用预训练的LLM计算语义嵌入获取它们的数值表示。一旦获得所有内容项的统一表示，可以通过计算适当的相似度度量来进行推荐，而无需进行额外的学习。我们在一个合成的多模态暗示环境中演示了我们的方法，其中输入包括表格、文本和视觉数据。

    We present a method for zero-shot recommendation of multimodal non-stationary content that leverages recent advancements in the field of generative AI. We propose rendering inputs of different modalities as textual descriptions and to utilize pre-trained LLMs to obtain their numerical representations by computing semantic embeddings. Once unified representations of all content items are obtained, the recommendation can be performed by computing an appropriate similarity metric between them without any additional learning. We demonstrate our approach on a synthetic multimodal nudging environment, where the inputs consist of tabular, textual, and visual data.
    
[^3]: NS4AR: 一种新的、专注于采样区域的图形推荐系统采样方法

    NS4AR: A new, focused on sampling areas sampling method in graphical recommendation Systems. (arXiv:2307.07321v1 [cs.IR])

    [http://arxiv.org/abs/2307.07321](http://arxiv.org/abs/2307.07321)

    本文提出了一种新的图形推荐系统采样方法，通过对样本区域进行划分并使用AdaSim对区域赋予不同的权重，形成正负样本集，并提出了一个子集选择模型来缩小核心负样本的数量。

    

    图形推荐系统的有效性取决于负采样的数量和质量。本文选择了一些典型的推荐系统模型，并将这些模型上的一些最新的负采样策略作为基线。基于典型的图形推荐模型，我们将样本区域划分为指定的n个区域，并使用AdaSim对这些区域赋予不同的权重，形成正样本集和负样本集。由于负样本的数量和重要性，我们还提出了一个子集选择模型来缩小核心负样本。

    The effectiveness of graphical recommender system depends on the quantity and quality of negative sampling. This paper selects some typical recommender system models, as well as some latest negative sampling strategies on the models as baseline. Based on typical graphical recommender model, we divide sample region into assigned-n areas and use AdaSim to give different weight to these areas to form positive set and negative set. Because of the volume and significance of negative items, we also proposed a subset selection model to narrow the core negative samples.
    
[^4]: 推荐系统的数据增广：一种基于最大边际矩阵分解的半监督方法

    Data augmentation for recommender system: A semi-supervised approach using maximum margin matrix factorization. (arXiv:2306.13050v1 [cs.IR])

    [http://arxiv.org/abs/2306.13050](http://arxiv.org/abs/2306.13050)

    本研究提出了一种基于最大边际矩阵分解的半监督方法来增广和细化协同过滤算法的评级预测。该方法利用自我训练来评估评分的置信度，并通过系统的数据增广策略来提高算法性能。

    

    协同过滤已成为推荐系统开发的常用方法，其中，根据用户的过去喜好和其他用户的可用偏好信息预测其对新物品的评分。尽管CF方法很受欢迎，但其性能通常受观察到的条目的稀疏性的极大限制。本研究探讨最大边际矩阵分解（MMMF）的数据增广和细化方面，该方法是广泛接受的用于评级预测的CF技术，之前尚未进行研究。我们利用CF算法的固有特性来评估单个评分的置信度，并提出了一种基于自我训练的半监督评级增强方法。我们假设任何CF算法的预测低置信度是由于训练数据的某些不足，因此，通过采用系统的数据增广策略，可以提高算法的性能。

    Collaborative filtering (CF) has become a popular method for developing recommender systems (RS) where ratings of a user for new items is predicted based on her past preferences and available preference information of other users. Despite the popularity of CF-based methods, their performance is often greatly limited by the sparsity of observed entries. In this study, we explore the data augmentation and refinement aspects of Maximum Margin Matrix Factorization (MMMF), a widely accepted CF technique for the rating predictions, which have not been investigated before. We exploit the inherent characteristics of CF algorithms to assess the confidence level of individual ratings and propose a semi-supervised approach for rating augmentation based on self-training. We hypothesize that any CF algorithm's predictions with low confidence are due to some deficiency in the training data and hence, the performance of the algorithm can be improved by adopting a systematic data augmentation strategy
    
[^5]: Mol-Instructions: 一个大规模生物分子指令数据集，为大语言模型提供支持

    Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models. (arXiv:2306.08018v1 [q-bio.QM])

    [http://arxiv.org/abs/2306.08018](http://arxiv.org/abs/2306.08018)

    Mol-Instructions是一个专门为生物分子领域设计的综合指令数据集，可以显著提高大语言模型在生物领域中的适应能力和认知敏锐度。

    

    大语言模型（LLM）以其卓越的任务处理能力和创新的输出，在许多领域推动了重大进展。然而，它们在生物分子研究等专业领域的熟练应用还受到限制。为了解决这个挑战，我们介绍了Mol-Instructions，这是一个经过精心策划、专门针对生物分子领域设计的综合指令数据集。Mol-Instructions由三个关键组成部分组成：分子导向指令、蛋白质导向指令和生物分子文本指令，每个部分都被策划用于增强LLM对生物分子特性和行为的理解和预测能力。通过对代表性LLM的广泛指令调整实验，我们强调了Mol-Instructions在增强大模型在生物分子研究复杂领域内的适应能力和认知敏锐度方面的潜力，从而促进生物分子领域的进一步发展。

    Large Language Models (LLMs), with their remarkable task-handling capabilities and innovative outputs, have catalyzed significant advancements across a spectrum of fields. However, their proficiency within specialized domains such as biomolecular studies remains limited. To address this challenge, we introduce Mol-Instructions, a meticulously curated, comprehensive instruction dataset expressly designed for the biomolecular realm. Mol-Instructions is composed of three pivotal components: molecule-oriented instructions, protein-oriented instructions, and biomolecular text instructions, each curated to enhance the understanding and prediction capabilities of LLMs concerning biomolecular features and behaviors. Through extensive instruction tuning experiments on the representative LLM, we underscore the potency of Mol-Instructions to enhance the adaptability and cognitive acuity of large models within the complex sphere of biomolecular studies, thereby promoting advancements in the biomol
    

