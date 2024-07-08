# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys)](https://arxiv.org/abs/2404.00579) | 生成模型在现代推荐系统中展现出了强大的潜力，能够同时处理用户-项目交互历史、文本、图像和视频等复杂数据，为推荐系统带来了新的可能性。 |
| [^2] | [Detecting LLM-Assisted Writing in Scientific Communication: Are We There Yet?.](http://arxiv.org/abs/2401.16807) | 这项研究评估了四种先进的文本检测器对LLM辅助写作的表现，发现它们的性能不如一个简单的检测器。研究认为需要开发专门用于LLM辅助写作的特定检测器，以解决当前承认实践中的挑战。 |
| [^3] | [Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models.](http://arxiv.org/abs/2401.10690) | 本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。 |
| [^4] | [Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency.](http://arxiv.org/abs/2401.10545) | 该研究揭示了基于ChatGPT的推荐系统的偏见问题，并研究了提示设计策略对推荐质量的影响。实验结果显示，在RecLLMs中引入特定的系统角色和提示策略可以增强推荐的公平性和多样性，同时GPT-based模型倾向于推荐最新和更多样化的电影流派。 |
| [^5] | [Temporal Interest Network for Click-Through Rate Prediction.](http://arxiv.org/abs/2308.08487) | 本文提出了时间兴趣网络（TIN），用于捕捉行为与目标之间的四重语义和时间相关性，以预测点击率的效果和已有方法对这种相关性的学习程度尚不清楚。 |

# 详细

[^1]: 使用生成模型的现代推荐系统（Gen-RecSys）综述

    A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys)

    [https://arxiv.org/abs/2404.00579](https://arxiv.org/abs/2404.00579)

    生成模型在现代推荐系统中展现出了强大的潜力，能够同时处理用户-项目交互历史、文本、图像和视频等复杂数据，为推荐系统带来了新的可能性。

    

    传统的推荐系统（RS）通常使用用户-项目评分历史作为主要数据来源，协同过滤是主要方法之一。然而，生成模型最近已经具备了建模和采样复杂数据分布的能力，不仅可以涵盖用户-项目交互历史，还可以包括文本、图像和视频，为新颖的推荐任务解锁了丰富的数据。通过这篇全面的、多学科的调研，我们旨在探讨使用生成模型（Gen-RecSys）在RS中的关键进展，包括：基于交互驱动的生成模型的基础概述；大型语言模型（LLM）在生成推荐、检索和对话推荐中的应用；以及用于处理和生成RS中的图像和视频内容的多模态模型的整合。我们的整体视角使我们能够强调评估所需范式的重要性。

    arXiv:2404.00579v1 Announce Type: cross  Abstract: Traditional recommender systems (RS) have used user-item rating histories as their primary data source, with collaborative filtering being one of the principal methods. However, generative models have recently developed abilities to model and sample from complex data distributions, including not only user-item interaction histories but also text, images, and videos - unlocking this rich data for novel recommendation tasks. Through this comprehensive and multi-disciplinary survey, we aim to connect the key advancements in RS using Generative Models (Gen-RecSys), encompassing: a foundational overview of interaction-driven generative models; the application of large language models (LLM) for generative recommendation, retrieval, and conversational recommendation; and the integration of multimodal models for processing and generating image and video content in RS. Our holistic perspective allows us to highlight necessary paradigms for eval
    
[^2]: 在科学交流中检测LLM辅助写作：我们已经到达了吗？

    Detecting LLM-Assisted Writing in Scientific Communication: Are We There Yet?. (arXiv:2401.16807v1 [cs.IR])

    [http://arxiv.org/abs/2401.16807](http://arxiv.org/abs/2401.16807)

    这项研究评估了四种先进的文本检测器对LLM辅助写作的表现，发现它们的性能不如一个简单的检测器。研究认为需要开发专门用于LLM辅助写作的特定检测器，以解决当前承认实践中的挑战。

    

    大型语言模型（LLMs），如ChatGPT，在文本生成方面产生了重大影响，尤其是在写作辅助领域。尽管伦理考虑强调了在科学交流中透明地承认LLM的使用的重要性，但真实的承认仍然很少见。鼓励准确承认LLM辅助写作的一个潜在途径涉及使用自动检测器。我们对四个前沿的LLM生成文本检测器进行了评估，发现它们的性能不如一个简单的临时检测器，该检测器设计用于识别在LLM大量出现时的突然写作风格变化。我们认为，开发专门用于LLM辅助写作检测的专用检测器是必要的。这样的检测器可以在促进对LLM参与科学交流的更真实认可、解决当前承认实践中的挑战方面发挥关键作用。

    Large Language Models (LLMs), exemplified by ChatGPT, have significantly reshaped text generation, particularly in the realm of writing assistance. While ethical considerations underscore the importance of transparently acknowledging LLM use, especially in scientific communication, genuine acknowledgment remains infrequent. A potential avenue to encourage accurate acknowledging of LLM-assisted writing involves employing automated detectors. Our evaluation of four cutting-edge LLM-generated text detectors reveals their suboptimal performance compared to a simple ad-hoc detector designed to identify abrupt writing style changes around the time of LLM proliferation. We contend that the development of specialized detectors exclusively dedicated to LLM-assisted writing detection is necessary. Such detectors could play a crucial role in fostering more authentic recognition of LLM involvement in scientific communication, addressing the current challenges in acknowledgment practices.
    
[^3]: 超越RMSE和MAE：引入EAUC来揭示偏见和不公平的迪亚德回归模型中的隐藏因素

    Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models. (arXiv:2401.10690v1 [cs.LG])

    [http://arxiv.org/abs/2401.10690](http://arxiv.org/abs/2401.10690)

    本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。

    

    迪亚德回归模型用于预测一对实体的实值结果，在许多领域中都是基础的（例如，在推荐系统中预测用户对产品的评分），在许多其他领域中也有许多潜力但尚未深入探索（例如，在个性化药理学中近似确定患者的适当剂量）。本研究中，我们证明个体实体观察值分布的非均匀性导致了最先进模型中的严重偏见预测，偏向于实体的观察过去值的平均值，并在另类但同样重要的情况下提供比随机预测更差的预测能力。我们表明，全局错误度量标准如均方根误差（RMSE）和平均绝对误差（MAE）不足以捕捉到这种现象，我们将其命名为另类偏见，并引入另类-曲线下面积（EAUC）作为一个新的补充度量，可以在所有研究的模型中量化它。

    Dyadic regression models, which predict real-valued outcomes for pairs of entities, are fundamental in many domains (e.g. predicting the rating of a user to a product in Recommender Systems) and promising and under exploration in many others (e.g. approximating the adequate dosage of a drug for a patient in personalized pharmacology). In this work, we demonstrate that non-uniformity in the observed value distributions of individual entities leads to severely biased predictions in state-of-the-art models, skewing predictions towards the average of observed past values for the entity and providing worse-than-random predictive power in eccentric yet equally important cases. We show that the usage of global error metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) is insufficient to capture this phenomenon, which we name eccentricity bias, and we introduce Eccentricity-Area Under the Curve (EAUC) as a new complementary metric that can quantify it in all studied models
    
[^4]: 理解ChatGPT基推荐系统中的偏见：供应商公平性、时间稳定性和最新性研究

    Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency. (arXiv:2401.10545v1 [cs.IR])

    [http://arxiv.org/abs/2401.10545](http://arxiv.org/abs/2401.10545)

    该研究揭示了基于ChatGPT的推荐系统的偏见问题，并研究了提示设计策略对推荐质量的影响。实验结果显示，在RecLLMs中引入特定的系统角色和提示策略可以增强推荐的公平性和多样性，同时GPT-based模型倾向于推荐最新和更多样化的电影流派。

    

    该研究探讨了使用大型语言模型（RecLLMs）的推荐系统的细微能力和固有偏见，重点研究了基于ChatGPT的系统。研究了生成模型和传统协同过滤模型在电影推荐中的差异行为。本研究主要调查了提示设计策略及其对推荐质量的各个方面（包括准确性、供应商公平性、多样性、稳定性、流行类型和时效性）的影响。我们的实验分析表明，在RecLLMs中引入特定的“系统角色”和“提示策略”显著影响其性能。例如，基于角色的提示可以增强推荐的公平性和多样性，减轻流行偏见。我们发现，虽然基于GPT的模型并不总是能与传统协同过滤基线模型的性能匹配，但它们倾向于推荐更新、更多样化的电影流派。值得注意的是，GPT-base

    This study explores the nuanced capabilities and inherent biases of Recommender Systems using Large Language Models (RecLLMs), with a focus on ChatGPT-based systems. It studies into the contrasting behaviors of generative models and traditional collaborative filtering models in movie recommendations. The research primarily investigates prompt design strategies and their impact on various aspects of recommendation quality, including accuracy, provider fairness, diversity, stability, genre dominance, and temporal freshness (recency).  Our experimental analysis reveals that the introduction of specific 'system roles' and 'prompt strategies' in RecLLMs significantly influences their performance. For instance, role-based prompts enhance fairness and diversity in recommendations, mitigating popularity bias. We find that while GPT-based models do not always match the performance of CF baselines, they exhibit a unique tendency to recommend newer and more diverse movie genres. Notably, GPT-base
    
[^5]: 点击率预测的时间兴趣网络

    Temporal Interest Network for Click-Through Rate Prediction. (arXiv:2308.08487v1 [cs.IR])

    [http://arxiv.org/abs/2308.08487](http://arxiv.org/abs/2308.08487)

    本文提出了时间兴趣网络（TIN），用于捕捉行为与目标之间的四重语义和时间相关性，以预测点击率的效果和已有方法对这种相关性的学习程度尚不清楚。

    

    用户行为的历史是预测点击率最重要的特征之一，因为它们与目标项目具有强烈的语义和时间相关性。虽然已有文献分别研究了这些相关性，但尚未分析它们的组合，即行为语义、目标语义、行为时间和目标时间的四重相关性。这种相关性对性能的影响以及现有方法学习这种相关性的程度尚不清楚。为了填补这一空白，我们在实践中测量了四重相关性，并观察到直观而强大的四重模式。我们测量了几种代表性的用户行为方法的学习相关性，但令人惊讶的是，它们都没有学习到这样的模式，特别是时间模式。在本文中，我们提出了时间兴趣网络（TIN）来捕捉行为与目标之间的四重语义和时间相关性。

    The history of user behaviors constitutes one of the most significant characteristics in predicting the click-through rate (CTR), owing to their strong semantic and temporal correlation with the target item. While the literature has individually examined each of these correlations, research has yet to analyze them in combination, that is, the quadruple correlation of (behavior semantics, target semantics, behavior temporal, and target temporal). The effect of this correlation on performance and the extent to which existing methods learn it remain unknown. To address this gap, we empirically measure the quadruple correlation and observe intuitive yet robust quadruple patterns. We measure the learned correlation of several representative user behavior methods, but to our surprise, none of them learn such a pattern, especially the temporal one.  In this paper, we propose the Temporal Interest Network (TIN) to capture the quadruple semantic and temporal correlation between behaviors and th
    

