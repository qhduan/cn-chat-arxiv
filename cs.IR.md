# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [S+t-SNE - Bringing dimensionality reduction to data streams](https://arxiv.org/abs/2403.17643) | S+t-SNE是t-SNE算法的改进版本，在处理数据流时具有增量更新和盲目漂移管理的特点，能够实现高效的降维和信息可视化。 |
| [^2] | [CFaiRLLM: Consumer Fairness Evaluation in Large-Language Model Recommender System](https://arxiv.org/abs/2403.05668) | 这项研究引入了一个全面的评估框架 CFaiRLLM，旨在评估和减轻 RecLLMs 中消费者端的偏见 |

# 详细

[^1]: S+t-SNE - 将降维引入数据流

    S+t-SNE - Bringing dimensionality reduction to data streams

    [https://arxiv.org/abs/2403.17643](https://arxiv.org/abs/2403.17643)

    S+t-SNE是t-SNE算法的改进版本，在处理数据流时具有增量更新和盲目漂移管理的特点，能够实现高效的降维和信息可视化。

    

    我们提出了S+t-SNE，这是t-SNE算法的一种改进，旨在处理无限数据流。S+t-SNE的核心思想是随着新数据的到来逐步更新t-SNE嵌入，确保可扩展性和适应性，以处理流式场景。通过在每一步选择最重要的点，该算法确保可扩展性同时保持信息可视化。采用盲目方法进行漂移管理调整嵌入空间，促进不断可视化不断发展的数据动态。我们的实验评估证明了S+t-SNE的有效性和效率。结果突显了其在流式场景中捕捉模式的能力。我们希望我们的方法为研究人员和从业者提供一个实时工具，用于理解和解释高维数据。

    arXiv:2403.17643v1 Announce Type: new  Abstract: We present S+t-SNE, an adaptation of the t-SNE algorithm designed to handle infinite data streams. The core idea behind S+t-SNE is to update the t-SNE embedding incrementally as new data arrives, ensuring scalability and adaptability to handle streaming scenarios. By selecting the most important points at each step, the algorithm ensures scalability while keeping informative visualisations. Employing a blind method for drift management adjusts the embedding space, facilitating continuous visualisation of evolving data dynamics. Our experimental evaluations demonstrate the effectiveness and efficiency of S+t-SNE. The results highlight its ability to capture patterns in a streaming scenario. We hope our approach offers researchers and practitioners a real-time tool for understanding and interpreting high-dimensional data.
    
[^2]: CFaiRLLM：大型语言模型推荐系统中的消费者公平评估

    CFaiRLLM: Consumer Fairness Evaluation in Large-Language Model Recommender System

    [https://arxiv.org/abs/2403.05668](https://arxiv.org/abs/2403.05668)

    这项研究引入了一个全面的评估框架 CFaiRLLM，旨在评估和减轻 RecLLMs 中消费者端的偏见

    

    在推荐系统不断发展的过程中，像ChatGPT这样的大型语言模型的整合标志着引入了基于语言模型的推荐（RecLLM）的新时代。虽然这些进展承诺提供前所未有的个性化和效率，但也引发了对公平性的重要关切，特别是在推荐可能无意中继续或放大与敏感用户属性相关的偏见的情况下。为了解决这些问题，我们的研究引入了一个全面的评估框架CFaiRLLM，旨在评估（从而减轻）RecLLMs中消费者端的偏见。

    arXiv:2403.05668v1 Announce Type: new  Abstract: In the evolving landscape of recommender systems, the integration of Large Language Models (LLMs) such as ChatGPT marks a new era, introducing the concept of Recommendation via LLM (RecLLM). While these advancements promise unprecedented personalization and efficiency, they also bring to the fore critical concerns regarding fairness, particularly in how recommendations might inadvertently perpetuate or amplify biases associated with sensitive user attributes. In order to address these concerns, our study introduces a comprehensive evaluation framework, CFaiRLLM, aimed at evaluating (and thereby mitigating) biases on the consumer side within RecLLMs.   Our research methodically assesses the fairness of RecLLMs by examining how recommendations might vary with the inclusion of sensitive attributes such as gender, age, and their intersections, through both similarity alignment and true preference alignment. By analyzing recommendations gener
    

