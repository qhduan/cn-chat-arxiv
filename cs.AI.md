# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nomic Embed: Training a Reproducible Long Context Text Embedder](https://rss.arxiv.org/abs/2402.01613) | Nomic Embed是第一个完全可复现、开源、开放权重、开放数据的8192上下文长度英文文本嵌入器，在短上下文和长上下文任务上优于OpenAI Ada-002和OpenAI text-embedding-3-small。 |
| [^2] | [Towards Stable Machine Learning Model Retraining via Slowly Varying Sequences](https://arxiv.org/abs/2403.19871) | 通过混合整数优化算法，以保持一致的分析洞见为重点，在重新训练机器学习模型中实现比贪婪训练更强稳定性，同时在模型性能上有小幅、可控的牺牲。 |
| [^3] | [People Attribute Purpose to Autonomous Vehicles When Explaining Their Behavior](https://arxiv.org/abs/2403.08828) | 人们会给自主车辆的行为赋予目的属性，并在生成解释和评估这些解释时表现出对目的论解释的倾向。 |
| [^4] | [Better Understandings and Configurations in MaxSAT Local Search Solvers via Anytime Performance Analysis](https://arxiv.org/abs/2403.06568) | 通过任意性能分析显示出MaxSAT本地搜索求解器在不同时间预算下的性能优劣变化 |
| [^5] | [Autoencoder-based General Purpose Representation Learning for Customer Embedding](https://arxiv.org/abs/2402.18164) | 设计了基于自动编码器的框架用于构建通用嵌入，展示简单模型在嵌入复杂表格数据时优于复杂模型，并将框架应用于生成表示AWS客户的嵌入，显著节省开发时间并观察到下游模型的改进。 |
| [^6] | [IDENAS: Internal Dependency Exploration for Neural Architecture Search.](http://arxiv.org/abs/2310.17250) | IDENAS是一种集成神经架构搜索和特征选择的方法，通过探索内部依赖性来提高分类任务的性能。 |
| [^7] | [CIDER: Category-Guided Intent Disentanglement for Accurate Personalized News Recommendation.](http://arxiv.org/abs/2310.09401) | CIDER是一种基于类别引导的个性化新闻推荐框架，通过意图分离和一致性的新闻表示来准确理解新闻文章的多个意图，并区分用户不同的后阅读偏好。 |
| [^8] | [Large Linguistic Models: Analyzing theoretical linguistic abilities of LLMs.](http://arxiv.org/abs/2305.00948) | 本研究展示了大型语言模型(LLMs)在语言任务上性能不断提高，且首次展示了它们能够生成连贯和有效的语言数据分析。分析和评估它们的元语言能力有助于我们理解它们的一般能力并对语言学理论模型提供新的认识。 |
| [^9] | [On the Interdependence of Reliance Behavior and Accuracy in AI-Assisted Decision-Making.](http://arxiv.org/abs/2304.08804) | 该论文分析了AI辅助决策中依赖行为和准确性之间的相互关系，并提出了一个视觉框架来更好地理解这种关系。该框架揭示了当人类在决策中过度依赖AI时，改善信任可能会降低准确性的有趣属性。 |

# 详细

[^1]: Nomic Embed：训练可复现的长上下文文本嵌入器

    Nomic Embed: Training a Reproducible Long Context Text Embedder

    [https://rss.arxiv.org/abs/2402.01613](https://rss.arxiv.org/abs/2402.01613)

    Nomic Embed是第一个完全可复现、开源、开放权重、开放数据的8192上下文长度英文文本嵌入器，在短上下文和长上下文任务上优于OpenAI Ada-002和OpenAI text-embedding-3-small。

    

    本技术报告描述了nomic-embed-text-v1的训练，这是第一个完全可复现、开源、开放权重、开放数据的8192上下文长度英文文本嵌入模型，在短上下文和长上下文任务上均优于OpenAI Ada-002和OpenAI text-embedding-3-small。我们在Apache 2许可下发布了训练代码和模型权重。与其他开源模型相比，我们还发布了一个包含2.35亿个策划文本对的训练数据加载器，可以完全复现nomic-embed-text-v1。你可以在https://github.com/nomic-ai/contrastors找到模型的代码和数据。

    This technical report describes the training of nomic-embed-text-v1, the first fully reproducible, open-source, open-weights, open-data, 8192 context length English text embedding model that outperforms both OpenAI Ada-002 and OpenAI text-embedding-3-small on short and long-context tasks. We release the training code and model weights under an Apache 2 license. In contrast with other open-source models, we release a training data loader with 235 million curated text pairs that allows for the full replication of nomic-embed-text-v1. You can find code and data to replicate the model at https://github.com/nomic-ai/contrastors
    
[^2]: 通过缓慢变化的序列实现稳定的机器学习模型重新训练

    Towards Stable Machine Learning Model Retraining via Slowly Varying Sequences

    [https://arxiv.org/abs/2403.19871](https://arxiv.org/abs/2403.19871)

    通过混合整数优化算法，以保持一致的分析洞见为重点，在重新训练机器学习模型中实现比贪婪训练更强稳定性，同时在模型性能上有小幅、可控的牺牲。

    

    重新训练机器学习模型仍然是实际机器学习模型部署的重要任务。现有方法主要关注贪婪方法，以找到表现最佳的模型，而不考虑通过不同的重新训练演变来保持训练模型结构的稳定性。在这项研究中，我们开发了一种混合整数优化算法，全面考虑了通过不同的数据批次更新重新训练机器学习模型的问题。我们的方法侧重于保留一致的分析洞见 - 这对于模型可解释性、实施简易性和与用户建立信任至关重要 - 通过使用可以直接纳入优化问题的自定义定义的距离度量。重要的是，我们的方法在真实的生产案例研究中表现出比贪婪训练模型更强的稳定性，同时在模型性能上有小幅、可控的牺牲。

    arXiv:2403.19871v1 Announce Type: cross  Abstract: Retraining machine learning models remains an important task for real-world machine learning model deployment. Existing methods focus largely on greedy approaches to find the best-performing model without considering the stability of trained model structures across different retraining evolutions. In this study, we develop a mixed integer optimization algorithm that holistically considers the problem of retraining machine learning models across different data batch updates. Our method focuses on retaining consistent analytical insights - which is important to model interpretability, ease of implementation, and fostering trust with users - by using custom-defined distance metrics that can be directly incorporated into the optimization problem. Importantly, our method shows stronger stability than greedily trained models with a small, controllable sacrifice in model performance in a real-world production case study. Finally, important an
    
[^3]: 当解释自主车辆的行为时，人们会给予其属性目的

    People Attribute Purpose to Autonomous Vehicles When Explaining Their Behavior

    [https://arxiv.org/abs/2403.08828](https://arxiv.org/abs/2403.08828)

    人们会给自主车辆的行为赋予目的属性，并在生成解释和评估这些解释时表现出对目的论解释的倾向。

    

    一款优秀的可解释人工智能系统的标志是用户可以理解并采取行动的解释。许多情况下，这需要系统提供可理解的因果或反事实解释。认知科学可以帮助我们理解用户可能期望的解释类型，以及在哪种格式下呈现这些解释。本文简要回顾了认知科学解释方面的相关文献，特别关注目的论，即以达到目的为解释决策的倾向。然后，我们报告了人们如何为自主车辆的行为产生解释以及他们如何评估这些解释的经验数据。在第一项调查中，参与者（n = 54）观看了道路场景的视频，并被要求为车辆的行为生成机械的、反事实的或目的论的言语解释。在第二项调查中，另一组参与者（n = 356）对这些进行评分。

    arXiv:2403.08828v1 Announce Type: cross  Abstract: A hallmark of a good XAI system is explanations that users can understand and act on. In many cases, this requires a system to offer causal or counterfactual explanations that are intelligible. Cognitive science can help us understand what kinds of explanations users might expect, and in which format to frame these explanations. We briefly review relevant literature from the cognitive science of explanation, particularly as it concerns teleology, the tendency to explain a decision in terms of the purpose it was meant to achieve. We then report empirical data on how people generate explanations for the behavior of autonomous vehicles, and how they evaluate these explanations. In a first survey, participants (n=54) were shown videos of a road scene and asked to generate either mechanistic, counterfactual, or teleological verbal explanations for a vehicle's actions. In the second survey, a different set of participants (n=356) rated these
    
[^4]: 通过任意性能分析更好地理解和配置MaxSAT本地搜索求解器

    Better Understandings and Configurations in MaxSAT Local Search Solvers via Anytime Performance Analysis

    [https://arxiv.org/abs/2403.06568](https://arxiv.org/abs/2403.06568)

    通过任意性能分析显示出MaxSAT本地搜索求解器在不同时间预算下的性能优劣变化

    

    尽管已经提出了许多用于MaxSAT问题的求解器，并且诸如MaxSAT Evaluations之类的基准环境提供了一个平台，用于比较最先进的求解器，但现有的评估通常是基于在给定运行时间预算内获得的最佳解的质量来评估的。然而，仅考虑特定时间预算内最终获得的解可能会限制我们理解求解器在收敛过程中的行为。本文证明了经验累积分布函数可用于比较MaxSAT本地搜索求解器在多个问题实例和不同时间预算下的任意性能。评估揭示了求解器性能的差异，并显示出求解器的（不）优势随着不同运行时间的调整。这项工作还展示了定量和高方差的评估

    arXiv:2403.06568v1 Announce Type: new  Abstract: Though numerous solvers have been proposed for the MaxSAT problem, and the benchmark environment such as MaxSAT Evaluations provides a platform for the comparison of the state-of-the-art solvers, existing assessments were usually evaluated based on the quality, e.g., fitness, of the best-found solutions obtained within a given running time budget. However, concerning solely the final obtained solutions regarding specific time budgets may restrict us from comprehending the behavior of the solvers along the convergence process. This paper demonstrates that Empirical Cumulative Distribution Functions can be used to compare MaxSAT local search solvers' anytime performance across multiple problem instances and various time budgets. The assessment reveals distinctions in solvers' performance and displays that the (dis)advantages of solvers adjust along different running times. This work also exhibits that the quantitative and high variance ass
    
[^5]: 基于自动编码器的通用表示学习用于客户嵌入

    Autoencoder-based General Purpose Representation Learning for Customer Embedding

    [https://arxiv.org/abs/2402.18164](https://arxiv.org/abs/2402.18164)

    设计了基于自动编码器的框架用于构建通用嵌入，展示简单模型在嵌入复杂表格数据时优于复杂模型，并将框架应用于生成表示AWS客户的嵌入，显著节省开发时间并观察到下游模型的改进。

    

    最近几年，利用数据的领域特定基础结构及其生成因素进行表示学习，在各种用例无关应用中取得成功。然而，表格数据的多样性和复杂性使得通过多维向量在潜在空间中表示这些结构具有挑战性。我们设计了一个基于自动编码器的框架用于构建通用嵌入，评估了不同自动编码器架构的性能，并展示了简单模型在嵌入高度复杂表格数据时优于复杂模型。我们将我们的框架应用于生成插拔式、丰富和匿名化的表示AWS客户的嵌入，可用于任何模型，节省开发时间高达45％，并观察到下游模型的显著改进。此外，我们提出了一种对于多层收缩自动编码器重构损失计算的重要改进。

    arXiv:2402.18164v1 Announce Type: cross  Abstract: In recent years, exploiting the domain-specific underlying structure of data and its generative factors for representation learning has shown success in various use-case agnostic applications. However, the diversity and complexity of tabular data have made it challenging to represent these structures in a latent space through multi-dimensional vectors. We design an autoencoder-based framework for building general purpose embeddings, we assess the performance of different autoencoder architectures, and show simpler models outperform complex ones in embedding highly complex tabular data. We apply our framework to produce plug-and-play, rich, and anonymized embeddings representing AWS customers for usage in any model, saving up to 45% of development time, and observe significant improvements in downstream models. Moreover, we propose a significant improvement to the calculation of reconstruction loss for multi-layer contractive autoencode
    
[^6]: IDENAS: 内部依赖性探索用于神经架构搜索

    IDENAS: Internal Dependency Exploration for Neural Architecture Search. (arXiv:2310.17250v1 [cs.LG])

    [http://arxiv.org/abs/2310.17250](http://arxiv.org/abs/2310.17250)

    IDENAS是一种集成神经架构搜索和特征选择的方法，通过探索内部依赖性来提高分类任务的性能。

    

    机器学习是从不同数据集中提取有价值信息和进行各种预测的强大工具。传统算法依赖于明确定义的输入和输出变量，然而，在某些情况下，输入和输出变量之间的区别以及模型的底层关联（输入和输出）层是未知的。神经架构搜索（NAS）和特征选择已成为这些场景中的有希望的解决方案。该研究提出了IDENAS，一种基于内部依赖性的神经架构搜索方法，将NAS与特征选择相结合。该方法在涉及1D传感器和2D图像数据的分类问题中探索了完整的参数空间的内部依赖性。IDENAS采用了修改的编码器-解码器模型和顺序前向搜索（SFS）算法，将输入-输出配置搜索与嵌入式特征选择相结合。实验结果证明了IDENAS的优越性能。

    Machine learning is a powerful tool for extracting valuable information and making various predictions from diverse datasets. Traditional algorithms rely on well-defined input and output variables however, there are scenarios where the distinction between the input and output variables and the underlying, associated (input and output) layers of the model, are unknown. Neural Architecture Search (NAS) and Feature Selection have emerged as promising solutions in such scenarios. This research proposes IDENAS, an Internal Dependency-based Exploration for Neural Architecture Search, integrating NAS with feature selection. The methodology explores internal dependencies in the complete parameter space for classification involving 1D sensor and 2D image data as well. IDENAS employs a modified encoder-decoder model and the Sequential Forward Search (SFS) algorithm, combining input-output configuration search with embedded feature selection. Experimental results demonstrate IDENASs superior perf
    
[^7]: CIDER: 基于类别引导的意图分离方法用于准确的个性化新闻推荐

    CIDER: Category-Guided Intent Disentanglement for Accurate Personalized News Recommendation. (arXiv:2310.09401v1 [cs.IR])

    [http://arxiv.org/abs/2310.09401](http://arxiv.org/abs/2310.09401)

    CIDER是一种基于类别引导的个性化新闻推荐框架，通过意图分离和一致性的新闻表示来准确理解新闻文章的多个意图，并区分用户不同的后阅读偏好。

    

    个性化新闻推荐旨在帮助用户找到与其兴趣相符的新闻文章，这在缓解用户信息过载问题方面起到至关重要的作用。尽管许多最近的研究致力于改进用户和新闻的表示方法，但以下挑战很少被研究：（C1）如何准确理解一篇新闻文章中包含的多个意图？以及（C2）如何区分用户点击历史中对新闻文章有不同后阅读偏好的情况？为了同时解决这两个挑战，在本文中，我们提出了一种新的个性化新闻推荐框架（CIDER），它利用（1）基于类别引导的意图分离来解决（C1）和（2）基于一致性的新闻表示来解决（C2）。此外，我们将类别预测纳入CIDER的训练过程作为辅助任务，这提供了额外的监督信号，以增强意图分离。在两个真实数据集上进行了广泛的实验。

    Personalized news recommendation aims to assist users in finding news articles that align with their interests, which plays a pivotal role in mitigating users' information overload problem. Although many recent works have been studied for better user and news representations, the following challenges have been rarely studied: (C1) How to precisely comprehend a range of intents coupled within a news article? and (C2) How to differentiate news articles with varying post-read preferences in users' click history? To tackle both challenges together, in this paper, we propose a novel personalized news recommendation framework (CIDER) that employs (1) category-guided intent disentanglement for (C1) and (2) consistency-based news representation for (C2). Furthermore, we incorporate a category prediction into the training process of CIDER as an auxiliary task, which provides supplementary supervisory signals to enhance intent disentanglement. Extensive experiments on two real-world datasets rev
    
[^8]: 大型语言模型：分析LLM的理论语言能力

    Large Linguistic Models: Analyzing theoretical linguistic abilities of LLMs. (arXiv:2305.00948v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.00948](http://arxiv.org/abs/2305.00948)

    本研究展示了大型语言模型(LLMs)在语言任务上性能不断提高，且首次展示了它们能够生成连贯和有效的语言数据分析。分析和评估它们的元语言能力有助于我们理解它们的一般能力并对语言学理论模型提供新的认识。

    

    大型语言模型(LLMs)的性能最近已经提高到了能够在许多语言任务上表现良好的程度。我们在这里展示了，这些模型也可以生成连贯和有效的语言数据的形式分析，展示了大型语言模型对其元语言能力分析的巨大潜力。LLMs主要是通过文本形式的语言数据进行训练；分析和评估它们的元语言能力改进了我们对它们的一般能力的理解，并对语言学中的理论模型提供了新的认识。在本文中，我们通过专注于形式语言学的三个子领域：句法、音韵学和语义学，探究了GPT-4的元语言能力。我们提出了一个关于大型语言模型元语言分析的研究计划，提出了实验设计，提供了一般指导方针，讨论了限制，并为这个研究方向提供了未来的方向。这个研究还有助于揭示大型语言模型的潜在能力和理论模型的新视角。

    The performance of large language models (LLMs) has recently improved to the point where the models can perform well on many language tasks. We show here that for the first time, the models can also generate coherent and valid formal analyses of linguistic data and illustrate the vast potential of large language models for analyses of their metalinguistic abilities. LLMs are primarily trained on language data in the form of text; analyzing and evaluating their metalinguistic abilities improves our understanding of their general capabilities and sheds new light on theoretical models in linguistics. In this paper, we probe into GPT-4's metalinguistic capabilities by focusing on three subfields of formal linguistics: syntax, phonology, and semantics. We outline a research program for metalinguistic analyses of large language models, propose experimental designs, provide general guidelines, discuss limitations, and offer future directions for this line of research. This line of inquiry als
    
[^9]: 关于AI辅助决策中依赖行为与准确性的相互关系

    On the Interdependence of Reliance Behavior and Accuracy in AI-Assisted Decision-Making. (arXiv:2304.08804v1 [cs.HC])

    [http://arxiv.org/abs/2304.08804](http://arxiv.org/abs/2304.08804)

    该论文分析了AI辅助决策中依赖行为和准确性之间的相互关系，并提出了一个视觉框架来更好地理解这种关系。该框架揭示了当人类在决策中过度依赖AI时，改善信任可能会降低准确性的有趣属性。

    

    在AI辅助决策中，将人类置于决策环路中央的主要承诺是，他们应该能够通过符合其正确的和覆盖其错误的建议来补充AI系统。然而实践中，我们经常看到人类倾向于过度或不足地依赖AI建议，这意味着他们要么依从错误的建议，要么覆盖正确的建议。这种依赖行为对决策准确性有害。在这项工作中，我们阐述并分析了在AI辅助决策中依赖行为和准确性之间的相互关系，这在以前的工作中很大程度上被忽视了。我们还提出了一个视觉框架，使这种相互关系更加具体化。该框架帮助我们解释和比较实证研究结果，并获得对AI辅助决策干预（例如解释）影响的细致理解。最后，我们从框架中推出了几个有趣的属性：（i）当人类不足地依赖AI建议时，改善信任将显着提高准确性，但在他们过度依赖时，信任的改善却可能降低准确性。

    In AI-assisted decision-making, a central promise of putting a human in the loop is that they should be able to complement the AI system by adhering to its correct and overriding its mistaken recommendations. In practice, however, we often see that humans tend to over- or under-rely on AI recommendations, meaning that they either adhere to wrong or override correct recommendations. Such reliance behavior is detrimental to decision-making accuracy. In this work, we articulate and analyze the interdependence between reliance behavior and accuracy in AI-assisted decision-making, which has been largely neglected in prior work. We also propose a visual framework to make this interdependence more tangible. This framework helps us interpret and compare empirical findings, as well as obtain a nuanced understanding of the effects of interventions (e.g., explanations) in AI-assisted decision-making. Finally, we infer several interesting properties from the framework: (i) when humans under-rely o
    

