# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structure Guided Large Language Model for SQL Generation](https://arxiv.org/abs/2402.13284) | 通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。 |
| [^2] | [Exploring Value Biases: How LLMs Deviate Towards the Ideal](https://arxiv.org/abs/2402.11005) | 研究发现大型语言模型（LLMs）在给出响应时存在一个价值偏好的机制，倾向于偏向理想状态，这种偏差会对不同应用场景产生重要影响。 |
| [^3] | [Don't Push the Button! Exploring Data Leakage Risks in Machine Learning and Transfer Learning.](http://arxiv.org/abs/2401.13796) | 本文讨论了机器学习中的数据泄露问题，即未预期的信息污染训练数据，影响模型性能评估，用户可能由于缺乏理解而忽视关键步骤，导致乐观的性能估计在实际场景中不成立。 |
| [^4] | [Interpretable Anomaly Detection via Discrete Optimization.](http://arxiv.org/abs/2303.14111) | 该论文提出了一个通过学习有限自动机进行异常检测的框架，并通过约束优化算法和新的正则化方案提高了可解释性。 |
| [^5] | [Don't Get Me Wrong: How to Apply Deep Visual Interpretations to Time Series.](http://arxiv.org/abs/2203.07861) | 该论文提出了一个针对时间序列分类和分割任务的框架，通过六个度量来评估基于梯度、传播或干扰的事后可视化解释方法。实验结果表明，这些方法对于时间序列的解释具有较高的可信度和有效性。 |

# 详细

[^1]: 结构引导的大型语言模型用于SQL生成

    Structure Guided Large Language Model for SQL Generation

    [https://arxiv.org/abs/2402.13284](https://arxiv.org/abs/2402.13284)

    通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。

    

    生成准确的结构化查询语言（SQL）是一个长期存在的问题，特别是在将用户的语义查询与结构化数据库匹配，然后生成结构化SQL方面。现有模型通常将查询和数据库模式输入到LLM中，并依赖LLM执行语义-结构匹配并生成结构化SQL。然而，这种解决方案忽略了用户查询和数据库中的结构信息，而这些信息可以用来增强结构化SQL的生成。这一疏忽可能导致不准确或无法执行的SQL生成。为了充分利用结构，我们提出了一个结构到SQL的框架，利用固有的结构信息来改善LLM的SQL生成。具体地，我们介绍了我们的结构引导SQL（SGU-SQL）生成模型。

    arXiv:2402.13284v1 Announce Type: cross  Abstract: Generating accurate Structured Querying Language (SQL) is a long-standing problem, especially in matching users' semantic queries with structured databases and then generating structured SQL. Existing models typically input queries and database schemas into the LLM and rely on the LLM to perform semantic-structure matching and generate structured SQL. However, such solutions overlook the structural information within user queries and databases, which can be utilized to enhance the generation of structured SQL. This oversight can lead to inaccurate or unexecutable SQL generation. To fully exploit the structure, we propose a structure-to-SQL framework, which leverages the inherent structure information to improve the SQL generation of LLMs. Specifically, we introduce our Structure Guided SQL~(SGU-SQL) generation model. SGU-SQL first links user queries and databases in a structure-enhanced manner. It then decomposes complicated linked str
    
[^2]: 探究价值偏好：LLMs偏向理想状态的偏差

    Exploring Value Biases: How LLMs Deviate Towards the Ideal

    [https://arxiv.org/abs/2402.11005](https://arxiv.org/abs/2402.11005)

    研究发现大型语言模型（LLMs）在给出响应时存在一个价值偏好的机制，倾向于偏向理想状态，这种偏差会对不同应用场景产生重要影响。

    

    大型语言模型（LLMs）被部署在各种应用中，并且它们的响应对社会产生着越来越大的影响。理解LLMs在给出响应时的非故意机制对于解释它们的性能并辨别它们在现实世界应用中的偏差至关重要。这类似于人类研究中，这种无意识的响应被称为抽样。我们研究了LLMs的这种抽样现象，发现LLMs的抽样倾向于偏爱高价值选项。价值偏好对应于从最可能的响应向LLM中代表的理想价值的转变。实际上，即便是通过上下文提示学习到的新实体，这种效果也能够再现。我们表明这种偏差表现在意想不到的地方，并对选择典型实例等相关应用场景产生影响。结果显示，价值偏好在不同分类的LLMs中都很明显。

    arXiv:2402.11005v1 Announce Type: cross  Abstract: Large-Language-Models (LLMs) are deployed in a wide range of applications, and their response has an increasing social impact. Understanding the non-deliberate(ive) mechanism of LLMs in giving responses is essential in explaining their performance and discerning their biases in real-world applications. This is analogous to human studies, where such inadvertent responses are referred to as sampling. We study this sampling of LLMs in light of value bias and show that the sampling of LLMs tends to favour high-value options. Value bias corresponds to this shift of response from the most likely towards an ideal value represented in the LLM. In fact, this effect can be reproduced even with new entities learnt via in-context prompting. We show that this bias manifests in unexpected places and has implications on relevant application scenarios, like choosing exemplars. The results show that value bias is strong in LLMs across different categor
    
[^3]: 不要按按钮！探索机器学习和迁移学习中的数据泄露风险

    Don't Push the Button! Exploring Data Leakage Risks in Machine Learning and Transfer Learning. (arXiv:2401.13796v1 [cs.LG])

    [http://arxiv.org/abs/2401.13796](http://arxiv.org/abs/2401.13796)

    本文讨论了机器学习中的数据泄露问题，即未预期的信息污染训练数据，影响模型性能评估，用户可能由于缺乏理解而忽视关键步骤，导致乐观的性能估计在实际场景中不成立。

    

    机器学习（ML）在各个领域取得了革命性的进展，为多个领域提供了预测能力。然而，随着ML工具的日益可获得性，许多从业者缺乏深入的ML专业知识，采用了“按按钮”方法，利用用户友好的界面而忽视了底层算法的深入理解。虽然这种方法提供了便利，但它引发了对结果可靠性的担忧，导致了错误的性能评估等挑战。本文解决了ML中的一个关键问题，即数据泄露，其中未预期的信息污染了训练数据，影响了模型的性能评估。由于缺乏理解，用户可能会无意中忽视关键步骤，从而导致在现实场景中可能不成立的乐观性能估计。评估性能与实际在新数据上的性能的差异是一个重要的关注点。本文特别将ML中的数据泄露分为不同类别，并讨论了相关解决方法。

    Machine Learning (ML) has revolutionized various domains, offering predictive capabilities in several areas. However, with the increasing accessibility of ML tools, many practitioners, lacking deep ML expertise, adopt a "push the button" approach, utilizing user-friendly interfaces without a thorough understanding of underlying algorithms. While this approach provides convenience, it raises concerns about the reliability of outcomes, leading to challenges such as incorrect performance evaluation. This paper addresses a critical issue in ML, known as data leakage, where unintended information contaminates the training data, impacting model performance evaluation. Users, due to a lack of understanding, may inadvertently overlook crucial steps, leading to optimistic performance estimates that may not hold in real-world scenarios. The discrepancy between evaluated and actual performance on new data is a significant concern. In particular, this paper categorizes data leakage in ML, discussi
    
[^4]: 通过离散优化实现可解释性异常检测

    Interpretable Anomaly Detection via Discrete Optimization. (arXiv:2303.14111v1 [cs.LG])

    [http://arxiv.org/abs/2303.14111](http://arxiv.org/abs/2303.14111)

    该论文提出了一个通过学习有限自动机进行异常检测的框架，并通过约束优化算法和新的正则化方案提高了可解释性。

    

    异常检测在许多应用领域中都是必不可少的，例如网络安全、执法、医学和欺诈保护。然而，目前深度学习方法的决策过程往往难以理解，这通常限制了它们的实际应用性。为了克服这个限制，我们提出了一个学习框架，可以从序列数据中学习可解释性的异常检测器。具体来说，我们考虑从给定的未标记序列多重集中学习确定性有限自动机 （DFA）的任务。我们证明了这个问题是计算难题，并基于约束优化开发了两个学习算法。此外，我们为优化问题引入了新的正则化方案，以提高我们的DFA的整体可解释性。通过原型实现，我们证明我们的方法在准确性和F1分数方面表现出有望的结果。

    Anomaly detection is essential in many application domains, such as cyber security, law enforcement, medicine, and fraud protection. However, the decision-making of current deep learning approaches is notoriously hard to understand, which often limits their practical applicability. To overcome this limitation, we propose a framework for learning inherently interpretable anomaly detectors from sequential data. More specifically, we consider the task of learning a deterministic finite automaton (DFA) from a given multi-set of unlabeled sequences. We show that this problem is computationally hard and develop two learning algorithms based on constraint optimization. Moreover, we introduce novel regularization schemes for our optimization problems that improve the overall interpretability of our DFAs. Using a prototype implementation, we demonstrate that our approach shows promising results in terms of accuracy and F1 score.
    
[^5]: 不要误会我：如何将深度视觉解释应用于时间序列

    Don't Get Me Wrong: How to Apply Deep Visual Interpretations to Time Series. (arXiv:2203.07861v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2203.07861](http://arxiv.org/abs/2203.07861)

    该论文提出了一个针对时间序列分类和分割任务的框架，通过六个度量来评估基于梯度、传播或干扰的事后可视化解释方法。实验结果表明，这些方法对于时间序列的解释具有较高的可信度和有效性。

    

    在许多应用中，正确解释和理解深度学习模型非常重要。针对图像和自然语言处理的解释性视觉解释方法允许领域专家验证和理解几乎任何深度学习模型。然而，当推广到任意时间序列时，它们在本质上更加复杂和多样化。一个可视化解释是否解释了有效的推理或捕捉了实际特征是难以判断的。因此，我们需要客观评估来获得可信的质量指标，而不是盲目信任。我们提出了一个框架，包括六个正交度量，用于针对时间序列分类和分割任务的基于梯度、传播或干扰的事后视觉解释方法。实验研究包括了常见的时间序列神经网络架构和九种可视化解释方法。我们使用UCR r等多样的数据集评估了这些可视化解释方法。

    The correct interpretation and understanding of deep learning models are essential in many applications. Explanatory visual interpretation approaches for image, and natural language processing allow domain experts to validate and understand almost any deep learning model. However, they fall short when generalizing to arbitrary time series, which is inherently less intuitive and more diverse. Whether a visualization explains valid reasoning or captures the actual features is difficult to judge. Hence, instead of blind trust, we need an objective evaluation to obtain trustworthy quality metrics. We propose a framework of six orthogonal metrics for gradient-, propagation- or perturbation-based post-hoc visual interpretation methods for time series classification and segmentation tasks. An experimental study includes popular neural network architectures for time series and nine visual interpretation methods. We evaluate the visual interpretation methods with diverse datasets from the UCR r
    

