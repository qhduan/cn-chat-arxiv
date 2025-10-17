# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Intrusion Detection with Domain-Invariant Representation Learning in Latent Space.](http://arxiv.org/abs/2312.17300) | 本研究提出了一种使用多任务学习的两阶段表示学习技术，通过培养潜在空间中的特征，包括本地和跨领域特征，以增强对未知分布领域的泛化效果。此外，通过最小化先验和潜在空间之间的互信息来分离潜在空间，并且在多个网络安全数据集上评估了模型的效能。 |
| [^2] | [Evaluation of feature selection performance for identification of best effective technical indicators on stock market price prediction.](http://arxiv.org/abs/2310.09903) | 本研究评估了特征选择方法在股市价格预测中的性能，通过选择最佳的技术指标组合来实现最少误差的预测。研究结果表明，不同的包装器特征选择方法在不同的机器学习方法中具有不同的表现。 |
| [^3] | [An analysis of the derivative-free loss method for solving PDEs.](http://arxiv.org/abs/2309.16829) | 本研究分析了一种无导数损失方法在使用神经网络求解椭圆型偏微分方程的方法。我们发现训练损失偏差与时间间隔和空间梯度成正比，与行走者大小成反比，同时时间间隔必须足够长。我们提供了数值测试结果以支持我们的分析。 |
| [^4] | [Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT).](http://arxiv.org/abs/2307.01225) | 通过提出的解释性和透明性驱动的检测与转换（IT-DT）框架，我们在检测和转换文本对抗示例方面注重解释性和透明性。这个框架利用了注意力图、集成梯度和模型反馈等技术，在检测阶段有助于识别对对抗性分类有贡献的显著特征和扰动词语，并在转换阶段使用预训练的嵌入和模型反馈来生成扰动词语的最佳替代，以将对抗性示例转换为正常示例。 |
| [^5] | [Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning.](http://arxiv.org/abs/2306.05494) | 本文对于基于机器学习的网络入侵检测系统(NIDS)的对抗性攻击进行了分类，同时探究了持续再训练对NIDS对抗性攻击的影响。实验表明，即使没有对抗性训练，持续再训练也可以减少对抗性攻击的影响。 |
| [^6] | [The Principle of Uncertain Maximum Entropy.](http://arxiv.org/abs/2305.09868) | 介绍了不确定最大熵原理，该原理可以处理模型元素不可观测的情况，并优于特定条件下的最大熵方法。同时将黑匣子机器学习模型的输出用作不确定机器熵框架的输入，性能得到了提高。 |
| [^7] | [Towards Automated Urban Planning: When Generative and ChatGPT-like AI Meets Urban Planning.](http://arxiv.org/abs/2304.03892) | 本文探讨了城市规划与人工智能的交叉应用，重点是自动化用地配置，通过对抗学习、生成神经网络、深度编码器-解码器网络、对话式 AI 和地理空间和时间机器学习等技术，AI 可以为现代城市规划带来不少创新与贡献。 |

# 详细

[^1]: 在潜在空间中通过领域不变表示学习改善入侵检测

    Improving Intrusion Detection with Domain-Invariant Representation Learning in Latent Space. (arXiv:2312.17300v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2312.17300](http://arxiv.org/abs/2312.17300)

    本研究提出了一种使用多任务学习的两阶段表示学习技术，通过培养潜在空间中的特征，包括本地和跨领域特征，以增强对未知分布领域的泛化效果。此外，通过最小化先验和潜在空间之间的互信息来分离潜在空间，并且在多个网络安全数据集上评估了模型的效能。

    

    领域泛化聚焦于利用来自具有丰富训练数据和标签的多个相关领域的知识，增强对未知分布（IN）和超出分布（OOD）领域的推理。在我们的研究中，我们引入了一种两阶段表示学习技术，使用多任务学习。这种方法旨在从跨越多个领域的特征中培养一个潜在空间，包括本地和跨领域，以增强对IN和OOD领域的泛化。此外，我们尝试通过最小化先验与潜在空间之间的互信息来分离潜在空间，有效消除虚假特征相关性。综合而言，联合优化将促进领域不变特征学习。我们使用标准分类指标评估模型在多个网络安全数据集上的效能，对比了现代领域泛化方法的结果。

    Domain generalization focuses on leveraging knowledge from multiple related domains with ample training data and labels to enhance inference on unseen in-distribution (IN) and out-of-distribution (OOD) domains. In our study, we introduce a two-phase representation learning technique using multi-task learning. This approach aims to cultivate a latent space from features spanning multiple domains, encompassing both native and cross-domains, to amplify generalization to IN and OOD territories. Additionally, we attempt to disentangle the latent space by minimizing the mutual information between the prior and latent space, effectively de-correlating spurious feature correlations. Collectively, the joint optimization will facilitate domain-invariant feature learning. We assess the model's efficacy across multiple cybersecurity datasets, using standard classification metrics on both unseen IN and OOD sets, and juxtapose the results with contemporary domain generalization methods.
    
[^2]: 评估特征选择在股市价格预测中的性能，以确定最有效的技术指标

    Evaluation of feature selection performance for identification of best effective technical indicators on stock market price prediction. (arXiv:2310.09903v1 [q-fin.ST])

    [http://arxiv.org/abs/2310.09903](http://arxiv.org/abs/2310.09903)

    本研究评估了特征选择方法在股市价格预测中的性能，通过选择最佳的技术指标组合来实现最少误差的预测。研究结果表明，不同的包装器特征选择方法在不同的机器学习方法中具有不同的表现。

    

    鉴于技术指标对股市预测的影响，特征选择对选择最佳指标至关重要。一种考虑在特征选择过程中模型性能的特征选择方法是包装器特征选择方法。本研究旨在通过特征选择鉴定出最少误差的预测股市价格的最佳股市指标组合。为评估包装器特征选择技术对股市预测的影响，本文在过去10年苹果公司的数据上使用了10个评估器和123个技术指标进行了SFS和SBS的考察。此外，通过提出的方法，将由3天时间窗口创建的数据转化为适用于回归方法的输入。从观察结果可以得出：（1）每种包装器特征选择方法在不同的机器学习方法中具有不同的结果，每种方法在不同的预测准确性上也有所不同。

    Due to the influence of many factors, including technical indicators on stock market prediction, feature selection is important to choose the best indicators. One of the feature selection methods that consider the performance of models during feature selection is the wrapper feature selection method. The aim of this research is to identify a combination of the best stock market indicators through feature selection to predict the stock market price with the least error. In order to evaluate the impact of wrapper feature selection techniques on stock market prediction, in this paper SFS and SBS with 10 estimators and 123 technical indicators have been examined on the last 10 years of Apple Company. Also, by the proposed method, the data created by the 3-day time window were converted to the appropriate input for regression methods. Based on the results observed: (1) Each wrapper feature selection method has different results with different machine learning methods, and each method is mor
    
[^3]: 无导数损失方法在求解偏微分方程中的分析

    An analysis of the derivative-free loss method for solving PDEs. (arXiv:2309.16829v1 [math.NA])

    [http://arxiv.org/abs/2309.16829](http://arxiv.org/abs/2309.16829)

    本研究分析了一种无导数损失方法在使用神经网络求解椭圆型偏微分方程的方法。我们发现训练损失偏差与时间间隔和空间梯度成正比，与行走者大小成反比，同时时间间隔必须足够长。我们提供了数值测试结果以支持我们的分析。

    

    本研究分析了无导数损失方法在使用神经网络求解一类椭圆型偏微分方程中的应用。无导数损失方法采用费曼-卡克公式，结合随机行走者及其对应的平均值。我们考察了费曼-卡克公式中与时间间隔相关的影响，以及行走者大小对计算效率、可训练性和采样误差的影响。我们的分析表明，训练损失偏差与时间间隔和神经网络的空间梯度成正比，与行走者大小成反比。同时，我们还表明时间间隔必须足够长才能训练网络。这些分析结果说明，在时间间隔的最优下界基础上，我们可以选择尽可能小的行走者大小。我们还提供了支持我们分析的数值测试。

    This study analyzes the derivative-free loss method to solve a certain class of elliptic PDEs using neural networks. The derivative-free loss method uses the Feynman-Kac formulation, incorporating stochastic walkers and their corresponding average values. We investigate the effect of the time interval related to the Feynman-Kac formulation and the walker size in the context of computational efficiency, trainability, and sampling errors. Our analysis shows that the training loss bias is proportional to the time interval and the spatial gradient of the neural network while inversely proportional to the walker size. We also show that the time interval must be sufficiently long to train the network. These analytic results tell that we can choose the walker size as small as possible based on the optimal lower bound of the time interval. We also provide numerical tests supporting our analysis.
    
[^4]: 解释性和透明性驱动的文本对抗示例的检测与转换（IT-DT）

    Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT). (arXiv:2307.01225v1 [cs.CL])

    [http://arxiv.org/abs/2307.01225](http://arxiv.org/abs/2307.01225)

    通过提出的解释性和透明性驱动的检测与转换（IT-DT）框架，我们在检测和转换文本对抗示例方面注重解释性和透明性。这个框架利用了注意力图、集成梯度和模型反馈等技术，在检测阶段有助于识别对对抗性分类有贡献的显著特征和扰动词语，并在转换阶段使用预训练的嵌入和模型反馈来生成扰动词语的最佳替代，以将对抗性示例转换为正常示例。

    

    基于Transformer的文本分类器如BERT、Roberta、T5和GPT-3在自然语言处理方面展示了令人印象深刻的性能。然而，它们对于对抗性示例的脆弱性提出了安全风险。现有的防御方法缺乏解释性，很难理解对抗性分类并识别模型的漏洞。为了解决这个问题，我们提出了解释性和透明性驱动的检测与转换（IT-DT）框架。它专注于在检测和转换文本对抗示例时的解释性和透明性。IT-DT利用注意力图、集成梯度和模型反馈等技术进行解释性检测。这有助于识别对对抗性分类有贡献的显著特征和扰动词语。在转换阶段，IT-DT利用预训练的嵌入和模型反馈来生成扰动词语的最佳替代。通过找到合适的替换，我们的目标是将对抗性示例转换为正常示例。

    Transformer-based text classifiers like BERT, Roberta, T5, and GPT-3 have shown impressive performance in NLP. However, their vulnerability to adversarial examples poses a security risk. Existing defense methods lack interpretability, making it hard to understand adversarial classifications and identify model vulnerabilities. To address this, we propose the Interpretability and Transparency-Driven Detection and Transformation (IT-DT) framework. It focuses on interpretability and transparency in detecting and transforming textual adversarial examples. IT-DT utilizes techniques like attention maps, integrated gradients, and model feedback for interpretability during detection. This helps identify salient features and perturbed words contributing to adversarial classifications. In the transformation phase, IT-DT uses pre-trained embeddings and model feedback to generate optimal replacements for perturbed words. By finding suitable substitutions, we aim to convert adversarial examples into
    
[^5]: 神经网络中对抗性漏洞攻击的实用性测试：动态学习的影响

    Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning. (arXiv:2306.05494v1 [cs.CR])

    [http://arxiv.org/abs/2306.05494](http://arxiv.org/abs/2306.05494)

    本文对于基于机器学习的网络入侵检测系统(NIDS)的对抗性攻击进行了分类，同时探究了持续再训练对NIDS对抗性攻击的影响。实验表明，即使没有对抗性训练，持续再训练也可以减少对抗性攻击的影响。

    

    机器学习被广泛应用于网络入侵检测系统(NIDS)中，由于其自动化的特性和在处理和分类大量数据上的高精度。但机器学习存在缺陷，其中最大的问题之一是对抗性攻击，其目的是使机器学习模型产生错误的预测。本文提出了两个独特的贡献：对抗性攻击对基于机器学习的NIDS实用性问题的分类和对持续训练对NIDS对抗性攻击的影响进行了研究。我们的实验表明，即使没有对抗性训练，持续再训练也可以减少对抗性攻击的影响。虽然对抗性攻击可能会危及基于机器学习的NIDS，但持续再训练可带来一定的缓解效果。

    Machine Learning (ML) has become ubiquitous, and its deployment in Network Intrusion Detection Systems (NIDS) is inevitable due to its automated nature and high accuracy in processing and classifying large volumes of data. However, ML has been found to have several flaws, on top of them are adversarial attacks, which aim to trick ML models into producing faulty predictions. While most adversarial attack research focuses on computer vision datasets, recent studies have explored the practicality of such attacks against ML-based network security entities, especially NIDS.  This paper presents two distinct contributions: a taxonomy of practicality issues associated with adversarial attacks against ML-based NIDS and an investigation of the impact of continuous training on adversarial attacks against NIDS. Our experiments indicate that continuous re-training, even without adversarial training, can reduce the effect of adversarial attacks. While adversarial attacks can harm ML-based NIDSs, ou
    
[^6]: 不确定最大熵原理

    The Principle of Uncertain Maximum Entropy. (arXiv:2305.09868v1 [cs.IT])

    [http://arxiv.org/abs/2305.09868](http://arxiv.org/abs/2305.09868)

    介绍了不确定最大熵原理，该原理可以处理模型元素不可观测的情况，并优于特定条件下的最大熵方法。同时将黑匣子机器学习模型的输出用作不确定机器熵框架的输入，性能得到了提高。

    

    最大熵原理在信息理论中的引入，为统计力学，机器学习和生态学等各个领域的发展做出了贡献。其得到的解决方案作为催化剂，促进研究人员将他们的经验观察映射到获取无偏模型，同时加深了对复杂系统和现象的理解。然而，在模型元素不直接可观测的情况下，例如存在噪声或眼部遮挡的情况下，标准最大熵方法可能会失败，因为它们无法匹配特征约束。在这里，我们展示了不确定最大熵原理作为一种方法，尽管存在任意噪声观察，它同时将所有可用信息编码，而且优于一些特定条件下的最大熵方法的准确度。此外，我们将黑匣子机器学习模型的输出用作不确定机器熵框架的输入，从而在与最大似然算法相比时建立了改进的性能。

    The principle of maximum entropy, as introduced by Jaynes in information theory, has contributed to advancements in various domains such as Statistical Mechanics, Machine Learning, and Ecology. Its resultant solutions have served as a catalyst, facilitating researchers in mapping their empirical observations to the acquisition of unbiased models, whilst deepening the understanding of complex systems and phenomena. However, when we consider situations in which the model elements are not directly observable, such as when noise or ocular occlusion is present, possibilities arise for which standard maximum entropy approaches may fail, as they are unable to match feature constraints. Here we show the Principle of Uncertain Maximum Entropy as a method that both encodes all available information in spite of arbitrarily noisy observations while surpassing the accuracy of some ad-hoc methods. Additionally, we utilize the output of a black-box machine learning model as input into an uncertain ma
    
[^7]: 自动化城市规划：生成式和聊天式 AI 相结合的城市规划探索

    Towards Automated Urban Planning: When Generative and ChatGPT-like AI Meets Urban Planning. (arXiv:2304.03892v1 [cs.AI])

    [http://arxiv.org/abs/2304.03892](http://arxiv.org/abs/2304.03892)

    本文探讨了城市规划与人工智能的交叉应用，重点是自动化用地配置，通过对抗学习、生成神经网络、深度编码器-解码器网络、对话式 AI 和地理空间和时间机器学习等技术，AI 可以为现代城市规划带来不少创新与贡献。

    

    城市规划领域和人工智能领域曾经是独立发展的，但现在两个领域开始交叉汇合，互相借鉴和受益。本文介绍了城市规划从可持续性、生活、经济、灾害和环境等方面的重要性，回顾了城市规划的基本概念，并将这些概念与机器学习的关键开放问题联系起来，包括对抗学习、生成神经网络、深度编码器-解码器网络、对话式 AI 以及地理空间和时间机器学习等，评估了 AI 如何为现代城市规划做出贡献。因此，一个核心问题是自动化用地配置，即从周围的地理空间、人类移动、社交媒体、环境和经济活动中为目标区域生成土地用途和建筑配置。最后，本文勾画了集成 AI 和城市规划面临的一些挑战和潜在解决方案。

    The two fields of urban planning and artificial intelligence (AI) arose and developed separately. However, there is now cross-pollination and increasing interest in both fields to benefit from the advances of the other. In the present paper, we introduce the importance of urban planning from the sustainability, living, economic, disaster, and environmental perspectives. We review the fundamental concepts of urban planning and relate these concepts to crucial open problems of machine learning, including adversarial learning, generative neural networks, deep encoder-decoder networks, conversational AI, and geospatial and temporal machine learning, thereby assaying how AI can contribute to modern urban planning. Thus, a central problem is automated land-use configuration, which is formulated as the generation of land uses and building configuration for a target area from surrounding geospatial, human mobility, social media, environment, and economic activities. Finally, we delineate some 
    

