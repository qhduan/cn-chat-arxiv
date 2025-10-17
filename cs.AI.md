# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Domain-Independent Dynamic Programming.](http://arxiv.org/abs/2401.13883) | 本文提出了一种领域无关的动态规划方法，并介绍了基于状态转移系统的动态规划描述语言。实验证明，该方法在许多组合优化问题上优于传统的混合整数规划和约束规划方法。 |
| [^2] | [Quantum Polar Metric Learning: Efficient Classically Learned Quantum Embeddings.](http://arxiv.org/abs/2312.01655) | 本论文提出了一种称为量子极坐标度量学习(QPMeL)的方法，通过经典模型学习量子比特的极坐标形式的参数，然后使用浅层PQC和可训练的门层来创建量子态和学习纠缠。与QMeL相比，QPMeL具有更高效的计算性能和可扩展性。 |
| [^3] | [Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT).](http://arxiv.org/abs/2307.01225) | 通过提出的解释性和透明性驱动的检测与转换（IT-DT）框架，我们在检测和转换文本对抗示例方面注重解释性和透明性。这个框架利用了注意力图、集成梯度和模型反馈等技术，在检测阶段有助于识别对对抗性分类有贡献的显著特征和扰动词语，并在转换阶段使用预训练的嵌入和模型反馈来生成扰动词语的最佳替代，以将对抗性示例转换为正常示例。 |
| [^4] | [Towards Automated Urban Planning: When Generative and ChatGPT-like AI Meets Urban Planning.](http://arxiv.org/abs/2304.03892) | 本文探讨了城市规划与人工智能的交叉应用，重点是自动化用地配置，通过对抗学习、生成神经网络、深度编码器-解码器网络、对话式 AI 和地理空间和时间机器学习等技术，AI 可以为现代城市规划带来不少创新与贡献。 |

# 详细

[^1]: 领域无关的动态规划方法

    Domain-Independent Dynamic Programming. (arXiv:2401.13883v1 [cs.AI])

    [http://arxiv.org/abs/2401.13883](http://arxiv.org/abs/2401.13883)

    本文提出了一种领域无关的动态规划方法，并介绍了基于状态转移系统的动态规划描述语言。实验证明，该方法在许多组合优化问题上优于传统的混合整数规划和约束规划方法。

    

    对于组合优化问题，基于模型的范例如混合整数规划 (MIP) 和约束规划 (CP) 旨在解耦问题的建模和求解过程，这是声明性问题求解的“圣杯”。我们提出了领域无关的动态规划（DIDP），这是一种基于动态规划 (DP) 的新的基于模型的方法。虽然DP并不新鲜，但通常它被作为一种特定问题的方法来实现。我们引入了动态规划描述语言 (DyPDL)，一种基于状态转移系统的形式化语言，灵感来自于AI规划。我们展示了启发式搜索算法可以用来求解DyPDL模型，并提出了七种DIDP求解器。我们在常见的11个组合优化问题类别的基准实例上，将我们的DIDP求解器与商业MIP和CP求解器进行了实验比较（分别求解MIP和CP模型）。结果显示DIDP在九个问题类别中优于MIP，也优于CP在九个问题类别中。

    For combinatorial optimization problems, model-based paradigms such as mixed-integer programming (MIP) and constraint programming (CP) aim to decouple modeling and solving a problem: the `holy grail' of declarative problem solving. We propose domain-independent dynamic programming (DIDP), a new model-based paradigm based on dynamic programming (DP). While DP is not new, it has typically been implemented as a problem-specific method. We introduce Dynamic Programming Description Language (DyPDL), a formalism to define DP models based on a state transition system, inspired by AI planning. We show that heuristic search algorithms can be used to solve DyPDL models and propose seven DIDP solvers. We experimentally compare our DIDP solvers with commercial MIP and CP solvers (solving MIP and CP models, respectively) on common benchmark instances of eleven combinatorial optimization problem classes. We show that DIDP outperforms MIP in nine problem classes, CP also in nine problem classes, and 
    
[^2]: 量子极坐标度量学习: 高效经典学习的量子嵌入

    Quantum Polar Metric Learning: Efficient Classically Learned Quantum Embeddings. (arXiv:2312.01655v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2312.01655](http://arxiv.org/abs/2312.01655)

    本论文提出了一种称为量子极坐标度量学习(QPMeL)的方法，通过经典模型学习量子比特的极坐标形式的参数，然后使用浅层PQC和可训练的门层来创建量子态和学习纠缠。与QMeL相比，QPMeL具有更高效的计算性能和可扩展性。

    

    深度度量学习在经典数据范畴中表现出极有潜力的结果，创建了分离明显的特征空间。这个想法也被应用到量子计算机中，通过量子度量学习(QMeL)。QMeL包括两个步骤，首先使用经典模型将数据压缩以适应有限数量的量子比特，然后使用参数化量子电路(PQC)在希尔伯特空间中创建更好的分离效果。然而，在嘈杂中间规模量子(NISQ)设备上，QMeL解决方案导致电路宽度和深度较大，从而限制了可扩展性。我们提出了一种称为量子极坐标度量学习(QPMeL)的方法，它使用经典模型学习一个量子比特的极坐标形式的参数。然后，我们利用仅包含$R_y$和$R_z$门的浅层PQC创建量子态，并利用可训练的$ZZ(\theta)$门层学习纠缠。电路还通过SWAP测试计算保真度，用于我们提出的保真度三元损失函数的训练，用于同时训练经典和量子模型。

    Deep metric learning has recently shown extremely promising results in the classical data domain, creating well-separated feature spaces. This idea was also adapted to quantum computers via Quantum Metric Learning(QMeL). QMeL consists of a 2 step process with a classical model to compress the data to fit into the limited number of qubits, then train a Parameterized Quantum Circuit(PQC) to create better separation in Hilbert Space. However, on Noisy Intermediate Scale Quantum (NISQ) devices. QMeL solutions result in high circuit width and depth, both of which limit scalability. We propose Quantum Polar Metric Learning (QPMeL) that uses a classical model to learn the parameters of the polar form of a qubit. We then utilize a shallow PQC with $R_y$ and $R_z$ gates to create the state and a trainable layer of $ZZ(\theta)$-gates to learn entanglement. The circuit also computes fidelity via a SWAP Test for our proposed Fidelity Triplet Loss function, used to train both classical and quantum 
    
[^3]: 解释性和透明性驱动的文本对抗示例的检测与转换（IT-DT）

    Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT). (arXiv:2307.01225v1 [cs.CL])

    [http://arxiv.org/abs/2307.01225](http://arxiv.org/abs/2307.01225)

    通过提出的解释性和透明性驱动的检测与转换（IT-DT）框架，我们在检测和转换文本对抗示例方面注重解释性和透明性。这个框架利用了注意力图、集成梯度和模型反馈等技术，在检测阶段有助于识别对对抗性分类有贡献的显著特征和扰动词语，并在转换阶段使用预训练的嵌入和模型反馈来生成扰动词语的最佳替代，以将对抗性示例转换为正常示例。

    

    基于Transformer的文本分类器如BERT、Roberta、T5和GPT-3在自然语言处理方面展示了令人印象深刻的性能。然而，它们对于对抗性示例的脆弱性提出了安全风险。现有的防御方法缺乏解释性，很难理解对抗性分类并识别模型的漏洞。为了解决这个问题，我们提出了解释性和透明性驱动的检测与转换（IT-DT）框架。它专注于在检测和转换文本对抗示例时的解释性和透明性。IT-DT利用注意力图、集成梯度和模型反馈等技术进行解释性检测。这有助于识别对对抗性分类有贡献的显著特征和扰动词语。在转换阶段，IT-DT利用预训练的嵌入和模型反馈来生成扰动词语的最佳替代。通过找到合适的替换，我们的目标是将对抗性示例转换为正常示例。

    Transformer-based text classifiers like BERT, Roberta, T5, and GPT-3 have shown impressive performance in NLP. However, their vulnerability to adversarial examples poses a security risk. Existing defense methods lack interpretability, making it hard to understand adversarial classifications and identify model vulnerabilities. To address this, we propose the Interpretability and Transparency-Driven Detection and Transformation (IT-DT) framework. It focuses on interpretability and transparency in detecting and transforming textual adversarial examples. IT-DT utilizes techniques like attention maps, integrated gradients, and model feedback for interpretability during detection. This helps identify salient features and perturbed words contributing to adversarial classifications. In the transformation phase, IT-DT uses pre-trained embeddings and model feedback to generate optimal replacements for perturbed words. By finding suitable substitutions, we aim to convert adversarial examples into
    
[^4]: 自动化城市规划：生成式和聊天式 AI 相结合的城市规划探索

    Towards Automated Urban Planning: When Generative and ChatGPT-like AI Meets Urban Planning. (arXiv:2304.03892v1 [cs.AI])

    [http://arxiv.org/abs/2304.03892](http://arxiv.org/abs/2304.03892)

    本文探讨了城市规划与人工智能的交叉应用，重点是自动化用地配置，通过对抗学习、生成神经网络、深度编码器-解码器网络、对话式 AI 和地理空间和时间机器学习等技术，AI 可以为现代城市规划带来不少创新与贡献。

    

    城市规划领域和人工智能领域曾经是独立发展的，但现在两个领域开始交叉汇合，互相借鉴和受益。本文介绍了城市规划从可持续性、生活、经济、灾害和环境等方面的重要性，回顾了城市规划的基本概念，并将这些概念与机器学习的关键开放问题联系起来，包括对抗学习、生成神经网络、深度编码器-解码器网络、对话式 AI 以及地理空间和时间机器学习等，评估了 AI 如何为现代城市规划做出贡献。因此，一个核心问题是自动化用地配置，即从周围的地理空间、人类移动、社交媒体、环境和经济活动中为目标区域生成土地用途和建筑配置。最后，本文勾画了集成 AI 和城市规划面临的一些挑战和潜在解决方案。

    The two fields of urban planning and artificial intelligence (AI) arose and developed separately. However, there is now cross-pollination and increasing interest in both fields to benefit from the advances of the other. In the present paper, we introduce the importance of urban planning from the sustainability, living, economic, disaster, and environmental perspectives. We review the fundamental concepts of urban planning and relate these concepts to crucial open problems of machine learning, including adversarial learning, generative neural networks, deep encoder-decoder networks, conversational AI, and geospatial and temporal machine learning, thereby assaying how AI can contribute to modern urban planning. Thus, a central problem is automated land-use configuration, which is formulated as the generation of land uses and building configuration for a target area from surrounding geospatial, human mobility, social media, environment, and economic activities. Finally, we delineate some 
    

