# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Shill-Proof Auctions](https://arxiv.org/abs/2404.00475) | 本文研究了免疫作弊的拍卖形式，发现荷兰式拍卖（设有适当保留价）是唯一的最优且强免疫作弊的拍卖，同时荷兰式拍卖（没有保留价）是唯一同时高效和弱免疫作弊的先验独立拍卖。 |
| [^2] | [Hedonic Prices and Quality Adjusted Price Indices Powered by AI.](http://arxiv.org/abs/2305.00044) | 本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。 |
| [^3] | [Recent Contributions to Theories of Discrimination.](http://arxiv.org/abs/2205.05994) | 本文对歧视理论进行了综述并提出了新的研究成果，包括特定学习和信令环境下的思考，算法学习和决策制定，以及引入行为偏差和误设信念的代理。此外，该文确定了一个由歧视性社会规范和歧视性制度设计理论组成的歧视性机构模型类别，探讨了关于歧视测量和歧视分类的问题。 |

# 详细

[^1]: 免疫作弊拍卖

    Shill-Proof Auctions

    [https://arxiv.org/abs/2404.00475](https://arxiv.org/abs/2404.00475)

    本文研究了免疫作弊的拍卖形式，发现荷兰式拍卖（设有适当保留价）是唯一的最优且强免疫作弊的拍卖，同时荷兰式拍卖（没有保留价）是唯一同时高效和弱免疫作弊的先验独立拍卖。

    

    在单品拍卖中，一个欺诈性的卖家可能会伪装成一个或多个竞标者，以操纵成交价格。本文对那些免疫作弊的拍卖格式进行了表征：一个利润最大化的卖家没有任何动机提交任何虚假报价。我们区分了强免疫作弊，即一个了解竞标者估值的卖家永远无法从作弊中获利，和弱免疫作弊，它仅要求从作弊中得到的平衡预期利润为非正。荷兰式拍卖（设有适当保留价）是唯一的最优和强免疫作弊拍卖。此外，荷兰式拍卖（没有保留价）是唯一的具有先验独立性的拍卖，既高效又弱免疫作弊。虽然存在多种策略证明、弱免疫作弊和最优拍卖；任何最优拍卖只能满足集合 {静态、策略证明、弱免疫作弊} 中的两个性质。

    arXiv:2404.00475v1 Announce Type: new  Abstract: In a single-item auction, a duplicitous seller may masquerade as one or more bidders in order to manipulate the clearing price. This paper characterizes auction formats that are shill-proof: a profit-maximizing seller has no incentive to submit any shill bids. We distinguish between strong shill-proofness, in which a seller with full knowledge of bidders' valuations can never profit from shilling, and weak shill-proofness, which requires only that the expected equilibrium profit from shilling is nonpositive. The Dutch auction (with suitable reserve) is the unique optimal and strongly shill-proof auction. Moreover, the Dutch auction (with no reserve) is the unique prior-independent auction that is both efficient and weakly shill-proof. While there are a multiplicity of strategy-proof, weakly shill-proof, and optimal auctions; any optimal auction can satisfy only two properties in the set {static, strategy-proof, weakly shill-proof}.
    
[^2]: 由人工智能驱动的享乐价格和质量调整价格指数

    Hedonic Prices and Quality Adjusted Price Indices Powered by AI. (arXiv:2305.00044v1 [econ.GN])

    [http://arxiv.org/abs/2305.00044](http://arxiv.org/abs/2305.00044)

    本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。

    

    在当今的经济环境下，使用电子记录准确地实时测量价格指数的变化对于跟踪通胀和生产率至关重要。本文开发了经验享乐模型，能够处理大量未结构化的产品数据（文本、图像、价格和数量），并输出精确的享乐价格估计和派生指数。为实现这一目标，我们使用深度神经网络从文本描述和图像中生成抽象的产品属性或”特征“，然后使用这些属性来估算享乐价格函数。具体地，我们使用基于transformers的大型语言模型将有关产品的文本信息转换为数字特征，使用训练或微调过的产品描述信息，使用残差网络模型将产品图像转换为数字特征。为了产生估计的享乐价格函数，我们再次使用多任务神经网络，训练以在所有时间段同时预测产品的价格。

    Accurate, real-time measurements of price index changes using electronic records are essential for tracking inflation and productivity in today's economic environment. We develop empirical hedonic models that can process large amounts of unstructured product data (text, images, prices, quantities) and output accurate hedonic price estimates and derived indices. To accomplish this, we generate abstract product attributes, or ``features,'' from text descriptions and images using deep neural networks, and then use these attributes to estimate the hedonic price function. Specifically, we convert textual information about the product to numeric features using large language models based on transformers, trained or fine-tuned using product descriptions, and convert the product image to numeric features using a residual network model. To produce the estimated hedonic price function, we again use a multi-task neural network trained to predict a product's price in all time periods simultaneousl
    
[^3]: 歧视理论的近期贡献

    Recent Contributions to Theories of Discrimination. (arXiv:2205.05994v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2205.05994](http://arxiv.org/abs/2205.05994)

    本文对歧视理论进行了综述并提出了新的研究成果，包括特定学习和信令环境下的思考，算法学习和决策制定，以及引入行为偏差和误设信念的代理。此外，该文确定了一个由歧视性社会规范和歧视性制度设计理论组成的歧视性机构模型类别，探讨了关于歧视测量和歧视分类的问题。

    

    本文综述了关于歧视理论的文献，主要关注于新的研究成果。最近的理论扩展了传统的基于品味和统计歧视框架，通过考虑学习和信号环境的特定特点，经常使用新的信息和机制设计语言；通过分析算法的学习和决策制定；并引入具有行为偏差和误设信念的代理。该综述还尝试缩小经济学角度对“歧视理论”的看法与社会科学文献中更广泛的歧视研究之间的差距。在这方面，我首先通过确定一个由歧视性社会规范和歧视性制度设计理论组成的歧视性机构模型类别做出贡献。其次，讨论涉及歧视测量和将歧视分类为偏见或统计学，直接或系统性以及世界性等问题。

    This paper surveys the literature on theories of discrimination, focusing mainly on new contributions. Recent theories expand on the traditional taste-based and statistical discrimination frameworks by considering specific features of learning and signaling environments, often using novel informationand mechanism-design language; analyzing learning and decision making by algorithms; and introducing agents with behavioral biases and misspecified beliefs. This survey also attempts to narrow the gap between the economic perspective on ``theories of discrimination'' and the broader study of discrimination in the social science literature. In that respect, I first contribute by identifying a class of models of discriminatory institutions, made up of theories of discriminatory social norms and discriminatory institutional design. Second, I discuss issues relating to the measurement of discrimination, and the classification of discrimination as bias or statistical, direct or systemic, and a
    

