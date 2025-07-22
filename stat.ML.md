# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Attention with Markov: A Framework for Principled Analysis of Transformers via Markov Chains](https://arxiv.org/abs/2402.04161) | 提出了一个新的框架，通过马尔可夫链的视角研究了注意力模型的顺序建模能力，理论上刻画了单层Transformer的损失景观并发现了全局最小值和坏局部最小值的存在。 |
| [^2] | [Score-based Causal Representation Learning: Linear and General Transformations](https://arxiv.org/abs/2402.00849) | 这篇论文提出了一种基于得分的算法类，用于干预范围内的因果表示学习，涵盖了线性和一般转化。算法保证了可识别性和实现性，并且通过创造性地将得分函数与因果表示学习相结合。 |
| [^3] | [Quantum Learning Theory Beyond Batch Binary Classification.](http://arxiv.org/abs/2302.07409) | 这篇论文通过研究量子学习理论拓展了批处理多类学习、在线布尔学习和在线多类学习，并提出了第一个具有量子示例的在线学习模型。 |

# 详细

[^1]: 基于马尔可夫链的注意力模型的规范分析框架：通过马尔可夫链研究Transformer的顺序建模能力

    Attention with Markov: A Framework for Principled Analysis of Transformers via Markov Chains

    [https://arxiv.org/abs/2402.04161](https://arxiv.org/abs/2402.04161)

    提出了一个新的框架，通过马尔可夫链的视角研究了注意力模型的顺序建模能力，理论上刻画了单层Transformer的损失景观并发现了全局最小值和坏局部最小值的存在。

    

    近年来，基于注意力的Transformer在包括自然语言在内的多个领域取得了巨大成功。其中一个关键因素是生成式预训练过程，模型在此过程中通过自回归的方式在大型文本语料库上进行训练。为了揭示这一现象，我们提出了一个新的框架，通过马尔可夫链的视角，允许理论和系统实验来研究Transformer的顺序建模能力。受到自然语言的马尔可夫性质的启发，我们将数据建模为一个马尔可夫源，并利用这个框架系统地研究数据分布特性、Transformer架构、学到的分布和最终模型性能之间的相互作用。特别地，我们理论上刻画了单层Transformer的损失景观，并展示了全局最小值和坏局部最小值的存在，这取决于具体的数据性质。

    In recent years, attention-based transformers have achieved tremendous success across a variety of disciplines including natural languages. A key ingredient behind their success is the generative pretraining procedure, during which these models are trained on a large text corpus in an auto-regressive manner. To shed light on this phenomenon, we propose a new framework that allows both theory and systematic experiments to study the sequential modeling capabilities of transformers through the lens of Markov chains. Inspired by the Markovianity of natural languages, we model the data as a Markovian source and utilize this framework to systematically study the interplay between the data-distributional properties, the transformer architecture, the learnt distribution, and the final model performance. In particular, we theoretically characterize the loss landscape of single-layer transformers and show the existence of global minima and bad local minima contingent upon the specific data chara
    
[^2]: 基于得分的因果表示学习：线性和一般的转化

    Score-based Causal Representation Learning: Linear and General Transformations

    [https://arxiv.org/abs/2402.00849](https://arxiv.org/abs/2402.00849)

    这篇论文提出了一种基于得分的算法类，用于干预范围内的因果表示学习，涵盖了线性和一般转化。算法保证了可识别性和实现性，并且通过创造性地将得分函数与因果表示学习相结合。

    

    本篇论文针对一般非参数潜在因果模型和将潜在变量映射到观测变量的未知转化，研究了基于干预的因果表示学习（CRL）。研究了线性和一般的转化。这篇论文同时讨论了可识别性和实现性两个方面。可识别性是指确定算法不相关的条件，以确保恢复真实的潜在因果变量和潜在因果图。实现性是指算法方面，解决设计算法来实现可识别保证的问题。通过将得分函数（即密度函数对数的梯度）与CRL之间建立新联系，本文设计了一种得分为基础的算法类，确保了可识别性和实现性。首先，本文专注于线性转化，并展示了每个n个随机硬干预下该转化的因果表示可识别。

    This paper addresses intervention-based causal representation learning (CRL) under a general nonparametric latent causal model and an unknown transformation that maps the latent variables to the observed variables. Linear and general transformations are investigated. The paper addresses both the \emph{identifiability} and \emph{achievability} aspects. Identifiability refers to determining algorithm-agnostic conditions that ensure recovering the true latent causal variables and the latent causal graph underlying them. Achievability refers to the algorithmic aspects and addresses designing algorithms that achieve identifiability guarantees. By drawing novel connections between \emph{score functions} (i.e., the gradients of the logarithm of density functions) and CRL, this paper designs a \emph{score-based class of algorithms} that ensures both identifiability and achievability. First, the paper focuses on \emph{linear} transformations and shows that one stochastic hard intervention per n
    
[^3]: 《超越批处理二元分类的量子学习理论》

    Quantum Learning Theory Beyond Batch Binary Classification. (arXiv:2302.07409v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.07409](http://arxiv.org/abs/2302.07409)

    这篇论文通过研究量子学习理论拓展了批处理多类学习、在线布尔学习和在线多类学习，并提出了第一个具有量子示例的在线学习模型。

    

    Arunachalam和de Wolf（2018）证明了在可实现和糊涂设置下，量子批处理学习布尔函数的样本复杂性与相应的经典样本复杂性具有相同的形式和数量级。在本文中，我们将这个明显令人惊讶的结果推广到了批处理多类学习、在线布尔学习和在线多类学习。对于我们的在线学习结果，我们首先考虑了Dawid和Tewari（2022）经典模型的自适应对手变体。然后，我们引入了第一个（据我们所知）具有量子示例的在线学习模型。

    Arunachalam and de Wolf (2018) showed that the sample complexity of quantum batch learning of boolean functions, in the realizable and agnostic settings, has the same form and order as the corresponding classical sample complexities. In this paper, we extend this, ostensibly surprising, message to batch multiclass learning, online boolean learning, and online multiclass learning. For our online learning results, we first consider an adaptive adversary variant of the classical model of Dawid and Tewari (2022). Then, we introduce the first (to the best of our knowledge) model of online learning with quantum examples.
    

