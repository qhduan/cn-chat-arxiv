# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Link Prediction under Heterophily: A Physics-Inspired Graph Neural Network Approach](https://arxiv.org/abs/2402.14802) | 图神经网络在异质图上的链路预测面临学习能力和表达能力方面的挑战，本论文提出了受物理启发的方法以增强节点分类性能。 |
| [^2] | [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377) | 这项研究展示了n-gram语言模型的价值，并介绍了一个名为infini-gram的引擎，它可以以毫秒级的延迟计算任意n的n-gram概率，使得在神经大型语言模型中对文本进行更准确的分析成为可能。 |
| [^3] | [GATSY: Graph Attention Network for Music Artist Similarity.](http://arxiv.org/abs/2311.00635) | GATSY是一个基于图注意力网络的音乐艺术家相似性推荐系统，可以灵活地处理多样性和关联性，并在不依赖手工特征的情况下取得卓越的性能结果。 |

# 详细

[^1]: 在异质性下的链路预测: 受物理启发的图神经网络方法

    Link Prediction under Heterophily: A Physics-Inspired Graph Neural Network Approach

    [https://arxiv.org/abs/2402.14802](https://arxiv.org/abs/2402.14802)

    图神经网络在异质图上的链路预测面临学习能力和表达能力方面的挑战，本论文提出了受物理启发的方法以增强节点分类性能。

    

    最近几年，由于其在对图表示的真实世界现象建模方面的灵活性，图神经网络（GNNs）已成为各种深度学习领域的事实标准。然而，GNNs的消息传递机制在学习能力和表达能力方面面临挑战，这限制了在异质图上实现高性能的能力，其中相邻节点经常具有不同的标签。大多数现有解决方案主要局限于针对节点分类任务的特定基准。这种狭窄的焦点限制了链路预测在多个应用中的潜在影响，包括推荐系统。例如，在社交网络中，两个用户可能由于某种潜在原因而连接，这使得提前预测这种连接具有挑战性。受物理启发的GNNs（如GRAFF）对提高节点分类性能提供了显著的贡献。

    arXiv:2402.14802v1 Announce Type: new  Abstract: In the past years, Graph Neural Networks (GNNs) have become the `de facto' standard in various deep learning domains, thanks to their flexibility in modeling real-world phenomena represented as graphs. However, the message-passing mechanism of GNNs faces challenges in learnability and expressivity, hindering high performance on heterophilic graphs, where adjacent nodes frequently have different labels. Most existing solutions addressing these challenges are primarily confined to specific benchmarks focused on node classification tasks. This narrow focus restricts the potential impact that link prediction under heterophily could offer in several applications, including recommender systems. For example, in social networks, two users may be connected for some latent reason, making it challenging to predict such connections in advance. Physics-Inspired GNNs such as GRAFF provided a significant contribution to enhance node classification perf
    
[^2]: 无限-gram：将无限n-gram语言模型扩展到万亿标记

    Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens

    [https://arxiv.org/abs/2401.17377](https://arxiv.org/abs/2401.17377)

    这项研究展示了n-gram语言模型的价值，并介绍了一个名为infini-gram的引擎，它可以以毫秒级的延迟计算任意n的n-gram概率，使得在神经大型语言模型中对文本进行更准确的分析成为可能。

    

    在神经大型语言模型（LLM）时代，n-gram语言模型还具有相关性吗？我们的答案是肯定的，并且我们展示了它们在文本分析和改进神经LLM方面的价值。然而，这需要在两个方面对n-gram模型进行现代化。首先，我们将它们与神经LLM相同的数据规模训练- 1.4万亿个标记。这是迄今为止构建的最大的n-gram模型。其次，现有的n-gram模型使用的n很小，这妨碍了它们的性能；相反，我们允许n可以是任意大的，通过引入一个新的无限-gram LM与回退。我们开发了一个名为infini-gram的引擎，它可以通过后缀数组计算无限-gram（以及任意n的n-gram）概率，并且具有毫秒级的延迟，而无需预先计算n-gram计数表（这将非常昂贵）。无限-gram框架和infini-gram引擎使我们能够对人类写作和机器生成的文本进行许多新颖和有意思的分析：我们发现无限-gram LM...

    Are n-gram language models still relevant in this era of neural large language models (LLMs)? Our answer is yes, and we show their values in both text analysis and improving neural LLMs. Yet this necessitates modernizing n-gram models in two aspects. First, we train them at the same data scale as neural LLMs -- 1.4 trillion tokens. This is the largest n-gram model ever built. Second, existing n-gram models use small n which hinders their performance; we instead allow n to be arbitrarily large, by introducing a new $\infty$-gram LM with backoff. Instead of pre-computing n-gram count tables (which would be very expensive), we develop an engine named infini-gram -- powered by suffix arrays -- that can compute $\infty$-gram (as well as n-gram with arbitrary n) probabilities with millisecond-level latency. The $\infty$-gram framework and infini-gram engine enable us to conduct many novel and interesting analyses of human-written and machine-generated text: we find that the $\infty$-gram LM 
    
[^3]: GATSY: 音乐艺术家相似性的图注意力网络

    GATSY: Graph Attention Network for Music Artist Similarity. (arXiv:2311.00635v1 [cs.IR])

    [http://arxiv.org/abs/2311.00635](http://arxiv.org/abs/2311.00635)

    GATSY是一个基于图注意力网络的音乐艺术家相似性推荐系统，可以灵活地处理多样性和关联性，并在不依赖手工特征的情况下取得卓越的性能结果。

    

    艺术家相似性问题已经成为社会和科学环境中的重要课题。现代研究解决方案根据用户的喜好来促进音乐发现。然而，定义艺术家之间的相似性可能涉及多个方面，甚至与主观角度相关，并且经常影响推荐结果。本文提出了GATSY，这是一个建立在图注意力网络上的推荐系统，并由艺术家的聚类嵌入驱动。所提出的框架利用输入数据的图拓扑结构，在不过分依赖手工特征的情况下取得了卓越的性能结果。这种灵活性使我们能够在音乐数据集中引入虚构的艺术家，与以前不相关的艺术家建立联系，并根据可能的异质来源获得推荐。实验结果证明了该方法相对于现有解决方案的有效性。

    The artist similarity quest has become a crucial subject in social and scientific contexts. Modern research solutions facilitate music discovery according to user tastes. However, defining similarity among artists may involve several aspects, even related to a subjective perspective, and it often affects a recommendation. This paper presents GATSY, a recommendation system built upon graph attention networks and driven by a clusterized embedding of artists. The proposed framework takes advantage of a graph topology of the input data to achieve outstanding performance results without relying heavily on hand-crafted features. This flexibility allows us to introduce fictitious artists in a music dataset, create bridges to previously unrelated artists, and get recommendations conditioned by possibly heterogeneous sources. Experimental results prove the effectiveness of the proposed method with respect to state-of-the-art solutions.
    

