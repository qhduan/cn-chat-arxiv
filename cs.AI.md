# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Finding Decision Tree Splits in Streaming and Massively Parallel Models](https://arxiv.org/abs/2403.19867) | 提出了在数据流学习中计算决策树最佳分割点的算法，能够在流式计算和大规模并行模型中高效运行 |
| [^2] | [Citation-Enhanced Generation for LLM-based Chatbot](https://arxiv.org/abs/2402.16063) | 提出一种基于引文增强的LLM聊天机器人生成方法，采用检索模块搜索支持文档来解决幻觉内容产生的问题。 |
| [^3] | [Unveiling Molecular Moieties through Hierarchical Graph Explainability](https://arxiv.org/abs/2402.01744) | 本论文提出了一种使用图神经网络和分层可解释人工智能技术的方法，能够准确预测生物活性并找到与之相关的最重要的成分。 |
| [^4] | [Multi-Step Deductive Reasoning Over Natural Language: An Empirical Study on Out-of-Distribution Generalisation](https://arxiv.org/abs/2207.14000) | 提出了IMA-GloVe-GA，一个用于自然语言表达的多步推理的迭代神经推理网络，在超领域泛化方面具有更好的性能表现。 |
| [^5] | [Asynchronous Graph Generators.](http://arxiv.org/abs/2309.17335) | 异步图生成器（AGG）是一种新型的图神经网络架构，通过节点生成进行数据插补，并隐式学习传感器测量的因果图表示，取得了state-of-the-art的结果。 |
| [^6] | [Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation.](http://arxiv.org/abs/2303.12973) | 本文提出了多种不确定性校准技术，以改进推荐系统中倾向性估计的效果。经过实验验证，校准后的IPS估计器在Coat和yahoo数据集上表现更好。 |

# 详细

[^1]: 在流式和大规模并行模型中找到决策树分割点

    Finding Decision Tree Splits in Streaming and Massively Parallel Models

    [https://arxiv.org/abs/2403.19867](https://arxiv.org/abs/2403.19867)

    提出了在数据流学习中计算决策树最佳分割点的算法，能够在流式计算和大规模并行模型中高效运行

    

    在这项工作中，我们提出了一种数据流算法，用于计算决策树学习中的最优分割点。具体而言，给定观测数据流$x_i$及其标签$y_i$，目标是找到将数据分为两组的最佳分割点$j$，使得均方误差（回归问题）或误分类率（分类问题）最小化。我们提供了多种快速的数据流算法，这些算法在这些问题中使用亚线性空间和少量次数的遍历。这些算法还可以扩展到大规模并行计算模型中。尽管不能直接比较，但我们的工作与Domingos和Hulten的开创性工作（KDD 2000）相互补充。

    arXiv:2403.19867v1 Announce Type: cross  Abstract: In this work, we provide data stream algorithms that compute optimal splits in decision tree learning. In particular, given a data stream of observations $x_i$ and their labels $y_i$, the goal is to find the optimal split point $j$ that divides the data into two sets such that the mean squared error (for regression) or misclassification rate (for classification) is minimized. We provide various fast streaming algorithms that use sublinear space and a small number of passes for these problems. These algorithms can also be extended to the massively parallel computation model. Our work, while not directly comparable, complements the seminal work of Domingos and Hulten (KDD 2000).
    
[^2]: 基于引文增强的LLM聊天机器人生成

    Citation-Enhanced Generation for LLM-based Chatbot

    [https://arxiv.org/abs/2402.16063](https://arxiv.org/abs/2402.16063)

    提出一种基于引文增强的LLM聊天机器人生成方法，采用检索模块搜索支持文档来解决幻觉内容产生的问题。

    

    大型语言模型（LLMs）在各种情景下展现出强大的通用智能，包括将它们集成到聊天机器人中。然而，基于LLM的聊天机器人面临的一个重要挑战是在回复中可能产生虚构内容，这严重限制了它们的适用性。本文提出了一种新颖的后续引用增强生成（CEG）方法，结合检索论证。与先前侧重于预防生成过程中幻觉的研究不同，我们的方法以后续方式解决了这个问题。它结合了一个检索模块来搜索与生成内容相关的支持文档，并采用基于自然语言推理的方法。

    arXiv:2402.16063v1 Announce Type: cross  Abstract: Large language models (LLMs) exhibit powerful general intelligence across diverse scenarios, including their integration into chatbots. However, a vital challenge of LLM-based chatbots is that they may produce hallucinated content in responses, which significantly limits their applicability. Various efforts have been made to alleviate hallucination, such as retrieval augmented generation and reinforcement learning with human feedback, but most of them require additional training and data annotation. In this paper, we propose a novel post-hoc \textbf{C}itation-\textbf{E}nhanced \textbf{G}eneration (\textbf{CEG}) approach combined with retrieval argumentation. Unlike previous studies that focus on preventing hallucinations during generation, our method addresses this issue in a post-hoc way. It incorporates a retrieval module to search for supporting documents relevant to the generated content, and employs a natural language inference-ba
    
[^3]: 通过分层图解释揭示分子成分

    Unveiling Molecular Moieties through Hierarchical Graph Explainability

    [https://arxiv.org/abs/2402.01744](https://arxiv.org/abs/2402.01744)

    本论文提出了一种使用图神经网络和分层可解释人工智能技术的方法，能够准确预测生物活性并找到与之相关的最重要的成分。

    

    背景：图神经网络（GNN）作为一种强大的工具，在支持体外虚拟筛选方面已经出现多年。在这项工作中，我们提出了一种使用图卷积架构实现高精度多靶标筛选的GNN。我们还设计了一种分层可解释人工智能（XAI）技术，通过利用信息传递机制，在原子、环和整个分子层面上直接捕获信息，从而找到与生物活性预测相关的最重要的成分。结果：我们在支持虚拟筛选方面的二十个细胞周期依赖性激酶靶标上报道了一种最先进的GNN分类器。我们的分类器超越了作者提出的先前最先进方法。此外，我们还设计了一个仅针对CDK1的高灵敏度版本的GNN，以使用我们的解释器来避免多类别模型固有的偏差。分层解释器已经由一位专家化学家在19个CDK1批准药物上进行了验证。

    Background: Graph Neural Networks (GNN) have emerged in very recent years as a powerful tool for supporting in silico Virtual Screening. In this work we present a GNN which uses Graph Convolutional architectures to achieve very accurate multi-target screening. We also devised a hierarchical Explainable Artificial Intelligence (XAI) technique to catch information directly at atom, ring, and whole molecule level by leveraging the message passing mechanism. In this way, we find the most relevant moieties involved in bioactivity prediction. Results: We report a state-of-the-art GNN classifier on twenty Cyclin-dependent Kinase targets in support of VS. Our classifier outperforms previous SOTA approaches proposed by the authors. Moreover, a CDK1-only high-sensitivity version of the GNN has been designed to use our explainer in order to avoid the inherent bias of multi-class models. The hierarchical explainer has been validated by an expert chemist on 19 approved drugs on CDK1. Our explainer 
    
[^4]: 自然语言上的多步演绎推理：基于超领域泛化的实证研究

    Multi-Step Deductive Reasoning Over Natural Language: An Empirical Study on Out-of-Distribution Generalisation

    [https://arxiv.org/abs/2207.14000](https://arxiv.org/abs/2207.14000)

    提出了IMA-GloVe-GA，一个用于自然语言表达的多步推理的迭代神经推理网络，在超领域泛化方面具有更好的性能表现。

    

    将深度学习与符号逻辑推理结合起来，旨在充分利用这两个领域的成功，并引起了越来越多的关注。受DeepLogic启发，该模型经过端到端训练，用于执行逻辑程序推理，我们介绍了IMA-GloVe-GA，这是一个用自然语言表达的多步推理的迭代神经推理网络。在我们的模型中，推理是使用基于RNN的迭代内存神经网络进行的，其中包含一个门关注机制。我们在PARARULES、CONCEPTRULES V1和CONCEPTRULES V2三个数据集上评估了IMA-GloVe-GA。实验结果表明，带有门关注机制的DeepLogic比DeepLogic和其他RNN基线模型能够实现更高的测试准确性。我们的模型在规则被打乱时比RoBERTa-Large实现了更好的超领域泛化性能。此外，为了解决当前多步推理数据集中推理深度不平衡的问题

    arXiv:2207.14000v2 Announce Type: replace-cross  Abstract: Combining deep learning with symbolic logic reasoning aims to capitalize on the success of both fields and is drawing increasing attention. Inspired by DeepLogic, an end-to-end model trained to perform inference on logic programs, we introduce IMA-GloVe-GA, an iterative neural inference network for multi-step reasoning expressed in natural language. In our model, reasoning is performed using an iterative memory neural network based on RNN with a gate attention mechanism. We evaluate IMA-GloVe-GA on three datasets: PARARULES, CONCEPTRULES V1 and CONCEPTRULES V2. Experimental results show DeepLogic with gate attention can achieve higher test accuracy than DeepLogic and other RNN baseline models. Our model achieves better out-of-distribution generalisation than RoBERTa-Large when the rules have been shuffled. Furthermore, to address the issue of unbalanced distribution of reasoning depths in the current multi-step reasoning datase
    
[^5]: 异步图生成器

    Asynchronous Graph Generators. (arXiv:2309.17335v1 [cs.LG])

    [http://arxiv.org/abs/2309.17335](http://arxiv.org/abs/2309.17335)

    异步图生成器（AGG）是一种新型的图神经网络架构，通过节点生成进行数据插补，并隐式学习传感器测量的因果图表示，取得了state-of-the-art的结果。

    

    我们引入了异步图生成器（AGG），这是一种用于多通道时间序列的新型图神经网络架构。AGG将观测值建模为动态图上的节点，并通过转导式节点生成进行数据插补。AGG不依赖于循环组件或对时间规律的假设，使用可学习的嵌入将测量值、时间戳和元数据直接表示在节点中，并利用注意机制来学习变量之间的关系。这样，所提出的架构隐式地学习传感器测量的因果图表示，可以基于未见时间戳和元数据对新的测量进行预测。我们将所提出的AGG在概念和实证两方面与之前的工作进行了比较，并简要讨论了数据增强对AGG性能的影响。实验结果表明，AGG在t

    We introduce the asynchronous graph generator (AGG), a novel graph neural network architecture for multi-channel time series which models observations as nodes on a dynamic graph and can thus perform data imputation by transductive node generation. Completely free from recurrent components or assumptions about temporal regularity, AGG represents measurements, timestamps and metadata directly in the nodes via learnable embeddings, to then leverage attention to learn expressive relationships across the variables of interest. This way, the proposed architecture implicitly learns a causal graph representation of sensor measurements which can be conditioned on unseen timestamps and metadata to predict new measurements by an expansion of the learnt graph. The proposed AGG is compared both conceptually and empirically to previous work, and the impact of data augmentation on the performance of AGG is also briefly discussed. Our experiments reveal that AGG achieved state-of-the-art results in t
    
[^6]: 推荐系统中反事实倾向估计的不确定性校准

    Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation. (arXiv:2303.12973v1 [cs.AI])

    [http://arxiv.org/abs/2303.12973](http://arxiv.org/abs/2303.12973)

    本文提出了多种不确定性校准技术，以改进推荐系统中倾向性估计的效果。经过实验验证，校准后的IPS估计器在Coat和yahoo数据集上表现更好。

    

    在推荐系统中，由于选择偏差，许多评分信息都丢失了，这被称为非随机缺失。反事实逆倾向评分（IPS）被用于衡量每个观察到的评分的填充错误。虽然在多种情况下有效，但我们认为IPS估计的性能受到倾向性估计不确定性的限制。本文提出了多种代表性的不确定性校准技术，以改进推荐系统中倾向性估计的不确定性校准。通过对偏误和推广界限的理论分析表明，经过校准的IPS估计器优于未校准的IPS估计器。 Coat和yahoo数据集上的实验结果表明，不确定性校准得到改进，从而使推荐结果更好。

    In recommendation systems, a large portion of the ratings are missing due to the selection biases, which is known as Missing Not At Random. The counterfactual inverse propensity scoring (IPS) was used to weight the imputation error of every observed rating. Although effective in multiple scenarios, we argue that the performance of IPS estimation is limited due to the uncertainty miscalibration of propensity estimation. In this paper, we propose the uncertainty calibration for the propensity estimation in recommendation systems with multiple representative uncertainty calibration techniques. Theoretical analysis on the bias and generalization bound shows the superiority of the calibrated IPS estimator over the uncalibrated one. Experimental results on the coat and yahoo datasets shows that the uncertainty calibration is improved and hence brings the better recommendation results.
    

