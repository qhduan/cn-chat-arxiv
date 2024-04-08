# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RAT: Retrieval-Augmented Transformer for Click-Through Rate Prediction](https://arxiv.org/abs/2404.02249) | RAT模型是为了解决当前CTR预测模型仅关注样本内特征交互而忽略跨样本关系的问题，通过检索相似样本构建增强输入，实现了对样本内和跨样本的全面特征交互推理，提高了CTR预测的效果。 |
| [^2] | [Survey of Computerized Adaptive Testing: A Machine Learning Perspective](https://arxiv.org/abs/2404.00712) | 本文以机器学习视角综述了计算机自适应测试（CAT），重点解析其测试问题选择算法和如何优化认知诊断模型、题库构建和测试控制。 |
| [^3] | [Text mining arXiv: a look through quantitative finance papers.](http://arxiv.org/abs/2401.01751) | 本文通过文本挖掘技术和自然语言处理方法，研究了arXiv上的量化金融论文，发现了关于该领域的时间趋势、最常被引用的研究人员和期刊，以及不同算法进行主题建模的比较。 |
| [^4] | [Generalized Rainbow Differential Privacy.](http://arxiv.org/abs/2309.05871) | 通过随机图着色，我们引入了一种名为彩虹差分隐私的新的差分隐私框架，其中不同的彩虹将连接的数据集图分割成不同的区域，并且我们证明了存在一个唯一的最优$(\epsilon,\delta)$-DP机制来保护边界上具有相同彩虹的数据集的隐私。 |

# 详细

[^1]: RAT: 检索增强变换器用于点击率预测

    RAT: Retrieval-Augmented Transformer for Click-Through Rate Prediction

    [https://arxiv.org/abs/2404.02249](https://arxiv.org/abs/2404.02249)

    RAT模型是为了解决当前CTR预测模型仅关注样本内特征交互而忽略跨样本关系的问题，通过检索相似样本构建增强输入，实现了对样本内和跨样本的全面特征交互推理，提高了CTR预测的效果。

    

    预测点击率（CTR）是Web应用程序的基本任务，其中一个关键问题是设计有效的特征交互模型。目前的方法主要集中于对单个样本内的特征交互进行建模，而忽略了可以作为参考背景来增强预测的潜在跨样本关系。为弥补这种不足，本文开发了一种检索增强变换器（RAT），旨在获取样本内和跨样本之间的细粒度特征交互。通过检索相似样本，我们为每个目标样本构建增强输入。然后利用级联注意力构建Transformer层，以捕获样本内和跨样本特征交互，促进全面推理以改善CTR预测的同时保持效率。对真实世界数据集的大量实验证实了RAT的有效性，并提出了

    arXiv:2404.02249v1 Announce Type: cross  Abstract: Predicting click-through rates (CTR) is a fundamental task for Web applications, where a key issue is to devise effective models for feature interactions. Current methodologies predominantly concentrate on modeling feature interactions within an individual sample, while overlooking the potential cross-sample relationships that can serve as a reference context to enhance the prediction. To make up for such deficiency, this paper develops a Retrieval-Augmented Transformer (RAT), aiming to acquire fine-grained feature interactions within and across samples. By retrieving similar samples, we construct augmented input for each target sample. We then build Transformer layers with cascaded attention to capture both intra- and cross-sample feature interactions, facilitating comprehensive reasoning for improved CTR prediction while retaining efficiency. Extensive experiments on real-world datasets substantiate the effectiveness of RAT and sugge
    
[^2]: 计算机自适应测试综述：机器学习视角

    Survey of Computerized Adaptive Testing: A Machine Learning Perspective

    [https://arxiv.org/abs/2404.00712](https://arxiv.org/abs/2404.00712)

    本文以机器学习视角综述了计算机自适应测试（CAT），重点解析其测试问题选择算法和如何优化认知诊断模型、题库构建和测试控制。

    

    计算机自适应测试（CAT）提供了一种高效、量身定制的评估考生熟练程度的方法，通过根据他们的表现动态调整测试问题。CAT广泛应用于教育、医疗、体育和社会学等多个领域，彻底改变了测试实践。然而，随着大规模测试的增加复杂性，CAT已经融合了机器学习技术。本文旨在提供一个以机器学习为重点的CAT综述，从新的角度解读这种自适应测试方法。通过研究CAT适应性核心的测试问题选择算法，我们揭示了其功能。此外，我们探讨了认知诊断模型、题库构建和CAT中的测试控制，探索了机器学习如何优化这些组成部分。通过对当前情况的分析，

    arXiv:2404.00712v1 Announce Type: cross  Abstract: Computerized Adaptive Testing (CAT) provides an efficient and tailored method for assessing the proficiency of examinees, by dynamically adjusting test questions based on their performance. Widely adopted across diverse fields like education, healthcare, sports, and sociology, CAT has revolutionized testing practices. While traditional methods rely on psychometrics and statistics, the increasing complexity of large-scale testing has spurred the integration of machine learning techniques. This paper aims to provide a machine learning-focused survey on CAT, presenting a fresh perspective on this adaptive testing method. By examining the test question selection algorithm at the heart of CAT's adaptivity, we shed light on its functionality. Furthermore, we delve into cognitive diagnosis models, question bank construction, and test control within CAT, exploring how machine learning can optimize these components. Through an analysis of curre
    
[^3]: 文本挖掘arXiv：对量化金融论文的观察

    Text mining arXiv: a look through quantitative finance papers. (arXiv:2401.01751v1 [cs.DL])

    [http://arxiv.org/abs/2401.01751](http://arxiv.org/abs/2401.01751)

    本文通过文本挖掘技术和自然语言处理方法，研究了arXiv上的量化金融论文，发现了关于该领域的时间趋势、最常被引用的研究人员和期刊，以及不同算法进行主题建模的比较。

    

    本文利用文本挖掘技术和自然语言处理方法，探索了arXiv预印本服务器上的论文，旨在发现这个庞大的研究集合中隐藏的有价值的见解。我们研究了从1997年到2022年在arXiv上发布的量化金融论文的内容。我们从整个文档中提取和分析关键信息，包括引用，以了解随时间变化的主题趋势，并找出这个领域中最常被引用的研究人员和期刊。此外，我们还比较了多种算法来进行主题建模，包括最先进的方法。

    This paper explores articles hosted on the arXiv preprint server with the aim to uncover valuable insights hidden in this vast collection of research. Employing text mining techniques and through the application of natural language processing methods, we examine the contents of quantitative finance papers posted in arXiv from 1997 to 2022. We extract and analyze crucial information from the entire documents, including the references, to understand the topics trends over time and to find out the most cited researchers and journals on this domain. Additionally, we compare numerous algorithms to perform topic modeling, including state-of-the-art approaches.
    
[^4]: 广义彩虹差分隐私

    Generalized Rainbow Differential Privacy. (arXiv:2309.05871v1 [cs.CR])

    [http://arxiv.org/abs/2309.05871](http://arxiv.org/abs/2309.05871)

    通过随机图着色，我们引入了一种名为彩虹差分隐私的新的差分隐私框架，其中不同的彩虹将连接的数据集图分割成不同的区域，并且我们证明了存在一个唯一的最优$(\epsilon,\delta)$-DP机制来保护边界上具有相同彩虹的数据集的隐私。

    

    我们研究了一种通过随机图着色来设计差分隐私(DP)机制的新框架，称为彩虹差分隐私。在这个框架中，数据集是图中的节点，两个相邻的数据集通过边相连。图中的每个数据集都有一种对机制可能的输出的偏好排序，这些排序被称为彩虹。不同的彩虹将相连的数据集图分割成不同的区域。我们证明，如果在这些区域边界处的DP机制被固定，并且它在所有具有相同彩虹边界的数据集上的行为相同，那么存在一个唯一的最优$(\epsilon,\delta)$-DP机制(只要边界条件有效)并且可以用闭形式表示出来。我们的证明技巧基于优势排序和DP之间的有趣关系，适用于任意有限数量的颜色和$(\epsilon,\delta)$-DP，改进了先前只适用于至多三种颜色的结果。

    We study a new framework for designing differentially private (DP) mechanisms via randomized graph colorings, called rainbow differential privacy. In this framework, datasets are nodes in a graph, and two neighboring datasets are connected by an edge. Each dataset in the graph has a preferential ordering for the possible outputs of the mechanism, and these orderings are called rainbows. Different rainbows partition the graph of connected datasets into different regions. We show that if a DP mechanism at the boundary of such regions is fixed and it behaves identically for all same-rainbow boundary datasets, then a unique optimal $(\epsilon,\delta)$-DP mechanism exists (as long as the boundary condition is valid) and can be expressed in closed-form. Our proof technique is based on an interesting relationship between dominance ordering and DP, which applies to any finite number of colors and for $(\epsilon,\delta)$-DP, improving upon previous results that only apply to at most three color
    

