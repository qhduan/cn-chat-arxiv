# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Model Stock: All we need is just a few fine-tuned models](https://arxiv.org/abs/2403.19522) | 本文提出了一种高效的微调方法，只使用少量模型就能获得优越的性能，通过权重空间和层次加权平均技术超越了现有的模型方法。 |
| [^2] | [The Developmental Landscape of In-Context Learning](https://arxiv.org/abs/2402.02364) | 在transformers模型中，我们展示了在上下文学习中的离散发展阶段，并引入了两种方法来检测这些阶段的关键里程碑。我们使用行为和结构度量验证了这些方法的有效性。 |
| [^3] | [Bagged Regularized $k$-Distances for Anomaly Detection](https://arxiv.org/abs/2312.01046) | 本文提出了一种称为Bagged Regularized $k$-Distances for Anomaly Detection (BRDAD)的基于距离的算法，通过将非监督异常检测问题转化为凸优化问题，成功解决了基于距离算法中超参数选择的敏感性挑战，并通过包集成方法解决了处理大规模数据集时的效率问题。 |
| [^4] | [Scalable manifold learning by uniform landmark sampling and constrained locally linear embedding.](http://arxiv.org/abs/2401.01100) | 通过均匀地标抽样和约束局部线性嵌入，提出了一种可伸缩的流形学习方法，可以有效处理大规模和高维数据，并解决全局结构失真和可伸缩性问题。 |
| [^5] | [Size Generalizability of Graph Neural Networks on Biological Data: Insights and Practices from the Spectral Perspective.](http://arxiv.org/abs/2305.15611) | 本文通过谱角度的方法，研究了GNNs的尺寸可泛化性问题，并在真实生物数据集上进行了实验，发现GNNs在度分布和谱分布偏移时均表现敏感，在同一数据集的大图上的性能仍然下降，揭示了 GNNs的尺寸可泛化性问题。 |
| [^6] | [Gradient Leakage Defense with Key-Lock Module for Federated Learning.](http://arxiv.org/abs/2305.04095) | 本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。 |
| [^7] | [A Generative Modeling Approach Using Quantum Gates.](http://arxiv.org/abs/2303.16955) | 本文提出了一种使用量子门生成新样本的生成建模方法，并在不同数据集上进行了实验证明其有效性。 |

# 详细

[^1]: 模型库：我们只需要几个经过良好调整的模型

    Model Stock: All we need is just a few fine-tuned models

    [https://arxiv.org/abs/2403.19522](https://arxiv.org/abs/2403.19522)

    本文提出了一种高效的微调方法，只使用少量模型就能获得优越的性能，通过权重空间和层次加权平均技术超越了现有的模型方法。

    

    本文介绍了一种高效的大型预训练模型微调方法，提供强大的内分布（ID）和外分布（OOD）性能。与需要大量微调模型进行平均的传统做法不同，我们的方法使用更少的模型来获得最终权重，同时产生更高的准确性。从微调权重的权重空间中汲取关键见解，我们揭示了性能和接近权重空间中心的强连接。基于此，我们引入一种方法，通过仅使用两个微调模型来近似中心接近的权重，可在训练期间或之后应用。我们的创新的逐层权重平均技术超越了Model Soup等最先进的模型方法，仅利用两个微调模型。这种策略可以被称为模型库，突出了它依赖于选择少量模型来进行综合的特点。

    arXiv:2403.19522v1 Announce Type: new  Abstract: This paper introduces an efficient fine-tuning method for large pre-trained models, offering strong in-distribution (ID) and out-of-distribution (OOD) performance. Breaking away from traditional practices that need a multitude of fine-tuned models for averaging, our approach employs significantly fewer models to achieve final weights yet yield superior accuracy. Drawing from key insights in the weight space of fine-tuned weights, we uncover a strong link between the performance and proximity to the center of weight space. Based on this, we introduce a method that approximates a center-close weight using only two fine-tuned models, applicable during or after training. Our innovative layer-wise weight averaging technique surpasses state-of-the-art model methods such as Model Soup, utilizing only two fine-tuned models. This strategy can be aptly coined Model Stock, highlighting its reliance on selecting a minimal number of models to draw a 
    
[^2]: 在上下文中学习的发展景观

    The Developmental Landscape of In-Context Learning

    [https://arxiv.org/abs/2402.02364](https://arxiv.org/abs/2402.02364)

    在transformers模型中，我们展示了在上下文学习中的离散发展阶段，并引入了两种方法来检测这些阶段的关键里程碑。我们使用行为和结构度量验证了这些方法的有效性。

    

    我们展示了在transformers中，当它们通过语言建模或线性回归任务进行训练时，上下文学习是如何以离散的发展阶段出现的。我们引入了两种方法来检测分隔这些阶段的关键里程碑，通过探测参数空间和函数空间中种群损失的几何特征。我们使用一系列行为和结构度量研究这些新方法揭示的阶段，以建立它们的有效性。

    We show that in-context learning emerges in transformers in discrete developmental stages, when they are trained on either language modeling or linear regression tasks. We introduce two methods for detecting the milestones that separate these stages, by probing the geometry of the population loss in both parameter space and function space. We study the stages revealed by these new methods using a range of behavioral and structural metrics to establish their validity.
    
[^3]: Bagged Regularized $k$-Distances用于异常检测

    Bagged Regularized $k$-Distances for Anomaly Detection

    [https://arxiv.org/abs/2312.01046](https://arxiv.org/abs/2312.01046)

    本文提出了一种称为Bagged Regularized $k$-Distances for Anomaly Detection (BRDAD)的基于距离的算法，通过将非监督异常检测问题转化为凸优化问题，成功解决了基于距离算法中超参数选择的敏感性挑战，并通过包集成方法解决了处理大规模数据集时的效率问题。

    

    本文考虑非监督异常检测的范式，即在没有标记的情况下识别数据集中的异常值。尽管基于距离的方法对于非监督异常检测具有较好的性能，但它们对最近邻数量的选择非常敏感。为此，我们提出了一种新的基于距离的算法，称为Bagged Regularized $k$-Distances for Anomaly Detection (BRDAD)，将非监督异常检测问题转化为凸优化问题。我们的BRDAD算法通过最小化替代风险（即经验风险的有限样本上界）来选择权重，以用于密度估计的带权重的$k$-distances。这种方法成功解决了基于距离算法中超参数选择的敏感性挑战。此外，在处理大规模数据集时，我们还可以通过包集成的方法来解决效率问题。

    We consider the paradigm of unsupervised anomaly detection, which involves the identification of anomalies within a dataset in the absence of labeled examples. Though distance-based methods are top-performing for unsupervised anomaly detection, they suffer heavily from the sensitivity to the choice of the number of the nearest neighbors. In this paper, we propose a new distance-based algorithm called bagged regularized $k$-distances for anomaly detection (BRDAD) converting the unsupervised anomaly detection problem into a convex optimization problem. Our BRDAD algorithm selects the weights by minimizing the surrogate risk, i.e., the finite sample bound of the empirical risk of the bagged weighted $k$-distances for density estimation (BWDDE). This approach enables us to successfully address the sensitivity challenge of the hyperparameter choice in distance-based algorithms. Moreover, when dealing with large-scale datasets, the efficiency issues can be addressed by the incorporated baggi
    
[^4]: 通过均匀地标抽样和约束局部线性嵌入实现可伸缩的流形学习

    Scalable manifold learning by uniform landmark sampling and constrained locally linear embedding. (arXiv:2401.01100v1 [cs.LG])

    [http://arxiv.org/abs/2401.01100](http://arxiv.org/abs/2401.01100)

    通过均匀地标抽样和约束局部线性嵌入，提出了一种可伸缩的流形学习方法，可以有效处理大规模和高维数据，并解决全局结构失真和可伸缩性问题。

    

    流形学习是机器学习和数据科学中的关键方法，旨在揭示高维空间中复杂非线性流形内在的低维结构。通过利用流形假设，已经开发了各种非线性降维技术来促进可视化、分类、聚类和获得关键洞察。虽然现有的流形学习方法取得了显著的成功，但仍然存在全局结构中的大量失真问题，这阻碍了对底层模式的理解。可伸缩性问题也限制了它们处理大规模数据的适用性。在这里，我们提出了一种可伸缩的流形学习(scML)方法，可以以有效的方式处理大规模和高维数据。它通过寻找一组地标来构建整个数据的低维骨架，然后将非地标引入地标空间中

    As a pivotal approach in machine learning and data science, manifold learning aims to uncover the intrinsic low-dimensional structure within complex nonlinear manifolds in high-dimensional space. By exploiting the manifold hypothesis, various techniques for nonlinear dimension reduction have been developed to facilitate visualization, classification, clustering, and gaining key insights. Although existing manifold learning methods have achieved remarkable successes, they still suffer from extensive distortions incurred in the global structure, which hinders the understanding of underlying patterns. Scalability issues also limit their applicability for handling large-scale data. Here, we propose a scalable manifold learning (scML) method that can manipulate large-scale and high-dimensional data in an efficient manner. It starts by seeking a set of landmarks to construct the low-dimensional skeleton of the entire data and then incorporates the non-landmarks into the landmark space based 
    
[^5]: 基于谱角度剖析生物数据中图神经网络的尺寸可泛化性：观点和实践

    Size Generalizability of Graph Neural Networks on Biological Data: Insights and Practices from the Spectral Perspective. (arXiv:2305.15611v1 [cs.LG])

    [http://arxiv.org/abs/2305.15611](http://arxiv.org/abs/2305.15611)

    本文通过谱角度的方法，研究了GNNs的尺寸可泛化性问题，并在真实生物数据集上进行了实验，发现GNNs在度分布和谱分布偏移时均表现敏感，在同一数据集的大图上的性能仍然下降，揭示了 GNNs的尺寸可泛化性问题。

    

    本文探讨了图神经网络 (GNNs) 是否具有从小图中学习的知识可推广到同一领域的大图中。之前的研究表明，不同大小的图之间的分布偏移，尤其是度分布，可能会导致图分类任务的性能下降。然而，在生物数据集中，度数是有界的，因此度分布的偏移很小。即使度分布偏移很小，我们观察到GNNs在同一数据集的大图上的性能仍然下降，暗示有其他原因。事实上，以往对于真实数据集中各种图尺寸引起的分布偏移类型和属性的探索不足。此外，以前的尺寸可泛化性分析大多集中在空间领域。为填补这些空白，我们采用谱角度去研究GNNs在生物图数据上的尺寸可泛化性。我们首先提出一个新框架来模拟各种类型的度分布偏移，并利用它来测试GNNs 在真实生物数据集上的尺寸可泛化性。我们的实验表明，除了度分布偏移外，GNNs 还对图大小变化引起的谱分布偏移很敏感。我们进一步分析了不同的GNN模型的影响，并表明，一些模型比其他模型更具有尺寸泛化性。本文展示了关于GNNs尺寸可泛化性问题的新观点和实践，并为该领域的未来研究提供了有益的洞察和建议。

    We investigate the question of whether the knowledge learned by graph neural networks (GNNs) from small graphs is generalizable to large graphs in the same domain. Prior works suggest that the distribution shift, particularly in the degree distribution, between graphs of different sizes can lead to performance degradation in the graph classification task. However, this may not be the case for biological datasets where the degrees are bounded and the distribution shift of degrees is small. Even with little degree distribution shift, our observations show that GNNs' performance on larger graphs from the same datasets still degrades, suggesting other causes. In fact, there has been a lack of exploration in real datasets to understand the types and properties of distribution shifts caused by various graph sizes. Furthermore, previous analyses of size generalizability mostly focus on the spatial domain.  To fill these gaps, we take the spectral perspective and study the size generalizabilit
    
[^6]: 基于密钥锁模块的联邦学习梯度泄露防御

    Gradient Leakage Defense with Key-Lock Module for Federated Learning. (arXiv:2305.04095v1 [cs.LG])

    [http://arxiv.org/abs/2305.04095](http://arxiv.org/abs/2305.04095)

    本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。

    

    联邦学习是一种广泛采用的隐私保护机器学习方法，其中私有数据保持本地，允许安全计算和本地模型梯度与第三方参数服务器之间的交换。然而，最近的研究发现，通过共享的梯度可能会危及隐私并恢复敏感信息。本研究提供了详细的分析和对梯度泄漏问题的新视角。这些理论工作导致了一种新的梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构。只有锁定的梯度被传输到参数服务器进行全局模型聚合。我们提出的学习方法对梯度泄露攻击具有抵抗力，并且所设计和训练的密钥锁模块可以确保，没有密钥锁模块的私有信息：a) 无法从共享的梯度中重建私有训练数据。

    Federated Learning (FL) is a widely adopted privacy-preserving machine learning approach where private data remains local, enabling secure computations and the exchange of local model gradients between local clients and third-party parameter servers. However, recent findings reveal that privacy may be compromised and sensitive information potentially recovered from shared gradients. In this study, we offer detailed analysis and a novel perspective on understanding the gradient leakage problem. These theoretical works lead to a new gradient leakage defense technique that secures arbitrary model architectures using a private key-lock module. Only the locked gradient is transmitted to the parameter server for global model aggregation. Our proposed learning method is resistant to gradient leakage attacks, and the key-lock module is designed and trained to ensure that, without the private information of the key-lock module: a) reconstructing private training data from the shared gradient is
    
[^7]: 一种使用量子门的生成建模方法

    A Generative Modeling Approach Using Quantum Gates. (arXiv:2303.16955v1 [quant-ph])

    [http://arxiv.org/abs/2303.16955](http://arxiv.org/abs/2303.16955)

    本文提出了一种使用量子门生成新样本的生成建模方法，并在不同数据集上进行了实验证明其有效性。

    

    近些年来，量子计算作为解决复杂计算问题的有前途的技术，开始变得越来越流行。生成建模是一种技术，可以让我们学习并生成类似于原数据集的新数据样本。本文提出了一种使用量子门生成新样本的生成建模方法。我们从简要介绍量子计算和生成建模开始。接着，我们描述了我们的方法，这种方法涉及到将数据集编码成量子态，并使用量子门来操作这些状态以生成新样本。我们还提供了我们方法的数学细节，并通过在各种数据集上的实验结果来证明其有效性。

    In recent years, quantum computing has emerged as a promising technology for solving complex computational problems. Generative modeling is a technique that allows us to learn and generate new data samples similar to the original dataset. In this paper, we propose a generative modeling approach using quantum gates to generate new samples from a given dataset. We start with a brief introduction to quantum computing and generative modeling. Then, we describe our proposed approach, which involves encoding the dataset into quantum states and using quantum gates to manipulate these states to generate new samples. We also provide mathematical details of our approach and demonstrate its effectiveness through experimental results on various datasets.
    

