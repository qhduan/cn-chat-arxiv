# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficiently Assemble Normalization Layers and Regularization for Federated Domain Generalization](https://arxiv.org/abs/2403.15605) | 引入了一种新颖的FedDG架构方法gPerXAN，通过规范化方案和引导正则化器配合工作，实现了个性化显式组装规范化，有助于客户端模型对领域特征进行有选择性过滤。 |
| [^2] | [Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](https://arxiv.org/abs/2402.15751) | 提出了一种稀疏MeZO方法，通过仅对精心选择的参数子集应用零阶优化，实现了在零阶LLM微调中减少参数以获得更好性能的目标 |
| [^3] | [Large (and Deep) Factor Models](https://arxiv.org/abs/2402.06635) | 本文通过证明一个足够宽而任意深的神经网络训练出来的投资组合优化模型与大型因子模型等效，打开了深度学习在此领域中的黑盒子，并提供了一种封闭形式的推导方法。研究实证了不同架构选择对模型性能的影响，并证明了随着深度增加，模型在足够多数据下的表现逐渐提升，直至达到饱和。 |
| [^4] | [Variational DAG Estimation via State Augmentation With Stochastic Permutations](https://arxiv.org/abs/2402.02644) | 使用状态扩展和随机排列进行变分DAG估计的方法可以超越竞争的贝叶斯和非贝叶斯基准方法，从而在估计贝叶斯网络结构方面取得更好的性能。 |
| [^5] | [A Survey on Generative Modeling with Limited Data, Few Shots, and Zero Shot.](http://arxiv.org/abs/2307.14397) | 本文调查了在有限数据、少样本和零样本条件下学习生成模型的情况，并提出了关于任务和方法的分类体系，研究了它们之间的相互作用，并探讨了未来的研究方向。 |
| [^6] | [Learning with Subset Stacking.](http://arxiv.org/abs/2112.06251) | 提出了一种新的回归算法LESS，通过生成以随机点为中心的子集并训练局部预测器，然后以新颖的方式组合预测器得到整体预测器。在多个数据集上测试表明，LESS是一种有竞争力且高效的监督学习方法。 |

# 详细

[^1]: 为联邦领域泛化高效组合规范化层与正则化方法

    Efficiently Assemble Normalization Layers and Regularization for Federated Domain Generalization

    [https://arxiv.org/abs/2403.15605](https://arxiv.org/abs/2403.15605)

    引入了一种新颖的FedDG架构方法gPerXAN，通过规范化方案和引导正则化器配合工作，实现了个性化显式组装规范化，有助于客户端模型对领域特征进行有选择性过滤。

    

    领域转移是机器学习中一个严峻的问题，会导致模型在未知领域测试时性能下降。联邦领域泛化（FedDG）旨在以隐私保护的方式使用协作客户端训练全局模型，能够很好地泛化到可能存在领域转移的未知客户端。然而，大多数现有的FedDG方法可能会导致额外的数据泄露隐私风险，或者在客户端通信和计算成本方面产生显著开销，这在联邦学习范式中是主要关注的问题。为了解决这些挑战，我们引入了一种新颖的FedDG架构方法，即gPerXAN，它依赖于一个规范化方案与引导正则化器配合工作。具体来说，我们精心设计了个性化显式组装规范化，以强制客户端模型有选择地过滤对本地数据有偏向的特定领域特征。

    arXiv:2403.15605v1 Announce Type: cross  Abstract: Domain shift is a formidable issue in Machine Learning that causes a model to suffer from performance degradation when tested on unseen domains. Federated Domain Generalization (FedDG) attempts to train a global model using collaborative clients in a privacy-preserving manner that can generalize well to unseen clients possibly with domain shift. However, most existing FedDG methods either cause additional privacy risks of data leakage or induce significant costs in client communication and computation, which are major concerns in the Federated Learning paradigm. To circumvent these challenges, here we introduce a novel architectural method for FedDG, namely gPerXAN, which relies on a normalization scheme working with a guiding regularizer. In particular, we carefully design Personalized eXplicitly Assembled Normalization to enforce client models selectively filtering domain-specific features that are biased towards local data while ret
    
[^2]: 稀疏MeZO：在零阶LLM微调中减少参数以获得更好性能

    Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning

    [https://arxiv.org/abs/2402.15751](https://arxiv.org/abs/2402.15751)

    提出了一种稀疏MeZO方法，通过仅对精心选择的参数子集应用零阶优化，实现了在零阶LLM微调中减少参数以获得更好性能的目标

    

    在针对特定任务进行大型语言模型（LLMs）微调通常会产生令人印象深刻的结果，但由于基于梯度的训练中的反向传播而导致内存效率低下。最近提出的高效利用存储器的零阶（MeZO）优化器旨在解决这个问题，在训练过程中只需要前向传递，使其更符合内存友好性。然而，零阶优化中梯度估计的质量往往取决于数据的维数，这可能解释了为什么与各种任务中的标准微调相比，MeZO仍然表现出显著的性能下降。受到参数高效微调（PEFT）成功的启发，本文介绍了稀疏MeZO，这是一种新颖的内存高效的零阶优化方法，仅将ZO应用于精心选择的参数子集。我们提出了一种简单而有效的参数选择方案，获得了显著的性能提升。

    arXiv:2402.15751v1 Announce Type: cross  Abstract: While fine-tuning large language models (LLMs) for specific tasks often yields impressive results, it comes at the cost of memory inefficiency due to back-propagation in gradient-based training. Memory-efficient Zeroth-order (MeZO) optimizers, recently proposed to address this issue, only require forward passes during training, making them more memory-friendly. However, the quality of gradient estimates in zeroth order optimization often depends on the data dimensionality, potentially explaining why MeZO still exhibits significant performance drops compared to standard fine-tuning across various tasks. Inspired by the success of Parameter-Efficient Fine-Tuning (PEFT), this paper introduces Sparse MeZO, a novel memory-efficient zeroth-order optimization approach that applies ZO only to a carefully chosen subset of parameters. We propose a simple yet effective parameter selection scheme that yields significant performance gains with Spar
    
[^3]: 大型（和深度）因子模型

    Large (and Deep) Factor Models

    [https://arxiv.org/abs/2402.06635](https://arxiv.org/abs/2402.06635)

    本文通过证明一个足够宽而任意深的神经网络训练出来的投资组合优化模型与大型因子模型等效，打开了深度学习在此领域中的黑盒子，并提供了一种封闭形式的推导方法。研究实证了不同架构选择对模型性能的影响，并证明了随着深度增加，模型在足够多数据下的表现逐渐提升，直至达到饱和。

    

    我们打开了深度学习在投资组合优化中的黑盒子，并证明了一个足够宽而任意深的神经网络(DNN)被训练用来最大化随机贴现因子(SDF)的夏普比率等效于一个大型因子模型(LFM)：一个使用许多非线性特征的线性因子定价模型。这些特征的性质取决于DNN的体系结构，在一种明确可追踪的方式下。这使得首次可以推导出封闭形式的端到端训练的基于DNN的SDF。我们通过实证评估了LFMs，并展示了各种架构选择如何影响SDF的性能。我们证明了深度复杂性的优点：随着足够多的数据，DNN-SDF的外样总体表现会随着神经网络的深度而增加，当隐藏层达到约100层时达到饱和。

    We open up the black box behind Deep Learning for portfolio optimization and prove that a sufficiently wide and arbitrarily deep neural network (DNN) trained to maximize the Sharpe ratio of the Stochastic Discount Factor (SDF) is equivalent to a large factor model (LFM): A linear factor pricing model that uses many non-linear characteristics. The nature of these characteristics depends on the architecture of the DNN in an explicit, tractable fashion. This makes it possible to derive end-to-end trained DNN-based SDFs in closed form for the first time. We evaluate LFMs empirically and show how various architectural choices impact SDF performance. We document the virtue of depth complexity: With enough data, the out-of-sample performance of DNN-SDF is increasing in the NN depth, saturating at huge depths of around 100 hidden layers.
    
[^4]: 通过状态扩展和随机排列的方法进行变分DAG估计

    Variational DAG Estimation via State Augmentation With Stochastic Permutations

    [https://arxiv.org/abs/2402.02644](https://arxiv.org/abs/2402.02644)

    使用状态扩展和随机排列进行变分DAG估计的方法可以超越竞争的贝叶斯和非贝叶斯基准方法，从而在估计贝叶斯网络结构方面取得更好的性能。

    

    从观测数据中估计贝叶斯网络的结构，即有向无环图（DAG），是一个在统计和计算上都很困难的问题，在因果发现等领域有着重要应用。贝叶斯方法在解决这个任务方面是一个有希望的方向，因为它们允许进行不确定性量化，并处理众所周知的可识别性问题。从概率推断的角度来看，主要的挑战是（i）表示满足DAG约束的图的分布和（ii）估计底层组合空间的后验概率。我们提出了一种方法，通过在DAG和排列的扩展空间上构建联合分布来解决这些挑战。我们通过变分推断进行后验估计，在其中利用了离散分布的连续松弛。我们展示了我们的方法在一系列合成和实际数据上能够超越竞争的贝叶斯和非贝叶斯基准方法。

    Estimating the structure of a Bayesian network, in the form of a directed acyclic graph (DAG), from observational data is a statistically and computationally hard problem with essential applications in areas such as causal discovery. Bayesian approaches are a promising direction for solving this task, as they allow for uncertainty quantification and deal with well-known identifiability issues. From a probabilistic inference perspective, the main challenges are (i) representing distributions over graphs that satisfy the DAG constraint and (ii) estimating a posterior over the underlying combinatorial space. We propose an approach that addresses these challenges by formulating a joint distribution on an augmented space of DAGs and permutations. We carry out posterior estimation via variational inference, where we exploit continuous relaxations of discrete distributions. We show that our approach can outperform competitive Bayesian and non-Bayesian benchmarks on a range of synthetic and re
    
[^5]: 关于有限数据、少样本和零样本情况下生成建模的调查

    A Survey on Generative Modeling with Limited Data, Few Shots, and Zero Shot. (arXiv:2307.14397v1 [cs.CV])

    [http://arxiv.org/abs/2307.14397](http://arxiv.org/abs/2307.14397)

    本文调查了在有限数据、少样本和零样本条件下学习生成模型的情况，并提出了关于任务和方法的分类体系，研究了它们之间的相互作用，并探讨了未来的研究方向。

    

    在机器学习中，生成建模旨在学习生成与训练数据分布统计相似的新数据。本文调查了在有限数据、少样本和零样本条件下学习生成模型的情况，称为数据约束下的生成建模（GM-DC）。这是一个重要的主题，当数据获取具有挑战性时，例如医疗应用。我们讨论了背景、挑战，并提出了两个分类体系：一个是GM-DC任务分类，另一个是GM-DC方法分类。重要的是，我们研究了不同GM-DC任务和方法之间的相互作用。此外，我们还强调了研究空白、研究趋势和未来探索的潜在途径。项目网站：https://gmdc-survey.github.io。

    In machine learning, generative modeling aims to learn to generate new data statistically similar to the training data distribution. In this paper, we survey learning generative models under limited data, few shots and zero shot, referred to as Generative Modeling under Data Constraint (GM-DC). This is an important topic when data acquisition is challenging, e.g. healthcare applications. We discuss background, challenges, and propose two taxonomies: one on GM-DC tasks and another on GM-DC approaches. Importantly, we study interactions between different GM-DC tasks and approaches. Furthermore, we highlight research gaps, research trends, and potential avenues for future exploration. Project website: https://gmdc-survey.github.io.
    
[^6]: 学习与子集叠加

    Learning with Subset Stacking. (arXiv:2112.06251v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2112.06251](http://arxiv.org/abs/2112.06251)

    提出了一种新的回归算法LESS，通过生成以随机点为中心的子集并训练局部预测器，然后以新颖的方式组合预测器得到整体预测器。在多个数据集上测试表明，LESS是一种有竞争力且高效的监督学习方法。

    

    我们提出了一种新的回归算法，该算法从一组输入-输出对中进行学习。我们的算法适用于输入变量与输出变量之间的关系在预测空间中表现出异质行为的群体。该算法首先生成以输入空间中的随机点为中心的子集，然后为每个子集训练一个局部预测器。然后这些预测器以一种新颖的方式组合在一起，形成一个整体预测器。我们将此算法称为“学习与子集叠加”或LESS，因为它类似于叠加回归器的方法。我们将LESS与多个数据集上的最先进方法进行测试性能比较。我们的比较结果表明，LESS是一种有竞争力的监督学习方法。此外，我们观察到LESS在计算时间上也非常高效，并且可以直接进行并行实现。

    We propose a new regression algorithm that learns from a set of input-output pairs. Our algorithm is designed for populations where the relation between the input variables and the output variable exhibits a heterogeneous behavior across the predictor space. The algorithm starts with generating subsets that are concentrated around random points in the input space. This is followed by training a local predictor for each subset. Those predictors are then combined in a novel way to yield an overall predictor. We call this algorithm ``LEarning with Subset Stacking'' or LESS, due to its resemblance to the method of stacking regressors. We compare the testing performance of LESS with state-of-the-art methods on several datasets. Our comparison shows that LESS is a competitive supervised learning method. Moreover, we observe that LESS is also efficient in terms of computation time and it allows a straightforward parallel implementation.
    

