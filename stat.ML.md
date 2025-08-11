# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Random Feature Model.](http://arxiv.org/abs/2310.04417) | 本研究提出了一种以扩散模型为灵感的深度随机特征模型，它具有可解释性并可在数量相同的可训练参数下与全连接神经网络提供可比较的数值结果。通过推导得分匹配的属性，我们扩展了现有随机特征结果，并得出了样本数据分布与真实分布之间的泛化边界。 |
| [^2] | [Double Normalizing Flows: Flexible Bayesian Gaussian Process ODEs Learning.](http://arxiv.org/abs/2309.09222) | 这项研究将标准化流引入高斯过程常微分方程(ODE)模型，使其具备更灵活和表达性强的先验分布和非高斯的后验推断，从而提高了贝叶斯高斯过程ODE的准确性和不确定性估计。 |
| [^3] | [Hypergraph Structure Inference From Data Under Smoothness Prior.](http://arxiv.org/abs/2308.14172) | 本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。 |

# 详细

[^1]: 扩散随机特征模型

    Diffusion Random Feature Model. (arXiv:2310.04417v1 [stat.ML])

    [http://arxiv.org/abs/2310.04417](http://arxiv.org/abs/2310.04417)

    本研究提出了一种以扩散模型为灵感的深度随机特征模型，它具有可解释性并可在数量相同的可训练参数下与全连接神经网络提供可比较的数值结果。通过推导得分匹配的属性，我们扩展了现有随机特征结果，并得出了样本数据分布与真实分布之间的泛化边界。

    

    扩散概率模型已成功用于生成从噪声中产生的数据。然而，大多数扩散模型计算成本高昂，难以解释，缺乏理论依据。另一方面，由于其可解释性，随机特征模型变得越来越受欢迎，但其在复杂机器学习任务中的应用仍然有限。在本工作中，我们提出了一种受扩散模型启发的深度随机特征模型，它既具有可解释性，又能给出与具有相同可训练参数数量的全连接神经网络相当的数值结果。具体而言，我们扩展了现有的随机特征结果，利用得分匹配的属性导出了样本数据分布与真实分布之间的泛化边界。我们通过在时尚MNIST数据集和乐器音频数据上生成样本来验证我们的发现。

    Diffusion probabilistic models have been successfully used to generate data from noise. However, most diffusion models are computationally expensive and difficult to interpret with a lack of theoretical justification. Random feature models on the other hand have gained popularity due to their interpretability but their application to complex machine learning tasks remains limited. In this work, we present a diffusion model-inspired deep random feature model that is interpretable and gives comparable numerical results to a fully connected neural network having the same number of trainable parameters. Specifically, we extend existing results for random features and derive generalization bounds between the distribution of sampled data and the true distribution using properties of score matching. We validate our findings by generating samples on the fashion MNIST dataset and instrumental audio data.
    
[^2]: 双重标准化流：灵活的贝叶斯高斯过程ODE学习

    Double Normalizing Flows: Flexible Bayesian Gaussian Process ODEs Learning. (arXiv:2309.09222v1 [cs.LG])

    [http://arxiv.org/abs/2309.09222](http://arxiv.org/abs/2309.09222)

    这项研究将标准化流引入高斯过程常微分方程(ODE)模型，使其具备更灵活和表达性强的先验分布和非高斯的后验推断，从而提高了贝叶斯高斯过程ODE的准确性和不确定性估计。

    

    最近，高斯过程被用来建模连续动力系统的向量场。对于这样的模型，贝叶斯推断已经得到了广泛研究，并应用于时间序列预测等任务，提供不确定性估计。然而，先前的高斯过程常微分方程(ODE)模型在具有非高斯过程先验的数据集上可能表现不佳，因为它们的约束先验和均值场后验可能缺乏灵活性。为了解决这个限制，我们引入了标准化流来重新参数化ODE的向量场，从而得到一个更灵活、更表达性的先验分布。此外，由于标准化流的解析可计算的概率密度函数，我们将它们应用于GP ODE的后验推断，生成一个非高斯的后验。通过这些标准化流的双重应用，我们的模型在贝叶斯高斯过程ODE中提高了准确性和不确定性估计。

    Recently, Gaussian processes have been utilized to model the vector field of continuous dynamical systems. Bayesian inference for such models \cite{hegde2022variational} has been extensively studied and has been applied in tasks such as time series prediction, providing uncertain estimates. However, previous Gaussian Process Ordinary Differential Equation (ODE) models may underperform on datasets with non-Gaussian process priors, as their constrained priors and mean-field posteriors may lack flexibility. To address this limitation, we incorporate normalizing flows to reparameterize the vector field of ODEs, resulting in a more flexible and expressive prior distribution. Additionally, due to the analytically tractable probability density functions of normalizing flows, we apply them to the posterior inference of GP ODEs, generating a non-Gaussian posterior. Through these dual applications of normalizing flows, our model improves accuracy and uncertainty estimates for Bayesian Gaussian P
    
[^3]: 从数据中基于光滑性先验推断超图结构

    Hypergraph Structure Inference From Data Under Smoothness Prior. (arXiv:2308.14172v1 [cs.LG])

    [http://arxiv.org/abs/2308.14172](http://arxiv.org/abs/2308.14172)

    本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。

    

    超图在处理涉及多个实体的高阶关系数据中非常重要。在没有明确超图可用的情况下，希望能够从节点特征中推断出有意义的超图结构，以捕捉数据内在的关系。然而，现有的方法要么采用简单预定义的规则，不能精确捕捉潜在超图结构的分布，要么学习超图结构和节点特征之间的映射，但需要大量标记数据（即预先存在的超图结构）进行训练。这两种方法都局限于实际情景中的应用。为了填补这一空白，我们提出了一种新的光滑性先验，使我们能够设计一种方法，在没有标记数据作为监督的情况下推断出每个潜在超边的概率。所提出的先验表示超边中的节点特征与包含该超边的超边的特征高度相关。

    Hypergraphs are important for processing data with higher-order relationships involving more than two entities. In scenarios where explicit hypergraphs are not readily available, it is desirable to infer a meaningful hypergraph structure from the node features to capture the intrinsic relations within the data. However, existing methods either adopt simple pre-defined rules that fail to precisely capture the distribution of the potential hypergraph structure, or learn a mapping between hypergraph structures and node features but require a large amount of labelled data, i.e., pre-existing hypergraph structures, for training. Both restrict their applications in practical scenarios. To fill this gap, we propose a novel smoothness prior that enables us to design a method to infer the probability for each potential hyperedge without labelled data as supervision. The proposed prior indicates features of nodes in a hyperedge are highly correlated by the features of the hyperedge containing th
    

