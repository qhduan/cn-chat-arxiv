# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Out-of-Distribution Detection via Deep Multi-Comprehension Ensemble](https://arxiv.org/abs/2403.16260) | 通过引入新颖的定性和定量模型集成评估方法，作者揭示了现有集成方法的关键缺陷，提出了提高传统模型集成维度的方法，以克服特征表示中的多样性限制。 |
| [^2] | [CASPER: Causality-Aware Spatiotemporal Graph Neural Networks for Spatiotemporal Time Series Imputation](https://arxiv.org/abs/2403.11960) | CASPER提出了一种因果关系感知的方法来处理时空时间序列数据插补问题，避免过度利用非因果关系，提高数据分析的准确性。 |
| [^3] | [Indirectly Parameterized Concrete Autoencoders](https://arxiv.org/abs/2403.00563) | 本文提出了间接参数化CAEs（IP-CAEs）来解决具体自编码器（CAEs）在稳定联合优化方面的问题，IP-CAEs在多个数据集上表现出显著且一致的改进。 |
| [^4] | [Multistatic-Radar RCS-Signature Recognition of Aerial Vehicles: A Bayesian Fusion Approach](https://arxiv.org/abs/2402.17987) | 提出了一种完全贝叶斯雷达自动目标识别的框架，采用最优贝叶斯融合来有效地汇总多个雷达的分类概率向量，以改进无人机雷达截面识别效果。 |
| [^5] | [Distributed Estimation and Inference for Semi-parametric Binary Response Models](https://arxiv.org/abs/2210.08393) | 通过一次性分治估计和多轮估计，本文提出了分布式环境下对半参数二元选择模型的新估计方法，实现了优化误差的超线性改进。 |
| [^6] | [Spectral Clustering for Discrete Distributions.](http://arxiv.org/abs/2401.13913) | 本文提出了一种基于谱聚类和分布相似度度量的框架来解决离散分布聚类问题。通过使用线性最优传输构建相似度矩阵，我们在聚类准确性和计算效率方面取得了显著的改进。 |
| [^7] | [Model-agnostic variable importance for predictive uncertainty: an entropy-based approach.](http://arxiv.org/abs/2310.12842) | 本文提出了一种基于熵的方法，通过扩展现有的解释性方法，可以理解不确定性感知模型中的预测来源和置信度，并利用改编后的特征重要性、部分依赖图和个体条件期望图等方法来测量特征对预测分布的熵和基于真实标签的对数似然的影响。 |
| [^8] | [Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation.](http://arxiv.org/abs/2310.02304) | 本文提出了一种自学优化器（STOP），通过递归自我改进的代码生成，使用融合了语言模型的脚手架程序来改进自身，从而生成性能更好的程序。 |
| [^9] | [Active Learning with Weak Supervision for Gaussian Processes.](http://arxiv.org/abs/2204.08335) | 本文提出了一种基于弱监督的主动学习算法，不仅选择要注释的观测结果，还选择要获得的注释精度，并在高斯过程中进行了实验验证。 |

# 详细

[^1]: 通过深度多理解集成实现越界检测

    Out-of-Distribution Detection via Deep Multi-Comprehension Ensemble

    [https://arxiv.org/abs/2403.16260](https://arxiv.org/abs/2403.16260)

    通过引入新颖的定性和定量模型集成评估方法，作者揭示了现有集成方法的关键缺陷，提出了提高传统模型集成维度的方法，以克服特征表示中的多样性限制。

    

    最近的研究强调了越界（OOD）特征表示领域规模对模型在OOD检测中效果的重要作用。因此，采用模型集成作为增强这一特征表示领域的突出策略已经成为一种突出的策略，利用预期的模型多样性。然而，我们引入了新颖的定性和定量模型集成评估方法，特别是损失盆/障碍可视化和自耦合指数，揭示了现有集成方法的一个关键缺陷。我们发现这些方法包含可进行仿射变换的权重，表现出有限的可变性，从而未能实现特征表示中所需的多样性。

    arXiv:2403.16260v1 Announce Type: cross  Abstract: Recent research underscores the pivotal role of the Out-of-Distribution (OOD) feature representation field scale in determining the efficacy of models in OOD detection. Consequently, the adoption of model ensembles has emerged as a prominent strategy to augment this feature representation field, capitalizing on anticipated model diversity.   However, our introduction of novel qualitative and quantitative model ensemble evaluation methods, specifically Loss Basin/Barrier Visualization and the Self-Coupling Index, reveals a critical drawback in existing ensemble methods. We find that these methods incorporate weights that are affine-transformable, exhibiting limited variability and thus failing to achieve the desired diversity in feature representation.   To address this limitation, we elevate the dimensions of traditional model ensembles, incorporating various factors such as different weight initializations, data holdout, etc., into di
    
[^2]: CASPER：因果关系感知时空图神经网络用于时空时间序列插补

    CASPER: Causality-Aware Spatiotemporal Graph Neural Networks for Spatiotemporal Time Series Imputation

    [https://arxiv.org/abs/2403.11960](https://arxiv.org/abs/2403.11960)

    CASPER提出了一种因果关系感知的方法来处理时空时间序列数据插补问题，避免过度利用非因果关系，提高数据分析的准确性。

    

    arXiv:2403.11960v1 公告类型：新 提要：时空时间序列是理解人类活动及其影响的基础，通常通过放置在不同位置的监测传感器收集。收集到的数据通常包含由于各种故障而导致的缺失值，这对数据分析有重要影响。为了填补缺失值，已经提出了许多方法。在恢复特定数据点时，大多数现有方法倾向于考虑与该点相关的所有信息，无论它们是否具有因果关系。在数据收集过程中，包括一些未知混杂因素是不可避免的，例如时间序列中的背景噪声和构建的传感器网络中的非因果快捷边。这些混杂因素可能在输入和输出之间开辟反向路径，换句话说，它们建立了输入和输出之间的非因果相关性。

    arXiv:2403.11960v1 Announce Type: new  Abstract: Spatiotemporal time series is the foundation of understanding human activities and their impacts, which is usually collected via monitoring sensors placed at different locations. The collected data usually contains missing values due to various failures, which have significant impact on data analysis. To impute the missing values, a lot of methods have been introduced. When recovering a specific data point, most existing methods tend to take into consideration all the information relevant to that point regardless of whether they have a cause-and-effect relationship. During data collection, it is inevitable that some unknown confounders are included, e.g., background noise in time series and non-causal shortcut edges in the constructed sensor network. These confounders could open backdoor paths between the input and output, in other words, they establish non-causal correlations between the input and output. Over-exploiting these non-causa
    
[^3]: 间接参数化具体自编码器

    Indirectly Parameterized Concrete Autoencoders

    [https://arxiv.org/abs/2403.00563](https://arxiv.org/abs/2403.00563)

    本文提出了间接参数化CAEs（IP-CAEs）来解决具体自编码器（CAEs）在稳定联合优化方面的问题，IP-CAEs在多个数据集上表现出显著且一致的改进。

    

    特征选择在数据高维或获取完整特征集成本高昂的情况下至关重要。最近基于神经网络的嵌入式特征选择的发展在广泛应用中表现出有希望的结果。具体自编码器（CAEs）被认为是嵌入式特征选择中的最先进技术，但可能难以实现稳定的联合优化，从而影响其训练时间和泛化能力。本文发现这种不稳定性与CAE学习重复选择有关。为了解决这个问题，我们提出了一种简单有效的改进：间接参数化CAEs（IP-CAEs）。IP-CAEs学习一个嵌入和从它到Gumbel-Softmax分布参数的映射。尽管实现简单，IP-CAE在多个数据集上的重构和分类任务中均表现出显著且一致的改进，无论是在泛化还是训练时间上。

    arXiv:2403.00563v1 Announce Type: new  Abstract: Feature selection is a crucial task in settings where data is high-dimensional or acquiring the full set of features is costly. Recent developments in neural network-based embedded feature selection show promising results across a wide range of applications. Concrete Autoencoders (CAEs), considered state-of-the-art in embedded feature selection, may struggle to achieve stable joint optimization, hurting their training time and generalization. In this work, we identify that this instability is correlated with the CAE learning duplicate selections. To remedy this, we propose a simple and effective improvement: Indirectly Parameterized CAEs (IP-CAEs). IP-CAEs learn an embedding and a mapping from it to the Gumbel-Softmax distributions' parameters. Despite being simple to implement, IP-CAE exhibits significant and consistent improvements over CAE in both generalization and training time across several datasets for reconstruction and classifi
    
[^4]: 多态雷达对空中飞行器雷达截面识别：一种贝叶斯融合方法

    Multistatic-Radar RCS-Signature Recognition of Aerial Vehicles: A Bayesian Fusion Approach

    [https://arxiv.org/abs/2402.17987](https://arxiv.org/abs/2402.17987)

    提出了一种完全贝叶斯雷达自动目标识别的框架，采用最优贝叶斯融合来有效地汇总多个雷达的分类概率向量，以改进无人机雷达截面识别效果。

    

    arXiv:2402.17987v1 公告类型：跨领域 摘要：无人机的雷达自动目标识别（RATR）涉及发射电磁波并对接收到的雷达回波执行目标类型识别，对国防和航空航天应用至关重要。先前的研究突出了多态雷达配置在RATR中优于单态雷达的优势。然而，多态雷达配置中的融合方法通常以概率方式次优地组合来自各个雷达的分类向量。为了解决这个问题，我们提出了一个完全贝叶斯RATR框架，采用最优贝叶斯融合（OBF）来聚合来自多个雷达的分类概率向量。OBF基于期望0-1损失，根据多个时间步骤的历史观测更新目标无人机类型的递归贝叶斯分类（RBC）后验分布。我们使用模拟的随机行走轨迹评估了这种方法，共涉及七种机动目标。

    arXiv:2402.17987v1 Announce Type: cross  Abstract: Radar Automated Target Recognition (RATR) for Unmanned Aerial Vehicles (UAVs) involves transmitting Electromagnetic Waves (EMWs) and performing target type recognition on the received radar echo, crucial for defense and aerospace applications. Previous studies highlighted the advantages of multistatic radar configurations over monostatic ones in RATR. However, fusion methods in multistatic radar configurations often suboptimally combine classification vectors from individual radars probabilistically. To address this, we propose a fully Bayesian RATR framework employing Optimal Bayesian Fusion (OBF) to aggregate classification probability vectors from multiple radars. OBF, based on expected 0-1 loss, updates a Recursive Bayesian Classification (RBC) posterior distribution for target UAV type, conditioned on historical observations across multiple time steps. We evaluate the approach using simulated random walk trajectories for seven dro
    
[^5]: 分布式估计和推断用于半参数二元响应模型

    Distributed Estimation and Inference for Semi-parametric Binary Response Models

    [https://arxiv.org/abs/2210.08393](https://arxiv.org/abs/2210.08393)

    通过一次性分治估计和多轮估计，本文提出了分布式环境下对半参数二元选择模型的新估计方法，实现了优化误差的超线性改进。

    

    现代技术的发展使得数据收集的规模前所未有，这给许多统计估计和推断问题带来了新挑战。本文研究了在分布式计算环境下对半参数二元选择模型的最大分数估计器，而无需预先指定噪声分布。传统的分治估计器在计算上昂贵，并受到机器数量的非正则约束的限制，这是由于目标函数的高度非光滑性质导致的。我们提出了(1)在平滑目标之后进行一次性分治估计以放宽约束，以及(2)通过迭代平滑完全去除约束的多轮估计。我们指定了一种自适应的核平滑器选择，通过顺序缩小带宽在多次迭代中实现了对优化误差的超线性改进。

    arXiv:2210.08393v3 Announce Type: replace-cross  Abstract: The development of modern technology has enabled data collection of unprecedented size, which poses new challenges to many statistical estimation and inference problems. This paper studies the maximum score estimator of a semi-parametric binary choice model under a distributed computing environment without pre-specifying the noise distribution. An intuitive divide-and-conquer estimator is computationally expensive and restricted by a non-regular constraint on the number of machines, due to the highly non-smooth nature of the objective function. We propose (1) a one-shot divide-and-conquer estimator after smoothing the objective to relax the constraint, and (2) a multi-round estimator to completely remove the constraint via iterative smoothing. We specify an adaptive choice of kernel smoother with a sequentially shrinking bandwidth to achieve the superlinear improvement of the optimization error over the multiple iterations. The
    
[^6]: 离散分布的谱聚类

    Spectral Clustering for Discrete Distributions. (arXiv:2401.13913v1 [cs.LG])

    [http://arxiv.org/abs/2401.13913](http://arxiv.org/abs/2401.13913)

    本文提出了一种基于谱聚类和分布相似度度量的框架来解决离散分布聚类问题。通过使用线性最优传输构建相似度矩阵，我们在聚类准确性和计算效率方面取得了显著的改进。

    

    离散分布聚类（D2C）通常通过Wasserstein质心方法来解决。这些方法在一个共同的假设下工作，即聚类可以通过质心很好地表示，但在许多实际应用中可能不成立。在本文中，我们提出了一个简单而有效的基于谱聚类和分布相似度度量（例如最大均值差异和Wasserstein距离）的框架来解决D2C问题。为了提高可扩展性，我们提出使用线性最优传输在大型数据集上高效地构建相似度矩阵。我们提供了理论保证，保证了所提方法在聚类分布方面的成功。在合成数据和真实数据上的实验证明，我们的方法在聚类准确性和计算效率方面大大优于基准方法。

    Discrete distribution clustering (D2C) was often solved by Wasserstein barycenter methods. These methods are under a common assumption that clusters can be well represented by barycenters, which may not hold in many real applications. In this work, we propose a simple yet effective framework based on spectral clustering and distribution affinity measures (e.g., maximum mean discrepancy and Wasserstein distance) for D2C. To improve the scalability, we propose to use linear optimal transport to construct affinity matrices efficiently on large datasets. We provide theoretical guarantees for the success of the proposed methods in clustering distributions. Experiments on synthetic and real data show that our methods outperform the baselines largely in terms of both clustering accuracy and computational efficiency.
    
[^7]: 针对预测不确定性的模型无关变量重要性：一种基于熵的方法

    Model-agnostic variable importance for predictive uncertainty: an entropy-based approach. (arXiv:2310.12842v1 [stat.ML])

    [http://arxiv.org/abs/2310.12842](http://arxiv.org/abs/2310.12842)

    本文提出了一种基于熵的方法，通过扩展现有的解释性方法，可以理解不确定性感知模型中的预测来源和置信度，并利用改编后的特征重要性、部分依赖图和个体条件期望图等方法来测量特征对预测分布的熵和基于真实标签的对数似然的影响。

    

    为了相信机器学习算法的预测结果，必须理解导致这些预测的因素。对于概率和不确定性感知的模型来说，不仅需要理解预测本身的原因，还要理解模型对这些预测的置信度。本文展示了如何将现有的解释性方法扩展到不确定性感知的模型，并如何利用这些扩展来理解模型预测分布中的不确定性来源。特别是通过改编排列特征重要性、部分依赖图和个体条件期望图，我们证明可以获得对模型行为的新见解，并且可以使用这些方法来衡量特征对预测分布的熵和基于该分布的真实标签的对数似然的影响。通过使用两个数据集的实验，我们验证了所提方法的有效性。

    In order to trust the predictions of a machine learning algorithm, it is necessary to understand the factors that contribute to those predictions. In the case of probabilistic and uncertainty-aware models, it is necessary to understand not only the reasons for the predictions themselves, but also the model's level of confidence in those predictions. In this paper, we show how existing methods in explainability can be extended to uncertainty-aware models and how such extensions can be used to understand the sources of uncertainty in a model's predictive distribution. In particular, by adapting permutation feature importance, partial dependence plots, and individual conditional expectation plots, we demonstrate that novel insights into model behaviour may be obtained and that these methods can be used to measure the impact of features on both the entropy of the predictive distribution and the log-likelihood of the ground truth labels under that distribution. With experiments using both s
    
[^8]: 自学优化器（STOP）：递归自我改进的代码生成

    Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation. (arXiv:2310.02304v1 [cs.CL])

    [http://arxiv.org/abs/2310.02304](http://arxiv.org/abs/2310.02304)

    本文提出了一种自学优化器（STOP），通过递归自我改进的代码生成，使用融合了语言模型的脚手架程序来改进自身，从而生成性能更好的程序。

    

    最近几年的人工智能系统（例如思维树和程序辅助语言模型）取得了一些重要进展，通过提供一个“脚手架”程序来解决问题，该程序构建了多次调用语言模型以生成更好的输出。脚手架程序通常使用Python等编程语言编写。在这项工作中，我们使用了一个融合了语言模型的脚手架程序来改进自身。我们从一个种子“改进器”开始，通过多次查询语言模型并返回最佳解决方案，根据给定的效用函数来改进输入程序。然后，我们运行这个种子改进器来改进自身。在一系列细分任务中，得到的改进改进器生成的程序在性能上明显优于种子改进器。随后，我们对语言模型提出的各种自我改进策略进行了分析，包括波束搜索、遗传算法和模拟退火。由于语言模型本身没有改变，这并不是一种增长领域。

    Several recent advances in AI systems (e.g., Tree-of-Thoughts and Program-Aided Language Models) solve problems by providing a "scaffolding" program that structures multiple calls to language models to generate better outputs. A scaffolding program is written in a programming language such as Python. In this work, we use a language-model-infused scaffolding program to improve itself. We start with a seed "improver" that improves an input program according to a given utility function by querying a language model several times and returning the best solution. We then run this seed improver to improve itself. Across a small set of downstream tasks, the resulting improved improver generates programs with significantly better performance than its seed improver. Afterward, we analyze the variety of self-improvement strategies proposed by the language model, including beam search, genetic algorithms, and simulated annealing. Since the language models themselves are not altered, this is not fu
    
[^9]: 基于弱监督的高斯过程主动学习

    Active Learning with Weak Supervision for Gaussian Processes. (arXiv:2204.08335v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2204.08335](http://arxiv.org/abs/2204.08335)

    本文提出了一种基于弱监督的主动学习算法，不仅选择要注释的观测结果，还选择要获得的注释精度，并在高斯过程中进行了实验验证。

    

    对于监督学习，进行数据注释需要耗费大量成本。当注释预算有限时，可以使用主动学习来选择和注释那些可能在模型性能上获得最大收益的观测结果。我们提出了一种主动学习算法，除了选择要注释的观测结果外，还选择要获得的注释精度。假定具有低精度的注释比具有高精度的注释更便宜，这使得模型能够在相同的注释预算下探索输入空间的更大部分。我们在先前针对高斯过程提出的BALD目标基础上构建了我们的获取函数，并在实验中展示了能够调整主动学习循环中的注释精度的优势。

    Annotating data for supervised learning can be costly. When the annotation budget is limited, active learning can be used to select and annotate those observations that are likely to give the most gain in model performance. We propose an active learning algorithm that, in addition to selecting which observation to annotate, selects the precision of the annotation that is acquired. Assuming that annotations with low precision are cheaper to obtain, this allows the model to explore a larger part of the input space, with the same annotation budget. We build our acquisition function on the previously proposed BALD objective for Gaussian Processes, and empirically demonstrate the gains of being able to adjust the annotation precision in the active learning loop.
    

