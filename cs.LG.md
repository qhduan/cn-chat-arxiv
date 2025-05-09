# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Comparing Hyper-optimized Machine Learning Models for Predicting Efficiency Degradation in Organic Solar Cells](https://arxiv.org/abs/2404.00173) | 该研究通过超优化的机器学习模型，成功预测有机太阳能电池效率退化，准确度高且具有实用价值。 |
| [^2] | [A Survey on Consumer IoT Traffic: Security and Privacy](https://arxiv.org/abs/2403.16149) | 本调查针对消费者物联网（CIoT）流量分析从安全和隐私的角度出发，总结了CIoT流量分析的新特征、最新进展和挑战，认为通过流量分析可以揭示CIoT领域中的安全和隐私问题。 |
| [^3] | [DyCE: Dynamic Configurable Exiting for Deep Learning Compression and Scaling](https://arxiv.org/abs/2403.01695) | 介绍了DyCE，一个动态可配置的提前退出框架，将设计考虑从彼此和基础模型解耦 |
| [^4] | [Exploring Learning Complexity for Downstream Data Pruning](https://arxiv.org/abs/2402.05356) | 本文提出了一种将学习复杂性作为分类和回归任务的评分函数，以解决在有限计算资源下微调过程中过度参数化模型的问题。 |
| [^5] | [Bounding Consideration Probabilities in Consider-Then-Choose Ranking Models.](http://arxiv.org/abs/2401.11016) | 在考虑-然后-选择的排名模型中，我们提出了一种方法来确定考虑概率的界限。尽管不能准确确定考虑概率，但在已知备选方案效用的情况下，我们可以推断出备选概率的相对大小的界限。 |
| [^6] | [Exact nonlinear state estimation.](http://arxiv.org/abs/2310.10976) | 本文引入了一种新的非线性估计理论，该理论试图弥合现有数据同化方法中的差距。具体而言，推导出了一个能够推广至任意非高斯分布的共轭变换滤波器 (CTF)，并提出了其集合近似版本 (ECTF)。 |
| [^7] | [Connecting NTK and NNGP: A Unified Theoretical Framework for Neural Network Learning Dynamics in the Kernel Regime.](http://arxiv.org/abs/2309.04522) | 本文提出了一个马尔可夫近似学习模型，统一了神经切向核（NTK）和神经网络高斯过程（NNGP）核，用于描述无限宽度深层网络的学习动力学。 |
| [^8] | [Multi-Objective Optimization Using the R2 Utility.](http://arxiv.org/abs/2305.11774) | 本文提出将多目标优化问题转化为一组单目标问题进行解决，并介绍了R2效用函数作为适当的目标函数。该效用函数单调且次模，可以使用贪心优化算法计算全局最优解。 |

# 详细

[^1]: 比较超优化的机器学习模型以预测有机太阳能电池效率退化

    Comparing Hyper-optimized Machine Learning Models for Predicting Efficiency Degradation in Organic Solar Cells

    [https://arxiv.org/abs/2404.00173](https://arxiv.org/abs/2404.00173)

    该研究通过超优化的机器学习模型，成功预测有机太阳能电池效率退化，准确度高且具有实用价值。

    

    本文提出了一组最优化的机器学习（ML）模型，来表示多层结构ITO/PEDOT:PSS/P3HT:PCBM/Al聚合物有机太阳能电池（OSCs）的功率转换效率（PCE）所遭受的时间退化。为此，我们生成了一个包含996条数据的数据库，其中包括关于制造过程和环境条件的7个变量，超过180天。然后，我们依靠一个软件框架，汇集了一系列自动化ML协议，通过简单的命令行界面顺序地针对我们的数据库执行，从而轻松地通过详尽的基准测试来超优化和随机化ML模型的种子，以获得最佳模型。所达到的准确度达到了广泛超过0.90的系数确定值（R2），而均方根误差（RMSE）、平方误差（SSE）和平均绝对误

    arXiv:2404.00173v1 Announce Type: new  Abstract: This work presents a set of optimal machine learning (ML) models to represent the temporal degradation suffered by the power conversion efficiency (PCE) of polymeric organic solar cells (OSCs) with a multilayer structure ITO/PEDOT:PSS/P3HT:PCBM/Al. To that aim, we generated a database with 996 entries, which includes up to 7 variables regarding both the manufacturing process and environmental conditions for more than 180 days. Then, we relied on a software framework that brings together a conglomeration of automated ML protocols that execute sequentially against our database by simply command-line interface. This easily permits hyper-optimizing and randomizing seeds of the ML models through exhaustive benchmarking so that optimal models are obtained. The accuracy achieved reaches values of the coefficient determination (R2) widely exceeding 0.90, whereas the root mean squared error (RMSE), sum of squared error (SSE), and mean absolute er
    
[^2]: 消费者物联网流量的调查：安全与隐私

    A Survey on Consumer IoT Traffic: Security and Privacy

    [https://arxiv.org/abs/2403.16149](https://arxiv.org/abs/2403.16149)

    本调查针对消费者物联网（CIoT）流量分析从安全和隐私的角度出发，总结了CIoT流量分析的新特征、最新进展和挑战，认为通过流量分析可以揭示CIoT领域中的安全和隐私问题。

    

    在过去几年里，消费者物联网（CIoT）已经进入了公众生活。尽管CIoT提高了人们日常生活的便利性，但也带来了新的安全和隐私问题。我们尝试通过流量分析这一安全领域中的流行方法，找出研究人员可以从流量分析中了解CIoT安全和隐私方面的内容。本调查从安全和隐私角度探讨了CIoT流量分析中的新特征、CIoT流量分析的最新进展以及尚未解决的挑战。我们从2018年1月至2023年12月收集了310篇与CIoT流量分析有关的安全和隐私角度的论文，总结了识别了CIoT新特征的CIoT流量分析过程。然后，我们根据五个应用目标详细介绍了现有的研究工作：设备指纹识别、用户活动推断、恶意行为检测、隐私泄露以及通信模式识别。

    arXiv:2403.16149v1 Announce Type: cross  Abstract: For the past few years, the Consumer Internet of Things (CIoT) has entered public lives. While CIoT has improved the convenience of people's daily lives, it has also brought new security and privacy concerns. In this survey, we try to figure out what researchers can learn about the security and privacy of CIoT by traffic analysis, a popular method in the security community. From the security and privacy perspective, this survey seeks out the new characteristics in CIoT traffic analysis, the state-of-the-art progress in CIoT traffic analysis, and the challenges yet to be solved. We collected 310 papers from January 2018 to December 2023 related to CIoT traffic analysis from the security and privacy perspective and summarized the process of CIoT traffic analysis in which the new characteristics of CIoT are identified. Then, we detail existing works based on five application goals: device fingerprinting, user activity inference, malicious
    
[^3]: DyCE：用于深度学习压缩和扩展的动态可配置退出

    DyCE: Dynamic Configurable Exiting for Deep Learning Compression and Scaling

    [https://arxiv.org/abs/2403.01695](https://arxiv.org/abs/2403.01695)

    介绍了DyCE，一个动态可配置的提前退出框架，将设计考虑从彼此和基础模型解耦

    

    现代深度学习（DL）模型需要在资源受限环境中有效部署时，使用缩放和压缩技术。大多数现有技术，如修剪和量化，通常是静态的。另一方面，动态压缩方法（如提前退出）通过识别输入样本的困难程度并根据需要分配计算来降低复杂性。动态方法，尽管具有更高的灵活性和与静态方法共存的潜力，但在实现上面临重大挑战，因为动态部分的任何变化都会影响后续过程。此外，大多数当前的动态压缩设计都是单片的，与基础模型紧密集成，从而使其难以适应新颖基础模型。本文介绍了DyCE，一种动态可配置的提前退出框架，从而使设计考虑相互解耦以及与基础模型

    arXiv:2403.01695v1 Announce Type: cross  Abstract: Modern deep learning (DL) models necessitate the employment of scaling and compression techniques for effective deployment in resource-constrained environments. Most existing techniques, such as pruning and quantization are generally static. On the other hand, dynamic compression methods, such as early exits, reduce complexity by recognizing the difficulty of input samples and allocating computation as needed. Dynamic methods, despite their superior flexibility and potential for co-existing with static methods, pose significant challenges in terms of implementation due to any changes in dynamic parts will influence subsequent processes. Moreover, most current dynamic compression designs are monolithic and tightly integrated with base models, thereby complicating the adaptation to novel base models. This paper introduces DyCE, an dynamic configurable early-exit framework that decouples design considerations from each other and from the 
    
[^4]: 探索用于下游数据修剪的学习复杂性

    Exploring Learning Complexity for Downstream Data Pruning

    [https://arxiv.org/abs/2402.05356](https://arxiv.org/abs/2402.05356)

    本文提出了一种将学习复杂性作为分类和回归任务的评分函数，以解决在有限计算资源下微调过程中过度参数化模型的问题。

    

    过度参数化的预训练模型对于有限的计算资源的微调构成了巨大的挑战。一个直观的解决方案是从微调数据集中修剪掉信息较少的样本。提出了一系列基于训练的评分函数来量化数据子集的信息性，但由于参数更新的繁重，修剪成本变得不可忽视。为了高效修剪，将几何方法的相似度评分函数从基于训练的方法适应为无需训练的方法是可行的。然而，我们凭经验证明这种适应扭曲了原始的修剪并导致下游任务表现不佳。在本文中，我们提出将学习复杂性（LC）作为分类和回归任务的评分函数。具体来说，学习复杂性被定义为具有不同容量的子网络的平均预测置信度，它包含了在一个收敛模型中的数据处理。

    The over-parameterized pre-trained models pose a great challenge to fine-tuning with limited computation resources. An intuitive solution is to prune the less informative samples from the fine-tuning dataset. A series of training-based scoring functions are proposed to quantify the informativeness of the data subset but the pruning cost becomes non-negligible due to the heavy parameter updating. For efficient pruning, it is viable to adapt the similarity scoring function of geometric-based methods from training-based to training-free. However, we empirically show that such adaption distorts the original pruning and results in inferior performance on the downstream tasks. In this paper, we propose to treat the learning complexity (LC) as the scoring function for classification and regression tasks. Specifically, the learning complexity is defined as the average predicted confidence of subnets with different capacities, which encapsulates data processing within a converged model. Then we
    
[^5]: 在考虑-然后-选择排名模型中确定考虑概率的界限

    Bounding Consideration Probabilities in Consider-Then-Choose Ranking Models. (arXiv:2401.11016v1 [cs.LG])

    [http://arxiv.org/abs/2401.11016](http://arxiv.org/abs/2401.11016)

    在考虑-然后-选择的排名模型中，我们提出了一种方法来确定考虑概率的界限。尽管不能准确确定考虑概率，但在已知备选方案效用的情况下，我们可以推断出备选概率的相对大小的界限。

    

    选择理论中一种常见的观点认为，个体在做出选择之前，会先进行两步的过程，首先选择一些备选方案进行考虑，然后从所得的考虑集合中进行选择。然而，在这种“考虑然后选择”的情景下推断未观察到的考虑集合（或者备选方案的考虑概率）面临着重大挑战，因为即使是对于具有强独立性假设的简单考虑模型，在已知备选方案效用的情况下也无法确定身份。我们考虑将考虑-然后-选择模型自然地扩展到top-k排名的情景，假设排名是根据Plackett-Luce模型在采样了考虑集合后构建的。尽管在这种情景下备选方案的考虑概率仍旧不能确定，但我们证明了在获得备选方案效用的知识的情况下，我们可以推断出备选概率相对大小的界限。此外，通过对期望考虑集合大小的条件进行推导，我们得到绝对界限。

    A common theory of choice posits that individuals make choices in a two-step process, first selecting some subset of the alternatives to consider before making a selection from the resulting consideration set. However, inferring unobserved consideration sets (or item consideration probabilities) in this "consider then choose" setting poses significant challenges, because even simple models of consideration with strong independence assumptions are not identifiable, even if item utilities are known. We consider a natural extension of consider-then-choose models to a top-$k$ ranking setting, where we assume rankings are constructed according to a Plackett-Luce model after sampling a consideration set. While item consideration probabilities remain non-identified in this setting, we prove that knowledge of item utilities allows us to infer bounds on the relative sizes of consideration probabilities. Additionally, given a condition on the expected consideration set size, we derive absolute u
    
[^6]: 精确非线性状态估计

    Exact nonlinear state estimation. (arXiv:2310.10976v1 [stat.ME])

    [http://arxiv.org/abs/2310.10976](http://arxiv.org/abs/2310.10976)

    本文引入了一种新的非线性估计理论，该理论试图弥合现有数据同化方法中的差距。具体而言，推导出了一个能够推广至任意非高斯分布的共轭变换滤波器 (CTF)，并提出了其集合近似版本 (ECTF)。

    

    地球科学中的大多数数据同化方法基于高斯假设。尽管这些假设方便了高效的算法，但它们会导致分析偏差和后续预测恶化。非参数、基于粒子的数据同化算法具有更高的准确性，但其在高维模型中的应用仍面临操作上的挑战。本文借鉴了生成人工智能领域的最新进展，提出了一种试图弥合数据同化方法中现有差距的新的非线性估计理论。具体而言，推导出了一个共轭变换滤波器 (CTF)，并显示其能够推广至任意非高斯分布。新的滤波器具有几个优点，例如能够保留先前状态中的统计关系并收敛至高精度的观测值。同时还提出了新理论的一个集合近似 (ECTF)。

    The majority of data assimilation (DA) methods in the geosciences are based on Gaussian assumptions. While these assumptions facilitate efficient algorithms, they cause analysis biases and subsequent forecast degradations. Non-parametric, particle-based DA algorithms have superior accuracy, but their application to high-dimensional models still poses operational challenges. Drawing inspiration from recent advances in the field of generative artificial intelligence (AI), this article introduces a new nonlinear estimation theory which attempts to bridge the existing gap in DA methodology. Specifically, a Conjugate Transform Filter (CTF) is derived and shown to generalize the celebrated Kalman filter to arbitrarily non-Gaussian distributions. The new filter has several desirable properties, such as its ability to preserve statistical relationships in the prior state and convergence to highly accurate observations. An ensemble approximation of the new theory (ECTF) is also presented and va
    
[^7]: 连接NTK和NNGP：神经网络学习动力学在核区域的统一理论框架

    Connecting NTK and NNGP: A Unified Theoretical Framework for Neural Network Learning Dynamics in the Kernel Regime. (arXiv:2309.04522v1 [cs.LG])

    [http://arxiv.org/abs/2309.04522](http://arxiv.org/abs/2309.04522)

    本文提出了一个马尔可夫近似学习模型，统一了神经切向核（NTK）和神经网络高斯过程（NNGP）核，用于描述无限宽度深层网络的学习动力学。

    

    人工神经网络近年来在机器学习领域取得了革命性的进展，但其学习过程缺乏一个完整的理论框架。对于无限宽度网络，已经取得了重大进展。在这个范式中，使用了两种不同的理论框架来描述网络的输出：一种基于神经切向核（NTK）的框架，假设了线性化的梯度下降动力学；另一种是基于神经网络高斯过程（NNGP）核的贝叶斯框架。然而，这两种框架之间的关系一直不明确。本文通过一个马尔可夫近似学习模型，统一了这两种不同的理论，用于描述随机初始化的无限宽度深层网络的学习动力学。我们推导出了在学习过程中和学习后的网络输入-输出函数的精确分析表达式，并引入了一个新的时间相关的神经动态核（NDK），这个核可以同时产生NTK和NNGP。

    Artificial neural networks have revolutionized machine learning in recent years, but a complete theoretical framework for their learning process is still lacking. Substantial progress has been made for infinitely wide networks. In this regime, two disparate theoretical frameworks have been used, in which the network's output is described using kernels: one framework is based on the Neural Tangent Kernel (NTK) which assumes linearized gradient descent dynamics, while the Neural Network Gaussian Process (NNGP) kernel assumes a Bayesian framework. However, the relation between these two frameworks has remained elusive. This work unifies these two distinct theories using a Markov proximal learning model for learning dynamics in an ensemble of randomly initialized infinitely wide deep networks. We derive an exact analytical expression for the network input-output function during and after learning, and introduce a new time-dependent Neural Dynamical Kernel (NDK) from which both NTK and NNGP
    
[^8]: 使用R2效用的多目标优化

    Multi-Objective Optimization Using the R2 Utility. (arXiv:2305.11774v1 [math.OC])

    [http://arxiv.org/abs/2305.11774](http://arxiv.org/abs/2305.11774)

    本文提出将多目标优化问题转化为一组单目标问题进行解决，并介绍了R2效用函数作为适当的目标函数。该效用函数单调且次模，可以使用贪心优化算法计算全局最优解。

    

    多目标优化的目标是确定描述多目标之间最佳权衡的点集合。为了解决这个矢量值优化问题，从业者常常使用标量化函数将多目标问题转化为一组单目标问题。这组标量化问题可以使用传统的单目标优化技术来解决。在这项工作中，我们将这个约定形式化为一个通用的数学框架。我们展示了这种策略如何有效地将原始的多目标优化问题重新转化为定义在集合上的单目标优化问题。针对这个新问题的适当类别的目标函数是R2效用函数，它被定义为标量化优化问题的加权积分。我们证明了这个效用函数是单调的和次模的集合函数，可以通过贪心优化算法有效地计算出全局最优解。

    The goal of multi-objective optimization is to identify a collection of points which describe the best possible trade-offs between the multiple objectives. In order to solve this vector-valued optimization problem, practitioners often appeal to the use of scalarization functions in order to transform the multi-objective problem into a collection of single-objective problems. This set of scalarized problems can then be solved using traditional single-objective optimization techniques. In this work, we formalise this convention into a general mathematical framework. We show how this strategy effectively recasts the original multi-objective optimization problem into a single-objective optimization problem defined over sets. An appropriate class of objective functions for this new problem is the R2 utility function, which is defined as a weighted integral over the scalarized optimization problems. We show that this utility function is a monotone and submodular set function, which can be op
    

