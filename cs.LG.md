# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Trading off Consistency and Dimensionality of Convex Surrogates for the Mode](https://arxiv.org/abs/2402.10818) | 通过在低维替代空间中的凸多面体顶点上嵌入结果，并探究单纯形中的一致性区域，权衡了替代损失维度、问题实例数量。 |
| [^2] | [Deeper or Wider: A Perspective from Optimal Generalization Error with Sobolev Loss](https://arxiv.org/abs/2402.00152) | 本文比较了更深的神经网络和更宽的神经网络在Sobolev损失的最优泛化误差方面的表现，研究发现神经网络的架构受多种因素影响，参数数量更多倾向于选择更宽的网络，而样本点数量和损失函数规则性更高倾向于选择更深的网络。 |
| [^3] | [Hidden Minima in Two-Layer ReLU Networks](https://arxiv.org/abs/2312.16819) | 本文研究了两层ReLU网络中的隐藏极小值现象，并提出方法来研究这些隐藏极小值的独特解析性质。 |
| [^4] | [Manipulating Feature Visualizations with Gradient Slingshots.](http://arxiv.org/abs/2401.06122) | 本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。 |
| [^5] | [Quadratic Time-Frequency Analysis of Vibration Signals for Diagnosing Bearing Faults.](http://arxiv.org/abs/2401.01172) | 本文提出了一种融合时间频率分析和深度学习技术的方法，用于在实际条件下诊断带有时间变化速度和不同噪声水平的轴承故障。这种方法有效地解析与不同轴承故障相关的独特动态模式。 |
| [^6] | [Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand.](http://arxiv.org/abs/2310.20350) | 本文提出了一种结合形状完成和抓取预测的方法，实现了快速灵活的多指抓取。通过使用基于深度图像的形状完成模块和基于预测的抓取预测器，实现了在具有有限或无先验知识的情况下，对物体进行抓取的任务。 |
| [^7] | [Optimal Conditional Inference in Adaptive Experiments.](http://arxiv.org/abs/2309.12162) | 我们研究了在自适应实验中进行条件推断的问题，证明了在没有进一步限制的情况下，仅使用最后一批结果进行推断是最优的；当实验的自适应方面是位置不变的时，我们还发现了额外的信息；在停止时间、分配概率和目标参数仅依赖于数据的多面体事件集合的情况下，我们推导出了计算可行且最优的条件推断程序。 |
| [^8] | [Shape Completion with Prediction of Uncertain Regions.](http://arxiv.org/abs/2308.00377) | 该论文提出了两种方法来处理在给定模糊物体视图时可能存在的物体部分的不确定区域预测问题。研究表明这些方法可以作为任何预测空间占用的方法的直接扩展，通过后处理占用评分或直接预测不确定性指标来实现。这些方法与已知的概率形状完成方法进行了比较，并使用自动生成的深度图像数据集进行了验证。 |
| [^9] | [Generative Language Models on Nucleotide Sequences of Human Genes.](http://arxiv.org/abs/2307.10634) | 本研究开发了一种生成语言模型，用于处理人类基因的核苷酸序列，填补了DNA序列生成模型研究的空白。 |
| [^10] | [Topology-Aware Loss for Aorta and Great Vessel Segmentation in Computed Tomography Images.](http://arxiv.org/abs/2307.03137) | 本文介绍了一种新的拓扑感知损失函数，通过持久同调来惩罚计算机断层扫描图像中主动脉和大血管分割结果与真实值之间的拓扑差异。这种方法能够改善分割任务的性能，尤其是针对具有固有几何特征的对象。 |
| [^11] | [Unsupervised Cross-Domain Soft Sensor Modelling via A Deep Bayesian Particle Flow Framework.](http://arxiv.org/abs/2306.04919) | 该论文提出了一种基于深度粒子流贝叶斯框架的跨领域软测量无监督建模方法，能够解决标签缺失、领域适应性和数据时间一致性等问题，提高软测量建模的准确性。 |
| [^12] | [On the Global Convergence of Risk-Averse Policy Gradient Methods with Expected Conditional Risk Measures.](http://arxiv.org/abs/2301.10932) | 本论文研究了具有期望条件风险度量的风险厌恶策略梯度方法，提出了策略梯度更新，证明了其在约束和无约束情况下的全局收敛性和迭代复杂度，并测试了REINFORCE和actor-critic算法的风险厌恶变体来展示方法的实用价值和风险控制的重要性。 |

# 详细

[^1]: 在模型的凸替代品的一致性和维度之间进行权衡

    Trading off Consistency and Dimensionality of Convex Surrogates for the Mode

    [https://arxiv.org/abs/2402.10818](https://arxiv.org/abs/2402.10818)

    通过在低维替代空间中的凸多面体顶点上嵌入结果，并探究单纯形中的一致性区域，权衡了替代损失维度、问题实例数量。

    

    在多类分类中，必须将结果嵌入到至少有$n-1$维的实数空间中，以设计一种一致的替代损失函数，这会导致"正确"的分类，而不受数据分布的影响。在信息检索和结构化预测任务等需要大量n时，优化n-1维替代常常是棘手的。我们研究了在多类分类中如何权衡替代损失维度、问题实例数量以及在单纯形上约束一致性区域的方法。我们跟随过去的研究，探讨了一种直观的嵌入过程，将结果映射到低维替代空间中的凸多面体的顶点上。我们展示了在每个点质量分布周围存在单纯形的全维子集，其中一致性成立，但是，少于n-1维度的情况下，存在一些分布，对于这些分布，一种现象性是

    arXiv:2402.10818v1 Announce Type: new  Abstract: In multiclass classification over $n$ outcomes, the outcomes must be embedded into the reals with dimension at least $n-1$ in order to design a consistent surrogate loss that leads to the "correct" classification, regardless of the data distribution. For large $n$, such as in information retrieval and structured prediction tasks, optimizing a surrogate in $n-1$ dimensions is often intractable. We investigate ways to trade off surrogate loss dimension, the number of problem instances, and restricting the region of consistency in the simplex for multiclass classification. Following past work, we examine an intuitive embedding procedure that maps outcomes into the vertices of convex polytopes in a low-dimensional surrogate space. We show that full-dimensional subsets of the simplex exist around each point mass distribution for which consistency holds, but also, with less than $n-1$ dimensions, there exist distributions for which a phenomeno
    
[^2]: 更深还是更宽: 从Sobolev损失的最优泛化误差角度看

    Deeper or Wider: A Perspective from Optimal Generalization Error with Sobolev Loss

    [https://arxiv.org/abs/2402.00152](https://arxiv.org/abs/2402.00152)

    本文比较了更深的神经网络和更宽的神经网络在Sobolev损失的最优泛化误差方面的表现，研究发现神经网络的架构受多种因素影响，参数数量更多倾向于选择更宽的网络，而样本点数量和损失函数规则性更高倾向于选择更深的网络。

    

    构建神经网络的架构是机器学习界一个具有挑战性的追求，到底是更深还是更宽一直是一个持续的问题。本文探索了更深的神经网络（DeNNs）和具有有限隐藏层的更宽的神经网络（WeNNs）在Sobolev损失的最优泛化误差方面的比较。通过分析研究发现，神经网络的架构可以受到多种因素的显著影响，包括样本点的数量，神经网络内的参数以及损失函数的规则性。具体而言，更多的参数倾向于选择WeNNs，而更多的样本点和更高的损失函数规则性倾向于选择DeNNs。最后，我们将这个理论应用于使用深度Ritz和物理感知神经网络（PINN）方法解决偏微分方程的问题。

    Constructing the architecture of a neural network is a challenging pursuit for the machine learning community, and the dilemma of whether to go deeper or wider remains a persistent question. This paper explores a comparison between deeper neural networks (DeNNs) with a flexible number of layers and wider neural networks (WeNNs) with limited hidden layers, focusing on their optimal generalization error in Sobolev losses. Analytical investigations reveal that the architecture of a neural network can be significantly influenced by various factors, including the number of sample points, parameters within the neural networks, and the regularity of the loss function. Specifically, a higher number of parameters tends to favor WeNNs, while an increased number of sample points and greater regularity in the loss function lean towards the adoption of DeNNs. We ultimately apply this theory to address partial differential equations using deep Ritz and physics-informed neural network (PINN) methods,
    
[^3]: 两层ReLU网络中的隐藏极小值

    Hidden Minima in Two-Layer ReLU Networks

    [https://arxiv.org/abs/2312.16819](https://arxiv.org/abs/2312.16819)

    本文研究了两层ReLU网络中的隐藏极小值现象，并提出方法来研究这些隐藏极小值的独特解析性质。

    

    本文考虑拟合具有$d$个输入、$k$个神经元以及由目标网络生成的标签的两层ReLU网络所涉及的优化问题。最近发现了两种无穷族的虚假极小值，每个$d$对应一个极小值。属于第一类的极小值的损失在$d$增加时收敛于零。在第二类中，损失保持远离于零。那么，如何避免属于后一类的极小值呢？幸运的是，这样的极小值从不会被标准优化方法检测到。受到此现象性质的问题的启发，我们开发了研究隐藏极小值独特解析性质的方法。根据现有的分析，两种类型的Hessian谱在$O(d^{-1/2})$项模意义下一致 -- 不太乐观。因此，我们的研究通过研究损失被最小化或最大化的曲线进行，通常称为切线。

    arXiv:2312.16819v2 Announce Type: replace  Abstract: The optimization problem associated to fitting two-layer ReLU networks having $d$~inputs, $k$~neurons, and labels generated by a target network, is considered. Two types of infinite families of spurious minima, giving one minimum per $d$, were recently found. The loss at minima belonging to the first type converges to zero as $d$ increases. In the second type, the loss remains bounded away from zero. That being so, how may one avoid minima belonging to the latter type? Fortunately, such minima are never detected by standard optimization methods. Motivated by questions concerning the nature of this phenomenon, we develop methods to study distinctive analytic properties of hidden minima.   By existing analyses, the Hessian spectrum of both types agree modulo $O(d^{-1/2})$-terms -- not promising. Thus, rather, our investigation proceeds by studying curves along which the loss is minimized or maximized, generally referred to as tangency 
    
[^4]: 用梯度弹射操纵特征可视化

    Manipulating Feature Visualizations with Gradient Slingshots. (arXiv:2401.06122v1 [cs.LG])

    [http://arxiv.org/abs/2401.06122](http://arxiv.org/abs/2401.06122)

    本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。

    

    深度神经网络(DNNs)能够学习复杂而多样化的表示，然而，学习到的概念的语义性质仍然未知。解释DNNs学习到的概念的常用方法是激活最大化(AM)，它生成一个合成的输入信号，最大化激活网络中的特定神经元。在本文中，我们研究了这种方法对于对抗模型操作的脆弱性，并引入了一种新的方法来操纵特征可视化，而不改变模型结构或对模型的决策过程产生显著影响。我们评估了我们的方法对几个神经网络模型的效果，并展示了它隐藏特定神经元功能的能力，在模型审核过程中使用选择的目标解释屏蔽了原始解释。作为一种补救措施，我们提出了一种防止这种操纵的防护措施，并提供了定量证据，证明了它的有效性。

    Deep Neural Networks (DNNs) are capable of learning complex and versatile representations, however, the semantic nature of the learned concepts remains unknown. A common method used to explain the concepts learned by DNNs is Activation Maximization (AM), which generates a synthetic input signal that maximally activates a particular neuron in the network. In this paper, we investigate the vulnerability of this approach to adversarial model manipulations and introduce a novel method for manipulating feature visualization without altering the model architecture or significantly impacting the model's decision-making process. We evaluate the effectiveness of our method on several neural network models and demonstrate its capabilities to hide the functionality of specific neurons by masking the original explanations of neurons with chosen target explanations during model auditing. As a remedy, we propose a protective measure against such manipulations and provide quantitative evidence which 
    
[^5]: 振动信号的二次时间频率分析用于诊断轴承故障

    Quadratic Time-Frequency Analysis of Vibration Signals for Diagnosing Bearing Faults. (arXiv:2401.01172v1 [cs.LG])

    [http://arxiv.org/abs/2401.01172](http://arxiv.org/abs/2401.01172)

    本文提出了一种融合时间频率分析和深度学习技术的方法，用于在实际条件下诊断带有时间变化速度和不同噪声水平的轴承故障。这种方法有效地解析与不同轴承故障相关的独特动态模式。

    

    轴承故障的诊断对于降低维修成本和设备停机至关重要。轴承故障是机器振动的主要原因，分析其信号形态可以揭示其健康状况。然而，现有的方法主要针对控制环境进行优化，忽略了实际条件下的时间变化的转速和振动的非平稳性。本文提出了一种时间频率分析和深度学习技术的融合方法，用于在时间变化速度和不同噪声水平下诊断轴承故障。首先，我们制定了轴承故障引起的振动，并讨论了它们的非平稳性与轴承固有和操作参数之间的联系。我们还阐述了二次时间频率分布，并验证了它们解析与不同轴承故障相关的独特动态模式的有效性。基于此，我们设计了一个时间频率卷积神经网络。

    Diagnosis of bearing faults is paramount to reducing maintenance costs and operational breakdowns. Bearing faults are primary contributors to machine vibrations, and analyzing their signal morphology offers insights into their health status. Unfortunately, existing approaches are optimized for controlled environments, neglecting realistic conditions such as time-varying rotational speeds and the vibration's non-stationary nature. This paper presents a fusion of time-frequency analysis and deep learning techniques to diagnose bearing faults under time-varying speeds and varying noise levels. First, we formulate the bearing fault-induced vibrations and discuss the link between their non-stationarity and the bearing's inherent and operational parameters. We also elucidate quadratic time-frequency distributions and validate their effectiveness in resolving distinctive dynamic patterns associated with different bearing faults. Based on this, we design a time-frequency convolutional neural n
    
[^6]: 将形状完成和抓取预测结合，实现快速灵活的多指抓取

    Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand. (arXiv:2310.20350v1 [cs.RO])

    [http://arxiv.org/abs/2310.20350](http://arxiv.org/abs/2310.20350)

    本文提出了一种结合形状完成和抓取预测的方法，实现了快速灵活的多指抓取。通过使用基于深度图像的形状完成模块和基于预测的抓取预测器，实现了在具有有限或无先验知识的情况下，对物体进行抓取的任务。

    

    在辅助机器人中，对于具有有限或无先验知识的物体进行抓取是一项非常重要的技能。然而，在这种普适情况下，尤其是在观测能力有限和利用多指手进行灵活抓取时，仍然存在一个开放的问题。我们提出了一种新颖、快速和高保真度的深度学习流程，由基于单个深度图像的形状完成模块和基于预测的物体形状的抓取预测器组成。形状完成网络基于VQDIF，在任意查询点上预测空间占用值。作为抓取预测器，我们使用了两阶段架构，首先使用自回归模型生成手姿势，然后回归每个姿势的手指关节配置。关键因素是足够的数据真实性和增强，以及在训练过程中对困难情况的特殊关注。在物理机器人平台上进行的实验表明，成功地实现了抓取。

    Grasping objects with limited or no prior knowledge about them is a highly relevant skill in assistive robotics. Still, in this general setting, it has remained an open problem, especially when it comes to only partial observability and versatile grasping with multi-fingered hands. We present a novel, fast, and high fidelity deep learning pipeline consisting of a shape completion module that is based on a single depth image, and followed by a grasp predictor that is based on the predicted object shape. The shape completion network is based on VQDIF and predicts spatial occupancy values at arbitrary query points. As grasp predictor, we use our two-stage architecture that first generates hand poses using an autoregressive model and then regresses finger joint configurations per pose. Critical factors turn out to be sufficient data realism and augmentation, as well as special attention to difficult cases during training. Experiments on a physical robot platform demonstrate successful gras
    
[^7]: 自适应实验中的最优条件推断

    Optimal Conditional Inference in Adaptive Experiments. (arXiv:2309.12162v1 [stat.ME])

    [http://arxiv.org/abs/2309.12162](http://arxiv.org/abs/2309.12162)

    我们研究了在自适应实验中进行条件推断的问题，证明了在没有进一步限制的情况下，仅使用最后一批结果进行推断是最优的；当实验的自适应方面是位置不变的时，我们还发现了额外的信息；在停止时间、分配概率和目标参数仅依赖于数据的多面体事件集合的情况下，我们推导出了计算可行且最优的条件推断程序。

    

    我们研究了批量赌徒实验，并考虑了在实现停止时间、分配概率和目标参数的条件下进行推断的问题，其中所有这些可能都是根据实验的最后一批信息进行自适应选择的。在没有对实验进行进一步限制的情况下，我们证明仅使用最后一批结果进行推断是最优的。当实验的自适应方面被认为是位置不变的，即当我们将所有批次-臂的平均值都向一个常数移动时，我们证明数据中还存在额外的信息，可以通过一个额外的批次-臂均值的线性函数来捕捉。在更严格的情况下，停止时间、分配概率和目标参数被认为仅依赖于数据通过一个多面体事件的集合，我们推导出了计算可行且最优的条件推断程序。

    We study batched bandit experiments and consider the problem of inference conditional on the realized stopping time, assignment probabilities, and target parameter, where all of these may be chosen adaptively using information up to the last batch of the experiment. Absent further restrictions on the experiment, we show that inference using only the results of the last batch is optimal. When the adaptive aspects of the experiment are known to be location-invariant, in the sense that they are unchanged when we shift all batch-arm means by a constant, we show that there is additional information in the data, captured by one additional linear function of the batch-arm means. In the more restrictive case where the stopping time, assignment probabilities, and target parameter are known to depend on the data only through a collection of polyhedral events, we derive computationally tractable and optimal conditional inference procedures.
    
[^8]: 带有不确定区域预测的形状完成

    Shape Completion with Prediction of Uncertain Regions. (arXiv:2308.00377v1 [cs.CV])

    [http://arxiv.org/abs/2308.00377](http://arxiv.org/abs/2308.00377)

    该论文提出了两种方法来处理在给定模糊物体视图时可能存在的物体部分的不确定区域预测问题。研究表明这些方法可以作为任何预测空间占用的方法的直接扩展，通过后处理占用评分或直接预测不确定性指标来实现。这些方法与已知的概率形状完成方法进行了比较，并使用自动生成的深度图像数据集进行了验证。

    

    形状完成，即从部分观测预测物体的完整几何形状，对于几个下游任务非常重要，尤其是机器人操作。当基于物体形状重建进行规划或实际抓取的预测时，指示严重几何不确定性是必不可少的。特别是在给定模糊的物体视图时，在整个物体部分存在 irreducible uncertainty 的扩展区域。为了处理这种重要情况，我们提出了两种新方法来预测这些不确定区域，这两种方法都可以作为预测局部空间占用的任何方法的直接扩展，一种是通过后处理占用评分，另一种是通过直接预测不确定性指标。我们将这些方法与两种已知的概率形状完成方法进行了比较。此外，我们还生成了一个基于ShapeNet的数据集，其中包含了真实渲染的物体视图深度图像及其带有地面真值标注。

    Shape completion, i.e., predicting the complete geometry of an object from a partial observation, is highly relevant for several downstream tasks, most notably robotic manipulation. When basing planning or prediction of real grasps on object shape reconstruction, an indication of severe geometric uncertainty is indispensable. In particular, there can be an irreducible uncertainty in extended regions about the presence of entire object parts when given ambiguous object views. To treat this important case, we propose two novel methods for predicting such uncertain regions as straightforward extensions of any method for predicting local spatial occupancy, one through postprocessing occupancy scores, the other through direct prediction of an uncertainty indicator. We compare these methods together with two known approaches to probabilistic shape completion. Moreover, we generate a dataset, derived from ShapeNet, of realistically rendered depth images of object views with ground-truth annot
    
[^9]: 人类基因核苷酸序列的生成语言模型

    Generative Language Models on Nucleotide Sequences of Human Genes. (arXiv:2307.10634v1 [q-bio.GN])

    [http://arxiv.org/abs/2307.10634](http://arxiv.org/abs/2307.10634)

    本研究开发了一种生成语言模型，用于处理人类基因的核苷酸序列，填补了DNA序列生成模型研究的空白。

    

    自然语言处理领域的语言模型，特别是基于Transformer的模型，取得了巨大的成功。然而，在DNA相关的生物信息学领域，生成模型的研究相对较少。因此，本研究旨在开发一种类似于GPT-3的自回归生成语言模型，用于处理人类基因的核苷酸序列。考虑到处理整个DNA序列需要大量计算资源，我们决定在更小的尺度上进行研究，重点关注人类基因的核苷酸序列，而不是整个DNA。这个决策并不改变问题的结构，因为DNA和基因都可以看作由四种不同的核苷酸组成的一维序列。

    Language models, primarily transformer-based ones, obtained colossal success in NLP. To be more precise, studies like BERT in NLU and works such as GPT-3 for NLG are very crucial. DNA sequences are very close to natural language in terms of structure, so if the DNA-related bioinformatics domain is concerned, discriminative models, like DNABert, exist. Yet, the generative side of the coin is mainly unexplored to the best of our knowledge. Consequently, we focused on developing an autoregressive generative language model like GPT-3 for DNA sequences. Because working with whole DNA sequences is challenging without substantial computational resources, we decided to carry out our study on a smaller scale, focusing on nucleotide sequences of human genes, unique parts in DNA with specific functionalities, instead of the whole DNA. This decision did not change the problem structure a lot due to the fact that both DNA and genes can be seen as 1D sequences consisting of four different nucleotide
    
[^10]: 计算机断层扫描图像中主动脉和大血管分割的拓扑感知损失

    Topology-Aware Loss for Aorta and Great Vessel Segmentation in Computed Tomography Images. (arXiv:2307.03137v1 [eess.IV])

    [http://arxiv.org/abs/2307.03137](http://arxiv.org/abs/2307.03137)

    本文介绍了一种新的拓扑感知损失函数，通过持久同调来惩罚计算机断层扫描图像中主动脉和大血管分割结果与真实值之间的拓扑差异。这种方法能够改善分割任务的性能，尤其是针对具有固有几何特征的对象。

    

    当使用标准损失函数训练分割网络时，网络并没有明确被要求学习图像的全局不变性，如对象的形状和多个对象之间的几何关系。然而，将这些不变性纳入网络训练中可能有助于改善各种分割任务的性能，尤其是当它们是需要分割的对象的固有特性时。本文以计算机断层扫描（CT）图像中主动脉和大血管的分割为例，这些血管由于人体解剖学，通常在身体中以特定的几何形状出现，并在2D CT图像上主要呈现为圆形对象。本文通过引入一种新的拓扑感知损失函数，通过持久同调惩罚地面真实值和预测之间的拓扑差异来解决这个问题。这与先前提出的分割网络设计不同，先前的设计是将阈值滤波应用于预测图像的似然函数。

    Segmentation networks are not explicitly imposed to learn global invariants of an image, such as the shape of an object and the geometry between multiple objects, when they are trained with a standard loss function. On the other hand, incorporating such invariants into network training may help improve performance for various segmentation tasks when they are the intrinsic characteristics of the objects to be segmented. One example is segmentation of aorta and great vessels in computed tomography (CT) images where vessels are found in a particular geometry in the body due to the human anatomy and they mostly seem as round objects on a 2D CT image. This paper addresses this issue by introducing a new topology-aware loss function that penalizes topology dissimilarities between the ground truth and prediction through persistent homology. Different from the previously suggested segmentation network designs, which apply the threshold filtration on a likelihood function of the prediction map 
    
[^11]: 一种基于深度贝叶斯粒子流框架的跨领域软测量无监督建模方法

    Unsupervised Cross-Domain Soft Sensor Modelling via A Deep Bayesian Particle Flow Framework. (arXiv:2306.04919v1 [cs.LG])

    [http://arxiv.org/abs/2306.04919](http://arxiv.org/abs/2306.04919)

    该论文提出了一种基于深度粒子流贝叶斯框架的跨领域软测量无监督建模方法，能够解决标签缺失、领域适应性和数据时间一致性等问题，提高软测量建模的准确性。

    

    数据驱动的软测量对于通过可靠的状态推断实现精确感知至关重要。然而，由于存在标签缺失、领域适应性和数据时间一致性等问题，开发具有代表性的软测量模型具有挑战性。为了解决这些问题，我们提出了一种基于深度粒子流贝叶斯 (DPFB) 框架，用于在无目标状态标签情况下进行跨领域软测量建模。具体来说，首先制定了一个顺序贝叶斯目标，以执行潜在的跨领域软感知问题的最大似然估计。在框架核心，我们结合物理学启发的粒子流，通过优化顺序贝叶斯目标来执行模型提取的潜在和隐藏特征的精确贝叶斯更新。由此，这些贡献使得该框架能够学习一个有机的近似后验特征表示，能够表征复杂的跨领域系统动力学并实现有效的软测量建模。

    Data-driven soft sensors are essential for achieving accurate perception through reliable state inference. However, developing representative soft sensor models is challenged by issues such as missing labels, domain adaptability, and temporal coherence in data. To address these challenges, we propose a deep Particle Flow Bayes (DPFB) framework for cross-domain soft sensor modeling in the absence of target state labels. In particular, a sequential Bayes objective is first formulated to perform the maximum likelihood estimation underlying the cross-domain soft sensing problem. At the core of the framework, we incorporate a physics-inspired particle flow that optimizes the sequential Bayes objective to perform an exact Bayes update of the model extracted latent and hidden features. As a result, these contributions enable the proposed framework to learn a cohesive approximate posterior feature representation capable of characterizing complex cross-domain system dynamics and performing effe
    
[^12]: 关于具有期望条件风险度量的风险厌恶策略梯度方法的全局收敛性

    On the Global Convergence of Risk-Averse Policy Gradient Methods with Expected Conditional Risk Measures. (arXiv:2301.10932v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.10932](http://arxiv.org/abs/2301.10932)

    本论文研究了具有期望条件风险度量的风险厌恶策略梯度方法，提出了策略梯度更新，证明了其在约束和无约束情况下的全局收敛性和迭代复杂度，并测试了REINFORCE和actor-critic算法的风险厌恶变体来展示方法的实用价值和风险控制的重要性。

    

    风险敏感的强化学习已经成为控制不确定结果和确保各种顺序决策问题的可靠性能的流行工具。虽然针对风险敏感的强化学习已经开发出了策略梯度方法，但这些方法是否具有与风险中性情况下相同的全局收敛保证还不清楚。本文考虑了一类动态时间一致风险度量，称为期望条件风险度量（ECRM），并为基于ECRM的目标函数推导出策略梯度更新。在约束直接参数化和无约束softmax参数化下，我们提供了相应的风险厌恶策略梯度算法的全局收敛性和迭代复杂度。我们进一步测试了REINFORCE和actor-critic算法的风险厌恶变体，以展示我们的方法的有效性和风险控制的重要性。

    Risk-sensitive reinforcement learning (RL) has become a popular tool to control the risk of uncertain outcomes and ensure reliable performance in various sequential decision-making problems. While policy gradient methods have been developed for risk-sensitive RL, it remains unclear if these methods enjoy the same global convergence guarantees as in the risk-neutral case. In this paper, we consider a class of dynamic time-consistent risk measures, called Expected Conditional Risk Measures (ECRMs), and derive policy gradient updates for ECRM-based objective functions. Under both constrained direct parameterization and unconstrained softmax parameterization, we provide global convergence and iteration complexities of the corresponding risk-averse policy gradient algorithms. We further test risk-averse variants of REINFORCE and actor-critic algorithms to demonstrate the efficacy of our method and the importance of risk control.
    

