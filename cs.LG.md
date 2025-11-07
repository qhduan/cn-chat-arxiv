# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Unified Kernel for Neural Network Learning](https://arxiv.org/abs/2403.17467) | 本文提出了统一神经内核(UNK)，可以描述神经网络的学习动态，并在有限的学习步骤下表现出类似于NTK的行为，当学习步骤逼近无穷大时收敛到NNGP。 |
| [^2] | [A General Theory for Kernel Packets: from state space model to compactly supported basis](https://arxiv.org/abs/2402.04022) | 该论文提出了一种从状态空间模型到紧支持基的核分组的通用理论，该理论可以用于降低高斯过程的训练和预测时间，并且通过适当的线性组合产生了$m$个紧支持的核分组函数。 |
| [^3] | [EERO: Early Exit with Reject Option for Efficient Classification with limited budget](https://arxiv.org/abs/2402.03779) | EERO 是一种早期退出与拒绝选项的新方法，通过使用多个分类器来选择每个实例的退出头，以实现高效分类。实验结果表明，它可以有效管理预算分配并提高准确性。 |
| [^4] | [Generative Noisy-Label Learning by Implicit Dicriminative Approximation with Partial Label Prior.](http://arxiv.org/abs/2308.01184) | 本文提出了一种新的生成噪声标签学习方法，直接关联数据和干净标签，通过使用判别的近似方法来隐式估计生成模型，解决了传统方法中的复杂公式、难以训练的生成模型和无信息先验的问题。 |
| [^5] | [Improving Expressivity of Graph Neural Networks using Localization.](http://arxiv.org/abs/2305.19659) | 本文提出了Weisfeiler-Leman (WL)算法的局部版本，用于解决子图计数问题并提高图神经网络的表达能力，同时，也给出了一些时间和空间效率更高的$k-$WL变体和分裂技术。 |
| [^6] | [Approximate non-linear model predictive control with safety-augmented neural networks.](http://arxiv.org/abs/2304.09575) | 本文提出了一种基于神经网络(NNs)的非线性模型预测控制(MPC)的近似方法，称为安全增强，可以使解决方案在线可行并具有收敛和约束条件的确定保证。 |

# 详细

[^1]: 一个统一的神经网络学习内核

    A Unified Kernel for Neural Network Learning

    [https://arxiv.org/abs/2403.17467](https://arxiv.org/abs/2403.17467)

    本文提出了统一神经内核(UNK)，可以描述神经网络的学习动态，并在有限的学习步骤下表现出类似于NTK的行为，当学习步骤逼近无穷大时收敛到NNGP。

    

    过去几十年来，人们对神经网络学习和内核学习之间的区别和联系表现出极大的兴趣。最近的进展在连接无限宽神经网络和高斯过程方面取得了理论上的进展。出现了两种主流方法：神经网络高斯过程(NNGP)和神经切向核(NTK)。前者基于贝叶斯推断，代表了零阶核，而后者基于梯度下降的切向空间，是第一阶核。在本文中，我们提出了统一神经内核(UNK)，该内核表征了神经网络在梯度下降和参数初始化中的学习动态。所提出的UNK内核保持了NNGP和NTK的极限特性，表现出类似于NTK的行为，但有有限的学习步骤，并且当学习步骤接近无穷大时收敛到NNGP。此外，我们还从理论上对UNK内核进行了分析。

    arXiv:2403.17467v1 Announce Type: cross  Abstract: Past decades have witnessed a great interest in the distinction and connection between neural network learning and kernel learning. Recent advancements have made theoretical progress in connecting infinite-wide neural networks and Gaussian processes. Two predominant approaches have emerged: the Neural Network Gaussian Process (NNGP) and the Neural Tangent Kernel (NTK). The former, rooted in Bayesian inference, represents a zero-order kernel, while the latter, grounded in the tangent space of gradient descents, is a first-order kernel. In this paper, we present the Unified Neural Kernel (UNK), which characterizes the learning dynamics of neural networks with gradient descents and parameter initialization. The proposed UNK kernel maintains the limiting properties of both NNGP and NTK, exhibiting behaviors akin to NTK with a finite learning step and converging to NNGP as the learning step approaches infinity. Besides, we also theoreticall
    
[^2]: 一种从状态空间模型到紧支持基的核分组的通用理论

    A General Theory for Kernel Packets: from state space model to compactly supported basis

    [https://arxiv.org/abs/2402.04022](https://arxiv.org/abs/2402.04022)

    该论文提出了一种从状态空间模型到紧支持基的核分组的通用理论，该理论可以用于降低高斯过程的训练和预测时间，并且通过适当的线性组合产生了$m$个紧支持的核分组函数。

    

    众所周知，高斯过程（GP）的状态空间（SS）模型公式可以将其训练和预测时间降低到O（n）（n为数据点个数）。我们证明了一个m维的GP的SS模型公式等价于我们引入的一个概念，称为通用右核分组（KP）：一种用于GP协方差函数K的变换，使得对于任意$t \leq t_1$，$0 \leq j \leq m-1$和$m+1$个连续点$t_i$，都满足$\sum_{i=0}^{m}a_iD_t^{(j)}K(t,t_i)=0$，其中${D}_t^{(j)}f(t)$表示在$t$上作用的第j阶导数。我们将这个思想扩展到了GP的向后SS模型公式，得到了下一个$m$个连续点的左核分组的概念：$\sum_{i=0}^{m}b_i{D}_t^{(j)}K(t,t_{m+i})=0$，对于任意$t\geq t_{2m}$。通过结合左右核分组，可以证明这些协方差函数的适当线性组合产生了$m$个紧支持的核分组函数：对于任意$t\not\in(t_0,t_{2m})$和$j=0,\cdots,m-1$，$\phi^{(j)}(t)=0$。

    It is well known that the state space (SS) model formulation of a Gaussian process (GP) can lower its training and prediction time both to O(n) for n data points. We prove that an $m$-dimensional SS model formulation of GP is equivalent to a concept we introduce as the general right Kernel Packet (KP): a transformation for the GP covariance function $K$ such that $\sum_{i=0}^{m}a_iD_t^{(j)}K(t,t_i)=0$ holds for any $t \leq t_1$, 0 $\leq j \leq m-1$, and $m+1$ consecutive points $t_i$, where ${D}_t^{(j)}f(t) $ denotes $j$-th order derivative acting on $t$. We extend this idea to the backward SS model formulation of the GP, leading to the concept of the left KP for next $m$ consecutive points: $\sum_{i=0}^{m}b_i{D}_t^{(j)}K(t,t_{m+i})=0$ for any $t\geq t_{2m}$. By combining both left and right KPs, we can prove that a suitable linear combination of these covariance functions yields $m$ compactly supported KP functions: $\phi^{(j)}(t)=0$ for any $t\not\in(t_0,t_{2m})$ and $j=0,\cdots,m-1$
    
[^3]: EERO: 早期退出与拒绝选项用于有限预算下的高效分类

    EERO: Early Exit with Reject Option for Efficient Classification with limited budget

    [https://arxiv.org/abs/2402.03779](https://arxiv.org/abs/2402.03779)

    EERO 是一种早期退出与拒绝选项的新方法，通过使用多个分类器来选择每个实例的退出头，以实现高效分类。实验结果表明，它可以有效管理预算分配并提高准确性。

    

    先进的机器学习模型的不断复杂化要求创新的方法来有效管理计算资源。其中一种方法是早期退出策略，通过提供缩短简单数据实例处理路径的机制，实现自适应计算。在本文中，我们提出了EERO，一种将早期退出问题转化为使用具有拒绝选项的多个分类器问题的新方法，以便更好地选择每个实例的退出头。我们使用指数权重聚合来校准不同头部退出的概率，以保证一个固定的预算。我们考虑贝叶斯风险、预算约束和头部特定预算消耗等因素。通过在Cifar和ImageNet数据集上使用ResNet-18模型和ConvNext架构进行的实验结果表明，我们的方法不仅能有效管理预算分配，还能提高过度考虑场景中的准确性。

    The increasing complexity of advanced machine learning models requires innovative approaches to manage computational resources effectively. One such method is the Early Exit strategy, which allows for adaptive computation by providing a mechanism to shorten the processing path for simpler data instances. In this paper, we propose EERO, a new methodology to translate the problem of early exiting to a problem of using multiple classifiers with reject option in order to better select the exiting head for each instance. We calibrate the probabilities of exiting at the different heads using aggregation with exponential weights to guarantee a fixed budget .We consider factors such as Bayesian risk, budget constraints, and head-specific budget consumption. Experimental results, conducted using a ResNet-18 model and a ConvNext architecture on Cifar and ImageNet datasets, demonstrate that our method not only effectively manages budget allocation but also enhances accuracy in overthinking scenar
    
[^4]: 通过部分标签先验的隐式判别逼近进行生成噪声标签学习

    Generative Noisy-Label Learning by Implicit Dicriminative Approximation with Partial Label Prior. (arXiv:2308.01184v1 [cs.CV])

    [http://arxiv.org/abs/2308.01184](http://arxiv.org/abs/2308.01184)

    本文提出了一种新的生成噪声标签学习方法，直接关联数据和干净标签，通过使用判别的近似方法来隐式估计生成模型，解决了传统方法中的复杂公式、难以训练的生成模型和无信息先验的问题。

    

    对于带有噪声标签的学习问题，已经使用了判别模型和生成模型进行研究。尽管判别模型由于其简单的建模和更高效的计算训练过程而在该领域占主导地位，但生成模型能够更有效地分解干净和噪声标签，并改善标签转换矩阵的估计。然而，生成方法使用了复杂的公式来最大化噪声标签和数据的联合似然，这只间接优化了与数据和干净标签相关的感兴趣的模型。此外，这些方法依赖于很难训练的生成模型，并倾向于使用无信息的干净标签先验。在本文中，我们提出了一个新的生成噪声标签学习方法来解决这三个问题。首先，我们提出了一种新的模型优化方法，直接关联数据和干净标签。其次，通过使用判别的近似方法来隐式估计生成模型。

    The learning with noisy labels has been addressed with both discriminative and generative models. Although discriminative models have dominated the field due to their simpler modeling and more efficient computational training processes, generative models offer a more effective means of disentangling clean and noisy labels and improving the estimation of the label transition matrix. However, generative approaches maximize the joint likelihood of noisy labels and data using a complex formulation that only indirectly optimizes the model of interest associating data and clean labels. Additionally, these approaches rely on generative models that are challenging to train and tend to use uninformative clean label priors. In this paper, we propose a new generative noisy-label learning approach that addresses these three issues. First, we propose a new model optimisation that directly associates data and clean labels. Second, the generative model is implicitly estimated using a discriminative m
    
[^5]: 利用局部化提高图神经网络的表达能力

    Improving Expressivity of Graph Neural Networks using Localization. (arXiv:2305.19659v1 [cs.LG])

    [http://arxiv.org/abs/2305.19659](http://arxiv.org/abs/2305.19659)

    本文提出了Weisfeiler-Leman (WL)算法的局部版本，用于解决子图计数问题并提高图神经网络的表达能力，同时，也给出了一些时间和空间效率更高的$k-$WL变体和分裂技术。

    

    本文提出了Weisfeiler-Leman (WL)算法的局部版本，旨在增加表达能力并减少计算负担。我们专注于子图计数问题，并为任意$k$给出$k-$WL的局部版本。我们分析了Local $k-$WL的作用，并证明其比$k-$WL更具表现力，并且至多与$(k+1)-$WL一样具有表现力。我们给出了一些模式的特征，如果两个图是Local $k-$WL等价的，则它们的子图和诱导子图的计数是不变的。我们还介绍了$k-$WL的两个变体：层$k-$WL和递归$k-$WL。这些方法的时间和空间效率比在整个图上应用$k-$WL更高。我们还提出了一种分裂技术，使用$1-$WL即可保证所有大小不超过4的诱导子图的准确计数。相同的方法可以使用$k>1$进一步扩展到更大的模式。我们还将Local $k-$WL的表现力与其他GNN层次结构进行了比较。

    In this paper, we propose localized versions of Weisfeiler-Leman (WL) algorithms in an effort to both increase the expressivity, as well as decrease the computational overhead. We focus on the specific problem of subgraph counting and give localized versions of $k-$WL for any $k$. We analyze the power of Local $k-$WL and prove that it is more expressive than $k-$WL and at most as expressive as $(k+1)-$WL. We give a characterization of patterns whose count as a subgraph and induced subgraph are invariant if two graphs are Local $k-$WL equivalent. We also introduce two variants of $k-$WL: Layer $k-$WL and recursive $k-$WL. These methods are more time and space efficient than applying $k-$WL on the whole graph. We also propose a fragmentation technique that guarantees the exact count of all induced subgraphs of size at most 4 using just $1-$WL. The same idea can be extended further for larger patterns using $k>1$. We also compare the expressive power of Local $k-$WL with other GNN hierarc
    
[^6]: 基于安全增强神经网络的非线性近似模型预测控制

    Approximate non-linear model predictive control with safety-augmented neural networks. (arXiv:2304.09575v1 [eess.SY])

    [http://arxiv.org/abs/2304.09575](http://arxiv.org/abs/2304.09575)

    本文提出了一种基于神经网络(NNs)的非线性模型预测控制(MPC)的近似方法，称为安全增强，可以使解决方案在线可行并具有收敛和约束条件的确定保证。

    

    模型预测控制(MPC)可以实现对于一般非线性系统的稳定性和约束条件的满足，但需要进行计算开销很大的在线优化。本文研究了通过神经网络(NNs)对这种MPC控制器的近似，以实现快速的在线评估。我们提出了安全增强，尽管存在近似不准确性，但可以获得收敛和约束条件的确定保证。我们使用NN近似MPC的整个输入序列，这使得我们在线验证它是否是MPC问题的可行解。当该解决方案不可行或成本更高时，我们基于标准MPC技术将NN解决方案替换为安全候选解。我们的方法仅需要对NN进行一次评估和对输入序列进行在线前向积分，这在资源受限系统上的计算速度很快。所提出的控制框架在三个不同复杂度的非线性MPC基准上进行了演示，展示了计算效率。

    Model predictive control (MPC) achieves stability and constraint satisfaction for general nonlinear systems, but requires computationally expensive online optimization. This paper studies approximations of such MPC controllers via neural networks (NNs) to achieve fast online evaluation. We propose safety augmentation that yields deterministic guarantees for convergence and constraint satisfaction despite approximation inaccuracies. We approximate the entire input sequence of the MPC with NNs, which allows us to verify online if it is a feasible solution to the MPC problem. We replace the NN solution by a safe candidate based on standard MPC techniques whenever it is infeasible or has worse cost. Our method requires a single evaluation of the NN and forward integration of the input sequence online, which is fast to compute on resource-constrained systems. The proposed control framework is illustrated on three non-linear MPC benchmarks of different complexity, demonstrating computational
    

