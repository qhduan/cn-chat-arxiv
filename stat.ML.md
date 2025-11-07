# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A General Theory for Kernel Packets: from state space model to compactly supported basis](https://arxiv.org/abs/2402.04022) | 该论文提出了一种从状态空间模型到紧支持基的核分组的通用理论，该理论可以用于降低高斯过程的训练和预测时间，并且通过适当的线性组合产生了$m$个紧支持的核分组函数。 |
| [^2] | [EERO: Early Exit with Reject Option for Efficient Classification with limited budget](https://arxiv.org/abs/2402.03779) | EERO 是一种早期退出与拒绝选项的新方法，通过使用多个分类器来选择每个实例的退出头，以实现高效分类。实验结果表明，它可以有效管理预算分配并提高准确性。 |

# 详细

[^1]: 一种从状态空间模型到紧支持基的核分组的通用理论

    A General Theory for Kernel Packets: from state space model to compactly supported basis

    [https://arxiv.org/abs/2402.04022](https://arxiv.org/abs/2402.04022)

    该论文提出了一种从状态空间模型到紧支持基的核分组的通用理论，该理论可以用于降低高斯过程的训练和预测时间，并且通过适当的线性组合产生了$m$个紧支持的核分组函数。

    

    众所周知，高斯过程（GP）的状态空间（SS）模型公式可以将其训练和预测时间降低到O（n）（n为数据点个数）。我们证明了一个m维的GP的SS模型公式等价于我们引入的一个概念，称为通用右核分组（KP）：一种用于GP协方差函数K的变换，使得对于任意$t \leq t_1$，$0 \leq j \leq m-1$和$m+1$个连续点$t_i$，都满足$\sum_{i=0}^{m}a_iD_t^{(j)}K(t,t_i)=0$，其中${D}_t^{(j)}f(t)$表示在$t$上作用的第j阶导数。我们将这个思想扩展到了GP的向后SS模型公式，得到了下一个$m$个连续点的左核分组的概念：$\sum_{i=0}^{m}b_i{D}_t^{(j)}K(t,t_{m+i})=0$，对于任意$t\geq t_{2m}$。通过结合左右核分组，可以证明这些协方差函数的适当线性组合产生了$m$个紧支持的核分组函数：对于任意$t\not\in(t_0,t_{2m})$和$j=0,\cdots,m-1$，$\phi^{(j)}(t)=0$。

    It is well known that the state space (SS) model formulation of a Gaussian process (GP) can lower its training and prediction time both to O(n) for n data points. We prove that an $m$-dimensional SS model formulation of GP is equivalent to a concept we introduce as the general right Kernel Packet (KP): a transformation for the GP covariance function $K$ such that $\sum_{i=0}^{m}a_iD_t^{(j)}K(t,t_i)=0$ holds for any $t \leq t_1$, 0 $\leq j \leq m-1$, and $m+1$ consecutive points $t_i$, where ${D}_t^{(j)}f(t) $ denotes $j$-th order derivative acting on $t$. We extend this idea to the backward SS model formulation of the GP, leading to the concept of the left KP for next $m$ consecutive points: $\sum_{i=0}^{m}b_i{D}_t^{(j)}K(t,t_{m+i})=0$ for any $t\geq t_{2m}$. By combining both left and right KPs, we can prove that a suitable linear combination of these covariance functions yields $m$ compactly supported KP functions: $\phi^{(j)}(t)=0$ for any $t\not\in(t_0,t_{2m})$ and $j=0,\cdots,m-1$
    
[^2]: EERO: 早期退出与拒绝选项用于有限预算下的高效分类

    EERO: Early Exit with Reject Option for Efficient Classification with limited budget

    [https://arxiv.org/abs/2402.03779](https://arxiv.org/abs/2402.03779)

    EERO 是一种早期退出与拒绝选项的新方法，通过使用多个分类器来选择每个实例的退出头，以实现高效分类。实验结果表明，它可以有效管理预算分配并提高准确性。

    

    先进的机器学习模型的不断复杂化要求创新的方法来有效管理计算资源。其中一种方法是早期退出策略，通过提供缩短简单数据实例处理路径的机制，实现自适应计算。在本文中，我们提出了EERO，一种将早期退出问题转化为使用具有拒绝选项的多个分类器问题的新方法，以便更好地选择每个实例的退出头。我们使用指数权重聚合来校准不同头部退出的概率，以保证一个固定的预算。我们考虑贝叶斯风险、预算约束和头部特定预算消耗等因素。通过在Cifar和ImageNet数据集上使用ResNet-18模型和ConvNext架构进行的实验结果表明，我们的方法不仅能有效管理预算分配，还能提高过度考虑场景中的准确性。

    The increasing complexity of advanced machine learning models requires innovative approaches to manage computational resources effectively. One such method is the Early Exit strategy, which allows for adaptive computation by providing a mechanism to shorten the processing path for simpler data instances. In this paper, we propose EERO, a new methodology to translate the problem of early exiting to a problem of using multiple classifiers with reject option in order to better select the exiting head for each instance. We calibrate the probabilities of exiting at the different heads using aggregation with exponential weights to guarantee a fixed budget .We consider factors such as Bayesian risk, budget constraints, and head-specific budget consumption. Experimental results, conducted using a ResNet-18 model and a ConvNext architecture on Cifar and ImageNet datasets, demonstrate that our method not only effectively manages budget allocation but also enhances accuracy in overthinking scenar
    

