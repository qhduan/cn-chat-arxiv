# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Sinkhorn's Algorithm and Choice Modeling.](http://arxiv.org/abs/2310.00260) | 该论文研究了使用Luce的选择公理为基础的一类选择和排名模型，证明了与最大似然估计问题等效的经典矩阵平衡问题。通过Sinkhorn算法，将选择建模算法统一为矩阵平衡算法的特例。论文还解决了Sinkhorn算法研究中的重要问题，包括对于非负矩阵的全局线性收敛和尖锐渐近速度的描述。 |
| [^2] | [Reparametrization and the Semiparametric Bernstein-von-Mises Theorem.](http://arxiv.org/abs/2306.03816) | 本文提出了一种参数化形式，该形式可以通过生成Neyman正交矩条件来降低对干扰参数的敏感度，从而可以用于去偏贝叶斯推断中的后验分布，同时在参数速率下对低维参数进行真实值的收缩，并在半参数效率界的方差下进行渐近正态分布。 |

# 详细

[^1]: 关于Sinkhorn算法和选择建模的论文

    On Sinkhorn's Algorithm and Choice Modeling. (arXiv:2310.00260v1 [math.OC])

    [http://arxiv.org/abs/2310.00260](http://arxiv.org/abs/2310.00260)

    该论文研究了使用Luce的选择公理为基础的一类选择和排名模型，证明了与最大似然估计问题等效的经典矩阵平衡问题。通过Sinkhorn算法，将选择建模算法统一为矩阵平衡算法的特例。论文还解决了Sinkhorn算法研究中的重要问题，包括对于非负矩阵的全局线性收敛和尖锐渐近速度的描述。

    

    对于基于Luce选择公理的广泛选择和排名模型，包括Bradley-Terry-Luce和Plackett-Luce模型，我们证明了相关的最大似然估计问题等价于具有目标行和列和的经典矩阵平衡问题。这个观点打开了两个看似不相关的研究领域之间的交流之门，并使我们能够将选择建模文献中的现有算法统一为Sinkhorn矩阵平衡算法的特殊实例或类似物。我们从这些联系中获得启发，并解决了Sinkhorn算法研究中的重要开放性问题。我们首先证明了当矩阵平衡问题存在有限解时，Sinkhorn算法对于非负矩阵的全局线性收敛。我们通过数据构建的二分图的代数连通性来描述这种全局收敛速度。接下来，我们还得出了线性收敛的尖锐渐近速度。

    For a broad class of choice and ranking models based on Luce's choice axiom, including the Bradley--Terry--Luce and Plackett--Luce models, we show that the associated maximum likelihood estimation problems are equivalent to a classic matrix balancing problem with target row and column sums. This perspective opens doors between two seemingly unrelated research areas, and allows us to unify existing algorithms in the choice modeling literature as special instances or analogs of Sinkhorn's celebrated algorithm for matrix balancing. We draw inspirations from these connections and resolve important open problems on the study of Sinkhorn's algorithm. We first prove the global linear convergence of Sinkhorn's algorithm for non-negative matrices whenever finite solutions to the matrix balancing problem exist. We characterize this global rate of convergence in terms of the algebraic connectivity of the bipartite graph constructed from data. Next, we also derive the sharp asymptotic rate of line
    
[^2]: 重参数化与半参数Bernstein-von-Mises定理

    Reparametrization and the Semiparametric Bernstein-von-Mises Theorem. (arXiv:2306.03816v1 [math.ST])

    [http://arxiv.org/abs/2306.03816](http://arxiv.org/abs/2306.03816)

    本文提出了一种参数化形式，该形式可以通过生成Neyman正交矩条件来降低对干扰参数的敏感度，从而可以用于去偏贝叶斯推断中的后验分布，同时在参数速率下对低维参数进行真实值的收缩，并在半参数效率界的方差下进行渐近正态分布。

    

    本文考虑了部分线性模型的贝叶斯推断。我们的方法利用了回归函数的一个参数化形式，该形式专门用于估计所关心的低维参数。参数化的关键特性是生成了一个Neyman正交矩条件，这意味着对干扰参数的估计低维参数不太敏感。我们的大样本分析支持了这种说法。特别地，我们推导出充分的条件，使得低维参数的后验在参数速率下对真实值收缩，并且在半参数效率界的方差下渐近地正态分布。这些条件相对于回归模型的原始参数化允许更大类的干扰参数。总的来说，我们得出结论，一个嵌入了Neyman正交性的参数化方法可以成为半参数推断中的一个有用工具，以去偏后验分布。

    This paper considers Bayesian inference for the partially linear model. Our approach exploits a parametrization of the regression function that is tailored toward estimating a low-dimensional parameter of interest. The key property of the parametrization is that it generates a Neyman orthogonal moment condition meaning that the low-dimensional parameter is less sensitive to the estimation of nuisance parameters. Our large sample analysis supports this claim. In particular, we derive sufficient conditions under which the posterior for the low-dimensional parameter contracts around the truth at the parametric rate and is asymptotically normal with a variance that coincides with the semiparametric efficiency bound. These conditions allow for a larger class of nuisance parameters relative to the original parametrization of the regression model. Overall, we conclude that a parametrization that embeds Neyman orthogonality can be a useful device for debiasing posterior distributions in semipa
    

