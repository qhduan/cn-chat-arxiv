# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Geometric structure of shallow neural networks and constructive ${\mathcal L}^2$ cost minimization.](http://arxiv.org/abs/2309.10370) | 本文提供了浅层神经网络的几何结构解释，并通过基于${\mathcal L}^2$代价最小化的构造方法获得了一个具有优越性能的网络。 |
| [^2] | [Gradient is All You Need?.](http://arxiv.org/abs/2306.09778) | 本文提供了一种新的角度分析了基于梯度的学习算法，将一种新的多粒子无导数优化方法解释为梯度下降的随机松弛方法。此优化方法证明了零阶方法并不一定低效或不具备泛化能力，并且可以在丰富类别的非光滑和非凸目标函数下全局收敛于全局最小值。 |

# 详细

[^1]: 浅层神经网络的几何结构和基于${\mathcal L}^2$代价最小化的构造方法

    Geometric structure of shallow neural networks and constructive ${\mathcal L}^2$ cost minimization. (arXiv:2309.10370v1 [cs.LG])

    [http://arxiv.org/abs/2309.10370](http://arxiv.org/abs/2309.10370)

    本文提供了浅层神经网络的几何结构解释，并通过基于${\mathcal L}^2$代价最小化的构造方法获得了一个具有优越性能的网络。

    

    本文给出了一个几何解释：浅层神经网络的结构由一个隐藏层、一个斜坡激活函数、一个${\mathcal L}^2$谱范类（或者Hilbert-Schmidt）的代价函数、输入空间${\mathbb R}^M$、输出空间${\mathbb R}^Q$（其中$Q\leq M$），以及训练输入样本数量$N>QM$所特征。我们证明了代价函数的最小值具有$O(\delta_P)$的上界，其中$\delta_P$衡量了训练输入的信噪比。我们使用适应于属于同一输出向量$y_j$的训练输入向量$\overline{x_{0,j}}$的投影来获得近似的优化器，其中$j=1,\dots,Q$。在特殊情况$M=Q$下，我们明确确定了代价函数的一个确切退化局部最小值；这个尖锐的值与对于$Q\leq M$所获得的上界之间有一个相对误差$O(\delta_P^2)$。上界证明的方法提供了一个构造性训练的网络；我们证明它测度了$Q$维空间中的给定输出。

    In this paper, we provide a geometric interpretation of the structure of shallow neural networks characterized by one hidden layer, a ramp activation function, an ${\mathcal L}^2$ Schatten class (or Hilbert-Schmidt) cost function, input space ${\mathbb R}^M$, output space ${\mathbb R}^Q$ with $Q\leq M$, and training input sample size $N>QM$. We prove an upper bound on the minimum of the cost function of order $O(\delta_P$ where $\delta_P$ measures the signal to noise ratio of training inputs. We obtain an approximate optimizer using projections adapted to the averages $\overline{x_{0,j}}$ of training input vectors belonging to the same output vector $y_j$, $j=1,\dots,Q$. In the special case $M=Q$, we explicitly determine an exact degenerate local minimum of the cost function; the sharp value differs from the upper bound obtained for $Q\leq M$ by a relative error $O(\delta_P^2)$. The proof of the upper bound yields a constructively trained network; we show that it metrizes the $Q$-dimen
    
[^2]: 梯度真的是你所需要的一切吗？

    Gradient is All You Need?. (arXiv:2306.09778v1 [cs.LG])

    [http://arxiv.org/abs/2306.09778](http://arxiv.org/abs/2306.09778)

    本文提供了一种新的角度分析了基于梯度的学习算法，将一种新的多粒子无导数优化方法解释为梯度下降的随机松弛方法。此优化方法证明了零阶方法并不一定低效或不具备泛化能力，并且可以在丰富类别的非光滑和非凸目标函数下全局收敛于全局最小值。

    

    本文提供了一种新的分析方法，通过将一种新的多粒子无导数优化方法结合梯度下降看作随机松弛方法，来解释基于梯度的学习算法的理论理解。通过粒子之间的通讯，这种优化方法表现出类似于随机梯度下降的行为，证明了零阶方法并不一定低效或不具备泛化能力，并且可以在非光滑和非凸目标函数的丰富类别下全局收敛于全局最小值。

    In this paper we provide a novel analytical perspective on the theoretical understanding of gradient-based learning algorithms by interpreting consensus-based optimization (CBO), a recently proposed multi-particle derivative-free optimization method, as a stochastic relaxation of gradient descent. Remarkably, we observe that through communication of the particles, CBO exhibits a stochastic gradient descent (SGD)-like behavior despite solely relying on evaluations of the objective function. The fundamental value of such link between CBO and SGD lies in the fact that CBO is provably globally convergent to global minimizers for ample classes of nonsmooth and nonconvex objective functions, hence, on the one side, offering a novel explanation for the success of stochastic relaxations of gradient descent. On the other side, contrary to the conventional wisdom for which zero-order methods ought to be inefficient or not to possess generalization abilities, our results unveil an intrinsic gradi
    

