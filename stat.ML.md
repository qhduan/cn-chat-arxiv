# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the rates of convergence for learning with convolutional neural networks](https://arxiv.org/abs/2403.16459) | 该研究提出了对具有一定权重约束的CNNs的新逼近上界，以及对前馈神经网络的覆盖数做了新的分析，为基于CNNs的学习问题推导了收敛速率，并在学习平滑函数和二元分类方面取得了极小最优的结果。 |
| [^2] | [Stochastic optimal transport in Banach Spaces for regularized estimation of multivariate quantiles](https://arxiv.org/abs/2302.00982) | 提出了一种新的随机算法，用于利用傅里叶系数解决熵最优输运问题，并研究了其在无限维Banach空间中的几乎肯定收敛性和性能表现。 |
| [^3] | [Geometric structure of shallow neural networks and constructive ${\mathcal L}^2$ cost minimization.](http://arxiv.org/abs/2309.10370) | 本文提供了浅层神经网络的几何结构解释，并通过基于${\mathcal L}^2$代价最小化的构造方法获得了一个具有优越性能的网络。 |
| [^4] | [Learning High-Dimensional Nonparametric Differential Equations via Multivariate Occupation Kernel Functions.](http://arxiv.org/abs/2306.10189) | 本论文提出了一种线性方法，通过多元占位核函数在高维状态空间中学习非参数ODE系统，可以解决显式公式按二次方缩放的问题。这种方法在高度非线性的数据和图像数据中都具有通用性。 |
| [^5] | [Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning.](http://arxiv.org/abs/2305.15612) | 该论文提出了一种基于密度比估计和半监督学习的贝叶斯优化方法，通过使用监督分类器代替密度比来估计全局最优解的两组数据的类别概率，避免了分类器对全局解决方案过于自信的问题。 |

# 详细

[^1]: 关于使用卷积神经网络进行学习收敛速率的研究

    On the rates of convergence for learning with convolutional neural networks

    [https://arxiv.org/abs/2403.16459](https://arxiv.org/abs/2403.16459)

    该研究提出了对具有一定权重约束的CNNs的新逼近上界，以及对前馈神经网络的覆盖数做了新的分析，为基于CNNs的学习问题推导了收敛速率，并在学习平滑函数和二元分类方面取得了极小最优的结果。

    

    我们研究了卷积神经网络（CNNs）的逼近和学习能力。第一个结果证明了在权重上有一定约束条件下CNNs的新逼近上界。第二个结果给出了对前馈神经网络的覆盖数的新分析，其中CNNs是其特例。该分析详细考虑了权重的大小，在某些情况下给出了比现有文献更好的上界。利用这两个结果，我们能够推导基于CNNs的估计器在许多学习问题中的收敛速率。特别地，我们在非参数回归设置中为基于CNNs的最小二乘学习平滑函数建立了极小最优的收敛速率。对于二元分类，我们推导了具有铰链损失和逻辑损失的CNN分类器的收敛速度。同时还表明所得到的速率在几种情况下是极小最优的。

    arXiv:2403.16459v1 Announce Type: new  Abstract: We study the approximation and learning capacities of convolutional neural networks (CNNs). Our first result proves a new approximation bound for CNNs with certain constraint on the weights. Our second result gives a new analysis on the covering number of feed-forward neural networks, which include CNNs as special cases. The analysis carefully takes into account the size of the weights and hence gives better bounds than existing literature in some situations. Using these two results, we are able to derive rates of convergence for estimators based on CNNs in many learning problems. In particular, we establish minimax optimal convergence rates of the least squares based on CNNs for learning smooth functions in the nonparametric regression setting. For binary classification, we derive convergence rates for CNN classifiers with hinge loss and logistic loss. It is also shown that the obtained rates are minimax optimal in several settings.
    
[^2]: 在Banach空间中的随机最优输运用于多元分位数的正则化估计

    Stochastic optimal transport in Banach Spaces for regularized estimation of multivariate quantiles

    [https://arxiv.org/abs/2302.00982](https://arxiv.org/abs/2302.00982)

    提出了一种新的随机算法，用于利用傅里叶系数解决熵最优输运问题，并研究了其在无限维Banach空间中的几乎肯定收敛性和性能表现。

    

    我们介绍了一种新的随机算法，用于解决两个绝对连续概率测度$\mu$和$\nu$之间的熵最优输运（EOT）。我们的工作受蒙日-坎托罗维奇分位数的特定设置启发，其中源测度$\mu$要么是单位超立方体上的均匀分布，要么是球面均匀分布。利用源测度的知识，我们建议通过其傅里叶系数来参数化坎托罗维奇对偶势能。通过这种方式，我们的随机算法的每次迭代都会减少到两个傅里叶变换，使我们能够利用快速傅里叶变换（FFT）来实现求解EOT的快速数值方法。我们研究了我们的随机算法在取值于无限维Banach空间中的几乎肯定收敛性。然后，通过数值实验，我们展示了我们的方法在计算中的性能表现。

    arXiv:2302.00982v2 Announce Type: replace-cross  Abstract: We introduce a new stochastic algorithm for solving entropic optimal transport (EOT) between two absolutely continuous probability measures $\mu$ and $\nu$. Our work is motivated by the specific setting of Monge-Kantorovich quantiles where the source measure $\mu$ is either the uniform distribution on the unit hypercube or the spherical uniform distribution. Using the knowledge of the source measure, we propose to parametrize a Kantorovich dual potential by its Fourier coefficients. In this way, each iteration of our stochastic algorithm reduces to two Fourier transforms that enables us to make use of the Fast Fourier Transform (FFT) in order to implement a fast numerical method to solve EOT. We study the almost sure convergence of our stochastic algorithm that takes its values in an infinite-dimensional Banach space. Then, using numerical experiments, we illustrate the performances of our approach on the computation of regular
    
[^3]: 浅层神经网络的几何结构和基于${\mathcal L}^2$代价最小化的构造方法

    Geometric structure of shallow neural networks and constructive ${\mathcal L}^2$ cost minimization. (arXiv:2309.10370v1 [cs.LG])

    [http://arxiv.org/abs/2309.10370](http://arxiv.org/abs/2309.10370)

    本文提供了浅层神经网络的几何结构解释，并通过基于${\mathcal L}^2$代价最小化的构造方法获得了一个具有优越性能的网络。

    

    本文给出了一个几何解释：浅层神经网络的结构由一个隐藏层、一个斜坡激活函数、一个${\mathcal L}^2$谱范类（或者Hilbert-Schmidt）的代价函数、输入空间${\mathbb R}^M$、输出空间${\mathbb R}^Q$（其中$Q\leq M$），以及训练输入样本数量$N>QM$所特征。我们证明了代价函数的最小值具有$O(\delta_P)$的上界，其中$\delta_P$衡量了训练输入的信噪比。我们使用适应于属于同一输出向量$y_j$的训练输入向量$\overline{x_{0,j}}$的投影来获得近似的优化器，其中$j=1,\dots,Q$。在特殊情况$M=Q$下，我们明确确定了代价函数的一个确切退化局部最小值；这个尖锐的值与对于$Q\leq M$所获得的上界之间有一个相对误差$O(\delta_P^2)$。上界证明的方法提供了一个构造性训练的网络；我们证明它测度了$Q$维空间中的给定输出。

    In this paper, we provide a geometric interpretation of the structure of shallow neural networks characterized by one hidden layer, a ramp activation function, an ${\mathcal L}^2$ Schatten class (or Hilbert-Schmidt) cost function, input space ${\mathbb R}^M$, output space ${\mathbb R}^Q$ with $Q\leq M$, and training input sample size $N>QM$. We prove an upper bound on the minimum of the cost function of order $O(\delta_P$ where $\delta_P$ measures the signal to noise ratio of training inputs. We obtain an approximate optimizer using projections adapted to the averages $\overline{x_{0,j}}$ of training input vectors belonging to the same output vector $y_j$, $j=1,\dots,Q$. In the special case $M=Q$, we explicitly determine an exact degenerate local minimum of the cost function; the sharp value differs from the upper bound obtained for $Q\leq M$ by a relative error $O(\delta_P^2)$. The proof of the upper bound yields a constructively trained network; we show that it metrizes the $Q$-dimen
    
[^4]: 通过多元占位核函数学习高维非参数微分方程

    Learning High-Dimensional Nonparametric Differential Equations via Multivariate Occupation Kernel Functions. (arXiv:2306.10189v1 [stat.ML])

    [http://arxiv.org/abs/2306.10189](http://arxiv.org/abs/2306.10189)

    本论文提出了一种线性方法，通过多元占位核函数在高维状态空间中学习非参数ODE系统，可以解决显式公式按二次方缩放的问题。这种方法在高度非线性的数据和图像数据中都具有通用性。

    

    从$d$维状态空间中$n$个轨迹快照中学习非参数的常微分方程（ODE）系统需要学习$d$个函数。除非具有额外的系统属性知识，例如稀疏性和对称性，否则显式的公式按二次方缩放。在这项工作中，我们提出了一种使用向量值再生核希尔伯特空间提供的隐式公式学习的线性方法。通过将ODE重写为更弱的积分形式，我们随后进行最小化并推导出我们的学习算法。最小化问题的解向量场依赖于与解轨迹相关的多元占位核函数。我们通过对高度非线性的模拟和真实数据进行实验证实了我们的方法，其中$d$可能超过100。我们进一步通过从图像数据学习非参数一阶拟线性偏微分方程展示了所提出的方法的多样性。

    Learning a nonparametric system of ordinary differential equations (ODEs) from $n$ trajectory snapshots in a $d$-dimensional state space requires learning $d$ functions of $d$ variables. Explicit formulations scale quadratically in $d$ unless additional knowledge about system properties, such as sparsity and symmetries, is available. In this work, we propose a linear approach to learning using the implicit formulation provided by vector-valued Reproducing Kernel Hilbert Spaces. By rewriting the ODEs in a weaker integral form, which we subsequently minimize, we derive our learning algorithm. The minimization problem's solution for the vector field relies on multivariate occupation kernel functions associated with the solution trajectories. We validate our approach through experiments on highly nonlinear simulated and real data, where $d$ may exceed 100. We further demonstrate the versatility of the proposed method by learning a nonparametric first order quasilinear partial differential 
    
[^5]: 基于密度比估计的半监督学习贝叶斯优化

    Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning. (arXiv:2305.15612v1 [cs.LG])

    [http://arxiv.org/abs/2305.15612](http://arxiv.org/abs/2305.15612)

    该论文提出了一种基于密度比估计和半监督学习的贝叶斯优化方法，通过使用监督分类器代替密度比来估计全局最优解的两组数据的类别概率，避免了分类器对全局解决方案过于自信的问题。

    

    贝叶斯优化在科学与工程的多个领域受到了广泛关注，因为它能高效地找到昂贵黑盒函数的全局最优解。通常，一个概率回归模型，如高斯过程、随机森林和贝叶斯神经网络，被广泛用作替代函数，用于模拟在给定输入和训练数据集的情况下函数评估的显式分布。除了基于概率回归的贝叶斯优化，基于密度比估计的贝叶斯优化已被提出来估计相对于全局最优解相对接近和相对远离的两组密度比。为了进一步发展这一研究，可以使用监督分类器来估计这两组的类别概率，而不是密度比。然而，此策略中使用的监督分类器倾向于对全局解决方案过于自信。

    Bayesian optimization has attracted huge attention from diverse research areas in science and engineering, since it is capable of finding a global optimum of an expensive-to-evaluate black-box function efficiently. In general, a probabilistic regression model, e.g., Gaussian processes, random forests, and Bayesian neural networks, is widely used as a surrogate function to model an explicit distribution over function evaluations given an input to estimate and a training dataset. Beyond the probabilistic regression-based Bayesian optimization, density ratio estimation-based Bayesian optimization has been suggested in order to estimate a density ratio of the groups relatively close and relatively far to a global optimum. Developing this line of research further, a supervised classifier can be employed to estimate a class probability for the two groups instead of a density ratio. However, the supervised classifiers used in this strategy tend to be overconfident for a global solution candid
    

