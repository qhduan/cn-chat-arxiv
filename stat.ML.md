# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Non-negative Contrastive Learning](https://arxiv.org/abs/2403.12459) | 非负对比学习(NCL)是对非负矩阵分解(NMF)的重新演绎，通过对特征施加非负约束来获得可解释的特征，保留了NMF的可解释属性，从而得到比标准对比学习(CL)更稀疏和解耦的表示 |
| [^2] | [Imputation using training labels and classification via label imputation.](http://arxiv.org/abs/2311.16877) | 本论文提出一种在填充缺失数据时将标签与输入堆叠的方法，能够显著提高填充效果，并同时填充标签和输入。该方法适用于各种类型的数据，且在实验证明具有有希望的准确性结果。 |
| [^3] | [Mixtures of Gaussians are Privately Learnable with a Polynomial Number of Samples.](http://arxiv.org/abs/2309.03847) | 通过多项式数量的样本和差分隐私约束，我们提出了一个可以估计高斯混合物的方法，并证明了这个方法的有效性，而无需对GMMs做任何结构性假设。 |
| [^4] | [Linear Convergence of Black-Box Variational Inference: Should We Stick the Landing?.](http://arxiv.org/abs/2307.14642) | 本文证明了带有控制变量的黑盒变分推断在完美变分族规范下以几何速度收敛，为BBVI提供了收敛性保证，同时提出了对熵梯度估计器的改进，对比了STL估计器，并给出了明确的非渐近复杂度保证。 |
| [^5] | [A Majorization-Minimization Gauss-Newton Method for 1-Bit Matrix Completion.](http://arxiv.org/abs/2304.13940) | 本文提出了一种基于主导-最小化原则，通过低秩矩阵补全解决1比特矩阵补全问题的新方法，称为MMGN。通过应用高斯-牛顿方法，MMGN具有更快的速度和更准确的结果，同时还不太受到潜在矩阵尖锐度的影响。 |

# 详细

[^1]: 非负对比学习

    Non-negative Contrastive Learning

    [https://arxiv.org/abs/2403.12459](https://arxiv.org/abs/2403.12459)

    非负对比学习(NCL)是对非负矩阵分解(NMF)的重新演绎，通过对特征施加非负约束来获得可解释的特征，保留了NMF的可解释属性，从而得到比标准对比学习(CL)更稀疏和解耦的表示

    

    深度表示在以黑盒方式转移到下游任务时表现出了良好的性能。然而，它们固有的不可解释性仍然是一个重大挑战，因为这些特征通常对人类理解而言是不透明的。在本文中，我们提出了非负对比学习（NCL），这是对非负矩阵分解（NMF）的复兴，旨在得出可解释的特征。NCL的力量在于强制将非负约束应用于特征，这让人想起NMF能够提取与样本集群紧密对齐的特征的能力。NCL不仅在数学上与NMF目标很好地对齐，而且保留了NMF的可解释属性，使得与标准对比学习（CL）相比，得到了更加稀疏和解耦的表示。从理论上，我们为NCL的可识别性和下游泛化性能提供了保证。从经验上看，我们展示了这些

    arXiv:2403.12459v1 Announce Type: cross  Abstract: Deep representations have shown promising performance when transferred to downstream tasks in a black-box manner. Yet, their inherent lack of interpretability remains a significant challenge, as these features are often opaque to human understanding. In this paper, we propose Non-negative Contrastive Learning (NCL), a renaissance of Non-negative Matrix Factorization (NMF) aimed at deriving interpretable features. The power of NCL lies in its enforcement of non-negativity constraints on features, reminiscent of NMF's capability to extract features that align closely with sample clusters. NCL not only aligns mathematically well with an NMF objective but also preserves NMF's interpretability attributes, resulting in a more sparse and disentangled representation compared to standard contrastive learning (CL). Theoretically, we establish guarantees on the identifiability and downstream generalization of NCL. Empirically, we show that these 
    
[^2]: 使用训练标签进行填充和通过标签填充进行分类

    Imputation using training labels and classification via label imputation. (arXiv:2311.16877v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.16877](http://arxiv.org/abs/2311.16877)

    本论文提出一种在填充缺失数据时将标签与输入堆叠的方法，能够显著提高填充效果，并同时填充标签和输入。该方法适用于各种类型的数据，且在实验证明具有有希望的准确性结果。

    

    在实际应用中，缺失数据是一个常见的问题。已经开发了各种填充方法来处理缺失数据。然而，尽管训练数据通常都有标签，但常见的填充方法通常只依赖于输入而忽略标签。在这项工作中，我们阐述了将标签堆叠到输入中可以显着提高输入的填充效果。此外，我们提出了一种分类策略，该策略将预测的测试标签初始化为缺失值，并将标签与输入堆叠在一起进行填充。这样可以同时填充标签和输入。而且，该技术能够处理具有缺失标签的训练数据，无需任何先前的填充，并且适用于连续型、分类型或混合型数据。实验证明在准确性方面取得了有希望的结果。

    Missing data is a common problem in practical settings. Various imputation methods have been developed to deal with missing data. However, even though the label is usually available in the training data, the common practice of imputation usually only relies on the input and ignores the label. In this work, we illustrate how stacking the label into the input can significantly improve the imputation of the input. In addition, we propose a classification strategy that initializes the predicted test label with missing values and stacks the label with the input for imputation. This allows imputing the label and the input at the same time. Also, the technique is capable of handling data training with missing labels without any prior imputation and is applicable to continuous, categorical, or mixed-type data. Experiments show promising results in terms of accuracy.
    
[^3]: 高斯混合物可以通过多项式数量的样本进行差分隐私学习

    Mixtures of Gaussians are Privately Learnable with a Polynomial Number of Samples. (arXiv:2309.03847v1 [stat.ML])

    [http://arxiv.org/abs/2309.03847](http://arxiv.org/abs/2309.03847)

    通过多项式数量的样本和差分隐私约束，我们提出了一个可以估计高斯混合物的方法，并证明了这个方法的有效性，而无需对GMMs做任何结构性假设。

    

    我们研究了在差分隐私(DP)约束下估计高斯混合物的问题。我们的主要结果是，使用$\tilde{O}(k^2 d^4 \log(1/\delta) / \alpha^2 \varepsilon)$个样本即可在满足$(\varepsilon, \delta)$-DP的条件下估计$k$个高斯混合物，使其达到总变差距离$\alpha$。这是该问题的第一个有限样本复杂性上限，而无需对GMMs做任何结构性假设。为了解决这个问题，我们构建了一个新的框架，该框架对于其他任务可能也有用。在高层次上，我们展示了如果一个分布类（比如高斯分布）是（1）可列表译码的并且（2）在总变差距离方面具有“局部小”覆盖[ BKSW19]，则其混合物类是私密可学习的。证明绕过了一个已知障碍，表明与高斯分布不同，GMMs不具有局部小的覆盖[AAL21]。

    We study the problem of estimating mixtures of Gaussians under the constraint of differential privacy (DP). Our main result is that $\tilde{O}(k^2 d^4 \log(1/\delta) / \alpha^2 \varepsilon)$ samples are sufficient to estimate a mixture of $k$ Gaussians up to total variation distance $\alpha$ while satisfying $(\varepsilon, \delta)$-DP. This is the first finite sample complexity upper bound for the problem that does not make any structural assumptions on the GMMs.  To solve the problem, we devise a new framework which may be useful for other tasks. On a high level, we show that if a class of distributions (such as Gaussians) is (1) list decodable and (2) admits a "locally small'' cover [BKSW19] with respect to total variation distance, then the class of its mixtures is privately learnable. The proof circumvents a known barrier indicating that, unlike Gaussians, GMMs do not admit a locally small cover [AAL21].
    
[^4]: 黑盒变分推断的线性收敛性：我们应该坚持到底吗？

    Linear Convergence of Black-Box Variational Inference: Should We Stick the Landing?. (arXiv:2307.14642v1 [stat.ML])

    [http://arxiv.org/abs/2307.14642](http://arxiv.org/abs/2307.14642)

    本文证明了带有控制变量的黑盒变分推断在完美变分族规范下以几何速度收敛，为BBVI提供了收敛性保证，同时提出了对熵梯度估计器的改进，对比了STL估计器，并给出了明确的非渐近复杂度保证。

    

    我们证明了带有控制变量的黑盒变分推断（BBVI），特别是着陆稳定（STL）估计器，在完美变分族规范下收敛于几何（传统上称为“线性”）速度。特别地，我们证明了STL估计器的梯度方差的二次界限，该界限包括了误指定的变分族。结合先前关于二次方差条件的工作，这直接暗示了在使用投影随机梯度下降的情况下BBVI的收敛性。我们还改进了现有对于正常封闭形式熵梯度估计器的分析，这使得我们能够将其与STL估计器进行比较，并为两者提供明确的非渐进复杂度保证。

    We prove that black-box variational inference (BBVI) with control variates, particularly the sticking-the-landing (STL) estimator, converges at a geometric (traditionally called "linear") rate under perfect variational family specification. In particular, we prove a quadratic bound on the gradient variance of the STL estimator, one which encompasses misspecified variational families. Combined with previous works on the quadratic variance condition, this directly implies convergence of BBVI with the use of projected stochastic gradient descent. We also improve existing analysis on the regular closed-form entropy gradient estimators, which enables comparison against the STL estimator and provides explicit non-asymptotic complexity guarantees for both.
    
[^5]: 1比特矩阵补全的主导-最小化高斯牛顿方法

    A Majorization-Minimization Gauss-Newton Method for 1-Bit Matrix Completion. (arXiv:2304.13940v1 [stat.ML])

    [http://arxiv.org/abs/2304.13940](http://arxiv.org/abs/2304.13940)

    本文提出了一种基于主导-最小化原则，通过低秩矩阵补全解决1比特矩阵补全问题的新方法，称为MMGN。通过应用高斯-牛顿方法，MMGN具有更快的速度和更准确的结果，同时还不太受到潜在矩阵尖锐度的影响。

    

    在1比特矩阵补全中，旨在从部分二进制观测值中估计潜在的低秩矩阵。我们提出了一种称为MMGN的1比特矩阵补全新方法。我们的方法基于主导-最小化（MM）原则，在我们的设置中产生一系列标准低秩矩阵补全问题。我们通过明确强制假定的低秩结构的分解方法解决这些子问题，然后应用高斯-牛顿方法。我们的数值研究和对实际数据的应用表明，MMGN输出的估计结果与现有方法相比较具有可比性且更准确、速度通常更快，并且对潜在矩阵的尖锐度不太敏感。

    In 1-bit matrix completion, the aim is to estimate an underlying low-rank matrix from a partial set of binary observations. We propose a novel method for 1-bit matrix completion called MMGN. Our method is based on the majorization-minimization (MM) principle, which yields a sequence of standard low-rank matrix completion problems in our setting. We solve each of these sub-problems by a factorization approach that explicitly enforces the assumed low-rank structure and then apply a Gauss-Newton method. Our numerical studies and application to a real-data example illustrate that MMGN outputs comparable if not more accurate estimates, is often significantly faster, and is less sensitive to the spikiness of the underlying matrix than existing methods.
    

