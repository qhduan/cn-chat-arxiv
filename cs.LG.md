# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scaling on-chip photonic neural processors using arbitrarily programmable wave propagation](https://arxiv.org/abs/2402.17750) | 提出并演示了一种利用任意可编程波传播的2D波导器件，通过组合光电导增益和电光效应实现对板块的折射率进行大规模并行调制 |
| [^2] | [Uncertainty Quantification in Anomaly Detection with Cross-Conformal $p$-Values](https://arxiv.org/abs/2402.16388) | 针对异常检测系统中不确定性量化的需求，提出了一种新颖的框架，称为交叉一致异常检测，通过校准模型的不确定性提供统计保证。 |
| [^3] | [On Rate-Optimal Partitioning Classification from Observable and from Privatised Data](https://arxiv.org/abs/2312.14889) | 研究了在放宽条件下的分区分类方法的收敛速率，提出了绝对连续分量的新特性，计算了分类错误概率的精确收敛率 |

# 详细

[^1]: 利用任意可编程波传播来扩展芯片光子神经处理器的规模

    Scaling on-chip photonic neural processors using arbitrarily programmable wave propagation

    [https://arxiv.org/abs/2402.17750](https://arxiv.org/abs/2402.17750)

    提出并演示了一种利用任意可编程波传播的2D波导器件，通过组合光电导增益和电光效应实现对板块的折射率进行大规模并行调制

    

    用于神经网络的芯片光子处理器在速度和能量效率方面具有潜在优势，但尚未达到能够胜过电子处理器的规模。设计芯片光子学的主导范式是制作由相对笨重的离散元件构成的网络，这些元件通过一维波导连接。一个更紧凑的替代方案是避免明确定义任何元件，而是通过在两个维度中自由传播的波直接塑造光子处理器的连续衬底来执行计算。我们提出并展示了一种可以快速重新编程空间折射率$n(x,z)$的设备，从而实现对设备中波传播的任意控制。我们的设备，一维可编程波导，将光电导增益与电光效应结合，实现了对板块的折射率的并行调制。

    arXiv:2402.17750v1 Announce Type: cross  Abstract: On-chip photonic processors for neural networks have potential benefits in both speed and energy efficiency but have not yet reached the scale at which they can outperform electronic processors. The dominant paradigm for designing on-chip photonics is to make networks of relatively bulky discrete components connected by one-dimensional waveguides. A far more compact alternative is to avoid explicitly defining any components and instead sculpt the continuous substrate of the photonic processor to directly perform the computation using waves freely propagating in two dimensions. We propose and demonstrate a device whose refractive index as a function of space, $n(x,z)$, can be rapidly reprogrammed, allowing arbitrary control over the wave propagation in the device. Our device, a 2D-programmable waveguide, combines photoconductive gain with the electro-optic effect to achieve massively parallel modulation of the refractive index of a slab
    
[^2]: 具有交叉一致$p$-值的异常检测中的不确定性量化

    Uncertainty Quantification in Anomaly Detection with Cross-Conformal $p$-Values

    [https://arxiv.org/abs/2402.16388](https://arxiv.org/abs/2402.16388)

    针对异常检测系统中不确定性量化的需求，提出了一种新颖的框架，称为交叉一致异常检测，通过校准模型的不确定性提供统计保证。

    

    随着可靠、可信和可解释机器学习的重要性日益增加，对异常检测系统进行不确定性量化的要求变得愈发重要。在这种情况下，有效控制类型I错误率($\alpha$)而又不损害系统的统计功率($1-\beta$)可以建立信任，并减少与假发现相关的成本，特别是当后续程序昂贵时。利用符合预测原则的方法有望通过校准模型的不确定性为异常检测提供相应的统计保证。该工作引入了一个新颖的异常检测框架，称为交叉一致异常检测，建立在为预测任务设计的著名交叉一致方法之上。通过这种方法，他填补了在归纳一致异常检测环境中扩展先前研究的自然研究空白

    arXiv:2402.16388v1 Announce Type: cross  Abstract: Given the growing significance of reliable, trustworthy, and explainable machine learning, the requirement of uncertainty quantification for anomaly detection systems has become increasingly important. In this context, effectively controlling Type I error rates ($\alpha$) without compromising the statistical power ($1-\beta$) of these systems can build trust and reduce costs related to false discoveries, particularly when follow-up procedures are expensive. Leveraging the principles of conformal prediction emerges as a promising approach for providing respective statistical guarantees by calibrating a model's uncertainty. This work introduces a novel framework for anomaly detection, termed cross-conformal anomaly detection, building upon well-known cross-conformal methods designed for prediction tasks. With that, it addresses a natural research gap by extending previous works in the context of inductive conformal anomaly detection, rel
    
[^3]: 论从可观测和私密数据中实现速率最优分区分类

    On Rate-Optimal Partitioning Classification from Observable and from Privatised Data

    [https://arxiv.org/abs/2312.14889](https://arxiv.org/abs/2312.14889)

    研究了在放宽条件下的分区分类方法的收敛速率，提出了绝对连续分量的新特性，计算了分类错误概率的精确收敛率

    

    在这篇论文中，我们重新审视了分区分类的经典方法，并研究了在放宽条件下的收敛速率，包括可观测（非私密）和私密数据。我们假设特征向量$X$取值于$\mathbb{R}^d$，其标签为$Y$。之前关于分区分类器的结果基于强密度假设，这种假设限制较大，我们通过简单的例子加以证明。我们假设$X$的分布是绝对连续分布和离散分布的混合体，其中绝对连续分量集中于一个$d_a$维子空间。在这里，我们在更宽松的条件下研究了这个问题：除了标准的Lipschitz和边际条件外，我们还引入了绝对连续分量的一个新特性，通过该特性计算了分类错误概率的精确收敛率，对于...

    arXiv:2312.14889v2 Announce Type: replace-cross  Abstract: In this paper we revisit the classical method of partitioning classification and study its convergence rate under relaxed conditions, both for observable (non-privatised) and for privatised data. Let the feature vector $X$ take values in $\mathbb{R}^d$ and denote its label by $Y$. Previous results on the partitioning classifier worked with the strong density assumption, which is restrictive, as we demonstrate through simple examples. We assume that the distribution of $X$ is a mixture of an absolutely continuous and a discrete distribution, such that the absolutely continuous component is concentrated to a $d_a$ dimensional subspace. Here, we study the problem under much milder assumptions: in addition to the standard Lipschitz and margin conditions, a novel characteristic of the absolutely continuous component is introduced, by which the exact convergence rate of the classification error probability is calculated, both for the
    

