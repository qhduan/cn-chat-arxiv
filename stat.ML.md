# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transformers Provably Learn Feature-Position Correlations in Masked Image Modeling](https://arxiv.org/abs/2403.02233) | 本文提供了关于使用MIM自监督预训练学习transformers的首个端到端理论，揭示了transformers如何学习到在具有空间结构的数据分布上突显特征-位置相关性的本地和多样化注意模式 |
| [^2] | [Multi-objective Differentiable Neural Architecture Search](https://arxiv.org/abs/2402.18213) | 提出了一种新颖的NAS算法，可以在一个搜索运行中编码用户对性能和硬件指标之间的权衡偏好，生成精心选择的多设备架构。 |
| [^3] | [Sparse Linear Regression and Lattice Problems](https://arxiv.org/abs/2402.14645) | 本文提供了关于稀疏线性回归在所有高效算法的平均情况困难性的证据，假设格问题的最坏情况困难性。 |

# 详细

[^1]: Transformers在Masked Image Modeling中能够证明学习特征-位置相关性

    Transformers Provably Learn Feature-Position Correlations in Masked Image Modeling

    [https://arxiv.org/abs/2403.02233](https://arxiv.org/abs/2403.02233)

    本文提供了关于使用MIM自监督预训练学习transformers的首个端到端理论，揭示了transformers如何学习到在具有空间结构的数据分布上突显特征-位置相关性的本地和多样化注意模式

    

    Masked image modeling (MIM)是一种新兴的自监督视觉预训练方法，它从未屏蔽的图像中预测随机屏蔽的补丁。然而，对于基于transformers的MIM的理论理解相当有限。本文提供了有关使用MIM自监督预训练学习一层transformers的首个端到端理论。我们提出了transformers如何学习到在具有空间结构的数据分布上突显特征-位置相关性的本地和多样化注意模式的理论机制。

    arXiv:2403.02233v1 Announce Type: new  Abstract: Masked image modeling (MIM), which predicts randomly masked patches from unmasked ones, has emerged as a promising approach in self-supervised vision pretraining. However, the theoretical understanding of MIM is rather limited, especially with the foundational architecture of transformers. In this paper, to the best of our knowledge, we provide the first end-to-end theory of learning one-layer transformers with softmax attention in MIM self-supervised pretraining. On the conceptual side, we posit a theoretical mechanism of how transformers, pretrained with MIM, produce empirically observed local and diverse attention patterns on data distributions with spatial structures that highlight feature-position correlations. On the technical side, our end-to-end analysis of the training dynamics of softmax-based transformers accommodates both input and position embeddings simultaneously, which is developed based on a novel approach to track the i
    
[^2]: 多目标可微神经架构搜索

    Multi-objective Differentiable Neural Architecture Search

    [https://arxiv.org/abs/2402.18213](https://arxiv.org/abs/2402.18213)

    提出了一种新颖的NAS算法，可以在一个搜索运行中编码用户对性能和硬件指标之间的权衡偏好，生成精心选择的多设备架构。

    

    多目标优化（MOO）中的Pareto前沿轮廓剖析是具有挑战性的，尤其是在像神经网络训练这样的昂贵目标中。 相对于传统的NAS方法，我们提出了一种新颖的NAS算法，该算法在一个搜索运行中编码用户对性能和硬件指标之间的权衡偏好，并生成精心选择的多设备架构。为此，我们通过一个超网络参数化跨多个设备和多个目标的联合架构分布，超网络可以根据硬件特征和偏好向量进行条件化，实现零次搜索。

    arXiv:2402.18213v1 Announce Type: new  Abstract: Pareto front profiling in multi-objective optimization (MOO), i.e. finding a diverse set of Pareto optimal solutions, is challenging, especially with expensive objectives like neural network training. Typically, in MOO neural architecture search (NAS), we aim to balance performance and hardware metrics across devices. Prior NAS approaches simplify this task by incorporating hardware constraints into the objective function, but profiling the Pareto front necessitates a search for each constraint. In this work, we propose a novel NAS algorithm that encodes user preferences for the trade-off between performance and hardware metrics, and yields representative and diverse architectures across multiple devices in just one search run. To this end, we parameterize the joint architectural distribution across devices and multiple objectives via a hypernetwork that can be conditioned on hardware features and preference vectors, enabling zero-shot t
    
[^3]: 稀疏线性回归和格问题

    Sparse Linear Regression and Lattice Problems

    [https://arxiv.org/abs/2402.14645](https://arxiv.org/abs/2402.14645)

    本文提供了关于稀疏线性回归在所有高效算法的平均情况困难性的证据，假设格问题的最坏情况困难性。

    

    稀疏线性回归（SLR）是统计学中一个研究良好的问题，其中给定设计矩阵 $X\in\mathbb{R}^{m\times n}$ 和响应向量 $y=X\theta^*+w$，其中 $\theta^*$ 是 $k$-稀疏向量（即，$\|\theta^*\|_0\leq k$），$w$ 是小的、任意的噪声，目标是找到一个 $k$-稀疏的 $\widehat{\theta} \in \mathbb{R}^n$，使得均方预测误差 $\frac{1}{m}\|X\widehat{\theta}-X\theta^*\|^2_2$ 最小化。虽然 $\ell_1$-松弛方法如基 Pursuit、Lasso 和 Dantzig 选择器在设计矩阵条件良好时解决了 SLR，但没有已知通用算法，也没有任何关于在所有高效算法的平均情况设置中的困难性的正式证据。

    arXiv:2402.14645v1 Announce Type: new  Abstract: Sparse linear regression (SLR) is a well-studied problem in statistics where one is given a design matrix $X\in\mathbb{R}^{m\times n}$ and a response vector $y=X\theta^*+w$ for a $k$-sparse vector $\theta^*$ (that is, $\|\theta^*\|_0\leq k$) and small, arbitrary noise $w$, and the goal is to find a $k$-sparse $\widehat{\theta} \in \mathbb{R}^n$ that minimizes the mean squared prediction error $\frac{1}{m}\|X\widehat{\theta}-X\theta^*\|^2_2$. While $\ell_1$-relaxation methods such as basis pursuit, Lasso, and the Dantzig selector solve SLR when the design matrix is well-conditioned, no general algorithm is known, nor is there any formal evidence of hardness in an average-case setting with respect to all efficient algorithms.   We give evidence of average-case hardness of SLR w.r.t. all efficient algorithms assuming the worst-case hardness of lattice problems. Specifically, we give an instance-by-instance reduction from a variant of the bo
    

