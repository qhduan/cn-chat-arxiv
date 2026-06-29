# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Recovering Governing Equations from Solution Data: Identifiability Bounds for Linear and Nonlinear ODEs](https://arxiv.org/abs/2606.27285) | 本文通过引入豪斯多夫距离作为度量，首次为从解数据中恢复线性和非线性常微分方程建立了定量可辨识性界，填补了理论空白。 |
| [^2] | [Topology-Informed Neural Networks for Flood Detection in Optical and Synthetic Aperture Radar Imagery](https://arxiv.org/abs/2606.26204) | 本文提出了一种结合拓扑信息的神经网络方法，用于在光学和合成孔径雷达影像中更准确、可解释地检测洪水，克服了云层遮挡和现有模型不透明的局限。 |
| [^3] | [Dual-Prototype Disentanglement: A Context-Aware Enhancement Framework for Time Series Forecasting](https://arxiv.org/abs/2601.16632) | 提出一种模型无关的辅助框架DPAD，通过动态双原型库解耦常见与罕见时间模式，使预测模型获得上下文感知的自适应能力。 |
| [^4] | [Monte Carlo with kernel-based Gibbs measures: Guarantees for probabilistic herding](https://arxiv.org/abs/2402.11736) | 该论文研究了一种联合概率分布，其支持趋于最小化最坏情况误差，证明了它在最坏情况积分误差集中不等式上优于i.i.d.蒙特卡罗。 |

# 详细

[^1]: 从解数据中恢复控制方程：线性和非线性常微分方程的可辨识性界

    Recovering Governing Equations from Solution Data: Identifiability Bounds for Linear and Nonlinear ODEs

    [https://arxiv.org/abs/2606.27285](https://arxiv.org/abs/2606.27285)

    本文通过引入豪斯多夫距离作为度量，首次为从解数据中恢复线性和非线性常微分方程建立了定量可辨识性界，填补了理论空白。

    

    从观测到的解数据中学习控制方程是科学机器学习中的一个基本挑战（参见文献\cite{bruntonDiscoveringGoverningEquations2016,kovachkiNeuralOperatorLearning2023,longPDENetLearningPDEs2018,rudyDatadrivenDiscoveryPartial2017,raonicConvolutionalNeuralOperators2023}），然而，关于在何种理论条件下可以从多个解观测中唯一且稳定地辨识出真实常微分方程（ODE）的研究仍基本空白，并且文献中缺乏对此类学习任务样本复杂性的定量分析。为填补这一空白，我们引入解集上的豪斯多夫距离作为比较微分方程的自然度量，因为它捕捉了所有允许初始条件下两个方程之间的最坏情况分离，从而编码了辨识问题的极小极大结构。我们针对一大类控制常微分方程建立了可辨识性界。

    arXiv:2606.27285v1 Announce Type: new  Abstract: Learning governing equations from observed solution data is a fundamental challenge in scientific machine learning \cite{bruntonDiscoveringGoverningEquations2016,kovachkiNeuralOperatorLearning2023,longPDENetLearningPDEs2018,rudyDatadrivenDiscoveryPartial2017,raonicConvolutionalNeuralOperators2023}, yet the theoretical conditions under which a ground-truth ODE can be uniquely and stably identified from multiple solution observations remain largely undeveloped, and no quantitative analysis of the sample complexity of such learning tasks exists in the literature. To address this gap, we introduce the Hausdorff distance on solution sets as the natural metric for comparing differential equations, since it captures the worst-case separation between two equations over all admissible initial conditions and thus encodes the minimax structure of the identification problem. We establish identifiability bounds for governing ODEs across a wide class 
    
[^2]: 面向光学与合成孔径雷达影像洪水检测的拓扑信息神经网络

    Topology-Informed Neural Networks for Flood Detection in Optical and Synthetic Aperture Radar Imagery

    [https://arxiv.org/abs/2606.26204](https://arxiv.org/abs/2606.26204)

    本文提出了一种结合拓扑信息的神经网络方法，用于在光学和合成孔径雷达影像中更准确、可解释地检测洪水，克服了云层遮挡和现有模型不透明的局限。

    

    arXiv:2606.26204v1 发布类型：新  摘要：洪水频繁影响全球各地。快速准确的洪水检测对于应急响应和及时减少人员及经济损失至关重要。卫星数据可用性的不断扩大以及人工智能的进步增强了对环境灾害的监测能力，但许多洪水事件仍难以检测，因为云层会遮挡光学卫星图像。Rambour等人引入了SEN12-FLOOD数据集，并使用ResNet-50卷积神经网络骨干网络提取每幅图像的特征，然后将这些特征输入到门控循环单元网络中，证明与单图像基线相比，时间信息可以显著提高准确性。最近，Chamatidis等人表明，视觉变换器可以与流行的卷积架构一起实现强大的性能。然而，这些模型通常作为不透明的黑箱运行，使得解释变得困难。

    arXiv:2606.26204v1 Announce Type: new  Abstract: Floods frequently impact regions around the world. Rapid and accurate flood detection is crucial for emergency response and timely mitigation of human and economic loss. The expanding availability of satellite data and advances in artificial intelligence have enhanced monitoring of environmental hazards, but many flood events remain challenging to detect because cloud cover obscures optical satellite imagery. Rambour et al. introduced the SEN12-FLOOD dataset and extracted per-image features using a ResNet-50 convolutional neural network backbone, then fed these features into a gated recurrent unit network to show that temporal information can substantially improve accuracy compared to single-image baselines. More recently, Chamatidis et al. showed that a vision transformer can achieve strong performance with popular convolutional architectures. However, these models typically function as opaque black boxes, making it difficult to interpr
    
[^3]: 双原型解耦：一种面向时间序列预测的上下文感知增强框架

    Dual-Prototype Disentanglement: A Context-Aware Enhancement Framework for Time Series Forecasting

    [https://arxiv.org/abs/2601.16632](https://arxiv.org/abs/2601.16632)

    提出一种模型无关的辅助框架DPAD，通过动态双原型库解耦常见与罕见时间模式，使预测模型获得上下文感知的自适应能力。

    

    时间序列预测在深度学习的推动下取得了显著进展。虽然主流方法通过修改架构或引入新颖的增强策略来提升预测性能，但它们往往无法动态解耦并利用时间序列中固有的复杂、交织的时间模式，从而学习到缺乏上下文感知能力的静态平均化表示。为解决这一问题，我们提出了双原型自适应解耦框架（DPAD），这是一种模型无关的辅助方法，使预测模型具备模式解耦和上下文感知自适应能力。具体来说，我们构建了一个动态双原型库（DDP），包含一个具有强时间先验的公共模式库（用于捕获主流趋势或季节模式）和一个动态记忆关键但罕见事件的罕见模式库，然后通过一个双原...

    arXiv:2601.16632v4 Announce Type: replace-cross  Abstract: Time series forecasting has witnessed significant progress with deep learning. While prevailing approaches enhance forecasting performance by modifying architectures or introducing novel enhancement strategies, they often fail to dynamically disentangle and leverage the complex, intertwined temporal patterns inherent in time series, thus resulting in the learning of static, averaged representations that lack context-aware capabilities. To address this, we propose the Dual-Prototype Adaptive Disentanglement framework (DPAD), a model-agnostic auxiliary method that equips forecasting models with the ability of pattern disentanglement and context-aware adaptation. Specifically, we construct a Dynamic Dual-Prototype bank (DDP), comprising a common pattern bank with strong temporal priors to capture prevailing trend or seasonal patterns, and a rare pattern bank dynamically memorizing critical yet infrequent events, and then an Dual-P
    
[^4]: 基于核Gibbs测度的蒙特卡罗：概率随机放牧的保证

    Monte Carlo with kernel-based Gibbs measures: Guarantees for probabilistic herding

    [https://arxiv.org/abs/2402.11736](https://arxiv.org/abs/2402.11736)

    该论文研究了一种联合概率分布，其支持趋于最小化最坏情况误差，证明了它在最坏情况积分误差集中不等式上优于i.i.d.蒙特卡罗。

    

    Kernel herding属于一类确定性的四位数法，旨在通过再生核希尔伯特空间（RKHS）上的最坏情况积分误差。尽管有很强的实验支持，但在通常情况下，即RKHS是无限维时，证明这种最坏情况误差以比标准积分节点数量的平方根更快的速率减少是困难的。在这篇理论论文中，我们研究了一个关于积分节点的联合概率分布，其支持趋于最小化与核放牧相同的最坏情况误差。我们证明它优于i.i.d.蒙特卡罗，意味着在最坏情况积分误差上具有更紧的集中不等式。尽管尚未提高速率，但这表明了研究Gibbs测度的数学工具可以帮助理解核放牧及其变体在计算上的改进程度

    arXiv:2402.11736v1 Announce Type: new  Abstract: Kernel herding belongs to a family of deterministic quadratures that seek to minimize the worst-case integration error over a reproducing kernel Hilbert space (RKHS). In spite of strong experimental support, it has revealed difficult to prove that this worst-case error decreases at a faster rate than the standard square root of the number of quadrature nodes, at least in the usual case where the RKHS is infinite-dimensional. In this theoretical paper, we study a joint probability distribution over quadrature nodes, whose support tends to minimize the same worst-case error as kernel herding. We prove that it does outperform i.i.d. Monte Carlo, in the sense of coming with a tighter concentration inequality on the worst-case integration error. While not improving the rate yet, this demonstrates that the mathematical tools of the study of Gibbs measures can help understand to what extent kernel herding and its variants improve on computation
    

