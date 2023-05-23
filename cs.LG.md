# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On First-Order Meta-Reinforcement Learning with Moreau Envelopes.](http://arxiv.org/abs/2305.12216) | 本文提出了一种Moreau包络元强化学习算法（MEMRL），通过利用Moreau包络代理正则化器，可以学习一个可以适应任务分布的元策略。 |
| [^2] | [Taming Resource Heterogeneity In Distributed ML Training With Dynamic Batching.](http://arxiv.org/abs/2305.12213) | 本文提出一种动态分批技术，用于分布式数据并行训练，可以根据每个工作节点的资源可用性和吞吐量调整小批量大小，从而平衡所有工作节点的迭代时间，减少分布式机器学习模型训练时间。 |
| [^3] | [Vocabulary for Universal Approximation: A Linguistic Perspective of Mapping Compositions.](http://arxiv.org/abs/2305.12205) | 本文探讨了通用逼近的词汇，证明了有限“词汇”存在并可用于逼近任何连续映射$f$和紧致域$\Omega$中的每个点，误差小于$\varepsilon$。 |
| [^4] | [GraVAC: Adaptive Compression for Communication-Efficient Distributed DL Training.](http://arxiv.org/abs/2305.12201) | GraVAC提出了一个动态调整压缩因子的框架，通过评估模型进展和评估与压缩相关的梯度信息损失来进行训练。GraVAC可以在不需要任何关于模型或其超参数的先前假设的情况下，达到与先前最先进的压缩方法相同或更好的翻译准确性。在CIFAR-10和ImageNet数据集上，相对于静态压缩对应物，GraVAC可以将通信减少高达87％和75％。 |
| [^5] | [Do We Need an Encoder-Decoder to Model Dynamical Systems on Networks?.](http://arxiv.org/abs/2305.12185) | 本论文揭示了使用潜在嵌入来模拟网络动力系统的不足，并提出了一种无嵌入的替代方案，通过三个正确长期行为测试的验证表明该方案的可行性。 |
| [^6] | [Model Debiasing via Gradient-based Explanation on Representation.](http://arxiv.org/abs/2305.12178) | 本文提出了一种新的公平性框架，通过梯度说明找到两个模型焦点进行去偏见处理，提高下游任务模型的预测性能。 |
| [^7] | [A Scalable Neural Network for DSIC Affine Maximizer Auction Design.](http://arxiv.org/abs/2305.12162) | 该论文提出了一种可扩展的神经网络AMenuNet来构造AMAs参数和生成候选分配，解决了现有方法在占优策略激励兼容性和可扩展性方面的限制，其在协商一致的价值和社会残余价值方面优于强基线模型。 |
| [^8] | [Model-based adaptation for sample efficient transfer in reinforcement learning control of parameter-varying systems.](http://arxiv.org/abs/2305.12158) | 本文提出一种基于模型的方法来解决强化学习中样本有效性问题，通过模型转换实现正向迁移，并用作强化学习过程的初始化以达到新的最优值。 |
| [^9] | [(Machine) Learning to Be Like Thee? For Algorithm Education, Not Training.](http://arxiv.org/abs/2305.12157) | 本文认为，机器学习算法需要接受算法教育，以改进其决策道德，而非仅仅停留在训练上。对于AI伦理决策，解决方案在于对ML进行算法教育。机器学习算法并不是与我们的人类特质分开而存在的，而是我们最内在的偏见和偏见的一种体现。 |
| [^10] | [Normalizing flow sampling with Langevin dynamics in the latent space.](http://arxiv.org/abs/2305.12149) | 本文提出了一种在潜在空间中使用 Langevin 动力学的采样方法，以克服归一化流可能面临的复杂分布问题，并能够轻松地融入任何 NF 结构中。 |
| [^11] | [Learning Horn Envelopes via Queries from Large Language Models.](http://arxiv.org/abs/2305.12143) | 该研究提出了一种从经过训练的神经网络中提取知识的方法，可以学习到最紧Horn逼近目标理论的新算法，并在预训练的语言模型中运用于揭示基于职业的性别偏见规则。 |
| [^12] | [Privacy in Multimodal Federated Human Activity Recognition.](http://arxiv.org/abs/2305.12134) | 本文研究了多模式联邦人类活动识别中的隐私问题。通过一个特定的系统，联邦学习可以提供更好的隐私保护，同时不会损失人类活动识别的准确性。 |
| [^13] | [Loss Spike in Training Neural Networks.](http://arxiv.org/abs/2305.12133) | 本文研究神经网络训练过程中损失值峰值现象的机制，并发现最大特征值的第一个特征向量的偏差主要受低频成分占据。低频成分可以被训练数据和测试数据很好地捕获，所以导致具有良好和劣质泛化能力的解决方案都可以很好地学习低频成分，但劣质泛化能力的解决方案可能会过度拟合高频成分，良好泛化能力的解决方案具有更平滑的损失函数。 |
| [^14] | [Can Public Large Language Models Help Private Cross-device Federated Learning?.](http://arxiv.org/abs/2305.12132) | 本文探讨在差分私有联邦学习中如何利用大型公共语言模型提升隐私和效用权衡，并提出一种分布匹配算法提高公共数据的训练效率和隐私性，为训练私有模型提供有效方法。 |
| [^15] | [Non-stationary Online Convex Optimization with Arbitrary Delays.](http://arxiv.org/abs/2305.12131) | 本文研究了任意时延的非稳态在线凸优化，提出了一种简单的算法DOGD，并证明它能在最坏情况下获得$O(\sqrt{dT}(P_T+1))$的动态遗憾界，同时当延迟不改变梯度到达顺序时，自动将动态遗憾减少到$O(\sqrt{S}(1+P_T))$。 |
| [^16] | [Lifting the Curse of Capacity Gap in Distilling Language Models.](http://arxiv.org/abs/2305.12129) | 本文提出了一种新的知识蒸馏方法（MiniMoE），通过增加学生的容量而不明显增加推理计算解除容量差异诅咒，并在GLUE和CoNLL上进行了实验验证。 |
| [^17] | [ACA-Net: Towards Lightweight Speaker Verification using Asymmetric Cross Attention.](http://arxiv.org/abs/2305.12121) | 本文提出了一种轻量级的全局上下文感知说话人嵌入提取器，使用不对称交叉注意力(ACA)代替时间池化，具有高效的全局特征提取和适应时间变化的优点。 |
| [^18] | [Annealing Self-Distillation Rectification Improves Adversarial Training.](http://arxiv.org/abs/2305.12118) | 本研究提出了退火自蒸馏校正(ADR)方法，其能生成软标签用作更好的指导机制，准确反映在对抗训练中攻击下的分布变化，提高模型的鲁棒性，并实现了平滑的插入性整合到其他对抗性训练技术中。 |
| [^19] | [GFDC: A Granule Fusion Density-Based Clustering with Evidential Reasoning.](http://arxiv.org/abs/2305.12114) | 提出了一种基于证据推理的颗粒聚类算法GFDC，该算法可以克服基于密度的聚类算法在衡量全局密度、确定合理的聚类中心或结构、准确地分配样本以及处理具有大密度差异的数据方面表现不佳的缺点。与此同时，GFDC 采用了三种新颖的颗粒融合策略来将颗粒组合成稳定的聚类结构，有助于检测具有任意形状的聚类。 |
| [^20] | [Meta Neural Coordination.](http://arxiv.org/abs/2305.12109) | 元神经协调是为了解决元学习中如何表示和推理其他学习算法在不同环境下操作和表现以及传统深度神经网络在预测中的不确定性问题而提出的新方法。 |
| [^21] | [Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems.](http://arxiv.org/abs/2305.12102) | 本文介绍了一种名为“特征复用”的框架，它使用单一的表示空间 能够高效有效地学习高质量的特征嵌入，同时区分不同的分类特征。通过在多个公共数据集和新数据集“Web-Available Image Search (WAIS)”上的测试，我们展示了这种方法的优于现有技术的表现。 |
| [^22] | [Stability, Generalization and Privacy: Precise Analysis for Random and NTK Features.](http://arxiv.org/abs/2305.12100) | 本论文研究了ERM训练模型对抗强大黑盒攻击的安全问题，并通过两个指标量化模型安全性：单个样本的稳定性和查询与原始数据特征的对齐。在研究中，通过研究RF和NTK回归，证明随着泛化能力的提高，隐私保护可以得到加强。 |
| [^23] | [Make Transformer Great Again for Time Series Forecasting: Channel Aligned Robust Dual Transformer.](http://arxiv.org/abs/2305.12095) | 本文提出了一种通道对齐鲁棒双Transformer模型，通过双Transformer结构和鲁棒损失函数的引入，解决了Transformer在时间序列预测中的关键缺点，显著提高了预测精度和效率。 |
| [^24] | [UP5: Unbiased Foundation Model for Fairness-aware Recommendation.](http://arxiv.org/abs/2305.12090) | 本研究提出了一种新颖的基础模型UP5，它采用反事实公平促进技术来消除大型语言模型中的偏见，从而实现面向公平性的推荐。 |
| [^25] | [Game-Theoretical Analysis of Reviewer Rewards in Peer-Review Journal Systems: Analysis and Experimental Evaluation using Deep Reinforcement Learning.](http://arxiv.org/abs/2305.12088) | 本文针对同行评审期刊中代金券奖励制度可能会导致评审人员二元决策的问题，提出了一种替代奖励系统，有效地促进了更全面的审查，实验结果表明该系统更加平衡且稳定。 |
| [^26] | [Semi-Supervised Graph Imbalanced Regression.](http://arxiv.org/abs/2305.12087) | 本文提出了一种半监督框架来解决图形回归任务中稀有标签值示例不足的问题，通过自我训练，平衡训练数据并减少模型偏差。其中，采用一种新的回归置信度测量方法为代表稀有标签的更多图形伪标注，并在数据平衡后使用伪标签的方法在潜在空间中增加图形以生成更多的稀有标签示例。 |
| [^27] | [SneakyPrompt: Evaluating Robustness of Text-to-image Generative Models' Safety Filters.](http://arxiv.org/abs/2305.12082) | 本文提出了第一个自动攻击框架SneakyPrompt，以评估最先进的文本生成图像模型中的安全过滤器的鲁棒性，该框架的关键洞见是搜索备选令牌来绕过安全过滤器。 |
| [^28] | [AnyPredict: Foundation Model for Tabular Prediction.](http://arxiv.org/abs/2305.12081) | 本文提出了一种名为 AnyPredict 的表格预测基础模型，使用数据引擎整合领域内和广泛的领域外数据集，以克服模式不匹配和预测目标异质性等方面的障碍。 |
| [^29] | [GELU Activation Function in Deep Learning: A Comprehensive Mathematical Analysis and Performance.](http://arxiv.org/abs/2305.12073) | 本文对GELU激活函数进行了全面的数学分析和广泛的实验比较，证明了它在深度学习模型中具有优越的性能和适用性。 |
| [^30] | [Dynamic Gradient Balancing for Enhanced Adversarial Attacks on Multi-Task Models.](http://arxiv.org/abs/2305.12066) | 本文提出了动态梯度平衡攻击（DGBA）框架来攻击多任务模型，并通过实验回答了多任务模型的对抗攻击的安全性、多任务攻击和对抗训练是否增强多任务模型的鲁棒性等安全研究问题。 |
| [^31] | [Efficient Multimodal Neural Networks for Trigger-less Voice Assistants.](http://arxiv.org/abs/2305.12063) | 本文提出了一种新方法，使用基于神经网络的音频-手势多模融合系统来实现无需触发器的语音助手，该方法可以更好地理解音频和手势数据之间的时间相关性，通用性强，并可以快速启动，提高资产开发流程的生产率。 |
| [^32] | [Mechanical Property Design of Bio-compatible Mg alloys using Machine-Learning Algorithms.](http://arxiv.org/abs/2305.12060) | 本研究开发了一个机器学习模型来预测生物相容性镁合金的屈服强度，通过遗传算法的优化，平衡了其力学强度和生物相容性。 |
| [^33] | [DADIN: Domain Adversarial Deep Interest Network for Cross Domain Recommender Systems.](http://arxiv.org/abs/2305.12058) | 论文提出了一种创新性的深度跨领域点击率预测模型——领域对抗深度兴趣网络（DADIN），该模型通过引入领域不可知层和特别设计的损失，创新地实现了两个领域的联合分布对齐，并采用对抗训练的方式与点击率预测损失一起进行优化，相比竞争基线算法提升明显。 |
| [^34] | [Uniform-in-Time Wasserstein Stability Bounds for (Noisy) Stochastic Gradient Descent.](http://arxiv.org/abs/2305.12056) | 本文通过建立学习理论和应用概率之间的联系，提出了一种证明随机优化算法Wasserstein稳定性界限的统一指南，并在随机梯度下降上验证了该方法的有效性，包括强凸损失和带添加噪声的非凸损失。 |
| [^35] | [SIDAR: Synthetic Image Dataset for Alignment & Restoration.](http://arxiv.org/abs/2305.12036) | SIDAR是一种通过使用3D渲染生成多视角、多光照、多阴影、多遮挡等真实场景的方式来解决图像对齐和图像修复问题的数据增强方法。 |
| [^36] | [The Waymo Open Sim Agents Challenge.](http://arxiv.org/abs/2305.12032) | Waymo开放模拟代理挑战赛提出使用真实、互动的智能体仿真以促进自动驾驶行为模型的评估和训练，是该领域的首个公开挑战赛，旨在推动逼真模拟器的设计。 |
| [^37] | [Learning Continually on a Sequence of Graphs -- The Dynamical System Way.](http://arxiv.org/abs/2305.12030) | 该论文研究的是如何进行图数据的持续学习，并面临着由于数据基础特性所导致的理论与方法上的挑战。 |
| [^38] | [MultiTurnCleanup: A Benchmark for Multi-Turn Spoken Conversational Transcript Cleanup.](http://arxiv.org/abs/2305.12029) | 本研究提出了MultiTurnCleanup任务，收集了新的数据集MultiTurnCleanup1，针对口语会话转录中的不连续现象进行探讨并提供了两个可用于未来研究的基准测试模型。 |
| [^39] | [Energy-efficient memcapacitive physical reservoir computing system for temporal data processing.](http://arxiv.org/abs/2305.12025) | 本文研究了一种基于能效随态电容器的物理储层计算系统，解决了时间数据的分类任务和分析时间序列数据的问题。在实验中，系统能够实现较高的准确率和较小的均方误差。 |
| [^40] | [Chemellia: An Ecosystem for Atomistic Scientific Machine Learning.](http://arxiv.org/abs/2305.12010) | Chemellia是一个开源原子层机器学习框架，库设计关注点分离、互操作性和透明度。其重要贡献是实现一种用于材料属性预测的晶体图卷积神经网络。 |
| [^41] | [Advising OpenMP Parallelization via a Graph-Based Approach with Transformers.](http://arxiv.org/abs/2305.11999) | 本文提出了一种名为OMPify的新方法，该方法基于Transformer模型，通过对串行代码的分析，自动检测和预测并行代码中的OpenMP编译指示符和共享内存属性。 |
| [^42] | [Robust Counterfactual Explanations for Neural Networks With Probabilistic Guarantees.](http://arxiv.org/abs/2305.11997) | 本文提出了一种可靠的神经网络反事实解释方法，该方法可以针对自然发生的模型变化提供高概率的鲁棒性。 |
| [^43] | [Survey on software ISP methods based on Deep Learning.](http://arxiv.org/abs/2305.11994) | 本文综述了基于深度学习的软件图像信号处理方法，包括去马赛克、降噪和增强等多个过程，研究并分析了最新的几项研究，并对方法进行了比较和改进点的探讨。 |
| [^44] | [Productive Crop Field Detection: A New Dataset and Deep Learning Benchmark Results.](http://arxiv.org/abs/2305.11990) | 本研究提出了一个高质量的数据集，使用半监督和最先进的深度学习方法自动检测高产农田，获得了很高的准确性，有望为农民提供帮助. |
| [^45] | [OL-Transformer: A Fast and Universal Surrogate Simulator for Optical Multilayer Thin Film Structures.](http://arxiv.org/abs/2305.11984) | 该论文提出了OL-Transformer用于光学多层薄膜结构，可以预测多达$10^{25}$种不同多层结构的精确反射和透射光谱，同时具有快速的计算速度。 |
| [^46] | [Sequential Memory with Temporal Predictive Coding.](http://arxiv.org/abs/2305.11982) | 该论文提出了一种基于PC的新型时序记忆模型，称为时间预测编码（tPC），可以通过生物可行的神经实现准确地记忆和检索连续输入。其中tPC可以被看作是一种经典异向性霍普菲尔德网络（AHN），具有更稳定的性能，并且可以编码上下文相关信息，区分在序列中出现的重复元素。 |
| [^47] | [AutoCoreset: An Automatic Practical Coreset Construction Framework.](http://arxiv.org/abs/2305.11980) | AutoCoreset是一个自动构建高质量Coreset的通用且实用的框架，用户只需提供数据和成本函数即可，无需其他计算，可用于任何数据和成本函数。 |
| [^48] | [Unsupervised Change Point Detection for heterogeneous sensor signals.](http://arxiv.org/abs/2305.11976) | 本文研究了无监督变点检测技术，该技术灵活适用于各种数据源，无需大量训练数据和重新校准模型。 |
| [^49] | [Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization.](http://arxiv.org/abs/2305.11965) | 本文提出了一种具有个性化温度的对比损失用于自监督学习，根据数据分布自动调整温度以使得训练更加有效。 |
| [^50] | [Towards understanding neural collapse in supervised contrastive learning with the information bottleneck method.](http://arxiv.org/abs/2305.11957) | 本文将神经网络崩溃建模为信息瓶颈问题，证明神经网络崩溃导致良好的泛化，特别是当它接近分类问题的最优信息瓶颈解时。 |
| [^51] | [OPTWIN: Drift identification with optimal sub-windows.](http://arxiv.org/abs/2305.11942) | 本文提出了OPTWIN概念漂移检测器，使用滑动子窗口方法检测概念漂移，获得了更高的准确性和更低的假阳性率。 |
| [^52] | [Inductive CaloFlow.](http://arxiv.org/abs/2305.11934) | iCaloFlow是一个基于归纳流的快速探测器模拟框架，可以以高达以往10-100倍的分辨率进行快速、高保真度模拟。 |
| [^53] | [PyTorch Hyperparameter Tuning -- A Tutorial for spotPython.](http://arxiv.org/abs/2305.11930) | 本文介绍了如何将spotPython超参数调谐器集成到PyTorch训练工作流中，以提高机器或深度学习模型的性能，以CIFAR10图像分类器为例。 |
| [^54] | [Energy-frugal and Interpretable AI Hardware Design using Learning Automata.](http://arxiv.org/abs/2305.11928) | 本论文通过使用学习自动机实现了节能的AI硬件设计，同时保持了模型的解释性和准确性。 |
| [^55] | [Where does a computer vision model make mistakes? Using interactive visualizations to find where and how CV models can improve.](http://arxiv.org/abs/2305.11927) | 研究使用交互式可视化工具在创建计算机视觉分类和检测模型时帮助用户识别和改进模型上的问题，有效减少设计师的工作量。 |
| [^56] | [MParrotTTS: Multilingual Multi-speaker Text to Speech Synthesis in Low Resource Setting.](http://arxiv.org/abs/2305.11926) | MParrotTTS是一个统一的多语言、多说话人文本转语音合成模型，以自监督语音表示为基础；它可以在低资源环境中仅使用少量有监督数据就适应于新语言，并在不需要平行或双语语料的情况下传递说话人特定的语音特征。 |
| [^57] | [An Approach to Multiple Comparison Benchmark Evaluations that is Stable Under Manipulation of the Comparate Set.](http://arxiv.org/abs/2305.11921) | 本文提出了一种新的基准比较结果展示方法——多元比较矩阵（MCM），使得比较集合稳定无误差，可避免常用方法存在的无意和有意的操纵空间，并且其采用Python实现，已在公开提供。 |
| [^58] | [Interpretable neural architecture search and transfer learning for understanding sequence dependent enzymatic reactions.](http://arxiv.org/abs/2305.11917) | Elektrum是一个深度学习框架，使用可解释的神经网络模型预测酶反应，利用有限但洁净的体外数据和噪声但丰富的体内数据。Elektrum可以通过迁移学习揭示酶活性的关键序列相关决定因素，并发现潜在的治疗干预靶点。 |
| [^59] | [PINNs error estimates for nonlinear equations in $\mathbb{R}$-smooth Banach spaces.](http://arxiv.org/abs/2305.11915) | 本文研究了在$\mathbb{R}$-光滑Banach空间中支持PINNs误差估计的非线性方程，提出了一种可用于限制残差的Bramble-Hilbert引理。 |
| [^60] | [Machine learning for phase-resolved reconstruction of nonlinear ocean wave surface elevations from sparse remote sensing data.](http://arxiv.org/abs/2305.11913) | 本文提出了一种基于神经网络的方法，利用高度现实的合成训练数据对稀疏雷达数据进行相位相关的波浪表面重建。 |
| [^61] | [Machine Learning and VIIRS Satellite Retrievals for Skillful Fuel Moisture Content Monitoring in Wildfire Management.](http://arxiv.org/abs/2305.11910) | 本研究利用机器学习模型结合国家水资源模型和数值天气预测模型，以及卫星检索来预测美国连续本土上的死亡燃料湿度检索，超过了既有的每日和每小时气候统计方法。VIIRS检索对预测FMC有重要贡献。 |
| [^62] | [Sequential Best-Arm Identification with Application to Brain-Computer Interface.](http://arxiv.org/abs/2305.11908) | 本论文提出了一种序列最优臂识别方法，应用于脑-机接口中的拼写系统。利用预训练的大型语言模型，可以更快地进行学习并提高信息传输速率。 |
| [^63] | [Properties of the ENCE and other MAD-based calibration metrics.](http://arxiv.org/abs/2305.11905) | 本文讨论ENEC和基于z分数（ZVE）方差的校准误差；指出在校准良好或几乎校准的数据集上，误差与分组数量的平方根成比例关系，提出一种解决方案以推断ENCE和ZVE的值，并提供统计校准测试。 |
| [^64] | [Long-lead forecasts of wintertime air stagnation index in southern China using oceanic memory effects.](http://arxiv.org/abs/2305.11901) | 该研究基于海洋记忆效应开发了一个LSTM模型，结合过去的ASI和尼娜指数可以实现更好的ASI预测，为提前制定空气质量管理计划提供了帮助。 |
| [^65] | [An Automated Power Conservation System (APCS) using Particle Photon and Smartphone.](http://arxiv.org/abs/2305.11889) | 本文介绍了一种使用Particle Photon和智能手机的自动化节能系统（APCS）。该系统使用IR传感器检测教室内人员的存在，并自动控制灯和风扇的开关，节省电力消耗和节约宝贵自然资源。通过智能手机应用程序对系统进行控制和监控，易于实施和维护。 |
| [^66] | [Enhancing Short-Term Wind Speed Forecasting using Graph Attention and Frequency-Enhanced Mechanisms.](http://arxiv.org/abs/2305.11526) | 本文提出了一种基于图注意力和频率增强机制的风速预测模型GFST-WSF，能够有效提高短期风速预测的准确性。 |
| [^67] | [Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling.](http://arxiv.org/abs/2305.10769) | 本文提出了一种名为“追赶蒸馏”的方法，通过调整传统采样算法，让速度估计模型的当前时刻输出与其先前时刻输出和地面真实标签对齐，从而实现只需一次训练便能加速采样的效果。 |
| [^68] | [Discounted Thompson Sampling for Non-Stationary Bandit Problems.](http://arxiv.org/abs/2305.10718) | 该论文提出了一种针对非稳态多臂赌博机问题的折扣汤普森抽样算法（DS-TS），可以解决突然性变化和平滑性变化的问题，并且在两种情况下具有近乎最优的遗憾上限。 |
| [^69] | [Tensor Products and Hyperdimensional Computing.](http://arxiv.org/abs/2305.10572) | 本文探索了张量积在超维计算中的数学关系，将其确定为中心表示，并发现它是最通用、最具表现力和最压缩的表示，同时具有无误差解绑和检测的能力。 |
| [^70] | [Deep Multiple Instance Learning with Distance-Aware Self-Attention.](http://arxiv.org/abs/2305.10552) | 本文提出具有距离感知自注意力的深度多示例学习模型，该模型能根据补丁之间的空间关系动态调整权重，从而在多个基准数据集上提高了分类性能。 |
| [^71] | [Appliance Detection Using Very Low-Frequency Smart Meter Time Series.](http://arxiv.org/abs/2305.10352) | 本文对时间序列分类器在极低频智能电表数据中检测不同家电的存在/缺失进行了深入评估和比较，结果表明...... |
| [^72] | [Rethinking Data Augmentation for Tabular Data in Deep Learning.](http://arxiv.org/abs/2305.10308) | 本研究提出了一种新的表格数据增强方法“随机连续嵌入”（Random Continuous Embedding，RCE），能够提高 Transformer-based 预训练模型的自监督学习性能，大幅优于现有方法，并使得自监督学习模型能够在监督表格学习中优于树形方法。 |
| [^73] | [Executive Voiced Laughter and Social Approval: An Explorative Machine Learning Study.](http://arxiv.org/abs/2305.09485) | 本文探究了行政沟通中的发声笑声对于社会认可的积极影响，特别是当发生双向笑声时。结果表明，这种影响随着组织业绩的下降而增加。 |
| [^74] | [Protein Complex Invariant Embedding with Cross-Gate MLP is A One-Shot Antibody Designer.](http://arxiv.org/abs/2305.09480) | 本文提出了一种深度生成模型，可以一次性地共同设计抗体CDR的1D序列和3D结构，解决几何建模和低效推断的问题。 |
| [^75] | [Seeing is Believing: Brain-Inspired Modular Training for Mechanistic Interpretability.](http://arxiv.org/abs/2305.08746) | BIMT方法使得神经网络更加模块化和可诠释，并且能够直接展示模块化结构，为许多简单任务提供了有用的信息，并可以补充当前的机理解释策略。 |
| [^76] | [Balancing Privacy and Utility of Spatio-Temporal Data for Taxi-Demand Prediction.](http://arxiv.org/abs/2305.08107) | 本文提出了使用联合学习进行出租车需求预测的方法，该方法允许多个参与方训练机器学习模型并保持数据私密和安全。文章对于类别不平衡、数据稀缺和模型泛化等技术挑战提出了解决方案，最终在实际数据集上展示了具有竞争力的表现。 |
| [^77] | [CodeT5+: Open Code Large Language Models for Code Understanding and Generation.](http://arxiv.org/abs/2305.07922) | CodeT5+是一组灵活组合的编码器-解码器LLM族，用于代码，混合了多种不同的预训练目标，包括代码生成、自然语言处理和程序合成，可以适应多种不同的下游代码任务，并且在实验中比现有代码-specific LLMs实现了最先进的性能。 |
| [^78] | [Understanding Model Averaging in Federated Learning on Heterogeneous Data.](http://arxiv.org/abs/2305.07845) | 本文研究了异构数据联邦学习中的模型平均技术，通过可视化损失/错误景观揭示了客户端模型环绕全局模型在一个共同的盆地内，并且发现全局模型在早期训练后的误差主要来自客户端数据集和全局数据集之间非重叠的数据及全局模型与客户端模型之间的最大距离两个因素。 |
| [^79] | [Online Learning Under A Separable Stochastic Approximation Framework.](http://arxiv.org/abs/2305.07484) | 本篇论文提出了一种新的在线学习算法，通过分离随机逼近框架，使用递归最小二乘算法和随机梯度下降算法分别更新模型的线性和非线性参数。此算法在多个数据集上表现出高效和有效性。 |
| [^80] | [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.](http://arxiv.org/abs/2305.07185) | MEGABYTE是一种基于多尺度Transformer的解码器架构，能够对超过一百万字节的序列进行端到端的可微建模，在训练和生成过程中提高了性能并降低了成本，同时证明了在大规模上下文无需标记的自回归序列建模的可行性。 |
| [^81] | [Deep Visual-Genetic Biometrics for Taxonomic Classification of Rare Species.](http://arxiv.org/abs/2305.06695) | 本文提出了一种利用对齐的视觉-遗传推理空间来提高少量图像数据珍稀物种分类的方法，该方法通过深度嵌入模型实现对齐，适用于提高稀有物种的长尾识别，并且可以显著有益于仅基于视觉的稀有物种识别。 |
| [^82] | [ST-GIN: An Uncertainty Quantification Approach in Traffic Data Imputation with Spatio-temporal Graph Attention and Bidirectional Recurrent United Neural Networks.](http://arxiv.org/abs/2305.06480) | 本研究提出了一种创新的交通数据插值方法，利用图注意力和双向神经网络捕捉时空相关性，实验结果表明在处理缺失值方面优于其他基准技术。 |
| [^83] | [Semi-Asynchronous Federated Edge Learning Mechanism via Over-the-air Computation.](http://arxiv.org/abs/2305.04066) | 本文提出了一种半异步聚合FEEL机制PAOTA，以改善数据和设备存在显著异质性的情况下FEEL系统的训练效率，通过调整边缘设备的上行传输功率来最小化FEEL全局模型的收敛上界。实验结果表明，所提出的机制在达到相同的目标精度下，训练速度显著快于具有空中计算方案的传统同步FEEL机制。 |
| [^84] | [GPT for Semi-Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering.](http://arxiv.org/abs/2305.03403) | 介绍了一种名为CAAFE的上下文感知自动特征工程方法，它利用大型语言模型根据数据集描述生成更多具有语义意义的特征，能够提高大多数数据集的性能，平均ROC AUC表现提高至0.822。 |
| [^85] | [Sensitive Data Detection with High-Throughput Machine Learning Models in Electrical Health Records.](http://arxiv.org/abs/2305.03169) | 该论文使用机器学习算法来识别结构化数据中的敏感变量，以便便于去身份化过程。该算法可以解决不同数据集PHI字段异质性的问题。 |
| [^86] | [Class-Distribution-Aware Pseudo Labeling for Semi-Supervised Multi-Label Learning.](http://arxiv.org/abs/2305.02795) | 本论文提出了一种面向半监督多标签学习的类别分布感知伪标记方法，能够在控制伪标签数目的情况下，更准确地逼近真实分布，从而实现更好的多标签分类性能。 |
| [^87] | [Leveraging Language Representation for Material Recommendation, Ranking, and Exploration.](http://arxiv.org/abs/2305.01101) | 本文提出了一种新型材料发现框架，利用材料科学特定语言模型的自然语言嵌入作为材料的组成和结构特征进行表示，并且联合采用了表示相似性召回候选材料和基于多任务学习对候选材料进行目标属性排名的方案。通过这种方法，可以更好地探索广阔的材料搜索空间，并确定高性能候选材料。 |
| [^88] | [Dynamic Transfer Learning across Graphs.](http://arxiv.org/abs/2305.00664) | 该论文提出了一个新的问题：在动态图形环境下如何有效地进行跨图迁移学习，主要解决了领域演化对泛化性能的影响。 |
| [^89] | [Are Emergent Abilities of Large Language Models a Mirage?.](http://arxiv.org/abs/2304.15004) | 研究指出大型语言模型所谓的新兴技能是研究者分析的产物，不是模型行为的基本变化。研究还展示了度量标准选择和可能研究人员的偏见，可能导致这种新兴技能的出现。 |
| [^90] | [Functional Diffusion Maps.](http://arxiv.org/abs/2304.14378) | 本研究关注一种非线性流形学习方法：扩散映射。本文阐述如何将这种方法应用于功能数据，并将其与功能主成分分析进行比较。 |
| [^91] | [Interpretable Neural-Symbolic Concept Reasoning.](http://arxiv.org/abs/2304.14068) | 本文提出了第一个基于概念嵌入的可解释概念模型DCR，能够在多个数据集上实现接近最先进的准确性，相对于最先进的可解释概念模型提高了高达+25％，并产生能够解释其预测的人类可理解规则和真值度，适应性强。 |
| [^92] | [Quantum Natural Policy Gradients: Towards Sample-Efficient Reinforcement Learning.](http://arxiv.org/abs/2304.13571) | 本文提出了量子自然策略梯度(QNPG)算法，利用了量子费舍尔信息矩阵的高效近似方法，提高了强化学习的效率，实验结果表明，相比基于一阶梯度的训练，QNPG具有更快的收敛速度和稳定性，可以减少样本复杂度。 |
| [^93] | [The Disharmony Between BN and ReLU Causes Gradient Explosion, but is Offset by the Correlation Between Activations.](http://arxiv.org/abs/2304.11692) | 本研究阐述了BN和ReLU之间的不和谐是导致梯度爆炸的主要原因，同时发现输入之间的相关性可以缓解这个问题。提出一种基于二阶优化算法的自适应学习率算法，在大批量训练中表现优异，并可替代WarmUp，在小批量训练中也表现不错。 |
| [^94] | [IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies.](http://arxiv.org/abs/2304.10573) | 本文重新解释隐式Q学习(IQL)作为Actor-Critic方法，提出使用扩散行为策略和评判器权重来平衡奖励最大化和与行为策略的分歧。这个方法能够处理复杂和多峰特征的Actor问题。 |
| [^95] | [Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields.](http://arxiv.org/abs/2304.06706) | 本文提出了一种 Zip-NeRF 技术，将 mip-NeRF 360 和基于网格的模型相结合，以实现抗锯齿、提高训练速度并降低误差率。 |
| [^96] | [SAMM (Segment Any Medical Model): A 3D Slicer Integration to SAM.](http://arxiv.org/abs/2304.05622) | 介绍了Segment Any Medical Model (SAMM)，它是用于3D Slicer的SAM的扩展。SAMM在医学图像分割上表现良好，在实时性和通用性方面都有很好的性能，可以推断出掩模。 |
| [^97] | [Why think step-by-step? Reasoning emerges from the locality of experience.](http://arxiv.org/abs/2304.03843) | 本文通过语言模型研究何时以及为什么推理是有帮助的，测试推理在训练数据由相互影响强烈的局部变量集群组成时是否有效。通过一步步的推理，能够将准确的局部推理链接在一起，以估算在训练中没有同时观察到的变量之间的关系。 |
| [^98] | [Torch-Choice: A PyTorch Package for Large-Scale Choice Modelling with Python.](http://arxiv.org/abs/2304.01906) | 本文介绍了一款名为 Torch-Choice 的 PyTorch 软件包，用于管理数据库、构建多项式Logit和嵌套Logit模型，并支持GPU加速，具有灵活性和高效性。 |
| [^99] | [Deep Ranking Ensembles for Hyperparameter Optimization.](http://arxiv.org/abs/2303.15212) | 研究提出了一种新型方法，元学习神经网络代理来优化超参数配置的性能排名，并通过集成来建模不确定性，该方法在超参数优化方面取得了最新最先进的结果。 |
| [^100] | [Multi-modal Variational Autoencoders for normative modelling across multiple imaging modalities.](http://arxiv.org/abs/2303.12706) | 本文提出了一种多模态规范建模框架，能够更好地检测出多种成像和生物变量中的异常性，特别适用于研究带有异质性的疾病。 |
| [^101] | [Self-supervised Meta-Prompt Learning with Meta-Gradient Regularization for Few-shot Generalization.](http://arxiv.org/abs/2303.12314) | 提出了一种自我监督元提示学习框架SUPMER，包括元梯度正则化，用于少样本泛化，通过锚定的元训练任务和基于课程的任务增强丰富了任务分布，解决了在少样本情况下良好初始化软提示和过拟合的问题。 |
| [^102] | [Stochastic Nonsmooth Convex Optimization with Heavy-Tailed Noises.](http://arxiv.org/abs/2303.12277) | 本文分析了具有重尾噪声的随机非光滑凸优化问题，并填补了在函数非光滑场景下的研究空白。 |
| [^103] | [STDLens: Model Hijacking-resilient Federated Learning for Object Detection.](http://arxiv.org/abs/2303.11511) | STDLens 是一种可以防止FL受到模型挟持的攻击的安全方法。它基于三层的取证框架来识别和排除特殊的梯度，并恢复FL的性能。STDLens在物体检测方面实现了最先进的性能并且具有防止模型挟持的鲁棒性。 |
| [^104] | [Reflexion: an autonomous agent with dynamic memory and self-reflection.](http://arxiv.org/abs/2303.11366) | 本文提出 Reflexion 方法，给智能体赋予了动态记忆和自我反思能力，以增强其任务特定的行动选择能力。 |
| [^105] | [Boosting Semi-Supervised Few-Shot Object Detection with SoftER Teacher.](http://arxiv.org/abs/2303.05739) | 本文研究了半监督Few-Shot目标检测任务，发现未标记数据对提高半监督FSOD有好处。受此发现启发，我们引入了SoftER Teacher，一种强大的检测器，用于改进FSOD，不需要丰富的标签，并能在性能方面超越强有力的监督检测器，而且不会出现灾难性遗忘。 |
| [^106] | [Improved Sample Complexity Bounds for Distributionally Robust Reinforcement Learning.](http://arxiv.org/abs/2303.02783) | 本文改进了分布式鲁棒强化学习的样本复杂度界限，提出了稳健分阶段价值学习（RPVL）算法来解决表格剧情学习环境下的不确定性问题。 |
| [^107] | [Stochastic Gradient Descent under Markovian Sampling Schemes.](http://arxiv.org/abs/2302.14428) | 本文研究了基于马尔科夫抽样方案的随机梯度下降算法，提出了MC-SAG算法实现了用于分布式算法的通信效率高的token 算法。 |
| [^108] | [Change is Hard: A Closer Look at Subpopulation Shift.](http://arxiv.org/abs/2302.12254) | 本文分析了子群体转变的各种机制，对20个最先进的算法在12个领域内进行了全面的基准测试，发现现有算法只能应对某些转变，进一步地，提出一种简单易行的选择标准来改善现有算法性能。 |
| [^109] | [One Fits All:Power General Time Series Analysis by Pretrained LM.](http://arxiv.org/abs/2302.11939) | 本论文提出了一种称为 Frozen Pretrained Transformer (FPT) 的预训练模型，利用从数十亿标记训练出来的语言或 CV 模型进行时间序列分析的所有主要类型任务的微调，进而使其在所有任务中都具备着最先进的性能和泛化能力。 |
| [^110] | [On discrete symmetries of robotics systems: A group-theoretic and data-driven analysis.](http://arxiv.org/abs/2302.10433) | 本文研究了运动系统的离散形态对称性，并提出了一个理论和实践框架，用于识别系统的形态对称群，并分析对称性在数据增强和控制设计中的应用。 |
| [^111] | [Dividing and Conquering a BlackBox to a Mixture of Interpretable Models: Route, Interpret, Repeat.](http://arxiv.org/abs/2302.10289) | 本文提出了一种从黑盒模型中构建可解释模型的方法。该方法将黑盒模型分成可解释模型的混合物和残差网络，并使用一阶逻辑对可解释模型进行基本推理。此方法在多个数据集上表现优异且产生高度可解释的模型。 |
| [^112] | [Navya3DSeg -- Navya 3D Semantic Segmentation Dataset & split generation for autonomous vehicles.](http://arxiv.org/abs/2302.08292) | Navya3DSeg是一个新的、具有多样标签空间的数据集，对应于大规模生产级操作领域，可用于自动驾驶感知，同时提供了一个新的方法来生成顺序数据集拆分。 |
| [^113] | [Predicting municipalities in financial distress: a machine learning approach enhanced by domain expertise.](http://arxiv.org/abs/2302.05780) | 本文运用机器学习模型结合会计司法专家的知识和经验，预测意大利城市的财政困境。 |
| [^114] | [DeepVATS: Deep Visual Analytics for Time Series.](http://arxiv.org/abs/2302.03858) | DeepVATS是一个开源的工具，用于对时间序列数据进行深度视觉分析，通过自监督的方式训练掩码时间序列自编码器，重建时间序列的补丁，并将模型嵌入中所包含的知识投影到交互式图中，从而可以轻松识别时间序列模式和异常。 |
| [^115] | [Beyond Statistical Similarity: Rethinking Metrics for Deep Generative Models in Engineering Design.](http://arxiv.org/abs/2302.02913) | 本文提供了一篇深度学习在工程设计中度量方法的综述和指南。传统的基于似然性的统计度量方法在对工程应用的要求上可能无法充分捕捉，因此本文编辑了一组全面的新度量标准，旨在解决传统度量标准的缺点，并更好地与工程设计的需求相一致。通过案例研究，本文展示了这些度量标准如何应用于评估深度生成模型在工程设计中的性能，并发现这些度量标准在捕捉设计的重要细微差别方面表现优于传统的统计度量标准。 |
| [^116] | [Precursor recommendation for inorganic synthesis by machine learning materials similarity from scientific literature.](http://arxiv.org/abs/2302.02303) | 本研究利用从科学文献中提取的合成配方的知识库，自动学习推荐新目标材料的前体材料，通过学习材料化学相似性并将新目标材料的合成参照类似材料的先前合成程序，成功率至少为82%。 |
| [^117] | [User-centric Heterogeneous-action Deep Reinforcement Learning for Virtual Reality in the Metaverse over Wireless Networks.](http://arxiv.org/abs/2302.01471) | 本文提出了一种用户中心异构行动深度强化学习（UCHA-DRL）算法，可以使用在无线网络的元宇宙虚拟现实中。该算法可以联合优化服务器向用户下行通信的信道访问安排和传输功率来提高用户的效用。 |
| [^118] | [Constrained Online Two-stage Stochastic Optimization: Near Optimal Algorithms via Adversarial Learning.](http://arxiv.org/abs/2302.00997) | 在线两阶段随机优化算法的累计目标值最小化，同时保证长期平均第二阶段决策结果属于一个集合。采用对抗性学习算法从在线两阶段问题中开发在线算法，其遗憾界可以降至嵌入对抗性学习算法的遗憾界，并在各种设置下获得了新的结果。 |
| [^119] | [Sharp Variance-Dependent Bounds in Reinforcement Learning: Best of Both Worlds in Stochastic and Deterministic Environments.](http://arxiv.org/abs/2301.13446) | 本研究将马尔可夫决策过程的方差相关遗憾界限应用到强化学习中，提出了两个新的环境规范来表征环境的方差属性，并设计出基于模型和无模型的算法，对于随机和确定性环境同时极小极大最优的界限是第一次被证明出来的。 |
| [^120] | [FedEBA+: Towards Fair and Effective Federated Learning via Entropy-Based Model.](http://arxiv.org/abs/2301.12407) | FedEBA+是一种新的联邦学习算法，它采用公平聚合方案和对齐更新方法，在同时提高全局模型性能的同时提高公平性。实验证明FedEBA+优于其他公平性联邦学习方法。 |
| [^121] | [BiBench: Benchmarking and Analyzing Network Binarization.](http://arxiv.org/abs/2301.11233) | BiBench提供了一个严谨设计的基准测试和深入分析，揭示了网络二值化面临的挑战和特性，为从业者和研究人员提供了有价值的参考。 |
| [^122] | [Which Experiences Are Influential for Your Agent? Policy Iteration with Turn-over Dropout.](http://arxiv.org/abs/2301.11168) | 本文提出了一种名为PI+ToD的方法，它通过淘汰正则化来高效估算经验对RL代理性能的影响。 |
| [^123] | [Bias-to-Text: Debiasing Unknown Visual Biases through Language Interpretation.](http://arxiv.org/abs/2301.11104) | 本文提出了基于语言解释的去偏见(B2T)框架，通过分析图像标题中的关键词，比较关键词和图像之间的相似性，识别和减缓视觉模型中的偏见，并提出了针对零样本分类器和文本到图像扩散模型的去偏见策略。 |
| [^124] | [On the Importance of Noise Scheduling for Diffusion Models.](http://arxiv.org/abs/2301.10972) | 本研究实证发现，噪声调度对于扩散生成模型的表现至关重要，最优的噪声调度策略会随着任务的不同而有所不同，并且，在图像大小增加时最优噪声调度策略会朝更嘈杂的方向转移。将输入数据按比例缩放并保持噪声调度函数不变，能够在不同图像大小上获得良好的效果。 |
| [^125] | [Sampling-based Nystr\"om Approximation and Kernel Quadrature.](http://arxiv.org/abs/2301.09517) | 本文提出了一种基于抽样的Nyström逼近方法用于核积分。同时，引入了一种非i.i.d.地标点的理论保证方法，使得提高了逼近的精度。 |
| [^126] | [The #DNN-Verification problem: Counting Unsafe Inputs for Deep Neural Networks.](http://arxiv.org/abs/2301.07068) | 本文提出了#DNN-Verification问题，即计算违反特定安全性质的DNN输入配置数量的问题。作者提出了一种新的方法和一种随机的近似方法，分别给出了确切的违规计数和可证明概率界，并在安全关键基准测试上进行了实验比较。 |
| [^127] | [Federated Transfer-Ordered-Personalized Learning for Driver Monitoring Application.](http://arxiv.org/abs/2301.04829) | 本文提出了一个名为FedTOP的联邦迁移有序个性化学习框架，用于解决数据和系统异构性、大规模并行通信资源、恶意攻击和数据污染等问题。实验结果显示，该框架在两个数据集上的测试客户端分别达到92.32％和95.96％的准确率。与基线相比，在准确率方面提高了462％，并降低了37.46％的通信成本。 |
| [^128] | [Improving Sequential Recommendation Models with an Enhanced Loss Function.](http://arxiv.org/abs/2301.00979) | 本文研究了顺序推荐模型常用损失函数的优劣，提出了一种改进的损失函数。实验表明，这种改进的损失函数可以显著提升多种顺序推荐模型的性能。 |
| [^129] | [Black-box language model explanation by context length probing.](http://arxiv.org/abs/2212.14815) | 该论文提出了一个模型不可知的新颖解释技术：上下文长度探测，通过跟踪模型预测与可用上下文长度的关系来对不同上下文分配不同的重要性得分。该方法适用于大型预训练语言模型，并有利于研究远距离依赖性。 |
| [^130] | [KNIFE: Distilling Reasoning Knowledge From Free-Text Rationales.](http://arxiv.org/abs/2212.09721) | 通过KNIFE，可以从自由文本理由中提取推理知识，进而在小型语言模型中提高推理能力。 |
| [^131] | [Physics-constrained deep learning postprocessing of temperature and humidity.](http://arxiv.org/abs/2212.04487) | 本研究提出了一种物理约束下的深度学习后处理方法，通过整合气象专业知识以解析方程式的形式，实现物理一致性。研究发现在瑞士地面天气的后处理中，通过强制执行热力学状态方程来约束神经网络可以产生物理上一致的温湿度预测结果。 |
| [^132] | [Neural DAEs: Constrained neural networks.](http://arxiv.org/abs/2211.14302) | 本文研究了将辅助代数轨迹信息明确添加到神经网络中的影响，并通过稳定化和投影方法合并信息，对多体摆和分子动力学情景进行了模拟实验。该方法易于实现，对训练性能影响有限，在推理方面给出了显著提升。 |
| [^133] | [OpenFE: Automated Feature Generation with Expert-level Performance.](http://arxiv.org/abs/2211.12507) | 本文提出的自动特征生成工具OpenFE可以与机器学习专家提供的结果相媲美，具有高效和准确的特点，通过新颖的特征提升方法和两阶段修剪算法实现。 |
| [^134] | [Towards Adversarially Robust Recommendation from Adaptive Fraudster Detection.](http://arxiv.org/abs/2211.11534) | 本文提出了一种针对推荐系统的MetaC恶意攻击，并设计了一种自适应欺诈者检测模块PDR，明确考虑标签的不确定性，提高了推荐系统的鲁棒性。 |
| [^135] | [GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-distribution Generalization Perspective.](http://arxiv.org/abs/2211.08073) | 本文提出了第一个创建名为方法的统一基准的尝试，用于评估NLP模型中的OOD鲁棒性，该基准包括13个公开可用的OOD测试数据集，并在21个常用的PLMs上对8个经典NLP任务进行评估。 |
| [^136] | [Optimal Privacy Preserving for Federated Learning in Mobile Edge Computing.](http://arxiv.org/abs/2211.07166) | 本文介绍了一种在移动边缘计算环境下最优隐私保护的联邦学习方法，通过联合优化量化和二项机制参数以及通信资源，最大化收敛速度并保证DP要求。 |
| [^137] | [The Sample Complexity of Online Contract Design.](http://arxiv.org/abs/2211.05732) | 本文解决了在线合同设计中一个悬而未决的问题，证明了指数级的$m$个样本就足以学习一个近乎最优的合同。 |
| [^138] | [A Survey on Quantum Reinforcement Learning.](http://arxiv.org/abs/2211.03464) | 本文综述了量子强化学习的最新进展和相关文献，重点介绍了基于噪声中等规模量子设备的变分量子电路在经典强化学习中的应用，以及基于未来容错硬件的量子强化学习算法，其中一些具有可证明的量子优势。 |
| [^139] | [An Empirical Bayes Analysis of Object Trajectory Representation Models.](http://arxiv.org/abs/2211.01696) | 本论文对对象轨迹表示模型的复杂度和拟合误差之间的权衡进行了经验分析，发现简单的线性模型就能够高度重现真实世界的轨迹，通过使用经验贝叶斯方法可以为轨迹跟踪问题中必要的运动模型提供信息，并可以帮助规范预测模型。 |
| [^140] | [Sequence-Based Plan Feasibility Prediction for Efficient Task and Motion Planning.](http://arxiv.org/abs/2211.01576) | 本文提出一种基于学习的任务和动作规划（TAMP）算法，通过序列预测辅助搜索，显著提高了规划效率，运行时间缩短了80%。 |
| [^141] | [RGMIM: Region-Guided Masked Image Modeling for COVID-19 Detection.](http://arxiv.org/abs/2211.00313) | 本论文提出了一种针对COVID-19检测的新颖区域引导的掩膜图像建模方法，该方法通过利用肺掩模信息来识别有效区域，以学习更有用的COVID-19检测信息。 |
| [^142] | [Variance reduced Shapley value estimation for trustworthy data valuation.](http://arxiv.org/abs/2210.16835) | 本文提出了一种名为VRDS的分层抽样方法，用于评估数据价值，以缩小排列抽样方法的估计方差，并在不同类型的数据集和数据删除应用程序中得到验证。 |
| [^143] | [A Training and Inference Strategy Using Noisy and Enhanced Speech as Target for Speech Enhancement without Clean Speech.](http://arxiv.org/abs/2210.15368) | 为解决语音增强领域中“无清晰语音”的挑战，提出了使用增强语音作为目标的训练和推理策略，即使在域内和域外噪声差异较大的情况下仍能有效，实验结果优于基线方法。 |
| [^144] | [DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models.](http://arxiv.org/abs/2210.14896) | 介绍了DiffusionDB数据集，这是一个规模庞大的文本到图像提示数据集，总计包含1400万张图像和180万个唯一提示。该数据集被用来帮助研究人员解决文本提示生成图像时所需的适当提示的问题，并指出了一些特定的提示样式和超参数值可能导致模型错误，甚至生成误导信息。 |
| [^145] | [Broken Neural Scaling Laws.](http://arxiv.org/abs/2210.14891) | 本文提出了一个平滑破碎的幂律函数形式，可以准确地模拟和外推深度神经网络的缩放行为，适用于各种架构和大量不同任务，包括视觉、语言、音频、视频、生成建模、对比学习、机器人、不确定性估计/校准、对抗鲁棒性、分子、计算机编程/编码、数学单词问题、算术、无监督/自监督学习和强化学习。 |
| [^146] | [Layer-Neighbor Sampling -- Defusing Neighborhood Explosion in GNNs.](http://arxiv.org/abs/2210.13339) | 本文提出了一种新的采样算法，名为LABOR，旨在替代现有的Neighbor Sampling算法。相比之下，它能够采样更少的顶点但不会牺牲质量，并且在相同的顶点采样预算约束下，收敛更快，可以使用更大的批处理大小。 |
| [^147] | [Horizon-Free and Variance-Dependent Reinforcement Learning for Latent Markov Decision Processes.](http://arxiv.org/abs/2210.11604) | 本文研究了具有上下文后见性的 LMDP 强化学习遗憾最小化问题。通过设计一个新颖的模型基础算法框架，我们证明了一个与计划视野对数相关的 $\widetilde{O}\left(\sqrt{M \Gamma S A K}\right)$ 遗憾度上限，并对 alpha 向量的总方差进行分析。同时，我们提出了一个 $\Omega\left(\sqrt{M S A K}\right)$ 的遗憾度下限，它在 $\Gamma=2$ 时证明了我们的上界是最优的。 |
| [^148] | [DICTDIS: Dictionary Constrained Disambiguation for Improved NMT.](http://arxiv.org/abs/2210.06996) | DICTDIS是一种新颖有词典约束的NMT系统，其利用多个字典候选项进行训练，实现了从多义词中消除翻译歧义的目的，提高了翻译质量。 |
| [^149] | [Multi-User Reinforcement Learning with Low Rank Rewards.](http://arxiv.org/abs/2210.05355) | 本文提出了一个新的多用户强化学习算法，可以在用户奖励矩阵具有低秩结构的情况下显著降低样本复杂度。 |
| [^150] | [Multi-CLS BERT: An Efficient Alternative to Traditional Ensembling.](http://arxiv.org/abs/2210.05043) | Multi-CLS BERT是一种高效的BERT模型集成方法，通过使用多个CLS标记并鼓励它们多样性，有效地提高了整体准确性和置信度估计。 |
| [^151] | [FaDIn: Fast Discretized Inference for Hawkes Processes with General Parametric Kernels.](http://arxiv.org/abs/2210.04635) | 本论文提出了一种使用具有有限支持的一般参数核进行TPP推理的高效解决方案，该方法采用了离散化方法，并通过多项实验证明了该方法的统计和计算效率。 |
| [^152] | [GNM: A General Navigation Model to Drive Any Robot.](http://arxiv.org/abs/2210.03370) | 通用目标条件模型GNM可以通过多样化的机器人数据实现更强大、更健壮的导航性能，并能驱动任何具有适当视觉感知输入的机器人。 |
| [^153] | [Hierarchical Adversarial Inverse Reinforcement Learning.](http://arxiv.org/abs/2210.01969) | 本文提出了一种分层对抗逆强化学习算法，能够在复杂任务中学习到具有层次结构的最优策略，比现有的方法更加有效。 |
| [^154] | [Efficient Quantum Agnostic Improper Learning of Decision Trees.](http://arxiv.org/abs/2210.00212) | 本文提出了第一个无需成员查询在多项式时间内学习大小为$t$的决策树的算法，并成功量化了Kalai和Kanade的不知性增强算法，得到了第一个高效的量子不知性增强算法。 |
| [^155] | [On Quantum Speedups for Nonconvex Optimization via Quantum Tunneling Walks.](http://arxiv.org/abs/2209.14501) | 本文探究了使用量子隧穿行走（QTW）在非凸优化问题中产生的量子加速效应。当不同的局部极小值由高但薄的势垒分隔而极小值是平坦的时候，QTW对比经典随机梯度下降（SGD）实现了加速。 |
| [^156] | [L2XGNN: Learning to Explain Graph Neural Networks.](http://arxiv.org/abs/2209.14402) | L2XGNN提出了一个框架来解释图神经网络，通过选择解释子图（模体）实现忠实的解释。该框架能够识别负责预测图属性的模体，并实现与基线方法相同的分类精度。 |
| [^157] | [Quantification before Selection: Active Dynamics Preference for Robust Reinforcement Learning.](http://arxiv.org/abs/2209.11596) | 本论文提出主动动态偏好(ADP)方法，在稳健强化学习中使用熵量化动态偏好来活跃地平衡策略的探索和利用，有效避免策略的过于保守或过于乐观，提高了策略在各种目标域中的鲁棒性，超过现有的state-of-the-art方法。 |
| [^158] | [Psychologically-informed chain-of-thought prompts for metaphor understanding in large language models.](http://arxiv.org/abs/2209.08141) | 本文介绍了一种在大语言模型中使用心理学引导思维的方法来增强隐喻理解的性能。这种方法使用了隐含变量和关系来选择正确的释义。 |
| [^159] | [Detection of Interacting Variables for Generalized Linear Models via Neural Networks.](http://arxiv.org/abs/2209.08030) | 本文提出了一种使用神经网络和模型特定交互检测方法来自动化寻找GLMs中应添加的交互作用以提高其预测能力的方法。 |
| [^160] | [Overhead-Free Blockage Detection and Precoding Through Physics-Based Graph Neural Networks: LIDAR Data Meets Ray Tracing.](http://arxiv.org/abs/2209.07350) | 本文提出了一种基于物理学图神经网络的无开销阻塞检测和预编码方法，使用激光雷达数据进行分类，通过光线追踪得到信道估计，经过逐步训练和设计预编码器，达到了较高的性能。 |
| [^161] | [On the Reuse Bias in Off-Policy Reinforcement Learning.](http://arxiv.org/abs/2209.07074) | 本文揭示了离线强化学习中一个新的偏见问题：重复使用偏见，提出了一种简单有效的方法——重复使用感知重要性加权（RAW）来解决这个问题，并证明RAW显著提高了离线方法的样本效率和鲁棒性。 |
| [^162] | [The Mori-Zwanzig formulation of deep learning.](http://arxiv.org/abs/2209.05544) | 本文提出了基于莫里-茨旺齐格形式主义的深度学习新表述，引入了神经网络记忆的新概念，并通过线性算子方程直接向前和向后传播感兴趣的量。收缩映射理论被用来开发记忆衰减随网络层数增加的充分条件。 |
| [^163] | [Bispectral Neural Networks.](http://arxiv.org/abs/2209.03416) | 本文提出了一种称为双谱神经网络的模型，能够从数据的隐含对称性中学习群、不可约表示和对应的完全不变映射，具有强大的基于不变性的对抗鲁棒性。 |
| [^164] | [Shaken, and Stirred: Long-Range Dependencies Enable Robust Outlier Detection with PixelCNN++.](http://arxiv.org/abs/2208.13579) | 本文提出了两种双射变换方法（“搅拌”和“摇晃”），用于改善深度自回归模型PixelCNN++似然性中的低级偏差，并隔离长程依赖的贡献。这些方法可以在评估时很容易计算，并且在离群检测方面表现出了很好的效果。 |
| [^165] | [LAMDA-SSL: Semi-Supervised Learning in Python.](http://arxiv.org/abs/2208.04610) | LAMDA-SSL是一款Python半监督学习工具包，提供了详细的用法文档和实现的所有算法，极大地方便了用户使用。 |
| [^166] | [Learning to Learn to Predict Performance Regressions in Production at Meta.](http://arxiv.org/abs/2208.04351) | 本文介绍了在Meta研究和部署的基于机器学习的回归预测流程，研究结果显示性能预测问题的固有难度，SuperPerforator模型表现最佳。 |
| [^167] | [DIVISION: Memory Efficient Training via Dual Activation Precision.](http://arxiv.org/abs/2208.04187) | 本文提出一种内存高效的DNN训练方法DIVISION，通过保留LFC的高精度，将HFC压缩成低精度的轻量副本，显著减少了内存成本。 |
| [^168] | [Rankability and Linear Ordering Problem: New Probabilistic Insight and Algorithms.](http://arxiv.org/abs/2208.03860) | 该论文提出了一种基于概率模型和Slater谱概念的算法，能够判断和解决线性排序问题中数据是否可排序并给出有意义的解释。 |
| [^169] | [Differentiable Agent-based Epidemiology.](http://arxiv.org/abs/2207.09714) | 本文介绍一种可扩展、可微分的基于Agent的流行病模拟设计——GradABM，能够在较短时间内快速模拟百万级别的人口，并与深度神经网络集成和接受异构数据源，为校准、预测和评估政策干预提供了便利。 |
| [^170] | [Generalizable Memory-driven Transformer for Multivariate Long Sequence Time-series Forecasting.](http://arxiv.org/abs/2207.07827) | 本文提出了一种通用记忆驱动变压器，通过集成多个时间序列特征来驱动预测过程，逐步引入噪声以增强泛化能力，在多个数据集上实现了更优秀的预测性能。 |
| [^171] | [Learning Deep Time-index Models for Time Series Forecasting.](http://arxiv.org/abs/2207.06046) | 本文提出了DeepTime框架，一种用于学习深度时间索引模型的元优化框架，能够有效地预测时间序列，并在真实世界数据集上获得了与先进方法相当的结果。 |
| [^172] | [Big Learning.](http://arxiv.org/abs/2207.03899) | 大型学习是一种把大规模完整/不完整的训练数据中固有信息同步建模并使用一个通用基础模型，利用所有联合/条件/边际数据能力并统一传统机器学习范式的通用学习范式。 |
| [^173] | [Persistent homology-based descriptor for machine-learning potential of amorphous structures.](http://arxiv.org/abs/2206.13727) | 本文提出了一种基于持续同调的二维描述符PD，用于构建表示原子配置的不变描述符，在机器学习势的构建中，PD能够捕获不同长度的相关性，并能够提高模型性能。 |
| [^174] | [Good Time to Ask: A Learning Framework for Asking for Help in Embodied Visual Navigation.](http://arxiv.org/abs/2206.10606) | 该论文提出了一种学习框架，使得机器人可以主动寻求帮助以解决表达需求式视觉导航任务中存在的难题，并展示了其在缺乏反馈信息的情况下依然稳健的抗干扰能力。 |
| [^175] | [Personalized Subgraph Federated Learning.](http://arxiv.org/abs/2206.10206) | 本文提出一个新的子图联邦学习问题，即个性化子图联邦学习，专注于联合改进相关的本地GNN并提出了一个新颖的框架FED-PUB来解决这个问题。在实验中表现出了优异的性能。 |
| [^176] | [Classification Utility, Fairness, and Compactness via Tunable Information Bottleneck and R\'enyi Measures.](http://arxiv.org/abs/2206.10043) | 本文提出了一种新型公平表示学习方法(RFIB)，兼顾了表示效用、公平性和紧凑性，并将其应用于图像和表格数据分类中。该方法考虑到了人口平等和等化赔率等不同的公平性约束，通过涉及经典信息瓶颈(IB)度量的损失函数实现。实验表明，该方法能够在实现公平性的同时保持较高的准确性。 |
| [^177] | [PrivHAR: Recognizing Human Actions From Privacy-preserving Lens.](http://arxiv.org/abs/2206.03891) | 该论文提出了一种优化框架，通过参数化摄像头镜头，对视频进行处理以保护个人隐私，并在维护活动识别特征的同时进行了模拟和硬件实验的验证。 |
| [^178] | [Topological Deep Learning: Going Beyond Graph Data.](http://arxiv.org/abs/2206.00606) | 本文提出了一个拓扑深度学习的框架，其中包含组合复合体这一新型拓扑域。组合复合体结合了超图和胞腔复合体的优点，允许构建分层高阶关系。 |
| [^179] | [Generalized Supervised Contrastive Learning.](http://arxiv.org/abs/2206.00384) | 本文提出了一种广义的有监督对比损失，通过充分利用标签分布来增强有监督对比损失的能力，从而适应各种现有技术，包括CutMix和知识蒸馏。在几个基准数据集上进行了实验，结果显示该方法优于现有的有监督对比学习方法，并在CIFAR-10，CIFAR-100和ImageNet上实现了最先进的结果。 |
| [^180] | [SepIt: Approaching a Single Channel Speech Separation Bound.](http://arxiv.org/abs/2205.11801) | 该论文提出了一种接近单通道语音分离界限的方法SepIt，在实验中表现优于最先进的神经网络方法，尤其是在5个和10个说话人的情况下仍有提高空间。 |
| [^181] | [Memorization and Optimization in Deep Neural Networks with Minimum Over-parameterization.](http://arxiv.org/abs/2205.10217) | 本文提供了一个最小超参数化的深度神经网络中最小的NTK特征值的下界，具有次线性层宽的深层神经网络是强大的记忆器和优化器，只要参数数量超过样本数量。 |
| [^182] | [Projection-free Online Learning with Arbitrary Delays.](http://arxiv.org/abs/2204.04964) | 该论文提出了一种能够应对任意延迟的无投影在线算法，与现有算法相比，其具有更好的遗憾保证。 |
| [^183] | [Personalized incentives as feedback design in generalized Nash equilibrium problems.](http://arxiv.org/abs/2203.12948) | 本文研究了对称互动的非单调广义纳什均衡问题，并提出了一种半分散纳什均衡寻找算法，其中，协调员通过整合代理的反馈来学习代理的伪梯度并为其设计个性化的激励，代理计算扩展博弈的解并反馈测量给协调员，该算法可返回静态设置下的纳什均衡。 |
| [^184] | [Multi-Agent Active Search using Detection and Location Uncertainty.](http://arxiv.org/abs/2203.04524) | 本文提出了一种新的算法，可以同时处理目标探测和位置不确定性，可以优化多智能体主动搜索的效率。 |
| [^185] | [Rethinking Efficient Lane Detection via Curve Modeling.](http://arxiv.org/abs/2203.02431) | 本文提出了一种新的参数曲线车道检测方法，在计算上更加简单且容易处理优化问题，并且在多个数据集中实现了最优的性能表现，可以作为未来车道检测领域的新基准。 |
| [^186] | [Graph Attention Retrospective.](http://arxiv.org/abs/2202.13060) | 图注意力网络是一种能够从邻居节点的特征中聚合信息的模型，通过对上下文随机块模型的节点分类问题进行研究，证明了在“易”区间内，它能够区分跨类和内类边缘并维护重要边缘的权重。 |
| [^187] | [Adaptive and Robust Multi-Task Learning.](http://arxiv.org/abs/2202.05250) | 本文提出一系列自适应方法，能够同时处理多任务学习的相似性和差异性，并具有统计保证和鲁棒性。 |
| [^188] | [Deep Discriminative to Kernel Generative Networks for Calibrated Inference.](http://arxiv.org/abs/2201.13001) | 该论文提出了将判别网络转换为生成网络的方法，用高斯核替换多面体中的仿射函数来生成模型，解决了内部和外部数据校准问题，并在 CIFAR-10，CIFAR-100 和 SVHN 等基准数据集上测试了方法的有效性。 |
| [^189] | [A Robust and Flexible EM Algorithm for Mixtures of Elliptical Distributions with Missing Data.](http://arxiv.org/abs/2201.12020) | 本研究提出了一种鲁棒且灵活的EM算法，用于处理缺失数据的混合椭圆分布，解决了非高斯数据插补的问题。 |
| [^190] | [PoNet: Pooling Network for Efficient Token Mixing in Long Sequences.](http://arxiv.org/abs/2110.02442) | PoNet是一种池化网络，可用于长序列中的token混合，其具有线性复杂度，并可以比Transformer更好地处理长序列。该方法提供了一种高效且有效的自注意替代方案。 |
| [^191] | [Barycentric-alignment and reconstruction loss minimization for domain generalization.](http://arxiv.org/abs/2109.01902) | 本论文提出了一个新的理论上界，它不包含双重依赖性的术语，在领域归纳中优化了未见域的风险上界。 |
| [^192] | [Exploring the Context Generalizability in Spatiotemporal Crowd Flow Prediction: Benchmark and Guideline.](http://arxiv.org/abs/2106.16046) | 本文研究了时空人群流量预测中的上下文泛化性，建立了基准，提出了通用分类法，为上下文选择和建模提供了指南。 |
| [^193] | [Identifiability of interaction kernels in mean-field equations of interacting particles.](http://arxiv.org/abs/2106.05565) | 本研究探索了相互作用微粒平均场方程中相互作用核的可辨识性问题，并确定了二次损失函数仅在特定的函数空间中才具有唯一的最小化器。此外，对于计算实践，研究证明了反问题的病态性质，需要进行正则化处理。 |
| [^194] | [Encoding physics to learn reaction-diffusion processes.](http://arxiv.org/abs/2106.04781) | 通过将测量数据和物理知识的有限先验统一来进行机器学习提供了一种新的解决非线性反应-扩散过程的途径。 |
| [^195] | [A first look into the carbon footprint of federated learning.](http://arxiv.org/abs/2102.07627) | 本文是对联邦学习的碳足迹进行的首次系统研究，发现FL的碳排放量最多可达传统学习的两倍以上。 |
| [^196] | [And/or trade-off in artificial neurons: impact on adversarial robustness.](http://arxiv.org/abs/2102.07389) | 本文研究了人工神经元中与/或函数的连续性对分类鲁棒性的影响，提出了增加类AND神经元在网络中比例的措施，实验结果显示该方法具有潜在的应用前景。 |
| [^197] | [Natural Language Specification of Reinforcement Learning Policies through Differentiable Decision Trees.](http://arxiv.org/abs/2101.07140) | 本文提出了一个新的可微分决策树框架，允许人们通过自然语言指定初始行为模型，并将其转换为词汇决策树，为机器人的强化学习策略提供指导。 |
| [^198] | [Variance-Reduced Off-Policy TDC Learning: Non-Asymptotic Convergence Analysis.](http://arxiv.org/abs/2010.13272) | 本文提出了离策略TDC算法的方差缩减方案，并在i.i.d.和马尔可夫抽样中分析了其收敛率，结果表明该算法实现了与已知最佳下限相匹配的i.i.d.样本复杂度和接近最优的马尔可夫样本复杂度。 |
| [^199] | [Deep Dynamic Factor Models.](http://arxiv.org/abs/2007.11887) | 提出了一种深度神经网络框架，Deep Dynamic Factor Model (D$^2$FM)，能够将数百个宏观经济和金融时间序列可用信息编码为少量的未观察到的潜在状态，并能提高现在预测和预测实验中的表现。 |
| [^200] | [Leveraging End-to-End Speech Recognition with Neural Architecture Search.](http://arxiv.org/abs/1912.05946) | 本文研究表明，通过神经架构搜索可以在非常低的计算成本情况下显著提高深度语音模型的准确性，取得了与最先进结果相当的水平。 |
| [^201] | [SHARP: An Adaptable, Energy-Efficient Accelerator for Recurrent Neural Network.](http://arxiv.org/abs/1911.01258) | 论文提出了一种适应性和节能型的循环神经网络加速器，通过智能的瓦片分派机制，实现了处理不同RNN维度的高计算效率。 |
| [^202] | [The Theory Behind Overfitting, Cross Validation, Regularization, Bagging, and Boosting: Tutorial.](http://arxiv.org/abs/1905.12787) | 该论文介绍了过拟合、交叉验证、正则化、装袋法和提升法的相关理论，包括定义和具体实现，并给出了AdaBoost的泛化误差上限的具体计算方法。 |
| [^203] | [Eigenvalue and Generalized Eigenvalue Problems: Tutorial.](http://arxiv.org/abs/1903.11240) | 本文是一篇阐述特征值和广义特征值问题的教程。特征值和广义特征值问题可用于各种机器学习算法中，如主成分分析和Fisher判别分析等。 |
| [^204] | [Recovery Bounds on Class-Based Optimal Transport: A Sum-of-Norms Regularization Framework.](http://arxiv.org/abs/1903.03850) | 该论文提出了一个基于范数求和正则化项的凸性最优传输程序，在几何假设条件下可证明恢复基础类结构。该论文还提供了一种加速的近端算法，并提出了一种新的唯一性优化方式。实验表明，新的正则化程序不仅可以更好地保留数据中的类结构，还可以在数据几何形状方面提供额外的鲁棒性。 |

# 详细

[^1]: 关于Moreau包络的一阶元强化学习

    On First-Order Meta-Reinforcement Learning with Moreau Envelopes. (arXiv:2305.12216v1 [cs.LG])

    [http://arxiv.org/abs/2305.12216](http://arxiv.org/abs/2305.12216)

    本文提出了一种Moreau包络元强化学习算法（MEMRL），通过利用Moreau包络代理正则化器，可以学习一个可以适应任务分布的元策略。

    

    元强化学习(MRL)是一种训练智能体在新环境和任务中快速适应的有前途的框架。在本文中，我们研究了基于策略梯度的MRL问题，提出了一种利用Moreau包络代理正则化器来共同学习可以适应每个任务环境的元策略的新算法。我们的算法称为Moreau包络元强化学习(MEMRL)，它通过梯度优化和Moreau包络正则化的组合有效地更新策略参数，学习可以适应任务分布的元策略。Moreau包络提供了策略优化问题的平滑近似，使我们能够应用标准优化技术并收敛到适当的稳定点。我们对MEMRL算法进行了详细的分析，展示了对于非凸问题，它可以以亚线性的收敛速度达到一阶稳定点。

    Meta-Reinforcement Learning (MRL) is a promising framework for training agents that can quickly adapt to new environments and tasks. In this work, we study the MRL problem under the policy gradient formulation, where we propose a novel algorithm that uses Moreau envelope surrogate regularizers to jointly learn a meta-policy that is adjustable to the environment of each individual task. Our algorithm, called Moreau Envelope Meta-Reinforcement Learning (MEMRL), learns a meta-policy that can adapt to a distribution of tasks by efficiently updating the policy parameters using a combination of gradient-based optimization and Moreau Envelope regularization. Moreau Envelopes provide a smooth approximation of the policy optimization problem, which enables us to apply standard optimization techniques and converge to an appropriate stationary point. We provide a detailed analysis of the MEMRL algorithm, where we show a sublinear convergence rate to a first-order stationary point for non-convex p
    
[^2]: 用动态分批技术平衡分布式机器学习培训中的资源异质性

    Taming Resource Heterogeneity In Distributed ML Training With Dynamic Batching. (arXiv:2305.12213v1 [cs.LG])

    [http://arxiv.org/abs/2305.12213](http://arxiv.org/abs/2305.12213)

    本文提出一种动态分批技术，用于分布式数据并行训练，可以根据每个工作节点的资源可用性和吞吐量调整小批量大小，从而平衡所有工作节点的迭代时间，减少分布式机器学习模型训练时间。

    

    目前分布式模型训练的技术和系统大多都假设集群由具有恒定资源可用性的同构服务器组成，然而集群异质性在计算基础设施中是普遍存在的，并且是低廉的瞬态资源（例如 EC2 spot 实例）的基本特性。在本文中，我们开发了一种动态分批技术，用于分布式数据并行训练，根据每个工作节点的资源可用性和吞吐量调整小批量大小。我们的小批量控制器旨在平衡所有工作节点的迭代时间，并促进使用具有不同数量 CPU 和 GPU 资源的服务器组成的集群进行训练。该变量小批量技术使用比例控制和 PID 控制器的思想来找到稳定的小批量大小。我们的实证评估表明，动态分批可以将异构集群上的模型训练时间缩短超过 4 倍。

    Current techniques and systems for distributed model training mostly assume that clusters are comprised of homogeneous servers with a constant resource availability. However, cluster heterogeneity is pervasive in computing infrastructure, and is a fundamental characteristic of low-cost transient resources (such as EC2 spot instances). In this paper, we develop a dynamic batching technique for distributed data-parallel training that adjusts the mini-batch sizes on each worker based on its resource availability and throughput. Our mini-batch controller seeks to equalize iteration times on all workers, and facilitates training on clusters comprised of servers with different amounts of CPU and GPU resources. This variable mini-batch technique uses proportional control and ideas from PID controllers to find stable mini-batch sizes. Our empirical evaluation shows that dynamic batching can reduce model training times by more than 4x on heterogeneous clusters.
    
[^3]: 通用逼近的词汇：一种将映射组合看作语言的视角

    Vocabulary for Universal Approximation: A Linguistic Perspective of Mapping Compositions. (arXiv:2305.12205v1 [cs.LG])

    [http://arxiv.org/abs/2305.12205](http://arxiv.org/abs/2305.12205)

    本文探讨了通用逼近的词汇，证明了有限“词汇”存在并可用于逼近任何连续映射$f$和紧致域$\Omega$中的每个点，误差小于$\varepsilon$。

    

    近年来，基于深度学习的序列建模，如语言模型，受到了广泛关注和成功的应用，这促使研究人员探索将非连续问题转化为连续形式的可能性。本文沿着这个思路，将深度神经网络表示为一系列映射函数的组合，其中每个组合可视为一个“单词”。然而，线性映射的权重是未确定的，因此需要无限数量的单词。本文研究有限情况，构建性地证明了通用逼近的有限“词汇”$V=\{\phi_i: \mathbb{R}^d \to \mathbb{R}^d | i=1,...,n\}$存在，其中$n = O(d^2)$。也就是说，对于任何连续映射$f: \mathbb{R}^d \to \mathbb{R}^d$、紧致域$\Omega$和$\varepsilon>0$，存在映射序列$\phi_{i_1}, ..., \phi_{i_m} \in V, m \in \mathbb{Z}_+$，使得组合$\phi_{i_m}$能够逼近$f$和$\Omega$中的每个点，且误差小于$\varepsilon$。

    In recent years, deep learning-based sequence modelings, such as language models, have received much attention and success, which pushes researchers to explore the possibility of transforming non-sequential problems into a sequential form. Following this thought, deep neural networks can be represented as composite functions of a sequence of mappings, linear or nonlinear, where each composition can be viewed as a \emph{word}. However, the weights of linear mappings are undetermined and hence require an infinite number of words. In this article, we investigate the finite case and constructively prove the existence of a finite \emph{vocabulary} $V=\{\phi_i: \mathbb{R}^d \to \mathbb{R}^d | i=1,...,n\}$ with $n=O(d^2)$ for the universal approximation. That is, for any continuous mapping $f: \mathbb{R}^d \to \mathbb{R}^d$, compact domain $\Omega$ and $\varepsilon>0$, there is a sequence of mappings $\phi_{i_1}, ..., \phi_{i_m} \in V, m \in \mathbb{Z}_+$, such that the composition $\phi_{i_m
    
[^4]: GraVAC：适应性压缩用于高效的分布式深度学习训练

    GraVAC: Adaptive Compression for Communication-Efficient Distributed DL Training. (arXiv:2305.12201v1 [cs.LG])

    [http://arxiv.org/abs/2305.12201](http://arxiv.org/abs/2305.12201)

    GraVAC提出了一个动态调整压缩因子的框架，通过评估模型进展和评估与压缩相关的梯度信息损失来进行训练。GraVAC可以在不需要任何关于模型或其超参数的先前假设的情况下，达到与先前最先进的压缩方法相同或更好的翻译准确性。在CIFAR-10和ImageNet数据集上，相对于静态压缩对应物，GraVAC可以将通信减少高达87％和75％。

    

    分布式数据并行（DDP）训练通过多个设备在数据子集上进行训练并聚合更新来提高应用程序的整体吞吐量。每次迭代的周期性同步产生了相当大的开销，受最先进的神经网络越来越大和复杂的影响更加严重。尽管许多梯度压缩技术提出了减少通信成本的方式，但最佳压缩因子导致最大速度提升或最小数据交换的问题仍然是一个开放式的问题，因为它与压缩质量、模型大小和结构、硬件、网络拓扑和带宽有关。我们提出了GraVAC，一个动态调整压缩因子的框架，通过评估模型进展和评估与压缩相关的梯度信息损失来进行训练。GraVAC以在线的黑盒方式工作，不需要任何关于模型或其超参数的先前假设，同时实现与先前最先进的压缩方法相同或更好的翻译准确性。对于CIFAR-10和ImageNet数据集，相对于其静态压缩对应物，GraVAC将通信减少了高达87％和75％，而收敛精度相同。

    Distributed data-parallel (DDP) training improves overall application throughput as multiple devices train on a subset of data and aggregate updates to produce a globally shared model. The periodic synchronization at each iteration incurs considerable overhead, exacerbated by the increasing size and complexity of state-of-the-art neural networks. Although many gradient compression techniques propose to reduce communication cost, the ideal compression factor that leads to maximum speedup or minimum data exchange remains an open-ended problem since it varies with the quality of compression, model size and structure, hardware, network topology and bandwidth. We propose GraVAC, a framework to dynamically adjust compression factor throughout training by evaluating model progress and assessing gradient information loss associated with compression. GraVAC works in an online, black-box manner without any prior assumptions about a model or its hyperparameters, while achieving the same or better
    
[^5]: 我们需要编码器-解码器来模拟网络上的动力系统吗？

    Do We Need an Encoder-Decoder to Model Dynamical Systems on Networks?. (arXiv:2305.12185v1 [cs.LG])

    [http://arxiv.org/abs/2305.12185](http://arxiv.org/abs/2305.12185)

    本论文揭示了使用潜在嵌入来模拟网络动力系统的不足，并提出了一种无嵌入的替代方案，通过三个正确长期行为测试的验证表明该方案的可行性。

    

    随着深度学习在模拟动力系统方面的普及，我们揭示了一个与网络动力学建模相关且未被重视的误解。受图神经网络的强烈影响，潜在顶点嵌入被自然地采用在许多神经动力网络模型中。然而，我们证明了嵌入倾向于产生适应观测良好但同时具有错误动力行为的模型。我们提出了三个正确长期行为的测试，并说明了嵌入式动力模型如何未通过这些测试，并分析其中的原因，特别是通过拓扑共轭的角度。通过这样做，我们表明避免使用嵌入可以避免困难。我们提出了一种简单的基于参数化两个加性矢量场分量的无嵌入替代方案。通过广泛的实验，我们验证了所提出的替代方案的优越性。

    As deep learning gains popularity in modelling dynamical systems, we expose an underappreciated misunderstanding relevant to modelling dynamics on networks. Strongly influenced by graph neural networks, latent vertex embeddings are naturally adopted in many neural dynamical network models. However, we show that embeddings tend to induce a model that fits observations well but simultaneously has incorrect dynamical behaviours. Recognising that previous studies narrowly focus on short-term predictions during the transient phase of a flow, we propose three tests for correct long-term behaviour, and illustrate how an embedding-based dynamical model fails these tests, and analyse the causes, particularly through the lens of topological conjugacy. In doing so, we show that the difficulties can be avoided by not using embedding. We propose a simple embedding-free alternative based on parametrising two additive vector-field components. Through extensive experiments, we verify that the proposed
    
[^6]: 基于梯度说明的表示法去偏见模型

    Model Debiasing via Gradient-based Explanation on Representation. (arXiv:2305.12178v1 [cs.LG])

    [http://arxiv.org/abs/2305.12178](http://arxiv.org/abs/2305.12178)

    本文提出了一种新的公平性框架，通过梯度说明找到两个模型焦点进行去偏见处理，提高下游任务模型的预测性能。

    

    机器学习系统会对某些人口统计学群体产生偏见，即不公平现象。近期的解决方法是通过分离式表示学习学习潜在码（即表示法），然后丢弃与敏感属性（如性别）相关的码。但这些方法在处理现实数据（特别是非结构化数据）时，可能会遗漏代理属性（敏感属性的代理），并且受到不完全分离的影响，导致公平性能下降和下游任务中损失有用信息。本文提出了一种新的公平性框架，针对敏感属性和代理属性进行去偏见处理，提高下游任务模型的预测性能而不需要完全分离。主要思路是利用梯度说明找到两个模型焦点：1）其中一个焦点用于预测值，2）另一个焦点用于代理属性，然后对潜在码进行修正以减轻这些属性之间的相关性。

    Machine learning systems produce biased results towards certain demographic groups, known as the fairness problem. Recent approaches to tackle this problem learn a latent code (i.e., representation) through disentangled representation learning and then discard the latent code dimensions correlated with sensitive attributes (e.g., gender). Nevertheless, these approaches may suffer from incomplete disentanglement and overlook proxy attributes (proxies for sensitive attributes) when processing real-world data, especially for unstructured data, causing performance degradation in fairness and loss of useful information for downstream tasks. In this paper, we propose a novel fairness framework that performs debiasing with regard to both sensitive attributes and proxy attributes, which boosts the prediction performance of downstream task models without complete disentanglement. The main idea is to, first, leverage gradient-based explanation to find two model focuses, 1) one focus for predicti
    
[^7]: 一种可扩展的神经网络用于DSIC仿射极大价拍卖设计

    A Scalable Neural Network for DSIC Affine Maximizer Auction Design. (arXiv:2305.12162v1 [cs.GT])

    [http://arxiv.org/abs/2305.12162](http://arxiv.org/abs/2305.12162)

    该论文提出了一种可扩展的神经网络AMenuNet来构造AMAs参数和生成候选分配，解决了现有方法在占优策略激励兼容性和可扩展性方面的限制，其在协商一致的价值和社会残余价值方面优于强基线模型。

    

    自动拍卖设计旨在通过机器学习寻找经验上高收入的机制。现有的多物品拍卖情景的工作可以粗略地分为RegretNet类和仿射极大价（AMAs）方法。然而，前者不能严格保证占优策略激励兼容性（DSIC），而后者因为分配候选人数过多而面临可扩展性问题。为解决这些限制，我们提出了AMenuNet，一种可扩展的神经网络，它从出价人和物品表示中构造AMA参数（甚至包括分配菜单）。由于AMA的属性，AMenuNet始终是DSIC和个人理性（IR）的，通过神经网络生成候选分配来增强可伸缩性。此外，AMenuNet是置换等变的，其参数数量不受拍卖规模的影响。我们进行了大量实验，证明AMenuNet在协商一致的价值和社会残余价值方面优于强基线模型。

    Automated auction design aims to find empirically high-revenue mechanisms through machine learning. Existing works on multi item auction scenarios can be roughly divided into RegretNet-like and affine maximizer auctions (AMAs) approaches. However, the former cannot strictly ensure dominant strategy incentive compatibility (DSIC), while the latter faces scalability issue due to the large number of allocation candidates. To address these limitations, we propose AMenuNet, a scalable neural network that constructs the AMA parameters (even including the allocation menu) from bidder and item representations. AMenuNet is always DSIC and individually rational (IR) due to the properties of AMAs, and it enhances scalability by generating candidate allocations through a neural network. Additionally, AMenuNet is permutation equivariant, and its number of parameters is independent of auction scale. We conduct extensive experiments to demonstrate that AMenuNet outperforms strong baselines in both co
    
[^8]: 基于模型的适应性强化学习控制在参数变化系统的样本有效迁移中的应用

    Model-based adaptation for sample efficient transfer in reinforcement learning control of parameter-varying systems. (arXiv:2305.12158v1 [eess.SY])

    [http://arxiv.org/abs/2305.12158](http://arxiv.org/abs/2305.12158)

    本文提出一种基于模型的方法来解决强化学习中样本有效性问题，通过模型转换实现正向迁移，并用作强化学习过程的初始化以达到新的最优值。

    

    本文提出基于模型的控制方法来解决强化学习算法的样本有效性问题。我们提出一种模型转换方法，通过将控制策略应用于目标系统，实现正向迁移，从而为强化学习过程提供初始化，以便其收敛到新的最优值。

    In this paper, we leverage ideas from model-based control to address the sample efficiency problem of reinforcement learning (RL) algorithms. Accelerating learning is an active field of RL highly relevant in the context of time-varying systems. Traditional transfer learning methods propose to use prior knowledge of the system behavior to devise a gradual or immediate data-driven transformation of the control policy obtained through RL. Such transformation is usually computed by estimating the performance of previous control policies based on measurements recently collected from the system. However, such retrospective measures have debatable utility with no guarantees of positive transfer in most cases. Instead, we propose a model-based transformation, such that when actions from a control policy are applied to the target system, a positive transfer is achieved. The transformation can be used as an initialization for the reinforcement learning process to converge to a new optimum. We va
    
[^9]: 机器学习需要受教育吗？——论算法教育而非训练 (arXiv:2305.12157v1 [cs.LG])

    (Machine) Learning to Be Like Thee? For Algorithm Education, Not Training. (arXiv:2305.12157v1 [cs.LG])

    [http://arxiv.org/abs/2305.12157](http://arxiv.org/abs/2305.12157)

    本文认为，机器学习算法需要接受算法教育，以改进其决策道德，而非仅仅停留在训练上。对于AI伦理决策，解决方案在于对ML进行算法教育。机器学习算法并不是与我们的人类特质分开而存在的，而是我们最内在的偏见和偏见的一种体现。

    

    本文认为机器学习（ML）算法必须要受教育。ML训练的算法的道德决策在人类社会中随处可见，有时会逆转政府、非政府组织和公民社会在过去几十年中取得的社会进步，或者仍在努力实现这些进步。尽管这些算法的决策对人类社会具有不可估量的影响，但它们却是已知的最无知的代理机构之一（数据不完整、不包容或有偏见）。ML算法并不是与我们的人类特质分开而存在的，而是我们最内在的偏见和偏见的一种体现。一些研究致力于责任分配作为解决非道德AI行为的策略。然而，本文认为，AI伦理决策的解决方案在于对ML进行算法教育（而非训练）。从ML和儿童社会责任教育之间的类比中，本文提供了负责任和可持续发展的明确方向。

    This paper argues that Machine Learning (ML) algorithms must be educated. ML-trained algorithms moral decisions are ubiquitous in human society. Sometimes reverting the societal advances governments, NGOs and civil society have achieved with great effort in the last decades or are yet on the path to be achieved. While their decisions have an incommensurable impact on human societies, these algorithms are within the least educated agents known (data incomplete, un-inclusive, or biased). ML algorithms are not something separate from our human idiosyncrasy but an enactment of our most implicit prejudices and biases. Some research is devoted to responsibility assignment as a strategy to tackle immoral AI behaviour. Yet this paper argues that the solution for AI ethical decision-making resides in algorithm education (as opposed to the training) of ML. Drawing from an analogy between ML and child education for social responsibility, the paper offers clear directions for responsible and susta
    
[^10]: 在潜空间中使用 Langevin 动力学的归一化流采样

    Normalizing flow sampling with Langevin dynamics in the latent space. (arXiv:2305.12149v1 [stat.ML])

    [http://arxiv.org/abs/2305.12149](http://arxiv.org/abs/2305.12149)

    本文提出了一种在潜在空间中使用 Langevin 动力学的采样方法，以克服归一化流可能面临的复杂分布问题，并能够轻松地融入任何 NF 结构中。

    

    归一化流（NF）使用连续生成器将简单的潜在分布（例如高斯分布）映射到与训练数据集相关联的经验目标分布。通过最小化变分目标来训练后，学习到的映射提供了目标分布的近似生成模型。本文提出了一种新的马尔可夫链蒙特卡罗算法，将目标分布在潜域中采样，然后将其传输回目标域，以克服当应对复杂分布时可能出现的问题。该方法依赖于潜空间中的 Metropolis 调整 Langevin 动力学，并且可以轻松地融入任何 NF 结构中。我们展示了我们的方法在玩具和真实数据集上的有效性。

    Normalizing flows (NF) use a continuous generator to map a simple latent (e.g. Gaussian) distribution, towards an empirical target distribution associated with a training data set. Once trained by minimizing a variational objective, the learnt map provides an approximate generative model of the target distribution. Since standard NF implement differentiable maps, they may suffer from pathological behaviors when targeting complex distributions. For instance, such problems may appear for distributions on multi-component topologies or characterized by multiple modes with high probability regions separated by very unlikely areas. A typical symptom is the explosion of the Jacobian norm of the transformation in very low probability areas. This paper proposes to overcome this issue thanks to a new Markov chain Monte Carlo algorithm to sample from the target distribution in the latent domain before transporting it back to the target domain. The approach relies on a Metropolis adjusted Langevin
    
[^11]: 通过与大型语言模型的查询学习Horn包络

    Learning Horn Envelopes via Queries from Large Language Models. (arXiv:2305.12143v1 [cs.LG])

    [http://arxiv.org/abs/2305.12143](http://arxiv.org/abs/2305.12143)

    该研究提出了一种从经过训练的神经网络中提取知识的方法，可以学习到最紧Horn逼近目标理论的新算法，并在预训练的语言模型中运用于揭示基于职业的性别偏见规则。

    

    我们研究了一种从训练的神经网络中提取知识的方法，该方法基于Angluin的精确学习模型，使用成员和等价性查询到一个oracle。在这种方法中，oracle是一个经过训练的神经网络。我们考虑了Angluin用于学习Horn理论的经典算法，并研究了必要的变化，以使其适用于从神经网络中学习。特别地，我们必须考虑到经过训练的神经网络可能不会像Horn oracle那样行事，这意味着它们的潜在目标理论可能不是Horn。我们提出了一种旨在提取目标理论“最紧Horn逼近”的新算法，并保证在指数时间（在最坏情况下）内终止，在目标具有多项式数量的非Horn示例的情况下，在多项式时间内终止。为了展示这种方法的适用性，我们对预训练的语言模型进行实验，提取揭示基于职业的性别偏见的规则。

    We investigate an approach for extracting knowledge from trained neural networks based on Angluin's exact learning model with membership and equivalence queries to an oracle. In this approach, the oracle is a trained neural network. We consider Angluin's classical algorithm for learning Horn theories and study the necessary changes to make it applicable to learn from neural networks. In particular, we have to consider that trained neural networks may not behave as Horn oracles, meaning that their underlying target theory may not be Horn. We propose a new algorithm that aims at extracting the ``tightest Horn approximation'' of the target theory and that is guaranteed to terminate in exponential time (in the worst case) and in polynomial time if the target has polynomially many non-Horn examples. To showcase the applicability of the approach, we perform experiments on pre-trained language models and extract rules that expose occupation-based gender biases.
    
[^12]: 多模式联邦人类活动识别中的隐私问题

    Privacy in Multimodal Federated Human Activity Recognition. (arXiv:2305.12134v1 [cs.LG])

    [http://arxiv.org/abs/2305.12134](http://arxiv.org/abs/2305.12134)

    本文研究了多模式联邦人类活动识别中的隐私问题。通过一个特定的系统，联邦学习可以提供更好的隐私保护，同时不会损失人类活动识别的准确性。

    

    人类活动识别（HAR）的训练数据往往包含隐私信息或由不合作实体持有。联邦学习（FL）通过在边缘设备上训练机器学习模型来解决这些问题。本文研究了在用户、环境和传感器级别上隐私对联邦HAR的影响。我们表明，FL对HAR的性能取决于FL系统的隐私保护程度，并且主要取决于来自不同传感器的数据的配置。尽管避免数据共享并在人类或环境级别上假设隐私，如之前的工作所做的那样，精度会降低5-7％。然而，将这种隐私延伸到模态级别并严格分离多个客户端之间的传感器数据可能会导致精度降低19-42％。由于这种形式的隐私是HAR中被要求的道德利用被动传感方法所必需的，因此我们实现了一种系统，在该系统中客户端相互训练一个通用的FL模型和一个每种模态一个的组级模型。我们的评估表明，这种方法可以在不牺牲HAR准确性的情况下提高隐私保护。

    Human Activity Recognition (HAR) training data is often privacy-sensitive or held by non-cooperative entities. Federated Learning (FL) addresses such concerns by training ML models on edge clients. This work studies the impact of privacy in federated HAR at a user, environment, and sensor level. We show that the performance of FL for HAR depends on the assumed privacy level of the FL system and primarily upon the colocation of data from different sensors. By avoiding data sharing and assuming privacy at the human or environment level, as prior works have done, the accuracy decreases by 5-7%. However, extending this to the modality level and strictly separating sensor data between multiple clients may decrease the accuracy by 19-42%. As this form of privacy is necessary for the ethical utilisation of passive sensing methods in HAR, we implement a system where clients mutually train both a general FL model and a group-level one per modality. Our evaluation shows that this method leads to
    
[^13]: 神经网络训练中的损失峰值研究

    Loss Spike in Training Neural Networks. (arXiv:2305.12133v1 [cs.LG])

    [http://arxiv.org/abs/2305.12133](http://arxiv.org/abs/2305.12133)

    本文研究神经网络训练过程中损失值峰值现象的机制，并发现最大特征值的第一个特征向量的偏差主要受低频成分占据。低频成分可以被训练数据和测试数据很好地捕获，所以导致具有良好和劣质泛化能力的解决方案都可以很好地学习低频成分，但劣质泛化能力的解决方案可能会过度拟合高频成分，良好泛化能力的解决方案具有更平滑的损失函数。

    

    本文研究了神经网络训练过程中出现的损失值峰值现象所背后的机制。研究发现，在具有较小损失的区域，一旦训练进入该区域，训练就会变得不稳定，损失值呈指数式增长，即出现损失峰值现象。当训练进入平坦区域时，训练会变得稳定。本研究发现，损失Hessian矩阵的最大特征值（$\lambda_{\mathrm{max}}$）的第一个特征向量的偏差主要由低频成分占据。由于低频成分可以非常快速地被捕获（频率原理），因此会出现急剧下降的现象。在分析损失峰值的基础上，我们重新审视了$\lambda_{\mathrm{max}}$平坦性和泛化能力之间的关系。对于实际数据集，低频往往占据主导地位，并且可以被训练数据和测试数据所很好地捕获。因此，具有良好泛化能力和具有劣质泛化能力的解决方案都可以很好地学习低频成分，因此它们在损失函数上的性质相似。但是，劣质泛化能力的解决方案可能会过度拟合高频成分，而良好泛化能力的解决方案具有更平滑的损失函数。

    In this work, we study the mechanism underlying loss spikes observed during neural network training. When the training enters a region, which has a smaller-loss-as-sharper (SLAS) structure, the training becomes unstable and loss exponentially increases once it is too sharp, i.e., the rapid ascent of the loss spike. The training becomes stable when it finds a flat region. The deviation in the first eigen direction (with maximum eigenvalue of the loss Hessian ($\lambda_{\mathrm{max}}$) is found to be dominated by low-frequency. Since low-frequency is captured very fast (frequency principle), the rapid descent is then observed. Inspired by our analysis of loss spikes, we revisit the link between $\lambda_{\mathrm{max}}$ flatness and generalization. For real datasets, low-frequency is often dominant and well-captured by both the training data and the test data. Then, a solution with good generalization and a solution with bad generalization can both learn low-frequency well, thus, they hav
    
[^14]: 公共大型语言模型能否帮助私有交叉设备联邦学习？

    Can Public Large Language Models Help Private Cross-device Federated Learning?. (arXiv:2305.12132v1 [cs.LG])

    [http://arxiv.org/abs/2305.12132](http://arxiv.org/abs/2305.12132)

    本文探讨在差分私有联邦学习中如何利用大型公共语言模型提升隐私和效用权衡，并提出一种分布匹配算法提高公共数据的训练效率和隐私性，为训练私有模型提供有效方法。

    

    本文研究了（差分）私有联邦学习（FL）中的语言模型。交叉设备FL中的语言模型相对较小，在训练中的大规模并行性参与下可以使用有意义的形式化用户级差分隐私（DP）保证进行训练。最近，公共数据已用于改善大型和小型语言模型的隐私-效用权衡。在本研究中，我们系统地研究了使用大规模公共数据和LLMs来帮助设备上的FL模型进行差分私有训练，并通过蒸馏技术进一步改善隐私-效用权衡。此外，我们提出了一种新颖的分布匹配算法，通过理论基础对公共数据进行接近私有数据分布的采样，显著提高了（预）训练公共数据的样本效率。所提出的方法是通过利用公共大型语言模型训练私有模型的高效和有效方法。

    We study (differentially) private federated learning (FL) of language models. The language models in cross-device FL are relatively small, which can be trained with meaningful formal user-level differential privacy (DP) guarantees when massive parallelism in training is enabled by the participation of a moderate size of users. Recently, public data has been used to improve privacy-utility trade-offs for both large and small language models. In this work, we provide a systematic study of using large-scale public data and LLMs to help differentially private training of on-device FL models, and further improve the privacy-utility tradeoff by techniques of distillation. Moreover, we propose a novel distribution matching algorithm with theoretical grounding to sample public data close to private data distribution, which significantly improves the sample efficiency of (pre-)training on public data. The proposed method is efficient and effective for training private model by taking advantage 
    
[^15]: 任意时延的非稳态在线凸优化

    Non-stationary Online Convex Optimization with Arbitrary Delays. (arXiv:2305.12131v1 [cs.LG])

    [http://arxiv.org/abs/2305.12131](http://arxiv.org/abs/2305.12131)

    本文研究了任意时延的非稳态在线凸优化，提出了一种简单的算法DOGD，并证明它能在最坏情况下获得$O(\sqrt{dT}(P_T+1))$的动态遗憾界，同时当延迟不改变梯度到达顺序时，自动将动态遗憾减少到$O(\sqrt{S}(1+P_T))$。

    

    最近，以梯度或其他函数信息可以任意延迟为特点的在线凸优化（OCO）引起了越来越多的关注。与之前研究稳态环境的研究不同，本文研究了非稳态环境下的延迟OCO，并旨在最小化与任何比较器序列相关的动态遗憾。为此，我们首先提出了一个简单的算法，即DOGD，该算法根据其到达顺序为每个延迟梯度执行渐变下降步骤。尽管它很简单，但我们的新型分析表明，DOGD可以在最坏情况下获得$O(\sqrt{dT}(P_T+1))$的动态遗憾界，其中$d$是最大延迟，$T$是时间跨度，$P_T$是比较器的路径长度。更重要的是，在延迟不改变渐变的到达顺序的情况下，它可以自动将动态遗憾减少到$O(\sqrt{S}(1+P_T))$，其中$S$是延迟之和。此外，我们将DOGD扩展为更通用的算法，并证明它实现了与DOGD相同的遗憾界。广泛的模拟表明了所提出算法的有效性和效率。

    Online convex optimization (OCO) with arbitrary delays, in which gradients or other information of functions could be arbitrarily delayed, has received increasing attention recently. Different from previous studies that focus on stationary environments, this paper investigates the delayed OCO in non-stationary environments, and aims to minimize the dynamic regret with respect to any sequence of comparators. To this end, we first propose a simple algorithm, namely DOGD, which performs a gradient descent step for each delayed gradient according to their arrival order. Despite its simplicity, our novel analysis shows that DOGD can attain an $O(\sqrt{dT}(P_T+1)$ dynamic regret bound in the worst case, where $d$ is the maximum delay, $T$ is the time horizon, and $P_T$ is the path length of comparators. More importantly, in case delays do not change the arrival order of gradients, it can automatically reduce the dynamic regret to $O(\sqrt{S}(1+P_T))$, where $S$ is the sum of delays. Furtherm
    
[^16]: 解除预训练语言模型中的容量差异诅咒

    Lifting the Curse of Capacity Gap in Distilling Language Models. (arXiv:2305.12129v1 [cs.CL])

    [http://arxiv.org/abs/2305.12129](http://arxiv.org/abs/2305.12129)

    本文提出了一种新的知识蒸馏方法（MiniMoE），通过增加学生的容量而不明显增加推理计算解除容量差异诅咒，并在GLUE和CoNLL上进行了实验验证。

    

    预训练语言模型在各种下游任务中表现出色，但不幸的是，它们需要大量的推理计算。知识蒸馏通过师生范式为小型模型压缩预训练语言模型，但当师生之间的容量差距很大时，容量差距诅咒会出现，导致蒸馏语言模型不足。虽然已有几项研究填补了这一差距，但诅咒仍未得到很好的解决。在本文中，我们旨在通过增加学生的容量而不明显增加推理计算来解除容量差异诅咒。受混合专家(Sparse Activation Regime of Mixture of Experts (MoE))的启发，我们提出了最小专家(MiniMoE)的混合物，这为学生引入了额外的参数，但几乎没有引入任何额外的推理计算。在GLUE和CoNLL上的实验结果表明，MiniMoE的魔力消除了容量差距诅咒。

    Pretrained language models (LMs) have shown compelling performance on various downstream tasks, but unfortunately they require a tremendous amount of inference compute. Knowledge distillation finds a path to compress LMs to small ones with a teacher-student paradigm. However, when the capacity gap between the teacher and the student is large, a curse of capacity gap appears, invoking a deficiency in distilling LMs. While a few studies have been carried out to fill the gap, the curse is not yet well tackled. In this paper, we aim at lifting the curse of capacity gap via enlarging the capacity of the student without notably increasing the inference compute. Largely motivated by sparse activation regime of mixture of experts (MoE), we propose a mixture of minimal experts (MiniMoE), which imposes extra parameters to the student but introduces almost no additional inference compute. Experimental results on GLUE and CoNLL demonstrate the curse of capacity gap is lifted by the magic of MiniMo
    
[^17]: ACA-Net: 采用不对称交叉注意力的轻量级说话人验证方法

    ACA-Net: Towards Lightweight Speaker Verification using Asymmetric Cross Attention. (arXiv:2305.12121v1 [cs.SD])

    [http://arxiv.org/abs/2305.12121](http://arxiv.org/abs/2305.12121)

    本文提出了一种轻量级的全局上下文感知说话人嵌入提取器，使用不对称交叉注意力(ACA)代替时间池化，具有高效的全局特征提取和适应时间变化的优点。

    

    本文提出ACA-Net，一种轻量级的全局上下文感知说话人嵌入提取器，用于说话人验证。ACA-Net使用不对称交叉注意力(ACA)代替时间池化，将大的变长序列压缩成小的固定大小的向量。在ACA-Net中，我们构建了一个多层汇聚块，使用ACA从变长度输入中生成固定长度的身份向量。与现有的SV模型不同，ACA-Net通过全局注意力作为有效的全局特征提取器，适应时间变化而不是应用于时间维度的固定函数。我们在WSJ0-1talker数据集上的实验表明，仅使用1/5的参数，ACA-Net在EER方面相对于强基线提高了5%。

    In this paper, we propose ACA-Net, a lightweight, global context-aware speaker embedding extractor for Speaker Verification (SV) that improves upon existing work by using Asymmetric Cross Attention (ACA) to replace temporal pooling. ACA is able to distill large, variable-length sequences into small, fixed-sized latents by attending a small query to large key and value matrices. In ACA-Net, we build a Multi-Layer Aggregation (MLA) block using ACA to generate fixed-sized identity vectors from variable-length inputs. Through global attention, ACA-Net acts as an efficient global feature extractor that adapts to temporal variability unlike existing SV models that apply a fixed function for pooling over the temporal dimension which may obscure information about the signal's non-stationary temporal variability. Our experiments on the WSJ0-1talker show ACA-Net outperforms a strong baseline by 5\% relative improvement in EER using only 1/5 of the parameters.
    
[^18]: 退火自蒸馏校正改进了对抗训练

    Annealing Self-Distillation Rectification Improves Adversarial Training. (arXiv:2305.12118v1 [cs.LG])

    [http://arxiv.org/abs/2305.12118](http://arxiv.org/abs/2305.12118)

    本研究提出了退火自蒸馏校正(ADR)方法，其能生成软标签用作更好的指导机制，准确反映在对抗训练中攻击下的分布变化，提高模型的鲁棒性，并实现了平滑的插入性整合到其他对抗性训练技术中。

    

    标准的对抗训练中，模型被优化以适应可接受的对抗扰动预算内的一热标签。然而，忽略由扰动带来的基础分布变化，导致了强健的过拟合问题。为了解决这个问题，增强对抗性鲁棒性，我们分析了强健模型的特征，并确定强健模型倾向于生成更平滑和更良好校准的输出。基于这一观测结果，我们提出了一种简单而有效的方法——退火自蒸馏校正(ADR)，该方法生成软标签作为更好的指导机制，能准确反映在对抗训练中攻击下的分布变化。通过使用ADR，我们可以获得修正的分布，显著改善模型的鲁棒性，而不需要预训练模型或额外的计算。此外，我们的方法通过替换卷积层以实现平滑的插入性整合到其他对抗性训练技术中。

    In standard adversarial training, models are optimized to fit one-hot labels within allowable adversarial perturbation budgets. However, the ignorance of underlying distribution shifts brought by perturbations causes the problem of robust overfitting. To address this issue and enhance adversarial robustness, we analyze the characteristics of robust models and identify that robust models tend to produce smoother and well-calibrated outputs. Based on the observation, we propose a simple yet effective method, Annealing Self-Distillation Rectification (ADR), which generates soft labels as a better guidance mechanism that accurately reflects the distribution shift under attack during adversarial training. By utilizing ADR, we can obtain rectified distributions that significantly improve model robustness without the need for pre-trained models or extensive extra computation. Moreover, our method facilitates seamless plug-and-play integration with other adversarial training techniques by repl
    
[^19]: GFDC：一种基于证据推理的颗粒聚类算法

    GFDC: A Granule Fusion Density-Based Clustering with Evidential Reasoning. (arXiv:2305.12114v1 [cs.LG])

    [http://arxiv.org/abs/2305.12114](http://arxiv.org/abs/2305.12114)

    提出了一种基于证据推理的颗粒聚类算法GFDC，该算法可以克服基于密度的聚类算法在衡量全局密度、确定合理的聚类中心或结构、准确地分配样本以及处理具有大密度差异的数据方面表现不佳的缺点。与此同时，GFDC 采用了三种新颖的颗粒融合策略来将颗粒组合成稳定的聚类结构，有助于检测具有任意形状的聚类。

    

    目前，基于密度的聚类算法因其能够检测具有任意形状的类群而被广泛应用。然而，它们在衡量全局密度、确定合理的聚类中心或结构、准确地分配样本以及处理具有大密度差异的数据方面表现不佳。为了克服它们的缺点，本文提出了一种基于证据推理的颗粒聚类算法（GFDC）。首先，通过一种稀疏度量指标来测量样本的局部和全局密度。然后在高密度和低密度区域中生成信息颗粒，帮助处理具有显著密度差异的聚类。此外，采用了三种新颖的颗粒融合策略来将颗粒组合成稳定的聚类结构，有助于检测具有任意形状的聚类。最后，通过基于Dempster-Shafer理论开发的分配方法来分配不稳定的样本。使用GFDC后，可以得到合理的聚类结果。

    Currently, density-based clustering algorithms are widely applied because they can detect clusters with arbitrary shapes. However, they perform poorly in measuring global density, determining reasonable cluster centers or structures, assigning samples accurately and handling data with large density differences among clusters. To overcome their drawbacks, this paper proposes a granule fusion density-based clustering with evidential reasoning (GFDC). Both local and global densities of samples are measured by a sparse degree metric first. Then information granules are generated in high-density and low-density regions, assisting in processing clusters with significant density differences. Further, three novel granule fusion strategies are utilized to combine granules into stable cluster structures, helping to detect clusters with arbitrary shapes. Finally, by an assignment method developed from Dempster-Shafer theory, unstable samples are assigned. After using GFDC, a reasonable clustering
    
[^20]: 元神经协调

    Meta Neural Coordination. (arXiv:2305.12109v1 [cs.LG])

    [http://arxiv.org/abs/2305.12109](http://arxiv.org/abs/2305.12109)

    元神经协调是为了解决元学习中如何表示和推理其他学习算法在不同环境下操作和表现以及传统深度神经网络在预测中的不确定性问题而提出的新方法。

    

    元学习旨在开发能够从其他学习算法中学习以适应新的和不断变化的环境的算法。这需要模型来描述其他学习算法在不同环境下的操作和表现，类似于在心理理论中表达和推理心理状态。此外，传统深度神经网络预测的不确定性问题凸显了世界的局部可预知性，需要对多个预测同时进行表示。神经模块之间的协调促进了不同的模块之间相互认知其信念和渴望的表示。模块化和分散的神经网络之间的神经协调是构建能够灵活和适应性地交互的自主智能机器的基本先决条件。本文提出了几个证据，证明一种解决以上问题的新途径，称为元神经协调。

    Meta-learning aims to develop algorithms that can learn from other learning algorithms to adapt to new and changing environments. This requires a model of how other learning algorithms operate and perform in different contexts, which is similar to representing and reasoning about mental states in the theory of mind. Furthermore, the problem of uncertainty in the predictions of conventional deep neural networks highlights the partial predictability of the world, requiring the representation of multiple predictions simultaneously. This is facilitated by coordination among neural modules, where different modules' beliefs and desires are attributed to others. The neural coordination among modular and decentralized neural networks is a fundamental prerequisite for building autonomous intelligence machines that can interact flexibly and adaptively. In this work, several pieces of evidence demonstrate a new avenue for tackling the problems above, termed Meta Neural Coordination. We discuss th
    
[^21]: 统一嵌入：面向 Web 规模 ML 系统的经过验证的特征表示

    Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems. (arXiv:2305.12102v1 [cs.LG])

    [http://arxiv.org/abs/2305.12102](http://arxiv.org/abs/2305.12102)

    本文介绍了一种名为“特征复用”的框架，它使用单一的表示空间 能够高效有效地学习高质量的特征嵌入，同时区分不同的分类特征。通过在多个公共数据集和新数据集“Web-Available Image Search (WAIS)”上的测试，我们展示了这种方法的优于现有技术的表现。

    

    高效、有效地学习高质量的特征嵌入对于 Web 规模的机器学习系统的性能至关重要。标准方法是将每个特征值表示为一个 d 维嵌入，引入数百亿个参数，而这些特征的基数非常高。这个瓶颈导致了备选嵌入算法的重大进展。本文介绍了一个简单但非常有效的框架，即“特征复用”，在许多不同的分类特征之间使用一个单一的表示空间。我们的理论和实证分析表明，复用的嵌入可以分解为每个组成特征的组件，使得模型可以区分特征。我们展示了复用的嵌入在几个公共数据集上优于现有技术。此外，我们引入了一个名为“Web-Available Image Search (WAIS)”的新数据集，以严格评估 Web 规模下的新嵌入算法。我们邀请社区通过提出可以准确、高效地将数百万张图像嵌入和分类到成千上万个类别的新模型来贡献 WAIS 挑战。

    Learning high-quality feature embeddings efficiently and effectively is critical for the performance of web-scale machine learning systems. A typical model ingests hundreds of features with vocabularies on the order of millions to billions of tokens. The standard approach is to represent each feature value as a d-dimensional embedding, introducing hundreds of billions of parameters for extremely high-cardinality features. This bottleneck has led to substantial progress in alternative embedding algorithms. Many of these methods, however, make the assumption that each feature uses an independent embedding table. This work introduces a simple yet highly effective framework, Feature Multiplexing, where one single representation space is used across many different categorical features. Our theoretical and empirical analysis reveals that multiplexed embeddings can be decomposed into components from each constituent feature, allowing models to distinguish between features. We show that multip
    
[^22]: 稳定性、泛化性和隐私保护：对于随机特征和NTK特征的精确分析

    Stability, Generalization and Privacy: Precise Analysis for Random and NTK Features. (arXiv:2305.12100v1 [stat.ML])

    [http://arxiv.org/abs/2305.12100](http://arxiv.org/abs/2305.12100)

    本论文研究了ERM训练模型对抗强大黑盒攻击的安全问题，并通过两个指标量化模型安全性：单个样本的稳定性和查询与原始数据特征的对齐。在研究中，通过研究RF和NTK回归，证明随着泛化能力的提高，隐私保护可以得到加强。

    

    深度学习模型容易受到恢复攻击，引起用户隐私保护的担忧。针对经验风险最小化（ERM）等常见算法通常不能直接实施安全保障的问题，本文研究了ERM训练模型对抗特定强大黑盒子攻击的安全问题。我们的分析通过两个看似不同但有联系的指标来量化模型安全性：一是相对于单个训练样本的模型稳定性，另一个是攻击查询和原始数据特征的特征对齐。虽然前者在学习理论中已经得到了很好的阐述，并与经典工作中的泛化误差相关，但在我们的研究中，第二种特性是新颖的。我们的关键技术结果为两种原型设置提供了特征对齐的精确刻画：随机特征（RF）和神经切向核（NTK）回归。这证明，随着泛化能力的提高，隐私保护能够得到加强，同时揭示了其他有趣的性质。

    Deep learning models can be vulnerable to recovery attacks, raising privacy concerns to users, and widespread algorithms such as empirical risk minimization (ERM) often do not directly enforce safety guarantees. In this paper, we study the safety of ERM-trained models against a family of powerful black-box attacks. Our analysis quantifies this safety via two separate terms: (i) the model stability with respect to individual training samples, and (ii) the feature alignment between the attacker query and the original data. While the first term is well established in learning theory and it is connected to the generalization error in classical work, the second one is, to the best of our knowledge, novel. Our key technical result provides a precise characterization of the feature alignment for the two prototypical settings of random features (RF) and neural tangent kernel (NTK) regression. This proves that privacy strengthens with an increase in the generalization capability, unveiling also
    
[^23]: 使Transformer在时间序列预测中再次卓越：通道对齐鲁棒双Transformer

    Make Transformer Great Again for Time Series Forecasting: Channel Aligned Robust Dual Transformer. (arXiv:2305.12095v1 [cs.LG])

    [http://arxiv.org/abs/2305.12095](http://arxiv.org/abs/2305.12095)

    本文提出了一种通道对齐鲁棒双Transformer模型，通过双Transformer结构和鲁棒损失函数的引入，解决了Transformer在时间序列预测中的关键缺点，显著提高了预测精度和效率。

    

    最近的研究表明，深度学习方法，尤其是Transformer和MLP，在时间序列预测方面具有巨大的优势。尽管在NLP和CV方面获得了成功，但许多研究发现，与MLP相比，Transformer在时间序列预测方面的效果不佳。在本文中，我们设计了一种特殊的Transformer，即通道对齐鲁棒双Transformer（CARD），以解决Transformer在时间序列预测中的关键缺点。首先，CARD引入了双Transformer结构，使其能够捕捉信号之间的时间相关性和多个变量在时间上的动态依赖。其次，我们引入了一种用于时间序列预测的鲁棒损失函数，以减轻潜在的过度拟合问题。这种新的损失函数基于预测不确定性加权预测在有限时间内的重要性。我们对多个长期和短期预测数据集进行的评估表明，CARD在精度和效率方面显著优于现有的方法。

    Recent studies have demonstrated the great power of deep learning methods, particularly Transformer and MLP, for time series forecasting. Despite its success in NLP and CV, many studies found that Transformer is less effective than MLP for time series forecasting. In this work, we design a special Transformer, i.e., channel-aligned robust dual Transformer (CARD for short), that addresses key shortcomings of Transformer in time series forecasting. First, CARD introduces a dual Transformer structure that allows it to capture both temporal correlations among signals and dynamical dependence among multiple variables over time. Second, we introduce a robust loss function for time series forecasting to alleviate the potential overfitting issue. This new loss function weights the importance of forecasting over a finite horizon based on prediction uncertainties. Our evaluation of multiple long-term and short-term forecasting datasets demonstrates that CARD significantly outperforms state-of-th
    
[^24]: UP5: 面向公平性推荐的无偏基础模型

    UP5: Unbiased Foundation Model for Fairness-aware Recommendation. (arXiv:2305.12090v1 [cs.IR])

    [http://arxiv.org/abs/2305.12090](http://arxiv.org/abs/2305.12090)

    本研究提出了一种新颖的基础模型UP5，它采用反事实公平促进技术来消除大型语言模型中的偏见，从而实现面向公平性的推荐。

    

    基于大型语言模型（LLM）等基础模型的最新进展，已将它们推到了推荐系统（RS）的前沿。此外，RS中的公平性很关键，因为许多用户将其用于决策和需求履行。然而，目前尚缺乏对推荐基础模型展示公平性水平和公平处理不同用户群组的适当方法的理解。本文侧重于用户方面的不公平问题，并通过彻底检查表明，LLMs中存在不公平性，导致不公平的推荐结果。为了消除LLM中的偏差以实现面向公平性的推荐，我们引入了一种基于反事实公平促进技术的新型无偏P5（UP5）基础模型。CFP包括两个子模块：个性化前缀提示和Prompt混合，从而增强了个体敏感属性的公平性。

    Recent advancements in foundation models such as large language models (LLM) have propelled them to the forefront of recommender systems (RS). Moreover, fairness in RS is critical since many users apply it for decision-making and demand fulfillment. However, at present, there is a lack of understanding regarding the level of fairness exhibited by recommendation foundation models and the appropriate methods for equitably treating different groups of users in foundation models. In this paper, we focus on user-side unfairness problem and show through a thorough examination that there is unfairness involved in LLMs that lead to unfair recommendation results. To eliminate bias from LLM for fairness-aware recommendation, we introduce a novel Unbiased P5 (UP5) foundation model based on Counterfactually-Fair-Prompting (CFP) techniques. CFP includes two sub-modules: a personalized prefix prompt that enhances fairness with respect to individual sensitive attributes, and a Prompt Mixture that int
    
[^25]: 基于深度强化学习的同行评审期刊系统中评审奖励的博弈论分析与实验评估

    Game-Theoretical Analysis of Reviewer Rewards in Peer-Review Journal Systems: Analysis and Experimental Evaluation using Deep Reinforcement Learning. (arXiv:2305.12088v1 [cs.AI])

    [http://arxiv.org/abs/2305.12088](http://arxiv.org/abs/2305.12088)

    本文针对同行评审期刊中代金券奖励制度可能会导致评审人员二元决策的问题，提出了一种替代奖励系统，有效地促进了更全面的审查，实验结果表明该系统更加平衡且稳定。

    

    本文利用数学准确性和博弈论策略洞察力，探讨了开放获取学术出版中的评审奖励领域。我们将现有的代金券奖励系统概念化为一个两个玩家的博弈，并识别了可能导致评审人员倾向于二元决策的潜在缺点。为解决这个问题，我们提出并数学上形式化了一种替代奖励系统，旨在减轻这种偏见，促进更全面的审查。我们运用严格的博弈论分析和深度强化学习模拟，对两种系统的属性和结果进行了详细的研究。我们的结果强调了两种系统之间的显着差异，我们提出的系统展现出了更平衡的决策分配和更加稳定的特点。这项研究不仅扩充了有关评审奖励系统的数学理解，而且提供了解决当前代金券奖励系统偏见问题的潜在解决方案。

    In this paper, we navigate the intricate domain of reviewer rewards in open-access academic publishing, leveraging the precision of mathematics and the strategic acumen of game theory. We conceptualize the prevailing voucher-based reviewer reward system as a two-player game, subsequently identifying potential shortcomings that may incline reviewers towards binary decisions. To address this issue, we propose and mathematically formalize an alternative reward system with the objective of mitigating this bias and promoting more comprehensive reviews. We engage in a detailed investigation of the properties and outcomes of both systems, employing rigorous game-theoretical analysis and deep reinforcement learning simulations. Our results underscore a noteworthy divergence between the two systems, with our proposed system demonstrating a more balanced decision distribution and enhanced stability. This research not only augments the mathematical understanding of reviewer reward systems, but it
    
[^26]: 半监督图形不平衡回归

    Semi-Supervised Graph Imbalanced Regression. (arXiv:2305.12087v1 [cs.LG])

    [http://arxiv.org/abs/2305.12087](http://arxiv.org/abs/2305.12087)

    本文提出了一种半监督框架来解决图形回归任务中稀有标签值示例不足的问题，通过自我训练，平衡训练数据并减少模型偏差。其中，采用一种新的回归置信度测量方法为代表稀有标签的更多图形伪标注，并在数据平衡后使用伪标签的方法在潜在空间中增加图形以生成更多的稀有标签示例。

    

    在回归任务中，由于某些连续标签值的观测很难收集，因此注释数据中很容易出现数据不平衡。在分子和聚合物属性预测中，注释的图形数据集通常很小，因为标注它们需要昂贵的设备和工作量。为了解决图形回归任务中稀有标签值示例不足的问题，我们提出了一种半监督框架，通过自我训练逐步平衡训练数据并减少模型偏差。训练数据的平衡通过以下方式实现：（1）用一种新的回归置信度测量方法为代表稀有标签的更多图形伪标注，并从不平衡的注释数据中逆分布采样它们的子集以识别未标注数据中的高质量样例；(2)在数据平衡后，使用伪标签的方法在潜在空间中增加图形，以生成更多的稀有标签示例，而无需手动标注。后者是利用潜在空间生成模型来人工生成更多的稀有标签示例。

    Data imbalance is easily found in annotated data when the observations of certain continuous label values are difficult to collect for regression tasks. When they come to molecule and polymer property predictions, the annotated graph datasets are often small because labeling them requires expensive equipment and effort. To address the lack of examples of rare label values in graph regression tasks, we propose a semi-supervised framework to progressively balance training data and reduce model bias via self-training. The training data balance is achieved by (1) pseudo-labeling more graphs for under-represented labels with a novel regression confidence measurement and (2) augmenting graph examples in latent space for remaining rare labels after data balancing with pseudo-labels. The former is to identify quality examples from unlabeled data whose labels are confidently predicted and sample a subset of them with a reverse distribution from the imbalanced annotated data. The latter collabor
    
[^27]: SneakyPrompt：评估文本生成图像模型安全过滤器的鲁棒性

    SneakyPrompt: Evaluating Robustness of Text-to-image Generative Models' Safety Filters. (arXiv:2305.12082v1 [cs.LG])

    [http://arxiv.org/abs/2305.12082](http://arxiv.org/abs/2305.12082)

    本文提出了第一个自动攻击框架SneakyPrompt，以评估最先进的文本生成图像模型中的安全过滤器的鲁棒性，该框架的关键洞见是搜索备选令牌来绕过安全过滤器。

    

    文本生成图像模型，如Stable Diffusion和DALL$\cdot$E 2等，由于它们在现实世界中的广泛应用而受到广泛关注。文本生成图像模型面临的一个挑战性问题是生成不安全内容，例如与暴力和成人相关的内容。因此，常见做法是部署所谓的安全过滤器，基于文本或图像特征阻止不安全内容。先前的工作研究了此类安全过滤器的可能绕过方式。然而，现有的工作在很大程度上是手动完成并专门针对Stable Diffusion官方的安全过滤器。此外，基于我们的评估，Stable Diffusion的安全过滤器的绕过比率仅为23.51％。在本文中，我们提出了第一个自动攻击框架SneakyPrompt，以评估最先进的文本生成图像模型中现实世界安全过滤器的鲁棒性。我们的关键洞见是搜索备选令牌来绕过安全过滤器。

    Text-to-image generative models such as Stable Diffusion and DALL$\cdot$E 2 have attracted much attention since their publication due to their wide application in the real world. One challenging problem of text-to-image generative models is the generation of Not-Safe-for-Work (NSFW) content, e.g., those related to violence and adult. Therefore, a common practice is to deploy a so-called safety filter, which blocks NSFW content based on either text or image features. Prior works have studied the possible bypass of such safety filters. However, existing works are largely manual and specific to Stable Diffusion's official safety filter. Moreover, the bypass ratio of Stable Diffusion's safety filter is as low as 23.51% based on our evaluation.  In this paper, we propose the first automated attack framework, called SneakyPrompt, to evaluate the robustness of real-world safety filters in state-of-the-art text-to-image generative models. Our key insight is to search for alternative tokens in 
    
[^28]: AnyPredict: 表格预测的基础模型

    AnyPredict: Foundation Model for Tabular Prediction. (arXiv:2305.12081v1 [cs.LG])

    [http://arxiv.org/abs/2305.12081](http://arxiv.org/abs/2305.12081)

    本文提出了一种名为 AnyPredict 的表格预测基础模型，使用数据引擎整合领域内和广泛的领域外数据集，以克服模式不匹配和预测目标异质性等方面的障碍。

    

    基础模型是在大规模数据上预先训练的模型，可以在许多下游任务中表现良好。它们在自然语言处理和计算机视觉方面取得了显著的成功。然而，这种模型在表格预测任务中的使用受到限制，主要问题包括 (1) 缺乏大规模和多样化的带有标准标签的表格数据集，以及 (2) 不同领域之间的模式不匹配和预测目标的异质性。本文提出了一种方法，用于构建基于 AnyPredict 的表格预测基础模型的大规模训练数据，包括领域内和广泛的领域外数据集。该方法使用数据引擎，利用大型语言模型 (LLM) 来整合表格样本，克服了不同模式表格之间的障碍，并使用“学习，注释和审计”流程将领域外数据与目标任务对齐。扩展的训练数据使预训练的 AnyPredict 能够支持每个表格领域。

    Foundation models are pre-trained on massive data to perform well across many downstream tasks. They have demonstrated significant success in natural language processing and computer vision. Nonetheless, the use of such models in tabular prediction tasks has been limited, with the main hurdles consisting of (1) the lack of large-scale and diverse tabular datasets with standardized labels and (2) the schema mismatch and predictive target heterogeneity across domains.  This paper proposes a method for building training data at scale for tabular prediction foundation models (AnyPredict) using both in-domain and a wide range of out-domain datasets. The method uses a data engine that leverages large language models (LLMs) to consolidate tabular samples to overcome the barrier across tables with varying schema and align out-domain data with the target task using a ``learn, annotate, and audit'' pipeline. The expanded training data enables the pre-trained AnyPredict to support every tabular d
    
[^29]: 深度学习中的GELU激活函数：全面的数学分析和性能评估

    GELU Activation Function in Deep Learning: A Comprehensive Mathematical Analysis and Performance. (arXiv:2305.12073v1 [cs.LG])

    [http://arxiv.org/abs/2305.12073](http://arxiv.org/abs/2305.12073)

    本文对GELU激活函数进行了全面的数学分析和广泛的实验比较，证明了它在深度学习模型中具有优越的性能和适用性。

    

    在深度学习模型中，选择最合适的激活函数是影响其学习能力、稳定性和计算效率的关键因素。近年来，高斯误差线性单元（GELU）激活函数已经成为一种主流方法，在各种应用中超越了传统的激活函数，如修正线性单元（ReLU）。本文对GELU激活函数进行了严格的数学分析，详细探讨了其可微性、有界性、平稳性和光滑性等性质。此外，我们对GELU函数进行了广泛的实验比较，利用在CIFAR-10、CIFAR-100和STL-10数据集上训练的残差卷积网络作为实证测试基础。我们的结果证明了GELU相对于其他激活函数的卓越性能，确立了它在广泛的深度学习模型中的适用性。

    Selecting the most suitable activation function is a critical factor in the effectiveness of deep learning models, as it influences their learning capacity, stability, and computational efficiency. In recent years, the Gaussian Error Linear Unit (GELU) activation function has emerged as a dominant method, surpassing traditional functions such as the Rectified Linear Unit (ReLU) in various applications. This study presents a rigorous mathematical investigation of the GELU activation function, exploring its differentiability, boundedness, stationarity, and smoothness properties in detail. Additionally, we conduct an extensive experimental comparison of the GELU function against a broad range of alternative activation functions, utilizing a residual convolutional network trained on the CIFAR-10, CIFAR-100, and STL-10 datasets as the empirical testbed. Our results demonstrate the superior performance of GELU compared to other activation functions, establishing its suitability for a wide ra
    
[^30]: 多任务模型的动态梯度平衡增强对抗攻击

    Dynamic Gradient Balancing for Enhanced Adversarial Attacks on Multi-Task Models. (arXiv:2305.12066v1 [cs.LG])

    [http://arxiv.org/abs/2305.12066](http://arxiv.org/abs/2305.12066)

    本文提出了动态梯度平衡攻击（DGBA）框架来攻击多任务模型，并通过实验回答了多任务模型的对抗攻击的安全性、多任务攻击和对抗训练是否增强多任务模型的鲁棒性等安全研究问题。

    

    多任务学习 (MTL) 创建了一个名为多任务模型的单个机器学习模型，可以同时执行多个任务。虽然单任务分类器的安全性得到了广泛的研究，但对于多任务模型，存在着几个关键的安全性研究问题，包括: 1）多任务模型对单任务对抗机器学习攻击的安全性如何？2）能否设计对抗性攻击来同时攻击多个任务？ 3）任务共享和对抗训练是否增加了多任务模型对对抗攻击的鲁棒性？本文通过仔细分析和严格的实验回答了这些问题。首先，我们开发了单任务白盒攻击的初级转化并分析了其固有缺陷。然后，我们提出了一种新的攻击框架，动态梯度平衡攻击（DGBA）。我们的框架把攻击多任务模型的问题作为一种基于平均相对损失变化的优化问题。

    Multi-task learning (MTL) creates a single machine learning model called multi-task model to simultaneously perform multiple tasks. Although the security of single task classifiers has been extensively studied, there are several critical security research questions for multi-task models including 1) How secure are multi-task models to single task adversarial machine learning attacks, 2) Can adversarial attacks be designed to attack multiple tasks simultaneously, and 3) Does task sharing and adversarial training increase multi-task model robustness to adversarial attacks? In this paper, we answer these questions through careful analysis and rigorous experimentation. First, we develop na\"ive adaptation of single-task white-box attacks and analyze their inherent drawbacks. We then propose a novel attack framework, Dynamic Gradient Balancing Attack (DGBA). Our framework poses the problem of attacking a multi-task model as an optimization problem based on averaged relative loss change, whi
    
[^31]: 无需触发器语音助手的高效多模神经网络

    Efficient Multimodal Neural Networks for Trigger-less Voice Assistants. (arXiv:2305.12063v1 [cs.LG])

    [http://arxiv.org/abs/2305.12063](http://arxiv.org/abs/2305.12063)

    本文提出了一种新方法，使用基于神经网络的音频-手势多模融合系统来实现无需触发器的语音助手，该方法可以更好地理解音频和手势数据之间的时间相关性，通用性强，并可以快速启动，提高资产开发流程的生产率。

    

    语音助手（VA）采用多模互动以增强人机交互的方式正在迅速增长。智能手表现在已经融合了无需显式触发器的VA调用方法，如Raise To Speak（RTS），用户将手表举起并向VA说话而无需显式触发器。当前最先进的RTS系统依靠启发式和设计的有限状态机来融合手势和音频数据以进行多模决策。然而，这些方法存在一些限制，包括适应性有限、可扩展性不足和人类产生的偏差。在这项工作中，我们提出了一种基于神经网络的音频-手势多模融合系统，其具有以下特点：（1）更好地理解音频和手势数据之间的时间相关性，从而进行精确的调用（2）在广泛的环境和场景下具有通用性（3）轻便且可在低功率设备上部署，如智能手表，并具有快速启动时间（4）提高资产开发流程的生产率。

    The adoption of multimodal interactions by Voice Assistants (VAs) is growing rapidly to enhance human-computer interactions. Smartwatches have now incorporated trigger-less methods of invoking VAs, such as Raise To Speak (RTS), where the user raises their watch and speaks to VAs without an explicit trigger. Current state-of-the-art RTS systems rely on heuristics and engineered Finite State Machines to fuse gesture and audio data for multimodal decision-making. However, these methods have limitations, including limited adaptability, scalability, and induced human biases. In this work, we propose a neural network based audio-gesture multimodal fusion system that (1) Better understands temporal correlation between audio and gesture data, leading to precise invocations (2) Generalizes to a wide range of environments and scenarios (3) Is lightweight and deployable on low-power devices, such as smartwatches, with quick launch times (4) Improves productivity in asset development processes.
    
[^32]: 机器学习算法设计生物相容性镁合金的力学性能

    Mechanical Property Design of Bio-compatible Mg alloys using Machine-Learning Algorithms. (arXiv:2305.12060v1 [cond-mat.mtrl-sci])

    [http://arxiv.org/abs/2305.12060](http://arxiv.org/abs/2305.12060)

    本研究开发了一个机器学习模型来预测生物相容性镁合金的屈服强度，通过遗传算法的优化，平衡了其力学强度和生物相容性。

    

    镁合金因其生物相容性、可控腐蚀速率以及与自然骨骼在刚度和密度方面的相似性而成为暂时性生物植入物的有吸引力选择。然而，低的机械强度阻碍了它们作为心血管支架和骨替代品的使用。虽然可以进行合金工程，以实现所需的机械强度，但使用常规实验方法优化生物相容性镁合金的力学性能非常耗时和昂贵。因此，可以利用人工智能来简化合金设计流程并缩短所需时间。在本研究中，开发了一个机器学习模型，可预测生物相容性镁合金的屈服强度，$R^2$的准确度为91％。接下来，将预测模型用作遗传算法的适应度函数，以优化合金中元素的质量百分比，并在力学强度和生物相容性之间获得平衡。

    Magnesium alloys are attractive options for temporary bio-implants because of their biocompatibility, controlled corrosion rate, and similarity to natural bone in terms of stiffness and density. Nevertheless, their low mechanical strength hinders their use as cardiovascular stents and bone substitutes. While it is possible to engineer alloys with the desired mechanical strength, optimizing the mechanical properties of biocompatible magnesium alloys using conventional experimental methods is time-consuming and expensive. Therefore, Artificial Intelligence (AI) can be leveraged to streamline the alloy design process and reduce the required time. In this study, a machine learning model was developed to predict the yield strength (YS) of biocompatible magnesium alloys with an $R^2$ accuracy of 91\%. The predictive model was then validated using the CALPHAD technique and thermodynamics calculations. Next, the predictive model was employed as the fitness function of a genetic algorithm to op
    
[^33]: DADIN: 面向跨域推荐系统的领域对抗深度兴趣网络

    DADIN: Domain Adversarial Deep Interest Network for Cross Domain Recommender Systems. (arXiv:2305.12058v1 [cs.IR])

    [http://arxiv.org/abs/2305.12058](http://arxiv.org/abs/2305.12058)

    论文提出了一种创新性的深度跨领域点击率预测模型——领域对抗深度兴趣网络（DADIN），该模型通过引入领域不可知层和特别设计的损失，创新地实现了两个领域的联合分布对齐，并采用对抗训练的方式与点击率预测损失一起进行优化，相比竞争基线算法提升明显。

    

    点击率预测是推荐系统的主要任务之一，用户针对不同项目进行点击以获取推荐结果。针对数据稀疏性、用户-项目交互的长尾分布和项目或用户的冷启动等问题，提出了跨领域点击率预测模型。为了使源域到目标域的知识转移更加顺畅，提出了创新性的深度跨领域点击率预测模型——领域对抗深度兴趣网络 (DADIN)，将跨域推荐任务转化为领域适应问题。通过引入领域不可知层和特别设计的损失，创新地实现了两个领域的联合分布对齐，并采用对抗训练的方式与点击率预测损失一起进行优化。实验结果表明，在华为数据集上，DADIN 的曲线下面积 (AUC) 比最具竞争力的基线高出0.08％，高出0.7％。

    Click-Through Rate (CTR) prediction is one of the main tasks of the recommendation system, which is conducted by a user for different items to give the recommendation results. Cross-domain CTR prediction models have been proposed to overcome problems of data sparsity, long tail distribution of user-item interactions, and cold start of items or users. In order to make knowledge transfer from source domain to target domain more smoothly, an innovative deep learning cross-domain CTR prediction model, Domain Adversarial Deep Interest Network (DADIN) is proposed to convert the cross-domain recommendation task into a domain adaptation problem. The joint distribution alignment of two domains is innovatively realized by introducing domain agnostic layers and specially designed loss, and optimized together with CTR prediction loss in a way of adversarial training. It is found that the Area Under Curve (AUC) of DADIN is 0.08% higher than the most competitive baseline on Huawei dataset and is 0.7
    
[^34]: （带噪声的）随机梯度下降的时间均匀Wasserstein稳定性界限

    Uniform-in-Time Wasserstein Stability Bounds for (Noisy) Stochastic Gradient Descent. (arXiv:2305.12056v1 [stat.ML])

    [http://arxiv.org/abs/2305.12056](http://arxiv.org/abs/2305.12056)

    本文通过建立学习理论和应用概率之间的联系，提出了一种证明随机优化算法Wasserstein稳定性界限的统一指南，并在随机梯度下降上验证了该方法的有效性，包括强凸损失和带添加噪声的非凸损失。

    

    算法稳定性是一个重要的概念，对于推导实践算法的泛化界限已被证明是有用的。过去十年已经见证了不同损失函数所应用的不同算法的稳定性界限的增加。虽然这些界限照亮了优化算法的各种属性，但每个案例的分析通常需要不同的证明技术和显著不同的数学工具。在本研究中，我们在学习理论和应用概率之间建立了新的联系，并介绍了一种证明随机优化算法的Wasserstein稳定性界限的统一指南。我们在随机梯度下降（SGD）上阐述了我们的方法，并获得了强凸损失和带添加噪声的非凸损失的时间均匀稳定性界限（即，界限不随迭代次数增加而增加），在这些情况下，我们恢复了与先前文献相似的结果或将它们扩展到更广泛。

    Algorithmic stability is an important notion that has proven powerful for deriving generalization bounds for practical algorithms. The last decade has witnessed an increasing number of stability bounds for different algorithms applied on different classes of loss functions. While these bounds have illuminated various properties of optimization algorithms, the analysis of each case typically required a different proof technique with significantly different mathematical tools. In this study, we make a novel connection between learning theory and applied probability and introduce a unified guideline for proving Wasserstein stability bounds for stochastic optimization algorithms. We illustrate our approach on stochastic gradient descent (SGD) and we obtain time-uniform stability bounds (i.e., the bound does not increase with the number of iterations) for strongly convex losses and non-convex losses with additive noise, where we recover similar results to the prior art or extend them to mor
    
[^35]: SIDAR：用于对齐和修复的合成图像数据集

    SIDAR: Synthetic Image Dataset for Alignment & Restoration. (arXiv:2305.12036v1 [cs.CV])

    [http://arxiv.org/abs/2305.12036](http://arxiv.org/abs/2305.12036)

    SIDAR是一种通过使用3D渲染生成多视角、多光照、多阴影、多遮挡等真实场景的方式来解决图像对齐和图像修复问题的数据增强方法。

    

    图像对齐和图像修复是经典的计算机视觉任务。然而，仍缺乏足够的数据集来训练和评估端到端深度学习模型。获取用于图像对齐的基准数据需要精细的运动结构方法或光流系统，通常只提供大量的图像对应关系，而在基础图像序列中仅引入少量的场景变化。而替代方法则利用现有图像数据上的随机透视扭曲。然而，这只提供了平凡的扭曲，缺乏真实场景的复杂性和变化性。相反，我们提出的数据增强方法通过使用3D渲染来克服数据缺乏的问题：将图像添加为平面的纹理，然后添加不同的光照条件、阴影和遮挡到场景中。场景从多个视角渲染，提供了多样化的图像集，其中的对齐和/或退化得到了已知的基准数据，因此为图像对齐和修复任务提供了适当的训练和评估数据。

    Image alignment and image restoration are classical computer vision tasks. However, there is still a lack of datasets that provide enough data to train and evaluate end-to-end deep learning models. Obtaining ground-truth data for image alignment requires sophisticated structure-from-motion methods or optical flow systems that often do not provide enough data variance, i.e., typically providing a high number of image correspondences, while only introducing few changes of scenery within the underlying image sequences. Alternative approaches utilize random perspective distortions on existing image data. However, this only provides trivial distortions, lacking the complexity and variance of real-world scenarios. Instead, our proposed data augmentation helps to overcome the issue of data scarcity by using 3D rendering: images are added as textures onto a plane, then varying lighting conditions, shadows, and occlusions are added to the scene. The scene is rendered from multiple viewpoints, g
    
[^36]: Waymo开放模拟代理挑战赛

    The Waymo Open Sim Agents Challenge. (arXiv:2305.12032v1 [cs.CV])

    [http://arxiv.org/abs/2305.12032](http://arxiv.org/abs/2305.12032)

    Waymo开放模拟代理挑战赛提出使用真实、互动的智能体仿真以促进自动驾驶行为模型的评估和训练，是该领域的首个公开挑战赛，旨在推动逼真模拟器的设计。

    

    本文定义了Waymo开放模拟代理挑战赛(WOSAC)。通过与真实、互动的智能体进行仿真是自动驾驶软件开发的关键任务。WOSAC是第一个公开的挑战赛，旨在解决该任务并提出相应的评估指标。该挑战的目标是激发设计逼真模拟器的兴趣，以用于评估和训练自动驾驶的行为模型。我们概述了评估方法，并展示了几种基准仿真代理方法的初步结果。

    In this work, we define the Waymo Open Sim Agents Challenge (WOSAC). Simulation with realistic, interactive agents represents a key task for autonomous vehicle software development. WOSAC is the first public challenge to tackle this task and propose corresponding metrics. The goal of the challenge is to stimulate the design of realistic simulators that can be used to evaluate and train a behavior model for autonomous driving. We outline our evaluation methodology and present preliminary results for a number of different baseline simulation agent methods.
    
[^37]: 学习连续的图序列 -- 动力系统方法

    Learning Continually on a Sequence of Graphs -- The Dynamical System Way. (arXiv:2305.12030v1 [cs.LG])

    [http://arxiv.org/abs/2305.12030](http://arxiv.org/abs/2305.12030)

    该论文研究的是如何进行图数据的持续学习，并面临着由于数据基础特性所导致的理论与方法上的挑战。

    

    持续学习(CL)是一个领域，关注于学习一系列相互关联的任务，这些任务通常是以回归或分类的方式定义的。近年来，当这些任务是使用欧几里得数据定义的时，如图像，CL已经被广泛研究。然而，当与CL任务相对应的数据是非欧几里德的时，如图形、点云或流形，欧几里得度量意义下的相似性概念并不适用。因此，为非欧几里德数据开发CL面临着几个理论和方法上的挑战。特别是，对于图中的CL需要显式地模拟节点和边的非平稳行为。

    Continual learning~(CL) is a field concerned with learning a series of inter-related task with the tasks typically defined in the sense of either regression or classification. In recent years, CL has been studied extensively when these tasks are defined using Euclidean data-- data, such as images, that can be described by a set of vectors in an n-dimensional real space. However, the literature is quite sparse, when the data corresponding to a CL task is nonEuclidean-- data , such as graphs, point clouds or manifold, where the notion of similarity in the sense of Euclidean metric does not hold. For instance, a graph is described by a tuple of vertices and edges and similarities between two graphs is not well defined through a Euclidean metric. Due to this fundamental nature of the data, developing CL for nonEuclidean data presents several theoretical and methodological challenges. In particular, CL for graphs requires explicit modelling of nonstationary behavior of vertices and edges an
    
[^38]: MultiTurnCleanup：用于多轮口语会话转录清理的基准测试

    MultiTurnCleanup: A Benchmark for Multi-Turn Spoken Conversational Transcript Cleanup. (arXiv:2305.12029v1 [cs.CL])

    [http://arxiv.org/abs/2305.12029](http://arxiv.org/abs/2305.12029)

    本研究提出了MultiTurnCleanup任务，收集了新的数据集MultiTurnCleanup1，针对口语会话转录中的不连续现象进行探讨并提供了两个可用于未来研究的基准测试模型。

    

    目前的语调不连续检测模型侧重于单个说话者的每个话语。然而，口语会话转录中的许多不连续现象都发生在多轮对话中，这影响了人类的可读性和下游 NLP 任务的性能。本研究通过提出创新的“MultiTurnCleanup”任务，针对口语会话转录中的不连续现象进行探讨，并收集了新的数据集MultiTurnCleanup1。我们设计了一种数据标注模式以收集高质量的数据集，提供了广泛的数据分析。此外，我们利用两种建模方法进行实验评估，作为未来研究的基准测试。

    Current disfluency detection models focus on individual utterances each from a single speaker. However, numerous discontinuity phenomena in spoken conversational transcripts occur across multiple turns, hampering human readability and the performance of downstream NLP tasks. This study addresses these phenomena by proposing an innovative Multi-Turn Cleanup task for spoken conversational transcripts and collecting a new dataset, MultiTurnCleanup1. We design a data labeling schema to collect the high-quality dataset and provide extensive data analysis. Furthermore, we leverage two modeling approaches for experimental evaluation as benchmarks for future research.
    
[^39]: 基于能效随态电容器的物理储层计算系统用于时间数据处理

    Energy-efficient memcapacitive physical reservoir computing system for temporal data processing. (arXiv:2305.12025v1 [cs.LG])

    [http://arxiv.org/abs/2305.12025](http://arxiv.org/abs/2305.12025)

    本文研究了一种基于能效随态电容器的物理储层计算系统，解决了时间数据的分类任务和分析时间序列数据的问题。在实验中，系统能够实现较高的准确率和较小的均方误差。

    

    储层计算是一种高效的机器学习框架，通过从输入信号中提取特征并将其映射到高维空间来处理时间数据。物理储层可使用磁旋电子、原子开关网络、硅光学模块、铁电晶体管和易失性存储器来实现。然而，这些设备由于其电阻性质本质上存在能量耗散问题，导致功耗增加。因此，采用电容存储器设备可提供更为能效的解决方案。在这里，我们利用模拟和实验中近似某些短期突触可塑性功能的易失生物膜基质量作为储层，解决分类任务和分析时间序列数据。我们的系统在口音数字分类中实现了98％的准确率，在二阶非线性回归任务中获得了0.0012的归一化均方误差。

    Reservoir computing is a highly efficient machine learning framework for processing temporal data by extracting features from the input signal and mapping them into higher dimensional spaces. Physical reservoir layers have been realized using spintronic oscillators, atomic switch networks, silicon photonic modules, ferroelectric transistors, and volatile memristors. However, these devices are intrinsically energy-dissipative due to their resistive nature, which leads to increased power consumption. Therefore, capacitive memory devices can provide a more energy-efficient approach. Here, we leverage volatile biomembrane-based memcapacitors that closely mimic certain short-term synaptic plasticity functions as reservoirs to solve classification tasks and analyze time-series data in simulation and experimentally. Our system achieves a 98% accuracy rate for spoken digit classification and a normalized mean square error of 0.0012 in a second-order non-linear regression task. Further, to demo
    
[^40]: Chemellia: 用于原子层科学机器学习的生态系统

    Chemellia: An Ecosystem for Atomistic Scientific Machine Learning. (arXiv:2305.12010v1 [cs.CE])

    [http://arxiv.org/abs/2305.12010](http://arxiv.org/abs/2305.12010)

    Chemellia是一个开源原子层机器学习框架，库设计关注点分离、互操作性和透明度。其重要贡献是实现一种用于材料属性预测的晶体图卷积神经网络。

    

    Chemellia是一个基于Julia编程语言的原子层机器学习的开源框架。该框架利用了Julia的高速度以及通过多重派生范式共享和重复使用代码和接口的能力。Chemellia旨在利用现有的接口，并在可能的情况下避免“重复造轮子”。Chemellia生态系统的一个关键方面是ChemistryFeaturization界面，用于定义和编码特征——旨在最大程度地提高特征化方案及其元素之间的互操作性，保持编码特征的来源，并确保易于解码和重新配置，以启用特征工程实验。这体现了Chemellia生态系统的总体设计原则：关注点分离、互操作性和透明度。我们通过讨论用于材料属性预测的晶体图卷积神经网络的实现来说明这些原则。

    Chemellia is an open-source framework for atomistic machine learning in the Julia programming language. The framework takes advantage of Julia's high speed as well as the ability to share and reuse code and interfaces through the paradigm of multiple dispatch. Chemellia is designed to make use of existing interfaces and avoid ``reinventing the wheel'' wherever possible. A key aspect of the Chemellia ecosystem is the ChemistryFeaturization interface for defining and encoding features -- it is designed to maximize interoperability between featurization schemes and elements thereof, to maintain provenance of encoded features, and to ensure easy decodability and reconfigurability to enable feature engineering experiments. This embodies the overall design principles of the Chemellia ecosystem: separation of concerns, interoperability, and transparency. We illustrate these principles by discussing the implementation of crystal graph convolutional neural networks for material property predict
    
[^41]: 一种基于Transformer的图形方法为OpenMP并行化提供建议

    Advising OpenMP Parallelization via a Graph-Based Approach with Transformers. (arXiv:2305.11999v1 [cs.DC])

    [http://arxiv.org/abs/2305.11999](http://arxiv.org/abs/2305.11999)

    本文提出了一种名为OMPify的新方法，该方法基于Transformer模型，通过对串行代码的分析，自动检测和预测并行代码中的OpenMP编译指示符和共享内存属性。

    

    利用多核架构的全部潜力，需要共享内存并行化方案。当前最常见的解决方案是OpenMP并行编程接口。虽然手动编写并行代码是复杂和费力的，但是许多确定性源到源（S2S）编译器已经涌现，旨在自动化将串行代码转换为并行代码的过程。然而，最近的研究表明，在许多情况下这些编译器是不实际的。在本文中，我们将AI和自然语言处理（NLP）领域的最新进展与大量的开源代码相结合，以解决自动并行化的问题。具体而言，我们提出了一种新方法，称为OMPify，通过串行代码来检测和预测并行代码中的OpenMP编译指示符和共享内存属性，OMPify基于基于Transformer的模型，利用源代码的基于图形的表示。

    There is an ever-present need for shared memory parallelization schemes to exploit the full potential of multi-core architectures. The most common parallelization API addressing this need today is OpenMP. Nevertheless, writing parallel code manually is complex and effort-intensive. Thus, many deterministic source-to-source (S2S) compilers have emerged, intending to automate the process of translating serial to parallel code. However, recent studies have shown that these compilers are impractical in many scenarios. In this work, we combine the latest advancements in the field of AI and natural language processing (NLP) with the vast amount of open-source code to address the problem of automatic parallelization. Specifically, we propose a novel approach, called OMPify, to detect and predict the OpenMP pragmas and shared-memory attributes in parallel code, given its serial version. OMPify is based on a Transformer-based model that leverages a graph-based representation of source code that
    
[^42]: 具有概率保证的神经网络鲁棒的反事实解释

    Robust Counterfactual Explanations for Neural Networks With Probabilistic Guarantees. (arXiv:2305.11997v1 [stat.ML])

    [http://arxiv.org/abs/2305.11997](http://arxiv.org/abs/2305.11997)

    本文提出了一种可靠的神经网络反事实解释方法，该方法可以针对自然发生的模型变化提供高概率的鲁棒性。

    

    针对神经网络发现偏移，通过使用稳定性度量来量化反事实解释对可能的模型变化的鲁棒性。通过在反事实解释优化中引入正则化项来将生成的反事实解释靠近数据流形，从而实现了对自然发生的模型变化的高概率鲁棒性。新的算法在合成和现实世界数据集上进行实验，证明了其有效性。

    There is an emerging interest in generating robust counterfactual explanations that would remain valid if the model is updated or changed even slightly. Towards finding robust counterfactuals, existing literature often assumes that the original model $m$ and the new model $M$ are bounded in the parameter space, i.e., $\|\text{Params}(M){-}\text{Params}(m)\|{<}\Delta$. However, models can often change significantly in the parameter space with little to no change in their predictions or accuracy on the given dataset. In this work, we introduce a mathematical abstraction termed \emph{naturally-occurring} model change, which allows for arbitrary changes in the parameter space such that the change in predictions on points that lie on the data manifold is limited. Next, we propose a measure -- that we call \emph{Stability} -- to quantify the robustness of counterfactuals to potential model changes for differentiable models, e.g., neural networks. Our main contribution is to show that counter
    
[^43]: 基于深度学习的软件图像信号处理方法综述

    Survey on software ISP methods based on Deep Learning. (arXiv:2305.11994v1 [cs.LG])

    [http://arxiv.org/abs/2305.11994](http://arxiv.org/abs/2305.11994)

    本文综述了基于深度学习的软件图像信号处理方法，包括去马赛克、降噪和增强等多个过程，研究并分析了最新的几项研究，并对方法进行了比较和改进点的探讨。

    

    相机的整个图像信号处理器（ISP）依靠多个过程将来自彩色滤波阵列（CFA）传感器的数据转换，例如去马赛克、降噪和增强。这些过程可以通过某些硬件或软件来执行。近年来，深度学习已经成为了其中一些过程的解决方案，甚至可以使用单个神经网络替代整个ISP。在本文中，我们调查了该领域内的几项最新研究，并对这些方法进行了深入的分析和比较，包括结果及未来研究的可能改进点。

    The entire Image Signal Processor (ISP) of a camera relies on several processes to transform the data from the Color Filter Array (CFA) sensor, such as demosaicing, denoising, and enhancement. These processes can be executed either by some hardware or via software. In recent years, Deep Learning has emerged as one solution for some of them or even to replace the entire ISP using a single neural network for the task. In this work, we investigated several recent pieces of research in this area and provide deeper analysis and comparison among them, including results and possible points of improvement for future researchers.
    
[^44]: 高产农田检测：一个新的数据集和深度学习基准结果

    Productive Crop Field Detection: A New Dataset and Deep Learning Benchmark Results. (arXiv:2305.11990v1 [cs.CV])

    [http://arxiv.org/abs/2305.11990](http://arxiv.org/abs/2305.11990)

    本研究提出了一个高质量的数据集，使用半监督和最先进的深度学习方法自动检测高产农田，获得了很高的准确性，有望为农民提供帮助.

    

    在精准农业中，检测高产农田是一项必要的实践，使得农民可以单独评估操作绩效并比较不同的种子品种、农药和肥料。然而，手动识别高产农田往往是一项耗时且容易出错的任务。以往的研究尝试使用先进的机器学习算法检测农田，但往往缺乏高质量的标记数据。在这种情况下，我们提出了一个高质量的数据集，它是通过机器操作结合随着时间推移而跟踪的Sentinel-2图像生成的。据我们所知，这是第一个通过使用这种技术克服标记样本不足的数据集。接着，我们应用半监督无标签数据分类和最先进的有监督和自监督深度学习方法来自动检测高产农田。最终结果表明在正无标记学习中具有很高的准确性，这非常适合

    In precision agriculture, detecting productive crop fields is an essential practice that allows the farmer to evaluate operating performance separately and compare different seed varieties, pesticides, and fertilizers. However, manually identifying productive fields is often a time-consuming and error-prone task. Previous studies explore different methods to detect crop fields using advanced machine learning algorithms, but they often lack good quality labeled data. In this context, we propose a high-quality dataset generated by machine operation combined with Sentinel-2 images tracked over time. As far as we know, it is the first one to overcome the lack of labeled samples by using this technique. In sequence, we apply a semi-supervised classification of unlabeled data and state-of-the-art supervised and self-supervised deep learning methods to detect productive crop fields automatically. Finally, the results demonstrate high accuracy in Positive Unlabeled learning, which perfectly fi
    
[^45]: OL-Transformer：用于光学多层薄膜结构的快速通用代理模拟器

    OL-Transformer: A Fast and Universal Surrogate Simulator for Optical Multilayer Thin Film Structures. (arXiv:2305.11984v1 [cs.LG])

    [http://arxiv.org/abs/2305.11984](http://arxiv.org/abs/2305.11984)

    该论文提出了OL-Transformer用于光学多层薄膜结构，可以预测多达$10^{25}$种不同多层结构的精确反射和透射光谱，同时具有快速的计算速度。

    

    基于深度学习的方法最近被证明是用于光学多层薄膜结构的快速准确的代理模拟器。然而，现有的方法仅适用于具有不同材料排列方式的有限类型的结构，限制了它们向多样化和通用化结构的应用。在这里，我们提出了Opto-Layer（OL）Transformer作为巨量结构的通用替代模拟器。结合结构序列化技术，我们的模型可以预测多达$10^{25}$种不同多层结构的精确反射和透射光谱，同时相较于物理求解器仍然实现了6倍时间加速。进一步的研究表明，普遍的学习能力来自于我们的模型首先学习物理嵌入，然后使用自我注意机制来捕捉每层之间的光物质相互作用的隐藏关系。

    Deep learning-based methods have recently been established as fast and accurate surrogate simulators for optical multilayer thin film structures. However, existing methods only work for limited types of structures with different material arrangements, preventing their applications towards diverse and universal structures. Here, we propose the Opto-Layer (OL) Transformer to act as a universal surrogate simulator for enormous types of structures. Combined with the technique of structure serialization, our model can predict accurate reflection and transmission spectra for up to $10^{25}$ different multilayer structures, while still achieving a six-fold time speedup compared to physical solvers. Further investigation reveals that the general learning ability comes from the fact that our model first learns the physical embeddings and then uses the self-attention mechanism to capture the hidden relationship of light-matter interaction between each layer.
    
[^46]: 带有时间预测编码的时序记忆

    Sequential Memory with Temporal Predictive Coding. (arXiv:2305.11982v1 [q-bio.NC])

    [http://arxiv.org/abs/2305.11982](http://arxiv.org/abs/2305.11982)

    该论文提出了一种基于PC的新型时序记忆模型，称为时间预测编码（tPC），可以通过生物可行的神经实现准确地记忆和检索连续输入。其中tPC可以被看作是一种经典异向性霍普菲尔德网络（AHN），具有更稳定的性能，并且可以编码上下文相关信息，区分在序列中出现的重复元素。

    

    对于生物体存储事件序列的时间顺序至关重要，然而大脑中支配时序记忆的计算机制仍不清楚。本文受到神经科学理论和预测编码（PC）在静态存储任务中的成功启示，提出了一种基于PC的新型时序记忆模型，称为时间预测编码（tPC）。我们展示了我们的tPC模型可以通过生物可行的神经实现准确地记忆和检索连续输入。重要的是，我们的分析研究表明，tPC可以被看作是一种具有隐式统计白化过程的经典异向性霍普菲尔德网络（AHN），这会在结构化输入的时序记忆任务中导致更稳定的性能。此外，我们发现具有多层结构的tPC可以编码上下文相关信息，因此可以区分在序列中出现的重复元素。

    Memorizing the temporal order of event sequences is critical for the survival of biological agents. However, the computational mechanism underlying sequential memory in the brain remains unclear. Inspired by neuroscience theories and recent successes in applying predictive coding (PC) to static memory tasks, in this work we propose a novel PC-based model for sequential memory, called temporal predictive coding (tPC). We show that our tPC models can memorize and retrieve sequential inputs accurately with a biologically plausible neural implementation. Importantly, our analytical study reveals that tPC can be viewed as a classical Asymmetric Hopfield Network (AHN) with an implicit statistical whitening process, which leads to more stable performance in sequential memory tasks of structured inputs. Moreover, we find that tPC with a multi-layer structure can encode context-dependent information, thus distinguishing between repeating elements appearing in a sequence, a computation attribute
    
[^47]: AutoCoreset：一个自动实用的Coreset构建框架。

    AutoCoreset: An Automatic Practical Coreset Construction Framework. (arXiv:2305.11980v1 [cs.LG])

    [http://arxiv.org/abs/2305.11980](http://arxiv.org/abs/2305.11980)

    AutoCoreset是一个自动构建高质量Coreset的通用且实用的框架，用户只需提供数据和成本函数即可，无需其他计算，可用于任何数据和成本函数。

    

    Coreset是一个与输入集合紧密相似于某种特定查询的损失函数的加权子集。由于Coreset对许多应用程序具有优势，并已被广泛用于机器学习，因此Coreset成为了普遍的研究领域。然而，当前Coreset的构建往往是问题依赖的，即对于每个问题，通常都会建议使用一种新的Coreset构建算法，这一过程可能需要时间，或者对于该领域的新研究人员来说可能很难。即使是通用框架，用户也需要完成其他（问题相关的）计算或证明。此外，许多问题并没有（可证明的）小的Coreset，限制了其应用性。为此，我们提出了一个自动实用的框架来构建Coreset，它只需要用户输入数据和期望的成本函数，无需用户完成任何其他任务相关的计算。为此，我们将将大数据集转换为小Coreset简化为一个简单的优化问题，并由我们提出的框架自动解决。我们的框架是通用的，适用于任何数据和成本函数，并且也是实用的，因为它能快速构建Coreset，同时保证高质量的逼近。

    A coreset is a tiny weighted subset of an input set, that closely resembles the loss function, with respect to a certain set of queries. Coresets became prevalent in machine learning as they have shown to be advantageous for many applications. While coreset research is an active research area, unfortunately, coresets are constructed in a problem-dependent manner, where for each problem, a new coreset construction algorithm is usually suggested, a process that may take time or may be hard for new researchers in the field. Even the generic frameworks require additional (problem-dependent) computations or proofs to be done by the user. Besides, many problems do not have (provable) small coresets, limiting their applicability. To this end, we suggest an automatic practical framework for constructing coresets, which requires (only) the input data and the desired cost function from the user, without the need for any other task-related computation to be done by the user. To do so, we reduce t
    
[^48]: 异构传感器信号的无监督变点检测

    Unsupervised Change Point Detection for heterogeneous sensor signals. (arXiv:2305.11976v1 [cs.LG])

    [http://arxiv.org/abs/2305.11976](http://arxiv.org/abs/2305.11976)

    本文研究了无监督变点检测技术，该技术灵活适用于各种数据源，无需大量训练数据和重新校准模型。

    

    变点检测是时间序列数据分析中至关重要的一个方面，因为变点的存在表明生成数据的过程发生了突然而显著的变化。虽然随着时间的推移，许多变点检测算法已被开发出来，但在特定问题中选择合适的算法仍然具有挑战性。算法的选择严重依赖于问题的性质和底层数据源。在本文中，我们将专门考察无监督技术，因为它们适用于各种数据源，而无需大量标注的训练数据和模型的重新校准。介绍并评估了几个标准来比较算法。

    Change point detection is a crucial aspect of analyzing time series data, as the presence of a change point indicates an abrupt and significant change in the process generating the data. While many algorithms for the problem of change point detection have been developed over time, it can be challenging to select the appropriate algorithm for a specific problem. The choice of the algorithm heavily depends on the nature of the problem and the underlying data source. In this paper, we will exclusively examine unsupervised techniques due to their flexibility in the application to various data sources without the requirement for abundant annotated training data and the re-calibration of the model. The examined methods will be introduced and evaluated based on several criteria to compare the algorithms.
    
[^49]: 不是所有的语义都是平等的：具有自定义温度的对比自监督学习

    Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization. (arXiv:2305.11965v1 [cs.LG])

    [http://arxiv.org/abs/2305.11965](http://arxiv.org/abs/2305.11965)

    本文提出了一种具有个性化温度的对比损失用于自监督学习，根据数据分布自动调整温度以使得训练更加有效。

    

    本文旨在通过原则性和系统性的方式，优化具有个性化温度的对比损失，用于自监督学习。普遍做法是将全局温度参数τ用于所有数据，忽略了“不是所有的语义都是平等的”这个事实，特别是在数据展示长尾分布时，不同的锚点数据可能具有不同数量的类似语义的样本。我们提出了一种基于分布鲁棒性优化（DRO）的新型鲁棒对比损失，为我们提供了有关τ的影响的直觉和自动温度个性化的机制。然后，我们提出了一种有效的随机算法来优化鲁棒性对比损失，具有可证明的收敛保证，而不需要使用大型小批量大小。理论和实验结果表明，我们的算法自动学习每个样本的合适τ。具体来说，具有频繁语义的样本使用较大温度以保持难度。

    In this paper, we aim to optimize a contrastive loss with individualized temperatures in a principled and systematic manner for self-supervised learning. The common practice of using a global temperature parameter $\tau$ ignores the fact that ``not all semantics are created equal", meaning that different anchor data may have different numbers of samples with similar semantics, especially when data exhibits long-tails. First, we propose a new robust contrastive loss inspired by distributionally robust optimization (DRO), providing us an intuition about the effect of $\tau$ and a mechanism for automatic temperature individualization. Then, we propose an efficient stochastic algorithm for optimizing the robust contrastive loss with a provable convergence guarantee without using large mini-batch sizes. Theoretical and experimental results show that our algorithm automatically learns a suitable $\tau$ for each sample. Specifically, samples with frequent semantics use large temperatures to k
    
[^50]: 通过信息瓶颈方法探索监督对比学习中神经网络崩溃的理解

    Towards understanding neural collapse in supervised contrastive learning with the information bottleneck method. (arXiv:2305.11957v1 [cs.LG])

    [http://arxiv.org/abs/2305.11957](http://arxiv.org/abs/2305.11957)

    本文将神经网络崩溃建模为信息瓶颈问题，证明神经网络崩溃导致良好的泛化，特别是当它接近分类问题的最优信息瓶颈解时。

    

    神经网络崩溃是指在超出性能平台训练时，深度神经网络最后一层激活的几何学表现。目前存在的问题包括神经网络崩溃是否会导致更好的泛化，如果是，超出性能平台的训练如何帮助神经网络崩溃。本文将神经网络崩溃建模为信息瓶颈问题，以探究是否存在这样一种紧凑的表示，并发现其与泛化性的关联。我们证明神经网络崩溃导致良好的泛化，特别是当它接近分类问题的最优信息瓶颈解时。最近的研究表明，使用相同的对比损失目标独立训练的两个深度神经网络是线性可识别的，这意味着得到的表示等效于矩阵变换。我们利用线性可识别性来近似信息瓶颈问题的解析解。这个近似表明，当类平均值相等时，最优解非常接近端到端模型，并提供了进一步的理论分析。

    Neural collapse describes the geometry of activation in the final layer of a deep neural network when it is trained beyond performance plateaus. Open questions include whether neural collapse leads to better generalization and, if so, why and how training beyond the plateau helps. We model neural collapse as an information bottleneck (IB) problem in order to investigate whether such a compact representation exists and discover its connection to generalization. We demonstrate that neural collapse leads to good generalization specifically when it approaches an optimal IB solution of the classification problem. Recent research has shown that two deep neural networks independently trained with the same contrastive loss objective are linearly identifiable, meaning that the resulting representations are equivalent up to a matrix transformation. We leverage linear identifiability to approximate an analytical solution of the IB problem. This approximation demonstrates that when class means exh
    
[^51]: OPTWIN: 使用最优子窗口进行漂移识别

    OPTWIN: Drift identification with optimal sub-windows. (arXiv:2305.11942v1 [cs.LG])

    [http://arxiv.org/abs/2305.11942](http://arxiv.org/abs/2305.11942)

    本文提出了OPTWIN概念漂移检测器，使用滑动子窗口方法检测概念漂移，获得了更高的准确性和更低的假阳性率。

    

    在线学习（OL）是一个在学术界和工业界日益受到关注的研究领域。OL的主要挑战之一是概念漂移的内在存在，概念漂移通常被定义为随时间而来的入站数据流的统计属性的不可预见性变化。目前的概念漂移检测器表现很好，即存在较低的假阴性率，但它们仍然倾向于在概念漂移检测中表现出较高的假阳性率。本文提出了OPTWIN，即“OPTimal WINdow”概念漂移检测器。OPTWIN使用滑动窗口方法分析数据流的子窗口，以更高的准确性和更低的假阳性率检测概念漂移。OPTWIN的实验评估显示它在假阳性和真阳性率之间达到了更好的平衡，优于现有的检测器。

    Online Learning (OL) is a field of research that is increasingly gaining attention both in academia and industry. One of the main challenges of OL is the inherent presence of concept drifts, which are commonly defined as unforeseeable changes in the statistical properties of an incoming data stream over time. The detection of concept drifts typically involves analyzing the error rates produced by an underlying OL algorithm in order to identify if a concept drift occurred or not, such that the OL algorithm can adapt accordingly. Current concept-drift detectors perform very well, i.e., with low false negative rates, but they still tend to exhibit high false positive rates in the concept-drift detection. This may impact the performance of the learner and result in an undue amount of computational resources spent on retraining a model that actually still performs within its expected range. In this paper, we propose OPTWIN, our "OPTimal WINdow" concept drift detector. OPTWIN uses a sliding 
    
[^52]: 基于归纳流的快速粒子探测器模拟框架

    Inductive CaloFlow. (arXiv:2305.11934v1 [physics.ins-det])

    [http://arxiv.org/abs/2305.11934](http://arxiv.org/abs/2305.11934)

    iCaloFlow是一个基于归纳流的快速探测器模拟框架，可以以高达以往10-100倍的分辨率进行快速、高保真度模拟。

    

    模拟粒子探测器响应是大型强子对撞机计算流程中最昂贵的步骤。最近的研究表明，归一化流可以加快此过程，并实现前所未有的精度水平，但将此方法扩展到与未来探测器升级相关的更高分辨率时会导致限制性的内存约束。为了克服这个问题，我们介绍了基于归纳系列归一化流的快速探测器模拟框架iCaloFlow，它是在成对的连续能量沉积层中训练的。为了增加采样速度而不失表现力，我们进一步使用师生蒸馏。正如我们在CaloChallenge2022的数据集2和数据集3中展示的那样，iCaloFlow可以实现归一化流在进行快速、高保真度模拟时的潜力，这些模拟对应的探测器几何约比以前考虑的高10-100倍。

    Simulating particle detector response is the single most expensive step in the Large Hadron Collider computational pipeline. Recently it was shown that normalizing flows can accelerate this process while achieving unprecedented levels of accuracy, but scaling this approach up to higher resolutions relevant for future detector upgrades leads to prohibitive memory constraints. To overcome this problem, we introduce Inductive CaloFlow (iCaloFlow), a framework for fast detector simulation based on an inductive series of normalizing flows trained on the pattern of energy depositions in pairs of consecutive calorimeter layers. We further use a teacher-student distillation to increase sampling speed without loss of expressivity. As we demonstrate with Datasets 2 and 3 of the CaloChallenge2022, iCaloFlow can realize the potential of normalizing flows in performing fast, high-fidelity simulation on detector geometries that are ~ 10 - 100 times higher granularity than previously considered.
    
[^53]: PyTorch的超参数调整——面向spotPython的教程

    PyTorch Hyperparameter Tuning -- A Tutorial for spotPython. (arXiv:2305.11930v1 [cs.LG])

    [http://arxiv.org/abs/2305.11930](http://arxiv.org/abs/2305.11930)

    本文介绍了如何将spotPython超参数调谐器集成到PyTorch训练工作流中，以提高机器或深度学习模型的性能，以CIFAR10图像分类器为例。

    

    超参数调整（或超参数优化）的目标是优化超参数以提高机器或深度学习模型的性能。spotPython是知名超参数调谐器SPOT的Python版本，SPOT已经在R编程环境中为统计分析开发了十年以上。PyTorch是一种基于GPU和CPU的深度学习优化张量库。本文展示了如何将spotPython超参数调谐器集成到PyTorch训练工作流中。以CIFAR10图像分类器为例，介绍了spotPython以及与Ray Tune的简短比较。本文讨论了两种方法的优缺点。我们展示了spotPython的使用经验，以及如何使用hook在训练过程中自动调整参数。

    The goal of hyperparameter tuning (or hyperparameter optimization) is to optimize the hyperparameters to improve the performance of the machine or deep learning model. spotPython (``Sequential Parameter Optimization Toolbox in Python'') is the Python version of the well-known hyperparameter tuner SPOT, which has been developed in the R programming environment for statistical analysis for over a decade. PyTorch is an optimized tensor library for deep learning using GPUs and CPUs. This document shows how to integrate the spotPython hyperparameter tuner into the PyTorch training workflow. As an example, the results of the CIFAR10 image classifier are used. In addition to an introduction to spotPython, this tutorial also includes a brief comparison with Ray Tune, a Python library for running experiments and tuning hyperparameters. This comparison is based on the PyTorch hyperparameter tuning tutorial. The advantages and disadvantages of both approaches are discussed. We show that spotPytho
    
[^54]: 使用学习自动机的节能且可解释AI硬件设计

    Energy-frugal and Interpretable AI Hardware Design using Learning Automata. (arXiv:2305.11928v1 [cs.AI])

    [http://arxiv.org/abs/2305.11928](http://arxiv.org/abs/2305.11928)

    本论文通过使用学习自动机实现了节能的AI硬件设计，同时保持了模型的解释性和准确性。

    

    在微边缘计算环境下，能效是实现强大人工智能应用的重要需求。通过节能的计算资源配置实现硬件加速是降低能耗的有效方法。然而，许多新兴应用还需要采用可解释决策模型，以确立责任和透明度。在真实数据场景中提供可达状态需要额外的资源，这给能效设计带来了冲突性的挑战。最近，提出了一种新的机器学习算法——Tsetlin机器，该算法基于有限状态自动机原理，与算术不同，受益于自然逻辑支撑。本文研究了如何通过适当调整超参数来实现节能的人工智能硬件设计，并保持高效的学习效果。为了展示其潜力，我们在不同的优化技术下在可编程逻辑门阵列（FPGAs）上实现了Tsetlin机器算法，并使用标准基准数据集评估其性能。实验结果表明，通过利用学习自动机，我们可以在不牺牲模型的解释性和准确性的情况下，实现显著的能源节约。

    Energy efficiency is a crucial requirement for enabling powerful artificial intelligence applications at the microedge. Hardware acceleration with frugal architectural allocation is an effective method for reducing energy. Many emerging applications also require the systems design to incorporate interpretable decision models to establish responsibility and transparency. The design needs to provision for additional resources to provide reachable states in real-world data scenarios, defining conflicting design tradeoffs between energy efficiency. is challenging.  Recently a new machine learning algorithm, called the Tsetlin machine, has been proposed. The algorithm is fundamentally based on the principles of finite-state automata and benefits from natural logic underpinning rather than arithmetic. In this paper, we investigate methods of energy-frugal artificial intelligence hardware design by suitably tuning the hyperparameters, while maintaining high learning efficacy. To demonstrate i
    
[^55]: 计算机视觉模型哪些地方会出错？使用交互式可视化工具找出并改进CV模型

    Where does a computer vision model make mistakes? Using interactive visualizations to find where and how CV models can improve. (arXiv:2305.11927v1 [cs.HC])

    [http://arxiv.org/abs/2305.11927](http://arxiv.org/abs/2305.11927)

    研究使用交互式可视化工具在创建计算机视觉分类和检测模型时帮助用户识别和改进模型上的问题，有效减少设计师的工作量。

    

    创建计算机视觉模型仍然是一个复杂和繁琐的过程，而使终端用户可以构建、检查和改进这些模型的交互式机器学习模型视角已经在解决这些问题方面取得了一些进展。为了提高具有不同级别机器学习专业技能的终端用户的体验，我们在Sprite的上下文中设计和评估了两个交互式可视化工具，这是一个用于为从视频中抽取的图像创建CV分类和检测模型的系统。我们研究了这些可视化工具如何作为机器学习循环的一部分，帮助用户识别（评估）和选择（规划）模型存在问题的图像，并改善正在训练的模型。我们发现，使用这些可视化工具的用户在更广泛的模型错误类型和一个或多个模型的预测行为的评估和比较方面发现了更多的图像，从而减少了设计师创建和改进CV模型所需的潜在工作量。

    Creating Computer Vision (CV) models remains a complex and taxing practice for end-users to build, inspect, and improve these models. Interactive ML perspectives have helped address some of these issues by considering a teacher-in-the-loop where planning, teaching, and evaluating tasks take place. To improve the experience of end-users with various levels of ML expertise, we designed and evaluated two interactive visualizations in the context of Sprite, a system for creating CV classification and detection models for images originating from videos. We study how these visualizations, as part of the machine teaching loop, help users identify (evaluate) and select (plan) images where a model is struggling and improve the model being trained. We found that users who had used the visualizations found more images across a wider set of potential types of model errors, as well as in assessing and contrasting the prediction behavior of one or more models, thus reducing the potential effort requ
    
[^56]: MParrotTTS：低资源环境下的多语言多说话人文本转语音合成

    MParrotTTS: Multilingual Multi-speaker Text to Speech Synthesis in Low Resource Setting. (arXiv:2305.11926v1 [cs.SD])

    [http://arxiv.org/abs/2305.11926](http://arxiv.org/abs/2305.11926)

    MParrotTTS是一个统一的多语言、多说话人文本转语音合成模型，以自监督语音表示为基础；它可以在低资源环境中仅使用少量有监督数据就适应于新语言，并在不需要平行或双语语料的情况下传递说话人特定的语音特征。

    

    我们介绍了MParrotTTS，这是一个统一的多语言、多说话人文本转语音(TTS)合成模型，可以产生高质量的语音。MParrotTTS受益于模块化培训范式，利用自监督语音表示，以最小的监督数据适应于新语言，并在训练自监督后骨干中对未见过的语言进行泛化。此外，MParrotTTS不需要任何双语或平行示例的训练，可以在语音中传递语音，同时保留说话人的特定特征，例如使用法语演讲者的声音和口音合成流利的印地语语音。我们在六种语言上提出了广泛的结果，包括并行和跨语言综合的语音自然度和说话人相似度。所提出的模型在只使用少量监督训练数据的情况下，优于最先进的多语言TTS模型和基线。我们的模型可在https://paper2438.github.io/tts上找到。

    We present MParrotTTS, a unified multilingual, multi-speaker text-to-speech (TTS) synthesis model that can produce high-quality speech. Benefiting from a modularized training paradigm exploiting self-supervised speech representations, MParrotTTS adapts to a new language with minimal supervised data and generalizes to languages not seen while training the self-supervised backbone. Moreover, without training on any bilingual or parallel examples, MParrotTTS can transfer voices across languages while preserving the speaker-specific characteristics, e.g., synthesizing fluent Hindi speech using a French speaker's voice and accent. We present extensive results on six languages in terms of speech naturalness and speaker similarity in parallel and cross-lingual synthesis. The proposed model outperforms the state-of-the-art multilingual TTS models and baselines, using only a small fraction of supervised training data. Speech samples from our model can be found at https://paper2438.github.io/tts
    
[^57]: 一种对比集合稳定无误差的多重比较基准评估方法

    An Approach to Multiple Comparison Benchmark Evaluations that is Stable Under Manipulation of the Comparate Set. (arXiv:2305.11921v1 [stat.ME])

    [http://arxiv.org/abs/2305.11921](http://arxiv.org/abs/2305.11921)

    本文提出了一种新的基准比较结果展示方法——多元比较矩阵（MCM），使得比较集合稳定无误差，可避免常用方法存在的无意和有意的操纵空间，并且其采用Python实现，已在公开提供。

    

    基准评估是计算机科学和机器学习中广泛使用的衡量进步的方法。然而，目前常用的方法对于多个算法在多个数据集上的基准比较结果分析和展示，如Dem\v{s}ar（2006）引入的关键差异图存在重大缺陷，并且我们发现这些方法存在无意和有意的操纵空间。为了解决这些问题，我们提出了一种新的基准比较结果展示方法——多元比较矩阵（MCM），该方法优先考虑成对比较，排除了现有方法中操纵实验结果的方式。MCM可用于显示全对比结果，或显示一个或多个选择的算法与技术的对比结果。MCM采用Python实现，并公开提供。

    The measurement of progress using benchmarks evaluations is ubiquitous in computer science and machine learning. However, common approaches to analyzing and presenting the results of benchmark comparisons of multiple algorithms over multiple datasets, such as the critical difference diagram introduced by Dem\v{s}ar (2006), have important shortcomings and, we show, are open to both inadvertent and intentional manipulation. To address these issues, we propose a new approach to presenting the results of benchmark comparisons, the Multiple Comparison Matrix (MCM), that prioritizes pairwise comparisons and precludes the means of manipulating experimental results in existing approaches. MCM can be used to show the results of an all-pairs comparison, or to show the results of a comparison between one or more selected algorithms and the state of the art. MCM is implemented in Python and is publicly available.
    
[^58]: 可解释的神经架构搜索与迁移学习用于理解序列依赖的酶反应

    Interpretable neural architecture search and transfer learning for understanding sequence dependent enzymatic reactions. (arXiv:2305.11917v1 [q-bio.MN])

    [http://arxiv.org/abs/2305.11917](http://arxiv.org/abs/2305.11917)

    Elektrum是一个深度学习框架，使用可解释的神经网络模型预测酶反应，利用有限但洁净的体外数据和噪声但丰富的体内数据。Elektrum可以通过迁移学习揭示酶活性的关键序列相关决定因素，并发现潜在的治疗干预靶点。

    

    精细调节的酶途径控制着细胞过程，它们的失调可能导致疾病。为这些途径创建预测性和可解释性模型具有挑战性，因为这些途径的复杂性以及细胞和基因组背景的复杂性。在这里，我们介绍了Elektrum，一个深度学习框架，通过数据驱动和生物物理解释模型，确定生化系统动力学，从而解决这些挑战。首先，它使用体外动力学测定快速假设高质量的可解释动力学神经网络（KINN），用于预测反应速率。然后，利用新颖的迁移学习步骤，将KINN作为中介层插入更深的卷积神经网络中，微调反应相关的体内结果的预测。Elektrum有效利用了有限但洁净的体外数据和捕获细胞背景的噪声但丰富的体内数据。我们将Elektrum应用于理解与非酒精性脂肪肝病相关的碳水化合物和脂质代谢中涉及的酶反应。我们证明Elektrum利用迁移学习（1）优于最先进的模型预测体内反应速率; (2) 揭示酶活性的关键序列相关决定因素; 以及（3）发现治疗干预的潜在靶点。

    Finely-tuned enzymatic pathways control cellular processes, and their dysregulation can lead to disease. Creating predictive and interpretable models for these pathways is challenging because of the complexity of the pathways and of the cellular and genomic contexts. Here we introduce Elektrum, a deep learning framework which addresses these challenges with data-driven and biophysically interpretable models for determining the kinetics of biochemical systems. First, it uses in vitro kinetic assays to rapidly hypothesize an ensemble of high-quality Kinetically Interpretable Neural Networks (KINNs) that predict reaction rates. It then employs a novel transfer learning step, where the KINNs are inserted as intermediary layers into deeper convolutional neural networks, fine-tuning the predictions for reaction-dependent in vivo outcomes. Elektrum makes effective use of the limited, but clean in vitro data and the noisy, yet plentiful in vivo data that captures cellular context. We apply Ele
    
[^59]: 在$\mathbb{R}$-光滑Banach空间中，PINNs误差估计非线性方程的研究

    PINNs error estimates for nonlinear equations in $\mathbb{R}$-smooth Banach spaces. (arXiv:2305.11915v1 [math.FA])

    [http://arxiv.org/abs/2305.11915](http://arxiv.org/abs/2305.11915)

    本文研究了在$\mathbb{R}$-光滑Banach空间中支持PINNs误差估计的非线性方程，提出了一种可用于限制残差的Bramble-Hilbert引理。

    

    本文以算子形式描述了一类支持PINN误差估计的PDE，并且对于$L^p$空间，我们得到了一个Bramble-Hilbert引理，作为与PINN残差边界的工具。

    In the paper, we describe in operator form classes of PDEs that admit PINN's error estimation. Also, for $L^p$ spaces, we obtain a Bramble-Hilbert type lemma that is a tool for PINN's residuals bounding.
    
[^60]: 用于从稀疏远程传感器数据重建非线性海洋波浪表面相位的机器学习方法

    Machine learning for phase-resolved reconstruction of nonlinear ocean wave surface elevations from sparse remote sensing data. (arXiv:2305.11913v1 [physics.ao-ph])

    [http://arxiv.org/abs/2305.11913](http://arxiv.org/abs/2305.11913)

    本文提出了一种基于神经网络的方法，利用高度现实的合成训练数据对稀疏雷达数据进行相位相关的波浪表面重建。

    

    准确预测相位相关的水波条件对于海洋工程的决策至关重要。然而，远程监测波浪预测模型的初始化首先需要从类似雷达的稀疏测量中重建波浪表面。现有的重建方法要么依赖于计算密集型的优化过程，要么依赖于简化的模型假设，这会影响整个预测过程的实时性或准确性。因此，我们提出了一种基于U-Net和Fourier神经算子（FNO）结构的神经网络方法，用于相位相关的波浪表面重建。我们的方法利用具有高度现实性的合成训练数据，这些数据在均匀的一维网格上由波浪模拟的高阶谱方法和几何雷达建模方法生成。研究结果表明，两种模型都可以提供准确的波浪重建结果。

    Accurate short-term prediction of phase-resolved water wave conditions is crucial for decision-making in ocean engineering. However, the initialization of remote-sensing-based wave prediction models first requires a reconstruction of wave surfaces from sparse measurements like radar. Existing reconstruction methods either rely on computationally intensive optimization procedures or simplistic modeling assumptions that compromise real-time capability or accuracy of the entire prediction process. We therefore address these issues by proposing a novel approach for phase-resolved wave surface reconstruction using neural networks based on the U-Net and Fourier neural operator (FNO) architectures. Our approach utilizes synthetic yet highly realistic training data on uniform one-dimensional grids, that is generated by the high-order spectral method for wave simulation and a geometric radar modeling approach. The investigation reveals that both models deliver accurate wave reconstruction resul
    
[^61]: 机器学习与VIIRS卫星检索在野火管理的燃料湿度监测中的技术应用

    Machine Learning and VIIRS Satellite Retrievals for Skillful Fuel Moisture Content Monitoring in Wildfire Management. (arXiv:2305.11910v1 [cs.LG])

    [http://arxiv.org/abs/2305.11910](http://arxiv.org/abs/2305.11910)

    本研究利用机器学习模型结合国家水资源模型和数值天气预测模型，以及卫星检索来预测美国连续本土上的死亡燃料湿度检索，超过了既有的每日和每小时气候统计方法。VIIRS检索对预测FMC有重要贡献。

    

    监测植被燃料湿度对于野火的管理和减轻影响至关重要。利用野外燃料湿度观测、数值天气预报模型和卫星检索相结合的方法，开发了机器学习模型来估计美国连续本土上的死亡燃料湿度检索。本研究利用National Water Model和High-Resolution Rapid Refresh（HRRR）数值天气预报模型的变量以及表面属性，以及Suomi-NPP卫星系统上的VIIRS仪器的表面反射率和陆地表面温度检索，训练了机器学习模型。通过广泛的超参调优，与每日气候统计误差（+44％）和每小时气候统计误差（+24％）相比，得到了技术娴熟的FMC模型。此外，VIIRS检索作为一个群体对估计FMC是重要的预测因子，有显著的贡献。

    Monitoring the fuel moisture content (FMC) of vegetation is crucial for managing and mitigating the impact of wildland fires. The combination of in situ FMC observations with numerical weather prediction (NWP) models and satellite retrievals has enabled the development of machine learning (ML) models to estimate dead FMC retrievals over the contiguous US (CONUS). In this study, ML models were trained using variables from the National Water Model and the High-Resolution Rapid Refresh (HRRR) NWP models, and static variables characterizing the surface properties, as well as surface reflectances and land surface temperature (LST) retrievals from the VIIRS instrument on board the Suomi-NPP satellite system. Extensive hyper-parameter optimization yielded skillful FMC models compared to a daily climatography RMSE (+44\%) and to an hourly climatography RMSE (+24\%). Furthermore, VIIRS retrievals were important predictors for estimating FMC, contributing significantly as a group due to their hi
    
[^62]: 序列最优臂识别及其在脑-机接口中的应用

    Sequential Best-Arm Identification with Application to Brain-Computer Interface. (arXiv:2305.11908v1 [cs.HC])

    [http://arxiv.org/abs/2305.11908](http://arxiv.org/abs/2305.11908)

    本论文提出了一种序列最优臂识别方法，应用于脑-机接口中的拼写系统。利用预训练的大型语言模型，可以更快地进行学习并提高信息传输速率。

    

    脑-机接口是一种使大脑与外部设备或计算机系统直接通信的技术，它允许个体只使用思维与设备进行交互，并具有在医学、康复和人体增强等领域中广泛应用的潜力。 基于脑电图（EEG）和事件相关电位（ERP）的拼写器系统是一种类型的脑-机接口，它允许用户在不使用物理键盘的情况下拼写单词，而是通过记录和解释在不同的刺激呈现范例下的脑信号。传统的非自适应范例将每个单词选择视为独立的，导致了漫长的学习过程。为了提高采样效率，我们将问题转化为多臂老虎机中一系列最优臂识别任务。利用预训练的大型语言模型（LLM），我们利用从先前任务中学习到的先验知识来通知和促进后续任务。我们提出的方法与现有方法相比具有更快的学习速度和更高的信息传输速率。我们在模拟ERP拼写实验和真实的EEG打字任务中展示了我们方法的有效性。

    A brain-computer interface (BCI) is a technology that enables direct communication between the brain and an external device or computer system. It allows individuals to interact with the device using only their thoughts, and holds immense potential for a wide range of applications in medicine, rehabilitation, and human augmentation. An electroencephalogram (EEG) and event-related potential (ERP)-based speller system is a type of BCI that allows users to spell words without using a physical keyboard, but instead by recording and interpreting brain signals under different stimulus presentation paradigms. Conventional non-adaptive paradigms treat each word selection independently, leading to a lengthy learning process. To improve the sampling efficiency, we cast the problem as a sequence of best-arm identification tasks in multi-armed bandits. Leveraging pre-trained large language models (LLMs), we utilize the prior knowledge learned from previous tasks to inform and facilitate subsequent
    
[^63]: 关于ENCE和其他基于MAD的校准度量的特性

    Properties of the ENCE and other MAD-based calibration metrics. (arXiv:2305.11905v1 [cs.LG])

    [http://arxiv.org/abs/2305.11905](http://arxiv.org/abs/2305.11905)

    本文讨论ENEC和基于z分数（ZVE）方差的校准误差；指出在校准良好或几乎校准的数据集上，误差与分组数量的平方根成比例关系，提出一种解决方案以推断ENCE和ZVE的值，并提供统计校准测试。

    

    「期望归一化校准误差（ENCE）」是机器学习中用于评估回归问题预测不确定性质量的常见校准统计量，其估计基于校准数据的分组。本文展示了ENCE的一个令人恼火的特性，即在校准良好或几乎校准的数据集上，它与分组数量的平方根成比例关系。类似的行为还影响了基于z分数（ZVE）方差的校准误差，并且在这两种情况下，此特性是使用平均绝对偏差（MAD）统计量估计校准误差的结果。因此，如何选择分组数以可靠地估计校准误差统计量成为一个问题。提出了一种解决方案，用于推断ENCE和ZVE的值，假设数据集已经校准，并同时提供统计校准测试。同时还表明，对于不断增加的分组密度，ZVE在渐近意义下等价于ENCE。

    The Expected Normalized Calibration Error (ENCE) is a popular calibration statistic used in Machine Learning to assess the quality of prediction uncertainties for regression problems. Estimation of the ENCE is based on the binning of calibration data. In this short note, I illustrate an annoying property of the ENCE, i.e. its proportionality to the square root of the number of bins for well calibrated or nearly calibrated datasets. A similar behavior affects the calibration error based on the variance of z-scores (ZVE), and in both cases this property is a consequence of the use of a Mean Absolute Deviation (MAD) statistic to estimate calibration errors. Hence, the question arises of which number of bins to choose for a reliable estimation of calibration error statistics. A solution is proposed to infer ENCE and ZVE values that do not depend on the number of bins for datasets assumed to be calibrated, providing simultaneously a statistical calibration test. It is also shown that the ZV
    
[^64]: 利用海洋记忆效应预测中国南方冬季空气稳定指数的长期预报

    Long-lead forecasts of wintertime air stagnation index in southern China using oceanic memory effects. (arXiv:2305.11901v1 [physics.ao-ph])

    [http://arxiv.org/abs/2305.11901](http://arxiv.org/abs/2305.11901)

    该研究基于海洋记忆效应开发了一个LSTM模型，结合过去的ASI和尼娜指数可以实现更好的ASI预测，为提前制定空气质量管理计划提供了帮助。

    

    不利于空气污染物的稀释和清除，是导致空气污染的主要因素之一。空气稳定指数（ASI）是测量大气清除空气污染物能力的一项重要气象指标。因此，进行长期ASI预报对于提前制定空气质量管理计划至关重要。本研究发现，由海表温度异常推导出的秋季尼娜指数与中国南方冬季ASI呈负相关，为预测冬季ASI提供了前景。我们开发了一个基于LSTM的模型来预测未来的ASI。结果表明，多元输入（过去的ASI和尼娜指数）比单元输入（仅过去的ASI）具有更好的预测性能。该模型在实际和预测ASI之间实现了0.778的相关系数，表现出高度的一致性。

    Stagnant weather condition is one of the major contributors to air pollution as it is favorable for the formation and accumulation of pollutants. To measure the atmosphere's ability to dilute air pollutants, Air Stagnation Index (ASI) has been introduced as an important meteorological index. Therefore, making long-lead ASI forecasts is vital to make plans in advance for air quality management. In this study, we found that autumn Ni\~no indices derived from sea surface temperature (SST) anomalies show a negative correlation with wintertime ASI in southern China, offering prospects for a prewinter forecast. We developed an LSTM-based model to predict the future wintertime ASI. Results demonstrated that multivariate inputs (past ASI and Ni\~no indices) achieve better forecast performance than univariate input (only past ASI). The model achieves a correlation coefficient of 0.778 between the actual and predicted ASI, exhibiting a high degree of consistency.
    
[^65]: 一种使用Particle Photon和智能手机的自动化节能系统（APCS）

    An Automated Power Conservation System (APCS) using Particle Photon and Smartphone. (arXiv:2305.11889v1 [cs.HC])

    [http://arxiv.org/abs/2305.11889](http://arxiv.org/abs/2305.11889)

    本文介绍了一种使用Particle Photon和智能手机的自动化节能系统（APCS）。该系统使用IR传感器检测教室内人员的存在，并自动控制灯和风扇的开关，节省电力消耗和节约宝贵自然资源。通过智能手机应用程序对系统进行控制和监控，易于实施和维护。

    

    如今，人们在生活的各个方面都使用电力，因此电力消耗逐渐增加。由于人为疏忽、日光等各种原因，电力可能会浪费。因此，节约能源是当务之急。本文介绍了一种“自动化节能系统（APCS）”的制作，具有多种好处，如节省电力消耗，从而节省组织的电费，消除人为干预和手动开关灯和电气装置所需的人力，最重要的是通过减少电气能耗来节约宝贵的自然资源。该项目使用了两个IR传感器，用于检测教室中人员的存在。当APCS检测到有人存在时，它会自动打开教室中的风扇和灯，在人离开时自动关闭。该系统通过智能手机应用程序进行控制和监控，方便用户随时访问系统。由于易于实施和维护，APCS已被证明是高效和具有成本效益的。

    Nowadays, people use electricity in all aspects of their lives so that electricity consumption increases gradually. There can be wastage of electricity due to various reasons, such as human negligence, daylighting, etc. Hence, conservation of energy is the need of the day. This paper deals with the fabrication of an "Automated Power Conservation System (APCS)" that has multiple benefits like saving on power consumption there by saving on electricity bills of the organization, eliminating human involvement and manpower which is often required to manually toggle the lights and electrical devices on/off, and last but most importantly conserve the precious natural resources by reducing electrical energy consumption. Two IR sensors are used in this project and these two sensors are used for detecting the presence of a person in the classroom. When the existence of the person is detected by the APCS it automatically turns on the fans and lights in that classroom and during the absence they w
    
[^66]: 基于图注意力和频率增强机制的短期风速预测方法

    Enhancing Short-Term Wind Speed Forecasting using Graph Attention and Frequency-Enhanced Mechanisms. (arXiv:2305.11526v1 [cs.LG])

    [http://arxiv.org/abs/2305.11526](http://arxiv.org/abs/2305.11526)

    本文提出了一种基于图注意力和频率增强机制的风速预测模型GFST-WSF，能够有效提高短期风速预测的准确性。

    

    在风电大规模集成电网中，风力的高可变性和随机性对电力系统的安全和稳定运行带来了巨大挑战。风力预测是解决这个问题的有效方法，其中风速预测是至关重要的方面。本文提出了一种基于图注意力和频率增强机制的图注意力频率增强时空风速预测模型（GFST-WSF），以提高短期风速预测的准确性。GFST-WSF包括用于提取时间特征的Transformer架构和用于提取空间特征的图注意力网络（GAT）。GAT被专门设计用于捕捉风速站之间的复杂空间依赖关系，从而有效地聚合图中相邻节点的信息，从而增强数据的空间表示。为了模拟邻近风场之间的风速相关的时间滞后

    The safe and stable operation of power systems is greatly challenged by the high variability and randomness of wind power in large-scale wind-power-integrated grids. Wind power forecasting is an effective solution to tackle this issue, with wind speed forecasting being an essential aspect. In this paper, a Graph-attentive Frequency-enhanced Spatial-Temporal Wind Speed Forecasting model based on graph attention and frequency-enhanced mechanisms, i.e., GFST-WSF, is proposed to improve the accuracy of short-term wind speed forecasting. The GFST-WSF comprises a Transformer architecture for temporal feature extraction and a Graph Attention Network (GAT) for spatial feature extraction. The GAT is specifically designed to capture the complex spatial dependencies among wind speed stations to effectively aggregate information from neighboring nodes in the graph, thus enhancing the spatial representation of the data. To model the time lag in wind speed correlation between adjacent wind farms cau
    
[^67]: 追赶蒸馏：加速采样只需一次训练

    Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling. (arXiv:2305.10769v1 [cs.LG])

    [http://arxiv.org/abs/2305.10769](http://arxiv.org/abs/2305.10769)

    本文提出了一种名为“追赶蒸馏”的方法，通过调整传统采样算法，让速度估计模型的当前时刻输出与其先前时刻输出和地面真实标签对齐，从而实现只需一次训练便能加速采样的效果。

    

    扩散概率模型在各种机器学习领域取得了令人瞩目的进展。然而，为了实现高质量的合成样本，通常需要执行大量的采样步骤，这阻碍了实时样本合成的可能性。传统的通过知识蒸馏加速采样的算法依赖于预训练的模型权重和离散时间步骤场景，需要额外的培训课程才能实现他们的目标。为了解决这些问题，我们提出了追赶蒸馏（CUD），它鼓励速度估计模型的当前时刻输出“追赶”其先前时刻输出。具体而言，CUD调整了原始的常微分方程（ODE）训练目标，以使当前时刻输出与地面真实标签和先前时刻输出对齐，利用基于龙格-库塔的多步对齐蒸馏进行精确的ODE估计，同时防止异步更新。

    Diffusion Probability Models (DPMs) have made impressive advancements in various machine learning domains. However, achieving high-quality synthetic samples typically involves performing a large number of sampling steps, which impedes the possibility of real-time sample synthesis. Traditional accelerated sampling algorithms via knowledge distillation rely on pre-trained model weights and discrete time step scenarios, necessitating additional training sessions to achieve their goals. To address these issues, we propose the Catch-Up Distillation (CUD), which encourages the current moment output of the velocity estimation model ``catch up'' with its previous moment output. Specifically, CUD adjusts the original Ordinary Differential Equation (ODE) training objective to align the current moment output with both the ground truth label and the previous moment output, utilizing Runge-Kutta-based multi-step alignment distillation for precise ODE estimation while preventing asynchronous updates
    
[^68]: 针对非稳态赌博机问题的折扣汤普森抽样算法

    Discounted Thompson Sampling for Non-Stationary Bandit Problems. (arXiv:2305.10718v1 [cs.LG])

    [http://arxiv.org/abs/2305.10718](http://arxiv.org/abs/2305.10718)

    该论文提出了一种针对非稳态多臂赌博机问题的折扣汤普森抽样算法（DS-TS），可以解决突然性变化和平滑性变化的问题，并且在两种情况下具有近乎最优的遗憾上限。

    

    近年来，非稳态多臂赌博机问题受到了显著关注。NS-MAB通常在两种情况下进行建模：突然性变化和平滑性变化。在本文中，我们提出了带有高斯先验的折扣汤普森采样算法（DS-TS）以解决这两个非稳态设置。我们的算法通过将折扣因子纳入汤普森采样来被动适应变化。DS-TS方法经过实验验证，但缺乏对遗憾上限的分析。在温和的假设下，我们证明了带有高斯先验的DS-TS可以在突然性变化的情况下实现近乎最优的遗憾上限（$\tilde{O} (\sqrt {TB_T})$），在平滑性变化的情况下实现 $\tilde{O}(T^{\beta})$ 的近乎最优遗憾上限，其中 $T$ 是时间步数，$B_T$ 是断点数，$\beta$ 与收益分布的平滑性有关，$\tilde{O}$ 是对数遗憾上限。

    Non-stationary multi-armed bandit (NS-MAB) problems have recently received significant attention. NS-MAB are typically modelled in two scenarios: abruptly changing, where reward distributions remain constant for a certain period and change at unknown time steps, and smoothly changing, where reward distributions evolve smoothly based on unknown dynamics. In this paper, we propose Discounted Thompson Sampling (DS-TS) with Gaussian priors to address both non-stationary settings. Our algorithm passively adapts to changes by incorporating a discounted factor into Thompson Sampling. DS-TS method has been experimentally validated, but analysis of the regret upper bound is currently lacking. Under mild assumptions, we show that DS-TS with Gaussian priors can achieve nearly optimal regret bound on the order of $\tilde{O}(\sqrt{TB_T})$ for abruptly changing and $\tilde{O}(T^{\beta})$ for smoothly changing, where $T$ is the number of time steps, $B_T$ is the number of breakpoints, $\beta$ is asso
    
[^69]: 张量积与超维计算

    Tensor Products and Hyperdimensional Computing. (arXiv:2305.10572v1 [stat.ML])

    [http://arxiv.org/abs/2305.10572](http://arxiv.org/abs/2305.10572)

    本文探索了张量积在超维计算中的数学关系，将其确定为中心表示，并发现它是最通用、最具表现力和最压缩的表示，同时具有无误差解绑和检测的能力。

    

    在之前对图嵌入的分析基础上，我们将一些结果推广和拓展到向量符号结构 (VSA) 和超维计算 (HDC) 的一般设置中。重要的是，我们探索超叠加、正交和张量积之间的数学关系。我们将张量积表示确定为中心表示，并具有一套独特的属性。这包括它是最通用和最具表现力的表示，也是最压缩的表示，具有无误差解绑和检测的能力。

    Following up on a previous analysis of graph embeddings, we generalize and expand some results to the general setting of vector symbolic architectures (VSA) and hyperdimensional computing (HDC). Importantly, we explore the mathematical relationship between superposition, orthogonality, and tensor product. We establish the tensor product representation as the central representation, with a suite of unique properties. These include it being the most general and expressive representation, as well as being the most compressed representation that has errorrless unbinding and detection.
    
[^70]: 具有距离感知自注意力的深度多示例学习

    Deep Multiple Instance Learning with Distance-Aware Self-Attention. (arXiv:2305.10552v1 [cs.CV])

    [http://arxiv.org/abs/2305.10552](http://arxiv.org/abs/2305.10552)

    本文提出具有距离感知自注意力的深度多示例学习模型，该模型能根据补丁之间的空间关系动态调整权重，从而在多个基准数据集上提高了分类性能。

    

    传统的监督学习任务要求对训练集中的每个实例进行标记，但在许多实际应用中，标记仅对实例的集合（包）可用。这种问题设置被称为多重示例学习（MIL），在医疗领域尤其相关，高分辨率图像被分成较小的补丁，但标签适用于整个图像。最近的MIL模型能够通过采用自我关注来捕捉补丁之间的对应关系，使它们能够根据包中所有其他补丁对每个补丁进行不同的加权。然而，这些方法仍然没有考虑较大图像中补丁之间的相对空间关系，这在计算病理学中尤为重要。为此，我们引入了一种新的MIL模型，具有距离感知自注意力（DAS-MIL），它在建模补丁之间的交互作用时明确考虑相对空间信息。与现有相关模型不同，DAS-MIL使用距离感知注意机制根据补丁之间的距离动态调整补丁权重，从而提高了在多个基准数据集上的分类性能。

    Traditional supervised learning tasks require a label for every instance in the training set, but in many real-world applications, labels are only available for collections (bags) of instances. This problem setting, known as multiple instance learning (MIL), is particularly relevant in the medical domain, where high-resolution images are split into smaller patches, but labels apply to the image as a whole. Recent MIL models are able to capture correspondences between patches by employing self-attention, allowing them to weigh each patch differently based on all other patches in the bag. However, these approaches still do not consider the relative spatial relationships between patches within the larger image, which is especially important in computational pathology. To this end, we introduce a novel MIL model with distance-aware self-attention (DAS-MIL), which explicitly takes into account relative spatial information when modelling the interactions between patches. Unlike existing rela
    
[^71]: 采用极低频智能电表时间序列进行家电检测

    Appliance Detection Using Very Low-Frequency Smart Meter Time Series. (arXiv:2305.10352v1 [eess.SP])

    [http://arxiv.org/abs/2305.10352](http://arxiv.org/abs/2305.10352)

    本文对时间序列分类器在极低频智能电表数据中检测不同家电的存在/缺失进行了深入评估和比较，结果表明......

    

    近年来，智能电表被广泛采用，以改善智能电网系统的管理，这些电表通常以极低的频率（每30分钟）收集能源消耗数据，以更准确地向客户计费。为了提供更个性化的建议，下一步是检测客户拥有的家电，由于极低的计量读数频率，这是一个具有挑战性的问题。尽管家电检测问题可以被视为时间序列分类问题，并且已经在文献中提出了许多这样的分类器，但没有研究将它们应用于这个具体的问题并进行比较。本文提出了对最新时间序列分类器在极低频智能电表数据中检测不同家电的存在/缺失进行深入评估和比较的研究。我们报告了5个真实数据集的结果。我们首先研究了13个时序分类器检测质量的影响。

    In recent years, smart meters have been widely adopted by electricity suppliers to improve the management of the smart grid system. These meters usually collect energy consumption data at a very low frequency (every 30min), enabling utilities to bill customers more accurately. To provide more personalized recommendations, the next step is to detect the appliances owned by customers, which is a challenging problem, due to the very-low meter reading frequency. Even though the appliance detection problem can be cast as a time series classification problem, with many such classifiers having been proposed in the literature, no study has applied and compared them on this specific problem. This paper presents an in-depth evaluation and comparison of state-of-the-art time series classifiers applied to detecting the presence/absence of diverse appliances in very low-frequency smart meter data. We report results with five real datasets. We first study the impact of the detection quality of 13 di
    
[^72]: 深度学习中考虑表格数据数据增强的新思路

    Rethinking Data Augmentation for Tabular Data in Deep Learning. (arXiv:2305.10308v1 [cs.LG])

    [http://arxiv.org/abs/2305.10308](http://arxiv.org/abs/2305.10308)

    本研究提出了一种新的表格数据增强方法“随机连续嵌入”（Random Continuous Embedding，RCE），能够提高 Transformer-based 预训练模型的自监督学习性能，大幅优于现有方法，并使得自监督学习模型能够在监督表格学习中优于树形方法。

    

    表格数据是机器学习中最广泛使用的数据格式。虽然在有监督学习中，树形方法优于深度学习方法；但最近的文献报告称，Transformer-based 预训练模型的自监督学习优于树形方法。在关于表格数据的自监督学习的现有文献中，对比学习是主导方法。然而，由于表格数据的独特结构和高复杂性，表格数据的数据增强一直是困难的。此外，现有方法将模型结构、自监督学习方法和数据增强三个主要组成部分一起提出。因此，以往的研究在综合考虑这些组成部分的情况下进行对比，每个组成部分对实际性能的影响还不清楚。本研究关注数据增强，以解决这些限制。具体地，我们提出了一种新的数据增强方法“随机连续嵌入”（RCE），通过向连续变量注入噪声来生成增强的表格数据。我们在几个基准数据集上评估了我们的方法，并表明 RCE 在使用 Transformer-based 模型进行自监督学习时一致优于现有的数据增强方法。我们还进行筛选研究以显示 RCE 的有效性，并证明 RCE 使 Transformer-based 模型的自监督学习可在监督表格学习中优于树形方法。

    Tabular data is the most widely used data format in machine learning (ML). While tree-based methods outperform DL-based methods in supervised learning, recent literature reports that self-supervised learning with Transformer-based models outperforms tree-based methods. In the existing literature on self-supervised learning for tabular data, contrastive learning is the predominant method. In contrastive learning, data augmentation is important to generate different views. However, data augmentation for tabular data has been difficult due to the unique structure and high complexity of tabular data. In addition, three main components are proposed together in existing methods: model structure, self-supervised learning methods, and data augmentation. Therefore, previous works have compared the performance without comprehensively considering these components, and it is not clear how each component affects the actual performance.  In this study, we focus on data augmentation to address these 
    
[^73]: 行政人员的发声笑声与社会认可：机器学习研究的探究。 (arXiv:2305.09485v1 [经济学.GN])

    Executive Voiced Laughter and Social Approval: An Explorative Machine Learning Study. (arXiv:2305.09485v1 [econ.GN])

    [http://arxiv.org/abs/2305.09485](http://arxiv.org/abs/2305.09485)

    本文探究了行政沟通中的发声笑声对于社会认可的积极影响，特别是当发生双向笑声时。结果表明，这种影响随着组织业绩的下降而增加。

    

    我们研究了行政人员沟通中的发声笑声以及它对社会认可的影响。我们将笑声，情感作为信息和信息媒介对公司的社会评价的研究相结合，假设行政人员沟通中的发声笑声对社会认可有积极影响，社会认可是指受众对一个组织的亲和力的感知。我们认为，与众笑的效果尤其强，即在给定的沟通场合中，聚焦的行政人员和观众同时发笑的次数。最后，结合情感作为信息和人类认知的负面偏见，我们假设笑声对社会认可的积极影响随着组织业绩的下降而增加。我们在902个德国巴林德斯利加足球新闻发布会和媒体十大数据中进行测试，应用最先进的机器学习方法进行笑声检测，并找到了部分支持我们想法的结果。

    We study voiced laughter in executive communication and its effect on social approval. Integrating research on laughter, affect-as-information, and infomediaries' social evaluations of firms, we hypothesize that voiced laughter in executive communication positively affects social approval, defined as audience perceptions of affinity towards an organization. We surmise that the effect of laughter is especially strong for joint laughter, i.e., the number of instances in a given communication venue for which the focal executive and the audience laugh simultaneously. Finally, combining the notions of affect-as-information and negativity bias in human cognition, we hypothesize that the positive effect of laughter on social approval increases with bad organizational performance. We find partial support for our ideas when testing them on panel data comprising 902 German Bundesliga soccer press conferences and media tenor, applying state-of-the-art machine learning approaches for laughter dete
    
[^74]: 交叉门控多层感知机下的蛋白质复合物不变嵌入是一种一次性抗体设计器

    Protein Complex Invariant Embedding with Cross-Gate MLP is A One-Shot Antibody Designer. (arXiv:2305.09480v1 [q-bio.BM])

    [http://arxiv.org/abs/2305.09480](http://arxiv.org/abs/2305.09480)

    本文提出了一种深度生成模型，可以一次性地共同设计抗体CDR的1D序列和3D结构，解决几何建模和低效推断的问题。

    

    抗体是由免疫系统产生的针对外来物质或抗原的重要蛋白质。抗体的特异性由其互补决定区（CDR）决定，CDR位于抗体链的可变区域中，形成与抗原结合的位点。以往的研究利用复杂的技术生成CDR，但它们遭受了几何建模不足的问题。此外，常见的迭代精化策略导致了低效的推断。本文提出了一种深度生成模型，可以一次性地共同设计CDR的1D序列和3D结构。为了实现这一目标，我们将抗体CDR设计分为两个阶段：（i）蛋白质结构的几何建模和（ii）序列结构共学习。我们开发了一种蛋白质复合物不变嵌入，可捕捉蛋白质骨架原子（包括Cα、N、C和O原子）之间的内部和外部组分相互作用，以实现全面的几何建模。

    Antibodies are crucial proteins produced by the immune system in response to foreign substances or antigens. The specificity of an antibody is determined by its complementarity-determining regions (CDRs), which are located in the variable domains of the antibody chains and form the antigen-binding site. Previous studies have utilized complex techniques to generate CDRs, but they suffer from inadequate geometric modeling. Moreover, the common iterative refinement strategies lead to an inefficient inference. In this paper, we propose a deep generative model that can co-design 1D sequences and 3D structures of CDRs in a one-shot manner. To achieve this, we decouple the antibody CDR design into two stages: (i) geometric modeling of protein structures and (ii) sequence-structure co-learning. We develop a protein complex invariant embedding that captures both intra- and inter-component interactions among the backbone atoms including C$\alpha$, N, C, and O atoms to achieve comprehensive geome
    
[^75]: 见证就是信仰：脑启发模块化训练促进机理诠释

    Seeing is Believing: Brain-Inspired Modular Training for Mechanistic Interpretability. (arXiv:2305.08746v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2305.08746](http://arxiv.org/abs/2305.08746)

    BIMT方法使得神经网络更加模块化和可诠释，并且能够直接展示模块化结构，为许多简单任务提供了有用的信息，并可以补充当前的机理解释策略。

    

    我们提出了一种名为脑启发模块化训练（Brain-Inspired Modular Training, BIMT）的方法，旨在使神经网络更加模块化和可诠释。BIMT从大脑受启发，将神经元嵌入到几何空间中，并通过成本与神经元连接长度成正比的方式增强损失函数。我们证明了BIMT可以为许多简单任务发现有用的模块化神经网络，揭示了符号公式中的组合结构、可解释的决策边界和分类特征，以及算法数据集中的数学结构。直接眼睛看到模块的能力可以补充当前的机理解释策略，例如探针，干预或凝视所有权重。

    We introduce Brain-Inspired Modular Training (BIMT), a method for making neural networks more modular and interpretable. Inspired by brains, BIMT embeds neurons in a geometric space and augments the loss function with a cost proportional to the length of each neuron connection. We demonstrate that BIMT discovers useful modular neural networks for many simple tasks, revealing compositional structures in symbolic formulas, interpretable decision boundaries and features for classification, and mathematical structure in algorithmic datasets. The ability to directly see modules with the naked eye can complement current mechanistic interpretability strategies such as probes, interventions or staring at all weights.
    
[^76]: 平衡出租车时空数据的隐私和效用用于需求预测

    Balancing Privacy and Utility of Spatio-Temporal Data for Taxi-Demand Prediction. (arXiv:2305.08107v1 [cs.LG])

    [http://arxiv.org/abs/2305.08107](http://arxiv.org/abs/2305.08107)

    本文提出了使用联合学习进行出租车需求预测的方法，该方法允许多个参与方训练机器学习模型并保持数据私密和安全。文章对于类别不平衡、数据稀缺和模型泛化等技术挑战提出了解决方案，最终在实际数据集上展示了具有竞争力的表现。

    

    出租车需求预测是机器学习的重要应用，它可以使出租车提供设施优化其运营，城市规划者改善交通基础设施和服务。然而，这些系统中使用敏感数据引发了隐私和安全方面的担忧。在本文中，我们提出了使用联合学习进行出租车需求预测，允许多个参与方在保持数据私密和安全的同时训练机器学习模型。这可以使组织在不得以获取数据的情况下构建模型。尽管联合学习为出租车需求预测带来了潜在的益处，但它也面临着一些技术挑战，例如类别不平衡，某些参与方的数据稀缺以及需要确保模型泛化以适应不同的设施和地理区域。为了有效地应对这些挑战，我们提出了一个系统，利用区域无关的编码进行地理特征处理，自适应正则化来平衡数据稀缺，以及数据增强来解决类别不平衡问题。我们在真实的时空出租车需求数据集上评估了我们的方法，并展示了它在保护隐私和保持数据效用方面达到了有竞争力的表现。

    Taxi-demand prediction is an important application of machine learning that enables taxi-providing facilities to optimize their operations and city planners to improve transportation infrastructure and services. However, the use of sensitive data in these systems raises concerns about privacy and security. In this paper, we propose the use of federated learning for taxi-demand prediction that allows multiple parties to train a machine learning model on their own data while keeping the data private and secure. This can enable organizations to build models on data they otherwise would not be able to access. Despite its potential benefits, federated learning for taxi-demand prediction poses several technical challenges, such as class imbalance, data scarcity among some parties, and the need to ensure model generalization to accommodate diverse facilities and geographic regions. To effectively address these challenges, we propose a system that utilizes region-independent encoding for geogr
    
[^77]: CodeT5+: 用于代码理解和生成的开放代码大型语言模型

    CodeT5+: Open Code Large Language Models for Code Understanding and Generation. (arXiv:2305.07922v1 [cs.CL])

    [http://arxiv.org/abs/2305.07922](http://arxiv.org/abs/2305.07922)

    CodeT5+是一组灵活组合的编码器-解码器LLM族，用于代码，混合了多种不同的预训练目标，包括代码生成、自然语言处理和程序合成，可以适应多种不同的下游代码任务，并且在实验中比现有代码-specific LLMs实现了最先进的性能。

    

    预训练在大量源代码上的大型语言模型(LLMs)在代码智能方面取得了显著进展。然而，现有的代码LLM在架构和预训练任务方面有两个主要限制。首先，它们通常采用特定的架构(仅编码器或仅解码器)或依赖于不同下游任务的统一编码器-解码器网络。前一种范式受到应用灵活性的限制，而在后一种范式中，模型被视为所有任务的单一系统，导致在某些任务的子集上性能不优。其次，它们通常采用有限的预训练目标，这些目标可能与某些下游任务不相关，因此会导致性能显著下降。为了解决这些限制，我们提出了“CodeT5+”，这是一组编码器-解码器LLM族，用于代码，其中组件模块可以灵活组合以适应各种下游代码任务。这种灵活性是通过我们提出的混合预训练目标实现的，包括代码生成，自然语言处理和程序合成。我们在几个与代码相关的下游任务上进行了广泛实验，证明CodeT5+相对于现有的代码特定LLM实现了最先进的性能。

    Large language models (LLMs) pretrained on vast source code have achieved prominent progress in code intelligence. However, existing code LLMs have two main limitations in terms of architecture and pretraining tasks. First, they often adopt a specific architecture (encoder-only or decoder-only) or rely on a unified encoder-decoder network for different downstream tasks. The former paradigm is limited by inflexibility in applications while in the latter, the model is treated as a single system for all tasks, leading to suboptimal performance on a subset of tasks. Secondly, they often employ a limited set of pretraining objectives which might not be relevant to some downstream tasks and hence result in substantial performance degrade. To address these limitations, we propose ``CodeT5+'', a family of encoder-decoder LLMs for code in which component modules can be flexibly combined to suit a wide range of downstream code tasks. Such flexibility is enabled by our proposed mixture of pretrai
    
[^78]: 理解异构数据联邦学习中的模型平均

    Understanding Model Averaging in Federated Learning on Heterogeneous Data. (arXiv:2305.07845v1 [cs.LG])

    [http://arxiv.org/abs/2305.07845](http://arxiv.org/abs/2305.07845)

    本文研究了异构数据联邦学习中的模型平均技术，通过可视化损失/错误景观揭示了客户端模型环绕全局模型在一个共同的盆地内，并且发现全局模型在早期训练后的误差主要来自客户端数据集和全局数据集之间非重叠的数据及全局模型与客户端模型之间的最大距离两个因素。

    

    模型平均是联邦学习中广泛采用的一种技术，它会聚集训练于异构数据上的多个客户端模型以获得表现良好的全局模型。然而，其成功背后的原理尚不是很清楚。本文通过可视化损失/错误景观来研究模型平均的几何特性，揭示了客户端模型环绕全局模型在一个共同的盆地内，并且即使全局模型表现优异，也可能偏离盆地底部。进一步的分析表明，全局模型在早期训练后的误差主要来自客户端数据集和全局数据集之间非重叠的数据及全局模型与客户端模型之间的最大距离两个因素。

    Model averaging, a widely adopted technique in federated learning (FL), aggregates multiple client models trained on heterogeneous data to obtain a well-performed global model. However, the rationale behind its success is not well understood. To shed light on this issue, we investigate the geometric properties of model averaging by visualizing the loss/error landscape. The geometrical visualization shows that the client models surround the global model within a common basin, and the global model may deviate from the bottom of the basin even though it performs better than the client models. To further understand this phenomenon, we decompose the expected prediction error of the global model into five factors related to client models. Specifically, we find that the global-model error after early training mainly comes from i) the client-model error on non-overlapping data between client datasets and the global dataset and ii) the maximal distance between the global and client models. Insp
    
[^79]: 分离随机逼近框架下的在线学习算法

    Online Learning Under A Separable Stochastic Approximation Framework. (arXiv:2305.07484v1 [cs.LG])

    [http://arxiv.org/abs/2305.07484](http://arxiv.org/abs/2305.07484)

    本篇论文提出了一种新的在线学习算法，通过分离随机逼近框架，使用递归最小二乘算法和随机梯度下降算法分别更新模型的线性和非线性参数。此算法在多个数据集上表现出高效和有效性。

    

    本文提出了一个基于分离随机逼近框架的在线学习算法，适用于一类机器学习模型。我们的想法的重点在于观察模型中某些参数比其他参数更容易优化。本文重点关注一类线性参数较多的机器学习模型。我们的算法使用递归最小二乘（RLS）算法来更新线性参数，然后基于更新后的线性参数，使用随机梯度下降（SGD）算法来更新非线性参数。这个算法可以理解为块坐标梯度下降方法的随机逼近版本，在这个版本中，其中一部分参数使用二阶随机梯度下降方法更新，而另一部分参数使用一阶随机梯度下降更新。虽然该算法对于非凸问题的全局收敛性没有讨论，但在几个数据集上的实证结果证明了其高效和有效性。

    We propose an online learning algorithm for a class of machine learning models under a separable stochastic approximation framework. The essence of our idea lies in the observation that certain parameters in the models are easier to optimize than others. In this paper, we focus on models where some parameters have a linear nature, which is common in machine learning. In one routine of the proposed algorithm, the linear parameters are updated by the recursive least squares (RLS) algorithm, which is equivalent to a stochastic Newton method; then, based on the updated linear parameters, the nonlinear parameters are updated by the stochastic gradient method (SGD). The proposed algorithm can be understood as a stochastic approximation version of block coordinate gradient descent approach in which one part of the parameters is updated by a second-order SGD method while the other part is updated by a first-order SGD. Global convergence of the proposed online algorithm for non-convex cases is 
    
[^80]: MEGABYTE: 基于多尺度Transformer的百万字节序列预测

    MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers. (arXiv:2305.07185v1 [cs.LG])

    [http://arxiv.org/abs/2305.07185](http://arxiv.org/abs/2305.07185)

    MEGABYTE是一种基于多尺度Transformer的解码器架构，能够对超过一百万字节的序列进行端到端的可微建模，在训练和生成过程中提高了性能并降低了成本，同时证明了在大规模上下文无需标记的自回归序列建模的可行性。

    

    自回归transformer模型在短序列上表现良好，但对于高分辨率图像、播客、代码或图书等长序列的处理能力较差。我们提出了Megabyte，一种多尺度解码器架构，能够对超过一百万字节的序列进行端到端的可微建模。Megabyte将序列分为图块，并在图块内使用局部子模型，在图块之间使用全局模型。这使得子二次自注意、更大的前馈层和更好的解码并行性得以实现，提高了训练和生成过程的性能，同时降低了成本。广泛的实验表明，Megabyte可以使基于字节的模型在长上下文语言建模方面与基于子词的模型相媲美，在ImageNet上实现了最先进的密度估计，可以模拟来自原始文件的音频。这些结果证明了在大规模上下文无需标记的自回归序列建模的可行性。

    Autoregressive transformers are spectacular models for short sequences but scale poorly to long sequences such as high-resolution images, podcasts, code, or books. We proposed Megabyte, a multi-scale decoder architecture that enables end-to-end differentiable modeling of sequences of over one million bytes. Megabyte segments sequences into patches and uses a local submodel within patches and a global model between patches. This enables sub-quadratic self-attention, much larger feedforward layers for the same compute, and improved parallelism during decoding -- unlocking better performance at reduced cost for both training and generation. Extensive experiments show that Megabyte allows byte-level models to perform competitively with subword models on long context language modeling, achieve state-of-the-art density estimation on ImageNet, and model audio from raw files. Together, these results establish the viability of tokenization-free autoregressive sequence modeling at scale.
    
[^81]: 深度视觉和遗传生物测定用于少量图像数据珍稀物种分类

    Deep Visual-Genetic Biometrics for Taxonomic Classification of Rare Species. (arXiv:2305.06695v1 [cs.CV])

    [http://arxiv.org/abs/2305.06695](http://arxiv.org/abs/2305.06695)

    本文提出了一种利用对齐的视觉-遗传推理空间来提高少量图像数据珍稀物种分类的方法，该方法通过深度嵌入模型实现对齐，适用于提高稀有物种的长尾识别，并且可以显著有益于仅基于视觉的稀有物种识别。

    

    在生物应用中，视觉和遗传生物测定通常用于识别物种和个体。然而，在计算上增强少量图像数据稀有类别的视觉分类方面，该领域尚未进行尝试。因此，本文提出了对齐的视觉-遗传推理空间，旨在隐式编码跨域关联以提高性能。我们首次证明了这种对齐可以通过深度嵌入模型实现，并且该方法直接适用于提高稀有物种的长尾识别（LTR）。我们通过应用于32个物种、超过30,000个浮游有孔虫壳的显微图像并与独立的遗传数据样本一起使用来实验室展现了该概念的效力。最重要的是，对从业者而言，我们展示了视觉-遗传对齐可以显著有益于仅基于视觉的稀有物种识别。

    Visual as well as genetic biometrics are routinely employed to identify species and individuals in biological applications. However, no attempts have been made in this domain to computationally enhance visual classification of rare classes with little image data via genetics. In this paper, we thus propose aligned visual-genetic inference spaces with the aim to implicitly encode cross-domain associations for improved performance. We demonstrate for the first time that such alignment can be achieved via deep embedding models and that the approach is directly applicable to boosting long-tailed recognition (LTR) particularly for rare species. We experimentally demonstrate the efficacy of the concept via application to microscopic imagery of 30k+ planktic foraminifer shells across 32 species when used together with independent genetic data samples. Most importantly for practitioners, we show that visual-genetic alignment can significantly benefit visual-only recognition of the rarest speci
    
[^82]: ST-GIN:一种具有时空图注意力和双向循环联合神经网络的交通数据插值不确定性量化方法

    ST-GIN: An Uncertainty Quantification Approach in Traffic Data Imputation with Spatio-temporal Graph Attention and Bidirectional Recurrent United Neural Networks. (arXiv:2305.06480v1 [cs.LG])

    [http://arxiv.org/abs/2305.06480](http://arxiv.org/abs/2305.06480)

    本研究提出了一种创新的交通数据插值方法，利用图注意力和双向神经网络捕捉时空相关性，实验结果表明在处理缺失值方面优于其他基准技术。

    

    交通数据是智能交通系统中的基本组成部分。然而，从环路检测器或类似来源收集的现实世界交通数据通常包含缺失值(MVs)，这可能会对相关应用和研究产生不利影响。为了恢复这些缺失值，研究人员利用数字统计、张量分解和深度学习技术等方法实现了数据插值。本文提出了一种创新的深度学习方法用于插值缺失数据。该方法采用图注意架构来捕捉交通数据中存在的空间相关性，同时利用双向神经网络学习时间信息。实验结果表明，我们提出的方法优于所有其他基准技术，证明其有效性。

    Traffic data serves as a fundamental component in both research and applications within intelligent transportation systems. However, real-world transportation data, collected from loop detectors or similar sources, often contain missing values (MVs), which can adversely impact associated applications and research. Instead of discarding this incomplete data, researchers have sought to recover these missing values through numerical statistics, tensor decomposition, and deep learning techniques. In this paper, we propose an innovative deep-learning approach for imputing missing data. A graph attention architecture is employed to capture the spatial correlations present in traffic data, while a bidirectional neural network is utilized to learn temporal information. Experimental results indicate that our proposed method outperforms all other benchmark techniques, thus demonstrating its effectiveness.
    
[^83]: 基于空中计算的半异步联邦边缘学习机制

    Semi-Asynchronous Federated Edge Learning Mechanism via Over-the-air Computation. (arXiv:2305.04066v1 [cs.LG])

    [http://arxiv.org/abs/2305.04066](http://arxiv.org/abs/2305.04066)

    本文提出了一种半异步聚合FEEL机制PAOTA，以改善数据和设备存在显著异质性的情况下FEEL系统的训练效率，通过调整边缘设备的上行传输功率来最小化FEEL全局模型的收敛上界。实验结果表明，所提出的机制在达到相同的目标精度下，训练速度显著快于具有空中计算方案的传统同步FEEL机制。

    

    空中计算是提高联邦边缘学习（FEEL）效率的有效传输方案。然而，现有的具有空中计算方案的FEEL系统通常在每个全局轮次中采用传统的同步聚合机制，而这些机制容易受到滞后者的影响。本文提出了一种基于空中计算方案的半异步聚合FEEL机制（PAOTA），以改善数据和设备存在显著异质性的情况下FEEL系统的训练效率。考虑到来自边缘设备模型更新的陈旧性和发散性，我们通过在每个聚合期调整边缘设备的上行传输功率来最小化FEEL全局模型的收敛上界。模拟结果表明，我们提出的算法实现了接近理想的局部SGD的收敛性能。此外，在相同的目标准确度下，所提出的机制的训练速度显着快于具有空中计算方案的传统同步FEEL机制。

    Over-the-air Computation (AirComp) has been demonstrated as an effective transmission scheme to boost the efficiency of federated edge learning (FEEL). However, existing FEEL systems with AirComp scheme often employ traditional synchronous aggregation mechanisms for local model aggregation in each global round, which suffer from the stragglers issues. In this paper, we propose a semi-asynchronous aggregation FEEL mechanism with AirComp scheme (PAOTA) to improve the training efficiency of the FEEL system in the case of significant heterogeneity in data and devices. Taking the staleness and divergence of model updates from edge devices into consideration, we minimize the convergence upper bound of the FEEL global model by adjusting the uplink transmit power of edge devices at each aggregation period. The simulation results demonstrate that our proposed algorithm achieves convergence performance close to that of the ideal Local SGD. Furthermore, with the same target accuracy, the training
    
[^84]: GPT用于半自动化数据科学：引入CAAFE实现上下文感知自动特征工程

    GPT for Semi-Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering. (arXiv:2305.03403v1 [cs.AI])

    [http://arxiv.org/abs/2305.03403](http://arxiv.org/abs/2305.03403)

    介绍了一种名为CAAFE的上下文感知自动特征工程方法，它利用大型语言模型根据数据集描述生成更多具有语义意义的特征，能够提高大多数数据集的性能，平均ROC AUC表现提高至0.822。

    

    随着自动化机器学习（AutoML）领域的发展，将领域知识纳入这些系统中变得越来越重要。我们利用大型语言模型（LLMs）的强大功能提出了一种方法来实现这一目标。具体地，我们介绍了一种用于表格数据的特征工程方法，名为上下文感知自动特征工程（CAAFE），它利用LLM根据数据集的描述生成更多具有语义意义的特征。该方法产生用于创建新特征的Python代码，并提供生成特征的效用说明。尽管方法论上很简单，但CAAFE提高了14个数据集中11个数据集的性能，与2个数据集并列，只有1个数据集性能下降，从而使所有数据集的平均ROC AUC表现从0.798提升至0.822。对于所评估的数据集，这一改进与使用随机森林（AUC 0.782）代替逻辑回归（AUC 0.754）所获得的平均改进相似。此外，

    As the field of automated machine learning (AutoML) advances, it becomes increasingly important to include domain knowledge within these systems. We present an approach for doing so by harnessing the power of large language models (LLMs). Specifically, we introduce Context-Aware Automated Feature Engineering (CAAFE), a feature engineering method for tabular datasets that utilizes an LLM to generate additional semantically meaningful features for tabular datasets based on their descriptions. The method produces both Python code for creating new features and explanations for the utility of the generated features.  Despite being methodologically simple, CAAFE enhances performance on 11 out of 14 datasets, ties on 2 and looses on 1 - boosting mean ROC AUC performance from 0.798 to 0.822 across all datasets. On the evaluated datasets, this improvement is similar to the average improvement achieved by using a random forest (AUC 0.782) instead of logistic regression (AUC 0.754).  Furthermore,
    
[^85]: 机器学习模型在电子健康记录中的敏感数据检测

    Sensitive Data Detection with High-Throughput Machine Learning Models in Electrical Health Records. (arXiv:2305.03169v1 [cs.CR])

    [http://arxiv.org/abs/2305.03169](http://arxiv.org/abs/2305.03169)

    该论文使用机器学习算法来识别结构化数据中的敏感变量，以便便于去身份化过程。该算法可以解决不同数据集PHI字段异质性的问题。

    

    在大数据时代，医疗保健提供者、社区和研究人员需要分享数据并合作改善健康结果、获取有价值的见解和推进研究。1996年《健康保险流通与责任法案》(HIPAA)是一项联邦法律，旨在通过制定保护健康信息的规定来保护敏感健康信息。然而，在数据共享之前，HIPAA没有提供有效的检测或删除PHI的工具。本文旨在探讨使用机器学习算法来识别结构化数据中的敏感变量，从而便于去身份化过程。

    In the era of big data, there is an increasing need for healthcare providers, communities, and researchers to share data and collaborate to improve health outcomes, generate valuable insights, and advance research. The Health Insurance Portability and Accountability Act of 1996 (HIPAA) is a federal law designed to protect sensitive health information by defining regulations for protected health information (PHI). However, it does not provide efficient tools for detecting or removing PHI before data sharing. One of the challenges in this area of research is the heterogeneous nature of PHI fields in data across different parties. This variability makes rule-based sensitive variable identification systems that work on one database fail on another. To address this issue, our paper explores the use of machine learning algorithms to identify sensitive variables in structured data, thus facilitating the de-identification process. We made a key observation that the distributions of metadata of
    
[^86]: 面向半监督多标签学习的类别分布感知伪标记方法

    Class-Distribution-Aware Pseudo Labeling for Semi-Supervised Multi-Label Learning. (arXiv:2305.02795v1 [cs.LG])

    [http://arxiv.org/abs/2305.02795](http://arxiv.org/abs/2305.02795)

    本论文提出了一种面向半监督多标签学习的类别分布感知伪标记方法，能够在控制伪标签数目的情况下，更准确地逼近真实分布，从而实现更好的多标签分类性能。

    

    伪标记是一种利用未标记数据信息的流行且有效方法。然而，传统的基于实例的伪标记方法通常根据其预测概率为每个未标记实例分配一个伪标签。由于真实标签数目未知，这些方法在半监督多标签学习（SSMLL）场景下难以很好地推广，因为它们会面临引入假正标签或忽略真正标签的风险。本文提出了一种面向SSMLL问题的类别分布感知伪标记（CAP）方法，鼓励伪标签的类别分布接近真实分布。具体而言，我们设计了一个包括类别感知阈值的正则化学习框架来控制每个类别的伪标签数目。鉴于标记和未标记的示例是根据同一分布采样的，我们通过利用标记示例的类别分布确定阈值并随着模型参数一起更新。在几个基准数据集上的实验结果表明，我们提出的方法在多标签分类性能方面显著优于现有的最先进方法。

    Pseudo labeling is a popular and effective method to leverage the information of unlabeled data. Conventional instance-aware pseudo labeling methods often assign each unlabeled instance with a pseudo label based on its predicted probabilities. However, due to the unknown number of true labels, these methods cannot generalize well to semi-supervised multi-label learning (SSMLL) scenarios, since they would suffer from the risk of either introducing false positive labels or neglecting true positive ones. In this paper, we propose to solve the SSMLL problems by performing Class-distribution-Aware Pseudo labeling (CAP), which encourages the class distribution of pseudo labels to approximate the true one. Specifically, we design a regularized learning framework consisting of the class-aware thresholds to control the number of pseudo labels for each class. Given that the labeled and unlabeled examples are sampled according to the same distribution, we determine the thresholds by exploiting th
    
[^87]: 利用语言表示进行材料推荐、排名和探索

    Leveraging Language Representation for Material Recommendation, Ranking, and Exploration. (arXiv:2305.01101v1 [cond-mat.mtrl-sci])

    [http://arxiv.org/abs/2305.01101](http://arxiv.org/abs/2305.01101)

    本文提出了一种新型材料发现框架，利用材料科学特定语言模型的自然语言嵌入作为材料的组成和结构特征进行表示，并且联合采用了表示相似性召回候选材料和基于多任务学习对候选材料进行目标属性排名的方案。通过这种方法，可以更好地探索广阔的材料搜索空间，并确定高性能候选材料。

    

    利用机器学习的新兴技术，数据驱动的材料发现和设计已经得到了加速。虽然在学习材料结构与性质关系方面取得了巨大进展，但能够有效探索广阔的材料搜索空间并确定高性能候选材料的方法仍然十分有限。本文介绍了一种材料发现框架，它利用从材料科学特定语言模型中得出的自然语言嵌入作为组成和结构特征的表示。该发现框架由一个联合方案组成，给定一个查询材料，首先基于表示相似性召回候选材料，再通过多任务学习对候选材料进行目标属性排名。语言表示中编码的上下文知识被发现可以传达有关材料性质和结构的信息，使得相似性分析和高性能材料发现变得更加可行。

    Data-driven approaches for material discovery and design have been accelerated by emerging efforts in machine learning. While there is enormous progress towards learning the structure to property relationship of materials, methods that allow for general representations of crystals to effectively explore the vast material search space and identify high-performance candidates remain limited. In this work, we introduce a material discovery framework that uses natural language embeddings derived from material science-specific language models as representations of compositional and structural features. The discovery framework consists of a joint scheme that, given a query material, first recalls candidates based on representational similarity, and ranks the candidates based on target properties through multi-task learning. The contextual knowledge encoded in language representations is found to convey information about material properties and structures, enabling both similarity analysis fo
    
[^88]: 跨图动态迁移学习

    Dynamic Transfer Learning across Graphs. (arXiv:2305.00664v1 [cs.LG])

    [http://arxiv.org/abs/2305.00664](http://arxiv.org/abs/2305.00664)

    该论文提出了一个新的问题：在动态图形环境下如何有效地进行跨图迁移学习，主要解决了领域演化对泛化性能的影响。

    

    在许多高风险领域中，跨图传输知识起着关键作用，包括运输网络、电子商务网络、神经科学和金融领域。我们提出了一个新问题：在动态设置下，考虑已观察到的具有标签的源图和标签稀疏的目标图，如何有效地表征不断变化的领域偏差，并优化目标域在下一个时间戳的泛化性能？为了回答这个问题，我们首次提出了跨图动态迁移学习设置下的一般化界限，这意味着泛化性能由领域演化控制。

    Transferring knowledge across graphs plays a pivotal role in many high-stake domains, ranging from transportation networks to e-commerce networks, from neuroscience to finance. To date, the vast majority of existing works assume both source and target domains are sampled from a universal and stationary distribution. However, many real-world systems are intrinsically dynamic, where the underlying domains are evolving over time. To bridge the gap, we propose to shift the problem to the dynamic setting and ask: given the label-rich source graphs and the label-scarce target graphs observed in previous T timestamps, how can we effectively characterize the evolving domain discrepancy and optimize the generalization performance of the target domain at the incoming T+1 timestamp? To answer the question, for the first time, we propose a generalization bound under the setting of dynamic transfer learning across graphs, which implies the generalization performance is dominated by domain evolution
    
[^89]: 大型语言模型所表现的新兴技能是否为幻觉？

    Are Emergent Abilities of Large Language Models a Mirage?. (arXiv:2304.15004v1 [cs.AI])

    [http://arxiv.org/abs/2304.15004](http://arxiv.org/abs/2304.15004)

    研究指出大型语言模型所谓的新兴技能是研究者分析的产物，不是模型行为的基本变化。研究还展示了度量标准选择和可能研究人员的偏见，可能导致这种新兴技能的出现。

    

    最近的研究声称，大型语言模型展示了新兴技能，这些技能在更小规模的模型中不存在，但在更大规模的模型中存在。新兴技能让人感到困惑的是两方面：它们的清晰度，似乎瞬间从不存在到存在，以及它们的不可预测性，似乎在不可预见的模型规模下出现。本文提出了新兴技能的另一种解释，即对于特定任务和模型族，当分析固定的模型输出时，可以选择导致推断出新兴技能或不导致推断出新兴技能的度量标准。因此，我们的解释表明，现有的新兴技能声明是研究人员分析的产物，而不是特定任务中模型行为的基本变化。我们在一个简单的数学模型中提出了我们的解释，然后通过三种互补的方式进行了测试：我们(1)制作、测试并验证了关于报告的新兴技能的度量选择的三个预测效应；(2)展示了模型架构和训练程序的简单变化会在一个已经确定的任务中产生大的新兴能力差异；(3)展示所谓的新兴技能可以通过有意优化所选择的评估指标来实现。总的来说，我们认为目前大型语言模型中新兴能力的声明很可能并不是真实存在的，而是度量标准任意选择和可能的研究人员偏见的产物。

    Recent work claims that large language models display emergent abilities, abilities not present in smaller-scale models that are present in larger-scale models. What makes emergent abilities intriguing is two-fold: their sharpness, transitioning seemingly instantaneously from not present to present, and their unpredictability, appearing at seemingly unforeseeable model scales. Here, we present an alternative explanation for emergent abilities: that for a particular task and model family, when analyzing fixed model outputs, one can choose a metric which leads to the inference of an emergent ability or another metric which does not. Thus, our alternative suggests that existing claims of emergent abilities are creations of the researcher's analyses, not fundamental changes in model behavior on specific tasks with scale. We present our explanation in a simple mathematical model, then test it in three complementary ways: we (1) make, test and confirm three predictions on the effect of metri
    
[^90]: 功能扩散映射

    Functional Diffusion Maps. (arXiv:2304.14378v1 [cs.LG])

    [http://arxiv.org/abs/2304.14378](http://arxiv.org/abs/2304.14378)

    本研究关注一种非线性流形学习方法：扩散映射。本文阐述如何将这种方法应用于功能数据，并将其与功能主成分分析进行比较。

    

    如今，许多现实世界的数据集可以被视为是功能性的，也就是说生成它们的过程是连续的。这种类型数据的一个基本特性是，理论上它们属于无限维空间。尽管在实践中，我们通常只能得到有限数量的观察结果，它们仍然是高维的，因此降维方法至关重要。在这方面，功能数据分析的主要现有方法是功能主成分分析。尽管如此，这种经典技术假设数据位于一个线性流形中，因此当这个假设不成立时可能会出现问题。本研究聚焦于一种非线性流形学习方法：扩散映射。本文解释了如何将这种多变量方法扩展到功能数据，并将其行为与功能主成分分析在不同的模拟和实际例子中进行了比较。

    Nowadays many real-world datasets can be considered as functional, in the sense that the processes which generate them are continuous. A fundamental property of this type of data is that in theory they belong to an infinite-dimensional space. Although in practice we usually receive finite observations, they are still high-dimensional and hence dimensionality reduction methods are crucial. In this vein, the main state-of-the-art method for functional data analysis is Functional PCA. Nevertheless, this classic technique assumes that the data lie in a linear manifold, and hence it could have problems when this hypothesis is not fulfilled. In this research, attention has been placed on a non-linear manifold learning method: Diffusion Maps. The article explains how to extend this multivariate method to functional data and compares its behavior against Functional PCA over different simulated and real examples.
    
[^91]: 可解释的神经符号概念推理

    Interpretable Neural-Symbolic Concept Reasoning. (arXiv:2304.14068v1 [cs.AI])

    [http://arxiv.org/abs/2304.14068](http://arxiv.org/abs/2304.14068)

    本文提出了第一个基于概念嵌入的可解释概念模型DCR，能够在多个数据集上实现接近最先进的准确性，相对于最先进的可解释概念模型提高了高达+25％，并产生能够解释其预测的人类可理解规则和真值度，适应性强。

    

    深度学习方法具有高度的准确性，但它们不透明的决策过程阻止了它们获得完全的人类信任。概念模型旨在通过学习一组人类可理解的概念来解决这个问题。然而，最先进的概念模型依赖于高维概念嵌入表示，缺乏明确的语义含义，因此质疑其决策过程的可解释性。为了克服这个限制，我们提出了Deep Concept Reasoner(DCR)，这是第一个基于概念嵌入的可解释概念模型。在DCR中，神经网络不直接进行任务预测，而是使用概念嵌入建立语法规则结构。然后DCR在有意义的概念真值度上执行这些规则，以不可微分的方式提供最终的可解释和语义一致的预测。我们的实验表明，DCR：(i)在多个数据集上实现接近最先进的准确性，同时相对于最先进的可解释概念模型提高了高达+25％;(ii)产生能够解释其预测的人类可理解规则和真值度;(iii)很容易适应新领域。

    Deep learning methods are highly accurate, yet their opaque decision process prevents them from earning full human trust. Concept-based models aim to address this issue by learning tasks based on a set of human-understandable concepts. However, state-of-the-art concept-based models rely on high-dimensional concept embedding representations which lack a clear semantic meaning, thus questioning the interpretability of their decision process. To overcome this limitation, we propose the Deep Concept Reasoner (DCR), the first interpretable concept-based model that builds upon concept embeddings. In DCR, neural networks do not make task predictions directly, but they build syntactic rule structures using concept embeddings. DCR then executes these rules on meaningful concept truth degrees to provide a final interpretable and semantically-consistent prediction in a differentiable manner. Our experiments show that DCR: (i) improves up to +25% w.r.t. state-of-the-art interpretable concept-based
    
[^92]: 量子自然策略梯度：向样本高效增强学习迈进

    Quantum Natural Policy Gradients: Towards Sample-Efficient Reinforcement Learning. (arXiv:2304.13571v1 [quant-ph] CROSS LISTED)

    [http://arxiv.org/abs/2304.13571](http://arxiv.org/abs/2304.13571)

    本文提出了量子自然策略梯度(QNPG)算法，利用了量子费舍尔信息矩阵的高效近似方法，提高了强化学习的效率，实验结果表明，相比基于一阶梯度的训练，QNPG具有更快的收敛速度和稳定性，可以减少样本复杂度。

    

    强化学习是人工智能领域的一个快速发展的方向，但学习过程通常很耗费资源。使用变分量子电路作为函数逼近器可以减少成本，提高强化学习效率。本文中，我们提出了量子自然策略梯度(QNPG)算法，该算法利用了量子费舍尔信息矩阵的高效近似方法，是一种二阶梯度的基于策略的算法。在Contextual Bandits环境下的实验结果表明，QNPG 比基于一阶梯度的训练具有更快的收敛速度和稳定性，从而减少了样本复杂度，进一步展示了我们方法的实际可行性，并在12量子比特的硬件设备上进行了训练。

    Reinforcement learning is a growing field in AI with a lot of potential. Intelligent behavior is learned automatically through trial and error in interaction with the environment. However, this learning process is often costly. Using variational quantum circuits as function approximators can reduce this cost. In order to implement this, we propose the quantum natural policy gradient (QNPG) algorithm -- a second-order gradient-based routine that takes advantage of an efficient approximation of the quantum Fisher information matrix. We experimentally demonstrate that QNPG outperforms first-order based training on Contextual Bandits environments regarding convergence speed and stability and thereby reduces the sample complexity. Furthermore, we provide evidence for the practical feasibility of our approach by training on a 12-qubit hardware device.
    
[^93]: BN与ReLU之间的不和谐引起梯度爆炸，但被激活之间的相关性所抵消。

    The Disharmony Between BN and ReLU Causes Gradient Explosion, but is Offset by the Correlation Between Activations. (arXiv:2304.11692v1 [cs.LG])

    [http://arxiv.org/abs/2304.11692](http://arxiv.org/abs/2304.11692)

    本研究阐述了BN和ReLU之间的不和谐是导致梯度爆炸的主要原因，同时发现输入之间的相关性可以缓解这个问题。提出一种基于二阶优化算法的自适应学习率算法，在大批量训练中表现优异，并可替代WarmUp，在小批量训练中也表现不错。

    

    基于批标准化和ReLU等激活函数的深度神经网络可能会在训练初期由于时间梯度爆炸而出现不稳定。我们解释了ReLU如何比预期更多地减少方差，以及批标准化如何在恢复期间放大梯度，导致前向传播保持稳定而梯度爆炸。此外，我们还讨论了深度神经网络在训练过程中的动力学变化以及输入之间的相关性如何缓解这个问题。最后，我们提出了一种灵感来自二阶优化算法的更好的自适应学习率算法，在大批量训练中优于现有的学习率缩放方法，并可替换小批量训练中的WarmUp。

    Deep neural networks based on batch normalization and ReLU-like activation functions can experience instability during the early stages of training due to the high gradient induced by temporal gradient explosion. We explain how ReLU reduces variance more than expected, and how batch normalization amplifies the gradient during recovery, which causes gradient explosion while forward propagation remains stable. Additionally, we discuss how the dynamics of a deep neural network change during training and how the correlation between inputs can alleviate this problem. Lastly, we propose a better adaptive learning rate algorithm inspired by second-order optimization algorithms, which outperforms existing learning rate scaling methods in large batch training and can also replace WarmUp in small batch training.
    
[^94]: IDQL: 作为一种扩散策略的Actor-Critic方法的隐式Q学习。 (arXiv:2304.10573v1 [cs.LG])

    IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies. (arXiv:2304.10573v1 [cs.LG])

    [http://arxiv.org/abs/2304.10573](http://arxiv.org/abs/2304.10573)

    本文重新解释隐式Q学习(IQL)作为Actor-Critic方法，提出使用扩散行为策略和评判器权重来平衡奖励最大化和与行为策略的分歧。这个方法能够处理复杂和多峰特征的Actor问题。

    

    有效的离线RL方法需要正确处理超出分布的行为。隐式Q学习（IQL）通过仅使用数据集行动通过修改后的Bellman Backup来训练Q函数来解决此问题。但是，不清楚哪个策略实际上实现了此隐含训练的Q函数所代表的值。在本文中，我们将IQL重新解释为Actor-Critic方法，通过广义化评判目标并将其连接到行为规范化的隐式Actor来实现。这种泛化显示了引入的Actor如何平衡奖励最大化和与行为策略的分歧，具体的损失选择决定了这种权衡的性质。值得注意的是，这个Actor可以表现出复杂和多峰的特征，这表明了利用优势加权回归（AWR）中使用的条件高斯Actor的拟合问题。相反，我们建议使用来自参数化扩散行为策略的样本和由评判器计算的权重，然后将其导入。

    Effective offline RL methods require properly handling out-of-distribution actions. Implicit Q-learning (IQL) addresses this by training a Q-function using only dataset actions through a modified Bellman backup. However, it is unclear which policy actually attains the values represented by this implicitly trained Q-function. In this paper, we reinterpret IQL as an actor-critic method by generalizing the critic objective and connecting it to a behavior-regularized implicit actor. This generalization shows how the induced actor balances reward maximization and divergence from the behavior policy, with the specific loss choice determining the nature of this tradeoff. Notably, this actor can exhibit complex and multimodal characteristics, suggesting issues with the conditional Gaussian actor fit with advantage weighted regression (AWR) used in prior methods. Instead, we propose using samples from a diffusion parameterized behavior policy and weights computed from the critic to then importa
    
[^95]: Zip-NeRF：抗锯齿网格化神经辐射场

    Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields. (arXiv:2304.06706v1 [cs.CV])

    [http://arxiv.org/abs/2304.06706](http://arxiv.org/abs/2304.06706)

    本文提出了一种 Zip-NeRF 技术，将 mip-NeRF 360 和基于网格的模型相结合，以实现抗锯齿、提高训练速度并降低误差率。

    

    神经辐射场（NeRF）的网格化表示可以加速训练，但缺乏对比例的明确理解，容易引入锯齿或丢失场景内容。本文提出了一种将渲染和信号处理思想用于将 mip-NeRF 360 和基于网格的模型相结合，误差率比先前的技术低8%到76%，并比 mip-NeRF 360 快22倍的方法。

    Neural Radiance Field training can be accelerated through the use of grid-based representations in NeRF's learned mapping from spatial coordinates to colors and volumetric density. However, these grid-based approaches lack an explicit understanding of scale and therefore often introduce aliasing, usually in the form of jaggies or missing scene content. Anti-aliasing has previously been addressed by mip-NeRF 360, which reasons about sub-volumes along a cone rather than points along a ray, but this approach is not natively compatible with current grid-based techniques. We show how ideas from rendering and signal processing can be used to construct a technique that combines mip-NeRF 360 and grid-based models such as Instant NGP to yield error rates that are 8% - 76% lower than either prior technique, and that trains 22x faster than mip-NeRF 360.
    
[^96]: SAMM（Segment Any Medical Model）：用于SAM的3D Slicer集成

    SAMM (Segment Any Medical Model): A 3D Slicer Integration to SAM. (arXiv:2304.05622v1 [eess.IV])

    [http://arxiv.org/abs/2304.05622](http://arxiv.org/abs/2304.05622)

    介绍了Segment Any Medical Model (SAMM)，它是用于3D Slicer的SAM的扩展。SAMM在医学图像分割上表现良好，在实时性和通用性方面都有很好的性能，可以推断出掩模。

    

    Segment Anything Model（SAM）是一个新的图像分割工具，使用迄今为止最大的分割数据集进行训练。该模型表明它可以创建高质量的图像分割掩模，具有良好的实时性和通用性。然而，在医学图像上的性能需要进一步验证。为了协助在医学图像上开发，评估和利用SAM，我们介绍了Segment Any Medical Model（SAMM），它是SAM在3D Slicer上的扩展。3D Slicer是一个广泛使用于医学影像处理和可视化软件的开源软件。这个开源扩展程序及其演示已发布在GitHub上（https://github.com/bingogome/samm）。SAMM在完整周期中实现了0.6秒的延迟，并可以实时推断出图像掩模。

    The Segment Anything Model (SAM) is a new image segmentation tool trained with the largest segmentation dataset at this time. The model has demonstrated that it can create high-quality masks for image segmentation with good promptability and generalizability. However, the performance of the model on medical images requires further validation. To assist with the development, assessment, and utilization of SAM on medical images, we introduce Segment Any Medical Model (SAMM), an extension of SAM on 3D Slicer, a widely-used open-source image processing and visualization software that has been extensively used in the medical imaging community. This open-source extension to 3D Slicer and its demonstrations are posted on GitHub (https://github.com/bingogome/samm). SAMM achieves 0.6-second latency of a complete cycle and can infer image masks in nearly real-time.
    
[^97]: 为什么要逐步思考？推理源于经验的局部性。

    Why think step-by-step? Reasoning emerges from the locality of experience. (arXiv:2304.03843v1 [cs.AI])

    [http://arxiv.org/abs/2304.03843](http://arxiv.org/abs/2304.03843)

    本文通过语言模型研究何时以及为什么推理是有帮助的，测试推理在训练数据由相互影响强烈的局部变量集群组成时是否有效。通过一步步的推理，能够将准确的局部推理链接在一起，以估算在训练中没有同时观察到的变量之间的关系。

    

    人类有着强大而神秘的推理能力。通过一系列纯粹的思维步骤，我们可以推理出我们无法直接得出的推论 - 尽管我们从世界上没有得到任何额外数据。同样地，大型语言模型可以通过一步步的推理，在回答问题之前生成中间步骤，从而更好地完成复杂的任务。我们使用语言模型研究何时以及为什么推理是有帮助的，测试推理在训练数据由相互影响强烈的局部变量集群组成时是否有效。这些训练条件能够将准确的局部推理链接在一起，以估算在训练中没有同时观察到的变量之间的关系。我们使用贝叶斯网络定义的联合分布的样品对自回归变压器进行训练，但每个样品只包括其中的一部分变量。我们比较使用推理生成的变量子集与使用完整集合进行训练的方案的性能。

    Humans have a powerful and mysterious capacity to reason. By working through a series of purely mental steps, we can make inferences we would not be capable of making directly -- despite that fact that we get no additional data from the world. Similarly, large language models can perform better at complex tasks through chain-of-thought reasoning, where they generate intermediate steps before answering a question. We use language models to investigate the questions of when and why reasoning is helpful, testing the hypothesis that reasoning is effective when training data consisting of local clusters of variables that influence each other strongly. These training conditions enable the chaining of accurate local inferences in order to estimate relationships between variables that were not seen together in training. We train an autoregressive transformer on samples from joint distributions defined by Bayes nets, but only include a subset of all the variables in each sample. We compare lang
    
[^98]: Torch-Choice: 用Python实现大规模选择建模的PyTorch包

    Torch-Choice: A PyTorch Package for Large-Scale Choice Modelling with Python. (arXiv:2304.01906v1 [cs.LG])

    [http://arxiv.org/abs/2304.01906](http://arxiv.org/abs/2304.01906)

    本文介绍了一款名为 Torch-Choice 的 PyTorch 软件包，用于管理数据库、构建多项式Logit和嵌套Logit模型，并支持GPU加速，具有灵活性和高效性。

    

    $\texttt{torch-choice}$ 是一款开源软件包，使用Python和PyTorch实现灵活、快速的选择建模。它提供了 $\texttt{ChoiceDataset}$ 数据结构，以便灵活而高效地管理数据库。本文演示了如何从各种格式的数据库中构建 $\texttt{ChoiceDataset}$，并展示了 $\texttt{ChoiceDataset}$ 的各种功能。该软件包实现了两种常用的模型: 多项式Logit和嵌套Logit模型，并支持模型估计期间的正则化。该软件包还支持使用GPU进行估计，使其可以扩展到大规模数据集而且在计算上更高效。模型可以使用R风格的公式字符串或Python字典进行初始化。最后，我们比较了 $\texttt{torch-choice}$ 和 R中的 $\texttt{mlogit}$ 在以下几个方面的计算效率: (1) 观测数增加时，(2) 协变量个数增加时， (3) 测试数升高时。

    The $\texttt{torch-choice}$ is an open-source library for flexible, fast choice modeling with Python and PyTorch. $\texttt{torch-choice}$ provides a $\texttt{ChoiceDataset}$ data structure to manage databases flexibly and memory-efficiently. The paper demonstrates constructing a $\texttt{ChoiceDataset}$ from databases of various formats and functionalities of $\texttt{ChoiceDataset}$. The package implements two widely used models, namely the multinomial logit and nested logit models, and supports regularization during model estimation. The package incorporates the option to take advantage of GPUs for estimation, allowing it to scale to massive datasets while being computationally efficient. Models can be initialized using either R-style formula strings or Python dictionaries. We conclude with a comparison of the computational efficiencies of $\texttt{torch-choice}$ and $\texttt{mlogit}$ in R as (1) the number of observations increases, (2) the number of covariates increases, and (3) th
    
[^99]: 深度排名集成用于超参数优化

    Deep Ranking Ensembles for Hyperparameter Optimization. (arXiv:2303.15212v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.15212](http://arxiv.org/abs/2303.15212)

    研究提出了一种新型方法，元学习神经网络代理来优化超参数配置的性能排名，并通过集成来建模不确定性，该方法在超参数优化方面取得了最新最先进的结果。

    

    自动优化机器学习算法的超参数是人工智能中的主要问题之一。现有的超参数优化工作训练了代理模型来近似超参数响应面作为回归任务。相反，我们假设训练代理的最佳策略是保持超参数配置的性能排名作为一个学习排名问题。因此，我们提出了一种新颖的方法，元学习神经网络代理来优化配置的性能排名，同时通过集成的方式对其不确定性进行建模。在一个大规模的实验协议中，包含12个基线，16个HPO搜索空间和86个数据集/任务，我们证明了我们的方法在HPO方面取得了新的最先进的结果。

    Automatically optimizing the hyperparameters of Machine Learning algorithms is one of the primary open questions in AI. Existing work in Hyperparameter Optimization (HPO) trains surrogate models for approximating the response surface of hyperparameters as a regression task. In contrast, we hypothesize that the optimal strategy for training surrogates is to preserve the ranks of the performances of hyperparameter configurations as a Learning to Rank problem. As a result, we present a novel method that meta-learns neural network surrogates optimized for ranking the configurations' performances while modeling their uncertainty via ensembling. In a large-scale experimental protocol comprising 12 baselines, 16 HPO search spaces and 86 datasets/tasks, we demonstrate that our method achieves new state-of-the-art results in HPO.
    
[^100]: 多模态变分自编码器用于跨多种成像模态进行规范建模

    Multi-modal Variational Autoencoders for normative modelling across multiple imaging modalities. (arXiv:2303.12706v1 [cs.CV])

    [http://arxiv.org/abs/2303.12706](http://arxiv.org/abs/2303.12706)

    本文提出了一种多模态规范建模框架，能够更好地检测出多种成像和生物变量中的异常性，特别适用于研究带有异质性的疾病。

    

    研究常见神经疾病的挑战之一是疾病异质性，包括病因、神经成像特征、合并症或基因变异的差异。规范建模已成为研究这种人群的流行方法，其中对生理系统的“正常”行为进行建模，并可以用于个体层面上检测与疾病病理相关的偏差。对于许多异质性疾病，我们预计会观察到多种神经成像和生物变量的异常。然而，到目前为止，规范模型主要是为了研究单一成像模态而开发的。我们旨在开发一种多模态规范建模框架，在多个模态的变量中聚合异常性，并且比单模式基线更能检测到偏差。我们提出了两种用于检测T1和DTI数据中的个体层面偏差的多模态VAE规范模型。与单模式基线相比，我们提出的模型能够更好地检测到轻度认知受损的受试者中的偏差，证明了多模态规范建模用于疾病异质性的潜力。

    One of the challenges of studying common neurological disorders is disease heterogeneity including differences in causes, neuroimaging characteristics, comorbidities, or genetic variation. Normative modelling has become a popular method for studying such cohorts where the 'normal' behaviour of a physiological system is modelled and can be used at subject level to detect deviations relating to disease pathology. For many heterogeneous diseases, we expect to observe abnormalities across a range of neuroimaging and biological variables. However, thus far, normative models have largely been developed for studying a single imaging modality. We aim to develop a multi-modal normative modelling framework where abnormality is aggregated across variables of multiple modalities and is better able to detect deviations than uni-modal baselines. We propose two multi-modal VAE normative models to detect subject level deviations across T1 and DTI data. Our proposed models were better able to detect di
    
[^101]: 具有元梯度正则化的自监督元提示学习用于少样本泛化

    Self-supervised Meta-Prompt Learning with Meta-Gradient Regularization for Few-shot Generalization. (arXiv:2303.12314v1 [cs.CL])

    [http://arxiv.org/abs/2303.12314](http://arxiv.org/abs/2303.12314)

    提出了一种自我监督元提示学习框架SUPMER，包括元梯度正则化，用于少样本泛化，通过锚定的元训练任务和基于课程的任务增强丰富了任务分布，解决了在少样本情况下良好初始化软提示和过拟合的问题。

    

    提示调整是一种参数有效的方法，它学习软提示并使冻结的语言模型执行特定的下游任务。尽管有效，但提示调整在少样本情况下一方面严重依赖于良好的软提示初始化。另一方面，它很容易导致过度拟合。现有的方法利用预训练或监督元学习来初始化软提示，但它们不能对未见下游任务进行数据有效的泛化。为了解决以上问题，本文提出了一种新的自我监督元提示学习框架，其中包括元梯度正则化，用于少样本泛化（SUPMER）。我们首先设计了一组自监督锚定的元训练任务，具有不同的任务格式，并通过基于课程的任务增强进一步丰富了任务分布。然后将一种新的元梯度正则化方法集成到元提示学习中。它元学习在少样本情况下如何转换原始梯度。

    Prompt tuning is a parameter-efficient method, which learns soft prompts and conditions frozen language models to perform specific downstream tasks. Though effective, prompt tuning under few-shot settings on the one hand heavily relies on a good initialization of soft prompts. On the other hand, it can easily result in overfitting. Existing works leverage pre-training or supervised meta-learning to initialize soft prompts but they cannot data-efficiently generalize to unseen downstream tasks. To address the above problems, this paper proposes a novel Self-sUpervised meta-Prompt learning framework with meta-gradient Regularization for few-shot generalization (SUPMER). We first design a set of self-supervised anchor meta-training tasks with different task formats and further enrich the task distribution with curriculum-based task augmentation. Then a novel meta-gradient regularization method is integrated into meta-prompt learning. It meta-learns to transform the raw gradients during few
    
[^102]: 带有重尾噪声的随机非光滑凸优化

    Stochastic Nonsmooth Convex Optimization with Heavy-Tailed Noises. (arXiv:2303.12277v1 [math.OC])

    [http://arxiv.org/abs/2303.12277](http://arxiv.org/abs/2303.12277)

    本文分析了具有重尾噪声的随机非光滑凸优化问题，并填补了在函数非光滑场景下的研究空白。

    

    最近，一些研究将随机优化问题考虑在重尾噪声范式下，即假设随机梯度和真实梯度之间的差异具有有限的 $p$ 阶矩（例如被某个 $\sigma \geq0$ 上界限制为 $\sigma^{p}$），其中 $p\in (1,2]$，这不仅泛化了传统的有限方差假设（$p=2$），而且在许多不同的任务中都被观察到。在这个具有挑战性的假设下，针对凸或非凸问题已经取得了很多新进展，然而，大多数只考虑光滑的目标函数。相反，在函数非光滑时，人们尚未充分探索并完全理解这个问题。本文旨在通过对带有重尾噪声的随机非光滑凸优化提供全面分析来填补这一关键空白。我们重新考虑了一个简单的基于裁剪的算法，然而，这个算法只被证明能以期望方式收敛，但在附加

    Recently, several studies consider the stochastic optimization problem but in a heavy-tailed noise regime, i.e., the difference between the stochastic gradient and the true gradient is assumed to have a finite $p$-th moment (say being upper bounded by $\sigma^{p}$ for some $\sigma\geq0$) where $p\in(1,2]$, which not only generalizes the traditional finite variance assumption ($p=2$) but also has been observed in practice for several different tasks. Under this challenging assumption, lots of new progress has been made for either convex or nonconvex problems, however, most of which only consider smooth objectives. In contrast, people have not fully explored and well understood this problem when functions are nonsmooth. This paper aims to fill this crucial gap by providing a comprehensive analysis of stochastic nonsmooth convex optimization with heavy-tailed noises. We revisit a simple clipping-based algorithm, whereas, which is only proved to converge in expectation but under the additi
    
[^103]: STDLens：基于模型挟持的物体检测联邦学习的安全防护方法

    STDLens: Model Hijacking-resilient Federated Learning for Object Detection. (arXiv:2303.11511v1 [cs.CR])

    [http://arxiv.org/abs/2303.11511](http://arxiv.org/abs/2303.11511)

    STDLens 是一种可以防止FL受到模型挟持的攻击的安全方法。它基于三层的取证框架来识别和排除特殊的梯度，并恢复FL的性能。STDLens在物体检测方面实现了最先进的性能并且具有防止模型挟持的鲁棒性。

    

    联邦学习（FL）作为协同学习框架在分布式客户端中训练基于深度学习的物体检测模型已经越来越受欢迎。尽管它具有诸多优点，FL容易受到模型挟持的攻击。攻击者可以仅仅利用一小部分可以被攻击的客户端控制物体检测系统的正确性，通过植入特殊梯度实现攻击。本文提出了一种名为STDLens的安全方法以保护FL免受此类攻击。我们首先调查现有的缓解机制并分析它们在空间聚类分析梯度时由于固有误差而产生的失败情况。基于这些洞见，我们提出了一个三层的取证框架来识别和排除这种特殊的梯度，并在FL过程中恢复性能。我们考虑了三种类型的自适应攻击，并展示了STDLens对高级对手具有的稳健性。大量实验表明，STDLens在物体检测方面实现了最先进的性能，并且具有防止模型挟持的鲁棒性。

    Federated Learning (FL) has been gaining popularity as a collaborative learning framework to train deep learning-based object detection models over a distributed population of clients. Despite its advantages, FL is vulnerable to model hijacking. The attacker can control how the object detection system should misbehave by implanting Trojaned gradients using only a small number of compromised clients in the collaborative learning process. This paper introduces STDLens, a principled approach to safeguarding FL against such attacks. We first investigate existing mitigation mechanisms and analyze their failures caused by the inherent errors in spatial clustering analysis on gradients. Based on the insights, we introduce a three-tier forensic framework to identify and expel Trojaned gradients and reclaim the performance over the course of FL. We consider three types of adaptive attacks and demonstrate the robustness of STDLens against advanced adversaries. Extensive experiments show that STD
    
[^104]: Reflexion：具有动态记忆和自我反思的自主智能体

    Reflexion: an autonomous agent with dynamic memory and self-reflection. (arXiv:2303.11366v1 [cs.AI])

    [http://arxiv.org/abs/2303.11366](http://arxiv.org/abs/2303.11366)

    本文提出 Reflexion 方法，给智能体赋予了动态记忆和自我反思能力，以增强其任务特定的行动选择能力。

    

    最近决策大型语言模型（LLM）代理的发展在各种基准测试中展现出卓越的性能。然而，这些最先进的方法通常需要内部模型微调、外部模型微调或在定义的状态空间上进行策略优化。由于高质量训练数据的稀缺性或缺乏良好定义的状态空间，实现这些方法可能会具有挑战性。此外，这些代理没有人类决策过程固有的某些品质，特别是从错误中学习的能力。通过反思，人类可以通过试错过程高效地解决新的问题。在最近的研究基础上，我们提出 Reflexion，一种将动态记忆和自我反思能力赋予智能体的方法，以增强其现有的推理轨迹和任务特定的行动选择能力。为了实现完全自动化，我们介绍了一种简单而有效的方法。

    Recent advancements in decision-making large language model (LLM) agents have demonstrated impressive performance across various benchmarks. However, these state-of-the-art approaches typically necessitate internal model fine-tuning, external model fine-tuning, or policy optimization over a defined state space. Implementing these methods can prove challenging due to the scarcity of high-quality training data or the lack of well-defined state space. Moreover, these agents do not possess certain qualities inherent to human decision-making processes, specifically the ability to learn from mistakes. Self-reflection allows humans to efficiently solve novel problems through a process of trial and error. Building on recent research, we propose Reflexion, an approach that endows an agent with dynamic memory and self-reflection capabilities to enhance its existing reasoning trace and task-specific action choice abilities. To achieve full automation, we introduce a straightforward yet effective 
    
[^105]: 使用SoftER Teacher增强半监督Few-Shot目标检测

    Boosting Semi-Supervised Few-Shot Object Detection with SoftER Teacher. (arXiv:2303.05739v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.05739](http://arxiv.org/abs/2303.05739)

    本文研究了半监督Few-Shot目标检测任务，发现未标记数据对提高半监督FSOD有好处。受此发现启发，我们引入了SoftER Teacher，一种强大的检测器，用于改进FSOD，不需要丰富的标签，并能在性能方面超越强有力的监督检测器，而且不会出现灾难性遗忘。

    

    Few-shot目标检测（FSOD）是一个新兴的问题，旨在从少量样本中检测新概念。现有的FSOD方法假定有丰富的基础标签来适应新对象。本文研究了半监督FSOD任务，考虑到基础和新标签同时很少的现实情况。我们探索了未标记数据的实用性，并发现其通过区域提议的方式提高了半监督FSOD的显着能力。受此发现的启发，我们引入了SoftER Teacher，一种结合区域提议上的伪标记和表示学习的强大检测器，以利用未标记数据改进FSOD，而无需依赖丰富的标签。广泛的实验表明，SoftER Teacher超越了强有力的监督检测器的新性能，仅使用所需基础标签的10％，而不会出现之前方法中观察到的灾难性遗忘。我们的工作还揭示了半监督学习和Few-Shot目标检测之间潜在关系的可能性。

    Few-shot object detection (FSOD) is an emerging problem aimed at detecting novel concepts from few exemplars. Existing approaches to FSOD assume abundant base labels to adapt to novel objects. This paper studies the task of semi-supervised FSOD by considering a realistic scenario in which both base and novel labels are simultaneously scarce. We explore the utility of unlabeled data and discover its remarkable ability to boost semi-supervised FSOD by way of region proposals. Motivated by this finding, we introduce SoftER Teacher, a robust detector combining pseudo-labeling with representation learning on region proposals, to harness unlabeled data for improved FSOD without relying on abundant labels. Extensive experiments show that SoftER Teacher surpasses the novel performance of a strong supervised detector using only 10% of required base labels, without experiencing catastrophic forgetting observed in prior approaches. Our work also sheds light on a potential relationship between sem
    
[^106]: 分布式鲁棒强化学习的样本复杂性界限的改进

    Improved Sample Complexity Bounds for Distributionally Robust Reinforcement Learning. (arXiv:2303.02783v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.02783](http://arxiv.org/abs/2303.02783)

    本文改进了分布式鲁棒强化学习的样本复杂度界限，提出了稳健分阶段价值学习（RPVL）算法来解决表格剧情学习环境下的不确定性问题。

    

    本文考虑了在训练环境与测试环境之间参数不匹配的情况下学习控制策略的问题。我们将其制定为一个分布式鲁棒强化学习(DR-RL)问题，其中目标是学习在不确定集中针对环境最坏的随机模型下最大化价值函数的策略。我们专注于表格剧情学习环境，在不确定集被定义在名义（训练）环境的生成模型周围的情况下，算法可以访问该环境。我们提出了稳健分阶段价值学习(RPVL)算法来解决用四种不同发散度指定的不确定集的问题: 全变分、卡方、Kullback-Leibler和Wasserstein。我们证明了我们的算法达到了 $\tilde{\mathcal{O}}(|\mathcal{S}||\mathcal{A}| H^{5})$ 样本复杂性，这比现有结果平均好了一倍的 $|\mathcal{S}|$

    We consider the problem of learning a control policy that is robust against the parameter mismatches between the training environment and testing environment. We formulate this as a distributionally robust reinforcement learning (DR-RL) problem where the objective is to learn the policy which maximizes the value function against the worst possible stochastic model of the environment in an uncertainty set. We focus on the tabular episodic learning setting where the algorithm has access to a generative model of the nominal (training) environment around which the uncertainty set is defined. We propose the Robust Phased Value Learning (RPVL) algorithm to solve this problem for the uncertainty sets specified by four different divergences: total variation, chi-square, Kullback-Leibler, and Wasserstein. We show that our algorithm achieves $\tilde{\mathcal{O}}(|\mathcal{S}||\mathcal{A}| H^{5})$ sample complexity, which is uniformly better than the existing results by a factor of $|\mathcal{S}|
    
[^107]: 基于马尔科夫抽样方案的随机梯度下降算法研究

    Stochastic Gradient Descent under Markovian Sampling Schemes. (arXiv:2302.14428v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2302.14428](http://arxiv.org/abs/2302.14428)

    本文研究了基于马尔科夫抽样方案的随机梯度下降算法，提出了MC-SAG算法实现了用于分布式算法的通信效率高的token 算法。

    

    本文研究了一种变形的随机梯度下降算法，其中优化器只能访问马尔科夫抽样方案。这些方案涵盖从具有随机行走者（token算法）的分散优化到RL和在线系统识别问题的应用。我们专注于在对基础马尔科夫链和优化函数施加最不限制性的假设的情况下获得收敛速度。我们首先揭示了样本随机梯度沿着马尔可夫链路径抽样的方法的理论下界，使出现了对基础马尔可夫链的命中时间的依赖性。然后，我们研究了比之前作品更温和的规律性假设下的Markov链SGD（MC-SGD）。最后，我们介绍了MC-SAG，这是MC-SGD的一种带有方差缩减的替代方案，仅取决于马尔可夫链的碰撞时间，因此获得了通信效率高的token 算法。

    We study a variation of vanilla stochastic gradient descent where the optimizer only has access to a Markovian sampling scheme. These schemes encompass applications that range from decentralized optimization with a random walker (token algorithms), to RL and online system identification problems. We focus on obtaining rates of convergence under the least restrictive assumptions possible on the underlying Markov chain and on the functions optimized. We first unveil the theoretical lower bound for methods that sample stochastic gradients along the path of a Markov chain, making appear a dependency in the hitting time of the underlying Markov chain. We then study Markov chain SGD (MC-SGD) under much milder regularity assumptions than prior works. We finally introduce MC-SAG, an alternative to MC-SGD with variance reduction, that only depends on the hitting time of the Markov chain, therefore obtaining a communication-efficient token algorithm.
    
[^108]: 改变很难：子群体转变的深入探究

    Change is Hard: A Closer Look at Subpopulation Shift. (arXiv:2302.12254v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.12254](http://arxiv.org/abs/2302.12254)

    本文分析了子群体转变的各种机制，对20个最先进的算法在12个领域内进行了全面的基准测试，发现现有算法只能应对某些转变，进一步地，提出一种简单易行的选择标准来改善现有算法性能。

    

    机器学习模型通常在训练数据中代表性不足的子群体上表现不佳。然而，对于导致子群体转变的机制以及算法在如此不同的转变中如何进行普遍化，我们知之甚少。在这项工作中，我们对子群体转变进行了细致的分析。首先，我们提出了一个统一的框架来剖析和解释子群体中的常见转变。然后，我们在视觉、语言和医疗领域的12个真实数据集上对20个最先进的算法进行了全面的基准测试。通过训练10,000多个模型得到的结果，我们揭示了未来在这个领域取得进展的有趣观察结果。首先，现有算法仅能在某些类型的转变上提高子群体的鲁棒性，而在其他类型的转变上则不能。此外，虽然当前算法依赖于群体标注的验证数据进行模型选择，但我们发现基于最差类别准确度的简单选择标准其实非常有效。

    Machine learning models often perform poorly on subgroups that are underrepresented in the training data. Yet, little is understood on the variation in mechanisms that cause subpopulation shifts, and how algorithms generalize across such diverse shifts at scale. In this work, we provide a fine-grained analysis of subpopulation shift. We first propose a unified framework that dissects and explains common shifts in subgroups. We then establish a comprehensive benchmark of 20 state-of-the-art algorithms evaluated on 12 real-world datasets in vision, language, and healthcare domains. With results obtained from training over 10,000 models, we reveal intriguing observations for future progress in this space. First, existing algorithms only improve subgroup robustness over certain types of shifts but not others. Moreover, while current algorithms rely on group-annotated validation data for model selection, we find that a simple selection criterion based on worst-class accuracy is surprisingly
    
[^109]: 一站式解决方案：利用预训练 LM 进行强大的时间序列分析

    One Fits All:Power General Time Series Analysis by Pretrained LM. (arXiv:2302.11939v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.11939](http://arxiv.org/abs/2302.11939)

    本论文提出了一种称为 Frozen Pretrained Transformer (FPT) 的预训练模型，利用从数十亿标记训练出来的语言或 CV 模型进行时间序列分析的所有主要类型任务的微调，进而使其在所有任务中都具备着最先进的性能和泛化能力。

    

    尽管预训练模型在自然语言处理 (NLP) 和计算机视觉 (CV) 领域取得了巨大成功，但在通用时间序列分析领域取得的进展有限。与 NLP 和 CV 不同的是，这些领域采用统一模型即可执行不同的任务，而在每个时间序列分析任务中，专门设计的方法仍然占据主导地位，如分类、异常检测、预测和少样本学习。阻碍预训练模型发展的主要挑战是缺乏大量用于训练的数据。在本文中，我们通过利用从数十亿标记训练出来的语言或 CV 模型，来解决这一挑战，用于时间序列分析。具体而言，我们避免改变预训练语言或图像模型中残差块中的自注意力和前向传递层。这种模型被称为冻结的预训练变压器 (FPT)，通过对涉及时间序列分析的所有主要类型的任务进行微调进行评估，包括分类、异常检测、预测和少样本学习等。实验结果证明，FPT 在所有任务中都具有最先进的性能和泛化能力。

    Although we have witnessed great success of pre-trained models in natural language processing (NLP) and computer vision (CV), limited progress has been made for general time series analysis. Unlike NLP and CV where a unified model can be used to perform different tasks, specially designed approach still dominates in each time series analysis task such as classification, anomaly detection, forecasting, and few-shot learning. The main challenge that blocks the development of pre-trained model for time series analysis is the lack of a large amount of data for training. In this work, we address this challenge by leveraging language or CV models, pre-trained from billions of tokens, for time series analysis. Specifically, we refrain from altering the self-attention and feedforward layers of the residual blocks in the pre-trained language or image model. This model, known as the Frozen Pretrained Transformer (FPT), is evaluated through fine-tuning on all major types of tasks involving time s
    
[^110]: 机器人系统的离散对称性: 基于群论和数据驱动分析的研究

    On discrete symmetries of robotics systems: A group-theoretic and data-driven analysis. (arXiv:2302.10433v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2302.10433](http://arxiv.org/abs/2302.10433)

    本文研究了运动系统的离散形态对称性，并提出了一个理论和实践框架，用于识别系统的形态对称群，并分析对称性在数据增强和控制设计中的应用。

    

    本文对运动系统的离散形态对称性进行了全面的研究，这在生物和人工运动系统中经常观察到，例如多腿、游泳和飞行的动物/机器人/虚拟角色。这些对称性源自系统形态中一个或多个平面/轴的对称性存在，导致身体部件的谐波复制和分布。我们阐述了形态对称性如何延伸到系统动力学、最优控制策略以及与系统动力学演化相关的所有本体感和外感测量中的对称性。在数据驱动方法的背景下，对称性代表一种归纳偏置，证明了数据增强或对称函数逼近器的使用。为了解决这个问题，我们提出了一个理论和实践框架，用于识别系统的形态对称群G并描述其在本体感和外感测量、最优控制策略以及系统动力学方面的对称性。该框架涉及到数据驱动和群论工具，例如主成分分析，置换测试和表示理论。提出的方法学通过确定一个仿生脊椎机器人的对称群并分析其对数据增强和控制设计的影响来进行说明。

    We present a comprehensive study on discrete morphological symmetries of dynamical systems, which are commonly observed in biological and artificial locomoting systems, such as legged, swimming, and flying animals/robots/virtual characters. These symmetries arise from the presence of one or more planes/axis of symmetry in the system's morphology, resulting in harmonious duplication and distribution of body parts. Significantly, we characterize how morphological symmetries extend to symmetries in the system's dynamics, optimal control policies, and in all proprioceptive and exteroceptive measurements related to the system's dynamics evolution. In the context of data-driven methods, symmetry represents an inductive bias that justifies the use of data augmentation or symmetric function approximators. To tackle this, we present a theoretical and practical framework for identifying the system's morphological symmetry group $\G$ and characterizing the symmetries in proprioceptive and exteroc
    
[^111]: 将黑匣子分解为可解释模型的混合物：路线规划，解释，重复。

    Dividing and Conquering a BlackBox to a Mixture of Interpretable Models: Route, Interpret, Repeat. (arXiv:2302.10289v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.10289](http://arxiv.org/abs/2302.10289)

    本文提出了一种从黑盒模型中构建可解释模型的方法。该方法将黑盒模型分成可解释模型的混合物和残差网络，并使用一阶逻辑对可解释模型进行基本推理。此方法在多个数据集上表现优异且产生高度可解释的模型。

    

    机器学习模型设计要么从解释性模型开始，要么从黑盒开始并事后解释。黑盒模型灵活但难以解释，而解释性模型本质上是可解释的。然而，解释性模型需要广泛的机器学习知识，并且往往比它们的黑盒变体不够灵活和表现不佳。本文旨在模糊黑盒的事后解释和构建可解释模型之间的界限。我们从黑盒开始，迭代地Carve出一种混合解释模型（MoIE）和一个残余网络。每个可解释模型专门处理一个样本子集，并使用一阶逻辑(FOL)对其进行解释，从黑盒中提供基本推理概念。我们通过灵活的残差路由其余的样本。我们在残转网络上重复该方法，直到所有可解释模型解释所需比例的数据。我们进行了大量实验，结果表明我们的路线规划，解释和重复方法在各种数据集上优于目前几种黑匣子模型解释方法，并产生高度可解释的模型。

    ML model design either starts with an interpretable model or a Blackbox and explains it post hoc. Blackbox models are flexible but difficult to explain, while interpretable models are inherently explainable. Yet, interpretable models require extensive ML knowledge and tend to be less flexible and underperforming than their Blackbox variants. This paper aims to blur the distinction between a post hoc explanation of a Blackbox and constructing interpretable models. Beginning with a Blackbox, we iteratively carve out a mixture of interpretable experts (MoIE) and a residual network. Each interpretable model specializes in a subset of samples and explains them using First Order Logic (FOL), providing basic reasoning on concepts from the Blackbox. We route the remaining samples through a flexible residual. We repeat the method on the residual network until all the interpretable models explain the desired proportion of data. Our extensive experiments show that our route, interpret, and repeat
    
[^112]: Navya3DSeg -- 用于自动驾驶的Navya三维语义分割数据集和拆分生成

    Navya3DSeg -- Navya 3D Semantic Segmentation Dataset & split generation for autonomous vehicles. (arXiv:2302.08292v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.08292](http://arxiv.org/abs/2302.08292)

    Navya3DSeg是一个新的、具有多样标签空间的数据集，对应于大规模生产级操作领域，可用于自动驾驶感知，同时提供了一个新的方法来生成顺序数据集拆分。

    

    自动驾驶感知目前严重依赖于基于深度学习的体系结构，需要大规模注释的数据集，具有相应的策划与标注成本。三维语义数据对于核心感知任务如障碍检测和自我定位非常有用。我们提出了一个新的数据集Navya3DSeg，具有多样的标签空间，对应于大规模生产级操作领域，包括来自13个国家的城市，乡村，工业区和大学。它包含23个带标签序列和25个没有标签的补充序列，旨在探讨基于点云的自监督和半监督语义分割基准。我们还提出了一种基于迭代多标签分层的顺序数据集拆分生成的新方法，并演示了比SemanticKITTI数据集提出的原始拆分+1.2％ mIoU的改进。这是一个完整的语义分割基准。

    Autonomous driving (AD) perception today relies heavily on deep learning based architectures requiring large scale annotated datasets with their associated costs for curation and annotation. The 3D semantic data are useful for core perception tasks such as obstacle detection and ego-vehicle localization. We propose a new dataset, Navya 3D Segmentation (Navya3DSeg), with a diverse label space corresponding to a large scale production grade operational domain, including rural, urban, industrial sites and universities from 13 countries. It contains 23 labeled sequences and 25 supplementary sequences without labels, designed to explore self-supervised and semi-supervised semantic segmentation benchmarks on point clouds. We also propose a novel method for sequential dataset split generation based on iterative multi-label stratification, and demonstrated to achieve a +1.2% mIoU improvement over the original split proposed by SemanticKITTI dataset. A complete benchmark for semantic segmentati
    
[^113]: 预测财政困境下的城市:一种机器学习方法

    Predicting municipalities in financial distress: a machine learning approach enhanced by domain expertise. (arXiv:2302.05780v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.05780](http://arxiv.org/abs/2302.05780)

    本文运用机器学习模型结合会计司法专家的知识和经验，预测意大利城市的财政困境。

    

    尽管与私营公司的破产相比可以类比，城市的财政困境对社区的福祉有着更严重的影响。预测城市的财政困境可能是一个复杂的任务，因为它涉及了理解许多因素，这些因素可能会影响到城市的财政健康状况。本文通过评估机器学习模型来预测意大利城市的财政困境。通过将会计司法专家的专业知识和经验纳入特征提取过程，可以确保模型考虑到了与城市财政健康状况相关的广泛信息。本研究的结果表明，结合领域专家的知识和经验使用机器学习模型可以更好地预测城市的财政困境。

    Financial distress of municipalities, although comparable to bankruptcy of private companies, has a far more serious impact on the well-being of communities. For this reason, it is essential to detect deficits as soon as possible. Predicting financial distress in municipalities can be a complex task, as it involves understanding a wide range of factors that can affect a municipality's financial health. In this paper, we evaluate machine learning models to predict financial distress in Italian municipalities. Accounting judiciary experts have specialized knowledge and experience in evaluating the financial performance, and they use a range of indicators to make their assessments. By incorporating these indicators in the feature extraction process, we can ensure that the model is taking into account a wide range of information that is relevant to the financial health of municipalities. The results of this study indicate that using machine learning models in combination with the knowledge
    
[^114]: DeepVATS：面向时间序列的深度视觉分析

    DeepVATS: Deep Visual Analytics for Time Series. (arXiv:2302.03858v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.03858](http://arxiv.org/abs/2302.03858)

    DeepVATS是一个开源的工具，用于对时间序列数据进行深度视觉分析，通过自监督的方式训练掩码时间序列自编码器，重建时间序列的补丁，并将模型嵌入中所包含的知识投影到交互式图中，从而可以轻松识别时间序列模式和异常。

    

    深度视觉分析（DVA）近来兴起，目的是通过深度学习支持可视交互系统，以提供大规模数据处理能力并在不同数据和领域中统一其实现。本文介绍了DeepVATS，这是一个开源工具，将DVA领域引入时间序列数据。DeepVATS以自监督方式训练掩码时间序列自编码器，重建时间序列的补丁，并将模型嵌入中所包含的知识投影到交互式图中，从而可以轻松识别时间序列的模式和异常。工具包括用于数据处理流水线和模型训练的后端以及带有交互式用户界面的前端。我们在合成和实际数据集上运行实验，证明了DeepVATS的实用价值。代码公开在https://github.com/vrodriguez上。

    The field of Deep Visual Analytics (DVA) has recently arisen from the idea of developing Visual Interactive Systems supported by deep learning, in order to provide them with large-scale data processing capabilities and to unify their implementation across different data and domains. In this paper we present DeepVATS, an open-source tool that brings the field of DVA into time series data. DeepVATS trains, in a self-supervised way, a masked time series autoencoder that reconstructs patches of a time series, and projects the knowledge contained in the embeddings of that model in an interactive plot, from which time series patterns and anomalies emerge and can be easily spotted. The tool includes a back-end for data processing pipeline and model training, as well as a front-end with a interactive user interface. We report on results that validate the utility of DeepVATS, running experiments on both synthetic and real datasets. The code is publicly available on https://github.com/vrodriguez
    
[^115]: 超越统计相似性：重新思考机器学习在工程设计中的度量方法

    Beyond Statistical Similarity: Rethinking Metrics for Deep Generative Models in Engineering Design. (arXiv:2302.02913v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02913](http://arxiv.org/abs/2302.02913)

    本文提供了一篇深度学习在工程设计中度量方法的综述和指南。传统的基于似然性的统计度量方法在对工程应用的要求上可能无法充分捕捉，因此本文编辑了一组全面的新度量标准，旨在解决传统度量标准的缺点，并更好地与工程设计的需求相一致。通过案例研究，本文展示了这些度量标准如何应用于评估深度生成模型在工程设计中的性能，并发现这些度量标准在捕捉设计的重要细微差别方面表现优于传统的统计度量标准。

    

    深度生成模型，如变分自编码器（VAEs），生成对抗网络（GANs），扩散模型和Transformer等，在图像和语音合成、自然语言处理和药物开发等各种应用中显示出巨大的潜力。然而，在工程设计问题中应用这些模型时，评估这些模型的性能可能会很具有挑战性，因为传统的基于似然性的统计度量方法可能无法充分捕捉工程应用的要求。本文旨在提供一篇深度学习在工程设计中的度量指南和综述。首先，我们总结了深度生成模型的“经典”评估度量标准，这些标准基于机器学习理论和典型的计算机应用，然后使用案例研究，强调了这些度量标准为何很少能够转化为设计问题但又因缺乏确立的替代选择而经常使用。接下来，我们编辑了一组全面的新度量标准，旨在解决传统度量标准的缺点，并更好地与工程设计的需求相一致。我们演示了如何应用这些度量标准来评估深度生成模型在工程设计应用中的性能。我们的结果表明，提出的度量方法在捕捉设计的重要细微差别方面优于传统的统计度量标准，因此在工程设计情境中为深度生成模型提供了更准确的评估。

    Deep generative models, such as Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Diffusion Models, and Transformers, have shown great promise in a variety of applications, including image and speech synthesis, natural language processing, and drug discovery. However, when applied to engineering design problems, evaluating the performance of these models can be challenging, as traditional statistical metrics based on likelihood may not fully capture the requirements of engineering applications. This paper doubles as a review and a practical guide to evaluation metrics for deep generative models (DGMs) in engineering design. We first summarize well-accepted `classic' evaluation metrics for deep generative models grounded in machine learning theory and typical computer science applications. Using case studies, we then highlight why these metrics seldom translate well to design problems but see frequent use due to the lack of established alternatives. Next, we curat
    
[^116]: 基于机器学习材料相似性从科学文献中推荐配方来促进无机合成前体预测

    Precursor recommendation for inorganic synthesis by machine learning materials similarity from scientific literature. (arXiv:2302.02303v2 [cond-mat.mtrl-sci] UPDATED)

    [http://arxiv.org/abs/2302.02303](http://arxiv.org/abs/2302.02303)

    本研究利用从科学文献中提取的合成配方的知识库，自动学习推荐新目标材料的前体材料，通过学习材料化学相似性并将新目标材料的合成参照类似材料的先前合成程序，成功率至少为82%。

    

    合成预测是推动先进材料快速设计的关键因素。然而，对于无机材料来说，确定合成变量，如前体材料的选择，是具有挑战性的，因为在加热过程中反应顺序不是很清楚。本研究利用从科学文献中提取的29,900个固态合成配方的知识库，自动学习推荐新目标材料的前体材料。数据驱动方法通过学习材料化学相似性，并将新目标材料的合成参照类似材料的先前合成程序，模拟人类合成设计。当针对2,654个未见过的测试目标材料提出五种前体组合时，推荐策略的成功率至少为82％。我们的方法以数学形式捕捉了几十年的经验合成数据，使得它可以用于推荐引擎和自治。

    Synthesis prediction is a key accelerator for the rapid design of advanced materials. However, determining synthesis variables such as the choice of precursor materials is challenging for inorganic materials because the sequence of reactions during heating is not well understood. In this work, we use a knowledge base of 29,900 solid-state synthesis recipes, text-mined from the scientific literature, to automatically learn which precursors to recommend for the synthesis of a novel target material. The data-driven approach learns chemical similarity of materials and refers the synthesis of a new target to precedent synthesis procedures of similar materials, mimicking human synthesis design. When proposing five precursor sets for each of 2,654 unseen test target materials, the recommendation strategy achieves a success rate of at least 82%. Our approach captures decades of heuristic synthesis data in a mathematical form, making it accessible for use in recommendation engines and autonomou
    
[^117]: 用户中心异构行动深度强化学习在无线网络的元宇宙虚拟现实中的应用

    User-centric Heterogeneous-action Deep Reinforcement Learning for Virtual Reality in the Metaverse over Wireless Networks. (arXiv:2302.01471v2 [cs.NI] UPDATED)

    [http://arxiv.org/abs/2302.01471](http://arxiv.org/abs/2302.01471)

    本文提出了一种用户中心异构行动深度强化学习（UCHA-DRL）算法，可以使用在无线网络的元宇宙虚拟现实中。该算法可以联合优化服务器向用户下行通信的信道访问安排和传输功率来提高用户的效用。

    

    随着各种技术的发展，元宇宙正在崛起。虚拟现实技术是元宇宙中虚拟宇宙的支撑，能够为用户带来高度沉浸式的体验。在元宇宙中，移动性备受强调，虚拟现实设备通过减少本地的计算能力来减轻重量。针对元宇宙服务器和多个虚拟现实用户的系统，本文考虑了两种情况：（i）服务器生成帧并将它们传输给用户；（ii）用户在本地生成帧，因此耗费设备能量。此外，在元宇宙中的多用户虚拟现实场景中，用户对于帧率有不同的特点和需求。因此，我们联合优化服务器向用户下行通信的信道访问安排（包括帧生成位置的决策）和传输功率来提高用户的效用。这个联合优化问题被形式化为一个用户中心异构行动深度强化学习（UCHA-DRL）问题，并采用一种新的Q网络体系结构，称为异构行动Q网络（HAQN）来求解。仿真结果表明，我们提出的UCHA-DRL算法在收敛速度、用户切换次数和用户平均帧率方面优于传统的Q学习算法和深度Q网络（DQN）算法。

    The Metaverse is emerging as maturing technologies are empowering the different facets. Virtual Reality (VR) technologies serve as the backbone of the virtual universe within the Metaverse to offer a highly immersive user experience. As mobility is emphasized in the Metaverse context, VR devices reduce their weights at the sacrifice of local computation abilities. In this paper, for a system consisting of a Metaverse server and multiple VR users, we consider two cases of (i) the server generating frames and transmitting them to users, and (ii) users generating frames locally and thus consuming device energy. Moreover, in our multi-user VR scenario for the Metaverse, users have different characteristics and demands for Frames Per Second (FPS). Then the channel access arrangement (including the decisions on frame generation location), and transmission powers for the downlink communications from the server to the users are jointly optimized to improve the utilities of users. This joint op
    
[^118]: 受限在线两阶段随机优化：通过对抗学习获得近似最优算法

    Constrained Online Two-stage Stochastic Optimization: Near Optimal Algorithms via Adversarial Learning. (arXiv:2302.00997v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.00997](http://arxiv.org/abs/2302.00997)

    在线两阶段随机优化算法的累计目标值最小化，同时保证长期平均第二阶段决策结果属于一个集合。采用对抗性学习算法从在线两阶段问题中开发在线算法，其遗憾界可以降至嵌入对抗性学习算法的遗憾界，并在各种设置下获得了新的结果。

    

    我们考虑一个在线两阶段随机优化问题，其具有有限的$T$期紧约束条件。在每个时间段，我们先作出第一阶段决策，然后观察模型参数的实现，最后从取决于第一阶段决策和模型参数的可行集中做出第二阶段决策。我们旨在最小化累计目标值，同时保证长期平均的第二阶段决策属于一个集合。我们利用对抗性学习算法从在线两阶段问题中开发在线算法。此外，我们算法的遗憾界可以降至嵌入对抗性学习算法的遗憾界。基于我们的框架，在各种设置下我们都获得了新的结果。当每个时间段的模型参数都是从相同的分布中抽取的时候，我们得到了最先进的$O（\sqrt{T}）$遗憾界，这比之前的特殊情况下的界有所提升。我们的算法还可以抵抗模型的敌对性扰动。

    We consider an online two-stage stochastic optimization with long-term constraints over a finite horizon of $T$ periods. At each period, we take the first-stage action, observe a model parameter realization and then take the second-stage action from a feasible set that depends both on the first-stage decision and the model parameter. We aim to minimize the cumulative objective value while guaranteeing that the long-term average second-stage decision belongs to a set. We develop online algorithms for the online two-stage problem from adversarial learning algorithms. Also, the regret bound of our algorithm cam be reduced to the regret bound of embedded adversarial learning algorithms. Based on our framework, we obtain new results under various settings. When the model parameter at each period is drawn from identical distributions, we derive state-of-art $O(\sqrt{T})$ regret that improves previous bounds under special cases. Our algorithm is also robust to adversarial corruptions of model
    
[^119]: 强化学习中的尖锐方差相关界限：随机和确定性环境的最佳结合

    Sharp Variance-Dependent Bounds in Reinforcement Learning: Best of Both Worlds in Stochastic and Deterministic Environments. (arXiv:2301.13446v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.13446](http://arxiv.org/abs/2301.13446)

    本研究将马尔可夫决策过程的方差相关遗憾界限应用到强化学习中，提出了两个新的环境规范来表征环境的方差属性，并设计出基于模型和无模型的算法，对于随机和确定性环境同时极小极大最优的界限是第一次被证明出来的。

    

    本文研究马尔可夫决策过程（MDPs）的方差相关遗憾界限。具有方差相关遗憾保证的算法可以自动利用具有低方差（例如，在确定性MDP上享有常量遗憾）的环境。现有算法要么独立于方差要么次优。我们首先提出两个新的环境规范来表征环境的细粒度方差属性。对于基于模型的方法，我们设计了MVP算法(Zhang等，2021a)的变种，并使用新的分析技术展示了该算法相对于我们提出的规范享有方差相关的界限。特别地，这一界限对于随机和确定性MDP同时是极小极大最优的，这是其种类中的第一个结果。我们进一步通过设计一种参考函数的算法以及一个新的带有上限加倍参考更新进度表的策略启动了关于具有方差相关遗憾界限的无模型算法的研究。最后，我们还提供了一些启示。

    We study variance-dependent regret bounds for Markov decision processes (MDPs). Algorithms with variance-dependent regret guarantees can automatically exploit environments with low variance (e.g., enjoying constant regret on deterministic MDPs). The existing algorithms are either variance-independent or suboptimal. We first propose two new environment norms to characterize the fine-grained variance properties of the environment. For model-based methods, we design a variant of the MVP algorithm (Zhang et al., 2021a) and use new analysis techniques show to this algorithm enjoys variance-dependent bounds with respect to our proposed norms. In particular, this bound is simultaneously minimax optimal for both stochastic and deterministic MDPs, the first result of its kind. We further initiate the study on model-free algorithms with variance-dependent regret bounds by designing a reference-function-based algorithm with a novel capped-doubling reference update schedule. Lastly, we also provid
    
[^120]: FedEBA+：基于熵的模型实现公平和有效联邦学习

    FedEBA+: Towards Fair and Effective Federated Learning via Entropy-Based Model. (arXiv:2301.12407v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12407](http://arxiv.org/abs/2301.12407)

    FedEBA+是一种新的联邦学习算法，它采用公平聚合方案和对齐更新方法，在同时提高全局模型性能的同时提高公平性。实验证明FedEBA+优于其他公平性联邦学习方法。

    

    确保公平性是联邦学习中至关重要的方面，它使模型在所有客户端上保持一致表现。然而，设计一种可以同时提高全局模型性能和促进公平的联邦学习算法仍然是一个艰巨的挑战，因为实现后者通常需要与前者的权衡。为了解决这一问题，我们提出了一种新的联邦学习算法FedEBA+，它在同时提高全局模型性能的同时提高公平性，该算法采用公平聚合方案和对齐更新方法。此外，我们提供了理论收敛分析，证明了FedEBA+的公平性。大量实验表明FedEBA+在公平性和全局模型性能方面均优于其他SOTA的公平联邦学习方法。

    Ensuring fairness is a crucial aspect of Federated Learning (FL), which enables the model to perform consistently across all clients. However, designing an FL algorithm that simultaneously improves global model performance and promotes fairness remains a formidable challenge, as achieving the latter often necessitates a trade-off with the former.To address this challenge, we propose a new FL algorithm, FedEBA+, which enhances fairness while simultaneously improving global model performance. FedEBA+ incorporates a fair aggregation scheme that assigns higher weights to underperforming clients and an alignment update method. In addition, we provide theoretical convergence analysis and show the fairness of FedEBA+. Extensive experiments demonstrate that FedEBA+ outperforms other SOTA fairness FL methods in terms of both fairness and global model performance.
    
[^121]: BiBench：网络二值化基准测试和分析

    BiBench: Benchmarking and Analyzing Network Binarization. (arXiv:2301.11233v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.11233](http://arxiv.org/abs/2301.11233)

    BiBench提供了一个严谨设计的基准测试和深入分析，揭示了网络二值化面临的挑战和特性，为从业者和研究人员提供了有价值的参考。

    

    网络二值化成为最有前途的压缩方法之一，通过最小化比特宽度，提供了非凡的计算和内存节省。然而，最近的研究表明，在现实场景下，将现有的二值化算法应用于不同的任务、架构和硬件仍然不是一件容易的事情。二值化的常见挑战，如精度降低和效率限制，表明其属性尚未完全理解。为了弥补这一空白，我们提出了BiBench，一个严谨设计的网络二值化基准测试，并进行了深入分析。我们首先仔细审查了实际生产中对二值化的要求，为全面公正地进行研究定义了评估轨道和度量标准。然后，我们评估和分析了一系列在操作员级别和影响广泛的里程碑二值化算法。我们的基准测试揭示了 1）二值化操作员对最终精度有关键影响；2）不应低估网络拓扑和超参数调整的重要性；以及3）不同算法在不同任务和架构下的性能差异显着。我们的研究为网络二值化的特性和挑战提供了深入见解，为从业者和研究人员提供了有价值的参考。

    Network binarization emerges as one of the most promising compression approaches offering extraordinary computation and memory savings by minimizing the bit-width. However, recent research has shown that applying existing binarization algorithms to diverse tasks, architectures, and hardware in realistic scenarios is still not straightforward. Common challenges of binarization, such as accuracy degradation and efficiency limitation, suggest that its attributes are not fully understood. To close this gap, we present BiBench, a rigorously designed benchmark with in-depth analysis for network binarization. We first carefully scrutinize the requirements of binarization in the actual production and define evaluation tracks and metrics for a comprehensive and fair investigation. Then, we evaluate and analyze a series of milestone binarization algorithms that function at the operator level and with extensive influence. Our benchmark reveals that 1) the binarized operator has a crucial impact o
    
[^122]: 哪些经验可以影响机器人的行为？具有淘汰正则化的策略迭代方法研究

    Which Experiences Are Influential for Your Agent? Policy Iteration with Turn-over Dropout. (arXiv:2301.11168v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.11168](http://arxiv.org/abs/2301.11168)

    本文提出了一种名为PI+ToD的方法，它通过淘汰正则化来高效估算经验对RL代理性能的影响。

    

    在经验回放的强化学习（RL）中，存储在回放缓冲区中的经验会影响RL代理的性能。有关经验影响的信息对于经验清理和分析等各种目的都非常有价值。一个估计单个经验影响的方法是代理比较，但当经验数量很多时，这种方法的成本是难以承受的。在本文中，我们提出了PI+ToD作为一种有效估算经验影响的方法。 PI+ToD是一种策略迭代方法，通过利用淘汰正则化来高效估算经验的影响。我们在MuJoCo环境中的实验中展示了PI + ToD的效率。

    In reinforcement learning (RL) with experience replay, experiences stored in a replay buffer influence the RL agent's performance. Information about the influence is valuable for various purposes, including experience cleansing and analysis. One method for estimating the influence of individual experiences is agent comparison, but it is prohibitively expensive when there is a large number of experiences. In this paper, we present PI+ToD as a method for efficiently estimating the influence of experiences. PI+ToD is a policy iteration that efficiently estimates the influence of experiences by utilizing turn-over dropout. We demonstrate the efficiency of PI+ToD with experiments in MuJoCo environments.
    
[^123]: 基于语言解释的去偏见: 通过语言解释消除未知的视觉偏见

    Bias-to-Text: Debiasing Unknown Visual Biases through Language Interpretation. (arXiv:2301.11104v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.11104](http://arxiv.org/abs/2301.11104)

    本文提出了基于语言解释的去偏见(B2T)框架，通过分析图像标题中的关键词，比较关键词和图像之间的相似性，识别和减缓视觉模型中的偏见，并提出了针对零样本分类器和文本到图像扩散模型的去偏见策略。

    

    模型中的偏见在部署机器学习系统时构成重要问题，但以可解释的方式诊断这些偏见可能具有挑战性。为了解决这个问题，我们引入了去偏见(B2T)框架，该框架利用语言解释来识别和缓解视觉模型中的偏见，例如图象分类器和文本生成模型。我们对视觉偏差的语言描述提供了可解释的形式，使得能够发现新的偏见并有效地对模型进行去偏见。为了实现这一点，我们分析了被误预测或生成的图像标题中的常见关键词。在这里，我们提出了新的评分函数，通过比较偏见关键词和图像之间的相似性来避免标题中的偏见。此外，我们还提出了使用B2T框架中的偏见关键词对零样本分类器和文本到图像扩散模型进行去偏见的策略。我们展示了我们的框架在各种图像分类和生成任务上的有效性。

    Biases in models pose a critical issue when deploying machine learning systems, but diagnosing them in an explainable manner can be challenging. To address this, we introduce the bias-to-text (B2T) framework, which uses language interpretation to identify and mitigate biases in vision models, such as image classifiers and text-to-image generative models. Our language descriptions of visual biases provide explainable forms that enable the discovery of novel biases and effective model debiasing. To achieve this, we analyze common keywords in the captions of mispredicted or generated images. Here, we propose novel score functions to avoid biases in captions by comparing the similarities between bias keywords and those images. Additionally, we present strategies to debias zero-shot classifiers and text-to-image diffusion models using the bias keywords from the B2T framework. We demonstrate the effectiveness of our framework on various image classification and generation tasks. For classifi
    
[^124]: 噪声调度对扩散模型的重要性研究

    On the Importance of Noise Scheduling for Diffusion Models. (arXiv:2301.10972v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.10972](http://arxiv.org/abs/2301.10972)

    本研究实证发现，噪声调度对于扩散生成模型的表现至关重要，最优的噪声调度策略会随着任务的不同而有所不同，并且，在图像大小增加时最优噪声调度策略会朝更嘈杂的方向转移。将输入数据按比例缩放并保持噪声调度函数不变，能够在不同图像大小上获得良好的效果。

    

    本文通过实证研究，探究了噪声调度策略对去噪扩散生成模型的影响。本研究得出了三个结论：（1）噪声调度对性能至关重要，最优的噪声调度策略取决于任务（例如图像大小）；（2）在图像大小增加时，最优噪声调度会朝更嘈杂的方向转移（由于像素冗余增加）；（3）仅通过将输入数据按比例$b$缩放并保持噪声调度函数不变（相当于将logSNR向上移动$\log b$），便是一种适用于各种图像大小的良好策略。将这一简单方法与最近提出的递归接口网络（RIN）结合起来，能够在ImageNet上生成1024$\times$1024分辨率的高保真度图像。这一方法能够以单步、端到端方式生成多样且高质量的图像。

    We empirically study the effect of noise scheduling strategies for denoising diffusion generative models. There are three findings: (1) the noise scheduling is crucial for the performance, and the optimal one depends on the task (e.g., image sizes), (2) when increasing the image size, the optimal noise scheduling shifts towards a noisier one (due to increased redundancy in pixels), and (3) simply scaling the input data by a factor of $b$ while keeping the noise schedule function fixed (equivalent to shifting the logSNR by $\log b$) is a good strategy across image sizes. This simple recipe, when combined with recently proposed Recurrent Interface Network (RIN), yields state-of-the-art pixel-based diffusion models for high-resolution images on ImageNet, enabling single-stage, end-to-end generation of diverse and high-fidelity images at 1024$\times$1024 resolution (without upsampling/cascades).
    
[^125]: 基于抽样的Nyström逼近和核积分。

    Sampling-based Nystr\"om Approximation and Kernel Quadrature. (arXiv:2301.09517v2 [math.NA] UPDATED)

    [http://arxiv.org/abs/2301.09517](http://arxiv.org/abs/2301.09517)

    本文提出了一种基于抽样的Nyström逼近方法用于核积分。同时，引入了一种非i.i.d.地标点的理论保证方法，使得提高了逼近的精度。

    

    我们分析与概率测量相关的正定核的Nyström逼近。我们首先证明了传统Nyström逼近在连续区间中使用i.i.d.抽样和奇异值分解的改进误差界，证明技巧借鉴了统计学习理论。我们进一步引入了Nyström逼近中的子空间精细选择，这是适用于非i.i.d.地标点的理论保证。最后，我们讨论了它们在凸核积分中的应用，并给出了新的理论保证以及数值观察。

    We analyze the Nystr\"om approximation of a positive definite kernel associated with a probability measure. We first prove an improved error bound for the conventional Nystr\"om approximation with i.i.d. sampling and singular-value decomposition in the continuous regime; the proof techniques are borrowed from statistical learning theory. We further introduce a refined selection of subspaces in Nystr\"om approximation with theoretical guarantees that is applicable to non-i.i.d. landmark points. Finally, we discuss their application to convex kernel quadrature and give novel theoretical guarantees as well as numerical observations.
    
[^126]: 深度神经网络不安全输入计数的#DNN-Verification问题

    The #DNN-Verification problem: Counting Unsafe Inputs for Deep Neural Networks. (arXiv:2301.07068v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2301.07068](http://arxiv.org/abs/2301.07068)

    本文提出了#DNN-Verification问题，即计算违反特定安全性质的DNN输入配置数量的问题。作者提出了一种新的方法和一种随机的近似方法，分别给出了确切的违规计数和可证明概率界，并在安全关键基准测试上进行了实验比较。

    

    深度神经网络（DNN）在需要高度安全性的关键任务中，例如自动驾驶中越来越被采用。虽然最先进的验证器可以用来检查DNN是否不安全，即是否存在至少一种不安全的输入配置，但它们的是/否输出对于其他目的（如屏蔽、模型选择或培训改进）的信息不足够详细。在本文中，我们介绍了#DNN-Verification问题，它涉及计算导致DNN违反特定安全性质的输入配置数量。我们分析了这个问题的复杂性，并提出了一种新的方法，它返回确切的违规计数。由于该问题的#P完备性，我们还提出了一种随机的近似方法，该方法提供了正确计数的可证明概率界，同时显著降低了计算要求。我们在一组安全关键基准测试上呈现了实验结果，比较了我们的方法与最先进的验证器和基于计数的启发式算法。

    Deep Neural Networks are increasingly adopted in critical tasks that require a high level of safety, e.g., autonomous driving. While state-of-the-art verifiers can be employed to check whether a DNN is unsafe w.r.t. some given property (i.e., whether there is at least one unsafe input configuration), their yes/no output is not informative enough for other purposes, such as shielding, model selection, or training improvements. In this paper, we introduce the #DNN-Verification problem, which involves counting the number of input configurations of a DNN that result in a violation of a particular safety property. We analyze the complexity of this problem and propose a novel approach that returns the exact count of violations. Due to the #P-completeness of the problem, we also propose a randomized, approximate method that provides a provable probabilistic bound of the correct count while significantly reducing computational requirements. We present experimental results on a set of safety-cr
    
[^127]: 面向驾驶员监测应用的联邦迁移有序个性化学习

    Federated Transfer-Ordered-Personalized Learning for Driver Monitoring Application. (arXiv:2301.04829v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.04829](http://arxiv.org/abs/2301.04829)

    本文提出了一个名为FedTOP的联邦迁移有序个性化学习框架，用于解决数据和系统异构性、大规模并行通信资源、恶意攻击和数据污染等问题。实验结果显示，该框架在两个数据集上的测试客户端分别达到92.32％和95.96％的准确率。与基线相比，在准确率方面提高了462％，并降低了37.46％的通信成本。

    

    联邦学习通过共享在本地数据上训练的客户端模型参数来实现协作学习，提高学习效率，在物联网中具有广泛应用前景。本文提出了一个名为FedTOP的联邦迁移有序个性化学习框架，用于解决数据和系统异构性、大规模并行通信资源、恶意攻击和数据污染等问题。论文在两个真实数据集上进行了测试，并比较了三种扩展的性能 - 迁移、有序和个性化。实验结果表明，该框架在两个数据集上的测试客户端分别达到92.32％和95.96％的准确率。与基线相比，在准确率方面提高了462％，并降低了37.46％的通信成本。

    Federated learning (FL) shines through in the internet of things (IoT) with its ability to realize collaborative learning and improve learning efficiency by sharing client model parameters trained on local data. Although FL has been successfully applied to various domains, including driver monitoring applications (DMAs) on the internet of vehicles (IoV), its usages still face some open issues, such as data and system heterogeneity, large-scale parallelism communication resources, malicious attacks, and data poisoning. This paper proposes a federated transfer-ordered-personalized learning (FedTOP) framework to address the above problems and test on two real-world datasets with and without system heterogeneity. The performance of the three extensions, transfer, ordered, and personalized, is compared by an ablation study and achieves 92.32% and 95.96% accuracy on the test clients of two datasets, respectively. Compared to the baseline, there is a 462% improvement in accuracy and a 37.46% 
    
[^128]: 一种改进的损失函数提升顺序推荐模型

    Improving Sequential Recommendation Models with an Enhanced Loss Function. (arXiv:2301.00979v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.00979](http://arxiv.org/abs/2301.00979)

    本文研究了顺序推荐模型常用损失函数的优劣，提出了一种改进的损失函数。实验表明，这种改进的损失函数可以显著提升多种顺序推荐模型的性能。

    

    最近，人们对于顺序推荐模型进行了大量的基准测试和复现/改进现有模型的工作。本文通过分析常用的顺序推荐损失函数的优劣，提出了一种改进的损失函数来充分利用它们的优点。实验结果表明，这种改进的损失函数显著提升了 GRU4Rec，SASRec，SR-GNN和 S3Rec等模型的性能。

    There has been a growing interest in benchmarking sequential recommendation models and reproducing/improving existing models. For example, Rendle et al. improved matrix factorization models by tuning their parameters and hyperparameters. Petrov and Macdonald developed a more efficient and effective implementation of BERT4Rec, which resolved inconsistencies in performance comparison between BERT4Rec and SASRec in previous works. In particular, BERT4Rec and SASRec share a similar network structure, with the main difference lying in their training objective/loss function. Therefore, we analyzed the advantages and disadvantages of commonly used loss functions in sequential recommendation and proposed an improved loss function that leverages their strengths. We conduct extensive experiments on two influential open-source libraries, and the results demonstrate that our improved loss function significantly enhances the performance of GRU4Rec, SASRec, SR-GNN, and S3Rec models, improving their 
    
[^129]: 通过上下文长度探究黑匣子语言模型解释

    Black-box language model explanation by context length probing. (arXiv:2212.14815v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.14815](http://arxiv.org/abs/2212.14815)

    该论文提出了一个模型不可知的新颖解释技术：上下文长度探测，通过跟踪模型预测与可用上下文长度的关系来对不同上下文分配不同的重要性得分。该方法适用于大型预训练语言模型，并有利于研究远距离依赖性。

    

    大型语言模型的广泛采用强调了改善其可解释性的必要性。我们提出了一种新颖的解释技术：上下文长度探测，它基于跟踪模型预测作为可用上下文长度的函数，并允许对不同上下文分配不同的重要性得分。该技术是模型不可知的，不依赖于除计算token级概率之外的模型内部访问。我们将上下文长度探测应用于大型预训练语言模型，并提供了一些初始的分析和见解，包括研究远距离依赖性的潜力。方法的源代码和交互式演示可用。

    The increasingly widespread adoption of large language models has highlighted the need for improving their explainability. We present context length probing, a novel explanation technique for causal language models, based on tracking the predictions of a model as a function of the length of available context, and allowing to assign differential importance scores to different contexts. The technique is model-agnostic and does not rely on access to model internals beyond computing token-level probabilities. We apply context length probing to large pre-trained language models and offer some initial analyses and insights, including the potential for studying long-range dependencies. The source code and an interactive demo of the method are available.
    
[^130]: KNIFE: 从自由文本理由中提取推理知识

    KNIFE: Distilling Reasoning Knowledge From Free-Text Rationales. (arXiv:2212.09721v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09721](http://arxiv.org/abs/2212.09721)

    通过KNIFE，可以从自由文本理由中提取推理知识，进而在小型语言模型中提高推理能力。

    

    语言模型在许多语言推理任务中表现出色，但它们的意外错误引起了对它们的推理能力的怀疑。因此，越来越多的人对微调/提示语言模型感兴趣，这些语言模型包含任务实例和其关联的自由文本理由（FTR），这些理由解释了预测正确任务输出的正确推理过程。然而，现有的微调方法无法提高语言模型的性能，而提示需要过大（即>50B）的语言模型才能良好工作。我们提出了KNIFE，证明从FTR中可以有效地提取推理知识，将其灌输到小型（即<1B）的语言模型中，从而提高语言模型的性能。首先，KNIFE对一个师生语言模型进行微调（给定任务输入和FTR），以预测任务输出，将推理知识从FTR转移至师生隐藏状态。其次，KNIFE对一个学生语言模型进行微调（仅给定任务输入），以使其隐藏状态类似于师生模型的隐藏状态，从而提高学生模型的性能。

    Language models (LMs) have yielded impressive results on many language reasoning tasks, but their unexpected errors raise doubts about their reasoning abilities. In light of this, there is growing interest in finetuning/prompting LMs with both task instances and their associated free-text rationales (FTRs), which explain the correct reasoning process for predicting the correct task output (i.e., how to be "right for the right reasons"). However, existing finetuning methods fail to improve LM performance, while prompting needs prohibitively large (i.e., >50B) LMs to work well. We propose KNIFE, which shows that reasoning knowledge can be effectively distilled from FTRs into a small (i.e., <1B) LM and improve the LM's performance. First, KNIFE finetunes a teacher LM (given task input and FTR) to predict the task output, transferring reasoning knowledge from the FTRs to the teacher's hidden states. Second, KNIFE finetunes a student LM (given task input only) such that its hidden states ar
    
[^131]: 物理约束下的温湿度深度学习后处理

    Physics-constrained deep learning postprocessing of temperature and humidity. (arXiv:2212.04487v2 [physics.ao-ph] UPDATED)

    [http://arxiv.org/abs/2212.04487](http://arxiv.org/abs/2212.04487)

    本研究提出了一种物理约束下的深度学习后处理方法，通过整合气象专业知识以解析方程式的形式，实现物理一致性。研究发现在瑞士地面天气的后处理中，通过强制执行热力学状态方程来约束神经网络可以产生物理上一致的温湿度预测结果。

    

    天气预报中心目前依赖于统计后处理方法来最小化预报误差。这提高了预报技能，但可能导致违反物理原理或忽略变量之间依赖关系的预测，这可能对下游应用和后处理模型的可信度有问题，特别是当它们基于新的机器学习方法时。借鉴物理知识在机器学习中的最新进展，我们建议通过整合气象专业知识以解析方程式的形式，在深度学习的后处理模型中实现物理一致性。应用于瑞士地面天气的后处理中，我们发现通过强制执行热力学状态方程来约束神经网络会产生物理上一致的温湿度预测结果，而不会影响性能。我们的方法在数据稀缺时尤其有优势，我们的研究结果表明，将领域专业知识纳入深度学习后处理模型是可行的。

    Weather forecasting centers currently rely on statistical postprocessing methods to minimize forecast error. This improves skill but can lead to predictions that violate physical principles or disregard dependencies between variables, which can be problematic for downstream applications and for the trustworthiness of postprocessing models, especially when they are based on new machine learning approaches. Building on recent advances in physics-informed machine learning, we propose to achieve physical consistency in deep learning-based postprocessing models by integrating meteorological expertise in the form of analytic equations. Applied to the post-processing of surface weather in Switzerland, we find that constraining a neural network to enforce thermodynamic state equations yields physically-consistent predictions of temperature and humidity without compromising performance. Our approach is especially advantageous when data is scarce, and our findings suggest that incorporating doma
    
[^132]: 约束神经网络：神经DAEs

    Neural DAEs: Constrained neural networks. (arXiv:2211.14302v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.14302](http://arxiv.org/abs/2211.14302)

    本文研究了将辅助代数轨迹信息明确添加到神经网络中的影响，并通过稳定化和投影方法合并信息，对多体摆和分子动力学情景进行了模拟实验。该方法易于实现，对训练性能影响有限，在推理方面给出了显著提升。

    

    本文研究了将辅助代数轨迹信息明确添加到动态系统的神经网络中的影响。我们从微分代数方程和流形上的微分方程领域汲取灵感，并在残差神经网络中实现相关方法，尽管存在一些基本情境上的差异。通过稳定化和投影方法，将约束或辅助信息效果合并，并通过对多体摆和分子动力学情景的模拟实验展示了何时使用哪种方法。我们的一些方法易于在现有代码中实现，并对训练性能影响有限，同时在推理方面给出了显著的提升。

    This article investigates the effect of explicitly adding auxiliary algebraic trajectory information to neural networks for dynamical systems. We draw inspiration from the field of differential-algebraic equations and differential equations on manifolds and implement related methods in residual neural networks, despite some fundamental scenario differences. Constraint or auxiliary information effects are incorporated through stabilization as well as projection methods, and we show when to use which method based on experiments involving simulations of multi-body pendulums and molecular dynamics scenarios. Several of our methods are easy to implement in existing code and have limited impact on training performance while giving significant boosts in terms of inference.
    
[^133]: OpenFE: 具有专家级性能的自动特征生成工具

    OpenFE: Automated Feature Generation with Expert-level Performance. (arXiv:2211.12507v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.12507](http://arxiv.org/abs/2211.12507)

    本文提出的自动特征生成工具OpenFE可以与机器学习专家提供的结果相媲美，具有高效和准确的特点，通过新颖的特征提升方法和两阶段修剪算法实现。

    

    自动特征生成的目标是使机器学习专家摆脱手动特征生成的繁琐任务，这对于提高表格数据的学习性能至关重要。自动特征生成的主要挑战是从大量候选特征中高效准确地识别有效特征。本文提出了OpenFE，一种自动特征生成工具，可以与机器学习专家提供的结果相媲美。OpenFE通过两个组件实现高效和准确：1）一种新颖的特征提升方法，用于准确地评估候选特征的增量性能；2）一种两阶段修剪算法，以粗到细的方式进行特征修剪。在十个基准数据集上的广泛实验表明，OpenFE比现有基线方法表现更好。我们进一步在两个Kaggle比赛中对OpenFE进行了评估，这些比赛有数千个数据科学团队参与。

    The goal of automated feature generation is to liberate machine learning experts from the laborious task of manual feature generation, which is crucial for improving the learning performance of tabular data. The major challenge in automated feature generation is to efficiently and accurately identify effective features from a vast pool of candidate features. In this paper, we present OpenFE, an automated feature generation tool that provides competitive results against machine learning experts. OpenFE achieves high efficiency and accuracy with two components: 1) a novel feature boosting method for accurately evaluating the incremental performance of candidate features and 2) a two-stage pruning algorithm that performs feature pruning in a coarse-to-fine manner. Extensive experiments on ten benchmark datasets show that OpenFE outperforms existing baseline methods by a large margin. We further evaluate OpenFE in two Kaggle competitions with thousands of data science teams participating. 
    
[^134]: 从自适应欺诈者检测探究对抗鲁棒的推荐系统

    Towards Adversarially Robust Recommendation from Adaptive Fraudster Detection. (arXiv:2211.11534v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2211.11534](http://arxiv.org/abs/2211.11534)

    本文提出了一种针对推荐系统的MetaC恶意攻击，并设计了一种自适应欺诈者检测模块PDR，明确考虑标签的不确定性，提高了推荐系统的鲁棒性。

    

    推荐系统在节点注入攻击下的鲁棒性备受关注。最近，提出了基于GNN的推荐系统GraphRfi，它有效减轻了注入的虚假用户的影响。但是，我们展示了GraphRfi仍然容易受到攻击，因为其欺诈者检测组件的监督性质，在实践中很难获得干净的标签。我们提出了一个强大的MetaC恶意攻击，针对GNN-based和MF-based推荐系统。根据我们从易受攻击性分析中得到的见解，我们设计了一种自适应欺诈者检测模块，明确考虑了标签不确定性。该模块可以作为不同推荐系统的插件，形成一个稳健的框架（PDR）。全面的实验表明，我们的防御方法在攻击下优于其他基准方法。总体而言，我们的工作强调了在构建欺诈者检测模块时考虑标签不确定性的重要性，并提供了改善推荐系统对节点注入攻击鲁棒性的实用解决方案。

    The robustness of recommender systems under node injection attacks has garnered significant attention. Recently, GraphRfi, a GNN-based recommender system, was proposed and shown to effectively mitigate the impact of injected fake users. However, we demonstrate that GraphRfi remains vulnerable to attacks due to the supervised nature of its fraudster detection component, where obtaining clean labels is challenging in practice. In particular, we propose a powerful poisoning attack, MetaC, against both GNN-based and MF-based recommender systems. Furthermore, we analyze why GraphRfi fails under such an attack. Then, based on our insights obtained from vulnerability analysis, we design an adaptive fraudster detection module that explicitly considers label uncertainty. This module can serve as a plug-in for different recommender systems, resulting in a robust framework named PDR. Comprehensive experiments show that our defense approach outperforms other benchmark methods under attacks. Overal
    
[^135]: GLUE-X: 从ODD普适性角度评估自然语言理解模型

    GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-distribution Generalization Perspective. (arXiv:2211.08073v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.08073](http://arxiv.org/abs/2211.08073)

    本文提出了第一个创建名为方法的统一基准的尝试，用于评估NLP模型中的OOD鲁棒性，该基准包括13个公开可用的OOD测试数据集，并在21个常用的PLMs上对8个经典NLP任务进行评估。

    

    预训练语言模型（PLMs）通过利用大量的训练数据，已知可以提高自然语言理解模型的泛化性能。然而，许多NLP任务中的ODD普适性问题仍然存在，这限制了这些方法在现实世界中的部署。本文提出了第一个创建名为方法的统一基准的尝试，用于评估NLP模型中的OOD鲁棒性，强调OOD鲁棒性的重要性，并提供如何衡量模型的鲁棒性以及如何改善模型的见解。该基准包括13个公开可用的OOD测试数据集，并在21个常用的PLMs（包括GPT-3和GPT-3.5）上对8个经典NLP任务进行评估。我们的研究结果确认了在所有设置下，与ID准确度相比，存在显着的性能下降，需要改善NLP任务中的OOD准确度。

    Pre-trained language models (PLMs) are known to improve the generalization performance of natural language understanding models by leveraging large amounts of data during the pre-training phase. However, the out-of-distribution (OOD) generalization problem remains a challenge in many NLP tasks, limiting the real-world deployment of these methods. This paper presents the first attempt at creating a unified benchmark named \method for evaluating OOD robustness in NLP models, highlighting the importance of OOD robustness and providing insights on how to measure the robustness of a model and how to improve it. The benchmark includes 13 publicly available datasets for OOD testing, and evaluations are conducted on 8 classic NLP tasks over 21 popularly used PLMs, including GPT-3 and GPT-3.5. Our findings confirm the need for improved OOD accuracy in NLP tasks, as significant performance degradation was observed in all settings compared to in-distribution (ID) accuracy.
    
[^136]: 移动边缘计算环境下联邦学习的最优隐私保护

    Optimal Privacy Preserving for Federated Learning in Mobile Edge Computing. (arXiv:2211.07166v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.07166](http://arxiv.org/abs/2211.07166)

    本文介绍了一种在移动边缘计算环境下最优隐私保护的联邦学习方法，通过联合优化量化和二项机制参数以及通信资源，最大化收敛速度并保证DP要求。

    

    在无线网络上，采用量化和有意添加的噪声的联邦学习是一种有前途的方法，可以保护用户的差分隐私，同时减少无线资源的使用。具体而言，联邦学习过程可以与来自多个用户的基于二项机制的量化更新进行融合。但是，优化量化参数，通信资源（例如传输功率、带宽和量化比特）以及添加的噪声，以保证DP要求和学习的FL模型的性能仍然是一个开放且具有挑战性的问题。本文旨在联合优化量化和二项机制参数以及通信资源，以在无线网络和DP要求的约束下最大化收敛速度。为此，我们首先推导了FL与量化/噪声的DP预算估计，该估计比现有的上界更紧。然后，我们提供了收敛速度的理论界限。

    Federated Learning (FL) with quantization and deliberately added noise over wireless networks is a promising approach to preserve user differential privacy (DP) while reducing wireless resources. Specifically, an FL process can be fused with quantized Binomial mechanism-based updates contributed by multiple users. However, optimizing quantization parameters, communication resources (e.g., transmit power, bandwidth, and quantization bits), and the added noise to guarantee the DP requirement and performance of the learned FL model remains an open and challenging problem. This article aims to jointly optimize the quantization and Binomial mechanism parameters and communication resources to maximize the convergence rate under the constraints of the wireless network and DP requirement. To that end, we first derive a novel DP budget estimation of the FL with quantization/noise that is tighter than the state-of-the-art bound. We then provide a theoretical bound on the convergence rate. This t
    
[^137]: 在线合同设计的样本复杂度

    The Sample Complexity of Online Contract Design. (arXiv:2211.05732v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2211.05732](http://arxiv.org/abs/2211.05732)

    本文解决了在线合同设计中一个悬而未决的问题，证明了指数级的$m$个样本就足以学习一个近乎最优的合同。

    

    本文研究在线情境下的隐藏-行动委托问题。在每轮中，委托人发布一份合同，根据每个结果规定代理人的支付。代理人然后做出一个最大化她自己效用的战略行动选择，但直接观察不到行动。委托人观察结果并从代理人的行动选择中获得效用。根据过去的观察，委托人动态地调整合同，目标是最大化其效用。我们引入了一种在线学习算法，并给出了其Stackelberg遗憾的上界。我们证明，在合同空间为$[0,1]^m$时，Stackelberg遗憾的上界为$\widetilde O(\sqrt{m} \cdot T^{1-1/(2m+1)})$，下界为$\Omega(T^{1-1/(m+2)})$，其中$\widetilde O$排除对数因子。 这个结果表明，指数级的$m$个样本就足以学习一个近乎最优的合同，解决了在线合同设计中的一个悬而未决的问题。

    We study the hidden-action principal-agent problem in an online setting. In each round, the principal posts a contract that specifies the payment to the agent based on each outcome. The agent then makes a strategic choice of action that maximizes her own utility, but the action is not directly observable by the principal. The principal observes the outcome and receives utility from the agent's choice of action. Based on past observations, the principal dynamically adjusts the contracts with the goal of maximizing her utility.  We introduce an online learning algorithm and provide an upper bound on its Stackelberg regret. We show that when the contract space is $[0,1]^m$, the Stackelberg regret is upper bounded by $\widetilde O(\sqrt{m} \cdot T^{1-1/(2m+1)})$, and lower bounded by $\Omega(T^{1-1/(m+2)})$, where $\widetilde O$ omits logarithmic factors. This result shows that exponential-in-$m$ samples are sufficient and necessary to learn a near-optimal contract, resolving an open probl
    
[^138]: 量子强化学习综述

    A Survey on Quantum Reinforcement Learning. (arXiv:2211.03464v1 [quant-ph] CROSS LISTED)

    [http://arxiv.org/abs/2211.03464](http://arxiv.org/abs/2211.03464)

    本文综述了量子强化学习的最新进展和相关文献，重点介绍了基于噪声中等规模量子设备的变分量子电路在经典强化学习中的应用，以及基于未来容错硬件的量子强化学习算法，其中一些具有可证明的量子优势。

    

    量子强化学习是量子计算和机器学习交叉领域中的新兴领域。本文将提供量子强化学习文献的广泛概述，但我们特别强调最近的发展。我们关注的是已经可用的噪声中等规模量子设备，这些设备包括变分量子电路，它在传统的强化学习框架下充当函数逼近器。此外，我们还调查了基于未来容错硬件的量子强化学习算法，其中一些具有可证明的量子优势。我们提供了对该领域的俯瞰以及对文献中部分内容的总结和评论。

    Quantum reinforcement learning is an emerging field at the intersection of quantum computing and machine learning. While we intend to provide a broad overview of the literature on quantum reinforcement learning (our interpretation of this term will be clarified below), we put particular emphasis on recent developments. With a focus on already available noisy intermediate-scale quantum devices, these include variational quantum circuits acting as function approximators in an otherwise classical reinforcement learning setting. In addition, we survey quantum reinforcement learning algorithms based on future fault-tolerant hardware, some of which come with a provable quantum advantage. We provide both a birds-eye-view of the field, as well as summaries and reviews for selected parts of the literature.
    
[^139]: 对象轨迹表示模型的经验贝叶斯分析

    An Empirical Bayes Analysis of Object Trajectory Representation Models. (arXiv:2211.01696v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.01696](http://arxiv.org/abs/2211.01696)

    本论文对对象轨迹表示模型的复杂度和拟合误差之间的权衡进行了经验分析，发现简单的线性模型就能够高度重现真实世界的轨迹，通过使用经验贝叶斯方法可以为轨迹跟踪问题中必要的运动模型提供信息，并可以帮助规范预测模型。

    

    我们对对象轨迹建模中的模型复杂度和拟合误差之间的权衡进行了深入的经验分析。通过分析多个大型公共数据集，我们发现简单的线性模型在相关时间范围内使用较少的模型复杂度就能够高度重现真实世界的轨迹。这一发现允许将轨迹跟踪和预测作为贝叶斯过滤问题进行公式化。我们采用经验贝叶斯方法，从数据中估计模型参数的先验分布，这些先验分布可以为轨迹跟踪问题中必要的运动模型提供信息，并可以帮助规范预测模型。我们主张在轨迹预测任务中使用线性轨迹表示模型，因为它们目前并不会限制预测性能。

    We present an in-depth empirical analysis of the trade-off between model complexity and fit error in modelling object trajectories. Analyzing several large public datasets, we show that simple linear models do represent real-world trajectories with high fidelity over relevant time scales at very moderate model complexity. This finding allows the formulation of trajectory tracking and prediction as a Bayesian filtering problem. Using an Empirical Bayes approach, we estimate prior distributions over model parameters from the data. These prior distributions inform the motion models necessary in the trajectory tracking problem and can help regularize prediction models. We argue for the use of linear trajectory representation models in trajectory prediction tasks as they do not limit prediction performance currently.
    
[^140]: 基于序列的计划可行性预测用于高效的任务和动作规划

    Sequence-Based Plan Feasibility Prediction for Efficient Task and Motion Planning. (arXiv:2211.01576v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2211.01576](http://arxiv.org/abs/2211.01576)

    本文提出一种基于学习的任务和动作规划（TAMP）算法，通过序列预测辅助搜索，显著提高了规划效率，运行时间缩短了80%。

    

    本文提出了一种学习可用的任务和动作规划（TAMP）算法，用于在存在多个关节和可移动障碍物的环境中解决移动操作问题。我们的思路是通过学习的计划可行性预测器对传统TAMP规划器的搜索过程进行偏置。算法的核心是PIGINet，这是一种新颖的基于Transformer的学习方法，它接收任务计划、目标和初始状态，并预测与任务计划相关联的运动轨迹的概率。我们将PIGINet集成到一个TAMP规划器中，该规划器生成一组多样化的高层任务计划，按照预测的可行性排序，并依次进行细化。我们对七种厨房重排问题的TAMP算法运行时间进行评估，将其性能与非学习基准进行比较。实验结果表明，PIGINet显着提高了规划效率，在状态较小的问题上缩短了运行时间80%。

    We present a learning-enabled Task and Motion Planning (TAMP) algorithm for solving mobile manipulation problems in environments with many articulated and movable obstacles. Our idea is to bias the search procedure of a traditional TAMP planner with a learned plan feasibility predictor. The core of our algorithm is PIGINet, a novel Transformer-based learning method that takes in a task plan, the goal, and the initial state, and predicts the probability of finding motion trajectories associated with the task plan. We integrate PIGINet within a TAMP planner that generates a diverse set of high-level task plans, sorts them by their predicted likelihood of feasibility, and refines them in that order. We evaluate the runtime of our TAMP algorithm on seven families of kitchen rearrangement problems, comparing its performance to that of non-learning baselines. Our experiments show that PIGINet substantially improves planning efficiency, cutting down runtime by 80% on problems with small state
    
[^141]: RGMIM: 区域引导的掩膜图像建模用于COVID-19检测。

    RGMIM: Region-Guided Masked Image Modeling for COVID-19 Detection. (arXiv:2211.00313v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.00313](http://arxiv.org/abs/2211.00313)

    本论文提出了一种针对COVID-19检测的新颖区域引导的掩膜图像建模方法，该方法通过利用肺掩模信息来识别有效区域，以学习更有用的COVID-19检测信息。

    

    目的：自监督学习正在快速推进医学领域的计算机辅助诊断。掩膜图像建模（MIM）是一种自监督学习方法，它掩盖了一组输入像素并试图预测遮盖的像素。传统的MIM方法通常采用随机掩膜策略。与普通图像相比，医学图像往往具有用于疾病检测的小区域。因此，我们在本文中专注于解决这个问题，在自动COVID-19识别方面进行评估。方法：本文提出了一种新颖的区域引导的掩膜图像建模方法（RGMIM）用于COVID-19检测。在我们的方法中，我们设计了一种新的掩膜策略，利用肺掩模信息来识别有效区域，以学习更有用的COVID-19检测信息。我们将所提出的方法与五种自监督学习技术（MAE，SKD，Cross，BYOL和SimSiam）进行对比。我们提出了定量评估。

    Purpose: Self-supervised learning is rapidly advancing computer-aided diagnosis in the medical field. Masked image modeling (MIM) is one of the self-supervised learning methods that masks a subset of input pixels and attempts to predict the masked pixels. Traditional MIM methods often employ a random masking strategy. In comparison to ordinary images, medical images often have a small region of interest for disease detection. Consequently, we focus on fixing the problem in this work, which is evaluated by automatic COVID-19 identification. Methods: In this study, we propose a novel region-guided masked image modeling method (RGMIM) for COVID-19 detection in this paper. In our method, we devise a new masking strategy that employed lung mask information to identify valid regions to learn more useful information for COVID-19 detection. The proposed method was contrasted with five self-supervised learning techniques (MAE, SKD, Cross, BYOL, and, SimSiam). We present a quantitative evaluatio
    
[^142]: 可信数据价值评估的方差缩小Shapley值估计

    Variance reduced Shapley value estimation for trustworthy data valuation. (arXiv:2210.16835v5 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.16835](http://arxiv.org/abs/2210.16835)

    本文提出了一种名为VRDS的分层抽样方法，用于评估数据价值，以缩小排列抽样方法的估计方差，并在不同类型的数据集和数据删除应用程序中得到验证。

    

    数据价值评估，特别是在算法预测和决策中量化数据价值，是数据交易场景中的一个基本问题。目前最广泛使用的方法是定义数据Shapley值，然后通过排序抽样算法进行近似计算。为了弥补排列抽样算法的大估计方差，我们提出了一种更稳健的数据估价方法，使用分层抽样，并命名为方差缩小数据Shapley值（VRDS）。我们理论上展示了如何进行分层抽样，每个层抽多少样本，以及VRDS的样本复杂度分析。最后，我们在不同类型的数据集和数据删除应用程序中说明了VRDS的有效性。

    Data valuation, especially quantifying data value in algorithmic prediction and decision-making, is a fundamental problem in data trading scenarios. The most widely used method is to define the data Shapley and approximate it by means of the permutation sampling algorithm. To make up for the large estimation variance of the permutation sampling that hinders the development of the data marketplace, we propose a more robust data valuation method using stratified sampling, named variance reduced data Shapley (VRDS for short). We theoretically show how to stratify, how many samples are taken at each stratum, and the sample complexity analysis of VRDS. Finally, the effectiveness of VRDS is illustrated in different types of datasets and data removal applications.
    
[^143]: 一种使用噪声增强语音作为目标的训练和推理策略，用于无清晰语音的语音增强

    A Training and Inference Strategy Using Noisy and Enhanced Speech as Target for Speech Enhancement without Clean Speech. (arXiv:2210.15368v3 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2210.15368](http://arxiv.org/abs/2210.15368)

    为解决语音增强领域中“无清晰语音”的挑战，提出了使用增强语音作为目标的训练和推理策略，即使在域内和域外噪声差异较大的情况下仍能有效，实验结果优于基线方法。

    

    无清晰语音是发展语音增强系统的实际挑战，这意味着它们的训练准则和评估指标之间存在不可避免的不匹配。为了应对这种不利情况，我们提出了一种训练和推理策略，其中使用增强语音作为目标来改进先前提出的噪声目标训练（NyTT）。由于域内噪声与外部噪声的同质性是NyTT有效性的关键，我们通过混音训练多个学生模型，包括：1）使用教师模型估计的语音和噪声进行增强目标训练，或者2）使用原始的噪声语音和教师模型估计的噪声进行噪声目标训练。实验结果表明，我们提出的方法优于几种基线方法，特别是在教师/学生推理方面，其中预测的清晰语音是通过教师和最终学生模型成功地推导出来的。

    The lack of clean speech is a practical challenge to the development of speech enhancement systems, which means that there is an inevitable mismatch between their training criterion and evaluation metric. In response to this unfavorable situation, we propose a training and inference strategy that additionally uses enhanced speech as a target by improving the previously proposed noisy-target training (NyTT). Because homogeneity between in-domain noise and extraneous noise is the key to the effectiveness of NyTT, we train various student models by remixing 1) the teacher model's estimated speech and noise for enhanced-target training or 2) raw noisy speech and the teacher model's estimated noise for noisy-target training. Experimental results show that our proposed method outperforms several baselines, especially with the teacher/student inference, where predicted clean speech is derived successively through the teacher and final student models.
    
[^144]: DiffusionDB: 文本到图像生成模型的大规模提示画廊数据集

    DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models. (arXiv:2210.14896v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.14896](http://arxiv.org/abs/2210.14896)

    介绍了DiffusionDB数据集，这是一个规模庞大的文本到图像提示数据集，总计包含1400万张图像和180万个唯一提示。该数据集被用来帮助研究人员解决文本提示生成图像时所需的适当提示的问题，并指出了一些特定的提示样式和超参数值可能导致模型错误，甚至生成误导信息。

    

    随着扩散模型的最新进展，用户可以通过编写自然语言提示生成高质量图像。然而，生成具有所需细节的图像需要适当的提示，而且往往不清楚模型对不同提示的反应或最佳提示是什么。为了帮助研究人员解决这些关键挑战，我们介绍了DiffusionDB，这是第一个大规模的文本到图像提示数据集，总计6.5TB，包含使用Stable Diffusion生成的1400万张图像，180万个唯一提示和由真实用户指定的超参数。我们分析了提示的语法和语义特征，并指出了可能导致模型错误的特定超参数值和提示样式，并提供了潜在有害模型使用的证据，如生成误导信息。这个人为驱动的数据集的空前规模和多样性为了解提示和生成图像之间相互作用提供了激动人心的研究机会。

    With recent advancements in diffusion models, users can generate high-quality images by writing text prompts in natural language. However, generating images with desired details requires proper prompts, and it is often unclear how a model reacts to different prompts or what the best prompts are. To help researchers tackle these critical challenges, we introduce DiffusionDB, the first large-scale text-to-image prompt dataset totaling 6.5TB, containing 14 million images generated by Stable Diffusion, 1.8 million unique prompts, and hyperparameters specified by real users. We analyze the syntactic and semantic characteristics of prompts. We pinpoint specific hyperparameter values and prompt styles that can lead to model errors and present evidence of potentially harmful model usage, such as the generation of misinformation. The unprecedented scale and diversity of this human-actuated dataset provide exciting research opportunities in understanding the interplay between prompts and generat
    
[^145]: 破碎的神经缩放定律

    Broken Neural Scaling Laws. (arXiv:2210.14891v7 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.14891](http://arxiv.org/abs/2210.14891)

    本文提出了一个平滑破碎的幂律函数形式，可以准确地模拟和外推深度神经网络的缩放行为，适用于各种架构和大量不同任务，包括视觉、语言、音频、视频、生成建模、对比学习、机器人、不确定性估计/校准、对抗鲁棒性、分子、计算机编程/编码、数学单词问题、算术、无监督/自监督学习和强化学习。

    This paper proposes a smoothly broken power law functional form (referred to as a Broken Neural Scaling Law (BNSL)) that accurately models and extrapolates the scaling behaviors of deep neural networks for various architectures and a large and diverse set of tasks, including vision, language, audio, video, generative modeling, contrastive learning, robotics, uncertainty estimation/calibration, adversarial robustness, molecules, computer programming/coding, math word problems, arithmetic, unsupervised/self-supervised learning, and reinforcement learning.

    我们提出了一个平滑破碎的幂律函数形式（我们称之为破碎的神经缩放定律（BNSL）），它准确地模拟和外推了深度神经网络的缩放行为（即感兴趣的评估指标随用于训练的计算量、模型参数数量、训练数据集大小或上游性能变化而变化）对于各种架构和大量不同任务中的每个任务，包括大规模视觉、语言、音频、视频、扩散、生成建模、多模态学习、对比学习、AI对齐、机器人、超出分布（OOD）泛化、持续学习、不确定性估计/校准、超出分布检测、对抗鲁棒性、蒸馏、分子、计算机编程/编码、数学单词问题、算术、无监督/自监督学习和强化学习。

    We present a smoothly broken power law functional form (referred to by us as a Broken Neural Scaling Law (BNSL)) that accurately models and extrapolates the scaling behaviors of deep neural networks (i.e. how the evaluation metric of interest varies as the amount of compute used for training, number of model parameters, training dataset size, or upstream performance varies) for various architectures and for each of various tasks within a large and diverse set of upstream and downstream tasks, in zero-shot, prompted, and fine-tuned settings. This set includes large-scale vision, language, audio, video, diffusion, generative modeling, multimodal learning, contrastive learning, AI alignment, robotics, out-of-distribution (OOD) generalization, continual learning, uncertainty estimation / calibration, out-of-distribution detection, adversarial robustness, distillation, molecules, computer programming/coding, math word problems, arithmetic, unsupervised/self-supervised learning, and reinforc
    
[^146]: Layer-Neighbor Sampling -- GNN中缓解邻居爆炸问题的采样算法

    Layer-Neighbor Sampling -- Defusing Neighborhood Explosion in GNNs. (arXiv:2210.13339v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13339](http://arxiv.org/abs/2210.13339)

    本文提出了一种新的采样算法，名为LABOR，旨在替代现有的Neighbor Sampling算法。相比之下，它能够采样更少的顶点但不会牺牲质量，并且在相同的顶点采样预算约束下，收敛更快，可以使用更大的批处理大小。

    

    近年来，图神经网络（GNN）受到了广泛关注，但在大规模训练方面仍存在挑战。采用小批量训练和采样可用于缓解此问题。然而，现有方法要么受到邻域爆炸现象的影响，要么性能较差。为了解决这些问题，我们提出了一种新的采样算法LAyer-neighBOR sampling（LABOR）。它被设计成Neighbor Sampling（NS）的直接替代品，具有相同的扩展因子超参数，同时采样的顶点数最多少7倍，不会牺牲质量。通过设计，每个顶点估计器的方差与单个顶点上的NS相匹配。此外，在相同的顶点采样预算约束下，LABOR比现有的层采样方法收敛更快，并且可以使用比NS大112倍的批处理大小。

    Graph Neural Networks (GNNs) have received significant attention recently, but training them at a large scale remains a challenge. Mini-batch training coupled with sampling is used to alleviate this challenge. However, existing approaches either suffer from the neighborhood explosion phenomenon or have poor performance. To address these issues, we propose a new sampling algorithm called LAyer-neighBOR sampling (LABOR). It is designed to be a direct replacement for Neighbor Sampling (NS) with the same fanout hyperparameter while sampling up to 7 times fewer vertices, without sacrificing quality. By design, the variance of the estimator of each vertex matches NS from the point of view of a single vertex. Moreover, under the same vertex sampling budget constraints, LABOR converges faster than existing layer sampling approaches and can use up to 112 times larger batch sizes compared to NS.
    
[^147]: 无视规划地推广横跨变量的强化学习

    Horizon-Free and Variance-Dependent Reinforcement Learning for Latent Markov Decision Processes. (arXiv:2210.11604v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.11604](http://arxiv.org/abs/2210.11604)

    本文研究了具有上下文后见性的 LMDP 强化学习遗憾最小化问题。通过设计一个新颖的模型基础算法框架，我们证明了一个与计划视野对数相关的 $\widetilde{O}\left(\sqrt{M \Gamma S A K}\right)$ 遗憾度上限，并对 alpha 向量的总方差进行分析。同时，我们提出了一个 $\Omega\left(\sqrt{M S A K}\right)$ 的遗憾度下限，它在 $\Gamma=2$ 时证明了我们的上界是最优的。

    

    本文研究了具有后见性上下文的潜在马尔可夫决策过程 (LMDPs) 强化学习 (RL) 的遗憾最小化问题。我们设计了一个新颖的基于模型的算法框架，可以通过模型乐观或值乐观求解器实例化。我们证明了一个关于遗憾度的较小量级为 $\widetilde{O}\left(\sqrt{M \Gamma S A K}\right)$ 的界限，其中 $M$ 是上下文数量，$S$ 是状态数量，$A$ 是动作数量，$K$ 是回合数量，而 $\Gamma \le S$ 是任何状态-动作对的最大转移次数。遗憾度只在规划视野中以对数形式缩放，所以 LMDP 的规划视野的第一个(几乎)无视界限就被产生了。我们的论证的关键是对 alpha 向量的总方差进行分析，该方差通过递归技术进行了仔细的限制。我们通过一个新的 $\Omega\left(\sqrt{M S A K}\right)$ 遗憾性下限补充了我们的正补结果，并证明了当 $\Gamma=2$ 时，我们的上界是极小化最优的。

    We study regret minimization for reinforcement learning (RL) in Latent Markov Decision Processes (LMDPs) with context in hindsight. We design a novel model-based algorithmic framework which can be instantiated with both a model-optimistic and a value-optimistic solver. We prove an $\widetilde{O}\left(\sqrt{M \Gamma S A K}\right)$ regret bound where $M$ is the number of contexts, $S$ is the number of states, $A$ is the number of actions, $K$ is the number of episodes, and $\Gamma \le S$ is the maximum transition degree of any state-action pair. The regret bound only scales logarithmically with the planning horizon, thus yielding the first (nearly) horizon-free regret bound for LMDP. Key in our proof is an analysis of the total variance of alpha vectors, which is carefully bounded by a recursion-based technique. We complement our positive result with a novel $\Omega\left(\sqrt{M S A K}\right)$ regret lower bound with $\Gamma = 2$, which shows our upper bound minimax optimal when $\Gamma$
    
[^148]: DICTDIS：基于词典约束的神经机器翻译消歧方法对 NMT 的改进

    DICTDIS: Dictionary Constrained Disambiguation for Improved NMT. (arXiv:2210.06996v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.06996](http://arxiv.org/abs/2210.06996)

    DICTDIS是一种新颖有词典约束的NMT系统，其利用多个字典候选项进行训练，实现了从多义词中消除翻译歧义的目的，提高了翻译质量。

    

    领域特定的神经机器翻译系统（例如教育应用程序）在多语言社会中帮助使信息对一组多样化的用户可访问是具有社会意义的。这种 NMT 系统应该具有词汇约束并从领域特定的词典中汲取。由于单词的多义性，词典中可能会为源单词或短语呈现多个候选翻译。这时，NMT 模型需要选择与语境最相关的候选翻译。先前的工作主要忽略了这个问题，而侧重于单个候选约束设置，其中目标词或短语被单个约束替换。在本文中，我们提出了一种名为DICTDIS的词典约束 NMT 系统，该系统消除了从字典中得出的多个候选翻译的歧义。我们通过将训练数据与多个字典候选项进行增量来实现这一点，从而在训练期间积极鼓励消除歧义。

    Domain-specific neural machine translation (NMT) systems (\eg, in educational applications) are socially significant with the potential to help make information accessible to a diverse set of users in multilingual societies. It is desirable that such NMT systems be lexically constrained and draw from domain-specific dictionaries. Dictionaries could present multiple candidate translations for a source word/phrase due to the polysemous nature of words. The onus is then on the NMT model to choose the contextually most appropriate candidate. Prior work has largely ignored this problem and focused on the single candidate constraint setting wherein the target word or phrase is replaced by a single constraint. In this work we present \dictdis, a lexically constrained NMT system that disambiguates between multiple candidate translations derived from dictionaries. We achieve this by augmenting training data with multiple dictionary candidates to actively encourage disambiguation during training
    
[^149]: 低秩奖励下的多用户强化学习

    Multi-User Reinforcement Learning with Low Rank Rewards. (arXiv:2210.05355v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.05355](http://arxiv.org/abs/2210.05355)

    本文提出了一个新的多用户强化学习算法，可以在用户奖励矩阵具有低秩结构的情况下显著降低样本复杂度。

    

    本文考虑了协作多用户强化学习问题。在此设置中，有多个用户具有相同的状态-动作空间和转移概率，但具有不同的奖励。在假设N个用户的奖励矩阵具有低秩结构的情况下，我们能否设计具有显着较低样本复杂度的算法，与为每个用户分别学习MDP的算法相比。我们的主要贡献是一种算法，它与N个用户特定的MDP一起探索奖励，并可以在两个关键设置中高效地学习奖励：表格MDP和线性MDP。当N很大且秩是常数时，每个MDP的样本复杂度对状态空间大小取对数，这代表了在状态空间大小上的指数降低（与标准的“非协作”相比）。

    In this work, we consider the problem of collaborative multi-user reinforcement learning. In this setting there are multiple users with the same state-action space and transition probabilities but with different rewards. Under the assumption that the reward matrix of the $N$ users has a low-rank structure -- a standard and practically successful assumption in the offline collaborative filtering setting -- the question is can we design algorithms with significantly lower sample complexity compared to the ones that learn the MDP individually for each user. Our main contribution is an algorithm which explores rewards collaboratively with $N$ user-specific MDPs and can learn rewards efficiently in two key settings: tabular MDPs and linear MDPs. When $N$ is large and the rank is constant, the sample complexity per MDP depends logarithmically over the size of the state-space, which represents an exponential reduction (in the state-space size) when compared to the standard ``non-collaborative
    
[^150]: Multi-CLS BERT：一种高效的传统组合方法替代方案

    Multi-CLS BERT: An Efficient Alternative to Traditional Ensembling. (arXiv:2210.05043v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.05043](http://arxiv.org/abs/2210.05043)

    Multi-CLS BERT是一种高效的BERT模型集成方法，通过使用多个CLS标记并鼓励它们多样性，有效地提高了整体准确性和置信度估计。

    

    在CLS-based预测任务中，组合BERT模型通常可以显著提高准确性，但需要更多的计算和内存。本文提出了一种新的集成方法，Multi-CLS BERT，它使用多个CLS标记和一个鼓励多样性的参数化和目标，几乎与单个BERT模型一样高效。实验结果表明，Multi-CLS BERT可靠地提高了整体准确性和置信度估计。

    Ensembling BERT models often significantly improves accuracy, but at the cost of significantly more computation and memory footprint. In this work, we propose Multi-CLS BERT, a novel ensembling method for CLS-based prediction tasks that is almost as efficient as a single BERT model. Multi-CLS BERT uses multiple CLS tokens with a parameterization and objective that encourages their diversity. Thus instead of fine-tuning each BERT model in an ensemble (and running them all at test time), we need only fine-tune our single Multi-CLS BERT model (and run the one model at test time, ensembling just the multiple final CLS embeddings). To test its effectiveness, we build Multi-CLS BERT on top of a state-of-the-art pretraining method for BERT (Aroca-Ouellette and Rudzicz, 2020). In experiments on GLUE and SuperGLUE we show that our Multi-CLS BERT reliably improves both overall accuracy and confidence estimation. When only 100 training samples are available in GLUE, the Multi-CLS BERT_Base model 
    
[^151]: FaDIn: 针对具有一般参数核的Hawkes过程的快速离散化推断

    FaDIn: Fast Discretized Inference for Hawkes Processes with General Parametric Kernels. (arXiv:2210.04635v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.04635](http://arxiv.org/abs/2210.04635)

    本论文提出了一种使用具有有限支持的一般参数核进行TPP推理的高效解决方案，该方法采用了离散化方法，并通过多项实验证明了该方法的统计和计算效率。

    

    时间点过程是建模事件数据的自然工具。在所有的时间点过程模型中，Hawkes过程被证明是最广泛使用的，主要是由于它们对于各种应用的适当建模，特别是在考虑指数或非参数核时。尽管非参数核是一种选择，但这些模型需要大型数据集。而指数核更具数据效率，对于立即触发更多事件的特定应用更有效，但对于需要估计延迟的应用（如神经科学），它们不太适用。本研究旨在提供一种使用具有有限支持的一般参数核进行TPP推理的高效解决方案。所开发的解决方案包括利用事件的离散化的快速$\ell_2$梯度求解器。在理论上支持离散化的使用后，通过多种实验，证明了该新方法的统计和计算效率。

    Temporal point processes (TPP) are a natural tool for modeling event-based data. Among all TPP models, Hawkes processes have proven to be the most widely used, mainly due to their adequate modeling for various applications, particularly when considering exponential or non-parametric kernels. Although non-parametric kernels are an option, such models require large datasets. While exponential kernels are more data efficient and relevant for specific applications where events immediately trigger more events, they are ill-suited for applications where latencies need to be estimated, such as in neuroscience. This work aims to offer an efficient solution to TPP inference using general parametric kernels with finite support. The developed solution consists of a fast $\ell_2$ gradient-based solver leveraging a discretized version of the events. After theoretically supporting the use of discretization, the statistical and computational efficiency of the novel approach is demonstrated through va
    
[^152]: GNM: 通用导航模型驱动任何机器人

    GNM: A General Navigation Model to Drive Any Robot. (arXiv:2210.03370v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2210.03370](http://arxiv.org/abs/2210.03370)

    通用目标条件模型GNM可以通过多样化的机器人数据实现更强大、更健壮的导航性能，并能驱动任何具有适当视觉感知输入的机器人。

    

    学习是视觉导航的强有力工具，但基于学习的策略的能力受到有限训练数据的限制。本文研究如何在来自多个不同但结构相似的机器人的数据基础上训练面向视觉导航的通用目标条件模型，并实现在各种环境和机身上的广泛泛化。我们分析了有效的机器人间数据共享的必要设计决策，包括使用时间上下文和标准化的动作空间，并证明了从异构数据集训练的全局策略优于任何单一数据集上训练的策略。我们收集了6个不同机器人的60小时导航轨迹，并在一系列新机器人上部署训练后的GNM，包括一个欠驱动的四旋翼飞行器。我们发现，训练多样化的数据可以提高和更加稳健的导航性能，而GNM可以驱动任何具有适当视觉感知输入的机器人。

    Learning provides a powerful tool for vision-based navigation, but the capabilities of learning-based policies are constrained by limited training data. If we could combine data from all available sources, including multiple kinds of robots, we could train more powerful navigation models. In this paper, we study how a general goal-conditioned model for vision-based navigation can be trained on data obtained from many distinct but structurally similar robots, and enable broad generalization across environments and embodiments. We analyze the necessary design decisions for effective data sharing across robots, including the use of temporal context and standardized action spaces, and demonstrate that an omnipolicy trained from heterogeneous datasets outperforms policies trained on any single dataset. We curate 60 hours of navigation trajectories from 6 distinct robots, and deploy the trained GNM on a range of new robots, including an underactuated quadrotor. We find that training on diver
    
[^153]: 分层对抗逆强化学习

    Hierarchical Adversarial Inverse Reinforcement Learning. (arXiv:2210.01969v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.01969](http://arxiv.org/abs/2210.01969)

    本文提出了一种分层对抗逆强化学习算法，能够在复杂任务中学习到具有层次结构的最优策略，比现有的方法更加有效。

    

    模仿学习（IL）一般用于从演示中恢复专家策略。然而，对于高度复杂的、长时程任务，恢复单一整体策略是困难的，而专家策略通常包含子任务层次结构。因此，研究者开发了分层模仿学习（HIL）方法，通过在选项框架中显式地建模任务中的活动结构来学习分层策略。现有的HIL方法要么忽视了子任务结构与学习策略之间的因果关系，要么无法同时在分层框架中学习高级别和低级别策略，导致亚最优。本文提出了一种新的HIL算法——分层对抗逆强化学习（H-AIRL），它在最新的IL算法AIRL上扩展了一步选项框架，重新定义了AIRL目标。

    Imitation Learning (IL) has been proposed to recover the expert policy from demonstrations. However, it would be difficult to learn a single monolithic policy for highly-complex long-horizon tasks of which the expert policy usually contains subtask hierarchies. Therefore, Hierarchical Imitation Learning (HIL) has been developed to learn a hierarchical policy from expert demonstrations through explicitly modelling the activity structure in a task with the option framework. Existing HIL methods either overlook the causal relationship between the subtask structure and the learned policy, or fail to learn the high-level and low-level policy in the hierarchical framework in conjuncture, which leads to suboptimality. In this work, we propose a novel HIL algorithm -Hierarchical Adversarial Inverse Reinforcement Learning (H-AIRL), which extends a state-of-the-art (SOTA) IL algorithm -- AIRL, with the one-step option framework. Specifically, we redefine the AIRL objectives on the extended sta
    
[^154]: 高效量子不可知不当决策树学习

    Efficient Quantum Agnostic Improper Learning of Decision Trees. (arXiv:2210.00212v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2210.00212](http://arxiv.org/abs/2210.00212)

    本文提出了第一个无需成员查询在多项式时间内学习大小为$t$的决策树的算法，并成功量化了Kalai和Kanade的不知性增强算法，得到了第一个高效的量子不知性增强算法。

    

    不知性设置是最类似于学习对抗噪声的PAC模型的最困难的泛化。在本文中，我们提出了一个Poly$(n,t,{\frac{1}{\varepsilon}})$量子算法，用于在不知性设置中无需成员查询即可学习大小为$t$的决策树，并且实例间具有均匀边际。我们的算法是第一个在多项式时间内学习决策树的算法（经典或量子），且无需成员查询。我们展示了如何通过设计量子版本的Goldreich-Levin算法，使用高度偏置的函数预言机来构建量子不知性弱学习器。我们展示了如何量化Kalai和Kanade（NIPS 2009）的不知性增强算法，以获得第一个高效的量子不知性增强算法。我们的量子增强算法在适应性量子增强算法中，所有弱学习器偏差依赖性方面都具有多项式改进，同时保留了在$V$中的标准加速度。

    The agnostic setting is the hardest generalization of the PAC model since it is akin to learning with adversarial noise. In this paper, we give a poly$(n,t,{\frac{1}{\varepsilon}})$ quantum algorithm for learning size $t$ decision trees with uniform marginal over instances, in the agnostic setting, without membership queries. Our algorithm is the first algorithm (classical or quantum) for learning decision trees in polynomial time without membership queries. We show how to construct a quantum agnostic weak learner by designing a quantum version of the classical Goldreich-Levin algorithm that works with strongly biased function oracles. We show how to quantize the agnostic boosting algorithm by Kalai and Kanade (NIPS 2009) to obtain the first efficient quantum agnostic boosting algorithm. Our quantum boosting algorithm has a polynomial improvement in the dependence of the bias of the weak learner over all adaptive quantum boosting algorithms while retaining the standard speedup in the V
    
[^155]: 关于量子隧穿行走在非凸优化中的加速效应

    On Quantum Speedups for Nonconvex Optimization via Quantum Tunneling Walks. (arXiv:2209.14501v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2209.14501](http://arxiv.org/abs/2209.14501)

    本文探究了使用量子隧穿行走（QTW）在非凸优化问题中产生的量子加速效应。当不同的局部极小值由高但薄的势垒分隔而极小值是平坦的时候，QTW对比经典随机梯度下降（SGD）实现了加速。

    

    经常情况下，经典算法不能有效解决局部极小值由高能壁隔开的非凸优化问题。本文探讨了通过利用量子隧穿的全局影响可能产生的非凸优化量子加速效应。具体而言，我们介绍了一种称为量子隧穿行走（QTW）的量子算法，并将其应用于局部极小值大约是全局极小值的非凸问题上。我们发现，在不同局部极小值之间的障碍很高但很薄，同时极小值是平坦的情况下，QTW相对于经典随机梯度下降（SGD）实现了量子加速。基于这一观察，我们构建了一个特定的双阱景观，在已知井陷的情况下，经典算法无法有效地攻击一个目标井陷，但是在给定合适的初始状态附近，QTW可以。最后，我们用数值实验验证了我们的发现。

    Classical algorithms are often not effective for solving nonconvex optimization problems where local minima are separated by high barriers. In this paper, we explore possible quantum speedups for nonconvex optimization by leveraging the global effect of quantum tunneling. Specifically, we introduce a quantum algorithm termed the quantum tunneling walk (QTW) and apply it to nonconvex problems where local minima are approximately global minima. We show that QTW achieves quantum speedup over classical stochastic gradient descents (SGD) when the barriers between different local minima are high but thin and the minima are flat. Based on this observation, we construct a specific double-well landscape, where classical algorithms cannot efficiently hit one target well knowing the other well but QTW can when given proper initial states near the known well. Finally, we corroborate our findings with numerical experiments.
    
[^156]: L2XGNN：学习解释图神经网络

    L2XGNN: Learning to Explain Graph Neural Networks. (arXiv:2209.14402v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.14402](http://arxiv.org/abs/2209.14402)

    L2XGNN提出了一个框架来解释图神经网络，通过选择解释子图（模体）实现忠实的解释。该框架能够识别负责预测图属性的模体，并实现与基线方法相同的分类精度。

    

    图神经网络（GNNs）是一类流行的机器学习模型。在学习解释（L2X）范式的启发下，我们提出了L2XGNN，这是一个可解释的GNN框架，通过设计提供忠实的解释。L2XGNN学习一种机制，用于选择解释子图（模体），这些子图仅用于GNN的信息传递操作中。对模体施加这样的限制通常会导致更易解释和更有效的解释。在几种数据集上的实验表明，L2XGNN实现了与基线方法相同的分类精度，同时确保仅使用提供的解释来进行预测。此外，我们还表明L2XGNN能够识别负责预测图属性的模体。

    Graph Neural Networks (GNNs) are a popular class of machine learning models. Inspired by the learning to explain (L2X) paradigm, we propose L2XGNN, a framework for explainable GNNs which provides faithful explanations by design. L2XGNN learns a mechanism for selecting explanatory subgraphs (motifs) which are exclusively used in the GNNs message-passing operations. L2XGNN is able to select, for each input graph, a subgraph with specific properties such as being sparse and connected. Imposing such constraints on the motifs often leads to more interpretable and effective explanations. Experiments on several datasets suggest that L2XGNN achieves the same classification accuracy as baseline methods using the entire input graph while ensuring that only the provided explanations are used to make predictions. Moreover, we show that L2XGNN is able to identify motifs responsible for the graph's properties it is intended to predict.
    
[^157]: 量化优于选择：使用主动动态偏好进行稳健强化学习

    Quantification before Selection: Active Dynamics Preference for Robust Reinforcement Learning. (arXiv:2209.11596v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.11596](http://arxiv.org/abs/2209.11596)

    本论文提出主动动态偏好(ADP)方法，在稳健强化学习中使用熵量化动态偏好来活跃地平衡策略的探索和利用，有效避免策略的过于保守或过于乐观，提高了策略在各种目标域中的鲁棒性，超过现有的state-of-the-art方法。

    

    在现实世界中部署策略或处理不同动态系统中的未知动态失配，对于训练鲁棒性策略至关重要。领域随机化(DR)是一种简单而优雅的方法，它训练一个保守的策略来抵消不同的动态系统，而不需要关于目标系统参数的专家知识。然而，现有的研究表明，通过DR训练的策略往往过于保守，在目标域中表现不佳。我们的关键洞察是，具有不同参数的动态系统为策略提供了不同程度的难度，而在系统中表现良好的难度由于策略的演变而不断变化。如果我们能够在运行过程中积极地采样适合策略难度的系统，就可以稳定训练过程，防止策略过于保守或过于乐观。为了落实这个想法，我们引入了主动动态偏好~(ADP)，通过熵量化动态偏好并主动选择策略与之交互的动态系统，在探索和利用之间平衡。我们证明了ADP显著提高了策略在各种目标域中的鲁棒性，在几个基准任务上优于现有的状态-of-the-art方法。

    Training a robust policy is critical for policy deployment in real-world systems or dealing with unknown dynamics mismatch in different dynamic systems. Domain Randomization~(DR) is a simple and elegant approach that trains a conservative policy to counter different dynamic systems without expert knowledge about the target system parameters. However, existing works reveal that the policy trained through DR tends to be over-conservative and performs poorly in target domains. Our key insight is that dynamic systems with different parameters provide different levels of difficulty for the policy, and the difficulty of behaving well in a system is constantly changing due to the evolution of the policy. If we can actively sample the systems with proper difficulty for the policy on the fly, it will stabilize the training process and prevent the policy from becoming over-conservative or over-optimistic. To operationalize this idea, we introduce Active Dynamics Preference~(ADP), which quantifie
    
[^158]: 大语言模型中用心理学启发的思维链触发识别隐含变量和推理关系进行隐喻理解

    Psychologically-informed chain-of-thought prompts for metaphor understanding in large language models. (arXiv:2209.08141v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2209.08141](http://arxiv.org/abs/2209.08141)

    本文介绍了一种在大语言模型中使用心理学引导思维的方法来增强隐喻理解的性能。这种方法使用了隐含变量和关系来选择正确的释义。

    

    语言理解的概率模型是研究人们语言使用的有价值的工具，但它们需要手动设计。相比之下，大语言模型（LLMs）是用跨领域的文本进行训练的，但它们缺乏概率模型的结构和可解释性。本文采用思维链触发方式来将概率模型中的结构引入LLMs中，以隐喻理解为例来探究这一方法。我们的思维链触发方式导致语言模型推断隐含变量，并思考它们之间的关系，以选择适当的隐喻释义。所选择的隐含变量和关系都基于认知心理学中的隐喻理解理论。我们将这些提示应用于GPT-3的两个最大版本，并显示它们可以提高释义选择任务的性能。

    Probabilistic models of language understanding are valuable tools for investigating human language use. However, they need to be hand-designed for a particular domain. In contrast, large language models (LLMs) are trained on text that spans a wide array of domains, but they lack the structure and interpretability of probabilistic models. In this paper, we use chain-of-thought prompts to introduce structures from probabilistic models into LLMs. We explore this approach in the case of metaphor understanding. Our chain-of-thought prompts lead language models to infer latent variables and reason about their relationships in order to choose appropriate paraphrases for metaphors. The latent variables and relationships chosen are informed by theories of metaphor understanding from cognitive psychology. We apply these prompts to the two largest versions of GPT-3 and show that they can improve performance in a paraphrase selection task.
    
[^159]: 通过神经网络检测广义线性模型的交互变量

    Detection of Interacting Variables for Generalized Linear Models via Neural Networks. (arXiv:2209.08030v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2209.08030](http://arxiv.org/abs/2209.08030)

    本文提出了一种使用神经网络和模型特定交互检测方法来自动化寻找GLMs中应添加的交互作用以提高其预测能力的方法。

    

    广义线性模型（GLMs）是保险公司经常使用的建模方法，其质量取决于交互变量的选择。寻找交互作用对于具有许多变量的数据集来说非常耗时，很大程度上依赖于精算师的专业判断，通常依赖于视觉性能指标。因此，本文提出了一种自动化寻找GLMs中应添加的交互作用以提高其预测能力的方法。我们的方法依赖于神经网络和模型特定的交互检测方法，这比传统方法（如Friedman H统计量或SHAP值）要快。在数字研究中，我们提供了我们的方法在人工生成数据以及开源数据上的结果。

    The quality of generalized linear models (GLMs), frequently used by insurance companies, depends on the choice of interacting variables. The search for interactions is time-consuming, especially for data sets with a large number of variables, depends much on expert judgement of actuaries, and often relies on visual performance indicators. Therefore, we present an approach to automating the process of finding interactions that should be added to GLMs to improve their predictive power. Our approach relies on neural networks and a model-specific interaction detection method, which is computationally faster than the traditionally used methods like Friedman H-Statistic or SHAP values. In numerical studies, we provide the results of our approach on artificially generated data as well as open-source data.
    
[^160]: 基于物理学图神经网络的无开销阻塞检测和预编码：LIDAR 数据遇上光线追踪

    Overhead-Free Blockage Detection and Precoding Through Physics-Based Graph Neural Networks: LIDAR Data Meets Ray Tracing. (arXiv:2209.07350v2 [cs.IT] UPDATED)

    [http://arxiv.org/abs/2209.07350](http://arxiv.org/abs/2209.07350)

    本文提出了一种基于物理学图神经网络的无开销阻塞检测和预编码方法，使用激光雷达数据进行分类，通过光线追踪得到信道估计，经过逐步训练和设计预编码器，达到了较高的性能。

    

    本文针对多输入多输出（MIMO）链路，解决了无需通信开销的阻塞检测和预编码设计问题。阻塞检测通过基于物理的图神经网络（GNN）对激光雷达（LIDAR）数据进行分类实现。对于预编码设计，我们通过对 LIDAR 数据产生的 3D 表面进行光线追踪，得到一个初步的信道估计，再进行逐步训练和设计预编码器。数值模拟表明，阻塞检测成功率达到 95%。我们的数字预编码达到了 90% 的容量，而模拟预编码性能优于以往利用 LIDAR 进行预编码设计的工作。

    In this letter, we address blockage detection and precoder design for multiple-input multiple-output (MIMO) links, without communication overhead required. Blockage detection is achieved by classifying light detection and ranging (LIDAR) data through a physics-based graph neural network (GNN). For precoder design, a preliminary channel estimate is obtained by running ray tracing on a 3D surface obtained from LIDAR data. This estimate is successively refined and the precoder is designed accordingly. Numerical simulations show that blockage detection is successful with 95% accuracy. Our digital precoding achieves 90% of the capacity and analog precoding outperforms previous works exploiting LIDAR for precoder design.
    
[^161]: 关于离线策略强化学习中重复使用偏见的研究

    On the Reuse Bias in Off-Policy Reinforcement Learning. (arXiv:2209.07074v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.07074](http://arxiv.org/abs/2209.07074)

    本文揭示了离线强化学习中一个新的偏见问题：重复使用偏见，提出了一种简单有效的方法——重复使用感知重要性加权（RAW）来解决这个问题，并证明RAW显著提高了离线方法的样本效率和鲁棒性。

    

    重要性采样是离线评估中常用的技术，它通过重新加权回放缓冲区中的轨迹收益来提高样本效率。然而，使用重要性采样训练可能不稳定，并且以前解决这个问题的尝试主要集中在分析重要性采样的方差上。本文揭示了这种不稳定性也与一个新的重复使用偏见有关——由回放缓冲区的评估和优化重复使用造成的离线评估中的偏差。我们在理论上展示了当前策略的离线评估和优化与回放缓冲区的数据导致目标的过高估计，这可能导致错误的梯度更新并退化性能。我们进一步提供一个重复使用偏见的高概率上限，并展示通过引入离线算法的稳定性概念，可以通过控制上限的某一项来控制重复使用偏差。基于这些分析，我们提出了一个简单而有效的方法，称为重复使用感知重要性加权（RAW），来纠正重复使用偏见并提高离线策略强化学习的稳定性。我们还提供了实证证据来证明，RAW可以显着提高离线方法的样本效率和鲁棒性，包括DDPG、SAC和TD3。

    Importance sampling (IS) is a popular technique in off-policy evaluation, which re-weights the return of trajectories in the replay buffer to boost sample efficiency. However, training with IS can be unstable and previous attempts to address this issue mainly focus on analyzing the variance of IS. In this paper, we reveal that the instability is also related to a new notion of Reuse Bias of IS -- the bias in off-policy evaluation caused by the reuse of the replay buffer for evaluation and optimization. We theoretically show that the off-policy evaluation and optimization of the current policy with the data from the replay buffer result in an overestimation of the objective, which may cause an erroneous gradient update and degenerate the performance. We further provide a high-probability upper bound of the Reuse Bias, and show that controlling one term of the upper bound can control the Reuse Bias by introducing the concept of stability for off-policy algorithms. Based on these analyses
    
[^162]: 深度学习的莫里-茨旺齐格表述

    The Mori-Zwanzig formulation of deep learning. (arXiv:2209.05544v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.05544](http://arxiv.org/abs/2209.05544)

    本文提出了基于莫里-茨旺齐格形式主义的深度学习新表述，引入了神经网络记忆的新概念，并通过线性算子方程直接向前和向后传播感兴趣的量。收缩映射理论被用来开发记忆衰减随网络层数增加的充分条件。

    

    本文基于不可逆统计力学的莫里-茨旺齐格（MZ）形式主义，提出了深度学习的新表述。这种新的表述基于深度神经网络和离散动力系统之间的对偶关系，通过线性算子方程直接向前和向后传播感兴趣的量（条件期望和概率密度函数）。这些新方程可以作为开发深度神经网络新的有效参数化的起点，并提供了一种新的通过算子理论方法研究深度学习的框架。所提出的MZ形式主义自然引入了神经网络记忆的新概念，在低维建模和参数化中起着 fundamental 的作用。通过使用收缩映射理论，我们开发出了记忆衰减随网络层数增加的充分条件。

    We develop a new formulation of deep learning based on the Mori-Zwanzig (MZ) formalism of irreversible statistical mechanics. The new formulation is built upon the well-known duality between deep neural networks and discrete dynamical systems, and it allows us to directly propagate quantities of interest (conditional expectations and probability density functions) forward and backward through the network by means of exact linear operator equations. Such new equations can be used as a starting point to develop new effective parameterizations of deep neural networks, and provide a new framework to study deep-learning via operator theoretic methods. The proposed MZ formulation of deep learning naturally introduces a new concept, i.e., the memory of the neural network, which plays a fundamental role in low-dimensional modeling and parameterization. By using the theory of contraction mappings, we develop sufficient conditions for the memory of the neural network to decay with the number of 
    
[^163]: 双谱神经网络

    Bispectral Neural Networks. (arXiv:2209.03416v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.03416](http://arxiv.org/abs/2209.03416)

    本文提出了一种称为双谱神经网络的模型，能够从数据的隐含对称性中学习群、不可约表示和对应的完全不变映射，具有强大的基于不变性的对抗鲁棒性。

    

    本文提出了一种神经网络架构，双谱神经网络(BNNs)，用于学习表示在紧致可交换群在定义信号的空间上的作用下具有不变性的表示。该模型结合了双谱的思想，即是一个解析定义的群不变量，它是完整的——也就是说，它保留了所有信号结构，同时只去除了由于群作用引起的变化。在这里，我们证明了BNNs能够通过数据中的隐含对称性同时学习群、它们的不可约表示和对应的等变和完全不变映射。此外，我们证明了完整性属性赋予了这些网络强大的基于不变性的对抗鲁棒性。这项工作将Bispectral Neural Networks确立为稳健不变表示学习的强大计算原语。

    We present a neural network architecture, Bispectral Neural Networks (BNNs) for learning representations that are invariant to the actions of compact commutative groups on the space over which a signal is defined. The model incorporates the ansatz of the bispectrum, an analytically defined group invariant that is complete -- that is, it preserves all signal structure while removing only the variation due to group actions. Here, we demonstrate that BNNs are able to simultaneously learn groups, their irreducible representations, and corresponding equivariant and complete-invariant maps purely from the symmetries implicit in data. Further, we demonstrate that the completeness property endows these networks with strong invariance-based adversarial robustness. This work establishes Bispectral Neural Networks as a powerful computational primitive for robust invariant representation learning
    
[^164]: 摇晃着前行：基于PixelCNN++的长程依赖关系实现健壮离群检测

    Shaken, and Stirred: Long-Range Dependencies Enable Robust Outlier Detection with PixelCNN++. (arXiv:2208.13579v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.13579](http://arxiv.org/abs/2208.13579)

    本文提出了两种双射变换方法（“搅拌”和“摇晃”），用于改善深度自回归模型PixelCNN++似然性中的低级偏差，并隔离长程依赖的贡献。这些方法可以在评估时很容易计算，并且在离群检测方面表现出了很好的效果。

    

    可靠的离群检测对于深度学习模型的实际应用至关重要。尽管已经进行了广泛的研究，但由深度生成模型产生的似然性通常被认为在离群检测方面不实用。首先，深度生成模型似然性易受低级输入统计的偏见影响。其次，许多最近的解决方案对纠正这些偏见的计算成本很高，或者在复杂的自然数据集上泛化能力差。在本文中，我们探讨了一种基于最先进的深度自回归模型PixelCNN++的离群检测方法。我们表明，PixelCNN++似然性中的偏见主要来自于基于局部依赖关系的预测。我们提出了两种双射变换族--“搅拌”和“摇晃”，这可以改善低级偏差，并将长程依赖关系的贡献隔离在PixelCNN++的似然性中。这些变换成本低廉，并且在评估时可以很容易地计算。我们对我们的方法进行了广泛的测试。

    Reliable outlier detection is critical for real-world deployment of deep learning models. Although extensively studied, likelihoods produced by deep generative models have been largely dismissed as being impractical for outlier detection. First, deep generative model likelihoods are readily biased by low-level input statistics. Second, many recent solutions for correcting these biases are computationally expensive, or do not generalize well to complex, natural datasets. Here, we explore outlier detection with a state-of-the-art deep autoregressive model: PixelCNN++. We show that biases in PixelCNN++ likelihoods arise primarily from predictions based on local dependencies. We propose two families of bijective transformations -- ``stirring'' and ``shaking'' -- which ameliorate low-level biases and isolate the contribution of long-range dependencies to PixelCNN++ likelihoods. These transformations are inexpensive and readily computed at evaluation time. We test our approaches extensively 
    
[^165]: LAMDA-SSL：Python中的半监督学习工具包

    LAMDA-SSL: Semi-Supervised Learning in Python. (arXiv:2208.04610v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.04610](http://arxiv.org/abs/2208.04610)

    LAMDA-SSL是一款Python半监督学习工具包，提供了详细的用法文档和实现的所有算法，极大地方便了用户使用。

    

    LAMDA-SSL在GitHub上开源，其详细用法文档可以在https://ygzwqzd.github.io/LAMDA-SSL/上获得。该文档从不同角度详细介绍了LAMDA-SSL，并可以分为四个部分。第一部分介绍了LAMDA-SSL的设计理念、特点和功能。第二部分通过丰富的例子详细说明了LAMDA-SSL的用法。第三部分介绍了由LAMDA-SSL实现的所有算法，帮助用户快速了解和选择SSL算法。第四部分展示了LAMDA-SSL的API。这份详细的文档极大地降低了用户熟悉LAMDA-SSL工具包和SSL算法的成本。

    LAMDA-SSL is open-sourced on GitHub and its detailed usage documentation is available at https://ygzwqzd.github.io/LAMDA-SSL/. This documentation introduces LAMDA-SSL in detail from various aspects and can be divided into four parts. The first part introduces the design idea, features and functions of LAMDA-SSL. The second part shows the usage of LAMDA-SSL by abundant examples in detail. The third part introduces all algorithms implemented by LAMDA-SSL to help users quickly understand and choose SSL algorithms. The fourth part shows the APIs of LAMDA-SSL. This detailed documentation greatly reduces the cost of familiarizing users with LAMDA-SSL toolkit and SSL algorithms.
    
[^166]: 学习学习在Meta的生产中预测性能回归

    Learning to Learn to Predict Performance Regressions in Production at Meta. (arXiv:2208.04351v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2208.04351](http://arxiv.org/abs/2208.04351)

    本文介绍了在Meta研究和部署的基于机器学习的回归预测流程，研究结果显示性能预测问题的固有难度，SuperPerforator模型表现最佳。

    

    在生产环境中捕捉和归因于代码变更引起的性能回归是困难的；预测它们在前期更加困难。本文是关于自动学习预测软件性能回归的入门介绍，我们在Meta研究和部署了一个基于机器学习的回归预测流水线后获得了一些经验。本文报告了一个比较研究结果，包括四个逐渐增加复杂度的机器学习模型：(1) 模糊代码，(2) 词袋模型，(3) 预先训练好的Transformer，和(4) 自定义Transformer模型，名为SuperPerforator。我们的研究显示了性能预测问题的固有难度，这一问题的特点是良性变更对恶性回归变更数量的巨大不平衡。我们的结果还质疑了Transformer架构在性能预测上的普适性：一个预训练的CodeBERT方法表现出惊人的 poor 表现；我们高度自定义的SuperPerforator模型--采用高级技术，如无效代码移除、合成数据增强和随机权重平均--表现最佳。最后，我们总结了所学经验，并勾勒出未来研究的有趣方向。

    Catching and attributing code change-induced performance regressions in production is hard; predicting them beforehand, even harder. A primer on automatically learning to predict performance regressions in software, this article gives an account of the experiences we gained when researching and deploying an ML-based regression prediction pipeline at Meta. In this paper, we report on a comparative study with four ML models of increasing complexity, from (1) code-opaque, over (2) Bag of Words, (3) off-the-shelve Transformer-based, to (4) a bespoke Transformer-based model, coined SuperPerforator. Our investigation shows the inherent difficulty of the performance prediction problem, which is characterized by a large imbalance of benign onto regressing changes. Our results also call into question the general applicability of Transformer-based architectures for performance prediction: an off-the-shelve CodeBERT-based approach had surprisingly poor performance; our highly customized SuperPerf
    
[^167]: DIVISION: 通过双激活精度实现高效内存训练

    DIVISION: Memory Efficient Training via Dual Activation Precision. (arXiv:2208.04187v5 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.04187](http://arxiv.org/abs/2208.04187)

    本文提出一种内存高效的DNN训练方法DIVISION，通过保留LFC的高精度，将HFC压缩成低精度的轻量副本，显著减少了内存成本。

    

    激活压缩训练提供了一个减少深度神经网络训练内存成本的解决方案。然而，目前最先进的工作将量化位宽搜索与训练结合在一起，使得过程复杂且不够透明。为此，我们提出了一种简单有效的压缩DNN训练方法。我们的方法得到了如下结论：神经网络反向传播主要利用激活图的低频组成部分（LFC)，而大部分内存是用来缓存训练期间的高频组成部分（HFC）。这表明激活图的HFC在DNN训练期间高度冗余且易于压缩，这也启发了我们提出的Dual Activation Precision (DIVISION)。在训练过程中，DIVISION保留LFC的高精度副本，并将HFC压缩成低数值精度的轻量副本。这可以显著降低内存成本而不会对DNN训练产生负面影响。

    Activation compressed training provides a solution towards reducing the memory cost of training deep neural networks~(DNNs). However, state-of-the-art work combines a search of quantization bit-width with the training, which makes the procedure complicated and less transparent. To this end, we propose a simple and effective method to compress DNN training. Our method is motivated by an instructive observation: DNN backward propagation mainly utilizes the low-frequency component (LFC) of the activation maps, while the majority of memory is for caching the high-frequency component (HFC) during the training. This indicates the HFC of activation maps is highly redundant and compressible during DNN training, which inspires our proposed Dual Activation Precision (DIVISION). During the training, DIVISION preserves the high-precision copy of LFC and compresses the HFC into a light-weight copy with low numerical precision. This can significantly reduce the memory cost without negatively affecti
    
[^168]: 排序可行性及线性排序问题：新的概率洞见与算法

    Rankability and Linear Ordering Problem: New Probabilistic Insight and Algorithms. (arXiv:2208.03860v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.03860](http://arxiv.org/abs/2208.03860)

    该论文提出了一种基于概率模型和Slater谱概念的算法，能够判断和解决线性排序问题中数据是否可排序并给出有意义的解释。

    

    线性排序问题（LOP）常用于许多研究领域中，其目的在于对M个对象进行排序。虽然已经努力制定高效的LOP算法，但验证数据是否是可排序的，也就是LOP解是否具有有意义的解释，却得到了很少的关注。为了解决这个问题，我们采用概率角度，将成对比较的结果建模为具有公共参数的伯努利变量，并从观测数据中估计参数。由于所需枚举的蛮力方法具有O（M！）的禁止复杂性，因此我们重新构思了这个问题，并引入了一个推广Slater指数的Slater谱概念，然后设计了一个算法来找到谱，并具有O（M^3 2^M）的复杂度，对于中等大小的M而言是可管理的。此外，通过对算法进行了微小的修改，我们能够找到所有可排序性和排名的分解。

    The linear ordering problem (LOP), which consists in ordering M objects from their pairwise comparisons, is commonly applied in many areas of research. While efforts have been made to devise efficient LOP algorithms, verification of whether the data are rankable, that is, if the linear ordering problem (LOP) solutions have a meaningful interpretation, received much less attention. To address this problem, we adopt a probabilistic perspective where the results of pairwise comparisons are modeled as Bernoulli variables with a common parameter and we estimate the latter from the observed data. The brute-force approach to the required enumeration has a prohibitive complexity of O(M !), so we reformulate the problem and introduce a concept of the Slater spectrum that generalizes the Slater index, and then devise an algorithm to find the spectrum with complexity O(M^3 2^M) that is manageable for moderate values of M. Furthermore, with a minor modification of the algorithm, we are able to fin
    
[^169]: 可微分的基于Agent的流行病模拟

    Differentiable Agent-based Epidemiology. (arXiv:2207.09714v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.09714](http://arxiv.org/abs/2207.09714)

    本文介绍一种可扩展、可微分的基于Agent的流行病模拟设计——GradABM，能够在较短时间内快速模拟百万级别的人口，并与深度神经网络集成和接受异构数据源，为校准、预测和评估政策干预提供了便利。

    

    机械模拟器是流行病学探索复杂，动态感染在不同条件下并在不确定的环境中导航的不可或缺工具。 基于代理的模型（ABM）是一种越来越受欢迎的仿真范例，可以用细节表示接触交互的异质性和个体行为的代理。 然而，传统ABM框架不可微分并且在可扩展性上存在挑战；因此，将它们连接到辅助数据源是不平凡的。 在本文中，我们介绍了GradABM：一种可扩展的，可微分的基于代理模型的设计，适合于使用自动微分进行基于梯度的学习。 GradABM可以在商品硬件上快速模拟数百万规模的人口，并与深度神经网络集成和接受异构数据源。 这为校准，预测和评估政策干预提供了一系列实际的好处。

    Mechanistic simulators are an indispensable tool for epidemiology to explore the behavior of complex, dynamic infections under varying conditions and navigate uncertain environments. Agent-based models (ABMs) are an increasingly popular simulation paradigm that can represent the heterogeneity of contact interactions with granular detail and agency of individual behavior. However, conventional ABM frameworks are not differentiable and present challenges in scalability; due to which it is non-trivial to connect them to auxiliary data sources. In this paper, we introduce GradABM: a scalable, differentiable design for agent-based modeling that is amenable to gradient-based learning with automatic differentiation. GradABM can quickly simulate million-size populations in few seconds on commodity hardware, integrate with deep neural networks and ingest heterogeneous data sources. This provides an array of practical benefits for calibration, forecasting, and evaluating policy interventions. We
    
[^170]: 多元长序列时间序列预测的通用记忆驱动变压器

    Generalizable Memory-driven Transformer for Multivariate Long Sequence Time-series Forecasting. (arXiv:2207.07827v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.07827](http://arxiv.org/abs/2207.07827)

    本文提出了一种通用记忆驱动变压器，通过集成多个时间序列特征来驱动预测过程，逐步引入噪声以增强泛化能力，在多个数据集上实现了更优秀的预测性能。

    

    多元长序列时间序列预测(M-LSTF)是一个实际但具有挑战性的问题。与传统的时间序列预测任务不同，M-LSTF任务从两个方面更具挑战性：1) M-LSTF模型需要同时学习多个时间特征之间的时间序列模式；2)在滚动预测设置中，两个连续训练样本之间的相似度随着预测长度的增加而增加，这使得模型更易于过拟合。本文提出了一种通用记忆驱动变压器来解决M-LSTF问题。具体而言，我们首先提出了一个全局层面的记忆组件，通过集成多个时间序列特征来驱动预测过程。此外，我们采用渐进式的方式来训练我们的模型，以增强其泛化能力，逐步在训练样本中引入伯努利噪声。在多个领域的五个不同数据集上进行了大量实验。实验结果表明，我们提出的模型优于现有的方法，并在所有数据集上实现了更优异的预测性能。

    Multivariate long sequence time-series forecasting (M-LSTF) is a practical but challenging problem. Unlike traditional timer-series forecasting tasks, M-LSTF tasks are more challenging from two aspects: 1) M-LSTF models need to learn time-series patterns both within and between multiple time features; 2) Under the rolling forecasting setting, the similarity between two consecutive training samples increases with the increasing prediction length, which makes models more prone to overfitting. In this paper, we propose a generalizable memory-driven Transformer to target M-LSTF problems. Specifically, we first propose a global-level memory component to drive the forecasting procedure by integrating multiple time-series features. In addition, we adopt a progressive fashion to train our model to increase its generalizability, in which we gradually introduce Bernoulli noises to training samples. Extensive experiments have been performed on five different datasets across multiple fields. Exper
    
[^171]: 学习深度时间索引模型用于时间序列预测

    Learning Deep Time-index Models for Time Series Forecasting. (arXiv:2207.06046v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.06046](http://arxiv.org/abs/2207.06046)

    本文提出了DeepTime框架，一种用于学习深度时间索引模型的元优化框架，能够有效地预测时间序列，并在真实世界数据集上获得了与先进方法相当的结果。

    

    深度学习被广泛应用于时间序列预测中，导致历史价值模型类别中涌现出大量新方法。虽然时间索引模型具有吸引人的特性，比如能够对底层时间序列动态性建模，但仍未受到足够的关注。本文提出了DeepTime框架，这是一种元优化框架，用于学习克服这些限制的深度时间索引模型，在长序列时间序列预测设置下在真实世界数据集上展开了广泛的实验。实验结果表明，我们的方法与现有技术相比具有竞争力，并且具有高效和准确的预测模型。

    Deep learning has been actively applied to time series forecasting, leading to a deluge of new methods, belonging to the class of historical-value models. Yet, despite the attractive properties of time-index models, such as being able to model the continuous nature of underlying time series dynamics, little attention has been given to them. Indeed, while naive deep time-index models are far more expressive than the manually predefined function representations of classical time-index models, they are inadequate for forecasting, being unable to generalize to unseen time steps due to the lack of inductive bias. In this paper, we propose DeepTime, a meta-optimization framework to learn deep time-index models which overcome these limitations, yielding an efficient and accurate forecasting model. Extensive experiments on real world datasets in the long sequence time-series forecasting setting demonstrate that our approach achieves competitive results with state-of-the-art methods, and is hig
    
[^172]: 大型学习

    Big Learning. (arXiv:2207.03899v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.03899](http://arxiv.org/abs/2207.03899)

    大型学习是一种把大规模完整/不完整的训练数据中固有信息同步建模并使用一个通用基础模型，利用所有联合/条件/边际数据能力并统一传统机器学习范式的通用学习范式。

    

    大规模/基础模型的最新进展为深度学习开辟了一条充满希望的道路，其中路线图稳步从大数据到大模型到（新引入的）大型学习。具体而言，大型学习通过同步建模可能存在的多个/全部联合/条件/边际数据分布，利用大规模完整/不完整的训练数据中固有的信息，使用一个通用基础模型。我们揭示大型学习具有以下特点：（i）是大多数现有基础模型的基础；（ii）具有对完整/不完整的训练数据和可信数据任务的非凡灵活性；（iii）能够使用一个通用模型提供所有联合/条件/边际数据能力；（iv）统一传统机器学习范式并启用它们的灵活协作，体现为通用学习范式。进行了不同的实验，验证了当前方法的有效性。

    Recent advances in big/foundation models reveal a promising path for deep learning, where the roadmap steadily moves from big data to big models to (the newly-introduced) big learning. Specifically, the big learning exhaustively exploits the information inherent in its large-scale complete/incomplete training data, by simultaneously modeling many/all joint/conditional/marginal data distributions across potentially diverse domains, with one universal foundation model. We reveal that big learning ($i$) underlies most existing foundation models, ($ii$) is equipped with extraordinary flexibilities for complete/incomplete training data and trustworthy data tasks, ($iii$) is capable of delivering all joint/conditional/marginal data capabilities with one universal model, and ($iv$) unifies conventional machine learning paradigms and enables their flexible cooperations, manifested as a universal learning paradigm. Diverse experiments are carried out to validate the effectiveness of the present
    
[^173]: 基于持续同调的描述符用于无定形材料机器学习势的预测

    Persistent homology-based descriptor for machine-learning potential of amorphous structures. (arXiv:2206.13727v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.13727](http://arxiv.org/abs/2206.13727)

    本文提出了一种基于持续同调的二维描述符PD，用于构建表示原子配置的不变描述符，在机器学习势的构建中，PD能够捕获不同长度的相关性，并能够提高模型性能。

    

    在凝聚态物理学中，高精度预测无定形材料的物理性质是一项挑战性的任务。机器学习势是实现这一目标的一种有前途的方法，它是用于计算机费时的从头开始计算的替代方法。在应用机器学习势时，构建表示原子配置的描述符至关重要。这些描述符应该对对称操作具有不变性。使用原子位置的平滑重叠和图神经网络（GNN）的手工制作表示法是构建对称不变描述符的方法的示例。本研究提出了一种基于持续同调（PH）的二维描述符，基于持续同调产生的二维表征，称为持续同调图（PD），来构建描述符。首先，我们证明了从PD获得的归一化的二维直方图可以预测不同密度下无定形碳（aC）的每个原子的平均能量，即使使用简单模型也可以实现。其次，PD的分析揭示了描述符捕获了aC中不同的相关长度。最后，我们使用基于PD的描述符训练了一个神经网络势，并证明其优于基于GNN和平滑重叠的描述符。我们的结果表明，使用持续同调基础描述符是构建适用于无定形材料的机器学习势的一种有前景的方法。

    High-accuracy prediction of the physical properties of amorphous materials is challenging in condensed-matter physics. A promising method to achieve this is machine-learning potentials, which is an alternative to computationally demanding ab initio calculations. When applying machine-learning potentials, the construction of descriptors to represent atomic configurations is crucial. These descriptors should be invariant to symmetry operations. Handcrafted representations using a smooth overlap of atomic positions and graph neural networks (GNN) are examples of methods used for constructing symmetry-invariant descriptors. In this study, we propose a novel descriptor based on a persistence diagram (PD), a two-dimensional representation of persistent homology (PH). First, we demonstrated that the normalized two-dimensional histogram obtained from PD could predict the average energy per atom of amorphous carbon (aC) at various densities, even when using a simple model. Second, an analysis o
    
[^174]: 好时机: 表达需求式视觉导航中启示求助的学习框架

    Good Time to Ask: A Learning Framework for Asking for Help in Embodied Visual Navigation. (arXiv:2206.10606v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.10606](http://arxiv.org/abs/2206.10606)

    该论文提出了一种学习框架，使得机器人可以主动寻求帮助以解决表达需求式视觉导航任务中存在的难题，并展示了其在缺乏反馈信息的情况下依然稳健的抗干扰能力。

    

    实际上，询问求助比在未知位置的空间内搜索要更加高效。我们提出了一个学习框架，使得代理能够在这种感知视觉任务中主动地寻求帮助，其中反馈信息告知代理目标在其视野中的位置。为了模拟现实情况下老师不总是在场的场景，我们提出了一个训练课程，在这个过程中反馈信息并不总是可用的。我们制定了一个不确定性度量目标位置，在经验结果的基础上展示了该方法的有效性，即使在缺乏反馈信息的情况下代理依然保持了抗干扰能力。

    In reality, it is often more efficient to ask for help than to search the entire space to find an object with an unknown location. We present a learning framework that enables an agent to actively ask for help in such embodied visual navigation tasks, where the feedback informs the agent of where the goal is in its view. To emulate the real-world scenario that a teacher may not always be present, we propose a training curriculum where feedback is not always available. We formulate an uncertainty measure of where the goal is and use empirical results to show that through this approach, the agent learns to ask for help effectively while remaining robust when feedback is not available.
    
[^175]: 个性化子图联邦学习

    Personalized Subgraph Federated Learning. (arXiv:2206.10206v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.10206](http://arxiv.org/abs/2206.10206)

    本文提出一个新的子图联邦学习问题，即个性化子图联邦学习，专注于联合改进相关的本地GNN并提出了一个新颖的框架FED-PUB来解决这个问题。在实验中表现出了优异的性能。

    

    全局图的子图可能分布在多个设备上，由于隐私限制而只能本地访问，尽管子图之间可能存在链接。最近提出的子图联邦学习方法处理跨本地子图的缺失链接，同时在其上分布式地训练图神经网络。然而，它们忽视了全局图不同社区组成的子图之间的不可避免的异质性，因此导致本地GNN模型的不兼容知识的崩溃。为此，我们提出了一个新的子图联邦学习问题，即个性化子图联邦学习，它专注于联合改进相关的本地GNN，而不是学习单个全局模型，并提出了一个新颖的框架FEDerated Personalized sUBgraph learning (FED-PUB)来解决这个问题。由于服务器无法访问每个客户端中的子图，FED-PUB利用随机图作为输入，利用本地GNN的功能嵌入来利用子图之间的相关性，同时保护隐私。我们在公共数据集和真实数据集上进行了实验证明，FED-PUB在精度、通信效率和收敛速度方面都达到了最先进的水平。

    Subgraphs of a larger global graph may be distributed across multiple devices, and only locally accessible due to privacy restrictions, although there may be links between subgraphs. Recently proposed subgraph Federated Learning (FL) methods deal with those missing links across local subgraphs while distributively training Graph Neural Networks (GNNs) on them. However, they have overlooked the inevitable heterogeneity between subgraphs comprising different communities of a global graph, consequently collapsing the incompatible knowledge from local GNN models. To this end, we introduce a new subgraph FL problem, personalized subgraph FL, which focuses on the joint improvement of the interrelated local GNNs rather than learning a single global model, and propose a novel framework, FEDerated Personalized sUBgraph learning (FED-PUB), to tackle it. Since the server cannot access the subgraph in each client, FED-PUB utilizes functional embeddings of the local GNNs using random graphs as inpu
    
[^176]: 可调信息瓶颈和Rényi度量通过分类效用、公平性和紧凑性

    Classification Utility, Fairness, and Compactness via Tunable Information Bottleneck and R\'enyi Measures. (arXiv:2206.10043v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.10043](http://arxiv.org/abs/2206.10043)

    本文提出了一种新型公平表示学习方法(RFIB)，兼顾了表示效用、公平性和紧凑性，并将其应用于图像和表格数据分类中。该方法考虑到了人口平等和等化赔率等不同的公平性约束，通过涉及经典信息瓶颈(IB)度量的损失函数实现。实验表明，该方法能够在实现公平性的同时保持较高的准确性。

    

    在机器学习中，设计可以准确获取信息而不歧视任何敏感属性的算法对于社会接受AI用于关键应用场景至关重要。本文提出了一种新型公平表示学习方法，称为Rényi公平信息瓶颈方法(RFIB)，它兼顾了表示的效用、公平性和紧凑性，并将其应用于图像和表格数据分类中。与以往的工作相比，我们的方法考虑到了人口平等和等化赔率等不同的公平性约束，使得满足这两个准则更加精细。利用变分方法，我们证明了我们的目标可以产生一个损失函数，其中涉及了经典信息瓶颈(IB)度量，并在两个Rényi$(\alpha)$度量中建立了一个上界，用于衡量输入和表示之间的紧凑度。实验表明，我们的方法在实现公平性的同时，也取得了有竞争力的准确性。

    Designing machine learning algorithms that are accurate yet fair, not discriminating based on any sensitive attribute, is of paramount importance for society to accept AI for critical applications. In this article, we propose a novel fair representation learning method termed the R\'enyi Fair Information Bottleneck Method (RFIB) which incorporates constraints for utility, fairness, and compactness (compression) of representation, and apply it to image and tabular data classification. A key attribute of our approach is that we consider - in contrast to most prior work - both demographic parity and equalized odds as fairness constraints, allowing for a more nuanced satisfaction of both criteria. Leveraging a variational approach, we show that our objectives yield a loss function involving classical Information Bottleneck (IB) measures and establish an upper bound in terms of two R\'enyi measures of order $\alpha$ on the mutual information IB term measuring compactness between the input a
    
[^177]: PrivHAR：从隐私保护角度识别人类活动

    PrivHAR: Recognizing Human Actions From Privacy-preserving Lens. (arXiv:2206.03891v2 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2206.03891](http://arxiv.org/abs/2206.03891)

    该论文提出了一种优化框架，通过参数化摄像头镜头，对视频进行处理以保护个人隐私，并在维护活动识别特征的同时进行了模拟和硬件实验的验证。

    

    数字摄像机的广泛使用引发了人们对隐私和安全的日益关注，特别是在识别人类活动的应用中。本文提出了一种优化框架，通过参数化摄像头镜头，成功地降低了视频质量，以抑制隐私属性并防止敌对攻击，同时保留了活动识别相关特征，提供强大的视觉隐私保护。我们进行了大量模拟和硬件实验来验证我们的方法。

    The accelerated use of digital cameras prompts an increasing concern about privacy and security, particularly in applications such as action recognition. In this paper, we propose an optimizing framework to provide robust visual privacy protection along the human action recognition pipeline. Our framework parameterizes the camera lens to successfully degrade the quality of the videos to inhibit privacy attributes and protect against adversarial attacks while maintaining relevant features for activity recognition. We validate our approach with extensive simulations and hardware experiments.
    
[^178]: 拓扑深度学习：超越图数据

    Topological Deep Learning: Going Beyond Graph Data. (arXiv:2206.00606v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.00606](http://arxiv.org/abs/2206.00606)

    本文提出了一个拓扑深度学习的框架，其中包含组合复合体这一新型拓扑域。组合复合体结合了超图和胞腔复合体的优点，允许构建分层高阶关系。

    

    拓扑深度学习是一个快速发展的领域，与开发支持于拓扑域上的深度学习模型有关，例如单纯复合体、胞腔复合体和超图。这些拓扑域在科学计算中广泛应用。在本文中，我们提出了一个建立在更丰富数据结构之上的统一深度学习框架，包括拓扑域。我们首先介绍组合复合体，这是一种新型的拓扑域。组合复合体可以看作是保持某些理想性质的图的推广。类似于超图，组合复合体对关系集合不施加任何约束。此外，组合复合体允许构建分层高阶关系，类似于单纯和胞腔复合体中的关系。因此，组合复合体推广并结合了超图和胞腔复合体的有用特性。

    Topological deep learning is a rapidly growing field that pertains to the development of deep learning models for data supported on topological domains such as simplicial complexes, cell complexes, and hypergraphs, which generalize many domains encountered in scientific computations. In this paper, we present a unifying deep learning framework built upon a richer data structure that includes widely adopted topological domains.  Specifically, we first introduce combinatorial complexes, a novel type of topological domain. Combinatorial complexes can be seen as generalizations of graphs that maintain certain desirable properties. Similar to hypergraphs, combinatorial complexes impose no constraints on the set of relations. In addition, combinatorial complexes permit the construction of hierarchical higher-order relations, analogous to those found in simplicial and cell complexes. Thus, combinatorial complexes generalize and combine useful traits of both hypergraphs and cell complexes, whi
    
[^179]: 广义的有监督对比学习

    Generalized Supervised Contrastive Learning. (arXiv:2206.00384v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2206.00384](http://arxiv.org/abs/2206.00384)

    本文提出了一种广义的有监督对比损失，通过充分利用标签分布来增强有监督对比损失的能力，从而适应各种现有技术，包括CutMix和知识蒸馏。在几个基准数据集上进行了实验，结果显示该方法优于现有的有监督对比学习方法，并在CIFAR-10，CIFAR-100和ImageNet上实现了最先进的结果。

    

    随着对自监督学习中对比学习的最新进展，有监督对比学习已成功将这些对比方法扩展到有监督的上下文中，在各种数据集上优于交叉熵。然而，有监督对比学习固有地使用以一位热目标向量的形式表示的标签信息，这种结构无法适应利用标签信息作为概率分布的方法，如CutMix和知识蒸馏。在本文中，我们介绍了一种广义的有监督对比损失，该损失度量了标签相似性和潜在相似性之间的交叉熵。这个概念通过充分利用标签分布，并使各种现有技术适应于训练现代神经网络而增强了有监督对比损失的能力。利用这种广义的有监督对比损失，我们为图像分类构建了一个定制的损失函数，并在几个基准数据集上进行了实验，包括CIFAR-10，CIFAR-100和ImageNet。提出的方法显示出比现有的有监督对比学习方法更好的效果，并在CIFAR-10，CIFAR-100和ImageNet上实现了最先进的结果。

    With the recent promising results of contrastive learning in the self-supervised learning paradigm, supervised contrastive learning has successfully extended these contrastive approaches to supervised contexts, outperforming cross-entropy on various datasets. However, supervised contrastive learning inherently employs label information in a binary form--either positive or negative--using a one-hot target vector. This structure struggles to adapt to methods that exploit label information as a probability distribution, such as CutMix and knowledge distillation. In this paper, we introduce a generalized supervised contrastive loss, which measures cross-entropy between label similarity and latent similarity. This concept enhances the capabilities of supervised contrastive loss by fully utilizing the label distribution and enabling the adaptation of various existing techniques for training modern neural networks. Leveraging this generalized supervised contrastive loss, we construct a tailor
    
[^180]: SepIt: 接近单通道语音分离界限的方法

    SepIt: Approaching a Single Channel Speech Separation Bound. (arXiv:2205.11801v4 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2205.11801](http://arxiv.org/abs/2205.11801)

    该论文提出了一种接近单通道语音分离界限的方法SepIt，在实验中表现优于最先进的神经网络方法，尤其是在5个和10个说话人的情况下仍有提高空间。

    

    我们提出了一种单通道语音分离任务的上界，该上界基于对语音短时段性质的假设。使用该上界，我们能够展示，虽然最近的方法已经取得了在少数几个说话人的情况下显著的进展，但在5个和10个说话人的情况下还有提高的空间。接着，我们引入了深度神经网络SepIt来迭代地改进不同说话人的估计。在测试时，SepIt对于每个测试样本的迭代次数是可变的，基于我们分析中出现的互信息标准。通过广泛的实验，SepIt在2、3、5和10个说话人的情况下胜过了最先进的神经网络方法。

    We present an upper bound for the Single Channel Speech Separation task, which is based on an assumption regarding the nature of short segments of speech. Using the bound, we are able to show that while the recent methods have made significant progress for a few speakers, there is room for improvement for five and ten speakers. We then introduce a Deep neural network, SepIt, that iteratively improves the different speakers' estimation. At test time, SpeIt has a varying number of iterations per test sample, based on a mutual information criterion that arises from our analysis. In an extensive set of experiments, SepIt outperforms the state-of-the-art neural networks for 2, 3, 5, and 10 speakers.
    
[^181]: 带有最小超参数化的深度神经网络中的记忆化与优化

    Memorization and Optimization in Deep Neural Networks with Minimum Over-parameterization. (arXiv:2205.10217v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2205.10217](http://arxiv.org/abs/2205.10217)

    本文提供了一个最小超参数化的深度神经网络中最小的NTK特征值的下界，具有次线性层宽的深层神经网络是强大的记忆器和优化器，只要参数数量超过样本数量。

    

    神经切向核（NTK）已成为提供深度神经网络记忆化、优化和泛化保证的强大工具。部分学者已研究了至少一层具有$\Omega(N)$个神经元的两层和深层网络的NTK谱，其中$N$是训练样本数量。此外，越来越多的证据表明，具有次线性层宽的深层神经网络是强大的记忆器和优化器，只要参数数量超过样本数量即可。因此，一个自然的开放性问题是在这种具有挑战性的次线性设置下，NTK是否存在良好的条件。在本文中，我们肯定地回答了这个问题。我们的主要技术贡献是提供了一个深度神经网络中最小的NTK特征值的下界，即参数数量大约为$\Omega(N)$，因此神经元数量至少为$\Omega(\sqrt{N})$。为展示我们算法的适用性，我们在多项任务上进行了实证分析。

    The Neural Tangent Kernel (NTK) has emerged as a powerful tool to provide memorization, optimization and generalization guarantees in deep neural networks. A line of work has studied the NTK spectrum for two-layer and deep networks with at least a layer with $\Omega(N)$ neurons, $N$ being the number of training samples. Furthermore, there is increasing evidence suggesting that deep networks with sub-linear layer widths are powerful memorizers and optimizers, as long as the number of parameters exceeds the number of samples. Thus, a natural open question is whether the NTK is well conditioned in such a challenging sub-linear setup. In this paper, we answer this question in the affirmative. Our key technical contribution is a lower bound on the smallest NTK eigenvalue for deep networks with the minimum possible over-parameterization: the number of parameters is roughly $\Omega(N)$ and, hence, the number of neurons is as little as $\Omega(\sqrt{N})$. To showcase the applicability of our N
    
[^182]: 任意延迟的无投影在线算法

    Projection-free Online Learning with Arbitrary Delays. (arXiv:2204.04964v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2204.04964](http://arxiv.org/abs/2204.04964)

    该论文提出了一种能够应对任意延迟的无投影在线算法，与现有算法相比，其具有更好的遗憾保证。

    

    近来，由于在处理具有复杂约束的高维问题时具有高效性，避免了投影操作并通过较少的计算如线性优化（LO）计算的无投影在线学习受到了广泛关注。但是，以往的研究假定任何查询梯度都会立即显示，这在实践中可能并不成立，从而限制了其应用。为了解决这个限制，我们将在线Frank-Wolfe（OFW）算法和在线平滑无投影（OSPF）算法推广到延迟设置中，其中查询的梯度可以被任意回合延迟。具体而言，我们推广的OFW的主要思想是在收到任何延迟的梯度后执行类似于原始OFW的更新，并为每一轮播放最新的决策。此外，OSPF的基本改变是用考虑任意延迟的加权和替换了查询梯度的总和。在各种延迟假设下，我们建立了广义OFW和OSPF的亚线性遗憾保证，并显示它们的遗憾界与设计用于无延迟设置的现有算法相当或更好。

    Projection-free online learning, which eschews the projection operation via less expensive computations such as linear optimization (LO), has received much interest recently due to its efficiency in handling high-dimensional problems with complex constraints. However, previous studies assume that any queried gradient is revealed immediately, which may not hold in practice and limits their applications. To address this limitation, we generalize the online Frank-Wolfe (OFW) algorithm and the online smooth projection-free (OSPF) algorithm, which are state-of-the-art LO-based projection-free online algorithms for non-smooth and smooth functions respectively, into a delayed setting where queried gradients can be delayed by arbitrary rounds. Specifically, the main idea of our generalized OFW is to perform an update similar to the original OFW after receiving any delayed gradient, and play the latest decision for each round. Moreover, the essential change on OSPF is to replace the sum of quer
    
[^183]: 个性化激励作为广义纳什均衡问题中的反馈设计

    Personalized incentives as feedback design in generalized Nash equilibrium problems. (arXiv:2203.12948v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2203.12948](http://arxiv.org/abs/2203.12948)

    本文研究了对称互动的非单调广义纳什均衡问题，并提出了一种半分散纳什均衡寻找算法，其中，协调员通过整合代理的反馈来学习代理的伪梯度并为其设计个性化的激励，代理计算扩展博弈的解并反馈测量给协调员，该算法可返回静态设置下的纳什均衡。

    

    本文研究了对称互动的静态和动态非单调广义纳什均衡问题，这些问题已知具有潜势。然而，在实际情况下可能存在这样一种情况，即底层潜在函数的正式表达式不可用，并设计了一种半分散纳什均衡寻找算法。在所提出的两层方案中，协调员迭代地整合代理的（可能是噪声和零散的）反馈以学习代理的伪梯度，然后为他们设计个性化的激励。在代理方面，代理接收到个性化的激励，计算扩展博弈的解，然后向协调员返回反馈测量。在静态设置下，我们的算法在协调员拥有标准学习策略的情况下返回纳什均衡，而在可调整的常数误差内返回纳什均衡。

    We investigate both stationary and time-varying, nonmonotone generalized Nash equilibrium problems that exhibit symmetric interactions among the agents, which are known to be potential. As may happen in practical cases, however, we envision a scenario in which the formal expression of the underlying potential function is not available, and we design a semi-decentralized Nash equilibrium seeking algorithm. In the proposed two-layer scheme, a coordinator iteratively integrates the (possibly noisy and sporadic) agents' feedback to learn the pseudo-gradients of the agents, and then design personalized incentives for them. On their side, the agents receive those personalized incentives, compute a solution to an extended game, and then return feedback measurements to the coordinator. In the stationary setting, our algorithm returns a Nash equilibrium in case the coordinator is endowed with standard learning policies, while it returns a Nash equilibrium up to a constant, yet adjustable, error
    
[^184]: 使用探测和位置不确定性的多智能体主动搜索

    Multi-Agent Active Search using Detection and Location Uncertainty. (arXiv:2203.04524v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2203.04524](http://arxiv.org/abs/2203.04524)

    本文提出了一种新的算法，可以同时处理目标探测和位置不确定性，可以优化多智能体主动搜索的效率。

    

    在环境监测或灾害响应任务等应用中，主动搜索涉及到自主代理在搜索空间中使用决策算法检测目标，这些算法会根据历史观察结果进行调整。主动搜索算法必须处理两种类型的不确定性：检测不确定性和位置不确定性。在机器人领域中，更常见的方法是专注于位置不确定性，通过将检测概率阈值设定为零或一来消除检测不确定性。相反，在稀疏信号处理文献中，通常假定目标位置准确，并专注于其检测的不确定性。在本文中，我们首先提出一种推理方法来同时处理目标检测和位置不确定性。然后，基于这个推理方法构建一个使用汤普森抽样的决策算法，使得分散式多智能体主动搜索成为可能。我们进行了模拟实验，证明我们的算法在检测率和资源效率方面优于现有方法。

    Active search, in applications like environment monitoring or disaster response missions, involves autonomous agents detecting targets in a search space using decision making algorithms that adapt to the history of their observations. Active search algorithms must contend with two types of uncertainty: detection uncertainty and location uncertainty. The more common approach in robotics is to focus on location uncertainty and remove detection uncertainty by thresholding the detection probability to zero or one. In contrast, it is common in the sparse signal processing literature to assume the target location is accurate and instead focus on the uncertainty of its detection. In this work, we first propose an inference method to jointly handle both target detection and location uncertainty. We then build a decision making algorithm on this inference method that uses Thompson sampling to enable decentralized multi-agent active search. We perform simulation experiments to show that our algo
    
[^185]: 曲线建模下的车道检测方法重新思考

    Rethinking Efficient Lane Detection via Curve Modeling. (arXiv:2203.02431v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2203.02431](http://arxiv.org/abs/2203.02431)

    本文提出了一种新的参数曲线车道检测方法，在计算上更加简单且容易处理优化问题，并且在多个数据集中实现了最优的性能表现，可以作为未来车道检测领域的新基准。

    

    本论文提出了一种新的基于参数曲线的RGB图像车道检测方法。与现有的基于分割和基于点检测的方法不同，曲线方法可以自然地学习整体车道表示，而不需要解码预测或制定大量锚点的启发式方法。为了处理现有多项式曲线方法的优化困难，我们提出利用参数Bézier曲线，因为它易于计算、稳定且具有较高的自由变换度数。此外，我们还提出了一种可变形卷积的特征翻转融合方法，以利用驾驶场景中车道的对称性质。所提出的方法在流行的LLAMAS基准测试上实现了新的最优性能。它也在TuSimple和CULane数据集上实现了良好的准确性，同时保持较低的延迟（> 150 FPS）和小模型大小（<10M）。我们的方法可以作为新的基准，以在路标检测领域推动进一步的研究。

    This paper presents a novel parametric curve-based method for lane detection in RGB images. Unlike state-of-the-art segmentation-based and point detection-based methods that typically require heuristics to either decode predictions or formulate a large sum of anchors, the curve-based methods can learn holistic lane representations naturally. To handle the optimization difficulties of existing polynomial curve methods, we propose to exploit the parametric B\'ezier curve due to its ease of computation, stability, and high freedom degrees of transformations. In addition, we propose the deformable convolution-based feature flip fusion, for exploiting the symmetry properties of lanes in driving scenes. The proposed method achieves a new state-of-the-art performance on the popular LLAMAS benchmark. It also achieves favorable accuracy on the TuSimple and CULane datasets, while retaining both low latency (> 150 FPS) and small model size (< 10M). Our method can serve as a new baseline, to shed 
    
[^186]: 图注意力回顾

    Graph Attention Retrospective. (arXiv:2202.13060v5 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.13060](http://arxiv.org/abs/2202.13060)

    图注意力网络是一种能够从邻居节点的特征中聚合信息的模型，通过对上下文随机块模型的节点分类问题进行研究，证明了在“易”区间内，它能够区分跨类和内类边缘并维护重要边缘的权重。

    

    基于图的学习是机器学习中快速发展的一个子领域，应用于社交网络、引文网络和生物信息学。其中最流行的模型之一是图注意力网络。它们被引入使节点能够以非统一的方式从邻居节点的特征中聚合信息，与简单的图卷积不同，后者不能区分节点的邻居。本文在理论上研究了图注意网络的行为。我们对上下文随机块模型的节点分类问题证明了图注意机制的多个性能结果。在该模型中，节点特征来自于高斯混合模型，边缘来自于随机块模型。我们证明，在“易”区间内，高斯分布均值之间的距离足够大时，图注意力可以区分跨类和内类边缘。因此，它维护了重要边缘的权重。

    Graph-based learning is a rapidly growing sub-field of machine learning with applications in social networks, citation networks, and bioinformatics. One of the most popular models is graph attention networks. They were introduced to allow a node to aggregate information from features of neighbor nodes in a non-uniform way, in contrast to simple graph convolution which does not distinguish the neighbors of a node. In this paper, we theoretically study the behaviour of graph attention networks. We prove multiple results on the performance of the graph attention mechanism for the problem of node classification for a contextual stochastic block model. Here, the node features are obtained from a mixture of Gaussians and the edges from a stochastic block model. We show that in an "easy" regime, where the distance between the means of the Gaussians is large enough, graph attention is able to distinguish inter-class from intra-class edges. Thus it maintains the weights of important edges and s
    
[^187]: 自适应鲁棒的多任务学习

    Adaptive and Robust Multi-Task Learning. (arXiv:2202.05250v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2202.05250](http://arxiv.org/abs/2202.05250)

    本文提出一系列自适应方法，能够同时处理多任务学习的相似性和差异性，并具有统计保证和鲁棒性。

    

    本论文研究了解决从不同来源收集的多个数据集并对每个数据集学习一个模型的多任务学习问题。我们提出了一系列自适应方法，自动利用任务之间的相似性，同时处理它们之间的差异。我们证明了这些方法的统计保证，并证明它们对异常任务具有鲁棒性。通过合成和实际数据集的数值实验，证明了我们的新方法的功效。

    We study the multi-task learning problem that aims to simultaneously analyze multiple datasets collected from different sources and learn one model for each of them. We propose a family of adaptive methods that automatically utilize possible similarities among those tasks while carefully handling their differences. We derive sharp statistical guarantees for the methods and prove their robustness against outlier tasks. Numerical experiments on synthetic and real datasets demonstrate the efficacy of our new methods.
    
[^188]: 深度判别到核生成网络的定标推断方法

    Deep Discriminative to Kernel Generative Networks for Calibrated Inference. (arXiv:2201.13001v5 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.13001](http://arxiv.org/abs/2201.13001)

    该论文提出了将判别网络转换为生成网络的方法，用高斯核替换多面体中的仿射函数来生成模型，解决了内部和外部数据校准问题，并在 CIFAR-10，CIFAR-100 和 SVHN 等基准数据集上测试了方法的有效性。

    

    判别与生成网络在人工智能和自然智能的研究中都有其重要性，我们提出了一种将二者相结合的方法，将深度判别网络转换为核生成网络。我们将深度模型视为广义的划分规则，并使用高斯核替换由训练数据构成的多面体中的仿射函数，来获得生成模型。实验证明了我们方法的有效性。

    The fight between discriminative versus generative goes deep, in both the study of artificial and natural intelligence. In our view, both camps have complementary values. So, we sought to synergistically combine them. Here, we propose a methodology to convert deep discriminative networks to kernel generative networks. We leveraged the fact that deep models, including both random forests and deep networks, learn internal representations which are unions of polytopes with affine activation functions to conceptualize them both as generalized partitioning rules. We replace the affine function in each polytope populated by the training data with Gaussian kernel that results in a generative model. Theoretically, we derive the conditions under which our generative models are a consistent estimator of the corresponding class conditional density. Moreover, our proposed models obtain well calibrated posteriors for in-distribution, and extrapolate beyond the training data to handle out-of-distrib
    
[^189]: 一种鲁棒而灵活的椭圆分布混合缺失数据EM算法

    A Robust and Flexible EM Algorithm for Mixtures of Elliptical Distributions with Missing Data. (arXiv:2201.12020v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2201.12020](http://arxiv.org/abs/2201.12020)

    本研究提出了一种鲁棒且灵活的EM算法，用于处理缺失数据的混合椭圆分布，解决了非高斯数据插补的问题。

    

    本文解决了噪声和非高斯数据缺失数据插补的问题。Gaussian混合模型的期望极大算法已经被证明在一定程度上比基于k近邻或基于多重方程链的插补方法表现更好。但是，Gaussian混合模型对于异构数据是非鲁棒的，当数据受到离群点或遵循非高斯分布的影响时会导致性能估计较差。为了克服这个问题，本文研究了一种新的EM算法，用于椭圆分布的混合物，具有处理潜在缺失数据的特性。本文表明，这个问题可以归结为在通用条件下（即每个样本都是从一个可能不同的椭圆分布混合物中抽取的），估计angular Gaussian分布的混合物。

    This paper tackles the problem of missing data imputation for noisy and non-Gaussian data. A classical imputation method, the Expectation Maximization (EM) algorithm for Gaussian mixture models, has shown interesting properties when compared to other popular approaches such as those based on k-nearest neighbors or on multiple imputations by chained equations. However, Gaussian mixture models are known to be non-robust to heterogeneous data, which can lead to poor estimation performance when the data is contaminated by outliers or follows non-Gaussian distributions. To overcome this issue, a new EM algorithm is investigated for mixtures of elliptical distributions with the property of handling potential missing data. This paper shows that this problem reduces to the estimation of a mixture of Angular Gaussian distributions under generic assumptions (i.e., each sample is drawn from a mixture of elliptical distributions, which is possibly different for one sample to another). In that case
    
[^190]: PoNet：长序列中高效Token混合的池化网络

    PoNet: Pooling Network for Efficient Token Mixing in Long Sequences. (arXiv:2110.02442v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2110.02442](http://arxiv.org/abs/2110.02442)

    PoNet是一种池化网络，可用于长序列中的token混合，其具有线性复杂度，并可以比Transformer更好地处理长序列。该方法提供了一种高效且有效的自注意替代方案。

    

    基于Transformer的模型在各种NLP、视觉和语音任务中取得了巨大的成功。然而，Transformer的核心自注意机制和序列长度的平方时间和内存复杂度，阻碍了Transformer-based模型在处理长序列时的应用。为了缓解这个问题，有许多方法被提出，例如稀疏注意机制、低秩矩阵近似和可扩展的核函数，以及替代自注意的token混合。我们提出了一种用于长序列中token混合的新型池化网络(PoNet)，其复杂度为线性。我们设计了多粒度池化和池化融合来捕捉不同级别的上下文信息，并将其与token的交互结合起来。在长序列基准测试中，PoNet显著优于Transformer，并实现了有竞争力的准确性，而且在所有在GPU上测量的序列长度上，它仅比最快的模型FNet稍慢。我们还能可视化PoNet的注意力图，以展示它可以有效地混合具有不同上下文信息的token。我们的方法为处理长序列提供了一种高效且有效的自注意替代方案。

    Transformer-based models have achieved great success in various NLP, vision, and speech tasks. However, the core of Transformer, the self-attention mechanism, has a quadratic time and memory complexity with respect to the sequence length, which hinders applications of Transformer-based models to long sequences. Many approaches have been proposed to mitigate this problem, such as sparse attention mechanisms, low-rank matrix approximations and scalable kernels, and token mixing alternatives to self-attention. We propose a novel Pooling Network (PoNet) for token mixing in long sequences with linear complexity. We design multi-granularity pooling and pooling fusion to capture different levels of contextual information and combine their interactions with tokens. On the Long Range Arena benchmark, PoNet significantly outperforms Transformer and achieves competitive accuracy, while being only slightly slower than the fastest model, FNet, across all sequence lengths measured on GPUs. We also c
    
[^191]: 多个领域下通用的质心对齐和重构损失最小化领域归纳论文翻译

    Barycentric-alignment and reconstruction loss minimization for domain generalization. (arXiv:2109.01902v6 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2109.01902](http://arxiv.org/abs/2109.01902)

    本论文提出了一个新的理论上界，它不包含双重依赖性的术语，在领域归纳中优化了未见域的风险上界。

    

    本文推进了机器学习中领域归纳（DG）理论和实践。我们考虑了典型的DG设置，其中假设由表示映射和标记函数组成。在这个设置中，大多数流行的DG方法旨在通过最小化未见域中的分类风险的已知上界共同学习表示和标记函数。然而，在实践中，基于这个理论上界的方法忽略了一个由于其对表示映射和未知最优标记函数的双重依赖关系而无法直接优化的术语。为了弥合理论与实践之间的差距，我们引入了一个新的上界，它不包含这种双重依赖性的术语，从而产生了一个可以完全优化的未见域风险上界。我们的推导利用了将最优传输度量与信息相连的经典和最近的传输不等式。

    This paper advances the theory and practice of Domain Generalization (DG) in machine learning. We consider the typical DG setting where the hypothesis is composed of a representation mapping followed by a labeling function. Within this setting, the majority of popular DG methods aim to jointly learn the representation and the labeling functions by minimizing a well-known upper bound for the classification risk in the unseen domain. In practice, however, methods based on this theoretical upper bound ignore a term that cannot be directly optimized due to its dual dependence on both the representation mapping and the unknown optimal labeling function in the unseen domain. To bridge this gap between theory and practice, we introduce a new upper bound that is free of terms having such dual dependence, resulting in a fully optimizable risk upper bound for the unseen domain. Our derivation leverages classical and recent transport inequalities that link optimal transport metrics with informati
    
[^192]: 探索时空人群流量预测中的上下文泛化性：基准和指南

    Exploring the Context Generalizability in Spatiotemporal Crowd Flow Prediction: Benchmark and Guideline. (arXiv:2106.16046v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2106.16046](http://arxiv.org/abs/2106.16046)

    本文研究了时空人群流量预测中的上下文泛化性，建立了基准，提出了通用分类法，为上下文选择和建模提供了指南。

    

    上下文特征是构建时空人群流量预测（STCFP）模型的重要数据来源。然而，应用上下文的困难在于不同场景中上下文特征（例如天气、假日和兴趣点）和上下文建模技术的未知泛化性。本文建立了一个基准，由大规模时空人群流量数据、上下文数据和最先进的时空预测模型组成。我们在几个城市人群流量预测场景中进行了全面的实验研究，以定量研究不同上下文特征和建模技术的泛化性。特别地，我们基于对流行研究的广泛调查，开发了上下文建模技术的通用分类法。我们使用了数百万条记录和丰富的上下文数据，训练和测试了数百种模型以捕捉上下文泛化性。我们的研究为STCFP中的上下文选择和建模提供了指南。

    Contextual features are important data sources for building spatiotemporal crowd flow prediction (STCFP) models. However, the difficulty of applying context lies in the unknown generalizability of both contextual features (e.g., weather, holiday, and points of interests) and context modeling techniques across different scenarios. In this paper, we build a benchmark composed of large-scale spatiotemporal crowd flow data, contextual data, and state-of-the-art spatiotemporal prediction models. We conduct a comprehensive experimental study to quantitatively investigate the generalizability of different contextual features and modeling techniques in several urban crowd flow prediction scenarios (including bike flow, metro passenger flow, electric vehicle charging demand and so on). In particular, we develop a general taxonomy of context modeling techniques based on extensive investigations in prevailing research. With millions of records and rich context data, we have trained and tested hun
    
[^193]: 相互作用微粒平均场方程中的相互作用核可辨识性研究

    Identifiability of interaction kernels in mean-field equations of interacting particles. (arXiv:2106.05565v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2106.05565](http://arxiv.org/abs/2106.05565)

    本研究探索了相互作用微粒平均场方程中相互作用核的可辨识性问题，并确定了二次损失函数仅在特定的函数空间中才具有唯一的最小化器。此外，对于计算实践，研究证明了反问题的病态性质，需要进行正则化处理。

    

    本研究探讨了相互作用微粒或代理人的平均场方程中相互作用核的可辨识性问题，这是各种科学和工程领域日益关注的领域。主要关注点在于确定数据相关函数空间，其中二次损失函数拥有唯一的最小化器。我们考虑了两个数据自适应的$L^2$空间：一个是由数据自适应权重衡量的，另一个使用勒贝格测度。在每个$L^2$空间中，我们表明可辨识性的函数空间是与求积分算子相关的RKHS闭包。与之前的研究相辅相成，本研究完成了具有有限或无限微粒的相互作用微粒系统的可辨识性的全面描述，突显了这两种情况之间的关键差异。此外，可辨识性分析对于计算实践具有重要影响。它表明反问题是病态的，需要正则化处理。

    This study examines the identifiability of interaction kernels in mean-field equations of interacting particles or agents, an area of growing interest across various scientific and engineering fields. The main focus is identifying data-dependent function spaces where a quadratic loss functional possesses a unique minimizer. We consider two data-adaptive $L^2$ spaces: one weighted by a data-adaptive measure and the other using the Lebesgue measure. In each $L^2$ space, we show that the function space of identifiability is the closure of the RKHS associated with the integral operator of inversion.  Alongside prior research, our study completes a full characterization of identifiability in interacting particle systems with either finite or infinite particles, highlighting critical differences between these two settings. Moreover, the identifiability analysis has important implications for computational practice. It shows that the inverse problem is ill-posed, necessitating regularization.
    
[^194]: 将物理建模用于学习反应-扩散过程

    Encoding physics to learn reaction-diffusion processes. (arXiv:2106.04781v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2106.04781](http://arxiv.org/abs/2106.04781)

    通过将测量数据和物理知识的有限先验统一来进行机器学习提供了一种新的解决非线性反应-扩散过程的途径。

    

    建模复杂的时空动态系统，例如反应-扩散过程，大多依赖于偏微分方程(PDE)。然而，由于一些尚未充分探索的动态系统（如化学、生物、地质、物理和生态系统）缺乏先验知识，且用于描述系统变量非线性过程的PDE公式不明确，因此预测这种系统的演化仍然是一个具有挑战性的任务。通过机器学习将测量数据和有限的先验物理知识统一起来，为我们提供了解决这个问题的新途径。现有的物理信息学习范式通过软约束惩罚实施物理定律，其解决方案的质量主要取决于超参数的试错适当设置。由于这些方法的核心仍然根植于黑盒神经网络，所得模型通常缺乏可解释性并且存在外推的重要问题。

    Modeling complex spatiotemporal dynamical systems, such as the reaction-diffusion processes, have largely relied on partial differential equations (PDEs). However, due to insufficient prior knowledge on some under-explored dynamical systems, such as those in chemistry, biology, geology, physics and ecology, and the lack of explicit PDE formulation used for describing the nonlinear process of the system variables, to predict the evolution of such a system remains a challenging task. Unifying measurement data and our limited prior physics knowledge via machine learning provides us with a new path to solving this problem. Existing physics-informed learning paradigms impose physics laws through soft penalty constraints, whose solution quality largely depends on a trial-and-error proper setting of hyperparameters. Since the core of such methods is still rooted in black-box neural networks, the resulting model generally lacks interpretability and suffers from critical issues of extrapolation
    
[^195]: 联邦学习的碳足迹初探

    A first look into the carbon footprint of federated learning. (arXiv:2102.07627v6 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2102.07627](http://arxiv.org/abs/2102.07627)

    本文是对联邦学习的碳足迹进行的首次系统研究，发现FL的碳排放量最多可达传统学习的两倍以上。

    

    尽管深度学习技术取得了令人瞩目的成果，但由数据中心进行的训练过程也引起了严重的隐私和环境问题。针对这一问题，出现了集中式训练的替代方法，如联邦学习（FL）。然而，与FL相关的潜在环境影响尚不明确或未被探究。这篇论文针对FL的碳足迹进行了首次系统研究。我们首先提出了一个严格的模型来量化碳足迹，从而方便研究FL设计与碳排放之间的关系。接着，我们将FL的碳足迹与传统的集中式学习进行了比较。研究发现，根据不同的配置，FL的碳排放量最多可达传统学习的两倍以上。

    Despite impressive results, deep learning-based technologies also raise severe privacy and environmental concerns induced by the training procedure often conducted in data centers. In response, alternatives to centralized training such as Federated Learning (FL) have emerged. Perhaps unexpectedly, FL is starting to be deployed at a global scale by companies that must adhere to new legal demands and policies originating from governments and social groups advocating for privacy protection. \textit{However, the potential environmental impact related to FL remains unclear and unexplored. This paper offers the first-ever systematic study of the carbon footprint of FL.} First, we propose a rigorous model to quantify the carbon footprint, hence facilitating the investigation of the relationship between FL design and carbon emissions. Then, we compare the carbon footprint of FL to traditional centralized learning. Our findings show that, depending on the configuration, FL can emit up to two or
    
[^196]: 人工神经元中的与/或权衡：对抗鲁棒性影响的研究

    And/or trade-off in artificial neurons: impact on adversarial robustness. (arXiv:2102.07389v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2102.07389](http://arxiv.org/abs/2102.07389)

    本文研究了人工神经元中与/或函数的连续性对分类鲁棒性的影响，提出了增加类AND神经元在网络中比例的措施，实验结果显示该方法具有潜在的应用前景。

    

    尽管神经网络取得了很大的成功，但分类鲁棒性仍然是一个问题，特别是在对抗性样本上表现尤为突出。本文通过关注人工神经元中实现的函数连续性，从纯AND门到纯OR门的范围来解决这个挑战。我们的假设是，在网络中存在足够数量的类OR的神经元会导致分类的脆弱性和增加对抗攻击的脆弱性。我们定义了类AND的神经元，并提出了增加它们在网络中比例的措施。这些措施涉及将输入放缩到[-1,1]区间并减少S型激活函数陡峭部分的点数。我们方法的一个关键部分是比较当神经元用实际数据集和称为“乱序数据集”的随机版本输入时的输出分布。在MNIST数据集上的实验结果表明，我们的方法很有前途。

    Despite the success of neural networks, the issue of classification robustness remains, particularly highlighted by adversarial examples. In this paper, we address this challenge by focusing on the continuum of functions implemented in artificial neurons, ranging from pure AND gates to pure OR gates. Our hypothesis is that the presence of a sufficient number of OR-like neurons in a network can lead to classification brittleness and increased vulnerability to adversarial attacks. We define AND-like neurons and propose measures to increase their proportion in the network. These measures involve rescaling inputs to the [-1,1] interval and reducing the number of points in the steepest section of the sigmoidal activation function. A crucial component of our method is the comparison between a neuron's output distribution when fed with the actual dataset and a randomised version called the "scrambled dataset." Experimental results on the MNIST dataset suggest that our approach holds promise a
    
[^197]: 可微分决策树通过自然语言规范RL策略

    Natural Language Specification of Reinforcement Learning Policies through Differentiable Decision Trees. (arXiv:2101.07140v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2101.07140](http://arxiv.org/abs/2101.07140)

    本文提出了一个新的可微分决策树框架，允许人们通过自然语言指定初始行为模型，并将其转换为词汇决策树，为机器人的强化学习策略提供指导。

    

    本文提出了一种新的人工智能政策规范过程，可以使人类与机器人共同启动强化学习策略。该过程包含两个步骤：政策规范与政策优化。我们开发了一个新的协作框架，允许人通过非结构化自然语言指定初始行为模型，并将其转换为词汇决策树来启动和解释一个自主代理的行为。

    Human-AI policy specification is a novel procedure we define in which humans can collaboratively warm-start a robot's reinforcement learning policy. This procedure is comprised of two steps; (1) Policy Specification, i.e. humans specifying the behavior they would like their companion robot to accomplish, and (2) Policy Optimization, i.e. the robot applying reinforcement learning to improve the initial policy. Existing approaches to enabling collaborative policy specification are often unintelligible black-box methods, and are not catered towards making the autonomous system accessible to a novice end-user. In this paper, we develop a novel collaborative framework to allow humans to initialize and interpret an autonomous agent's behavior. Through our framework, we enable humans to specify an initial behavior model via unstructured, natural language (NL), which we convert to lexical decision trees. Next, we leverage these translated specifications, to warm-start reinforcement learning an
    
[^198]: 方差缩减的离策略TDC学习: 非渐进收敛分析

    Variance-Reduced Off-Policy TDC Learning: Non-Asymptotic Convergence Analysis. (arXiv:2010.13272v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2010.13272](http://arxiv.org/abs/2010.13272)

    本文提出了离策略TDC算法的方差缩减方案，并在i.i.d.和马尔可夫抽样中分析了其收敛率，结果表明该算法实现了与已知最佳下限相匹配的i.i.d.样本复杂度和接近最优的马尔可夫样本复杂度。

    

    方差缩减技术已成功应用于时间差分（TD）学习，并有助于提高策略评估的样本复杂度。然而，现有的工作要么将方差缩减应用于较不流行的单时间尺度TD算法，要么将其应用于两个时间尺度的GTD算法，但仅使用有限数目的i.i.d.样本，并且两种算法仅适用于在线策略设置。在这项工作中，我们为离策略设置中的两个时间尺度TDC算法开发了方差缩减方案，并分析了它在i.i.d.和马尔可夫抽样中的非渐进收敛率。在i.i.d.设置中，我们的算法与已知最佳下限相匹配$\tilde{O}(\epsilon^{-1}$)。在马尔可夫设置中，我们的算法实现了近乎最优的样本复杂度$O(\epsilon^{-1} \log {\epsilon}^{-1})$。实验表明，所提出的方差缩减TDC算法的渐近收敛误差比传统的TDC和方差缩减GTD算法都小。

    Variance reduction techniques have been successfully applied to temporal-difference (TD) learning and help to improve the sample complexity in policy evaluation. However, the existing work applied variance reduction to either the less popular one time-scale TD algorithm or the two time-scale GTD algorithm but with a finite number of i.i.d.\ samples, and both algorithms apply to only the on-policy setting. In this work, we develop a variance reduction scheme for the two time-scale TDC algorithm in the off-policy setting and analyze its non-asymptotic convergence rate over both i.i.d.\ and Markovian samples. In the i.i.d.\ setting, our algorithm {matches the best-known lower bound $\tilde{O}(\epsilon^{-1}$).} In the Markovian setting, our algorithm achieves the state-of-the-art sample complexity $O(\epsilon^{-1} \log {\epsilon}^{-1})$ that is near-optimal. Experiments demonstrate that the proposed variance-reduced TDC achieves a smaller asymptotic convergence error than both the conventi
    
[^199]: 深度动态因子模型

    Deep Dynamic Factor Models. (arXiv:2007.11887v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2007.11887](http://arxiv.org/abs/2007.11887)

    提出了一种深度神经网络框架，Deep Dynamic Factor Model (D$^2$FM)，能够将数百个宏观经济和金融时间序列可用信息编码为少量的未观察到的潜在状态，并能提高现在预测和预测实验中的表现。

    

    提出了一种新颖的深度神经网络框架——Deep Dynamic Factor Model (D$^2$FM)，能够将数百个宏观经济和金融时间序列可用信息编码为少量的未观察到的潜在状态。与传统动态因子模型（DFMs）类似，但由于自编码器神经网络结构，与观测值之间允许非线性关系。然而，设计上，模型的潜在状态仍可像标准因子模型那样进行解释。在使用美国数据进行全实时外样本现在预测和预测实验以及蒙特卡罗模拟中，D$^2$FM的表现优于最先进的DFM。

    A novel deep neural network framework -- that we refer to as Deep Dynamic Factor Model (D$^2$FM) --, is able to encode the information available, from hundreds of macroeconomic and financial time-series into a handful of unobserved latent states. While similar in spirit to traditional dynamic factor models (DFMs), differently from those, this new class of models allows for nonlinearities between factors and observables due to the autoencoder neural network structure. However, by design, the latent states of the model can still be interpreted as in a standard factor model. Both in a fully real-time out-of-sample nowcasting and forecasting exercise with US data and in a Monte Carlo experiment, the D$^2$FM improves over the performances of a state-of-the-art DFM.
    
[^200]: 利用神经架构搜索提升端到端语音识别效果

    Leveraging End-to-End Speech Recognition with Neural Architecture Search. (arXiv:1912.05946v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/1912.05946](http://arxiv.org/abs/1912.05946)

    本文研究表明，通过神经架构搜索可以在非常低的计算成本情况下显著提高深度语音模型的准确性，取得了与最先进结果相当的水平。

    

    深度神经网络已经被证实在自动语音识别方面优于许多传统机器学习算法。本文研究表明，通过有效实施神经架构搜索可以在非常低的计算成本情况下显著提高深度语音模型的准确性。在使用流行的LibriSpeech和TIMIT基准测试中进行的音素识别测试证明了这一事实，该方法能够在几个小时之内（不到一天），比基于注意力机制的seq2seq模型快多次，探测和训练新的候选模型。我们的方法在LibriSpeech语料库上的测试误差率（WER）为7％，在TIMIT语料库上的音素误差率（PER）为13％，达到了与最先进结果相当的水平。

    Deep neural networks (DNNs) have been demonstrated to outperform many traditional machine learning algorithms in Automatic Speech Recognition (ASR). In this paper, we show that a large improvement in the accuracy of deep speech models can be achieved with effective Neural Architecture Optimization at a very low computational cost. Phone recognition tests with the popular LibriSpeech and TIMIT benchmarks proved this fact by displaying the ability to discover and train novel candidate models within a few hours (less than a day) many times faster than the attention-based seq2seq models. Our method achieves test error of 7% Word Error Rate (WER) on the LibriSpeech corpus and 13% Phone Error Rate (PER) on the TIMIT corpus, on par with state-of-the-art results.
    
[^201]: SHARP：一种适应性和节能型的循环神经网络加速器

    SHARP: An Adaptable, Energy-Efficient Accelerator for Recurrent Neural Network. (arXiv:1911.01258v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1911.01258](http://arxiv.org/abs/1911.01258)

    论文提出了一种适应性和节能型的循环神经网络加速器，通过智能的瓦片分派机制，实现了处理不同RNN维度的高计算效率。

    

    循环神经网络（RNN）在自动语音识别等任务中的有效性引发了对RNN推理加速的兴趣。由于RNN计算的递归性和数据依赖性，先前的工作设计了专门针对RNN计算模式的定制化架构，为某些选择的模型大小获得了高计算效率。然而，鉴于RNN的维度对于不同任务变化很大，将这种效率推广到各种不同配置非常关键。在这项工作中，我们认为适应性是当今RNN加速器所缺少的关键特征。特别地，我们首先显示了现有的GPU、FPGA和ASIC架构的最新RNN实现在资源利用率和适应性方面存在问题。为了解决这些问题，我们提出了一种智能的基于瓦片的分派机制，以增加RNN计算的适应性，以便有效地处理数据。

    The effectiveness of Recurrent Neural Networks (RNNs) for tasks such as Automatic Speech Recognition has fostered interest in RNN inference acceleration. Due to the recurrent nature and data dependencies of RNN computations, prior work has designed customized architectures specifically tailored to the computation pattern of RNN, getting high computation efficiency for certain chosen model sizes. However, given that the dimensionality of RNNs varies a lot for different tasks, it is crucial to generalize this efficiency to diverse configurations. In this work, we identify adaptiveness as a key feature that is missing from today's RNN accelerators. In particular, we first show the problem of low resource-utilization and low adaptiveness for the state-of-the-art RNN implementations on GPU, FPGA and ASIC architectures. To solve these issues, we propose an intelligent tiled-based dispatching mechanism for increasing the adaptiveness of RNN computation, in order to efficiently handle the data
    
[^202]: 过拟合、交叉验证、正则化、装袋法和提升法背后的理论：教程

    The Theory Behind Overfitting, Cross Validation, Regularization, Bagging, and Boosting: Tutorial. (arXiv:1905.12787v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/1905.12787](http://arxiv.org/abs/1905.12787)

    该论文介绍了过拟合、交叉验证、正则化、装袋法和提升法的相关理论，包括定义和具体实现，并给出了AdaBoost的泛化误差上限的具体计算方法。

    

    在这篇教程性的论文中，我们首先定义了随机变量和分类/预测模型的均方误差、方差、协方差和偏差。然后，我们利用Stein的无偏风险估计器（SURE）制定了模型的真实和泛化误差，包括训练和验证/测试实例。利用得到的真实和泛化误差，我们定义了过拟合、欠拟合和泛化。我们介绍了交叉验证和两个著名的例子，即K倍交叉验证和留一法交叉验证。我们简要介绍了广义交叉验证，然后转向正则化，在这里我们再次使用SURE。我们对$\ell_2$和$\ell_1$范数正则化进行了研究。然后，我们展示了自举聚合（bagging）如何降低估计方差。我们介绍了提升法，特别是AdaBoost，并解释了它作为一个加性模型和最大间隔模型（即支持向量机（SVM））的原理。给出了AdaBoost的泛化误差上限，包括指数损失和0-1损失。最后，我们总结了教程的主要内容。

    In this tutorial paper, we first define mean squared error, variance, covariance, and bias of both random variables and classification/predictor models. Then, we formulate the true and generalization errors of the model for both training and validation/test instances where we make use of the Stein's Unbiased Risk Estimator (SURE). We define overfitting, underfitting, and generalization using the obtained true and generalization errors. We introduce cross validation and two well-known examples which are $K$-fold and leave-one-out cross validations. We briefly introduce generalized cross validation and then move on to regularization where we use the SURE again. We work on both $\ell_2$ and $\ell_1$ norm regularizations. Then, we show that bootstrap aggregating (bagging) reduces the variance of estimation. Boosting, specifically AdaBoost, is introduced and it is explained as both an additive model and a maximum margin model, i.e., Support Vector Machine (SVM). The upper bound on the gener
    
[^203]: 特征值和广义特征值问题：教程

    Eigenvalue and Generalized Eigenvalue Problems: Tutorial. (arXiv:1903.11240v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/1903.11240](http://arxiv.org/abs/1903.11240)

    本文是一篇阐述特征值和广义特征值问题的教程。特征值和广义特征值问题可用于各种机器学习算法中，如主成分分析和Fisher判别分析等。

    

    本文是特征值和广义特征值问题的教程。我们首先介绍了特征值问题、特征值分解（谱分解）和广义特征值问题。然后，我们提到了导致特征值和广义特征值问题的优化问题。我们还提供了机器学习中的例子，包括主成分分析、核监督主成分分析和Fisher判别分析，这些方法都会导致特征值和广义特征值问题。最后，我们介绍了解决特征值和广义特征值问题的方法。

    This paper is a tutorial for eigenvalue and generalized eigenvalue problems. We first introduce eigenvalue problem, eigen-decomposition (spectral decomposition), and generalized eigenvalue problem. Then, we mention the optimization problems which yield to the eigenvalue and generalized eigenvalue problems. We also provide examples from machine learning, including principal component analysis, kernel supervised principal component analysis, and Fisher discriminant analysis, which result in eigenvalue and generalized eigenvalue problems. Finally, we introduce the solutions to both eigenvalue and generalized eigenvalue problems.
    
[^204]: 基于类结构的最优传输恢复界限: 一种范数求和正则化框架

    Recovery Bounds on Class-Based Optimal Transport: A Sum-of-Norms Regularization Framework. (arXiv:1903.03850v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1903.03850](http://arxiv.org/abs/1903.03850)

    该论文提出了一个基于范数求和正则化项的凸性最优传输程序，在几何假设条件下可证明恢复基础类结构。该论文还提供了一种加速的近端算法，并提出了一种新的唯一性优化方式。实验表明，新的正则化程序不仅可以更好地保留数据中的类结构，还可以在数据几何形状方面提供额外的鲁棒性。

    

    我们开展了一个新的理论框架，用于理解尊重类结构的最优传输方案。为此，我们提出了一个带有范数求和正则化项的凸性最优传输程序，该程序在几何假设条件下可证明恢复基础类结构。此外，我们推导出一种加速的近端算法，该算法具有闭式投影和近端操作符方案，从而为计算最优传输计划提供了更可扩展的算法。我们提供了一种新的唯一性优化方式，即使在缺乏强凸性的情况下，也可得到最优解。我们的实验表明，与以前的正则化程序相比，新的正则化程序不仅可以更好地保留数据中的类结构，还可以在数据几何形状方面提供额外的鲁棒性。

    We develop a novel theoretical framework for understating OT schemes respecting a class structure. For this purpose, we propose a convex OT program with a sum-of-norms regularization term, which provably recovers the underlying class structure under geometric assumptions. Furthermore, we derive an accelerated proximal algorithm with a closed-form projection and proximal operator scheme, thereby affording a more scalable algorithm for computing optimal transport plans. We provide a novel argument for the uniqueness of the optimum even in the absence of strong convexity. Our experiments show that the new regularizer not only results in a better preservation of the class structure in the data but also yields additional robustness to the data geometry, compared to previous regularizers.
    

