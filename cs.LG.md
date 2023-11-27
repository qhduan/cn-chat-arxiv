# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion.](http://arxiv.org/abs/2311.01017) | 本论文提出了一种通过离散扩散学习无监督的自动驾驶世界模型的新方法，通过使用VQVAE对传感器观察进行标记化并通过离散扩散预测未来，我们的模型在点云观察中实现了显著改进，将1秒预测的SOTA Chamfer距离降低了65%以上。 |
| [^2] | [Zero Coordinate Shift: Whetted Automatic Differentiation for Physics-informed Operator Learning.](http://arxiv.org/abs/2311.00860) | 本文提出了一种用于物理约束操作学习的新型自动微分算法，通过零坐标移动（ZCS）的技巧，将所需导数的复杂度从“多根多叶”简化为“一根多叶”，从而显著提高了性能。 |
| [^3] | [Upgrading VAE Training With Unlimited Data Plans Provided by Diffusion Models.](http://arxiv.org/abs/2310.19653) | 这项研究通过在预训练的扩散模型生成的样本上进行训练，有效减轻了VAE中编码器的过拟合问题。 |
| [^4] | [Course Correcting Koopman Representations.](http://arxiv.org/abs/2310.15386) | 本文修正了Koopman表示的方法，并提出了一种称为“周期重新编码”的机制，用于准确捕捉非线性动力系统中的长期动态。 |
| [^5] | [Physics-Informed Graph Convolutional Networks: Towards a generalized framework for complex geometries.](http://arxiv.org/abs/2310.14948) | 本研究提出了物理信息图卷积网络作为解决复杂几何体中偏微分方程问题的广义框架，并结合经典数值求解器从而解决了物理信息框架在复杂几何体上的问题。 |
| [^6] | [Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models.](http://arxiv.org/abs/2310.13102) | 本文提出了一种粒子引导的方法，通过超越独立样本的常见假设，提高了生成模型的多样性和采样效率。在实验中，我们在条件图像生成和分子构象生成上进行了测试，并取得了显著的结果。 |
| [^7] | [Sensitivity-Aware Amortized Bayesian Inference.](http://arxiv.org/abs/2310.11122) | 本文提出了一种敏感性感知的摊销贝叶斯推断方法，通过权重共享和神经网络来进行似然和先验规范的训练，以及对数据扰动和预处理程序的敏感性评估。 |
| [^8] | [SD-PINN: Deep Learning based Spatially Dependent PDEs Recovery.](http://arxiv.org/abs/2310.10970) | SD-PINN是一种基于深度学习的方法，能够通过一个神经网络恢复空间相关的偏微分方程（PDE）系数，无需领域特定的物理专业知识，并且对噪声具有稳健性。同时，它能够通过空间变化低秩假设恢复没有可用测量数据的位置的系数。 |
| [^9] | [Grokking as the Transition from Lazy to Rich Training Dynamics.](http://arxiv.org/abs/2310.06110) | 研究发现洞察现象可能是由神经网络从懒惰训练动态过渡到丰富的特征学习模式的结果，通过跟踪足够的统计量，发现洞察是在网络首先尝试拟合核回归解决方案后，进行后期特征学习找到通用解决方案之后的结果。 |
| [^10] | [Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts.](http://arxiv.org/abs/2310.05898) | Lion是通过程序搜索发现的新优化器，在训练大型AI模型方面表现出有希望的结果，具有更高的内存效率。尽管其理论基础不明确，但基于连续时间和离散时间分析，我们证明Lion是一种理论上新颖且有原则的方法，可在最小化一般损失函数的同时强制执行边界约束。 |
| [^11] | [Learning Intra- and Inter-Cell Differences for Accurate Battery Lifespan Prediction across Diverse Conditions.](http://arxiv.org/abs/2310.05052) | 该论文介绍了一种跨不同条件精确预测电池寿命的方法，通过捕捉目标电池和参考电池之间的电信号差异，无论材料和老化条件如何，在扩展特征空间的同时为通用的电池寿命预测框架铺平了道路。 |
| [^12] | [Fast, Expressive SE$(n)$ Equivariant Networks through Weight-Sharing in Position-Orientation Space.](http://arxiv.org/abs/2310.02970) | 该论文通过在位置-方向空间中共享权重，提出了一种快速、表达力强的SE$(n)$等变网络。他们基于同态空间理论，推导出几何优化的边属性，并将权重共享形式化为对等处理相同点对的消息函数。他们在处理3D点云时，开发了一个高效的等变群卷积网络，并选择了$\mathbb{R}^3 {\times} S^2$作为最佳的处理空间。 |
| [^13] | [EGraFFBench: Evaluation of Equivariant Graph Neural Network Force Fields for Atomistic Simulations.](http://arxiv.org/abs/2310.02428) | EGraFFBench对六种EGraFF算法进行了系统的基准测试，以评估其在原子模拟中的性能，并提出了新的数据集和度量标准。 |
| [^14] | [How Over-Parameterization Slows Down Gradient Descent in Matrix Sensing: The Curses of Symmetry and Initialization.](http://arxiv.org/abs/2310.01769) | 该论文研究了过参数化如何影响矩阵感知问题中梯度下降的收敛行为，在对称和非对称设置下给出了不同的收敛速度。 |
| [^15] | [A path-norm toolkit for modern networks: consequences, promises and challenges.](http://arxiv.org/abs/2310.01225) | 本文介绍了适用于现代神经网络的路径范数工具包，可以包括具有偏差、跳跃连接和最大池化的通用DAG ReLU网络。这个工具包恢复或超越了已知的路径范数界限，并挑战了基于路径范数的一些具体承诺。 |
| [^16] | [The Noise Geometry of Stochastic Gradient Descent: A Quantitative and Analytical Characterization.](http://arxiv.org/abs/2310.00692) | 本文对随机梯度下降（SGD）中的噪声几何进行了全面的理论研究，发现噪声与损失函数的局部几何特征有利的一致性。通过实验证明，SGD在逃脱尖锐极小值时与GD形成鲜明对比，逃脱方向在平坦方向上有显著分量。 |
| [^17] | [SIMD Dataflow Co-optimization for Efficient Neural Networks Inferences on CPUs.](http://arxiv.org/abs/2310.00574) | 我们提出了一种通过共优化数据流和SIMD实现来高效地在CPU上进行神经网络推理的方法，实验结果表明，这种方法能够在保持准确性的同时大幅提升推理速度。 |
| [^18] | [Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization.](http://arxiv.org/abs/2310.00116) | 本文提出了一种基于动态边界最大化和改进的Lipschitz正则化的认证鲁棒性训练算法，通过增加输出空间中的边界和正则化模型的Lipschitz常数来提高深度分类器对抗性扰动的鲁棒性。 |
| [^19] | [Navigating the Design Space of Equivariant Diffusion-Based Generative Models for De Novo 3D Molecule Generation.](http://arxiv.org/abs/2309.17296) | 本论文探索了E(3)等变扩散模型的设计空间，提出了EQGAT-diff模型，通过在连续的原子位置和分类的化学元素与键类型之间的交互中改进，其在从头设计3D分子方面的性能显著超过已有模型。 |
| [^20] | [Why do Angular Margin Losses work well for Semi-Supervised Anomalous Sound Detection?.](http://arxiv.org/abs/2309.15643) | 角边距损失与辅助任务结合在半监督异常声音检测中表现出色，通过最小化角边距损失同时达到最小化紧凑性损失和防止学习平凡解的效果。 |
| [^21] | [Associative Transformer Is A Sparse Representation Learner.](http://arxiv.org/abs/2309.12862) | 关联变换器（AiT）是一种采用低秩显式记忆和关联记忆的稀疏表示学习器，通过联合端到端训练实现模块特化和注意力瓶颈的形成。 |
| [^22] | [Mitigating Over-Smoothing and Over-Squashing using Augmentations of Forman-Ricci Curvature.](http://arxiv.org/abs/2309.09384) | 本文提出了一种使用Forman-Ricci曲率扩展的方法来减轻图神经网络中的过度平滑和过度压缩问题。通过观察离散曲率，可以添加或删除边以减轻这两种效应。 |
| [^23] | [Symplectic Structure-Aware Hamiltonian (Graph) Embeddings.](http://arxiv.org/abs/2309.04885) | 本文提出了SAH-GNN，一种在图神经网络中应用结构感知的辛系统哈密顿嵌入方法。与传统方法不同，SAH-GNN通过在训练过程中自适应学习辛结构，避免了依赖预定义标准辛结构形式的限制，并能够适应不同的图数据集，同时保持物理意义上的能量守恒。 |
| [^24] | [Echocardiographic View Classification with Integrated Out-of-Distribution Detection for Enhanced Automatic Echocardiographic Analysis.](http://arxiv.org/abs/2308.16483) | ECHO-VICODE是一个深度学习框架，通过训练来分类超声心动图的31个视图类别，并具有集成的离群检测功能，可以显著降低超声心动图中的错误可能性。 |
| [^25] | [Karasu: A Collaborative Approach to Efficient Cluster Configuration for Big Data Analytics.](http://arxiv.org/abs/2308.11792) | Karasu是一种通过促进类似基础设施、框架、算法或数据集上工作的用户之间的数据共享，实现更高效的资源配置配置文件分析的方法。 |
| [^26] | [Graph of Thoughts: Solving Elaborate Problems with Large Language Models.](http://arxiv.org/abs/2308.09687) | 想法图（GoT）是一种新的框架，它超越了现有的提示范式，通过将大型语言模型（LLM）的信息建模为任意图形，将LLM想法组合成具有协同效应的结果，提炼整个思维网络的本质，或者使用反馈环路增强思维。GoT在不同任务上展示出优势，并可以通过新的想法转换进行扩展，使LLM的推理更接近人类思维。 |
| [^27] | [Kernel-Based Tests for Likelihood-Free Hypothesis Testing.](http://arxiv.org/abs/2308.09043) | 本文介绍了一种基于核的无似然假设检验方法，解决了对已知属于两个类别的输入进行分类的问题，在无似然推断领域，通过将标记样本通过正向模拟获得，未标记样本通过实验收集，给出了一个权衡m和n的方法。 |
| [^28] | [On Neural Quantum Support Vector Machines.](http://arxiv.org/abs/2308.08467) | 本文介绍了神经量子支持向量机，利用量子核，扩展了神经支持向量机的训练算法。 |
| [^29] | [Real Robot Challenge 2022: Learning Dexterous Manipulation from Offline Data in the Real World.](http://arxiv.org/abs/2308.07741) | Real Robot Challenge 2022为RL和机器人学界之间的桥梁，允许参与者在真实机器人上从离线数据中学习灵巧操作任务，解决了在模拟中得到的见解不能转化到真实机器人的问题。 |
| [^30] | [The SocialAI School: Insights from Developmental Psychology Towards Artificial Socio-Cultural Agents.](http://arxiv.org/abs/2307.07871) | 该论文讨论了AI研究应该受发展心理学启发，并研究使代理能够进入文化的社会认知能力。提出了社会AI学校工具以便于进行相关实验。 |
| [^31] | [LeCo: Lightweight Compression via Learning Serial Correlations.](http://arxiv.org/abs/2306.15374) | 使用机器学习来自动消除序列冗余以实现出色的压缩比和解压缩性能的LeCo是一种轻量级压缩框架，通过学习序列相关性，它能够在压缩比和随机访问速度上实现帕累托改进。 |
| [^32] | [Equivariant flow matching.](http://arxiv.org/abs/2306.15030) | 本文介绍了一种基于最优输运流匹配的等变CNF训练目标，可以提高等变CNF的可扩展性和实际应用。 |
| [^33] | [Review of compressed embedding layers and their applications for recommender systems.](http://arxiv.org/abs/2306.13724) | 论文综述了可训练的、压缩的嵌入层在压缩大型神经网络推荐系统中的应用，并提供了相关实验结果。 |
| [^34] | [Broadening the perspective for sustainable AI: Comprehensive sustainability criteria and indicators for AI systems.](http://arxiv.org/abs/2306.13686) | 本文提出了SCAIS框架，包含一组19个可持续性标准和67个指标，旨在促进和结构化关于可持续人工智能的讨论。这种跨学科方法为实现人工智能系统的可持续发展提供了基础。 |
| [^35] | [Task-Robust Pre-Training for Worst-Case Downstream Adaptation.](http://arxiv.org/abs/2306.12070) | 本文提出了一种任务鲁棒的预训练方法，将上游任务分成几个代表性任务并应用极小极大损失进行预训练，以保证模型能够在下游任务中具有均匀良好的性能。 |
| [^36] | [A Bayesian Take on Gaussian Process Networks.](http://arxiv.org/abs/2306.11380) | 该论文提出了一种基于高斯过程和贝叶斯方法的网络模型，通过蒙特卡罗和马尔可夫链蒙特卡罗方法采样网络结构的后验分布。该方法在恢复网络的图形结构方面优于最先进的算法，并提供了后验概率的准确近似。 |
| [^37] | [Blockchain-Enabled Federated Learning: A Reference Architecture Design, Implementation, and Verification.](http://arxiv.org/abs/2306.10841) | 本文提出了一种基于区块链的联邦学习参考架构，通过结合联邦学习和区块链技术，实现了去中心化、协作的机器学习系统，并保护了数据隐私和用户控制的身份。该架构使用去中心化标识符进行身份验证，通过智能合约实现强大的安全性和高效的去中心化，并能根据需求集成各种额外的元素，是一个适用范围广泛的 BCFL 解决方案。 |
| [^38] | [MARBLE: Music Audio Representation Benchmark for Universal Evaluation.](http://arxiv.org/abs/2306.10548) | 本论文介绍了MARBLE，一个音乐音频表征通用评估基准，它为音乐理解领域的研究和发展提供了一个全面和可持续性的基础，并提供各种音乐信息检索（MIR）任务的基准。 |
| [^39] | [PEAR: Primitive enabled Adaptive Relabeling for boosting Hierarchical Reinforcement Learning.](http://arxiv.org/abs/2306.06394) | PEAR是一种基于原始操作的自适应重标记方法，用于Boosting层次强化学习。它通过对专家演示进行自适应重标记来生成高效的子目标监督，并通过联合优化强化学习和模仿学习来训练分层代理。实验结果显示，PEAR能够在具有挑战性的机器人环境中取得良好的性能。 |
| [^40] | [Neural Algorithmic Reasoning for Combinatorial Optimisation.](http://arxiv.org/abs/2306.06064) | 本文提出了一种用于组合优化问题的神经算法推理方法，旨在解决旅行商问题。该方法是通过在TSP实例训练之前，将神经模型用相关算法进行预训练来实现的。实验结果表明，该方法可以显著提高TSP问题的解决效率。 |
| [^41] | [Inverse Approximation Theory for Nonlinear Recurrent Neural Networks.](http://arxiv.org/abs/2305.19190) | 该论文证明了使用RNNs逼近非线性序列关系的逆近似定理，进一步将先前在线性RNNs中识别出的记忆难题推广到了一般的非线性情况，并提出了一个有原则的重新参数化方法来克服这些限制。 |
| [^42] | [Explainable Machine Learning for Categorical and Mixed Data with Lossless Visualization.](http://arxiv.org/abs/2305.18437) | 本文提出了一些数值编码和可视化方法，以支持机器学习算法处理混合数据，并提出了可解释的多分类模型和SRG算法来生成解释性分类模型。 |
| [^43] | [FAVAS: Federated AVeraging with ASynchronous clients.](http://arxiv.org/abs/2305.16099) | 本研究提出了FAVAS算法，是一种用于在资源有限环境下训练DNNs的新型中心化异步联邦学习框架。实验结果表明FAVAS算法优于当前方法。 |
| [^44] | [Collective Relational Inference for learning physics-consistent heterogeneous particle interactions.](http://arxiv.org/abs/2305.00557) | 本论文提出了一种新的概率方法用于学习异质性粒子相互作用的集体关系推断，与现有方法相比，该方法集体地推断不同边的相互作用类型，使用物理感应的图神经网络来学习具有物理一致性的成对相互作用，并在推断准确性和保持物理保真度方面一致优于现有方法。 |
| [^45] | [Towards Quantifying Calibrated Uncertainty via Deep Ensembles in Multi-output Regression Task.](http://arxiv.org/abs/2303.16210) | 本研究探究了在多输出回归任务中应用深度集合量化校准不确定性的方法，提出了该方法的改进框架，其在回归准确性、不确定性估计可靠性和训练效率方面具有优越表现。 |
| [^46] | [Spatio-Temporal Graph Neural Networks for Predictive Learning in Urban Computing: A Survey.](http://arxiv.org/abs/2303.14483) | 本综述介绍了面向城市计算的时空图神经网络预测学习领域的发展现状，包括其框架、实现方法和应用场景，以及当前的研究热点和挑战，提出了该领域未来的发展方向和应用前景。 |
| [^47] | [Large statistical learning models effectively forecast diverse chaotic systems.](http://arxiv.org/abs/2303.08011) | 该论文研究了混沌预测的大规模实验，发现基于人工神经网络的大规模、领域不可知的时间序列预测方法表现出了相当强大的性能，尤其是分层神经基础函数模型表现最佳。 |
| [^48] | [Interpretable and Intervenable Ultrasonography-based Machine Learning Models for Pediatric Appendicitis.](http://arxiv.org/abs/2302.14460) | 本研究开发了可解释的机器学习模型，利用超声影像预测儿科疑似阑尾炎的诊断、管理和严重程度。模型使用了超声影像和临床、实验室数据进行训练，并推广了概念瓶颈模型到多视图和不完整概念集的预测问题。 |
| [^49] | [A Deep Learning Method for Comparing Bayesian Hierarchical Models.](http://arxiv.org/abs/2301.11873) | 这个论文提出了一种深度学习方法，用于比较贝叶斯层次模型。该方法通过支持分摊推断，能够高效地进行模型比较和性能验证。同时，作者还对四个层次证据积累模型进行了比较。 |
| [^50] | [Bounding Box-based Multi-objective Bayesian Optimization of Risk Measures under Input Uncertainty.](http://arxiv.org/abs/2301.11588) | 该论文提出了一种基于边界框的多目标贝叶斯优化方法，能够在输入不确定性下高效地识别风险衡量定义的帕累托前沿。该方法具有理论保证，并通过构建高概率边界框和选择下一个评估点的方法来减少不确定性。 |
| [^51] | [Visual Dexterity: In-hand Dexterous Manipulation from Depth.](http://arxiv.org/abs/2211.11744) | 通过使用深度相机的读数，我们提出了一种通用物体重新定向控制器，可以实时、动态地重新定向复杂和新颖的物体形状，中位数重新定向时间接近于七秒。该控制器经过强化学习在仿真环境中训练，并在实际世界中对未用于训练的新物体形状进行了评估。 |
| [^52] | [Supervised Feature Compression based on Counterfactual Analysis.](http://arxiv.org/abs/2211.09894) | 该论文提出了一种基于反事实分析的监督特征压缩方法，利用此方法可以构建出类似于黑盒模型最优决策树，该决策树具备可解释性和紧凑性，并在真实数据集上有效。 |
| [^53] | [Automatically Score Tissue Images Like a Pathologist by Transfer Learning.](http://arxiv.org/abs/2209.05954) | 该算法通过选择性迁移学习从多个小辅助集中提取知识，从具有“相似”特征的组织图像中学习染色模式，以实现像病理学家一样自动评分组织图像的目标。 |
| [^54] | [FAIR4Cov: Fused Audio Instance and Representation for COVID-19 Detection.](http://arxiv.org/abs/2204.10581) | FAIR4Cov是一种针对COVID-19检测的方法，它提出了一种融合身体声音的波形和谱图表示的关节特征向量，可以有效地检测COVID-19患者，胜过了其他方法。 |
| [^55] | [Networked Time Series Prediction with Incomplete Data.](http://arxiv.org/abs/2110.02271) | 本文研究了具有不完整数据的网络时间序列（NETS）预测问题。提出了NETS-ImpGAN深度学习框架，可以训练不完整数据，并引入了图时序注意力网络来捕捉时间序列之间的相关性和时间相关性。 |
| [^56] | [On the Tightness of the Moment Accountant for DP-SGD.](http://arxiv.org/abs/2102.09030) | 通过改进Moment Accountant方法，DP-SGD具有可关闭形式的$(\epsilon，\delta)$-DP保证，并且其保证接近是紧密的，具有最小的计算成本。 |

# 详细

[^1]: 通过离散扩散学习无监督的自动驾驶世界模型

    Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion. (arXiv:2311.01017v1 [cs.CV])

    [http://arxiv.org/abs/2311.01017](http://arxiv.org/abs/2311.01017)

    本论文提出了一种通过离散扩散学习无监督的自动驾驶世界模型的新方法，通过使用VQVAE对传感器观察进行标记化并通过离散扩散预测未来，我们的模型在点云观察中实现了显著改进，将1秒预测的SOTA Chamfer距离降低了65%以上。

    

    学习世界模型可以以无监督的方式教会智能体世界的运作方式。尽管它可以看作是序列建模的特殊情况，但在自动驾驶等机器人应用中，与使用生成预训练转换器（GPT）扩展语言模型相比，扩展世界模型的进展相对较慢。我们指出了两个主要瓶颈：处理复杂和无结构的观察空间以及具有可扩展性的生成模型。因此，我们提出了一种新颖的世界建模方法，首先使用VQVAE对传感器观察进行标记化，然后通过离散扩散预测未来。为了有效地并行解码和去噪标记，我们将遮蔽生成图像转换器转换为离散扩散框架，并进行了一些简单的改进，取得了显着的改进效果。当应用于点云观察的世界模型学习时，我们的模型将1秒预测的SOTA Chamfer距离降低了65%以上。

    Learning world models can teach an agent how the world works in an unsupervised manner. Even though it can be viewed as a special case of sequence modeling, progress for scaling world models on robotic applications such as autonomous driving has been somewhat less rapid than scaling language models with Generative Pre-trained Transformers (GPT). We identify two reasons as major bottlenecks: dealing with complex and unstructured observation space, and having a scalable generative model. Consequently, we propose a novel world modeling approach that first tokenizes sensor observations with VQVAE, then predicts the future via discrete diffusion. To efficiently decode and denoise tokens in parallel, we recast Masked Generative Image Transformer into the discrete diffusion framework with a few simple changes, resulting in notable improvement. When applied to learning world models on point cloud observations, our model reduces prior SOTA Chamfer distance by more than 65% for 1s prediction, an
    
[^2]: 零坐标移动：针对物理约束操作学习的优化自动微分方法

    Zero Coordinate Shift: Whetted Automatic Differentiation for Physics-informed Operator Learning. (arXiv:2311.00860v1 [cs.LG])

    [http://arxiv.org/abs/2311.00860](http://arxiv.org/abs/2311.00860)

    本文提出了一种用于物理约束操作学习的新型自动微分算法，通过零坐标移动（ZCS）的技巧，将所需导数的复杂度从“多根多叶”简化为“一根多叶”，从而显著提高了性能。

    

    自动微分（AD）是物理约束机器学习中的关键步骤，用于计算网络输出相对于坐标的高阶导数。本文提出了一种新颖且轻量级的算法，用于进行针对物理约束操作学习的自动微分，称为零坐标移动（ZCS）的技巧。ZCS引入了一个标量值的叶变量，用于每个空间或时间维度，通过将所需导数从“多根多叶”简化为“一根多叶”，从而实现了性能的巨大提升。ZCS很容易在当前的深度学习库中实现；我们使用DeepXDE软件包进行了自己的实现。我们进行了全面的基准分析和多个案例研究，训练物理约束的DeepONets来解决无数据的偏微分方程（PDE）。结果表明，ZCS一直通过降低GPU内存消耗提供了改进效果。

    Automatic differentiation (AD) is a critical step in physics-informed machine learning, required for computing the high-order derivatives of network output w.r.t. coordinates. In this paper, we present a novel and lightweight algorithm to conduct such AD for physics-informed operator learning, as we call the trick of Zero Coordinate Shift (ZCS). Instead of making all sampled coordinates leaf variables, ZCS introduces only one scalar-valued leaf variable for each spatial or temporal dimension, leading to a game-changing performance leap by simplifying the wanted derivatives from "many-roots-many-leaves" to "one-root-many-leaves". ZCS is easy to implement with current deep learning libraries; our own implementation is by extending the DeepXDE package. We carry out a comprehensive benchmark analysis and several case studies, training physics-informed DeepONets to solve partial differential equations (PDEs) without data. The results show that ZCS has persistently brought down GPU memory co
    
[^3]: 使用扩散模型提供的无限数据计划升级VAE训练

    Upgrading VAE Training With Unlimited Data Plans Provided by Diffusion Models. (arXiv:2310.19653v1 [stat.ML])

    [http://arxiv.org/abs/2310.19653](http://arxiv.org/abs/2310.19653)

    这项研究通过在预训练的扩散模型生成的样本上进行训练，有效减轻了VAE中编码器的过拟合问题。

    

    变分自编码器（VAE）是一种常用的表示学习模型，但其编码器容易过拟合，因为它们是在有限的训练集上进行训练，而不是真实（连续）数据分布$p_{\mathrm{data}}(\mathbf{x})$。与之相反，扩散模型通过固定编码器避免了这个问题。这使得它们的表示不太可解释，但简化了训练，可以精确和连续地逼近$p_{\mathrm{data}}(\mathbf{x})$。在本文中，我们展示了通过在预训练的扩散模型生成的样本上训练，可以有效减轻VAE中编码器的过拟合问题。这些结果有些出人意料，因为最近的研究发现，在使用另一个生成模型生成的数据上训练时，生成性能会下降。我们分析了使用我们的方法训练的VAE的泛化性能、分摊差距和鲁棒性。

    Variational autoencoders (VAEs) are popular models for representation learning but their encoders are susceptible to overfitting (Cremer et al., 2018) because they are trained on a finite training set instead of the true (continuous) data distribution $p_{\mathrm{data}}(\mathbf{x})$. Diffusion models, on the other hand, avoid this issue by keeping the encoder fixed. This makes their representations less interpretable, but it simplifies training, enabling accurate and continuous approximations of $p_{\mathrm{data}}(\mathbf{x})$. In this paper, we show that overfitting encoders in VAEs can be effectively mitigated by training on samples from a pre-trained diffusion model. These results are somewhat unexpected as recent findings (Alemohammad et al., 2023; Shumailov et al., 2023) observe a decay in generative performance when models are trained on data generated by another generative model. We analyze generalization performance, amortization gap, and robustness of VAEs trained with our pro
    
[^4]: 修正Koopman表示的方法

    Course Correcting Koopman Representations. (arXiv:2310.15386v1 [cs.LG])

    [http://arxiv.org/abs/2310.15386](http://arxiv.org/abs/2310.15386)

    本文修正了Koopman表示的方法，并提出了一种称为“周期重新编码”的机制，用于准确捕捉非线性动力系统中的长期动态。

    

    Koopman表示旨在学习非线性动力系统中导致潜在空间线性动力学的特征。从理论上讲，这些特征可以用于简化非线性动力系统建模和控制中的许多问题。在本文中，我们研究了此问题的自动编码器方法，并探讨了它们在建模动力学方面的不同应用，特别是在长期预测未来状态方面。我们发现在潜在空间中预测未来状态存在一些限制，并提出了一种称为“周期重新编码”的推理时间机制，以实现长期动态的准确捕捉。我们通过在低维和高维非线性动力系统上的实验证明了该方法的合理性和实用性。

    Koopman representations aim to learn features of nonlinear dynamical systems (NLDS) which lead to linear dynamics in the latent space. Theoretically, such features can be used to simplify many problems in modeling and control of NLDS. In this work we study autoencoder formulations of this problem, and different ways they can be used to model dynamics, specifically for future state prediction over long horizons. We discover several limitations of predicting future states in the latent space and propose an inference-time mechanism, which we refer to as Periodic Reencoding, for faithfully capturing long term dynamics. We justify this method both analytically and empirically via experiments in low and high dimensional NLDS.
    
[^5]: 物理信息图卷积网络：面向复杂几何的广义框架

    Physics-Informed Graph Convolutional Networks: Towards a generalized framework for complex geometries. (arXiv:2310.14948v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.14948](http://arxiv.org/abs/2310.14948)

    本研究提出了物理信息图卷积网络作为解决复杂几何体中偏微分方程问题的广义框架，并结合经典数值求解器从而解决了物理信息框架在复杂几何体上的问题。

    

    自从[9]的开创性工作以及他们的物理信息神经网络（PINNs）之后，已经有很多工作致力于使用深度学习模型来解决偏微分方程（PDEs）。然而，仍然存在一些挑战，例如将这些模型扩展到复杂的三维几何体，以及如何将这些方法与经典的数值求解器结合起来进行研究。在本工作中，我们基于图神经网络和传统数值技术中用于求解偏微分方程的网格之间的相似性，证明了将图神经网络用于这些问题的合理性。在证明了物理信息框架在复杂几何体上计算PDE残差时存在问题后，我们提出了一种替代方案，将经典的数值求解器与物理信息框架相结合。最后，我们提出了一种实现该方法的方案，并在一个不规则几何体上的三维问题上进行了测试。

    Since the seminal work of [9] and their Physics-Informed neural networks (PINNs), many efforts have been conducted towards solving partial differential equations (PDEs) with Deep Learning models. However, some challenges remain, for instance the extension of such models to complex three-dimensional geometries, and a study on how such approaches could be combined to classical numerical solvers. In this work, we justify the use of graph neural networks for these problems, based on the similarity between these architectures and the meshes used in traditional numerical techniques for solving partial differential equations. After proving an issue with the Physics-Informed framework for complex geometries, during the computation of PDE residuals, an alternative procedure is proposed, by combining classical numerical solvers and the Physics-Informed framework. Finally, we propose an implementation of this approach, that we test on a three-dimensional problem on an irregular geometry.
    
[^6]: 粒子引导：非独立同分布样本多样性采样与扩散模型

    Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models. (arXiv:2310.13102v1 [cs.LG])

    [http://arxiv.org/abs/2310.13102](http://arxiv.org/abs/2310.13102)

    本文提出了一种粒子引导的方法，通过超越独立样本的常见假设，提高了生成模型的多样性和采样效率。在实验中，我们在条件图像生成和分子构象生成上进行了测试，并取得了显著的结果。

    

    鉴于生成模型的广泛成功，已经有大量的研究致力于加快其采样时间。然而，为了获取多样性样本，生成模型通常需要进行多次采样，这会造成与采样时间无关的成本。本文处理了如何通过超越独立样本的常见假设来提高多样性和采样效率的问题。我们提出了粒子引导，一种基于扩散的生成采样的扩展，其中的联合粒子时变位势强制实现多样性。我们从理论上分析了粒子引导产生的联合分布，以及它对位势选择的影响和与其他学科方法的联系。在实证方面，我们在条件图像生成的设置中进行了测试，我们能够增加多样性而不影响质量，并在分子构象生成中降低了平均13%的先进技术中值误差。

    In light of the widespread success of generative models, a significant amount of research has gone into speeding up their sampling time. However, generative models are often sampled multiple times to obtain a diverse set incurring a cost that is orthogonal to sampling time. We tackle the question of how to improve diversity and sample efficiency by moving beyond the common assumption of independent samples. We propose particle guidance, an extension of diffusion-based generative sampling where a joint-particle time-evolving potential enforces diversity. We analyze theoretically the joint distribution that particle guidance generates, its implications on the choice of potential, and the connections with methods in other disciplines. Empirically, we test the framework both in the setting of conditional image generation, where we are able to increase diversity without affecting quality, and molecular conformer generation, where we reduce the state-of-the-art median error by 13% on average
    
[^7]: 敏感性感知的摊销贝叶斯推断

    Sensitivity-Aware Amortized Bayesian Inference. (arXiv:2310.11122v1 [stat.ML])

    [http://arxiv.org/abs/2310.11122](http://arxiv.org/abs/2310.11122)

    本文提出了一种敏感性感知的摊销贝叶斯推断方法，通过权重共享和神经网络来进行似然和先验规范的训练，以及对数据扰动和预处理程序的敏感性评估。

    

    贝叶斯推断是在不确定性下进行概率推理和决策的强大框架。现代贝叶斯工作流程中的基本选择涉及似然函数和先验分布的规范、后验逼近器和数据。每个选择都可以显着影响基于模型的推断和后续决策，因此需要进行敏感性分析。在这项工作中，我们提出了一种多方面的方法，将敏感性分析整合到摊销贝叶斯推断（ABI，即基于神经网络的模拟推断）中。首先，我们利用权重共享在训练过程中编码替代似然和先验规范之间的结构相似性，以最小的计算开销。其次，我们利用神经网络的快速推断来评估对各种数据扰动或预处理程序的敏感性。与大多数其他贝叶斯方法相比，这两个步骤都避免了昂贵的计算。

    Bayesian inference is a powerful framework for making probabilistic inferences and decisions under uncertainty. Fundamental choices in modern Bayesian workflows concern the specification of the likelihood function and prior distributions, the posterior approximator, and the data. Each choice can significantly influence model-based inference and subsequent decisions, thereby necessitating sensitivity analysis. In this work, we propose a multifaceted approach to integrate sensitivity analyses into amortized Bayesian inference (ABI, i.e., simulation-based inference with neural networks). First, we utilize weight sharing to encode the structural similarities between alternative likelihood and prior specifications in the training process with minimal computational overhead. Second, we leverage the rapid inference of neural networks to assess sensitivity to various data perturbations or pre-processing procedures. In contrast to most other Bayesian approaches, both steps circumvent the costly
    
[^8]: SD-PINN: 基于深度学习的空间相关偏微分方程恢复

    SD-PINN: Deep Learning based Spatially Dependent PDEs Recovery. (arXiv:2310.10970v1 [cs.LG])

    [http://arxiv.org/abs/2310.10970](http://arxiv.org/abs/2310.10970)

    SD-PINN是一种基于深度学习的方法，能够通过一个神经网络恢复空间相关的偏微分方程（PDE）系数，无需领域特定的物理专业知识，并且对噪声具有稳健性。同时，它能够通过空间变化低秩假设恢复没有可用测量数据的位置的系数。

    

    物理知识驱动的神经网络（PINN）能够直接从物理测量中恢复在整个空间域中保持不变的偏微分方程（PDE）系数。在本文中，我们提出了一种空间相关的物理知识驱动神经网络（SD-PINN），它通过一个单一的神经网络来恢复空间相关的PDE系数，消除了对领域特定的物理专业知识的要求。所提出的方法由于加入了物理约束而对噪声具有稳健性。它还能够将PDE系数的空间变化低秩假设纳入考虑，从而恢复没有可用测量数据的位置的系数。

    The physics-informed neural network (PINN) is capable of recovering partial differential equation (PDE) coefficients that remain constant throughout the spatial domain directly from physical measurements. In this work, we propose a spatially dependent physics-informed neural network (SD-PINN), which enables the recovery of coefficients in spatially-dependent PDEs using a single neural network, eliminating the requirement for domain-specific physical expertise. The proposed method exhibits robustness to noise owing to the incorporation of physical constraints. It can also incorporate the low-rank assumption of the spatial variation for the PDE coefficients to recover the coefficients at locations without available measurements.
    
[^9]: 从懒惰到丰富训练动态的洞察力

    Grokking as the Transition from Lazy to Rich Training Dynamics. (arXiv:2310.06110v1 [stat.ML])

    [http://arxiv.org/abs/2310.06110](http://arxiv.org/abs/2310.06110)

    研究发现洞察现象可能是由神经网络从懒惰训练动态过渡到丰富的特征学习模式的结果，通过跟踪足够的统计量，发现洞察是在网络首先尝试拟合核回归解决方案后，进行后期特征学习找到通用解决方案之后的结果。

    

    我们提出了洞察现象，即神经网络的训练损失在测试损失之前大幅下降，可能是由于神经网络从懒惰的训练动态转变为丰富的特征学习模式。为了说明这一机制，我们研究了在没有正则化的情况下，使用Vanilla梯度下降方法在多项式回归问题上进行的两层神经网络的训练，该训练展现了无法用现有理论解释的洞察现象。我们确定了该网络测试损失的足够统计量，并通过训练跟踪这些统计量揭示了洞察现象的发生。我们发现，在这种情况下，网络首先尝试使用初始特征拟合核回归解决方案，接着在训练损失已经很低的情况下进行后期特征学习，从而找到了一个能够泛化的解决方案。我们发现，洞察产生的关键因素是特征学习的速率，这可以通过缩放网络参数来精确控制。

    We propose that the grokking phenomenon, where the train loss of a neural network decreases much earlier than its test loss, can arise due to a neural network transitioning from lazy training dynamics to a rich, feature learning regime. To illustrate this mechanism, we study the simple setting of vanilla gradient descent on a polynomial regression problem with a two layer neural network which exhibits grokking without regularization in a way that cannot be explained by existing theories. We identify sufficient statistics for the test loss of such a network, and tracking these over training reveals that grokking arises in this setting when the network first attempts to fit a kernel regression solution with its initial features, followed by late-time feature learning where a generalizing solution is identified after train loss is already low. We find that the key determinants of grokking are the rate of feature learning -- which can be controlled precisely by parameters that scale the ne
    
[^10]: 狮子秘密地解决受限制优化问题：正如李雅普诺夫所预测的。

    Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts. (arXiv:2310.05898v1 [cs.LG])

    [http://arxiv.org/abs/2310.05898](http://arxiv.org/abs/2310.05898)

    Lion是通过程序搜索发现的新优化器，在训练大型AI模型方面表现出有希望的结果，具有更高的内存效率。尽管其理论基础不明确，但基于连续时间和离散时间分析，我们证明Lion是一种理论上新颖且有原则的方法，可在最小化一般损失函数的同时强制执行边界约束。

    

    通过程序搜索发现的新优化器Lion（进化的符号动量）在训练大型AI模型方面显示出有希望的结果。它在训练效果上与AdamW相当或更好，并具有更高的内存效率。正如我们可以从随机搜索程序的结果中期待的，Lion集成了几个现有算法的元素，包括符号动量、独立的权重衰减、Polak和Nesterov动量，但又不属于任何现有的理论基础优化器类别。因此，尽管Lion作为广泛任务的通用优化器表现良好，但其理论基础仍然不明确。这种缺乏理论的明确性限制了进一步增强和扩展Lion的可能性。本文旨在揭开Lion的神秘面纱。基于连续时间和离散时间分析，我们证明Lion是一种理论上新颖且有原则的方法，可在最小化一般损失函数$f(x)$的同时强制执行边界约束。

    Lion (Evolved Sign Momentum), a new optimizer discovered through program search, has shown promising results in training large AI models. It performs comparably or favorably to AdamW but with greater memory efficiency. As we can expect from the results of a random search program, Lion incorporates elements from several existing algorithms, including signed momentum, decoupled weight decay, Polak, and Nesterov momentum, but does not fit into any existing category of theoretically grounded optimizers. Thus, even though Lion appears to perform well as a general-purpose optimizer for a wide range of tasks, its theoretical basis remains uncertain. This lack of theoretical clarity limits opportunities to further enhance and expand Lion's efficacy.  This work aims to demystify Lion. Based on both continuous-time and discrete-time analysis, we demonstrate that Lion is a theoretically novel and principled approach for minimizing a general loss function $f(x)$ while enforcing a bound constraint 
    
[^11]: 跨不同条件精确预测电池寿命的学习细胞内和细胞间差异

    Learning Intra- and Inter-Cell Differences for Accurate Battery Lifespan Prediction across Diverse Conditions. (arXiv:2310.05052v2 [eess.SP] UPDATED)

    [http://arxiv.org/abs/2310.05052](http://arxiv.org/abs/2310.05052)

    该论文介绍了一种跨不同条件精确预测电池寿命的方法，通过捕捉目标电池和参考电池之间的电信号差异，无论材料和老化条件如何，在扩展特征空间的同时为通用的电池寿命预测框架铺平了道路。

    

    电池寿命预测对电池研究和开发具有重要的实际价值。目前，许多数据驱动模型依赖于特定目标电池的早期电信号来预测它们的寿命。一个常见的不足是，大多数现有方法都是基于特定老化条件开发的，这不仅限制了它们的模型能力，而且降低了它们在预测不同条件下的退化效果。因此，这些模型通常无法充分利用其他条件下可用的丰富历史数据。为了解决这个问题，我们引入了一种方法，明确捕捉目标电池和参考电池之间的电信号差异，无论它们的材料和老化条件如何，来预测目标电池的寿命。通过这种细胞间差异，我们不仅扩展了特征空间，还为通用的电池寿命预测框架铺平了道路。显著的是，我们的方法能够在不同条件下精确预测电池寿命。

    Battery life prediction holds significant practical value for battery research and development. Currently, many data-driven models rely on early electrical signals from specific target batteries to predict their lifespan. A common shortfall is that most existing methods are developed based on specific aging conditions, which not only limits their model's capability but also diminishes their effectiveness in predicting degradation under varied conditions. As a result, these models often miss out on fully benefiting from the rich historical data available under other conditions. Here, to address above, we introduce an approach that explicitly captures differences between electrical signals of a target battery and a reference battery, irrespective of their materials and aging conditions, to forecast the target battery life. Through this inter-cell difference, we not only enhance the feature space but also pave the way for a universal battery life prediction framework. Remarkably, our mode
    
[^12]: 快速、表达力强的SE$(n)$等变网络通过在位置-方向空间中共享权重

    Fast, Expressive SE$(n)$ Equivariant Networks through Weight-Sharing in Position-Orientation Space. (arXiv:2310.02970v1 [cs.LG])

    [http://arxiv.org/abs/2310.02970](http://arxiv.org/abs/2310.02970)

    该论文通过在位置-方向空间中共享权重，提出了一种快速、表达力强的SE$(n)$等变网络。他们基于同态空间理论，推导出几何优化的边属性，并将权重共享形式化为对等处理相同点对的消息函数。他们在处理3D点云时，开发了一个高效的等变群卷积网络，并选择了$\mathbb{R}^3 {\times} S^2$作为最佳的处理空间。

    

    我们基于同态空间理论推导出用于灵活的消息传递框架的“几何优化边属性”。我们将卷积神经网络中的权重共享形式化为对等地处理并且应该被平等对待的点对的消息函数共享。我们定义了等价类，这些等价类在群中进行变换时是相同的，并且推导出唯一标识这些类别的属性。通过在这些属性上进行条件化，可以实现权重共享。作为该理论的应用，我们开发了一个高效的等变群卷积网络来处理3D点云。同态空间理论告诉我们如何在位置$\mathbb{R}^3$、位置和方向$\mathbb{R}^3 {\times} S^2$的同态空间以及群SE$(3)$上的特征图上进行群卷积。在这些选择中，$\mathbb{R}^3 {\times} S^2$是一个最佳选择，因为它具有处理方向信息的能力。

    Based on the theory of homogeneous spaces we derive \textit{geometrically optimal edge attributes} to be used within the flexible message passing framework. We formalize the notion of weight sharing in convolutional networks as the sharing of message functions over point-pairs that should be treated equally. We define equivalence classes of point-pairs that are identical up to a transformation in the group and derive attributes that uniquely identify these classes. Weight sharing is then obtained by conditioning message functions on these attributes. As an application of the theory, we develop an efficient equivariant group convolutional network for processing 3D point clouds. The theory of homogeneous spaces tells us how to do group convolutions with feature maps over the homogeneous space of positions $\mathbb{R}^3$, position and orientations $\mathbb{R}^3 {\times} S^2$, and the group SE$(3)$ itself. Among these, $\mathbb{R}^3 {\times} S^2$ is an optimal choice due to the ability to 
    
[^13]: EGraFFBench: 用于原子模拟的等变图神经网络力场评估

    EGraFFBench: Evaluation of Equivariant Graph Neural Network Force Fields for Atomistic Simulations. (arXiv:2310.02428v1 [cs.LG])

    [http://arxiv.org/abs/2310.02428](http://arxiv.org/abs/2310.02428)

    EGraFFBench对六种EGraFF算法进行了系统的基准测试，以评估其在原子模拟中的性能，并提出了新的数据集和度量标准。

    

    通过利用图的固有对称性，等变图神经网络力场(EGraFF)在建模原子系统中的复杂相互作用方面表现出巨大的潜力。最近的研究引发了对新型架构的开发潮，这些架构将等变性的归纳偏见与图变换器和消息传递等架构创新结合起来，以建模原子相互作用。然而，我们目前对这些使用EGraFF进行实际原子模拟任务的彻底评估还缺乏。为了达到这个目的，我们在这里对6种EGraFF算法(NequIP, Allegro, BOTNet, MACE, Equiformer, TorchMDNet)进行了系统的基准测试，以了解它们在实际原子模拟中的能力和限制。除了对基于基准测试文献的八个现有数据集进行彻底评估和分析外，我们还发布了两个新的基准数据集，提出了四个新的度量标准和三个新的具有挑战性的任务。

    Equivariant graph neural networks force fields (EGraFFs) have shown great promise in modelling complex interactions in atomic systems by exploiting the graphs' inherent symmetries. Recent works have led to a surge in the development of novel architectures that incorporate equivariance-based inductive biases alongside architectural innovations like graph transformers and message passing to model atomic interactions. However, thorough evaluations of these deploying EGraFFs for the downstream task of real-world atomistic simulations, is lacking. To this end, here we perform a systematic benchmarking of 6 EGraFF algorithms (NequIP, Allegro, BOTNet, MACE, Equiformer, TorchMDNet), with the aim of understanding their capabilities and limitations for realistic atomistic simulations. In addition to our thorough evaluation and analysis on eight existing datasets based on the benchmarking literature, we release two new benchmark datasets, propose four new metrics, and three new challenging tasks.
    
[^14]: 过参数化如何减缓矩阵感知中的梯度下降：对称性和初始化的问题。

    How Over-Parameterization Slows Down Gradient Descent in Matrix Sensing: The Curses of Symmetry and Initialization. (arXiv:2310.01769v1 [cs.LG])

    [http://arxiv.org/abs/2310.01769](http://arxiv.org/abs/2310.01769)

    该论文研究了过参数化如何影响矩阵感知问题中梯度下降的收敛行为，在对称和非对称设置下给出了不同的收敛速度。

    

    本文详细阐述了过参数化如何改变梯度下降在矩阵感知问题中的收敛行为。在对称设置中，通过对称参数化学习未知的半正定矩阵，我们给出了过参数化情况下（$k>r$）随机初始化梯度下降的新型$\Omega (1/T^2)$下界，与精确参数化情况（$k=r$）的收敛速度$\exp (-\Omega (T))$形成鲜明对比。接下来，我们研究了不对称设置，其中$M^* \in \mathbb{R}^{n_1 \times n_2}$是未知矩阵，采用非对称参数化学习。

    This paper rigorously shows how over-parameterization changes the convergence behaviors of gradient descent (GD) for the matrix sensing problem, where the goal is to recover an unknown low-rank ground-truth matrix from near-isotropic linear measurements. First, we consider the symmetric setting with the symmetric parameterization where $M^* \in \mathbb{R}^{n \times n}$ is a positive semi-definite unknown matrix of rank $r \ll n$, and one uses a symmetric parameterization $XX^\top$ to learn $M^*$. Here $X \in \mathbb{R}^{n \times k}$ with $k > r$ is the factor matrix. We give a novel $\Omega (1/T^2)$ lower bound of randomly initialized GD for the over-parameterized case ($k >r$) where $T$ is the number of iterations. This is in stark contrast to the exact-parameterization scenario ($k=r$) where the convergence rate is $\exp (-\Omega (T))$. Next, we study asymmetric setting where $M^* \in \mathbb{R}^{n_1 \times n_2}$ is the unknown matrix of rank $r \ll \min\{n_1,n_2\}$, and one uses an 
    
[^15]: 一种适用于现代网络的路径范数工具包：影响、前景和挑战

    A path-norm toolkit for modern networks: consequences, promises and challenges. (arXiv:2310.01225v1 [stat.ML])

    [http://arxiv.org/abs/2310.01225](http://arxiv.org/abs/2310.01225)

    本文介绍了适用于现代神经网络的路径范数工具包，可以包括具有偏差、跳跃连接和最大池化的通用DAG ReLU网络。这个工具包恢复或超越了已知的路径范数界限，并挑战了基于路径范数的一些具体承诺。

    

    本文介绍了第一个完全能够包括具有偏差、跳跃连接和最大池化的通用DAG ReLU网络的路径范数工具包。这个工具包不仅适用于最广泛的基于路径范数的现代神经网络，还可以恢复或超越已知的此类范数的最尖锐界限。这些扩展的路径范数还享有路径范数的常规优点：计算简便、对网络的对称性具有不变性，在前馈网络上比操作符范数的乘积（另一种常用的复杂度度量）具有更好的锐度。工具包的多功能性和易于实施使我们能够通过数值评估在ImageNet上对ResNet的最尖锐界限来挑战基于路径范数的具体承诺。

    This work introduces the first toolkit around path-norms that is fully able to encompass general DAG ReLU networks with biases, skip connections and max pooling. This toolkit notably allows us to establish generalization bounds for real modern neural networks that are not only the most widely applicable path-norm based ones, but also recover or beat the sharpest known bounds of this type. These extended path-norms further enjoy the usual benefits of path-norms: ease of computation, invariance under the symmetries of the network, and improved sharpness on feedforward networks compared to the product of operators' norms, another complexity measure most commonly used.  The versatility of the toolkit and its ease of implementation allow us to challenge the concrete promises of path-norm-based generalization bounds, by numerically evaluating the sharpest known bounds for ResNets on ImageNet.
    
[^16]: 随机梯度下降的噪声几何：定量和分析特征的研究

    The Noise Geometry of Stochastic Gradient Descent: A Quantitative and Analytical Characterization. (arXiv:2310.00692v1 [cs.LG])

    [http://arxiv.org/abs/2310.00692](http://arxiv.org/abs/2310.00692)

    本文对随机梯度下降（SGD）中的噪声几何进行了全面的理论研究，发现噪声与损失函数的局部几何特征有利的一致性。通过实验证明，SGD在逃脱尖锐极小值时与GD形成鲜明对比，逃脱方向在平坦方向上有显著分量。

    

    实证研究表明，随机梯度下降（SGD）中的噪声与损失函数的局部几何特征有利的一致性。然而，对于这种现象的理论和定量解释仍然不足。本文对过参数化线性模型和两层神经网络的上述“噪声几何”进行了全面的理论研究。我们细致地研究了平均和方向的一致性，特别关注样本大小和输入数据退化对一致性强度的影响。作为特定应用，我们利用噪声几何特征研究了SGD如何从尖锐极小值中逃脱，发现逃脱方向在平坦方向上有显著分量，这与只在最尖锐方向逃脱的梯度下降方法GD形成鲜明对比。为了验证我们的理论发现，我们进行了合成和真实世界的实验。

    Empirical studies have demonstrated that the noise in stochastic gradient descent (SGD) aligns favorably with the local geometry of loss landscape. However, theoretical and quantitative explanations for this phenomenon remain sparse. In this paper, we offer a comprehensive theoretical investigation into the aforementioned {\em noise geometry} for over-parameterized linear (OLMs) models and two-layer neural networks. We scrutinize both average and directional alignments, paying special attention to how factors like sample size and input data degeneracy affect the alignment strength. As a specific application, we leverage our noise geometry characterizations to study how SGD escapes from sharp minima, revealing that the escape direction has significant components along flat directions. This is in stark contrast to GD, which escapes only along the sharpest directions. To substantiate our theoretical findings, both synthetic and real-world experiments are provided.
    
[^17]: SIMD数据流共优化用于CPU上高效的神经网络推理

    SIMD Dataflow Co-optimization for Efficient Neural Networks Inferences on CPUs. (arXiv:2310.00574v2 [cs.AR] UPDATED)

    [http://arxiv.org/abs/2310.00574](http://arxiv.org/abs/2310.00574)

    我们提出了一种通过共优化数据流和SIMD实现来高效地在CPU上进行神经网络推理的方法，实验结果表明，这种方法能够在保持准确性的同时大幅提升推理速度。

    

    我们针对在CPU上部署神经网络所面临的挑战提出了解决方案，特别关注的是在保持准确性的同时最小化推理时间。我们的新颖方法是利用神经网络的数据流（即计算顺序），通过启发式引导分析和代码生成框架来探索数据重用机会，从而实现各种单指令多数据（SIMD）实现以实现优化的神经网络执行。我们的结果表明，将输出保持在SIMD寄存器中的数据流同时最大化输入和权重重用，在各种推理工作负载下始终能够获得最佳性能，相比今天的神经网络优化实现，8位神经网络的加速比可达3倍，而二进制神经网络的加速比可达4.8倍。

    We address the challenges associated with deploying neural networks on CPUs, with a particular focus on minimizing inference time while maintaining accuracy. Our novel approach is to use the dataflow (i.e., computation order) of a neural network to explore data reuse opportunities using heuristic-guided analysis and a code generation framework, which enables exploration of various Single Instruction, Multiple Data (SIMD) implementations to achieve optimized neural network execution. Our results demonstrate that the dataflow that keeps outputs in SIMD registers while also maximizing both input and weight reuse consistently yields the best performance for a wide variety of inference workloads, achieving up to 3x speedup for 8-bit neural networks, and up to 4.8x speedup for binary neural networks, respectively, over the optimized implementations of neural networks today.
    
[^18]: 动态边界最大化和改进的Lipschitz正则化的认证鲁棒性

    Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization. (arXiv:2310.00116v1 [cs.LG])

    [http://arxiv.org/abs/2310.00116](http://arxiv.org/abs/2310.00116)

    本文提出了一种基于动态边界最大化和改进的Lipschitz正则化的认证鲁棒性训练算法，通过增加输出空间中的边界和正则化模型的Lipschitz常数来提高深度分类器对抗性扰动的鲁棒性。

    

    为了提高深度分类器对抗性扰动的鲁棒性，已经提出了许多方法，例如设计具有更好鲁棒性性质的新架构（例如，Lipschitz-capped网络）或修改训练过程本身（例如，最小-最大优化，约束学习或正则化）。然而，这些方法对于增加输入（特征）空间中的边界可能并不有效。因此，越来越多的人开始对开发能够直接操纵输入空间中的决策边界的训练过程感兴趣。在本文中，我们在该类别的最新发展基础上，开发了一种鲁棒训练算法，其目标是在输出（logit）空间中增加边界，并沿着脆弱方向正则化模型的Lipschitz常数。我们证明这两个目标可以直接促进输入空间中更大的边界。为此，我们开发了一种可扩展的方法来计算...

    To improve the robustness of deep classifiers against adversarial perturbations, many approaches have been proposed, such as designing new architectures with better robustness properties (e.g., Lipschitz-capped networks), or modifying the training process itself (e.g., min-max optimization, constrained learning, or regularization). These approaches, however, might not be effective at increasing the margin in the input (feature) space. As a result, there has been an increasing interest in developing training procedures that can directly manipulate the decision boundary in the input space. In this paper, we build upon recent developments in this category by developing a robust training algorithm whose objective is to increase the margin in the output (logit) space while regularizing the Lipschitz constant of the model along vulnerable directions. We show that these two objectives can directly promote larger margins in the input space. To this end, we develop a scalable method for calcula
    
[^19]: 导航等变扩散基生成模型的设计空间，用于从头生成3D分子

    Navigating the Design Space of Equivariant Diffusion-Based Generative Models for De Novo 3D Molecule Generation. (arXiv:2309.17296v1 [cs.LG])

    [http://arxiv.org/abs/2309.17296](http://arxiv.org/abs/2309.17296)

    本论文探索了E(3)等变扩散模型的设计空间，提出了EQGAT-diff模型，通过在连续的原子位置和分类的化学元素与键类型之间的交互中改进，其在从头设计3D分子方面的性能显著超过已有模型。

    

    深度生成扩散模型是材料科学和药物发现中从头设计3D分子的一种有前途的途径。然而，它们在大分子结构和有限的训练数据方面的性能仍受到限制。为了解决这个问题，我们探索了E(3)等变扩散模型的设计空间，重点关注以前的空白点。我们进行了广泛的比较分析，评估了连续和离散状态空间之间的相互作用。在这个调查中，我们引入了EQGAT-diff模型，其在QM9和GEOM-Drugs数据集上的性能始终大大超过已建立模型。与其他模型不同的是，EQGAT-diff采用连续的原子位置，而化学元素和键类型是分类的，并采用时间相关的损失加权，这显著提高了训练收敛和生成样本的质量。为进一步增强扩散模型对有限训练数据的适用性，

    Deep generative diffusion models are a promising avenue for de novo 3D molecular design in material science and drug discovery. However, their utility is still constrained by suboptimal performance with large molecular structures and limited training data. Addressing this gap, we explore the design space of E(3) equivariant diffusion models, focusing on previously blank spots. Our extensive comparative analysis evaluates the interplay between continuous and discrete state spaces. Out of this investigation, we introduce the EQGAT-diff model, which consistently surpasses the performance of established models on the QM9 and GEOM-Drugs datasets by a large margin. Distinctively, EQGAT-diff takes continuous atomic positions while chemical elements and bond types are categorical and employ a time-dependent loss weighting that significantly increases training convergence and the quality of generated samples. To further strengthen the applicability of diffusion models to limited training data, 
    
[^20]: 为什么角边距损失在半监督异常声音检测中表现出色？

    Why do Angular Margin Losses work well for Semi-Supervised Anomalous Sound Detection?. (arXiv:2309.15643v1 [eess.AS])

    [http://arxiv.org/abs/2309.15643](http://arxiv.org/abs/2309.15643)

    角边距损失与辅助任务结合在半监督异常声音检测中表现出色，通过最小化角边距损失同时达到最小化紧凑性损失和防止学习平凡解的效果。

    

    最先进的异常声音检测系统通常利用角边距损失来通过一个辅助任务学习合适的声学数据表示，该任务通常是一个监督或自监督分类任务。其基本思想是为了解决这个辅助任务，需要在学习的表示中捕捉到关于正常数据的特定信息，并且这些信息也足以区分正常和异常样本。特别是在噪声条件下，基于角边距损失的判别模型往往明显优于基于生成模型或单类模型的系统。本研究的目标是调查为什么在辅助任务中使用角边距损失对于检测异常声音效果良好。通过理论和实验证明，最小化角边距损失也最小化了紧凑性损失，同时固有地防止学习平凡的解。此外，m

    State-of-the-art anomalous sound detection systems often utilize angular margin losses to learn suitable representations of acoustic data using an auxiliary task, which usually is a supervised or self-supervised classification task. The underlying idea is that, in order to solve this auxiliary task, specific information about normal data needs to be captured in the learned representations and that this information is also sufficient to differentiate between normal and anomalous samples. Especially in noisy conditions, discriminative models based on angular margin losses tend to significantly outperform systems based on generative or one-class models. The goal of this work is to investigate why using angular margin losses with auxiliary tasks works well for detecting anomalous sounds. To this end, it is shown, both theoretically and experimentally, that minimizing angular margin losses also minimizes compactness loss while inherently preventing learning trivial solutions. Furthermore, m
    
[^21]: 关联变换器是一种稀疏表示学习器

    Associative Transformer Is A Sparse Representation Learner. (arXiv:2309.12862v1 [cs.LG])

    [http://arxiv.org/abs/2309.12862](http://arxiv.org/abs/2309.12862)

    关联变换器（AiT）是一种采用低秩显式记忆和关联记忆的稀疏表示学习器，通过联合端到端训练实现模块特化和注意力瓶颈的形成。

    

    在传统的Transformer模型中，出现了一种新兴的基于稀疏交互的注意力机制，这种机制与生物原理更为接近。包括Set Transformer和Perceiver在内的方法采用了与有限能力的潜在空间相结合的交叉注意力机制。基于最近对全局工作空间理论和关联记忆的神经科学研究，我们提出了关联变换器（AiT）。AiT引入了低秩显式记忆，既可以作为先验来指导共享工作空间的瓶颈注意力，又可以作为关联记忆的吸引子。通过联合端到端训练，这些先验自然地发展出模块的特化，每个模块对形成注意力瓶颈的归纳偏好有所贡献。瓶颈可以促进输入之间为将信息写入内存而进行竞争。我们展示了AiT是一种稀疏表示学习器。

    Emerging from the monolithic pairwise attention mechanism in conventional Transformer models, there is a growing interest in leveraging sparse interactions that align more closely with biological principles. Approaches including the Set Transformer and the Perceiver employ cross-attention consolidated with a latent space that forms an attention bottleneck with limited capacity. Building upon recent neuroscience studies of Global Workspace Theory and associative memory, we propose the Associative Transformer (AiT). AiT induces low-rank explicit memory that serves as both priors to guide bottleneck attention in the shared workspace and attractors within associative memory of a Hopfield network. Through joint end-to-end training, these priors naturally develop module specialization, each contributing a distinct inductive bias to form attention bottlenecks. A bottleneck can foster competition among inputs for writing information into the memory. We show that AiT is a sparse representation 
    
[^22]: 使用Forman-Ricci曲率的扩展来减轻过度平滑和过度压缩问题

    Mitigating Over-Smoothing and Over-Squashing using Augmentations of Forman-Ricci Curvature. (arXiv:2309.09384v1 [cs.LG])

    [http://arxiv.org/abs/2309.09384](http://arxiv.org/abs/2309.09384)

    本文提出了一种使用Forman-Ricci曲率扩展的方法来减轻图神经网络中的过度平滑和过度压缩问题。通过观察离散曲率，可以添加或删除边以减轻这两种效应。

    

    虽然图神经网络（GNNs）在不同领域的图结构数据学习中取得了成功，但最近描述了几个潜在的陷阱。这些包括无法准确利用编码在长距离连接中的信息（过度压缩），以及在网络深度增加时难以区分附近节点的学习表示（过度平滑）。一种有效的表征这两种效应的方法是离散曲率：导致过度压缩效应的长距离连接具有低曲率，而导致过度平滑的边具有高曲率。这个观察引发了一些重连技术，通过增加或删除边来减轻过度平滑和过度压缩问题。已经提出了几种利用图特征（如曲率或图拉普拉斯算子的谱）的重连方法。然而，现有方法，特别是基于曲率的方法，通常需要昂贵的子图操作。

    While Graph Neural Networks (GNNs) have been successfully leveraged for learning on graph-structured data across domains, several potential pitfalls have been described recently. Those include the inability to accurately leverage information encoded in long-range connections (over-squashing), as well as difficulties distinguishing the learned representations of nearby nodes with growing network depth (over-smoothing). An effective way to characterize both effects is discrete curvature: Long-range connections that underlie over-squashing effects have low curvature, whereas edges that contribute to over-smoothing have high curvature. This observation has given rise to rewiring techniques, which add or remove edges to mitigate over-smoothing and over-squashing. Several rewiring approaches utilizing graph characteristics, such as curvature or the spectrum of the graph Laplacian, have been proposed. However, existing methods, especially those based on curvature, often require expensive subr
    
[^23]: 结构感知的辛系统哈密顿（图）嵌入

    Symplectic Structure-Aware Hamiltonian (Graph) Embeddings. (arXiv:2309.04885v1 [cs.LG])

    [http://arxiv.org/abs/2309.04885](http://arxiv.org/abs/2309.04885)

    本文提出了SAH-GNN，一种在图神经网络中应用结构感知的辛系统哈密顿嵌入方法。与传统方法不同，SAH-GNN通过在训练过程中自适应学习辛结构，避免了依赖预定义标准辛结构形式的限制，并能够适应不同的图数据集，同时保持物理意义上的能量守恒。

    

    在传统的图神经网络（GNNs）中，固定嵌入流形的假设常常限制了其对不同图几何结构的适应性。最近，提出了基于哈密顿系统的GNNs，通过将物理定律纳入节点特征更新中，来解决这类嵌入的动态特性。在这项工作中，我们提出了SAH-GNN，一种新颖的方法，将哈密顿动力学推广到更灵活的节点特征更新中。与现有的受哈密顿启发的GNNs不同，SAH-GNN在训练过程中采用辛斯蒂费尔流形上的黎曼优化，自适应地学习潜在的辛结构，从而规避了现有依赖预定义标准辛结构形式的哈密顿GNNs的局限性。这一创新使得SAH-GNN能够在没有大量超参数调整的情况下自动适应各种图数据集。此外，它在训练过程中保持能量守恒，使得隐式哈密顿系统具有物理意义。

    In traditional Graph Neural Networks (GNNs), the assumption of a fixed embedding manifold often limits their adaptability to diverse graph geometries. Recently, Hamiltonian system-inspired GNNs are proposed to address the dynamic nature of such embeddings by incorporating physical laws into node feature updates. In this work, we present SAH-GNN, a novel approach that generalizes Hamiltonian dynamics for more flexible node feature updates. Unlike existing Hamiltonian-inspired GNNs, SAH-GNN employs Riemannian optimization on the symplectic Stiefel manifold to adaptively learn the underlying symplectic structure during training, circumventing the limitations of existing Hamiltonian GNNs that rely on a pre-defined form of standard symplectic structure. This innovation allows SAH-GNN to automatically adapt to various graph datasets without extensive hyperparameter tuning. Moreover, it conserves energy during training such that the implicit Hamiltonian system is physically meaningful. To thi
    
[^24]: 具有集成离群检测的超声心动图视图分类，以增强自动超声心动图分析

    Echocardiographic View Classification with Integrated Out-of-Distribution Detection for Enhanced Automatic Echocardiographic Analysis. (arXiv:2308.16483v1 [eess.SP])

    [http://arxiv.org/abs/2308.16483](http://arxiv.org/abs/2308.16483)

    ECHO-VICODE是一个深度学习框架，通过训练来分类超声心动图的31个视图类别，并具有集成的离群检测功能，可以显著降低超声心动图中的错误可能性。

    

    在快速发展的自动超声心动图分析和解释领域中，自动视图分类是一个关键但具有挑战性的任务，这是由于超声心动图数据的固有复杂性和可变性。本研究提出了ECHO-VICODE（超声心动图视图分类与离群检测），这是一个基于深度学习的新型框架，通过训练来分类31个类别，超过了先前的研究，并展示了其处理多种超声心动图视图的能力。此外，ECHO-VICODE还加入了一个集成的离群检测功能，利用相对马氏距离有效识别常见的“接近离群”实例。通过大量实验，我们展示了ECHO-VICODE在视图分类和离群检测方面的出色性能，显著降低了超声心动图中潜在错误的可能性。

    In the rapidly evolving field of automatic echocardiographic analysis and interpretation, automatic view classification is a critical yet challenging task, owing to the inherent complexity and variability of echocardiographic data. This study presents ECHOcardiography VIew Classification with Out-of-Distribution dEtection (ECHO-VICODE), a novel deep learning-based framework that effectively addresses this challenge by training to classify 31 classes, surpassing previous studies and demonstrating its capacity to handle a wide range of echocardiographic views. Furthermore, ECHO-VICODE incorporates an integrated out-of-distribution (OOD) detection function, leveraging the relative Mahalanobis distance to effectively identify 'near-OOD' instances commonly encountered in echocardiographic data. Through extensive experimentation, we demonstrated the outstanding performance of ECHO-VICODE in terms of view classification and OOD detection, significantly reducing the potential for errors in ech
    
[^25]: Karasu:一种用于大数据分析的高效集群配置的协作方法

    Karasu: A Collaborative Approach to Efficient Cluster Configuration for Big Data Analytics. (arXiv:2308.11792v1 [cs.DC])

    [http://arxiv.org/abs/2308.11792](http://arxiv.org/abs/2308.11792)

    Karasu是一种通过促进类似基础设施、框架、算法或数据集上工作的用户之间的数据共享，实现更高效的资源配置配置文件分析的方法。

    

    由于机器类型和集群规模等配置选项的广泛多样性，选择适合大数据分析作业的正确资源是困难的。由于糟糕的选择可能对资源效率、成本和能源使用产生重大影响，自动化方法越来越受欢迎。大多数现有方法依赖于对重复工作负载进行配置文件分析，以寻找接近最优解的解决方案。由于冷启动问题，这通常导致费时且昂贵的配置文件分析阶段。然而，不同用户的大数据分析作业可以共享许多共同特性：它们通常在类似的基础设施上操作，使用类似的算法在类似的框架中实现。共享聚合配置文件分析运行以协作解决冷启动问题的潜力还没有得到充分探索。我们提出了Karasu，一种促进在类似基础设施、框架、算法或数据集上工作的用户之间共享数据的更高效的资源配置配置文件分析方法。

    Selecting the right resources for big data analytics jobs is hard because of the wide variety of configuration options like machine type and cluster size. As poor choices can have a significant impact on resource efficiency, cost, and energy usage, automated approaches are gaining popularity. Most existing methods rely on profiling recurring workloads to find near-optimal solutions over time. Due to the cold-start problem, this often leads to lengthy and costly profiling phases. However, big data analytics jobs across users can share many common properties: they often operate on similar infrastructure, using similar algorithms implemented in similar frameworks. The potential in sharing aggregated profiling runs to collaboratively address the cold start problem is largely unexplored.  We present Karasu, an approach to more efficient resource configuration profiling that promotes data sharing among users working with similar infrastructures, frameworks, algorithms, or datasets. Karasu tr
    
[^26]: 想法图：用大型语言模型解决复杂问题

    Graph of Thoughts: Solving Elaborate Problems with Large Language Models. (arXiv:2308.09687v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.09687](http://arxiv.org/abs/2308.09687)

    想法图（GoT）是一种新的框架，它超越了现有的提示范式，通过将大型语言模型（LLM）的信息建模为任意图形，将LLM想法组合成具有协同效应的结果，提炼整个思维网络的本质，或者使用反馈环路增强思维。GoT在不同任务上展示出优势，并可以通过新的想法转换进行扩展，使LLM的推理更接近人类思维。

    

    我们介绍了一种名为想法图（Graph of Thoughts，GoT）的框架，它在大型语言模型（LLM）的提示能力上超越了Chain-of-Thought或Tree of Thoughts（ToT）等范式。GoT的关键思想和主要优势在于能够将LLM生成的信息建模为任意图形，其中信息单元（"LLM想法"）是顶点，边表示这些顶点之间的依赖关系。这种方法使得将任意LLM想法组合成具有协同效应的结果、提炼整个思维网络的本质或者使用反馈环路增强思维成为可能。我们证明GoT在不同任务上比最先进的方法有优势，例如在排序任务上质量提高了62%，同时成本降低了超过31%。我们确保GoT能够通过新的想法转换进行扩展，从而可以用于开创新的提示方案。这项工作使得LLM的推理更接近人类思维。

    We introduce Graph of Thoughts (GoT): a framework that advances prompting capabilities in large language models (LLMs) beyond those offered by paradigms such as Chain-of-Thought or Tree of Thoughts (ToT). The key idea and primary advantage of GoT is the ability to model the information generated by an LLM as an arbitrary graph, where units of information ("LLM thoughts") are vertices, and edges correspond to dependencies between these vertices. This approach enables combining arbitrary LLM thoughts into synergistic outcomes, distilling the essence of whole networks of thoughts, or enhancing thoughts using feedback loops. We illustrate that GoT offers advantages over state of the art on different tasks, for example increasing the quality of sorting by 62% over ToT, while simultaneously reducing costs by >31%. We ensure that GoT is extensible with new thought transformations and thus can be used to spearhead new prompting schemes. This work brings the LLM reasoning closer to human thinki
    
[^27]: 基于核的无似然假设检验方法

    Kernel-Based Tests for Likelihood-Free Hypothesis Testing. (arXiv:2308.09043v1 [stat.ML])

    [http://arxiv.org/abs/2308.09043](http://arxiv.org/abs/2308.09043)

    本文介绍了一种基于核的无似然假设检验方法，解决了对已知属于两个类别的输入进行分类的问题，在无似然推断领域，通过将标记样本通过正向模拟获得，未标记样本通过实验收集，给出了一个权衡m和n的方法。

    

    从两个平衡类别的n个观测中，考虑对额外m个已知属于其中一个类别的输入进行分类的任务。该问题的特殊情况已经被广泛研究：当完全了解类别分布时（n=∞），最优解是使用似然比检验；当m=1时，对应二分类问题；当m≈n时，等同于两样本检验。中间的情况出现在无似然推断领域，其中标记样本通过运行正向模拟获得，而未标记样本通过实验收集。最近的研究发现，m和n之间存在根本性的权衡：增加数据样本m会减少所需的训练/模拟数据量n。在本研究中，我们（a）引入了一个常常遇到的情况，即未标记样本来自两个类别的混合物；（b）研究了最小化风险的方法，其中风险定义为误分类概率的上界。

    Given $n$ observations from two balanced classes, consider the task of labeling an additional $m$ inputs that are known to all belong to \emph{one} of the two classes. Special cases of this problem are well-known: with complete knowledge of class distributions ($n=\infty$) the problem is solved optimally by the likelihood-ratio test; when $m=1$ it corresponds to binary classification; and when $m\approx n$ it is equivalent to two-sample testing. The intermediate settings occur in the field of likelihood-free inference, where labeled samples are obtained by running forward simulations and the unlabeled sample is collected experimentally. In recent work it was discovered that there is a fundamental trade-off between $m$ and $n$: increasing the data sample $m$ reduces the amount $n$ of training/simulation data needed. In this work we (a) introduce a generalization where unlabeled samples come from a mixture of the two classes -- a case often encountered in practice; (b) study the minimax 
    
[^28]: 论神经量子支持向量机

    On Neural Quantum Support Vector Machines. (arXiv:2308.08467v1 [quant-ph])

    [http://arxiv.org/abs/2308.08467](http://arxiv.org/abs/2308.08467)

    本文介绍了神经量子支持向量机，利用量子核，扩展了神经支持向量机的训练算法。

    

    在 \cite{simon2023algorithms} 中，我们介绍了四种用于训练神经支持向量机（NSVMs）的算法，并证明了其可行性。在本文中，我们引入了神经量子支持向量机，即具有量子核的NSVMs，并将我们的结果扩展到这个情景中。

    In \cite{simon2023algorithms} we introduced four algorithms for the training of neural support vector machines (NSVMs) and demonstrated their feasibility. In this note we introduce neural quantum support vector machines, that is, NSVMs with a quantum kernel, and extend our results to this setting.
    
[^29]: Real Robot Challenge 2022: 在真实世界中从离线数据中学习灵巧操作

    Real Robot Challenge 2022: Learning Dexterous Manipulation from Offline Data in the Real World. (arXiv:2308.07741v1 [cs.RO])

    [http://arxiv.org/abs/2308.07741](http://arxiv.org/abs/2308.07741)

    Real Robot Challenge 2022为RL和机器人学界之间的桥梁，允许参与者在真实机器人上从离线数据中学习灵巧操作任务，解决了在模拟中得到的见解不能转化到真实机器人的问题。

    

    在真实机器人上进行实验在时间和成本上要求很高。因此，增强学习（RL）社区的很大一部分使用模拟器来开发和评估算法。然而，从模拟中得到的见解不一定能够转化到真实机器人上，尤其是对于涉及复杂环境交互的任务。因此，Real Robot Challenge 2022作为RL和机器人学界之间的桥梁，让参与者能够像在模拟中一样轻松地远程实验真实机器人。在过去几年中，离线增强学习已经成熟为一种从预先收集的数据集中学习的有希望的范式，减轻了对昂贵在线交互的依赖.因此，我们要求参与者从提供的真实机器人数据集中学习两个灵巧操作任务，包括推动、抓取和手内定位。进行了广泛的软件文档化，并在基于仿真的初步阶段进行了实验。

    Experimentation on real robots is demanding in terms of time and costs. For this reason, a large part of the reinforcement learning (RL) community uses simulators to develop and benchmark algorithms. However, insights gained in simulation do not necessarily translate to real robots, in particular for tasks involving complex interactions with the environment. The Real Robot Challenge 2022 therefore served as a bridge between the RL and robotics communities by allowing participants to experiment remotely with a real robot - as easily as in simulation.  In the last years, offline reinforcement learning has matured into a promising paradigm for learning from pre-collected datasets, alleviating the reliance on expensive online interactions. We therefore asked the participants to learn two dexterous manipulation tasks involving pushing, grasping, and in-hand orientation from provided real-robot datasets. An extensive software documentation and an initial stage based on a simulation of the re
    
[^30]: 社会AI学校：从发展心理学到人工社会文化代理的观点

    The SocialAI School: Insights from Developmental Psychology Towards Artificial Socio-Cultural Agents. (arXiv:2307.07871v1 [cs.AI])

    [http://arxiv.org/abs/2307.07871](http://arxiv.org/abs/2307.07871)

    该论文讨论了AI研究应该受发展心理学启发，并研究使代理能够进入文化的社会认知能力。提出了社会AI学校工具以便于进行相关实验。

    

    发展心理学家长期以来已经确立了社会认知能力在人类智力中的重要性。这些能力使我们能够进入、参与和从人类文化中受益。社会交互代理的AI研究大多关注多智能体环境中文化的出现（通常没有强烈的发展心理学基础）。我们认为AI研究应该受心理学启发，并研究能够进入文化的社会认知能力。我们讨论了Michael Tomasello和Jerome Bruner的理论，介绍了他们的一些概念，并概述了关键概念和社会认知能力。我们提出了社会AI学校——一个包括定制参数化环境的工具，简化了关于这些概念的实验。我们展示了使用RL代理和大型语言模型进行此类实验的示例。这项工作的主要动机是吸引AI社区围绕这些概念进行讨论和研究。

    Developmental psychologists have long-established the importance of socio-cognitive abilities in human intelligence. These abilities enable us to enter, participate and benefit from human culture. AI research on social interactive agents mostly concerns the emergence of culture in a multi-agent setting (often without a strong grounding in developmental psychology). We argue that AI research should be informed by psychology and study socio-cognitive abilities enabling to enter a culture too. We discuss the theories of Michael Tomasello and Jerome Bruner to introduce some of their concepts to AI and outline key concepts and socio-cognitive abilities. We present The SocialAI school - a tool including a customizable parameterized uite of procedurally generated environments, which simplifies conducting experiments regarding those concepts. We show examples of such experiments with RL agents and Large Language Models. The main motivation of this work is to engage the AI community around the 
    
[^31]: LeCo：通过学习序列相关性实现轻量级压缩

    LeCo: Lightweight Compression via Learning Serial Correlations. (arXiv:2306.15374v1 [cs.DB])

    [http://arxiv.org/abs/2306.15374](http://arxiv.org/abs/2306.15374)

    使用机器学习来自动消除序列冗余以实现出色的压缩比和解压缩性能的LeCo是一种轻量级压缩框架，通过学习序列相关性，它能够在压缩比和随机访问速度上实现帕累托改进。

    

    轻量级数据压缩是一种关键技术，它使得列式存储在分析查询方面展示出卓越的性能。尽管之前有关基于字典编码来逼近Shannon熵的研究已经很全面，但鲜有之前的工作系统地利用列的序列相关性来进行压缩。在本文中，我们提出了LeCo（即学习压缩），这是一种使用机器学习自动消除序列冗余以实现出色的压缩比和解压缩性能的框架。LeCo提供了一种通用的方法来实现这一目标，在我们的框架下，现有的（临时的）算法，如参考帧（Frame-of-Reference），Delta编码和游程编码（Run-Length Encoding）都是特例。我们对三个合成数据集和六个真实数据集进行的微基准测试显示，LeCo原型在压缩比和随机访问速度上相比现有解决方案取得了帕累托改进。当将LeCo集成时

    Lightweight data compression is a key technique that allows column stores to exhibit superior performance for analytical queries. Despite a comprehensive study on dictionary-based encodings to approach Shannon's entropy, few prior works have systematically exploited the serial correlation in a column for compression. In this paper, we propose LeCo (i.e., Learned Compression), a framework that uses machine learning to remove the serial redundancy in a value sequence automatically to achieve an outstanding compression ratio and decompression performance simultaneously. LeCo presents a general approach to this end, making existing (ad-hoc) algorithms such as Frame-of-Reference (FOR), Delta Encoding, and Run-Length Encoding (RLE) special cases under our framework. Our microbenchmark with three synthetic and six real-world data sets shows that a prototype of LeCo achieves a Pareto improvement on both compression ratio and random access speed over the existing solutions. When integrating LeC
    
[^32]: 等变流匹配

    Equivariant flow matching. (arXiv:2306.15030v1 [stat.ML])

    [http://arxiv.org/abs/2306.15030](http://arxiv.org/abs/2306.15030)

    本文介绍了一种基于最优输运流匹配的等变CNF训练目标，可以提高等变CNF的可扩展性和实际应用。

    

    标准化流是一类特别适用于物理学中概率分布建模的深度生成模型。其中，流的准确似然性质可以实现对已知目标能量函数的加权重重和无偏观测量的计算。例如，Boltzmann生成器通过训练流生成处于平衡状态的多体系统（如小分子和蛋白质）样本，解决了统计物理学中长期存在的采样问题。为了构建有效的模型，也很关键将目标能量的对称性纳入模型中，这可以通过等变连续标准化流（CNF）来实现。然而，CNF的训练和样本生成的计算开销较大，这限制了它们的可扩展性和实际应用。在本文中，我们引入了等变流匹配，一种新的等变CNF训练目标，其基于最近提出的最优输运流匹配方法。

    Normalizing flows are a class of deep generative models that are especially interesting for modeling probability distributions in physics, where the exact likelihood of flows allows reweighting to known target energy functions and computing unbiased observables. For instance, Boltzmann generators tackle the long-standing sampling problem in statistical physics by training flows to produce equilibrium samples of many-body systems such as small molecules and proteins. To build effective models for such systems, it is crucial to incorporate the symmetries of the target energy into the model, which can be achieved by equivariant continuous normalizing flows (CNFs). However, CNFs can be computationally expensive to train and generate samples from, which has hampered their scalability and practical application. In this paper, we introduce equivariant flow matching, a new training objective for equivariant CNFs that is based on the recently proposed optimal transport flow matching. Equivarian
    
[^33]: 可训练的压缩嵌入层及其在推荐系统上的应用综述

    Review of compressed embedding layers and their applications for recommender systems. (arXiv:2306.13724v1 [cs.LG])

    [http://arxiv.org/abs/2306.13724](http://arxiv.org/abs/2306.13724)

    论文综述了可训练的、压缩的嵌入层在压缩大型神经网络推荐系统中的应用，并提供了相关实验结果。

    

    我们回顾了可训练的、压缩的嵌入层的文献，并讨论了它们在压缩巨型神经推荐系统方面的适用性。我们还报告了使用我们的压缩嵌入层所测得的结果。

    We review the literature on trainable, compressed embedding layers and discuss their applicability for compressing gigantic neural recommender systems. We also report the results we measured with our compressed embedding layers.
    
[^34]: 拓展可持续人工智能的视角： 人工智能系统的综合可持续性标准和指标

    Broadening the perspective for sustainable AI: Comprehensive sustainability criteria and indicators for AI systems. (arXiv:2306.13686v1 [cs.CY])

    [http://arxiv.org/abs/2306.13686](http://arxiv.org/abs/2306.13686)

    本文提出了SCAIS框架，包含一组19个可持续性标准和67个指标，旨在促进和结构化关于可持续人工智能的讨论。这种跨学科方法为实现人工智能系统的可持续发展提供了基础。

    

    人工智能系统的增加使用导致了多方面的社会、环境和经济后果，包括非透明的决策过程、歧视、不平等加剧、人工智能模型的能量消耗和温室气体排放，以及经济实力的集中。本文通过考虑可持续发展的多方面性，为“可持续人工智能”的理念提供了实质性的支持。提出了SCAIS框架（人工智能系统的可持续性标准和指标），其中包含一组19个可持续性标准和67个指标，这些标准和指标基于批判性审查和专家研讨的结果。这种跨学科方法为促进和结构化关于可持续人工智能的讨论提供了独特的整体性视角。此外，它提供了一个具体框架，为AI系统的后续发展和评估打下了基础。

    The increased use of AI systems is associated with multi-faceted societal, environmental, and economic consequences. These include non-transparent decision-making processes, discrimination, increasing inequalities, rising energy consumption and greenhouse gas emissions in AI model development and application, and an increasing concentration of economic power. By considering the multi-dimensionality of sustainability, this paper takes steps towards substantiating the call for an overarching perspective on "sustainable AI". It presents the SCAIS Framework (Sustainability Criteria and Indicators for Artificial Intelligence Systems) which contains a set 19 sustainability criteria for sustainable AI and 67 indicators that is based on the results of a critical review and expert workshops. This interdisciplinary approach contributes a unique holistic perspective to facilitate and structure the discourse on sustainable AI. Further, it provides a concrete framework that lays the foundation for 
    
[^35]: 对最坏情况下游任务适应性的任务鲁棒预训练

    Task-Robust Pre-Training for Worst-Case Downstream Adaptation. (arXiv:2306.12070v1 [cs.CV])

    [http://arxiv.org/abs/2306.12070](http://arxiv.org/abs/2306.12070)

    本文提出了一种任务鲁棒的预训练方法，将上游任务分成几个代表性任务并应用极小极大损失进行预训练，以保证模型能够在下游任务中具有均匀良好的性能。

    

    预训练在转移到下游任务时取得了显着的成功。在机器学习中，我们关心模型不仅具有良好的性能，而且在合理的条件变化下的行为。当预训练基础模型时，同样的哲学也适用。然而，基础模型可能并不会在一系列相关下游任务中均匀地表现良好。本文考虑预训练一个模型，保证其在下游任务中具有均匀良好的性能，我们称此目标为下游任务鲁棒性。我们的方法首先将上游任务分成几个代表性任务，并应用简单的minimax loss 进行预训练，然后设计了一个高效的算法来解决极小极大问题，并表明我们的方法优于先前的基线。

    Pre-training has achieved remarkable success when transferred to downstream tasks. In machine learning, we care about not only the good performance of a model but also its behavior under reasonable shifts of condition. The same philosophy holds when pre-training a foundation model. However, the foundation model may not uniformly behave well for a series of related downstream tasks. This happens, for example, when conducting mask recovery regression where the recovery ability or the training instances diverge like pattern features are extracted dominantly on pre-training, but semantic features are also required on a downstream task. This paper considers pre-training a model that guarantees a uniformly good performance over the downstream tasks. We call this goal as $\textit{downstream-task robustness}$. Our method first separates the upstream task into several representative ones and applies a simple minimax loss for pre-training. We then design an efficient algorithm to solve the minim
    
[^36]: 高斯过程网络的贝叶斯方法

    A Bayesian Take on Gaussian Process Networks. (arXiv:2306.11380v1 [stat.ML])

    [http://arxiv.org/abs/2306.11380](http://arxiv.org/abs/2306.11380)

    该论文提出了一种基于高斯过程和贝叶斯方法的网络模型，通过蒙特卡罗和马尔可夫链蒙特卡罗方法采样网络结构的后验分布。该方法在恢复网络的图形结构方面优于最先进的算法，并提供了后验概率的准确近似。

    

    高斯过程网络（GPNs）是一类有向图模型，其使用高斯过程作为网络中每个变量给定其父变量的条件期望的先验分布。该模型允许以紧凑但灵活的方式描述连续联合分布，对变量之间的依赖关系仅做最少的参数假设。GPNs的贝叶斯结构学习需要计算网络结构的后验分布，即使在低维情况下，这也是计算上不可行的。本文实现了蒙特卡罗和马尔可夫链蒙特卡罗方法来从网络结构的后验分布中采样。因此，该方法遵循贝叶斯范式，通过边缘似然比较模型，并计算GPN特征的后验概率。模拟研究表明，我们的方法在恢复网络的图形结构方面优于最先进的算法，并提供其后验的准确近似。

    Gaussian Process Networks (GPNs) are a class of directed graphical models which employ Gaussian processes as priors for the conditional expectation of each variable given its parents in the network. The model allows describing continuous joint distributions in a compact but flexible manner with minimal parametric assumptions on the dependencies between variables. Bayesian structure learning of GPNs requires computing the posterior over graphs of the network and is computationally infeasible even in low dimensions. This work implements Monte Carlo and Markov Chain Monte Carlo methods to sample from the posterior distribution of network structures. As such, the approach follows the Bayesian paradigm, comparing models via their marginal likelihood and computing the posterior probability of the GPN features. Simulation studies show that our method outperforms state-of-the-art algorithms in recovering the graphical structure of the network and provides an accurate approximation of its poste
    
[^37]: 区块链支持的联邦学习：参考架构设计、实现和验证

    Blockchain-Enabled Federated Learning: A Reference Architecture Design, Implementation, and Verification. (arXiv:2306.10841v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.10841](http://arxiv.org/abs/2306.10841)

    本文提出了一种基于区块链的联邦学习参考架构，通过结合联邦学习和区块链技术，实现了去中心化、协作的机器学习系统，并保护了数据隐私和用户控制的身份。该架构使用去中心化标识符进行身份验证，通过智能合约实现强大的安全性和高效的去中心化，并能根据需求集成各种额外的元素，是一个适用范围广泛的 BCFL 解决方案。

    

    本文提出了一种创新的基于区块链的联邦学习（BCFL）参考架构，该架构将联邦学习和区块链技术的优势结合起来。这导致了一个去中心化的、协作的机器学习系统，尊重数据隐私和用户控制的身份。我们的架构战略性地采用基于去中心化标识符（DID）的身份验证系统，允许参与者使用其自主 DID 安全地认证并获得对联邦学习平台的访问权限，这些信息被记录在区块链上。通过执行智能合约来确保强大的安全性和高效的去中心化是我们方法的关键方面。此外，我们的 BCFL 参考架构提供了显著的可扩展性，能够根据特定需求和用例集成各种额外的元素，使其成为广泛适用的 BCFL 解决方案。

    This paper presents an innovative reference architecture for blockchain-enabled federated learning (BCFL), a state-of-the-art approach that amalgamates the strengths of federated learning and blockchain technology. This results in a decentralized, collaborative machine learning system that respects data privacy and user-controlled identity. Our architecture strategically employs a decentralized identifier (DID)-based authentication system, allowing participants to authenticate and then gain access to the federated learning platform securely using their self-sovereign DIDs, which are recorded on the blockchain. Ensuring robust security and efficient decentralization through the execution of smart contracts is a key aspect of our approach. Moreover, our BCFL reference architecture provides significant extensibility, accommodating the integration of various additional elements, as per specific requirements and use cases, thereby rendering it an adaptable solution for a wide range of BCFL 
    
[^38]: MARBLE：音乐音频表征通用评估基准

    MARBLE: Music Audio Representation Benchmark for Universal Evaluation. (arXiv:2306.10548v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2306.10548](http://arxiv.org/abs/2306.10548)

    本论文介绍了MARBLE，一个音乐音频表征通用评估基准，它为音乐理解领域的研究和发展提供了一个全面和可持续性的基础，并提供各种音乐信息检索（MIR）任务的基准。

    

    在艺术与人工智能（AI）之间交叉的广泛时代中，例如图像生成和虚构共创，音乐的AI仍然相对初步，特别是在音乐理解方面。针对这个问题，本论文介绍了一个通用的音乐音频表征评估基准MARBLE，旨在提供各种音乐信息检索（MIR）任务的基准，通过定义包括声学，演奏，乐谱和高级描述在内的四个层次的综合分类法。然后，我们基于8个公共可用数据集上的14项任务建立了一个统一的协议，提供了所有基于音乐录音开发的开放源代码的预训练模型的表征的公平和标准的评估。此外，MARBLE提供了一个易于使用、可扩展和可重用的工具库，以支持社区驱动的客观基准评估。

    In the era of extensive intersection between art and Artificial Intelligence (AI), such as image generation and fiction co-creation, AI for music remains relatively nascent, particularly in music understanding. This is evident in the limited work on deep music representations, the scarcity of large-scale datasets, and the absence of a universal and community-driven benchmark. To address this issue, we introduce the Music Audio Representation Benchmark for universaL Evaluation, termed MARBLE. It aims to provide a benchmark for various Music Information Retrieval (MIR) tasks by defining a comprehensive taxonomy with four hierarchy levels, including acoustic, performance, score, and high-level description. We then establish a unified protocol based on 14 tasks on 8 public-available datasets, providing a fair and standard assessment of representations of all open-sourced pre-trained models developed on music recordings as baselines. Besides, MARBLE offers an easy-to-use, extendable, and re
    
[^39]: PEAR: 基于原始操作的自适应重标记用于Boosting层次强化学习

    PEAR: Primitive enabled Adaptive Relabeling for boosting Hierarchical Reinforcement Learning. (arXiv:2306.06394v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.06394](http://arxiv.org/abs/2306.06394)

    PEAR是一种基于原始操作的自适应重标记方法，用于Boosting层次强化学习。它通过对专家演示进行自适应重标记来生成高效的子目标监督，并通过联合优化强化学习和模仿学习来训练分层代理。实验结果显示，PEAR能够在具有挑战性的机器人环境中取得良好的性能。

    

    层次强化学习（HRL）利用时间抽象和增加的探索性能解决复杂的长期任务。然而，由于固有的非静态性，分层代理难以训练。我们提出了基于原始操作的自适应重标记（PEAR），这是一个两阶段方法，我们首先对少量专家演示进行自适应重标记，产生高效的子目标监督，然后通过使用强化学习（RL）和模仿学习（IL）联合优化HRL代理。我们进行理论分析来$(i)$限制我们方法的次优性，和$(ii)$推导出使用RL和IL的广义即插即用的框架进行联合优化。PEAR使用一些专家演示，并对任务结构进行最小的限制假设。此外，它可以轻松与典型的模型自由RL算法集成，产生一个实用的HRL算法。我们在具有挑战性的机器人环境上进行了实验。

    Hierarchical reinforcement learning (HRL) has the potential to solve complex long horizon tasks using temporal abstraction and increased exploration. However, hierarchical agents are difficult to train due to inherent non-stationarity. We present primitive enabled adaptive relabeling (PEAR), a two-phase approach where we first perform adaptive relabeling on a few expert demonstrations to generate efficient subgoal supervision, and then jointly optimize HRL agents by employing reinforcement learning (RL) and imitation learning (IL). We perform theoretical analysis to $(i)$ bound the sub-optimality of our approach, and $(ii)$ derive a generalized plug-and-play framework for joint optimization using RL and IL. PEAR uses a handful of expert demonstrations and makes minimal limiting assumptions on the task structure. Additionally, it can be easily integrated with typical model free RL algorithms to produce a practical HRL algorithm. We perform experiments on challenging robotic environments
    
[^40]: 用于组合优化的神经算法推理

    Neural Algorithmic Reasoning for Combinatorial Optimisation. (arXiv:2306.06064v1 [cs.NE])

    [http://arxiv.org/abs/2306.06064](http://arxiv.org/abs/2306.06064)

    本文提出了一种用于组合优化问题的神经算法推理方法，旨在解决旅行商问题。该方法是通过在TSP实例训练之前，将神经模型用相关算法进行预训练来实现的。实验结果表明，该方法可以显著提高TSP问题的解决效率。

    

    使用神经网络解决NP难/完全组合问题是一个挑战性的研究领域，旨在超越传统的近似算法。其长期目标是通过学习仅从训练数据生成更优解来超越手工设计的启发式算法，而旅行商问题(TSP)是经常被这些方法瞄准的一个重要的组合优化问题。然而，目前用于解决TSP的基于神经网络的方法常常忽略了问题固有的“算法”本质。与此相反，设计用于TSP的启发式方法常常利用诸如查找最小生成树之类的成熟算法。在本文中，我们提出利用神经算法推理的最新进展来改进TSP问题的学习。具体来说，我们建议在对TSP实例进行训练之前，在相关算法上对我们的神经模型进行预训练。我们的结果表明，使用这种学习方法可以显著提高TSP问题的解决效率。

    Solving NP-hard/complete combinatorial problems with neural networks is a challenging research area that aims to surpass classical approximate algorithms. The long-term objective is to outperform hand-designed heuristics for NP-hard/complete problems by learning to generate superior solutions solely from training data. The Travelling Salesman Problem (TSP) is a prominent combinatorial optimisation problem often targeted by such approaches. However, current neural-based methods for solving TSP often overlook the inherent "algorithmic" nature of the problem. In contrast, heuristics designed for TSP frequently leverage well-established algorithms, such as those for finding the minimum spanning tree. In this paper, we propose leveraging recent advancements in neural algorithmic reasoning to improve the learning of TSP problems. Specifically, we suggest pre-training our neural model on relevant algorithms before training it on TSP instances. Our results demonstrate that, using this learning
    
[^41]: 非线性循环神经网络的逆近似理论

    Inverse Approximation Theory for Nonlinear Recurrent Neural Networks. (arXiv:2305.19190v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.19190](http://arxiv.org/abs/2305.19190)

    该论文证明了使用RNNs逼近非线性序列关系的逆近似定理，进一步将先前在线性RNNs中识别出的记忆难题推广到了一般的非线性情况，并提出了一个有原则的重新参数化方法来克服这些限制。

    

    我们证明了使用RNNs来逼近非线性序列关系的逆近似定理。这是近似理论中的一种称为Bernstein型结果的结果，它在假设目标函数可以通过假设空间有效逼近的条件下推导出目标函数的属性。特别地，我们展示了非线性序列关系可以被具有hardtanh/tanh激活函数的RNNs稳定逼近的时候，必须具有一个指数衰减的记忆结构--这个概念可以被明确定义。这将先前在线性RNNs中识别出的记忆难题推广到了一般的非线性情况，并量化了RNN架构在学习具有长期记忆的序列关系时的重要限制。基于分析，我们提出了一个有原则的重新参数化方法来克服这些限制。我们的理论结果通过数值实验进行了确认。

    We prove an inverse approximation theorem for the approximation of nonlinear sequence-to-sequence relationships using RNNs. This is a so-called Bernstein-type result in approximation theory, which deduces properties of a target function under the assumption that it can be effectively approximated by a hypothesis space. In particular, we show that nonlinear sequence relationships, viewed as functional sequences, that can be stably approximated by RNNs with hardtanh/tanh activations must have an exponential decaying memory structure -- a notion that can be made precise. This extends the previously identified curse of memory in linear RNNs into the general nonlinear setting, and quantifies the essential limitations of the RNN architecture for learning sequential relationships with long-term memory. Based on the analysis, we propose a principled reparameterization method to overcome the limitations. Our theoretical results are confirmed by numerical experiments.
    
[^42]: 可解释机器学习在类别和混合数据上的应用：无损可视化

    Explainable Machine Learning for Categorical and Mixed Data with Lossless Visualization. (arXiv:2305.18437v1 [cs.LG])

    [http://arxiv.org/abs/2305.18437](http://arxiv.org/abs/2305.18437)

    本文提出了一些数值编码和可视化方法，以支持机器学习算法处理混合数据，并提出了可解释的多分类模型和SRG算法来生成解释性分类模型。

    

    为混合数据构建准确可解释的机器学习模型一直是算法面对非数值数据的挑战。本文提出了数值编码方案和无损可视化方法，为机器学习算法支持准确可解释的模型，提出了可解释的多分类模型和演示其重要作用。论文提出了一种分类混合数据类型的方法，并提出了一种工具包，以对混合数据的所有内部操作实现可解释性。论文还提出了一种新的“顺序规则生成（SRG）”算法，用于生成可解释的分类模型，并在多个计算实验中成功评估该算法。

    Building accurate and interpretable Machine Learning (ML) models for heterogeneous/mixed data is a long-standing challenge for algorithms designed for numeric data. This work focuses on developing numeric coding schemes for non-numeric attributes for ML algorithms to support accurate and explainable ML models, methods for lossless visualization of n-D non-numeric categorical data with visual rule discovery in these visualizations, and accurate and explainable ML models for categorical data. This study proposes a classification of mixed data types and analyzes their important role in Machine Learning. It presents a toolkit for enforcing interpretability of all internal operations of ML algorithms on mixed data with a visual data exploration on mixed data. A new Sequential Rule Generation (SRG) algorithm for explainable rule generation with categorical data is proposed and successfully evaluated in multiple computational experiments. This work is one of the steps to the full scope ML alg
    
[^43]: FAVAS: 带有异步客户端的联邦平均的新型中心化框架

    FAVAS: Federated AVeraging with ASynchronous clients. (arXiv:2305.16099v1 [cs.LG])

    [http://arxiv.org/abs/2305.16099](http://arxiv.org/abs/2305.16099)

    本研究提出了FAVAS算法，是一种用于在资源有限环境下训练DNNs的新型中心化异步联邦学习框架。实验结果表明FAVAS算法优于当前方法。

    

    本文提出了一种新型的中心化异步联邦学习框架FAVAS，用于在资源有限的环境下训练深度神经网络。尽管联邦学习越来越受欢迎，但在大型无线网络上伸缩同步通信变得越来越困难。此外，由于客户端通常具有不同的计算资源和计算速度，异步更新可能会导致显着的偏差（对“快速”客户端更有利）。因此，FL的实际部署需要处理在通信/资源受限的环境中具有强烈变化的计算速度的用户。我们提供了FAVAS在平滑的非凸环境中的收敛性保证，并仔细比较了获得的收敛保证与现有边界（如果有）的差异。实验结果表明，FAVAS算法在标准基准测试中优于当前方法。

    In this paper, we propose a novel centralized Asynchronous Federated Learning (FL) framework, FAVAS, for training Deep Neural Networks (DNNs) in resource-constrained environments. Despite its popularity, ``classical'' federated learning faces the increasingly difficult task of scaling synchronous communication over large wireless networks. Moreover, clients typically have different computing resources and therefore computing speed, which can lead to a significant bias (in favor of ``fast'' clients) when the updates are asynchronous. Therefore, practical deployment of FL requires to handle users with strongly varying computing speed in communication/resource constrained setting. We provide convergence guarantees for FAVAS in a smooth, non-convex environment and carefully compare the obtained convergence guarantees with existing bounds, when they are available. Experimental results show that the FAVAS algorithm outperforms current methods on standard benchmarks.
    
[^44]: 学习具有物理一致性的异质性粒子相互作用的集体关系推断

    Collective Relational Inference for learning physics-consistent heterogeneous particle interactions. (arXiv:2305.00557v1 [cs.LG])

    [http://arxiv.org/abs/2305.00557](http://arxiv.org/abs/2305.00557)

    本论文提出了一种新的概率方法用于学习异质性粒子相互作用的集体关系推断，与现有方法相比，该方法集体地推断不同边的相互作用类型，使用物理感应的图神经网络来学习具有物理一致性的成对相互作用，并在推断准确性和保持物理保真度方面一致优于现有方法。

    

    相互作用粒子系统在自然界和工程中无处不在。揭示粒子相互作用定律具有基本重要性，但由于底层配置复杂性而具有极大的挑战性。最近开发的机器学习方法在发现同质系统粒子轨迹中的成对相互作用方面显示出极大的潜力。然而，它们无法揭示异质系统中的相互作用，而这种系统在现实中普遍存在，其中多个相互作用类型同时存在，并且需要关系推断。在这里，我们提出了一种新的概率方法用于关系推断，与现有方法相比，具有两个独特的特征：首先，它集体地推断不同边的相互作用类型；其次，它使用物理感应的图神经网络来学习具有物理一致性的成对相互作用。我们在几个基准数据集上评估了所提出的方法，并证明其在推断的相互作用准确性和保持物理保真度方面一致优于现有方法。具体而言，我们的方法确定了具有重要物理意义的新型相互作用类型，揭示了统治系统的隐藏物理原理，并在提高物理性质的预测方面显示出极大的潜力。

    Interacting particle systems are ubiquitous in nature and engineering. Revealing particle interaction laws is of fundamental importance but also particularly challenging due to underlying configurational complexities. Recently developed machine learning methods show great potential in discovering pairwise interactions from particle trajectories in homogeneous systems. However, they fail to reveal interactions in heterogeneous systems that are prevalent in reality, where multiple interaction types coexist simultaneously and relational inference is required. Here, we propose a novel probabilistic method for relational inference, which possesses two distinctive characteristics compared to existing methods. First, it infers the interaction types of different edges collectively, and second, it uses a physics-induced graph neural network to learn physics-consistent pairwise interactions. We evaluate the proposed methodology across several benchmark datasets and demonstrate that it is consist
    
[^45]: 深度集合在多输出回归任务中量化校准不确定性的探究

    Towards Quantifying Calibrated Uncertainty via Deep Ensembles in Multi-output Regression Task. (arXiv:2303.16210v1 [cs.LG])

    [http://arxiv.org/abs/2303.16210](http://arxiv.org/abs/2303.16210)

    本研究探究了在多输出回归任务中应用深度集合量化校准不确定性的方法，提出了该方法的改进框架，其在回归准确性、不确定性估计可靠性和训练效率方面具有优越表现。

    

    深度集合是逼近贝叶斯推断的一种简单直接的方法，已被成功应用于许多分类任务。本研究旨在全面探究该方法在多输出回归任务中的应用，以预测导弹结构的空气动力性能。通过仔细研究集合中神经网络数量的影响，观察到估计的不确定性普遍存在低估的趋势。在此背景下，提出了一种应用事后校准的深度集合框架，并证明其改进的不确定性量化性能。直观地将其与高斯过程回归进行比较，这是工程中最常用的不确定性量化模型，结果表明在回归准确性、估计不确定性的可靠性和训练效率方面具有卓越的表现。最后，本文也研究了所提出框架对贝叶斯优化结果的影响。

    Deep ensemble is a simple and straightforward approach for approximating Bayesian inference and has been successfully applied to many classification tasks. This study aims to comprehensively investigate this approach in the multi-output regression task to predict the aerodynamic performance of a missile configuration. By scrutinizing the effect of the number of neural networks used in the ensemble, an obvious trend toward underconfidence in estimated uncertainty is observed. In this context, we propose the deep ensemble framework that applies the post-hoc calibration method, and its improved uncertainty quantification performance is demonstrated. It is compared with Gaussian process regression, the most prevalent model for uncertainty quantification in engineering, and is proven to have superior performance in terms of regression accuracy, reliability of estimated uncertainty, and training efficiency. Finally, the impact of the suggested framework on the results of Bayesian optimizatio
    
[^46]: 面向城市计算的时空图神经网络预测学习综述

    Spatio-Temporal Graph Neural Networks for Predictive Learning in Urban Computing: A Survey. (arXiv:2303.14483v1 [cs.LG])

    [http://arxiv.org/abs/2303.14483](http://arxiv.org/abs/2303.14483)

    本综述介绍了面向城市计算的时空图神经网络预测学习领域的发展现状，包括其框架、实现方法和应用场景，以及当前的研究热点和挑战，提出了该领域未来的发展方向和应用前景。

    

    随着先进传感器和大型数据库技术的发展，越来越多的城市系统时空数据被记录和存储。这些数据的演化模式的预测学习是城市计算中基本但重要的循环，可以更好地支持城市智能管理决策，特别是在交通、环境、安全、公共卫生等领域。由于传统的统计学习和深度学习方法很难捕捉城市时空数据的复杂相关性，近年来提出了时空图神经网络（STGNN）的框架。STGNN通过集成图神经网络（GNN）和各种时间学习方法实现了复杂时空依赖关系的提取。然而，对于不同的预测学习任务，有效设计空间依赖学习模块、时间依赖学习模块、以及它们之间相互作用的方法仍然具有挑战性。

    With the development of sophisticated sensors and large database technologies, more and more spatio-temporal data in urban systems are recorded and stored. Predictive learning for the evolution patterns of these spatio-temporal data is a basic but important loop in urban computing, which can better support urban intelligent management decisions, especially in the fields of transportation, environment, security, public health, etc. Since traditional statistical learning and deep learning methods can hardly capture the complex correlations in the urban spatio-temporal data, the framework of spatio-temporal graph neural network (STGNN) has been proposed in recent years. STGNNs enable the extraction of complex spatio-temporal dependencies by integrating graph neural networks (GNNs) and various temporal learning methods. However, for different predictive learning tasks, it is a challenging problem to effectively design the spatial dependencies learning modules, temporal dependencies learnin
    
[^47]: 大规模统计学习模型有效地预测各种混沌系统

    Large statistical learning models effectively forecast diverse chaotic systems. (arXiv:2303.08011v1 [cs.LG])

    [http://arxiv.org/abs/2303.08011](http://arxiv.org/abs/2303.08011)

    该论文研究了混沌预测的大规模实验，发现基于人工神经网络的大规模、领域不可知的时间序列预测方法表现出了相当强大的性能，尤其是分层神经基础函数模型表现最佳。

    

    传统上混沌和不可预测是同义词，但最近统计预测的进展表明，大型机器学习模型可以从复杂系统的长时间观测中获得意想不到的见解。在本文中，我们对规模上的混沌预测进行了研究，通过对 135 种不同低维混沌系统的众包数据库进行 24 种代表性最高的多元预测方法的大规模比较。我们发现，基于人工神经网络的大规模的领域不可知时间序列预测方法始终展现出强大的预测性能，在某些情况下可以产生持续数十个李雅普诺夫时间的准确预测。最佳的混沌预测结果由最近引入的分层神经基础函数模型实现，但即使是通用的变压器和循环神经网络也表现出强大的性能。然而，物理启发式混合方法如神经常微分方程和储层计算机的性能更好，尤其是在更小的数据集上。

    Chaos and unpredictability are traditionally synonymous, yet recent advances in statistical forecasting suggest that large machine learning models can derive unexpected insight from extended observation of complex systems. Here, we study the forecasting of chaos at scale, by performing a large-scale comparison of 24 representative state-of-the-art multivariate forecasting methods on a crowdsourced database of 135 distinct low-dimensional chaotic systems. We find that large, domain-agnostic time series forecasting methods based on artificial neural networks consistently exhibit strong forecasting performance, in some cases producing accurate predictions lasting for dozens of Lyapunov times. Best-in-class results for forecasting chaos are achieved by recently-introduced hierarchical neural basis function models, though even generic transformers and recurrent neural networks perform strongly. However, physics-inspired hybrid methods like neural ordinary equations and reservoir computers c
    
[^48]: 可解释和能够干预的超声成像机器学习模型用于儿科阑尾炎

    Interpretable and Intervenable Ultrasonography-based Machine Learning Models for Pediatric Appendicitis. (arXiv:2302.14460v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.14460](http://arxiv.org/abs/2302.14460)

    本研究开发了可解释的机器学习模型，利用超声影像预测儿科疑似阑尾炎的诊断、管理和严重程度。模型使用了超声影像和临床、实验室数据进行训练，并推广了概念瓶颈模型到多视图和不完整概念集的预测问题。

    

    阑尾炎是儿科腹部手术的最常见原因之一。随着机器学习的最新进展，数据驱动的决策支持可以帮助临床医生诊断和管理患者，同时减少非关键手术的数量。以往的阑尾炎决策支持系统主要关注临床、实验室、评分和计算机断层扫描数据，主要忽视了腹部超声成像，这是一种非侵入性和方便获得的诊断方法。为此，我们开发和验证了利用超声影像预测疑似阑尾炎诊断、管理和严重程度的可解释机器学习模型。我们的模型使用了由579名儿科患者的1709幅超声影像，以及临床和实验室数据构成的数据集进行训练。我们的方法学贡献在于将概念瓶颈模型推广到具有多个视图和不完整概念集的预测问题。值得注意的是，这样的模型适用于干预操作。

    Appendicitis is among the most frequent reasons for pediatric abdominal surgeries. With recent advances in machine learning, data-driven decision support could help clinicians diagnose and manage patients while reducing the number of non-critical surgeries. Previous decision support systems for appendicitis focused on clinical, laboratory, scoring and computed tomography data, mainly ignoring abdominal ultrasound, a noninvasive and readily available diagnostic modality. To this end, we developed and validated interpretable machine learning models for predicting the diagnosis, management and severity of suspected appendicitis using ultrasound images. Our models were trained on a dataset comprising 579 pediatric patients with 1709 ultrasound images accompanied by clinical and laboratory data. Our methodological contribution is the generalization of concept bottleneck models to prediction problems with multiple views and incomplete concept sets. Notably, such models lend themselves to int
    
[^49]: 比较贝叶斯层次模型的深度学习方法

    A Deep Learning Method for Comparing Bayesian Hierarchical Models. (arXiv:2301.11873v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.11873](http://arxiv.org/abs/2301.11873)

    这个论文提出了一种深度学习方法，用于比较贝叶斯层次模型。该方法通过支持分摊推断，能够高效地进行模型比较和性能验证。同时，作者还对四个层次证据积累模型进行了比较。

    

    贝叶斯模型比较（BMC）提供了一种基于原则的方法来评估竞争计算模型的相对优势，并将不确定性传播到模型选择决策中。然而，由于高维嵌套参数结构，BMC在常见的层次模型中常常难以计算。为了解决这个难题，我们提出了一种深度学习方法，用于对任何可实例化为概率程序的层次模型集进行BMC。由于我们的方法支持分摊推断，它可以在任何实际数据应用之前，对后验模型概率进行高效的重新估计和快速性能验证。在一系列广泛的验证研究中，我们对比了我们的方法与最先进的桥式抽样方法的性能，并展示了在所有BMC设置中出色的分摊推断能力。然后，我们展示了我们的方法，通过比较先前被认为是四个层次证据积累模型。

    Bayesian model comparison (BMC) offers a principled approach for assessing the relative merits of competing computational models and propagating uncertainty into model selection decisions. However, BMC is often intractable for the popular class of hierarchical models due to their high-dimensional nested parameter structure. To address this intractability, we propose a deep learning method for performing BMC on any set of hierarchical models which can be instantiated as probabilistic programs. Since our method enables amortized inference, it allows efficient re-estimation of posterior model probabilities and fast performance validation prior to any real-data application. In a series of extensive validation studies, we benchmark the performance of our method against the state-of-the-art bridge sampling method and demonstrate excellent amortized inference across all BMC settings. We then showcase our method by comparing four hierarchical evidence accumulation models that have previously b
    
[^50]: 基于边界框的多目标贝叶斯优化在输入不确定性下的风险衡量

    Bounding Box-based Multi-objective Bayesian Optimization of Risk Measures under Input Uncertainty. (arXiv:2301.11588v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.11588](http://arxiv.org/abs/2301.11588)

    该论文提出了一种基于边界框的多目标贝叶斯优化方法，能够在输入不确定性下高效地识别风险衡量定义的帕累托前沿。该方法具有理论保证，并通过构建高概率边界框和选择下一个评估点的方法来减少不确定性。

    

    在本研究中，我们提出了一种新颖的多目标贝叶斯优化（MOBO）方法，用于在输入不确定性（IU）存在的情况下高效地识别由风险衡量定义的帕累托前沿（PF）。现有的IU下帕累托优化的BO方法是特定风险或者没有理论保证的，而我们提出的方法涉及一般风险衡量并具有理论保证。所提方法的基本思想是假设黑箱函数的高斯过程（GP）模型，并使用GP模型构建风险衡量的高概率边界框。此外，为了减少非支配边界框的不确定性，我们提出了一种使用基于边界框的拟距离的最大值定义的最大最小距离选择下一个评估点的方法。作为理论分析，我们证明了该算法可以在有限次迭代中返回任意精确的解。

    In this study, we propose a novel multi-objective Bayesian optimization (MOBO) method to efficiently identify the Pareto front (PF) defined by risk measures for black-box functions under the presence of input uncertainty (IU). Existing BO methods for Pareto optimization in the presence of IU are risk-specific or without theoretical guarantees, whereas our proposed method addresses general risk measures and has theoretical guarantees. The basic idea of the proposed method is to assume a Gaussian process (GP) model for the black-box function and to construct high-probability bounding boxes for the risk measures using the GP model. Furthermore, in order to reduce the uncertainty of non-dominated bounding boxes, we propose a method of selecting the next evaluation point using a maximin distance defined by the maximum value of a quasi distance based on bounding boxes. As theoretical analysis, we prove that the algorithm can return an arbitrary-accurate solution in a finite number of iterati
    
[^51]: 通过深度感知实现手持灵巧操作

    Visual Dexterity: In-hand Dexterous Manipulation from Depth. (arXiv:2211.11744v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2211.11744](http://arxiv.org/abs/2211.11744)

    通过使用深度相机的读数，我们提出了一种通用物体重新定向控制器，可以实时、动态地重新定向复杂和新颖的物体形状，中位数重新定向时间接近于七秒。该控制器经过强化学习在仿真环境中训练，并在实际世界中对未用于训练的新物体形状进行了评估。

    

    手持物体的重新定向对于执行许多灵巧操作任务非常必要，例如在当前机器人无法触及的结构不太完善的环境中使用工具。之前的研究建立了重新定向系统，假设以下情况之一或多种情况同时存在：仅重新定向具有简单形状的特定物体、重新定向范围有限、慢速或准静态操作、仅模拟结果、需要专用且昂贵的传感器套件以及其他不适用于实际部署的限制。我们提出了一种不做这些假设的通用物体重新定向控制器。它使用来自单个普通深度摄像机的读数，以实时方式通过任意旋转动态重新定向复杂且新颖的物体形状，中位数重新定向时间接近于七秒。该控制器经过强化学习在仿真环境中进行训练，并在未用于训练的新物体形状上在实际世界中进行评估，包括 ...

    In-hand object reorientation is necessary for performing many dexterous manipulation tasks, such as tool use in less structured environments that remain beyond the reach of current robots. Prior works built reorientation systems assuming one or many of the following: reorienting only specific objects with simple shapes, limited range of reorientation, slow or quasistatic manipulation, simulation-only results, the need for specialized and costly sensor suites, and other constraints which make the system infeasible for real-world deployment. We present a general object reorientation controller that does not make these assumptions. It uses readings from a single commodity depth camera to dynamically reorient complex and new object shapes by any rotation in real-time, with the median reorientation time being close to seven seconds. The controller is trained using reinforcement learning in simulation and evaluated in the real world on new object shapes not used for training, including the m
    
[^52]: 基于反事实分析的监督特征压缩

    Supervised Feature Compression based on Counterfactual Analysis. (arXiv:2211.09894v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.09894](http://arxiv.org/abs/2211.09894)

    该论文提出了一种基于反事实分析的监督特征压缩方法，利用此方法可以构建出类似于黑盒模型最优决策树，该决策树具备可解释性和紧凑性，并在真实数据集上有效。

    

    反事实解释已成为事后可解释机器学习的事实标准。该工作旨在利用反事实解释识别预训练黑盒模型的重要决策边界。该信息用于在数据集中构建一种可调整细度的特征的监督离散化。使用离散化的数据集，可以训练出类似于黑盒模型的最优决策树，但具有可解释性和紧凑性。在真实数据集上的数值实验表明了该方法在准确性和稀疏性方面的有效性。

    Counterfactual Explanations are becoming a de-facto standard in post-hoc interpretable machine learning. For a given classifier and an instance classified in an undesired class, its counterfactual explanation corresponds to small perturbations of that instance that allows changing the classification outcome. This work aims to leverage Counterfactual Explanations to detect the important decision boundaries of a pre-trained black-box model. This information is used to build a supervised discretization of the features in the dataset with a tunable granularity. Using the discretized dataset, an optimal Decision Tree can be trained that resembles the black-box model, but that is interpretable and compact. Numerical results on real-world datasets show the effectiveness of the approach in terms of accuracy and sparsity.
    
[^53]: 利用迁移学习像病理学家一样自动评分组织图像

    Automatically Score Tissue Images Like a Pathologist by Transfer Learning. (arXiv:2209.05954v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.05954](http://arxiv.org/abs/2209.05954)

    该算法通过选择性迁移学习从多个小辅助集中提取知识，从具有“相似”特征的组织图像中学习染色模式，以实现像病理学家一样自动评分组织图像的目标。

    

    癌症是全球第二大死亡原因。早期诊断癌症可以挽救很多生命。病理学家必须手动查看组织微阵列 (TMA) 图像以识别肿瘤，这可能会耗费时间、不一致且主观。现有的自动检测肿瘤的算法要么没有达到病理学家的准确水平，要么需要大量的人工参与。主要挑战是具有不同形状、大小和位置的 TMA 图像可能具有相同的得分。由于医疗组织中的隐私问题和限制，学习 TMA 图像的染色模式受到严重限制。来自不同癌症类型的 TMA 图像可能具有共同的特征，提供了有价值的信息，但直接使用会损害准确性。通过选择性迁移学习来自多个小辅助集的知识，所提出的算法能够提取显示“类似”的组织图像的知识，从而在TMA图像评分方面取得了很好的成果。

    Cancer is the second leading cause of death in the world. Diagnosing cancer early on can save many lives. Pathologists have to look at tissue microarray (TMA) images manually to identify tumors, which can be time-consuming, inconsistent and subjective. Existing algorithms that automatically detect tumors have either not achieved the accuracy level of a pathologist or require substantial human involvements. A major challenge is that TMA images with different shapes, sizes, and locations can have the same score. Learning staining patterns in TMA images requires a huge number of images, which are severely limited due to privacy concerns and regulations in medical organizations. TMA images from different cancer types may have common characteristics that could provide valuable information, but using them directly harms the accuracy. By selective transfer learning from multiple small auxiliary sets, the proposed algorithm is able to extract knowledge from tissue images showing a ``similar" s
    
[^54]: FAIR4Cov：用于 COVID-19 检测的融合音频实例和表示

    FAIR4Cov: Fused Audio Instance and Representation for COVID-19 Detection. (arXiv:2204.10581v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2204.10581](http://arxiv.org/abs/2204.10581)

    FAIR4Cov是一种针对COVID-19检测的方法，它提出了一种融合身体声音的波形和谱图表示的关节特征向量，可以有效地检测COVID-19患者，胜过了其他方法。

    

    基于身体声音的分类技术长期以来一直被研究用于支持诊断决策，特别是在肺部疾病方面。针对 COVID-19 疫情的紧迫性，越来越多的模型被开发来基于声学输入识别 COVID-19 患者。大多数模型侧重于咳嗽，因为干咳是 COVID-19 最为人所知的症状。然而，呼吸和言语等其他身体声音也被发现与 COVID-19 相关。在这项工作中，我们提出了 FAIR4Cov，它不依赖于特定的身体声音，而是提出了一种融合身体声音的波形和谱图表示的关节特征向量。FAIR4Cov 的核心组件是一个自注意融合单元，它的训练目的是建立多个身体声音和音频表示的关系并将其集成到一个紧凑的特征向量中。我们在两个公共数据集上设置了实验，并在不同场景下评估了我们的提议方法，包括跨数据集评估和早期检测设置。实验结果表明，FAIR4Cov 胜过了现有方法，并展示了利用各种身体声音检测 COVID-19 患者的能力。

    Audio-based classification techniques on body sounds have long been studied to support diagnostic decisions, particularly in pulmonary diseases. In response to the urgency of the COVID-19 pandemic, a growing number of models are developed to identify COVID-19 patients based on acoustic input. Most models focus on cough because the dry cough is the best-known symptom of COVID-19. However, other body sounds, such as breath and speech, have also been revealed to correlate with COVID-19 as well. In this work, rather than relying on a specific body sound, we propose Fused Audio Instance and Representation for COVID-19 Detection (FAIR4Cov). It relies on constructing a joint feature vector obtained from a plurality of body sounds in waveform and spectrogram representation. The core component of FAIR4Cov is a self-attention fusion unit that is trained to establish the relation of multiple body sounds and audio representations and integrate it into a compact feature vector. We set up our experi
    
[^55]: 具有不完整数据的网络时间序列预测

    Networked Time Series Prediction with Incomplete Data. (arXiv:2110.02271v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2110.02271](http://arxiv.org/abs/2110.02271)

    本文研究了具有不完整数据的网络时间序列（NETS）预测问题。提出了NETS-ImpGAN深度学习框架，可以训练不完整数据，并引入了图时序注意力网络来捕捉时间序列之间的相关性和时间相关性。

    

    网络时间序列（NETS）是给定图上的一组时间序列，每个节点对应一个时间序列。它在智能交通、环境监测和移动网络管理等领域都有广泛应用。在这些应用中，一个重要的任务是基于历史值和底层图来预测NETS的未来值。大多数现有的方法需要完整的数据进行训练。然而，在现实世界的情况下，由于传感器故障、不完全感知覆盖等原因，数据缺失是常见的。本文研究了具有不完整数据的NETS预测问题。我们提出了NETS-ImpGAN，一种可以在历史和未来的缺失值上训练的新型深度学习框架。此外，我们提出了新颖的图时序注意力网络，通过引入注意力机制来捕捉时间序列之间的相关性和时间相关性。我们在三个真实数据集上进行了大量实验。

    A networked time series (NETS) is a family of time series on a given graph, one for each node. It has found a wide range of applications from intelligent transportation, environment monitoring to mobile network management. An important task in such applications is to predict the future values of a NETS based on its historical values and the underlying graph. Most existing methods require complete data for training. However, in real-world scenarios, it is not uncommon to have missing data due to sensor malfunction, incomplete sensing coverage, etc. In this paper, we study the problem of NETS prediction with incomplete data. We propose NETS-ImpGAN, a novel deep learning framework that can be trained on incomplete data with missing values in both history and future. Furthermore, we propose novel Graph Temporal Attention Networks by incorporating the attention mechanism to capture both inter-time series correlations and temporal correlations. We conduct extensive experiments on three real-
    
[^56]: 论DP-SGD的Moment Accountant方法的紧密性

    On the Tightness of the Moment Accountant for DP-SGD. (arXiv:2102.09030v8 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2102.09030](http://arxiv.org/abs/2102.09030)

    通过改进Moment Accountant方法，DP-SGD具有可关闭形式的$(\epsilon，\delta)$-DP保证，并且其保证接近是紧密的，具有最小的计算成本。

    

    为了提供差分隐私，在差分隐私SGD（DP-SGD）中，在执行剪切操作后，向本地SGD更新添加标准差为$ \sigma $的高斯噪声。通过非平凡地改进Moment Accountant方法，我们证明了一个封闭形式的$(\epsilon，\delta)$-DP保证：如果$ \sigma=\sqrt{ 2(\epsilon+\ln(1/\delta))/\epsilon} $，则DP-SGD是$ (\epsilon \leq 1/2，\delta = 1 / N) $-DP，其中$T$至少为$ \approx 2k^2/\epsilon$， $(2/e)^2k^2-1/2\geq \ln(N)$，其中$T$是回合的总数，$ K = kN $是梯度计算的总数，其中$ k $用数据集的大小$N$的时代数量来衡量。我们证明我们的表达式接近紧，在$T$小于约为$ 8 $倍于下界$ \approx 2k^2/\epsilon$的常数因子时，$(\epsilon，\delta)$-DP保证将被违反。选择最小可能值的$T \approx 2k^2/\epsilon$不仅会导致接近密集的DP保证，而且还会最小化计算成本。

    In order to provide differential privacy, Gaussian noise with standard deviation $\sigma$ is added to local SGD updates after performing a clipping operation in Differential Private SGD (DP-SGD). By non-trivially improving the moment account method we prove a closed form $(\epsilon,\delta)$-DP guarantee: DP-SGD is $(\epsilon\leq 1/2,\delta=1/N)$-DP if $\sigma=\sqrt{2(\epsilon +\ln(1/\delta))/\epsilon}$ with $T$ at least $\approx 2k^2/\epsilon$ and $(2/e)^2k^2-1/2\geq \ln(N)$, where $T$ is the total number of rounds, and $K=kN$ is the total number of gradient computations where $k$ measures $K$ in number of epochs of size $N$ of the local data set. We prove that our expression is close to tight in that if $T$ is more than a constant factor $\approx 8$ smaller than the lower bound $\approx 2k^2/\epsilon$, then the $(\epsilon,\delta)$-DP guarantee is violated. Choosing the smallest possible value $T\approx 2k^2/\epsilon$ not only leads to a close to tight DP guarantee, but also minimizes 
    

