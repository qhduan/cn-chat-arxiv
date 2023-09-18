# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sparse Autoencoders Find Highly Interpretable Features in Language Models.](http://arxiv.org/abs/2309.08600) | 本研究通过稀疏自编码器在语言模型中发现了一组高度可解释和单一义的特征，从而解决了神经网络内部多义性的问题。 |
| [^2] | [Attention-Only Transformers and Implementing MLPs with Attention Heads.](http://arxiv.org/abs/2309.08593) | 该论文证明了通过使用带有内部维度为1的掩模注意力头实现MLP神经元，可以将MLP和注意力Transformer转换为仅注意力的Transformer。同时，该论文还证明了注意力头可以分别执行MLP的组成部分，并且可以在其权重矩阵中编码任意的掩码模式。 |
| [^3] | [Chain-of-Thought Reasoning is a Policy Improvement Operator.](http://arxiv.org/abs/2309.08589) | 大型语言模型SECToR通过链式思考推理成功地自学新技能， |
| [^4] | [Compositional Foundation Models for Hierarchical Planning.](http://arxiv.org/abs/2309.08587) | 本研究提出了一种基于组合式基础模型的层次规划方法，通过利用语言、视觉和动作数据的多个专家模型，解决了长期目标任务。通过符号计划、视频扩散和逆动力学模型的结合，实现了在新环境中做出有效决策的能力。 |
| [^5] | [Replacing softmax with ReLU in Vision Transformers.](http://arxiv.org/abs/2309.08586) | 在视觉变换器中，用ReLU替换softmax的注意力机制可以在计算性能上接近或匹配softmax注意力，并且通过序列长度进行除法可以缓解精度下降的问题。 |
| [^6] | [A Bayesian Approach to Robust Inverse Reinforcement Learning.](http://arxiv.org/abs/2309.08571) | 这篇论文提出了一种贝叶斯方法，用于稳健的离线模型导向的逆强化学习。通过同时估计专家的奖励函数和主观模型的环境动态，利用先验分布参数化专家对环境模型的准确性，提出了高效的算法。实验证明，当先验地认为专家对环境具有高度准确的模型时，估计的策略表现出稳健性能，并且优于最先进的离线IRL算法。 |
| [^7] | [Neural Network Driven, Interactive Design for Nonlinear Optical Molecules Based on Group Contribution Method.](http://arxiv.org/abs/2309.08570) | 本研究提出了一种基于神经网络的交互式设计方法，结合群组贡献方法和机器学习技术，能够准确快速地设计非线性光学分子。该方法不仅能够准确预测不同分子的光学性质，还可以实现有效的结构搜索。 |
| [^8] | [Local Differential Privacy in Graph Neural Networks: a Reconstruction Approach.](http://arxiv.org/abs/2309.08569) | 本研究提出了一种学习框架，旨在为用户提供节点隐私保护，并通过在节点级别对特征和标签数据进行随机化扰动来实现。通过频率估计和重构方法，实现了对扰动数据中特征和标签的恢复。 |
| [^9] | [Open-vocabulary Keyword-spotting with Adaptive Instance Normalization.](http://arxiv.org/abs/2309.08561) | 本论文提出了一种名为AdaKWS的新方法，用于解决自动语音识别中的开放词汇关键词检测任务。通过训练文本编码器输出关键词条件的归一化参数，并将其应用于处理听觉输入，我们在多语言基准测试上取得了显著的性能改进。 |
| [^10] | [Deep Reinforcement Learning for Efficient and Fair Allocation of Health Care Resources.](http://arxiv.org/abs/2309.08560) | 本研究使用强化学习方法，通过整合个体患者的疾病进展和患者间的相互作用效应，来优化医疗资源的分配策略，旨在提高分配的公平性和整体患者结果。 |
| [^11] | [HINT: Healthy Influential-Noise based Training to Defend against Data Poisoning Attacks.](http://arxiv.org/abs/2309.08549) | 本论文提出了一种名为健康影响力噪声训练的高效稳健训练方法，该方法使用影响函数制造了有助于加强分类模型对抗数据污染攻击的健康噪声，并且在仅修改训练数据的子集时也能有效运行。 |
| [^12] | [Towards Robust Continual Learning with Bayesian Adaptive Moment Regularization.](http://arxiv.org/abs/2309.08546) | 基于贝叶斯自适应时刻正则化的鲁棒性持续学习方法能够在机器人应用中有效地解决灾难性遗忘问题，并具有轻量级和任务实验室等优势。 |
| [^13] | [Efficient and robust Sensor Placement in Complex Environments.](http://arxiv.org/abs/2309.08545) | 本文解决了在复杂环境中高效稳健的传感器部署问题，提出了一种贪婪算法来设计最小传感器集合以实现多覆盖约束，并探索了深度学习技术加速算法的评估。 |
| [^14] | [Towards Last-layer Retraining for Group Robustness with Fewer Annotations.](http://arxiv.org/abs/2309.08534) | 本研究发现，在没有群体标注和只有少量类别标注的情况下，最后一层再训练仍然可以有效提高最差群体准确性，从而优于经验风险最小化在整个数据集上的表现，可以实现群体鲁棒性的提升。 |
| [^15] | [Scaling Laws for Sparsely-Connected Foundation Models.](http://arxiv.org/abs/2309.08520) | 本研究探讨了参数稀疏性对基于大规模数据集训练的Transformer模型在视觉和语言领域中的尺度行为影响，并通过实验证明了权重稀疏性、非零参数数量和训练数据量之间的尺度定律。研究结果可以帮助确定对于给定的有效模型大小和训练预算，所需的最佳稀疏水平。同时，研究还拓展了对不同稀疏结构和策略的探究，揭示了权重稀疏性的能力和限制。 |
| [^16] | [Generalised Probabilistic Diffusion Scale-Spaces.](http://arxiv.org/abs/2309.08511) | 本研究提出了概率扩散模型的广义尺度空间理论，探索了它与经典图像滤波的关系，以及与扩散和渗透滤波的概念和经验联系。 |
| [^17] | [Deep-learning-powered data analysis in plankton ecology.](http://arxiv.org/abs/2309.08500) | 深度学习在浮游生态学数据分析中的应用提供了客观的方案，能加快分析速度、减少实验偏差，并促进浮游生态相关研究的发展。 |
| [^18] | [P-ROCKET: Pruning Random Convolution Kernels for Time Series Classification.](http://arxiv.org/abs/2309.08499) | 本研究提出了一种名为P-ROCKET的方法，通过在特征选择的角度删除卷积核，从而实现对时间序列分类中的随机卷积核进行剪枝。 |
| [^19] | [Towards Word-Level End-to-End Neural Speaker Diarization with Auxiliary Network.](http://arxiv.org/abs/2309.08489) | 本文提出了一种名为WEEND的词级端到端神经网络方法，通过使用辅助网络，实现了同时进行自动语音识别和发言人分离，并在2个发言人的短片场景中取得了优于基线系统的性能。 |
| [^20] | [Deep Multi-Agent Reinforcement Learning for Decentralized Active Hypothesis Testing.](http://arxiv.org/abs/2309.08477) | 这个论文提出了一个分布式主动假设测试（AHT）问题的解决方法，通过多智能体强化学习，设计一个策略来在有限通信通道上合作完成任务，将贝叶斯风险最小化。 |
| [^21] | [On the limitations of data-driven weather forecasting models.](http://arxiv.org/abs/2309.08473) | 数据驱动的机器学习天气预报模型不具备传统基于物理的模型的准确性和物理一致性，它们在预测技能上的优势很大程度上可以归因于这些特殊性。 |
| [^22] | [Explaining Search Result Stances to Opinionated People.](http://arxiv.org/abs/2309.08460) | 这项研究探讨了向有观点的人解释搜索结果立场的效果，发现立场标签和解释可以帮助用户消费更多不同的搜索结果，但没有发现系统性观点改变的证据。 |
| [^23] | [Mixture Encoder Supporting Continuous Speech Separation for Meeting Recognition.](http://arxiv.org/abs/2309.08454) | 本研究将混合编码器方法从两个说话人情况扩展到了更自然的会议环境，包括任意数量的说话人和动态重叠。实验证明，该方法在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。 |
| [^24] | [Toward responsible face datasets: modeling the distribution of a disentangled latent space for sampling face images from demographic groups.](http://arxiv.org/abs/2309.08442) | 本文提出了一种方法，通过建模和采样分解潜空间的方法来生成任意组合的人口群体，以解决现代人脸识别系统中数据集偏见导致的不公平关注问题。 |
| [^25] | [MIML: Multiplex Image Machine Learning for High Precision Cell Classification via Mechanical Traits within Microfluidic Systems.](http://arxiv.org/abs/2309.08421) | 本研究开发了一种新颖的机器学习框架MIML，该框架通过将无标记细胞图像与生物力学属性数据相结合，实现了高精度细胞分类。该方法利用了形态信息，将细胞属性理解得更全面，相较于仅考虑单一数据类型的模型，实现了98.3％的分类精度。该方法已在白细胞和肿瘤细胞分类中得到证明，并具有更广泛的应用潜力。 |
| [^26] | [FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning.](http://arxiv.org/abs/2309.08420) | 提出了一种名为FedDCSR的联邦跨领域顺序推荐框架，通过解缠表示学习来处理不同领域之间的序列特征异质性，并保护数据隐私。 |
| [^27] | [A new method of modeling the multi-stage decision-making process of CRT using machine learning with uncertainty quantification.](http://arxiv.org/abs/2309.08415) | 本研究提出了一种使用机器学习和不确定性量化建模的多阶段决策过程方法，用于预测心力衰竭患者对心脏再同步治疗的反应。该模型能够推荐收集额外的SPECT MPI变量，以提高预测准确性。 |
| [^28] | [Make Deep Networks Shallow Again.](http://arxiv.org/abs/2309.08414) | 通过残差连接的概念将顺序深层架构替换为并行浅层架构，大大减轻了梯度消失问题，并提出了通过截断高阶项的方式来得到宽广层的结构。 |
| [^29] | [Constraint-Free Structure Learning with Smooth Acyclic Orientations.](http://arxiv.org/abs/2309.08406) | 本文提出了一种无约束的连续优化方案COSMO，用于非环结构学习。通过定义一个可微近似的方向矩阵，并使用单一优先向量进行参数化，我们可以得到一个平滑的方向矩阵和相应的非环邻接矩阵，而无需在任何步骤中评估非环性。尽管没有显式约束，但我们证明COSMO始终收敛到一个非环解。这种方法不仅渐近快速，而且比其他有约束方法具有更小的误差。 |
| [^30] | [Optimizing Modular Robot Composition: A Lexicographic Genetic Algorithm Approach.](http://arxiv.org/abs/2309.08399) | 本论文提出了一种将遗传算法与词典式评估相结合的方法来优化模块化机器人的组合，以克服以往方法中存在的设计空间不足和适应复杂任务的问题，并证明了这种方法在比以往范围更大的搜索空间中表现出更好的性能。 |
| [^31] | [Exploring Meta Information for Audio-based Zero-shot Bird Classification.](http://arxiv.org/abs/2309.08398) | 该研究探索了如何利用元信息来改善基于音频的零样本鸟类分类，并通过连接不同的元数据和音频特征获得最佳结果。 |
| [^32] | [Learning by Self-Explaining.](http://arxiv.org/abs/2309.08395) | 学习通过自我解释（LSX）是一种新的学习范式，通过给予解释和批评者的反馈来改进学习者的性能。这种方法适用于图像分类等基本任务，并有潜力在人工智能研究中发挥作用。 |
| [^33] | [Efficient Graphics Representation with Differentiable Indirection.](http://arxiv.org/abs/2309.08387) | 本论文介绍了一种新的学习原语，使用可微分的多尺度查找表作为图形管线中传统计算和数据操作的有效替代方法。它在多个图形任务中展现了灵活性和高效性。 |
| [^34] | [A Unified View Between Tensor Hypergraph Neural Networks And Signal Denoising.](http://arxiv.org/abs/2309.08385) | 这篇论文提出了一种统一视角，将张量超图神经网络（HyperGNNs）和超图信号去噪（HyperGSD）联系起来，并通过设计了张量超图迭代网络（T-HGIN）来应用于信号去噪问题。实验结果显示了该方法的潜在应用价值。 |
| [^35] | [Adaptive Priority Reweighing for Generalizing Fairness Improvement.](http://arxiv.org/abs/2309.08375) | 本文提出了一种新颖的自适应重新加权方法，通过优先考虑靠近决策边界的样本并分配较高的权重，提高了公平分类器的泛化能力。 |
| [^36] | [Understanding the limitations of self-supervised learning for tabular anomaly detection.](http://arxiv.org/abs/2309.08374) | 本研究探讨了自监督学习在表格异常检测中的限制。通过多个实验发现，自监督学习得到的表征并不能提高表格异常检测的性能，这是由于神经网络引入了无关的特征。然而，使用神经网络表示的子空间可以恢复性能。 |
| [^37] | [Continual Learning with Deep Streaming Regularized Discriminant Analysis.](http://arxiv.org/abs/2309.08353) | 本文提出了深度流正则化判别分析下的持续学习方法，能够有效解决使用非同分布数据增量更新模型导致的灾难性遗忘问题。在ImageNet数据集上实验证明，该方法优于批量学习和现有的流式学习算法。 |
| [^38] | [Convergence of ADAM with Constant Step Size in Non-Convex Settings: A Simple Proof.](http://arxiv.org/abs/2309.08339) | 本文分析了ADAM在非凸设置中具有恒定步长的收敛性，给出了步长达到几乎肯定渐近收敛的充分条件，并提供了确定性ADAM在处理平滑非凸函数时达到近似临界性所需的运行时间界限。 |
| [^39] | [Let's Predict Who Will Move to a New Job.](http://arxiv.org/abs/2309.08333) | 本文讨论了如何使用机器学习来预测谁会换工作，包括数据预处理和使用多种ML算法。为了提高性能，使用了合成少数过采样技术。评估模型时使用了精度、召回率、F1-Score和准确率等指标。 |
| [^40] | [Estimation of Counterfactual Interventions under Uncertainties.](http://arxiv.org/abs/2309.08332) | 本论文提出了一种通过采用层次贝叶斯方法来解决连续情况下反事实干预估计中的不确定性的方法。通过推导贝叶斯变形高斯过程的反事实分布，实现了对非高斯分布和非可加情况的建模。 |
| [^41] | [Heteroskedastic conformal regression.](http://arxiv.org/abs/2309.08313) | 本文研究了使用标准化和Mondrian符合规范的方法如何构建自适应的预测区间，以解决回归问题中的异方差噪声。 |
| [^42] | [A Real-Time Active Speaker Detection System Integrating an Audio-Visual Signal with a Spatial Querying Mechanism.](http://arxiv.org/abs/2309.08295) | 本文介绍了一个实时活动说话人检测系统，通过将音频-视觉信号与空间查询机制整合，利用低功耗边缘计算实现。该系统具有优雅的退化性能，能够在计算预算耗尽的情况下仍然有效运行，并在真实会议数据集上表现出良好的性能。 |
| [^43] | [Large Intestine 3D Shape Refinement Using Point Diffusion Models for Digital Phantom Generation.](http://arxiv.org/abs/2309.08289) | 本研究利用几何深度学习和去噪扩散概率模型优化大肠的分割结果，并结合先进的表面重构模型，实现对大肠3D形状的精化恢复。 |
| [^44] | [Sampling-Free Probabilistic Deep State-Space Models.](http://arxiv.org/abs/2309.08256) | 本文提出了一种无需采样的概率深度状态空间模型，通过使用第一个确定性推断算法，实现了高效的训练和测试近似。 |
| [^45] | [Cross-lingual Knowledge Distillation via Flow-based Voice Conversion for Robust Polyglot Text-To-Speech.](http://arxiv.org/abs/2309.08255) | 本文提出了一个跨语言语音合成的框架，使用语音转换和文本到语音模型，优于基于多语种模型的最先进方法，特别适用于资源匮乏的情况。 |
| [^46] | [Quantitative and Qualitative Evaluation of Reinforcement Learning Policies for Autonomous Vehicles.](http://arxiv.org/abs/2309.08254) | 本文使用强化学习算法（PPO）针对自主驾驶车辆的选择进行了优化，通过最小化时间和污染来缓解交通阻塞问题，经实证分析和定性评估证明了方法的有效性和实用性。 |
| [^47] | [Deep Nonnegative Matrix Factorization with Beta Divergences.](http://arxiv.org/abs/2309.08249) | 本文提出了一种使用Beta散度的深度非负矩阵分解方法，应用于面部特征提取、文档主题识别和高光谱图像材料识别。 |
| [^48] | [A Geometric Perspective on Autoencoders.](http://arxiv.org/abs/2309.08247) | 本文从几何角度研究了自编码器框架，并提出了解决多解和畸变表示问题的几何方法。 |
| [^49] | [Topological Node2vec: Enhanced Graph Embedding via Persistent Homology.](http://arxiv.org/abs/2309.08241) | 通过引入拓扑损失项和适应持续同调度量的熵正则化，我们改进了Node2vec方法，使其能够更好地还原输入图的拓扑结构。 |
| [^50] | [Ensuring Toplogical Data-Structure Preservation under Autoencoder Compression due to Latent Space Regularization in Gauss--Legendre nodes.](http://arxiv.org/abs/2309.08228) | 通过在高斯-勒让德节点上进行潜在空间正则化，我们的研究提出了一种新的无监督自编码器，能够确保在压缩过程中保持拓扑数据结构的完整性。 |
| [^51] | [VERSE: Virtual-Gradient Aware Streaming Lifelong Learning with Anytime Inference.](http://arxiv.org/abs/2309.08227) | 这项研究提出了一种具有实时推理能力的流式终身学习方法，采用虚拟梯度进行连续表示学习，借助语义记忆来抑制灾难性遗忘，并在多样化的数据上进行了广泛实验。 |
| [^52] | [Unified Risk Analysis for Weakly Supervised Learning.](http://arxiv.org/abs/2309.08216) | 本文提出了一个框架，为弱监督学习提供了全面的理解和统一的方法论。该框架的表达部分提供了对弱监督形成的统一解释，并包含15种现有的弱监督设置；引导生成的减少图在弱监督学习中提供了全面的连接。该框架的分析部分提供了一种系统的进行风险重写的方法，从而提供了一种新的去污分布策略。 |
| [^53] | [HM-Conformer: A Conformer-based audio deepfake detection system with hierarchical pooling and multi-level classification token aggregation methods.](http://arxiv.org/abs/2309.08208) | HM-Conformer是一种音频深度伪造检测系统，利用分层汇聚和多级分类令牌聚合方法，能够有效地捕捉并检测音频深度伪造的欺骗证据。 |
| [^54] | [Gaussian Processes with Linear Multiple Kernel: Spectrum Design and Distributed Learning for Multi-Dimensional Data.](http://arxiv.org/abs/2309.08201) | 本文研究了高斯过程与线性多核在多维数据上的应用，提出了一种新的格点谱混合核公式，减少了超参数数量，同时保留了优化结构和逼近能力。通过引入分布式算法，使大规模超参数优化变得可行。 |
| [^55] | [An Explainable Deep-learning Model of Proton Auroras on Mars.](http://arxiv.org/abs/2309.08195) | 这项研究开发了一个纯数据驱动模型，使用火星大气和挥发物演化 (MAVEN) 的观测资料，来解释火星质子极光。通过训练人工神经网络，可以准确重现每个Ly alpha辐射的强度，并对观测结果进行忠实重构。 |
| [^56] | [A Precision-Scalable RISC-V DNN Processor with On-Device Learning Capability at the Extreme Edge.](http://arxiv.org/abs/2309.08186) | 这篇论文提出了一种具有精度可扩展性的RISC-V DNN处理器，该处理器可以支持多种精度级别的定点DNN推断，并通过改进的FP16操作增强了在设备学习能力。 |
| [^57] | [Unveiling Invariances via Neural Network Pruning.](http://arxiv.org/abs/2309.08171) | 该论文提出了一种通过神经网络剪枝来学习捕捉数据相关的不变性的新型网络架构的框架。实验证明，这种学习的网络架构在视觉和表格数据集上都比密集神经网络表现出色，不仅效率高，而且效果好。 |
| [^58] | [To Predict or to Reject: Causal Effect Estimation with Uncertainty on Networked Data.](http://arxiv.org/abs/2309.08165) | 本文提出了一种基于不确定性的图深度核学习框架来处理网络数据上因果效应估计中的正性假设违反问题，并在实验证明了该方法的优越性。 |
| [^59] | [AdSEE: Investigating the Impact of Image Style Editing on Advertisement Attractiveness.](http://arxiv.org/abs/2309.08159) | 本文研究了图像样式编辑对广告吸引力的影响。通过引入基于StyleGAN的面部语义编辑和反转，并结合传统的视觉和文本特征，我们提出了AdSEE方法，可用于预测在线广告的点击率。通过对QQ-AD数据集的评估，验证了AdSEE的有效性。 |
| [^60] | [A Testbed for Automating and Analysing Mobile Devices and their Applications.](http://arxiv.org/abs/2309.08158) | 这个研究介绍了一个测试平台，可以自动化生成和标记逼真的移动设备应用程序流量，为改善网络态势感知提供了机器学习技术的工具。 |
| [^61] | [Two-Step Knowledge Distillation for Tiny Speech Enhancement.](http://arxiv.org/abs/2309.08144) | 本文提出了一种新颖的两步法知识蒸馏方法用于微弱语音增强模型。方法首先使用知识蒸馏目标预训练学生模型，然后切换到完全监督训练。同时，引入细粒度相似性保持的知识蒸馏损失，将学生模型的激活内部格拉姆矩阵与教师模型匹配。实验证明，该方法在高压缩和低信噪比条件下表现出显著的性能提升。 |
| [^62] | [Audio Difference Learning for Audio Captioning.](http://arxiv.org/abs/2309.08141) | 本研究引入了音频差异学习方法，通过创建特征表示空间来改进音频字幕生成。该方法使用参考音频和输入音频，生成描述它们差异的字幕，同时提出了一种独特的混合技术来消除差异和原始输入之间的需求。 |
| [^63] | [PromptTTS++: Controlling Speaker Identity in Prompt-Based Text-to-Speech Using Natural Language Descriptions.](http://arxiv.org/abs/2309.08140) | PromptTTS++是一种基于提示的文本转语音系统，可以使用自然语言描述控制说话者身份。与现有研究不同，该方法利用说话者提示来学习自然语言描述与声学特征的映射。 |
| [^64] | [Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates.](http://arxiv.org/abs/2309.08125) | Oobleck采用流水线模板和已复制模型状态来实现对大型模型的弹性分布式训练，并通过有效利用资源和快速恢复来提供高吞吐量。在评估中，Oobleck在吞吐量上胜过了Bamboo和Varuna等最先进的容错解决方案。 |
| [^65] | [Fast and Accurate Deep Loop Closing and Relocalization for Reliable LiDAR SLAM.](http://arxiv.org/abs/2309.08086) | 本文提出了一个名为LCR-Net的多头网络，利用新颖的特征提取和姿态感知机制来快速准确地处理循环关闭和位置再定位任务。实验结果表明，LCR-Net在候选检索、闭环点云配准和多数据集连续再定位等任务中表现优异，超过了当前最先进的方法，并具有出色的泛化能力。 |
| [^66] | [Supervised Stochastic Neighbor Embedding Using Contrastive Learning.](http://arxiv.org/abs/2309.08077) | 该论文将自监督对比学习方法扩展到全监督设置中，允许有效利用标签信息，并在保留数据集邻域信息的同时，将同一类别的样本聚集在一起，将不同类别的样本聚集分开。 |
| [^67] | [Morphologically-Aware Consensus Computation via Heuristics-based IterATive Optimization (MACCHIatO).](http://arxiv.org/abs/2309.08066) | 本文提出了一种基于启发式迭代优化的方法，利用精心选择的距离的Fréchet均值构建二进制或概率一致性分割。与STAPLE方法相比，该方法不受图像背景大小和先验选择的影响。 |
| [^68] | [Traveling Waves Encode the Recent Past and Enhance Sequence Learning.](http://arxiv.org/abs/2309.08045) | 本论文介绍了Wave-RNN (wRNN)模型，展示了旅行波机制如何有效地编码最近的过去，并在合成记忆任务中比波动模型表现更好。 |
| [^69] | [How many Neurons do we need? A refined Analysis for Shallow Networks trained with Gradient Descent.](http://arxiv.org/abs/2309.08044) | 本论文在神经切向核（NTK）范式下，通过分析用梯度下降训练的两层神经网络的泛化性质，改进了现有结果，并得出了快速收敛的速度。此外，我们证明了训练过程中权重保持在初始位置附近，半径与回归函数的平滑度和NTK的积分算子的特征值衰减程度有关。 |
| [^70] | [On Prediction Feature Assignment in the Heckman Selection Model.](http://arxiv.org/abs/2309.08043) | 本文研究在非随机缺失样本选择偏差下的预测模型性能降低问题，提出了Heckman-FA框架来获取恰当的预测特征。 |
| [^71] | [USM-SCD: Multilingual Speaker Change Detection Based on Large Pretrained Foundation Models.](http://arxiv.org/abs/2309.08023) | USM-SCD是一种基于大型预训练基础模型的多语种演讲者转换检测模型，通过微调模型参数，可以同时检测演讲者转换并为96种语言执行自动语音识别。在实验中表现出了优异的性能。 |
| [^72] | [CRYPTO-MINE: Cryptanalysis via Mutual Information Neural Estimation.](http://arxiv.org/abs/2309.08019) | CRYPTO-MINE是一种通过神经网络估计互信息的新方法，应用于选择明文攻击中明文和密文之间的互信息估计。该方法可用于分析密码系统的计算安全性和信息泄露与输入分布之间的关系。 |
| [^73] | [An Automated Machine Learning Approach for Detecting Anomalous Peak Patterns in Time Series Data from a Research Watershed in the Northeastern United States Critical Zone.](http://arxiv.org/abs/2309.07992) | 本文提出了一个自动化机器学习框架，用于检测美国东北地区临界地带研究流域传感器生成的时间序列数据中的异常峰值模式。通过合成生成带有标记的数据集和自动超参数优化机制，该框架克服了标记数据和选择合适的深度学习模型的挑战。 |
| [^74] | [Folding Attention: Memory and Power Optimization for On-Device Transformer-based Streaming Speech Recognition.](http://arxiv.org/abs/2309.07988) | 本论文提出了一种名为折叠注意力的技术，在基于Transformer的流式语音识别模型中，通过减少线性投影层的数量，显著减小了模型大小，提高了内存和功耗效率，实验证明可以将模型大小减小24%、功耗减小23%。 |
| [^75] | [Viewpoint Textual Inversion: Unleashing Novel View Synthesis with Pretrained 2D Diffusion Models.](http://arxiv.org/abs/2309.07986) | 本研究展示了通过预训练的2D图像扩散模型，可以从仅有2D监督的情况下提取出3D结构信息，并利用该信息进行3D视觉任务。通过观点神经文本倒置（ViewNeTI）方法，我们可以控制生成图像中对象的3D视点，有效解决新颖视图合成问题，并在单视图情况下具有良好的语义细节和逼真度。 |
| [^76] | [SLMIA-SR: Speaker-Level Membership Inference Attacks against Speaker Recognition Systems.](http://arxiv.org/abs/2309.07983) | 这项研究提出了SLMIA-SR，这是针对说话人识别系统的第一个针对说话人级别成员推断攻击。与传统的示例级攻击不同，这种攻击方法可以确定一个给定的声音是否与训练数据中的任何声音有关，无论它们是否相同。这对实践非常有用，因为训练和推断声音通常是不同的，而且考虑到说话人识别的开放性质，也是有意义的。 |
| [^77] | [Uncertainty quantification for learned ISTA.](http://arxiv.org/abs/2309.07982) | 本文提出了一种严谨的方法来获得LISTA估计量的置信区间，为模型-based深度学习解决方案中的不确定性提供了理论支持。 |
| [^78] | [A Data Source for Reasoning Embodied Agents.](http://arxiv.org/abs/2309.07974) | 本研究提出了一个与体验智能体集成的新数据生成器，用于机器推理。该生成器生成的数据包括模板化的文本查询和答案，并与编码为数据库的世界状态相匹配。通过实验发现，当前模型可以回答一些关于世界状态的问题，但在其他问题上存在困难。 |
| [^79] | [Complex-Valued Neural Networks for Data-Driven Signal Processing and Signal Understanding.](http://arxiv.org/abs/2309.07948) | 该论文介绍了一个基于PyTorch构建的软件包，用于实现复数值神经网络操作和架构的轻量级接口。该软件包的目标是为信号处理、感知和通信等领域提供高效的复数值模型支持。 |
| [^80] | [TiBGL: Template-induced Brain Graph Learning for Functional Neuroimaging Analysis.](http://arxiv.org/abs/2309.07947) | TiBGL是一种模板引导的脑图学习框架，用于功能性神经影像分析。它具有判别和可解释能力，旨在通过学习功能连接数据的有用特征来改进神经疾病的诊断效率。 |
| [^81] | [Slow Invariant Manifolds of Singularly Perturbed Systems via Physics-Informed Machine Learning.](http://arxiv.org/abs/2309.07946) | 通过物理知识引导的机器学习方法用于计算奇异摄动系统的慢不变流形，提供了显式形式的函数来构建和数值积分缩减模型，并通过三个基准问题的评估证明了其有效性。 |
| [^82] | [Masked Generative Modeling with Enhanced Sampling Scheme.](http://arxiv.org/abs/2309.07945) | 本文提出了一种增强的采样方案 (ESS)，用于掩码非自回归生成建模。该方案能够确保样本的多样性和保真度，并由三个阶段组成：简单迭代解码、关键反向采样和关键重采样。简单迭代解码用于采样标记集，关键反向采样和关键重采样用于掩盖不真实的标记并重建被掩盖的标记，以提高采样的保真度。 |
| [^83] | [Voxtlm: unified decoder-only models for consolidating speech recognition/synthesis and speech/text continuation tasks.](http://arxiv.org/abs/2309.07937) | Voxtlm是一个统一的只解码模型，能够在语音识别、语音合成、文本生成和语音延续等任务上取得显著的改善。 |
| [^84] | [Landscape-Sketch-Step: An AI/ML-Based Metaheuristic for Surrogate Optimization Problems.](http://arxiv.org/abs/2309.07936) | Landscape-Sketch-Step是一种基于AI/ML的元启发式方法，结合了机器学习、随机优化和强化学习技术，用于解决成本函数评估昂贵、不可访问或禁止的代理优化问题。 |
| [^85] | [Racing Control Variable Genetic Programming for Symbolic Regression.](http://arxiv.org/abs/2309.07934) | 提出了一种称为Racing Control Variable Genetic Programming (Racing-CVGP) 的方法，它通过同时进行多个实验计划来加速符号回归过程，并克服了固定实验计划选择不佳导致发现过程延迟的限制。 |
| [^86] | [Generative AI.](http://arxiv.org/abs/2309.07930) | "生成型人工智能"指的是能够从训练数据中生成新颖有意义内容的计算技术，如文本、图像或音频。本文提供了生成型人工智能在社会技术系统中的概念，并介绍了模型、系统和应用的示例。同时，提出了当前生成型人工智能的限制，并提出了对商业与信息系统工程研究的议程，包括研究机会和挑战。 |
| [^87] | [Prompting Segmentation with Sound is Generalizable Audio-Visual Source Localizer.](http://arxiv.org/abs/2309.07929) | 本研究提出了一种使用声音提示进行分割的泛化音频-视觉源定位器，在零样本和少样本情况下实现音频-视觉定位和分割任务。通过引入编码器提示解码器范式、构建语义感知音频提示和相关适配器来解决数据稀缺性和不同数据分布的困境。 |
| [^88] | [Virchow: A Million-Slide Digital Pathology Foundation Model.](http://arxiv.org/abs/2309.07778) | Virchow是一个数百万参数的深度神经网络基础模型，通过在数百万张全数字病理学切片图像上进行自监督学习训练，有效解决了计算病理学任务中数据不足的问题，并在多个下游任务上超越了最先进的系统。 |
| [^89] | [Structure-Preserving Transformers for Sequences of SPD Matrices.](http://arxiv.org/abs/2309.07579) | 本文介绍了一种保持序列的对称正定矩阵的黎曼几何特性的结构保持变压器机制，并将其应用于自动睡眠分期，取得了高水平的阶段性能。 |
| [^90] | [Frequency Convergence of Complexon Shift Operators.](http://arxiv.org/abs/2309.07169) | 本文研究了拓扑信号处理中复合子的可转移性，通过构造边际复合子和复合移位算子，研究其特征值和特征向量，并证明了复合子收敛时对应的复合移位算子的特征值会收敛到极限复合子的特征值。这些结果拓展了图信号处理框架。 |
| [^91] | [Collectionless Artificial Intelligence.](http://arxiv.org/abs/2309.06938) | 本文提出了无集合原则的学习协议的思路，其中机器在环境交互背景中掌握认知技能，避免了数据集集中化的风险。 |
| [^92] | [Uncertainty-aware Traffic Prediction under Missing Data.](http://arxiv.org/abs/2309.06800) | 本研究提出了一种考虑不确定性的交通预测方法，可以处理缺失数据和测量不确定性，并适用于风险敏感任务和决策导向问题。 |
| [^93] | [Convergence of Gradient-based MAML in LQR.](http://arxiv.org/abs/2309.06588) | 本研究调查了在LQR中应用MAML时的局部收敛特性，并提供了保持动态系统稳定性的局部收敛保证。论文通过简单的数值结果展示了MAML在LQR任务中的收敛性质。 |
| [^94] | [Explainable Graph Neural Network for Alzheimer's Disease And Related Dementias Risk Prediction.](http://arxiv.org/abs/2309.06584) | 这项研究提出了一种可解释的图神经网络方法来预测阿尔茨海默病和相关痴呆症的风险。通过将机器学习与索赔数据相结合，不仅能发现额外的风险因素，还能揭示不同医学代码之间的关联。通过评估关系重要性和其对风险预测的影响，该方法能提供全面的解释。 |
| [^95] | [Practical Homomorphic Aggregation for Byzantine ML.](http://arxiv.org/abs/2309.05395) | 本文介绍了一种适用于拜占庭式机器学习的实用同态聚合方法，以应对拜占庭节点和服务器隐私侵犯的问题。 |
| [^96] | [CenTime: Event-Conditional Modelling of Censoring in Survival Analysis.](http://arxiv.org/abs/2309.03851) | CenTime是一种新的生存分析方法，通过创新的事件条件审查机制直接估计事件发生的时间，在处理未被审查的数据时具有良好的鲁棒性和准确性。 |
| [^97] | [Scalable Learning of Intrusion Responses through Recursive Decomposition.](http://arxiv.org/abs/2309.03292) | 本文提出了一种通过递归分解方法实现可扩展学习入侵响应的技术，该技术通过解决并行子游戏和计算阈值结构的最佳响应策略来提高效率。 |
| [^98] | [PROMISE: Preconditioned Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates.](http://arxiv.org/abs/2309.02014) | 本文介绍了一套基于草图的预条件随机梯度算法套件PROMISE，用于解决大规模凸优化问题，相比传统方法，在默认超参数下在机器学习问题上表现更优。 |
| [^99] | [Representing Edge Flows on Graphs via Sparse Cell Complexes.](http://arxiv.org/abs/2309.01632) | 本文针对图边缘的流量数据，提出了一种通过稀疏细胞复合体来表示边流的方法。我们将图结构转化为一个单纯复合体，利用Hodge-Laplacian的特征向量和关联矩阵进行Hodge分解，得到梯度、旋量和谐波流的表示。同时，我们引入了细胞推断优化问题，通过添加细胞来增强观测到的图，使表示稀疏可解释。实验证明这个问题是NP难的，我们提出了一个高效的近似算法。 |
| [^100] | [Symbolically integrating tensor networks over various random tensors -- the second version of Python RTNI.](http://arxiv.org/abs/2309.01167) | 我们升级了Python RTNI的第二版，可以对不同随机张量进行符号性整合，支持Haar分布的酉矩阵、正交矩阵和正态分布的张量。通过导出TensorNetwork格式的张量网络，可以进行低维计算，并解释了数学原理和张量网络图之间的关系。 |
| [^101] | [Majorana Demonstrator Data Release for AI/ML Applications.](http://arxiv.org/abs/2308.10856) | 此数据发布包含Majorana示范器实验的校准数据子集，旨在支持人工智能和机器学习算法在该数据上的训练和测试。 |
| [^102] | [MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models.](http://arxiv.org/abs/2308.09729) | 本论文通过使用知识图谱来激发大型语言模型，解决了整合新知识、产生幻觉和决策过程不透明等问题，并通过生成思维导图展示了模型的推理路径，实验证明这种方法可以取得显著的实证增益。 |
| [^103] | [Feature Enforcing PINN (FE-PINN): A Framework to Learn the Underlying-Physics Features Before Target Task.](http://arxiv.org/abs/2308.08873) | FE-PINN是一种学习底层物理特征的框架，在主训练之前以低计算成本解决问题的模式。与传统PINN相比，FE-PINN通过执行一系列子任务来解决损失函数不平衡的问题，并具有快速训练和更高的求解速度。 |
| [^104] | [LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs.](http://arxiv.org/abs/2308.08469) | 这项工作提出了LLM4TS方法，利用预训练的LLMs增强时间序列预测能力。通过两阶段微调和参数高效微调，提高了LLMs处理时间序列数据的能力。 |
| [^105] | [DCNFIS: Deep Convolutional Neuro-Fuzzy Inference System.](http://arxiv.org/abs/2308.06378) | 本文介绍了一种新的深度卷积神经模糊推理系统（DCNFIS），它通过将模糊逻辑和深度学习模型相结合，实现了提高透明度而不损失准确性的目标。DCNFIS在准确性上与现有卷积神经网络相当，并且胜过了最先进的深度模糊系统。通过模糊规则提取的解释可以提高模型的可解释性。 |
| [^106] | [Transferable Graph Neural Fingerprint Models for Quick Response to Future Bio-Threats.](http://arxiv.org/abs/2308.01921) | 该论文提出了一种可转移的图神经指纹模型，用于快速应对未来的生物威胁。通过利用包含30万种候选药物和23个冠状病毒蛋白靶的COVID-19药物对接数据集，训练了高通量虚拟COVID-19药物筛选的图神经指纹模型。与传统指纹方法相比，该模型在对接得分上具有较高的预测准确性，并且提出了可转移的图神经指纹方法，能够适用于未知的靶点。 |
| [^107] | [Towards Head Computed Tomography Image Reconstruction Standardization with Deep Learning Assisted Automatic Detection.](http://arxiv.org/abs/2307.16440) | 提出了一种用深度学习辅助自动检测实现头部CT图像三维重建的方法，该方法可以提高重建的准确性和可重复性，减少了手动干预。通过识别和评估眶下线标志点，实现在重建之前自动重新格式化图像。 |
| [^108] | [Discovering interpretable elastoplasticity models via the neural polynomial method enabled symbolic regressions.](http://arxiv.org/abs/2307.13149) | 本文介绍了一种通过神经多项式方法实现可解释的弹性塑性模型的机器学习方法，该方法通过分为两个步骤，先通过监督学习得到一组特征映射，再通过符号回归将其转化为数学公式，从而克服了传统神经网络模型的缺乏可解释性的问题。 |
| [^109] | [Policy Gradient Optimal Correlation Search for Variance Reduction in Monte Carlo simulation and Maximum Optimal Transport.](http://arxiv.org/abs/2307.12703) | 该论文提出了一种新的算法，通过引入相关路径来降低蒙特卡洛模拟中的方差，从而估计随机微分方程解的函数。通过政策梯度和强化学习技术，使用深度神经网络近似最优相关函数并进行校准。这与最大最优传输问题有关。 |
| [^110] | [Rethinking Data Distillation: Do Not Overlook Calibration.](http://arxiv.org/abs/2307.12463) | 本文指出经过蒸馏的数据无法很好地进行校准，因为在这种情况下，网络的 logits 分布更加集中，并且语义明确但与分类任务无关的信息会丢失。为了解决这个问题，我们提出了遮蔽温度缩放 (MTS) 和遮蔽蒸馏训练 (MDT) 方法，以获得更好的校准结果。 |
| [^111] | [Solving multiphysics-based inverse problems with learned surrogates and constraints.](http://arxiv.org/abs/2307.11099) | 本论文将学习代理和学习约束相结合用于解决基于多物理的反问题，通过该方法不仅改善了对流体流动性质的反演精度，而且为反演多模态数据提供了一个有效的解决方案。 |
| [^112] | [On the Constrained Time-Series Generation Problem.](http://arxiv.org/abs/2307.01717) | 这篇论文研究了约束时间序列生成问题。在实际应用中，合成时间序列被广泛用于增强历史时间序列数据集，提高机器学习算法的性能，放大稀有事件的发生，以及创建反事实情景。然而，现有的方法在满足约束方面存在问题，需要重新训练且计算代价高，或者在复杂约束条件下不切实际。 |
| [^113] | [Engression: Extrapolation for Nonlinear Regression?.](http://arxiv.org/abs/2307.00835) | Engression是一种非线性回归方法，通过使用分布回归技术和预加性噪声模型，在训练样本范围边界外也能可靠地进行外推。 |
| [^114] | [Feature Selection: A perspective on inter-attribute cooperation.](http://arxiv.org/abs/2306.16559) | 本文综述了辅助特征间协作的过滤特征选择方法的最新研究进展，并总结了不同方法在文献中的贡献。同时提出了当前存在的问题和挑战，以确定未来有前景的研究和发展方向。 |
| [^115] | [SEEDS: Emulation of Weather Forecast Ensembles with Diffusion Models.](http://arxiv.org/abs/2306.14066) | 本文提出了利用生成人工智能技术在规模上生成集合预测的方法，并产生了与完整的GEFS 31成员集合相似的预测能力，并且很好地模拟了大规模集合的统计数据。 |
| [^116] | [Text-Driven Foley Sound Generation With Latent Diffusion Model.](http://arxiv.org/abs/2306.10359) | 本文提出了一种基于扩散模型的Foley音效生成系统，可进行文本条件的生成。我们通过迁移学习对系统进行微调，并引入可训练的层来改善文本嵌入，同时也改进了生成的波形。 |
| [^117] | [Neural Network Compression using Binarization and Few Full-Precision Weights.](http://arxiv.org/abs/2306.08960) | 本文提出了一种自动修剪二进制化（APB）的新型神经网络压缩技术，通过结合量化和修剪，利用少量全精度权重来增强二进制网络的表示能力，并通过高效的算法在CPU上实现了高速的量化矩阵乘法运算。 |
| [^118] | [Simulation and Prediction of Countercurrent Spontaneous Imbibition at Early and Late Times Using Physics-Informed Neural Networks.](http://arxiv.org/abs/2306.05554) | 本文通过物理信息神经网络模型对多孔材料中的逆流自发渗透过程进行了早期和晚期的模拟和预测，并使用改变变量技术来改进模型性能。 |
| [^119] | [Exploiting Noise as a Resource for Computation and Learning in Spiking Neural Networks.](http://arxiv.org/abs/2305.16044) | 本文提出了噪声脉冲神经元网络（NSNN）和噪声驱动学习规则（NDL），展示了噪声可以作为计算和学习的资源，并为一般脉冲神经元网络提供了一个框架。研究还展示了NSNNs在图像分类和语音识别等实际任务中的适用性，表明它们是未来神经形态计算系统的潜在有力工具。 |
| [^120] | [AUC Optimization from Multiple Unlabeled Datasets.](http://arxiv.org/abs/2305.15776) | 本文提出了一种从多个未标记数据集中构建AUC优化模型的方法，该方法在实践和理论上都有效。 |
| [^121] | [Improving Speech Emotion Recognition Performance using Differentiable Architecture Search.](http://arxiv.org/abs/2305.14402) | 该论文提出使用DARTS优化联合CNN和LSTM的体系结构以提高语音情绪识别性能，并在实验中证明了其优于以往最好的结果。 |
| [^122] | [Predictive change point detection for heterogeneous data.](http://arxiv.org/abs/2305.06630) | 该论文提出了一种基于“预测与比较”机器学习模型的变点监测框架，它能够比现有的在线监测方法更好地控制误报率和失控平均运行长度。该方法使用ARIMA模型和LSTM递归神经网络模型进行预测，具有很强的推广性能。 |
| [^123] | [Wasserstein Dictionaries of Persistence Diagrams.](http://arxiv.org/abs/2304.14852) | 本文提出了一种基于Wasserstein字典的持久图的紧凑编码方法，并在实验中证明了其有效性，可以应用于数据降维和压缩。 |
| [^124] | [CVRecon: Rethinking 3D Geometric Feature Learning For Neural Reconstruction.](http://arxiv.org/abs/2304.14633) | 研究团队提出了一种基于代价体的3D神经重建框架CVRecon，利用丰富的几何嵌入来促进3D几何特征学习。通过引入射线上下文补偿代价体（RCCV），有效提高了视角相关信息的完整性和鲁棒性，并在各种度量方面显着提高了重建质量。 |
| [^125] | [Surrogate Assisted Generation of Human-Robot Interaction Scenarios.](http://arxiv.org/abs/2304.13787) | 本文提出了基于替代模型的人机交互场景生成方法，可以高效合成多样化的挑战性数据集，以便评估和理解人机交互系统的优劣，可以在实际交互中重现这些场景。 |
| [^126] | [Unconstrained Parametrization of Dissipative and Contracting Neural Ordinary Differential Equations.](http://arxiv.org/abs/2304.02976) | 本文介绍了一种连续时间的深度神经网络，通过结合神经常微分方程和循环平衡网络的结构，使得网络具有收缩和耗散性质。此外提出的非约束参数化方法使得该网络学习的参数量得以增加。 |
| [^127] | [BoundaryCAM: A Boundary-based Refinement Framework for Weakly Supervised Semantic Segmentation of Medical Images.](http://arxiv.org/abs/2303.07853) | BoundaryCAM提出了一种基于边界的弱监督的优化框架，能够预测对象位置，实现精细的高精度分割掩模。 |
| [^128] | [Data-Centric AI: Deep Generative Differentiable Feature Selection via Discrete Subsetting as Continuous Embedding Space Optimization.](http://arxiv.org/abs/2302.13221) | 该论文提出一种将离散特征子集作为连续嵌入空间优化的深度生成可微分特征选择方法，解决了在高维小样本数据集中通用、准确和维度无关的特征选择问题。 |
| [^129] | [A Comprehensive Review and a Taxonomy of Edge Machine Learning: Requirements, Paradigms, and Techniques.](http://arxiv.org/abs/2302.08571) | 这篇论文综述了边缘机器学习的需求、范式和技术，并强调了其在保护隐私、实现低延迟的实时性能和资源优化方面的重要性。 |
| [^130] | [Guiding Pretraining in Reinforcement Learning with Large Language Models.](http://arxiv.org/abs/2302.06692) | 这项研究提出了一种使用大型语言模型在强化学习中引导预训练的方法，通过奖励代理根据语言模型建议的目标来塑造探索策略，使代理朝着人类有意义且可能有用的行为方向发展，无需人类的介入。 |
| [^131] | [Five policy uses of algorithmic transparency and explainability.](http://arxiv.org/abs/2302.03080) | 本论文通过案例研究展示了算法透明性和可解释性在政策环境中的五种应用方式：对解释的具体要求；在算法内部治理的非约束性指南中；适用于高度管制环境的法规；旨在提高算法法律责任的实用性的指南；以及对模型和数据透明性的广泛要求。 |
| [^132] | [A Survey on Deep Learning based Time Series Analysis with Frequency Transformation.](http://arxiv.org/abs/2302.02173) | 近期，频率变换（FT）在深度学习时间序列分析中得到广泛应用，显著提高了准确性和效率。本文系统回顾和总结了基于FT的深度学习时间序列模型的研究进展，并探讨了其优势、限制以及主要方法。 |
| [^133] | [3D-Spatiotemporal Forecasting the Expansion of Supernova Shells Using Deep Learning toward High-Resolution Galaxy Simulations.](http://arxiv.org/abs/2302.00026) | 本文开发了一个深度学习模型，3D-MIM，用于预测超新星爆炸后的壳层扩张，通过在平滑粒子流体动力学模拟中检测并预测超新星影响粒子所在的壳层形状，解决了高分辨率星系模拟中超新星积分时间步长问题。 |
| [^134] | [Emerging Synergies in Causality and Deep Generative Models: A Survey.](http://arxiv.org/abs/2301.12351) | 这项综述探讨了因果性和深度生成模型之间的新兴协同作用，阐明了将因果性原则融入DGM中的方法，以及在大规模生成模型中应用因果性的研究前沿。 |
| [^135] | [SPEC5G: A Dataset for 5G Cellular Network Protocol Analysis.](http://arxiv.org/abs/2301.09201) | SPEC5G是首个公共5G数据集，用于5G蜂窝网络协议的安全性分析和文本摘要。 |
| [^136] | [Temporal Saliency Detection Towards Explainable Transformer-based Timeseries Forecasting.](http://arxiv.org/abs/2212.07771) | 这项研究提出了一种名为Temporal Saliency Detection (TSD)的方法，利用基于注意力机制的架构实现了多步时间序列预测，并通过压缩多头注意力进行显著性模式的多分辨率分析。 |
| [^137] | [TIDE: Time Derivative Diffusion for Deep Learning on Graphs.](http://arxiv.org/abs/2212.02483) | 本文提出了一种新方法 TIDE，通过时间导数图扩散克服了图神经网络中消息传递框架的结构限制，实现了高效地中长距离通信，并在图神经网络任务中达到了 state-of-the-art 的性能表现。 |
| [^138] | [On counterfactual inference with unobserved confounding.](http://arxiv.org/abs/2211.08209) | 本研究提出了一种在观测研究中应对未观察到混淆因素进行反事实推断的方法，通过建模条件分布，学习了各单位的反事实分布，并提供了一个均方误差的界限。 |
| [^139] | [BAFFLE: Backdoor Attack in Offline Reinforcement Learning.](http://arxiv.org/abs/2210.04688) | 本文研究离线增强学习中的后门攻击，通过向数据中添加扰动，使得智能体在注入触发器的观测值上采取低奖励动作，从而提出了BAFFLE方法。 |
| [^140] | [Critical Learning Periods for Multisensory Integration in Deep Networks.](http://arxiv.org/abs/2210.04643) | 对于多感官集成，神经网络在早期训练阶段接受适当相关信号至关重要，而干扰学习过程可能会永久损害技能的发展。早期瞬态动力学对最终的系统性能和学习表示具有决定性影响。 |
| [^141] | [DPA-1: Pretraining of Attention-based Deep Potential Model for Molecular Simulation.](http://arxiv.org/abs/2208.08236) | DPA-1是一种具有新颖注意力机制的深度势能模型，能够高效表示原子系统的构象和化学空间，并且在分子模拟中能够通过预训练和微调取得卓越的性能。 |
| [^142] | [Optimal scheduling of entropy regulariser for continuous-time linear-quadratic reinforcement learning.](http://arxiv.org/abs/2208.04466) | 本研究利用熵正则化的松弛随机控制视角设计了连续时间线性二次强化学习算法，并通过探索性控制方法和近端策略更新方法实现了探索和利用的权衡，以解决有限时间线性二次强化学习问题。 |
| [^143] | [Tac2Pose: Tactile Object Pose Estimation from the First Touch.](http://arxiv.org/abs/2204.11701) | Tac2Pose是一种从第一次触觉中估计物体姿态的方法，通过在仿真中学习物体的感知模型，根据触觉观测估计可能的物体姿态，并通过对比学习进行匹配。这种方法只需要一次真实触觉观测即可定位物体。 |
| [^144] | [GP-BART: a novel Bayesian additive regression trees approach using Gaussian processes.](http://arxiv.org/abs/2204.02112) | GP-BART是一种使用高斯过程的新型贝叶斯加法回归树方法，相比标准BART模型，它具有更好的平滑性和明确的协方差结构假设，在多种情境下显示出超越传统建模方法的性能。 |
| [^145] | [Application of Quantum Density Matrix in Classical Question Answering and Classical Image Classification.](http://arxiv.org/abs/2203.11155) | 该论文将量子密度矩阵应用于经典问答和图像分类中，证明了其可以提高任务的效率，尤其在图像分类中取得了优秀的性能表现。 |
| [^146] | [Don't Get Me Wrong: How to Apply Deep Visual Interpretations to Time Series.](http://arxiv.org/abs/2203.07861) | 该论文提出了一个针对时间序列分类和分割任务的框架，通过六个度量来评估基于梯度、传播或干扰的事后可视化解释方法。实验结果表明，这些方法对于时间序列的解释具有较高的可信度和有效性。 |
| [^147] | [HAKE: A Knowledge Engine Foundation for Human Activity Understanding.](http://arxiv.org/abs/2202.06851) | 本论文提出了一个名为HAKE的知识引擎，用于人类活动理解。该引擎通过将像素映射到中间空间，并使用逻辑规则推断语义，展现出了优越的泛化能力和性能。 |
| [^148] | [Physics-enhanced deep surrogates for PDEs.](http://arxiv.org/abs/2111.05841) | 这种物理增强的深度代理方法通过结合低保真度的物理模拟器和神经网络生成器来开发复杂物理系统的快速代理模型，能够在精确性和成本之间取得更好的平衡。 |
| [^149] | [MixStyle Neural Networks for Domain Generalization and Adaptation.](http://arxiv.org/abs/2107.02053) | MixStyle是一个简单的模块，用于提高神经网络对于领域转移的泛化性能。它通过在训练过程中混合两个随机实例的特征统计来合成新领域，从而实现数据增强。MixStyle易于实现，适用于各类学习范式。 |
| [^150] | [Constraint-Based Causal Discovery using Partial Ancestral Graphs in the presence of Cycles.](http://arxiv.org/abs/2005.00610) | 本研究证明了在涉及反馈的系统生成的观察数据中，应用Fast Causal Inference (FCI)算法可以得到正确的结果，该算法可以被用于一致地估计因果关系的存在和缺失、直接因果关系的存在和缺失、混淆因素的缺失以及因果图中特定循环的缺失。 |

# 详细

[^1]: 稀疏自编码器在语言模型中发现高度可解释的特征

    Sparse Autoencoders Find Highly Interpretable Features in Language Models. (arXiv:2309.08600v1 [cs.LG])

    [http://arxiv.org/abs/2309.08600](http://arxiv.org/abs/2309.08600)

    本研究通过稀疏自编码器在语言模型中发现了一组高度可解释和单一义的特征，从而解决了神经网络内部多义性的问题。

    

    神经网络内部理解的一个障碍是多义性，其中神经元在多个语义不同的上下文中激活。多义性使我们无法找到简洁的、人类可理解的解释来解释神经网络内部的工作。多义性的一个猜测原因是叠加效应，即神经网络通过将特征分配给激活空间中的一个过完备方向集合，而不是个别神经元，表示更多的特征。在这里，我们尝试使用稀疏自编码器来确定这些方向，以重构语言模型的内部激活。这些自编码器学习到的一组稀疏激活特征比其他方法鉴定出的方向更可解释和单一义，解释性是通过自动化方法衡量的。删除这些特征可以实现精确的模型编辑，例如通过删除这些特征可以改变模型输出。

    One of the roadblocks to a better understanding of neural networks' internals is \textit{polysemanticity}, where neurons appear to activate in multiple, semantically distinct contexts. Polysemanticity prevents us from identifying concise, human-understandable explanations for what neural networks are doing internally. One hypothesised cause of polysemanticity is \textit{superposition}, where neural networks represent more features than they have neurons by assigning features to an overcomplete set of directions in activation space, rather than to individual neurons. Here, we attempt to identify those directions, using sparse autoencoders to reconstruct the internal activations of a language model. These autoencoders learn sets of sparsely activating features that are more interpretable and monosemantic than directions identified by alternative approaches, where interpretability is measured by automated methods. Ablating these features enables precise model editing, for example, by remo
    
[^2]: 仅使用注意力的Transformer和使用注意力头实现MLPs

    Attention-Only Transformers and Implementing MLPs with Attention Heads. (arXiv:2309.08593v1 [cs.LG])

    [http://arxiv.org/abs/2309.08593](http://arxiv.org/abs/2309.08593)

    该论文证明了通过使用带有内部维度为1的掩模注意力头实现MLP神经元，可以将MLP和注意力Transformer转换为仅注意力的Transformer。同时，该论文还证明了注意力头可以分别执行MLP的组成部分，并且可以在其权重矩阵中编码任意的掩码模式。

    

    Transformer架构被广泛应用于机器学习模型，由注意力头和多层感知器（MLPs）交替组成。我们证明了只要MLP的激活函数来自限制类（包括SiLU和接近的ReLU和GeLU），就可以通过带有内部维度为1的掩模注意力头来实现MLP神经元。这样就可以将MLP和注意力Transformer转换为仅注意力的Transformer，但代价是大大增加了注意力头的数量。我们还证明了注意力头可以分别执行MLP的组成部分（线性变换和激活函数）。最后，我们证明了注意力头可以在其权重矩阵中编码任意的掩码模式，并且这个近似误差可以任意小。

    The transformer architecture is widely used in machine learning models and consists of two alternating sublayers: attention heads and MLPs. We prove that an MLP neuron can be implemented by a masked attention head with internal dimension 1 so long as the MLP's activation function comes from a restricted class including SiLU and close approximations of ReLU and GeLU. This allows one to convert an MLP-and-attention transformer into an attention-only transformer at the cost of greatly increasing the number of attention heads. We also prove that attention heads can perform the components of an MLP (linear transformations and activation functions) separately. Finally, we prove that attention heads can encode arbitrary masking patterns in their weight matrices to within arbitrarily small error.
    
[^3]: 链式思考推理是一种策略改进操作

    Chain-of-Thought Reasoning is a Policy Improvement Operator. (arXiv:2309.08589v1 [cs.LG])

    [http://arxiv.org/abs/2309.08589](http://arxiv.org/abs/2309.08589)

    大型语言模型SECToR通过链式思考推理成功地自学新技能，

    

    大型语言模型以其令人赞叹的新能力令世界为之惊叹。然而，它们目前缺乏自我学习新技能的能力，而是依赖于接受大量由人类生成的数据的训练。我们介绍了SECToR（通过链式思考推理实现自我教育），这是一个概念验证，证明语言模型可以通过链式思考推理成功地自学新技能。受到以前在强化学习（Silver等人，2017）和人类认知（Kahneman，2011）中的相关工作的启发，SECToR首先使用链式思考推理逐渐思考问题。然后，SECToR通过微调模型生成相同的答案，这次不再使用链式思考推理。通过SECToR训练的语言模型自主学会了进行多达29位数字的加法运算，而没有任何超过6位数字的基准真实示例，仅通过初始的监督微调阶段。我们的核心假设是...

    Large language models have astounded the world with fascinating new capabilities. However, they currently lack the ability to teach themselves new skills, relying instead on being trained on large amounts of human-generated data. We introduce SECToR (Self-Education via Chain-of-Thought Reasoning), a proof-of-concept demonstration that language models can successfully teach themselves new skills using chain-of-thought reasoning. Inspired by previous work in both reinforcement learning (Silver et al., 2017) and human cognition (Kahneman, 2011), SECToR first uses chain-of-thought reasoning to slowly think its way through problems. SECToR then fine-tunes the model to generate those same answers, this time without using chain-of-thought reasoning. Language models trained via SECToR autonomously learn to add up to 29-digit numbers without any access to any ground truth examples beyond an initial supervised fine-tuning phase consisting only of numbers with 6 or fewer digits. Our central hypot
    
[^4]: 基于组合式基础模型的层次规划

    Compositional Foundation Models for Hierarchical Planning. (arXiv:2309.08587v1 [cs.LG])

    [http://arxiv.org/abs/2309.08587](http://arxiv.org/abs/2309.08587)

    本研究提出了一种基于组合式基础模型的层次规划方法，通过利用语言、视觉和动作数据的多个专家模型，解决了长期目标任务。通过符号计划、视频扩散和逆动力学模型的结合，实现了在新环境中做出有效决策的能力。

    

    在新环境中做出有效决策需要进行跨空间和时间尺度的层次推理。本文提出了一种基于组合式基础模型的层次规划方法，利用多个专家模型分别对语言、视觉和动作数据进行训练，共同解决长期目标任务。我们利用一个大型语言模型构建在环境中扎根的符号计划，并通过大型视频扩散模型来实现。生成的视频计划通过逆动力学模型与视觉-动作控制相结合。为了在此层次结构中进行有效推理，我们通过迭代改进强制保持模型的一致性。

    To make effective decisions in novel environments with long-horizon goals, it is crucial to engage in hierarchical reasoning across spatial and temporal scales. This entails planning abstract subgoal sequences, visually reasoning about the underlying plans, and executing actions in accordance with the devised plan through visual-motor control. We propose Compositional Foundation Models for Hierarchical Planning (HiP), a foundation model which leverages multiple expert foundation model trained on language, vision and action data individually jointly together to solve long-horizon tasks. We use a large language model to construct symbolic plans that are grounded in the environment through a large video diffusion model. Generated video plans are then grounded to visual-motor control, through an inverse dynamics model that infers actions from generated videos. To enable effective reasoning within this hierarchy, we enforce consistency between the models via iterative refinement. We illustr
    
[^5]: 在视觉变换器中用ReLU替换softmax

    Replacing softmax with ReLU in Vision Transformers. (arXiv:2309.08586v1 [cs.CV])

    [http://arxiv.org/abs/2309.08586](http://arxiv.org/abs/2309.08586)

    在视觉变换器中，用ReLU替换softmax的注意力机制可以在计算性能上接近或匹配softmax注意力，并且通过序列长度进行除法可以缓解精度下降的问题。

    

    先前的研究观察到当将注意力softmax替换为ReLU这样的逐点激活时，精度会下降。在视觉变换器的背景下，我们发现当通过序列长度除以注意力下降被缓解。我们在ImageNet-21k数据集上训练小型到大型视觉变换器的实验表明，在计算方面，ReLU注意力可以达到或匹配softmax注意力的性能。

    Previous research observed accuracy degradation when replacing the attention softmax with a point-wise activation such as ReLU. In the context of vision transformers, we find that this degradation is mitigated when dividing by sequence length. Our experiments training small to large vision transformers on ImageNet-21k indicate that ReLU-attention can approach or match the performance of softmax-attention in terms of scaling behavior as a function of compute.
    
[^6]: 一种贝叶斯方法用于稳健的逆强化学习

    A Bayesian Approach to Robust Inverse Reinforcement Learning. (arXiv:2309.08571v1 [cs.LG])

    [http://arxiv.org/abs/2309.08571](http://arxiv.org/abs/2309.08571)

    这篇论文提出了一种贝叶斯方法，用于稳健的离线模型导向的逆强化学习。通过同时估计专家的奖励函数和主观模型的环境动态，利用先验分布参数化专家对环境模型的准确性，提出了高效的算法。实验证明，当先验地认为专家对环境具有高度准确的模型时，估计的策略表现出稳健性能，并且优于最先进的离线IRL算法。

    

    我们考虑一种贝叶斯方法用于离线模型导向的逆强化学习(IRL)。所提出的框架通过同时估计专家的奖励函数和主观模型的环境动态，区别于现有的离线模型导向的IRL方法。我们利用一类先验分布来参数化专家对环境的模型的准确性，以开发在高维环境中估计专家奖励和主观动态的高效算法。我们的分析揭示了一个新的见解，即当先验地认为专家对环境具有高度准确的模型时，估计的策略表现出稳健性能。我们在MuJoCo环境中验证了这一观察，并展示了我们的算法胜过最先进的离线IRL算法。

    We consider a Bayesian approach to offline model-based inverse reinforcement learning (IRL). The proposed framework differs from existing offline model-based IRL approaches by performing simultaneous estimation of the expert's reward function and subjective model of environment dynamics. We make use of a class of prior distributions which parameterizes how accurate the expert's model of the environment is to develop efficient algorithms to estimate the expert's reward and subjective dynamics in high-dimensional settings. Our analysis reveals a novel insight that the estimated policy exhibits robust performance when the expert is believed (a priori) to have a highly accurate model of the environment. We verify this observation in the MuJoCo environments and show that our algorithms outperform state-of-the-art offline IRL algorithms.
    
[^7]: 基于群组贡献方法的非线性光学分子的神经网络驱动的交互式设计

    Neural Network Driven, Interactive Design for Nonlinear Optical Molecules Based on Group Contribution Method. (arXiv:2309.08570v1 [stat.ML])

    [http://arxiv.org/abs/2309.08570](http://arxiv.org/abs/2309.08570)

    本研究提出了一种基于神经网络的交互式设计方法，结合群组贡献方法和机器学习技术，能够准确快速地设计非线性光学分子。该方法不仅能够准确预测不同分子的光学性质，还可以实现有效的结构搜索。

    

    本文报道了一种基于Lewis模型群组贡献方法（LGC）- 多阶段贝叶斯神经网络（msBNN）- 进化算法（EA）框架，用于合理设计D-Pi-A型有机小分子非线性光学材料。通过结合msBNN和校正的Lewis模型群组贡献方法（cLGC），可以准确高效地获得分子的不同光学性质 - 仅使用小型数据集进行训练。此外，通过使用专为LGC设计的EA模型，可以实现良好的结构搜索。详细讨论了该框架表现良好的逻辑原因。考虑到这种理论引导的机器学习框架结合了化学原理和数据驱动工具，很可能被证明在更广泛的领域中解决分子设计相关问题时有效。

    A Lewis-mode group contribution method (LGC) -- multi-stage Bayesian neural network (msBNN) -- evolutionary algorithm (EA) framework is reported for rational design of D-Pi-A type organic small-molecule nonlinear optical materials is presented. Upon combination of msBNN and corrected Lewis-mode group contribution method (cLGC), different optical properties of molecules are afforded accurately and efficiently - by using only a small data set for training. Moreover, by employing the EA model designed specifically for LGC, structural search is well achievable. The logical origins of the well performance of the framework are discussed in detail. Considering that such a theory guided, machine learning framework combines chemical principles and data-driven tools, most likely, it will be proven efficient to solve molecular design related problems in wider fields.
    
[^8]: 图神经网络中的局部差分隐私：一种重构方法

    Local Differential Privacy in Graph Neural Networks: a Reconstruction Approach. (arXiv:2309.08569v1 [cs.LG])

    [http://arxiv.org/abs/2309.08569](http://arxiv.org/abs/2309.08569)

    本研究提出了一种学习框架，旨在为用户提供节点隐私保护，并通过在节点级别对特征和标签数据进行随机化扰动来实现。通过频率估计和重构方法，实现了对扰动数据中特征和标签的恢复。

    

    图神经网络在各种应用中对建模复杂图数据取得了巨大成功。然而，有关GNN的隐私保护的研究还很有限。在本文中，我们提出了一个学习框架，可以在不丧失太多效用的情况下为用户提供节点隐私保护。我们关注一种去中心化的差分隐私概念，即局部差分隐私，并在数据被集中服务器进行模型训练之前，对节点级别的特征和标签数据应用随机化机制进行扰动。具体而言，我们研究了在高维特征设置中应用随机化机制的方法，并提出了具有严格隐私保证的LDP协议。基于随机化数据的统计分析中的频率估计，我们开发了重构方法来近似从扰动数据中恢复特征和标签。我们还制定了这个学习框架，利用了图聚类中的频率估计。

    Graph Neural Networks have achieved tremendous success in modeling complex graph data in a variety of applications. However, there are limited studies investigating privacy protection in GNNs. In this work, we propose a learning framework that can provide node privacy at the user level, while incurring low utility loss. We focus on a decentralized notion of Differential Privacy, namely Local Differential Privacy, and apply randomization mechanisms to perturb both feature and label data at the node level before the data is collected by a central server for model training. Specifically, we investigate the application of randomization mechanisms in high-dimensional feature settings and propose an LDP protocol with strict privacy guarantees. Based on frequency estimation in statistical analysis of randomized data, we develop reconstruction methods to approximate features and labels from perturbed data. We also formulate this learning framework to utilize frequency estimates of graph cluste
    
[^9]: 自适应实例归一化的开放词汇关键词检测

    Open-vocabulary Keyword-spotting with Adaptive Instance Normalization. (arXiv:2309.08561v1 [eess.AS])

    [http://arxiv.org/abs/2309.08561](http://arxiv.org/abs/2309.08561)

    本论文提出了一种名为AdaKWS的新方法，用于解决自动语音识别中的开放词汇关键词检测任务。通过训练文本编码器输出关键词条件的归一化参数，并将其应用于处理听觉输入，我们在多语言基准测试上取得了显著的性能改进。

    

    开放词汇关键词检测是自动语音识别（ASR）中的关键和具有挑战性的任务，其重点是识别口语中用户定义的关键词。关键词检测方法通常将音频和关键词映射到一个联合嵌入空间中，以获得一些相关性得分。在本文中，我们提出了AdaKWS，一种新方法用于关键词检测，在这种方法中，训练一个文本编码器以输出关键词条件归一化参数。这些参数用于处理听觉输入。我们使用具有挑战性和多样化的多语言基准进行了全面的评估，并显示出与最近的关键词检测和ASR基准相比的显着改进。此外，我们研究了我们的方法在训练过程中未见的低资源语言上的有效性。结果表明与基准方法相比存在显着的性能改进。

    Open vocabulary keyword spotting is a crucial and challenging task in automatic speech recognition (ASR) that focuses on detecting user-defined keywords within a spoken utterance. Keyword spotting methods commonly map the audio utterance and keyword into a joint embedding space to obtain some affinity score. In this work, we propose AdaKWS, a novel method for keyword spotting in which a text encoder is trained to output keyword-conditioned normalization parameters. These parameters are used to process the auditory input. We provide an extensive evaluation using challenging and diverse multi-lingual benchmarks and show significant improvements over recent keyword spotting and ASR baselines. Furthermore, we study the effectiveness of our approach on low-resource languages that were unseen during the training. The results demonstrate a substantial performance improvement compared to baseline methods.
    
[^10]: 用于高效且公平分配医疗资源的深度强化学习

    Deep Reinforcement Learning for Efficient and Fair Allocation of Health Care Resources. (arXiv:2309.08560v1 [cs.LG])

    [http://arxiv.org/abs/2309.08560](http://arxiv.org/abs/2309.08560)

    本研究使用强化学习方法，通过整合个体患者的疾病进展和患者间的相互作用效应，来优化医疗资源的分配策略，旨在提高分配的公平性和整体患者结果。

    

    医疗资源的稀缺性可能导致不可避免的配给问题。例如，通气机的供应通常有限，特别是在公共卫生紧急情况或资源有限的医疗环境中，如COVID-19大流行期间。目前，针对医疗资源分配的协议并没有普遍接受的标准，导致各国政府根据不同的标准和基于启发式协议来优先考虑患者。在本研究中，我们研究了使用强化学习来优化重症护理资源分配策略，以公平有效地配给资源。我们提出了基于变换器的深度Q网络，用于将个体患者的病情进展和患者间的相互作用效应整合到重症护理资源分配中。我们的目标是提高分配的公平性和整体患者结果。我们的实验表明，我们的方法显著减少了过度配给资源的情况。

    Scarcity of health care resources could result in the unavoidable consequence of rationing. For example, ventilators are often limited in supply, especially during public health emergencies or in resource-constrained health care settings, such as amid the pandemic of COVID-19. Currently, there is no universally accepted standard for health care resource allocation protocols, resulting in different governments prioritizing patients based on various criteria and heuristic-based protocols. In this study, we investigate the use of reinforcement learning for critical care resource allocation policy optimization to fairly and effectively ration resources. We propose a transformer-based deep Q-network to integrate the disease progression of individual patients and the interaction effects among patients during the critical care resource allocation. We aim to improve both fairness of allocation and overall patient outcomes. Our experiments demonstrate that our method significantly reduces exces
    
[^11]: 基于健康影响力噪声的训练来抵御数据污染攻击的方法

    HINT: Healthy Influential-Noise based Training to Defend against Data Poisoning Attacks. (arXiv:2309.08549v1 [cs.LG])

    [http://arxiv.org/abs/2309.08549](http://arxiv.org/abs/2309.08549)

    本论文提出了一种名为健康影响力噪声训练的高效稳健训练方法，该方法使用影响函数制造了有助于加强分类模型对抗数据污染攻击的健康噪声，并且在仅修改训练数据的子集时也能有效运行。

    

    虽然已经提出了许多防御方法来防止来自不可信数据源的潜在污染攻击，但大多数研究仅针对特定攻击进行防御，这给了攻击者许多可利用的机会。在本论文中，我们提出了一种基于影响函数的高效稳健训练方法，名为健康影响力噪声训练。通过使用影响函数，我们制造了有助于加强分类模型对抗污染攻击的健康噪声，同时不会对测试数据的泛化能力产生显著影响。此外，我们的方法可以在仅修改训练数据的子集时有效运行，而不是如几种之前的方法中那样向所有示例添加噪声。我们在两个图像数据集上进行了全面评估，并考虑不同的实际攻击场景下的最新攻击技术。我们的实证结果表明，H

    While numerous defense methods have been proposed to prohibit potential poisoning attacks from untrusted data sources, most research works only defend against specific attacks, which leaves many avenues for an adversary to exploit. In this work, we propose an efficient and robust training approach to defend against data poisoning attacks based on influence functions, named Healthy Influential-Noise based Training. Using influence functions, we craft healthy noise that helps to harden the classification model against poisoning attacks without significantly affecting the generalization ability on test data. In addition, our method can perform effectively when only a subset of the training data is modified, instead of the current method of adding noise to all examples that has been used in several previous works. We conduct comprehensive evaluations over two image datasets with state-of-the-art poisoning attacks under different realistic attack scenarios. Our empirical results show that H
    
[^12]: 基于贝叶斯自适应时刻正则化的鲁棒性持续学习

    Towards Robust Continual Learning with Bayesian Adaptive Moment Regularization. (arXiv:2309.08546v1 [cs.LG])

    [http://arxiv.org/abs/2309.08546](http://arxiv.org/abs/2309.08546)

    基于贝叶斯自适应时刻正则化的鲁棒性持续学习方法能够在机器人应用中有效地解决灾难性遗忘问题，并具有轻量级和任务实验室等优势。

    

    为了追求长期自主性，机器人代理必须不断适应不断变化的环境并学习解决新任务。持续学习试图克服灾难性遗忘的挑战，即学习解决新任务导致模型忘记先前学到的信息。基于先验的持续学习方法对于机器人应用具有吸引力，因为它们在空间效率上很高，并且通常不会随着任务数量的增加而增加计算复杂性。尽管具有这些理想的特性，但基于先验的方法通常在重要的基准测试中失败，因此与基于记忆的方法相比，在潜在应用方面有限。我们引入了贝叶斯自适应时刻正则化（BAdam），一种新的基于先验的方法，它更好地约束参数增长，降低灾难性遗忘。我们的方法在机器人应用中具有一系列理想的特性，例如轻量级和任务实验室。

    The pursuit of long-term autonomy mandates that robotic agents must continuously adapt to their changing environments and learn to solve new tasks. Continual learning seeks to overcome the challenge of catastrophic forgetting, where learning to solve new tasks causes a model to forget previously learnt information. Prior-based continual learning methods are appealing for robotic applications as they are space efficient and typically do not increase in computational complexity as the number of tasks grows. Despite these desirable properties, prior-based approaches typically fail on important benchmarks and consequently are limited in their potential applications compared to their memory-based counterparts. We introduce Bayesian adaptive moment regularization (BAdam), a novel prior-based method that better constrains parameter growth, leading to lower catastrophic forgetting. Our method boasts a range of desirable properties for robotic applications such as being lightweight and task lab
    
[^13]: 复杂环境中高效稳健的传感器部署

    Efficient and robust Sensor Placement in Complex Environments. (arXiv:2309.08545v1 [cs.LG])

    [http://arxiv.org/abs/2309.08545](http://arxiv.org/abs/2309.08545)

    本文解决了在复杂环境中高效稳健的传感器部署问题，提出了一种贪婪算法来设计最小传感器集合以实现多覆盖约束，并探索了深度学习技术加速算法的评估。

    

    我们解决了在复杂环境中高效无阻的监视或通信问题。一方面，人们希望使用最少数量的传感器覆盖环境。另一方面，考虑到传感器故障或对抗攻击的情况，设计具有鲁棒性的解决方案通常很重要。本文解决了设计能够实现多覆盖约束的最小传感器集合的挑战，即环境中的每个点都被一定数量的传感器覆盖。我们提出了一种贪婪算法来实现这个目标。此外，我们探索了深度学习技术来加速贪婪算法中所制定的目标函数的评估。神经网络的训练揭示了数据的几何属性对网络性能的显著影响，特别是在最终阶段。通过考虑这些属性，我们讨论了使用贪婪算法和ε-贪婪算法生成数据的差异。

    We address the problem of efficient and unobstructed surveillance or communication in complex environments. On one hand, one wishes to use a minimal number of sensors to cover the environment. On the other hand, it is often important to consider solutions that are robust against sensor failure or adversarial attacks. This paper addresses these challenges of designing minimal sensor sets that achieve multi-coverage constraints -- every point in the environment is covered by a prescribed number of sensors. We propose a greedy algorithm to achieve the objective. Further, we explore deep learning techniques to accelerate the evaluation of the objective function formulated in the greedy algorithm. The training of the neural network reveals that the geometric properties of the data significantly impact the network's performance, particularly at the end stage. By taking into account these properties, we discuss the differences in using greedy and $\epsilon$-greedy algorithms to generate data 
    
[^14]: 朝着使用更少标注实现群体鲁棒性的最后一层再训练

    Towards Last-layer Retraining for Group Robustness with Fewer Annotations. (arXiv:2309.08534v1 [cs.LG])

    [http://arxiv.org/abs/2309.08534](http://arxiv.org/abs/2309.08534)

    本研究发现，在没有群体标注和只有少量类别标注的情况下，最后一层再训练仍然可以有效提高最差群体准确性，从而优于经验风险最小化在整个数据集上的表现，可以实现群体鲁棒性的提升。

    

    神经网络的经验风险最小化(ERM)容易过度依赖虚假相关性，并在少数群体上具有较差的泛化性能。最近的深度特征再赋权(DFR)技术通过简单的最后一层再训练实现了最先进的群体保护性能，但它需要保留群体和类别的标注，并构建一个群体平衡的再赋权数据集。在这项工作中，我们研究了这个不切实际的要求，并发现即使没有群体标注（除了模型选择），只有少量的类别标注，最后一层再训练仍然可以出人意料地有效。我们首先证明了即使再赋权数据集中仅有一小部分最差群体数据，最后一层再训练仍然可以显著提高最差群体准确性。这意味着通过保留一部分训练数据来重新训练最后一层，可以在没有额外数据或标注的情况下，显著优于对整个数据集进行ERM。为了进一步提高群体鲁棒性，我们...

    Empirical risk minimization (ERM) of neural networks is prone to over-reliance on spurious correlations and poor generalization on minority groups. The recent deep feature reweighting (DFR) technique achieves state-of-the-art group robustness via simple last-layer retraining, but it requires held-out group and class annotations to construct a group-balanced reweighting dataset. In this work, we examine this impractical requirement and find that last-layer retraining can be surprisingly effective with no group annotations (other than for model selection) and only a handful of class annotations. We first show that last-layer retraining can greatly improve worst-group accuracy even when the reweighting dataset has only a small proportion of worst-group data. This implies a "free lunch" where holding out a subset of training data to retrain the last layer can substantially outperform ERM on the entire dataset with no additional data or annotations. To further improve group robustness, we i
    
[^15]: 针对稀疏连接模型的尺度定律研究

    Scaling Laws for Sparsely-Connected Foundation Models. (arXiv:2309.08520v1 [cs.LG])

    [http://arxiv.org/abs/2309.08520](http://arxiv.org/abs/2309.08520)

    本研究探讨了参数稀疏性对基于大规模数据集训练的Transformer模型在视觉和语言领域中的尺度行为影响，并通过实验证明了权重稀疏性、非零参数数量和训练数据量之间的尺度定律。研究结果可以帮助确定对于给定的有效模型大小和训练预算，所需的最佳稀疏水平。同时，研究还拓展了对不同稀疏结构和策略的探究，揭示了权重稀疏性的能力和限制。

    

    本文研究了参数稀疏性对基于大规模数据集训练的Transformer（即“基础模型”）在视觉和语言领域中的尺度行为的影响。在这个设定下，我们首次确定了描述权重稀疏性、非零参数数量和训练数据量之间关系的尺度定律，并在ViT/JFT-4B和T5/C4上通过实验证明了该定律在模型和数据规模上的适用性。这些结果使我们能够确定“最佳稀疏性”，即对于给定的有效模型大小和训练预算，能够获得最佳性能的稀疏水平。对于固定数量的非零参数，我们发现最佳稀疏性随着用于训练的数据量的增加而增加。我们还扩展了研究范围，包括不同的稀疏结构（如硬件友好的n:m模式）和策略（如从预训练稠密模型开始）。本研究结果对于理解权重稀疏性在不同参数和条件下的能力和限制具有重要意义。

    We explore the impact of parameter sparsity on the scaling behavior of Transformers trained on massive datasets (i.e., "foundation models"), in both vision and language domains. In this setting, we identify the first scaling law describing the relationship between weight sparsity, number of non-zero parameters, and amount of training data, which we validate empirically across model and data scales; on ViT/JFT-4B and T5/C4. These results allow us to characterize the "optimal sparsity", the sparsity level which yields the best performance for a given effective model size and training budget. For a fixed number of non-zero parameters, we identify that the optimal sparsity increases with the amount of data used for training. We also extend our study to different sparsity structures (such as the hardware-friendly n:m pattern) and strategies (such as starting from a pretrained dense model). Our findings shed light on the power and limitations of weight sparsity across various parameter and c
    
[^16]: 广义概率扩散尺度空间

    Generalised Probabilistic Diffusion Scale-Spaces. (arXiv:2309.08511v1 [eess.IV])

    [http://arxiv.org/abs/2309.08511](http://arxiv.org/abs/2309.08511)

    本研究提出了概率扩散模型的广义尺度空间理论，探索了它与经典图像滤波的关系，以及与扩散和渗透滤波的概念和经验联系。

    

    概率扩散模型在从学习的分布中采样新图像方面表现出色。最初由物理学中的漂移扩散概念推动，它们通过正向过程应用图像扰动，如噪声和模糊，从而产生可处理的概率分布。相应的学习逆过程生成图像，并可在附加信息条件下进行调整，从而导致各种实际应用。目前大部分研究重点放在实践导向的扩展上。相比之下，理论背景仍然大部分未被探索，尤其是与漂移扩散的关系。为了阐明与经典图像滤波的连接，我们提出了概率扩散模型的广义尺度空间理论。此外，我们展示了与扩散和渗透滤波的概念和经验联系。

    Probabilistic diffusion models excel at sampling new images from learned distributions. Originally motivated by drift-diffusion concepts from physics, they apply image perturbations such as noise and blur in a forward process that results in a tractable probability distribution. A corresponding learned reverse process generates images and can be conditioned on side information, which leads to a wide variety of practical applications. Most of the research focus currently lies on practice-oriented extensions. In contrast, the theoretical background remains largely unexplored, in particular the relations to drift-diffusion. In order to shed light on these connections to classical image filtering, we propose a generalised scale-space theory for probabilistic diffusion models. Moreover, we show conceptual and empirical connections to diffusion and osmosis filters.
    
[^17]: 深度学习在浮游生态学数据分析中的应用

    Deep-learning-powered data analysis in plankton ecology. (arXiv:2309.08500v1 [physics.bio-ph])

    [http://arxiv.org/abs/2309.08500](http://arxiv.org/abs/2309.08500)

    深度学习在浮游生态学数据分析中的应用提供了客观的方案，能加快分析速度、减少实验偏差，并促进浮游生态相关研究的发展。

    

    深度学习算法的实施为浮游生态学带来了新的视角。作为替代传统方法的一种新途径，深度学习提供了客观的方案来研究不同环境中的浮游生物。我们提供了基于深度学习的方法概述，包括浮游植物和浮游动物图像的检测和分类、觅食和游动行为分析，以及生态建模。深度学习有望加快分析速度，减少人为实验偏差，从而实现在相关的时间和空间尺度上获取数据，并提高再现性。我们还讨论了深度学习的局限性，并展示了深度学习架构如何演变以减少不准确结果。最后，我们提出了深度学习在浮游生态研究中特别可能推动发展的机会。这些例子附带了详细的教程和代码示例，使读者能够应用所描述的方法。

    The implementation of deep learning algorithms has brought new perspectives to plankton ecology. Emerging as an alternative approach to established methods, deep learning offers objective schemes to investigate plankton organisms in diverse environments. We provide an overview of deep-learning-based methods including detection and classification of phytoand zooplankton images, foraging and swimming behaviour analysis, and finally ecological modelling. Deep learning has the potential to speed up the analysis and reduce the human experimental bias, thus enabling data acquisition at relevant temporal and spatial scales with improved reproducibility. We also discuss shortcomings and show how deep learning architectures have evolved to mitigate imprecise readouts. Finally, we suggest opportunities where deep learning is particularly likely to catalyze plankton research. The examples are accompanied by detailed tutorials and code samples that allow readers to apply the methods described in
    
[^18]: P-ROCKET: 针对时间序列分类的随机卷积核剪枝

    P-ROCKET: Pruning Random Convolution Kernels for Time Series Classification. (arXiv:2309.08499v1 [cs.LG])

    [http://arxiv.org/abs/2309.08499](http://arxiv.org/abs/2309.08499)

    本研究提出了一种名为P-ROCKET的方法，通过在特征选择的角度删除卷积核，从而实现对时间序列分类中的随机卷积核进行剪枝。

    

    在最近几年，两个时间序列分类模型ROCKET和MINIROCKET因其低训练成本和最先进的准确性而受到广泛关注。ROCKET和MINIROCKET利用无需训练的随机一维卷积核，可以快速从时间序列数据中提取特征，从而实现线性分类器的高效拟合。然而，为了全面捕捉有用的特征，需要大量的随机卷积核，这对于资源受限的设备来说是不兼容的。因此，我们设计了一种启发式进化算法S-ROCKET，用于识别和剪枝冗余的卷积核。然而，进化算法本身的特性导致在S-ROCKET中评估卷积核是一个耗时的过程。本文中，与直接评估具有非显著差异的随机卷积核的S-ROCKET不同，我们从特征选择的角度删除卷积核，通过消除序列中的相关连接来实现。

    In recent years, two time series classification models, ROCKET and MINIROCKET, have attracted much attention for their low training cost and state-of-the-art accuracy. Utilizing random 1-D convolutional kernels without training, ROCKET and MINIROCKET can rapidly extract features from time series data, allowing for the efficient fitting of linear classifiers. However, to comprehensively capture useful features, a large number of random kernels are required, which is incompatible for resource-constrained devices. Therefore, a heuristic evolutionary algorithm named S-ROCKET is devised to recognize and prune redundant kernels. Nevertheless, the inherent nature of evolutionary algorithms renders the evaluation of kernels within S-ROCKET an unacceptable time-consuming process. In this paper, diverging from S-ROCKET, which directly evaluates random kernels with nonsignificant differences, we remove kernels from a feature selection perspective by eliminating associating connections in the sequ
    
[^19]: 朝向词级端到端神经发言人分离与辅助网络

    Towards Word-Level End-to-End Neural Speaker Diarization with Auxiliary Network. (arXiv:2309.08489v1 [eess.AS])

    [http://arxiv.org/abs/2309.08489](http://arxiv.org/abs/2309.08489)

    本文提出了一种名为WEEND的词级端到端神经网络方法，通过使用辅助网络，实现了同时进行自动语音识别和发言人分离，并在2个发言人的短片场景中取得了优于基线系统的性能。

    

    尽管标准的发言人分离试图回答“谁在什么时候说了什么”，但现实中大多数相关应用更关心确定“谁说了什么”。无论是传统的模块化方法还是最近的端到端神经分离（EEND），都需要一个额外的自动语音识别（ASR）模型和一个协调算法来将说话者标签与识别的单词关联起来。在本文中，我们提出了一种带有辅助网络的词级端到端神经分离（WEEND），这是一种多任务学习算法，它在相同的神经架构中执行端到端ASR和发言人分离。也就是说，当语音被识别时，同时为每个识别的单词预测说话者标签。实验结果表明，WEEND在所有两个发言人的短片场景上优于基线系统，并且能够推广到5分钟的音频长度。尽管在3个或更多发言人的情况下，相对于基线系统，不能达到最佳性能。

    While standard speaker diarization attempts to answer the question "who spoken when", most of relevant applications in reality are more interested in determining "who spoken what". Whether it is the conventional modularized approach or the more recent end-to-end neural diarization (EEND), an additional automatic speech recognition (ASR) model and an orchestration algorithm are required to associate the speaker labels with recognized words. In this paper, we propose Word-level End-to-End Neural Diarization (WEEND) with auxiliary network, a multi-task learning algorithm that performs end-to-end ASR and speaker diarization in the same neural architecture. That is, while speech is being recognized, speaker labels are predicted simultaneously for each recognized word. Experimental results demonstrate that WEEND outperforms the turn-based diarization baseline system on all 2-speaker short-form scenarios and has the capability to generalize to audio lengths of 5 minutes. Although 3+speaker co
    
[^20]: 分布式主动假设测试的深度多智能体强化学习

    Deep Multi-Agent Reinforcement Learning for Decentralized Active Hypothesis Testing. (arXiv:2309.08477v1 [stat.ML])

    [http://arxiv.org/abs/2309.08477](http://arxiv.org/abs/2309.08477)

    这个论文提出了一个分布式主动假设测试（AHT）问题的解决方法，通过多智能体强化学习，设计一个策略来在有限通信通道上合作完成任务，将贝叶斯风险最小化。

    

    我们考虑了分布式主动假设测试（AHT）问题的一个分布式形式，在这个问题中，多个智能体从环境中收集到带噪声的观测数据，目的是识别出正确的假设。在每个时间步骤中，智能体可以选择一个采样动作，这些不同的动作会导致从不同分布中抽取观测数据，每个分布与一个特定的假设相关联。智能体通过在有限速率的通信通道上进行消息交换来合作完成任务。目标是设计一个多智能体策略，将贝叶斯风险最小化。这种风险包括采样成本和智能体在声明假设时产生的联合终端成本。在AHT问题中推导出最优的结构化策略通常在数学上是难以处理的，即使是在单个智能体的背景下也是如此。因此，最近的研究工作转向深度学习方法来解决这些问题，这些方法包括...

    We consider a decentralized formulation of the active hypothesis testing (AHT) problem, where multiple agents gather noisy observations from the environment with the purpose of identifying the correct hypothesis. At each time step, agents have the option to select a sampling action. These different actions result in observations drawn from various distributions, each associated with a specific hypothesis. The agents collaborate to accomplish the task, where message exchanges between agents are allowed over a rate-limited communications channel. The objective is to devise a multi-agent policy that minimizes the Bayes risk. This risk comprises both the cost of sampling and the joint terminal cost incurred by the agents upon making a hypothesis declaration. Deriving optimal structured policies for AHT problems is generally mathematically intractable, even in the context of a single agent. As a result, recent efforts have turned to deep learning methodologies to address these problems, whi
    
[^21]: 数据驱动天气预报模型的局限性研究

    On the limitations of data-driven weather forecasting models. (arXiv:2309.08473v1 [stat.ML])

    [http://arxiv.org/abs/2309.08473](http://arxiv.org/abs/2309.08473)

    数据驱动的机器学习天气预报模型不具备传统基于物理的模型的准确性和物理一致性，它们在预测技能上的优势很大程度上可以归因于这些特殊性。

    

    机器学习在天气和气候预测领域产生了深远影响。最近的发展是数据驱动的机器学习预测模型的出现，它们通常声称比传统的基于物理的模型具有更高的性能。在这项工作中，我们研究了当前一代机器学习模型之一Pangu-Weather的预测方面的一些问题，重点关注预测的准确性和物理一致性以及这些特征与感知预测性能之间的关系。主要结论是Pangu-Weather的预测，以及类似的机器学习模型，不具备基于物理的模型的准确性和物理一致性，而它们在传统的确定性预测技能指标上的优势很大程度上可以归因于这些特殊性。与其他当前的后处理技术类似。

    As in many other areas of engineering and applied science, Machine Learning (ML) is having a profound impact in the domain of Weather and Climate Prediction. A very recent development in this area has been the emergence of fully data-driven ML prediction models which routinely claim superior performance to that of traditional physics-based models. In this work, we examine some aspects of the forecasts produced by an exemplar of the current generation of ML models, Pangu-Weather, with a focus on the fidelity and physical consistency of those forecasts and how these characteristics relate to perceived forecast performance. The main conclusion is that Pangu-Weather forecasts, and by extension those of similar ML models, do not have the fidelity and physical consistency of physics-based models and their advantage in accuracy on traditional deterministic metrics of forecast skill can be attributed, to a large extent, to these peculiarities. Similarly to other current post-processing technol
    
[^22]: 向有观点的人解释搜索结果立场

    Explaining Search Result Stances to Opinionated People. (arXiv:2309.08460v1 [cs.IR])

    [http://arxiv.org/abs/2309.08460](http://arxiv.org/abs/2309.08460)

    这项研究探讨了向有观点的人解释搜索结果立场的效果，发现立场标签和解释可以帮助用户消费更多不同的搜索结果，但没有发现系统性观点改变的证据。

    

    人们在形成观点之前使用网络搜索引擎找到信息，这可能导致具有不同影响水平的实际决策。搜索的认知努力可能使有观点的用户容易受到认知偏见的影响，例如确认偏见。在本文中，我们调查立场标签及其解释是否可以帮助用户消费更多不同的搜索结果。我们自动对三个主题（知识产权、校服和无神论）的搜索结果进行分类和标记，分为反对、中立和支持，并为这些标签生成解释。在一项用户研究中（N =203），我们调查了搜索结果立场偏见（平衡 vs 偏见）和解释水平（纯文本、仅标签、标签和解释）是否会影响被点击的搜索结果的多样性。我们发现立场标签和解释可以导致更多样化的搜索结果消费。然而，我们并没有发现系统性观点改变的证据。

    People use web search engines to find information before forming opinions, which can lead to practical decisions with different levels of impact. The cognitive effort of search can leave opinionated users vulnerable to cognitive biases, e.g., the confirmation bias. In this paper, we investigate whether stance labels and their explanations can help users consume more diverse search results. We automatically classify and label search results on three topics (i.e., intellectual property rights, school uniforms, and atheism) as against, neutral, and in favor, and generate explanations for these labels. In a user study (N =203), we then investigate whether search result stance bias (balanced vs biased) and the level of explanation (plain text, label only, label and explanation) influence the diversity of search results clicked. We find that stance labels and explanations lead to a more diverse search result consumption. However, we do not find evidence for systematic opinion change among us
    
[^23]: 混合编码器支持连续语音分离用于会议识别

    Mixture Encoder Supporting Continuous Speech Separation for Meeting Recognition. (arXiv:2309.08454v1 [eess.AS])

    [http://arxiv.org/abs/2309.08454](http://arxiv.org/abs/2309.08454)

    本研究将混合编码器方法从两个说话人情况扩展到了更自然的会议环境，包括任意数量的说话人和动态重叠。实验证明，该方法在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。

    

    自动语音识别（ASR）的许多实际应用需要处理重叠的语音。一种常见的方法是首先将语音分离成无重叠的流，然后对生成的信号进行ASR。最近，提出了在ASR模型中包含混合编码器的方法。该混合编码器利用原始重叠的语音来减轻语音分离引入的伪影效果。然而，先前的方法仅针对两个说话人的情况。在这项工作中，我们将这种方法扩展到更自然的会议环境，包括任意数量的说话人和动态重叠。我们使用不同的语音分离器（包括强大的TF-GridNet模型）评估性能。实验证明，在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。此外，实验还展示了TF-GridNet的强大分离能力，大大缩小了先前方法的差距。

    Many real-life applications of automatic speech recognition (ASR) require processing of overlapped speech. A commonmethod involves first separating the speech into overlap-free streams and then performing ASR on the resulting signals. Recently, the inclusion of a mixture encoder in the ASR model has been proposed. This mixture encoder leverages the original overlapped speech to mitigate the effect of artifacts introduced by the speech separation. Previously, however, the method only addressed two-speaker scenarios. In this work, we extend this approach to more natural meeting contexts featuring an arbitrary number of speakers and dynamic overlaps. We evaluate the performance using different speech separators, including the powerful TF-GridNet model. Our experiments show state-of-the-art performance on the LibriCSS dataset and highlight the advantages of the mixture encoder. Furthermore, they demonstrate the strong separation of TF-GridNet which largely closes the gap between previous m
    
[^24]: 朝着负责任的人脸数据集：对从人口群体中采样人脸图像的分解潜空间分布建模

    Toward responsible face datasets: modeling the distribution of a disentangled latent space for sampling face images from demographic groups. (arXiv:2309.08442v1 [cs.CV])

    [http://arxiv.org/abs/2309.08442](http://arxiv.org/abs/2309.08442)

    本文提出了一种方法，通过建模和采样分解潜空间的方法来生成任意组合的人口群体，以解决现代人脸识别系统中数据集偏见导致的不公平关注问题。

    

    最近，一些现代人脸识别系统被曝光出可能对特定人口群体进行歧视，并可能导致对性别和出身等各种面部属性的不公平关注。原因在于被用于训练这些模型的数据集中存在偏见和不平衡的人口统计数据。然而，采集一个各个人口统计数据都平衡的大规模数据集是不可行的。因此，本文探讨了一个替代方案，即生成一个具有平衡性和可能无偏见的合成数据集，以用于训练、正则化或评估基于深度学习的人脸识别模型。我们提出使用一个简单的方法来建模和采样一个StyleGAN潜空间的分解投影，以生成任意组合的人口群体（例如 $hispanic-female$）。我们的实验证明，我们可以有效地合成任意组合的人口群体，且这些身份与原始训练集中的身份不同。

    Recently, it has been exposed that some modern facial recognition systems could discriminate specific demographic groups and may lead to unfair attention with respect to various facial attributes such as gender and origin. The main reason are the biases inside datasets, unbalanced demographics, used to train theses models. Unfortunately, collecting a large-scale balanced dataset with respect to various demographics is impracticable.  In this paper, we investigate as an alternative the generation of a balanced and possibly bias-free synthetic dataset that could be used to train, to regularize or to evaluate deep learning-based facial recognition models. We propose to use a simple method for modeling and sampling a disentangled projection of a StyleGAN latent space to generate any combination of demographic groups (e.g. $hispanic-female$). Our experiments show that we can synthesis any combination of demographic groups effectively and the identities are different from the original traini
    
[^25]: MIML: 通过微流控系统内的机械特性对高精度细胞分类进行多重图像机器学习

    MIML: Multiplex Image Machine Learning for High Precision Cell Classification via Mechanical Traits within Microfluidic Systems. (arXiv:2309.08421v1 [eess.IV])

    [http://arxiv.org/abs/2309.08421](http://arxiv.org/abs/2309.08421)

    本研究开发了一种新颖的机器学习框架MIML，该框架通过将无标记细胞图像与生物力学属性数据相结合，实现了高精度细胞分类。该方法利用了形态信息，将细胞属性理解得更全面，相较于仅考虑单一数据类型的模型，实现了98.3％的分类精度。该方法已在白细胞和肿瘤细胞分类中得到证明，并具有更广泛的应用潜力。

    

    无标记细胞分类有助于为进一步使用或检查提供原始细胞，然而现有技术在特异性和速度方面往往不足。在本研究中，我们通过开发一种新颖的机器学习框架MIML来解决这些局限性。该架构将无标记细胞图像与生物力学属性数据相结合，利用每个细胞固有的广阔且常常被低估的形态信息。通过整合这两种类型的数据，我们的模型提供了对细胞属性更全面的理解，利用了传统机器学习模型中通常被丢弃的形态信息。这种方法使细胞分类精度达到了惊人的98.3％，大大优于仅考虑单一数据类型的模型。MIML已被证明在白细胞和肿瘤细胞分类中有效，并具有更广泛的应用潜力。

    Label-free cell classification is advantageous for supplying pristine cells for further use or examination, yet existing techniques frequently fall short in terms of specificity and speed. In this study, we address these limitations through the development of a novel machine learning framework, Multiplex Image Machine Learning (MIML). This architecture uniquely combines label-free cell images with biomechanical property data, harnessing the vast, often underutilized morphological information intrinsic to each cell. By integrating both types of data, our model offers a more holistic understanding of the cellular properties, utilizing morphological information typically discarded in traditional machine learning models. This approach has led to a remarkable 98.3\% accuracy in cell classification, a substantial improvement over models that only consider a single data type. MIML has been proven effective in classifying white blood cells and tumor cells, with potential for broader applicatio
    
[^26]: FedDCSR: 通过解缠表示学习实现联邦跨领域顺序推荐

    FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning. (arXiv:2309.08420v1 [cs.LG])

    [http://arxiv.org/abs/2309.08420](http://arxiv.org/abs/2309.08420)

    提出了一种名为FedDCSR的联邦跨领域顺序推荐框架，通过解缠表示学习来处理不同领域之间的序列特征异质性，并保护数据隐私。

    

    近年来，利用来自多个领域的用户序列数据的跨领域顺序推荐(CSR)受到了广泛关注。然而，现有的CSR方法需要在领域之间共享原始用户数据，这违反了《通用数据保护条例》(GDPR)。因此，有必要将联邦学习(FL)和CSR相结合，充分利用不同领域的知识，同时保护数据隐私。然而，不同领域之间的序列特征异质性对FL的整体性能有显著影响。在本文中，我们提出了FedDCSR，这是一种通过解缠表示学习的新型联邦跨领域顺序推荐框架。具体而言，为了解决不同领域之间的序列特征异质性，我们引入了一种称为领域内-领域间序列表示解缠(SRD)的方法，将用户序列特征解缠成领域共享和领域专属特征。

    Cross-domain Sequential Recommendation (CSR) which leverages user sequence data from multiple domains has received extensive attention in recent years. However, the existing CSR methods require sharing origin user data across domains, which violates the General Data Protection Regulation (GDPR). Thus, it is necessary to combine federated learning (FL) and CSR to fully utilize knowledge from different domains while preserving data privacy. Nonetheless, the sequence feature heterogeneity across different domains significantly impacts the overall performance of FL. In this paper, we propose FedDCSR, a novel federated cross-domain sequential recommendation framework via disentangled representation learning. Specifically, to address the sequence feature heterogeneity across domains, we introduce an approach called inter-intra domain sequence representation disentanglement (SRD) to disentangle the user sequence features into domain-shared and domain-exclusive features. In addition, we design
    
[^27]: 使用机器学习和不确定性量化对CRT的多阶段决策过程进行建模的新方法

    A new method of modeling the multi-stage decision-making process of CRT using machine learning with uncertainty quantification. (arXiv:2309.08415v1 [cs.LG])

    [http://arxiv.org/abs/2309.08415](http://arxiv.org/abs/2309.08415)

    本研究提出了一种使用机器学习和不确定性量化建模的多阶段决策过程方法，用于预测心力衰竭患者对心脏再同步治疗的反应。该模型能够推荐收集额外的SPECT MPI变量，以提高预测准确性。

    

    目的。本研究旨在创建一个多阶段的机器学习模型，用于预测心力衰竭（HF）患者心脏再同步治疗（CRT）的反应。该模型利用不确定性量化来推荐在基线临床变量和心电图（ECG）的特征不足时收集额外的单光子发射计算机体层摄影心肌灌注显像（SPECT MPI）变量。方法。本研究纳入了218名接受静息门控SPECT MPI的患者。CRT反应被定义为6个月随访时左室射血分数（LVEF）增加> 5%。通过组合两个集成模型创建了一个多阶段的机器学习模型。结果。CRT的反应率为55.5%（n = 121），整体男性占61.0%（n = 133），平均年龄62.0岁，LVEF为27.7。该多阶段模型的性能与集成模型2（利用了额外的SPECT数据）相似，AUC分别为0.75和0.77，准确性分别为0.71和...

    Aims. The purpose of this study is to create a multi-stage machine learning model to predict cardiac resynchronization therapy (CRT) response for heart failure (HF) patients. This model exploits uncertainty quantification to recommend additional collection of single-photon emission computed tomography myocardial perfusion imaging (SPECT MPI) variables if baseline clinical variables and features from electrocardiogram (ECG) are not sufficient. Methods. 218 patients who underwent rest-gated SPECT MPI were enrolled in this study. CRT response was defined as an increase in left ventricular ejection fraction (LVEF) > 5% at a 6 month follow-up. A multi-stage ML model was created by combining two ensemble models. Results. The response rate for CRT was 55.5% (n = 121) with overall male gender 61.0% (n = 133), an average age of 62.0, and LVEF of 27.7. The multi-stage model performed similarly to Ensemble 2 (which utilized the additional SPECT data) with AUC of 0.75 vs. 0.77, accuracy of 0.71 vs
    
[^28]: 再次使深层网络变浅

    Make Deep Networks Shallow Again. (arXiv:2309.08414v1 [cs.LG])

    [http://arxiv.org/abs/2309.08414](http://arxiv.org/abs/2309.08414)

    通过残差连接的概念将顺序深层架构替换为并行浅层架构，大大减轻了梯度消失问题，并提出了通过截断高阶项的方式来得到宽广层的结构。

    

    深度神经网络在复杂应用中有着良好的成功记录，因此被视为最佳架构选择。然而，长期以来它们的主要缺点一直是梯度消失问题，导致数值优化算法无法收敛。通过残差连接的概念，取得了重大突破--在常规层的旁边构建了一个恒等映射。这个概念适用于具有相同维度的层堆叠，并且大大减轻了梯度消失问题。残差连接层堆叠可以表达为类似泰勒展开的项的扩展。这种展开方式提出了截断高阶项的可能性，并可以得到一个由所有初始堆叠层以并行方式组成的单个宽广层的结构。换句话说，将顺序深层架构替换为并行浅层架构。在这一理论的推动下，我们研究了性能可能的上限

    Deep neural networks have a good success record and are thus viewed as the best architecture choice for complex applications. Their main shortcoming has been, for a long time, the vanishing gradient which prevented the numerical optimization algorithms from acceptable convergence. A breakthrough has been achieved by the concept of residual connections -- an identity mapping parallel to a conventional layer. This concept is applicable to stacks of layers of the same dimension and substantially alleviates the vanishing gradient problem. A stack of residual connection layers can be expressed as an expansion of terms similar to the Taylor expansion. This expansion suggests the possibility of truncating the higher-order terms and receiving an architecture consisting of a single broad layer composed of all initially stacked layers in parallel. In other words, a sequential deep architecture is substituted by a parallel shallow one. Prompted by this theory, we investigated the performance capa
    
[^29]: 不受限的平滑有向无环图结构学习

    Constraint-Free Structure Learning with Smooth Acyclic Orientations. (arXiv:2309.08406v1 [cs.LG])

    [http://arxiv.org/abs/2309.08406](http://arxiv.org/abs/2309.08406)

    本文提出了一种无约束的连续优化方案COSMO，用于非环结构学习。通过定义一个可微近似的方向矩阵，并使用单一优先向量进行参数化，我们可以得到一个平滑的方向矩阵和相应的非环邻接矩阵，而无需在任何步骤中评估非环性。尽管没有显式约束，但我们证明COSMO始终收敛到一个非环解。这种方法不仅渐近快速，而且比其他有约束方法具有更小的误差。

    

    结构学习问题是将由有向无环图（DAG）生成的数据正确地重构其弧的问题。在这种情况下，可微化方法使用连续松弛的非环性质对优化问题进行约束或规范化。评估图的非环性的计算成本与节点数量呈三次方关系，严重影响可扩展性。本文介绍了COSMO，一种无约束连续优化方案，用于非环结构学习。在我们的方法的核心，我们定义了一个可微近似的方向矩阵，其由一个优先向量参数化。与以前的工作不同，我们的参数化得到了一个平滑的方向矩阵和相应的非环邻接矩阵，而不需要在任何步骤中评估非环性。尽管没有显式约束，我们证明COSMO始终收敛到一个非环解。除了渐近快速外，我们的经验分析还表明COSMO与其他有约束方法相比具有更小的误差。

    The structure learning problem consists of fitting data generated by a Directed Acyclic Graph (DAG) to correctly reconstruct its arcs. In this context, differentiable approaches constrain or regularize the optimization problem using a continuous relaxation of the acyclicity property. The computational cost of evaluating graph acyclicity is cubic on the number of nodes and significantly affects scalability. In this paper we introduce COSMO, a constraint-free continuous optimization scheme for acyclic structure learning. At the core of our method, we define a differentiable approximation of an orientation matrix parameterized by a single priority vector. Differently from previous work, our parameterization fits a smooth orientation matrix and the resulting acyclic adjacency matrix without evaluating acyclicity at any step. Despite the absence of explicit constraints, we prove that COSMO always converges to an acyclic solution. In addition to being asymptotically faster, our empirical ana
    
[^30]: 优化模块化机器人组合：一种词典遗传算法的方法

    Optimizing Modular Robot Composition: A Lexicographic Genetic Algorithm Approach. (arXiv:2309.08399v1 [cs.RO])

    [http://arxiv.org/abs/2309.08399](http://arxiv.org/abs/2309.08399)

    本论文提出了一种将遗传算法与词典式评估相结合的方法来优化模块化机器人的组合，以克服以往方法中存在的设计空间不足和适应复杂任务的问题，并证明了这种方法在比以往范围更大的搜索空间中表现出更好的性能。

    

    工业机器人被设计为通用硬件，这限制了它们适应任务需求或环境变化的能力。而模块化机器人则提供了灵活性，可以轻松定制以适应不同的需求。机器人的形态，即机器人的形式和结构，对主要性能指标--采购成本、周期时间和能源效率有着重要影响。然而，针对特定任务的最佳模块组合仍然是一个尚未解决的问题，在开发任务定制的模块化机器人中面临重大障碍。以往的方法要么无法充分探索设计空间，要么无法适应复杂任务。我们提出了将遗传算法与词典式评估解决方案候选的组合，以克服这个问题，并在可能组合的数量上比先前的工作范围更大的搜索空间中进行导航。我们证明了我们的方法优于最先进的技术。

    Industrial robots are designed as general-purpose hardware, which limits their ability to adapt to changing task requirements or environments. Modular robots, on the other hand, offer flexibility and can be easily customized to suit diverse needs. The morphology, i.e., the form and structure of a robot, significantly impacts the primary performance metrics acquisition cost, cycle time, and energy efficiency. However, identifying an optimal module composition for a specific task remains an open problem, presenting a substantial hurdle in developing task-tailored modular robots. Previous approaches either lack adequate exploration of the design space or the possibility to adapt to complex tasks. We propose combining a genetic algorithm with a lexicographic evaluation of solution candidates to overcome this problem and navigate search spaces exceeding those in prior work by magnitudes in the number of possible compositions. We demonstrate that our approach outperforms a state-of-the-art b
    
[^31]: 探索基于元信息的基于音频的零样本鸟类分类

    Exploring Meta Information for Audio-based Zero-shot Bird Classification. (arXiv:2309.08398v1 [cs.SD])

    [http://arxiv.org/abs/2309.08398](http://arxiv.org/abs/2309.08398)

    该研究探索了如何利用元信息来改善基于音频的零样本鸟类分类，并通过连接不同的元数据和音频特征获得最佳结果。

    

    被动声学监测和机器学习的进步已经为计算生物声学研究提供了大量数据集。然而，对于稀有和代表性不足的物种来说，数据稀缺仍然是一个问题。本研究通过使用丰富和多样的元数据，以鸟类物种为例进行了探索，研究了如何利用元信息来改善零样本音频分类。我们研究了三种不同的元数据来源：通过(S)BERT编码的文本鸟鸣描述，功能特性(AVONET)和鸟类生活史(BLH)特征。作为音频特征，我们提取音频频谱图变换器(AST)嵌入，并通过采用单个线性层将其投影到辅助信息的维度上。然后，我们采用点积作为兼容性函数，并使用标准的零样本学习排名铰链损失确定正确的类别。通过连接AVONET和BLH特征，我们获得了最佳结果。

    Advances in passive acoustic monitoring and machine learning have led to the procurement of vast datasets for computational bioacoustic research. Nevertheless, data scarcity is still an issue for rare and underrepresented species. This study investigates how meta-information can improve zero-shot audio classification, utilising bird species as an example case study due to the availability of rich and diverse metadata. We investigate three different sources of metadata: textual bird sound descriptions encoded via (S)BERT, functional traits (AVONET), and bird life-history (BLH) characteristics. As audio features, we extract audio spectrogram transformer (AST) embeddings and project them to the dimension of the auxiliary information by adopting a single linear layer. Then, we employ the dot product as compatibility function and a standard zero-shot learning ranking hinge loss to determine the correct class. The best results are achieved by concatenating the AVONET and BLH features attaini
    
[^32]: 学习通过自我解释

    Learning by Self-Explaining. (arXiv:2309.08395v1 [cs.AI])

    [http://arxiv.org/abs/2309.08395](http://arxiv.org/abs/2309.08395)

    学习通过自我解释（LSX）是一种新的学习范式，通过给予解释和批评者的反馈来改进学习者的性能。这种方法适用于图像分类等基本任务，并有潜力在人工智能研究中发挥作用。

    

    人工智能研究长期以来一直从生物学中寻找灵感，特别是人类智能。与目前主要将解释视为模型检查手段的人工智能研究相比，从心理学中发现自我解释在代理学习过程中的好处有些被忽视了。受到这个启发，我们引入了一种新的学习范式，称为学习通过自我解释 (LSX)。其中的基本思想是，一个学习模块 (学习者) 执行一个基本任务，比如图像分类，并对其决策进行解释。随后，一个内部批评者模块基于原始任务评估这些解释的质量。最后，学习者通过批评者的反馈得到改进，并根据需要重复这个循环。背后的直觉是，如果批评者能够根据相应的解释执行相同的任务，则该解释被认为是“好”的。尽管有许多实现可能性，但本文旨在提供关于实施学习通过自我解释的一般指导原则。有待进一步的研究和实践来探索这一学习范式的潜力。

    Artificial intelligence (AI) research has a long track record of drawing inspirations from findings from biology, in particular human intelligence. In contrast to current AI research that mainly treats explanations as a means for model inspection, a somewhat neglected finding from human psychology is the benefit of self-explaining in an agents' learning process. Motivated by this, we introduce a novel learning paradigm, termed Learning by Self-Explaining (LSX). The underlying idea is that a learning module (learner) performs a base task, e.g. image classification, and provides explanations to its decisions. An internal critic module next evaluates the quality of these explanations given the original task. Finally, the learner is refined with the critic's feedback and the loop is repeated as required. The intuition behind this is that an explanation is considered "good" if the critic can perform the same task given the respective explanation. Despite many implementation possibilities th
    
[^33]: 使用可微间接引用的高效图形表示

    Efficient Graphics Representation with Differentiable Indirection. (arXiv:2309.08387v1 [cs.GR])

    [http://arxiv.org/abs/2309.08387](http://arxiv.org/abs/2309.08387)

    本论文介绍了一种新的学习原语，使用可微分的多尺度查找表作为图形管线中传统计算和数据操作的有效替代方法。它在多个图形任务中展现了灵活性和高效性。

    

    我们引入了可微间接引用——一种新颖的学习原语，它使用可微分的多尺度查找表作为图形管线中传统计算和数据操作的有效替代方法。我们在几个图形任务中展示了其灵活性，包括几何和图像表示、纹理映射、着色和辐射场表示。在所有情况下，可微间接引用可以无缝集成到现有架构中，快速训练，并产生多样且高效的结果。

    We introduce differentiable indirection -- a novel learned primitive that employs differentiable multi-scale lookup tables as an effective substitute for traditional compute and data operations across the graphics pipeline. We demonstrate its flexibility on a number of graphics tasks, i.e., geometric and image representation, texture mapping, shading, and radiance field representation. In all cases, differentiable indirection seamlessly integrates into existing architectures, trains rapidly, and yields both versatile and efficient results.
    
[^34]: 《张量超图神经网络与信号去噪之间的统一视角》

    A Unified View Between Tensor Hypergraph Neural Networks And Signal Denoising. (arXiv:2309.08385v1 [cs.LG])

    [http://arxiv.org/abs/2309.08385](http://arxiv.org/abs/2309.08385)

    这篇论文提出了一种统一视角，将张量超图神经网络（HyperGNNs）和超图信号去噪（HyperGSD）联系起来，并通过设计了张量超图迭代网络（T-HGIN）来应用于信号去噪问题。实验结果显示了该方法的潜在应用价值。

    

    张量超图神经网络（HyperGNNs）和超图信号去噪（HyperGSD）是高阶网络建模中的两个基本主题。了解这两个领域之间的联系对于从HyperGSD的角度设计新的HyperGNNs以及反之亦然非常有用。特别地，张量超图卷积网络（T-HGCN）已经成为在超图上保留高阶交互的强大结构，并且这项工作展示了HyperGSD问题与T-HGCN之间的等价关系。受到这一有趣结果的启发，我们进一步设计了一种基于HyperGSD问题的张量超图迭代网络（T-HGIN），它在每个单层中利用了多步更新方案的优势。数值实验被进行以展示所提出的T-HGIN方法的有希望的应用。

    Hypergraph Neural networks (HyperGNNs) and hypergraph signal denoising (HyperGSD) are two fundamental topics in higher-order network modeling. Understanding the connection between these two domains is particularly useful for designing novel HyperGNNs from a HyperGSD perspective, and vice versa. In particular, the tensor-hypergraph convolutional network (T-HGCN) has emerged as a powerful architecture for preserving higher-order interactions on hypergraphs, and this work shows an equivalence relation between a HyperGSD problem and the T-HGCN. Inspired by this intriguing result, we further design a tensor-hypergraph iterative network (T-HGIN) based on the HyperGSD problem, which takes advantage of a multi-step updating scheme in every single layer. Numerical experiments are conducted to show the promising applications of the proposed T-HGIN approach.
    
[^35]: 自适应优先级重新加权以提高公平性泛化能力

    Adaptive Priority Reweighing for Generalizing Fairness Improvement. (arXiv:2309.08375v1 [cs.LG])

    [http://arxiv.org/abs/2309.08375](http://arxiv.org/abs/2309.08375)

    本文提出了一种新颖的自适应重新加权方法，通过优先考虑靠近决策边界的样本并分配较高的权重，提高了公平分类器的泛化能力。

    

    随着机器学习应用在关键决策领域的普及，对算法公平性的呼声越来越大。尽管已经通过学习公平约束来改善算法的公平性的各种方式，但它们在测试集上的性能并不能很好地推广。需要一种性能有前景且具有更好泛化能力的公平算法。本文提出了一种新颖的自适应重新加权方法，以消除训练数据和测试数据之间分布偏移对模型泛化能力的影响。大多数先前的重新加权方法提议为每个（子）组分配一个统一的权重。相反，我们的方法细粒度地建模了样本预测与决策边界的距离。我们的自适应重新加权方法优先考虑靠近决策边界的样本，并分配较高的权重来提高公平分类器的泛化能力。进行了大量实验验证了其泛化能力。

    With the increasing penetration of machine learning applications in critical decision-making areas, calls for algorithmic fairness are more prominent. Although there have been various modalities to improve algorithmic fairness through learning with fairness constraints, their performance does not generalize well in the test set. A performance-promising fair algorithm with better generalizability is needed. This paper proposes a novel adaptive reweighing method to eliminate the impact of the distribution shifts between training and test data on model generalizability. Most previous reweighing methods propose to assign a unified weight for each (sub)group. Rather, our method granularly models the distance from the sample predictions to the decision boundary. Our adaptive reweighing method prioritizes samples closer to the decision boundary and assigns a higher weight to improve the generalizability of fair classifiers. Extensive experiments are performed to validate the generalizability 
    
[^36]: 理解自监督学习在表格异常检测中的限制

    Understanding the limitations of self-supervised learning for tabular anomaly detection. (arXiv:2309.08374v1 [cs.LG])

    [http://arxiv.org/abs/2309.08374](http://arxiv.org/abs/2309.08374)

    本研究探讨了自监督学习在表格异常检测中的限制。通过多个实验发现，自监督学习得到的表征并不能提高表格异常检测的性能，这是由于神经网络引入了无关的特征。然而，使用神经网络表示的子空间可以恢复性能。

    

    尽管自监督学习已经改进了计算机视觉和自然语言处理中的异常检测，但表格数据是否可以从中受益尚不清楚。本文探讨了自监督学习在表格异常检测中的限制。我们在26个基准数据集上进行了多个实验，涉及各种预训练任务，以了解这种情况的原因。我们的结果证实，与使用原始数据表示相比，通过自监督学习得到的表征并不能提高表格异常检测的性能。我们展示了这是由于神经网络引入了无关的特征，从而降低了异常检测器的有效性。然而，我们证明了使用神经网络表示的子空间可以恢复性能。

    While self-supervised learning has improved anomaly detection in computer vision and natural language processing, it is unclear whether tabular data can benefit from it. This paper explores the limitations of self-supervision for tabular anomaly detection. We conduct several experiments spanning various pretext tasks on 26 benchmark datasets to understand why this is the case. Our results confirm representations derived from self-supervision do not improve tabular anomaly detection performance compared to using the raw representations of the data. We show this is due to neural networks introducing irrelevant features, which reduces the effectiveness of anomaly detectors. However, we demonstrate that using a subspace of the neural network's representation can recover performance.
    
[^37]: 深度流正则化判别分析下的持续学习

    Continual Learning with Deep Streaming Regularized Discriminant Analysis. (arXiv:2309.08353v1 [cs.CV])

    [http://arxiv.org/abs/2309.08353](http://arxiv.org/abs/2309.08353)

    本文提出了深度流正则化判别分析下的持续学习方法，能够有效解决使用非同分布数据增量更新模型导致的灾难性遗忘问题。在ImageNet数据集上实验证明，该方法优于批量学习和现有的流式学习算法。

    

    持续学习在现实世界的机器学习应用中越来越受到追捧，因为它能够以更接近人类的方式进行学习。传统的机器学习方法无法实现这一点，因为用非同分布的数据增量更新模型会导致灾难性遗忘，覆盖了现有的表示。尽管传统的持续学习方法主要集中在批量学习上，即按顺序从大规模标记数据中进行学习，但这种方法不适用于我们希望直接集成新数据的实际应用。这就需要对流式学习进行范式转变。本文提出了一个流式版本的正则化判别分析作为解决这一挑战的方法。我们将算法与卷积神经网络相结合，并证明在ImageNet上优于批量学习和现有的流式学习算法。

    Continual learning is increasingly sought after in real world machine learning applications, as it enables learning in a more human-like manner. Conventional machine learning approaches fail to achieve this, as incrementally updating the model with non-identically distributed data leads to catastrophic forgetting, where existing representations are overwritten. Although traditional continual learning methods have mostly focused on batch learning, which involves learning from large collections of labeled data sequentially, this approach is not well-suited for real-world applications where we would like new data to be integrated directly. This necessitates a paradigm shift towards streaming learning. In this paper, we propose a streaming version of regularized discriminant analysis as a solution to this challenge. We combine our algorithm with a convolutional neural network and demonstrate that it outperforms both batch learning and existing streaming learning algorithms on the ImageNet 
    
[^38]: ADAM在非凸设置中具有恒定步长的收敛性：一个简单的证明

    Convergence of ADAM with Constant Step Size in Non-Convex Settings: A Simple Proof. (arXiv:2309.08339v1 [cs.LG])

    [http://arxiv.org/abs/2309.08339](http://arxiv.org/abs/2309.08339)

    本文分析了ADAM在非凸设置中具有恒定步长的收敛性，给出了步长达到几乎肯定渐近收敛的充分条件，并提供了确定性ADAM在处理平滑非凸函数时达到近似临界性所需的运行时间界限。

    

    在神经网络训练中，RMSProp和ADAM仍然是广泛使用的优化算法。它们的性能关键之一在于选择适当的步长，这会显著影响它们的有效性。值得注意的是，这些算法的性能可以因选择的步长而变化很大。此外，关于它们的理论收敛性问题仍然是一个感兴趣的话题。在本文中，我们在非凸设置中对ADAM的恒定步长版本进行了理论分析。我们证明了步长达到几乎肯定渐近收敛到零的充分条件，而只需最小的假设。我们还给出了确定性ADAM在处理平滑非凸函数时达到近似临界性所需的运行时间界限。

    In neural network training, RMSProp and ADAM remain widely favoured optimization algorithms. One of the keys to their performance lies in selecting the correct step size, which can significantly influence their effectiveness. It is worth noting that these algorithms performance can vary considerably, depending on the chosen step sizes. Additionally, questions about their theoretical convergence properties continue to be a subject of interest. In this paper, we theoretically analyze a constant stepsize version of ADAM in the non-convex setting. We show sufficient conditions for the stepsize to achieve almost sure asymptotic convergence of the gradients to zero with minimal assumptions. We also provide runtime bounds for deterministic ADAM to reach approximate criticality when working with smooth, non-convex functions.
    
[^39]: 让我们预测谁会换工作

    Let's Predict Who Will Move to a New Job. (arXiv:2309.08333v1 [cs.LG])

    [http://arxiv.org/abs/2309.08333](http://arxiv.org/abs/2309.08333)

    本文讨论了如何使用机器学习来预测谁会换工作，包括数据预处理和使用多种ML算法。为了提高性能，使用了合成少数过采样技术。评估模型时使用了精度、召回率、F1-Score和准确率等指标。

    

    任何一家公司的人力资源部门都面临着预测申请人是否会寻找新工作或者留在公司的挑战。在本文中，我们讨论了如何使用机器学习（ML）来预测谁会换工作。首先，将数据预处理成适合ML模型的格式。为了处理分类特征，应用数据编码并执行几种ML算法，包括随机森林（RF）、逻辑回归（LR）、决策树（DT）和极限梯度提升（XGBoost）。为了提高ML模型的性能，使用合成少数过采样技术（SMOTE）进行保留。使用精度、召回率、F1-Score和准确率等决策支持度量来评估模型。

    Any company's human resources department faces the challenge of predicting whether an applicant will search for a new job or stay with the company. In this paper, we discuss how machine learning (ML) is used to predict who will move to a new job. First, the data is pre-processed into a suitable format for ML models. To deal with categorical features, data encoding is applied and several MLA (ML Algorithms) are performed including Random Forest (RF), Logistic Regression (LR), Decision Tree (DT), and eXtreme Gradient Boosting (XGBoost). To improve the performance of ML models, the synthetic minority oversampling technique (SMOTE) is used to retain them. Models are assessed using decision support metrics such as precision, recall, F1-Score, and accuracy.
    
[^40]: 不确定情况下反事实干预的估计

    Estimation of Counterfactual Interventions under Uncertainties. (arXiv:2309.08332v1 [cs.LG])

    [http://arxiv.org/abs/2309.08332](http://arxiv.org/abs/2309.08332)

    本论文提出了一种通过采用层次贝叶斯方法来解决连续情况下反事实干预估计中的不确定性的方法。通过推导贝叶斯变形高斯过程的反事实分布，实现了对非高斯分布和非可加情况的建模。

    

    反事实分析是人类每天直观进行的活动，例如“我应该怎么做才能使贷款获得批准？”这样的反事实问题也引导了科学假设的制定。更正式地说，通过推断对系统行为的过去观察的假设干预的效果，它提供了关于系统潜在改进的见解，这在各种工业应用中起着重要作用。由于这种分析的假设性质，反事实分布本质上是模棱两可的。这种模棱两可在连续设置中尤其具有挑战性，因为对于相同观察存在许多解释。在本文中，我们通过采用一种层次贝叶斯方法来解决这个问题，该方法明确地模拟了这种不确定性。具体而言，我们推导了贝叶斯变形高斯过程的反事实分布，从而允许非高斯分布和非可加的情况。

    Counterfactual analysis is intuitively performed by humans on a daily basis eg. "What should I have done differently to get the loan approved?". Such counterfactual questions also steer the formulation of scientific hypotheses. More formally it provides insights about potential improvements of a system by inferring the effects of hypothetical interventions into a past observation of the system's behaviour which plays a prominent role in a variety of industrial applications. Due to the hypothetical nature of such analysis, counterfactual distributions are inherently ambiguous. This ambiguity is particularly challenging in continuous settings in which a continuum of explanations exist for the same observation. In this paper, we address this problem by following a hierarchical Bayesian approach which explicitly models such uncertainty. In particular, we derive counterfactual distributions for a Bayesian Warped Gaussian Process thereby allowing for non-Gaussian distributions and non-additi
    
[^41]: 异方差拟合置信回归

    Heteroskedastic conformal regression. (arXiv:2309.08313v1 [stat.ML])

    [http://arxiv.org/abs/2309.08313](http://arxiv.org/abs/2309.08313)

    本文研究了使用标准化和Mondrian符合规范的方法如何构建自适应的预测区间，以解决回归问题中的异方差噪声。

    

    符合规范的预测以及特定的拆分符合规范的预测提供了一种无分布的方法来估计具有统计保证的预测区间。最近的研究表明，当专注于边际覆盖时，即在校准数据集上，该方法产生的预测区间平均包含预定义覆盖水平的真实值，拆分符合规范的预测可以产生最先进的预测区间。然而，这样的区间通常不是自适应的，这对于具有异方差噪声的回归问题可能是有问题的。本文试图阐明如何使用标准化和Mondrian符合规范的方法来构建自适应的预测区间。我们以系统的方式提出理论和实验结果来研究这些方法。

    Conformal prediction, and split conformal prediction as a specific implementation, offer a distribution-free approach to estimating prediction intervals with statistical guarantees. Recent work has shown that split conformal prediction can produce state-of-the-art prediction intervals when focusing on marginal coverage, i.e., on a calibration dataset the method produces on average prediction intervals that contain the ground truth with a predefined coverage level. However, such intervals are often not adaptive, which can be problematic for regression problems with heteroskedastic noise. This paper tries to shed new light on how adaptive prediction intervals can be constructed using methods such as normalized and Mondrian conformal prediction. We present theoretical and experimental results in which these methods are investigated in a systematic way.
    
[^42]: 一个实时活动说话人检测系统，将音频-视觉信号与空间查询机制整合在一起

    A Real-Time Active Speaker Detection System Integrating an Audio-Visual Signal with a Spatial Querying Mechanism. (arXiv:2309.08295v1 [eess.AS])

    [http://arxiv.org/abs/2309.08295](http://arxiv.org/abs/2309.08295)

    本文介绍了一个实时活动说话人检测系统，通过将音频-视觉信号与空间查询机制整合，利用低功耗边缘计算实现。该系统具有优雅的退化性能，能够在计算预算耗尽的情况下仍然有效运行，并在真实会议数据集上表现出良好的性能。

    

    我们介绍了一种独特的实时、因果关系的基于神经网络的活动说话人检测系统，经过低功耗边缘计算优化。该系统驱动一个虚拟影视模块，并且部署在商业设备上。该系统使用来自麦克风阵列和360度相机的数据。我们的网络每个参与者只需要127MFLOPs，对于一个有14个参与者的会议。与以前的工作不同，当计算预算耗尽时，我们检查了我们的网络的错误率，并发现它表现出了优雅的退化，即使在这种情况下，系统仍然能够运行得相当好。与传统的方向估计方法不同，我们的网络学习查询可用的声学数据，并考虑到检测到的头部位置。我们在一个包含最多14个参与者、重叠的语音和其他挑战性场景的真实会议数据集上训练和评估我们的算法。

    We introduce a distinctive real-time, causal, neural network-based active speaker detection system optimized for low-power edge computing. This system drives a virtual cinematography module and is deployed on a commercial device. The system uses data originating from a microphone array and a 360-degree camera. Our network requires only 127 MFLOPs per participant, for a meeting with 14 participants. Unlike previous work, we examine the error rate of our network when the computational budget is exhausted, and find that it exhibits graceful degradation, allowing the system to operate reasonably well even in this case. Departing from conventional DOA estimation approaches, our network learns to query the available acoustic data, considering the detected head locations. We train and evaluate our algorithm on a realistic meetings dataset featuring up to 14 participants in the same meeting, overlapped speech, and other challenging scenarios.
    
[^43]: 利用点扩散模型对大肠的3D形状进行精化以生成数字幻影

    Large Intestine 3D Shape Refinement Using Point Diffusion Models for Digital Phantom Generation. (arXiv:2309.08289v1 [cs.CV])

    [http://arxiv.org/abs/2309.08289](http://arxiv.org/abs/2309.08289)

    本研究利用几何深度学习和去噪扩散概率模型优化大肠的分割结果，并结合先进的表面重构模型，实现对大肠3D形状的精化恢复。

    

    准确建模人体器官在构建虚拟成像试验的计算仿真中起着至关重要的作用。然而，从计算机断层扫描中生成解剖学上可信的器官表面重建仍然对人体结构中的许多器官来说是个挑战。在处理大肠时，这个挑战尤为明显。在这项研究中，我们利用几何深度学习和去噪扩散概率模型的最新进展来优化大肠分割结果。首先，我们将器官表示为从3D分割掩模表面采样得到的点云。随后，我们使用分层变分自编码器获得器官形状的全局和局部潜在表示。我们在分层潜在空间中训练两个条件去噪扩散模型来进行形状精化。为了进一步提高我们的方法，我们还结合了一种先进的表面重构模型，从而实现形状的更好恢复。

    Accurate 3D modeling of human organs plays a crucial role in building computational phantoms for virtual imaging trials. However, generating anatomically plausible reconstructions of organ surfaces from computed tomography scans remains challenging for many structures in the human body. This challenge is particularly evident when dealing with the large intestine. In this study, we leverage recent advancements in geometric deep learning and denoising diffusion probabilistic models to refine the segmentation results of the large intestine. We begin by representing the organ as point clouds sampled from the surface of the 3D segmentation mask. Subsequently, we employ a hierarchical variational autoencoder to obtain global and local latent representations of the organ's shape. We train two conditional denoising diffusion models in the hierarchical latent space to perform shape refinement. To further enhance our method, we incorporate a state-of-the-art surface reconstruction model, allowin
    
[^44]: 无需采样的概率深度状态空间模型

    Sampling-Free Probabilistic Deep State-Space Models. (arXiv:2309.08256v1 [cs.LG])

    [http://arxiv.org/abs/2309.08256](http://arxiv.org/abs/2309.08256)

    本文提出了一种无需采样的概率深度状态空间模型，通过使用第一个确定性推断算法，实现了高效的训练和测试近似。

    

    很多现实世界中的动态系统可以用状态空间模型（SSM）来描述。在这种表述中，每个观察值都由一个潜在状态发射，该状态遵循一阶马尔可夫动力学。概率深度状态空间模型（ProDSSM）将这一框架推广到未知参数形式的动态系统中，其中过渡模型和发射模型由具有不确定权重的神经网络描述。本文提出了针对这类模型的第一个确定性推断算法。我们的框架可以进行高效的训练和测试近似。在实验中，我们证明我们的新方法可以用于各种任务，并在预测性能和计算预算之间取得了卓越的平衡。

    Many real-world dynamical systems can be described as State-Space Models (SSMs). In this formulation, each observation is emitted by a latent state, which follows first-order Markovian dynamics. A Probabilistic Deep SSM (ProDSSM) generalizes this framework to dynamical systems of unknown parametric form, where the transition and emission models are described by neural networks with uncertain weights. In this work, we propose the first deterministic inference algorithm for models of this type. Our framework allows efficient approximations for training and testing. We demonstrate in our experiments that our new method can be employed for a variety of tasks and enjoys a superior balance between predictive performance and computational budget.
    
[^45]: 通过基于流的语音转换实现跨语言知识蒸馏，用于鲁棒的多语种语音合成

    Cross-lingual Knowledge Distillation via Flow-based Voice Conversion for Robust Polyglot Text-To-Speech. (arXiv:2309.08255v1 [eess.AS])

    [http://arxiv.org/abs/2309.08255](http://arxiv.org/abs/2309.08255)

    本文提出了一个跨语言语音合成的框架，使用语音转换和文本到语音模型，优于基于多语种模型的最先进方法，特别适用于资源匮乏的情况。

    

    在这项工作中，我们引入了一个跨语言语音合成的框架，其中包括一个上游语音转换（VC）模型和一个下游文本到语音（TTS）模型。我们的框架包括4个阶段。在前两个阶段中，我们使用VC模型将目标区域的话语转换为目标说话者的声音。在第三个阶段，将转换后的数据与目标语言录音中的语言特征和持续时间结合起来，然后用于训练一个单说话人声学模型。最后，最后一个阶段将训练一个与语言无关的声码器。我们的评估结果显示，这种提出的范例优于基于训练大型多语种TTS模型的最先进方法。此外，我们的实验证明了我们的方法在不同的模型架构、语言、说话者和数据量方面的鲁棒性。此外，我们的解决方案在资源匮乏的情况下特别有益。

    In this work, we introduce a framework for cross-lingual speech synthesis, which involves an upstream Voice Conversion (VC) model and a downstream Text-To-Speech (TTS) model. The proposed framework consists of 4 stages. In the first two stages, we use a VC model to convert utterances in the target locale to the voice of the target speaker. In the third stage, the converted data is combined with the linguistic features and durations from recordings in the target language, which are then used to train a single-speaker acoustic model. Finally, the last stage entails the training of a locale-independent vocoder. Our evaluations show that the proposed paradigm outperforms state-of-the-art approaches which are based on training a large multilingual TTS model. In addition, our experiments demonstrate the robustness of our approach with different model architectures, languages, speakers and amounts of data. Moreover, our solution is especially beneficial in low-resource settings.
    
[^46]: 自主驾驶车辆的强化学习策略的定量和定性评估

    Quantitative and Qualitative Evaluation of Reinforcement Learning Policies for Autonomous Vehicles. (arXiv:2309.08254v1 [cs.AI])

    [http://arxiv.org/abs/2309.08254](http://arxiv.org/abs/2309.08254)

    本文使用强化学习算法（PPO）针对自主驾驶车辆的选择进行了优化，通过最小化时间和污染来缓解交通阻塞问题，经实证分析和定性评估证明了方法的有效性和实用性。

    

    在不断变化的交通环境中优化交通动力学非常重要，特别是在自动驾驶车辆（AVs）与人驾驶车辆并存的情况下。本文提出了一种使用近端策略优化（PPO）强化学习算法来优化AVs选择的新方法。我们通过学习一种策略来最小化交通阻塞（即最小化横过米兰的环形道的时间）并减少污染。通过经验分析，我们证明了我们的方法可以减少时间和污染水平。此外，我们使用先进的驾驶舱定性评估了学到的策略，以评估其在接近真实世界条件下的性能。为了评估策略的实用性和可接受性，我们通过模拟器进行了人类参与者的评估，重点关注交通平稳性和安全感等一系列指标。总的来说，我们的研究结果表明，人驾驶车辆的感知和行车平滑性方面，我们的方法非常实用。

    Optimizing traffic dynamics in an evolving transportation landscape is crucial, particularly in scenarios where autonomous vehicles (AVs) with varying levels of autonomy coexist with human-driven cars. This paper presents a novel approach to optimizing choices of AVs using Proximal Policy Optimization (PPO), a reinforcement learning algorithm. We learned a policy to minimize traffic jams (i.e., minimize the time to cross the scenario) and to minimize pollution in a roundabout in Milan, Italy. Through empirical analysis, we demonstrate that our approach can reduce time and pollution levels. Furthermore, we qualitatively evaluate the learned policy using a cutting-edge cockpit to assess its performance in near-real-world conditions. To gauge the practicality and acceptability of the policy, we conducted evaluations with human participants using the simulator, focusing on a range of metrics like traffic smoothness and safety perception. In general, our findings show that human-driven vehi
    
[^47]: 带有Beta散度的深度非负矩阵分解

    Deep Nonnegative Matrix Factorization with Beta Divergences. (arXiv:2309.08249v1 [cs.LG])

    [http://arxiv.org/abs/2309.08249](http://arxiv.org/abs/2309.08249)

    本文提出了一种使用Beta散度的深度非负矩阵分解方法，应用于面部特征提取、文档主题识别和高光谱图像材料识别。

    

    深度非负矩阵分解（deep NMF）最近成为一种有价值的技术，用于在不同尺度上提取多层特征。然而，所有现有的深度NMF模型和算法主要都以最小二乘误差为评估标准，这可能不是评估多样化数据集近似质量的最合适指标。例如，当处理音频信号和文档等数据类型时，广泛认可的是$\beta$-divergences提供了更适合的替代方案。本文基于$\beta$-divergences开发了新的深度NMF模型和算法，并将这些技术应用于面部特征提取、文档集合中的主题识别以及高光谱图像中材料的识别。

    Deep Nonnegative Matrix Factorization (deep NMF) has recently emerged as a valuable technique for extracting multiple layers of features across different scales. However, all existing deep NMF models and algorithms have primarily centered their evaluation on the least squares error, which may not be the most appropriate metric for assessing the quality of approximations on diverse datasets. For instance, when dealing with data types such as audio signals and documents, it is widely acknowledged that $\beta$-divergences offer a more suitable alternative. In this paper, we develop new models and algorithms for deep NMF using $\beta$-divergences. Subsequently, we apply these techniques to the extraction of facial features, the identification of topics within document collections, and the identification of materials within hyperspectral images.
    
[^48]: 对自编码器的几何角度的研究

    A Geometric Perspective on Autoencoders. (arXiv:2309.08247v1 [cs.LG])

    [http://arxiv.org/abs/2309.08247](http://arxiv.org/abs/2309.08247)

    本文从几何角度研究了自编码器框架，并提出了解决多解和畸变表示问题的几何方法。

    

    本文提出了自编码器框架的几何方面，尽管其重要性，但被相对较少地认识到。给定一组几乎位于某个较低维度流形上的高维数据点，自编码器同时学习流形和其坐标图。这种几何角度自然引发了一些问题，比如“有限的数据点对应于单一的流形吗？”或者“只有一个坐标图可以表示流形吗？”对这些问题的回答是否定的，这意味着给定一个数据集，有多个解的自编码器。因此，它们有时会产生具有严重畸变的潜在空间表示的错误流形。在本文中，我们介绍了解决这些问题的最近的几何方法。

    This paper presents the geometric aspect of the autoencoder framework, which, despite its importance, has been relatively less recognized. Given a set of high-dimensional data points that approximately lie on some lower-dimensional manifold, an autoencoder learns the \textit{manifold} and its \textit{coordinate chart}, simultaneously. This geometric perspective naturally raises inquiries like "Does a finite set of data points correspond to a single manifold?" or "Is there only one coordinate chart that can represent the manifold?". The responses to these questions are negative, implying that there are multiple solution autoencoders given a dataset. Consequently, they sometimes produce incorrect manifolds with severely distorted latent space representations. In this paper, we introduce recent geometric approaches that address these issues.
    
[^49]: 基于持续同调的拓扑Node2vec：增强图嵌入方法

    Topological Node2vec: Enhanced Graph Embedding via Persistent Homology. (arXiv:2309.08241v1 [stat.ML])

    [http://arxiv.org/abs/2309.08241](http://arxiv.org/abs/2309.08241)

    通过引入拓扑损失项和适应持续同调度量的熵正则化，我们改进了Node2vec方法，使其能够更好地还原输入图的拓扑结构。

    

    Node2vec是一种图嵌入方法，它学习了加权图每个节点的向量表示，同时尽力保持节点之间的相对距离和全局结构。数值实验表明，Node2vec难以再现输入图的拓扑结构。为了解决这个问题，我们在Node2vec的训练损失中引入了一个拓扑损失项，该损失项试图将生成的嵌入的持续同调图与输入图的持续同调图尽可能地对齐。我们根据计算优化传输中的结果，精心调整了熵正则化的持续同调度量，使我们能够以可微分的方式衡量持续同调图之间的差异。通过梯度下降最小化我们修改后的损失函数，可以重建输入图的几何和拓扑结构。我们使用一些示例合成图展示了这种方法的好处。

    Node2vec is a graph embedding method that learns a vector representation for each node of a weighted graph while seeking to preserve relative proximity and global structure. Numerical experiments suggest Node2vec struggles to recreate the topology of the input graph. To resolve this we introduce a topological loss term to be added to the training loss of Node2vec which tries to align the persistence diagram (PD) of the resulting embedding as closely as possible to that of the input graph. Following results in computational optimal transport, we carefully adapt entropic regularization to PD metrics, allowing us to measure the discrepancy between PDs in a differentiable way. Our modified loss function can then be minimized through gradient descent to reconstruct both the geometry and the topology of the input graph. We showcase the benefits of this approach using demonstrative synthetic examples.
    
[^50]: 在高斯－勒让德节点上由于潜在空间正则化而压缩中保持拓扑数据结构完整性的确保

    Ensuring Toplogical Data-Structure Preservation under Autoencoder Compression due to Latent Space Regularization in Gauss--Legendre nodes. (arXiv:2309.08228v1 [cs.LG])

    [http://arxiv.org/abs/2309.08228](http://arxiv.org/abs/2309.08228)

    通过在高斯-勒让德节点上进行潜在空间正则化，我们的研究提出了一种新的无监督自编码器，能够确保在压缩过程中保持拓扑数据结构的完整性。

    

    我们为一般的无监督自编码器制定了一个数据无关的潜在空间正则化约束。该正则化基于在勒让德节点上对自编码器的雅可比矩阵进行采样，这些节点是高斯-勒让德积分的中心。重新审视这个经典问题能够证明，经过正则化的自编码器能够将初始数据流形一对一地重新嵌入到其潜在表示中。实验证明，之前提出的正则化策略（如收缩自编码）在简单示例中已经导致了拓扑缺陷，而基于卷积的（变分）自编码器也是如此。相比之下，通过我们的贡献，标准的多层感知器神经网络在正则化的情况下已经确保了拓扑完整性。这个观察结果适用于经典的FashionMNIST数据集以及MRI脑部扫描的真实世界编码问题，这表明在各个领域中，对于复杂的高维数据，可靠的低维表示已得以确保。

    We formulate a data independent latent space regularisation constraint for general unsupervised autoencoders. The regularisation rests on sampling the autoencoder Jacobian in Legendre nodes, being the centre of the Gauss-Legendre quadrature. Revisiting this classic enables to prove that regularised autoencoders ensure a one-to-one re-embedding of the initial data manifold to its latent representation. Demonstrations show that prior proposed regularisation strategies, such as contractive autoencoding, cause topological defects already for simple examples, and so do convolutional based (variational) autoencoders. In contrast, topological preservation is ensured already by standard multilayer perceptron neural networks when being regularised due to our contribution. This observation extends through the classic FashionMNIST dataset up to real world encoding problems for MRI brain scans, suggesting that, across disciplines, reliable low dimensional representations of complex high-dimensiona
    
[^51]: VERSE：具有实时推理能力的虚拟梯度感知流转学习

    VERSE: Virtual-Gradient Aware Streaming Lifelong Learning with Anytime Inference. (arXiv:2309.08227v1 [cs.LG])

    [http://arxiv.org/abs/2309.08227](http://arxiv.org/abs/2309.08227)

    这项研究提出了一种具有实时推理能力的流式终身学习方法，采用虚拟梯度进行连续表示学习，借助语义记忆来抑制灾难性遗忘，并在多样化的数据上进行了广泛实验。

    

    终身学习是指在训练AI代理的同时，防止其遗忘以前获得的知识的问题。现有的方法大多关注在静态环境下的终身学习，并且缺乏在快速变化的动态环境中减轻遗忘的能力。流式终身学习是终身学习中一个具有挑战性的设置，其目标是在动态的非平稳环境中进行连续学习而不遗忘。我们引入一种新颖的终身学习方法，该方法是流式的，仅需要对数据进行一次遍历，可以以类增量的方式学习，并且可以进行即时评估（实时推理）。为了实现这些，我们提出了用于连续表示学习的虚拟梯度，以防止灾难性遗忘，并借助基于指数移动平均的语义记忆进一步提高性能。我们在多样化的数据上进行了广泛的实验。

    Lifelong learning, also referred to as continual learning, is the problem of training an AI agent continuously while also preventing it from forgetting its previously acquired knowledge. Most of the existing methods primarily focus on lifelong learning within a static environment and lack the ability to mitigate forgetting in a quickly-changing dynamic environment. Streaming lifelong learning is a challenging setting of lifelong learning with the goal of continuous learning in a dynamic non-stationary environment without forgetting. We introduce a novel approach to lifelong learning, which is streaming, requires a single pass over the data, can learn in a class-incremental manner, and can be evaluated on-the-fly (anytime inference). To accomplish these, we propose virtual gradients for continual representation learning to prevent catastrophic forgetting and leverage an exponential-moving-average-based semantic memory to further enhance performance. Extensive experiments on diverse data
    
[^52]: 弱监督学习的统一风险分析

    Unified Risk Analysis for Weakly Supervised Learning. (arXiv:2309.08216v1 [cs.LG])

    [http://arxiv.org/abs/2309.08216](http://arxiv.org/abs/2309.08216)

    本文提出了一个框架，为弱监督学习提供了全面的理解和统一的方法论。该框架的表达部分提供了对弱监督形成的统一解释，并包含15种现有的弱监督设置；引导生成的减少图在弱监督学习中提供了全面的连接。该框架的分析部分提供了一种系统的进行风险重写的方法，从而提供了一种新的去污分布策略。

    

    在弱监督学习的繁荣研究中，我们认识到弱监督情景背后机制的统一解释缺失，更不用说对于经验风险最小化方法中关键的风险重写问题的系统处理了。在本文中，我们引入了一个框架，为弱监督学习提供了全面的理解和统一的方法论。该框架的表达部分利用了一个污染视角，提供了弱监督形成的统一解释，并包含了15种现有的弱监督设置。引导生成的减少图在弱监督学习中提供了全面的连接。该框架的分析部分被视为一个去污过程，提供了一种系统的进行风险重写的方法。除了传统的逆矩阵方法，我们还设计了一种称为边际链的新策略，旨在去污分布。我们证明了该方法的可行性。

    Among the flourishing research of weakly supervised learning (WSL), we recognize the lack of a unified interpretation of the mechanism behind the weakly supervised scenarios, let alone a systematic treatment of the risk rewrite problem, a crucial step in the empirical risk minimization approach. In this paper, we introduce a framework providing a comprehensive understanding and a unified methodology for WSL. The formulation component of the framework, leveraging a contamination perspective, provides a unified interpretation of how weak supervision is formed and subsumes fifteen existing WSL settings. The induced reduction graphs offer comprehensive connections over WSLs. The analysis component of the framework, viewed as a decontamination process, provides a systematic method of conducting risk rewrite. In addition to the conventional inverse matrix approach, we devise a novel strategy called marginal chain aiming to decontaminate distributions. We justify the feasibility of the propos
    
[^53]: HM-Conformer: 一种基于Conformer的音频深度伪造检测系统，具有分层汇聚和多级分类令牌聚合方法。

    HM-Conformer: A Conformer-based audio deepfake detection system with hierarchical pooling and multi-level classification token aggregation methods. (arXiv:2309.08208v1 [cs.SD])

    [http://arxiv.org/abs/2309.08208](http://arxiv.org/abs/2309.08208)

    HM-Conformer是一种音频深度伪造检测系统，利用分层汇聚和多级分类令牌聚合方法，能够有效地捕捉并检测音频深度伪造的欺骗证据。

    

    音频深度伪造检测（ADD）是检测由文本到语音或语音转换系统生成的欺骗攻击的任务。用于区分伪造和真实话语的欺骗证据可能存在于输入特征的局部或全局。为了捕捉这些证据，Conformer结合了Transformer和CNN，具有适合的结构。然而，由于Conformer是为序列到序列任务而设计的，直接应用于ADD任务可能不是最优的。为了解决这个限制，我们提出了HM-Conformer，采用了两个组件：（1）分层汇聚方法，逐步减少序列长度以消除重复信息；（2）多级分类令牌聚合方法，利用分类令牌从不同的块中收集信息。由于这些组件的存在，HM-Conformer可以通过处理各种序列长度并聚合它们来高效地检测到欺骗证据。

    Audio deepfake detection (ADD) is the task of detecting spoofing attacks generated by text-to-speech or voice conversion systems. Spoofing evidence, which helps to distinguish between spoofed and bona-fide utterances, might exist either locally or globally in the input features. To capture these, the Conformer, which consists of Transformers and CNN, possesses a suitable structure. However, since the Conformer was designed for sequence-to-sequence tasks, its direct application to ADD tasks may be sub-optimal. To tackle this limitation, we propose HM-Conformer by adopting two components: (1) Hierarchical pooling method progressively reducing the sequence length to eliminate duplicated information (2) Multi-level classification token aggregation method utilizing classification tokens to gather information from different blocks. Owing to these components, HM-Conformer can efficiently detect spoofing evidence by processing various sequence lengths and aggregating them. In experimental resu
    
[^54]: 高斯过程与线性多核：频谱设计和多维数据的分布式学习

    Gaussian Processes with Linear Multiple Kernel: Spectrum Design and Distributed Learning for Multi-Dimensional Data. (arXiv:2309.08201v1 [cs.LG])

    [http://arxiv.org/abs/2309.08201](http://arxiv.org/abs/2309.08201)

    本文研究了高斯过程与线性多核在多维数据上的应用，提出了一种新的格点谱混合核公式，减少了超参数数量，同时保留了优化结构和逼近能力。通过引入分布式算法，使大规模超参数优化变得可行。

    

    高斯过程（GPs）已成为机器学习和信号处理的重要技术。GP建模的关键组成部分是核函数的选择，线性多核（LMKs）因其强大的建模能力和可解释性而成为一个吸引人的核函数类。本文重点研究格点谱混合（GSM）核，它是一种可以近似任意平稳核的LMK。具体来说，我们提出了一种新的GSM核公式，用于多维数据，相比现有公式减少了超参数的数量，同时保留了有利的优化结构和逼近能力。此外，为了使GSM核中的大规模超参数优化变得可行，我们首先引入了分布式SCA（DSCA）算法。在此基础上，我们基于交替方向乘子法（ADMM）框架提出了双重分布式SCA（D$^2$SCA）算法，使我们能够合作地进行优化。

    Gaussian processes (GPs) have emerged as a prominent technique for machine learning and signal processing. A key component in GP modeling is the choice of kernel, and linear multiple kernels (LMKs) have become an attractive kernel class due to their powerful modeling capacity and interpretability. This paper focuses on the grid spectral mixture (GSM) kernel, an LMK that can approximate arbitrary stationary kernels. Specifically, we propose a novel GSM kernel formulation for multi-dimensional data that reduces the number of hyper-parameters compared to existing formulations, while also retaining a favorable optimization structure and approximation capability. In addition, to make the large-scale hyper-parameter optimization in the GSM kernel tractable, we first introduce the distributed SCA (DSCA) algorithm. Building on this, we propose the doubly distributed SCA (D$^2$SCA) algorithm based on the alternating direction method of multipliers (ADMM) framework, which allows us to cooperativ
    
[^55]: 火星质子极光的可解释深度学习模型

    An Explainable Deep-learning Model of Proton Auroras on Mars. (arXiv:2309.08195v1 [astro-ph.EP])

    [http://arxiv.org/abs/2309.08195](http://arxiv.org/abs/2309.08195)

    这项研究开发了一个纯数据驱动模型，使用火星大气和挥发物演化 (MAVEN) 的观测资料，来解释火星质子极光。通过训练人工神经网络，可以准确重现每个Ly alpha辐射的强度，并对观测结果进行忠实重构。

    

    火星白天侧广泛观察到质子极光，被认为是氢 Ly alpha (121.6 nm) 辐射在120至150公里高度之间的显著增强。太阳风质子作为高能中性原子穿过火星热层进入大气层，被认为是质子极光的原因。因此，理解质子极光对于描绘太阳风与火星大气相互作用至关重要。最近观测到局部"斑块状"质子极光，暗示在不稳定的太阳风条件下，质子可能直接沉积到火星大气中。在这里，我们利用火星大气和挥发物演化 (MAVEN) 非现场观测和边缘扫描的 Ly alpha 辐射资料开发了一个纯数据驱动模型来模拟质子极光。我们训练了一个人工神经网络，可以以0.95的Pearson相关性重现每个Ly alpha辐射的强度，并对观测结果进行忠实重构。

    Proton auroras are widely observed on the day side of Mars, identified as a significant intensity enhancement in the hydrogen Ly alpha (121.6 nm) emission between 120 and 150~km altitudes. Solar wind protons penetrating as energetic neutral atoms into the Martian thermosphere are thought to be responsible for these auroras. Understanding proton auroras is therefore important for characterizing the solar wind interaction with the atmosphere of Mars. Recent observations of spatially localized "patchy" proton auroras suggest a possible direct deposition of protons into the atmosphere of Mars during unstable solar wind conditions. Here, we develop a purely data-driven model of proton auroras using Mars Atmosphere and Volatile EvolutioN (MAVEN) in situ observations and limb scans of Ly alpha emissions between 2014 and 2022. We train an artificial neural network that reproduces individual Ly alpha intensities with a Pearson correlation of 0.95 along with a faithful reconstruction of the obse
    
[^56]: 一种具有精度可扩展性的在极限边缘具有在设备学习能力的RISC-V DNN处理器

    A Precision-Scalable RISC-V DNN Processor with On-Device Learning Capability at the Extreme Edge. (arXiv:2309.08186v1 [cs.AR])

    [http://arxiv.org/abs/2309.08186](http://arxiv.org/abs/2309.08186)

    这篇论文提出了一种具有精度可扩展性的RISC-V DNN处理器，该处理器可以支持多种精度级别的定点DNN推断，并通过改进的FP16操作增强了在设备学习能力。

    

    极限边缘平台，例如车载智能设备，需要高效部署量化的深度神经网络（DNN），以便在能源、内存和计算资源有限的情况下实现智能应用。然而，由于量化水平的变化，许多边缘设备难以提高各种量化DNN的推断吞吐量，并且这些设备缺乏浮点（FP）支持的在设备学习能力，这阻碍了它们在确保数据隐私的同时提高模型准确性。为了解决以上挑战，我们提出了一种具有精度可扩展性的RISC-V DNN处理器，具有在设备学习能力。它可以方便地进行2位到16位的多种精度级别的定点DNN推断，并通过改进的FP16操作来增强在设备学习。此外，我们采用多种方法，如FP16乘法器重用和多精度整数乘法器重用，以及FPGA资源的平衡映射，大大提高了性能。

    Extreme edge platforms, such as in-vehicle smart devices, require efficient deployment of quantized deep neural networks (DNNs) to enable intelligent applications with limited amounts of energy, memory, and computing resources. However, many edge devices struggle to boost inference throughput of various quantized DNNs due to the varying quantization levels, and these devices lack floating-point (FP) support for on-device learning, which prevents them from improving model accuracy while ensuring data privacy. To tackle the challenges above, we propose a precision-scalable RISC-V DNN processor with on-device learning capability. It facilitates diverse precision levels of fixed-point DNN inference, spanning from 2-bit to 16-bit, and enhances on-device learning through improved support with FP16 operations. Moreover, we employ multiple methods such as FP16 multiplier reuse and multi-precision integer multiplier reuse, along with balanced mapping of FPGA resources, to significantly improve 
    
[^57]: 通过神经网络剪枝揭示不变性

    Unveiling Invariances via Neural Network Pruning. (arXiv:2309.08171v1 [cs.LG])

    [http://arxiv.org/abs/2309.08171](http://arxiv.org/abs/2309.08171)

    该论文提出了一种通过神经网络剪枝来学习捕捉数据相关的不变性的新型网络架构的框架。实验证明，这种学习的网络架构在视觉和表格数据集上都比密集神经网络表现出色，不仅效率高，而且效果好。

    

    不变性描述了对数据底层语义没有影响的转换。保持自然不变性的神经网络具有良好的归纳偏差和出色的性能。因此，现代网络被手工设计用来处理众所周知的不变性（例如平移）。我们提出了一个框架，通过剪枝来学习捕捉数据相关的不变性的新型网络架构。我们学到的网络架构在视觉和表格数据集上都比密集神经网络在效率和效果上都表现出色。我们在3个视觉和40个表格数据集上展示了我们的框架。

    Invariance describes transformations that do not alter data's underlying semantics. Neural networks that preserve natural invariance capture good inductive biases and achieve superior performance. Hence, modern networks are handcrafted to handle well-known invariances (ex. translations). We propose a framework to learn novel network architectures that capture data-dependent invariances via pruning. Our learned architectures consistently outperform dense neural networks on both vision and tabular datasets in both efficiency and effectiveness. We demonstrate our framework on multiple deep learning models across 3 vision and 40 tabular datasets.
    
[^58]: 预测还是拒绝：网络数据上的因果效应估计与不确定性

    To Predict or to Reject: Causal Effect Estimation with Uncertainty on Networked Data. (arXiv:2309.08165v1 [cs.LG])

    [http://arxiv.org/abs/2309.08165](http://arxiv.org/abs/2309.08165)

    本文提出了一种基于不确定性的图深度核学习框架来处理网络数据上因果效应估计中的正性假设违反问题，并在实验证明了该方法的优越性。

    

    由于网络观察数据的不平衡性，对于某些个体的因果效应预测可能严重违反正性/重叠假设，导致估计不可靠。然而，关于网络数据个体级治疗效应估计的这种潜在风险在很大程度上未被充分探索。为了创建一个更可信赖的因果效应估计器，我们提出了基于不确定性感知的图深度核学习 (GraphDKL) 框架，并通过Lipschitz约束来建模预测不确定性以识别不可靠的估计。据我们所知，GraphDKL是第一个在执行图上的因果效应估计时处理正性假设违反的框架。通过大量实验证明了我们所提出的方法在网络数据上的不确定性感知因果效应估计方面的优越性。

    Due to the imbalanced nature of networked observational data, the causal effect predictions for some individuals can severely violate the positivity/overlap assumption, rendering unreliable estimations. Nevertheless, this potential risk of individual-level treatment effect estimation on networked data has been largely under-explored. To create a more trustworthy causal effect estimator, we propose the uncertainty-aware graph deep kernel learning (GraphDKL) framework with Lipschitz constraint to model the prediction uncertainty with Gaussian process and identify unreliable estimations. To the best of our knowledge, GraphDKL is the first framework to tackle the violation of positivity assumption when performing causal effect estimation with graphs. With extensive experiments, we demonstrate the superiority of our proposed method in uncertainty-aware causal effect estimation on networked data.
    
[^59]: AdSEE: 研究图像样式编辑对广告吸引力的影响

    AdSEE: Investigating the Impact of Image Style Editing on Advertisement Attractiveness. (arXiv:2309.08159v1 [cs.CV])

    [http://arxiv.org/abs/2309.08159](http://arxiv.org/abs/2309.08159)

    本文研究了图像样式编辑对广告吸引力的影响。通过引入基于StyleGAN的面部语义编辑和反转，并结合传统的视觉和文本特征，我们提出了AdSEE方法，可用于预测在线广告的点击率。通过对QQ-AD数据集的评估，验证了AdSEE的有效性。

    

    在电子商务网站、社交媒体平台和搜索引擎中，在线广告是重要的元素。随着移动浏览的日益流行，许多在线广告都通过封面图片以及文本描述来吸引用户的注意力。最近的各种研究致力于通过考虑视觉特征来预测在线广告的点击率，或者通过组合最佳的广告元素来增强可见性。本文提出了广告样式编辑和吸引力增强（AdSEE），探讨了广告图像的语义编辑是否会影响或改变在线广告的受欢迎程度。我们引入了基于StyleGAN的面部语义编辑和反转，对广告图像进行训练，并使用基于GAN的面部潜在表示以及传统的视觉和文本特征来预测点击率。通过一个名为QQ-AD的大型数据集，包含20,527个样本，我们对AdSEE进行了评估。

    Online advertisements are important elements in e-commerce sites, social media platforms, and search engines. With the increasing popularity of mobile browsing, many online ads are displayed with visual information in the form of a cover image in addition to text descriptions to grab the attention of users. Various recent studies have focused on predicting the click rates of online advertisements aware of visual features or composing optimal advertisement elements to enhance visibility. In this paper, we propose Advertisement Style Editing and Attractiveness Enhancement (AdSEE), which explores whether semantic editing to ads images can affect or alter the popularity of online advertisements. We introduce StyleGAN-based facial semantic editing and inversion to ads images and train a click rate predictor attributing GAN-based face latent representations in addition to traditional visual and textual features to click rates. Through a large collected dataset named QQ-AD, containing 20,527 
    
[^60]: 一个自动化和分析移动设备及其应用的测试平台

    A Testbed for Automating and Analysing Mobile Devices and their Applications. (arXiv:2309.08158v1 [cs.NI])

    [http://arxiv.org/abs/2309.08158](http://arxiv.org/abs/2309.08158)

    这个研究介绍了一个测试平台，可以自动化生成和标记逼真的移动设备应用程序流量，为改善网络态势感知提供了机器学习技术的工具。

    

    随着网络攻击的复杂性和严重性增加，对改进网络态势感知的需求日益凸显。由于移动电话的动态行为和在网络上的缺乏可见性，它们对网络态势感知构成了重大威胁。机器学习技术通过向管理员提供有关构成其网络的设备和活动的见解，增强了态势感知。为了开发用于态势感知的机器学习技术，需要一个能够生成和标记网络流量的测试平台。然而，当前的测试平台无法自动化生成和标记逼真的网络流量。为了解决这个问题，我们描述了一个测试平台，该平台可以自动化移动设备上的应用程序以生成和标记逼真的流量。通过这个测试平台，我们创建了两个标记的网络流量数据集。我们对测试平台的自动化可靠性进行了分析，并对这些数据集进行了应用程序分类任务的基准测试。

    The need for improved network situational awareness has been highlighted by the growing complexity and severity of cyber-attacks. Mobile phones pose a significant risk to network situational awareness due to their dynamic behaviour and lack of visibility on a network. Machine learning techniques enhance situational awareness by providing administrators insight into the devices and activities which form their network. Developing machine learning techniques for situational awareness requires a testbed to generate and label network traffic. Current testbeds, however, are unable to automate the generation and labelling of realistic network traffic. To address this, we describe a testbed which automates applications on mobile devices to generate and label realistic traffic. From this testbed, two labelled datasets of network traffic have been created. We provide an analysis of the testbed automation reliability and benchmark the datasets for the task of application classification.
    
[^61]: 两步法知识蒸馏用于微弱语音增强

    Two-Step Knowledge Distillation for Tiny Speech Enhancement. (arXiv:2309.08144v1 [cs.SD])

    [http://arxiv.org/abs/2309.08144](http://arxiv.org/abs/2309.08144)

    本文提出了一种新颖的两步法知识蒸馏方法用于微弱语音增强模型。方法首先使用知识蒸馏目标预训练学生模型，然后切换到完全监督训练。同时，引入细粒度相似性保持的知识蒸馏损失，将学生模型的激活内部格拉姆矩阵与教师模型匹配。实验证明，该方法在高压缩和低信噪比条件下表现出显著的性能提升。

    

    对于嵌入式音频机器学习应用而言，微型的因果模型至关重要。模型压缩可以通过将大型教师模型的知识蒸馏到更小的学生模型中来实现。在本文中，我们提出了一种新颖的两步法来进行微弱语音增强模型的蒸馏。与标准方法中使用蒸馏损失和监督损失的加权混合不同，我们首先只使用知识蒸馏（KD）目标来预训练学生模型，然后切换到完全监督训练方案。我们还提出了一种新颖的细粒度相似性保持的KD损失，旨在将学生模型的激活内部格拉姆矩阵与教师模型的格拉姆矩阵匹配。我们的方法在多个方面都取得了显著的改进，尤其在恶劣条件下，包括高压缩和低信噪比（SNR），与基线相比，在输入SNR为-5 dB和63倍压缩下，信号失真比分别提高了0.9 dB和1.1 dB。

    Tiny, causal models are crucial for embedded audio machine learning applications. Model compression can be achieved via distilling knowledge from a large teacher into a smaller student model. In this work, we propose a novel two-step approach for tiny speech enhancement model distillation. In contrast to the standard approach of a weighted mixture of distillation and supervised losses, we firstly pre-train the student using only the knowledge distillation (KD) objective, after which we switch to a fully supervised training regime. We also propose a novel fine-grained similarity-preserving KD loss, which aims to match the student's intra-activation Gram matrices to that of the teacher. Our method demonstrates broad improvements, but particularly shines in adverse conditions including high compression and low signal to noise ratios (SNR), yielding signal to distortion ratio gains of 0.9 dB and 1.1 dB, respectively, at -5 dB input SNR and 63x compression compared to baseline.
    
[^62]: 音频差异学习用于音频字幕生成

    Audio Difference Learning for Audio Captioning. (arXiv:2309.08141v1 [eess.AS])

    [http://arxiv.org/abs/2309.08141](http://arxiv.org/abs/2309.08141)

    本研究引入了音频差异学习方法，通过创建特征表示空间来改进音频字幕生成。该方法使用参考音频和输入音频，生成描述它们差异的字幕，同时提出了一种独特的混合技术来消除差异和原始输入之间的需求。

    

    本研究引入了一种新的训练范式，即音频差异学习，用于改进音频字幕生成。所提出的学习方法的基本概念是创建一个保留音频之间关系的特征表示空间，从而能够生成详细描述复杂音频信息的字幕。该方法使用参考音频和输入音频，通过共享编码器将它们转换为特征表示。然后，从这些差异特征生成字幕描述它们的差异。此外，提出了一种独特的技术，涉及将输入音频与额外音频混合，并使用额外音频作为参考。这样，混合音频与参考音频之间的差异回到原始输入音频。这允许将原始输入的字幕作为其差异的字幕使用，消除了为差异添加额外注释的需求。

    This study introduces a novel training paradigm, audio difference learning, for improving audio captioning. The fundamental concept of the proposed learning method is to create a feature representation space that preserves the relationship between audio, enabling the generation of captions that detail intricate audio information. This method employs a reference audio along with the input audio, both of which are transformed into feature representations via a shared encoder. Captions are then generated from these differential features to describe their differences. Furthermore, a unique technique is proposed that involves mixing the input audio with additional audio, and using the additional audio as a reference. This results in the difference between the mixed audio and the reference audio reverting back to the original input audio. This allows the original input's caption to be used as the caption for their difference, eliminating the need for additional annotations for the difference
    
[^63]: PromptTTS++：使用自然语言描述控制提示式文本转语音中的说话者身份

    PromptTTS++: Controlling Speaker Identity in Prompt-Based Text-to-Speech Using Natural Language Descriptions. (arXiv:2309.08140v1 [eess.AS])

    [http://arxiv.org/abs/2309.08140](http://arxiv.org/abs/2309.08140)

    PromptTTS++是一种基于提示的文本转语音系统，可以使用自然语言描述控制说话者身份。与现有研究不同，该方法利用说话者提示来学习自然语言描述与声学特征的映射。

    

    我们提出了PromptTTS++，一种基于提示的文本转语音（TTS）合成系统，它允许使用自然语言描述来控制说话者身份。为了在基于提示的TTS框架中控制说话者身份，我们引入了说话者提示的概念，该提示描述了语音特征（如中性、年轻、老年和沉闷），旨在与说话风格大致独立。由于目前没有包含说话者提示的大规模数据集，我们首先使用LibriTTS-R语料库构建了一个基于手动注释的说话者提示数据集。然后，我们采用基于扩散的声学模型与混合密度网络来建模训练数据中的多样化说话者因素。与之前仅依赖样式提示的研究不同，样式提示仅描述了说话者个性化的有限方面，如音调、说话速度和能量，我们的方法利用额外的说话者提示来有效地学习从自然语言描述到声学特征的映射。

    We propose PromptTTS++, a prompt-based text-to-speech (TTS) synthesis system that allows control over speaker identity using natural language descriptions. To control speaker identity within the prompt-based TTS framework, we introduce the concept of speaker prompt, which describes voice characteristics (e.g., gender-neutral, young, old, and muffled) designed to be approximately independent of speaking style. Since there is no large-scale dataset containing speaker prompts, we first construct a dataset based on the LibriTTS-R corpus with manually annotated speaker prompts. We then employ a diffusion-based acoustic model with mixture density networks to model diverse speaker factors in the training data. Unlike previous studies that rely on style prompts describing only a limited aspect of speaker individuality, such as pitch, speaking speed, and energy, our method utilizes an additional speaker prompt to effectively learn the mapping from natural language descriptions to the acoustic f
    
[^64]: Oobleck：使用流水线模板实现大型模型的弹性分布式训练

    Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates. (arXiv:2309.08125v1 [cs.DC])

    [http://arxiv.org/abs/2309.08125](http://arxiv.org/abs/2309.08125)

    Oobleck采用流水线模板和已复制模型状态来实现对大型模型的弹性分布式训练，并通过有效利用资源和快速恢复来提供高吞吐量。在评估中，Oobleck在吞吐量上胜过了Bamboo和Varuna等最先进的容错解决方案。

    

    Oobleck通过采用规定的容错率，可实现对大型深度神经网络模型的弹性分布式训练。它采用了规划-执行的协同设计方法，首先生成一组异构的流水线模板，并实例化至少$ f + 1 $个逻辑等效的流水线副本，以容纳任何$f$个同时故障。在执行过程中，它依赖于跨副本的已复制模型状态来提供快速恢复。Oobleck可以可靠地保证在$f$个或更少的同时故障后，初始创建的流水线模板的某种组合可以用于覆盖所有可用资源，从而始终避免资源闲置。在具有数十亿参数的大型深度神经网络模型上的评估表明，Oobleck提供了一致高吞吐量，并且在吞吐量上胜过了Bamboo和Varuna等最先进的容错解决方案。

    Oobleck enables resilient distributed training of large DNN models with guaranteed fault tolerance. It takes a planning-execution co-design approach, where it first generates a set of heterogeneous pipeline templates and instantiates at least $f+1$ logically equivalent pipeline replicas to tolerate any $f$ simultaneous failures. During execution, it relies on already-replicated model states across the replicas to provide fast recovery. Oobleck provably guarantees that some combination of the initially created pipeline templates can be used to cover all available resources after $f$ or fewer simultaneous failures, thereby avoiding resource idling at all times. Evaluation on large DNN models with billions of parameters shows that Oobleck provides consistently high throughput, and it outperforms state-of-the-art fault tolerance solutions like Bamboo and Varuna by up to $13.9x$.
    
[^65]: 快速准确的深度循环关闭和位置再定位以实现可靠的LiDAR SLAM

    Fast and Accurate Deep Loop Closing and Relocalization for Reliable LiDAR SLAM. (arXiv:2309.08086v1 [cs.RO])

    [http://arxiv.org/abs/2309.08086](http://arxiv.org/abs/2309.08086)

    本文提出了一个名为LCR-Net的多头网络，利用新颖的特征提取和姿态感知机制来快速准确地处理循环关闭和位置再定位任务。实验结果表明，LCR-Net在候选检索、闭环点云配准和多数据集连续再定位等任务中表现优异，超过了当前最先进的方法，并具有出色的泛化能力。

    

    循环关闭和位置再定位是建立可靠和稳定的长期SLAM的关键技术，用于解决姿态估计漂移和退化问题。本文首先在一个统一的框架下表述了循环关闭和位置再定位。然后，我们提出了一种新颖的多头网络LCR-Net来有效地处理这两个任务。它利用了新颖的特征提取和姿态感知机制来精确估计LiDAR扫描对之间的相似性和6自由度姿态。最后，我们将LCR-Net整合到一个SLAM系统中，在室外驾驶环境中实现了稳健准确的在线LiDAR SLAM。我们通过三种从循环关闭和位置再定位得出的设置，包括候选检索、闭环点云配准和多数据集的连续再定位，对LCR-Net进行了全面评估。结果表明，LCR-Net在这三个任务中表现出色，超过了当前最先进的方法，并展示了出色的泛化能力。

    Loop closing and relocalization are crucial techniques to establish reliable and robust long-term SLAM by addressing pose estimation drift and degeneration. This article begins by formulating loop closing and relocalization within a unified framework. Then, we propose a novel multi-head network LCR-Net to tackle both tasks effectively. It exploits novel feature extraction and pose-aware attention mechanism to precisely estimate similarities and 6-DoF poses between pairs of LiDAR scans. In the end, we integrate our LCR-Net into a SLAM system and achieve robust and accurate online LiDAR SLAM in outdoor driving environments. We thoroughly evaluate our LCR-Net through three setups derived from loop closing and relocalization, including candidate retrieval, closed-loop point cloud registration, and continuous relocalization using multiple datasets. The results demonstrate that LCR-Net excels in all three tasks, surpassing the state-of-the-art methods and exhibiting a remarkable generalizati
    
[^66]: 使用对比学习的监督型随机邻域嵌入

    Supervised Stochastic Neighbor Embedding Using Contrastive Learning. (arXiv:2309.08077v1 [cs.LG])

    [http://arxiv.org/abs/2309.08077](http://arxiv.org/abs/2309.08077)

    该论文将自监督对比学习方法扩展到全监督设置中，允许有效利用标签信息，并在保留数据集邻域信息的同时，将同一类别的样本聚集在一起，将不同类别的样本聚集分开。

    

    随机邻域嵌入（SNE）方法t-SNE和UMAP是两种常用的数据可视化降维方法。对比学习，尤其是自监督对比学习（SSCL），在从无标签数据中嵌入特征方面取得了巨大成功。本研究在保留数据集邻域信息的范围内，将自监督对比学习方法扩展到全监督设置中，使我们能够有效利用标签信息。在低维嵌入空间中，将同一类别的样本聚集在一起，同时将不同类别的样本聚集分开。

    Stochastic neighbor embedding (SNE) methods $t$-SNE, UMAP are two most popular dimensionality reduction methods for data visualization. Contrastive learning, especially self-supervised contrastive learning (SSCL), has showed great success in embedding features from unlabeled data. The conceptual connection between SNE and SSCL has been exploited. In this work, within the scope of preserving neighboring information of a dataset, we extend the self-supervised contrastive approach to the fully-supervised setting, allowing us to effectively leverage label information. Clusters of samples belonging to the same class are pulled together in low-dimensional embedding space, while simultaneously pushing apart clusters of samples from different classes.
    
[^67]: 基于启发式迭代优化的形态学感知一致性计算（MACCHIatO）的研究

    Morphologically-Aware Consensus Computation via Heuristics-based IterATive Optimization (MACCHIatO). (arXiv:2309.08066v1 [cs.CV])

    [http://arxiv.org/abs/2309.08066](http://arxiv.org/abs/2309.08066)

    本文提出了一种基于启发式迭代优化的方法，利用精心选择的距离的Fréchet均值构建二进制或概率一致性分割。与STAPLE方法相比，该方法不受图像背景大小和先验选择的影响。

    

    从多个二进制或概率掩模中提取一致性分割是解决各种任务的重要方法，如解析标注者间的差异性或多个神经网络输出的融合。本文首先证明了STAPLE算法的输出受到图像背景大小和先验选择的影响。然后我们提出了一种新的方法，基于精心选择的距离的Fréchet均值构建二进制或概率一致性分割，使其完全独立于图像背景大小。我们提供了一种启发式方法来优化此准则，从而使一个体素的类别完全由它与不同掩模的体素级距离、它所属的连通组件和分割它的标注者组决定。我们在多个数据集上与STAPLE方法进行了广泛比较。

    The extraction of consensus segmentations from several binary or probabilistic masks is important to solve various tasks such as the analysis of inter-rater variability or the fusion of several neural network outputs. One of the most widely used methods to obtain such a consensus segmentation is the STAPLE algorithm. In this paper, we first demonstrate that the output of that algorithm is heavily impacted by the background size of images and the choice of the prior. We then propose a new method to construct a binary or a probabilistic consensus segmentation based on the Fr\'{e}chet means of carefully chosen distances which makes it totally independent of the image background size. We provide a heuristic approach to optimize this criterion such that a voxel's class is fully determined by its voxel-wise distance to the different masks, the connected component it belongs to and the group of raters who segmented it. We compared extensively our method on several datasets with the STAPLE met
    
[^68]: 旅行波编码最近的过去并增强序列学习

    Traveling Waves Encode the Recent Past and Enhance Sequence Learning. (arXiv:2309.08045v1 [cs.NE])

    [http://arxiv.org/abs/2309.08045](http://arxiv.org/abs/2309.08045)

    本论文介绍了Wave-RNN (wRNN)模型，展示了旅行波机制如何有效地编码最近的过去，并在合成记忆任务中比波动模型表现更好。

    

    神经活动的旅行波现象在大脑的不同区域和尺度上都有所观察到，然而，它们在计算角色上的具体作用仍存在争议。一个基于物理的假设认为，皮质层可以像波动场一样，通过沿着皮质表面传播的波动来存储顺序刺激的短期记忆。然而，由于缺乏一个简单的递归神经网络架构能够展现出这种波动，迄今为止，这个想法的计算意义一直是假设性的。在这项工作中，我们引入了一个模型来填补这个空白，我们称之为Wave-RNN (wRNN)，并展示了连通性约束和初始化在波动动力学出现中起到了关键作用。然后，我们经验证实了这样的架构的确通过一系列合成记忆任务有效地编码了最近的过去，在这些任务中，wRNN比波动模型学习更快、表现更好。

    Traveling waves of neural activity have been observed throughout the brain at a diversity of regions and scales; however, their precise computational role is still debated. One physically grounded hypothesis suggests that the cortical sheet may act like a wave-field capable of storing a short-term memory of sequential stimuli through induced waves traveling across the cortical surface. To date, however, the computational implications of this idea have remained hypothetical due to the lack of a simple recurrent neural network architecture capable of exhibiting such waves. In this work, we introduce a model to fill this gap, which we denote the Wave-RNN (wRNN), and demonstrate how both connectivity constraints and initialization play a crucial role in the emergence of wave-like dynamics. We then empirically show how such an architecture indeed efficiently encodes the recent past through a suite of synthetic memory tasks where wRNNs learn faster and perform significantly better than wave-
    
[^69]: 我们需要多少个神经元？用梯度下降训练的浅层网络的精细分析

    How many Neurons do we need? A refined Analysis for Shallow Networks trained with Gradient Descent. (arXiv:2309.08044v1 [stat.ML])

    [http://arxiv.org/abs/2309.08044](http://arxiv.org/abs/2309.08044)

    本论文在神经切向核（NTK）范式下，通过分析用梯度下降训练的两层神经网络的泛化性质，改进了现有结果，并得出了快速收敛的速度。此外，我们证明了训练过程中权重保持在初始位置附近，半径与回归函数的平滑度和NTK的积分算子的特征值衰减程度有关。

    

    我们在神经切向核（NTK）范式下，分析了用梯度下降（GD）训练的两层神经网络的泛化性质。对于早停的GD，我们导出了快速收敛的速度，这在非参数回归和再生核希尔伯特空间的框架中已知是最小值的最优解。在这过程中，我们精确地追踪了泛化所需的隐藏层神经元数量，并改进了现有结果。我们进一步展示了训练过程中权重保持在初始位置附近的情况，其半径取决于回归函数的平滑度和与NTK相关联的积分算子的特征值衰减程度。

    We analyze the generalization properties of two-layer neural networks in the neural tangent kernel (NTK) regime, trained with gradient descent (GD). For early stopped GD we derive fast rates of convergence that are known to be minimax optimal in the framework of non-parametric regression in reproducing kernel Hilbert spaces. On our way, we precisely keep track of the number of hidden neurons required for generalization and improve over existing results. We further show that the weights during training remain in a vicinity around initialization, the radius being dependent on structural assumptions such as degree of smoothness of the regression function and eigenvalue decay of the integral operator associated to the NTK.
    
[^70]: 关于在Heckman选择模型中的预测特征分配问题

    On Prediction Feature Assignment in the Heckman Selection Model. (arXiv:2309.08043v1 [cs.LG])

    [http://arxiv.org/abs/2309.08043](http://arxiv.org/abs/2309.08043)

    本文研究在非随机缺失样本选择偏差下的预测模型性能降低问题，提出了Heckman-FA框架来获取恰当的预测特征。

    

    在非随机缺失（MNAR）样本选择偏差的情况下，预测模型的性能往往下降。本文关注MNAR样本选择偏差的一个经典例子，即一部分样本具有非随机缺失结果。Heckman选择模型及其变种通常用于处理这种类型的样本选择偏差。Heckman模型使用两个不同的方程来模拟样本的预测和选择，其中选择特征包括所有预测特征。在使用Heckman模型时，必须从选择特征集中正确选择预测特征。然而，对于Heckman模型来说，选择正确的预测特征是一项具有挑战性的任务，尤其是当选择特征的数量较多时。现有的使用Heckman模型的方法通常提供一个手动选择的预测特征集。在本文中，我们提出了Heckman-FA作为一种新的数据驱动框架来获得预测特征。

    Under missing-not-at-random (MNAR) sample selection bias, the performance of a prediction model is often degraded. This paper focuses on one classic instance of MNAR sample selection bias where a subset of samples have non-randomly missing outcomes. The Heckman selection model and its variants have commonly been used to handle this type of sample selection bias. The Heckman model uses two separate equations to model the prediction and selection of samples, where the selection features include all prediction features. When using the Heckman model, the prediction features must be properly chosen from the set of selection features. However, choosing the proper prediction features is a challenging task for the Heckman model. This is especially the case when the number of selection features is large. Existing approaches that use the Heckman model often provide a manually chosen set of prediction features. In this paper, we propose Heckman-FA as a novel data-driven framework for obtaining pr
    
[^71]: 基于大型预训练基础模型的多语种演讲者转换检测模型（USM-SCD）

    USM-SCD: Multilingual Speaker Change Detection Based on Large Pretrained Foundation Models. (arXiv:2309.08023v1 [eess.AS])

    [http://arxiv.org/abs/2309.08023](http://arxiv.org/abs/2309.08023)

    USM-SCD是一种基于大型预训练基础模型的多语种演讲者转换检测模型，通过微调模型参数，可以同时检测演讲者转换并为96种语言执行自动语音识别。在实验中表现出了优异的性能。

    

    我们提出了一种多语种演讲者转换检测模型（USM-SCD），可以同时检测演讲者转换并为96种语言执行自动语音识别。该模型是从一个经过大量受监督和无监督数据训练的语音基础模型进行调整而来的，展示了从大型通用基础模型到下游任务的微调的实用性。通过一系列消融研究，我们分析了这个多语种演讲者转换检测模型的性能。我们展示了USM-SCD模型可以在由来自96种语言的数据构成的测试集上实现超过75%的平均演讲者转换检测F1得分。在美式英语上，USM-SCD模型可以在各种公共和内部测试集上实现85.8%的演讲者转换检测F1得分，相对于之前的单语言基准模型提高了21%。我们还展示了只需要微调可训练模型参数的四分之一就可以实现最佳模型性能。

    We introduce a multilingual speaker change detection model (USM-SCD) that can simultaneously detect speaker turns and perform ASR for 96 languages. This model is adapted from a speech foundation model trained on a large quantity of supervised and unsupervised data, demonstrating the utility of fine-tuning from a large generic foundation model for a downstream task. We analyze the performance of this multilingual speaker change detection model through a series of ablation studies. We show that the USM-SCD model can achieve more than 75% average speaker change detection F1 score across a test set that consists of data from 96 languages. On American English, the USM-SCD model can achieve an 85.8% speaker change detection F1 score across various public and internal test sets, beating the previous monolingual baseline model by 21% relative. We also show that we only need to fine-tune one-quarter of the trainable model parameters to achieve the best model performance. The USM-SCD model exhib
    
[^72]: CRYPTO-MINE: 通过互信息神经估计进行密码分析

    CRYPTO-MINE: Cryptanalysis via Mutual Information Neural Estimation. (arXiv:2309.08019v1 [cs.CR])

    [http://arxiv.org/abs/2309.08019](http://arxiv.org/abs/2309.08019)

    CRYPTO-MINE是一种通过神经网络估计互信息的新方法，应用于选择明文攻击中明文和密文之间的互信息估计。该方法可用于分析密码系统的计算安全性和信息泄露与输入分布之间的关系。

    

    互信息（MI）作为评估密码系统效率的指标具有广泛的历史。然而，在高维空间中估计未知随机变量之间的互信息是具有挑战性的。机器学习的最新进展使得使用神经网络估计互信息成为可能。本文提出了互信息估计在密码学领域的新应用。我们建议将这种方法直接应用于选择明文攻击中明文和密文之间的估计互信息。如果有的话，加密中的泄露信息可能会被对手利用来破坏密码系统的计算安全性。我们通过对多个加密方案和基准方法进行经验分析来评估我们方法的效率。此外，我们还扩展了对提供个体保密性的基于网络编码的密码系统的分析，并研究了信息泄露和输入分布之间的关系。

    The use of Mutual Information (MI) as a measure to evaluate the efficiency of cryptosystems has an extensive history. However, estimating MI between unknown random variables in a high-dimensional space is challenging. Recent advances in machine learning have enabled progress in estimating MI using neural networks. This work presents a novel application of MI estimation in the field of cryptography. We propose applying this methodology directly to estimate the MI between plaintext and ciphertext in a chosen plaintext attack. The leaked information, if any, from the encryption could potentially be exploited by adversaries to compromise the computational security of the cryptosystem. We evaluate the efficiency of our approach by empirically analyzing multiple encryption schemes and baseline approaches. Furthermore, we extend the analysis to novel network coding-based cryptosystems that provide individual secrecy and study the relationship between information leakage and input distribution
    
[^73]: 使用自动化机器学习方法检测美国东北地区临界地带内研究流域时间序列数据的异常峰行为

    An Automated Machine Learning Approach for Detecting Anomalous Peak Patterns in Time Series Data from a Research Watershed in the Northeastern United States Critical Zone. (arXiv:2309.07992v1 [cs.LG])

    [http://arxiv.org/abs/2309.07992](http://arxiv.org/abs/2309.07992)

    本文提出了一个自动化机器学习框架，用于检测美国东北地区临界地带研究流域传感器生成的时间序列数据中的异常峰值模式。通过合成生成带有标记的数据集和自动超参数优化机制，该框架克服了标记数据和选择合适的深度学习模型的挑战。

    

    本文提出了一种自动化机器学习框架，旨在帮助水文学家检测美国东北地区临界地带研究流域传感器生成的时间序列数据中的异常情况。该框架专注于识别峰值模式异常，这可能是由于传感器故障或自然现象引起的。然而，使用分类方法进行异常检测存在挑战，例如需要标记数据作为基准和选择最适合给定任务和数据集的深度学习模型。为了解决这些挑战，我们的框架通过将合成的峰值模式注入到合成生成的时间序列数据中生成带有标记的数据集，并结合自动化的超参数优化机制。该机制从五种选择的模型中生成一个具有最佳架构和训练参数的优化模型实例，即时序卷积网络（

    This paper presents an automated machine learning framework designed to assist hydrologists in detecting anomalies in time series data generated by sensors in a research watershed in the northeastern United States critical zone. The framework specifically focuses on identifying peak-pattern anomalies, which may arise from sensor malfunctions or natural phenomena. However, the use of classification methods for anomaly detection poses challenges, such as the requirement for labeled data as ground truth and the selection of the most suitable deep learning model for the given task and dataset. To address these challenges, our framework generates labeled datasets by injecting synthetic peak patterns into synthetically generated time series data and incorporates an automated hyperparameter optimization mechanism. This mechanism generates an optimized model instance with the best architectural and training parameters from a pool of five selected models, namely Temporal Convolutional Network (
    
[^74]: 折叠注意力：面向设备的Transformer流式语音识别的内存和功耗优化

    Folding Attention: Memory and Power Optimization for On-Device Transformer-based Streaming Speech Recognition. (arXiv:2309.07988v1 [cs.LG])

    [http://arxiv.org/abs/2309.07988](http://arxiv.org/abs/2309.07988)

    本论文提出了一种名为折叠注意力的技术，在基于Transformer的流式语音识别模型中，通过减少线性投影层的数量，显著减小了模型大小，提高了内存和功耗效率，实验证明可以将模型大小减小24%、功耗减小23%。

    

    基于Transformer的模型在语音识别中表现出色。现有的用于优化Transformer推断的努力，通常针对长上下文应用，主要集中在简化注意力得分计算上。然而，流式语音识别模型通常每次只处理有限数量的令牌，因此注意力得分计算在这种情况下并不是瓶颈所在。相反，瓶颈在于多头注意力和前馈网络的线性投影层，它们构成了模型大小的相当部分，并对计算、内存和功耗的使用产生重要影响。为了解决这一瓶颈，我们提出了折叠注意力，这是一种针对这些线性层的技术，显著减小了模型大小，并提高了内存和功耗效率。设备上的基于Transformer的流式语音识别模型的实验证明，折叠注意力可以将模型大小（和相应的内存消耗）减小多达24%，并将功耗减小多达23%，而无需进行补充。

    Transformer-based models excel in speech recognition. Existing efforts to optimize Transformer inference, typically for long-context applications, center on simplifying attention score calculations. However, streaming speech recognition models usually process a limited number of tokens each time, making attention score calculation less of a bottleneck. Instead, the bottleneck lies in the linear projection layers of multi-head attention and feedforward networks, constituting a substantial portion of the model size and contributing significantly to computation, memory, and power usage.  To address this bottleneck, we propose folding attention, a technique targeting these linear layers, significantly reducing model size and improving memory and power efficiency. Experiments on on-device Transformer-based streaming speech recognition models show that folding attention reduces model size (and corresponding memory consumption) by up to 24% and power consumption by up to 23%, all without comp
    
[^75]: 观点文本倒置：通过预训练的2D扩散模型释放新颖的视图合成

    Viewpoint Textual Inversion: Unleashing Novel View Synthesis with Pretrained 2D Diffusion Models. (arXiv:2309.07986v1 [cs.CV])

    [http://arxiv.org/abs/2309.07986](http://arxiv.org/abs/2309.07986)

    本研究展示了通过预训练的2D图像扩散模型，可以从仅有2D监督的情况下提取出3D结构信息，并利用该信息进行3D视觉任务。通过观点神经文本倒置（ViewNeTI）方法，我们可以控制生成图像中对象的3D视点，有效解决新颖视图合成问题，并在单视图情况下具有良好的语义细节和逼真度。

    

    文本到图像扩散模型可以理解对象之间的空间关系，但它们是否能够仅通过2D监督来表示世界的真实3D结构？我们证明，是的，3D知识被编码在2D图像扩散模型（如稳定扩散模型）中，我们展示了这种结构可以用于3D视觉任务。我们的方法，观点神经文本倒置（ViewNeTI），可以控制生成图像中对象的3D视点。我们训练一个小型神经映射器，用于获取相机视点参数并预测文本编码器的潜在向量；然后利用这些潜在向量来调整扩散生成过程，生成具有所需相机视点的图像。ViewNeTI自然解决了新颖视图合成（NVS）问题。通过利用被冻结的扩散模型作为先验知识，我们可以用很少的输入视图来解决NVS问题；我们甚至可以进行单视图新颖视图合成。与之前的方法相比，我们的单视图NVS预测具有良好的语义细节和逼真度。

    Text-to-image diffusion models understand spatial relationship between objects, but do they represent the true 3D structure of the world from only 2D supervision? We demonstrate that yes, 3D knowledge is encoded in 2D image diffusion models like Stable Diffusion, and we show that this structure can be exploited for 3D vision tasks. Our method, Viewpoint Neural Textual Inversion (ViewNeTI), controls the 3D viewpoint of objects in generated images from frozen diffusion models. We train a small neural mapper to take camera viewpoint parameters and predict text encoder latents; the latents then condition the diffusion generation process to produce images with the desired camera viewpoint.  ViewNeTI naturally addresses Novel View Synthesis (NVS). By leveraging the frozen diffusion model as a prior, we can solve NVS with very few input views; we can even do single-view novel view synthesis. Our single-view NVS predictions have good semantic details and photorealism compared to prior methods.
    
[^76]: SLMIA-SR: Speaker-Level Membership Inference Attacks against Speaker Recognition Systems

    SLMIA-SR: Speaker-Level Membership Inference Attacks against Speaker Recognition Systems. (arXiv:2309.07983v1 [cs.CR])

    [http://arxiv.org/abs/2309.07983](http://arxiv.org/abs/2309.07983)

    这项研究提出了SLMIA-SR，这是针对说话人识别系统的第一个针对说话人级别成员推断攻击。与传统的示例级攻击不同，这种攻击方法可以确定一个给定的声音是否与训练数据中的任何声音有关，无论它们是否相同。这对实践非常有用，因为训练和推断声音通常是不同的，而且考虑到说话人识别的开放性质，也是有意义的。

    

    成员推断攻击允许对手确定一个特定示例是否包含在模型的训练数据集中。虽然先前的研究已经证实了在各种应用中进行此类攻击的可行性，但没有一个研究专注于说话人识别（SR），这是一种有前途的基于声音的生物特征识别技术。在这项工作中，我们提出了SLMIA-SR，这是第一个针对SR量身定制的成员推断攻击。与传统的示例级攻击不同，我们的攻击特点是说话人级别的成员推断，即确定给定推断声音中是否有任何给定说话人的声音，无论它们是否与给定推断声音相同。这在实践中非常有用，因为训练和推断声音通常是不同的，而且考虑到SR的开放性质，也是有意义的，即识别说话人往往没有出现在训练数据中。我们利用两个训练目标：内部接近度和外部远离度，来进行攻击。

    Membership inference attacks allow adversaries to determine whether a particular example was contained in the model's training dataset. While previous works have confirmed the feasibility of such attacks in various applications, none has focused on speaker recognition (SR), a promising voice-based biometric recognition technique. In this work, we propose SLMIA-SR, the first membership inference attack tailored to SR. In contrast to conventional example-level attack, our attack features speaker-level membership inference, i.e., determining if any voices of a given speaker, either the same as or different from the given inference voices, have been involved in the training of a model. It is particularly useful and practical since the training and inference voices are usually distinct, and it is also meaningful considering the open-set nature of SR, namely, the recognition speakers were often not present in the training data. We utilize intra-closeness and inter-farness, two training objec
    
[^77]: 为学习的ISTA进行不确定性量化

    Uncertainty quantification for learned ISTA. (arXiv:2309.07982v1 [stat.ML])

    [http://arxiv.org/abs/2309.07982](http://arxiv.org/abs/2309.07982)

    本文提出了一种严谨的方法来获得LISTA估计量的置信区间，为模型-based深度学习解决方案中的不确定性提供了理论支持。

    

    近年来，基于模型的深度学习方法在逆问题中已经引起越来越多的关注，因为它们在数值性能和解释性方面都处于最前沿。此外，结合先验领域知识可以使训练更加高效，因为较少的参数数量允许使用较小的数据集进行训练。在这些基于模型的学习技术中，算法展开方案脱颖而出。尽管它们的快速发展与传统的高维统计方法密切相关，但它们缺乏确定性估计，对于不确定性量化的理论仍然存在困难。本文提出了一种严谨的方法来获得LISTA估计量的置信区间，从而为填补这一空白迈出了一步。

    Model-based deep learning solutions to inverse problems have attracted increasing attention in recent years as they bridge state-of-the-art numerical performance with interpretability. In addition, the incorporated prior domain knowledge can make the training more efficient as the smaller number of parameters allows the training step to be executed with smaller datasets. Algorithm unrolling schemes stand out among these model-based learning techniques. Despite their rapid advancement and their close connection to traditional high-dimensional statistical methods, they lack certainty estimates and a theory for uncertainty quantification is still elusive. This work provides a step towards closing this gap proposing a rigorous way to obtain confidence intervals for the LISTA estimator.
    
[^78]: 用于推理体验智能体的数据源

    A Data Source for Reasoning Embodied Agents. (arXiv:2309.07974v1 [cs.LG])

    [http://arxiv.org/abs/2309.07974](http://arxiv.org/abs/2309.07974)

    本研究提出了一个与体验智能体集成的新数据生成器，用于机器推理。该生成器生成的数据包括模板化的文本查询和答案，并与编码为数据库的世界状态相匹配。通过实验发现，当前模型可以回答一些关于世界状态的问题，但在其他问题上存在困难。

    

    最近在使用机器学习模型进行推理任务方面取得了进展，这得益于新颖的模型架构、大规模的预训练协议以及专门用于微调的推理数据集。在这项工作中，为了进一步推动这些进展，我们引入了一个与体验智能体集成的新数据生成器用于机器推理。生成的数据包括模板化的文本查询和答案，与编码为数据库的世界状态相匹配。这些世界状态是世界动态和智能体行为的结果。我们展示了几种基准模型在训练集实例化上的结果。这些基准模型包括在数据库的文本格式化表示上进行微调的预训练语言模型，以及在知识图表示的图结构Transformer上操作的模型。我们发现这些模型可以回答一些关于世界状态的问题，但在其他问题上存在困难。这些结果暗示了设计神经网络推理新的研究方向。

    Recent progress in using machine learning models for reasoning tasks has been driven by novel model architectures, large-scale pre-training protocols, and dedicated reasoning datasets for fine-tuning. In this work, to further pursue these advances, we introduce a new data generator for machine reasoning that integrates with an embodied agent. The generated data consists of templated text queries and answers, matched with world-states encoded into a database. The world-states are a result of both world dynamics and the actions of the agent. We show the results of several baseline models on instantiations of train sets. These include pre-trained language models fine-tuned on a text-formatted representation of the database, and graph-structured Transformers operating on a knowledge-graph representation of the database. We find that these models can answer some questions about the world-state, but struggle with others. These results hint at new research directions in designing neural reaso
    
[^79]: 基于复数的神经网络用于数据驱动的信号处理和信号理解

    Complex-Valued Neural Networks for Data-Driven Signal Processing and Signal Understanding. (arXiv:2309.07948v1 [eess.SP])

    [http://arxiv.org/abs/2309.07948](http://arxiv.org/abs/2309.07948)

    该论文介绍了一个基于PyTorch构建的软件包，用于实现复数值神经网络操作和架构的轻量级接口。该软件包的目标是为信号处理、感知和通信等领域提供高效的复数值模型支持。

    

    复数值神经网络已经出现，对于信号处理、感知和通信领域的许多任务具有卓越的建模性能。然而，目前发展复数值模型需要开发基本的深度学习操作，例如线性或卷积层，因为现代的深度学习框架如PyTorch和TensorFlow对于复数值神经网络的支持不足。本文概述了一个建立在PyTorch上的软件包，旨在实现常见复数值神经网络操作和架构的轻量级接口。类似于自然语言理解（NLU）最近在基于文本的智能方面取得了巨大进展，射频信号理解（RFSU）是一个有前途的领域，它使用信号力学基础洞察力与数据驱动建模能力的混合方法扩展了传统信号处理算法。值得注意的是，我们包括了线性、卷积和...

    Complex-valued neural networks have emerged boasting superior modeling performance for many tasks across the signal processing, sensing, and communications arenas. However, developing complex-valued models currently demands development of basic deep learning operations, such as linear or convolution layers, as modern deep learning frameworks like PyTorch and Tensor flow do not adequately support complex-valued neural networks. This paper overviews a package built on PyTorch with the intention of implementing light-weight interfaces for common complex-valued neural network operations and architectures. Similar to natural language understanding (NLU), which as recently made tremendous leaps towards text-based intelligence, RF Signal Understanding (RFSU) is a promising field extending conventional signal processing algorithms using a hybrid approach of signal mechanics-based insight with data-driven modeling power. Notably, we include efficient implementations for linear, convolution, and
    
[^80]: TiBGL: 模板引导的脑图学习用于功能性神经影像分析

    TiBGL: Template-induced Brain Graph Learning for Functional Neuroimaging Analysis. (arXiv:2309.07947v1 [cs.AI])

    [http://arxiv.org/abs/2309.07947](http://arxiv.org/abs/2309.07947)

    TiBGL是一种模板引导的脑图学习框架，用于功能性神经影像分析。它具有判别和可解释能力，旨在通过学习功能连接数据的有用特征来改进神经疾病的诊断效率。

    

    近年来，功能性磁共振成像已成为研究人类大脑功能连接网络的有力工具。相关研究表明，人脑的功能连接网络可以提高神经疾病诊断的效率。然而，功能性神经影像领域仍存在两个挑战限制着进展。首先，功能连接数据中存在大量噪音和冗余信息，导致性能不佳。其次，现有的脑网络模型往往偏向于分类性能或对学习模型背后的神经科学发现的解释。为了应对这些挑战，本文提出了一种新颖的脑图学习框架，称为模板引导的脑图学习（TiBGL），具有判别和可解释能力。受到与功能连接相关的医学发现的启发，TiBGL的目标是通过模板引导方法来学习功能连接数据的有用特征。

    In recent years, functional magnetic resonance imaging has emerged as a powerful tool for investigating the human brain's functional connectivity networks. Related studies demonstrate that functional connectivity networks in the human brain can help to improve the efficiency of diagnosing neurological disorders. However, there still exist two challenges that limit the progress of functional neuroimaging. Firstly, there exists an abundance of noise and redundant information in functional connectivity data, resulting in poor performance. Secondly, existing brain network models have tended to prioritize either classification performance or the interpretation of neuroscience findings behind the learned models. To deal with these challenges, this paper proposes a novel brain graph learning framework called Template-induced Brain Graph Learning (TiBGL), which has both discriminative and interpretable abilities. Motivated by the related medical findings on functional connectivites, TiBGL prop
    
[^81]: 通过物理知识引导的机器学习方法来计算奇异摄动系统的慢不变流形

    Slow Invariant Manifolds of Singularly Perturbed Systems via Physics-Informed Machine Learning. (arXiv:2309.07946v1 [math.DS])

    [http://arxiv.org/abs/2309.07946](http://arxiv.org/abs/2309.07946)

    通过物理知识引导的机器学习方法用于计算奇异摄动系统的慢不变流形，提供了显式形式的函数来构建和数值积分缩减模型，并通过三个基准问题的评估证明了其有效性。

    

    我们提出了一种通过物理知识引导的机器学习方法，用于近似计算奇异摄动系统的慢不变流形，并提供了显式形式的函数来便于构建和数值积分缩减模型。该方案在几何奇异摄动理论框架下解决与不变方程（IE）对应的偏微分方程。为了解决IE，我们采用了两种神经网络结构，即前馈神经网络（FNNs）和随机投影神经网络（RPNNs），利用符号微分来计算学习过程所需的梯度。我们通过三个基准问题，即Michaelis-Menten反应机制、靶向介导的药物分布反应机制和3D Sel'kov模型，评估了我们的PIML方法的效率。结果表明，我们提出的PIML方案能够提供等价或甚至更高精度的近似解。

    We present a physics-informed machine-learning (PIML) approach for the approximation of slow invariant manifolds (SIMs) of singularly perturbed systems, providing functionals in an explicit form that facilitate the construction and numerical integration of reduced order models (ROMs). The proposed scheme solves a partial differential equation corresponding to the invariance equation (IE) within the Geometric Singular Perturbation Theory (GSPT) framework. For the solution of the IE, we used two neural network structures, namely feedforward neural networks (FNNs), and random projection neural networks (RPNNs), with symbolic differentiation for the computation of the gradients required for the learning process. The efficiency of our PIML method is assessed via three benchmark problems, namely the Michaelis-Menten, the target mediated drug disposition reaction mechanism, and the 3D Sel'kov model. We show that the proposed PIML scheme provides approximations, of equivalent or even higher ac
    
[^82]: 增强采样方案的掩码非自回归生成建模

    Masked Generative Modeling with Enhanced Sampling Scheme. (arXiv:2309.07945v1 [cs.LG])

    [http://arxiv.org/abs/2309.07945](http://arxiv.org/abs/2309.07945)

    本文提出了一种增强的采样方案 (ESS)，用于掩码非自回归生成建模。该方案能够确保样本的多样性和保真度，并由三个阶段组成：简单迭代解码、关键反向采样和关键重采样。简单迭代解码用于采样标记集，关键反向采样和关键重采样用于掩盖不真实的标记并重建被掩盖的标记，以提高采样的保真度。

    

    本文提出了一种用于掩码非自回归生成建模的新型采样方案。我们分析了TimeVQVAE、MaskGIT和Token-Critic在采样过程中的局限性，并提出了增强采样方案 (ESS) 来克服这些限制。ESS明确确保了样本的多样性和保真度，由三个阶段组成：简单迭代解码、关键反向采样和关键重采样。ESS首先使用MaskGIT中提出的简单迭代解码来采样一个标记集，以确保样本的多样性。然后，标记集经过关键反向采样，掩盖导致不真实样本的标记。在此之后，关键重采样重建被掩盖的标记，直到达到最终采样步骤以确保高度保真度。关键重采样使用来自自我Token-Critic获得的置信度分数更好地衡量采样标记的真实性，而关键反向采样使用量化潜变量空间的结构。

    This paper presents a novel sampling scheme for masked non-autoregressive generative modeling. We identify the limitations of TimeVQVAE, MaskGIT, and Token-Critic in their sampling processes, and propose Enhanced Sampling Scheme (ESS) to overcome these limitations. ESS explicitly ensures both sample diversity and fidelity, and consists of three stages: Naive Iterative Decoding, Critical Reverse Sampling, and Critical Resampling. ESS starts by sampling a token set using the naive iterative decoding as proposed in MaskGIT, ensuring sample diversity. Then, the token set undergoes the critical reverse sampling, masking tokens leading to unrealistic samples. After that, critical resampling reconstructs masked tokens until the final sampling step is reached to ensure high fidelity. Critical resampling uses confidence scores obtained from a self-Token-Critic to better measure the realism of sampled tokens, while critical reverse sampling uses the structure of the quantized latent vector space
    
[^83]: Voxtlm: 统一的只解码模型，用于合并语音识别/合成和语音/文本补充任务

    Voxtlm: unified decoder-only models for consolidating speech recognition/synthesis and speech/text continuation tasks. (arXiv:2309.07937v1 [eess.AS])

    [http://arxiv.org/abs/2309.07937](http://arxiv.org/abs/2309.07937)

    Voxtlm是一个统一的只解码模型，能够在语音识别、语音合成、文本生成和语音延续等任务上取得显著的改善。

    

    我们提出了一个只解码语言模型VoxtLM，能够执行四个任务：语音识别、语音合成、文本生成和语音延续。VoxtLM将文本词汇与自监督语音特征中的离散语音令牌进行整合，并使用特殊令牌实现多任务学习。与单任务模型相比，VoxtLM在语音合成方面显示了显著的改善，语音可理解性从28.9提高到5.6，客观质量从2.68提高到3.90。VoxtLM还改善了语音生成和语音识别性能。VoxtLM使用公开可用的数据进行训练，并将提供训练脚本和模型检查点的开源代码，以实现完全可复现的工作。

    We propose a decoder-only language model, VoxtLM, that can perform four tasks: speech recognition, speech synthesis, text generation, and speech continuation. VoxtLM integrates text vocabulary with discrete speech tokens from self-supervised speech features and uses special tokens to enable multitask learning. Compared to a single-task model, VoxtLM exhibits a significant improvement in speech synthesis, with improvements in both speech intelligibility from 28.9 to 5.6 and objective quality from 2.68 to 3.90. VoxtLM also improves speech generation and speech recognition performance over the single-task counterpart. VoxtLM is trained with publicly available data and training recipes and model checkpoints will be open-sourced to make fully reproducible work.
    
[^84]: Landscape-Sketch-Step: 一种基于AI/ML的元启发式方法解决代理优化问题

    Landscape-Sketch-Step: An AI/ML-Based Metaheuristic for Surrogate Optimization Problems. (arXiv:2309.07936v1 [cs.LG])

    [http://arxiv.org/abs/2309.07936](http://arxiv.org/abs/2309.07936)

    Landscape-Sketch-Step是一种基于AI/ML的元启发式方法，结合了机器学习、随机优化和强化学习技术，用于解决成本函数评估昂贵、不可访问或禁止的代理优化问题。

    

    本文介绍了一种新的全局优化启发式方法，用于在成本函数的评估非常昂贵、不可访问或甚至禁止的场景下进行优化。该方法称为Landscape-Sketch-Step（LSS），结合了机器学习、随机优化和强化学习技术，依赖于先前采样点的历史信息，以明智地选择应评估成本函数的参数值。与复制交换蒙特卡洛方法相比，该方法所需的成本函数评估次数与模拟退火方法相当，这在高通量计算或高性能计算任务等环境中尤为重要，因为评估要么计算成本高昂，要么需要很长时间才能完成。该方法与标准的代理优化技术也不同，因为它不构建代理模型。

    In this paper, we introduce a new heuristics for global optimization in scenarios where extensive evaluations of the cost function are expensive, inaccessible, or even prohibitive. The method, which we call Landscape-Sketch-and-Step (LSS), combines Machine Learning, Stochastic Optimization, and Reinforcement Learning techniques, relying on historical information from previously sampled points to make judicious choices of parameter values where the cost function should be evaluated at. Unlike optimization by Replica Exchange Monte Carlo methods, the number of evaluations of the cost function required in this approach is comparable to that used by Simulated Annealing, quality that is especially important in contexts like high-throughput computing or high-performance computing tasks, where evaluations are either computationally expensive or take a long time to be performed. The method also differs from standard Surrogate Optimization techniques, for it does not construct a surrogate model
    
[^85]: 使用竞速控制变量遗传编程进行符号回归

    Racing Control Variable Genetic Programming for Symbolic Regression. (arXiv:2309.07934v1 [cs.NE])

    [http://arxiv.org/abs/2309.07934](http://arxiv.org/abs/2309.07934)

    提出了一种称为Racing Control Variable Genetic Programming (Racing-CVGP) 的方法，它通过同时进行多个实验计划来加速符号回归过程，并克服了固定实验计划选择不佳导致发现过程延迟的限制。

    

    符号回归是人工智能科学中最重要的任务之一，它从实验数据中发现控制方程。基于遗传编程、蒙特卡洛树搜索或深度强化学习的流行方法可以从固定数据集中学习符号回归。尤其是在学习涉及多个变量的复杂方程时，它们需要海量的数据集和长时间的训练。最近，引入了控制变量遗传编程（CVGP），它通过从设计的控制变量实验中发现方程来加速回归过程。但是，在CVGP中实验集是先验固定的，我们观察到实验计划的次优选择会显著延迟发现过程。为了克服这个限制，我们提出了竞速控制变量遗传编程（Racing-CVGP），它同时进行多个实验计划。类似于选择好的符号方程的选择方案被用于选择实验计划。

    Symbolic regression, as one of the most crucial tasks in AI for science, discovers governing equations from experimental data. Popular approaches based on genetic programming, Monte Carlo tree search, or deep reinforcement learning learn symbolic regression from a fixed dataset. They require massive datasets and long training time especially when learning complex equations involving many variables. Recently, Control Variable Genetic Programming (CVGP) has been introduced which accelerates the regression process by discovering equations from designed control variable experiments. However, the set of experiments is fixed a-priori in CVGP and we observe that sub-optimal selection of experiment schedules delay the discovery process significantly. To overcome this limitation, we propose Racing Control Variable Genetic Programming (Racing-CVGP), which carries out multiple experiment schedules simultaneously. A selection scheme similar to that used in selecting good symbolic equations in the 
    
[^86]: 生成型人工智能

    Generative AI. (arXiv:2309.07930v1 [cs.AI])

    [http://arxiv.org/abs/2309.07930](http://arxiv.org/abs/2309.07930)

    "生成型人工智能"指的是能够从训练数据中生成新颖有意义内容的计算技术，如文本、图像或音频。本文提供了生成型人工智能在社会技术系统中的概念，并介绍了模型、系统和应用的示例。同时，提出了当前生成型人工智能的限制，并提出了对商业与信息系统工程研究的议程，包括研究机会和挑战。

    

    "生成型人工智能"一词指的是能够从训练数据中生成看似新颖有意义的内容，如文本、图像或音频的计算技术。这种技术的广泛应用，例如Dall-E 2，GPT-4和Copilot，正在彻底改变我们工作和与他人交流的方式。在本文中，我们将生成型人工智能形容为社会技术系统中的一种实体，并提供了模型、系统和应用的示例。基于此，我们介绍了当前生成型人工智能的限制，并提出了对商业与信息系统工程（BISE）研究的议程。与以往的研究不同，我们重点讨论了信息系统背景下的生成型人工智能，并在此基础上讨论了BISE社区独特的机遇和挑战，并提出了对BISE研究的有影响的方向的建议。

    The term "generative AI" refers to computational techniques that are capable of generating seemingly new, meaningful content such as text, images, or audio from training data. The widespread diffusion of this technology with examples such as Dall-E 2, GPT-4, and Copilot is currently revolutionizing the way we work and communicate with each other. In this article, we provide a conceptualization of generative AI as an entity in socio-technical systems and provide examples of models, systems, and applications. Based on that, we introduce limitations of current generative AI and provide an agenda for Business & Information Systems Engineering (BISE) research. Different from previous works, we focus on generative AI in the context of information systems, and, to this end, we discuss several opportunities and challenges that are unique to the BISE community and make suggestions for impactful directions for BISE research.
    
[^87]: 使用声音提示进行分割的泛化音频-视觉源定位器

    Prompting Segmentation with Sound is Generalizable Audio-Visual Source Localizer. (arXiv:2309.07929v1 [cs.CV])

    [http://arxiv.org/abs/2309.07929](http://arxiv.org/abs/2309.07929)

    本研究提出了一种使用声音提示进行分割的泛化音频-视觉源定位器，在零样本和少样本情况下实现音频-视觉定位和分割任务。通过引入编码器提示解码器范式、构建语义感知音频提示和相关适配器来解决数据稀缺性和不同数据分布的困境。

    

    在从未同时看到物体和听到其声音的情况下，模型是否仍然能够准确地从输入音频中定位其视觉位置？在这项工作中，我们关注零样本和少样本情况下的音频-视觉定位和分割任务。为了实现这个目标，我们引入了编码器提示解码器的范式，与现有方法不同，现有方法主要使用编码器融合解码器范式从融合音频-视觉特征中解码定位信息，我们旨在借助预训练模型的丰富知识来更好地适应数据稀缺性和不同数据分布的困境。具体地，我们首先提出构建语义感知音频提示（SAP）来帮助视觉基础模型关注有声对象，同时也鼓励视觉和音频模态之间的语义差距缩小。然后，我们开发了一个相关适配器（ColA）来保持最小的训练工作量并维持模型性能。

    Never having seen an object and heard its sound simultaneously, can the model still accurately localize its visual position from the input audio? In this work, we concentrate on the Audio-Visual Localization and Segmentation tasks but under the demanding zero-shot and few-shot scenarios. To achieve this goal, different from existing approaches that mostly employ the encoder-fusion-decoder paradigm to decode localization information from the fused audio-visual feature, we introduce the encoder-prompt-decoder paradigm, aiming to better fit the data scarcity and varying data distribution dilemmas with the help of abundant knowledge from pre-trained models. Specifically, we first propose to construct Semantic-aware Audio Prompt (SAP) to help the visual foundation model focus on sounding objects, meanwhile, the semantic gap between the visual and audio modalities is also encouraged to shrink. Then, we develop a Correlation Adapter (ColA) to keep minimal training efforts as well as maintain 
    
[^88]: Virchow: 数百万张全数字病理学基础模型

    Virchow: A Million-Slide Digital Pathology Foundation Model. (arXiv:2309.07778v1 [eess.IV])

    [http://arxiv.org/abs/2309.07778](http://arxiv.org/abs/2309.07778)

    Virchow是一个数百万参数的深度神经网络基础模型，通过在数百万张全数字病理学切片图像上进行自监督学习训练，有效解决了计算病理学任务中数据不足的问题，并在多个下游任务上超越了最先进的系统。

    

    计算病理学利用人工智能通过分析全数字切片图像实现精准医学和决策支持系统，有潜力彻底改变癌症的诊断和治疗。然而，实现这个目标的一个主要挑战是对于许多特定的计算病理学任务，数据量不足以进行开发。为了应对这个挑战，我们创建了Virchow，一个632百万参数的深度神经网络基础模型，用于计算病理学。通过自监督学习，Virchow在1.5百万个不同组织样本的苏木精和伊红染色全数字切片图像上进行训练，这比之前的研究数据量大得多。在包括瓦片级全癌检测和亚型以及幻灯片级生物标志物预测在内的下游任务上，Virchow在来自与预训练数据相同人群的内部数据集和外部公开数据集上均胜过最先进的系统。

    Computational pathology uses artificial intelligence to enable precision medicine and decision support systems through the analysis of whole slide images. It has the potential to revolutionize the diagnosis and treatment of cancer. However, a major challenge to this objective is that for many specific computational pathology tasks the amount of data is inadequate for development. To address this challenge, we created Virchow, a 632 million parameter deep neural network foundation model for computational pathology. Using self-supervised learning, Virchow is trained on 1.5 million hematoxylin and eosin stained whole slide images from diverse tissue groups, which is orders of magnitude more data than previous works. When evaluated on downstream tasks including tile-level pan-cancer detection and subtyping and slide-level biomarker prediction, Virchow outperforms state-of-the-art systems both on internal datasets drawn from the same population as the pretraining data as well as external pu
    
[^89]: 保持结构的变压器用于序列的SPD矩阵

    Structure-Preserving Transformers for Sequences of SPD Matrices. (arXiv:2309.07579v1 [cs.LG])

    [http://arxiv.org/abs/2309.07579](http://arxiv.org/abs/2309.07579)

    本文介绍了一种保持序列的对称正定矩阵的黎曼几何特性的结构保持变压器机制，并将其应用于自动睡眠分期，取得了高水平的阶段性能。

    

    近年来，基于变压器的自注意力机制已成功应用于各种上下文相关的数据类型的分析，从文本到图像等，包括非欧几里得几何的数据。本文提出了一种这样的机制，用于分类序列的对称正定矩阵，并在整个分析过程中保持它们的黎曼几何特性。我们将我们的方法应用于来自标准数据集中的脑电图协方差矩阵序列的自动睡眠分期，取得了高水平的阶段性能。

    In recent years, Transformer-based auto-attention mechanisms have been successfully applied to the analysis of a variety of context-reliant data types, from texts to images and beyond, including data from non-Euclidean geometries. In this paper, we present such a mechanism, designed to classify sequences of Symmetric Positive Definite matrices while preserving their Riemannian geometry throughout the analysis. We apply our method to automatic sleep staging on timeseries of EEG-derived covariance matrices from a standard dataset, obtaining high levels of stage-wise performance.
    
[^90]: 复合移位算子的频率收敛问题

    Frequency Convergence of Complexon Shift Operators. (arXiv:2309.07169v1 [eess.SP])

    [http://arxiv.org/abs/2309.07169](http://arxiv.org/abs/2309.07169)

    本文研究了拓扑信号处理中复合子的可转移性，通过构造边际复合子和复合移位算子，研究其特征值和特征向量，并证明了复合子收敛时对应的复合移位算子的特征值会收敛到极限复合子的特征值。这些结果拓展了图信号处理框架。

    

    拓扑信号处理(TSP)利用单纯形复合来建模比顶点和边更高阶的结构。本文研究了TSP的可转移性，通过一种称为复合子的广义高阶图的版本。我们回顾了复合子的概念，即单纯形复合序列的极限[1]。受图移位算子的积分算子形式的启发，我们根据复合子的所有可能尺寸的组件构造了边际复合子和复合移位算子(CSO)。我们研究了CSO的特征值和特征向量，并将它们与一类新的加权邻接矩阵相关联。我们证明，当一个单纯形复合序列收敛到一个复合子时，相应的CSO的特征值收敛到极限复合子的特征值。这些结果暗示了在大型单纯形复合或单纯形复合序列上的学习可转移性，从而推广了图信号处理框架。

    Topological signal processing (TSP) utilizes simplicial complexes to model structures with higher order than vertices and edges. In this paper, we study the transferability of TSP via a generalized higher-order version of graphon, known as complexon. We recall the notion of a complexon as the limit of a simplicial complex sequence [1]. Inspired by the integral operator form of graphon shift operators, we construct a marginal complexon and complexon shift operator (CSO) according to components of all possible dimensions from the complexon. We investigate the CSO's eigenvalues and eigenvectors, and relate them to a new family of weighted adjacency matrices. We prove that when a simplicial complex sequence converges to a complexon, the eigenvalues of the corresponding CSOs converge to that of the limit complexon. These results hint at learning transferability on large simplicial complexes or simplicial complex sequences, which generalize the graphon signal processing framework.
    
[^91]: 无集合的人工智能

    Collectionless Artificial Intelligence. (arXiv:2309.06938v1 [cs.AI])

    [http://arxiv.org/abs/2309.06938](http://arxiv.org/abs/2309.06938)

    本文提出了无集合原则的学习协议的思路，其中机器在环境交互背景中掌握认知技能，避免了数据集集中化的风险。

    

    大体上，处理庞大数据集被认为是机器学习进展和相关领域中壮观结果的基本组成部分，对于这种数据集的集中化存在着越来越多的风险意识。本文支持一种新的学习协议思路，其中机器在真正以环境交互为中心的类人认知背景下掌握认知技能。这意味着学习协议需要遵循无集合原则，即在每个时间点，从环境中获取的数据被用于更新当前环境内部表示，并且代理不能对时间流进行记录。基本上，不能存储来自传感器的时间信息，从而促进了无集合原则的发展。

    By and large, the professional handling of huge data collections is regarded as a fundamental ingredient of the progress of machine learning and of its spectacular results in related disciplines, with a growing agreement on risks connected to the centralization of such data collections. This paper sustains the position that the time has come for thinking of new learning protocols where machines conquer cognitive skills in a truly human-like context centered on environmental interactions. This comes with specific restrictions on the learning protocol according to the collectionless principle, which states that, at each time instant, data acquired from the environment is processed with the purpose of contributing to update the current internal representation of the environment, and that the agent is not given the privilege of recording the temporal stream. Basically, there is neither permission to store the temporal information coming from the sensors, thus promoting the development of s
    
[^92]: 缺失数据下的不确定性交通预测

    Uncertainty-aware Traffic Prediction under Missing Data. (arXiv:2309.06800v1 [cs.LG])

    [http://arxiv.org/abs/2309.06800](http://arxiv.org/abs/2309.06800)

    本研究提出了一种考虑不确定性的交通预测方法，可以处理缺失数据和测量不确定性，并适用于风险敏感任务和决策导向问题。

    

    交通预测是一个重要的课题，因为它在交通领域有广泛的应用。近期，许多研究取得了很好的结果。然而，大多数研究假设预测位置有完整或至少部分的历史记录，不能扩展到无历史记录的位置。在现实场景中，由于预算限制和安装可行性问题，传感器的部署可能受限，这使得大多数当前模型不适用。虽然少数文献尝试在缺失位置上插补交通状态，但这些方法需要与传感器位置同时观测的数据，使它们不适用于预测任务。另一个缺点是缺乏对预测不确定性的测量，使得之前的工作不适用于风险敏感的任务或涉及决策的情况。为了填补这一空白，受到先前的归纳图神经网络的启发，本文提出了一种考虑不确定性的方法。

    Traffic prediction is a crucial topic because of its broad scope of applications in the transportation domain. Recently, various studies have achieved promising results. However, most studies assume the prediction locations have complete or at least partial historical records and cannot be extended to non-historical recorded locations. In real-life scenarios, the deployment of sensors could be limited due to budget limitations and installation availability, which makes most current models not applicable. Though few pieces of literature tried to impute traffic states at the missing locations, these methods need the data simultaneously observed at the locations with sensors, making them not applicable to prediction tasks. Another drawback is the lack of measurement of uncertainty in prediction, making prior works unsuitable for risk-sensitive tasks or involving decision-making. To fill the gap, inspired by the previous inductive graph neural network, this work proposed an uncertainty-awa
    
[^93]: Gradient-based MAML在LQR中的收敛性研究

    Convergence of Gradient-based MAML in LQR. (arXiv:2309.06588v1 [eess.SY])

    [http://arxiv.org/abs/2309.06588](http://arxiv.org/abs/2309.06588)

    本研究调查了在LQR中应用MAML时的局部收敛特性，并提供了保持动态系统稳定性的局部收敛保证。论文通过简单的数值结果展示了MAML在LQR任务中的收敛性质。

    

    本研究的主要目标是探索在线性系统二次优化控制（LQR）中应用Model-agnostic Meta-learning（MAML）时的局部收敛特性。MAML及其变体已成为快速适应新任务的流行技术，通过利用在回归、分类和强化学习等领域的先前学习知识。然而，由于非凸性和其结构，MAML的理论保证仍然未知，这使得在动态系统设置中确保稳定性更具挑战性。本研究重点研究了MAML在LQR设置中的局部收敛性保证，同时保持动态系统的稳定性。该论文还提供了简单的数值结果，以展示MAML在LQR任务中的收敛性质。

    The main objective of this research paper is to investigate the local convergence characteristics of Model-agnostic Meta-learning (MAML) when applied to linear system quadratic optimal control (LQR). MAML and its variations have become popular techniques for quickly adapting to new tasks by leveraging previous learning knowledge in areas like regression, classification, and reinforcement learning. However, its theoretical guarantees remain unknown due to non-convexity and its structure, making it even more challenging to ensure stability in the dynamic system setting. This study focuses on exploring MAML in the LQR setting, providing its local convergence guarantees while maintaining the stability of the dynamical system. The paper also presents simple numerical results to demonstrate the convergence properties of MAML in LQR tasks.
    
[^94]: 可解释的图神经网络用于阿尔茨海默病和相关痴呆症风险预测

    Explainable Graph Neural Network for Alzheimer's Disease And Related Dementias Risk Prediction. (arXiv:2309.06584v1 [cs.LG])

    [http://arxiv.org/abs/2309.06584](http://arxiv.org/abs/2309.06584)

    这项研究提出了一种可解释的图神经网络方法来预测阿尔茨海默病和相关痴呆症的风险。通过将机器学习与索赔数据相结合，不仅能发现额外的风险因素，还能揭示不同医学代码之间的关联。通过评估关系重要性和其对风险预测的影响，该方法能提供全面的解释。

    

    阿尔茨海默病和相关痴呆症（ADRD）在美国是第六大死亡原因，准确的ADRD风险预测具有重要意义。虽然最近在ADRD风险预测方面取得了一定进展，但大部分依赖于图像分析，而并非所有患者在ADRD诊断前都接受医学影像检查。将机器学习与索赔数据相结合可以揭示额外的风险因素并发现不同医学代码之间的相互关联。我们的目标是利用图神经网络（GNN）和索赔数据进行ADRD风险预测。为了解决这些预测背后缺乏可解释原因的问题，我们引入了一种创新方法来评估关系重要性及其对ADRD风险预测的影响，确保全面解释。我们使用变分正则化编码器-解码器图神经网络（VGNN）来估计ADRD可能性。我们创建了三种情景来评估模型的效率，使用了随机森林和轻梯度...

    Alzheimer's disease and related dementias (ADRD) ranks as the sixth leading cause of death in the US, underlining the importance of accurate ADRD risk prediction. While recent advancement in ADRD risk prediction have primarily relied on imaging analysis, yet not all patients undergo medical imaging before an ADRD diagnosis. Merging machine learning with claims data can reveal additional risk factors and uncover interconnections among diverse medical codes. Our goal is to utilize Graph Neural Networks (GNNs) with claims data for ADRD risk prediction. Addressing the lack of human-interpretable reasons behind these predictions, we introduce an innovative method to evaluate relationship importance and its influence on ADRD risk prediction, ensuring comprehensive interpretation.  We employed Variationally Regularized Encoder-decoder Graph Neural Network (VGNN) for estimating ADRD likelihood. We created three scenarios to assess the model's efficiency, using Random Forest and Light Gradient 
    
[^95]: 适用于拜占庭式机器学习的实用同态聚合

    Practical Homomorphic Aggregation for Byzantine ML. (arXiv:2309.05395v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.05395](http://arxiv.org/abs/2309.05395)

    本文介绍了一种适用于拜占庭式机器学习的实用同态聚合方法，以应对拜占庭节点和服务器隐私侵犯的问题。

    

    由于数据的大规模可用性，机器学习算法正在分布式拓扑中部署，不同的节点通过与中央服务器交换与模型相关的信息（例如梯度）来共同训练其个体数据上的机器学习模型。然而，分布式学习方案容易受到两种威胁。首先，拜占庭式节点可以通过向服务器发送不正确的信息（例如错误的梯度）单独破坏学习过程。缓解此类行为的标准方法是在服务器上使用非线性鲁棒聚合方法。其次，服务器可以侵犯节点的隐私。最近的攻击已经表明，交换（未加密的）梯度使得一个好奇的服务器能够恢复出所有节点的数据。同态加密（HE），一种金标准安全原语，已经广泛研究作为非拜占庭场景中分布式学习的隐私保护解决方案。

    Due to the large-scale availability of data, machine learning (ML) algorithms are being deployed in distributed topologies, where different nodes collaborate to train ML models over their individual data by exchanging model-related information (e.g., gradients) with a central server. However, distributed learning schemes are notably vulnerable to two threats. First, Byzantine nodes can single-handedly corrupt the learning by sending incorrect information to the server, e.g., erroneous gradients. The standard approach to mitigate such behavior is to use a non-linear robust aggregation method at the server. Second, the server can violate the privacy of the nodes. Recent attacks have shown that exchanging (unencrypted) gradients enables a curious server to recover the totality of the nodes' data. The use of homomorphic encryption (HE), a gold standard security primitive, has extensively been studied as a privacy-preserving solution to distributed learning in non-Byzantine scenarios. Howev
    
[^96]: CenTime: 事件条件模型在生存分析中的应用

    CenTime: Event-Conditional Modelling of Censoring in Survival Analysis. (arXiv:2309.03851v1 [cs.LG])

    [http://arxiv.org/abs/2309.03851](http://arxiv.org/abs/2309.03851)

    CenTime是一种新的生存分析方法，通过创新的事件条件审查机制直接估计事件发生的时间，在处理未被审查的数据时具有良好的鲁棒性和准确性。

    

    生存分析是一种有价值的工具，可以基于基线观测来估计特定事件（如死亡或癌症复发）发生的时间。这在医疗保健中非常有用，可以根据患者数据预测临床重要事件的预后。然而，现有方法常常存在局限性；有些方法只关注将患者按生存能力进行排名，忽视了对实际事件时间的估计；而其他方法将问题视为分类任务，忽视了事件的时间顺序结构。此外，有效利用被审查样本（训练数据点，其中确切事件时间不可知）对于提高模型的预测准确性至关重要。在本文中，我们引入了CenTime，一种新的生存分析方法，直接估计事件发生的时间。我们的方法具有创新的事件条件审查机制，即使没有未被审查的数据，也能表现出良好的鲁棒性。我们证明了我们的方法在准确性上的优势。

    Survival analysis is a valuable tool for estimating the time until specific events, such as death or cancer recurrence, based on baseline observations. This is particularly useful in healthcare to prognostically predict clinically important events based on patient data. However, existing approaches often have limitations; some focus only on ranking patients by survivability, neglecting to estimate the actual event time, while others treat the problem as a classification task, ignoring the inherent time-ordered structure of the events. Furthermore, the effective utilization of censored samples - training data points where the exact event time is unknown - is essential for improving the predictive accuracy of the model. In this paper, we introduce CenTime, a novel approach to survival analysis that directly estimates the time to event. Our method features an innovative event-conditional censoring mechanism that performs robustly even when uncensored data is scarce. We demonstrate that ou
    
[^97]: 通过递归分解实现可扩展学习入侵响应

    Scalable Learning of Intrusion Responses through Recursive Decomposition. (arXiv:2309.03292v1 [eess.SY])

    [http://arxiv.org/abs/2309.03292](http://arxiv.org/abs/2309.03292)

    本文提出了一种通过递归分解方法实现可扩展学习入侵响应的技术，该技术通过解决并行子游戏和计算阈值结构的最佳响应策略来提高效率。

    

    我们研究了针对IT基础设施的自动化入侵应对，并将攻击者和防御者之间的交互形式建模为部分观测的随机游戏。为了解决这个游戏，我们采用了一种方法，攻击和防御策略通过强化学习和自我对弈进行协同演化，以达到平衡。之前的研究中提出的解决方案证明了这种方法对于小型基础设施的可行性，但面对实际情境由于基础设施规模的指数级增长而无法扩展。为了解决这个问题，我们提出了一种将游戏递归分解成可以并行解决的子游戏的方法。应用最优停止理论，我们证明了这些子游戏中的最佳响应策略具有阈值结构，这允许我们高效地计算它们。为了解决分解的游戏，我们提出了一个名为Decompositional Fictitious Self-Play (DFSP) 的算法，通过随机自我对弈学习纳什均衡。

    We study automated intrusion response for an IT infrastructure and formulate the interaction between an attacker and a defender as a partially observed stochastic game. To solve the game we follow an approach where attack and defense strategies co-evolve through reinforcement learning and self-play toward an equilibrium. Solutions proposed in previous work prove the feasibility of this approach for small infrastructures but do not scale to realistic scenarios due to the exponential growth in computational complexity with the infrastructure size. We address this problem by introducing a method that recursively decomposes the game into subgames which can be solved in parallel. Applying optimal stopping theory we show that the best response strategies in these subgames exhibit threshold structures, which allows us to compute them efficiently. To solve the decomposed game we introduce an algorithm called Decompositional Fictitious Self-Play (DFSP), which learns Nash equilibria through stoc
    
[^98]: PROMISE: 通过引入可扩展曲率估计的预条件随机优化方法

    PROMISE: Preconditioned Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates. (arXiv:2309.02014v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2309.02014](http://arxiv.org/abs/2309.02014)

    本文介绍了一套基于草图的预条件随机梯度算法套件PROMISE，用于解决大规模凸优化问题，相比传统方法，在默认超参数下在机器学习问题上表现更优。

    

    本文介绍了PROMISE（通过引入可扩展曲率估计的预条件随机优化方法），这是一套基于草图的预条件随机梯度算法套件，用于解决机器学习中出现的大规模凸优化问题。PROMISE包括SVRG、SAGA和Katyusha的预条件版本；每个算法都有强大的理论分析和有效的默认超参数值。相比之下，传统的随机梯度方法需要仔细调节超参数才能成功，在机器学习中普遍存在的病态条件下性能会下降。实验证明，通过使用默认超参数值，所提出的算法在由基准机器学习问题汇编的51个岭回归和逻辑回归问题上优于或与流行的调整后的随机梯度优化器相匹配。

    This paper introduces PROMISE ($\textbf{Pr}$econditioned Stochastic $\textbf{O}$ptimization $\textbf{M}$ethods by $\textbf{I}$ncorporating $\textbf{S}$calable Curvature $\textbf{E}$stimates), a suite of sketching-based preconditioned stochastic gradient algorithms for solving large-scale convex optimization problems arising in machine learning. PROMISE includes preconditioned versions of SVRG, SAGA, and Katyusha; each algorithm comes with a strong theoretical analysis and effective default hyperparameter values. In contrast, traditional stochastic gradient methods require careful hyperparameter tuning to succeed, and degrade in the presence of ill-conditioning, a ubiquitous phenomenon in machine learning. Empirically, we verify the superiority of the proposed algorithms by showing that, using default hyperparameter values, they outperform or match popular tuned stochastic gradient optimizers on a test bed of $51$ ridge and logistic regression problems assembled from benchmark machine l
    
[^99]: 通过稀疏细胞复合体在图上表示边流

    Representing Edge Flows on Graphs via Sparse Cell Complexes. (arXiv:2309.01632v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2309.01632](http://arxiv.org/abs/2309.01632)

    本文针对图边缘的流量数据，提出了一种通过稀疏细胞复合体来表示边流的方法。我们将图结构转化为一个单纯复合体，利用Hodge-Laplacian的特征向量和关联矩阵进行Hodge分解，得到梯度、旋量和谐波流的表示。同时，我们引入了细胞推断优化问题，通过添加细胞来增强观测到的图，使表示稀疏可解释。实验证明这个问题是NP难的，我们提出了一个高效的近似算法。

    

    在许多机器学习和信号处理任务中，获取稀疏可解释的可观测数据表示是至关重要的。对于表示沿图边缘的流动的数据，一种直观可解释的获取表示的方法是将图结构提升到一个单纯复合体：相关Hodge-Laplacian的特征向量，以及相应单纯复合体的关联矩阵，可引导出Hodge分解，用于以梯度，旋量和谐波流的形式表示观测到的数据。在本文中，我们将这种方法推广到细胞复合体，并引入细胞推断优化问题，即通过添加一组细胞来增强观测到的图，使得关联Hodge Laplacian的特征向量能够提供对图上观测到的边缘流的稀疏可解释表示。我们证明了这个问题是NP难的，并引入了一个高效的近似算法。

    Obtaining sparse, interpretable representations of observable data is crucial in many machine learning and signal processing tasks. For data representing flows along the edges of a graph, an intuitively interpretable way to obtain such representations is to lift the graph structure to a simplicial complex: The eigenvectors of the associated Hodge-Laplacian, respectively the incidence matrices of the corresponding simplicial complex then induce a Hodge decomposition, which can be used to represent the observed data in terms of gradient, curl, and harmonic flows. In this paper, we generalize this approach to cellular complexes and introduce the cell inference optimization problem, i.e., the problem of augmenting the observed graph by a set of cells, such that the eigenvectors of the associated Hodge Laplacian provide a sparse, interpretable representation of the observed edge flows on the graph. We show that this problem is NP-hard and introduce an efficient approximation algorithm for i
    
[^100]: 符号性地整合不同随机张量的张量网络计算 - Python RTNI的第二版本

    Symbolically integrating tensor networks over various random tensors -- the second version of Python RTNI. (arXiv:2309.01167v2 [physics.comp-ph] UPDATED)

    [http://arxiv.org/abs/2309.01167](http://arxiv.org/abs/2309.01167)

    我们升级了Python RTNI的第二版，可以对不同随机张量进行符号性整合，支持Haar分布的酉矩阵、正交矩阵和正态分布的张量。通过导出TensorNetwork格式的张量网络，可以进行低维计算，并解释了数学原理和张量网络图之间的关系。

    

    我们正在升级RTNI的Python版本，该版本能够符号性地整合Haar分布的酉矩阵上的张量网络。现在，PyRTNI2还可以处理Haar分布的正交矩阵以及实数和复数正态分布的张量。此外，它可以将张量网络以TensorNetwork的格式导出，这样可以使用具体的张量进行进一步计算，即使是低维情况下的计算，其中Weingarten函数与高维情况下的函数不同。教程笔记本可以在GitHub上找到：https://github.com/MotohisaFukuda/PyRTNI2。在本文中，我们解释了程序背后的数学原理，并展示了可以使用它进行的各种张量网络计算。关于前者，我们将上述随机矩阵和张量的逐元素矩阵微积分解释为张量网络图，认为这种观点是自然的，将微积分中的delta函数与张量网络图中的边相关联。

    We are upgrading the Python-version of RTNI, which symbolically integrates tensor networks over the Haar-distributed unitary matrices. Now, PyRTNI2 can treat the Haar-distributed orthogonal matrices and the real and complex normal Gaussian tensors as well. Moreover, it can export tensor networks in the format of TensorNetwork so that one can make further calculations with concrete tensors, even for low dimensions, where the Weingarten functions differ from the ones for high dimensions. The tutorial notebooks are found at GitHub: https://github.com/MotohisaFukuda/PyRTNI2. In this paper, we explain maths behind the program and show what kind of tensor network calculations can be made with it. For the former, we interpret the element-wise moment calculus of the above random matrices and tensors in terms of tensor network diagrams, and argue that the view is natural, relating delta functions in the calculus to edges in tensor network diagrams.
    
[^101]: 用于人工智能和机器学习应用的Majorana示范器数据发布

    Majorana Demonstrator Data Release for AI/ML Applications. (arXiv:2308.10856v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2308.10856](http://arxiv.org/abs/2308.10856)

    此数据发布包含Majorana示范器实验的校准数据子集，旨在支持人工智能和机器学习算法在该数据上的训练和测试。

    

    此数据发布包含Majorana示范器实验的校准数据子集。每个Majorana事件都有原始的锗探测器波形、脉冲形状识别切割和校准后的最终能量，所有这些数据以HDF5文件格式与相关元数据一同分享。此发布旨在支持对我们的数据进行人工智能（AI）和机器学习（ML）算法的训练和测试。

    The enclosed data release consists of a subset of the calibration data from the Majorana Demonstrator experiment. Each Majorana event is accompanied by raw Germanium detector waveforms, pulse shape discrimination cuts, and calibrated final energies, all shared in an HDF5 file format along with relevant metadata. This release is specifically designed to support the training and testing of Artificial Intelligence (AI) and Machine Learning (ML) algorithms upon our data. This document is structured as follows. Section I provides an overview of the dataset's content and format; Section II outlines the location of this dataset and the method for accessing it; Section III presents the NPML Machine Learning Challenge associated with this dataset; Section IV contains a disclaimer from the Majorana collaboration regarding the use of this dataset; Appendix A contains technical details of this data release. Please direct questions about the material provided within this release to liaobo77@ucsd.ed
    
[^102]: MindMap：知识图谱激发大型语言模型的思维图思考方法

    MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models. (arXiv:2308.09729v1 [cs.AI])

    [http://arxiv.org/abs/2308.09729](http://arxiv.org/abs/2308.09729)

    本论文通过使用知识图谱来激发大型语言模型，解决了整合新知识、产生幻觉和决策过程不透明等问题，并通过生成思维导图展示了模型的推理路径，实验证明这种方法可以取得显著的实证增益。

    

    通常，大型语言模型存在无法整合新知识、产生幻觉和决策过程不透明等限制。本文探讨了如何利用知识图谱（KG）来激发大型语言模型，以解决整合最新知识和引发模型思维路径的问题。具体来说，我们构建了一个提示管道，使大型语言模型能够理解KG输入并利用隐含知识和检索到的外部知识进行推理。此外，我们研究了引发大型语言模型执行推理和生成答案的思维导图。研究发现，生成的思维导图基于知识的本体论，展示了大型语言模型的推理路径，从而为生产环境中的推理提供了探索和评估的可能性。对三个问答数据集的实验证明，MindMap提示方法带来了显著的实证增益。

    LLMs usually exhibit limitations in their ability to incorporate new knowledge, the generation of hallucinations, and the transparency of their decision-making process. In this paper, we explore how to prompt LLMs with knowledge graphs (KG), working as a remedy to engage LLMs with up-to-date knowledge and elicit the reasoning pathways from LLMs. Specifically, we build a prompting pipeline that endows LLMs with the capability of comprehending KG inputs and inferring with a combined implicit knowledge and the retrieved external knowledge. In addition, we investigate eliciting the mind map on which LLMs perform the reasoning and generate the answers. It is identified that the produced mind map exhibits the reasoning pathways of LLMs grounded on the ontology of knowledge, hence bringing the prospects of probing and gauging LLM inference in production. The experiments on three question & answering datasets also show that MindMap prompting leads to a striking empirical gain. For instance, pr
    
[^103]: 特征强化物理信息神经网络（FE-PINN）：在目标任务之前学习底层物理特征的框架

    Feature Enforcing PINN (FE-PINN): A Framework to Learn the Underlying-Physics Features Before Target Task. (arXiv:2308.08873v1 [cs.LG])

    [http://arxiv.org/abs/2308.08873](http://arxiv.org/abs/2308.08873)

    FE-PINN是一种学习底层物理特征的框架，在主训练之前以低计算成本解决问题的模式。与传统PINN相比，FE-PINN通过执行一系列子任务来解决损失函数不平衡的问题，并具有快速训练和更高的求解速度。

    

    本文介绍了一种名为特征强化物理信息神经网络（FE-PINN）的新型无数据框架。该框架能够在主训练循环之前以较低的计算成本学习任何问题的底层模式。由于存在偏微分残差和边界条件均方误差两个项，普通PINN的损失函数不平衡。FE-PINN通过只需一分钟的训练，而不是耗时数小时的超参数调优来解决这个挑战。FE-PINN通过执行一系列子任务来完成这个过程。第一个子任务学习有关底层物理的有用特征。然后，模型在目标任务上进行训练以完善计算。FE-PINN应用于三个基准问题：圆柱体上的流动、二维热传导以及计算入口速度的逆问题。FE-PINN可以分别加速15倍、2倍和5倍地解决每个案例。另外

    In this work, a new data-free framework called Feature Enforcing Physics Informed Neural Network (FE-PINN) is introduced. This framework is capable of learning the underlying pattern of any problem with low computational cost before the main training loop. The loss function of vanilla PINN due to the existence of two terms of partial differential residuals and boundary condition mean squared error is imbalanced. FE-PINN solves this challenge with just one minute of training instead of time-consuming hyperparameter tuning for loss function that can take hours. The FE-PINN accomplishes this process by performing a sequence of sub-tasks. The first sub-task learns useful features about the underlying physics. Then, the model trains on the target task to refine the calculations. FE-PINN is applied to three benchmarks, flow over a cylinder, 2D heat conduction, and an inverse problem of calculating inlet velocity. FE-PINN can solve each case with, 15x, 2x, and 5x speed up accordingly. Another
    
[^104]: LLM4TS:使用预训练的LLM进行两阶段微调用于时间序列预测

    LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs. (arXiv:2308.08469v1 [cs.LG])

    [http://arxiv.org/abs/2308.08469](http://arxiv.org/abs/2308.08469)

    这项工作提出了LLM4TS方法，利用预训练的LLMs增强时间序列预测能力。通过两阶段微调和参数高效微调，提高了LLMs处理时间序列数据的能力。

    

    在这项工作中，我们利用预训练的大型语言模型（LLMs）来增强时间序列预测。借鉴了自然语言处理和计算机视觉统一模型的日益增长的兴趣，我们设想创建一个类似的模型用于长期时间序列预测。由于缺乏大规模的时间序列数据来构建稳健的基础模型，我们的方法LLM4TS专注于利用预训练的LLMs的优势。通过将时间序列修补与时间编码相结合，我们提高了LLMs处理时间序列数据的能力。受到聊天机器人领域的有监督微调的启发，我们优先进行两阶段的微调过程：首先进行有监督微调以使LLMs适应时间序列数据，然后进行任务特定的下游微调。此外，为了在不进行大量参数调整的情况下发挥预训练LLMs的灵活性，我们采用了几种参数高效微调（PEFT）技术。

    In this work, we leverage pre-trained Large Language Models (LLMs) to enhance time-series forecasting. Mirroring the growing interest in unifying models for Natural Language Processing and Computer Vision, we envision creating an analogous model for long-term time-series forecasting. Due to limited large-scale time-series data for building robust foundation models, our approach LLM4TS focuses on leveraging the strengths of pre-trained LLMs. By combining time-series patching with temporal encoding, we have enhanced the capability of LLMs to handle time-series data effectively. Inspired by the supervised fine-tuning in chatbot domains, we prioritize a two-stage fine-tuning process: first conducting supervised fine-tuning to orient the LLM towards time-series data, followed by task-specific downstream fine-tuning. Furthermore, to unlock the flexibility of pre-trained LLMs without extensive parameter adjustments, we adopt several Parameter-Efficient Fine-Tuning (PEFT) techniques. Drawing o
    
[^105]: DCNFIS：深度卷积神经模糊推理系统

    DCNFIS: Deep Convolutional Neuro-Fuzzy Inference System. (arXiv:2308.06378v1 [cs.AI])

    [http://arxiv.org/abs/2308.06378](http://arxiv.org/abs/2308.06378)

    本文介绍了一种新的深度卷积神经模糊推理系统（DCNFIS），它通过将模糊逻辑和深度学习模型相结合，实现了提高透明度而不损失准确性的目标。DCNFIS在准确性上与现有卷积神经网络相当，并且胜过了最先进的深度模糊系统。通过模糊规则提取的解释可以提高模型的可解释性。

    

    在可解释的人工智能中，透明度与准确性之间存在一个著名的权衡。本文介绍了一种新的深度网络设计，通过将模糊逻辑和深度学习模型相结合，实现了提高透明度但不损失准确性的目标。我们设计了一个深度卷积神经模糊推理系统（DCNFIS），并在四个著名数据集上展示了它与三个现有卷积神经网络的相同准确性。我们进一步发现，DCNFIS在性能上胜过了最先进的深度模糊系统。然后，我们利用模糊逻辑的透明度，从DCNFIS中编码的模糊规则中提取解释，以渐变映射的形式展示。我们还利用Fashion-MNIST数据集对这些解释的特性进行了深入研究。

    A key challenge in eXplainable Artificial Intelligence is the well-known tradeoff between the transparency of an algorithm (i.e., how easily a human can directly understand the algorithm, as opposed to receiving a post-hoc explanation), and its accuracy. We report on the design of a new deep network that achieves improved transparency without sacrificing accuracy. We design a deep convolutional neuro-fuzzy inference system (DCNFIS) by hybridizing fuzzy logic and deep learning models and show that DCNFIS performs as accurately as three existing convolutional neural networks on four well-known datasets. We furthermore that DCNFIS outperforms state-of-the-art deep fuzzy systems. We then exploit the transparency of fuzzy logic by deriving explanations, in the form of saliency maps, from the fuzzy rules encoded in DCNFIS. We investigate the properties of these explanations in greater depth using the Fashion-MNIST dataset.
    
[^106]: 可转移的图神经指纹模型快速应对未来生物威胁

    Transferable Graph Neural Fingerprint Models for Quick Response to Future Bio-Threats. (arXiv:2308.01921v1 [q-bio.BM])

    [http://arxiv.org/abs/2308.01921](http://arxiv.org/abs/2308.01921)

    该论文提出了一种可转移的图神经指纹模型，用于快速应对未来的生物威胁。通过利用包含30万种候选药物和23个冠状病毒蛋白靶的COVID-19药物对接数据集，训练了高通量虚拟COVID-19药物筛选的图神经指纹模型。与传统指纹方法相比，该模型在对接得分上具有较高的预测准确性，并且提出了可转移的图神经指纹方法，能够适用于未知的靶点。

    

    基于配体结合亲和力的药物分子快速筛选是药物发现管线中的重要步骤。图神经指纹是一种用于开发高通量和高准确性分子对接代理的有希望方法。在这项研究中，我们建立了一个包含约30万种药物候选物和23个冠状病毒蛋白靶的COVID-19药物对接数据集。利用这个数据集，我们训练了图神经指纹对接模型，用于高通量虚拟COVID-19药物筛选。图神经指纹模型在对接得分上具有很高的预测准确性，对大多数对接靶点的均方误差低于0.21 kcal/mol，相比传统圆形指纹方法有显著改进。为了使神经指纹适用于未知的靶点，我们还提出了一种在多个靶点上训练的可转移的图神经指纹方法。

    Fast screening of drug molecules based on the ligand binding affinity is an important step in the drug discovery pipeline. Graph neural fingerprint is a promising method for developing molecular docking surrogates with high throughput and great fidelity. In this study, we built a COVID-19 drug docking dataset of about 300,000 drug candidates on 23 coronavirus protein targets. With this dataset, we trained graph neural fingerprint docking models for high-throughput virtual COVID-19 drug screening. The graph neural fingerprint models yield high prediction accuracy on docking scores with the mean squared error lower than $0.21$ kcal/mol for most of the docking targets, showing significant improvement over conventional circular fingerprint methods. To make the neural fingerprints transferable for unknown targets, we also propose a transferable graph neural fingerprint method trained on multiple targets. With comparable accuracy to target-specific graph neural fingerprint models, the transf
    
[^107]: 用深度学习辅助自动检测实现头部计算机断层成像重建标准化

    Towards Head Computed Tomography Image Reconstruction Standardization with Deep Learning Assisted Automatic Detection. (arXiv:2307.16440v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2307.16440](http://arxiv.org/abs/2307.16440)

    提出了一种用深度学习辅助自动检测实现头部CT图像三维重建的方法，该方法可以提高重建的准确性和可重复性，减少了手动干预。通过识别和评估眶下线标志点，实现在重建之前自动重新格式化图像。

    

    头部计算机断层成像(CT)图像的三维重建揭示了组织结构的复杂空间关系，从而帮助准确诊断。然而，在临床环境中，由于技术人员摆位不良、患者身体限制或CT扫描仪倾斜角度限制，获得理想的头部CT扫描是具有挑战性的。手动格式化和重建不仅引入主观性，而且耗费时间和劳动资源。为了解决这些问题，我们提出了一种高效的自动头部CT图像三维重建方法，提高了准确性和可重复性，减少了手动干预。我们的方法利用基于深度学习的物体检测算法，识别和评估眶下线标志点，以在重建之前自动重新格式化图像。鉴于关于头部CT图像背景下物体检测算法的评估较少，我们进行了比较。

    Three-dimensional (3D) reconstruction of head Computed Tomography (CT) images elucidates the intricate spatial relationships of tissue structures, thereby assisting in accurate diagnosis. Nonetheless, securing an optimal head CT scan without deviation is challenging in clinical settings, owing to poor positioning by technicians, patient's physical constraints, or CT scanner tilt angle restrictions. Manual formatting and reconstruction not only introduce subjectivity but also strain time and labor resources. To address these issues, we propose an efficient automatic head CT images 3D reconstruction method, improving accuracy and repeatability, as well as diminishing manual intervention. Our approach employs a deep learning-based object detection algorithm, identifying and evaluating orbitomeatal line landmarks to automatically reformat the images prior to reconstruction. Given the dearth of existing evaluations of object detection algorithms in the context of head CT images, we compared
    
[^108]: 通过神经多项式方法实现可解释的弹性塑性模型的发现

    Discovering interpretable elastoplasticity models via the neural polynomial method enabled symbolic regressions. (arXiv:2307.13149v1 [cs.CE])

    [http://arxiv.org/abs/2307.13149](http://arxiv.org/abs/2307.13149)

    本文介绍了一种通过神经多项式方法实现可解释的弹性塑性模型的机器学习方法，该方法通过分为两个步骤，先通过监督学习得到一组特征映射，再通过符号回归将其转化为数学公式，从而克服了传统神经网络模型的缺乏可解释性的问题。

    

    传统神经网络弹性塑性模型通常被认为缺乏可解释性。本文介绍了一种两步机器学习方法，可以返回专家可解释的数学模型。具体而言，我们引入了一个替代模型，其中屈服曲面是通过监督学习得到的一组单变量特征映射来表示的。然后，通过符号回归将这组单变量神经网络映射函数重新解释为数学形式。这种分而治之的方法具有几个重要优势。首先，它使我们能够克服符号回归算法的扩展问题。从实际角度来看，它提高了用不同编程语言编写的偏微分方程求解器的学习模型的可移植性。最后，它使我们能够对材料的属性（如凸性和对称性）有一个具体的理解。

    Conventional neural network elastoplasticity models are often perceived as lacking interpretability. This paper introduces a two-step machine-learning approach that returns mathematical models interpretable by human experts. In particular, we introduce a surrogate model where yield surfaces are expressed in terms of a set of single-variable feature mappings obtained from supervised learning. A postprocessing step is then used to re-interpret the set of single-variable neural network mapping functions into mathematical form through symbolic regression. This divide-and-conquer approach provides several important advantages. First, it enables us to overcome the scaling issue of symbolic regression algorithms. From a practical perspective, it enhances the portability of learned models for partial differential equation solvers written in different programming languages. Finally, it enables us to have a concrete understanding of the attributes of the materials, such as convexity and symmetri
    
[^109]: 政策梯度最优相关搜索用于蒙特卡洛模拟和最大最优传输中的方差降低

    Policy Gradient Optimal Correlation Search for Variance Reduction in Monte Carlo simulation and Maximum Optimal Transport. (arXiv:2307.12703v1 [stat.ML])

    [http://arxiv.org/abs/2307.12703](http://arxiv.org/abs/2307.12703)

    该论文提出了一种新的算法，通过引入相关路径来降低蒙特卡洛模拟中的方差，从而估计随机微分方程解的函数。通过政策梯度和强化学习技术，使用深度神经网络近似最优相关函数并进行校准。这与最大最优传输问题有关。

    

    我们提出了一种用于估计$f(X_T)$的方差降低算法，其中$X$是某个随机微分方程的解，$f$是一个测试函数。新的估计器是$(f(X^1_T) + f(X^2_T))/2$，其中$X^1$和$X^2$具有与$X$相同的边际分布，但路径上存在相关性，以降低方差。最优相关函数$\rho$由深度神经网络近似，并通过政策梯度和强化学习技术在$(X^1, X^2)$的轨迹上进行校准。在给定边际分布的情况下找到最优耦合与最大最优传输有关联。

    We propose a new algorithm for variance reduction when estimating $f(X_T)$ where $X$ is the solution to some stochastic differential equation and $f$ is a test function. The new estimator is $(f(X^1_T) + f(X^2_T))/2$, where $X^1$ and $X^2$ have same marginal law as $X$ but are pathwise correlated so that to reduce the variance. The optimal correlation function $\rho$ is approximated by a deep neural network and is calibrated along the trajectories of $(X^1, X^2)$ by policy gradient and reinforcement learning techniques. Finding an optimal coupling given marginal laws has links with maximum optimal transport.
    
[^110]: 重新思考数据蒸馏：不要忽视校准

    Rethinking Data Distillation: Do Not Overlook Calibration. (arXiv:2307.12463v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2307.12463](http://arxiv.org/abs/2307.12463)

    本文指出经过蒸馏的数据无法很好地进行校准，因为在这种情况下，网络的 logits 分布更加集中，并且语义明确但与分类任务无关的信息会丢失。为了解决这个问题，我们提出了遮蔽温度缩放 (MTS) 和遮蔽蒸馏训练 (MDT) 方法，以获得更好的校准结果。

    

    在经过蒸馏的数据上训练的神经网络经常产生过于自信的输出，并需要通过校准方法进行修正。现有的校准方法，如温度缩放和混合训练，在原始的大规模数据上训练的神经网络上效果良好。然而，我们发现这些方法无法对从大源数据集蒸馏出的数据进行校准。本文中，我们展示了蒸馏数据会导致网络无法校准，原因是（i）最大logit分布更为集中，以及（ii）在分类任务无关但语义意义明确的信息损失。为了解决这个问题，我们提出了遮蔽温度缩放（MTS）和遮蔽蒸馏训练（MDT）方法，以减轻蒸馏数据的限制，并在保持数据蒸馏效率的同时实现更好的校准结果。

    Neural networks trained on distilled data often produce over-confident output and require correction by calibration methods. Existing calibration methods such as temperature scaling and mixup work well for networks trained on original large-scale data. However, we find that these methods fail to calibrate networks trained on data distilled from large source datasets. In this paper, we show that distilled data lead to networks that are not calibratable due to (i) a more concentrated distribution of the maximum logits and (ii) the loss of information that is semantically meaningful but unrelated to classification tasks. To address this problem, we propose Masked Temperature Scaling (MTS) and Masked Distillation Training (MDT) which mitigate the limitations of distilled data and achieve better calibration results while maintaining the efficiency of dataset distillation.
    
[^111]: 用学习的代理和约束解决基于多物理的反问题

    Solving multiphysics-based inverse problems with learned surrogates and constraints. (arXiv:2307.11099v1 [physics.geo-ph])

    [http://arxiv.org/abs/2307.11099](http://arxiv.org/abs/2307.11099)

    本论文将学习代理和学习约束相结合用于解决基于多物理的反问题，通过该方法不仅改善了对流体流动性质的反演精度，而且为反演多模态数据提供了一个有效的解决方案。

    

    在地质碳封存监测中，当多模态时变数据昂贵且数值模拟成本高昂时，解决基于多物理的反问题可能具有挑战性。我们通过将计算成本低廉的学习代理与学习约束相结合来克服这些挑战。这种组合不仅能够大大改善对重要流体流动性质（渗透率）的反演，还能为反演多模态数据（包括井测量和主动源时变地震数据）提供一个自然的平台。通过添加学习约束，我们得到了一个计算可行的反演方法，其精度仍然准确。这通过包含一个经过训练的深度神经网络（称为归一化流），使模型迭代保持在分布内，从而保证了作为代理的经过训练的傅里叶神经算子的准确性，这些算子用于代替涉及部分计算昂贵的多相流模拟。

    Solving multiphysics-based inverse problems for geological carbon storage monitoring can be challenging when multimodal time-lapse data are expensive to collect and costly to simulate numerically. We overcome these challenges by combining computationally cheap learned surrogates with learned constraints. Not only does this combination lead to vastly improved inversions for the important fluid-flow property, permeability, it also provides a natural platform for inverting multimodal data including well measurements and active-source time-lapse seismic data. By adding a learned constraint, we arrive at a computationally feasible inversion approach that remains accurate. This is accomplished by including a trained deep neural network, known as a normalizing flow, which forces the model iterates to remain in-distribution, thereby safeguarding the accuracy of trained Fourier neural operators that act as surrogates for the computationally expensive multiphase flow simulations involving partia
    
[^112]: 关于约束时间序列生成问题的研究

    On the Constrained Time-Series Generation Problem. (arXiv:2307.01717v1 [cs.LG])

    [http://arxiv.org/abs/2307.01717](http://arxiv.org/abs/2307.01717)

    这篇论文研究了约束时间序列生成问题。在实际应用中，合成时间序列被广泛用于增强历史时间序列数据集，提高机器学习算法的性能，放大稀有事件的发生，以及创建反事实情景。然而，现有的方法在满足约束方面存在问题，需要重新训练且计算代价高，或者在复杂约束条件下不切实际。

    

    合成时间序列经常在实际应用中用于增加历史时间序列数据集，以提高机器学习算法的性能，放大稀有事件的发生，并创建由时间序列描述的反事实情景。分布相似性（我们称之为真实性）以及满足一定数值约束是反事实时间序列场景生成请求中常见的要求。例如，美联储发布了给定约束时间序列的合成市场压力情景，供金融机构评估其在假设性衰退中的表现。现有的生成约束时间序列的方法通常通过对损失函数进行惩罚来强制满足约束，并拒绝不符合约束的样本。然而，如果我们改变约束条件，这些方法需要重新训练，而拒绝抽样可能在计算上是昂贵的，或者在复杂约束条件下是不切实际的。

    Synthetic time series are often used in practical applications to augment the historical time series dataset for better performance of machine learning algorithms, amplify the occurrence of rare events, and also create counterfactual scenarios described by the time series. Distributional-similarity (which we refer to as realism) as well as the satisfaction of certain numerical constraints are common requirements in counterfactual time series scenario generation requests. For instance, the US Federal Reserve publishes synthetic market stress scenarios given by the constrained time series for financial institutions to assess their performance in hypothetical recessions. Existing approaches for generating constrained time series usually penalize training loss to enforce constraints, and reject non-conforming samples. However, these approaches would require re-training if we change constraints, and rejection sampling can be computationally expensive, or impractical for complex constraints.
    
[^113]: Engression: 非线性回归的外推方法

    Engression: Extrapolation for Nonlinear Regression?. (arXiv:2307.00835v1 [stat.ME])

    [http://arxiv.org/abs/2307.00835](http://arxiv.org/abs/2307.00835)

    Engression是一种非线性回归方法，通过使用分布回归技术和预加性噪声模型，在训练样本范围边界外也能可靠地进行外推。

    

    外推对于许多统计学和机器学习应用至关重要，因为常常会遇到超出训练样本范围的测试数据。然而，对于非线性模型来说，外推是一个巨大的挑战。传统模型在这方面通常遇到困难：树集成模型在支持范围外提供连续的预测，而神经网络的预测往往变得不可控。这项工作旨在提供一种非线性回归方法，其可靠性在训练样本范围边界不会立即崩溃。我们的主要贡献是一种名为“engression”的新方法，它是一种预加性噪声模型的分布回归技术，其中噪声添加到协变量上并应用非线性转换。我们的实验结果表明，该模型通常适用于许多真实数据集。我们展示engression可以在一些假设下成功进行外推，例如严格限制噪声大小。

    Extrapolation is crucial in many statistical and machine learning applications, as it is common to encounter test data outside the training support. However, extrapolation is a considerable challenge for nonlinear models. Conventional models typically struggle in this regard: while tree ensembles provide a constant prediction beyond the support, neural network predictions tend to become uncontrollable. This work aims at providing a nonlinear regression methodology whose reliability does not break down immediately at the boundary of the training support. Our primary contribution is a new method called `engression' which, at its core, is a distributional regression technique for pre-additive noise models, where the noise is added to the covariates before applying a nonlinear transformation. Our experimental results indicate that this model is typically suitable for many real data sets. We show that engression can successfully perform extrapolation under some assumptions such as a strictl
    
[^114]: 特征选择：对属性间协作的视角

    Feature Selection: A perspective on inter-attribute cooperation. (arXiv:2306.16559v1 [cs.LG])

    [http://arxiv.org/abs/2306.16559](http://arxiv.org/abs/2306.16559)

    本文综述了辅助特征间协作的过滤特征选择方法的最新研究进展，并总结了不同方法在文献中的贡献。同时提出了当前存在的问题和挑战，以确定未来有前景的研究和发展方向。

    

    高维数据对数据挖掘和机器学习中的学习任务构成了挑战。特征选择是处理维度缩减的一种有效技术，通常是在应用学习算法之前的重要数据处理步骤。在过去几十年中，过滤特征选择方法从简单的单变量相关性排序算法发展到更复杂的相关性-冗余权衡和基于多元依赖性的方法。这种捕捉多变量依赖的趋势旨在通过特征间的互相合作获取关于类别的独特信息。本文对辅助特征间协作的过滤特征选择方法的最新研究工作进行了全面的调查，并总结了文献中不同方法的贡献。此外，还介绍了当前存在的问题和挑战，以确定未来有前景的研究和发展方向。

    High-dimensional datasets depict a challenge for learning tasks in data mining and machine learning. Feature selection is an effective technique in dealing with dimensionality reduction. It is often an essential data processing step prior to applying a learning algorithm. Over the decades, filter feature selection methods have evolved from simple univariate relevance ranking algorithms to more sophisticated relevance-redundancy trade-offs and to multivariate dependencies-based approaches in recent years. This tendency to capture multivariate dependence aims at obtaining unique information about the class from the intercooperation among features. This paper presents a comprehensive survey of the state-of-the-art work on filter feature selection methods assisted by feature intercooperation, and summarizes the contributions of different approaches found in the literature. Furthermore, current issues and challenges are introduced to identify promising future research and development.
    
[^115]: SEEDS：利用扩散模型仿真天气预测集合

    SEEDS: Emulation of Weather Forecast Ensembles with Diffusion Models. (arXiv:2306.14066v1 [cs.LG])

    [http://arxiv.org/abs/2306.14066](http://arxiv.org/abs/2306.14066)

    本文提出了利用生成人工智能技术在规模上生成集合预测的方法，并产生了与完整的GEFS 31成员集合相似的预测能力，并且很好地模拟了大规模集合的统计数据。

    

    在不确定未来天气时，概率预测对决策非常重要。主要的方法是使用预测集合来表示和量化数值天气预报的不确定性。然而，产生集合的计算成本很高。本文提出利用最近的生成人工智能技术在规模上生成集合预测的方法。我们的方法从5成员集合GEFS重新预报数据集中学习基于数据驱动的概率扩散模型。该模型可以有效地进行采样，以产生联合情况下真实的天气预测，这些情况可以基于操作GEFS预测系统的少数成员条件化。根据ERA5分析评估，生成的集合与完整的GEFS 31成员集合具有相似的预测能力，并且很好地模拟了大规模集合的统计数据。我们还将相同的方法应用于开发扩散模型，进行生成后处理。模型可以基于少数预测成员条件化地生成类似于物理大模型集合的预测集合。

    Probabilistic forecasting is crucial to decision-making under uncertainty about future weather. The dominant approach is to use an ensemble of forecasts to represent and quantify uncertainty in operational numerical weather prediction. However, generating ensembles is computationally costly. In this paper, we propose to generate ensemble forecasts at scale by leveraging recent advances in generative artificial intelligence. Our approach learns a data-driven probabilistic diffusion model from the 5-member ensemble GEFS reforecast dataset. The model can then be sampled efficiently to produce realistic weather forecasts, conditioned on a few members of the operational GEFS forecasting system. The generated ensembles have similar predictive skill as the full GEFS 31-member ensemble, evaluated against ERA5 reanalysis, and emulate well the statistics of large physics-based ensembles. We also apply the same methodology to developing a diffusion model for generative post-processing: the model 
    
[^116]: 基于潜在扩散模型的文本驱动Foley音效生成

    Text-Driven Foley Sound Generation With Latent Diffusion Model. (arXiv:2306.10359v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2306.10359](http://arxiv.org/abs/2306.10359)

    本文提出了一种基于扩散模型的Foley音效生成系统，可进行文本条件的生成。我们通过迁移学习对系统进行微调，并引入可训练的层来改善文本嵌入，同时也改进了生成的波形。

    

    Foley音效生成旨在为多媒体内容生成背景音效。先前的模型通常使用大量有标签的开发集作为输入（例如，单个数字或one-hot向量）。本文提出了一种基于扩散模型的Foley音效生成系统，可进行文本条件的生成。为了缓解数据稀缺问题，我们的模型首先使用大规模数据集进行预训练，然后通过对比语言-音频配对（CLAP）技术进行迁移学习来对该任务进行微调。我们观察到，文本编码器提取的特征嵌入可以显著影响生成模型的性能。因此，我们在编码器之后引入可训练的层来改善编码器产生的文本嵌入。此外，我们通过同时生成多个候选音频片段并选择最佳片段来进一步改进生成的波形，最佳片段是根据嵌入之间相似性得分确定的。

    Foley sound generation aims to synthesise the background sound for multimedia content. Previous models usually employ a large development set with labels as input (e.g., single numbers or one-hot vector). In this work, we propose a diffusion model based system for Foley sound generation with text conditions. To alleviate the data scarcity issue, our model is initially pre-trained with large-scale datasets and fine-tuned to this task via transfer learning using the contrastive language-audio pertaining (CLAP) technique. We have observed that the feature embedding extracted by the text encoder can significantly affect the performance of the generation model. Hence, we introduce a trainable layer after the encoder to improve the text embedding produced by the encoder. In addition, we further refine the generated waveform by generating multiple candidate audio clips simultaneously and selecting the best one, which is determined in terms of the similarity score between the embedding of the 
    
[^117]: 使用二进制化和少量全精度权重的神经网络压缩

    Neural Network Compression using Binarization and Few Full-Precision Weights. (arXiv:2306.08960v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.08960](http://arxiv.org/abs/2306.08960)

    本文提出了一种自动修剪二进制化（APB）的新型神经网络压缩技术，通过结合量化和修剪，利用少量全精度权重来增强二进制网络的表示能力，并通过高效的算法在CPU上实现了高速的量化矩阵乘法运算。

    

    量化和修剪是两种有效的深度神经网络模型压缩方法。本文提出了自动修剪二进制化（APB），这是一种结合了量化和修剪的新型压缩技术。APB利用少量全精度权重增强了二进制网络的表示能力。我们的技术在最大化网络准确性的同时最小化了其内存占用，通过决定每个权重是应该进行二进制化还是保持全精度。我们展示了如何通过将其分解为二进制和稀疏-稠密矩阵乘法来高效地执行使用APB压缩的前向传递。此外，我们在CPU上设计了两种新颖的高效量化矩阵乘法算法，利用了高效的位操作。所提出的算法比现有的最先进解决方案快6.9倍和1.5倍。我们对APB在两个广泛采用的模型压缩数据集CIFAR10和Imag上进行了广泛评估。

    Quantization and pruning are two effective Deep Neural Networks model compression methods. In this paper, we propose Automatic Prune Binarization (APB), a novel compression technique combining quantization with pruning. APB enhances the representational capability of binary networks using a few full-precision weights. Our technique jointly maximizes the accuracy of the network while minimizing its memory impact by deciding whether each weight should be binarized or kept in full precision. We show how to efficiently perform a forward pass through layers compressed using APB by decomposing it into a binary and a sparse-dense matrix multiplication. Moreover, we design two novel efficient algorithms for extremely quantized matrix multiplication on CPU, leveraging highly efficient bitwise operations. The proposed algorithms are 6.9x and 1.5x faster than available state-of-the-art solutions. We extensively evaluate APB on two widely adopted model compression datasets, namely CIFAR10 and Imag
    
[^118]: 物理信息神经网络在逆流自发渗透中的应用和预测：早期和晚期的模拟

    Simulation and Prediction of Countercurrent Spontaneous Imbibition at Early and Late Times Using Physics-Informed Neural Networks. (arXiv:2306.05554v1 [physics.comp-ph])

    [http://arxiv.org/abs/2306.05554](http://arxiv.org/abs/2306.05554)

    本文通过物理信息神经网络模型对多孔材料中的逆流自发渗透过程进行了早期和晚期的模拟和预测，并使用改变变量技术来改进模型性能。

    

    逆流自发渗透（COUCSI）是一种多孔材料中的过程，其中润湿相取代了非润湿相的位置。本文首次探讨了物理信息神经网络（PINNs）在解决早期（ET）和晚期（LT）COUCSI问题中的应用。同时，我们还研究了改变变量技术以改进PINNs的性能。我们通过改变自变量将COUCSI问题分别用XT-，XY-和Z-三种等效形式进行描述：第一个描述了饱和度作为规范化位置X和时间T的函数;第二个描述了X和Y=T^0.5作为函数的饱和度;第三个作为Z=X/T^0.5的唯一函数（仅在ET下有效）。该PINN模型使用前馈神经网络生成，并基于最小化加权损失函数进行训练，包括物理信息丢失项和与初始边界条件相对应的项。没有合成或实验数据被调用。

    Countercurrent spontaneous imbibition (COUCSI) is a process in porous materials in which a wetting phase displaces non-wetting phase. In this work, we investigate for the first time the application of Physics-Informed Neural Networks (PINNs) in solving the 1D COUCSI problem in both early (ET) and late (LT) times. Also novel, we examine the Change-of-Variables technique for improving the performance of PINNs. We formulated the COUCSI problem in three equivalent forms by changing the independent variables: XT-, XY-, and Z-formulations. The first describes saturation as function of normalized position X and time T; the second as function of X and Y=T^0.5; and the third as a sole function of Z=X/T^0.5 (valid only at ET). The PINN model was generated using a feed-forward neural network and trained based on minimizing a weighted loss function, including the physics-informed loss term and terms corresponding to the initial and boundary conditions. No synthetical or experimental data were invo
    
[^119]: 在脉冲神经网络中将噪声作为计算和学习资源

    Exploiting Noise as a Resource for Computation and Learning in Spiking Neural Networks. (arXiv:2305.16044v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2305.16044](http://arxiv.org/abs/2305.16044)

    本文提出了噪声脉冲神经元网络（NSNN）和噪声驱动学习规则（NDL），展示了噪声可以作为计算和学习的资源，并为一般脉冲神经元网络提供了一个框架。研究还展示了NSNNs在图像分类和语音识别等实际任务中的适用性，表明它们是未来神经形态计算系统的潜在有力工具。

    

    脉冲神经元网络是大脑非凡信息处理能力的基础，并已成为神经形态智能的支柱模型。本文介绍了噪声脉冲神经元网络（NSNN）和噪声驱动学习规则（NDL），采用带有噪声神经元动力学的脉冲神经元模型。该方法显示噪声可以作为计算和学习的资源，并理论上为一般脉冲神经元网络提供了一个框架。此外，NDL为代理梯度提供了深入的生物学合理性。通过将各种SNN架构和算法结合起来，我们展示了我们的方法表现出竞争性能，并且比确定性SNNs表现出更好的鲁棒性。此外，本文还展示了NSNNs在图像分类和语音识别等实际任务中的适用性，表明它们是未来神经形态计算系统的潜在有力工具。

    Networks of spiking neurons underpin the extraordinary information-processing capabilities of the brain and have emerged as pillar models in neuromorphic intelligence. Despite extensive research on spiking neural networks (SNNs), most are established on deterministic models. Integrating noise into SNNs leads to biophysically more realistic neural dynamics and may benefit model performance. This work presents the noisy spiking neural network (NSNN) and the noise-driven learning rule (NDL) by introducing a spiking neuron model incorporating noisy neuronal dynamics. Our approach shows how noise may act as a resource for computation and learning and theoretically provides a framework for general SNNs. Moreover, NDL provides an insightful biological rationale for surrogate gradients. By incorporating various SNN architectures and algorithms, we show that our approach exhibits competitive performance and improved robustness against challenging perturbations than deterministic SNNs. Additiona
    
[^120]: 多个未标记数据集的AUC优化

    AUC Optimization from Multiple Unlabeled Datasets. (arXiv:2305.15776v1 [cs.LG])

    [http://arxiv.org/abs/2305.15776](http://arxiv.org/abs/2305.15776)

    本文提出了一种从多个未标记数据集中构建AUC优化模型的方法，该方法在实践和理论上都有效。

    

    弱监督学习旨在在缺乏完美监督的情况下赋予机器学习能力，这引起了研究人员的极大关注。在各种类型的弱监督学习中，最具挑战性的案例之一是仅了解类别先验知识的多个未标记(U)数据集的学习或称为U^m学习。本文研究了从多个未标记数据集中构建最大化分类器成对排名能力的AUC (ROC曲线下面积) 优化模型的问题。我们提出了U^m-AUC，一种将U^m数据转换为多标记AUC优化问题并能够有效训练的AUC优化方法。我们理论上和实证上证明了所提出的U^m-AUC的有效性。

    Weakly supervised learning aims to empower machine learning when the perfect supervision is unavailable, which has drawn great attention from researchers. Among various types of weak supervision, one of the most challenging cases is to learn from multiple unlabeled (U) datasets with only a little knowledge of the class priors, or U$^m$ learning for short. In this paper, we study the problem of building an AUC (area under ROC curve) optimization model from multiple unlabeled datasets, which maximizes the pairwise ranking ability of the classifier. We propose U$^m$-AUC, an AUC optimization approach that converts the U$^m$ data into a multi-label AUC optimization problem, and can be trained efficiently. We show that the proposed U$^m$-AUC is effective theoretically and empirically.
    
[^121]: 使用不同iable的搜索体系架构提高语音情绪识别性能。

    Improving Speech Emotion Recognition Performance using Differentiable Architecture Search. (arXiv:2305.14402v1 [cs.SD])

    [http://arxiv.org/abs/2305.14402](http://arxiv.org/abs/2305.14402)

    该论文提出使用DARTS优化联合CNN和LSTM的体系结构以提高语音情绪识别性能，并在实验中证明了其优于以往最好的结果。

    

    语音情绪识别(SER)是实现情感感知交互的关键因素。深度学习(DL)改善了SER模型的性能，但设计DL体系结构需要先前的经验和实验评估。鼓励地，神经体系结构搜索(NAS)允许自动搜索最优DL模型。特别地，可区分的体系结构搜索(DARTS)是一种使用NAS搜索最优化模型的有效方法。在本文中，我们提出DARTS用于联合CNN和LSTM的体系结构，以改善SER性能。我们选择CNN LSTM耦合的原因是结果表明类似的模型能够提高性能。虽然SER研究人员已将CNN和RNN分别考虑，但DARTs同时用于CNN和LSTM的可行性仍需要探索。通过对IEMOCAP数据集的实验，我们证明了我们的方法优于使用DA的最佳报告结果。

    Speech Emotion Recognition (SER) is a critical enabler of emotion-aware communication in human-computer interactions. Deep Learning (DL) has improved the performance of SER models by improving model complexity. However, designing DL architectures requires prior experience and experimental evaluations. Encouragingly, Neural Architecture Search (NAS) allows automatic search for an optimum DL model. In particular, Differentiable Architecture Search (DARTS) is an efficient method of using NAS to search for optimised models. In this paper, we propose DARTS for a joint CNN and LSTM architecture for improving SER performance. Our choice of the CNN LSTM coupling is inspired by results showing that similar models offer improved performance. While SER researchers have considered CNNs and RNNs separately, the viability of using DARTs jointly for CNN and LSTM still needs exploration. Experimenting with the IEMOCAP dataset, we demonstrate that our approach outperforms best-reported results using DA
    
[^122]: 异质数据的预测性变点检测

    Predictive change point detection for heterogeneous data. (arXiv:2305.06630v1 [cs.LG])

    [http://arxiv.org/abs/2305.06630](http://arxiv.org/abs/2305.06630)

    该论文提出了一种基于“预测与比较”机器学习模型的变点监测框架，它能够比现有的在线监测方法更好地控制误报率和失控平均运行长度。该方法使用ARIMA模型和LSTM递归神经网络模型进行预测，具有很强的推广性能。

    

    引入一个名为“预测与比较”的机器学习模型辅助的变点检测（CPD）框架，并与其他在线CPD例程进行比较，结果表明该方法在误报率和失控平均运行长度方面表现更优。该方法的重点是通过使用更复杂的预测模型（预测步骤）代替通常使用的趋势估计函数（如滑动平均），并将其预测与实际数据进行比较（比较步骤），从而改善顺序分析中的标准方法，例如CUSUM规则，以提高这些质量指标。

    A change point detection (CPD) framework assisted by a predictive machine learning model called ''Predict and Compare'' is introduced and characterised in relation to other state-of-the-art online CPD routines which it outperforms in terms of false positive rate and out-of-control average run length. The method's focus is on improving standard methods from sequential analysis such as the CUSUM rule in terms of these quality measures.  This is achieved by replacing typically used trend estimation functionals such as the running mean with more sophisticated predictive models (Predict step), and comparing their prognosis with actual data (Compare step). The two models used in the Predict step are the ARIMA model and the LSTM recursive neural network. However, the framework is formulated in general terms, so as to allow the use of other prediction or comparison methods than those tested here. The power of the method is demonstrated in a tribological case study in which change points separa
    
[^123]: 基于Wasserstein字典的持久图的紧凑编码

    Wasserstein Dictionaries of Persistence Diagrams. (arXiv:2304.14852v1 [cs.LG])

    [http://arxiv.org/abs/2304.14852](http://arxiv.org/abs/2304.14852)

    本文提出了一种基于Wasserstein字典的持久图的紧凑编码方法，并在实验中证明了其有效性，可以应用于数据降维和压缩。

    

    本文提出了一个计算框架，用于以原子图字典的加权Wasserstein barycenters [99]，[101] 的形式对一组持久图进行简洁编码。我们引入了一种多尺度梯度下降方法，用于有效解决相应的最小化问题，该方法将Barycenter权重的优化与Atom图的优化交错进行。我们的方法利用了两个子问题梯度的解析表达式以确保快速迭代，并且还利用了共享内存并行性。对公共合奏的广泛实验证明了我们方法的有效性，最大示例的Wasserstein字典计算时间在数分钟之内。我们在两个应用中展示了我们的贡献的效用。首先，我们将Wassserstein字典应用于数据降维，并通过仅用其重量来紧凑地表示Persistence图来可靠地压缩它们。

    This paper presents a computational framework for the concise encoding of an ensemble of persistence diagrams, in the form of weighted Wasserstein barycenters [99], [101] of a dictionary of atom diagrams. We introduce a multi-scale gradient descent approach for the efficient resolution of the corresponding minimization problem, which interleaves the optimization of the barycenter weights with the optimization of the atom diagrams. Our approach leverages the analytic expressions for the gradient of both sub-problems to ensure fast iterations and it additionally exploits shared-memory parallelism. Extensive experiments on public ensembles demonstrate the efficiency of our approach, with Wasserstein dictionary computations in the orders of minutes for the largest examples. We show the utility of our contributions in two applications. First, we apply Wassserstein dictionaries to data reduction and reliably compress persistence diagrams by concisely representing them with their weights in t
    
[^124]: CVRecon: 重新思考神经重建的3D几何特征学习

    CVRecon: Rethinking 3D Geometric Feature Learning For Neural Reconstruction. (arXiv:2304.14633v1 [cs.CV])

    [http://arxiv.org/abs/2304.14633](http://arxiv.org/abs/2304.14633)

    研究团队提出了一种基于代价体的3D神经重建框架CVRecon，利用丰富的几何嵌入来促进3D几何特征学习。通过引入射线上下文补偿代价体（RCCV），有效提高了视角相关信息的完整性和鲁棒性，并在各种度量方面显着提高了重建质量。

    

    最近使用图像序列进行神经重建的进展取得了显着进展。但是，由于缺乏深度信息，现有的基于体积的技术仅沿整个相机光线复制对象表面的2D图像特征。我们认为这种复制会在空洞和遮挡空间中引入噪声，从而产生高质量的3D几何体成形方面产生挑战。受传统多视角立体方法的启发，我们提出了一种端到端的3D神经重建框架CVRecon，旨在利用代价体中丰富的几何嵌入来促进3D几何特征学习。此外，我们提出了一种新颖的3D几何特征表示法——射线上下文补偿代价体（RCCV），它具有更好的完整性和鲁棒性，可以编码视角相关信息。通过全面的实验，我们证明了我们的方法在各种度量方面显着提高了重建质量，并恢复了清晰的

    Recent advances in neural reconstruction using posed image sequences have made remarkable progress. However, due to the lack of depth information, existing volumetric-based techniques simply duplicate 2D image features of the object surface along the entire camera ray. We contend this duplication introduces noise in empty and occluded spaces, posing challenges for producing high-quality 3D geometry. Drawing inspiration from traditional multi-view stereo methods, we propose an end-to-end 3D neural reconstruction framework CVRecon, designed to exploit the rich geometric embedding in the cost volumes to facilitate 3D geometric feature learning. Furthermore, we present Ray-contextual Compensated Cost Volume (RCCV), a novel 3D geometric feature representation that encodes view-dependent information with improved integrity and robustness. Through comprehensive experiments, we demonstrate that our approach significantly improves the reconstruction quality in various metrics and recovers clear
    
[^125]: 基于替代模型的人机交互场景生成

    Surrogate Assisted Generation of Human-Robot Interaction Scenarios. (arXiv:2304.13787v1 [cs.RO])

    [http://arxiv.org/abs/2304.13787](http://arxiv.org/abs/2304.13787)

    本文提出了基于替代模型的人机交互场景生成方法，可以高效合成多样化的挑战性数据集，以便评估和理解人机交互系统的优劣，可以在实际交互中重现这些场景。

    

    随着人机交互系统的发展，不同环境和用户下评估和理解这些系统的优缺点变得越来越困难。为此，以往的方法通过算法生成了多样的场景，揭示了共享控制遥操作任务的系统失效情况。然而，这些方法需要通过模拟机器人策略和人类行为来直接评估生成的场景。这些评估所需的计算成本限制了它们在更复杂的领域的适用性。因此，我们提出了通过替代模型来预测人类和机器人行为来增强场景生成系统的建议。在共享控制遥操作域和更复杂的共享工作空间协作任务中，我们展示了替代模型辅助的场景生成可以高效地合成具有挑战性的多样数据集。我们展示了这些故障在真实世界中的交互中是可重现的。

    As human-robot interaction (HRI) systems advance, so does the difficulty of evaluating and understanding the strengths and limitations of these systems in different environments and with different users. To this end, previous methods have algorithmically generated diverse scenarios that reveal system failures in a shared control teleoperation task. However, these methods require directly evaluating generated scenarios by simulating robot policies and human actions. The computational cost of these evaluations limits their applicability in more complex domains. Thus, we propose augmenting scenario generation systems with surrogate models that predict both human and robot behaviors. In the shared control teleoperation domain and a more complex shared workspace collaboration task, we show that surrogate assisted scenario generation efficiently synthesizes diverse datasets of challenging scenarios. We demonstrate that these failures are reproducible in real-world interactions.
    
[^126]: 无约束参数化的耗散性和收缩性神经常微分方程

    Unconstrained Parametrization of Dissipative and Contracting Neural Ordinary Differential Equations. (arXiv:2304.02976v1 [eess.SY])

    [http://arxiv.org/abs/2304.02976](http://arxiv.org/abs/2304.02976)

    本文介绍了一种连续时间的深度神经网络，通过结合神经常微分方程和循环平衡网络的结构，使得网络具有收缩和耗散性质。此外提出的非约束参数化方法使得该网络学习的参数量得以增加。

    

    本文介绍和研究了一类连续时间的深度神经网络，提出的架构源于神经常微分方程和最近引入的循环平衡网络（RENs）的模型结构相结合。我们展示了如何赋予我们提出的NodeRENs收缩和耗散性——对于健壮的学习和控制至关重要的属性。最重要的是，与RENs一样，我们推导了收缩和耗散NodeRENs的参数化，这些参数没有约束，因此能够学习大量的参数。我们在非线性系统识别的案例研究中验证了NodeRENs的属性，包括处理不规则采样数据的可能性。

    In this work, we introduce and study a class of Deep Neural Networks (DNNs) in continuous-time. The proposed architecture stems from the combination of Neural Ordinary Differential Equations (Neural ODEs) with the model structure of recently introduced Recurrent Equilibrium Networks (RENs). We show how to endow our proposed NodeRENs with contractivity and dissipativity -- crucial properties for robust learning and control. Most importantly, as for RENs, we derive parametrizations of contractive and dissipative NodeRENs which are unconstrained, hence enabling their learning for a large number of parameters. We validate the properties of NodeRENs, including the possibility of handling irregularly sampled data, in a case study in nonlinear system identification.
    
[^127]: BoundaryCAM：一种基于边界的弱监督医学图像语义分割优化框架

    BoundaryCAM: A Boundary-based Refinement Framework for Weakly Supervised Semantic Segmentation of Medical Images. (arXiv:2303.07853v1 [cs.CV])

    [http://arxiv.org/abs/2303.07853](http://arxiv.org/abs/2303.07853)

    BoundaryCAM提出了一种基于边界的弱监督的优化框架，能够预测对象位置，实现精细的高精度分割掩模。

    

    仅利用图像级别监督的弱监督语义分割（WSSS）是解决分割网络需求的一种有前途的方法，尤其是对于在给定数据集中生成大量像素级掩模。然而，大多数最先进的图像级WSSS技术缺乏对图像中包含的几何特征的理解，因为网络无法从仅图像级别标签中导出任何对象边界信息。为了解决这个缺陷，我们提出了我们的新型BoundaryCAM框架，该框架采用最先进的类激活图结合各种后处理技术，以实现精细的高精度分割掩模。为了实现这一目标，我们调查了一种最先进的无监督语义分割网络，该网络可用于构建边界图，以使BoundaryCAM能够高精度预测对象位置。

    Weakly Supervised Semantic Segmentation (WSSS) with only image-level supervision is a promising approach to deal with the need for Segmentation networks, especially for generating a large number of pixel-wise masks in a given dataset. However, most state-of-the-art image-level WSSS techniques lack an understanding of the geometric features embedded in the images since the network cannot derive any object boundary information from just image-level labels. We define a boundary here as the line separating an object and its background, or two different objects. To address this drawback, we propose our novel BoundaryCAM framework, which deploys state-of-the-art class activation maps combined with various post-processing techniques in order to achieve fine-grained higher-accuracy segmentation masks. To achieve this, we investigate a state-of-the-art unsupervised semantic segmentation network that can be used to construct a boundary map, which enables BoundaryCAM to predict object locations w
    
[^128]: 数据中心人工智能：通过离散子集作为连续嵌入空间优化实现深度生成可微分特征选择

    Data-Centric AI: Deep Generative Differentiable Feature Selection via Discrete Subsetting as Continuous Embedding Space Optimization. (arXiv:2302.13221v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.13221](http://arxiv.org/abs/2302.13221)

    该论文提出一种将离散特征子集作为连续嵌入空间优化的深度生成可微分特征选择方法，解决了在高维小样本数据集中通用、准确和维度无关的特征选择问题。

    

    特征选择（FS）旨在为给定的下游任务找到最佳特征子集，例如过滤器、包装器和嵌入式方法。但在许多实际应用中，FS的标准在不同领域中变化，并且当数据是高维和小样本时，FS容易出现问题。选择的特征子集是否可以更通用、准确和维度无关？我们将这个问题泛化为一个深度可微分特征选择任务，并提出了一个新的视角：将离散特征子集作为连续嵌入空间优化。我们开发了一个通用和原则性的框架，包括深度特征子集编码器、准确性评估器、解码器和梯度上升优化器。这个框架实现了四个步骤：1) 特征-准确性训练数据准备；2) 深度特征子集嵌入；3) 梯度优化搜索；4) 特征子集重建。我们提出了新的技术洞见：将强化作为训练数据生成器、多样化的集成模型视为搜索加速器、多尺度的特征选择和逐渐增强的探索

    Feature Selection (FS), such as filter, wrapper, and embedded methods, aims to find the optimal feature subset for a given downstream task. However, in many real-world practices, 1) the criteria of FS vary across domains; 2) FS is brittle when data is a high-dimensional and small sample size. Can selected feature subsets be more generalized, accurate, and input dimensionality agnostic? We generalize this problem into a deep differentiable feature selection task and propose a new perspective: discrete feature subsetting as continuous embedding space optimization. We develop a generic and principled framework including a deep feature subset encoder, accuracy evaluator, decoder, and gradient ascent optimizer. This framework implements four steps: 1) features-accuracy training data preparation; 2) deep feature subset embedding; 3) gradient-optimized search; 4) feature subset reconstruction. We develop new technical insights: reinforcement as a training data generator, ensembles of diverse 
    
[^129]: 边缘机器学习的综述与分类：需求，范式和技术

    A Comprehensive Review and a Taxonomy of Edge Machine Learning: Requirements, Paradigms, and Techniques. (arXiv:2302.08571v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.08571](http://arxiv.org/abs/2302.08571)

    这篇论文综述了边缘机器学习的需求、范式和技术，并强调了其在保护隐私、实现低延迟的实时性能和资源优化方面的重要性。

    

    边缘计算(EC)和人工智能(AI)的结合提出了边缘AI的概念，为了保护隐私，实现低延迟的实时性能和资源优化，提供了接近最终用户环境的智能解决方案。机器学习(ML)作为近年来AI中最先进的分支，在边缘环境中展示了令人鼓舞的结果和应用。然而，边缘驱动的ML解决方案更加复杂，因为它同时考虑到边缘计算和AI领域的约束，并且期望这些解决方案在边缘ML的需求方面高效且适应性强，如数据处理，模型压缩，分布式推理和高级学习范式。尽管在学术界和工业界都受到了边缘ML的关注，但我们注意到缺乏对现有边缘ML技术的完整调查，以提供一个共同的理解。

    The union of Edge Computing (EC) and Artificial Intelligence (AI) has brought forward the Edge AI concept to provide intelligent solutions close to the end-user environment, for privacy preservation, low latency to real-time performance, and resource optimization. Machine Learning (ML), as the most advanced branch of AI in the past few years, has shown encouraging results and applications in the edge environment. Nevertheless, edge-powered ML solutions are more complex to realize due to the joint constraints from both edge computing and AI domains, and the corresponding solutions are expected to be efficient and adapted in technologies such as data processing, model compression, distributed inference, and advanced learning paradigms for Edge ML requirements. Despite the fact that a great deal of the attention garnered by Edge ML is gained in both the academic and industrial communities, we noticed the lack of a complete survey on existing Edge ML technologies to provide a common unders
    
[^130]: 使用大型语言模型在强化学习中引导预训练

    Guiding Pretraining in Reinforcement Learning with Large Language Models. (arXiv:2302.06692v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.06692](http://arxiv.org/abs/2302.06692)

    这项研究提出了一种使用大型语言模型在强化学习中引导预训练的方法，通过奖励代理根据语言模型建议的目标来塑造探索策略，使代理朝着人类有意义且可能有用的行为方向发展，无需人类的介入。

    

    强化学习算法在没有密集且形状良好的奖励函数的情况下通常很困难。通过奖励代理访问新颖状态或转换的内在动机探索方法可以解决这个限制，但在大型环境中，这些方法对下游任务的相关性有限。我们描述了一种利用文本语料库中的背景知识来塑造探索策略的方法。这种方法称为ELLM（使用LLMs进行探索），通过给代理奖励其达成由语言模型基于代理当前状态描述所提出的目标，引导代理朝着人类有意义且可能有用的行为方向发展，无需人类的介入。我们在Crafter游戏环境和Housekeep机器人模拟器中评估了ELLM，结果表明，经过ELLM训练的代理在预训练阶段有更好的常识行为覆盖率，并且通常与人类行为相匹配。

    Reinforcement learning algorithms typically struggle in the absence of a dense, well-shaped reward function. Intrinsically motivated exploration methods address this limitation by rewarding agents for visiting novel states or transitions, but these methods offer limited benefits in large environments where most discovered novelty is irrelevant for downstream tasks. We describe a method that uses background knowledge from text corpora to shape exploration. This method, called ELLM (Exploring with LLMs) rewards an agent for achieving goals suggested by a language model prompted with a description of the agent's current state. By leveraging large-scale language model pretraining, ELLM guides agents toward human-meaningful and plausibly useful behaviors without requiring a human in the loop. We evaluate ELLM in the Crafter game environment and the Housekeep robotic simulator, showing that ELLM-trained agents have better coverage of common-sense behaviors during pretraining and usually matc
    
[^131]: 五种算法透明性和可解释性的政策应用

    Five policy uses of algorithmic transparency and explainability. (arXiv:2302.03080v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.03080](http://arxiv.org/abs/2302.03080)

    本论文通过案例研究展示了算法透明性和可解释性在政策环境中的五种应用方式：对解释的具体要求；在算法内部治理的非约束性指南中；适用于高度管制环境的法规；旨在提高算法法律责任的实用性的指南；以及对模型和数据透明性的广泛要求。

    

    "算法系统应该具有“透明性”和“可解释性”的观念在政府、公司和倡导组织制定的许多共识原则中很常见。但政策和法律行为者究竟要求这些技术概念的哪些方面，以及他们的要求与机器学习文献中开发的可解释性技术相比如何？为了更好地连接政策和技术社区，我们提供了案例研究，说明算法透明性和可解释性在政策环境中的五种应用方式：对解释的具体要求；在算法内部治理的非约束性指南中；适用于高度管制环境的法规；旨在提高算法法律责任的实用性的指南；以及对模型和数据透明性的广泛要求。案例研究涵盖了从对特定类型解释的精确要求到非具体要求的范围。

    The notion that algorithmic systems should be "transparent" and "explainable" is common in the many statements of consensus principles developed by governments, companies, and advocacy organizations. But what exactly do policy and legal actors want from these technical concepts, and how do their desiderata compare with the explainability techniques developed in the machine learning literature? In hopes of better connecting the policy and technical communities, we provide case studies illustrating five ways in which algorithmic transparency and explainability have been used in policy settings: specific requirements for explanations; in nonbinding guidelines for internal governance of algorithms; in regulations applicable to highly regulated settings; in guidelines meant to increase the utility of legal liability for algorithms; and broad requirements for model and data transparency. The case studies span a spectrum from precise requirements for specific types of explanations to nonspeci
    
[^132]: 基于频率变换的深度学习时间序列分析综述

    A Survey on Deep Learning based Time Series Analysis with Frequency Transformation. (arXiv:2302.02173v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02173](http://arxiv.org/abs/2302.02173)

    近期，频率变换（FT）在深度学习时间序列分析中得到广泛应用，显著提高了准确性和效率。本文系统回顾和总结了基于FT的深度学习时间序列模型的研究进展，并探讨了其优势、限制以及主要方法。

    

    最近，频率变换（FT）越来越多地被纳入深度学习模型中，可以显著提高时间序列分析的最新准确性和效率。频率变换的优势，如高效性和全局视角，在各种时间序列任务和应用中被迅速探索和利用，展示了频率变换作为一种新的深度学习范式在时间序列分析领域的潜力。尽管这个新兴领域受到了越来越多的关注和研究，但目前还缺乏对基于频率变换的深度学习时间序列模型的系统回顾和深入分析。目前还不清楚为什么频率变换可以提升时间序列分析的效果，以及它在该领域的限制是什么。为了填补这些空白，我们提供了一份全面的综述，系统调查和总结了基于频率变换的深度学习时间序列分析的最新研究进展。具体而言，我们探讨了主要的方法。

    Recently, frequency transformation (FT) has been increasingly incorporated into deep learning models to significantly enhance state-of-the-art accuracy and efficiency in time series analysis. The advantages of FT, such as high efficiency and a global view, have been rapidly explored and exploited in various time series tasks and applications, demonstrating the promising potential of FT as a new deep learning paradigm for time series analysis. Despite the growing attention and the proliferation of research in this emerging field, there is currently a lack of a systematic review and in-depth analysis of deep learning-based time series models with FT. It is also unclear why FT can enhance time series analysis and what its limitations in the field are. To address these gaps, we present a comprehensive review that systematically investigates and summarizes the recent research advancements in deep learning-based time series analysis with FT. Specifically, we explore the primary approaches us
    
[^133]: 使用深度学习预测超新星壳层扩张的3D时空预测方法，用于高分辨率星系模拟

    3D-Spatiotemporal Forecasting the Expansion of Supernova Shells Using Deep Learning toward High-Resolution Galaxy Simulations. (arXiv:2302.00026v2 [astro-ph.GA] UPDATED)

    [http://arxiv.org/abs/2302.00026](http://arxiv.org/abs/2302.00026)

    本文开发了一个深度学习模型，3D-MIM，用于预测超新星爆炸后的壳层扩张，通过在平滑粒子流体动力学模拟中检测并预测超新星影响粒子所在的壳层形状，解决了高分辨率星系模拟中超新星积分时间步长问题。

    

    超新星在星系形成和演化中起着重要作用。在使用大规模并行计算进行高分辨率星系模拟时，超新星的短积分时间步长成为严重瓶颈。为了解决这个问题，一种可能的解决方案是使用Hamiltonian分裂方法，即将需要短积分时间步长的区域与整个系统分开积分。为了将这种方法应用于平滑粒子流体动力学模拟中受超新星影响的粒子，我们需要在随后的全局步骤中提前检测到这些粒子所在的壳层的形状。本文中，我们开发了一个名为3D-MIM的深度学习模型来预测超新星爆炸后的壳层扩张。通过对带有粒子质量$m_{\rm gas}$ = 1 M$_\odot$的湍流云模拟进行训练，该模型能够准确地再现出各向异性的壳层形状，其中密度逐渐下降。

    Supernova (SN) plays an important role in galaxy formation and evolution. In high-resolution galaxy simulations using massively parallel computing, short integration timesteps for SNe are serious bottlenecks. This is an urgent issue that needs to be resolved for future higher-resolution galaxy simulations. One possible solution would be to use the Hamiltonian splitting method, in which regions requiring short timesteps are integrated separately from the entire system. To apply this method to the particles affected by SNe in a smoothed-particle hydrodynamics simulation, we need to detect the shape of the shell on and within which such SN-affected particles reside during the subsequent global step in advance. In this paper, we develop a deep learning model, 3D-MIM, to predict a shell expansion after a SN explosion. Trained on turbulent cloud simulations with particle mass $m_{\rm gas}$~=~1 M$_\odot$, the model accurately reproduces the anisotropic shell shape, where densities decrease by
    
[^134]: 因果性和深度生成模型中的新兴协同作用：一项综述

    Emerging Synergies in Causality and Deep Generative Models: A Survey. (arXiv:2301.12351v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12351](http://arxiv.org/abs/2301.12351)

    这项综述探讨了因果性和深度生成模型之间的新兴协同作用，阐明了将因果性原则融入DGM中的方法，以及在大规模生成模型中应用因果性的研究前沿。

    

    在人工智能领域，了解和建模数据生成过程（DGP）的追求至关重要。深度生成模型（DGM）在捕捉复杂数据分布方面表现出色，但通常在泛化能力和可解释性方面表现不足。而因果性则提供了一种结构化的方法来理解驱动数据生成的机制，并突显了这些过程中固有的因果效应动力学。虽然因果性在可解释性和外推能力方面表现出色，但却面临着高维空间中的复杂性。意识到它们之间的协同潜力，我们深入探讨了因果性和DGM的交汇点。我们阐明了因果性原则在DGM中的整合，探讨了使用DGM进行因果识别的方法，并对因果性在大规模生成模型中的新兴研究前沿，尤其是大型语言模型（LLM）中的生成性问题提供了见解。我们介绍了方法论，突出了开放的挑战和机会。

    In the field of artificial intelligence (AI), the quest to understand and model data-generating processes (DGPs) is of paramount importance. Deep generative models (DGMs) have proven adept in capturing complex data distributions but often fall short in generalization and interpretability. On the other hand, causality offers a structured lens to comprehend the mechanisms driving data generation and highlights the causal-effect dynamics inherent in these processes. While causality excels in interpretability and the ability to extrapolate, it grapples with intricacies of high-dimensional spaces. Recognizing the synergistic potential, we delve into the confluence of causality and DGMs. We elucidate the integration of causal principles within DGMs, investigate causal identification using DGMs, and navigate an emerging research frontier of causality in large-scale generative models, particularly generative large language models (LLMs). We offer insights into methodologies, highlight open cha
    
[^135]: SPEC5G：用于5G蜂窝网络协议分析的数据集

    SPEC5G: A Dataset for 5G Cellular Network Protocol Analysis. (arXiv:2301.09201v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.09201](http://arxiv.org/abs/2301.09201)

    SPEC5G是首个公共5G数据集，用于5G蜂窝网络协议的安全性分析和文本摘要。

    

    5G是第五代蜂窝网络协议，是最先进的全球无线标准，能够以提高速度和降低延迟的方式连接几乎所有人和物。因此，其发展、分析和安全性非常重要。然而，目前5G协议的开发和安全分析方法都是完全手动的，比如属性提取、协议摘要和协议规范和实现的语义分析。为了减少这种手动工作，本文提出了SPEC5G，这是首个用于自然语言处理研究的公共5G数据集。该数据集包含来自13094份蜂窝网络规范和13个网站的3,547,586个句子，总计134M个单词。通过利用在自然语言处理任务上取得最先进结果的大规模预训练语言模型，我们使用这个数据集进行与安全相关的文本分类和摘要。安全相关的文本分类可以

    5G is the 5th generation cellular network protocol. It is the state-of-the-art global wireless standard that enables an advanced kind of network designed to connect virtually everyone and everything with increased speed and reduced latency. Therefore, its development, analysis, and security are critical. However, all approaches to the 5G protocol development and security analysis, e.g., property extraction, protocol summarization, and semantic analysis of the protocol specifications and implementations are completely manual. To reduce such manual effort, in this paper, we curate SPEC5G the first-ever public 5G dataset for NLP research. The dataset contains 3,547,586 sentences with 134M words, from 13094 cellular network specifications and 13 online websites. By leveraging large-scale pre-trained language models that have achieved state-of-the-art results on NLP tasks, we use this dataset for security-related text classification and summarization. Security-related text classification ca
    
[^136]: 面向可解释的基于Transformer的时间序列预测的时间显著性检测

    Temporal Saliency Detection Towards Explainable Transformer-based Timeseries Forecasting. (arXiv:2212.07771v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.07771](http://arxiv.org/abs/2212.07771)

    这项研究提出了一种名为Temporal Saliency Detection (TSD)的方法，利用基于注意力机制的架构实现了多步时间序列预测，并通过压缩多头注意力进行显著性模式的多分辨率分析。

    

    尽管在许多基于Transformer的模型中取得了显著的进展，但长期多步预测的时间序列预测任务仍然是一个持续的挑战，特别是在可解释性方面。本文的目标是基于常用的显著性图解释DNN的思想，构建一种基于注意力机制的架构，能够通过与适当的注意力头建立连接，自动编码与显著性相关的时间模式。因此，本文介绍了一种名为Temporal Saliency Detection (TSD) 的有效方法，它在多步时间序列预测中利用了注意力机制。虽然我们提出的架构遵循常规的编码器-解码器结构，但在编码器组件中经历了重大的改进，其中我们采用了受U-Net风格架构启发的一系列信息收缩和扩展模块。TSD方法通过压缩多头注意力实现了显著性模式的多分辨率分析。

    Despite the notable advancements in numerous Transformer-based models, the task of long multi-horizon time series forecasting remains a persistent challenge, especially towards explainability. Focusing on commonly used saliency maps in explaining DNN in general, our quest is to build attention-based architecture that can automatically encode saliency-related temporal patterns by establishing connections with appropriate attention heads. Hence, this paper introduces Temporal Saliency Detection (TSD), an effective approach that builds upon the attention mechanism and applies it to multi-horizon time series prediction. While our proposed architecture adheres to the general encoder-decoder structure, it undergoes a significant renovation in the encoder component, wherein we incorporate a series of information contracting and expanding blocks inspired by the U-Net style architecture. The TSD approach facilitates the multiresolution analysis of saliency patterns by condensing multi-heads, th
    
[^137]: TIDE：用于图上深度学习的时间导数扩散

    TIDE: Time Derivative Diffusion for Deep Learning on Graphs. (arXiv:2212.02483v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.02483](http://arxiv.org/abs/2212.02483)

    本文提出了一种新方法 TIDE，通过时间导数图扩散克服了图神经网络中消息传递框架的结构限制，实现了高效地中长距离通信，并在图神经网络任务中达到了 state-of-the-art 的性能表现。

    

    图神经网络的一个重要范式是基于消息传递框架的。在这个框架中，信息通信仅在相邻节点之间实现。使用这种范式的方法的挑战是确保节点之间的高效和准确的长距离通信，因为深度卷积网络容易产生过度平滑。在本文中，我们提出了一种基于时间导数图扩散（TIDE）的新方法，以克服消息传递框架的这些结构限制。我们的方法允许优化扩散的空间范围，适用于各种任务和网络通道，从而实现中长距离通信的高效率。此外，我们还展示了我们的架构设计也使本地消息传递成为可能，从而继承了本地消息传递方法的能力。我们在广泛使用的图基准和合成网格和图数据集上展示，所提出的框架优于	state-of-the-art 方法。

    A prominent paradigm for graph neural networks is based on the message-passing framework. In this framework, information communication is realized only between neighboring nodes. The challenge of approaches that use this paradigm is to ensure efficient and accurate long-distance communication between nodes, as deep convolutional networks are prone to oversmoothing. In this paper, we present a novel method based on time derivative graph diffusion (TIDE) to overcome these structural limitations of the message-passing framework. Our approach allows for optimizing the spatial extent of diffusion across various tasks and network channels, thus enabling medium and long-distance communication efficiently. Furthermore, we show that our architecture design also enables local message-passing and thus inherits from the capabilities of local message-passing approaches. We show that on both widely used graph benchmarks and synthetic mesh and graph datasets, the proposed framework outperforms state-
    
[^138]: 关于未观察到的混淆因素下的反事实推断

    On counterfactual inference with unobserved confounding. (arXiv:2211.08209v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.08209](http://arxiv.org/abs/2211.08209)

    本研究提出了一种在观测研究中应对未观察到混淆因素进行反事实推断的方法，通过建模条件分布，学习了各单位的反事实分布，并提供了一个均方误差的界限。

    

    在观测研究中，我们的目标是利用每个单位只有一个包含协变量、干预和结果的$p$维样本来学习每个单位的反事实分布，这些单位是独立但异质的。具体而言，我们允许存在未观察到的混淆因素，它引入了干预和结果之间的统计偏差，并加剧了单位之间的异质性。将结果的条件分布建模为指数族，我们将学习单位级反事实分布简化为学习具有异质参数和仅有一个样本的$n$个指数族分布。我们引入了一个凸优化目标，将所有$n$个样本汇集起来共同学习所有$n$个参数向量，并提供了一个单位级均方误差的界限，该界限与参数空间的度量熵成线性关系。例如，当参数是$k$个已知向量的$s$稀疏线性组合时，误差为$O(k\sqrt{\frac{s\log p}{n}})$。

    Given an observational study with $n$ independent but heterogeneous units, our goal is to learn the counterfactual distribution for each unit using only one $p$-dimensional sample per unit containing covariates, interventions, and outcomes. Specifically, we allow for unobserved confounding that introduces statistical biases between interventions and outcomes as well as exacerbates the heterogeneity across units. Modeling the conditional distribution of the outcomes as an exponential family, we reduce learning the unit-level counterfactual distributions to learning $n$ exponential family distributions with heterogeneous parameters and only one sample per distribution. We introduce a convex objective that pools all $n$ samples to jointly learn all $n$ parameter vectors, and provide a unit-wise mean squared error bound that scales linearly with the metric entropy of the parameter space. For example, when the parameters are $s$-sparse linear combination of $k$ known vectors, the error is $
    
[^139]: BAFFLE: 离线增强学习中的后门攻击

    BAFFLE: Backdoor Attack in Offline Reinforcement Learning. (arXiv:2210.04688v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.04688](http://arxiv.org/abs/2210.04688)

    本文研究离线增强学习中的后门攻击，通过向数据中添加扰动，使得智能体在注入触发器的观测值上采取低奖励动作，从而提出了BAFFLE方法。

    

    越来越多的研究关注于强化学习（RL）方法，允许智能体通过与环境的交互中收集的试错经验进行学习。最近，离线RL成为一种流行的RL范例，因为它节省了与环境的交互。在离线RL中，数据提供者共享大规模的预先收集的数据集，其他人可以在不与环境交互的情况下训练高质量的智能体。这种范例在机器人控制、自动驾驶等关键任务中表现出有效性。然而，较少关注研究离线RL系统的安全威胁。本文关注后门攻击，其中一些扰动被添加到数据（观测值）中，使得在给定正常观测值的情况下，智能体采取高奖励的动作，在注入触发器的观测值上采取低奖励的动作。在本文中，我们提出了BAFFLE（离线增强学习中的后门攻击），这是一种方法。

    A growing body of research has focused on the Reinforcement Learning (RL) methods which allow the agent to learn from trial-and-error experiences gathered during the interaction with the environment. Recently, offline RL becomes a popular RL paradigm because it saves the interactions with environments. In offline RL, data providers share large pre-collected datasets, and others can train high-quality agents without interacting with the environments. This paradigm has demonstrated effectiveness in critical tasks like robot control, autonomous driving, etc. However, less attention is paid to investigating the security threats to the offline RL system. This paper focuses on backdoor attacks, where some perturbations are added to the data (observations) such that given normal observations, the agent takes high-rewards actions, and low-reward actions on observations injected with triggers. In this paper, we propose Baffle (Backdoor Attack for Offline Reinforcement Learning), an approach tha
    
[^140]: 多感官集成在深度网络中的关键学习期

    Critical Learning Periods for Multisensory Integration in Deep Networks. (arXiv:2210.04643v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.04643](http://arxiv.org/abs/2210.04643)

    对于多感官集成，神经网络在早期训练阶段接受适当相关信号至关重要，而干扰学习过程可能会永久损害技能的发展。早期瞬态动力学对最终的系统性能和学习表示具有决定性影响。

    

    我们展示了神经网络整合来自不同来源信息的能力在早期训练阶段接受适当相关信号的情况下至关重要。干扰学习过程可能会永久损害技能的发展，无论是在人造系统还是生物系统中，这种现象被称为关键学习期。我们展示了关键学习期源于复杂而不稳定的早期瞬态动力学，这对训练系统的最终性能和学习表示具有决定性影响。这一证据挑战了通过分析宽而浅的网络得出的认为神经网络的早期学习动态是简单的、类似于线性模型的观点。实际上，我们展示了即使是深度线性网络在多源集成方面也会出现关键学习期，而浅层网络则不会。

    We show that the ability of a neural network to integrate information from diverse sources hinges critically on being exposed to properly correlated signals during the early phases of training. Interfering with the learning process during this initial stage can permanently impair the development of a skill, both in artificial and biological systems where the phenomenon is known as a critical learning period. We show that critical periods arise from the complex and unstable early transient dynamics, which are decisive of final performance of the trained system and their learned representations. This evidence challenges the view, engendered by analysis of wide and shallow networks, that early learning dynamics of neural networks are simple, akin to those of a linear model. Indeed, we show that even deep linear networks exhibit critical learning periods for multi-source integration, while shallow networks do not. To better understand how the internal representations change according to di
    
[^141]: DPA-1: 运用注意力机制的深度势能模型在分子模拟中的预训练

    DPA-1: Pretraining of Attention-based Deep Potential Model for Molecular Simulation. (arXiv:2208.08236v4 [physics.chem-ph] UPDATED)

    [http://arxiv.org/abs/2208.08236](http://arxiv.org/abs/2208.08236)

    DPA-1是一种具有新颖注意力机制的深度势能模型，能够高效表示原子系统的构象和化学空间，并且在分子模拟中能够通过预训练和微调取得卓越的性能。

    

    机器学习辅助建模的原子间势能能量面（PES）正彻底改变分子模拟领域。随着高质量电子结构数据的积累，一个能够预先训练所有可用数据并在下游任务中通过少量额外工作进行微调的模型将使该领域进入一个新阶段。本文提出了DPA-1，一种具有新颖注意力机制的深度势能模型，对原子系统的构象和化学空间具有高效表示能力，并且能够学习到PES。我们在多个系统上测试了DPA-1，并观察到与现有基准相比表现出更好的性能。当在包含56个元素的大规模数据集上进行预训练时，DPA-1可以在各种下游任务中取得极大的样本效率改进。令人惊讶的是，对于不同的元素，学到的类型嵌入参数在潜在空间中形成了一个"螺旋"形状，并且与它们的p

    Machine learning assisted modeling of the inter-atomic potential energy surface (PES) is revolutionizing the field of molecular simulation. With the accumulation of high-quality electronic structure data, a model that can be pretrained on all available data and finetuned on downstream tasks with a small additional effort would bring the field to a new stage. Here we propose DPA-1, a Deep Potential model with a novel attention mechanism, which is highly effective for representing the conformation and chemical spaces of atomic systems and learning the PES. We tested DPA-1 on a number of systems and observed superior performance compared with existing benchmarks. When pretrained on large-scale datasets containing 56 elements, DPA-1 can be successfully applied to various downstream tasks with a great improvement of sample efficiency. Surprisingly, for different elements, the learned type embedding parameters form a $spiral$ in the latent space and have a natural correspondence with their p
    
[^142]: 连续时间线性二次强化学习中熵正则化的最优调度

    Optimal scheduling of entropy regulariser for continuous-time linear-quadratic reinforcement learning. (arXiv:2208.04466v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.04466](http://arxiv.org/abs/2208.04466)

    本研究利用熵正则化的松弛随机控制视角设计了连续时间线性二次强化学习算法，并通过探索性控制方法和近端策略更新方法实现了探索和利用的权衡，以解决有限时间线性二次强化学习问题。

    

    本研究使用熵正则化的松弛随机控制视角作为设计强化学习算法的基础框架。在这里，Agent通过生成符合最优松弛策略的噪声控制与环境进行交互。噪声策略一方面可以探索空间并促进学习，但另一方面也会引入偏差，将正概率分配给非最优动作。这种探索与利用的权衡由熵正则化的强度来确定。我们研究了两种熵正则化形式得到的算法：探索性控制方法，在成本目标中添加熵；近端策略更新方法，在连续的Episode之间对策略差异进行熵惩罚。我们重点研究了有限时间连续时间线性二次强化学习问题，其中具有未知漂移系数的线性动力学受到四次方约束的控制。

    This work uses the entropy-regularised relaxed stochastic control perspective as a principled framework for designing reinforcement learning (RL) algorithms. Herein agent interacts with the environment by generating noisy controls distributed according to the optimal relaxed policy. The noisy policies on the one hand, explore the space and hence facilitate learning but, on the other hand, introduce bias by assigning a positive probability to non-optimal actions. This exploration-exploitation trade-off is determined by the strength of entropy regularisation. We study algorithms resulting from two entropy regularisation formulations: the exploratory control approach, where entropy is added to the cost objective, and the proximal policy update approach, where entropy penalises policy divergence between consecutive episodes. We focus on the finite horizon continuous-time linear-quadratic (LQ) RL problem, where a linear dynamics with unknown drift coefficients is controlled subject to quadr
    
[^143]: Tac2Pose：从第一次接触中的触觉对象姿态估计

    Tac2Pose: Tactile Object Pose Estimation from the First Touch. (arXiv:2204.11701v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2204.11701](http://arxiv.org/abs/2204.11701)

    Tac2Pose是一种从第一次触觉中估计物体姿态的方法，通过在仿真中学习物体的感知模型，根据触觉观测估计可能的物体姿态，并通过对比学习进行匹配。这种方法只需要一次真实触觉观测即可定位物体。

    

    本文介绍了Tac2Pose，一种针对已知对象的从第一次接触中估计触觉姿态的对象特定方法。给定物体几何形状，我们在仿真中学习了一个量身定制的感知模型，可以根据触觉观测来估计可能的物体姿态的概率分布。为此，我们模拟了一组密集的物体姿态在传感器上产生的接触形状。然后，给定从传感器中获得的新接触形状，我们使用对比学习学习的对象特定嵌入将其与预先计算的集合进行匹配。我们使用针对对象无关的校准步骤将RGB触觉观测映射到二值接触形状，从传感器中获得接触形状。这个映射可以在对象和传感器实例之间重复使用，是唯一使用真实传感器数据进行训练的步骤。这样就可以通过第一次真实触觉观测来定位物体。重要的是，它可以产生姿态分布，并可以将其他传感器数据整合到姿态估计中。

    In this paper, we present Tac2Pose, an object-specific approach to tactile pose estimation from the first touch for known objects. Given the object geometry, we learn a tailored perception model in simulation that estimates a probability distribution over possible object poses given a tactile observation. To do so, we simulate the contact shapes that a dense set of object poses would produce on the sensor. Then, given a new contact shape obtained from the sensor, we match it against the pre-computed set using an object-specific embedding learned using contrastive learning. We obtain contact shapes from the sensor with an object-agnostic calibration step that maps RGB tactile observations to binary contact shapes. This mapping, which can be reused across object and sensor instances, is the only step trained with real sensor data. This results in a perception model that localizes objects from the first real tactile observation. Importantly, it produces pose distributions and can incorpor
    
[^144]: GP-BART: 一种使用高斯过程的新型贝叶斯加法回归树方法

    GP-BART: a novel Bayesian additive regression trees approach using Gaussian processes. (arXiv:2204.02112v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2204.02112](http://arxiv.org/abs/2204.02112)

    GP-BART是一种使用高斯过程的新型贝叶斯加法回归树方法，相比标准BART模型，它具有更好的平滑性和明确的协方差结构假设，在多种情境下显示出超越传统建模方法的性能。

    

    贝叶斯加法回归树 (BART) 模型是一种集成方法，由于其始终强大的预测性能和量化不确定性的能力，在回归任务中广泛且成功地使用。BART通过一组缩减先验将“弱”树模型组合起来，其中每个树解释了数据中的一小部分变异性。然而，标准BART模型中缺乏平滑性并且对观测值之间的协方差结构没有明确假设，这在需要这些假设的情况下可能导致性能较差。高斯过程贝叶斯加法回归树 (GP-BART) 模型是对BART的扩展，它通过假设每个终端节点的预测服从高斯过程先验来解决这一限制。通过对模拟和实际数据的应用来证明了模型的有效性，在各种情境下超越了传统建模方法的性能。

    The Bayesian additive regression trees (BART) model is an ensemble method extensively and successfully used in regression tasks due to its consistently strong predictive performance and its ability to quantify uncertainty. BART combines "weak" tree models through a set of shrinkage priors, whereby each tree explains a small portion of the variability in the data. However, the lack of smoothness and the absence of an explicit covariance structure over the observations in standard BART can yield poor performance in cases where such assumptions would be necessary. The Gaussian processes Bayesian additive regression trees (GP-BART) model is an extension of BART which addresses this limitation by assuming Gaussian process (GP) priors for the predictions of each terminal node among all trees. The model's effectiveness is demonstrated through applications to simulated and real-world data, surpassing the performance of traditional modeling approaches in various scenarios.
    
[^145]: 量子密度矩阵在经典问答和图像分类中的应用

    Application of Quantum Density Matrix in Classical Question Answering and Classical Image Classification. (arXiv:2203.11155v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2203.11155](http://arxiv.org/abs/2203.11155)

    该论文将量子密度矩阵应用于经典问答和图像分类中，证明了其可以提高任务的效率，尤其在图像分类中取得了优秀的性能表现。

    

    量子密度矩阵可表示整个量子系统的全部信息，将密度矩阵用于经典问答任务可以更加有效地实现问题回答。本论文设计了一种基于LSTM的新机制，以应对输入为矩阵的情况，并将该机制应用于卷积神经网络进行QA问题的求解，同时也证明了量子密度矩阵可以增强经典图像分类中的特征信息和特征之间的关系。实验结果表明，该新框架在CIFAR-10数据集上的性能优于传统的基于CNN的分类方法。

    Quantum density matrix represents all the information of the entire quantum system, and novel models of meaning employing density matrices naturally model linguistic phenomena such as hyponymy and linguistic ambiguity, among others in quantum question answering tasks. Naturally, we argue that applying the quantum density matrix into classical Question Answering (QA) tasks can show more effective performance. Specifically, we (i) design a new mechanism based on Long Short-Term Memory (LSTM) to accommodate the case when the inputs are matrixes; (ii) apply the new mechanism to QA problems with Convolutional Neural Network (CNN) and gain the LSTM-based QA model with the quantum density matrix. Experiments of our new model on TREC-QA and WIKI-QA data sets show encouraging results. Similarly, we argue that the quantum density matrix can also enhance the image feature information and the relationship between the features for the classical image classification. Thus, we (i) combine density mat
    
[^146]: 不要误会我：如何将深度视觉解释应用于时间序列

    Don't Get Me Wrong: How to Apply Deep Visual Interpretations to Time Series. (arXiv:2203.07861v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2203.07861](http://arxiv.org/abs/2203.07861)

    该论文提出了一个针对时间序列分类和分割任务的框架，通过六个度量来评估基于梯度、传播或干扰的事后可视化解释方法。实验结果表明，这些方法对于时间序列的解释具有较高的可信度和有效性。

    

    在许多应用中，正确解释和理解深度学习模型非常重要。针对图像和自然语言处理的解释性视觉解释方法允许领域专家验证和理解几乎任何深度学习模型。然而，当推广到任意时间序列时，它们在本质上更加复杂和多样化。一个可视化解释是否解释了有效的推理或捕捉了实际特征是难以判断的。因此，我们需要客观评估来获得可信的质量指标，而不是盲目信任。我们提出了一个框架，包括六个正交度量，用于针对时间序列分类和分割任务的基于梯度、传播或干扰的事后视觉解释方法。实验研究包括了常见的时间序列神经网络架构和九种可视化解释方法。我们使用UCR r等多样的数据集评估了这些可视化解释方法。

    The correct interpretation and understanding of deep learning models are essential in many applications. Explanatory visual interpretation approaches for image, and natural language processing allow domain experts to validate and understand almost any deep learning model. However, they fall short when generalizing to arbitrary time series, which is inherently less intuitive and more diverse. Whether a visualization explains valid reasoning or captures the actual features is difficult to judge. Hence, instead of blind trust, we need an objective evaluation to obtain trustworthy quality metrics. We propose a framework of six orthogonal metrics for gradient-, propagation- or perturbation-based post-hoc visual interpretation methods for time series classification and segmentation tasks. An experimental study includes popular neural network architectures for time series and nine visual interpretation methods. We evaluate the visual interpretation methods with diverse datasets from the UCR r
    
[^147]: HAKE:人类活动理解的知识引擎基础

    HAKE: A Knowledge Engine Foundation for Human Activity Understanding. (arXiv:2202.06851v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2202.06851](http://arxiv.org/abs/2202.06851)

    本论文提出了一个名为HAKE的知识引擎，用于人类活动理解。该引擎通过将像素映射到中间空间，并使用逻辑规则推断语义，展现出了优越的泛化能力和性能。

    

    人类活动理解在人工智能领域引起了广泛的关注，涉及到健康护理和行为分析等多个应用领域。尽管深度学习取得了一些进展，但仍然面临挑战。通常，像物体识别一样的解决方案试图直接将像素映射到语义，但活动模式与物体模式非常不同，从而阻碍了成功。在这项工作中，我们提出了一种新的范例，将任务分为两个阶段：首先将像素映射到由原子活动基元构成的中间空间，然后使用可解释的逻辑规则对检测到的基元进行编程以推断语义。为了得到一个具有代表性的基元空间，我们构建了一个包含26+M个基元标签和逻辑规则的知识库，这些规则是通过人类先验知识或自动发现得到的。我们的框架，人类活动知识引擎（HAKE），在具有挑战性的基准测试上表现出了卓越的泛化能力和性能。

    Human activity understanding is of widespread interest in artificial intelligence and spans diverse applications like health care and behavior analysis. Although there have been advances in deep learning, it remains challenging. The object recognition-like solutions usually try to map pixels to semantics directly, but activity patterns are much different from object patterns, thus hindering success. In this work, we propose a novel paradigm to reformulate this task in two stages: first mapping pixels to an intermediate space spanned by atomic activity primitives, then programming detected primitives with interpretable logic rules to infer semantics. To afford a representative primitive space, we build a knowledge base including 26+ M primitive labels and logic rules from human priors or automatic discovering. Our framework, the Human Activity Knowledge Engine (HAKE), exhibits superior generalization ability and performance upon canonical methods on challenging benchmarks. Code and data
    
[^148]: 物理增强的深度代理用于偏微分方程

    Physics-enhanced deep surrogates for PDEs. (arXiv:2111.05841v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2111.05841](http://arxiv.org/abs/2111.05841)

    这种物理增强的深度代理方法通过结合低保真度的物理模拟器和神经网络生成器来开发复杂物理系统的快速代理模型，能够在精确性和成本之间取得更好的平衡。

    

    许多物理和工程应用需要传统上用资源密集型的高保真数值求解器计算的偏微分方程属性评估。数据驱动的代理模型提供了一种高效的替代方法，但训练成本很高。新兴应用将获益于具有改进的准确性-成本平衡的代理模型，同时在大规模上进行研究。本文提出了一种"物理增强的深度代理"（"PEDS"）方法，用于开发复杂物理系统的快速代理模型，该系统由偏微分方程描述。具体而言，提出了低保真度可解释的物理模拟器和神经网络生成器的组合，通过端到端训练全局匹配昂贵高保真数值求解器的输出。在扩散、反应扩散和电磁散射模型的三个示例测试用例上的实验证明，PEDS代理比一个例子加上的。。

    Many physics and engineering applications demand Partial Differential Equations (PDE) property evaluations that are traditionally computed with resource-intensive high-fidelity numerical solvers. Data-driven surrogate models provide an efficient alternative but come with a significant cost of training. Emerging applications would benefit from surrogates with an improved accuracy-cost tradeoff, while studied at scale. Here we present a "physics-enhanced deep-surrogate" ("PEDS") approach towards developing fast surrogate models for complex physical systems, which is described by PDEs. Specifically, a combination of a low-fidelity, explainable physics simulator and a neural network generator is proposed, which is trained end-to-end to globally match the output of an expensive high-fidelity numerical solver. Experiments on three exemplar testcases, diffusion, reaction-diffusion, and electromagnetic scattering models, show that a PEDS surrogate can be up to 3$\times$ more accurate than an e
    
[^149]: MixStyle神经网络用于领域泛化和适应性的翻译和摘要机器人。

    MixStyle Neural Networks for Domain Generalization and Adaptation. (arXiv:2107.02053v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2107.02053](http://arxiv.org/abs/2107.02053)

    MixStyle是一个简单的模块，用于提高神经网络对于领域转移的泛化性能。它通过在训练过程中混合两个随机实例的特征统计来合成新领域，从而实现数据增强。MixStyle易于实现，适用于各类学习范式。

    

    神经网络在具有领域转移的未见数据上表现不佳，这是机器学习和人工智能中的一个长期存在的问题。为了克服这个问题，我们提出了MixStyle，这是一个简单的即插即用，无需参数的模块，可以提高领域泛化性能，而无需收集更多的数据或增加模型容量。MixStyle的设计很简单：在训练过程中，在一个前向传播中将两个随机实例的特征统计混合。这个想法是基于最新的风格转换研究发现的，特征统计捕捉到图像风格信息，而图像风格本质上定义了视觉领域。因此，混合特征统计可以被看作是在特征空间中合成新领域的一种高效方式，从而实现了数据增强。MixStyle很容易用几行代码实现，不需要修改训练目标，并且可以适用于各种学习范式，包括监督领域泛化，半监督领域自适应等。

    Neural networks do not generalize well to unseen data with domain shifts -- a longstanding problem in machine learning and AI. To overcome the problem, we propose MixStyle, a simple plug-and-play, parameter-free module that can improve domain generalization performance without the need to collect more data or increase model capacity. The design of MixStyle is simple: it mixes the feature statistics of two random instances in a single forward pass during training. The idea is grounded by the finding from recent style transfer research that feature statistics capture image style information, which essentially defines visual domains. Therefore, mixing feature statistics can be seen as an efficient way to synthesize new domains in the feature space, thus achieving data augmentation. MixStyle is easy to implement with a few lines of code, does not require modification to training objectives, and can fit a variety of learning paradigms including supervised domain generalization, semi-supervi
    
[^150]: 在循环存在的情况下，基于约束的因果推断利用部分祖先图

    Constraint-Based Causal Discovery using Partial Ancestral Graphs in the presence of Cycles. (arXiv:2005.00610v3 [math.ST] UPDATED)

    [http://arxiv.org/abs/2005.00610](http://arxiv.org/abs/2005.00610)

    本研究证明了在涉及反馈的系统生成的观察数据中，应用Fast Causal Inference (FCI)算法可以得到正确的结果，该算法可以被用于一致地估计因果关系的存在和缺失、直接因果关系的存在和缺失、混淆因素的缺失以及因果图中特定循环的缺失。

    

    虽然反馈回路在许多复杂系统中起着重要作用，但在大部分因果推断文献中忽视了它们的存在，因为通常假设系统从一开始就是非循环的。当将为非循环环境设计的因果推断算法应用于涉及反馈的系统生成的数据时，我们不会期望得到正确的结果。本研究表明，出人意料的是，快速因果推断（FCI）算法在应用于涉及反馈的系统生成的观察数据时的输出是正确的。具体而言，我们证明了对于由简单且$\sigma$-可信结构性因果模型（SCM）生成的观察数据，FCI是可靠而完整的，并且可以用于一致地估计：（i）因果关系的存在和缺失，（ii）直接因果关系的存在和缺失，（iii）混淆因素的缺失，以及（iv）因果图中特定循环的缺失。

    While feedback loops are known to play important roles in many complex systems, their existence is ignored in a large part of the causal discovery literature, as systems are typically assumed to be acyclic from the outset. When applying causal discovery algorithms designed for the acyclic setting on data generated by a system that involves feedback, one would not expect to obtain correct results. In this work, we show that -- surprisingly -- the output of the Fast Causal Inference (FCI) algorithm is correct if it is applied to observational data generated by a system that involves feedback. More specifically, we prove that for observational data generated by a simple and $\sigma$-faithful Structural Causal Model (SCM), FCI is sound and complete, and can be used to consistently estimate (i) the presence and absence of causal relations, (ii) the presence and absence of direct causal relations, (iii) the absence of confounders, and (iv) the absence of specific cycles in the causal graph o
    

