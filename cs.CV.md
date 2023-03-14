# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generalizing and Decoupling Neural Collapse via Hyperspherical Uniformity Gap.](http://arxiv.org/abs/2303.06484) | 本文提出了一个广义神经坍塌假设，有效地包含了原始神经坍塌，并将其分解为两个目标：最小化类内变异性和最大化类间可分性。使用超球统一性作为量化这两个目标的统一框架，并提出了一个通用目标——超球统一性差（HUG），它由类间和类内超球统一性之间的差异定义。 |
| [^2] | [Knowledge Distillation for Efficient Sequences of Training Runs.](http://arxiv.org/abs/2303.06480) | 本文研究了如何利用先前运行中的计算来减少未来运行成本的问题，使用知识蒸馏（KD），通过将未来运行与来自先前运行的KD相结合，可以显著减少训练这些模型所需的时间，KD的开销降低了80-90％，对准确性影响很小，并在整体成本方面实现了巨大的帕累托改进。 |
| [^3] | [ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation.](http://arxiv.org/abs/2303.06458) | ZeroNLG是一个零样本学习框架，可以处理多个NLG任务，包括图像到文本、视频到文本和文本到文本，跨越英语、中文、德语和法语。它不需要任何标记的下游对进行训练，通过将不同的领域投影到共享的公共潜在空间中的相应坐标，桥接不同领域之间的差异。 |
| [^4] | [DECOMPL: Decompositional Learning with Attention Pooling for Group Activity Recognition from a Single Volleyball Image.](http://arxiv.org/abs/2303.06439) | 本文提出了一种新的排球视频团体活动识别技术DECOMPL，它由两个互补的分支组成，使用选择性的注意力池化提取特征，考虑参与者的当前配置，并从框坐标中提取空间信息。同时，本文发现排球数据集的标签方案降低了活动中的团体概念。 |
| [^5] | [Probing neural representations of scene perception in a hippocampally dependent task using artificial neural networks.](http://arxiv.org/abs/2303.06367) | 本研究利用人工神经网络探究场景感知的神经表征在海马依赖任务中的应用，设计了一个新型场景感知基准，证明了DNNs可以学习从不同自我中心视角观察的场景的能力。 |
| [^6] | [Explainable AI for Time Series via Virtual Inspection Layers.](http://arxiv.org/abs/2303.06365) | 本文提出了一种虚拟检查层，将时间序列转换为可解释的表示，并允许通过本地XAI方法将相关性归因传播到该表示。我们将一系列XAI方法的适用性扩展到需要转换后才能解释输入的领域。我们展示了DFT-LRP在各种时间序列分类设置中的有用性，如音频和电子健康记录。 |
| [^7] | [MetaViewer: Towards A Unified Multi-View Representation.](http://arxiv.org/abs/2303.06329) | 该论文提出了一种新颖的基于双层优化的多视图学习框架MetaViewer，通过统一到特定的方式学习表示，避免了手动预先指定的融合函数和混合在特征中的视图专用冗余信息可能会降低所得表示的质量的问题。 |
| [^8] | [MLP-SRGAN: A Single-Dimension Super Resolution GAN using MLP-Mixer.](http://arxiv.org/abs/2303.06298) | MLP-SRGAN是一种单维超分辨率GAN，使用MLP-Mixer和卷积层进行上采样，可用于FLAIR MRI图像的超分辨率重建，提出了新的图像质量度量方法。 |
| [^9] | [Stabilizing Transformer Training by Preventing Attention Entropy Collapse.](http://arxiv.org/abs/2303.06296) | 本文研究了Transformer的训练动态，发现低注意力熵伴随着高训练不稳定性，提出了一种简单而有效的解决方案$\sigma$Reparam，成功地防止了注意力层中的熵崩溃，促进了更稳定的训练。 |
| [^10] | [CoNIC Challenge: Pushing the Frontiers of Nuclear Detection, Segmentation, Classification and Counting.](http://arxiv.org/abs/2303.06274) | CoNIC挑战使用最大的数据集评估核分割和细胞组成，刺激了可重复的细胞识别算法的开发，发现嗜酸性粒细胞和中性粒细胞在肿瘤中发挥重要作用。 |
| [^11] | [HYperbolic Self-Paced Learning for Self-Supervised Skeleton-based Action Representations.](http://arxiv.org/abs/2303.06242) | 本文提出了一种新的超似曲自适应模型（HYSP）用于学习基于骨架的动作表示，采用自监督学习，使用数据增强来生成同一样本的两个视图，并通过将一个视图与另一个视图匹配来学习，使用超似曲不确定性来确定算法学习速度，假设不确定性较小的样本应更强烈地推动训练，具有更大的权重和速度。 |
| [^12] | [Do we need entire training data for adversarial training?.](http://arxiv.org/abs/2303.06241) | 本文提出了一种新的对抗训练方法，通过对训练数据集进行筛选，仅使用易受对抗攻击的样本进行训练，从而减少训练时间。 |
| [^13] | [Compressive Sensing with Tensorized Autoencoder.](http://arxiv.org/abs/2303.06235) | 本文提出了一种利用张量环因子分解的自动编码器来恢复图像的方法，该方法在修复和去噪应用中表现出更好的重建质量。 |
| [^14] | [MCROOD: Multi-Class Radar Out-Of-Distribution Detection.](http://arxiv.org/abs/2303.06232) | 本文提出了一种基于重建的多类OOD检测器，该检测器在雷达距离多普勒图像（RDIs）上运行。检测器旨在将除坐、站或走的人以外的任何移动物体分类为OOD。作者还提供了一种简单而有效的预处理技术，以检测呼吸等微小的人体运动。在实验中，该方法表现优于最先进的OOD检测方法。 |
| [^15] | [A POV-based Highway Vehicle Trajectory Dataset and Prediction Architecture.](http://arxiv.org/abs/2303.06202) | 介绍了一个基于POV的高速公路车辆轨迹数据集和预测架构，其中的Carolinas Highway Dataset（CHD）包括160万帧和338,000个车辆轨迹。 |
| [^16] | [Optimizing Federated Learning for Medical Image Classification on Distributed Non-iid Datasets with Partial Labels.](http://arxiv.org/abs/2303.06180) | 本文提出了FedFBN，一个联邦学习框架，使用预训练的网络作为模型后端，并在整个训练过程中冻结批量归一化层，以优化分布式非独立同分布数据和部分标签的医学图像分类。 |
| [^17] | [Overcoming Bias in Pretrained Models by Manipulating the Finetuning Dataset.](http://arxiv.org/abs/2303.06167) | 本文研究了预训练模型中的偏见问题，发现微调模型可以继承预训练模型的偏见，但通过对微调数据集进行干预可以纠正这种偏见，而且对性能的影响很小。这表明仔细策划微调数据集对于减少下游任务中的偏见非常重要，这样做甚至可以弥补预训练模型中的偏见。 |
| [^18] | [Learning the Legibility of Visual Text Perturbations.](http://arxiv.org/abs/2303.05077) | 本文提出了一种学习模型来预测扰动字符串的易读性，并根据其易读性对候选扰动进行排名的方法，填补了保持易读性的文本扰动的系统性表征的空白。 |
| [^19] | [Prismer: A Vision-Language Model with An Ensemble of Experts.](http://arxiv.org/abs/2303.02506) | Prismer是一种数据和参数高效的视觉语言模型，它利用了一组领域专家的集合，通过汇集这些专家知识并将其适应于各种视觉语言推理任务，实现了与当前最先进模型竞争的微调和少样本学习性能，同时需要少至两个数量级的训练数据。 |
| [^20] | [Mesh-SORT: Simple and effective location-wise tracker with lost management strategies.](http://arxiv.org/abs/2302.14415) | Mesh-SORT是一种简单而有效的位置跟踪器，通过网格分割帧并提出相应的位置感知的丢失管理策略和不同的匹配策略，解决了跟踪中的不良检测器和遮挡问题。 |
| [^21] | [Rethinking Semi-Supervised Medical Image Segmentation: A Variance-Reduction Perspective.](http://arxiv.org/abs/2302.01735) | 本文提出了ARCO，一种半监督对比学习（CL）框架，其中包括医学图像分割中的分层组采样理论。通过方差缩减估计的概念来构建ARCO，并表明某些方差缩减技术在医学图像分割中特别有益。 |
| [^22] | [LDMIC: Learning-based Distributed Multi-view Image Coding.](http://arxiv.org/abs/2301.09799) | LDMIC是一种基于学习的分布式多视图图像编码框架，通过独立编码器和联合上下文传输模块实现了全局视图间的相关性捕捉，对几何关系不敏感。 |
| [^23] | [SegViz: A federated-learning based framework for multi-organ segmentation on heterogeneous data sets with partial annotations.](http://arxiv.org/abs/2301.07074) | SegViz是一种基于联邦学习的框架，用于从分布式的非i.i.d数据集中训练具有部分注释的分割模型。使用FedBN作为聚合策略的SegViz框架在外部BTCV集上表现出优异的性能，分割的dice分数分别为0.93、0.83、0.55和0.75。 |
| [^24] | [Image Data Augmentation Approaches: A Comprehensive Survey and Future directions.](http://arxiv.org/abs/2301.02830) | 本文综述了图像数据增强方法，提供了全面的分类法和每种技术的优缺点，并给出了数据增强对图像分类、目标检测和语义分割等计算机视觉任务的全面结果。 |
| [^25] | [One-shot domain adaptation in video-based assessment of surgical skills.](http://arxiv.org/abs/2301.00812) | 本文提出了一种元学习模型A-VBANet，可以通过一次性学习提供领域不可知的手术技能分类，成功地适应了模拟任务和腹腔镜胆囊切除术，为基于视频的手术技能评估提供了领域不可知程序。 |
| [^26] | [UniDA3D: Unified Domain Adaptive 3D Semantic Segmentation Pipeline.](http://arxiv.org/abs/2212.10390) | 本文提出了UniDA3D，一种统一的域自适应三维语义分割管道，通过设计统一的源和目标主动采样策略，可以解决三维分割领域中的多个自适应任务，并探索了实现多模态采样策略的可能性。 |
| [^27] | [Fine-tuned CLIP Models are Efficient Video Learners.](http://arxiv.org/abs/2212.03640) | 本文提出了一个简单的视频细调CLIP（ViFi-CLIP）基线，通过从CLIP图像编码器的帧级处理，接着进行特征池化和与相应文本嵌入的相似度匹配，有效地将图像级别的CLIP表示转移到视频中，从而弥合了从图像到视频的领域差距。 |
| [^28] | [P{\O}DA: Prompt-driven Zero-shot Domain Adaptation.](http://arxiv.org/abs/2212.03241) | 本文提出了一种基于提示的零样本领域自适应方法，通过利用预训练的对比视觉语言模型（CLIP）来优化源特征的仿射变换，将其引导到目标文本嵌入中，同时保留其内容和语义。实验表明，该方法在几个数据集上显著优于基于CLIP的风格转移基线，用于下游任务。 |
| [^29] | [Unifying Vision, Text, and Layout for Universal Document Processing.](http://arxiv.org/abs/2212.02623) | 本文提出了通用文档处理（UDOP）模型，将文本、图像和布局模态以及各种任务格式统一起来，通过一种新颖的Transformer模型实现预训练和多域下游任务的统一，同时实现了高质量的神经文档编辑和内容定制。 |
| [^30] | [Dunhuang murals contour generation network based on convolution and self-attention fusion.](http://arxiv.org/abs/2212.00935) | 本文提出了一种基于卷积和自注意力融合的敦煌壁画轮廓生成网络，旨在解决传统卷积神经网络在感受野扩大时丢失局部细节信息的问题，以生成合理的壁画轮廓图。 |
| [^31] | [From Forks to Forceps: A New Framework for Instance Segmentation of Surgical Instruments.](http://arxiv.org/abs/2211.16200) | 本研究提出了一种新的神经网络框架，将分类模块作为现有实例分割模型的新阶段添加，用于改善手术器械实例分割中的分类问题。该模块包括多尺度掩模注意力，用于关注器械区域并掩盖分散的背景特征。 |
| [^32] | [VideoFACT: Detecting Video Forgeries Using Attention, Scene Context, and Forensic Traces.](http://arxiv.org/abs/2211.15775) | 本文提出了一种新的网络VideoFACT，它利用取证嵌入、上下文嵌入和深度自我注意机制来检测和定位各种视频伪造和操纵，克服了现有网络在分析视频时面临的挑战。 |
| [^33] | [SinFusion: Training Diffusion Models on a Single Image or Video.](http://arxiv.org/abs/2211.11743) | 本文提出了一种在单张图像或视频上训练扩散模型的方法，称为SinFusion。该模型可以解决各种图像/视频特定的操作任务，包括从少量帧中学习单个输入视频的运动和动态，生成相同动态场景的多样化新视频样本，将短视频推广为长视频（向前和向后）并执行视频上采样。 |
| [^34] | [LA-VocE: Low-SNR Audio-visual Speech Enhancement using Neural Vocoders.](http://arxiv.org/abs/2211.10999) | LA-VocE是一种新的音频视觉语音增强方法，使用神经声码器将从嘈杂的音频视觉语音预测的mel频谱图转换为波形音频，适用于多种语言和不同水平的背景噪声和语音干扰。 |
| [^35] | [HMOE: Hypernetwork-based Mixture of Experts for Domain Generalization.](http://arxiv.org/abs/2211.08253) | 本文提出了一种新的领域泛化方法HMOE，它不需要领域标签，更具可解释性，使用超网络生成专家权重，能够在低维向量空间中探索专家的相似性，实验结果表明HMOE可以划分混合数据并取得更好的效果。 |
| [^36] | [Efficient brain age prediction from 3D MRI volumes using 2D projections.](http://arxiv.org/abs/2211.05762) | 本文提出了一种使用2D投影从3D MRI体积中高效预测脑龄的方法，相比于使用3D CNN，该方法在计算速度上有两个数量级的提升，对于没有3D CNN昂贵GPU硬件的研究人员非常有用。 |
| [^37] | [A Machine Learning Tutorial for Operational Meteorology, Part II: Neural Networks and Deep Learning.](http://arxiv.org/abs/2211.00147) | 本文讨论了机器学习在气象学中的应用，特别是神经网络和深度学习。涵盖了感知器、人工神经网络、卷积神经网络和U型网络等方法。 |
| [^38] | [ViTASD: Robust Vision Transformer Baselines for Autism Spectrum Disorder Facial Diagnosis.](http://arxiv.org/abs/2210.16943) | 本文提出了一种使用Vision Transformer进行儿童ASD计算分析的方法，该方法从大型面部表情数据集中提取知识，并提供模型结构可转移性。在标准ASD面部分析基准测试上进行的大量实验表明，ViTASD-L实现了新的最先进水平。 |
| [^39] | [Play It Back: Iterative Attention for Audio Recognition.](http://arxiv.org/abs/2210.11328) | 该论文提出了一种基于注意力的架构，通过选择性重复跨越音频序列的最具区分性的声音来进行关注，最终实现了在三个音频分类基准测试中始终实现最先进的性能。 |
| [^40] | [Self-Supervised Geometric Correspondence for Category-Level 6D Object Pose Estimation in the Wild.](http://arxiv.org/abs/2210.07199) | 本文提出了一种自监督学习方法，直接在大规模真实世界物体视频上进行类别级6D姿态估计。通过表面嵌入学习了输入图像和规范形状之间的密集对应关系，并提出了新颖的几何循环一致性损失。学习到的对应关系可以应用于6D姿态估计和其他任务。 |
| [^41] | [PDEBENCH: An Extensive Benchmark for Scientific Machine Learning.](http://arxiv.org/abs/2210.07182) | PDEBench是一个基于偏微分方程的时间依赖性模拟任务基准套件，包括代码和数据，可用于对新型机器学习模型的性能进行基准测试，同时还可以与经典数值模拟和机器学习基线进行比较。 |
| [^42] | [Symmetry Defense Against CNN Adversarial Perturbation Attacks.](http://arxiv.org/abs/2210.04087) | 本文提出了一种对称防御方法，通过翻转或水平翻转对称对抗样本来提高对抗性鲁棒性，同时使用子群对称性进行分类。 |
| [^43] | [Differentiable Parsing and Visual Grounding of Natural Language Instructions for Object Placement.](http://arxiv.org/abs/2210.00215) | ParaGon是一种可微分的自然语言指令解析和视觉定位方法，通过将语言指令解析为以对象为中心的图形表示，以单独定位对象，并使用一种新颖的基于粒子的图神经网络来推理关于带有不确定性的物体放置。 |
| [^44] | [Learning Transferable Spatiotemporal Representations from Natural Script Knowledge.](http://arxiv.org/abs/2209.15280) | 本文提出了一种新的预文本任务，利用自然语言知识来提高可转移的时空表示学习，通过关注学习到的视频表示来对打乱的ASR脚本进行排序，从而提高视频理解的进展。 |
| [^45] | [Recipro-CAM: Fast gradient-free visual explanations for convolutional neural networks.](http://arxiv.org/abs/2209.14074) | Recipro-CAM是一种快速无梯度的可解释性卷积神经网络的视觉解释方法，通过对提取的特征图进行空间掩蔽，利用激活图和目标类别的网络预测之间的相关性，解决了CAM和Grad-CAM方法的架构限制和梯度计算负担问题，具有更短的执行时间，适用于实际解决方案。 |
| [^46] | [EGFR Mutation Prediction of Lung Biopsy Images using Deep Learning.](http://arxiv.org/abs/2208.12506) | 本文使用深度学习技术，通过对肺活检图像进行分析，实现了对EGFR突变的预测，为肺癌治疗提供了更经济、更快捷的诊断方法。 |
| [^47] | [Evaluating Continual Test-Time Adaptation for Contextual and Semantic Domain Shifts.](http://arxiv.org/abs/2208.08767) | 本文评估了针对上下文和语义领域漂移的连续测试时间适应性，无需标签。研究发现，连续测试时间适应性（CoTTA）是一种有效的方法。 |
| [^48] | [Localized Sparse Incomplete Multi-view Clustering.](http://arxiv.org/abs/2208.02998) | 本文提出了一种名为局部稀疏不完整多视图聚类（LSIMVC）的方法，通过优化稀疏正则化和新颖的图嵌入多视图矩阵分解模型，从不完整的多视图数据中学习稀疏和结构化的共识潜在表示，以解决不完整多视图聚类中的问题。 |
| [^49] | [$L_2$BN: Enhancing Batch Normalization by Equalizing the $L_2$ Norms of Features.](http://arxiv.org/abs/2207.02625) | 本文提出了一种$L_2$BN方法，通过等化样本特征的$L_2$范数来增强批量归一化，可以增强内类别特征的紧凑性并扩大跨类别特征的差异，易于实现，可以用作神经网络的基本归一化方法。 |
| [^50] | [Bootstrapping Semi-supervised Medical Image Segmentation with Anatomical-aware Contrastive Distillation.](http://arxiv.org/abs/2206.02307) | 本文提出了一种基于解剖感知对比蒸馏的半监督医学图像分割引导启动方法，通过软标记负样本和捕获更多语义上相似的特征来解决医学图像数据不平衡的问题。 |
| [^51] | [Visuomotor Control in Multi-Object Scenes Using Object-Aware Representations.](http://arxiv.org/abs/2205.06333) | 本文探讨了使用物体感知表示学习技术进行机器人任务的有效性，以解决当前方法学习任务特定表示不能很好地转移到其他任务的问题，以及由监督方法学习的表示需要大量标记数据集的问题。 |
| [^52] | [Representation Learning by Detecting Incorrect Location Embeddings.](http://arxiv.org/abs/2204.04788) | 本文提出了一种新的自监督学习（SSL）损失，用于图像表示学习。通过检测错误的位置嵌入，我们可以提高深度神经网络的泛化能力，使其更加鲁棒。我们称这种方法为DILEMMA，将其应用于MoCoV3、DINO和SimCLR，分别显示它们的性能提高了4.41%、3.97%和0.5%。 |
| [^53] | [Generative Modeling Helps Weak Supervision (and Vice Versa).](http://arxiv.org/abs/2203.12023) | 本文提出了一种融合程序弱监督和生成对抗网络的模型，通过对齐离散潜在变量和弱监督派生的标签估计，改善了未观察到的标签的估计，实现了数据增强。 |
| [^54] | [Towards Targeted Change Detection with Heterogeneous Remote Sensing Images for Forest Mortality Mapping.](http://arxiv.org/abs/2203.00049) | 本文提出了一种基于图像到图像转换和单类分类的新方法，用于检测生态系统某些干扰的弱信号，以绘制由几何蛾爆发引起的森林死亡率在稀疏森林-苔原生态过渡带中的地图。 |
| [^55] | [Practical Network Acceleration with Tiny Sets.](http://arxiv.org/abs/2202.07861) | 本文提出了一种基于删除块的方法，用于加速稀缺训练样本的网络，通过提出的可恢复性概念选择要删除的块，提出了一种名为PRACTISE的算法，仅使用微小的训练图像集来加速网络，PRACTISE的表现优于以往的方法。 |
| [^56] | [I-Tuning: Tuning Frozen Language Models with Image for Lightweight Image Captioning.](http://arxiv.org/abs/2202.06574) | 本文提出了一种轻量级的图像字幕生成框架（I-Tuning），通过设计新颖的交叉注意力模块将不可训练的预训练语言解码器和视觉编码器连接起来，使得模型包含的可训练参数少，训练速度快，同时在三个图像字幕生成基准测试上实现了与大规模基线系统相当或更好的性能，但需要的可训练参数和训练数据量都少得多。 |
| [^57] | [Temporal Sentence Grounding in Videos: A Survey and Future Directions.](http://arxiv.org/abs/2201.08071) | 本文综述了视频中的时间句子定位（TSGV）的基本概念和当前研究现状，以及未来研究方向。TSGV旨在从未经修剪的视频中检索与语言查询语义对应的时间时刻，连接计算机视觉和自然语言，是两个社区研究人员的重点关注点。 |
| [^58] | [Fingerprint Presentation Attack Detection by Channel-wise Feature Denoising.](http://arxiv.org/abs/2111.07620) | 本文提出了一种基于通道特征去噪的指纹呈现攻击检测方法，通过处理先前研究中忽略的冗余噪声信息来学习指纹图像的重要特征，具有较好的鲁棒性和准确性。 |
| [^59] | [A comprehensive review of Binary Neural Network.](http://arxiv.org/abs/2110.06804) | 本文全面综述了二进制神经网络的最新发展，重点关注1位激活和1位卷积网络的权重，这些网络可以在微小的受限设备上实现和嵌入，并节省大量存储、计算成本和能量消耗。 |

# 详细

[^1]: 通过超球统一性差填补神经坍塌的泛化和解耦

    Generalizing and Decoupling Neural Collapse via Hyperspherical Uniformity Gap. (arXiv:2303.06484v1 [cs.LG])

    [http://arxiv.org/abs/2303.06484](http://arxiv.org/abs/2303.06484)

    本文提出了一个广义神经坍塌假设，有效地包含了原始神经坍塌，并将其分解为两个目标：最小化类内变异性和最大化类间可分性。使用超球统一性作为量化这两个目标的统一框架，并提出了一个通用目标——超球统一性差（HUG），它由类间和类内超球统一性之间的差异定义。

    This paper proposes a generalized neural collapse hypothesis that effectively subsumes the original neural collapse and decomposes it into two objectives: minimizing intra-class variability and maximizing inter-class separability. The authors use hyperspherical uniformity as a unified framework to quantify these objectives and propose a general objective, hyperspherical uniformity gap (HUG), which is defined by the difference between inter-class and intra-class hyperspherical uniformity.

    神经坍塌现象描述了深度神经网络的底层几何对称性，其中深度学习的特征和分类器都收敛于一个等角紧框架。已经证明，交叉熵损失和均方误差都可以导致神经坍塌。我们消除了神经坍塌对特征维度和类别数量的关键假设，然后提出了一个广义神经坍塌假设，有效地包含了原始神经坍塌。受神经坍塌描述神经网络训练目标的启发，我们将广义神经坍塌分解为两个目标：最小化类内变异性和最大化类间可分性。然后，我们使用超球统一性（它描述了单位超球上均匀性的程度）作为量化这两个目标的统一框架。最后，我们提出了一个通用目标——超球统一性差（HUG），它由类间和类内超球统一性之间的差异定义。

    The neural collapse (NC) phenomenon describes an underlying geometric symmetry for deep neural networks, where both deeply learned features and classifiers converge to a simplex equiangular tight frame. It has been shown that both cross-entropy loss and mean square error can provably lead to NC. We remove NC's key assumption on the feature dimension and the number of classes, and then present a generalized neural collapse (GNC) hypothesis that effectively subsumes the original NC. Inspired by how NC characterizes the training target of neural networks, we decouple GNC into two objectives: minimal intra-class variability and maximal inter-class separability. We then use hyperspherical uniformity (which characterizes the degree of uniformity on the unit hypersphere) as a unified framework to quantify these two objectives. Finally, we propose a general objective -- hyperspherical uniformity gap (HUG), which is defined by the difference between inter-class and intra-class hyperspherical un
    
[^2]: 高效训练序列的知识蒸馏

    Knowledge Distillation for Efficient Sequences of Training Runs. (arXiv:2303.06480v1 [cs.LG])

    [http://arxiv.org/abs/2303.06480](http://arxiv.org/abs/2303.06480)

    本文研究了如何利用先前运行中的计算来减少未来运行成本的问题，使用知识蒸馏（KD），通过将未来运行与来自先前运行的KD相结合，可以显著减少训练这些模型所需的时间，KD的开销降低了80-90％，对准确性影响很小，并在整体成本方面实现了巨大的帕累托改进。

    This paper studies how to reduce the cost of future runs by utilizing the computation invested in previous runs using knowledge distillation (KD). Augmenting future runs with KD from previous runs dramatically reduces the time necessary to train these models, and the overhead of KD can be reduced by 80-90% with minimal effect on accuracy, resulting in vast pareto-improvements in overall cost.

    在许多实际场景中，如超参数搜索或使用新数据进行持续重新训练，相关的训练运行会按顺序执行多次。目前的做法是从头开始独立训练每个模型。我们研究了利用先前运行中的计算来减少未来运行成本的问题，使用知识蒸馏（KD）。我们发现，将未来运行与来自先前运行的KD相结合，可以显著减少训练这些模型所需的时间，即使考虑到KD的开销。我们通过两种策略改进了这些结果，将KD的开销降低了80-90％，对准确性影响很小，并在整体成本方面实现了巨大的帕累托改进。我们得出结论，KD是减少实践中训练最终模型之前昂贵的准备工作成本的有前途的途径。

    In many practical scenarios -- like hyperparameter search or continual retraining with new data -- related training runs are performed many times in sequence. Current practice is to train each of these models independently from scratch. We study the problem of exploiting the computation invested in previous runs to reduce the cost of future runs using knowledge distillation (KD). We find that augmenting future runs with KD from previous runs dramatically reduces the time necessary to train these models, even taking into account the overhead of KD. We improve on these results with two strategies that reduce the overhead of KD by 80-90% with minimal effect on accuracy and vast pareto-improvements in overall cost. We conclude that KD is a promising avenue for reducing the cost of the expensive preparatory work that precedes training final models in practice.
    
[^3]: ZeroNLG: 将领域对齐和自编码用于零样本多模态和多语言自然语言生成

    ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation. (arXiv:2303.06458v1 [cs.CL])

    [http://arxiv.org/abs/2303.06458](http://arxiv.org/abs/2303.06458)

    ZeroNLG是一个零样本学习框架，可以处理多个NLG任务，包括图像到文本、视频到文本和文本到文本，跨越英语、中文、德语和法语。它不需要任何标记的下游对进行训练，通过将不同的领域投影到共享的公共潜在空间中的相应坐标，桥接不同领域之间的差异。

    ZeroNLG is a zero-shot learning framework that can handle multiple NLG tasks, including image-to-text, video-to-text, and text-to-text, across English, Chinese, German, and French. It does not require any labeled downstream pairs for training, and bridges the differences between different domains by projecting them to corresponding coordinates in a shared common latent space.

    自然语言生成（NLG）接受以图像、视频或文本形式的输入数据，并生成相应的自然语言文本作为输出。现有的NLG方法主要采用监督方法，并且严重依赖于耦合的数据到文本对。然而，对于许多有针对性的场景和非英语语言，往往没有足够数量的标记数据。为了放松对下游任务标记数据的依赖性，我们提出了一个直观有效的零样本学习框架ZeroNLG，它可以处理多个NLG任务，包括图像到文本（图像字幕）、视频到文本（视频字幕）和文本到文本（神经机器翻译），跨越英语、中文、德语和法语在一个统一的框架内。ZeroNLG不需要任何标记的下游对进行训练。在训练期间，ZeroNLG（i）将不同的领域（跨模态和语言）投影到共享的公共潜在空间中的相应坐标；（ii）桥接差异

    Natural Language Generation (NLG) accepts input data in the form of images, videos, or text and generates corresponding natural language text as output. Existing NLG methods mainly adopt a supervised approach and rely heavily on coupled data-to-text pairs. However, for many targeted scenarios and for non-English languages, sufficient quantities of labeled data are often not available. To relax the dependency on labeled data of downstream tasks, we propose an intuitive and effective zero-shot learning framework, ZeroNLG, which can deal with multiple NLG tasks, including image-to-text (image captioning), video-to-text (video captioning), and text-to-text (neural machine translation), across English, Chinese, German, and French within a unified framework. ZeroNLG does not require any labeled downstream pairs for training. During training, ZeroNLG (i) projects different domains (across modalities and languages) to corresponding coordinates in a shared common latent space; (ii) bridges diff
    
[^4]: DECOMPL: 一种基于注意力池化的分解学习技术，用于从单个排球图像中识别团体活动

    DECOMPL: Decompositional Learning with Attention Pooling for Group Activity Recognition from a Single Volleyball Image. (arXiv:2303.06439v1 [cs.CV])

    [http://arxiv.org/abs/2303.06439](http://arxiv.org/abs/2303.06439)

    本文提出了一种新的排球视频团体活动识别技术DECOMPL，它由两个互补的分支组成，使用选择性的注意力池化提取特征，考虑参与者的当前配置，并从框坐标中提取空间信息。同时，本文发现排球数据集的标签方案降低了活动中的团体概念。

    This paper proposes a novel GAR technique for volleyball videos, DECOMPL, which consists of two complementary branches, using selective attention pooling to extract features, considering the current configuration of actors and extracting spatial information from box coordinates. The paper also reveals that the labeling scheme of the Volleyball dataset degrades the group concept in activities.

    团体活动识别旨在检测场景中多个参与者执行的活动。先前的工作基于RGB、光流或关键点数据类型对时空特征进行建模。然而，同时使用时间性和这些数据类型会显著增加计算复杂度。我们的假设是，仅使用RGB数据而不考虑时间性，可以在几乎不损失准确性的情况下保持性能。为此，我们提出了一种新的排球视频团体活动识别技术DECOMPL，它由两个互补的分支组成。在视觉分支中，它使用选择性的注意力池化提取特征。在坐标分支中，它考虑参与者的当前配置，并从框坐标中提取空间信息。此外，我们分析了排球数据集，发现其标签方案降低了活动中的团体概念。

    Group Activity Recognition (GAR) aims to detect the activity performed by multiple actors in a scene. Prior works model the spatio-temporal features based on the RGB, optical flow or keypoint data types. However, using both the temporality and these data types altogether increase the computational complexity significantly. Our hypothesis is that by only using the RGB data without temporality, the performance can be maintained with a negligible loss in accuracy. To that end, we propose a novel GAR technique for volleyball videos, DECOMPL, which consists of two complementary branches. In the visual branch, it extracts the features using attention pooling in a selective way. In the coordinate branch, it considers the current configuration of the actors and extracts the spatial information from the box coordinates. Moreover, we analyzed the Volleyball dataset that the recent literature is mostly based on, and realized that its labeling scheme degrades the group concept in the activities to
    
[^5]: 利用人工神经网络探究场景感知的神经表征在海马依赖任务中的应用

    Probing neural representations of scene perception in a hippocampally dependent task using artificial neural networks. (arXiv:2303.06367v1 [cs.CV])

    [http://arxiv.org/abs/2303.06367](http://arxiv.org/abs/2303.06367)

    本研究利用人工神经网络探究场景感知的神经表征在海马依赖任务中的应用，设计了一个新型场景感知基准，证明了DNNs可以学习从不同自我中心视角观察的场景的能力。

    This study uses artificial neural networks to explore neural representations of scene perception in a hippocampally dependent task, and demonstrates that DNNs can learn the ability to transform scenes viewed from different egocentric perspectives, using a novel scene perception benchmark.

    通过反向传播训练的深度人工神经网络(DNNs)提供了哺乳动物视觉系统的有效模型，准确地捕捉了从初级视觉皮层到下颞皮质(IT)的神经响应层次结构。然而，这些网络解释更高皮层区域的表征能力相对较弱，研究也相对较少。我们描述了一个受海马依赖任务启发的新型场景感知基准，旨在探究DNNs转换从不同自我中心视角观察的场景的能力。使用受颞叶结构和海马之间连接启发的网络架构，我们证明了使用三元组损失训练的DNNs可以学习这个任务。此外，通过强制执行分解的潜在空间

    Deep artificial neural networks (DNNs) trained through backpropagation provide effective models of the mammalian visual system, accurately capturing the hierarchy of neural responses through primary visual cortex to inferior temporal cortex (IT). However, the ability of these networks to explain representations in higher cortical areas is relatively lacking and considerably less well researched. For example, DNNs have been less successful as a model of the egocentric to allocentric transformation embodied by circuits in retrosplenial and posterior parietal cortex. We describe a novel scene perception benchmark inspired by a hippocampal dependent task, designed to probe the ability of DNNs to transform scenes viewed from different egocentric perspectives. Using a network architecture inspired by the connectivity between temporal lobe structures and the hippocampus, we demonstrate that DNNs trained using a triplet loss can learn this task. Moreover, by enforcing a factorized latent space
    
[^6]: 通过虚拟检查层实现时间序列的可解释性人工智能

    Explainable AI for Time Series via Virtual Inspection Layers. (arXiv:2303.06365v1 [cs.LG])

    [http://arxiv.org/abs/2303.06365](http://arxiv.org/abs/2303.06365)

    本文提出了一种虚拟检查层，将时间序列转换为可解释的表示，并允许通过本地XAI方法将相关性归因传播到该表示。我们将一系列XAI方法的适用性扩展到需要转换后才能解释输入的领域。我们展示了DFT-LRP在各种时间序列分类设置中的有用性，如音频和电子健康记录。

    This paper proposes a virtual inspection layer that transforms time series into an interpretable representation and allows for relevance attributions to be propagated to this representation via local XAI methods. The applicability of a family of XAI methods is extended to domains where the input is only interpretable after a transformation. The usefulness of DFT-LRP is demonstrated in various time series classification settings, such as audio and electronic health records.

    最近几年，可解释人工智能（XAI）领域取得了很大进展，但主要是在计算机视觉和自然语言处理方面。对于时间序列，由于输入通常不可解释，因此只有有限的XAI研究可用。在这项工作中，我们提出了一种虚拟检查层，将时间序列转换为可解释的表示，并允许通过本地XAI方法（如逐层相关传播（LRP））将相关性归因传播到该表示。通过这种方式，我们将一系列XAI方法的适用性扩展到需要转换后才能解释输入的领域（例如语音）。在这里，我们专注于傅里叶变换，这在时间序列解释和LRP中被广泛应用，并将我们的方法称为DFT-LRP。我们展示了DFT-LRP在各种时间序列分类设置中的有用性，如音频和电子健康记录。我们展示了如何使用DFT-LRP来可视化和解释模型的决策。

    The field of eXplainable Artificial Intelligence (XAI) has greatly advanced in recent years, but progress has mainly been made in computer vision and natural language processing. For time series, where the input is often not interpretable, only limited research on XAI is available. In this work, we put forward a virtual inspection layer, that transforms the time series to an interpretable representation and allows to propagate relevance attributions to this representation via local XAI methods like layer-wise relevance propagation (LRP). In this way, we extend the applicability of a family of XAI methods to domains (e.g. speech) where the input is only interpretable after a transformation. Here, we focus on the Fourier transformation which is prominently applied in the interpretation of time series and LRP and refer to our method as DFT-LRP. We demonstrate the usefulness of DFT-LRP in various time series classification settings like audio and electronic health records. We showcase how 
    
[^7]: MetaViewer: 朝着统一的多视图表示迈进

    MetaViewer: Towards A Unified Multi-View Representation. (arXiv:2303.06329v1 [cs.CV])

    [http://arxiv.org/abs/2303.06329](http://arxiv.org/abs/2303.06329)

    该论文提出了一种新颖的基于双层优化的多视图学习框架MetaViewer，通过统一到特定的方式学习表示，避免了手动预先指定的融合函数和混合在特征中的视图专用冗余信息可能会降低所得表示的质量的问题。

    This paper proposes a novel bi-level-optimization-based multi-view learning framework, MetaViewer, which learns the representation in a uniform-to-specific manner, avoiding the problem of manually pre-specify fusion functions and view-private redundant information mixed in features that potentially degrade the quality of the derived representation.

    现有的多视图表示学习方法通常遵循特定到统一的流程，从每个视图中提取潜在特征，然后融合或对齐它们以获得统一的对象表示。然而，手动预先指定的融合函数和混合在特征中的视图专用冗余信息可能会降低所得表示的质量。为了克服这些问题，我们提出了一种新颖的基于双层优化的多视图学习框架，其中表示是以统一到特定的方式学习的。具体而言，我们训练一个元学习器，即MetaViewer，在外层优化中学习融合和建模视图共享的元表示。从这个元表示开始，需要在内层训练视图特定的基学习器，以快速重构相应的视图。MetaViewer最终通过观察所有视图上从统一到特定的重构过程来更新，并学习最佳融合方案。

    Existing multi-view representation learning methods typically follow a specific-to-uniform pipeline, extracting latent features from each view and then fusing or aligning them to obtain the unified object representation. However, the manually pre-specify fusion functions and view-private redundant information mixed in features potentially degrade the quality of the derived representation. To overcome them, we propose a novel bi-level-optimization-based multi-view learning framework, where the representation is learned in a uniform-to-specific manner. Specifically, we train a meta-learner, namely MetaViewer, to learn fusion and model the view-shared meta representation in outer-level optimization. Start with this meta representation, view-specific base-learners are then required to rapidly reconstruct the corresponding view in inner-level. MetaViewer eventually updates by observing reconstruction processes from uniform to specific over all views, and learns an optimal fusion scheme that
    
[^8]: MLP-SRGAN: 使用MLP-Mixer的单维超分辨率GAN

    MLP-SRGAN: A Single-Dimension Super Resolution GAN using MLP-Mixer. (arXiv:2303.06298v1 [cs.CV])

    [http://arxiv.org/abs/2303.06298](http://arxiv.org/abs/2303.06298)

    MLP-SRGAN是一种单维超分辨率GAN，使用MLP-Mixer和卷积层进行上采样，可用于FLAIR MRI图像的超分辨率重建，提出了新的图像质量度量方法。

    MLP-SRGAN is a single-dimension Super Resolution GAN that utilizes MLP-Mixers and convolutional layers for upsampling, and can be used for super-resolution reconstruction of FLAIR MRI images. New image quality metrics were proposed.

    我们提出了一种新的架构，称为MLP-SRGAN，它是一种单维超分辨率生成对抗网络（SRGAN），利用多层感知器混合器（MLP-Mixer）以及卷积层在切片方向上进行上采样。 MLP-SRGAN使用MSSEG2挑战数据集中的高分辨率（HR）FLAIR MRI进行训练和验证。该方法应用于三个低空间分辨率的多中心FLAIR数据集（CAIN，ADNI，CCNA）的图像，以检查在保留（未见）临床数据上的性能。将上采样结果与几种最先进的SR网络进行比较。对于具有高分辨率（HR）基本事实的图像，使用峰值信噪比（PSNR）和结构相似性指数（SSIM）来衡量上采样性能。提出了几种新的结构，无参考图像质量度量，以在缺乏基础事实的情况下量化锐度（边缘强度），噪声（熵）和模糊度（低频信息）。

    We propose a novel architecture called MLP-SRGAN, which is a single-dimension Super Resolution Generative Adversarial Network (SRGAN) that utilizes Multi-Layer Perceptron Mixers (MLP-Mixers) along with convolutional layers to upsample in the slice direction. MLP-SRGAN is trained and validated using high resolution (HR) FLAIR MRI from the MSSEG2 challenge dataset. The method was applied to three multicentre FLAIR datasets (CAIN, ADNI, CCNA) of images with low spatial resolution in the slice dimension to examine performance on held-out (unseen) clinical data. Upsampled results are compared to several state-of-the-art SR networks. For images with high resolution (HR) ground truths, peak-signal-to-noise-ratio (PSNR) and structural similarity index (SSIM) are used to measure upsampling performance. Several new structural, no-reference image quality metrics were proposed to quantify sharpness (edge strength), noise (entropy), and blurriness (low frequency information) in the absence of groun
    
[^9]: 防止注意力熵崩溃的Transformer训练稳定性研究

    Stabilizing Transformer Training by Preventing Attention Entropy Collapse. (arXiv:2303.06296v1 [cs.LG])

    [http://arxiv.org/abs/2303.06296](http://arxiv.org/abs/2303.06296)

    本文研究了Transformer的训练动态，发现低注意力熵伴随着高训练不稳定性，提出了一种简单而有效的解决方案$\sigma$Reparam，成功地防止了注意力层中的熵崩溃，促进了更稳定的训练。

    This paper investigates the training dynamics of Transformers and proposes a simple and efficient solution, $\sigma$Reparam, to prevent entropy collapse in the attention layers, promoting more stable training.

    训练稳定性对于Transformer至关重要。本文通过研究注意力层的演变来探究Transformer的训练动态。特别地，我们在训练过程中跟踪每个注意力头的注意力熵，这是模型锐度的代理。我们发现，在不同的架构和任务中存在一种常见模式，即低注意力熵伴随着高训练不稳定性，这可能采取振荡损失或发散的形式。我们将病态低注意力熵，对应高度集中的注意力分数，称为$\textit{熵崩溃}$。作为一种解决方案，我们提出了$\sigma$Reparam，一种简单而有效的解决方案，其中我们使用谱归一化和额外的学习标量重新参数化所有线性层。我们证明了所提出的重新参数化成功地防止了注意力层中的熵崩溃，促进了更稳定的训练。此外，我们

    Training stability is of great importance to Transformers. In this work, we investigate the training dynamics of Transformers by examining the evolution of the attention layers. In particular, we track the attention entropy for each attention head during the course of training, which is a proxy for model sharpness. We identify a common pattern across different architectures and tasks, where low attention entropy is accompanied by high training instability, which can take the form of oscillating loss or divergence. We denote the pathologically low attention entropy, corresponding to highly concentrated attention scores, as $\textit{entropy collapse}$. As a remedy, we propose $\sigma$Reparam, a simple and efficient solution where we reparametrize all linear layers with spectral normalization and an additional learned scalar. We demonstrate that the proposed reparameterization successfully prevents entropy collapse in the attention layers, promoting more stable training. Additionally, we 
    
[^10]: CoNIC挑战：推动核检测、分割、分类和计数的前沿（arXiv:2303.06274v1 [cs.CV]）

    CoNIC Challenge: Pushing the Frontiers of Nuclear Detection, Segmentation, Classification and Counting. (arXiv:2303.06274v1 [cs.CV])

    [http://arxiv.org/abs/2303.06274](http://arxiv.org/abs/2303.06274)

    CoNIC挑战使用最大的数据集评估核分割和细胞组成，刺激了可重复的细胞识别算法的开发，发现嗜酸性粒细胞和中性粒细胞在肿瘤中发挥重要作用。

    The CoNIC challenge used the largest dataset to evaluate nuclear segmentation and cellular composition, stimulated the development of reproducible algorithms for cellular recognition, and found that eosinophils and neutrophils play an important role in tumors.

    核检测、分割和形态测量是帮助我们进一步了解组织学和患者预后关系的关键。为了推动这一领域的创新，我们使用目前最大的数据集设置了一个社区广泛的挑战，以评估核分割和细胞组成。我们的挑战名为CoNIC，刺激了可重复的细胞识别算法的开发，并在公共排行榜上进行实时结果检查。我们基于1,658个结肠组织的全切片图像对表现最佳的模型进行了广泛的后挑战分析。每个模型检测到约7亿个细胞核，相关特征用于不良增生分级和生存分析，我们证明了挑战对先前最先进技术的改进导致了下游性能的显著提升。我们的发现还表明，嗜酸性粒细胞和中性粒细胞在肿瘤中发挥重要作用。

    Nuclear detection, segmentation and morphometric profiling are essential in helping us further understand the relationship between histology and patient outcome. To drive innovation in this area, we setup a community-wide challenge using the largest available dataset of its kind to assess nuclear segmentation and cellular composition. Our challenge, named CoNIC, stimulated the development of reproducible algorithms for cellular recognition with real-time result inspection on public leaderboards. We conducted an extensive post-challenge analysis based on the top-performing models using 1,658 whole-slide images of colon tissue. With around 700 million detected nuclei per model, associated features were used for dysplasia grading and survival analysis, where we demonstrated that the challenge's improvement over the previous state-of-the-art led to significant boosts in downstream performance. Our findings also suggest that eosinophils and neutrophils play an important role in the tumour m
    
[^11]: 基于双视角的超似曲自适应学习用于自监督骨架动作表示

    HYperbolic Self-Paced Learning for Self-Supervised Skeleton-based Action Representations. (arXiv:2303.06242v1 [cs.CV])

    [http://arxiv.org/abs/2303.06242](http://arxiv.org/abs/2303.06242)

    本文提出了一种新的超似曲自适应模型（HYSP）用于学习基于骨架的动作表示，采用自监督学习，使用数据增强来生成同一样本的两个视图，并通过将一个视图与另一个视图匹配来学习，使用超似曲不确定性来确定算法学习速度，假设不确定性较小的样本应更强烈地推动训练，具有更大的权重和速度。

    This paper proposes a novel HYperbolic Self-Paced model (HYSP) for learning skeleton-based action representations, which adopts self-supervision and uses data augmentations to generate two views of the same sample, and learns by matching one to the other. It uses hyperbolic uncertainty to determine the algorithmic learning pace, assuming that less uncertain samples should be more strongly driving the training, with a larger weight and pace.

    自适应学习在一些任务中有益，例如弱监督学习和领域自适应，可以选择和排序训练样本序列，从易到难。然而，它在无监督学习中的适用性仍未被探索，其中任务的知识在训练期间成熟。我们提出了一种新的超似曲自适应模型（HYSP）用于学习基于骨架的动作表示。HYSP采用自监督：它使用数据增强来生成同一样本的两个视图，并通过将一个视图（称为在线）与另一个视图（目标）匹配来学习。我们建议使用超似曲不确定性来确定算法学习速度，假设不确定性较小的样本应更强烈地推动训练，具有更大的权重和速度。超似曲不确定性是采用的超似曲神经网络的副产品，它在训练期间成熟，与额外成本相比，没有额外成本。

    Self-paced learning has been beneficial for tasks where some initial knowledge is available, such as weakly supervised learning and domain adaptation, to select and order the training sample sequence, from easy to complex. However its applicability remains unexplored in unsupervised learning, whereby the knowledge of the task matures during training. We propose a novel HYperbolic Self-Paced model (HYSP) for learning skeleton-based action representations. HYSP adopts self-supervision: it uses data augmentations to generate two views of the same sample, and it learns by matching one (named online) to the other (the target). We propose to use hyperbolic uncertainty to determine the algorithmic learning pace, under the assumption that less uncertain samples should be more strongly driving the training, with a larger weight and pace. Hyperbolic uncertainty is a by-product of the adopted hyperbolic neural networks, it matures during training and it comes with no extra cost, compared to the e
    
[^12]: 对抗训练是否需要使用整个训练数据集？

    Do we need entire training data for adversarial training?. (arXiv:2303.06241v1 [cs.CV])

    [http://arxiv.org/abs/2303.06241](http://arxiv.org/abs/2303.06241)

    本文提出了一种新的对抗训练方法，通过对训练数据集进行筛选，仅使用易受对抗攻击的样本进行训练，从而减少训练时间。

    This paper proposes a new adversarial training method that reduces training time by selecting only the adversarially-prone samples from the training dataset.

    深度神经网络（DNN）被用于解决许多领域的问题，包括自动驾驶汽车和医学图像等安全关键领域。DNN对抗攻击的脆弱性已经被广泛关注。近年来，已经提出了许多方法来通过对抗训练来解决这个问题。几乎所有的方法都会为整个训练数据集生成对抗性示例，从而大大增加了训练时间。我们展示了通过仅使用训练数据的子集进行对抗训练，可以减少任何对抗训练算法的训练时间。为了选择子集，我们从训练数据中过滤出易受对抗攻击的样本。我们对所有训练样本执行简单的对抗攻击，以过滤出这个子集。在这个攻击中，我们向每个像素添加一个小扰动和几条网格线到输入图像中。我们对易受对抗攻击的子集进行对抗训练，并且...

    Deep Neural Networks (DNNs) are being used to solve a wide range of problems in many domains including safety-critical domains like self-driving cars and medical imagery. DNNs suffer from vulnerability against adversarial attacks. In the past few years, numerous approaches have been proposed to tackle this problem by training networks using adversarial training. Almost all the approaches generate adversarial examples for the entire training dataset, thus increasing the training time drastically. We show that we can decrease the training time for any adversarial training algorithm by using only a subset of training data for adversarial training. To select the subset, we filter the adversarially-prone samples from the training data. We perform a simple adversarial attack on all training examples to filter this subset. In this attack, we add a small perturbation to each pixel and a few grid lines to the input image.  We perform adversarial training on the adversarially-prone subset and mi
    
[^13]: 带张量自编码器的压缩感知

    Compressive Sensing with Tensorized Autoencoder. (arXiv:2303.06235v1 [cs.CV])

    [http://arxiv.org/abs/2303.06235](http://arxiv.org/abs/2303.06235)

    本文提出了一种利用张量环因子分解的自动编码器来恢复图像的方法，该方法在修复和去噪应用中表现出更好的重建质量。

    This paper proposes a method for image recovery using an autoencoder with tensor ring factorization, which achieves better reconstruction quality in inpainting and denoising applications.

    深度网络可以训练成将图像映射到低维潜在空间的工具。在许多情况下，集合中的不同图像是彼此的关节版本；例如，同一物体具有不同的照明、背景或姿势。此外，在许多情况下，图像的某些部分可能会受到噪声或缺失条目的影响。在本文中，我们的目标是利用数据的结构先验来恢复图像，而没有访问地面真实（干净）图像。这样的恢复问题属于压缩感知领域。我们建议在嵌入空间上学习带有张量环因子分解的自动编码器，以对数据施加结构约束。特别地，我们在自动编码器的瓶颈层中使用张量环结构，利用结构化数据集的软标签。我们通过实验证明了所提出的方法在修复和去噪应用中的有效性。所得到的方法实现了更好的重建质量。

    Deep networks can be trained to map images into a low-dimensional latent space. In many cases, different images in a collection are articulated versions of one another; for example, same object with different lighting, background, or pose. Furthermore, in many cases, parts of images can be corrupted by noise or missing entries. In this paper, our goal is to recover images without access to the ground-truth (clean) images using the articulations as structural prior of the data. Such recovery problems fall under the domain of compressive sensing. We propose to learn autoencoder with tensor ring factorization on the the embedding space to impose structural constraints on the data. In particular, we use a tensor ring structure in the bottleneck layer of the autoencoder that utilizes the soft labels of the structured dataset. We empirically demonstrate the effectiveness of the proposed approach for inpainting and denoising applications. The resulting method achieves better reconstruction qu
    
[^14]: MCROOD: 多类雷达超出分布检测

    MCROOD: Multi-Class Radar Out-Of-Distribution Detection. (arXiv:2303.06232v1 [cs.CV])

    [http://arxiv.org/abs/2303.06232](http://arxiv.org/abs/2303.06232)

    本文提出了一种基于重建的多类OOD检测器，该检测器在雷达距离多普勒图像（RDIs）上运行。检测器旨在将除坐、站或走的人以外的任何移动物体分类为OOD。作者还提供了一种简单而有效的预处理技术，以检测呼吸等微小的人体运动。在实验中，该方法表现优于最先进的OOD检测方法。

    This paper proposes a reconstruction-based multi-class OOD detector that operates on radar range doppler images (RDIs). The detector aims to classify any moving object other than a person sitting, standing, or walking as OOD. The authors also provide a simple yet effective pre-processing technique to detect minor human body movements like breathing. The method outperforms state-of-the-art OOD detection methods in experiments.

    最近，由于其在安全部署现代深度学习（DL）架构中的关键作用，超出分布（OOD）检测受到特别关注。本文提出了一种基于重建的多类OOD检测器，该检测器在雷达距离多普勒图像（RDIs）上运行。检测器旨在将除坐、站或走的人以外的任何移动物体分类为OOD。我们还提供了一种简单而有效的预处理技术，以检测呼吸等微小的人体运动。这个简单的想法被称为呼吸检测器（RESPD），可以减轻OOD检测的负担，特别是对于人坐和人站的类别。在我们收集的60GHz短距离FMCW雷达数据集上，我们分别为坐、站和走三个类别实现了97.45％、92.13％和96.58％的AUROC。我们进行了大量实验，并表明我们的方法优于最先进的OOD检测方法。此外，我们的流程比第二好的方法快24倍，并且是v

    Out-of-distribution (OOD) detection has recently received special attention due to its critical role in safely deploying modern deep learning (DL) architectures. This work proposes a reconstruction-based multi-class OOD detector that operates on radar range doppler images (RDIs). The detector aims to classify any moving object other than a person sitting, standing, or walking as OOD. We also provide a simple yet effective pre-processing technique to detect minor human body movements like breathing. The simple idea is called respiration detector (RESPD) and eases the OOD detection, especially for human sitting and standing classes. On our dataset collected by 60GHz short-range FMCW Radar, we achieve AUROCs of 97.45%, 92.13%, and 96.58% for sitting, standing, and walking classes, respectively. We perform extensive experiments and show that our method outperforms state-of-the-art (SOTA) OOD detection methods. Also, our pipeline performs 24 times faster than the second-best method and is v
    
[^15]: 基于POV的高速公路车辆轨迹数据集和预测架构

    A POV-based Highway Vehicle Trajectory Dataset and Prediction Architecture. (arXiv:2303.06202v1 [cs.CV])

    [http://arxiv.org/abs/2303.06202](http://arxiv.org/abs/2303.06202)

    介绍了一个基于POV的高速公路车辆轨迹数据集和预测架构，其中的Carolinas Highway Dataset（CHD）包括160万帧和338,000个车辆轨迹。

    Introducing a POV-based highway vehicle trajectory dataset and prediction architecture, including Carolinas Highway Dataset (CHD) with 1.6 million frames and 338,000 vehicle trajectories captured at eight locations in Carolinas.

    提供多个视角（POV）的车辆轨迹数据集对于各种交通安全和管理应用非常有价值。尽管轨迹数据集很丰富，但很少提供全面和多样化的驾驶场景，捕捉各种高速公路布局、合并车道和配置的多个视点。这限制了它们捕捉驾驶员、车辆和道路基础设施之间微妙互动的能力。我们介绍了Carolinas Highway Dataset（CHD），这是一个车辆轨迹、检测和跟踪数据集。CHD是在Carolinas的八个位置拍摄的高速公路视频中捕获的160万帧，包括338,000个车辆轨迹。

    Vehicle Trajectory datasets that provide multiple point-of-views (POVs) can be valuable for various traffic safety and management applications. Despite the abundance of trajectory datasets, few offer a comprehensive and diverse range of driving scenes, capturing multiple viewpoints of various highway layouts, merging lanes, and configurations. This limits their ability to capture the nuanced interactions between drivers, vehicles, and the roadway infrastructure. We introduce the \emph{Carolinas Highway Dataset (CHD\footnote{\emph{CHD} available at: \url{https://github.com/TeCSAR-UNCC/Carolinas\_Dataset}})}, a vehicle trajectory, detection, and tracking dataset. \emph{CHD} is a collection of 1.6 million frames captured in highway-based videos from eye-level and high-angle POVs at eight locations across Carolinas with 338,000 vehicle trajectories. The locations, timing of recordings, and camera angles were carefully selected to capture various road geometries, traffic patterns, lighting 
    
[^16]: 针对分布式非独立同分布数据和部分标签的医学图像分类，优化联邦学习

    Optimizing Federated Learning for Medical Image Classification on Distributed Non-iid Datasets with Partial Labels. (arXiv:2303.06180v1 [cs.LG])

    [http://arxiv.org/abs/2303.06180](http://arxiv.org/abs/2303.06180)

    本文提出了FedFBN，一个联邦学习框架，使用预训练的网络作为模型后端，并在整个训练过程中冻结批量归一化层，以优化分布式非独立同分布数据和部分标签的医学图像分类。

    This paper proposes FedFBN, a federated learning framework that uses pretrained networks as the model backend and freezes the batch normalization layers throughout the training process to optimize medical image classification on distributed non-iid datasets with partial labels.

    大量的胸部X光数据集已经带头使用深度学习进行异常检测。然而，这些数据集专注于检测可能存在的一部分疾病标签，因此使它们成为分布式和非独立同分布的部分标签数据集。最近的文献指出，批量归一化层对于联邦学习的收敛具有影响，因为它们与具有部分标签的非独立同分布数据相关的域漂移。为此，我们提出了FedFBN，这是一个联邦学习框架，它从迁移学习中汲取灵感，使用预训练的网络作为模型后端，并在整个训练过程中冻结批量归一化层。我们使用合成iid玩具数据集和大规模非iid数据集评估FedFBN与当前FL策略。我们的结果表明，FedFBN优于使用分布式和非独立同分布数据训练全局模型的当前聚合策略。

    Numerous large-scale chest x-ray datasets have spearheaded expert-level detection of abnormalities using deep learning. However, these datasets focus on detecting a subset of disease labels that could be present, thus making them distributed and non-iid with partial labels. Recent literature has indicated the impact of batch normalization layers on the convergence of federated learning due to domain shift associated with non-iid data with partial labels. To that end, we propose FedFBN, a federated learning framework that draws inspiration from transfer learning by using pretrained networks as the model backend and freezing the batch normalization layers throughout the training process. We evaluate FedFBN with current FL strategies using synthetic iid toy datasets and large-scale non-iid datasets across scenarios with partial and complete labels. Our results demonstrate that FedFBN outperforms current aggregation strategies for training global models using distributed and non-iid data w
    
[^17]: 通过操作微调数据集来克服预训练模型中的偏见

    Overcoming Bias in Pretrained Models by Manipulating the Finetuning Dataset. (arXiv:2303.06167v1 [cs.CV])

    [http://arxiv.org/abs/2303.06167](http://arxiv.org/abs/2303.06167)

    本文研究了预训练模型中的偏见问题，发现微调模型可以继承预训练模型的偏见，但通过对微调数据集进行干预可以纠正这种偏见，而且对性能的影响很小。这表明仔细策划微调数据集对于减少下游任务中的偏见非常重要，这样做甚至可以弥补预训练模型中的偏见。

    This paper investigates the bias problem in pretrained models and finds that finetuned models can inherit the biases of pretrained models, but these biases can be corrected by manipulating the finetuning dataset with little impact on performance. This implies that careful curation of the finetuning dataset is important for reducing biases on a downstream task, and doing so can even compensate for bias in the pretrained model.

    转移学习通过允许在大规模数据集上预训练的模型的表达特征被微调到更小、更具领域特定性的数据集的目标任务中而受益。然而，有人担心这些预训练模型可能带有自己的偏见，这些偏见会传播到微调模型中。在这项工作中，我们研究了偏见，当偏见被概念化为目标任务和敏感属性之间的虚假相关性以及数据集中特定群体的代表性不足时。在偏见的两种概念下，我们发现(1)在预训练模型的基础上微调的模型确实可以继承它们的偏见，但(2)通过对微调数据集进行相对较小的干预，这种偏见可以得到纠正，而且对性能的影响往往可以忽略不计。我们的发现意味着，仔细策划微调数据集对于减少下游任务中的偏见非常重要，这样做甚至可以弥补预训练模型中的偏见。

    Transfer learning is beneficial by allowing the expressive features of models pretrained on large-scale datasets to be finetuned for the target task of smaller, more domain-specific datasets. However, there is a concern that these pretrained models may come with their own biases which would propagate into the finetuned model. In this work, we investigate bias when conceptualized as both spurious correlations between the target task and a sensitive attribute as well as underrepresentation of a particular group in the dataset. Under both notions of bias, we find that (1) models finetuned on top of pretrained models can indeed inherit their biases, but (2) this bias can be corrected for through relatively minor interventions to the finetuning dataset, and often with a negligible impact to performance. Our findings imply that careful curation of the finetuning dataset is important for reducing biases on a downstream task, and doing so can even compensate for bias in the pretrained model.
    
[^18]: 学习视觉文本扰动的易读性

    Learning the Legibility of Visual Text Perturbations. (arXiv:2303.05077v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.05077](http://arxiv.org/abs/2303.05077)

    本文提出了一种学习模型来预测扰动字符串的易读性，并根据其易读性对候选扰动进行排名的方法，填补了保持易读性的文本扰动的系统性表征的空白。

    This paper proposes a method to learn models that predict the legibility of a perturbed string and rank candidate perturbations based on their legibility, filling the gap in systematically characterizing the legibility of text perturbations while preserving it.

    许多NLP中的对抗性攻击会扰动输入以产生在视觉上相似但对模型性能有负面影响的字符串（例如'ergo' $\rightarrow$ '$\epsilon$rgo'），这些字符串对人类来说是易读的。尽管保持易读性是文本扰动的必要条件，但很少有工作对其进行系统性的表征；相反，易读性通常通过对扰动的性质和程度的直觉约束来实现。特别地，尚不清楚在保持易读性的情况下可以扰动多少输入，或如何量化扰动字符串的易读性。在这项工作中，我们通过学习模型来预测扰动字符串的易读性，并根据其易读性对候选扰动进行排名，以填补这一空白。为此，我们收集并发布了LEGIT，一个人类注释的数据集，其中包括视觉上扰动文本的易读性。使用这个数据集，我们构建了基于文本和视觉的模型，可以达到$0.91$的F1分数，以预测输入是否易读。

    Many adversarial attacks in NLP perturb inputs to produce visually similar strings ('ergo' $\rightarrow$ '$\epsilon$rgo') which are legible to humans but degrade model performance. Although preserving legibility is a necessary condition for text perturbation, little work has been done to systematically characterize it; instead, legibility is typically loosely enforced via intuitions around the nature and extent of perturbations. Particularly, it is unclear to what extent can inputs be perturbed while preserving legibility, or how to quantify the legibility of a perturbed string. In this work, we address this gap by learning models that predict the legibility of a perturbed string, and rank candidate perturbations based on their legibility. To do so, we collect and release LEGIT, a human-annotated dataset comprising the legibility of visually perturbed text. Using this dataset, we build both text- and vision-based models which achieve up to $0.91$ F1 score in predicting whether an input
    
[^19]: Prismer: 一种具有专家集合的视觉语言模型

    Prismer: A Vision-Language Model with An Ensemble of Experts. (arXiv:2303.02506v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.02506](http://arxiv.org/abs/2303.02506)

    Prismer是一种数据和参数高效的视觉语言模型，它利用了一组领域专家的集合，通过汇集这些专家知识并将其适应于各种视觉语言推理任务，实现了与当前最先进模型竞争的微调和少样本学习性能，同时需要少至两个数量级的训练数据。

    Prismer is a data- and parameter-efficient vision-language model that leverages an ensemble of domain experts, achieving fine-tuned and few-shot learning performance competitive with current state-of-the-art models, whilst requiring up to two orders of magnitude less training data.

    最近的视觉语言模型展示了令人印象深刻的多模态生成能力。然而，通常它们需要在大规模数据集上训练庞大的模型。作为一种更可扩展的替代方案，我们介绍了Prismer，一种数据和参数高效的视觉语言模型，它利用了一组领域专家的集合。Prismer只需要训练少量组件，大部分网络权重从现成的预训练领域专家中继承，并在训练期间保持冻结状态。通过利用来自各种领域的专家，我们展示了Prismer可以有效地汇集这些专家知识并将其适应于各种视觉语言推理任务。在我们的实验中，我们展示了Prismer实现了与当前最先进模型竞争的微调和少样本学习性能，同时需要少至两个数量级的训练数据。代码可在https://github.com/NVlabs/prismer获得。

    Recent vision-language models have shown impressive multi-modal generation capabilities. However, typically they require training huge models on massive datasets. As a more scalable alternative, we introduce Prismer, a data- and parameter-efficient vision-language model that leverages an ensemble of domain experts. Prismer only requires training of a small number of components, with the majority of network weights inherited from readily-available, pre-trained domain experts, and kept frozen during training. By leveraging experts from a wide range of domains, we show that Prismer can efficiently pool this expert knowledge and adapt it to various vision-language reasoning tasks. In our experiments, we show that Prismer achieves fine-tuned and few-shot learning performance which is competitive with current state-of-the-art models, whilst requiring up to two orders of magnitude less training data. Code is available at https://github.com/NVlabs/prismer.
    
[^20]: Mesh-SORT: 简单而有效的位置跟踪器及其丢失管理策略

    Mesh-SORT: Simple and effective location-wise tracker with lost management strategies. (arXiv:2302.14415v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.14415](http://arxiv.org/abs/2302.14415)

    Mesh-SORT是一种简单而有效的位置跟踪器，通过网格分割帧并提出相应的位置感知的丢失管理策略和不同的匹配策略，解决了跟踪中的不良检测器和遮挡问题。

    Mesh-SORT is a simple and effective location-wise tracker that solves the problem of bad detectors and occlusions in tracking by dividing frames into grids and proposing corresponding location-aware loss management strategies and different matching strategies.

    多目标跟踪(MOT)由于其在交通和行人检测等领域的潜在应用而受到广泛关注。我们注意到，通过检测进行跟踪可能会受到噪声检测器产生的误差的影响，例如在遮挡之前的不精确边界框，而且在大多数跟踪场景中，物体往往会在特定位置移动和丢失。为了解决这个问题，我们提出了一种新的跟踪器来处理不良检测器和遮挡。首先，我们提出了一种位置感知的子区域识别方法，将帧等分为网格。然后，我们提出了相应的位置感知的丢失管理策略和不同的匹配策略。结果，Mesh-SORT的消融研究证明了其有效性，并使MOT17数据集上的3%碎片化、7.2% ID切换下降和0.4% MOTA改进与基线相比。最后，我们分析了其在特定场景下的局限性，并讨论了未来的工作。

    Multi-Object Tracking (MOT) has gained extensive attention in recent years due to its potential applications in traffic and pedestrian detection. We note that tracking by detection may suffer from errors generated by noise detectors, such as an imprecise bounding box before the occlusions, and observed that in most tracking scenarios, objects tend to move and lost within specific locations. To counter this, we present a novel tracker to deal with the bad detector and occlusions. Firstly, we proposed a location-wise sub-region recognition method which equally divided the frame, which we called mesh. Then we proposed corresponding location-wise loss management strategies and different matching strategies. The resulting Mesh-SORT, ablation studies demonstrate its effectiveness and made 3% fragmentation 7.2% ID switches drop and 0.4% MOTA improvement compared to the baseline on MOT17 datasets. Finally, we analyze its limitation on the specific scene and discussed what future works can be e
    
[^21]: 重新思考半监督医学图像分割：方差缩减的视角

    Rethinking Semi-Supervised Medical Image Segmentation: A Variance-Reduction Perspective. (arXiv:2302.01735v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.01735](http://arxiv.org/abs/2302.01735)

    本文提出了ARCO，一种半监督对比学习（CL）框架，其中包括医学图像分割中的分层组采样理论。通过方差缩减估计的概念来构建ARCO，并表明某些方差缩减技术在医学图像分割中特别有益。

    This paper proposes ARCO, a semi-supervised contrastive learning (CL) framework with stratified group sampling theory in medical image segmentation. The concept of variance-reduced estimation is used to build ARCO, and certain variance-reduction techniques are shown to be particularly beneficial in medical image segmentation.

    对于医学图像分割，对比学习是提高视觉表示质量的主要方法，通过对比语义相似和不相似的样本对来实现。这是通过观察到，在没有访问地面真实标签的情况下，如果采样具有真正不同解剖特征的负样本，则可以显着提高性能。然而，在现实中，这些样本可能来自相似的解剖特征，模型可能难以区分少数尾类样本，使得尾类更容易被错误分类，这通常导致模型崩溃。在本文中，我们提出了ARCO，一种半监督对比学习（CL）框架，其中包括医学图像分割中的分层组采样理论。特别是，我们首先提出通过方差缩减估计的概念来构建ARCO，并表明某些方差缩减技术在医学图像分割中特别有益。

    For medical image segmentation, contrastive learning is the dominant practice to improve the quality of visual representations by contrasting semantically similar and dissimilar pairs of samples. This is enabled by the observation that without accessing ground truth label, negative examples with truly dissimilar anatomical features, if sampled, can significantly improve the performance. In reality, however, these samples may come from similar anatomical features and the models may struggle to distinguish the minority tail-class samples, making the tail classes more prone to misclassification, both of which typically lead to model collapse. In this paper, we propose ARCO, a semi-supervised contrastive learning (CL) framework with stratified group sampling theory in medical image segmentation. In particular, we first propose building ARCO through the concept of variance-reduced estimation, and show that certain variance-reduction techniques are particularly beneficial in medical image se
    
[^22]: LDMIC：基于学习的分布式多视图图像编码

    LDMIC: Learning-based Distributed Multi-view Image Coding. (arXiv:2301.09799v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2301.09799](http://arxiv.org/abs/2301.09799)

    LDMIC是一种基于学习的分布式多视图图像编码框架，通过独立编码器和联合上下文传输模块实现了全局视图间的相关性捕捉，对几何关系不敏感。

    LDMIC is a learning-based distributed multi-view image coding framework that captures global inter-view correlations through independent encoders and a joint context transfer module based on the cross-attention mechanism, which is insensitive to geometric relations.

    多视图图像压缩在3D相关应用中起着至关重要的作用。现有方法采用预测编码架构，需要联合编码压缩相应的视差和残差信息。这要求相机之间进行协作，并强制执行不同视图之间的极线几何约束，这使得在具有随机重叠视野的分布式相机系统中部署这些方法具有挑战性。同时，分布式源编码理论表明，可以通过独立编码和联合解码实现相关源的高效数据压缩，这激发了我们设计基于学习的分布式多视图图像编码（LDMIC）框架的动机。通过独立编码器，LDMIC引入了一个简单而有效的基于交叉注意机制的联合上下文传输模块，以有效捕捉全局视图间的相关性，对几何关系不敏感。

    Multi-view image compression plays a critical role in 3D-related applications. Existing methods adopt a predictive coding architecture, which requires joint encoding to compress the corresponding disparity as well as residual information. This demands collaboration among cameras and enforces the epipolar geometric constraint between different views, which makes it challenging to deploy these methods in distributed camera systems with randomly overlapping fields of view. Meanwhile, distributed source coding theory indicates that efficient data compression of correlated sources can be achieved by independent encoding and joint decoding, which motivates us to design a learning-based distributed multi-view image coding (LDMIC) framework. With independent encoders, LDMIC introduces a simple yet effective joint context transfer module based on the cross-attention mechanism at the decoder to effectively capture the global inter-view correlations, which is insensitive to the geometric relation
    
[^23]: SegViz：基于联邦学习的多器官分割框架，适用于具有部分注释的异构数据集

    SegViz: A federated-learning based framework for multi-organ segmentation on heterogeneous data sets with partial annotations. (arXiv:2301.07074v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.07074](http://arxiv.org/abs/2301.07074)

    SegViz是一种基于联邦学习的框架，用于从分布式的非i.i.d数据集中训练具有部分注释的分割模型。使用FedBN作为聚合策略的SegViz框架在外部BTCV集上表现出优异的性能，分割的dice分数分别为0.93、0.83、0.55和0.75。

    SegViz is a federated learning-based framework for training segmentation models from distributed non-i.i.d datasets with partial annotations. The SegViz framework using FedBN as the aggregation strategy demonstrated excellent performance on the external BTCV set with dice scores of 0.93, 0.83, 0.55, and 0.75 for segmentation.

    分割是医学图像深度学习中最基本的任务之一，由于其多个下游临床应用而备受关注。然而，为医学图像生成手动注释是耗时的、需要高技能的、昂贵的工作，特别是对于3D图像。一个潜在的解决方案是从多个组的部分注释数据集中聚合知识，使用联邦学习协作训练全局模型。为此，我们提出了SegViz，一种基于联邦学习的框架，用于从分布式的非i.i.d数据集中训练具有部分注释的分割模型。将SegViz的性能与分别在每个数据集上单独训练模型以及集中聚合所有数据集并训练单个模型进行比较。使用FedBN作为聚合策略的SegViz框架在外部BTCV集上表现出优异的性能，分割的dice分数分别为0.93、0.83、0.55和0.75。

    Segmentation is one of the most primary tasks in deep learning for medical imaging, owing to its multiple downstream clinical applications. However, generating manual annotations for medical images is time-consuming, requires high skill, and is an expensive effort, especially for 3D images. One potential solution is to aggregate knowledge from partially annotated datasets from multiple groups to collaboratively train global models using Federated Learning. To this end, we propose SegViz, a federated learning-based framework to train a segmentation model from distributed non-i.i.d datasets with partial annotations. The performance of SegViz was compared against training individual models separately on each dataset as well as centrally aggregating all the datasets in one place and training a single model. The SegViz framework using FedBN as the aggregation strategy demonstrated excellent performance on the external BTCV set with dice scores of 0.93, 0.83, 0.55, and 0.75 for segmentation 
    
[^24]: 图像数据增强方法：综述与未来方向

    Image Data Augmentation Approaches: A Comprehensive Survey and Future directions. (arXiv:2301.02830v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.02830](http://arxiv.org/abs/2301.02830)

    本文综述了图像数据增强方法，提供了全面的分类法和每种技术的优缺点，并给出了数据增强对图像分类、目标检测和语义分割等计算机视觉任务的全面结果。

    This article provides a comprehensive survey of advanced data augmentation techniques for computer vision tasks, including a novel taxonomy and evaluation of each technique's strengths and weaknesses. The article also presents comprehensive results of the data augmentation effect on popular computer vision tasks such as image classification, object detection, and semantic segmentation.

    深度学习算法在各种计算机视觉任务中表现出了显著的性能。然而，由于标记数据有限，导致网络过拟合问题，即网络在未见过的数据上的性能比训练数据差。因此，它限制了性能的提高。为了应对这个问题，提出了各种技术，如dropout、归一化和高级数据增强。其中，数据增强旨在通过包括样本多样性来扩大数据集大小，近来成为热门话题。在本文中，我们重点关注高级数据增强技术。我们提供了数据增强的背景、一个新颖而全面的分类法、以及每种技术的优点和缺点（在可能的情况下）。我们还提供了数据增强对三个流行的计算机视觉任务（如图像分类、目标检测和语义分割）的全面结果。

    Deep learning (DL) algorithms have shown significant performance in various computer vision tasks. However, having limited labelled data lead to a network overfitting problem, where network performance is bad on unseen data as compared to training data. Consequently, it limits performance improvement. To cope with this problem, various techniques have been proposed such as dropout, normalization and advanced data augmentation. Among these, data augmentation, which aims to enlarge the dataset size by including sample diversity, has been a hot topic in recent times. In this article, we focus on advanced data augmentation techniques. we provide a background of data augmentation, a novel and comprehensive taxonomy of reviewed data augmentation techniques, and the strengths and weaknesses (wherever possible) of each technique. We also provide comprehensive results of the data augmentation effect on three popular computer vision tasks, such as image classification, object detection and seman
    
[^25]: 一次性领域自适应在基于视频的手术技能评估中的应用

    One-shot domain adaptation in video-based assessment of surgical skills. (arXiv:2301.00812v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.00812](http://arxiv.org/abs/2301.00812)

    本文提出了一种元学习模型A-VBANet，可以通过一次性学习提供领域不可知的手术技能分类，成功地适应了模拟任务和腹腔镜胆囊切除术，为基于视频的手术技能评估提供了领域不可知程序。

    This paper proposes a meta-learning model, A-VBANet, that can deliver domain-agnostic surgical skill classification via one-shot learning. The model successfully adapts to simulated tasks and laparoscopic cholecystectomy, providing a domain-agnostic procedure for video-based assessment of surgical skills.

    深度学习已经实现了手术技能的自动和客观评估。然而，深度学习模型需要大量数据，并且受限于其训练领域。这阻止了它们过渡到数据有限的新任务。因此，领域自适应对于在现实生活中实现深度学习至关重要。在这里，我们提出了一种元学习模型A-VBANet，它可以通过一次性学习提供领域不可知的手术技能分类。我们在五个腹腔镜和机器人手术模拟器上开发了A-VBANet。此外，我们在腹腔镜胆囊切除术的手术室视频上进行了测试。我们的模型成功地适应了模拟任务，准确率高达99.5%（一次性）和99.9%（少量样本），在腹腔镜胆囊切除术中的准确率为89.7%。我们首次提供了基于视频的手术技能评估的领域不可知程序。这种方法的一个重要影响是它允许使用来自手术模拟器的数据来评估手术表现。

    Deep Learning (DL) has achieved automatic and objective assessment of surgical skills. However, DL models are data-hungry and restricted to their training domain. This prevents them from transitioning to new tasks where data is limited. Hence, domain adaptation is crucial to implement DL in real life. Here, we propose a meta-learning model, A-VBANet, that can deliver domain-agnostic surgical skill classification via one-shot learning. We develop the A-VBANet on five laparoscopic and robotic surgical simulators. Additionally, we test it on operating room (OR) videos of laparoscopic cholecystectomy. Our model successfully adapts with accuracies up to 99.5% in one-shot and 99.9% in few-shot settings for simulated tasks and 89.7% for laparoscopic cholecystectomy. For the first time, we provide a domain-agnostic procedure for video-based assessment of surgical skills. A significant implication of this approach is that it allows the use of data from surgical simulators to assess performance 
    
[^26]: UniDA3D: 统一的域自适应三维语义分割管道

    UniDA3D: Unified Domain Adaptive 3D Semantic Segmentation Pipeline. (arXiv:2212.10390v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.10390](http://arxiv.org/abs/2212.10390)

    本文提出了UniDA3D，一种统一的域自适应三维语义分割管道，通过设计统一的源和目标主动采样策略，可以解决三维分割领域中的多个自适应任务，并探索了实现多模态采样策略的可能性。

    This paper proposes UniDA3D, a unified domain adaptive 3D semantic segmentation pipeline, which can tackle several adaptation tasks in 3D segmentation field by designing a unified source-and-target active sampling strategy, and investigates the possibility of achieving a multi-modal sampling strategy.

    目前的三维语义分割模型是在现成的公共基准上训练的，但当这些训练良好的模型部署到新领域时，它们将不可避免地面临识别精度下降的挑战。本文介绍了一种统一的域自适应三维语义分割管道（UniDA3D），以增强弱泛化能力，并弥合域之间的点分布差距。与之前只关注单一自适应任务的研究不同，UniDA3D可以通过设计统一的源和目标主动采样策略来解决三维分割领域中的多个自适应任务，该策略从源域和目标域中选择最具信息量的子集以实现有效的模型自适应。此外，受到多模态二维-三维数据集的崛起的影响，UniDA3D探索了实现多模态采样策略的可能性，通过开发跨模态特征交互模块，可以提取代表性对。

    State-of-the-art 3D semantic segmentation models are trained on off-the-shelf public benchmarks, but they will inevitably face the challenge of recognition accuracy drop when these well-trained models are deployed to a new domain. In this paper, we introduce a Unified Domain Adaptive 3D semantic segmentation pipeline (UniDA3D) to enhance the weak generalization ability, and bridge the point distribution gap between domains. Different from previous studies that only focus on a single adaptation task, UniDA3D can tackle several adaptation tasks in 3D segmentation field, by designing a unified source-and-target active sampling strategy, which selects a maximally-informative subset from both source and target domains for effective model adaptation. Besides, benefiting from the rise of multi-modal 2D-3D datasets, UniDA3D investigates the possibility of achieving a multi-modal sampling strategy, by developing a cross-modality feature interaction module that can extract a representative pair 
    
[^27]: 细调CLIP模型是高效的视频学习器

    Fine-tuned CLIP Models are Efficient Video Learners. (arXiv:2212.03640v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.03640](http://arxiv.org/abs/2212.03640)

    本文提出了一个简单的视频细调CLIP（ViFi-CLIP）基线，通过从CLIP图像编码器的帧级处理，接着进行特征池化和与相应文本嵌入的相似度匹配，有效地将图像级别的CLIP表示转移到视频中，从而弥合了从图像到视频的领域差距。

    This paper proposes a simple Video Fine-tuned CLIP (ViFi-CLIP) baseline, which effectively transfers image-level CLIP representations to videos by frame-level processing from CLIP image-encoder followed by feature pooling and similarity matching with corresponding text embeddings, thus bridging the domain gap from images to videos.

    通过图像-文本对的大规模多模态训练，CLIP模型具有强大的泛化能力。由于在类似规模上对视频进行训练是不可行的，因此最近的方法集中于将基于图像的CLIP有效地转移到视频领域。在这个追求中，添加了新的参数模块来学习时间信息和帧间关系，这需要精心设计。此外，当所得到的模型在视频上进行学习时，它们往往会过度拟合给定的任务分布，并且缺乏泛化方面。这引出了以下问题：如何有效地将图像级别的CLIP表示转移到视频中？在这项工作中，我们展示了一个简单的视频细调CLIP（ViFi-CLIP）基线通常足以弥合从图像到视频的领域差距。我们的定性分析表明，从CLIP图像编码器的帧级处理，接着进行特征池化和与相应文本嵌入的相似度匹配，有助于提高模型的性能。

    Large-scale multi-modal training with image-text pairs imparts strong generalization to CLIP model. Since training on a similar scale for videos is infeasible, recent approaches focus on the effective transfer of image-based CLIP to the video domain. In this pursuit, new parametric modules are added to learn temporal information and inter-frame relationships which require meticulous design efforts. Furthermore, when the resulting models are learned on videos, they tend to overfit on the given task distribution and lack in generalization aspect. This begs the following question: How to effectively transfer image-level CLIP representations to videos? In this work, we show that a simple Video Fine-tuned CLIP (ViFi-CLIP) baseline is generally sufficient to bridge the domain gap from images to videos. Our qualitative analysis illustrates that the frame-level processing from CLIP image-encoder followed by feature pooling and similarity matching with corresponding text embeddings helps in imp
    
[^28]: P{\O}DA: 基于提示的零样本领域自适应

    P{\O}DA: Prompt-driven Zero-shot Domain Adaptation. (arXiv:2212.03241v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.03241](http://arxiv.org/abs/2212.03241)

    本文提出了一种基于提示的零样本领域自适应方法，通过利用预训练的对比视觉语言模型（CLIP）来优化源特征的仿射变换，将其引导到目标文本嵌入中，同时保留其内容和语义。实验表明，该方法在几个数据集上显著优于基于CLIP的风格转移基线，用于下游任务。

    This paper proposes a prompt-driven zero-shot domain adaptation method, which leverages a pretrained contrastive vision-language model (CLIP) to optimize affine transformations of source features, steering them towards target text embeddings, while preserving their content and semantics. Experiments demonstrate that the method significantly outperforms CLIP-based style transfer baselines on several datasets for the downstream task at hand.

    领域自适应在计算机视觉领域得到了广泛的研究，但仍需要在训练时访问目标图像，这在某些不常见的情况下可能是不可行的。本文提出了“基于提示的零样本领域自适应”任务，其中我们仅使用目标域的单个通用文本描述（即提示）来调整在源域上训练的模型。首先，我们利用预训练的对比视觉语言模型（CLIP）来优化源特征的仿射变换，将其引导到目标文本嵌入中，同时保留其内容和语义。其次，我们展示了增强的特征可以用于执行语义分割的零样本领域自适应。实验表明，我们的方法在几个数据集上显著优于基于CLIP的风格转移基线，用于下游任务。我们的基于提示的方法甚至在某些数据集上优于一次性无监督领域自适应，并且gi

    Domain adaptation has been vastly investigated in computer vision but still requires access to target images at train time, which might be intractable in some uncommon conditions. In this paper, we propose the task of `Prompt-driven Zero-shot Domain Adaptation', where we adapt a model trained on a source domain using only a single general textual description of the target domain, i.e., a prompt. First, we leverage a pretrained contrastive vision-language model (CLIP) to optimize affine transformations of source features, steering them towards target text embeddings, while preserving their content and semantics. Second, we show that augmented features can be used to perform zero-shot domain adaptation for semantic segmentation. Experiments demonstrate that our method significantly outperforms CLIP-based style transfer baselines on several datasets for the downstream task at hand. Our prompt-driven approach even outperforms one-shot unsupervised domain adaptation on some datasets, and gi
    
[^29]: 统一视觉、文本和布局的通用文档处理

    Unifying Vision, Text, and Layout for Universal Document Processing. (arXiv:2212.02623v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.02623](http://arxiv.org/abs/2212.02623)

    本文提出了通用文档处理（UDOP）模型，将文本、图像和布局模态以及各种任务格式统一起来，通过一种新颖的Transformer模型实现预训练和多域下游任务的统一，同时实现了高质量的神经文档编辑和内容定制。

    This paper proposes the Universal Document Processing (UDOP) model, which unifies text, image, and layout modalities together with varied task formats, and achieves pretraining and multi-domain downstream tasks unification through a novel Transformer model. It also achieves high-quality neural document editing and content customization.

    我们提出了通用文档处理（UDOP），这是一个基础的文档AI模型，它将文本、图像和布局模态以及各种任务格式（包括文档理解和生成）统一起来。UDOP利用文本内容和文档图像之间的空间相关性，用一个统一的表示来建模图像、文本和布局模态。通过一种新颖的Vision-Text-Layout Transformer，UDOP将预训练和多域下游任务统一到基于提示的序列生成方案中。UDOP在大规模无标签文档语料库和多样化标记数据上使用创新的自监督目标进行预训练。UDOP还通过遮蔽图像重建学习从文本和布局模态生成文档图像。据我们所知，这是文档AI领域中第一次使用一个模型同时实现高质量的神经文档编辑和内容定制。我们的方法在8个文档处理基准数据集上取得了最先进的结果。

    We propose Universal Document Processing (UDOP), a foundation Document AI model which unifies text, image, and layout modalities together with varied task formats, including document understanding and generation. UDOP leverages the spatial correlation between textual content and document image to model image, text, and layout modalities with one uniform representation. With a novel Vision-Text-Layout Transformer, UDOP unifies pretraining and multi-domain downstream tasks into a prompt-based sequence generation scheme. UDOP is pretrained on both large-scale unlabeled document corpora using innovative self-supervised objectives and diverse labeled data. UDOP also learns to generate document images from text and layout modalities via masked image reconstruction. To the best of our knowledge, this is the first time in the field of document AI that one model simultaneously achieves high-quality neural document editing and content customization. Our method sets the state-of-the-art on 8 Docu
    
[^30]: 基于卷积和自注意力融合的敦煌壁画轮廓生成网络

    Dunhuang murals contour generation network based on convolution and self-attention fusion. (arXiv:2212.00935v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.00935](http://arxiv.org/abs/2212.00935)

    本文提出了一种基于卷积和自注意力融合的敦煌壁画轮廓生成网络，旨在解决传统卷积神经网络在感受野扩大时丢失局部细节信息的问题，以生成合理的壁画轮廓图。

    This paper proposes a Dunhuang murals contour generation network based on convolution and self-attention fusion, aiming to solve the problem of losing local detail information when the receptive field is enlarged in traditional convolutional neural networks, in order to generate reasonable mural contour drawings.

    敦煌壁画是中国风格和民族风格的集合，形成了一个独立的中国式佛教艺术。它具有非常高的历史和文化价值以及研究意义。其中，敦煌壁画的线条高度概括和表现力强，反映了角色独特的性格和复杂的内心情感。因此，壁画的轮廓图对于敦煌文化的研究具有重要意义。敦煌壁画的轮廓生成属于图像边缘检测，是计算机视觉的重要分支，旨在提取图像中显著的轮廓信息。虽然基于卷积的深度学习网络通过探索图像的上下文和语义特征在图像边缘提取方面取得了良好的结果。但是，随着感受野的扩大，一些局部细节信息会丢失。这使得它们无法生成合理的壁画轮廓图。在本文中，我们提出了一种基于卷积和自注意力融合的敦煌壁画轮廓生成网络。

    Dunhuang murals are a collection of Chinese style and national style, forming a self-contained Chinese-style Buddhist art. It has very high historical and cultural value and research significance. Among them, the lines of Dunhuang murals are highly general and expressive. It reflects the character's distinctive character and complex inner emotions. Therefore, the outline drawing of murals is of great significance to the research of Dunhuang Culture. The contour generation of Dunhuang murals belongs to image edge detection, which is an important branch of computer vision, aims to extract salient contour information in images. Although convolution-based deep learning networks have achieved good results in image edge extraction by exploring the contextual and semantic features of images. However, with the enlargement of the receptive field, some local detail information is lost. This makes it impossible for them to generate reasonable outline drawings of murals. In this paper, we propose 
    
[^31]: 从叉子到钳子：一种新的手术器械实例分割框架

    From Forks to Forceps: A New Framework for Instance Segmentation of Surgical Instruments. (arXiv:2211.16200v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.16200](http://arxiv.org/abs/2211.16200)

    本研究提出了一种新的神经网络框架，将分类模块作为现有实例分割模型的新阶段添加，用于改善手术器械实例分割中的分类问题。该模块包括多尺度掩模注意力，用于关注器械区域并掩盖分散的背景特征。

    

    微创手术和相关应用需要在实例级别上对手术工具进行分类和分割。手术工具在外观上相似，长而细，且以角度处理。将自然图像训练的最先进的实例分割模型微调用于器械分割时，往往难以区分器械类别。我们的研究表明，虽然边界框和分割掩模通常准确，但分类头误分类了手术器械的类别标签。我们提出了一个新的神经网络框架，将分类模块作为现有实例分割模型的新阶段添加。该模块专门用于改善现有模型生成的器械掩模的分类。该模块包括多尺度掩模注意力，该注意力关注器械区域并掩盖分散的背景特征。我们建议使用度量学习来训练分类器模块。

    Minimally invasive surgeries and related applications demand surgical tool classification and segmentation at the instance level. Surgical tools are similar in appearance and are long, thin, and handled at an angle. The fine-tuning of state-of-the-art (SOTA) instance segmentation models trained on natural images for instrument segmentation has difficulty discriminating instrument classes. Our research demonstrates that while the bounding box and segmentation mask are often accurate, the classification head mis-classifies the class label of the surgical instrument. We present a new neural network framework that adds a classification module as a new stage to existing instance segmentation models. This module specializes in improving the classification of instrument masks generated by the existing model. The module comprises multi-scale mask attention, which attends to the instrument region and masks the distracting background features. We propose training our classifier module using metr
    
[^32]: VideoFACT: 使用注意力、场景上下文和取证痕迹检测视频伪造

    VideoFACT: Detecting Video Forgeries Using Attention, Scene Context, and Forensic Traces. (arXiv:2211.15775v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.15775](http://arxiv.org/abs/2211.15775)

    本文提出了一种新的网络VideoFACT，它利用取证嵌入、上下文嵌入和深度自我注意机制来检测和定位各种视频伪造和操纵，克服了现有网络在分析视频时面临的挑战。

    This paper proposes a new network, VideoFACT, which utilizes forensic embeddings, context embeddings, and a deep self-attention mechanism to detect and localize a wide variety of video forgeries and manipulations, overcoming challenges faced by existing networks when analyzing videos.

    假视频代表了一个重要的误导威胁。虽然现有的取证网络已经在图像伪造方面表现出强大的性能，但最近在Adobe VideoSham数据集上报告的结果表明，这些网络无法识别视频中的虚假内容。在本文中，我们展示了这是由于视频编码引入了取证痕迹的局部变化。为了应对现有网络在分析视频时面临的挑战，我们的网络利用取证嵌入来捕捉操纵留下的痕迹，上下文嵌入来控制视频编码引入的取证痕迹的变化，以及深度自我注意机制来估计局部取证嵌入的质量和相对重要性。我们创建了几个新的视频伪造数据集，并使用这些数据集以及公开可用的数据进行实验。

    Fake videos represent an important misinformation threat. While existing forensic networks have demonstrated strong performance on image forgeries, recent results reported on the Adobe VideoSham dataset show that these networks fail to identify fake content in videos. In this paper, we show that this is due to video coding, which introduces local variation into forensic traces. In response, we propose VideoFACT - a new network that is able to detect and localize a wide variety of video forgeries and manipulations. To overcome challenges that existing networks face when analyzing videos, our network utilizes both forensic embeddings to capture traces left by manipulation, context embeddings to control for variation in forensic traces introduced by video coding, and a deep self-attention mechanism to estimate the quality and relative importance of local forensic embeddings. We create several new video forgery datasets and use these, along with publicly available data, to experimentally e
    
[^33]: SinFusion：在单张图像或视频上训练扩散模型

    SinFusion: Training Diffusion Models on a Single Image or Video. (arXiv:2211.11743v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.11743](http://arxiv.org/abs/2211.11743)

    本文提出了一种在单张图像或视频上训练扩散模型的方法，称为SinFusion。该模型可以解决各种图像/视频特定的操作任务，包括从少量帧中学习单个输入视频的运动和动态，生成相同动态场景的多样化新视频样本，将短视频推广为长视频（向前和向后）并执行视频上采样。

    

    扩散模型在图像和视频生成方面取得了巨大的进展，超过了GAN在质量和多样性方面。然而，它们通常是在非常大的数据集上训练的，并且不自然地适应于操作给定的输入图像或视频。在本文中，我们展示了如何通过在单个输入图像或视频上训练扩散模型来解决这个问题。我们的图像/视频特定扩散模型（SinFusion）学习单个图像或视频的外观和动态，同时利用扩散模型的条件能力。它可以解决各种图像/视频特定的操作任务。特别地，我们的模型可以从少量帧中学习单个输入视频的运动和动态。然后，它可以生成相同动态场景的多样化新视频样本，将短视频推广为长视频（向前和向后）并执行视频上采样。这些任务中的大多数都无法通过当前的视频特定生成方法实现。

    Diffusion models exhibited tremendous progress in image and video generation, exceeding GANs in quality and diversity. However, they are usually trained on very large datasets and are not naturally adapted to manipulate a given input image or video. In this paper we show how this can be resolved by training a diffusion model on a single input image or video. Our image/video-specific diffusion model (SinFusion) learns the appearance and dynamics of the single image or video, while utilizing the conditioning capabilities of diffusion models. It can solve a wide array of image/video-specific manipulation tasks. In particular, our model can learn from few frames the motion and dynamics of a single input video. It can then generate diverse new video samples of the same dynamic scene, extrapolate short videos into long ones (both forward and backward in time) and perform video upsampling. Most of these tasks are not realizable by current video-specific generation methods.
    
[^34]: LA-VocE: 使用神经声码器的低信噪比音频视觉语音增强

    LA-VocE: Low-SNR Audio-visual Speech Enhancement using Neural Vocoders. (arXiv:2211.10999v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2211.10999](http://arxiv.org/abs/2211.10999)

    LA-VocE是一种新的音频视觉语音增强方法，使用神经声码器将从嘈杂的音频视觉语音预测的mel频谱图转换为波形音频，适用于多种语言和不同水平的背景噪声和语音干扰。

    LA-VocE is a new audio-visual speech enhancement method that uses a neural vocoder to convert mel-spectrograms predicted from noisy audio-visual speech via a transformer-based architecture into waveform audio, and is applicable to multiple languages and different levels of background noise and speech interference.

    音频视觉语音增强旨在通过利用音频本身以及目标说话者的唇部运动从嘈杂的环境中提取干净的语音。这种方法已经被证明比仅使用音频的语音增强方法更有效，特别是对于消除干扰语音。尽管语音合成方面取得了最近的进展，但大多数音频视觉方法仍然使用频谱映射/掩蔽来重现干净的音频，通常会在现有的语音增强架构中添加视觉骨干。在这项工作中，我们提出了LA-VocE，一种新的两阶段方法，通过基于Transformer的架构从嘈杂的音频视觉语音预测mel频谱图，然后使用神经声码器（HiFi-GAN）将它们转换为波形音频。我们在数千个说话者和11种以上不同的语言上训练和评估我们的框架，并研究我们的模型适应不同水平的背景噪声和语音干扰的能力。我们的实验表明

    Audio-visual speech enhancement aims to extract clean speech from a noisy environment by leveraging not only the audio itself but also the target speaker's lip movements. This approach has been shown to yield improvements over audio-only speech enhancement, particularly for the removal of interfering speech. Despite recent advances in speech synthesis, most audio-visual approaches continue to use spectral mapping/masking to reproduce the clean audio, often resulting in visual backbones added to existing speech enhancement architectures. In this work, we propose LA-VocE, a new two-stage approach that predicts mel-spectrograms from noisy audio-visual speech via a transformer-based architecture, and then converts them into waveform audio using a neural vocoder (HiFi-GAN). We train and evaluate our framework on thousands of speakers and 11+ different languages, and study our model's ability to adapt to different levels of background noise and speech interference. Our experiments show that 
    
[^35]: HMOE: 基于超网络的专家混合模型用于领域泛化

    HMOE: Hypernetwork-based Mixture of Experts for Domain Generalization. (arXiv:2211.08253v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.08253](http://arxiv.org/abs/2211.08253)

    本文提出了一种新的领域泛化方法HMOE，它不需要领域标签，更具可解释性，使用超网络生成专家权重，能够在低维向量空间中探索专家的相似性，实验结果表明HMOE可以划分混合数据并取得更好的效果。

    This paper proposes a novel domain generalization method called HMOE, which does not rely on domain labels and is more interpretable. HMOE uses hypernetworks to generate experts' weights, which allows experts to share useful meta-knowledge and enables exploring experts' similarities in a low-dimensional vector space. Experimental results show that HMOE can divide mixed data and achieve better performance.

    由于领域转移，机器学习系统通常无法很好地推广到与训练数据不同的领域，这就是领域泛化（DG）的目的。尽管已经开发了各种各样的DG方法，但大多数缺乏可解释性，并且需要在许多实际场景中不可用的领域标签。本文提出了一种新的DG方法，称为HMOE：基于超网络的专家混合模型（MoE），它不依赖于领域标签，并且更具可解释性。MoE在识别数据中的异质模式方面证明了其有效性。对于DG问题，异质性正是由于领域转移而产生的。HMOE使用超网络将向量作为输入来生成专家权重，这使得专家可以共享有用的元知识，并能够在低维向量空间中探索专家的相似性。我们在公平和统一的基准测试-DomainBed下将HMOE与其他DG算法进行比较。我们的广泛实验表明，HMOE可以划分混合数据并取得更好的效果。

    Due to domain shift, machine learning systems typically fail to generalize well to domains different from those of training data, which is what domain generalization (DG) aims to address. Although various DG methods have been developed, most of them lack interpretability and require domain labels that are not available in many real-world scenarios. This paper presents a novel DG method, called HMOE: Hypernetwork-based Mixture of Experts (MoE), which does not rely on domain labels and is more interpretable. MoE proves effective in identifying heterogeneous patterns in data. For the DG problem, heterogeneity arises exactly from domain shift. HMOE uses hypernetworks taking vectors as input to generate experts' weights, which allows experts to share useful meta-knowledge and enables exploring experts' similarities in a low-dimensional vector space. We compare HMOE with other DG algorithms under a fair and unified benchmark-DomainBed. Our extensive experiments show that HMOE can divide mixe
    
[^36]: 使用2D投影从3D MRI体积中高效预测脑龄

    Efficient brain age prediction from 3D MRI volumes using 2D projections. (arXiv:2211.05762v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2211.05762](http://arxiv.org/abs/2211.05762)

    本文提出了一种使用2D投影从3D MRI体积中高效预测脑龄的方法，相比于使用3D CNN，该方法在计算速度上有两个数量级的提升，对于没有3D CNN昂贵GPU硬件的研究人员非常有用。

    This paper proposes an efficient method for predicting brain age from 3D MRI volumes using 2D projections, which is two orders of magnitude faster than using 3D CNNs and is important for researchers without access to expensive GPU hardware.

    在高分辨率医学体积上使用3D CNN非常计算密集，特别是对于像英国生物库这样的大型数据集，该库旨在扫描10万个受试者。在这里，我们证明了使用2D CNN在3D体积的几个2D投影（代表轴向，矢状面和冠状面切片的平均值和标准差）上进行预测脑龄时，可以获得合理的测试准确性。使用我们的方法，使用单个GPU进行的一次训练时，20324个受试者需要20-50秒，比小型3D CNN快两个数量级。这些结果对于没有3D CNN昂贵GPU硬件的研究人员非常重要。

    Using 3D CNNs on high resolution medical volumes is very computationally demanding, especially for large datasets like the UK Biobank which aims to scan 100,000 subjects. Here we demonstrate that using 2D CNNs on a few 2D projections (representing mean and standard deviation across axial, sagittal and coronal slices) of the 3D volumes leads to reasonable test accuracy when predicting the age from brain volumes. Using our approach, one training epoch with 20,324 subjects takes 20 - 50 seconds using a single GPU, which two orders of magnitude faster compared to a small 3D CNN. These results are important for researchers who do not have access to expensive GPU hardware for 3D CNNs.
    
[^37]: 操作气象学的机器学习教程，第二部分：神经网络和深度学习

    A Machine Learning Tutorial for Operational Meteorology, Part II: Neural Networks and Deep Learning. (arXiv:2211.00147v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.00147](http://arxiv.org/abs/2211.00147)

    本文讨论了机器学习在气象学中的应用，特别是神经网络和深度学习。涵盖了感知器、人工神经网络、卷积神经网络和U型网络等方法。

    This paper discusses the application of machine learning in meteorology, specifically neural networks and deep learning. It covers methods such as perceptrons, artificial neural networks, convolutional neural networks, and U-networks.

    在过去的十年中，机器学习在气象学中的应用迅速增长。特别是神经网络和深度学习的使用率前所未有。为了填补缺乏以气象学视角涵盖神经网络的资源，本文以平易近人的语言格式讨论了机器学习方法，针对操作气象学界。这是一对旨在为气象学家提供机器学习资源的论文中的第二篇。第一篇论文侧重于传统的机器学习方法（例如随机森林），而本文则涵盖了广泛的神经网络和深度学习方法。具体而言，本文涵盖了感知器、人工神经网络、卷积神经网络和U型网络。与第一篇论文一样，本文讨论了与神经网络及其训练相关的术语。然后，本文提供了每种方法背后的一些直觉，并以展示每种方法的实例来结束。

    Over the past decade the use of machine learning in meteorology has grown rapidly. Specifically neural networks and deep learning have been used at an unprecedented rate. In order to fill the dearth of resources covering neural networks with a meteorological lens, this paper discusses machine learning methods in a plain language format that is targeted for the operational meteorological community. This is the second paper in a pair that aim to serve as a machine learning resource for meteorologists. While the first paper focused on traditional machine learning methods (e.g., random forest), here a broad spectrum of neural networks and deep learning methods are discussed. Specifically this paper covers perceptrons, artificial neural networks, convolutional neural networks and U-networks. Like the part 1 paper, this manuscript discusses the terms associated with neural networks and their training. Then the manuscript provides some intuition behind every method and concludes by showing ea
    
[^38]: ViTASD：自闭症谱系障碍面部诊断的强健视觉Transformer基线

    ViTASD: Robust Vision Transformer Baselines for Autism Spectrum Disorder Facial Diagnosis. (arXiv:2210.16943v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.16943](http://arxiv.org/abs/2210.16943)

    本文提出了一种使用Vision Transformer进行儿童ASD计算分析的方法，该方法从大型面部表情数据集中提取知识，并提供模型结构可转移性。在标准ASD面部分析基准测试上进行的大量实验表明，ViTASD-L实现了新的最先进水平。

    This paper proposes a method for computational analysis of pediatric ASD using Vision Transformer, which extracts knowledge from large facial expression datasets and offers model structure transferability. Extensive experiments on standard ASD facial analysis benchmarks show that ViTASD-L achieves a new state-of-the-art.

    自闭症谱系障碍（ASD）是一种终身神经发育障碍，在全球范围内具有非常高的患病率。由于缺乏良好的基线，ASD面部分析在儿科患者中的研究进展受到了阻碍。本文提出了使用Vision Transformer（ViT）进行儿童ASD计算分析的方法。所提出的模型，称为ViTASD，从大型面部表情数据集中提取知识，并提供模型结构可转移性。具体而言，ViTASD采用普通的ViT从患者的面部图像中提取特征，并采用轻量级解码器和高斯过程层来增强ASD分析的鲁棒性。在标准ASD面部分析基准测试上进行的大量实验表明，我们的方法优于所有代表性的ASD面部分析方法，而ViTASD-L实现了新的最先进水平。我们的代码和预训练模型可在https://github.com/IrohX上获得。

    Autism spectrum disorder (ASD) is a lifelong neurodevelopmental disorder with very high prevalence around the world. Research progress in the field of ASD facial analysis in pediatric patients has been hindered due to a lack of well-established baselines. In this paper, we propose the use of the Vision Transformer (ViT) for the computational analysis of pediatric ASD. The presented model, known as ViTASD, distills knowledge from large facial expression datasets and offers model structure transferability. Specifically, ViTASD employs a vanilla ViT to extract features from patients' face images and adopts a lightweight decoder with a Gaussian Process layer to enhance the robustness for ASD analysis. Extensive experiments conducted on standard ASD facial analysis benchmarks show that our method outperforms all of the representative approaches in ASD facial analysis, while the ViTASD-L achieves a new state-of-the-art. Our code and pretrained models are available at https://github.com/IrohX
    
[^39]: 回放：迭代注意力用于音频识别

    Play It Back: Iterative Attention for Audio Recognition. (arXiv:2210.11328v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2210.11328](http://arxiv.org/abs/2210.11328)

    该论文提出了一种基于注意力的架构，通过选择性重复跨越音频序列的最具区分性的声音来进行关注，最终实现了在三个音频分类基准测试中始终实现最先进的性能。

    The paper proposes an end-to-end attention-based architecture that attends over the most discriminative sounds across the audio sequence through selective repetition, achieving consistently state-of-the-art performance across three audio-classification benchmarks.

    听觉认知的一个关键功能是随着时间的推移将特征声音与其相应的语义关联起来。人类试图区分细粒度音频类别时，通常会重播相同的区分性声音以增加其预测置信度。我们提出了一种端到端的基于注意力的架构，通过选择性重复跨越音频序列的最具区分性的声音来进行关注。我们的模型最初使用完整的音频序列，并通过插槽注意力迭代地细化重播的时间段。在每次播放时，所选段使用较小的跳跃长度重播，这代表了这些段内更高分辨率的特征。我们展示了我们的方法可以在三个音频分类基准测试中始终实现最先进的性能：AudioSet、VGG-Sound和EPIC-KITCHENS-100。

    A key function of auditory cognition is the association of characteristic sounds with their corresponding semantics over time. Humans attempting to discriminate between fine-grained audio categories, often replay the same discriminative sounds to increase their prediction confidence. We propose an end-to-end attention-based architecture that through selective repetition attends over the most discriminative sounds across the audio sequence. Our model initially uses the full audio sequence and iteratively refines the temporal segments replayed based on slot attention. At each playback, the selected segments are replayed using a smaller hop length which represents higher resolution features within these segments. We show that our method can consistently achieve state-of-the-art performance across three audio-classification benchmarks: AudioSet, VGG-Sound, and EPIC-KITCHENS-100.
    
[^40]: 自监督几何对应用于野外类别级6D物体姿态估计

    Self-Supervised Geometric Correspondence for Category-Level 6D Object Pose Estimation in the Wild. (arXiv:2210.07199v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.07199](http://arxiv.org/abs/2210.07199)

    本文提出了一种自监督学习方法，直接在大规模真实世界物体视频上进行类别级6D姿态估计。通过表面嵌入学习了输入图像和规范形状之间的密集对应关系，并提出了新颖的几何循环一致性损失。学习到的对应关系可以应用于6D姿态估计和其他任务。

    This paper proposes a self-supervised learning approach for category-level 6D object pose estimation in the wild, which reconstructs the canonical 3D shape of an object category and learns dense correspondences between input images and the canonical shape via surface embedding. The proposed novel geometrical cycle-consistency losses construct cycles across 2D-3D spaces, across different instances and different time steps. The learned correspondence can be applied for 6D pose estimation and other tasks.

    尽管6D物体姿态估计在计算机视觉和机器人领域有广泛的应用，但由于缺乏注释，它仍然远未解决。当转向类别级6D姿态时，问题变得更加具有挑战性，因为需要对未见实例进行泛化。目前的方法受到从模拟或从人类收集的注释的限制。在本文中，我们通过引入一种自监督学习方法，直接在大规模真实世界物体视频上进行类别级6D姿态估计，克服了这一障碍。我们的框架重构了物体类别的规范3D形状，并通过表面嵌入学习了输入图像和规范形状之间的密集对应关系。对于训练，我们提出了新颖的几何循环一致性损失，它们在2D-3D空间、不同实例和不同时间步之间构建循环。学习到的对应关系可以应用于6D姿态估计和其他任务。

    While 6D object pose estimation has wide applications across computer vision and robotics, it remains far from being solved due to the lack of annotations. The problem becomes even more challenging when moving to category-level 6D pose, which requires generalization to unseen instances. Current approaches are restricted by leveraging annotations from simulation or collected from humans. In this paper, we overcome this barrier by introducing a self-supervised learning approach trained directly on large-scale real-world object videos for category-level 6D pose estimation in the wild. Our framework reconstructs the canonical 3D shape of an object category and learns dense correspondences between input images and the canonical shape via surface embedding. For training, we propose novel geometrical cycle-consistency losses which construct cycles across 2D-3D spaces, across different instances and different time steps. The learned correspondence can be applied for 6D pose estimation and othe
    
[^41]: PDEBENCH：科学机器学习的广泛基准测试

    PDEBENCH: An Extensive Benchmark for Scientific Machine Learning. (arXiv:2210.07182v6 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.07182](http://arxiv.org/abs/2210.07182)

    PDEBench是一个基于偏微分方程的时间依赖性模拟任务基准套件，包括代码和数据，可用于对新型机器学习模型的性能进行基准测试，同时还可以与经典数值模拟和机器学习基线进行比较。

    PDEBench is a benchmark suite of time-dependent simulation tasks based on Partial Differential Equations (PDEs), which includes code and data to benchmark the performance of novel machine learning models against both classical numerical simulations and machine learning baselines.

    近年来，基于机器学习的物理系统建模受到了越来越多的关注。尽管取得了一些令人印象深刻的进展，但仍缺乏易于使用但具有挑战性和代表性的科学ML基准测试。我们介绍了PDEBench，这是一个基于偏微分方程（PDE）的时间依赖性模拟任务基准套件。PDEBench包括代码和数据，以对新型机器学习模型的性能进行基准测试，同时还可以与经典数值模拟和机器学习基线进行比较。我们提出的基准问题集具有以下独特特征：（1）与现有基准测试相比，PDE的范围更广，从相对常见的示例到更现实和困难的问题；（2）与先前的工作相比，准备好使用的数据集更大，包括跨更多初始和边界条件以及PDE参数的多个模拟运行；（3）更广泛的基准测试，包括更多的性能指标和评估方法。

    Machine learning-based modeling of physical systems has experienced increased interest in recent years. Despite some impressive progress, there is still a lack of benchmarks for Scientific ML that are easy to use but still challenging and representative of a wide range of problems. We introduce PDEBench, a benchmark suite of time-dependent simulation tasks based on Partial Differential Equations (PDEs). PDEBench comprises both code and data to benchmark the performance of novel machine learning models against both classical numerical simulations and machine learning baselines. Our proposed set of benchmark problems contribute the following unique features: (1) A much wider range of PDEs compared to existing benchmarks, ranging from relatively common examples to more realistic and difficult problems; (2) much larger ready-to-use datasets compared to prior work, comprising multiple simulation runs across a larger number of initial and boundary conditions and PDE parameters; (3) more exte
    
[^42]: 对抗性CNN扰动攻击的对称防御

    Symmetry Defense Against CNN Adversarial Perturbation Attacks. (arXiv:2210.04087v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.04087](http://arxiv.org/abs/2210.04087)

    本文提出了一种对称防御方法，通过翻转或水平翻转对称对抗样本来提高对抗性鲁棒性，同时使用子群对称性进行分类。

    This paper proposes a symmetry defense method to improve adversarial robustness by flipping or horizontally flipping symmetric adversarial samples, and uses subgroup symmetries for classification.

    卷积神经网络分类器（CNN）容易受到对抗性攻击，这些攻击会扰动原始样本以欺骗分类器，例如自动驾驶汽车的道路标志图像分类器。CNN在对称样本的分类中也缺乏不变性，因为CNN可以以不同的方式对称样本进行分类。考虑到CNN缺乏对抗性鲁棒性和CNN缺乏不变性，对称对抗样本的分类可能与其错误分类不同。本文通过设计一种对称防御来回答这个问题，在对抗者不知道防御的情况下，将对称对抗样本翻转或水平翻转后再进行分类。对于知道防御的对手，防御设计了一个Klein四个对称子群，其中包括水平翻转和像素反转对称性。对称防御使用子群对称性进行分类，以提高对抗性鲁棒性。

    Convolutional neural network classifiers (CNNs) are susceptible to adversarial attacks that perturb original samples to fool classifiers such as an autonomous vehicle's road sign image classifier. CNNs also lack invariance in the classification of symmetric samples because CNNs can classify symmetric samples differently. Considered together, the CNN lack of adversarial robustness and the CNN lack of invariance mean that the classification of symmetric adversarial samples can differ from their incorrect classification. Could symmetric adversarial samples revert to their correct classification? This paper answers this question by designing a symmetry defense that inverts or horizontally flips adversarial samples before classification against adversaries unaware of the defense. Against adversaries aware of the defense, the defense devises a Klein four symmetry subgroup that includes the horizontal flip and pixel inversion symmetries. The symmetry defense uses the subgroup symmetries in ac
    
[^43]: 可微分的自然语言指令解析和视觉定位在物体放置任务中的应用

    Differentiable Parsing and Visual Grounding of Natural Language Instructions for Object Placement. (arXiv:2210.00215v4 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2210.00215](http://arxiv.org/abs/2210.00215)

    ParaGon是一种可微分的自然语言指令解析和视觉定位方法，通过将语言指令解析为以对象为中心的图形表示，以单独定位对象，并使用一种新颖的基于粒子的图神经网络来推理关于带有不确定性的物体放置。

    ParaGon is a differentiable method for natural language instruction parsing and visual grounding in object placement tasks. It parses language instructions into an object-centric graph representation to ground objects individually and uses a novel particle-based graph neural network to reason about object placements with uncertainty.

    我们提出了一种新的方法，PARsing And visual GrOuNding (ParaGon)，用于在物体放置任务中对自然语言进行定位。自然语言通常用组合性和歧义性描述对象和空间关系，这是有效语言定位的两个主要障碍。对于组合性，ParaGon将语言指令解析为以对象为中心的图形表示，以单独定位对象。对于歧义性，ParaGon使用一种新颖的基于粒子的图神经网络来推理关于带有不确定性的物体放置。本质上，ParaGon将解析算法集成到概率的数据驱动学习框架中。它是完全可微分的，并从数据中端到端地训练，以对抗复杂的，模糊的语言输入。

    We present a new method, PARsing And visual GrOuNding (ParaGon), for grounding natural language in object placement tasks. Natural language generally describes objects and spatial relations with compositionality and ambiguity, two major obstacles to effective language grounding. For compositionality, ParaGon parses a language instruction into an object-centric graph representation to ground objects individually. For ambiguity, ParaGon uses a novel particle-based graph neural network to reason about object placements with uncertainty. Essentially, ParaGon integrates a parsing algorithm into a probabilistic, data-driven learning framework. It is fully differentiable and trained end-to-end from data for robustness against complex, ambiguous language input.
    
[^44]: 从自然语言知识中学习可转移的时空表示

    Learning Transferable Spatiotemporal Representations from Natural Script Knowledge. (arXiv:2209.15280v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.15280](http://arxiv.org/abs/2209.15280)

    本文提出了一种新的预文本任务，利用自然语言知识来提高可转移的时空表示学习，通过关注学习到的视频表示来对打乱的ASR脚本进行排序，从而提高视频理解的进展。

    This paper proposes a new pretext task to boost transferable spatiotemporal representation learning by exploiting natural language knowledge, which sorts shuffled ASR scripts by attending to learned video representations, and thus improves the progress of video understanding.

    近年来，预训练大规模视频数据已成为学习可转移的时空表示的常见方法。尽管取得了一些进展，但现有方法大多局限于高度策划的数据集（例如K400），并展示出令人不满意的开箱即用表示。我们认为这是因为它们只捕捉像素级别的知识而不是时空语义，这阻碍了视频理解的进一步进展。受到图像文本预训练（例如CLIP）的巨大成功的启发，我们迈出了利用语言语义提高可转移的时空表示学习的第一步。我们引入了一个新的预文本任务，即“Turning to Video for Transcript Sorting（TVTS）”，通过关注学习到的视频表示来对打乱的ASR脚本进行排序。我们不依赖于描述性标题，纯粹从视频中学习，即利用自然转录的语音知识在时间上提供嘈杂但有用的语义。

    Pre-training on large-scale video data has become a common recipe for learning transferable spatiotemporal representations in recent years. Despite some progress, existing methods are mostly limited to highly curated datasets (e.g., K400) and exhibit unsatisfactory out-of-the-box representations. We argue that it is due to the fact that they only capture pixel-level knowledge rather than spatiotemporal semantics, which hinders further progress in video understanding. Inspired by the great success of image-text pre-training (e.g., CLIP), we take the first step to exploit language semantics to boost transferable spatiotemporal representation learning. We introduce a new pretext task, Turning to Video for Transcript Sorting (TVTS), which sorts shuffled ASR scripts by attending to learned video representations. We do not rely on descriptive captions and learn purely from video, i.e., leveraging the natural transcribed speech knowledge to provide noisy but useful semantics over time. Our me
    
[^45]: Recipro-CAM: 基于快速无梯度的可解释性卷积神经网络的视觉解释方法

    Recipro-CAM: Fast gradient-free visual explanations for convolutional neural networks. (arXiv:2209.14074v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.14074](http://arxiv.org/abs/2209.14074)

    Recipro-CAM是一种快速无梯度的可解释性卷积神经网络的视觉解释方法，通过对提取的特征图进行空间掩蔽，利用激活图和目标类别的网络预测之间的相关性，解决了CAM和Grad-CAM方法的架构限制和梯度计算负担问题，具有更短的执行时间，适用于实际解决方案。

    Recipro-CAM is a fast gradient-free visual explanation method for interpretable convolutional neural networks. It solves the architectural constraints and gradient computing burden issues of CAM and Grad-CAM methods by spatially masking the extracted feature maps to exploit the correlation between activation maps and network predictions for target classes, with shorter execution time and practical applicability.

    卷积神经网络（CNN）是计算机视觉中广泛使用的深度学习架构。然而，它的黑盒本质使得难以解释模型的行为。为了缓解这个问题，AI从业者探索了可解释性AI方法，如Class Activation Map（CAM）和Grad-CAM。虽然这些方法表现出了很好的前景，但它们受到架构限制或梯度计算负担的限制。为了解决这个问题，提出了Score-CAM和Ablation-CAM作为无梯度方法，但与基于CAM或Grad-CAM的方法相比，它们具有更长的执行时间，使它们不适合实际解决方案，尽管它们解决了梯度相关问题并启用了推理模式XAI。为了解决这个挑战，我们提出了一种快速无梯度的Recipro-CAM方法。我们的方法涉及对提取的特征图进行空间掩蔽，以利用激活图和目标类别的网络预测之间的相关性。

    The Convolutional Neural Network (CNN) is a widely used deep learning architecture for computer vision. However, its black box nature makes it difficult to interpret the behavior of the model. To mitigate this issue, AI practitioners have explored explainable AI methods like Class Activation Map (CAM) and Grad-CAM. Although these methods have shown promise, they are limited by architectural constraints or the burden of gradient computing. To overcome this issue, Score-CAM and Ablation-CAM have been proposed as gradient-free methods, but they have longer execution times compared to CAM or Grad-CAM based methods, making them unsuitable for real-world solution though they resolved gradient related issues and enabled inference mode XAI. To address this challenge, we propose a fast gradient-free Reciprocal CAM (Recipro-CAM) method. Our approach involves spatially masking the extracted feature maps to exploit the correlation between activation maps and network predictions for target classes.
    
[^46]: 使用深度学习预测肺活检图像中的EGFR突变

    EGFR Mutation Prediction of Lung Biopsy Images using Deep Learning. (arXiv:2208.12506v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2208.12506](http://arxiv.org/abs/2208.12506)

    本文使用深度学习技术，通过对肺活检图像进行分析，实现了对EGFR突变的预测，为肺癌治疗提供了更经济、更快捷的诊断方法。

    This paper uses deep learning technology to predict EGFR mutations through analysis of lung biopsy images, providing a more economical and faster diagnostic method for lung cancer treatment.

    肺癌治疗的标准诊断程序涉及组织学亚型分型和随后的关键驱动基因突变检测，如EGFR。尽管分子分析可以揭示驱动基因突变，但该过程通常昂贵且耗时。基于深度学习的图像分析提供了一种更经济的选择，可以直接从全幻灯片图像（WSIs）中发现驱动基因突变。在这项工作中，我们使用定制的深度学习管道进行弱监督，以识别hematoxylin和eosin染色的WSIs中EGFR突变的形态学相关性，以及检测肿瘤和组织学亚型。我们通过在两个肺癌数据集（TCGA和来自印度的私人数据集）上进行严格的实验和消融研究来证明我们管道的有效性。使用我们的管道，我们实现了肿瘤检测的平均曲线下面积（AUC）为0.964，组织学亚型为0.942。

    The standard diagnostic procedures for targeted therapies in lung cancer treatment involve histological subtyping and subsequent detection of key driver mutations, such as EGFR. Even though molecular profiling can uncover the driver mutation, the process is often expensive and time-consuming. Deep learning-oriented image analysis offers a more economical alternative for discovering driver mutations directly from whole slide images (WSIs). In this work, we used customized deep learning pipelines with weak supervision to identify the morphological correlates of EGFR mutation from hematoxylin and eosin-stained WSIs, in addition to detecting tumor and histologically subtyping it. We demonstrate the effectiveness of our pipeline by conducting rigorous experiments and ablation studies on two lung cancer datasets - TCGA and a private dataset from India. With our pipeline, we achieved an average area under the curve (AUC) of 0.964 for tumor detection, and 0.942 for histological subtyping betwe
    
[^47]: 评估针对上下文和语义领域漂移的连续测试时间适应性

    Evaluating Continual Test-Time Adaptation for Contextual and Semantic Domain Shifts. (arXiv:2208.08767v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2208.08767](http://arxiv.org/abs/2208.08767)

    本文评估了针对上下文和语义领域漂移的连续测试时间适应性，无需标签。研究发现，连续测试时间适应性（CoTTA）是一种有效的方法。

    This paper evaluates continual test-time adaptation for contextual and semantic domain shifts without labels. The study finds that Continual Test-Time Adaptation (CoTTA) is an effective method.

    本文旨在将预训练的卷积神经网络在测试时间内连续适应领域漂移，无需标签。我们评估了最新技术，如预测时间批量归一化（BN）、测试熵最小化（TENT）和连续测试时间适应性（CoTTA），并在两个现实和具有挑战性的领域漂移源上进行了测试。

    In this paper, our goal is to adapt a pre-trained convolutional neural network to domain shifts at test time. We do so continually with the incoming stream of test batches, without labels. The existing literature mostly operates on artificial shifts obtained via adversarial perturbations of a test image. Motivated by this, we evaluate the state of the art on two realistic and challenging sources of domain shifts, namely contextual and semantic shifts. Contextual shifts correspond to the environment types, for example, a model pre-trained on indoor context has to adapt to the outdoor context on CORe-50. Semantic shifts correspond to the capture types, for example a model pre-trained on natural images has to adapt to cliparts, sketches, and paintings on DomainNet. We include in our analysis recent techniques such as Prediction-Time Batch Normalization (BN), Test Entropy Minimization (TENT) and Continual Test-Time Adaptation (CoTTA). Our findings are three-fold: i) Test-time adaptation me
    
[^48]: 局部稀疏不完整多视图聚类

    Localized Sparse Incomplete Multi-view Clustering. (arXiv:2208.02998v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2208.02998](http://arxiv.org/abs/2208.02998)

    本文提出了一种名为局部稀疏不完整多视图聚类（LSIMVC）的方法，通过优化稀疏正则化和新颖的图嵌入多视图矩阵分解模型，从不完整的多视图数据中学习稀疏和结构化的共识潜在表示，以解决不完整多视图聚类中的问题。

    This paper proposes a method named Localized Sparse Incomplete Multi-view Clustering (LSIMVC) to learn a sparse and structured consensus latent representation from incomplete multi-view data by optimizing a sparse regularized and novel graph embedded multi-view matrix factorization model, which aims to solve the clustering problem on incomplete multi-view data with partial view missing.

    近年来，不完整多视图聚类在解决部分视图缺失的不完整多视图数据聚类问题方面受到越来越多的关注。虽然已经开发了许多方法，但大多数方法要么不能灵活处理具有任意缺失视图的不完整多视图数据，要么不考虑视图之间信息不平衡的负面因素。此外，一些方法没有充分探索所有不完整视图的局部结构。为了解决这些问题，本文提出了一种简单而有效的方法，称为局部稀疏不完整多视图聚类（LSIMVC）。与现有方法不同，LSIMVC旨在通过优化稀疏正则化和新颖的图嵌入多视图矩阵分解模型来从不完整的多视图数据中学习稀疏和结构化的共识潜在表示。具体而言，在基于矩阵分解的新颖模型中，基于l1范数的稀疏正则化被用于学习局部结构，以及新颖的图嵌入被用于处理视图之间的信息不平衡。

    Incomplete multi-view clustering, which aims to solve the clustering problem on the incomplete multi-view data with partial view missing, has received more and more attention in recent years. Although numerous methods have been developed, most of the methods either cannot flexibly handle the incomplete multi-view data with arbitrary missing views or do not consider the negative factor of information imbalance among views. Moreover, some methods do not fully explore the local structure of all incomplete views. To tackle these problems, this paper proposes a simple but effective method, named localized sparse incomplete multi-view clustering (LSIMVC). Different from the existing methods, LSIMVC intends to learn a sparse and structured consensus latent representation from the incomplete multi-view data by optimizing a sparse regularized and novel graph embedded multi-view matrix factorization model. Specifically, in such a novel model based on the matrix factorization, a l1 norm based spa
    
[^49]: $L_2$BN: 通过等化特征的$L_2$范数来增强批量归一化

    $L_2$BN: Enhancing Batch Normalization by Equalizing the $L_2$ Norms of Features. (arXiv:2207.02625v5 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2207.02625](http://arxiv.org/abs/2207.02625)

    本文提出了一种$L_2$BN方法，通过等化样本特征的$L_2$范数来增强批量归一化，可以增强内类别特征的紧凑性并扩大跨类别特征的差异，易于实现，可以用作神经网络的基本归一化方法。

    This paper proposes an $L_2$BN method to enhance batch normalization by equalizing the $l_2$ norms of sample features, which can strengthen the compactness of intra-class features and enlarge the discrepancy of inter-class features, easy to implement, and can be used as a basic normalization method for neural networks.

    本文表明，样本特征的$L_2$范数差异会妨碍批量归一化获得更加显著的跨类别特征和更加紧凑的内类别特征。为了解决这个问题，我们提出了一种直观但有效的方法来等化样本特征的$L_2$范数。具体来说，我们在将样本特征输入批量归一化之前对每个样本特征进行$L_2$归一化，因此特征具有相同的数量级。由于所提出的方法结合了$L_2$归一化和批量归一化，因此我们将其命名为$L_2$BN。$L_2$BN可以增强内类别特征的紧凑性并扩大跨类别特征的差异。$L_2$BN易于实现，可以在没有任何额外参数或超参数的情况下发挥其作用。因此，它可以用作神经网络的基本归一化方法。我们通过对各种模型进行广泛的实验评估了$L_2$BN的有效性，用于图像分类。

    In this paper, we show that the difference in $l_2$ norms of sample features can hinder batch normalization from obtaining more distinguished inter-class features and more compact intra-class features. To address this issue, we propose an intuitive but effective method to equalize the $l_2$ norms of sample features. Concretely, we $l_2$-normalize each sample feature before feeding them into batch normalization, and therefore the features are of the same magnitude. Since the proposed method combines the $l_2$ normalization and batch normalization, we name our method $L_2$BN. The $L_2$BN can strengthen the compactness of intra-class features and enlarge the discrepancy of inter-class features. The $L_2$BN is easy to implement and can exert its effect without any additional parameters or hyper-parameters. Therefore, it can be used as a basic normalization method for neural networks. We evaluate the effectiveness of $L_2$BN through extensive experiments with various models on image classif
    
[^50]: 基于解剖感知对比蒸馏的半监督医学图像分割引导启动

    Bootstrapping Semi-supervised Medical Image Segmentation with Anatomical-aware Contrastive Distillation. (arXiv:2206.02307v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2206.02307](http://arxiv.org/abs/2206.02307)

    本文提出了一种基于解剖感知对比蒸馏的半监督医学图像分割引导启动方法，通过软标记负样本和捕获更多语义上相似的特征来解决医学图像数据不平衡的问题。

    This paper proposes a semi-supervised medical image segmentation bootstrapping method based on anatomical-aware contrastive distillation, which solves the problem of imbalanced medical image data by softly labeling negative samples and capturing more semantically similar features.

    对比学习已经在医学图像分割的注释稀缺问题上显示出了巨大的潜力。现有的方法通常假设标记和未标记的医学图像具有平衡的类分布。然而，现实中的医学图像数据通常是不平衡的（即多类标签不平衡），这自然地产生模糊的轮廓并通常错误地标记罕见的对象。此外，所有负样本是否同样负面仍不清楚。在这项工作中，我们提出了ACTION，一种解剖感知对比蒸馏框架，用于半监督医学图像分割。具体而言，我们首先通过软标记负样本而不是正负对之间的二元监督来开发迭代对比蒸馏算法。与正样本相比，我们还从随机选择的负样本集中捕获更多语义上相似的特征，以强制执行采样数据的多样性。其次，我们提出了一种基于解剖感知的启动方法，以更好地利用有限的标记数据。

    Contrastive learning has shown great promise over annotation scarcity problems in the context of medical image segmentation. Existing approaches typically assume a balanced class distribution for both labeled and unlabeled medical images. However, medical image data in reality is commonly imbalanced (i.e., multi-class label imbalance), which naturally yields blurry contours and usually incorrectly labels rare objects. Moreover, it remains unclear whether all negative samples are equally negative. In this work, we present ACTION, an Anatomical-aware ConTrastive dIstillatiON framework, for semi-supervised medical image segmentation. Specifically, we first develop an iterative contrastive distillation algorithm by softly labeling the negatives rather than binary supervision between positive and negative pairs. We also capture more semantically similar features from the randomly chosen negative set compared to the positives to enforce the diversity of the sampled data. Second, we raise a m
    
[^51]: 使用物体感知表示在多物体场景中进行视觉运动控制

    Visuomotor Control in Multi-Object Scenes Using Object-Aware Representations. (arXiv:2205.06333v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2205.06333](http://arxiv.org/abs/2205.06333)

    本文探讨了使用物体感知表示学习技术进行机器人任务的有效性，以解决当前方法学习任务特定表示不能很好地转移到其他任务的问题，以及由监督方法学习的表示需要大量标记数据集的问题。

    This paper explores the effectiveness of using object-aware representation learning techniques for robotic tasks, to address the problem that current methodologies learn task specific representations that do not necessarily transfer well to other tasks, and that representations learned by supervised methods require large labeled datasets for each task that are expensive to collect in the real world.

    场景的感知理解以及其不同组件之间的关系对于成功完成机器人任务至关重要。表示学习已被证明是一种强大的技术，但大多数当前的方法学习任务特定的表示，不一定能够很好地转移到其他任务。此外，由监督方法学习的表示需要大量标记数据集，这在现实世界中收集起来很昂贵。使用自监督学习从未标记的数据中获取表示可以缓解这个问题。然而，当前的自监督表示学习方法大多是物体无关的，我们证明了由此得到的表示对于具有许多组件的场景的通用机器人任务是不足够的。在本文中，我们探讨了使用物体感知表示学习技术进行机器人任务的有效性。

    Perceptual understanding of the scene and the relationship between its different components is important for successful completion of robotic tasks. Representation learning has been shown to be a powerful technique for this, but most of the current methodologies learn task specific representations that do not necessarily transfer well to other tasks. Furthermore, representations learned by supervised methods require large labeled datasets for each task that are expensive to collect in the real world. Using self-supervised learning to obtain representations from unlabeled data can mitigate this problem. However, current self-supervised representation learning methods are mostly object agnostic, and we demonstrate that the resulting representations are insufficient for general purpose robotics tasks as they fail to capture the complexity of scenes with many components. In this paper, we explore the effectiveness of using object-aware representation learning techniques for robotic tasks. 
    
[^52]: 通过检测错误的位置嵌入进行表示学习

    Representation Learning by Detecting Incorrect Location Embeddings. (arXiv:2204.04788v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2204.04788](http://arxiv.org/abs/2204.04788)

    本文提出了一种新的自监督学习（SSL）损失，用于图像表示学习。通过检测错误的位置嵌入，我们可以提高深度神经网络的泛化能力，使其更加鲁棒。我们称这种方法为DILEMMA，将其应用于MoCoV3、DINO和SimCLR，分别显示它们的性能提高了4.41%、3.97%和0.5%。

    

    本文提出了一种新的自监督学习（SSL）损失，用于图像表示学习。我们认为深度神经网络的泛化能力与其区分对象形状的能力有关。由于对象形状与其部件的位置有关，因此我们提出检测那些被人为移位的部件。我们用图像令牌表示对象部件，并训练ViT检测哪个令牌与错误的位置嵌入组合。然后，我们引入输入的稀疏性，使模型更加鲁棒，以应对遮挡并加速训练。我们称这种方法为DILEMMA，即检测错误位置嵌入和掩蔽输入。我们将DILEMMA应用于MoCoV3、DINO和SimCLR，并在相同的训练时间内，在ImageNet-1K上进行线性探测转移，分别显示它们的性能提高了4.41%、3.97%和0.5%。我们还展示了MAE与我们的完全微调改进的结果。

    In this paper, we introduce a novel self-supervised learning (SSL) loss for image representation learning. There is a growing belief that generalization in deep neural networks is linked to their ability to discriminate object shapes. Since object shape is related to the location of its parts, we propose to detect those that have been artificially misplaced. We represent object parts with image tokens and train a ViT to detect which token has been combined with an incorrect positional embedding. We then introduce sparsity in the inputs to make the model more robust to occlusions and to speed up the training. We call our method DILEMMA, which stands for Detection of Incorrect Location EMbeddings with MAsked inputs. We apply DILEMMA to MoCoV3, DINO and SimCLR and show an improvement in their performance of respectively 4.41%, 3.97%, and 0.5% under the same training time and with a linear probing transfer on ImageNet-1K. We also show full fine-tuning improvements of MAE combined with our 
    
[^53]: 生成建模有助于弱监督（反之亦然）

    Generative Modeling Helps Weak Supervision (and Vice Versa). (arXiv:2203.12023v6 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2203.12023](http://arxiv.org/abs/2203.12023)

    本文提出了一种融合程序弱监督和生成对抗网络的模型，通过对齐离散潜在变量和弱监督派生的标签估计，改善了未观察到的标签的估计，实现了数据增强。

    This paper proposes a model that fuses programmatic weak supervision and generative adversarial networks, improving the estimate of unobserved labels by aligning discrete latent variables and weak supervision derived label estimate, and enabling data augmentation through weak supervision.

    许多有前途的监督式机器学习应用在获取足够数量和质量的标记数据方面面临困难，从而造成昂贵的瓶颈。为了克服这些限制，研究了不依赖于基本真实标签的技术，包括弱监督和生成建模。虽然这些技术似乎可以共同使用，相互改进，但如何在它们之间建立接口尚不为人所知。在这项工作中，我们提出了一种融合程序弱监督和生成对抗网络的模型，并提供了理论上的理由来支持这种融合。所提出的方法捕捉数据中的离散潜在变量以及弱监督派生的标签估计。两者的对齐允许更好地建模弱监督来源的样本相关准确性，从而改善未观察到的标签的估计。这是第一种通过弱监督实现数据增强的方法。

    Many promising applications of supervised machine learning face hurdles in the acquisition of labeled data in sufficient quantity and quality, creating an expensive bottleneck. To overcome such limitations, techniques that do not depend on ground truth labels have been studied, including weak supervision and generative modeling. While these techniques would seem to be usable in concert, improving one another, how to build an interface between them is not well-understood. In this work, we propose a model fusing programmatic weak supervision and generative adversarial networks and provide theoretical justification motivating this fusion. The proposed approach captures discrete latent variables in the data alongside the weak supervision derived label estimate. Alignment of the two allows for better modeling of sample-dependent accuracies of the weak supervision sources, improving the estimate of unobserved labels. It is the first approach to enable data augmentation through weakly supervi
    
[^54]: 面向异构遥感图像的目标变化检测，用于森林死亡率映射

    Towards Targeted Change Detection with Heterogeneous Remote Sensing Images for Forest Mortality Mapping. (arXiv:2203.00049v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2203.00049](http://arxiv.org/abs/2203.00049)

    本文提出了一种基于图像到图像转换和单类分类的新方法，用于检测生态系统某些干扰的弱信号，以绘制由几何蛾爆发引起的森林死亡率在稀疏森林-苔原生态过渡带中的地图。

    

    最近已经开发了几种用于异构遥感数据（例如合成孔径雷达（SAR）和多光谱辐射计）的变化检测的通用方法。然而，这些方法不适合检测生态系统某些干扰的弱信号。为了解决这个问题，我们提出了一种基于图像到图像转换和单类分类（OCC）的新方法。我们旨在使用多源卫星图像绘制由几何蛾爆发引起的森林死亡率在稀疏森林-苔原生态过渡带中的地图。事件前和事件后的图像分别由Landsat-5和RADARSAT-2收集。使用最近的深度学习方法进行变化感知图像转换，我们在两个卫星各自的域中计算差异图像。这些差异与原始的事件前和事件后的图像堆叠，并传递给在目标变化类的小样本上训练的OCC。分类器产生一个

    Several generic methods have recently been developed for change detection in heterogeneous remote sensing data, such as images from synthetic aperture radar (SAR) and multispectral radiometers. However, these are not well suited to detect weak signatures of certain disturbances of ecological systems. To resolve this problem we propose a new approach based on image-to-image translation and one-class classification (OCC). We aim to map forest mortality caused by an outbreak of geometrid moths in a sparsely forested forest-tundra ecotone using multisource satellite images. The images preceding and following the event are collected by Landsat-5 and RADARSAT-2, respectively. Using a recent deep learning method for change-aware image translation, we compute difference images in both satellites' respective domains. These differences are stacked with the original pre- and post-event images and passed to an OCC trained on a small sample from the targeted change class. The classifier produces a 
    
[^55]: 用微小数据集实现网络加速的实践

    Practical Network Acceleration with Tiny Sets. (arXiv:2202.07861v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2202.07861](http://arxiv.org/abs/2202.07861)

    本文提出了一种基于删除块的方法，用于加速稀缺训练样本的网络，通过提出的可恢复性概念选择要删除的块，提出了一种名为PRACTISE的算法，仅使用微小的训练图像集来加速网络，PRACTISE的表现优于以往的方法。

    

    由于数据隐私问题，使用微小的训练集加速网络已成为实践中的一个关键需求。以往的方法主要采用滤波器级别的剪枝来加速稀缺训练样本的网络。在本文中，我们揭示了在这种情况下，删除块是一种基本上更优越的方法。它享有更高的加速比，并在少样本情况下产生更好的延迟-准确性性能。为了选择要删除的块，我们提出了一个新的概念，即可恢复性，用于衡量压缩网络的恢复难度。我们的可恢复性对于选择要删除的块是高效和有效的。最后，我们提出了一种名为PRACTISE的算法，仅使用微小的训练图像集来加速网络。PRACTISE的表现优于以往的方法。对于22％的延迟降低，PRACTISE在ImageNet-1k上平均超过以往方法7％。它还具有高度的泛化能力，在数据隐私场景和各种网络架构下表现良好。

    Due to data privacy issues, accelerating networks with tiny training sets has become a critical need in practice. Previous methods mainly adopt filter-level pruning to accelerate networks with scarce training samples. In this paper, we reveal that dropping blocks is a fundamentally superior approach in this scenario. It enjoys a higher acceleration ratio and results in a better latency-accuracy performance under the few-shot setting. To choose which blocks to drop, we propose a new concept namely recoverability to measure the difficulty of recovering the compressed network. Our recoverability is efficient and effective for choosing which blocks to drop. Finally, we propose an algorithm named PRACTISE to accelerate networks using only tiny sets of training images. PRACTISE outperforms previous methods by a significant margin. For 22% latency reduction, PRACTISE surpasses previous methods by on average 7% on ImageNet-1k. It also enjoys high generalization ability, working well under data
    
[^56]: I-Tuning: 利用图像对冻结语言模型进行轻量级图像字幕生成的调整

    I-Tuning: Tuning Frozen Language Models with Image for Lightweight Image Captioning. (arXiv:2202.06574v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2202.06574](http://arxiv.org/abs/2202.06574)

    本文提出了一种轻量级的图像字幕生成框架（I-Tuning），通过设计新颖的交叉注意力模块将不可训练的预训练语言解码器和视觉编码器连接起来，使得模型包含的可训练参数少，训练速度快，同时在三个图像字幕生成基准测试上实现了与大规模基线系统相当或更好的性能，但需要的可训练参数和训练数据量都少得多。

    This paper proposes a lightweight image captioning framework (I-Tuning) that connects the non-trainable pre-trained language decoder GPT2 and vision encoder CLIP-ViT with a novel I-Tuning cross-attention module. The framework contains fewer trainable parameters and achieves comparable or better performance than large-scale baseline systems on three image captioning benchmarks, while requiring much fewer trainable parameters and training data compared with state-of-the-art baselines.

    图像字幕生成是一项传统的视觉与语言任务，旨在生成图像的语言描述。最近的研究集中在扩大模型规模和训练数据量，这显著增加了模型训练的成本。与这些高成本模型不同，我们引入了一个轻量级的图像字幕生成框架（I-Tuning），其中包含少量可训练参数。我们设计了一种新颖的I-Tuning交叉注意力模块，将不可训练的预训练语言解码器GPT2和视觉编码器CLIP-ViT连接起来。由于大多数参数在训练期间不需要更新，因此我们的框架轻巧快速。在三个图像字幕生成基准测试上进行的实验结果表明，我们的框架实现了与大规模基线系统相当或更好的性能。但与最先进的基线相比，我们的模型包含多达10倍少的可训练参数，并且需要更少的数据进行训练。

    Image Captioning is a traditional vision-and-language task that aims to generate the language description of an image. Recent studies focus on scaling up the model size and the number of training data, which significantly increase the cost of model training. Different to these heavy-cost models, we introduce a lightweight image captioning framework (I-Tuning), which contains a small number of trainable parameters. We design a novel I-Tuning cross-attention module to connect the non-trainable pre-trained language decoder GPT2 and vision encoder CLIP-ViT. Since most parameters are not required to be updated during training, our framework is lightweight and fast. Experimental results conducted on three image captioning benchmarks reveal that our framework achieves comparable or better performance than the large-scale baseline systems. But our models contain up to 10 times fewer trainable parameters and require much fewer data for training compared with state-of-the-art baselines.
    
[^57]: 视频中的时间句子定位：综述与未来方向

    Temporal Sentence Grounding in Videos: A Survey and Future Directions. (arXiv:2201.08071v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2201.08071](http://arxiv.org/abs/2201.08071)

    本文综述了视频中的时间句子定位（TSGV）的基本概念和当前研究现状，以及未来研究方向。TSGV旨在从未经修剪的视频中检索与语言查询语义对应的时间时刻，连接计算机视觉和自然语言，是两个社区研究人员的重点关注点。

    This survey summarizes the fundamental concepts and current research status of temporal sentence grounding in videos (TSGV), also known as natural language video localization (NLVL) or video moment retrieval (VMR), as well as future research directions. TSGV aims to retrieve a temporal moment that semantically corresponds to a language query from an untrimmed video, connecting computer vision and natural language, and has drawn significant attention from researchers in both communities.

    视频中的时间句子定位（TSGV），又称自然语言视频定位（NLVL）或视频时刻检索（VMR），旨在从未经修剪的视频中检索与语言查询语义对应的时间时刻。连接计算机视觉和自然语言，TSGV引起了两个社区研究人员的重视。本综述试图提供TSGV中基本概念和当前研究现状的总结，以及未来研究方向。作为背景，我们以教程的形式介绍了TSGV中功能组件的常见结构：从原始视频和语言查询的特征提取到目标时刻的答案预测。然后，我们回顾了多模态理解和交互的技术，这是TSGV的重点关注点，以实现两种模态之间的有效对齐。我们构建了TSGV技术的分类法，并详细阐述了不同类别的方法及其优缺点。

    Temporal sentence grounding in videos (TSGV), \aka natural language video localization (NLVL) or video moment retrieval (VMR), aims to retrieve a temporal moment that semantically corresponds to a language query from an untrimmed video. Connecting computer vision and natural language, TSGV has drawn significant attention from researchers in both communities. This survey attempts to provide a summary of fundamental concepts in TSGV and current research status, as well as future research directions. As the background, we present a common structure of functional components in TSGV, in a tutorial style: from feature extraction from raw video and language query, to answer prediction of the target moment. Then we review the techniques for multimodal understanding and interaction, which is the key focus of TSGV for effective alignment between the two modalities. We construct a taxonomy of TSGV techniques and elaborate the methods in different categories with their strengths and weaknesses. La
    
[^58]: 基于通道特征去噪的指纹呈现攻击检测

    Fingerprint Presentation Attack Detection by Channel-wise Feature Denoising. (arXiv:2111.07620v2 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2111.07620](http://arxiv.org/abs/2111.07620)

    本文提出了一种基于通道特征去噪的指纹呈现攻击检测方法，通过处理先前研究中忽略的冗余噪声信息来学习指纹图像的重要特征，具有较好的鲁棒性和准确性。

    This paper proposes a novel channel-wise feature denoising fingerprint presentation attack detection (CFD-PAD) method, which learns important features of fingerprint images by handling the redundant noise information ignored in previous studies, and exhibits good robustness and accuracy under various attack types.

    由于攻击材料的多样性，指纹识别系统（AFRS）容易受到恶意攻击。因此，提出有效的指纹呈现攻击检测（PAD）方法对于AFRS的安全性和可靠性至关重要。然而，当前的PAD方法在新的攻击类型设置下往往表现出较差的鲁棒性。因此，本文提出了一种新颖的基于通道特征去噪的指纹PAD（CFD-PAD）方法，通过处理先前研究中忽略的冗余噪声信息来学习指纹图像的重要特征。该方法通过权衡每个通道的重要性并识别出有区别性的通道和“噪声”通道来学习指纹图像的重要特征。然后，在特征图中抑制“噪声”通道的传播以减少干扰。具体而言，设计了PA-Adaptation损失来约束特征分布，使活体指纹的特征分布更聚合，而欺诈指纹的特征分布更分散。实验结果表明，所提出的方法在各种攻击类型下均具有较好的鲁棒性和准确性。

    Due to the diversity of attack materials, fingerprint recognition systems (AFRSs) are vulnerable to malicious attacks. It is thus important to propose effective fingerprint presentation attack detection (PAD) methods for the safety and reliability of AFRSs. However, current PAD methods often exhibit poor robustness under new attack types settings. This paper thus proposes a novel channel-wise feature denoising fingerprint PAD (CFD-PAD) method by handling the redundant noise information ignored in previous studies. The proposed method learns important features of fingerprint images by weighing the importance of each channel and identifying discriminative channels and "noise" channels. Then, the propagation of "noise" channels is suppressed in the feature map to reduce interference. Specifically, a PA-Adaptation loss is designed to constrain the feature distribution to make the feature distribution of live fingerprints more aggregate and that of spoof fingerprints more disperse. Experime
    
[^59]: 二进制神经网络的全面综述

    A comprehensive review of Binary Neural Network. (arXiv:2110.06804v4 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2110.06804](http://arxiv.org/abs/2110.06804)

    本文全面综述了二进制神经网络的最新发展，重点关注1位激活和1位卷积网络的权重，这些网络可以在微小的受限设备上实现和嵌入，并节省大量存储、计算成本和能量消耗。

    This article provides a comprehensive overview of recent developments in Binary Neural Networks (BNN), with a focus on 1-bit activations and 1-bit convolution networks. These networks can be implemented and embedded on tiny restricted devices, saving significant storage, computation cost, and energy consumption.

    深度学习（DL）最近改变了智能系统的发展，并被广泛应用于许多实际应用中。尽管DL具有各种好处和潜力，但在不同的计算受限和能量受限设备中需要进行DL处理。研究二进制神经网络（BNN）等具有改变游戏规则的技术以增加深度学习能力是很自然的。最近在BNN方面取得了显着进展，因为它们可以在微小的受限设备上实现和嵌入，并节省大量存储、计算成本和能量消耗。然而，几乎所有的BNN行为都会带来额外的内存、计算成本和更高的性能。本文提供了BNN最近发展的完整概述。本文专门关注1位激活和1位卷积网络的权重，与以前的调查混合使用低位作品相反。它对BNN的开发进行了全面调查。

    Deep learning (DL) has recently changed the development of intelligent systems and is widely adopted in many real-life applications. Despite their various benefits and potentials, there is a high demand for DL processing in different computationally limited and energy-constrained devices. It is natural to study game-changing technologies such as Binary Neural Networks (BNN) to increase deep learning capabilities. Recently remarkable progress has been made in BNN since they can be implemented and embedded on tiny restricted devices and save a significant amount of storage, computation cost, and energy consumption. However, nearly all BNN acts trade with extra memory, computation cost, and higher performance. This article provides a complete overview of recent developments in BNN. This article focuses exclusively on 1-bit activations and weights 1-bit convolution networks, contrary to previous surveys in which low-bit works are mixed in. It conducted a complete investigation of BNN's dev
    

