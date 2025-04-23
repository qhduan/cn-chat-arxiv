# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Facilitating Reinforcement Learning for Process Control Using Transfer Learning: Perspectives](https://arxiv.org/abs/2404.00247) | 本文从迁移学习的角度探讨了如何将其与强化学习相结合，为过程控制带来新的可能性。 |
| [^2] | [Cross--domain Fiber Cluster Shape Analysis for Language Performance Cognitive Score Prediction](https://arxiv.org/abs/2403.19001) | 本研究通过新颖的框架SFFormer，结合了多头交叉注意力特征融合模块，基于dMRI纤维束追踪，预测了主观语言表现，拓展了脑结构与人类认知功能的关联研究。 |
| [^3] | [Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models trained on Corrupted Data](https://arxiv.org/abs/2403.08728) | 提出了一种使用环境扩散后验采样解决逆问题的框架，能在受损数据上训练的扩散模型上表现出色，并在图像恢复和MRI模型训练中取得优越性能。 |
| [^4] | [QuaCer-C: Quantitative Certification of Knowledge Comprehension in LLMs](https://arxiv.org/abs/2402.15929) | 本文提出了一种新颖的认证框架QuaCer-C，用于正式认证大型语言模型中知识理解的能力，证书定量化且包含高置信度的概率界限，研究发现，随着参数数量的增加，知识理解能力提高，Mistral模型在这一评估中表现不如其他模型。 |
| [^5] | [WikiMT++ Dataset Card.](http://arxiv.org/abs/2309.13259) | WikiMT++是一个扩展和精细版本的WikiMusicText数据集，包含了1010个经过策划的ABC记谱法的主题曲。它添加了客观属性和主观情感属性，增强了数据集的应用场景和可用性，并通过CLaMP来纠正属性，提高准确性和完整性。 |
| [^6] | [AI-Copilot for Business Optimisation: A Framework and A Case Study in Production Scheduling.](http://arxiv.org/abs/2309.13218) | 这篇论文提出了一个AI-企业优化的协同辅助系统，通过采用大型语言模型和微调预训练模型的方法，实现了减少人类专业知识需求的目标。 |

# 详细

[^1]: 利用迁移学习促进过程控制的强化学习：观点

    Facilitating Reinforcement Learning for Process Control Using Transfer Learning: Perspectives

    [https://arxiv.org/abs/2404.00247](https://arxiv.org/abs/2404.00247)

    本文从迁移学习的角度探讨了如何将其与强化学习相结合，为过程控制带来新的可能性。

    

    本文从迁移学习的角度，为过程控制中的深度强化学习（DRL）提供了深入见解。我们分析了在过程工业领域应用DRL所面临的挑战，以及引入迁移学习的必要性。此外，我们为未来研究方向提供了建议和展望，探讨了如何将迁移学习与DRL结合起来加强过程控制。

    arXiv:2404.00247v1 Announce Type: cross  Abstract: This paper provides insights into deep reinforcement learning (DRL) for process control from the perspective of transfer learning. We analyze the challenges of applying DRL in the field of process industries and the necessity of introducing transfer learning. Furthermore, recommendations and prospects are provided for future research directions on how transfer learning can be integrated with DRL to empower process control.
    
[^2]: 跨领域的纤维簇形状分析用于语言表现认知分数预测

    Cross--domain Fiber Cluster Shape Analysis for Language Performance Cognitive Score Prediction

    [https://arxiv.org/abs/2403.19001](https://arxiv.org/abs/2403.19001)

    本研究通过新颖的框架SFFormer，结合了多头交叉注意力特征融合模块，基于dMRI纤维束追踪，预测了主观语言表现，拓展了脑结构与人类认知功能的关联研究。

    

    形状在计算机图形学中扮演重要角色，提供了有关对象形态和功能的信息特征。脑成像中的形状分析可帮助解释人脑结构和功能的相关性。本研究调查了大脑的3D白质连接的形状及其与人类认知功能的潜在预测关系。我们使用扩散磁共振成像（dMRI）纤维束追踪将大脑连接重建为3D点序列。为了描述每个连接，我们提取了12个形状描述符以及传统的dMRI连接和组织微结构特征。我们引入了一种新颖的框架，形状融合纤维簇变换器（SFFormer），利用多头交叉注意力特征融合模块基于dMRI纤维束追踪来预测特定个体的语言表现。我们在一个大型数据集上评估了该方法的性能。

    arXiv:2403.19001v1 Announce Type: cross  Abstract: Shape plays an important role in computer graphics, offering informative features to convey an object's morphology and functionality. Shape analysis in brain imaging can help interpret structural and functionality correlations of the human brain. In this work, we investigate the shape of the brain's 3D white matter connections and its potential predictive relationship to human cognitive function. We reconstruct brain connections as sequences of 3D points using diffusion magnetic resonance imaging (dMRI) tractography. To describe each connection, we extract 12 shape descriptors in addition to traditional dMRI connectivity and tissue microstructure features. We introduce a novel framework, Shape--fused Fiber Cluster Transformer (SFFormer), that leverages a multi-head cross-attention feature fusion module to predict subject-specific language performance based on dMRI tractography. We assess the performance of the method on a large dataset
    
[^3]: 使用环境扩散后验采样：在受损数据上训练的扩散模型解决逆问题

    Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models trained on Corrupted Data

    [https://arxiv.org/abs/2403.08728](https://arxiv.org/abs/2403.08728)

    提出了一种使用环境扩散后验采样解决逆问题的框架，能在受损数据上训练的扩散模型上表现出色，并在图像恢复和MRI模型训练中取得优越性能。

    

    我们提供了一个框架，用于使用从线性受损数据中学习的扩散模型解决逆问题。我们的方法，Ambient Diffusion Posterior Sampling (A-DPS)，利用一个预先在一种类型的损坏数据上进行过训练的生成模型，以在可能来自不同前向过程（例如图像模糊）的测量条件下执行后验采样。我们在标准自然图像数据集（CelebA、FFHQ 和 AFHQ）上测试了我们的方法的有效性，并展示了 A-DPS 有时在速度和性能上都能胜过在清洁数据上训练的模型，用于几个图像恢复任务。我们进一步扩展了环境扩散框架，以仅访问傅里叶子采样的多线圈 MRI 测量数据来训练 MRI 模型，其加速因子为不同的加速因子（R=2、4、6、8）。我们再次观察到，在高度子采样数据上训练的模型更适用于解决高加速 MRI 逆问题。

    arXiv:2403.08728v1 Announce Type: cross  Abstract: We provide a framework for solving inverse problems with diffusion models learned from linearly corrupted data. Our method, Ambient Diffusion Posterior Sampling (A-DPS), leverages a generative model pre-trained on one type of corruption (e.g. image inpainting) to perform posterior sampling conditioned on measurements from a potentially different forward process (e.g. image blurring). We test the efficacy of our approach on standard natural image datasets (CelebA, FFHQ, and AFHQ) and we show that A-DPS can sometimes outperform models trained on clean data for several image restoration tasks in both speed and performance. We further extend the Ambient Diffusion framework to train MRI models with access only to Fourier subsampled multi-coil MRI measurements at various acceleration factors (R=2, 4, 6, 8). We again observe that models trained on highly subsampled data are better priors for solving inverse problems in the high acceleration r
    
[^4]: QuaCer-C：大型语言模型中知识理解的定量认证

    QuaCer-C: Quantitative Certification of Knowledge Comprehension in LLMs

    [https://arxiv.org/abs/2402.15929](https://arxiv.org/abs/2402.15929)

    本文提出了一种新颖的认证框架QuaCer-C，用于正式认证大型语言模型中知识理解的能力，证书定量化且包含高置信度的概率界限，研究发现，随着参数数量的增加，知识理解能力提高，Mistral模型在这一评估中表现不如其他模型。

    

    大型语言模型（LLMs）在多个基准测试中展现出令人印象深刻的表现。然而，传统研究并未对LLMs的表现提供正式的保证。本文提出了一种新颖的LLM认证框架QuaCer-C，我们在此对知名LLMs的知识理解能力进行正式认证。我们的证书是定量的 - 它们包括对目标LLM在任何相关知识理解提示上给出正确答案的概率的高置信度紧密界限。我们针对Llama、Vicuna和Mistral LLMs的证书表明，知识理解能力随参数数量的增加而提高，并且Mistral模型在这一评估中表现不如其他模型。

    arXiv:2402.15929v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated impressive performance on several benchmarks. However, traditional studies do not provide formal guarantees on the performance of LLMs. In this work, we propose a novel certification framework for LLM, QuaCer-C, wherein we formally certify the knowledge-comprehension capabilities of popular LLMs. Our certificates are quantitative - they consist of high-confidence, tight bounds on the probability that the target LLM gives the correct answer on any relevant knowledge comprehension prompt. Our certificates for the Llama, Vicuna, and Mistral LLMs indicate that the knowledge comprehension capability improves with an increase in the number of parameters and that the Mistral model is less performant than the rest in this evaluation.
    
[^5]: WikiMT++数据集卡片

    WikiMT++ Dataset Card. (arXiv:2309.13259v1 [cs.IR])

    [http://arxiv.org/abs/2309.13259](http://arxiv.org/abs/2309.13259)

    WikiMT++是一个扩展和精细版本的WikiMusicText数据集，包含了1010个经过策划的ABC记谱法的主题曲。它添加了客观属性和主观情感属性，增强了数据集的应用场景和可用性，并通过CLaMP来纠正属性，提高准确性和完整性。

    

    WikiMT++是WikiMusicText（WikiMT）的扩展和精细版本，包含了1010个经过策划的ABC记谱法的主题曲。为了扩展WikiMT的应用场景，我们添加了客观属性（专辑、歌词、视频）和主观情感属性（12个情感形容词）和情感4Q（Russell 4Q），增强了其在音乐信息检索、条件音乐生成、自动作曲和情感分类等方面的可用性。此外，我们还实现了CLaMP来纠正从WikiMT继承的属性，以减少原始数据收集过程中引入的错误，增强了数据集的准确性和完整性。

    WikiMT++ is an expanded and refined version of WikiMusicText (WikiMT), featuring 1010 curated lead sheets in ABC notation. To expand application scenarios of WikiMT, we add both objective (album, lyrics, video) and subjective emotion (12 emotion adjectives) and emo\_4q (Russell 4Q) attributes, enhancing its usability for music information retrieval, conditional music generation, automatic composition, and emotion classification, etc. Additionally, CLaMP is implemented to correct the attributes inherited from WikiMT to reduce errors introduced during original data collection and enhance the accuracy and completeness of our dataset.
    
[^6]: AI-企业优化的协同辅助：一个框架和在生产调度中的案例研究。

    AI-Copilot for Business Optimisation: A Framework and A Case Study in Production Scheduling. (arXiv:2309.13218v1 [cs.AI])

    [http://arxiv.org/abs/2309.13218](http://arxiv.org/abs/2309.13218)

    这篇论文提出了一个AI-企业优化的协同辅助系统，通过采用大型语言模型和微调预训练模型的方法，实现了减少人类专业知识需求的目标。

    

    企业优化是寻找和实施高效和具有成本效益的运营方式，以为企业带来竞争优势的过程。综合问题表述是企业优化的一个重要组成部分，它围绕着人类专业知识展开，因此很有可能成为瓶颈。随着大型语言模型（LLMs）的最新进展，通过人工智能（AI）可以潜在地减少问题表述中所需的人类专业知识。然而，开发用于问题表述的LLM具有挑战性，由于训练数据要求、令牌限制以及LLM中缺乏适当的性能度量。为了减少大量训练数据的需求，最近人们开始关注对预训练的LLM进行微调以适应下游任务，而不是从头开始训练一个特定任务的LLM。在本文中，我们采用了这种方法，提出了一个AI-企业优化的协同辅助系统。

    Business optimisation is the process of finding and implementing efficient and cost-effective means of operation to bring a competitive advantage for businesses. Synthesizing problem formulations is an integral part of business optimisation which is centred around human expertise, thus with a high potential of becoming a bottleneck. With the recent advancements in Large Language Models (LLMs), human expertise needed in problem formulation can potentially be minimized using Artificial Intelligence (AI). However, developing a LLM for problem formulation is challenging, due to training data requirements, token limitations, and the lack of appropriate performance metrics in LLMs. To minimize the requirement of large training data, considerable attention has recently been directed towards fine-tuning pre-trained LLMs for downstream tasks, rather than training a LLM from scratch for a specific task. In this paper, we adopt this approach and propose an AI-Copilot for business optimisation by 
    

