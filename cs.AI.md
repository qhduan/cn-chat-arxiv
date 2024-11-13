# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Wait, It's All Token Noise? Always Has Been: Interpreting LLM Behavior Using Shapley Value](https://arxiv.org/abs/2404.01332) | 使用Shapley值方法解释LLM行为，揭示了所谓的“令牌噪音”效应，揭示了LLMs的决策在很大程度上受到提示组件的影响 |
| [^2] | [Securing GNNs: Explanation-Based Identification of Backdoored Training Graphs](https://arxiv.org/abs/2403.18136) | 提出了一种基于解释的方法来识别GNN中的后门训练图，设计了七种新的度量指标以更有效地检测后门攻击，并且通过自适应攻击进行了方法评估。 |
| [^3] | [LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection](https://arxiv.org/abs/2403.17465) | LaRE^2 提出了一种基于潜在重构误差的方法用于检测扩散生成的图像，通过引入潜在重构误差（LaRE）和误差引导特征细化模块（EGRE）实现了对特征的有效提取和增强，从而区分真实和生成图像。 |
| [^4] | [Smooth Sensitivity for Learning Differentially-Private yet Accurate Rule Lists](https://arxiv.org/abs/2403.13848) | 通过建立Gini不纯度的平滑敏感度并将其应用于提出DP贪婪规则列表算法，本文改善了差异保护模型的准确性问题。 |
| [^5] | [DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers](https://arxiv.org/abs/2402.16914) | 将恶意提示分解为独立的子提示使得LLM越狱攻击更难被检测 |
| [^6] | [Dynamic planning in hierarchical active inference](https://arxiv.org/abs/2402.11658) | 通过研究在动态规划领域中模拟工具使用的目标，我们深入探讨了主动推断中的动态规划，该领域考虑到生物目标导向行为的两个关键方面 |
| [^7] | [Into the Unknown: Self-Learning Large Language Models](https://arxiv.org/abs/2402.09147) | 本研究关注自学习大型语言模型的核心问题：如何学习未知知识。提出了一种自学习框架，通过自我评估和识别未知点来独立学习以前未知的知识。实验证明该方法对于减少幻觉评分、实现高效LLM更新以及知识交流具有重要意义。 |
| [^8] | [Explainable Identification of Hate Speech towards Islam using Graph Neural Networks](https://arxiv.org/abs/2311.04916) | 使用图神经网络解释和识别伊斯兰教仇恨言论，模型在保持出色性能的同时能够解释相关性和因果关系。 |
| [^9] | [Multi-Agent Dynamic Relational Reasoning for Social Robot Navigation.](http://arxiv.org/abs/2401.12275) | 本文提出了一种多Agent动态关系推理方法，通过明确推断关系结构的演化，来实现在社交机器人导航中的有效性。方法包括推断超边缘以实现群体推理和轨迹预测器生成未来状态。 |
| [^10] | [CFASL: Composite Factor-Aligned Symmetry Learning for Disentanglement in Variational AutoEncoder.](http://arxiv.org/abs/2401.08897) | CFASL是一种用于解缠学习的新方法，它将对称性学习与VAE集成，无需任何数据集因子信息的先验知识，具有三个新特征：对齐潜在向量维度到可学习对称代码簿中的对称性，学习复合对称性来表达未知因素的变化，以及引入群等变编码器和解码器来训练VAE。 |
| [^11] | [When Geoscience Meets Foundation Models: Towards General Geoscience Artificial Intelligence System.](http://arxiv.org/abs/2309.06799) | 地球科学基础模型通过整合大量跨学科数据来模拟和理解地球系统动态，具有广阔的应用前景和创新潜力，但仍面临验证和核实、规模性、可解释性、知识表示和社会偏差等挑战。 |
| [^12] | [A Deep Recurrent-Reinforcement Learning Method for Intelligent AutoScaling of Serverless Functions.](http://arxiv.org/abs/2308.05937) | 该论文介绍了一种用于智能自动缩放无服务器函数的深度循环强化学习方法，针对波动的工作负载和严格的性能约束，通过建立一个适应性策略来实现最大化期望目标。 |
| [^13] | [Levin Tree Search with Context Models.](http://arxiv.org/abs/2305.16945) | 本文提出了一种新的具有上下文模型的Levin树搜索算法，通过将神经网络替换为上下文模型，实现了LTS损失的凸优化，并在多个基准测试中取得了明显优于LTS+NN的结果。 |
| [^14] | [Explicit and Implicit Semantic Ranking Framework.](http://arxiv.org/abs/2304.04918) | 本文提出了一个名为sRank的通用语义学习排名框架，它使用transformer模型，能够在智能回复和环境临床智能等真实应用中，实现11.7%的离线准确度提升。 |

# 详细

[^1]: 等等，这都是令牌噪音？一直就是吗：利用 Shapley 值解释 LLM 行为

    Wait, It's All Token Noise? Always Has Been: Interpreting LLM Behavior Using Shapley Value

    [https://arxiv.org/abs/2404.01332](https://arxiv.org/abs/2404.01332)

    使用Shapley值方法解释LLM行为，揭示了所谓的“令牌噪音”效应，揭示了LLMs的决策在很大程度上受到提示组件的影响

    

    大型语言模型（LLMs）的出现为模拟人类行为和认知过程开辟了新的可能性，潜在应用包括市场研究和消费者行为分析等各个领域。然而，由于LLMs的显著差异暗示了不同的基础过程在起作用，以及LLMs对提示变化的敏感性，利用LLMs作为人类主体的替代仍然存在不确定性。本文提出了一种基于合作博弈理论中Shapley值的新方法来解释LLM行为，并量化每个提示组件对模型输出的相对贡献。通过两个应用--一个离散选择实验和一个认知偏见调查，我们展示了Shapley值方法如何揭示我们所谓的“令牌噪音”效应，即LLM决策受到的影响严重偏向于

    arXiv:2404.01332v1 Announce Type: cross  Abstract: The emergence of large language models (LLMs) has opened up exciting possibilities for simulating human behavior and cognitive processes, with potential applications in various domains, including marketing research and consumer behavior analysis. However, the validity of utilizing LLMs as stand-ins for human subjects remains uncertain due to glaring divergences that suggest fundamentally different underlying processes at play and the sensitivity of LLM responses to prompt variations. This paper presents a novel approach based on Shapley values from cooperative game theory to interpret LLM behavior and quantify the relative contribution of each prompt component to the model's output. Through two applications-a discrete choice experiment and an investigation of cognitive biases-we demonstrate how the Shapley value method can uncover what we term "token noise" effects, a phenomenon where LLM decisions are disproportionately influenced by 
    
[^2]: 保护GNN：基于解释的后门训练图识别

    Securing GNNs: Explanation-Based Identification of Backdoored Training Graphs

    [https://arxiv.org/abs/2403.18136](https://arxiv.org/abs/2403.18136)

    提出了一种基于解释的方法来识别GNN中的后门训练图，设计了七种新的度量指标以更有效地检测后门攻击，并且通过自适应攻击进行了方法评估。

    

    Graph Neural Networks (GNNs)已经在许多领域流行起来，但它们容易受到后门攻击，这可能会损害它们的性能和道德应用。检测这些攻击对于保持GNN分类任务的可靠性和安全性至关重要，但有效的检测技术并不多见。我们观察到，尽管图级解释能够提供一些有限的见解，但它们在检测后门触发器方面的有效性是不一致且不完整的。为弥补这一差距，我们提取并转换GNN解释机制的次要输出，设计了七种更有效地检测后门攻击的新度量。此外，我们还开发了一种自适应攻击来严格评估我们的方法。我们在多个基准数据集上测试了我们的方法，并检查其对各种攻击模型的有效性。我们的结果表明，我们的方法可以取得较高的效果。

    arXiv:2403.18136v1 Announce Type: cross  Abstract: Graph Neural Networks (GNNs) have gained popularity in numerous domains, yet they are vulnerable to backdoor attacks that can compromise their performance and ethical application. The detection of these attacks is crucial for maintaining the reliability and security of GNN classification tasks, but effective detection techniques are lacking. Following an initial investigation, we observed that while graph-level explanations can offer limited insights, their effectiveness in detecting backdoor triggers is inconsistent and incomplete. To bridge this gap, we extract and transform secondary outputs of GNN explanation mechanisms, designing seven novel metrics that more effectively detect backdoor attacks. Additionally, we develop an adaptive attack to rigorously evaluate our approach. We test our method on multiple benchmark datasets and examine its efficacy against various attack models. Our results show that our method can achieve high de
    
[^3]: LaRE^2: 基于潜在重构误差的扩散生成图像检测方法

    LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection

    [https://arxiv.org/abs/2403.17465](https://arxiv.org/abs/2403.17465)

    LaRE^2 提出了一种基于潜在重构误差的方法用于检测扩散生成的图像，通过引入潜在重构误差（LaRE）和误差引导特征细化模块（EGRE）实现了对特征的有效提取和增强，从而区分真实和生成图像。

    

    arXiv:2403.17465v1 类型：交叉 摘要：扩散模型的发展显著提高了图像生成质量，使真实图像和生成图像之间的区分变得越来越困难。尽管这一进展令人印象深刻，但也引发了重要的隐私和安全问题。为了解决这一问题，我们提出了一种新颖的基于潜在重构误差引导特征细化方法（LaRE^2）来检测扩散生成的图像。我们提出了潜在重构误差（LaRE），作为潜在空间中生成图像检测的第一个基于重构误差的特征。LaRE在特征提取效率方面超越了现有方法，同时保留了区分真假所需的关键线索。为了利用LaRE，我们提出了一种误差引导特征细化模块（EGRE），它可以通过LaRE引导的方式细化图像特征，以增强特征的区分能力。

    arXiv:2403.17465v1 Announce Type: cross  Abstract: The evolution of Diffusion Models has dramatically improved image generation quality, making it increasingly difficult to differentiate between real and generated images. This development, while impressive, also raises significant privacy and security concerns. In response to this, we propose a novel Latent REconstruction error guided feature REfinement method (LaRE^2) for detecting the diffusion-generated images. We come up with the Latent Reconstruction Error (LaRE), the first reconstruction-error based feature in the latent space for generated image detection. LaRE surpasses existing methods in terms of feature extraction efficiency while preserving crucial cues required to differentiate between the real and the fake. To exploit LaRE, we propose an Error-Guided feature REfinement module (EGRE), which can refine the image feature guided by LaRE to enhance the discriminativeness of the feature. Our EGRE utilizes an align-then-refine m
    
[^4]: 用于学习差异保护但准确规则列表的平滑敏感度

    Smooth Sensitivity for Learning Differentially-Private yet Accurate Rule Lists

    [https://arxiv.org/abs/2403.13848](https://arxiv.org/abs/2403.13848)

    通过建立Gini不纯度的平滑敏感度并将其应用于提出DP贪婪规则列表算法，本文改善了差异保护模型的准确性问题。

    

    差异保护（DP）机制可以嵌入到机器学习算法的设计中，以保护所得模型免受隐私泄露的影响，尽管这通常伴随着明显的准确性损失。本文旨在通过建立Gini不纯度的平滑敏感度并利用这一特性来提出一个DP贪婪规则列表算法，以改善这种权衡。我们的理论分析和实验结果表明，集成平滑敏感度的DP规则列表模型具有比使用全局敏感度的其他DP框架更高的准确性。

    arXiv:2403.13848v1 Announce Type: cross  Abstract: Differentially-private (DP) mechanisms can be embedded into the design of a machine learningalgorithm to protect the resulting model against privacy leakage, although this often comes with asignificant loss of accuracy. In this paper, we aim at improving this trade-off for rule lists modelsby establishing the smooth sensitivity of the Gini impurity and leveraging it to propose a DP greedyrule list algorithm. In particular, our theoretical analysis and experimental results demonstrate thatthe DP rule lists models integrating smooth sensitivity have higher accuracy that those using otherDP frameworks based on global sensitivity.
    
[^5]: DrAttack: 提示分解和重构使强大的LLM越狱者

    DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers

    [https://arxiv.org/abs/2402.16914](https://arxiv.org/abs/2402.16914)

    将恶意提示分解为独立的子提示使得LLM越狱攻击更难被检测

    

    本文发现将恶意提示分解为独立的子提示能够有效模糊其潜在的恶意意图，使之以片段化、不易检测的形式呈现，从而解决了这些局限性。我们引入了一个用于越狱攻击的自动提示分解和重构框架（DrAttack）。DrAttack包括三个关键组件：(a) 将原始提示进行“分解”为子提示，(b) 通过上下文学习中的语义上相似但隐含的“重构”这些子提示

    arXiv:2402.16914v1 Announce Type: cross  Abstract: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but h
    
[^6]: 分层主动推断中的动态规划

    Dynamic planning in hierarchical active inference

    [https://arxiv.org/abs/2402.11658](https://arxiv.org/abs/2402.11658)

    通过研究在动态规划领域中模拟工具使用的目标，我们深入探讨了主动推断中的动态规划，该领域考虑到生物目标导向行为的两个关键方面

    

    通过动态规划，我们指的是人类大脑推断和施加与认知决策相关的运动轨迹的能力。最近的一个范式，主动推断，为生物有机体适应带来了基本见解，不断努力最小化预测误差以将自己限制在与生命兼容的状态。在过去的几年里，许多研究表明人类和动物行为可以解释为主动推断过程，无论是作为离散决策还是连续运动控制，都激发了机器人技术和人工智能中的创新解决方案。然而，文献缺乏对如何有效地在变化环境中规划行动的全面展望。我们设定了对工具使用进行建模的目标，深入研究了主动推断中的动态规划主题，牢记两个生物目标导向行为的关键方面：理解……

    arXiv:2402.11658v1 Announce Type: new  Abstract: By dynamic planning, we refer to the ability of the human brain to infer and impose motor trajectories related to cognitive decisions. A recent paradigm, active inference, brings fundamental insights into the adaptation of biological organisms, constantly striving to minimize prediction errors to restrict themselves to life-compatible states. Over the past years, many studies have shown how human and animal behavior could be explained in terms of an active inferential process -- either as discrete decision-making or continuous motor control -- inspiring innovative solutions in robotics and artificial intelligence. Still, the literature lacks a comprehensive outlook on how to effectively plan actions in changing environments. Setting ourselves the goal of modeling tool use, we delve into the topic of dynamic planning in active inference, keeping in mind two crucial aspects of biological goal-directed behavior: the capacity to understand a
    
[^7]: 未知之中：自学习大型语言模型

    Into the Unknown: Self-Learning Large Language Models

    [https://arxiv.org/abs/2402.09147](https://arxiv.org/abs/2402.09147)

    本研究关注自学习大型语言模型的核心问题：如何学习未知知识。提出了一种自学习框架，通过自我评估和识别未知点来独立学习以前未知的知识。实验证明该方法对于减少幻觉评分、实现高效LLM更新以及知识交流具有重要意义。

    

    我们解决了自学习大型语言模型（LLM）的主要问题：即如何学习自己不知道的知识。我们提出了一种自学习LLM框架，通过对自己的幻觉进行自我评估，使LLM能够独立地学习以前未知的知识。通过使用幻觉评分，我们引入了一个称为“未知点”的新概念，并提出了一种外部和三种内部方法来自动识别未知点。这有助于创建一个自学习循环，专注于未知点中的知识差距，从而减少幻觉评分。我们还开发了用于评估LLM自学习能力的评估指标。我们的实验证明，已经进行了微调或对齐的7B-Mistral模型在自学习方面表现出色。我们的自学习概念可以实现更高效的LLM更新，并为知识交流开辟新的可能性。它还可能增加公众的信任。

    arXiv:2402.09147v1 Announce Type: new Abstract: We address the main problem of self-learning LLM: the question of what to learn. We propose a self-learning LLM framework that enables an LLM to independently learn previously unknown knowledge through self-assessment of their own hallucinations. Using the hallucination score, we introduce a new concept of Points in The Unknown (PiUs), along with one extrinsic and three intrinsic methods for automatic PiUs identification. It facilitates the creation of a self-learning loop that focuses exclusively on the knowledge gap in Points in The Unknown, resulting in a reduced hallucination score. We also developed evaluation metrics for gauging an LLM's self-learning capability. Our experiments revealed that 7B-Mistral models that have been finetuned or aligned are capable of self-learning considerably well. Our self-learning concept allows more efficient LLM updates and opens new perspectives for knowledge exchange. It may also increase public tru
    
[^8]: 使用图神经网络解释伊斯兰教仇恨言论的研究

    Explainable Identification of Hate Speech towards Islam using Graph Neural Networks

    [https://arxiv.org/abs/2311.04916](https://arxiv.org/abs/2311.04916)

    使用图神经网络解释和识别伊斯兰教仇恨言论，模型在保持出色性能的同时能够解释相关性和因果关系。

    

    伊斯兰教仇恨言论在在线社交互动平台上是一个普遍存在的挑战。识别和消除这种仇恨是迈向和谐与和平未来的关键一步。本研究提出了一种新的范例，利用图神经网络来识别和解释针对伊斯兰教的仇恨言论。利用图神经网络发现、提取并利用不同数据点之间的关系的内在能力，我们的模型始终能够在保持出色性能的同时提供对潜在相关性和因果关系的解释。

    arXiv:2311.04916v2 Announce Type: cross  Abstract: Islamophobic language is a prevalent challenge on online social interaction platforms. Identifying and eliminating such hatred is a crucial step towards a future of harmony and peace. This study presents a novel paradigm for identifying and explaining hate speech towards Islam using graph neural networks. Utilizing the intrinsic ability of graph neural networks to find, extract, and use relationships across disparate data points, our model consistently achieves outstanding performance while offering explanations for the underlying correlations and causation.
    
[^9]: 多Agent动态关系推理用于社交机器人导航

    Multi-Agent Dynamic Relational Reasoning for Social Robot Navigation. (arXiv:2401.12275v1 [cs.RO])

    [http://arxiv.org/abs/2401.12275](http://arxiv.org/abs/2401.12275)

    本文提出了一种多Agent动态关系推理方法，通过明确推断关系结构的演化，来实现在社交机器人导航中的有效性。方法包括推断超边缘以实现群体推理和轨迹预测器生成未来状态。

    

    社交机器人导航在日常生活的各种情景下可以提供帮助，但需要安全的人机交互和高效的轨迹规划。在多Agent交互系统中，建模成对的关系已经被广泛研究，但是捕捉更大规模的群体活动的能力有限。在本文中，我们提出了一种系统的关系推理方法，通过明确推断正在演变的关系结构，展示了其在多Agent轨迹预测和社交机器人导航中的有效性。除了节点对之间的边缘（即Agent），我们还提出了推断超边缘的方法，以自适应地连接多个节点，以便进行群体推理。我们的方法推断动态演化的关系图和超图，以捕捉关系的演化，轨迹预测器利用这些图来生成未来状态。同时，我们提出了对锐度和逻辑稀疏性进行正则化的方法。

    Social robot navigation can be helpful in various contexts of daily life but requires safe human-robot interactions and efficient trajectory planning. While modeling pairwise relations has been widely studied in multi-agent interacting systems, the ability to capture larger-scale group-wise activities is limited. In this paper, we propose a systematic relational reasoning approach with explicit inference of the underlying dynamically evolving relational structures, and we demonstrate its effectiveness for multi-agent trajectory prediction and social robot navigation. In addition to the edges between pairs of nodes (i.e., agents), we propose to infer hyperedges that adaptively connect multiple nodes to enable group-wise reasoning in an unsupervised manner. Our approach infers dynamically evolving relation graphs and hypergraphs to capture the evolution of relations, which the trajectory predictor employs to generate future states. Meanwhile, we propose to regularize the sharpness and sp
    
[^10]: CFASL：用于变分自编码器中的解缠学习的复合因子对齐对称学习

    CFASL: Composite Factor-Aligned Symmetry Learning for Disentanglement in Variational AutoEncoder. (arXiv:2401.08897v1 [cs.LG])

    [http://arxiv.org/abs/2401.08897](http://arxiv.org/abs/2401.08897)

    CFASL是一种用于解缠学习的新方法，它将对称性学习与VAE集成，无需任何数据集因子信息的先验知识，具有三个新特征：对齐潜在向量维度到可学习对称代码簿中的对称性，学习复合对称性来表达未知因素的变化，以及引入群等变编码器和解码器来训练VAE。

    

    输入和潜在向量的对称性为VAE中的解缠学习提供了宝贵的见解。然而，只有少数几篇论文提出了一种无监督方法，甚至这些方法在训练数据中也需要已知的因子信息。我们提出了一种新的方法，Composite Factor-Aligned Symmetry Learning (CFASL)，将其集成到VAE中，用于学习基于对称性的解缠，无监督学习中不需要任何数据集因子信息的知识。CFASL包括三个用于学习基于对称性的解缠的新特征：1)注入归纳偏置，将潜在向量维度对齐到明确可学习的对称代码簿中的因子对齐对称性；2)学习一个复合对称性，通过学习代码簿中的因子对齐对称性，来表达两个随机样本之间的未知因素的变化；3)在训练VAE时，引入具有群等变编码器和解码器的两个条件。此外，我们提出了一种扩展的评估指标。

    Symmetries of input and latent vectors have provided valuable insights for disentanglement learning in VAEs.However, only a few works were proposed as an unsupervised method, and even these works require known factor information in training data. We propose a novel method, Composite Factor-Aligned Symmetry Learning (CFASL), which is integrated into VAEs for learning symmetry-based disentanglement in unsupervised learning without any knowledge of the dataset factor information.CFASL incorporates three novel features for learning symmetry-based disentanglement: 1) Injecting inductive bias to align latent vector dimensions to factor-aligned symmetries within an explicit learnable symmetry codebook 2) Learning a composite symmetry to express unknown factors change between two random samples by learning factor-aligned symmetries within the codebook 3) Inducing group equivariant encoder and decoder in training VAEs with the two conditions. In addition, we propose an extended evaluation metri
    
[^11]: 当地球科学遇见基础模型：走向通用地球科学人工智能系统

    When Geoscience Meets Foundation Models: Towards General Geoscience Artificial Intelligence System. (arXiv:2309.06799v1 [cs.AI])

    [http://arxiv.org/abs/2309.06799](http://arxiv.org/abs/2309.06799)

    地球科学基础模型通过整合大量跨学科数据来模拟和理解地球系统动态，具有广阔的应用前景和创新潜力，但仍面临验证和核实、规模性、可解释性、知识表示和社会偏差等挑战。

    

    地球科学基础模型通过整合大量跨学科数据来模拟和理解地球系统动态，代表了地球科学领域的一种革命性方法。作为一种数据中心的人工智能范式，它们从百万亿字节的结构化和非结构化数据中揭示出洞察力。灵活的任务规范、多样化的输入和输出以及多模态的知识表示使得综合分析成为可能。至关重要的是，地球科学模型的可扩展性和可推广性允许解决与地球系统相互作用相关的多种预测、模拟和决策挑战。领域专家和计算机科学家之间的合作推动了这些宝贵工具在理解我们地球的过去、现在和未来方面的创新。然而，验证和核实、规模性、可解释性、知识表示和社会偏差仍然面临挑战。展望未来，增强验证和核实、规模性、解释性、知识表示和社会偏差方面的能力，将有助于推动地球科学人工智能系统的发展。

    Geoscience foundation models represent a revolutionary approach in the field of Earth sciences by integrating massive cross-disciplinary data to simulate and understand the Earth systems dynamics. As a data-centric artificial intelligence (AI) paradigm, they uncover insights from petabytes of structured and unstructured data. Flexible task specification, diverse inputs and outputs and multi-modal knowledge representation enable comprehensive analysis infeasible with individual data sources. Critically, the scalability and generalizability of geoscience models allow for tackling diverse prediction, simulation, and decision challenges related to Earth systems interactions. Collaboration between domain experts and computer scientists leads to innovations in these invaluable tools for understanding the past, present, and future of our planet. However, challenges remain in validation and verification, scale, interpretability, knowledge representation, and social bias. Going forward, enhanci
    
[^12]: 一种用于智能自动缩放无服务器函数的深度循环强化学习方法

    A Deep Recurrent-Reinforcement Learning Method for Intelligent AutoScaling of Serverless Functions. (arXiv:2308.05937v1 [cs.DC])

    [http://arxiv.org/abs/2308.05937](http://arxiv.org/abs/2308.05937)

    该论文介绍了一种用于智能自动缩放无服务器函数的深度循环强化学习方法，针对波动的工作负载和严格的性能约束，通过建立一个适应性策略来实现最大化期望目标。

    

    函数即服务（FaaS）引入了一种轻量级的基于函数的云执行模型，在物联网边缘数据处理和异常检测等应用中具有相关性。虽然云服务提供商提供了几乎无限的函数弹性，但这些应用经常遇到波动的工作负载和更严格的性能约束。典型的云服务提供商策略是根据基于监控的阈值（如CPU或内存）来经验性地确定和调整所需的函数实例以适应需求和性能，即"自动缩放"。然而，阈值配置要么需要专家知识，要么需要历史数据或对环境的完整视图，使得自动缩放成为缺乏适应性解决方案的性能瓶颈。强化学习算法已被证明在分析复杂的云环境中是有益的，并产生适应性策略以最大化期望目标。

    Function-as-a-Service (FaaS) introduces a lightweight, function-based cloud execution model that finds its relevance in applications like IoT-edge data processing and anomaly detection. While CSP offer a near-infinite function elasticity, these applications often experience fluctuating workloads and stricter performance constraints. A typical CSP strategy is to empirically determine and adjust desired function instances, "autoscaling", based on monitoring-based thresholds such as CPU or memory, to cope with demand and performance. However, threshold configuration either requires expert knowledge, historical data or a complete view of environment, making autoscaling a performance bottleneck lacking an adaptable solution.RL algorithms are proven to be beneficial in analysing complex cloud environments and result in an adaptable policy that maximizes the expected objectives. Most realistic cloud environments usually involve operational interference and have limited visibility, making them
    
[^13]: 具有上下文模型的Levin树搜索

    Levin Tree Search with Context Models. (arXiv:2305.16945v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.16945](http://arxiv.org/abs/2305.16945)

    本文提出了一种新的具有上下文模型的Levin树搜索算法，通过将神经网络替换为上下文模型，实现了LTS损失的凸优化，并在多个基准测试中取得了明显优于LTS+NN的结果。

    

    Levin Tree Search（LTS）是一种利用策略（动作的概率分布）的搜索算法，并具有关于达到目标节点之前扩展次数的理论保证，这取决于策略的质量。我们将这个保证称为LTS损失，可以将其作为优化表示策略的神经网络（LTS+NN）的损失函数。在这项工作中，我们展示了神经网络可以用在线压缩文献中的参数化上下文模型来替代（LTS+CM）。我们证明了在这种新模型下LTS损失是凸的，从而可以使用标准凸优化工具，并且对于给定的解轨迹集合，在在线设置中可以获得到最优参数的收敛保证——而神经网络无法提供这样的保证。新的LTS+CM算法在几个基准测试中与LTS+NN相比表现出明显优势：Sokoban（Boxoban）、The Witness和24-Sliding Tile Puzzle（STP）。

    Levin Tree Search (LTS) is a search algorithm that makes use of a policy (a probability distribution over actions) and comes with a theoretical guarantee on the number of expansions before reaching a goal node, depending on the quality of the policy. This guarantee can be used as a loss function, which we call the LTS loss, to optimize neural networks representing the policy (LTS+NN). In this work we show that the neural network can be substituted with parameterized context models originating from the online compression literature (LTS+CM). We show that the LTS loss is convex under this new model, which allows for using standard convex optimization tools, and obtain convergence guarantees to the optimal parameters in an online setting for a given set of solution trajectories -- guarantees that cannot be provided for neural networks. The new LTS+CM algorithm compares favorably against LTS+NN on several benchmarks: Sokoban (Boxoban), The Witness, and the 24-Sliding Tile puzzle (STP). The
    
[^14]: 显式和隐式语义排序框架

    Explicit and Implicit Semantic Ranking Framework. (arXiv:2304.04918v1 [cs.IR])

    [http://arxiv.org/abs/2304.04918](http://arxiv.org/abs/2304.04918)

    本文提出了一个名为sRank的通用语义学习排名框架，它使用transformer模型，能够在智能回复和环境临床智能等真实应用中，实现11.7%的离线准确度提升。

    

    在许多实际应用中，核心难题是将一个查询与一个可变且有限的文档集中的最佳文档进行匹配。现有的工业解决方案，特别是延迟受限的服务，通常依赖于相似性算法，这些算法为了速度而牺牲了质量。本文介绍了一个通用的语义学习排名框架，自我训练语义交叉关注排名（sRank）。这个基于transformer的框架使用线性成对损失，具有可变的训练批量大小、实现质量提升和高效率，并已成功应用于微软公司的两个工业任务：智能回复（SR）和环境临床智能（ACI）的真实大规模数据集上。在智能回复中，$sRank$通过基于消费者和支持代理信息的预定义解决方案选择最佳答案，帮助用户实时获得技术支持。在SR任务上，$sRank$实现了11.7%的离线top-one准确度提升，比之前的系统更加优秀。

    The core challenge in numerous real-world applications is to match an inquiry to the best document from a mutable and finite set of candidates. Existing industry solutions, especially latency-constrained services, often rely on similarity algorithms that sacrifice quality for speed. In this paper we introduce a generic semantic learning-to-rank framework, Self-training Semantic Cross-attention Ranking (sRank). This transformer-based framework uses linear pairwise loss with mutable training batch sizes and achieves quality gains and high efficiency, and has been applied effectively to show gains on two industry tasks at Microsoft over real-world large-scale data sets: Smart Reply (SR) and Ambient Clinical Intelligence (ACI). In Smart Reply, $sRank$ assists live customers with technical support by selecting the best reply from predefined solutions based on consumer and support agent messages. It achieves 11.7% gain in offline top-one accuracy on the SR task over the previous system, and 
    

