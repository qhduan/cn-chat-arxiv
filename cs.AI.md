# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Preference-Based Planning in Stochastic Environments: From Partially-Ordered Temporal Goals to Most Preferred Policies](https://arxiv.org/abs/2403.18212) | 使用偏序时序目标，将部分有序偏好映射到MDP策略偏好，并通过引入序理论实现最优策略的合成。 |
| [^2] | [ENOT: Expectile Regularization for Fast and Accurate Training of Neural Optimal Transport](https://arxiv.org/abs/2403.03777) | 通过期望回归正则化，本论文提出了一种新的神经优化传输（NOT）训练程序扩展，能够有效地估计最优输运方案，并使学习变得稳定。 |
| [^3] | [Polyhedral Complex Derivation from Piecewise Trilinear Networks](https://arxiv.org/abs/2402.10403) | 本文以三线性插值方法作为位置编码，提出了理论见解和分析网格提取方法，将高维曲面转换为平面，并引入了一种近似交点的方法，拓展了更广泛的应用。 |
| [^4] | [Synthesizing Sentiment-Controlled Feedback For Multimodal Text and Image Data](https://arxiv.org/abs/2402.07640) | 该论文提出了一个可控的多模态反馈合成系统，能够根据文本和图像输入生成具有特定情感（积极或消极）的反馈，有着广泛的应用价值。 |
| [^5] | [Mitigating Open-Vocabulary Caption Hallucinations](https://arxiv.org/abs/2312.03631) | 提出了在开放词汇设置中解决图像字幕幻觉问题的框架，并提出了一种新方法MOCHa来缓解幻觉 |
| [^6] | [Path-based Explanation for Knowledge Graph Completion.](http://arxiv.org/abs/2401.02290) | 基于路径的KGC解释器Power-Link通过引入图加权技术，实现了可解释的知识图谱补全，推动了模型透明度和可靠性的提升。 |
| [^7] | [FedSN: A General Federated Learning Framework over LEO Satellite Networks.](http://arxiv.org/abs/2311.01483) | FedSN是一个通用的联邦学习框架，用于解决在LEO卫星网络中的异构计算和存储能力、有限的上行速率以及模型陈旧等关键挑战。 |
| [^8] | [Efficient Anatomical labeling of Pulmonary Tree Structures via Implicit Point-Graph Networks.](http://arxiv.org/abs/2309.17329) | 本文介绍了一种通过隐式点图网络高效解剖标记肺部树状结构的方法，提供了SOTA准确度和可用的表面，同时还提供了一个用于评估该方法的数据集。 |
| [^9] | [Encode-Store-Retrieve: Enhancing Memory Augmentation through Language-Encoded Egocentric Perception.](http://arxiv.org/abs/2308.05822) | 本研究提出了一种记忆增强系统，它利用自然语言编码视频数据并将其存储在向量数据库中，通过利用大型视觉语言模型的强大功能来进行语言编码的过程。 |
| [^10] | [SIA-FTP: A Spoken Instruction Aware Flight Trajectory Prediction Framework.](http://arxiv.org/abs/2305.01661) | 提出一种语音指令感知的飞行轨迹预测框架，通过融合即时的语音指令和飞行轨迹表示，解决了语音指令和飞行轨迹的模态差距问题，在多个真实世界数据集上表现优异。 |

# 详细

[^1]: 在随机环境中基于偏序时序目标的首选规划

    Preference-Based Planning in Stochastic Environments: From Partially-Ordered Temporal Goals to Most Preferred Policies

    [https://arxiv.org/abs/2403.18212](https://arxiv.org/abs/2403.18212)

    使用偏序时序目标，将部分有序偏好映射到MDP策略偏好，并通过引入序理论实现最优策略的合成。

    

    人类偏好并非总是通过完全的线性顺序来表示：使用部分有序偏好来表达不可比较的结果是自然的。在这项工作中，我们考虑在随机系统中做决策和概率规划，这些系统被建模为马尔可夫决策过程（MDPs），给定一组有序偏好的时间延伸目标。具体而言，每个时间延伸目标都是使用线性时序逻辑有限轨迹（LTL$_f$）中的公式来表示的。为了根据部分有序偏好进行规划，我们引入了序理论来将对时间目标的偏好映射到对MDP策略的偏好。因此，在随机顺序下的一个最优选策略将导致MDP中有限路径上的一个随机非支配概率分布。为了合成一个最优选策略，我们的技术方法包括两个关键步骤。在第一步中，我们开发了一个程序...

    arXiv:2403.18212v1 Announce Type: cross  Abstract: Human preferences are not always represented via complete linear orders: It is natural to employ partially-ordered preferences for expressing incomparable outcomes. In this work, we consider decision-making and probabilistic planning in stochastic systems modeled as Markov decision processes (MDPs), given a partially ordered preference over a set of temporally extended goals. Specifically, each temporally extended goal is expressed using a formula in Linear Temporal Logic on Finite Traces (LTL$_f$). To plan with the partially ordered preference, we introduce order theory to map a preference over temporal goals to a preference over policies for the MDP. Accordingly, a most preferred policy under a stochastic ordering induces a stochastic nondominated probability distribution over the finite paths in the MDP. To synthesize a most preferred policy, our technical approach includes two key steps. In the first step, we develop a procedure to
    
[^2]: ENOT：期望回归用于神经优化传输的快速和准确训练

    ENOT: Expectile Regularization for Fast and Accurate Training of Neural Optimal Transport

    [https://arxiv.org/abs/2403.03777](https://arxiv.org/abs/2403.03777)

    通过期望回归正则化，本论文提出了一种新的神经优化传输（NOT）训练程序扩展，能够有效地估计最优输运方案，并使学习变得稳定。

    

    我们提出了一种新的神经优化传输（NOT）训练程序扩展，通过特定的共轭势正则化能够准确和高效地估计最优输运方案。现有NOT求解器的主要瓶颈在于找到共轭算子（即c-transform）的接近精确近似的过程，这要么通过优化最小-最大目标，要么通过计算密集型的对初始近似预测的精细调整来完成。我们通过提出一种新的、在期望回归形式上强制适应性条件于学习对偶势的理论上合理化损失来解决这两个问题。这样的正则化提供了可能共轭势分布的上限估计，并使学习变得稳定，消除了对额外广泛微调的需求。我们正式证明了我们的方法的效率。

    arXiv:2403.03777v1 Announce Type: cross  Abstract: We present a new extension for Neural Optimal Transport (NOT) training procedure, capable of accurately and efficiently estimating optimal transportation plan via specific regularisation on conjugate potentials. The main bottleneck of existing NOT solvers is associated with the procedure of finding a near-exact approximation of the conjugate operator (i.e., the c-transform), which is done either by optimizing over maximin objectives or by the computationally-intensive fine-tuning of the initial approximated prediction. We resolve both issues by proposing a new, theoretically justified loss in the form of expectile regularization that enforces binding conditions on the learning dual potentials. Such a regularization provides the upper bound estimation over the distribution of possible conjugate potentials and makes the learning stable, eliminating the need for additional extensive finetuning. We formally justify the efficiency of our me
    
[^3]: 从分段三线性网络中导出多面体复合体

    Polyhedral Complex Derivation from Piecewise Trilinear Networks

    [https://arxiv.org/abs/2402.10403](https://arxiv.org/abs/2402.10403)

    本文以三线性插值方法作为位置编码，提出了理论见解和分析网格提取方法，将高维曲面转换为平面，并引入了一种近似交点的方法，拓展了更广泛的应用。

    

    最近关于深度神经网络可视化的进展揭示了它们结构的见解，并且可以从连续分段仿射（CPWA）函数中提取网格。与此同时，神经表面表示学习的发展包括非线性位置编码，解决了诸如谱偏差之类的问题；然而，这在应用基于CPWA函数的网格提取技术方面带来了挑战。我们聚焦于三线性插值方法作为位置编码，提供了理论见解和分析的网格提取，展示了在奇拿尔约束下将高维曲面转换为三线性区域内的平面的过程。此外，我们引入了一种方法来近似三个高维曲面之间的交点，从而扩展了更广泛的应用。通过汉明距离和效率以及角距离来经验性地验证正确性和简洁性，同时检查了t之间的相关性

    arXiv:2402.10403v1 Announce Type: cross  Abstract: Recent advancements in visualizing deep neural networks provide insights into their structures and mesh extraction from Continuous Piecewise Affine (CPWA) functions. Meanwhile, developments in neural surface representation learning incorporate non-linear positional encoding, addressing issues like spectral bias; however, this poses challenges in applying mesh extraction techniques based on CPWA functions. Focusing on trilinear interpolating methods as positional encoding, we present theoretical insights and an analytical mesh extraction, showing the transformation of hypersurfaces to flat planes within the trilinear region under the eikonal constraint. Moreover, we introduce a method for approximating intersecting points among three hypersurfaces contributing to broader applications. We empirically validate correctness and parsimony through chamfer distance and efficiency, and angular distance, while examining the correlation between t
    
[^4]: 合成对多模态文本和图片数据的情感控制反馈

    Synthesizing Sentiment-Controlled Feedback For Multimodal Text and Image Data

    [https://arxiv.org/abs/2402.07640](https://arxiv.org/abs/2402.07640)

    该论文提出了一个可控的多模态反馈合成系统，能够根据文本和图像输入生成具有特定情感（积极或消极）的反馈，有着广泛的应用价值。

    

    生成对多模态输入（包括文本和图片）的情感控制反馈能够弥补人机交互领域的一个关键差距，使系统能够提供具有同理心、准确性和引人入胜的回应。这种能力在医疗、营销和教育等领域有着深远的应用。为此，我们构建了一个大规模的可控多模态反馈合成（CMFeed）数据集，并提出了一个可控的反馈合成系统。所提出的系统包括一个编码器、解码器和控制性模块，用于处理文本和视觉输入。它使用Transformer和Faster R-CNN网络提取文本和视觉特征，并将它们结合起来生成反馈。CMFeed数据集包含图片、文本、对帖子的反应、带有相关性评分的人类评论以及对评论的反应。对帖子和评论的反应被用来训练提出的模型以产生具有特定（积极或消极）情感的反馈。

    The ability to generate sentiment-controlled feedback in response to multimodal inputs, comprising both text and images, addresses a critical gap in human-computer interaction by enabling systems to provide empathetic, accurate, and engaging responses. This capability has profound applications in healthcare, marketing, and education. To this end, we construct a large-scale Controllable Multimodal Feedback Synthesis (CMFeed) dataset and propose a controllable feedback synthesis system. The proposed system includes an encoder, decoder, and controllability block for textual and visual inputs. It extracts textual and visual features using a transformer and Faster R-CNN networks and combines them to generate feedback. The CMFeed dataset encompasses images, text, reactions to the post, human comments with relevance scores, and reactions to the comments. The reactions to the post and comments are utilized to train the proposed model to produce feedback with a particular (positive or negative)
    
[^5]: 缓解开放词汇描述幻觉

    Mitigating Open-Vocabulary Caption Hallucinations

    [https://arxiv.org/abs/2312.03631](https://arxiv.org/abs/2312.03631)

    提出了在开放词汇设置中解决图像字幕幻觉问题的框架，并提出了一种新方法MOCHa来缓解幻觉

    

    近年来，图像条件的文本生成取得了快速进展，但图像字幕仍然存在幻觉的基本问题，即生成与给定图像无法推断的虚假细节。现有方法在图像字幕中大多使用封闭词汇对象列表来缓解或评估幻觉，忽略了实践中发生的大多数幻觉类型。为此，我们提出了一个框架，以应对开放词汇设置中图像字幕中的幻觉，包括量化它们的存在并优化以减轻这种幻觉。我们的OpenCHAIR基准利用生成基础模型来评估开放词汇描述幻觉，在多样性和准确性方面都超过了流行的CHAIR基准。为了在序列级别上缓解开放词汇的幻觉，我们提出了MOCHa，一种利用进展的方法

    arXiv:2312.03631v2 Announce Type: replace-cross  Abstract: While recent years have seen rapid progress in image-conditioned text generation, image captioning still suffers from the fundamental issue of hallucinations, namely, the generation of spurious details that cannot be inferred from the given image. Existing methods largely use closed-vocabulary object lists to mitigate or evaluate hallucinations in image captioning, ignoring most types of hallucinations that occur in practice. To this end, we propose a framework for addressing hallucinations in image captioning in the open-vocabulary setting, including quantifying their presence and optimizing to mitigate such hallucinations. Our OpenCHAIR benchmark leverages generative foundation models to evaluate open-vocabulary caption hallucinations, surpassing the popular CHAIR benchmark in both diversity and accuracy. To mitigate open-vocabulary hallucinations at the sequence level, we propose MOCHa, an approach harnessing advancements in
    
[^6]: 基于路径的知识图谱补全的解释方法

    Path-based Explanation for Knowledge Graph Completion. (arXiv:2401.02290v1 [cs.LG])

    [http://arxiv.org/abs/2401.02290](http://arxiv.org/abs/2401.02290)

    基于路径的KGC解释器Power-Link通过引入图加权技术，实现了可解释的知识图谱补全，推动了模型透明度和可靠性的提升。

    

    近年来，图神经网络（GNNs）通过建模实体和关系的交互在知识图谱补全（KGC）任务中取得了巨大成功。然而，对预测结果的解释却没有得到必要的关注。对基于GNN的KGC模型结果进行适当解释，可以增加模型的透明度，并帮助研究人员开发更可靠的模型。现有的KGC解释方法主要依赖于实例/子图的方法，而在某些场景下，路径可以提供更友好和可解释的解释。然而，还没有对生成基于路径的知识图谱解释方法进行充分探索。为了填补这一空白，我们提出了Power-Link，这是第一个探索基于路径的KGC解释器。我们设计了一种新颖的图加权技术，使得可以以完全可并行化和内存高效的训练方案生成基于路径的解释。我们还引入了三个新的度量指标，用于评估解释的质量和有效性。

    Graph Neural Networks (GNNs) have achieved great success in Knowledge Graph Completion (KGC) by modelling how entities and relations interact in recent years. However, the explanation of the predicted facts has not caught the necessary attention. Proper explanations for the results of GNN-based KGC models increase model transparency and help researchers develop more reliable models. Existing practices for explaining KGC tasks rely on instance/subgraph-based approaches, while in some scenarios, paths can provide more user-friendly and interpretable explanations. Nonetheless, the methods for generating path-based explanations for KGs have not been well-explored. To address this gap, we propose Power-Link, the first path-based KGC explainer that explores GNN-based models. We design a novel simplified graph-powering technique, which enables the generation of path-based explanations with a fully parallelisable and memory-efficient training scheme. We further introduce three new metrics for 
    
[^7]: FedSN：一个适用于LEO卫星网络的通用联邦学习框架

    FedSN: A General Federated Learning Framework over LEO Satellite Networks. (arXiv:2311.01483v1 [cs.LG])

    [http://arxiv.org/abs/2311.01483](http://arxiv.org/abs/2311.01483)

    FedSN是一个通用的联邦学习框架，用于解决在LEO卫星网络中的异构计算和存储能力、有限的上行速率以及模型陈旧等关键挑战。

    

    最近，许多低地球轨道（LEO）卫星已经由商业公司成功地发射和部署到太空中，如SpaceX。由于LEO卫星配备了多模传感器，它们不仅用于通信，还用于各种机器学习应用，如空间调制识别、遥感图像分类等。然而，由于与LEO卫星的有限接触时间（例如5分钟），地面站（GS）可能无法下载如此大量的原始感测数据进行集中模型训练。因此，联邦学习（FL）已经成为解决这个问题的有希望的解决方案，通过在设备上进行训练。不幸的是，要在LEO卫星上使用FL，我们仍然面临三个关键挑战，即i）异构计算和存储能力，ii）有限的上行速率，以及iii）模型陈旧问题。为此，我们提出了一种名为FedSN的通用FL框架来解决上述挑战，一

    Recently, a large number of Low Earth Orbit (LEO) satellites have been launched and deployed successfully in space by commercial companies, such as SpaceX. Due to multimodal sensors equipped by the LEO satellites, they serve not only for communication but also for various machine learning applications, such as space modulation recognition, remote sensing image classification, etc. However, the ground station (GS) may be incapable of downloading such a large volume of raw sensing data for centralized model training due to the limited contact time with LEO satellites (e.g. 5 minutes). Therefore, federated learning (FL) has emerged as the promising solution to address this problem via on-device training. Unfortunately, to enable FL on LEO satellites, we still face three critical challenges that are i) heterogeneous computing and memory capabilities, ii) limited uplink rate, and iii) model staleness. To this end, we propose FedSN as a general FL framework to tackle the above challenges, an
    
[^8]: 通过隐式点图网络高效解剖标记肺部树状结构

    Efficient Anatomical labeling of Pulmonary Tree Structures via Implicit Point-Graph Networks. (arXiv:2309.17329v1 [cs.CV])

    [http://arxiv.org/abs/2309.17329](http://arxiv.org/abs/2309.17329)

    本文介绍了一种通过隐式点图网络高效解剖标记肺部树状结构的方法，提供了SOTA准确度和可用的表面，同时还提供了一个用于评估该方法的数据集。

    

    肺部疾病在全球范围内是导致死亡的主要原因之一。治愈肺部疾病需要更好地理解肺部系统内的许多复杂的3D树状结构，如气道、动脉和静脉。在理论上，它们可以通过高分辨率图像堆栈进行建模。然而，基于密集体素网格的标准CNN方法代价过高。为了解决这个问题，我们引入了一种基于点的方法，保留了树骨架的图连通性，并结合了隐式表面表示。它以较低的计算成本提供了SOTA准确度，生成的模型具有可用的表面。由于公开可访问的数据稀缺，我们还整理了一套广泛的数据集来评估我们的方法，并将其公开。

    Pulmonary diseases rank prominently among the principal causes of death worldwide. Curing them will require, among other things, a better understanding of the many complex 3D tree-shaped structures within the pulmonary system, such as airways, arteries, and veins. In theory, they can be modeled using high-resolution image stacks. Unfortunately, standard CNN approaches operating on dense voxel grids are prohibitively expensive. To remedy this, we introduce a point-based approach that preserves graph connectivity of tree skeleton and incorporates an implicit surface representation. It delivers SOTA accuracy at a low computational cost and the resulting models have usable surfaces. Due to the scarcity of publicly accessible data, we have also curated an extensive dataset to evaluate our approach and will make it public.
    
[^9]: 编码-存储-检索：通过语言编码的自我中心感知增强记忆

    Encode-Store-Retrieve: Enhancing Memory Augmentation through Language-Encoded Egocentric Perception. (arXiv:2308.05822v1 [cs.CV])

    [http://arxiv.org/abs/2308.05822](http://arxiv.org/abs/2308.05822)

    本研究提出了一种记忆增强系统，它利用自然语言编码视频数据并将其存储在向量数据库中，通过利用大型视觉语言模型的强大功能来进行语言编码的过程。

    

    我们依赖于自己的记忆来编码、存储和检索我们的经历。然而，记忆间隔有时会发生。实现记忆增强的一种有希望的方法是通过使用增强现实头戴式显示设备来捕捉和保留自我中心的视频，这种做法通常被称为生活记录。然而，由于当前技术缺乏高效编码和存储如此大量的视频数据的能力，从庞大的视频存档中检索特定信息需要大量的计算能力，进一步复杂了快速访问所需内容的任务。

    We depend on our own memory to encode, store, and retrieve our experiences. However, memory lapses can occur. One promising avenue for achieving memory augmentation is through the use of augmented reality head-mounted displays to capture and preserve egocentric videos, a practice commonly referred to as life logging. However, a significant challenge arises from the sheer volume of video data generated through life logging, as the current technology lacks the capability to encode and store such large amounts of data efficiently. Further, retrieving specific information from extensive video archives requires substantial computational power, further complicating the task of quickly accessing desired content. To address these challenges, we propose a memory augmentation system that involves leveraging natural language encoding for video data and storing them in a vector database. This approach harnesses the power of large vision language models to perform the language encoding process. Add
    
[^10]: SIA-FTP: 一种语音指令感知的飞行轨迹预测框架

    SIA-FTP: A Spoken Instruction Aware Flight Trajectory Prediction Framework. (arXiv:2305.01661v1 [cs.SD])

    [http://arxiv.org/abs/2305.01661](http://arxiv.org/abs/2305.01661)

    提出一种语音指令感知的飞行轨迹预测框架，通过融合即时的语音指令和飞行轨迹表示，解决了语音指令和飞行轨迹的模态差距问题，在多个真实世界数据集上表现优异。

    

    通过语音通讯进行地空协商是确保空中交通管制（ATC）操作安全和效率的重要前提。但是，随着交通流量的增加，由于人为因素导致的错误指令给ATC安全带来了巨大威胁。现有的飞行轨迹预测（FTP）方法主要依赖于历史轨迹的飞行状态，在实时机动指令的预测上会出现显著的延迟，这不利于冲突检测。本文提出了一种名为SIA-FTP的语音指令感知FTP框架，通过包含即时的语音指令来支持高机动FTP任务。为了解决模态差距并最小化数据需求，我们提出了一种联合注意机制来融合语音指令嵌入和飞行轨迹表示。在多个真实世界数据集上评估了所提出的SIA-FTP，与现有的FTP方法相比取得了显著的改进。

    Ground-air negotiation via speech communication is a vital prerequisite for ensuring safety and efficiency in air traffic control (ATC) operations. However, with the increase in traffic flow, incorrect instructions caused by human factors bring a great threat to ATC safety. Existing flight trajectory prediction (FTP) approaches primarily rely on the flight status of historical trajectory, leading to significant delays in the prediction of real-time maneuvering instruction, which is not conducive to conflict detection. A major reason is that spoken instructions and flight trajectories are presented in different modalities in the current air traffic control (ATC) system, bringing great challenges to considering the maneuvering instruction in the FTP tasks. In this paper, a spoken instruction-aware FTP framework, called SIA-FTP, is innovatively proposed to support high-maneuvering FTP tasks by incorporating instant spoken instruction. To address the modality gap and minimize the data requ
    

