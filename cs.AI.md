# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SHIELD: A regularization technique for eXplainable Artificial Intelligence](https://arxiv.org/abs/2404.02611) | SHIELD引入了一种正则化技术，通过隐藏部分输入数据并评估预测结果的差异，从而改善了可解释人工智能模型的质量。 |
| [^2] | [Robust Diffusion Models for Adversarial Purification](https://arxiv.org/abs/2403.16067) | 提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。 |
| [^3] | [M-HOF-Opt: Multi-Objective Hierarchical Output Feedback Optimization via Multiplier Induced Loss Landscape Scheduling](https://arxiv.org/abs/2403.13728) | 提出了一种新的方法，通过多目标分层输出反馈优化的方式，利用乘子诱导的损失景观调度解决神经网络参数化的复杂损失函数优化问题。 |
| [^4] | [The VampPrior Mixture Model](https://arxiv.org/abs/2402.04412) | 本论文提出了VampPrior混合模型（VMM），它是一种新颖的DLVM先验，可用于深度潜变量模型的集成和聚类，通过改善当前聚类先验的不足，并提出了一个清晰区分变分和先验参数的推理过程。使用VMM的变分自动编码器在基准数据集上取得了强大的聚类性能，将VMM与scVI相结合可以显著提高其性能，并自动将细胞分组为具有生物意义的聚类。 |
| [^5] | [Hypergraph Structure Inference From Data Under Smoothness Prior.](http://arxiv.org/abs/2308.14172) | 本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。 |
| [^6] | [Addressing Selection Bias in Computerized Adaptive Testing: A User-Wise Aggregate Influence Function Approach.](http://arxiv.org/abs/2308.11912) | 本文研究了计算机自适应测试中存在的选择偏差问题，并提出了一种基于用户的聚合影响函数方法来解决该问题。 |
| [^7] | [HowkGPT: Investigating the Detection of ChatGPT-generated University Student Homework through Context-Aware Perplexity Analysis.](http://arxiv.org/abs/2305.18226) | 本研究介绍了一种基于元数据的 AI 生成的大学生作业检测方法 HowkGPT，通过计算困惑度得分来区分学生提交和 ChatGPT 生成的作业，进一步提高分析的精度，以帮助维护学术诚信和防止作弊。 |
| [^8] | [SketchOGD: Memory-Efficient Continual Learning.](http://arxiv.org/abs/2305.16424) | SketchOGD提出了一种内存高效的解决灾难性遗忘的方法，通过采用在线草图算法，将模型梯度压缩为固定大小的矩阵，从而改进了现有的算法——正交梯度下降（OGD）。 |

# 详细

[^1]: SHIELD: 一种用于可解释人工智能的正则化技术

    SHIELD: A regularization technique for eXplainable Artificial Intelligence

    [https://arxiv.org/abs/2404.02611](https://arxiv.org/abs/2404.02611)

    SHIELD引入了一种正则化技术，通过隐藏部分输入数据并评估预测结果的差异，从而改善了可解释人工智能模型的质量。

    

    随着人工智能系统在各个领域变得不可或缺，对可解释性的需求与日俱增。尽管科学界的努力主要集中在为模型获取更好的解释上，但重要的是不要忽视这个解释过程对改善训练的潜力。虽然现有的努力主要集中在为黑盒模型生成和评估解释上，但直接通过这些评估来增强模型仍存在关键差距。本文介绍了SHIELD（选择性隐藏输入评估学习动态），这是一种适用于可解释人工智能的正则化技术，旨在通过隐藏部分输入数据并评估预测结果的差异来改善模型质量。与传统方法相比，SHIELD正则化无缝集成到目标函数中，提高了模型的可解释性同时也改善了性能

    arXiv:2404.02611v1 Announce Type: new  Abstract: As Artificial Intelligence systems become integral across domains, the demand for explainability grows. While the effort by the scientific community is focused on obtaining a better explanation for the model, it is important not to ignore the potential of this explanation process to improve training as well. While existing efforts primarily focus on generating and evaluating explanations for black-box models, there remains a critical gap in directly enhancing models through these evaluations. This paper introduces SHIELD (Selective Hidden Input Evaluation for Learning Dynamics), a regularization technique for explainable artificial intelligence designed to improve model quality by concealing portions of input data and assessing the resulting discrepancy in predictions. In contrast to conventional approaches, SHIELD regularization seamlessly integrates into the objective function, enhancing model explainability while also improving perfor
    
[^2]: 针对对抗净化的强大扩散模型

    Robust Diffusion Models for Adversarial Purification

    [https://arxiv.org/abs/2403.16067](https://arxiv.org/abs/2403.16067)

    提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。

    

    基于扩散模型（DM）的对抗净化（AP）已被证明是对抗训练（AT）最有力的替代方法。然而，这些方法忽略了预训练的扩散模型本身对对抗攻击并不稳健这一事实。此外，扩散过程很容易破坏语义信息，在反向过程后生成高质量图像但与原始输入图像完全不同，导致标准精度下降。为了解决这些问题，一个自然的想法是利用对抗训练策略重新训练或微调预训练的扩散模型，然而这在计算上是禁止的。我们提出了一种新颖的具有对抗引导的稳健反向过程，它独立于给定的预训练DMs，并且避免了重新训练或微调DMs。这种强大的引导不仅可以确保生成的净化示例保留更多的语义内容，还可以...

    arXiv:2403.16067v1 Announce Type: cross  Abstract: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also m
    
[^3]: M-HOF-Opt: 多目标分层输出反馈优化：基于乘子诱导损失景观调度的方法

    M-HOF-Opt: Multi-Objective Hierarchical Output Feedback Optimization via Multiplier Induced Loss Landscape Scheduling

    [https://arxiv.org/abs/2403.13728](https://arxiv.org/abs/2403.13728)

    提出了一种新的方法，通过多目标分层输出反馈优化的方式，利用乘子诱导的损失景观调度解决神经网络参数化的复杂损失函数优化问题。

    

    当一个神经网络参数化的损失函数由许多项组成时，在优化过程中对权重乘子的组合选择形成了一个具有挑战性的问题。为了解决这个问题，我们提出了一个概率图模型（PGM），用于联合模型参数和乘子演化过程，具有基于超体积的似然，促进每个损失项的多目标下降。相应的参数和乘子估计作为一个顺序决策过程被转化为一个最优控制问题，其中多目标下降目标被分层地分派到一系列约束优化子问题中。子问题约束根据帕累托支配自动适应并作为低层乘子控制器调度损失景观的设定点，通过每个损失项的输出反馈来运行。我们的方法是无乘子的，并且在时代尺度上运行。

    arXiv:2403.13728v1 Announce Type: new  Abstract: When a neural network parameterized loss function consists of many terms, the combinatorial choice of weight multipliers during the optimization process forms a challenging problem. To address this, we proposed a probabilistic graphical model (PGM) for the joint model parameter and multiplier evolution process, with a hypervolume based likelihood that promotes multi-objective descent of each loss term. The corresponding parameter and multiplier estimation as a sequential decision process is then cast into an optimal control problem, where the multi-objective descent goal is dispatched hierarchically into a series of constraint optimization sub-problems. The sub-problem constraint automatically adapts itself according to Pareto dominance and serves as the setpoint for the low level multiplier controller to schedule loss landscapes via output feedback of each loss term. Our method is multiplier-free and operates at the timescale of epochs,
    
[^4]: VampPrior混合模型

    The VampPrior Mixture Model

    [https://arxiv.org/abs/2402.04412](https://arxiv.org/abs/2402.04412)

    本论文提出了VampPrior混合模型（VMM），它是一种新颖的DLVM先验，可用于深度潜变量模型的集成和聚类，通过改善当前聚类先验的不足，并提出了一个清晰区分变分和先验参数的推理过程。使用VMM的变分自动编码器在基准数据集上取得了强大的聚类性能，将VMM与scVI相结合可以显著提高其性能，并自动将细胞分组为具有生物意义的聚类。

    

    当前用于深度潜变量模型（DLVMs）的聚类先验需要预先定义聚类的数量，并且容易受到较差的初始化的影响。解决这些问题可以通过同时执行集成和聚类的方式极大地改进基于深度学习的scRNA-seq分析。我们将VampPrior（Tomczak和Welling，2018）调整为Dirichlet过程高斯混合模型，得到VampPrior混合模型（VMM），这是一种新颖的DLVM先验。我们提出了一个推理过程，交替使用变分推理和经验贝叶斯，以清楚地区分变分和先验参数。在基准数据集上使用VMM的变分自动编码器获得了极具竞争力的聚类性能。将VMM与广受欢迎的scRNA-seq集成方法scVI（Lopez等，2018）相结合，显著改善了其性能，并自动将细胞分组为具有生物意义的聚类。

    Current clustering priors for deep latent variable models (DLVMs) require defining the number of clusters a-priori and are susceptible to poor initializations. Addressing these deficiencies could greatly benefit deep learning-based scRNA-seq analysis by performing integration and clustering simultaneously. We adapt the VampPrior (Tomczak & Welling, 2018) into a Dirichlet process Gaussian mixture model, resulting in the VampPrior Mixture Model (VMM), a novel prior for DLVMs. We propose an inference procedure that alternates between variational inference and Empirical Bayes to cleanly distinguish variational and prior parameters. Using the VMM in a Variational Autoencoder attains highly competitive clustering performance on benchmark datasets. Augmenting scVI (Lopez et al., 2018), a popular scRNA-seq integration method, with the VMM significantly improves its performance and automatically arranges cells into biologically meaningful clusters.
    
[^5]: 从数据中基于光滑性先验推断超图结构

    Hypergraph Structure Inference From Data Under Smoothness Prior. (arXiv:2308.14172v1 [cs.LG])

    [http://arxiv.org/abs/2308.14172](http://arxiv.org/abs/2308.14172)

    本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。

    

    超图在处理涉及多个实体的高阶关系数据中非常重要。在没有明确超图可用的情况下，希望能够从节点特征中推断出有意义的超图结构，以捕捉数据内在的关系。然而，现有的方法要么采用简单预定义的规则，不能精确捕捉潜在超图结构的分布，要么学习超图结构和节点特征之间的映射，但需要大量标记数据（即预先存在的超图结构）进行训练。这两种方法都局限于实际情景中的应用。为了填补这一空白，我们提出了一种新的光滑性先验，使我们能够设计一种方法，在没有标记数据作为监督的情况下推断出每个潜在超边的概率。所提出的先验表示超边中的节点特征与包含该超边的超边的特征高度相关。

    Hypergraphs are important for processing data with higher-order relationships involving more than two entities. In scenarios where explicit hypergraphs are not readily available, it is desirable to infer a meaningful hypergraph structure from the node features to capture the intrinsic relations within the data. However, existing methods either adopt simple pre-defined rules that fail to precisely capture the distribution of the potential hypergraph structure, or learn a mapping between hypergraph structures and node features but require a large amount of labelled data, i.e., pre-existing hypergraph structures, for training. Both restrict their applications in practical scenarios. To fill this gap, we propose a novel smoothness prior that enables us to design a method to infer the probability for each potential hyperedge without labelled data as supervision. The proposed prior indicates features of nodes in a hyperedge are highly correlated by the features of the hyperedge containing th
    
[^6]: 解决计算机自适应测试中的选择偏差问题：一种基于用户的聚合影响函数方法

    Addressing Selection Bias in Computerized Adaptive Testing: A User-Wise Aggregate Influence Function Approach. (arXiv:2308.11912v1 [cs.LG])

    [http://arxiv.org/abs/2308.11912](http://arxiv.org/abs/2308.11912)

    本文研究了计算机自适应测试中存在的选择偏差问题，并提出了一种基于用户的聚合影响函数方法来解决该问题。

    

    计算机自适应测试（CAT）是一种广泛使用的高效测试模式，可以根据受试者在测试领域的熟练程度进行适应。CAT需要预先训练的项目简介，因为CAT根据已注册项目的简介实时评估学生，并使用候选项目的简介选择下一个要指导的项目。然而，获取这样的项目简介是一个昂贵的过程，涉及收集大量密集的项目响应数据，然后在收集的数据上训练诊断模型。在本文中，我们探讨了利用CAT服务中收集的响应数据的可能性。我们首先展示了这带来的独特挑战，原因是CAT引入了固有的选择偏差，即熟练程度更高的学生会收到更难的问题。实际上，当使用CAT响应数据进行简单训练诊断模型时，我们观察到项目简介与实际情况显著偏离。为了解决选择偏差问题，我们提出了基于用户的聚合影响函数方法。

    Computerized Adaptive Testing (CAT) is a widely used, efficient test mode that adapts to the examinee's proficiency level in the test domain. CAT requires pre-trained item profiles, for CAT iteratively assesses the student real-time based on the registered items' profiles, and selects the next item to administer using candidate items' profiles. However, obtaining such item profiles is a costly process that involves gathering a large, dense item-response data, then training a diagnostic model on the collected data. In this paper, we explore the possibility of leveraging response data collected in the CAT service. We first show that this poses a unique challenge due to the inherent selection bias introduced by CAT, i.e., more proficient students will receive harder questions. Indeed, when naively training the diagnostic model using CAT response data, we observe that item profiles deviate significantly from the ground-truth. To tackle the selection bias issue, we propose the user-wise agg
    
[^7]: HowkGPT: 基于上下文感知困惑度分析的 ChatGPT 生成的大学生作业检测研究

    HowkGPT: Investigating the Detection of ChatGPT-generated University Student Homework through Context-Aware Perplexity Analysis. (arXiv:2305.18226v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.18226](http://arxiv.org/abs/2305.18226)

    本研究介绍了一种基于元数据的 AI 生成的大学生作业检测方法 HowkGPT，通过计算困惑度得分来区分学生提交和 ChatGPT 生成的作业，进一步提高分析的精度，以帮助维护学术诚信和防止作弊。

    

    随着大型语言模型（LLM）在文本生成任务中的使用越来越普遍，人们担心它们可能会危及学术诚信。教育部门目前正在努力区分学生提交的家庭作业和AI生成的作业。本文通过引入 HowkGPT 标识由 AI 生成的作业来解决这一挑战。HowkGPT 基于一组学术作业和相应元数据构建，并使用预训练的 LLM 计算学生提交和 ChatGPT 生成的回答的困惑度得分。然后，这些得分有助于建立区分提交作业来源的阈值。鉴于学术工作的特殊性和上下文性质，HowkGPT 还通过定义从元数据中导出的类别特定的阈值来进一步提高分析的精度。本研究强调了在 LLM 文本生成时期维护学术诚信和防止作弊的有效策略的关键性需求。

    As the use of Large Language Models (LLMs) in text generation tasks proliferates, concerns arise over their potential to compromise academic integrity. The education sector currently tussles with distinguishing student-authored homework assignments from AI-generated ones. This paper addresses the challenge by introducing HowkGPT, designed to identify homework assignments generated by AI. HowkGPT is built upon a dataset of academic assignments and accompanying metadata [17] and employs a pretrained LLM to compute perplexity scores for student-authored and ChatGPT-generated responses. These scores then assist in establishing a threshold for discerning the origin of a submitted assignment. Given the specificity and contextual nature of academic work, HowkGPT further refines its analysis by defining category-specific thresholds derived from the metadata, enhancing the precision of the detection. This study emphasizes the critical need for effective strategies to uphold academic integrity a
    
[^8]: SketchOGD：内存高效的持续学习

    SketchOGD: Memory-Efficient Continual Learning. (arXiv:2305.16424v1 [cs.LG])

    [http://arxiv.org/abs/2305.16424](http://arxiv.org/abs/2305.16424)

    SketchOGD提出了一种内存高效的解决灾难性遗忘的方法，通过采用在线草图算法，将模型梯度压缩为固定大小的矩阵，从而改进了现有的算法——正交梯度下降（OGD）。

    

    当机器学习模型在一系列任务上持续训练时，它们容易忘记先前任务上学习到的知识，这种现象称为灾难性遗忘。现有的解决灾难性遗忘的方法往往涉及存储过去任务的信息，这意味着内存使用是确定实用性的主要因素。本文提出了一种内存高效的解决灾难性遗忘的方法，改进了一种已有的算法——正交梯度下降（OGD）。OGD利用先前模型梯度来找到维持先前数据点性能的权重更新。然而，由于存储先前模型梯度的内存成本随算法运行时间增长而增加，因此OGD不适用于任意长时间跨度的连续学习。针对这个问题，本文提出了SketchOGD。SketchOGD采用在线草图算法，将模型梯度压缩为固定大小的矩阵。

    When machine learning models are trained continually on a sequence of tasks, they are liable to forget what they learned on previous tasks -- a phenomenon known as catastrophic forgetting. Proposed solutions to catastrophic forgetting tend to involve storing information about past tasks, meaning that memory usage is a chief consideration in determining their practicality. This paper proposes a memory-efficient solution to catastrophic forgetting, improving upon an established algorithm known as orthogonal gradient descent (OGD). OGD utilizes prior model gradients to find weight updates that preserve performance on prior datapoints. However, since the memory cost of storing prior model gradients grows with the runtime of the algorithm, OGD is ill-suited to continual learning over arbitrarily long time horizons. To address this problem, this paper proposes SketchOGD. SketchOGD employs an online sketching algorithm to compress model gradients as they are encountered into a matrix of a fix
    

