# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Empowering Biomedical Discovery with AI Agents](https://arxiv.org/abs/2404.02831) | AI科学家代理结合人类创造力和专业知识，利用机器学习工具赋能生物医学研究，对多个领域产生影响。 |
| [^2] | [The Larger the Better? Improved LLM Code-Generation via Budget Reallocation](https://arxiv.org/abs/2404.00725) | 较小的语言模型可以在相同预算下产生可靠的改进，但在无法进行单元测试的情况下，较小的模型选择排名次于较大模型的单个输出。 |
| [^3] | [ReMamber: Referring Image Segmentation with Mamba Twister](https://arxiv.org/abs/2403.17839) | 提出了ReMamber，一种整合了Mamba和多模态Mamba Twister块的新型RIS架构，通过其独特的通道和空间扭曲机制实现图像-文本交互，取得了三个基准测试的最新技术成果 |
| [^4] | [AutoRE: Document-Level Relation Extraction with Large Language Models](https://arxiv.org/abs/2403.14888) | AutoRE 是一种端到端的文档级关系抽取模型，采用了一种名为RHF的新颖关系抽取范式，可有效处理分布在文档中的多个关系和三元组事实。 |
| [^5] | [A Unified Framework for Model Editing](https://arxiv.org/abs/2403.14236) | 这个统一框架结合了“定位和编辑”模型编辑技术，最大化保留某些向量表示并记忆新事实信息。 |
| [^6] | [Identifying Semantic Induction Heads to Understand In-Context Learning](https://arxiv.org/abs/2402.13055) | 该研究通过分析注意力头的操作，揭示了结合了句法依赖和知识图关系的语义感应头的出现，从而更好地理解了大型语言模型的上下文学习能力。 |
| [^7] | [HyperMoE: Towards Better Mixture of Experts via Transferring Among Experts](https://arxiv.org/abs/2402.12656) | HyperMoE通过Hypernetworks框架整合知识传递的概念，解决了在专家选择过程中专家知识稀疏性和可用性之间的矛盾。 |
| [^8] | [3D Diffuser Actor: Policy Diffusion with 3D Scene Representations](https://arxiv.org/abs/2402.10885) | 通过策略扩散和3D场景表示相结合，提出了3D Diffuser Actor，一个神经策略架构，可以根据语言指令构建3D视觉场景表示，并对机器人末端执行器的3D旋转和平移进行迭代去噪。 |
| [^9] | [Unsupervised Outlier Detection using Random Subspace and Subsampling Ensembles of Dirichlet Process Mixtures.](http://arxiv.org/abs/2401.00773) | 提出一种基于随机子空间和子抽样集合的Dirichlet过程高斯混合模型的无监督异常检测方法，提高了计算效率和检测器的鲁棒性。 |
| [^10] | [P-ROCKET: Pruning Random Convolution Kernels for Time Series Classification.](http://arxiv.org/abs/2309.08499) | 本研究提出了一种名为P-ROCKET的方法，通过在特征选择的角度删除卷积核，从而实现对时间序列分类中的随机卷积核进行剪枝。 |
| [^11] | [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants.](http://arxiv.org/abs/2308.16884) | Belebele是一个包含122种语言变体的多选机器阅读理解数据集，可用于评估文本模型在高、中和低资源语言中的性能。尽管英语为中心的大型语言模型在跨语言转移方面表现良好，但小型多语言遮蔽语言模型在其他语言上表现更佳。 |
| [^12] | [The most likely common cause.](http://arxiv.org/abs/2306.17557) | 对于因果不充分的情况下的共同原因问题，我们使用广义最大似然方法来识别共同原因C，与最大熵原则密切相关。对于两个二元对称变量的研究揭示了类似于二阶相变的条件概率非解析行为。 |
| [^13] | [Hierarchical Classification of Research Fields in the "Web of Science" Using Deep Learning.](http://arxiv.org/abs/2302.00390) | 本文提出了一个使用深度学习进行层次分类的系统，可以自动将学术出版物通过抽象进行分类，实现了对研究活动在不同层次结构中的全面分类，并允许跨学科和跨领域的单标签和多标签分类。 |

# 详细

[^1]: 利用AI代理赋能生物医学发现

    Empowering Biomedical Discovery with AI Agents

    [https://arxiv.org/abs/2404.02831](https://arxiv.org/abs/2404.02831)

    AI科学家代理结合人类创造力和专业知识，利用机器学习工具赋能生物医学研究，对多个领域产生影响。

    

    我们设想“AI科学家”作为系统，能够进行怀疑性学习和推理，通过将机器学习工具与实验平台集成的协作代理，赋能生物医学研究。生物医学AI代理不是要剔除人类在发现过程中的作用，而是结合人类的创造力和专业知识，利用AI分析大型数据集、导航假设空间，并执行重复性任务的能力。这些代理擅长各种任务，包括自我评估和规划发现工作流程。它们利用大型语言模型和生成模型进行结构化记忆，以便持续学习，并利用机器学习工具整合科学知识、生物原理和理论。AI代理可以影响从混合细胞模拟、可编程控制表型到细胞电路设计以及新疗法开发等领域。

    arXiv:2404.02831v1 Announce Type: new  Abstract: We envision 'AI scientists' as systems capable of skeptical learning and reasoning that empower biomedical research through collaborative agents that integrate machine learning tools with experimental platforms. Rather than taking humans out of the discovery process, biomedical AI agents combine human creativity and expertise with AI's ability to analyze large datasets, navigate hypothesis spaces, and execute repetitive tasks. AI agents are proficient in a variety of tasks, including self-assessment and planning of discovery workflows. These agents use large language models and generative models to feature structured memory for continual learning and use machine learning tools to incorporate scientific knowledge, biological principles, and theories. AI agents can impact areas ranging from hybrid cell simulation, programmable control of phenotypes, and the design of cellular circuits to the development of new therapies.
    
[^2]: 越大越好吗？通过预算重新分配改进LLM代码生成

    The Larger the Better? Improved LLM Code-Generation via Budget Reallocation

    [https://arxiv.org/abs/2404.00725](https://arxiv.org/abs/2404.00725)

    较小的语言模型可以在相同预算下产生可靠的改进，但在无法进行单元测试的情况下，较小的模型选择排名次于较大模型的单个输出。

    

    人们普遍认为，大型语言模型(LLMs)比较小的模型更好。然而，更大的模型在推断过程中也需要更多的时间和计算资源。这就引出了一个问题：当两个模型在相同的预算下运行时会发生什么？（例如，计算资源，运行时间）。为了解决这个问题，我们分析了各种大小的代码生成LLMs，并进行比较，例如运行一个70B模型一次与从13B模型生成五个输出并选择一个的情况。我们的研究结果表明，在标准单元测试设置中，反复使用较小的模型可以产生一致的改进，在五个任务中最高可达15%的增益。另一方面，在无法进行单元测试的情况下，从较小模型中基于排名的候选选择表现不及来自较大模型的单个输出。我们的结果突显了使用较小模型而非较大模型的潜力。

    arXiv:2404.00725v1 Announce Type: cross  Abstract: It is a common belief that large language models (LLMs) are better than smaller-sized ones. However, larger models also require significantly more time and compute during inference. This begs the question: what happens when both models operate under the same budget? (e.g., compute, run-time). To address this question, we analyze code generation LLMs of various sizes and make comparisons such as running a 70B model once vs. generating five outputs from a 13B model and selecting one. Our findings reveal that, in a standard unit-test setup, the repeated use of smaller models can yield consistent improvements, with gains of up to 15% across five tasks. On the other hand, in scenarios where unit-tests are unavailable, a ranking-based selection of candidates from the smaller model falls short of the performance of a single output from larger ones. Our results highlight the potential of using smaller models instead of larger ones, and the imp
    
[^3]: ReMamber：使用Mamba Twister实现引用图像分割

    ReMamber: Referring Image Segmentation with Mamba Twister

    [https://arxiv.org/abs/2403.17839](https://arxiv.org/abs/2403.17839)

    提出了ReMamber，一种整合了Mamba和多模态Mamba Twister块的新型RIS架构，通过其独特的通道和空间扭曲机制实现图像-文本交互，取得了三个基准测试的最新技术成果

    

    引用图像分割（RIS）利用变换器在解释复杂的视觉-语言任务方面取得了巨大成功。然而，二次计算成本使其在捕捉远程视觉-语言依赖性方面消耗资源。幸运的是，Mamba通过高效的线性复杂度在处理方面解决了这个问题。然而，将Mamba直接应用于多模态交互会面临挑战，主要原因是因为通道交互不足，无法有效融合多模态数据。在本文中，我们提出了ReMamber，这是一种整合了Mamba和多模态Mamba Twister块强大功能的新型RIS架构。Mamba Twister通过其独特的通道和空间扭曲机制明确建模图像-文本交互，并通过其独特的通道和空间扭曲机制融合文本和视觉特征。我们在三个具有挑战性的基准测试中取得了最新技术成果。此外，我们对ReMamber进行了彻底分析，并讨论其他...

    arXiv:2403.17839v1 Announce Type: cross  Abstract: Referring Image Segmentation (RIS) leveraging transformers has achieved great success on the interpretation of complex visual-language tasks. However, the quadratic computation cost makes it resource-consuming in capturing long-range visual-language dependencies. Fortunately, Mamba addresses this with efficient linear complexity in processing. However, directly applying Mamba to multi-modal interactions presents challenges, primarily due to inadequate channel interactions for the effective fusion of multi-modal data. In this paper, we propose ReMamber, a novel RIS architecture that integrates the power of Mamba with a multi-modal Mamba Twister block. The Mamba Twister explicitly models image-text interaction, and fuses textual and visual features through its unique channel and spatial twisting mechanism. We achieve the state-of-the-art on three challenging benchmarks. Moreover, we conduct thorough analyses of ReMamber and discuss other
    
[^4]: AutoRE：使用大型语言模型进行文档级关系抽取

    AutoRE: Document-Level Relation Extraction with Large Language Models

    [https://arxiv.org/abs/2403.14888](https://arxiv.org/abs/2403.14888)

    AutoRE 是一种端到端的文档级关系抽取模型，采用了一种名为RHF的新颖关系抽取范式，可有效处理分布在文档中的多个关系和三元组事实。

    

    大型语言模型(LLMs)展示了在理解和生成文本方面的异常能力，这激励着许多研究人员利用它们进行信息抽取(IE)任务，包括关系抽取(RE)。然而，大多数现有方法主要设计用于句子级关系抽取(SentRE)任务，这通常涵盖了单个句子内的一组关系和三元组事实。此外，一些方法采用将关系作为候选选择集成到提示模板中的方式，导致在处理分布在给定文档中的多个关系和三元组事实时效率低下，性能亚优，并在处理文档级关系抽取(DocRE)任务时面临独特挑战。为了克服这些限制，我们介绍了AutoRE，这是一个端到端的DocRE模型，采用了一种名为RHF(Re

    arXiv:2403.14888v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated exceptional abilities in comprehending and generating text, motivating numerous researchers to utilize them for Information Extraction (IE) purposes, including Relation Extraction (RE). Nonetheless, most existing methods are predominantly designed for Sentence-level Relation Extraction (SentRE) tasks, which typically encompass a restricted set of relations and triplet facts within a single sentence. Furthermore, certain approaches resort to treating relations as candidate choices integrated into prompt templates, leading to inefficient processing and suboptimal performance when tackling Document-Level Relation Extraction (DocRE) tasks, which entail handling multiple relations and triplet facts distributed across a given document, posing distinct challenges. To overcome these limitations, we introduce AutoRE, an end-to-end DocRE model that adopts a novel RE extraction paradigm named RHF (Re
    
[^5]: 一个统一的模型编辑框架

    A Unified Framework for Model Editing

    [https://arxiv.org/abs/2403.14236](https://arxiv.org/abs/2403.14236)

    这个统一框架结合了“定位和编辑”模型编辑技术，最大化保留某些向量表示并记忆新事实信息。

    

    模型编辑是一个不断发展的领域，专注于更新模型中嵌入的知识。在各种方法中，ROME和MEMIT作为主要的“定位和编辑”模型编辑技术脱颖而出。而MEMIT可以批量编辑记忆，ROME则一次只能改变一个事实。本文引入了一个统一的框架，将ROME和MEMIT纳入一个单一的概念框架，优化同一目标，我们称之为“保存-记忆”目标。该目标旨在在记忆新事实信息的同时保留某些选定向量的表示。具体来说，ROME使用等式约束优化此目标，而MEMIT采用更灵活的最小二乘约束。除了批量编辑外，MEMIT还可以在多个层面编辑模型。我们将编辑的分布从多个层面分开，区别于优化目标。

    arXiv:2403.14236v1 Announce Type: cross  Abstract: Model editing is a growing area focused on updating the knowledge embedded within models. Among the various methodologies, ROME and MEMIT stand out as leading "locate-and-edit" model editing techniques. While MEMIT enables batched editing of memories, ROME is limited to changing one fact at a time. This paper introduces a unifying framework that brings ROME and MEMIT under a single conceptual umbrella, optimizing for the same goal, which we call the "preservation-memorization" objective. This objective aims to preserve the representations of certain selected vectors while memorizing the representations of new factual information. Specifically, ROME optimizes this objective using an equality constraint, whereas MEMIT employs a more flexible least-square constraint. In addition to making batched edits, MEMIT also edits the model at multiple layers. We disentangle the distribution of edits to multiple layers from the optimization objectiv
    
[^6]: 识别语义感应头以理解上下文学习

    Identifying Semantic Induction Heads to Understand In-Context Learning

    [https://arxiv.org/abs/2402.13055](https://arxiv.org/abs/2402.13055)

    该研究通过分析注意力头的操作，揭示了结合了句法依赖和知识图关系的语义感应头的出现，从而更好地理解了大型语言模型的上下文学习能力。

    

    虽然大型语言模型(LLMs)已经展示出卓越的性能，但它们推理逻辑的不透明性引发了对其可靠性的担忧。为了更好地理解LLMs，我们对注意力头的操作进行了详细分析，并旨在更好地理解LLMs的上下文学习。具体而言，我们研究了注意力头是否编码了自然语言中存在的两种类型的关系：从句子中解析的句法依赖和知识图中的关系。我们发现某些注意力头表现出一种模式，即当关注头标记时，它们会回忆起尾标记，并增加这些尾标记的输出逻辑。更重要的是，这种语义感应头的制定与语言模型上下文学习能力的出现存在密切关联。语义注意力头的研究推动了我们的

    arXiv:2402.13055v1 Announce Type: cross  Abstract: Although large language models (LLMs) have demonstrated remarkable performance, the lack of transparency in their inference logic raises concerns about their trustworthiness. To gain a better understanding of LLMs, we conduct a detailed analysis of the operations of attention heads and aim to better understand the in-context learning of LLMs. Specifically, we investigate whether attention heads encode two types of relationships between tokens present in natural languages: the syntactic dependency parsed from sentences and the relation within knowledge graphs. We find that certain attention heads exhibit a pattern where, when attending to head tokens, they recall tail tokens and increase the output logits of those tail tokens. More crucially, the formulation of such semantic induction heads has a close correlation with the emergence of the in-context learning ability of language models. The study of semantic attention heads advances our
    
[^7]: HyperMoE: 通过专家之间的知识传递实现更好的专家混合

    HyperMoE: Towards Better Mixture of Experts via Transferring Among Experts

    [https://arxiv.org/abs/2402.12656](https://arxiv.org/abs/2402.12656)

    HyperMoE通过Hypernetworks框架整合知识传递的概念，解决了在专家选择过程中专家知识稀疏性和可用性之间的矛盾。

    

    混合专家(MoE)在语言模型中被证明有效地增强了模型的能力，通过动态地将每个输入标记路由到特定的专家子集进行处理。尽管取得了成功，但大多数现有方法在专家知识的稀疏性和可用性之间面临挑战：通过增加对专家知识的使用来增强性能，往往会导致在专家选择过程中稀疏度减少。为了缓解这一矛盾，我们提出了HyperMoE，这是一个建立在Hypernetworks之上的新颖MoE框架。该框架将MoE的计算过程与多任务学习中的知识传递概念进行了集成。基于未选择专家信息生成的特定模块作为补充信息，允许未被选中的专家的知识在保持选择稀疏性的同时被使用。

    arXiv:2402.12656v1 Announce Type: cross  Abstract: The Mixture of Experts (MoE) for language models has been proven effective in augmenting the capacity of models by dynamically routing each input token to a specific subset of experts for processing. Despite the success, most existing methods face a challenge for balance between sparsity and the availability of expert knowledge: enhancing performance through increased use of expert knowledge often results in diminishing sparsity during expert selection. To mitigate this contradiction, we propose HyperMoE, a novel MoE framework built upon Hypernetworks. This framework integrates the computational processes of MoE with the concept of knowledge transferring in multi-task learning. Specific modules generated based on the information of unselected experts serve as supplementary information, which allows the knowledge of experts not selected to be used while maintaining selection sparsity. Our comprehensive empirical evaluations across multi
    
[^8]: 基于3D场景表示的3D扩散器Actor：通过策略扩散进行机器人操作

    3D Diffuser Actor: Policy Diffusion with 3D Scene Representations

    [https://arxiv.org/abs/2402.10885](https://arxiv.org/abs/2402.10885)

    通过策略扩散和3D场景表示相结合，提出了3D Diffuser Actor，一个神经策略架构，可以根据语言指令构建3D视觉场景表示，并对机器人末端执行器的3D旋转和平移进行迭代去噪。

    

    我们将扩散策略和3D场景表示相结合，用于机器人操作。扩散策略通过条件扩散模型学习基于机器人和环境状态的动作分布。最近，它们已经表现出优于确定性和其他基于状态的动作分布学习方法。3D机器人策略使用从单个或多个摄像头视角获取的感应深度聚合的3D场景特征表示。它们已经证明比其2D对应物在摄像机视角上具有更好的泛化能力。我们统一了这两条线路的工作，并提出了3D扩散器Actor，这是一个神经策略架构，它在给定语言指令的情况下，构建视觉场景的3D表示，并在其上进行条件迭代去噪机器人末端执行器的3D旋转和平移。在每个去噪迭代中，我们的模型将末端执行器姿态估计表示为3D场景令牌，并预测t

    arXiv:2402.10885v1 Announce Type: cross  Abstract: We marry diffusion policies and 3D scene representations for robot manipulation. Diffusion policies learn the action distribution conditioned on the robot and environment state using conditional diffusion models. They have recently shown to outperform both deterministic and alternative state-conditioned action distribution learning methods. 3D robot policies use 3D scene feature representations aggregated from a single or multiple camera views using sensed depth. They have shown to generalize better than their 2D counterparts across camera viewpoints. We unify these two lines of work and present 3D Diffuser Actor, a neural policy architecture that, given a language instruction, builds a 3D representation of the visual scene and conditions on it to iteratively denoise 3D rotations and translations for the robot's end-effector. At each denoising iteration, our model represents end-effector pose estimates as 3D scene tokens and predicts t
    
[^9]: 使用随机子空间和Dirichlet过程混合模型的子抽样集合进行无监督异常检测

    Unsupervised Outlier Detection using Random Subspace and Subsampling Ensembles of Dirichlet Process Mixtures. (arXiv:2401.00773v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.00773](http://arxiv.org/abs/2401.00773)

    提出一种基于随机子空间和子抽样集合的Dirichlet过程高斯混合模型的无监督异常检测方法，提高了计算效率和检测器的鲁棒性。

    

    概率混合模型被认为是一种有价值的工具，用于无监督异常检测，因为它们具有解释性，并且在统计原理上有直观基础。在这个框架内，Dirichlet过程混合模型作为传统有限混合模型在聚类和异常检测任务中的一个引人注目的替代选择。然而，尽管它们明显具有优势，但在无监督异常检测中广泛采用Dirichlet过程混合模型受到与构建检测器过程中的计算效率和对异常值的敏感性有关的挑战的阻碍。为了解决这些挑战，我们提出了一种基于Dirichlet过程高斯混合模型集合的新型异常检测方法。所提出的方法是一种完全无监督的算法，利用了随机子空间和子抽样集合，不仅确保了高效计算，还增强了结果异常检测器的鲁棒性。

    Probabilistic mixture models are acknowledged as a valuable tool for unsupervised outlier detection owing to their interpretability and intuitive grounding in statistical principles. Within this framework, Dirichlet process mixture models emerge as a compelling alternative to conventional finite mixture models for both clustering and outlier detection tasks. However, despite their evident advantages, the widespread adoption of Dirichlet process mixture models in unsupervised outlier detection has been hampered by challenges related to computational inefficiency and sensitivity to outliers during the construction of detectors. To tackle these challenges, we propose a novel outlier detection method based on ensembles of Dirichlet process Gaussian mixtures. The proposed method is a fully unsupervised algorithm that capitalizes on random subspace and subsampling ensembles, not only ensuring efficient computation but also enhancing the robustness of the resulting outlier detector. Moreover,
    
[^10]: P-ROCKET: 针对时间序列分类的随机卷积核剪枝

    P-ROCKET: Pruning Random Convolution Kernels for Time Series Classification. (arXiv:2309.08499v1 [cs.LG])

    [http://arxiv.org/abs/2309.08499](http://arxiv.org/abs/2309.08499)

    本研究提出了一种名为P-ROCKET的方法，通过在特征选择的角度删除卷积核，从而实现对时间序列分类中的随机卷积核进行剪枝。

    

    在最近几年，两个时间序列分类模型ROCKET和MINIROCKET因其低训练成本和最先进的准确性而受到广泛关注。ROCKET和MINIROCKET利用无需训练的随机一维卷积核，可以快速从时间序列数据中提取特征，从而实现线性分类器的高效拟合。然而，为了全面捕捉有用的特征，需要大量的随机卷积核，这对于资源受限的设备来说是不兼容的。因此，我们设计了一种启发式进化算法S-ROCKET，用于识别和剪枝冗余的卷积核。然而，进化算法本身的特性导致在S-ROCKET中评估卷积核是一个耗时的过程。本文中，与直接评估具有非显著差异的随机卷积核的S-ROCKET不同，我们从特征选择的角度删除卷积核，通过消除序列中的相关连接来实现。

    In recent years, two time series classification models, ROCKET and MINIROCKET, have attracted much attention for their low training cost and state-of-the-art accuracy. Utilizing random 1-D convolutional kernels without training, ROCKET and MINIROCKET can rapidly extract features from time series data, allowing for the efficient fitting of linear classifiers. However, to comprehensively capture useful features, a large number of random kernels are required, which is incompatible for resource-constrained devices. Therefore, a heuristic evolutionary algorithm named S-ROCKET is devised to recognize and prune redundant kernels. Nevertheless, the inherent nature of evolutionary algorithms renders the evaluation of kernels within S-ROCKET an unacceptable time-consuming process. In this paper, diverging from S-ROCKET, which directly evaluates random kernels with nonsignificant differences, we remove kernels from a feature selection perspective by eliminating associating connections in the sequ
    
[^11]: Belebele基准数据集：122种语言变体的并行阅读理解数据集

    The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. (arXiv:2308.16884v1 [cs.CL])

    [http://arxiv.org/abs/2308.16884](http://arxiv.org/abs/2308.16884)

    Belebele是一个包含122种语言变体的多选机器阅读理解数据集，可用于评估文本模型在高、中和低资源语言中的性能。尽管英语为中心的大型语言模型在跨语言转移方面表现良好，但小型多语言遮蔽语言模型在其他语言上表现更佳。

    

    我们提出了Belebele，一个包含122种语言变体的多选机器阅读理解（MRC）数据集。该数据集极大地扩展了自然语言理解（NLU）基准的语言覆盖范围，使得可以评估文本模型在高、中和低资源语言中的性能。每个问题都基于Flores-200数据集中的一个短篇文章，并提供了四个多选答案。问题经过精心策划，以区分具有不同通用语言理解水平的模型。单独的英语数据集已经足够困难，可以挑战最先进的语言模型。由于完全并行，该数据集可以直接比较所有语言的模型性能。我们使用该数据集评估多语言遮蔽语言模型（MLMs）和大型语言模型（LLMs）的能力。我们展示了广泛的结果，并发现尽管英语为中心的LLMs之间存在显著的跨语言转移，但小型MLMs在其他语言上的表现相对较好。

    We present Belebele, a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants. Significantly expanding the language coverage of natural language understanding (NLU) benchmarks, this dataset enables the evaluation of text models in high-, medium-, and low-resource languages. Each question is based on a short passage from the Flores-200 dataset and has four multiple-choice answers. The questions were carefully curated to discriminate between models with different levels of general language comprehension. The English dataset on its own proves difficult enough to challenge state-of-the-art language models. Being fully parallel, this dataset enables direct comparison of model performance across all languages. We use this dataset to evaluate the capabilities of multilingual masked language models (MLMs) and large language models (LLMs). We present extensive results and find that despite significant cross-lingual transfer in English-centric LLMs, much small
    
[^12]: 最可能的共同原因

    The most likely common cause. (arXiv:2306.17557v1 [physics.data-an])

    [http://arxiv.org/abs/2306.17557](http://arxiv.org/abs/2306.17557)

    对于因果不充分的情况下的共同原因问题，我们使用广义最大似然方法来识别共同原因C，与最大熵原则密切相关。对于两个二元对称变量的研究揭示了类似于二阶相变的条件概率非解析行为。

    

    对于两个随机变量A和B的共同原因原则在因果不充分的情况下进行了研究，当它们的共同原因C被认为已经存在，但只观测到了A和B的联合概率。因此，C不能被唯一确定（潜在混杂因子问题）。我们展示了广义最大似然方法可以应用于这种情况，并且允许识别与共同原因原则一致的C。它与最大熵原则密切相关。对两个二元对称变量的研究揭示了条件概率的非解析行为，类似于二阶相变。这发生在观察到的概率分布从相关到反相关的过渡期间。讨论了广义似然方法与其他方法（如预测似然和最小共同原因熵）之间的关系。

    The common cause principle for two random variables $A$ and $B$ is examined in the case of causal insufficiency, when their common cause $C$ is known to exist, but only the joint probability of $A$ and $B$ is observed. As a result, $C$ cannot be uniquely identified (the latent confounder problem). We show that the generalized maximum likelihood method can be applied to this situation and allows identification of $C$ that is consistent with the common cause principle. It closely relates to the maximum entropy principle. Investigation of the two binary symmetric variables reveals a non-analytic behavior of conditional probabilities reminiscent of a second-order phase transition. This occurs during the transition from correlation to anti-correlation in the observed probability distribution. The relation between the generalized likelihood approach and alternative methods, such as predictive likelihood and the minimum common cause entropy, is discussed. The consideration of the common cause
    
[^13]: 使用深度学习在"Web of Science"中对研究领域进行层次分类

    Hierarchical Classification of Research Fields in the "Web of Science" Using Deep Learning. (arXiv:2302.00390v2 [cs.DL] UPDATED)

    [http://arxiv.org/abs/2302.00390](http://arxiv.org/abs/2302.00390)

    本文提出了一个使用深度学习进行层次分类的系统，可以自动将学术出版物通过抽象进行分类，实现了对研究活动在不同层次结构中的全面分类，并允许跨学科和跨领域的单标签和多标签分类。

    

    本文提出了一个层次分类系统，通过使用抽象将学术出版物自动分类到三级层次标签集（学科、领域、子领域）中，以多类别设置进行分类。该系统可以通过文章的知识生产和引用的影响，对研究活动在所述层次结构中进行全面的分类，并允许这些活动被归为多个类别。该分类系统在Microsoft Academic Graph (版本2018-05-17)的160 million份摘要片段中区分了44个学科、718个领域和1,485个子领域。我们以模块化和分布式方式进行批量训练，以解决和允许跨学科和跨领域的单标签和多标签分类。总共，我们在所有考虑的模型（卷积神经网络，循环神经网络，变形器）中进行了3,140次实验。分类准确率> 90％。

    This paper presents a hierarchical classification system that automatically categorizes a scholarly publication using its abstract into a three-tier hierarchical label set (discipline, field, subfield) in a multi-class setting. This system enables a holistic categorization of research activities in the mentioned hierarchy in terms of knowledge production through articles and impact through citations, permitting those activities to fall into multiple categories. The classification system distinguishes 44 disciplines, 718 fields and 1,485 subfields among 160 million abstract snippets in Microsoft Academic Graph (version 2018-05-17). We used batch training in a modularized and distributed fashion to address and allow for interdisciplinary and interfield classifications in single-label and multi-label settings. In total, we have conducted 3,140 experiments in all considered models (Convolutional Neural Networks, Recurrent Neural Networks, Transformers). The classification accuracy is > 90%
    

