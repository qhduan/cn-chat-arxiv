# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [UFID: A Unified Framework for Input-level Backdoor Detection on Diffusion Models](https://arxiv.org/abs/2404.01101) | 扩散模型容易受到后门攻击，本文提出了一个统一的框架用于输入级后门检测，弥补了该领域的空白，并不需要访问模型的白盒信息。 |
| [^2] | [Towards Stable Machine Learning Model Retraining via Slowly Varying Sequences](https://arxiv.org/abs/2403.19871) | 通过混合整数优化算法，以保持一致的分析洞见为重点，在重新训练机器学习模型中实现比贪婪训练更强稳定性，同时在模型性能上有小幅、可控的牺牲。 |
| [^3] | [Fisher-Rao Gradient Flows of Linear Programs and State-Action Natural Policy Gradients](https://arxiv.org/abs/2403.19448) | 研究了基于Fisher信息矩阵的自然梯度方法在线性规划中的应用，展示了线性收敛性，提出了改进现有结果的熵正则化误差估计，并对扰动的Fisher-Rao梯度流和自然梯度流的次线性收敛性进行了研究。 |
| [^4] | [A Semantic Search Engine for Mathlib4](https://arxiv.org/abs/2403.13310) | 提出了一个用于Mathlib4的语义搜索引擎，能够接受非正式查询并找到相关定理，为解决在mathlib4中搜索困难问题提供了新的方法。 |
| [^5] | [Statistical Mechanics of Dynamical System Identification](https://arxiv.org/abs/2403.01723) | 统计力学提供了一种稀疏方程发现算法的分析方法，通过两级贝叶斯推断问题来平衡数据拟合和简洁性，特别在低数据极限中能够量化不确定性。 |
| [^6] | [Autoencoder-based General Purpose Representation Learning for Customer Embedding](https://arxiv.org/abs/2402.18164) | 设计了基于自动编码器的框架用于构建通用嵌入，展示简单模型在嵌入复杂表格数据时优于复杂模型，并将框架应用于生成表示AWS客户的嵌入，显著节省开发时间并观察到下游模型的改进。 |
| [^7] | [World Model on Million-Length Video And Language With RingAttention](https://arxiv.org/abs/2402.08268) | 该论文介绍了一个使用百万长度的视频和语言序列进行联合建模的环形注意力世界模型。该模型通过利用视频序列中的时间信息和语言的文本知识以及逐渐增加上下文大小的方法提高了AI辅助人类的能力。 |
| [^8] | [Which Pretrain Samples to Rehearse when Finetuning Pretrained Models?](https://arxiv.org/abs/2402.08096) | 本文提出了一种新的微调预训练模型的采样方案"mix-cd"，通过识别和优先处理实际面临遗忘的样本，以缓解微调过程中的知识遗忘问题。该方法简单、易于实现，并能在现有模型中无缝运用，有效地保持预训练模型的性能。 |
| [^9] | [IDENAS: Internal Dependency Exploration for Neural Architecture Search.](http://arxiv.org/abs/2310.17250) | IDENAS是一种集成神经架构搜索和特征选择的方法，通过探索内部依赖性来提高分类任务的性能。 |
| [^10] | [Pseudo-Bayesian Optimization.](http://arxiv.org/abs/2310.09766) | 本文提出了伪贝叶斯优化，并通过研究最小要求的公理框架，构建了能确保黑盒优化收敛性的算法。 |
| [^11] | [SelfFed: Self-supervised Federated Learning for Data Heterogeneity and Label Scarcity in IoMT.](http://arxiv.org/abs/2307.01514) | 这篇论文提出了一种名为SelfFed的自监督联邦学习框架，用于解决IoMT中的数据异质性和标签匮乏问题。该框架包括预训练和微调两个阶段，通过分散训练和增强建模来克服数据异质性和标签稀缺问题。 |
| [^12] | [Broadcasting in random recursive dags.](http://arxiv.org/abs/2306.01727) | 该论文研究了一个均匀的$k$-dag广播模型，确定了与$p$和$k$有关的阈值，并讨论了大多数规则的误差率。 |

# 详细

[^1]: UFID: 一个统一的框架用于扩散模型上的输入级后门检测

    UFID: A Unified Framework for Input-level Backdoor Detection on Diffusion Models

    [https://arxiv.org/abs/2404.01101](https://arxiv.org/abs/2404.01101)

    扩散模型容易受到后门攻击，本文提出了一个统一的框架用于输入级后门检测，弥补了该领域的空白，并不需要访问模型的白盒信息。

    

    扩散模型容易受到后门攻击，即恶意攻击者在训练阶段通过对部分训练样本进行毒化来注入后门。为了减轻后门攻击的威胁，对后门检测进行了大量研究。然而，没有人为扩散模型设计了专门的后门检测方法，使得这一领域较少被探索。此外，大多数先前的方法主要集中在传统神经网络的分类任务上，很难轻松地将其适应生成任务上的后门检测。此外，大多数先前的方法需要访问模型权重和架构的白盒访问，或概率logits作为额外信息，这并不总是切实可行的。在本文中

    arXiv:2404.01101v1 Announce Type: cross  Abstract: Diffusion Models are vulnerable to backdoor attacks, where malicious attackers inject backdoors by poisoning some parts of the training samples during the training stage. This poses a serious threat to the downstream users, who query the diffusion models through the API or directly download them from the internet. To mitigate the threat of backdoor attacks, there have been a plethora of investigations on backdoor detections. However, none of them designed a specialized backdoor detection method for diffusion models, rendering the area much under-explored. Moreover, these prior methods mainly focus on the traditional neural networks in the classification task, which cannot be adapted to the backdoor detections on the generative task easily. Additionally, most of the prior methods require white-box access to model weights and architectures, or the probability logits as additional information, which are not always practical. In this paper
    
[^2]: 通过缓慢变化的序列实现稳定的机器学习模型重新训练

    Towards Stable Machine Learning Model Retraining via Slowly Varying Sequences

    [https://arxiv.org/abs/2403.19871](https://arxiv.org/abs/2403.19871)

    通过混合整数优化算法，以保持一致的分析洞见为重点，在重新训练机器学习模型中实现比贪婪训练更强稳定性，同时在模型性能上有小幅、可控的牺牲。

    

    重新训练机器学习模型仍然是实际机器学习模型部署的重要任务。现有方法主要关注贪婪方法，以找到表现最佳的模型，而不考虑通过不同的重新训练演变来保持训练模型结构的稳定性。在这项研究中，我们开发了一种混合整数优化算法，全面考虑了通过不同的数据批次更新重新训练机器学习模型的问题。我们的方法侧重于保留一致的分析洞见 - 这对于模型可解释性、实施简易性和与用户建立信任至关重要 - 通过使用可以直接纳入优化问题的自定义定义的距离度量。重要的是，我们的方法在真实的生产案例研究中表现出比贪婪训练模型更强的稳定性，同时在模型性能上有小幅、可控的牺牲。

    arXiv:2403.19871v1 Announce Type: cross  Abstract: Retraining machine learning models remains an important task for real-world machine learning model deployment. Existing methods focus largely on greedy approaches to find the best-performing model without considering the stability of trained model structures across different retraining evolutions. In this study, we develop a mixed integer optimization algorithm that holistically considers the problem of retraining machine learning models across different data batch updates. Our method focuses on retaining consistent analytical insights - which is important to model interpretability, ease of implementation, and fostering trust with users - by using custom-defined distance metrics that can be directly incorporated into the optimization problem. Importantly, our method shows stronger stability than greedily trained models with a small, controllable sacrifice in model performance in a real-world production case study. Finally, important an
    
[^3]: Fisher-Rao线性规划和状态-动作自然策略梯度的梯度流

    Fisher-Rao Gradient Flows of Linear Programs and State-Action Natural Policy Gradients

    [https://arxiv.org/abs/2403.19448](https://arxiv.org/abs/2403.19448)

    研究了基于Fisher信息矩阵的自然梯度方法在线性规划中的应用，展示了线性收敛性，提出了改进现有结果的熵正则化误差估计，并对扰动的Fisher-Rao梯度流和自然梯度流的次线性收敛性进行了研究。

    

    Kakade的自然策略梯度方法近年来得到广泛研究，表明在有或无正则化的情况下具有线性收敛性。我们研究了另一种基于状态-动作分布的Fisher信息矩阵的自然梯度方法，但在理论方面接受度较低。在这里，状态-动作分布在状态-动作多面体内遵循Fisher-Rao梯度流，相对于线性势。因此，我们更全面地研究线性规划的Fisher-Rao梯度流，并显示了线性收敛性，其速率取决于线性规划的几何特性。换句话说，这提供了线性规划的熵正则化引起的误差估计，这改进了现有结果。我们拓展了这些结果，并展示了对扰动的Fisher-Rao梯度流和自然梯度流的次线性收敛性，直到逼近误差。

    arXiv:2403.19448v1 Announce Type: cross  Abstract: Kakade's natural policy gradient method has been studied extensively in the last years showing linear convergence with and without regularization. We study another natural gradient method which is based on the Fisher information matrix of the state-action distributions and has received little attention from the theoretical side. Here, the state-action distributions follow the Fisher-Rao gradient flow inside the state-action polytope with respect to a linear potential. Therefore, we study Fisher-Rao gradient flows of linear programs more generally and show linear convergence with a rate that depends on the geometry of the linear program. Equivalently, this yields an estimate on the error induced by entropic regularization of the linear program which improves existing results. We extend these results and show sublinear convergence for perturbed Fisher-Rao gradient flows and natural gradient flows up to an approximation error. In particul
    
[^4]: 一个用于Mathlib4的语义搜索引擎

    A Semantic Search Engine for Mathlib4

    [https://arxiv.org/abs/2403.13310](https://arxiv.org/abs/2403.13310)

    提出了一个用于Mathlib4的语义搜索引擎，能够接受非正式查询并找到相关定理，为解决在mathlib4中搜索困难问题提供了新的方法。

    

    交互式定理证明器Lean使得可以验证正式数学证明，并且得到一个不断扩大的社区的支持。该生态系统的核心是其数学库mathlib4，为扩展范围的数学理论的形式化奠定了基础。然而，在mathlib4中搜索定理可能具有挑战性。为了成功在mathlib4中搜索，用户通常需要熟悉其命名约定或文档字符串。因此，创建一个语义搜索引擎，可以方便地被具有不同熟悉程度的mathlib4的个人使用是非常重要的。在本文中，我们提出了一个用于mathlib4的语义搜索引擎，可以接受非正式查询并找到相关定理。我们还建立了一个用于评估各种mathlib4搜索引擎性能的基准。

    arXiv:2403.13310v1 Announce Type: cross  Abstract: The interactive theorem prover, Lean, enables the verification of formal mathematical proofs and is backed by an expanding community. Central to this ecosystem is its mathematical library, mathlib4, which lays the groundwork for the formalization of an expanding range of mathematical theories. However, searching for theorems in mathlib4 can be challenging. To successfully search in mathlib4, users often need to be familiar with its naming conventions or documentation strings. Therefore, creating a semantic search engine that can be used easily by individuals with varying familiarity with mathlib4 is very important. In this paper, we present a semantic search engine for mathlib4 that accepts informal queries and finds the relevant theorems. We also establish a benchmark for assessing the performance of various search engines for mathlib4.
    
[^5]: 动力学系统识别的统计力学

    Statistical Mechanics of Dynamical System Identification

    [https://arxiv.org/abs/2403.01723](https://arxiv.org/abs/2403.01723)

    统计力学提供了一种稀疏方程发现算法的分析方法，通过两级贝叶斯推断问题来平衡数据拟合和简洁性，特别在低数据极限中能够量化不确定性。

    

    从观测到的噪声数据中恢复动力学方程是系统识别的核心挑战。我们发展了一种统计力学方法来分析稀疏方程发现算法，这些算法通常通过对超参数的试错选择平衡数据拟合和简洁性。在这个框架中，统计力学提供了分析复杂性和适应性之间相互作用的工具，类似于熵与能量之间的分析。为了建立这种类比，我们将优化过程定义为一个将变量选择与系数值分开的两级贝叶斯推断问题，并使得后验参数分布可以以闭式形式计算。采用统计力学概念（如自由能和配分函数）的一个关键优势在于在低数据极限中量化不确定性，这在真实世界应用中经常遇到。

    arXiv:2403.01723v1 Announce Type: cross  Abstract: Recovering dynamical equations from observed noisy data is the central challenge of system identification. We develop a statistical mechanical approach to analyze sparse equation discovery algorithms, which typically balance data fit and parsimony through a trial-and-error selection of hyperparameters. In this framework, statistical mechanics offers tools to analyze the interplay between complexity and fitness, in analogy to that done between entropy and energy. To establish this analogy, we define the optimization procedure as a two-level Bayesian inference problem that separates variable selection from coefficient values and enables the computation of the posterior parameter distribution in closed form. A key advantage of employing statistical mechanical concepts, such as free energy and the partition function, is in the quantification of uncertainty, especially in in the low-data limit; frequently encountered in real-world applicati
    
[^6]: 基于自动编码器的通用表示学习用于客户嵌入

    Autoencoder-based General Purpose Representation Learning for Customer Embedding

    [https://arxiv.org/abs/2402.18164](https://arxiv.org/abs/2402.18164)

    设计了基于自动编码器的框架用于构建通用嵌入，展示简单模型在嵌入复杂表格数据时优于复杂模型，并将框架应用于生成表示AWS客户的嵌入，显著节省开发时间并观察到下游模型的改进。

    

    最近几年，利用数据的领域特定基础结构及其生成因素进行表示学习，在各种用例无关应用中取得成功。然而，表格数据的多样性和复杂性使得通过多维向量在潜在空间中表示这些结构具有挑战性。我们设计了一个基于自动编码器的框架用于构建通用嵌入，评估了不同自动编码器架构的性能，并展示了简单模型在嵌入高度复杂表格数据时优于复杂模型。我们将我们的框架应用于生成插拔式、丰富和匿名化的表示AWS客户的嵌入，可用于任何模型，节省开发时间高达45％，并观察到下游模型的显著改进。此外，我们提出了一种对于多层收缩自动编码器重构损失计算的重要改进。

    arXiv:2402.18164v1 Announce Type: cross  Abstract: In recent years, exploiting the domain-specific underlying structure of data and its generative factors for representation learning has shown success in various use-case agnostic applications. However, the diversity and complexity of tabular data have made it challenging to represent these structures in a latent space through multi-dimensional vectors. We design an autoencoder-based framework for building general purpose embeddings, we assess the performance of different autoencoder architectures, and show simpler models outperform complex ones in embedding highly complex tabular data. We apply our framework to produce plug-and-play, rich, and anonymized embeddings representing AWS customers for usage in any model, saving up to 45% of development time, and observe significant improvements in downstream models. Moreover, we propose a significant improvement to the calculation of reconstruction loss for multi-layer contractive autoencode
    
[^7]: 百万长度视频和语言的环形注意力世界模型

    World Model on Million-Length Video And Language With RingAttention

    [https://arxiv.org/abs/2402.08268](https://arxiv.org/abs/2402.08268)

    该论文介绍了一个使用百万长度的视频和语言序列进行联合建模的环形注意力世界模型。该模型通过利用视频序列中的时间信息和语言的文本知识以及逐渐增加上下文大小的方法提高了AI辅助人类的能力。

    

    当前的语言模型在理解难以用文字描述的世界方面表现不佳，并且在处理复杂的长篇任务时遇到困难。视频序列提供了只有语言和静态图像所不具备的宝贵时间信息，因此它们在与语言进行联合建模时具有吸引力。这种模型可以对人类的文本知识和物理世界进行理解，为辅助人类提供更广泛的人工智能能力。然而，从百万个标记的视频和语言序列中学习面临着记忆约束、计算复杂性和数据有限性的挑战。为了应对这些挑战，我们策划了一个包含多样化视频和书籍的大型数据集，利用环形注意力技术对长序列进行可扩展的训练，逐渐增加上下文大小从4K到1M个标记。本文的贡献如下：

    Current language models fall short in understanding aspects of the world not easily described in words, and struggle with complex, long-form tasks. Video sequences offer valuable temporal information absent in language and static images, making them attractive for joint modeling with language. Such models could develop a understanding of both human textual knowledge and the physical world, enabling broader AI capabilities for assisting humans. However, learning from millions of tokens of video and language sequences poses challenges due to memory constraints, computational complexity, and limited datasets. To address these challenges, we curate a large dataset of diverse videos and books, utilize the RingAttention technique to scalably train on long sequences, and gradually increase context size from 4K to 1M tokens. This paper makes the following contributions: (a) Largest context size neural network: We train one of the largest context size transformers on long video and language seq
    
[^8]: 在微调预训练模型时重新练习哪些预训练样本更好？

    Which Pretrain Samples to Rehearse when Finetuning Pretrained Models?

    [https://arxiv.org/abs/2402.08096](https://arxiv.org/abs/2402.08096)

    本文提出了一种新的微调预训练模型的采样方案"mix-cd"，通过识别和优先处理实际面临遗忘的样本，以缓解微调过程中的知识遗忘问题。该方法简单、易于实现，并能在现有模型中无缝运用，有效地保持预训练模型的性能。

    

    在文本和视觉任务中，微调预训练基础模型已成为事实上的方法。这种方法的一个已知问题是在微调过程中会遗忘预训练知识。从预训练数据集中随机选择样本来进行重新练习是缓解遗忘的常见方法。然而，我们发现随机混合不经意地包括了模型尚未遗忘或无法学习的样本。我们提出了一种新的采样方案"mix-cd"，用于识别和优先处理实际面临遗忘的样本，我们称之为"collateral damage"。由于直接识别"collateral damage"样本计算成本高昂，我们提出了一种通过跟踪微调样本的统计信息来估计这类样本分布的过程。我们的方法简洁轻量，易于实现，并可以无缝集成到现有模型中，具有有效地保持预训练性能而无需额外计算开销的能力。

    Fine-tuning pretrained foundational models on specific tasks is now the de facto approach for text and vision tasks. A known pitfall of this approach is the forgetting of pretraining knowledge that happens during finetuning. Rehearsing samples randomly from the pretrain dataset is a common approach to alleviate such forgetting. However, we find that random mixing unintentionally includes samples which are not (yet) forgotten or unlearnable by the model. We propose a novel sampling scheme, mix-cd, that identifies and prioritizes samples that actually face forgetting, which we call collateral damage. Since directly identifying collateral damage samples is computationally expensive, we propose a procedure to estimate the distribution of such samples by tracking the statistics of finetuned samples. Our approach is lightweight, easy to implement, and can be seamlessly integrated into existing models, offering an effective means to retain pretrain performance without additional computational
    
[^9]: IDENAS: 内部依赖性探索用于神经架构搜索

    IDENAS: Internal Dependency Exploration for Neural Architecture Search. (arXiv:2310.17250v1 [cs.LG])

    [http://arxiv.org/abs/2310.17250](http://arxiv.org/abs/2310.17250)

    IDENAS是一种集成神经架构搜索和特征选择的方法，通过探索内部依赖性来提高分类任务的性能。

    

    机器学习是从不同数据集中提取有价值信息和进行各种预测的强大工具。传统算法依赖于明确定义的输入和输出变量，然而，在某些情况下，输入和输出变量之间的区别以及模型的底层关联（输入和输出）层是未知的。神经架构搜索（NAS）和特征选择已成为这些场景中的有希望的解决方案。该研究提出了IDENAS，一种基于内部依赖性的神经架构搜索方法，将NAS与特征选择相结合。该方法在涉及1D传感器和2D图像数据的分类问题中探索了完整的参数空间的内部依赖性。IDENAS采用了修改的编码器-解码器模型和顺序前向搜索（SFS）算法，将输入-输出配置搜索与嵌入式特征选择相结合。实验结果证明了IDENAS的优越性能。

    Machine learning is a powerful tool for extracting valuable information and making various predictions from diverse datasets. Traditional algorithms rely on well-defined input and output variables however, there are scenarios where the distinction between the input and output variables and the underlying, associated (input and output) layers of the model, are unknown. Neural Architecture Search (NAS) and Feature Selection have emerged as promising solutions in such scenarios. This research proposes IDENAS, an Internal Dependency-based Exploration for Neural Architecture Search, integrating NAS with feature selection. The methodology explores internal dependencies in the complete parameter space for classification involving 1D sensor and 2D image data as well. IDENAS employs a modified encoder-decoder model and the Sequential Forward Search (SFS) algorithm, combining input-output configuration search with embedded feature selection. Experimental results demonstrate IDENASs superior perf
    
[^10]: 伪贝叶斯优化

    Pseudo-Bayesian Optimization. (arXiv:2310.09766v1 [stat.ML])

    [http://arxiv.org/abs/2310.09766](http://arxiv.org/abs/2310.09766)

    本文提出了伪贝叶斯优化，并通过研究最小要求的公理框架，构建了能确保黑盒优化收敛性的算法。

    

    贝叶斯优化是一种优化昂贵黑盒函数的流行方法。其关键思想是使用一个替代模型来近似目标，并且重要的是量化相关的不确定性，从而实现探索和开发之间的平衡的顺序搜索。高斯过程(GP)一直是替代模型的首选，因为它具有贝叶斯的不确定性量化能力和建模灵活性。然而，它的挑战也引发了一系列收敛性更显得不明显的备选方案。在本文中，我们通过研究引出最小要求的公理框架来确保黑盒优化的收敛性，以应用于除了GP相关方法之外的情况。此外，我们利用我们的框架中的设计自由，我们称之为伪贝叶斯优化，来构建经验上更优的算法。特别地，我们展示了如何使用简单的局部回归和一个适应问题特性的代理模型来实现这一目标。

    Bayesian Optimization is a popular approach for optimizing expensive black-box functions. Its key idea is to use a surrogate model to approximate the objective and, importantly, quantify the associated uncertainty that allows a sequential search of query points that balance exploitation-exploration. Gaussian process (GP) has been a primary candidate for the surrogate model, thanks to its Bayesian-principled uncertainty quantification power and modeling flexibility. However, its challenges have also spurred an array of alternatives whose convergence properties could be more opaque. Motivated by these, we study in this paper an axiomatic framework that elicits the minimal requirements to guarantee black-box optimization convergence that could apply beyond GP-related methods. Moreover, we leverage the design freedom in our framework, which we call Pseudo-Bayesian Optimization, to construct empirically superior algorithms. In particular, we show how using simple local regression, and a sui
    
[^11]: SelfFed: 自监督的联邦学习用于IoMT中的数据异质性和标签匮乏问题

    SelfFed: Self-supervised Federated Learning for Data Heterogeneity and Label Scarcity in IoMT. (arXiv:2307.01514v1 [cs.LG])

    [http://arxiv.org/abs/2307.01514](http://arxiv.org/abs/2307.01514)

    这篇论文提出了一种名为SelfFed的自监督联邦学习框架，用于解决IoMT中的数据异质性和标签匮乏问题。该框架包括预训练和微调两个阶段，通过分散训练和增强建模来克服数据异质性和标签稀缺问题。

    

    基于自监督学习的联邦学习范式在行业和研究领域中引起了很大的兴趣，因为它可以协作学习未标记但孤立的数据。然而，自监督的联邦学习策略在标签稀缺和数据异质性（即数据分布不同）方面存在性能下降的问题。在本文中，我们提出了适用于医疗物联网（IoMT）的SelfFed框架。我们的SelfFed框架分为两个阶段。第一个阶段是预训练范式，使用基于Swin Transformer的编码器以分散的方式进行增强建模。SelfFed框架的第一个阶段有助于克服数据异质性问题。第二个阶段是微调范式，引入对比网络和一种在有限标记数据上进行训练的新型聚合策略，用于目标任务的分散训练。这个微调阶段克服了标签稀缺问题。

    Self-supervised learning in federated learning paradigm has been gaining a lot of interest both in industry and research due to the collaborative learning capability on unlabeled yet isolated data. However, self-supervised based federated learning strategies suffer from performance degradation due to label scarcity and diverse data distributions, i.e., data heterogeneity. In this paper, we propose the SelfFed framework for Internet of Medical Things (IoMT). Our proposed SelfFed framework works in two phases. The first phase is the pre-training paradigm that performs augmentive modeling using Swin Transformer based encoder in a decentralized manner. The first phase of SelfFed framework helps to overcome the data heterogeneity issue. The second phase is the fine-tuning paradigm that introduces contrastive network and a novel aggregation strategy that is trained on limited labeled data for a target task in a decentralized manner. This fine-tuning stage overcomes the label scarcity problem
    
[^12]: 在随机递归有向无环图中的广播

    Broadcasting in random recursive dags. (arXiv:2306.01727v1 [stat.ML])

    [http://arxiv.org/abs/2306.01727](http://arxiv.org/abs/2306.01727)

    该论文研究了一个均匀的$k$-dag广播模型，确定了与$p$和$k$有关的阈值，并讨论了大多数规则的误差率。

    

    一个均匀的$k$-dag通过从现有节点中均匀随机选择$k$个父节点来推广均匀的随机递归树。它以$k$个“根”开始。每个$k$个根节点都被分配一个位。这些位通过一个嘈杂的信道传播。每个父节点的位都以概率$p$发生变化，并进行大多数表决。当所有节点都接收到它们的位后，$k$-dag被显示，不识别根节点。目标是估计所有根节点中的大多数位。我们确定了$p$的阈值，作为一个关于$k$的函数，使得所有节点的大多数规则产生错误$c+o(1)$的概率小于$1/2$。在阈值以上，大多数规则的错误概率为$1/2+o(1)$。

    A uniform $k$-{\sc dag} generalizes the uniform random recursive tree by picking $k$ parents uniformly at random from the existing nodes. It starts with $k$ ''roots''. Each of the $k$ roots is assigned a bit. These bits are propagated by a noisy channel. The parents' bits are flipped with probability $p$, and a majority vote is taken. When all nodes have received their bits, the $k$-{\sc dag} is shown without identifying the roots. The goal is to estimate the majority bit among the roots. We identify the threshold for $p$ as a function of $k$ below which the majority rule among all nodes yields an error $c+o(1)$ with $c<1/2$. Above the threshold the majority rule errs with probability $1/2+o(1)$.
    

