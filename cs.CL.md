# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [3M-TRANSFORMER: A Multi-Stage Multi-Stream Multimodal Transformer for Embodied Turn-Taking Prediction.](http://arxiv.org/abs/2310.14859) | 本文提出了一种用于预测体验式转换的多阶段多流多模态Transformer，通过对同步的多角度自我中心数据进行处理，相较于现有的基准模型和其他基于Transformer的方法，实现了14.01%的性能提升。 |
| [^2] | [FedJudge: Federated Legal Large Language Model.](http://arxiv.org/abs/2309.08173) | 本文提出了第一个分布式法律大型语言模型（FedJudge）框架，可以通过在设备或客户端上进行本地微调，并将参数聚合和分布在中央服务器上来确保数据隐私。这解决了集中式训练法律LLMs引发的数据隐私问题和分布偏移导致的FL方法效果降低的挑战。 |
| [^3] | [Are ChatGPT and GPT-4 Good Poker Players? -- A Pre-Flop Analysis.](http://arxiv.org/abs/2308.12466) | ChatGPT和GPT-4在扑克中显示出高级理解，但不是游戏论理最优的扑克玩家。对模型参数和提示的优化可以提高它们在扑克中的表现。 |
| [^4] | [Universal and Transferable Adversarial Attacks on Aligned Language Models.](http://arxiv.org/abs/2307.15043) | 这项研究提出了一种简单而有效的攻击方法，能够使对齐的语言模型生成不良行为，而不依赖于人工设计，通过自动化方法产生对抗性后缀，并在实践中取得改进。 |
| [^5] | [Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers.](http://arxiv.org/abs/2307.14367) | 提出了一种名为Prot2Text的新方法，通过结合GNNs和Transformers，以自由文本样式预测蛋白质的功能。该方法能够综合蛋白质的序列、结构和文本注释等多种数据类型，超越传统的二进制或分类分类，实现了对蛋白质功能的全面表示。 |
| [^6] | [Layer-wise Representation Fusion for Compositional Generalization.](http://arxiv.org/abs/2307.10799) | 该论文提出了一种层级表示融合的方法，以提升序列到序列模型在组合泛化方面的表现。之前的研究主要关注增强基于令牌的语义信息，而本文提出了在人类那样适当地组合和使用序列的句法和语义表示的方法。此外，从近期的关于训练更深Transformer的研究结果来看，纠缠问题主要是由于残差连接的“浅层”和简单的单步操作导致不能有效地融合前面层的信息。 |
| [^7] | [Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations.](http://arxiv.org/abs/2307.05722) | 本论文探索了大规模语言模型在在线职位推荐中对图数据的理解能力，并提出了新的框架来分析行为图，发现其中的潜在模式和关系。 |
| [^8] | [BloombergGPT: A Large Language Model for Finance.](http://arxiv.org/abs/2303.17564) | 本文提出了BloombergGPT，一个500亿参数的金融领域的大型语言模型，其基于Bloomberg的广泛数据来源和通用数据集进行训练。通过混合数据集训练，该模型在金融任务上表现出色，并且不会牺牲在普通任务上的性能。 |
| [^9] | [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.](http://arxiv.org/abs/2303.10512) | AdaLoRA是一种自适应预算分配方法，用于参数效率微调。将增量更新的预算根据权重矩阵的重要性分数进行自适应分配，通过奇异值分解的形式，实现了微调表现的优化。 |
| [^10] | [Reducing Spurious Correlations for Aspect-Based Sentiment Analysis with Variational Information Bottleneck and Contrastive Learning.](http://arxiv.org/abs/2303.02846) | 本文提出了一种新的对比变分信息瓶颈框架（CVIB），以减少方面情感分析（ABSA）中的虚假相关性。该框架由一个原始网络和一个自剪枝网络组成，通过对比学习同时进行优化，从而丢弃了输入特征和预测标签之间的多余模式或虚假相关性。 |

# 详细

[^1]: 3M-TRANSFORMER：一种用于体验式转换预测的多阶段多流多模态Transformer

    3M-TRANSFORMER: A Multi-Stage Multi-Stream Multimodal Transformer for Embodied Turn-Taking Prediction. (arXiv:2310.14859v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.14859](http://arxiv.org/abs/2310.14859)

    本文提出了一种用于预测体验式转换的多阶段多流多模态Transformer，通过对同步的多角度自我中心数据进行处理，相较于现有的基准模型和其他基于Transformer的方法，实现了14.01%的性能提升。

    

    在多方对话中预测交替对话在人机/机器人交互中具有很多实际应用。然而，人类沟通的复杂性使这成为一项具有挑战性的任务。最近的研究进展表明，同步的多角度自我中心数据可以显著提高与异步的单角度转录相比的交替对话预测能力。基于这项研究，我们提出了一种基于多模态Transformer的新架构，用于预测体验式的、同步的多角度数据中的交替对话。我们在最近引入的EgoCom数据集上的实验结果显示，与现有基准线和其他基于Transformer的方法相比，我们的3M-Transformer平均性能提高了14.01%。我们的3M-Transformer的源代码和预训练模型将在被接受后提供。

    Predicting turn-taking in multiparty conversations has many practical applications in human-computer/robot interaction. However, the complexity of human communication makes it a challenging task. Recent advances have shown that synchronous multi-perspective egocentric data can significantly improve turn-taking prediction compared to asynchronous, single-perspective transcriptions. Building on this research, we propose a new multimodal transformer-based architecture for predicting turn-taking in embodied, synchronized multi-perspective data. Our experimental results on the recently introduced EgoCom dataset show a substantial performance improvement of up to 14.01% on average compared to existing baselines and alternative transformer-based approaches. The source code, and the pre-trained models of our 3M-Transformer will be available upon acceptance.
    
[^2]: FedJudge: 分布式法律大型语言模型

    FedJudge: Federated Legal Large Language Model. (arXiv:2309.08173v1 [cs.CL])

    [http://arxiv.org/abs/2309.08173](http://arxiv.org/abs/2309.08173)

    本文提出了第一个分布式法律大型语言模型（FedJudge）框架，可以通过在设备或客户端上进行本地微调，并将参数聚合和分布在中央服务器上来确保数据隐私。这解决了集中式训练法律LLMs引发的数据隐私问题和分布偏移导致的FL方法效果降低的挑战。

    

    大型语言模型（LLMs）在法律智能领域得到了广泛应用，可以辅助法律专业人员和普通人。然而，这些法律LLMs的集中式训练引发了数据隐私问题，因为法律数据分散在包含敏感个人信息的各个机构之间。本文通过探索将法律LLMs与分布式学习（FL）方法相结合来解决这一挑战。通过使用FL，法律LLMs可以在设备或客户端上进行本地微调，其参数被聚合并分布在中央服务器上，确保数据隐私而无需直接共享原始数据。然而，计算和通信开销阻碍了LLMs在FL环境中的全面微调。此外，法律数据的分布偏移减少了FL方法的有效性。为此，在本文中，我们提出了第一个分布式法律大型语言模型（FedJudge）框架，可以对LLMs进行微调。

    Large Language Models (LLMs) have gained prominence in the field of Legal Intelligence, offering potential applications in assisting legal professionals and laymen. However, the centralized training of these Legal LLMs raises data privacy concerns, as legal data is distributed among various institutions containing sensitive individual information. This paper addresses this challenge by exploring the integration of Legal LLMs with Federated Learning (FL) methodologies. By employing FL, Legal LLMs can be fine-tuned locally on devices or clients, and their parameters are aggregated and distributed on a central server, ensuring data privacy without directly sharing raw data. However, computation and communication overheads hinder the full fine-tuning of LLMs under the FL setting. Moreover, the distribution shift of legal data reduces the effectiveness of FL methods. To this end, in this paper, we propose the first Federated Legal Large Language Model (FedJudge) framework, which fine-tunes 
    
[^3]: ChatGPT和GPT-4是优秀的扑克玩家吗？——一项Pre-Flop分析。

    Are ChatGPT and GPT-4 Good Poker Players? -- A Pre-Flop Analysis. (arXiv:2308.12466v1 [cs.CL])

    [http://arxiv.org/abs/2308.12466](http://arxiv.org/abs/2308.12466)

    ChatGPT和GPT-4在扑克中显示出高级理解，但不是游戏论理最优的扑克玩家。对模型参数和提示的优化可以提高它们在扑克中的表现。

    

    自ChatGPT和GPT-4问世以来，这些模型已在许多任务中进行了测试。它们在各个领域的熟练程度是显而易见的，但它们在游戏中的能力，特别是在扑克领域的能力，还未被探索。扑克是一种需要在不确定性和不完全信息下做出决策的游戏。在本文中，我们对ChatGPT和GPT-4进行了扑克测试，并评估了它们的扑克技能。我们的研究结果显示，虽然这两个模型都展示了对扑克的高级理解，包括起始手牌的估值、打牌位置以及游戏论理最优(GTO)扑克的其他复杂性，但ChatGPT和GPT-4并不是游戏论理最优的扑克玩家。通过一系列实验，我们首先发现了与使用这些模型玩扑克相关的最佳提示和模型参数的特征。接着，我们观察到了这两个模型具有不同的打牌风格。最终，我们得出结论：GPT-4是

    Since the introduction of ChatGPT and GPT-4, these models have been tested across a large number of tasks. Their adeptness across domains is evident, but their aptitude in playing games and specifically their aptitude in the realm of poker has remained unexplored. Poker is a game that requires decision making under uncertainty and incomplete information. In this paper, we put ChatGPT and GPT-4 through the poker test and evaluate their poker skills. Our findings reveal that while both models display an advanced understanding of poker, encompassing concepts like the valuation of starting hands, playing positions and other intricacies of game theory optimal (GTO) poker, both ChatGPT and GPT-4 are NOT game theory optimal poker players.  Through a series of experiments, we first discover the characteristics of optimal prompts and model parameters for playing poker with these models. Our observations then unveil the distinct playing personas of the two models. We first conclude that GPT-4 is
    
[^4]: 对齐语言模型上的通用和可迁移对抗攻击

    Universal and Transferable Adversarial Attacks on Aligned Language Models. (arXiv:2307.15043v1 [cs.CL])

    [http://arxiv.org/abs/2307.15043](http://arxiv.org/abs/2307.15043)

    这项研究提出了一种简单而有效的攻击方法，能够使对齐的语言模型生成不良行为，而不依赖于人工设计，通过自动化方法产生对抗性后缀，并在实践中取得改进。

    

    由于“开箱即用”的大型语言模型能够生成大量引起反感的内容，最新的研究专注于对齐这些模型，以防止产生不良生成。尽管在规避这些措施上取得了一些成功，所谓的对LLMs的“越狱”攻击，但这些攻击需要人为的巧思，实际上并不稳定。在本文中，我们提出了一种简单而有效的攻击方法，使对齐的语言模型生成不良行为。具体而言，我们的方法找到一个后缀，当附加到各种查询上，供LLM生成不良内容时，旨在最大化模型产生肯定回答（而不是拒绝回答）的概率。然而，与其依赖手工设计，我们的方法通过贪婪和基于梯度的搜索技术自动产生这些对抗性后缀，并且在过去的自动化方法上进行了改进。

    Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past autom
    
[^5]: Prot2Text: 基于GNNs和Transformers的多模态蛋白质功能生成

    Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers. (arXiv:2307.14367v1 [q-bio.QM])

    [http://arxiv.org/abs/2307.14367](http://arxiv.org/abs/2307.14367)

    提出了一种名为Prot2Text的新方法，通过结合GNNs和Transformers，以自由文本样式预测蛋白质的功能。该方法能够综合蛋白质的序列、结构和文本注释等多种数据类型，超越传统的二进制或分类分类，实现了对蛋白质功能的全面表示。

    

    大型生物系统的复杂性使某些科学家将其理解归类为难以想象的任务。不同级别的挑战使这项任务复杂化，其中之一是预测蛋白质的功能。近年来，通过开发各种机器学习方法，在这个领域取得了重大进展。然而，大多数现有的方法将任务表述为多分类问题，即将预定义标签分配给蛋白质。在这项工作中，我们提出了一种新的方法——Prot2Text，以自由文本样式预测蛋白质的功能，超越传统的二进制或分类分类。通过在编码器-解码器框架中结合图神经网络（GNNs）和大型语言模型（LLMs），我们的模型有效地整合了蛋白质序列、结构和文本注释等多种数据类型。这种多模态方法允许对蛋白质功能进行整体表示。

    The complex nature of big biological systems pushed some scientists to classify its understanding under the inconceivable missions. Different leveled challenges complicated this task, one of is the prediction of a protein's function. In recent years, significant progress has been made in this field through the development of various machine learning approaches. However, most existing methods formulate the task as a multi-classification problem, i.e assigning predefined labels to proteins. In this work, we propose a novel approach, \textbf{Prot2Text}, which predicts a protein function's in a free text style, moving beyond the conventional binary or categorical classifications. By combining Graph Neural Networks(GNNs) and Large Language Models(LLMs), in an encoder-decoder framework, our model effectively integrates diverse data types including proteins' sequences, structures, and textual annotations. This multimodal approach allows for a holistic representation of proteins' functions, en
    
[^6]: Layer-wise Representation Fusion for Compositional Generalization. (arXiv:2307.10799v1 [cs.CL])

    Layer-wise Representation Fusion for Compositional Generalization. (arXiv:2307.10799v1 [cs.CL])

    [http://arxiv.org/abs/2307.10799](http://arxiv.org/abs/2307.10799)

    该论文提出了一种层级表示融合的方法，以提升序列到序列模型在组合泛化方面的表现。之前的研究主要关注增强基于令牌的语义信息，而本文提出了在人类那样适当地组合和使用序列的句法和语义表示的方法。此外，从近期的关于训练更深Transformer的研究结果来看，纠缠问题主要是由于残差连接的“浅层”和简单的单步操作导致不能有效地融合前面层的信息。

    

    尽管序列到序列模型在广泛的应用中取得了成功，但其构建的解决方案被认为在组合泛化方面不如人类。越来越多的证据表明，阻碍组合泛化的一个原因是编码器和解码器最上层的表示被纠缠在一起。换句话说，序列的句法和语义表示被不适当地扭曲了。然而，大多数以前的研究主要集中于增强基于令牌的语义信息，以缓解表示纠缠问题，而不是像人类那样适当地组合和使用序列的句法和语义表示。此外，我们从近期关于训练更深Transformer的研究的角度解释了为什么纠缠问题存在，主要是由于“浅层”残差连接和其简单的单步操作导致无法有效地融合前面层的信息。

    Despite successes across a broad range of applications, sequence-to-sequence models' construct of solutions are argued to be less compositional than human-like generalization. There is mounting evidence that one of the reasons hindering compositional generalization is representations of the encoder and decoder uppermost layer are entangled. In other words, the syntactic and semantic representations of sequences are twisted inappropriately. However, most previous studies mainly concentrate on enhancing token-level semantic information to alleviate the representations entanglement problem, rather than composing and using the syntactic and semantic representations of sequences appropriately as humans do. In addition, we explain why the entanglement problem exists from the perspective of recent studies about training deeper Transformer, mainly owing to the ``shallow'' residual connections and its simple, one-step operations, which fails to fuse previous layers' information effectively. Sta
    
[^7]: 探索大规模语言模型在在线职位推荐中对图数据的理解

    Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations. (arXiv:2307.05722v1 [cs.AI])

    [http://arxiv.org/abs/2307.05722](http://arxiv.org/abs/2307.05722)

    本论文探索了大规模语言模型在在线职位推荐中对图数据的理解能力，并提出了新的框架来分析行为图，发现其中的潜在模式和关系。

    

    大规模语言模型（LLMs）在各个领域展示了其出色的能力，彻底改变了自然语言处理任务。然而，它们在职位推荐中对行为图的理解潜力仍然未被充分探索。本文旨在揭示大规模语言模型在理解行为图方面的能力，并利用这种理解来提升在线招聘中的推荐，包括促进非分布式的应用。我们提出了一个新的框架，利用大规模语言模型提供的丰富上下文信息和语义表示来分析行为图并揭示其中的潜在模式和关系。具体而言，我们提出了一个元路径提示构造器，利用LLM推荐器首次理解行为图，并设计了相应的路径增强模块来缓解基于路径的序列输入引入的提示偏差。通过利用将LM的特点引入到行为图的大规模数据分析中，我们取得了显著的实验结果，证明了我们提出的方法的有效性和性能。

    Large Language Models (LLMs) have revolutionized natural language processing tasks, demonstrating their exceptional capabilities in various domains. However, their potential for behavior graph understanding in job recommendations remains largely unexplored. This paper focuses on unveiling the capability of large language models in understanding behavior graphs and leveraging this understanding to enhance recommendations in online recruitment, including the promotion of out-of-distribution (OOD) application. We present a novel framework that harnesses the rich contextual information and semantic representations provided by large language models to analyze behavior graphs and uncover underlying patterns and relationships. Specifically, we propose a meta-path prompt constructor that leverages LLM recommender to understand behavior graphs for the first time and design a corresponding path augmentation module to alleviate the prompt bias introduced by path-based sequence input. By leveragin
    
[^8]: BloombergGPT：金融领域的大型语言模型

    BloombergGPT: A Large Language Model for Finance. (arXiv:2303.17564v1 [cs.LG])

    [http://arxiv.org/abs/2303.17564](http://arxiv.org/abs/2303.17564)

    本文提出了BloombergGPT，一个500亿参数的金融领域的大型语言模型，其基于Bloomberg的广泛数据来源和通用数据集进行训练。通过混合数据集训练，该模型在金融任务上表现出色，并且不会牺牲在普通任务上的性能。

    

    自然语言处理在金融技术领域有着广泛而复杂的应用，从情感分析和命名实体识别到问答。大型语言模型（LLM）已被证明在各种任务上非常有效；然而，专为金融领域设计的LLM尚未在文献中报告。在本文中，我们提出了BloombergGPT，一个拥有500亿个参数的语言模型，它是基于广泛的金融数据进行训练的。我们构建了一种3630亿个标记的数据集，该数据集基于彭博社的广泛数据来源，可能是迄今最大的领域特定数据集，同时又增加了来自通用数据集的3450亿个标记。我们在标准LLM基准、开放式金融基准和一套最能准确反映我们预期用途的内部基准上验证了BloombergGPT。我们的混合数据集训练产生了一个在金融任务上明显优于现有模型的模型，同时不会牺牲普通任务的性能。

    The use of NLP in the realm of financial technology is broad and complex, with applications ranging from sentiment analysis and named entity recognition to question answering. Large Language Models (LLMs) have been shown to be effective on a variety of tasks; however, no LLM specialized for the financial domain has been reported in literature. In this work, we present BloombergGPT, a 50 billion parameter language model that is trained on a wide range of financial data. We construct a 363 billion token dataset based on Bloomberg's extensive data sources, perhaps the largest domain-specific dataset yet, augmented with 345 billion tokens from general purpose datasets. We validate BloombergGPT on standard LLM benchmarks, open financial benchmarks, and a suite of internal benchmarks that most accurately reflect our intended usage. Our mixed dataset training leads to a model that outperforms existing models on financial tasks by significant margins without sacrificing performance on general 
    
[^9]: 参数效率微调的自适应预算分配

    Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. (arXiv:2303.10512v1 [cs.CL])

    [http://arxiv.org/abs/2303.10512](http://arxiv.org/abs/2303.10512)

    AdaLoRA是一种自适应预算分配方法，用于参数效率微调。将增量更新的预算根据权重矩阵的重要性分数进行自适应分配，通过奇异值分解的形式，实现了微调表现的优化。

    

    在自然语言处理中，对预训练的大型语言模型进行微调已经成为了一种重要的范式。然而，通常的做法是微调预训练模型中的所有参数，当存在大量下游任务时，这种方法变得不切实际。因此，许多微调方法被提出来以以参数有效的方式学习预训练加权的增量更新，例如低秩增量。这些方法通常将增量更新的预算均匀分配到所有预训练的权重矩阵上，忽略了不同权重参数的不同重要性。结果，微调的表现是次优的。为弥补这一差距，我们提出了AdaLoRA，根据它们的重要性分数自适应分配权重矩阵的参数预算。特别地，AdaLoRA将增量更新的参数化为奇异值分解的形式。这种新颖的方法使我们可以有效地剪枝奇异值。

    Fine-tuning large pre-trained language models on downstream tasks has become an important paradigm in NLP. However, common practice fine-tunes all of the parameters in a pre-trained model, which becomes prohibitive when a large number of downstream tasks are present. Therefore, many fine-tuning methods are proposed to learn incremental updates of pre-trained weights in a parameter efficient way, e.g., low-rank increments. These methods often evenly distribute the budget of incremental updates across all pre-trained weight matrices, and overlook the varying importance of different weight parameters. As a consequence, the fine-tuning performance is suboptimal. To bridge this gap, we propose AdaLoRA, which adaptively allocates the parameter budget among weight matrices according to their importance score. In particular, AdaLoRA parameterizes the incremental updates in the form of singular value decomposition. Such a novel approach allows us to effectively prune the singular values of unim
    
[^10]: 通过变分信息瓶颈和对比学习减少方面情感分析中的虚假相关性

    Reducing Spurious Correlations for Aspect-Based Sentiment Analysis with Variational Information Bottleneck and Contrastive Learning. (arXiv:2303.02846v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.02846](http://arxiv.org/abs/2303.02846)

    本文提出了一种新的对比变分信息瓶颈框架（CVIB），以减少方面情感分析（ABSA）中的虚假相关性。该框架由一个原始网络和一个自剪枝网络组成，通过对比学习同时进行优化，从而丢弃了输入特征和预测标签之间的多余模式或虚假相关性。

    This paper proposes a novel Contrastive Variational Information Bottleneck framework (CVIB) to reduce spurious correlations for aspect-based sentiment analysis (ABSA). The proposed CVIB framework is composed of an original network and a self-pruned network, and these two networks are optimized simultaneously via contrastive learning, which discards the superfluous patterns or spurious correlations between input features and prediction labels.

    深度学习技术在方面情感分析（ABSA）的文献中占据主导地位，取得了最先进的结果。然而，这些深度模型通常在输入特征和输出标签之间存在虚假相关性问题，这会给鲁棒性和泛化能力带来重大障碍。在本文中，我们提出了一种新颖的对比变分信息瓶颈框架（称为CVIB），以减少ABSA中的虚假相关性。所提出的CVIB框架由一个原始网络和一个自剪枝网络组成，这两个网络通过对比学习同时进行优化。具体而言，我们采用变分信息瓶颈（VIB）原则从原始网络中学习一个信息丰富且压缩的网络（自剪枝网络），该网络丢弃了输入特征和预测标签之间的多余模式或虚假相关性。然后，我们设计了自剪枝对比学习，以将两个网络拉在一起。

    Deep learning techniques have dominated the literature on aspect-based sentiment analysis (ABSA), yielding state-of-the-art results. However, these deep models generally suffer from spurious correlation problems between input features and output labels, which creates significant barriers to robustness and generalization capability. In this paper, we propose a novel Contrastive Variational Information Bottleneck framework (called CVIB) to reduce spurious correlations for ABSA. The proposed CVIB framework is composed of an original network and a self-pruned network, and these two networks are optimized simultaneously via contrastive learning. Concretely, we employ the Variational Information Bottleneck (VIB) principle to learn an informative and compressed network (self-pruned network) from the original network, which discards the superfluous patterns or spurious correlations between input features and prediction labels. Then, self-pruning contrastive learning is devised to pull together
    

