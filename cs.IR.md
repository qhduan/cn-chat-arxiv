# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Streamlined Data Fusion: Unleashing the Power of Linear Combination with Minimal Relevance Judgments.](http://arxiv.org/abs/2309.04981) | 本研究发现，仅使用20％-50％的相关文档，通过多元线性回归训练得到的权重与使用传统方法得到的权重非常接近，从而实现了更高效和可负担的数据融合方法。 |
| [^2] | [Multi-modal Extreme Classification.](http://arxiv.org/abs/2309.04961) | 本文提出了一种名为MUFIN的技术，用于处理具有数百万个标签的多模态极端分类任务，该技术在产品推荐和竞标查询预测中具有应用前景。 |
| [^3] | [A multiple k-means cluster ensemble framework for clustering citation trajectories.](http://arxiv.org/abs/2309.04949) | 本文提出了一种基于特征的多个k-means聚类集成框架，用于聚类引用轨迹。这有助于理解知识传播过程，并解决了现有方法依赖参数、定义模糊和只捕捉极端轨迹的问题。 |
| [^4] | [RecAD: Towards A Unified Library for Recommender Attack and Defense.](http://arxiv.org/abs/2309.04884) | RecAD是一个旨在建立推荐攻击和防御的开放基准的统一库，通过整合数据集、源代码、参数设置、运行日志、攻击知识、攻击预算和评估结果，为研究人员提供一个可复现的研究流程。 |
| [^5] | [Exploring Music Genre Classification: Algorithm Analysis and Deployment Architecture.](http://arxiv.org/abs/2309.04861) | 本文研究了音乐流派分类，使用了数字信号处理和深度学习技术，并提出了一种新颖的算法，可以从音频信号中提取特征并进行分类。该算法在GTZAN数据集上取得高精度，同时还提出了端到端的部署架构，可用于音乐应用程序的集成。 |
| [^6] | [CPMR: Context-Aware Incremental Sequential Recommendation with Pseudo-Multi-Task Learning.](http://arxiv.org/abs/2309.04802) | CPMR是一个基于上下文感知的增量顺序推荐系统，通过创建静态嵌入、历史时间状态和上下文时间状态的三个表示，准确地建模了用户随时间变化的表示和兴趣动态的演化。 |
| [^7] | [A Comprehensive Survey on Deep Learning Techniques in Educational Data Mining.](http://arxiv.org/abs/2309.04761) | 本调研综合审查了在教育数据挖掘中深度学习技术的最新研究进展，包括对知识跟踪、学生不良行为检测、性能预测和个性化推荐等典型教育场景的应用。同时提供了公共数据集和处理工具的综合概述，并指出了未来的研究方向。 |
| [^8] | [Data Augmentation for Conversational AI.](http://arxiv.org/abs/2309.04739) | 本教程提供了对话式人工智能中数据增强的综述，包括对话增强、开放域和任务导向的对话生成以及评估模型。此外，还讨论了当前的挑战和未来的发展方向，以帮助推动该领域的发展。 |
| [^9] | [Analysis of Disinformation and Fake News Detection Using Fine-Tuned Large Language Model.](http://arxiv.org/abs/2309.04704) | 本研究考虑使用LLM模型通过细调实现虚假信息和假新闻的深入分析，揭示复杂的风格和叙事，并提取命名实体的情感，以此作为监督机器学习模型中的预测性特征。 |
| [^10] | [Financial News Analytics Using Fine-Tuned Llama 2 GPT Model.](http://arxiv.org/abs/2308.13032) | 本研究通过精细调整的Llama 2模型实现了金融新闻的多任务分析，包括文本分析、摘要和情感提取等。实验结果显示，提取的命名实体情感可以作为有监督机器学习模型的预测特征。 |
| [^11] | [EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction.](http://arxiv.org/abs/2304.10711) | 本文提出了一种自适应特征交互学习模型EulerNet，它采用欧拉公式将高阶特征交互映射到复杂向量空间中学习，从而在保持效率的同时提高模型能力。 |

# 详细

[^1]: 精简数据融合: 以最少的相关性判断释放线性组合的力量

    Streamlined Data Fusion: Unleashing the Power of Linear Combination with Minimal Relevance Judgments. (arXiv:2309.04981v1 [cs.IR])

    [http://arxiv.org/abs/2309.04981](http://arxiv.org/abs/2309.04981)

    本研究发现，仅使用20％-50％的相关文档，通过多元线性回归训练得到的权重与使用传统方法得到的权重非常接近，从而实现了更高效和可负担的数据融合方法。

    

    线性组合是信息检索任务中一种强大的数据融合方法，它能够根据不同的情境调整权重。然而，传统上实现最优权重训练通常需要对大部分文档进行人工相关性判断，这是一项费时费力的过程。在本研究中，我们探讨了仅使用20％-50％的相关文档获取接近最优权重的可行性。通过对四个TREC数据集进行实验，我们发现使用这种减少的数据集进行多元线性回归训练得到的权重与使用TREC官方"qrels"得到的权重非常接近。我们的研究结果揭示了更高效、更经济的数据融合潜力，使研究人员和从业者能够在更少的工作量下充分享受其所带来的好处。

    Linear combination is a potent data fusion method in information retrieval tasks, thanks to its ability to adjust weights for diverse scenarios. However, achieving optimal weight training has traditionally required manual relevance judgments on a large percentage of documents, a labor-intensive and expensive process. In this study, we investigate the feasibility of obtaining near-optimal weights using a mere 20\%-50\% of relevant documents. Through experiments on four TREC datasets, we find that weights trained with multiple linear regression using this reduced set closely rival those obtained with TREC's official "qrels." Our findings unlock the potential for more efficient and affordable data fusion, empowering researchers and practitioners to reap its full benefits with significantly less effort.
    
[^2]: 多模态极端分类

    Multi-modal Extreme Classification. (arXiv:2309.04961v1 [cs.IR])

    [http://arxiv.org/abs/2309.04961](http://arxiv.org/abs/2309.04961)

    本文提出了一种名为MUFIN的技术，用于处理具有数百万个标签的多模态极端分类任务，该技术在产品推荐和竞标查询预测中具有应用前景。

    

    本文针对具有数百万个标签的极端分类任务，发展了一种名为MUFIN的技术，其中数据点和标签具有视觉和文本描述符。将MUFIN应用于数百万个产品的产品推荐和竞标查询预测中。当代的多模态方法通常依赖于仅嵌入式方法。另一方面，XC方法利用分类器架构提供比仅嵌入式方法更高的准确性，但主要专注于基于文本的分类任务。MUFIN通过将多模态分类重新制定为具有数百万个标签的XC问题来弥合这一差距。这提出了两个挑战：开发能够提供足够表达力以实现对数百万个标签进行准确分类的多模态架构；以及在标签数量的对数尺度上扩展训练和推理例程。MUFIN基于交叉的方法开发了一种架构。

    This paper develops the MUFIN technique for extreme classification (XC) tasks with millions of labels where datapoints and labels are endowed with visual and textual descriptors. Applications of MUFIN to product-to-product recommendation and bid query prediction over several millions of products are presented. Contemporary multi-modal methods frequently rely on purely embedding-based methods. On the other hand, XC methods utilize classifier architectures to offer superior accuracies than embedding only methods but mostly focus on text-based categorization tasks. MUFIN bridges this gap by reformulating multi-modal categorization as an XC problem with several millions of labels. This presents the twin challenges of developing multi-modal architectures that can offer embeddings sufficiently expressive to allow accurate categorization over millions of labels; and training and inference routines that scale logarithmically in the number of labels. MUFIN develops an architecture based on cros
    
[^3]: 用于聚类引用轨迹的多个K-means聚类集成框架

    A multiple k-means cluster ensemble framework for clustering citation trajectories. (arXiv:2309.04949v1 [cs.SI])

    [http://arxiv.org/abs/2309.04949](http://arxiv.org/abs/2309.04949)

    本文提出了一种基于特征的多个k-means聚类集成框架，用于聚类引用轨迹。这有助于理解知识传播过程，并解决了现有方法依赖参数、定义模糊和只捕捉极端轨迹的问题。

    

    引用成熟时间因文章而异，然而所有文章的影响力都是在一个固定窗口内衡量的。对它们的引用轨迹进行聚类有助于理解知识扩散过程，并揭示并非所有文章在发表后都立即获得成功。此外，对轨迹进行聚类也对论文影响力推荐算法至关重要。由于引用时间序列具有非线性和非平稳特性，这是一个具有挑战性的问题。先前的工作提出了一组任意的阈值和基于规则的固定方法。所有的方法主要都依赖于参数，因此在定义相似的轨迹和关于特定数目的模糊性方面导致了不一致性。大多数研究只捕捉了极端的轨迹。因此，需要一个通用的聚类框架。本文提出了一个基于特征的多个k-means聚类集成框架。

    Citation maturity time varies for different articles. However, the impact of all articles is measured in a fixed window. Clustering their citation trajectories helps understand the knowledge diffusion process and reveals that not all articles gain immediate success after publication. Moreover, clustering trajectories is necessary for paper impact recommendation algorithms. It is a challenging problem because citation time series exhibit significant variability due to non linear and non stationary characteristics. Prior works propose a set of arbitrary thresholds and a fixed rule based approach. All methods are primarily parameter dependent. Consequently, it leads to inconsistencies while defining similar trajectories and ambiguities regarding their specific number. Most studies only capture extreme trajectories. Thus, a generalised clustering framework is required. This paper proposes a feature based multiple k means cluster ensemble framework. 1,95,783 and 41,732 well cited articles f
    
[^4]: RecAD: 向统一的推荐攻击和防御库迈进

    RecAD: Towards A Unified Library for Recommender Attack and Defense. (arXiv:2309.04884v1 [cs.IR])

    [http://arxiv.org/abs/2309.04884](http://arxiv.org/abs/2309.04884)

    RecAD是一个旨在建立推荐攻击和防御的开放基准的统一库，通过整合数据集、源代码、参数设置、运行日志、攻击知识、攻击预算和评估结果，为研究人员提供一个可复现的研究流程。

    

    近年来，推荐系统已经成为我们日常生活中无处不在的一部分，然而由于不断增长的商业和社会价值，它们面临着被攻击的高风险。尽管在推荐攻击和防御方面取得了重要的研究进展，但该领域缺乏广泛认可的基准标准，导致性能比较不公平且实验可信度有限。为了解决这个问题，我们提出了RecAD，一个旨在建立推荐攻击和防御的开放基准的统一库。RecAD通过整合不同的数据集、标准源代码、超参数设置、运行日志、攻击知识、攻击预算和评估结果，初步建立了一个统一的基准测试流程，以实现可复现的研究。该基准测试旨在全面且可持续，涵盖攻击、防御和评估任务，使更多的研究人员能够轻松地追随和贡献这个有前景的领域。

    In recent years, recommender systems have become a ubiquitous part of our daily lives, while they suffer from a high risk of being attacked due to the growing commercial and social values. Despite significant research progress in recommender attack and defense, there is a lack of a widely-recognized benchmarking standard in the field, leading to unfair performance comparison and limited credibility of experiments. To address this, we propose RecAD, a unified library aiming at establishing an open benchmark for recommender attack and defense. RecAD takes an initial step to set up a unified benchmarking pipeline for reproducible research by integrating diverse datasets, standard source codes, hyper-parameter settings, running logs, attack knowledge, attack budget, and evaluation results. The benchmark is designed to be comprehensive and sustainable, covering both attack, defense, and evaluation tasks, enabling more researchers to easily follow and contribute to this promising field. RecA
    
[^5]: 探索音乐流派分类：算法分析与部署架构

    Exploring Music Genre Classification: Algorithm Analysis and Deployment Architecture. (arXiv:2309.04861v1 [cs.SD])

    [http://arxiv.org/abs/2309.04861](http://arxiv.org/abs/2309.04861)

    本文研究了音乐流派分类，使用了数字信号处理和深度学习技术，并提出了一种新颖的算法，可以从音频信号中提取特征并进行分类。该算法在GTZAN数据集上取得高精度，同时还提出了端到端的部署架构，可用于音乐应用程序的集成。

    

    随着各种流媒体应用的出现，音乐流派分类变得越来越重要。如今，在一个复杂的音乐应用程序中，我们无法想象仅通过艺术家的名字和歌曲标题来搜索音乐。正确分类音乐一直很困难，因为与音乐相关的信息，如地区、艺术家、专辑或非专辑，是如此多变。本文提出了一项关于音乐流派分类的研究，使用了数字信号处理（DSP）和深度学习（DL）技术的组合。提出了一种新颖的算法，利用DSP和DL方法从音频信号中提取相关特征，并将其分类到各种流派中。该算法在GTZAN数据集上进行了测试，并取得了高精度。还提出了一种端到端的部署架构，用于集成到与音乐相关的应用程序中。对算法的性能进行了分析，并讨论了改进的未来方向。

    Music genre classification has become increasingly critical with the advent of various streaming applications. Nowadays, we find it impossible to imagine using the artist's name and song title to search for music in a sophisticated music app. It is always difficult to classify music correctly because the information linked to music, such as region, artist, album, or non-album, is so variable. This paper presents a study on music genre classification using a combination of Digital Signal Processing (DSP) and Deep Learning (DL) techniques. A novel algorithm is proposed that utilizes both DSP and DL methods to extract relevant features from audio signals and classify them into various genres. The algorithm was tested on the GTZAN dataset and achieved high accuracy. An end-to-end deployment architecture is also proposed for integration into music-related applications. The performance of the algorithm is analyzed and future directions for improvement are discussed. The proposed DSP and DL-b
    
[^6]: CPMR: 基于上下文感知的增量顺序推荐与伪多任务学习

    CPMR: Context-Aware Incremental Sequential Recommendation with Pseudo-Multi-Task Learning. (arXiv:2309.04802v1 [cs.IR])

    [http://arxiv.org/abs/2309.04802](http://arxiv.org/abs/2309.04802)

    CPMR是一个基于上下文感知的增量顺序推荐系统，通过创建静态嵌入、历史时间状态和上下文时间状态的三个表示，准确地建模了用户随时间变化的表示和兴趣动态的演化。

    

    用户进行互动的动机可以分为静态偏好和动态兴趣。为了准确地建模用户随时间变化的表示，最近的顺序推荐研究利用信息传播和演化从批量到达的互动中进行挖掘。然而，他们忽略了在上下文场景中人们很容易受到其他用户的最近行为的影响，并且在所有历史互动中应用演化会稀释最近互动的重要性，从而无法准确地建模兴趣动态的演化。为了解决这个问题，我们提出了一种基于上下文感知的伪多任务推荐系统（CPMR），通过为每个用户和项目创建三个表示（静态嵌入、历史时间状态和上下文时间状态），来建模历史和上下文情境中的演化。为了同时提高时间状态演化和增量推荐的性能。

    The motivations of users to make interactions can be divided into static preference and dynamic interest. To accurately model user representations over time, recent studies in sequential recommendation utilize information propagation and evolution to mine from batches of arriving interactions. However, they ignore the fact that people are easily influenced by the recent actions of other users in the contextual scenario, and applying evolution across all historical interactions dilutes the importance of recent ones, thus failing to model the evolution of dynamic interest accurately. To address this issue, we propose a Context-Aware Pseudo-Multi-Task Recommender System (CPMR) to model the evolution in both historical and contextual scenarios by creating three representations for each user and item under different dynamics: static embedding, historical temporal states, and contextual temporal states. To dually improve the performance of temporal states evolution and incremental recommenda
    
[^7]: 在教育数据挖掘中深度学习技术的综合调研

    A Comprehensive Survey on Deep Learning Techniques in Educational Data Mining. (arXiv:2309.04761v1 [cs.LG])

    [http://arxiv.org/abs/2309.04761](http://arxiv.org/abs/2309.04761)

    本调研综合审查了在教育数据挖掘中深度学习技术的最新研究进展，包括对知识跟踪、学生不良行为检测、性能预测和个性化推荐等典型教育场景的应用。同时提供了公共数据集和处理工具的综合概述，并指出了未来的研究方向。

    

    教育数据挖掘(EDM)作为研究的重要领域，利用计算技术来分析教育数据。随着教育数据的复杂性和多样性增加，深度学习技术在解决分析和建模这些数据所面临的挑战方面表现出了显著的优势。本调研旨在系统地审查深度学习在EDM领域的最新研究进展。我们首先提供了关于EDM和深度学习的简要介绍，强调了它们在现代教育环境中的重要性。接下来，我们详细回顾了在四个典型教育场景中应用的深度学习技术，包括知识跟踪、学生不良行为检测、性能预测和个性化推荐。此外，我们还提供了EDM的公共数据集和处理工具的综合概述。最后，我们指出了该研究领域的新兴趋势和未来方向。

    Educational Data Mining (EDM) has emerged as a vital field of research, which harnesses the power of computational techniques to analyze educational data. With the increasing complexity and diversity of educational data, Deep Learning techniques have shown significant advantages in addressing the challenges associated with analyzing and modeling this data. This survey aims to systematically review the state-of-the-art in EDM with Deep Learning. We begin by providing a brief introduction to EDM and Deep Learning, highlighting their relevance in the context of modern education. Next, we present a detailed review of Deep Learning techniques applied in four typical educational scenarios, including knowledge tracing, undesirable student detecting, performance prediction, and personalized recommendation. Furthermore, a comprehensive overview of public datasets and processing tools for EDM is provided. Finally, we point out emerging trends and future directions in this research area.
    
[^8]: 对话式人工智能的数据增强

    Data Augmentation for Conversational AI. (arXiv:2309.04739v1 [cs.CL])

    [http://arxiv.org/abs/2309.04739](http://arxiv.org/abs/2309.04739)

    本教程提供了对话式人工智能中数据增强的综述，包括对话增强、开放域和任务导向的对话生成以及评估模型。此外，还讨论了当前的挑战和未来的发展方向，以帮助推动该领域的发展。

    

    对话系统的发展已经彻底改变了信息获取方式，超越了单一查询的限制。然而，开发对话系统需要大量的训练数据，在资源有限的领域和语言中具有挑战性。传统的数据收集方法，如众包，需要大量的人力和时间，因此在此情景下效率低下。数据增强（DA）是一种缓解对话系统中数据稀缺问题的有效方法。本教程全面且最新地概述了在对话系统中使用的DA方法，包括对话增强、开放域和任务导向的对话生成以及不同的评估模型的范式。我们还讨论了当前的挑战和未来的发展方向，以帮助研究人员和从业者进一步推动这一领域的发展。

    Advancements in conversational systems have revolutionized information access, surpassing the limitations of single queries. However, developing dialogue systems requires a large amount of training data, which is a challenge in low-resource domains and languages. Traditional data collection methods like crowd-sourcing are labor-intensive and time-consuming, making them ineffective in this context. Data augmentation (DA) is an affective approach to alleviate the data scarcity problem in conversational systems. This tutorial provides a comprehensive and up-to-date overview of DA approaches in the context of conversational systems. It highlights recent advances in conversation augmentation, open domain and task-oriented conversation generation, and different paradigms of evaluating these models. We also discuss current challenges and future directions in order to help researchers and practitioners to further advance the field in this area.
    
[^9]: 通过优化大型语言模型进行虚假信息和假新闻的检测分析

    Analysis of Disinformation and Fake News Detection Using Fine-Tuned Large Language Model. (arXiv:2309.04704v1 [cs.CL])

    [http://arxiv.org/abs/2309.04704](http://arxiv.org/abs/2309.04704)

    本研究考虑使用LLM模型通过细调实现虚假信息和假新闻的深入分析，揭示复杂的风格和叙事，并提取命名实体的情感，以此作为监督机器学习模型中的预测性特征。

    

    本文考虑使用LLM（Llama 2大型语言模型）通过细调进行虚假信息分析和假新闻的检测。采用了基于PEFT/LoRA的细调方法。研究中，该模型对以下任务进行了细调：揭示虚假信息和宣传叙事的文本分析，事实核查，假新闻检测，操纵分析以及提取带有情感的命名实体。所得结果表明，经过细调的Llama 2模型能够对文本进行深入分析，并揭示复杂的风格和叙事。带有情感的命名实体可以作为监督机器学习模型中的预测性特征。

    The paper considers the possibility of fine-tuning Llama 2 large language model (LLM) for the disinformation analysis and fake news detection. For fine-tuning, the PEFT/LoRA based approach was used. In the study, the model was fine-tuned for the following tasks: analysing a text on revealing disinformation and propaganda narratives, fact checking, fake news detection, manipulation analytics, extracting named entities with their sentiments. The obtained results show that the fine-tuned Llama 2 model can perform a deep analysis of texts and reveal complex styles and narratives. Extracted sentiments for named entities can be considered as predictive features in supervised machine learning models.
    
[^10]: 使用精细调整的Llama 2 GPT模型进行金融新闻分析

    Financial News Analytics Using Fine-Tuned Llama 2 GPT Model. (arXiv:2308.13032v1 [cs.CL])

    [http://arxiv.org/abs/2308.13032](http://arxiv.org/abs/2308.13032)

    本研究通过精细调整的Llama 2模型实现了金融新闻的多任务分析，包括文本分析、摘要和情感提取等。实验结果显示，提取的命名实体情感可以作为有监督机器学习模型的预测特征。

    

    本文考虑了使用精细调整的Llama 2 Large Language Model (LLM) 对金融新闻进行多任务分析的可能性。通过PEFT/LoRA方法对模型进行精细调整，主要包括从金融市场角度分析文本、突出文本的主要观点、对文本进行摘要和提取具有适当情感的命名实体等任务。实验结果表明，经过精细调整的Llama 2模型能够进行多任务的金融新闻分析，其响应的结构可以部分为结构化文本，另一部分数据可以采用JSON格式进一步处理。提取的命名实体情感可以被视为具有定量目标变量的监督机器学习模型的预测特征。

    The paper considers the possibility to fine-tune Llama 2 Large Language Model (LLM) for the multitask analysis of financial news. For fine-tuning, the PEFT/LoRA based approach was used. In the study, the model was fine-tuned for the following tasks: analysing a text from financial market perspectives, highlighting main points of a text, summarizing a text and extracting named entities with appropriate sentiments. The obtained results show that the fine-tuned Llama 2 model can perform a multitask financial news analysis with a specified structure of response, part of response can be a structured text and another part of data can have JSON format for further processing. Extracted sentiments for named entities can be considered as predictive features in supervised machine learning models with quantitative target variables.
    
[^11]: EulerNet: 基于欧拉公式的复杂向量空间特征交互学习以实现点击率预测

    EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction. (arXiv:2304.10711v1 [cs.IR])

    [http://arxiv.org/abs/2304.10711](http://arxiv.org/abs/2304.10711)

    本文提出了一种自适应特征交互学习模型EulerNet，它采用欧拉公式将高阶特征交互映射到复杂向量空间中学习，从而在保持效率的同时提高模型能力。

    

    在点击率预测任务中，学习高阶特征交互是非常关键的。然而，在在线电子商务平台中，由于海量特征的存在，计算高阶特征交互非常耗时。大多数现有方法手动设计最大阶数，并从中过滤出无用的交互。尽管它们减少了高阶特征组合的指数级增长所引起的高计算成本，但由于受到受限的特征阶数的次优学习的影响，它们仍然会受到模型能力下降的影响。保持模型能力并同时保持其效率的解决方案是一个技术挑战，该问题尚未得到充分解决。为了解决这个问题，我们提出了一个自适应特征交互学习模型，名为EulerNet，在该模型中，通过根据欧拉公式进行空间映射在复杂向量空间中学习特征交互。

    Learning effective high-order feature interactions is very crucial in the CTR prediction task. However, it is very time-consuming to calculate high-order feature interactions with massive features in online e-commerce platforms. Most existing methods manually design a maximal order and further filter out the useless interactions from them. Although they reduce the high computational costs caused by the exponential growth of high-order feature combinations, they still suffer from the degradation of model capability due to the suboptimal learning of the restricted feature orders. The solution to maintain the model capability and meanwhile keep it efficient is a technical challenge, which has not been adequately addressed. To address this issue, we propose an adaptive feature interaction learning model, named as EulerNet, in which the feature interactions are learned in a complex vector space by conducting space mapping according to Euler's formula. EulerNet converts the exponential power
    

