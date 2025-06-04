# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Checkpoint Merging via Bayesian Optimization in LLM Pretraining](https://arxiv.org/abs/2403.19390) | 通过贝叶斯优化，我们提出了LLM预训练中的检查点合并方法，展现了在最小成本下增强预训练的能力以及在不同领域展示鲁棒泛化能力的特点。 |
| [^2] | [Large Language Models are Parallel Multilingual Learners](https://arxiv.org/abs/2403.09073) | 通过将输入翻译为多种语言，为大型语言模型提供多语言平行输入，显著增强了它们的理解能力，实验证明多语言输入可以超越传统学习方法，并发现了神经元激活的反直觉现象 |
| [^3] | [UltraWiki: Ultra-fine-grained Entity Set Expansion with Negative Seed Entities](https://arxiv.org/abs/2403.04247) | 使用负种子实体进行超细粒度实体集扩展，解决了传统方法在超细粒度语义类别表示中的问题。 |
| [^4] | [GPTVQ: The Blessing of Dimensionality for LLM Quantization](https://arxiv.org/abs/2402.15319) | 通过增加量化维度，GPTVQ方法在大型语言模型的量化中取得了新的最优结果，不仅显著改善了大小与准确性的权衡，还提高了处理效率。 |
| [^5] | [Conti Inc.: Understanding the Internal Discussions of a large Ransomware-as-a-Service Operator with Machine Learning.](http://arxiv.org/abs/2308.16061) | Conti公司的聊天记录泄露给我们提供了了解勒索软件服务运营商内部运作的机会。使用机器学习技术和可视化策略，研究发现业务、技术、内部任务管理、恶意软件和客户服务是Conti成员讨论的主要主题。 |
| [^6] | [A Small-Scale Switch Transformer and NLP-based Model for Clinical Narratives Classification.](http://arxiv.org/abs/2303.12892) | 本研究提出了一个简化的Switch Transformer框架，并从头开始训练，取得了在小型法语临床文本分类任务中比预训练的BERT模型更好的效果，采用Switch Transformer的专家混合机制有助于提高识别准确度，最终在测试集上实现了87％的准确率、87％的精度和86％的召回率。 |

# 详细

[^1]: 在LLM预训练中通过贝叶斯优化进行检查点合并

    Checkpoint Merging via Bayesian Optimization in LLM Pretraining

    [https://arxiv.org/abs/2403.19390](https://arxiv.org/abs/2403.19390)

    通过贝叶斯优化，我们提出了LLM预训练中的检查点合并方法，展现了在最小成本下增强预训练的能力以及在不同领域展示鲁棒泛化能力的特点。

    

    大型语言模型（LLMs）如GPT-4和Gemini的迅速增长突显了在它们的训练过程中对资源的强烈需求，由于巨大的计算和环境成本，这提出了重大挑战。为了缓解这一问题，我们提出了LLM预训练中的检查点合并。该方法利用具有共享训练轨迹的LLM检查点，并通过贝叶斯优化对最佳合并权重进行广泛的搜索空间探索。通过各种实验，我们展示了：（1）我们提出的方法展示了增强预训练的能力，类似于在最小成本下获得重大收益的机会；（2）尽管我们提出的方法需要一个给定的保留数据集，但仍展示了跨多个领域的稳健泛化能力，这是预训练中的一个关键方面。

    arXiv:2403.19390v1 Announce Type: new  Abstract: The rapid proliferation of large language models (LLMs) such as GPT-4 and Gemini underscores the intense demand for resources during their training processes, posing significant challenges due to substantial computational and environmental costs. To alleviate this issue, we propose checkpoint merging in pretraining LLM. This method utilizes LLM checkpoints with shared training trajectories, and is rooted in an extensive search space exploration for the best merging weight via Bayesian optimization. Through various experiments, we demonstrate that: (1) Our proposed methodology exhibits the capacity to augment pretraining, presenting an opportunity akin to obtaining substantial benefits at minimal cost; (2) Our proposed methodology, despite requiring a given held-out dataset, still demonstrates robust generalization capabilities across diverse domains, a pivotal aspect in pretraining.
    
[^2]: 大型语言模型是并行多语言学习者

    Large Language Models are Parallel Multilingual Learners

    [https://arxiv.org/abs/2403.09073](https://arxiv.org/abs/2403.09073)

    通过将输入翻译为多种语言，为大型语言模型提供多语言平行输入，显著增强了它们的理解能力，实验证明多语言输入可以超越传统学习方法，并发现了神经元激活的反直觉现象

    

    在这项研究中，我们揭示了多语言大型语言模型（LLMs）的上下文学习（ICL）能力：通过将输入翻译成多种语言，我们为LLMs提供了多语言平行输入（PiM），显著增强了它们的理解能力。为测试这种能力，我们设计了包括8个典型数据集、7种语言和8种最先进的多语言LLMs在内的大量实验证明结果显示，（1）整合更多语言可以帮助PiM进一步超越传统的ICL；（2）即使与基准性能低劣的翻译结合也是有帮助的。此外，通过检查LLMs中激活的神经元，我们发现了一个令人意外但有趣的现象。与常见观点相反，PiM并不会激活比单语输入更多的神经元来利用从多种语言学习到的知识，而实际上是抑制神经元并促进更精确的神经。

    arXiv:2403.09073v1 Announce Type: new  Abstract: In this study, we reveal an in-context learning (ICL) capability of multilingual large language models (LLMs): by translating the input to several languages, we provide Parallel Input in Multiple Languages (PiM) to LLMs, which significantly enhances their comprehension abilities. To test this capability, we design extensive experiments encompassing 8 typical datasets, 7 languages and 8 state-of-the-art multilingual LLMs. Experimental results show that (1) incorporating more languages help PiM surpass the conventional ICL further; (2) even combining with the translations that are inferior to baseline performance can also help. Moreover, by examining the activated neurons in LLMs, we discover a counterintuitive but interesting phenomenon. Contrary to the common thought that PiM would activate more neurons than monolingual input to leverage knowledge learned from diverse languages, PiM actually inhibits neurons and promotes more precise neu
    
[^3]: UltraWiki: 使用负种子实体进行超细粒度实体集扩展

    UltraWiki: Ultra-fine-grained Entity Set Expansion with Negative Seed Entities

    [https://arxiv.org/abs/2403.04247](https://arxiv.org/abs/2403.04247)

    使用负种子实体进行超细粒度实体集扩展，解决了传统方法在超细粒度语义类别表示中的问题。

    

    实体集扩展(ESE)旨在识别属于与给定种子实体相同语义类别的新实体。传统方法主要依赖正种子实体来表示目标语义类别，这对超细粒度语义类别的表示构成挑战。超细粒度语义类别是基于带有更具体属性约束的细粒度语义类别定义的。仅使用正种子实体描述会引起两个问题：(i) 超细粒度语义类别之间的歧义。(ii) 无法定义“不想要”的语义。由于这些固有缺陷，以前的方法很难解决超细粒度ESE(Ultra-ESE)。为了解决这个问题，我们首先引入了输入中的负种子实体，它们属于与正种子实体相同的细粒度语义类别，但在某些属性上有所不同。负种子实体消除

    arXiv:2403.04247v1 Announce Type: new  Abstract: Entity Set Expansion (ESE) aims to identify new entities belonging to the same semantic class as a given set of seed entities. Traditional methods primarily relied on positive seed entities to represent a target semantic class, which poses challenge for the representation of ultra-fine-grained semantic classes. Ultra-fine-grained semantic classes are defined based on fine-grained semantic classes with more specific attribute constraints. Describing it with positive seed entities alone cause two issues: (i) Ambiguity among ultra-fine-grained semantic classes. (ii) Inability to define "unwanted" semantic. Due to these inherent shortcomings, previous methods struggle to address the ultra-fine-grained ESE (Ultra-ESE). To solve this issue, we first introduce negative seed entities in the inputs, which belong to the same fine-grained semantic class as the positive seed entities but differ in certain attributes. Negative seed entities eliminate
    
[^4]: GPTVQ：LLM量化中维度的福音

    GPTVQ: The Blessing of Dimensionality for LLM Quantization

    [https://arxiv.org/abs/2402.15319](https://arxiv.org/abs/2402.15319)

    通过增加量化维度，GPTVQ方法在大型语言模型的量化中取得了新的最优结果，不仅显著改善了大小与准确性的权衡，还提高了处理效率。

    

    在这项工作中，我们展示了通过增加量化维度可以显著改善神经网络量化的大小与准确性权衡。我们提出了GPTVQ方法，这是一种新的快速后训练向量量化（VQ）方法，适用于大型语言模型（LLMs）。我们的方法交替进行一个或多个列的量化，并使用来自每层输出重建MSE的Hessian信息来更新其余未量化的权重。量化码书使用一种高效的数据感知版本的EM算法进行初始化。然后，通过使用整数量化和基于SVD的压缩进一步压缩码书。GPTVQ在诸如Llama-v2和Mistral等各种LLMs上建立了新的最新技术，大小与准确性之间的权衡。此外，我们的方法高效：在单个H100上，处理一个Llamav2-70B需要3至11小时。

    arXiv:2402.15319v1 Announce Type: cross  Abstract: In this work we show that the size versus accuracy trade-off of neural network quantization can be significantly improved by increasing the quantization dimensionality. We propose the GPTVQ method, a new fast method for post-training vector quantization (VQ) that scales well to Large Language Models (LLMs). Our method interleaves quantization of one or more columns with updates to the remaining unquantized weights, using information from the Hessian of the per-layer output reconstruction MSE. Quantization codebooks are initialized using an efficient data-aware version of the EM algorithm. The codebooks are then updated, and further compressed by using integer quantization and SVD-based compression. GPTVQ establishes a new state-of-the art in the size vs accuracy trade-offs on a wide range of LLMs such as Llama-v2 and Mistral. Furthermore, our method is efficient: on a single H100 it takes between 3 and 11 hours to process a Llamav2-70B
    
[^5]: Conti公司：通过机器学习了解一个大型勒索软件服务运营商的内部讨论

    Conti Inc.: Understanding the Internal Discussions of a large Ransomware-as-a-Service Operator with Machine Learning. (arXiv:2308.16061v1 [cs.CR])

    [http://arxiv.org/abs/2308.16061](http://arxiv.org/abs/2308.16061)

    Conti公司的聊天记录泄露给我们提供了了解勒索软件服务运营商内部运作的机会。使用机器学习技术和可视化策略，研究发现业务、技术、内部任务管理、恶意软件和客户服务是Conti成员讨论的主要主题。

    

    勒索软件服务（RaaS）正在增加勒索软件攻击的规模和复杂性。了解RaaS背后的内部运作一直是个挑战，因为此类活动是非法的。最近Conti公司泄露的聊天记录给我们提供了一个了解这类组织内部运作的良机。本文使用自然语言处理（NLP）和潜在狄利克雷分配（LDA）等机器学习技术以及可视化策略，分析了Conti公司聊天记录中的主要主题讨论。发现了五个讨论主题：1）业务，2）技术，3）内部任务/管理，4）恶意软件，5）客户服务/问题解决。此外，Conti成员的主题分布显示，只有4%的人进行了专门的讨论，而几乎所有人（96%）都是全能型，意味着他们的讨论都围绕着这五个主题展开。

    Ransomware-as-a-service (RaaS) is increasing the scale and complexity of ransomware attacks. Understanding the internal operations behind RaaS has been a challenge due to the illegality of such activities. The recent chat leak of the Conti RaaS operator, one of the most infamous ransomware operators on the international scene, offers a key opportunity to better understand the inner workings of such organizations. This paper analyzes the main topic discussions in the Conti chat leak using machine learning techniques such as Natural Language Processing (NLP) and Latent Dirichlet Allocation (LDA), as well as visualization strategies. Five discussion topics are found: 1) Business, 2) Technical, 3) Internal tasking/Management, 4) Malware, and 5) Customer Service/Problem Solving. Moreover, the distribution of topics among Conti members shows that only 4% of individuals have specialized discussions while almost all individuals (96%) are all-rounders, meaning that their discussions revolve aro
    
[^6]: 用于临床叙述分类的小规模交换变压器和基于NLP的模型

    A Small-Scale Switch Transformer and NLP-based Model for Clinical Narratives Classification. (arXiv:2303.12892v1 [cs.CL])

    [http://arxiv.org/abs/2303.12892](http://arxiv.org/abs/2303.12892)

    本研究提出了一个简化的Switch Transformer框架，并从头开始训练，取得了在小型法语临床文本分类任务中比预训练的BERT模型更好的效果，采用Switch Transformer的专家混合机制有助于提高识别准确度，最终在测试集上实现了87％的准确率、87％的精度和86％的召回率。

    

    近年来，基于变压器的模型（如交换变压器）在自然语言处理任务中取得了显著的结果。然而，这些模型通常过于复杂并需要大量的预训练，这限制了它们在有限数据的小型临床文本分类任务中的有效性。在本研究中，我们提出了一个简化的Switch Transformer框架，并从头开始在CHU Sainte-Justine医院的小型法语临床文本分类数据集上进行了训练。我们的结果表明，简化的小规模变压器模型优于预训练的BERT模型，包括DistillBERT、CamemBERT、FlauBERT和FrALBERT。此外，使用Switch Transformer的专家混合机制有助于捕获多样的模式；因此，所提出的方法比具有自我注意机制的传统变压器获得更好的结果。最后，我们提出的框架在测试集上实现了87％的准确率，87％的精度和86％的召回率，突显了其在小型临床文本分类任务中的潜力。

    In recent years, Transformer-based models such as the Switch Transformer have achieved remarkable results in natural language processing tasks. However, these models are often too complex and require extensive pre-training, which limits their effectiveness for small clinical text classification tasks with limited data. In this study, we propose a simplified Switch Transformer framework and train it from scratch on a small French clinical text classification dataset at CHU Sainte-Justine hospital. Our results demonstrate that the simplified small-scale Transformer models outperform pre-trained BERT-based models, including DistillBERT, CamemBERT, FlauBERT, and FrALBERT. Additionally, using a mixture of expert mechanisms from the Switch Transformer helps capture diverse patterns; hence, the proposed approach achieves better results than a conventional Transformer with the self-attention mechanism. Finally, our proposed framework achieves an accuracy of 87\%, precision at 87\%, and recall 
    

