# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Safety Implications of Explainable Artificial Intelligence in End-to-End Autonomous Driving](https://arxiv.org/abs/2403.12176) | 自动驾驶中可解释人工智能的安全影响对于确保车辆自动化安全至关重要，但当前研究中安全性和可解释性方面往往被分开砠。 |
| [^2] | [Symmetry-Breaking Augmentations for Ad Hoc Teamwork](https://arxiv.org/abs/2402.09984) | 本研究提出了一种称为对称破缺增强的方法，通过增加训练队友的行为多样性来提高人工智能代理与新队友合作的性能。实验证明了该方法的有效性。 |
| [^3] | [Embedding Ontologies via Incoprorating Extensional and Intensional Knowledge](https://arxiv.org/abs/2402.01677) | 本文提出了一种新型本体嵌入方法EIKE，通过整合外延知识和内涵知识，在外延空间和内涵空间中表示本体，并采用基于几何的方法和预训练的语言模型对实例、概念和关系进行嵌入建模。 |
| [^4] | [Exploring the Frontiers of LLMs in Psychological Applications: A Comprehensive Review.](http://arxiv.org/abs/2401.01519) | 本文综述了大型语言模型（LLMs）在心理学应用中的前沿，包括如何模拟人类认知和行为、提供创新工具进行文献回顾、假设生成、实验设计等。 |
| [^5] | [Recognize Any Regions.](http://arxiv.org/abs/2311.01373) | 本文提出了一种名为RegionSpot的新型、通用且高效的区域识别架构，旨在解决在计算机视觉中理解无约束图像中区域的语义的挑战。 |
| [^6] | [GLoRE: Evaluating Logical Reasoning of Large Language Models.](http://arxiv.org/abs/2310.09107) | 本论文介绍了GLoRE，一个评估大型语言模型逻辑推理能力的基准，实验结果表明开放式LLM模型的逻辑推理能力需要提高。研究提出了一种自一致性探测方法和微调方法来改进ChatGPT和开放式LLM的性能。 |
| [^7] | [Explainable Attention for Few-shot Learning and Beyond.](http://arxiv.org/abs/2310.07800) | 本研究介绍了一种用于少样本学习的解释性注意力框架，旨在通过识别输入数据的关键部分来提高模型性能。 |
| [^8] | [AI-Copilot for Business Optimisation: A Framework and A Case Study in Production Scheduling.](http://arxiv.org/abs/2309.13218) | 这篇论文提出了一个AI-企业优化的协同辅助系统，通过采用大型语言模型和微调预训练模型的方法，实现了减少人类专业知识需求的目标。 |
| [^9] | [Decidability of Querying First-Order Theories via Countermodels of Finite Width.](http://arxiv.org/abs/2304.06348) | 通过有限宽度的反模型查询一阶理论的可决定性并提出分割宽度，使其能够捕获实际相关的查询语言 |

# 详细

[^1]: 自动驾驶中可解释人工智能的安全影响

    Safety Implications of Explainable Artificial Intelligence in End-to-End Autonomous Driving

    [https://arxiv.org/abs/2403.12176](https://arxiv.org/abs/2403.12176)

    自动驾驶中可解释人工智能的安全影响对于确保车辆自动化安全至关重要，但当前研究中安全性和可解释性方面往往被分开砠。

    

    末端到末端学习管道正在逐渐改变高度自主车辆的持续发展，这主要归功于深度学习的进步、大规模训练数据集的可用性以及综合传感器设备的改进。然而，当代学习方法在实时决策中缺乏可解释性，妨碍了用户的信任，并减弱了这类车辆的广泛部署和商业化。此外，当这些汽车参与或导致交通事故时，问题会变得更加严重。这种缺点从社会和法律的角度引起了严重的安全担忧。因此，在末端到末端自动驾驶中解释性是促进车辆自动化安全的关键。然而，当今最先进技术中研究人员通常将自动驾驶的安全性和可解释性方面分开研究。在本文中，我们旨在弥合这一差距

    arXiv:2403.12176v1 Announce Type: cross  Abstract: The end-to-end learning pipeline is gradually creating a paradigm shift in the ongoing development of highly autonomous vehicles, largely due to advances in deep learning, the availability of large-scale training datasets, and improvements in integrated sensor devices. However, a lack of interpretability in real-time decisions with contemporary learning methods impedes user trust and attenuates the widespread deployment and commercialization of such vehicles. Moreover, the issue is exacerbated when these cars are involved in or cause traffic accidents. Such drawback raises serious safety concerns from societal and legal perspectives. Consequently, explainability in end-to-end autonomous driving is essential to enable the safety of vehicular automation. However, the safety and explainability aspects of autonomous driving have generally been investigated disjointly by researchers in today's state of the art. In this paper, we aim to brid
    
[^2]: 对于临时团队合作的对称破缺增强

    Symmetry-Breaking Augmentations for Ad Hoc Teamwork

    [https://arxiv.org/abs/2402.09984](https://arxiv.org/abs/2402.09984)

    本研究提出了一种称为对称破缺增强的方法，通过增加训练队友的行为多样性来提高人工智能代理与新队友合作的性能。实验证明了该方法的有效性。

    

    在许多协作环境中，人工智能（AI）代理必须能够适应使用未知或先前未观察到的策略的新队友。对于AI代理来说，这通常对人类来说很简单，但却是一项具有挑战性的任务。例如，如果一个AI代理在训练集中学会了与只在一侧道路上行驶的其他车辆并行驶，那么即使这些车辆的行为只是在左右对称上进行了翻转，它也可能难以适应与相反方向上行驶的驾驶员进行协调。为了解决这个问题，我们引入了对称破缺增强（SBA），通过应用对称翻转操作来增加训练队友的行为多样性。通过学习对增强后的队友的最佳响应，我们的代理能够接触到更广泛的行为约定，从而提高与新队友合作时的性能。我们在两个设置中进行了实验验证，并证明了我们的方法的有效性。

    arXiv:2402.09984v1 Announce Type: cross  Abstract: In many collaborative settings, artificial intelligence (AI) agents must be able to adapt to new teammates that use unknown or previously unobserved strategies. While often simple for humans, this can be challenging for AI agents. For example, if an AI agent learns to drive alongside others (a training set) that only drive on one side of the road, it may struggle to adapt this experience to coordinate with drivers on the opposite side, even if their behaviours are simply flipped along the left-right symmetry. To address this we introduce symmetry-breaking augmentations (SBA), which increases diversity in the behaviour of training teammates by applying a symmetry-flipping operation. By learning a best-response to the augmented set of teammates, our agent is exposed to a wider range of behavioural conventions, improving performance when deployed with novel teammates. We demonstrate this experimentally in two settings, and show that our a
    
[^3]: 通过整合外延知识和内涵知识嵌入本体

    Embedding Ontologies via Incoprorating Extensional and Intensional Knowledge

    [https://arxiv.org/abs/2402.01677](https://arxiv.org/abs/2402.01677)

    本文提出了一种新型本体嵌入方法EIKE，通过整合外延知识和内涵知识，在外延空间和内涵空间中表示本体，并采用基于几何的方法和预训练的语言模型对实例、概念和关系进行嵌入建模。

    

    本体包含领域内丰富的知识，可以分为两个类别，即外延知识和内涵知识。外延知识提供关于本体中特定概念所属的具体实例的信息，而内涵知识详细描述了概念之间的内在属性、特征和语义关联。然而，现有的本体嵌入方法未能同时充分考虑外延知识和内涵知识。在本文中，我们提出了一种名为EIKE（Extensional and Intensional Knowledge Embedding）的新型本体嵌入方法，通过在外延空间和内涵空间中表示本体。EIKE提出了一个统一的框架，用于将实例、概念及其关系嵌入到本体中，采用基于几何的方法对外延知识进行建模，并使用预训练的语言模型对内涵知识进行建模。

    Ontologies contain rich knowledge within domain, which can be divided into two categories, namely extensional knowledge and intensional knowledge. Extensional knowledge provides information about the concrete instances that belong to specific concepts in the ontology, while intensional knowledge details inherent properties, characteristics, and semantic associations among concepts. However, existing ontology embedding approaches fail to take both extensional knowledge and intensional knowledge into fine consideration simultaneously. In this paper, we propose a novel ontology embedding approach named EIKE (Extensional and Intensional Knowledge Embedding) by representing ontologies in two spaces, called extensional space and intensional space. EIKE presents a unified framework for embedding instances, concepts and their relations in an ontology, applying a geometry-based method to model extensional knowledge and a pretrained language model to model intensional knowledge, which can captur
    
[^4]: 探索LLMs在心理学应用中的前沿：一份综述

    Exploring the Frontiers of LLMs in Psychological Applications: A Comprehensive Review. (arXiv:2401.01519v1 [cs.LG])

    [http://arxiv.org/abs/2401.01519](http://arxiv.org/abs/2401.01519)

    本文综述了大型语言模型（LLMs）在心理学应用中的前沿，包括如何模拟人类认知和行为、提供创新工具进行文献回顾、假设生成、实验设计等。

    

    本文探索了大型语言模型（LLMs）在心理学应用中的前沿。心理学经历了几次理论变革，当前人工智能（AI）和机器学习，特别是LLMs的使用有望开启新的研究方向。我们详细探讨了LLMs如ChatGPT在心理学研究中的转变。文章讨论了LLMs在认知与行为心理学、临床与咨询心理学、教育与发展心理学以及社会与文化心理学等心理学分支中的影响，强调了它们模拟人类认知和行为方面的潜力。本文深入探讨了这些模型模拟人类文本生成的能力，为心理学中的文献回顾、假设生成、实验设计、实验对象、数据分析、学术写作和同行评审等提供创新工具。虽然LLMs在推动研究方法学方面起着重要作用，

    This paper explores the frontiers of large language models (LLMs) in psychology applications. Psychology has undergone several theoretical changes, and the current use of Artificial Intelligence (AI) and Machine Learning, particularly LLMs, promises to open up new research directions. We provide a detailed exploration of how LLMs like ChatGPT are transforming psychological research. It discusses the impact of LLMs across various branches of psychology, including cognitive and behavioral, clinical and counseling, educational and developmental, and social and cultural psychology, highlighting their potential to simulate aspects of human cognition and behavior. The paper delves into the capabilities of these models to emulate human-like text generation, offering innovative tools for literature review, hypothesis generation, experimental design, experimental subjects, data analysis, academic writing, and peer review in psychology. While LLMs are essential in advancing research methodologie
    
[^5]: 认证任何区域

    Recognize Any Regions. (arXiv:2311.01373v1 [cs.CV])

    [http://arxiv.org/abs/2311.01373](http://arxiv.org/abs/2311.01373)

    本文提出了一种名为RegionSpot的新型、通用且高效的区域识别架构，旨在解决在计算机视觉中理解无约束图像中区域的语义的挑战。

    

    理解无约束图像中各个区域或块的语义，例如在开放世界物体检测中，代表了一项关键而具有挑战性的计算机视觉任务。在基于强大的图像级视觉语言（ViL）基础模型如CLIP的成功基础上，最近的努力要么通过使用广泛的区域-标签对集合从头开始训练对比模型，要么将检测模型的输出与区域建议的图像级表示对齐，以发挥它们的能力。尽管取得了显著进展，但这些方法都受到计算密集型的训练需求、数据噪声的影响以及环境信息的不足等限制。为了解决这些问题，我们探索了现成的基础模型的协同潜力，利用它们在定位和语义方面的各自优势。我们引入了一种新颖的、通用的、高效的区域识别架构，称为RegionSpot。

    Understanding the semantics of individual regions or patches within unconstrained images, such as in open-world object detection, represents a critical yet challenging task in computer vision. Building on the success of powerful image-level vision-language (ViL) foundation models like CLIP, recent efforts have sought to harness their capabilities by either training a contrastive model from scratch with an extensive collection of region-label pairs or aligning the outputs of a detection model with image-level representations of region proposals. Despite notable progress, these approaches are plagued by computationally intensive training requirements, susceptibility to data noise, and deficiency in contextual information. To address these limitations, we explore the synergistic potential of off-the-shelf foundation models, leveraging their respective strengths in localization and semantics. We introduce a novel, generic, and efficient region recognition architecture, named RegionSpot, de
    
[^6]: GLoRE: 评估大型语言模型的逻辑推理能力

    GLoRE: Evaluating Logical Reasoning of Large Language Models. (arXiv:2310.09107v1 [cs.CL])

    [http://arxiv.org/abs/2310.09107](http://arxiv.org/abs/2310.09107)

    本论文介绍了GLoRE，一个评估大型语言模型逻辑推理能力的基准，实验结果表明开放式LLM模型的逻辑推理能力需要提高。研究提出了一种自一致性探测方法和微调方法来改进ChatGPT和开放式LLM的性能。

    

    最近，包括GPT-4和新兴社区模型在内的大型语言模型(LLMs)展示了显著的通用语言理解能力。然而，对这些LLMs的逻辑推理能力进行评估的尝试还很少，而这是自然语言理解的一个重要方面。为了鼓励进一步研究，我们引入了GLoRE，一个精心组织的通用逻辑推理评估基准，包含了12个覆盖三种不同类型任务的数据集。我们的实验结果显示，与人类和监督微调的性能相比，开放式LLM模型的逻辑推理能力需要进一步提高；ChatGPT和GPT-4展示了较强的逻辑推理能力，GPT-4大幅超过了ChatGPT。我们提出了一种自一致性探测方法来提高ChatGPT的准确性，以及一种微调方法来提高开放式LLM的性能。

    Recently, large language models (LLMs), including notable models such as GPT-4 and burgeoning community models, have showcased significant general language understanding abilities. However, there has been a scarcity of attempts to assess the logical reasoning capacities of these LLMs, an essential facet of natural language understanding. To encourage further investigation in this area, we introduce GLoRE, a meticulously assembled General Logical Reasoning Evaluation benchmark comprised of 12 datasets that span three different types of tasks. Our experimental results show that compared to the performance of human and supervised fine-tuning, the logical reasoning capabilities of open LLM models necessitate additional improvement; ChatGPT and GPT-4 show a strong capability of logical reasoning, with GPT-4 surpassing ChatGPT by a large margin. We propose a self-consistency probing method to enhance the accuracy of ChatGPT and a fine-tuned method to boost the performance of an open LLM. We 
    
[^7]: 解释性注意力用于少样本学习及其它领域

    Explainable Attention for Few-shot Learning and Beyond. (arXiv:2310.07800v1 [cs.AI])

    [http://arxiv.org/abs/2310.07800](http://arxiv.org/abs/2310.07800)

    本研究介绍了一种用于少样本学习的解释性注意力框架，旨在通过识别输入数据的关键部分来提高模型性能。

    

    注意力机制在增强学习模型中显示出了有希望的潜力，它可以通过识别输入数据中显著的部分来提高模型的性能。这在数据收集和标记方面存在挑战，导致训练样本有限的情况下尤其有价值。受人类认知过程启发，我们认为，如果将AI基线暴露于原始数据的关键部分而不是整个输入数据集，类似于人类的感知，那么它的性能可能会更准确、更可靠。然而，选择这些信息性数据部分的任务，即硬注意力寻找，是一个巨大的挑战。在少量训练样本的情况下，现有的研究很难找到这些信息性区域，原因是大量的训练参数无法从有限的样本中有效学习。在本研究中，我们介绍了一种实用的框架，用于实现可解释的硬注意力寻找。

    Attention mechanisms have exhibited promising potential in enhancing learning models by identifying salient portions of input data. This is particularly valuable in scenarios where limited training samples are accessible due to challenges in data collection and labeling. Drawing inspiration from human recognition processes, we posit that an AI baseline's performance could be more accurate and dependable if it is exposed to essential segments of raw data rather than the entire input dataset, akin to human perception. However, the task of selecting these informative data segments, referred to as hard attention finding, presents a formidable challenge. In situations with few training samples, existing studies struggle to locate such informative regions due to the large number of training parameters that cannot be effectively learned from the available limited samples. In this study, we introduce a novel and practical framework for achieving explainable hard attention finding, specifically
    
[^8]: AI-企业优化的协同辅助：一个框架和在生产调度中的案例研究。

    AI-Copilot for Business Optimisation: A Framework and A Case Study in Production Scheduling. (arXiv:2309.13218v1 [cs.AI])

    [http://arxiv.org/abs/2309.13218](http://arxiv.org/abs/2309.13218)

    这篇论文提出了一个AI-企业优化的协同辅助系统，通过采用大型语言模型和微调预训练模型的方法，实现了减少人类专业知识需求的目标。

    

    企业优化是寻找和实施高效和具有成本效益的运营方式，以为企业带来竞争优势的过程。综合问题表述是企业优化的一个重要组成部分，它围绕着人类专业知识展开，因此很有可能成为瓶颈。随着大型语言模型（LLMs）的最新进展，通过人工智能（AI）可以潜在地减少问题表述中所需的人类专业知识。然而，开发用于问题表述的LLM具有挑战性，由于训练数据要求、令牌限制以及LLM中缺乏适当的性能度量。为了减少大量训练数据的需求，最近人们开始关注对预训练的LLM进行微调以适应下游任务，而不是从头开始训练一个特定任务的LLM。在本文中，我们采用了这种方法，提出了一个AI-企业优化的协同辅助系统。

    Business optimisation is the process of finding and implementing efficient and cost-effective means of operation to bring a competitive advantage for businesses. Synthesizing problem formulations is an integral part of business optimisation which is centred around human expertise, thus with a high potential of becoming a bottleneck. With the recent advancements in Large Language Models (LLMs), human expertise needed in problem formulation can potentially be minimized using Artificial Intelligence (AI). However, developing a LLM for problem formulation is challenging, due to training data requirements, token limitations, and the lack of appropriate performance metrics in LLMs. To minimize the requirement of large training data, considerable attention has recently been directed towards fine-tuning pre-trained LLMs for downstream tasks, rather than training a LLM from scratch for a specific task. In this paper, we adopt this approach and propose an AI-Copilot for business optimisation by 
    
[^9]: 通过有限宽度的反模型查询一阶理论的可决定性

    Decidability of Querying First-Order Theories via Countermodels of Finite Width. (arXiv:2304.06348v1 [cs.LO])

    [http://arxiv.org/abs/2304.06348](http://arxiv.org/abs/2304.06348)

    通过有限宽度的反模型查询一阶理论的可决定性并提出分割宽度，使其能够捕获实际相关的查询语言

    

    我们提出了一个通用框架，基于具有结构简单的反模型的存在性（通过某些类型的宽度量来衡量，包括树宽和团宽等），为广泛的逻辑蕴含问题（简称查询）的可决定性提供了支持。作为我们框架的一个重要特例，我们确定了展现出宽度有限有限通用模型集的逻辑，保证了各种同态封闭查询的可决定性，包括了各种实际相关的查询语言。作为一个特别强大的宽度量，我们提出了Blumensath的分割宽度，该量包含了各种通常考虑的宽度量，具有非常有利的计算和结构特性。针对普遍展现存在性规则为一个展示案例，我们解释了有限分割宽度规则集包含其他已知的抽象可决定类，但借助现有的分层和受控规则集概念，也使我们能够捕获实际相关的查询语言，例如正则，连接和布尔连接查询。我们以存在规则的形式为重点，补充我们的理论结果，并进行了彻底的实验评估，展示了我们的框架在各种高级知识处理场景中的实际适用性和可伸缩性。

    We propose a generic framework for establishing the decidability of a wide range of logical entailment problems (briefly called querying), based on the existence of countermodels that are structurally simple, gauged by certain types of width measures (with treewidth and cliquewidth as popular examples). As an important special case of our framework, we identify logics exhibiting width-finite finitely universal model sets, warranting decidable entailment for a wide range of homomorphism-closed queries, subsuming a diverse set of practically relevant query languages. As a particularly powerful width measure, we propose Blumensath's partitionwidth, which subsumes various other commonly considered width measures and exhibits highly favorable computational and structural properties. Focusing on the formalism of existential rules as a popular showcase, we explain how finite partitionwidth sets of rules subsume other known abstract decidable classes but -- leveraging existing notions of strat
    

