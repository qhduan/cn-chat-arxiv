# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoTRIZ: Artificial Ideation with TRIZ and Large Language Models](https://arxiv.org/abs/2403.13002) | 本文提出了AutoTRIZ，一种利用大型语言模型自动化和增强TRIZ方法的人工创意工具，为设计自动化和可解释创意提供了一种新颖方法。 |
| [^2] | [NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning](https://arxiv.org/abs/2403.07376) | 本文提出了一种名为NavCoT的新策略，在视觉与语言导航中通过学习解耦推理，实现了自主导航决策，有效减轻了领域差距。 |
| [^3] | [GhostWriter: Augmenting Collaborative Human-AI Writing Experiences Through Personalization and Agency](https://arxiv.org/abs/2402.08855) | GhostWriter是一个AI增强的写作设计探针，通过个性化和代理增强用户的写作体验。它利用大型语言模型（LLMs）隐式学习用户的写作风格，并允许用户通过手动样式编辑和批注来控制系统的写作风格。 |
| [^4] | [AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts](https://arxiv.org/abs/2402.07625) | 本论文介绍了一种自主数据选择策略，利用语言模型进行数学文本的自动评估和选择，并通过连续预训练显著提高了数学推理能力。主要创新包括利用元提示语言模型作为验证器，发布了高质量的AutoMathText数据集，并实现了预训练令牌效率的提升。 |
| [^5] | [Large Language Models: A Survey](https://arxiv.org/abs/2402.06196) | 大型语言模型（LLMs）吸引了很多关注，因为它们在自然语言任务上的强大表现。该研究领域发展迅速，包括了各种著名的LLMs、构建和增强LLMs的技术、以及流行的LLM数据集和评估指标。 |
| [^6] | [Promoting Segment Anything Model towards Highly Accurate Dichotomous Image Segmentation](https://arxiv.org/abs/2401.00248) | 将段分离任意模型推进至高度准确的二元图像分割，通过提出DIS-SAM框架，成功改进SAM模型在细节方面的表现，实现了显著增强的分割精度。 |
| [^7] | [Cross-domain Random Pre-training with Prototypes for Reinforcement Learning](https://arxiv.org/abs/2302.05614) | 提出了CRPTpro框架，利用原型进行跨领域自监督随机预训练，提高预训练效率，并实现在不同领域中定义的视觉控制RL任务。 |
| [^8] | [Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending.](http://arxiv.org/abs/2401.16458) | 本文研究了如何利用P2P借贷平台上借款人提供的文本描述来构建风险指标。结果显示，利用大型语言模型生成的风险评分可以明显提高信用风险分类器的性能。 |
| [^9] | [Mixup Your Own Pairs.](http://arxiv.org/abs/2309.16633) | 本文提出了一种名为SupReMix的方法，通过混合样本，特别是混合负样本和混合正样本，来解决回归问题中表示学习的挑战。这种方法能够提供更好的性能和更准确的回归结果。 |
| [^10] | [Notation3 as an Existential Rule Language.](http://arxiv.org/abs/2308.07332) | 本文研究了Notation3与存在规则之间的关系，并提出了一个将部分Notation3直接映射到存在规则的方法，从而提高了Notation3推理的效率。 |
| [^11] | [Introducing Foundation Models as Surrogate Models: Advancing Towards More Practical Adversarial Attacks.](http://arxiv.org/abs/2307.06608) | 本文将对抗攻击重新设定为下游任务，通过生成图像噪声来满足新兴趋势，并将基础模型引入作为代理模型。虽然基础模型的表现不佳，但通过在特征空间中进行分析，我们发现缺乏对应的特征。 |
| [^12] | [The Representational Status of Deep Learning Models.](http://arxiv.org/abs/2303.12032) | 该论文澄清了深度学习模型的表征状态。尽管通常称为“表征”，但实际上它们更适合理解为高度理想化的模型，这一结果对可解释的AI有着直接影响，也引起了哲学家对其在未来科学研究中的作用的关注。 |

# 详细

[^1]: AutoTRIZ：利用TRIZ和大型语言模型的人工创意

    AutoTRIZ: Artificial Ideation with TRIZ and Large Language Models

    [https://arxiv.org/abs/2403.13002](https://arxiv.org/abs/2403.13002)

    本文提出了AutoTRIZ，一种利用大型语言模型自动化和增强TRIZ方法的人工创意工具，为设计自动化和可解释创意提供了一种新颖方法。

    

    研究人员和创新者在开发思维方法方面做出了巨大努力，比如形态分析和类比设计，以辅助工程设计创意，解决问题和推动创新。在这些方法中，TRIZ作为最著名的方法脱颖而出，被广泛应用于系统化创新。然而，TRIZ资源和概念的复杂性，以及其对用户知识、经验和推理能力的依赖，限制了其实用性。本文提出了AutoTRIZ，一种利用大型语言模型（LLMs）自动化和增强TRIZ方法的人工创意工具。通过利用LLMs的广泛知识和先进推理能力，AutoTRIZ提供了一种新颖的利用人工智能进行设计自动化和可解释创意的方法。我们通过对矛盾检测和比较方面的一致性实验来证明并评估AutoTRIZ的有效性。

    arXiv:2403.13002v1 Announce Type: cross  Abstract: Researchers and innovators have made enormous efforts in developing ideation methods, such as morphological analysis and design-by-analogy, to aid engineering design ideation for problem solving and innovation. Among these, TRIZ stands out as the most well-known approach, widely applied for systematic innovation. However, the complexity of TRIZ resources and concepts, coupled with its reliance on users' knowledge, experience, and reasoning capabilities, limits its practicability. This paper proposes AutoTRIZ, an artificial ideation tool that leverages large language models (LLMs) to automate and enhance the TRIZ methodology. By leveraging the broad knowledge and advanced reasoning capabilities of LLMs, AutoTRIZ offers a novel approach to design automation and interpretable ideation with artificial intelligence. We demonstrate and evaluate the effectiveness of AutoTRIZ through consistency experiments in contradiction detection and compa
    
[^2]: NavCoT: 通过学习解耦推理提升基于LLM的视觉与语言导航

    NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning

    [https://arxiv.org/abs/2403.07376](https://arxiv.org/abs/2403.07376)

    本文提出了一种名为NavCoT的新策略，在视觉与语言导航中通过学习解耦推理，实现了自主导航决策，有效减轻了领域差距。

    

    视觉与语言导航(VLN)作为具有重要研究价值的具身人工智能问题，需要一个具身代理根据自然语言指示穿越复杂的3D环境。最近的研究突出了大型语言模型(LLMs)在VLN中提高导航推理准确性和可解释性的潜力。然而，它们主要在离线方式下的使用通常在VLN任务和LLM训练语料库之间遭受显著的领域差距。本文引入了一种名为导航思维链(NavCoT)的新型策略，我们通过完成领域内高效参数训练，实现自主导航决策，有效减轻领域差距的成本。具体地，在每个时间步，LLM被提示通过作为世界模型来预测导航思维链：1)根据

    arXiv:2403.07376v1 Announce Type: cross  Abstract: Vision-and-Language Navigation (VLN), as a crucial research problem of Embodied AI, requires an embodied agent to navigate through complex 3D environments following natural language instructions. Recent research has highlighted the promising capacity of large language models (LLMs) in VLN by improving navigational reasoning accuracy and interpretability. However, their predominant use in an offline manner usually suffers from substantial domain gap between the VLN task and the LLM training corpus. This paper introduces a novel strategy called Navigational Chain-of-Thought (NavCoT), where we fulfill parameter-efficient in-domain training to enable self-guided navigational decision, leading to a significant mitigation of the domain gap in a cost-effective manner. Specifically, at each timestep, the LLM is prompted to forecast the navigational chain-of-thought by: 1) acting as a world model to imagine the next observation according to the
    
[^3]: GhostWriter:通过个性化和代理增强协作人工智能写作体验

    GhostWriter: Augmenting Collaborative Human-AI Writing Experiences Through Personalization and Agency

    [https://arxiv.org/abs/2402.08855](https://arxiv.org/abs/2402.08855)

    GhostWriter是一个AI增强的写作设计探针，通过个性化和代理增强用户的写作体验。它利用大型语言模型（LLMs）隐式学习用户的写作风格，并允许用户通过手动样式编辑和批注来控制系统的写作风格。

    

    大型语言模型（LLMs）在提供不同形式的写作辅助方面越来越流行，并且具有无处不在的应用。然而，由于个性化和控制能力有限，LLM驱动的写作系统可能会使用户感到沮丧，当用户缺乏提示工程经验时，这种情况可能加剧。我们认为设计可以解决这些挑战之一，并引入GhostWriter，这是一个AI增强的写作设计探针，用户可以通过增强的代理和个性化来进行写作。GhostWriter利用LLMs在用户编写的过程中隐式学习用户所期望的写作风格，同时允许通过手动样式编辑和批注进行显式教学。我们研究了18名参与者在两个不同的写作任务中使用GhostWriter，观察到它帮助用户编写个性化的文本生成，并通过提供多种方式控制系统的写作风格来增强用户的能力。从这项研究中，我们提出了一些见解。

    arXiv:2402.08855v1 Announce Type: cross Abstract: Large language models (LLMs) are becoming more prevalent and have found a ubiquitous use in providing different forms of writing assistance. However, LLM-powered writing systems can frustrate users due to their limited personalization and control, which can be exacerbated when users lack experience with prompt engineering. We see design as one way to address these challenges and introduce GhostWriter, an AI-enhanced writing design probe where users can exercise enhanced agency and personalization. GhostWriter leverages LLMs to learn the user's intended writing style implicitly as they write, while allowing explicit teaching moments through manual style edits and annotations. We study 18 participants who use GhostWriter on two different writing tasks, observing that it helps users craft personalized text generations and empowers them by providing multiple ways to control the system's writing style. From this study, we present insights re
    
[^4]: AutoMathText：使用语言模型进行数学文本的自主数据选择

    AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts

    [https://arxiv.org/abs/2402.07625](https://arxiv.org/abs/2402.07625)

    本论文介绍了一种自主数据选择策略，利用语言模型进行数学文本的自动评估和选择，并通过连续预训练显著提高了数学推理能力。主要创新包括利用元提示语言模型作为验证器，发布了高质量的AutoMathText数据集，并实现了预训练令牌效率的提升。

    

    为了通过持续的预训练改善语言模型在数学推理方面的能力，我们引入了一种新颖的策略，利用基础语言模型进行自主数据选择。与传统的有人工标注数据的监督微调或训练过的分类器不同，我们的方法利用元提示语言模型作为零样本验证器，自主评估和选择高质量的数学内容，并发布了经过策划的开源AutoMathText数据集，其中包含超过200GB的数据。为了证明我们方法的有效性，我们对AutoMathText数据集进行了连续预训练，使得7B参数的Mistral语言模型在MATH数据集上的下游性能大幅提升，而令牌数量比之前的连续预训练工作减少了几个数量级。我们的方法展示了基准的预训练令牌效率提高了2倍，突显了我们方法在增强中的潜力。

    To improve language models' proficiency in mathematical reasoning via continual pretraining, we introduce a novel strategy that leverages base language models for autonomous data selection. Departing from conventional supervised fine-tuning or trained classifiers with human-annotated data, our approach utilizes meta-prompted language models as zero-shot verifiers to autonomously evaluate and select high-quality mathematical content, and we release the curated open-source AutoMathText dataset encompassing over 200GB of data. To demonstrate the efficacy of our method, we continuously pretrained a 7B-parameter Mistral language model on the AutoMathText dataset, achieving substantial improvements in downstream performance on the MATH dataset with a token amount reduced by orders of magnitude compared to previous continuous pretraining works. Our method showcases a 2 times increase in pretraining token efficiency compared to baselines, underscoring the potential of our approach in enhancing
    
[^5]: 大型语言模型：一项调查

    Large Language Models: A Survey

    [https://arxiv.org/abs/2402.06196](https://arxiv.org/abs/2402.06196)

    大型语言模型（LLMs）吸引了很多关注，因为它们在自然语言任务上的强大表现。该研究领域发展迅速，包括了各种著名的LLMs、构建和增强LLMs的技术、以及流行的LLM数据集和评估指标。

    

    大型语言模型（LLMs）由于其在各种自然语言任务上的出色表现而受到了很多关注，自2022年11月ChatGPT发布以来。LLMs通过在大量文本数据上训练模型的数十亿参数来获得广泛的通用语言理解和生成能力，这符合缩放定律的预测。LLMs的研究领域尽管非常新，但在许多不同方面正在快速发展。在本文中，我们回顾了一些最著名的LLMs，包括三个流行的LLM系列（GPT、LLaMA、PaLM），并讨论了它们的特点、贡献和限制。我们还概述了构建和增强LLMs的技术。然后，我们调查了为LLM训练、微调和评估准备的流行数据集，审查了广泛使用的LLM评估指标，并比较了几个流行LLM在一组代表性基准上的性能。

    Large Language Models (LLMs) have drawn a lot of attention due to their strong performance on a wide range of natural language tasks, since the release of ChatGPT in November 2022. LLMs' ability of general-purpose language understanding and generation is acquired by training billions of model's parameters on massive amounts of text data, as predicted by scaling laws \cite{kaplan2020scaling,hoffmann2022training}. The research area of LLMs, while very recent, is evolving rapidly in many different ways. In this paper, we review some of the most prominent LLMs, including three popular LLM families (GPT, LLaMA, PaLM), and discuss their characteristics, contributions and limitations. We also give an overview of techniques developed to build, and augment LLMs. We then survey popular datasets prepared for LLM training, fine-tuning, and evaluation, review widely used LLM evaluation metrics, and compare the performance of several popular LLMs on a set of representative benchmarks. Finally, we co
    
[^6]: 将“段分离任意模型”推进至高度准确的二元图像分割

    Promoting Segment Anything Model towards Highly Accurate Dichotomous Image Segmentation

    [https://arxiv.org/abs/2401.00248](https://arxiv.org/abs/2401.00248)

    将段分离任意模型推进至高度准确的二元图像分割，通过提出DIS-SAM框架，成功改进SAM模型在细节方面的表现，实现了显著增强的分割精度。

    

    Segment Anything Model (SAM)代表了计算机视觉基础模型的重大突破，提供了大规模图像分割模型。然而，尽管SAM的零-shot表现，其分割蒙版缺乏细粒度细节，特别是在准确描绘对象边界方面。我们对SAM是否可以作为基础模型进一步改进以实现高度精确的对象分割（即称为二元图像分割DIS）抱有很高期望。为解决这一问题，我们提出了DIS-SAM，将SAM推进至DIS，具有极高的精确细节。DIS-SAM是一个专门为高度准确分割而设计的框架，保持了SAM的可促进设计。DIS-SAM采用了两阶段方法，将SAM与专门用于DIS的修改后的IS-Net集成在一起。尽管简单，DIS-SAM相比SAM和HQ-SA表现出显着增强的分割精度。

    arXiv:2401.00248v2 Announce Type: replace-cross  Abstract: The Segment Anything Model (SAM) represents a significant breakthrough into foundation models for computer vision, providing a large-scale image segmentation model. However, despite SAM's zero-shot performance, its segmentation masks lack fine-grained details, particularly in accurately delineating object boundaries. We have high expectations regarding whether SAM, as a foundation model, can be improved towards highly accurate object segmentation, which is known as dichotomous image segmentation (DIS). To address this issue, we propose DIS-SAM, which advances SAM towards DIS with extremely accurate details. DIS-SAM is a framework specifically tailored for highly accurate segmentation, maintaining SAM's promptable design. DIS-SAM employs a two-stage approach, integrating SAM with a modified IS-Net dedicated to DIS. Despite its simplicity, DIS-SAM demonstrates significantly enhanced segmentation accuracy compared to SAM and HQ-SA
    
[^7]: 具有原型的跨领域随机预训练用于强化学习

    Cross-domain Random Pre-training with Prototypes for Reinforcement Learning

    [https://arxiv.org/abs/2302.05614](https://arxiv.org/abs/2302.05614)

    提出了CRPTpro框架，利用原型进行跨领域自监督随机预训练，提高预训练效率，并实现在不同领域中定义的视觉控制RL任务。

    

    此工作已提交给IEEE进行可能的出版。 CRPTpro提出了一种用于基于图像的RL的跨领域自监督随机预训练框架，利用原型。 CRPTpro采用了跨领域随机策略，可以轻松快速地从多个领域中抽样多样化数据，以提高预训练效率。此外，通过提出一种新颖的内在损失进行原型表示学习，以在不同领域中预训练有效且通用的编码器。在没有微调的情况下，跨领域编码器可以高效地应用于不同领域中定义的具有挑战性的下游视觉控制RL任务。 与以前的方法如APT和Proto-RL相比，CRP

    arXiv:2302.05614v2 Announce Type: replace-cross  Abstract: This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible. Task-agnostic cross-domain pre-training shows great potential in image-based Reinforcement Learning (RL) but poses a big challenge. In this paper, we propose CRPTpro, a Cross-domain self-supervised Random Pre-Training framework with prototypes for image-based RL. CRPTpro employs cross-domain random policy to easily and quickly sample diverse data from multiple domains, to improve pre-training efficiency. Moreover, prototypical representation learning with a novel intrinsic loss is proposed to pre-train an effective and generic encoder across different domains. Without finetuning, the cross-domain encoder can be implemented for challenging downstream visual-control RL tasks defined in different domains efficiently. Compared with prior arts like APT and Proto-RL, CRP
    
[^8]: 信用风险与大型语言模型相结合：从P2P借贷的贷款描述中构建风险指标。

    Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending. (arXiv:2401.16458v1 [q-fin.RM])

    [http://arxiv.org/abs/2401.16458](http://arxiv.org/abs/2401.16458)

    本文研究了如何利用P2P借贷平台上借款人提供的文本描述来构建风险指标。结果显示，利用大型语言模型生成的风险评分可以明显提高信用风险分类器的性能。

    

    P2P借贷作为一种独特的融资机制，通过在线平台将借款人与放款人联系起来。然而，P2P借贷面临信息不对称的挑战，因为放款人往往缺乏足够的数据来评估借款人的信用价值。本文提出了一种新颖的方法来解决这个问题，即利用借款人在贷款申请过程中提供的文本描述。我们的方法涉及使用大型语言模型（LLM）处理这些文本描述，LLM是一种能够识别文本中的模式和语义的强大工具。将迁移学习应用于将LLM适应特定任务。我们从Lending Club数据集的分析结果显示，BERT生成的风险评分显著提高了信用风险分类器的性能。然而，基于LLM的系统固有的不透明性，以及潜在偏差的不确定性，限制了其应用。

    Peer-to-peer (P2P) lending has emerged as a distinctive financing mechanism, linking borrowers with lenders through online platforms. However, P2P lending faces the challenge of information asymmetry, as lenders often lack sufficient data to assess the creditworthiness of borrowers. This paper proposes a novel approach to address this issue by leveraging the textual descriptions provided by borrowers during the loan application process. Our methodology involves processing these textual descriptions using a Large Language Model (LLM), a powerful tool capable of discerning patterns and semantics within the text. Transfer learning is applied to adapt the LLM to the specific task at hand.  Our results derived from the analysis of the Lending Club dataset show that the risk score generated by BERT, a widely used LLM, significantly improves the performance of credit risk classifiers. However, the inherent opacity of LLM-based systems, coupled with uncertainties about potential biases, unders
    
[^9]: 混合你自己的对比对

    Mixup Your Own Pairs. (arXiv:2309.16633v1 [cs.LG])

    [http://arxiv.org/abs/2309.16633](http://arxiv.org/abs/2309.16633)

    本文提出了一种名为SupReMix的方法，通过混合样本，特别是混合负样本和混合正样本，来解决回归问题中表示学习的挑战。这种方法能够提供更好的性能和更准确的回归结果。

    

    在表示学习中，回归问题传统上比分类问题受到的关注较少。直接应用为分类设计的表示学习技术到回归问题往往会导致潜空间中碎片化的表示，从而产生次优的性能。本文认为，由于忽视了两个关键方面：序序感知和难度，对于回归问题而言，对比学习的潜能被忽视了。为了解决这些挑战，我们提倡“混合自己的对比对进行监督性对比回归”，而不仅仅依靠真实/增强样本。具体来说，我们提出了混合式监督对比回归学习（SupReMix）。它在嵌入级别上以锚点包含的混合（锚点和一个不同的负样本的混合）作为困难负对，以锚点排除的混合（两个不同的负样本的混合）作为困难正对。这一策略形成了困难样本对学习的方式。

    In representation learning, regression has traditionally received less attention than classification. Directly applying representation learning techniques designed for classification to regression often results in fragmented representations in the latent space, yielding sub-optimal performance. In this paper, we argue that the potential of contrastive learning for regression has been overshadowed due to the neglect of two crucial aspects: ordinality-awareness and hardness. To address these challenges, we advocate "mixup your own contrastive pairs for supervised contrastive regression", instead of relying solely on real/augmented samples. Specifically, we propose Supervised Contrastive Learning for Regression with Mixup (SupReMix). It takes anchor-inclusive mixtures (mixup of the anchor and a distinct negative sample) as hard negative pairs and anchor-exclusive mixtures (mixup of two distinct negative samples) as hard positive pairs at the embedding level. This strategy formulates harde
    
[^10]: Notation3作为一种存在规则语言

    Notation3 as an Existential Rule Language. (arXiv:2308.07332v1 [cs.AI])

    [http://arxiv.org/abs/2308.07332](http://arxiv.org/abs/2308.07332)

    本文研究了Notation3与存在规则之间的关系，并提出了一个将部分Notation3直接映射到存在规则的方法，从而提高了Notation3推理的效率。

    

    Notation3逻辑（\nthree）是RDF的扩展，允许用户编写引入新的空白节点到RDF图中的规则。许多应用程序（例如本体映射）依赖于此功能，因为空白节点在Web上广泛存在，直接使用或作为辅助结构。然而，涵盖该逻辑非常重要功能的快速\nthree推理器的数量相对有限。另一方面，像VLog或Nemo之类的引擎不直接支持语义Web规则格式，但是它们是为非常相似的构造（存在规则）开发和优化的。在本文中，我们研究了具有空白节点的\nthree规则与存在规则之间的关系。我们确定了一个可以直接映射到存在规则的\nthree子集，并定义了这样一个映射，保持了\nthree公式的等价性。为了进一步说明在某些情况下\nthree推理可以受益于我们的转换，我们使用该映射进行分析。

    Notation3 Logic (\nthree) is an extension of RDF that allows the user to write rules introducing new blank nodes to RDF graphs. Many applications (e.g., ontology mapping) rely on this feature as blank nodes -- used directly or in auxiliary constructs -- are omnipresent on the Web. However, the number of fast \nthree reasoners covering this very important feature of the logic is rather limited. On the other hand, there are engines like VLog or Nemo which do not directly support Semantic Web rule formats but which are developed and optimized for very similar constructs: existential rules. In this paper, we investigate the relation between \nthree rules with blank nodes in their heads and existential rules. We identify a subset of \nthree which can be mapped directly to existential rules and define such a mapping preserving the equivalence of \nthree formulae. In order to also illustrate that in some cases \nthree reasoning could benefit from our translation, we then employ this mapping i
    
[^11]: 将基础模型作为代理模型引入：朝着更实用的对抗攻击迈进

    Introducing Foundation Models as Surrogate Models: Advancing Towards More Practical Adversarial Attacks. (arXiv:2307.06608v1 [cs.LG])

    [http://arxiv.org/abs/2307.06608](http://arxiv.org/abs/2307.06608)

    本文将对抗攻击重新设定为下游任务，通过生成图像噪声来满足新兴趋势，并将基础模型引入作为代理模型。虽然基础模型的表现不佳，但通过在特征空间中进行分析，我们发现缺乏对应的特征。

    

    最近，无盒对抗攻击成为了最实用且具有挑战性的攻击方式，攻击者无法访问模型的架构、权重和训练数据。然而，在无盒设置中，对于代理模型选择过程的潜力和灵活性缺乏认识。受到利用基础模型解决下游任务的兴趣的启发，本文采用了1）将对抗攻击重新设定为下游任务，具体而言，是生成图像噪声以满足新兴趋势；2）将基础模型引入作为代理模型的创新思想。通过利用非鲁棒特征的概念，我们阐述了选择代理模型的两个指导原则，以解释为什么基础模型是这一角色的最佳选择。然而，矛盾地的是，我们观察到这些基础模型表现不佳。通过在特征空间中分析这种意外行为，我们归因于缺乏上述指导原则所需的特征。

    Recently, the no-box adversarial attack, in which the attacker lacks access to the model's architecture, weights, and training data, become the most practical and challenging attack setup. However, there is an unawareness of the potential and flexibility inherent in the surrogate model selection process on no-box setting. Inspired by the burgeoning interest in utilizing foundational models to address downstream tasks, this paper adopts an innovative idea that 1) recasting adversarial attack as a downstream task. Specifically, image noise generation to meet the emerging trend and 2) introducing foundational models as surrogate models. Harnessing the concept of non-robust features, we elaborate on two guiding principles for surrogate model selection to explain why the foundational model is an optimal choice for this role. However, paradoxically, we observe that these foundational models underperform. Analyzing this unexpected behavior within the feature space, we attribute the lackluster
    
[^12]: 深度学习模型的表征状态

    The Representational Status of Deep Learning Models. (arXiv:2303.12032v1 [cs.AI])

    [http://arxiv.org/abs/2303.12032](http://arxiv.org/abs/2303.12032)

    该论文澄清了深度学习模型的表征状态。尽管通常称为“表征”，但实际上它们更适合理解为高度理想化的模型，这一结果对可解释的AI有着直接影响，也引起了哲学家对其在未来科学研究中的作用的关注。

    

    本文旨在澄清深度学习模型（DLMs）的表征状态。由于功能和关系概念的混淆，尽管通常称为“表征”，但这意味着含糊不清。本文认为，虽然DLM以关系意义上的表征其目标，但最好理解为高度理想化的模型。这个结果对可解释的AI（XAI）有直接影响，并引导哲学关注DLM表征的理想化性质及其在未来科学研究中的作用。

    This paper aims to clarify the representational status of Deep Learning Models (DLMs). While commonly referred to as 'representations', what this entails is ambiguous due to a conflation of functional and relational conceptions of representation. This paper argues that while DLMs represent their targets in a relational sense, they are best understood as highly idealized models. This result has immediate implications for explainable AI (XAI) and directs philosophical attention toward examining the idealized nature of DLM representations and their role in future scientific investigation.
    

