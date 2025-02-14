# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simplicity Bias of Transformers to Learn Low Sensitivity Functions](https://arxiv.org/abs/2403.06925) | Transformers在不同数据模态上具有低敏感性，这种简单性偏差有助于解释其在视觉和语言任务中的优越性能。 |
| [^2] | [A Cognitive Evaluation Benchmark of Image Reasoning and Description for Large Vision Language Models](https://arxiv.org/abs/2402.18409) | 提出了一个新颖的评估基准，用于评估大型视觉语言模型的认知能力，发现LVLMs与人类之间存在较大的认知能力差距。 |
| [^3] | [Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator](https://arxiv.org/abs/2402.17767) | 实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。 |
| [^4] | [Sym-Q: Adaptive Symbolic Regression via Sequential Decision-Making](https://arxiv.org/abs/2402.05306) | Sym-Q是一个基于强化学习的模型，通过将符号回归重新定义为顺序决策任务来解决现有模型在泛化性和适应性方面的挑战。通过利用监督演示和奖励信号，Sym-Q能够根据拟合精度的质量改进表达式。 |
| [^5] | [The Developmental Landscape of In-Context Learning](https://arxiv.org/abs/2402.02364) | 在transformers模型中，我们展示了在上下文学习中的离散发展阶段，并引入了两种方法来检测这些阶段的关键里程碑。我们使用行为和结构度量验证了这些方法的有效性。 |
| [^6] | [The Calibration Gap between Model and Human Confidence in Large Language Models.](http://arxiv.org/abs/2401.13835) | 该论文研究了大型语言模型在传达置信度方面模型和人类之间存在的差距，并发现默认解释会导致用户过高估计模型置信度和准确性。 |
| [^7] | [Improving Factual Consistency of Text Summarization by Adversarially Decoupling Comprehension and Embellishment Abilities of LLMs.](http://arxiv.org/abs/2310.19347) | 本文提出了一个名为DECENT的方法，通过对抗解耦LLMs的理解和修饰能力，提高文本摘要的事实一致性。同时，采用了一种探测技术来弥补训练过程中对真与假的敏感性不足的问题。 |
| [^8] | [Multi-level Asymmetric Contrastive Learning for Medical Image Segmentation Pre-training.](http://arxiv.org/abs/2309.11876) | 本论文提出了一种针对医学图像分割的自我监督预训练方法，通过多级非对称对比学习的框架，在编码器和解码器同时进行预训练，提供更好的分割模型初始化。 |
| [^9] | [Exploring Large Language Models for Knowledge Graph Completion.](http://arxiv.org/abs/2308.13916) | 本文研究了利用大型语言模型（LLM）进行知识图谱补全的方法，并引入了一种创新的框架（知识图谱LLM），以提高三元组分类和关系预测的性能。 |
| [^10] | [On the Creativity of Large Language Models.](http://arxiv.org/abs/2304.00008) | 这篇论文探讨了大型语言模型的创造性问题，分析了与之相关的机器创造性的难点和易点，并重点分析了这些技术在创意产业中的社会影响。 |
| [^11] | [Multi-modal Multi-kernel Graph Learning for Autism Prediction and Biomarker Discovery.](http://arxiv.org/abs/2303.03388) | 本文提出了一种名为MMKGL的新方法，能够解决多模态集成中各模态之间的负面影响，并从多个图中提取异质信息，以进行自闭症的预测和生物标志物的发现。 |
| [^12] | [Checking Trustworthiness of Probabilistic Computations in a Typed Natural Deduction System.](http://arxiv.org/abs/2206.12934) | 本文介绍了一种名为TPTND的概率类型自然演算系统，该系统能够检验并推导概率计算过程的可信性，具有可检查性的优势。 |
| [^13] | [MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection.](http://arxiv.org/abs/2203.13310) | 本文介绍了一种名为MonoDETR的深度引导Transformer框架，用于单目3D目标检测。相比于传统的方法，MonoDETR通过引入深度信息来指导整个检测过程，提高了对场景的理解和目标的准确性。 |

# 详细

[^1]: Transformers学习低敏感性函数的简单性偏差

    Simplicity Bias of Transformers to Learn Low Sensitivity Functions

    [https://arxiv.org/abs/2403.06925](https://arxiv.org/abs/2403.06925)

    Transformers在不同数据模态上具有低敏感性，这种简单性偏差有助于解释其在视觉和语言任务中的优越性能。

    

    Transformers在许多任务中取得了最先进的准确性和鲁棒性，但对它们具有的归纳偏差以及这些偏差如何与其他神经网络架构不同的理解仍然难以捉摸。本文中，我们将模型对输入中的随机更改的敏感性概念化为一种简单性偏差的概念，这为解释transformers在不同数据模态上的简单性和谱偏差提供了统一的度量标准。我们展示了transformers在视觉和语言任务中比其他替代架构（如LSTMs、MLPs和CNNs）具有更低的敏感性。我们还展示了低敏感性偏差与改进性能的相关性。

    arXiv:2403.06925v1 Announce Type: cross  Abstract: Transformers achieve state-of-the-art accuracy and robustness across many tasks, but an understanding of the inductive biases that they have and how those biases are different from other neural network architectures remains elusive. Various neural network architectures such as fully connected networks have been found to have a simplicity bias towards simple functions of the data; one version of this simplicity bias is a spectral bias to learn simple functions in the Fourier space. In this work, we identify the notion of sensitivity of the model to random changes in the input as a notion of simplicity bias which provides a unified metric to explain the simplicity and spectral bias of transformers across different data modalities. We show that transformers have lower sensitivity than alternative architectures, such as LSTMs, MLPs and CNNs, across both vision and language tasks. We also show that low-sensitivity bias correlates with impro
    
[^2]: 一个针对大型视觉语言模型图像推理和描述的认知评估基准

    A Cognitive Evaluation Benchmark of Image Reasoning and Description for Large Vision Language Models

    [https://arxiv.org/abs/2402.18409](https://arxiv.org/abs/2402.18409)

    提出了一个新颖的评估基准，用于评估大型视觉语言模型的认知能力，发现LVLMs与人类之间存在较大的认知能力差距。

    

    尽管大型视觉语言模型(LVLMs)近年来取得了成功，但它们很少受到全面的认知能力测试。受到人类认知测试中广泛使用的“偷饼干”任务的启发，我们提出了一个新颖的评估基准，利用具有丰富语义的图像评估LVLMs的高级认知能力。它定义了八种推理能力，并包括图像描述任务和视觉问答任务。我们对知名LVLMs进行的评估表明，在LVLMs和人类之间仍存在较大的认知能力差距。

    arXiv:2402.18409v1 Announce Type: new  Abstract: Large Vision Language Models (LVLMs), despite their recent success, are hardly comprehensively tested for their cognitive abilities. Inspired by the prevalent use of the "Cookie Theft" task in human cognition test, we propose a novel evaluation benchmark to evaluate high-level cognitive ability of LVLMs using images with rich semantics. It defines eight reasoning capabilities and consists of an image description task and a visual question answering task. Our evaluation on well-known LVLMs shows that there is still a large gap in cognitive ability between LVLMs and humans.
    
[^3]: 在现实世界中使用商品移动操作器打开橱柜和抽屉

    Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator

    [https://arxiv.org/abs/2402.17767](https://arxiv.org/abs/2402.17767)

    实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。

    

    在这项工作中，我们构建了一个端到端系统，使商品移动操作器（Stretch RE2）能够在多样的以前未见的真实世界环境中拉开橱柜和抽屉。我们在31个不同的物体和13个不同真实世界环境中进行了4天的实际测试。我们的系统在零击打下，对在未知环境中新颖的橱柜和抽屉的打开率达到61%。对失败模式的分析表明，感知误差是我们系统面临的最重要挑战。

    arXiv:2402.17767v1 Announce Type: cross  Abstract: Pulling open cabinets and drawers presents many difficult technical challenges in perception (inferring articulation parameters for objects from onboard sensors), planning (producing motion plans that conform to tight task constraints), and control (making and maintaining contact while applying forces on the environment). In this work, we build an end-to-end system that enables a commodity mobile manipulator (Stretch RE2) to pull open cabinets and drawers in diverse previously unseen real world environments. We conduct 4 days of real world testing of this system spanning 31 different objects from across 13 different real world environments. Our system achieves a success rate of 61% on opening novel cabinets and drawers in unseen environments zero-shot. An analysis of the failure modes suggests that errors in perception are the most significant challenge for our system. We will open source code and models for others to replicate and bui
    
[^4]: Sym-Q：通过顺序决策进行自适应符号回归

    Sym-Q: Adaptive Symbolic Regression via Sequential Decision-Making

    [https://arxiv.org/abs/2402.05306](https://arxiv.org/abs/2402.05306)

    Sym-Q是一个基于强化学习的模型，通过将符号回归重新定义为顺序决策任务来解决现有模型在泛化性和适应性方面的挑战。通过利用监督演示和奖励信号，Sym-Q能够根据拟合精度的质量改进表达式。

    

    符号回归具有从实证数据中揭示潜在数学和物理关系的巨大潜力。虽然现有的基于Transformer的模型在这个领域取得了显著成功，但它们在泛化性和适应性方面面临挑战。通常，当输出表达式不足以适应实验数据时，这些模型缺乏有效的机制来适应或修改表达式。这种缺乏灵活性限制了它们在实际场景中的应用，特别是在发现未知的物理或生物关系方面。受到人类专家如何改进和调整表达式的启发，我们引入了一种新颖的基于强化学习的模型Symbolic Q-network（Sym-Q），将符号回归重新定义为顺序决策任务。Sym-Q利用监督演示并根据奖励信号来改进表达式，奖励信号指示拟合精度的质量。它独特的能力可以处理复杂性。

    Symbolic regression holds great potential for uncovering underlying mathematical and physical relationships from empirical data. While existing transformer-based models have recently achieved significant success in this domain, they face challenges in terms of generalizability and adaptability. Typically, in cases where the output expressions do not adequately fit experimental data, the models lack efficient mechanisms to adapt or modify the expression. This inflexibility hinders their application in real-world scenarios, particularly in discovering unknown physical or biological relationships. Inspired by how human experts refine and adapt expressions, we introduce Symbolic Q-network (Sym-Q), a novel reinforcement learning-based model that redefines symbolic regression as a sequential decision-making task. Sym-Q leverages supervised demonstrations and refines expressions based on reward signals indicating the quality of fitting precision. Its distinctive ability to manage the complexi
    
[^5]: 在上下文中学习的发展景观

    The Developmental Landscape of In-Context Learning

    [https://arxiv.org/abs/2402.02364](https://arxiv.org/abs/2402.02364)

    在transformers模型中，我们展示了在上下文学习中的离散发展阶段，并引入了两种方法来检测这些阶段的关键里程碑。我们使用行为和结构度量验证了这些方法的有效性。

    

    我们展示了在transformers中，当它们通过语言建模或线性回归任务进行训练时，上下文学习是如何以离散的发展阶段出现的。我们引入了两种方法来检测分隔这些阶段的关键里程碑，通过探测参数空间和函数空间中种群损失的几何特征。我们使用一系列行为和结构度量研究这些新方法揭示的阶段，以建立它们的有效性。

    We show that in-context learning emerges in transformers in discrete developmental stages, when they are trained on either language modeling or linear regression tasks. We introduce two methods for detecting the milestones that separate these stages, by probing the geometry of the population loss in both parameter space and function space. We study the stages revealed by these new methods using a range of behavioral and structural metrics to establish their validity.
    
[^6]: 语言模型中模型和人类置信度之间的校准差距

    The Calibration Gap between Model and Human Confidence in Large Language Models. (arXiv:2401.13835v1 [cs.LG])

    [http://arxiv.org/abs/2401.13835](http://arxiv.org/abs/2401.13835)

    该论文研究了大型语言模型在传达置信度方面模型和人类之间存在的差距，并发现默认解释会导致用户过高估计模型置信度和准确性。

    

    为了使大型语言模型（LLM）能够获得人类的信任，它们需要在某种意义上实现良好的校准，即能够准确评估和传达它们的预测正确的可能性。最近的研究关注了LLM内部置信度评估的质量，但问题仍然是LLM能够如何将这种内部模型置信度传达给人类用户。本文探讨了人类对LLM响应的外部置信度与模型内部置信度之间的差距。通过涉及多项选择题的实验，我们系统地检查了人类用户识别LLM输出可信度的能力。我们的研究重点分为两个方面：（1）评估用户对真实LLM置信度的感知和（2）调查个性化解释对该感知的影响。研究结果显示，LLM的默认解释往往会导致用户过高估计模型的置信度和准确性。通过修改解释的方式可以减小这种误差。

    For large language models (LLMs) to be trusted by humans they need to be well-calibrated in the sense that they can accurately assess and communicate how likely it is that their predictions are correct. Recent work has focused on the quality of internal LLM confidence assessments, but the question remains of how well LLMs can communicate this internal model confidence to human users. This paper explores the disparity between external human confidence in an LLM's responses and the internal confidence of the model. Through experiments involving multiple-choice questions, we systematically examine human users' ability to discern the reliability of LLM outputs. Our study focuses on two key areas: (1) assessing users' perception of true LLM confidence and (2) investigating the impact of tailored explanations on this perception. The research highlights that default explanations from LLMs often lead to user overestimation of both the model's confidence and its' accuracy. By modifying the expl
    
[^7]: 通过对LLMs的理解和修饰能力进行对抗解耦，提高文本摘要的事实一致性改进

    Improving Factual Consistency of Text Summarization by Adversarially Decoupling Comprehension and Embellishment Abilities of LLMs. (arXiv:2310.19347v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.19347](http://arxiv.org/abs/2310.19347)

    本文提出了一个名为DECENT的方法，通过对抗解耦LLMs的理解和修饰能力，提高文本摘要的事实一致性。同时，采用了一种探测技术来弥补训练过程中对真与假的敏感性不足的问题。

    

    尽管大型语言模型（LLMs）在文本摘要方面取得了近期的进展，但它们经常会生成与原始文章事实不一致的摘要，被称为文本生成中的“幻觉”。与之前的小型模型（如BART，T5）不同，当前的LLMs在制造愚蠢错误方面较少，但制造了更复杂的错误，例如加入因果关系、添加错误细节和过度泛化等。这些幻觉很难通过传统方法检测出来，这给提高文本摘要的事实一致性带来了很大挑战。在本文中，我们提出了一种对抗解耦方法来分离LLMs的理解和修饰能力（DECENT）。此外，我们采用一种基于探测的参数高效技术，以弥补LLMs在训练过程中对真与假的敏感性不足的问题。通过这种方式，LLMs对于修饰和理解的概念更加清晰，从而能够更准确地执行指令。

    Despite the recent progress in text summarization made by large language models (LLMs), they often generate summaries that are factually inconsistent with original articles, known as "hallucinations" in text generation. Unlike previous small models (e.g., BART, T5), current LLMs make fewer silly mistakes but more sophisticated ones, such as imposing cause and effect, adding false details, and overgeneralizing, etc. These hallucinations are challenging to detect through traditional methods, which poses great challenges for improving the factual consistency of text summarization. In this paper, we propose an adversarially DEcoupling method to disentangle the Comprehension and EmbellishmeNT abilities of LLMs (DECENT). Furthermore, we adopt a probing-based parameter-efficient technique to cover the shortage of sensitivity for true and false in the training process of LLMs. In this way, LLMs are less confused about embellishing and understanding, thus can execute the instructions more accur
    
[^8]: 多级非对称对比学习在医学图像分割预训练中的应用

    Multi-level Asymmetric Contrastive Learning for Medical Image Segmentation Pre-training. (arXiv:2309.11876v1 [cs.CV])

    [http://arxiv.org/abs/2309.11876](http://arxiv.org/abs/2309.11876)

    本论文提出了一种针对医学图像分割的自我监督预训练方法，通过多级非对称对比学习的框架，在编码器和解码器同时进行预训练，提供更好的分割模型初始化。

    

    对比学习是一种从无标签数据中学习图像级表示的强大技术，为解决大规模预训练和有限标注数据之间的困境提供了一种有前途的方法。然而，大多数现有的对比学习策略主要针对自然图像的下游任务设计，因此当直接应用于医学图像（其下游任务通常是分割）时，它们往往是次优的甚至不如从头开始训练。在这项工作中，我们提出了一种名为JCL的新型非对称对比学习框架，用于医学图像分割的自我监督预训练。具体来说，（1）我们提出了一种新颖的非对称对比学习策略，同时在一阶段内对编码器和解码器进行预训练，以提供更好的分割模型初始化。 （2）我们设计了一个多级对比损失，用于考虑特征级别、图像级别和像素级别投影的对应关系。

    Contrastive learning, which is a powerful technique for learning image-level representations from unlabeled data, leads a promising direction to dealing with the dilemma between large-scale pre-training and limited labeled data. However, most existing contrastive learning strategies are designed mainly for downstream tasks of natural images, therefore they are sub-optimal and even worse than learning from scratch when directly applied to medical images whose downstream tasks are usually segmentation. In this work, we propose a novel asymmetric contrastive learning framework named JCL for medical image segmentation with self-supervised pre-training. Specifically, (1) A novel asymmetric contrastive learning strategy is proposed to pre-train both encoder and decoder simultaneously in one-stage to provide better initialization for segmentation models. (2) A multi-level contrastive loss is designed to take the correspondence among feature-level, image-level and pixel-level projections, resp
    
[^9]: 探索大型语言模型用于知识图谱补全

    Exploring Large Language Models for Knowledge Graph Completion. (arXiv:2308.13916v1 [cs.CL])

    [http://arxiv.org/abs/2308.13916](http://arxiv.org/abs/2308.13916)

    本文研究了利用大型语言模型（LLM）进行知识图谱补全的方法，并引入了一种创新的框架（知识图谱LLM），以提高三元组分类和关系预测的性能。

    

    知识图谱在众多人工智能任务中发挥着重要作用，但经常面临不完整性的问题。在本研究中，我们探索了利用大型语言模型（LLM）进行知识图谱补全的方法。我们将知识图谱中的三元组视为文本序列，并引入了一种创新的框架，称为知识图谱LLM（KG-LLM），来对这些三元组进行建模。我们的技术利用三元组的实体和关系描述作为提示，并利用响应进行预测。对各种基准知识图谱的实验表明，我们的方法在三元组分类和关系预测等任务中达到了最先进的性能。我们还发现，微调相对较小的模型（例如LLaMA-7B，ChatGLM-6B）优于最新的ChatGPT和GPT-4。

    Knowledge graphs play a vital role in numerous artificial intelligence tasks, yet they frequently face the issue of incompleteness. In this study, we explore utilizing Large Language Models (LLM) for knowledge graph completion. We consider triples in knowledge graphs as text sequences and introduce an innovative framework called Knowledge Graph LLM (KG-LLM) to model these triples. Our technique employs entity and relation descriptions of a triple as prompts and utilizes the response for predictions. Experiments on various benchmark knowledge graphs demonstrate that our method attains state-of-the-art performance in tasks such as triple classification and relation prediction. We also find that fine-tuning relatively smaller models (e.g., LLaMA-7B, ChatGLM-6B) outperforms recent ChatGPT and GPT-4.
    
[^10]: 关于大型语言模型的创造性研究

    On the Creativity of Large Language Models. (arXiv:2304.00008v1 [cs.AI])

    [http://arxiv.org/abs/2304.00008](http://arxiv.org/abs/2304.00008)

    这篇论文探讨了大型语言模型的创造性问题，分析了与之相关的机器创造性的难点和易点，并重点分析了这些技术在创意产业中的社会影响。

    

    大型语言模型(LLMs)正在颠覆人工智能的多个领域。其中最显著的应用之一是创作，例如诗歌或故事：生成的输出通常具有惊人的质量。但是，一个自然的问题是：LLMs真的可以被认为是创造性的吗？在本文中，我们首先通过创造性理论的角度分析了LLMs的发展，探讨了关键的未解决问题和挑战。然后，我们在与LLMs相关的机器创造性方面确定了一组“易”和“难”问题，并对其进行了讨论。最后，我们分析了这些技术在创意产业中的社会影响。

    Large Language Models (LLMs) are revolutionizing several areas of Artificial Intelligence. One of the most remarkable applications is creative writing, e.g., poetry or storytelling: the generated outputs are often of astonishing quality. However, a natural question arise: can LLMs really be considered creative? In this article we firstly analyze the development of LLMs under the lens of creativity theories, investigating the key open questions and challenges. Then, we identify a set of "easy" and "hard" problems in machine creativity, discussing them in relation to LLMs. Finally, we analyze the societal impact of these technologies with a particular focus on the creative industries.
    
[^11]: 基于多模态多核图学习的自闭症预测与生物标志物发现

    Multi-modal Multi-kernel Graph Learning for Autism Prediction and Biomarker Discovery. (arXiv:2303.03388v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.03388](http://arxiv.org/abs/2303.03388)

    本文提出了一种名为MMKGL的新方法，能够解决多模态集成中各模态之间的负面影响，并从多个图中提取异质信息，以进行自闭症的预测和生物标志物的发现。

    

    基于图学习的多模态集成和分类是疾病预测中最具挑战性的障碍之一。我们提出了一种名为MMKGL的新方法来有效抵消多模态集成过程中各模态之间负面影响，并从图中提取异质信息。具体地，我们提出了多模态图嵌入模块，并通过自适应学习生成多个图，然后提出多核图学习模块，从多模态图中提取异质信息。在不同层次上聚合多模态图中的信息，实现了对自闭症的预测和生物标志物的发现。

    Due to its complexity, graph learning-based multi-modal integration and classification is one of the most challenging obstacles for disease prediction. To effectively offset the negative impact between modalities in the process of multi-modal integration and extract heterogeneous information from graphs, we propose a novel method called MMKGL (Multi-modal Multi-Kernel Graph Learning). For the problem of negative impact between modalities, we propose a multi-modal graph embedding module to construct a multi-modal graph. Different from conventional methods that manually construct static graphs for all modalities, each modality generates a separate graph by adaptive learning, where a function graph and a supervision graph are introduced for optimization during the multi-graph fusion embedding process. We then propose a multi-kernel graph learning module to extract heterogeneous information from the multi-modal graph. The information in the multi-modal graph at different levels is aggregat
    
[^12]: 在类型化自然演算系统中检验概率计算的可信性

    Checking Trustworthiness of Probabilistic Computations in a Typed Natural Deduction System. (arXiv:2206.12934v2 [cs.LO] UPDATED)

    [http://arxiv.org/abs/2206.12934](http://arxiv.org/abs/2206.12934)

    本文介绍了一种名为TPTND的概率类型自然演算系统，该系统能够检验并推导概率计算过程的可信性，具有可检查性的优势。

    

    本文介绍了一种名为 TPTND 的概率类型自然演算系统，该系统旨在推导有关概率计算过程的可信性属性，例如当今人工智能应用程序中的那些属性。TPTND 中的推导被解释为从给定的分类分布中提取 n 个可能复杂输出样本的过程。我们将这些输出样本的可信性形式化为一种假设测试，即计算出现有的频率与预期的概率之间的距离。这个演算系统的主要优势在于能够检查这种可信性的概念。我们为推理过程中出现的项提供了计算语义，并定义了逻辑运算符以及信任运算符的引入和消解规则。我们重点介绍了系统的结构和元理论属性，尤其是能够确定哪些项演化和逻辑规则应用时，计算仍然是可信的。

    In this paper we present the probabilistic typed natural deduction calculus TPTND, designed to reason about and derive trustworthiness properties of probabilistic computational processes, like those underlying current AI applications. Derivability in TPTND is interpreted as the process of extracting $n$ samples of possibly complex outputs with a certain frequency from a given categorical distribution. We formalize trust for such outputs as a form of hypothesis testing on the distance between such frequency and the intended probability. The main advantage of the calculus is to render such notion of trustworthiness checkable. We present a computational semantics for the terms over which we reason and then the semantics of TPTND, where logical operators as well as a Trust operator are defined through introduction and elimination rules. We illustrate structural and metatheoretical properties, with particular focus on the ability to establish under which term evolutions and logical rules ap
    
[^13]: MonoDETR：深度引导的单目3D目标检测的Transformer

    MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection. (arXiv:2203.13310v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2203.13310](http://arxiv.org/abs/2203.13310)

    本文介绍了一种名为MonoDETR的深度引导Transformer框架，用于单目3D目标检测。相比于传统的方法，MonoDETR通过引入深度信息来指导整个检测过程，提高了对场景的理解和目标的准确性。

    

    单目三维目标检测一直是自动驾驶中一项具有挑战性的任务。大多数现有方法是根据传统的二维检测器首先定位目标中心，然后通过邻近特征预测三维属性。然而，仅仅使用局部视觉特征是不足以理解场景级别的三维空间结构并忽略了远距离的目标深度关系。在本文中，我们引入了第一个采用深度引导Transformer的单目检测框架，称为MonoDETR。我们将基本的Transformer进行了修改，使其具有深度感知，并通过上下文深度线索来指导整个检测过程。具体而言，在捕捉物体外观的视觉编码器的同时，我们引入了预测前景深度图，并专门设计了一个深度编码器来提取非局部深度嵌入。然后，我们将三维目标候选物形式化为可学习的查询，并提出了一个深度引导的解码器来进行目标-场景深度交互。通过这种方式，每个目标都可以得到更全面的深度感知和更准确的三维检测结果。

    Monocular 3D object detection has long been a challenging task in autonomous driving. Most existing methods follow conventional 2D detectors to first localize object centers, and then predict 3D attributes by neighboring features. However, only using local visual features is insufficient to understand the scene-level 3D spatial structures and ignores the long-range inter-object depth relations. In this paper, we introduce the first DETR framework for Monocular DEtection with a depth-guided TRansformer, named MonoDETR. We modify the vanilla transformer to be depth-aware and guide the whole detection process by contextual depth cues. Specifically, concurrent to the visual encoder that captures object appearances, we introduce to predict a foreground depth map, and specialize a depth encoder to extract non-local depth embeddings. Then, we formulate 3D object candidates as learnable queries and propose a depth-guided decoder to conduct object-scene depth interactions. In this way, each obj
    

