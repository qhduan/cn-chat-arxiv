# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Instruction-Driven Game Engines on Large Language Models](https://arxiv.org/abs/2404.00276) | 通过在大型语言模型上开发指令驱动游戏引擎，使用户可以通过简单的自然语言指令创建游戏，从而降低游戏开发的难度。 |
| [^2] | [Robust Diffusion Models for Adversarial Purification](https://arxiv.org/abs/2403.16067) | 提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。 |
| [^3] | [Exploring Multi-modal Neural Scene Representations With Applications on Thermal Imaging](https://arxiv.org/abs/2403.11865) | 本文对神经场景表示在多模态学习中的应用进行了全面评估，并提出了四种不同的策略以将第二模态（非RGB）纳入NeRFs中，通过选择热成像作为第二模态来挑战神经场景表示的整合。 |
| [^4] | [Self-Supervised Multiple Instance Learning for Acute Myeloid Leukemia Classification](https://arxiv.org/abs/2403.05379) | 自本研究发现自监督预训练编码器在多实例学习中实现了可比较的性能，展示了自监督学习在急性髓细胞白血病分类中的潜力，这为一种经济高效且节约数据的解决方案。 |
| [^5] | [Quantum Many-Body Physics Calculations with Large Language Models](https://arxiv.org/abs/2403.03154) | 使用大型语言模型准确地执行理论物理研究论文中关键的Hartree-Fock方法计算。 |
| [^6] | [Distillation Contrastive Decoding: Improving LLMs Reasoning with Contrastive Decoding and Distillation](https://arxiv.org/abs/2402.14874) | 该研究提出了一种叫做蒸馏对比解码（DCD）的方法，通过结合对比提示与蒸馏技术，有效提升了大型语言模型（LLM）在推理任务上的性能表现，超过了传统的对比解码方法，并在多个基准数据集上取得了显著成果。 |
| [^7] | [Federated Complex Qeury Answering](https://arxiv.org/abs/2402.14609) | 研究了在多源知识图谱上回答复杂查询的联邦式方法，解决了知识图谱中的隐私保护和答案检索的挑战 |
| [^8] | [Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization](https://arxiv.org/abs/2401.16352) | 提出了一种新的对净化的对抗训练（AToP）流程，通过随机转换的扰动破坏和通过对抗损失微调净化器模型，同时提升了鲁棒性和泛化性能。 |
| [^9] | [Rational Sensibility: LLM Enhanced Empathetic Response Generation Guided by Self-presentation Theory.](http://arxiv.org/abs/2312.08702) | 本文通过以自我展示理论为指导，设计了一种创新的分类方法，将历史对话分成合理和理性的句子，并通过注意力机制来阐明上下文，从而增强同理心回应生成的能力。 |
| [^10] | [Model Merging by Uncertainty-Based Gradient Matching.](http://arxiv.org/abs/2310.12808) | 本论文通过不确定性梯度匹配的方法，提出了一种新的模型合并方案，该方案能够减少梯度不匹配，从而提高了模型合并的性能并对超参数更具鲁棒性。 |
| [^11] | [OpsEval: A Comprehensive Task-Oriented AIOps Benchmark for Large Language Models.](http://arxiv.org/abs/2310.07637) | OpsEval是一个全面任务导向的AIOps基准测试，评估了大型语言模型在有线网络操作、5G通信操作和数据库操作等关键场景下的能力水平，为提供针对AIOps定制的LLMs的优化方向。 |
| [^12] | [On the Vulnerability of Fairness Constrained Learning to Malicious Noise.](http://arxiv.org/abs/2307.11892) | 这项研究考虑了公正约束学习对恶意噪声的脆弱性，发现使用随机分类器可以在精度上只损失$\Theta(\alpha)$和$O(\sqrt{\alpha})$，对应不同的公正约束要求。 |
| [^13] | [DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model.](http://arxiv.org/abs/2306.01001) | 本文提出了一种扩散模型中的负荷预测不确定性量化方法，采用Seq2Seq网络结构来分离两种类型的不确定性并处理异常情况，不仅着眼于预测条件期望值。 |
| [^14] | [On the Interdependence of Reliance Behavior and Accuracy in AI-Assisted Decision-Making.](http://arxiv.org/abs/2304.08804) | 该论文分析了AI辅助决策中依赖行为和准确性之间的相互关系，并提出了一个视觉框架来更好地理解这种关系。该框架揭示了当人类在决策中过度依赖AI时，改善信任可能会降低准确性的有趣属性。 |

# 详细

[^1]: 大型语言模型上的指令驱动游戏引擎

    Instruction-Driven Game Engines on Large Language Models

    [https://arxiv.org/abs/2404.00276](https://arxiv.org/abs/2404.00276)

    通过在大型语言模型上开发指令驱动游戏引擎，使用户可以通过简单的自然语言指令创建游戏，从而降低游戏开发的难度。

    

    Instruction-Driven Game Engine (IDGE) 项目旨在通过使大型语言模型（LLM）遵循自由形式的游戏规则并自动生成游戏过程来使游戏开发民主化。IDGE允许用户通过发出简单的自然语言指令来创建游戏，从而显著降低了游戏开发的障碍。我们将IDGE的学习过程视为下一个状态预测任务，模型自回归地预测玩家行动给出的游戏状态。这是一个具有挑战性的任务，因为游戏状态的计算必须准确；否则，轻微的错误可能会破坏游戏过程。为了解决这个问题，我们以课程方式训练IDGE，逐渐增加模型对复杂场景的接触。

    arXiv:2404.00276v1 Announce Type: new  Abstract: The Instruction-Driven Game Engine (IDGE) project aims to democratize game development by enabling a large language model (LLM) to follow free-form game rules and autonomously generate game-play processes. The IDGE allows users to create games by issuing simple natural language instructions, which significantly lowers the barrier for game development. We approach the learning process for IDGEs as a Next State Prediction task, wherein the model autoregressively predicts in-game states given player actions. It is a challenging task because the computation of in-game states must be precise; otherwise, slight errors could disrupt the game-play. To address this, we train the IDGE in a curriculum manner that progressively increases the model's exposure to complex scenarios.   Our initial progress lies in developing an IDGE for Poker, a universally cherished card game. The engine we've designed not only supports a wide range of poker variants b
    
[^2]: 针对对抗净化的强大扩散模型

    Robust Diffusion Models for Adversarial Purification

    [https://arxiv.org/abs/2403.16067](https://arxiv.org/abs/2403.16067)

    提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。

    

    基于扩散模型（DM）的对抗净化（AP）已被证明是对抗训练（AT）最有力的替代方法。然而，这些方法忽略了预训练的扩散模型本身对对抗攻击并不稳健这一事实。此外，扩散过程很容易破坏语义信息，在反向过程后生成高质量图像但与原始输入图像完全不同，导致标准精度下降。为了解决这些问题，一个自然的想法是利用对抗训练策略重新训练或微调预训练的扩散模型，然而这在计算上是禁止的。我们提出了一种新颖的具有对抗引导的稳健反向过程，它独立于给定的预训练DMs，并且避免了重新训练或微调DMs。这种强大的引导不仅可以确保生成的净化示例保留更多的语义内容，还可以...

    arXiv:2403.16067v1 Announce Type: cross  Abstract: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also m
    
[^3]: 利用热成像探索多模态神经场景表示并应用

    Exploring Multi-modal Neural Scene Representations With Applications on Thermal Imaging

    [https://arxiv.org/abs/2403.11865](https://arxiv.org/abs/2403.11865)

    本文对神经场景表示在多模态学习中的应用进行了全面评估，并提出了四种不同的策略以将第二模态（非RGB）纳入NeRFs中，通过选择热成像作为第二模态来挑战神经场景表示的整合。

    

    神经辐射场（NeRFs）在一组RGB图像上训练时迅速发展为新的事实标准，用于新视角合成任务。本文在多模态学习的背景下对神经场景表示（如NeRFs）进行了全面评估。具体而言，我们提出了四种不同的策略，用于如何将第二模态（非RGB）纳入NeRFs中：（1）独立地从头训练每种模态；（2）在RGB上进行预训练，然后在第二模态上进行微调；（3）添加第二分支；（4）添加一个单独的组件来预测（颜色）额外模态的值。我们选择了热成像作为第二模态，因为从辐射度来看，它与RGB有很大差异，这使得将其整合到神经场景表示中具有挑战性。

    arXiv:2403.11865v1 Announce Type: cross  Abstract: Neural Radiance Fields (NeRFs) quickly evolved as the new de-facto standard for the task of novel view synthesis when trained on a set of RGB images. In this paper, we conduct a comprehensive evaluation of neural scene representations, such as NeRFs, in the context of multi-modal learning. Specifically, we present four different strategies of how to incorporate a second modality, other than RGB, into NeRFs: (1) training from scratch independently on both modalities; (2) pre-training on RGB and fine-tuning on the second modality; (3) adding a second branch; and (4) adding a separate component to predict (color) values of the additional modality. We chose thermal imaging as second modality since it strongly differs from RGB in terms of radiosity, making it challenging to integrate into neural scene representations. For the evaluation of the proposed strategies, we captured a new publicly available multi-view dataset, ThermalMix, consisti
    
[^4]: 自监督多实例学习用于急性髓细胞白血病分类

    Self-Supervised Multiple Instance Learning for Acute Myeloid Leukemia Classification

    [https://arxiv.org/abs/2403.05379](https://arxiv.org/abs/2403.05379)

    自本研究发现自监督预训练编码器在多实例学习中实现了可比较的性能，展示了自监督学习在急性髓细胞白血病分类中的潜力，这为一种经济高效且节约数据的解决方案。

    

    自动疾病诊断使用医学图像分析依赖深度学习，通常需要大量标记数据集进行监督模型训练。急性髓细胞白血病（AML）等疾病由于在单个细胞水平上稀缺且昂贵的标注而面临挑战。多实例学习（MIL）解决了弱标记场景，但通常需要用标记数据训练的强大编码器。在本研究中，我们探索了自监督学习（SSL）作为基于MIL的AML亚型分类的预训练方法，从血涂片中去除了编码器训练期间的标记数据需求。我们研究了三种最先进的SSL方法SimCLR、SwAV和DINO，并将它们的性能与监督预训练进行了比较。我们的研究结果表明，SSL预训练编码器实现了可比较的性能，展示了SSL在MIL中的潜力。这一突破提供了一种经济高效且节约数据的解决方案，

    arXiv:2403.05379v1 Announce Type: cross  Abstract: Automated disease diagnosis using medical image analysis relies on deep learning, often requiring large labeled datasets for supervised model training. Diseases like Acute Myeloid Leukemia (AML) pose challenges due to scarce and costly annotations on a single-cell level. Multiple Instance Learning (MIL) addresses weakly labeled scenarios but necessitates powerful encoders typically trained with labeled data. In this study, we explore Self-Supervised Learning (SSL) as a pre-training approach for MIL-based AML subtype classification from blood smears, removing the need for labeled data during encoder training. We investigate the three state-of-the-art SSL methods SimCLR, SwAV, and DINO, and compare their performance against supervised pre-training. Our findings show that SSL-pretrained encoders achieve comparable performance, showcasing the potential of SSL in MIL. This breakthrough offers a cost-effective and data-efficient solution, pr
    
[^5]: 使用大型语言模型进行量子多体物理计算

    Quantum Many-Body Physics Calculations with Large Language Models

    [https://arxiv.org/abs/2403.03154](https://arxiv.org/abs/2403.03154)

    使用大型语言模型准确地执行理论物理研究论文中关键的Hartree-Fock方法计算。

    

    大型语言模型（LLMs）展示了在多个领域（包括数学和科学推理）执行复杂任务的前所未有能力。我们证明，通过精心设计的提示，LLMs可以准确地执行理论物理研究论文中的关键计算。我们专注于量子物理中广泛使用的近似方法：Hartree-Fock方法，该方法需要进行分析性的多步计算，导出近似哈密顿量和相应的自洽方程。为了使用LLMs进行计算，我们设计了多步提示模板，将分析计算拆分为标准步骤，并为问题特定信息留出占位符。我们评估了GPT-4在执行过去十年的15篇研究论文中的计算表现，结果表明，通过修正中间步骤，它可以正确地推导出最终的Hartree-Fock哈密顿量。

    arXiv:2403.03154v1 Announce Type: cross  Abstract: Large language models (LLMs) have demonstrated an unprecedented ability to perform complex tasks in multiple domains, including mathematical and scientific reasoning. We demonstrate that with carefully designed prompts, LLMs can accurately carry out key calculations in research papers in theoretical physics. We focus on a broadly used approximation method in quantum physics: the Hartree-Fock method, requiring an analytic multi-step calculation deriving approximate Hamiltonian and corresponding self-consistency equations. To carry out the calculations using LLMs, we design multi-step prompt templates that break down the analytic calculation into standardized steps with placeholders for problem-specific information. We evaluate GPT-4's performance in executing the calculation for 15 research papers from the past decade, demonstrating that, with correction of intermediate steps, it can correctly derive the final Hartree-Fock Hamiltonian i
    
[^6]: 蒸馏对比解码：利用对比解码和蒸馏提升LLM的推理能力

    Distillation Contrastive Decoding: Improving LLMs Reasoning with Contrastive Decoding and Distillation

    [https://arxiv.org/abs/2402.14874](https://arxiv.org/abs/2402.14874)

    该研究提出了一种叫做蒸馏对比解码（DCD）的方法，通过结合对比提示与蒸馏技术，有效提升了大型语言模型（LLM）在推理任务上的性能表现，超过了传统的对比解码方法，并在多个基准数据集上取得了显著成果。

    

    我们提出了一种称为蒸馏对比解码（DCD）的简单方法，以增强大型语言模型（LLMs）在推理过程中的推理能力。与先前依赖于较小的业余模型或隐藏状态差异分析的方法不同，DCD采用了对比式思维引导和先进的蒸馏技术，包括Dropout和量化。这种方法有效地解决了对比解码（CD）的局限性，后者通常需要专家和业余模型，从而增加计算资源需求。通过将对比提示与蒸馏相结合，DCD消除了对业余模型的需求并减少了内存使用。我们的评估表明，DCD显著增强了LLM在各种推理基准测试中的性能，在GSM8K和StrategyQA数据集中均超过了CD和现有方法。

    arXiv:2402.14874v1 Announce Type: cross  Abstract: We propose a straightforward approach called Distillation Contrastive Decoding (DCD) to enhance the reasoning capabilities of Large Language Models (LLMs) during inference. In contrast to previous approaches that relied on smaller amateur models or analysis of hidden state differences, DCD employs Contrastive Chain-of-thought Prompting and advanced distillation techniques, including Dropout and Quantization. This approach effectively addresses the limitations of Contrastive Decoding (CD), which typically requires both an expert and an amateur model, thus increasing computational resource demands. By integrating contrastive prompts with distillation, DCD obviates the need for an amateur model and reduces memory usage. Our evaluations demonstrate that DCD significantly enhances LLM performance across a range of reasoning benchmarks, surpassing both CD and existing methods in the GSM8K and StrategyQA datasets.
    
[^7]: 联邦式复杂查询答案方法研究

    Federated Complex Qeury Answering

    [https://arxiv.org/abs/2402.14609](https://arxiv.org/abs/2402.14609)

    研究了在多源知识图谱上回答复杂查询的联邦式方法，解决了知识图谱中的隐私保护和答案检索的挑战

    

    知识图谱中的复杂逻辑查询答案是一个具有挑战性的任务，已经得到广泛研究。执行复杂逻辑推理的能力是必不可少的，并支持各种基于图推理的下游任务，比如搜索引擎。最近提出了一些方法，将知识图谱实体和逻辑查询表示为嵌入向量，并从知识图谱中找到逻辑查询的答案。然而，现有的方法主要集中在查询单个知识图谱上，并不能应用于多个图形。此外，直接共享带有敏感信息的知识图谱可能会带来隐私风险，使得共享和构建一个聚合知识图谱用于推理以检索查询答案是不切实际的。因此，目前仍然不清楚如何在多源知识图谱上回答查询。一个实体可能涉及到多个知识图谱，对多个知识图谱进行推理，并在多源知识图谱上回答复杂查询对于发现知识是重要的。

    arXiv:2402.14609v1 Announce Type: cross  Abstract: Complex logical query answering is a challenging task in knowledge graphs (KGs) that has been widely studied. The ability to perform complex logical reasoning is essential and supports various graph reasoning-based downstream tasks, such as search engines. Recent approaches are proposed to represent KG entities and logical queries into embedding vectors and find answers to logical queries from the KGs. However, existing proposed methods mainly focus on querying a single KG and cannot be applied to multiple graphs. In addition, directly sharing KGs with sensitive information may incur privacy risks, making it impractical to share and construct an aggregated KG for reasoning to retrieve query answers. Thus, it remains unknown how to answer queries on multi-source KGs. An entity can be involved in various knowledge graphs and reasoning on multiple KGs and answering complex queries on multi-source KGs is important in discovering knowledge 
    
[^8]: 对净化的对抗训练（AToP）：提升鲁棒性和泛化性能

    Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization

    [https://arxiv.org/abs/2401.16352](https://arxiv.org/abs/2401.16352)

    提出了一种新的对净化的对抗训练（AToP）流程，通过随机转换的扰动破坏和通过对抗损失微调净化器模型，同时提升了鲁棒性和泛化性能。

    

    深度神经网络被认为易受设计精良的对抗攻击影响。基于对抗训练（AT）的最成功防御技术可以实现特定攻击下的最佳鲁棒性，但无法很好地泛化到未知攻击。基于对抗净化（AP）的另一有效防御技术可以增强泛化性能，但无法实现最佳鲁棒性。与此同时，这两种方法都存在一个共同的局限性，即标准准确性降级。为了缓解这些问题，我们提出了一种新的流程，称为对净化的对抗训练（AToP），包括两个组件：通过随机转换（RT）破坏扰动，以避免对已知攻击的过度学习，从而实现对未知攻击的鲁棒性泛化；以及通过对抗损失对净化器模型进行微调（FT），以提高鲁棒性。为了评估我们的方法，我们在一种...

    arXiv:2401.16352v2 Announce Type: replace-cross  Abstract: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel pipeline called Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks resulting in the robustness generalization to unseen attacks and FT is essential for the improvement of robustness. To evaluate our method in an e
    
[^9]: 理性情感：以自我展示理论为指导的增强型同理心回应生成的LLM方法

    Rational Sensibility: LLM Enhanced Empathetic Response Generation Guided by Self-presentation Theory. (arXiv:2312.08702v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2312.08702](http://arxiv.org/abs/2312.08702)

    本文通过以自我展示理论为指导，设计了一种创新的分类方法，将历史对话分成合理和理性的句子，并通过注意力机制来阐明上下文，从而增强同理心回应生成的能力。

    

    在对话中准确表达人类行为的能力对于同理心至关重要。尽管有许多研究致力于通过引入外部知识来改进模型的认知能力，但对于对话本身的理性表达和合理的表现方面，却受到了有限的关注，而这些是认知同理心的关键组成部分。在社会学中，我们借鉴了自我展示理论，设计了一种创新的分类方法，将历史对话分成合理和理性的句子，并通过设计的注意力机制来阐明上下文。然而，对话中的理性信息受到限制，并且先前方法中使用的外部知识存在语义矛盾和狭窄视野的限制。考虑到LLM在智能代理领域的卓越表现，我们采用LLaMA2-70b作为理性大脑来分析深远的逻辑信息。

    Having the ability to empathize is crucial for accurately representing human behavior during conversations. Despite numerous research aim to improve the cognitive capability of models by incorporating external knowledge, there has been limited attention on the sensible and rational expression of the conversation itself, which are crucial components of the cognitive empathy. Guided by self-presentation theory in sociology, we have designed an innovative categorical approach that segregates historical dialogues into sensible and rational sentences and subsequently elucidate the context through the designed attention mechanism. However, the rational information within the conversation is restricted and the external knowledge used in previous methods have limitations of semantic contradiction and narrow vision field. Considering the impressive performance of LLM in the domain of intelligent agent. We employ LLaMA2-70b as a rational brain to analyze the profound logical information maintain
    
[^10]: 基于不确定性梯度匹配的模型合并

    Model Merging by Uncertainty-Based Gradient Matching. (arXiv:2310.12808v1 [cs.LG])

    [http://arxiv.org/abs/2310.12808](http://arxiv.org/abs/2310.12808)

    本论文通过不确定性梯度匹配的方法，提出了一种新的模型合并方案，该方案能够减少梯度不匹配，从而提高了模型合并的性能并对超参数更具鲁棒性。

    

    在不同数据集上训练的模型可以通过参数的加权平均来合并，但为什么会起作用，什么情况下会失败？在这里，我们将加权平均的不准确性与梯度不匹配联系起来，并提出了一种新的基于不确定性的方案，通过减少不匹配来提高性能。这种联系还揭示了其他方案（如平均值、任务算术和Fisher加权平均）中的隐含假设。我们的新方法在大型语言模型和视觉转换器方面都在性能和超参数鲁棒性方面得到了一致的改进。

    Models trained on different datasets can be merged by a weighted-averaging of their parameters, but why does it work and when can it fail? Here, we connect the inaccuracy of weighted-averaging to mismatches in the gradients and propose a new uncertainty-based scheme to improve the performance by reducing the mismatch. The connection also reveals implicit assumptions in other schemes such as averaging, task arithmetic, and Fisher-weighted averaging. Our new method gives consistent improvements for large language models and vision transformers, both in terms of performance and robustness to hyperparameters.
    
[^11]: OpsEval: 用于大型语言模型的全面任务导向的AIOps基准测试

    OpsEval: A Comprehensive Task-Oriented AIOps Benchmark for Large Language Models. (arXiv:2310.07637v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.07637](http://arxiv.org/abs/2310.07637)

    OpsEval是一个全面任务导向的AIOps基准测试，评估了大型语言模型在有线网络操作、5G通信操作和数据库操作等关键场景下的能力水平，为提供针对AIOps定制的LLMs的优化方向。

    

    大型语言模型(Large Language Models, LLMs)在翻译、总结和生成等NLP相关任务中表现出了显著的能力。LLMs在特定领域中应用，特别是在AIOps（面向IT运维的人工智能）中，由于其先进的信息汇总、报告分析和API调用能力而具有巨大的潜力。然而，当前LLMs在AIOps任务中的性能尚未确定。此外，需要一个全面的基准测试来引导针对AIOps定制的LLMs的优化。与现有的专注于评估网络配置等特定领域的基准测试不同，本文提出了OpsEval，这是一个专为LLMs设计的全面任务导向的AIOps基准测试。OpsEval首次对LLMs在三个关键场景（有线网络操作、5G通信操作和数据库操作）以及不同的能力水平（知识回忆、分析思考）进行评估。

    Large language models (LLMs) have exhibited remarkable capabilities in NLP-related tasks such as translation, summarizing, and generation. The application of LLMs in specific areas, notably AIOps (Artificial Intelligence for IT Operations), holds great potential due to their advanced abilities in information summarizing, report analyzing, and ability of API calling. Nevertheless, the performance of current LLMs in AIOps tasks is yet to be determined. Furthermore, a comprehensive benchmark is required to steer the optimization of LLMs tailored for AIOps. Compared with existing benchmarks that focus on evaluating specific fields like network configuration, in this paper, we present \textbf{OpsEval}, a comprehensive task-oriented AIOps benchmark designed for LLMs. For the first time, OpsEval assesses LLMs' proficiency in three crucial scenarios (Wired Network Operation, 5G Communication Operation, and Database Operation) at various ability levels (knowledge recall, analytical thinking, an
    
[^12]: 关于受恶意噪声影响的公正约束学习的脆弱性

    On the Vulnerability of Fairness Constrained Learning to Malicious Noise. (arXiv:2307.11892v1 [cs.LG])

    [http://arxiv.org/abs/2307.11892](http://arxiv.org/abs/2307.11892)

    这项研究考虑了公正约束学习对恶意噪声的脆弱性，发现使用随机分类器可以在精度上只损失$\Theta(\alpha)$和$O(\sqrt{\alpha})$，对应不同的公正约束要求。

    

    我们考虑了公正约束学习对训练数据中微小恶意噪声的脆弱性。Konstantinov和Lampert (2021)在这个问题上进行了研究，并展示了负面结果，表明在不平衡的群组大小下存在一些数据分布，任何适当的学习器都会表现出较高的脆弱性。在这里，我们展示了更乐观的观点，如果允许随机分类器，则情况更加细致。例如，对于人口统计学平等性，我们显示只会产生$\Theta(\alpha)$的精度损失，其中$\alpha$是恶意噪声率，甚至可以与没有公正约束的情况完全匹配。对于机会均等性，我们显示只会产生$O(\sqrt{\alpha})$的损失，并给出一个匹配的$\Omega(\sqrt{\alpha})$的下界。相比之下，Konstantinov和Lampert (2021)示范了对于适当的学习器，这两个概念的精度损失都是$\Omega(1)$。关键的技术创新是

    We consider the vulnerability of fairness-constrained learning to small amounts of malicious noise in the training data. Konstantinov and Lampert (2021) initiated the study of this question and presented negative results showing there exist data distributions where for several fairness constraints, any proper learner will exhibit high vulnerability when group sizes are imbalanced. Here, we present a more optimistic view, showing that if we allow randomized classifiers, then the landscape is much more nuanced. For example, for Demographic Parity we show we can incur only a $\Theta(\alpha)$ loss in accuracy, where $\alpha$ is the malicious noise rate, matching the best possible even without fairness constraints. For Equal Opportunity, we show we can incur an $O(\sqrt{\alpha})$ loss, and give a matching $\Omega(\sqrt{\alpha})$lower bound. In contrast, Konstantinov and Lampert (2021) showed for proper learners the loss in accuracy for both notions is $\Omega(1)$. The key technical novelty 
    
[^13]: DiffLoad:扩散模型中的负荷预测不确定性量化

    DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model. (arXiv:2306.01001v1 [cs.LG])

    [http://arxiv.org/abs/2306.01001](http://arxiv.org/abs/2306.01001)

    本文提出了一种扩散模型中的负荷预测不确定性量化方法，采用Seq2Seq网络结构来分离两种类型的不确定性并处理异常情况，不仅着眼于预测条件期望值。

    

    电力负荷预测对电力系统的决策制定，如机组投入和能源管理等具有重要意义。近年来，各种基于自监督神经网络的方法已经被应用于电力负荷预测，以提高预测准确性和捕捉不确定性。然而，大多数现有的方法是基于高斯似然方法的，它旨在在给定的协变量下准确估计分布期望值。这种方法很难适应存在分布偏移和异常值的时间数据。在本文中，我们提出了一种基于扩散的Seq2seq结构来估计本体不确定性，并使用鲁棒的加性柯西分布来估计物象不确定性。我们展示了我们的方法能够分离两种类型的不确定性并处理突变情况，而不是准确预测条件期望。

    Electrical load forecasting is of great significance for the decision makings in power systems, such as unit commitment and energy management. In recent years, various self-supervised neural network-based methods have been applied to electrical load forecasting to improve forecasting accuracy and capture uncertainties. However, most current methods are based on Gaussian likelihood methods, which aim to accurately estimate the distribution expectation under a given covariate. This kind of approach is difficult to adapt to situations where temporal data has a distribution shift and outliers. In this paper, we propose a diffusion-based Seq2seq structure to estimate epistemic uncertainty and use the robust additive Cauchy distribution to estimate aleatoric uncertainty. Rather than accurately forecasting conditional expectations, we demonstrate our method's ability in separating two types of uncertainties and dealing with the mutant scenarios.
    
[^14]: 关于AI辅助决策中依赖行为与准确性的相互关系

    On the Interdependence of Reliance Behavior and Accuracy in AI-Assisted Decision-Making. (arXiv:2304.08804v1 [cs.HC])

    [http://arxiv.org/abs/2304.08804](http://arxiv.org/abs/2304.08804)

    该论文分析了AI辅助决策中依赖行为和准确性之间的相互关系，并提出了一个视觉框架来更好地理解这种关系。该框架揭示了当人类在决策中过度依赖AI时，改善信任可能会降低准确性的有趣属性。

    

    在AI辅助决策中，将人类置于决策环路中央的主要承诺是，他们应该能够通过符合其正确的和覆盖其错误的建议来补充AI系统。然而实践中，我们经常看到人类倾向于过度或不足地依赖AI建议，这意味着他们要么依从错误的建议，要么覆盖正确的建议。这种依赖行为对决策准确性有害。在这项工作中，我们阐述并分析了在AI辅助决策中依赖行为和准确性之间的相互关系，这在以前的工作中很大程度上被忽视了。我们还提出了一个视觉框架，使这种相互关系更加具体化。该框架帮助我们解释和比较实证研究结果，并获得对AI辅助决策干预（例如解释）影响的细致理解。最后，我们从框架中推出了几个有趣的属性：（i）当人类不足地依赖AI建议时，改善信任将显着提高准确性，但在他们过度依赖时，信任的改善却可能降低准确性。

    In AI-assisted decision-making, a central promise of putting a human in the loop is that they should be able to complement the AI system by adhering to its correct and overriding its mistaken recommendations. In practice, however, we often see that humans tend to over- or under-rely on AI recommendations, meaning that they either adhere to wrong or override correct recommendations. Such reliance behavior is detrimental to decision-making accuracy. In this work, we articulate and analyze the interdependence between reliance behavior and accuracy in AI-assisted decision-making, which has been largely neglected in prior work. We also propose a visual framework to make this interdependence more tangible. This framework helps us interpret and compare empirical findings, as well as obtain a nuanced understanding of the effects of interventions (e.g., explanations) in AI-assisted decision-making. Finally, we infer several interesting properties from the framework: (i) when humans under-rely o
    

