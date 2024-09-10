# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring the Mystery of Influential Data for Mathematical Reasoning](https://arxiv.org/abs/2404.01067) | 提出了适用于数学推理的质量感知多样选择（QaDS）策略，并验证其在选择具有影响力的数据上的优越性；扩大了数据规模、用QaDS选择的通用数据进行训练对数学推理有帮助，最后定义了最佳混合OpenMathMix。 |
| [^2] | [Disentangling Length from Quality in Direct Preference Optimization](https://arxiv.org/abs/2403.19159) | 针对直接偏好优化中的长度问题展开研究，揭示了DPO中显著的利用情况，并将其与分布外引导联系起来。 |
| [^3] | [Fact-and-Reflection (FaR) Improves Confidence Calibration of Large Language Models](https://arxiv.org/abs/2402.17124) | 提出了Fact-and-Reflection（FaR）提示策略，通过引入已知“事实”并要求模型“反思”，在两个步骤中改进了大型语言模型的置信校准 |
| [^4] | [CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark](https://arxiv.org/abs/2401.11944) | CMMMU是一个旨在评估大型多模型模型在大学级学科知识和深思熟虑推理任务中表现的中文大规模多学科多模态理解基准，为填补在非英语环境中评估先进知识和推理能力的空白而设计。 |
| [^5] | [Rethinking the BERT-like Pretraining for DNA Sequences.](http://arxiv.org/abs/2310.07644) | 重新考虑了基于DNA序列的BERT-like预训练方法，通过使用K-mer重叠标记化，在下游任务的微调阶段和预训练过程中都取得了一致的性能改善。 |
| [^6] | [PointLLM: Empowering Large Language Models to Understand Point Clouds.](http://arxiv.org/abs/2308.16911) | PointLLM是一种使大型语言模型理解点云的方法，它利用点云编码器和强大的LLM将几何、外观和语言信息融合，并通过人类指导生成环境上恰当的响应。该方法通过收集大规模的点-文本指令对数据集进行两阶段的训练，以提高模型的感知能力和泛化能力。 |
| [^7] | [Synthetic Dataset for Evaluating Complex Compositional Knowledge for Natural Language Inference.](http://arxiv.org/abs/2307.05034) | 该论文介绍了一个名为SICCK的合成数据集以及一种新颖的分析方法，用于评估自然语言推理中复杂组合知识的性能。研究发现，在零-shot和微调情况下，神经网络推理模型能够很好地捕捉结构和语义组合的变化。 |
| [^8] | [MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.](http://arxiv.org/abs/2305.14795) | 本文提出了一种基准测试MQuAKE，通过多跳问题评估编辑模型是否能够正确回答因编辑事实而答案应该改变的问题。研究发现当前的知识编辑方法可以准确召回已编辑的事实，但在多跳问题上表现灾难性失败。 |
| [^9] | [Using Natural Language Explanations to Rescale Human Judgments.](http://arxiv.org/abs/2305.14770) | 本文提出使用自然语言解释来调整标注者之间存在的尺度不一致，解决了主观NLP任务中标注者之间分歧的问题。 |

# 详细

[^1]: 探索数学推理中数据对推理的影响之谜

    Exploring the Mystery of Influential Data for Mathematical Reasoning

    [https://arxiv.org/abs/2404.01067](https://arxiv.org/abs/2404.01067)

    提出了适用于数学推理的质量感知多样选择（QaDS）策略，并验证其在选择具有影响力的数据上的优越性；扩大了数据规模、用QaDS选择的通用数据进行训练对数学推理有帮助，最后定义了最佳混合OpenMathMix。

    

    选择对下游任务微调具有影响力的数据是性能和计算效率的关键因素。最近的研究表明，仅使用有限数据进行训练在通用任务上可以表现出卓越性能。然而，对于数学推理任务，这种可行性尚未得到验证。为此，针对数学推理存在两个开放问题：如何选择具有影响力的数据以及什么是具有影响力的数据组成。对于前者，我们提出了一个适用于数学推理的质量感知多样选择（QaDS）策略。与其他选择策略进行比较验证了QaDS的优越性。对于后者，我们首先扩大了我们的设置并探索了具有影响力的数据组成。我们进行了一系列实验并强调：扩大推理数据规模，并训练选用QaDS选择的通用数据是有益的。然后，我们将我们的最佳混合定义为OpenMathMix。

    arXiv:2404.01067v1 Announce Type: new  Abstract: Selecting influential data for fine-tuning on downstream tasks is a key factor for both performance and computation efficiency. Recent works have shown that training with only limited data can show a superior performance on general tasks. However, the feasibility on mathematical reasoning tasks has not been validated. To go further, there exist two open questions for mathematical reasoning: how to select influential data and what is an influential data composition. For the former one, we propose a Quality-aware Diverse Selection (QaDS) strategy adaptable for mathematical reasoning. A comparison with other selection strategies validates the superiority of QaDS. For the latter one, we first enlarge our setting and explore the influential data composition. We conduct a series of experiments and highlight: scaling up reasoning data, and training with general data selected by QaDS is helpful. Then, we define our optimal mixture as OpenMathMix
    
[^2]: 在直接偏好优化中将长度与质量分离

    Disentangling Length from Quality in Direct Preference Optimization

    [https://arxiv.org/abs/2403.19159](https://arxiv.org/abs/2403.19159)

    针对直接偏好优化中的长度问题展开研究，揭示了DPO中显著的利用情况，并将其与分布外引导联系起来。

    

    Reinforcement Learning from Human Feedback (RLHF)是最近大型语言模型成功的关键组成部分。然而，RLHF被认为利用了人类偏好中的偏见，比如冗长性。精心格式化和雄辩的答案通常会被用户更高评价，即使它们在帮助性和客观性上较低。一些方法已经被开发来控制这些偏见，在古典RLHF文献中这个问题已有所探讨，但对于直接对齐算法如直接偏好优化（DPO）这个问题相对较少探索。与古典RLHF不同，DPO不训练单独的奖励模型或直接使用强化学习，因此之前用来控制冗长性的方法无法直接应用于这种情况。我们的工作做出了几点贡献。首次在DPO环境中研究长度问题，显示DPO中存在显著的利用，并将其与分布外引导相关联。

    arXiv:2403.19159v1 Announce Type: new  Abstract: Reinforcement Learning from Human Feedback (RLHF) has been a crucial component in the recent success of Large Language Models. However, RLHF is know to exploit biases in human preferences, such as verbosity. A well-formatted and eloquent answer is often more highly rated by users, even when it is less helpful and objective. A number of approaches have been developed to control those biases in the classical RLHF literature, but the problem remains relatively under-explored for Direct Alignment Algorithms such as Direct Preference Optimization (DPO). Unlike classical RLHF, DPO does not train a separate reward model or use reinforcement learning directly, so previous approaches developed to control verbosity cannot be directly applied to this setting. Our work makes several contributions. For the first time, we study the length problem in the DPO setting, showing significant exploitation in DPO and linking it to out-of-distribution bootstra
    
[^3]: Fact-and-Reflection（FaR）改善大型语言模型的置信校准

    Fact-and-Reflection (FaR) Improves Confidence Calibration of Large Language Models

    [https://arxiv.org/abs/2402.17124](https://arxiv.org/abs/2402.17124)

    提出了Fact-and-Reflection（FaR）提示策略，通过引入已知“事实”并要求模型“反思”，在两个步骤中改进了大型语言模型的置信校准

    

    要使LLM值得信赖，其置信水平应与实际表现良好校准。尽管现在普遍认为LLM的表现在很大程度上受到提示的影响，但提示LLM中的置信校准尚未得到彻底探讨。本文探讨了不同提示策略如何影响LLM的置信校准以及如何改进。我们在问答环境中对六种提示方法进行了大量实验，我们观察到，尽管这些方法有助于改进LLM的预期校准，但也会导致LLM在响应某些实例时过于自信。受人类认知启发，我们提出了Fact-and-Reflection（FaR）提示，它通过两个步骤改善了LLM的校准。首先，FaR从LLM中获取与输入提示相关的已知“事实”。然后要求模型“反思”它们以生成最终答案。

    arXiv:2402.17124v1 Announce Type: new  Abstract: For a LLM to be trustworthy, its confidence level should be well-calibrated with its actual performance. While it is now common sense that LLM performances are greatly impacted by prompts, the confidence calibration in prompting LLMs has yet to be thoroughly explored. In this paper, we explore how different prompting strategies influence LLM confidence calibration and how it could be improved. We conduct extensive experiments on six prompting methods in the question-answering context and we observe that, while these methods help improve the expected LLM calibration, they also trigger LLMs to be over-confident when responding to some instances. Inspired by human cognition, we propose Fact-and-Reflection (FaR) prompting, which improves the LLM calibration in two steps. First, FaR elicits the known "facts" that are relevant to the input prompt from the LLM. And then it asks the model to "reflect" over them to generate the final answer. Expe
    
[^4]: CMMMU：一个中国大规模多学科多模态理解基准

    CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark

    [https://arxiv.org/abs/2401.11944](https://arxiv.org/abs/2401.11944)

    CMMMU是一个旨在评估大型多模型模型在大学级学科知识和深思熟虑推理任务中表现的中文大规模多学科多模态理解基准，为填补在非英语环境中评估先进知识和推理能力的空白而设计。

    

    随着大型多模型模型(LMMs)的能力不断提升，评估LMMs的表现日益成为一个迫切的需求。此外，在评估LMMs在中文等非英语环境中先进知识和推理能力方面存在更大差距。我们引入了CMMMU，一个新的中文大规模多学科多模态理解基准，旨在评估LMMs在需要大学水平学科知识和深思熟虑推理的任务中的表现。CMMMU受到了MMMUs的标注和分析模式的启发并严格遵循。CMMMU包括来自大学考试、测验和教科书的1.2万个手动收集的多模态问题，涵盖六个核心学科：艺术与设计、商业、科学、健康与医学、人文社科以及技术与工程，就像其伙伴MMMMU一样。这些问题涵盖30个学科，包括39个高度异质的图像。

    arXiv:2401.11944v2 Announce Type: replace-cross  Abstract: As the capabilities of large multimodal models (LMMs) continue to advance, evaluating the performance of LMMs emerges as an increasing need. Additionally, there is an even larger gap in evaluating the advanced knowledge and reasoning abilities of LMMs in non-English contexts such as Chinese. We introduce CMMMU, a new Chinese Massive Multi-discipline Multimodal Understanding benchmark designed to evaluate LMMs on tasks demanding college-level subject knowledge and deliberate reasoning in a Chinese context. CMMMU is inspired by and strictly follows the annotation and analysis pattern of MMMU.   CMMMU includes 12k manually collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering, like its companion, MMMU. These questions span 30 subjects and comprise 39 highly heterogeneous image 
    
[^5]: 重新考虑基于DNA序列的BERT-like预训练

    Rethinking the BERT-like Pretraining for DNA Sequences. (arXiv:2310.07644v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.07644](http://arxiv.org/abs/2310.07644)

    重新考虑了基于DNA序列的BERT-like预训练方法，通过使用K-mer重叠标记化，在下游任务的微调阶段和预训练过程中都取得了一致的性能改善。

    

    随着在自然语言处理领域中大规模预训练的成功，将其应用于生命科学领域的趋势日益增长。特别是基于DNA序列的预训练方法因其捕捉基因的通用信息的潜力而受到关注。然而，现有的DNA序列预训练方法主要依赖于从自然语言处理领域直接引入的BERT预训练方法，缺乏全面的理解和专门定制的方法。为了填补这一研究空白，我们首先进行了一系列的探索性实验，并获得了几个有启发性的观察结果：1）在下游任务的微调阶段，使用K-mer重叠标记化而不是K-mer非重叠标记化时，重叠和非重叠的预训练权重均表现出一致的性能改善。2）在预训练过程中，使用K-mer重叠标记化会迅速产生清晰的K-mer嵌入，并将损失降低到非常低的水平。

    With the success of large-scale pretraining in NLP, there is an increasing trend of applying it to the domain of life sciences. In particular, pretraining methods based on DNA sequences have garnered growing attention due to their potential to capture generic information about genes. However, existing pretraining methods for DNA sequences largely rely on direct adoptions of BERT pretraining from NLP, lacking a comprehensive understanding and a specifically tailored approach. To address this research gap, we first conducted a series of exploratory experiments and gained several insightful observations: 1) In the fine-tuning phase of downstream tasks, when using K-mer overlapping tokenization instead of K-mer non-overlapping tokenization, both overlapping and non-overlapping pretraining weights show consistent performance improvement.2) During the pre-training process, using K-mer overlapping tokenization quickly produces clear K-mer embeddings and reduces the loss to a very low level, w
    
[^6]: PointLLM：赋予大型语言模型理解点云的能力

    PointLLM: Empowering Large Language Models to Understand Point Clouds. (arXiv:2308.16911v1 [cs.CV])

    [http://arxiv.org/abs/2308.16911](http://arxiv.org/abs/2308.16911)

    PointLLM是一种使大型语言模型理解点云的方法，它利用点云编码器和强大的LLM将几何、外观和语言信息融合，并通过人类指导生成环境上恰当的响应。该方法通过收集大规模的点-文本指令对数据集进行两阶段的训练，以提高模型的感知能力和泛化能力。

    

    大型语言模型（LLM）的前所未有的进展对自然语言处理产生了深远影响，但在3D理解领域仍有待完全发展。本文介绍了PointLLM，这是一项填补这一空白的初步工作，使LLM能够理解点云，并提供了超越2D视觉数据的新途径。PointLLM通过人类指导处理带有颜色的物体点云，并生成环境上恰当的响应，展示了其对点云和常识的掌握。具体来说，它利用了一个点云编码器和一个强大的LLM，有效地融合了几何、外观和语言信息。我们收集了一个新颖的数据集，包括66万个简单和7万个复杂的点-文本指令对，以实现两阶段的训练策略：首先对齐潜在空间，然后对统一模型进行指令调整。为了严格评估我们模型的感知能力和其泛化能力，我们建立了评估基准数据集进行实验。

    The unprecedented advancements in Large Language Models (LLMs) have created a profound impact on natural language processing but are yet to fully embrace the realm of 3D understanding. This paper introduces PointLLM, a preliminary effort to fill this gap, thereby enabling LLMs to understand point clouds and offering a new avenue beyond 2D visual data. PointLLM processes colored object point clouds with human instructions and generates contextually appropriate responses, illustrating its grasp of point clouds and common sense. Specifically, it leverages a point cloud encoder with a powerful LLM to effectively fuse geometric, appearance, and linguistic information. We collect a novel dataset comprising 660K simple and 70K complex point-text instruction pairs to enable a two-stage training strategy: initially aligning latent spaces and subsequently instruction-tuning the unified model. To rigorously evaluate our model's perceptual abilities and its generalization capabilities, we establis
    
[^7]: 用于评估自然语言推理中复杂组合知识的合成数据集

    Synthetic Dataset for Evaluating Complex Compositional Knowledge for Natural Language Inference. (arXiv:2307.05034v1 [cs.CL])

    [http://arxiv.org/abs/2307.05034](http://arxiv.org/abs/2307.05034)

    该论文介绍了一个名为SICCK的合成数据集以及一种新颖的分析方法，用于评估自然语言推理中复杂组合知识的性能。研究发现，在零-shot和微调情况下，神经网络推理模型能够很好地捕捉结构和语义组合的变化。

    

    我们介绍了一个名为Sentences Involving Complex Compositional Knowledge (SICCK)的合成数据集，以及一种新颖的分析方法，用于研究自然语言推理模型对逻辑组成性的性能。我们通过修改SICK数据集中的15个示例，生成了1,304个句子对。为此，我们使用一组短语 - 与自然逻辑中的普遍量词、存在量词、否定和其他概念修饰符相对应的修饰符 - 修改了原始文本。我们使用这些短语修改前提和假设的主语、谓语和宾语部分。最后，我们根据自然逻辑规则为这些修改后的文本标注相应的包含关系标签。我们对神经网络推理模型在零-shot和微调情况下对结构和语义组合变化的捕捉能力进行了初步验证。我们发现在这些情况下，NLI模型的性能表现良好。

    We introduce a synthetic dataset called Sentences Involving Complex Compositional Knowledge (SICCK) and a novel analysis that investigates the performance of Natural Language Inference (NLI) models to understand compositionality in logic. We produce 1,304 sentence pairs by modifying 15 examples from the SICK dataset (Marelli et al., 2014). To this end, we modify the original texts using a set of phrases - modifiers that correspond to universal quantifiers, existential quantifiers, negation, and other concept modifiers in Natural Logic (NL) (MacCartney, 2009). We use these phrases to modify the subject, verb, and object parts of the premise and hypothesis. Lastly, we annotate these modified texts with the corresponding entailment labels following NL rules. We conduct a preliminary verification of how well the change in the structural and semantic composition is captured by neural NLI models, in both zero-shot and fine-tuned scenarios. We found that the performance of NLI models under th
    
[^8]: MQuAKE：通过多跳问题评估语言模型中的知识编辑

    MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions. (arXiv:2305.14795v1 [cs.CL])

    [http://arxiv.org/abs/2305.14795](http://arxiv.org/abs/2305.14795)

    本文提出了一种基准测试MQuAKE，通过多跳问题评估编辑模型是否能够正确回答因编辑事实而答案应该改变的问题。研究发现当前的知识编辑方法可以准确召回已编辑的事实，但在多跳问题上表现灾难性失败。

    

    大型语言模型（LLM）中存储的信息很快就会过时，重新训练并非总是可行的选择。这促使人们开发了通过更新模型权重注入新事实的一系列技术。当前的评估方法非常有限，主要验证编辑事实的召回率，但更改一个事实应该会对模型的相关信念产生连锁反应。如果我们编辑英国首相为Rishi Sunak，那么对于“谁是英国首相的配偶”这个问题，我们应该得到一个不同的答案。在这项工作中，我们提出了一个基准MQuAKE（用于知识编辑的多跳问答），包括多跳问题，评估编辑后的模型是否正确回答那些因编辑事实而答案应该改变的问题。虽然我们发现当前的知识编辑方法可以准确召回已编辑的事实，但它们在构建的多跳问题上遭遇了灾难性失败。因此，我们建议对LLMs的评估必须超越简单的事实召回，并纳入更微妙的知识编辑质量评估。

    The information stored in large language models (LLMs) falls out of date quickly, and retraining from scratch is often not an option. This has recently given rise to a range of techniques for injecting new facts through updating model weights. Current evaluation paradigms are extremely limited, mainly validating the recall of edited facts, but changing one fact should cause rippling changes to the model's related beliefs. If we edit the UK Prime Minister to now be Rishi Sunak, then we should get a different answer to Who is married to the British Prime Minister? In this work, we present a benchmark MQuAKE (Multi-hop Question Answering for Knowledge Editing) comprising multi-hop questions that assess whether edited models correctly answer questions where the answer should change as an entailed consequence of edited facts. While we find that current knowledge-editing approaches can recall edited facts accurately, they fail catastrophically on the constructed multi-hop questions. We thus 
    
[^9]: 利用自然语言解释重新调整人类评价

    Using Natural Language Explanations to Rescale Human Judgments. (arXiv:2305.14770v1 [cs.CL])

    [http://arxiv.org/abs/2305.14770](http://arxiv.org/abs/2305.14770)

    本文提出使用自然语言解释来调整标注者之间存在的尺度不一致，解决了主观NLP任务中标注者之间分歧的问题。

    

    大型语言模型（LLM）的出现带来了需要高质量人标记数据的紧迫需求，特别是对于人的反馈和评估等过程。一种常见的做法是通过多个众包工作者的共识来标注数据。然而，不同的标注者可能对标注方案有不同的解释，除非接受了广泛的培训，否则对于主观的NLP任务，甚至受过训练的专家标注者也可能会出现巨大的分歧。我们展示了这些细微差别可以通过高质量的自然语言解释进行捕捉，提出了一种使用LLM在存在分歧时重新调整大小排序注释的方法。具体而言，我们将Likert评分和相应的自然语言解释输入LLM，并提示它产生一个数字得分。这个得分应该反映注释者对示例的基本评估。解释的存在使LLM能够在尺度使用差异存在的情况下使评级在标注者之间同质化。

    The rise of large language models (LLMs) has brought a critical need for high-quality human-labeled data, particularly for processes like human feedback and evaluation. A common practice is to label data via consensus annotation over the judgments of multiple crowdworkers. However, different annotators may have different interpretations of labeling schemes unless given extensive training, and for subjective NLP tasks, even trained expert annotators can diverge heavily. We show that these nuances can be captured by high quality natural language explanations, and propose a method to rescale ordinal annotation in the presence of disagreement using LLMs. Specifically, we feed Likert ratings and corresponding natural language explanations into an LLM and prompt it to produce a numeric score. This score should reflect the underlying assessment of the example by the annotator. The presence of explanations allows the LLM to homogenize ratings across annotators in spite of scale usage differenc
    

