# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Meaning Representations from Trajectories in Autoregressive Models.](http://arxiv.org/abs/2310.18348) | 本文提出了一种从自回归语言模型中提取意义表征的方法，通过考虑输入文本的所有可能轨迹的分布。这种方法可以模拟非对称关系，且在语义相似性任务上优于其他方法。 |
| [^2] | [SentMix-3L: A Bangla-English-Hindi Code-Mixed Dataset for Sentiment Analysis.](http://arxiv.org/abs/2310.18023) | SentMix-3L是一个用于情感分析的新颖数据集，包含孟加拉语、英语和印地语之间的代码混合数据。研究发现，在SentMix-3L上，使用GPT-3.5进行零-shot提示可以超过所有基于转换器的模型。 |
| [^3] | [An Attribution Method for Siamese Encoders.](http://arxiv.org/abs/2310.05703) | 本文提出了一种适用于Siamese编码器的局部归因方法，通过将集成梯度原理推广到具有多个输入的模型，该方法能够解释句子转换器模型中重要的预测令牌对，主要集中在名词和动词上。 |
| [^4] | [Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback.](http://arxiv.org/abs/2310.05199) | 本文提出了一种创新的解决方案，通过应用“专家的乘积”（PoE）技术来减轻强化学习中的长度偏差问题。在这个框架中，主要的专家关注理解人类意图，而偏见专家则致力于识别和捕捉长度偏差。 |
| [^5] | [Navigating Cultural Chasms: Exploring and Unlocking the Cultural POV of Text-To-Image Models.](http://arxiv.org/abs/2310.01929) | 本研究旨在探索和解锁文本到图像模型的文化视角，通过对TTI模型中嵌入的文化感知进行评估，揭示了这些模型的文化意识、文化区别和文化适应性。 |
| [^6] | [DeepSpeed-VisualChat: Multi-Round Multi-Image Interleave Chat via Multi-Modal Causal Attention.](http://arxiv.org/abs/2309.14327) | DeepSpeed-VisualChat是一个用于多轮多图交错聊天的框架，通过引入创新的多模态因果关注机制和数据融合技术，具有优越的可扩展性。 |
| [^7] | [On Separate Normalization in Self-supervised Transformers.](http://arxiv.org/abs/2309.12931) | 在自监督变形器中，通过为标记和[CLS]符号分别使用归一化层，可以更好地捕捉它们各自的特点并提高下游任务的性能。 |
| [^8] | [Explainability for Large Language Models: A Survey.](http://arxiv.org/abs/2309.01029) | 本文调研了大型语言模型的可解释性问题，提出了一个解释技术的分类法，并介绍了基于Transformer的语言模型的解释方法。同时，讨论了评估生成解释的度量标准，以及如何利用解释来调试模型和提高性能。 |
| [^9] | [Towards Understanding In-Context Learning with Contrastive Demonstrations and Saliency Maps.](http://arxiv.org/abs/2307.05052) | 本研究探索了对比演示和显著性图在上下文学习中的作用，并发现改变标签对显著性有显著影响，尤其对于更大的语言模型更为明显。在情感分析任务中，将表达情感的术语改为中性词并不像改变标签那样具有显著影响。另外，补充解释在提高上下文学习方面是有效的。 |
| [^10] | [Adapting Sentence Transformers for the Aviation Domain.](http://arxiv.org/abs/2305.09556) | 本研究提出了一种针对航空领域的句子变换器调整方法，在预训练阶段使用TSDAE模型进行改进，然后在少量注释的数据集上进行微调，实验结果表明在航空相关的自然语言处理任务中取得了最好的表现。 |
| [^11] | [SUR-adapter: Enhancing Text-to-Image Pre-trained Diffusion Models with Large Language Models.](http://arxiv.org/abs/2305.05189) | 本文提出了一个名为SUR-adapter的微调方法，用于增强预先训练的文本到图像扩散模型的语义理解和常识推理能力，以便在生成图片时使用简短的叙述提示。作者还构建了一个新的数据集SURD，并使用大型语言模型的知识进行了优化。 |
| [^12] | [Exploring Human-Like Translation Strategy with Large Language Models.](http://arxiv.org/abs/2305.04118) | 本文提出了一个名为MAPS的框架，使LLMs能够模仿人类翻译的过程，该过程包括分析源文本并提取关键词、主题和相关演示以指导翻译过程。该框架实验结果显示明显优于多个强基线，为开展使用LLM实现人类化翻译策略的有前途的方向提供了启示。 |

# 详细

[^1]: 意义表征来自自回归模型中的轨迹

    Meaning Representations from Trajectories in Autoregressive Models. (arXiv:2310.18348v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.18348](http://arxiv.org/abs/2310.18348)

    本文提出了一种从自回归语言模型中提取意义表征的方法，通过考虑输入文本的所有可能轨迹的分布。这种方法可以模拟非对称关系，且在语义相似性任务上优于其他方法。

    

    我们提出通过考虑扩展输入文本的所有可能轨迹的分布来从自回归语言模型中提取意义表征。这种策略是无提示的，不需要微调，并适用于任何预训练的自回归模型。此外，与基于向量的表征不同，基于分布的表征还可以通过使用似然函数之间的代数运算来建模非对称关系（例如，逻辑蕴涵的方向，上位词/下位词关系）。这些想法基于语义的分布视角，并与自动机理论中的标准构造相连接，但据我们所知，它们尚未应用于现代语言模型。我们通过实验证明，从大型模型获得的表征与人类注释很好地一致，在语义相似性任务上优于其他零样本和无提示方法，并可用于解决更复杂的蕴涵和包含任务。

    We propose to extract meaning representations from autoregressive language models by considering the distribution of all possible trajectories extending an input text. This strategy is prompt-free, does not require fine-tuning, and is applicable to any pre-trained autoregressive model. Moreover, unlike vector-based representations, distribution-based representations can also model asymmetric relations (e.g., direction of logical entailment, hypernym/hyponym relations) by using algebraic operations between likelihood functions. These ideas are grounded in distributional perspectives on semantics and are connected to standard constructions in automata theory, but to our knowledge they have not been applied to modern language models. We empirically show that the representations obtained from large models align well with human annotations, outperform other zero-shot and prompt-free methods on semantic similarity tasks, and can be used to solve more complex entailment and containment tasks 
    
[^2]: SentMix-3L: 用于情感分析的孟加拉语-英语-印地语混合代码数据集

    SentMix-3L: A Bangla-English-Hindi Code-Mixed Dataset for Sentiment Analysis. (arXiv:2310.18023v1 [cs.CL])

    [http://arxiv.org/abs/2310.18023](http://arxiv.org/abs/2310.18023)

    SentMix-3L是一个用于情感分析的新颖数据集，包含孟加拉语、英语和印地语之间的代码混合数据。研究发现，在SentMix-3L上，使用GPT-3.5进行零-shot提示可以超过所有基于转换器的模型。

    

    代码混合是一种研究很深的语言现象，指的是在文本或语音中混合使用两种或更多语言。已经构建了几个旨在训练代码混合计算模型的数据集。尽管多语言的代码混合很常见，但大多数可用的数据集只包含两种语言的代码混合。本文介绍了SentMix-3L，这是一个新颖的用于情感分析的数据集，其中包含孟加拉语、英语和印地语之间的代码混合数据。我们使用SentMix-3L进行了全面评估。我们展示了使用GPT-3.5进行零-shot提示在SentMix-3L上优于所有基于转换器的模型。

    Code-mixing is a well-studied linguistic phenomenon when two or more languages are mixed in text or speech. Several datasets have been build with the goal of training computational models for code-mixing. Although it is very common to observe code-mixing with multiple languages, most datasets available contain code-mixed between only two languages. In this paper, we introduce SentMix-3L, a novel dataset for sentiment analysis containing code-mixed data between three languages Bangla, English, and Hindi. We carry out a comprehensive evaluation using SentMix-3L. We show that zero-shot prompting with GPT-3.5 outperforms all transformer-based models on SentMix-3L.
    
[^3]: Siamese编码器的归因方法

    An Attribution Method for Siamese Encoders. (arXiv:2310.05703v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05703](http://arxiv.org/abs/2310.05703)

    本文提出了一种适用于Siamese编码器的局部归因方法，通过将集成梯度原理推广到具有多个输入的模型，该方法能够解释句子转换器模型中重要的预测令牌对，主要集中在名词和动词上。

    

    尽管句子转换器等Siamese编码器模型取得了成功，但人们对它们关注的输入方面知之甚少。一个障碍是它们的预测不能归因于个别特征，因为它们比较的是两个输入而不是一个输入。本文通过将集成梯度原理推广到具有多个输入的模型，推导出一种适用于Siamese编码器的局部归因方法。该解决方案采用特征对归因的形式，并可将其简化为句子转换器的令牌-令牌矩阵。我们的方法涉及引入集成雅可比矩阵，并继承了集成梯度的优势形式特性：它考虑了模型的完整计算图，并确保收敛到实际预测结果。一项实验表明，在句子转换器中，很少的令牌对往往可以解释大部分的预测，并且它们主要集中在名词和动词上。然而，为了获得准确的预测，它需要关注大多数的令牌。

    Despite the success of Siamese encoder models such as sentence transformers (ST), little is known about the aspects of inputs they pay attention to. A barrier is that their predictions cannot be attributed to individual features, as they compare two inputs rather than processing a single one. This paper derives a local attribution method for Siamese encoders by generalizing the principle of integrated gradients to models with multiple inputs. The solution takes the form of feature-pair attributions, and can be reduced to a token-token matrix for STs. Our method involves the introduction of integrated Jacobians and inherits the advantageous formal properties of integrated gradients: it accounts for the model's full computation graph and is guaranteed to converge to the actual prediction. A pilot study shows that in an ST few token-pairs can often explain large fractions of predictions, and it focuses on nouns and verbs. For accurate predictions, it however needs to attend to the majorit
    
[^4]: 宽松的嘴唇会使船沉没：减轻强化学习中的长度偏差问题

    Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback. (arXiv:2310.05199v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05199](http://arxiv.org/abs/2310.05199)

    本文提出了一种创新的解决方案，通过应用“专家的乘积”（PoE）技术来减轻强化学习中的长度偏差问题。在这个框架中，主要的专家关注理解人类意图，而偏见专家则致力于识别和捕捉长度偏差。

    

    人类反馈强化学习是重要的桥梁，将大型语言模型与人类和社会价值观对齐。这种对齐需要大量的人类反馈语料库来学习奖励模型，然后用于微调语言模型。然而，我们发现奖励模型常常会找到绕过预期目标的捷径，错误地假设人类更喜欢较长的回答。长度偏差的出现常常会导致模型倾向于较长的输出，但并不意味着这些输出中有更多有用的信息。在本文中，我们提出了一种创新的解决方案，应用了“专家的乘积”（PoE）技术来将奖励建模与序列长度的影响分离。在我们的框架中，主要的专家关注理解人类意图，而偏见专家则致力于识别和捕捉长度偏差。为了进一步增强偏见的学习，我们引入了扰动进入偏差部分。

    Reinforcement learning from human feedback serves as a crucial bridge, aligning large language models with human and societal values. This alignment requires a vast corpus of human feedback to learn a reward model, which is subsequently used to finetune language models. However, we have identified that the reward model often finds shortcuts to bypass its intended objectives, misleadingly assuming that humans prefer longer responses. The emergence of length bias often induces the model to favor longer outputs, yet it doesn't equate to an increase in helpful information within these outputs. In this paper, we propose an innovative solution, applying the Product-of-Experts (PoE) technique to separate reward modeling from the influence of sequence length. In our framework, the main expert concentrates on understanding human intents, while the biased expert targets the identification and capture of length bias. To further enhance the learning of bias, we introduce perturbations into the bia
    
[^5]: 穿越文化鸿沟：探索和解锁文本到图像模型的文化视角

    Navigating Cultural Chasms: Exploring and Unlocking the Cultural POV of Text-To-Image Models. (arXiv:2310.01929v1 [cs.CL])

    [http://arxiv.org/abs/2310.01929](http://arxiv.org/abs/2310.01929)

    本研究旨在探索和解锁文本到图像模型的文化视角，通过对TTI模型中嵌入的文化感知进行评估，揭示了这些模型的文化意识、文化区别和文化适应性。

    

    文本到图像（TTI）模型，例如DALL-E和StableDiffusion，在通过文本提示生成图像的零射模式方面具有卓越的能力，近来备受关注。作为文化的媒介，语言在这些模型的多语言能力中起着关键作用，从而塑造了它们的文化机制。在本研究中，我们通过描述文化维度，文化领域和文化概念的三个层次来探索TTI模型中嵌入的文化感知。我们提出了一套全面的评估技术，包括使用CLIP空间进行内在评估，使用视觉问答（VQA）模型进行外在评估以及人类评估，以识别TTI文化感知。为了促进我们的研究，我们引入了CulText2I数据集，该数据集来自四个不同的TTI模型，涵盖了十种语言。我们的实验揭示了这些模型的文化意识、文化区别和

    Text-To-Image (TTI) models, exemplified by DALL-E and StableDiffusion, have recently gained prominence for their remarkable zero-shot capabilities in generating images guided by textual prompts. Language, as a conduit of culture, plays a pivotal role in these models' multilingual capabilities, which in turn shape their cultural agency. In this study, we explore the cultural perception embedded in TTI models by characterizing culture across three hierarchical tiers: cultural dimensions, cultural domains, and cultural concepts. We propose a comprehensive suite of evaluation techniques, including intrinsic evaluations using the CLIP space, extrinsic evaluations with a Visual-Question-Answer (VQA) model, and human assessments, to discern TTI cultural perceptions. To facilitate our research, we introduce the CulText2I dataset, derived from four diverse TTI models and spanning ten languages. Our experiments reveal insights into these models' cultural awareness, cultural distinctions, and the
    
[^6]: DeepSpeed-VisualChat：通过多模态因果关注实现的多轮多图交错聊天

    DeepSpeed-VisualChat: Multi-Round Multi-Image Interleave Chat via Multi-Modal Causal Attention. (arXiv:2309.14327v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2309.14327](http://arxiv.org/abs/2309.14327)

    DeepSpeed-VisualChat是一个用于多轮多图交错聊天的框架，通过引入创新的多模态因果关注机制和数据融合技术，具有优越的可扩展性。

    

    现有的大部分多模态模型由于无法熟练地处理多图、多回合对话中交错的图像和文本输入，面临着在训练资源分配和数据可访问性方面的重要限制，这影响了它们在不同交互领域中的适应性和可扩展性。为了解决这个问题，我们提出了 DeepSpeed-VisualChat 框架，旨在通过融合多模态功能，集中提高大型视觉和语言模型处理交错输入的能力。我们的框架的显著特点在于：(1) 提供对多轮多图对话的开源支持，(2) 引入创新的多模态因果关注机制，以及 (3) 在现有数据集上使用数据融合技术，以确保多轮多图对话中的无缝交互。与现有框架相比，DeepSpeed-VisualChat 在可扩展性方面展现出卓越的表现，可达到 70B 参数。

    Most of the existing multi-modal models, hindered by their incapacity to adeptly manage interleaved image-and-text inputs in multi-image, multi-round dialogues, face substantial constraints in resource allocation for training and data accessibility, impacting their adaptability and scalability across varied interaction realms. To address this, we present the DeepSpeed-VisualChat framework, designed to optimize Large Language Models (LLMs) by incorporating multi-modal capabilities, with a focus on enhancing the proficiency of Large Vision and Language Models in handling interleaved inputs. Our framework is notable for (1) its open-source support for multi-round and multi-image dialogues, (2) introducing an innovative multi-modal causal attention mechanism, and (3) utilizing data blending techniques on existing datasets to assure seamless interactions in multi-round, multi-image conversations. Compared to existing frameworks, DeepSpeed-VisualChat shows superior scalability up to 70B para
    
[^7]: 自监督变形器中的分别归一化

    On Separate Normalization in Self-supervised Transformers. (arXiv:2309.12931v1 [cs.CL])

    [http://arxiv.org/abs/2309.12931](http://arxiv.org/abs/2309.12931)

    在自监督变形器中，通过为标记和[CLS]符号分别使用归一化层，可以更好地捕捉它们各自的特点并提高下游任务的性能。

    

    自监督变形器的训练方法在各个领域展现了显著的性能。以往的基于变形器的模型（如遮蔽自编码器）通常会为[CLS]符号和标记使用单独的归一化层。我们在本文中提出了一种简单的修改，为标记和[CLS]符号分别使用归一化层，以更好地捕捉它们各自的特点并增强下游任务的性能。我们的方法旨在缓解将相同的归一化统计数据应用于两种标记类型可能带来的负面效果，这些统计数据可能无法与它们各自的角色最佳匹配。通过使用单独的归一化层，我们经验证明[CLS]嵌入能够更好地编码全局语境信息，并在其非各向同性空间中分布更均匀。当用这两个单独的归一化层替换常规的归一化层时，我们观察到平均性能提升了2.7%。

    Self-supervised training methods for transformers have demonstrated remarkable performance across various domains. Previous transformer-based models, such as masked autoencoders (MAE), typically utilize a single normalization layer for both the [CLS] symbol and the tokens. We propose in this paper a simple modification that employs separate normalization layers for the tokens and the [CLS] symbol to better capture their distinct characteristics and enhance downstream task performance. Our method aims to alleviate the potential negative effects of using the same normalization statistics for both token types, which may not be optimally aligned with their individual roles. We empirically show that by utilizing a separate normalization layer, the [CLS] embeddings can better encode the global contextual information and are distributed more uniformly in its anisotropic space. When replacing the conventional normalization layer with the two separate layers, we observe an average 2.7% performa
    
[^8]: 大型语言模型的可解释性：一项调查

    Explainability for Large Language Models: A Survey. (arXiv:2309.01029v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.01029](http://arxiv.org/abs/2309.01029)

    本文调研了大型语言模型的可解释性问题，提出了一个解释技术的分类法，并介绍了基于Transformer的语言模型的解释方法。同时，讨论了评估生成解释的度量标准，以及如何利用解释来调试模型和提高性能。

    

    大型语言模型（LLMs）在自然语言处理中展示出令人印象深刻的能力。然而，它们的内部机制仍然不明确，这种缺乏透明度为下游应用带来了不必要的风险。因此，理解和解释这些模型对于阐明它们的行为、限制和社会影响至关重要。在本文中，我们引入了一个可解释性技术的分类法，并提供了一种结构化的概述方法，用于解释基于Transformer的语言模型。我们根据LLMs的训练范式将技术进行分类：传统的微调范式和提示范式。对于每个范式，我们总结了生成个体预测的局部解释和整体模型知识的全局解释的目标和主要方法。我们还讨论了评估生成解释的度量标准，并讨论了如何利用解释来调试模型和提高性能。

    Large language models (LLMs) have demonstrated impressive capabilities in natural language processing. However, their internal mechanisms are still unclear and this lack of transparency poses unwanted risks for downstream applications. Therefore, understanding and explaining these models is crucial for elucidating their behaviors, limitations, and social impacts. In this paper, we introduce a taxonomy of explainability techniques and provide a structured overview of methods for explaining Transformer-based language models. We categorize techniques based on the training paradigms of LLMs: traditional fine-tuning-based paradigm and prompting-based paradigm. For each paradigm, we summarize the goals and dominant approaches for generating local explanations of individual predictions and global explanations of overall model knowledge. We also discuss metrics for evaluating generated explanations, and discuss how explanations can be leveraged to debug models and improve performance. Lastly, 
    
[^9]: 探索对比演示和显著性图在上下文学习中的作用

    Towards Understanding In-Context Learning with Contrastive Demonstrations and Saliency Maps. (arXiv:2307.05052v1 [cs.CL])

    [http://arxiv.org/abs/2307.05052](http://arxiv.org/abs/2307.05052)

    本研究探索了对比演示和显著性图在上下文学习中的作用，并发现改变标签对显著性有显著影响，尤其对于更大的语言模型更为明显。在情感分析任务中，将表达情感的术语改为中性词并不像改变标签那样具有显著影响。另外，补充解释在提高上下文学习方面是有效的。

    

    本文研究了在大型语言模型的上下文学习(ICL)性能中，各种演示组件的作用。具体而言，我们探讨了标签、输入分布和补充解释等因素的影响，特别是在这些因素被修改或扰动时的影响。我们基于之前的工作，这些工作对于这些元素如何影响ICL给出了不一致的结果。为了探究这些问题，我们采用了可解释的自然语言处理(XNLP)方法，并利用对比演示的显著性图进行定性和定量分析。我们的研究结果表明，改变标签对显著性有显著影响，尤其对于更大的语言模型更为明显。我们对输入分布进行了粒度级别的分析，发现在情感分析任务中，将表达情感的术语改为中性词并不像改变标签那样具有显著影响。最后，我们发现补充解释在提高ICL方面的效果是存在的。

    We investigate the role of various demonstration components in the in-context learning (ICL) performance of large language models (LLMs). Specifically, we explore the impacts of ground-truth labels, input distribution, and complementary explanations, particularly when these are altered or perturbed. We build on previous work, which offers mixed findings on how these elements influence ICL. To probe these questions, we employ explainable NLP (XNLP) methods and utilize saliency maps of contrastive demonstrations for both qualitative and quantitative analysis. Our findings reveal that flipping ground-truth labels significantly affects the saliency, though it's more noticeable in larger LLMs. Our analysis of the input distribution at a granular level reveals that changing sentiment-indicative terms in a sentiment analysis task to neutral ones does not have as substantial an impact as altering ground-truth labels. Finally, we find that the effectiveness of complementary explanations in boos
    
[^10]: 针对航空领域进行句子变换器的适应性研究

    Adapting Sentence Transformers for the Aviation Domain. (arXiv:2305.09556v1 [cs.CL])

    [http://arxiv.org/abs/2305.09556](http://arxiv.org/abs/2305.09556)

    本研究提出了一种针对航空领域的句子变换器调整方法，在预训练阶段使用TSDAE模型进行改进，然后在少量注释的数据集上进行微调，实验结果表明在航空相关的自然语言处理任务中取得了最好的表现。

    

    学习有效的句子表示对于许多自然语言处理任务至关重要，包括语义搜索、语义文本相似度（STS）和聚类。虽然已经开发了多个用于句子嵌入学习的变形器模型，但是这些模型在处理具有唯一特征的专业领域时，如航空领域，可能无法发挥最佳性能，因为航空领域包含特殊术语、缩写词和非传统语法等领域特有特点。此外，缺乏标记的数据集使得难以专门训练航空领域的模型。为了解决这些挑战，我们提出了一种针对航空领域调整句子变换器的新方法。我们的方法是一个两阶段的过程，包括预训练和微调。在预训练阶段，我们使用含航空文本数据的变形器和序列去噪自编码器(TSDAE)作为输入来提高初始模型性能。随后，我们使用少量注释的航空数据集进行自然语言推理（NLI）任务来微调我们的模型。在几个与航空相关的自然语言处理任务上的实验结果表明，我们的方法明显优于基准变换模型，并在某些情况下取得了最新的结果。

    Learning effective sentence representations is crucial for many Natural Language Processing (NLP) tasks, including semantic search, semantic textual similarity (STS), and clustering. While multiple transformer models have been developed for sentence embedding learning, these models may not perform optimally when dealing with specialized domains like aviation, which has unique characteristics such as technical jargon, abbreviations, and unconventional grammar. Furthermore, the absence of labeled datasets makes it difficult to train models specifically for the aviation domain. To address these challenges, we propose a novel approach for adapting sentence transformers for the aviation domain. Our method is a two-stage process consisting of pre-training followed by fine-tuning. During pre-training, we use Transformers and Sequential Denoising AutoEncoder (TSDAE) with aviation text data as input to improve the initial model performance. Subsequently, we fine-tune our models using a Natural 
    
[^11]: SUR-adapter：用大型语言模型增强文本-图像预训练扩散模型

    SUR-adapter: Enhancing Text-to-Image Pre-trained Diffusion Models with Large Language Models. (arXiv:2305.05189v1 [cs.CL])

    [http://arxiv.org/abs/2305.05189](http://arxiv.org/abs/2305.05189)

    本文提出了一个名为SUR-adapter的微调方法，用于增强预先训练的文本到图像扩散模型的语义理解和常识推理能力，以便在生成图片时使用简短的叙述提示。作者还构建了一个新的数据集SURD，并使用大型语言模型的知识进行了优化。

    

    扩散模型是目前流行的文本到图像生成模型，可以通过文本提示生成具有高质量和内容丰富度的图像。但是，当输入的提示为简短的叙述时，现有模型在语义理解和常识推理方面存在一定限制，导致图像生成的质量较低。为了提高叙述提示的能力，我们提出了一种简单而有效的参数高效的微调方法，称为Semantic Understanding和Reasoning adapter（SUR-adapter），用于预先训练的扩散模型。为实现这一目标，我们首先收集和注释一个新的数据集SURD，其中包含超过57,000个语义修正的多模态样本。每个样本都包含一个简单的叙述提示，一个复杂的基于关键字的提示和一个高质量的图像。然后，我们将叙述提示的语义表示与复杂提示对齐，并通过大型语言模型的知识将其转移至我们的SUR-adapter中。

    Diffusion models, which have emerged to become popular text-to-image generation models, can produce high-quality and content-rich images guided by textual prompts. However, there are limitations to semantic understanding and commonsense reasoning in existing models when the input prompts are concise narrative, resulting in low-quality image generation. To improve the capacities for narrative prompts, we propose a simple-yet-effective parameter-efficient fine-tuning approach called the Semantic Understanding and Reasoning adapter (SUR-adapter) for pre-trained diffusion models. To reach this goal, we first collect and annotate a new dataset SURD which consists of more than 57,000 semantically corrected multi-modal samples. Each sample contains a simple narrative prompt, a complex keyword-based prompt, and a high-quality image. Then, we align the semantic representation of narrative prompts to the complex prompts and transfer knowledge of large language models (LLMs) to our SUR-adapter vi
    
[^12]: 使用大型语言模型探索人类化翻译策略

    Exploring Human-Like Translation Strategy with Large Language Models. (arXiv:2305.04118v1 [cs.CL])

    [http://arxiv.org/abs/2305.04118](http://arxiv.org/abs/2305.04118)

    本文提出了一个名为MAPS的框架，使LLMs能够模仿人类翻译的过程，该过程包括分析源文本并提取关键词、主题和相关演示以指导翻译过程。该框架实验结果显示明显优于多个强基线，为开展使用LLM实现人类化翻译策略的有前途的方向提供了启示。

    

    大型语言模型（LLMs）在各种场景下展现出了惊人的能力，表现出了接近甚至超越人类智能的水平。在其多种技能中，LLM的翻译能力受到了广泛的关注。与传统的机器翻译仅关注源目标映射不同，基于LLM的翻译可以潜在地模仿人类翻译的过程，该过程会采取许多准备步骤以确保高质量的翻译。本文旨在通过提出MAPS框架（Multi-Aspect Prompting and Selection）探索这种可能性。具体来说，我们使LLM首先分析给定源文本并提取三个与翻译相关的知识方面：关键词、主题和相关演示以指导翻译过程。为了过滤掉噪声和无用的知识，我们采用基于质量估计的选择机制。实验证明，我们的框架在多个语言对和翻译方向上显着优于多个强基线。这项工作为开展使用LLM实现人类化翻译策略的有前途的方向提供了启示。

    Large language models (LLMs) have demonstrated impressive capabilities in general scenarios, exhibiting a level of aptitude that approaches, in some aspects even surpasses, human-level intelligence. Among their numerous skills, the translation abilities of LLMs have received considerable attention. In contrast to traditional machine translation that focuses solely on source-target mapping, LLM-based translation can potentially mimic the human translation process that takes many preparatory steps to ensure high-quality translation. This work aims to explore this possibility by proposing the MAPS framework, which stands for Multi-Aspect Prompting and Selection. Specifically, we enable LLMs to first analyze the given source text and extract three aspects of translation-related knowledge: keywords, topics and relevant demonstrations to guide the translation process. To filter out the noisy and unhelpful knowledge, we employ a selection mechanism based on quality estimation. Experiments sug
    

