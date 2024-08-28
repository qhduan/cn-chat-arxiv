# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Computational Analysis of Lyric Similarity Perception](https://arxiv.org/abs/2404.02342) | 该研究通过比较分析计算方法对模拟歌词相似度与人类感知的关联，发现基于BERT模型嵌入、歌词音频和音素组件相似性的计算模型对感知上的歌词相似度具有指示作用。 |
| [^2] | [CMAT: A Multi-Agent Collaboration Tuning Framework for Enhancing Small Language Models](https://arxiv.org/abs/2404.01663) | CMAT框架引入了TinyAgent模型，并提出了一种新颖的系统，通过环境反馈进行自适应权重更新，增强了语言智能体的能力和长期记忆。 |
| [^3] | [Learning to Decode Collaboratively with Multiple Language Models](https://arxiv.org/abs/2403.03870) | 学习了一种协作解码方法，通过在标记级别交错生成来教授多个大型语言模型协作，无需直接监督，在特定任务中融合每个模型的专业知识，提高了联合系统的性能。 |
| [^4] | [DIVERSE: Deciphering Internet Views on the U.S. Military Through Video Comment Stance Analysis, A Novel Benchmark Dataset for Stance Classification](https://arxiv.org/abs/2403.03334) | 本文提出了一个名为DIVERSE的数据集，其中包含超过173,000条YouTube视频评论，标注了这些评论对美国军事视频的立场，采用了一种通过人类引导、机器辅助的标注方法，使用了句子中的弱信号作为支持指标。 |
| [^5] | [RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations](https://arxiv.org/abs/2402.17700) | RAVEL数据集介绍了一种新方法MDAS，该方法在解开语言模型表示方面取得了最新的成果，强调了跨激活特征的重要性。 |
| [^6] | [OWSM-CTC: An Open Encoder-Only Speech Foundation Model for Speech Recognition, Translation, and Language Identification](https://arxiv.org/abs/2402.12654) | 提出了OWSM-CTC，这是一种基于Connectionist Temporal Classification的新型仅编码器语音基础模型，训练有180k小时的公共音频数据，用于多语言自动语音识别（ASR）、语音翻译（ST）和语言识别。 |
| [^7] | [A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260) | 提出了一种新的基准 StrongREJECT，通过使用更高质量的问题，更好地区分有效和无效的空破解方法。 |
| [^8] | [LLaVA-Docent: Instruction Tuning with Multimodal Large Language Model to Support Art Appreciation Education](https://arxiv.org/abs/2402.06264) | 本研究利用多模态大型语言模型（MLLM）开发了LLaVA-Docent模型，以支持艺术鉴赏教育。通过综述文献和专家咨询，构建了数据框架，并使用该框架生成了虚拟对话数据集用于训练MLLM。该研究对于解决传统艺术鉴赏教育中的资源限制和主流教育中的科学技术工程和数学偏重具有重要意义。 |
| [^9] | [Unified Speech-Text Pretraining for Spoken Dialog Modeling](https://arxiv.org/abs/2402.05706) | 本研究提出了一个名为统一口语对话模型（USDM）的广泛语音文本模型框架，用于生成与输入语音相关的连贯口语回复。通过使用多步骤的语音文本推理方式和广义语音文本预训练方案，该方法能够有效捕捉跨模态语义，并生成自然流畅的口语回复。 |
| [^10] | [OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer.](http://arxiv.org/abs/2401.16658) | 该论文介绍了OWSM v3.1基于E-Branchformer的更好和更快的开放式Whisper风格语音模型。这个模型通过提高性能和效率，超越了之前的版本，并实现了更快的推理速度。该论文还公开发布了相关的数据和模型。 |
| [^11] | [LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning.](http://arxiv.org/abs/2311.12023) | LQ-LoRA是一种低秩加量化矩阵分解方法，用于内存高效的语言模型微调。它通过将每个预训练矩阵分解为高精度低秩部分和内存高效的量化部分，实现了动态配置量化参数以及对重构目标进行加权的优化，并在微调实验中表现出了优于QLoRA和GPTQ-LoRA的效果。 |
| [^12] | [Affective Visual Dialog: A Large-Scale Benchmark for Emotional Reasoning Based on Visually Grounded Conversations.](http://arxiv.org/abs/2308.16349) | 我们引入了一个名为AffectVisDial的大规模数据集，其中包含50,000个基于视觉的对话，我们训练了情感视觉对话模型来解决基于对话的问答、情感预测和情感解释任务，展示出了有希望的情感推理能力。 |
| [^13] | [Exploiting the Potential of Seq2Seq Models as Robust Few-Shot Learners.](http://arxiv.org/abs/2307.14856) | Seq2Seq模型作为少样本学习器的潜力在解码器和编码-解码模型中进行了广泛研究，提出了两种能有效提升Seq2Seq模型上下文学习能力的方法，并在各种任务中显示出显著的性能改进。 |

# 详细

[^1]: 歌词相似度感知的计算分析

    A Computational Analysis of Lyric Similarity Perception

    [https://arxiv.org/abs/2404.02342](https://arxiv.org/abs/2404.02342)

    该研究通过比较分析计算方法对模拟歌词相似度与人类感知的关联，发现基于BERT模型嵌入、歌词音频和音素组件相似性的计算模型对感知上的歌词相似度具有指示作用。

    

    在包含人声的音乐作品中，歌词对艺术表达起着重要作用。因此，先前的研究引入了推荐系统的概念，该系统建议类似于用户喜爱或个性化偏好的歌词，有助于在数百万音轨中发现歌词。然而，许多系统并未充分考虑人类对歌词相似度的感知，主要是由于该领域的研究有限。为弥补这一差距，我们进行了对计算方法建模歌词相似度与人类感知进行了比较分析。结果表明，基于预训练的BERT模型嵌入之间的相似性、歌词来源的音频以及音素组件的计算模型指示了感知上的歌词相似度。该发现强调了语义、风格和音韵相似性在人类感知中的重要性。

    arXiv:2404.02342v1 Announce Type: new  Abstract: In musical compositions that include vocals, lyrics significantly contribute to artistic expression. Consequently, previous studies have introduced the concept of a recommendation system that suggests lyrics similar to a user's favorites or personalized preferences, aiding in the discovery of lyrics among millions of tracks. However, many of these systems do not fully consider human perceptions of lyric similarity, primarily due to limited research in this area. To bridge this gap, we conducted a comparative analysis of computational methods for modeling lyric similarity with human perception. Results indicated that computational models based on similarities between embeddings from pre-trained BERT-based models, the audio from which the lyrics are derived, and phonetic components are indicative of perceptual lyric similarity. This finding underscores the importance of semantic, stylistic, and phonetic similarities in human perception abo
    
[^2]: CMAT: 用于增强小型语言模型的多智能体协作调整框架

    CMAT: A Multi-Agent Collaboration Tuning Framework for Enhancing Small Language Models

    [https://arxiv.org/abs/2404.01663](https://arxiv.org/abs/2404.01663)

    CMAT框架引入了TinyAgent模型，并提出了一种新颖的系统，通过环境反馈进行自适应权重更新，增强了语言智能体的能力和长期记忆。

    

    开放的大型语言模型（LLMs）显著推动了自然语言处理领域的发展，在各种任务中展现出卓越的性能。尽管LLMs取得了显著进展，但它们的有效操作仍然严重依赖于人类输入来准确引导对话流程，智能体调整是一种关键的优化技术，涉及人类对模型的调整，以更好地响应这种引导。针对这一依赖性，我们的工作引入了TinyAgent模型，该模型经过精心策划的高质量数据集训练。我们还提出了Collaborative Multi-Agent Tuning（CMAT）框架，这是一个创新性系统，旨在通过根据环境反馈进行自适应权重更新来增强语言智能体的能力。该框架促进了多个智能体之间的协作学习和实时适应，增强了它们的上下文感知和长期记忆。

    arXiv:2404.01663v1 Announce Type: new  Abstract: Open large language models (LLMs) have significantly advanced the field of natural language processing, showcasing impressive performance across various tasks.Despite the significant advancements in LLMs, their effective operation still relies heavily on human input to accurately guide the dialogue flow, with agent tuning being a crucial optimization technique that involves human adjustments to the model for better response to such guidance.Addressing this dependency, our work introduces the TinyAgent model, trained on a meticulously curated high-quality dataset. We also present the Collaborative Multi-Agent Tuning (CMAT) framework, an innovative system designed to augment language agent capabilities through adaptive weight updates based on environmental feedback. This framework fosters collaborative learning and real-time adaptation among multiple intelligent agents, enhancing their context-awareness and long-term memory. In this resear
    
[^3]: 学习多个语言模型的协作解码方法

    Learning to Decode Collaboratively with Multiple Language Models

    [https://arxiv.org/abs/2403.03870](https://arxiv.org/abs/2403.03870)

    学习了一种协作解码方法，通过在标记级别交错生成来教授多个大型语言模型协作，无需直接监督，在特定任务中融合每个模型的专业知识，提高了联合系统的性能。

    

    我们提出了一种方法，通过在标记级别交错它们的生成来教授多个大型语言模型（LLM）协作。我们将下一个标记由哪个LLM生成的决策建模为潜变量。通过在我们的潜变量模型下优化训练集的边际似然，基础LLM自动学习何时生成自身以及何时调用其中一个“助手”语言模型来生成，而无需直接监督。在解码过程中进行标记级协作允许融合每个模型的专业知识，以符合特定任务的方式。我们的协作解码在跨领域设置中特别有用，其中通用基础LLM学会调用领域专家模型。在执行指令、领域特定问答和推理任务时，我们展示了联合系统的性能优于单独模型。通过对学习到的潜

    arXiv:2403.03870v1 Announce Type: new  Abstract: We propose a method to teach multiple large language models (LLM) to collaborate by interleaving their generations at the token level. We model the decision of which LLM generates the next token as a latent variable. By optimizing the marginal likelihood of a training set under our latent variable model, the base LLM automatically learns when to generate itself and when to call on one of the ``assistant'' language models to generate, all without direct supervision. Token-level collaboration during decoding allows for a fusion of each model's expertise in a manner tailored to the specific task at hand. Our collaborative decoding is especially useful in cross-domain settings where a generalist base LLM learns to invoke domain expert models. On instruction-following, domain-specific QA, and reasoning tasks, we show that the performance of the joint system exceeds that of the individual models. Through qualitative analysis of the learned lat
    
[^4]: DIVERSE：通过视频评论态度分析解读互联网对美国军事的看法，一个用于立场分类的新颖基准数据集

    DIVERSE: Deciphering Internet Views on the U.S. Military Through Video Comment Stance Analysis, A Novel Benchmark Dataset for Stance Classification

    [https://arxiv.org/abs/2403.03334](https://arxiv.org/abs/2403.03334)

    本文提出了一个名为DIVERSE的数据集，其中包含超过173,000条YouTube视频评论，标注了这些评论对美国军事视频的立场，采用了一种通过人类引导、机器辅助的标注方法，使用了句子中的弱信号作为支持指标。

    

    社交媒体文本的立场检测是涉及识别在有争议主题上拥有相反观点的用户群组的下游任务的关键组成部分，如疫苗接种和争论中。具体来说，立场提供了对实体立场的指示。本文介绍了DIVERSE，这是一个包含对超过173,000个YouTube视频评论进行标注的数据集，标注了这些评论对于美国军事视频的立场。这些立场通过一种由人类引导、机器辅助的标注方法进行标注，该方法利用了句子中蕴含的语气弱信号作为支持指标，而非使用人类手动注释。这些弱信号包括仇恨言论和讽刺的存在，特定关键词的存在，文本的情感以及从两个大型语言模型中推断的立场。然后，在每个评论被注释之前，这些弱信号使用数据编程模型进行 consol

    arXiv:2403.03334v1 Announce Type: cross  Abstract: Stance detection of social media text is a key component of downstream tasks involving the identification of groups of users with opposing opinions on contested topics such as vaccination and within arguments. In particular, stance provides an indication of an opinion towards an entity. This paper introduces DIVERSE, a dataset of over 173,000 YouTube video comments annotated for their stance towards videos of the U.S. military. The stance is annotated through a human-guided, machine-assisted labeling methodology that makes use of weak signals of tone within the sentence as supporting indicators, as opposed to using manual annotations by humans. These weak signals consist of the presence of hate speech and sarcasm, the presence of specific keywords, the sentiment of the text, and the stance inference from two Large Language Models. The weak signals are then consolidated using a data programming model before each comment is annotated wit
    
[^5]: RAVEL: 在解开语言模型表示方面评估可解释性方法

    RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations

    [https://arxiv.org/abs/2402.17700](https://arxiv.org/abs/2402.17700)

    RAVEL数据集介绍了一种新方法MDAS，该方法在解开语言模型表示方面取得了最新的成果，强调了跨激活特征的重要性。

    

    个别神经元参与多个高级概念的表示。不同的可解释性方法在多大程度上能成功解开这些角色？为了帮助解决这个问题，我们介绍了RAVEL（Resolving Attribute-Value Entanglements in Language Models），这是一个数据集，可以实现对多种现有可解释性方法进行紧密控制的定量比较。我们利用由此产生的概念框架来定义新的Multi-task Distributed Alignment Search（MDAS）方法，该方法能够找到满足多个因果标准的分布式表示。以Llama2-7B作为目标语言模型，MDAS在RAVEL上取得了最新的成果，展示了超越神经元级别分析以识别跨激活的特征的重要性。我们在https://github.com/explanare/ravel上发布了我们的基准。

    arXiv:2402.17700v1 Announce Type: new  Abstract: Individual neurons participate in the representation of multiple high-level concepts. To what extent can different interpretability methods successfully disentangle these roles? To help address this question, we introduce RAVEL (Resolving Attribute-Value Entanglements in Language Models), a dataset that enables tightly controlled, quantitative comparisons between a variety of existing interpretability methods. We use the resulting conceptual framework to define the new method of Multi-task Distributed Alignment Search (MDAS), which allows us to find distributed representations satisfying multiple causal criteria. With Llama2-7B as the target language model, MDAS achieves state-of-the-art results on RAVEL, demonstrating the importance of going beyond neuron-level analyses to identify features distributed across activations. We release our benchmark at https://github.com/explanare/ravel.
    
[^6]: OWSM-CTC:一种用于语音识别、翻译和语言识别的开放编码器基础模型

    OWSM-CTC: An Open Encoder-Only Speech Foundation Model for Speech Recognition, Translation, and Language Identification

    [https://arxiv.org/abs/2402.12654](https://arxiv.org/abs/2402.12654)

    提出了OWSM-CTC，这是一种基于Connectionist Temporal Classification的新型仅编码器语音基础模型，训练有180k小时的公共音频数据，用于多语言自动语音识别（ASR）、语音翻译（ST）和语言识别。

    

    近来对能够在单个模型中执行多个语音处理任务的大型语音模型越来越感兴趣。这些模型通常采用编码器-解码器或仅解码器架构，因为它们在许多领域中非常流行且性能良好。然而，与非自回归模型相比，自回归模型在推断时可能会比较慢，并且还存在幻觉的潜在风险。尽管先前的研究观察到非自回归模型在小规模任务中产生了令人满意的结果，但尚不清楚它们是否可以扩展到不同语言和任务的语音转文本生成中。受Open Whisper-style Speech Model (OWSM)项目的启发，我们提出了OWSM-CTC，这是一种基于Connectionist Temporal Classification (CTC)的新型仅编码器的语音基础模型。它使用18万小时的公共音频数据进行训练，用于多语言自动语音识别（ASR）、语音翻译（ST）和语言识别。

    arXiv:2402.12654v1 Announce Type: new  Abstract: There has been an increasing interest in large speech models that can perform multiple speech processing tasks in a single model. Such models usually adopt the encoder-decoder or decoder-only architecture due to their popularity and good performance in many domains. However, autoregressive models can be slower during inference compared to non-autoregressive models and also have potential risks of hallucination. Though prior studies observed promising results of non-autoregressive models for certain tasks at small scales, it remains unclear if they can be scaled to speech-to-text generation in diverse languages and tasks. Inspired by the Open Whisper-style Speech Model (OWSM) project, we propose OWSM-CTC, a novel encoder-only speech foundation model based on Connectionist Temporal Classification (CTC). It is trained on 180k hours of public audio data for multilingual automatic speech recognition (ASR), speech translation (ST), and languag
    
[^7]: 一种用于空破解的强REJECT方法

    A StrongREJECT for Empty Jailbreaks

    [https://arxiv.org/abs/2402.10260](https://arxiv.org/abs/2402.10260)

    提出了一种新的基准 StrongREJECT，通过使用更高质量的问题，更好地区分有效和无效的空破解方法。

    

    大型语言模型（LLMs）的兴起引起了对“破解”的关注，这种破解允许模型被恶意使用。然而，目前没有标准的基准来衡量破解的严重程度，导致破解论文的作者不得不自行创建标准。我们表明这些基准经常包含模棱两可或无法回答的问题，并使用倾向于高估低质量模型响应的滥用潜力的评分标准。一些破解技术使问题更加严重，因为它们即使对于良性问题也会降低模型响应的质量：我们展示了几种破解技术显着降低了GPT-4在MMLU上的零射击表现。破解还会使从“未经审查”的开源模型中获取有害响应变得更加困难。我们提出了一个新的基准，StrongREJECT，通过使用更高质量的问题更好地区分有效和无效的破解方法。

    arXiv:2402.10260v1 Announce Type: cross  Abstract: The rise of large language models (LLMs) has drawn attention to the existence of "jailbreaks" that allow the models to be used maliciously. However, there is no standard benchmark for measuring the severity of a jailbreak, leaving authors of jailbreak papers to create their own. We show that these benchmarks often include vague or unanswerable questions and use grading criteria that are biased towards overestimating the misuse potential of low-quality model responses. Some jailbreak techniques make the problem worse by decreasing the quality of model responses even on benign questions: we show that several jailbreaking techniques substantially reduce the zero-shot performance of GPT-4 on MMLU. Jailbreaks can also make it harder to elicit harmful responses from an "uncensored" open-source model. We present a new benchmark, StrongREJECT, which better discriminates between effective and ineffective jailbreaks by using a higher-quality que
    
[^8]: LLaVA-Docent：利用多模态大型语言模型支持艺术鉴赏教育的教学调优

    LLaVA-Docent: Instruction Tuning with Multimodal Large Language Model to Support Art Appreciation Education

    [https://arxiv.org/abs/2402.06264](https://arxiv.org/abs/2402.06264)

    本研究利用多模态大型语言模型（MLLM）开发了LLaVA-Docent模型，以支持艺术鉴赏教育。通过综述文献和专家咨询，构建了数据框架，并使用该框架生成了虚拟对话数据集用于训练MLLM。该研究对于解决传统艺术鉴赏教育中的资源限制和主流教育中的科学技术工程和数学偏重具有重要意义。

    

    艺术鉴赏对于培养学习者的批判性思维和情感智力至关重要。然而，传统的艺术鉴赏教育常面临艺术资源有限的问题，特别是对于弱势学生，并且在主流教育中过度强调科学技术工程和数学科目。为了应对这些挑战，最近的技术进步为创新解决方案铺平了道路。本研究探索了多模态大型语言模型（MLLM）在艺术鉴赏教育中的应用，重点是开发了LLaVA-Docent模型来利用这些进展。我们的方法包括全面的文献综述和与领域专家的咨询，从而形成了一个强大的数据框架。利用这个框架，我们生成了一个虚拟对话数据集，该数据集被GPT-4利用。这个数据集对于训练MLLM（即LLaVA-Docent）起到了关键作用。六名研究人员进行了定量和定性评估。

    Art appreciation is vital in nurturing critical thinking and emotional intelligence among learners. However, traditional art appreciation education has often been hindered by limited access to art resources, especially for disadvantaged students, and an imbalanced emphasis on STEM subjects in mainstream education. In response to these challenges, recent technological advancements have paved the way for innovative solutions. This study explores the application of multi-modal large language models (MLLMs) in art appreciation education, focusing on developing LLaVA-Docent, a model that leverages these advancements. Our approach involved a comprehensive literature review and consultations with experts in the field, leading to developing a robust data framework. Utilizing this framework, we generated a virtual dialogue dataset that was leveraged by GPT-4. This dataset was instrumental in training the MLLM, named LLaVA-Docent. Six researchers conducted quantitative and qualitative evaluation
    
[^9]: 面向口语对话建模的统一语音文本预训练方法

    Unified Speech-Text Pretraining for Spoken Dialog Modeling

    [https://arxiv.org/abs/2402.05706](https://arxiv.org/abs/2402.05706)

    本研究提出了一个名为统一口语对话模型（USDM）的广泛语音文本模型框架，用于生成与输入语音相关的连贯口语回复。通过使用多步骤的语音文本推理方式和广义语音文本预训练方案，该方法能够有效捕捉跨模态语义，并生成自然流畅的口语回复。

    

    近期的研究表明，扩展大型语言模型（LLM）以直接理解和合成语音具有良好的结果，但用于口语对话建模的基于LLM的策略仍然难以实现，需要进一步研究。本文提出了一个广泛的语音文本LLM框架，命名为统一口语对话模型（USDM），以在不依赖于自动语音识别（ASR）或文本到语音（TTS）解决方案的情况下生成与给定输入语音相关的连贯口语回复和有机的韵律特征。我们的方法采用了一种多步骤的语音文本推理方式，利用了底层LLM所展示的推理链能力。我们还提出了一种广义的语音文本预训练方案，有助于捕捉跨模态语义。自动和人工评估结果表明，所提出的方法能够有效生成自然流畅的口语回复，并且优于之前的和级联的基线模型。详细的比较研究

    While recent work shows promising results in expanding the capabilities of large language models (LLM) to directly understand and synthesize speech, an LLM-based strategy for modeling spoken dialogs remains elusive and calls for further investigation. This work proposes an extensive speech-text LLM framework, named the Unified Spoken Dialog Model (USDM), to generate coherent spoken responses with organic prosodic features relevant to the given input speech without relying on automatic speech recognition (ASR) or text-to-speech (TTS) solutions. Our approach employs a multi-step speech-text inference scheme that leverages chain-of-reasoning capabilities exhibited by the underlying LLM. We also propose a generalized speech-text pretraining scheme that helps with capturing cross-modal semantics. Automatic and human evaluations show that the proposed approach is effective in generating natural-sounding spoken responses, outperforming both prior and cascaded baselines. Detailed comparative s
    
[^10]: OWSM v3.1: 基于E-Branchformer的更好和更快的开放式Whisper风格语音模型

    OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer. (arXiv:2401.16658v1 [cs.CL])

    [http://arxiv.org/abs/2401.16658](http://arxiv.org/abs/2401.16658)

    该论文介绍了OWSM v3.1基于E-Branchformer的更好和更快的开放式Whisper风格语音模型。这个模型通过提高性能和效率，超越了之前的版本，并实现了更快的推理速度。该论文还公开发布了相关的数据和模型。

    

    最近的研究倡导采用完全开放的基础模型来推动透明度和开放科学。作为一个初步的步骤，开放式Whisper风格语音模型(OWSM)使用公开可用的数据和开源工具重新复制了OpenAI的Whisper。为了复制Whisper，之前的OWSM v1到v3模型仍然基于Transformer，这可能导致性能不如其他最先进的语音编码器。在这项工作中，我们旨在提高OWSM的性能和效率，而无需额外的训练数据。我们提出了基于E-Branchformer的OWSM v3.1模型，有两个规模，即100M和1B。1B模型是目前公开可用的最大的基于E-Branchformer的语音模型。它在大部分评估基准上表现出比之前的OWSM v3更好的性能，同时演示了高达25%的更快推理速度。我们公开发布数据准备脚本、预训练模型和训练日志。

    Recent studies have advocated for fully open foundation models to promote transparency and open science. As an initial step, the Open Whisper-style Speech Model (OWSM) reproduced OpenAI's Whisper using publicly available data and open-source toolkits. With the aim of reproducing Whisper, the previous OWSM v1 through v3 models were still based on Transformer, which might lead to inferior performance compared to other state-of-the-art speech encoders. In this work, we aim to improve the performance and efficiency of OWSM without extra training data. We present E-Branchformer based OWSM v3.1 models at two scales, i.e., 100M and 1B. The 1B model is the largest E-Branchformer based speech model that has been made publicly available. It outperforms the previous OWSM v3 in a vast majority of evaluation benchmarks, while demonstrating up to 25% faster inference speed. We publicly release the data preparation scripts, pre-trained models and training logs.
    
[^11]: LQ-LoRA: 低秩加量化矩阵分解用于有效的语言模型微调

    LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning. (arXiv:2311.12023v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2311.12023](http://arxiv.org/abs/2311.12023)

    LQ-LoRA是一种低秩加量化矩阵分解方法，用于内存高效的语言模型微调。它通过将每个预训练矩阵分解为高精度低秩部分和内存高效的量化部分，实现了动态配置量化参数以及对重构目标进行加权的优化，并在微调实验中表现出了优于QLoRA和GPTQ-LoRA的效果。

    

    我们提出了一种简单的方法，用于对预训练语言模型进行内存高效的自适应。我们的方法使用迭代算法将每个预训练矩阵分解为高精度低秩部分和内存高效的量化部分。在微调过程中，量化部分保持固定，只有低秩部分被更新。我们提出了量化部分的整数线性规划表达，可以根据总体内存预算动态配置量化参数（例如比特宽度、块大小）给定每个矩阵。我们进一步探索了数据感知版本的算法，该算法使用Fisher信息矩阵的近似来加权矩阵分解过程中的重构目标。在RoBERTa和LLaMA-2（7B和70B）的微调实验中，我们的低秩加量化矩阵分解方法（LQ-LoRA）优于强基线方法QLoRA和GPTQ-LoRA，并实现了激进的量化。

    We propose a simple approach for memory-efficient adaptation of pretrained language models. Our approach uses an iterative algorithm to decompose each pretrained matrix into a high-precision low-rank component and a memory-efficient quantized component. During finetuning, the quantized component remains fixed and only the low-rank component is updated. We present an integer linear programming formulation of the quantization component which enables dynamic configuration of quantization parameters (e.g., bit-width, block size) for each matrix given an overall target memory budget. We further explore a data-aware version of the algorithm which uses an approximation of the Fisher information matrix to weight the reconstruction objective during matrix decomposition. Experiments on finetuning RoBERTa and LLaMA-2 (7B and 70B) demonstrate that our low-rank plus quantized matrix decomposition approach (LQ-LoRA) outperforms strong QLoRA and GPTQ-LoRA baselines and enables aggressive quantization
    
[^12]: 情感视觉对话：基于视觉对话理解情感形成的大规模基准

    Affective Visual Dialog: A Large-Scale Benchmark for Emotional Reasoning Based on Visually Grounded Conversations. (arXiv:2308.16349v1 [cs.CL])

    [http://arxiv.org/abs/2308.16349](http://arxiv.org/abs/2308.16349)

    我们引入了一个名为AffectVisDial的大规模数据集，其中包含50,000个基于视觉的对话，我们训练了情感视觉对话模型来解决基于对话的问答、情感预测和情感解释任务，展示出了有希望的情感推理能力。

    

    我们引入了情感视觉对话，作为一个测试平台，用于研究理解在基于视觉对话中情感形成的过程。这项任务涉及三项技能：（1）基于对话的问答，（2）基于对话的情感预测，以及（3）基于对话生成情感解释。我们的主要贡献是构建了一个大规模数据集，称为AffectVisDial，包含50,000个10轮的基于视觉的对话，还包括总结的情感归因和基于对话的情感解释，总共需要27180个工作小时。我们解释了收集该数据集的设计决策，并介绍了与对话参与者相关的提问者和回答者任务。我们训练和展示了来自最先进模型的坚实的情感视觉对话基线。值得注意的是，我们模型生成的回答显示出有希望的情感推理能力。

    We introduce Affective Visual Dialog, an emotion explanation and reasoning task as a testbed for research on understanding the formation of emotions in visually grounded conversations. The task involves three skills: (1) Dialog-based Question Answering (2) Dialog-based Emotion Prediction and (3) Affective emotion explanation generation based on the dialog. Our key contribution is the collection of a large-scale dataset, dubbed AffectVisDial, consisting of 50K 10-turn visually grounded dialogs as well as concluding emotion attributions and dialog-informed textual emotion explanations, resulting in a total of 27,180 working hours. We explain our design decisions in collecting the dataset and introduce the questioner and answerer tasks that are associated with the participants in the conversation. We train and demonstrate solid Affective Visual Dialog baselines adapted from state-of-the-art models. Remarkably, the responses generated by our models show promising emotional reasoning abilit
    
[^13]: 发挥Seq2Seq模型作为稳健少样本学习器的潜力

    Exploiting the Potential of Seq2Seq Models as Robust Few-Shot Learners. (arXiv:2307.14856v1 [cs.CL])

    [http://arxiv.org/abs/2307.14856](http://arxiv.org/abs/2307.14856)

    Seq2Seq模型作为少样本学习器的潜力在解码器和编码-解码模型中进行了广泛研究，提出了两种能有效提升Seq2Seq模型上下文学习能力的方法，并在各种任务中显示出显著的性能改进。

    

    在上下文学习中，只有解码器模型具有明显优势，而编码-解码（即Seq2Seq）模型在依赖于权重更新的方法中表现出色。最近，一些研究表明Seq2Seq模型可以进行少样本学习，但这仅限于与Seq2Seq体系结构相匹配的任务，如摘要和翻译。受到这些初始研究的启发，我们首次进行了广泛的实验，比较了解码器和编码-解码模型在各种任务的上下文少样本学习能力。此外，我们提出了两种能更有效地引发Seq2Seq模型上下文学习能力的方法：目标对齐提示和基于融合的方法。值得注意的是，我们的方法在性能上超过了一个体积是其六倍的解码器模型，并且相较于常规Seq2Seq模型显示出显著的性能改进。

    In-context learning, which offers substantial advantages over fine-tuning, is predominantly observed in decoder-only models, while encoder-decoder (i.e., seq2seq) models excel in methods that rely on weight updates. Recently, a few studies have demonstrated the feasibility of few-shot learning with seq2seq models; however, this has been limited to tasks that align well with the seq2seq architecture, such as summarization and translation. Inspired by these initial studies, we provide a first-ever extensive experiment comparing the in-context few-shot learning capabilities of decoder-only and encoder-decoder models on a broad range of tasks. Furthermore, we propose two methods to more effectively elicit in-context learning ability in seq2seq models: objective-aligned prompting and a fusion-based approach. Remarkably, our approach outperforms a decoder-only model that is six times larger and exhibits significant performance improvements compared to conventional seq2seq models across a var
    

