# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SSLCL: An Efficient Model-Agnostic Supervised Contrastive Learning Framework for Emotion Recognition in Conversations.](http://arxiv.org/abs/2310.16676) | SSLCL是一种高效的无模型偏向的监督对比学习框架，用于对话中的情感识别。它解决了基于SCL的方法中大批量大小的限制和与现有ERC模型不兼容的问题，并通过投影离散标签到密集表示中来改善特征的鲁棒性和泛化能力。 |
| [^2] | [Investigating Uncertainty Calibration of Aligned Language Models under the Multiple-Choice Setting.](http://arxiv.org/abs/2310.11732) | 本研究系统评估了在多选设置下对齐语言模型不确定性校准的影响。研究发现，在这种设置下存在两种不确定性，分别对答案决策和格式偏好负责。对齐模型过度自信的原因之一是这两种不确定性的混淆。 |
| [^3] | [Lifelong Sequence Generation with Dynamic Module Expansion and Adaptation.](http://arxiv.org/abs/2310.09886) | 这项研究提出了一种动态模块扩展和自适应的方法，旨在解决终身序列生成中的持续学习问题。该方法允许模型根据任务相关性动态决定获取新知识的架构，并选择相似的先前任务来帮助适应新任务。此外，还引入了动态梯度缩放来平衡学习过程，以避免对先前学到的知识的严重遗忘。 |
| [^4] | [A novel approach to measuring patent claim scope based on probabilities obtained from (large) language models.](http://arxiv.org/abs/2309.10003) | 本文提出了一种使用概率从语言模型中获得的自信息来测量专利权要求范围的新方法。该方法通过计算要求的发生概率和自信息来评估要求的信息量，进而反映出要求的范围。研究结果表明，不同类型的语言模型对范围测量的影响不同，最简单的模型可以将范围度量简化为单词或字符计数的倒数。此方法在九个系列的专利权要求上进行了验证，结果表明各系列的要求范围逐渐减小。 |
| [^5] | [Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation.](http://arxiv.org/abs/2308.15363) | 本文提出了一个大规模语言模型(LLMs)赋能的文本到SQL任务的基准评估，并基于实验结果提出了一种新的集成解决方案DAIL-SQL，刷新了Spider榜单并实现了86.6%的执行准确率。同时，强调了在提示工程中的词汇效率以实现高效经济的LLM-based文本到SQL解决方案，此外还对在上下文学习中应用开源LLMs进行了研究，并进行了任务特定的性能优化。 |
| [^6] | [Evaluating the Generation Capabilities of Large Chinese Language Models.](http://arxiv.org/abs/2308.04823) | 本文首次对大型中文语言模型在多个学科领域的生成能力进行了全面评估，并提出了Gscore作为衡量生成结果质量的综合指数。 |
| [^7] | [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models.](http://arxiv.org/abs/2306.07691) | 本文提出了一种名为StyleTTS 2的TTS模型，通过对抗训练和大型语音语言模型来实现人类级别的语音合成。与以往模型不同，StyleTTS 2将样式视为潜在随机变量，利用扩散模型来生成最适合文本的样式，同时受益于大型SLMs和新的可微分持续时间建模。 |
| [^8] | [Benchmarking Robustness of Adaptation Methods on Pre-trained Vision-Language Models.](http://arxiv.org/abs/2306.02080) | 研究针对预训练视觉语言模型的11种适应方法在不同污染情况下的鲁棒性，发现适应方法对文本污染更敏感，单独使用小型文本适配器比共享适配器更鲁棒，可获得可比较的干净性能。 |
| [^9] | [Emergent inabilities? Inverse scaling over the course of pretraining.](http://arxiv.org/abs/2305.14681) | 该研究探讨了模型在训练中针对特定任务的表现是否会下降，发现Pythia模型在更多的参数下，尽管整体表现良好，但在引述重复和重新定义数学两个任务的训练中性能下降。需要在任何时候测试模型在相应基准上的表现。 |
| [^10] | [Learning to Initialize: Can Meta Learning Improve Cross-task Generalization in Prompt Tuning?.](http://arxiv.org/abs/2302.08143) | 本文研究了元Prompt Tuning（MPT）如何帮助改善跨任务泛化能力。使用元学习可以从其他相关任务中学习初始化Prompt嵌入，我们提出并实验了代表性的元学习算法，并在大量的少样本任务中证明了MPT的有效性，特别是在分类任务中。 |
| [^11] | [Data Selection for Language Models via Importance Resampling.](http://arxiv.org/abs/2302.03169) | 通过重要性重采样方法，我们提出了一种高效且可扩展的数据选择框架（DSIR），可以在语言模型中选择适合的预训练数据集。我们使用KL减少作为数据度量来确定合适的特征空间，并在降维特征空间中估计重要性权重以进行数据选择。 |
| [^12] | [Talk the Walk: Synthetic Data Generation for Conversational Music Recommendation.](http://arxiv.org/abs/2301.11489) | TalkTheWalk 是一种新技术，通过利用精心策划的项目收藏中的领域专业知识来合成逼真高质量的会话数据，解决了构建会话式推荐系统所需的训练数据收集的困难。 |
| [^13] | [MEAL: Stable and Active Learning for Few-Shot Prompting.](http://arxiv.org/abs/2211.08358) | 本文提出了MEAL方法，是一个可以在少量样本下进行分类，稳定和活跃的少样本学习方法。方法包含两个贡献，一个是提出新颖的减少运行变异性的集成方法，另一个是引入AL准则用于数据选择。 |
| [^14] | [Unsupervised Opinion Summarization Using Approximate Geodesics.](http://arxiv.org/abs/2209.07496) | 本文提出了一种名为GeoSumm的新型系统，用于执行无监督的抽取式意见摘要。它使用基于编码器解码器的表示学习模型和近似测地线距离的得分机制，在保证性能的同时鲁棒性也得到了保证。 |
| [^15] | [Predicting Influenza A Viral Host Using PSSM and Word Embeddings.](http://arxiv.org/abs/2201.01140) | 该研究利用机器学习模型和特征提取方法，成功预测了甲型流感病毒的原始宿主，为早期和快速控制病毒传播提供了帮助。 |
| [^16] | [CoPaSul Manual -- Contour-based parametric and superpositional intonation stylization.](http://arxiv.org/abs/1612.04765) | CoPaSul工具包提供了自动的韵律标注和特征提取功能，使用了基于轮廓的参数化和叠加韵律风格化的方法。通过该工具包可以得到与韵律边界和突出性相关的特征，并可以通过系数聚类得到韵律轮廓类别。 |

# 详细

[^1]: SSLCL: 一种高效的无模型偏向的监督对比学习框架用于对话中的情感识别

    SSLCL: An Efficient Model-Agnostic Supervised Contrastive Learning Framework for Emotion Recognition in Conversations. (arXiv:2310.16676v1 [cs.CL])

    [http://arxiv.org/abs/2310.16676](http://arxiv.org/abs/2310.16676)

    SSLCL是一种高效的无模型偏向的监督对比学习框架，用于对话中的情感识别。它解决了基于SCL的方法中大批量大小的限制和与现有ERC模型不兼容的问题，并通过投影离散标签到密集表示中来改善特征的鲁棒性和泛化能力。

    

    对话中的情感识别 (ERC) 是自然语言处理领域中一个快速发展的任务，旨在检测对话中发言者表达的情感。最近，越来越多的ERC方法专注于利用监督对比学习 (SCL) 来增强学到的特征的鲁棒性和泛化能力。然而，当前ERC中基于SCL的方法受到大批量大小的限制，并且与大多数现有的ERC模型不兼容。为了解决这些挑战，我们提出了一种高效且无模型偏向的SCL框架，名为Supervised Sample-Label Contrastive Learning with Soft-HGR Maximal Correlation (SSLCL)，它消除了对大批量大小的需求，并且可以与现有的ERC模型无缝集成而不引入任何特定于模型的假设。具体来说，我们介绍了一种利用标签表示的新视角，通过将离散标签投影到密集表示中来实现。

    Emotion recognition in conversations (ERC) is a rapidly evolving task within the natural language processing community, which aims to detect the emotions expressed by speakers during a conversation. Recently, a growing number of ERC methods have focused on leveraging supervised contrastive learning (SCL) to enhance the robustness and generalizability of learned features. However, current SCL-based approaches in ERC are impeded by the constraint of large batch sizes and the lack of compatibility with most existing ERC models. To address these challenges, we propose an efficient and model-agnostic SCL framework named Supervised Sample-Label Contrastive Learning with Soft-HGR Maximal Correlation (SSLCL), which eliminates the need for a large batch size and can be seamlessly integrated with existing ERC models without introducing any model-specific assumptions. Specifically, we introduce a novel perspective on utilizing label representations by projecting discrete labels into dense embeddi
    
[^2]: 在多选设置下研究对齐语言模型的不确定性校准

    Investigating Uncertainty Calibration of Aligned Language Models under the Multiple-Choice Setting. (arXiv:2310.11732v1 [cs.LG])

    [http://arxiv.org/abs/2310.11732](http://arxiv.org/abs/2310.11732)

    本研究系统评估了在多选设置下对齐语言模型不确定性校准的影响。研究发现，在这种设置下存在两种不确定性，分别对答案决策和格式偏好负责。对齐模型过度自信的原因之一是这两种不确定性的混淆。

    

    尽管在对齐语言模型（LM）的实际应用中取得了显著进展，但它们倾向于与预训练的LM相比，在输出答案时表现出过度自信。本研究系统评估了对齐过程对多选设置下LM的基于逻辑的不确定性校准的影响。我们首先对对齐LM在校准方面与其预训练对应模型之间的差异进行了认真的实证研究。实验结果显示，在多选设置下，LM存在两种明显的不确定性，分别负责答案决策和LM的格式偏好。然后，我们通过在简单的合成对齐方案中进行微调，研究了这两种不确定性在对齐LM的校准中的作用，并得出结论，对齐LM过度自信的原因之一是这两种不确定性的混淆。此外，我们还检查了常见的事后校准方法的实用性。

    Despite the significant progress made in practical applications of aligned language models (LMs), they tend to be overconfident in output answers compared to the corresponding pre-trained LMs. In this work, we systematically evaluate the impact of the alignment process on logit-based uncertainty calibration of LMs under the multiple-choice setting. We first conduct a thoughtful empirical study on how aligned LMs differ in calibration from their pre-trained counterparts. Experimental results reveal that there are two distinct uncertainties in LMs under the multiple-choice setting, which are responsible for the answer decision and the format preference of the LMs, respectively. Then, we investigate the role of these two uncertainties on aligned LM's calibration through fine-tuning in simple synthetic alignment schemes and conclude that one reason for aligned LMs' overconfidence is the conflation of these two types of uncertainty. Furthermore, we examine the utility of common post-hoc cal
    
[^3]: 动态模块扩展和自适应的终身序列生成

    Lifelong Sequence Generation with Dynamic Module Expansion and Adaptation. (arXiv:2310.09886v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.09886](http://arxiv.org/abs/2310.09886)

    这项研究提出了一种动态模块扩展和自适应的方法，旨在解决终身序列生成中的持续学习问题。该方法允许模型根据任务相关性动态决定获取新知识的架构，并选择相似的先前任务来帮助适应新任务。此外，还引入了动态梯度缩放来平衡学习过程，以避免对先前学到的知识的严重遗忘。

    

    终身序列生成（LSG）是持续学习中的一个问题，旨在让模型在一系列生成任务上进行持续训练，以不断学习新的生成模式并避免遗忘先前的知识。现有的LSG方法主要关注维持旧知识，而对跨任务的知识传递关注较少。相比之下，人类可以通过利用先前获取的类似任务的知识更好地学习新任务。受人类学习范式的启发，我们提出了动态模块扩展和自适应（DMEA）的方法，该方法使模型能够根据任务相关性动态确定获取新知识的架构，并选择最相似的先前任务来促进对新任务的适应。此外，由于学习过程很容易偏向于当前任务，这可能导致更严重的遗忘先前学到的知识，因此我们提出了动态梯度缩放来平衡学习过程。

    Lifelong sequence generation (LSG), a problem in continual learning, aims to continually train a model on a sequence of generation tasks to learn constantly emerging new generation patterns while avoiding the forgetting of previous knowledge. Existing LSG methods mainly focus on maintaining old knowledge while paying little attention to knowledge transfer across tasks. In contrast, humans can better learn new tasks by leveraging previously acquired knowledge from similar tasks. Inspired by the learning paradigm of humans, we propose Dynamic Module Expansion and Adaptation (DMEA), which enables the model to dynamically determine the architecture for acquiring new knowledge based on task correlation and select the most similar previous tasks to facilitate adaptation to new tasks. In addition, as the learning process can easily be biased towards the current task which might cause more severe forgetting of previously learned knowledge, we propose dynamic gradient scaling to balance the lea
    
[^4]: 基于语言模型的概率测量专利权要求范围的新方法

    A novel approach to measuring patent claim scope based on probabilities obtained from (large) language models. (arXiv:2309.10003v1 [cs.CL])

    [http://arxiv.org/abs/2309.10003](http://arxiv.org/abs/2309.10003)

    本文提出了一种使用概率从语言模型中获得的自信息来测量专利权要求范围的新方法。该方法通过计算要求的发生概率和自信息来评估要求的信息量，进而反映出要求的范围。研究结果表明，不同类型的语言模型对范围测量的影响不同，最简单的模型可以将范围度量简化为单词或字符计数的倒数。此方法在九个系列的专利权要求上进行了验证，结果表明各系列的要求范围逐渐减小。

    

    本文提出了一种将专利权要求的范围测量为该要求所包含的自信息的倒数的方法。这种方法基于信息论，基于一个假设，即罕见的概念比平常的概念更具信息量，因为它更令人惊讶。自信息是从该要求的发生概率计算得出的，其中概率是根据语言模型计算的。本文考虑了五个语言模型，从最简单的模型（每个单词或字符均从均匀分布中抽取）到中等模型（使用平均词或字符频率），再到一个大型语言模型（GPT2）。有趣的是，最简单的语言模型将范围度量减少为单词或字符计数的倒数，这是先前作品中已经使用的度量标准。该方法应用于九个系列的针对不同发明的专利权要求，其中每个系列的要求范围逐渐减小。

    This work proposes to measure the scope of a patent claim as the reciprocal of the self-information contained in this claim. Grounded in information theory, this approach is based on the assumption that a rare concept is more informative than a usual concept, inasmuch as it is more surprising. The self-information is calculated from the probability of occurrence of that claim, where the probability is calculated in accordance with a language model. Five language models are considered, ranging from the simplest models (each word or character is drawn from a uniform distribution) to intermediate models (using average word or character frequencies), to a large language model (GPT2). Interestingly, the simplest language models reduce the scope measure to the reciprocal of the word or character count, a metric already used in previous works. Application is made to nine series of patent claims directed to distinct inventions, where the claims in each series have a gradually decreasing scope.
    
[^5]: 大语言模型赋能文本到SQL的研究：一个基准评估

    Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation. (arXiv:2308.15363v1 [cs.DB])

    [http://arxiv.org/abs/2308.15363](http://arxiv.org/abs/2308.15363)

    本文提出了一个大规模语言模型(LLMs)赋能的文本到SQL任务的基准评估，并基于实验结果提出了一种新的集成解决方案DAIL-SQL，刷新了Spider榜单并实现了86.6%的执行准确率。同时，强调了在提示工程中的词汇效率以实现高效经济的LLM-based文本到SQL解决方案，此外还对在上下文学习中应用开源LLMs进行了研究，并进行了任务特定的性能优化。

    

    大语言模型(LLMs)已经成为文本到SQL任务的一种新范式。然而，缺乏一个系统性的基准阻碍了设计有效、高效和经济的LLM-based文本到SQL解决方案的发展。为了解决这一挑战，本文首先对现有的提示工程方法进行了系统性和广泛的比较，包括问题表示、示例选择和示例组织，并根据实验结果详细阐述了它们的优缺点。基于这些发现，我们提出了一种新的集成解决方案，名为DAIL-SQL，刷新了Spider榜单，达到了86.6%的执行准确率，建立了一个新的标杆。为了实现高效经济的LLM-based文本到SQL解决方案，我们强调提示工程中的词汇效率，并在此度量下比较了之前的研究。此外，我们还研究了上下文学习中的开源LLMs，并用任务特定的监督进行了进一步的性能优化。

    Large language models (LLMs) have emerged as a new paradigm for Text-to-SQL task. However, the absence of a systematical benchmark inhibits the development of designing effective, efficient and economic LLM-based Text-to-SQL solutions. To address this challenge, in this paper, we first conduct a systematical and extensive comparison over existing prompt engineering methods, including question representation, example selection and example organization, and with these experimental results, we elaborates their pros and cons. Based on these findings, we propose a new integrated solution, named DAIL-SQL, which refreshes the Spider leaderboard with 86.6% execution accuracy and sets a new bar. Towards an efficient and economic LLM-based Text-to-SQL solution, we emphasize the token efficiency in prompt engineering and compare the prior studies under this metric. Additionally, we investigate open-source LLMs in in-context learning, and further enhance their performance with task-specific superv
    
[^6]: 评估大型中文语言模型的生成能力

    Evaluating the Generation Capabilities of Large Chinese Language Models. (arXiv:2308.04823v1 [cs.CL])

    [http://arxiv.org/abs/2308.04823](http://arxiv.org/abs/2308.04823)

    本文首次对大型中文语言模型在多个学科领域的生成能力进行了全面评估，并提出了Gscore作为衡量生成结果质量的综合指数。

    

    本文介绍了CG-Eval，这是第一个对大型中文语言模型在多个学科领域生成能力进行全面评估的研究。通过在科学工程、人文社科、数学计算、医师资格考试、司法考试和注册会计师考试六个学科中生成准确和相关的回答，评估了这些模型的性能。本文还提出了Gscore，这是一个由多个度量指标加权求和得到的综合指数，用于衡量模型生成结果与参考答案的质量。测试数据和测试结果可在此http URL找到。

    This paper presents CG-Eval, the first comprehensive evaluation of the generation capabilities of large Chinese language models across a wide range of academic disciplines. The models' performance was assessed based on their ability to generate accurate and relevant responses to different types of questions in six disciplines, namely, Science and Engineering, Humanities and Social Sciences, Mathematical Calculations, Medical Practitioner Qualification Examination, Judicial Examination, and Certified Public Accountant Examination. This paper also presents Gscore, a composite index derived from the weighted sum of multiple metrics to measure the quality of model's generation against a reference. The test data and test results can be found at this http URL
    
[^7]: StyleTTS 2：通过风格扩散和与大型语音语言模型的对抗训练实现人类级别的语音合成

    StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models. (arXiv:2306.07691v1 [eess.AS])

    [http://arxiv.org/abs/2306.07691](http://arxiv.org/abs/2306.07691)

    本文提出了一种名为StyleTTS 2的TTS模型，通过对抗训练和大型语音语言模型来实现人类级别的语音合成。与以往模型不同，StyleTTS 2将样式视为潜在随机变量，利用扩散模型来生成最适合文本的样式，同时受益于大型SLMs和新的可微分持续时间建模。

    

    本文提出了StyleTTS2，一种文本到语音（TTS）模型，该模型利用风格扩散和与大型语音语言模型（SLM）的对抗性训练来实现人类级别的TTS合成。 StyleTTS 2通过将样式建模为潜在的随机变量通过扩散模型来生成最适合文本的样式，无需参考语音，实现高效的潜在扩散，同时受益于扩散模型提供的多样化语音合成。此外，我们使用大型预先训练的SLMs（例如WavLM）作为鉴别器，并使用我们的新型可微分持续时间建模进行端到端的训练，从而提高了语音的自然度。 StyleTTS 2在单扬声器LJSpeech数据集上超越了人类录音，并在多扬声器VCTK数据集上与之匹配，经过母语为英语的人员评判。此外，当在LibriTTS数据集上进行训练时，我们的模型胜过了以前公开可用的零样本说话人语音合成模型。

    In this paper, we present StyleTTS 2, a text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis. StyleTTS 2 differs from its predecessor by modeling styles as a latent random variable through diffusion models to generate the most suitable style for the text without requiring reference speech, achieving efficient latent diffusion while benefiting from the diverse speech synthesis offered by diffusion models. Furthermore, we employ large pre-trained SLMs, such as WavLM, as discriminators with our novel differentiable duration modeling for end-to-end training, resulting in improved speech naturalness. StyleTTS 2 surpasses human recordings on the single-speaker LJSpeech dataset and matches it on the multispeaker VCTK dataset as judged by native English speakers. Moreover, when trained on the LibriTTS dataset, our model outperforms previous publicly available models for zero-shot speaker
    
[^8]: 预训练视觉语言模型适应方法的鲁棒性基准测试研究

    Benchmarking Robustness of Adaptation Methods on Pre-trained Vision-Language Models. (arXiv:2306.02080v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.02080](http://arxiv.org/abs/2306.02080)

    研究针对预训练视觉语言模型的11种适应方法在不同污染情况下的鲁棒性，发现适应方法对文本污染更敏感，单独使用小型文本适配器比共享适配器更鲁棒，可获得可比较的干净性能。

    

    提升预训练视觉语言模型在特定领域表现的各种适应方法，如 LoRA、prompts 和 adapters 等已被提出。然而，这些适应方法对于分布位移的鲁棒性尚未得到研究。本研究评估了11种广泛使用的适应方法在4个视觉语言数据集上的鲁棒性，考察了可用适应示例和适应过程中可训练参数大小的影响。具体地，引入了7个基准数据集，包括96种视觉和87种文本污损，以研究不同适应方法的鲁棒性。我们的分析揭示了：1）适应方法对文本污染比视觉污染更敏感。2) 全量微调并不总能提供最高的鲁棒性；相反，适配器可以实现更好的鲁棒性，并具有可比较的干净性能。3）与预期相反，我们的发现表明，单独使用小型文本适配器通常比在视觉和语言空间中共享适配器更鲁棒。

    Various adaptation methods, such as LoRA, prompts, and adapters, have been proposed to enhance the performance of pre-trained vision-language models in specific domains. The robustness of these adaptation methods against distribution shifts have not been studied. In this study, we assess the robustness of 11 widely-used adaptation methods across 4 vision-language datasets under multimodal corruptions. Concretely, we introduce 7 benchmark datasets, including 96 visual and 87 textual corruptions, to investigate the robustness of different adaptation methods, the impact of available adaptation examples, and the influence of trainable parameter size during adaptation. Our analysis reveals that: 1) Adaptation methods are more sensitive to text corruptions than visual corruptions. 2) Full fine-tuning does not consistently provide the highest robustness; instead, adapters can achieve better robustness with comparable clean performance. 3) Contrary to expectations, our findings indicate that i
    
[^9]: 训练过程中的逆比例缩放现象：累赘还是贡献？

    Emergent inabilities? Inverse scaling over the course of pretraining. (arXiv:2305.14681v1 [cs.CL])

    [http://arxiv.org/abs/2305.14681](http://arxiv.org/abs/2305.14681)

    该研究探讨了模型在训练中针对特定任务的表现是否会下降，发现Pythia模型在更多的参数下，尽管整体表现良好，但在引述重复和重新定义数学两个任务的训练中性能下降。需要在任何时候测试模型在相应基准上的表现。

    

    模型参数大小是否会导致逆比例缩放现象呢？这个研究探讨了语言模型在训练中针对特定任务的表现是否会下降，尽管整体表现仍然不错。研究发现在逆比例缩放挑战中的两个任务 - 引述重复和重新定义数学，Pythia模型运用更多的参数时，这两个任务在训练过程中的性能确实会下降，尽管这些模型在整体上表现良好。这突显了在任何时候如果模型被训练了额外的数据，那么需要测试模型在相应基准上的表现，即使它们的整体表现有所提高。

    Does inverse scaling only occur as a function of model parameter size, or can it also occur over the course of training? We carry out an exploratory study investigating whether, over the course of training on the language modeling task, the performance of language models at specific tasks can decrease while general performance remains high. We find that for two tasks from the Inverse Scaling Challenge - quote-repetition and redefine-math - this is indeed the case. Specifically, we find that for Pythia (Biderman et al., 2023) models with a higher number of parameters, performance decreases over the course of training at these two tasks, despite these models showing standard (positive) scaling overall. This highlights the importance of testing model performance at all relevant benchmarks any time they are trained on additional data, even if their overall performance improves.
    
[^10]: 学习初始化：元学习能否提高Prompt Tuning跨任务泛化能力？

    Learning to Initialize: Can Meta Learning Improve Cross-task Generalization in Prompt Tuning?. (arXiv:2302.08143v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.08143](http://arxiv.org/abs/2302.08143)

    本文研究了元Prompt Tuning（MPT）如何帮助改善跨任务泛化能力。使用元学习可以从其他相关任务中学习初始化Prompt嵌入，我们提出并实验了代表性的元学习算法，并在大量的少样本任务中证明了MPT的有效性，特别是在分类任务中。

    

    Prompt Tuning (PT)是一种只调整每个任务的一个额外标记序列的嵌入，同时保持预训练完成的语言模型（PLM）不变，已经在少样本学习中展现出了优异的性能。尽管如此，PT已经被证明极大地依赖于很好的Prompt嵌入的初始化。本文研究了元Prompt Tuning (MPT) 来系统地探索元学习如何帮助通过从其他相关任务学习初始化Prompt嵌入来改善（如果可以）PT中的跨任务泛化能力。我们在大量的少样本任务上使用广泛的实验和分析来经验分析了一系列代表性的元学习算法，分析不同源/目标任务配置下的各种调整设置。通过广泛的实验和分析，我们证明了MPT的有效性。我们发现它特别在分类任务上的提升是显著的。对于其他类型的任务，例如问题回答，我们观察到，虽然MPT可以在大多数情况下优于PT，

    Prompt tuning (PT) which only tunes the embeddings of an additional sequence of tokens per task, keeping the pre-trained language model (PLM) frozen, has shown remarkable performance in few-shot learning. Despite this, PT has been shown to rely heavily on good initialization of the prompt embeddings. In this work, we study meta prompt tuning (MPT) to systematically explore how meta-learning can help improve (if it can) cross-task generalization in PT through learning to initialize the prompt embeddings from other relevant tasks. We empirically analyze a representative set of meta learning algorithms in a wide range of adaptation settings with different source/target task configurations on a large set of few-shot tasks. With extensive experiments and analysis, we demonstrate the effectiveness of MPT. We find the improvement to be significant particularly on classification tasks. For other kinds of tasks such as question answering, we observe that while MPT can outperform PT in most case
    
[^11]: 通过重要性重采样进行语言模型的数据选择

    Data Selection for Language Models via Importance Resampling. (arXiv:2302.03169v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.03169](http://arxiv.org/abs/2302.03169)

    通过重要性重采样方法，我们提出了一种高效且可扩展的数据选择框架（DSIR），可以在语言模型中选择适合的预训练数据集。我们使用KL减少作为数据度量来确定合适的特征空间，并在降维特征空间中估计重要性权重以进行数据选择。

    

    选择适合的预训练数据集对于通用领域（如GPT-3）和特定领域（如Codex）的语言模型（LM）都至关重要。我们将这个问题形式化为从大型原始无标签数据集中选择一个子集，以匹配给定一些无标签目标样本的所需目标分布。鉴于原始文本数据的大规模和高维度，现有方法使用简单的启发式方法或专家手动策划数据。相反，我们扩展了在LM数据选择中使用的经典重要性重采样方法，以低维空间进行数据选择。我们提出了一种高效且可扩展的框架，称为数据选择与重要性重采样（DSIR），它在一个降维特征空间中估计重要性权重，以便根据这些权重进行重要性重采样数据选择。为了确定一个合适的特征空间，我们还展示了KL减少，一种在特征空间中衡量所选预训练数据与目标之间相似度的数据度量，具有较高的相关性。

    Selecting a suitable pretraining dataset is crucial for both general-domain (e.g., GPT-3) and domain-specific (e.g., Codex) language models (LMs). We formalize this problem as selecting a subset of a large raw unlabeled dataset to match a desired target distribution given some unlabeled target samples. Due to the large scale and dimensionality of the raw text data, existing methods use simple heuristics or use experts to manually curate data. Instead, we extend the classic importance resampling approach used in low-dimensions for LM data selection. We propose Data Selection with Importance Resampling (DSIR), an efficient and scalable framework that estimates importance weights in a reduced feature space for tractability and selects data with importance resampling according to these weights. To determine an appropriate feature space, we show that KL reduction, a data metric that measures the proximity between selected pretraining data and the target in a feature space, has high correlat
    
[^12]: Talk the Walk: 针对会话式音乐推荐的合成数据生成

    Talk the Walk: Synthetic Data Generation for Conversational Music Recommendation. (arXiv:2301.11489v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.11489](http://arxiv.org/abs/2301.11489)

    TalkTheWalk 是一种新技术，通过利用精心策划的项目收藏中的领域专业知识来合成逼真高质量的会话数据，解决了构建会话式推荐系统所需的训练数据收集的困难。

    

    推荐系统广泛存在，但用户往往很难在推荐质量较差时进行控制和调整。这促使了会话式推荐系统(CRSs)的发展，通过自然语言反馈提供对推荐的控制。然而，构建会话式推荐系统需要包含用户话语和涵盖多样化偏好范围的项目的会话训练数据。使用传统方法如众包，这样的数据收集起来非常困难。我们在项目集推荐的背景下解决了这个问题，注意到这个任务受到越来越多关注，动机在于音乐、新闻和食谱推荐等使用案例。我们提出了一种新技术TalkTheWalk，通过利用广泛可获得的精心策划的项目收藏中的领域专业知识来合成逼真高质量的会话数据，并展示了如何将其转化为相应的项目集策划。

    Recommendation systems are ubiquitous yet often difficult for users to control and adjust when recommendation quality is poor. This has motivated the development of conversational recommendation systems (CRSs), with control over recommendations provided through natural language feedback. However, building conversational recommendation systems requires conversational training data involving user utterances paired with items that cover a diverse range of preferences. Such data has proved challenging to collect scalably using conventional methods like crowdsourcing. We address it in the context of item-set recommendation, noting the increasing attention to this task motivated by use cases like music, news and recipe recommendation. We present a new technique, TalkTheWalk, that synthesizes realistic high-quality conversational data by leveraging domain expertise encoded in widely available curated item collections, showing how these can be transformed into corresponding item set curation c
    
[^13]: MEAL：少样本提示的稳定和活跃学习

    MEAL: Stable and Active Learning for Few-Shot Prompting. (arXiv:2211.08358v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.08358](http://arxiv.org/abs/2211.08358)

    本文提出了MEAL方法，是一个可以在少量样本下进行分类，稳定和活跃的少样本学习方法。方法包含两个贡献，一个是提出新颖的减少运行变异性的集成方法，另一个是引入AL准则用于数据选择。

    

    通过启动和提示，基础模型已经成为高效的少样本学习器，在少样本分类方面取得了巨大进展。然而，这种方法在不同的少样本集合（数据选择）和不同的微调运行（运行变异性）之间具有高变化率，这不仅阻碍了不同方法之间的公平比较，而且使得少样本学习对许多实际应用过于不可靠。为了缓解这些问题，我们提出了两个贡献，用于更稳定和有效的少样本学习：首先，我们提出了新颖的集成方法，并展示了它们可以大幅减少运行变异性。其次，我们引入了一种新的主动学习（AL）准则用于数据选择，并呈现了第一个专门针对基于提示的学习的AL方法。在实验中，我们展示了我们的联合方法MEAL（多提示微调与预测集成与主动学习）可以稳定地在少量样本下进行分类。

    Few-shot classification has made great strides due to foundation models that, through priming and prompting, are highly effective few-shot learners. However, this approach has high variance both across different sets of few shots (data selection) and across different finetuning runs (run variability). This is problematic not only because it impedes the fair comparison of different approaches, but especially because it makes few-shot learning too unreliable for many real-world applications. To alleviate these issues, we make two contributions for more stable and effective few-shot learning: First, we propose novel ensembling methods and show that they substantially reduce run variability. Second, we introduce a new active learning (AL) criterion for data selection and present the first AL-based approach specifically tailored towards prompt-based learning. In our experiments, we show that our combined method, MEAL (Multiprompt finetuning and prediction Ensembling with Active Learning), i
    
[^14]: 使用近似测地线的无监督意见摘要。

    Unsupervised Opinion Summarization Using Approximate Geodesics. (arXiv:2209.07496v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2209.07496](http://arxiv.org/abs/2209.07496)

    本文提出了一种名为GeoSumm的新型系统，用于执行无监督的抽取式意见摘要。它使用基于编码器解码器的表示学习模型和近似测地线距离的得分机制，在保证性能的同时鲁棒性也得到了保证。

    

    意见摘要是从用户评论中捕获流行观点的任务。本文介绍了一种名为Geodesic Summarizer (GeoSumm)的新型系统，用于执行无监督的抽取式意见摘要。GeoSumm涉及基于编码器解码器的表示学习模型，通过在多个解码器层上对预训练文本表示进行词典学习来生成文本的表示形式为潜在语义单元的分布。然后，我们使用这些表示来量化评论句子的相关性，使用一种基于近似测地线距离的得分机制。我们使用相关性评分来识别流行观点，从而组成普遍性和方面特定的摘要。我们提出的模型GeoSumm在三个意见摘要数据集上取得了最先进的性能。我们进行了额外的实验来分析我们的模型的功能，并展示了生成的摘要。

    Opinion summarization is the task of creating summaries capturing popular opinions from user reviews. In this paper, we introduce Geodesic Summarizer (GeoSumm), a novel system to perform unsupervised extractive opinion summarization. GeoSumm involves an encoder-decoder based representation learning model, that generates representations of text as a distribution over latent semantic units. GeoSumm generates these representations by performing dictionary learning over pre-trained text representations at multiple decoder layers. We then use these representations to quantify the relevance of review sentences using a novel approximate geodesic distance based scoring mechanism. We use the relevance scores to identify popular opinions in order to compose general and aspect-specific summaries. Our proposed model, GeoSumm, achieves state-of-the-art performance on three opinion summarization datasets. We perform additional experiments to analyze the functioning of our model and showcase the gene
    
[^15]: 利用PSSM和词嵌入预测甲型流感病毒宿主

    Predicting Influenza A Viral Host Using PSSM and Word Embeddings. (arXiv:2201.01140v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2201.01140](http://arxiv.org/abs/2201.01140)

    该研究利用机器学习模型和特征提取方法，成功预测了甲型流感病毒的原始宿主，为早期和快速控制病毒传播提供了帮助。

    

    流感病毒的快速突变威胁公共健康，可能引发致命的大流行病。然而，检测病毒的原始宿主在爆发期间或爆发后是困难的，因为流感病毒可以在不同的物种之间循环传播。因此，早期和快速检测病毒宿主将有助于减少病毒的进一步传播。我们使用基于位置特异性得分矩阵（PSSM）和学习自词嵌入和词编码的特征的各种机器学习模型来推断病毒的原始宿主。结果表明，基于PSSM的模型的性能达到了95%左右的MCC和96%左右的F1。使用词嵌入的模型得到的MCC约为96％，F1约为97％。

    The rapid mutation of the influenza virus threatens public health. Reassortment among viruses with different hosts can lead to a fatal pandemic. However, it is difficult to detect the original host of the virus during or after an outbreak as influenza viruses can circulate between different species. Therefore, early and rapid detection of the viral host would help reduce the further spread of the virus. We use various machine learning models with features derived from the position-specific scoring matrix (PSSM) and features learned from word embedding and word encoding to infer the origin host of viruses. The results show that the performance of the PSSM-based model reaches the MCC around 95%, and the F1 around 96%. The MCC obtained using the model with word embedding is around 96%, and the F1 is around 97%.
    
[^16]: CoPaSul手册--基于轮廓的参数化和叠加韵律风格化

    CoPaSul Manual -- Contour-based parametric and superpositional intonation stylization. (arXiv:1612.04765v11 [cs.CL] UPDATED)

    [http://arxiv.org/abs/1612.04765](http://arxiv.org/abs/1612.04765)

    CoPaSul工具包提供了自动的韵律标注和特征提取功能，使用了基于轮廓的参数化和叠加韵律风格化的方法。通过该工具包可以得到与韵律边界和突出性相关的特征，并可以通过系数聚类得到韵律轮廓类别。

    

    CoPaSul工具包的目的是自动的韵律标注和从音节到语句级别的韵律特征提取。在这个框架下，韵律被表示为全局和局部轮廓的叠加，这些轮廓在多项式系数的参数化描述下。在全局层面上（通常与但不一定限于语调短语相关），风格化用于以时间变化的F0水平和范围来表示音调。在局部层面上（例如，重音组），描述局部轮廓形状。通过这种参数化方法，可以得到几个与韵律边界和突出性相关的特征。此外，通过系数聚类，可以以自下而上的方式获得韵律轮廓类别。除了基于风格化的特征提取外，还可以使用标准的F0和能量测量（例如，平均值和方差）以及韵律方面的特征。

    The purposes of the CoPaSul toolkit are (1) automatic prosodic annotation and (2) prosodic feature extraction from syllable to utterance level. CoPaSul stands for contour-based, parametric, superpositional intonation stylization. In this framework intonation is represented as a superposition of global and local contours that are described parametrically in terms of polynomial coefficients. On the global level (usually associated but not necessarily restricted to intonation phrases) the stylization serves to represent register in terms of time-varying F0 level and range. On the local level (e.g. accent groups), local contour shapes are described. From this parameterization several features related to prosodic boundaries and prominence can be derived. Furthermore, by coefficient clustering prosodic contour classes can be obtained in a bottom-up way. Next to the stylization-based feature extraction also standard F0 and energy measures (e.g. mean and variance) as well as rhythmic aspects c
    

