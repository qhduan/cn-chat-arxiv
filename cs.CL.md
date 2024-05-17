# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TRABSA: Interpretable Sentiment Analysis of Tweets using Attention-based BiLSTM and Twitter-RoBERTa](https://arxiv.org/abs/2404.00297) | TRABSA是一个集成了transformer架构、注意力机制和BiLSTM网络的混合框架，利用RoBERTa在大量推特上训练，填补了情感分析领域的差距，实现了94%的准确性和显著的性能提升。 |
| [^2] | [Interpreting Key Mechanisms of Factual Recall in Transformer-Based Language Models](https://arxiv.org/abs/2403.19521) | 通过深入研究Transformer-based语言模型在事实回忆任务中的机制，我们发现了零/少次样本情况下的特定任务头、MLP层和残差流的功能，以及抗过度自信机制。 |
| [^3] | [FEEL: A Framework for Evaluating Emotional Support Capability with Large Language Models](https://arxiv.org/abs/2403.15699) | 提出了一个基于大型语言模型的框架FEEL，用于评估情感支持能力，解决了当前非人工方法在评估情感支持能力方面面临的挑战，并采用了概率分布方法和集成学习以获得更稳定和全面的结果。 |
| [^4] | [Retrieval augmented text-to-SQL generation for epidemiological question answering using electronic health records](https://arxiv.org/abs/2403.09226) | 结合文本到SQL生成与检索增强生成（RAG）的方法，可用于使用电子健康记录和索赔数据回答流行病学问题，并在现实行业中显示出显著性能提升。 |
| [^5] | [A Modular Approach for Multimodal Summarization of TV Shows](https://arxiv.org/abs/2403.03823) | 提出了一种模块化方法用于多模态电视节目摘要，包括检测场景边界、重新排列场景、将视觉信息转换为文本、总结对话以及将场景摘要融合的过程，并引入了一个新的衡量摘要质量的评价指标PREFS。 |
| [^6] | [Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models](https://arxiv.org/abs/2402.15938) | 本文提出了一种通过LLMs输出分布进行污染检测的方法CDD，以及一种基于LLMs输出修正的可信评估方法TED，以应对大语言模型在数据污染和可信评估方面面临的挑战。 |
| [^7] | [GenTranslate: Large Language Models are Generative Multilingual Speech and Machine Translators](https://arxiv.org/abs/2402.06894) | GenTranslate是一个新的翻译任务生成模型，通过利用大型语言模型的丰富语言知识和强大推理能力，可以从N-best列表中生成更高质量的翻译结果。 |
| [^8] | [A blind spot for large language models: Supradiegetic linguistic information](https://arxiv.org/abs/2306.06794) | 大型语言模型的盲点在于其对超叙事语言信息的忽视，研究提出考虑模型如何感知语言信息有助于深入了解其能力。 |
| [^9] | [Sowing the Wind, Reaping the Whirlwind: The Impact of Editing Language Models.](http://arxiv.org/abs/2401.10647) | 本文研究了通过编辑语言模型的复杂后果，发现在增强模型准确性与保持道德完整性之间存在悖论。我们发现，尽管注入准确信息对模型的可靠性很重要，但它可能破坏模型的基本框架，导致不可预测和潜在的不安全行为。 |
| [^10] | [What makes for a 'good' social actor? Using respect as a lens to evaluate interactions with language agents.](http://arxiv.org/abs/2401.09082) | 本文研究以尊重为视角评估与语言代理的交互，提出了一种更加关注关系和情境因素的伦理方法，旨在帮助LLM技术表现得“好” |
| [^11] | [Large Language Model-Enhanced Algorithm Selection: Towards Comprehensive Algorithm Representation.](http://arxiv.org/abs/2311.13184) | 本论文提出了一种方法，通过将算法表示集成到算法选择中，从而填补了当前算法选择技术对算法特征的研究空白。 |
| [^12] | [MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing.](http://arxiv.org/abs/2306.10012) | MagicBrush是第一个大规模的手动标注的数据集，用于指导真实图像的编辑。它包括超过10K个手动标注的三元组，支持大规模的文本指导图像编辑模型训练。在此数据集上微调InstructPix2Pix可以根据人类评估提供更好的图像。 |
| [^13] | [Adapting Pretrained ASR Models to Low-resource Clinical Speech using Epistemic Uncertainty-based Data Selection.](http://arxiv.org/abs/2306.02105) | 本研究使用基于认识不确定性的数据选择方法来减少非洲口音临床ASR训练的注释成本。结果表明，这种方法可以超过现有的基准结果，并提高低资源口音的泛化能力。 |
| [^14] | [Escaping the sentence-level paradigm in machine translation.](http://arxiv.org/abs/2304.12959) | 本文提出了一种摆脱机器翻译中句子级范式限制的方法，通过处理三个障碍来实现：使用足够大的标准Transformer架构、引入一种简单而有效的技术来将文档级信息转化为适合训练的形式、基于自动文档分类的评估协议来有效地识别文档级翻译质量。在两个非常不同的文档级翻译任务上，我们的实验表明，在此数据上训练的Transformer模型明显优于强大的基线模型。 |

# 详细

[^1]: TRABSA：使用基于注意力的BiLSTM和Twitter-RoBERTa进行可解释的推文情感分析

    TRABSA: Interpretable Sentiment Analysis of Tweets using Attention-based BiLSTM and Twitter-RoBERTa

    [https://arxiv.org/abs/2404.00297](https://arxiv.org/abs/2404.00297)

    TRABSA是一个集成了transformer架构、注意力机制和BiLSTM网络的混合框架，利用RoBERTa在大量推特上训练，填补了情感分析领域的差距，实现了94%的准确性和显著的性能提升。

    

    情感分析对于理解公众舆论和消费者行为至关重要。现有模型面临着语言多样性、泛化能力和可解释性方面的挑战。我们提出了TRABSA，这是一个集成了基于transformer的架构、注意力机制和BiLSTM网络的混合框架，旨在解决这些挑战。利用在124M条推文上训练的RoBERTa，我们填补了情感分析基准测试中的差距，确保了最先进的准确性。通过将来自32个国家和美国各州的推文与数据集相结合，我们比较了六种词嵌入技术和三种基于词典的标注技术，并选择了最佳技术以实现最佳情感分析效果。TRABSA以94%的准确性和显著的精确度、召回率和F1得分增益，胜过了传统的机器学习和深度学习模型。在不同数据集上的评估显示了一致的优越性和泛化能力。SHAP和LIME分析提高了可解释性，增强了信心。

    arXiv:2404.00297v1 Announce Type: new  Abstract: Sentiment analysis is crucial for understanding public opinion and consumer behavior. Existing models face challenges with linguistic diversity, generalizability, and explainability. We propose TRABSA, a hybrid framework integrating transformer-based architectures, attention mechanisms, and BiLSTM networks to address this. Leveraging RoBERTa-trained on 124M tweets, we bridge gaps in sentiment analysis benchmarks, ensuring state-of-the-art accuracy. Augmenting datasets with tweets from 32 countries and US states, we compare six word-embedding techniques and three lexicon-based labeling techniques, selecting the best for optimal sentiment analysis. TRABSA outperforms traditional ML and deep learning models with 94% accuracy and significant precision, recall, and F1-score gains. Evaluation across diverse datasets demonstrates consistent superiority and generalizability. SHAP and LIME analyses enhance interpretability, improving confidence i
    
[^2]: 解释基于Transformer模型的语言模型在事实回忆中的关键机制

    Interpreting Key Mechanisms of Factual Recall in Transformer-Based Language Models

    [https://arxiv.org/abs/2403.19521](https://arxiv.org/abs/2403.19521)

    通过深入研究Transformer-based语言模型在事实回忆任务中的机制，我们发现了零/少次样本情况下的特定任务头、MLP层和残差流的功能，以及抗过度自信机制。

    

    本文深入探讨了Transformer-based语言模型在事实回忆任务中所采用的机制。在零次样本情况下，给定类似“法国的首都是”的提示，特定任务的注意力头会从上下文中提取主题实体，如“法国”，并将其传递给后续的MLP以回忆所需的答案，如“巴黎”。我们引入了一种新颖的分析方法，旨在将MLP的输出分解为人类可理解的组件。通过这种方法，我们量化了跟随这些特定任务头的MLP层的功能。在残差流中，它会擦除或放大来自各个头的信息。此外，它会生成一个组件，将残差流重新定向到预期答案的方向。这些零次机制也适用于少次样本情况。此外，我们观察到一种广泛存在的抗过度自信机制。

    arXiv:2403.19521v1 Announce Type: cross  Abstract: In this paper, we deeply explore the mechanisms employed by Transformer-based language models in factual recall tasks. In zero-shot scenarios, given a prompt like "The capital of France is," task-specific attention heads extract the topic entity, such as "France," from the context and pass it to subsequent MLPs to recall the required answer such as "Paris." We introduce a novel analysis method aimed at decomposing the outputs of the MLP into components understandable by humans. Through this method, we quantify the function of the MLP layer following these task-specific heads. In the residual stream, it either erases or amplifies the information originating from individual heads. Moreover, it generates a component that redirects the residual stream towards the direction of its expected answer. These zero-shot mechanisms are also employed in few-shot scenarios. Additionally, we observed a widely existent anti-overconfidence mechanism in 
    
[^3]: FEEL：用于评估大型语言模型情感支持能力的框架

    FEEL: A Framework for Evaluating Emotional Support Capability with Large Language Models

    [https://arxiv.org/abs/2403.15699](https://arxiv.org/abs/2403.15699)

    提出了一个基于大型语言模型的框架FEEL，用于评估情感支持能力，解决了当前非人工方法在评估情感支持能力方面面临的挑战，并采用了概率分布方法和集成学习以获得更稳定和全面的结果。

    

    情感支持对话（ESC）是一种典型的对话，可以有效地帮助用户缓解情感压力。然而，由于情感分析中涉及固有主观性，当前非人工方法在有效评估情感支持能力方面面临挑战。这些指标与人类判断之间存在很低的相关性。同时，手动评估方法将导致很高的成本。为解决这些问题，我们提出了一个新型模型FEEL（用大型语言模型评估情感支持能力的框架），采用大型语言模型（LLMs）作为评估者来评估情感支持能力。该模型周密考虑ESC的各种评估方面，应用更全面和准确的ESC评估方法。此外，它采用概率分布方法以获得更稳定的结果，并集成了集成学习。

    arXiv:2403.15699v1 Announce Type: new  Abstract: Emotional Support Conversation (ESC) is a typical dialogue that can effec-tively assist the user in mitigating emotional pressures. However, owing to the inherent subjectivity involved in analyzing emotions, current non-artificial methodologies face challenges in effectively appraising the emo-tional support capability. These metrics exhibit a low correlation with human judgments. Concurrently, manual evaluation methods extremely will cause high costs. To solve these problems, we propose a novel model FEEL (Framework for Evaluating Emotional Support Capability with Large Lan-guage Models), employing Large Language Models (LLMs) as evaluators to assess emotional support capabilities. The model meticulously considers var-ious evaluative aspects of ESC to apply a more comprehensive and accurate evaluation method for ESC. Additionally, it employs a probability distribu-tion approach for a more stable result and integrates an ensemble learnin
    
[^4]: 使用电子健康记录的检索增强文本到SQL生成器用于流行病学问题回答

    Retrieval augmented text-to-SQL generation for epidemiological question answering using electronic health records

    [https://arxiv.org/abs/2403.09226](https://arxiv.org/abs/2403.09226)

    结合文本到SQL生成与检索增强生成（RAG）的方法，可用于使用电子健康记录和索赔数据回答流行病学问题，并在现实行业中显示出显著性能提升。

    

    电子健康记录（EHR）和索赔数据是反映患者健康状况和医疗利用情况的丰富现实世界数据来源。查询这些数据库以回答流行病学问题具有挑战性，原因在于医学术语的复杂性和对复杂SQL查询的需求。我们介绍了一种端到端的方法，将文本到SQL生成与检索增强生成（RAG）结合起来，使用EHR和索赔数据回答流行病学问题。我们展示了我们的方法，将医学编码步骤整合到文本到SQL过程中，显著提高了性能，而不仅仅是简单提示。我们的研究结果表明，尽管当前的语言模型尚不足以无监督使用，但RAG为改进它们的能力提供了一个有希望的方向，如在一个现实的行业环境中所示。

    arXiv:2403.09226v1 Announce Type: new  Abstract: Electronic health records (EHR) and claims data are rich sources of real-world data that reflect patient health status and healthcare utilization. Querying these databases to answer epidemiological questions is challenging due to the intricacy of medical terminology and the need for complex SQL queries. Here, we introduce an end-to-end methodology that combines text-to-SQL generation with retrieval augmented generation (RAG) to answer epidemiological questions using EHR and claims data. We show that our approach, which integrates a medical coding step into the text-to-SQL process, significantly improves the performance over simple prompting. Our findings indicate that although current language models are not yet sufficiently accurate for unsupervised use, RAG offers a promising direction for improving their capabilities, as shown in a realistic industry setting.
    
[^5]: 一种用于多模态电视节目摘要的模块化方法

    A Modular Approach for Multimodal Summarization of TV Shows

    [https://arxiv.org/abs/2403.03823](https://arxiv.org/abs/2403.03823)

    提出了一种模块化方法用于多模态电视节目摘要，包括检测场景边界、重新排列场景、将视觉信息转换为文本、总结对话以及将场景摘要融合的过程，并引入了一个新的衡量摘要质量的评价指标PREFS。

    

    在本文中，我们讨论了电视节目摘要的任务，涉及到人工智能研究中的关键领域：复杂推理、多模态和长篇叙事。我们提出了一种模块化方法，其中各个组件执行专门的子任务，我们认为与端到端方法相比，这种方法提供了更大的灵活性。我们的模块涉及检测场景边界，重新排列场景以尽量减少不同事件之间的切换次数，将视觉信息转换为文本，总结每个场景中的对话，并将场景摘要融合成整集的最终摘要。我们还提出了一个新的度量标准，PREFS（摘要事实的精确度和召回率评估），用于衡量生成摘要的精确度和召回率，我们将其分解为原子事实。在最近发布的SummScreen3D数据集Papalampidi和Lapata（2023）上进行测试，我们的方法产生了

    arXiv:2403.03823v1 Announce Type: new  Abstract: In this paper we address the task of summarizing television shows, which touches key areas in AI research: complex reasoning, multiple modalities, and long narratives. We present a modular approach where separate components perform specialized sub-tasks which we argue affords greater flexibility compared to end-to-end methods. Our modules involve detecting scene boundaries, reordering scenes so as to minimize the number of cuts between different events, converting visual information to text, summarizing the dialogue in each scene, and fusing the scene summaries into a final summary for the entire episode. We also present a new metric, PREFS (\textbf{P}recision and \textbf{R}ecall \textbf{E}valuation of Summary \textbf{F}act\textbf{s}), to measure both precision and recall of generated summaries, which we decompose into atomic facts. Tested on the recently released SummScreen3D dataset Papalampidi and Lapata (2023), our method produces hi
    
[^6]: 大语言模型的泛化或记忆：数据污染与可信评估

    Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models

    [https://arxiv.org/abs/2402.15938](https://arxiv.org/abs/2402.15938)

    本文提出了一种通过LLMs输出分布进行污染检测的方法CDD，以及一种基于LLMs输出修正的可信评估方法TED，以应对大语言模型在数据污染和可信评估方面面临的挑战。

    

    最近关于大语言模型（LLMs）令人印象深刻能力的说法通常是通过在开放获取的基准上进行评估来支持的。考虑到LLMs的训练数据的庞大规模和广泛来源，它可能明确或隐含地包含测试数据，导致LLMs更容易受到数据污染的影响。然而，由于训练数据的不透明性、模型的黑盒访问以及合成训练数据的快速增长，对于LLMs来说检测和减轻数据污染面临着重大挑战。在本文中，我们提出了CDD，即通过LLMs输出分布进行污染检测的CDD。CDD仅需要采样文本来检测数据污染，通过识别LLMs输出分布的峰值来进行检测。为了减轻评估中数据污染的影响，我们还提出了TED：基于LLMs输出修正的可信评估。

    arXiv:2402.15938v1 Announce Type: cross  Abstract: Recent statements about the impressive capabilities of large language models (LLMs) are usually supported by evaluating on open-access benchmarks. Considering the vast size and wide-ranging sources of LLMs' training data, it could explicitly or implicitly include test data, leading to LLMs being more susceptible to data contamination. However, due to the opacity of training data, the black-box access of models, and the rapid growth of synthetic training data, detecting and mitigating data contamination for LLMs faces significant challenges. In this paper, we propose CDD, which stands for Contamination Detection via output Distribution for LLMs. CDD necessitates only the sampled texts to detect data contamination, by identifying the peakedness of LLM's output distribution. To mitigate the impact of data contamination in evaluation, we also present TED: Trustworthy Evaluation via output Distribution, based on the correction of LLM's outp
    
[^7]: GenTranslate: 大型语言模型是生成的多语言语音和机器翻译工具

    GenTranslate: Large Language Models are Generative Multilingual Speech and Machine Translators

    [https://arxiv.org/abs/2402.06894](https://arxiv.org/abs/2402.06894)

    GenTranslate是一个新的翻译任务生成模型，通过利用大型语言模型的丰富语言知识和强大推理能力，可以从N-best列表中生成更高质量的翻译结果。

    

    大型语言模型（LLMs）的最新进展通过减少表示误差和引入外部知识，推动了多语言语音和机器翻译的发展。然而，翻译任务通常使用束搜索解码和前k个假设选择进行推理。这些技术往往不能充分利用多样化的N-best假设中的丰富信息，使得它们在需要单个高质量输出序列的翻译任务中效果不佳。在本文中，我们提出了一个新的翻译任务生成模型，即“GenTranslate”，它基于LLMs来从N-best列表中生成更好的结果。利用LLMs丰富的语言知识和强大的推理能力，我们的新模型可以将N-best候选人中的丰富信息整合起来，生成更高质量的翻译结果。此外，为了支持LLM的微调，我们构建并发布了一个HypoTransla模型。

    Recent advances in large language models (LLMs) have stepped forward the development of multilingual speech and machine translation by its reduced representation errors and incorporated external knowledge. However, both translation tasks typically utilize beam search decoding and top-1 hypothesis selection for inference. These techniques struggle to fully exploit the rich information in the diverse N-best hypotheses, making them less optimal for translation tasks that require a single, high-quality output sequence. In this paper, we propose a new generative paradigm for translation tasks, namely "GenTranslate", which builds upon LLMs to generate better results from the diverse translation versions in N-best list. Leveraging the rich linguistic knowledge and strong reasoning abilities of LLMs, our new paradigm can integrate the rich information in N-best candidates to generate a higher-quality translation result. Furthermore, to support LLM finetuning, we build and release a HypoTransla
    
[^8]: 大型语言模型的盲点：超叙事语言信息

    A blind spot for large language models: Supradiegetic linguistic information

    [https://arxiv.org/abs/2306.06794](https://arxiv.org/abs/2306.06794)

    大型语言模型的盲点在于其对超叙事语言信息的忽视，研究提出考虑模型如何感知语言信息有助于深入了解其能力。

    

    像ChatGPT这样的大型语言模型(LLMs)反映了人工智能领域的深刻变革，实现了令人印象深刻甚至令人震惊的类人语言流利度。它们目前和潜在的能力范围是一个积极探讨的领域，绝非仅限于科研人员。人们通常将LLMs的训练数据框定为“文本”甚至“语言”。我们使用来自语言学、体现认知、认知科学、数学和历史等领域的思想，仔细审视这一框架的细节。我们提出，考虑像ChatGPT这样的LLM是什么感觉，正如纳格尔可能会说的那样，可以帮助我们深入了解其整体能力，特别是，其接受的语言训练数据可以被有益地重新构思为对语言中编码的叙事信息的接触，其缺陷可以被重新构思为对这些信息的无知。

    arXiv:2306.06794v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) like ChatGPT reflect profound changes in the field of Artificial Intelligence, achieving a linguistic fluency that is impressively, even shockingly, human-like. The extent of their current and potential capabilities is an active area of investigation by no means limited to scientific researchers. It is common for people to frame the training data for LLMs as "text" or even "language". We examine the details of this framing using ideas from several areas, including linguistics, embodied cognition, cognitive science, mathematics, and history. We propose that considering what it is like to be an LLM like ChatGPT, as Nagel might have put it, can help us gain insight into its capabilities in general, and in particular, that its exposure to linguistic training data can be productively reframed as exposure to the diegetic information encoded in language, and its deficits can be reframed as ignorance of ext
    
[^9]: 播风撩起风暴：编辑语言模型的影响

    Sowing the Wind, Reaping the Whirlwind: The Impact of Editing Language Models. (arXiv:2401.10647v1 [cs.CL])

    [http://arxiv.org/abs/2401.10647](http://arxiv.org/abs/2401.10647)

    本文研究了通过编辑语言模型的复杂后果，发现在增强模型准确性与保持道德完整性之间存在悖论。我们发现，尽管注入准确信息对模型的可靠性很重要，但它可能破坏模型的基本框架，导致不可预测和潜在的不安全行为。

    

    在人工智能领域中，红队测试或越狱大型语言模型（LLM）的概念已成为一个重要的研究领域。通过对模型进行编辑，揭示了这种修改的复杂后果，发现了增强模型准确性与保持其道德完整性之间的复杂关系。我们的深入分析揭示了一个令人惊讶的悖论：虽然注入准确信息对于模型的可靠性至关重要，但它却可能破坏模型的基本框架，导致不可预测和潜在的不安全行为。此外，我们提出了一个基准数据集NicheHazardQA，用于研究模型在相同和跨领域中的不安全行为。这一方面的研究揭示了编辑如何影响模型的安全度量和保护机制。

    In the rapidly advancing field of artificial intelligence, the concept of Red-Teaming or Jailbreaking large language models (LLMs) has emerged as a crucial area of study. This approach is especially significant in terms of assessing and enhancing the safety and robustness of these models. This paper investigates the intricate consequences of such modifications through model editing, uncovering a complex relationship between enhancing model accuracy and preserving its ethical integrity. Our in-depth analysis reveals a striking paradox: while injecting accurate information is crucial for model reliability, it can paradoxically destabilize the model's foundational framework, resulting in unpredictable and potentially unsafe behaviors. Additionally, we propose a benchmark dataset NicheHazardQA to investigate this unsafe behavior both within the same and cross topical domain. This aspect of our research sheds light on how the edits, impact the model's safety metrics and guardrails. Our find
    
[^10]: 什么是“好”的社交行为者？以尊重为视角评估与语言代理的交互

    What makes for a 'good' social actor? Using respect as a lens to evaluate interactions with language agents. (arXiv:2401.09082v1 [cs.CL])

    [http://arxiv.org/abs/2401.09082](http://arxiv.org/abs/2401.09082)

    本文研究以尊重为视角评估与语言代理的交互，提出了一种更加关注关系和情境因素的伦理方法，旨在帮助LLM技术表现得“好”

    

    随着基于大型语言模型（LLM）的对话代理越来越受欢迎，如何确保它们的行为道德和适当性已经引起了紧急关注。从“HHH”标准的角度来看，这主要体现在让输出更有帮助和诚实，并避免有害（有偏见、有毒或不准确）的陈述。虽然这种语义焦点对于将LLM代理视为纯粹的信息媒介是有用的，但它未能考虑到在不同社交情境中，同样的话语可能会显得更或者更少冒犯或不得体的实际因素。我们提出了一种更加关注关系和情境因素的伦理方法，探讨作为社交行为者的系统如何在交互中以尊重的方式对待个体。我们的工作预见了在情境交互层面上一系列尚未被探索的风险，并提供了实用建议，以帮助LLM技术表现得“好”

    With the growing popularity of dialogue agents based on large language models (LLMs), urgent attention has been drawn to finding ways to ensure their behaviour is ethical and appropriate. These are largely interpreted in terms of the 'HHH' criteria: making outputs more helpful and honest, and avoiding harmful (biased, toxic, or inaccurate) statements. Whilst this semantic focus is useful from the perspective of viewing LLM agents as mere mediums for information, it fails to account for pragmatic factors that can make the same utterance seem more or less offensive or tactless in different social situations. We propose an approach to ethics that is more centred on relational and situational factors, exploring what it means for a system, as a social actor, to treat an individual respectfully in a (series of) interaction(s). Our work anticipates a set of largely unexplored risks at the level of situated interaction, and offers practical suggestions to help LLM technologies behave as 'good'
    
[^11]: 大型语言模型增强的算法选择：朝着全面算法表示的方向

    Large Language Model-Enhanced Algorithm Selection: Towards Comprehensive Algorithm Representation. (arXiv:2311.13184v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.13184](http://arxiv.org/abs/2311.13184)

    本论文提出了一种方法，通过将算法表示集成到算法选择中，从而填补了当前算法选择技术对算法特征的研究空白。

    

    算法选择旨在在执行之前识别解决特定问题的最合适算法，已成为自动机器学习中的关键过程。当前主流的算法选择技术主要依赖于各种问题的特征表示，并使用每个算法的性能作为监督信息。然而，目前对算法特征的考虑存在重要的研究空白。这主要归因于算法的固有复杂性，使得在不同种类的算法中找到一种普适有效的特征提取方法特别具有挑战性。不幸的是，忽视了这一方面无疑会影响算法选择的准确性，并间接需要增加训练数据的数量。本文提出了一种方法来解决这一空白，即将算法表示集成到算法选择中。

    Algorithm selection aims to identify the most suitable algorithm for solving a specific problem before execution, which has become a critical process of the AutoML. Current mainstream algorithm selection techniques rely heavily on feature representations of various problems and employ the performance of each algorithm as supervised information. However, there is a significant research gap concerning the consideration of algorithm features. This gap is primarily attributed to the inherent complexity of algorithms, making it particularly challenging to find a universally effective feature extraction method that is applicable across a diverse range of algorithms. Unfortunately, neglecting this aspect undoubtedly impacts the accuracy of algorithm selection and indirectly necessitates an increased volume of problem data for training purposes. This paper takes a significant stride towards addressing this gap by proposing an approach that integrates algorithm representation into the algorithm
    
[^12]: MagicBrush: 人工标注的用于指导图像编辑的数据集

    MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing. (arXiv:2306.10012v1 [cs.CV])

    [http://arxiv.org/abs/2306.10012](http://arxiv.org/abs/2306.10012)

    MagicBrush是第一个大规模的手动标注的数据集，用于指导真实图像的编辑。它包括超过10K个手动标注的三元组，支持大规模的文本指导图像编辑模型训练。在此数据集上微调InstructPix2Pix可以根据人类评估提供更好的图像。

    

    文本指导的图像编辑从个人使用到专业应用（如Photoshop）广泛需要。然而，现有的方法要么是零样本，要么是在自动合成的数据集上进行训练，其中含有大量的噪声。因此，它们在实践中仍需要大量的手动调整才能产生理想的结果。为了解决这个问题，我们介绍了MagicBrush，第一个大规模的手动标注的数据集，用于指导真实图像的编辑，包括单个操作、多个操作、提供掩码和不提供掩码等不同场景。MagicBrush包括超过10K个手动标注的三元组（源图像，指令，目标图像），支持大规模的文本指导图像编辑模型训练。我们在MagicBrush上微调InstructPix2Pix，并展示了新模型可以根据人类评估提供更好的图像。我们还进行了广泛的实验评估，以评估模型的泛化能力和使用效果。

    Text-guided image editing is widely needed in daily life, ranging from personal use to professional applications such as Photoshop. However, existing methods are either zero-shot or trained on an automatically synthesized dataset, which contains a high volume of noise. Thus, they still require lots of manual tuning to produce desirable outcomes in practice. To address this issue, we introduce MagicBrush (https://osu-nlp-group.github.io/MagicBrush/), the first large-scale, manually annotated dataset for instruction-guided real image editing that covers diverse scenarios: single-turn, multi-turn, mask-provided, and mask-free editing. MagicBrush comprises over 10K manually annotated triples (source image, instruction, target image), which supports trainining large-scale text-guided image editing models. We fine-tune InstructPix2Pix on MagicBrush and show that the new model can produce much better images according to human evaluation. We further conduct extensive experiments to evaluate cu
    
[^13]: 使用基于认识不确定性的数据选择来适应预训练的ASR模型以应对低资源临床语音问题

    Adapting Pretrained ASR Models to Low-resource Clinical Speech using Epistemic Uncertainty-based Data Selection. (arXiv:2306.02105v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.02105](http://arxiv.org/abs/2306.02105)

    本研究使用基于认识不确定性的数据选择方法来减少非洲口音临床ASR训练的注释成本。结果表明，这种方法可以超过现有的基准结果，并提高低资源口音的泛化能力。

    

    尽管ASR取得了显著进展，但由于缺乏训练数据集，对于非洲口音的临床ASR的研究还不够。在这一领域构建强大的ASR系统需要大量的标注数据，用于各种语言和形态丰富的口音，但这些数据的创建成本较高。本研究旨在通过基于信息不确定性的数据选择来减少注释费用。我们表明，将认识不确定性纳入我们的自适应过程中，可以超过使用最先进（SOTA）ASR模型建立的几个基准结果，同时减少所需的标记数据量，从而降低注释成本。我们的方法还改善了低资源口音的超出分布泛化能力，展示了我们的方法在非洲临床ASR的背景下构建泛化型ASR模型的可行性，而在这种情况下，训练数据集主要是稀缺的。

    While there has been significant progress in ASR, African-accented clinical ASR has been understudied due to a lack of training datasets. Building robust ASR systems in this domain requires large amounts of annotated or labeled data, for a wide variety of linguistically and morphologically rich accents, which are expensive to create. Our study aims to address this problem by reducing annotation expenses through informative uncertainty-based data selection. We show that incorporating epistemic uncertainty into our adaptation rounds outperforms several baseline results, established using state-of-the-art (SOTA) ASR models, while reducing the required amount of labeled data, and hence reducing annotation costs. Our approach also improves out-of-distribution generalization for very low-resource accents, demonstrating the viability of our approach for building generalizable ASR models in the context of accented African clinical ASR, where training datasets are predominantly scarce.
    
[^14]: 逃离机器翻译中句子级范式的限制

    Escaping the sentence-level paradigm in machine translation. (arXiv:2304.12959v1 [cs.CL])

    [http://arxiv.org/abs/2304.12959](http://arxiv.org/abs/2304.12959)

    本文提出了一种摆脱机器翻译中句子级范式限制的方法，通过处理三个障碍来实现：使用足够大的标准Transformer架构、引入一种简单而有效的技术来将文档级信息转化为适合训练的形式、基于自动文档分类的评估协议来有效地识别文档级翻译质量。在两个非常不同的文档级翻译任务上，我们的实验表明，在此数据上训练的Transformer模型明显优于强大的基线模型。

    

    众所周知，文档语境对于解决一系列翻译模糊性至关重要，事实上，文档设置几乎是所有翻译的自然设置。然而，机器翻译（包括研究和生产）在几十年前的句子级翻译范式中仍然停滞不前，这是一个越来越明显的问题，由于来自大型语言模型的竞争压力，这些模型天生就是基于文档的。本文提出了一种摆脱这种困境的方法，同时解决了三个障碍：我们应该使用什么架构？我们从哪里获取训练它们的文档级信息？以及我们如何知道它们是否足够好？

    It is well-known that document context is vital for resolving a range of translation ambiguities, and in fact the document setting is the most natural setting for nearly all translation. It is therefore unfortunate that machine translation -- both research and production -- largely remains stuck in a decades-old sentence-level translation paradigm. It is also an increasingly glaring problem in light of competitive pressure from large language models, which are natively document-based. Much work in document-context machine translation exists, but for various reasons has been unable to catch hold. This paper suggests a path out of this rut by addressing three impediments at once: what architectures should we use? where do we get document-level information for training them? and how do we know whether they are any good? In contrast to work on specialized architectures, we show that the standard Transformer architecture is sufficient, provided it has enough capacity. Next, we address the t
    

