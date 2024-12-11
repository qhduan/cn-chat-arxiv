# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SciNews: From Scholarly Complexities to Public Narratives -- A Dataset for Scientific News Report Generation](https://arxiv.org/abs/2403.17768) | 科学新闻报道生成的自动化提高了学术见解的可访问性，该研究提出了一个包含学术出版物和相应科学新闻报道的数据集，用于探索自动生成科学新闻报道的可能性。 |
| [^2] | [Benchmarking Chinese Commonsense Reasoning of LLMs: From Chinese-Specifics to Reasoning-Memorization Correlations](https://arxiv.org/abs/2403.14112) | CHARM是第一个用于全面深入评估大型语言模型中文常识推理能力的基准，研究发现LLM的语言导向性和任务领域会影响提示策略的有效性，并指出一些LLMs在记忆中文常识方面存在困难，而其他一些LLMs在推理上表现存在差异。 |
| [^3] | [Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision](https://arxiv.org/abs/2403.09472) | 通过从更简单的任务学习，实现对更难推理任务的有效泛化，提出了一种可扩展对齐方法。 |
| [^4] | [A Benchmark of Domain-Adapted Large Language Models for Generating Brief Hospital Course Summaries](https://arxiv.org/abs/2403.05720) | 介绍了一个新的基准测试，评估了用于生成简要住院病程摘要的大语言模型在健康保健领域中的性能并提出相应的自适应策略 |
| [^5] | [Is Cognition and Action Consistent or Not: Investigating Large Language Model's Personality](https://arxiv.org/abs/2402.14679) | 通过评估大型语言模型在表达人类个性特征方面的可靠性，研究认知与行为之间的一致性，以及提出对观察结果的心理理论和指标假设 |
| [^6] | [What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement](https://arxiv.org/abs/2402.01865) | 本文研究了语言模型更新中的遗忘现象，提出了一种预测上游实例遗忘的方法，以改进重播过程的可控性和解释性。根据预训练实例的预-softmax对数几率分数变化与在线学习实例的相似性，提出了一种部分可解释的预测模型，在BART模型上表现良好但在T5模型上失败。此外，还展示了基于内积的黑盒分类器。 |
| [^7] | [BPDec: Unveiling the Potential of Masked Language Modeling Decoder in BERT pretraining](https://arxiv.org/abs/2401.15861) | 本文揭示了BPDec（BERT预训练解码器）的潜力，强调增强的掩码语言建模解码器设计及研究在BERT预训练中的重要性。 |
| [^8] | [Will Sentiment Analysis Need Subculture? A New Data Augmentation Approach.](http://arxiv.org/abs/2309.00178) | 本文提出了一种新的数据增强方法SCDA，通过利用亚文化表达生成器为每个训练文本生成六个增强文本，以解决情感分析中面临的训练数据不足问题。实验证明了SCDA的有效性和潜力。 |
| [^9] | [ChatGPT vs Human-authored Text: Insights into Controllable Text Summarization and Sentence Style Transfer.](http://arxiv.org/abs/2306.07799) | 本文旨在系统地检查ChatGPT在两个可控生成任务中的表现，即ChatGPT能否适应不同的目标受众和写作风格。研究发现，人类产生的文体变化比ChatGPT表现出的更大，而生成的文本在一些特征上与人类样本有所不同，有时会包含事实错误或幻觉。 |
| [^10] | [Incorporating Distributions of Discourse Structure for Long Document Abstractive Summarization.](http://arxiv.org/abs/2305.16784) | 本文提出了一种名为'RSTformer'的摘要模型，该模型全面融合了话语关系类型和不确定性，并以修辞结构理论为基础，经过严格评估，表现明显优于现有的模型。 |

# 详细

[^1]: 从学术复杂性到公众叙事：科学新闻报道生成的数据集

    SciNews: From Scholarly Complexities to Public Narratives -- A Dataset for Scientific News Report Generation

    [https://arxiv.org/abs/2403.17768](https://arxiv.org/abs/2403.17768)

    科学新闻报道生成的自动化提高了学术见解的可访问性，该研究提出了一个包含学术出版物和相应科学新闻报道的数据集，用于探索自动生成科学新闻报道的可能性。

    

    科学新闻报道作为一个桥梁，巧妙地将复杂的研究文章翻译成与更广泛的公众 resonant 的报道。这种叙事的自动生成增强了学术见解的可访问性。在本文中，我们提出了一个新的语料库来促进这种范式的发展。我们的语料库包括九个学科领域中学术出版物及其相应科学新闻报道的平行编译。为了证明我们数据集的实用性和可靠性，我们进行了广泛分析，突出了科学新闻叙事和学术文稿之间的可读性和简洁性差异。我们使用最先进的文本生成模型基准测试我们的数据集。评估过程包括自动评估和人工评估，为未来探索自动生成科学新闻报道打下了基础。

    arXiv:2403.17768v1 Announce Type: cross  Abstract: Scientific news reports serve as a bridge, adeptly translating complex research articles into reports that resonate with the broader public. The automated generation of such narratives enhances the accessibility of scholarly insights. In this paper, we present a new corpus to facilitate this paradigm development. Our corpus comprises a parallel compilation of academic publications and their corresponding scientific news reports across nine disciplines. To demonstrate the utility and reliability of our dataset, we conduct an extensive analysis, highlighting the divergences in readability and brevity between scientific news narratives and academic manuscripts. We benchmark our dataset employing state-of-the-art text generation models. The evaluation process involves both automatic and human evaluation, which lays the groundwork for future explorations into the automated generation of scientific news reports. The dataset and code related 
    
[^2]: 评估大型语言模型中文常识推理能力：从中文特定到推理-记忆关联

    Benchmarking Chinese Commonsense Reasoning of LLMs: From Chinese-Specifics to Reasoning-Memorization Correlations

    [https://arxiv.org/abs/2403.14112](https://arxiv.org/abs/2403.14112)

    CHARM是第一个用于全面深入评估大型语言模型中文常识推理能力的基准，研究发现LLM的语言导向性和任务领域会影响提示策略的有效性，并指出一些LLMs在记忆中文常识方面存在困难，而其他一些LLMs在推理上表现存在差异。

    

    我们介绍了CHARM，这是第一个用于全面深入评估大型语言模型（LLMs）中文常识推理能力的基准，涵盖了全球已知和中文特有的常识。在CHARM上评估了7个英文和12个中文定向LLMs，采用了5种代表性提示策略来提高LLMs的推理能力，比如思维链。我们的研究结果表明，LLM的语言导向性和任务领域影响了提示策略的有效性，这丰富了以往的研究结果。我们构建了紧密关联的推理和记忆任务，并发现一些LLMs在记忆中文常识方面存在困难，影响了它们的推理能力，而其他一些LLMs在推理上表现存在差异，尽管记忆表现相似。我们还评估了LLMs的与记忆无关的推理能力，并分析了典型错误。

    arXiv:2403.14112v1 Announce Type: new  Abstract: We introduce CHARM, the first benchmark for comprehensively and in-depth evaluating the commonsense reasoning ability of large language models (LLMs) in Chinese, which covers both globally known and Chinese-specific commonsense. We evaluated 7 English and 12 Chinese-oriented LLMs on CHARM, employing 5 representative prompt strategies for improving LLMs' reasoning ability, such as Chain-of-Thought. Our findings indicate that the LLM's language orientation and the task's domain influence the effectiveness of the prompt strategy, which enriches previous research findings. We built closely-interconnected reasoning and memorization tasks, and found that some LLMs struggle with memorizing Chinese commonsense, affecting their reasoning ability, while others show differences in reasoning despite similar memorization performance. We also evaluated the LLMs' memorization-independent reasoning abilities and analyzed the typical errors. Our study pr
    
[^3]: 易于难的泛化：超越人类监督的可扩展对齐

    Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision

    [https://arxiv.org/abs/2403.09472](https://arxiv.org/abs/2403.09472)

    通过从更简单的任务学习，实现对更难推理任务的有效泛化，提出了一种可扩展对齐方法。

    

    当前人工智能对齐方法依赖于人类提供的演示或判断，由于这种方法，AI系统学习到的能力将受到人类能力的上界限制。这就带来了一个具有挑战性的研究问题：当系统的能力超过人类水平时，我们如何继续改进这些系统？本文在解决难度推理任务（如4-5级数学问题）的背景下回答了这个问题，通过从更简单的任务（如1-3级数学问题）中学习人类注释，我们将其称为“易于难的泛化”。我们的关键观点是，一个在更简单任务的监督下训练的评估器（奖励模型）可以有效地用于评分更难任务的候选解决方案，从而促进在不同难度任务间的易于难的泛化。基于这一观点，我们提出了一种新的可扩展对齐方法，首先训练处理督导

    arXiv:2403.09472v1 Announce Type: cross  Abstract: Current AI alignment methodologies rely on human-provided demonstrations or judgments, and the learned capabilities of AI systems would be upper-bounded by human capabilities as a result. This raises a challenging research question: How can we keep improving the systems when their capabilities have surpassed the levels of humans? This paper answers this question in the context of tackling hard reasoning tasks (e.g., level 4-5 MATH problems) via learning from human annotations on easier tasks (e.g., level 1-3 MATH problems), which we term as \textit{easy-to-hard generalization}. Our key insight is that an evaluator (reward model) trained on supervisions for easier tasks can be effectively used for scoring candidate solutions of harder tasks and hence facilitating easy-to-hard generalization over different levels of tasks. Based on this insight, we propose a novel approach to scalable alignment, which firstly trains the process-supervise
    
[^4]: 用于生成简要住院病程摘要的领域自适应大语言模型的基准测试

    A Benchmark of Domain-Adapted Large Language Models for Generating Brief Hospital Course Summaries

    [https://arxiv.org/abs/2403.05720](https://arxiv.org/abs/2403.05720)

    介绍了一个新的基准测试，评估了用于生成简要住院病程摘要的大语言模型在健康保健领域中的性能并提出相应的自适应策略

    

    简要住院病程（BHC）摘要是通过总结临床记录而生成的常见临床文件。虽然大型语言模型（LLMs）在自动化实际任务方面展现出显著能力，但它们在医疗应用（如BHC合成）中的能力尚未得到展示。为了使LLMs能够适应BHC合成，我们引入了一个新颖的基准测试，其中包含从MIMIC-IV记录中提取的经过预处理的数据集，封装了临床记录和简要住院病程（BHC）对。我们评估了两个通用LLMs和三个医疗领域适应的LLMs的性能，以改进从临床记录生成BHC。我们使用临床记录作为输入来生成BHC，采用基于提示的（使用上下文学习）和基于微调的自适应策略来应用于三个开源LLMs（Clinical-T5-Large，Llama2-13B，FLAN-UL2）和两个专有LLMs（GPT-3.5，GPT-4）。我们定量评估了性能。

    arXiv:2403.05720v1 Announce Type: cross  Abstract: Brief hospital course (BHC) summaries are common clinical documents generated by summarizing clinical notes. While large language models (LLMs) depict remarkable capabilities in automating real-world tasks, their capabilities for healthcare applications such as BHC synthesis have not been shown. To enable the adaptation of LLMs for BHC synthesis, we introduce a novel benchmark consisting of a pre-processed dataset extracted from MIMIC-IV notes, encapsulating clinical note, and brief hospital course (BHC) pairs. We assess the performance of two general-purpose LLMs and three healthcare-adapted LLMs to improve BHC synthesis from clinical notes. Using clinical notes as input for generating BHCs, we apply prompting-based (using in-context learning) and fine-tuning-based adaptation strategies to three open-source LLMs (Clinical-T5-Large, Llama2-13B, FLAN-UL2) and two proprietary LLMs (GPT-3.5, GPT-4). We quantitatively evaluate the performa
    
[^5]: 认知与行为一致还是不一致：研究大型语言模型的个性

    Is Cognition and Action Consistent or Not: Investigating Large Language Model's Personality

    [https://arxiv.org/abs/2402.14679](https://arxiv.org/abs/2402.14679)

    通过评估大型语言模型在表达人类个性特征方面的可靠性，研究认知与行为之间的一致性，以及提出对观察结果的心理理论和指标假设

    

    在这项研究中，我们通过回答人格问卷调查来探讨大型语言模型（LLMs）在表达类人个性特征方面的可靠性。我们的目标是评估LLMs所表达的个性倾向与它们实际“行为”之间的一致性，检验这些模型能够模拟类人个性模式的程度。通过全面分析LLM输出与已建立的人类基准之间的对比，我们试图了解LLMs中认知与行为之间的差异，并根据心理理论和指标对观察结果提出假设。

    arXiv:2402.14679v1 Announce Type: new  Abstract: In this study, we investigate the reliability of Large Language Models (LLMs) in professing human-like personality traits through responses to personality questionnaires. Our goal is to evaluate the consistency between LLMs' professed personality inclinations and their actual "behavior", examining the extent to which these models can emulate human-like personality patterns. Through a comprehensive analysis of LLM outputs against established human benchmarks, we seek to understand the cognition-action divergence in LLMs and propose hypotheses for the observed results based on psychological theories and metrics.
    
[^6]: 我的模型会忘记什么？语言模型改进中的被遗忘实例预测

    What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement

    [https://arxiv.org/abs/2402.01865](https://arxiv.org/abs/2402.01865)

    本文研究了语言模型更新中的遗忘现象，提出了一种预测上游实例遗忘的方法，以改进重播过程的可控性和解释性。根据预训练实例的预-softmax对数几率分数变化与在线学习实例的相似性，提出了一种部分可解释的预测模型，在BART模型上表现良好但在T5模型上失败。此外，还展示了基于内积的黑盒分类器。

    

    在实际应用中，语言模型会出现错误。然而，仅仅通过将模型更新为纠正错误实例，会导致灾难性的遗忘，更新后的模型在指导微调或上游训练阶段中学到的实例上出现错误。随机重播上游数据的效果不令人满意，往往伴随着较高的方差和较差的可控性。为了改善重播过程的可控性和解释性，我们试图预测由于模型更新而遗忘的上游实例。我们根据一组在线学习的实例和相应被遗忘的上游预训练实例训练预测模型。我们提出了一种部分可解释的预测模型，该模型基于这样的观察结果：预训练实例的预-softmax对数几率分数的变化类似于在线学习实例的变化，这在BART模型上表现出不错的效果，但在T5模型上失败。我们进一步展示了基于内积的黑盒分类器

    Language models deployed in the wild make errors. However, simply updating the model with the corrected error instances causes catastrophic forgetting -- the updated model makes errors on instances learned during the instruction tuning or upstream training phase. Randomly replaying upstream data yields unsatisfactory performance and often comes with high variance and poor controllability. To this end, we try to forecast upstream examples that will be forgotten due to a model update for improved controllability of the replay process and interpretability. We train forecasting models given a collection of online learned examples and corresponding forgotten upstream pre-training examples. We propose a partially interpretable forecasting model based on the observation that changes in pre-softmax logit scores of pretraining examples resemble that of online learned examples, which performs decently on BART but fails on T5 models. We further show a black-box classifier based on inner products 
    
[^7]: BPDec: 揭示BERT预训练中掩码语言建模解码器的潜力

    BPDec: Unveiling the Potential of Masked Language Modeling Decoder in BERT pretraining

    [https://arxiv.org/abs/2401.15861](https://arxiv.org/abs/2401.15861)

    本文揭示了BPDec（BERT预训练解码器）的潜力，强调增强的掩码语言建模解码器设计及研究在BERT预训练中的重要性。

    

    BERT（来自Transformer的双向编码表示）通过其在许多任务上出色的性能彻底改变了自然语言处理领域。然而，大多数研究人员主要集中在与模型结构相关的增强，例如相对位置嵌入和更有效的注意机制。还有一些人深入研究了与掩码语言建模相关的预训练技巧，包括整词掩码。DeBERTa引入了一种针对BERT编码器模型进行预训练的增强解码器，证明效果非常显著。我们认为围绕增强掩码语言建模解码器的设计和研究并未得到应有的重视。在本文中，我们提出了几种增强解码器的设计，并介绍了BPDec（BERT预训练解码器），这是一种用于建模训练的新方法。通常，预训练的BERT模型会针对特定的自然语

    arXiv:2401.15861v2 Announce Type: replace-cross  Abstract: BERT (Bidirectional Encoder Representations from Transformers) has revolutionized the field of natural language processing through its exceptional performance on numerous tasks. Yet, the majority of researchers have mainly concentrated on enhancements related to the model structure, such as relative position embedding and more efficient attention mechanisms. Others have delved into pretraining tricks associated with Masked Language Modeling, including whole word masking. DeBERTa introduced an enhanced decoder adapted for BERT's encoder model for pretraining, proving to be highly effective. We argue that the design and research around enhanced masked language modeling decoders have been underappreciated. In this paper, we propose several designs of enhanced decoders and introduce BPDec (BERT Pretraining Decoder), a novel method for modeling training. Typically, a pretrained BERT model is fine-tuned for specific Natural Language 
    
[^8]: 情感分析是否需要亚文化？一种新的数据增强方法

    Will Sentiment Analysis Need Subculture? A New Data Augmentation Approach. (arXiv:2309.00178v1 [cs.CL])

    [http://arxiv.org/abs/2309.00178](http://arxiv.org/abs/2309.00178)

    本文提出了一种新的数据增强方法SCDA，通过利用亚文化表达生成器为每个训练文本生成六个增强文本，以解决情感分析中面临的训练数据不足问题。实验证明了SCDA的有效性和潜力。

    

    著名谚语“笔能胜过剑”强调了文字表达在塑造情感方面所具有的强大影响力。事实上，精心打造的文字可以在文化中产生深远共鸣，传达深刻的情感。如今，互联网的普及促成了围绕当代社会环境聚集的亚文化。亚文化通过热衷追求新奇来巧妙地表达人类情感的复杂性，这在情感分析中是不可忽视的事实。本文旨在通过亚文化的视角丰富数据，以解决情感分析面临的训练数据不足问题。为此，提出了一种基于亚文化的数据增强（SCDA）新方法，通过创建六种不同亚文化表达生成器，为每个训练文本生成六个增强文本。大量实验证实了SCDA的有效性和潜力。结果还揭示了该方法对提高情感分析性能的启示。

    The renowned proverb that "The pen is mightier than the sword" underscores the formidable influence wielded by text expressions in shaping sentiments. Indeed, well-crafted written can deeply resonate within cultures, conveying profound sentiments. Nowadays, the omnipresence of the Internet has fostered a subculture that congregates around the contemporary milieu. The subculture artfully articulates the intricacies of human feelings by ardently pursuing the allure of novelty, a fact that cannot be disregarded in the sentiment analysis. This paper strives to enrich data through the lens of subculture, to address the insufficient training data faced by sentiment analysis. To this end, a new approach of subculture-based data augmentation (SCDA) is proposed, which engenders six enhanced texts for each training text by leveraging the creation of six diverse subculture expression generators. The extensive experiments attest to the effectiveness and potential of SCDA. The results also shed lig
    
[^9]: ChatGPT与人工撰写文本：可控文本摘要和句子风格转移的洞察

    ChatGPT vs Human-authored Text: Insights into Controllable Text Summarization and Sentence Style Transfer. (arXiv:2306.07799v1 [cs.CL])

    [http://arxiv.org/abs/2306.07799](http://arxiv.org/abs/2306.07799)

    本文旨在系统地检查ChatGPT在两个可控生成任务中的表现，即ChatGPT能否适应不同的目标受众和写作风格。研究发现，人类产生的文体变化比ChatGPT表现出的更大，而生成的文本在一些特征上与人类样本有所不同，有时会包含事实错误或幻觉。

    

    大规模语言模型（如ChatGPT）以其出色的能力从简短的自然语言提示生成连贯的文本引起了媒体的重视。本文旨在系统地检查ChatGPT在两个可控生成任务中的表现，即ChatGPT能否适应不同的目标受众（专家与一般人）和写作风格（正式与非正式）。此外，我们评估了生成文本的忠实度，并将模型的表现与人工撰写的文本进行了比较。我们的研究发现，人类产生的文体变化比ChatGPT表现出的更大，而生成的文本在诸如单词类型分布等几个特征上与人类样本有所不同。此外，我们发现当 ChatGPT 将文本适应特定风格时，有时会包含事实错误或幻觉。

    Large-scale language models, like ChatGPT, have garnered significant media attention and stunned the public with their remarkable capacity for generating coherent text from short natural language prompts. In this paper, we aim to conduct a systematic inspection of ChatGPT's performance in two controllable generation tasks, with respect to ChatGPT's ability to adapt its output to different target audiences (expert vs. layman) and writing styles (formal vs. informal). Additionally, we evaluate the faithfulness of the generated text, and compare the model's performance with human-authored texts. Our findings indicate that the stylistic variations produced by humans are considerably larger than those demonstrated by ChatGPT, and the generated texts diverge from human samples in several characteristics, such as the distribution of word types. Moreover, we observe that ChatGPT sometimes incorporates factual errors or hallucinations when adapting the text to suit a specific style.
    
[^10]: 结合话语结构分布的长文本自动摘要方法

    Incorporating Distributions of Discourse Structure for Long Document Abstractive Summarization. (arXiv:2305.16784v1 [cs.CL])

    [http://arxiv.org/abs/2305.16784](http://arxiv.org/abs/2305.16784)

    本文提出了一种名为'RSTformer'的摘要模型，该模型全面融合了话语关系类型和不确定性，并以修辞结构理论为基础，经过严格评估，表现明显优于现有的模型。

    

    对于文本摘要，话语结构在辨识文本核心内容方面起着关键作用。可惜的是，之前将修辞结构理论（RST）引入基于transformer的自动摘要模型的研究仅考虑了核心部分的注释，从而忽略了各种不同类型的话语关系。本文提出了一种名为'RSTformer'的新型摘要模型，该模型全面融合了话语关系类型和不确定性。我们的RST-attention机制是基于文档级修辞结构的Longformer框架的扩展。经过严格评估，本文提出的模型表现明显优于现有的模型，凸显出其在多个自动评估指标和人工评估上的卓越表现。

    For text summarization, the role of discourse structure is pivotal in discerning the core content of a text. Regrettably, prior studies on incorporating Rhetorical Structure Theory (RST) into transformer-based summarization models only consider the nuclearity annotation, thereby overlooking the variety of discourse relation types. This paper introduces the 'RSTformer', a novel summarization model that comprehensively incorporates both the types and uncertainty of rhetorical relations. Our RST-attention mechanism, rooted in document-level rhetorical structure, is an extension of the recently devised Longformer framework. Through rigorous evaluation, the model proposed herein exhibits significant superiority over state-of-the-art models, as evidenced by its notable performance on several automatic metrics and human evaluation.
    

