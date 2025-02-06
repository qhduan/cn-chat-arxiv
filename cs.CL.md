# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decoding Speculative Decoding](https://rss.arxiv.org/abs/2402.01528) | 推测解码是一种用于加速大型语言模型推断的技术，但我们的实验表明，选择的草稿模型生成的令牌被目标模型接受的概率越高，吞吐量越低。我们通过大量实验，分析了各种因素对推测解码效果的影响，并提出了一个分析模型来提高效率。 |
| [^2] | [Ukrainian Texts Classification: Exploration of Cross-lingual Knowledge Transfer Approaches](https://arxiv.org/abs/2404.02043) | 乌克兰文本分类领域探索跨语言知识传递方法，利用最新的NLP技术，测试了在毒性分类、文体分类和自然语言推理任务上的最佳设置。 |
| [^3] | [Lost in Overlap: Exploring Watermark Collision in LLMs](https://arxiv.org/abs/2403.10020) | 本研究探讨了在大型语言模型中关于水印冲突的问题，发现双水印冲突存在时会对水印算法的检测性能造成威胁。 |
| [^4] | [TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification with Minimal Supervision](https://arxiv.org/abs/2403.00165) | 本文提出了一种最小监督的分层文本分类方法，利用每个节点的唯一类名作为唯一监督，同时结合大型语言模型（LLM）提高分类性能。 |
| [^5] | [SymBa: Symbolic Backward Chaining for Multi-step Natural Language Reasoning](https://arxiv.org/abs/2402.12806) | SymBa提出了一种符号化向后推理方法，在多步自然语言推理中取得了显著的性能和效率提升，能够生成可解释的结构化证明。 |
| [^6] | [What Makes for Good Visual Instructions? Synthesizing Complex Visual Reasoning Instructions for Visual Instruction Tuning.](http://arxiv.org/abs/2311.01487) | 通过综合复杂的视觉推理任务，可以有效改善多模式大型语言模型在评估基准上的性能。我们提出了一种自动创建高质量复杂视觉推理指令的系统方法。 |
| [^7] | [Certifying LLM Safety against Adversarial Prompting.](http://arxiv.org/abs/2309.02705) | 本研究提出了首个具有可验证安全保证的框架——消除和检查，用于对抗敌对提示。通过逐个消除标记并使用安全过滤器检查生成的子序列，确保任何敌对修改的有害输入提示都能被正确标识为有害。 |

# 详细

[^1]: 解码推测解码

    Decoding Speculative Decoding

    [https://rss.arxiv.org/abs/2402.01528](https://rss.arxiv.org/abs/2402.01528)

    推测解码是一种用于加速大型语言模型推断的技术，但我们的实验表明，选择的草稿模型生成的令牌被目标模型接受的概率越高，吞吐量越低。我们通过大量实验，分析了各种因素对推测解码效果的影响，并提出了一个分析模型来提高效率。

    

    推测解码是一种常用的技术，用于加速大型语言模型（LLM）的推断，而不修改其结果。在对LLM进行推断时，推测解码使用较小的草稿模型生成推测令牌，然后使用目标LLM验证这些草稿令牌。推测解码提供的加速取决于草稿模型的选择。普遍建议选择一个草稿模型，该模型生成的令牌被LLM接受的概率很高，以实现最高吞吐量。然而，我们的实验结果与之相反，随着生成的令牌被目标模型接受的概率增加，吞吐量减少。为了理解这一现象，我们进行了大量实验，对影响推测解码的不同因素进行了表征，并研究了这些因素如何相互作用和影响加速效果。基于我们的实验结果，我们描述了一个分析模型，可以使用该模型来进行决策，提高推测解码的效率。

    Speculative Decoding is a widely used technique to speed up inference for Large Language Models (LLMs) without modifying its outcome. When performing inference on an LLM, speculative decoding uses a smaller draft model which generates speculative tokens and then uses the target LLM to verify those draft tokens. The speedup provided by speculative decoding heavily depends on the choice of the draft model. It has been widely suggested to select a draft model that provides a high probability of the generated token being accepted by the LLM to achieve the highest throughput. However, our experiments indicate the contrary with throughput diminishing as the probability of generated tokens to be accepted by the target model increases. To understand this phenomenon, we perform extensive experiments to characterize the different factors that affect speculative decoding and how those factors interact and affect the speedups. Based on our experiments we describe an analytical model which can be u
    
[^2]: 乌克兰文本分类：跨语言知识传递方法的探索

    Ukrainian Texts Classification: Exploration of Cross-lingual Knowledge Transfer Approaches

    [https://arxiv.org/abs/2404.02043](https://arxiv.org/abs/2404.02043)

    乌克兰文本分类领域探索跨语言知识传递方法，利用最新的NLP技术，测试了在毒性分类、文体分类和自然语言推理任务上的最佳设置。

    

    虽然在自然语言处理文本分类领域存在大量标记数据集，但各种语言可用数据的不平衡问题依然显而易见。乌克兰语作为一种仍可从跨语言方法的持续完善中受益的语言。鉴于我们所了解，针对典型文本分类任务，乌克兰语语料库极度匮乏。在这项工作中，我们利用自然语言处理领域的最新进展，探索跨语言知识传递方法，避免手动数据整理：大型多语言编码器和翻译系统、LLMs，以及语言适配器。我们在三个文本分类任务上测试这些方法--毒性分类、文体分类和自然语言推理--提供了最佳设置的"配方"。

    arXiv:2404.02043v1 Announce Type: cross  Abstract: Despite the extensive amount of labeled datasets in the NLP text classification field, the persistent imbalance in data availability across various languages remains evident. Ukrainian, in particular, stands as a language that still can benefit from the continued refinement of cross-lingual methodologies. Due to our knowledge, there is a tremendous lack of Ukrainian corpora for typical text classification tasks. In this work, we leverage the state-of-the-art advances in NLP, exploring cross-lingual knowledge transfer methods avoiding manual data curation: large multilingual encoders and translation systems, LLMs, and language adapters. We test the approaches on three text classification tasks -- toxicity classification, formality classification, and natural language inference -- providing the "recipe" for the optimal setups.
    
[^3]: 在重叠中迷失：探索LLMs中的水印冲突

    Lost in Overlap: Exploring Watermark Collision in LLMs

    [https://arxiv.org/abs/2403.10020](https://arxiv.org/abs/2403.10020)

    本研究探讨了在大型语言模型中关于水印冲突的问题，发现双水印冲突存在时会对水印算法的检测性能造成威胁。

    

    由于大型语言模型（LLMs）在生成内容方面的普及，引发了关于文本版权的担忧。水印方法，特别是基于logit的方法，将不可察觉的标识嵌入文本中，以解决这些挑战。然而，水印方法在不同LLMs上的广泛应用导致了一种不可避免的问题，即在常见任务（如问答和改写）中发生的水印冲突。本研究关注双水印冲突，即同一文本中同时存在两个水印的情况。研究表明，水印冲突对上游和下游水印算法的检测器的检测性能构成威胁。

    arXiv:2403.10020v1 Announce Type: new  Abstract: The proliferation of large language models (LLMs) in generating content raises concerns about text copyright. Watermarking methods, particularly logit-based approaches, embed imperceptible identifiers into text to address these challenges. However, the widespread use of watermarking across diverse LLMs has led to an inevitable issue known as watermark collision during common tasks like question answering and paraphrasing. This study focuses on dual watermark collisions, where two watermarks are present simultaneously in the same text. The research demonstrates that watermark collision poses a threat to detection performance for detectors of both upstream and downstream watermark algorithms.
    
[^4]: TELEClass: 税务学丰富和LLM增强的最小监督分层文本分类

    TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification with Minimal Supervision

    [https://arxiv.org/abs/2403.00165](https://arxiv.org/abs/2403.00165)

    本文提出了一种最小监督的分层文本分类方法，利用每个节点的唯一类名作为唯一监督，同时结合大型语言模型（LLM）提高分类性能。

    

    分层文本分类旨在将每个文档分类为标签Taxonomy中的一组类别。本文旨在研究使用最少监督：仅使用每个节点的唯一类名作为监督来进行分层文本分类。最近，大型语言模型（LLM）通过零提示在各种任务上表现出竞争性能，但这种方法在分层设置中表现较差，因为在提示中包含大而结构化的标签空间是无效的。另一方面，以前的弱监督分层文本分类方法仅利用原始的Taxonomy骨架，忽略了文本语料库中隐藏的丰富信息，这些信息可以用作额外的类别指示信息。

    arXiv:2403.00165v1 Announce Type: new  Abstract: Hierarchical text classification aims to categorize each document into a set of classes in a label taxonomy. Most earlier works focus on fully or semi-supervised methods that require a large amount of human annotated data which is costly and time-consuming to acquire. To alleviate human efforts, in this paper, we work on hierarchical text classification with the minimal amount of supervision: using the sole class name of each node as the only supervision. Recently, large language models (LLM) show competitive performance on various tasks through zero-shot prompting, but this method performs poorly in the hierarchical setting, because it is ineffective to include the large and structured label space in a prompt. On the other hand, previous weakly-supervised hierarchical text classification methods only utilize the raw taxonomy skeleton and ignore the rich information hidden in the text corpus that can serve as additional class-indicative 
    
[^5]: SymBa：符号化向后推理用于多步自然语言推理

    SymBa: Symbolic Backward Chaining for Multi-step Natural Language Reasoning

    [https://arxiv.org/abs/2402.12806](https://arxiv.org/abs/2402.12806)

    SymBa提出了一种符号化向后推理方法，在多步自然语言推理中取得了显著的性能和效率提升，能够生成可解释的结构化证明。

    

    最近大型语言模型（LLMs）展示了在一系列思维提示中出色的推理能力，但忠实的多步推理依然是一个挑战。我们专注于向后推理，即通过逻辑规则递归地分解查询，直到证明为止。为了解决当前向后推理实现的局限性，我们提出了SymBa（符号化向后推理）。在SymBa中，符号化自顶向下求解器控制整个证明过程，当求解器遇到死胡同时，才调用LLM生成单个推理步骤。通过这种新颖的求解器-LLM集成，SymBa在各种多步推理基准（ProofWriter，Birds-Electricity，GSM8k，CLUTRR-TF，ECtHR Article 6）中相比向后推理基线取得了性能、证明忠实性和效率显著提高，能够生成可解释的结构化证明。

    arXiv:2402.12806v1 Announce Type: new  Abstract: Large Language Models (LLMs) have recently demonstrated remarkable reasoning ability as in Chain-of-thought prompting, but faithful multi-step reasoning remains a challenge. We specifically focus on backward chaining, where the query is recursively decomposed using logical rules until proven. To address the limitations of current backward chaining implementations, we propose SymBa (Symbolic Backward Chaining). In SymBa, the symbolic top-down solver controls the entire proof process and the LLM is called to generate a single reasoning step only when the solver encounters a dead end. By this novel solver-LLM integration, while being able to produce an interpretable, structured proof, SymBa achieves significant improvement in performance, proof faithfulness, and efficiency in diverse multi-step reasoning benchmarks (ProofWriter, Birds-Electricity, GSM8k, CLUTRR-TF, ECtHR Article 6) compared to backward chaining baselines.
    
[^6]: 优秀的视觉指导有什么特点？综合复杂的视觉推理指令用于视觉指导调整

    What Makes for Good Visual Instructions? Synthesizing Complex Visual Reasoning Instructions for Visual Instruction Tuning. (arXiv:2311.01487v1 [cs.CV])

    [http://arxiv.org/abs/2311.01487](http://arxiv.org/abs/2311.01487)

    通过综合复杂的视觉推理任务，可以有效改善多模式大型语言模型在评估基准上的性能。我们提出了一种自动创建高质量复杂视觉推理指令的系统方法。

    

    视觉指导调整是提高多模式大型语言模型（MLLMs）的零样本泛化能力的重要方法。最近提出了许多着眼于不同焦点和特征的视觉指导数据集，使得MLLMs在评估基准上取得了令人惊讶的结果。为了开发更强大的MLLMs，本文旨在研究一个更基本的问题：“什么样的视觉指导才是好的？”通过进行全面的实证研究，我们发现侧重于复杂视觉推理任务的指导对于改善MLLMs在评估基准上的性能特别有效。基于这一发现，我们设计了一个系统的方法来自动创建高质量的复杂视觉推理指令。我们的方法采用合成-复杂化-重构的范式，利用多个阶段逐渐增加指令的复杂性，同时保证质量。

    Visual instruction tuning is an essential approach to improving the zero-shot generalization capability of Multi-modal Large Language Models (MLLMs). A surge of visual instruction datasets with various focuses and characteristics have been proposed recently, enabling MLLMs to achieve surprising results on evaluation benchmarks. To develop more capable MLLMs, in this paper, we aim to investigate a more fundamental question: ``what makes for good visual instructions?''. By conducting a comprehensive empirical study, we find that instructions focused on complex visual reasoning tasks are particularly effective in improving the performance of MLLMs on evaluation benchmarks. Building upon this finding, we design a systematic approach to automatically creating high-quality complex visual reasoning instructions. Our approach employs a synthesis-complication-reformulation paradigm, leveraging multiple stages to gradually increase the complexity of the instructions while guaranteeing quality. B
    
[^7]: 证明LLM对抗敌对提示的安全性

    Certifying LLM Safety against Adversarial Prompting. (arXiv:2309.02705v1 [cs.CL])

    [http://arxiv.org/abs/2309.02705](http://arxiv.org/abs/2309.02705)

    本研究提出了首个具有可验证安全保证的框架——消除和检查，用于对抗敌对提示。通过逐个消除标记并使用安全过滤器检查生成的子序列，确保任何敌对修改的有害输入提示都能被正确标识为有害。

    

    为了确保语言模型的输出安全，公开使用的大型语言模型（LLM）引入了所谓的“模型对齐”防护措施。一个对齐的语言模型应该拒绝用户的请求生成有害内容。然而，这种安全措施容易受到敌对提示的攻击，敌对提示包含恶意设计的标记序列，以规避模型的安全防护并导致生成有害内容。在这项工作中，我们介绍了可验证安全保证的第一个对抗敌对提示的框架——消除和检查。我们逐个消除标记，并使用安全过滤器检查生成的子序列。如果安全过滤器检测到任何子序列或输入提示有害，我们的过程将将输入提示标记为有害。这保证了对于某个特定大小的有害输入提示的任何敌对修改也将被标记为有害。我们对抗三种攻击模式：i)敌对后缀，即附加敌对序列…

    Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial prompts, which contain maliciously designed token sequences to circumvent the model's safety guards and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We erase tokens individually and inspect the resulting subsequences using a safety filter. Our procedure labels the input prompt as harmful if any subsequences or the input prompt are detected as harmful by the filter. This guarantees that any adversarial modification of a harmful prompt up to a certain size is also labeled harmful. We defend against three attack modes: i) adversarial suffix, which appends an adversarial seq
    

