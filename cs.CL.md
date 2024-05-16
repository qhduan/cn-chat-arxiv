# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey of Large Language Models in Medicine: Principles, Applications, and Challenges](https://rss.arxiv.org/abs/2311.05112) | 本综述提供了医学中大型语言模型（LLMs）的原理、应用和挑战的全面概述。同时回答了医学LLMs的构建、下游性能、实际应用、挑战以及更好构建和利用的问题。旨在为构建有效的医学LLMs提供见解和实用资源。 |
| [^2] | [Efficient Pruning of Large Language Model with Adaptive Estimation Fusion](https://arxiv.org/abs/2403.10799) | 提出了一种简单而高效的剪枝方法，能够自适应地模拟每个子结构的重要性，并根据多层结构的结果自适应地融合粗粒度和细粒度的估计。 |
| [^3] | [BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://arxiv.org/abs/2402.04291) | BiLLM是一种针对预训练LLMs的1位后训练量化方案，通过识别重要的权重和优化二值化，成功实现了高准确度的推理。 |
| [^4] | [LLM Voting: Human Choices and AI Collective Decision Making](https://arxiv.org/abs/2402.01766) | 本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并揭示了LLMs与人类在决策和偏见方面的差异。研究发现，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。 |
| [^5] | [Not My Voice! A Taxonomy of Ethical and Safety Harms of Speech Generators](https://arxiv.org/abs/2402.01708) | 语音生成技术的广泛应用给社会带来了伦理和安全风险。本文通过分析语音生成事件，总结出特定伤害模式，并将其分类。这些特定伤害涉及到受影响个体的曝光程度以及相关利益相关者和技术系统之间的相互作用。 |
| [^6] | [Multilingual Text-to-Image Generation Magnifies Gender Stereotypes and Prompt Engineering May Not Help You](https://arxiv.org/abs/2401.16092) | 多语言文本到图像生成模型存在性别偏见；通过MAGBIG评估模型时，发现模型对不同语言具有重要差异；我们呼吁研究多语言模型领域消除性别偏见。 |
| [^7] | [ChatQA: Building GPT-4 Level Conversational QA Models.](http://arxiv.org/abs/2401.10225) | ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。 |
| [^8] | [Beyond Turing: A Comparative Analysis of Approaches for Detecting Machine-Generated Text.](http://arxiv.org/abs/2311.12373) | 该论文对比了三种不同方法用于检测人类生成和机器生成文本的能力，并发现它们在性能上存在显著差异，为进一步研究和开发具有鲁棒性和高度区分性的模型提供了重要参考。 |
| [^9] | [Kid-Whisper: Towards Bridging the Performance Gap in Automatic Speech Recognition for Children VS. Adults.](http://arxiv.org/abs/2309.07927) | 本文主要研究利用My Science Tutor（MyST）儿童语音语料库和更有效的数据预处理来改进自动语音识别（ASR）系统对儿童语音的识别性能。将Whisper系统整合到儿童语音识别中，显示了表现可行和高效。 |
| [^10] | [Tackling Fake News in Bengali: Unraveling the Impact of Summarization vs. Augmentation on Pre-trained Language Models.](http://arxiv.org/abs/2307.06979) | 本论文研究了孟加拉语中假新闻的检测问题。通过使用总结和扩充技术，结合预训练语言模型，提出了一种四重方法来分类孟加拉语的假新闻文章。研究表明，总结和扩充在孟加拉语假新闻检测中具有有效性。 |
| [^11] | [VisualGPTScore: Visio-Linguistic Reasoning with Multimodal Generative Pre-Training Scores.](http://arxiv.org/abs/2306.01879) | 我们提出了VisualGPTScore方法，能够使用多模态生成分数捕捉文本标题可能性，并在图像条件语言模型上进行计算，具备组合推理能力。 |
| [^12] | [Tailoring Instructions to Student's Learning Levels Boosts Knowledge Distillation.](http://arxiv.org/abs/2305.09651) | 本文提出了一种个性化指导的学习技术，称为LGTM，其利用蒸馏效应选择样本以增强学生的泛化能力，在GLUE基准测试的6个文本分类任务中优于10个常见的知识蒸馏基线算法。 |
| [^13] | [Cleansing Jewel: A Neural Spelling Correction Model Built On Google OCR-ed Tibetan Manuscripts.](http://arxiv.org/abs/2304.03427) | 本文提出了一种基于谷歌OCR扫描的藏文手稿的神经拼写纠错模型，可以自动纠正OCR输出中的噪声。 |

# 详细

[^1]: 医学中的大型语言模型调查：原理、应用和挑战

    A Survey of Large Language Models in Medicine: Principles, Applications, and Challenges

    [https://rss.arxiv.org/abs/2311.05112](https://rss.arxiv.org/abs/2311.05112)

    本综述提供了医学中大型语言模型（LLMs）的原理、应用和挑战的全面概述。同时回答了医学LLMs的构建、下游性能、实际应用、挑战以及更好构建和利用的问题。旨在为构建有效的医学LLMs提供见解和实用资源。

    

    大型语言模型（LLMs），如ChatGPT，由于其理解和生成人类语言的能力而受到了广泛关注。在人工智能和临床医学中，LLMs在协助医生进行患者护理方面正在成为一个有前景的研究方向。本综述提供了医学中LLMs的原理、应用和面临的挑战的全面概述。我们回答了以下具体问题：1）如何构建医学LLMs？2）什么是医学LLMs的下游性能评估指标？3）在现实临床实践中，如何利用医学LLMs？4）使用医学LLMs会出现哪些挑战？5）如何更好地构建和利用医学LLMs？本综述旨在提供关于医学中LLMs的机遇和挑战的见解，并作为构建有效的医学LLMs的实用资源。我们还维护并定期更新一个清单

    Large language models (LLMs), such as ChatGPT, have received substantial attention due to their capabilities for understanding and generating human language. LLMs in medicine to assist physicians for patient care are emerging as a promising research direction in both artificial intelligence and clinical medicine. This review provides a comprehensive overview of the principles, applications, and challenges faced by LLMs in medicine. We address the following specific questions: 1) How should medical LLMs be built? 2) What are the measures for the downstream performance of medical LLMs? 3) How should medical LLMs be utilized in real-world clinical practice? 4) What challenges arise from the use of medical LLMs? and 5) How should we better construct and utilize medical LLMs? This review aims to provide insights into the opportunities and challenges of LLMs in medicine, and serve as a practical resource for constructing effective medical LLMs. We also maintain and regularly updated list of 
    
[^2]: 使用自适应估计融合高效剪枝大型语言模型

    Efficient Pruning of Large Language Model with Adaptive Estimation Fusion

    [https://arxiv.org/abs/2403.10799](https://arxiv.org/abs/2403.10799)

    提出了一种简单而高效的剪枝方法，能够自适应地模拟每个子结构的重要性，并根据多层结构的结果自适应地融合粗粒度和细粒度的估计。

    

    大型语言模型（LLMs）已经成为许多生成性下游任务中至关重要的组成部分，这导致在资源受限设备上高效部署它们成为不可避免的趋势和重大挑战。结构化剪枝是解决这一挑战的广泛应用方法。然而，当处理多个解码器层的复杂结构时，通常的方法往往采用常见的估计方法进行剪枝。这些方法导致特定下游任务精度下降。本文介绍了一种简单而有效的方法，可自适应地模拟每个子结构的重要性。同时，它可以基于复杂和多层结构的结果，自适应地融合粗粒度和细粒度的估计。我们设计的所有方面都无缝集成到端到端的剪枝框架中。与主流数据集上的最先进方法相比，我们的实验结果表明

    arXiv:2403.10799v1 Announce Type: cross  Abstract: Large language models (LLMs) have become crucial for many generative downstream tasks, leading to an inevitable trend and significant challenge to deploy them efficiently on resource-constrained devices. Structured pruning is a widely used method to address this challenge. However, when dealing with the complex structure of the multiple decoder layers, general methods often employ common estimation approaches for pruning. These approaches lead to a decline in accuracy for specific downstream tasks. In this paper, we introduce a simple yet efficient method that adaptively models the importance of each substructure. Meanwhile, it can adaptively fuse coarse-grained and finegrained estimations based on the results from complex and multilayer structures. All aspects of our design seamlessly integrate into the endto-end pruning framework. Our experimental results, compared with state-of-the-art methods on mainstream datasets, demonstrate ave
    
[^3]: BiLLM: 推动LLMs的后训练量化极限

    BiLLM: Pushing the Limit of Post-Training Quantization for LLMs

    [https://arxiv.org/abs/2402.04291](https://arxiv.org/abs/2402.04291)

    BiLLM是一种针对预训练LLMs的1位后训练量化方案，通过识别重要的权重和优化二值化，成功实现了高准确度的推理。

    

    预训练的大型语言模型（LLMs）具有出色的通用语言处理能力，但对内存和计算资源有很大的需求。作为一种强大的压缩技术，二值化可以将模型权重极大地减少到仅1位，降低了昂贵的计算和内存需求。然而，现有的量化技术在超低位宽下无法保持LLM的性能。针对这一挑战，我们提出了BiLLM，这是一种针对预训练LLM定制的开创性的1位后训练量化方案。基于LLMs的权重分布，BiLLM首先识别和结构选择重要的权重，并通过有效的二值化残差逼近策略来最小化压缩损失。此外，考虑到非重要权重的钟形分布，我们提出了一种最佳分割搜索方法，以准确地将它们分组和二值化。BiLLM首次实现了高准确度的推理。

    Pretrained large language models (LLMs) exhibit exceptional general language processing capabilities but come with significant demands on memory and computational resources. As a powerful compression technology, binarization can extremely reduce model weights to a mere 1 bit, lowering the expensive computation and memory requirements. However, existing quantization techniques fall short of maintaining LLM performance under ultra-low bit-widths. In response to this challenge, we present BiLLM, a groundbreaking 1-bit post-training quantization scheme tailored for pretrained LLMs. Based on the weight distribution of LLMs, BiLLM first identifies and structurally selects salient weights, and minimizes the compression loss through an effective binary residual approximation strategy. Moreover, considering the bell-shaped distribution of the non-salient weights, we propose an optimal splitting search to group and binarize them accurately. BiLLM achieving for the first time high-accuracy infere
    
[^4]: LLM投票：人类选择和AI集体决策

    LLM Voting: Human Choices and AI Collective Decision Making

    [https://arxiv.org/abs/2402.01766](https://arxiv.org/abs/2402.01766)

    本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并揭示了LLMs与人类在决策和偏见方面的差异。研究发现，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。

    

    本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并与人类投票模式进行了对比。我们的方法包括进行人类投票实验以建立人类偏好的基准，并与LLM代理进行平行实验。研究聚焦于集体结果和个体偏好，揭示了人类和LLMs之间在决策和固有偏见方面的差异。我们观察到LLMs在偏好多样性和一致性之间存在权衡，相比人类选民的多样偏好，LLMs有更趋向于一致选择的倾向。这一发现表明，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。

    This paper investigates the voting behaviors of Large Language Models (LLMs), particularly OpenAI's GPT4 and LLaMA2, and their alignment with human voting patterns. Our approach included a human voting experiment to establish a baseline for human preferences and a parallel experiment with LLM agents. The study focused on both collective outcomes and individual preferences, revealing differences in decision-making and inherent biases between humans and LLMs. We observed a trade-off between preference diversity and alignment in LLMs, with a tendency towards more uniform choices as compared to the diverse preferences of human voters. This finding indicates that LLMs could lead to more homogenized collective outcomes when used in voting assistance, underscoring the need for cautious integration of LLMs into democratic processes.
    
[^5]: 不是我的声音！语音生成器的伦理和安全伤害分类法

    Not My Voice! A Taxonomy of Ethical and Safety Harms of Speech Generators

    [https://arxiv.org/abs/2402.01708](https://arxiv.org/abs/2402.01708)

    语音生成技术的广泛应用给社会带来了伦理和安全风险。本文通过分析语音生成事件，总结出特定伤害模式，并将其分类。这些特定伤害涉及到受影响个体的曝光程度以及相关利益相关者和技术系统之间的相互作用。

    

    人工智能广泛采用语音生成技术给社会带来了一系列重大的伦理和安全风险，亟需解决。例如，在美国，越来越多的语音生成事件与警察受到恶作剧袭击有关，匿名行为者制造合成的声音打电话给警察，要求关闭学校和医院，或者以暴力手段进入无辜市民的家中。这样的事件表明，多模态生成人工智能的风险和伤害并不存在于孤立状态，而是源于多个利益相关者和技术人工智能系统之间的互动。本文通过分析语音生成事件，研究特定伤害模式的出现。我们发现特定伤害可以根据受影响个体的曝光程度进行分类，即他们是语音生成系统的主体、与之互动、受其影响或被排除在外。同样，特定伤害也与相关利益相关者和技术系统之间的相互作用有关。

    The rapid and wide-scale adoption of AI to generate human speech poses a range of significant ethical and safety risks to society that need to be addressed. For example, a growing number of speech generation incidents are associated with swatting attacks in the United States, where anonymous perpetrators create synthetic voices that call police officers to close down schools and hospitals, or to violently gain access to innocent citizens' homes. Incidents like this demonstrate that multimodal generative AI risks and harms do not exist in isolation, but arise from the interactions of multiple stakeholders and technical AI systems. In this paper we analyse speech generation incidents to study how patterns of specific harms arise. We find that specific harms can be categorised according to the exposure of affected individuals, that is to say whether they are a subject of, interact with, suffer due to, or are excluded from speech generation systems. Similarly, specific harms are also a con
    
[^6]: 多语言文本到图像生成放大了性别刻板印象，并且修正工程可能无法帮助您

    Multilingual Text-to-Image Generation Magnifies Gender Stereotypes and Prompt Engineering May Not Help You

    [https://arxiv.org/abs/2401.16092](https://arxiv.org/abs/2401.16092)

    多语言文本到图像生成模型存在性别偏见；通过MAGBIG评估模型时，发现模型对不同语言具有重要差异；我们呼吁研究多语言模型领域消除性别偏见。

    

    最近，文本到图像生成模型在图像质量、灵活性和文本对齐方面取得了令人惊讶的结果，并因此在越来越多的应用中得到应用。通过改善多语言能力，更多的社群现在可以访问这种技术。然而，正如我们将展示的那样，多语言模型与单语模型一样受到(性别)偏见的困扰。此外，人们自然期望这些模型在不同语言之间提供类似的结果，但事实并非如此，不同语言之间存在重要的差异。因此，我们提出了一个旨在促进没有性别偏见的多语言模型研究的新基准MAGBIG。我们研究了多语言T2I模型是否通过MAGBIG放大了性别偏见。为此，我们使用多语言提示请求特定职业或特质的人像图像(使用形容词)。我们的结果不仅表明模型偏离了规范的假设，...

    Text-to-image generation models have recently achieved astonishing results in image quality, flexibility, and text alignment and are consequently employed in a fast-growing number of applications. Through improvements in multilingual abilities, a larger community now has access to this kind of technology. Yet, as we will show, multilingual models suffer similarly from (gender) biases as monolingual models. Furthermore, the natural expectation is that these models will provide similar results across languages, but this is not the case and there are important differences between languages. Thus, we propose a novel benchmark MAGBIG intending to foster research in multilingual models without gender bias. We investigate whether multilingual T2I models magnify gender bias with MAGBIG. To this end, we use multilingual prompts requesting portrait images of persons of a certain occupation or trait (using adjectives). Our results show not only that models deviate from the normative assumption th
    
[^7]: ChatQA: 构建GPT-4级对话问答模型

    ChatQA: Building GPT-4 Level Conversational QA Models. (arXiv:2401.10225v1 [cs.CL])

    [http://arxiv.org/abs/2401.10225](http://arxiv.org/abs/2401.10225)

    ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。

    

    在这项工作中，我们介绍了ChatQA，一系列具有GPT-4级别准确性的对话问答模型。具体地，我们提出了一个两阶段的指令调整方法，可以显著提高大型语言模型（LLM）在零-shot对话问答中的结果。为了处理对话问答中的检索问题，我们在多轮问答数据集上进行了密集检索器的微调，这样可以提供与使用最先进的查询重写模型相当的结果，同时大大降低部署成本。值得注意的是，我们的ChatQA-70B可以在10个对话问答数据集的平均分上超过GPT-4（54.14 vs. 53.90），而不依赖于OpenAI GPT模型的任何合成数据。

    In this work, we introduce ChatQA, a family of conversational question answering (QA) models, that obtain GPT-4 level accuracies. Specifically, we propose a two-stage instruction tuning method that can significantly improve the zero-shot conversational QA results from large language models (LLMs). To handle retrieval in conversational QA, we fine-tune a dense retriever on a multi-turn QA dataset, which provides comparable results to using the state-of-the-art query rewriting model while largely reducing deployment cost. Notably, our ChatQA-70B can outperform GPT-4 in terms of average score on 10 conversational QA datasets (54.14 vs. 53.90), without relying on any synthetic data from OpenAI GPT models.
    
[^8]: 超越图灵: 检测机器生成文本的方法比较分析

    Beyond Turing: A Comparative Analysis of Approaches for Detecting Machine-Generated Text. (arXiv:2311.12373v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2311.12373](http://arxiv.org/abs/2311.12373)

    该论文对比了三种不同方法用于检测人类生成和机器生成文本的能力，并发现它们在性能上存在显著差异，为进一步研究和开发具有鲁棒性和高度区分性的模型提供了重要参考。

    

    在预训练语言模型（PLMs）生成文本方面取得了显著进展，然而区分人类生成和机器生成的文本越来越具有挑战性。本论文对三种不同的方法进行了深入评估，用于解决这个任务：传统的浅层学习、语言模型（LM）微调和多语言模型微调。这些方法在广泛的机器生成文本上经过严格测试，提供了一个评估它们在区分人类和机器构造语言方面能力的基准。结果显示不同方法的性能存在显著差异，因此强调了在这一关键NLP领域的持续需求。本研究提供了有价值的见解，并为未来旨在创建强大且高度区分性模型的研究铺平了道路。

    Significant progress has been made on text generation by pre-trained language models (PLMs), yet distinguishing between human and machine-generated text poses an escalating challenge. This paper offers an in-depth evaluation of three distinct methods used to address this task: traditional shallow learning, Language Model (LM) fine-tuning, and Multilingual Model fine-tuning. These approaches are rigorously tested on a wide range of machine-generated texts, providing a benchmark of their competence in distinguishing between human-authored and machine-authored linguistic constructs. The results reveal considerable differences in performance across methods, thus emphasizing the continued need for advancement in this crucial area of NLP. This study offers valuable insights and paves the way for future research aimed at creating robust and highly discriminative models.
    
[^9]: Kid-Whisper: 助力填补儿童与成人自动语音识别性能差距的研究

    Kid-Whisper: Towards Bridging the Performance Gap in Automatic Speech Recognition for Children VS. Adults. (arXiv:2309.07927v1 [eess.AS])

    [http://arxiv.org/abs/2309.07927](http://arxiv.org/abs/2309.07927)

    本文主要研究利用My Science Tutor（MyST）儿童语音语料库和更有效的数据预处理来改进自动语音识别（ASR）系统对儿童语音的识别性能。将Whisper系统整合到儿童语音识别中，显示了表现可行和高效。

    

    最近自动语音识别（ASR）系统的进展，例如Whisper，展示了这些系统在足够的数据条件下接近人类水平的性能潜力。然而，这一进展并不适用于儿童ASR，原因是适用于儿童的专用数据库的可用性有限，且儿童语音具有与成人不同的特征。最近的一项研究调查了利用My Science Tutor（MyST）儿童语音语料库提高Whisper识别儿童语音的性能。本文在这些研究结果的基础上，通过更有效的数据预处理增强了MyST数据集的实用性。我们还强调了改进儿童ASR性能的重要挑战。结果展示了将Whisper有效整合到儿童语音识别中的可行性和高效性。

    Recent advancements in Automatic Speech Recognition (ASR) systems, exemplified by Whisper, have demonstrated the potential of these systems to approach human-level performance given sufficient data. However, this progress doesn't readily extend to ASR for children due to the limited availability of suitable child-specific databases and the distinct characteristics of children's speech. A recent study investigated leveraging the My Science Tutor (MyST) children's speech corpus to enhance Whisper's performance in recognizing children's speech. This paper builds on these findings by enhancing the utility of the MyST dataset through more efficient data preprocessing. We also highlight important challenges towards improving children's ASR performance. The results showcase the viable and efficient integration of Whisper for effective children's speech recognition.
    
[^10]: 解决孟加拉语中的假新闻问题：揭示总结与扩充对预训练语言模型的影响

    Tackling Fake News in Bengali: Unraveling the Impact of Summarization vs. Augmentation on Pre-trained Language Models. (arXiv:2307.06979v1 [cs.CL])

    [http://arxiv.org/abs/2307.06979](http://arxiv.org/abs/2307.06979)

    本论文研究了孟加拉语中假新闻的检测问题。通过使用总结和扩充技术，结合预训练语言模型，提出了一种四重方法来分类孟加拉语的假新闻文章。研究表明，总结和扩充在孟加拉语假新闻检测中具有有效性。

    

    随着社交媒体和在线新闻来源的兴起，假新闻已成为全球性的重大问题。然而，在像孟加拉语这样的低资源语言中检测假新闻在研究中受到了有限的关注。本文提出了一种方法，利用总结和扩充技术以及五种预训练语言模型来分类孟加拉语的假新闻文章。我们的方法包括将英语新闻文章进行翻译，并使用扩充技术来解决假新闻文章的不足问题。我们的研究还着重于通过总结新闻来解决基于BERT模型的令牌长度限制。通过广泛的实验和严格的评估，我们展示了总结和扩充在孟加拉语假新闻检测中的有效性。我们使用三个独立的测试数据集来评估我们的模型。当将BanglaBERT基础模型与扩充技术相结合时，取得了令人印象深刻的准确性。

    With the rise of social media and online news sources, fake news has become a significant issue globally. However, the detection of fake news in low resource languages like Bengali has received limited attention in research. In this paper, we propose a methodology consisting of four distinct approaches to classify fake news articles in Bengali using summarization and augmentation techniques with five pre-trained language models. Our approach includes translating English news articles and using augmentation techniques to curb the deficit of fake news articles. Our research also focused on summarizing the news to tackle the token length limitation of BERT based models. Through extensive experimentation and rigorous evaluation, we show the effectiveness of summarization and augmentation in the case of Bengali fake news detection. We evaluated our models using three separate test datasets. The BanglaBERT Base model, when combined with augmentation techniques, achieved an impressive accurac
    
[^11]: VisualGPTScore: 多模态生成预训练分数的视觉语义推理。

    VisualGPTScore: Visio-Linguistic Reasoning with Multimodal Generative Pre-Training Scores. (arXiv:2306.01879v1 [cs.CV])

    [http://arxiv.org/abs/2306.01879](http://arxiv.org/abs/2306.01879)

    我们提出了VisualGPTScore方法，能够使用多模态生成分数捕捉文本标题可能性，并在图像条件语言模型上进行计算，具备组合推理能力。

    

    本文提出了一种名为 VisualGPTScore 的方法，使用多模态生成分数来捕捉文本标题可能性，并使用图像条件语言模型在图像上运算。与传统观点认为的VLM只是无意义的单词袋模型不同，我们的 VisualGPTScore 在 ARO 和 Crepe 等最近提出的图像文本检索基准测试中展现了顶尖的性能，证明了其具备组合推理能力。

    Vision-language models (VLMs) discriminatively pre-trained with contrastive image-text matching losses such as $P(\text{match}|\text{text}, \text{image})$ have been criticized for lacking compositional understanding. This means they might output similar scores even if the original caption is rearranged into a different semantic statement. To address this, we propose to use the ${\bf V}$isual ${\bf G}$enerative ${\bf P}$re-${\bf T}$raining Score (${\bf VisualGPTScore}$) of $P(\text{text}|\text{image})$, a $\textit{multimodal generative}$ score that captures the likelihood of a text caption conditioned on an image using an image-conditioned language model. Contrary to the belief that VLMs are mere bag-of-words models, our off-the-shelf VisualGPTScore demonstrates top-tier performance on recently proposed image-text retrieval benchmarks like ARO and Crepe that assess compositional reasoning. Furthermore, we factorize VisualGPTScore into a product of the $\textit{marginal}$ P(text) and the
    
[^12]: 个性化指导有助于知识蒸馏

    Tailoring Instructions to Student's Learning Levels Boosts Knowledge Distillation. (arXiv:2305.09651v1 [cs.CL])

    [http://arxiv.org/abs/2305.09651](http://arxiv.org/abs/2305.09651)

    本文提出了一种个性化指导的学习技术，称为LGTM，其利用蒸馏效应选择样本以增强学生的泛化能力，在GLUE基准测试的6个文本分类任务中优于10个常见的知识蒸馏基线算法。

    

    先前研究表明，能力超群的教师模型并不一定能够让学生水平得到提升，这凸显了当前教师培训实践和有效知识传授之间的不一致性。为了提高教师培训过程的指导效果，本文引入了蒸馏效应的概念，以确定每个训练样本对学生泛化能力的影响。我们提出了一种名为学好教师很重要（LGTM）的有效训练技术，以将蒸馏效应纳入教师的学习过程中。通过优先选择可能提升学生泛化能力的样本，我们的LGTM在GLUE基准测试的6个文本分类任务中优于10个常见的知识蒸馏基线算法。

    It has been commonly observed that a teacher model with superior performance does not necessarily result in a stronger student, highlighting a discrepancy between current teacher training practices and effective knowledge transfer. In order to enhance the guidance of the teacher training process, we introduce the concept of distillation influence to determine the impact of distillation from each training sample on the student's generalization ability. In this paper, we propose Learning Good Teacher Matters (LGTM), an efficient training technique for incorporating distillation influence into the teacher's learning process. By prioritizing samples that are likely to enhance the student's generalization ability, our LGTM outperforms 10 common knowledge distillation baselines on 6 text classification tasks in the GLUE benchmark.
    
[^13]: 基于谷歌OCR扫描的藏文手稿的神经拼写纠错模型

    Cleansing Jewel: A Neural Spelling Correction Model Built On Google OCR-ed Tibetan Manuscripts. (arXiv:2304.03427v1 [cs.CL])

    [http://arxiv.org/abs/2304.03427](http://arxiv.org/abs/2304.03427)

    本文提出了一种基于谷歌OCR扫描的藏文手稿的神经拼写纠错模型，可以自动纠正OCR输出中的噪声。

    

    人文学者在研究历史、宗教和社会政治结构等方面经常依赖于古代手稿。虽然OCR技术可以将这些宝贵手稿数字化，但多数手稿因磨损而过时，OCR程序没办法识别翻页的虚淡或污渍。本文提出了一种基于谷歌OCR扫描的藏文手稿的神经拼写纠错模型，可以自动纠正OCR输出中的噪声。本文分为四个部分：数据集、模型架构、训练和分析。首先，我们对原始藏文电子文本语料库进行了特征工程，并将其转化为两组结构化数据框——一组匹配的玩具数据和一组匹配的真实数据。然后，我们在Transformer架构中实现了置信度得分机制来执行拼写校正任务。根据损失和字符错误率，我们的Transformer + 置信度得分机制比其他常用的拼写校正算法表现更好。

    Scholars in the humanities rely heavily on ancient manuscripts to study history, religion, and socio-political structures in the past. Many efforts have been devoted to digitizing these precious manuscripts using OCR technology, but most manuscripts were blemished over the centuries so that an Optical Character Recognition (OCR) program cannot be expected to capture faded graphs and stains on pages. This work presents a neural spelling correction model built on Google OCR-ed Tibetan Manuscripts to auto-correct OCR-ed noisy output. This paper is divided into four sections: dataset, model architecture, training and analysis. First, we feature-engineered our raw Tibetan etext corpus into two sets of structured data frames -- a set of paired toy data and a set of paired real data. Then, we implemented a Confidence Score mechanism into the Transformer architecture to perform spelling correction tasks. According to the Loss and Character Error Rate, our Transformer + Confidence score mechani
    

