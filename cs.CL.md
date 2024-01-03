# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents.](http://arxiv.org/abs/2310.19923) | Jina Embeddings 2是一个能够处理长篇文档的文本嵌入模型，突破了传统512个标记限制，提供了高达8192个标记的容量。 |
| [^2] | [Gaining Wisdom from Setbacks: Aligning Large Language Models via Mistake Analysis.](http://arxiv.org/abs/2310.10477) | 该论文介绍了一种基于错误分析的对齐策略，通过暴露大型语言模型的错误输出并进行评估，以理解内部原因。通过这种方法，有毒回应可以转化为模型对齐的指导调谐语料，从而提高模型的安全性并训练其进行自我批评。 |
| [^3] | [Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model.](http://arxiv.org/abs/2310.09520) | 该论文介绍了一种名为Reward-Augmented Decoding (RAD)的文本生成方法，使用小型的单向奖励模型来鼓励语言模型生成具有特定属性的文本。RAD在生成非有害和情感受控文本方面表现最佳，并且在非常大的语言模型上也很有效。 |
| [^4] | [The Cambridge Law Corpus: A Corpus for Legal AI Research.](http://arxiv.org/abs/2309.12269) | 剑桥法律语料库是一个用于法律人工智能研究的语料库，包含来自英国的超过250,000个法庭案例。在该语料库的基础上，我们提供了案例结果的专家注解，并使用多个模型进行了案例结果提取的训练和评估，为研究提供了基准。 |
| [^5] | [Evidence of Human-Like Visual-Linguistic Integration in Multimodal Large Language Models During Predictive Language Processing.](http://arxiv.org/abs/2308.06035) | 这篇论文研究了多模态大语言模型（mLLMs）在预测语言处理过程中与人类的视觉-语言集成能力是否一致的问题，并通过实验验证了mLLMs的多模态输入方法可以减少认知负荷，提高感知和理解能力。 |
| [^6] | [RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model.](http://arxiv.org/abs/2306.11300) | 本文提出了一个新的框架RS5M，该框架包括领域基础模型（DFM），用于实现通用基础模型（GFM）和领域特定下游任务之间的转换。另外，还介绍了一个遥感领域的大规模图像-文本配对数据集RS5M，该数据集是通过过滤公开可用的图像-文本配对数据集并使用预训练的视觉-语言基础模型为标签数据集生成标题。 |
| [^7] | [Language Models are Bounded Pragmatic Speakers.](http://arxiv.org/abs/2305.17760) | 本文提出了一个概率认知模型，称为有限实用说话者，用于表征不同变体的语言模型的操作方式。经过人类反馈的强化学习微调的大型语言模型具有概念上类似于 快与慢思考模型的思维模型，而这种思维模型被归因于人类。此研究凸显了采用认知概率建模方法对语言模型的理解、评估和推进的价值。 |
| [^8] | [ArtGPT-4: Artistic Vision-Language Understanding with Adapter-enhanced MiniGPT-4.](http://arxiv.org/abs/2305.07490) | ArtGPT-4是一种基于适配器增强的MiniGPT-4模型，专注于解决图像理解方面的问题，能够在短时间内训练出具备良好视觉语言理解能力的多模态模型。 |
| [^9] | [In-depth analysis of music structure as a self-organized network.](http://arxiv.org/abs/2303.13631) | 本文介绍了一种利用Essential Element Network (EEN)算法将音频编码成文本并进行相关性计算和优化应用于聚类系数的频率和排名的方法，得到了音乐的深层结构信息，为厘清音乐结构提供了新方法。 |

# 详细

[^1]: Jina Embeddings 2: 面向长篇文档的8192-Token通用文本嵌入模型

    Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents. (arXiv:2310.19923v1 [cs.CL])

    [http://arxiv.org/abs/2310.19923](http://arxiv.org/abs/2310.19923)

    Jina Embeddings 2是一个能够处理长篇文档的文本嵌入模型，突破了传统512个标记限制，提供了高达8192个标记的容量。

    

    文本嵌入模型已经成为将句子转化为固定大小特征向量的强大工具，这些向量包含了语义信息。尽管这些模型对于信息检索、语义聚类和文本重排序等任务至关重要，但大多数现有的开源模型，尤其是基于BERT等架构构建的模型，难以表示长篇文档，并且常常会进行截断。为了缓解这个挑战，一种常见的方法是将文档分割成更小的段落进行嵌入。然而，这种策略会导致更大的向量集合，进而增加内存消耗，并且在向量搜索时会出现计算密集和延迟升高的问题。为了解决这些挑战，我们介绍了Jina Embeddings 2，这是一个开源的文本嵌入模型，可以容纳高达8192个标记。该模型旨在突破传统的512个标记限制，能够灵活处理长篇文档。

    Text embedding models have emerged as powerful tools for transforming sentences into fixed-sized feature vectors that encapsulate semantic information. While these models are essential for tasks like information retrieval, semantic clustering, and text re-ranking, most existing open-source models, especially those built on architectures like BERT, struggle to represent lengthy documents and often resort to truncation. One common approach to mitigate this challenge involves splitting documents into smaller paragraphs for embedding. However, this strategy results in a much larger set of vectors, consequently leading to increased memory consumption and computationally intensive vector searches with elevated latency.  To address these challenges, we introduce Jina Embeddings 2, an open-source text embedding model capable of accommodating up to 8192 tokens. This model is designed to transcend the conventional 512-token limit and adeptly process long documents. Jina Embeddings 2 not only ach
    
[^2]: 从挫折中获得智慧：通过错误分析对齐大型语言模型

    Gaining Wisdom from Setbacks: Aligning Large Language Models via Mistake Analysis. (arXiv:2310.10477v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.10477](http://arxiv.org/abs/2310.10477)

    该论文介绍了一种基于错误分析的对齐策略，通过暴露大型语言模型的错误输出并进行评估，以理解内部原因。通过这种方法，有毒回应可以转化为模型对齐的指导调谐语料，从而提高模型的安全性并训练其进行自我批评。

    

    大型语言模型（LLMs）的快速发展既带来了机遇，也带来了挑战，特别是在意外生成有害和有毒回应方面。传统的对齐方法致力于引导LLMs朝着期望的性能发展并保护它们免受恶意内容的侵害，而本研究提出了一种基于错误分析的全新对齐策略，通过有意暴露LLMs的缺陷输出并进行深入评估，以完全理解内部原因，通过自然语言分析。因此，有毒回应可以转化为模型对齐的指导调谐语料，LLMs不仅可以避免生成有缺陷的回应，还可以训练其进行自我批评，发挥其辨别有毒内容的内在能力。实验结果表明，所提出的方法在安全指令遵循方面优于传统的对齐技术，同时还保持了卓越的效率。

    The rapid advancement of large language models (LLMs) presents both opportunities and challenges, particularly concerning unintentional generation of harmful and toxic responses. While the traditional alignment methods strive to steer LLMs towards desired performance and shield them from malicious content, this study proposes a novel alignment strategy rooted in mistake analysis by exposing LLMs to flawed outputs purposefully and then conducting a thorough assessment to fully comprehend internal reasons via natural language analysis. Thus, toxic responses can be transformed into instruction tuning corpus for model alignment, and LLMs can not only be deterred from generating flawed responses but also trained to self-criticize, leveraging its innate ability to discriminate toxic content. Experimental results demonstrate that the proposed method outperforms conventional alignment techniques for safety instruction following, while maintaining superior efficiency.
    
[^3]: Reward-Augmented Decoding: 使用单向奖励模型实现高效的受控文本生成

    Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model. (arXiv:2310.09520v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.09520](http://arxiv.org/abs/2310.09520)

    该论文介绍了一种名为Reward-Augmented Decoding (RAD)的文本生成方法，使用小型的单向奖励模型来鼓励语言模型生成具有特定属性的文本。RAD在生成非有害和情感受控文本方面表现最佳，并且在非常大的语言模型上也很有效。

    

    尽管大型语言模型已经在许多应用中证明了其有效性，但它们通常生成的文本存在问题或者缺乏所需的属性。本文提出了一种名为Reward-Augmented Decoding (RAD)的文本生成方法，它利用一个小型的单向奖励模型来鼓励语言模型生成具有特定属性的文本。具体而言，RAD利用奖励模型对生成的文本进行评分，并通过重新调整采样概率来更倾向于高奖励的标记。通过使用单向奖励模型，RAD能够缓存先前生成步骤的激活值，降低计算开销。通过在生成非有害和情感受控文本方面的实验，我们证明RAD在仅改变生成过程的方法中表现最佳，并且与涉及重新训练语言模型的最先进方法相当。我们进一步验证了RAD在非常大的语言模型上的有效性。

    While large language models have proven effective in a huge range of downstream applications, they often generate text that is problematic or lacks a desired attribute. In this paper, we introduce Reward-Augmented Decoding (RAD), a text generation procedure that uses a small unidirectional reward model to encourage a language model to generate text that has certain properties. Specifically, RAD uses the reward model to score generations as they are produced and rescales sampling probabilities to favor high-reward tokens. By using a unidirectional reward model, RAD can cache activations from prior generation steps to decrease computational overhead. Through experiments on generating non-toxic and sentiment-controlled text, we demonstrate that RAD performs best among methods that change only the generation procedure and matches the performance of state-of-the-art methods that involve re-training the language model. We further validate that RAD is effective on very large language models w
    
[^4]: 剑桥法律语料库：用于法律人工智能研究的语料库

    The Cambridge Law Corpus: A Corpus for Legal AI Research. (arXiv:2309.12269v1 [cs.CL])

    [http://arxiv.org/abs/2309.12269](http://arxiv.org/abs/2309.12269)

    剑桥法律语料库是一个用于法律人工智能研究的语料库，包含来自英国的超过250,000个法庭案例。在该语料库的基础上，我们提供了案例结果的专家注解，并使用多个模型进行了案例结果提取的训练和评估，为研究提供了基准。

    

    我们介绍了剑桥法律语料库（CLC），这是一个用于法律人工智能研究的语料库。它包含了来自英国的超过250,000个法庭案例。大部分案例来自21世纪，但该语料库包括了16世纪以来的案例。本文介绍了该语料库的首次发布，包括原始文本和元数据。在语料库的基础上，我们提供了638个案例的法律专家对案例结果的注解。我们使用我们的标注数据，训练和评估了GPT-3、GPT-4和RoBERTa模型进行案例结果提取，以提供基准。我们还进行了广泛的法律和伦理讨论，以解决这些材料可能具有敏感性的问题。因此，该语料库只会在一定限制下用于研究目的。

    We introduce the Cambridge Law Corpus (CLC), a corpus for legal AI research. It consists of over 250 000 court cases from the UK. Most cases are from the 21st century, but the corpus includes cases as old as the 16th century. This paper presents the first release of the corpus, containing the raw text and meta-data. Together with the corpus, we provide annotations on case outcomes for 638 cases, done by legal experts. Using our annotated data, we have trained and evaluated case outcome extraction with GPT-3, GPT-4 and RoBERTa models to provide benchmarks. We include an extensive legal and ethical discussion to address the potentially sensitive nature of this material. As a consequence, the corpus will only be released for research purposes under certain restrictions.
    
[^5]: 多模态大语言模型在预测语言处理期间表现出人类视觉-语言集成的证据

    Evidence of Human-Like Visual-Linguistic Integration in Multimodal Large Language Models During Predictive Language Processing. (arXiv:2308.06035v1 [cs.AI])

    [http://arxiv.org/abs/2308.06035](http://arxiv.org/abs/2308.06035)

    这篇论文研究了多模态大语言模型（mLLMs）在预测语言处理过程中与人类的视觉-语言集成能力是否一致的问题，并通过实验验证了mLLMs的多模态输入方法可以减少认知负荷，提高感知和理解能力。

    

    大语言模型（LLMs）的先进语言处理能力引发了关于它们是否能够复制人类认知过程的争议。LLMs和人类在语言处理方面的一个区别在于，语言输入通常建立在多个知觉模态上，而大多数LLMs仅处理基于文本的信息。多模态基础使人类能够整合视觉背景与语言信息，从而对即将出现的单词的空间施加限制，减少认知负荷，提高感知和理解能力。最近的多模态LLMs（mLLMs）结合了视觉和语言嵌入空间，并使用变压器类型的注意机制进行下一个单词的预测。在多大程度上，基于多模态输入的预测语言处理在mLLMs和人类中吻合？为了回答这个问题，200名被试观看了短的视听剪辑，并估计了即将出现的动词或名词的可预测性。

    The advanced language processing abilities of large language models (LLMs) have stimulated debate over their capacity to replicate human-like cognitive processes. One differentiating factor between language processing in LLMs and humans is that language input is often grounded in more than one perceptual modality, whereas most LLMs process solely text-based information. Multimodal grounding allows humans to integrate - e.g. visual context with linguistic information and thereby place constraints on the space of upcoming words, reducing cognitive load and improving perception and comprehension. Recent multimodal LLMs (mLLMs) combine visual and linguistic embedding spaces with a transformer type attention mechanism for next-word prediction. To what extent does predictive language processing based on multimodal input align in mLLMs and humans? To answer this question, 200 human participants watched short audio-visual clips and estimated the predictability of an upcoming verb or noun. The 
    
[^6]: RS5M：用于遥感视觉-语言基础模型的大规模视觉-语言数据集

    RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model. (arXiv:2306.11300v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.11300](http://arxiv.org/abs/2306.11300)

    本文提出了一个新的框架RS5M，该框架包括领域基础模型（DFM），用于实现通用基础模型（GFM）和领域特定下游任务之间的转换。另外，还介绍了一个遥感领域的大规模图像-文本配对数据集RS5M，该数据集是通过过滤公开可用的图像-文本配对数据集并使用预训练的视觉-语言基础模型为标签数据集生成标题。

    

    利用大量图像-文本配对数据进行预训练的视觉-语言基础模型展示了前所未有的图像-文本关联能力，在各种下游任务中取得了显著的成果。关键挑战是如何利用已有的大规模预训练的视觉-语言基础模型，在域相关的下游任务中进行领域特定的迁移。本文提出了一个新的框架，包括领域基础模型（DFM），弥合了通用基础模型（GFM）和领域特定下游任务之间的差距。此外，我们还介绍了一个遥感领域（RS）的图像-文本配对数据集RS5M，其中包含了500万张带有英文描述的RS图像。该数据集是通过过滤公开可用的图像-文本配对数据集，并使用预训练的视觉-语言基础模型为仅带标签的RS数据集生成标题。这是第一个大规模的RS图像-文本配对数据集。

    Pre-trained Vision-Language Foundation Models utilizing extensive image-text paired data have demonstrated unprecedented image-text association capabilities, achieving remarkable results across various downstream tasks. A critical challenge is how to make use of existing large-scale pre-trained VLMs, which are trained on common objects, to perform the domain-specific transfer for accomplishing domain-related downstream tasks. In this paper, we propose a new framework that includes the Domain Foundation Model (DFM), bridging the gap between the General Foundation Model (GFM) and domain-specific downstream tasks. Moreover, we present an image-text paired dataset in the field of remote sensing (RS), RS5M, which has 5 million RS images with English descriptions. The dataset is obtained from filtering publicly available image-text paired datasets and captioning label-only RS datasets with pre-trained VLM. These constitute the first large-scale RS image-text paired dataset. Additionally, we 
    
[^7]: 语言模型是有限实用说话者

    Language Models are Bounded Pragmatic Speakers. (arXiv:2305.17760v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.17760](http://arxiv.org/abs/2305.17760)

    本文提出了一个概率认知模型，称为有限实用说话者，用于表征不同变体的语言模型的操作方式。经过人类反馈的强化学习微调的大型语言模型具有概念上类似于 快与慢思考模型的思维模型，而这种思维模型被归因于人类。此研究凸显了采用认知概率建模方法对语言模型的理解、评估和推进的价值。

    

    本文提出了一个概率认知模型，称为有限实用说话者，用于表征不同变体的语言模型的操作方式。特别地，我们展示了经过人类反馈的强化学习微调的大型语言模型（Ouyang等人，2022）具有概念上类似于 快与慢思考模型（Kahneman，2011）的思维模型，而这种思维模型被心理学家们归因于人类。我们讨论了从人类反馈中的强化学习作为快与慢思考模型的局限性，并提出了扩展这个框架的途径。本研究实质上凸显了采用认知概率建模方法来获得对语言模型的理解、评估和推进方面的深刻见解的价值。

    How do language models "think"? This paper formulates a probabilistic cognitive model called the bounded pragmatic speaker, which can characterize the operation of different variations of language models. Specifically, we demonstrate that large language models fine-tuned with reinforcement learning from human feedback (Ouyang et al., 2022) embody a model of thought that conceptually resembles a fast-and-slow model (Kahneman, 2011), which psychologists have attributed to humans. We discuss the limitations of reinforcement learning from human feedback as a fast-and-slow model of thought and propose avenues for expanding this framework. In essence, our research highlights the value of adopting a cognitive probabilistic modeling approach to gain insights into the comprehension, evaluation, and advancement of language models.
    
[^8]: ArtGPT-4: 基于适配器增强的MiniGPT-4模型的艺术视觉语言理解

    ArtGPT-4: Artistic Vision-Language Understanding with Adapter-enhanced MiniGPT-4. (arXiv:2305.07490v1 [cs.CL])

    [http://arxiv.org/abs/2305.07490](http://arxiv.org/abs/2305.07490)

    ArtGPT-4是一种基于适配器增强的MiniGPT-4模型，专注于解决图像理解方面的问题，能够在短时间内训练出具备良好视觉语言理解能力的多模态模型。

    

    近年来，大型语言模型在自然语言处理领域取得了显著进展，比如ChatGPT和GPT-4等模型在多种语言任务上取得了惊人的能力。但是，对这样的大规模模型进行训练是具有挑战性的，而找到与模型规模匹配的数据集通常也很困难。微调和使用新方法训练参数较少的模型已经成为克服这些挑战的有效方法。MiniGPT-4模型便是其中之一，该模型通过运用新颖的预训练模型和革新性的培训策略实现了与GPT-4相当的视觉语言理解能力。但是，该模型在图像理解方面仍然面临一些挑战，特别是在艺术图片方面。ArtGPT-4是一种新型的多模态模型，旨在应对这些局限。ArtGPT-4使用Tesla A100设备对图像-文本对进行训练，仅用了约200GB的数据，在2小时内就能展示出图像。

    In recent years, large language models (LLMs) have made significant progress in natural language processing (NLP), with models like ChatGPT and GPT-4 achieving impressive capabilities in various linguistic tasks. However, training models on such a large scale is challenging, and finding datasets that match the model's scale is often difficult. Fine-tuning and training models with fewer parameters using novel methods have emerged as promising approaches to overcome these challenges. One such model is MiniGPT-4, which achieves comparable vision-language understanding to GPT-4 by leveraging novel pre-training models and innovative training strategies. However, the model still faces some challenges in image understanding, particularly in artistic pictures. A novel multimodal model called ArtGPT-4 has been proposed to address these limitations. ArtGPT-4 was trained on image-text pairs using a Tesla A100 device in just 2 hours, using only about 200 GB of data. The model can depict images wit
    
[^9]: 音乐结构的自组织网络分析

    In-depth analysis of music structure as a self-organized network. (arXiv:2303.13631v1 [cs.SD])

    [http://arxiv.org/abs/2303.13631](http://arxiv.org/abs/2303.13631)

    本文介绍了一种利用Essential Element Network (EEN)算法将音频编码成文本并进行相关性计算和优化应用于聚类系数的频率和排名的方法，得到了音乐的深层结构信息，为厘清音乐结构提供了新方法。

    

    自然语言中的词汇不仅传递信息，还随着文明和人类迁移而演变。音乐也是如此。为了理解音乐背后的复杂结构，我们引入了一个叫做Essential Element Network (EEN)的算法将音频编码成文本。该网络通过计算音调、时间和音量之间的相关性得到，通过优化EEN算法以生成Zipf定律应用于聚类系数的频率和排名，我们可以将语义关系视为词汇并生成它们的映射。我们将这些编码后的词汇映射到音调-时间空间中，有助于我们系统地组织音乐深层结构中的句法。相比于其他深度学习方法的黑盒子特性，我们的算法提供了对音乐背后复杂网络的精确描述。因此，这些过程积累的经验和属性不仅为此类应用提供了新的方法，同时也为许多其他相关领域的研究提供了探索的路径。

    Words in a natural language not only transmit information but also evolve with the development of civilization and human migration. The same is true for music. To understand the complex structure behind the music, we introduced an algorithm called the Essential Element Network (EEN) to encode the audio into text. The network is obtained by calculating the correlations between scales, time, and volume. Optimizing EEN to generate Zipfs law for the frequency and rank of the clustering coefficient enables us to generate and regard the semantic relationships as words. We map these encoded words into the scale-temporal space, which helps us organize systematically the syntax in the deep structure of music. Our algorithm provides precise descriptions of the complex network behind the music, as opposed to the black-box nature of other deep learning approaches. As a result, the experience and properties accumulated through these processes can offer not only a new approach to the applications of
    

