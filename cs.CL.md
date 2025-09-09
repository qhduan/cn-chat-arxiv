# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Repetition Improves Language Model Embeddings](https://arxiv.org/abs/2402.15449) | 回声嵌入方法通过重复输入来提取信息，解决了自回归模型无法包含后续令牌信息的限制，实验结果表明其能够最大程度充分利用高质量的语言模型进行嵌入。 |
| [^2] | [AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling](https://arxiv.org/abs/2402.12226) | AnyGPT是一个统一的多模态语言模型，通过离散表示实现各种模态的统一处理，能够在不改变大型语言模型架构或训练方式的情况下稳定训练，为新模态的无缝整合提供了可能。 |
| [^3] | [Grammaticality illusion or ambiguous interpretation? Event-related potentials reveal the nature of the missing-NP effect in Mandarin centre-embedded structures](https://arxiv.org/abs/2402.11282) | 汉语中缺失NP的双重中心嵌套结构并不是语法性错觉，而是动词歧义解释的含糊解释。 |
| [^4] | [Support or Refute: Analyzing the Stance of Evidence to Detect Out-of-Context Mis- and Disinformation.](http://arxiv.org/abs/2311.01766) | 本研究提出了一种基于多模态证据的立场抽取网络（SEN）来检测上下文错误的误导信息。通过考虑不同证据的立场，我们提供了一种更准确的检测方法，并引入了基于共现关系的支持-反驳分数。这种方法在公共大规模数据上进行的实验证明了其有效性。 |
| [^5] | [Multiple Noises in Diffusion Model for Semi-Supervised Multi-Domain Translation.](http://arxiv.org/abs/2309.14394) | 本文提出了一种多噪声扩散模型（MDD）用于半监督多域翻译，通过引入噪声级别来对缺失的域进行建模，实现了任意域之间的翻译而不需要训练单独的模型。 |

# 详细

[^1]: 重复改善语言模型嵌入

    Repetition Improves Language Model Embeddings

    [https://arxiv.org/abs/2402.15449](https://arxiv.org/abs/2402.15449)

    回声嵌入方法通过重复输入来提取信息，解决了自回归模型无法包含后续令牌信息的限制，实验结果表明其能够最大程度充分利用高质量的语言模型进行嵌入。

    

    最近改进从自回归大型语言模型（LLMs）中提取文本嵌入的方法主要集中在改进数据、骨干预训练语言模型或通过指令改进任务差异化上。在这项工作中，我们解决了自回归模型的一个架构限制：令牌嵌入不能包含来自输入中后续令牌的信息。为了解决这一限制，我们提出了一种简单的方法，“回声嵌入”，其中我们在上下文中将输入重复两次，并从第二次出现中提取嵌入。我们展示了早期令牌的回声嵌入可以编码关于后续令牌的信息，从而使我们能够最大程度地利用高质量的LLMs进行嵌入。在MTEB排行榜上，回声嵌入在零射击中比经典嵌入提高了超过9%，在微调时提高了约0.7%。使用Mistral-7B模型的回声嵌入实现了与当前最先进模型的比较。

    arXiv:2402.15449v1 Announce Type: new  Abstract: Recent approaches to improving the extraction of text embeddings from autoregressive large language models (LLMs) have largely focused on improvements to data, backbone pretrained language models, or improving task-differentiation via instructions. In this work, we address an architectural limitation of autoregressive models: token embeddings cannot contain information from tokens that appear later in the input. To address this limitation, we propose a simple approach, "echo embeddings," in which we repeat the input twice in context and extract embeddings from the second occurrence. We show that echo embeddings of early tokens can encode information about later tokens, allowing us to maximally leverage high-quality LLMs for embeddings. On the MTEB leaderboard, echo embeddings improve over classical embeddings by over 9% zero-shot and by around 0.7% when fine-tuned. Echo embeddings with a Mistral-7B model achieve state-of-the-art compared
    
[^2]: AnyGPT：统一的多模式离散序列建模语言模型

    AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling

    [https://arxiv.org/abs/2402.12226](https://arxiv.org/abs/2402.12226)

    AnyGPT是一个统一的多模态语言模型，通过离散表示实现各种模态的统一处理，能够在不改变大型语言模型架构或训练方式的情况下稳定训练，为新模态的无缝整合提供了可能。

    

    我们介绍了 AnyGPT，这是一个任意多模式语言模型，利用离散表示统一处理各种模态，包括语音、文本、图像和音乐。AnyGPT 可以稳定训练，无需对当前大型语言模型（LLM）架构或训练范式进行任何改动。相反，它仅依赖于数据级预处理，促进了新模态的无缝集成到LLM中，类似于新语言的整合。我们构建了一个多模式文本中心的数据集，用于多模式对齐预训练。利用生成模型，我们合成了第一个大规模任意多模式指令数据集。它包括108k个多轮对话示例，精细地交织各种模态，从而使模型能够处理多模态输入和输出的任意组合。实验结果表明，AnyGPT能够促进...

    arXiv:2402.12226v1 Announce Type: cross  Abstract: We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages. We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs. Experimental results demonstrate that AnyGPT is capable of facilitat
    
[^3]: 中心嵌套结构中的缺失NP效应的性质：事件相关电位揭示晕眩错觉还是含糊解释？

    Grammaticality illusion or ambiguous interpretation? Event-related potentials reveal the nature of the missing-NP effect in Mandarin centre-embedded structures

    [https://arxiv.org/abs/2402.11282](https://arxiv.org/abs/2402.11282)

    汉语中缺失NP的双重中心嵌套结构并不是语法性错觉，而是动词歧义解释的含糊解释。

    

    在几种语言中，在双重中心嵌套结构中省略动词短语（VP）会产生一个语法性错觉。类似的错觉也表现在汉语缺失NP的双重中心嵌套结构中。然而，关于它的本质尚无共识。我们认为，与其把它看作是语法性错觉，不如将动词的歧义解释视为最能解释汉语中这一现象的方式。为了进一步支持这一假设，我们在减少复杂度的情况下进行了两项与自嵌入关系从句放置在句子主语位置相结合相近双中心嵌套结构的脑电图（EEG）实验。实验1表明，在这种结构中同样会表现出类似的现象，证据是缺少P600效应而存在N400效应。在实验2中，通过提供语义线索以减少歧义，消除了这种错觉，证据是存在P600效应。我们解释了这些结果

    arXiv:2402.11282v1 Announce Type: new  Abstract: In several languages, omitting a verb phrase (VP) in double centre-embedded structures creates a grammaticality illusion. Similar illusion also exhibited in Mandarin missing-NP double centre-embedded structures. However, there is no consensus on its very nature. Instead of treating it as grammaticality illusion, we argue that ambiguous interpretations of verbs can best account for this phenomenon in Mandarin. To further support this hypothesis, we conducted two electroencephalography (EEG) experiments on quasi double centre-embedded structures whose complexity is reduced by placing the self-embedding relative clauses into the sentence's subject position. Experiment 1 showed that similar phenomenon even exhibited in this structure, evidenced by an absence of P600 effect and a presence of N400 effect. In Experiment 2, providing semantic cues to reduce ambiguity dispelled this illusion, as evidenced by a P600 effect. We interpret the result
    
[^4]: 支持还是反驳：分析证据立场以检测上下文错误的误导信息

    Support or Refute: Analyzing the Stance of Evidence to Detect Out-of-Context Mis- and Disinformation. (arXiv:2311.01766v1 [cs.CL])

    [http://arxiv.org/abs/2311.01766](http://arxiv.org/abs/2311.01766)

    本研究提出了一种基于多模态证据的立场抽取网络（SEN）来检测上下文错误的误导信息。通过考虑不同证据的立场，我们提供了一种更准确的检测方法，并引入了基于共现关系的支持-反驳分数。这种方法在公共大规模数据上进行的实验证明了其有效性。

    

    在线误导信息已经成为一个国家级的社会问题，是各种在线伤害的主要来源之一。其中一种常见的误导信息形式是上下文错误（OOC）信息，其中不同的信息被错误地关联起来，例如真实图像与虚假的文本标题或误导性的文本描述。尽管一些研究试图通过外部证据来抵御上下文错误的误导信息，但它们往往忽视了不同立场的不同证据的作用。受到证据立场代表不同检测结果的偏见的启发，我们提出了一种能够在统一框架中提取多模态证据的立场的立场抽取网络（SEN）。此外，我们还引入了基于命名实体的共现关系计算的支持-反驳分数到文本SEN中。对公共大规模数据的大量实验证明了我们的方法的有效性。

    Mis- and disinformation online have become a major societal problem as major sources of online harms of different kinds. One common form of mis- and disinformation is out-of-context (OOC) information, where different pieces of information are falsely associated, e.g., a real image combined with a false textual caption or a misleading textual description. Although some past studies have attempted to defend against OOC mis- and disinformation through external evidence, they tend to disregard the role of different pieces of evidence with different stances. Motivated by the intuition that the stance of evidence represents a bias towards different detection results, we propose a stance extraction network (SEN) that can extract the stances of different pieces of multi-modal evidence in a unified framework. Moreover, we introduce a support-refutation score calculated based on the co-occurrence relations of named entities into the textual SEN. Extensive experiments on a public large-scale data
    
[^5]: 多噪声扩散模型用于半监督多域翻译

    Multiple Noises in Diffusion Model for Semi-Supervised Multi-Domain Translation. (arXiv:2309.14394v1 [cs.CL])

    [http://arxiv.org/abs/2309.14394](http://arxiv.org/abs/2309.14394)

    本文提出了一种多噪声扩散模型（MDD）用于半监督多域翻译，通过引入噪声级别来对缺失的域进行建模，实现了任意域之间的翻译而不需要训练单独的模型。

    

    域间翻译涉及在给定源域条件下生成目标域样本。大多数现有方法都集中在固定的输入和输出域上，即它们仅适用于特定的配置（例如对于两个域，要么$D_1\rightarrow{}D_2$，要么$D_2\rightarrow{}D_1$）。本文提出了Multi-Domain Diffusion（MDD）方法，这是一种用于半监督多域翻译的条件扩散框架。与以往的方法不同，MDD不需要定义输入和输出域，允许在一组域的任何分区之间进行翻译（例如$(D_1, D_2)\rightarrow{}D_3$，$D_2\rightarrow{}(D_1, D_3)$，$D_3\rightarrow{}D_1$等），而无需为每个域配置训练单独的模型。MDD的关键思想是利用扩散模型的噪声形式，通过为每个域引入一个噪声级别，以自然的方式对缺失的域进行建模。这将传统的翻译问题转化为一个通过噪声建模来解决的问题。

    Domain-to-domain translation involves generating a target domain sample given a condition in the source domain. Most existing methods focus on fixed input and output domains, i.e. they only work for specific configurations (i.e. for two domains, either $D_1\rightarrow{}D_2$ or $D_2\rightarrow{}D_1$). This paper proposes Multi-Domain Diffusion (MDD), a conditional diffusion framework for multi-domain translation in a semi-supervised context. Unlike previous methods, MDD does not require defining input and output domains, allowing translation between any partition of domains within a set (such as $(D_1, D_2)\rightarrow{}D_3$, $D_2\rightarrow{}(D_1, D_3)$, $D_3\rightarrow{}D_1$, etc. for 3 domains), without the need to train separate models for each domain configuration. The key idea behind MDD is to leverage the noise formulation of diffusion models by incorporating one noise level per domain, which allows missing domains to be modeled with noise in a natural way. This transforms the tra
    

