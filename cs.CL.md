# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfare](https://arxiv.org/abs/2606.26104) | 研究发现，在微调语言模型时，使用断言式确定性、道德词汇和情感词汇等特征会显著增强模型对动物福利的支持，而模糊语言和具体感官描述则会削弱这一立场。 |
| [^2] | [Helpfulness Hurts: Domain-Dependent Degradation of Mid-Trained Compassion Values Under Post-Training](https://arxiv.org/abs/2606.26102) | 论文发现，后训练中的有益性训练（如SFT和GRPO）会显著削弱语言模型在中期训练中获得的动物同情价值观，而编码训练的影响较小，且这一现象在多个数据集和训练范式上得到验证。 |
| [^3] | [Small edits, large models: How Wikipedia advocacy shapes LLM values](https://arxiv.org/abs/2606.24890) | 本文发现，一小群维基百科编辑者通过少量有针对性的编辑，就能显著影响大语言模型对特定主题（如动物福利）的价值观输出，揭示了众包内容对AI系统行为的塑造能力。 |
| [^4] | [HiQA: A Hierarchical Contextual Augmentation RAG for Massive Documents QA](https://arxiv.org/abs/2402.01767) | HiQA是一个先进的多文档问答框架，使用分层的上下文增强和多路径检索机制，解决了大规模文档问答中的检索准确性问题，并在多文档环境中展示了最先进的性能。 |

# 详细

[^1]: 断言，而非描述：改变大语言模型对动物福利推理的语言特征

    Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfare

    [https://arxiv.org/abs/2606.26104](https://arxiv.org/abs/2606.26104)

    研究发现，在微调语言模型时，使用断言式确定性、道德词汇和情感词汇等特征会显著增强模型对动物福利的支持，而模糊语言和具体感官描述则会削弱这一立场。

    

    arXiv:2606.26104v1 公告类型：交叉 摘要：动物福利倡导者撰写了大量文本，而这些文本越来越多地用于训练语言模型，随后数百万用户会向这些模型询问动物福利问题。通过在一个预留的动物福利基准测试上使用词汇匹配的立场对比探针，我们测量了十种语言特征分别作为微调数据使用时，如何改变Llama-3.2-1B对支持动物福利推理的偏好。其中八种特征产生了统计上显著的转变。七种特征使模型朝向更强的支持动物福利推理方向移动：断言式确定性、明确的道德词汇、情感词汇、评价性主张、叙事结构、描绘的伤害严重程度以及即时时间框架。两种特征使模型朝相反方向移动：模糊语言和具体感官描述都削弱了支持动物福利的立场。第一人称视角没有统计上显著的影响。对于任何撰写旨在影响语言模型的动物福利文本的人来说，实际建议是使用断言式确定性、明确的道德词汇和情感词汇，同时避免模糊语言和具体感官描述。

    arXiv:2606.26104v1 Announce Type: cross  Abstract: Animal-welfare advocates produce a lot of writing, and increasingly that writing trains the language models that millions of people then ask about animal welfare. Using vocabulary-matched stance-contrast probes on a held-out animal-welfare benchmark, we measure how each of ten linguistic features changes Llama-3.2-1B's preference for pro-animal-welfare reasoning when used as fine-tuning data. Eight of the ten features produce statistically significant shifts. Seven move the model toward stronger pro-animal-welfare reasoning: assertive certainty, explicit moral vocabulary, emotion words, evaluative claims, narrative structure, depicted harm severity, and immediate temporal framing. Two move it the other way: hedged language and concrete sensory description both dilute the pro-animal-welfare stance. First-person perspective has no statistically significant effect. The practical recommendation for anyone writing animal-welfare text that m
    
[^2]: 有益性有害：后训练中基于领域的中期训练同情价值观退化

    Helpfulness Hurts: Domain-Dependent Degradation of Mid-Trained Compassion Values Under Post-Training

    [https://arxiv.org/abs/2606.26102](https://arxiv.org/abs/2606.26102)

    论文发现，后训练中的有益性训练（如SFT和GRPO）会显著削弱语言模型在中期训练中获得的动物同情价值观，而编码训练的影响较小，且这一现象在多个数据集和训练范式上得到验证。

    

    摘要：标准后训练流程采用监督微调（SFT）和强化学习（RL）来使语言模型具有有益性，但这些过程可能会无意中削弱预训练期间灌输的价值观。我们研究了后训练数据的领域是否会对基于同情导向的合成数据进行中期训练的Llama 3.1 8B模型中的动物同情价值观保持产生差异化影响，使用了SFT（通过Dolly-15k的有益性与通过Magicoder-110K的编码）和GRPO（通过RLHFlow的有益性与通过Magicoder的编码），并在动物伤害基准（AHB 2.2）和不确定性下的道德推理基准（MORU）上进行评估。与编码训练相比，有益性训练在AHB上显著降低了动物同情（SFT：35.7%对65.2%；GRPO：18.7%对32.0%），这一结果在两个独立的有益性数据集和两种训练范式上得到复现。在英文MORU项目中，有益性训练也降低了通用道德推理。

    arXiv:2606.26102v1 Announce Type: cross  Abstract: Standard post-training pipelines apply supervised fine-tuning (SFT) and reinforcement learning (RL) to make language models helpful, but these processes may inadvertently degrade values instilled during pre-training. We investigate whether the domain of post-training data differentially affects the retention of animal compassion values in a Llama 3.1 8B model mid-trained on compassion-oriented synthetic data, using both SFT (helpfulness via Dolly-15k vs. coding via Magicoder-110K) and GRPO (helpfulness via RLHFlow vs. coding via Magicoder), evaluated on the Animal Harm Benchmark (AHB 2.2) and MORU benchmark (Moral Reasoning Under Uncertainty). Helpfulness training significantly degrades animal compassion relative to coding training on AHB (SFT: 35.7% vs. 65.2%; GRPO: 18.7% vs. 32.0%), replicating across two independent helpfulness datasets and two training paradigms. On English MORU items, helpfulness training degrades general moral re
    
[^3]: 微小编辑，巨大模型：维基百科倡导如何塑造大语言模型的价值观

    Small edits, large models: How Wikipedia advocacy shapes LLM values

    [https://arxiv.org/abs/2606.24890](https://arxiv.org/abs/2606.24890)

    本文发现，一小群维基百科编辑者通过少量有针对性的编辑，就能显著影响大语言模型对特定主题（如动物福利）的价值观输出，揭示了众包内容对AI系统行为的塑造能力。

    

    arXiv:2606.24890v2 公告类型：替换交叉 摘要：一小群志愿者仅通过编辑维基百科，就能塑造AI系统讨论动物福利的方式吗？我们证明他们可以。维基百科几乎出现在每个主要语言模型训练数据集中，并且其权重高于网络爬取文本。亲动物维基百科编辑者（PAW）是一群倡导者，他们在相关文章中添加强调动物福利的内容，共对115个页面进行了125次编辑。利用基于梯度的数据归因方法（Bergson；MAGIC），我们追踪了这些编辑如何影响语言模型行为。在Llama 3.1 8B上进行的TrackStar检索归因发现，对于动物福利查询，PAW编辑的部分占最高归因文档的68%（p < 0.0001），而对于关于同一公司的不相关查询，这一比例仅为52%（p = 0.53）：模型将PAW内容特别关联到动物福利主题，而非一般实体。在Llama-3.2-1B上进行的MAGIC反事实影响估计表明，这些编辑显著改变了模型对相关问题的回应。

    arXiv:2606.24890v2 Announce Type: replace-cross  Abstract: Can a small group of volunteers shape how AI systems discuss animal welfare, just by editing Wikipedia? We show that they can. Wikipedia appears in nearly every major language model training dataset and is weighted more heavily than web-crawled text. The Pro-Animal Wikipedians (PAW), a group of advocates who add sourced animal welfare content to relevant articles, have made 125 edits across 115 pages. Using gradient-based data attribution (Bergson; MAGIC), we traced how these edits influence language model behavior. TrackStar retrieval attribution on Llama 3.1 8B found that PAW-edited sections made up 68 percent of the highest-attributed documents for animal welfare queries (p < 0.0001) but only 52 percent for unrelated queries about the same companies (p = 0.53): the model links PAW content specifically to animal welfare topics, not to the entities in general. MAGIC counterfactual influence estimation on Llama-3.2-1B, run acro
    
[^4]: HiQA：一种用于大规模文档问答的分层上下文增强的RAG模型

    HiQA: A Hierarchical Contextual Augmentation RAG for Massive Documents QA

    [https://arxiv.org/abs/2402.01767](https://arxiv.org/abs/2402.01767)

    HiQA是一个先进的多文档问答框架，使用分层的上下文增强和多路径检索机制，解决了大规模文档问答中的检索准确性问题，并在多文档环境中展示了最先进的性能。

    

    随着利用外部工具的语言模型代理迅速发展，使用补充文档和检索增强生成（RAG）方法的问答（QA）方法学取得了重要进展。这种进步提高了语言模型的回答质量，并减轻了幻觉的出现。然而，当面临大量无法区分的文档时，这些方法在检索准确性方面表现有限，给实际应用带来了显著挑战。针对这些新兴的挑战，我们提出了HiQA，这是一个先进的多文档问答（MDQA）框架，将级联的元数据整合到内容中，同时具备多路径检索机制。我们还发布了一个名为MasQA的基准来评估和研究MDQA。最后，HiQA在多文档环境中展示了最先进的性能。

    As language model agents leveraging external tools rapidly evolve, significant progress has been made in question-answering(QA) methodologies utilizing supplementary documents and the Retrieval-Augmented Generation (RAG) approach. This advancement has improved the response quality of language models and alleviates the appearance of hallucination. However, these methods exhibit limited retrieval accuracy when faced with massive indistinguishable documents, presenting notable challenges in their practical application. In response to these emerging challenges, we present HiQA, an advanced framework for multi-document question-answering (MDQA) that integrates cascading metadata into content as well as a multi-route retrieval mechanism. We also release a benchmark called MasQA to evaluate and research in MDQA. Finally, HiQA demonstrates the state-of-the-art performance in multi-document environments.
    

