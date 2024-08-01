# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Plan for Language Modeling from Unlabeled Data](https://arxiv.org/abs/2404.00614) | 通过自监督学习目标训练一个用于规划未来写作过程的模块，扩展了成功的语言模型公式到更抽象的规划中，改善了语言建模的性能，特别是在文本结构方面，同时新的规划模块可以大规模训练并轻松与社区共享。 |
| [^2] | [FrameQuant: Flexible Low-Bit Quantization for Transformers](https://arxiv.org/abs/2403.06082) | 提出一种简单的方案，通过融合框架将Transformer模型量化为仅两位，仅有轻微精度下降。 |
| [^3] | [Designing Informative Metrics for Few-Shot Example Selection](https://arxiv.org/abs/2403.03861) | 提出了一种基于复杂度的提示选择方法，用于将示例与测试句子的句法-语义复杂度对齐，在少样本NER任务中取得了显著的性能提升。 |
| [^4] | [Human vs. Machine: Language Models and Wargames](https://arxiv.org/abs/2403.03407) | 人工智能大型语言模型在战争游戏中与人类响应存在一致性，但也存在显著的差异，这表明在政策制定者交出自主权或听从基于AI的战略建议之前应谨慎对待。 |
| [^5] | [Training Language Model Agents without Modifying Language Models](https://arxiv.org/abs/2402.11359) | 提出一种新的方法，在不修改语言模型的情况下训练语言模型代理，通过进化代理的功能来解决下游任务 |
| [^6] | [AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models.](http://arxiv.org/abs/2401.09002) | 本研究提出一种新方法评估大型语言模型上越狱攻击效果，引入粗粒度和细粒度评估框架，提供了更全面和细致的评估角度，并开发了专门的真实数据集作为基准，为未来研究建立了基础资源。 |
| [^7] | [LLM in a flash: Efficient Large Language Model Inference with Limited Memory.](http://arxiv.org/abs/2312.11514) | 本文提出了一种在有限内存条件下高效运行大型语言模型的方法，通过将模型参数存储在闪存中并按需传输到DRAM的方式来解决内存限制的挑战。该方法通过构建推理成本模型并优化数据传输和读取方式，引入了窗口化和行列绑定两种主要技术。 |
| [^8] | [LLMs may Dominate Information Access: Neural Retrievers are Biased Towards LLM-Generated Texts.](http://arxiv.org/abs/2310.20501) | 近期的研究发现，大型语言模型（LLMs）对信息检索系统产生了一种偏见，倾向于将LLM生成的文档排名较高。这种“来源偏见”可能对信息访问产生重大影响。 |
| [^9] | [CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models.](http://arxiv.org/abs/2310.08753) | CompA提出了由两个专家注释的音频-语言模型组合推理基准数据集，用于评估ALMs在理解音频中声音事件的顺序和属性绑定方面的表现。 |
| [^10] | [Characterizing Learning Curves During Language Model Pre-Training: Learning, Forgetting, and Stability.](http://arxiv.org/abs/2308.15419) | 本研究通过从五个英语语言模型预训练运行中提取学习曲线，揭示了语言模型在预训练期间的学习过程。结果表明，语言模型在学习生成更长、更连贯的文本之前，会生成短而重复的短语。同时，频繁出现的标记在预训练过程中更早学习，具有更小的变异性，并且很少被遗忘。较短、更频繁的上下文与稳定和快速获得的预测有关。词类的影响较小，但名词倾向于较晚获得且遗忘率较低。 |
| [^11] | [SILO Language Models: Isolating Legal Risk In a Nonparametric Datastore.](http://arxiv.org/abs/2308.04430) | SILO是一种新的语言模型，通过在推理过程中对非参数化的数据存储进行查询，实现在面临法律风险和模型性能之间的权衡，并支持数据归属和数据生产者退出模型的功能。 |

# 详细

[^1]: 从未标记数据中学习语言建模规划

    Learning to Plan for Language Modeling from Unlabeled Data

    [https://arxiv.org/abs/2404.00614](https://arxiv.org/abs/2404.00614)

    通过自监督学习目标训练一个用于规划未来写作过程的模块，扩展了成功的语言模型公式到更抽象的规划中，改善了语言建模的性能，特别是在文本结构方面，同时新的规划模块可以大规模训练并轻松与社区共享。

    

    通过训练来预测未标记语料库中的下一个标记，大型语言模型学会执行许多任务，而无需任何标记数据。然而，它们的下一个标记预测目标可以说限制了它们在需要规划的场景中的性能，比如写作一篇连贯的文章。在这篇论文中，我们通过自监督学习目标训练一个用于规划未来写作过程的模块。通过根据生成的潜在计划进行条件化，我们的模型以无监督的方式将成功的语言模型公式扩展到更抽象的规划中。实验上，我们证明了我们的方法在一般情况下改善了语言建模的性能，特别是在文本结构方面。由于我们的框架使用的是无监督且外部于语言模型的规划模块，因此新的规划模块可以大规模训练，并且能够轻松地与社区共享。

    arXiv:2404.00614v1 Announce Type: cross  Abstract: By training to predict the next token in an unlabeled corpus, large language models learn to perform many tasks without any labeled data. However, their next-token-prediction objective arguably limits their performance in scenarios that require planning, such as writing a coherent article. In this paper, we train a module for planning the future writing process via a self-supervised learning objective. By conditioning on generated latent plans, our model extends the successful language model formula to more abstract planning in an unsupervised way. Empirically, we demonstrate that our method improves language modeling performance in general, particularly with respect to the text structure. Because our framework uses a planner module that is unsupervised and external to the language model, new planner modules can be trained at large scale and easily be shared with the community.
    
[^2]: FrameQuant: Transformer的灵活低比特量化方法

    FrameQuant: Flexible Low-Bit Quantization for Transformers

    [https://arxiv.org/abs/2403.06082](https://arxiv.org/abs/2403.06082)

    提出一种简单的方案，通过融合框架将Transformer模型量化为仅两位，仅有轻微精度下降。

    

    Transformer是许多视觉和自然语言处理任务强大基础模型的支柱。然而，它们的计算和内存/存储空间占用较大，因此为这些模型提供服务往往需要昂贵的高端硬件。为了缓解这一困难，后训练量化试图修改预训练模型并将其量化为八位或更低的位数，显着提高计算/内存/延迟效率。既可以成功将这些模型量化为四位，但性能有所损失。在这项工作中，我们概述了一个简单的方案，将基于Transformer的模型量化为仅两位（加一些额外开销），仅会有轻微的精度下降。我们的制定关键在于从谐波分析中借鉴了一种称为融合框架的概念。我们的主要发现是，量化不应该在原始权重空间中进行，而是应该在融合框架表示中进行。

    arXiv:2403.06082v1 Announce Type: cross  Abstract: Transformers are the backbone of powerful foundation models for many Vision and Natural Language Processing tasks. But their compute and memory/storage footprint is large, and so, serving such models is expensive often requiring high-end hardware. To mitigate this difficulty, Post-Training Quantization seeks to modify a pre-trained model and quantize it to eight bits or lower, significantly boosting compute/memory/latency efficiency. Such models have been successfully quantized to four bits with some performance loss. In this work, we outline a simple scheme to quantize Transformer-based models to just two bits (plus some overhead) with only a small drop in accuracy. Key to our formulation is a concept borrowed from Harmonic analysis called Fusion Frames. Our main finding is that the quantization must take place not in the original weight space, but instead in the Fusion Frame representations. If quantization is interpreted as the addi
    
[^3]: 为少样本示例选择设计信息度量

    Designing Informative Metrics for Few-Shot Example Selection

    [https://arxiv.org/abs/2403.03861](https://arxiv.org/abs/2403.03861)

    提出了一种基于复杂度的提示选择方法，用于将示例与测试句子的句法-语义复杂度对齐，在少样本NER任务中取得了显著的性能提升。

    

    预训练语言模型（PLMs）在提供适当格式的示例时展现出了卓越的少样本学习能力。然而，选择“最佳”示例仍然是一个未解决的挑战。我们提出了一种基于复杂度的提示选择方法，适用于序列标注任务。该方法避免了训练一个专门用于选择示例的模型，而是使用特定的度量标准来对齐测试句子和示例的句法-语义复杂度。我们使用句子和单词级别的度量标准，将示例的复杂度与考虑中的（测试）句子进行匹配。我们的结果表明，我们的方法能够从PLMs中提取出更好的性能：在少样本NER上实现了最先进的性能，在CoNLL2003数据集上对GPT-4的F1分数实现了5%的绝对改善。我们还在像GPT-j-6B这样的较小模型中看到了高达28.85个点（F1/Acc.）的显著增益。

    arXiv:2403.03861v1 Announce Type: new  Abstract: Pretrained language models (PLMs) have shown remarkable few-shot learning capabilities when provided with properly formatted examples. However, selecting the "best" examples remains an open challenge. We propose a complexity-based prompt selection approach for sequence tagging tasks. This approach avoids the training of a dedicated model for selection of examples, and instead uses certain metrics to align the syntactico-semantic complexity of test sentences and examples. We use both sentence- and word-level metrics to match the complexity of examples to the (test) sentence being considered. Our results demonstrate that our approach extracts greater performance from PLMs: it achieves state-of-the-art performance on few-shot NER, achieving a 5% absolute improvement in F1 score on the CoNLL2003 dataset for GPT-4. We also see large gains of upto 28.85 points (F1/Acc.) in smaller models like GPT-j-6B.
    
[^4]: 人类对抗机器：语言模型与战争游戏

    Human vs. Machine: Language Models and Wargames

    [https://arxiv.org/abs/2403.03407](https://arxiv.org/abs/2403.03407)

    人工智能大型语言模型在战争游戏中与人类响应存在一致性，但也存在显著的差异，这表明在政策制定者交出自主权或听从基于AI的战略建议之前应谨慎对待。

    

    战争游戏在军事战略的发展和国家对威胁或攻击的响应中有着悠久的历史。人工智能（AI）的出现承诺了更好的决策制定和增强的军事效果。然而，关于AI系统，尤其是大型语言模型（LLMs），与人类的行为有何不同仍存在争议。为此，我们进行了一项战争游戏实验，共有107位国家安全专家人类参与者参与，旨在研究在一个虚构的美中情景中的危机升级，并比较人类参与者与LLM模拟响应之间的差异。我们发现LLM和人类响应存在显著一致性，但在战争游戏中模拟和人类参与者之间也存在显著的定量和定性差异，这促使决策者在交出自主权或遵循基于AI的战略建议之前谨慎对待。

    arXiv:2403.03407v1 Announce Type: cross  Abstract: Wargames have a long history in the development of military strategy and the response of nations to threats or attacks. The advent of artificial intelligence (AI) promises better decision-making and increased military effectiveness. However, there is still debate about how AI systems, especially large language models (LLMs), behave as compared to humans. To this end, we use a wargame experiment with 107 national security expert human players designed to look at crisis escalation in a fictional US-China scenario and compare human players to LLM-simulated responses. We find considerable agreement in the LLM and human responses but also significant quantitative and qualitative differences between simulated and human players in the wargame, motivating caution to policymakers before handing over autonomy or following AI-based strategy recommendations.
    
[^5]: 在不修改语言模型的情况下训练语言模型代理

    Training Language Model Agents without Modifying Language Models

    [https://arxiv.org/abs/2402.11359](https://arxiv.org/abs/2402.11359)

    提出一种新的方法，在不修改语言模型的情况下训练语言模型代理，通过进化代理的功能来解决下游任务

    

    研究人员和实践者最近已经将强大的大型语言模型（LLMs）重新定义为代理，使它们能够通过使用专门的功能自动化地完成复杂任务。为了促进LLM代理的发展，我们提出了一种在不修改LLM权重的情况下训练LLM代理的新范式，当LLM难以或无法进行修改时尤其有用。受到人类不断锻造工具以适应现实任务的启发，而不是改变我们的生物结构以适应一组静态工具，我们提出逐步锻造代理的功能，以更好地解决下游任务，而不是修改LLM权重。通过将这些功能视为可学习的“代理参数”并利用人工智能模型训练的基本思想，我们开发了AgentOptimizer，利用LLM更新代理的功能，并设计了一种代理训练算法

    arXiv:2402.11359v1 Announce Type: new  Abstract: Researchers and practitioners have recently reframed powerful Large Language Models (LLMs) as agents, enabling them to automate complex tasks largely via the use of specialized functions. To facilitate the development of LLM agents, we present a novel paradigm of training LLM agents without modifying the LLM weights, which is particularly useful when the LLMs are difficult or inaccessible for modifications. Inspired by how humans continuously forge tools to adapt to real-world tasks, rather than change our biological structure to fit a static set of tools, we propose to progressively forge agent's functions to better solve the downstream tasks instead of modifying the LLM weights. By treating the functions as learnable `agent parameters' and leveraging the fundamental idea of model training in artificial intelligence, we develop AgentOptimizer that employs the LLM to update agents' functions and devise an agent training algorithm with tw
    
[^6]: 评估大型语言模型上越狱攻击效果的方法研究

    AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models. (arXiv:2401.09002v1 [cs.CL])

    [http://arxiv.org/abs/2401.09002](http://arxiv.org/abs/2401.09002)

    本研究提出一种新方法评估大型语言模型上越狱攻击效果，引入粗粒度和细粒度评估框架，提供了更全面和细致的评估角度，并开发了专门的真实数据集作为基准，为未来研究建立了基础资源。

    

    在我们的研究中，我们开创性地提出了一种评估大型语言模型（LLMs）上越狱攻击效果的新方法，与传统的健壮性评估方法不同。我们的研究引入了两个不同的评估框架：粗粒度评估和细粒度评估。每个框架都使用从0到1的评分范围，提供了独特的视角，能够更全面和细致地评估攻击效果，并帮助攻击者更好地优化攻击提示。此外，我们还开发了一个专门用于越狱任务的全面的真实数据集。这个数据集不仅是我们当前研究的关键基准，也为未来研究建立了一个基础资源，可以在这个不断发展的领域中进行一致和比较的分析。通过与传统评估方法的精心比较，我们发现我们的评估方法与之相一致。

    In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation alig
    
[^7]: 闪存LLM：在有限内存下高效运行大型语言模型

    LLM in a flash: Efficient Large Language Model Inference with Limited Memory. (arXiv:2312.11514v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.11514](http://arxiv.org/abs/2312.11514)

    本文提出了一种在有限内存条件下高效运行大型语言模型的方法，通过将模型参数存储在闪存中并按需传输到DRAM的方式来解决内存限制的挑战。该方法通过构建推理成本模型并优化数据传输和读取方式，引入了窗口化和行列绑定两种主要技术。

    

    大型语言模型（LLM）在现代自然语言处理中起着至关重要的作用，在各种任务中表现出色。然而，它们庞大的计算和内存需求带来了挑战，特别是对于具有有限DRAM容量的设备而言。本文通过将模型参数存储在闪存中，并按需将其传输到DRAM的方式，解决了超过可用DRAM容量的LLM高效运行的挑战。我们的方法涉及构建一个考虑闪存特性的推理成本模型，引导我们在两个关键领域进行优化：减少从闪存传输的数据量，并以较大、更连续的块读取数据。在这个受硬件启发的框架内，我们引入了两个主要技术。首先，“窗口化”通过重复使用之前激活的神经元来策略性地减少数据传输，其次，“行列绑定”适应了闪存的顺序数据访问特点，

    Large language models (LLMs) are central to modern natural language processing, delivering exceptional performance in various tasks. However, their substantial computational and memory requirements present challenges, especially for devices with limited DRAM capacity. This paper tackles the challenge of efficiently running LLMs that exceed the available DRAM capacity by storing the model parameters in flash memory, but bringing them on demand to DRAM. Our method involves constructing an inference cost model that takes into account the characteristics of flash memory, guiding us to optimize in two critical areas: reducing the volume of data transferred from flash and reading data in larger, more contiguous chunks. Within this hardware-informed framework, we introduce two principal techniques. First, "windowing" strategically reduces data transfer by reusing previously activated neurons, and second, "row-column bundling", tailored to the sequential data access strengths of flash memory, 
    
[^8]: LLM可能主导信息访问：神经检索器对LLM生成的文本存在偏见。

    LLMs may Dominate Information Access: Neural Retrievers are Biased Towards LLM-Generated Texts. (arXiv:2310.20501v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.20501](http://arxiv.org/abs/2310.20501)

    近期的研究发现，大型语言模型（LLMs）对信息检索系统产生了一种偏见，倾向于将LLM生成的文档排名较高。这种“来源偏见”可能对信息访问产生重大影响。

    

    最近，大型语言模型（LLMs）的出现在信息检索（IR）应用，尤其是在网络搜索方面，彻底改变了范式。由于其在生成类人文本方面的卓越能力，LLMs在互联网上创造了大量的文本。因此，LLMs时代的IR系统面临一个新的挑战：索引的文档不仅是由人类撰写的，而且还包括由LLMs自动生成的文档。这些LLM生成的文档如何影响IR系统是一个紧迫且尚未探索的问题。在这项工作中，我们在涉及人类编写和LLM生成的文本的不同IR模型的场景中进行了定量评估。令人惊讶的是，我们的研究结果表明，神经检索模型倾向于将LLM生成的文档排名较高。我们将这种神经检索模型对LLM生成文本的偏见称为“来源偏见”。此外，我们发现这种偏见不仅限于f方相当的情况，而且在分类任务上也存在。

    Recently, the emergence of large language models (LLMs) has revolutionized the paradigm of information retrieval (IR) applications, especially in web search. With their remarkable capabilities in generating human-like texts, LLMs have created enormous texts on the Internet. As a result, IR systems in the LLMs era are facing a new challenge: the indexed documents now are not only written by human beings but also automatically generated by the LLMs. How these LLM-generated documents influence the IR systems is a pressing and still unexplored question. In this work, we conduct a quantitative evaluation of different IR models in scenarios where both human-written and LLM-generated texts are involved. Surprisingly, our findings indicate that neural retrieval models tend to rank LLM-generated documents higher. We refer to this category of biases in neural retrieval models towards the LLM-generated text as the \textbf{source bias}. Moreover, we discover that this bias is not confined to the f
    
[^9]: CompA: 解决音频-语言模型中的组合推理差距

    CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models. (arXiv:2310.08753v1 [cs.SD])

    [http://arxiv.org/abs/2310.08753](http://arxiv.org/abs/2310.08753)

    CompA提出了由两个专家注释的音频-语言模型组合推理基准数据集，用于评估ALMs在理解音频中声音事件的顺序和属性绑定方面的表现。

    

    音频的基本特性是其组合性。使用对比方法（例如CLAP）训练的音频-语言模型（ALMs）能够学习音频和语言模态之间的共享表示，从而在许多下游应用中提高性能，包括零样本音频分类、音频检索等。然而，这些模型在有效执行组合推理方面的能力还很少被探索，需要进一步的研究。本文提出了CompA，这是一个由两个专家注释的基准数据集，其中大多数是真实世界的音频样本，用于评估ALMs的组合推理能力。我们的CompA-order评估ALMs在理解音频中声音事件的顺序或发生时的表现如何，而CompA-attribute评估声音事件的属性绑定。每个基准数据集中的实例包含两个音频-标题对，其中两个音频具有相同的声音事件，但组合方式不同。

    A fundamental characteristic of audio is its compositional nature. Audio-language models (ALMs) trained using a contrastive approach (e.g., CLAP) that learns a shared representation between audio and language modalities have improved performance in many downstream applications, including zero-shot audio classification, audio retrieval, etc. However, the ability of these models to effectively perform compositional reasoning remains largely unexplored and necessitates additional research. In this paper, we propose CompA, a collection of two expert-annotated benchmarks with a majority of real-world audio samples, to evaluate compositional reasoning in ALMs. Our proposed CompA-order evaluates how well an ALM understands the order or occurrence of acoustic events in audio, and CompA-attribute evaluates attribute binding of acoustic events. An instance from either benchmark consists of two audio-caption pairs, where both audios have the same acoustic events but with different compositions. A
    
[^10]: 语言模型预训练期间学习曲线的特征化：学习、遗忘和稳定性

    Characterizing Learning Curves During Language Model Pre-Training: Learning, Forgetting, and Stability. (arXiv:2308.15419v1 [cs.CL])

    [http://arxiv.org/abs/2308.15419](http://arxiv.org/abs/2308.15419)

    本研究通过从五个英语语言模型预训练运行中提取学习曲线，揭示了语言模型在预训练期间的学习过程。结果表明，语言模型在学习生成更长、更连贯的文本之前，会生成短而重复的短语。同时，频繁出现的标记在预训练过程中更早学习，具有更小的变异性，并且很少被遗忘。较短、更频繁的上下文与稳定和快速获得的预测有关。词类的影响较小，但名词倾向于较晚获得且遗忘率较低。

    

    在本文中，我们从五个自回归英语语言模型预训练运行中提取学习曲线，用于上下文中的100万个标记。我们观察到，在学习生成更长、更连贯文本之前，语言模型会生成短而重复的短语。我们定量描述了单个上下文中标记的学习曲线的最终surprisal、运行内变异性、获得年龄、遗忘度和跨运行变异性。更频繁的标记达到较低的最终surprisal，其内部和预训练运行间变异性较小，学习得更早，并且在预训练过程中很少被“遗忘”。更高的n-gram概率进一步加强了这些效果。与目标标记无关，较短、更频繁的上下文与较稳定和快速获得的预测略有相关。词类的影响也较小，尽管名词倾向于较晚获得且遗忘率较低。

    How do language models learn to make predictions during pre-training? To study this question, we extract learning curves from five autoregressive English language model pre-training runs, for 1M tokens in context. We observe that the language models generate short repetitive phrases before learning to generate longer and more coherent text. We quantify the final surprisal, within-run variability, age of acquisition, forgettability, and cross-run variability of learning curves for individual tokens in context. More frequent tokens reach lower final surprisals, exhibit less variability within and across pre-training runs, are learned earlier, and are less likely to be "forgotten" during pre-training. Higher n-gram probabilities further accentuate these effects. Independent of the target token, shorter and more frequent contexts correlate with marginally more stable and quickly acquired predictions. Effects of part-of-speech are also small, although nouns tend to be acquired later and les
    
[^11]: SILO语言模型：在非参数化数据存储中隔离法律风险

    SILO Language Models: Isolating Legal Risk In a Nonparametric Datastore. (arXiv:2308.04430v1 [cs.CL])

    [http://arxiv.org/abs/2308.04430](http://arxiv.org/abs/2308.04430)

    SILO是一种新的语言模型，通过在推理过程中对非参数化的数据存储进行查询，实现在面临法律风险和模型性能之间的权衡，并支持数据归属和数据生产者退出模型的功能。

    

    在对将语言模型（LMs）训练在受版权或受其他限制的数据上的合法性进行激烈辩论的同时，我们展示了仅在低风险文本（例如过期版权图书或政府文件）上训练时，模型性能显著下降的问题，原因是该文本的规模和领域覆盖有限。我们提出了SILO，一种新的语言模型，在推理过程中管理这种风险-性能权衡。SILO通过以下方式构建：（1）在我们策划的新语料库“开放许可证语料库”（OLC）上训练参数化的LM，该语料库包含228B个公共领域和许可文本。（2）通过非参数化的数据存储（例如包含受版权保护的图书或新闻的数据）对其进行扩充，该数据存储仅在推理过程中被查询。该数据存储允许使用高风险数据而无需对其进行训练，支持句级数据归属，并使数据生产者可以通过从存储中删除内容来选择退出模型。这些功能可以促进对数据使用规范的遵循。

    The legality of training language models (LMs) on copyrighted or otherwise restricted data is under intense debate. However, as we show, model performance significantly degrades if trained only on low-risk text (e.g., out-of-copyright books or government documents), due to its limited size and domain coverage. We present SILO, a new language model that manages this risk-performance tradeoff during inference. SILO is built by (1) training a parametric LM on Open License Corpus (OLC), a new corpus we curate with 228B tokens of public domain and permissively licensed text and (2) augmenting it with a more general and easily modifiable nonparametric datastore (e.g., containing copyrighted books or news) that is only queried during inference. The datastore allows use of high-risk data without training on it, supports sentence-level data attribution, and enables data producers to opt out from the model by removing content from the store. These capabilities can foster compliance with data-use
    

