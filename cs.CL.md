# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dataverse: Open-Source ETL (Extract, Transform, Load) Pipeline for Large Language Models](https://arxiv.org/abs/2403.19340) | Dataverse是一个面向大型语言模型的开源ETL管道，提供了用户友好的设计和易于定制的处理器添加功能，旨在成为LLM开发的重要工具，并开源整个库以促进社区贡献。 |
| [^2] | [AURA: Natural Language Reasoning for Aleatoric Uncertainty in Rationales](https://arxiv.org/abs/2402.14337) | 提出了在自然语言推理中处理引发模式合理性不确定性的不完美理由的方法，实施了使用熵分数和模型先验信念来指导模型的策略，并在实证中展示了方法相对于敌对理由的稳健性能优势 |
| [^3] | [De-identification is not always enough](https://arxiv.org/abs/2402.00179) | 研究表明，仅仅进行去识别操作并不能有效保护隐私。本文提出了使用大型语言模型生成合成临床笔记的方法，并评估了其在临床任务中的性能。同时，还发现利用合成数据训练模型可以提高会员推理攻击的成功率。 |
| [^4] | [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.](http://arxiv.org/abs/2401.15077) | EAGLE是一个无损加速语言模型推理的框架，通过在次顶层特征层面上自回归推理，并解决采样不确定性问题，实现了比传统方法更快3倍的速度。 |
| [^5] | [Diversifying Question Generation over Knowledge Base via External Natural Questions.](http://arxiv.org/abs/2309.14362) | 通过引入新的多样性评估指标，我们提出了一种通过外部自然问题在知识库上进行多样化问题生成的方法。同时，我们设计了一个双模型框架来解决如何增强多样化问题生成的挑战。 |

# 详细

[^1]: Dataverse：用于大型语言模型的开源ETL（抽取、转换、加载）管道

    Dataverse: Open-Source ETL (Extract, Transform, Load) Pipeline for Large Language Models

    [https://arxiv.org/abs/2403.19340](https://arxiv.org/abs/2403.19340)

    Dataverse是一个面向大型语言模型的开源ETL管道，提供了用户友好的设计和易于定制的处理器添加功能，旨在成为LLM开发的重要工具，并开源整个库以促进社区贡献。

    

    为了解决规模化数据处理所面临的挑战，我们提出了Dataverse，一个统一的面向大型语言模型（LLMs）的开源抽取-转换-加载（ETL）管道，其核心具有用户友好的设计。在Dataverse中，通过基于块的界面轻松添加自定义处理器，使用户可以方便高效地使用Dataverse构建自己的ETL管道。我们希望Dataverse将成为LLM开发的重要工具，并开放整个库以欢迎社区贡献。此外，我们提供了一个简洁的、两分钟的系统演示视频，展示其功能和实现。

    arXiv:2403.19340v1 Announce Type: cross  Abstract: To address the challenges associated with data processing at scale, we propose Dataverse, a unified open-source Extract-Transform-Load (ETL) pipeline for large language models (LLMs) with a user-friendly design at its core. Easy addition of custom processors with block-based interface in Dataverse allows users to readily and efficiently use Dataverse to build their own ETL pipeline. We hope that Dataverse will serve as a vital tool for LLM development and open source the entire library to welcome community contribution. Additionally, we provide a concise, two-minute video demonstration of our system, illustrating its capabilities and implementation.
    
[^2]: AURA：自然语言推理中的模式合理性不确定性

    AURA: Natural Language Reasoning for Aleatoric Uncertainty in Rationales

    [https://arxiv.org/abs/2402.14337](https://arxiv.org/abs/2402.14337)

    提出了在自然语言推理中处理引发模式合理性不确定性的不完美理由的方法，实施了使用熵分数和模型先验信念来指导模型的策略，并在实证中展示了方法相对于敌对理由的稳健性能优势

    

    回策背后的理由不仅解释了模型决策，而且提升了语言模型在复杂推理任务上的推理能力。然而，获得无懈可击的理由通常是不可能的。此外，估计理由足够忠实以鼓励模型表现的程度并不是微不足道的。因此，这些推理任务通常迫使模型在不理想的理由下输出正确答案，并且与模型完全有能力的情况相比是次优的。在这项工作中，我们提出了如何应对引发模式合理性不确定性的不完美理由。我们首先用给定理由的熵分数来定义模糊的理由，使用模型先验信念作为信息量。然后根据理由的模糊性来引导模型选择两种不同的推理模型中的一种。我们在实证上论证了我们提出的方法相对于理由的敌对质量产生了稳健的性能优势。

    arXiv:2402.14337v1 Announce Type: new  Abstract: Rationales behind answers not only explain model decisions but boost language models to reason well on complex reasoning tasks. However, obtaining impeccable rationales is often impossible. Besides, it is non-trivial to estimate the degree to which the rationales are faithful enough to encourage model performance. Thus, such reasoning tasks often compel models to output correct answers under undesirable rationales and are sub-optimal compared to what the models are fully capable of. In this work, we propose how to deal with imperfect rationales causing aleatoric uncertainty. We first define the ambiguous rationales with entropy scores of given rationales, using model prior beliefs as informativeness. We then guide models to select one of two different reasoning models according to the ambiguity of rationales. We empirically argue that our proposed method produces robust performance superiority against the adversarial quality of rationale
    
[^3]: 不仅仅去识别可能是不够的

    De-identification is not always enough

    [https://arxiv.org/abs/2402.00179](https://arxiv.org/abs/2402.00179)

    研究表明，仅仅进行去识别操作并不能有效保护隐私。本文提出了使用大型语言模型生成合成临床笔记的方法，并评估了其在临床任务中的性能。同时，还发现利用合成数据训练模型可以提高会员推理攻击的成功率。

    

    对于共享隐私敏感数据，常常将去识别视为足够保护隐私的措施。合成数据也被认为是一种保护隐私的替代方法。最近在生成数值和表格数据模型方面取得的成功以及大型生成语言模型的突破引发了一个问题：合成的临床笔记是否可以作为研究目的的真实笔记的可行替代品。在这项工作中，我们证明了：（i）对真实临床笔记的去识别并不能保护记录免遭会员推理攻击；（ii）提出了一种使用当前最先进的大型语言模型生成合成临床笔记的新方法；（iii）在临床领域任务中评估了合成生成笔记的性能；（iv）提出了一种利用合成数据训练目标模型的会员推理攻击方法。我们观察到，当合成生成的笔记与真实笔记相似时，这种攻击的成功率增加。

    For sharing privacy-sensitive data, de-identification is commonly regarded as adequate for safeguarding privacy. Synthetic data is also being considered as a privacy-preserving alternative. Recent successes with numerical and tabular data generative models and the breakthroughs in large generative language models raise the question of whether synthetically generated clinical notes could be a viable alternative to real notes for research purposes. In this work, we demonstrated that (i) de-identification of real clinical notes does not protect records against a membership inference attack, (ii) proposed a novel approach to generate synthetic clinical notes using the current state-of-the-art large language models, (iii) evaluated the performance of the synthetically generated notes in a clinical domain task, and (iv) proposed a way to mount a membership inference attack where the target model is trained with synthetic data. We observed that when synthetically generated notes closely match
    
[^4]: EAGLE: 推测采样需要重新思考特征不确定性

    EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. (arXiv:2401.15077v1 [cs.LG])

    [http://arxiv.org/abs/2401.15077](http://arxiv.org/abs/2401.15077)

    EAGLE是一个无损加速语言模型推理的框架，通过在次顶层特征层面上自回归推理，并解决采样不确定性问题，实现了比传统方法更快3倍的速度。

    

    自回归解码使得大型语言模型（LLMs）的推理变得耗时。我们提出了一个简单的框架，EAGLE（用于提高语言模型效率的外推算法），实现了无损加速。与传统的推测采样方法不同，EAGLE在更规律的（次顶层）特征层面上自回归进行编写，并通过整合提前一个时间步的标记来解决下一个特征预测问题中的采样不确定性。EAGLE所提供的加速是无损的：它不需要微调目标LLM，并且生成的文本与原始的自回归解码的分布相同。截至本文提交时，EAGLE是已知推测采样家族中速度最快的框架。在MT-bench上，EAGLE比原始解码快3倍，比Lookahead快2倍，比Medusa快1.6倍。使用gpt-fast，EAGLE平均每秒达到160个标记与LLaMA2-Chat搭配。

    Auto-regressive decoding makes the inference of Large Language Models (LLMs) time-consuming. We propose a simple framework, EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency), for lossless acceleration. Unlike traditional speculative sampling methods, EAGLE operates the drafting process auto-regressively at the more regular (second-top-layer) feature level and addresses the sampling uncertainty issues in the next-feature prediction problems by integrating tokens from one time step ahead. The acceleration provided by EAGLE is lossless: it involves no fine-tuning of the target LLM, and the generated text maintains the same distribution as that of vanilla auto-regressive decoding. As of the submission of this paper, EAGLE is the fastest known framework within the speculative sampling family. On MT-bench, EAGLE is 3x faster than vanilla decoding, 2x faster than Lookahead, and 1.6x faster than Medusa. Using gpt-fast, EAGLE attains on average 160 tokens/s with LLaMA2-Chat 
    
[^5]: 通过外部自然问题在知识库上进行多样化问题生成

    Diversifying Question Generation over Knowledge Base via External Natural Questions. (arXiv:2309.14362v1 [cs.CL])

    [http://arxiv.org/abs/2309.14362](http://arxiv.org/abs/2309.14362)

    通过引入新的多样性评估指标，我们提出了一种通过外部自然问题在知识库上进行多样化问题生成的方法。同时，我们设计了一个双模型框架来解决如何增强多样化问题生成的挑战。

    

    先前的知识库问题生成方法主要集中在提高单个生成问题的质量。我们认为，人类出色的改写能力表明相同的语义可以通过不同的表达来传达。以上观点使得多样化问题生成成为一个有趣的任务，其中第一个挑战是多样性评估指标。当前的指标不足以评估多样性，因为它们仅计算生成问题中唯一n-gram的比例，更倾向于衡量重复而非真正的多样性。因此，我们设计了一个新的多样性评估指标，它衡量每个实例的前k个生成问题之间的多样性，同时确保它们与基准问题相关。显然，第二个挑战是如何增强多样化问题生成。为了解决这个问题，我们引入了一个由两个选择模型交织而成的双模型框架。

    Previous methods on knowledge base question generation (KBQG) primarily focus on enhancing the quality of a single generated question. Recognizing the remarkable paraphrasing ability of humans, we contend that diverse texts should convey the same semantics through varied expressions. The above insights make diversifying question generation an intriguing task, where the first challenge is evaluation metrics for diversity. Current metrics inadequately assess the above diversity since they calculate the ratio of unique n-grams in the generated question itself, which leans more towards measuring duplication rather than true diversity. Accordingly, we devise a new diversity evaluation metric, which measures the diversity among top-k generated questions for each instance while ensuring their relevance to the ground truth. Clearly, the second challenge is how to enhance diversifying question generation. To address this challenge, we introduce a dual model framework interwoven by two selection
    

