# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Compact Speech Translation Models via Discrete Speech Units Pretraining](https://arxiv.org/abs/2402.19333) | 通过在离散语音单元上预训练较小模型，以蒸馏SSL模型的知识，实现了紧凑的语音翻译模型，具有短推理管道和适用于低资源环境等优点 |
| [^2] | [Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions](https://arxiv.org/abs/2402.18060) | 在回答医学问题方面，大型语言模型在处理具有挑战性的实际临床案例上的表现是关键，因此构建了两个结构化数据集进行评估。 |
| [^3] | [Case-Based or Rule-Based: How Do Transformers Do the Math?](https://arxiv.org/abs/2402.17709) | transformers在数学问题中采用基于案例的推理而非基于规则的推理。 |
| [^4] | [Ouroboros: Speculative Decoding with Large Model Enhanced Drafting](https://arxiv.org/abs/2402.13720) | Ouroboros通过构建短小草案并引入候选短语池的方法提高了大语言模型推理的加速效率 |
| [^5] | [Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment](https://arxiv.org/abs/2402.13561) | 该论文提出了一种认知视觉语言映射器（CVLM），通过增强视觉知识对齐，在多模态理解中取得了重要进展，特别是在挑战知识型视觉问题回答方面。 |
| [^6] | [Improving Demonstration Diversity by Human-Free Fusing for Text-to-SQL](https://arxiv.org/abs/2402.10663) | 本文提出了一种通过无需人类参与的多次迭代合成来改善文本到SQL演示的多样性，并构建了高多样性演示池，提高了多样性并降低标注成本。 |
| [^7] | [Generating Chain-of-Thoughts with a Direct Pairwise-Comparison Approach to Searching for the Most Promising Intermediate Thought](https://arxiv.org/abs/2402.06918) | 本文提出了一种基于直接两两比较的方法，通过利用LLMs的噪声反馈，直接识别出最有潜力的中间思维，从而生成优秀的思维链。 |
| [^8] | [SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models](https://arxiv.org/abs/2402.05935) | 本论文介绍了SPHINX-X，一种扩展的多模态大型语言模型系列。通过改进架构和训练效率，我们成功构建了一系列参数大小和多语言能力不同的MLLMs，与数据和参数规模有强相关性。 |
| [^9] | [Large Language Models As Faithful Explainers](https://arxiv.org/abs/2402.04678) | 本论文提出了一个生成解释框架（xLLM），用于提高大型语言模型（LLMs）自然语言格式解释的可信度。通过一个评估器来量化解释的可信度，并通过迭代优化过程来提高可信度。 |
| [^10] | [Caught in the Quicksand of Reasoning, Far from AGI Summit: Evaluating LLMs' Mathematical and Coding Competency through Ontology-guided Interventions](https://arxiv.org/abs/2401.09395) | 通过引入数学和编码问题的扰动本体以及两个数据集，作者评估了LLMs在数字推理和编码任务中的能力，在全面评估中发现所有模型在扰动问题上表现显著下降，表明当前的LLMs缺乏稳健性。 |
| [^11] | [Unlocking Anticipatory Text Generation: A Constrained Approach for Large Language Models Decoding](https://arxiv.org/abs/2312.06149) | 提出了将文本生成形式化为未来受限生成问题的方法，以最小化不良行为并强制执行对指令的忠实性，并通过LLMs有效指导文本生成。 |
| [^12] | [MFAS: Emotion Recognition through Multiple Perspectives Fusion Architecture Search Emulating Human Cognition.](http://arxiv.org/abs/2306.09361) | 该论文提出了一种基于多角度融合结构搜索的情感识别框架，模拟人类的认知过程，能够从连续的角度捕捉更全面的情感信息。 |
| [^13] | [On the Impact of Voice Anonymization on Speech-Based COVID-19 Detection.](http://arxiv.org/abs/2304.02181) | 研究探讨了语音匿名化在 COVID-19 检测应用中的影响。研究发现，匿名化方法可能会对语音诊断系统的准确性产生显著影响。 |

# 详细

[^1]: 通过离散语音单元预训练实现紧凑的语音翻译模型

    Compact Speech Translation Models via Discrete Speech Units Pretraining

    [https://arxiv.org/abs/2402.19333](https://arxiv.org/abs/2402.19333)

    通过在离散语音单元上预训练较小模型，以蒸馏SSL模型的知识，实现了紧凑的语音翻译模型，具有短推理管道和适用于低资源环境等优点

    

    使用自监督学习（SSL）作为模型初始化如今在语音翻译（ST）中获得强大结果是常见的。然而，它们也会占用大量内存，阻碍了设备部署。本文利用SSL模型通过在其离散语音单元（DSU）上预训练较小模型。我们在1）Filterbank-to-DSU和2）DSU-to-Translation数据上预训练编码器-解码器模型，然后取自1）的编码器和来自2）的解码器来初始化一个新模型，在有限的语音翻译数据上微调。通过使用DSU预训练来提炼SSL模型的知识，最终模型变得紧凑。我们的方法相比于使用DSU作为模型输入有几个优点，比如推理管道更短和对（DSU）标记化的鲁棒性。与ASR预训练相比，它不需要转录，使其适用于资源匮乏的环境。在CoVoST-2 X-En上的评估显示我们的方法是

    arXiv:2402.19333v1 Announce Type: new  Abstract: Using Self-Supervised Learning (SSL) as model initialization is now common to obtain strong results in Speech Translation (ST). However, they also impose a large memory footprint, hindering on-device deployment. In this paper, we leverage the SSL models by pretraining smaller models on their Discrete Speech Units (DSU). We pretrain encoder-decoder models on 1) Filterbank-to-DSU and 2) DSU-to-Translation data, and take the encoder from 1) and the decoder from 2) to initialise a new model, finetuning this on limited speech-translation data. The final model becomes compact by using the DSU pretraining to distil the knowledge of the SSL model. Our method has several benefits over using DSU as model inputs, such as shorter inference pipeline and robustness over (DSU) tokenization. In contrast to ASR pretraining, it does not require transcripts, making it applicable to low-resource settings. Evaluation on CoVoST-2 X-En shows that our method is
    
[^2]: 在回答和解释具有挑战性的医学问题上对大型语言模型的基准测试

    Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions

    [https://arxiv.org/abs/2402.18060](https://arxiv.org/abs/2402.18060)

    在回答医学问题方面，大型语言模型在处理具有挑战性的实际临床案例上的表现是关键，因此构建了两个结构化数据集进行评估。

    

    LLMs在回答医学问题方面表现出色，例如通过医学执照考试。然而，大多数现有的基准测试依赖于委员会考试问题或一般医学问题，无法捕捉真实临床案例的复杂性。此外，缺乏答案的参考解释阻碍了对模型解释的评估，这对支持医生做出复杂的医疗决策至关重要。为解决这些挑战，我们构建了两个新数据集：JAMA临床挑战和Medbullets。JAMA临床挑战包含基于具有挑战性的临床案例的问题，而Medbullets包含类似USMLE Step 2&3风格的临床问题。两个数据集均以多项选择问题-回答任务的结构化形式呈现，每个问题都附有专家撰写的解释。我们使用各种提示在这两个数据集上评估了四个LLMs。实验表明

    arXiv:2402.18060v1 Announce Type: new  Abstract: LLMs have demonstrated impressive performance in answering medical questions, such as passing medical licensing examinations. However, most existing benchmarks rely on board exam questions or general medical questions, falling short in capturing the complexity of realistic clinical cases. Moreover, the lack of reference explanations for answers hampers the evaluation of model explanations, which are crucial to supporting doctors in making complex medical decisions. To address these challenges, we construct two new datasets: JAMA Clinical Challenge and Medbullets. JAMA Clinical Challenge consists of questions based on challenging clinical cases, while Medbullets comprises USMLE Step 2&3 style clinical questions. Both datasets are structured as multiple-choice question-answering tasks, where each question is accompanied by an expert-written explanation. We evaluate four LLMs on the two datasets using various prompts. Experiments demonstrat
    
[^3]: 基于案例还是基于规则：变压器如何进行数学计算？

    Case-Based or Rule-Based: How Do Transformers Do the Math?

    [https://arxiv.org/abs/2402.17709](https://arxiv.org/abs/2402.17709)

    transformers在数学问题中采用基于案例的推理而非基于规则的推理。

    

    尽管现代大型语言模型在各种复杂任务中表现出色，但仍然难以处理一些对人类来说简单且直观的数学问题，例如加法。然而，我们可以轻松学习加法的基本规则，并将其应用于任意长度的新问题，而大型语言模型却难以做到。相反，它们可能依赖于在训练语料库中看到的类似“案例”来获取帮助。我们将这两种不同的推理机制定义为“基于规则的推理”和“基于案例的推理”。由于基于规则的推理对于获得系统化概括能力至关重要，我们旨在探究变压器究竟是使用基于规则还是基于案例的推理来解决数学问题。通过精心设计的五个数学任务的干预实验，我们确认变压器正在执行基于案例的推理，无论是否使用草稿本，这与之前的观察结果一致。

    arXiv:2402.17709v1 Announce Type: new  Abstract: Despite the impressive performance in a variety of complex tasks, modern large language models (LLMs) still have trouble dealing with some math problems that are simple and intuitive for humans, such as addition. While we can easily learn basic rules of addition and apply them to new problems of any length, LLMs struggle to do the same. Instead, they may rely on similar "cases" seen in the training corpus for help. We define these two different reasoning mechanisms as "rule-based reasoning" and "case-based reasoning". Since rule-based reasoning is essential for acquiring the systematic generalization ability, we aim to explore exactly whether transformers use rule-based or case-based reasoning for math problems. Through carefully designed intervention experiments on five math tasks, we confirm that transformers are performing case-based reasoning, no matter whether scratchpad is used, which aligns with the previous observations that tran
    
[^4]: Ouroboros: 大模型增强草案的猜测解码技术

    Ouroboros: Speculative Decoding with Large Model Enhanced Drafting

    [https://arxiv.org/abs/2402.13720](https://arxiv.org/abs/2402.13720)

    Ouroboros通过构建短小草案并引入候选短语池的方法提高了大语言模型推理的加速效率

    

    通过构建短小高效的小模型起草草案，然后要求大语言模型以无自回归方式进行验证和修正，以最小化时间开销。当验证后可以生成更长的草稿，但也会导致相当大的尝试和错误成本。由于高验证失败概率，现有解码方法不能一次起草太多内容进行验证，实现次优的推理加速。

    arXiv:2402.13720v1 Announce Type: new  Abstract: Drafting-then-verifying decoding methods such as speculative decoding are widely adopted training-free methods to accelerate the inference of large language models (LLMs). Instead of employing an autoregressive process to decode tokens sequentially, speculative decoding initially creates drafts with an efficient small model. Then LLMs are required to conduct verification and correction in a non-autoregressive fashion to minimize time overhead. Generating longer drafts can lead to even more significant speedups once verified, but also incurs substantial trial and error costs if it fails. Suffering from the high verification failure probability, existing decoding methods cannot draft too much content for verification at one time, achieving sub-optimal inference acceleration. In this paper, we introduce Ouroboros, which constructs a phrase candidate pool from the verification process of LLMs to provide candidates for draft generation of the
    
[^5]: 认知视觉语言映射器：通过增强视觉知识对齐推进多模态理解

    Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment

    [https://arxiv.org/abs/2402.13561](https://arxiv.org/abs/2402.13561)

    该论文提出了一种认知视觉语言映射器（CVLM），通过增强视觉知识对齐，在多模态理解中取得了重要进展，特别是在挑战知识型视觉问题回答方面。

    

    评估和反思当前大型多模态模型（LMMs）的现状，我们观察到广泛使用的视觉语言投影方法（如Q-former或MLP）侧重于图像-文本描述的对齐，但忽略了视觉知识维度的对齐，即将视觉与其相关知识连接起来。视觉知识在分析、推断和解释视觉信息方面起着重要作用，有助于提高基于知识的视觉问题答案的准确性。本文主要探讨通过视觉语言知识对齐来改进LMMs，特别针对挑战知识型视觉问答（VQA）。为此，我们提出了一个认知视觉语言映射器（CVLM），其中包含一个预训练的视觉知识对齐器（VKA）和一个用于多模态指令调节阶段的细粒度知识适配器（FKA）。具体来说，我们基于

    arXiv:2402.13561v1 Announce Type: new  Abstract: Evaluating and Rethinking the current landscape of Large Multimodal Models (LMMs), we observe that widely-used visual-language projection approaches (e.g., Q-former or MLP) focus on the alignment of image-text descriptions yet ignore the visual knowledge-dimension alignment, i.e., connecting visuals to their relevant knowledge. Visual knowledge plays a significant role in analyzing, inferring, and interpreting information from visuals, helping improve the accuracy of answers to knowledge-based visual questions. In this paper, we mainly explore improving LMMs with visual-language knowledge alignment, especially aimed at challenging knowledge-based visual question answering (VQA). To this end, we present a Cognitive Visual-Language Mapper (CVLM), which contains a pretrained Visual Knowledge Aligner (VKA) and a Fine-grained Knowledge Adapter (FKA) used in the multimodal instruction tuning stage. Specifically, we design the VKA based on the 
    
[^6]: 通过无需人类参与的融合方法改善文本到SQL的演示多样性

    Improving Demonstration Diversity by Human-Free Fusing for Text-to-SQL

    [https://arxiv.org/abs/2402.10663](https://arxiv.org/abs/2402.10663)

    本文提出了一种通过无需人类参与的多次迭代合成来改善文本到SQL演示的多样性，并构建了高多样性演示池，提高了多样性并降低标注成本。

    

    目前，基于大型语言模型（LLMs）的上下文学习方法已成为文本到SQL研究的主流。先前的工作讨论了如何从人标记的演示池中选择与用户问题相关的演示。然而，人工标注存在着多样性不足和标注成本高的限制。因此，在本文中，我们讨论了如何衡量和改善文本到SQL演示的多样性。我们提出了一个度量演示多样性的指标，并通过实验分析了现有标记数据的不足之处。基于上述发现，我们提出了一种通过无需人类参与的多次迭代合成来构建高多样性演示池的融合方法（Fused），提高了多样性并降低标注成本。我们的方法在有/无人类标注的情况下平均提高了3.2%和5.0%。

    arXiv:2402.10663v1 Announce Type: new  Abstract: Currently, the in-context learning method based on large language models (LLMs) has become the mainstream of text-to-SQL research. Previous works have discussed how to select demonstrations related to the user question from a human-labeled demonstration pool. However, human labeling suffers from the limitations of insufficient diversity and high labeling overhead. Therefore, in this paper, we discuss how to measure and improve the diversity of the demonstrations for text-to-SQL. We present a metric to measure the diversity of the demonstrations and analyze the insufficient of the existing labeled data by experiments. Based on the above discovery, we propose fusing iteratively for demonstrations (Fused) to build a high-diversity demonstration pool through human-free multiple-iteration synthesis, improving diversity and lowering label cost. Our method achieves an average improvement of 3.2% and 5.0% with and without human labeling on sever
    
[^7]: 用直接的两两比较方法生成思维链，以搜索最有潜力的中间思维

    Generating Chain-of-Thoughts with a Direct Pairwise-Comparison Approach to Searching for the Most Promising Intermediate Thought

    [https://arxiv.org/abs/2402.06918](https://arxiv.org/abs/2402.06918)

    本文提出了一种基于直接两两比较的方法，通过利用LLMs的噪声反馈，直接识别出最有潜力的中间思维，从而生成优秀的思维链。

    

    为了提高大型语言模型(LLMs)处理复杂推理问题的能力，提出了思维链(Chain-of-Thoughts, CoT)方法，用于指导LLMs进行逐步推理，从简单到复杂的问题解决。目前最先进的生成这种思维链的方法涉及互动协作，学习者生成候选中间思维，由LLMs评估，引导生成后续思维。然而，一个广泛但未被充分研究的问题是，LLMs的评估通常存在噪声和不可靠性，可能误导生成过程，选择不够有潜力的中间思维。本文受Vapnik原则的启发，提出了一种新的基于比较的CoT生成算法，直接根据LLMs的噪声反馈确定最有潜力的思维。在每一轮中，我们随机配对中间思维，并直接促使LLMs从每对中选择更有潜力的思维。

    To improve the ability of the large language model (LLMs) to handle complex reasoning problems, chain-of-thoughts (CoT) methods were proposed to guide LLMs to reason step-by-step, facilitating problem solving from simple to complex tasks. State-of-the-art approaches for generating such a chain involve interactive collaboration, where the learner generates candidate intermediate thoughts, evaluated by the LLM, guiding the generation of subsequent thoughts. However, a widespread yet understudied problem is that the evaluation from the LLM is typically noisy and unreliable, potentially misleading the generation process in selecting promising intermediate thoughts. In this paper, motivated by Vapnik's principle, we propose a novel comparison-based CoT generation algorithm that directly identifies the most promising thoughts with the noisy feedback from the LLM. In each round, we randomly pair intermediate thoughts and directly prompt the LLM to select the more promising one from each pair,
    
[^8]: SPHINX-X: 扩展数据和参数用于一系列多模态大型语言模型

    SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models

    [https://arxiv.org/abs/2402.05935](https://arxiv.org/abs/2402.05935)

    本论文介绍了SPHINX-X，一种扩展的多模态大型语言模型系列。通过改进架构和训练效率，我们成功构建了一系列参数大小和多语言能力不同的MLLMs，与数据和参数规模有强相关性。

    

    我们提出SPHINX-X，一种基于SPHINX开发的广泛多模态大型语言模型（MLLM）系列。为了改善架构和训练效率，我们通过移除冗余的视觉编码器、绕过完全填充的子图像，并将多阶段训练简化成为一阶段的全集合模式，修改了SPHINX框架。为了充分发挥MLLM的潜力，我们组装了一个综合的跨语言、跨视觉和视觉-语言任务的多领域、多模态的数据集，涵盖了公开可用的资源。我们进一步使用我们的OCR密集和Mark数据集丰富这个收集，扩展了多样性和普适性。通过对不同基础LLM进行训练，包括TinyLlama1.1B、InternLM2-7B、LLaMA2-13B和Mixtral8x7B，我们获得了一系列参数大小和多语言能力变化的MLLMs。全面的基准测试揭示了多模态性能与数据和参数规模之间的强相关性。

    We propose SPHINX-X, an extensive Multimodality Large Language Model (MLLM) series developed upon SPHINX. To improve the architecture and training efficiency, we modify the SPHINX framework by removing redundant visual encoders, bypassing fully-padded sub-images with skip tokens, and simplifying multi-stage training into a one-stage all-in-one paradigm. To fully unleash the potential of MLLMs, we assemble a comprehensive multi-domain and multimodal dataset covering publicly available resources in language, vision, and vision-language tasks. We further enrich this collection with our curated OCR intensive and Set-of-Mark datasets, extending the diversity and generality. By training over different base LLMs including TinyLlama1.1B, InternLM2-7B, LLaMA2-13B, and Mixtral8x7B, we obtain a spectrum of MLLMs that vary in parameter size and multilingual capabilities. Comprehensive benchmarking reveals a strong correlation between the multi-modal performance with the data and parameter scales. 
    
[^9]: 大型语言模型作为可信的解释器

    Large Language Models As Faithful Explainers

    [https://arxiv.org/abs/2402.04678](https://arxiv.org/abs/2402.04678)

    本论文提出了一个生成解释框架（xLLM），用于提高大型语言模型（LLMs）自然语言格式解释的可信度。通过一个评估器来量化解释的可信度，并通过迭代优化过程来提高可信度。

    

    近年来，大型语言模型(LLMs)通过利用其丰富的内部知识和推理能力，已经能够熟练解决复杂的任务。然而，这种复杂性阻碍了传统的以输入为重点的解释算法来解释LLMs的复杂决策过程。为了解决这个问题，最近出现了一种自我解释机制，通过自然语言的形式进行单向推理，从而实现对LLMs预测的解释。然而，这种自然语言解释经常因为缺乏可信度而受到批评，因为这些解释可能不准确地反映LLMs的决策行为。在这项工作中，我们引入了一个生成解释框架xLLM，以提高LLMs自然语言格式的解释的可信度。具体而言，我们提出了一个评估器来量化自然语言解释的可信度，并通过xLLM的迭代优化过程来提高可信度，目标是最大程度地提高可信度。

    Large Language Models (LLMs) have recently become proficient in addressing complex tasks by utilizing their rich internal knowledge and reasoning ability. Consequently, this complexity hinders traditional input-focused explanation algorithms for explaining the complex decision-making processes of LLMs. Recent advancements have thus emerged for self-explaining their predictions through a single feed-forward inference in a natural language format. However, natural language explanations are often criticized for lack of faithfulness since these explanations may not accurately reflect the decision-making behaviors of the LLMs. In this work, we introduce a generative explanation framework, xLLM, to improve the faithfulness of the explanations provided in natural language formats for LLMs. Specifically, we propose an evaluator to quantify the faithfulness of natural language explanation and enhance the faithfulness by an iterative optimization process of xLLM, with the goal of maximizing the 
    
[^10]: 被理性的流沙所困，远离AGI峰会：通过本体引导干预评估LLMs的数学和编码能力

    Caught in the Quicksand of Reasoning, Far from AGI Summit: Evaluating LLMs' Mathematical and Coding Competency through Ontology-guided Interventions

    [https://arxiv.org/abs/2401.09395](https://arxiv.org/abs/2401.09395)

    通过引入数学和编码问题的扰动本体以及两个数据集，作者评估了LLMs在数字推理和编码任务中的能力，在全面评估中发现所有模型在扰动问题上表现显著下降，表明当前的LLMs缺乏稳健性。

    

    最近大型语言模型（LLMs）的先进发展展示了在现有逻辑推理基准测试中取得了引人注目的成果，其中一些模型甚至超过了人类表现。然而，它们在推理任务中的实际能力和稳健性仍然是一个未解之谜。因此，本文关注两个流行的推理任务：算术推理和代码生成。特别是，我们引入了：（i）数学和编码问题的通用扰动本体，（ii）一种半自动方法来应用这些扰动，以及（iii）两个数据集MORE和CORE，分别用于扰动数学和编码问题，以探究LLM在数字推理和编码任务中的能力极限。通过对封闭源和开源LLMs的全面评估，我们展示了所有模型对扰动问题的显著性能下降，表明当前的LLMs缺乏稳健性。

    arXiv:2401.09395v2 Announce Type: replace  Abstract: Recent advancements in Large Language Models (LLMs) have showcased striking results on existing logical reasoning benchmarks, with some models even surpassing human performance. However, the true depth of their competencies and robustness in reasoning tasks remains an open question. To this end, in this paper, we focus on two popular reasoning tasks: arithmetic reasoning and code generation. Particularly, we introduce: (i) a general ontology of perturbations for maths and coding questions, (ii) a semi-automatic method to apply these perturbations, and (iii) two datasets, MORE and CORE, respectively, of perturbed maths and coding problems to probe the limits of LLM capabilities in numeric reasoning and coding tasks. Through comprehensive evaluations of both closed-source and open-source LLMs, we show a significant performance drop across all the models against the perturbed questions, suggesting that the current LLMs lack robust probl
    
[^11]: 解锁预测性文本生成：对大型语言模型解码的受限方法

    Unlocking Anticipatory Text Generation: A Constrained Approach for Large Language Models Decoding

    [https://arxiv.org/abs/2312.06149](https://arxiv.org/abs/2312.06149)

    提出了将文本生成形式化为未来受限生成问题的方法，以最小化不良行为并强制执行对指令的忠实性，并通过LLMs有效指导文本生成。

    

    大型语言模型(LLMs)展现了强大的文本生成能力。然而，对于给定提示或指令实现最佳结果可能具有挑战性，特别是对于十亿级别的模型。此外，不良行为如毒性或幻觉可能会显现。在这项工作中，我们提出将文本生成形式化为未来受限生成问题，以最小化不良行为并强制执行对指令的忠实性。使用LLMs实现未来约束满足度的估计引导文本生成过程。我们的广泛实验表明所提出的方法在三个不同的文本生成任务中的有效性：关键词受限生成、毒性减少等。

    arXiv:2312.06149v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have demonstrated a powerful ability for text generation. However, achieving optimal results with a given prompt or instruction can be challenging, especially for billion-sized models. Additionally, undesired behaviors such as toxicity or hallucinations can manifest. While much larger models (e.g., ChatGPT) may demonstrate strength in mitigating these issues, there is still no guarantee of complete prevention. In this work, we propose formalizing text generation as a future-constrained generation problem to minimize undesirable behaviors and enforce faithfulness to instructions. The estimation of future constraint satisfaction, accomplished using LLMs, guides the text generation process. Our extensive experiments demonstrate the effectiveness of the proposed approach across three distinct text generation tasks: keyword-constrained generation (Lin et al., 2020), toxicity reduction (Gehman et al., 202
    
[^12]: MFAS: 基于多角度融合结构搜索的情感识别，模拟人类认知

    MFAS: Emotion Recognition through Multiple Perspectives Fusion Architecture Search Emulating Human Cognition. (arXiv:2306.09361v1 [eess.AS])

    [http://arxiv.org/abs/2306.09361](http://arxiv.org/abs/2306.09361)

    该论文提出了一种基于多角度融合结构搜索的情感识别框架，模拟人类的认知过程，能够从连续的角度捕捉更全面的情感信息。

    

    语音情感识别旨在识别和分析与人类类似的情绪状态。完美的情感识别可以极大地改善各种人机交互任务。受人类理解情感的过程的启发，我们证明了与量化建模相比，从连续的角度理解语音内容，类似于人类的理解，能够使模型捕捉更全面的情感信息。此外，考虑到人类根据语音中存在的某些线索调整情感单词的文本语义的感知，我们设计了一个新的搜索空间并搜索两种信息的最佳融合策略。实验结果进一步验证了调整感知的重要性。基于这些观察结果，我们提出了一种新的框架，称为Multiple perspectives Fusion Architecture Search(MFAS)。

    Speech emotion recognition aims to identify and analyze emotional states in target speech similar to humans. Perfect emotion recognition can greatly benefit a wide range of human-machine interaction tasks. Inspired by the human process of understanding emotions, we demonstrate that compared to quantized modeling, understanding speech content from a continuous perspective, akin to human-like comprehension, enables the model to capture more comprehensive emotional information. Additionally, considering that humans adjust their perception of emotional words in textual semantic based on certain cues present in speech, we design a novel search space and search for the optimal fusion strategy for the two types of information. Experimental results further validate the significance of this perception adjustment. Building on these observations, we propose a novel framework called Multiple perspectives Fusion Architecture Search (MFAS). Specifically, we utilize continuous-based knowledge to capt
    
[^13]: 关于声音匿名化对基于语音的COVID-19检测的影响研究

    On the Impact of Voice Anonymization on Speech-Based COVID-19 Detection. (arXiv:2304.02181v1 [cs.CL])

    [http://arxiv.org/abs/2304.02181](http://arxiv.org/abs/2304.02181)

    研究探讨了语音匿名化在 COVID-19 检测应用中的影响。研究发现，匿名化方法可能会对语音诊断系统的准确性产生显著影响。

    

    随着深度学习的发展，基于语音的应用正蓬勃发展，从个人助理、情感计算到远程疾病诊断。由于声音同时包含语言和语用信息（如语音音调、语调、语速、声音大小），因此保护说话者的隐私和身份的声音匿名化引起了广泛的关注。近年来，声音隐私问题已经出现，重点是去除说话者身份，同时保留语言内容。然而，对于情感计算和疾病监测应用而言，语用内容可能更为关键。遗憾的是，匿名化可能对这些系统产生的影响仍然不明确。在本文中，我们填补了这个空白，并专注于一个特定的健康监测应用：基于语音的COVID-19诊断。我们测试了两种流行的匿名化方法及其对五种最先进的COVID-19诊断系统的影响。

    With advances seen in deep learning, voice-based applications are burgeoning, ranging from personal assistants, affective computing, to remote disease diagnostics. As the voice contains both linguistic and paralinguistic information (e.g., vocal pitch, intonation, speech rate, loudness), there is growing interest in voice anonymization to preserve speaker privacy and identity. Voice privacy challenges have emerged over the last few years and focus has been placed on removing speaker identity while keeping linguistic content intact. For affective computing and disease monitoring applications, however, the paralinguistic content may be more critical. Unfortunately, the effects that anonymization may have on these systems are still largely unknown. In this paper, we fill this gap and focus on one particular health monitoring application: speech-based COVID-19 diagnosis. We test two popular anonymization methods and their impact on five different state-of-the-art COVID-19 diagnostic system
    

