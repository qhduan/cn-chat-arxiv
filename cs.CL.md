# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLM Agent Operating System](https://arxiv.org/abs/2403.16971) | 提出了一种将大型语言模型嵌入操作系统中的LLM代理操作系统，旨在优化资源分配、促进代理间上下文切换、实现并发执行以及为代理提供工具服务。 |
| [^2] | [Negative Yields Positive: Unified Dual-Path Adapter for Vision-Language Models](https://arxiv.org/abs/2403.12964) | 本研究在视觉-语言模型的微调中引入了双学习概念，提出了DualAdapter方法，通过正面和负面两方面的双路径适配，同时进行补充正向选择和负向排除，从而提高了在下游任务中的整体识别准确性。 |
| [^3] | [Logits of API-Protected LLMs Leak Proprietary Information](https://arxiv.org/abs/2403.09539) | 大多数现代LLM受到softmax瓶颈影响，可以以较低成本获取API保护的LLM的非公开信息和解锁多种功能 |
| [^4] | [Curry-DPO: Enhancing Alignment using Curriculum Learning & Ranked Preferences](https://arxiv.org/abs/2403.07230) | 提出了一种名为Curry-DPO的方法，在直接偏好优化(DPO)中利用课程学习方法，通过构建多个偏好对来训练模型，相比于标准单一对DPO设置有着更好的性能表现。 |
| [^5] | [Accelerating Greedy Coordinate Gradient via Probe Sampling](https://arxiv.org/abs/2403.01251) | 研究引入了一种名为“探查采样”的新算法，通过动态确定草稿模型和目标模型的相似度，来加速贪婪坐标梯度算法，实现高达5.6倍的加速。 |
| [^6] | [Emotion Classification in Low and Moderate Resource Languages](https://arxiv.org/abs/2402.18424) | 通过跨语言情感分类器，在低和中等资源语言中实现情感分类，展示了两种迁移学习方法的有效性。 |
| [^7] | [Hal-Eval: A Universal and Fine-grained Hallucination Evaluation Framework for Large Vision Language Models](https://arxiv.org/abs/2402.15721) | 本论文提出了Hal-Eval，一个通用和细粒度的幻觉评估框架，引入了新的幻觉分类法，专注于事件幻觉，通过生成和过滤细粒度幻觉数据来评估大型视觉语言模型对各种幻觉的处理能力。 |
| [^8] | [Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks](https://arxiv.org/abs/2401.17263) | 该论文提出了一种鲁棒的提示优化算法（RPO）用于对抗语言模型的破解攻击，通过梯度优化来确保输出的无害性，并成功降低了攻击成功率。 |
| [^9] | [PAC Neural Prediction Set Learning to Quantify the Uncertainty of Generative Language Models.](http://arxiv.org/abs/2307.09254) | 本文提出了一种使用神经网络来量化生成式语言模型不确定性的PAC神经预测集学习方法，通过在多种语言数据集和模型上的实验证明，相比于标准基准方法，我们的方法平均提高了63％的量化不确定性。 |

# 详细

[^1]: LLM Agent Operating System

    LLM Agent Operating System

    [https://arxiv.org/abs/2403.16971](https://arxiv.org/abs/2403.16971)

    提出了一种将大型语言模型嵌入操作系统中的LLM代理操作系统，旨在优化资源分配、促进代理间上下文切换、实现并发执行以及为代理提供工具服务。

    

    arXiv:2403.16971v1 公告类型: 跨领域 摘要: 部署大型语言模型（LLM）智能代理存在诸多挑战，会损害它们的效率和功效。其中包括代理请求在LLM上的次优调度和资源分配、在代理和LLM之间交互时保持上下文的困难，以及将具有不同能力和专业化的异构代理集成在一起的复杂性。代理数量和复杂性的快速增加进一步加剧了这些问题，通常会导致资源瓶颈和次优资源利用。受到这些挑战的启发，本文提出了AIOS，一种LLM代理操作系统，它将大型语言模型嵌入操作系统（OS）中。具体地，AIOS旨在优化资源分配，促进代理之间的上下文切换，实现代理的并发执行，为代理提供工具服务。

    arXiv:2403.16971v1 Announce Type: cross  Abstract: The integration and deployment of large language model (LLM)-based intelligent agents have been fraught with challenges that compromise their efficiency and efficacy. Among these issues are sub-optimal scheduling and resource allocation of agent requests over the LLM, the difficulties in maintaining context during interactions between agent and LLM, and the complexities inherent in integrating heterogeneous agents with different capabilities and specializations. The rapid increase of agent quantity and complexity further exacerbates these issues, often leading to bottlenecks and sub-optimal utilization of resources. Inspired by these challenges, this paper presents AIOS, an LLM agent operating system, which embeds large language model into operating systems (OS). Specifically, AIOS is designed to optimize resource allocation, facilitate context switch across agents, enable concurrent execution of agents, provide tool service for agents
    
[^2]: 负得正：用于视觉语言模型的统一双路径适配器

    Negative Yields Positive: Unified Dual-Path Adapter for Vision-Language Models

    [https://arxiv.org/abs/2403.12964](https://arxiv.org/abs/2403.12964)

    本研究在视觉-语言模型的微调中引入了双学习概念，提出了DualAdapter方法，通过正面和负面两方面的双路径适配，同时进行补充正向选择和负向排除，从而提高了在下游任务中的整体识别准确性。

    

    最近，大规模预训练的视觉-语言模型（VLMs）展示了学习开放世界视觉表示方面的巨大潜力，并通过高效微调在各种下游任务中展现出卓越性能。在这项工作中，我们创新地将双学习概念引入微调VLMs中，即我们不仅学习图像是什么，还学习图像不是什么。基于这一概念，我们提出了一种新颖的DualAdapter方法，使VLMs能够从正面和负面两方面进行双路径适配，仅使用有限的注释样本。在推理阶段，我们的DualAdapter通过针对目标类别同时进行补充正向选择和负向排除，实现了统一预测，从而提高了VLMs在下游任务中的整体识别准确性。我们广泛的实验结果跨越15个数据集，验证了所提出的DualAda

    arXiv:2403.12964v1 Announce Type: cross  Abstract: Recently, large-scale pre-trained Vision-Language Models (VLMs) have demonstrated great potential in learning open-world visual representations, and exhibit remarkable performance across a wide range of downstream tasks through efficient fine-tuning. In this work, we innovatively introduce the concept of dual learning into fine-tuning VLMs, i.e., we not only learn what an image is, but also what an image isn't. Building on this concept, we introduce a novel DualAdapter approach to enable dual-path adaptation of VLMs from both positive and negative perspectives with only limited annotated samples. In the inference stage, our DualAdapter performs unified predictions by simultaneously conducting complementary positive selection and negative exclusion across target classes, thereby enhancing the overall recognition accuracy of VLMs in downstream tasks. Our extensive experimental results across 15 datasets validate that the proposed DualAda
    
[^3]: API保护的LLMs的标志泄露专有信息

    Logits of API-Protected LLMs Leak Proprietary Information

    [https://arxiv.org/abs/2403.09539](https://arxiv.org/abs/2403.09539)

    大多数现代LLM受到softmax瓶颈影响，可以以较低成本获取API保护的LLM的非公开信息和解锁多种功能

    

    大型语言模型（LLMs）的商业化导致了高级API-only接入专有模型的常见实践。在这项工作中，我们展示了即使对于模型架构有保守的假设，也可以从相对较少的API查询中学习关于API保护的LLM的大量非公开信息（例如，使用OpenAI的gpt-3.5-turbo仅花费不到1000美元）。我们的发现集中在一个关键观察上：大多数现代LLM受到了softmax瓶颈的影响，这限制了模型输出到完整输出空间的线性子空间。我们表明，这导致了一个模型图像或模型签名，从而以较低的成本解锁了几种功能：有效发现LLM的隐藏大小，获取完整词汇输出，检测和消除不同模型更新，识别给定单个完整LLM输出的源LLM，以及...

    arXiv:2403.09539v1 Announce Type: cross  Abstract: The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption about the model architecture, it is possible to learn a surprisingly large amount of non-public information about an API-protected LLM from a relatively small number of API queries (e.g., costing under $1,000 for OpenAI's gpt-3.5-turbo). Our findings are centered on one key observation: most modern LLMs suffer from a softmax bottleneck, which restricts the model outputs to a linear subspace of the full output space. We show that this lends itself to a model image or a model signature which unlocks several capabilities with affordable cost: efficiently discovering the LLM's hidden size, obtaining full-vocabulary outputs, detecting and disambiguating different model updates, identifying the source LLM given a single full LLM output, and eve
    
[^4]: Curry-DPO：利用课程学习和排名偏好增强对齐

    Curry-DPO: Enhancing Alignment using Curriculum Learning & Ranked Preferences

    [https://arxiv.org/abs/2403.07230](https://arxiv.org/abs/2403.07230)

    提出了一种名为Curry-DPO的方法，在直接偏好优化(DPO)中利用课程学习方法，通过构建多个偏好对来训练模型，相比于标准单一对DPO设置有着更好的性能表现。

    

    直接偏好优化(DPO)是一种有效的技术，利用成对偏好数据(通常是每个用户提示选择和拒绝的响应对)将LLMs与人类偏好对齐。在实践中，对于给定提示可能会存在多个响应，这些响应的质量相对于彼此而言有所不同。有了这些多个响应的质量评级，我们提出利用这些响应为给定提示创建多个偏好对。我们的工作侧重于通过课程学习方法系统地利用构建的多个偏好对来进行DPO训练。特别是，我们根据不同的标准将这些多个偏好数据对从易到难(模拟课程训练)排序。我们详细比较了我们提出的方法与标准单一对DPO设置。我们的方法，我们称之为Curry-DPO，在MTbench、Vicuna、Wiz上始终表现出增强的性能收益。

    arXiv:2403.07230v1 Announce Type: cross  Abstract: Direct Preference Optimization (DPO) is an effective technique that leverages pairwise preference data (usually one chosen and rejected response pair per user prompt) to align LLMs to human preferences. In practice, multiple responses can exist for a given prompt with varying quality relative to each other. With availability of such quality ratings for multiple responses, we propose utilizing these responses to create multiple preference pairs for a given prompt. Our work focuses on systematically using the constructed multiple preference pair in DPO training via curriculum learning methodology. In particular, we order these multiple pairs of preference data from easy to hard (emulating curriculum training) according to various criteria. We show detailed comparisons of our proposed approach to the standard single-pair DPO setting. Our method, which we call Curry-DPO consistently shows increased performance gains on MTbench, Vicuna, Wiz
    
[^5]: 通过探查采样加速贪婪坐标梯度

    Accelerating Greedy Coordinate Gradient via Probe Sampling

    [https://arxiv.org/abs/2403.01251](https://arxiv.org/abs/2403.01251)

    研究引入了一种名为“探查采样”的新算法，通过动态确定草稿模型和目标模型的相似度，来加速贪婪坐标梯度算法，实现高达5.6倍的加速。

    

    大型语言模型（LLMs）的安全性已成为一个中心问题，考虑到它们的快速发展和广泛应用。研究表明，贪婪坐标梯度（GCG）在构建包含对抗后缀的提示时非常有效，以破坏被认为是安全的LLMs，但GCG的优化耗时较长，限制了其实用性。为了减少GCG的时间成本并实现对LLMs安全性更全面的研究，在这项工作中，我们研究了一种称为“探查采样”的新算法，以加速GCG算法。该算法的核心是一种机制，动态确定较小草稿模型的预测与目标模型的提示候选预测的相似程度。当目标模型与草稿模型相似时，我们大量依赖于草稿模型来过滤大量潜在提示候选，以减少计算时间。探查采样使用Llam实现高达5.6倍的加速。

    arXiv:2403.01251v1 Announce Type: new  Abstract: Safety of Large Language Models (LLMs) has become a central issue given their rapid progress and wide applications. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing prompts containing adversarial suffixes to break the presumingly safe LLMs, but the optimization of GCG is time-consuming and limits its practicality. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$ to accelerate the GCG algorithm. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates to reduce the computation time. Probe sampling achieves up to $5.6$ times speedup using Llam
    
[^6]: 低资源和中等资源语言中的情感分类

    Emotion Classification in Low and Moderate Resource Languages

    [https://arxiv.org/abs/2402.18424](https://arxiv.org/abs/2402.18424)

    通过跨语言情感分类器，在低和中等资源语言中实现情感分类，展示了两种迁移学习方法的有效性。

    

    能够分析全球范围内人们情绪状态是很重要的。全球有7100多种活跃语言，为每种语言构建情感分类是一项劳动密集型工作。特别是对于低资源和濒危语言，建立情感分类可能非常具有挑战性。我们提出了一种跨语言情感分类器，我们在资源丰富的语言（例如我们的工作中的英语）上训练情感分类器，并将学习迁移到低资源和中等资源的语言。我们比较并对比了从高资源语言到低资源或中等资源语言的两种迁移学习方法。一种方法将高资源语言的标注投影到低资源和中等资源语言的平行语料库中，另一种方法直接将高资源语言的学习迁移到其他语言。我们展示了我们的方法在6种语言上的有效性：Fa

    arXiv:2402.18424v1 Announce Type: cross  Abstract: It is important to be able to analyze the emotional state of people around the globe. There are 7100+ active languages spoken around the world and building emotion classification for each language is labor intensive. Particularly for low-resource and endangered languages, building emotion classification can be quite challenging. We present a cross-lingual emotion classifier, where we train an emotion classifier with resource-rich languages (i.e. \textit{English} in our work) and transfer the learning to low and moderate resource languages. We compare and contrast two approaches of transfer learning from a high-resource language to a low or moderate-resource language. One approach projects the annotation from a high-resource language to low and moderate-resource language in parallel corpora and the other one uses direct transfer from high-resource language to the other languages. We show the efficacy of our approaches on 6 languages: Fa
    
[^7]: Hal-Eval: 一种面向大型视觉语言模型的通用和细粒度幻觉评估框架

    Hal-Eval: A Universal and Fine-grained Hallucination Evaluation Framework for Large Vision Language Models

    [https://arxiv.org/abs/2402.15721](https://arxiv.org/abs/2402.15721)

    本论文提出了Hal-Eval，一个通用和细粒度的幻觉评估框架，引入了新的幻觉分类法，专注于事件幻觉，通过生成和过滤细粒度幻觉数据来评估大型视觉语言模型对各种幻觉的处理能力。

    

    大型视觉语言模型具有非凡的能力，但在图片和其描述之间存在幻觉不一致。以往对LVLMs进行的幻觉评估研究发现了关于对象、属性和关系的幻觉，但忽略了围绕虚构实体创建整个叙事的复杂幻觉。本文引入了一种精细的幻觉分类法，其中包括一个新的类别：事件幻觉。然后，我们利用先进的LLMs生成和过滤由各种类型的幻觉组成的细粒度幻觉数据，特别关注事件幻觉，为在我们的通用评估框架内集成辨别和生成评估方法奠定基础。所提出的基准可以独特地评估LVLMs处理广泛幻觉的能力，使其成为一个可靠和全面的工具。

    arXiv:2402.15721v1 Announce Type: new  Abstract: Large Vision Language Models exhibit remarkable capabilities but struggle with hallucinations inconsistencies between images and their descriptions. Previous hallucination evaluation studies on LVLMs have identified hallucinations in terms of objects, attributes, and relations but overlooked complex hallucinations that create an entire narrative around a fictional entity. In this paper, we introduce a refined taxonomy of hallucinations, featuring a new category: Event Hallucination. We then utilize advanced LLMs to generate and filter fine grained hallucinatory data consisting of various types of hallucinations, with a particular focus on event hallucinations, laying the groundwork for integrating discriminative and generative evaluation methods within our universal evaluation framework. The proposed benchmark distinctively assesses LVLMs ability to tackle a broad spectrum of hallucinations, making it a reliable and comprehensive tool fo
    
[^8]: 鲁棒的提示优化用于对抗语言模型的破解攻击

    Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks

    [https://arxiv.org/abs/2401.17263](https://arxiv.org/abs/2401.17263)

    该论文提出了一种鲁棒的提示优化算法（RPO）用于对抗语言模型的破解攻击，通过梯度优化来确保输出的无害性，并成功降低了攻击成功率。

    

    尽管在人工智能对齐方面取得了一些进展，但语言模型（LM）仍然容易受到对抗性攻击或破解攻击的影响，其中对手修改输入提示以诱导有害行为。虽然已经提出了一些防御方法，但它们仅关注狭窄的威胁模型，并不能提供强大的防御。为了实现强大的防御，我们首次提出了用于对抗破解攻击的对抗目标，并提出了一种名为鲁棒提示优化（RPO）的算法，该算法利用基于梯度的令牌优化来确保输出的无害性。通过这种方法，我们得到了一个易于访问的后缀，显著改善了对破解攻击的强韧性，包括优化过程中出现的破解攻击以及未知的破解攻击，将攻击成功率从84%降低到8.66%，在20个破解攻击中。此外，我们还发现RPO对正常LM使用的影响较小，在适应性攻击下仍然有效，并且可以迁移到黑盒模型中，降低攻击成功率。

    Despite advances in AI alignment, language models (LM) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries modify input prompts to induce harmful behavior. While some defenses have been proposed, they focus on narrow threat models and fall short of a strong defense, which we posit should be effective, universal, and practical. To achieve this, we propose the first adversarial objective for defending LMs against jailbreaking attacks and an algorithm, robust prompt optimization (RPO), that uses gradient-based token optimization to enforce harmless outputs. This results in an easily accessible suffix that significantly improves robustness to both jailbreaks seen during optimization and unknown, held-out jailbreaks, reducing the attack success rate on Starling-7B from 84% to 8.66% across 20 jailbreaks. In addition, we find that RPO has a minor effect on normal LM use, is successful under adaptive attacks, and can transfer to black-box models, reducing the success
    
[^9]: 用于量化生成式语言模型不确定性的PAC神经预测集学习

    PAC Neural Prediction Set Learning to Quantify the Uncertainty of Generative Language Models. (arXiv:2307.09254v1 [cs.LG])

    [http://arxiv.org/abs/2307.09254](http://arxiv.org/abs/2307.09254)

    本文提出了一种使用神经网络来量化生成式语言模型不确定性的PAC神经预测集学习方法，通过在多种语言数据集和模型上的实验证明，相比于标准基准方法，我们的方法平均提高了63％的量化不确定性。

    

    学习和量化模型的不确定性是增强模型可信度的关键任务。由于对生成虚构事实的担忧，最近兴起的生成式语言模型（GLM）特别强调可靠的不确定性量化的需求。本文提出了一种学习神经预测集模型的方法，该方法能够以可能近似正确（PAC）的方式量化GLM的不确定性。与现有的预测集模型通过标量值参数化不同，我们提出通过神经网络参数化预测集，实现更精确的不确定性量化，但仍满足PAC保证。通过在四种类型的语言数据集和六种类型的模型上展示，我们的方法相比标准基准方法平均提高了63％的量化不确定性。

    Uncertainty learning and quantification of models are crucial tasks to enhance the trustworthiness of the models. Importantly, the recent surge of generative language models (GLMs) emphasizes the need for reliable uncertainty quantification due to the concerns on generating hallucinated facts. In this paper, we propose to learn neural prediction set models that comes with the probably approximately correct (PAC) guarantee for quantifying the uncertainty of GLMs. Unlike existing prediction set models, which are parameterized by a scalar value, we propose to parameterize prediction sets via neural networks, which achieves more precise uncertainty quantification but still satisfies the PAC guarantee. We demonstrate the efficacy of our method on four types of language datasets and six types of models by showing that our method improves the quantified uncertainty by $63\%$ on average, compared to a standard baseline method.
    

