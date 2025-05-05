# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Wiki-TabNER:Advancing Table Interpretation Through Named Entity Recognition](https://arxiv.org/abs/2403.04577) | 本文提出了一个新的挑战性数据集，并介绍了一个旨在解决实体链接任务的新问题：单元格内的命名实体识别，并提出了一个提示框架用于评估大型语言模型在这一新任务上的效果。 |
| [^2] | [FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning](https://arxiv.org/abs/2402.18789) | FlexLLM是第一个可以在同一迭代中共同提供推理和参数高效微调请求的系统，通过引入标记级微调机制实现共享GPU资源的高效利用 |
| [^3] | [Rethinking Scientific Summarization Evaluation: Grounding Explainable Metrics on Facet-aware Benchmark](https://arxiv.org/abs/2402.14359) | 该论文提出了一种基于方面感知的评估指标（FM），利用大型语言模型对摘要进行高级语义匹配，提供了一种全面评估科学摘要的方法。 |
| [^4] | [AutoPlanBench: Automatically generating benchmarks for LLM planners from PDDL](https://arxiv.org/abs/2311.09830) | AutoPlanBench是一种新方法，可以自动转换PDDL规划基准测试为文本描述，并提供了相应的基准测试数据集。研究表明，当前最好的LLM规划器在某些规划任务上表现优秀，但对于其他任务来说仍存在挑战。 |

# 详细

[^1]: Wiki-TabNER:通过命名实体识别推进表格解释

    Wiki-TabNER:Advancing Table Interpretation Through Named Entity Recognition

    [https://arxiv.org/abs/2403.04577](https://arxiv.org/abs/2403.04577)

    本文提出了一个新的挑战性数据集，并介绍了一个旨在解决实体链接任务的新问题：单元格内的命名实体识别，并提出了一个提示框架用于评估大型语言模型在这一新任务上的效果。

    

    arXiv:2403.04577v1 发布类型：新摘要：网络表格包含大量宝贵知识，激发了旨在解决表格解释（TI）任务的表格语言模型。本文分析了用于评估TI任务的广泛使用的基准数据集，特别关注实体链接任务。我们的分析显示，该数据集过于简化，可能降低其用于全面评估的有效性，并未准确代表表格在现实世界中的外观。为克服这一缺点，我们构建并注释了一个更具挑战性的新数据集。除了介绍新数据集外，我们还介绍了一个旨在解决实体链接任务的新问题：单元格内的命名实体识别。最后，我们提出了一个提示框架，用于评估新开发的大型语言模型（LLMs）在这一新的TI任务上。我们在各种设置下对提示LLMs进行实验证明，其中我们同时使用了随机

    arXiv:2403.04577v1 Announce Type: new  Abstract: Web tables contain a large amount of valuable knowledge and have inspired tabular language models aimed at tackling table interpretation (TI) tasks. In this paper, we analyse a widely used benchmark dataset for evaluation of TI tasks, particularly focusing on the entity linking task. Our analysis reveals that this dataset is overly simplified, potentially reducing its effectiveness for thorough evaluation and failing to accurately represent tables as they appear in the real-world. To overcome this drawback, we construct and annotate a new more challenging dataset. In addition to introducing the new dataset, we also introduce a novel problem aimed at addressing the entity linking task: named entity recognition within cells. Finally, we propose a prompting framework for evaluating the newly developed large language models (LLMs) on this novel TI task. We conduct experiments on prompting LLMs under various settings, where we use both random
    
[^2]: FlexLLM：一种用于共同提供大型语言模型推理和参数高效微调的系统

    FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning

    [https://arxiv.org/abs/2402.18789](https://arxiv.org/abs/2402.18789)

    FlexLLM是第一个可以在同一迭代中共同提供推理和参数高效微调请求的系统，通过引入标记级微调机制实现共享GPU资源的高效利用

    

    Parameter-efficient finetuning（PEFT）是一种广泛使用的技术，用于为不同任务调整大型语言模型。通常，服务提供商会为用户创建单独的系统，以执行PEFT模型微调和推理任务。这是因为现有系统无法处理包含推理和PEFT微调请求混合的工作负载。因此，共享的GPU资源利用不足，导致效率低下。为解决这一问题，我们提出了FlexLLM，这是第一个可以在同一迭代中为推理和参数高效微调请求提供服务的系统。我们的系统利用这两个任务的互补性质，并利用共享的GPU资源来共同运行它们，使用一种称为共同提供的方法。为实现这一目标，FlexLLM引入了一种新颖的标记级微调机制，将序列的微调计算分解为更小的标记级计算，并使用依赖并行化。

    arXiv:2402.18789v1 Announce Type: cross  Abstract: Parameter-efficient finetuning (PEFT) is a widely used technique to adapt large language models for different tasks. Service providers typically create separate systems for users to perform PEFT model finetuning and inference tasks. This is because existing systems cannot handle workloads that include a mix of inference and PEFT finetuning requests. As a result, shared GPU resources are underutilized, leading to inefficiencies. To address this problem, we present FlexLLM, the first system that can serve inference and parameter-efficient finetuning requests in the same iteration. Our system leverages the complementary nature of these two tasks and utilizes shared GPU resources to run them jointly, using a method called co-serving. To achieve this, FlexLLM introduces a novel token-level finetuning mechanism, which breaks down the finetuning computation of a sequence into smaller token-level computations and uses dependent parallelization
    
[^3]: 重新思考科学摘要评估：基于方面感知基准的可解释度指标

    Rethinking Scientific Summarization Evaluation: Grounding Explainable Metrics on Facet-aware Benchmark

    [https://arxiv.org/abs/2402.14359](https://arxiv.org/abs/2402.14359)

    该论文提出了一种基于方面感知的评估指标（FM），利用大型语言模型对摘要进行高级语义匹配，提供了一种全面评估科学摘要的方法。

    

    预训练和大型语言模型（LLMs）的摘要能力在一般领域中得到了广泛验证，但它们在涉及复杂句子和专业知识的科学语料库中的使用较少被评估。该论文提出了科学摘要的概念和实验分析，突出了传统评估方法（如$n$-gram、嵌入比较和问答）在提供解释、把握科学概念或识别关键内容方面的不足之处。随后，我们介绍了Facet-aware Metric（FM），利用LLMs进行高级语义匹配，根据不同方面评估摘要。这种面向方面的方法通过将评估任务分解为更简单的子任务，为摘要提供了全面的评估。鉴于该领域缺乏评估基准，我们精心策划了一个基于方面的科学摘要数据集（FD）。

    arXiv:2402.14359v1 Announce Type: new  Abstract: The summarization capabilities of pretrained and large language models (LLMs) have been widely validated in general areas, but their use in scientific corpus, which involves complex sentences and specialized knowledge, has been less assessed. This paper presents conceptual and experimental analyses of scientific summarization, highlighting the inadequacies of traditional evaluation methods, such as $n$-gram, embedding comparison, and QA, particularly in providing explanations, grasping scientific concepts, or identifying key content. Subsequently, we introduce the Facet-aware Metric (FM), employing LLMs for advanced semantic matching to evaluate summaries based on different aspects. This facet-aware approach offers a thorough evaluation of abstracts by decomposing the evaluation task into simpler subtasks.Recognizing the absence of an evaluation benchmark in this domain, we curate a Facet-based scientific summarization Dataset (FD) with 
    
[^4]: AutoPlanBench: 从PDDL自动生成LLM规划器的基准测试

    AutoPlanBench: Automatically generating benchmarks for LLM planners from PDDL

    [https://arxiv.org/abs/2311.09830](https://arxiv.org/abs/2311.09830)

    AutoPlanBench是一种新方法，可以自动转换PDDL规划基准测试为文本描述，并提供了相应的基准测试数据集。研究表明，当前最好的LLM规划器在某些规划任务上表现优秀，但对于其他任务来说仍存在挑战。

    

    LLMs（逻辑-概率模型）在规划任务中的应用越来越广泛，但是它们在规划和推理方面的能力尚不明确。我们提出了AutoPlanBench，一种将PDDL中的规划基准测试自动转换为文本描述的新方法，并提供了使用我们方法创建的基准测试数据集。我们展示了最好的LLM规划器在某些规划任务上表现良好，但其他任务仍然超出了当前方法的能力范围。

    LLMs are being increasingly used for planning-style tasks, but their capabilities for planning and reasoning are poorly understood. We present AutoPlanBench, a novel method for automatically converting planning benchmarks written in PDDL into textual descriptions and offer a benchmark dataset created with our method. We show that while the best LLM planners do well on some planning tasks, others remain out of reach of current methods.
    

