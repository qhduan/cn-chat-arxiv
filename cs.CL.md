# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability](https://arxiv.org/abs/2403.04483) | 该论文提出了一个名为GraphInstruct的基准，用于评估和增强大规模语言模型的图理解能力，并通过构建GraphLM和提出GraphLM+模型实现了显著的图推理能力增强。 |
| [^2] | [Large Language Models As Faithful Explainers](https://arxiv.org/abs/2402.04678) | 本论文提出了一个生成解释框架（xLLM），用于提高大型语言模型（LLMs）自然语言格式解释的可信度。通过一个评估器来量化解释的可信度，并通过迭代优化过程来提高可信度。 |
| [^3] | [DocFinQA: A Long-Context Financial Reasoning Dataset](https://arxiv.org/abs/2401.06915) | 引入了一个长文档财务问答任务，将平均上下文长度从700个词扩展到123k个词，对于大型语言模型在金融领域具有重要挑战。 |

# 详细

[^1]: 使用图理解和推理功能增强大规模语言模型的GraphInstruct

    GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability

    [https://arxiv.org/abs/2403.04483](https://arxiv.org/abs/2403.04483)

    该论文提出了一个名为GraphInstruct的基准，用于评估和增强大规模语言模型的图理解能力，并通过构建GraphLM和提出GraphLM+模型实现了显著的图推理能力增强。

    

    评估和增强大规模语言模型（LLMs）的通用能力一直是一个重要的研究课题。图是现实世界中常见的数据结构，理解图数据对于推进通用智能至关重要。为了评估和增强LLMs的图理解能力，在本文中，我们提出了一个名为GraphInstruct的基准，全面包括21个经典图推理任务，提供多样的图生成流水线和详细的推理步骤。基于GraphInstruct，我们进一步通过高效的指导调整构建了GraphLM，展示出显著的图理解能力。为了增强LLM的图推理能力，我们提出了一种步骤掩码训练策略，并构建了一个名为GraphLM+的模型。作为增强LLMs图理解和推理能力的先驱性努力之一，我们进行了大量实验。

    arXiv:2403.04483v1 Announce Type: new  Abstract: Evaluating and enhancing the general capabilities of large language models (LLMs) has been an important research topic. Graph is a common data structure in the real world, and understanding graph data is a crucial part for advancing general intelligence. To evaluate and enhance the graph understanding abilities of LLMs, in this paper, we propose a benchmark named GraphInstruct, which comprehensively includes 21 classical graph reasoning tasks, providing diverse graph generation pipelines and detailed reasoning steps. Based on GraphInstruct, we further construct GraphLM through efficient instruction-tuning, which shows prominent graph understanding capability. In order to enhance the LLM with graph reasoning capability as well, we propose a step mask training strategy, and construct a model named GraphLM+. As one of the pioneering efforts to enhance the graph understanding and reasoning abilities of LLMs, extensive experiments have demons
    
[^2]: 大型语言模型作为可信的解释器

    Large Language Models As Faithful Explainers

    [https://arxiv.org/abs/2402.04678](https://arxiv.org/abs/2402.04678)

    本论文提出了一个生成解释框架（xLLM），用于提高大型语言模型（LLMs）自然语言格式解释的可信度。通过一个评估器来量化解释的可信度，并通过迭代优化过程来提高可信度。

    

    近年来，大型语言模型(LLMs)通过利用其丰富的内部知识和推理能力，已经能够熟练解决复杂的任务。然而，这种复杂性阻碍了传统的以输入为重点的解释算法来解释LLMs的复杂决策过程。为了解决这个问题，最近出现了一种自我解释机制，通过自然语言的形式进行单向推理，从而实现对LLMs预测的解释。然而，这种自然语言解释经常因为缺乏可信度而受到批评，因为这些解释可能不准确地反映LLMs的决策行为。在这项工作中，我们引入了一个生成解释框架xLLM，以提高LLMs自然语言格式的解释的可信度。具体而言，我们提出了一个评估器来量化自然语言解释的可信度，并通过xLLM的迭代优化过程来提高可信度，目标是最大程度地提高可信度。

    Large Language Models (LLMs) have recently become proficient in addressing complex tasks by utilizing their rich internal knowledge and reasoning ability. Consequently, this complexity hinders traditional input-focused explanation algorithms for explaining the complex decision-making processes of LLMs. Recent advancements have thus emerged for self-explaining their predictions through a single feed-forward inference in a natural language format. However, natural language explanations are often criticized for lack of faithfulness since these explanations may not accurately reflect the decision-making behaviors of the LLMs. In this work, we introduce a generative explanation framework, xLLM, to improve the faithfulness of the explanations provided in natural language formats for LLMs. Specifically, we propose an evaluator to quantify the faithfulness of natural language explanation and enhance the faithfulness by an iterative optimization process of xLLM, with the goal of maximizing the 
    
[^3]: DocFinQA：一个长文本财务推理数据集

    DocFinQA: A Long-Context Financial Reasoning Dataset

    [https://arxiv.org/abs/2401.06915](https://arxiv.org/abs/2401.06915)

    引入了一个长文档财务问答任务，将平均上下文长度从700个词扩展到123k个词，对于大型语言模型在金融领域具有重要挑战。

    

    对于大型语言模型（LLMs）在金融领域发挥作用，需要研究现实任务和数据。金融专业人士经常与长达数百页的文档进行交互，但大多数金融研究数据集仅处理这些文档的简短摘录。为了解决这个问题，我们引入了一个长文档财务问答任务。我们通过在现有FinQA数据集中的7,437个问题中增加完整文档上下文，将FinQA中平均上下文长度从不到700个词扩展到DocFinQA中的123k个词。我们在检索式QA管道和长文本语言模型上进行了大量实验。即使对于最先进的系统，DocFinQA也是一个巨大挑战。我们还对DocFinQA中最长文档进行了案例研究，并发现模型在这些文档上特别困难。解决这些挑战。

    arXiv:2401.06915v2 Announce Type: replace-cross  Abstract: For large language models (LLMs) to be effective in the financial domain -- where each decision can have a significant impact -- it is necessary to investigate realistic tasks and data. Financial professionals often interact with documents that are hundreds of pages long, but most financial research datasets only deal with short excerpts from these documents. To address this, we introduce a long-document financial QA task. We augment 7,437 questions from the existing FinQA dataset with the full-document context, extending the average context length from under 700 words in FinQA to 123k words in DocFinQA. We conduct extensive experiments over retrieval-based QA pipelines and long-context language models. DocFinQA proves a significant challenge for even state-of-the-art systems. We also provide a case-study on the longest documents in DocFinQA and find that models particularly struggle on these documents. Addressing these challen
    

