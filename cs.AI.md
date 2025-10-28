# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability](https://arxiv.org/abs/2403.04483) | 该论文提出了一个名为GraphInstruct的基准，用于评估和增强大规模语言模型的图理解能力，并通过构建GraphLM和提出GraphLM+模型实现了显著的图推理能力增强。 |
| [^2] | [Diffusion Models Meet Contextual Bandits with Large Action Spaces](https://arxiv.org/abs/2402.10028) | 本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。 |
| [^3] | [Large Language Models As Faithful Explainers](https://arxiv.org/abs/2402.04678) | 本论文提出了一个生成解释框架（xLLM），用于提高大型语言模型（LLMs）自然语言格式解释的可信度。通过一个评估器来量化解释的可信度，并通过迭代优化过程来提高可信度。 |
| [^4] | [DocFinQA: A Long-Context Financial Reasoning Dataset](https://arxiv.org/abs/2401.06915) | 引入了一个长文档财务问答任务，将平均上下文长度从700个词扩展到123k个词，对于大型语言模型在金融领域具有重要挑战。 |
| [^5] | [Online POMDP Planning with Anytime Deterministic Guarantees.](http://arxiv.org/abs/2310.01791) | 本文中，我们推导出在线POMDP规划中一个简化解决方案与理论上最优解之间的确定性关系，以解决目前近似算法只能提供概率性和通常呈现渐进性保证的限制。 |
| [^6] | [Graph Neural Architecture Search with GPT-4.](http://arxiv.org/abs/2310.01436) | 本文将GPT-4集成到图神经网络架构搜索（GNAS）中，提出了一种新的GPT-4基于的GNAS方法（GPT4GNAS），通过设计新的提示来引导GPT-4生成更准确的图神经网络，实验证明嵌入GPT-4到GNAS中优于现有方法。 |
| [^7] | [Learnable Behavior Control: Breaking Atari Human World Records via Sample-Efficient Behavior Selection.](http://arxiv.org/abs/2305.05239) | 本文提出了一个通用的Learnable Behavioral Control (LBC)框架，使得行为选择空间得到扩大，并通过基于赌博机的元控制器实现行为控制。在Atari游戏上，我们的代理已经达到10个游戏的人类水平，并在7个游戏中达到了目前的最高分。 |
| [^8] | [Representer Theorems for Metric and Preference Learning: A Geometric Perspective.](http://arxiv.org/abs/2304.03720) | 该论文提出了度量学习和偏好学习的新的表现定理，解决了度量学习任务以三元组比较为基础的表现定理问题。这种表现定理可以用内积诱导的范数来表示。 |

# 详细

[^1]: 使用图理解和推理功能增强大规模语言模型的GraphInstruct

    GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability

    [https://arxiv.org/abs/2403.04483](https://arxiv.org/abs/2403.04483)

    该论文提出了一个名为GraphInstruct的基准，用于评估和增强大规模语言模型的图理解能力，并通过构建GraphLM和提出GraphLM+模型实现了显著的图推理能力增强。

    

    评估和增强大规模语言模型（LLMs）的通用能力一直是一个重要的研究课题。图是现实世界中常见的数据结构，理解图数据对于推进通用智能至关重要。为了评估和增强LLMs的图理解能力，在本文中，我们提出了一个名为GraphInstruct的基准，全面包括21个经典图推理任务，提供多样的图生成流水线和详细的推理步骤。基于GraphInstruct，我们进一步通过高效的指导调整构建了GraphLM，展示出显著的图理解能力。为了增强LLM的图推理能力，我们提出了一种步骤掩码训练策略，并构建了一个名为GraphLM+的模型。作为增强LLMs图理解和推理能力的先驱性努力之一，我们进行了大量实验。

    arXiv:2403.04483v1 Announce Type: new  Abstract: Evaluating and enhancing the general capabilities of large language models (LLMs) has been an important research topic. Graph is a common data structure in the real world, and understanding graph data is a crucial part for advancing general intelligence. To evaluate and enhance the graph understanding abilities of LLMs, in this paper, we propose a benchmark named GraphInstruct, which comprehensively includes 21 classical graph reasoning tasks, providing diverse graph generation pipelines and detailed reasoning steps. Based on GraphInstruct, we further construct GraphLM through efficient instruction-tuning, which shows prominent graph understanding capability. In order to enhance the LLM with graph reasoning capability as well, we propose a step mask training strategy, and construct a model named GraphLM+. As one of the pioneering efforts to enhance the graph understanding and reasoning abilities of LLMs, extensive experiments have demons
    
[^2]: 扩散模型与大动作空间情境强化学习的结合

    Diffusion Models Meet Contextual Bandits with Large Action Spaces

    [https://arxiv.org/abs/2402.10028](https://arxiv.org/abs/2402.10028)

    本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。

    

    由于动作空间较大，有效的探索是情境强化学习中的一个关键挑战。本文通过利用预训练的扩散模型来捕捉动作之间的相关性，设计了扩散汤普森采样（dTS）方法，实现了高效的探索。我们为dTS方法提供了理论和算法基础，并通过实证评估展示了它的优越性能。

    arXiv:2402.10028v1 Announce Type: cross  Abstract: Efficient exploration is a key challenge in contextual bandits due to the large size of their action space, where uninformed exploration can result in computational and statistical inefficiencies. Fortunately, the rewards of actions are often correlated and this can be leveraged to explore them efficiently. In this work, we capture such correlations using pre-trained diffusion models; upon which we design diffusion Thompson sampling (dTS). Both theoretical and algorithmic foundations are developed for dTS, and empirical evaluation also shows its favorable performance.
    
[^3]: 大型语言模型作为可信的解释器

    Large Language Models As Faithful Explainers

    [https://arxiv.org/abs/2402.04678](https://arxiv.org/abs/2402.04678)

    本论文提出了一个生成解释框架（xLLM），用于提高大型语言模型（LLMs）自然语言格式解释的可信度。通过一个评估器来量化解释的可信度，并通过迭代优化过程来提高可信度。

    

    近年来，大型语言模型(LLMs)通过利用其丰富的内部知识和推理能力，已经能够熟练解决复杂的任务。然而，这种复杂性阻碍了传统的以输入为重点的解释算法来解释LLMs的复杂决策过程。为了解决这个问题，最近出现了一种自我解释机制，通过自然语言的形式进行单向推理，从而实现对LLMs预测的解释。然而，这种自然语言解释经常因为缺乏可信度而受到批评，因为这些解释可能不准确地反映LLMs的决策行为。在这项工作中，我们引入了一个生成解释框架xLLM，以提高LLMs自然语言格式的解释的可信度。具体而言，我们提出了一个评估器来量化自然语言解释的可信度，并通过xLLM的迭代优化过程来提高可信度，目标是最大程度地提高可信度。

    Large Language Models (LLMs) have recently become proficient in addressing complex tasks by utilizing their rich internal knowledge and reasoning ability. Consequently, this complexity hinders traditional input-focused explanation algorithms for explaining the complex decision-making processes of LLMs. Recent advancements have thus emerged for self-explaining their predictions through a single feed-forward inference in a natural language format. However, natural language explanations are often criticized for lack of faithfulness since these explanations may not accurately reflect the decision-making behaviors of the LLMs. In this work, we introduce a generative explanation framework, xLLM, to improve the faithfulness of the explanations provided in natural language formats for LLMs. Specifically, we propose an evaluator to quantify the faithfulness of natural language explanation and enhance the faithfulness by an iterative optimization process of xLLM, with the goal of maximizing the 
    
[^4]: DocFinQA：一个长文本财务推理数据集

    DocFinQA: A Long-Context Financial Reasoning Dataset

    [https://arxiv.org/abs/2401.06915](https://arxiv.org/abs/2401.06915)

    引入了一个长文档财务问答任务，将平均上下文长度从700个词扩展到123k个词，对于大型语言模型在金融领域具有重要挑战。

    

    对于大型语言模型（LLMs）在金融领域发挥作用，需要研究现实任务和数据。金融专业人士经常与长达数百页的文档进行交互，但大多数金融研究数据集仅处理这些文档的简短摘录。为了解决这个问题，我们引入了一个长文档财务问答任务。我们通过在现有FinQA数据集中的7,437个问题中增加完整文档上下文，将FinQA中平均上下文长度从不到700个词扩展到DocFinQA中的123k个词。我们在检索式QA管道和长文本语言模型上进行了大量实验。即使对于最先进的系统，DocFinQA也是一个巨大挑战。我们还对DocFinQA中最长文档进行了案例研究，并发现模型在这些文档上特别困难。解决这些挑战。

    arXiv:2401.06915v2 Announce Type: replace-cross  Abstract: For large language models (LLMs) to be effective in the financial domain -- where each decision can have a significant impact -- it is necessary to investigate realistic tasks and data. Financial professionals often interact with documents that are hundreds of pages long, but most financial research datasets only deal with short excerpts from these documents. To address this, we introduce a long-document financial QA task. We augment 7,437 questions from the existing FinQA dataset with the full-document context, extending the average context length from under 700 words in FinQA to 123k words in DocFinQA. We conduct extensive experiments over retrieval-based QA pipelines and long-context language models. DocFinQA proves a significant challenge for even state-of-the-art systems. We also provide a case-study on the longest documents in DocFinQA and find that models particularly struggle on these documents. Addressing these challen
    
[^5]: 具有任意确定性保证的在线POMDP规划

    Online POMDP Planning with Anytime Deterministic Guarantees. (arXiv:2310.01791v1 [cs.AI])

    [http://arxiv.org/abs/2310.01791](http://arxiv.org/abs/2310.01791)

    本文中，我们推导出在线POMDP规划中一个简化解决方案与理论上最优解之间的确定性关系，以解决目前近似算法只能提供概率性和通常呈现渐进性保证的限制。

    

    在现实场景中，自主智能体经常遇到不确定性并基于不完整信息做出决策。在不确定性下的规划可以使用部分可观察的马尔科夫决策过程（POMDP）进行数学建模。然而，寻找POMDP的最优规划在计算上是昂贵的，只有在小规模任务中可行。近年来，近似算法（如树搜索和基于采样的方法）已经成为解决较大问题的先进POMDP求解器。尽管这些算法有效，但它们仅提供概率性和通常呈现渐进性保证，这是由于它们依赖于采样的缘故。为了解决这些限制，我们推导出一个简化解决方案与理论上最优解之间的确定性关系。首先，我们推导出选择一组观测以在计算每个后验节点时分支的边界。

    Autonomous agents operating in real-world scenarios frequently encounter uncertainty and make decisions based on incomplete information. Planning under uncertainty can be mathematically formalized using partially observable Markov decision processes (POMDPs). However, finding an optimal plan for POMDPs can be computationally expensive and is feasible only for small tasks. In recent years, approximate algorithms, such as tree search and sample-based methodologies, have emerged as state-of-the-art POMDP solvers for larger problems. Despite their effectiveness, these algorithms offer only probabilistic and often asymptotic guarantees toward the optimal solution due to their dependence on sampling. To address these limitations, we derive a deterministic relationship between a simplified solution that is easier to obtain and the theoretically optimal one. First, we derive bounds for selecting a subset of the observations to branch from while computing a complete belief at each posterior nod
    
[^6]: 使用GPT-4的图神经网络架构搜索

    Graph Neural Architecture Search with GPT-4. (arXiv:2310.01436v1 [cs.LG])

    [http://arxiv.org/abs/2310.01436](http://arxiv.org/abs/2310.01436)

    本文将GPT-4集成到图神经网络架构搜索（GNAS）中，提出了一种新的GPT-4基于的GNAS方法（GPT4GNAS），通过设计新的提示来引导GPT-4生成更准确的图神经网络，实验证明嵌入GPT-4到GNAS中优于现有方法。

    

    图神经网络架构搜索（GNAS）在自动设计图神经网络方面取得了有希望的结果。然而，GNAS仍然需要大量的人工劳动和丰富的领域知识来设计搜索空间和搜索策略。本文将GPT-4集成到GNAS中，提出了一种基于GPT-4的图神经网络架构搜索方法（简称为GPT4GNAS）。我们的方法的基本思想是为GPT-4设计一类新的提示，以指导GPT-4进行图神经网络架构的生成任务。这些提示包括GNAS的搜索空间、搜索策略和搜索反馈的描述。通过迭代地运行具有提示的GPT-4，GPT4GNAS能够生成更准确的图神经网络，并快速收敛。实验结果表明，嵌入GPT-4到GNAS中优于现有最先进的GNAS方法。

    Graph Neural Architecture Search (GNAS) has shown promising results in automatically designing graph neural networks. However, GNAS still requires intensive human labor with rich domain knowledge to design the search space and search strategy. In this paper, we integrate GPT-4 into GNAS and propose a new GPT-4 based Graph Neural Architecture Search method (GPT4GNAS for short). The basic idea of our method is to design a new class of prompts for GPT-4 to guide GPT-4 toward the generative task of graph neural architectures. The prompts consist of descriptions of the search space, search strategy, and search feedback of GNAS. By iteratively running GPT-4 with the prompts, GPT4GNAS generates more accurate graph neural networks with fast convergence. Experimental results show that embedding GPT-4 into GNAS outperforms the state-of-the-art GNAS methods.
    
[^7]: 可学习的行为控制：通过高效行为选择打破Atari人类世界记录

    Learnable Behavior Control: Breaking Atari Human World Records via Sample-Efficient Behavior Selection. (arXiv:2305.05239v1 [cs.LG])

    [http://arxiv.org/abs/2305.05239](http://arxiv.org/abs/2305.05239)

    本文提出了一个通用的Learnable Behavioral Control (LBC)框架，使得行为选择空间得到扩大，并通过基于赌博机的元控制器实现行为控制。在Atari游戏上，我们的代理已经达到10个游戏的人类水平，并在7个游戏中达到了目前的最高分。

    

    在深度强化学习中，探索问题是主要挑战之一。最近，一些有希望的工作尝试使用基于群体的方法来处理这个问题，通过从不同探索策略的人群中收集具有不同行为的样本。自适应策略选择已被用于行为控制。然而，行为选择空间在很大程度上受到预定义策略种群的限制，这进一步限制了行为多样性。在本文中，我们提出了一个通用框架称为可学习的行为控制（LBC）来解决这种限制。该框架a)通过从所有策略中制定混合行为映射，实现了显著扩大的行为选择空间；b)构建了一个统一的可学习的行为选择过程。我们将LBC引入分布式离线演员-评论家方法中，并通过基于赌博机的元控制器优化行为映射的选择来实现行为控制。我们的代理已经在10个Atari游戏中达到了人类水平，并在7个游戏中达到了目前的最高分。我们还展示了LBC框架的良好泛化能力，并在机器人控制任务上进行了测试。

    The exploration problem is one of the main challenges in deep reinforcement learning (RL). Recent promising works tried to handle the problem with population-based methods, which collect samples with diverse behaviors derived from a population of different exploratory policies. Adaptive policy selection has been adopted for behavior control. However, the behavior selection space is largely limited by the predefined policy population, which further limits behavior diversity. In this paper, we propose a general framework called Learnable Behavioral Control (LBC) to address the limitation, which a) enables a significantly enlarged behavior selection space via formulating a hybrid behavior mapping from all policies; b) constructs a unified learnable process for behavior selection. We introduce LBC into distributed off-policy actor-critic methods and achieve behavior control via optimizing the selection of the behavior mappings with bandit-based meta-controllers. Our agents have achieved 10
    
[^8]: 度量学习与偏好学习的表现定理：基于几何的视角

    Representer Theorems for Metric and Preference Learning: A Geometric Perspective. (arXiv:2304.03720v1 [cs.LG])

    [http://arxiv.org/abs/2304.03720](http://arxiv.org/abs/2304.03720)

    该论文提出了度量学习和偏好学习的新的表现定理，解决了度量学习任务以三元组比较为基础的表现定理问题。这种表现定理可以用内积诱导的范数来表示。

    

    我们探讨了希尔伯特空间中的度量学习和偏好学习问题，并获得了一种新的度量学习和偏好学习的表现定理。我们的关键观察是，表现定理可以根据问题结构内在的内积所诱导的范数来表示。此外，我们展示了如何将我们的框架应用于三元组比较的度量学习任务，并展示它导致了一个简单且自包含的该任务的表现定理。在再生核希尔伯特空间(RKHS)的情况下，我们展示了学习问题的解可以使用类似于经典表现定理的核术语表示。

    We explore the metric and preference learning problem in Hilbert spaces. We obtain a novel representer theorem for the simultaneous task of metric and preference learning. Our key observation is that the representer theorem can be formulated with respect to the norm induced by the inner product inherent in the problem structure. Additionally, we demonstrate how our framework can be applied to the task of metric learning from triplet comparisons and show that it leads to a simple and self-contained representer theorem for this task. In the case of Reproducing Kernel Hilbert Spaces (RKHS), we demonstrate that the solution to the learning problem can be expressed using kernel terms, akin to classical representer theorems.
    

