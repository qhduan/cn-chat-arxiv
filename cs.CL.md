# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [K-Level Reasoning with Large Language Models](https://rss.arxiv.org/abs/2402.01521) | 该论文探索了使用大型语言模型进行动态决策推理的能力，并提出了一种名为"K级推理"的新颖推理方法。通过博弈论的试验，发现现有的推理方法在动态环境中容易出错，而"K级推理"可以解决这个问题。 |
| [^2] | [PROMPT-SAW: Leveraging Relation-Aware Graphs for Textual Prompt Compression](https://arxiv.org/abs/2404.00489) | 提出了PROMPT-SAW模型，利用关系感知图来实现文本提示的压缩，提高了提示的可读性和可解释性。 |
| [^3] | [CoLLEGe: Concept Embedding Generation for Large Language Models](https://arxiv.org/abs/2403.15362) | CoLLEGe是一个元学习框架，能够为大型语言模型生成灵活的新概念嵌入，用于现代化少样本概念学习。 |
| [^4] | [Pragmatic Competence Evaluation of Large Language Models for Korean](https://arxiv.org/abs/2403.12675) | 该研究将大型语言模型的评估从对嵌入知识的基准拓展到了探索语用能力，结果显示在韩语环境下，GPT-4在传统和人工评估设置中表现出色，而HyperCLOVA X也在开放式问题评估中取得了不错的成绩。 |
| [^5] | [ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547) | ActiveRAG是一个创新的RAG框架，通过引入主动学习机制，利用知识构建和认知联结机制来提升大型语言模型（LLMs）的内在认知，实现了明显的性能提升。 |
| [^6] | [Tables as Images? Exploring the Strengths and Limitations of LLMs on Multimodal Representations of Tabular Data](https://arxiv.org/abs/2402.12424) | 本研究探讨了LLM在解释表格数据方面的有效性，比较了文本和图像表格表示对LLM性能的影响，为在表格相关任务上有效使用LLM提供了见解。 |
| [^7] | [MORL-Prompt: An Empirical Analysis of Multi-Objective Reinforcement Learning for Discrete Prompt Optimization](https://arxiv.org/abs/2402.11711) | 本研究将多目标优化技术应用于基于强化学习的离散提示优化，为解决奖励平衡问题提供了新视角。 |
| [^8] | [BLT: Can Large Language Models Handle Basic Legal Text?](https://arxiv.org/abs/2311.09693) | 大型语言模型在处理基础法律文本方面表现不佳，但通过针对性微调，甚至较小的模型也能在测试中表现出色，提升了相关法律任务的表现。 |
| [^9] | [Experimental Contexts Can Facilitate Robust Semantic Property Inference in Language Models, but Inconsistently.](http://arxiv.org/abs/2401.06640) | 本研究通过控制实验环境的方式，发现语言模型在属性继承任务中表现出了一定的非平凡能力，但这种能力是不一致的。 |
| [^10] | [Investigating Chain-of-thought with ChatGPT for Stance Detection on Social Media.](http://arxiv.org/abs/2304.03087) | 本文研究了利用ChatGPT进行立场检测中，无参数的思维链（CoT）方法的有效性，证明其表现优越，并探讨了相关挑战。 |

# 详细

[^1]: 使用大型语言模型进行K级推理

    K-Level Reasoning with Large Language Models

    [https://rss.arxiv.org/abs/2402.01521](https://rss.arxiv.org/abs/2402.01521)

    该论文探索了使用大型语言模型进行动态决策推理的能力，并提出了一种名为"K级推理"的新颖推理方法。通过博弈论的试验，发现现有的推理方法在动态环境中容易出错，而"K级推理"可以解决这个问题。

    

    虽然大型语言模型（LLMs）已经展示了其在复杂推理任务上的能力，但在动态、交互和竞争场景（如商业战略和股票市场分析）中的性能仍然未被充分探索。为了填补这个空白，我们正式探索LLMs在快速变化环境中的决策推理能力。我们引入了两个基于博弈论的试验，以模拟现实世界中动态决策的复杂性。这些挑战具有明确定义，可以对LLMs的动态推理能力进行清晰、可控和精确的评估。通过大量实验，我们发现现有的推理方法在需要k级思考的动态环境中容易出错 - 这是之前研究中未解决的关键概念。为了解决这个问题，我们提出了一种新颖的LLMs推理方法，命名为“K级推理”。该方法采用对手的视角，从递归角度运用基于k级思考的推理。

    While Large Language Models (LLMs) have demonstrated their proficiency in complex reasoning tasks, their performance in dynamic, interactive, and competitive scenarios - such as business strategy and stock market analysis - remains underexplored. To bridge this gap, we formally explore the dynamic reasoning capabilities of LLMs for decision-making in rapidly evolving environments. We introduce two game theory-based pilot challenges that mirror the complexities of real-world dynamic decision-making. These challenges are well-defined, enabling clear, controllable, and precise evaluation of LLMs' dynamic reasoning abilities. Through extensive experiments, we find that existing reasoning methods tend to falter in dynamic settings that require k-level thinking - a key concept not tackled by previous works. To address this, we propose a novel reasoning approach for LLMs, named "K-Level Reasoning". This approach adopts the perspective of rivals to recursively employ k-level thinking based on 
    
[^2]: PROMPT-SAW：利用关系感知图进行文本提示压缩

    PROMPT-SAW: Leveraging Relation-Aware Graphs for Textual Prompt Compression

    [https://arxiv.org/abs/2404.00489](https://arxiv.org/abs/2404.00489)

    提出了PROMPT-SAW模型，利用关系感知图来实现文本提示的压缩，提高了提示的可读性和可解释性。

    

    大型语言模型(LLMs)在多种不同的自然语言处理任务中展现出卓越的能力。提示是LLM推理中的基本工具，但我们观察到超长提示会带来显著的成本。现有的压缩长提示的尝试导致压缩提示在可读性和可解释性方面表现不佳，对提示效用产生有害影响。为了解决这一问题，我们提出了PROMPT-SAW：通过关系感知图进行提示压缩，这是一种针对任务不可知和任务感知提示的有效策略。PROMPT-SAW使用提示的文本信息构建图形，在图形中提取关键信息元素，从而得出压缩提示。我们还提出了GSM8K-AUG，即现有GSM8k基准的扩展版本，用于任务不可知提示，以提供全面的评估平台。

    arXiv:2404.00489v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown exceptional abilities for multiple different natural language processing tasks. While prompting is a crucial tool for LLM inference, we observe that there is a significant cost associated with exceedingly lengthy prompts. Existing attempts to compress lengthy prompts lead to sub-standard results in terms of readability and interpretability of the compressed prompt, with a detrimental impact on prompt utility. To address this, we propose PROMPT-SAW: Prompt compresSion via Relation AWare graphs, an effective strategy for prompt compression over task-agnostic and task-aware prompts. PROMPT-SAW uses the prompt's textual information to build a graph, later extracts key information elements in the graph to come up with the compressed prompt. We also propose GSM8K-AUG, i.e., an extended version of the existing GSM8k benchmark for task-agnostic prompts in order to provide a comprehensive evaluation platf
    
[^3]: CoLLEGe: 大型语言模型的概念嵌入生成

    CoLLEGe: Concept Embedding Generation for Large Language Models

    [https://arxiv.org/abs/2403.15362](https://arxiv.org/abs/2403.15362)

    CoLLEGe是一个元学习框架，能够为大型语言模型生成灵活的新概念嵌入，用于现代化少样本概念学习。

    

    当前语言模型无法快速学习新概念，通常需要更复杂的微调过程才能学习得更稳健。本文引入了一种名为CoLLEGe（Concept Learning with Language Embedding Generation）的新方法，用于现代化的少样本概念学习。CoLLEGe是一个元学习框架，能够使用少量示例句子或定义生成新概念的灵活嵌入。我们的主要元学习目标只是促进语言模型在随后的句子中进行下一个词预测，使其与语言模型的预训练兼容。

    arXiv:2403.15362v1 Announce Type: cross  Abstract: Current language models are unable to quickly learn new concepts on the fly, often requiring a more involved finetuning process to learn robustly. Prompting in-context is not robust to context distractions, and often fails to confer much information about the new concepts. Classic methods for few-shot word learning in NLP, relying on global word vectors, are less applicable to large language models. In this paper, we introduce a novel approach named CoLLEGe (Concept Learning with Language Embedding Generation) to modernize few-shot concept learning. CoLLEGe is a meta-learning framework capable of generating flexible embeddings for new concepts using a small number of example sentences or definitions. Our primary meta-learning objective is simply to facilitate a language model to make next word predictions in forthcoming sentences, making it compatible with language model pretraining. We design a series of tasks to test new concept lear
    
[^4]: 对韩语大型语言模型的语用能力评估

    Pragmatic Competence Evaluation of Large Language Models for Korean

    [https://arxiv.org/abs/2403.12675](https://arxiv.org/abs/2403.12675)

    该研究将大型语言模型的评估从对嵌入知识的基准拓展到了探索语用能力，结果显示在韩语环境下，GPT-4在传统和人工评估设置中表现出色，而HyperCLOVA X也在开放式问题评估中取得了不错的成绩。

    

    目前对大型语言模型（LLMs）的评估主要依赖于着重于测试其嵌入知识的基准，通过多项选择题（MCQs）来进行评估，这种格式非常适合自动评估。我们的研究将此评估拓展到探索LLM的语用能力--在先进的LLM出现之前鲜有研究，特别是在韩语环境下。我们采用两种不同的评估设置：传统的自动评估适配的MCQ格式，以及由人类专家评估的开放式问题（OEQs），用以检查LLM的叙事回应能力，而无需预先定义选项。我们的研究发现，GPT-4表现优异，在MCQ和OEQ设置中得分分别为81.11和85.69，而以韩语为优化目标的HyperCLOVA X在OEQ设置中表现出色，得分为81.56，与GPT-4相比，仅有4.13分的微小差距。

    arXiv:2403.12675v1 Announce Type: new  Abstract: The current evaluation of Large Language Models (LLMs) predominantly relies on benchmarks focusing on their embedded knowledge by testing through multiple-choice questions (MCQs), a format inherently suited for automated evaluation. Our study extends this evaluation to explore LLMs' pragmatic competence--a facet previously underexamined before the advent of sophisticated LLMs, specifically in the context of Korean. We employ two distinct evaluation setups: the conventional MCQ format, adapted for automatic evaluation, and Open-Ended Questions (OEQs), assessed by human experts, to examine LLMs' narrative response capabilities without predefined options. Our findings reveal that GPT-4 excels, scoring 81.11 and 85.69 in the MCQ and OEQ setups, respectively, with HyperCLOVA X, an LLM optimized for Korean, closely following, especially in the OEQ setup, demonstrating a score of 81.56 with a marginal difference of 4.13 points compared to GPT-4
    
[^5]: ActiveRAG: 通过主动学习揭示知识的宝藏

    ActiveRAG: Revealing the Treasures of Knowledge via Active Learning

    [https://arxiv.org/abs/2402.13547](https://arxiv.org/abs/2402.13547)

    ActiveRAG是一个创新的RAG框架，通过引入主动学习机制，利用知识构建和认知联结机制来提升大型语言模型（LLMs）的内在认知，实现了明显的性能提升。

    

    arXiv:2402.13547v1 公告类型：新摘要：检索增强生成（RAG）引入了一种新的大型语言模型（LLM）范例，有助于解决知识密集型任务。然而，当前的RAG模型将LLMs定位为被动的知识接收器，从而限制了它们学习和理解外部知识的能力。本文提出了ActiveRAG，它是一种创新的RAG框架，从被动知识获取转变为主动学习机制。这种方法利用知识构建机制通过将外部知识与先前获取或记忆的知识相关联来更深入地理解外部知识。随后，它设计了认知联结机制以合并来自思维和知识构建链的成果，从而校准LLMs的内在认知。我们的实验结果表明，ActiveRAG超越了先前的RAG模型，在问题回答上实现了5%的改进。

    arXiv:2402.13547v1 Announce Type: new  Abstract: Retrieval Augmented Generation (RAG) has introduced a new paradigm for Large Language Models (LLMs), aiding in the resolution of knowledge-intensive tasks. However, current RAG models position LLMs as passive knowledge receptors, thereby restricting their capacity for learning and comprehending external knowledge. In this paper, we present ActiveRAG, an innovative RAG framework that shifts from passive knowledge acquisition to an active learning mechanism. This approach utilizes the Knowledge Construction mechanism to develop a deeper understanding of external knowledge by associating it with previously acquired or memorized knowledge. Subsequently, it designs the Cognitive Nexus mechanism to incorporate the outcomes from both chains of thought and knowledge construction, thereby calibrating the intrinsic cognition of LLMs. Our experimental results demonstrate that ActiveRAG surpasses previous RAG models, achieving a 5% improvement on qu
    
[^6]: 表格作为图片？探讨LLM在多模态表格数据表示上的优势和局限性

    Tables as Images? Exploring the Strengths and Limitations of LLMs on Multimodal Representations of Tabular Data

    [https://arxiv.org/abs/2402.12424](https://arxiv.org/abs/2402.12424)

    本研究探讨了LLM在解释表格数据方面的有效性，比较了文本和图像表格表示对LLM性能的影响，为在表格相关任务上有效使用LLM提供了见解。

    

    在本文中，我们通过不同的提示策略和数据格式研究了各种LLM在解释表格数据方面的有效性。我们的分析涵盖了六个针对与表格相关任务的基准，如问答和事实核查。我们首次介绍了LLM在基于图像的表格表示上的表现评估。具体地，我们比较了五种基于文本和三种基于图像的表格表示，展示了表示和提示对LLM性能的影响。我们的研究为在表格相关任务上有效使用LLM提供了见解。

    arXiv:2402.12424v1 Announce Type: cross  Abstract: In this paper, we investigate the effectiveness of various LLMs in interpreting tabular data through different prompting strategies and data formats. Our analysis extends across six benchmarks for table-related tasks such as question-answering and fact-checking. We introduce for the first time the assessment of LLMs' performance on image-based table representations. Specifically, we compare five text-based and three image-based table representations, demonstrating the influence of representation and prompting on LLM performance. Our study provides insights into the effective use of LLMs on table-related tasks.
    
[^7]: MORL-Prompt: 离散提示优化的多目标强化学习的实证分析

    MORL-Prompt: An Empirical Analysis of Multi-Objective Reinforcement Learning for Discrete Prompt Optimization

    [https://arxiv.org/abs/2402.11711](https://arxiv.org/abs/2402.11711)

    本研究将多目标优化技术应用于基于强化学习的离散提示优化，为解决奖励平衡问题提供了新视角。

    

    基于RL的技术可以用于搜索提示，将其输入目标语言模型以最大化一组用户指定的奖励函数。然而，在许多目标应用中，自然奖励函数彼此之间存在紧张关系--例如，在风格转移任务中，内容保留与风格匹配之间存在矛盾。当前技术侧重于最大化奖励函数的平均值，这未必会导致取得各种奖励平衡的提示--这个问题在多目标和鲁棒优化文献中得到了深入研究。本文将几种多目标优化技术调整为基于RL的离散提示优化--其中有两种考虑帕累托奖励面积的方法，另外一种选择有益于所有奖励的更新方向。我们在两个NLP任务上对这些方法进行了实证分析：风格转移和机器翻译。

    arXiv:2402.11711v1 Announce Type: new  Abstract: RL-based techniques can be used to search for prompts that when fed into a target language model maximize a set of user-specified reward functions. However, in many target applications, the natural reward functions are in tension with one another -- for example, content preservation vs. style matching in style transfer tasks. Current techniques focus on maximizing the average of reward functions, which does not necessarily lead to prompts that achieve balance across rewards -- an issue that has been well-studied in the multi-objective and robust optimization literature. In this paper, we adapt several techniques for multi-objective optimization to RL-based discrete prompt optimization -- two that consider volume of the Pareto reward surface, and another that chooses an update direction that benefits all rewards simultaneously. We conduct an empirical analysis of these methods on two NLP tasks: style transfer and machine translation, each
    
[^8]: BLT: 大型语言模型能处理基础法律文本吗？

    BLT: Can Large Language Models Handle Basic Legal Text?

    [https://arxiv.org/abs/2311.09693](https://arxiv.org/abs/2311.09693)

    大型语言模型在处理基础法律文本方面表现不佳，但通过针对性微调，甚至较小的模型也能在测试中表现出色，提升了相关法律任务的表现。

    

    我们发现像GPT-4、Claude和{PaLM 2}这样的最好的公开可用的LLM在处理基础法律文本方面表现不佳。我们引入了一个基准，其中包含律师和法律助理期望LLM零-shot处理的任务，比如查找证词文件的某一行或合同的某个子部分的文本。LLM在这个基准上的差劲表现对它们在法律实践中的可靠性提出了质疑。然而，针对这些任务进行微调甚至使一个较小的模型在我们的测试集上表现接近完美，并且还提升了相关法律任务的表现。这些结果表明，许多领域所需的简单行为在基础LLM中可能不存在，除非有领域专家的额外参与。

    arXiv:2311.09693v2 Announce Type: replace-cross  Abstract: We find that the best publicly available LLMs like GPT-4, Claude, and {PaLM 2} currently perform poorly at basic legal text handling. We introduce a benchmark consisting of tasks that lawyers and paralegals would expect LLMs to handle zero-shot, such as looking up the text at a line of a witness deposition or at a subsection of a contract. LLMs' poor performance on this benchmark casts into doubt their reliability as-is for legal practice. However, fine-tuning for these tasks brings even a smaller model to near-perfect performance on our test set and also raises performance on a related legal task. These results suggest that many simple behaviors needed for a domain may not be present in foundational LLMs, without additional engagement from subject matter experts.
    
[^9]: 实验环境能够促进语言模型在稳健的语义属性推断中的表现，但不一致。

    Experimental Contexts Can Facilitate Robust Semantic Property Inference in Language Models, but Inconsistently. (arXiv:2401.06640v1 [cs.CL])

    [http://arxiv.org/abs/2401.06640](http://arxiv.org/abs/2401.06640)

    本研究通过控制实验环境的方式，发现语言模型在属性继承任务中表现出了一定的非平凡能力，但这种能力是不一致的。

    

    最近的无人监督评估凸显了语言模型（LMs）在执行意义提取方面的重要限制。然而，众所周知，在引入实验环境（如上下文示例和指导）的情况下，LMs的表现可以显著提高。那么这是否适用于先前研究的意义敏感任务呢？我们在控制上下文示例和指导内容的前提下，对实验环境对于提高LMs在执行属性继承任务中的鲁棒性的程度进行了案例研究，该任务是预先表明LMs无法完成的任务。我们的研究发现，实验环境确实可以导致LMs在属性继承行为方面表现出非平凡的能力。然而，这种能力是不一致的：通过对任务进行最小改写，发现一些LMs从输入中捕捉到浅层的非语义式启发式信息，这表明计算机的行为具有不一致性。

    Recent zero-shot evaluations have highlighted important limitations in the abilities of language models (LMs) to perform meaning extraction. However, it is now well known that LMs can demonstrate radical improvements in the presence of experimental contexts such as in-context examples and instructions. How well does this translate to previously studied meaning-sensitive tasks? We present a case-study on the extent to which experimental contexts can improve LMs' robustness in performing property inheritance -- predicting semantic properties of novel concepts, a task that they have been previously shown to fail on. Upon carefully controlling the nature of the in-context examples and the instructions, our work reveals that they can indeed lead to non-trivial property inheritance behavior in LMs. However, this ability is inconsistent: with a minimal reformulation of the task, some LMs were found to pick up on shallow, non-semantic heuristics from their inputs, suggesting that the computati
    
[^10]: 利用ChatGPT探究思维链在社交媒体中的立场检测

    Investigating Chain-of-thought with ChatGPT for Stance Detection on Social Media. (arXiv:2304.03087v1 [cs.CL])

    [http://arxiv.org/abs/2304.03087](http://arxiv.org/abs/2304.03087)

    本文研究了利用ChatGPT进行立场检测中，无参数的思维链（CoT）方法的有效性，证明其表现优越，并探讨了相关挑战。

    

    立场检测是预测文本中针对目标的态度，随着社交媒体的兴起已受到关注。传统方法包括传统机器学习、早期深度神经网络和预训练微调模型。然而，随着非常大的预训练语言模型（VLPLMs）如ChatGPT（GPT-3.5）的发展，传统方法面临部署挑战。不需要反向传播训练的无参数思维链（CoT）方法已成为一种有希望的替代方法。本文研究了CoT在立场检测任务中的有效性，展示了其优越的精度并讨论了相关的挑战。

    Stance detection predicts attitudes towards targets in texts and has gained attention with the rise of social media. Traditional approaches include conventional machine learning, early deep neural networks, and pre-trained fine-tuning models. However, with the evolution of very large pre-trained language models (VLPLMs) like ChatGPT (GPT-3.5), traditional methods face deployment challenges. The parameter-free Chain-of-Thought (CoT) approach, not requiring backpropagation training, has emerged as a promising alternative. This paper examines CoT's effectiveness in stance detection tasks, demonstrating its superior accuracy and discussing associated challenges.
    

