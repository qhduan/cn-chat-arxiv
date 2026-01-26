# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fine-Tuning, Prompting, In-Context Learning and Instruction-Tuning: How Many Labelled Samples Do We Need?](https://arxiv.org/abs/2402.12819) | 专门模型通常只需少量标记样本（100-1000个）就能与通用模型持平甚至更好，取决于任务的复杂性和结果的变化。 |

# 详细

[^1]: 微调、提示、上下文学习和指导微调：我们需要多少标记样本？

    Fine-Tuning, Prompting, In-Context Learning and Instruction-Tuning: How Many Labelled Samples Do We Need?

    [https://arxiv.org/abs/2402.12819](https://arxiv.org/abs/2402.12819)

    专门模型通常只需少量标记样本（100-1000个）就能与通用模型持平甚至更好，取决于任务的复杂性和结果的变化。

    

    当解决具有有限标记数据的任务时，研究人员可以选择使用通用的大型语言模型而不进行进一步更新，或者使用少量示例来调整专门的较小模型。 当有足够的标记可用时，专门的模型在许多自然语言处理任务上表现优于通用模型。 在这项工作中，我们旨在调查专门模型需要多少标记样本才能实现这种出色的性能，同时考虑结果的变化。观察提示、上下文学习、微调和指导微调的行为，识别它们在增加不同复杂性任务的标记训练样本数量时的收支平衡点，我们发现专门模型通常只需少量样本（100-1000个）就能与通用模型持平甚至更好。 同时，所需的标记数据量强烈依赖于任务的复杂性和结果的变化。

    arXiv:2402.12819v1 Announce Type: cross  Abstract: When solving a task with limited labelled data, researchers can either use a general large language model without further update, or use the few examples to tune a specialised smaller model. When enough labels are available, the specialised models outperform the general ones on many NLP tasks. In this work, we aim to investigate how many labelled samples are required for the specialised models to achieve this superior performance, while taking the results variance into consideration. Observing the behaviour of prompting, in-context learning, fine-tuning and instruction-tuning, identifying their break-even points when increasing number of labelled training samples across three tasks of varying complexity, we find that the specialised models often need only few samples ($100-1000$) to be on par or better than the general ones. At the same time, the amount of required labelled data strongly depends on the task complexity and results varia
    

