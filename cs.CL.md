# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The pitfalls of next-token prediction](https://arxiv.org/abs/2403.06963) | 论文揭示了在某些任务类别中，教师强制方法可能无法在第一时间学习到准确的下一个标记预测器，进而导致模型失败的一般机制。 |

# 详细

[^1]: 下一个标记预测的陷阱

    The pitfalls of next-token prediction

    [https://arxiv.org/abs/2403.06963](https://arxiv.org/abs/2403.06963)

    论文揭示了在某些任务类别中，教师强制方法可能无法在第一时间学习到准确的下一个标记预测器，进而导致模型失败的一般机制。

    

    一篇关于下一个标记预测的论文。我们提出了一个直观的担忧：一个仅仅基于下一个标记预测的模型是否能忠实地模拟人类智能。我们认为下一个标记预测中经常混淆的两个阶段 -- 自回归推断和教师强制训练 -- 必须被区别对待。我们描述了一个一般机制，展示了教师强制如何失败，并设计了一个最小化计划任务，在这个任务中Transformer和Mamba架构在实践中以这种方式失败 -- 尽管任务本身很容易学习。

    arXiv:2403.06963v1 Announce Type: cross  Abstract: Can a mere next-token predictor faithfully model human intelligence? We crystallize this intuitive concern, which is fragmented in the literature. As a starting point, we argue that the two often-conflated phases of next-token prediction -- autoregressive inference and teacher-forced training -- must be treated distinctly. The popular criticism that errors can compound during autoregressive inference, crucially assumes that teacher-forcing has learned an accurate next-token predictor. This assumption sidesteps a more deep-rooted problem we expose: in certain classes of tasks, teacher-forcing can simply fail to learn an accurate next-token predictor in the first place. We describe a general mechanism of how teacher-forcing can fail, and design a minimal planning task where both the Transformer and the Mamba architecture empirically fail in that manner -- remarkably, despite the task being straightforward to learn. We provide preliminary
    

