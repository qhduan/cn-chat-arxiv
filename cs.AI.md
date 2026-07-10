# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?](https://arxiv.org/abs/2606.26428) | 本文提出Play2Perfect框架，通过任务无关的玩耍式预训练获取可重用的操作先验知识，并证明玩耍过程中的物体多样性、目标多样性和探索策略是提升精密装配性能的关键因素。 |
| [^2] | [Narration-of-Thought: Inference-Time Scaffolding for Defeasible Ethical Reasoning in Large Language Models](https://arxiv.org/abs/2606.26366) | 本文提出一种名为“思维叙述”的系统提示方法，通过将思维链结构化为五个特定部分，显著减少了大型语言模型在伦理推理中忽视利益相关者和压制不确定性的错误，无需额外训练或微调。 |

# 详细

[^1]: 完美游戏：灵巧操作预训练中什么因素对精密装配至关重要？

    Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?

    [https://arxiv.org/abs/2606.26428](https://arxiv.org/abs/2606.26428)

    本文提出Play2Perfect框架，通过任务无关的玩耍式预训练获取可重用的操作先验知识，并证明玩耍过程中的物体多样性、目标多样性和探索策略是提升精密装配性能的关键因素。

    

    arXiv:2606.26428v1 公告类型：交叉 摘要：多指机器人有望实现人类手部的速度和灵巧性，但精密装配等挑战性问题仍然难以解决。这些任务接触密集，使得模仿学习的数据收集变得困难，且奖励稀疏，使得直接使用强化学习进行探索变得棘手。因此，先前的研究通过使用专用夹爪、工具附件和环境固定装置来结构化问题，取得了进展。在这项工作中，我们认为，在机器人能够完善精密装配之前，它必须首先学会“玩耍”。我们进一步提出问题：在玩耍学习的过程中，哪些因素对精密装配至关重要？我们提出了Play2Perfect，这是一个任务无关的强化学习框架，通过对多样化的物体和目标进行玩耍式预训练，然后在精密装配任务上进行精炼。玩耍的目标是获取可重用的操作先验知识，例如抓取、手中重新定向和姿态到达。

    arXiv:2606.26428v1 Announce Type: cross  Abstract: Multi-fingered robots promise the speed and dexterity of human hands, yet challenging problems such as precise assembly have remained out of reach. These tasks are contact-rich, making data collection for imitation learning difficult, and sparse-reward, making direct exploration with reinforcement learning (RL) intractable. Consequently, prior work has made progress by structuring the problem with specialized grippers, tool attachments, and environment fixtures. In this work, we argue that before a robot can perfect precise assembly, it must first learn to play. We further ask the question: what factors in the process of learning to play matter for precise assembly? We propose Play2Perfect, an RL framework for task-agnostic pretraining through play on diverse objects and goals, which is then perfected on precise assembly. The goal of play is to acquire reusable manipulation priors, such as grasping, in-hand reorientation and pose reach
    
[^2]: 思维叙述：大型语言模型中可废止伦理推理的推理时脚手架

    Narration-of-Thought: Inference-Time Scaffolding for Defeasible Ethical Reasoning in Large Language Models

    [https://arxiv.org/abs/2606.26366](https://arxiv.org/abs/2606.26366)

    本文提出一种名为“思维叙述”的系统提示方法，通过将思维链结构化为五个特定部分，显著减少了大型语言模型在伦理推理中忽视利益相关者和压制不确定性的错误，无需额外训练或微调。

    

    arXiv:2606.26366v1 公告类型：新 摘要：针对道德困境的标准思维链存在两种失败模式：利益相关者崩溃（思维链中最多只提及一个与结果相关的当事方）和不确定性抑制（在做出行动承诺前，没有明确提及未知或保留意见）。我们引入了思维叙述（NoT），这是一种系统提示，将思维链结构化为五个部分：主角、利益相关者、两步后果、不确定性、然后承诺。NoT无需额外训练、参数或微调。在来自三家供应商的四个生成器上的100个每日困境场景中，NoT将每个模型上的利益相关者崩溃率从高达31%降至1%以下，将不确定性抑制率从高达72%降至1-24%。一个匹配预算的详细思维链控制实验排除了令牌消耗作为有效成分；NoT在四个生成器中的三个上，在利益相关者数量上保持了+0.79至+0.90的Cliff's delta优势，在不确定性得分上保持了+0.65至+0.93的优势，并且一个部分消融实验归因了这些改进。

    arXiv:2606.26366v1 Announce Type: new  Abstract: Standard chain-of-thought on moral dilemmas exhibits two failure modes: stakeholder collapse (the trace names at most one party with a stake in the outcome) and uncertainty suppression (no explicit unknowns or hedges before committing to an action). We introduce narration-of-thought (NoT), a system prompt that structures chain-of-thought into five sections: protagonist, stakeholders, two-step consequences, uncertainty, then commitment. NoT adds no training, parameters, or fine-tuning. On 100 DailyDilemmas scenarios across four generators from three vendors, NoT cuts stakeholder collapse from up to 31% to under 1% and uncertainty suppression from up to 72% to 1-24% on every model. A matched-budget verbose-CoT control rules out token spend as the active ingredient; NoT retains Cliff's delta advantages of +0.79 to +0.90 on stakeholder count and +0.65 to +0.93 on uncertainty score for three of four generators, and a section ablation attribut
    

