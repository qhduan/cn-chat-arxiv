# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conformal Temporal Logic Planning using Large Language Models: Knowing When to Do What and When to Ask for Help.](http://arxiv.org/abs/2309.10092) | 本文提出了一个使用大型语言模型的一致时间逻辑规划方法，用于解决多个高级子任务的移动机器人运动规划问题。其中的一个关键挑战是如何以正确性的角度推理机器人计划与基于自然语言的逻辑任务的关系。 |

# 详细

[^1]: 使用大型语言模型的一致时间逻辑规划：知道何时做什么和何时寻求帮助。

    Conformal Temporal Logic Planning using Large Language Models: Knowing When to Do What and When to Ask for Help. (arXiv:2309.10092v1 [cs.RO])

    [http://arxiv.org/abs/2309.10092](http://arxiv.org/abs/2309.10092)

    本文提出了一个使用大型语言模型的一致时间逻辑规划方法，用于解决多个高级子任务的移动机器人运动规划问题。其中的一个关键挑战是如何以正确性的角度推理机器人计划与基于自然语言的逻辑任务的关系。

    

    本文解决了一个新的移动机器人运动规划问题，任务是以自然语言（NL）表达并以时间和逻辑顺序完成多个高级子任务。为了正式定义这样的任务，我们利用基于NL的原子谓词在LTL上定义了模型。这与相关的规划方法形成对比，这些方法在原子谓词上定义了捕捉所需低级系统配置的LTL任务。我们的目标是设计机器人计划，满足基于NL的原子命题定义的LTL任务。在这个设置中出现的一个新的技术挑战在于推理机器人计划的正确性与这些LTL编码的任务的关系。为了解决这个问题，我们提出了HERACLEs，一个分层一致的自然语言规划器，它依赖于现有工具的新型整合，包括（i）自动机理论，以确定机器人应该完成的NL指定的子任务以推进任务进展；

    This paper addresses a new motion planning problem for mobile robots tasked with accomplishing multiple high-level sub-tasks, expressed using natural language (NL), in a temporal and logical order. To formally define such missions, we leverage LTL defined over NL-based atomic predicates modeling the considered NL-based sub-tasks. This is contrast to related planning approaches that define LTL tasks over atomic predicates capturing desired low-level system configurations. Our goal is to design robot plans that satisfy LTL tasks defined over NL-based atomic propositions. A novel technical challenge arising in this setup lies in reasoning about correctness of a robot plan with respect to such LTL-encoded tasks. To address this problem, we propose HERACLEs, a hierarchical conformal natural language planner, that relies on a novel integration of existing tools that include (i) automata theory to determine the NL-specified sub-task the robot should accomplish next to make mission progress; (
    

