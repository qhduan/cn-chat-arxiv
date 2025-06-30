# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought](https://arxiv.org/abs/2403.05518) | 引入偏差增强的一致性训练（BCT）可以显著减少链式思维中的偏见推理问题，尤其是通过训练模型在带有和不带有偏置特征的提示下进行一致的推理。 |
| [^2] | [Programming Distributed Collective Processes in the eXchange Calculus.](http://arxiv.org/abs/2401.11212) | 本研究在交换演算中考虑了集合设备的动态合作行为，提出了分布式集体过程的抽象表示，用于编程计算集体的行为。 |
| [^3] | [On CNF formulas irredundant with respect to unit clause propagation.](http://arxiv.org/abs/2309.01750) | 对于单子句传播而言，在CNF公式中不可简化的公式，其大小与最小可等价的公式大小的比值最大为n^2，其中n是变量数量。一般上界不会小于n/ln n倍。 |

# 详细

[^1]: 通过偏差增强一致性训练减少链式思维中的偏见推理

    Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought

    [https://arxiv.org/abs/2403.05518](https://arxiv.org/abs/2403.05518)

    引入偏差增强的一致性训练（BCT）可以显著减少链式思维中的偏见推理问题，尤其是通过训练模型在带有和不带有偏置特征的提示下进行一致的推理。

    

    虽然链式思维提示（CoT）有潜力改善语言模型推理的可解释性，但它可能会系统性地歪曲影响模型行为的因素--比如，合理化答案以符合用户意见而不提及此偏见。为了减轻这一偏见推理问题，我们引入了偏差增强的一致性训练（BCT），这是一种无监督的微调方案，旨在训练模型在带有和不带有偏置特征的提示下进行一致的推理。我们构建了一个测试单元，针对七个问答任务测试了九种形式的有偏推理，发现将BCT应用于带有一种偏见的GPT-3.5-Turbo可以将有偏推理的比例在未知任务上降低86%。此外，这个模型推广到其他形式的偏见，平均将未知偏见上的有偏推理减少了37%。由于BCT将未知偏见泛化并且不需要金标签，这种方法可能会有助于

    arXiv:2403.05518v1 Announce Type: cross  Abstract: While chain-of-thought prompting (CoT) has the potential to improve the explainability of language model reasoning, it can systematically misrepresent the factors influencing models' behavior--for example, rationalizing answers in line with a user's opinion without mentioning this bias. To mitigate this biased reasoning problem, we introduce bias-augmented consistency training (BCT), an unsupervised fine-tuning scheme that trains models to give consistent reasoning across prompts with and without biasing features. We construct a suite testing nine forms of biased reasoning on seven question-answering tasks, and find that applying BCT to GPT-3.5-Turbo with one bias reduces the rate of biased reasoning by 86% on held-out tasks. Moreover, this model generalizes to other forms of bias, reducing biased reasoning on held-out biases by an average of 37%. As BCT generalizes to held-out biases and does not require gold labels, this method may h
    
[^2]: 在交换演算中编程分布式集体过程

    Programming Distributed Collective Processes in the eXchange Calculus. (arXiv:2401.11212v1 [cs.DC])

    [http://arxiv.org/abs/2401.11212](http://arxiv.org/abs/2401.11212)

    本研究在交换演算中考虑了集合设备的动态合作行为，提出了分布式集体过程的抽象表示，用于编程计算集体的行为。

    

    最近的趋势如物联网（IoT）提出了在几乎所有环境中密集和多尺度部署计算设备的愿景。一个突出的工程挑战围绕着编程这种计算生态系统的集体自适应行为。这需要能够捕捉概念（动态合作设备群组）和集体任务（由合奏组执行的联合活动）的抽象。在这项工作中，我们考虑与邻居交互并以几乎同步的感知-计算-交互循环执行的设备集合，其中计算由一个将感知值和传入消息映射到输出和传出消息的单个程序给出。为了支持整个计算集体的编程，我们提出了分布式集体过程的抽象，它可以同时定义合奏组的形成逻辑和它的集体任务。我们在交换演算中形式化了这种抽象。

    Recent trends like the Internet of Things (IoT) suggest a vision of dense and multi-scale deployments of computing devices in nearly all kinds of environments. A prominent engineering challenge revolves around programming the collective adaptive behaviour of such computational ecosystems. This requires abstractions able to capture concepts like ensembles (dynamic groups of cooperating devices) and collective tasks (joint activities carried out by ensembles). In this work, we consider collections of devices interacting with neighbours and that execute in nearly-synchronised sense-compute-interact rounds, where the computation is given by a single program mapping sensing values and incoming messages to output and outcoming messages. To support programming whole computational collectives, we propose the abstraction of a distributed collective process, which can be used to define at once the ensemble formation logic and its collective task. We formalise the abstraction in the eXchange Calc
    
[^3]: 关于相对于单子句传播不可简化的CNF公式

    On CNF formulas irredundant with respect to unit clause propagation. (arXiv:2309.01750v2 [math.CO] UPDATED)

    [http://arxiv.org/abs/2309.01750](http://arxiv.org/abs/2309.01750)

    对于单子句传播而言，在CNF公式中不可简化的公式，其大小与最小可等价的公式大小的比值最大为n^2，其中n是变量数量。一般上界不会小于n/ln n倍。

    

    如果两个CNF公式在单子句传播（UCP）方面的行为相同，则它们被称为ucp等价。如果移除任意一个子句会导致一个与原始公式在ucp方面不等价的公式，则称该公式为ucp不可简化。根据已知结果，ucp不可简化公式的大小与最小ucp等价公式的大小的比值最大为n^2，其中n是变量的数量。我们展示了对称确定Horn函数的一个ucp不可简化公式的例子，其大小比最小的ucp等价公式大n/ln n倍，因此，上述比值的一般上界不能小于这个值。

    Two CNF formulas are called ucp-equivalent, if they behave in the same way with respect to the unit clause propagation (UCP). A formula is called ucp-irredundant, if removing any clause leads to a formula which is not ucp-equivalent to the original one. As a consequence of known results, the ratio of the size of a ucp-irredundant formula and the size of a smallest ucp-equivalent formula is at most $n^2$, where $n$ is the number of the variables. We demonstrate an example of a ucp-irredundant formula for a symmetric definite Horn function which is larger than a smallest ucp-equivalent formula by a factor $\Omega(n/\ln n)$ and, hence, a general upper bound on the above ratio cannot be smaller than this.
    

