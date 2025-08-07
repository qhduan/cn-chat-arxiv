# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Strong Priority and Determinacy in Timed CCS](https://arxiv.org/abs/2403.04618) | 引入了一种新的调度机制“顺序构造减少”，旨在实现多播并发通信的确定性，扩展了CCS的技术设置，证明了构造减少的汇聚属性，展示了在一些语法限制下运算符的结构连贯性。 |

# 详细

[^1]: 时标CCS中的强优先级和确定性

    Strong Priority and Determinacy in Timed CCS

    [https://arxiv.org/abs/2403.04618](https://arxiv.org/abs/2403.04618)

    引入了一种新的调度机制“顺序构造减少”，旨在实现多播并发通信的确定性，扩展了CCS的技术设置，证明了构造减少的汇聚属性，展示了在一些语法限制下运算符的结构连贯性。

    

    在具有优先级的经典进程代数理论的基础上，我们确定了一种名为“顺序构造减少”的新调度机制，旨在捕捉同步编程的本质。这种评估策略的独特属性是通过构造实现多播并发通信的确定性。特别是，这使我们能够模拟具有对缺失反应的共享内存多线程，因为它是Esterel编程语言的核心。在通过时钟和优先级扩展的CCS的技术设置中，对于我们称为“结构连贯”的大类过程，我们证明了构造减少的汇聚属性。我们进一步展示，在一些称为“可枢纽”的语法限制下，前缀、求和、并行组成、限制和隐藏的运算符保持结构连贯。这涵盖了一个严格更大的过程类。

    arXiv:2403.04618v1 Announce Type: cross  Abstract: Building on the classical theory of process algebra with priorities, we identify a new scheduling mechanism, called "sequentially constructive reduction" which is designed to capture the essence of synchronous programming. The distinctive property of this evaluation strategy is to achieve determinism-by-construction for multi-cast concurrent communication. In particular, it permits us to model shared memory multi-threading with reaction to absence as it lies at the core of the programming language Esterel. In the technical setting of CCS extended by clocks and priorities, we prove for a large class of processes, which we call "structurally coherent" the confluence property for constructive reductions. We further show that under some syntactic restrictions, called "pivotable" the operators of prefix, summation, parallel composition, restriction and hiding preserve structural coherence. This covers a strictly larger class of processes co
    

