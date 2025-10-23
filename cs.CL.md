# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LV-Eval: A Balanced Long-Context Benchmark with 5 Length Levels Up to 256K](https://arxiv.org/abs/2402.05136) | LV-Eval是一个具有五个长度级别的长上下文基准测试，支持256k上下文长度，并具有混淆事实、关键词和短语替换以及基于关键词回忆的度量设计等关键技术，旨在减少知识泄漏和提供更客观的评估。 |

# 详细

[^1]: LV-Eval:一个平衡的长上下文基准测试，具有5个长度级别，最多可达256K

    LV-Eval: A Balanced Long-Context Benchmark with 5 Length Levels Up to 256K

    [https://arxiv.org/abs/2402.05136](https://arxiv.org/abs/2402.05136)

    LV-Eval是一个具有五个长度级别的长上下文基准测试，支持256k上下文长度，并具有混淆事实、关键词和短语替换以及基于关键词回忆的度量设计等关键技术，旨在减少知识泄漏和提供更客观的评估。

    

    最先进的大型语言模型（LLMs）现在声称支持的上下文长度可以达到256k甚至更多。相比之下，主流基准测试的平均上下文长度不足（5k-21k），并且它们容易出现知识泄漏和不准确的评估指标，导致评估结果偏见。本文介绍了LV-Eval，一个具有五个长度级别（16k，32k，64k，128k和256k）的具有挑战性的长上下文基准测试，最多可达256k个单词。LV-Eval包含两个主要任务，单跳问答和多跳问答，包含11个双语数据集。LV-Eval的设计融合了三个关键技术，即混淆事实插入、关键词和短语替换以及基于关键词回忆的度量设计。LV-Eval的优点包括对不同上下文长度的可控评估、具有混淆事实的具有挑战性的测试实例、减少的知识泄漏以及更客观的评估。我们在LV-Eval上评估了10个LLMs，并进行了消融研究

    State-of-the-art large language models (LLMs) are now claiming remarkable supported context lengths of 256k or even more. In contrast, the average context lengths of mainstream benchmarks are insufficient (5k-21k), and they suffer from potential knowledge leakage and inaccurate metrics, resulting in biased evaluation. This paper introduces LV-Eval, a challenging long-context benchmark with five length levels (16k, 32k, 64k, 128k, and 256k) reaching up to 256k words. LV-Eval features two main tasks, single-hop QA and multi-hop QA, comprising 11 bilingual datasets. The design of LV-Eval has incorporated three key techniques, namely confusing facts insertion, keyword and phrase replacement, and keyword-recall-based metric design. The advantages of LV-Eval include controllable evaluation across different context lengths, challenging test instances with confusing facts, mitigated knowledge leakage, and more objective evaluations. We evaluate 10 LLMs on LV-Eval and conduct ablation studies o
    

