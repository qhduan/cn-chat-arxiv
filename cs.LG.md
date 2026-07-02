# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PersistentKV: Page-Aware Decode Scheduling for Long-Context LLM Serving on Commodity GPUs](https://arxiv.org/abs/2606.26666) | 本文提出PersistentKV，一种针对商用GPU上长上下文LLM服务的页面感知解码调度引擎，通过按KV头组映射工作、重用K/V块和紧凑工作队列调度，优化了页面感知解码的效率。 |

# 详细

[^1]: PersistentKV：面向长上下文LLM在商用GPU上服务的页面感知解码调度

    PersistentKV: Page-Aware Decode Scheduling for Long-Context LLM Serving on Commodity GPUs

    [https://arxiv.org/abs/2606.26666](https://arxiv.org/abs/2606.26666)

    本文提出PersistentKV，一种针对商用GPU上长上下文LLM服务的页面感知解码调度引擎，通过按KV头组映射工作、重用K/V块和紧凑工作队列调度，优化了页面感知解码的效率。

    

    自回归大型语言模型（LLM）服务正越来越多地受到键值（KV）缓存移动的限制，而非密集矩阵乘法。现代分页注意力系统减少了KV缓存碎片，而成熟的核函数（如FlashInfer）提供了高度优化的原生分页解码注意力。然而，最佳的单核实现并不总是最佳的服务调度方案：低活跃度的长上下文解码可能无法充分利用商用GPU，而混合序列长度则在精确长度启动和粗粒度填充批次之间引入了张力。我们提出了PersistentKV，一种原生的块表解码注意力引擎，并针对分组查询注意力（GQA）进行了页面感知调度研究。PersistentKV按KV头组映射工作，设计用于跨分组查询头重用K、V块，支持原生页表，并添加了一个紧凑的工作队列调度，该调度仅执行非空的行-KV头-序列拆分任务。在RTX 3060上

    arXiv:2606.26666v1 Announce Type: new  Abstract: Autoregressive large language model (LLM) serving is increasingly limited by key-value (KV) cache movement rather than dense matrix multiplication. Modern paged-attention systems reduce KV-cache fragmentation and mature kernels such as FlashInfer provide highly optimized native-paged decode attention. However, the best single-kernel implementation is not always the best serving schedule: low-active long-context decode can under-utilize commodity GPUs, while mixed sequence lengths introduce a tension between many exact-length launches and coarse padded batches. We present PersistentKV, a native block-table decode attention engine and page-aware scheduling study for grouped-query attention (GQA). PersistentKV maps work by KV-head group, is designed to reuse K,V tiles across grouped query heads, supports native page tables, and adds a compact workqueue schedule that executes only non-empty row-KV-head-sequence-split tasks. On an RTX 3060 wi
    

