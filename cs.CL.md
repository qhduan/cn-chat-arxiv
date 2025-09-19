# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast Multipole Attention: A Divide-and-Conquer Attention Mechanism for Long Sequences.](http://arxiv.org/abs/2310.11960) | 提出了一种名为快速多极化注意力的新型注意力机制，它使用分治策略将注意力的时间和内存复杂度从O(n^2)降低到O(n log n)或O(n)，同时保持了全局感知范围。 |

# 详细

[^1]: 快速多极化注意力：一种用于长序列的分治注意力机制

    Fast Multipole Attention: A Divide-and-Conquer Attention Mechanism for Long Sequences. (arXiv:2310.11960v1 [cs.CL])

    [http://arxiv.org/abs/2310.11960](http://arxiv.org/abs/2310.11960)

    提出了一种名为快速多极化注意力的新型注意力机制，它使用分治策略将注意力的时间和内存复杂度从O(n^2)降低到O(n log n)或O(n)，同时保持了全局感知范围。

    

    基于Transformer的模型已在许多领域取得了最先进的性能。然而，自注意力对于输入长度的二次复杂度限制了Transformer模型在长序列上的适用性。为了解决这个问题，我们提出了快速多极化注意力，一种使用分治策略来减少注意力时间和内存复杂度的新型注意力机制，将长度为n的序列的注意力复杂度从O(n^2)降低到O(n log n)或O(n)，同时保持了全局感知范围。这种分层方法将查询、键和值分为O(log n)级的分辨率，较远距离的组群越来越大，并学习计算组群数量的权重。因此，以高效分层的方式在较低的分辨率中考虑远离彼此的标记之间的相互作用。快速多极化注意力的总体复杂度为O(n)或O(n log n)。

    Transformer-based models have achieved state-of-the-art performance in many areas. However, the quadratic complexity of self-attention with respect to the input length hinders the applicability of Transformer-based models to long sequences. To address this, we present Fast Multipole Attention, a new attention mechanism that uses a divide-and-conquer strategy to reduce the time and memory complexity of attention for sequences of length $n$ from $\mathcal{O}(n^2)$ to $\mathcal{O}(n \log n)$ or $O(n)$, while retaining a global receptive field. The hierarchical approach groups queries, keys, and values into $\mathcal{O}( \log n)$ levels of resolution, where groups at greater distances are increasingly larger in size and the weights to compute group quantities are learned. As such, the interaction between tokens far from each other is considered in lower resolution in an efficient hierarchical manner. The overall complexity of Fast Multipole Attention is $\mathcal{O}(n)$ or $\mathcal{O}(n \
    

