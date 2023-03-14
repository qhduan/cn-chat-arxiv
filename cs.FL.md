# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Self-Attention Networks Can Process Bounded Hierarchical Languages.](http://arxiv.org/abs/2105.11115) | 本文证明了自注意力网络可以处理深度受限的$\mathsf{Dyck}_{k}$子集$\mathsf{Dyck}_{k,D}$，这更好地捕捉了自然语言的有界层次结构。 |

# 详细

[^1]: 自注意力网络可以处理有界层次语言

    Self-Attention Networks Can Process Bounded Hierarchical Languages. (arXiv:2105.11115v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2105.11115](http://arxiv.org/abs/2105.11115)

    本文证明了自注意力网络可以处理深度受限的$\mathsf{Dyck}_{k}$子集$\mathsf{Dyck}_{k,D}$，这更好地捕捉了自然语言的有界层次结构。

    This paper proves that self-attention networks can process the depth-bounded subset of $\mathsf{Dyck}_{k}$, $\mathsf{Dyck}_{k,D}$, which better captures the bounded hierarchical structure of natural language.

    尽管自注意力网络在自然语言处理中表现出色，但最近证明它们在处理具有分层结构的形式语言（例如$\mathsf{Dyck}_k$，由$k$种嵌套括号组成的语言）方面存在局限性。这表明自然语言可以用模型很好地近似，而这些模型对于形式语言来说过于弱，或者说层次结构和递归在自然语言中的作用可能是有限的。我们通过证明自注意力网络可以处理$\mathsf{Dyck}_{k,D}$来限定这一含义，其中$\mathsf{Dyck}_{k,D}$是深度受限的$\mathsf{Dyck}_{k}$子集，它更好地捕捉了自然语言的有界层次结构。具体而言，我们构建了一个具有$D+1$层和$O(\log k)$内存大小（每个令牌每层）的硬注意力网络，用于识别$\mathsf{Dyck}_{k,D}$，以及一个具有两层和$O(\log k)$内存大小的软注意力网络，用于生成$\mathsf{Dyck}_{k,D}$。实验表明，软注意力网络可以在$\mathsf{Dyck}_{k,D}$上生成正确的序列。

    Despite their impressive performance in NLP, self-attention networks were recently proved to be limited for processing formal languages with hierarchical structure, such as $\mathsf{Dyck}_k$, the language consisting of well-nested parentheses of $k$ types. This suggested that natural language can be approximated well with models that are too weak for formal languages, or that the role of hierarchy and recursion in natural language might be limited. We qualify this implication by proving that self-attention networks can process $\mathsf{Dyck}_{k, D}$, the subset of $\mathsf{Dyck}_{k}$ with depth bounded by $D$, which arguably better captures the bounded hierarchical structure of natural language. Specifically, we construct a hard-attention network with $D+1$ layers and $O(\log k)$ memory size (per token per layer) that recognizes $\mathsf{Dyck}_{k, D}$, and a soft-attention network with two layers and $O(\log k)$ memory size that generates $\mathsf{Dyck}_{k, D}$. Experiments show that s
    

