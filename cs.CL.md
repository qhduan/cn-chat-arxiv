# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](https://arxiv.org/abs/2403.14773) | StreamingT2V是一种自回归方法，用于生成长视频，可以产生80、240、600、1200帧甚至更多帧的视频，并具有平滑的过渡。 |
| [^2] | [Local Grammar-Based Coding Revisited.](http://arxiv.org/abs/2209.13636) | 本文重新审视了最小局部基于语法的编码问题，并提出了一种新的、更简单、更普遍的证明方法，证明了最小分块编码具有强大的普遍性。同时，通过实验也表明，最小分块编码中的规则数量不能明确区分长记忆和无记忆的源。 |

# 详细

[^1]: StreamingT2V: 一种一致、动态和可扩展的基于文本的长视频生成方法

    StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text

    [https://arxiv.org/abs/2403.14773](https://arxiv.org/abs/2403.14773)

    StreamingT2V是一种自回归方法，用于生成长视频，可以产生80、240、600、1200帧甚至更多帧的视频，并具有平滑的过渡。

    

    arXiv:2403.14773v1 公告类型: 交叉 摘要: 文本到视频的扩散模型可以生成遵循文本指令的高质量视频，使得创建多样化和个性化内容变得更加容易。然而，现有方法大多集中在生成高质量的短视频（通常为16或24帧），当天真地扩展到长视频合成的情况时，通常会出现硬裁剪。为了克服这些限制，我们引入了StreamingT2V，这是一种自回归方法，用于生成80、240、600、1200或更多帧的长视频，具有平滑的过渡。主要组件包括：（i）一种名为条件注意力模块（CAM）的短期记忆块，通过注意机制将当前生成条件设置为先前块提取的特征，实现一致的块过渡，（ii）一种名为外观保存模块的长期记忆块，从第一个视频块中提取高级场景和对象特征，以防止th

    arXiv:2403.14773v1 Announce Type: cross  Abstract: Text-to-video diffusion models enable the generation of high-quality videos that follow text instructions, making it easy to create diverse and individual content. However, existing approaches mostly focus on high-quality short video generation (typically 16 or 24 frames), ending up with hard-cuts when naively extended to the case of long video synthesis. To overcome these limitations, we introduce StreamingT2V, an autoregressive approach for long video generation of 80, 240, 600, 1200 or more frames with smooth transitions. The key components are:(i) a short-term memory block called conditional attention module (CAM), which conditions the current generation on the features extracted from the previous chunk via an attentional mechanism, leading to consistent chunk transitions, (ii) a long-term memory block called appearance preservation module, which extracts high-level scene and object features from the first video chunk to prevent th
    
[^2]: 本文重新审视了局部基于语法的编码问题

    Local Grammar-Based Coding Revisited. (arXiv:2209.13636v2 [cs.IT] UPDATED)

    [http://arxiv.org/abs/2209.13636](http://arxiv.org/abs/2209.13636)

    本文重新审视了最小局部基于语法的编码问题，并提出了一种新的、更简单、更普遍的证明方法，证明了最小分块编码具有强大的普遍性。同时，通过实验也表明，最小分块编码中的规则数量不能明确区分长记忆和无记忆的源。

    

    本文重新审视了最小局部基于语法的编码问题。在这个设置中，局部基于语法的编码器逐个符号地对语法进行编码，而最小语法变换通过局部语法编码的长度在预设的语法类别中最小化语法长度。已知，这样的最小编码对于严格正熵率的情况具有强大的普遍性，而最小语法中的规则数量构成了源的互信息的上界。尽管完全最小编码可能是不可行的，但受限的最小分块编码可以有效计算。本文提出了一种新的、更简单、更普适的最小分块编码强大普遍性的证明方法，不受熵率的限制。该证明基于对排名概率的简单的Zipfian界限。顺便提一下，我们还通过实验证明，最小分块编码中的规则数量不能明确区分长记忆和无记忆的源。

    We revisit the problem of minimal local grammar-based coding. In this setting, the local grammar encoder encodes grammars symbol by symbol, whereas the minimal grammar transform minimizes the grammar length in a preset class of grammars as given by the length of local grammar encoding. It has been known that such minimal codes are strongly universal for a strictly positive entropy rate, whereas the number of rules in the minimal grammar constitutes an upper bound for the mutual information of the source. Whereas the fully minimal code is likely intractable, the constrained minimal block code can be efficiently computed. In this article, we present a new, simpler, and more general proof of strong universality of the minimal block code, regardless of the entropy rate. The proof is based on a simple Zipfian bound for ranked probabilities. By the way, we also show empirically that the number of rules in the minimal block code cannot clearly discriminate between long-memory and memoryless s
    

