# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [S+t-SNE - Bringing dimensionality reduction to data streams](https://arxiv.org/abs/2403.17643) | S+t-SNE是t-SNE算法的改进版本，在处理数据流时具有增量更新和盲目漂移管理的特点，能够实现高效的降维和信息可视化。 |
| [^2] | [Evolution of ESG-focused DLT Research: An NLP Analysis of the Literature.](http://arxiv.org/abs/2308.12420) | 本研究通过NLP分析了ESG主导的DLT研究的演化，通过构建引用网络和命名实体识别任务，对DLT在ESG背景下的发展进行了文献综述。 |

# 详细

[^1]: S+t-SNE - 将降维引入数据流

    S+t-SNE - Bringing dimensionality reduction to data streams

    [https://arxiv.org/abs/2403.17643](https://arxiv.org/abs/2403.17643)

    S+t-SNE是t-SNE算法的改进版本，在处理数据流时具有增量更新和盲目漂移管理的特点，能够实现高效的降维和信息可视化。

    

    我们提出了S+t-SNE，这是t-SNE算法的一种改进，旨在处理无限数据流。S+t-SNE的核心思想是随着新数据的到来逐步更新t-SNE嵌入，确保可扩展性和适应性，以处理流式场景。通过在每一步选择最重要的点，该算法确保可扩展性同时保持信息可视化。采用盲目方法进行漂移管理调整嵌入空间，促进不断可视化不断发展的数据动态。我们的实验评估证明了S+t-SNE的有效性和效率。结果突显了其在流式场景中捕捉模式的能力。我们希望我们的方法为研究人员和从业者提供一个实时工具，用于理解和解释高维数据。

    arXiv:2403.17643v1 Announce Type: new  Abstract: We present S+t-SNE, an adaptation of the t-SNE algorithm designed to handle infinite data streams. The core idea behind S+t-SNE is to update the t-SNE embedding incrementally as new data arrives, ensuring scalability and adaptability to handle streaming scenarios. By selecting the most important points at each step, the algorithm ensures scalability while keeping informative visualisations. Employing a blind method for drift management adjusts the embedding space, facilitating continuous visualisation of evolving data dynamics. Our experimental evaluations demonstrate the effectiveness and efficiency of S+t-SNE. The results highlight its ability to capture patterns in a streaming scenario. We hope our approach offers researchers and practitioners a real-time tool for understanding and interpreting high-dimensional data.
    
[^2]: ESG主导的DLT研究的演化：对文献进行NLP分析

    Evolution of ESG-focused DLT Research: An NLP Analysis of the Literature. (arXiv:2308.12420v1 [cs.IR])

    [http://arxiv.org/abs/2308.12420](http://arxiv.org/abs/2308.12420)

    本研究通过NLP分析了ESG主导的DLT研究的演化，通过构建引用网络和命名实体识别任务，对DLT在ESG背景下的发展进行了文献综述。

    

    分布式账本技术(DLT)迅速发展，需要全面了解其各个组成部分。然而，针对DLT的环境、可持续性和治理(ESG)组成部分的系统文献综述还不足。为填补这一空白，我们选择了107篇种子文献，构建了一个包含63,083个参考文献的引用网络，并将其精炼为24,539篇文献的语料库进行分析。然后，我们根据一个已建立的技术分类法从46篇论文中标记了命名实体，并通过找出DLT的ESG要素来完善这个分类法。利用基于transformer的语言模型，我们对一个预先训练的语言模型进行了细化调整，用于命名实体识别任务，使用我们标记的数据集。我们利用我们调整后的语言模型对语料库进行了精简，得到了505篇关键论文，通过命名实体和时间图分析，促进了对DLT在ESG背景下的演化的文献综述。

    Distributed Ledger Technologies (DLTs) have rapidly evolved, necessitating comprehensive insights into their diverse components. However, a systematic literature review that emphasizes the Environmental, Sustainability, and Governance (ESG) components of DLT remains lacking. To bridge this gap, we selected 107 seed papers to build a citation network of 63,083 references and refined it to a corpus of 24,539 publications for analysis. Then, we labeled the named entities in 46 papers according to twelve top-level categories derived from an established technology taxonomy and enhanced the taxonomy by pinpointing DLT's ESG elements. Leveraging transformer-based language models, we fine-tuned a pre-trained language model for a Named Entity Recognition (NER) task using our labeled dataset. We used our fine-tuned language model to distill the corpus to 505 key papers, facilitating a literature review via named entities and temporal graph analysis on DLT evolution in the context of ESG. Our con
    

