# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LIST: Learning to Index Spatio-Textual Data for Embedding based Spatial Keyword Queries](https://arxiv.org/abs/2403.07331) | 提出了一种名为LIST的新技术，通过学习为基于嵌入的空间关键词查询建立空间文本数据索引，以加速top-k搜索过程。 |

# 详细

[^1]: LIST: 学习为基于嵌入的空间关键词查询建立空间文本数据索引

    LIST: Learning to Index Spatio-Textual Data for Embedding based Spatial Keyword Queries

    [https://arxiv.org/abs/2403.07331](https://arxiv.org/abs/2403.07331)

    提出了一种名为LIST的新技术，通过学习为基于嵌入的空间关键词查询建立空间文本数据索引，以加速top-k搜索过程。

    

    随着空间文本数据的普及，“Top-k KNN空间关键词查询（TkQs）”已经在许多实际应用中发现，它基于一个评价空间和文本相关性的排名函数返回一个对象列表。现有的用于TkQs的geo-textual索引使用传统的检索模型（如BM25）来计算文本相关性，并通常利用简单的线性函数来计算空间相关性，但其效果有限。为了提高效果，最近提出了几种深度学习模型，但它们存在严重的效率问题。据我们所知，目前没有为加速这些深度学习模型的top-k搜索过程专门设计的有效索引。为了解决这些问题，我们提出了一种新技术，通过学习为回答基于嵌入的空间关键词查询（称为LIST）建立空间文本数据索引。LIST具有两个新颖组件。

    arXiv:2403.07331v1 Announce Type: new  Abstract: With the proliferation of spatio-textual data, Top-k KNN spatial keyword queries (TkQs), which return a list of objects based on a ranking function that evaluates both spatial and textual relevance, have found many real-life applications. Existing geo-textual indexes for TkQs use traditional retrieval models like BM25 to compute text relevance and usually exploit a simple linear function to compute spatial relevance, but its effectiveness is limited. To improve effectiveness, several deep learning models have recently been proposed, but they suffer severe efficiency issues. To the best of our knowledge, there are no efficient indexes specifically designed to accelerate the top-k search process for these deep learning models.   To tackle these issues, we propose a novel technique, which Learns to Index the Spatio-Textual data for answering embedding based spatial keyword queries (called LIST). LIST is featured with two novel components. F
    

