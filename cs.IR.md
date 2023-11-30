# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLMRec: Large Language Models with Graph Augmentation for Recommendation.](http://arxiv.org/abs/2311.00423) | LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。 |
| [^2] | [FASER: Binary Code Similarity Search through the use of Intermediate Representations.](http://arxiv.org/abs/2310.03605) | 本论文提出了一种名为FASER的方法，通过使用中间表示进行二进制代码相似性搜索。该方法可以跨架构地识别函数，并明确编码函数的语义，以支持各种应用场景。 |
| [^3] | [DiskANN++: Efficient Page-based Search over Isomorphic Mapped Graph Index using Query-sensitivity Entry Vertex.](http://arxiv.org/abs/2310.00402) | 提出了一个优化的DiskANN++方法，使用查询敏感的入口顶点在同构映射图索引上进行高效基于页面的搜索。 |

# 详细

[^1]: LLMRec: 使用图增强的大型语言模型用于推荐系统

    LLMRec: Large Language Models with Graph Augmentation for Recommendation. (arXiv:2311.00423v1 [cs.IR])

    [http://arxiv.org/abs/2311.00423](http://arxiv.org/abs/2311.00423)

    LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。

    

    数据稀疏性一直是推荐系统中的一个挑战，之前的研究尝试通过引入附加信息来解决这个问题。然而，这种方法往往会带来噪声、可用性问题和数据质量低下等副作用，从而影响对用户偏好的准确建模，进而对推荐性能产生不利影响。鉴于大型语言模型（LLM）在知识库和推理能力方面的最新进展，我们提出了一个名为LLMRec的新框架，它通过采用三种简单而有效的基于LLM的图增强策略来增强推荐系统。我们的方法利用在线平台（如Netflix，MovieLens）中丰富的内容，在三个方面增强交互图：（i）加强用户-物品交互边，（ii）增强对物品节点属性的理解，（iii）进行用户节点建模，直观地表示用户特征。

    The problem of data sparsity has long been a challenge in recommendation systems, and previous studies have attempted to address this issue by incorporating side information. However, this approach often introduces side effects such as noise, availability issues, and low data quality, which in turn hinder the accurate modeling of user preferences and adversely impact recommendation performance. In light of the recent advancements in large language models (LLMs), which possess extensive knowledge bases and strong reasoning capabilities, we propose a novel framework called LLMRec that enhances recommender systems by employing three simple yet effective LLM-based graph augmentation strategies. Our approach leverages the rich content available within online platforms (e.g., Netflix, MovieLens) to augment the interaction graph in three ways: (i) reinforcing user-item interaction egde, (ii) enhancing the understanding of item node attributes, and (iii) conducting user node profiling, intuiti
    
[^2]: FASER: 通过中间表示进行二进制代码相似性搜索

    FASER: Binary Code Similarity Search through the use of Intermediate Representations. (arXiv:2310.03605v1 [cs.CR])

    [http://arxiv.org/abs/2310.03605](http://arxiv.org/abs/2310.03605)

    本论文提出了一种名为FASER的方法，通过使用中间表示进行二进制代码相似性搜索。该方法可以跨架构地识别函数，并明确编码函数的语义，以支持各种应用场景。

    

    能够识别跨架构软件中感兴趣的函数对于分析恶意软件、保护软件供应链或进行漏洞研究都是有用的。跨架构二进制代码相似性搜索已在许多研究中探索，并使用了各种不同的数据来源来实现其目标。通常使用的数据来源包括从二进制文件中提取的常见结构，如函数控制流图或二进制级调用图，反汇编过程的输出或动态分析方法的输出。其中一种受到较少关注的数据来源是二进制中间表示。二进制中间表示具有两个有趣的属性：它们的跨架构性质以及明确编码函数的语义以支持下游使用。在本文中，我们提出了一种名为FASER的函数字符串编码表示方法，它结合了长文档转换技术。

    Being able to identify functions of interest in cross-architecture software is useful whether you are analysing for malware, securing the software supply chain or conducting vulnerability research. Cross-Architecture Binary Code Similarity Search has been explored in numerous studies and has used a wide range of different data sources to achieve its goals. The data sources typically used draw on common structures derived from binaries such as function control flow graphs or binary level call graphs, the output of the disassembly process or the outputs of a dynamic analysis approach. One data source which has received less attention is binary intermediate representations. Binary Intermediate representations possess two interesting properties: they are cross architecture by their very nature and encode the semantics of a function explicitly to support downstream usage. Within this paper we propose Function as a String Encoded Representation (FASER) which combines long document transforme
    
[^3]: DiskANN++: 使用查询敏感的入口顶点在同构映射图索引上进行高效基于页面的搜索

    DiskANN++: Efficient Page-based Search over Isomorphic Mapped Graph Index using Query-sensitivity Entry Vertex. (arXiv:2310.00402v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.00402](http://arxiv.org/abs/2310.00402)

    提出了一个优化的DiskANN++方法，使用查询敏感的入口顶点在同构映射图索引上进行高效基于页面的搜索。

    

    给定一个向量数据集X和一个查询向量xq，基于图的近似最近邻搜索(ANNS)旨在构建一个图索引G，并通过在G上搜索来近似返回与xq的最小距离向量。基于图的ANNS的主要缺点是图索引太大，无法适应尤其是大规模X的内存。为了解决这个问题，提出了一种基于产品量化(PQ)的混合方法DiskANN，它在内存中存储低维度的PQ索引，并在SSD中保留图索引，从而减小内存开销同时确保高搜索准确性。然而，它存在两个重要的I/O问题，严重影响了整体效率：(1)从入口顶点到查询邻域的长路径导致大量的I/O请求，以及(2)在路由过程中的冗余I/O请求。为了解决上述问题，我们提出了优化的DiskANN++。

    Given a vector dataset $\mathcal{X}$ and a query vector $\vec{x}_q$, graph-based Approximate Nearest Neighbor Search (ANNS) aims to build a graph index $G$ and approximately return vectors with minimum distances to $\vec{x}_q$ by searching over $G$. The main drawback of graph-based ANNS is that a graph index would be too large to fit into the memory especially for a large-scale $\mathcal{X}$. To solve this, a Product Quantization (PQ)-based hybrid method called DiskANN is proposed to store a low-dimensional PQ index in memory and retain a graph index in SSD, thus reducing memory overhead while ensuring a high search accuracy. However, it suffers from two I/O issues that significantly affect the overall efficiency: (1) long routing path from an entry vertex to the query's neighborhood that results in large number of I/O requests and (2) redundant I/O requests during the routing process. We propose an optimized DiskANN++ to overcome above issues. Specifically, for the first issue, we pre
    

