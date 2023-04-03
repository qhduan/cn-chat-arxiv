# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Task Oriented Conversational Modelling With Subjective Knowledge.](http://arxiv.org/abs/2303.17695) | 本文提出了一种改进知识选择模块的实体检索方法，并探讨了一种潜在的关键字提取方法，以提高任务导向交互建模系统的性能。 |
| [^2] | [Bounded Simplex-Structured Matrix Factorization: Algorithms, Identifiability and Applications.](http://arxiv.org/abs/2209.12638) | 提出了一种新的低秩矩阵分解模型BSSMF，它的矩阵W每列的元素属于给定的区间，而H的列是随机的，推广了NMF和SSMF，适用于矩阵元素属于给定区间的情况下，具有易于理解的分解和离散结构，适用于主题建模和社区检测等应用。 |

# 详细

[^1]: 面向任务的主观知识交互建模

    Task Oriented Conversational Modelling With Subjective Knowledge. (arXiv:2303.17695v1 [cs.CL])

    [http://arxiv.org/abs/2303.17695](http://arxiv.org/abs/2303.17695)

    本文提出了一种改进知识选择模块的实体检索方法，并探讨了一种潜在的关键字提取方法，以提高任务导向交互建模系统的性能。

    

    现有的对话模型都是基于数据库和API的系统来处理的。但是，用户的问题经常需要处理这些系统无法处理的信息。然而，这些问题的答案可以在客户评价和常见问题解答中找到。DSTC-11提出了一个由三个部分组成的管道，包括知识寻求回合检测、知识选择和响应生成，从而创建一个基于主观知识的交互式模型。本文着重于改进知识选择模块，以提高整个系统的性能。我们提出了一种实体检索方法，它可以实现准确和更快的知识搜索。我们提出的基于命名实体识别(NER)的实体检索方法比基线模型快了7倍。此外，我们还探讨了一种潜在的关键字提取方法，可以提高知识选择的准确性。初步结果显示了4\%的改进。

    Existing conversational models are handled by a database(DB) and API based systems. However, very often users' questions require information that cannot be handled by such systems. Nonetheless, answers to these questions are available in the form of customer reviews and FAQs. DSTC-11 proposes a three stage pipeline consisting of knowledge seeking turn detection, knowledge selection and response generation to create a conversational model grounded on this subjective knowledge. In this paper, we focus on improving the knowledge selection module to enhance the overall system performance. In particular, we propose entity retrieval methods which result in an accurate and faster knowledge search. Our proposed Named Entity Recognition (NER) based entity retrieval method results in 7X faster search compared to the baseline model. Additionally, we also explore a potential keyword extraction method which can improve the accuracy of knowledge selection. Preliminary results show a 4 \% improvement
    
[^2]: 有界单纯形结构矩阵分解：算法、可识别性和应用

    Bounded Simplex-Structured Matrix Factorization: Algorithms, Identifiability and Applications. (arXiv:2209.12638v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.12638](http://arxiv.org/abs/2209.12638)

    提出了一种新的低秩矩阵分解模型BSSMF，它的矩阵W每列的元素属于给定的区间，而H的列是随机的，推广了NMF和SSMF，适用于矩阵元素属于给定区间的情况下，具有易于理解的分解和离散结构，适用于主题建模和社区检测等应用。

    

    本文提出了一种新的低秩矩阵分解模型，称为有界单纯形结构矩阵分解（BSSMF）。给定一个输入矩阵X和一个分解秩r，BSSMF在矩阵W中寻找具有r列的矩阵和在矩阵H中寻找具有r行的矩阵，使得X≈WH ，其中W的每列中的元素都是有界的，即它们属于给定的区间，而H的列属于概率单纯形，即H是列随机的。BSSMF推广了非负矩阵分解（NMF）和单纯形结构矩阵分解（SSMF）。BSSMF特别适用于输入矩阵X的元素属于给定区间的情况；例如，当X的行表示图像时，或者X是类似Netflix和MovieLens数据集中的评分矩阵时，其中X的元素属于区间[1,5]。单纯形结构矩阵H不仅可以提供易于理解的分解，从而对X的列空间进行软聚类，而且还赋予H的列离散结构，使其非常适合用于如主题建模和社区检测等应用。我们开发了有效的BSSMF优化算法，建立了其可识别性保证，并在合成和实际数据集上展示了BSSMF的有效性。

    In this paper, we propose a new low-rank matrix factorization model dubbed bounded simplex-structured matrix factorization (BSSMF). Given an input matrix $X$ and a factorization rank $r$, BSSMF looks for a matrix $W$ with $r$ columns and a matrix $H$ with $r$ rows such that $X \approx WH$ where the entries in each column of $W$ are bounded, that is, they belong to given intervals, and the columns of $H$ belong to the probability simplex, that is, $H$ is column stochastic. BSSMF generalizes nonnegative matrix factorization (NMF), and simplex-structured matrix factorization (SSMF). BSSMF is particularly well suited when the entries of the input matrix $X$ belong to a given interval; for example when the rows of $X$ represent images, or $X$ is a rating matrix such as in the Netflix and MovieLens datasets where the entries of $X$ belong to the interval $[1,5]$. The simplex-structured matrix $H$ not only leads to an easily understandable decomposition providing a soft clustering of the colu
    

