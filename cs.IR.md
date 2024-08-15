# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CFRet-DVQA: Coarse-to-Fine Retrieval and Efficient Tuning for Document Visual Question Answering](https://arxiv.org/abs/2403.00816) | 该研究提出了一种名为CFRet-DVQA的方法，通过检索和高效调优，解决了文档视觉问答中定位信息和限制模型输入的长度等问题，进一步提升了答案的生成性能。 |

# 详细

[^1]: CFRet-DVQA：粗到精检索和高效调优用于文档视觉问答

    CFRet-DVQA: Coarse-to-Fine Retrieval and Efficient Tuning for Document Visual Question Answering

    [https://arxiv.org/abs/2403.00816](https://arxiv.org/abs/2403.00816)

    该研究提出了一种名为CFRet-DVQA的方法，通过检索和高效调优，解决了文档视觉问答中定位信息和限制模型输入的长度等问题，进一步提升了答案的生成性能。

    

    文档视觉问答（DVQA）是一个涉及根据图像内容回答查询的任务。现有工作仅限于定位单页内的信息，不支持跨页面问答交互。此外，对模型输入的标记长度限制可能导致与答案相关的部分被截断。在本研究中，我们引入了一种简单但有效的方法学，称为CFRet-DVQA，重点放在检索和高效调优上，以有效解决这一关键问题。为此，我们首先从文档中检索与所提问题相关的多个片段。随后，我们利用大型语言模型（LLM）的先进推理能力，通过指导调优进一步增强其性能。该方法使得生成的答案与文档标签的风格相符。实验演示了...

    arXiv:2403.00816v1 Announce Type: cross  Abstract: Document Visual Question Answering (DVQA) is a task that involves responding to queries based on the content of images. Existing work is limited to locating information within a single page and does not facilitate cross-page question-and-answer interaction. Furthermore, the token length limitation imposed on inputs to the model may lead to truncation of segments pertinent to the answer. In this study, we introduce a simple but effective methodology called CFRet-DVQA, which focuses on retrieval and efficient tuning to address this critical issue effectively. For that, we initially retrieve multiple segments from the document that correlate with the question at hand. Subsequently, we leverage the advanced reasoning abilities of the large language model (LLM), further augmenting its performance through instruction tuning. This approach enables the generation of answers that align with the style of the document labels. The experiments demo
    

