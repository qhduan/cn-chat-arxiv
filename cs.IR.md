# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Taxonomy of Mathematical Plagiarism.](http://arxiv.org/abs/2401.16969) | 本论文建立了数学内容重用的分类法，并分析了剽窃检测和数学内容相似性的最佳方法。研究发现，目前最佳方法在剽窃和数学内容相似性方面的表现依然不理想。这些发现将为剽窃检测系统、推荐系统和问答系统等领域的研究提供帮助。 |
| [^2] | [Improving Text Embeddings with Large Language Models.](http://arxiv.org/abs/2401.00368) | 本文介绍了一种使用只用合成数据和少量训练步骤获取高质量文本嵌入的简单方法，并且在没有使用标记数据的情况下，在竞争激烈的文本嵌入基准上取得了强大的性能。 |

# 详细

[^1]: 数学剽窃分类法

    Taxonomy of Mathematical Plagiarism. (arXiv:2401.16969v1 [cs.IR])

    [http://arxiv.org/abs/2401.16969](http://arxiv.org/abs/2401.16969)

    本论文建立了数学内容重用的分类法，并分析了剽窃检测和数学内容相似性的最佳方法。研究发现，目前最佳方法在剽窃和数学内容相似性方面的表现依然不理想。这些发现将为剽窃检测系统、推荐系统和问答系统等领域的研究提供帮助。

    

    剽窃问题是一个紧迫的关注点，尤其是在大型语言模型的可用性下更为突出。现有的剽窃检测系统可以可靠地找到复制和适度改写的文本，但在数学科学中的思想剽窃方面表现不佳，因为数学科学中使用了严格的数学符号。我们做出了两个贡献。首先，我们通过对可能存在剽窃的122个科学文档进行注释，建立了数学内容重用的分类法。其次，我们对刚刚建立的分类法上最佳表现的剽窃检测方法和数学内容相似性进行了分析。我们发现，对于剽窃和数学内容相似性，表现最佳的方法分别达到了0.06和0.16的整体检测分数（PlagDet）。这些最佳方法未能检测出七种新建立的数学相似性类型中的大部分案例。我们的贡献将有助于剽窃检测系统、推荐系统、问答系统和其他相关研究的发展。

    Plagiarism is a pressing concern, even more so with the availability of large language models. Existing plagiarism detection systems reliably find copied and moderately reworded text but fail for idea plagiarism, especially in mathematical science, which heavily uses formal mathematical notation. We make two contributions. First, we establish a taxonomy of mathematical content reuse by annotating potentially plagiarised 122 scientific document pairs. Second, we analyze the best-performing approaches to detect plagiarism and mathematical content similarity on the newly established taxonomy. We found that the best-performing methods for plagiarism and math content similarity achieve an overall detection score (PlagDet) of 0.06 and 0.16, respectively. The best-performing methods failed to detect most cases from all seven newly established math similarity types. Outlined contributions will benefit research in plagiarism detection systems, recommender systems, question-answering systems, an
    
[^2]: 用大型语言模型改善文本嵌入

    Improving Text Embeddings with Large Language Models. (arXiv:2401.00368v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.00368](http://arxiv.org/abs/2401.00368)

    本文介绍了一种使用只用合成数据和少量训练步骤获取高质量文本嵌入的简单方法，并且在没有使用标记数据的情况下，在竞争激烈的文本嵌入基准上取得了强大的性能。

    

    在本文中，我们介绍了一种新颖且简单的方法，仅使用合成数据和少于1k个训练步骤即可获得高质量的文本嵌入。与现有方法不同，现有方法往往依赖多阶段中间预训练，使用数十亿个弱监督文本对进行训练，然后再使用少量标记数据进行微调，我们的方法不需要构建复杂的训练流程，也不依赖于通常受任务多样性和语言覆盖范围限制的手动收集的数据集。我们利用专有的LLM来为近100种语言的数十万个文本嵌入任务生成多样的合成数据。然后，我们使用标准的对比损失在合成数据上微调开源的只有解码器的LLM。实验证明，我们的方法在竞争激烈的文本嵌入基准上取得了出色的性能，而且没有使用任何标记数据。此外，当与合成数据和标记数据的混合进行微调时，我们的模型创造了新的

    In this paper, we introduce a novel and simple method for obtaining high-quality text embeddings using only synthetic data and less than 1k training steps. Unlike existing methods that often depend on multi-stage intermediate pre-training with billions of weakly-supervised text pairs, followed by fine-tuning with a few labeled datasets, our method does not require building complex training pipelines or relying on manually collected datasets that are often constrained by task diversity and language coverage. We leverage proprietary LLMs to generate diverse synthetic data for hundreds of thousands of text embedding tasks across nearly 100 languages. We then fine-tune open-source decoder-only LLMs on the synthetic data using standard contrastive loss. Experiments demonstrate that our method achieves strong performance on highly competitive text embedding benchmarks without using any labeled data. Furthermore, when fine-tuned with a mixture of synthetic and labeled data, our model sets new
    

