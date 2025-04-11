# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SD-HuBERT: Self-Distillation Induces Syllabic Organization in HuBERT.](http://arxiv.org/abs/2310.10803) | 本研究提出了SD-HuBERT模型，通过采用自我蒸馏目标进行微调，实现了在学习语音句子级表示时音节组织的出现，模型能够在语音中划定明确的边界，并展现出显著的音节结构。该研究还提出了一个新的基准任务用于评估语音的句子级表示，与之前的模型相比，在无监督音节发现和学习句子级表示方面表现优异。 |
| [^2] | [Comparative Performance Evaluation of Large Language Models for Extracting Molecular Interactions and Pathway Knowledge.](http://arxiv.org/abs/2307.08813) | 本研究评估了不同大型语言模型在提取分子相互作用和通路知识方面的有效性，并讨论了未来机遇和挑战。 |
| [^3] | [BUCA: A Binary Classification Approach to Unsupervised Commonsense Question Answering.](http://arxiv.org/abs/2305.15932) | 本文提出了一种更简单的二分类方法，将下游的多项选择题回答任务转换为二分类任务，根据合理性对所有候选答案进行排名，以实现无监督常识问题回答，相较于现有使用知识图谱的UCR方法，我们的方法更为节省数据。 |
| [^4] | [Medical Intervention Duration Estimation Using Language-enhanced Transformer Encoder with Medical Prompts.](http://arxiv.org/abs/2303.17408) | 使用语言增强Transformer编码器，并结合医学提示，将结构化、非结构化的临床数据投影到一个语言潜空间中，以实现更精确的医学干预持续时间估计。 |

# 详细

[^1]: SD-HuBERT: 自我蒸馏诱导HuBERT中的音节组织

    SD-HuBERT: Self-Distillation Induces Syllabic Organization in HuBERT. (arXiv:2310.10803v1 [cs.CL])

    [http://arxiv.org/abs/2310.10803](http://arxiv.org/abs/2310.10803)

    本研究提出了SD-HuBERT模型，通过采用自我蒸馏目标进行微调，实现了在学习语音句子级表示时音节组织的出现，模型能够在语音中划定明确的边界，并展现出显著的音节结构。该研究还提出了一个新的基准任务用于评估语音的句子级表示，与之前的模型相比，在无监督音节发现和学习句子级表示方面表现优异。

    

    自我监督学习（SSL）中的数据驱动单元发现开启了口语语言处理的新时代。然而，发现的单元往往仍处于音素空间，限制了SSL表示的实用性。在这里，我们展示了在学习语音的句子级表示时，音节组织的出现。特别地，我们采用“自我蒸馏”目标来微调预训练的HuBERT，并加入一个汇聚标记来总结整个句子。在没有任何监督的情况下，得到的模型在语音中划定了明确的边界，并且帧间的表示显示出显著的音节结构。我们证明这种出现的结构很大程度上与真实音节对应。此外，我们提出了一个新的基准任务，Spoken Speech ABX，用于评估语音的句子级表示。与之前的模型相比，我们的模型在无监督音节发现和学习句子级表示方面表现优异。

    Data-driven unit discovery in self-supervised learning (SSL) of speech has embarked on a new era of spoken language processing. Yet, the discovered units often remain in phonetic space, limiting the utility of SSL representations. Here, we demonstrate that a syllabic organization emerges in learning sentence-level representation of speech. In particular, we adopt "self-distillation" objective to fine-tune the pretrained HuBERT with an aggregator token that summarizes the entire sentence. Without any supervision, the resulting model draws definite boundaries in speech, and the representations across frames show salient syllabic structures. We demonstrate that this emergent structure largely corresponds to the ground truth syllables. Furthermore, we propose a new benchmark task, Spoken Speech ABX, for evaluating sentence-level representation of speech. When compared to previous models, our model outperforms in both unsupervised syllable discovery and learning sentence-level representatio
    
[^2]: 大型语言模型在提取分子相互作用和通路知识方面的比较性能评估

    Comparative Performance Evaluation of Large Language Models for Extracting Molecular Interactions and Pathway Knowledge. (arXiv:2307.08813v1 [cs.CL])

    [http://arxiv.org/abs/2307.08813](http://arxiv.org/abs/2307.08813)

    本研究评估了不同大型语言模型在提取分子相互作用和通路知识方面的有效性，并讨论了未来机遇和挑战。

    

    理解蛋白质相互作用和通路知识对于揭示生物系统的复杂性和研究生物功能和复杂疾病的基本机制至关重要。尽管现有的数据库提供了来自文献和其他源的策划生物数据，但它们往往不完整且维护工作繁重，因此需要替代方法。在本研究中，我们提出利用大型语言模型的能力，通过自动从相关科学文献中提取这些知识来解决这些问题。为了实现这个目标，在这项工作中，我们调查了不同大型语言模型在识别蛋白质相互作用、通路和基因调控关系等任务中的有效性。我们对不同模型的性能进行了彻底评估，突出了重要的发现，并讨论了这种方法所面临的未来机遇和挑战。代码和数据集链接可在论文中找到。

    Understanding protein interactions and pathway knowledge is crucial for unraveling the complexities of living systems and investigating the underlying mechanisms of biological functions and complex diseases. While existing databases provide curated biological data from literature and other sources, they are often incomplete and their maintenance is labor-intensive, necessitating alternative approaches. In this study, we propose to harness the capabilities of large language models to address these issues by automatically extracting such knowledge from the relevant scientific literature. Toward this goal, in this work, we investigate the effectiveness of different large language models in tasks that involve recognizing protein interactions, pathways, and gene regulatory relations. We thoroughly evaluate the performance of various models, highlight the significant findings, and discuss both the future opportunities and the remaining challenges associated with this approach. The code and d
    
[^3]: BUCA：一种用于无监督常识问题回答的二分类方法

    BUCA: A Binary Classification Approach to Unsupervised Commonsense Question Answering. (arXiv:2305.15932v1 [cs.CL])

    [http://arxiv.org/abs/2305.15932](http://arxiv.org/abs/2305.15932)

    本文提出了一种更简单的二分类方法，将下游的多项选择题回答任务转换为二分类任务，根据合理性对所有候选答案进行排名，以实现无监督常识问题回答，相较于现有使用知识图谱的UCR方法，我们的方法更为节省数据。

    

    随着常识推理数据集的构建变得越来越昂贵且在范围上不可避免地受限，无监督的常识推理(UCR)变得越来越流行。UCR的一种流行方法是利用外部知识将语言模型进行微调(例如，知识图谱)，但这通常需要大量的训练样例。在本文中，我们提出将下游的多项选择题回答任务转换为一个更简单的二分类任务，通过对所有候选答案的合理性进行排名来完成。为了训练模型，我们将知识图谱三元组转换为合理和不合理的文本。广泛的实验结果显示了我们的方法在各种多项选择问题回答基准测试中的有效性。此外，与使用KG的现有UCR方法相比，我们的方法更节省数据。我们的代码可在https://github.com/probe2/BUCA上获取。

    Unsupervised commonsense reasoning (UCR) is becoming increasingly popular as the construction of commonsense reasoning datasets is expensive, and they are inevitably limited in their scope. A popular approach to UCR is to fine-tune language models with external knowledge (e.g., knowledge graphs), but this usually requires a large number of training examples. In this paper, we propose to transform the downstream multiple choice question answering task into a simpler binary classification task by ranking all candidate answers according to their reasonableness. To this end, for training the model, we convert the knowledge graph triples into reasonable and unreasonable texts. Extensive experimental results show the effectiveness of our approach on various multiple choice question answering benchmarks. Furthermore, compared with existing UCR approaches using KGs, ours is less data hungry. Our code is available at https://github.com/probe2/BUCA.
    
[^4]: 基于医学提示的语言增强Transformer编码器的医疗干预持续时间估计

    Medical Intervention Duration Estimation Using Language-enhanced Transformer Encoder with Medical Prompts. (arXiv:2303.17408v1 [cs.CL])

    [http://arxiv.org/abs/2303.17408](http://arxiv.org/abs/2303.17408)

    使用语言增强Transformer编码器，并结合医学提示，将结构化、非结构化的临床数据投影到一个语言潜空间中，以实现更精确的医学干预持续时间估计。

    

    近年来，基于电子病历(EHRs)估计医疗干预的持续时间在临床决策支持领域引起了重视。然而，当前的模型主要关注结构化数据，忽略了来自非结构化的临床自由文本数据的信息。为了解决这个问题，我们提出了一个新颖的语言增强Transformer-based框架，它使用经过预训练的句子编码器将所有相关的临床数据模态（连续、分类、二进制和自由文本特征）投影到一个协调的语言潜空间中，借助医学提示。所提出的方法使得不同模态的信息在单元变压器编码器中集成起来，从而实现更准确的医学干预持续时间估计。我们在美国（ICU住院时间估计）和亚洲（手术持续时间预测）医学数据集上的实验结果证明了我们提出的框架的有效性。

    In recent years, estimating the duration of medical intervention based on electronic health records (EHRs) has gained significant attention in the filed of clinical decision support. However, current models largely focus on structured data, leaving out information from the unstructured clinical free-text data. To address this, we present a novel language-enhanced transformer-based framework, which projects all relevant clinical data modalities (continuous, categorical, binary, and free-text features) into a harmonized language latent space using a pre-trained sentence encoder with the help of medical prompts. The proposed method enables the integration of information from different modalities within the cell transformer encoder and leads to more accurate duration estimation for medical intervention. Our experimental results on both US-based (length of stay in ICU estimation) and Asian (surgical duration prediction) medical datasets demonstrate the effectiveness of our proposed framewor
    

