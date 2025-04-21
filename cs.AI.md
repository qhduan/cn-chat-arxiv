# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks](https://arxiv.org/abs/2404.02151) | 展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。 |
| [^2] | [Unsupervised LLM Adaptation for Question Answering](https://arxiv.org/abs/2402.12170) | 提出了无监督LLM适应问答任务，通过利用预训练的LLM和目标领域的未标记文档，实现在新领域回答问题的目标。 |
| [^3] | [Exploring Value Biases: How LLMs Deviate Towards the Ideal](https://arxiv.org/abs/2402.11005) | 研究发现大型语言模型（LLMs）在给出响应时存在一个价值偏好的机制，倾向于偏向理想状态，这种偏差会对不同应用场景产生重要影响。 |
| [^4] | [Foundational theories of hesitant fuzzy sets and hesitant fuzzy information systems and their applications for multi-strength intelligent classifiers](https://arxiv.org/abs/2311.04256) | 本文提出了基于犹豫模糊集的多种包含关系定义、犹豫模糊信息系统的基础命题和基于多强度智能分类器的健康状态诊断方法。 |
| [^5] | [Cooperation Is All You Need.](http://arxiv.org/abs/2305.10449) | 引入了一种基于“本地处理器民主”的算法Cooperator，该算法在强化学习中表现比Transformer算法更好。 |
| [^6] | [On the Shift Invariance of Max Pooling Feature Maps in Convolutional Neural Networks.](http://arxiv.org/abs/2209.11740) | 本文研究了卷积神经网络中最大池化特征图的位移不变性问题，并提出了一种近似复数模的条件，实现了位移稳定性。实验证实了理论的有效性。 |

# 详细

[^1]: 用简单自适应攻击越狱功能对齐的LLM

    Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

    [https://arxiv.org/abs/2404.02151](https://arxiv.org/abs/2404.02151)

    展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。

    

    我们展示了即使是最新的安全对齐的LLM也不具有抵抗简单自适应越狱攻击的稳健性。首先，我们展示了如何成功利用对logprobs的访问进行越狱：我们最初设计了一个对抗性提示模板（有时会适应目标LLM），然后我们在后缀上应用随机搜索以最大化目标logprob（例如token“Sure”），可能会进行多次重启。通过这种方式，我们实现了对GPT-3.5/4、Llama-2-Chat-7B/13B/70B、Gemma-7B和针对GCG攻击进行对抗训练的HarmBench上的R2D2等几乎100%的攻击成功率--根据GPT-4的评判。我们还展示了如何通过转移或预填充攻击以100%的成功率对所有不暴露logprobs的Claude模型进行越狱。此外，我们展示了如何在受污染的模型中使用对一组受限制的token执行随机搜索以查找木马字符串的方法--这项任务与许多其他任务共享相同的属性。

    arXiv:2404.02151v1 Announce Type: cross  Abstract: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many s
    
[^2]: 无监督LLM适应问答任务

    Unsupervised LLM Adaptation for Question Answering

    [https://arxiv.org/abs/2402.12170](https://arxiv.org/abs/2402.12170)

    提出了无监督LLM适应问答任务，通过利用预训练的LLM和目标领域的未标记文档，实现在新领域回答问题的目标。

    

    大型语言模型（LLM）通过自监督训练学习大规模训练数据集中的多样化知识。接着通过指导微调，LLM能够返回多样问题的正确信息。然而，将这些预训练的LLM调整到新的目标领域，如不同组织或时期，用于问答任务会产生很高的注释成本。为解决这一挑战，我们提出了一个新颖的任务，即无监督LLM适应问答任务。在这个任务中，我们利用预训练的LLM、一个公开可用的问答数据集（源数据）和目标域的未标记文档。我们的目标是学习LLM，使其能够回答关于目标领域的问题。我们引入了一个合成数据集和两个真实数据集来评估在源数据和目标数据上微调的模型，并揭示了一些有趣的见解；（i）微调模型展示了提供正确答案的能力

    arXiv:2402.12170v1 Announce Type: cross  Abstract: Large language models (LLM) learn diverse knowledge present in the large-scale training dataset via self-supervised training. Followed by instruction-tuning, LLM acquires the ability to return correct information for diverse questions. However, adapting these pre-trained LLMs to new target domains, such as different organizations or periods, for the question-answering (QA) task incurs a substantial annotation cost. To tackle this challenge, we propose a novel task, unsupervised LLM adaptation for question answering. In this task, we leverage a pre-trained LLM, a publicly available QA dataset (source data), and unlabeled documents from the target domain. Our goal is to learn LLM that can answer questions about the target domain. We introduce one synthetic and two real datasets to evaluate models fine-tuned on the source and target data, and reveal intriguing insights; (i) fine-tuned models exhibit the ability to provide correct answers 
    
[^3]: 探究价值偏好：LLMs偏向理想状态的偏差

    Exploring Value Biases: How LLMs Deviate Towards the Ideal

    [https://arxiv.org/abs/2402.11005](https://arxiv.org/abs/2402.11005)

    研究发现大型语言模型（LLMs）在给出响应时存在一个价值偏好的机制，倾向于偏向理想状态，这种偏差会对不同应用场景产生重要影响。

    

    大型语言模型（LLMs）被部署在各种应用中，并且它们的响应对社会产生着越来越大的影响。理解LLMs在给出响应时的非故意机制对于解释它们的性能并辨别它们在现实世界应用中的偏差至关重要。这类似于人类研究中，这种无意识的响应被称为抽样。我们研究了LLMs的这种抽样现象，发现LLMs的抽样倾向于偏爱高价值选项。价值偏好对应于从最可能的响应向LLM中代表的理想价值的转变。实际上，即便是通过上下文提示学习到的新实体，这种效果也能够再现。我们表明这种偏差表现在意想不到的地方，并对选择典型实例等相关应用场景产生影响。结果显示，价值偏好在不同分类的LLMs中都很明显。

    arXiv:2402.11005v1 Announce Type: cross  Abstract: Large-Language-Models (LLMs) are deployed in a wide range of applications, and their response has an increasing social impact. Understanding the non-deliberate(ive) mechanism of LLMs in giving responses is essential in explaining their performance and discerning their biases in real-world applications. This is analogous to human studies, where such inadvertent responses are referred to as sampling. We study this sampling of LLMs in light of value bias and show that the sampling of LLMs tends to favour high-value options. Value bias corresponds to this shift of response from the most likely towards an ideal value represented in the LLM. In fact, this effect can be reproduced even with new entities learnt via in-context prompting. We show that this bias manifests in unexpected places and has implications on relevant application scenarios, like choosing exemplars. The results show that value bias is strong in LLMs across different categor
    
[^4]: 犹豫模糊集及其应用于多强度智能分类器的基础理论

    Foundational theories of hesitant fuzzy sets and hesitant fuzzy information systems and their applications for multi-strength intelligent classifiers

    [https://arxiv.org/abs/2311.04256](https://arxiv.org/abs/2311.04256)

    本文提出了基于犹豫模糊集的多种包含关系定义、犹豫模糊信息系统的基础命题和基于多强度智能分类器的健康状态诊断方法。

    

    犹豫模糊集在某些不确定和犹豫的情况下被广泛使用。在集合中，包含关系是一个重要且基础的定义。因此，作为一种集合，犹豫模糊集需要一个明确的包含关系定义。基于离散形式的犹豫模糊隶属度，本文提出了几种适用于犹豫模糊集的包含关系。随后，介绍了一些犹豫模糊集的基础命题，以及犹豫模糊集族的命题。针对参数减少，提出了犹豫模糊信息系统的一些基础命题，并给出了一个示例和算法来说明参数减少的过程。最后，提出了一种多强度智能分类器，用于对复杂系统进行健康状态诊断。

    arXiv:2311.04256v3 Announce Type: replace  Abstract: Hesitant fuzzy sets are widely used in certain instances of uncertainty and hesitation. In sets, the inclusion relationship is an important and foundational definition. Thus, as a kind of set, hesitant fuzzy sets require an explicit definition of inclusion relationship. Based on the hesitant fuzzy membership degree of discrete form, several kinds of inclusion relationships for hesitant fuzzy sets are proposed in this work. Then, some foundational propositions of hesitant fuzzy sets are presented, along with propositions of families of hesitant fuzzy sets. Some foundational propositions of hesitant fuzzy information systems are proposed with respect to parameter reductions and an example and an algorithm are given to illustrate the processes of parameter reduction. Finally, a multi-strength intelligent classifier is proposed to make health state diagnoses for complex systems.
    
[^5]: 合作是你所需要的。 （arXiv:2305.10449v1 [cs.LG]）

    Cooperation Is All You Need. (arXiv:2305.10449v1 [cs.LG])

    [http://arxiv.org/abs/2305.10449](http://arxiv.org/abs/2305.10449)

    引入了一种基于“本地处理器民主”的算法Cooperator，该算法在强化学习中表现比Transformer算法更好。

    

    在超越“树突民主”之上，我们引入了一个名为Cooperator的“本地处理器民主”。在这里，我们将它们与基于Transformers的机器学习算法（例如ChatGPT）在置换不变神经网络强化学习（RL）中的功能进行比较。 Transformers基于长期以来的“积分-发射”“点”神经元的概念，而Cooperator则受到最近神经生物学突破的启示，这些突破表明，精神生活的细胞基础取决于新皮层中具有两个功能上不同点的上皮神经元。我们表明，当用于RL时，基于Cooperator的算法学习速度比基于Transformer的算法快得多，即使它们具有相同数量的参数。

    Going beyond 'dendritic democracy', we introduce a 'democracy of local processors', termed Cooperator. Here we compare their capabilities when used in permutation-invariant neural networks for reinforcement learning (RL), with machine learning algorithms based on Transformers, such as ChatGPT. Transformers are based on the long-standing conception of integrate-and-fire 'point' neurons, whereas Cooperator is inspired by recent neurobiological breakthroughs suggesting that the cellular foundations of mental life depend on context-sensitive pyramidal neurons in the neocortex which have two functionally distinct points. We show that when used for RL, an algorithm based on Cooperator learns far quicker than that based on Transformer, even while having the same number of parameters.
    
[^6]: 关于卷积神经网络中最大池化特征图的位移不变性

    On the Shift Invariance of Max Pooling Feature Maps in Convolutional Neural Networks. (arXiv:2209.11740v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.11740](http://arxiv.org/abs/2209.11740)

    本文研究了卷积神经网络中最大池化特征图的位移不变性问题，并提出了一种近似复数模的条件，实现了位移稳定性。实验证实了理论的有效性。

    

    本文致力于改善卷积神经网络（CNN）在图像分类领域中的数学可解释性。具体而言，我们解决了在其第一层中出现的不稳定性问题。当在像ImageNet这样的数据集上进行训练时，其第一层往往学习到与方向边通滤波器非常相似的参数。使用这样的Gabor滤波器进行子采样卷积容易出现混叠问题，导致对输入的小偏移敏感。在这个背景下，我们建立了最大池化算子近似复数模的条件，使其几乎具有位移不变性。然后，我们推导了子采样卷积后最大池化的位移稳定性度量。特别地，我们强调了滤波器的频率和方向在实现稳定性方面的关键作用。通过考虑基于双树复小波包变换的确定性特征提取器，即离散Gabor的一种特殊情况，我们通过实验证实了我们的理论。

    This paper focuses on improving the mathematical interpretability of convolutional neural networks (CNNs) in the context of image classification. Specifically, we tackle the instability issue arising in their first layer, which tends to learn parameters that closely resemble oriented band-pass filters when trained on datasets like ImageNet. Subsampled convolutions with such Gabor-like filters are prone to aliasing, causing sensitivity to small input shifts. In this context, we establish conditions under which the max pooling operator approximates a complex modulus, which is nearly shift invariant. We then derive a measure of shift invariance for subsampled convolutions followed by max pooling. In particular, we highlight the crucial role played by the filter's frequency and orientation in achieving stability. We experimentally validate our theory by considering a deterministic feature extractor based on the dual-tree complex wavelet packet transform, a particular case of discrete Gabor
    

