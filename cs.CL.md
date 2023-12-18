# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knowledge Editing for Large Language Models: A Survey.](http://arxiv.org/abs/2310.16218) | 大型语言模型(LLMs)在学术和工业领域具有巨大潜力。本文综述了LLMs的知识编辑问题，强调了需要开发有效和高效的技术来更新预训练LLMs以纳入新知识的重要性。 |
| [^2] | [SteloCoder: a Decoder-Only LLM for Multi-Language to Python Code Translation.](http://arxiv.org/abs/2310.15539) | SteloCoder是一个仅解码的基于StarCoder的LLM，在多语言到Python代码翻译中取得了显著的性能提升。它采用Mixture-of-Experts（MoE）技术和门控网络，通过对StarCoder进行微调获得专家，并使用低秩自适应方法（LoRA）技术来限制每个专家的大小。 |
| [^3] | [SeqXGPT: Sentence-Level AI-Generated Text Detection.](http://arxiv.org/abs/2310.08903) | 本文介绍了SeqXGPT，这是一种句子级AI生成文本检测方法。通过利用白盒LLMs的对数概率列表作为特征，SeqXGPT在句子级别的AIGT检测中取得了良好的效果。 |
| [^4] | [Dementia Assessment Using Mandarin Speech with an Attention-based Speech Recognition Encoder.](http://arxiv.org/abs/2310.03985) | 本文提出了一个基于注意力的语音识别模型，用于构建一个普通话言语痴呆评估系统。通过训练模型并提取编码器，实现了在阿尔茨海默病检测和临床痴呆评分预测方面的显著提升。 |
| [^5] | [C-Pack: Packaged Resources To Advance General Chinese Embedding.](http://arxiv.org/abs/2309.07597) | C-Pack是一套推进普通汉语嵌入领域的资源，包括全面汉语文本嵌入基准、大规模文本嵌入数据集和涵盖多个尺寸的嵌入模型系列。该资源集在C-MTEB基准上实现了最高+10%的表现，并通过整合和优化一套训练方法进一步提升了效果。此外，C-Pack还发布了英语文本嵌入数据和模型，实现了最先进的性能。该资源集可公开获取。 |
| [^6] | [Say Goodbye to RNN-T Loss: A Novel CIF-based Transducer Architecture for Automatic Speech Recognition.](http://arxiv.org/abs/2307.14132) | 本文提出了一种名为CIF-Transducer的新型模型，将连续积分和火机制与RNN-T模型结合起来，实现了高效的对齐，并放弃了RNN-T Loss，从而减少了计算量，并使预测网络发挥更重要的作用。实验证明CIF-T在自动语音识别中取得了最先进的结果。 |
| [^7] | [Personality testing of GPT-3: Limited temporal reliability, but highlighted social desirability of GPT-3's personality instruments results.](http://arxiv.org/abs/2306.04308) | 本研究探讨了GPT-3 Davinci-003聊天机器人的人格特质，发现其具有良好的社交渴望和亲社会特质，但在不同时间的一致性存在限制。 |
| [^8] | [LLMs Can Understand Encrypted Prompt: Towards Privacy-Computing Friendly Transformers.](http://arxiv.org/abs/2305.18396) | 本文中，研究人员通过使用隐私计算友好的近似方法替换transformer架构中计算和通信密集的运算符，实现了大幅降低私有推断成本的效果，并在保持准确性的前提下实现了计算加速和通信开销降低。 |

# 详细

[^1]: 大型语言模型的知识编辑：一项综述

    Knowledge Editing for Large Language Models: A Survey. (arXiv:2310.16218v1 [cs.CL])

    [http://arxiv.org/abs/2310.16218](http://arxiv.org/abs/2310.16218)

    大型语言模型(LLMs)在学术和工业领域具有巨大潜力。本文综述了LLMs的知识编辑问题，强调了需要开发有效和高效的技术来更新预训练LLMs以纳入新知识的重要性。

    

    大型语言模型(LLMs)近期以其出色的理解、分析和生成文本的能力，根据其广博的知识和推理能力，改变了学术和工业领域的格局。然而，LLMs的一个主要缺点是它们在预训练时需要大量计算资源，因为其参数数量前所未有。当需要频繁引入新知识到预训练模型中时，这个缺点更加显著。因此，开发有效和高效的技术来更新预训练LLMs是必不可少的。传统方法是通过直接微调将新知识编码到预训练LLMs中。然而，简单地重新训练LLMs可能计算资源密集，并且存在将与模型更新无关的有价值的预训练知识退化的风险。最近，基于知识的模型编辑(KME)引起了越来越多的关注，旨在精确修改LLMs以纳入特定的知识。

    Large language models (LLMs) have recently transformed both the academic and industrial landscapes due to their remarkable capacity to understand, analyze, and generate texts based on their vast knowledge and reasoning ability. Nevertheless, one major drawback of LLMs is their substantial computational cost for pre-training due to their unprecedented amounts of parameters. The disadvantage is exacerbated when new knowledge frequently needs to be introduced into the pre-trained model. Therefore, it is imperative to develop effective and efficient techniques to update pre-trained LLMs. Traditional methods encode new knowledge in pre-trained LLMs through direct fine-tuning. However, naively re-training LLMs can be computationally intensive and risks degenerating valuable pre-trained knowledge irrelevant to the update in the model. Recently, Knowledge-based Model Editing (KME) has attracted increasing attention, which aims to precisely modify the LLMs to incorporate specific knowledge, wit
    
[^2]: SteloCoder:一种仅解码的用于多语言到Python代码翻译的LLM

    SteloCoder: a Decoder-Only LLM for Multi-Language to Python Code Translation. (arXiv:2310.15539v1 [cs.CL])

    [http://arxiv.org/abs/2310.15539](http://arxiv.org/abs/2310.15539)

    SteloCoder是一个仅解码的基于StarCoder的LLM，在多语言到Python代码翻译中取得了显著的性能提升。它采用Mixture-of-Experts（MoE）技术和门控网络，通过对StarCoder进行微调获得专家，并使用低秩自适应方法（LoRA）技术来限制每个专家的大小。

    

    最近关注大规模语言模型（LLM），StarCoder和Code Llama分别展示了在代码生成方面的出色性能。然而，在代码翻译功能上仍然需要改进和有效训练技术。为了解决这个问题，我们介绍了SteloCoder，一种仅解码的基于StarCoder的LLM，专为多编程语言到Python代码翻译而设计。具体而言，SteloCoder实现了C ++，C＃，JavaScript，Java或PHP到Python代码翻译，而无需指定输入编程语言。我们通过引入专家组混合（Mixture-of-Experts，MoE）技术和一个控制多任务的门控网络来修改StarCoder模型架构。我们通过对StarCoder进行微调来获得专家。具体而言，我们使用了低秩自适应方法（Low-Rank Adaptive Method，LoRA）技术，将每个专家的大小限制为StarCoder参数数量的仅0.06％。同时，为了增强tr

    With the recent focus on Large Language Models (LLMs), both StarCoder (Li et al., 2023) and Code Llama (Rozi\`ere et al., 2023) have demonstrated remarkable performance in code generation. However, there is still a need for improvement in code translation functionality with efficient training techniques. In response to this, we introduce SteloCoder, a decoder-only StarCoder-based LLM designed specifically for multi-programming language-to-Python code translation. In particular, SteloCoder achieves C++, C#, JavaScript, Java, or PHP-to-Python code translation without specifying the input programming language. We modified StarCoder model architecture by incorporating a Mixture-of-Experts (MoE) technique featuring five experts and a gating network for multi-task handling. Experts are obtained by StarCoder fine-tuning. Specifically, we use a Low-Rank Adaptive Method (LoRA) technique, limiting each expert size as only 0.06% of number of StarCoder's parameters. At the same time, to enhance tr
    
[^3]: SeqXGPT: 句子级AI生成文本检测

    SeqXGPT: Sentence-Level AI-Generated Text Detection. (arXiv:2310.08903v1 [cs.CL])

    [http://arxiv.org/abs/2310.08903](http://arxiv.org/abs/2310.08903)

    本文介绍了SeqXGPT，这是一种句子级AI生成文本检测方法。通过利用白盒LLMs的对数概率列表作为特征，SeqXGPT在句子级别的AIGT检测中取得了良好的效果。

    

    广泛应用的大型语言模型(LLMs)可以生成类似人类的内容，引发了对LLMs滥用的担忧。因此，建立强大的AI生成文本（AIGT）检测器非常重要。目前的工作只考虑文档级别的AIGT检测，因此在本文中，我们首先通过合成一个数据集，该数据集包含由LLMs修改过的句子和由人类编写的句子，引入了一个句子级别的检测挑战。然后，我们提出了SeqXGPT，一种利用白盒LLMs的对数概率列表作为句子级AIGT检测特征的新方法。这些特征类似于语音处理中的“波浪”，LLMs无法研究其组成。因此，我们基于卷积和自注意力网络构建了SeqXGPT。我们在句子和文档级别的检测挑战中进行了测试。实验结果显示之前的方法在句子级别的检测中存在困难。

    Widely applied large language models (LLMs) can generate human-like content, raising concerns about the abuse of LLMs. Therefore, it is important to build strong AI-generated text (AIGT) detectors. Current works only consider document-level AIGT detection, therefore, in this paper, we first introduce a sentence-level detection challenge by synthesizing a dataset that contains documents that are polished with LLMs, that is, the documents contain sentences written by humans and sentences modified by LLMs. Then we propose \textbf{Seq}uence \textbf{X} (Check) \textbf{GPT}, a novel method that utilizes log probability lists from white-box LLMs as features for sentence-level AIGT detection. These features are composed like \textit{waves} in speech processing and cannot be studied by LLMs. Therefore, we build SeqXGPT based on convolution and self-attention networks. We test it in both sentence and document-level detection challenges. Experimental results show that previous methods struggle in
    
[^4]: 使用基于注意力的语音识别编码器进行普通话言语的痴呆评估

    Dementia Assessment Using Mandarin Speech with an Attention-based Speech Recognition Encoder. (arXiv:2310.03985v1 [cs.CL])

    [http://arxiv.org/abs/2310.03985](http://arxiv.org/abs/2310.03985)

    本文提出了一个基于注意力的语音识别模型，用于构建一个普通话言语痴呆评估系统。通过训练模型并提取编码器，实现了在阿尔茨海默病检测和临床痴呆评分预测方面的显著提升。

    

    痴呆诊断需要一系列不同的测试方法，这是复杂且耗时的。痴呆的早期检测非常重要，因为它可以防止病情进一步恶化。本文利用语音识别模型构建了一个针对普通话使用者在图片描述任务中的痴呆评估系统。通过在与真实世界情境非常相似的语音数据上训练基于注意力的语音识别模型，我们显著提高了模型的识别能力。随后，我们从语音识别模型中提取编码器，并添加了一个线性层用于痴呆评估。我们收集了来自99名被试的普通话语音数据，并从当地医院获取了他们的临床评估数据。在阿尔茨海默病检测中，我们实现了92.04%的准确性，并在临床痴呆评分预测中达到了9%的平均绝对误差。

    Dementia diagnosis requires a series of different testing methods, which is complex and time-consuming. Early detection of dementia is crucial as it can prevent further deterioration of the condition. This paper utilizes a speech recognition model to construct a dementia assessment system tailored for Mandarin speakers during the picture description task. By training an attention-based speech recognition model on voice data closely resembling real-world scenarios, we have significantly enhanced the model's recognition capabilities. Subsequently, we extracted the encoder from the speech recognition model and added a linear layer for dementia assessment. We collected Mandarin speech data from 99 subjects and acquired their clinical assessments from a local hospital. We achieved an accuracy of 92.04% in Alzheimer's disease detection and a mean absolute error of 9% in clinical dementia rating score prediction.
    
[^5]: C-Pack: 推进普通汉语嵌入的打包资源

    C-Pack: Packaged Resources To Advance General Chinese Embedding. (arXiv:2309.07597v1 [cs.CL])

    [http://arxiv.org/abs/2309.07597](http://arxiv.org/abs/2309.07597)

    C-Pack是一套推进普通汉语嵌入领域的资源，包括全面汉语文本嵌入基准、大规模文本嵌入数据集和涵盖多个尺寸的嵌入模型系列。该资源集在C-MTEB基准上实现了最高+10%的表现，并通过整合和优化一套训练方法进一步提升了效果。此外，C-Pack还发布了英语文本嵌入数据和模型，实现了最先进的性能。该资源集可公开获取。

    

    我们介绍了C-Pack，这是一套显著推进普通汉语嵌入领域的资源。C-Pack包括三个关键资源。1）C-MTEB是一个涵盖6个任务和35个数据集的全面汉语文本嵌入基准。2）C-MTP是一个从标记和未标记的汉语语料库中策划的大规模文本嵌入数据集，用于训练嵌入模型。3）C-TEM是一个涵盖多个尺寸的嵌入模型系列。我们的模型在C-MTEB上的表现优于之前的所有汉语文本嵌入达到了发布时的最高+10%。我们还整合和优化了C-TEM的整套训练方法。除了我们关于普通汉语嵌入的资源外，我们还发布了我们的英语文本嵌入数据和模型。这些英语模型在MTEB基准上实现了最先进的性能；与此同时，我们发布的英语数据比汉语数据大2倍。所有这些资源都可以在https://github.com/FlagOpen/FlagEmbedding上公开获取。

    We introduce C-Pack, a package of resources that significantly advance the field of general Chinese embeddings. C-Pack includes three critical resources. 1) C-MTEB is a comprehensive benchmark for Chinese text embeddings covering 6 tasks and 35 datasets. 2) C-MTP is a massive text embedding dataset curated from labeled and unlabeled Chinese corpora for training embedding models. 3) C-TEM is a family of embedding models covering multiple sizes. Our models outperform all prior Chinese text embeddings on C-MTEB by up to +10% upon the time of the release. We also integrate and optimize the entire suite of training methods for C-TEM. Along with our resources on general Chinese embedding, we release our data and models for English text embeddings. The English models achieve state-of-the-art performance on MTEB benchmark; meanwhile, our released English data is 2 times larger than the Chinese data. All these resources are made publicly available at https://github.com/FlagOpen/FlagEmbedding.
    
[^6]: 告别RNN-T Loss：一种新的基于CIF的转录器架构用于自动语音识别

    Say Goodbye to RNN-T Loss: A Novel CIF-based Transducer Architecture for Automatic Speech Recognition. (arXiv:2307.14132v1 [cs.SD])

    [http://arxiv.org/abs/2307.14132](http://arxiv.org/abs/2307.14132)

    本文提出了一种名为CIF-Transducer的新型模型，将连续积分和火机制与RNN-T模型结合起来，实现了高效的对齐，并放弃了RNN-T Loss，从而减少了计算量，并使预测网络发挥更重要的作用。实验证明CIF-T在自动语音识别中取得了最先进的结果。

    

    RNN-T模型在ASR中广泛使用，依靠RNN-T Loss实现输入音频和目标序列的长度对齐。然而，RNN-T Loss的实现复杂性和基于对齐的优化目标导致计算冗余和预测网络角色的减少。在本文中，我们提出了一种名为CIF-Transducer（CIF-T）的新型模型，它将连续积分和火（CIF）机制与RNN-T模型结合起来，实现高效的对齐。通过这种方式，放弃了RNN-T Loss，从而减少了计算量，并使预测网络发挥更重要的作用。我们还引入了Funnel-CIF、Context Blocks、Unified Gating和Bilinear Pooling联合网络以及辅助训练策略来进一步提高性能。在178小时的AISHELL-1和10000小时的WenetSpeech数据集上的实验证明，与RNN-T模型相比，CIF-T以更低的计算开销实现了最先进的结果。

    RNN-T models are widely used in ASR, which rely on the RNN-T loss to achieve length alignment between input audio and target sequence. However, the implementation complexity and the alignment-based optimization target of RNN-T loss lead to computational redundancy and a reduced role for predictor network, respectively. In this paper, we propose a novel model named CIF-Transducer (CIF-T) which incorporates the Continuous Integrate-and-Fire (CIF) mechanism with the RNN-T model to achieve efficient alignment. In this way, the RNN-T loss is abandoned, thus bringing a computational reduction and allowing the predictor network a more significant role. We also introduce Funnel-CIF, Context Blocks, Unified Gating and Bilinear Pooling joint network, and auxiliary training strategy to further improve performance. Experiments on the 178-hour AISHELL-1 and 10000-hour WenetSpeech datasets show that CIF-T achieves state-of-the-art results with lower computational overhead compared to RNN-T models.
    
[^7]: GPT-3的人格测试：时间可靠性有限，但凸显了社交渴望的人格工具结果。

    Personality testing of GPT-3: Limited temporal reliability, but highlighted social desirability of GPT-3's personality instruments results. (arXiv:2306.04308v1 [cs.AI])

    [http://arxiv.org/abs/2306.04308](http://arxiv.org/abs/2306.04308)

    本研究探讨了GPT-3 Davinci-003聊天机器人的人格特质，发现其具有良好的社交渴望和亲社会特质，但在不同时间的一致性存在限制。

    

    为了评估聊天机器人GPT-3 Davinci-003的潜在应用和限制，本研究探讨了应用于聊天机器人及其个性化资料的人格问卷的时间可靠性。在两个不同的场合，心理问卷被应用于聊天机器人，然后将回答与人类基准数据进行比较。研究结果显示，聊天机器人的回答有不同程度的一致性，有些量表表现出良好的一致性，而有些则表现出较差的一致性。总体而言，Davinci-003显示出一个社交渴望和亲社会的人格特质，尤其是在亲和力领域。然而，聊天机器人回答的基础，无论是由主观自我反思还是预定算法驱动，尚不确定。

    To assess the potential applications and limitations of chatbot GPT-3 Davinci-003, this study explored the temporal reliability of personality questionnaires applied to the chatbot and its personality profile. Psychological questionnaires were administered to the chatbot on two separate occasions, followed by a comparison of the responses to human normative data. The findings revealed varying levels of agreement in the chatbot's responses over time, with some scales displaying excellent while others demonstrated poor agreement. Overall, Davinci-003 displayed a socially desirable and pro-social personality profile, particularly in the domain of communion. However, the underlying basis of the chatbot's responses, whether driven by conscious self-reflection or predetermined algorithms, remains uncertain.
    
[^8]: LLM可以理解加密提示：面向隐私计算友好的Transformers

    LLMs Can Understand Encrypted Prompt: Towards Privacy-Computing Friendly Transformers. (arXiv:2305.18396v1 [cs.LG])

    [http://arxiv.org/abs/2305.18396](http://arxiv.org/abs/2305.18396)

    本文中，研究人员通过使用隐私计算友好的近似方法替换transformer架构中计算和通信密集的运算符，实现了大幅降低私有推断成本的效果，并在保持准确性的前提下实现了计算加速和通信开销降低。

    

    先前的研究尝试在服务器客户端环境中为基于transformer的大型语言模型 (LLMs) 构建私有推断框架，其中服务器持有模型参数，客户端输入私有数据进行推断。然而，当私有输入通过原始LLMs进行前向传播时，这些框架会产生显着的开销。在本文中，我们展示了通过用隐私计算友好的近似替换transformer架构中计算和通信密集的运算符可以大大降低私有推断成本，对模型性能的影响微乎其微。与最新的Iron（NeurIPS 2022）相比，我们的隐私计算友好的模型推断管道在计算上实现了$5 \times$的加速，在通信开销上实现了80\%的降低，同时几乎保持了相同的准确性。

    Prior works have attempted to build private inference frameworks for transformer-based large language models (LLMs) in a server-client setting, where the server holds the model parameters and the client inputs the private data for inference. However, these frameworks impose significant overhead when the private inputs are forward propagated through the original LLMs. In this paper, we show that substituting the computation- and communication-heavy operators in the transformer architecture with privacy-computing friendly approximations can greatly reduce the private inference costs with minor impact on model performance. Compared to the state-of-the-art Iron (NeurIPS 2022), our privacy-computing friendly model inference pipeline achieves a $5\times$ acceleration in computation and an 80\% reduction in communication overhead, while retaining nearly identical accuracy.
    

