# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Do Not Worry if You Do Not Have Data: Building Pretrained Language Models Using Translationese](https://arxiv.org/abs/2403.13638) | 本文探讨了使用Translationese合成数据作为预训练语言模型的实用性，展示了在英语以外的语言中使用机器翻译创建的合成数据进行LMs预训练的有效性，并提出了通过使用轻量级TinyLMs预训练来过滤合成数据的方法。 |
| [^2] | [Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding](https://arxiv.org/abs/2402.12374) | Sequoia是一种可扩展、稳健且硬件感知的推测解码算法，通过引入动态规划算法优化标记树结构、采用新颖的采样和验证方法实现稳健性能以及硬件感知的树优化器最大化推测性能。 |
| [^3] | [End-to-End Evaluation for Low-Latency Simultaneous Speech Translation.](http://arxiv.org/abs/2308.03415) | 本文提出了一个端到端的评估框架，用于评估低延迟语音翻译的各个方面。通过该框架，我们比较了不同方法的性能，并进行了全面的评估。 |
| [^4] | [Relation-Aware Network with Attention-Based Loss for Few-Shot Knowledge Graph Completion.](http://arxiv.org/abs/2306.09519) | 本文提出了一种新颖的RANA框架，利用有策略地选择相关负样本和设计基于注意力机制的损失函数来更好地利用负样本并缓解零损失问题，同时设计了一种动态的关系感知实体编码来捕获不同关系下实体的不同表示。 |

# 详细

[^1]: 不必担心如果您没有数据：利用Translationese构建预训练语言模型

    Do Not Worry if You Do Not Have Data: Building Pretrained Language Models Using Translationese

    [https://arxiv.org/abs/2403.13638](https://arxiv.org/abs/2403.13638)

    本文探讨了使用Translationese合成数据作为预训练语言模型的实用性，展示了在英语以外的语言中使用机器翻译创建的合成数据进行LMs预训练的有效性，并提出了通过使用轻量级TinyLMs预训练来过滤合成数据的方法。

    

    在本文中，我们探讨了将机器翻译创建的合成数据Translationese用作预训练语言模型（LMs）的实用性。预训练需要大量的单语数据，对于英语以外的语言，这些数据大部分是不可用的。近年来，人们越来越关注使用合成数据来解决这种数据稀缺性问题。我们以英语和Indic语言为例，将网络抓取的单语文档（干净的）翻译成目标语言。然后，我们在这些Translationese数据（合成数据）上训练包含28M和85M参数的语言模型。我们展示了它们在下游自然语言理解和生成任务中的性能与在干净数据上预训练的LMs相比，NLU任务的性能仅差3.56％，NLG任务的差异为1.51％。此外，我们提出了使用在干净数据上预训练的轻量级TinyLMs来高效过滤合成数据的方法，这显著提高了性能。

    arXiv:2403.13638v1 Announce Type: new  Abstract: In this paper, we explore the utility of \textit{Translationese} as synthetic data created using machine translation for pre-training language models (LMs). Pre-training requires vast amounts of monolingual data, which is mostly unavailable for languages other than English. Recently, there has been a growing interest in using synthetic data to address this data scarcity. We take the case of English and Indic languages and translate web-crawled monolingual documents (clean) into the target language. Then, we train language models containing 28M and 85M parameters on this translationese data (synthetic). We show that their performance on downstream natural language understanding and generative tasks is only 3.56\% poorer on NLU tasks and 1.51\% on NLG tasks than LMs pre-trained on clean data. Further, we propose the use of lightweight \textit{TinyLMs} pre-trained on clean data to filter synthetic data efficiently which significantly improv
    
[^2]: Sequoia: 可扩展、稳健且硬件感知的推测解码

    Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding

    [https://arxiv.org/abs/2402.12374](https://arxiv.org/abs/2402.12374)

    Sequoia是一种可扩展、稳健且硬件感知的推测解码算法，通过引入动态规划算法优化标记树结构、采用新颖的采样和验证方法实现稳健性能以及硬件感知的树优化器最大化推测性能。

    

    随着大型语言模型（LLMs）的使用增多，使用这些模型进行高效推理变得日益重要。虽然最近推测解码已经成为加速推理的一个有前途的方向，但现有方法在扩展到较大的推测预算、适应不同超参数和硬件方面存在局限性。本文介绍了Sequoia，一个可扩展、稳健且硬件感知的用于推测解码的算法。为了实现更好的可扩展性，Sequoia引入了一个动态规划算法来找到用于被推测标记的最佳树结构。为了实现稳健的推测性能，Sequoia使用了一种新颖的采样和验证方法，该方法在不同解码温度下优于先前的方法。最后，Sequoia引入了一种硬件感知的树优化器，通过自动选择给定情况下的标记树大小和深度来最大化推测性能。

    arXiv:2402.12374v1 Announce Type: new  Abstract: As the usage of large language models (LLMs) grows, performing efficient inference with these models becomes increasingly important. While speculative decoding has recently emerged as a promising direction for speeding up inference, existing methods are limited in their ability to scale to larger speculation budgets, and adapt to different hyperparameters and hardware. This paper introduces Sequoia, a scalable, robust, and hardware-aware algorithm for speculative decoding. To attain better scalability, Sequoia introduces a dynamic programming algorithm to find the optimal tree structure for the speculated tokens. To achieve robust speculative performance, Sequoia uses a novel sampling and verification method that outperforms prior work across different decoding temperatures. Finally, Sequoia introduces a hardware-aware tree optimizer that maximizes speculative performance by automatically selecting the token tree size and depth for a giv
    
[^3]: 低延迟同时语音翻译的端到端评估

    End-to-End Evaluation for Low-Latency Simultaneous Speech Translation. (arXiv:2308.03415v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.03415](http://arxiv.org/abs/2308.03415)

    本文提出了一个端到端的评估框架，用于评估低延迟语音翻译的各个方面。通过该框架，我们比较了不同方法的性能，并进行了全面的评估。

    

    近年来，低延迟语音翻译的挑战引起了研究界的广泛关注，许多出版物和共享任务也证明了这一点。因此，在实际场景中评估这些不同的方法非常重要。然而，目前只有系统的特定方面被评估，并且往往无法比较不同的方法。在这项工作中，我们提出了第一个在实际条件下执行和评估低延迟语音翻译各个方面的框架。评估是以端到端的方式进行的，包括音频的分段以及不同组成部分的运行时间。其次，我们使用该框架比较了不同的低延迟语音翻译方法。我们评估了具有修订输出选项的模型以及具有固定输出方法。此外，我们直接比较了最先进的级联系统和端到端系统。最后，该框架基于一个统一的度量来评估低延迟语音翻译性能，并提供了一个全面的评估结果。

    The challenge of low-latency speech translation has recently draw significant interest in the research community as shown by several publications and shared tasks. Therefore, it is essential to evaluate these different approaches in realistic scenarios. However, currently only specific aspects of the systems are evaluated and often it is not possible to compare different approaches.  In this work, we propose the first framework to perform and evaluate the various aspects of low-latency speech translation under realistic conditions. The evaluation is carried out in an end-to-end fashion. This includes the segmentation of the audio as well as the run-time of the different components.  Secondly, we compare different approaches to low-latency speech translation using this framework. We evaluate models with the option to revise the output as well as methods with fixed output. Furthermore, we directly compare state-of-the-art cascaded as well as end-to-end systems. Finally, the framework all
    
[^4]: 关系感知网络基于注意力损失的小样本知识图谱补全

    Relation-Aware Network with Attention-Based Loss for Few-Shot Knowledge Graph Completion. (arXiv:2306.09519v1 [cs.CL])

    [http://arxiv.org/abs/2306.09519](http://arxiv.org/abs/2306.09519)

    本文提出了一种新颖的RANA框架，利用有策略地选择相关负样本和设计基于注意力机制的损失函数来更好地利用负样本并缓解零损失问题，同时设计了一种动态的关系感知实体编码来捕获不同关系下实体的不同表示。

    

    小样本知识图谱补全旨在利用少量参考实体对预测关系的未见事实。现有方法随机选择一个负采样来最小化基于边界的排名损失，但这容易导致零损失问题。此外，实体在不同的上下文中应该具有不同的表征。为了解决这些问题，我们提出了一种新颖的关系感知网络基于注意力损失的框架。具体而言，我们通过有策略地选择相关负样本和设计基于注意力机制的损失函数来更好地利用丰富的负样本并缓解零损失问题。直觉上，与正样本更相似的负样本将对模型贡献更大。此外，我们设计了一种动态的关系感知实体编码来捕捉不同关系下实体的不同表示。三个基准数据集上的实验结果表明，相比最先进的方法，所提出的RANA框架的有效性。

    Few-shot knowledge graph completion (FKGC) task aims to predict unseen facts of a relation with few-shot reference entity pairs. Current approaches randomly select one negative sample for each reference entity pair to minimize a margin-based ranking loss, which easily leads to a zero-loss problem if the negative sample is far away from the positive sample and then out of the margin. Moreover, the entity should have a different representation under a different context. To tackle these issues, we propose a novel Relation-Aware Network with Attention-Based Loss (RANA) framework. Specifically, to better utilize the plentiful negative samples and alleviate the zero-loss issue, we strategically select relevant negative samples and design an attention-based loss function to further differentiate the importance of each negative sample. The intuition is that negative samples more similar to positive samples will contribute more to the model. Further, we design a dynamic relation-aware entity en
    

