# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator](https://arxiv.org/abs/2402.17767) | 实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。 |
| [^2] | [Navigating Neural Space: Revisiting Concept Activation Vectors to Overcome Directional Divergence](https://arxiv.org/abs/2202.03482) | 本文重新审视了概念激活向量（CAVs）在建模人类可理解的概念中的应用，并引入了基于模式的CAVs来提供更准确的概念方向。 |
| [^3] | [Diverse Neural Audio Embeddings -- Bringing Features back !.](http://arxiv.org/abs/2309.08751) | 本文通过在音频分类任务中学习多样化的特征表示，包括领域特定的音高、音色和神经表示，以及端到端架构，为学习稳健、多样化的表示铺平了道路，并显著提高了性能。 |
| [^4] | [Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models.](http://arxiv.org/abs/2308.15022) | 递归总结在大型语言模型中实现长期对话记忆，可以提高对话系统在长对话中记忆重要信息的能力。 |
| [^5] | [JEN-1: Text-Guided Universal Music Generation with Omnidirectional Diffusion Models.](http://arxiv.org/abs/2308.04729) | JEN-1是一个高保真度通用音乐生成模型，通过结合自回归和非自回归训练，实现了文本引导的音乐生成、音乐修补和延续等生成任务，在文本音乐对齐和音乐质量方面表现出优越性，同时保持计算效率。 |

# 详细

[^1]: 在现实世界中使用商品移动操作器打开橱柜和抽屉

    Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator

    [https://arxiv.org/abs/2402.17767](https://arxiv.org/abs/2402.17767)

    实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。

    

    在这项工作中，我们构建了一个端到端系统，使商品移动操作器（Stretch RE2）能够在多样的以前未见的真实世界环境中拉开橱柜和抽屉。我们在31个不同的物体和13个不同真实世界环境中进行了4天的实际测试。我们的系统在零击打下，对在未知环境中新颖的橱柜和抽屉的打开率达到61%。对失败模式的分析表明，感知误差是我们系统面临的最重要挑战。

    arXiv:2402.17767v1 Announce Type: cross  Abstract: Pulling open cabinets and drawers presents many difficult technical challenges in perception (inferring articulation parameters for objects from onboard sensors), planning (producing motion plans that conform to tight task constraints), and control (making and maintaining contact while applying forces on the environment). In this work, we build an end-to-end system that enables a commodity mobile manipulator (Stretch RE2) to pull open cabinets and drawers in diverse previously unseen real world environments. We conduct 4 days of real world testing of this system spanning 31 different objects from across 13 different real world environments. Our system achieves a success rate of 61% on opening novel cabinets and drawers in unseen environments zero-shot. An analysis of the failure modes suggests that errors in perception are the most significant challenge for our system. We will open source code and models for others to replicate and bui
    
[^2]: 领航神经空间：重新审视概念激活向量以克服方向差异

    Navigating Neural Space: Revisiting Concept Activation Vectors to Overcome Directional Divergence

    [https://arxiv.org/abs/2202.03482](https://arxiv.org/abs/2202.03482)

    本文重新审视了概念激活向量（CAVs）在建模人类可理解的概念中的应用，并引入了基于模式的CAVs来提供更准确的概念方向。

    

    随着对于理解神经网络预测策略的兴趣日益增长，概念激活向量（CAVs）已成为一种流行的工具，用于在潜在空间中建模人类可理解的概念。通常，CAVs是通过利用线性分类器来计算的，该分类器优化具有给定概念和无给定概念的样本的潜在表示的可分离性。然而，在本文中我们展示了这种以可分离性为导向的计算方法会导致与精确建模概念方向的实际目标发散的解决方案。这种差异可以归因于分散方向的显著影响，即与概念无关的信号，这些信号被线性模型的滤波器（即权重）捕获以优化类别可分性。为了解决这个问题，我们引入基于模式的CAVs，仅关注概念信号，从而提供更准确的概念方向。我们评估了各种CAV方法与真实概念方向的对齐程度。

    With a growing interest in understanding neural network prediction strategies, Concept Activation Vectors (CAVs) have emerged as a popular tool for modeling human-understandable concepts in the latent space. Commonly, CAVs are computed by leveraging linear classifiers optimizing the separability of latent representations of samples with and without a given concept. However, in this paper we show that such a separability-oriented computation leads to solutions, which may diverge from the actual goal of precisely modeling the concept direction. This discrepancy can be attributed to the significant influence of distractor directions, i.e., signals unrelated to the concept, which are picked up by filters (i.e., weights) of linear models to optimize class-separability. To address this, we introduce pattern-based CAVs, solely focussing on concept signals, thereby providing more accurate concept directions. We evaluate various CAV methods in terms of their alignment with the true concept dire
    
[^3]: 多样的神经音频嵌入 - 恢复特征！

    Diverse Neural Audio Embeddings -- Bringing Features back !. (arXiv:2309.08751v1 [cs.SD])

    [http://arxiv.org/abs/2309.08751](http://arxiv.org/abs/2309.08751)

    本文通过在音频分类任务中学习多样化的特征表示，包括领域特定的音高、音色和神经表示，以及端到端架构，为学习稳健、多样化的表示铺平了道路，并显著提高了性能。

    

    随着现代人工智能架构的出现，从端到端的架构开始流行。这种转变导致了神经架构在没有领域特定偏见/知识的情况下进行训练，根据任务进行优化。本文中，我们通过多样的特征表示（在本例中是领域特定的）学习音频嵌入。对于涉及数百种声音分类的情况，我们学习分别针对音高、音色和神经表示等多样的音频属性建立稳健的嵌入，同时也通过端到端架构进行学习。我们观察到手工制作的嵌入，例如基于音高和音色的嵌入，虽然单独使用时无法击败完全端到端的表示，但将这些嵌入与端到端嵌入相结合可以显著提高性能。这项工作将为在端到端模型中引入一些领域专业知识来学习稳健、多样化的表示铺平道路，并超越仅训练端到端模型的性能。

    With the advent of modern AI architectures, a shift has happened towards end-to-end architectures. This pivot has led to neural architectures being trained without domain-specific biases/knowledge, optimized according to the task. We in this paper, learn audio embeddings via diverse feature representations, in this case, domain-specific. For the case of audio classification over hundreds of categories of sound, we learn robust separate embeddings for diverse audio properties such as pitch, timbre, and neural representation, along with also learning it via an end-to-end architecture. We observe handcrafted embeddings, e.g., pitch and timbre-based, although on their own, are not able to beat a fully end-to-end representation, yet adding these together with end-to-end embedding helps us, significantly improve performance. This work would pave the way to bring some domain expertise with end-to-end models to learn robust, diverse representations, surpassing the performance of just training 
    
[^4]: 递归总结在大型语言模型中实现长期对话记忆

    Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models. (arXiv:2308.15022v1 [cs.CL])

    [http://arxiv.org/abs/2308.15022](http://arxiv.org/abs/2308.15022)

    递归总结在大型语言模型中实现长期对话记忆，可以提高对话系统在长对话中记忆重要信息的能力。

    

    大多数开放领域的对话系统在长期对话中容易遗忘重要信息。现有方法通常训练特定的检索器或总结器从过去获取关键信息，这需要耗费时间且高度依赖标记数据的质量。为了缓解这个问题，我们提出使用大型语言模型（LLMs）递归生成总结/记忆，以增强长期记忆能力。具体而言，我们的方法首先刺激LLMs记住小对话上下文，然后递归地使用之前的记忆和随后的对话内容产生新的记忆。最后，LLM可以在最新记忆的帮助下轻松生成高度一致的响应。我们使用ChatGPT和text-davinci-003进行评估，对广泛使用的公共数据集进行的实验证明我们的方法在长对话中可以生成更一致的响应。值得注意的是，我们的方法是实现LLM建模的潜在解决方案。

    Most open-domain dialogue systems suffer from forgetting important information, especially in a long-term conversation. Existing works usually train the specific retriever or summarizer to obtain key information from the past, which is time-consuming and highly depends on the quality of labeled data. To alleviate this problem, we propose to recursively generate summaries/ memory using large language models (LLMs) to enhance long-term memory ability. Specifically, our method first stimulates LLMs to memorize small dialogue contexts and then recursively produce new memory using previous memory and following contexts. Finally, the LLM can easily generate a highly consistent response with the help of the latest memory. We evaluate our method using ChatGPT and text-davinci-003, and the experiments on the widely-used public dataset show that our method can generate more consistent responses in a long-context conversation. Notably, our method is a potential solution to enable the LLM to model
    
[^5]: JEN-1：具有全向扩散模型的文本引导通用音乐生成

    JEN-1: Text-Guided Universal Music Generation with Omnidirectional Diffusion Models. (arXiv:2308.04729v1 [cs.SD])

    [http://arxiv.org/abs/2308.04729](http://arxiv.org/abs/2308.04729)

    JEN-1是一个高保真度通用音乐生成模型，通过结合自回归和非自回归训练，实现了文本引导的音乐生成、音乐修补和延续等生成任务，在文本音乐对齐和音乐质量方面表现出优越性，同时保持计算效率。

    

    随着深度生成模型的进步，音乐生成引起了越来越多的关注。然而，基于文本描述生成音乐（即文本到音乐）仍然具有挑战性，原因是音乐结构的复杂性和高采样率的要求。尽管任务的重要性，当前的生成模型在音乐质量、计算效率和泛化能力方面存在局限性。本文介绍了JEN-1，这是一个用于文本到音乐生成的通用高保真模型。JEN-1是一个结合了自回归和非自回归训练的扩散模型。通过上下文学习，JEN-1可以执行各种生成任务，包括文本引导的音乐生成、音乐修补以及延续。评估结果表明，JEN-1在文本音乐对齐和音乐质量方面表现出优越性，同时保持计算效率。我们的演示可在此网址获取：http://URL

    Music generation has attracted growing interest with the advancement of deep generative models. However, generating music conditioned on textual descriptions, known as text-to-music, remains challenging due to the complexity of musical structures and high sampling rate requirements. Despite the task's significance, prevailing generative models exhibit limitations in music quality, computational efficiency, and generalization. This paper introduces JEN-1, a universal high-fidelity model for text-to-music generation. JEN-1 is a diffusion model incorporating both autoregressive and non-autoregressive training. Through in-context learning, JEN-1 performs various generation tasks including text-guided music generation, music inpainting, and continuation. Evaluations demonstrate JEN-1's superior performance over state-of-the-art methods in text-music alignment and music quality while maintaining computational efficiency. Our demos are available at this http URL
    

