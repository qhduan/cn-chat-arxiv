# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Defending Our Privacy With Backdoors.](http://arxiv.org/abs/2310.08320) | 本研究提出了一种基于后门攻击的防御方法，通过对模型进行策略性插入后门，对齐敏感短语与中性术语的嵌入，以删除训练数据中的私人信息。实证结果显示该方法的有效性。 |
| [^2] | [Norm Tweaking: High-performance Low-bit Quantization of Large Language Models.](http://arxiv.org/abs/2309.02784) | 本文介绍了一种称为“norm tweaking”的技术，通过调整量化的激活分布来实现高精度的低比特量化，以提高大型语言模型的压缩性能。 |
| [^3] | [PMET: Precise Model Editing in a Transformer.](http://arxiv.org/abs/2308.08742) | 该论文通过分析Transformer模型中的隐藏状态，发现多头自注意力编码了某些通用知识提取模式，因此在进行模型编辑时，不需要更新多头自注意力的权重。 |
| [^4] | [AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation.](http://arxiv.org/abs/2305.09515) | 本文提出了一种自回归扩散模型（AR-Diffusion）用于文本生成，通过动态数量的降噪步骤，确保左侧标记的生成影响右侧标记的生成。 |
| [^5] | [StarCoder: may the source be with you!.](http://arxiv.org/abs/2305.06161) | 本研究介绍了一个具有15.5B参数和8K上下文长度的大型语言模型——StarCoder，其可以进行快速大批量推理。经评估证明，在Python上表现优异，能够通过人工评估获得40\%的pass@1的得分，且在其他程序中也表现出令人满意的性能。 |
| [^6] | [Structure-CLIP: Enhance Multi-modal Language Representations with Structure Knowledge.](http://arxiv.org/abs/2305.06152) | Structure-CLIP使用文本中的结构化知识，使用场景图强化多模态语言表示，从而在图像-文本匹配任务中展现了更好的性能。 |
| [^7] | [Learning to Program with Natural Language.](http://arxiv.org/abs/2304.10464) | 该论文提出了一种用自然语言作为编程语言并通过学习编程方法让大语言模型直接生成自然语言程序并指导推理的方法。实验结果表明，这种方法在解决编程任务上比基线方法有更高的成功率。 |
| [^8] | [USNID: A Framework for Unsupervised and Semi-supervised New Intent Discovery.](http://arxiv.org/abs/2304.07699) | 该论文提出了一个名为USNID的框架，用于无监督和半监督的新意图发现，解决了利用有限或无标记数据时难以捕捉复杂语义的问题，并设计了聚类机制来提高自我监督目标的质量，从而发现细粒度的意图簇。 |
| [^9] | [Evaluation of GPT and BERT-based models on identifying protein-protein interactions in biomedical text.](http://arxiv.org/abs/2303.17728) | 该论文评估了预先训练的语言模型(GPT和BERT)识别生物医学文本中蛋白质相互作用的性能, 结果显示BERT模型表现最佳，其中PubMedBERT具有最高的精度和F1分数，BioM-ALBERT具有最高的召回率。 |

# 详细

[^1]: 使用后门技术保护我们的隐私

    Defending Our Privacy With Backdoors. (arXiv:2310.08320v1 [cs.LG])

    [http://arxiv.org/abs/2310.08320](http://arxiv.org/abs/2310.08320)

    本研究提出了一种基于后门攻击的防御方法，通过对模型进行策略性插入后门，对齐敏感短语与中性术语的嵌入，以删除训练数据中的私人信息。实证结果显示该方法的有效性。

    

    在使用未经筛选、常常包含敏感信息的网页数据训练大型人工智能模型的情况下，隐私问题成为了一个重要的关注点。其中一个问题是，攻击者可以利用隐私攻击的方法提取出训练数据的信息。然而，如何在不降低模型性能的情况下去除特定信息是一个不容易解决且具有挑战性的问题。我们提出了一个基于后门攻击的简单而有效的防御方法，用于从模型中删除私人信息，如个人姓名，特别是针对文本编码器的。具体而言，通过策略性地插入后门，我们将敏感短语的嵌入与中性术语的嵌入对齐，例如用"a person"代替人名。我们的实证结果通过对零样本分类器使用专门的隐私攻击测试表明了我们基于后门的防御方法的效果。我们的方法提供了一个新的"双重用途"的视角。

    The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names of individuals from models, and focus in this work on text encoders. Specifically, through strategic insertion of backdoors, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's name. Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides not only a new "dual-use" perspecti
    
[^2]: Norm调整：大型语言模型的高性能低比特量化

    Norm Tweaking: High-performance Low-bit Quantization of Large Language Models. (arXiv:2309.02784v1 [cs.LG])

    [http://arxiv.org/abs/2309.02784](http://arxiv.org/abs/2309.02784)

    本文介绍了一种称为“norm tweaking”的技术，通过调整量化的激活分布来实现高精度的低比特量化，以提高大型语言模型的压缩性能。

    

    随着大型语言模型（LLMs）的尺寸不断增大，在保持精度的前提下进行模型压缩已成为部署的关键挑战。虽然一些量化方法，如GPTQ，在实现可接受的4比特权重量化方面取得了进展，但尝试更低位的量化往往导致严重的性能降低。在本文中，我们引入了一种称为“norm tweaking”的技术，它可以作为当前PTQ方法的插件，实现高精度和成本高效。我们的方法受到一项观察的启示，即使调整量化的激活分布以与其浮点对应物匹配，也可以恢复LLMs的准确性。为了实现这一点，我们精心设计了一个调整策略，包括生成校准数据和通道距离约束，以更新归一化层的权重以获得更好的泛化性能。我们在各种数据集上进行了大量实验，使用了几个开源的LLMs。

    As the size of large language models (LLMs) continues to grow, model compression without sacrificing accuracy has become a crucial challenge for deployment. While some quantization methods, such as GPTQ, have made progress in achieving acceptable 4-bit weight-only quantization, attempts at lower bit quantization often result in severe performance degradation. In this paper, we introduce a technique called norm tweaking, which can be used as a plugin in current PTQ methods to achieve high precision while being cost-efficient. Our approach is inspired by the observation that rectifying the quantized activation distribution to match its float counterpart can readily restore accuracy for LLMs. To achieve this, we carefully design a tweaking strategy that includes calibration data generation and channel-wise distance constraint to update the weights of normalization layers for better generalization. We conduct extensive experiments on various datasets using several open-sourced LLMs. Our me
    
[^3]: PMET: 在Transformer中的精确模型编辑

    PMET: Precise Model Editing in a Transformer. (arXiv:2308.08742v1 [cs.CL])

    [http://arxiv.org/abs/2308.08742](http://arxiv.org/abs/2308.08742)

    该论文通过分析Transformer模型中的隐藏状态，发现多头自注意力编码了某些通用知识提取模式，因此在进行模型编辑时，不需要更新多头自注意力的权重。

    

    模型编辑技术可以以较低的成本修改大型语言模型中的少量知识，并且已经取得了显著的成功。现有方法假设Transformer层隐藏状态是前馈网络的键值内存的值。它们通常优化Transformer层隐藏状态来记忆目标知识，并将其用于更新大型语言模型中前馈网络的权重。然而，Transformer层隐藏状态的信息流来自三个部分：多头自注意力、前馈网络和残差连接。现有方法忽视了Transformer层隐藏状态包含了前馈网络特别需要的信息这一事实。因此，模型编辑的性能下降。为了实现更精确的模型编辑，我们分析了多头自注意力和前馈网络的隐藏状态，发现多头自注意力编码了某些通用知识提取模式。这意味着当引入新知识时，多头自注意力的权重不需要更新。

    Model editing techniques modify a minor proportion of knowledge in Large Language Models (LLMs) at a relatively low cost, which have demonstrated notable success. Existing methods assume Transformer Layer (TL) hidden states are values of key-value memories of the Feed-Forward Network (FFN). They usually optimize the TL hidden states to memorize target knowledge and use it to update the weights of the FFN in LLMs. However, the information flow of TL hidden states comes from three parts: Multi-Head Self-Attention (MHSA), FFN, and residual connections. Existing methods neglect the fact that the TL hidden states contains information not specifically required for FFN. Consequently, the performance of model editing decreases. To achieve more precise model editing, we analyze hidden states of MHSA and FFN, finding that MHSA encodes certain general knowledge extraction patterns. This implies that MHSA weights do not require updating when new knowledge is introduced. Based on above findings, we
    
[^4]: AR-Diffusion：自回归扩散模型用于文本生成

    AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation. (arXiv:2305.09515v1 [cs.CL])

    [http://arxiv.org/abs/2305.09515](http://arxiv.org/abs/2305.09515)

    本文提出了一种自回归扩散模型（AR-Diffusion）用于文本生成，通过动态数量的降噪步骤，确保左侧标记的生成影响右侧标记的生成。

    

    扩散模型由于其出色的性能，在图像生成领域引起了广泛的关注。最近，这种成功已经扩展到了通过同时生成序列中的所有标记来实现文本生成。然而，自然语言相对于图像具有更为明显的序列依赖性，现有的大多数语言模型都是使用自左向右的自回归方法进行训练的。为了解决自然语言固有的序列特征，我们引入了自回归扩散（AR-Diffusion）模型。AR-Diffusion确保右侧标记的生成取决于左侧标记的生成，这种机制是通过采用动态数量的降噪步骤来实现的，这些步骤根据标记位置而变化。这导致左侧的标记经历的降噪步骤比右侧的标记少，从而使它们能够更早地生成并随后影响右侧标记的生成。

    Diffusion models have gained significant attention in the realm of image generation due to their exceptional performance. Their success has been recently expanded to text generation via generating all tokens within a sequence concurrently. However, natural language exhibits a far more pronounced sequential dependency in comparison to images, and the majority of existing language models are trained utilizing a left-to-right auto-regressive approach. To account for the inherent sequential characteristic of natural language, we introduce Auto-Regressive Diffusion (AR-Diffusion). AR-Diffusion ensures that the generation of tokens on the right depends on the generated ones on the left, a mechanism achieved through employing a dynamic number of denoising steps that vary based on token position. This results in tokens on the left undergoing fewer denoising steps than those on the right, thereby enabling them to generate earlier and subsequently influence the generation of tokens on the right.
    
[^5]: StarCoder: 源代码与你同在！

    StarCoder: may the source be with you!. (arXiv:2305.06161v1 [cs.CL])

    [http://arxiv.org/abs/2305.06161](http://arxiv.org/abs/2305.06161)

    本研究介绍了一个具有15.5B参数和8K上下文长度的大型语言模型——StarCoder，其可以进行快速大批量推理。经评估证明，在Python上表现优异，能够通过人工评估获得40\%的pass@1的得分，且在其他程序中也表现出令人满意的性能。

    

    BigCode社区是一个开放的科学合作组织，致力于开发代表代码的大型语言模型（Code LLMs）的负责任发展。该文介绍了StarCoder和StarCoderBase，这是具有15.5B参数模型和8K上下文长度、填充能力以及多种查询注意力实现的快速大批量推理的模型。我们对StarCoderBase的1万亿个标记进行 fine-tuning，创建了StarCoder。我们进行了迄今为止最全面的Code LLMs评估，并表明StarCoderBase优于支持多种编程语言的每个开放Code LLM，并与OpenAI code-cushman-001模型相匹配或优于该模型。此外，StarCoder在Python上也表现出优异性能，能够通过人工评估获得40\%的pass@1的得分，并仍然保持其在其他程序中的性能。

    The BigCode community, an open-scientific collaboration working on the responsible development of Large Language Models for Code (Code LLMs), introduces StarCoder and StarCoderBase: 15.5B parameter models with 8K context length, infilling capabilities and fast large-batch inference enabled by multi-query attention. StarCoderBase is trained on 1 trillion tokens sourced from The Stack, a large collection of permissively licensed GitHub repositories with inspection tools and an opt-out process. We fine-tuned StarCoderBase on 35B Python tokens, resulting in the creation of StarCoder. We perform the most comprehensive evaluation of Code LLMs to date and show that StarCoderBase outperforms every open Code LLM that supports multiple programming languages and matches or outperforms the OpenAI code-cushman-001 model. Furthermore, StarCoder outperforms every model that is fine-tuned on Python, can be prompted to achieve 40\% pass@1 on HumanEval, and still retains its performance on other program
    
[^6]: Structure-CLIP: 结合结构知识优化多模态语言表示

    Structure-CLIP: Enhance Multi-modal Language Representations with Structure Knowledge. (arXiv:2305.06152v1 [cs.CL])

    [http://arxiv.org/abs/2305.06152](http://arxiv.org/abs/2305.06152)

    Structure-CLIP使用文本中的结构化知识，使用场景图强化多模态语言表示，从而在图像-文本匹配任务中展现了更好的性能。

    

    大规模的视觉-语言预训练在各种下游任务中展现出了很好的性能，并在多模态理解和生成任务中取得了显著的进展。然而，现有方法在需要对文本进行详细语义理解的图像-文本匹配任务上通常表现较差。尽管已经有一些研究在解决这个问题，但它们没有充分利用句子中存在的结构化知识来增强多模态语言表示，导致性能较差。本文提出了一个端到端的框架Structure-CLIP，该框架结合了从文本中提取的隐式详细语义，以增强精细的语义表示。具体而言，(1)我们使用场景图来更加关注文本中的详细语义学习，并充分探索细粒度语义之间的结构化知识，(2)我们结合场景图的知识强化框架来充分利用这些信息。

    Large-scale vision-language pre-training has shown promising advances on various downstream tasks and achieved significant performance in multi-modal understanding and generation tasks. However, existing methods often perform poorly on image-text matching tasks that require a detailed semantics understanding of the text. Although there have been some works on this problem, they do not sufficiently exploit the structural knowledge present in sentences to enhance multi-modal language representations, which leads to poor performance. In this paper, we present an end-to-end framework Structure-CLIP, which integrates latent detailed semantics from the text to enhance fine-grained semantic representations. Specifically, (1) we use scene graphs in order to pay more attention to the detailed semantic learning in the text and fully explore structured knowledge between fine-grained semantics, and (2) we utilize the knowledge-enhanced framework with the help of the scene graph to make full use of
    
[^7]: 用自然语言学习编程

    Learning to Program with Natural Language. (arXiv:2304.10464v1 [cs.CL])

    [http://arxiv.org/abs/2304.10464](http://arxiv.org/abs/2304.10464)

    该论文提出了一种用自然语言作为编程语言并通过学习编程方法让大语言模型直接生成自然语言程序并指导推理的方法。实验结果表明，这种方法在解决编程任务上比基线方法有更高的成功率。

    

    大语言模型在各种基本自然语言任务中表现出卓越性能，这引起了实现人工通用智能的希望。为了更好地完成复杂任务，我们需要利用大语言模型进行编程，然后按照程序生成特定的解决方案。我们提出使用自然语言作为一种新的编程语言来描述任务过程，使它们易于人类和大语言模型理解。虽然大语言模型能够直接生成自然语言程序，但这些程序可能仍然存在错误或不完整的步骤。因此，我们进一步提出了学习编程（LP）的方法，要求大语言模型从复杂任务的训练数据集中学习自然语言程序，然后使用学习到的程序来指导推理。我们在AMPS（高中数学）和Math（竞赛数学问题）数据集上的实验证明了我们方法的有效性。在测试ChatGP解决编程任务时，LP能够实现80%的成功率，优于基线方法。

    Large Language Models (LLMs) have shown remarkable performance in various basic natural language tasks, which raises hopes for achieving Artificial General Intelligence. To better complete complex tasks, we need LLMs to program for the task and then follow the program to generate a specific solution for the test sample. We propose using natural language as a new programming language to describe task procedures, making them easily understandable to both humans and LLMs. LLMs are capable of directly generating natural language programs, but these programs may still contain factual errors or incomplete steps. Therefore, we further propose the Learning to Program (LP) method to ask LLMs themselves to learn natural language programs from the training dataset of complex tasks and then use the learned program to guide inference. Our experiments on the AMPS (high school math) and Math (competition mathematics problems) datasets demonstrate the effectiveness of our approach. When testing ChatGP
    
[^8]: USNID: 无监督和半监督新意图发现的框架

    USNID: A Framework for Unsupervised and Semi-supervised New Intent Discovery. (arXiv:2304.07699v1 [cs.CL])

    [http://arxiv.org/abs/2304.07699](http://arxiv.org/abs/2304.07699)

    该论文提出了一个名为USNID的框架，用于无监督和半监督的新意图发现，解决了利用有限或无标记数据时难以捕捉复杂语义的问题，并设计了聚类机制来提高自我监督目标的质量，从而发现细粒度的意图簇。

    

    新意图发现对自然语言处理非常有价值，使我们更好地理解用户需求并提供友好的服务。然而，在有限或没有标记数据的情况下，大多数现有方法难以捕捉离散文本表示的复杂语义。为了解决这个问题，我们提出了一种名为USNID的新框架，用于无监督和半监督新意图发现，具有三个关键技术：充分利用无监督或半监督数据挖掘浅层语义相似性关系；设计聚类机制解决簇分配不一致的问题；捕获无监督或半监督数据中的高级语义，通过同时优化聚类和自我监督来发现细粒度的意图簇。

    New intent discovery is of great value to natural language processing, allowing for a better understanding of user needs and providing friendly services. However, most existing methods struggle to capture the complicated semantics of discrete text representations when limited or no prior knowledge of labeled data is available. To tackle this problem, we propose a novel framework called USNID for unsupervised and semi-supervised new intent discovery, which has three key technologies. First, it takes full use of unsupervised or semi-supervised data to mine shallow semantic similarity relations and provide well-initialized representations for clustering. Second, it designs a centroid-guided clustering mechanism to address the issue of cluster allocation inconsistency and provide high-quality self-supervised targets for representation learning. Third, it captures high-level semantics in unsupervised or semi-supervised data to discover fine-grained intent-wise clusters by optimizing both cl
    
[^9]: 基于GPT和BERT的模型在生物医学文本中鉴定蛋白质相互作用的评估

    Evaluation of GPT and BERT-based models on identifying protein-protein interactions in biomedical text. (arXiv:2303.17728v1 [cs.CL])

    [http://arxiv.org/abs/2303.17728](http://arxiv.org/abs/2303.17728)

    该论文评估了预先训练的语言模型(GPT和BERT)识别生物医学文本中蛋白质相互作用的性能, 结果显示BERT模型表现最佳，其中PubMedBERT具有最高的精度和F1分数，BioM-ALBERT具有最高的召回率。

    

    检测蛋白质相互作用(PPIs)对于理解遗传机制、疾病发病机理和药物设计至关重要。然而，随着生物医学文献的快速增长，需要自动化和准确提取PPIs以促进科学知识的发掘。已经预先训练的语言模型，如生成式预训练变压器(GPT)和双向编码器表示变压器(BERT)，在自然语言处理(NLP)任务上表现出有希望的结果。我们使用手动编制的LLL基准语料库评估了各种GPT和BERT模型的PPI识别性能，该语料库包含77个句子中的164个PPIs。BERT模型取得了最佳的性能，其中PubMedBERT具有最高的精度(85.17%)和F1分数(86.47%)，BioM-ALBERT具有最高的召回率(93.83%)。尽管GPT-4没有专门针对生物医学文本进行训练，但其性能可与其他模型相媲美。

    Detecting protein-protein interactions (PPIs) is crucial for understanding genetic mechanisms, disease pathogenesis, and drug design. However, with the fast-paced growth of biomedical literature, there is a growing need for automated and accurate extraction of PPIs to facilitate scientific knowledge discovery. Pre-trained language models, such as generative pre-trained transformer (GPT) and bidirectional encoder representations from transformers (BERT), have shown promising results in natural language processing (NLP) tasks. We evaluated the PPI identification performance of various GPT and BERT models using a manually curated benchmark corpus of 164 PPIs in 77 sentences from learning language in logic (LLL). BERT-based models achieved the best overall performance, with PubMedBERT achieving the highest precision (85.17%) and F1-score (86.47%) and BioM-ALBERT achieving the highest recall (93.83%). Despite not being explicitly trained for biomedical texts, GPT-4 achieved comparable perfo
    

