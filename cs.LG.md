# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modeling Latent Selection with Structural Causal Models.](http://arxiv.org/abs/2401.06925) | 本文介绍了一种在结构因果模型中对潜在选择进行建模的方法，并展示了它如何帮助进行因果推理任务，包括处理选择偏差。 |
| [^2] | [FinGPT: Open-Source Financial Large Language Models.](http://arxiv.org/abs/2306.06031) | FinGPT是一个开源的金融大型语言模型，提供了可访问和透明的资源来开发金融LLMs，其重要性在于自动数据筛选管道和轻量级低秩适应技术。 |
| [^3] | [Global universal approximation of functional input maps on weighted spaces.](http://arxiv.org/abs/2306.03303) | 本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。 |
| [^4] | [Beyond Statistical Similarity: Rethinking Metrics for Deep Generative Models in Engineering Design.](http://arxiv.org/abs/2302.02913) | 本文提供了一篇深度学习在工程设计中度量方法的综述和指南。传统的基于似然性的统计度量方法在对工程应用的要求上可能无法充分捕捉，因此本文编辑了一组全面的新度量标准，旨在解决传统度量标准的缺点，并更好地与工程设计的需求相一致。通过案例研究，本文展示了这些度量标准如何应用于评估深度生成模型在工程设计中的性能，并发现这些度量标准在捕捉设计的重要细微差别方面表现优于传统的统计度量标准。 |

# 详细

[^1]: 用结构因果模型对潜在选择进行建模

    Modeling Latent Selection with Structural Causal Models. (arXiv:2401.06925v1 [cs.AI])

    [http://arxiv.org/abs/2401.06925](http://arxiv.org/abs/2401.06925)

    本文介绍了一种在结构因果模型中对潜在选择进行建模的方法，并展示了它如何帮助进行因果推理任务，包括处理选择偏差。

    

    选择偏倚在现实世界的数据中是普遍存在的，如果不正确处理可能导致误导性结果。我们引入了对结构因果模型（SCMs）进行条件操作的方法，以从因果的角度对潜在选择进行建模。我们展示了条件操作将具有明确潜在选择机制的SCM转换为没有此类选择机制的SCM，这在一定程度上编码了根据原始SCM选择的亚总体的因果语义。此外，我们还展示了该条件操作保持SCMs的简洁性，无环性和线性性，并与边际化操作相符合。由于这些特性与边际化和干预结合起来，条件操作为在潜在细节已经去除的因果模型中进行因果推理任务提供了一个有价值的工具。我们通过例子演示了如何将因果推断的经典结果推广以包括选择偏倚。

    Selection bias is ubiquitous in real-world data, and can lead to misleading results if not dealt with properly. We introduce a conditioning operation on Structural Causal Models (SCMs) to model latent selection from a causal perspective. We show that the conditioning operation transforms an SCM with the presence of an explicit latent selection mechanism into an SCM without such selection mechanism, which partially encodes the causal semantics of the selected subpopulation according to the original SCM. Furthermore, we show that this conditioning operation preserves the simplicity, acyclicity, and linearity of SCMs, and commutes with marginalization. Thanks to these properties, combined with marginalization and intervention, the conditioning operation offers a valuable tool for conducting causal reasoning tasks within causal models where latent details have been abstracted away. We demonstrate by example how classical results of causal inference can be generalized to include selection b
    
[^2]: FinGPT：开源金融大型语言模型

    FinGPT: Open-Source Financial Large Language Models. (arXiv:2306.06031v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.06031](http://arxiv.org/abs/2306.06031)

    FinGPT是一个开源的金融大型语言模型，提供了可访问和透明的资源来开发金融LLMs，其重要性在于自动数据筛选管道和轻量级低秩适应技术。

    

    大型语言模型（LLMs）展示了在各个领域革新自然语言处理任务的潜力，引起了金融领域的浓厚兴趣。获得高质量的金融数据是金融LLMs（FinLLMs）的第一个挑战。在这篇论文中，我们提出了一个针对金融领域的开源大型语言模型FinGPT。与专有模型不同，FinGPT采用数据为中心的方法，为研究人员和从业者提供可访问和透明的资源来开发他们的金融LLMs。我们强调自动数据筛选管道和轻量级低秩适应技术在建立FinGPT中的重要性。此外，我们展示了几个潜在的应用作为用户的基础，如机器顾问、算法交易和论 。

    Large language models (LLMs) have shown the potential of revolutionizing natural language processing tasks in diverse domains, sparking great interest in finance. Accessing high-quality financial data is the first challenge for financial LLMs (FinLLMs). While proprietary models like BloombergGPT have taken advantage of their unique data accumulation, such privileged access calls for an open-source alternative to democratize Internet-scale financial data.  In this paper, we present an open-source large language model, FinGPT, for the finance sector. Unlike proprietary models, FinGPT takes a data-centric approach, providing researchers and practitioners with accessible and transparent resources to develop their FinLLMs. We highlight the importance of an automatic data curation pipeline and the lightweight low-rank adaptation technique in building FinGPT. Furthermore, we showcase several potential applications as stepping stones for users, such as robo-advising, algorithmic trading, and l
    
[^3]: 带权重空间上功能性输入映射的全局普适逼近

    Global universal approximation of functional input maps on weighted spaces. (arXiv:2306.03303v1 [stat.ML])

    [http://arxiv.org/abs/2306.03303](http://arxiv.org/abs/2306.03303)

    本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。

    

    我们引入了所谓的功能性输入神经网络，定义在可能是无限维带权重空间上，其值也在可能是无限维的输出空间中。为此，我们使用一个加性族作为隐藏层映射，以及一个非线性激活函数应用于每个隐藏层。依靠带权重空间上的Stone-Weierstrass定理，我们可以证明连续函数的推广的全局普适逼近结果，超越了常规紧集逼近。这特别适用于通过功能性输入神经网络逼近（非先见之明的）路径空间函数。作为带权Stone-Weierstrass定理的进一步应用，我们证明了线性函数签名的全局普适逼近结果。我们还在这个设置中引入了高斯过程回归的观点，并展示了签名内核的再生核希尔伯特空间是某些高斯过程的Cameron-Martin空间。

    We introduce so-called functional input neural networks defined on a possibly infinite dimensional weighted space with values also in a possibly infinite dimensional output space. To this end, we use an additive family as hidden layer maps and a non-linear activation function applied to each hidden layer. Relying on Stone-Weierstrass theorems on weighted spaces, we can prove a global universal approximation result for generalizations of continuous functions going beyond the usual approximation on compact sets. This then applies in particular to approximation of (non-anticipative) path space functionals via functional input neural networks. As a further application of the weighted Stone-Weierstrass theorem we prove a global universal approximation result for linear functions of the signature. We also introduce the viewpoint of Gaussian process regression in this setting and show that the reproducing kernel Hilbert space of the signature kernels are Cameron-Martin spaces of certain Gauss
    
[^4]: 超越统计相似性：重新思考机器学习在工程设计中的度量方法

    Beyond Statistical Similarity: Rethinking Metrics for Deep Generative Models in Engineering Design. (arXiv:2302.02913v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02913](http://arxiv.org/abs/2302.02913)

    本文提供了一篇深度学习在工程设计中度量方法的综述和指南。传统的基于似然性的统计度量方法在对工程应用的要求上可能无法充分捕捉，因此本文编辑了一组全面的新度量标准，旨在解决传统度量标准的缺点，并更好地与工程设计的需求相一致。通过案例研究，本文展示了这些度量标准如何应用于评估深度生成模型在工程设计中的性能，并发现这些度量标准在捕捉设计的重要细微差别方面表现优于传统的统计度量标准。

    

    深度生成模型，如变分自编码器（VAEs），生成对抗网络（GANs），扩散模型和Transformer等，在图像和语音合成、自然语言处理和药物开发等各种应用中显示出巨大的潜力。然而，在工程设计问题中应用这些模型时，评估这些模型的性能可能会很具有挑战性，因为传统的基于似然性的统计度量方法可能无法充分捕捉工程应用的要求。本文旨在提供一篇深度学习在工程设计中的度量指南和综述。首先，我们总结了深度生成模型的“经典”评估度量标准，这些标准基于机器学习理论和典型的计算机应用，然后使用案例研究，强调了这些度量标准为何很少能够转化为设计问题但又因缺乏确立的替代选择而经常使用。接下来，我们编辑了一组全面的新度量标准，旨在解决传统度量标准的缺点，并更好地与工程设计的需求相一致。我们演示了如何应用这些度量标准来评估深度生成模型在工程设计应用中的性能。我们的结果表明，提出的度量方法在捕捉设计的重要细微差别方面优于传统的统计度量标准，因此在工程设计情境中为深度生成模型提供了更准确的评估。

    Deep generative models, such as Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Diffusion Models, and Transformers, have shown great promise in a variety of applications, including image and speech synthesis, natural language processing, and drug discovery. However, when applied to engineering design problems, evaluating the performance of these models can be challenging, as traditional statistical metrics based on likelihood may not fully capture the requirements of engineering applications. This paper doubles as a review and a practical guide to evaluation metrics for deep generative models (DGMs) in engineering design. We first summarize well-accepted `classic' evaluation metrics for deep generative models grounded in machine learning theory and typical computer science applications. Using case studies, we then highlight why these metrics seldom translate well to design problems but see frequent use due to the lack of established alternatives. Next, we curat
    

