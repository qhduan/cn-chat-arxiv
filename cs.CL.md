# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SongComposer: A Large Language Model for Lyric and Melody Composition in Song Generation](https://arxiv.org/abs/2402.17645) | SongComposer提出了一种用于歌曲生成的大型语言模型，采用符号化的歌曲表示，实现了LLM可以明确创作歌曲的能力。 |
| [^2] | [Multilingual Text-to-Image Generation Magnifies Gender Stereotypes and Prompt Engineering May Not Help You](https://arxiv.org/abs/2401.16092) | 多语言文本到图像生成模型存在性别偏见；通过MAGBIG评估模型时，发现模型对不同语言具有重要差异；我们呼吁研究多语言模型领域消除性别偏见。 |
| [^3] | [A Neural Lambda Calculus: Neurosymbolic AI meets the foundations of computing and functional programming.](http://arxiv.org/abs/2304.09276) | 本文提出了一种神经λ演算法，使用λ语言编程，研究神经网络在执行整个程序的能力，旨在拓展神经网络在符号人工智能领域的应用。 |
| [^4] | [Efficient distributed representations beyond negative sampling.](http://arxiv.org/abs/2303.17475) | 本文介绍了一种高效的分布式表示（嵌入）学习方法，通过线性时间估计softmax归一化常数来实现学习过程，该方法优于负采样方法并在多项测试中验证了其有效性。 |

# 详细

[^1]: SongComposer：一种用于歌曲生成的大型语言模型，用于歌词和旋律创作

    SongComposer: A Large Language Model for Lyric and Melody Composition in Song Generation

    [https://arxiv.org/abs/2402.17645](https://arxiv.org/abs/2402.17645)

    SongComposer提出了一种用于歌曲生成的大型语言模型，采用符号化的歌曲表示，实现了LLM可以明确创作歌曲的能力。

    

    我们提出了SongComposer，一个为歌曲创作而设计的创新型LLM。它能够理解和生成歌曲中的旋律和歌词，通过利用LLM的能力在符号化的歌曲表示中生成。现有的与音乐相关的LLM将音乐视为量化的音频信号，而这种隐式编码导致了编码效率低下和灵活性差。相比之下，我们采用了符号化的歌曲表示，这是人类为音乐设计的成熟和高效的方式，并使LLM能够像人类一样明确地创作歌曲。在实践中，我们设计了一种新颖的元组设计，用于格式化歌词和旋律中的三个音符属性（音高、持续时间和休止时间），从而保证LLM对音乐符号的正确理解，并实现歌词和旋律之间的精确对齐。为了向LLM灌输基本的音乐理解，我们精心收集了SongCompose-PT，一个大规模的歌曲预训练数据集，其中包括了歌词、旋律和成对的

    arXiv:2402.17645v1 Announce Type: cross  Abstract: We present SongComposer, an innovative LLM designed for song composition. It could understand and generate melodies and lyrics in symbolic song representations, by leveraging the capability of LLM. Existing music-related LLM treated the music as quantized audio signals, while such implicit encoding leads to inefficient encoding and poor flexibility. In contrast, we resort to symbolic song representation, the mature and efficient way humans designed for music, and enable LLM to explicitly compose songs like humans. In practice, we design a novel tuple design to format lyric and three note attributes (pitch, duration, and rest duration) in the melody, which guarantees the correct LLM understanding of musical symbols and realizes precise alignment between lyrics and melody. To impart basic music understanding to LLM, we carefully collected SongCompose-PT, a large-scale song pretraining dataset that includes lyrics, melodies, and paired ly
    
[^2]: 多语言文本到图像生成放大了性别刻板印象，并且修正工程可能无法帮助您

    Multilingual Text-to-Image Generation Magnifies Gender Stereotypes and Prompt Engineering May Not Help You

    [https://arxiv.org/abs/2401.16092](https://arxiv.org/abs/2401.16092)

    多语言文本到图像生成模型存在性别偏见；通过MAGBIG评估模型时，发现模型对不同语言具有重要差异；我们呼吁研究多语言模型领域消除性别偏见。

    

    最近，文本到图像生成模型在图像质量、灵活性和文本对齐方面取得了令人惊讶的结果，并因此在越来越多的应用中得到应用。通过改善多语言能力，更多的社群现在可以访问这种技术。然而，正如我们将展示的那样，多语言模型与单语模型一样受到(性别)偏见的困扰。此外，人们自然期望这些模型在不同语言之间提供类似的结果，但事实并非如此，不同语言之间存在重要的差异。因此，我们提出了一个旨在促进没有性别偏见的多语言模型研究的新基准MAGBIG。我们研究了多语言T2I模型是否通过MAGBIG放大了性别偏见。为此，我们使用多语言提示请求特定职业或特质的人像图像(使用形容词)。我们的结果不仅表明模型偏离了规范的假设，...

    Text-to-image generation models have recently achieved astonishing results in image quality, flexibility, and text alignment and are consequently employed in a fast-growing number of applications. Through improvements in multilingual abilities, a larger community now has access to this kind of technology. Yet, as we will show, multilingual models suffer similarly from (gender) biases as monolingual models. Furthermore, the natural expectation is that these models will provide similar results across languages, but this is not the case and there are important differences between languages. Thus, we propose a novel benchmark MAGBIG intending to foster research in multilingual models without gender bias. We investigate whether multilingual T2I models magnify gender bias with MAGBIG. To this end, we use multilingual prompts requesting portrait images of persons of a certain occupation or trait (using adjectives). Our results show not only that models deviate from the normative assumption th
    
[^3]: 一种神经λ演算法：神经符号人工智能遇见计算和函数式编程的基础。

    A Neural Lambda Calculus: Neurosymbolic AI meets the foundations of computing and functional programming. (arXiv:2304.09276v1 [cs.LG])

    [http://arxiv.org/abs/2304.09276](http://arxiv.org/abs/2304.09276)

    本文提出了一种神经λ演算法，使用λ语言编程，研究神经网络在执行整个程序的能力，旨在拓展神经网络在符号人工智能领域的应用。

    

    在过去几十年中，基于深度神经网络的模型成为了机器学习中的主导范式。最近，人们越来越认为在符号学习中使用人工神经网络是越来越相关的。为了研究神经网络在符号人工智能领域的能力，研究人员已经探索了深度神经网络学习数学构造（如加法和乘法）、逻辑推理（如定理证明器）甚至执行计算机程序的能力。然而，后者对于神经网络来说是太复杂的任务，结果并不总是成功的，并且往往需要在学习过程中引入有偏见的元素，以限制可能要执行的程序的范围。在这项工作中，我们将分析神经网络学习如何执行整个程序的能力。为此，我们提出了一种不同的方法。我们不使用命令式编程语言，而是采用λ语言进行编程。

    Over the last decades, deep neural networks based-models became the dominant paradigm in machine learning. Further, the use of artificial neural networks in symbolic learning has been seen as increasingly relevant recently. To study the capabilities of neural networks in the symbolic AI domain, researchers have explored the ability of deep neural networks to learn mathematical constructions, such as addition and multiplication, logic inference, such as theorem provers, and even the execution of computer programs. The latter is known to be too complex a task for neural networks. Therefore, the results were not always successful, and often required the introduction of biased elements in the learning process, in addition to restricting the scope of possible programs to be executed. In this work, we will analyze the ability of neural networks to learn how to execute programs as a whole. To do so, we propose a different approach. Instead of using an imperative programming language, with com
    
[^4]: 超越负采样的高效分布式表示方法

    Efficient distributed representations beyond negative sampling. (arXiv:2303.17475v1 [cs.LG])

    [http://arxiv.org/abs/2303.17475](http://arxiv.org/abs/2303.17475)

    本文介绍了一种高效的分布式表示（嵌入）学习方法，通过线性时间估计softmax归一化常数来实现学习过程，该方法优于负采样方法并在多项测试中验证了其有效性。

    

    本文介绍了一种高效的学习分布式表示（也称为嵌入）的方法。该方法通过最小化一个类似于Word2Vec算法中引入并在多个工作中采用的目标函数来实现。优化计算的瓶颈是softmax归一化常数的计算，这需要与样本大小呈二次比例的操作数。这种复杂度不适用于大型数据集，所以负采样是一个常见的解决方法，可以在与样本大小线性相关的时间内获得分布式表示。然而，负采样会改变损失函数，因此解决的是与最初提出的不同的优化问题。我们的贡献在于展示如何通过线性时间估计softmax归一化常数，从而设计了一种有效的优化策略来学习分布式表示。我们使用不同的数据集进行测试，并展示了我们的方法在嵌入质量和训练时间方面优于负采样。

    This article describes an efficient method to learn distributed representations, also known as embeddings. This is accomplished minimizing an objective function similar to the one introduced in the Word2Vec algorithm and later adopted in several works. The optimization computational bottleneck is the calculation of the softmax normalization constants for which a number of operations scaling quadratically with the sample size is required. This complexity is unsuited for large datasets and negative sampling is a popular workaround, allowing one to obtain distributed representations in linear time with respect to the sample size. Negative sampling consists, however, in a change of the loss function and hence solves a different optimization problem from the one originally proposed. Our contribution is to show that the sotfmax normalization constants can be estimated in linear time, allowing us to design an efficient optimization strategy to learn distributed representations. We test our ap
    

