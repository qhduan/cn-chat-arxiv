# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Masked Autoencoders Are Robust Neural Architecture Search Learners](https://arxiv.org/abs/2311.12086) | 提出了一种基于掩码自编码器的新型神经架构搜索框架，无需标记数据，在搜索过程中使用图像重建任务代替监督学习目标，具有鲁棒性、性能和泛化能力，并通过引入多尺度解码器解决了性能崩溃问题。 |
| [^2] | [Comparing Time-Series Analysis Approaches Utilized in Research Papers to Forecast COVID-19 Cases in Africa: A Literature Review.](http://arxiv.org/abs/2310.03606) | 本文献综述比较了在预测非洲COVID-19病例中使用的各种时间序列分析方法，突出了它们的有效性和局限性。 |
| [^3] | [On Memorization and Privacy Risks of Sharpness Aware Minimization.](http://arxiv.org/abs/2310.00488) | 本研究通过对过度参数化模型中的数据记忆的剖析，揭示了尖锐意识最小化算法在非典型数据点上实现的泛化收益。同时，也发现了与此算法相关的更高隐私风险，并提出了缓解策略，以达到更理想的准确度与隐私权衡。 |

# 详细

[^1]: 掩码自编码器是鲁棒的神经架构搜索学习器

    Masked Autoencoders Are Robust Neural Architecture Search Learners

    [https://arxiv.org/abs/2311.12086](https://arxiv.org/abs/2311.12086)

    提出了一种基于掩码自编码器的新型神经架构搜索框架，无需标记数据，在搜索过程中使用图像重建任务代替监督学习目标，具有鲁棒性、性能和泛化能力，并通过引入多尺度解码器解决了性能崩溃问题。

    

    神经架构搜索（NAS）目前严重依赖标记数据，而获取标记数据既昂贵又耗时。本文提出了一种基于掩码自编码器（MAE）的新型NAS框架，它在搜索过程中消除了对标记数据的需求。通过将监督学习目标替换为图像重建任务，我们的方法使得能够在不损害性能和泛化能力的情况下鲁棒地发现网络架构。此外，我们通过引入多尺度解码器解决了在无监督范式中广泛使用的可微架构搜索（DARTS）方法遇到的性能崩溃问题。通过在各种搜索空间和数据集上进行大量实验，我们展示了所提方法的有效性和鲁棒性，为其胜过基准方法提供了实证证据。

    arXiv:2311.12086v2 Announce Type: replace  Abstract: Neural Architecture Search (NAS) currently relies heavily on labeled data, which is both expensive and time-consuming to acquire. In this paper, we propose a novel NAS framework based on Masked Autoencoders (MAE) that eliminates the need for labeled data during the search process. By replacing the supervised learning objective with an image reconstruction task, our approach enables the robust discovery of network architectures without compromising performance and generalization ability. Additionally, we address the problem of performance collapse encountered in the widely-used Differentiable Architecture Search (DARTS) method in the unsupervised paradigm by introducing a multi-scale decoder. Through extensive experiments conducted on various search spaces and datasets, we demonstrate the effectiveness and robustness of the proposed method, providing empirical evidence of its superiority over baseline approaches.
    
[^2]: 将用于研究论文的时间序列分析方法与预测非洲COVID-19病例的比较：文献综述

    Comparing Time-Series Analysis Approaches Utilized in Research Papers to Forecast COVID-19 Cases in Africa: A Literature Review. (arXiv:2310.03606v1 [cs.LG])

    [http://arxiv.org/abs/2310.03606](http://arxiv.org/abs/2310.03606)

    本文献综述比较了在预测非洲COVID-19病例中使用的各种时间序列分析方法，突出了它们的有效性和局限性。

    

    本文献综述旨在比较在预测非洲COVID-19病例中使用的各种时间序列分析方法。该研究对2020年1月至2023年7月发表的英文研究论文进行了系统搜索，重点关注在非洲COVID-19数据集上使用时间序列分析方法的论文。该过程使用了包括PubMed、谷歌学术、Scopus和科学引文索引等多种数据库。研究论文经过评估过程，提取了关于时间序列分析模型的实施和性能的相关信息。该研究突出了不同的方法学，并评估了它们在预测病毒传播方面的有效性和局限性。本综述的结果可以为该领域提供更深入的见解，未来的研究应考虑这些见解，以改进时间序列分析模型并探索不同方法之间的整合。

    This literature review aimed to compare various time-series analysis approaches utilized in forecasting COVID-19 cases in Africa. The study involved a methodical search for English-language research papers published between January 2020 and July 2023, focusing specifically on papers that utilized time-series analysis approaches on COVID-19 datasets in Africa. A variety of databases including PubMed, Google Scholar, Scopus, and Web of Science were utilized for this process. The research papers underwent an evaluation process to extract relevant information regarding the implementation and performance of the time-series analysis models. The study highlighted the different methodologies employed, evaluating their effectiveness and limitations in forecasting the spread of the virus. The result of this review could contribute deeper insights into the field, and future research should consider these insights to improve time series analysis models and explore the integration of different appr
    
[^3]: 关于尖锐意识最小化的记忆和隐私风险研究

    On Memorization and Privacy Risks of Sharpness Aware Minimization. (arXiv:2310.00488v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.00488](http://arxiv.org/abs/2310.00488)

    本研究通过对过度参数化模型中的数据记忆的剖析，揭示了尖锐意识最小化算法在非典型数据点上实现的泛化收益。同时，也发现了与此算法相关的更高隐私风险，并提出了缓解策略，以达到更理想的准确度与隐私权衡。

    

    在许多最近的研究中，设计寻求神经网络损失优化中更平坦的极值的算法成为焦点，因为有经验证据表明这会在许多数据集上导致更好的泛化性能。在这项工作中，我们通过过度参数化模型中的数据记忆视角来剖析这些性能收益。我们定义了一个新的度量指标，帮助我们确定相对于普通SGD，寻求更平坦极值的算法在哪些数据点上表现更好。我们发现，尖锐意识最小化（SAM）所实现的泛化收益在非典型数据点上特别显著，这需要记忆。这一认识帮助我们揭示与SAM相关的更高的隐私风险，并通过详尽的实证评估进行验证。最后，我们提出缓解策略，以实现更理想的准确度与隐私权衡。

    In many recent works, there is an increased focus on designing algorithms that seek flatter optima for neural network loss optimization as there is empirical evidence that it leads to better generalization performance in many datasets. In this work, we dissect these performance gains through the lens of data memorization in overparameterized models. We define a new metric that helps us identify which data points specifically do algorithms seeking flatter optima do better when compared to vanilla SGD. We find that the generalization gains achieved by Sharpness Aware Minimization (SAM) are particularly pronounced for atypical data points, which necessitate memorization. This insight helps us unearth higher privacy risks associated with SAM, which we verify through exhaustive empirical evaluations. Finally, we propose mitigation strategies to achieve a more desirable accuracy vs privacy tradeoff.
    

