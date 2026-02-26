# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE](https://arxiv.org/abs/2402.13604) | 通过OccCANINE工具，我们成功打破了HISCO障碍，实现了自动化职业标准化，从而大大简化了对职业描述的处理和分类过程，为经济学、经济历史等领域的职业结构分析提供了高效且准确的数据。 |
| [^2] | [Incentive-Aware Synthetic Control: Accurate Counterfactual Estimation via Incentivized Exploration](https://arxiv.org/abs/2312.16307) | 本论文提出了一种为了解决合成对照方法中"重叠"假设的问题的激励感知合成对照方法。该方法通过激励单位采取通常不会考虑的干预措施，提供与激励相容的干预建议，从而实现在面板数据环境中准确估计反事实效果。 |

# 详细

[^1]: 打破HISCO障碍：使用OccCANINE进行自动职业标准化

    Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE

    [https://arxiv.org/abs/2402.13604](https://arxiv.org/abs/2402.13604)

    通过OccCANINE工具，我们成功打破了HISCO障碍，实现了自动化职业标准化，从而大大简化了对职业描述的处理和分类过程，为经济学、经济历史等领域的职业结构分析提供了高效且准确的数据。

    

    这篇论文介绍了一种新工具OccCANINE，可自动将职业描述转换为HISCO分类系统。处理和分类职业描述涉及的手动工作容易出错、繁琐且耗时。我们对一个现有的语言模型（CANINE）进行了微调，使其能够在几秒钟到几分钟内自动完成此过程，而以前需要数天甚至数周。该模型在来自22个不同来源贡献的13种语言中的1400万对职业描述和HISCO代码上进行训练。我们的方法表现出精度、召回率和准确率均超过90%。我们的工具突破了象征性HISCO障碍，并使这些数据可供经济学、经济历史和各种相关学科中的职业结构分析使用。

    arXiv:2402.13604v1 Announce Type: new  Abstract: This paper introduces a new tool, OccCANINE, to automatically transform occupational descriptions into the HISCO classification system. The manual work involved in processing and classifying occupational descriptions is error-prone, tedious, and time-consuming. We finetune a preexisting language model (CANINE) to do this automatically thereby performing in seconds and minutes what previously took days and weeks. The model is trained on 14 million pairs of occupational descriptions and HISCO codes in 13 different languages contributed by 22 different sources. Our approach is shown to have accuracy, recall and precision above 90 percent. Our tool breaks the metaphorical HISCO barrier and makes this data readily available for analysis of occupational structures with broad applicability in economics, economic history and various related disciplines.
    
[^2]: 激励感知合成对照方法：通过激励探索进行准确的反事实估计

    Incentive-Aware Synthetic Control: Accurate Counterfactual Estimation via Incentivized Exploration

    [https://arxiv.org/abs/2312.16307](https://arxiv.org/abs/2312.16307)

    本论文提出了一种为了解决合成对照方法中"重叠"假设的问题的激励感知合成对照方法。该方法通过激励单位采取通常不会考虑的干预措施，提供与激励相容的干预建议，从而实现在面板数据环境中准确估计反事实效果。

    

    我们考虑合成对照方法（SCMs）的设定，这是一种在面板数据环境中估计被治疗对象的治疗效应的经典方法。我们揭示了SCMs中经常被忽视但普遍存在的“重叠”假设：一个被治疗的单位可以被写成保持控制的单位的某种组合（通常是凸或线性组合）。我们展示了如果单位选择自己的干预措施，并且单位之间的异质性足够大，以至于他们偏好不同的干预措施，重叠将不成立。为了解决这个问题，我们提出了一个框架，通过激励具有不同偏好的单位来采取他们通常不会考虑的干预措施。具体来说，我们利用信息设计和在线学习的工具，提出了一种SCM，通过为单位提供与激励相容的干预建议，在面板数据环境中激励探索。

    arXiv:2312.16307v2 Announce Type: replace-cross Abstract: We consider the setting of synthetic control methods (SCMs), a canonical approach used to estimate the treatment effect on the treated in a panel data setting. We shed light on a frequently overlooked but ubiquitous assumption made in SCMs of "overlap": a treated unit can be written as some combination -- typically, convex or linear combination -- of the units that remain under control. We show that if units select their own interventions, and there is sufficiently large heterogeneity between units that prefer different interventions, overlap will not hold. We address this issue by proposing a framework which incentivizes units with different preferences to take interventions they would not normally consider. Specifically, leveraging tools from information design and online learning, we propose a SCM that incentivizes exploration in panel data settings by providing incentive-compatible intervention recommendations to units. We e
    

