# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Personalized Negative Reservoir for Incremental Learning in Recommender Systems](https://arxiv.org/abs/2403.03993) | 推荐系统中的个性化负采样技术在增量学习中的应用，解决了更新推荐系统模型时遇到的遗忘灾难问题。 |
| [^2] | [CODE-ACCORD: A Corpus of Building Regulatory Data for Rule Generation towards Automatic Compliance Checking](https://arxiv.org/abs/2403.02231) | 介绍了一个独特的数据集CODE-ACCORD，旨在解决自动合规检查中解释建筑法规的挑战，成为机器可读规则生成的基础。 |

# 详细

[^1]: 个性化负采样在推荐系统增量学习中的应用

    Personalized Negative Reservoir for Incremental Learning in Recommender Systems

    [https://arxiv.org/abs/2403.03993](https://arxiv.org/abs/2403.03993)

    推荐系统中的个性化负采样技术在增量学习中的应用，解决了更新推荐系统模型时遇到的遗忘灾难问题。

    

    推荐系统已成为在线平台的重要组成部分。每天训练数据量不断扩大，用户互动次数不断增加。探索更大更具表现力的模型已成为改善用户体验的必要追求。然而，这种进展带来了更大的计算负担。在商业环境中，一旦推荐系统模型被训练和部署，通常需要频繁更新以适应新的客户数据。累积起来，数据量的增加必将使得从头开始进行全量重训练变得计算上不可行。仅仅在新数据上进行简单微调会遇到已被广泛记录的遗忘灾难问题。尽管负采样在使用隐式反馈进行训练中是至关重要的一部分，但目前并不存在专门针对增量学习的技术。

    arXiv:2403.03993v1 Announce Type: cross  Abstract: Recommender systems have become an integral part of online platforms. Every day the volume of training data is expanding and the number of user interactions is constantly increasing. The exploration of larger and more expressive models has become a necessary pursuit to improve user experience. However, this progression carries with it an increased computational burden. In commercial settings, once a recommendation system model has been trained and deployed it typically needs to be updated frequently as new client data arrive. Cumulatively, the mounting volume of data is guaranteed to eventually make full batch retraining of the model from scratch computationally infeasible. Naively fine-tuning solely on the new data runs into the well-documented problem of catastrophic forgetting. Despite the fact that negative sampling is a crucial part of training with implicit feedback, no specialized technique exists that is tailored to the increme
    
[^2]: CODE-ACCORD：用于规则生成的建筑法规数据语料库

    CODE-ACCORD: A Corpus of Building Regulatory Data for Rule Generation towards Automatic Compliance Checking

    [https://arxiv.org/abs/2403.02231](https://arxiv.org/abs/2403.02231)

    介绍了一个独特的数据集CODE-ACCORD，旨在解决自动合规检查中解释建筑法规的挑战，成为机器可读规则生成的基础。

    

    自动合规检查（ACC）在建筑、工程和施工（AEC）领域内的自动合规检查需要自动解释建筑法规，以发挥其全部潜力。然而，从文本规则中提取信息以将其转换为机器可读格式由于自然语言的复杂性以及仅能支持先进的机器学习技术的有限资源而成为一个挑战。为了解决这一挑战，我们介绍了一个独特的数据集CODE-ACCORD，这是在欧盟Horizon ACCORD项目下编制的。CODE-ACCORD包含862个来自英格兰和芬兰建筑法规的自包含句子。与我们的核心目标一致，即促进从文本中提取信息以生成机器可读规则，每个句子都注释了实体和关系。实体代表特定组件，如“窗户”和“烟雾探测器”，而re

    arXiv:2403.02231v1 Announce Type: new  Abstract: Automatic Compliance Checking (ACC) within the Architecture, Engineering, and Construction (AEC) sector necessitates automating the interpretation of building regulations to achieve its full potential. However, extracting information from textual rules to convert them to a machine-readable format has been a challenge due to the complexities associated with natural language and the limited resources that can support advanced machine-learning techniques. To address this challenge, we introduce CODE-ACCORD, a unique dataset compiled under the EU Horizon ACCORD project. CODE-ACCORD comprises 862 self-contained sentences extracted from the building regulations of England and Finland. Aligned with our core objective of facilitating information extraction from text for machine-readable rule generation, each sentence was annotated with entities and relations. Entities represent specific components such as "window" and "smoke detectors", while re
    

