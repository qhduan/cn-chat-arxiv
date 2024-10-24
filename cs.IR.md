# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Post-Training Attribute Unlearning in Recommender Systems](https://arxiv.org/abs/2403.06737) | 本文提出了一种后训练属性取消学习（PoT-AU）的方法，通过设计两部分损失函数，旨在在推荐系统中保护用户的敏感属性。 |
| [^2] | [LLM-Assisted Multi-Teacher Continual Learning for Visual Question Answering in Robotic Surgery](https://arxiv.org/abs/2402.16664) | LLM辅助的多教师持续学习为机器人手术中的视觉问答系统更新提供了解决新任务需求的方法，同时解决了外科领域中的大领域转变和数据不平衡问题。 |
| [^3] | [From Keywords to Structured Summaries: Streamlining Scholarly Knowledge Access](https://arxiv.org/abs/2402.14622) | 该论文突出了信息检索引擎在科学界的重要性，并提出了一种通过结构化记录和先进信息技术工具实现的解决方案，以革新研究人员访问和过滤文章的方式。 |

# 详细

[^1]: 在推荐系统中进行后训练属性遗忘

    Post-Training Attribute Unlearning in Recommender Systems

    [https://arxiv.org/abs/2403.06737](https://arxiv.org/abs/2403.06737)

    本文提出了一种后训练属性取消学习（PoT-AU）的方法，通过设计两部分损失函数，旨在在推荐系统中保护用户的敏感属性。

    

    随着推荐系统中日益增长的隐私问题，推荐取消学习越来越受到关注。现有研究主要使用训练数据，即模型输入，作为取消学习目标。然而，即使模型在训练过程中没有明确遇到，攻击者仍可以从模型中提取私人信息。我们将这些未见信息称为属性，并将其视为取消学习目标。为了保护用户的敏感属性，属性取消学习（AU）旨在使目标属性难以分辨。本文侧重于AU的一个严格但实际的设置，即后训练属性取消学习（PoT-AU），其中取消学习只能在推荐模型训练完成后执行。为了解决推荐系统中的PoT-AU问题，我们提出了一个两部分损失函数。第一部分是可区分性损失，我们设计了一个基于分布的度量

    arXiv:2403.06737v1 Announce Type: new  Abstract: With the growing privacy concerns in recommender systems, recommendation unlearning is getting increasing attention. Existing studies predominantly use training data, i.e., model inputs, as unlearning target. However, attackers can extract private information from the model even if it has not been explicitly encountered during training. We name this unseen information as \textit{attribute} and treat it as unlearning target. To protect the sensitive attribute of users, Attribute Unlearning (AU) aims to make target attributes indistinguishable. In this paper, we focus on a strict but practical setting of AU, namely Post-Training Attribute Unlearning (PoT-AU), where unlearning can only be performed after the training of the recommendation model is completed. To address the PoT-AU problem in recommender systems, we propose a two-component loss function. The first component is distinguishability loss, where we design a distribution-based meas
    
[^2]: LLM辅助的多教师持续学习在机器人手术中的视觉问答

    LLM-Assisted Multi-Teacher Continual Learning for Visual Question Answering in Robotic Surgery

    [https://arxiv.org/abs/2402.16664](https://arxiv.org/abs/2402.16664)

    LLM辅助的多教师持续学习为机器人手术中的视觉问答系统更新提供了解决新任务需求的方法，同时解决了外科领域中的大领域转变和数据不平衡问题。

    

    视觉问答(VQA)在促进机器人辅助手术教育方面可能至关重要。在实践中，学员的需求不断发展，比如学习更多种类的手术，适应不同的机器人，以及为一种手术学习新的外科器械和技术。因此，在机器人手术中需要通过多个资源的顺序数据流持续更新VQA系统，以解决新任务。在外科场景中，存储成本和患者数据隐私通常限制了在更新模型时旧数据的可用性，这需要一个无样本的持续学习(CL)设置。然而，先前的研究忽视了外科领域的两个重要问题：i)来自不同科室或临床中心收集的各种外科手术的大领域转变，ii)由于外科器械或活动的不均匀出现而导致的严重数据不平衡。

    arXiv:2402.16664v1 Announce Type: new  Abstract: Visual question answering (VQA) can be fundamentally crucial for promoting robotic-assisted surgical education. In practice, the needs of trainees are constantly evolving, such as learning more surgical types, adapting to different robots, and learning new surgical instruments and techniques for one surgery. Therefore, continually updating the VQA system by a sequential data stream from multiple resources is demanded in robotic surgery to address new tasks. In surgical scenarios, the storage cost and patient data privacy often restrict the availability of old data when updating the model, necessitating an exemplar-free continual learning (CL) setup. However, prior studies overlooked two vital problems of the surgical domain: i) large domain shifts from diverse surgical operations collected from multiple departments or clinical centers, and ii) severe data imbalance arising from the uneven presence of surgical instruments or activities du
    
[^3]: 从关键词到结构化摘要: 精简学术知识获取

    From Keywords to Structured Summaries: Streamlining Scholarly Knowledge Access

    [https://arxiv.org/abs/2402.14622](https://arxiv.org/abs/2402.14622)

    该论文突出了信息检索引擎在科学界的重要性，并提出了一种通过结构化记录和先进信息技术工具实现的解决方案，以革新研究人员访问和过滤文章的方式。

    

    这篇短文强调了信息检索引擎在科学界日益重要，指出传统基于关键词的搜索引擎由于出版物数量不断增加而效率低下。提出的解决方案涉及结构化记录，支持先进的信息技术工具，包括可视化仪表板，以彻底改变研究人员如何访问和过滤文章，取代传统的文本密集型方法。这一愿景通过一个以“传染病的繁殖数估计”研究主题为中心的概念验证得以体现，使用经过调整的大型语言模型(LLM)自动创建结构化记录以填充一个超越关键词的后端数据库。结果是一个下一代信息检索方法，可在https://orkg.org/usecases/r0-estimates 上访问。

    arXiv:2402.14622v1 Announce Type: cross  Abstract: This short paper highlights the growing importance of information retrieval (IR) engines in the scientific community, addressing the inefficiency of traditional keyword-based search engines due to the rising volume of publications. The proposed solution involves structured records, underpinning advanced information technology (IT) tools, including visualization dashboards, to revolutionize how researchers access and filter articles, replacing the traditional text-heavy approach. This vision is exemplified through a proof of concept centered on the ``reproductive number estimate of infectious diseases'' research theme, using a fine-tuned large language model (LLM) to automate the creation of structured records to populate a backend database that now goes beyond keywords. The result is a next-generation IR method accessible at https://orkg.org/usecases/r0-estimates.
    

