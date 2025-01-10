# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synergizing Spatial Optimization with Large Language Models for Open-Domain Urban Itinerary Planning](https://arxiv.org/abs/2402.07204) | 本文提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程，通过结合空间优化和大型语言模型(LLM)，提供个性化的城市行程定制服务。 |
| [^2] | [LLMs as Workers in Human-Computational Algorithms? Replicating Crowdsourcing Pipelines with LLMs.](http://arxiv.org/abs/2307.10168) | 本文研究探索了LLMs是否可以复制更复杂的众包流水线，并发现现代LLMs在模拟人类计算算法中的能力上有一定的成功，但受多种因素影响。文章强调了为LLMs提供人类面向的安全保障的重要性，并讨论了训练人类和LLMs互补技能的潜力。 |
| [^3] | [BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs.](http://arxiv.org/abs/2303.00915) | BiomedCLIP是一个从1500万科学图像-文本对中预训练的多模态生物医学基础模型，其基于大规模的PMC-15M数据集进行训练，该数据集比现有的生物医学多模态数据集大两个数量级，并成功应用于生物医学图像任务的检索、分类和视觉问题回答等方面。 |

# 详细

[^1]: 结合空间优化和大型语言模型的开放领域城市行程规划

    Synergizing Spatial Optimization with Large Language Models for Open-Domain Urban Itinerary Planning

    [https://arxiv.org/abs/2402.07204](https://arxiv.org/abs/2402.07204)

    本文提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程，通过结合空间优化和大型语言模型(LLM)，提供个性化的城市行程定制服务。

    

    本文首次提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程。OUIP与传统行程规划不同，传统规划限制了用户表达更详细的需求，阻碍了真正的个性化。最近，大型语言模型(LLM)在处理多样化任务方面表现出潜力。然而，由于非实时信息、不完整的知识和不足的空间意识，它们无法独立地提供满意的用户体验。鉴于此，我们提出了一个名为ItiNera的OUIP系统，将空间优化与大型语言模型(LLM)相结合，根据用户需求提供个性化的城市行程定制服务。具体来说，我们开发了一个基于LLM的流水线，用于提取和更新兴趣点特征，以创建用户自己的个性化兴趣点数据库。对于每个用户请求，我们利用LLM进行协同实现优化。

    In this paper, we for the first time propose the task of Open-domain Urban Itinerary Planning (OUIP) for citywalk, which directly generates itineraries based on users' requests described in natural language. OUIP is different from conventional itinerary planning, which limits users from expressing more detailed needs and hinders true personalization. Recently, large language models (LLMs) have shown potential in handling diverse tasks. However, due to non-real-time information, incomplete knowledge, and insufficient spatial awareness, they are unable to independently deliver a satisfactory user experience in OUIP. Given this, we present ItiNera, an OUIP system that synergizes spatial optimization with Large Language Models (LLMs) to provide services that customize urban itineraries based on users' needs. Specifically, we develop an LLM-based pipeline for extracting and updating POI features to create a user-owned personalized POI database. For each user request, we leverage LLM in coop
    
[^2]: LLM作为人-计算算法中的工作者？用LLM复制众包流水线。

    LLMs as Workers in Human-Computational Algorithms? Replicating Crowdsourcing Pipelines with LLMs. (arXiv:2307.10168v1 [cs.CL])

    [http://arxiv.org/abs/2307.10168](http://arxiv.org/abs/2307.10168)

    本文研究探索了LLMs是否可以复制更复杂的众包流水线，并发现现代LLMs在模拟人类计算算法中的能力上有一定的成功，但受多种因素影响。文章强调了为LLMs提供人类面向的安全保障的重要性，并讨论了训练人类和LLMs互补技能的潜力。

    

    LLM已经显示出在众包任务中复制人类行为的潜力，而这些任务以前被认为只有人类才能完成。然而，目前的研究主要集中在简单的原子任务上。我们探索LLM是否可以复制更复杂的众包流水线。我们发现现代LLM可以模拟某些众包工作者在这些“人类计算算法”中的能力，但成功的程度是可变的，并受到请求者对LLM能力的理解、子任务所需的特定技能以及执行这些子任务的最佳交互方式的影响。我们反思了人类和LLM对指示的不同敏感性，强调为LLM提供面向人类的安全保障的重要性，并讨论了训练具有互补技能的人类和LLM的潜力。关键是，我们展示了复制众包流水线提供了一个有价值的平台来研究LLM在不同任务上的相对优势（通过交叉验证

    LLMs have shown promise in replicating human-like behavior in crowdsourcing tasks that were previously thought to be exclusive to human abilities. However, current efforts focus mainly on simple atomic tasks. We explore whether LLMs can replicate more complex crowdsourcing pipelines. We find that modern LLMs can simulate some of crowdworkers' abilities in these "human computation algorithms," but the level of success is variable and influenced by requesters' understanding of LLM capabilities, the specific skills required for sub-tasks, and the optimal interaction modality for performing these sub-tasks. We reflect on human and LLMs' different sensitivities to instructions, stress the importance of enabling human-facing safeguards for LLMs, and discuss the potential of training humans and LLMs with complementary skill sets. Crucially, we show that replicating crowdsourcing pipelines offers a valuable platform to investigate (1) the relative strengths of LLMs on different tasks (by cross
    
[^3]: BiomedCLIP：一种从一千五百万科学图像-文本对进行预训练的多模态生物医学基础模型

    BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. (arXiv:2303.00915v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.00915](http://arxiv.org/abs/2303.00915)

    BiomedCLIP是一个从1500万科学图像-文本对中预训练的多模态生物医学基础模型，其基于大规模的PMC-15M数据集进行训练，该数据集比现有的生物医学多模态数据集大两个数量级，并成功应用于生物医学图像任务的检索、分类和视觉问题回答等方面。

    

    生物医学数据本质上是多模态的，包括物理测量和自然语言叙述。一个通用的生物医学人工智能模型需要同时处理不同的数据模态，包括文本和图像。因此，训练一个有效的通用生物医学模型需要高质量的多模态数据，例如平行的图像-文本对。在这里，我们提供了一个新颖的数据集PMC-15M，比现有的生物医学多模态数据集（如MIMIC-CXR）大两个数量级，并涵盖了各种各样的生物医学图像类型。PMC-15M包含了来自440万科学论文的1500万个生物医学图像-文本对。基于PMC-15M，我们训练了BiomedCLIP，一个多模态基础模型，并进行了领域特定的自适应，以适用于生物医学视觉-语言处理。我们在标准的生物医学图像任务，从检索到分类到视觉问题回答（VQA）方面进行了大量的实验和消融研究。

    Biomedical data is inherently multimodal, comprising physical measurements and natural language narratives. A generalist biomedical AI model needs to simultaneously process different modalities of data, including text and images. Therefore, training an effective generalist biomedical model requires high-quality multimodal data, such as parallel image-text pairs. Here, we present PMC-15M, a novel dataset that is two orders of magnitude larger than existing biomedical multimodal datasets such as MIMIC-CXR, and spans a diverse range of biomedical image types. PMC-15M contains 15 million biomedical image-text pairs collected from 4.4 million scientific articles. Based on PMC-15M, we have pretrained BiomedCLIP, a multimodal foundation model, with domain-specific adaptations tailored to biomedical vision-language processing. We conducted extensive experiments and ablation studies on standard biomedical imaging tasks from retrieval to classification to visual question-answering (VQA). BiomedC
    

