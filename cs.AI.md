# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Traffic Signal Control via Genetic Programming](https://arxiv.org/abs/2403.17328) | 提出了一种新的基于学习的方法用于解决复杂交叉路口的信号控制问题，通过设计阶段紧急性概念和可解释的树结构，可以在信号转换期间选择激活的信号相位。 |
| [^2] | [InkSight: Offline-to-Online Handwriting Conversion by Learning to Read and Write](https://arxiv.org/abs/2402.05804) | InkSight是一个可以将离线手写转换为在线手写的系统，通过结合阅读和书写先验知识，在多样化的照片中有效地Derendering手写文本。 |
| [^3] | [OpsEval: A Comprehensive Task-Oriented AIOps Benchmark for Large Language Models.](http://arxiv.org/abs/2310.07637) | OpsEval是一个全面任务导向的AIOps基准测试，评估了大型语言模型在有线网络操作、5G通信操作和数据库操作等关键场景下的能力水平，为提供针对AIOps定制的LLMs的优化方向。 |
| [^4] | [Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures.](http://arxiv.org/abs/2307.15220) | 通过观看手术视频讲座，我们提出了一种新方法，SurgVLP，通过利用手术视频讲座中的语音和视觉信息进行多模态表示学习，并解决了手术相关语言挑战。 |

# 详细

[^1]: 通过遗传编程学习交通信号控制

    Learning Traffic Signal Control via Genetic Programming

    [https://arxiv.org/abs/2403.17328](https://arxiv.org/abs/2403.17328)

    提出了一种新的基于学习的方法用于解决复杂交叉路口的信号控制问题，通过设计阶段紧急性概念和可解释的树结构，可以在信号转换期间选择激活的信号相位。

    

    交通信号控制对提高交通效率至关重要。最近，基于学习的方法，特别是深度强化学习（DRL），在寻求更有效的交通信号控制策略方面取得了巨大成功。然而，在DRL中奖励的设计高度依赖领域知识才能收敛到有效策略，而最终策略也存在解释困难。在本工作中，提出了一种新的面向复杂路口的信号控制的学习方法。在我们的方法中，我们为每个信号相设计了一个阶段紧急性的概念。在信号变换期间，交通灯控制策略根据阶段紧急性选择要激活的下一个相位。然后，我们提出将紧急功能表示为可解释的树结构。紧急功能可以根据当前道路条件为特定相位计算相位紧急性。

    arXiv:2403.17328v1 Announce Type: new  Abstract: The control of traffic signals is crucial for improving transportation efficiency. Recently, learning-based methods, especially Deep Reinforcement Learning (DRL), garnered substantial success in the quest for more efficient traffic signal control strategies. However, the design of rewards in DRL highly demands domain knowledge to converge to an effective policy, and the final policy also presents difficulties in terms of explainability. In this work, a new learning-based method for signal control in complex intersections is proposed. In our approach, we design a concept of phase urgency for each signal phase. During signal transitions, the traffic light control strategy selects the next phase to be activated based on the phase urgency. We then proposed to represent the urgency function as an explainable tree structure. The urgency function can calculate the phase urgency for a specific phase based on the current road conditions. Genetic 
    
[^2]: InkSight：通过学习阅读和书写实现离线到在线手写转换

    InkSight: Offline-to-Online Handwriting Conversion by Learning to Read and Write

    [https://arxiv.org/abs/2402.05804](https://arxiv.org/abs/2402.05804)

    InkSight是一个可以将离线手写转换为在线手写的系统，通过结合阅读和书写先验知识，在多样化的照片中有效地Derendering手写文本。

    

    数字笔记正在变得越来越受欢迎，提供了一种耐用、可编辑和易于索引的存储笔记的方式，即矢量化形式的数字墨水。然而，这种笔记方式与传统的纸笔记方式之间仍存在显著差距，而传统纸笔记方式仍受到绝大多数人的青睐。我们的工作InkSight旨在弥合这种差距，使实体笔记者能够轻松地将他们的作品（离线手写）转换为数字墨水（在线手写），这个过程我们称之为Derendering。之前关于此主题的研究集中在图像的几何属性上，导致了在训练领域之外的有限泛化能力。我们的方法结合了阅读和书写的先验知识，允许在缺乏大量配对样本的情况下训练模型，而这些配对样本很难获取。据我们所知，这是第一个有效地对具有多样化视觉特征和背景的任意照片中的手写文本进行Derendering的工作。

    Digital note-taking is gaining popularity, offering a durable, editable, and easily indexable way of storing notes in the vectorized form, known as digital ink. However, a substantial gap remains between this way of note-taking and traditional pen-and-paper note-taking, a practice still favored by a vast majority. Our work, InkSight, aims to bridge the gap by empowering physical note-takers to effortlessly convert their work (offline handwriting) to digital ink (online handwriting), a process we refer to as Derendering. Prior research on the topic has focused on the geometric properties of images, resulting in limited generalization beyond their training domains. Our approach combines reading and writing priors, allowing training a model in the absence of large amounts of paired samples, which are difficult to obtain. To our knowledge, this is the first work that effectively derenders handwritten text in arbitrary photos with diverse visual characteristics and backgrounds. Furthermore,
    
[^3]: OpsEval: 用于大型语言模型的全面任务导向的AIOps基准测试

    OpsEval: A Comprehensive Task-Oriented AIOps Benchmark for Large Language Models. (arXiv:2310.07637v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.07637](http://arxiv.org/abs/2310.07637)

    OpsEval是一个全面任务导向的AIOps基准测试，评估了大型语言模型在有线网络操作、5G通信操作和数据库操作等关键场景下的能力水平，为提供针对AIOps定制的LLMs的优化方向。

    

    大型语言模型(Large Language Models, LLMs)在翻译、总结和生成等NLP相关任务中表现出了显著的能力。LLMs在特定领域中应用，特别是在AIOps（面向IT运维的人工智能）中，由于其先进的信息汇总、报告分析和API调用能力而具有巨大的潜力。然而，当前LLMs在AIOps任务中的性能尚未确定。此外，需要一个全面的基准测试来引导针对AIOps定制的LLMs的优化。与现有的专注于评估网络配置等特定领域的基准测试不同，本文提出了OpsEval，这是一个专为LLMs设计的全面任务导向的AIOps基准测试。OpsEval首次对LLMs在三个关键场景（有线网络操作、5G通信操作和数据库操作）以及不同的能力水平（知识回忆、分析思考）进行评估。

    Large language models (LLMs) have exhibited remarkable capabilities in NLP-related tasks such as translation, summarizing, and generation. The application of LLMs in specific areas, notably AIOps (Artificial Intelligence for IT Operations), holds great potential due to their advanced abilities in information summarizing, report analyzing, and ability of API calling. Nevertheless, the performance of current LLMs in AIOps tasks is yet to be determined. Furthermore, a comprehensive benchmark is required to steer the optimization of LLMs tailored for AIOps. Compared with existing benchmarks that focus on evaluating specific fields like network configuration, in this paper, we present \textbf{OpsEval}, a comprehensive task-oriented AIOps benchmark designed for LLMs. For the first time, OpsEval assesses LLMs' proficiency in three crucial scenarios (Wired Network Operation, 5G Communication Operation, and Database Operation) at various ability levels (knowledge recall, analytical thinking, an
    
[^4]: 通过观看数百个手术视频讲座学习多模态表示

    Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures. (arXiv:2307.15220v1 [cs.CV])

    [http://arxiv.org/abs/2307.15220](http://arxiv.org/abs/2307.15220)

    通过观看手术视频讲座，我们提出了一种新方法，SurgVLP，通过利用手术视频讲座中的语音和视觉信息进行多模态表示学习，并解决了手术相关语言挑战。

    

    最近在外科计算机视觉应用方面的进展主要依靠完全监督方法，主要使用视觉数据。这些方法依赖于手动注释的手术视频来预测一组固定的对象类别，限制了它们在未见手术程序和后续任务上的通用性。在这项工作中，我们提出了一个观点，即通过开放的手术电子学习平台提供的手术视频讲座可以为多模态表示学习提供有效的监督信号，而无需依赖手动注释。我们通过使用多个互补的自动语音识别系统生成文本转录来解决手术视频讲座中存在的手术相关语言挑战。然后，我们提出了一种新的方法，SurgVLP - 手术视觉语言预训练，用于多模态表示学习。SurgVLP构建了一种新的对比学习目标，将视频剪辑嵌入与相应的文本嵌入对齐。

    Recent advancements in surgical computer vision applications have been driven by fully-supervised methods, primarily using only visual data. These methods rely on manually annotated surgical videos to predict a fixed set of object categories, limiting their generalizability to unseen surgical procedures and downstream tasks. In this work, we put forward the idea that the surgical video lectures available through open surgical e-learning platforms can provide effective supervisory signals for multi-modal representation learning without relying on manual annotations. We address the surgery-specific linguistic challenges present in surgical video lectures by employing multiple complementary automatic speech recognition systems to generate text transcriptions. We then present a novel method, SurgVLP - Surgical Vision Language Pre-training, for multi-modal representation learning. SurgVLP constructs a new contrastive learning objective to align video clip embeddings with the corresponding m
    

