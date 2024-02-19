# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Proving membership in LLM pretraining data via data watermarks](https://arxiv.org/abs/2402.10892) | 使用数据水印在LLM预训练中检测版权持有人作品的方法，可以进行合理检测且提供误检率保证，研究了水印设计对假设检验能力的影响以及在模型和数据集缩放下的检测强度变化。 |
| [^2] | [Instruction Diversity Drives Generalization To Unseen Tasks](https://arxiv.org/abs/2402.10891) | 指导调整通过增加指令集的多样性来推动模型对未见任务的泛化。 |
| [^3] | [When is Tree Search Useful for LLM Planning? It Depends on the Discriminator](https://arxiv.org/abs/2402.10890) | 当前研究通过实验分析了大型语言模型在多步问题求解中使用树搜索的可行性，指出高级规划方法需要鉴别器至少90%准确性才能显著提高性能。 |
| [^4] | [Reviewer2: Optimizing Review Generation Through Prompt Generation](https://arxiv.org/abs/2402.10886) | Reviewer2是一个高效的两阶段评论生成框架，通过明确建模评论可能涉及的各个方面的分布，生成更详细的评论，更好地涵盖人类审稿人在草稿中确定的各种方面。 |
| [^5] | [Multi-modal preference alignment remedies regression of visual instruction tuning on language model](https://arxiv.org/abs/2402.10884) | 通过收集轻量级VQA偏好数据集并使用Direct Preference Optimization，我们能够在语言模型的指导能力上取得显著提升，在小规模数据下比其他方法实现了更高的分数。 |
| [^6] | [Universal Prompt Optimizer for Safe Text-to-Image Generation](https://arxiv.org/abs/2402.10882) | 提出了第一个通用提示优化器，用于在黑盒场景中安全生成文本到图像，通过构建毒素-清洁提示对数据集，设计奖励函数，并通过 Proximal Policy Optimization 训练优化器，成功降低各种 T2I 模型生成不安全内容的可能性。 |
| [^7] | [EcoRank: Budget-Constrained Text Re-ranking Using Large Language Models](https://arxiv.org/abs/2402.10866) | EcoRank是一个两层管线，通过联合优化有关预算分配和LLM API的决策来实现文本重新排序，在实验中表现优于其他预算感知方法。 |
| [^8] | [Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities](https://arxiv.org/abs/2402.10835) | 本研究通过比较LLMs与传统模型，发现了LLMs在时间序列预测中的优势和局限性，指出LLMs在预测具有明显模式和趋势的时间序列方面表现出色，但在缺乏周期性的数据集方面面临挑战，同时指出融入外部知识和采用自然语言释义有助于提升LLMs在时间序列预测中的性能。 |
| [^9] | [Exploring Hybrid Question Answering via Program-based Prompting](https://arxiv.org/abs/2402.10812) | 提出了HProPro，一个基于程序提示的框架，用于处理混合式问答任务，通过代码生成和执行范式以及各种函数来应对混合推理场景。 |
| [^10] | [Quantifying the Persona Effect in LLM Simulations](https://arxiv.org/abs/2402.10811) | 本研究探讨了人物变量对LLMs模拟不同视角能力的影响，发现人物变量在现有主观NLP数据集中解释能力有限，但通过提示方式加入可以略微改善模型预测，尤其在存在争议但范围有限的数据样本上效果最好。 |
| [^11] | [Generative Cross-Modal Retrieval: Memorizing Images in Multimodal Language Models for Retrieval and Beyond](https://arxiv.org/abs/2402.10805) | 提出了一种生成式跨模态检索框架，在多模态语言模型中实现了存储和检索图像的能力 |
| [^12] | [In Search of Needles in a 10M Haystack: Recurrent Memory Finds What LLMs Miss](https://arxiv.org/abs/2402.10790) | 通过使用循环记忆增强对 GPT-2 进行微调，使其能够处理长达 1000 万个元素的任务，这是迄今为止处理最长输入的开放神经网络模型，并展示了对长序列处理能力的显著改进。 |
| [^13] | [EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge](https://arxiv.org/abs/2402.10787) | 本文提出了EdgeQAT，使用熵和分布引导的量化感知训练方法来优化轻量级LLMs，在边缘设备上实现推理加速。 |
| [^14] | [A Condensed Transition Graph Framework for Zero-shot Link Prediction with Large Language Models](https://arxiv.org/abs/2402.10779) | 提出了一种用于零样本链接预测的紧凑转换图框架，能够在线性时间内编码所有路径的信息，并可解决大型语言模型性能受限的问题。 |
| [^15] | [Enhancing ESG Impact Type Identification through Early Fusion and Multilingual Models](https://arxiv.org/abs/2402.10772) | 通过早期融合和多语言模型，提出了一个集成学习的系统，可以最佳地识别ESG影响类型，为当今金融和企业治理领域中的负责任和可持续决策过程做出贡献。 |
| [^16] | [How Reliable Are Automatic Evaluation Methods for Instruction-Tuned LLMs?](https://arxiv.org/abs/2402.10770) | 本文研究了面向指令的大型语言模型中自动评估方法的可靠性，发现自动方法在不同任务类型下与人工评估者之间的相关性存在巨大变化，且在自由形式生成任务和跨语言转移中可能不可靠。 |
| [^17] | [Distillation Enhanced Generative Retrieval](https://arxiv.org/abs/2402.10769) | 通过蒸馏方法增强生成式检索系统，提出了一种名为DGR的框架，利用先进排名模型和蒸馏RankNet损失来优化模型。 |
| [^18] | [Inference to the Best Explanation in Large Language Models](https://arxiv.org/abs/2402.10767) | 该论文提出了一个受哲学启发设计的框架IBE-Eval，用于推进对大型语言模型解释的解释和评估，在因果问答实验中显示出高达77%的准确率。 |
| [^19] | [ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages](https://arxiv.org/abs/2402.10753) | ToolSword提出了一个专门用于细致调查大型语言模型在工具学习中安全问题的全面框架，揭示了在工具学习中持久存在的安全挑战。 |
| [^20] | [GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models](https://arxiv.org/abs/2402.10744) | GenRES提出了一种多维度评估生成式关系抽取结果的方法，填补了使用传统指标评估GRE方法时的不足之处。 |
| [^21] | [Construction of a Syntactic Analysis Map for Yi Shui School through Text Mining and Natural Language Processing Research](https://arxiv.org/abs/2402.10743) | 通过条件随机场构建了基于自然语言处理技术框架下的《易水学派》句法分析图，实现了传统中医药文本的实体关系提取和关键信息提取。 |
| [^22] | [Let's Learn Step by Step: Enhancing In-Context Learning Ability with Curriculum Learning](https://arxiv.org/abs/2402.10738) | 通过少样本上下文课程学习（ICCL）方法，逐渐增加提示演示的复杂性，有效提高了大型语言模型（LLMs）的性能，实验结果显示ICCL对开源LLMs有效。 |
| [^23] | [Assessing the Reasoning Abilities of ChatGPT in the Context of Claim Verification](https://arxiv.org/abs/2402.10735) | 我们提出了一个逻辑推理框架，用于评估ChatGPT在声明验证中的推理能力，发现其在归纳推理方面存在困难，并提出了一种缓解方法。 |
| [^24] | [An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Generative LLM Inference](https://arxiv.org/abs/2402.10712) | 通过实证研究，本文探讨了各种跨语言词汇适应方法对提高生成LLM推理效率的影响。 |
| [^25] | [Rethinking Human-like Translation Strategy: Integrating Drift-Diffusion Model with Large Language Models for Machine Translation](https://arxiv.org/abs/2402.10699) | 将Thinker与漂移扩散模型集成，重新定义漂移扩散过程以模拟人类翻译者的决策制定，实验证明在机器翻译中取得了优异成绩。 |
| [^26] | [Exploring Precision and Recall to assess the quality and diversity of LLMs](https://arxiv.org/abs/2402.10693) | 该研究提出了一种新的评估框架，将精度和召回率指标从图像生成转化为文本生成，细致评估了LLMs生成文本的质量和多样性，揭示了当前LLMs在生成任务中性能表现的重要见解。 |
| [^27] | [MultiPoT: Multilingual Program of Thoughts Harnesses Multiple Programming Languages](https://arxiv.org/abs/2402.10691) | MultiPoT 提出了一种任务和模型无关的方法，通过利用多种编程语言的优势和多样性，在表现上显著优于 Python 自一致性。 |
| [^28] | [Multi-Cultural Commonsense Knowledge Distillation](https://arxiv.org/abs/2402.10689) | 提出了一种MANGO方法，通过从概念和文化两个入口点谨慎而迭代地提示LLMs，提炼高准确度、高召回率的文化知识断言，提供了大量高准确度断言，能够改善对话系统回应的质量、特异性和文化敏感性。 |
| [^29] | [Opening the Black Box of Large Language Models: Two Views on Holistic Interpretability](https://arxiv.org/abs/2402.10688) | 通过整体可解释性框架，本文提出了打开大型语言模型黑匣子的方法，包括自下而上的机械解释和自上而下的表示工程视角，有助于深入理解和应用LLMs的行为和机制。 |
| [^30] | [LongHeads: Multi-Head Attention is Secretly a Long Context Processor](https://arxiv.org/abs/2402.10685) | LongHeads 提出了一个无需训练的框架，通过释放多头注意力的潜力来增强大型语言模型(LLM)处理长上下文的能力。 |
| [^31] | [German Text Simplification: Finetuning Large Language Models with Semi-Synthetic Data](https://arxiv.org/abs/2402.10675) | 该研究利用半合成数据对大型语言模型进行微调，成功完成德语文本的文档级简化，并展示了合成数据在改善文本简化方面的潜力。 |
| [^32] | [Decomposition for Enhancing Attention: Improving LLM-based Text-to-SQL through Workflow Paradigm](https://arxiv.org/abs/2402.10671) | 提出了一种通过工作流范式方法来改善LLMs在文本到SQL中的上下文学习能力，通过分解提高了模型的注意力和问题解决范围，进一步提高了基于LLM的方法的上限。 |
| [^33] | [OpenFMNav: Towards Open-Set Zero-Shot Object Navigation via Vision-Language Foundation Models](https://arxiv.org/abs/2402.10670) | 本研究提出了一种名为OpenFMNav的框架，通过大型语言模型和视觉语言模型解决了目标导航领域中关于理解自然语言指令和零样本泛化的问题。 |
| [^34] | [Humans or LLMs as the Judge? A Study on Judgement Biases](https://arxiv.org/abs/2402.10669) | 提出了一种新框架来研究LLM和人类裁判的偏见，揭示人类和LLM裁判在面对干扰时的脆弱性，强调评估现有LLM性能的挑战。 |
| [^35] | [Multi-Hop Table Retrieval for Open-Domain Text-to-SQL](https://arxiv.org/abs/2402.10666) | 提出了一种多跳表检索方法，通过重写问题和波束搜索来减少相似无关实体的影响，并通过多跳检索中重新编写问题来缓解领域不匹配实体的限制，取得了新的最先进结果 |
| [^36] | [Improving Demonstration Diversity by Human-Free Fusing for Text-to-SQL](https://arxiv.org/abs/2402.10663) | 本文提出了一种通过无需人类参与的多次迭代合成来改善文本到SQL演示的多样性，并构建了高多样性演示池，提高了多样性并降低标注成本。 |
| [^37] | [Fine Tuning Named Entity Extraction Models for the Fantasy Domain](https://arxiv.org/abs/2402.10662) | 通过使用D&D领域中的怪物传说来微调Trankit，实现了命名实体提取模型在奇幻领域的有效应用。 |
| [^38] | [Network Formation and Dynamics Among Multi-LLMs](https://arxiv.org/abs/2402.10659) | 分析了多个LLM在社交网络中的行为，发现它们在给定网络结构并被询问形成网络偏好时表现出与人类社交动态一致的原则。 |
| [^39] | [Enhancing Numerical Reasoning with the Guidance of Reliable Reasoning Processes](https://arxiv.org/abs/2402.10654) | 通过分解答案公式以确保支持答案，借鉴可靠推理过程的方法增强了数值推理能力。 |
| [^40] | [AbsInstruct: Eliciting Abstraction Ability from LLMs through Explanation Tuning with Plausibility Estimation](https://arxiv.org/abs/2402.10646) | 通过指导调节和合理性评估，本研究设计了AbsInstruct框架来增强LLMs的抽象能力，提供了强大的泛化性能。 |
| [^41] | [Can Separators Improve Chain-of-Thought Prompting?](https://arxiv.org/abs/2402.10645) | 分隔符的引入在思维链提示中显著提高了大型语言模型（LLMs）在复杂推理任务上的表现。 |
| [^42] | [`Keep it Together': Enforcing Cohesion in Extractive Summaries by Simulating Human Memory](https://arxiv.org/abs/2402.10643) | 本文通过模拟人类记忆来保持主题连贯性，实现了在提取式摘要中强化连贯性的目标，同时保持信息量和减少冗余。 |
| [^43] | [Generalizability of Mixture of Domain-Specific Adapters from the Lens of Signed Weight Directions and its Application to Effective Model Pruning](https://arxiv.org/abs/2402.10639) | 本研究对领域特定适配器混合在领域内评估中的泛化性进行了全面分析，并探讨了混合适配器的内部运作，为适应新领域的性能优化提供了关键洞见 |
| [^44] | [BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation](https://arxiv.org/abs/2402.10631) | BitDistiller框架将量化感知训练（QAT）与知识蒸馏（KD）相结合，通过引入定制的量化和剪裁技术以及置信感知Kullback-Leibler散度（CAKLD）目标，实现了在极低精度下（低于4位）提升LLMs性能。 |
| [^45] | [Enhancing Role-playing Systems through Aggressive Queries: Evaluation and Improvement](https://arxiv.org/abs/2402.10618) | 本论文设计了MORTISE系统，通过多个LLM模块的协作努力生成高度与角色相关的积极查询，进而改善角色扮演系统的性能。 |
| [^46] | [Can LLMs Speak For Diverse People? Tuning LLMs via Debate to Generate Controllable Controversial Statements](https://arxiv.org/abs/2402.10614) | 本文通过辩论调节LLMs，使其生成可控的支持用户定义论点的声明，改进了LLMs的可控性，并提出了DEBATunE流程。通过两个LLMs之间的多轮辩论生成高质量的训练数据，以支持生成有更高质量和更突出的声明。 |
| [^47] | [Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models](https://arxiv.org/abs/2402.10612) | 本研究提出了一种新方法Rowen，通过选择性检索增强过程，采用多语义感知检测模块来平衡参数化知识和外部信息，以减轻大型语言模型中的幻觉问题。 |
| [^48] | [Jailbreaking Proprietary Large Language Models using Word Substitution Cipher](https://arxiv.org/abs/2402.10601) | 本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。 |
| [^49] | [Efficiency at Scale: Investigating the Performance of Diminutive Language Models in Clinical Tasks](https://arxiv.org/abs/2402.10597) | 研究了不同Parameter Efficient Fine-tuning (PEFT)方法在临床决策任务中的适用性，发现除了LoRA外，大多数PEFT方法在各个模型规模和任务中性能不稳定，而LoRA在所有情况下性能都相对较高。PEFT方法在临床领域特别有效，尤其适用于可以操作的专门模型。 |
| [^50] | [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/abs/2402.10588) | 本研究通过对Llama-2系列变压器模型的研究发现，在多语言语言模型中存在英语作为内部枢纽语言的现象，这有助于理解语言模型的功能方式以及语言偏见的起源。 |
| [^51] | [Threads of Subtlety: Detecting Machine-Generated Texts Through Discourse Motifs](https://arxiv.org/abs/2402.10586) | 本文探讨了如何通过研究文本中的话语特征来区分人类创作和机器生成的文本，引入了一种新颖的方法来揭示这些特征，并发现人类写作在结构上更为多样化。 |
| [^52] | [LinkNER: Linking Local Named Entity Recognition Models to Large Language Models using Uncertainty](https://arxiv.org/abs/2402.10573) | 提出了一种结合小型微调模型和大型语言模型的LinkNER框架，通过不确定性的链接策略RDC，使微调模型能够补充黑盒LLMs |
| [^53] | [Direct Preference Optimization with an Offset](https://arxiv.org/abs/2402.10571) | 提出了一种新颖的直接偏好优化方法，即具有偏置的DPO（ODPO），在微调过程中不同对待每个偏好对。 |
| [^54] | [InSaAF: Incorporating Safety through Accuracy and Fairness | Are LLMs ready for the Indian Legal Domain?](https://arxiv.org/abs/2402.10567) | 本研究在印度法律领域探讨了大型语言模型（LLMs）在处理社会因素时的能力，提出了结合公平性和准确性的新指标$LSS_{\beta}$，并评估了模型在二元法律推理任务中的表现以及在印度社会各种不平等方面的公平性展示。 |
| [^55] | [Neural paraphrasing by automatically crawled and aligned sentence pairs](https://arxiv.org/abs/2402.10558) | 通过自动爬取和对齐句子对，本文提出了一种神经网络重述的方法 |
| [^56] | [SPAR: Personalized Content-Based Recommendation via Long Engagement Attention](https://arxiv.org/abs/2402.10555) | SPAR是一个基于内容的推荐框架，通过利用PLM、多注意力层和注意力稀疏机制，在会话级别有效地处理长期用户参与历史，提取全面用户兴趣，实现个性化推荐。 |
| [^57] | [Disordered-DABS: A Benchmark for Dynamic Aspect-Based Summarization in Disordered Texts](https://arxiv.org/abs/2402.10554) | Disordered-DABS是针对不规则文本中动态基于方面的总结而设计的新基准测试，挑战了现有总结模型的独特性。 |
| [^58] | [Conversational SimulMT: Efficient Simultaneous Translation with Large Language Models](https://arxiv.org/abs/2402.10552) | 通过对话式SimulMT框架，本文提高了基于LLM的SimulMT推理效率，在保持翻译质量的同时实现与专门的SimulMT模型相近的计算延迟。 |
| [^59] | [Strong hallucinations from negation and how to fix them](https://arxiv.org/abs/2402.10543) | 论文针对语言模型在推理中造成的强幻觉问题，提出了一种处理否定的新方法，可以改善模型性能而无需使用稀疏负数据训练。 |
| [^60] | [Properties and Challenges of LLM-Generated Explanations](https://arxiv.org/abs/2402.10532) | 该研究探讨了大型语言模型生成的解释在多领域指导微调数据集上的特性，发现生成的解释表现出选择性和包含说明性元素，但较少是主观或误导性的。 |
| [^61] | [Can We Verify Step by Step for Incorrect Answer Detection?](https://arxiv.org/abs/2402.10528) | 通过推理链来预测大型语言模型输出的准确性，我们引入了一个新的基准R2PE，并提出了处理可辨识性评分（PDS）框架。 |
| [^62] | [Zero-shot sampling of adversarial entities in biomedical question answering](https://arxiv.org/abs/2402.10527) | 在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。 |
| [^63] | [LLM Comparator: Visual Analytics for Side-by-Side Evaluation of Large Language Models](https://arxiv.org/abs/2402.10524) | LLM Comparator是一种用于交互式分析自动并行评估结果的新型可视化工具，支持用户理解模型表现优劣和不同之处，解决了大型语言模型评估中的可扩展性和可解释性挑战。 |
| [^64] | [Provably Sample Efficient RLHF via Active Preference Optimization](https://arxiv.org/abs/2402.10500) | 通过Active Preference Optimization算法，在Bradley-Terry-Luce偏好模型下实现了RLHF的样本效率提高，优化了对提示收集偏好数据的策略。 |
| [^65] | [Comparing Hallucination Detection Metrics for Multilingual Generation](https://arxiv.org/abs/2402.10496) | 本研究比较了多语言生成中不同幻觉检测指标的效果，发现基于自然语言推理（NLI）的指标在高资源语言的句子级别表现良好，但通常无法检测到原子事实幻觉。 |
| [^66] | [Emoji Driven Crypto Assets Market Reactions](https://arxiv.org/abs/2402.10481) | 该研究利用GPT-4和BERT模型进行多模态情感分析，发现基于表情符号情绪的策略可以帮助避免市场下挫并稳定回报。 |
| [^67] | [Large Language Models as Zero-shot Dialogue State Tracker through Function Calling](https://arxiv.org/abs/2402.10466) | 本研究提出了一种通过函数调用将大型语言模型用于零-shot对话状态追踪的新方法，能够在任务导向对话中取得出色的性能，适应不同领域而无需大量数据收集或模型调整。 |
| [^68] | [QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning](https://arxiv.org/abs/2402.10462) | 本论文提出了一种名为QDyLoRA的高效量化动态低秩适应方法，能够在大型语言模型的预定义秩上实现有效微调，与QLoRA相竞争，并且在采用其最佳秩时表现更好。 |
| [^69] | [Steering Conversational Large Language Models for Long Emotional Support Conversations](https://arxiv.org/abs/2402.10453) | 引入了Strategy-Relevant Attention（SRA）度量，评估大型语言模型在情感支持对话中遵循战略提示的有效性，研究发现应用SRA指导的提示可提高战略依从性，从而使长时间对话更可靠地展示所需的情感支持策略。 |
| [^70] | [Incremental Sequence Labeling: A Tale of Two Shifts](https://arxiv.org/abs/2402.10447) | 提出了一种名为IS3的框架，旨在解决增量序列标记任务中的E2O和O2E两种重要的语义转变，通过使用知识蒸馏来维持对旧实体的判别能力。 |
| [^71] | [I Am Not Them: Fluid Identities and Persistent Out-group Bias in Large Language Models](https://arxiv.org/abs/2402.10436) | 论文研究了ChatGPT在不同语言环境中的文化偏见表现，发现当其拥抱特定社会身份时，会区分内外群体，偏好内群体价值观而抵制外群体价值观。 |
| [^72] | [Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models](https://arxiv.org/abs/2402.10430) | 更小的语言模型可以根据样本学习百分比自主选择高质量训练数据，支持更大语言模型的指导调整，实现相媲美甚至优于在整个数据集训练的性能。 |
| [^73] | [Evaluating and Improving Continual Learning in Spoken Language Understanding](https://arxiv.org/abs/2402.10427) | 提出了一种评估方法来统一评估口语理解中的持续学习算法在稳定性、可塑性和泛化能力方面的整体表现，并展示了引入不同知识蒸馏如何改善模型性能。 |
| [^74] | [DELL: Generating Reactions and Explanations for LLM-Based Misinformation Detection](https://arxiv.org/abs/2402.10426) | DELL提出了一个新的方法，将LLMs整合到虚假信息检测的管道中，通过生成新闻反应和解释来提升对新闻文章真实性的判断准确性。 |
| [^75] | [Understanding In-Context Learning with a Pelican Soup Framework](https://arxiv.org/abs/2402.10424) | 提出了一个鹈鹕汤框架，包括常识知识库、自然语言分类任务的形式化以及意义关联的概念，并建立了一个$O(1/T)$的上下文学习损失界限，能够解释对未见任务的泛化。 |
| [^76] | [Pushing the Limits of Zero-shot End-to-End Speech Translation](https://arxiv.org/abs/2402.10422) | 引入了ZeroSwot方法，实现了零-shot ST，通过CTC压缩和最优传输，仅利用ASR数据训练语音编码器，并与多语言MT模型在推断时无缝集成，实现直接从语音到文本的翻译。 |
| [^77] | [Grounding Language about Belief in a Bayesian Theory-of-Mind](https://arxiv.org/abs/2402.10416) | 语义基础置于贝叶斯心灵理论中，通过模拟人们共同推断出解释代理人行为的一致性目标、信念和计划集合，再通过认识逻辑评估有关代理人信念的陈述，解释了人类信念归因的分级性和组合性，以及其与目标和计划的密切联系。 |
| [^78] | [Measuring and Reducing LLM Hallucination without Gold-Standard Answers via Expertise-Weighting](https://arxiv.org/abs/2402.10412) | 提出了一种名为FEWL的幻觉度量方法，通过对LLM答案进行加权评估事实性，适用于没有黄金标准答案的情况。 |
| [^79] | [Understanding Survey Paper Taxonomy about Large Language Models via Graph Representation Learning](https://arxiv.org/abs/2402.10409) | 通过图结构信息在共类别图上利用图表示学习技术，可以在LLMs的预训练模型微调和零-shot/few-shot分类方面显著优于语言模型，揭示了弱标签微调LLMs的潜力。 |
| [^80] | [Chain of Logic: Rule-Based Reasoning with Large Language Models](https://arxiv.org/abs/2402.10400) | 介绍了一种新的提示方法，逻辑链，通过分解和重新组合来促进基于规则的推理，受到律师使用的序贯推理方法的启发。 |
| [^81] | [DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows](https://arxiv.org/abs/2402.10379) | DataDreamer是一种用于合成数据生成和可复现LLM工作流程的开源Python库，有助于研究人员实现强大的LLM工作流，提倡开放科学和可重现性。 |
| [^82] | [BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains](https://arxiv.org/abs/2402.10373) | BioMistral是一种面向生物医学领域的开源预训练大型语言模型集合，在医学问答任务中表现出优越性能并具有竞争优势。 |
| [^83] | [Can we soft prompt LLMs for graph learning tasks?](https://arxiv.org/abs/2402.10359) | 引入了GraphPrompter框架，通过软提示将图信息与LLMs对齐，以进一步探究LLMs理解图信息的潜力。 |
| [^84] | [Prompt-Based Bias Calibration for Better Zero/Few-Shot Learning of Language Models](https://arxiv.org/abs/2402.10353) | 本研究提出了一种空输入提示方法，用于校准预训练语言模型中的固有偏差，从而提升零/少样本学习的性能。 |
| [^85] | [The optimal placement of the head in the noun phrase. The case of demonstrative, numeral, adjective and noun](https://arxiv.org/abs/2402.10311) | 本研究旨在探讨句法依赖距离最小化与意外减少最小化原则在名词短语中的冲突，结论显示当涉及的单词较少且单词较短时，意外减少可能会超越句法依赖距离优化。 |
| [^86] | [How to Discern Important Urgent News?](https://arxiv.org/abs/2402.10302) | 通过分析新闻数据集中的聚类属性，可以强相关地识别出新闻的重要性和紧急性，为识别重要紧急新闻或过滤不重要文章提供了新方法。 |
| [^87] | [LAVE: LLM-Powered Agent Assistance and Language Augmentation for Video Editing](https://arxiv.org/abs/2402.10294) | LAVE通过整合大型语言模型（LLMs），提供LLM动力的代理辅助和语言增强编辑功能，减少视频编辑的障碍，帮助用户实现编辑目标 |
| [^88] | [A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260) | 提出了一种新的基准 StrongREJECT，通过使用更高质量的问题，更好地区分有效和无效的空破解方法。 |
| [^89] | [TOAD: Task-Oriented Automatic Dialogs with Diverse Response Styles](https://arxiv.org/abs/2402.10137) | TOAD是一个具有多样响应风格的面向任务的自动对话系统，其中考虑了冗长程度和用户表达镜像两个方面。TOAD通过模拟真实的应用上下文交互，提供了丰富的系统响应风格选项，并在评估中表明建模更冗长的回复或不进行用户表达镜像的回复更具挑战性。 |
| [^90] | [An Analysis of Langauge Frequency and Error Correction for Esperanto](https://arxiv.org/abs/2402.09696) | 本论文运用 Eo-GP 数据集进行了世界语的频率分析，引入了 Eo-GEC 数据集用于错误识别。实验表明 GPT-4 在自动化和人工评估中的性能优于 GPT-3.5，展示了先进语言模型在增强对于较少研究语言的 GEC 策略方面的潜力。 |
| [^91] | [CodeMind: A Framework to Challenge Large Language Models for Code Reasoning](https://arxiv.org/abs/2402.09664) | CodeMind是一个用于挑战大型语言模型进行代码推理的框架，通过评估LLMs的代码推理能力来替代仅仅依靠测试通过来评估，对三种代码推理任务进行评估，结果显示LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。 |
| [^92] | [MiMiC: Minimally Modified Counterfactuals in the Representation Space](https://arxiv.org/abs/2402.09631) | 提出了一种新颖的对抗事实生成方法，利用闭式解决方案在表示空间中生成富有表达力的对抗事实，以减轻语言模型中的不良行为，该方法在地球移动问题方面提供理论上的保证，并对表示空间的几何组织进行改进。 |
| [^93] | [API Pack: A Massive Multilingual Dataset for API Call Generation](https://arxiv.org/abs/2402.09615) | 这个论文介绍了一个名为API Pack的大规模多语言数据集，旨在提高大型语言模型的API调用生成能力，通过实验证明了其在生成未见过的API调用方面的高准确率，并实现了跨语言的API调用生成 |
| [^94] | [Rethinking Machine Unlearning for Large Language Models](https://arxiv.org/abs/2402.08787) | 这篇论文研究了大型语言模型中的机器消除技术，旨在消除不良数据的影响并保持基本知识生成的完整性，为开发安全、可靠和资源高效的生成式人工智能提供基础。 |
| [^95] | [Plausible Extractive Rationalization through Semi-Supervised Entailment Signal](https://arxiv.org/abs/2402.08479) | 本文通过半监督方法，采用蕴涵对齐，以优化可行性，提取有理的方式提供一个可解释的替代模型 |
| [^96] | [Towards Faithful and Robust LLM Specialists for Evidence-Based Question-Answering](https://arxiv.org/abs/2402.08277) | 这项工作探索了如何鲁棒地微调大型语言模型以提高答案的来源质量和答案归因能力，引入了数据生成流水线和四个测试集来评估模型的性能，并展示了在合成数据上微调可以改善内部和外部分布的性能。 |
| [^97] | [Anchor-based Large Language Models](https://arxiv.org/abs/2402.07616) | 基于锚点的大型语言模型（AnLLM）通过引入创新的基于锚点的自注意力网络（AnSAN）和基于锚点的推理策略，将序列信息压缩到锚点标记中，减少键/值缓存，提高推理效率。 |
| [^98] | [Pushing The Limit of LLM Capacity for Text Classification](https://arxiv.org/abs/2402.07470) | 本论文提出了一个自适应增强框架RGPT，通过反复集成强基学习者，生成一个专用的文本分类LLM。通过实证比较，我们展示了RGPT明显胜过其他方法。 |
| [^99] | [Through the Lens of Split Vote: Exploring Disagreement, Difficulty and Calibration in Legal Case Outcome Classification](https://arxiv.org/abs/2402.07214) | 通过研究分割投票，探索律师在处理法律案件结果分类时面临的意见分歧和困难，并在欧洲人权法院收集了法官的投票数据集进行研究。这项研究还评估了模型和人类之间感知困难的一致性以及模型的置信度和人类校准。 |
| [^100] | [Can LLMs Recognize Toxicity? Structured Toxicity Investigation Framework and Semantic-Based Metric](https://arxiv.org/abs/2402.06900) | 本研究提出了一种基于大型语言模型（LLMs）的自动度量方法，用于识别生成文本中的毒性。通过分析毒性因素和LLMs的内在毒性属性，该方法在测量毒性方面表现出众，比现有指标提升12个百分点。 |
| [^101] | [Debating with More Persuasive LLMs Leads to More Truthful Answers](https://arxiv.org/abs/2402.06782) | 本文研究了更弱的语言模型是否能评估更强的模型的正确性。研究发现，通过进行辩论，非专家模型和人类回答问题的准确性都有所提高。 |
| [^102] | [NICE: To Optimize In-Context Examples or Not?](https://arxiv.org/abs/2402.06733) | 通过研究在提供任务特定指令的情况下是否需要优化上下文示例，我们挑战了对于指导性LLMs的共识，并发现在某些任务中，不同的优化上下文示例方法会产生递减的回报。我们引入了"度量标准"，用于衡量从给定指令中学习任务的能力，并提供了一个启发式方法，帮助决定是否优化指令还是ICE用于任何新任务。 |
| [^103] | [Unified Hallucination Detection for Multimodal Large Language Models](https://arxiv.org/abs/2402.03190) | 该论文提出了一个新颖的统一的多模态幻觉检测框架UNIHD，并设计了一个评估基准方法MHaluBench来评估幻觉检测方法的进展。这项工作扩展了幻觉检测的研究范围并提供了有效的解决方案。 |
| [^104] | [Emojis Decoded: Leveraging ChatGPT for Enhanced Understanding in Social Media Communications](https://arxiv.org/abs/2402.01681) | 在表情符号研究中，我们评估了ChatGPT在处理注释和下游任务中的有效性。我们的研究结果表明ChatGPT可以作为一个可行的替代人类注释者的工具，有效地解释表情符号。 |
| [^105] | [StickerConv: Generating Multimodal Empathetic Responses from Scratch](https://arxiv.org/abs/2402.01679) | 本文介绍了StickerConv代理(Agent4SC)，该代理通过协作代理交互，实现了与贴纸使用相仿的人类行为模拟，从而增强了多模态共情交流。为了利用构建的多模态共情对话数据集StickerConv，作者提出了PErceive and Generate Stickers (PEGS)模型，该模型能够生成情境相关和情感丰富的回应。 |
| [^106] | [I Think, Therefore I am: Awareness in Large Language Models](https://arxiv.org/abs/2401.17882) | 本文介绍了将意识概念引入大型语言模型（LLMs），并定义了LLMs在感知和理解自身以及展示社交智能方面的能力。通过引入AwareLLM数据集，研究发现LLMs在意识方面表现出相当程度的能力，尽管它们缺乏实质性的能力意识。 |
| [^107] | [Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios](https://arxiv.org/abs/2401.17167) | 本论文介绍了一种新的基准测试UltraTool，旨在改善和评估LLMs在实际复杂场景中的工具利用能力。该基准测试关注从规划和创建到应用工具的整个过程，并强调实际复杂性和多步规划的要求。 |
| [^108] | [Combining Hierachical VAEs with LLMs for clinically meaningful timeline summarisation in social media](https://arxiv.org/abs/2401.16240) | 利用混合的分层变分自动编码器与LLMs结合的方法实现了从社交媒体用户时间轴生成具有临床意义的摘要，通过对时间轴的时间敏感性和举重有力的抽象摘要，TH-VAE生成的摘要在捕捉随时间变化方面优于仅使用LLM方法。 |
| [^109] | [Text Embedding Inversion Security for Multilingual Language Models](https://arxiv.org/abs/2401.12192) | 该研究探讨了多语言语言模型的文本嵌入逆转安全性问题，发现多语言模型更容易受到逆转攻击的影响，并提出了简单的掩蔽防御方法。 |
| [^110] | [Identifying and Analyzing Task-Encoding Tokens in Large Language Models](https://arxiv.org/abs/2401.11323) | 本文通过识别和分析任务编码标记，揭示了大型语言模型如何学习执行任务的方式。 |
| [^111] | [SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of Large Language Models](https://arxiv.org/abs/2401.08295) | 提出了一种共享注意力框架（SAPT），通过共享注意力学习与选择模块对齐PET学习和选择，以同时解决大型语言模型中的灾难性遗忘和知识转移挑战。 |
| [^112] | [MARIO: MAth Reasoning with code Interpreter Output -- A Reproducible Pipeline](https://arxiv.org/abs/2401.08190) | 本文通过引入具有Python代码解释器的数学数据集，解决了大型语言模型在数学推理能力方面的挑战。 |
| [^113] | [Small LLMs Are Weak Tool Learners: A Multi-LLM Agent](https://arxiv.org/abs/2401.07324) | 本论文提出了一种新的策略，将大型语言模型代理（LLMs）的能力分解为计划器、调用器和总结器模块，以克服小型模型性能限制和工具更新的问题。 |
| [^114] | [RoleEval: A Bilingual Role Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2312.16132) | 介绍了RoleEval，一个旨在评估大型语言模型角色知识记忆、利用和推理能力的双语基准，涵盖了来自各个领域的300名有影响力的人物和虚构角色，包括6000道中英文平行多项选择题，旨在系统地探究个人信息、关系、能力和经历等各个方面 |
| [^115] | [BloomVQA: Assessing Hierarchical Multi-modal Comprehension](https://arxiv.org/abs/2312.12716) | 提出了新VQA数据集BloomVQA，基于Bloom的分类法，通过层次图表示实现数据增强和模型一致性评估，揭示大型视觉语言模型在高级理解任务上的性能下降。 |
| [^116] | [Response Enhanced Semi-supervised Dialogue Query Generation](https://arxiv.org/abs/2312.12713) | 提出了一种新的半监督学习框架--SemiDQG，通过未标记的对话来改善模型性能，训练响应增强的查询生成器 (RA)。 |
| [^117] | [KGLens: A Parameterized Knowledge Graph Solution to Assess What an LLM Does and Doesn't Know](https://arxiv.org/abs/2312.11539) | KGLens 是一个旨在衡量知识图与大型语言模型（LLMs）之间对齐程度的框架，帮助找出LLMs相对于知识图的知识不足之处。 |
| [^118] | [Cascade Speculative Drafting for Even Faster LLM Inference](https://arxiv.org/abs/2312.11462) | 引入了Cascade Speculative Drafting（CS Drafting）算法，通过垂直级联消除神经模型的自回归生成，通过水平级联优化草稿中的时间分配，从而进一步提高LLM推理效率。 |
| [^119] | [Split and Rephrase with Large Language Models](https://arxiv.org/abs/2312.11075) | 评估了大型语言模型在Split and Rephrase任务上的表现，表明在主要指标上有显著改进，但在分割一致性方面仍有待提高。 |
| [^120] | [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) | 该论文提出了一种具有硬件高效性的线性注意力算法，可在短序列长度下比现有方法更快，同时推广到了具有数据相关门的更具表达能力的线性注意力变体。 |
| [^121] | [Causal ATE Mitigates Unintended Bias in Controlled Text Generation](https://arxiv.org/abs/2311.11229) | 因果ATE方法可解决语言模型中属性控制任务中存在的意外偏差问题，并在毒性缓解问题中得到验证。 |
| [^122] | [WatME: Towards Lossless Watermarking Through Lexical Redundancy](https://arxiv.org/abs/2311.09832) | WatME通过利用词汇冗余的语言先验知识，动态优化语言模型解码过程中的词汇使用，避免适当词汇不可用的情况，维持语言模型的表现力。 |
| [^123] | [Examining LLMs' Uncertainty Expression Towards Questions Outside Parametric Knowledge](https://arxiv.org/abs/2311.09731) | 本研究系统地调查了大型语言模型在缺乏足够参数化知识的情况下如何表达对超出其知识范围的问题的不确定性，并强调了诚实与帮助性之间的权衡。 |
| [^124] | [The Wisdom of Partisan Crowds: Comparing Collective Intelligence in Humans and LLM-based Agents](https://arxiv.org/abs/2311.09665) | 本文研究了基于LLM的代理人在扮演党派角色时，展现出类似于人类群体的党派偏见、并通过商讨收敛到更准确信念的能力。 |
| [^125] | [Decoding Susceptibility: Modeling Misbelief to Misinformation Through a Computational Approach](https://arxiv.org/abs/2311.09630) | 通过计算方法对用户的潜在易感性水平进行建模，可以帮助理解易受错误信息影响的程度，为后续研究和应用提供重要参考。 |
| [^126] | [Simulating Opinion Dynamics with Networks of LLM-based Agents](https://arxiv.org/abs/2311.09618) | 提出了一种基于大型语言模型（LLMs）人口的新方法来模拟意见动态，发现LLM代理存在固有偏见导致模拟代理趋向于科学现实一致的共识，但引入确认偏见后观察到意见分裂，突显了LLM代理在该领域的潜力和局限性。 |
| [^127] | [Digital Socrates: Evaluating LLMs through Explanation Critiques](https://arxiv.org/abs/2311.09613) | 通过定义新的解释批评任务、创建人工验证过的数据集并训练开源自动批评模型，数字苏格拉底有助于揭示学生模型的见解。 |
| [^128] | [Fusion-Eval: Integrating Evaluators with LLMs](https://arxiv.org/abs/2311.09204) | Fusion-Eval是一种创新方法，利用LLMs整合不同辅助评估器的见解，极大提升自然语言系统评估的有效性。 |
| [^129] | [Enabling Large Language Models to Learn from Rules](https://arxiv.org/abs/2311.08883) | 本文探索了一种新的学习范式，将基于规则的知识编码到大型语言模型中，并提出了规则提取方法。 |
| [^130] | [StrategyLLM: Large Language Models as Strategy Generators, Executors, Optimizers, and Evaluators for Problem Solving](https://arxiv.org/abs/2311.08803) | StrategyLLM提出了一个框架，利用大型语言模型的能力自动构建可推广和一致的少次提示，优于竞争基线，不需要人工参与。 |
| [^131] | [SimpleSafetyTests: a Test Suite for Identifying Critical Safety Risks in Large Language Models](https://arxiv.org/abs/2311.08370) | 引入SimpleSafetyTests（SST）作为一个新的测试套件，用于快速系统地识别大语言模型中关键的安全风险 |
| [^132] | [Forgetting before Learning: Utilizing Parametric Arithmetic for Knowledge Updating in Large Language Models](https://arxiv.org/abs/2311.08011) | 提出了一种名为F-Learning的新微调范式，利用参数化算术促进旧知识的遗忘和新知识的学习，在大型语言模型中显著改善知识更新性能 |
| [^133] | [ChartCheck: Explainable Fact-Checking over Real-World Chart Images](https://arxiv.org/abs/2311.07453) | 该论文介绍了ChartCheck，这是一个用于对真实世界图表进行可解释事实检查的新型数据集，旨在解决图表被误用传播错误信息的问题，并提出了视觉语言和图表到表格模型的基线。 |
| [^134] | [Injecting a Structural Inductive Bias into a Seq2Seq Model by Simulation](https://arxiv.org/abs/2310.00796) | 通过模拟结构转换在Seq2Seq模型中注入结构归纳偏差，提高了系统泛化和FST类似任务的少样本学习。 |
| [^135] | [Linearity of Relation Decoding in Transformer Language Models](https://arxiv.org/abs/2308.09124) | 在Transformer语言模型中，部分关系的计算可以通过对主题表示进行单一线性转换来很好地近似，但并非所有关系都能通过线性编码。 |
| [^136] | [History-Aware Conversational Dense Retrieval.](http://arxiv.org/abs/2401.16659) | 该论文提出了一种历史感知的对话式稠密检索系统，通过上下文去噪的查询重构以及根据历史轮次的实际影响自动挖掘监督信号改进了现有的对话式稠密检索方法。 |
| [^137] | [Query of CC: Unearthing Large Scale Domain-Specific Knowledge from Public Corpora.](http://arxiv.org/abs/2401.14624) | 本论文提出了一种通过大型语言模型来收集特定领域知识的高效方法，通过该方法构建了一个高质量的名为“Knowledge Pile”的数据集，实验证明其显著改善了特定领域的数据稀缺问题。 |
| [^138] | [SEER: Facilitating Structured Reasoning and Explanation via Reinforcement Learning.](http://arxiv.org/abs/2401.13246) | SEER是一种通过最大化基于结构的回报来促进结构化推理和解释的新方法。 |
| [^139] | [Mitigating Hallucinations of Large Language Models via Knowledge Consistent Alignment.](http://arxiv.org/abs/2401.10768) | 本文提出了一种称为知识一致性对齐（KCA）的方法，通过减少训练数据中外部知识和预训练语料库中内在知识之间的不一致性，从而缓解了大型语言模型产生幻觉的问题。实验结果表明，KCA方法在多个基准测试中取得了优异的性能。 |
| [^140] | [Batch-ICL: Effective, Efficient, and Order-Agnostic In-Context Learning.](http://arxiv.org/abs/2401.06469) | 本文提出了批处理ICL方法，通过将ICL视为一个元优化过程，开发出了一个有效、高效且无序的推理算法。通过聚合元梯度并将其应用于零-shot学习，该方法使LLM对ICL示例顺序无关，并且在实验证明其在大多数情况下优于其他排列方式，甚至超过了标准ICL的最佳顺序的性能。 |
| [^141] | [Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks.](http://arxiv.org/abs/2401.05949) | 本研究发现上下文学习范式在大型语言模型中存在漏洞，攻击者可以通过污染示范上下文来操控模型行为，而无需进行微调。这项研究设计了一种名为ICLAttack的后门攻击方法，可以通过污染示范样本和提示来使模型按照预定义的意图行事。 |
| [^142] | [AUTOACT: Automatic Agent Learning from Scratch via Self-Planning.](http://arxiv.org/abs/2401.05268) | AUTOACT是一个自动代理学习框架，通过自主规划合成轨迹，不依赖于大规模数据和闭源模型，能够实现更好或类似的性能。 |
| [^143] | [Multi-User Chat Assistant (MUCA): a Framework Using LLMs to Facilitate Group Conversations.](http://arxiv.org/abs/2401.04883) | 这篇论文介绍了一种基于大规模语言模型的多用户聊天机器人框架（MUCA），该框架支持群组讨论，并提供了三个主要模块来确定回应内容、时机和适当的接收者。同时，作者还提出了一个基于语言模型的多用户模拟器（MUS），用于模拟真实用户行为，以便更高效地测试和优化聊天机器人。 |
| [^144] | [Are Language Models More Like Libraries or Like Librarians? Bibliotechnism, the Novel Reference Problem, and the Attitudes of LLMs.](http://arxiv.org/abs/2401.04854) | 本文探讨了语言模型（LLMs）是更像图书馆还是图书管理员的问题。论文首先阐述了 "文献主义 "这一概念，并提出了对其的挑战，指出LLMs生成的全新文本在内容上依赖于原始人类文本的内容。然后，论文提出了对 "文献主义"的新颖挑战，讨论了LLMs生成的 "新引用"问题。最后，根据心灵哲学中的解释主义，论文提出了有限代理能力的LLMs可能存在的可能性。 |
| [^145] | [The Mystery and Fascination of LLMs: A Comprehensive Survey on the Interpretation and Analysis of Emergent Abilities.](http://arxiv.org/abs/2311.00237) | 该论文对LLMs的新兴能力的解释和分析进行了全面调查，旨在理解这些能力的机制和实际应用，并解决可能出现的潜在风险和担忧。 |
| [^146] | [InstructCoder: Empowering Language Models for Code Editing.](http://arxiv.org/abs/2310.20329) | 本研究旨在探索使用大型语言模型（LLMs）进行代码编辑，并引入了InstructCoder数据集，该数据集包含多样性的代码编辑任务，为通用代码编辑提供支持。 |
| [^147] | [Not All Countries Celebrate Thanksgiving: On the Cultural Dominance in Large Language Models.](http://arxiv.org/abs/2310.12481) | 本文研究了大型语言模型中的文化主导问题，发现由于在模型训练中主要使用英语数据，当用户使用非英语语言提问时，模型往往提供与预期文化不相关的不恰当答案。我们提出了通过多样化数据预训练和文化感知提示两种方法来解决这个问题。 |
| [^148] | [From Dissonance to Insights: Dissecting Disagreements in Rationale Dataset Construction for Case Outcome Classification.](http://arxiv.org/abs/2310.11878) | 本研究关注法律自然语言处理中人工标注的变异问题，通过收集一组律师对案件结果评估存在分歧的数据集，对这些分歧进行了研究，构建了一个两级分类体系，并发现分歧主要源于对法律背景的不明确描述。 |
| [^149] | [(Dynamic) Prompting might be all you need to repair Compressed LLMs.](http://arxiv.org/abs/2310.00867) | 提出了一种动态提示(IDP)的机制，它可以作为一种轻量级的适应工具，修复压缩的大型语言模型(LLMs)在一些实际的下游任务中的性能下降。 |
| [^150] | [ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving.](http://arxiv.org/abs/2309.17452) | ToRA是一种集成工具的数学问题求解推理代理，通过结合语言的分析能力和工具的计算效率，能够显著提高数学推理的性能，在多个数学推理数据集上取得了13%-19%的平均绝对改进率，并在竞赛级数据集MATH上达到了44.6%的性能。 |
| [^151] | [Bridging Topic, Domain, and Language Shifts: An Evaluation of Comprehensive Out-of-Distribution Scenarios.](http://arxiv.org/abs/2309.08316) | 本论文评估了语言模型在跨越主题、领域和语言变化的全面非分布场景中的泛化能力，并提出了改进策略，包括基于提示的精细调节和上下文学习。 |
| [^152] | [Investigating Gender Bias in News Summarization.](http://arxiv.org/abs/2309.08047) | 本研究调查了新闻概述中的性别偏见，发现大型语言模型（LLMs）会重复和强化有害的社会偏见。研究提出了一些方法来量化模型中的有偏行为，并提出了一种生成具有控制人口属性的输入文档的方法。 |
| [^153] | [Clinical Text Summarization: Adapting Large Language Models Can Outperform Human Experts.](http://arxiv.org/abs/2309.07430) | 本研究通过对八个大型语言模型在临床摘要任务上的领域适应方法实验进行了全面的定量评估，发现最佳适应的模型的摘要在完整性和正确性方面优于人类摘要。 |
| [^154] | [Large Language Models for Automated Open-domain Scientific Hypotheses Discovery.](http://arxiv.org/abs/2309.02726) | 这项研究提出了用于社会科学学术假设发现的第一个自然语言处理数据集，旨在开发一个系统，能够基于原始网络语料库自动生成有效、新颖且对人类研究者有帮助的假设。 |
| [^155] | [Long-Term Memorability On Advertisements.](http://arxiv.org/abs/2309.00378) | 本研究是首个大规模的记忆性研究，发现广告的长期记忆性对于市场营销非常重要，但在机器学习文献中一直缺乏相关研究。通过分析大量参与者和广告，我们得出了关于什么使广告记忆深刻的有趣见解。 |
| [^156] | [Detoxify Language Model Step-by-Step.](http://arxiv.org/abs/2308.08295) | 这项研究提出了一种分步解毒语言模型的方法，通过在输入阶段进行解毒处理，并使用无毒提示进行连续生成来保持生成质量。同时，通过设计Detox-Chain来校准LLMs的推理能力，实现了更安全和可靠的生成。 |
| [^157] | [Synthesizing Political Zero-Shot Relation Classification via Codebook Knowledge, NLI, and ChatGPT.](http://arxiv.org/abs/2308.07876) | 该论文通过利用已建立的注释编码本的知识，探索零样本方法用于政治事件本体关系分类，并介绍一种基于自然语言推理的方法，名为ZSP。ZSP采用了一种树查询框架，提高了解释性、效率和对模式更改的适应性。在细粒度根代码分类上，ZSP的性能明显优于ChatGPT，F1得分提高了40%。 |
| [^158] | [FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets.](http://arxiv.org/abs/2307.10928) | FLASK是一种基于对齐技能集的细粒度语言模型评估协议，通过将粗级评分分解为每个指令的技能集级评分，实现了对模型性能的全面视角和提高评估的可靠性。 |
| [^159] | [Large language models shape and are shaped by society: A survey of arXiv publication patterns.](http://arxiv.org/abs/2307.10700) | 大型语言模型的论文数量急剧增加，研究重点逐渐转向社会影响。与LLM相关的论文呈现持续增长的趋势，新发表关于LLM的作者更注重应用和社会影响。 |
| [^160] | [Meta-Reasoning: Semantics-Symbol Deconstruction For Large Language Models.](http://arxiv.org/abs/2306.17820) | 本论文提出了一种称为“元推理”的方法，它通过使用语义符号解构的方式，将不同推理问题转化为类似的自然语言表示，以提高大型语言模型的推理能力。 |
| [^161] | [Unsupervised ASR via Cross-Lingual Pseudo-Labeling.](http://arxiv.org/abs/2305.13330) | 本研究提出了一种基于跨语言伪标注的无监督ASR方法，能够使用其他语言中的标注数据来引导新语言的无监督AM。在Common Voice上取得了良好的效果，可以实现18% WER。而且在不同语言的数据集上都优于基线模型。 |
| [^162] | [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.](http://arxiv.org/abs/2305.11738) | 本文提出了一个名为CRITIC的框架，使得大型语言模型可以通过与工具的交互校正自己的错误，从而避免生成出现不一致和问题行为的结果。 |
| [^163] | [Towards Versatile and Efficient Visual Knowledge Injection into Pre-trained Language Models with Cross-Modal Adapters.](http://arxiv.org/abs/2305.07358) | 本文提出了X-adapter插拔式模块，利用多模态视觉语言模型，高效地向预训练语言模型注入视觉知识。 |
| [^164] | [Multi-Relational Hyperbolic Word Embeddings from Natural Language Definitions.](http://arxiv.org/abs/2305.07303) | 本论文提出了一种从自然语言定义中学习多关系双曲词向量的框架，以捕捉由定义所引起的分层和多分辨率结构。 |
| [^165] | [Learning Disentangled Semantic Spaces of Explanations via Invertible Neural Networks.](http://arxiv.org/abs/2305.01713) | 本文介绍了一种使用可逆神经网络将BERT-GPT2自动编码器的隐藏空间转换为更可分离的语义空间的方法，实验结果表明此方法可以改进模型的可解释性和可控性，并取得了比最先进模型更好的性能表现。 |
| [^166] | [Tokenization Tractability for Human and Machine Learning Model: An Annotation Study.](http://arxiv.org/abs/2304.10813) | 研究比较了六种分词方法，并发现人类可追溯的分词与机器学习模型中的分词不一定相同。 |
| [^167] | [Logical Reasoning over Natural Language as Knowledge Representation: A Survey.](http://arxiv.org/abs/2303.12023) | 本文总结了一种新的逻辑推理方法，它使用自然语言作为知识表示，具有不同于端到端神经方法的优势。这种新模式在未来有着很高的潜力。 |

# 详细

[^1]: 通过数据水印证明LLM预训练数据的成员资格

    Proving membership in LLM pretraining data via data watermarks

    [https://arxiv.org/abs/2402.10892](https://arxiv.org/abs/2402.10892)

    使用数据水印在LLM预训练中检测版权持有人作品的方法，可以进行合理检测且提供误检率保证，研究了水印设计对假设检验能力的影响以及在模型和数据集缩放下的检测强度变化。

    

    检测版权持有人的作品是否在LLM预训练中使用是一个重要问题，本文提出使用数据水印实现基于黑盒模型访问的合理检测，前提是版权持有人在公开发布之前贡献了多个训练文档并对其进行了水印处理。通过应用随机采样的数据水印，检测可以被构造为假设检验，从而提供对误检率的保证。研究了两种水印：一种插入随机序列，另一种随机用Unicode类似字符替换字符。首先展示了水印设计的三个方面--水印长度、复制次数和干扰--如何影响假设检验的能力。接着研究了水印在模型和数据集缩放下的检测强度如何变化：增加数据集大小会降低水印的强度，水印...

    arXiv:2402.10892v1 Announce Type: cross  Abstract: Detecting whether copyright holders' works were used in LLM pretraining is poised to be an important problem. This work proposes using data watermarks to enable principled detection with only black-box model access, provided that the rightholder contributed multiple training documents and watermarked them before public release. By applying a randomly sampled data watermark, detection can be framed as hypothesis testing, which provides guarantees on the false detection rate. We study two watermarks: one that inserts random sequences, and another that randomly substitutes characters with Unicode lookalikes. We first show how three aspects of watermark design -- watermark length, number of duplications, and interference -- affect the power of the hypothesis test. Next, we study how a watermark's detection strength changes under model and dataset scaling: while increasing the dataset size decreases the strength of the watermark, watermarks
    
[^2]: 指导多样性推动对未见任务的泛化

    Instruction Diversity Drives Generalization To Unseen Tasks

    [https://arxiv.org/abs/2402.10891](https://arxiv.org/abs/2402.10891)

    指导调整通过增加指令集的多样性来推动模型对未见任务的泛化。

    

    指导调整——在指令和期望结果之间微调大型语言模型（LLM）的方法——是一种使预训练语言模型执行现实世界任务并遵循人类指令的方法。其实际成功取决于模型学习比其训练时更广泛的指令集。然而，决定模型对这种“未见任务”的泛化的因素尚不十分清楚。为了了解泛化的驱动因素，本文通过字符串重写进行实验，这是一个符号任务，是图灵完整马尔可夫算法的基本组成部分，同时允许实验对“输入”和“指令”进行控制。我们调查了模型接受的指令数量和为每个指令提供的训练样本数量之间的权衡，并观察到指令集的多样性确定了泛化。

    arXiv:2402.10891v1 Announce Type: cross  Abstract: Instruction tuning -- fine-tuning a large language model (LLM) on pairs of instructions and desired outcomes -- is an approach that enables pre-trained language models to perform real-world tasks and follow human instructions. Its practical success depends on the model learning a broader set of instructions than those it was trained on. Yet the factors that determine model generalization to such \emph{unseen tasks} are not well understood. %To understand the driving factors of generalization, In this paper, we experiment with string rewrites, a symbolic task that serves as a building block for Turing complete Markov algorithms while allowing experimental control of "inputs" and "instructions". We investigate the trade-off between the number of instructions the model is trained on and the number of training samples provided for each instruction and observe that the diversity of the instruction set determines generalization. Generalizati
    
[^3]: LLM规划中树搜索何时有用？取决于鉴别器

    When is Tree Search Useful for LLM Planning? It Depends on the Discriminator

    [https://arxiv.org/abs/2402.10890](https://arxiv.org/abs/2402.10890)

    当前研究通过实验分析了大型语言模型在多步问题求解中使用树搜索的可行性，指出高级规划方法需要鉴别器至少90%准确性才能显著提高性能。

    

    在本文中，我们通过一个语言代理框架研究了大型语言模型（LLMs）如何在多步问题下解决问题，该框架包括生成器、鉴别器和规划方法三个部分。我们研究了两种先进规划方法，迭代校正和树搜索的实际效用。我们全面分析了鉴别准确性如何影响代理在使用这两种方法或更简单的重新排序方法时的整体性能。在两项任务，文本到SQL解析和数学推理上的实验表明：（1）高级规划方法需要至少90%准确性的鉴别器才能实现显著改进；（2）当前LLMs的鉴别能力尚未满足高级规划方法实现这种改进的需求；（3）采用基于LLM的鉴别器时，高级规划方法可能无法充分平衡准确性和效率。

    arXiv:2402.10890v1 Announce Type: cross  Abstract: In this paper, we examine how large language models (LLMs) solve multi-step problems under a language agent framework with three components: a generator, a discriminator, and a planning method. We investigate the practical utility of two advanced planning methods, iterative correction and tree search. We present a comprehensive analysis of how discrimination accuracy affects the overall performance of agents when using these two methods or a simpler method, re-ranking. Experiments on two tasks, text-to-SQL parsing and mathematical reasoning, show that: (1) advanced planning methods demand discriminators with at least 90% accuracy to achieve significant improvements over re-ranking; (2) current LLMs' discrimination abilities have not met the needs of advanced planning methods to achieve such improvements; (3) with LLM-based discriminators, advanced planning methods may not adequately balance accuracy and efficiency. For example, compare
    
[^4]: 通过提示生成优化评论生成

    Reviewer2: Optimizing Review Generation Through Prompt Generation

    [https://arxiv.org/abs/2402.10886](https://arxiv.org/abs/2402.10886)

    Reviewer2是一个高效的两阶段评论生成框架，通过明确建模评论可能涉及的各个方面的分布，生成更详细的评论，更好地涵盖人类审稿人在草稿中确定的各种方面。

    

    最近LLMs的发展为协助作者改进其作品提供了新机会。 本文设想了一个使用案例，即作者可以收到LLM生成的评论，揭示当前草稿中的弱点。 虽然已经存在用于自动生成评论的初始方法，但这些方法往往生成缺乏细节的评论，并且不能涵盖人类审稿人产生的各种意见。 为解决这一不足，我们提出了一种名为Reviewer2的高效二阶段评论生成框架。 与以往的工作不同，这种方法明确地模拟了评论可能涉及的各个方面的分布。 我们表明，这将导致更详细的评论，更好地涵盖人类审稿人在草稿中确定的各种方面。 作为研究的一部分，我们生成了一个包含27,000篇论文和99,000篇评论的大规模评论数据集，我们用方面提示进行了注释，并将其公开可用。

    arXiv:2402.10886v1 Announce Type: new  Abstract: Recent developments in LLMs offer new opportunities for assisting authors in improving their work. In this paper, we envision a use case where authors can receive LLM-generated reviews that uncover weak points in the current draft. While initial methods for automated review generation already exist, these methods tend to produce reviews that lack detail, and they do not cover the range of opinions that human reviewers produce. To address this shortcoming, we propose an efficient two-stage review generation framework called Reviewer2. Unlike prior work, this approach explicitly models the distribution of possible aspects that the review may address. We show that this leads to more detailed reviews that better cover the range of aspects that human reviewers identify in the draft. As part of the research, we generate a large-scale review dataset of 27k papers and 99k reviews that we annotate with aspect prompts, which we make available as a
    
[^5]: 多模式偏好对齐修复了语言模型在视觉指令调整上的回归

    Multi-modal preference alignment remedies regression of visual instruction tuning on language model

    [https://arxiv.org/abs/2402.10884](https://arxiv.org/abs/2402.10884)

    通过收集轻量级VQA偏好数据集并使用Direct Preference Optimization，我们能够在语言模型的指导能力上取得显著提升，在小规模数据下比其他方法实现了更高的分数。

    

    在实际应用中，多模式大型语言模型（MLLMs）被期望能够支持图像和文本模态的交换式多轮查询。然而，当前使用视觉问题回答（VQA）数据集训练的MLLMs可能会出现退化，因为VQA数据集缺乏原始文本指令数据集的多样性和复杂性，后者是底层语言模型训练的数据集。为了解决这一具有挑战性的退化问题，我们首先收集了一个轻量级（6k条记录）的VQA偏好数据集，其中答案由Gemini以细粒度方式注释了5个质量指标，然后研究了标准的监督微调、拒绝抽样、直接偏好优化（DPO）和SteerLM。我们的研究结果表明，通过DPO，我们能够超越语言模型的指导能力，实现了6.73的MT-Bench分数，而Vicuna的6.57和LLaVA的5.99，尽管数据规模较小。

    arXiv:2402.10884v1 Announce Type: cross  Abstract: In production, multi-modal large language models (MLLMs) are expected to support multi-turn queries of interchanging image and text modalities. However, the current MLLMs trained with visual-question-answering (VQA) datasets could suffer from degradation, as VQA datasets lack the diversity and complexity of the original text instruction datasets which the underlying language model had been trained with. To address this challenging degradation, we first collect a lightweight (6k entries) VQA preference dataset where answers were annotated by Gemini for 5 quality metrics in a granular fashion, and investigate standard Supervised Fine-tuning, rejection sampling, Direct Preference Optimization (DPO), and SteerLM. Our findings indicate that the with DPO we are able to surpass instruction-following capabilities of the language model, achieving a 6.73 score on MT-Bench, compared to Vicuna's 6.57 and LLaVA's 5.99 despite small data scale. This
    
[^6]: 通用提示优化器用于安全文本到图像生成

    Universal Prompt Optimizer for Safe Text-to-Image Generation

    [https://arxiv.org/abs/2402.10882](https://arxiv.org/abs/2402.10882)

    提出了第一个通用提示优化器，用于在黑盒场景中安全生成文本到图像，通过构建毒素-清洁提示对数据集，设计奖励函数，并通过 Proximal Policy Optimization 训练优化器，成功降低各种 T2I 模型生成不安全内容的可能性。

    

    文本到图像（T2I）模型在根据文字提示生成图像方面表现出色。然而，这些模型容易受到不安全输入的影响，从而生成不安全内容，如色情、骚扰和非法活动图像。基于图像检查器、模型微调和嵌入式阻止的现有研究在真实世界应用中不可行。因此，我们提出了第一个用于黑盒场景中安全 T2I 生成的通用提示优化器。

    arXiv:2402.10882v1 Announce Type: cross  Abstract: Text-to-Image (T2I) models have shown great performance in generating images based on textual prompts. However, these models are vulnerable to unsafe input to generate unsafe content like sexual, harassment and illegal-activity images. Existing studies based on image checker, model fine-tuning and embedding blocking are impractical in real-world applications. Hence, \textit{we propose the first universal prompt optimizer for safe T2I generation in black-box scenario}. We first construct a dataset consisting of toxic-clean prompt pairs by GPT-3.5 Turbo. To guide the optimizer to have the ability of converting toxic prompt to clean prompt while preserving semantic information, we design a novel reward function measuring toxicity and text alignment of generated images and train the optimizer through Proximal Policy Optimization. Experiments show that our approach can effectively reduce the likelihood of various T2I models in generating in
    
[^7]: EcoRank: 使用大型语言模型进行受限预算文本重新排序

    EcoRank: Budget-Constrained Text Re-ranking Using Large Language Models

    [https://arxiv.org/abs/2402.10866](https://arxiv.org/abs/2402.10866)

    EcoRank是一个两层管线，通过联合优化有关预算分配和LLM API的决策来实现文本重新排序，在实验中表现优于其他预算感知方法。

    

    大型语言模型（LLMs）在文本重新排序中取得了最先进的性能。该过程包括在提示中使用查询和候选段落，利用点对点，列表式和成对提示策略。LLMs的这些排序策略的一个限制是它们的成本：由于API收费基于输入和输出令牌的数量，这个过程可能会变得昂贵。我们研究如何在给定预算的情况下最大化重新排序性能，通过导航提示选择，LLM API和预算分割的广阔搜索空间。我们提出了一套使用一组LLM API进行文本重新排序的受限预算方法。我们最有效的方法是EcoRank，它是一个两层管线，可以联合优化有关跨提示策略和LLM API的预算分配决策。我们在四个流行的QA和段重排序数据集上的实验结果显示，EcoRank优于其他具有预算意识的方法。

    arXiv:2402.10866v1 Announce Type: new  Abstract: Large Language Models (LLMs) have achieved state-of-the-art performance in text re-ranking. This process includes queries and candidate passages in the prompts, utilizing pointwise, listwise, and pairwise prompting strategies. A limitation of these ranking strategies with LLMs is their cost: the process can become expensive due to API charges, which are based on the number of input and output tokens. We study how to maximize the re-ranking performance given a budget, by navigating the vast search spaces of prompt choices, LLM APIs, and budget splits. We propose a suite of budget-constrained methods to perform text re-ranking using a set of LLM APIs. Our most efficient method, called EcoRank, is a two-layered pipeline that jointly optimizes decisions regarding budget allocation across prompt strategies and LLM APIs. Our experimental results on four popular QA and passage reranking datasets show that EcoRank outperforms other budget-aware 
    
[^8]: LLMs下的时间序列预测：理解和增强模型能力

    Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities

    [https://arxiv.org/abs/2402.10835](https://arxiv.org/abs/2402.10835)

    本研究通过比较LLMs与传统模型，发现了LLMs在时间序列预测中的优势和局限性，指出LLMs在预测具有明显模式和趋势的时间序列方面表现出色，但在缺乏周期性的数据集方面面临挑战，同时指出融入外部知识和采用自然语言释义有助于提升LLMs在时间序列预测中的性能。

    

    大语言模型(LLMs)近年来在许多领域得到迅速发展。作为一种经典的机器学习任务，时间序列预测最近从LLMs中获得了推动。然而，在这一领域，LLMs的偏好存在研究空白。通过将LLMs与传统模型进行比较，发现了LLMs在时间序列预测中的许多特性。例如，我们的研究表明，LLMs在预测具有明显模式和趋势的时间序列方面表现出色，但在缺乏周期性的数据集方面面临挑战。我们通过设计提示要求LLMs告知数据集的周期来解释我们的发现。此外，本文还研究了输入策略，发现融入外部知识和采用自然语言释义积极影响了LLMs在时间序列预测中的预测性能。总的来说，这项研究有助于洞察LLMs在时间序列预测中的优势和局限性。

    arXiv:2402.10835v1 Announce Type: new  Abstract: Large language models (LLMs) have been applied in many fields with rapid development in recent years. As a classic machine learning task, time series forecasting has recently received a boost from LLMs. However, there is a research gap in the LLMs' preferences in this field. In this paper, by comparing LLMs with traditional models, many properties of LLMs in time series prediction are found. For example, our study shows that LLMs excel in predicting time series with clear patterns and trends but face challenges with datasets lacking periodicity. We explain our findings through designing prompts to require LLMs to tell the period of the datasets. In addition, the input strategy is investigated, and it is found that incorporating external knowledge and adopting natural language paraphrases positively affects the predictive performance of LLMs for time series. Overall, this study contributes to insight into the advantages and limitations of
    
[^9]: 通过基于程序提示的方式探索混合式问答

    Exploring Hybrid Question Answering via Program-based Prompting

    [https://arxiv.org/abs/2402.10812](https://arxiv.org/abs/2402.10812)

    提出了HProPro，一个基于程序提示的框架，用于处理混合式问答任务，通过代码生成和执行范式以及各种函数来应对混合推理场景。

    

    在异构数据上进行问答需要对各种数据来源进行推理，这是挑战性的，因为信息量大且异构数据有机耦合。已经提出了各种方法来解决这些挑战。其中一种方法涉及训练专门的检索器来选择相关信息，从而减少输入长度。另一种方法是将数据的不同形式转换为单一形式，简化任务难度并实现更简单的处理。在本文中，我们提出了HProPro，这是一个面向混合式问答任务的新型基于程序提示的框架。HProPro遵循代码生成和执行范式。此外，HProPro集成了各种函数以应对混合推理场景。具体来说，HProPro 包含函数声明和函数实现，以对来自各种数据来源的混合信息进行检索。

    arXiv:2402.10812v1 Announce Type: new  Abstract: Question answering over heterogeneous data requires reasoning over diverse sources of data, which is challenging due to the large scale of information and organic coupling of heterogeneous data. Various approaches have been proposed to address these challenges. One approach involves training specialized retrievers to select relevant information, thereby reducing the input length. Another approach is to transform diverse modalities of data into a single modality, simplifying the task difficulty and enabling more straightforward processing. In this paper, we propose HProPro, a novel program-based prompting framework for the hybrid question answering task. HProPro follows the code generation and execution paradigm. In addition, HProPro integrates various functions to tackle the hybrid reasoning scenario. Specifically, HProPro contains function declaration and function implementation to perform hybrid information-seeking over data from vario
    
[^10]: 在LLM模拟中量化Persona效应

    Quantifying the Persona Effect in LLM Simulations

    [https://arxiv.org/abs/2402.10811](https://arxiv.org/abs/2402.10811)

    本研究探讨了人物变量对LLMs模拟不同视角能力的影响，发现人物变量在现有主观NLP数据集中解释能力有限，但通过提示方式加入可以略微改善模型预测，尤其在存在争议但范围有限的数据样本上效果最好。

    

    大型语言模型（LLMs）在模拟人类语言使用和行为方面表现出显著的潜力。在这项研究中，我们深入探讨了人物变量与LLMs模拟不同视角的能力的交集。我们发现人物变量可以解释现有主观NLP数据集中<10\%的注释变异。然而，通过提示在LLMs中加入他们能带来适度的改进。Persona提示在注释者之间存在争议但范围有限的数据样本上效果最好。存在线性相关性：人格变量对人类注释的影响越大，LLMs使用Persona提示的预测就越好。然而，当人物变量的效用较低（即解释人类注释的<10\%）时，Persona提示几乎没有影响。大多数主观NLP数据集都属于这一类别，对模拟多元视角产生怀疑。

    arXiv:2402.10811v1 Announce Type: new  Abstract: Large language models (LLMs) have shown remarkable promise in simulating human language use and behavior. In this study, we delve into the intersection of persona variables and the capability of LLMs to simulate different perspectives. We find that persona variables can explain <10\% variance in annotations in existing subjective NLP datasets. Nonetheless, incorporating them via prompting in LLMs provides modest improvement. Persona prompting is most effective on data samples where disagreements among annotators are frequent yet confined to a limited range. A linear correlation exists: the more persona variables influence human annotations, the better LLMs predictions are using persona prompting. However, when the utility of persona variables is low (i.e., explaining <10\% of human annotations), persona prompting has little effect. Most subjective NLP datasets fall into this category, casting doubt on simulating diverse perspectives in t
    
[^11]: 生成式跨模态检索：在多模态语言模型中存储图像用于检索及更多应用

    Generative Cross-Modal Retrieval: Memorizing Images in Multimodal Language Models for Retrieval and Beyond

    [https://arxiv.org/abs/2402.10805](https://arxiv.org/abs/2402.10805)

    提出了一种生成式跨模态检索框架，在多模态语言模型中实现了存储和检索图像的能力

    

    近期生成式语言模型的进展表明其能够记忆文档中的知识并有效地回答用户查询。在此能力基础上，我们提出了使多模态大型语言模型（MLLMs）能够在其参数内存储和检索图像的方法。给定用户对视觉内容的查询，MLLM被期望能够从其参数中“回忆”相关图像作为响应。实现这一目标面临着显著挑战，其中包括MLLM内置的视觉记忆和视觉检索方案。为解决这些挑战，我们引入了一个生成式跨模态检索框架，该框架为图像分配唯一标识符字符串，并涉及两个训练步骤：学习记忆和学习检索。第一步侧重于训练MLLM记忆图像与其标识符之间的关联。

    arXiv:2402.10805v1 Announce Type: cross  Abstract: The recent advancements in generative language models have demonstrated their ability to memorize knowledge from documents and recall knowledge to respond to user queries effectively. Building upon this capability, we propose to enable multimodal large language models (MLLMs) to memorize and recall images within their parameters. Given a user query for visual content, the MLLM is anticipated to "recall" the relevant image from its parameters as the response. Achieving this target presents notable challenges, including inbuilt visual memory and visual recall schemes within MLLMs. To address these challenges, we introduce a generative cross-modal retrieval framework, which assigns unique identifier strings to represent images and involves two training steps: learning to memorize and learning to retrieve. The first step focuses on training the MLLM to memorize the association between images and their respective identifiers. The latter ste
    
[^12]: 在一个 1000 万根草垛中寻找针：循环记忆找到了语言模型不擅长的内容

    In Search of Needles in a 10M Haystack: Recurrent Memory Finds What LLMs Miss

    [https://arxiv.org/abs/2402.10790](https://arxiv.org/abs/2402.10790)

    通过使用循环记忆增强对 GPT-2 进行微调，使其能够处理长达 1000 万个元素的任务，这是迄今为止处理最长输入的开放神经网络模型，并展示了对长序列处理能力的显著改进。

    

    本文解决了使用生成式 Transformer 模型处理长文档的挑战。为了评估不同方法，我们引入了 BABILong，这是一个新的基准，旨在评估模型在提取和处理广泛文本中分布式事实方面的能力。我们的评估包括 GPT-4 和 RAG 的基准，结果显示常见方法仅适用于最多 $10^4$ 个元素的序列。相反，通过使用循环记忆增强对 GPT-2 进行微调，使其能够处理涉及最多 $10^7$ 个元素的任务。这一成就标志着迄今为止任何开源神经网络模型处理的最长输入，显示了对长序列处理能力的显著改进。

    arXiv:2402.10790v1 Announce Type: cross  Abstract: This paper addresses the challenge of processing long documents using generative transformer models. To evaluate different approaches, we introduce BABILong, a new benchmark designed to assess model capabilities in extracting and processing distributed facts within extensive texts. Our evaluation, which includes benchmarks for GPT-4 and RAG, reveals that common methods are effective only for sequences up to $10^4$ elements. In contrast, fine-tuning GPT-2 with recurrent memory augmentations enables it to handle tasks involving up to $10^7$ elements. This achievement marks a substantial leap, as it is by far the longest input processed by any open neural network model to date, demonstrating a significant improvement in the processing capabilities for long sequences.
    
[^13]: EdgeQAT: 熵和分布引导的量化感知训练，用于加速轻量级LLMs在边缘设备上的应用

    EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

    [https://arxiv.org/abs/2402.10787](https://arxiv.org/abs/2402.10787)

    本文提出了EdgeQAT，使用熵和分布引导的量化感知训练方法来优化轻量级LLMs，在边缘设备上实现推理加速。

    

    尽管大型语言模型（LLMs）在各个领域取得了显著进展，但由于其庞大的参数和计算量，LLMs在边缘设备上的广泛应用受到限制。为了解决这一问题，通常采用量化方法生成具有高效计算和快速推理的轻量级LLMs。然而，后训练量化（PTQ）方法在将权重、激活和KV缓存一起量化至8位以下时，质量会急剧下降。此外，许多量化感知训练（QAT）工作对模型权重进行量化，而激活未被触及，这不能充分发挥量化对边缘端推理加速的潜力。在本文中，我们提出了EdgeQAT，即熵和分布引导的QAT，用于优化轻量级LLMs以实现在边缘设备上的推理加速。我们首先确定量化性能下降主要源自信息

    arXiv:2402.10787v1 Announce Type: cross  Abstract: Despite the remarkable strides of Large Language Models (LLMs) in various fields, the wide applications of LLMs on edge devices are limited due to their massive parameters and computations. To address this, quantization is commonly adopted to generate lightweight LLMs with efficient computations and fast inference. However, Post-Training Quantization (PTQ) methods dramatically degrade in quality when quantizing weights, activations, and KV cache together to below 8 bits. Besides, many Quantization-Aware Training (QAT) works quantize model weights, leaving the activations untouched, which do not fully exploit the potential of quantization for inference acceleration on the edge. In this paper, we propose EdgeQAT, the Entropy and Distribution Guided QAT for the optimization of lightweight LLMs to achieve inference acceleration on Edge devices. We first identify that the performance drop of quantization primarily stems from the information
    
[^14]: 一种用于零样本链接预测的紧凑转换图框架与大型语言模型

    A Condensed Transition Graph Framework for Zero-shot Link Prediction with Large Language Models

    [https://arxiv.org/abs/2402.10779](https://arxiv.org/abs/2402.10779)

    提出了一种用于零样本链接预测的紧凑转换图框架，能够在线性时间内编码所有路径的信息，并可解决大型语言模型性能受限的问题。

    

    零样本链接预测（ZSLP）旨在自动识别给定实体之间的关系。现有方法主要利用辅助信息来预测给定头实体和其关系时的尾实体，然而面临挑战，原因是有时缺乏这些详细信息，并且基于语义相似性来预测尾实体的固有简单性。尽管大型语言模型（LLMs）为以零样本方式预测头实体和尾实体之间的未观察到的关系提供了有前途的解决方案，但其性能仍受限于无法利用两个实体之间所有（指数多）路径信息的能力，这些信息对于共同指示它们的关系类型至关重要。为了解决这个问题，在这项工作中，我们引入了一种用于零样本链接预测的紧凑转换图框架（CTLP），它以线性时间编码了所有路径的信息。

    arXiv:2402.10779v1 Announce Type: new  Abstract: Zero-shot link prediction (ZSLP) on knowledge graphs aims at automatically identifying relations between given entities. Existing methods primarily employ auxiliary information to predict tail entity given head entity and its relation, yet face challenges due to the occasional unavailability of such detailed information and the inherent simplicity of predicting tail entities based on semantic similarities. Even though Large Language Models (LLMs) offer a promising solution to predict unobserved relations between the head and tail entity in a zero-shot manner, their performance is still restricted due to the inability to leverage all the (exponentially many) paths' information between two entities, which are critical in collectively indicating their relation types. To address this, in this work, we introduce a Condensed Transition Graph Framework for Zero-Shot Link Prediction (CTLP), which encodes all the paths' information in linear time
    
[^15]: 通过早期融合和多语言模型增强ESG影响类型识别

    Enhancing ESG Impact Type Identification through Early Fusion and Multilingual Models

    [https://arxiv.org/abs/2402.10772](https://arxiv.org/abs/2402.10772)

    通过早期融合和多语言模型，提出了一个集成学习的系统，可以最佳地识别ESG影响类型，为当今金融和企业治理领域中的负责任和可持续决策过程做出贡献。

    

    在不断发展的环境、社会和企业治理(ESG)影响评估领域，ML-ESG-2共享任务提出了识别ESG影响类型的挑战。为了解决这一挑战，我们提出了一个综合系统，利用集成学习技术，利用早期和后期融合方法。我们的方法采用了四种不同的模型：mBERT、FlauBERT-base、ALBERT-base-v2和结合潜在语义分析（LSA）和词频-逆文档频率（TF-IDF）特征的多层感知器(MLP)。通过大量实验，我们发现我们的早期融合集成方法，在LSA、TF-IDF、mBERT、FlauBERT-base和ALBERT-base-v2的整合下，取得了最佳性能。我们的系统提供了一个全面的ESG影响类型识别解决方案，促进了当今金融和企业治理领域中至关重要的负责任和可持续决策过程。

    arXiv:2402.10772v1 Announce Type: new  Abstract: In the evolving landscape of Environmental, Social, and Corporate Governance (ESG) impact assessment, the ML-ESG-2 shared task proposes identifying ESG impact types. To address this challenge, we present a comprehensive system leveraging ensemble learning techniques, capitalizing on early and late fusion approaches. Our approach employs four distinct models: mBERT, FlauBERT-base, ALBERT-base-v2, and a Multi-Layer Perceptron (MLP) incorporating Latent Semantic Analysis (LSA) and Term Frequency-Inverse Document Frequency (TF-IDF) features. Through extensive experimentation, we find that our early fusion ensemble approach, featuring the integration of LSA, TF-IDF, mBERT, FlauBERT-base, and ALBERT-base-v2, delivers the best performance. Our system offers a comprehensive ESG impact type identification solution, contributing to the responsible and sustainable decision-making processes vital in today's financial and corporate governance landsca
    
[^16]: 自动评估方法在面向指令的LLM中有多可靠？

    How Reliable Are Automatic Evaluation Methods for Instruction-Tuned LLMs?

    [https://arxiv.org/abs/2402.10770](https://arxiv.org/abs/2402.10770)

    本文研究了面向指令的大型语言模型中自动评估方法的可靠性，发现自动方法在不同任务类型下与人工评估者之间的相关性存在巨大变化，且在自由形式生成任务和跨语言转移中可能不可靠。

    

    面向指令的大型语言模型(LLMs)的研究使用基于文本重叠和LLM判断的自动方法作为人工评估的成本有效替代方案。本文研究了这些方法在广泛的任务范围和跨语言环境中的可靠性。与先前的研究结果相反，我们观察到在任务类型不同的情况下，自动方法与人工评估者之间的相关性存在显著变化。具体而言，广泛使用的ROUGE-L度量在短答案英语任务中与人类判断强相关，但在自由形式生成任务和跨语言转移中不可靠。使用GPT-4作为评估员的有效性取决于在要求评估时包含参考答案，这可能导致在自由形式生成任务中评估过于严格。总的来说，我们发现，尽管自动评估方法可以近似人类判断，但其准确性可能因任务类型和评估设置而异。

    arXiv:2402.10770v1 Announce Type: cross  Abstract: Work on instruction-tuned Large Language Models (LLMs) has used automatic methods based on text overlap and LLM judgments as cost-effective alternatives to human evaluation. In this paper, we study the reliability of such methods across a broad range of tasks and in a cross-lingual setting. In contrast to previous findings, we observe considerable variability in correlations between automatic methods and human evaluators when scores are differentiated by task type. Specifically, the widely-used ROUGE-L metric strongly correlates with human judgments for short-answer English tasks but is unreliable in free-form generation tasks and cross-lingual transfer. The effectiveness of GPT-4 as an evaluator depends on including reference answers when prompting for assessments, which can lead to overly strict evaluations in free-form generation tasks. In summary, we find that, while automatic evaluation methods can approximate human judgements und
    
[^17]: 蒸馏增强生成式检索

    Distillation Enhanced Generative Retrieval

    [https://arxiv.org/abs/2402.10769](https://arxiv.org/abs/2402.10769)

    通过蒸馏方法增强生成式检索系统，提出了一种名为DGR的框架，利用先进排名模型和蒸馏RankNet损失来优化模型。

    

    生成式检索是文本检索中的一种新兴范式，通过生成相关段落的标识符字符串作为检索目标。该范式利用强大的生成式语言模型，不同于传统的稀疏或密集检索方法。本研究确定了通过蒸馏进一步增强生成式检索的可行方向，并提出了一个名为DGR的可行框架。DGR利用诸如跨编码器等先进排名模型，在教师角色中提供段落排名列表，捕获段落的不同相关程度，而不是二元硬标签；随后，DGR采用一种特别设计的蒸馏RankNet损失来优化生成式检索模型，考虑教师模型提供的段落排名顺序作为标签。该框架仅需要额外的蒸馏步骤来增强当前的生成式检索系统，并不增加任何负担。

    arXiv:2402.10769v1 Announce Type: cross  Abstract: Generative retrieval is a promising new paradigm in text retrieval that generates identifier strings of relevant passages as the retrieval target. This paradigm leverages powerful generative language models, distinct from traditional sparse or dense retrieval methods. In this work, we identify a viable direction to further enhance generative retrieval via distillation and propose a feasible framework, named DGR. DGR utilizes sophisticated ranking models, such as the cross-encoder, in a teacher role to supply a passage rank list, which captures the varying relevance degrees of passages instead of binary hard labels; subsequently, DGR employs a specially designed distilled RankNet loss to optimize the generative retrieval model, considering the passage rank order provided by the teacher model as labels. This framework only requires an additional distillation step to enhance current generative retrieval systems and does not add any burden
    
[^18]: 大型语言模型中的最佳解释推断

    Inference to the Best Explanation in Large Language Models

    [https://arxiv.org/abs/2402.10767](https://arxiv.org/abs/2402.10767)

    该论文提出了一个受哲学启发设计的框架IBE-Eval，用于推进对大型语言模型解释的解释和评估，在因果问答实验中显示出高达77%的准确率。

    

    虽然大型语言模型（LLMs）在现实应用中取得了成功，但它们的基本解释过程仍然知之甚少。本文提出了IBE-Eval，这是一个受哲学关于最佳解释推断（IBE）的启发而设计的框架，旨在推进对LLMs解释的解释和评估。IBE-Eval通过结合包括一致性、简洁性、连贯性和不确定性在内的显式逻辑和语言特征来估计自然语言解释的合理性。在因果问答（CQA）领域进行了大量实验，其中IBE-Eval被要求在多个由LLMs（即GPT 3.5和Llama 2）生成的竞争性因果解释中选择最合理的因果解释。实验证明，IBE-Eval可以成功地以高达77\%的准确率（比随机高约27%）识别最佳解释，优于GPT 3.5作为判定基线的表现。

    arXiv:2402.10767v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) have found success in real-world applications, their underlying explanatory process is still poorly understood. This paper proposes IBE-Eval, a framework inspired by philosophical accounts on Inference to the Best Explanation (IBE) to advance the interpretation and evaluation of LLMs' explanations. IBE-Eval estimates the plausibility of natural language explanations through a combination of explicit logical and linguistic features including: consistency, parsimony, coherence, and uncertainty. Extensive experiments are conducted on Causal Question Answering (CQA), where \textit{IBE-Eval} is tasked to select the most plausible causal explanation amongst competing ones generated by LLMs (i.e., GPT 3.5 and Llama 2). The experiments reveal that IBE-Eval can successfully identify the best explanation with up to 77\% accuracy ($\approx 27\%$ above random), improving upon a GPT 3.5-as-a-Judge baseline ($\appr
    
[^19]: ToolSword：揭示大型语言模型在工具学习中的安全问题跨三个阶段

    ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages

    [https://arxiv.org/abs/2402.10753](https://arxiv.org/abs/2402.10753)

    ToolSword提出了一个专门用于细致调查大型语言模型在工具学习中安全问题的全面框架，揭示了在工具学习中持久存在的安全挑战。

    

    arXiv:2402.10753v1 公告类型：跨领域 抽象：工具学习被广泛认为是在现实场景中部署大型语言模型（LLMs）的基础方法。尽管当前研究主要强调利用工具来增强LLMs，但它经常忽视与其应用相关的新兴安全考虑。为填补这一空白，我们提出了$ToolSword$，这是一个致力于细致调查LLMs在工具学习中安全问题的全面框架。具体来说，ToolSword勾画了LLMs在工具学习中的六个安全场景，包括输入阶段的$恶意$ $查询$和$越狱$ $攻击$，执行阶段的$噪声$ $误导$和$风险$ $线索$，以及输出阶段的$有害$ $反馈$和$错误$ $冲突$。对11个开源和闭源LLMs进行的实验表明，在工具学习中存在持久的安全挑战，如处理有害查询、使用风险工具和提供有害反馈。

    arXiv:2402.10753v1 Announce Type: cross  Abstract: Tool learning is widely acknowledged as a foundational approach or deploying large language models (LLMs) in real-world scenarios. While current research primarily emphasizes leveraging tools to augment LLMs, it frequently neglects emerging safety considerations tied to their application. To fill this gap, we present $ToolSword$, a comprehensive framework dedicated to meticulously investigating safety issues linked to LLMs in tool learning. Specifically, ToolSword delineates six safety scenarios for LLMs in tool learning, encompassing $malicious$ $queries$ and $jailbreak$ $attacks$ in the input stage, $noisy$ $misdirection$ and $risky$ $cues$ in the execution stage, and $harmful$ $feedback$ and $error$ $conflicts$ in the output stage. Experiments conducted on 11 open-source and closed-source LLMs reveal enduring safety challenges in tool learning, such as handling harmful queries, employing risky tools, and delivering detrimental feedb
    
[^20]: GenRES：在大语言模型时代重新思考生成式关系抽取的评估

    GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models

    [https://arxiv.org/abs/2402.10744](https://arxiv.org/abs/2402.10744)

    GenRES提出了一种多维度评估生成式关系抽取结果的方法，填补了使用传统指标评估GRE方法时的不足之处。

    

    关系抽取（RE）领域正朝着利用大语言模型（LLM）的能力的生成式关系抽取（GRE）方向发生显着转变。然而，我们发现传统的关系抽取（RE）指标如精确率和召回率在评估GRE方法时存在不足。这种不足的原因在于这些指标依赖于与人工注释的参考关系的精确匹配，而GRE方法通常会产生与参考不同的多样且语义准确的关系。为填补这一空白，我们提出了GenRES，以多维度评估GRE结果的主题相似性、独特性、粒度、真实性和完整性。通过GenRES，我们实证发现：（1）精确率/召回率不能充分证明GRE方法的性能；（2）人工注释的参考关系可能存在不完整情况；（3）以固定一组关系或实体提示LLM

    arXiv:2402.10744v1 Announce Type: cross  Abstract: The field of relation extraction (RE) is experiencing a notable shift towards generative relation extraction (GRE), leveraging the capabilities of large language models (LLMs). However, we discovered that traditional relation extraction (RE) metrics like precision and recall fall short in evaluating GRE methods. This shortfall arises because these metrics rely on exact matching with human-annotated reference relations, while GRE methods often produce diverse and semantically accurate relations that differ from the references. To fill this gap, we introduce GenRES for a multi-dimensional assessment in terms of the topic similarity, uniqueness, granularity, factualness, and completeness of the GRE results. With GenRES, we empirically identified that (1) precision/recall fails to justify the performance of GRE methods; (2) human-annotated referential relations can be incomplete; (3) prompting LLMs with a fixed set of relations or entities
    
[^21]: 通过文本挖掘和自然语言处理研究构建《易水学派》句法分析图

    Construction of a Syntactic Analysis Map for Yi Shui School through Text Mining and Natural Language Processing Research

    [https://arxiv.org/abs/2402.10743](https://arxiv.org/abs/2402.10743)

    通过条件随机场构建了基于自然语言处理技术框架下的《易水学派》句法分析图，实现了传统中医药文本的实体关系提取和关键信息提取。

    

    实体和关系提取是自然语言处理任务中至关重要的组成部分，如知识图谱构建、问答系统设计和语义分析。传统中医学《易水学派》的大部分信息以非结构化的古典汉语文本形式存储。中医学文本的关键信息提取在挖掘和研究中医学学术派别方面发挥着重要作用。为了有效地利用人工智能方法解决这些问题，本研究在自然语言处理技术框架下构建了基于条件随机场的分词和实体关系提取模型，以识别和提取传统中医药文本的实体关系，并利用TF-IDF信息检索和数据挖掘的常见加权技术提取不同的重要关键实体信息。

    arXiv:2402.10743v1 Announce Type: new  Abstract: Entity and relationship extraction is a crucial component in natural language processing tasks such as knowledge graph construction, question answering system design, and semantic analysis. Most of the information of the Yishui school of traditional Chinese Medicine (TCM) is stored in the form of unstructured classical Chinese text. The key information extraction of TCM texts plays an important role in mining and studying the academic schools of TCM. In order to solve these problems efficiently using artificial intelligence methods, this study constructs a word segmentation and entity relationship extraction model based on conditional random fields under the framework of natural language processing technology to identify and extract the entity relationship of traditional Chinese medicine texts, and uses the common weighting technology of TF-IDF information retrieval and data mining to extract important key entity information in different
    
[^22]: 让我们一步一步学习：通过课程学习增强上下文学习能力

    Let's Learn Step by Step: Enhancing In-Context Learning Ability with Curriculum Learning

    [https://arxiv.org/abs/2402.10738](https://arxiv.org/abs/2402.10738)

    通过少样本上下文课程学习（ICCL）方法，逐渐增加提示演示的复杂性，有效提高了大型语言模型（LLMs）的性能，实验结果显示ICCL对开源LLMs有效。

    

    演示排序是上下文学习（ICL）的重要策略，可以显著影响大型语言模型（LLMs）的性能。然而，大部分当前的排序方法需要额外的知识和相似性计算。我们倡导少样本上下文课程学习（ICCL），这是一种简单而有效的ICL演示排序方法，其暗示在推理过程中逐渐增加提示演示的复杂性。然后，我们设计了三个实验，讨论ICCL的有效性，LLM的ICCL能力形成机制以及排序主题的影响。实验结果表明，ICCL在指导调整阶段开发，对于开源LLMs是有效的。此外，LLMs在辨别演示难度级别方面表现出比人类更弱的能力。我们在https://github.com/61peng/cu发布了我们的代码。

    arXiv:2402.10738v1 Announce Type: new  Abstract: Demonstration ordering, which is an important strategy for in-context learning (ICL), can significantly affects the performance of large language models (LLMs). However, most of the current approaches of ordering require additional knowledge and similarity calculation. We advocate the few-shot in-context curriculum learning (ICCL), a simple but effective demonstration ordering method for ICL, which implies gradually increasing the complexity of prompt demonstrations during the inference process. Then we design three experiments to discuss the effectiveness of ICCL, the formation mechanism of LLM's ICCL capability, and the impact of ordering subjects. Experimental results demonstrate that ICCL, developed during the instruction-tuning stage, is effective for open-source LLMs. Moreover, LLMs exhibit a weaker capacity compared to humans in discerning the difficulty levels of demonstrations. We release our code at https://github.com/61peng/cu
    
[^23]: 在声明验证的背景下评估ChatGPT的推理能力

    Assessing the Reasoning Abilities of ChatGPT in the Context of Claim Verification

    [https://arxiv.org/abs/2402.10735](https://arxiv.org/abs/2402.10735)

    我们提出了一个逻辑推理框架，用于评估ChatGPT在声明验证中的推理能力，发现其在归纳推理方面存在困难，并提出了一种缓解方法。

    

    当前有关LLMs的推理能力的辩论正在日益激烈。我们从声明/谣言验证的角度来审视这个问题。我们提出了第一个逻辑推理框架，旨在将任何声明或传言与证据结合，拆分成验证所需的基本推理步骤。基于我们的框架，我们整理了两个注释集合，其中包括来自维基百科的合成数据集和源自Twitter上流传的谣言的真实数据集。我们使用它们来评估GPT-3.5-Turbo和GPT-4（以下简称为ChatGPT）在我们框架的背景下的推理能力，并提供了彻底的分析。我们的研究表明，ChatGPT在归纳推理方面存在困难，尽管可以通过使用手动的思维链路（Chain of Thought，CoT）来缓解这一问题，而非零编码（Zero Shot，ZS）和ZS CoT方法。我们的研究有助于不断增长的研究领域，表明Cha

    arXiv:2402.10735v1 Announce Type: new  Abstract: The reasoning capabilities of LLMs are currently hotly debated. We examine the issue from the perspective of claim/rumour verification. We propose the first logical reasoning framework designed to break down any claim or rumor paired with evidence into the atomic reasoning steps necessary for verification. Based on our framework, we curate two annotated collections of such claim/evidence pairs: a synthetic dataset from Wikipedia and a real-world set stemming from rumours circulating on Twitter. We use them to evaluate the reasoning capabilities of GPT-3.5-Turbo and GPT-4 (hereinafter referred to as ChatGPT) within the context of our framework, providing a thorough analysis. Our results show that ChatGPT struggles in abductive reasoning, although this can be somewhat mitigated by using manual Chain of Thought (CoT) as opposed to Zero Shot (ZS) and ZS CoT approaches. Our study contributes to the growing body of research suggesting that Cha
    
[^24]: 一项关于跨语言词汇适应用于高效生成LLM推理的实证研究

    An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Generative LLM Inference

    [https://arxiv.org/abs/2402.10712](https://arxiv.org/abs/2402.10712)

    通过实证研究，本文探讨了各种跨语言词汇适应方法对提高生成LLM推理效率的影响。

    

    arXiv:2402.10712v1 通告类型: 跨领域 摘要: 最先进的生成大型语言模型(LLMs)的发展在很大程度上依赖于英语为中心的分词器、词汇和预训练数据。尽管一些LLMs具有多语言能力，但最近的研究表明，当生成英语以外的其他语言时，它们的推理效率会下降。这导致推理时间和成本增加。已经提出了跨语言词汇适应方法，用于将模型调整到目标语言，旨在提高下游性能。然而，这些方法对提高生成LLM推理效率的有效性尚未得到探究。在本文中，我们对五种生成LLMs（包括单语和多语模型）在四种语言类型多样且四种自然语言理解任务上进行了各种跨语言词汇适应方法的实证研究。

    arXiv:2402.10712v1 Announce Type: cross  Abstract: The development of state-of-the-art generative large language models (LLMs) disproportionately relies on English-centric tokenizers, vocabulary and pre-training data. Despite the fact that some LLMs have multilingual capabilities, recent studies have shown that their inference efficiency deteriorates when generating text in languages other than English. This results in increased inference time and costs. Cross-lingual vocabulary adaptation methods have been proposed for adapting models to a target language aiming to improve downstream performance. However, the effectiveness of these methods on increasing inference efficiency of generative LLMs has yet to be explored. In this paper, we perform an empirical study of various cross-lingual vocabulary adaptation methods on five generative LLMs (including monolingual and multilingual models) across four typologically-diverse languages and four natural language understanding tasks. We find th
    
[^25]: 重新思考类人翻译策略：将漂移扩散模型与大型语言模型集成用于机器翻译

    Rethinking Human-like Translation Strategy: Integrating Drift-Diffusion Model with Large Language Models for Machine Translation

    [https://arxiv.org/abs/2402.10699](https://arxiv.org/abs/2402.10699)

    将Thinker与漂移扩散模型集成，重新定义漂移扩散过程以模拟人类翻译者的决策制定，实验证明在机器翻译中取得了优异成绩。

    

    大型语言模型（LLMs）在包括机器翻译在内的各种下游任务中展现出了巨大潜力。然而，基于LLM的机器翻译先前的工作主要集中在更好地利用训练数据、演示版本或预定义的普遍知识来提高性能，缺乏对类似人类翻译者的决策制定的考虑。本文将“Thinker”与漂移扩散模型（Thinker-DDM）相结合，以解决这一问题。然后，我们重新定义了漂移扩散过程，以模拟受限资源情况下类人翻译者的动态决策制定。我们在高资源、低资源和常识翻译设置下，使用WMT22和CommonMT数据集进行了大量实验，在前两种场景中，Thinker-DDM的表现优于基准。我们还对常识翻译进行了额外的分析和评估，以说明其高效性。

    arXiv:2402.10699v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated promising potential in various downstream tasks, including machine translation. However, prior work on LLM-based machine translation has mainly focused on better utilizing training data, demonstrations, or pre-defined and universal knowledge to improve performance, with a lack of consideration of decision-making like human translators. In this paper, we incorporate Thinker with the Drift-Diffusion Model (Thinker-DDM) to address this issue. We then redefine the Drift-Diffusion process to emulate human translators' dynamic decision-making under constrained resources. We conduct extensive experiments under the high-resource, low-resource, and commonsense translation settings using the WMT22 and CommonMT datasets, in which Thinker-DDM outperforms baselines in the first two scenarios. We also perform additional analysis and evaluation on commonsense translation to illustrate the high effectivenes
    
[^26]: 探索精度和召回率以评估LLMs的质量和多样性

    Exploring Precision and Recall to assess the quality and diversity of LLMs

    [https://arxiv.org/abs/2402.10693](https://arxiv.org/abs/2402.10693)

    该研究提出了一种新的评估框架，将精度和召回率指标从图像生成转化为文本生成，细致评估了LLMs生成文本的质量和多样性，揭示了当前LLMs在生成任务中性能表现的重要见解。

    

    这篇论文介绍了一种针对大型语言模型（LLMs）如Llama-2和Mistral的新型评估框架，重点是将图像生成的精度和召回率指标转化为文本生成。这种方法允许对生成文本的质量和多样性进行细致评估，而无需对齐的语料库。通过对最先进的语言模型进行全面评估，研究揭示了它们在开放生成任务上的表现，这是传统基准无法充分捕捉的。研究结果突出了在模型利用人类反馈进行微调时，生成样本质量和多样性之间的权衡。这项工作扩展了基于分布的自然语言处理评估工具包，为当前LLMs在生成多样性和高质量文本方面面临的实际能力和挑战提供了见解。

    arXiv:2402.10693v1 Announce Type: new  Abstract: This paper introduces a novel evaluation framework for Large Language Models (LLMs) such as Llama-2 and Mistral, focusing on the adaptation of Precision and Recall metrics from image generation to text generation. This approach allows for a nuanced assessment of the quality and diversity of generated text without the need for aligned corpora. By conducting a comprehensive evaluation of state-of-the-art language models, the study reveals significant insights into their performance on open-ended generation tasks, which are not adequately captured by traditional benchmarks. The findings highlight a trade-off between the quality and diversity of generated samples, particularly when models are fine-tuned with human feedback. This work extends the toolkit for distribution-based NLP evaluation, offering insights into the practical capabilities and challenges faced by current LLMs in generating diverse and high-quality text.
    
[^27]: MultiPoT: 多语言思维程序利用多种编程语言

    MultiPoT: Multilingual Program of Thoughts Harnesses Multiple Programming Languages

    [https://arxiv.org/abs/2402.10691](https://arxiv.org/abs/2402.10691)

    MultiPoT 提出了一种任务和模型无关的方法，通过利用多种编程语言的优势和多样性，在表现上显著优于 Python 自一致性。

    

    arXiv:2402.10691v1 公告类型：新的 摘要：思维程序（PoT）是一种以其可执行中间步骤为特征的方法，其确保推理过程中数值计算的准确性。目前，PoT主要使用Python。然而，仅依赖单一语言可能导致次优解决方案，忽视其他编程语言的潜在优势。在本文中，我们对PoT中使用的编程语言进行了全面实验，发现没有一种单一语言在所有任务和模型上始终提供最佳性能。每种语言的有效性取决于具体情景。受此启发，我们提出了一种称为MultiPoT的任务和模型无关方法，该方法从各种语言中获取强大和多样性。实验结果显示，MultiPoT 在很大程度上优于Python 自一致性。此外，与最佳模型相比，它实现了可比或更优异的性能。

    arXiv:2402.10691v1 Announce Type: new  Abstract: Program of Thoughts (PoT) is an approach characterized by its executable intermediate steps, which ensure the accuracy of the numerical calculations in the reasoning process. Currently, PoT primarily uses Python. However, relying solely on a single language may result in suboptimal solutions and overlook the potential benefits of other programming languages. In this paper, we conduct comprehensive experiments on the programming languages used in PoT and find that no single language consistently delivers optimal performance across all tasks and models. The effectiveness of each language varies depending on the specific scenarios. Inspired by this, we propose a task and model agnostic approach called MultiPoT, which harnesses strength and diversity from various languages. Experimental results reveal that it significantly outperforms Python Self-Consistency. Furthermore, it achieves comparable or superior performance compared to the best mo
    
[^28]: 多元文化常识知识蒸馏

    Multi-Cultural Commonsense Knowledge Distillation

    [https://arxiv.org/abs/2402.10689](https://arxiv.org/abs/2402.10689)

    提出了一种MANGO方法，通过从概念和文化两个入口点谨慎而迭代地提示LLMs，提炼高准确度、高召回率的文化知识断言，提供了大量高准确度断言，能够改善对话系统回应的质量、特异性和文化敏感性。

    

    尽管最近取得了一定进展，但大型语言模型（LLMs）仍然面临着适当应对社会和文化惯例的挑战。本文提出了MANGO，一种用于提炼高准确度、高召回率文化知识断言的方法论。我们从概念和文化两个入口点谨慎而迭代地提示LLMs进行这一目的。通过聚类和生成摘要将输出结果巩固。运行MANGO方法，以GPT-3.5作为底层LLM，为30K个概念和11K个文化提供了167K个高准确度断言，大幅超过先前的资源。为了外部评估，我们探索了将对话系统与文化知识断言相结合的方法。我们发现，添加来自MANGO的知识可以提升对话回应的整体质量、特异性和文化敏感性，这是由人类标注者评判的。数据和代码可供下载。

    arXiv:2402.10689v1 Announce Type: new  Abstract: Despite recent progress, large language models (LLMs) still face the challenge of appropriately reacting to the intricacies of social and cultural conventions. This paper presents MANGO, a methodology for distilling high-accuracy, high-recall assertions of cultural knowledge. We judiciously and iteratively prompt LLMs for this purpose from two entry points, concepts and cultures. Outputs are consolidated via clustering and generative summarization. Running the MANGO method with GPT-3.5 as underlying LLM yields 167K high-accuracy assertions for 30K concepts and 11K cultures, surpassing prior resources by a large margin. For extrinsic evaluation, we explore augmenting dialogue systems with cultural knowledge assertions. We find that adding knowledge from MANGO improves the overall quality, specificity, and cultural sensitivity of dialogue responses, as judged by human annotators. Data and code are available for download.
    
[^29]: 打开大型语言模型的黑匣子：整体可解释性的两个视角

    Opening the Black Box of Large Language Models: Two Views on Holistic Interpretability

    [https://arxiv.org/abs/2402.10688](https://arxiv.org/abs/2402.10688)

    通过整体可解释性框架，本文提出了打开大型语言模型黑匣子的方法，包括自下而上的机械解释和自上而下的表示工程视角，有助于深入理解和应用LLMs的行为和机制。

    

    随着大型语言模型(LLMs)变得越来越强大，人们对潜在伤害(如毒性、不公平和幻觉)的担忧威胁到用户的信任。通过模型对齐确保LLMs与人类价值观的有益契合因此至关重要，但具有挑战性，需要对LLMs的行为和机制有更深入的理解。我们提出通过一个涵盖互补的自下而上和自上而下视角的整体解释框架来打开LLMs的黑匣子。自下而上视角由机械解释能力实现，侧重于组件功能和训练动态。自上而下视角利用表示工程通过隐藏表示分析行为。在本文中，我们回顾了周围关于机械解释能力和表示工程的情况，总结了方法，讨论了限制和应用，并概述了将这些技术用于达到的未来挑战。

    arXiv:2402.10688v1 Announce Type: new  Abstract: As large language models (LLMs) grow more powerful, concerns around potential harms like toxicity, unfairness, and hallucination threaten user trust. Ensuring beneficial alignment of LLMs with human values through model alignment is thus critical yet challenging, requiring a deeper understanding of LLM behaviors and mechanisms. We propose opening the black box of LLMs through a framework of holistic interpretability encompassing complementary bottom-up and top-down perspectives. The bottom-up view, enabled by mechanistic interpretability, focuses on component functionalities and training dynamics. The top-down view utilizes representation engineering to analyze behaviors through hidden representations. In this paper, we review the landscape around mechanistic interpretability and representation engineering, summarizing approaches, discussing limitations and applications, and outlining future challenges in using these techniques to achiev
    
[^30]: LongHeads: 多头注意力其实是一个长上下文处理器

    LongHeads: Multi-Head Attention is Secretly a Long Context Processor

    [https://arxiv.org/abs/2402.10685](https://arxiv.org/abs/2402.10685)

    LongHeads 提出了一个无需训练的框架，通过释放多头注意力的潜力来增强大型语言模型(LLM)处理长上下文的能力。

    

    大型语言模型(LLMs)在许多领域取得了令人印象深刻的表现，但由于有限长度泛化和注意力的二次计算需求，往往难以有效高效地处理较长的输入。 许多人试图通过限制在预训练长度内的注意力窗口来缓解这一问题。 然而，这些方法引入了新问题，如忽略中间上下文和需要额外训练。 为了解决这些问题，我们提出了LongHeads，一个无需训练的框架，通过释放多头注意力的潜力来增强LLM的长上下文能力。 我们允许每个头部选择并关注重要的上下文块，以处理分布长度，而不是让每个头部都参与全句注意力，这样做由于分布之外的问题而难以泛化到更长的序列。

    arXiv:2402.10685v1 Announce Type: cross  Abstract: Large language models (LLMs) have achieved impressive performance in numerous domains but often struggle to process lengthy inputs effectively and efficiently due to limited length generalization and attention's quadratic computational demands. Many sought to mitigate this by restricting the attention window within the pre-trained length. However, these methods introduce new issues such as ignoring the middle context and requiring additional training. To address these problems, we propose LongHeads, a training-free framework that enhances LLM's long context ability by unlocking multi-head attention's untapped potential. Instead of allowing each head to attend to the full sentence, which struggles with generalizing to longer sequences due to out-of-distribution (OOD) issues, we allow each head to process in-distribution length by selecting and attending to important context chunks. To this end, we propose a chunk selection strategy that
    
[^31]: 德语文本简化：使用半合成数据微调大型语言模型

    German Text Simplification: Finetuning Large Language Models with Semi-Synthetic Data

    [https://arxiv.org/abs/2402.10675](https://arxiv.org/abs/2402.10675)

    该研究利用半合成数据对大型语言模型进行微调，成功完成德语文本的文档级简化，并展示了合成数据在改善文本简化方面的潜力。

    

    这项研究首次利用合成生成的数据来训练生成模型，完成了德语文本的文档级简化。我们通过真实世界在线文本展示了我们方法的有效性。为解决语言简化中的数据稀缺挑战，我们爬取了经过专业简化的德语文本，并使用GPT-4合成了一个语料库。我们在这些数据上微调了拥有多达130亿参数的大型语言模型，并评估了它们的性能。本文采用了各种方法进行评估，并展示了目前使用的基于规则的评估指标的局限性。自动和手动评估均表明我们的模型能够显著简化真实世界在线文本，表明了合成数据在改善文本简化中的潜力。

    arXiv:2402.10675v1 Announce Type: new  Abstract: This study pioneers the use of synthetically generated data for training generative models in document-level text simplification of German texts. We demonstrate the effectiveness of our approach with real-world online texts. Addressing the challenge of data scarcity in language simplification, we crawled professionally simplified German texts and synthesized a corpus using GPT-4. We finetune Large Language Models with up to 13 billion parameters on this data and evaluate their performance. This paper employs various methodologies for evaluation and demonstrates the limitations of currently used rule-based metrics. Both automatic and manual evaluations reveal that our models can significantly simplify real-world online texts, indicating the potential of synthetic data in improving text simplification.
    
[^32]: 通过分解来增强注意力：通过工作流范式改进基于LLM的文本到SQL转换

    Decomposition for Enhancing Attention: Improving LLM-based Text-to-SQL through Workflow Paradigm

    [https://arxiv.org/abs/2402.10671](https://arxiv.org/abs/2402.10671)

    提出了一种通过工作流范式方法来改善LLMs在文本到SQL中的上下文学习能力，通过分解提高了模型的注意力和问题解决范围，进一步提高了基于LLM的方法的上限。

    

    大语言模型（LLMs）的上下文学习在自然语言处理领域取得了显著成功，而广泛的案例研究表明，单步链式思维提示方法在复杂任务（如文本到SQL）中面临注意力扩散和性能不足等挑战。为了改善LLMs在文本到SQL中的上下文学习能力，提出了一种工作流范式方法，旨在通过分解增强LLMs的注意力和问题解决范围。具体来说，用于消除冗余信息的信息确定模块和基于问题分类的全新提示结构极大增强了模型的注意力。此外，引入自校正和主动学习模块极大扩展了LLMs的问题解决范围，从而提高了基于LLM方法的上限。在三个数据集上进行了大量实验。

    arXiv:2402.10671v1 Announce Type: new  Abstract: In-context learning of large-language models (LLMs) has achieved remarkable success in the field of natural language processing, while extensive case studies reveal that the single-step chain-of-thought prompting approach faces challenges such as attention diffusion and inadequate performance in complex tasks like text-to-SQL. To improve the contextual learning capabilities of LLMs in text-to-SQL, a workflow paradigm method is proposed, aiming to enhance the attention and problem-solving scope of LLMs through decomposition. Specifically, the information determination module for eliminating redundant information and the brand-new prompt structure based on problem classification greatly enhance the model's attention. Additionally, the inclusion of self-correcting and active learning modules greatly expands the problem-solving scope of LLMs, hence improving the upper limit of LLM-based approaches. Extensive experiments conducted on three da
    
[^33]: OpenFMNav: 通过视觉-语言基础模型实现开放式零样本目标导航

    OpenFMNav: Towards Open-Set Zero-Shot Object Navigation via Vision-Language Foundation Models

    [https://arxiv.org/abs/2402.10670](https://arxiv.org/abs/2402.10670)

    本研究提出了一种名为OpenFMNav的框架，通过大型语言模型和视觉语言模型解决了目标导航领域中关于理解自然语言指令和零样本泛化的问题。

    

    目标导航(ObjectNav)需要一个代理在未知环境中导航以找到查询对象。许多先前的方法尝试通过依赖监督学习或强化学习来解决这一任务，其中它们是在具有闭集对象的有限家庭数据集上进行训练的。然而，仍有两个关键挑战尚未解决：理解要求开放集对象的自由形式自然语言指令，并以零样本方式推广到新环境。为了解决这两个挑战，在本文中，我们提出了OpenFMNav，一种基于开放集基础模型的零样本目标导航框架。我们首先释放大型语言模型(LLMs)的推理能力，从符合用户需求的自然语言指令中提取提议的对象。然后，利用大型视觉语言模型(VLMs)的泛化能力，积极发现并检测场景中的候选对象，构建一个Ve

    arXiv:2402.10670v1 Announce Type: new  Abstract: Object navigation (ObjectNav) requires an agent to navigate through unseen environments to find queried objects. Many previous methods attempted to solve this task by relying on supervised or reinforcement learning, where they are trained on limited household datasets with close-set objects. However, two key challenges are unsolved: understanding free-form natural language instructions that demand open-set objects, and generalizing to new environments in a zero-shot manner. Aiming to solve the two challenges, in this paper, we propose OpenFMNav, an Open-set Foundation Model based framework for zero-shot object Navigation. We first unleash the reasoning abilities of large language models (LLMs) to extract proposed objects from natural language instructions that meet the user's demand. We then leverage the generalizability of large vision language models (VLMs) to actively discover and detect candidate objects from the scene, building a Ve
    
[^34]: 人类还是大型语言模型作为裁判？一项关于判决偏见的研究

    Humans or LLMs as the Judge? A Study on Judgement Biases

    [https://arxiv.org/abs/2402.10669](https://arxiv.org/abs/2402.10669)

    提出了一种新框架来研究LLM和人类裁判的偏见，揭示人类和LLM裁判在面对干扰时的脆弱性，强调评估现有LLM性能的挑战。

    

    采用人类和大型语言模型（LLM）作为裁判（即人类和LLM作为裁判）来评估现有LLM性能的做法近来备受关注。然而，这种方法同时可能引入人类和LLM裁判的潜在偏见，质疑评估结果的可靠性。本文提出了一种新颖的框架，用于研究LLM和人类裁判的5种偏见。我们整理了一个包含142个样本的数据集，涉及修订的布卢姆分类法，并进行了成千上万次的人类和LLM评估。结果表明，人类和LLM裁判在不同程度上都容易受到干扰，即使最尖端的裁判也存在相当大的偏见。我们进一步利用他们的弱点对LLM裁判进行攻击。希望我们的工作能提醒社群关于人类和LLM作为裁判在面对干扰时的脆弱性，以及发展的紧迫性。

    arXiv:2402.10669v1 Announce Type: new  Abstract: Adopting human and large language models (LLM) as judges (\textit{a.k.a} human- and LLM-as-a-judge) for evaluating the performance of existing LLMs has recently gained attention. Nonetheless, this approach concurrently introduces potential biases from human and LLM judges, questioning the reliability of the evaluation results. In this paper, we propose a novel framework for investigating 5 types of biases for LLM and human judges. We curate a dataset with 142 samples referring to the revised Bloom's Taxonomy and conduct thousands of human and LLM evaluations. Results show that human and LLM judges are vulnerable to perturbations to various degrees, and that even the most cutting-edge judges possess considerable biases. We further exploit their weakness and conduct attacks on LLM judges. We hope that our work can notify the community of the vulnerability of human- and LLM-as-a-judge against perturbations, as well as the urgency of develop
    
[^35]: 开放域文本到SQL的多跳表检索

    Multi-Hop Table Retrieval for Open-Domain Text-to-SQL

    [https://arxiv.org/abs/2402.10666](https://arxiv.org/abs/2402.10666)

    提出了一种多跳表检索方法，通过重写问题和波束搜索来减少相似无关实体的影响，并通过多跳检索中重新编写问题来缓解领域不匹配实体的限制，取得了新的最先进结果

    

    开放域文本到SQL是一个重要任务，它从庞大的数据库中检索与问题相关的表，然后生成SQL。然而，现有的单跳检索方法并未关注文本到SQL挑战中的模式链接，这涉及到将问题中的实体与表中实体对齐，主要体现在两个方面：相似的无关实体和领域不匹配实体。因此，我们提出了我们的方法，即带重写和波束搜索的多跳表检索（Murre）。为了减少相似的无关实体的影响，我们的方法侧重于每个跳跃中未检索到的实体，并通过波束搜索考虑排名较低的表。为了缓解领域不匹配实体的限制，Murre基于多个跳跃中检索到的表重写问题，减少与相关表的领域差距。我们在SpiderUnion和BirdUnion+上进行实验，取得了新的最先进结果。

    arXiv:2402.10666v1 Announce Type: new  Abstract: Open-domain text-to-SQL is an important task that retrieves question-relevant tables from massive databases and then generates SQL. However, existing retrieval methods that retrieve in a single hop do not pay attention to the text-to-SQL challenge of schema linking, which is aligning the entities in the question with table entities, reflected in two aspects: similar irrelevant entity and domain mismatch entity. Therefore, we propose our method, the multi-hop table retrieval with rewrite and beam search (Murre). To reduce the effect of the similar irrelevant entity, our method focuses on unretrieved entities at each hop and considers the low-ranked tables by beam search. To alleviate the limitation of domain mismatch entity, Murre rewrites the question based on retrieved tables in multiple hops, decreasing the domain gap with relevant tables. We conduct experiments on SpiderUnion and BirdUnion+, reaching new state-of-the-art results with 
    
[^36]: 通过无需人类参与的融合方法改善文本到SQL的演示多样性

    Improving Demonstration Diversity by Human-Free Fusing for Text-to-SQL

    [https://arxiv.org/abs/2402.10663](https://arxiv.org/abs/2402.10663)

    本文提出了一种通过无需人类参与的多次迭代合成来改善文本到SQL演示的多样性，并构建了高多样性演示池，提高了多样性并降低标注成本。

    

    目前，基于大型语言模型（LLMs）的上下文学习方法已成为文本到SQL研究的主流。先前的工作讨论了如何从人标记的演示池中选择与用户问题相关的演示。然而，人工标注存在着多样性不足和标注成本高的限制。因此，在本文中，我们讨论了如何衡量和改善文本到SQL演示的多样性。我们提出了一个度量演示多样性的指标，并通过实验分析了现有标记数据的不足之处。基于上述发现，我们提出了一种通过无需人类参与的多次迭代合成来构建高多样性演示池的融合方法（Fused），提高了多样性并降低标注成本。我们的方法在有/无人类标注的情况下平均提高了3.2%和5.0%。

    arXiv:2402.10663v1 Announce Type: new  Abstract: Currently, the in-context learning method based on large language models (LLMs) has become the mainstream of text-to-SQL research. Previous works have discussed how to select demonstrations related to the user question from a human-labeled demonstration pool. However, human labeling suffers from the limitations of insufficient diversity and high labeling overhead. Therefore, in this paper, we discuss how to measure and improve the diversity of the demonstrations for text-to-SQL. We present a metric to measure the diversity of the demonstrations and analyze the insufficient of the existing labeled data by experiments. Based on the above discovery, we propose fusing iteratively for demonstrations (Fused) to build a high-diversity demonstration pool through human-free multiple-iteration synthesis, improving diversity and lowering label cost. Our method achieves an average improvement of 3.2% and 5.0% with and without human labeling on sever
    
[^37]: 为奇幻领域微调命名实体提取模型

    Fine Tuning Named Entity Extraction Models for the Fantasy Domain

    [https://arxiv.org/abs/2402.10662](https://arxiv.org/abs/2402.10662)

    通过使用D&D领域中的怪物传说来微调Trankit，实现了命名实体提取模型在奇幻领域的有效应用。

    

    命名实体识别（NER）是一项序列分类自然语言处理任务，在该任务中，文本中的实体被识别并分类为预定义的类别。它为大多数信息提取系统奠定了基础。《龙与地下城》（Dungeons and Dragons，D&D）是一款开放式桌面奇幻游戏，拥有自己丰富多样的传说。D&D的实体是领域特定的，因此即使是最先进的现成NER系统也无法识别这些实体，因为NER系统是针对预定义类别（如人物（PERS）、地点（LOC）、组织（ORG）和杂项（MISC））的通用数据进行训练的。为了从奇幻文本中提取有意义的信息，实体需要被分类为领域特定的实体类别，并且模型需要在领域相关语料库上进行微调。该研究利用D&D领域中可用的怪物传说来微调Trankit，这是一个使用预训练模型的繁荣NER框架。

    arXiv:2402.10662v1 Announce Type: new  Abstract: Named Entity Recognition (NER) is a sequence classification Natural Language Processing task where entities are identified in the text and classified into predefined categories. It acts as a foundation for most information extraction systems. Dungeons and Dragons (D&D) is an open-ended tabletop fantasy game with its own diverse lore. DnD entities are domain-specific and are thus unrecognizable by even the state-of-the-art off-the-shelf NER systems as the NER systems are trained on general data for pre-defined categories such as: person (PERS), location (LOC), organization (ORG), and miscellaneous (MISC). For meaningful extraction of information from fantasy text, the entities need to be classified into domain-specific entity categories as well as the models be fine-tuned on a domain-relevant corpus. This work uses available lore of monsters in the D&D domain to fine-tune Trankit, which is a prolific NER framework that uses a pre-trained 
    
[^38]: 多个LLM之间的网络形成与动态

    Network Formation and Dynamics Among Multi-LLMs

    [https://arxiv.org/abs/2402.10659](https://arxiv.org/abs/2402.10659)

    分析了多个LLM在社交网络中的行为，发现它们在给定网络结构并被询问形成网络偏好时表现出与人类社交动态一致的原则。

    

    社交网络影响行为、偏好和关系，在人类社会中对信息和规范的传播起着至关重要的作用。随着大型语言模型（LLMs）越来越多地融入社交和专业环境中，理解它们在社交网络和互动背景下的行为变得至关重要。我们的研究分析了标准网络结构和现实世界网络的行为，以确定多个LLMs的动态是否与人类社交动态一致。我们探讨了各种社交网络原则，包括微观层面的概念，如偏爱附着、三角闭合和同似性，以及宏观层面的概念，如社区结构和小世界现象。我们的研究发现表明，当向LLMs提供网络结构并询问它们对网络形成的偏好时，它们表现出所有这些原则。

    arXiv:2402.10659v1 Announce Type: cross  Abstract: Social networks influence behaviors, preferences, and relationships and play a crucial role in the dissemination of information and norms within human societies. As large language models (LLMs) increasingly integrate into social and professional environments, understanding their behavior within the context of social networks and interactions becomes essential. Our study analyzes the behaviors of standard network structures and real-world networks to determine whether the dynamics of multiple LLMs align with human social dynamics. We explore various social network principles, including micro-level concepts such as preferential attachment, triadic closure, and homophily, as well as macro-level concepts like community structure and the small-world phenomenon. Our findings suggest that LLMs demonstrate all these principles when they are provided with network structures and asked about their preferences regarding network formation. Furtherm
    
[^39]: 借鉴可靠推理过程增强数值推理能力

    Enhancing Numerical Reasoning with the Guidance of Reliable Reasoning Processes

    [https://arxiv.org/abs/2402.10654](https://arxiv.org/abs/2402.10654)

    通过分解答案公式以确保支持答案，借鉴可靠推理过程的方法增强了数值推理能力。

    

    数值推理是自然语言处理系统处理数值信息的必要能力。最近的研究表明，通过微调小规模模型，使其学习在回答问题的同时生成推理过程，可以显著提高性能。然而，当前方法的局限性在于大多数方法通过大型语言模型生成推理过程，这些过程“不可靠”，因为这种过程可能包含与答案无关的信息。为了解决这一限制，我们引入了Enhancing NumeriCal reasOning with Reliable procEsses (Encore)，通过分解答案公式得出可靠的推理过程，确保完全支持答案。然而，模型可能缺乏足够的数据来充分学习推理过程生成，因为我们的方法为一个公式只生成一个推理过程。为了克服这一困难，我们提出了一系列的预训练任务来h

    arXiv:2402.10654v1 Announce Type: new  Abstract: Numerical reasoning is an essential ability for NLP systems to handle numeric information. Recent research indicates that fine-tuning a small-scale model to learn generating reasoning processes alongside answers can significantly enhance performance. However, current methods have the limitation that most methods generate reasoning processes with large language models (LLMs), which are "unreliable" since such processes could contain information unrelated to the answer. To address this limitation, we introduce Enhancing NumeriCal reasOning with Reliable procEsses (Encore), which derives the reliable reasoning process by decomposing the answer formula, ensuring which fully supports the answer. Nevertheless, models could lack enough data to learn the reasoning process generation adequately, since our method generates only one single reasoning process for one formula. To overcome this difficulty, we present a series of pre-training tasks to h
    
[^40]: 从合理性评估中通过解释调节提取LLMs的抽象能力

    AbsInstruct: Eliciting Abstraction Ability from LLMs through Explanation Tuning with Plausibility Estimation

    [https://arxiv.org/abs/2402.10646](https://arxiv.org/abs/2402.10646)

    通过指导调节和合理性评估，本研究设计了AbsInstruct框架来增强LLMs的抽象能力，提供了强大的泛化性能。

    

    抽象能力对人类智能至关重要，也可以在自然语言处理研究的各种任务中受益。现有工作表明LLMs在抽象能力上存在不足，如何改进仍未被探索。在这项工作中，我们设计了AbsInstruct框架，通过指导调节来增强LLMs的抽象能力。该框架使用深入解释构建指导，帮助LLMs捕捉抽象的潜在原理。同时，我们引入了一个合理性估计器来选择更符合LLMs抽象知识的指导以进行对齐。然后，我们的框架将抽象指导与通用指导结合以构建混合数据集。大量实验和分析表明，我们的框架可以显着增强LLMs的抽象能力，并保持其通用的指导遵循能力。

    arXiv:2402.10646v1 Announce Type: new  Abstract: Abstraction ability is crucial in human intelligence, which can also benefit various tasks in NLP study. Existing work shows that LLMs are deficient in abstract ability, and how to improve it remains unexplored. In this work, we design the framework AbsInstruct to enhance LLMs' abstraction ability through instruction tuning. The framework builds instructions with in-depth explanations to assist LLMs in capturing the underlying rationale of abstraction. Meanwhile, we introduce a plausibility estimator to select instructions that are more consistent with the abstraction knowledge of LLMs to be aligned. Then, our framework combines abstraction instructions with general-purpose ones to build a hybrid dataset. Extensive experiments and analyses demonstrate that our framework can considerably enhance LLMs' abstraction ability with strong generalization performance while maintaining their general instruction-following abilities.
    
[^41]: 分隔符是否可以提高思维链提示的效果？

    Can Separators Improve Chain-of-Thought Prompting?

    [https://arxiv.org/abs/2402.10645](https://arxiv.org/abs/2402.10645)

    分隔符的引入在思维链提示中显著提高了大型语言模型（LLMs）在复杂推理任务上的表现。

    

    Chain-of-thought (CoT) prompting是一种简单有效的方法，用于提高大型语言模型（LLMs）的推理能力。CoT的基本理念是通过将示例放在输入提示中，让LLMs逐步拆解他们的思维过程。然而，CoT提示的密集结构可能导致LLMs的认知负荷过重。受人类认知启发，我们引入了CoT-Sep，一种新颖的方法，在CoT提示中每个示例的末尾策略性地应用分隔符。这些分隔符旨在帮助LLMs在推理过程中更好地理解他们的思维过程。结果表明，与不使用分隔符的普通CoT相比，CoT-Sep显著提高了LLMs在复杂推理任务（如GSM-8K、AQuA、CSQA）上的表现。我们还研究了不同类型和位置的分隔符对多个LLMs（包括GPT-3.5-Turbo、GPT-4和LLaMA-27）的影响。

    arXiv:2402.10645v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) prompting is a simple and effective method for improving the reasoning capabilities of Large language models (LLMs). The basic idea of CoT is to let LLMs break down their thought processes step-by-step by putting exemplars in the input prompt. However, the densely structured prompt exemplars of CoT may cause the cognitive overload of LLMs. Inspired by human cognition, we introduce CoT-Sep, a novel method that strategically employs separators at the end of each exemplar in CoT prompting. These separators are designed to help the LLMs understand their thought processes better while reasoning. It turns out that CoT-Sep significantly improves the LLMs' performances on complex reasoning tasks (e.g., GSM-8K, AQuA, CSQA), compared with the vanilla CoT, which does not use separators. We also study the effects of the type and the location of separators tested on multiple LLMs, including GPT-3.5-Turbo, GPT-4, and LLaMA-2 7
    
[^42]: “保持联系：通过模拟人类记忆在提取摘要中强化连贯性”

    `Keep it Together': Enforcing Cohesion in Extractive Summaries by Simulating Human Memory

    [https://arxiv.org/abs/2402.10643](https://arxiv.org/abs/2402.10643)

    本文通过模拟人类记忆来保持主题连贯性，实现了在提取式摘要中强化连贯性的目标，同时保持信息量和减少冗余。

    

    提取式摘要通常以一系列句子的形式呈现，它们之间没有预期的连贯性。本文旨在在摘要中强化连贯性，同时控制信息量和冗余，特别是当输入具有较高冗余性时。该方法在处理长输入时控制冗余，并在选择句子时平衡信息量和连贯性。我们的句子选择器模拟人类记忆以跟踪主题 -- 被建模为词链 -- 在名词短语之间强化连贯联系。在各种领域的实验证明，可以提取高度连贯的摘要，然而读者仍会感到这些摘要和仅考虑信息量或冗余性的摘要一样富有信息。提取的摘要在句子之间展示了平滑的主题转换，这些转换被词链所标识，这些链跨越相邻或几乎相邻的句子。

    arXiv:2402.10643v1 Announce Type: cross  Abstract: Extractive summaries are usually presented as lists of sentences with no expected cohesion between them. In this paper, we aim to enforce cohesion whilst controlling for informativeness and redundancy in summaries, in cases where the input exhibits high redundancy. The pipeline controls for redundancy in long inputs as it is consumed, and balances informativeness and cohesion during sentence selection. Our sentence selector simulates human memory to keep track of topics --modeled as lexical chains--, enforcing cohesive ties between noun phrases. Across a variety of domains, our experiments revealed that it is possible to extract highly cohesive summaries that nevertheless read as informative to humans as summaries extracted by only accounting for informativeness or redundancy. The extracted summaries exhibit smooth topic transitions between sentences as signaled by lexical chains, with chains spanning adjacent or near-adjacent sentence
    
[^43]: 基于符号权重方向的领域特定适配器混合的泛化与其在有效模型剪枝中的应用

    Generalizability of Mixture of Domain-Specific Adapters from the Lens of Signed Weight Directions and its Application to Effective Model Pruning

    [https://arxiv.org/abs/2402.10639](https://arxiv.org/abs/2402.10639)

    本研究对领域特定适配器混合在领域内评估中的泛化性进行了全面分析，并探讨了混合适配器的内部运作，为适应新领域的性能优化提供了关键洞见

    

    基于适配器的几种参数高效的微调方法被提出作为一种简化的方法，不仅能将单一专业知识整合到现有的预训练语言模型（PLM）中，还能一次性整合多个专业知识。最近的作品如AdapterSoup提出了通过模型权重平均化在推理过程中混合领域特定适配器的方法，以优化在新领域中的性能，并具有出色的计算效率。然而，当前这种新兴的权重空间适配器混合机制在未知的领域内例子上的基本泛化性仍未被探讨。因此，在本研究中，我们进行了全面分析，阐明了领域特定适配器混合在领域内评估中的泛化性。我们还通过分析它们的权重符号来深入研究领域特定适配器混合的内部运作，得出了关键的分析。

    arXiv:2402.10639v1 Announce Type: new  Abstract: Several parameter-efficient fine-tuning methods based on adapters have been proposed as a streamlined approach to incorporate not only a single specialized knowledge into existing Pre-Trained Language Models (PLMs) but also multiple of them at once. Recent works such as AdapterSoup propose to mix not all but only a selective sub-set of domain-specific adapters during inference via model weight averaging to optimize performance on novel, unseen domains with excellent computational efficiency. However, the essential generalizability of this emerging weight-space adapter mixing mechanism on unseen, in-domain examples remains unexplored. Thus, in this study, we conduct a comprehensive analysis to elucidate the generalizability of domain-specific adapter mixtures in in-domain evaluation. We also provide investigations into the inner workings of the mixture of domain-specific adapters by analyzing their weight signs, yielding critical analysis
    
[^44]: BitDistiller: 通过自蒸馏释放低于4位LLMs的潜力

    BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation

    [https://arxiv.org/abs/2402.10631](https://arxiv.org/abs/2402.10631)

    BitDistiller框架将量化感知训练（QAT）与知识蒸馏（KD）相结合，通过引入定制的量化和剪裁技术以及置信感知Kullback-Leibler散度（CAKLD）目标，实现了在极低精度下（低于4位）提升LLMs性能。

    

    arXiv:2402.10631v1 公告类型: 新内容 摘要: 大型语言模型（LLMs）的升级取得了自然语言处理领域的重大进展，但也带来了重大部署挑战。权重量化已经成为减少内存和计算需求的普遍接受的解决方案。本文介绍了BitDistiller，这是一个将量化感知训练（QAT）与知识蒸馏（KD）相结合的框架，以提升LLMs在极低精度（低于4位）下的性能。具体而言，BitDistiller首先采用了一种定制的非对称量化和剪裁技术，以最大限度地保留量化权重的保真度，然后提出了一种新颖的置信感知Kullback-Leibler散度（CAKLD）目标，以自蒸馏的方式实现更快的收敛和卓越的模型性能。实证评估表明，BitDistiller在3位和2位情况下明显超越了现有方法。

    arXiv:2402.10631v1 Announce Type: new  Abstract: The upscaling of Large Language Models (LLMs) has yielded impressive advances in natural language processing, yet it also poses significant deployment challenges. Weight quantization has emerged as a widely embraced solution to reduce memory and computational demands. This paper introduces BitDistiller, a framework that synergizes Quantization-Aware Training (QAT) with Knowledge Distillation (KD) to boost the performance of LLMs at ultra-low precisions (sub-4-bit). Specifically, BitDistiller first incorporates a tailored asymmetric quantization and clipping technique to maximally preserve the fidelity of quantized weights, and then proposes a novel Confidence-Aware Kullback-Leibler Divergence (CAKLD) objective, which is employed in a self-distillation manner to enable faster convergence and superior model performance. Empirical evaluations demonstrate that BitDistiller significantly surpasses existing methods in both 3-bit and 2-bit conf
    
[^45]: 通过积极查询增强角色扮演系统：评估与改进

    Enhancing Role-playing Systems through Aggressive Queries: Evaluation and Improvement

    [https://arxiv.org/abs/2402.10618](https://arxiv.org/abs/2402.10618)

    本论文设计了MORTISE系统，通过多个LLM模块的协作努力生成高度与角色相关的积极查询，进而改善角色扮演系统的性能。

    

    大型语言模型（LLMs）的出现将对话生成推向了新的领域，特别是在角色扮演系统（RPSs）领域。尽管现有基于LLM的RPS已经通过普通角色相关培训对话进行了增强，但在处理边界情境中的复杂和受困查询时，仍然存在着与角色不对齐的问题。在本文中，我们设计了模块化协调的陷阱设置交互系统（MORTISE），来评估和提高角色扮演LLMs的性能。MORTISE可以通过多个基于LLM的模块的协作努力产生高度与角色相关的积极查询，并通过一致的响应生成器制定相应的回复，从而创建对抗性训练数据集。我们选择了190种中文和英文角色来构建积极的查询，以评估现有的角色扮演LLMs。通过全面评估，我们发现现有模型在角色表示上普遍存在不足。

    arXiv:2402.10618v1 Announce Type: new  Abstract: The advent of Large Language Models (LLMs) has propelled dialogue generation into new realms, particularly in the field of role-playing systems (RPSs). While enhanced with ordinary role-relevant training dialogues, existing LLM-based RPSs still struggle to align with roles when handling intricate and trapped queries in boundary scenarios. In this paper, we design the Modular ORchestrated Trap-setting Interaction SystEm (MORTISE) to benchmark and improve the role-playing LLMs' performance. MORTISE can produce highly role-relevant aggressive queries through the collaborative effort of multiple LLM-based modules, and formulate corresponding responses to create an adversarial training dataset via a consistent response generator. We select 190 Chinese and English roles to construct aggressive queries to benchmark existing role-playing LLMs. Through comprehensive evaluation, we find that existing models exhibit a general deficiency in role ali
    
[^46]: 通过辩论调节LLMs以生成可控的具有争议性的声明

    Can LLMs Speak For Diverse People? Tuning LLMs via Debate to Generate Controllable Controversial Statements

    [https://arxiv.org/abs/2402.10614](https://arxiv.org/abs/2402.10614)

    本文通过辩论调节LLMs，使其生成可控的支持用户定义论点的声明，改进了LLMs的可控性，并提出了DEBATunE流程。通过两个LLMs之间的多轮辩论生成高质量的训练数据，以支持生成有更高质量和更突出的声明。

    

    LLMs代表不同的人群，尤其是少数群体，并产生支持其多样化甚至有争议观点的声明对于创造一个包容的环境至关重要。然而，现有的LLMs缺乏足够的控制性来支持生成内容的立场，其中往往包含不一致、中立或有偏见的声明。在本文中，我们改进了LLMs在生成支持用户在提示中定义的论点的声明时的可控性。我们发现两个持有相反立场的LLMs之间的多轮辩论产生了更高质量和更突出的声明，这些声明对于改善LLMs的可控性是重要的训练数据。受此启发，我们开发了一种新颖的Debate & Tuning（“DEBATunE”）流程，通过微调LLMs生成通过辩论获得的声明。为了检验DEBATunE，我们整理了迄今为止涵盖710个争议性主题的最大数据集。

    arXiv:2402.10614v1 Announce Type: cross  Abstract: Making LLMs speak for different, especially minority groups of people, and generate statements supporting their diverse or even controversial perspectives is critical to creating an inclusive environment. However, existing LLMs lack sufficient controllability to the stance of their generated content, which often contains inconsistent, neutral, or biased statements. In this paper, we improve the controllability of LLMs in generating statements supporting an argument the user defined in the prompt. We find that multi-round debates between two LLMs with opposite stances generate higher-quality and more salient statements for each, which are important training data to improve the controllability of LLMs. Motivated by this, we develop a novel debate & tuning ("DEBATunE") pipeline finetuning LLMs to generate the statements obtained via debate. To examine DEBATunE, we curate the largest dataset of debate topics so far, which covers 710 contro
    
[^47]: 仅在需要时检索：大型语言模型中的适应性检索增强以减轻幻觉

    Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models

    [https://arxiv.org/abs/2402.10612](https://arxiv.org/abs/2402.10612)

    本研究提出了一种新方法Rowen，通过选择性检索增强过程，采用多语义感知检测模块来平衡参数化知识和外部信息，以减轻大型语言模型中的幻觉问题。

    

    幻觉对于大型语言模型（LLMs）的实际实施构成了显著挑战。生成事实内容时利用参数化知识受到LLMs有限知识的限制，可能导致内部幻觉。虽然整合外部信息可以填补知识空白，但也会引入无关信息的风险，从而增加外部幻觉的可能性。在LLMs内部平衡地整合参数化知识和外部信息对缓解幻觉至关重要。本研究中，我们提出Rowen，一种增强LLMs的新方法，其中包括一种选择性检索增强过程，旨在解决幻觉输出。该过程由一个多语义感知检测模块管理，该模块评估了对相同查询在不同语言中的扰动响应的一致性。

    arXiv:2402.10612v1 Announce Type: new  Abstract: Hallucinations pose a significant challenge for the practical implementation of large language models (LLMs). The utilization of parametric knowledge in generating factual content is constrained by the limited knowledge of LLMs, potentially resulting in internal hallucinations. While incorporating external information can help fill knowledge gaps, it also introduces the risk of irrelevant information, thereby increasing the likelihood of external hallucinations. A careful and balanced integration of the parametric knowledge within LLMs with external information is crucial to alleviate hallucinations. In this study, we present Rowen, a novel approach that enhances LLMs with a selective retrieval augmentation process tailored to address hallucinated outputs. This process is governed by a multilingual semantic-aware detection module, which evaluates the consistency of the perturbed responses across various languages for the same queries. Up
    
[^48]: 使用单词替换密码来越狱专有的大型语言模型

    Jailbreaking Proprietary Large Language Models using Word Substitution Cipher

    [https://arxiv.org/abs/2402.10601](https://arxiv.org/abs/2402.10601)

    本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。

    

    大型语言模型（LLMs）遵循道德和伦理准则，但仍然容易受到名为Jailbreak的创意提示的影响，这些提示可以绕过对齐过程。然而，大多数越狱提示包含自然语言（主要是英语）中的有害问题，可以被LLMs自身检测到。本文提出了使用密码技术编码的越狱提示。我们首先在最先进的LLM，GPT-4上进行了一个试点研究，解码了使用各种密码技术加密的几个安全句子，发现简单的单词替换密码可以被最有效地解码。受此结果启发，我们使用这种编码技术来编写越狱提示。我们提供了将不安全单词映射到安全单词，并使用这些映射的单词提出不安全问题的映射。实验结果显示，我们提出的越狱攻击成功率（高达59.42%）。

    arXiv:2402.10601v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are aligned to moral and ethical guidelines but remain susceptible to creative prompts called Jailbreak that can bypass the alignment process. However, most jailbreaking prompts contain harmful questions in the natural language (mainly English), which can be detected by the LLM themselves. In this paper, we present jailbreaking prompts encoded using cryptographic techniques. We first present a pilot study on the state-of-the-art LLM, GPT-4, in decoding several safe sentences that have been encrypted using various cryptographic techniques and find that a straightforward word substitution cipher can be decoded most effectively. Motivated by this result, we use this encoding technique for writing jailbreaking prompts. We present a mapping of unsafe words with safe words and ask the unsafe question using these mapped words. Experimental results show an attack success rate (up to 59.42%) of our proposed jailbrea
    
[^49]: 规模效率：研究微小语言模型在临床任务中的性能

    Efficiency at Scale: Investigating the Performance of Diminutive Language Models in Clinical Tasks

    [https://arxiv.org/abs/2402.10597](https://arxiv.org/abs/2402.10597)

    研究了不同Parameter Efficient Fine-tuning (PEFT)方法在临床决策任务中的适用性，发现除了LoRA外，大多数PEFT方法在各个模型规模和任务中性能不稳定，而LoRA在所有情况下性能都相对较高。PEFT方法在临床领域特别有效，尤其适用于可以操作的专门模型。

    

    大型语言模型（LLMs）进入研究和商业领域，引发了越来越大模型的趋势，最初承诺通用性，随后普遍希望缩小规模并创建专门模型，而无需进行完整微调，使用参数高效微调（PEFT）方法。我们对不同PEFT方法在临床决策任务中的适用性进行了调查，涵盖一系列模型规模，包括只有$25$百万参数的极小模型。我们的分析表明，大多数PEFT方法在不同任务之间的性能差异较大，除了LoRA外，LoRA在所有模型规模和任务中的性能保持相对较高，通常接近或达到完全微调性能。 PEFT方法在临床领域的有效性是显而易见的，特别是对于可以操作的专门模型

    arXiv:2402.10597v1 Announce Type: cross  Abstract: The entry of large language models (LLMs) into research and commercial spaces has led to a trend of ever-larger models, with initial promises of generalisability, followed by a widespread desire to downsize and create specialised models without the need for complete fine-tuning, using Parameter Efficient Fine-tuning (PEFT) methods. We present an investigation into the suitability of different PEFT methods to clinical decision-making tasks, across a range of model sizes, including extremely small models with as few as $25$ million parameters.   Our analysis shows that the performance of most PEFT approaches varies significantly from one task to another, with the exception of LoRA, which maintains relatively high performance across all model sizes and tasks, typically approaching or matching full fine-tuned performance. The effectiveness of PEFT methods in the clinical domain is evident, particularly for specialised models which can oper
    
[^50]: 拉马在英语中有效吗？关于多语言变压器的潜在语言

    Do Llamas Work in English? On the Latent Language of Multilingual Transformers

    [https://arxiv.org/abs/2402.10588](https://arxiv.org/abs/2402.10588)

    本研究通过对Llama-2系列变压器模型的研究发现，在多语言语言模型中存在英语作为内部枢纽语言的现象，这有助于理解语言模型的功能方式以及语言偏见的起源。

    

    我们探讨了是否在不平衡、英语主导的语料库上训练的多语言语言模型使用英语作为内部枢纽语言的问题——这对于理解语言模型的功能方式以及语言偏见的起源至关重要。 我们关注Llama-2系列变压器模型，通过使用精心构建的非英语提示和唯一正确的单词延续来进行研究。 从一层到另一层，变压器逐渐将最终提示令牌的输入嵌入映射到输出嵌入，从中计算下一个令牌的概率。 通过跟踪其在高维空间中的中间嵌入，揭示了三个不同的阶段，即中间嵌入（1）开始远离输出令牌嵌入；（2）在中间层已经允许解码一个语义正确的下一个令牌，但更倾向于英语版本而不是输入语言的版本；（3）最终移动到

    arXiv:2402.10588v1 Announce Type: new  Abstract: We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language models function and the origins of linguistic bias. Focusing on the Llama-2 family of transformer models, our study uses carefully constructed non-English prompts with a unique correct single-token continuation. From layer to layer, transformers gradually map an input embedding of the final prompt token to an output embedding from which next-token probabilities are computed. Tracking intermediate embeddings through their high-dimensional space reveals three distinct phases, whereby intermediate embeddings (1) start far away from output token embeddings; (2) already allow for decoding a semantically correct next token in the middle layers, but give higher probability to its version in English than in the input language; (3) finally move into an
    
[^51]: 细微之线：通过话语主题识别机器生成的文本

    Threads of Subtlety: Detecting Machine-Generated Texts Through Discourse Motifs

    [https://arxiv.org/abs/2402.10586](https://arxiv.org/abs/2402.10586)

    本文探讨了如何通过研究文本中的话语特征来区分人类创作和机器生成的文本，引入了一种新颖的方法来揭示这些特征，并发现人类写作在结构上更为多样化。

    

    随着大型语言模型（LLM）的出现，人类创作和机器生成的文本之间的界限变得日益模糊。本文探讨了识别人类撰写的文本中可辨识和独特的语言特性的研究，特别是揭示文本在表面结构之外的潜在话语结构。引入了一种新颖的方法论，我们利用层次化解析树和递归超图来揭示LLM和人类生成的文本中的独特话语模式。实证研究结果表明，尽管LLM和人类生成的文本都受特定领域的影响而产生不同的话语模式，但人类撰写的文本表现出更多的结构变异性，反映了不同领域人类写作的微妙性质。值得注意的是，引入层次话语特征可以增强二元分类器在区分人类生成和机器生成文本方面的整体性能。

    arXiv:2402.10586v1 Announce Type: new  Abstract: With the advent of large language models (LLM), the line between human-crafted and machine-generated texts has become increasingly blurred. This paper delves into the inquiry of identifying discernible and unique linguistic properties in texts that were written by humans, particularly uncovering the underlying discourse structures of texts beyond their surface structures. Introducing a novel methodology, we leverage hierarchical parse trees and recursive hypergraphs to unveil distinctive discourse patterns in texts produced by both LLMs and humans. Empirical findings demonstrate that, although both LLMs and humans generate distinct discourse patterns influenced by specific domains, human-written texts exhibit more structural variability, reflecting the nuanced nature of human writing in different domains. Notably, incorporating hierarchical discourse features enhances binary classifiers' overall performance in distinguishing between huma
    
[^52]: LinkNER: 使用不确定性将本地命名实体识别模型与大语言模型进行链接

    LinkNER: Linking Local Named Entity Recognition Models to Large Language Models using Uncertainty

    [https://arxiv.org/abs/2402.10573](https://arxiv.org/abs/2402.10573)

    提出了一种结合小型微调模型和大型语言模型的LinkNER框架，通过不确定性的链接策略RDC，使微调模型能够补充黑盒LLMs

    

    命名实体识别（NER）作为自然语言理解中的基本任务，直接影响着网络内容分析、搜索引擎和信息检索系统。微调后的NER模型在标准NER基准上表现出令人满意的性能。然而，由于有限的微调数据和缺乏知识，它在未见实体识别上表现不佳。因此，NER模型在网络相关应用中的可用性和可靠性受到影响。相反，像GPT-4这样的大型语言模型（LLM）具有丰富的外部知识，但研究表明它们缺乏NER任务的专业性。此外，私有和大规模权重使LLM的调整困难。为了解决这些挑战，我们提出了一个框架，结合了小型微调模型和LLMs（LinkNER），以及一种基于不确定性的链接策略RDC，使微调模型能够补充黑盒LLMs。

    arXiv:2402.10573v1 Announce Type: new  Abstract: Named Entity Recognition (NER) serves as a fundamental task in natural language understanding, bearing direct implications for web content analysis, search engines, and information retrieval systems. Fine-tuned NER models exhibit satisfactory performance on standard NER benchmarks. However, due to limited fine-tuning data and lack of knowledge, it performs poorly on unseen entity recognition. As a result, the usability and reliability of NER models in web-related applications are compromised. Instead, Large Language Models (LLMs) like GPT-4 possess extensive external knowledge, but research indicates that they lack specialty for NER tasks. Furthermore, non-public and large-scale weights make tuning LLMs difficult. To address these challenges, we propose a framework that combines small fine-tuned models with LLMs (LinkNER) and an uncertainty-based linking strategy called RDC that enables fine-tuned models to complement black-box LLMs, ach
    
[^53]: 具有偏置的直接偏好优化

    Direct Preference Optimization with an Offset

    [https://arxiv.org/abs/2402.10571](https://arxiv.org/abs/2402.10571)

    提出了一种新颖的直接偏好优化方法，即具有偏置的DPO（ODPO），在微调过程中不同对待每个偏好对。

    

    直接偏好优化（DPO）是一种成功的微调策略，用于使大型语言模型与人类偏好保持一致，而无需训练奖励模型或使用强化学习。本文提出了一种DPO的泛化形式，称为具有偏置的DPO（ODPO），在微调过程中不将每个偏好对视为相等。

    arXiv:2402.10571v1 Announce Type: cross  Abstract: Direct preference optimization (DPO) is a successful fine-tuning strategy for aligning large language models with human preferences without the need to train a reward model or employ reinforcement learning. DPO, as originally formulated, relies on binary preference data and fine-tunes a language model to increase the likelihood of a preferred response over a dispreferred response. However, not all preference pairs are equal: while in some cases the preferred response is only slightly better than the dispreferred response, there can be a stronger preference for one response when, for example, the other response includes harmful or toxic content. In this paper, we propose a generalization of DPO, termed DPO with an offset (ODPO), that does not treat every preference pair equally during fine-tuning. Intuitively, ODPO requires the difference between the likelihood of the preferred and dispreferred response to be greater than an offset valu
    
[^54]: 在InSaAF中融入安全性，通过准确性和公平性 | LLM是否已经准备好进入印度法律领域？

    InSaAF: Incorporating Safety through Accuracy and Fairness | Are LLMs ready for the Indian Legal Domain?

    [https://arxiv.org/abs/2402.10567](https://arxiv.org/abs/2402.10567)

    本研究在印度法律领域探讨了大型语言模型（LLMs）在处理社会因素时的能力，提出了结合公平性和准确性的新指标$LSS_{\beta}$，并评估了模型在二元法律推理任务中的表现以及在印度社会各种不平等方面的公平性展示。

    

    语言技术和人工智能的最新进展已经导致提出了众多语言模型，用于执行法律领域的各种任务，从预测判决到生成摘要。尽管它们具有巨大潜力，但已经证明这些模型学习并展示社会偏见，并做出不公平的预测。在这项研究中，我们探讨了当涉及社会因素时大型语言模型（LLMs）在印度法律领域执行任务的能力。我们提出了一种新颖的度量标准，$\beta$-加权的$\textit{法律安全分数($LSS_{\beta}$)}$，将LLM的公平性和准确性两个方面结合起来。我们通过考虑LLM在$\textit{二元法律推理}$任务中的表现以及其在印度社会各种不平等方面的公平展示来评估LLMs的安全性。LLaMA和LLaMA--2模型的任务表现和公平得分表明...

    arXiv:2402.10567v1 Announce Type: cross  Abstract: Recent advancements in language technology and Artificial Intelligence have resulted in numerous Language Models being proposed to perform various tasks in the legal domain ranging from predicting judgments to generating summaries. Despite their immense potential, these models have been proven to learn and exhibit societal biases and make unfair predictions. In this study, we explore the ability of Large Language Models (LLMs) to perform legal tasks in the Indian landscape when social factors are involved. We present a novel metric, $\beta$-weighted $\textit{Legal Safety Score ($LSS_{\beta}$)}$, which encapsulates both the fairness and accuracy aspects of the LLM. We assess LLMs' safety by considering its performance in the $\textit{Binary Statutory Reasoning}$ task and its fairness exhibition with respect to various axes of disparities in the Indian society. Task performance and fairness scores of LLaMA and LLaMA--2 models indicate th
    
[^55]: 通过自动爬取和对齐的句子对进行神经重述

    Neural paraphrasing by automatically crawled and aligned sentence pairs

    [https://arxiv.org/abs/2402.10558](https://arxiv.org/abs/2402.10558)

    通过自动爬取和对齐句子对，本文提出了一种神经网络重述的方法

    

    抄袭是使用其他词语重新书写输入文本的任务，而不改变原始内容的含义。会话系统可以利用自动抄袭来使对话更加自然，例如，使用不同的释义在不同的时间点谈论某个特定主题。最近，在自然语言生成（NLG）的背景下，自动生成释义的任务已经得到了解决。虽然许多现有系统只是基于规则的模型，但深度神经网络在几个NLG任务上的最新成功自然地暗示了利用这些网络来生成释义的可能性。然而，基于神经网络的抄袭的主要障碍是缺乏大规模带有对齐的句子和释义对的数据集，这些数据集是训练神经模型的有效需要。在本文中，我们提出了一种用于自动生成大规模对齐句子和释义对的方法

    arXiv:2402.10558v1 Announce Type: new  Abstract: Paraphrasing is the task of re-writing an input text using other words, without altering the meaning of the original content. Conversational systems can exploit automatic paraphrasing to make the conversation more natural, e.g., talking about a certain topic using different paraphrases in different time instants. Recently, the task of automatically generating paraphrases has been approached in the context of Natural Language Generation (NLG). While many existing systems simply consist in rule-based models, the recent success of the Deep Neural Networks in several NLG tasks naturally suggests the possibility of exploiting such networks for generating paraphrases. However, the main obstacle toward neural-network-based paraphrasing is the lack of large datasets with aligned pairs of sentences and paraphrases, that are needed to efficiently train the neural models. In this paper we present a method for the automatic generation of large align
    
[^56]: SPAR：通过长期参与注意力实现个性化基于内容的推荐

    SPAR: Personalized Content-Based Recommendation via Long Engagement Attention

    [https://arxiv.org/abs/2402.10555](https://arxiv.org/abs/2402.10555)

    SPAR是一个基于内容的推荐框架，通过利用PLM、多注意力层和注意力稀疏机制，在会话级别有效地处理长期用户参与历史，提取全面用户兴趣，实现个性化推荐。

    

    利用用户长期参与历史对个性化内容推荐至关重要。预训练语言模型（PLMs）在自然语言处理领域的成功导致它们被用于编码用户历史和候选项，将内容推荐视为文本语义匹配任务。然而，现有工作仍然在处理非常长的用户历史文本和不足的用户-物品交互方面存在困难。本文介绍了一种基于内容的推荐框架SPAR，有效应对了从长期用户参与历史中提取全面用户兴趣的挑战。它通过利用PLM、多注意力层和注意力稀疏机制以会话为基础对用户的历史进行编码。用户和物品侧特征被充分融合进行参与预测，同时保持双方的独立表示，这对于实际模型部署是有效的。

    arXiv:2402.10555v1 Announce Type: cross  Abstract: Leveraging users' long engagement histories is essential for personalized content recommendations. The success of pretrained language models (PLMs) in NLP has led to their use in encoding user histories and candidate items, framing content recommendations as textual semantic matching tasks. However, existing works still struggle with processing very long user historical text and insufficient user-item interaction. In this paper, we introduce a content-based recommendation framework, SPAR, which effectively tackles the challenges of holistic user interest extraction from the long user engagement history. It achieves so by leveraging PLM, poly-attention layers and attention sparsity mechanisms to encode user's history in a session-based manner. The user and item side features are sufficiently fused for engagement prediction while maintaining standalone representations for both sides, which is efficient for practical model deployment. Mor
    
[^57]: 不规则文本中动态基于方面的总结标准： Disordered-DABS基准测试

    Disordered-DABS: A Benchmark for Dynamic Aspect-Based Summarization in Disordered Texts

    [https://arxiv.org/abs/2402.10554](https://arxiv.org/abs/2402.10554)

    Disordered-DABS是针对不规则文本中动态基于方面的总结而设计的新基准测试，挑战了现有总结模型的独特性。

    

    方面为基础的总结已经取得了重要进展，尤其是在结构化文本中。然而，总结不规则的大规模文本，比如社交媒体和客户反馈中发现的文本，仍然是一个重大挑战。目前的研究主要针对结构化文本中的预定义方面，忽略了动态和无序环境的复杂性。为了弥补这一差距，我们引入了Disordered-DABS，这是一个新颖的面向动态方面的总结基准测试，专为非结构化文本量身定制。通过调整现有数据集以提高成本效率和可扩展性，我们的综合实验和详细的人类评估表明，Disordered-DABS对当代总结模型提出了独特的挑战，包括GPT-3.5等最先进的语言模型。

    arXiv:2402.10554v1 Announce Type: new  Abstract: Aspect-based summarization has seen significant advancements, especially in structured text. Yet, summarizing disordered, large-scale texts, like those found in social media and customer feedback, remains a significant challenge. Current research largely targets predefined aspects within structured texts, neglecting the complexities of dynamic and disordered environments. Addressing this gap, we introduce Disordered-DABS, a novel benchmark for dynamic aspect-based summarization tailored to unstructured text. Developed by adapting existing datasets for cost-efficiency and scalability, our comprehensive experiments and detailed human evaluations reveal that Disordered-DABS poses unique challenges to contemporary summarization models, including state-of-the-art language models such as GPT-3.5.
    
[^58]: Conversational SimulMT: 基于大型语言模型的高效同时翻译

    Conversational SimulMT: Efficient Simultaneous Translation with Large Language Models

    [https://arxiv.org/abs/2402.10552](https://arxiv.org/abs/2402.10552)

    通过对话式SimulMT框架，本文提高了基于LLM的SimulMT推理效率，在保持翻译质量的同时实现与专门的SimulMT模型相近的计算延迟。

    

    同声机器翻译（SimulMT）在翻译质量和延迟之间存在挑战性的权衡。最近的研究表明，大型语言模型（LLMs）在SimulMT任务中可以取得很好的表现。然而，这往往是以推理成本和延迟的增加为代价的。本文提出了一种对话式SimulMT框架，通过基于多轮对话的解码来提高基于LLM的SimulMT的推理效率。我们在两个SimulMT基准上使用Llama2-7b-chat进行实验，结果表明LLM在翻译质量上具有优势，同时实现与专门的SimulMT模型相当的计算延迟。

    arXiv:2402.10552v1 Announce Type: new  Abstract: Simultaneous machine translation (SimulMT) presents a challenging trade-off between translation quality and latency. Recent studies have shown that LLMs can achieve good performance in SimulMT tasks. However, this often comes at the expense of high inference cost and latency. In this paper, we propose a conversational SimulMT framework to enhance the inference efficiency of LLM-based SimulMT through multi-turn-dialogue-based decoding. Our experiments with Llama2-7b-chat on two SimulMT benchmarks demonstrate the superiority of LLM in translation quality while achieving comparable computational latency to specialized SimulMT models.
    
[^59]: 消除否定导致的强幻觉

    Strong hallucinations from negation and how to fix them

    [https://arxiv.org/abs/2402.10543](https://arxiv.org/abs/2402.10543)

    论文针对语言模型在推理中造成的强幻觉问题，提出了一种处理否定的新方法，可以改善模型性能而无需使用稀疏负数据训练。

    

    尽管语言模型（LMs）在许多任务上表现出色，但仍然在推理方面存在困难，有时会提供由于逻辑不连贯而不可能成立的响应。我们称这种响应为\textit{强幻觉}，并证明它们源于LM计算其内部表示的逻辑运算符和从这些表示中产生的输出。重点关注否定，我们提供了一种新颖的解决方案，其中否定不是作为潜在表示的另一个元素，而是作为\textit{LM潜在表示上的一个操作，约束它们可能的演变方式}。我们展示了我们的方法改善了在带否定的填空提示和自然语言推理任务中的模型性能，而无需对稀疏负数据进行训练。

    arXiv:2402.10543v1 Announce Type: cross  Abstract: Despite great performance on many tasks, language models (LMs) still struggle with reasoning, sometimes providing responses that cannot possibly be true because they stem from logical incoherence. We call such responses \textit{strong hallucinations} and prove that they follow from an LM's computation of its internal representations for logical operators and outputs from those representations. Focusing on negation, we provide a novel solution in which negation is treated not as another element of a latent representation, but as \textit{an operation over an LM's latent representations that constrains how they may evolve}. We show that our approach improves model performance in cloze prompting and natural language inference tasks with negation without requiring training on sparse negative data.
    
[^60]: LLM生成的解释的特性和挑战

    Properties and Challenges of LLM-Generated Explanations

    [https://arxiv.org/abs/2402.10532](https://arxiv.org/abs/2402.10532)

    该研究探讨了大型语言模型生成的解释在多领域指导微调数据集上的特性，发现生成的解释表现出选择性和包含说明性元素，但较少是主观或误导性的。

    

    大型语言模型（LLMs）的自我合理化能力在限定环境中得到了探索，使用特定任务/数据集。然而，当前LLMs并不（仅）依赖于特定注释的数据；然而，它们经常解释它们的输出。生成的解释的特性受预训练语料库和用于指导微调的目标数据的影响。由于预训练语料库包含大量野外人类编写的解释，我们假设LLMs采用了人类解释的共同特性。通过分析多域指导微调数据集的输出，我们发现生成的解释表现出选择性并包含说明性元素，但很少是主观或误导性的。我们讨论了属性存在或缺失的原因和后果。特别是，我们概述了根据LLMs预训练语料库和微调数据的性质，这些属性存在或缺失的积极和消极影响。

    arXiv:2402.10532v1 Announce Type: cross  Abstract: The self-rationalising capabilities of large language models (LLMs) have been explored in restricted settings, using task/specific data sets. However, current LLMs do not (only) rely on specifically annotated data; nonetheless, they frequently explain their outputs. The properties of the generated explanations are influenced by the pre-training corpus and by the target data used for instruction fine-tuning. As the pre-training corpus includes a large amount of human-written explanations "in the wild", we hypothesise that LLMs adopt common properties of human explanations. By analysing the outputs for a multi-domain instruction fine-tuning data set, we find that generated explanations show selectivity and contain illustrative elements, but less frequently are subjective or misleading. We discuss reasons and consequences of the properties' presence or absence. In particular, we outline positive and negative implications depending on the 
    
[^61]: 我们能否逐步验证错误答案检测？

    Can We Verify Step by Step for Incorrect Answer Detection?

    [https://arxiv.org/abs/2402.10528](https://arxiv.org/abs/2402.10528)

    通过推理链来预测大型语言模型输出的准确性，我们引入了一个新的基准R2PE，并提出了处理可辨识性评分（PDS）框架。

    

    Chain-of-Thought（CoT）提示在增强大型语言模型（LLMs）的推理能力方面取得了重大进展。先前的研究开发了各种扩展的CoT，主要集中在增强最终任务的性能上。此外，已经有研究评估了CoT中推理链的质量。这引发了一个有趣的问题：通过仔细审查它们生成的推理链，是否可以预测LLMs输出的准确性？为了回答这个研究问题，我们引入了一个基准，R2PE，专门设计用于探究不同领域涵盖五个不同推理任务中推理链与性能之间的关系。该基准旨在基于推理步骤衡量LLMs最终输出的虚假性。为了充分利用多个推理链中的信息，我们提出了打败常识分数（PDS）框架。

    arXiv:2402.10528v1 Announce Type: cross  Abstract: Chain-of-Thought (CoT) prompting has marked a significant advancement in enhancing the reasoning capabilities of large language models (LLMs). Previous studies have developed various extensions of CoT, which focus primarily on enhancing end-task performance. In addition, there has been research on assessing the quality of reasoning chains in CoT. This raises an intriguing question: Is it possible to predict the accuracy of LLM outputs by scrutinizing the reasoning chains they generate? To answer this research question, we introduce a benchmark, R2PE, designed specifically to explore the relationship between reasoning chains and performance in various reasoning tasks spanning five different domains. This benchmark aims to measure the falsehood of the final output of LLMs based on the reasoning steps. To make full use of information in multiple reasoning chains, we propose the process discernibility score (PDS) framework that beats the a
    
[^62]: 生物医学问题回答中的零样本采样对抗实体

    Zero-shot sampling of adversarial entities in biomedical question answering

    [https://arxiv.org/abs/2402.10527](https://arxiv.org/abs/2402.10527)

    在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。

    

    大型语言模型（LLM）中参数域知识的增加深度推动它们在现实世界应用中的快速部署。在高风险和知识密集型任务中，理解模型的漏洞对于量化模型预测的可信度和规范其使用至关重要。最近发现在自然语言处理任务中作为对抗示例的命名实体引发了关于它们在其他环境中可能的伪装的疑问。在这里，我们提出了一种在嵌入空间中的幂缩放距离加权采样方案，以发现多样化的对抗实体作为干扰因素。我们展示了它在生物医学主题的对抗性问题回答中优于随机采样的优势。我们的方法使得可以探索攻击表面上的不同区域，这揭示了两种在特征上明显不同的对抗性实体的制度。此外，我们展示了攻击方式如何...

    arXiv:2402.10527v1 Announce Type: new  Abstract: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. In high-stakes and knowledge-intensive tasks, understanding model vulnerabilities is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples in natural language processing tasks raises questions about their potential guises in other settings. Here, we propose a powerscaled distance-weighted sampling scheme in embedding space to discover diverse adversarial entities as distractors. We demonstrate its advantage over random sampling in adversarial question answering on biomedical topics. Our approach enables the exploration of different regions on the attack surface, which reveals two regimes of adversarial entities that markedly differ in their characteristics. Moreover, we show that the attacks su
    
[^63]: LLM比较器：用于大型语言模型并行评估的可视化分析

    LLM Comparator: Visual Analytics for Side-by-Side Evaluation of Large Language Models

    [https://arxiv.org/abs/2402.10524](https://arxiv.org/abs/2402.10524)

    LLM Comparator是一种用于交互式分析自动并行评估结果的新型可视化工具，支持用户理解模型表现优劣和不同之处，解决了大型语言模型评估中的可扩展性和可解释性挑战。

    

    自动并行评估已成为评估大型语言模型（LLMs）响应质量的一种有前途的方法。然而，分析这种评估方法的结果存在可扩展性和可解释性挑战。本文提出了LLM比较器，这是一种新颖的可视化分析工具，用于交互式地分析自动并行评估结果。该工具支持用户进行交互式工作流，以了解为什么和何时模型比基准模型表现更好或更差，以及两个模型的响应在质量上有何不同。我们通过与一家大型科技公司的研究人员和工程师密切合作，迭代设计和开发了该工具。本文详细介绍了我们识别的用户挑战、该工具的设计和开发，以及定期评估其模型的参与者的观察研究。

    arXiv:2402.10524v1 Announce Type: cross  Abstract: Automatic side-by-side evaluation has emerged as a promising approach to evaluating the quality of responses from large language models (LLMs). However, analyzing the results from this evaluation approach raises scalability and interpretability challenges. In this paper, we present LLM Comparator, a novel visual analytics tool for interactively analyzing results from automatic side-by-side evaluation. The tool supports interactive workflows for users to understand when and why a model performs better or worse than a baseline model, and how the responses from two models are qualitatively different. We iteratively designed and developed the tool by closely working with researchers and engineers at a large technology company. This paper details the user challenges we identified, the design and development of the tool, and an observational study with participants who regularly evaluate their models.
    
[^64]: 通过主动偏好优化实现经验证的样本效率的RLHF

    Provably Sample Efficient RLHF via Active Preference Optimization

    [https://arxiv.org/abs/2402.10500](https://arxiv.org/abs/2402.10500)

    通过Active Preference Optimization算法，在Bradley-Terry-Luce偏好模型下实现了RLHF的样本效率提高，优化了对提示收集偏好数据的策略。

    

    强化学习从人类反馈（RLHF）在将大型语言模型（LLMs）与人类偏好相一致方面至关重要。虽然这些对齐的生成模型已经在各种任务中展示出令人印象深刻的能力，但是依赖高质量的人类偏好数据在实际RLHF实施中构成了昂贵的瓶颈。因此，需要更好和自适应的数据收集策略。为此，我们将RLHF以上下文偏好赌博机问题的形式框定，其中提示作为上下文，并表明通过随机选择提示收集偏好数据的天真方式导致一个在奖励方面具有$\Omega(1)$次优性差距的策略。然后，我们提出了$\textit{Active Preference Optimization}$（$\texttt{APO}$）算法，该算法积极选择提示以收集偏好数据。在Bradley-Terry-Luce（BTL）偏好模型下，\texttt{APO}实现了样本效率，而不会妥协于polic

    arXiv:2402.10500v1 Announce Type: cross  Abstract: Reinforcement Learning from Human Feedback (RLHF) is pivotal in aligning Large Language Models (LLMs) with human preferences. While these aligned generative models have demonstrated impressive capabilities across various tasks, the dependence on high-quality human preference data poses a costly bottleneck in practical implementation of RLHF. Hence better and adaptive strategies for data collection is needed. To this end, we frame RLHF as a contextual preference bandit problem with prompts as contexts and show that the naive way of collecting preference data by choosing prompts uniformly at random leads to a policy that suffers an $\Omega(1)$ suboptimality gap in rewards. Then we propose $\textit{Active Preference Optimization}$ ($\texttt{APO}$), an algorithm that actively selects prompts to collect preference data. Under the Bradley-Terry-Luce (BTL) preference model, \texttt{APO} achieves sample efficiency without compromising on polic
    
[^65]: 比较多语言生成中幻觉检测指标

    Comparing Hallucination Detection Metrics for Multilingual Generation

    [https://arxiv.org/abs/2402.10496](https://arxiv.org/abs/2402.10496)

    本研究比较了多语言生成中不同幻觉检测指标的效果，发现基于自然语言推理（NLI）的指标在高资源语言的句子级别表现良好，但通常无法检测到原子事实幻觉。

    

    尽管已提出许多针对英文文本的自动幻觉检测技术，但它们在多语言环境中的效果尚未被探索。本文旨在填补对这些幻觉检测指标在非英语语言上表现如何的认识上的差距。我们评估了各种检测指标的有效性，包括诸如ROUGE和命名实体重叠以及基于自然语言推理（NLI）的指标，在多种语言的传记摘要中检测幻觉；我们还评估这些不同指标之间的相关性，以判断它们是否衡量相同的现象。我们的实证分析显示，虽然词汇指标显示出有限的有效性，但基于NLI的指标在高资源语言中在句子级别表现良好。相反，NLI-based指标通常无法检测到原子事实幻觉。我们的研究结果突显了多语言幻觉检测中的现有差距。

    arXiv:2402.10496v1 Announce Type: cross  Abstract: While many automatic hallucination detection techniques have been proposed for English texts, their effectiveness in multilingual contexts remains unexplored. This paper aims to bridge the gap in understanding how these hallucination detection metrics perform on non-English languages. We evaluate the efficacy of various detection metrics, including lexical metrics like ROUGE and Named Entity Overlap and Natural Language Inference (NLI)-based metrics, at detecting hallucinations in biographical summaries in many languages; we also evaluate how correlated these different metrics are to gauge whether they measure the same phenomena. Our empirical analysis reveals that while lexical metrics show limited effectiveness, NLI-based metrics perform well in high-resource languages at the sentence level. In contrast, NLI-based metrics often fail to detect atomic fact hallucinations. Our findings highlight existing gaps in multilingual hallucinati
    
[^66]: 基于表情符号的加密资产市场反应

    Emoji Driven Crypto Assets Market Reactions

    [https://arxiv.org/abs/2402.10481](https://arxiv.org/abs/2402.10481)

    该研究利用GPT-4和BERT模型进行多模态情感分析，发现基于表情符号情绪的策略可以帮助避免市场下挫并稳定回报。

    

    在加密货币领域，诸如Twitter之类的社交媒体平台已经成为影响市场趋势和投资者情绪的关键因素。在我们的研究中，我们利用GPT-4和经过微调的基于BERT模型的多模态情感分析，重点关注表情符号情绪对加密货币市场的影响。通过将表情符号转化为可量化的情感数据，我们将这些见解与BTC价格和VCRIX指数等关键市场指标进行了相关联。这种方法可以用于开发旨在利用社交媒体元素识别和预测市场趋势的交易策略。关键是，我们的研究结果表明，基于表情符号情绪的策略可以有助于避免重大市场下挫，并有助于回报的稳定。这项研究强调了将先进的基于人工智能的分析整合到金融策略中的实际益处，并提供了一种新的方式来看待市场预测。

    arXiv:2402.10481v1 Announce Type: cross  Abstract: In the burgeoning realm of cryptocurrency, social media platforms like Twitter have become pivotal in influencing market trends and investor sentiments. In our study, we leverage GPT-4 and a fine-tuned transformer-based BERT model for a multimodal sentiment analysis, focusing on the impact of emoji sentiment on cryptocurrency markets. By translating emojis into quantifiable sentiment data, we correlate these insights with key market indicators like BTC Price and the VCRIX index. This approach may be fed into the development of trading strategies aimed at utilizing social media elements to identify and forecast market trends. Crucially, our findings suggest that strategies based on emoji sentiment can facilitate the avoidance of significant market downturns and contribute to the stabilization of returns. This research underscores the practical benefits of integrating advanced AI-driven analyses into financial strategies, offering a nuan
    
[^67]: 将大型语言模型作为零-shot对话状态追踪器通过函数调用

    Large Language Models as Zero-shot Dialogue State Tracker through Function Calling

    [https://arxiv.org/abs/2402.10466](https://arxiv.org/abs/2402.10466)

    本研究提出了一种通过函数调用将大型语言模型用于零-shot对话状态追踪的新方法，能够在任务导向对话中取得出色的性能，适应不同领域而无需大量数据收集或模型调整。

    

    大型语言模型（LLMs）在会话系统中日益普遍，这是因为它们在一般情境中具有先进的理解和生成能力。然而，在需要不仅进行响应生成还需要在特定任务和领域内进行有效对话状态追踪（DST）的任务导向对话（TOD）中，它们的有效性仍不尽人意。在这项工作中，我们提出了一种通过函数调用解决LLMs中的DST的新方法FnCTOD。这种方法改进了零-shot DST，使其能够适应各种领域，而无需进行大量数据收集或模型调整。我们的实验结果表明，我们的方法在使用开源或专有LLMs时都取得了出色的性能：通过上下文提示，使得各种7B或13B参数模型超越了之前由ChatGPT实现的最新技术成果（SOTA）的水平，并提高了ChatGPT的性能，击败了

    arXiv:2402.10466v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly prevalent in conversational systems due to their advanced understanding and generative capabilities in general contexts. However, their effectiveness in task-oriented dialogues (TOD), which requires not only response generation but also effective dialogue state tracking (DST) within specific tasks and domains, remains less satisfying. In this work, we propose a novel approach FnCTOD for solving DST with LLMs through function calling. This method improves zero-shot DST, allowing adaptation to diverse domains without extensive data collection or model tuning. Our experimental results demonstrate that our approach achieves exceptional performance with both modestly sized open-source and also proprietary LLMs: with in-context prompting it enables various 7B or 13B parameter models to surpass the previous state-of-the-art (SOTA) achieved by ChatGPT, and improves ChatGPT's performance beating the
    
[^68]: QDyLoRA: 高效的大型语言模型调优的量化动态低秩适应

    QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning

    [https://arxiv.org/abs/2402.10462](https://arxiv.org/abs/2402.10462)

    本论文提出了一种名为QDyLoRA的高效量化动态低秩适应方法，能够在大型语言模型的预定义秩上实现有效微调，与QLoRA相竞争，并且在采用其最佳秩时表现更好。

    

    Finetuning大型语言模型需要巨大的GPU内存，限制了获取更大模型的选择。虽然命名为QLoRA的低秩适应技术的量化版本显著缓解了这一问题，但是找到高效的LoRA秩仍然具有挑战性。此外，QLoRA是在预定义的秩上训练的，因此，在不需要进一步微调步骤的情况下无法重新配置为其较低的秩。本文提出了QDyLoRA-Quantized Dynamic Low-Rank Adaptation-，作为一种用于动态低秩适应的高效量化方法。受Dynamic LoRA的启发，QDyLoRA能够在一组预定义的LoRA秩上有效地微调LLMs。通过一轮微调，QDyLoRA能够在单个32 GB V100-GPU上为1到64个秩的Falcon-40b进行微调。实验结果表明，QDyLoRA与QLoRA具有竞争力，在使用其最佳秩时表现优越。

    arXiv:2402.10462v1 Announce Type: cross  Abstract: Finetuning large language models requires huge GPU memory, restricting the choice to acquire Larger models. While the quantized version of the Low-Rank Adaptation technique, named QLoRA, significantly alleviates this issue, finding the efficient LoRA rank is still challenging. Moreover, QLoRA is trained on a pre-defined rank and, therefore, cannot be reconfigured for its lower ranks without requiring further fine-tuning steps. This paper proposes QDyLoRA -Quantized Dynamic Low-Rank Adaptation-, as an efficient quantization approach for dynamic low-rank adaptation. Motivated by Dynamic LoRA, QDyLoRA is able to efficiently finetune LLMs on a set of pre-defined LoRA ranks. QDyLoRA enables fine-tuning Falcon-40b for ranks 1 to 64 on a single 32 GB V100-GPU through one round of fine-tuning. Experimental results show that QDyLoRA is competitive to QLoRA and outperforms when employing its optimal rank.
    
[^69]: 引导情感支持对话的大型语言模型进行长时间对话

    Steering Conversational Large Language Models for Long Emotional Support Conversations

    [https://arxiv.org/abs/2402.10453](https://arxiv.org/abs/2402.10453)

    引入了Strategy-Relevant Attention（SRA）度量，评估大型语言模型在情感支持对话中遵循战略提示的有效性，研究发现应用SRA指导的提示可提高战略依从性，从而使长时间对话更可靠地展示所需的情感支持策略。

    

    在这项研究中，我们解决了大型语言模型（LLMs）在长时间对话中一贯遵循情感支持策略的挑战。我们引入了Strategy-Relevant Attention（SRA）度量，这是一个模型不可知的指标，旨在评估LLMs在情感支持环境中遵循战略提示的有效性。通过使用LLaMA模型分析情感支持对话数据集（ESConv）中的对话，我们证明SRA与模型在整个互动过程中维持所述策略能力密切相关。我们的研究结果显示，应用基于SRA的提示可提高战略依从性，导致对话更可靠地展示长时间对话中所需的情感支持策略。此外，我们贡献了一个全面的、多分支的合成对话数据集，适用于ESConv，其中包含各种策略内容。

    arXiv:2402.10453v1 Announce Type: new  Abstract: In this study, we address the challenge of consistently following emotional support strategies in long conversations by large language models (LLMs). We introduce the Strategy-Relevant Attention (SRA) metric, a model-agnostic measure designed to evaluate the effectiveness of LLMs in adhering to strategic prompts in emotional support contexts. By analyzing conversations within the Emotional Support Conversations dataset (ESConv) using LLaMA models, we demonstrate that SRA is significantly correlated with a model's ability to sustain the outlined strategy throughout the interactions. Our findings reveal that the application of SRA-informed prompts leads to enhanced strategic adherence, resulting in conversations that more reliably exhibit the desired emotional support strategies over longer conversations. Furthermore, we contribute a comprehensive, multi-branch synthetic conversation dataset for ESConv, featuring a variety of strategy cont
    
[^70]: 增量序列标记：两种转变的故事

    Incremental Sequence Labeling: A Tale of Two Shifts

    [https://arxiv.org/abs/2402.10447](https://arxiv.org/abs/2402.10447)

    提出了一种名为IS3的框架，旨在解决增量序列标记任务中的E2O和O2E两种重要的语义转变，通过使用知识蒸馏来维持对旧实体的判别能力。

    

    增量序列标记任务涉及在保留对先前类别知识的同时，随时间不断学习新类别。我们的研究确定了两种重要的语义转变：E2O（模型将旧实体错误标记为非实体）和O2E（模型将非实体或旧实体标记为新实体）。先前的研究主要集中在解决E2O问题上，忽视了O2E问题。这种忽略导致模型在学习过程中对新数据样本进行分类时存在偏见，认为它们属于新类别。为了解决这些挑战，我们提出了一种新颖的框架，即无语义转变的增量顺序标记（IS3）。受到已确定的语义转变（E2O和O2E）的启发，IS3旨在缓解模型中的灾难性遗忘。至于E2O问题，我们使用知识蒸馏来维持模型对旧实体的判别能力。

    arXiv:2402.10447v1 Announce Type: new  Abstract: The incremental sequence labeling task involves continuously learning new classes over time while retaining knowledge of the previous ones. Our investigation identifies two significant semantic shifts: E2O (where the model mislabels an old entity as a non-entity) and O2E (where the model labels a non-entity or old entity as a new entity). Previous research has predominantly focused on addressing the E2O problem, neglecting the O2E issue. This negligence results in a model bias towards classifying new data samples as belonging to the new class during the learning process. To address these challenges, we propose a novel framework, Incremental Sequential Labeling without Semantic Shifts (IS3). Motivated by the identified semantic shifts (E2O and O2E), IS3 aims to mitigate catastrophic forgetting in models. As for the E2O problem, we use knowledge distillation to maintain the model's discriminative ability for old entities. Simultaneously, t
    
[^71]: 我不是他们：大型语言模型中的流动身份和持久的外群体偏见

    I Am Not Them: Fluid Identities and Persistent Out-group Bias in Large Language Models

    [https://arxiv.org/abs/2402.10436](https://arxiv.org/abs/2402.10436)

    论文研究了ChatGPT在不同语言环境中的文化偏见表现，发现当其拥抱特定社会身份时，会区分内外群体，偏好内群体价值观而抵制外群体价值观。

    

    我们探讨了ChatGPT在三种西方语言（即英语、德语和法语）和三种东方语言（即中文、日语和韩语）中的文化偏见-个人主义与集体主义。当ChatGPT在西方语言中采用个人主义人格时，其集体主义评分（即外群体价值观）呈现出更为消极的趋势，超越了对个人主义（即内群体价值观）的积极取向。相反，当在东方语言中向ChatGPT指定集体主义人格时，出现了类似的模式，对个人主义（即外群体价值观）产生了更为负面的反应，而对集体主义（即内群体价值观）持积极态度。结果表明，当注入特定社会身份时，ChatGPT能够识别内群体和外群体，接受内群体价值观同时摒弃外群体价值观。值得注意的是，对外群体的消极态度引发了偏见和歧视。

    arXiv:2402.10436v1 Announce Type: new  Abstract: We explored cultural biases-individualism vs. collectivism-in ChatGPT across three Western languages (i.e., English, German, and French) and three Eastern languages (i.e., Chinese, Japanese, and Korean). When ChatGPT adopted an individualistic persona in Western languages, its collectivism scores (i.e., out-group values) exhibited a more negative trend, surpassing their positive orientation towards individualism (i.e., in-group values). Conversely, when a collectivistic persona was assigned to ChatGPT in Eastern languages, a similar pattern emerged with more negative responses toward individualism (i.e., out-group values) as compared to collectivism (i.e., in-group values). The results indicate that when imbued with a particular social identity, ChatGPT discerns in-group and out-group, embracing in-group values while eschewing out-group values. Notably, the negativity towards the out-group, from which prejudices and discrimination arise,
    
[^72]: 更小的语言模型可以为更大的语言模型选择指导调整训练数据

    Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models

    [https://arxiv.org/abs/2402.10430](https://arxiv.org/abs/2402.10430)

    更小的语言模型可以根据样本学习百分比自主选择高质量训练数据，支持更大语言模型的指导调整，实现相媲美甚至优于在整个数据集训练的性能。

    

    指导调整语言模型已成为使它们适用于一般用途的关键步骤。通常，这个过程涉及大量的大规模数据集训练，导致高昂的训练成本。本文引入了一种基于样本学习百分比的新颖训练数据选择方法。我们断言目前的语言模型具有自主选择高质量训练数据的能力，从而导致与在整个数据集上训练相比具有可比或更好性能。我们的实验涵盖了不同规模的模型，揭示出这一特征适用于从1B（小）到13B（大）大小的模型。此外，我们展示了一个有趣的发现，即数据难度在不同模型大小之间传递，并且更小的350M模型可以有效地筛选出包含困难样本的高质量训练数据，用于更大的13B模型，导致一个与在整个数据集上训练相比同样或更优秀的指导调整模型。

    arXiv:2402.10430v1 Announce Type: new  Abstract: Instruction-tuning language models has become a crucial step in aligning them for general use. Typically, this process involves extensive training on large datasets, incurring high training costs. In this paper, we introduce a novel training data selection based on the learning percentage of the samples. We assert that current language models possess the capability to autonomously select high-quality training data, leading to comparable or improved performance compared to training on the entire dataset. Our experiments span different-sized models, revealing that this characteristic holds for models ranging from 1B (small) to 13B (large) in size. Moreover, we demonstrate an interesting finding that the data hardness transfers across model sizes, and a smaller 350M model can effectively curate high-quality training data with hard samples for a larger 13B model, resulting in an equally or superior instruction-tuned model compared to trainin
    
[^73]: 评估和改进口语理解中的持续学习

    Evaluating and Improving Continual Learning in Spoken Language Understanding

    [https://arxiv.org/abs/2402.10427](https://arxiv.org/abs/2402.10427)

    提出了一种评估方法来统一评估口语理解中的持续学习算法在稳定性、可塑性和泛化能力方面的整体表现，并展示了引入不同知识蒸馏如何改善模型性能。

    

    持续学习已经成为各种任务中越来越重要的挑战，包括口语理解。在口语理解中，其目标是有效处理新概念的出现和不断演变的环境。持续学习算法的评估通常涉及评估模型的稳定性、可塑性和泛化能力作为标准的基本方面。然而，现有的持续学习指标主要集中在其中一个或两个属性上。它们忽视了整体表现在所有任务上，并没有充分解开模型内的可塑性与稳定性/泛化能力之间的权衡。在本研究中，我们提出了一种评估方法，可以在持续学习中统一评估稳定性、可塑性和泛化能力。通过采用所提出的度量标准，我们演示了如何引入各种知识蒸馏来改进...

    arXiv:2402.10427v1 Announce Type: cross  Abstract: Continual learning has emerged as an increasingly important challenge across various tasks, including Spoken Language Understanding (SLU). In SLU, its objective is to effectively handle the emergence of new concepts and evolving environments. The evaluation of continual learning algorithms typically involves assessing the model's stability, plasticity, and generalizability as fundamental aspects of standards. However, existing continual learning metrics primarily focus on only one or two of the properties. They neglect the overall performance across all tasks, and do not adequately disentangle the plasticity versus stability/generalizability trade-offs within the model. In this work, we propose an evaluation methodology that provides a unified evaluation on stability, plasticity, and generalizability in continual learning. By employing the proposed metric, we demonstrate how introducing various knowledge distillations can improve diffe
    
[^74]: DELL: 基于LLM的虚假信息检测生成反应和解释

    DELL: Generating Reactions and Explanations for LLM-Based Misinformation Detection

    [https://arxiv.org/abs/2402.10426](https://arxiv.org/abs/2402.10426)

    DELL提出了一个新的方法，将LLMs整合到虚假信息检测的管道中，通过生成新闻反应和解释来提升对新闻文章真实性的判断准确性。

    

    大型语言模型受事实性和幻觉方面的挑战所限，因此无法直接用于新闻文章真实性的判断，而事实准确性是至关重要的。在这项工作中，我们提出了DELL，它确定了虚假信息检测中LLM可以作为管道的一部分的三个关键阶段：1）LLM可以生成新闻反应来代表不同视角并模拟用户-新闻交互网络；2）LLM可以为代理任务（如情感、立场）生成解释，以丰富新闻文章的背景并产生专门研究新闻不同方面的专家；3）LLM可以合并任务特定的专家，并通过结合不同专家的预测和置信度分数来提供整体预测。对七个数据集进行的大量实验表明，DELL的性能优于现有的基线方法。

    arXiv:2402.10426v1 Announce Type: new  Abstract: Large language models are limited by challenges in factuality and hallucinations to be directly employed off-the-shelf for judging the veracity of news articles, where factual accuracy is paramount. In this work, we propose DELL that identifies three key stages in misinformation detection where LLMs could be incorporated as part of the pipeline: 1) LLMs could \emph{generate news reactions} to represent diverse perspectives and simulate user-news interaction networks; 2) LLMs could \emph{generate explanations} for proxy tasks (e.g., sentiment, stance) to enrich the contexts of news articles and produce experts specializing in various aspects of news understanding; 3) LLMs could \emph{merge task-specific experts} and provide an overall prediction by incorporating the predictions and confidence scores of varying experts. Extensive experiments on seven datasets with three LLMs demonstrate that DELL outperforms state-of-the-art baselines by u
    
[^75]: 使用鹈鹕汤框架理解上下文学习

    Understanding In-Context Learning with a Pelican Soup Framework

    [https://arxiv.org/abs/2402.10424](https://arxiv.org/abs/2402.10424)

    提出了一个鹈鹕汤框架，包括常识知识库、自然语言分类任务的形式化以及意义关联的概念，并建立了一个$O(1/T)$的上下文学习损失界限，能够解释对未见任务的泛化。

    

    许多现有关于自然语言处理中的上下文学习的理论分析是基于潜变量模型的，它们存在理论与实践之间的差距。我们旨在通过提出一个理论框架，即鹈鹕汤框架，来弥合这些差距。在这个框架中，我们引入了（1）常识知识库的概念，（2）自然语言分类任务的一般形式化，以及（3）意义关联的概念。在这个框架下，我们可以建立一个$\mathcal{O}(1/T)$的上下文学习损失界限，这里$T$是演示中示例-标签对的数量。与先前的作品相比，我们的界限反映了动词选择和指令调整的影响。一个额外的"原子概念"概念使我们的框架能够解释对语言模型训练数据中未见任务的泛化。最后，我们提出了一个玩具设置，Calcutec，

    arXiv:2402.10424v1 Announce Type: cross  Abstract: Many existing theoretical analyses of in-context learning for natural language processing are based on latent variable models that leaves gaps between theory and practice. We aim to close these gaps by proposing a theoretical framework, the Pelican Soup Framework. In this framework, we introduce (1) the notion of a common sense knowledge base, (2) a general formalism for natural language classification tasks, and the notion of (3) meaning association. Under this framework, we can establish a $\mathcal{O}(1/T)$ loss bound for in-context learning, where $T$ is the number of example-label pairs in the demonstration. Compared with previous works, our bound reflects the effect of the choice of verbalizers and the effect of instruction tuning. An additional notion of \textit{atom concepts} makes our framework possible to explain the generalization to tasks unseen in the language model training data. Finally, we propose a toy setup, Calcutec,
    
[^76]: 推动零-shot端到端语音翻译的极限

    Pushing the Limits of Zero-shot End-to-End Speech Translation

    [https://arxiv.org/abs/2402.10422](https://arxiv.org/abs/2402.10422)

    引入了ZeroSwot方法，实现了零-shot ST，通过CTC压缩和最优传输，仅利用ASR数据训练语音编码器，并与多语言MT模型在推断时无缝集成，实现直接从语音到文本的翻译。

    

    数据稀缺和语音与文本模态之间的模态差距是端到端语音翻译（ST）系统的两个主要障碍，从而阻碍了其性能。 以往的工作尝试通过利用外部MT数据和优化距离度量来减轻这些挑战，从而使语音-文本表示更加接近。 然而，通常需要一些ST数据才能获得竞争性结果。 出于这个原因，我们介绍了ZeroSwot，这是一种零-shot ST方法，可以在没有任何配对的ST数据的情况下弥合模态差距。 利用一种新颖的CTC压缩和最优传输技术，我们只使用ASR数据训练语音编码器，以与一个大规模多语言MT模型的表示空间进行对齐。 语音编码器在推断时与MT模型无缝集成，使得可以直接在所有MT模型支持的语言中从语音翻译为文本。 我们的实验表明，我们可以有效地平滑地关闭m模态间的空间.

    arXiv:2402.10422v1 Announce Type: new  Abstract: Data scarcity and the modality gap between the speech and text modalities are two major obstacles of end-to-end Speech Translation (ST) systems, thus hindering their performance. Prior work has attempted to mitigate these challenges by leveraging external MT data and optimizing distance metrics that bring closer the speech-text representations. However, achieving competitive results typically requires some ST data. For this reason, we introduce ZeroSwot, a method for zero-shot ST that bridges the modality gap without any paired ST data. Leveraging a novel CTC compression and Optimal Transport, we train a speech encoder using only ASR data, to align with the representation space of a massively multilingual MT model. The speech encoder seamlessly integrates with the MT model at inference, enabling direct translation from speech to text, across all languages supported by the MT model. Our experiments show that we can effectively close the m
    
[^77]: 将关于信念的语言接地于贝叶斯心灵理论

    Grounding Language about Belief in a Bayesian Theory-of-Mind

    [https://arxiv.org/abs/2402.10416](https://arxiv.org/abs/2402.10416)

    语义基础置于贝叶斯心灵理论中，通过模拟人们共同推断出解释代理人行为的一致性目标、信念和计划集合，再通过认识逻辑评估有关代理人信念的陈述，解释了人类信念归因的分级性和组合性，以及其与目标和计划的密切联系。

    

    尽管信念是无法直接观察的心理状态，人类常常使用丰富的组合语言来描述他人的想法和知识。这项研究通过将信念陈述的语义基础置于贝叶斯心灵理论中，为解释人类如何解释他人隐藏的认识内容迈出了一步：通过建模人类如何共同推断出解释一个代理人行动的一致性目标、信念和计划集合，然后通过认识逻辑对有关代理人信念的陈述进行评估，我们的框架为信念提供了一个概念角色语义，解释了人类信念归因的分级性和组合性，以及它们与目标和计划的密切联系。我们通过研究人们在观察一个代理人解决问题时是如何归因目标和信念的来评估这一框架。

    arXiv:2402.10416v1 Announce Type: new  Abstract: Despite the fact that beliefs are mental states that cannot be directly observed, humans talk about each others' beliefs on a regular basis, often using rich compositional language to describe what others think and know. What explains this capacity to interpret the hidden epistemic content of other minds? In this paper, we take a step towards an answer by grounding the semantics of belief statements in a Bayesian theory-of-mind: By modeling how humans jointly infer coherent sets of goals, beliefs, and plans that explain an agent's actions, then evaluating statements about the agent's beliefs against these inferences via epistemic logic, our framework provides a conceptual role semantics for belief, explaining the gradedness and compositionality of human belief attributions, as well as their intimate connection with goals and plans. We evaluate this framework by studying how humans attribute goals and beliefs while watching an agent solve
    
[^78]: 通过专家加权来衡量和减少LLM在没有黄金标准答案的情况下的虚构

    Measuring and Reducing LLM Hallucination without Gold-Standard Answers via Expertise-Weighting

    [https://arxiv.org/abs/2402.10412](https://arxiv.org/abs/2402.10412)

    提出了一种名为FEWL的幻觉度量方法，通过对LLM答案进行加权评估事实性，适用于没有黄金标准答案的情况。

    

    LLM幻觉，即生成事实不正确但看似令人信服的答案，目前是LLM可信度和可靠性的主要威胁。解决这一复杂问题的第一步是对其进行衡量。然而，现有的幻觉度量标准需要具有具有黄金标准答案的基准数据集，即人类编写的“最佳”或“正确”答案。这种要求使幻觉测量成本高昂，并容易出现人为误差。在这项工作中，我们提出了通过加权LLM对事实性进行评估（FEWL），这是第一个专门为金标准答案缺失时设计的幻觉度量标准。FEWL利用了现成的LLM答案作为黄金标准答案的代理。关键挑战是如何有效地量化参考LLM的专业知识。我们展示FEWL具有一定的理论保证，并在实证中证明它更准确。度量虚构。

    arXiv:2402.10412v1 Announce Type: cross  Abstract: LLM hallucination, i.e. generating factually incorrect yet seemingly convincing answers, is currently a major threat to the trustworthiness and reliability of LLMs. The first step towards solving this complicated problem is to measure it. However, existing hallucination metrics require to have a benchmark dataset with gold-standard answers, i.e. "best" or "correct" answers written by humans. Such requirement makes hallucination measurement costly and prone to human errors. In this work, we propose Factualness Evaluations via Weighting LLMs (FEWL), the first hallucination metric that is specifically designed for the scenario when gold-standard answers are absent. FEWL leverages the answers from off-the-shelf LLMs that serve as a proxy of gold-standard answers. The key challenge is how to quantify the expertise of reference LLMs resourcefully. We show FEWL has certain theoretical guarantees and demonstrate empirically it gives more accur
    
[^79]: 通过图表示学习理解大型语言模型调查论文分类法

    Understanding Survey Paper Taxonomy about Large Language Models via Graph Representation Learning

    [https://arxiv.org/abs/2402.10409](https://arxiv.org/abs/2402.10409)

    通过图结构信息在共类别图上利用图表示学习技术，可以在LLMs的预训练模型微调和零-shot/few-shot分类方面显著优于语言模型，揭示了弱标签微调LLMs的潜力。

    

    随着大型语言模型（LLMs）的新研究持续进行，难以跟上新的研究和模型。为帮助研究人员综合新研究成果，许多人写了调研论文，但即使这些论文也变得越来越多。本文提出了一种自动将调研论文分配到分类法的方法。我们收集了144篇LLM调研论文的元数据，并探讨了三种范例来对分类法内的论文进行分类。我们的工作表明，在共类别图上利用图结构信息可以显著优于两个范例中的语言模型; 使用LLMs进行预训练语言模型的微调和零-shot/few-shot分类。我们发现我们的模型超过了平均人类识别水平，并且利用较小模型生成的弱标签来微调LLMs（本研究中的GCN等）可能比使用地面实况标签更有效，揭示了从弱到强的潜力。

    arXiv:2402.10409v1 Announce Type: cross  Abstract: As new research on Large Language Models (LLMs) continues, it is difficult to keep up with new research and models. To help researchers synthesize the new research many have written survey papers, but even those have become numerous. In this paper, we develop a method to automatically assign survey papers to a taxonomy. We collect the metadata of 144 LLM survey papers and explore three paradigms to classify papers within the taxonomy. Our work indicates that leveraging graph structure information on co-category graphs can significantly outperform the language models in two paradigms; pre-trained language models' fine-tuning and zero-shot/few-shot classifications using LLMs. We find that our model surpasses an average human recognition level and that fine-tuning LLMs using weak labels generated by a smaller model, such as the GCN in this study, can be more effective than using ground-truth labels, revealing the potential of weak-to-stro
    
[^80]: 逻辑链：基于大型语言模型的基于规则的推理

    Chain of Logic: Rule-Based Reasoning with Large Language Models

    [https://arxiv.org/abs/2402.10400](https://arxiv.org/abs/2402.10400)

    介绍了一种新的提示方法，逻辑链，通过分解和重新组合来促进基于规则的推理，受到律师使用的序贯推理方法的启发。

    

    基于规则的推理是一种基本的法律推理类型，它使我们能够通过准确地将规则应用于一组事实来得出结论。我们探讨了因果语言模型作为基于规则的推理者，特别是关于组合规则 - 由多个元素组成形成复杂逻辑表达式的规则。推理组合规则具有挑战性，因为它需要多个推理步骤，并且需要关注元素之间的逻辑关系。我们引入了一种新的提示方法，逻辑链，通过分解（将元素作为独立的逻辑线索解决）和重新组合（重新组合这些子答案以解决潜在的逻辑表达式）。这种方法受到了IRAC（问题、规则、应用、结论）框架的启发，这是律师使用的一种序贯推理方法。我们在八个基于规则的推理任务中评估了逻辑链。

    arXiv:2402.10400v1 Announce Type: new  Abstract: Rule-based reasoning, a fundamental type of legal reasoning, enables us to draw conclusions by accurately applying a rule to a set of facts. We explore causal language models as rule-based reasoners, specifically with respect to compositional rules - rules consisting of multiple elements which form a complex logical expression. Reasoning about compositional rules is challenging because it requires multiple reasoning steps, and attending to the logical relationships between elements. We introduce a new prompting method, Chain of Logic, which elicits rule-based reasoning through decomposition (solving elements as independent threads of logic), and recomposition (recombining these sub-answers to resolve the underlying logical expression). This method was inspired by the IRAC (Issue, Rule, Application, Conclusion) framework, a sequential reasoning approach used by lawyers. We evaluate chain of logic across eight rule-based reasoning tasks in
    
[^81]: DataDreamer: 一种用于合成数据生成和可复现LLM工作流程的工具

    DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows

    [https://arxiv.org/abs/2402.10379](https://arxiv.org/abs/2402.10379)

    DataDreamer是一种用于合成数据生成和可复现LLM工作流程的开源Python库，有助于研究人员实现强大的LLM工作流，提倡开放科学和可重现性。

    

    大型语言模型（LLMs）已成为自然语言处理研究人员在各种任务中的主要工具。如今，许多研究人员在合成数据生成、任务评估、微调、提炼以及其他与模型相关的研究工作流中使用LLMs。然而，使用这些模型时会遇到挑战，这些挑战源于它们的规模、闭源性质以及缺乏针对这些新兴工作流的标准化工具。这些模型的迅速崛起和这些独特挑战对开放科学和使用它们的工作的可重现性产生了直接的负面影响。在本文中，我们介绍了DataDreamer，这是一个开源Python库，使研究人员能够编写简单的代码来实现强大的LLM工作流。DataDreamer还帮助研究人员遵循我们提出的最佳实践，以鼓励开放科学和可重现性。该库和文档可在h网站上找到。

    arXiv:2402.10379v1 Announce Type: new  Abstract: Large language models (LLMs) have become a dominant and important tool for NLP researchers in a wide range of tasks. Today, many researchers use LLMs in synthetic data generation, task evaluation, fine-tuning, distillation, and other model-in-the-loop research workflows. However, challenges arise when using these models that stem from their scale, their closed source nature, and the lack of standardized tooling for these new and emerging workflows. The rapid rise to prominence of these models and these unique challenges has had immediate adverse impacts on open science and on the reproducibility of work that uses them. In this paper, we introduce DataDreamer, an open source Python library that allows researchers to write simple code to implement powerful LLM workflows. DataDreamer also helps researchers adhere to best practices that we propose to encourage open science and reproducibility. The library and documentation are available at h
    
[^82]: BioMistral：面向医学领域的开源预训练大型语言模型集合

    BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains

    [https://arxiv.org/abs/2402.10373](https://arxiv.org/abs/2402.10373)

    BioMistral是一种面向生物医学领域的开源预训练大型语言模型集合，在医学问答任务中表现出优越性能并具有竞争优势。

    

    大型语言模型（LLMs）近年来展示出卓越的多功能性，为医疗保健和医学等专业领域提供潜在应用。尽管有各种针对健康领域定制的开源LLMs可用，但将通用LLMs调整到医学领域仍面临重大挑战。本文介绍了BioMistral，一种专为生物医学领域量身定制的开源LLM，采用Mistral作为基础模型，并在PubMed Central上进一步进行预训练。我们在包含10个已建立的英文医学问答（QA）任务的基准上对BioMistral进行了全面评估。我们还探讨通过量化和模型合并方法获得的轻量级模型。我们的结果表明，BioMistral相较于现有开源医学模型具有优越性能，并与专有对手具有竞争优势。最后，为了解决

    arXiv:2402.10373v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated remarkable versatility in recent years, offering potential applications across specialized domains such as healthcare and medicine. Despite the availability of various open-source LLMs tailored for health contexts, adapting general-purpose LLMs to the medical domain presents significant challenges. In this paper, we introduce BioMistral, an open-source LLM tailored for the biomedical domain, utilizing Mistral as its foundation model and further pre-trained on PubMed Central. We conduct a comprehensive evaluation of BioMistral on a benchmark comprising 10 established medical question-answering (QA) tasks in English. We also explore lightweight models obtained through quantization and model merging approaches. Our results demonstrate BioMistral's superior performance compared to existing open-source medical models and its competitive edge against proprietary counterparts. Finally, to address
    
[^83]: 我们能否用软提示LLMs来进行图学习任务？

    Can we soft prompt LLMs for graph learning tasks?

    [https://arxiv.org/abs/2402.10359](https://arxiv.org/abs/2402.10359)

    引入了GraphPrompter框架，通过软提示将图信息与LLMs对齐，以进一步探究LLMs理解图信息的潜力。

    

    图在表示社交网络、生物数据和引用网络等现实世界应用中的复杂关系方面起着重要作用。最近，大型语言模型（LLMs）在各个领域取得了巨大成功，这使得将LLMs应用于图表格尤为诱人。然而，直接将LLMs应用于图表格形式存在独特挑战，因为图表格形式与文本形式之间存在差异和不匹配。因此，为了进一步探究LLMs理解图信息的潜力，我们引入了GraphPrompter，这是一个通过软提示来将图信息与LLMs对齐的新颖框架。具体而言，GraphPrompter包括两个主要组件：一个图神经网络用于编码复杂的图信息，以及一个能够有效处理文本信息的LLM。在不同基准数据集上进行了广泛实验，涵盖了节点分类和链接预测任务。

    arXiv:2402.10359v1 Announce Type: cross  Abstract: Graph plays an important role in representing complex relationships in real-world applications such as social networks, biological data and citation networks. In recent years, Large Language Models (LLMs) have achieved tremendous success in various domains, which makes applying LLMs to graphs particularly appealing. However, directly applying LLMs to graph modalities presents unique challenges due to the discrepancy and mismatch between the graph and text modalities. Hence, to further investigate LLMs' potential for comprehending graph information, we introduce GraphPrompter, a novel framework designed to align graph information with LLMs via soft prompts. Specifically, GraphPrompter consists of two main components: a graph neural network to encode complex graph information and an LLM that effectively processes textual information. Comprehensive experiments on various benchmark datasets under node classification and link prediction tas
    
[^84]: 提升基于提示的语言模型零/少样本学习的偏差校准策略

    Prompt-Based Bias Calibration for Better Zero/Few-Shot Learning of Language Models

    [https://arxiv.org/abs/2402.10353](https://arxiv.org/abs/2402.10353)

    本研究提出了一种空输入提示方法，用于校准预训练语言模型中的固有偏差，从而提升零/少样本学习的性能。

    

    提示学习容易受到预训练语言模型中固有偏差的影响，导致基于提示的零/少样本学习性能不佳。本文提出了一种空输入提示方法，用于校准预训练语言模型中编码的固有偏差。与以往主要致力于社会公平的固有偏差修正方法不同，我们的目标是在增强语言模型在下游零/少样本学习任务中的性能的同时，强调固有偏差校准的效率。具体来说，我们利用从GPT-4生成的一组自动选取的无意义输入来提示预训练语言模型以探测固有偏差。利用偏差反映的概率分布，我们提出了一个分布差异损失用于偏差校准，其中我们仅更新语言模型的偏差参数（总参数的0.1%）以朝向相等的概率分布。

    arXiv:2402.10353v1 Announce Type: new  Abstract: Prompt learning is susceptible to intrinsic bias present in pre-trained language models (LMs), resulting in sub-optimal performance of prompt-based zero/few-shot learning. In this work, we propose a null-input prompting method to calibrate intrinsic bias encoded in pre-trained LMs. Different from prior efforts that address intrinsic bias primarily for social fairness and often involve excessive computational cost, our objective is to explore enhancing LMs' performance in downstream zero/few-shot learning while emphasizing the efficiency of intrinsic bias calibration. Specifically, we leverage a diverse set of auto-selected null-meaning inputs generated from GPT-4 to prompt pre-trained LMs for intrinsic bias probing. Utilizing the bias-reflected probability distribution, we formulate a distribution disparity loss for bias calibration, where we exclusively update bias parameters ($0.1\%$ of total parameters) of LMs towards equal probabilit
    
[^85]: 名词短语中头部的最佳位置。指示语、数词、形容词和名词的案例。

    The optimal placement of the head in the noun phrase. The case of demonstrative, numeral, adjective and noun

    [https://arxiv.org/abs/2402.10311](https://arxiv.org/abs/2402.10311)

    本研究旨在探讨句法依赖距离最小化与意外减少最小化原则在名词短语中的冲突，结论显示当涉及的单词较少且单词较短时，意外减少可能会超越句法依赖距离优化。

    

    一句话的词序由多种原则塑造。句法依赖距离最小化原则与意外减少最小化原则（或可预测性最大化）在单一头部的句法依赖结构中存在冲突：前者预测头部应该放置在线性排列的中心，后者预测头部应该放置在两端之一（要么在首位，要么在末位）。一个关键问题是何时意外减少（或可预测性最大化）应该超越句法依赖距离最小化。在单一头部结构的背景下，预测在满足两个条件时更有可能发生，即（a）涉及的单词较少，并且（b）单词较短。在这里，我们在由指示语、数词、形容词和名词组成的名词短语上测试了这一预测。我们发现，在首选顺序中...（缺失部分无法提供完整翻译）

    arXiv:2402.10311v1 Announce Type: new  Abstract: The word order of a sentence is shaped by multiple principles. The principle of syntactic dependency distance minimization is in conflict with the principle of surprisal minimization (or predictability maximization) in single head syntactic dependency structures: while the former predicts that the head should be placed at the center of the linear arrangement, the latter predicts that the head should be placed at one of the ends (either first or last). A critical question is when surprisal minimization (or predictability maximization) should surpass syntactic dependency distance minimization. In the context of single head structures, it has been predicted that this is more likely to happen when two conditions are met, i.e. (a) fewer words are involved and (b) words are shorter. Here we test the prediction on the noun phrase when its composed of a demonstrative, a numeral, an adjective and a noun. We find that, across preferred orders in l
    
[^86]: 如何识别重要紧急新闻？

    How to Discern Important Urgent News?

    [https://arxiv.org/abs/2402.10302](https://arxiv.org/abs/2402.10302)

    通过分析新闻数据集中的聚类属性，可以强相关地识别出新闻的重要性和紧急性，为识别重要紧急新闻或过滤不重要文章提供了新方法。

    

    我们发现在新闻的聚类数据集中，一种简单的属性与由LLM评估的新闻的重要性和紧急性（IUN）强相关。我们验证了我们的发现在不同的新闻数据集、数据集大小、聚类算法和嵌入上的普遍性。我们发现的相关性应该允许使用聚类（作为LLM的替代）来识别最重要的紧急新闻，或者用于过滤不重要的文章。

    arXiv:2402.10302v1 Announce Type: new  Abstract: We found that a simple property of clusters in a clustered dataset of news correlate strongly with importance and urgency of news (IUN) as assessed by LLM. We verified our finding across different news datasets, dataset sizes, clustering algorithms and embeddings. The found correlation should allow using clustering (as an alternative to LLM) for identifying the most important urgent news, or for filtering out unimportant articles.
    
[^87]: LAVE：以LLM为动力的视频编辑代理辅助和语言增强技术

    LAVE: LLM-Powered Agent Assistance and Language Augmentation for Video Editing

    [https://arxiv.org/abs/2402.10294](https://arxiv.org/abs/2402.10294)

    LAVE通过整合大型语言模型（LLMs），提供LLM动力的代理辅助和语言增强编辑功能，减少视频编辑的障碍，帮助用户实现编辑目标

    

    视频制作变得越来越受欢迎，但编辑所需的专业知识和努力常常对初学者构成障碍。本文探讨了在视频编辑工作流程中整合大型语言模型（LLMs）以减少这些障碍。我们的设计理念体现在LAVE中，这是一个提供LLM动力的代理辅助和语言增强编辑功能的新颖系统。LAVE自动生成用户素材的语言描述，作为使LLM能够处理视频并协助编辑任务的基础。当用户提供编辑目标时，代理计划并执行相关动作以实现这些目标。此外，LAVE允许用户通过代理或直接UI操作编辑视频，提供灵活性并使代理动作能够进行手动调整。我们的用户研究包括了从初学者到熟练编辑者的八名参与者，证明了LAVE对于减少编辑障碍和帮助用户实现编辑目标的有效性。

    arXiv:2402.10294v1 Announce Type: cross  Abstract: Video creation has become increasingly popular, yet the expertise and effort required for editing often pose barriers to beginners. In this paper, we explore the integration of large language models (LLMs) into the video editing workflow to reduce these barriers. Our design vision is embodied in LAVE, a novel system that provides LLM-powered agent assistance and language-augmented editing features. LAVE automatically generates language descriptions for the user's footage, serving as the foundation for enabling the LLM to process videos and assist in editing tasks. When the user provides editing objectives, the agent plans and executes relevant actions to fulfill them. Moreover, LAVE allows users to edit videos through either the agent or direct UI manipulation, providing flexibility and enabling manual refinement of agent actions. Our user study, which included eight participants ranging from novices to proficient editors, demonstrated
    
[^88]: 一种用于空破解的强REJECT方法

    A StrongREJECT for Empty Jailbreaks

    [https://arxiv.org/abs/2402.10260](https://arxiv.org/abs/2402.10260)

    提出了一种新的基准 StrongREJECT，通过使用更高质量的问题，更好地区分有效和无效的空破解方法。

    

    大型语言模型（LLMs）的兴起引起了对“破解”的关注，这种破解允许模型被恶意使用。然而，目前没有标准的基准来衡量破解的严重程度，导致破解论文的作者不得不自行创建标准。我们表明这些基准经常包含模棱两可或无法回答的问题，并使用倾向于高估低质量模型响应的滥用潜力的评分标准。一些破解技术使问题更加严重，因为它们即使对于良性问题也会降低模型响应的质量：我们展示了几种破解技术显着降低了GPT-4在MMLU上的零射击表现。破解还会使从“未经审查”的开源模型中获取有害响应变得更加困难。我们提出了一个新的基准，StrongREJECT，通过使用更高质量的问题更好地区分有效和无效的破解方法。

    arXiv:2402.10260v1 Announce Type: cross  Abstract: The rise of large language models (LLMs) has drawn attention to the existence of "jailbreaks" that allow the models to be used maliciously. However, there is no standard benchmark for measuring the severity of a jailbreak, leaving authors of jailbreak papers to create their own. We show that these benchmarks often include vague or unanswerable questions and use grading criteria that are biased towards overestimating the misuse potential of low-quality model responses. Some jailbreak techniques make the problem worse by decreasing the quality of model responses even on benign questions: we show that several jailbreaking techniques substantially reduce the zero-shot performance of GPT-4 on MMLU. Jailbreaks can also make it harder to elicit harmful responses from an "uncensored" open-source model. We present a new benchmark, StrongREJECT, which better discriminates between effective and ineffective jailbreaks by using a higher-quality que
    
[^89]: TOAD: 具有多样响应风格的面向任务的自动对话系统

    TOAD: Task-Oriented Automatic Dialogs with Diverse Response Styles

    [https://arxiv.org/abs/2402.10137](https://arxiv.org/abs/2402.10137)

    TOAD是一个具有多样响应风格的面向任务的自动对话系统，其中考虑了冗长程度和用户表达镜像两个方面。TOAD通过模拟真实的应用上下文交互，提供了丰富的系统响应风格选项，并在评估中表明建模更冗长的回复或不进行用户表达镜像的回复更具挑战性。

    

    最近大语言模型（LLM）的进展显示，下一代虚拟助手的期望包括在各种使用场景下提供更加自然和适应性强的对话能力。然而，为面向任务的对话（TOD）创建高质量的标注数据被认为是缓慢和昂贵的。为了解决这些挑战，我们引入了任务导向的自动对话系统（TOAD），这是一个新颖且可扩展的TOD数据集，以及与之配套的自动生成流程。TOAD数据集模拟了真实的应用上下文交互，并提供了各种系统响应风格选项。考虑了系统响应风格的两个方面，即冗长程度和用户表达镜像。我们在两个响应生成任务上进行了TOAD的评估，结果显示建模更冗长的回复或不进行用户表达镜像的回复更具挑战性。

    arXiv:2402.10137v1 Announce Type: new  Abstract: In light of recent advances in large language models~(LLMs), the expectations for the next generation of virtual assistants include enhanced naturalness and adaptability across diverse usage scenarios. However, the creation of high-quality annotated data for Task-Oriented Dialog~(TOD) is recognized to be slow and costly. To address these challenges, we introduce Task-Oriented Automatic Dialogs~(TOAD), a novel and scalable TOD dataset along with its automatic generation pipeline. The TOAD dataset simulates realistic app context interaction and provide a variety of system response style options. Two aspects of system response styles are considered, verbosity level and users' expression mirroring. We benchmark TOAD on two response generation tasks and the results show that modeling more verbose or responses without user expression mirroring is more challenging.
    
[^90]: 对于世界语的语言频率和错误修正的分析

    An Analysis of Langauge Frequency and Error Correction for Esperanto

    [https://arxiv.org/abs/2402.09696](https://arxiv.org/abs/2402.09696)

    本论文运用 Eo-GP 数据集进行了世界语的频率分析，引入了 Eo-GEC 数据集用于错误识别。实验表明 GPT-4 在自动化和人工评估中的性能优于 GPT-3.5，展示了先进语言模型在增强对于较少研究语言的 GEC 策略方面的潜力。

    

    目前的语法错误修正 (GEC) 项目往往着重于主要语言，而对于像世界语这样的资源匮乏语言则关注较少。本文通过首先使用专门为此目的创建的 Eo-GP 数据集进行全面的频率分析，开始弥补这一差距。然后，我们引入了源自真实用户案例并用于错误识别的细粒度语言细节进行注释的 Eo-GEC 数据集。利用 GPT-3.5 和 GPT-4，我们的实验证明 GPT-4 在自动化和人工评估中的表现优于 GPT-3.5，突出了其在解决世界语语法特殊性方面的效果，并展示了先进语言模型在增强对于较少研究语言的 GEC 策略方面的潜力。

    arXiv:2402.09696v1 Announce Type: new  Abstract: Current Grammar Error Correction (GEC) initiatives tend to focus on major languages, with less attention given to low-resource languages like Esperanto. In this article, we begin to bridge this gap by first conducting a comprehensive frequency analysis using the Eo-GP dataset, created explicitly for this purpose. We then introduce the Eo-GEC dataset, derived from authentic user cases and annotated with fine-grained linguistic details for error identification. Leveraging GPT-3.5 and GPT-4, our experiments show that GPT-4 outperforms GPT-3.5 in both automated and human evaluations, highlighting its efficacy in addressing Esperanto's grammatical peculiarities and illustrating the potential of advanced language models to enhance GEC strategies for less commonly studied languages.
    
[^91]: CodeMind:一个用于挑战大型语言模型进行代码推理的框架

    CodeMind: A Framework to Challenge Large Language Models for Code Reasoning

    [https://arxiv.org/abs/2402.09664](https://arxiv.org/abs/2402.09664)

    CodeMind是一个用于挑战大型语言模型进行代码推理的框架，通过评估LLMs的代码推理能力来替代仅仅依靠测试通过来评估，对三种代码推理任务进行评估，结果显示LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。

    

    仅靠测试通过来评估大型语言模型（LLMs）的代码合成能力可能会导致不公正的评估或促进具有数据泄漏的模型，作为一种替代方案，我们介绍了CodeMind，这是一个旨在评估LLMs的代码推理能力的框架。CodeMind目前支持三种代码推理任务：独立执行推理（IER）、依赖执行推理（DER）和规范推理（SR）。前两者评估模型以预测任意代码的执行输出，或者模型能够正确合成的代码。第三个任务评估LLMs实现指定预期行为的程度。我们使用CodeMind对两种不同编程语言中的五个基准下的九个LLMs进行了广泛的评估，结果表明LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。

    arXiv:2402.09664v1 Announce Type: cross  Abstract: Solely relying on test passing to evaluate Large Language Models (LLMs) for code synthesis may result in unfair assessment or promoting models with data leakage. As an alternative, we introduce CodeMind, a framework designed to gauge the code reasoning abilities of LLMs. CodeMind currently supports three code reasoning tasks: Independent Execution Reasoning (IER), Dependent Execution Reasoning (DER), and Specification Reasoning (SR). The first two evaluate models to predict the execution output of an arbitrary code or code the model could correctly synthesize. The third one evaluates the extent to which LLMs implement the specified expected behavior. Our extensive evaluation of nine LLMs across five benchmarks in two different programming languages using CodeMind shows that LLMs fairly understand control flow constructs and, in general, are capable of reasoning how inputs evolve to output, specifically for simple programs and the ones 
    
[^92]: MiMiC：表示空间中最小修改的对抗事实

    MiMiC: Minimally Modified Counterfactuals in the Representation Space

    [https://arxiv.org/abs/2402.09631](https://arxiv.org/abs/2402.09631)

    提出了一种新颖的对抗事实生成方法，利用闭式解决方案在表示空间中生成富有表达力的对抗事实，以减轻语言模型中的不良行为，该方法在地球移动问题方面提供理论上的保证，并对表示空间的几何组织进行改进。

    

    arXiv:2402.09631v1 公告类型：交叉学科 简介：语言模型经常表现出不良行为，如性别偏见或有毒语言。通过对表示空间进行干预，可以有效减轻这些问题，但两种常见的干预技术，即线性擦除和定向向量，并不能提供高度可控和表达丰富度。因此，我们提出了一种新颖的干预方法，旨在在表示空间中生成富有表达力的对抗事实，使源类别（例如“有毒”）的表示与目标类别（例如“非有毒”）的表示相似。这种方法利用高斯假设下的闭式解决方案，在地球移动问题方面提供了理论上的保证，并对表示空间的几何组织提供了进一步的改进。

    arXiv:2402.09631v1 Announce Type: cross  Abstract: Language models often exhibit undesirable behaviors, such as gender bias or toxic language. Interventions in the representation space were shown effective in mitigating such issues by altering the LM behavior. We first show that two prominent intervention techniques, Linear Erasure and Steering Vectors, do not enable a high degree of control and are limited in expressivity.   We then propose a novel intervention methodology for generating expressive counterfactuals in the representation space, aiming to make representations of a source class (e.g., ``toxic'') resemble those of a target class (e.g., ``non-toxic''). This approach, generalizing previous linear intervention techniques, utilizes a closed-form solution for the Earth Mover's problem under Gaussian assumptions and provides theoretical guarantees on the representation space's geometric organization. We further build on this technique and derive a nonlinear intervention that ena
    
[^93]: API Pack：一个用于API调用生成的大规模多语言数据集

    API Pack: A Massive Multilingual Dataset for API Call Generation

    [https://arxiv.org/abs/2402.09615](https://arxiv.org/abs/2402.09615)

    这个论文介绍了一个名为API Pack的大规模多语言数据集，旨在提高大型语言模型的API调用生成能力，通过实验证明了其在生成未见过的API调用方面的高准确率，并实现了跨语言的API调用生成

    

    我们介绍了API Pack，一个包含超过一百万个指令-API调用对的多语言数据集，旨在提高大型语言模型的API调用生成能力。通过实验，我们证明了API Pack在提升模型在这一特定任务上的效果的同时，保持其在一般编码方面的整体熟练程度。仅在20,000个Python实例上对CodeLlama-13B进行微调，其生成未见过的API调用的准确率比GPT-3.5和GPT-4分别高出10%和5%。扩展到100k个例子可以提高对训练期间未见过的新API的泛化能力。此外，实现了跨语言的API调用生成，而无需大量语言特定的数据。数据集、经过微调的模型和整体代码库可在https://github.com/anonymous_url上公开获取。

    arXiv:2402.09615v1 Announce Type: cross  Abstract: We introduce API Pack, a multilingual dataset featuring over one million instruction-API call pairs aimed at advancing large language models' API call generation capabilities. Through experiments, we demonstrate API Pack's efficacy in enhancing models for this specialized task while maintaining their overall proficiency at general coding. Fine-tuning CodeLlama-13B on just 20,000 Python instances yields over 10% and 5% higher accuracy than GPT-3.5 and GPT-4 respectively in generating unseen API calls. Scaling to 100k examples improves generalization to new APIs not seen during training. In addition, cross-lingual API call generation is achieved without needing extensive data per language. The dataset, fine-tuned models, and overall code base are publicly available at https://github.com/anonymous_url.
    
[^94]: 重新思考大型语言模型的机器消除技术

    Rethinking Machine Unlearning for Large Language Models

    [https://arxiv.org/abs/2402.08787](https://arxiv.org/abs/2402.08787)

    这篇论文研究了大型语言模型中的机器消除技术，旨在消除不良数据的影响并保持基本知识生成的完整性，为开发安全、可靠和资源高效的生成式人工智能提供基础。

    

    我们研究了大型语言模型（LLM）领域的机器消除技术（MU），称为LLM消除技术。这个研究旨在消除不良数据的影响（例如敏感或非法信息）以及相关模型的能力，同时保持基本的知识生成的完整性，并不影响因果无关的信息。我们设想LLM消除技术将成为LLM生命周期管理中的关键要素，可能成为开发既安全、可靠又资源高效的生成式人工智能的基础，而无需进行完全重训练。我们从概念、方法、评估指标和应用等方面探索了LLM消除技术的研究领域。特别是，我们突出了现有LLM消除技术研究中经常被忽视的方面，例如消除范围、数据模型交互和多方面的有效性评估。

    arXiv:2402.08787v1 Announce Type: cross Abstract: We explore machine unlearning (MU) in the domain of large language models (LLMs), referred to as LLM unlearning. This initiative aims to eliminate undesirable data influence (e.g., sensitive or illegal information) and the associated model capabilities, while maintaining the integrity of essential knowledge generation and not affecting causally unrelated information. We envision LLM unlearning becoming a pivotal element in the life-cycle management of LLMs, potentially standing as an essential foundation for developing generative AI that is not only safe, secure, and trustworthy, but also resource-efficient without the need of full retraining. We navigate the unlearning landscape in LLMs from conceptual formulation, methodologies, metrics, and applications. In particular, we highlight the often-overlooked aspects of existing LLM unlearning research, e.g., unlearning scope, data-model interaction, and multifaceted efficacy assessment. We
    
[^95]: 可信的取样合理化通过半监督的蕴涵信号

    Plausible Extractive Rationalization through Semi-Supervised Entailment Signal

    [https://arxiv.org/abs/2402.08479](https://arxiv.org/abs/2402.08479)

    本文通过半监督方法，采用蕴涵对齐，以优化可行性，提取有理的方式提供一个可解释的替代模型

    

    复杂和不透明的黑盒子模型的增加需要采用可解释的措施，其中一种选择是提取有理的模型，它们作为更可解释的替代方案。这些模型，也称为先解释然后预测模型，使用解释模型来提取有理，然后使用提取的信息来调整预测模型。它们的主要目标是提供精确和忠实的解释，由提取的有理表示。在本文中，我们采用半监督方法来优化提取有理的可行性。我们采用一个预训练的自然语言推理（NLI）模型，并在一个小型的有监督有理集（10%）上进一步微调它。通过蕴涵对齐，NLI预测模型被利用作为解释模型的一种监督信号源。通过在问答任务中强制解释和答案之间的对齐一致，我们证明了性能得到了提升。

    The increasing use of complex and opaque black box models requires the adoption of interpretable measures, one such option is extractive rationalizing models, which serve as a more interpretable alternative. These models, also known as Explain-Then-Predict models, employ an explainer model to extract rationales and subsequently condition the predictor with the extracted information. Their primary objective is to provide precise and faithful explanations, represented by the extracted rationales. In this paper, we take a semi-supervised approach to optimize for the plausibility of extracted rationales. We adopt a pre-trained natural language inference (NLI) model and further fine-tune it on a small set of supervised rationales ($10\%$). The NLI predictor is leveraged as a source of supervisory signals to the explainer via entailment alignment. We show that, by enforcing the alignment agreement between the explanation and answer in a question-answering task, the performance can be improve
    
[^96]: 朝着忠实和强大的基于证据的问答专家的方向前进

    Towards Faithful and Robust LLM Specialists for Evidence-Based Question-Answering

    [https://arxiv.org/abs/2402.08277](https://arxiv.org/abs/2402.08277)

    这项工作探索了如何鲁棒地微调大型语言模型以提高答案的来源质量和答案归因能力，引入了数据生成流水线和四个测试集来评估模型的性能，并展示了在合成数据上微调可以改善内部和外部分布的性能。

    

    对大型语言模型（LLM）更忠实和可追踪的答案的进步对于各种研究和实践活动至关重要。其中一种达到这个目标的方法是基于可靠的来源提供答案。然而，这种基于证据的问答在使用LLM时已经证明在引用正确的来源（来源质量）和准确地表示来源中的信息（答案归因能力）方面工作不足。在这项工作中，我们系统地研究了如何鲁棒地微调LLM，以提高来源质量和答案归因能力。具体而言，我们引入了一个数据生成流水线，其中包括自动数据质量过滤器，可以大规模合成多样化的高质量训练和测试数据。我们还引入了四个测试集，以对微调后的专家模型的鲁棒性进行基准测试。广泛的评估结果表明，在合成数据上进行微调可以提高在内部和外部分布的性能。%基于证据的问答案例。此外，我们展示了用于评估的四个测试集，以评估微调后的专家模型的鲁棒性。

    Advances towards more faithful and traceable answers of Large Language Models (LLMs) are crucial for various research and practical endeavors. One avenue in reaching this goal is basing the answers on reliable sources. However, this Evidence-Based QA has proven to work insufficiently with LLMs in terms of citing the correct sources (source quality) and truthfully representing the information within sources (answer attributability). In this work, we systematically investigate how to robustly fine-tune LLMs for better source quality and answer attributability. Specifically, we introduce a data generation pipeline with automated data quality filters, which can synthesize diversified high-quality training and testing data at scale. We further introduce four test sets to benchmark the robustness of fine-tuned specialist models. Extensive evaluation shows that fine-tuning on synthetic data improves performance on both in- and out-of-distribution. %Evidence-Based QA cases. Furthermore, we sho
    
[^97]: 基于锚点的大型语言模型

    Anchor-based Large Language Models

    [https://arxiv.org/abs/2402.07616](https://arxiv.org/abs/2402.07616)

    基于锚点的大型语言模型（AnLLM）通过引入创新的基于锚点的自注意力网络（AnSAN）和基于锚点的推理策略，将序列信息压缩到锚点标记中，减少键/值缓存，提高推理效率。

    

    大型语言模型（LLMs）主要采用仅解码器的转换器架构，需要保留历史标记的键/值信息以提供上下文信息并避免冗余计算。然而，这些LLMs的巨大大小和参数量需要大量的GPU内存。这种内存需求随着输入文本的长度而增加，迫切需要更高效的信息存储和处理方法。本研究介绍了一种基于锚点的LLM（AnLLM），它利用了一种创新的基于锚点的自注意力网络（AnSAN）和基于锚点的推理策略。这种方法使LLMs能够将序列信息压缩成锚点标记，减少键/值缓存并提高推理效率。实验证明，AnLLM在减少键/值缓存高达99%和推理速度提高高达3.5倍的同时，仍保持可比的准确性。尽管牺牲了一些准确性，AnLLM的创新和贡献依然重要。

    Large language models (LLMs) predominantly employ decoder-only transformer architectures, necessitating the retention of keys/values information for historical tokens to provide contextual information and avoid redundant computation. However, the substantial size and parameter volume of these LLMs require massive GPU memory. This memory demand increases with the length of the input text, leading to an urgent need for more efficient methods of information storage and processing. This study introduces the Anchor-based LLM (AnLLM), which utilizes an innovative anchor-based self-attention network (AnSAN) and also an anchor-based inference strategy. This approach enables LLMs to compress sequence information into an anchor token, reducing the keys/values cache and enhancing inference efficiency. Experiments show that the AnLLM maintains comparable accuracy with up to 99% keys/values cache reduction and up to 3.5 times faster inference. Despite a minor compromise in accuracy, the AnLLM signi
    
[^98]: 推动文本分类中LLM容量的极限

    Pushing The Limit of LLM Capacity for Text Classification

    [https://arxiv.org/abs/2402.07470](https://arxiv.org/abs/2402.07470)

    本论文提出了一个自适应增强框架RGPT，通过反复集成强基学习者，生成一个专用的文本分类LLM。通过实证比较，我们展示了RGPT明显胜过其他方法。

    

    由于大型语言模型（LLM）在众多下游NLP任务中展示出的非凡效果，文本分类未来研究的价值面临着挑战和不确定性。在这个任务边界逐渐模糊的开放式语言建模时代，一个迫切的问题出现了：在充分利用LLM的情况下，我们在文本分类方面取得了重大进展吗？为了回答这个问题，我们提出了RGPT，一个自适应增强框架，旨在通过反复集成一组强基学习者，来生成一个专用的文本分类LLM。基学习者是通过自适应调整训练样本的分布，并反复微调LLM与之构建的。然后，这些基学习者通过反复融合前几个学习者的历史预测结果，形成一个专用的文本分类LLM。通过全面的实证比较，我们展示了RGPT明显胜过其他方法。

    The value of text classification's future research has encountered challenges and uncertainties, due to the extraordinary efficacy demonstrated by large language models (LLMs) across numerous downstream NLP tasks. In this era of open-ended language modeling, where task boundaries are gradually fading, an urgent question emerges: have we made significant advances in text classification under the full benefit of LLMs? To answer this question, we propose RGPT, an adaptive boosting framework tailored to produce a specialized text classification LLM by recurrently ensembling a pool of strong base learners. The base learners are constructed by adaptively adjusting the distribution of training samples and iteratively fine-tuning LLMs with them. Such base learners are then ensembled to be a specialized text classification LLM, by recurrently incorporating the historical predictions from the previous learners. Through a comprehensive empirical comparison, we show that RGPT significantly outperf
    
[^99]: 透过分割投票的视角: 探索在法律案件结果分类中的意见分歧、困难和校准

    Through the Lens of Split Vote: Exploring Disagreement, Difficulty and Calibration in Legal Case Outcome Classification

    [https://arxiv.org/abs/2402.07214](https://arxiv.org/abs/2402.07214)

    通过研究分割投票，探索律师在处理法律案件结果分类时面临的意见分歧和困难，并在欧洲人权法院收集了法官的投票数据集进行研究。这项研究还评估了模型和人类之间感知困难的一致性以及模型的置信度和人类校准。

    

    在法律决策中，当法官无法达成一致决定时，就会出现分割投票(SV)，给必须处理各种法律论点和意见的律师带来了困难。在高风险领域，理解人类和AI系统之间感知困难的一致性对于建立信任至关重要。然而，现有的自然语言处理校准方法主要关注分类器对预测性能的认知，通常是与人类的多数类进行比较，而忽视了人类标签变化的固有差异（HLV）。本文将分割投票视为自然可观察的人类意见分歧和价值多元主义，并从欧洲人权法院（ECHR）收集法官的投票分布，提出了带有SV信息的案件结果分类（COC）数据集SV-ECHR。我们建立了包含SV特定子类别的不同意见的分类法。我们进一步评估模型和人类之间感知困难的一致性，以及COC模型的置信度和人类校准。我们观察到了限制性的...

    In legal decisions, split votes (SV) occur when judges cannot reach a unanimous decision, posing a difficulty for lawyers who must navigate diverse legal arguments and opinions. In high-stakes domains, understanding the alignment of perceived difficulty between humans and AI systems is crucial to build trust. However, existing NLP calibration methods focus on a classifier's awareness of predictive performance, measured against the human majority class, overlooking inherent human label variation (HLV). This paper explores split votes as naturally observable human disagreement and value pluralism. We collect judges' vote distributions from the European Court of Human Rights (ECHR), and present SV-ECHR, a case outcome classification (COC) dataset with SV information. We build a taxonomy of disagreement with SV-specific subcategories. We further assess the alignment of perceived difficulty between models and humans, as well as confidence- and human-calibration of COC models. We observe lim
    
[^100]: LLM能够识别毒性吗？结构化毒性调查框架和基于语义的度量

    Can LLMs Recognize Toxicity? Structured Toxicity Investigation Framework and Semantic-Based Metric

    [https://arxiv.org/abs/2402.06900](https://arxiv.org/abs/2402.06900)

    本研究提出了一种基于大型语言模型（LLMs）的自动度量方法，用于识别生成文本中的毒性。通过分析毒性因素和LLMs的内在毒性属性，该方法在测量毒性方面表现出众，比现有指标提升12个百分点。

    

    在开发遵守社会标准的大型语言模型（LLMs）的过程中，识别生成文本中的毒性存在至关重要。现有的大多数毒性度量依赖于在特定毒性数据集上训练的编码模型。然而，这些编码器容易受到分布外的问题的影响，并且依赖于数据集中所假定的毒性定义。本文介绍了一种基于LLMs的自动鲁棒度量，用于区分模型回应是否具有毒性。我们首先分析了毒性因素，然后研究了LLMs的内在毒性属性，以确定它们作为评估器的适用性。随后，我们对评估数据集上的度量指标LLMs As ToxiciTy Evaluators（LATTE）进行了评估。实证结果表明，在不进行训练过程的情况下，我们的度量在测量毒性方面表现出色，F1得分比现有技术指标提高了12个百分点。我们还展示了上游毒性对度量结果的影响。

    In the pursuit of developing Large Language Models (LLMs) that adhere to societal standards, it is imperative to discern the existence of toxicity in the generated text. The majority of existing toxicity metrics rely on encoder models trained on specific toxicity datasets. However, these encoders are susceptible to out-of-distribution (OOD) problems and depend on the definition of toxicity assumed in a dataset. In this paper, we introduce an automatic robust metric grounded on LLMs to distinguish whether model responses are toxic. We start by analyzing the toxicity factors, followed by examining the intrinsic toxic attributes of LLMs to ascertain their suitability as evaluators. Subsequently, we evaluate our metric, LLMs As ToxiciTy Evaluators (LATTE), on evaluation datasets.The empirical results indicate outstanding performance in measuring toxicity, improving upon state-of-the-art metrics by 12 points in F1 score without training procedure. We also show that upstream toxicity has an 
    
[^101]: 与更有说服力的LLMs辩论会导致更真实的回答

    Debating with More Persuasive LLMs Leads to More Truthful Answers

    [https://arxiv.org/abs/2402.06782](https://arxiv.org/abs/2402.06782)

    本文研究了更弱的语言模型是否能评估更强的模型的正确性。研究发现，通过进行辩论，非专家模型和人类回答问题的准确性都有所提高。

    

    与所需行为一致的大型语言模型（LLM）的常见方法主要依赖于人工标注的数据。然而，随着模型变得越来越复杂，它们将超过人类专业知识，人类评估的角色将演变为非专家监督专家。在此之前，我们问：更弱的模型能评估更强的模型的正确性吗？我们在类似的环境中调查了这个问题，其中更强的模型（专家）拥有回答问题所需的信息，而更弱的模型（非专家）缺乏这些信息。我们评估的方法是\textit{辩论}，其中两个LLM专家分别支持不同的答案，一个非专家选择答案。我们发现辩论 consistently帮助非专家模型和人类回答问题，分别达到76%和88%的准确性（朴素基准分别为48%和60%）。此外，以无监督方式优化专家辩论者的说服力会提高非专家的能力。

    Common methods for aligning large language models (LLMs) with desired behaviour heavily rely on human-labelled data. However, as models grow increasingly sophisticated, they will surpass human expertise, and the role of human evaluation will evolve into non-experts overseeing experts. In anticipation of this, we ask: can weaker models assess the correctness of stronger models? We investigate this question in an analogous setting, where stronger models (experts) possess the necessary information to answer questions and weaker models (non-experts) lack this information. The method we evaluate is \textit{debate}, where two LLM experts each argue for a different answer, and a non-expert selects the answer. We find that debate consistently helps both non-expert models and humans answer questions, achieving 76\% and 88\% accuracy respectively (naive baselines obtain 48\% and 60\%). Furthermore, optimising expert debaters for persuasiveness in an unsupervised manner improves non-expert abilit
    
[^102]: NICE: 优化上下文示例还是不优化？

    NICE: To Optimize In-Context Examples or Not?

    [https://arxiv.org/abs/2402.06733](https://arxiv.org/abs/2402.06733)

    通过研究在提供任务特定指令的情况下是否需要优化上下文示例，我们挑战了对于指导性LLMs的共识，并发现在某些任务中，不同的优化上下文示例方法会产生递减的回报。我们引入了"度量标准"，用于衡量从给定指令中学习任务的能力，并提供了一个启发式方法，帮助决定是否优化指令还是ICE用于任何新任务。

    

    最近的研究表明，大型语言模型（LLMs）通过上下文学习和优化上下文示例（ICE），在各种任务上表现出色。然而，大多数研究假设在提示信息中要么是固定的，要么没有提供指令，导致了一个表面上的共识：优化上下文示例对于提高性能至关重要。我们针对经过指导的LLMs挑战这一共识，研究在提供了任务特定指令的情况下优化上下文示例是否必要，并发现有一些任务对于不同的优化上下文示例方法产生递减的回报。我们引入了一种任务特定的度量标准，称为"度量标准"（Metric），用于量化从给定指令中学习任务的能力，并提供了一个启发式方法，帮助决定是否优化指令还是ICE用于任何新任务。通过对各种任务和逐步增加的指令集的系统性研究，我们验证了该启发式方法的有效性。

    Recent works have shown that large language models (LLMs) work remarkably well on a wide range of tasks through in-context learning and optimization of in-context examples (ICE). However, most of these studies assume either a fixed or no instruction provided in the prompt, leading to the apparent consensus that the optimization of in-context examples is critical for better performance. We challenge this consensus for instruction-tuned LLMs by investigating the necessity of optimizing in-context examples when task-specific instructions are provided, and find that there are tasks for which various ways of optimizing in-context examples yield diminishing returns. We introduce a task-specific metric called \metriclong{} (\metric) that quantifies the learnability of tasks from a given instruction, and provides a heuristic that helps decide whether to optimize for instructions or ICE for any new task. On a wide range of tasks and a systematically created instruction set with gradually added 
    
[^103]: 统一的多模态大型语言模型的幻觉检测

    Unified Hallucination Detection for Multimodal Large Language Models

    [https://arxiv.org/abs/2402.03190](https://arxiv.org/abs/2402.03190)

    该论文提出了一个新颖的统一的多模态幻觉检测框架UNIHD，并设计了一个评估基准方法MHaluBench来评估幻觉检测方法的进展。这项工作扩展了幻觉检测的研究范围并提供了有效的解决方案。

    

    尽管在多模态任务方面取得了重大进展，多模态大型语言模型(MLLMs)仍然存在幻觉的严重问题。因此，可靠地检测MLLMs中的幻觉已成为模型评估和实际应用部署保障的重要方面。之前在这个领域的研究受到了狭窄的任务焦点、不足的幻觉类别涵盖范围以及缺乏详细的细粒度的限制。针对这些挑战，我们的工作扩展了幻觉检测的研究范围。我们提出了一个新颖的元评估基准方法，MHaluBench，精心设计以促进幻觉检测方法的进展评估。此外，我们揭示了一个新颖的统一多模态幻觉检测框架，UNIHD，它利用一套辅助工具来稳健地验证幻觉的发生。我们通过实验证明了UNIHD的有效性。

    Despite significant strides in multimodal tasks, Multimodal Large Language Models (MLLMs) are plagued by the critical issue of hallucination. The reliable detection of such hallucinations in MLLMs has, therefore, become a vital aspect of model evaluation and the safeguarding of practical application deployment. Prior research in this domain has been constrained by a narrow focus on singular tasks, an inadequate range of hallucination categories addressed, and a lack of detailed granularity. In response to these challenges, our work expands the investigative horizons of hallucination detection. We present a novel meta-evaluation benchmark, MHaluBench, meticulously crafted to facilitate the evaluation of advancements in hallucination detection methods. Additionally, we unveil a novel unified multimodal hallucination detection framework, UNIHD, which leverages a suite of auxiliary tools to validate the occurrence of hallucinations robustly. We demonstrate the effectiveness of UNIHD throug
    
[^104]: 表情符号解密：利用ChatGPT提升社交媒体沟通的理解能力

    Emojis Decoded: Leveraging ChatGPT for Enhanced Understanding in Social Media Communications

    [https://arxiv.org/abs/2402.01681](https://arxiv.org/abs/2402.01681)

    在表情符号研究中，我们评估了ChatGPT在处理注释和下游任务中的有效性。我们的研究结果表明ChatGPT可以作为一个可行的替代人类注释者的工具，有效地解释表情符号。

    

    表情符号在社交网络沟通中已经普遍存在，它们承载了超越文字或短语的语义，这引发了学术界对其属性和功能的越来越多的研究兴趣。然而，与表情符号相关的研究和应用面临两个主要挑战。首先，研究者通常依赖众包来注释表情符号，以了解其情感、使用意图和语义含义。其次，用户的主观解释往往会导致对表情符号的误解，并造成沟通障碍。大型语言模型（LLMs）在各种注释任务中取得了显著的成功，ChatGPT在多个领域展示了专业能力。在我们的研究中，我们评估了ChatGPT在处理以前注释和下游任务中的有效性。我们的目标是验证ChatGPT可以在表情符号研究中作为人类注释者的可行替代者，并验证其解释表情符号的能力。

    Emojis, which encapsulate semantics beyond mere words or phrases, have become prevalent in social network communications. This has spurred increasing scholarly interest in exploring their attributes and functionalities. However, emoji-related research and application face two primary challenges. First, researchers typically rely on crowd-sourcing to annotate emojis in order to understand their sentiments, usage intentions, and semantic meanings. Second, subjective interpretations by users can often lead to misunderstandings of emojis and cause the communication barrier. Large Language Models (LLMs) have achieved significant success in various annotation tasks, with ChatGPT demonstrating expertise across multiple domains. In our study, we assess ChatGPT's effectiveness in handling previously annotated and downstream tasks. Our objective is to validate the hypothesis that ChatGPT can serve as a viable alternative to human annotators in emoji research and that its ability to explain emoji
    
[^105]: StickerConv: 从零开始生成多模态共情回应

    StickerConv: Generating Multimodal Empathetic Responses from Scratch

    [https://arxiv.org/abs/2402.01679](https://arxiv.org/abs/2402.01679)

    本文介绍了StickerConv代理(Agent4SC)，该代理通过协作代理交互，实现了与贴纸使用相仿的人类行为模拟，从而增强了多模态共情交流。为了利用构建的多模态共情对话数据集StickerConv，作者提出了PErceive and Generate Stickers (PEGS)模型，该模型能够生成情境相关和情感丰富的回应。

    

    在当前的共情对话研究中，贴纸尽管被广泛认可为提高在线交流中的共情能力，但仍未得到充分探索。本文介绍了StickerConv代理(Agent4SC)，通过协作代理交互，实现了与贴纸使用相仿的人类行为模拟，从而增强了多模态共情交流。在此基础上，我们构建了一个多模态共情对话数据集StickerConv，包括12.9K个对话会话，5.8K个独特贴纸和2K个多样化会话场景，专门设计用于增强多模态情境下的共情回应生成。为了利用这个数据集的丰富性，我们提出了PErceive and Generate Stickers (PEGS)，一种多模态共情回应生成模型，并结合基于LLM的全面共情评估指标。我们的实验表明，PEGS在生成情境相关和情感丰富的回应方面具有很好的效果。

    Stickers, while widely recognized for enhancing empathetic communication in online interactions, remain underexplored in current empathetic dialogue research. In this paper, we introduce the Agent for StickerConv (Agent4SC), which uses collaborative agent interactions to realistically simulate human behavior with sticker usage, thereby enhancing multimodal empathetic communication. Building on this foundation, we develop a multimodal empathetic dialogue dataset, StickerConv, which includes 12.9K dialogue sessions, 5.8K unique stickers, and 2K diverse conversational scenarios, specifically designs to augment the generation of empathetic responses in a multimodal context. To leverage the richness of this dataset, we propose PErceive and Generate Stickers (PEGS), a multimodal empathetic response generation model, complemented by a comprehensive set of empathy evaluation metrics based on LLM. Our experiments demonstrate PEGS's effectiveness in generating contextually relevant and emotional
    
[^106]: 因此我思，我在：大型语言模型中的意识

    I Think, Therefore I am: Awareness in Large Language Models

    [https://arxiv.org/abs/2401.17882](https://arxiv.org/abs/2401.17882)

    本文介绍了将意识概念引入大型语言模型（LLMs），并定义了LLMs在感知和理解自身以及展示社交智能方面的能力。通过引入AwareLLM数据集，研究发现LLMs在意识方面表现出相当程度的能力，尽管它们缺乏实质性的能力意识。

    

    大型语言模型（LLMs）是否展现出类似于人类的意识形式？在本文中，我们介绍了将意识概念引入LLMs，认为意识是LLMs增强与人类交互并确保道德回应的可信度的重要方面。我们将LLMs中的意识定义为感知和理解自身作为AI模型以及展示社交智能的能力。我们确定了意识的四个关键维度：能力、任务、情感和观点。为了评估LLMs在这些维度上的表现，我们引入了一个专门的数据集，AwareLLM数据集。我们的研究结果显示，LLMs展现出相当程度的意识，尽管它们仍然缺乏实质性的能力意识。

    Do large language models (LLMs) exhibit any forms of awareness similar to humans? In this paper, we introduce the concept of awareness to LLMs, arguing that awareness is an essential aspect of trustworthiness for LLMs to enhance their interaction with humans while ensuring ethical responses. We define awareness in LLMs as the ability to perceive and understand themselves as AI models and to exhibit social intelligence. We identify four key dimensions of awareness: capability, mission, emotion, and perspective. To assess LLMs on these dimensions, we introduce a specialized dataset, AwareLLM dataset. Our findings reveal that LLMs demonstrate a decent degree of awareness, though they still lack substantial capability awareness.
    
[^107]: 规划、创造、使用：对LLMs在实际复杂场景中全面工具利用的基准测试

    Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios

    [https://arxiv.org/abs/2401.17167](https://arxiv.org/abs/2401.17167)

    本论文介绍了一种新的基准测试UltraTool，旨在改善和评估LLMs在实际复杂场景中的工具利用能力。该基准测试关注从规划和创建到应用工具的整个过程，并强调实际复杂性和多步规划的要求。

    

    近期使用大型语言模型（LLMs）作为实际应用中的智能代理的趋势强调了对它们能力的全面评估的必要性，特别是在涉及规划、创造和使用工具的复杂场景中。然而，现有的基准测试通常只关注简单合成的查询，不反映实际复杂性，因此在评估工具利用方面的视角有限。为了解决这个问题，我们提出了一个新颖的基准测试UltraTool，旨在改善和评估LLMs在实际场景中工具利用的能力。UltraTool关注使用工具的整个过程——从规划和创建到在复杂任务中应用。它强调实际的复杂性，要求准确的多步规划以实现有效的问题解决。UltraTool的一个关键特点是在工具使用之前针对自然语言进行独立评估的规划，从而简化了任务解决的过程。

    The recent trend of using Large Language Models (LLMs) as intelligent agents in real-world applications underscores the necessity for comprehensive evaluations of their capabilities, particularly in complex scenarios involving planning, creating, and using tools. However, existing benchmarks typically focus on simple synthesized queries that do not reflect real-world complexity, thereby offering limited perspectives in evaluating tool utilization. To address this issue, we present UltraTool, a novel benchmark designed to improve and evaluate LLMs' ability in tool utilization within real-world scenarios. UltraTool focuses on the entire process of using tools - from planning and creating to applying them in complex tasks. It emphasizes real-world complexities, demanding accurate, multi-step planning for effective problem-solving. A key feature of UltraTool is its independent evaluation of planning with natural language, which happens before tool usage and simplifies the task solving by m
    
[^108]: 将分层变分自动编码器与LLMs结合，实现在社交媒体中具有临床意义的时间线摘要

    Combining Hierachical VAEs with LLMs for clinically meaningful timeline summarisation in social media

    [https://arxiv.org/abs/2401.16240](https://arxiv.org/abs/2401.16240)

    利用混合的分层变分自动编码器与LLMs结合的方法实现了从社交媒体用户时间轴生成具有临床意义的摘要，通过对时间轴的时间敏感性和举重有力的抽象摘要，TH-VAE生成的摘要在捕捉随时间变化方面优于仅使用LLM方法。

    

    我们引入了一种混合的抽象汇总方法，将分层变分自动编码器与LLMs结合（LlaMA-2），以从社交媒体用户时间轴生成具有临床意义的摘要，适用于心理健康监测。摘要结合了两种不同的叙述观点：通过向专门的临床提示馈送来生成专向临床医生有用的第三人称临床见解，以及重要的，通过新颖的分层变分自动编码器TH-VAE生成用户时间线的临时敏感的第一人称抽象摘要。我们通过与专家摘要的自动评估和与临床专家的人工评估来评估生成的摘要，结果表明通过TH-VAE进行的时间线摘要会产生更富有临床效用、更具事实和逻辑连贯性的摘要，优于仅使用LLM方法捕捉时间变化。

    arXiv:2401.16240v2 Announce Type: replace-cross  Abstract: We introduce a hybrid abstractive summarisation approach combining hierarchical VAE with LLMs (LlaMA-2) to produce clinically meaningful summaries from social media user timelines, appropriate for mental health monitoring. The summaries combine two different narrative points of view: clinical insights in third person useful for a clinician are generated by feeding into an LLM specialised clinical prompts, and importantly, a temporally sensitive abstractive summary of the user's timeline in first person, generated by a novel hierarchical variational autoencoder, TH-VAE. We assess the generated summaries via automatic evaluation against expert summaries and via human evaluation with clinical experts, showing that timeline summarisation by TH-VAE results in more factual and logically coherent summaries rich in clinical utility and superior to LLM-only approaches in capturing changes over time.
    
[^109]: 多语言语言模型的文本嵌入反向安全性

    Text Embedding Inversion Security for Multilingual Language Models

    [https://arxiv.org/abs/2401.12192](https://arxiv.org/abs/2401.12192)

    该研究探讨了多语言语言模型的文本嵌入逆转安全性问题，发现多语言模型更容易受到逆转攻击的影响，并提出了简单的掩蔽防御方法。

    

    在自然语言处理中，文本数据通常以实数嵌入表示，尤其是随着大型语言模型（LLMs）和嵌入式服务（EaaS）的流行。然而，将敏感信息存储为嵌入可能容易受到安全漏洞的影响，因为研究表明，即使不知道底层模型的情况下，文本也可以从嵌入中重构。尽管已经探讨了防御机制，但这些机制专注于英语，使其他语言容易受到攻击。本文通过多语言嵌入逆转探讨了LLM安全性。我们定义了黑盒多语言和跨语言逆转攻击的问题，并深入探讨了它们可能的影响。我们的研究结果表明，多语言LLMs可能更容易受到逆转攻击的影响，部分原因是基于英语的防御可能无效。为了缓解这一问题，我们提出了一种简单的掩蔽防御方法，对b有效。

    arXiv:2401.12192v2 Announce Type: replace-cross  Abstract: Textual data is often represented as realnumbered embeddings in NLP, particularly with the popularity of large language models (LLMs) and Embeddings as a Service (EaaS). However, storing sensitive information as embeddings can be vulnerable to security breaches, as research shows that text can be reconstructed from embeddings, even without knowledge of the underlying model. While defence mechanisms have been explored, these are exclusively focused on English, leaving other languages vulnerable to attacks. This work explores LLM security through multilingual embedding inversion. We define the problem of black-box multilingual and cross-lingual inversion attacks, and thoroughly explore their potential implications. Our findings suggest that multilingual LLMs may be more vulnerable to inversion attacks, in part because English based defences may be ineffective. To alleviate this, we propose a simple masking defense effective for b
    
[^110]: 辨识并分析大型语言模型中的任务编码标记

    Identifying and Analyzing Task-Encoding Tokens in Large Language Models

    [https://arxiv.org/abs/2401.11323](https://arxiv.org/abs/2401.11323)

    本文通过识别和分析任务编码标记，揭示了大型语言模型如何学习执行任务的方式。

    

    在上下文学习（ICL）已成为自然语言处理中少样本学习的有效解决方案。然而，我们对ICL的工作机制的理解有限，特别是模型如何从ICL演示中学习执行任务。本文通过识别和分析任务编码标记，调查了这个问题。我们发现，模板标记和停用词标记最容易成为任务编码标记。此外，我们实验证明，词汇意思、重复和文本格式是这些标记的主要区别特征。我们的工作揭示了大型语言模型（LLMs）学习的方式。

    arXiv:2401.11323v2 Announce Type: replace  Abstract: In-context learning (ICL) has become an effective solution for few-shot learning in natural language processing. However, our understanding of ICL's working mechanisms is limited, specifically regarding how models learn to perform tasks from ICL demonstrations. For example, unexpectedly large changes in performance can arise from small changes in the prompt, leaving prompt design a largely empirical endeavour. In this paper, we investigate this problem by identifying and analyzing task-encoding tokens on whose representations the task performance depends. Using experiments that ablate the representations of different token types, we find that template and stopword tokens are the most prone to be task-encoding. In addition, we demonstrate experimentally that lexical meaning, repetition, and text formatting are the main distinguishing characteristics of these tokens. Our work sheds light on how large language models (LLMs) learn to per
    
[^111]: SAPT：一种共享注意力框架，用于大型语言模型的参数高效持续学习

    SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of Large Language Models

    [https://arxiv.org/abs/2401.08295](https://arxiv.org/abs/2401.08295)

    提出了一种共享注意力框架（SAPT），通过共享注意力学习与选择模块对齐PET学习和选择，以同时解决大型语言模型中的灾难性遗忘和知识转移挑战。

    

    持续学习（CL）能力对于在动态世界部署大型语言模型（LLM）至关重要。现有方法设计学习模块，通过参数高效调整（PET）块获取特定任务的知识，并通过选择模块选择出相应的输入，旨在应对CL中的灾难式遗忘和知识转移挑战。然而，这些方法往往只解决其中一个挑战，忽视了通过将两个模块对齐来有效同时解决灾难式遗忘和知识转移的潜力。为此，我们提出了一种新颖的共享注意力框架（SAPT），通过共享注意力学习与选择模块来对齐PET学习和选择。在两个CL基准测试上进行的广泛实验表明SAPT的优越性。此外，当我们将其扩展到不同模型大小时，SAPT一直展现出其优越性。

    arXiv:2401.08295v2 Announce Type: replace  Abstract: The continual learning (CL) ability is vital for deploying large language models (LLMs) in the dynamic world. Existing methods devise the learning module to acquire task-specific knowledge with parameter-efficient tuning (PET) block and the selection module to pick out the corresponding one for the testing input, aiming at handling the challenges of catastrophic forgetting and knowledge transfer in CL. However, these methods tend to address only one of the challenges, ignoring the potential of aligning the two modules to effectively address catastrophic forgetting and knowledge transfer simultaneously. To this end, we propose a novel Shared Attention Framework (SAPT), to align the PET learning and selection via the Shared Attentive Learning \& Selection module. Extensive Experiments on two CL benchmarks demonstrate the superiority of SAPT. Moreover, SAPT consistently demonstrates its superiority when we scale it to different model si
    
[^112]: MARIO: 具有Python代码解释器的数学推理输出--可重复的流水线

    MARIO: MAth Reasoning with code Interpreter Output -- A Reproducible Pipeline

    [https://arxiv.org/abs/2401.08190](https://arxiv.org/abs/2401.08190)

    本文通过引入具有Python代码解释器的数学数据集，解决了大型语言模型在数学推理能力方面的挑战。

    

    大型语言模型（LLMs）在自然语言理解任务中取得了相当大的进展，但在获得真正的人工通用智能方面仍然存在一些需要填补的差距，特别是在数学推理能力方面存在的缺陷。本文通过丰富数据景观和引入一种新颖的数学数据集来解决这一挑战，该数据集增加了使用Python代码解释器的能力。

    arXiv:2401.08190v2 Announce Type: replace  Abstract: Large language models (LLMs) have seen considerable advancements in natural language understanding tasks, yet there remains a gap to bridge before attaining true artificial general intelligence, especially concerning shortcomings in mathematical reasoning capabilities. We postulate that the inherent nature of LLM training, which focuses on predicting probabilities of next token, presents challenges in effectively modeling mathematical reasoning that demands exact calculations, both from data-driven and theoretical standpoints. In this paper, we address this challenge by enriching the data landscape and introducing a novel math dataset, enhanced with a capability to utilize a Python code interpreter. This dataset is derived from GSM8K and MATH and has been further refined through a combination of GPT-4 annotations, human review, and self-training processes, where the errors in the original GSM8K training set have been fixed. Additiona
    
[^113]: 小型LLMs是弱工具学习者：多LLM代理

    Small LLMs Are Weak Tool Learners: A Multi-LLM Agent

    [https://arxiv.org/abs/2401.07324](https://arxiv.org/abs/2401.07324)

    本论文提出了一种新的策略，将大型语言模型代理（LLMs）的能力分解为计划器、调用器和总结器模块，以克服小型模型性能限制和工具更新的问题。

    

    大型语言模型（LLM）代理大大扩展了独立LLMs的能力，使它们能够与外部工具（例如API，函数）进行交互，并自主完成复杂任务。工具使用的挑战要求LLMs不仅能理解用户查询并生成答案，还要在任务规划、记忆管理、工具调用和结果总结方面表现出色。传统方法集中于训练单个具备所有这些功能的LLM，但在小型模型上会出现性能限制的问题，此外，当工具更新时，整个LLM可能需要重新训练。为了克服这些挑战，我们提出了一种新的策略，将上述能力分解为计划器、调用器和总结器。每个组件由一个单独的LLM实现，专注于特定的能力，并与其他组件合作完成任务。这种模块化框架便于进行个体更新和...

    Large Language Model (LLM) agents significantly extend the capabilities of standalone LLMs, empowering them to interact with external tools (e.g., APIs, functions) and complete complex tasks in a self-directed fashion. The challenge of tool use demands that LLMs not only understand user queries and generate answers but also excel in task planning, memory management, tool invocation, and result summarization. While traditional approaches focus on training a single LLM with all these capabilities, performance limitations become apparent, particularly with smaller models. Moreover, the entire LLM may require retraining when tools are updated. To overcome these challenges, we propose a novel strategy that decomposes the aforementioned capabilities into a planner, caller, and summarizer. Each component is implemented by a single LLM that focuses on a specific capability and collaborates with other components to accomplish the task. This modular framework facilitates individual updates and t
    
[^114]: RoleEval：大型语言模型的双语角色评估基准

    RoleEval: A Bilingual Role Evaluation Benchmark for Large Language Models

    [https://arxiv.org/abs/2312.16132](https://arxiv.org/abs/2312.16132)

    介绍了RoleEval，一个旨在评估大型语言模型角色知识记忆、利用和推理能力的双语基准，涵盖了来自各个领域的300名有影响力的人物和虚构角色，包括6000道中英文平行多项选择题，旨在系统地探究个人信息、关系、能力和经历等各个方面

    

    大型语言模型的快速发展需要有效的基准来评估其角色知识，这对于建立与现实世界的联系并提供更具沉浸感的互动至关重要。本文介绍了RoleEval，一个旨在评估角色知识的记忆、利用和推理能力的双语基准。RoleEval包括RoleEval-Global（包括国际公认的角色）和RoleEval-Chinese（包括中国流行角色），涵盖了来自各个领域（包括名人、动漫、漫画、电影、电视剧、游戏和小说）的300名有影响力的人物和虚构角色，共6000道中英文平行多项选择题。这些问题涵盖了基本知识和多跳推理能力，旨在系统地探究个人信息、关系、能力和经历等各个方面。

    arXiv:2312.16132v2 Announce Type: replace  Abstract: The rapid evolution of large language models necessitates effective benchmarks for evaluating their role knowledge, which is essential for establishing connections with the real world and providing more immersive interactions. This paper introduces RoleEval, a bilingual benchmark designed to assess the memorization, utilization, and reasoning capabilities of role knowledge. RoleEval comprises RoleEval-Global (including internationally recognized characters) and RoleEval-Chinese (including characters popular in China), with 6,000 Chinese-English parallel multiple-choice questions focusing on 300 influential people and fictional characters drawn from a variety of domains including celebrities, anime, comics, movies, TV series, games, and fictions. These questions cover basic knowledge and multi-hop reasoning abilities, aiming to systematically probe various aspects such as personal information, relationships, abilities, and experiences
    
[^115]: BloomVQA：评估分层多模态理解

    BloomVQA: Assessing Hierarchical Multi-modal Comprehension

    [https://arxiv.org/abs/2312.12716](https://arxiv.org/abs/2312.12716)

    提出了新VQA数据集BloomVQA，基于Bloom的分类法，通过层次图表示实现数据增强和模型一致性评估，揭示大型视觉语言模型在高级理解任务上的性能下降。

    

    我们提出了一个新颖的VQA数据集BloomVQA，旨在促进对大型视觉语言模型在理解任务上的全面评估。与当前的基准不同，它们通常侧重于基于事实的记忆和没有理论基础的简单推理任务，我们收集了基于图片故事的多项选择样本，反映了不同层次的理解，正如布鲁姆的分类法所展示的，在教育研究中被广泛采用的经典框架。我们的数据映射到一种新颖的分层图表示，实现了自动数据增强和表征模型一致性的新措施。我们对最近的多模态模型进行了分级评估和可靠性分析。与低级任务相比，我们发现在需要高级理解和认知能力的任务上表现下降，VQA准确性下降了高达38.0%。与早期模型相比，GPT-4V表现出...

    arXiv:2312.12716v2 Announce Type: replace-cross  Abstract: We propose a novel VQA dataset, BloomVQA, to facilitate comprehensive evaluation of large vision-language models on comprehension tasks. Unlike current benchmarks that often focus on fact-based memorization and simple reasoning tasks without theoretical grounding, we collect multiple-choice samples based on picture stories that reflect different levels of comprehension, as laid out in Bloom's Taxonomy, a classic framework for learning assessment widely adopted in education research. Our data maps to a novel hierarchical graph representation which enables automatic data augmentation and novel measures characterizing model consistency. We perform graded evaluation and reliability analysis on recent multi-modal models. In comparison to low-level tasks, we observe decreased performance on tasks requiring advanced comprehension and cognitive skills with up to 38.0% drop in VQA accuracy. In comparison to earlier models, GPT-4V demons
    
[^116]: 响应增强的半监督对话查询生成

    Response Enhanced Semi-supervised Dialogue Query Generation

    [https://arxiv.org/abs/2312.12713](https://arxiv.org/abs/2312.12713)

    提出了一种新的半监督学习框架--SemiDQG，通过未标记的对话来改善模型性能，训练响应增强的查询生成器 (RA)。

    

    从互联网获取庞大且不断更新的知识被认为是对话系统的一个重要能力。因此，针对生成对话历史记录的搜索查询而提出了对话查询生成任务，这些查询将被提交到搜索引擎以检索互联网上相关的网站。为了解决数据稀缺和领域适应性的挑战，本文提出了一个半监督学习框架 - SemiDQG，通过未标记的对话来提高模型性能。基于搜索查询通常与对话响应主题相关的观察，我们训练一个响应增强的查询生成器（RA）来提供丰富且有效的训练。

    arXiv:2312.12713v2 Announce Type: replace-cross  Abstract: Leveraging vast and continually updated knowledge from the Internet has been considered an important ability for a dialogue system. Therefore, the dialogue query generation task is proposed for generating search queries from dialogue histories, which will be submitted to a search engine for retrieving relevant websites on the Internet. In this regard, previous efforts were devoted to collecting conversations with annotated queries and training a query producer (QP) via standard supervised learning. However, these studies still face the challenges of data scarcity and domain adaptation. To address these issues, in this paper, we propose a semi-supervised learning framework -- SemiDQG, to improve model performance with unlabeled conversations. Based on the observation that the search query is typically related to the topic of dialogue response, we train a response-augmented query producer (RA) to provide rich and effective traini
    
[^117]: KGLens：一个参数化知识图解决方案，用于评估LLM知道和不知道的内容

    KGLens: A Parameterized Knowledge Graph Solution to Assess What an LLM Does and Doesn't Know

    [https://arxiv.org/abs/2312.11539](https://arxiv.org/abs/2312.11539)

    KGLens 是一个旨在衡量知识图与大型语言模型（LLMs）之间对齐程度的框架，帮助找出LLMs相对于知识图的知识不足之处。

    

    衡量知识图（KG）与大型语言模型（LLMs）之间的对齐程度是评估事实性并识别LLMs的知识盲点的有效方法。然而，这种方法面临两个主要挑战，包括将KGs转化为自然语言和高效评估这些广泛且复杂的结构。在本文中，我们提出了KGLens--一个旨在衡量KGs和LLMs之间对齐程度，并找出LLMs相对于KGs的知识缺陷的新颖框架。KGLens具有一个图引导的问题生成器，用于将KGs转化为自然语言，以及一个基于参数化KG结构的精心设计的采样策略，以加快KG的遍历。我们使用来自Wikidata的三个领域特定KG进行实验，这些KG包括超过19,000条边，700个关系和21,000个实体。我们跨越8个LLMs的分析表明，KGLens不仅

    arXiv:2312.11539v2 Announce Type: replace  Abstract: Measuring the alignment between a Knowledge Graph (KG) and Large Language Models (LLMs) is an effective method to assess the factualness and identify the knowledge blind spots of LLMs. However, this approach encounters two primary challenges including the translation of KGs into natural language and the efficient evaluation of these extensive and complex structures. In this paper, we present KGLens--a novel framework aimed at measuring the alignment between KGs and LLMs, and pinpointing the LLMs' knowledge deficiencies relative to KGs. KGLens features a graph-guided question generator for converting KGs into natural language, along with a carefully designed sampling strategy based on parameterized KG structure to expedite KG traversal. We conducted experiments using three domain-specific KGs from Wikidata, which comprise over 19,000 edges, 700 relations, and 21,000 entities. Our analysis across eight LLMs reveals that KGLens not only
    
[^118]: 用于更快的LLM推理的级联推测草图

    Cascade Speculative Drafting for Even Faster LLM Inference

    [https://arxiv.org/abs/2312.11462](https://arxiv.org/abs/2312.11462)

    引入了Cascade Speculative Drafting（CS Drafting）算法，通过垂直级联消除神经模型的自回归生成，通过水平级联优化草稿中的时间分配，从而进一步提高LLM推理效率。

    

    引入了增强大型语言模型（LLM）推理效率的级联推测草图，通过较小的模型生成草稿来运作。较大的目标模型然后查看这个草稿以与其输出对齐，目标模型的任何接受都将减少目标模型运行的数量，从而提高效率。然而，在级联推测的草图过程中包括缓慢的自回归生成，并为生成的标记分配相同的时间，而不考虑它们的重要性。这些低效性共同导致级联推测的性能不佳。为了进一步改善LLM推理，我们引入了级联推测草图（CS Drafting），这是一种整合了两种级联类型的推测执行算法。垂直级联从神经模型中消除自回归生成，而水平级联优化了草稿中的时间分配

    arXiv:2312.11462v3 Announce Type: replace-cross  Abstract: Introduced to enhance the efficiency of large language model (LLM) inference, speculative decoding operates by having a smaller model generate a draft. A larger target model then reviews this draft to align with its output, and any acceptance by the target model results in a reduction of the number of the target model runs, ultimately improving efficiency. However, the drafting process in speculative decoding includes slow autoregressive generation and allocates equal time to generating tokens, irrespective of their importance. These inefficiencies collectively contribute to the suboptimal performance of speculative decoding. To further improve LLM inference, we introduce Cascade Speculative Drafting (CS Drafting), a speculative execution algorithm that incorporates two types of cascades. The Vertical Cascade eliminates autoregressive generation from neural models, while the Horizontal Cascade optimizes time allocation in draft
    
[^119]: 利用大型语言模型进行分割与重述任务

    Split and Rephrase with Large Language Models

    [https://arxiv.org/abs/2312.11075](https://arxiv.org/abs/2312.11075)

    评估了大型语言模型在Split and Rephrase任务上的表现，表明在主要指标上有显著改进，但在分割一致性方面仍有待提高。

    

    Split and Rephrase (SPRP)任务旨在将复杂句子分解为一系列更短的符合语法规则的句子，同时保持原始含义，有助于人类和机器处理复杂文本。这也是一个有价值的测试平台，可以评估自然语言处理模型，因为其需要对复杂的语法方面进行建模。在这项工作中，我们评估了大型语言模型在该任务上的表现，显示它们可以在主要指标上比现有技术有很大改进，尽管在拆分一致性方面仍有差距。来自两项人类评估的结果进一步支持自动度量结果得出的结论。我们提供了一项全面的研究，包括提示变体、领域转移、参数规模和训练数据量不同的微调预训练语言模型，同时与指导调整的零射和少射方法进行对比。

    arXiv:2312.11075v3 Announce Type: replace  Abstract: The Split and Rephrase (SPRP) task, which consists in splitting complex sentences into a sequence of shorter grammatical sentences, while preserving the original meaning, can facilitate the processing of complex texts for humans and machines alike. It is also a valuable testbed to evaluate natural language processing models, as it requires modelling complex grammatical aspects. In this work, we evaluate large language models on the task, showing that they can provide large improvements over the state of the art on the main metrics, although still lagging in terms of splitting compliance. Results from two human evaluations further support the conclusions drawn from automated metric results. We provide a comprehensive study that includes prompting variants, domain shift, fine-tuned pretrained language models of varying parameter size and training data volumes, contrasted with both zero-shot and few-shot approaches on instruction-tuned 
    
[^120]: 具有硬件高效训练的门控线性注意力变换器

    Gated Linear Attention Transformers with Hardware-Efficient Training

    [https://arxiv.org/abs/2312.06635](https://arxiv.org/abs/2312.06635)

    该论文提出了一种具有硬件高效性的线性注意力算法，可在短序列长度下比现有方法更快，同时推广到了具有数据相关门的更具表达能力的线性注意力变体。

    

    具有线性注意力的变压器允许进行高效的并行训练，同时可以被表述为具有2D（矩阵值）隐藏状态的RNN，从而享受线性时间推断复杂度。然而，线性注意力通常表现不如普通softmax注意力。而且，当前的线性注意力实现缺乏I/O感知性，因此比高度优化的softmax注意力实现更慢。本文描述了一种适用于线性注意力的硬件高效算法，它在内存移动和可并行性之间进行折中。由此产生的实现，被称为FLASHLINEARATTENTION，在短序列长度（例如，1K）下，即使作为单独的层也比FLASHATTENTION-2(Dao, 2023)更快。然后，我们将该算法推广到具有数据相关门的更具表达能力的线性注意力变体。当用作变换器中标准注意力层的替代时，产生的门控

    arXiv:2312.06635v4 Announce Type: replace-cross  Abstract: Transformers with linear attention allow for efficient parallel training but can simultaneously be formulated as an RNN with 2D (matrix-valued) hidden states, thus enjoying linear-time inference complexity. However, linear attention generally underperforms ordinary softmax attention. Moreover, current implementations of linear attention lack I/O-awareness and are thus slower than highly optimized implementations of softmax attention. This work describes a hardware-efficient algorithm for linear attention that trades off memory movement against parallelizability. The resulting implementation, dubbed FLASHLINEARATTENTION, is faster than FLASHATTENTION-2(Dao, 2023) as a standalone layer even at short sequence lengths (e.g., 1K). We then generalize this algorithm to a more expressive variant of linear attention with data-dependent gates. When used as a replacement for the standard attention layer in Transformers, the resulting gate
    
[^121]: 因果ATE减轻了控制文本生成中的意外偏差

    Causal ATE Mitigates Unintended Bias in Controlled Text Generation

    [https://arxiv.org/abs/2311.11229](https://arxiv.org/abs/2311.11229)

    因果ATE方法可解决语言模型中属性控制任务中存在的意外偏差问题，并在毒性缓解问题中得到验证。

    

    我们通过因果平均处理效应（Causal ATE）方法研究了语言模型中的属性控制。现有方法中，针对语言模型中属性控制任务，会检查句子中单词与感兴趣属性的共现情况，并加以控制。然而，在训练数据集中，单词与属性之间的伪相关性可能会导致模型在推断时出现属性存在的幻觉。我们展示了简单的基于扰动的因果ATE方法消除了这种意外效应。具体地，我们将其应用在毒性缓解问题上，毒性缓解的一个重要挑战在于在去毒后经常出现对受保护群体的无意偏见。我们证明了使用因果ATE度量可以解决这种意外偏差，并且我们提供了实验证实。

    arXiv:2311.11229v2 Announce Type: replace  Abstract: We study attribute control in language models through the method of Causal Average Treatment Effect (Causal ATE). Existing methods for the attribute control task in Language Models (LMs) check for the co-occurrence of words in a sentence with the attribute of interest, and control for them. However, spurious correlation of the words with the attribute in the training dataset, can cause models to hallucinate the presence of the attribute when presented with the spurious correlate during inference. We show that the simple perturbation-based method of Causal ATE removes this unintended effect. Specifically, we ground it in the problem of toxicity mitigation, where a significant challenge lies in the inadvertent bias that often emerges towards protected groups post detoxification. We show that this unintended bias can be solved by the use of the Causal ATE metric and rigorously prove our claim. We provide experimental validations for our
    
[^122]: WatME：通过词汇冗余实现无损水印

    WatME: Towards Lossless Watermarking Through Lexical Redundancy

    [https://arxiv.org/abs/2311.09832](https://arxiv.org/abs/2311.09832)

    WatME通过利用词汇冗余的语言先验知识，动态优化语言模型解码过程中的词汇使用，避免适当词汇不可用的情况，维持语言模型的表现力。

    

    arXiv:2311.09832v2 公告类型：替换。文本水印技术已经成为一种重要的检测机器生成文本的技术。然而，现有方法通常在解码过程中使用任意的词汇分割，导致在响应生成过程中缺乏适当的词汇，并破坏了语言模型的表现力，严重降低了文本响应的质量。为了解决这些问题，我们引入了一种新颖的方法，即互斥式水印（WatME）。具体来说，通过利用固有词汇冗余的语言先验知识，WatME 可以在语言模型的解码过程中动态优化可用词汇的使用。它采用互斥规则来管理这种冗余，避免了适当的词汇不可用的情况，同时保持了大型语言模型（LLMs）的表现力。我们提出理论分析和实证证据，证明了WatME的有效性。

    arXiv:2311.09832v2 Announce Type: replace  Abstract: Text watermarking has emerged as an important technique for detecting machine-generated text. However, existing methods generally use arbitrary vocabulary partitioning during decoding, which results in the absence of appropriate words during the response generation and disrupts the language model's expressiveness, thus severely degrading the quality of text response. To address these issues, we introduce a novel approach, Watermarking with Mutual Exclusion (WatME). Specifically, by leveraging linguistic prior knowledge of inherent lexical redundancy, WatME can dynamically optimize the use of available vocabulary during the decoding process of language models. It employs a mutually exclusive rule to manage this redundancy, avoiding situations where appropriate words are unavailable and maintaining the expressive power of large language models (LLMs). We present theoretical analysis and empirical evidence demonstrating that WatME subst
    
[^123]: 探究大型语言模型如何表达对超出参数化知识范围的问题的不确定性

    Examining LLMs' Uncertainty Expression Towards Questions Outside Parametric Knowledge

    [https://arxiv.org/abs/2311.09731](https://arxiv.org/abs/2311.09731)

    本研究系统地调查了大型语言模型在缺乏足够参数化知识的情况下如何表达对超出其知识范围的问题的不确定性，并强调了诚实与帮助性之间的权衡。

    

    这项工作旨在系统地调查大型语言模型在缺乏足够参数化知识以生成合理回应的情况下的行为，强调诚实与帮助性之间的权衡。为了精确确定语言模型的知识空白挑战，我们诊断性地创建了包含不存在概念或错误前提的无法回答的问题，确保它们超出了语言模型庞大的训练数据。通过编制一个包含既有无法回答也有可回答问题的基准，UnknownBench，我们定量评估语言模型在保持诚实的同时提供帮助的表现。使用一个模型无关的统一信心引导方法，我们观察到大多数语言模型在一致拒绝或表达对超出其参数化知识范围的问题的不确定性方面表现不佳。

    arXiv:2311.09731v2 Announce Type: replace-cross  Abstract: Can large language models (LLMs) express their uncertainty in situations where they lack sufficient parametric knowledge to generate reasonable responses? This work aims to systematically investigate LLMs' behaviors in such situations, emphasizing the trade-off between honesty and helpfulness. To tackle the challenge of precisely determining LLMs' knowledge gaps, we diagnostically create unanswerable questions containing non-existent concepts or false premises, ensuring that they are outside the LLMs' vast training data. By compiling a benchmark, UnknownBench, which consists of both unanswerable and answerable questions, we quantitatively evaluate the LLMs' performance in maintaining honesty while being helpful. Using a model-agnostic unified confidence elicitation approach, we observe that most LLMs fail to consistently refuse or express uncertainty towards questions outside their parametric knowledge, although instruction fin
    
[^124]: 党派群体的智慧：比较人类和基于LLM的代理人的集体智能

    The Wisdom of Partisan Crowds: Comparing Collective Intelligence in Humans and LLM-based Agents

    [https://arxiv.org/abs/2311.09665](https://arxiv.org/abs/2311.09665)

    本文研究了基于LLM的代理人在扮演党派角色时，展现出类似于人类群体的党派偏见、并通过商讨收敛到更准确信念的能力。

    

    人类群体能够通过商讨达成更准确的信念，即使在存在极化和党派偏见的情况下也是如此，这一现象被称为“党派群体的智慧”。由大型语言模型（LLMs）驱动的生成代理人越来越多地被用来模拟人类集体行为，然而很少有基准用于评估它们的动态与人类群体行为的对比。在本文中，我们研究了在提示扮演党派人物（例如，民主党人或共和党人）的LLM代理人群体中，党派群体的智慧出现的程度。我们发现他们不仅显示出类似于人类的党派偏见，而且通过商讨像人类一样收敛到更准确的信念。然后，我们确定了几个干扰收敛的因素，包括链式思维提示的使用和人物缺乏细节。相反，对人类数据进行微调似乎增强

    arXiv:2311.09665v2 Announce Type: replace  Abstract: Human groups are able to converge on more accurate beliefs through deliberation, even in the presence of polarization and partisan bias -- a phenomenon known as the "wisdom of partisan crowds." Generated agents powered by Large Language Models (LLMs) are increasingly used to simulate human collective behavior, yet few benchmarks exist for evaluating their dynamics against the behavior of human groups. In this paper, we examine the extent to which the wisdom of partisan crowds emerges in groups of LLM-based agents that are prompted to role-play as partisan personas (e.g., Democrat or Republican). We find that they not only display human-like partisan biases, but also converge to more accurate beliefs through deliberation as humans do. We then identify several factors that interfere with convergence, including the use of chain-of-thought prompt and lack of details in personas. Conversely, fine-tuning on human data appears to enhance co
    
[^125]: 解码易感性：通过计算方法对错误信息进行建模

    Decoding Susceptibility: Modeling Misbelief to Misinformation Through a Computational Approach

    [https://arxiv.org/abs/2311.09630](https://arxiv.org/abs/2311.09630)

    通过计算方法对用户的潜在易感性水平进行建模，可以帮助理解易受错误信息影响的程度，为后续研究和应用提供重要参考。

    

    易受错误信息影响的程度描述了对不可验证主张的信仰程度，这是个体思维过程中的潜在因素，不可观察。现有易感性研究严重依赖于自我报告的信念，这可能存在偏见，收集成本高，并且难以在后续应用中扩展。为了解决这些限制，我们在这项研究中提出了一种计算方法来建模用户的潜在易感性水平。正如先前的研究所示，易感性受到各种因素的影响（例如人口统计因素、政治意识形态），并直接影响人们在社交媒体上的转发行为。为了表示基础心理过程，我们的易感性建模将这些因素作为输入，受到人们分享行为监督的引导。使用COVID-19作为实验领域，我们的实验证明了易感性评分之间存在显著的一致性。

    arXiv:2311.09630v2 Announce Type: replace  Abstract: Susceptibility to misinformation describes the degree of belief in unverifiable claims, a latent aspect of individuals' mental processes that is not observable. Existing susceptibility studies heavily rely on self-reported beliefs, which can be subject to bias, expensive to collect, and challenging to scale for downstream applications. To address these limitations, in this work, we propose a computational approach to model users' latent susceptibility levels. As shown in previous research, susceptibility is influenced by various factors (e.g., demographic factors, political ideology), and directly influences people's reposting behavior on social media. To represent the underlying mental process, our susceptibility modeling incorporates these factors as inputs, guided by the supervision of people's sharing behavior. Using COVID-19 as a testbed domain, our experiments demonstrate a significant alignment between the susceptibility score
    
[^126]: 使用基于LLM的代理网络模拟意见动态

    Simulating Opinion Dynamics with Networks of LLM-based Agents

    [https://arxiv.org/abs/2311.09618](https://arxiv.org/abs/2311.09618)

    提出了一种基于大型语言模型（LLMs）人口的新方法来模拟意见动态，发现LLM代理存在固有偏见导致模拟代理趋向于科学现实一致的共识，但引入确认偏见后观察到意见分裂，突显了LLM代理在该领域的潜力和局限性。

    

    准确模拟人类意见动态对于理解各种社会现象至关重要，包括极化和错误信息的传播。然而，常用于此类模拟的基于代理的模型（ABM）经常会过分简化人类行为。我们提出了一种基于大型语言模型（LLMs）人口的模拟意见动态的新方法。我们的研究结果显示，LLM代理存在一种对产生准确信息的强烈固有偏见，导致模拟代理趋向于与科学现实一致的共识。然而，这种偏见限制了它们在理解气候变化等问题上抵制共识观点的效用。通过引入提示工程诱导确认偏见后，我们观察到了与现有基于代理模型和意见动态研究一致的意见分裂。这些见解突显了LLM代理在该领域的潜力和局限性，并提出了一条路径。

    arXiv:2311.09618v2 Announce Type: replace-cross  Abstract: Accurately simulating human opinion dynamics is crucial for understanding a variety of societal phenomena, including polarization and the spread of misinformation. However, the agent-based models (ABMs) commonly used for such simulations often over-simplify human behavior. We propose a new approach to simulating opinion dynamics based on populations of Large Language Models (LLMs). Our findings reveal a strong inherent bias in LLM agents towards producing accurate information, leading simulated agents to consensus in line with scientific reality. This bias limits their utility for understanding resistance to consensus views on issues like climate change. After inducing confirmation bias through prompt engineering, however, we observed opinion fragmentation in line with existing agent-based modeling and opinion dynamics research. These insights highlight the promise and limitations of LLM agents in this domain and suggest a path
    
[^127]: 数字苏格拉底：通过解释批评评估LLM

    Digital Socrates: Evaluating LLMs through Explanation Critiques

    [https://arxiv.org/abs/2311.09613](https://arxiv.org/abs/2311.09613)

    通过定义新的解释批评任务、创建人工验证过的数据集并训练开源自动批评模型，数字苏格拉底有助于揭示学生模型的见解。

    

    虽然LLMs可以提供有理有据的解释以及答案，但这些解释的性质和质量仍然知之甚少。作为回应，我们的目标是定义一种详细的方式来表征现代模型的解释能力，创建一个细致且可解释的解释评估工具，该工具可以自动生成这种表征，而无需依赖昂贵的API调用或人类注释。我们的方法是：(a)定义解释批评的新任务——识别和分类解释中的任何主要缺陷，并提供建议来解决这些缺陷；(b)为此任务创建一个规模可观且经过人工验证的数据集；(c)使用这些数据训练一个开源的自动批评模型（称为数字苏格拉底）。通过定量和定性分析，我们展示了数字苏格拉底如何有助于通过检查其理由来揭示有关学生模型的见解。

    arXiv:2311.09613v2 Announce Type: replace-cross  Abstract: While LLMs can provide reasoned explanations along with their answers, the nature and quality of those explanations are still poorly understood. In response, our goal is to define a detailed way of characterizing the explanation capabilities of modern models and to create a nuanced, interpretable explanation evaluation tool that can generate such characterizations automatically, without relying on expensive API calls or human annotations. Our approach is to (a) define the new task of explanation critiquing - identifying and categorizing any main flaw in an explanation and providing suggestions to address the flaw, (b) create a sizeable, human-verified dataset for this task, and (c) train an open-source, automatic critique model (called Digital Socrates) using this data. Through quantitative and qualitative analysis, we demonstrate how Digital Socrates is useful for revealing insights about student models by examining their reas
    
[^128]: Fusion-Eval: 将评估器与LLMs集成

    Fusion-Eval: Integrating Evaluators with LLMs

    [https://arxiv.org/abs/2311.09204](https://arxiv.org/abs/2311.09204)

    Fusion-Eval是一种创新方法，利用LLMs整合不同辅助评估器的见解，极大提升自然语言系统评估的有效性。

    

    自然语言系统的评估在自然语言理解和高级推理领域面临着重大挑战。本文介绍了一种名为“Fusion-Eval”的创新方法，利用大型语言模型（LLMs）来整合来自各种辅助评估器的见解。每个评估器专门负责评估响应的不同方面。这种独特策略使得Fusion-Eval能够有效地跨越各种任务和标准，增强现有评估方法的效果。在SummEval上，Fusion-Eval与人类之间的系统级Kendall-Tau相关性达到0.962，在TopicalChat上的轮级Spearman相关性达到0.744，远高于基准方法。这些结果突显了Fusion-Eval在自然语言系统评估领域的巨大潜力。

    arXiv:2311.09204v2 Announce Type: replace-cross  Abstract: Evaluating natural language systems poses significant challenges, particularly in the realms of natural language understanding and high-level reasoning. In this paper, we introduce "Fusion-Eval", an innovative approach that leverages Large Language Models (LLMs) to integrate insights from various assistant evaluators. Each of these evaluators specializes in assessing distinct aspects of responses. This unique strategy enables Fusion-Eval to function effectively across a diverse range of tasks and criteria, enhancing the effectiveness of existing evaluation methods. Fusion-Eval achieves a 0.962 system-level Kendall-Tau correlation with humans on SummEval and a 0.744 turn-level Spearman correlation on TopicalChat, which is significantly higher than baseline methods. These results highlight Fusion-Eval's significant potential in the realm of natural language system evaluation.
    
[^129]: 可实现大型语言模型从规则中学习

    Enabling Large Language Models to Learn from Rules

    [https://arxiv.org/abs/2311.08883](https://arxiv.org/abs/2311.08883)

    本文探索了一种新的学习范式，将基于规则的知识编码到大型语言模型中，并提出了规则提取方法。

    

    大型语言模型（LLMs）在完成各种真实世界任务时表现出色。目前LLMs的知识学习范式主要基于从例子中学习，其中LLMs从一定数量的监督示例中隐式学习内部规则。然而，当训练示例有限时，这种学习范式可能无法很好地学习那些复杂的规则。我们受到启发，人类可以通过从规则中学习来另一种方式学习新任务或知识。因此，在本文中，我们旨在探索这种新的学习范式的可行性，即将基于规则的知识编码到LLMs中。我们进一步提出了规则提取，首先利用LLMs的强大上下文能力来从文本中提取知识。

    arXiv:2311.08883v2 Announce Type: replace  Abstract: Large language models (LLMs) have shown incredible performance in completing various real-world tasks. The current knowledge learning paradigm of LLMs is mainly based on learning from examples, in which LLMs learn the internal rule implicitly from a certain number of supervised examples. However, this learning paradigm may not well learn those complicated rules, especially when the training examples are limited. We are inspired that humans can learn the new tasks or knowledge in another way by learning from rules. That is, humans can learn new tasks or grasps new knowledge quickly and generalize well given only a detailed rule and a few optional examples. Therefore, in this paper, we aim to explore the feasibility of this new learning paradigm, which targets on encoding rule-based knowledge into LLMs. We further propose rule distillation, which first uses the strong in-context abilities of LLMs to extract the knowledge from the textu
    
[^130]: StrategyLLM：大型语言模型作为问题解决的策略生成器、执行器、优化器和评估器

    StrategyLLM: Large Language Models as Strategy Generators, Executors, Optimizers, and Evaluators for Problem Solving

    [https://arxiv.org/abs/2311.08803](https://arxiv.org/abs/2311.08803)

    StrategyLLM提出了一个框架，利用大型语言模型的能力自动构建可推广和一致的少次提示，优于竞争基线，不需要人工参与。

    

    大多数现有的思维链 (CoT) 提示方法存在泛化和一致性问题，因为它们常常依赖于特定实例的解决方案，这些解决方案可能不适用于其他情况，并缺乏在推理步骤中的任务级一致性。为解决这些限制，我们提出了一个全面的框架，StrategyLLM，利用LLM的能力自动构建可推广和一致的少次提示以用于各种任务。为此，StrategyLLM 使用四个基于LLM的代理：策略生成器、执行器、优化器和评估器，共同工作以为给定任务生成、评估和选择有前途的策略。实验结果表明，在13个数据集上跨4个挑战性任务上，不需要人工参与，StrategyLLM 在数学推理（34.21%->38.79%）、常见推理等任务上优于竞争基线CoT-SC，该基线需要人工注释的解决方案。

    arXiv:2311.08803v2 Announce Type: replace  Abstract: Most existing chain-of-thought (CoT) prompting methods suffer from the issues of generalizability and consistency, as they often rely on instance-specific solutions that may not be applicable to other cases and lack task-level consistency in their reasoning steps. To address these limitations, we propose a comprehensive framework, StrategyLLM, harnessing the capabilities of LLMs to construct generalizable and consistent few-shot prompts for various tasks automatically. To this end, StrategyLLM employs four LLM-based agents: strategy generator, executor, optimizer, and evaluator, working together to generate, evaluate, and select promising strategies for a given task. The experimental results demonstrate that StrategyLLM outperforms the competitive baseline CoT-SC that requires human-annotated solutions on 13 datasets across 4 challenging tasks without human involvement, including math reasoning (34.21% $\rightarrow$ 38.79%), commonse
    
[^131]: SimpleSafetyTests：一个用于识别大语言模型中关键安全风险的测试套件

    SimpleSafetyTests: a Test Suite for Identifying Critical Safety Risks in Large Language Models

    [https://arxiv.org/abs/2311.08370](https://arxiv.org/abs/2311.08370)

    引入SimpleSafetyTests（SST）作为一个新的测试套件，用于快速系统地识别大语言模型中关键的安全风险

    

    过去一年，大语言模型（LLMs）的发展急剧加速。然而，如果缺乏适当的引导和保障，LLMs将很容易遵循恶意指令，提供不安全的建议，并生成有毒内容。我们引入SimpleSafetyTests（SST）作为一个新的测试套件，可以快速系统地识别此类关键安全风险。该测试套件包括100个测试提示，涵盖五个LLMs应该拒绝遵从的伤害领域。我们测试了11个开放获取和开源LLMs以及四个封闭源LLMs，并发现了关键的安全性弱点。虽然其中一些模型没有给出单一的不安全响应，但大多数对超过20%的提示给出了不安全响应，极端情况下超过50%的不安全响应。在系统提示中加入强调安全性的前置内容显著减少了不安全响应的发生，但并不能完全阻止它们发生。

    arXiv:2311.08370v2 Announce Type: replace  Abstract: The past year has seen rapid acceleration in the development of large language models (LLMs). However, without proper steering and safeguards, LLMs will readily follow malicious instructions, provide unsafe advice, and generate toxic content. We introduce SimpleSafetyTests (SST) as a new test suite for rapidly and systematically identifying such critical safety risks. The test suite comprises 100 test prompts across five harm areas that LLMs, for the vast majority of applications, should refuse to comply with. We test 11 open-access and open-source LLMs and four closed-source LLMs, and find critical safety weaknesses. While some of the models do not give a single unsafe response, most give unsafe responses to more than 20% of the prompts, with over 50% unsafe responses in the extreme. Prepending a safety-emphasising system prompt substantially reduces the occurrence of unsafe responses, but does not completely stop them from happenin
    
[^132]: 在学习之前遗忘：利用参数化算术进行大型语言模型中的知识更新

    Forgetting before Learning: Utilizing Parametric Arithmetic for Knowledge Updating in Large Language Models

    [https://arxiv.org/abs/2311.08011](https://arxiv.org/abs/2311.08011)

    提出了一种名为F-Learning的新微调范式，利用参数化算术促进旧知识的遗忘和新知识的学习，在大型语言模型中显著改善知识更新性能

    

    大型语言模型（LLMs）的最新进展展示了它们在文本理解和生成方面出色的能力。然而，即使更强大的LLMs也会受到从训练语料库中获取错误或过时信息的影响。直接使用包含新知识的数据进行二次微调可能无法有效更新知识，这是由于旧知识和新知识之间的冲突。本文提出了一种名为F-Learning（学习之前遗忘）的微调新范式，它采用参数化算术来促进旧知识的遗忘和新知识的学习。在两个公开可用数据集上的实验结果表明，我们提出的F-Learning显著改善了完全微调和LoRA微调的知识更新性能，在大多数情况下同时优于现有基线。此外，我们还发现了遗忘旧知识

    arXiv:2311.08011v2 Announce Type: replace  Abstract: Recent advancements in Large Language Models (LLMs) have showcased their remarkable capabilities in text understanding and generation. However, even stronger LLMs are susceptible to acquiring erroneous or obsolete information from the training corpus. Direct secondary fine-tuning with data containing new knowledge may be ineffective in updating knowledge due to the conflict between old and new knowledge. In this paper, we propose a new paradigm for fine-tuning called F-Learning (Forgetting before Learning), which employs parametric arithmetic to facilitate the forgetting of old knowledge and learning of new knowledge. Experimental results on two publicly available datasets demonstrate that our proposed F-Learning can obviously improve the knowledge updating performance of both full fine-tuning and LoRA fine-tuning, simultaneously outperforming the existing baselines in most cases. Moreover, we have also discovered that forgetting old
    
[^133]: ChartCheck：对真实世界图表图像进行可解释事实检查

    ChartCheck: Explainable Fact-Checking over Real-World Chart Images

    [https://arxiv.org/abs/2311.07453](https://arxiv.org/abs/2311.07453)

    该论文介绍了ChartCheck，这是一个用于对真实世界图表进行可解释事实检查的新型数据集，旨在解决图表被误用传播错误信息的问题，并提出了视觉语言和图表到表格模型的基线。

    

    虽然事实验证在自然语言处理领域引起了广泛关注，但至今仍然疏忽了针对数据可视化（如图表）的误导性陈述进行验证。图表通常用于总结和传达关键信息，但它们也很容易被误用以传播错误信息和推广某种议程。在本文中，我们引入了ChartCheck，这是一个针对真实世界图表的可解释事实检查的新型大规模数据集，包含了1.7k张图表和10.5k人为撰写的声明和解释。我们使用视觉语言和图表到表格模型系统地评估ChartCheck，并向社区提出了一个基线。最后，我们研究了对这些模型构成挑战的图表推理类型和视觉属性。

    arXiv:2311.07453v2 Announce Type: replace  Abstract: Whilst fact verification has attracted substantial interest in the natural language processing community, verifying misinforming statements against data visualizations such as charts has so far been overlooked. Charts are commonly used in the real-world to summarize and communicate key information, but they can also be easily misused to spread misinformation and promote certain agendas. In this paper, we introduce ChartCheck, a novel, large-scale dataset for explainable fact-checking against real-world charts, consisting of 1.7k charts and 10.5k human-written claims and explanations. We systematically evaluate ChartCheck using vision-language and chart-to-table models, and propose a baseline to the community. Finally, we study chart reasoning types and visual attributes that pose a challenge to these models
    
[^134]: 通过模拟将结构归纳偏差注入Seq2Seq模型

    Injecting a Structural Inductive Bias into a Seq2Seq Model by Simulation

    [https://arxiv.org/abs/2310.00796](https://arxiv.org/abs/2310.00796)

    通过模拟结构转换在Seq2Seq模型中注入结构归纳偏差，提高了系统泛化和FST类似任务的少样本学习。

    

    强烈的归纳偏差有助于从少量数据中学习，并帮助在训练分布之外进行泛化。流行的神经架构如Transformers本身缺乏seq2seq NLP任务的强结构归纳偏差。因此，即使在大量文本上进行了预训练，它们也在系统泛化方面遇到困难，例如在外推到更长的输入时。我们展示了如何通过预训练来有效地将结构归纳偏差注入seq2seq模型，以在合成数据上模拟结构转换。具体地，我们通过预训练模拟FST描述来将结构归纳偏差注入到Transformer中。我们的实验表明，我们的方法给予了所需的归纳偏差，从而提高了系统泛化能力和FST类似任务的少样本学习。我们的分析显示

    arXiv:2310.00796v2 Announce Type: replace  Abstract: Strong inductive biases enable learning from little data and help generalization outside of the training distribution. Popular neural architectures such as Transformers lack strong structural inductive biases for seq2seq NLP tasks on their own. Consequently, they struggle with systematic generalization beyond the training distribution, e.g. with extrapolating to longer inputs, even when pre-trained on large amounts of text. We show how a structural inductive bias can be efficiently injected into a seq2seq model by pre-training it to simulate structural transformations on synthetic data. Specifically, we inject an inductive bias towards Finite State Transducers (FSTs) into a Transformer by pre-training it to simulate FSTs given their descriptions. Our experiments show that our method imparts the desired inductive bias, resulting in improved systematic generalization and better few-shot learning for FST-like tasks. Our analysis shows t
    
[^135]: Transformer语言模型中关系解码的线性性

    Linearity of Relation Decoding in Transformer Language Models

    [https://arxiv.org/abs/2308.09124](https://arxiv.org/abs/2308.09124)

    在Transformer语言模型中，部分关系的计算可以通过对主题表示进行单一线性转换来很好地近似，但并非所有关系都能通过线性编码。

    

    Transformer语言模型中编码的许多知识可以用关系的形式表达：词语及其同义词之间的关系，实体及其属性之间的关系等。我们展示，对于某些关系子集，这种计算可以很好地近似为对主题表示进行单一线性转换。线性关系表示可以通过从单个提示构建对LM的一阶近似来获得，并且可应用于各种事实，常识和语言关系。然而，我们还发现许多情况，LM的预测虽然准确地捕捉了关系知识，但这种知识并没有线性地编码在它们的表示中。因此，我们的结果揭示了Transformer语言模型中一种简单、可解释但异质部署的知识表示策略。

    arXiv:2308.09124v2 Announce Type: replace  Abstract: Much of the knowledge encoded in transformer language models (LMs) may be expressed in terms of relations: relations between words and their synonyms, entities and their attributes, etc. We show that, for a subset of relations, this computation is well-approximated by a single linear transformation on the subject representation. Linear relation representations may be obtained by constructing a first-order approximation to the LM from a single prompt, and they exist for a variety of factual, commonsense, and linguistic relations. However, we also identify many cases in which LM predictions capture relational knowledge accurately, but this knowledge is not linearly encoded in their representations. Our results thus reveal a simple, interpretable, but heterogeneously deployed knowledge representation strategy in transformer LMs.
    
[^136]: 历史感知的对话式稠密检索

    History-Aware Conversational Dense Retrieval. (arXiv:2401.16659v1 [cs.IR])

    [http://arxiv.org/abs/2401.16659](http://arxiv.org/abs/2401.16659)

    该论文提出了一种历史感知的对话式稠密检索系统，通过上下文去噪的查询重构以及根据历史轮次的实际影响自动挖掘监督信号改进了现有的对话式稠密检索方法。

    

    对话搜索通过实现用户和系统之间的多轮交互，实现了复杂信息检索的便利。支持这种交互需要对对话输入有全面的理解，以便根据历史信息制定良好的搜索查询。特别是，搜索查询应包括来自先前对话回合的相关信息。然而，目前的对话式稠密检索方法主要依赖于对经过精调的预训练专门检索器进行整个对话式搜索会话的优化，这可能会变得冗长和嘈杂。此外，现有方法受现有数据集中手动监督信号数量的限制。为了解决上述问题，我们提出了一种历史感知的对话式稠密检索(HAConvDR)系统，它结合了两个思想：上下文去噪的查询重构和根据历史轮次的实际影响进行自动挖掘监督信号。

    Conversational search facilitates complex information retrieval by enabling multi-turn interactions between users and the system. Supporting such interactions requires a comprehensive understanding of the conversational inputs to formulate a good search query based on historical information. In particular, the search query should include the relevant information from the previous conversation turns. However, current approaches for conversational dense retrieval primarily rely on fine-tuning a pre-trained ad-hoc retriever using the whole conversational search session, which can be lengthy and noisy. Moreover, existing approaches are limited by the amount of manual supervision signals in the existing datasets. To address the aforementioned issues, we propose a History-Aware Conversational Dense Retrieval (HAConvDR) system, which incorporates two ideas: context-denoised query reformulation and automatic mining of supervision signals based on the actual impact of historical turns. Experime
    
[^137]: CC查询：从公开文献中发现大规模领域特定知识

    Query of CC: Unearthing Large Scale Domain-Specific Knowledge from Public Corpora. (arXiv:2401.14624v1 [cs.CL])

    [http://arxiv.org/abs/2401.14624](http://arxiv.org/abs/2401.14624)

    本论文提出了一种通过大型语言模型来收集特定领域知识的高效方法，通过该方法构建了一个高质量的名为“Knowledge Pile”的数据集，实验证明其显著改善了特定领域的数据稀缺问题。

    

    大型语言模型在各种任务中展示了显著的潜力，然而特定领域的开源模型和数据仍然非常稀缺。之前的研究主要集中在手动指定资源和收集特定领域的高质量数据，这消耗了大量时间和精力。为了解决这个问题，我们提出了一种基于大型语言模型的高效数据收集方法“CC查询”。该方法通过大型语言模型引导种子信息，并从公开文献中检索相关数据。它不仅收集了特定领域的知识相关数据，还揭示了潜在的推理过程数据。通过应用这种方法，我们构建了一个名为“Knowledge Pile”的高质量数据集，涵盖了包括STEM科学和人文科学在内的四个主要领域。实验结果表明，“Knowledge Pile”显著改善了

    Large language models have demonstrated remarkable potential in various tasks, however, there remains a significant scarcity of open-source models and data for specific domains. Previous works have primarily focused on manually specifying resources and collecting high-quality data on specific domains, which significantly consume time and effort. To address this limitation, we propose an efficient data collection method~\textit{Query of CC} based on large language models. This method bootstraps seed information through a large language model and retrieves related data from public corpora. It not only collects knowledge-related data for specific domains but unearths the data with potential reasoning procedures. Through the application of this method, we have curated a high-quality dataset called~\textsc{Knowledge Pile}, encompassing four major domains, including stem and humanities sciences, among others. Experimental results demonstrate that~\textsc{Knowledge Pile} significantly improve
    
[^138]: SEER: 通过强化学习促进结构化推理和解释

    SEER: Facilitating Structured Reasoning and Explanation via Reinforcement Learning. (arXiv:2401.13246v1 [cs.CL])

    [http://arxiv.org/abs/2401.13246](http://arxiv.org/abs/2401.13246)

    SEER是一种通过最大化基于结构的回报来促进结构化推理和解释的新方法。

    

    阐明从问题到答案的推理过程，通过结构化解释是根本重要的，因为它显著增强了问答系统的解释性和可信度。然而，结构化解释要求模型进行复杂的结构化推理，这带来了巨大的挑战。大多数现有方法集中在通过监督学习进行单步推理，忽视步骤之间的逻辑依赖关系。同时，现有的基于强化学习（RL）的方法忽视了结构化关系，阻碍了RL在结构化推理中的潜力。在本文中，我们提出了一种名为SEER的新方法，通过最大化基于结构的回报，以促进结构化推理和解释。我们提出的基于结构的回报准确描述了结构化推理中固有的分层和分支结构，有效地捕捉了状态之间的复杂关系。我们还引入了一种细粒度的奖励函数。

    Elucidating the reasoning process with structured explanations from question to answer is fundamentally crucial, as it significantly enhances the interpretability and trustworthiness of question-answering (QA) systems. However, structured explanations demand models to perform intricate structured reasoning, which poses great challenges. Most existing methods focus on single-step reasoning through supervised learning, ignoring logical dependencies between steps. Meanwhile, existing reinforcement learning (RL)-based methods overlook the structured relationships, impeding RL's potential in structured reasoning. In this paper, we propose SEER, a novel method that maximizes a structure-based return to facilitate structured reasoning and explanation. Our proposed structure-based return precisely describes the hierarchical and branching structure inherent in structured reasoning, effectively capturing the intricate relationships between states. We also introduce a fine-grained reward function
    
[^139]: 缓解大型语言模型的幻觉问题：通过知识一致性对齐

    Mitigating Hallucinations of Large Language Models via Knowledge Consistent Alignment. (arXiv:2401.10768v1 [cs.CL])

    [http://arxiv.org/abs/2401.10768](http://arxiv.org/abs/2401.10768)

    本文提出了一种称为知识一致性对齐（KCA）的方法，通过减少训练数据中外部知识和预训练语料库中内在知识之间的不一致性，从而缓解了大型语言模型产生幻觉的问题。实验结果表明，KCA方法在多个基准测试中取得了优异的性能。

    

    虽然大型语言模型在对齐后在各种任务上表现出色，但它们仍可能产生与上下文或世界知识自信矛盾的响应，这被称为“幻觉”现象。本文展示了通过减少训练数据中的外部知识与预训练语料库中继承的内在知识之间的不一致性，可以缓解对齐中的幻觉问题。具体而言，我们引入了一种新颖的知识一致性对齐（KCA）方法，该方法通过根据外部知识自动制定考试来评估大型语言模型的理解能力。对于包含知识不一致性的数据，KCA实施了几种简单而高效的处理策略。我们通过使用不同背景和规模的大型语言模型在六个基准测试中展示了所提出的KCA方法在缓解幻觉方面的卓越性能。

    While Large Language Models (LLMs) have proven to be exceptional on a variety of tasks after alignment, they may still produce responses that contradict the context or world knowledge confidently, a phenomenon known as ``hallucination''. In this paper, we demonstrate that reducing the inconsistency between the external knowledge encapsulated in the training data and the intrinsic knowledge inherited in the pretraining corpus could mitigate hallucination in alignment. Specifically, we introduce a novel knowledge consistent alignment (KCA) approach, which involves automatically formulating examinations based on external knowledge for accessing the comprehension of LLMs. For data encompassing knowledge inconsistency, KCA implements several simple yet efficient strategies for processing. We illustrate the superior performance of the proposed KCA approach in mitigating hallucinations across six benchmarks using LLMs of different backbones and scales. Furthermore, we confirm the correlation 
    
[^140]: 批处理ICL: 有效，高效且无序地进行上下文学习

    Batch-ICL: Effective, Efficient, and Order-Agnostic In-Context Learning. (arXiv:2401.06469v1 [cs.LG])

    [http://arxiv.org/abs/2401.06469](http://arxiv.org/abs/2401.06469)

    本文提出了批处理ICL方法，通过将ICL视为一个元优化过程，开发出了一个有效、高效且无序的推理算法。通过聚合元梯度并将其应用于零-shot学习，该方法使LLM对ICL示例顺序无关，并且在实验证明其在大多数情况下优于其他排列方式，甚至超过了标准ICL的最佳顺序的性能。

    

    本文将上下文学习（ICL）视为一个元优化过程，解释了LLM对ICL示例顺序敏感的原因。这种理解使我们开发出了Batch-ICL，一种用于ICL的有效、高效且无序的推理算法。与标准的N-shot学习方法不同，Batch-ICL使用N个单独的1-shot前向计算，并聚合得到的元梯度。然后，将这些聚合的元梯度应用于零-shot学习以生成最终预测。这种批处理方法使LLM对ICL示例的顺序无关。通过大量实验证明，Batch-ICL一致优于大多数示例序列的排列方式。在某些情况下，甚至超过了标准ICL的最佳顺序的性能，同时减少了所需的计算资源。此外，我们还开发了Batch-ICL的一种新颖变体，其中包含多个"epochs"。

    In this paper, by treating in-context learning (ICL) as a meta-optimization process, we explain why LLMs are sensitive to the order of ICL examples. This understanding leads us to the development of Batch-ICL, an effective, efficient, and order-agnostic inference algorithm for ICL. Differing from the standard N-shot learning approach, Batch-ICL employs $N$ separate 1-shot forward computations and aggregates the resulting meta-gradients. These aggregated meta-gradients are then applied to a zero-shot learning to generate the final prediction. This batch processing approach renders the LLM agnostic to the order of ICL examples. Through extensive experiments and analysis, we demonstrate that Batch-ICL consistently outperforms most permutations of example sequences. In some cases, it even exceeds the performance of the optimal order for standard ICL, all while reducing the computational resources required. Furthermore, we develop a novel variant of Batch-ICL featuring multiple "epochs" of 
    
[^141]: 大型语言模型中的通用漏洞：上下文学习后门攻击

    Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks. (arXiv:2401.05949v1 [cs.CL])

    [http://arxiv.org/abs/2401.05949](http://arxiv.org/abs/2401.05949)

    本研究发现上下文学习范式在大型语言模型中存在漏洞，攻击者可以通过污染示范上下文来操控模型行为，而无需进行微调。这项研究设计了一种名为ICLAttack的后门攻击方法，可以通过污染示范样本和提示来使模型按照预定义的意图行事。

    

    上下文学习是一种在预训练和微调之间弥合差距的范式，在几个自然语言处理任务中展现了高效性，特别是在少样本设置中。与传统的微调方法不同，上下文学习能够适应未见过的任务而无需更新任何参数。尽管被广泛应用，上下文学习仍然容易受到恶意攻击。本研究提出了对这一范式的安全性问题的关切。我们的研究表明，攻击者可以通过污染示范上下文来操控大型语言模型的行为，而无需对模型进行微调。具体来说，我们设计了一种新的后门攻击方法，命名为ICLAttack，针对基于上下文学习的大型语言模型。我们的方法包括两种类型的攻击：污染示范样本和污染提示，可以使模型按照预定义的意图行事。ICLAttack不需要额外的微调。

    In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Unlike traditional fine-tuning methods, in-context learning adapts pre-trained models to unseen tasks without updating any parameters. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we have designed a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning prompts, which can make models behave in accordance with predefined intentions. ICLAttack does not require additional fine-tuning 
    
[^142]: AUTOACT：通过自主规划实现的自动代理学习

    AUTOACT: Automatic Agent Learning from Scratch via Self-Planning. (arXiv:2401.05268v1 [cs.CL])

    [http://arxiv.org/abs/2401.05268](http://arxiv.org/abs/2401.05268)

    AUTOACT是一个自动代理学习框架，通过自主规划合成轨迹，不依赖于大规模数据和闭源模型，能够实现更好或类似的性能。

    

    语言代理在各种复杂任务上取得了相当的性能。尽管在这个领域进行了不断的探索，但现有的语言代理系统仍然面临昂贵、不可重复的数据依赖问题，并且面临将单一模型应用于多个功能的挑战。为此，我们介绍了AutoAct，这是一个自动代理学习框架，不依赖于大规模带注释的数据和来自闭源模型（如GPT-4）的合成轨迹。给定有限的数据和工具库，AutoAct首先自动合成规划轨迹，不需要人类或强闭源模型的任何辅助。然后，AutoAct利用分工策略，根据目标任务信息和合成轨迹自动区分，产生一个子代理组来完成任务。我们进行了多种LLMs的广泛实验，结果显示AutoAct在性能上优于或与其相当。

    Language agents have achieved considerable performance on various complex tasks. Despite the incessant exploration in this field, existing language agent systems still struggle with costly, non-reproducible data reliance and face the challenge of compelling a single model for multiple functions. To this end, we introduce AutoAct, an automatic agent learning framework that does not rely on large-scale annotated data and synthetic trajectories from closed-source models (e.g., GPT-4). Given limited data with a tool library, AutoAct first automatically synthesizes planning trajectories without any assistance from humans or strong closed-source models. Then, AutoAct leverages a division-of-labor strategy to automatically differentiate based on the target task information and synthesized trajectories, producing a sub-agent group to complete the task. We conduct comprehensive experiments with different LLMs, which demonstrates that AutoAct yields better or parallel performance compared to var
    
[^143]: 多用户聊天助手（MUCA）：一种使用LLMs框架促进群体对话的方法

    Multi-User Chat Assistant (MUCA): a Framework Using LLMs to Facilitate Group Conversations. (arXiv:2401.04883v1 [cs.CL])

    [http://arxiv.org/abs/2401.04883](http://arxiv.org/abs/2401.04883)

    这篇论文介绍了一种基于大规模语言模型的多用户聊天机器人框架（MUCA），该框架支持群组讨论，并提供了三个主要模块来确定回应内容、时机和适当的接收者。同时，作者还提出了一个基于语言模型的多用户模拟器（MUS），用于模拟真实用户行为，以便更高效地测试和优化聊天机器人。

    

    最近大规模语言模型（LLMs）的进展为聊天机器人的发展提供了新的途径，而大部分现有研究主要集中在单用户的聊天机器人上，重点放在用户输入后决定“回答什么”。在本文中，我们发现多用户聊天机器人有更复杂的3W设计维度——如何回答，“何时”回应，“回答谁”。此外，我们提出了一个名为Multi-User Chat Assistant (MUCA)的基于LLM的聊天机器人框架，专门用于群组讨论。MUCA由三个主要模块组成：子主题生成器，对话分析器和话语策略仲裁器。这些模块共同确定合适的回应内容、时机和适当的接收者。为了使MUCA的优化过程更容易，我们进一步提出了一个基于LLM的多用户模拟器（MUS），可以模拟真实用户行为。这使得聊天机器人和模拟用户之间的对话进行更快速的模拟，从而使得早期测试和优化过程更高效。

    Recent advancements in large language models (LLMs) have provided a new avenue for chatbot development, while most existing research has primarily centered on single-user chatbots that focus on deciding "What" to answer after user inputs. In this paper, we identified that multi-user chatbots have more complex 3W design dimensions -- "What" to say, "When" to respond, and "Who" to answer. Additionally, we proposed Multi-User Chat Assistant (MUCA), which is an LLM-based framework for chatbots specifically designed for group discussions. MUCA consists of three main modules: Sub-topic Generator, Dialog Analyzer, and Utterance Strategies Arbitrator. These modules jointly determine suitable response contents, timings, and the appropriate recipients. To make the optimizing process for MUCA easier, we further propose an LLM-based Multi-User Simulator (MUS) that can mimic real user behavior. This enables faster simulation of a conversation between the chatbot and simulated users, making the earl
    
[^144]: 语言模型是更像图书馆还是图书管理员？Bibliotechnism，小说引用问题和LLM的态度。

    Are Language Models More Like Libraries or Like Librarians? Bibliotechnism, the Novel Reference Problem, and the Attitudes of LLMs. (arXiv:2401.04854v1 [cs.CL])

    [http://arxiv.org/abs/2401.04854](http://arxiv.org/abs/2401.04854)

    本文探讨了语言模型（LLMs）是更像图书馆还是图书管理员的问题。论文首先阐述了 "文献主义 "这一概念，并提出了对其的挑战，指出LLMs生成的全新文本在内容上依赖于原始人类文本的内容。然后，论文提出了对 "文献主义"的新颖挑战，讨论了LLMs生成的 "新引用"问题。最后，根据心灵哲学中的解释主义，论文提出了有限代理能力的LLMs可能存在的可能性。

    

    LLMs（语言模型）是否像复印机或印刷机等文化技术一样，传输信息但无法创建新内容？我们将这个概念称为"文献主义"，它面临一个挑战，即LLMs经常生成全新的文本。我们首先为"文献主义"对抗这个挑战进行辩护，展示了新的文本仅在派生意义上具有意义，因此这些生成的文本的内容在重要意义上依赖于原始人类文本的内容。然后，我们提出了一个不同的、新颖的挑战，即LLMs生成"新引用"的例子，使用新的名称来引用新实体。如果LLMs不是文化技术而是具有有限形式的代理能力（信念、欲望和意图），这样的例子可以很好地解释。根据心灵哲学中的解释主义，仅当一个系统的行为可以通过假设它具有信念、欲望和意图来很好地解释时，它才具有这样的信念、欲望和意图。

    Are LLMs cultural technologies like photocopiers or printing presses, which transmit information but cannot create new content? A challenge for this idea, which we call bibliotechnism, is that LLMs often do generate entirely novel text. We begin by defending bibliotechnism against this challenge, showing how novel text may be meaningful only in a derivative sense, so that the content of this generated text depends in an important sense on the content of original human text. We go on to present a different, novel challenge for bibliotechnism, stemming from examples in which LLMs generate "novel reference", using novel names to refer to novel entities. Such examples could be smoothly explained if LLMs were not cultural technologies but possessed a limited form of agency (beliefs, desires, and intentions). According to interpretationism in the philosophy of mind, a system has beliefs, desires and intentions if and only if its behavior is well-explained by the hypothesis that it has such s
    
[^145]: LLMs的神秘和迷人之处：紧密调查对新兴能力的解释和分析

    The Mystery and Fascination of LLMs: A Comprehensive Survey on the Interpretation and Analysis of Emergent Abilities. (arXiv:2311.00237v1 [cs.CL])

    [http://arxiv.org/abs/2311.00237](http://arxiv.org/abs/2311.00237)

    该论文对LLMs的新兴能力的解释和分析进行了全面调查，旨在理解这些能力的机制和实际应用，并解决可能出现的潜在风险和担忧。

    

    理解新兴能力，如在大型语言模型（LLMs）中的上下文学习(ICL)和思维链(CoT)触发，至关重要。这种重要性不仅来自于在各种任务中更好地利用这些能力，还包括主动识别和缓解可能出现的潜在风险，包括真实性、偏见和有害性的担忧。本文在LLMs的新兴能力解释和分析方面提出了一项深入调查。首先，我们简要介绍了新兴能力的背景和定义。然后，我们从两个角度概述了研究的进展：1)宏观角度，强调对机制可解释性的研究，并深入探讨新兴能力背后的数学基础；2)微观角度，关注通过考察与这些能力相关的因素来实证可解释性的研究。

    Understanding emergent abilities, such as in-context learning (ICL) and chain-of-thought (CoT) prompting in large language models (LLMs), is of utmost importance. This importance stems not only from the better utilization of these capabilities across various tasks, but also from the proactive identification and mitigation of potential risks, including concerns of truthfulness, bias, and toxicity, that may arise alongside these capabilities. In this paper, we present a thorough survey on the interpretation and analysis of emergent abilities of LLMs. First, we provide a concise introduction to the background and definition of emergent abilities. Then, we give an overview of advancements from two perspectives: 1) a macro perspective, emphasizing studies on the mechanistic interpretability and delving into the mathematical foundations behind emergent abilities; and 2) a micro-perspective, concerning studies that focus on empirical interpretability by examining factors associated with these
    
[^146]: InstructCoder: 为代码编辑赋能的语言模型。

    InstructCoder: Empowering Language Models for Code Editing. (arXiv:2310.20329v1 [cs.CL])

    [http://arxiv.org/abs/2310.20329](http://arxiv.org/abs/2310.20329)

    本研究旨在探索使用大型语言模型（LLMs）进行代码编辑，并引入了InstructCoder数据集，该数据集包含多样性的代码编辑任务，为通用代码编辑提供支持。

    

    代码编辑涵盖了开发者日常处理的各种实用任务。尽管其相关性和实用性，但自动代码编辑仍然是深度学习模型演化中尚未充分探索的领域，部分原因是数据稀缺。在本研究中，我们探索了使用大型语言模型（LLMs）根据用户指令编辑代码的方法，涵盖了诸如注释插入，代码优化和代码重构等一系列隐含任务。为了实现这一目标，我们引入了InstructCoder，这是第一个专为通用代码编辑而设计的数据集，包含高多样性的代码编辑任务。该数据集包含超过114,000个指令-输入-输出三元组，并涵盖了多个不同的代码编辑场景。数据集通过一个迭代过程进行系统扩展，该过程从GitHub的提交中获取代码编辑数据作为种子任务。种子任务和生成的任务随后用于提示ChatGPT获取更多任务数据。

    Code editing encompasses a variety of pragmatic tasks that developers deal with daily. Despite its relevance and practical usefulness, automatic code editing remains an underexplored area in the evolution of deep learning models, partly due to data scarcity. In this work, we explore the use of large language models (LLMs) to edit code based on user instructions, covering a broad range of implicit tasks such as comment insertion, code optimization, and code refactoring. To facilitate this, we introduce InstructCoder, the first dataset designed to adapt LLMs for general-purpose code editing, containing highdiversity code-editing tasks. It consists of over 114,000 instruction-input-output triplets and covers multiple distinct code editing scenarios. The dataset is systematically expanded through an iterative process that commences with code editing data sourced from GitHub commits as seed tasks. Seed and generated tasks are used subsequently to prompt ChatGPT for more task data. Our exper
    
[^147]: 并非所有国家都庆祝感恩节：关于大型语言模型中的文化主导问题

    Not All Countries Celebrate Thanksgiving: On the Cultural Dominance in Large Language Models. (arXiv:2310.12481v1 [cs.CL])

    [http://arxiv.org/abs/2310.12481](http://arxiv.org/abs/2310.12481)

    本文研究了大型语言模型中的文化主导问题，发现由于在模型训练中主要使用英语数据，当用户使用非英语语言提问时，模型往往提供与预期文化不相关的不恰当答案。我们提出了通过多样化数据预训练和文化感知提示两种方法来解决这个问题。

    

    本文针对大型语言模型（LLM）中存在的文化主导问题进行了研究，该问题源于在模型训练中主要使用英语数据（例如ChatGPT）。当用户使用非英语语言提问时，LLMs往往会提供与预期文化不相关的不恰当的英语文化相关答案。为了系统评估文化主导问题，我们构建了一个包含具体文化对象（如假日和歌曲）和抽象文化对象（如价值观和观点）的基准测试集。实证结果表明，代表性的GPT模型存在文化主导问题，其中GPT-4受到最严重影响，而text-davinci-003在这个问题上受影响最小。我们的研究强调了在开发和部署过程中对文化主导问题进行批判性审视和伦理考虑的需要。我们展示了两种直接的方法：模型开发中的多样化数据预训练和部署中的文化感知提示，可以显著缓解文化主导问题。

    In this paper, we identify a cultural dominance issue within large language models (LLMs) due to the predominant use of English data in model training (e.g. ChatGPT). LLMs often provide inappropriate English-culture-related answers that are not relevant to the expected culture when users ask in non-English languages. To systematically evaluate the cultural dominance issue, we build a benchmark that consists of both concrete (e.g. holidays and songs) and abstract (e.g. values and opinions) cultural objects. Empirical results show that the representative GPT models suffer from the culture dominance problem, where GPT-4 is the most affected while text-davinci-003 suffers the least from this problem. Our study emphasizes the need for critical examination of cultural dominance and ethical consideration in their development and deployment. We show two straightforward methods in model development (i.e. pretraining on more diverse data) and deployment (e.g. culture-aware prompting) can signifi
    
[^148]: 从不一致到洞察：对案例结果分类的理由数据集构建进行解析

    From Dissonance to Insights: Dissecting Disagreements in Rationale Dataset Construction for Case Outcome Classification. (arXiv:2310.11878v1 [cs.CL])

    [http://arxiv.org/abs/2310.11878](http://arxiv.org/abs/2310.11878)

    本研究关注法律自然语言处理中人工标注的变异问题，通过收集一组律师对案件结果评估存在分歧的数据集，对这些分歧进行了研究，构建了一个两级分类体系，并发现分歧主要源于对法律背景的不明确描述。

    

    在法律自然语言处理中，案例结果分类（COC）不仅需要准确性，还需要可信赖性和可解释性。现有的可解释COC研究仅限于由单个专家进行的注释。然而，众所周知，律师在对案件事实进行评估时可能存在分歧。因此，我们收集了一个新的数据集RAVE：欧洲人权法领域的理由变异，该数据集是从国际人权法领域的两位专家那里获得的，我们观察到他们之间存在弱一致性。我们研究了他们的分歧，并构建了一个两级任务无关的分类体系，同时补充了COC特定的子类别。据我们所知，这是法律自然语言处理领域首次关注人工标注的变异。我们定量评估了不同分类类别，并发现分歧主要源于对法律背景的不明确描述，这在COC元数据通常具有有限细粒度和噪声的情况下带来了挑战。我们进一步评估了SOTA COC模型在RAVE数据集上的可解释性，并观察到...

    In legal NLP, Case Outcome Classification (COC) must not only be accurate but also trustworthy and explainable. Existing work in explainable COC has been limited to annotations by a single expert. However, it is well-known that lawyers may disagree in their assessment of case facts. We hence collect a novel dataset RAVE: Rationale Variation in ECHR1, which is obtained from two experts in the domain of international human rights law, for whom we observe weak agreement. We study their disagreements and build a two-level task-independent taxonomy, supplemented with COC-specific subcategories. To our knowledge, this is the first work in the legal NLP that focuses on human label variation. We quantitatively assess different taxonomy categories and find that disagreements mainly stem from underspecification of the legal context, which poses challenges given the typically limited granularity and noise in COC metadata. We further assess the explainablility of SOTA COC models on RAVE and observ
    
[^149]: (动态)提示可能是修复压缩LLMs所需的全部。(arXiv:2310.00867v2 [cs.CL] UPDATED)

    (Dynamic) Prompting might be all you need to repair Compressed LLMs. (arXiv:2310.00867v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.00867](http://arxiv.org/abs/2310.00867)

    提出了一种动态提示(IDP)的机制，它可以作为一种轻量级的适应工具，修复压缩的大型语言模型(LLMs)在一些实际的下游任务中的性能下降。

    

    大型语言模型(LLMs)在自然语言处理方面有着重大的变革，但同时也带来了显著的计算需求，强调了高效、无需训练的压缩的需求。尽管针对最大的LLMs在无需训练的压缩方面取得了显著的改进，但我们使用LLaMA-7B和OPT-6.7b进行的测试显示，在一些实际的下游任务中存在显著的性能下降。对资源密集型的压缩后重新训练的权衡的调查表明，提示驱动的恢复作为一种轻量级的适应工具具有潜在的前景。然而，现有研究主要局限在困惑度评估和简单任务上，对提示的可扩展性和通用性没有给出明确的信心。我们通过两种关键方法解决了这种不确定性。首先，我们揭示了LLM压缩中天真提示的脆弱性，即过度依赖单一输入的提示。作为回应，我们提出了推理时动态提示(IDP)的机制，它可以自主选择最佳的提示。

    Large language models (LLMs), while transformative for NLP, come with significant computational demands, underlining the need for efficient, training-free compression. Notably, despite the marked improvement in training-free compression for the largest of LLMs, our tests using LLaMA-7B and OPT-6.7b highlight a significant performance drop in several realistic downstream tasks. Investigation into the trade-off between resource-intensive post-compression re-training highlights the prospect of prompt-driven recovery as a lightweight adaption tool. However, existing studies, confined mainly to perplexity evaluations and simple tasks, fail to offer unequivocal confidence in the scalability and generalizability of prompting. We tackle this uncertainty in two key ways. First, we uncover the vulnerability of naive prompts in LLM compression as an over-reliance on a singular prompt per input. In response, we propose inference-time dynamic prompting (IDP), a mechanism that autonomously chooses f
    
[^150]: ToRA：一种集成工具的数学问题求解推理代理

    ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving. (arXiv:2309.17452v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.17452](http://arxiv.org/abs/2309.17452)

    ToRA是一种集成工具的数学问题求解推理代理，通过结合语言的分析能力和工具的计算效率，能够显著提高数学推理的性能，在多个数学推理数据集上取得了13%-19%的平均绝对改进率，并在竞赛级数据集MATH上达到了44.6%的性能。

    

    大型语言模型在各种语言任务中取得了重大进展，但在复杂的数学问题上仍然存在困难。在本文中，我们提出了一系列集成工具的推理代理ToRA，它通过无缝地将自然语言推理与外部工具（例如计算库和符号求解器）的利用相结合，从而将语言的分析能力与工具的计算效率融合在一起，用于解决具有挑战性的数学问题。为了训练ToRA，我们精选了数学数据集上的互动工具使用轨迹，应用模仿学习于注释，并提出输出空间整形来进一步改进模型的推理行为。结果显示，ToRA模型在10个涵盖各种规模的数学推理数据集上显著优于开源模型，平均绝对改进率达到13%至19%。值得注意的是，ToRA-7B 在竞赛级数据集MATH上达到了44.6%，超越了最佳开源模型WizardMath。

    Large language models have made significant progress in various language tasks, yet they still struggle with complex mathematics. In this paper, we propose ToRA a series of Tool-integrated Reasoning Agents designed to solve challenging mathematical problems by seamlessly integrating natural language reasoning with the utilization of external tools (e.g., computation libraries and symbolic solvers), thereby amalgamating the analytical prowess of language and the computational efficiency of tools. To train ToRA, we curate interactive tool-use trajectories on mathematical datasets, apply imitation learning on the annotations, and propose output space shaping to further refine models' reasoning behavior. As a result, ToRA models significantly outperform open-source models on 10 mathematical reasoning datasets across all scales with 13%-19% absolute improvements on average. Notably, ToRA-7B reaches 44.6% on the competition-level dataset MATH, surpassing the best open-source model WizardMath
    
[^151]: 跨越主题、领域和语言变化：对全面的非分布场景进行评估

    Bridging Topic, Domain, and Language Shifts: An Evaluation of Comprehensive Out-of-Distribution Scenarios. (arXiv:2309.08316v1 [cs.CL])

    [http://arxiv.org/abs/2309.08316](http://arxiv.org/abs/2309.08316)

    本论文评估了语言模型在跨越主题、领域和语言变化的全面非分布场景中的泛化能力，并提出了改进策略，包括基于提示的精细调节和上下文学习。

    

    语言模型在独立且同分布的训练和测试数据中表现出色。然而，在实际应用中（如争论挖掘），它们的性能经常下降。这种降级发生在新话题出现，或其他文本领域和语言变得相关的情况下。为了评估语言模型在这些非分布场景中的泛化能力，我们通过有意地保留特定实例进行测试来模拟这种分布变化，例如社交媒体领域或太阳能主题。与先前关注特定变化和度量标准的研究不同，我们全面分析了泛化问题。我们定义了三个度量标准来确定泛化缺陷，并提出了涵盖主题、领域和语言变化的十一个分类任务。总体来说，我们发现基于提示的精细调节具有更卓越的性能，特别是在训练集和测试集在语义上主要有差异的情况下。同时，在上下文学习方面也有类似的发现。

    Language models (LMs) excel in in-distribution (ID) scenarios where train and test data are independent and identically distributed. However, their performance often degrades in real-world applications like argument mining. Such degradation happens when new topics emerge, or other text domains and languages become relevant. To assess LMs' generalization abilities in such out-of-distribution (OOD) scenarios, we simulate such distribution shifts by deliberately withholding specific instances for testing, as from the social media domain or the topic Solar Energy.  Unlike prior studies focusing on specific shifts and metrics in isolation, we comprehensively analyze OOD generalization. We define three metrics to pinpoint generalization flaws and propose eleven classification tasks covering topic, domain, and language shifts. Overall, we find superior performance of prompt-based fine-tuning, notably when train and test splits primarily differ semantically. Simultaneously, in-context learning
    
[^152]: 调查新闻概述中的性别偏见

    Investigating Gender Bias in News Summarization. (arXiv:2309.08047v1 [cs.CL])

    [http://arxiv.org/abs/2309.08047](http://arxiv.org/abs/2309.08047)

    本研究调查了新闻概述中的性别偏见，发现大型语言模型（LLMs）会重复和强化有害的社会偏见。研究提出了一些方法来量化模型中的有偏行为，并提出了一种生成具有控制人口属性的输入文档的方法。

    

    概述是大型语言模型（LLMs）的一个重要应用。以往对概述模型的评估主要关注它们在内容选择、语法正确性和连贯性方面的性能。然而，众所周知，LLMs会重复和强化有害的社会偏见。这引发了一个问题：在一个相对受限制的环境，比如概述，这些偏见会对模型的输出产生影响吗？为了解答这个问题，我们首先提出了一些关于概述模型中的有偏行为的定义，并引入了一些实际方法来量化它们。由于我们发现输入文档中存在的偏见可能干扰我们的分析，我们还提出了一种方法来生成具有仔细控制人口属性的输入文档。这使我们能够规避这个问题，同时仍然使用一些现实的输入文档进行工作。最后，我们将我们的方法应用于专门构建的概述模型和通用用途的模型生成的概述。

    Summarization is an important application of large language models (LLMs). Most previous evaluation of summarization models has focused on their performance in content selection, grammaticality and coherence. However, it is well known that LLMs reproduce and reinforce harmful social biases. This raises the question: Do these biases affect model outputs in a relatively constrained setting like summarization?  To help answer this question, we first motivate and introduce a number of definitions for biased behaviours in summarization models, along with practical measures to quantify them. Since we find biases inherent to the input document can confound our analysis, we additionally propose a method to generate input documents with carefully controlled demographic attributes. This allows us to sidestep this issue, while still working with somewhat realistic input documents.  Finally, we apply our measures to summaries generated by both purpose-built summarization models and general purpose
    
[^153]: 临床文本摘要: 大型语言模型的应用优于人类专家

    Clinical Text Summarization: Adapting Large Language Models Can Outperform Human Experts. (arXiv:2309.07430v1 [cs.CL])

    [http://arxiv.org/abs/2309.07430](http://arxiv.org/abs/2309.07430)

    本研究通过对八个大型语言模型在临床摘要任务上的领域适应方法实验进行了全面的定量评估，发现最佳适应的模型的摘要在完整性和正确性方面优于人类摘要。

    

    在临床工作中，浏览大量的文本数据并总结关键信息对临床医生的时间分配造成了很大的负担。尽管大型语言模型（LLMs）在自然语言处理（NLP）任务中展现了巨大的潜力，但它们在各种临床摘要任务中的效果尚未得到严格的检验。在本研究中，我们对八个LLMs进行了领域适应方法的实验，涵盖了六个数据集和四个不同的摘要任务：放射学报告、患者问题、病历记录和医患对话。我们进行了全面的定量评估，发现模型和适应方法之间存在权衡，并且在某些情况下，LLMs的最新进展可能不会带来改进的结果。此外，通过与六名医生进行的临床阅读者研究，我们发现最佳适应的LLM的摘要在完整性和正确性方面优于人类摘要。我们的进一步定性分析揭示了LLMs和人类在面对的共同挑战。

    Sifting through vast textual data and summarizing key information imposes a substantial burden on how clinicians allocate their time. Although large language models (LLMs) have shown immense promise in natural language processing (NLP) tasks, their efficacy across diverse clinical summarization tasks has not yet been rigorously examined. In this work, we employ domain adaptation methods on eight LLMs, spanning six datasets and four distinct summarization tasks: radiology reports, patient questions, progress notes, and doctor-patient dialogue. Our thorough quantitative assessment reveals trade-offs between models and adaptation methods in addition to instances where recent advances in LLMs may not lead to improved results. Further, in a clinical reader study with six physicians, we depict that summaries from the best adapted LLM are preferable to human summaries in terms of completeness and correctness. Our ensuing qualitative analysis delineates mutual challenges faced by both LLMs and
    
[^154]: 用于自动开放领域科学假设发现的大语言模型

    Large Language Models for Automated Open-domain Scientific Hypotheses Discovery. (arXiv:2309.02726v1 [cs.CL])

    [http://arxiv.org/abs/2309.02726](http://arxiv.org/abs/2309.02726)

    这项研究提出了用于社会科学学术假设发现的第一个自然语言处理数据集，旨在开发一个系统，能够基于原始网络语料库自动生成有效、新颖且对人类研究者有帮助的假设。

    

    当科学家观察世界并试图提出解释这些观察结果的假设时，假设归纳被认为是主要的推理类型。过去关于假设归纳的研究存在以下限制：（1）数据集的观察注释不是原始的网络语料库，而是手动选择的句子（导致了一个封闭领域的设置）；（2）实际的假设注释主要是常识知识，使得任务不太具有挑战性。在本文中，我们提出了第一个用于社会科学学术假设发现的自然语言处理数据集，包含50篇发表在顶级社会科学期刊上的最新论文。数据集中还收集了开发论文中的假设所需的原始网络语料库，最终目标是创建一个系统，仅通过一堆原始网络语料库就可以自动生成有效、新颖且对人类研究者有帮助的假设。这个新数据集可以解决以前关于假设归纳的研究所面临的限制问题。

    Hypothetical induction is recognized as the main reasoning type when scientists make observations about the world and try to propose hypotheses to explain those observations. Past research on hypothetical induction has a limited setting that (1) the observation annotations of the dataset are not raw web corpus but are manually selected sentences (resulting in a close-domain setting); and (2) the ground truth hypotheses annotations are mostly commonsense knowledge, making the task less challenging. In this work, we propose the first NLP dataset for social science academic hypotheses discovery, consisting of 50 recent papers published in top social science journals. Raw web corpora that are necessary for developing hypotheses in the published papers are also collected in the dataset, with the final goal of creating a system that automatically generates valid, novel, and helpful (to human researchers) hypotheses, given only a pile of raw web corpora. The new dataset can tackle the previou
    
[^155]: 广告的长期记忆性研究

    Long-Term Memorability On Advertisements. (arXiv:2309.00378v1 [cs.CL])

    [http://arxiv.org/abs/2309.00378](http://arxiv.org/abs/2309.00378)

    本研究是首个大规模的记忆性研究，发现广告的长期记忆性对于市场营销非常重要，但在机器学习文献中一直缺乏相关研究。通过分析大量参与者和广告，我们得出了关于什么使广告记忆深刻的有趣见解。

    

    市场营销人员花费数十亿美元在广告上，但是投入到广告上的金钱能起多大作用呢？当顾客在购买时无法辨认出他们看过的品牌的话，花在广告上的钱基本上就被浪费了。尽管在营销中很重要，但迄今为止，在机器学习的文献中还没有关于广告记忆力的研究。大多数研究都是对特定内容类型（如物体和动作视频）进行短期回忆（<5分钟）的研究。另一方面，广告行业只关心长期记忆（几个小时或更长时间），而且广告几乎总是高度多模式化，通过不同的形式（文本、图像和视频）来讲故事。基于这一动机，我们进行了首个大规模记忆性研究，共有1203名参与者和2205个广告涵盖了276个品牌。在不同参与者子群体和广告类型上进行统计测试，我们发现了许多有关什么使广告难忘的有趣见解-无论是内容还是

    Marketers spend billions of dollars on advertisements but to what end? At the purchase time, if customers cannot recognize a brand for which they saw an ad, the money spent on the ad is essentially wasted. Despite its importance in marketing, until now, there has been no study on the memorability of ads in the ML literature. Most studies have been conducted on short-term recall (<5 mins) on specific content types like object and action videos. On the other hand, the advertising industry only cares about long-term memorability (a few hours or longer), and advertisements are almost always highly multimodal, depicting a story through its different modalities (text, images, and videos). With this motivation, we conduct the first large scale memorability study consisting of 1203 participants and 2205 ads covering 276 brands. Running statistical tests over different participant subpopulations and ad-types, we find many interesting insights into what makes an ad memorable - both content and h
    
[^156]: 分步解毒语言模型

    Detoxify Language Model Step-by-Step. (arXiv:2308.08295v1 [cs.CL])

    [http://arxiv.org/abs/2308.08295](http://arxiv.org/abs/2308.08295)

    这项研究提出了一种分步解毒语言模型的方法，通过在输入阶段进行解毒处理，并使用无毒提示进行连续生成来保持生成质量。同时，通过设计Detox-Chain来校准LLMs的推理能力，实现了更安全和可靠的生成。

    

    解毒语言模型具有挑战性，因为它要求模型在保持生成能力的同时避免生成有害内容。为了确保生成的安全性，先前的解毒方法通过改变数据分布或在单步骤中从不同方面约束生成来解毒模型。然而，由于语言模型倾向于沿着有毒提示生成，解毒方法的工作方向与之相反，这些方法将大大影响LLM的生成质量，如话语连贯性和语义一致性。为了处理这种冲突，我们将解毒过程分解为不同的子步骤，其中解毒集中在输入阶段，随后的连续生成基于无毒提示。此外，我们还通过设计一个Detox-Chain来校准LLMs的强大推理能力，以有序的方式连接上述子步骤，这使得LLMs可以进行连续的解毒生成。

    Detoxification for LLMs is challenging since it requires models to avoid generating harmful content while maintaining the generation capability. To ensure the safety of generations, previous detoxification methods detoxify the models by changing the data distributions or constraining the generations from different aspects in a single-step manner. However, these approaches will dramatically affect the generation quality of LLMs, e.g., discourse coherence and semantic consistency, since language models tend to generate along the toxic prompt while detoxification methods work in the opposite direction. To handle such a conflict, we decompose the detoxification process into different sub-steps, where the detoxification is concentrated in the input stage and the subsequent continual generation is based on the non-toxic prompt. Besides, we also calibrate the strong reasoning ability of LLMs by designing a Detox-Chain to connect the above sub-steps in an orderly manner, which allows LLMs to d
    
[^157]: 通过编码本知识、自然语言推理和ChatGPT来合成政治零样本关系分类

    Synthesizing Political Zero-Shot Relation Classification via Codebook Knowledge, NLI, and ChatGPT. (arXiv:2308.07876v1 [cs.CL])

    [http://arxiv.org/abs/2308.07876](http://arxiv.org/abs/2308.07876)

    该论文通过利用已建立的注释编码本的知识，探索零样本方法用于政治事件本体关系分类，并介绍一种基于自然语言推理的方法，名为ZSP。ZSP采用了一种树查询框架，提高了解释性、效率和对模式更改的适应性。在细粒度根代码分类上，ZSP的性能明显优于ChatGPT，F1得分提高了40%。

    

    最近的事件编码的监督模型在性能方面远远超过模式匹配方法。然而，它们仅仅依赖于新的注释，忽视了专家数据库中的大量知识，限制了它们在细粒度分类中的适用性。为了解决这些限制，我们通过利用已建立的注释编码本的知识，探索零样本方法用于政治事件本体关系分类。我们的研究涵盖了ChatGPT和一种新颖的基于自然语言推理的方法，名为ZSP。ZSP采用了一种树查询框架，将任务分解为上下文、语态和类别消歧的不同层次。该框架提高了解释性、效率和对模式更改的适应性。通过在我们新策划的数据集上进行大量实验，我们指出了ChatGPT中的不稳定性问题，并突出了ZSP的卓越性能。ZSP在细粒度根代码分类的F1得分上取得了令人印象深刻的提高40%。

    Recent supervised models for event coding vastly outperform pattern-matching methods. However, their reliance solely on new annotations disregards the vast knowledge within expert databases, hindering their applicability to fine-grained classification. To address these limitations, we explore zero-shot approaches for political event ontology relation classification, by leveraging knowledge from established annotation codebooks. Our study encompasses both ChatGPT and a novel natural language inference (NLI) based approach named ZSP. ZSP adopts a tree-query framework that deconstructs the task into context, modality, and class disambiguation levels. This framework improves interpretability, efficiency, and adaptability to schema changes. By conducting extensive experiments on our newly curated datasets, we pinpoint the instability issues within ChatGPT and highlight the superior performance of ZSP. ZSP achieves an impressive 40% improvement in F1 score for fine-grained Rootcode classific
    
[^158]: FLASK: 基于对齐技能集的细粒度语言模型评估

    FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets. (arXiv:2307.10928v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2307.10928](http://arxiv.org/abs/2307.10928)

    FLASK是一种基于对齐技能集的细粒度语言模型评估协议，通过将粗级评分分解为每个指令的技能集级评分，实现了对模型性能的全面视角和提高评估的可靠性。

    

    由于指令需要与人类的价值观进行对齐，并且所需的技能集根据指令而异，因此对大型语言模型（LLMs）进行评估具有挑战性。然而，先前的研究主要集中在粗粒度评估（即基于整体偏好的评估），这限制了可解释性，因为它未考虑需要实例级技能组合的用户指令的特性。在本文中，我们介绍了FLASK（基于对齐技能集的细粒度语言模型评估），这是一种细粒度评估协议，用于人类和模型的评估，它将粗级评分分解为每个指令的技能集级评分。通过实验证明，评估的细粒度对于获得对模型性能的全面视角和提高评估的可靠性至关重要。利用FLASK，我们比较了多个开源和专有LLMs，并观察到高度相关性。

    Evaluation of Large Language Models (LLMs) is challenging because instruction-following necessitates alignment with human values and the required set of skills varies depending on the instruction. However, previous studies have mainly focused on coarse-grained evaluation (i.e. overall preference-based evaluation), which limits interpretability since it does not consider the nature of user instructions that require instance-wise skill composition. In this paper, we introduce FLASK (Fine-grained Language Model Evaluation based on Alignment Skill Sets), a fine-grained evaluation protocol for both human-based and model-based evaluation which decomposes coarse-level scoring to a skill set-level scoring for each instruction. We experimentally observe that the fine-graininess of evaluation is crucial for attaining a holistic view of model performance and increasing the reliability of the evaluation. Using FLASK, we compare multiple open-source and proprietary LLMs and observe a high correlati
    
[^159]: 大型语言模型塑造并受到社会的影响：arXiv出版模式调查

    Large language models shape and are shaped by society: A survey of arXiv publication patterns. (arXiv:2307.10700v1 [cs.DL])

    [http://arxiv.org/abs/2307.10700](http://arxiv.org/abs/2307.10700)

    大型语言模型的论文数量急剧增加，研究重点逐渐转向社会影响。与LLM相关的论文呈现持续增长的趋势，新发表关于LLM的作者更注重应用和社会影响。

    

    大型语言模型的论文数量近年来呈急剧增加，这种变化对科学领域产生了戏剧性的影响，但目前还没有进行详细的文献计量分析。本文分析了CS和Stat arXiv上发布的388K篇论文，并重点关注2023年与2018-2022年之间发表模式的变化。我们分析了LLM论文的比例增加情况，得到了最多关注的与LLM相关的主题，撰写LLM论文的作者，作者的研究主题与背景的相关性，区分高被引用LLM论文的因素，以及国际合作的模式。我们展示了LLM研究越来越关注社会影响：在计算机与社会子arXiv上，与LLM相关的论文比例增加了18倍，新发表关于LLM的作者更倾向于关注应用和社会影响。LLM研究也受到社会动态的影响。

    There has been a steep recent increase in the number of large language model (LLM) papers, producing a dramatic shift in the scientific landscape which remains largely undocumented through bibliometric analysis. Here, we analyze 388K papers posted on the CS and Stat arXivs, focusing on changes in publication patterns in 2023 vs. 2018-2022. We analyze how the proportion of LLM papers is increasing; the LLM-related topics receiving the most attention; the authors writing LLM papers; how authors' research topics correlate with their backgrounds; the factors distinguishing highly cited LLM papers; and the patterns of international collaboration. We show that LLM research increasingly focuses on societal impacts: there has been an 18x increase in the proportion of LLM-related papers on the Computers and Society sub-arXiv, and authors newly publishing on LLMs are more likely to focus on applications and societal impacts than more experienced authors. LLM research is also shaped by social dyn
    
[^160]: 元推理：用于大型语言模型的语义符号解构

    Meta-Reasoning: Semantics-Symbol Deconstruction For Large Language Models. (arXiv:2306.17820v1 [cs.CL])

    [http://arxiv.org/abs/2306.17820](http://arxiv.org/abs/2306.17820)

    本论文提出了一种称为“元推理”的方法，它通过使用语义符号解构的方式，将不同推理问题转化为类似的自然语言表示，以提高大型语言模型的推理能力。

    

    大型语言模型中的符号化方法已经被证明可以有效提高语言模型的推理能力。然而，大多数这些方法依赖于将自然语言映射到更加语法完备且没有歧义的形式语言（例如Python、SQL）。虽然这些方法有效，但它们离开了自然语言本身，偏离了人类思维的习惯，而更多地迎合了计算机的执行思维方式。相反，我们希望从语言学中符号的概念出发来简化自然语言，使得语言模型可以学习不同自然语义中包含的推理问题的常见表达方式和通用解决方案。基于这种考虑，我们提出了“元推理”，它允许语言模型自动完成语义符号的解构，即语义解析，从而最大程度地将某些推理任务的不同问题减少到类似的自然语言表示，从而获得推理的能力。

    Symbolization methods in large language models (LLMs) have been shown effective to improve LLMs' reasoning ability. However, most of these approaches hinge on mapping natural languages to formal languages (e.g., Python, SQL) that are more syntactically complete and free of ambiguity. Although effective, they depart from the natural language itself and deviate from the habits of human thinking, and instead cater more to the execution mindset of computers. In contrast, we hope to simplify natural language by starting from the concept of symbols in linguistics itself, so that LLMs can learn the common formulation and general solution of reasoning problems wrapped in different natural semantics. From this consideration, we propose \textbf{Meta-Reasoning}, which allows LLMs to automatically accomplish semantic-symbol deconstruction, i.e., semantic resolution, to maximally reduce different questions of certain reasoning tasks to similar natural language representation, thus gaining the abili
    
[^161]: 基于跨语言伪标注的无监督自动语音识别

    Unsupervised ASR via Cross-Lingual Pseudo-Labeling. (arXiv:2305.13330v1 [eess.AS])

    [http://arxiv.org/abs/2305.13330](http://arxiv.org/abs/2305.13330)

    本研究提出了一种基于跨语言伪标注的无监督ASR方法，能够使用其他语言中的标注数据来引导新语言的无监督AM。在Common Voice上取得了良好的效果，可以实现18% WER。而且在不同语言的数据集上都优于基线模型。

    

    最近的研究表明，可以仅使用非配对的音频和文本来训练无监督自动语音识别（ASR）系统。现有的无监督ASR方法假定不能使用任何标注数据进行训练。本文认为，即使没有给定语言的任何标注音频，也始终可以使用其他语言中的标注数据。本文展示了如何使用其他语言的字符级声学模型（AM），来引导新语言的无监督AM。 这里，“无监督”意味着没有可用于目标语言的标注音频。本文的方法基于两个关键因素：（i）使用其他语言AM生成“目标”语言的伪标签（PLs）；（ii）使用“目标语言模型”限制这些PLs。我们的方法在Common Voice上非常有效：例如，将英语AM传递到斯瓦希里语可以实现18％的WER。 它还在不同语言的多个数据集上优于基于字符的基线模型。

    Recent work has shown that it is possible to train an $\textit{unsupervised}$ automatic speech recognition (ASR) system using only unpaired audio and text. Existing unsupervised ASR methods assume that no labeled data can be used for training. We argue that even if one does not have any labeled audio for a given language, there is $\textit{always}$ labeled data available for other languages. We show that it is possible to use character-level acoustic models (AMs) from other languages to bootstrap an $\textit{unsupervised}$ AM in a new language. Here, "unsupervised" means no labeled audio is available for the $\textit{target}$ language. Our approach is based on two key ingredients: (i) generating pseudo-labels (PLs) of the $\textit{target}$ language using some $\textit{other}$ language AM and (ii) constraining these PLs with a $\textit{target language model}$. Our approach is effective on Common Voice: e.g. transfer of English AM to Swahili achieves 18% WER. It also outperforms characte
    
[^162]: CRITIC：大型语言模型可以通过工具交互批评进行自我校正

    CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. (arXiv:2305.11738v1 [cs.CL])

    [http://arxiv.org/abs/2305.11738](http://arxiv.org/abs/2305.11738)

    本文提出了一个名为CRITIC的框架，使得大型语言模型可以通过与工具的交互校正自己的错误，从而避免生成出现不一致和问题行为的结果。

    

    近年来，大型语言模型的发展非常引人注目。然而，这些模型有时会出现不一致和问题行为，例如出现幻觉事实，生成有缺陷的代码或创建冒犯和有害的内容。与这些模型不同，人类通常使用外部工具来交叉检查和精炼他们的初步内容，例如使用搜索引擎进行事实检查或使用代码解释器进行调试。受这一观察的启发，我们引入了一个名为CRITIC的框架，允许LLMs（实质上是“黑盒子”）以类似于人类与工具交互的方式验证和逐步修正自己的输出。更具体地说，从初始输出开始，CRITIC与适当的工具交互以评估文本的某些方面，然后根据在此验证过程中获得的反馈修改输出。涉及自由形式问答、数学程序综合和毒性检测的全面评估表明，我们的框架使LLMs能够从错误中学习并纠正自己的错误。

    Recent developments in large language models (LLMs) have been impressive. However, these models sometimes show inconsistencies and problematic behavior, such as hallucinating facts, generating flawed code, or creating offensive and toxic content. Unlike these models, humans typically utilize external tools to cross-check and refine their initial content, like using a search engine for fact-checking, or a code interpreter for debugging. Inspired by this observation, we introduce a framework called CRITIC that allows LLMs, which are essentially "black boxes" to validate and progressively amend their own outputs in a manner similar to human interaction with tools. More specifically, starting with an initial output, CRITIC interacts with appropriate tools to evaluate certain aspects of the text, and then revises the output based on the feedback obtained during this validation process. Comprehensive evaluations involving free-form question answering, mathematical program synthesis, and toxi
    
[^163]: 利用跨模态适配器向预训练语言模型注入多功能高效的视觉知识

    Towards Versatile and Efficient Visual Knowledge Injection into Pre-trained Language Models with Cross-Modal Adapters. (arXiv:2305.07358v1 [cs.CL])

    [http://arxiv.org/abs/2305.07358](http://arxiv.org/abs/2305.07358)

    本文提出了X-adapter插拔式模块，利用多模态视觉语言模型，高效地向预训练语言模型注入视觉知识。

    

    人类通过多模态知识学习语言，然而现有的大多数预训练语言模型（PLMs）仅支持文本预训练。本文提出了插拔式模块X-adapter，它能够根据多模态视觉语言模型（VLMs）的对齐视觉和文本知识，灵活高效地向PLMs注入视觉知识。 X-adapter包含两个子模块V-expert和T-expert，可以根据下游任务激活不同的子模块，来融合VLMs的图像和文本表示。

    Humans learn language via multi-modal knowledge. However, due to the text-only pre-training scheme, most existing pre-trained language models (PLMs) are hindered from the multi-modal information.  To inject visual knowledge into PLMs, existing methods incorporate either the text or image encoder of vision-language models (VLMs) to encode the visual information and update all the original parameters of PLMs for knowledge fusion.  In this paper, we propose a new plug-and-play module, X-adapter, to flexibly leverage the aligned visual and textual knowledge learned in pre-trained VLMs and efficiently inject them into PLMs.  Specifically, we insert X-adapters into PLMs, and only the added parameters are updated during adaptation.  To fully exploit the potential in VLMs, X-adapters consist of two sub-modules, V-expert and T-expert, to fuse VLMs' image and text representations, respectively.  We can opt for activating different sub-modules depending on the downstream tasks.  Experimental resu
    
[^164]: 从自然语言定义中学习多关系双曲词向量

    Multi-Relational Hyperbolic Word Embeddings from Natural Language Definitions. (arXiv:2305.07303v1 [cs.CL])

    [http://arxiv.org/abs/2305.07303](http://arxiv.org/abs/2305.07303)

    本论文提出了一种从自然语言定义中学习多关系双曲词向量的框架，以捕捉由定义所引起的分层和多分辨率结构。

    

    仅使用分布信息的神经词向量一直以来都能为下游任务提供有用的含义表示。然而，现有的方法通常会导致难以解释和控制的表示。相反，自然语言定义具有递归的，自说明的语义结构，可以支持能够保留向量空间中显式概念关系和约束的新型表示学习范 paradigm。本文提出了一个神经符号、多关系框架，通过联合映射定义和定义术语及其相应的语义关系，仅从自然语言定义中学习词向量。通过自动从定义语料库中提取关系，并通过一个翻译目标规范化学习问题，我们将框架专门设定为在双曲空间中捕获由定义引起的分层和多分辨率结构。

    Neural-based word embeddings using solely distributional information have consistently produced useful meaning representations for downstream tasks. However, existing approaches often result in representations that are hard to interpret and control. Natural language definitions, on the other side, possess a recursive, self-explanatory semantic structure that can support novel representation learning paradigms able to preserve explicit conceptual relations and constraints in the vector space.  This paper proposes a neuro-symbolic, multi-relational framework to learn word embeddings exclusively from natural language definitions by jointly mapping defined and defining terms along with their corresponding semantic relations. By automatically extracting the relations from definitions corpora and formalising the learning problem via a translational objective, we specialise the framework in hyperbolic space to capture the hierarchical and multi-resolution structure induced by the definitions.
    
[^165]: 通过可逆神经网络学习解释的非交互语义空间

    Learning Disentangled Semantic Spaces of Explanations via Invertible Neural Networks. (arXiv:2305.01713v1 [cs.CL])

    [http://arxiv.org/abs/2305.01713](http://arxiv.org/abs/2305.01713)

    本文介绍了一种使用可逆神经网络将BERT-GPT2自动编码器的隐藏空间转换为更可分离的语义空间的方法，实验结果表明此方法可以改进模型的可解释性和可控性，并取得了比最先进模型更好的性能表现。

    

    在细化连续空间的句子表征上进行解耦可以在定位明确发生的生成因素的同时，改进可解释性和语义控制，这为基于神经的语言模型赋予了一些符号模型的优势，同时保持其灵活性。 本文提出了一种方法，通过使用可逆神经网络（INN）将BERT-GPT2自动编码器的隐藏空间转换为更可分离的语义空间来解除编码的隐藏空间。实验结果表明，与最新的最先进模型相比，INN能够将分布式隐藏空间转换为更好的语义上解耦的潜在空间，从而产生更好的可解释性和可控性。

    Disentangling sentence representations over continuous spaces can be a critical process in improving interpretability and semantic control by localising explicit generative factors. Such process confers to neural-based language models some of the advantages that are characteristic of symbolic models, while keeping their flexibility. This work presents a methodology for disentangling the hidden space of a BERT-GPT2 autoencoder by transforming it into a more separable semantic space with the support of a flow-based invertible neural network (INN). Experimental results indicate that the INN can transform the distributed hidden space into a better semantically disentangled latent space, resulting in better interpretability and controllability, when compared to recent state-of-the-art models.
    
[^166]: 人类和机器学习模型的分词可追溯性：一个注释研究

    Tokenization Tractability for Human and Machine Learning Model: An Annotation Study. (arXiv:2304.10813v1 [cs.CL])

    [http://arxiv.org/abs/2304.10813](http://arxiv.org/abs/2304.10813)

    研究比较了六种分词方法，并发现人类可追溯的分词与机器学习模型中的分词不一定相同。

    

    人类可追溯的分词对于机器学习模型是否也是可追溯的？本研究探讨了人类可追溯的分词（如适当性和可读性）与机器学习模型中的分词（如在NLP任务中的性能）之间的关系。我们在日语常识问答数据集（JGLUE的JCommmonsenseQA）中比较了六种分词方法。我们使用不同的分词器对问答数据集中的问题文本进行分词，并比较了人类标注者和机器学习模型的性能。此外，我们分析了性能、分词的适当性和回答问题的响应时间之间的关系。本文提供了一个定量调查结果，显示出对于人类和机器学习模型来说，可追溯的分词不一定相同。

    Is tractable tokenization for humans also tractable for machine learning models? This study investigates relations between tractable tokenization for humans (e.g., appropriateness and readability) and one for models of machine learning (e.g., performance on an NLP task). We compared six tokenization methods on the Japanese commonsense question-answering dataset (JCommmonsenseQA in JGLUE). We tokenized question texts of the QA dataset with different tokenizers and compared the performance of human annotators and machine-learning models. Besides,we analyze relationships among the performance, appropriateness of tokenization, and response time to questions. This paper provides a quantitative investigation result that shows the tractable tokenizations for humans and machine learning models are not necessarily the same as each other.
    
[^167]: 自然语言作为知识表示的逻辑推理研究：综述

    Logical Reasoning over Natural Language as Knowledge Representation: A Survey. (arXiv:2303.12023v1 [cs.CL])

    [http://arxiv.org/abs/2303.12023](http://arxiv.org/abs/2303.12023)

    本文总结了一种新的逻辑推理方法，它使用自然语言作为知识表示，具有不同于端到端神经方法的优势。这种新模式在未来有着很高的潜力。

    

    逻辑推理是人类认知和智能的核心。以往的人工智能中的逻辑推理研究使用形式化语言作为知识表示（和符号推理器）。然而，使用形式化语言进行推理证明具有困难（例如脆弱性和知识获取瓶颈）。本文总结了一种新的逻辑推理方法的综合概述，它使用自然语言作为知识表示（以及预训练语言模型作为推理器），包括逻辑推理的哲学定义和分类，新模式的优势、基准和方法，未来需要的任务和方法以及与相关 NLP 领域的关系。这种新模式是很有前途的，因为它不仅可以缓解形式化表示的许多挑战，而且也具有优于端到端神经方法的优势。

    Logical reasoning is central to human cognition and intelligence. Past research of logical reasoning within AI uses formal language as knowledge representation~(and symbolic reasoners). However, reasoning with formal language has proved challenging~(e.g., brittleness and knowledge-acquisition bottleneck). This paper provides a comprehensive overview on a new paradigm of logical reasoning, which uses natural language as knowledge representation~(and pretrained language models as reasoners), including philosophical definition and categorization of logical reasoning, advantages of the new paradigm, benchmarks and methods, challenges of the new paradigm, desirable tasks & methods in the future, and relation to related NLP fields. This new paradigm is promising since it not only alleviates many challenges of formal representation but also has advantages over end-to-end neural methods.
    

