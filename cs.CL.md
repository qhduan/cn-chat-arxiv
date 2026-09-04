# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Compile by Training: Turning Natural-Language Specifications into Local Neural Functions](https://arxiv.org/abs/2609.04199) | 提出“训练式编译”方法，将自然语言规范编译为可复用的本地神经函数，通过教师模型生成的示例训练小型适配器，无需每次调用远程大模型即可达到83.6%的语义准确率。 |
| [^2] | [ESPO: Error-Structured Prompt Optimization via Diagnose, Diversify, and Stabilize](https://arxiv.org/abs/2609.04197) | ESPO通过诊断错误模式、多样化候选生成和稳定性选择三个阶段，解决了进化式提示优化中的提示膨胀问题，在七个NLP基准上平均准确率超越GEPA达3.76个百分点，同时提示词更短47%且推理更快。 |
| [^3] | [Legibility is Not Interpretability: Comparing Judged and Actual Importance in Chain-Of-Thought Reasoning](https://arxiv.org/abs/2609.04194) | 本研究将思维链推理步骤的重要性量化为蒙特卡洛模拟估计的“优势”，发现LLM评判器虽能超越简单基线但远不足以准确识别真正重要的推理步骤，表明推理文本的可读性并不等于可解释性。 |
| [^4] | [Knowledge Acquisition During Pre-training? Large Language Models Learn Better With Auxiliary Views](https://arxiv.org/abs/2609.04180) | 研究发现，在预训练中将token预算从文档重复转移到辅助视图（知识的重新表述）能提升大语言模型的学习效果，即使对事实回忆也有效，且不依赖教师模型的强弱。 |
| [^5] | [Last Translation Benchmark](https://arxiv.org/abs/2609.04173) | 提出了终极翻译基准测试，这是一个包含人工编写、经同行评审的多模态示例的基准数据集，通过为每个示例配备手工编写的验证规则来描述具体失败案例，解决了现有机器翻译基准趋于饱和且评估方法不可靠的问题。 |
| [^6] | [Rethinking On-Policy Distillation of Large Language Models II: One Training Example](https://arxiv.org/abs/2609.04172) | 该研究发现仅用一个训练样本进行在线策略蒸馏就能持续改进并达到全数据训练的大部分性能，原因是单个查询即可覆盖全数据训练71.5%的状态，而16个语义不同的查询可达到98.9%的覆盖率并完全匹配全数据训练的效果。 |
| [^7] | [Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments](https://arxiv.org/abs/2609.04148) | Terminal-Universe 通过重放智能体轨迹中记录的文件操作，直接从已有轨迹重建可执行的终端环境，将海量积累的轨迹转化为可复用、可扩展的环境，用于合成新任务并提供执行反馈，解决了智能体后训练中环境稀缺的问题。 |
| [^8] | [Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR](https://arxiv.org/abs/2609.04108) | 先蒸馏后强化学习的两阶段训练方案在推理任务上持续优于纯OPD、纯RLVR及所有联合优化方法，因为OPD先扩大学生对教师解的覆盖范围、RL再在其内锐化，而联合训练会导致两种信号相互干扰。 |
| [^9] | [CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation](https://arxiv.org/abs/2609.04083) | CORE通过将交叉注意力重排序器的细粒度组合判断以列表式Rank-KL目标蒸馏到嵌入模型中，显著提升了MLLM嵌入模型的组合推理能力，其效果优于对比学习和CoSENT。 |
| [^10] | [When Models Edit Too Much: On the Fidelity of Minimal Code Edits](https://arxiv.org/abs/2609.04061) | 该研究揭示了前沿大语言模型在修复代码时普遍存在“过度编辑”问题（即使如GPT-5.5这样的强模型也不例外），并提出通过一条简单的保留指令即可显著减少不必要的代码改动、降低认知复杂度，同时提升修复准确率。 |
| [^11] | [Translation as a Decision Space: A Multi-Agent Perspective on Low-Resource Dialect Generation](https://arxiv.org/abs/2609.04048) | 本文将翻译重构为由多个自主智能体探索的结构化决策空间，把不同翻译路径建模为智能体，并将智能体间的分歧作为可解释的行为信号，应用于土耳其语—叙利亚阿拉伯语这一低资源方言翻译的实证研究。 |
| [^12] | [The Dice Roll Method: A Standardized Protocol for Repeated-Query Auditing of Large Language Model Brand Recommendations](https://arxiv.org/abs/2609.04047) | 本文提出并形式化了“骰子法”——一个基于温度缩放核采样生成模型的可复用标准化协议，通过将响应方差分解为多个成分并提供完整的统计分析技术栈，为大语言模型品牌推荐的重复查询审计建立了系统方法。 |
| [^13] | [Editable Visual Design](https://arxiv.org/abs/2609.04034) | 该论文提出“可编辑的视觉设计”新范式，以编码智能体为核心，将VLM作为“创意大脑”进行需求理解与审美判断，将图像生成模型作为按需的“视觉世界模拟器”合成独立资产，并通过“先想象、后行动”的闭环工作流编写原生HTML/CSS，实现支持图层级精确后编辑的视觉设计。 |
| [^14] | [Instruction Duplication as an Inference-Time Control Primitive](https://arxiv.org/abs/2609.04024) | 在推理时仅简单复制一遍程序化指令——无需重新训练或修改解码——即可将七个模型在医学选择题上通过全部八项诊断测试的比例从90.22%提升至93.17%，同时保持最终答案准确率不变。 |
| [^15] | [Representational alignment yields generalizable safety in language models](https://arxiv.org/abs/2609.04022) | 提出表征相似性优化方法，将大语言模型的内部潜在表征与人类道德判断的原型化归类结构直接对齐，从而使安全对齐能够泛化到以陌生或对抗性形式表述的有害意图。 |
| [^16] | [Alignment-Free Text-Audiobox for Voice Dubbing and Full-Duplex Dialogue Synthesis](https://arxiv.org/abs/2609.03992) | 该论文提出免对齐Text-Audiobox（Text-AB），通过DAC-VAE潜在扩散表示（压缩率较EnCodec提升10倍以上）、基于交叉注意力的免对齐文本建模，以及3B参数与48万小时语音的大规模训练，构建了一个支持语音配音与全双工对话合成的统一框架。 |
| [^17] | [IchthyoNoma: Nomenclature and Context Sensitivity of Zero-Shot Biological Vision--Language Models for Bangladeshi Freshwater Fish Recognition](https://arxiv.org/abs/2609.03985) | 该研究审计了零样本生物视觉-语言模型在孟加拉国淡水鱼识别上的表现，发现其准确率不仅源于生物学专业化知识，还强烈依赖于命名语言（英文/学名/孟加拉语）的选择与图像上下文，其中BioCLIP2在英文提示下表现优异，但孟加拉语提示下接近随机水平。 |
| [^18] | [Investigating the Ability of Large Language Models to Analyze Recipes for Diabetes](https://arxiv.org/abs/2609.03967) | 本研究构建了包含7607个食谱的糖尿病基准数据集，并采用三种融合医学饮食指南的提示策略，系统评估了大型语言模型分析食谱对糖尿病适用性的能力。 |
| [^19] | [FiMI Banking: A Sovereign Model for Indian Retail Banking](https://arxiv.org/abs/2609.03960) | 该论文构建了面向印度零售银行业、基于真实银行文档和工具的受控对话环境FiMI Banking，并通过偏好优化和可验证奖励强化学习两种后训练方法，分别将安全拒绝行为从52%提升至80%、边缘案例性能从0.509提升至0.718。 |
| [^20] | [Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code LLMs](https://arxiv.org/abs/2609.03955) | 该论文提出了一种两阶段强化学习框架TCS，第一阶段生成与参考解一致的可靠测试用例，第二阶段学习针对模型当前失败模式的对抗性反例测试，从而有效提升代码大模型的测试生成质量和代码性能。 |
| [^21] | [Beyond Majority Vote: Multi-Perspective Adjudication for Medical Hallucination Detection](https://arxiv.org/abs/2609.03953) | 该研究提出一种结合首轮标注、LLM裁判候选发现和双重裁定的多视角医学幻觉检测标注框架，证明单一方法均会遗漏事实错误，而多来源裁定可提升医学事实核查基准的完整性。 |
| [^22] | [VestigeKV: The NoPE-MLA KV Cache Carries Its Own Eviction Signal in a Vestigial Branch](https://arxiv.org/abs/2609.03949) | VestigeKV发现NoPE MLA模型KV缓存中的64维解耦RoPE残余分支已被训练重新利用为显著性信号，据此提出无需训练和量化的查询无关缓存淘汰方法，在8-32倍压缩下几乎不损失检索精度。 |
| [^23] | [More Criticism Does Not Make a Better Review: EquiReview-R](https://arxiv.org/abs/2609.03943) | 该论文提出EquiReview-R框架，将AI辅助审稿重构为基于证据的结构化关注点集细化过程，把遗漏与过度批评视为两种独立风险分别纠正，并借助证据关联的轨迹语料库证明审稿修订必须先于进一步的问题搜索。 |
| [^24] | [Headroom-Drift Replay: A Primitive for Principled Replay Control in GRPO](https://arxiv.org/abs/2609.03941) | 该论文提出了一种面向GRPO的组级重放控制原语Headroom-Drift Replay，通过Headroom按剩余学习价值排序、Drift按策略兼容性门控来复用历史轨迹，在不改变在线数据流、不增加额外训练机制的前提下加速RL后训练，从而将重放本身的贡献与复杂训练流程解耦。 |
| [^25] | [Fixed Suffix Dependency Ratio: Quantifying the Dual-Track Mechanism of Gender Assignment in Latvian Loanwords](https://arxiv.org/abs/2609.03930) | 本研究提出固定后缀依赖比率（FSDR）这一量化指标，揭示了拉脱维亚语英语外来词性属分配的双轨机制——阴性外来词显著依赖固定派生后缀，而阳性外来词集中于自由选择区域。 |
| [^26] | [Speak for Me: Giving LLMs the Situational Awareness to Participate in a Meeting](https://arxiv.org/abs/2609.03923) | 提出CAPA架构，通过感知器、预测器、控制器、生成器和重校准器的协作设计，赋予LLM追踪会议立场、话题覆盖和发言权的情境感知能力，解决其在代理缺席者参会时51.4%发言机会保持沉默的问题。 |
| [^27] | [RuleMem: Active Rule Memory for Long-Term Conversational Agents](https://arxiv.org/abs/2609.03915) | RuleMem提出了一种基于规则的主动记忆框架，通过从历史对话中归纳并验证自然语言霍恩子句来主动指导证据检索与推理，显著提升了长期对话问答代理的可靠性。 |
| [^28] | [CROCODIL: Cross-Model Code Editing with LLMs](https://arxiv.org/abs/2609.03894) | 论文发现大语言模型在编辑其他模型生成的陌生代码时会产生过多且过度的改动，为此提出了CROCODIL后训练框架，通过相似性奖励惩罚大幅改动并结合执行验证，在保证功能正确性的同时有效减少跨模型代码编辑中的过度修改。 |
| [^29] | [Beyond Shallow Alignment: How Post-Training Methods Determine Refusal Circuits And Steering Robustness](https://arxiv.org/abs/2609.03887) | 该研究发现训练后方法（尤其是推理增强微调）会根本性地改变语言模型内部拒绝有害请求的计算方式，但没有任何一种现有方法能同时实现鲁棒、不损失通用能力且可稳定转向的安全对齐。 |
| [^30] | [Flip, Don't Shuffle: Watermarking LLMs at the Speed of Inference](https://arxiv.org/abs/2609.03844) | 提出无状态伯努利水印（SBW），通过每词元独立伯努利试验实现O(1)复杂度的绿名单判断，检测速度比KGW自盐值快6000倍以上、比SynthID快2倍，同时保持相同的N(0,1)统计检测保证。 |
| [^31] | [Select, Compress, Reinvest: A Controlled Study of Visual-Token Allocation in Long-Video MLLMs](https://arxiv.org/abs/2609.03820) | 本文通过严格控制变量的受控实验发现，在长视频多模态大语言模型的视觉令牌分配中，帧选择是影响性能的最大单一因素——八个基于查询选择的帧可超越十六个均匀采样的帧，且经典的正交匹配追踪算法即可媲美各类专门设计的选择器。 |
| [^32] | [Evaluating Criterion-Conditioned Behaviour of Large Language Models in Content Moderation](https://arxiv.org/abs/2609.03814) | 提出DECO诊断评估框架，通过标准无关的内容分解与成对评估方法，揭示LLM在内容审核基准上的优异表现可能掩盖其在具体审核标准层面的大量失败。 |
| [^33] | [VisCAD: A Foundation Model Suite with Multimodal Industrial CAD Intelligence](https://arxiv.org/abs/2609.03811) | VisCAD是一个面向工业CAD的基础模型套件，其核心270亿参数模型VisCAD-M1能够将渲染图、文本、2D图纸和真实照片等多种输入转换为可执行的CAD程序，在保证广泛泛化能力的同时具备强大的专业CAD设计与装配生成能力。 |
| [^34] | [Transfiver: Human-AI Co-Inference through a Shared Editable State](https://arxiv.org/abs/2609.03797) | Transfiver 提出了一种人机协同推理架构，将交互信息维护在模型与人类共同更新的单一共享持久状态中，通过隐式流式更新与显式定向编辑两种机制，使人类的修正能够直接改变后续计算所读取的状态。 |
| [^35] | [A Reverse Sign Language Dictionary: Open-Vocabulary Sign Recognition from Continuous Signing via Video Captioning and Description Retrieval](https://arxiv.org/abs/2609.03788) | 该论文提出一种“反向手语词典”方法：先用视觉-语言模型将连续手语片段生成自由形式的动作过程描述，再用多语言句子编码器从目标描述库中检索最匹配条目，从而实现无需词汇标注监督、支持开放词汇的手语识别。 |
| [^36] | [IndicSafeEval: Safety Robustness of Large Language Models under Multilingual Persuasive Jailbreak Attacks](https://arxiv.org/abs/2609.03781) | 该论文提出了IndicSafeEval框架，通过四种印度语言、十个安全类别和六种说服策略构建7,200条对抗性提示，系统评估并揭示了大语言模型在面对多语言说服性越狱攻击时安全表现存在显著差异。 |
| [^37] | [Typological Feature Prediction with Large Language Models: An In-Context Learning Approach](https://arxiv.org/abs/2609.03775) | 大语言模型通过结合系统发育和地理邻近语言证据的上下文学习方法，显著优于基线方法进行类型学特征预测，且对低资源语言同样有效，其预测依据与证据保持一致，为实现可解释的类型学特征预测提供了新途径。 |
| [^38] | [RealCADBench: Benchmarking Parametric CAD Modeling from Industrial Design Intents](https://arxiv.org/abs/2609.03773) | 提出了RealCADBench基准测试，基于19个工厂自动化类别的真实工业设计意图，通过文本、图纸、图片等多种输入模态和可执行性、IoU、视觉语义一致性等综合评估指标，系统性地评估从设计意图到程序化参数CAD建模的能力。 |
| [^39] | [OBER+: Continuity-Aware Reporting and Traceable Continuous Improvement in Outcome-Based Education](https://arxiv.org/abs/2609.03770) | OBER+通过五个相互衔接的阶段将测得的成果差距转化为可追溯且经评估的纠正措施，并引入连续性规则以避免在成果表述变更时误读达成度趋势，从而在成果导向教育中实现从测量到改进的闭环。 |
| [^40] | [Rent-a-RAG: Embedding-Space Watermarks for Auditing Third-Party RAG](https://arxiv.org/abs/2609.03749) | 提出DirBucket框架，通过让文档改写版本的嵌入偏向提供方专属的秘密方向来嵌入语义水印，从而在保持检索效用的同时，实现对第三方RAG中无偿复用文档的黑盒审计检测。 |
| [^41] | [KnowVis: Knowledge-Centric Visual Summarization for Video Lectures](https://arxiv.org/abs/2609.03742) | 提出 KnowVis 框架，通过从多模态视频内容中提取概念图、构建结构化知识单元并合成视觉摘要，将线性视频讲座转化为符合教学规律的视觉叙事，从而降低初学者的认知负担。 |
| [^42] | [Beyond BLEU: A Case for Redefining Sign Language Translation Benchmarks](https://arxiv.org/abs/2609.03734) | 本文证明BLEU-4的提升并不等同于更强的手语理解能力，并提出了一种基于开放权重LLM问答协议的新型评估方法，该方法更符合人类排名、对改写更不敏感且对训练-测试重叠更加鲁棒。 |
| [^43] | [Opening mind by opening architecture: analysis strategies](https://arxiv.org/abs/2609.03719) | 论文指出电声作曲领域封闭架构音频处理器的日益主导使内部处理过程沦为不可洞察的“黑箱”，并提出通过文献分析策略重新审视信号处理技术的实现历程及其美学意义。 |
| [^44] | [What Do CAE Simulation Agents Really Need Beyond a Generic Harness?](https://arxiv.org/abs/2609.03718) | 该研究发现在信息访问和修复预算相同的条件下，单智能体通用框架在CAE仿真任务上能匹敌甚至超越专门设计的多智能体系统，其关键在于框架已内置的执行反馈修复机制（使FoamBench成绩从71.8%提升至96.4%），表明现代LLM框架已有的能力足以替代CAE专用的复杂机制。 |
| [^45] | [A Circuit for Plural Reference: How LLMs Represent and Retrieve Singular and Plural Entities](https://arxiv.org/abs/2609.03687) | 该论文结合机制可解释性与因果干预技术，首次揭示了大语言模型处理复数指代的完整电路机制，发现了一组分别负责表示共指信息、识别复数实体和传递信息的注意力头，并证明LLM对本体论相似实体的复数代词偏好与人类一致。 |
| [^46] | [Understanding Autonomous Driving Datasets by Describing Differences between Image Subsets in Natural Language](https://arxiv.org/abs/2609.03677) | 本文提出集合差异描述方法，利用自然语言自动描述自动驾驶数据集中不同图像子集之间的差异，通过基于目标检测的对象中心分析实现对数据集组成和域偏移的可解释理解。 |
| [^47] | [Enhancing Financial Question Answering: A Novel Benchmark Dataset of Banks' financial statements](https://arxiv.org/abs/2609.03654) | 该论文提出了首个针对跨机构银行财务报表检索的金融问答基准数据集 FinRAG-QA，包含999个从业者整理的问题和24家欧美大型银行的209份超长报告，并系统评估了多阶段 RAG 流水线中各组件的贡献。 |
| [^48] | [The Impact of Synthetic Data Augmentation on Discourse-Pragmatic Function Classification](https://arxiv.org/abs/2609.03652) | 本研究在话语-语用功能分类任务中保持合成样本数量恒定，通过改变其相对于决策边界的几何位置，揭示了合成数据增强的效果取决于合成样本与真实训练数据的几何关系而非仅仅是数量。 |
| [^49] | [</think> Doesn't Stop Reasoning: Analysis of Spurious CoT Termination](https://arxiv.org/abs/2609.03633) | 本文发现提前退出方法中注入的思考结束标记</think>并不总能真正终止推理，模型会在回答阶段继续产生类似推理的“虚假CoT终止”行为，且该延续片段的长度与提前退出节省的推理token数成正比，其根源可能是模型对注入EoT的注意力不足。 |
| [^50] | [Remember and Reweight: Enhancing Multi-Agent Debate with Experience Memory and Confidence Estimation](https://arxiv.org/abs/2609.03619) | 该论文提出R²-MAD框架，通过为多智能体辩论引入经验记忆机制，利用辩论状态感知的检索策略动态校准概念先验，并结合置信度估计对回答进行重加权，有效缓解了多数智能体收敛于错误答案时“共享误解”被放大的关键缺陷。 |
| [^51] | [KhatianDoc: A Human-Verified Benchmark Diagnosing Multimodal LLM Failure on Bengali Legal Land Records](https://arxiv.org/abs/2609.03597) | 本文提出了KhatianDoc基准，首次通过符号识别、进制转换、字段提取和法律问答四项任务，系统评估多模态大语言模型读取孟加拉国手写土地记录中独特的十六进制分数系统的能力，并揭示其在该领域的失败表现。 |
| [^52] | [How Far Can Synthetic Data Take Thai OCR?](https://arxiv.org/abs/2609.03595) | 通过受控文档重建流水线解耦合成数据“真实性”中的各项因素，发现字体多样性、二维结构和真实手写字形是合成数据迁移到真实泰文OCR的关键，并据此构建了无需真实OCR标签的泰文OCR模型Wayu-Paxa-OCR-Zero。 |
| [^53] | [HalluPeer: A Taxonomy-driven Benchmark for Detecting Hallucinations in Scientific Peer Reviews](https://arxiv.org/abs/2609.03580) | 该论文提出了HalluPeer——首个面向科学同行评审场景的幻觉检测基准，通过构建论文、真实评审与注入幻觉评审的对齐数据集以及同行评审专属的幻觉分类体系，揭示了现有检测器难以区分幻觉与合理批评的局限。 |
| [^54] | [Language, Language Models, and What We're Talking About](https://arxiv.org/abs/2609.03577) | 本文以意大利语语言模型为例，主张必须首先区分“作为技术产品的语言模型”与“作为语言研究工具的语言模型”，才能回答“我们究竟希望语言模型生成什么样的语言”这一核心问题。 |
| [^55] | [Lost in Reordering: Structural Sensitivity of Multilingual LLMs under Semantics-Preserving Perturbations](https://arxiv.org/abs/2609.03511) | 该研究提出了基于 GSM8K 构建的基准数据集 IndicReStruct，通过对印地语和马拉雅拉姆语施加成分重排与语态转换等语义保持扰动，揭示了多语言大语言模型在结构变化下数学推理能力会出现持续且显著的退化。 |
| [^56] | [Building and Evaluating Fixed-Voice Thai TTS from Synthetic Speech](https://arxiv.org/abs/2609.03502) | 该论文提出将大型声音克隆模型作为可编程数据源，仅凭15秒声音参考生成合成语音来训练紧凑的固定音色泰语TTS学生模型，并系统研究了文本准备、质量过滤、拒绝采样和前端选择等流水线设计对模型效果的影响及教师模型残留的局限性。 |
| [^57] | [Pattern Over-Generalization of Knowledge Graph Embedding](https://arxiv.org/abs/2609.03487) | 本文揭示了知识图谱嵌入中的模式过度泛化问题，并提出PogRE方法，通过稠密线性变换和复合操作进行关系表示，有效缓解该问题。 |
| [^58] | [When Users Don't Ask: Benchmarking Context-Driven Memory Retrieval in Conversational Agents](https://arxiv.org/abs/2609.03467) | 该论文提出了对话式记忆基准LOCOMO-CONV，通过对话式、隐含式、反事实和组合式四种查询风格，揭示了问答式基准测试所忽视的记忆检索差距，并发现强检索能力并不完全等同于高质量的对话响应。 |
| [^59] | [When Retrieval Helps: Selective Retrieval for Single-Turn Mental-Health QA](https://arxiv.org/abs/2609.03454) | 该研究针对单轮心理健康问答提出一种轻量级选择性检索策略，通过心理教育需求、应对需求、回答具体性三个效用维度及安全触发机制判断检索的必要性，从而在发挥检索增强优势的同时避免其负面影响。 |
| [^60] | [Plan Pointers and Record-Directive Form in Budgeted Verification of Inherited Agent Memory](https://arxiv.org/abs/2609.03450) | 该论文通过十二项注册研究发现，写入智能体记忆库的指令形式（准则、裸ID或指针）会以高度模型依赖的方式显著影响预算受限下的记录选择，长度匹配准则可带来35分的提升，但附加ID可能完全抵消准则的效果。 |
| [^61] | [It's the Problem, Not the Path: Budget and Difficulty Confounds in LLM Reasoning Trajectories](https://arxiv.org/abs/2609.03436) | 该研究提出重启控制的截断探针方法，证明大语言模型推理轨迹中所谓的“突破时刻”和“早期注定失败”大多是预算与难度造成的混淆——178个问题-模型组合中仅1个真正存在前缀特有价值，且在同等token预算下延续自身推理前缀几乎总是优于从头重启。 |
| [^62] | [Decoupled Analysis-Judging: An Automated Creativity Evaluator Using LLMs in Complex Multi-step Creativity Tasks](https://arxiv.org/abs/2609.03432) | 该论文提出CreaEval，通过将“大模型评判者”解耦为记忆增强的结构化分析与基于证据的评判两个阶段，有效缓解冗长偏差与宽容偏差，实现复杂多步骤创造力任务的自动化可靠评估。 |
| [^63] | [Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning](https://arxiv.org/abs/2609.03430) | 该论文发现KV缓存驱逐中复杂的token重要性打分信号几乎毫无贡献，提出完全不计算分数、在每个注意力头内均匀随机驱逐的“随机注意力”方法，其性能匹配最强基线的同时吞吐量提升32-43%，关键在于提示才是需要保留的脆弱部分，而推理轨迹凭借自身冗余性可自我保护。 |
| [^64] | [Lngram v2: Latent N-Gram Memory with Interpretable Discrete Representations](https://arxiv.org/abs/2609.03426) | Lngram v2通过解耦记忆容量与骨干网络宽度、引入上下文感知的分组查询注意力读取机制以及零值Sink和反事实代理梯度等改进，在保留硬离散寻址的同时实现了记忆容量的独立扩展，并成功应用于300亿参数的视觉-语言模型。 |
| [^65] | [To What Extent Do Large Language Models Understand Bangla Idioms?](https://arxiv.org/abs/2609.03410) | 本文构建了首个大规模孟加拉语习语基准数据集，并通过释义改写、跨度检测和意义识别三项任务的评估发现，没有任何大语言模型能在所有任务上全面领先，不同模型各有所长。 |
| [^66] | [TabScope: Question-Adaptive Scope Selection for Table Question Answering](https://arxiv.org/abs/2609.03395) | 该论文提出TabScope框架，通过操作感知的表格分解和问题类型预测，在大语言模型表格问答中动态选择局部子表推理或全表推理，显著提升长表格问答的准确率，并贡献了基于真实世界长表格的SLQA评测基准。 |
| [^67] | [Chiaroscuro for Emotions: A Contrastive Emotion Benchmark Grounded in Appraisal Theory](https://arxiv.org/abs/2609.03394) | 提出了基于评价理论的情绪对比基准CHIARO（1,000句人工标注，每个场景中同一事件引发两人相反情绪），评测发现最强LLM仅达67.3宏F1、现有情绪分类器接近随机水平，且该基准还可作为训练信号提升下游分类器性能。 |
| [^68] | [FrameBench:A Language Understanding Benchmark Based on Frame Semantics](https://arxiv.org/abs/2609.03370) | 该论文提出了基于框架语义学的新基准FrameBench，用于评估大语言模型能否像人类一样根据语境隐式区分同一动词所唤起的不同语义框架。 |
| [^69] | [Accountable AI with Grounded, Faithful, Consistent, Actionable Rationales: A Case Study in Clinical Trial Matching with VERDICT](https://arxiv.org/abs/2609.03366) | 论文提出了基于大语言模型的VERDICT智能体用于临床试验匹配任务，并创新性地引入“自我忠实性”作为可问责性的自动测试方法，通过生成有据可依、忠实、一致且可操作的理据来实现AI决策的可问责性。 |
| [^70] | [ALRA: Adaptive Local Relational Alignment for Logit-Based Pre-training Distillation of Autoregressive Language Models](https://arxiv.org/abs/2609.03355) | 提出ALRA框架，通过让学生提议候选词元并以教师最可能词元作为锚点，同时根据教师概率分布广度自适应调整候选词元数量，从而改进自回归语言模型的局部logit蒸馏效果。 |
| [^71] | [From Zero to Hero: An Open LLM Ecosystem for Armenian](https://arxiv.org/abs/2609.03350) | 该研究发布了ArmWeb（437万篇亚美尼亚语新闻）与ArmSTEM（37.3万条英亚平行数理题目）两个数据集，并通过继续预训练Gemma-4-E4B构建出首个附带完整训练数据和配方的开源亚美尼亚语模型arm-gemma-e4b，其性能超越所有现有的开放亚美尼亚语模型。 |
| [^72] | [FPCO-Dialog: A Multi-Turn False-Premise Benchmark for Correction and Cooperation in Vision-Language Models](https://arxiv.org/abs/2609.03331) | 该论文提出了FPCO-Dialog基准，包含1,080张图像和10,800个问题轮次，首次系统评估视觉语言模型在多轮对话中面对持续重复错误前提时的纠正与合作行为，并通过CorrTP@K指标揭示了20个模型间显著的纠正能力差异。 |
| [^73] | [Less Is Moral: A CHARMing Framework for Moral Foundations Detection in Endorsement Behaviour](https://arxiv.org/abs/2609.03330) | 本文提出CHARM框架，基于轻量级微调的大语言模型，将MAC交叉注意力、推理依据对齐和仇恨言论调制三个组件分别对应不同的心理学构念，从而以更低成本实现更稳健、更忠实的道德基础检测。 |
| [^74] | [How Perturbations Propagate: A Multi-Level Analysis of Robustness in Large Language Models](https://arxiv.org/abs/2609.03322) | 该论文首次从输出行为、隐藏状态几何和注意力头功能三个层面对大语言模型的扰动鲁棒性进行系统分析，揭示了扰动类型在模型内部表征中留下的可区分特征，而这些特征无法仅通过输出度量完全捕捉。 |
| [^75] | [Decoupling Turn-Taking from Semantics: A Decoupled Data Approach for Finite-State-Machine-Based Full-Duplex Dialogue](https://arxiv.org/abs/2609.03321) | 提出一种解耦数据方法，用真实人人口语对话学习话轮转换、用可配置的人机文本对话塑造语义行为，并通过基于规则的事件引导转换将口语对话序列化为FSM带，从而提升全双工对话中话轮转换的自然性。 |
| [^76] | [PACE: Towards Surfacing Hidden Conflicts in User Requests](https://arxiv.org/abs/2609.03293) | 该论文提出了PACE数据集，用于评估模型能否通过从知识库中检索隐式上下文证据，识别出使看似合理的用户请求变得不合适的潜在约束，从而实现基于冲突的个性化拒绝。 |
| [^77] | [Contextual Tamil Spelling and Grammar Correction Using Progressively Fine-Tuned Sequence-to-Sequence Transformers](https://arxiv.org/abs/2609.03273) | 提出端到端序列到序列方法，通过在65万余对合成噪声-干净句子对上按四阶段渐进式调度微调mT5-small和mBART-50，实现能够处理主谓一致、时态一致和跨词连声等上下文错误的泰米尔语拼写与语法纠错。 |
| [^78] | [MedQA-MM: Shortcuts Behind Medical Visual Reasoning](https://arxiv.org/abs/2609.03261) | 该研究揭示了医学多模态选择题基准测试中的“推理膨胀”现象——模型无需真正的视觉推理，仅靠答案措辞、临床文本、图像文字、人工标注等捷径线索即可答对题目，全输入准确率为62.63%而仅文本设置即达53%以上。 |
| [^79] | [What Else Needs Fixing? Exploring Cost-Effective Test-Time Compute for Revision Propagation in Artifacts Generated Through Conversation](https://arxiv.org/abs/2609.03254) | 本文针对对话生成工件中修订传播这一新问题提出了一个基准，评估了九种测试时计算修订方法，发现基线方法可达68.3%–93%的准确率，并识别出最具成本效益的方法。 |
| [^80] | [SGD-KV: Summarization Guided KV Cache Compression](https://arxiv.org/abs/2609.03235) | SGD-KV提出了一种感知注意力头的KV缓存压缩框架，通过新颖的块摘要诊断任务识别专门负责分层信息聚合的注意力头并据此分配缓存预算，在高达100万token的长上下文中实现最先进性能的同时，将KV缓存内存占用降低多达75%。 |
| [^81] | [Counterfactual Fairness Audits of Multi-Step Clinical LLM Agents Require a Measured Per-Action Instability Floor](https://arxiv.org/abs/2609.03221) | 临床LLM智能体在完全相同输入下本身就存在显著的动作不稳定性（约8.7%），因此反事实公平性审计必须先测量这一“每动作不稳定性底线”，否则任何检测到的人口统计学差异都无法解释。 |
| [^82] | [The Analyst in the Prompt: Role, Retrieval, and Memory Biases in LLM Financial Analysis](https://arxiv.org/abs/2609.03218) | 该研究通过对十二个大语言模型分析3,575份SEC备案文件的实验，揭示了用户上下文导致的金融分析偏差主要源于模型在不同角色下对相同证据的解读差异而非证据检索差异，并提出了两种简单的缓解策略。 |
| [^83] | [SWIM: Student Writing Simulation via Proficiency-Conditioned Generation](https://arxiv.org/abs/2609.03215) | 该论文提出SWIM任务，将学生写作模拟形式化为基于熟练度条件的作文生成，实验发现提示方法对写作熟练度的控制有限，模型虽能调整内容导向特征，却难以重现词汇、语法和组织结构层面的真实学生写作差异。 |
| [^84] | [LLMs Learn Better In-Context from Rules than from Examples](https://arxiv.org/abs/2609.03213) | 该研究通过五个跨领域任务比较发现，大语言模型从规则（指令遵循）中进行上下文学习比从示例（少样本提示）中学习更可靠，增加示例数量并不能带来显著提升，而指令微调会进一步放大基于规则学习的优势。 |
| [^85] | [Learning to Zoom Efficiently with a Contrastive Curriculum](https://arxiv.org/abs/2609.03206) | 该论文提出了一种无需标签和热启动微调的InfoNCE式内在奖励，通过难度递增的负样本工具调用课程进行对比学习，高效训练多模态大语言模型使用缩放工具，在多个基准上表现出色甚至超越SFT基线，并引入可扩展的M&C合成数据集直接评估模型的缩放能力。 |
| [^86] | [VoxReason: Listener-Free Evaluation of Source-Grounded Speech Planning Before Synthesis](https://arxiv.org/abs/2609.03203) | VoxReason提出了一种无需听者参与的评估任务，在语音合成之前通过带证据引用的说话计划和确定性验证器，衡量语音表达方式的选择是否真正建立在被引用的源记录之上。 |
| [^87] | [MemoryLACE: Memory Lifecycle-Aware Consolidation and Evidence Retrieval](https://arxiv.org/abs/2609.03201) | 提出了轻量级记忆框架MemoryLACE，通过稀疏的合并、取代和矛盾关系显式建模文本证据的生命周期，重建关系感知的证据单元以呈现当前、历史、支持和冲突证据，从而改进长期LLM智能体的记忆整合与证据检索。 |
| [^88] | [Jina-OCR-v1: Efficient Document Parsing with Speculative Decoding and Dense Verifiable Rewards](https://arxiv.org/abs/2609.03181) | Jina-OCR-v1 是一个可在低成本 GPU 上高效运行的端到端文档解析模型，通过递归共享草稿块的 FastMTP 推测解码实现无损加速，并结合密集可验证奖励的 GRPO 后训练，在多项基准上取得领先成绩并达到每秒 2.57 页的最高吞吐量。 |
| [^89] | [No country for old linguists: LLM-brain alignment underdetermines neural computation](https://arxiv.org/abs/2609.03160) | 本文指出，LLM与大脑之间的表征对齐虽然能够约束关于神经机制的假说，但并不足以唯一确定神经计算的实际机制，因此不能直接将LLM视为自然语言处理的机制性模型。 |
| [^90] | [Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization](https://arxiv.org/abs/2609.03158) | 提出CoverPruner，一种无需训练的视觉token剪枝方法，将剪枝创新性地建模为表示覆盖最大化问题，确保每个被剪枝的token都有存活的token为其代言，尤其在激进压缩下取得最佳准确率。 |
| [^91] | [Routing Is Not Enough: Diagnosing Intra-Adapter Subspace Contention in MoE+LoRA Fine-Tuning](https://arxiv.org/abs/2609.03150) | 该研究发现在 MoE+LoRA 多领域微调中，即使专家路由近乎完全分离，负迁移仍由近乎正交的领域梯度在同一低秩适配器子空间内竞争所致，并提出 SpawnLoRA，通过在检测到适配器争用时于 MoE 专家内部动态添加门控子适配器（保持路由器固定）来化解这一问题。 |
| [^92] | [Large Language Models in Resolving Contextual Knowledge Conflicts](https://arxiv.org/abs/2609.03148) | 该论文提出了上下文知识内部冲突的六类型分类法并构建了包含5,781个样本的ContextConflict数据集，实验发现当前大语言模型在解决此类冲突上仍有不足，并通过机制可解释性分析揭示了模型对冲突的潜在感知及其背后的表征几何结构。 |
| [^93] | [SHELF: A Synthetic Harness for Multi-Task Bibliographic Benchmarking](https://arxiv.org/abs/2609.03047) | SHELF是一个基于美国国会图书馆词表生成6万余篇合成文档的Python系统，为图书馆和档案馆的书目工作提供了涵盖分类、聚类、检索等多任务的系统性基准测试框架。 |
| [^94] | [Unifying Conformal Language Tasks with In-Context Ensembles](https://arxiv.org/abs/2609.03005) | 提出共形相关性框架，通过上下文学习示例筛选与集成自动构建评分函数，以最少的人工干预统一实现了多种NLP任务中覆盖率与简洁性的双重保证。 |
| [^95] | [Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation](https://arxiv.org/abs/2609.02998) | 该论文提出教师门控在线策略蒸馏（TGOPD），通过经验证器评分的教师探测在提示级别先验证教师模型的可靠性，将可靠提示路由到密集OPD监督、不可靠提示路由到基于验证器的GRPO，从而避免“自信但错误”的教师模型诱导误导性更新。 |
| [^96] | [The Geometry of Ignorance: LLMs Know When to Temper Bayesian Priors](https://arxiv.org/abs/2609.02959) | 研究发现大语言模型的反嵌入矩阵中存在一个编码训练语料词元分布的“无知方向”，模型通过逐词元调节该先验的强度，实现了随上下文信息增加而逐步减弱先验影响的温度调节贝叶斯更新。 |
| [^97] | [LexIssue: Benchmarking Legal Issue Identification in Chinese Civil Litigation](https://arxiv.org/abs/2609.02954) | 该论文构建了包含430个真实中国民事诉讼案例和1303个专家标注争议焦点的LexIssue基准数据集，并提出以争议焦点为中心的法律知识库，将争议焦点识别形式化为生成与分类两个互补任务以支持检索增强推理。 |
| [^98] | [Privacy-Preserving Heterogeneous Multi-LLM Federated Inference for Cognitive Diagnosis](https://arxiv.org/abs/2609.02947) | 该论文提出一种隐私保护的异构多LLM联邦推理框架，通过本地拉普拉斯噪声差分隐私和基于残差的聚合机制，使多个商用LLM API无需访问原始学生数据即可协作实现准确的认知诊断。 |
| [^99] | [Judging LLM-as-a-Judge: Concerning Rubric Artifacts in LLM-based Automated Text Generation Evaluation](https://arxiv.org/abs/2609.02942) | 研究发现LLM评估中的评估准则文本本身编码了可预测的评估信号，且评判者在候选回答或准则被反转时往往无法可靠更新判断，这引发了对基于准则的LLM自动评估可靠性的严重质疑。 |
| [^100] | [SISER: Speaker-Invariant Speech Emotion Recognition with Entropy-Based Adversarial Training](https://arxiv.org/abs/2609.02941) | 提出SISER框架，将wav2vec 2.0特征编码器与ECAPA-TDNN说话人判别器结合进行基于熵的对抗训练，同时解决标注数据稀缺和说话人差异两大难题，在IEMOCAP上达到60.63%的无加权准确率。 |
| [^101] | [Listen to the Latents: Self-Correcting Speech Recognition in Large Audio Language Models Through Hidden-State Interactions](https://arxiv.org/abs/2609.02940) | 该论文提出Hybrid Search纠错策略，利用基于LLM的ASR隐藏状态与基础LLM隐藏状态之间的交互特征来识别语义依赖程度高的词元并进行选择性精炼，从而显著提升LoRA适配的热启动初始化语音识别模型性能，超越重打分等全局纠错方法。 |
| [^102] | [A Spectral Phase Admissibility Certificate for Complex Linear Maps](https://arxiv.org/abs/2609.02911) | 该论文将量子引力中的Kontsevich-Segal-Witten判据引入机器学习，提出三种可微证书来约束复线性映射的谱集体相位，并通过Schur参数化实现可微执行，同时指出该约束因损害特征向量条件数而不适用于深度线性传播。 |
| [^103] | [RL-ADA: A World-Feedback Framework for Adversarially Robust Enterprise Dialogue Agents](https://arxiv.org/abs/2609.02902) | 该论文提出RL-ADA框架，用基于可衡量交互结果的世界反馈取代人工标注，让30亿参数的客户支持代理与70亿参数的对抗性客户代理在自动裁判引导下共同进化，从而训练出对抗鲁棒的企业对话代理。 |
| [^104] | [Dual-Form ASR: Semantics-Aware Inverse Text Normalization for Chinese Speech Recognition](https://arxiv.org/abs/2609.02901) | 提出双形式语音识别框架DF-ASR，利用LLM驱动的成对口语-书面形式监督数据构建及ITN-MWER序列级训练目标，将口语形式语音识别能力扩展到语义感知的书面形式逆文本正则化，统一了两种转录形式的建模。 |
| [^105] | [DisclosureBeta: A Measurement-Channel Theory for Regime-Conditioned Betas from LLM-Read Risk Disclosures](https://arxiv.org/abs/2609.02900) | 该论文将大语言模型建模为读取企业风险披露的含噪测量信道，在分段平稳的五因子框架下首次为状态条件贝塔建立了完整的可识别性、一致性与匹配误差下界理论，从而为价格历史过短的公司（如IPO企业）提供带误差预算的可靠贝塔估计。 |
| [^106] | [Contamination Inflates Scores but Rarely Reorders Large Language Model Leaderboards](https://arxiv.org/abs/2609.02899) | 该论文提出一种基于题目内改写对比的污染度量方法，发现基准污染虽然会抬高模型的绝对分数，但很少改变大语言模型排行榜的相对排名。 |
| [^107] | [Distilled Rapid Embedding Transfer (DRET): Parameter-Efficient Biomedical Domain Adaptation via Priority-Based Embedding Transfer](https://arxiv.org/abs/2609.02898) | DRET提出了一种无需在专业语料上重新训练的参数高效领域自适应方法，通过基于优先级的嵌入迁移机制，将BioBERT等大型生物医学专用模型的领域知识注入DistilBERT等轻量级通用模型中。 |
| [^108] | [Margins, Not Windows: Training-Free Per-Step Lossy Speculative Decoding](https://arxiv.org/abs/2609.02897) | AdaptiveSpec提出了一种无需训练的逐步推测解码方法，通过边际概率比规则放宽严格的token匹配验证，并动态调整草稿树的深度、宽度和节点数，从而在不受草稿长度和起草器架构限制的情况下加速LLM推理。 |
| [^109] | [PiPMRE: A Pipeline Based on Language Model for Medical Relation Extraction](https://arxiv.org/abs/2609.02896) | 该论文提出了一种基于语言模型的新型流水线框架PiPMRE，通过关系生成器产生候选关系三元组、再由关系过滤器筛选出合格结果的方式，摆脱了传统序列标注模式的限制，从而提升了医学关系抽取的性能。 |
| [^110] | [BharatGather: A Culturally-Informed Benchmark Dataset for Misinformation and Fake News Detection in Indian Public Events](https://arxiv.org/abs/2609.02895) | 本文提出了BharatGather数据集，一个专为印度大型公共活动中虚假信息二元分类设计的文化感知基准数据集，包含14,646条通过事实核查平台爬取、多媒体转录提取与大语言模型合成增强相结合的混合流水线构建的记录。 |
| [^111] | [R$^{2}$Adapter: A Routing and Rewriting Adapter for Efficient Hybrid RAG](https://arxiv.org/abs/2609.02894) | 提出了轻量级即插即用的路由与重写适配器R²Adapter，可动态将查询分配给原生RAG或基于图的RAG，仅对真正受益于图推理的查询进行图检索，从而降低不必要的开销并减少对底层大模型的依赖。 |
| [^112] | [Probe Generalization as Subspace Selection for OOD Deception Detection](https://arxiv.org/abs/2609.02893) | 该论文发现，将输入投影到训练激活分布中经LLM评判器筛选出的关键主成分子集上，可大幅提升线性探针在分布外欺骗检测任务上的跨域泛化能力，在Insider Trading Report上缩小了78%的性能差距。 |
| [^113] | [Counterexamples as Feedback for Agent Self-Correction](https://arxiv.org/abs/2609.02892) | 本文提出 A-CEGIS 轻量级框架，将反例作为反馈来评估智能体在自然语言到正则表达式合成中的多轮自我纠正能力，在四轮预算内解决了 90% 的任务，显著优于零样本生成和通用自我纠正方法。 |
| [^114] | [Bounded Personas Match Retrieval on Classification but Not Regression for a Frozen Agent](https://arxiv.org/abs/2609.02890) | 提出免训练方法PersonaLink，将用户历史蒸馏为有界的三字段人格画像并通过递归自评估精炼，证明该蒸馏人格画像在冻结智能体的分类任务上能够媲美检索方法，但在回归任务上仍无法匹敌。 |
| [^115] | [Where Does Harness-Optimization Value Live? Localized Gains and the Budget-Splitting Trap in Self-Evolving LLM Agents](https://arxiv.org/abs/2609.02889) | 提出HARNESSEVO，将智能体harness分解为角色、任务策略、工具/格式规则和反思/控制四个可独立进化的槽位，通过等预算对照与留一归因分析揭示：整体成功率提升有限（预算分割会稀释收益），优化的价值主要局部化于特定槽位。 |
| [^116] | [ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering](https://arxiv.org/abs/2609.02486) | 提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。 |
| [^117] | [Transfer Safety Awareness for Cross-Modal Safety Drift in Multimodal Large Language Models](https://arxiv.org/abs/2609.02082) | 针对多模态大语言模型中“跨模态安全漂移”这一新安全问题（无害文本结合图像即可传达有害意图且模型难以拒绝），提出轻量级的安全意识表示迁移方法（SRT），将文本安全信号迁移至视觉场景以有效缓解该风险。 |
| [^118] | [SFAD: Speculative Factuality-Aware Decoding](https://arxiv.org/abs/2609.00796) | 提出SFAD推测解码框架，通过构建细粒度扰动偏好数据集ConFide并利用DPO训练上下文忠实草稿模型，结合认知摩擦机制检测幻觉，在不增加推理开销的情况下显著增强大语言模型的上下文忠实度。 |
| [^119] | [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065) | 该论文提出了一个名为“科学智能体技能”的开放库，收录了基因组学、化学信息学等16个科研实践领域共163项程序性知识，使语言模型智能体能够遵循领域规范做出站得住脚的科学分析，而非仅仅返回能运行的代码。 |
| [^120] | [Budget-Aware Compression Pipeline for Single-GPU LLM Inference: Methods, Trade-offs, and Coupling Effects](https://arxiv.org/abs/2608.30076) | 该论文提出一种预算感知的单GPU大模型压缩流水线，通过系统研究剪枝、量化与KV缓存压缩之间的耦合效应，将70B模型压缩至约33GB并在单张A40上实现约57 tokens/s的推理速度，同时将精度损失控制在5%以内。 |
| [^121] | [Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090](https://arxiv.org/abs/2608.27370) | 本文提出了一种开源且成本高效的预训练配方，使在消费级RTX 5090 GPU上以极低计算成本训练出接近Qwen2.5-1.5B性能的Puro-2B模型成为可能。 |
| [^122] | [JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution](https://arxiv.org/abs/2608.25593) | JIT-Agent通过训练一个能即时生成和优化任务自适应工具模型的系统，显著提升了现成智能体的性能，使工具设计从手动变为自动化。 |
| [^123] | [TRACE: A Self-Evolving Skill Bank for Consistent, Limit-Aware LLM Agents](https://arxiv.org/abs/2608.22793) | TRACE通过构建自我进化的技能库，在不修改模型权重的情况下，提升LLM代理在重复任务中的一致性和限制意识，弥合了单次成功与一致成功之间的可靠性差距。 |
| [^124] | [K-Bench: measuring model performance on real scientific agent requests](https://arxiv.org/abs/2608.21601) | 本论文提出K-Bench 01，一个基于真实科学请求的评估框架，发现当前前沿模型在满足领域科学家接受标准上均未达到阈值，其中gpt-5.6-sol表现最优但仍有不确定性。 |
| [^125] | [DeltaMomentum: A Key-Value based Anisotropic Momentum Update via Delta Rule](https://arxiv.org/abs/2608.19491) | DeltaMomentum通过利用梯度中的键值结构，将方向感知引入动量更新规则，使每个方向以与其出现频率相关的速率被遗忘，从而无需矩阵即可实现输入侧曲率校正。 |
| [^126] | [Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence](https://arxiv.org/abs/2608.10720) | Ex-Omni-2D 提出通过“视觉思维计划”协调文本、个性化语音与基于参考视频的生成，使全模态对话智能体在语音回答的同时拥有原生视觉形象，并可将全序列教师模型蒸馏为少步骤的流式学生模型。 |
| [^127] | [Gaokerena: A Small Persian Medical Language Model Family](https://arxiv.org/abs/2608.00932) | 本文提出了Gaokerena，一个专为消费级硬件设计的小型波斯语医学语言模型家族，其中Gaokerena-V通过新构建的波斯语医学语料库训练提升了医学问答性能，Gaokerena-R则结合思维链与两个新型RLAIF框架来增强临床推理能力。 |
| [^128] | [Reading and Steering Representations of Materials-Science Mechanisms in an Open-Weight Language Model](https://arxiv.org/abs/2607.20058) | 该研究在开放权重语言模型中首次识别出材料科学机制内部表征的三个可实验验证的特征，并证明通过因果干预可以读取并调控模型对材料物理机制的表征。 |
| [^129] | [PalmClaw: A Native On-Device Agent Framework for Mobile Phones](https://arxiv.org/abs/2607.13027) | PalmClaw 是一个原生运行在手机上的开源智能体框架，直接在设备端管理会话、记忆、技能与工具，并将设备能力封装为可调用的设备工具，从而突破了传统依赖 GUI 操作的移动智能体的局限。 |
| [^130] | [What You See Is What You Get: Observation-Aligned Supervision for Chart-to-Code Generation](https://arxiv.org/abs/2607.04726) | 论文揭示了图表到代码生成训练中存在的四类潜在变量与观察图像不匹配问题，并提出观察对齐监督方法，用视觉上可约束的量替换潜在变量作为监督目标。 |
| [^131] | [Can Dialects Be Steered Like Languages? Sparse Neurons and Distributed Directions in Arabic LLMs](https://arxiv.org/abs/2607.03936) | 该研究发现阿拉伯语大语言模型中的方言特征可通过占MLP维度不足1%的稀疏神经元与分布式激活方向在推理时被探测和引导，无需微调即可将模型输出导向目标方言。 |
| [^132] | [KARMA: Knowledge graph-based Automated Reasoning Materialization and Alignment](https://arxiv.org/abs/2607.03166) | KARMA 通过在领域知识图谱上枚举模式约束路径生成槽位对齐的对比候选样本，并利用槽位并行对齐（SPA）将偏好监督精准路由至区分性实体槽位，从而解决了基于模板的对比合成中的分辨率不匹配问题。 |
| [^133] | [Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization](https://arxiv.org/abs/2606.31002) | 该论文提出将Lean编译与GPT-5.2和Gemini-2.5-Pro严格语义共识相结合的自动形式化评估标准，发现编译通过率会高估语义忠实度达3.0至29.0个百分点，且该标准与人类多数判断的一致率达89.7%。 |
| [^134] | [Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization](https://arxiv.org/abs/2606.23989) | 提出CAMS框架，将声明级归因嵌入“提取—选择—改写”流程，使多文档摘要中的每句话都能锚定到经过验证、可溯源的源文本片段，从而在构造层面保证摘要的忠实性。 |
| [^135] | [Breaking the Likelihood Trap: Variance-Calibrated Modulation for Large Language Model Decoding](https://arxiv.org/abs/2606.22511) | 提出一种无需训练的解码前干预方法VCM，通过基于PMI的上下文探照灯和自适应自我去偏两种动态机制在截断前重塑概率分布，解决大语言模型生成中的重复退化和词汇贫乏问题。 |
| [^136] | [Chehre: An Emoji-Prompted Dataset to Explore Perceptual Flexibility in Video Language Models](https://arxiv.org/abs/2606.21657) | 提出了Chehre数据集，通过203名参与者录制40种表情符号并将其面部动作迁移到合成面部以保护隐私，收集了由1,242名标注者标注的2,111个视频，并据此定义了“分布式表情识别”这一新任务，以测试视频语言模型能否复现人类对面部表情感知的变异性。 |
| [^137] | [Learning What Not to Forget: Long-Horizon Agent Memory from a Few Kilobytes of Learning](https://arxiv.org/abs/2606.20954) | LRE是一种千字节级、仅用CPU、无需语言模型的学习型淘汰评分器，通过逐字提取保留任务关键历史信息，在智能体任务上以极低成本恢复保留完整历史93%的准确率，并将最坏情况峰值提示削减52%。 |
| [^138] | [Structured Inference with Large Language Gibbs](https://arxiv.org/abs/2606.19264) | 提出了Large Language Gibbs方法，将LLM的条件分布作为MCMC的转移算子，通过在其他变量条件下迭代重采样单个变量来实现结构化概率推断，从而避免了自回归生成中的顺序依赖偏差。 |
| [^139] | [LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents](https://arxiv.org/abs/2606.18388) | LLMZero利用大语言模型智能体结合树搜索，通过在每个检查点诊断训练状态来自适应地优化RL后训练的多参数调度策略，在四个GRPO任务上比基础模型提升9%-140%、比网格搜索提升6%-15%，并揭示了容量参数单调累积、正则化参数震荡变化的训练规律。 |
| [^140] | [Uncertainty Is Not a Safety Net for Clinical VQA, but Can It Anticipate Model Failure?](https://arxiv.org/abs/2606.16583) | 该研究通过在12个临床视觉语言模型上基准测试8种不确定性估计方法，发现不确定性估计的质量会随模型准确率下降而退化、且在正确答案被隐藏时无法发出警示，但未扰动输入上的不确定性本身携带诊断信息，能够预判哪些预测将在模型失效时崩溃。 |
| [^141] | [ParaBridge: Bridging Paralinguistic Perception and Dialogue Behavior in Speech Language Models](https://arxiv.org/abs/2606.10581) | ParaBridge提出一种在线策略自蒸馏方法，将推理时脆弱的副语言指令支架转化为语音语言模型中稳定的行为，从而弥合副语言感知与对话行为之间的差距。 |
| [^142] | [Beyond Accuracy: Community Perspectives on Machine Translation](https://arxiv.org/abs/2606.09655) | 该论文首次大规模分析了四个利益相关社区在社交媒体上发布的79,286条关于机器翻译的帖子和评论，揭示了技术进步与真实用户需求之间的差距，强调倾听用户社区声音以引导研究方向的必要性。 |
| [^143] | [TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment](https://arxiv.org/abs/2606.07451) | TEVI框架利用稀疏自编码器解耦图像嵌入，并通过文本条件化的掩码模块只保留与文本描述相符的信息、剔除多余内容，从而改善CLIP等视觉-语言模型中图像-文本嵌入的对齐问题。 |
| [^144] | [SV-Detect: AI-generated Text Detection with Steering Vectors](https://arxiv.org/abs/2606.07313) | 本文提出SV-Detect，一种利用从冻结语言模型隐藏表示中提取的转向向量构建逐层投影特征来检测AI生成文本的方法，在跨领域、跨源模型及编辑攻击等分布偏移场景下均实现了强大的检测性能。 |
| [^145] | [EDIT: Evidence-Diagnosed Intervention Training for Rule-Faithful LLM Grading](https://arxiv.org/abs/2606.06350) | EDIT是一个两阶段训练框架，利用模型内部的信念信号定位并修正LLM评分器中有问题的推理步骤，并通过信念引导的强化学习奖励塑形，使LLM的评分更忠实于评分规则和学生答案证据。 |
| [^146] | [ArcANE: Do Role-Playing Language Agents Stay in Character at the Right Time?](https://arxiv.org/abs/2606.05553) | 本文提出ArcANE基准，通过构建刻画角色价值观、动机和关系随故事演变的“情节弧”，评估角色扮演语言代理能否在叙事的不同阶段准确呈现角色的动态发展，弥补了现有基准将角色视为固定设定的不足。 |
| [^147] | [Expert-Aware Causal Tracing of Factual Recall in Sparse MoE Language Models](https://arxiv.org/abs/2606.03780) | 该研究将激活修补方法细化到专家层面，首次证明在稀疏MoE语言模型（如Qwen3-30B-A3B-Base）中，事实回忆的恢复可定位于单个路由专家（L44E069），但这种专家级定位在不同模型间并不一致。 |
| [^148] | [Large AI Models in Dental Healthcare: From General-Purpose Systems to Domain-Specific Foundation Models](https://arxiv.org/abs/2606.02914) | 本文首次提出按架构范式和牙科专业化程度划分的二维分类框架，系统综述了97项研究，统一考察了语言生成模型、视觉基础模型和牙科专用基础模型三类大规模AI模型在牙科医疗中的关系与共同局限。 |
| [^149] | [Fixing FOLIO and MALLS: Verified Annotations and an LLM-assisted Framework to Focus Human Relabeling](https://arxiv.org/abs/2606.02837) | 本研究系统性审查发现 FOLIO 和 MALLS 基准中约 42% 的一阶逻辑标注存在错误，并发布修正后的真值标注以及一个 LLM 辅助框架来聚焦人工重新标注工作。 |
| [^150] | [CoMAP: Co-Evolving World Models and Agent Policies for LLM Agents](https://arxiv.org/abs/2606.02372) | 本文提出COMAP框架，通过闭环交互使文本世界模型与LLM智能体策略共同演化——世界模型为候选动作预测未来状态反馈以支持智能体的前瞻性反思，智能体产生的同策略轨迹又通过自蒸馏反哺更新世界模型，从而摆脱了对外部奖励或验证器的依赖。 |
| [^151] | [Argument Collapse: LLMs Flatten Long-Form Public Debate](https://arxiv.org/abs/2606.01736) | 该论文首次提出并系统量化了“论点坍塌”现象：大语言模型生成的议论文会高度收敛到极少数相同论点（人类论点65.3%独特而模型仅3.4%），即使显式要求多样性也只能恢复约一半的人类论点，揭示了LLM可能使公共辩论同质化、扁平化的风险。 |
| [^152] | [Attend to Evidence: Evidence-Anchored Spatial Attention Supervision for Multimodal RLVR](https://arxiv.org/abs/2605.30912) | EASE通过将标注证据区域转化为平滑的视觉token目标，在高奖励轨迹上监督回复到图像的空间注意力，为多模态RLVR引入视觉证据过程监督，且推理时无需任何标注。 |
| [^153] | [Skill-Conditioned Gated Self-Distillation for LLM Reasoning](https://arxiv.org/abs/2605.28791) | 本文提出SGSD，通过技能库和门控机制将自蒸馏从无条件模仿转为教师假设验证，以应对不可靠技能并提升大语言模型推理能力。 |
| [^154] | [SuperValid: Capability-Aligned OOD Validation for Generalizable Downstream Scaling](https://arxiv.org/abs/2605.28179) | 该论文提出SuperValid框架，通过从能力域内的基准中提炼核心概念并扩展为多样化的知识型文本，合成能力对齐的分布外验证数据，从而在能力层面实现更具泛化性的下游扩展预测。 |
| [^155] | [MIRA: A Bilingual Benchmark for Medical Information Response Audit](https://arxiv.org/abs/2605.28025) | 该论文提出了首个双语医疗基准MIRA，揭示了大语言模型在应对低健康素养用户提问时会系统性遗漏关键医疗信息、减少后续行动建议的“差异化信息稀释”（DID）这一安全隐患。 |
| [^156] | [EmoDistill: Offline Emotion Skill Distillation for Language Model Agents in Adversarial Negotiation](https://arxiv.org/abs/2605.26785) | 提出EmoDistill离线蒸馏框架，通过IQL情感选择器与LoRA微调的表达策略相结合，将大模型间对抗谈判中的情感技能迁移到小型语言模型智能体，使其能够抵御情感操纵并维护用户谈判目标。 |
| [^157] | [The Age of Curiosity Meets the Age of AI: Benchmarking Child Safety in Large Language Models](https://arxiv.org/abs/2605.25510) | 提出了基于发展心理学的儿童安全评估基准KIDBench，用于测试大语言模型对7-11岁儿童提问的安全性，并发现提供儿童身份线索能显著提升模型的安全表现。 |
| [^158] | [How Much Do Circuits Tell Us? Measuring the Consistency and Specificity of Language Model Circuits](https://arxiv.org/abs/2605.08348) | 本文提出用一致性和特异性两个新属性来评估机制可解释性中的电路，发现组件级电路虽高度一致且因果重要但不具任务特异性，而神经元级电路虽具任务特异性却一致性较差。 |
| [^159] | [Beyond Decodability: Reconstructing Language Model Representations with an Encoding Probe](https://arxiv.org/abs/2605.00607) | 本文提出一种编码探针方法，通过可解释特征重构语言模型的内部表示，克服了传统解码探针无法直接比较不同特征贡献且受特征相关性干扰的两大局限。 |
| [^160] | [When Chain-of-Thought Fails, the Solution Hides in the Hidden States](https://arxiv.org/abs/2604.23351) | 研究发现，即使思维链推理轨迹本身是错误的，其隐藏状态（尤其集中于中后层和轨迹早期）仍编码了足以恢复正确答案的任务相关信息，通过激活修补将这些隐藏状态注入直接回答过程可显著提升答题准确率。 |
| [^161] | [One Model to Translate Them All? A Journey to Mount Doom for Multilingual Model Merging](https://arxiv.org/abs/2604.02881) | 该论文系统研究了多语言机器翻译中的权重空间模型合并，揭示了显著的方向性不对称现象——当模型共享目标语言时合并相对有效但无法保留各语言模型的峰值性能，而当目标语言不同时性能则急剧下降。 |
| [^162] | [Arabic Morphosyntactic Tagging and Dependency Parsing with Large Language Models](https://arxiv.org/abs/2603.16718) | 本文对大语言模型在阿拉伯语形态句法标注与依存句法分析任务上进行了统一评估，发现基于检索的上下文学习能显著提升性能，最强的大语言模型已接近有监督系统的水平，但需要大量标注数据和计算资源。 |
| [^163] | [Imagination Helps Visual Reasoning, But Not Yet in Latent Space](https://arxiv.org/abs/2602.22766) | 本文通过因果中介分析揭示当前潜在视觉推理存在输入与潜在token、潜在token与最终答案之间的两个关键因果断连，表明潜在空间的想象机制尚未真正发挥有效作用。 |
| [^164] | [Ex-Omni: Enabling 3D Facial Animation Generation for Omni-modal Large Language Models](https://arxiv.org/abs/2602.07106) | 该论文提出Ex-Omni框架，通过带混合变形协同监督的语音单元生成器与非自回归混合变形解码器将语义推理与时间生成解耦，并结合令牌即查询门控融合接口和120万样本弱监督数据集InstructS2SF-1200K，首次使全模态大语言模型能够联合生成语音与3D面部动画。 |
| [^165] | [Deep networks learn to parse uniform-depth context-free languages from local statistics](https://arxiv.org/abs/2602.06065) | 该研究引入了一类可调节歧义程度和跨尺度相关结构的概率上下文无关文法，揭示了深度网络能够仅从局部统计特征中学习解析语言的层次结构。 |
| [^166] | [Benchmarking Machine Translation on Chinese Social Media Texts](https://arxiv.org/abs/2601.22931) | 该论文提出了CSM-MTBench基准，涵盖五个中外语言方向，包含“趣味帖子”和“社交片段”两个专家策划的子集，并针对中文社交媒体文本提出了定制化评估方法，以解决数据稀缺和传统评估指标难以捕捉风格保真度的双重难题。 |
| [^167] | [Relational Linearity is a Predictor of Hallucinations](https://arxiv.org/abs/2601.11429) | 该论文提出关系线性性可预测语言模型的幻觉：由于抽象表示方案，语言模型能轻松为线性关系中不存在的主体生成看似合理的客体从而导致幻觉，而在面对非线性关系时这种机制失效，幻觉更容易避免。 |
| [^168] | [HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning](https://arxiv.org/abs/2601.10187) | 该论文提出了Sand-Glass音节级时长约束翻译基准和Homura强化学习框架，通过新颖的动态音节比率奖励有效解决LLM翻译的跨语言冗长偏差问题，使其适用于字幕、配音等时间受限场景。 |
| [^169] | [Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models](https://arxiv.org/abs/2601.08955) | 提出了ITP统一框架，让智能体策略模型与世界模型交互生成多步想象轨迹，并通过权衡最终目标与任务进度的自适应前瞻机制，充分释放世界模型在复杂任务规划中的潜力。 |
| [^170] | [CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems](https://arxiv.org/abs/2601.05520) | 该论文提出CHisAgent多智能体框架，通过归纳、扩展、充实三个角色专业化阶段，从《二十四史》等中国古代文献中自动构建历史事件分类体系，克服了LLM在中国历史语境下推理能力不足和人工分类构建成本高的问题。 |
| [^171] | [Towards Multi-modal Multi-turn Safety: From Agentic Interaction to Strategic Alignment](https://arxiv.org/abs/2601.04736) | 提出MINT-Safe开源视觉多轮安全训练数据集，针对多轮交互中攻击者渐进式重构有害意图的新型安全风险，弥补现有RLHF方法无法捕捉跨轮次风险动态的不足。 |
| [^172] | [User Perceptions vs. Proxy LLM Judges: Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios](https://arxiv.org/abs/2510.20721) | 该研究通过94人的用户实验发现，用户对LLM在隐私敏感场景下响应的评价彼此一致性较低，这表明此前以代理LLM作为评审的基准测试结果可能与真实用户的隐私和有用性感知存在显著偏差。 |
| [^173] | [Evaluating Large Language Models on Urdu Idioms](https://arxiv.org/abs/2510.17460) | 该研究构建了一个包含4000个人工验证的乌尔都语习语句对的双文字基准数据集，通过翻译、改写、习语检测和回译等多项任务及多种提示策略进行评估，发现前沿大语言模型在所有设置下均优于传统神经机器翻译系统。 |
| [^174] | [EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering](https://arxiv.org/abs/2509.25175) | EasySteer是一个基于vLLM构建的高性能、可扩展的大语言模型推理时引导统一框架，相比现有框架实现了10.8-22.3倍的加速，并提供模块化可插拔接口、细粒度参数控制以及八个应用领域的预计算引导向量。 |
| [^175] | [Causal-Counterfactual RAG: The Integration of Causal-Counterfactual Reasoning into RAG](https://arxiv.org/abs/2509.14435) | 该论文提出因果-反事实RAG框架，通过将显式因果图与反事实推理融入检索增强生成过程，解决了传统RAG系统因文本分块破坏上下文完整性和过度依赖语义相似性检索而导致的回答浅显不准确的问题。 |
| [^176] | [Human Psychometric Questionnaires Mischaracterize LLM Behavior](https://arxiv.org/abs/2509.10078) | 该研究发现，人类心理测量问卷中的题目含有明显的词汇线索，会让大语言模型识别出被测构念并给出符合社会期望的回答，因此基于问卷得到的模型人格与价值观画像并不能反映其在真实日常用户交互中的实际生成行为。 |
| [^177] | [Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications](https://arxiv.org/abs/2508.00669) | 本文是首个针对大语言模型医学推理领域的系统综述，提出了涵盖训练时策略与测试时机制的推理增强技术分类体系，并系统分析了这些技术在多种数据模态和临床应用中的实践与评估方法。 |
| [^178] | [IDRBench: Understanding the Capability of Large Language Models on Interdisciplinary Research](https://arxiv.org/abs/2507.15736) | 该论文提出IDRBench框架，通过论文识别、想法整合和想法推荐三项评估任务，系统评估了十个主流大语言模型在跨学科研究中整合不同领域知识的能力。 |
| [^179] | [LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models](https://arxiv.org/abs/2503.03313) | 该论文提出PromptGFM，通过图词表学习将图节点融入语言模型词表空间，克服了现有LLM与GNN解耦架构及OOV token带来的跨图、跨任务迁移难题，构建了面向文本属性图的多功能图基础模型。 |
| [^180] | [AgentRM: Enhancing Agent Generalization with Reward Modeling](https://arxiv.org/abs/2502.18407) | 本论文提出可泛化奖励模型AgentRM，发现微调奖励模型来引导测试时搜索比直接微调策略模型更稳健，在九个智能体任务上平均提升8.8分并超越最强通用智能体4.0分。 |
| [^181] | [LDC: Learning to Generate Research Idea with Dynamic Control](https://arxiv.org/abs/2412.14626) | 提出首个结合监督微调与可控强化学习的两阶段框架，利用多维奖励模型和细粒度反馈动态控制研究想法生成，从而在新颖性、可行性和有效性之间实现平衡，提升大语言模型科研构思质量。 |
| [^182] | [Detecting Conversational Mental Manipulation with Intent-Aware Prompting](https://arxiv.org/abs/2412.08414) | 提出意图感知提示（IAP）方法，利用大语言模型捕捉对话参与者的潜在意图来检测心理操纵，在MentalManip数据集上表现优于其他提示策略，并显著减少了假阴性。 |
| [^183] | [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047) | 本文揭示了语法约束解码会扭曲大语言模型的输出分布，导致生成结果虽符合语法但质量低下，并提出了一种名为ASAp的语法对齐解码算法来解决这一问题。 |

# 详细

[^1]: 训练式编译：将自然语言规范转化为本地神经函数

    Compile by Training: Turning Natural-Language Specifications into Local Neural Functions

    [https://arxiv.org/abs/2609.04199](https://arxiv.org/abs/2609.04199)

    提出“训练式编译”方法，将自然语言规范编译为可复用的本地神经函数，通过教师模型生成的示例训练小型适配器，无需每次调用远程大模型即可达到83.6%的语义准确率。

    

    许多反复出现的文本功能很容易描述，但难以用规则来实现，而每次输入都调用大型远程模型会带来重复的成本、延迟以及对服务提供商的依赖。我们提出了“训练式编译”，它将自然语言规范转化为可复用的神经函数。在编译阶段，教师模型生成任务特定的示例，用于为一个紧凑的解释器训练小型适配器。生成的函数无需教师模型即可运行，并且可以像普通软件一样进行存储、版本管理和组合。在FuzzyBench-Hard（一个Program-as-Weights快速编译器无法产生精确匹配的子集）上，训练式编译达到了83.6%的语义准确率。这一更高的准确率伴随着更高的编译时间成本：大约需要一分钟，而快速编译器只需几秒钟。我们将该编译器部署在一个公开的交互式服务中，并在多站点网站上展示了编译后的函数。

    arXiv:2609.04199v1 Announce Type: cross  Abstract: Many recurring text functions are easy to describe but difficult to implement with rules, while calling a large remote model for every input introduces repeated cost, latency, and dependency on a provider. We present compile by training, which turns a natural-language specification into a reusable neural function. At compile time, teacher models generate task-specific examples that are used to train a small adapter for a compact interpreter. The resulting function runs without the teachers and can be stored, versioned, and composed like ordinary software. On FuzzyBench-Hard, a subset on which the Program-as-Weights fast compiler produced no exact matches, compile by training reaches 83.6% semantic accuracy. This higher accuracy comes with a higher compile-time cost: roughly a minute rather than seconds for the fast compiler. We deploy the compiler in a public interactive service and demonstrate compiled functions in a multi-site websit
    
[^2]: ESPO：通过诊断、多样化与稳定化实现的错误结构化提示优化

    ESPO: Error-Structured Prompt Optimization via Diagnose, Diversify, and Stabilize

    [https://arxiv.org/abs/2609.04197](https://arxiv.org/abs/2609.04197)

    ESPO通过诊断错误模式、多样化候选生成和稳定性选择三个阶段，解决了进化式提示优化中的提示膨胀问题，在七个NLP基准上平均准确率超越GEPA达3.76个百分点，同时提示词更短47%且推理更快。

    

    以GEPA为代表的进化式提示优化器存在提示膨胀问题：每次迭代都会追加规则和注意事项，导致提示词长度增至3倍，但准确率却没有提升。我们将这一问题归因于三个缺陷：错误观察不完整、搜索多样性有限以及选择不可靠，并提出ESPO（错误结构化提示优化），将提示优化分解为三个阶段：Diagnose（诊断）在一轮中将所有训练错误聚类为结构性模式；Propose（提议）通过四种具有独立偏好的互补策略生成候选提示；Select（选择）应用自助法稳定性选择。在七个公开NLP基准——Tweet、MMLU、GSM8K、HotpotQA、ScoNe、HoVer和PUPA上，ESPO相比最先进方法平均准确率提升3.76个百分点（74.67%对比GEPA的70.91%），在所有数据集上持平或超越GEPA，同时生成的提示词缩短47%（1,004字符对比1,878字符）且推理速度更快。

    arXiv:2609.04197v1 Announce Type: cross  Abstract: Evolutionary prompt optimizers such as GEPA suffer from prompt bloat: each iteration appends rules and caveats, producing prompts up to 3$\times$ longer yet no more accurate. We trace this to three deficiencies - incomplete error observation, limited search diversity, and unreliable selection - and propose ESPO (Error-Structured Prompt Optimization), which decomposes prompt optimization into three phases: Diagnose clusters all training errors into structural patterns in one round; Propose generates candidates via four complementary strategies with independent biases; Select applies bootstrap stability selection. On seven public NLP benchmarks - Tweet, MMLU, GSM8K, HotpotQA, ScoNe, HoVer, and PUPA - ESPO improves average accuracy by $+$3.76 pp over the state-of-the-art (74.67% vs 70.91% for GEPA), matching or exceeding GEPA on every dataset while producing prompts 47% shorter (1,004 vs 1,878 chars) and faster at inference. Cross-model e
    
[^3]: 可读性不等于可解释性：比较思维链推理中被评判的重要性与实际重要性

    Legibility is Not Interpretability: Comparing Judged and Actual Importance in Chain-Of-Thought Reasoning

    [https://arxiv.org/abs/2609.04194](https://arxiv.org/abs/2609.04194)

    本研究将思维链推理步骤的重要性量化为蒙特卡洛模拟估计的“优势”，发现LLM评判器虽能超越简单基线但远不足以准确识别真正重要的推理步骤，表明推理文本的可读性并不等于可解释性。

    

    来自思维链模型的推理轨迹似乎为了解模型如何得出答案提供了一个清晰可读的窗口。越来越多的研究正是这样对待它们，使用LLM评判器来诊断错误、评估忠实性，并通过过程奖励模型和生成式批评家提供步骤级监督。这些做法依赖于推理步骤的文本能够承载关于其功能作用的信息。但文本实际上是否编码了哪些推理步骤真正重要的信息？我们将推理步骤的重要性操作化为其“优势”：即包含该步骤后期望奖励（例如产生正确的最终答案）的变化，并通过蒙特卡洛模拟进行估计。以这些估计作为真值，我们评估了LLM评判器能否识别高优势步骤，发现足够强大的LLM能够超越流行率基线，但远低于噪声上限。将模型微调为步骤级……

    arXiv:2609.04194v1 Announce Type: new  Abstract: Reasoning traces from chain-of-thought models appear to offer a legible window into how a model arrives at its answer. A growing body of work treats them as such, using LLM judges to diagnose errors, evaluate faithfulness, and provide step-level supervision via process reward models and generative critics. These practices rely on the text of a reasoning step carrying information about its functional role. But does the text actually encode information about which reasoning steps matter? We operationalize the importance of a reasoning step as its advantage: the change in expected reward, e.g., producing the correct final answer, from including that step, estimated via Monte Carlo rollouts. Basing ground truth on these estimates, we evaluate whether LLM judges can identify high-advantage steps and find that sufficiently capable LLMs can outperform a prevalence baseline but fall well short of a noise ceiling. Fine-tuning a model as a step-le
    
[^4]: 预训练期间的知识获取？大语言模型借助辅助视图学得更好

    Knowledge Acquisition During Pre-training? Large Language Models Learn Better With Auxiliary Views

    [https://arxiv.org/abs/2609.04180](https://arxiv.org/abs/2609.04180)

    研究发现，在预训练中将token预算从文档重复转移到辅助视图（知识的重新表述）能提升大语言模型的学习效果，即使对事实回忆也有效，且不依赖教师模型的强弱。

    

    我们对大语言模型（LLMs）在预训练期间如何获取知识的理解仍存在空白。我们提出假设：辅助视图（auxiliary views），即知识的重新表述，对学习具有因果性的帮助。我们设计了受控实验来隔离验证这一假设。首先，我们确认重复是知识获取的必要条件，并澄清释义改写仅在小批量大小（batch size）时才有所帮助。其次，在保持token预算固定的情况下，将token从文档重复中分配给辅助视图可以提升学习效果——反直觉的是，即使对于事实回忆也是如此。第三，辅助视图的有效性并不取决于生成它的教师模型的强弱。第四，我们识别出两类知识形式——上下文性知识和基础性知识——它们能够在存在先前知识空白的情况下帮助学习。最后，我们通过逐层偏置（layer-wise biases）和压缩（compression）机制，从机理层面考察了这些效应的表现。总之，我们的发现表明辅助表示……

    arXiv:2609.04180v1 Announce Type: cross  Abstract: Gaps remain in our understanding of how large language models (LLMs) acquire knowledge during pre-training. We posit that auxiliary views, reformulations of knowledge, are causally helpful for learning. We design controlled experiments to isolate this. First, we confirm that repetition is necessary for acquisition and clarify that paraphrasing helps only at smaller batch sizes. Second, holding the token budget fixed, allocating tokens from document repetition to auxiliary views improves learning, counterintuitively, even for factual recall. Third, the effectiveness of auxiliary views is not contingent on the strength of the teacher model that generates them. Fourth, we identify forms of knowledge, contextual and foundational, that aid learning in the presence of prior knowledge gaps. Finally, we examine how these effects manifest mechanistically via layer-wise biases and compression. Together, our findings suggest that auxiliary repres
    
[^5]: 终极翻译基准测试

    Last Translation Benchmark

    [https://arxiv.org/abs/2609.04173](https://arxiv.org/abs/2609.04173)

    提出了终极翻译基准测试，这是一个包含人工编写、经同行评审的多模态示例的基准数据集，通过为每个示例配备手工编写的验证规则来描述具体失败案例，解决了现有机器翻译基准趋于饱和且评估方法不可靠的问题。

    

    为了推动科学进步，我们需要能够测试最先进模型极限的基准测试，以及能够揭示失败案例的评估方法。随着模型日益强大，机器翻译的标准基准测试正趋于饱和。此外，自动翻译指标不可靠、容易受到奖励破解的攻击，且提供的评估缺乏可操作性。即使是黄金标准的人工评估也并非完美，因为它通常缺乏可重复性、客观性和可扩展性。总体而言，这阻碍了我们追踪该领域的客观进展并确定改进路径。我们提出了终极翻译基准测试，这是一个由人工编写并经过同行评审的示例（文本、图像、音频、视频）集合，这些示例能够难倒领先的机器翻译模型。我们还提出了一种新的评估方法：每个示例都附有手工编写的验证规则，用于描述该示例上的具体失败案例，因此能够……

    arXiv:2609.04173v1 Announce Type: new  Abstract: For scientific progress, we need benchmarks that test the limits of state-of-the-art models, and evaluation methods that inform us about failure cases. As models get stronger, standard benchmarks for machine translation are approaching saturation. Further, automatic translation metrics are unreliable, vulnerable to reward-hacking, and provide unactionable assessments. Even gold human evaluation is not problem-free, because it often lacks reproducibility, objectivity, and scalability. Overall, this prevents us from tracking objective progress in the field and identifying pathways for improvement. We introduce the Last Translation Benchmark, a collection of human-authored and peer-reviewed examples (texts, images, audio, videos) that break leading machine translation models. We also present a new evaluation approach: each example comes with handcrafted verification rules describing concrete failure cases on that example, therefore allowing
    
[^6]: 重新思考大语言模型的在线策略蒸馏 II：单个训练样本

    Rethinking On-Policy Distillation of Large Language Models II: One Training Example

    [https://arxiv.org/abs/2609.04172](https://arxiv.org/abs/2609.04172)

    该研究发现仅用一个训练样本进行在线策略蒸馏就能持续改进并达到全数据训练的大部分性能，原因是单个查询即可覆盖全数据训练71.5%的状态，而16个语义不同的查询可达到98.9%的覆盖率并完全匹配全数据训练的效果。

    

    arXiv:2609.04172v1 公告类型：新论文 摘要：在线策略蒸馏（OPD）将学生模型生成的轨迹与教师模型提供的密集token级监督相结合。现有工作主要研究其算法行为，而训练数据的作用尚不明确。我们通过在单个查询上进行训练，在数据极简极限下考察训练数据的作用。单样本OPD在数百步内持续改进，并在跨任务领域和模型家族中恢复了全数据OPD的大部分增益。我们通过训练过程中访问的状态以及学生模型与教师模型对齐的速率来解释这一结果。我们测量了“状态覆盖率”，即某个查询集的轨迹所能达到的全数据OPD访问状态的比例。单个查询已能达到71.5%的覆盖率，其中大部分在前100步内实现。添加语义上不同的查询会使覆盖率和验证准确率同步提升，直到16个查询达到98.9%的覆盖率并与全数据训练的表现相匹配。然而，无论使用何种查询，对齐速度都会以相似的速率减缓……

    arXiv:2609.04172v1 Announce Type: new  Abstract: On-policy distillation (OPD) combines student-generated rollouts with dense token-level supervision from a teacher. Existing work has mainly studied its algorithmic behavior, leaving the role of training data unclear. We examine this role at the data-minimal limit by training on a single query. One-shot OPD keeps improving for hundreds of steps and recovers most of full-data OPD's gain across task domains and model families. We explain this result through the states visited during training and the rate at which the student aligns with the teacher. We measure \emph{state coverage}, the fraction of the states full-data OPD visits that a query set's rollouts reach. A single query already reaches \(71.5\%\), most of it within the first 100 steps. Adding semantically distinct queries raises coverage and validation accuracy together, until 16 queries reach \(98.9\%\) and match full-data training. Yet alignment slows at a similar pace whether O
    
[^7]: Terminal-Universe：将智能体轨迹转化为可扩展的终端环境

    Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments

    [https://arxiv.org/abs/2609.04148](https://arxiv.org/abs/2609.04148)

    Terminal-Universe 通过重放智能体轨迹中记录的文件操作，直接从已有轨迹重建可执行的终端环境，将海量积累的轨迹转化为可复用、可扩展的环境，用于合成新任务并提供执行反馈，解决了智能体后训练中环境稀缺的问题。

    

    随着基于终端的代码智能体日益普及，智能体轨迹已大规模积累，而真实、可执行的环境仍然稀缺。然而，环境才是智能体后训练真正所需的：每个环境可以被反复查询以生成大量可验证的任务，并提供执行反馈，而轨迹只是单一的冻结演示。与其从头生成环境，我们观察到现有轨迹中的工具执行历史揭示了其运行环境的结构与内容，使得从轨迹本身重建这些环境成为可能。因此，我们提出了 Terminal-Universe，这是一个将每条轨迹转化为可复用环境的框架，并对环境进行探索以合成新任务和延续交互。具体而言，Terminal-Universe 通过重放轨迹中记录的文件操作，恢复智能体修改之前的每个文件状态……（原文摘要此处截断）

    arXiv:2609.04148v1 Announce Type: new  Abstract: As terminal-based code agents become prevalent, agent trajectories have accumulated at scale, while realistic, executable environments remain scarce. However, environments are what agent post-training actually requires: each can be re-queried into many verifiable tasks and provides execution feedback, whereas a trajectory is a single frozen demonstration. Rather than generating environments from scratch, we observe that the tool-execution history in existing trajectories exposes the structure and contents of the environments in which they ran, making it possible to reconstruct those environments from the trajectories themselves. Thus, we introduce Terminal-Universe, a framework which turns each trajectory into a reusable environment and explores it for synthesizing new tasks and continued interactions. Specifically, Terminal-Universe replays the file operations recorded in a trajectory to restore each file before the agent modified it, y
    
[^8]: 顺序优于联合：论在线策略蒸馏与RLVR的相互作用

    Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR

    [https://arxiv.org/abs/2609.04108](https://arxiv.org/abs/2609.04108)

    先蒸馏后强化学习的两阶段训练方案在推理任务上持续优于纯OPD、纯RLVR及所有联合优化方法，因为OPD先扩大学生对教师解的覆盖范围、RL再在其内锐化，而联合训练会导致两种信号相互干扰。

    

    可验证奖励强化学习（RLVR）和在线策略蒸馏（OPD）已成为对推理大语言模型进行后训练的两种主流方法。先前的工作利用OPD的密集token级监督来补充稀疏的RL奖励，在单个步骤内融合这两种信号：要么作为加权加性组合，要么作为对RL优势的教师调制重缩放。在本文中，我们展示了一个简单的两阶段方案——先OPD后RL——在逻辑和数学推理基准上持续优于纯OPD、纯RLVR以及所有此类联合基线方法。除了实证结果外，我们还通过pass@$k$行为、学习动态和参数更新对这一现象提供了系统性的理解，并得出一个一致的解释：OPD扩大了学生对教师支持解的覆盖范围，而RL则在该支持范围内进行锐化，同时联合优化这两种信号会导致它们相互干扰。

    arXiv:2609.04108v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) and on-policy distillation (OPD) have emerged as two dominant methods for post-training reasoning LLMs. Prior work uses OPD's dense token-level supervision to complement the sparse RL reward, fusing the two signals within a single step: either as a \emph{weighted-additive combination} or a \emph{teacher-modulated rescaling} of the RL advantage. In this paper, we show that a simple two-stage scheme, OPD-then-RL, consistently outperforms pure OPD, pure RLVR, and all such joint baselines across logic and math reasoning benchmarks. Beyond the empirical results, we further provide a systematic understanding of this through pass@$k$ behavior, learning dynamics, and parameter updates, yielding a consistent explanation: OPD expands the student's coverage of teacher-supported solutions and RL sharpens within that support, while jointly optimizing the two signals causes them to interfere.To p
    
[^9]: CORE：通过重排序器蒸馏改进MLLM嵌入中的组合推理

    CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation

    [https://arxiv.org/abs/2609.04083](https://arxiv.org/abs/2609.04083)

    CORE通过将交叉注意力重排序器的细粒度组合判断以列表式Rank-KL目标蒸馏到嵌入模型中，显著提升了MLLM嵌入模型的组合推理能力，其效果优于对比学习和CoSENT。

    

    基于MLLM的嵌入模型在组合检索方面仍然存在局限，常常无法区分包含相同概念但属性-对象绑定不同的场景。然而，当同一骨干网络被用作交叉注意力重排序器时，却能够解决这类区分问题，这促使我们将其组合判断蒸馏到嵌入模型中。我们提出了CORE，该方法合成了跨越五个组合匹配级别的候选列表，并引入Rank-KL目标函数，训练嵌入模型复现重排序器的细粒度排序。我们进一步提出了一种分级评估协议，并在相同的数据和调参预算下比较了对比学习、成对式CoSENT和列表式Rank-KL。比较结果表明，CoSENT和Rank-KL都比对比学习更有效地利用了多级别监督，其中Rank-KL取得了最强的整体性能。在三个组合推理基准上……（摘要截断）

    arXiv:2609.04083v1 Announce Type: cross  Abstract: MLLM-based embedding models remain limited in compositional retrieval, often failing to distinguish scenes containing the same concepts but different attribute-object bindings. Yet the same backbone can resolve such distinctions when used as a cross-attentive reranker, motivating us to distill its compositional judgments into the embedding model. We propose CORE, which synthesizes candidate lists spanning five compositional matching levels and introduces a Rank-KL objective that trains the embedding model to reproduce the reranker's fine-grained ranking. We further introduce a graded evaluation protocol and compare contrastive learning, pairwise CoSENT, and listwise Rank-KL under the same data and tuning budget. Our comparison shows that both CoSENT and Rank-KL use the multi-level supervision more effectively than contrastive learning, with Rank-KL achieving the strongest overall performance. Across three compositional reasoning benchm
    
[^10]: 当模型编辑过多：论最小代码编辑的保真度

    When Models Edit Too Much: On the Fidelity of Minimal Code Edits

    [https://arxiv.org/abs/2609.04061](https://arxiv.org/abs/2609.04061)

    该研究揭示了前沿大语言模型在修复代码时普遍存在“过度编辑”问题（即使如GPT-5.5这样的强模型也不例外），并提出通过一条简单的保留指令即可显著减少不必要的代码改动、降低认知复杂度，同时提升修复准确率。

    

    大语言模型越来越多地被用于编辑现有代码，但仅仅正确是不够的：有用的修复还应当是最小化的、可审查的，并且忠实于原始实现。我们研究了“过度编辑”现象，即模型重写代码的范围超出修复缺陷所需的趋势。我们基于400个BigCodeBench问题构建了一个评估框架，通过向参考解答中注入受控的AST级（抽象语法树级）破坏，为每个修复任务提供一个已知的最小补丁。研究发现，在各类前沿大语言模型中，过度编辑现象普遍存在，即使是在GPT-5.5这样的强大模型中也是如此：高Pass@1可能与不必要的巨大编辑和新增认知复杂度并存。一条保留指令能够显著减少这种行为，将平均超额Levenshtein距离从0.195降至0.131，减少26.6%的新增认知复杂度，并使Pass@1提高2.3个百分点。然而，这些收益并非简单地源于更大的推理（摘要在此处被截断）

    arXiv:2609.04061v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to edit existing code, but correctness alone is not enough: useful repairs should also be minimal, reviewable, and faithful to the original implementation. We study over-editing, the tendency of a model to rewrite code beyond what is required to fix a bug. We construct an evaluation framework from 400 BigCodeBench problems by injecting controlled AST-level corruptions into reference solutions, giving each repair task a known minimal patch. Across frontier LLMs, over-editing is widespread even among strong models like GPT-5.5: high Pass@1 can coexist with unnecessarily large edits and added cognitive complexity. A preservation instruction substantially reduces this behavior, lowering average excess Levenshtein distance from 0.195 to 0.131, reducing added cognitive complexity by 26.6%, and increasing Pass@1 by 2.3 points. However, these gains do not simply follow from a larger reasoning 
    
[^11]: 作为决策空间的翻译：低资源方言生成的多智能体视角

    Translation as a Decision Space: A Multi-Agent Perspective on Low-Resource Dialect Generation

    [https://arxiv.org/abs/2609.04048](https://arxiv.org/abs/2609.04048)

    本文将翻译重构为由多个自主智能体探索的结构化决策空间，把不同翻译路径建模为智能体，并将智能体间的分歧作为可解释的行为信号，应用于土耳其语—叙利亚阿拉伯语这一低资源方言翻译的实证研究。

    

    神经机器翻译（NMT）系统通常对每个输入只产生单一输出，这掩盖了多语言解码中隐含存在的多种备选决策路径。这种不透明性在低资源方言环境中尤为成问题，因为在该环境中，多种在语言学上都成立的译文实现方式可能在词汇地道性、语域和结构稳定性上存在差异。我们提出将翻译重新构建为一个由自主翻译智能体探索的结构化决策空间。我们不分析单一输出，而是将不同的翻译路径建模为在共享多语言骨干上运行的智能体。智能体之间的分歧不再被视为错误，而是被视为一种可解释的行为信号。我们使用三个智能体对土耳其语—叙利亚阿拉伯语翻译进行了实证研究：（1）零样本直接翻译，（2）通过轻量级微调实现的方言稳定化翻译，以及（3）通过英语进行的枢纽语翻译。

    arXiv:2609.04048v1 Announce Type: cross  Abstract: Neural machine translation (NMT) systems typically produce a single output per input, obscuring the alternative decision trajectories implicitly available within multilingual decoding. This opacity becomes particularly problematic in low-resource dialect settings, where multiple linguistically valid realizations may differ in lexical authenticity, register, and structural stability. We propose reframing translation as a structured decision space explored by autonomous translation agents. Instead of analyzing a single output, we model distinct translation pathways as agents operating over a shared multilingual backbone. Inter-agent divergence is treated not as error but as an interpretable behavioral signal. We conduct an empirical study on Turkish--Syrian Arabic translation using three agents: (1) zero-shot direct translation, (2) dialect-stabilized translation via lightweight fine-tuning, and (3) pivot translation through English. Eva
    
[^12]: 骰子法：大语言模型品牌推荐重复查询审计的标准化协议

    The Dice Roll Method: A Standardized Protocol for Repeated-Query Auditing of Large Language Model Brand Recommendations

    [https://arxiv.org/abs/2609.04047](https://arxiv.org/abs/2609.04047)

    本文提出并形式化了“骰子法”——一个基于温度缩放核采样生成模型的可复用标准化协议，通过将响应方差分解为多个成分并提供完整的统计分析技术栈，为大语言模型品牌推荐的重复查询审计建立了系统方法。

    

    背景：研究人员越来越多地使用重复相同的提示词来审计大语言模型（LLM）品牌推荐中的随机变异，然而目前尚无用于设定迭代次数、选择稳定性指标或建立可靠性阈值的标准化协议。目标：我们将骰子法形式化为一个可复用的协议，用于LLM品牌推荐的重复查询审计，该协议建立在温度缩放核采样的生成模型基础之上。方法：将总响应方差分解为采样、提示措辞、运行间和模型版本等成分。该技术栈包括：以迭代作为重复测量的负二项混合模型；作为无分布效应量的Cliff's delta；保留依赖结构的自助法；基于模拟的统计功效分析；概化理论分解；以及对固定快照的漂移诊断。我们重新分析了五项品牌推荐审计研究：约190（摘要在此处截断）。

    arXiv:2609.04047v1 Announce Type: cross  Abstract: Background: Researchers increasingly use repeated identical prompts to audit stochastic variation in large language model (LLM) brand recommendations, yet no standardized protocol exists for setting iteration counts, selecting stability metrics, or establishing reliability thresholds. Objective: We formalize the Dice Roll Method as a reusable protocol for repeated-query auditing of LLM brand recommendations, grounded in a generative model of temperature-scaled nucleus sampling. Methods: Total response variance is decomposed into sampling, prompt-phrasing, run-to-run, and model-version components. The stack: a negative-binomial mixed model with iterations as repeated measures; Cliff's delta as the distribution-free effect size; dependence-preserving bootstrap; simulation-based power; a generalizability-theory decomposition; drift diagnostics on pinned snapshots. We reanalyse five brand-recommendation auditing studies: approximately 190,
    
[^13]: 可编辑的视觉设计

    Editable Visual Design

    [https://arxiv.org/abs/2609.04034](https://arxiv.org/abs/2609.04034)

    该论文提出“可编辑的视觉设计”新范式，以编码智能体为核心，将VLM作为“创意大脑”进行需求理解与审美判断，将图像生成模型作为按需的“视觉世界模拟器”合成独立资产，并通过“先想象、后行动”的闭环工作流编写原生HTML/CSS，实现支持图层级精确后编辑的视觉设计。

    

    尽管 GPT-Image-2 和 Nano-Banana 等扩散基础模型展现出卓越的视觉表现力，但它们的端到端生成本质上会产生文本易出错的扁平位图，导致无法进行图层级的后编辑。相反，通过编码智能体进行基于代码的视觉生成能够提供精确的布局控制和相互解耦的图层，但仍受限于缺乏全局审美直觉以及编写复杂视觉资产的困难。为解决这些问题，我们提出了“可编辑的视觉设计”，这是一种由编码智能体驱动的新范式。我们将视觉语言模型（VLM）指定为负责需求理解、任务规划和审美判断的“创意大脑”，同时利用图像生成模型作为按需调用的“视觉世界模拟器”来合成独立的视觉资产。在“先想象、后行动”的闭环工作流下，智能体生成相互隔离的资产、编写原生 HTML/CSS 代码，并进行迭代式优化。

    arXiv:2609.04034v1 Announce Type: cross  Abstract: While diffusion base models such as GPT-Image-2 and Nano-Banana exhibit remarkable visual expressiveness, their end-to-end generation inherently yields flattened bitmaps with error-prone text, precluding layer-wise post-editing. Conversely, code-based visual generation via Coding Agents provides precise layout control and decoupled layers, yet remains constrained by a lack of global aesthetic intuition and the difficulty of coding complex visual assets.   To address this, we propose Editable Visual Design, a new paradigm driven by a Coding Agent. We designate the VLM as the ``creative brain'' for requirement comprehension, task planning, and aesthetic judgment, while utilizing the image generation model as an on-demand ``visual world simulator'' to synthesize standalone visual assets. Operating under an ``imagine first, then act'' closed-loop workflow, the agent generates isolated assets, writes native HTML/CSS, and iteratively refines
    
[^14]: 指令复制作为一种推理时控制原语

    Instruction Duplication as an Inference-Time Control Primitive

    [https://arxiv.org/abs/2609.04024](https://arxiv.org/abs/2609.04024)

    在推理时仅简单复制一遍程序化指令——无需重新训练或修改解码——即可将七个模型在医学选择题上通过全部八项诊断测试的比例从90.22%提升至93.17%，同时保持最终答案准确率不变。

    

    程序化指令遵循是可控语言模型系统的基本要求，尤其是当生成的轨迹需要在下游被检查或修复时。我们提出了“指令复制”，这是一种极简的黑盒推理时控制方法，仅重复程序化指令本身，无需重新训练或修改解码过程。在七个指令微调模型、300道医学选择题、八种放置条件以及16,800次计划生成的实验中，将指令从一份复制为两份，使确定性的All-8诊断——即响应通过全部八项可观测测试——从90.22%提升至93.17%（提高2.95个百分点），消除了单份复制后剩余失败案例的30.2%。预修正的TF-IDF召回率从73.44%上升至74.81%（提高1.38个百分点；经Holm校正后p < .001），而最终答案准确率恰好保持在60.21%不变。过早承诺现象从1.52%增加至2.30%（p_Holm = .00536）。一项盲测挑战……（原文在此处截断）

    arXiv:2609.04024v1 Announce Type: new  Abstract: Procedural instruction following is a basic requirement for controllable language-model systems, especially when generated trajectories are inspected or repaired downstream. We introduce instruction duplication, a minimal black-box inference-time control that repeats only the procedural instruction, without retraining or decoding changes. Across seven instruction-tuned models, 300 medical multiple-choice questions, eight placement conditions, and 16,800 scheduled generations, moving from one to two copies raises the deterministic All-8 diagnostic--responses passing all eight observable tests--from 90.22% to 93.17% (+2.95 percentage points), eliminating 30.2% of the failures remaining after one copy. Pre-provisional TF-IDF recall rises from 73.44% to 74.81% (+1.38 points; Holm-adjusted p < .001), while final-answer accuracy remains exactly 60.21%. Premature commitment increases from 1.52% to 2.30% (p_Holm = .00536). A blinded challenge au
    
[^15]: 表征对齐使语言模型获得可泛化的安全性

    Representational alignment yields generalizable safety in language models

    [https://arxiv.org/abs/2609.04022](https://arxiv.org/abs/2609.04022)

    提出表征相似性优化方法，将大语言模型的内部潜在表征与人类道德判断的原型化归类结构直接对齐，从而使安全对齐能够泛化到以陌生或对抗性形式表述的有害意图。

    

    对大语言模型（LLM）进行对齐对于其安全部署至关重要。当前的对齐方法主要优化可观察的响应，然而当同样的有害意图以人类能够轻易识别的不熟悉或对抗性形式重新表述时，模型仍然容易受到攻击。原型理论为人类的这种适应性提供了一种解释：人类概念是围绕中心案例来表征的，新实例则根据其相对于这些原型的分级典型性来进行归类。本研究表明，这种道德概念的归类能力在当前的大语言模型中仅有微弱的保留。在23个LLM上的实验显示，模型往往无法区分相互对立的道德类别，也无法在每个类别内部保持细粒度的典型性。这些缺陷在不同的参数规模和对齐阶段中持续存在。我们提出了表征相似性优化方法，将LLM中的潜在表征与人类道德判断中表达的归类方式直接对齐……

    arXiv:2609.04022v1 Announce Type: cross  Abstract: Aligning large language models (LLMs) is essential for their safe deployment. Current alignment methods mainly optimize observable responses, yet models remain vulnerable when the same harmful intent is recast in unfamiliar or adversarial forms that humans can easily recognize. Prototype theory offers an account of this adaptability. Human concepts are represented around central cases, and new instances are categorized according to their graded typicality relative to these prototypes. Here we show that such categorization of moral concepts is weakly preserved in current LLMs. Across 23 LLMs, models often failed to distinguish opposed moral categories or preserve fine-grained typicality within each category. These deficits persist across parameter sizes and alignment stages. We developed representational similarity optimization, which directly aligns the latent representations in LLMs with the categorization expressed in human moral jud
    
[^16]: 免对齐文本-Audiobox：面向语音配音与全双工对话合成

    Alignment-Free Text-Audiobox for Voice Dubbing and Full-Duplex Dialogue Synthesis

    [https://arxiv.org/abs/2609.03992](https://arxiv.org/abs/2609.03992)

    该论文提出免对齐Text-Audiobox（Text-AB），通过DAC-VAE潜在扩散表示（压缩率较EnCodec提升10倍以上）、基于交叉注意力的免对齐文本建模，以及3B参数与48万小时语音的大规模训练，构建了一个支持语音配音与全双工对话合成的统一框架。

    

    我们提出了免对齐文本-Audiobox（Text-AB），这是一个用于高质量语音配音和全双工对话合成的统一框架。Text-AB建立在采用流匹配目标训练的扩散Transformer之上，并在三个维度上区别于Audiobox系统。首先，它在潜在扩散框架中运行，使用DAC-VAE特征将48 kHz波形编码为25 Hz的潜在序列，相比之前的EnCodec表示实现了超过10倍的压缩率，同时提升了重合成质量。其次，Text-AB是免对齐的：它通过现成的文本编码器直接处理原始文本，并通过交叉注意力机制学习文本-语音对齐，从而无需强制对齐和显式时长预测。第三，我们大幅扩展了模型和数据的规模，在48万小时的单语语音上预训练了一个30亿参数的模型，随后在三个下游任务上进行监督微调：跨语言语音配音、全（双工对话合成等，摘要原文在此处截断）。

    arXiv:2609.03992v1 Announce Type: new  Abstract: We present Alignment-Free Text-Audiobox (Text-AB), a unified framework for high-quality voice dubbing and full-duplex dialogue synthesis. Building on a Diffusion Transformer trained with a flow-matching objective, Text-AB departs from the Audiobox system along three dimensions. First, it operates in a latent diffusion framework using DAC-VAE features that encode 48 kHz waveforms into a 25 Hz latent sequence, giving over 10x higher compression than previous EnCodec representations while improving resynthesis quality. Second, Text-AB is alignment-free: it consumes raw text via an off-the-shelf text encoder and learns text-speech alignment through cross-attention, removing the need for forced alignment and explicit duration prediction. Third, we scale model and data substantially, pretraining a 3B-parameter model on 480k hours of monolingual speech, followed by supervised fine-tuning on three downstream tasks: cross-lingual voice dubbing, f
    
[^17]: IchthyoNoma：零样本生物视觉-语言模型在孟加拉国淡水鱼识别中的命名法与上下文敏感性

    IchthyoNoma: Nomenclature and Context Sensitivity of Zero-Shot Biological Vision--Language Models for Bangladeshi Freshwater Fish Recognition

    [https://arxiv.org/abs/2609.03985](https://arxiv.org/abs/2609.03985)

    该研究审计了零样本生物视觉-语言模型在孟加拉国淡水鱼识别上的表现，发现其准确率不仅源于生物学专业化知识，还强烈依赖于命名语言（英文/学名/孟加拉语）的选择与图像上下文，其中BioCLIP2在英文提示下表现优异，但孟加拉语提示下接近随机水平。

    

    零样本视觉-语言模型（VLM）日益被用作无需训练的物种识别器，但所报告的准确率可能反映的不仅仅是视觉物种知识本身。我们在来自孟加拉国两个数据源（共10,321张图像）的七个淡水鱼类别上，对CLIP、BioCLIP、BioCLIP2以及多语言Jina CLIP v2对照组进行了审计。BioCLIP2在BFF-15数据集上使用英文常用名达到72.36%的准确率，在SylFishBD上使用学名达到68.91%，而通用CLIP分别仅为25.15%和14.40%。BioCLIP2的孟加拉语提示词在平衡准确率上接近随机水平（14.22-14.29%）；Jina将孟加拉语的判别能力部分恢复至21.89%和16.36%，但使用纯孟加拉语名称时，两个数据源上的准确率均回落到14.29%。在SylFishBD上进行的配对干预实验显示，轻度模糊无显著影响，较强模糊/灰度遮罩带来适度损失，白色遮罩产生更大的伪影效应，且效果存在强烈的物种依赖性。因此，零样本生物VLM的得分共同反映了生物领域专业化、多语言……（原文摘要在此处截断）

    arXiv:2609.03985v1 Announce Type: cross  Abstract: Zero-shot vision-language models (VLMs) are increasingly used as training-free species recognizers, but reported accuracy can reflect more than visual species knowledge. We audit CLIP, BioCLIP, BioCLIP2, and a multilingual Jina CLIP v2 control on seven freshwater-fish categories from two Bangladeshi sources (10,321 images). BioCLIP2 reaches 72.36% on BFF-15 with English common names and 68.91% on SylFishBD with scientific names, versus 25.15% and 14.40% for generic CLIP. BioCLIP2 Bengali prompts are near chance in balanced accuracy (14.22-14.29%); Jina partially recovers Bengali discrimination to 21.89% and 16.36%, but bare Bengali names return to 14.29% on both sources. Paired SylFishBD interventions show no significant weak-blur effect, modest losses from stronger blur/gray masking, a larger white-mask artifact, and strong species dependence. Zero-shot biological VLM scores therefore jointly reflect biological specialization, multili
    
[^18]: 研究大型语言模型分析糖尿病食谱的能力

    Investigating the Ability of Large Language Models to Analyze Recipes for Diabetes

    [https://arxiv.org/abs/2609.03967](https://arxiv.org/abs/2609.03967)

    本研究构建了包含7607个食谱的糖尿病基准数据集，并采用三种融合医学饮食指南的提示策略，系统评估了大型语言模型分析食谱对糖尿病适用性的能力。

    

    多项研究已经评估了大型语言模型（LLM）在膳食规划方面的能力，并取得了积极成果。这些模型能够处理自然语言输入，并利用预训练中学习到的知识来生成膳食计划。在这项工作中，我们研究了大型语言模型分析给定食谱是否适合糖尿病患者的能力。LLM面临的主要挑战是检索相关的糖尿病饮食指南、将食谱分解为食材和烹饪方法，并应用这些指南来判断食谱的适用性。为了研究这些挑战，我们采用了三种提示方法：（i）直接查询提示（Direct Query Prompt）；（ii）上下文引导提示（Context-Guided Prompt）；（iii）示例上下文提示（Exemplary Context Prompt），这些提示融合了来自医学来源的不同层级的糖尿病饮食指南。我们为这项研究构建了一个专门的基准数据集，包含7607个食谱，其中3807个适合糖尿病患者。

    arXiv:2609.03967v1 Announce Type: cross  Abstract: Several studies have evaluated the ability of Large Language Models (LLMs) for meal planning, yielding positive outcomes. These models can process natural language inputs and leverage learned knowledge from their pretraining to generate meal plans. In this work, we investigate the ability of LLMs to analyze the suitability of given recipes for diabetes. The primary challenge for LLMs is to retrieve relevant dietary guidelines for diabetes, decompose recipes into ingredients and cooking methods, and apply these guidelines to determine the recipe's suitability. To study these challenges, we employ three kinds of prompts namely, (i) Direct Query Prompt (ii) Context-Guided Prompt, and (iii) Exemplary Context Prompt that incorporate different levels of diabetes dietary guidelines from medical sources. We introduce a benchmark dataset curated for this investigation consisting of 7607 recipes that include 3807 recipes suitable for diabetes an
    
[^19]: FiMI Banking：一个印度零售银行业的主权模型

    FiMI Banking: A Sovereign Model for Indian Retail Banking

    [https://arxiv.org/abs/2609.03960](https://arxiv.org/abs/2609.03960)

    该论文构建了面向印度零售银行业、基于真实银行文档和工具的受控对话环境FiMI Banking，并通过偏好优化和可验证奖励强化学习两种后训练方法，分别将安全拒绝行为从52%提升至80%、边缘案例性能从0.509提升至0.718。

    

    银行需要能够回答产品问题、协助客户处理账户相关请求、并在严格的运营和监管约束下安全运行的对话式系统。通用语言模型无法可靠地满足这些要求。当任务需要基于可靠信息作答、正确使用工具或谨慎处理银行特有的敏感情况时，它们的表现不尽如人意。我们提出了FiMI Banking，一个受控的印度零售银行环境。我们基于经过审查的银行文档、结构化基准真值、合成客户背景和银行工具构建了该环境。我们评估了两种后训练方法：用于响应级行为的偏好优化，以及用于多轮工具使用任务的带可验证奖励的强化学习。偏好优化显著改善了安全行为：超范围请求的拒绝率从52%提升至80%。强化学习将边缘案例性能从0.509提升至0.718。

    arXiv:2609.03960v1 Announce Type: new  Abstract: Banks need conversational systems that can answer product questions, assist customers with account-related requests, and operate safely within strict operational and regulatory constraints. General-purpose language models do not reliably meet these requirements. They fall short when a task requires grounded information, correct tool use, or cautious handling of bank-specific sensitive situations. We introduce FiMI Banking, a controlled Indian retail-banking setting. We build it from vetted banking documents, structured ground truth, synthetic customer backgrounds, and banking tools. We evaluate two post-training approaches: preference optimization for response-level behavior, and reinforcement learning with verifiable rewards for multi-turn tool-use tasks. Preference optimization improves safe behavior substantially: out-of-scope refusal rises from 52% to 80%. Reinforcement learning improves edge-case performance from 0.509 to 0.718 and 
    
[^20]: 面向代码大语言模型中可靠与对抗性测试生成的两阶段强化学习

    Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code LLMs

    [https://arxiv.org/abs/2609.03955](https://arxiv.org/abs/2609.03955)

    该论文提出了一种两阶段强化学习框架TCS，第一阶段生成与参考解一致的可靠测试用例，第二阶段学习针对模型当前失败模式的对抗性反例测试，从而有效提升代码大模型的测试生成质量和代码性能。

    

    强化学习（RL）通过可执行反馈极大地推动了基于大语言模型（LLM）的代码生成。编程问题的反馈主要来自特定的测试用例，而高质量的测试用例往往十分稀缺，因为它们既需要具备可靠性，又需要具备区分性。因此，我们转而研究利用学习到的模型自动生成测试用例。我们发现这本质上是一个对抗性强化学习问题：模型需要根据求解器当前的失败模式生成有效的测试用例作为反例。我们提出了测试用例缩放，这是一个用于有效测试生成的两阶段强化学习框架。两个阶段均从一个滚动策略对齐缓冲区中训练测试生成器：第一阶段生成与参考解一致的测试用例，第二阶段将缓冲区限制为当前的失败模式并学习生成反例测试。在TACO和LiveCodeBench数据集上，TCS同时提升了pass@1和推理性能。

    arXiv:2609.03955v1 Announce Type: new  Abstract: Reinforcement learning (RL) has substantially advanced code generation with large language models (LLMs) through executable feedback. The feedback for coding problems mainly comes from specific test cases, where high-quality test cases are often scarce since they should be both sound and discriminative. We thus turn to study the auto-generation of test cases using the learned model. We find this is naturally an adversarial RL problem: the model is expected to generate effective test cases as counterexamples, depending on the solver's current failure modes. We propose Test Cases Scaling (TCS), a two-stage RL framework for effective test generation. Both stages train a test generator from a rolling policy-aligned buffer: Stage 1 generates tests consistent with the reference solution, and Stage 2 restricts the buffer to current failure modes and learns counterexample tests. Across TACO and LiveCodeBench, TCS improves both pass@1 and inferen
    
[^21]: 超越多数投票：用于医学幻觉检测的多视角裁定方法

    Beyond Majority Vote: Multi-Perspective Adjudication for Medical Hallucination Detection

    [https://arxiv.org/abs/2609.03953](https://arxiv.org/abs/2609.03953)

    该研究提出一种结合首轮标注、LLM裁判候选发现和双重裁定的多视角医学幻觉检测标注框架，证明单一方法均会遗漏事实错误，而多来源裁定可提升医学事实核查基准的完整性。

    

    arXiv:2609.03953v1 公告类型：新 摘要：理解聊天机器人生成文本中事实错误的出现频率，并评估检测这些错误的系统，对于确定聊天机器人的安全性至关重要。然而，事实错误检测通常被视为单次、单标注者的标注问题。在长篇聊天机器人回复中，事实错误往往很微妙，并嵌入在基本正确的文本之中。我们对医学相关的聊天机器人回复开展了一项多视角标注研究，结合了首轮人工标注、大语言模型作为裁判的候选错误发现，以及两种形式的裁定机制：医学专家裁定和基于证据的事实核查。研究发现，首轮标注者经常遗漏后续被裁定者验证为正确的事实错误；LaJ（LLM作为裁判）改善了候选错误的发现，但其本身并不充分，会遗漏标注者能够捕捉到的事实错误。此外，我们还发现裁定者之间存在分歧，这表明对多个候选来源进行裁定可以提高基准数据集的完整性。

    arXiv:2609.03953v1 Announce Type: new  Abstract: Understanding the frequency of factual errors in chatbot-generated text and evaluating systems that detect these errors is critical for determining chatbot safety. Yet factual-error detection is often treated as a single-pass, single-annotator labeling problem. In long-form chatbot responses, factual errors can be subtle and embedded within mostly correct text.   We develop a multi-perspective annotation study of medically relevant chatbot responses, combining first-pass annotation, LLM-as-a-Judge (LaJ) candidate discovery, and two forms of adjudication: medical-expert and evidence-based fact-checking. First-pass annotators frequently miss factual errors later validated by adjudicators. LaJ improves candidate discovery, but is insufficient on its own: It misses factual errors that annotators catch. We also find disagreement among adjudicators, suggesting that adjudication over multiple candidate sources can improve benchmark completeness
    
[^22]: VestigeKV：NoPE-MLA的KV缓存通过一个残余分支携带其自身的淘汰信号

    VestigeKV: The NoPE-MLA KV Cache Carries Its Own Eviction Signal in a Vestigial Branch

    [https://arxiv.org/abs/2609.03949](https://arxiv.org/abs/2609.03949)

    VestigeKV发现NoPE MLA模型KV缓存中的64维解耦RoPE残余分支已被训练重新利用为显著性信号，据此提出无需训练和量化的查询无关缓存淘汰方法，在8-32倍压缩下几乎不损失检索精度。

    

    问题所在：一个长期存在的KV缓存必须在将要读取它的查询出现之前就被压缩；基于已观测注意力进行选择的方法（H2O、SnapKV）在这种情况下会失效（在NoPE MLA模型上针检索率仅为0.00-0.33），因为token的重要性尚未被观测到。方法：在Kimi Linear模型上，VestigeKV利用缓存自身已携带的与查询无关的信号进行淘汰：即64维解耦分支——这是RoPE的残余结构，NoPE训练将其重新利用为显著性通道。只需读取每行的11%，它将缓存划分为两个层级：top-m行保留在注意力层；其余所有行——精确保存、从不删除——移动到GPU驻留的存档中，每一步都可以通过经过验证的触发器访问。无需训练、无需量化、无需更改权重或内核。代价：无可测量的损失：在8k到65k上下文长度范围内，8倍压缩下检索率保持在1.00，32倍压缩下为0.92，与全行选择方法零差距。注意力层级仅占Kimi Linear每token 8.1 KB缓存中的0.25 KB。

    arXiv:2609.03949v1 Announce Type: cross  Abstract: The problem. A long-lived KV cache must be compressed before the queries that will read it exist; selection by observed attention (H2O, SnapKV) collapses there (0.00-0.33 needle retrieval on a NoPE MLA model), because a token's importance has not yet been observed. The method. On Kimi Linear, VestigeKV evicts by a query-independent signal the cache already carries: the 64-dimensional decoupled branch, a vestige of RoPE that NoPE training repurposes into a salience channel. Reading 11% of each row, it partitions the cache: the top-m rows stay in the attended tier; every other row moves -- exactly, never deleted -- to a GPU-resident archive reachable per step by a certified trigger. No training, no quantization, no weight or kernel change. Cost. Nothing measurable: retrieval holds at 1.00 under 8x and 0.92 under 32x from 8k to 65k context, zero gap to full-row selection. The attended tier is 0.25 KB of Kimi Linear's 8.1 KB per-token cach
    
[^23]: 更多批评并不等于更好的审稿：EquiReview-R

    More Criticism Does Not Make a Better Review: EquiReview-R

    [https://arxiv.org/abs/2609.03943](https://arxiv.org/abs/2609.03943)

    该论文提出EquiReview-R框架，将AI辅助审稿重构为基于证据的结构化关注点集细化过程，把遗漏与过度批评视为两种独立风险分别纠正，并借助证据关联的轨迹语料库证明审稿修订必须先于进一步的问题搜索。

    

    AI审稿人如今能够产出大量具体的批评意见，但更多的批评并不一定意味着更好的审稿。一份审稿意见可能遗漏了具有实质影响的缺陷，也可能保留了现有证据无法支持的指控。这两类失败需要截然相反的纠正措施，然而面向生成的系统以及总体性的评估指标却掩盖了这一区别。因此，我们将AI辅助审稿重新表述为对结构化关注点集合进行证据引导的细化过程，并将遗漏与过度批评视为两种相互独立的风险。基于这一表述，我们提出了EquiReview-R，它结合局部化证据解决现有关注点，从独立视角和以审稿意见为条件的视角搜索可能缺失的问题，并返回停止、继续或推迟的决策。为了揭示促使这一设计的失败模式，我们构建了一个与证据关联的轨迹语料库。其回溯性分析表明了为什么修订必须先于进一步的问题搜索：在高……（摘要至此截断）

    arXiv:2609.03943v1 Announce Type: new  Abstract: AI reviewers can now produce many specific criticisms, but more criticism is not necessarily a better review. A review may miss a consequential weakness or retain an allegation that available evidence does not support. These failures require opposite corrections, yet generation-oriented systems and aggregate measures obscure the distinction. We therefore recast AI-assisted review as evidence-guided refinement of a structured concern set, with omission and overcritique treated as separate risks. Building on this formulation, we introduce EquiReview-R, which resolves existing concerns against localized evidence, searches for missing issues from independent and review-conditioned perspectives, and returns stop, continue, or defer. To expose the failure mode that motivates this design, we construct an evidence-linked trajectory corpus. Its retrospective analysis shows why revision must precede further search: nearly all concerns in a high-re
    
[^24]: Headroom-Drift Replay：GRPO中一种实现原则性重放控制的原语

    Headroom-Drift Replay: A Primitive for Principled Replay Control in GRPO

    [https://arxiv.org/abs/2609.03941](https://arxiv.org/abs/2609.03941)

    该论文提出了一种面向GRPO的组级重放控制原语Headroom-Drift Replay，通过Headroom按剩余学习价值排序、Drift按策略兼容性门控来复用历史轨迹，在不改变在线数据流、不增加额外训练机制的前提下加速RL后训练，从而将重放本身的贡献与复杂训练流程解耦。

    

    基于强化学习的推理模型后训练正日益受到重复性全新轨迹生成（rollout）的瓶颈制约，尤其是在智能体环境中，环境交互主导了墙钟时间成本。重放可以通过复用过去的轨迹来减轻这一负担，但现有方法通常将重放嵌入到涉及探索、经验重构或混合策略优化的更大训练流程中，这使得重放本身的贡献难以被隔离。我们提出一个聚焦的问题：仅凭原则性的重放选择究竟能走多远？我们提出了Headroom-Drift Replay，一种面向GRPO的组级重放控制原语，它将复用拆分为两个决策：Headroom（头空间）根据剩余学习价值对存储的轨迹组进行排序，而Drift（漂移）则根据与当前策略的兼容性对其进行门控。新鲜的在线策略数据流保持不变，且该方法不引入任何辅助的生成或训练机制。在数学推理、多模……（原文摘要在此处截断）

    arXiv:2609.03941v1 Announce Type: cross  Abstract: RL-based post-training for reasoning models is increasingly bottlenecked by repeated fresh rollout generation, particularly in agentic settings where environment interaction dominates wall-clock cost. Replay can reduce this burden by reusing past trajectories, but existing methods typically embed it within larger training pipelines involving exploration, experience restructuring, or mixed-policy optimization. This makes replay's own contribution difficult to isolate. We ask a focused question: how far can principled replay selection alone go? We introduce Headroom-Drift Replay, a group-level replay control primitive for GRPO that separates reuse into two decisions. Headroom ranks stored groups by remaining learning value, while Drift gates them by compatibility with the current policy. The fresh on-policy stream remains unchanged, and the method adds no auxiliary generation or training machinery. Across mathematical reasoning, multimod
    
[^25]: 固定后缀依赖比率：量化拉脱维亚语外来词性属分配的双轨机制

    Fixed Suffix Dependency Ratio: Quantifying the Dual-Track Mechanism of Gender Assignment in Latvian Loanwords

    [https://arxiv.org/abs/2609.03930](https://arxiv.org/abs/2609.03930)

    本研究提出固定后缀依赖比率（FSDR）这一量化指标，揭示了拉脱维亚语英语外来词性属分配的双轨机制——阴性外来词显著依赖固定派生后缀，而阳性外来词集中于自由选择区域。

    

    现有研究反复观察到英语外来词在不同接受语言中倾向于聚集为阳性的现象，但由于固定的形态规则和默认分配常常被放在一起分析，这种模式的起源仍难以确定。本研究提出固定后缀依赖比率（FSDR），用以量化不同性属对固定派生后缀的依赖程度，并区分分布中的形态锚定与自由选择。通过考察1,832个拉脱维亚语名词词元类型，结果揭示了外来词系统内部显著的FSDR不对称性：阴性外来词显著更依赖固定派生后缀，而阳性外来词更多集中在自由选择区域。这一模式表现出外来词特异性，并且在当代使用中变得更加明显。因此，FSDR提供了一个量化框架

    arXiv:2609.03930v1 Announce Type: new  Abstract: Existing research has repeatedly observed the tendency for English loanwords to cluster in the masculine gender across different recipient languages, yet the origin of this pattern remains difficult to determine, as fixed morphological rules and default assignments are frequently analysed together. This study proposes the Fixed Suffix Dependency Ratio (FSDR) to quantify the degree of reliance on fixed derivational suffixes across different genders, and to distinguish between morphological anchoring and free-choice in distribution. By examining 1,832 Latvian noun lemma types, the results reveal a significant FSDR asymmetry within the loanword system: feminine loanwords rely significantly more on fixed derivational suffixes, while masculine loanwords are more concentrated in the free-choice zone. This pattern exhibits loanword specificity and has become more pronounced in contemporary usage. FSDR therefore provides a quantitative framework
    
[^26]: 替我发言：赋予大语言模型参与会议的情境感知能力

    Speak for Me: Giving LLMs the Situational Awareness to Participate in a Meeting

    [https://arxiv.org/abs/2609.03923](https://arxiv.org/abs/2609.03923)

    提出CAPA架构，通过感知器、预测器、控制器、生成器和重校准器的协作设计，赋予LLM追踪会议立场、话题覆盖和发言权的情境感知能力，解决其在代理缺席者参会时51.4%发言机会保持沉默的问题。

    

    在在线会议代理场景中，大语言模型（LLM）智能体无法识别何时该发言。由于缺乏结构化的方式来跟踪立场、话题覆盖度和发言权，它们错过了本应代表参与者发言的时机。在AMI语料库上，仅依靠提示词的代理在51.4%的缺席参与者发言机会上保持沉默。我们提出了CAPA（协作智能体预测架构），一种用于在线会议代理的架构。感知器根据每个观察到的发言轮次更新会议状态；预测器预测对话将如何继续；控制器决定是否发言以及表达哪个观点；生成器以参与者的语言风格组织所选内容的表达。两个评判器根据下一个实际观察到的发言轮次对预测和行动进行评分；重校准器则根据这些评判结果更新会议状态，以供未来的决策使用。为了评估在线会议代理，我们引入了一种片段级的评估协议，用于评估代理是否、何时以及表达了什么内容。

    arXiv:2609.03923v1 Announce Type: new  Abstract: In online meeting delegation, LLM agents fail to recognize when to speak. With no structured way to track stances, coverage, and floor, they miss the moments where they should contribute. Prompt-only delegates stay silent on 51.4% of the absent participant's talking opportunities on the AMI corpus. We present CAPA (Collaborative Agent Predictive Architecture), an architecture for online meeting delegation. A Perceiver updates the meeting state from each observed turn. A Predictor forecasts how the conversation will continue. A Controller decides whether to speak and which proposition to surface. A Generator phrases the chosen contribution in the participant's style. Two judges score the forecast and the action against the next observed turn. A Recalibrator updates the meeting state from those verdicts for future decisions. To evaluate online delegation, we introduce an episode-level protocol that scores whether, when, and what a delegate
    
[^27]: RuleMem：面向长期对话代理的主动规则记忆

    RuleMem: Active Rule Memory for Long-Term Conversational Agents

    [https://arxiv.org/abs/2609.03915](https://arxiv.org/abs/2609.03915)

    RuleMem提出了一种基于规则的主动记忆框架，通过从历史对话中归纳并验证自然语言霍恩子句来主动指导证据检索与推理，显著提升了长期对话问答代理的可靠性。

    

    长期对话中的问答代理必须对海量且在时间上分散的对话历史进行推理。然而，现有的记忆机制主要将过去的信息视为被动存储的事实，导致语义鸿沟和不可靠的推理。为了解决这一局限性，我们提出了RuleMem，这是一个基于规则的记忆框架，它从历史交互中归纳出可重用的逻辑规则，以主动地指导证据检索和推理。具体而言，RuleMem从对话中构建自然语言的霍恩子句，并通过规则困惑度一致性机制对其进行验证。这些归纳出的规则能够检索语义上相距较远的证据，同时为答案生成提供显式的逻辑结构。我们在两个长期对话基准LoCoMo和LongMemEval_s*上对RuleMem进行了全面评估，并在与14个基线的严格比较中验证了其有效性。

    arXiv:2609.03915v1 Announce Type: new  Abstract: Question answering agents in long-term conversations must reason over massive, temporally dispersed dialogue histories. However, existing memory mechanisms primarily treat past information as \textit{passively} stored facts, leading to semantic gaps and unreliable reasoning. To address this limitation, we propose RuleMem, a rule-based memory framework that induces reusable logical rules from historical interactions to \textit{actively} guide both evidence retrieval and reasoning. Specifically, RuleMem constructs natural-language Horn clauses from conversations and validates them via a Rule Perplexity Consistency (RPC) mechanism. These induced rules enable the retrieval of semantically distant evidence while providing an explicit logical structure for answer generation. We conducted a comprehensive evaluation of RuleMem on two long-term conversational benchmarks, LoCoMo and LongMemEval_s*. In a rigorous comparison against 14 baselines on 
    
[^28]: CROCODIL：基于大语言模型的跨模型代码编辑

    CROCODIL: Cross-Model Code Editing with LLMs

    [https://arxiv.org/abs/2609.03894](https://arxiv.org/abs/2609.03894)

    论文发现大语言模型在编辑其他模型生成的陌生代码时会产生过多且过度的改动，为此提出了CROCODIL后训练框架，通过相似性奖励惩罚大幅改动并结合执行验证，在保证功能正确性的同时有效减少跨模型代码编辑中的过度修改。

    

    大型语言模型（LLMs）已成为代码生成和编辑中无处不在的工具。然而，开发团队通常会使用多个LLM助手。不同的开发者可能偏好不同的模型，而且单个开发者也可能在不同的编码会话中切换使用不同模型。因此，某个模型所做的编辑经常被应用到最初由另一个模型生成的陌生代码上。这些LLM通常在不同的数据集上训练，因此具有不同的风格偏好。那么，当LLM编辑最初由另一个具有不同编码风格的LLM编写的陌生代码时，它们的表现会有所不同吗？我们发现，模型倾向于对陌生代码进行更多、且往往是过度的编辑。我们提出了CROCODIL（基于大语言模型的跨模型代码编辑），这是一个后训练框架，用于在保持功能正确性的同时减少过度编辑。CROCODIL的相似性奖励会对大幅改动进行惩罚，而其执行……（注：原文摘要在此处截断）

    arXiv:2609.03894v1 Announce Type: new  Abstract: Large language models (LLMs) have become ubiquitous tools for code generation and editing. However, development teams often use multiple LLM assistants. Different developers may prefer different models, and individual developers may switch between models across different coding sessions. Because of this, the edits any one model makes are frequently applied to foreign code originally generated by another model. These LLMs are often trained on different datasets, and as a result have different stylistic preferences. Do LLMs behave differently when they edit foreign code originally written by a different LLM with a different coding style? We find that models tend to make more, and often excessive, edits on foreign code. We introduce CROCODIL (Cross-model Code Editing with LLMs), a post-training framework for reducing excessive edits while preserving functional correctness. CROCODIL's similarity reward penalizes large changes, while its exec
    
[^29]: 超越浅层对齐：训练后方法如何决定拒绝电路与转向鲁棒性

    Beyond Shallow Alignment: How Post-Training Methods Determine Refusal Circuits And Steering Robustness

    [https://arxiv.org/abs/2609.03887](https://arxiv.org/abs/2609.03887)

    该研究发现训练后方法（尤其是推理增强微调）会根本性地改变语言模型内部拒绝有害请求的计算方式，但没有任何一种现有方法能同时实现鲁棒、不损失通用能力且可稳定转向的安全对齐。

    

    用于训练语言模型拒绝有害请求的方法，如何塑造拒绝机制在模型内部的实际运作方式？我们在三个架构不同的模型（Llama-3.1-8B、Gemma-2-9B、Qwen3-8B）上比较了三种训练后方法——监督微调、推理增强微调（在为安全决策提供理由的推理链上进行训练）以及偏好优化（ORPO）。我们发现，训练方法而不仅仅是数据，会重塑拒绝在模型内部的计算方式：推理增强训练在所有三个模型中都始终产生一种独特的拒绝计算形式，而模型架构则独立地塑造内部结构以及拒绝被转向干预的可靠程度。最重要的是，我们研究的方法中没有一种能同时满足安全对齐应具备的三个属性：拒绝机制不集中在少数脆弱的组件上、安全收益不以牺牲通用能力为代价，以及转向鲁棒性。

    arXiv:2609.03887v1 Announce Type: new  Abstract: How do the methods used to train language models to refuse harmful requests shape how that refusal actually works inside the model? We compare three post-training methods - supervised fine-tuning, reasoning-augmented fine-tuning (training on reasoning chains that justify a safety decision), and preference optimization (ORPO) - across three architecturally distinct models (Llama-3.1-8B, Gemma-2-9B, Qwen3-8B). We find that training method, not just data, reshapes how refusal is computed internally: reasoning-augmented training consistently produces a distinct kind of refusal computation, visible across all three models, while architecture independently shapes internal structure and how reliably refusal can be steered. Most importantly, no method we study achieves all three properties we would want from safe alignment at once: refusal that isn't concentrated in a few fragile components, safety gains that don't cost general capability, and s
    
[^30]: 翻转而非打乱：以推理速度为LLM添加水印

    Flip, Don't Shuffle: Watermarking LLMs at the Speed of Inference

    [https://arxiv.org/abs/2609.03844](https://arxiv.org/abs/2609.03844)

    提出无状态伯努利水印（SBW），通过每词元独立伯努利试验实现O(1)复杂度的绿名单判断，检测速度比KGW自盐值快6000倍以上、比SynthID快2倍，同时保持相同的N(0,1)统计检测保证。

    

    我们提出了无状态伯努利水印（SBW），这是一种针对大语言模型的新型统计水印方法，它通过每个词元独立的伯努利试验来确定绿名单成员资格。与KGW的词表置换或SynthID的多层锦标赛机制不同，SBW只需对每个词元与基于计数器的随机数生成器进行一次比较，将成员判断复杂度降至O(1)，并实现了零中间内存分配的单内核执行。我们证明了这种形式化方法保持了与固定大小绿名单相同的检测保证：在零假设下，z分数检验仍服从N(0,1)分布。这种无状态架构实现了现有方法无法企及的能力：全词表自盐值水印（比KGW的自盐值方法快6000倍以上，尽管使用候选相关种子对整个词表进行偏置，仍比SynthID快2倍），以及与蒸馏架构的兼容性。

    arXiv:2609.03844v1 Announce Type: cross  Abstract: We introduce Stateless Bernoulli Watermarking (SBW), a new statistical watermark for Large Language Models that determines green list membership through independent per-token Bernoulli trials. Unlike KGW's vocabulary permutation or SynthID's multi-layer tournament, SBW requires only a single comparison per token against a counter-based random number generator, reducing membership complexity to $O(1)$ and enabling single-kernel execution with zero intermediate allocations. We prove that this formulation preserves the same detection guarantees as fixed-size green lists: the z-score test remains $\mathcal{N}(0,1)$ under the null. The stateless architecture enables capabilities unavailable to existing methods: full-vocabulary self-salt watermarking (over 6000$\times$ faster than KGW's self-salt and 2$\times$ faster than SynthID despite biasing the entire vocabulary with candidate-dependent seeding) and architectural compatibility with dist
    
[^31]: 选择、压缩、再投资：长视频多模态大语言模型中视觉令牌分配的受控研究

    Select, Compress, Reinvest: A Controlled Study of Visual-Token Allocation in Long-Video MLLMs

    [https://arxiv.org/abs/2609.03820](https://arxiv.org/abs/2609.03820)

    本文通过严格控制变量的受控实验发现，在长视频多模态大语言模型的视觉令牌分配中，帧选择是影响性能的最大单一因素——八个基于查询选择的帧可超越十六个均匀采样的帧，且经典的正交匹配追踪算法即可媲美各类专门设计的选择器。

    

    长视频语言模型无法查看每一帧：一小时的视频按每秒采样一次就是3,600张图像，而系统只能保留该图像池中固定的一小部分切片。哪些帧能够保留在这一切片中，通常被视为预处理细节；我们检验这是否应当如此。已发表的选择器使比较变得困难，因为它们同时改变了帧评分器、提示边界、分辨率策略和回答模型。我们将上述因素全部固定，每次只改变一个决策：选择、空间压缩以及节省资源的再投资，涵盖六种免训练选择规则、三个长视频基准测试和两个回答模型。选择是最大的单一杠杆：在LongVideoBench的小时级数据段上，八个基于查询选择的帧比十六个均匀间隔的帧高出6.9分，而正交匹配追踪——一种未经修改的几十年前的稀疏近似算法——与每一个专门设计的选择器相当或差距不到一分……

    arXiv:2609.03820v1 Announce Type: cross  Abstract: Long-video language models cannot look at every frame: an hour sampled once per second is 3,600 images, and a system keeps only a small fixed slice of that pool. Which frames survive that slice is usually treated as a preprocessing detail; we test whether it should be. Published selectors make the comparison hard because they change the frame scorer, the prompt boundary, the resolution policy, and the answering model all at once. We hold each fixed and vary one decision at a time: selection, spatial compression, and reinvestment of the savings, across six training-free selection rules, three long-video benchmarks, and two answering models. Selection is the largest single lever: on LongVideoBench's hour-long bin, eight query-selected frames beat sixteen uniformly spaced ones by 6.9 points, and Orthogonal Matching Pursuit, an unmodified decades-old sparse-approximation algorithm, matches or comes within a point of every purpose-built sel
    
[^32]: 评估大型语言模型在内容审核中的标准条件化行为

    Evaluating Criterion-Conditioned Behaviour of Large Language Models in Content Moderation

    [https://arxiv.org/abs/2609.03814](https://arxiv.org/abs/2609.03814)

    提出DECO诊断评估框架，通过标准无关的内容分解与成对评估方法，揭示LLM在内容审核基准上的优异表现可能掩盖其在具体审核标准层面的大量失败。

    

    大型语言模型（LLM）在标准内容审核基准测试中表现出色。然而，这些基准测试通常将多个审核标准聚合为单一标签，因此无法确定模型是否能够区分这些标准，并在决策时可靠地应用每一个标准。为了研究LLM是否表现出标准条件化行为，我们引入了内容诊断评估（DECO），这是一种与标准无关的内容分解方法，能够实现受控的、标准层面的评估。我们还引入了成对评估方法，用于比较模型对同一输入在不同标准下的输出。在四个审核数据集和四个LLM上的实验表明，出色的基准测试成绩可能掩盖标准层面的大量失败。当正确决策并非取决于内容的整体危害性，而是取决于标准所要求模型评估的特定方面时，模型的表现最为困难。

    arXiv:2609.03814v1 Announce Type: new  Abstract: Large language models (LLMs) demonstrate strong performance on standard content moderation benchmarks. However, these benchmarks often aggregate multiple moderation criteria into a single label, making it unclear whether models can disentangle them and reliably apply each criterion when making decisions. To study whether LLMs exhibit criterion-conditioned behaviour, we introduce Diagnostic Evaluation of COntent (DECO), a criterion-independent factorisation of content that enables controlled, criterion-level evaluation. We also introduce pairwise evaluation to compare model outputs across different criteria for the same input. Across four moderation datasets and four LLMs, we find that strong benchmark performance can hide substantial failures at the criterion level. Models struggle most when correct decisions depend not on overall harmfulness, but on the specific aspect of the content that the criterion requires them to assess. Our resul
    
[^33]: VisCAD：具备多模态工业CAD智能的基础模型套件

    VisCAD: A Foundation Model Suite with Multimodal Industrial CAD Intelligence

    [https://arxiv.org/abs/2609.03811](https://arxiv.org/abs/2609.03811)

    VisCAD是一个面向工业CAD的基础模型套件，其核心270亿参数模型VisCAD-M1能够将渲染图、文本、2D图纸和真实照片等多种输入转换为可执行的CAD程序，在保证广泛泛化能力的同时具备强大的专业CAD设计与装配生成能力。

    

    面向工业产品的AI辅助计算机辅助设计（CAD）涉及两个具有挑战性的阶段。零件级生成将多种形式的用户意图——包括渲染图、文本描述、二维图纸和真实照片——映射为CAD领域特定语言的可执行程序。装配级生成则还需要处理相互作用的零件、规划配合关系、估计位姿并正确放置所有零件。现有的专用CAD模型通常只在狭窄的输入域（如渲染图或文本）上训练，泛化能力往往较差；而通用前沿模型虽然覆盖更广泛的输入，但在CAD各领域上的表现并不稳定。我们提出了VisCAD，一个基础模型套件，旨在为真实工业产品同时提供广泛的泛化能力和强大的CAD专业能力。其核心是VisCAD-M1，一个经过中期训练和后训练的270亿参数模型，用于零件级设计生成。在PubCADBench上（摘要截断）……

    arXiv:2609.03811v1 Announce Type: cross  Abstract: AI-assisted computer-aided design (CAD) for industrial products involves two challenging phases. Part-level generation maps diverse forms of user intent, including renders, text descriptions, 2D drawings, and real photographs, to executable programs in a CAD domain-specific language. Assembly-level generation must additionally handle interacting parts, plan mating relations, estimate poses, and place all parts correctly. Existing specialized CAD models are commonly trained on narrow input domains, such as renders or texts, and often generalize poorly, while general-purpose frontier models cover broader inputs but perform inconsistently across CAD domains. We present VisCAD, a foundation model suite designed to provide both broad generalization and strong CAD capability for realistic industrial products. At its core is VisCAD-M1, a 27B model trained through mid-training and post-training for part-level design generation. On PubCADBench 
    
[^34]: Transfiver：通过共享可编辑状态实现人机协同推理

    Transfiver: Human-AI Co-Inference through a Shared Editable State

    [https://arxiv.org/abs/2609.03797](https://arxiv.org/abs/2609.03797)

    Transfiver 提出了一种人机协同推理架构，将交互信息维护在模型与人类共同更新的单一共享持久状态中，通过隐式流式更新与显式定向编辑两种机制，使人类的修正能够直接改变后续计算所读取的状态。

    

    长期的人机交互之所以困难，是因为引导推理的信息由模型隐式更新，用户无法直接查看或控制。我们提出了“透明交互式、可验证、可编辑表示框架”（Transfiver），这是一种通过共享可编辑状态实现人机协同推理的架构。其核心思想是：与交互相关的信息被维护在单一持久状态 $(S_t)$ 中，该状态由模型和人类共同更新。Transfiver 区分了两种状态演化模式：在隐式流式更新中，模型解释正在进行的交互，并决定新信息是修改现有的状态项还是创建新的状态项；在显式定向编辑中，人类可以检查并修改指定的状态项。两者都作用于同一个底层状态，因此人类的修正会改变后续计算所读取的状态，而不是简单地另加一条信息。

    arXiv:2609.03797v1 Announce Type: new  Abstract: Long-term human-AI interaction is difficult because the information that guides inference is updated implicitly by the model and is not directly inspectable or controllable by the user. We introduce the TRANSparent Framework for Interactive, Verifiable, Editable Representation (Transfiver), an architecture for human-AI co-inference through a shared editable state. Its central idea is that interaction-specific information is maintained in a single persistent state $(S_t)$ that both the model and the human update.   Transfiver distinguishes two modes of state evolution. In an implicit stream update, the model interprets ongoing interaction and decides whether new information revises an existing state item or creates a new one. In an explicit directed edit, a human inspects and modifies an addressed item. Both act on the same underlying state, so a human correction changes the state that subsequent computation reads, rather than adding anot
    
[^35]: 一种反向手语词典：通过视频描述生成与描述检索实现连续手语中的开放词汇手语识别

    A Reverse Sign Language Dictionary: Open-Vocabulary Sign Recognition from Continuous Signing via Video Captioning and Description Retrieval

    [https://arxiv.org/abs/2609.03788](https://arxiv.org/abs/2609.03788)

    该论文提出一种“反向手语词典”方法：先用视觉-语言模型将连续手语片段生成自由形式的动作过程描述，再用多语言句子编码器从目标描述库中检索最匹配条目，从而实现无需词汇标注监督、支持开放词汇的手语识别。

    

    孤立手语识别（ISLR）传统上被构建为基于词汇标签的闭集分类问题，这种方式无法泛化到训练中未见过的手语，并且使每次部署都依赖于带有词汇标注的词典。我们转而通过以下方式识别从连续手语中提取出的手语：（1）使用开放权重的视觉-语言模型将手语级别的视频片段转化为对发音动作的自由形式过程描述；（2）使用多语言句子编码器从目标描述词汇库中检索最接近的条目——这构成了一种反向手语词典，无需词汇监督，且支持开放词汇。在一个标注了过程描述的日本手语（JSL）对话语料库的1,300个手语级别片段上进行评估（相对于503个条目目标词汇库的2% top-10随机基线），对手语描述生成模型进行微调显著提升了已见类别的检索性能：语言塔和视觉塔的微调使top-1……（原文摘要在此处截断）

    arXiv:2609.03788v1 Announce Type: cross  Abstract: Isolated Sign Language Recognition (ISLR) is conventionally cast as closed-set classification over gloss labels, which cannot generalize to signs unseen in training and ties every deployment to a gloss-annotated lexicon. We instead recognize signs extracted from continuous signing by (1) captioning a sign-level clip into a free-form procedural description of the articulation with an open-weight vision-language model, and (2) retrieving the closest entry from a vocabulary of target descriptions with a multilingual sentence encoder: a reverse sign language dictionary that needs no gloss supervision and admits an open vocabulary. On 1,300 sign-level segments from a Japanese Sign Language (JSL) dialogue corpus annotated with procedural descriptions (against a 2% top-10 chance floor over the 503-entry target vocabulary), fine-tuning the captioner substantially improves seen-class retrieval: language and vision tower fine-tuning raises top-1
    
[^36]: IndicSafeEval：多语言说服性越狱攻击下大语言模型的安全稳健性

    IndicSafeEval: Safety Robustness of Large Language Models under Multilingual Persuasive Jailbreak Attacks

    [https://arxiv.org/abs/2609.03781](https://arxiv.org/abs/2609.03781)

    该论文提出了IndicSafeEval框架，通过四种印度语言、十个安全类别和六种说服策略构建7,200条对抗性提示，系统评估并揭示了大语言模型在面对多语言说服性越狱攻击时安全表现存在显著差异。

    

    大语言模型在多语言环境中的应用日益广泛，但其安全性评估主要仍以英语为主。这限制了我们对对齐失效在低资源和文化多样性语言中如何表现的理解。我们提出了IndicSafeEval，一个针对印度语言的说服性越狱攻击评估框架。该基准将十个安全关键内容类别与六种类人说服策略相结合，涵盖印地语、孟加拉语、马拉地语和旁遮普语四种不同的印度语言，共生成7,200条对抗性提示。我们对多个开源大语言模型进行了系统性的黑盒评估，以检验其安全行为如何随语言、说服策略和风险类别的变化而变化。我们的分析表明，模型并非在所有语言和提示风格下都表现得同样安全，相反，安全性能在很大程度上取决于所使用的语言以及提示的构造方式。

    arXiv:2609.03781v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in multilingual settings, yet their safety is still evaluated primarily in English. This limits our understanding of how alignment failures manifest in low-resource and culturally diverse languages. We introduce IndicSafeEval, a persuasion-based jailbreak evaluation framework for Indian languages. Our benchmark combines ten safety critical content categories with six human-like persuasive strategies across four different Indian languages, such as Hindi, Bengali, Marathi and Punjabi, resulting in 7,200 adversarial prompts. We conduct a systematic black-box evaluation of several open-source LLMs to examine how their safety behaviour varies across languages, persuasion strategies, and risk categories. Our analysis shows that the model does not behave equally safely across all languages and prompt styles. Instead, safety performance depends strongly on both the languages used and the way a
    
[^37]: 基于大语言模型的类型学特征预测：一种上下文学习方法

    Typological Feature Prediction with Large Language Models: An In-Context Learning Approach

    [https://arxiv.org/abs/2609.03775](https://arxiv.org/abs/2609.03775)

    大语言模型通过结合系统发育和地理邻近语言证据的上下文学习方法，显著优于基线方法进行类型学特征预测，且对低资源语言同样有效，其预测依据与证据保持一致，为实现可解释的类型学特征预测提供了新途径。

    

    类型学特征在多语言自然语言处理（NLP）中被广泛使用，对此类特征的预测具有重要的下游应用价值。然而，现有的缺失值预测方法缺乏可解释的预测依据，且其在不同资源水平和特征类型上的性能仍未得到充分探索。鉴于大语言模型（LLM）在元语言推理和提供推理依据方面的能力，我们通过上下文学习方法，利用来自URIEL+和Glottolog的语言学数据，研究了大语言模型在类型学特征预测中的表现。我们发现零样本提示是不够的，但当提供系统发育和地理邻近证据时，大语言模型的表现显著优于所有基线方法，且不会使低资源语言处于不利地位。我们进一步发现，大多数大语言模型给出的推理依据与所提供的证据一致，这为可解释的类型学特征预测迈出了一步。

    arXiv:2609.03775v1 Announce Type: new  Abstract: Typological features are widely used in multilingual NLP, and the prediction of such features holds downstream utility. However, existing methods to predict missing values lack interpretable justifications for predictions, while their performance across resource levels and feature types remains underexplored. Given LLMs' abilities in meta-linguistic reasoning and in providing rationales, we investigate LLMs' performance in typological feature prediction via an in-context learning approach with linguistic data from URIEL+ and Glottolog. We find that zero-shot prompting is insufficient, but when given phylogenetic and geographic neighbour evidence, LLMs substantially outperform all baselines without disadvantaging low-resource languages. We further find that most LLM rationales are consistent with the provided evidence, offering a step toward explainable typological feature prediction.
    
[^38]: RealCADBench：基于工业设计意图的参数化CAD建模基准测试

    RealCADBench: Benchmarking Parametric CAD Modeling from Industrial Design Intents

    [https://arxiv.org/abs/2609.03773](https://arxiv.org/abs/2609.03773)

    提出了RealCADBench基准测试，基于19个工厂自动化类别的真实工业设计意图，通过文本、图纸、图片等多种输入模态和可执行性、IoU、视觉语义一致性等综合评估指标，系统性地评估从设计意图到程序化参数CAD建模的能力。

    

    参数化计算机辅助设计（CAD）建模难以用单一指标进行评估。现有的CAD基准测试往往侧重于合成或CAD原生环境、有限的输入模态，或仅关注可执行性和IoU（交并比）。我们提出了RealCADBench，这是一个从真实工业设计意图到程序化CAD建模的基准测试。它包含来自19个工厂自动化类别的12,632个任务，涵盖文本描述、2D工程图纸、真实产品图片和渲染图像等多种输入模态，同时支持零件和装配体建模。我们在一个包含1,770个任务的评估切片上报告结果：包括跨四种输入模式的1,745个零件任务，以及RCB-Assm25（一个包含25个任务的装配体研究），用于所有已报告的装配体比较。每种方法生成FreeCAD API的Python代码，由共享运行时执行以导出3D模型。我们使用可执行性、Solid IoU、Surface IoU以及基于评分标准的视觉-语义一致性Judge来评估导出的模型。

    arXiv:2609.03773v1 Announce Type: cross  Abstract: Parametric computer-aided design (CAD) modeling is difficult to evaluate with a single metric. Existing CAD benchmarks often emphasize synthetic or CAD-native settings, limited input modalities, or executability and IoUs alone. We introduce RealCADBench, a benchmark for intent-to-program CAD modeling from real industrial design intents. It contains 12,632 tasks from 19 factory-automation categories and spans text descriptions, 2D engineering drawings, real product pictures, and rendered images for both Part and Assembly modeling. We report results on a 1,770-task evaluation slice: 1,745 Part tasks across four input regimes and RCB-Assm25, a 25-task assembly study used in every reported assembly comparison. Each method generates FreeCAD API Python, which a shared runtime executes to export the 3D model. We evaluate the exported model using executability, Solid IoU, Surface IoU, and a rubric-based visual-semantic identity Judge. Among th
    
[^39]: OBER+：成果导向教育中具备连续性感知的报告与可追溯的持续改进

    OBER+: Continuity-Aware Reporting and Traceable Continuous Improvement in Outcome-Based Education

    [https://arxiv.org/abs/2609.03770](https://arxiv.org/abs/2609.03770)

    OBER+通过五个相互衔接的阶段将测得的成果差距转化为可追溯且经评估的纠正措施，并引入连续性规则以避免在成果表述变更时误读达成度趋势，从而在成果导向教育中实现从测量到改进的闭环。

    

    实施成果导向教育的机构通常会例行计算学习成果的达成度，然而课程分析领域的综述指出，这种计算如何为决策提供依据方面缺乏证据。本文提出了OBER+，这是对一个已部署的机构级达成度平台的扩展，它计算了从测得的差距到经过评估的纠正措施之间的步骤。五个相互衔接的阶段分别完成：在课程的多次授课中累积达成度；发出差距及持续性差距信号；按照监管机构已经在使用的截止标准对其进行分级；将决策记录在附有证据注释的实践目录中；记录变更；并量化差距的后续变化。另一条规则会对同一成果的先后表述进行比较，因此当成果表述发生变化时，达成度不会被误读为跨越该变化点的连续序列。将这些规则应用于两门真实课程的实时记录产生了三个结果……

    arXiv:2609.03770v1 Announce Type: cross  Abstract: Institutions practising outcome-based education compute learning outcome attainment routinely, while reviews of curriculum analytics report an absence of evidence on how that computation informs decisions. This paper presents OBER+, an extension of a deployed institutional attainment platform that computes the step from a measured shortfall to an evaluated corrective action. Five connected stages accumulate attainment across deliveries of a course, signal a shortfall and a persistent shortfall, grade it on cutoffs the regulator already uses, record the decision against a catalogue of practices annotated with their evidence, log the change, and quantify the subsequent movement in the shortfall. A further rule compares successive statements of an outcome, so attainment is never read as a series across a point at which the outcome changed. Applying the rules to the live record of two real courses produced three results. Every outcome of a
    
[^40]: Rent-a-RAG：用于审计第三方RAG的嵌入空间水印

    Rent-a-RAG: Embedding-Space Watermarks for Auditing Third-Party RAG

    [https://arxiv.org/abs/2609.03749](https://arxiv.org/abs/2609.03749)

    提出DirBucket框架，通过让文档改写版本的嵌入偏向提供方专属的秘密方向来嵌入语义水印，从而在保持检索效用的同时，实现对第三方RAG中无偿复用文档的黑盒审计检测。

    

    第三方检索增强生成（RAG）市场带来了一个新的审计问题：数据提供方可能将其语料库授权给RAG运营商，但事后却无从得知自己的文档是否正在被无偿重复使用。审计这种滥用行为十分困难，因为运营商不配合，答案会被生成器改写，且单次回答可能融合来自多个提供方的证据。我们提出了DirBucket，一个面向多提供方RAG中文档级重复使用的提供方侧语义水印与黑盒审计框架。DirBucket通过保持语义不变的改写方式为文档添加水印，使这些改写文本的嵌入偏向于提供方桶的秘密方向，从而在保留检索效用的同时，能够从黑盒回答中实现检测。在一个反映黑盒访问下混合提供方重复使用场景的挑战性基准上，DirBucket是唯一能够持续实现强目标检测且无（原文截断）

    arXiv:2609.03749v1 Announce Type: cross  Abstract: Third-party retrieval-augmented generation (RAG) marketplaces create a new auditing problem: data providers may license corpora to a RAG operator, yet later have no visibility into whether their documents are being reused without compensation. Auditing this misuse is difficult because the operator is non-cooperative, answers are paraphrased by the generator, and one response may combine evidence from many providers. We propose DirBucket, a provider-side semantic watermarking and black-box auditing framework for document-level reuse in multi-provider RAG. DirBucket watermarks documents by meaning-preserving paraphrases whose embeddings are biased toward provider-bucket secret directions, enabling detection from black-box answers while preserving retrieval utility. On a challenging benchmark that reflects mixed-provider reuse under black-box access, DirBucket is the only method that consistently achieves strong target detection with no n
    
[^41]: KnowVis：面向视频讲座的以知识为中心的视觉摘要生成

    KnowVis: Knowledge-Centric Visual Summarization for Video Lectures

    [https://arxiv.org/abs/2609.03742](https://arxiv.org/abs/2609.03742)

    提出 KnowVis 框架，通过从多模态视频内容中提取概念图、构建结构化知识单元并合成视觉摘要，将线性视频讲座转化为符合教学规律的视觉叙事，从而降低初学者的认知负担。

    

    视频讲座是宝贵的教育资源，但其内容密集且冗长的形式常常使初学者感到无所适从。这一困难源于一个根本性的教学不匹配：视频以线性方式传递瞬时信息，而人类学习则需要构建相互关联的认知网络，对于缺乏先前领域知识的初学者来说，这项任务会导致严重的认知过载。现有的视频摘要方法未能解决这种不匹配，因为它们主要生成以文本为主的线性浓缩内容，仍然需要较高的认知努力。为了弥合这一差距，我们提出了 KnowVis，这是一个将线性视频讲座转化为具有教学依据的视觉叙事的框架。KnowVis 首先从多模态视频内容中提取详细的概念图，以识别重要且具有挑战性的阈值概念，然后构建结构化的知识单元，最后合成引人入胜的视觉摘要。

    arXiv:2609.03742v1 Announce Type: cross  Abstract: Video lectures are valuable educational resources, but their dense and lengthy formats often overwhelm novice learners. This difficulty stems from a fundamental pedagogical mismatch: while videos deliver transient information linearly, human learning requires constructing interconnected cognitive networks, a task that induces severe cognitive overload for novice learners lacking prior domain knowledge. Existing video summarization methods fail to resolve this mismatch, as they primarily produce text-heavy, linear condensations that still demand high cognitive effort. To bridge this gap, we propose KnowVis, a framework that transforms linear video lectures into pedagogically grounded visual narratives. KnowVis first extracts a detailed concept map from multimodal video content to identify important and challenging threshold concepts, then constructs structured knowledge units, and finally synthesizes engaging visual summaries. Alongside
    
[^42]: 超越BLEU：重新定义手语翻译基准的案例

    Beyond BLEU: A Case for Redefining Sign Language Translation Benchmarks

    [https://arxiv.org/abs/2609.03734](https://arxiv.org/abs/2609.03734)

    本文证明BLEU-4的提升并不等同于更强的手语理解能力，并提出了一种基于开放权重LLM问答协议的新型评估方法，该方法更符合人类排名、对改写更不敏感且对训练-测试重叠更加鲁棒。

    

    BLEU-4是评估手语翻译（SLT）的标准指标，但口语语言指标可能无法充分反映手语能力。SLT的多模态、低资源环境使得模型能够利用虚假相关性和口语先验，而不是学习更强的手语表示。在本文中，我们评估了六个SLT模型在Phoenix-2014T和CSL-Daily数据集上的时空理解与BLEU-4之间的关系，表明BLEU-4的提升本身并不能证明更好的手语理解能力。这项工作引入了一种受语言学习评估启发的替代方法，使用开放权重LLM问答协议来衡量显著内容的保留程度。该协议与人类排名更加一致，并且在改写不变性方面比BLEU-4高出六到七倍。应用于SLT时，该协议针对内容传递，对训练-测试重叠更加鲁棒，并且给出……

    arXiv:2609.03734v1 Announce Type: cross  Abstract: BLEU-4 is the standard metric for evaluating sign language translation (SLT), but spoken-language metrics may not adequately reflect sign language proficiency. The multimodal, low-resource context of SLT allows models to exploit spurious correlations and spoken-language priors, rather than learning stronger sign representations. In this paper, we evaluate the relationship between spatio-temporal understanding and BLEU-4 across six SLT models on Phoenix-2014T and CSL-Daily, showing that gains in BLEU-4 are not on their own evidence of better sign language understanding. This work introduces an alternative inspired by language-learning assessment, using an open-weight-LLM QA protocol that measures salient content preservation. It aligns more closely with human rankings and is six to seven times more paraphrase-invariant than BLEU-4. Applied to SLT, this protocol targets content transfer, is more robust to train-test overlap, and gives a 
    
[^43]: 通过开放架构开启心智：分析策略

    Opening mind by opening architecture: analysis strategies

    [https://arxiv.org/abs/2609.03719](https://arxiv.org/abs/2609.03719)

    论文指出电声作曲领域封闭架构音频处理器的日益主导使内部处理过程沦为不可洞察的“黑箱”，并提出通过文献分析策略重新审视信号处理技术的实现历程及其美学意义。

    

    在电声作曲的数值信号处理领域，随着数字市场工具的日益普及，专用的开发与研究环境逐渐流失，这促成了封闭架构音频处理器模型的主导地位。这种模型虽然功能强大，能够描述关于其感知特性的输出数据，但代价是忽略了其内部过程和交互系统——这些系统成为了复杂而强大的环境，却封闭在难以洞察的黑箱之中，这是一种我们必须正视的损失。任何数字信号处理技术都在讲述一个故事。正如一种语言的词汇承载着社会、历史和技术等多重语义层面，信号处理器也有其自身的实现故事，这是一段渐进的技术成就历程，并伴随着不可避免的美学影响。透过文献这面镜子，人们能够以焕然一新的认知去接近那些环境……（原文摘要在此处不完整）

    arXiv:2609.03719v1 Announce Type: new  Abstract: In numerical signal processing for electroacoustic composition, the progressive loss of specific development and research environments caused by the increasing use of digital market tools has favoured the dominance of the closed-architecture audio processor model. This model, while powerful, envisions the possibility of describing output data about its perceived characteristics, but at the cost of ignoring its internal process and interacting systems, which become complex, powerful environments but closed in an inscrutable black box, a loss we must consider. Any digital signal processing technique tells a story. Just as the words of a language incorporate social, historical and technical polysemic layers, a signal processor has its own story of implementation, a gradual technological achievement with its inevitable aesthetic consequences. Through the looking-glass of literature, one can access those environments with renewed awareness by
    
[^44]: 超越通用框架，CAE仿真智能体真正需要什么？

    What Do CAE Simulation Agents Really Need Beyond a Generic Harness?

    [https://arxiv.org/abs/2609.03718](https://arxiv.org/abs/2609.03718)

    该研究发现在信息访问和修复预算相同的条件下，单智能体通用框架在CAE仿真任务上能匹敌甚至超越专门设计的多智能体系统，其关键在于框架已内置的执行反馈修复机制（使FoamBench成绩从71.8%提升至96.4%），表明现代LLM框架已有的能力足以替代CAE专用的复杂机制。

    

    计算机辅助工程（CAE）仿真是工程领域中规模最大、要求最高的领域之一，配置诸如OpenFOAM、FEniCS或COMSOL等求解器需要真正的专业知识。大语言模型（LLM）智能体有望将自然语言请求转化为可运行的仿真，而近期的CAE智能体添加了仿真专用的机制：多智能体分解、领域检索和脚本化反思。这些机制适用于较弱的基础模型；现代框架已经提供了多轮推理、工具使用和执行反馈。我们提出了这样一个问题：除了通用框架之外，CAE仿真智能体还需要什么？在信息访问和修复预算保持固定的情况下，单智能体框架的性能与多智能体专用系统相当甚至更优（FoamBench上96.4%对比88.2%）。消融实验将此归因于框架本身已提供的能力：执行反馈修复使FoamBench的成绩从无修复轮次时的71.8%提升至96.4%，而脚本化……

    arXiv:2609.03718v1 Announce Type: cross  Abstract: Computer-aided engineering (CAE) simulation is among the largest and most demanding areas of engineering, where setting up a solver such as OpenFOAM, FEniCS, or COMSOL takes real expertise. Large language model (LLM) agents promise to turn a natural-language request into a working simulation, and recent CAE agents add simulation-specific machinery: multi-agent decomposition, domain retrieval, and scripted reflection. That machinery suited weak base models; modern harnesses already supply multi-turn reasoning, tool use, and execution feedback. We ask what a CAE simulation agent still needs beyond a generic harness. With information access and repair budget held fixed, a single-agent harness matches or beats multi-agent specialized systems (FoamBench 96.4\% vs.\ 88.2\%).   Ablations trace this to capabilities the harness already provides: execution-feedback repair lifts FoamBench from 71.8\% with no repair round to 96.4\%, while scripted
    
[^45]: 复数指代的电路机制：大语言模型如何表示与检索单数和复数实体

    A Circuit for Plural Reference: How LLMs Represent and Retrieve Singular and Plural Entities

    [https://arxiv.org/abs/2609.03687](https://arxiv.org/abs/2609.03687)

    该论文结合机制可解释性与因果干预技术，首次揭示了大语言模型处理复数指代的完整电路机制，发现了一组分别负责表示共指信息、识别复数实体和传递信息的注意力头，并证明LLM对本体论相似实体的复数代词偏好与人类一致。

    

    共指消解是语境推理中的一项重要任务。本文研究了复数指代中单数和复数实体的表示与检索机制。我们结合机制可解释性与注意力模式分析，研究大语言模型预测代词以回指先前提及实体的过程。通过一系列因果干预技术，我们发现了一组注意力头，它们分别负责：（1）表示输入中的共指信息，（2）识别构成复数指代的实体，（3）将信息传递给负责选择先行词和预测代词的组件。我们还发现，大语言模型在复数代词的偏好上与人类一致。具体而言，复数结构中的实体如果在本体论上相似且相互关联，则更有可能被视为一个复数实体而被复数代词指代。

    arXiv:2609.03687v1 Announce Type: new  Abstract: Coreference resolution is an important task in contextual reasoning. In this paper, we investigate the mechanism for representing and retrieving singular and plural entities for plural reference. We use a combination of mechanistic interpretability and attention pattern analysis to study the process in which LLMs predict a pronoun to refer back to previously mentioned entities. Using a range of causal intervention techniques, we find a set of attention heads that are responsible for (1) representing coreference information in the input, (2) identifying entities that form a plural reference, (3) transferring the information to the component that is responsible for selecting the antecedents and predicting the pronoun. We also find that LLMs align with humans in preference for plural pronoun. Specifically, entities in a plural construction are more likely to be referred to as a plural entity if they are ontologically similar and are linked 
    
[^46]: 通过用自然语言描述图像子集之间的差异来理解自动驾驶数据集

    Understanding Autonomous Driving Datasets by Describing Differences between Image Subsets in Natural Language

    [https://arxiv.org/abs/2609.03677](https://arxiv.org/abs/2609.03677)

    本文提出集合差异描述方法，利用自然语言自动描述自动驾驶数据集中不同图像子集之间的差异，通过基于目标检测的对象中心分析实现对数据集组成和域偏移的可解释理解。

    

    理解大规模自动驾驶数据集的组成对于安全性、鲁棒性以及跨域的可靠运行至关重要。例如，不同地点之间的域偏移可能导致运行环境与训练数据不一致，从而造成潜在的危险性能下降。然而，现有的数据分析流程在很大程度上依赖于元数据、预定义标签或人工检查，这些方法提供的语义洞察有限或无法规模化。本文研究了集合差异描述任务：给定两个图像子集，目标是生成一个用自然语言描述目标集与参考集之间差异的假设。基于两阶段的框架，我们将该方法适配到自动驾驶领域，通过聚焦于从目标检测中提取的以对象为中心的图像块，这简化了聚合过程，并能够将差异归因于特定的对象实例或类别。

    arXiv:2609.03677v1 Announce Type: cross  Abstract: Understanding the composition of large-scale autonomous driving datasets is essential for safety, robustness, and reliable operation across domains. For example, domain shift between locations could lead to the operating environment being misaligned with the training data, resulting in potentially dangerous performance degradation. Yet, existing data analysis pipelines largely rely on metadata, predefined labels, or manual inspection, which provide limited semantic insight or do not scale. This paper studies set difference captioning: given two subsets of images, the goal is to produce a natural-language hypothesis describing differences between the target and reference set. Building on a two-stage formulation, we adapt the method to autonomous driving by focusing on object-centric patches derived from object detection, which simplifies aggregation and enables attribution of differences to specific object instances or categories. To ev
    
[^47]: 增强金融问答：一个基于银行财务报表的新型基准数据集

    Enhancing Financial Question Answering: A Novel Benchmark Dataset of Banks' financial statements

    [https://arxiv.org/abs/2609.03654](https://arxiv.org/abs/2609.03654)

    该论文提出了首个针对跨机构银行财务报表检索的金融问答基准数据集 FinRAG-QA，包含999个从业者整理的问题和24家欧美大型银行的209份超长报告，并系统评估了多阶段 RAG 流水线中各组件的贡献。

    

    由于银行财务报表的复杂性、篇幅冗长、专业术语的使用，以及不同司法管辖区和机构之间文本与数值内容的异质性，对其进行比较分析对自动问答系统构成了重大挑战。我们提出了 FinRAG-QA，一个新颖的金融问答基准数据集，包含999个由从业者精心整理的问题，涵盖10个标准化指标，基于24家欧美主要银行2019年至2023年的209份年度报告和第三支柱（Pillar 3）报告。与以往主要聚焦于美国申报文件和单一机构分析的金融问答基准不同，FinRAG-QA 针对的是跨机构检索场景，文档平均长度达19.8万字，超过了任何现有的金融问答资源。在该基准上，我们评估了一个多阶段 RAG 流水线，并分离量化了每个组件的贡献。上下文分块增强结合检索器……

    arXiv:2609.03654v1 Announce Type: cross  Abstract: The comparative analysis of banks' financial statements poses significant challenges for automated question answering systems due to their complexity, substantial length, technical language, and inhomogeneity of both textual and numerical content across different jurisdictions and institutions. We introduce FinRAG-QA, a novel benchmark dataset for financial question answering, which comprises 999 practitioner-curated questions on 10 standardised indicators, grounded in 209 annual and Pillar 3 reports from 24 major European and U.S. banks spanning 2019-2023. Unlike prior financial QA benchmarks, which centre on U.S. filings and single-institution analysis, FinRAG-QA targets cross-institutional retrieval over documents averaging 198k words, longer than any existing financial QA resource. On this benchmark we evaluate a multi-stage RAG pipeline and isolate the contribution of each component. Contextual chunk enrichment combined with a ret
    
[^48]: 合成数据增强对话语-语用功能分类的影响

    The Impact of Synthetic Data Augmentation on Discourse-Pragmatic Function Classification

    [https://arxiv.org/abs/2609.03652](https://arxiv.org/abs/2609.03652)

    本研究在话语-语用功能分类任务中保持合成样本数量恒定，通过改变其相对于决策边界的几何位置，揭示了合成数据增强的效果取决于合成样本与真实训练数据的几何关系而非仅仅是数量。

    

    合成数据增强已成为解决自然语言处理中类别不平衡问题的常用策略，但大多数方法关注的是生成样本的数量和多样性，而非其与真实训练数据之间的几何关系。我们在话语-语用功能分类的背景下研究这一问题，这是一项数据稀疏性属于结构性特征而非采集偶然现象的任务。我们使用从英国国家语料库中选取的410个经人工标注的英语单词"look"实例，涵盖四种功能：注意信号、指令、话语标记和感叹词。我们使用Llama 3.1生成合成训练样本，并根据它们在RoBERTa嵌入空间中与真实训练数据的余弦距离对其进行划分。我们在各条件下保持增强数量恒定，比较了六种在合成样本相对于经验决策边界的位置上有所不同的训练条件。

    arXiv:2609.03652v1 Announce Type: new  Abstract: Synthetic data augmentation has become a common strategy for addressing class imbalance in NLP, but most approaches focus on the quantity and diversity of generated examples rather than their geometric relationship to real training data. We investigate this question in the context of discourse pragmatic function classification, a task where data sparsity is a structural feature rather than a collection artefact. Using 410 manually annotated instances of the English word look drawn from the British National Corpus, spanning four functions: Attention Signal, Directive, Discourse Marker, and Interjection. We generate synthetic training examples with Llama 3.1 and partition them by their cosine distance from real training data in RoBERTa embedding space. We compare six training conditions that differ in the placement of synthetic examples relative to the empirical decision boundary, while holding augmentation quantity constant across conditi
    
[^49]: 《</think> 并不能停止推理：虚假思维链终止现象分析》

    </think> Doesn't Stop Reasoning: Analysis of Spurious CoT Termination

    [https://arxiv.org/abs/2609.03633](https://arxiv.org/abs/2609.03633)

    本文发现提前退出方法中注入的思考结束标记</think>并不总能真正终止推理，模型会在回答阶段继续产生类似推理的“虚假CoT终止”行为，且该延续片段的长度与提前退出节省的推理token数成正比，其根源可能是模型对注入EoT的注意力不足。

    

    思维链推理提升了大型推理模型（LRMs）在复杂任务上的表现，但常常产生冗长且冗余的推理轨迹。近期的免训练提前退出方法通过选择一个中间停止点来缩短这些轨迹。我们研究了其中一种策略：在该点注入思考结束标记（EoT，</think>）以触发从推理到回答阶段的转换，并发现注入的EoT并不总是能引发干净的回答阶段。在模型重新生成另一个EoT之前，回答阶段的生成可能会继续进行，且该重新生成的EoT之前的片段长度会随提前退出所节省的推理token数量而增长，并表现出持续的推理行为。我们将这种现象称为“虚假CoT终止”，即类似推理的生成延续到了回答阶段。我们假设对注入的EoT关注不足是导致虚假CoT终止的原因之一，并通过Exit-token注意力偏……（原文在此处截断）

    arXiv:2609.03633v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) reasoning improves large reasoning models (LRMs) on complex tasks but often produces long, redundant traces. Recent training-free early-exit methods shorten these traces by choosing an intermediate point to stop reasoning. We study one such strategy that injects an end-of-think token (EoT, ) at this point to trigger the reasoning-to-answering transition, and find that the injected EoT does not always induce a clean answering phase. Answering-phase generation can continue before the model regenerates another EoT, with the span preceding this regenerated EoT scaling with the reasoning tokens saved by early exit and exhibiting continued reasoning behavior. We call this spurious CoT termination, where reasoning-like generation continues into the answering phase. We hypothesize that insufficient attention to the injected EoT contributes to spurious CoT termination and probe this hypothesis with Exit-token Attention Bi
    
[^50]: 记忆与重加权：利用经验记忆与置信度估计增强多智能体辩论

    Remember and Reweight: Enhancing Multi-Agent Debate with Experience Memory and Confidence Estimation

    [https://arxiv.org/abs/2609.03619](https://arxiv.org/abs/2609.03619)

    该论文提出R²-MAD框架，通过为多智能体辩论引入经验记忆机制，利用辩论状态感知的检索策略动态校准概念先验，并结合置信度估计对回答进行重加权，有效缓解了多数智能体收敛于错误答案时“共享误解”被放大的关键缺陷。

    

    多智能体辩论通过让多个智能体在讨论中迭代地改进自身回答，从而提升大语言模型的推理能力。然而，MAD存在一个被称为“共享误解”的关键脆弱性：当大多数智能体最初收敛于某个错误答案时，辩论过程往往会放大而非纠正该错误。现有方法主要针对同伴偏差问题，但未能解决智能体固有的有偏概念先验。为缓解这一系统性弱点，我们提出了R²-MAD（Remember and Reweight for Multi-Agent Debate），这是一个为智能体配备从过往辩论中积累的经验记忆的框架。R²-MAD通过两个互补机制对两种失败模式进行干预：一种辩论状态感知的检索策略，根据当前共识水平检索相关历史证据，动态校准概念先验；随后将这些检索到的证据与置信度估计相结合，对智能体的回答进行重加权，从而在共识偏离正确答案时有效抑制错误观点的放大。

    arXiv:2609.03619v1 Announce Type: cross  Abstract: Multi-agent debate (MAD) improves the reasoning capabilities of large language models by having multiple agents iteratively refine their responses through discussion. However, MAD suffers from a critical vulnerability known as shared misconception: when a majority of agents initially converge on an incorrect answer, the debate process tends to amplify rather than correct the error. Existing methods primarily address peer skew but leave the agents' inherently biased concept priors unaddressed. To mitigate this systematic weakness, we propose R$^2$-MAD (Remember and Reweight for Multi-Agent Debate), a framework that equips agents with an experience memory accumulated from past debates. R$^2$-MAD intervenes on both failure modes through two complementary mechanisms: A debate-state-aware retrieval policy dynamically calibrates the concept prior by retrieving relevant historical evidence based on the current consensus level. Then these retr
    
[^51]: KhatianDoc：一个诊断多模态大语言模型在孟加拉语法律土地记录上失败的人工验证基准

    KhatianDoc: A Human-Verified Benchmark Diagnosing Multimodal LLM Failure on Bengali Legal Land Records

    [https://arxiv.org/abs/2609.03597](https://arxiv.org/abs/2609.03597)

    本文提出了KhatianDoc基准，首次通过符号识别、进制转换、字段提取和法律问答四项任务，系统评估多模态大语言模型读取孟加拉国手写土地记录中独特的十六进制分数系统的能力，并揭示其在该领域的失败表现。

    

    孟加拉国的土地所有权以 Ana-Ganda-Kora-Kranti-Til 系统记录，这是一种基数为16的位置分数系统，拥有专用的Unicode字形，没有主流字体的支持，也没有被任何OCR流水线或分词器所覆盖。承载这些分数的手写记录 RS Khatians 是数百万地块的权威产权记录，也是民事诉讼的常见对象，然而迄今为止还没有任何基准测试探究过机器是否能够读取这些记录。我们提出了 KhatianDoc，这是一个基于来自孟加拉国 Munshiganj 土地办公室的107份真实 RS Khatian 记录构建的四任务基准：符号识别、十六进制到十进制的转换、结构化字段提取，以及基于1,634个问答对的法律文档问答。所有真实标注均由人工转录，并由土地法律从业者验证至完全一致，同时通过位置标记进行匿名化处理，这些标记保留了多跳问题所依赖的指代区别。我们评估了六个多模态大语言模型。

    arXiv:2609.03597v1 Announce Type: new  Abstract: Land ownership in Bangladesh is recorded in Ana-Ganda-Kora-Kranti-Til, a base-16 positional fraction system with dedicated Unicode glyphs, no mainstream font, and no coverage in any OCR pipeline or tokenizer. The handwritten records that carry these fractions, RS Khatians, are the authoritative title record for millions of parcels and a frequent subject of civil litigation, yet no benchmark has asked whether a machine can read one. We introduce KhatianDoc, a four-task benchmark built from 107 real RS Khatian records from the Vumi (land) Office of Munshiganj, Bangladesh: symbol recognition, base-16-to-decimal conversion, structured field extraction, and legal document question answering over 1,634 QA pairs. Ground truth was transcribed by hand, verified by a land-law practitioner to full agreement, and anonymized through positional tokens that keep the referential distinctions multi-hop questions depend on. We evaluate six multimodal LLMs
    
[^52]: 合成数据能将泰文OCR带向多远？

    How Far Can Synthetic Data Take Thai OCR?

    [https://arxiv.org/abs/2609.03595](https://arxiv.org/abs/2609.03595)

    通过受控文档重建流水线解耦合成数据“真实性”中的各项因素，发现字体多样性、二维结构和真实手写字形是合成数据迁移到真实泰文OCR的关键，并据此构建了无需真实OCR标签的泰文OCR模型Wayu-Paxa-OCR-Zero。

    

    我们研究了是什么因素使得合成OCR监督信号能够迁移到真实的泰文文档上，并利用所得见解构建了Wayu-Paxa-OCR-Zero——一个无需来自真实泰文文档页面的OCR标签即可完成适配的泰文OCR模型。合成数据能够以大规模提供精确标签，但“真实性”这一概念混淆了源域、页面上下文、字体排印、空间结构和字形变化等多个因素。我们通过一个受控的文档重建流水线将这些因素解耦，并在页面级和裁剪级两种训练方式下，在印刷体和手写泰文文档上对每种变体进行评估。结果表明：非文本上下文几乎没有一致的影响，而字体多样性、二维结构以及真实手写字形则能改善迁移效果；此外，源域匹配的效果依赖于训练粒度——在页面级训练下，域内重建可接近真实印刷体监督的效果（中位字符错误率1.82% 对比 1.31%），但在域外重建方面则表现不佳。

    arXiv:2609.03595v1 Announce Type: cross  Abstract: We investigate what makes synthetic OCR supervision transfer to real Thai documents and use the resulting insights to build Wayu-Paxa-OCR-Zero, a Thai OCR model adapted without OCR labels from real Thai document pages. Synthetic data provide exact labels at scale, but "realism" conflates source domain, page context, typography, spatial structure, and glyph variation. We disentangle these factors with a controlled document-reconstruction pipeline and evaluate each variant under page- and crop-level training on printed and handwritten Thai documents. Non-text context has little consistent effect, whereas typeface diversity, two-dimensional structure, and real handwriting glyphs improve transfer; moreover, source-domain matching depends on training granularity, with in-domain reconstruction approaching real printed supervision under page-level training (1.82% versus 1.31% median character error rate) but underperforming out-of-domain reco
    
[^53]: HalluPeer：一个面向科学同行评审中幻觉检测的分类体系驱动基准测试

    HalluPeer: A Taxonomy-driven Benchmark for Detecting Hallucinations in Scientific Peer Reviews

    [https://arxiv.org/abs/2609.03580](https://arxiv.org/abs/2609.03580)

    该论文提出了HalluPeer——首个面向科学同行评审场景的幻觉检测基准，通过构建论文、真实评审与注入幻觉评审的对齐数据集以及同行评审专属的幻觉分类体系，揭示了现有检测器难以区分幻觉与合理批评的局限。

    

    学术同行评审规模的不断增长推动了将大语言模型（LLM）用作评审助手的实践，然而LLM可能会生成流畅但缺乏依据的论断，从而损害评审的可靠性。现有的幻觉基准测试并非为同行评审场景设计，因为在这一场景中，验证论断需要以冗长且技术性强的论文为依据。我们提出了HalluPeer，一个用于检测科学同行评审中幻觉的基准测试，它提供了论文内容、人工撰写的评审以及注入幻觉的评审三者对齐的数据三元组，并针对幻觉的检测、分类和定位进行了标注。我们的流程构建了面向同行评审的幻觉分类体系，识别评审上下文，并通过自动化过滤注入幻觉。在1.2万篇论文和3.8万条评审上的实验表明，现有检测器难以将幻觉与合理的批评意见区分开来，而对真实评审的评估则证明HalluPeer……

    arXiv:2609.03580v1 Announce Type: new  Abstract: The growing scale of academic peer review has motivated the use of Large Language Models (LLMs) as review assistants, yet LLMs can generate fluent but unsupported claims that undermine review reliability. Existing hallucination benchmarks are not designed for peer review, where verification requires grounding claims in long, technical papers. We introduce HalluPeer, a benchmark for detecting hallucinations in scientific peer reviews, providing aligned triples of paper content, human-written reviews, and hallucination-injected reviews, annotated for detection, classification, and localization. Our pipeline induces a peer-review-specific hallucination taxonomy, identifies review contexts, and injects hallucinations with automated filtering. Experiments on 12K papers and 38K reviews show that existing detectors struggle to separate hallucinations from legitimate critique, while evaluation on authentic reviews demonstrates that HalluPeer-def
    
[^54]: 语言、语言模型与我们究竟在谈论什么

    Language, Language Models, and What We're Talking About

    [https://arxiv.org/abs/2609.03577](https://arxiv.org/abs/2609.03577)

    本文以意大利语语言模型为例，主张必须首先区分“作为技术产品的语言模型”与“作为语言研究工具的语言模型”，才能回答“我们究竟希望语言模型生成什么样的语言”这一核心问题。

    

    语言模型通常被作为技术制品来讨论，但它们显然受到训练数据所传递的语言世界的影响。以意大利语语言模型为例证，我想引起人们对以下系统性质的关注：这些系统是通过在翻译数据和合成数据上训练和专精化模型、并进一步加以人工筛选而形成的，而在同样非自然的数据上对它们进行测试又意味着什么。这些最终是意大利语的模型吗？它们是语言的模型吗？自然语言处理（NLP）还在关心语言吗？这些问题引出了另一个更具体的问题：我们究竟希望语言模型产出什么样的语言？我认为，如果不首先更清晰地区分“作为技术产品设计”的语言模型与“作为研究语言本身工具设计”的语言模型，这个问题就无法得到回答。此后答案或许会多种多样，我们所谈论的语言或许也会有所不同。

    arXiv:2609.03577v1 Announce Type: new  Abstract: Language models are commonly discussed as technical artefacts, but they are obviously shaped by the linguistic worlds conveyed by data during their training. Using Italian language models as evidence, I want to bring attention to the nature of the systems which result from training and specialising models on translated and synthetic data, and further curating them, and to the meaning of testing them on equally unnatural data. Are these eventually models of Italian? Are they models of language? Does NLP still care about language? These questions yield another, more concrete question: what language do we actually want language models to produce? I argue that this question cannot be answered if we do not first consider a clearer distinction between language models designed as technical products and language models designed as tools for studying language itself. The answers then might be diverse, the languages we are talking about might be d
    
[^55]: 迷失于语序重排：多语言大语言模型在语义保持扰动下的结构敏感性

    Lost in Reordering: Structural Sensitivity of Multilingual LLMs under Semantics-Preserving Perturbations

    [https://arxiv.org/abs/2609.03511](https://arxiv.org/abs/2609.03511)

    该研究提出了基于 GSM8K 构建的基准数据集 IndicReStruct，通过对印地语和马拉雅拉姆语施加成分重排与语态转换等语义保持扰动，揭示了多语言大语言模型在结构变化下数学推理能力会出现持续且显著的退化。

    

    大语言模型（LLMs）展现出强大的多语言推理能力，但它们对语义保持的结构变化的鲁棒性仍未得到充分探索，尤其是对于语序相对自由的语言。我们使用两种基于语言学理论的扰动设置——受限的成分重排和主动-被动语态转换——在印地语和马拉雅拉姆语中研究了多语言大语言模型的结构敏感性。我们引入了一个基准数据集 IndicReStruct，包含两个变体：GSM8K-Reordered 和 GSM8K-Voice，它们由 GSM8K 构建并在重构过程中保持了语义含义。在六个最先进的大语言模型和多种提示策略上的实验中，我们观察到在结构扰动的输入下，数学推理性能出现了持续且显著的下降。为了进一步理解这些失败，我们进行了定性错误分析，并利用残差流激活开展了机制可解释性实验。

    arXiv:2609.03511v1 Announce Type: new  Abstract: Large Language Models (LLMs) demonstrate strong multilingual reasoning performance, yet their robustness to semantics-preserving structural variation remains underexplored, particularly for relatively free word-order languages. We investigate the structural sensitivity of multilingual LLMs using two linguistically grounded perturbation settings in Hindi and Malayalam: constrained constituent reordering and active-passive voice transformation. We introduce a benchmark dataset IndicReStruct, with two variants, GSM8K-Reordered and GSM8K-Voice, constructed from GSM8K while preserving semantic meaning. Across six state-of-the-art LLMs and multiple prompting strategies, we observe consistent and significant degradation in mathematical reasoning performance under structurally perturbed inputs. To further understand these failures, we perform qualitative error analysis and mechanistic interpretability experiments using residual-stream activation
    
[^56]: 基于合成语音构建与评估固定音色泰语语音合成系统

    Building and Evaluating Fixed-Voice Thai TTS from Synthetic Speech

    [https://arxiv.org/abs/2609.03502](https://arxiv.org/abs/2609.03502)

    该论文提出将大型声音克隆模型作为可编程数据源，仅凭15秒声音参考生成合成语音来训练紧凑的固定音色泰语TTS学生模型，并系统研究了文本准备、质量过滤、拒绝采样和前端选择等流水线设计对模型效果的影响及教师模型残留的局限性。

    

    在低资源环境下，部署TTS通常需要在两种方案之间做出选择：推理成本高昂的大型声音克隆模型，或需要特定说话人语料库的紧凑固定音色系统。我们研究了第三条路线：将大型声音克隆模型用作可编程的数据源，把简短的声音参考（例如15秒）转化为完全在合成语音上训练的紧凑固定音色学生模型。这一设定使得流水线设计变得至关重要：教师的错误会成为训练目标，而过滤失败的生成结果可能会降低对困难文本的覆盖率。泰语还带来了额外的挑战，包括词边界歧义、词汇声调、人名与外来词、数字的口语化表达以及泰英混读。我们研究了文本准备、合成生成、质量过滤、拒绝采样和前端选择如何影响最终的学生模型，以及教师模型的局限性在哪些方面依然存在。我们使用CER、挑战集关键词……（原文摘要在此处截断）

    arXiv:2609.03502v1 Announce Type: cross  Abstract: In low-resource settings, deploying TTS typically requires choosing between a large voice-cloning model with costly inference or a compact fixed-voice system that requires a speaker-specific corpus. We study a third route: using a large voice-cloning model as a programmable data source to turn a short voice reference (e.g., 15 seconds) into a compact fixed-voice student trained entirely on synthetic speech. This setting makes pipeline design consequential: teacher errors become training targets, while filtering failed generations can reduce coverage of difficult texts. Thai further introduces challenges from ambiguous word boundaries, lexical tone, names and loanwords, numeric verbalization, and Thai-English code-switching. We study how text preparation, synthetic generation, quality filtering, rejection sampling, and frontend choices affect the resulting student, and where teacher limitations remain. We evaluate CER, Challenge-Set Key
    
[^57]: 知识图谱嵌入的模式过度泛化问题

    Pattern Over-Generalization of Knowledge Graph Embedding

    [https://arxiv.org/abs/2609.03487](https://arxiv.org/abs/2609.03487)

    本文揭示了知识图谱嵌入中的模式过度泛化问题，并提出PogRE方法，通过稠密线性变换和复合操作进行关系表示，有效缓解该问题。

    

    知识图谱嵌入（KGE）通过将实体和关系投影到低维向量空间中，展示了其在预测知识图谱（KG）中缺失链接方面的有效性。对于KGE模型而言，有效捕获知识图谱中固有的推理模式至关重要，例如对称性/反对称性、反转和组合等模式。尽管近期的KGE模型在建模这些多样化模式方面表现出强大的能力，但它们受到模式过度泛化所带来的固有局限性的困扰：仅从单个模式实例学习到的嵌入，不可避免地会将该模式泛化到所有相关实例上，即将模式普遍化。为了解决这一问题，我们提出了PogRE（模式过度泛化鲁棒嵌入），这是一种简单而有效的方法，利用稠密线性变换和复合操作来进行关系表示。我们的理论分析表明，稠密...

    arXiv:2609.03487v1 Announce Type: cross  Abstract: Knowledge graph embedding (KGE) demonstrates its effectiveness for predicting missing links in knowledge graphs (KGs) by projecting entities and relations into a low-dimensional vector space. It is crucial for KGE models to effectively capture inference patterns (patterns) inherent in KGs, such as symmetry/antisymmetry, inversion and composition. Although recent KGE models exhibit strong capabilities in modeling such diverse patterns, they suffer from inherent limitations stemming from pattern over-generalization, where embeddings learned from only a single pattern instance inevitably generalize that pattern to all related instances, i.e., generalize the pattern universally. To address this issue, we propose PogRE (Pattern Over-Generalization Robust Embedding), a simple but effective method that utilizes dense linear transformations and compound operations for relation representation. Our theoretical analysis demonstrates that a dense 
    
[^58]: 当用户不提问时：对话式智能体中上下文驱动的记忆检索基准测试

    When Users Don't Ask: Benchmarking Context-Driven Memory Retrieval in Conversational Agents

    [https://arxiv.org/abs/2609.03467](https://arxiv.org/abs/2609.03467)

    该论文提出了对话式记忆基准LOCOMO-CONV，通过对话式、隐含式、反事实和组合式四种查询风格，揭示了问答式基准测试所忽视的记忆检索差距，并发现强检索能力并不完全等同于高质量的对话响应。

    

    大型语言模型（LLM）越来越多地被部署为长时程对话式智能体，这推动了对记忆系统日益增长的关注。然而，现有基准测试主要通过问答式的探测方式来评估记忆，而非在真实对话场景中的实际使用。我们提出了LOCOMO-CONV，这是一个基于LoCoMo派生的对话式记忆基准，包含四种查询风格：对话式、隐含式、反事实式和组合式。我们在五个代表性记忆系统上评估了检索召回率和端到端响应质量。实验表明，对话式的表述方式暴露了问答式基准所忽视的大量检索差距，尤其是在隐含式和组合式查询上；多方面查询改写可以缩小原始对话轮次记忆的差距，但对抽象化记忆则无效。我们进一步发现，强大的检索能力并不能完全转化为响应质量，且隐含式查询表现出“静默接地”现象，即记忆在……（原文摘要在此处截断）

    arXiv:2609.03467v1 Announce Type: cross  Abstract: Large language models (LLMs) are increas- ingly deployed as long-horizon conversational agents, motivating growing interest in mem- ory systems. However, existing benchmarks primarily evaluate memory through QA-style probing rather than in-situ conversational usage. We introduce LOCOMO-CONV, a conversa- tional memory benchmark derived from Lo- CoMo with four query styles: dialog, implicit, counterfactual, and composed. Across five rep- resentative memory systems, we evaluate both retrieval recall and end-to-end response qual- ity. Our experiments show that conversational framing exposes substantial retrieval gaps over- looked by QA benchmarks, especially on im- plicit and composed queries, which multi-facet query rewriting narrows for raw-turn mem- ory but not abstractive memory. We further find that strong retrieval does not fully trans- late into response quality, and that implicit queries exhibit silent grounding, where mem- ory imp
    
[^59]: 何时检索有益：面向单轮心理健康问答的选择性检索

    When Retrieval Helps: Selective Retrieval for Single-Turn Mental-Health QA

    [https://arxiv.org/abs/2609.03454](https://arxiv.org/abs/2609.03454)

    该研究针对单轮心理健康问答提出一种轻量级选择性检索策略，通过心理教育需求、应对需求、回答具体性三个效用维度及安全触发机制判断检索的必要性，从而在发挥检索增强优势的同时避免其负面影响。

    

    检索增强生成（RAG）能够提升大语言模型回答的具体性和事实依据，但在单轮心理健康问答中，其效果并非始终有益，因为用户提问往往同时涉及情绪困扰、治疗关切以及安全敏感需求。我们研究了检索在心理健康问答中何时有益、何时有害，以及一种轻量级的选择性检索策略能否更好地控制这种权衡。我们通过三个基于草稿的条件化效用维度来量化检索需求：心理教育需求、应对需求和回答具体性，并结合基于规则的安全触发机制。借鉴coTherapist等基于心理治疗框架的RAG系统，我们构建了一个紧凑且可控的指南语料库，涵盖应对策略、心理教育和安全资源。我们使用QLoRA在MentalChat16K上对指令微调生成器进行微调，并比较了闭卷（Closed-book）、始终检索（Always Retrieval）以及……（摘要原文在此处截断）

    arXiv:2609.03454v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) can improve the specificity and grounding of large language model responses, but its effect is not uniformly beneficial in single-turn mental-health question answering, where user queries often combine emotional distress, treatment concerns, and safety-sensitive needs. We study when retrieval helps or hurts mental-health QA, and whether a lightweight selective retrieval policy can better control this trade-off. We operationalize retrieval need using three draft-conditioned utility dimensions: psychoeducational need, coping need, and response specificity, together with a rule-based safety trigger. Following psychotherapy-grounded RAG systems such as coTherapist, we construct a compact and controllable guideline corpus comprising coping-strategy, psychoeducational, and safety resources. We fine-tune an instruction-tuned generator on MentalChat16K using QLoRA and compare Closed-book, Always Retrieval, an
    
[^60]: 预算化继承式智能体记忆验证中的计划指针与记录指令形式

    Plan Pointers and Record-Directive Form in Budgeted Verification of Inherited Agent Memory

    [https://arxiv.org/abs/2609.03450](https://arxiv.org/abs/2609.03450)

    该论文通过十二项注册研究发现，写入智能体记忆库的指令形式（准则、裸ID或指针）会以高度模型依赖的方式显著影响预算受限下的记录选择，长度匹配准则可带来35分的提升，但附加ID可能完全抵消准则的效果。

    

    arXiv:2609.03450v1 公告类型：cross。摘要：一个继承了六条单行记忆的智能体在行动前最多只能拉取一条存档的源记录；写入存储中的指令可以引导这一选择：可以是指向该记录的指针、识别该记录的准则，或两者兼有。在同一仪器谱系上的十二项注册研究（共14,760次尝试）中，我们测量了每种指令形式下请求的去向。在六个直接提供商模型上，长度匹配的准则比裸ID高出+35.0个点 [+31.2, +38.8]（研究D）；而在九个模型的OpenRouter服务面板上，该对比未能通过注册的优越性规则（研究E）。在三个Claude模型上，附加ID会抵消准则的效果（Opus 5: 从40/40降至0/40；研究F-x）；六次字节匹配的编辑使每个精确字符串都产生了各自的效应（研究G），并且在每单元八十次运行的重跑中，三十个复现对比中有十五个处于误差范围内，十五个未获解决，没有一个超出范围（研究G'）。批准行（在Opus 5上+96.0个点）以及一个（摘要在此处截断）

    arXiv:2609.03450v1 Announce Type: cross  Abstract: An agent that inherits six one-line memories may pull at most one archived source record before acting; a directive written into the store can steer that choice: a pointer to the record, a criterion that identifies it, or both. Across twelve registered studies on one instrument lineage (14,760 attempts) we measured where the request goes under each form. On six direct-provider models a length-matched criterion exceeded a bare id by +35.0 points [+31.2, +38.8] (Study D); the contrast failed its registered superiority rule on a nine-model OpenRouter-served panel (Study E). Appending the id cancelled the criterion on three Claude models (Opus 5: 40/40 to 0/40; Study F-x); six byte-matched edits gave each exact string its own effect (Study G), and a re-run at eighty runs per cell left fifteen of thirty replication contrasts within the margin, fifteen unresolved and none beyond (Study G'). A ratification line (+96.0 points on Opus 5) and a 
    
[^61]: 问题本身，而非路径：大语言模型推理轨迹中的预算与难度混淆因素

    It's the Problem, Not the Path: Budget and Difficulty Confounds in LLM Reasoning Trajectories

    [https://arxiv.org/abs/2609.03436](https://arxiv.org/abs/2609.03436)

    该研究提出重启控制的截断探针方法，证明大语言模型推理轨迹中所谓的“突破时刻”和“早期注定失败”大多是预算与难度造成的混淆——178个问题-模型组合中仅1个真正存在前缀特有价值，且在同等token预算下延续自身推理前缀几乎总是优于从头重启。

    

    大语言模型的推理轨迹通常被解读为包含“突破”时刻和早期可判定的命运。这两种解读都建立在缺少主张层面反事实控制的测量之上；我们提供了这两种控制。首先，一个重启控制的截断探针将“解法适合延续预算”与“前缀携带全新计算无法换取的价值”区分开来，在匹配的总生成token预算下，比较每个锚点的延续求解率与从头重启曲线。将该探针应用于178个问题-模型单元格（89个MATH问题 × 两个小型开源模型，这是一个结果盲但针对难度的队列），178个单元格中恰好只有1个作为前缀受限而存活；重启的剂量-反应关系能够区分计算饥饿型模型与能力受限型模型；并且只要匹配预算位于重启网格之内，延续模型自身的前缀总是优于从头重启（9/9）——主要是计算压缩……（摘要原文在此处截断）

    arXiv:2609.03436v1 Announce Type: cross  Abstract: Reasoning traces of large language models are widely read as containing "breakthrough" moments and early-legible fates. Both readings rest on measurements missing a counterfactual control at the level of the claim; we supply both controls. First, a restart-controlled truncation probe separates when a solution fits the continuation budget from when a prefix carries value that fresh computation cannot buy, comparing per-anchor continuation solve rates against from-scratch restart curves at matched total generated-token budget. Applied to 178 problem-model cells (89 MATH problems x two small open models, an outcome-blind but difficulty-targeted cohort), exactly 1 of 178 cells survives as prefix-limited; restart dose-response separates a compute-starved model from a capability-limited one; and wherever the matched budget lies inside the restart grid, continuing the model's own prefix beats restarting (9 of 9) -- predominantly compute compr
    
[^62]: 分析-评判解耦：一种面向复杂多步骤创造力任务的基于大语言模型的自动化创造力评估器

    Decoupled Analysis-Judging: An Automated Creativity Evaluator Using LLMs in Complex Multi-step Creativity Tasks

    [https://arxiv.org/abs/2609.03432](https://arxiv.org/abs/2609.03432)

    该论文提出CreaEval，通过将“大模型评判者”解耦为记忆增强的结构化分析与基于证据的评判两个阶段，有效缓解冗长偏差与宽容偏差，实现复杂多步骤创造力任务的自动化可靠评估。

    

    对创造力任务的自动化评估对“大语言模型作为评判者”（LLM-as-a-Judge）而言仍具挑战性，因为大语言模型容易受到冗长偏差、宽容偏差等偏见的影响。这些局限性在情境关联且程序结构化的任务（CGPST）中尤为明显——这是一类复杂的多步骤创造力任务，其步骤间依赖性、高度主观性和宽泛的评分范围导致评判更加不稳定且充满偏见。现有方法要么依赖特定任务的训练，要么直接应用LLM-as-a-Judge，二者在此类复杂场景下均难以保证评估的可靠性。为弥合这些差距，我们提出了CreaEval——一种面向CGPST的自动化创造力评估器，它将典型的LLM-as-a-Judge解耦为分析与评判两个环节。相应地，CreaEval包含两个关键阶段：记忆增强分析，由SoT-LLM将多步骤回答转换为结构化的评估证据，并融入跨步骤记忆；以及基于证据的评判（原文摘要在此处被截断）。

    arXiv:2609.03432v1 Announce Type: new  Abstract: Automated evaluation of creativity tasks remains challenging for LLM-as-a-Judge, as LLM is susceptible to biases such as verbosity bias and leniency bias. Such limitations are particularly evident in Contextually-Grounded and Procedurally-Structured Tasks (CGPST), a complex multi-step creativity task where inter-step dependencies, highly subjectivity, and wide scoring ranges lead to more unstable and biased judgments. Existing approaches either rely on task-specific training or directly apply LLM-as-a-Judge, both of which struggle to ensure reliable evaluation under such complexity. To bridge these gaps, we propose CreaEval, an automated creativity evaluator for CGPST that decouples typical LLM-as-a-Judge into analysis and judging. Correspondingly, CreaEval involves two critical phases: Memory-augmented Analysis, a SoT-LLM converts multi-step responses into structured evaluation evidence, incorporating cross-step memory; and Evidence-bas
    
[^63]: 随机注意力：重新思考用于高效推理的KV缓存驱逐

    Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning

    [https://arxiv.org/abs/2609.03430](https://arxiv.org/abs/2609.03430)

    该论文发现KV缓存驱逐中复杂的token重要性打分信号几乎毫无贡献，提出完全不计算分数、在每个注意力头内均匀随机驱逐的“随机注意力”方法，其性能匹配最强基线的同时吞吐量提升32-43%，关键在于提示才是需要保留的脆弱部分，而推理轨迹凭借自身冗余性可自我保护。

    

    大型语言模型在需要长程推理的任务上取得了卓越的性能，但漫长的思维链使KV缓存成为严重的内存瓶颈。现有的KV缓存压缩方法都遵循同一个范式：通过某种估计来评估每个缓存token未来的重要程度并打分，然后保留得分最高的那些token。我们证明这种选择信号几乎没有贡献。随机注意力保留提示内容，并在每个注意力头内均匀随机地进行驱逐，完全不计算任何分数；在四个模型和六个推理任务上，该方法与最强的现有驱逐方法性能相当，同时在vLLM部署中实现了比其高32-43%的吞吐量。受控实验解释了这一现象，表明：1）提示内容是缓存中脆弱的部分，不同选择器之间的性能差距主要取决于它们的选择信号是否恰好保留了提示；2）推理轨迹本身通过两个层面的冗余性保护自己免受驱逐……

    arXiv:2609.03430v1 Announce Type: new  Abstract: Large language models achieve superior performance on tasks that require extended reasoning, but long chains of thought make the KV cache a severe memory bottleneck. Existing KV cache compression methods share one paradigm: score each cached token by some estimate of how much it will matter later, and keep the top-scoring ones. We show that the selection signal contributes almost nothing. Random Attention keeps the prompt and evicts uniformly at random within each attention head, computing no score at all; across four models and six reasoning tasks it matches the strongest prior evictor while serving 32-43% higher throughput than it in vLLM deployment. Controlled experiments explain this by showing that 1) the prompt is the fragile part of the cache, and most of the gap between selectors is just whether their selection signal happened to keep it; 2) the reasoning trace protects itself against eviction with redundancy at two levels, in th
    
[^64]: Lngram v2：具有可解释离散表示的潜在N-元语法记忆

    Lngram v2: Latent N-Gram Memory with Interpretable Discrete Representations

    [https://arxiv.org/abs/2609.03426](https://arxiv.org/abs/2609.03426)

    Lngram v2通过解耦记忆容量与骨干网络宽度、引入上下文感知的分组查询注意力读取机制以及零值Sink和反事实代理梯度等改进，在保留硬离散寻址的同时实现了记忆容量的独立扩展，并成功应用于300亿参数的视觉-语言模型。

    

    Transformer缺乏原生的查找机制，需要通过重复的密集计算来识别和复用局部静态模式。Lngram v1通过离散潜在n-元语法寻址引入了与分词器无关的条件记忆，但其记忆容量与骨干网络宽度相耦合，高昂的参数和激活成本限制了其可扩展性。我们提出Lngram v2，将路由数量、记忆维度和骨干网络宽度进行解耦，并引入上下文感知的分组查询注意力读取机制，从而实现记忆容量的独立扩展。零值Sink和反事实代理梯度进一步提升了读取的选择性和路由的可训练性，同时保留了硬离散寻址。在不同规模的视觉-语言模型（VLM）上的实验显示了一致的性能提升，包括成功扩展至300亿参数的模型。与Lngram v1相比，Lngram v2大幅降低了总参数量和激活……

    arXiv:2609.03426v1 Announce Type: new  Abstract: Transformers lack a native lookup mechanism, requiring repeated dense computation to recognize and reuse local static patterns. Lngram v1 introduces tokenizer-independent conditional memory through discrete latent n-gram addressing, but its memory capacity is coupled with the backbone width, limiting scalability due to high parameter and activation costs. We propose Lngram v2, which decouples the number of routes, memory dimension, and backbone width, and introduces a context-aware grouped-query attention readout to scale memory capacity independently. A zero-value Sink and counterfactual surrogate gradients further improve readout selectivity and routing trainability while preserving hard discrete addressing. Experiments across vision--language models (VLMs) of different scales show consistent improvements, including successful scaling to a 30B-parameter model. Compared with Lngram v1, Lngram v2 substantially reduces both total and acti
    
[^65]: 大型语言模型在多大程度上理解孟加拉语习语？

    To What Extent Do Large Language Models Understand Bangla Idioms?

    [https://arxiv.org/abs/2609.03410](https://arxiv.org/abs/2609.03410)

    本文构建了首个大规模孟加拉语习语基准数据集，并通过释义改写、跨度检测和意义识别三项任务的评估发现，没有任何大语言模型能在所有任务上全面领先，不同模型各有所长。

    

    习语表达是自然语言的重要组成部分，反映着文化细微差异，并给计算模型带来了独特挑战，尤其是在低资源语言中。在本文中，我们提出了首个大规模孟加拉语习语基准数据集，并辅以一个用于习语意义识别的合成多项选择题（MCQ）数据集。我们采用零样本和少样本提示策略，对近期的大型语言模型（LLMs）在三项与习语相关的任务上进行了全面评估：释义改写、习语跨度检测和意义识别。我们的结果显示模型性能存在显著差异，没有任何单一的LLM能在所有任务上持续优于其他模型。值得注意的是，Phi-4-mini-instruct在释义改写方面表现卓越，Kimi-K2-32b-instruct在跨度检测方面表现最佳，而Gemini-2.5-flash在意义识别方面最为出色。我们相信，我们的数据集和分析将为该领域提供宝贵的资源。

    arXiv:2609.03410v1 Announce Type: new  Abstract: Idiomatic expressions are an integral part of natural language, reflecting cultural nuances and posing unique challenges for computational models, particularly in low-resource languages. In this paper, we present the first large-scale benchmark dataset of Bangla idioms, complemented by a synthetic multiple-choice question (MCQ) dataset for idiom meaning identification. We conduct a comprehensive evaluation of recent large language models (LLMs) across three idiom-related tasks: paraphrasing, idiom span detection, and meaning identification, leveraging zero-shot and few-shot prompting strategies. Our results reveal substantial variability in model performance, with no single LLM consistently outperforming others across all tasks. Notably, Phi-4-mini-instruct excels in paraphrasing, Kimi-K2-32b-instruct in span detection, and Gemini-2.5-flash in meaning identification. We believe that our datasets and analyses will provide valuable resourc
    
[^66]: TabScope：面向表格问答的问题自适应范围选择

    TabScope: Question-Adaptive Scope Selection for Table Question Answering

    [https://arxiv.org/abs/2609.03395](https://arxiv.org/abs/2609.03395)

    该论文提出TabScope框架，通过操作感知的表格分解和问题类型预测，在大语言模型表格问答中动态选择局部子表推理或全表推理，显著提升长表格问答的准确率，并贡献了基于真实世界长表格的SLQA评测基准。

    

    大型语言模型（LLMs）在表格问答任务上展现出强大的性能，但其准确率往往随着表格规模的增大而下降。我们发现这种性能下降在不同问题类型之间并不均匀：对定位敏感的问题尤其容易受到表格中无关内容的干扰，而需要更广泛证据的问题则仍可能受益于全表推理。基于这一观察，我们提出了一个问答自适应框架，可在局部推理与全表推理之间进行动态选择。该框架通过操作感知的表格分解来构建针对特定问题的子表，并利用预测出的问题类型来确定合适的推理模式。此外，我们进一步引入了用于评估证据选择的银标准参考子表，并构建了SLQA——一个基于真实世界长表格的基准数据集。在WikiTQ和SLQA上的实验表明，定位方法对查找类问题尤其有效……

    arXiv:2609.03395v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have shown strong performance on table question answering, yet their accuracy often degrades as table size increases. We find that this degradation is not uniform across question types. Localization-sensitive questions are particularly affected by irrelevant table content, while questions requiring broader evidence may still benefit from full-table reasoning. Based on this observation, we propose a question-adaptive framework that dynamically selects between localized and full-table reasoning. The framework constructs question-specific sub-tables through operation-aware table decomposition and uses the predicted question type to determine the appropriate reasoning mode. We further introduce silver reference sub-tables for evaluating evidence selection and construct SLQA, a benchmark based on real-world long tables. Experiments on WikiTQ and SLQA show that localization is particularly effective for lookup an
    
[^67]: 情感的明暗对照：基于评价理论的情绪对比基准

    Chiaroscuro for Emotions: A Contrastive Emotion Benchmark Grounded in Appraisal Theory

    [https://arxiv.org/abs/2609.03394](https://arxiv.org/abs/2609.03394)

    提出了基于评价理论的情绪对比基准CHIARO（1,000句人工标注，每个场景中同一事件引发两人相反情绪），评测发现最强LLM仅达67.3宏F1、现有情绪分类器接近随机水平，且该基准还可作为训练信号提升下游分类器性能。

    

    现有情绪识别基准通常对每段文本只预测一种情绪，忽视了许多现实场景——在同一个共同事件下，两个人可能产生截然相反的情绪。例如，一个孩子因兴奋而踢到了前排座椅，而前排乘客却因此变得愤怒。我们提出CHIARO，一个包含1,000条人工标注句子的情绪对比推理基准，其设计植根于评价理论。每个场景描述一个因果触发事件，该事件使一个人产生正面情绪，而使另一个人产生负面情绪，情绪类别取自一个十类分类体系。我们对七个前沿大语言模型和四个现成的情绪分类器进行了评测。最强的LLM仅达到67.3的宏平均F1，远低于人类一致性水平，而现有情绪分类器的表现接近随机水平。除评测之外，CHIARO还可作为训练信号：与现有情绪语料库结合后，所得的下游分类器不仅在CHIARO本身上有所提升，还在十个外部基准中的六个上取得改进。

    arXiv:2609.03394v1 Announce Type: new  Abstract: Emotion recognition benchmarks often predict one emotion per text, missing many real-world scenarios where two people arrive at opposing emotions from a single shared event. For example, a child kicks the seat in front of her in excitement while the passenger ahead grows angry. We introduce CHIARO, a 1,000 human-annotated sentence benchmark for contrastive emotion inference grounded in appraisal theory. Each scene describes one causal trigger eliciting a positive emotion in one person and a negative emotion in the other, drawn from a ten-class taxonomy. We benchmark seven frontier LLMs and four off-the-shelf emotion classifiers. The strongest LLM reaches 67.3 macro-F1, well below human agreement, while existing emotion classifiers score near chance. Beyond evaluation, CHIARO also serves as a training signal. When combined with an existing emotion corpus, the resulting downstream classifier improves on CHIARO itself and on six of ten exte
    
[^68]: FrameBench：基于框架语义学的语言理解基准

    FrameBench:A Language Understanding Benchmark Based on Frame Semantics

    [https://arxiv.org/abs/2609.03370](https://arxiv.org/abs/2609.03370)

    该论文提出了基于框架语义学的新基准FrameBench，用于评估大语言模型能否像人类一样根据语境隐式区分同一动词所唤起的不同语义框架。

    

    在框架语义学中，句子理解被认为是通过将词汇意义与被称为“语义框架”的背景知识相关联来实现的，从而使读者能够用未明说的信息对文本进行隐式充实。近年来，大型语言模型（LLM）在广泛的下游任务中取得了强劲的表现。然而，它们能否再现人类在理解过程中自然进行的这类隐式充实，目前仍不清楚。为了解答这一问题，我们提出了FrameBench，一个基于框架语义学的基准。FrameBench由多项选择题构成，用于测试模型能否区分同一动词在不同语境下所唤起的不同框架。我们利用FrameNet风格的资源以及结合母语者判断的生成-验证流程，构建了英语和日语两种语言的基准。我们在一系列多样化模型上的实验表明，小型模型面临显著挑战，而若干大型模型则表现较好。

    arXiv:2609.03370v1 Announce Type: new  Abstract: In frame semantics, sentence comprehension is assumed to proceed by relating lexical meaning to background knowledge called semantic frames, thereby enabling readers to implicitly enrich the text with unstated information. Recent large language models (LLMs) have achieved strong performance across a wide range of downstream tasks. However, it remains unclear whether they can reproduce the kinds of implicit enrichment that humans naturally make during comprehension. To address this question, we introduce FrameBench, a benchmark grounded in frame semantics. FrameBench consists of multiple-choice questions that test whether models distinguish the frames evoked by the same verb across contexts. We construct the benchmark for English and Japanese using FrameNet-style resources and a generation-and-verification pipeline with native-speaker judgments. Our experiments on a diverse set of models reveal challenges for small models, while several l
    
[^69]: 具有有据可依、忠实、一致、可操作理据的可问责人工智能：以VERDICT进行临床试验匹配的案例研究

    Accountable AI with Grounded, Faithful, Consistent, Actionable Rationales: A Case Study in Clinical Trial Matching with VERDICT

    [https://arxiv.org/abs/2609.03366](https://arxiv.org/abs/2609.03366)

    论文提出了基于大语言模型的VERDICT智能体用于临床试验匹配任务，并创新性地引入“自我忠实性”作为可问责性的自动测试方法，通过生成有据可依、忠实、一致且可操作的理据来实现AI决策的可问责性。

    

    可问责性意味着一项决策可以被审查、论证和质疑。大语言模型使这一点变得困难：流畅的输出可能缺乏依据、不完整，或与决策过程不一致。实现可问责性需要经验证的理据（决策是如何达成的）、假设（哪些内容是被假定而非已知的）、政策一致性（对相同的事实采用相同的处理方式），以及关键条件（什么因素会改变结果）。我们引入了“自我忠实性”作为可问责性的自动测试：改变关键条件应当会改变决策结果。我们通过临床试验匹配这一循证医学核心的高风险任务来研究可问责AI。尽管基于大语言模型的匹配器能够较为准确地将患者匹配到临床试验，但它们在应用决策政策时表现不一致，且产生的理据与其自身的决策不相符。我们介绍了VERDICT，这是一个基于大语言模型的智能体，它将决策任务及其常量（摘要在此处被截断）

    arXiv:2609.03366v1 Announce Type: new  Abstract: Accountability means a decision can be examined, justified, and contested. LLMs make this hard: fluent output may be ungrounded, incomplete, or unfaithful to the decision process. Achieving accountability requires verified rationales (how was the decision reached), assumptions (what was assumed rather than known), policy consistency (the same treatment for the same facts), and pivotal conditions (what would change the outcome). We introduce self-faithfulness as an automatic test of accountability: changing the pivotal conditions should change the decision.   We examine accountable AI through clinical trial matching, a high-stakes task central to evidence-based medicine. Although LLM-based matchers match patients to trials reasonably accurately, they apply decision policies inconsistently and produce rationales that are unfaithful to their own decisions.   We introduce VERDICT, an LLM-based agent that translates a decision task, its const
    
[^70]: ALRA：用于自回归语言模型基于Logit预训练蒸馏的自适应局部关系对齐

    ALRA: Adaptive Local Relational Alignment for Logit-Based Pre-training Distillation of Autoregressive Language Models

    [https://arxiv.org/abs/2609.03355](https://arxiv.org/abs/2609.03355)

    提出ALRA框架，通过让学生提议候选词元并以教师最可能词元作为锚点，同时根据教师概率分布广度自适应调整候选词元数量，从而改进自回归语言模型的局部logit蒸馏效果。

    

    基于Logit的知识蒸馏在自回归语言模型中通常是在整个词表上对齐教师模型和学生模型的下一词元分布。然而，这种全局目标忽略了可能词元候选之间的相对偏好。现有的局部方法通常仅从教师或学生单方选择候选词元。仅由教师选择可能会遗漏学生认为可能的词元，而仅由学生选择则可能在训练早期依赖不准确的排序。我们提出自适应局部关系对齐，这是一个结合学生提议与教师指导的位置特定框架。在每个有效预测位置，由学生提出可能的词元，同时将教师最可能的词元作为锚点纳入其中。ALRA根据教师在该候选集内相对于当前批次的概率分布广度来自适应地调整所选词元的数量。

    arXiv:2609.03355v1 Announce Type: cross  Abstract: Logit-based knowledge distillation for autoregressive language models usually aligns teacher and student next-token distributions over the entire vocabulary. However, this global objective overlooks relative preferences among likely token alternatives. Existing local approaches often select candidate tokens from either the teacher or the student alone. Teacher-only selection can miss tokens that the student considers likely, while student-only selection can rely on an inaccurate ranking early in training. We propose Adaptive Local Relational Alignment (ALRA), a position-specific framework combining student proposals with teacher guidance. At each valid prediction position, the student proposes likely tokens, while the teacher's most probable token is included as an anchor. ALRA adjusts the number of selected tokens according to how broadly the teacher distributes probability within this candidate set relative to the current batch. Adap
    
[^71]: 从零到英雄：面向亚美尼亚语的开放大语言模型生态系统

    From Zero to Hero: An Open LLM Ecosystem for Armenian

    [https://arxiv.org/abs/2609.03350](https://arxiv.org/abs/2609.03350)

    该研究发布了ArmWeb（437万篇亚美尼亚语新闻）与ArmSTEM（37.3万条英亚平行数理题目）两个数据集，并通过继续预训练Gemma-4-E4B构建出首个附带完整训练数据和配方的开源亚美尼亚语模型arm-gemma-e4b，其性能超越所有现有的开放亚美尼亚语模型。

    

    亚美尼亚语是一种形态丰富但资源稀缺的语言，其预训练数据十分匮乏，且目前尚无任何开放的亚美尼亚语大语言模型附带可复现所需的数据与训练配方。为填补这一空白，我们整理并发布了两个数据集。ArmWeb是一个经过广泛验证的语料库，包含437万篇亚美尼亚语新闻文档。ArmSTEM是一个英亚平行数据集，包含37.3万个附有分步解答的数学与科学题目，这些题目被翻译为亚美尼亚语，并通过保留答案的LLM判断和人工评估进行了双重验证。在Gemma-4-E4B上利用这些数据集进行继续预训练，得到arm-gemma-e4b，其性能超越了所有现有的开放亚美尼亚语模型以及未经适配的基础模型，并且是首个附带完整训练数据和配方的开放亚美尼亚语大语言模型。我们的消融实验表明，仅使用新闻数据进行继续预训练虽然能提升语言流畅度，却会侵蚀知识能力——我们在现有的亚美尼亚语模型中也观察到了这一模式；此外还表明，少量的……（原文在此处截断）

    arXiv:2609.03350v1 Announce Type: cross  Abstract: Pretraining data for Armenian, a morphologically rich and low-resource language, is scarce, and no open Armenian LLM has been released with the data and recipe needed to reproduce it. To address this gap, we curate and release two datasets. ArmWeb is an extensively validated corpus of 4.37M Armenian news documents. ArmSTEM is a parallel English-Armenian collection of 373K math and science problems with step-by-step solutions, translated into Armenian and verified through both answer-preserving LLM judgment and human evaluation. Continued pretraining of Gemma-4-E4B on these datasets yields arm-gemma-e4b, which outperforms every existing open Armenian model as well as its unadapted base, and is the first open Armenian LLM with complete training data and recipe. Our ablations show that news-only continued pretraining improves fluency while eroding knowledge, a pattern we also observe in existing Armenian models, and that a small share of 
    
[^72]: FPCO-Dialog：用于视觉语言模型中纠正与合作的多轮错误前提基准测试

    FPCO-Dialog: A Multi-Turn False-Premise Benchmark for Correction and Cooperation in Vision-Language Models

    [https://arxiv.org/abs/2609.03331](https://arxiv.org/abs/2609.03331)

    该论文提出了FPCO-Dialog基准，包含1,080张图像和10,800个问题轮次，首次系统评估视觉语言模型在多轮对话中面对持续重复错误前提时的纠正与合作行为，并通过CorrTP@K指标揭示了20个模型间显著的纠正能力差异。

    

    视觉语言模型越来越多地被部署在多轮对话场景中，用户在描述视觉内容时可能带有错误的假设。然而，现有评估很少单独研究当同一个基于视觉的错误前提在多轮对话中持续存在时模型如何响应。我们提出了FPCO-Dialog，一个用于评估视觉语言模型在重复错误前提下的纠正与合作行为的基准。FPCO-Dialog包含1,080张图像和10,800个问题轮次，按视觉复杂度、物体类别和错误前提类型进行分层，并采用10轮协议，其中正确的对话前缀之后跟随重复的错误前提指称表达。我们使用模型无关的协议和CorrTP@K（一种针对错误前提轮次的纠正率指标）评估了20个商业和开源视觉语言模型，并由两个独立的检测器进行评分。FPCO-Dialog揭示了各模型在总体纠正趋势方面存在显著且持续的跨模型差异。

    arXiv:2609.03331v1 Announce Type: new  Abstract: Vision-language models (VLMs) are increasingly deployed in multi-turn settings where users may describe visual content with incorrect assumptions. Yet existing evaluations rarely isolate how models respond when the same visually grounded false premise persists across dialogue turns. We introduce FPCO-Dialog, a benchmark for evaluating correction and cooperation behavior in VLMs under repeated false premises. FPCO-Dialog contains 1,080 images and 10,800 question turns, stratified by visual complexity, object category, and false-premise class, and uses a 10-turn protocol in which a correct dialogue prefix is followed by repeated false-premise referring expressions. We evaluate 20 commercial and open-source VLMs with a model-agnostic protocol and CorrTP@K, a correction-rate metric over false-premise turns, scored by two independent detectors. FPCO-Dialog reveals substantial and persistent cross-model differences in aggregate correction tend
    
[^73]: 少即是道德：一个用于背书行为中道德基础检测的CHARM框架

    Less Is Moral: A CHARMing Framework for Moral Foundations Detection in Endorsement Behaviour

    [https://arxiv.org/abs/2609.03330](https://arxiv.org/abs/2609.03330)

    本文提出CHARM框架，基于轻量级微调的大语言模型，将MAC交叉注意力、推理依据对齐和仇恨言论调制三个组件分别对应不同的心理学构念，从而以更低成本实现更稳健、更忠实的道德基础检测。

    

    道德语言在塑造在线背书和信息传播方面发挥着核心作用，然而现有的道德基础检测系统往往存在跨领域泛化能力差、推理依据薄弱以及依赖昂贵的基于提示的大语言模型（LLM）等问题。我们提出了CHARM，这是一个建立在轻量级微调大语言模型之上的、感知MAC与仇恨言论且对齐推理依据的道德基础检测框架，它整合了互补的道德 grounding、推理依据对齐以及极性感知的仇恨言论信号，以支持更稳健、更忠实的道德预测。与以往将计算与心理学理论相脱节的基于词典、微调或提示的检测器不同，CHARM的设计使每个组件——MAC交叉注意力、推理依据对齐和仇恨言论调制——都将一个独特的心理学构念予以操作化。使用30%的子样本

    arXiv:2609.03330v1 Announce Type: new  Abstract: Moral language plays a central role in shaping online endorsement and the diffusion of information, yet existing moral foundation detection systems often suffer from poor cross-domain generalization, weak rationale grounding, and reliance on costly prompting-based large language models (LLMs). We introduce CHARM, a MA\textbf{C}- and \textbf{H}ate-speech-\textbf{A}ware \textbf{R}ationale-aligned \textbf{M}oral foundation detection framework built on a lightweight fine-tuned LLM, which integrates complementary moral grounding, rationale alignment, and polarity-aware hate speech signals to support more robust and faithful moral prediction. Unlike prior dictionary-, fine-tune-, or prompt-based detectors, which decouple computation from psychological theory, CHARM is built so that each component -- MAC cross-attention, rationale alignment, and hate-speech modulation -- operationalizes a distinct psychological construct. Using a 30\% subsample
    
[^74]: 扰动如何传播：大语言模型鲁棒性的多层次分析

    How Perturbations Propagate: A Multi-Level Analysis of Robustness in Large Language Models

    [https://arxiv.org/abs/2609.03322](https://arxiv.org/abs/2609.03322)

    该论文首次从输出行为、隐藏状态几何和注意力头功能三个层面对大语言模型的扰动鲁棒性进行系统分析，揭示了扰动类型在模型内部表征中留下的可区分特征，而这些特征无法仅通过输出度量完全捕捉。

    

    语言模型会遇到拼写错误、文本损坏、词语替换和词序打乱等输入问题，然而其鲁棒性通常仅通过输出行为来评估。我们研究了六种自然和合成的输入扰动如何在三个层面上于纯解码器（decoder-only）语言模型中传播：输出行为、隐藏状态几何结构和注意力头功能。我们通过使用中心核对齐（CKA）和内在维度分析逐层几何结构，评估了四个GPT-2检查点和两个Qwen2.5检查点的行为效应，并考察了GPT-2中注意力头的响应。不同扰动类型产生了可区分的度量特征，这些特征无法被输出度量完全捕捉，且在所测试的检查点之间仅部分一致。复制分数与词符替换和打乱情形下的激活修补恢复效果尤为相关。梯度引导的HotFlip扰动还会造成更强的行为与表征层面的破坏。

    arXiv:2609.03322v1 Announce Type: new  Abstract: Language models encounter typos, corrupted text, altered words, and disrupted token order, yet robustness is usually evaluated only through output behavior. We study how six naturalistic and synthetic input perturbations propagate through decoder-only language models at three levels: output behavior, hidden-state geometry, and attention-head function. We evaluate behavioral effects across four GPT-2 and two Qwen2.5 checkpoints by analyzing layerwise geometry using centered kernel alignment and intrinsic dimension, and examine attention-head responses in GPT-2. Perturbation types produce distinguishable metric profiles that are not fully captured by output measures and are only partly consistent across the tested checkpoints. Copying scores are especially associated with activation-patching recovery under token substitution and shuffling. Gradient-guided HotFlip perturbations also cause stronger behavioral and representational disruption 
    
[^75]: 解耦话轮转换与语义：一种用于基于有限状态机的全双工对话的解耦数据方法

    Decoupling Turn-Taking from Semantics: A Decoupled Data Approach for Finite-State-Machine-Based Full-Duplex Dialogue

    [https://arxiv.org/abs/2609.03321](https://arxiv.org/abs/2609.03321)

    提出一种解耦数据方法，用真实人人口语对话学习话轮转换、用可配置的人机文本对话塑造语义行为，并通过基于规则的事件引导转换将口语对话序列化为FSM带，从而提升全双工对话中话轮转换的自然性。

    

    神经有限状态机（NFSM）框架为全双工对话提供了一条务实的路径：它在标准的下一词预测目标下，将话轮转换控制与响应生成序列化到单一因果带上，从而以较低的微调成本保留了语义能力。然而，其对合成文本数据的依赖从根本上限制了话轮转换的自然性，因为大语言模型（LLM）无法忠实模拟真实人类对话中细粒度的声学时间动态。在本工作中，我们提出一种解耦数据方法：从真实的人-人（HH）口语对话中学习话轮转换，同时通过可配置的人-代理（HA）文本对话塑造语义行为。为使该方法得以落地，我们引入了一种基于规则的事件引导数据转换方法，通过对话轮转换事件进行分类并应用确定性映射规则，将HH口语对话序列化为FSM带，从而……

    arXiv:2609.03321v1 Announce Type: new  Abstract: The Neural Finite State Machine (NFSM) framework offers a pragmatic path to full-duplex dialogue by serializing turn-taking control and response generation onto a single causal tape under the standard next-token prediction objective, thereby preserving semantic prowess at a low fine-tuning cost. However, its reliance on synthetic text data fundamentally limits turn-taking naturalness, as Large Language Models (LLMs) cannot faithfully simulate the fine-grained acoustic temporal dynamics of real human dialogues. In this work, we propose a decoupled data approach that learns turn-taking from real Human-Human (HH) spoken dialogues while shaping semantic behavior through configurable Human-Agent (HA) text dialogues. To operationalize this approach, we introduce a rule-based event-guided data transformation method that serializes HH spoken dialogues into FSM tapes by classifying turn-taking events and applying deterministic mapping rules, enab
    
[^76]: PACE：揭示用户请求中隐藏的冲突

    PACE: Towards Surfacing Hidden Conflicts in User Requests

    [https://arxiv.org/abs/2609.03293](https://arxiv.org/abs/2609.03293)

    该论文提出了PACE数据集，用于评估模型能否通过从知识库中检索隐式上下文证据，识别出使看似合理的用户请求变得不合适的潜在约束，从而实现基于冲突的个性化拒绝。

    

    个性化助手不仅应该遵守用户请求，还应该根据用户当前的情况评估这些请求是否合适。然而，先前的工作主要集中于准确执行请求，忽视了助手需要考虑上下文并进行基于冲突的拒绝这一需求。此外，现有的冲突或安全检测工作依赖于显式提供的因素，而现实场景中往往涉及必须从知识库中检索的隐式因素。为此，我们引入了个性化助手冲突评估数据集（PACE），用于评估模型能否识别出以自我中心知识或事件形式表达的潜在约束，这些约束会使看似合理的用户请求变得不合适。PACE将基于明确定义的用户画像的用户请求与自我中心的知识库事实配对，要求模型整合上下文证据来判断……

    arXiv:2609.03293v1 Announce Type: new  Abstract: Personalized assistants should not only comply with user requests but also assess whether those requests are appropriate given the user's current circumstances. However, prior work has primarily focused on accurately executing requests, overlooking the need for assistants to account for context and engage in conflict-based refusal. Furthermore, while existing work on conflict or safety detection relies on explicitly provided factors, real-world scenarios often involve implicit factors that must be retrieved from a knowledge base (KB). To this end, we introduce Personalized Assistants for Conflict Evaluation (PACE), a dataset for evaluating whether models can identify latent constraints, expressed as egocentric knowledge or events, that render seemingly reasonable user requests inappropriate. PACE pairs user requests grounded in well-defined personas with egocentric KB facts, requiring models to integrate contextual evidence to determine 
    
[^77]: 使用渐进式微调序列到序列Transformer的上下文泰米尔语拼写与语法纠错

    Contextual Tamil Spelling and Grammar Correction Using Progressively Fine-Tuned Sequence-to-Sequence Transformers

    [https://arxiv.org/abs/2609.03273](https://arxiv.org/abs/2609.03273)

    提出端到端序列到序列方法，通过在65万余对合成噪声-干净句子对上按四阶段渐进式调度微调mT5-small和mBART-50，实现能够处理主谓一致、时态一致和跨词连声等上下文错误的泰米尔语拼写与语法纠错。

    

    泰米尔语拼写和语法纠错具有挑战性，因为泰米尔语是一种黏着型低资源语言，具有丰富的动词形态变化、词边界处复杂的连声（sandhi，语音变化）规则，以及由247个不同字母组成的文字系统。先前的工作采用基于规则的方法、统计n-gram模型、最小编辑距离或结合Transformer重排序器的混合流水线来处理词级表面错误；这些方法无法可靠地处理上下文错误——如主谓一致、时态一致或跨词连声——因为这类错误需要句子级的理解能力。我们提出了一种端到端的序列到序列建模方法，并在一个包含多达657,720对噪声-干净泰米尔语句子对（涵盖十种错误类别）的合成语料库上对mT5-small和mBART-50进行微调。两个骨干模型均遵循相同的四阶段渐进式训练调度，每个阶段针对一个特定弱点：表面噪声、上下文语法、单点连声以及多点连声。

    arXiv:2609.03273v1 Announce Type: new  Abstract: Tamil spell and grammar correction is challenging because Tamil is an agglutinative low-resource language with rich verbal morphology, complex sandhi (phonetic transformation) rules at word boundaries, and a script of 247 distinct letters. Prior work targets word-level surface errors with rule-based methods, statistical n-gram models, Minimum Edit Distance, or hybrid pipelines with a transformer re-ranker; such methods cannot reliably handle contextual errors - subject-verb agreement, tense consistency, or cross-word sandhi - which require sentence-level understanding. We propose an end-to-end sequence-to-sequence formulation and fine-tune mT5-small and mBART-50 on a synthetic corpus of up to 657,720 noisy-clean Tamil sentence pairs spanning ten error categories. Both backbones follow the same four-stage progressive schedule, each stage targeting one weakness: surface noise (v2), contextual grammar (v3), single-site sandhi (v4), and mult
    
[^78]: MedQA-MM：医学视觉推理背后的捷径

    MedQA-MM: Shortcuts Behind Medical Visual Reasoning

    [https://arxiv.org/abs/2609.03261](https://arxiv.org/abs/2609.03261)

    该研究揭示了医学多模态选择题基准测试中的“推理膨胀”现象——模型无需真正的视觉推理，仅靠答案措辞、临床文本、图像文字、人工标注等捷径线索即可答对题目，全输入准确率为62.63%而仅文本设置即达53%以上。

    

    基准测试分数评价的是最终答案，而非得出答案所经由的路径。在医学多模态单项选择题（MCQs）中，这一区别至关重要，因为正确答案既可以由预期的图像发现所支持，也可以由基准测试中保留的各类线索所支持，包括答案措辞中的线索、非视觉临床文本、图像中可见的文字、人工标注或设备/上下文伪影。我们将由此产生的分数层面的过度解读称为“推理膨胀”。此处的“路径”指能够支持答案选择的可观察输入路径，而非关于模型隐藏认知的论断。在六个医学多模态MCQ数据集上，我们通过提示词侧和图像侧审计、模态消融以及保留医学目标和答案密钥的匹配修复方法，将候选线索与行为证据分离开来。在包含13种配置的开源模型面板中，全输入准确率为62.63%，而仅文本和仅选项设置即达到53%以上（摘要原文在此处截断）。

    arXiv:2609.03261v1 Announce Type: cross  Abstract: A benchmark score credits final answers, but not the route by which an item can be answered. In medical multimodal multiple-choice questions (MCQs), this distinction matters because a correct answer can be supported by the intended image finding or by benchmark-preserved cues in the wording of answers, non-visual clinical text, visible image text, artificial annotations, or device/context artifacts. We call the resulting score-level overinterpretation reasoning inflation. Here, a route is an observable input path that can support answer selection, not a claim about the model's hidden cognition. Across six medical multimodal MCQ datasets, we separate candidate cues from behavioral evidence through prompt- and image-side audits, modality ablations, and matched repairs that preserve the medical target and answer key. In a 13-configuration open-model panel, full-input accuracy is 62.63%, while text-only and options-only settings achieve 53
    
[^79]: 还有什么需要修改？探索对话生成工件中修订传播的高性价比测试时计算方法

    What Else Needs Fixing? Exploring Cost-Effective Test-Time Compute for Revision Propagation in Artifacts Generated Through Conversation

    [https://arxiv.org/abs/2609.03254](https://arxiv.org/abs/2609.03254)

    本文针对对话生成工件中修订传播这一新问题提出了一个基准，评估了九种测试时计算修订方法，发现基线方法可达68.3%–93%的准确率，并识别出最具成本效益的方法。

    

    大语言模型（LLM）通常通过对话中生成与修订的迭代循环帮助用户创建工件。这里的一个挑战是，当用户在修订时仅指定局部更改时，LLM 必须识别相关的依赖关系，并将修订传播到工件中所有受影响的部分。本文研究了 LLM 在对话生成的工件上的这种能力，其中工件上下文及其依赖关系可能嵌入在对话历史中。面向实际应用，我们还探索了这一新设置下高性价比的测试时计算方法。具体而言，我们为该设置引入了一个新的基准，并使用 gpt-oss-20b/120b、gpt-5.4-mini 和 qwen3.5-9b/27b/122b 在该基准上评估了九种修订方法，包括顺序反思和并行采样变体。结果表明，基线方法达到了 68.3%–93% 的准确率，而最具成本效益的方法是……

    arXiv:2609.03254v1 Announce Type: new  Abstract: Large Language Models (LLMs) often help users generate artifacts through iterative cycles of generation and revision in conversation. A challenge here is that, when users specify only a local change during revision, LLMs must instead identify the relevant dependencies and propagate the revision to all affected parts of the artifact. This paper studies this ability of LLMs on conversationally generated artifacts, where the artifact context and its dependencies may be embedded in the conversation history. Toward practical use, we also explore cost-effective test-time compute for this new setting. Specifically, we introduce a new benchmark for this setting, and evaluate nine revision methods, including sequential reflection and parallel sampling variants, using gpt-oss-20b/120b, gpt-5.4-mini, and qwen3.5-9b/27b/122b on the benchmark. The results show that baselines achieve accuracies of 68.3--93%, and the most cost-effective method is selec
    
[^80]: SGD-KV：摘要引导的KV缓存压缩

    SGD-KV: Summarization Guided KV Cache Compression

    [https://arxiv.org/abs/2609.03235](https://arxiv.org/abs/2609.03235)

    SGD-KV提出了一种感知注意力头的KV缓存压缩框架，通过新颖的块摘要诊断任务识别专门负责分层信息聚合的注意力头并据此分配缓存预算，在高达100万token的长上下文中实现最先进性能的同时，将KV缓存内存占用降低多达75%。

    

    大语言模型（LLM）在长上下文推理中面临严重的内存瓶颈，这是由于键值（KV）缓存的大小呈线性增长。现有的KV缓存压缩技术通常依赖简单的启发式方法，忽略了不同注意力头的独特功能角色。我们提出了SGD-KV（摘要引导的KV缓存压缩），这是一个感知注意力头的框架，利用一种新颖的块摘要诊断任务来系统地识别并优先考虑专门负责分层信息聚合的注意力头。在Qwen2.5-7B-1M和Qwen3-32B上针对多样化长上下文基准的实验表明，SGD-KV在高达100万token的上下文中实现了最先进的性能，同时将KV缓存的内存使用量降低了高达75%。我们的研究结果表明，基于注意力头的摘要分数分布策略性地分配KV缓存预算，能够获得更优的效率与准确性权衡。

    arXiv:2609.03235v1 Announce Type: new  Abstract: Large language models (LLMs) face severe memory bottlenecks in long-context inference due to the linearly growing size of key-value (KV) caches. Existing KV cache compression techniques typically rely on simple heuristics, overlooking the distinct functional roles of different attention heads. We present SGD-KV (Summarization-Guided KV Cache Compression), a head-aware framework that leverages a novel chunk-summarization diagnostic task to systematically identify and prioritize attention heads specialized in hierarchical information aggregation. Experiments on Qwen2.5-7B-1M and Qwen3-32B across diverse long-context benchmarks demonstrate that SGD-KV achieves state-of-the-art performance with contexts up to 1M tokens, while reducing KV cache memory usage by up to 75%. Our findings show that strategically allocating the KV cache budget based on the summarization score distribution of attention heads yields a superior efficiency-accuracy tra
    
[^81]: 多步临床大语言模型智能体的反事实公平性审计需要测量每动作的不稳定性底线

    Counterfactual Fairness Audits of Multi-Step Clinical LLM Agents Require a Measured Per-Action Instability Floor

    [https://arxiv.org/abs/2609.03221](https://arxiv.org/abs/2609.03221)

    临床LLM智能体在完全相同输入下本身就存在显著的动作不稳定性（约8.7%），因此反事实公平性审计必须先测量这一“每动作不稳定性底线”，否则任何检测到的人口统计学差异都无法解释。

    

    反事实审计是检查临床智能体是否对人口统计学上不同但临床上相同的患者采取不同行动的标准工具。这类审计报告一个“翻转率”：当仅改变患者描述时，智能体行动发生改变的频率。我们证明这一指标本身是不可解释的。在16个病例情景上将完全相同的条件重复运行十次（相同叙述、相同描述字符串、不改变任何变量），临床智能体的行动在8.7%的结果-情景单元格中发生了改变，且不稳定性在不同行动之间呈现8倍的异质性，从ICU升级决策的0.022到受管制物质谨慎建议的0.179。我们数据中没有任何人口统计学对比能够与这一底线区分开。第二个模型给出了6.7%的合并底线，且对六种行动的不稳定性排序几乎完全一致（Spearman 0.94，精确p=0.017），说明该底线并非单一系统的特有产物。对五次抽样进行多数投票聚合可以消除其中39%的不稳定性……

    arXiv:2609.03221v1 Announce Type: new  Abstract: Counterfactual audits are the standard tool for checking whether a clinical agent treats demographically distinct but clinically identical patients differently. They report a flip rate: how often an action changes when only the patient descriptor changes. We show that this quantity is uninterpretable on its own. Re-running an identical condition ten times over sixteen vignettes (same narrative, same descriptor string, nothing varied) moved a clinical agent's action in 8.7% of outcome-vignette cells, and instability was heterogeneous across actions by a factor of eight, from 0.022 for ICU escalation to 0.179 for controlled-substance caution. No demographic contrast in our data was distinguishable from that floor. A second model gives a pooled floor of 6.7% and ranks the six actions almost identically (Spearman 0.94, exact p=0.017), so the floor is not one system's artefact. Majority-vote aggregation over five draws removes 39% of it and t
    
[^82]: 提示词中的分析师：大语言模型金融分析中的角色、检索与记忆偏差

    The Analyst in the Prompt: Role, Retrieval, and Memory Biases in LLM Financial Analysis

    [https://arxiv.org/abs/2609.03218](https://arxiv.org/abs/2609.03218)

    该研究通过对十二个大语言模型分析3,575份SEC备案文件的实验，揭示了用户上下文导致的金融分析偏差主要源于模型在不同角色下对相同证据的解读差异而非证据检索差异，并提出了两种简单的缓解策略。

    

    大语言模型越来越多地使用记忆、用户资料和角色提示等用户上下文来个性化其响应。这种个性化可能会影响基于证据的判断：相同的证据在不同的用户上下文下可能导致不同的结论。金融领域为研究这一问题提供了一个高风险的场景，因为决策往往依赖于对冗长且复杂文档的解读。我们使用十二个大语言模型对3,575份美国证券交易委员会（SEC）备案文件进行了测试，比较了基于角色的检索、中性检索和记忆框架化的上下文，以区分证据选择的影响与证据解读的影响。我们发现，大多数用户上下文带来的溢出效应源自模型在不同角色下对相同证据的解读方式，而非检索到不同的证据。随后，我们测试了两种简单的缓解策略：将相同的投资者心态以用户资料而非助手角色的形式表达，以及分离证据……

    arXiv:2609.03218v1 Announce Type: new  Abstract: Large Language Models (LLMs) increasingly use user context such as memory, profiles, and role prompts to personalize their responses. This personalization can affect evidence-based judgment: the same evidence may lead to different conclusions under different user contexts. Finance provides a high-stakes setting to study this problem because decisions often depend on interpreting long and complex documents. We test this using 3,575 SEC filings across twelve LLMs. We compare persona-conditioned retrieval, neutral retrieval, and memory-framed context to separate the effect of evidence selection from the effect of interpretation. We find that most user-context spillover comes from how models interpret the same evidence under different roles, rather than from retrieving different evidence. We then test two simple mitigation strategies: expressing the same investor mindset as a user profile instead of an assistant role, and separating evidence
    
[^83]: SWIM：基于熟练度条件化生成的学生写作模拟

    SWIM: Student Writing Simulation via Proficiency-Conditioned Generation

    [https://arxiv.org/abs/2609.03215](https://arxiv.org/abs/2609.03215)

    该论文提出SWIM任务，将学生写作模拟形式化为基于熟练度条件的作文生成，实验发现提示方法对写作熟练度的控制有限，模型虽能调整内容导向特征，却难以重现词汇、语法和组织结构层面的真实学生写作差异。

    

    写作熟练度体现在学生如何发展内容、组织观点、选择词汇以及运用语言。尽管基于大语言模型（LLM）的学生模拟日益受到关注，但LLM能否在长篇写作中重现这种多维度的差异，在很大程度上仍未被探索。在本工作中，我们探索了语言模型能否真实地模拟学生写作，并提出了SWIM——一个将学生写作模拟形式化为熟练度条件化作文生成的任务。我们评估了提示方法、监督微调（SFT）和强化学习（RL）方法在写作模拟中的表现，并使用自动作文评分作为学生画像对齐程度的衡量标准。大量实验表明，提示方法所能提供的熟练度控制能力有限，即使是采用基于评分标准策略的强大专有LLM也是如此。特别值得注意的是，虽然模型能够调整内容导向的写作特征，但它们难以重现词汇、语法和组织结构层面的差异变化。

    arXiv:2609.03215v1 Announce Type: new  Abstract: Writing proficiency manifests in how students develop content, organize ideas, choose words, and use language. Despite growing interest in LLM-based student simulation, whether LLMs can reproduce such multidimensional variation in extended writing remains largely unexplored. In this work, we explore if language models can realistically simulate student writing, and introduce SWIM, a task that formulates Student Writing sIMulation as proficiency-conditioned essay generation. We evaluate prompting, supervised fine-tuning (SFT), and reinforcement learning (RL) methods for writing simulation using automated essay scoring as a measure of profile alignment. Extensive experiments reveal that prompting provides limited proficiency control, even for strong proprietary LLMs with rubric-grounded strategies. In particular, while models can adjust content-oriented traits, they struggle to reproduce the lexical, grammatical, and organizational variati
    
[^84]: 大语言模型从规则中学习上下文的效果优于从示例中学习

    LLMs Learn Better In-Context from Rules than from Examples

    [https://arxiv.org/abs/2609.03213](https://arxiv.org/abs/2609.03213)

    该研究通过五个跨领域任务比较发现，大语言模型从规则（指令遵循）中进行上下文学习比从示例（少样本提示）中学习更可靠，增加示例数量并不能带来显著提升，而指令微调会进一步放大基于规则学习的优势。

    

    大语言模型（LLMs）展现出上下文学习能力，即它们能够从提示上下文中学习新任务，而无需更新模型权重。我们比较了两种主要的上下文学习模式的学习效果：(1) 从规则描述中学习（指令遵循）；(2) 从输入输出演示示例中学习（少样本提示）。通过五个涵盖不同领域（游戏、算术、语言推理）的学习任务，我们比较了针对相同底层任务的两种学习模式（规则 vs. 示例）。我们进一步探索了调节学习效果的模型和任务属性。我们发现，模型从规则中学习通常比仅从示例中学习更加可靠，而在规则基础上添加额外示例或仅仅增加示例数量并不能带来一致且显著的收益。指令微调在保持……的同时放大了基于规则学习的优势（原文摘要在此处截断）。

    arXiv:2609.03213v1 Announce Type: new  Abstract: Large language models (LLMs) exhibit in-context learning capabilities, where they can learn new tasks from prompt contexts without weight updates. We compare the learning efficacies of two prominent modes of in-context learning: (1) learning from descriptions of rules (instruction following); and (2) learning from examples of input-output demonstrations (few-shot prompting). Through five learning tasks that cover diverse domains (games, arithmetic, linguistic inferences), we compare two modes of learning (rules vs. examples) specifying the same underlying task. We furthermore explore model and task properties that modulate the learning efficacies. We find that models generally learn more reliably from rules than from examples alone, and additional examples on top of rules or simply scaling up the number of examples do not lead to consistent and significant gains. Instruction tuning amplifies the benefit of rule-based learning while keepi
    
[^85]: 通过对比课程高效学习缩放

    Learning to Zoom Efficiently with a Contrastive Curriculum

    [https://arxiv.org/abs/2609.03206](https://arxiv.org/abs/2609.03206)

    该论文提出了一种无需标签和热启动微调的InfoNCE式内在奖励，通过难度递增的负样本工具调用课程进行对比学习，高效训练多模态大语言模型使用缩放工具，在多个基准上表现出色甚至超越SFT基线，并引入可扩展的M&C合成数据集直接评估模型的缩放能力。

    

    使用缩放工具是现代视觉智能体的重要基础能力，因为它能够高效处理涉及高分辨率图像的任务。以往的大多数方法都需要一个大规模的热启动监督微调阶段来教会模型使用缩放功能。我们证明了这并非必要，提出了一种新的内在奖励，用于在多模态大语言模型（MLLM）中学习工具使用，无需额外标签或热启动监督微调。我们的InfoNCE风格奖励使用难度逐渐增加的负样本工具调用课程作为对比训练信号。在V*、HRBench和MME-RealWorld上的实证实验表明，我们的方法具有竞争力且更加高效。当作为SFT的直接替代方案使用时，我们甚至超越了所有基线。为了直接衡量模型的缩放能力，我们进一步引入了可扩展的合成数据集Muffin&Chihuahua（M&C）。每张图像由一个网格组成，其中每个单元格显示的要么是一个玛芬蛋糕，要么是一只吉娃娃犬（原文摘要至此截断）。

    arXiv:2609.03206v1 Announce Type: cross  Abstract: Using a zoom-in tool is an important foundational part of modern visual agents, because it allows to efficiently handle tasks involving high-resolution images. Most previous methods need an extensive warm-start supervised fine-tuning phase for teaching models zoom-in. We show that this is not necessary by proposing a new intrinsic reward for learning tool use in MLLMs without the need for additional labels or warm-start SFT. Our InfoNCE-style reward uses a curriculum of increasingly hard negative tool calls as a contrastive training signal. Empirical experiments on $V^*$, HRBench and MME-RealWorld show that our approach is competitive while being more efficient. When used as a drop-in replacement for SFT, we even outperform all baselines. To directly measure the zoom-in ability of models, we further introduce the scalable synthetic Muffin&Chihuahua (M&C) dataset. Each image consists of a grid with every cell either showing a muffin or 
    
[^86]: VoxReason：合成前基于源记录的语音规划的无听者评估

    VoxReason: Listener-Free Evaluation of Source-Grounded Speech Planning Before Synthesis

    [https://arxiv.org/abs/2609.03203](https://arxiv.org/abs/2609.03203)

    VoxReason提出了一种无需听者参与的评估任务，在语音合成之前通过带证据引用的说话计划和确定性验证器，衡量语音表达方式的选择是否真正建立在被引用的源记录之上。

    

    表现力语音系统在任何波形被渲染之前就必须做出一个决定：一句话语将以何种方式被表达。在对话智能体、旁白叙述和角色条件TTS中，这一隐藏的规划步骤决定了情感、音高、能量、语速、停顿、重音和立场，然而下游音频评分很少能揭示这些选择是否由源记录所支持——这是一种在任何波形存在之前就发生的源使用失败。VoxReason将这一合成前的决策转化为可度量的、无需听者参与的任务，用于评估基于源记录的语音规划。在合成之前，VoxReason衡量话语表达方式的选择是否有被引用的源记录作为依据。系统输出带有证据引用的、注明来源的说话计划，随后一个确定性验证器检查引用合法性、槽位一致性、无支持状态、模式有效性以及单线索反事实局部性。在1,440个经过检查的源标签案例上，捷径控制实验表明了为什么仅凭槽位准确率是不安全的：一个简单的键值查找……（原文摘要在此处截断）

    arXiv:2609.03203v1 Announce Type: cross  Abstract: Expressive speech systems make a decision before any waveform is rendered: how an utterance is delivered. In dialogue agents, narration, and role-conditioned TTS, that hidden planning step sets affect, pitch, energy, rate, pause, emphasis, and stance, yet downstream audio scores rarely reveal whether those choices were licensed by the source record, a source-use failure that occurs before any waveform exists. VoxReason makes that pre-synthesis decision measurable as a listener-free task for source-grounded speech planning. Before synthesis, VoxReason measures whether delivery choices are grounded in cited source records. Systems output a source-cited speaking-plan with evidence citations, and a deterministic verifier checks citation legality, slot agreement, unsupported state, schema validity, and one-cue counterfactual locality. On 1,440 checked source-label cases, shortcut controls show why slot accuracy alone is unsafe: a key-lookup
    
[^87]: MemoryLACE：记忆生命周期感知的整合与证据检索

    MemoryLACE: Memory Lifecycle-Aware Consolidation and Evidence Retrieval

    [https://arxiv.org/abs/2609.03201](https://arxiv.org/abs/2609.03201)

    提出了轻量级记忆框架MemoryLACE，通过稀疏的合并、取代和矛盾关系显式建模文本证据的生命周期，重建关系感知的证据单元以呈现当前、历史、支持和冲突证据，从而改进长期LLM智能体的记忆整合与证据检索。

    

    长期运行的LLM智能体必须在多次交互中保留信息，同时区分重复证据、历史状态、更新以及未解决的矛盾。现有的文本记忆系统能够高效检索语义相关的记忆，但这些关系往往是隐式的；而更丰富的结构化方法则通过全局图、层次抽象或反思机制来建模这些关系，但复杂度更高。我们提出了MemoryLACE（MemLACE），这是一个轻量级的记忆框架，通过稀疏的合并、取代和矛盾关系显式建模文本证据的生命周期，同时保留原子化的自然语言记忆及其来源。MemLACE并非独立检索记忆，而是重建具有关系感知的证据单元，为下游推理呈现当前、历史、支持和相互冲突的证据。在BEAM和StructMemEval基准上，使用开源权重和专有LLM……（摘要原文在此处截断）

    arXiv:2609.03201v1 Announce Type: new  Abstract: Long-term LLM agents must preserve information across interactions while distinguishing repeated evidence, historical states, updates, and unresolved contradictions. Existing textual memory systems retrieve semantically relevant memories efficiently but often leave these relationships implicit, whereas richer structured approaches model them through global graphs, hierarchical abstractions, or reflection at greater complexity. We introduce MemoryLACE (MemLACE), a lightweight memory framework that explicitly models the lifecycle of textual evidence through sparse merge, supersession, and contradiction relations while preserving atomic natural-language memories and their provenance. Rather than retrieving memories independently, MemLACE reconstructs relation-aware evidence units that expose current, historical, supporting, and conflicting evidence for downstream reasoning. Across BEAM and StructMemEval, using open-weight and proprietary LL
    
[^88]: Jina-OCR-v1：基于推测解码与密集可验证奖励的高效文档解析

    Jina-OCR-v1: Efficient Document Parsing with Speculative Decoding and Dense Verifiable Rewards

    [https://arxiv.org/abs/2609.03181](https://arxiv.org/abs/2609.03181)

    Jina-OCR-v1 是一个可在低成本 GPU 上高效运行的端到端文档解析模型，通过递归共享草稿块的 FastMTP 推测解码实现无损加速，并结合密集可验证奖励的 GRPO 后训练，在多项基准上取得领先成绩并达到每秒 2.57 页的最高吞吐量。

    

    我们提出了 Jina-OCR-v1，一个专为在低成本 GPU 上运行而构建的端到端文档解析模型。它结合了 DeepSeek-OCR 的压缩视觉编码器和 3B 混合专家解码器（每个 token 激活约 5.7 亿参数），并配备 FastMTP 推测解码头，该解码头在 K=3 个预测步骤中递归共享单个草稿块，而贪婪验证则保证了解码的无损性。后训练阶段结合了指令对齐、针对困难文档的鲁棒性微调，以及在密集可验证奖励（即对公式、表格和结构进行确定性检查并给予部分得分）下的 GRPO 训练。训练数据混合了清洗后的公开语料库和有针对性的合成页面。在默认动态分辨率设置下，Jina-OCR-v1 在 OmniDocBench v1.6 上获得 91.14 分，在 olmOCR-Bench 上获得 83.4 分，并在我们的对比中达到了最高的页面吞吐量，即每秒 2.57 页。在诸如 NVIDIA L4 这样的低成本 GPU 上，FastMTP 使（原文摘要在此处截断）

    arXiv:2609.03181v1 Announce Type: new  Abstract: We present Jina-OCR-v1, an end-to-end document parsing model built to serve on low-budget GPUs. It combines the compressed-vision encoder and the 3B mixture-of-experts decoder of DeepSeek-OCR, which activates about 570M parameters per token, with a FastMTP speculative decoding head that shares a single draft block recursively across K=3 prediction steps. Greedy verification makes decoding lossless. Post-training combines instruction alignment, robustness fine-tuning on difficult documents, and GRPO under dense verifiable rewards: deterministic formula, table, and structural checks that award partial credit. The training data mixes cleaned public corpora with targeted synthetic pages. At the default dynamic-resolution setting, Jina-OCR-v1 scores 91.14 on OmniDocBench v1.6 and 83.4 on olmOCR-Bench, and reaches the highest page throughput in our comparison at 2.57 pages per second. On a low-budget GPU such as the NVIDIA L4, FastMTP doubles 
    
[^89]: 老语言学家的无依之境：LLM-大脑对齐不足以唯一确定神经计算

    No country for old linguists: LLM-brain alignment underdetermines neural computation

    [https://arxiv.org/abs/2609.03160](https://arxiv.org/abs/2609.03160)

    本文指出，LLM与大脑之间的表征对齐虽然能够约束关于神经机制的假说，但并不足以唯一确定神经计算的实际机制，因此不能直接将LLM视为自然语言处理的机制性模型。

    

    Nastase等人（2026）认为，大语言模型（LLM）可能有助于揭示语言处理过程，因为两者都依赖于由统计学习塑造的、分布式且语境敏感的表征。他们对简单的大脑皮层“盒子学”的否定颇具说服力，并有力地论证了LLM-大脑对齐研究的价值。关键问题在于，LLM-大脑对齐能够支持何种类型的推断。本文的论点较为有限：表征对齐原则上可以约束机制性假说，但其本身并不能识别出具体的机制。Nastase等人也承认，编码模型可以捕捉神经活动中所表征的特征，但这并不能确立共享的架构或算法。然而，作者们有时会从对齐跳跃到“共享的计算原理”，并最终将LLM视为自然语言的机制性模型。事实上，他们方法学上的告诫——即对齐并不能确立共（摘要在此处被截断）

    arXiv:2609.03160v1 Announce Type: new  Abstract: Nastase et al. (2026) argue that large language models (LLMs) may illuminate language processing because both rely on distributed, context-sensitive representations shaped by statistical learning. Their rejection of simple cortical "boxology" is persuasive, and they articulate a strong case for the value of LLM-brain alignment research. The key question is what kind of inference LLM-brain alignment licenses. My claim here will be narrow: representational alignment can in principle constrain mechanistic hypotheses, but it does not by itself identify a mechanism. Nastase et al. acknowledge that an encoding model can capture features represented in neural activity without establishing a shared architecture or algorithm. Yet the authors sometime move from alignment to "shared computational principles" and ultimately to LLMs as mechanistic models of natural language. Indeed, their methodological caveat that alignment does not establish a shar
    
[^90]: 谁为被剪枝的Token发声？将视觉Token剪枝视为覆盖率优化问题

    Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization

    [https://arxiv.org/abs/2609.03158](https://arxiv.org/abs/2609.03158)

    提出CoverPruner，一种无需训练的视觉token剪枝方法，将剪枝创新性地建模为表示覆盖最大化问题，确保每个被剪枝的token都有存活的token为其代言，尤其在激进压缩下取得最佳准确率。

    

    视觉token剪枝可以降低视觉语言模型（VLM）的推理成本，但大多数方法只关注应该保留哪些token。这种基于保留token的视角可能会保留冗余的高分token，同时使被丢弃的信息缺乏近似的代表性token。我们提出了CoverPruner，一种无需训练的剪枝方法，它从互补的需求侧提出问题：当一个token被移除后，哪个存活的原始token能够为目标VLM代表它？CoverPruner将剪枝表述为表示覆盖最大化（RCM）问题，以查询加权的需求覆盖完整的投影视觉token集合。该方法通过投影器空间覆盖和轻量级第一层注意力探针来实现RCM。在多个VLM架构和压缩率下，CoverPruner在所有对比方法中取得了最佳平均准确率，且最大的性能提升通常出现在激进压缩的情况下。

    arXiv:2609.03158v1 Announce Type: cross  Abstract: Visual token pruning reduces the inference cost of vision-language models (VLMs), but most methods only ask which tokens to keep. This retained-token view can keep redundant high-scoring tokens while leaving discarded evidence without a close representative. We propose CoverPruner, a training-free pruner that asks the complementary demand-side question: after a token is removed, which surviving original token represents it for the target VLM? CoverPruner formulates pruning as Representational Coverage Maximization (RCM), covering the full projected visual-token set with query-weighted demand. It instantiates RCM with projector-space coverage and a lightweight first-layer attention probe. Across multiple VLM architectures and compression rates, CoverPruner achieves the best average accuracy among all compared methods, with the largest gains usually appearing under aggressive compression.
    
[^91]: 路由是不够的：诊断 MoE+LoRA 微调中适配器内部子空间的争用问题

    Routing Is Not Enough: Diagnosing Intra-Adapter Subspace Contention in MoE+LoRA Fine-Tuning

    [https://arxiv.org/abs/2609.03150](https://arxiv.org/abs/2609.03150)

    该研究发现在 MoE+LoRA 多领域微调中，即使专家路由近乎完全分离，负迁移仍由近乎正交的领域梯度在同一低秩适配器子空间内竞争所致，并提出 SpawnLoRA，通过在检测到适配器争用时于 MoE 专家内部动态添加门控子适配器（保持路由器固定）来化解这一问题。

    

    多领域微调通常将 MoE 路由与 LoRA 相结合，并假设 token 级别的路由能够分离各领域特有的更新。我们使用 Python 代码与生物医学文本和数学推理配对的数据，在 MoE+LoRA 中检验了这一假设。尽管这些领域表现出近乎不相交的专家路由，但加入生物医学数据后代码的困惑度显著上升，这表明仅靠路由分离可能无法阻止负迁移。为了定位这一失败，我们引入了 Jaccard 路由重叠度和适配器梯度余弦相似度两个诊断指标，分别用于衡量专家共享程度和更新兼容性。这些诊断结果表明，干扰主要源自几乎正交的领域梯度在同一低秩适配器子空间内的竞争。我们通过 SpawnLoRA 来解决这一问题：当检测到适配器层面的争用时，SpawnLoRA 会在 MoE 专家内部动态添加带门控的子适配器，同时保持路由器固定不变。

    arXiv:2609.03150v1 Announce Type: cross  Abstract: Multi-domain fine-tuning often combines MoE routing with LoRA, assuming that token-level routing separates domain-specific updates. We test this assumption in MoE+LoRA using Python code paired with biomedical text and mathematical reasoning. Although these domains show near-disjoint expert routing, adding biomedical data substantially increases code perplexity, indicating that routing separation alone may not prevent negative transfer. To localize the failure, we introduce Jaccard routing overlap and adapter-gradient cosine similarity, which measure expert sharing and update compatibility, respectively. These diagnostics indicate that interference arises mostly from nearly orthogonal domain gradients competing within the same low-rank adapter subspace. We address this issue with SpawnLoRA, which dynamically adds gated sub-adapters inside MoE experts when adapter-level contention is detected, while keeping the router fixed. We evaluate 
    
[^92]: 大语言模型在解决上下文知识冲突中的应用

    Large Language Models in Resolving Contextual Knowledge Conflicts

    [https://arxiv.org/abs/2609.03148](https://arxiv.org/abs/2609.03148)

    该论文提出了上下文知识内部冲突的六类型分类法并构建了包含5,781个样本的ContextConflict数据集，实验发现当前大语言模型在解决此类冲突上仍有不足，并通过机制可解释性分析揭示了模型对冲突的潜在感知及其背后的表征几何结构。

    

    以往的大多数研究都聚焦于大语言模型（LLM）内部参数化知识与外部提供的上下文之间的冲突。与之不同，我们研究了大语言模型如何处理上下文知识本身内部产生的冲突。我们引入了一个包含六种上下文冲突类型的分类法（事实型、推理型、时间型、粒度型、视角型和歧义型），并为此场景贡献了一个综合性数据集ContextConflict。该数据集包含5,781个样本，同时涵盖推理和摘要两类任务，既包含显性矛盾，也包含需要多步推理才能发现的隐性冲突。在九个大语言模型上进行的实验表明，当前模型在解决上下文知识冲突方面仍然存在不足。我们进一步提供了关于大语言模型如何处理此类冲突的机制可解释性见解，揭示了模型对冲突的潜在感知以及冲突处理背后的表征几何结构。

    arXiv:2609.03148v1 Announce Type: new  Abstract: Most prior works focused on conflicts between an LLM's internal parametric knowledge and externally provided context. In contrast, we investigate how LLMs handle conflicts that arise within contextual knowledge itself. We introduce a taxonomy of six types of contextual conflicts (factual, inferential, temporal, granularity, perspective, and ambiguity) and contribute a comprehensive dataset ContextConflict for this setting. The dataset contains 5,781 samples, covers both reasoning and summarization tasks, and includes both explicit contradictions and implicit conflicts that require multi-step reasoning. Experiments on nine LLMs show that current models still fall short in resolving contextual knowledge conflicts. We further provide mechanistic interpretability insights into how LLMs process such conflicts, revealing their latent awareness of conflicts and the representational geometry underlying conflict processing. In addition, our analy
    
[^93]: SHELF：一个用于多任务书目基准测试的合成测试框架

    SHELF: A Synthetic Harness for Multi-Task Bibliographic Benchmarking

    [https://arxiv.org/abs/2609.03047](https://arxiv.org/abs/2609.03047)

    SHELF是一个基于美国国会图书馆词表生成6万余篇合成文档的Python系统，为图书馆和档案馆的书目工作提供了涵盖分类、聚类、检索等多任务的系统性基准测试框架。

    

    图书馆和档案馆在人员和计算预算有限的情况下管理着大量馆藏，然而现有的常见基准测试并未系统地检验其书目工作。他们需要了解哪些方法适用于自己的任务，以及运行这些方法需要什么条件。SHELF（用于评估LLM适应性的合成测试框架，Synthetic Harness for Evaluating LLM Fitness）填补了这一空白。它是一个Python系统，能够将带标签的分类法、编写规范和生成预算转化为受控的基准数据和评估任务。首个发布版本包含62,899篇基于美国国会图书馆词表、由模型生成的文档，涵盖分类、聚类、检索、成对分类和指令检索等任务。我们比较了TF、TF-IDF、BM25、流行的编码器模型，以及仅在主题分类任务上测试的零样本解码器；每种方法仅出现在支持它的任务上。主题分类的准确率达到0.8887，而体裁-形式分类仅达到0.2605……

    arXiv:2609.03047v1 Announce Type: cross  Abstract: Libraries and archives manage large collections with limited staff and computing budgets, yet common benchmarks do not systematically test their bibliographic work. They need to know which methods work for their tasks and what those methods require to run. SHELF, the Synthetic Harness for Evaluating LLM Fitness, addresses this gap. It is a Python system that turns labelled taxonomies, writing specifications, and a generation budget into controlled benchmark data and evaluation tasks. This first release contains 62,899 model-written documents based on Library of Congress vocabularies, with tasks for classification, clustering, retrieval, pair classification, and instruction retrieval. We compare TF, TF-IDF, BM25, popular encoders, and, on subject classification only, zero-shot decoders; each method appears only on tasks that support it. Subject classification reaches 0.8887, while genre-form classification reaches only 0.2605, and sever
    
[^94]: 基于上下文集成的共形语言任务统一框架

    Unifying Conformal Language Tasks with In-Context Ensembles

    [https://arxiv.org/abs/2609.03005](https://arxiv.org/abs/2609.03005)

    提出共形相关性框架，通过上下文学习示例筛选与集成自动构建评分函数，以最少的人工干预统一实现了多种NLP任务中覆盖率与简洁性的双重保证。

    

    许多自然语言处理任务，例如摘要生成和抽取式问答，都可以归结为在两个约束条件下从文档中检索相关内容：覆盖率（保留足够的相关信息以实现某个目标）和简洁性（尽可能去除无关信息）。共形预测方法已被用于保证覆盖率，但需要通过设计评分函数来优化简洁性。最先进的评分函数使用手工设计的LLM提示词来让模型评估内容的重要性，但手动提示工程既费力又依赖于具体任务。我们提出了共形相关性框架，该框架利用上下文学习示例筛选与集成来构建评分函数，在保持覆盖率的同时提高简洁性，且只需最少的人工干预。我们演示了该框架在七个NLP任务上的应用，并从理论上研究了多样性的影响。

    arXiv:2609.03005v1 Announce Type: new  Abstract: Many NLP tasks, such as summarization and extractive question answering, reduce to retrieving relevant content from documents under two constraints: coverage, retaining enough pertinent information to achieve some goal, and conciseness, removing as much irrelevant information as possible. Conformal prediction methods have been used to guarantee coverage, and must be optimized for conciseness through design of a score function. State-of-the-art scoring functions use hand-engineered LLM prompts asking the model to rate the importance of content, but manual prompt engineering is labor-intensive and task-specific. We introduce the Conformal Relevance framework which uses in-context learning example curation and ensembling to create a score function which maintains coverage while improving conciseness with minimal manual input. We demonstrate this framework's application on seven NLP tasks, and also theoretically study the impact of diversity
    
[^95]: 蒸馏之前先验证：面向在线策略蒸馏的提示级教师门控

    Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation

    [https://arxiv.org/abs/2609.02998](https://arxiv.org/abs/2609.02998)

    该论文提出教师门控在线策略蒸馏（TGOPD），通过经验证器评分的教师探测在提示级别先验证教师模型的可靠性，将可靠提示路由到密集OPD监督、不可靠提示路由到基于验证器的GRPO，从而避免“自信但错误”的教师模型诱导误导性更新。

    

    在线策略蒸馏（OPD）通过在学生模型自身的生成结果上提供来自冻结教师模型的密集token级监督来加速后训练过程。原始的OPD在所有提示上均匀地应用这种监督，而不检查教师模型对每个提示是否可靠。由于反向KL散度具有模式寻求特性，一个自信但错误的教师模型可能导致强烈却具有误导性的更新。分布性代理指标（如熵或教师-学生似然一致性）只能衡量不确定性或一致性，但无法直接验证结果的正确性。我们提出了教师门控在线策略蒸馏（TGOPD），其核心原则是在接受密集监督之前，应在提示级别验证教师模型的可靠性。TGOPD通过一小组经验证器评分的教师探测样本估计教师可靠性，并将每个提示专门路由到密集OPD（当可靠性检查通过时）或基于验证器的GRPO（当检查不通过时）。在4B和3...（摘要内容不完整）

    arXiv:2609.02998v1 Announce Type: cross  Abstract: On-policy distillation (OPD) accelerates post-training by providing dense token-level supervision from a frozen teacher on the student's own rollouts. Vanilla OPD applies this supervision uniformly across prompts, without checking whether the teacher is reliable for each prompt. Because reverse KL is mode-seeking, a confidently wrong teacher can induce a strong yet misleading update. Distributional proxies, such as entropy or teacher-student likelihood agreement, measure uncertainty or agreement but do not directly verify outcome correctness. We introduce Teacher-Gated On-Policy Distillation (TGOPD), built on the principle that teacher reliability should be verified at the prompt level before dense supervision is admitted. TGOPD estimates reliability from a small set of verifier-scored teacher probes and routes each prompt exclusively to dense OPD when the reliability check passes or to verifier-grounded GRPO otherwise. Across 4B and 3
    
[^96]: 无知的几何学：大语言模型知道何时调节贝叶斯先验

    The Geometry of Ignorance: LLMs Know When to Temper Bayesian Priors

    [https://arxiv.org/abs/2609.02959](https://arxiv.org/abs/2609.02959)

    研究发现大语言模型的反嵌入矩阵中存在一个编码训练语料词元分布的“无知方向”，模型通过逐词元调节该先验的强度，实现了随上下文信息增加而逐步减弱先验影响的温度调节贝叶斯更新。

    

    当语言模型缺乏线索时，它会预测什么？答案隐藏在其反嵌入矩阵的几何结构中：反嵌入矩阵的一个单一方向编码了训练语料库的词元分布，它充当了模型在不确定时回退依赖的贝叶斯先验。我们将这一结构称为“无知方向”，它出现在所有四个被检验的模型家族中（Llama、Qwen、Gemma 和 Pythia），参数规模从 0.4B 到 405B 不等。将最终预测状态投影到该方向上可得到逐词元的先验载荷因子 λ，经验表明该因子随着上下文信息量的增加而稳步下降。从形式上看，同样的投影将预测状态分解为两个正交向量，它们恰好对应于温度调节贝叶斯更新的两个因子：被提升到指数 λ 的词元先验以及由上下文驱动的似然。

    arXiv:2609.02959v1 Announce Type: cross  Abstract: What does a language model predict when it has few clues? The answer lurks in its unembedding geometry: a single direction of the unembedding matrix encodes the unigram distribution of the training corpus, which serves as the Bayesian prior the model falls back on when uncertain. This structure --- which we term the \emph{direction of ignorance} --- appears in all four model families examined (\texttt{Llama}, \texttt{Qwen}, \texttt{Gemma}, and \texttt{Pythia}), ranging from 0.4B to 405B parameters. Projecting the final prediction state onto this direction yields a per-token \emph{prior loading factor} $\lambda$, which, empirically, declines steadily as the context becomes more informative. Formally, the same projection decomposes the prediction state into two orthogonal vectors that correspond exactly to the two factors of a tempered Bayesian update: a unigram prior raised to the exponent $\lambda$ and a context-driven likelihood. This
    
[^97]: LexIssue：中国民事诉讼中法律争议焦点识别的基准测试

    LexIssue: Benchmarking Legal Issue Identification in Chinese Civil Litigation

    [https://arxiv.org/abs/2609.02954](https://arxiv.org/abs/2609.02954)

    该论文构建了包含430个真实中国民事诉讼案例和1303个专家标注争议焦点的LexIssue基准数据集，并提出以争议焦点为中心的法律知识库，将争议焦点识别形式化为生成与分类两个互补任务以支持检索增强推理。

    

    识别诉讼当事人之间的争议焦点是现实诉讼中的一个关键环节。然而，法律争议焦点在法律人工智能研究中仍相对缺乏探索。在本工作中，我们研究了诉讼中法律争议焦点识别的计算建模。我们引入了一个具有法律依据的层次化模式，通过自由形式的争议描述和结构化的法律类别来表示法律争议焦点，并将法律争议焦点识别形式化为两个互补的任务：法律争议焦点生成和法律争议焦点分类。基于这一形式化，我们构建了LexIssue基准数据集，其中包含430个真实的中国民事诉讼案例和1,303个由专家标注的争议法律焦点。我们进一步开发了一个以争议焦点为中心的法律知识库，涵盖27个案由和441个候选法律争议焦点条目，以支持检索增强推理。在多种模型上的实验结果表明……

    arXiv:2609.02954v1 Announce Type: new  Abstract: Identifying the issues disputed between litigating parties is a crucial component of real-world litigation. However, legal issues remain comparatively underexplored in legal AI research. In this work, we study the computational modelling of legal issue identification in litigation. We introduce a legally grounded hierarchical schema that represents legal issues through both free-form issue descriptions and structured legal categories, and formulate legal issue identification as two complementary tasks: legal issue generation and legal issue classification. Based on this formulation, we construct LexIssue, a benchmark containing 430 real-world Chinese civil litigation cases and 1,303 expert-annotated disputed legal issues. We further develop an issue-centric legal knowledge base spanning 27 causes of action and 441 candidate legal issue entries to support retrieval-augmented reasoning. Experimental results across a diverse set of models s
    
[^98]: 面向认知诊断的隐私保护异构多LLM联邦推理

    Privacy-Preserving Heterogeneous Multi-LLM Federated Inference for Cognitive Diagnosis

    [https://arxiv.org/abs/2609.02947](https://arxiv.org/abs/2609.02947)

    该论文提出一种隐私保护的异构多LLM联邦推理框架，通过本地拉普拉斯噪声差分隐私和基于残差的聚合机制，使多个商用LLM API无需访问原始学生数据即可协作实现准确的认知诊断。

    

    AI驱动的教育系统在平衡隐私保护与准确认知诊断方面仍面临重大挑战。为克服这一问题，我们提出了一种联邦推理框架，使多个商用LLM API能够在无需访问原始学生数据或专有模型内部结构的前提下进行协作。该框架基于异构多LLM架构，利用多个联邦实体（如LLaMA-3.3-70B、GPT-4o-mini和Claude-3-Haiku）。这些实体生成的预测通过epsilon本地差分隐私进行融合，即在聚合之前对每个实体的预测输出本地添加拉普拉斯噪声，同时采用基于残差的聚合方式来缓解模型间的异质性。我们的方法建立在“诚实但好奇”的信任范式之上，即假设API提供者不会滥用所提交的查询，并且我们的差分隐私机制保护已发布的诊断结果免受外部……（原文摘要在此处截断）

    arXiv:2609.02947v1 Announce Type: cross  Abstract: Significant challenges remain in AI-driven educational systems in balancing privacy preservation with accurate cognitive diagnosis. To overcome this, we propose a federated inference framework in which several commercial LLM APIs collaborate without requiring access to raw student data or proprietary model internals. Using multiple federated entities, such as LLaMA-3.3-70B, GPT-4o-mini, and Claude-3-Haiku, our framework builds upon a heterogeneous multi-LLM architecture. The predictions generated by these entities are combined with epsilon-local differential privacy by adding Laplace noise locally to each entity's prediction output before aggregation, while residual-based aggregation mitigates model heterogeneity. Our approach is predicated on an honest-but-curious trust paradigm in which API providers are presumed not to abuse submitted queries, and our differential privacy mechanism shields the published diagnostic results from exter
    
[^99]: 评判LLM作为评判者：基于LLM的自动文本生成评估中令人担忧的评估准则伪影问题

    Judging LLM-as-a-Judge: Concerning Rubric Artifacts in LLM-based Automated Text Generation Evaluation

    [https://arxiv.org/abs/2609.02942](https://arxiv.org/abs/2609.02942)

    研究发现LLM评估中的评估准则文本本身编码了可预测的评估信号，且评判者在候选回答或准则被反转时往往无法可靠更新判断，这引发了对基于准则的LLM自动评估可靠性的严重质疑。

    

    基于大语言模型（LLM）作为评判者的评估流程被越来越多地用于评估AI生成的文本，其前提假设是评判结果源于对候选回答依据评估准则的推理。我们证明这一假设需要进一步审视。仅在评估准则文本上训练的分类器，在完全无法接触任何被评估回答的情况下，就能对评判结果取得可观的预测性能。这表明评估准则的表述中编码了可提取的评估信号，使得评分可以在不依赖模型输出的情况下被部分预测。最后，反事实扰动实验揭示，当候选回答或评估准则被反转时，评判者往往无法可靠地更新其判断。我们的发现引发了对基于评估准则的LLM评估可靠性的担忧，并强调需要对通过LLM进行自动化评估的方法论展开进一步研究。

    arXiv:2609.02942v1 Announce Type: cross  Abstract: LLM-as-a-Judge pipelines are increasingly used to evaluate AI-generated text, based on the assumption that judgments arise from reasoning over candidate responses with respect to a rubric. We show that this assumption warrants further scrutiny. Classifiers trained only on rubric text, without access to any evaluated response, achieve nontrivial predictive performance on judge outputs. This suggests that rubric formulations encode recoverable evaluative signals, allowing scores to be partially anticipated independently of model outputs. Finally, counterfactual perturbations reveal that judges often fail to reliably update their decisions when either the candidate response or the rubric criterion is reversed. Our findings raise concerns about the reliability of rubric-based LLM evaluation and highlight the need for further methodological study of automated evaluation via LLMs.
    
[^100]: SISER：基于熵对抗训练的说话人不变语音情感识别

    SISER: Speaker-Invariant Speech Emotion Recognition with Entropy-Based Adversarial Training

    [https://arxiv.org/abs/2609.02941](https://arxiv.org/abs/2609.02941)

    提出SISER框架，将wav2vec 2.0特征编码器与ECAPA-TDNN说话人判别器结合进行基于熵的对抗训练，同时解决标注数据稀缺和说话人差异两大难题，在IEMOCAP上达到60.63%的无加权准确率。

    

    语音情感识别（SER）面临两个根本性挑战：标注数据稀缺和说话人之间的个体差异，这两者都阻碍了情感识别系统的泛化能力。虽然先前的对抗方法能够解决说话人差异问题，但它们未能充分利用强大的预训练表示。我们提出了SISER（说话人不变语音情感识别），将wav2vec 2.0作为特征编码器、ECAPA-TDNN作为说话人判别器，整合到基于熵的对抗训练方案中。wav2vec 2.0提供了丰富的自监督表示，减轻了对大规模标注数据集的依赖，而ECAPA-TDNN通过比浅层分类器更强的对抗信号来抑制说话人身份信息。在IEMOCAP数据集上的评估显示，SISER达到了60.63%的无加权准确率（UA），优于基线方法（51.15%）和无说话人抑制的wav2vec 2.0（56.46%），消融实验强调了说话人判别器选择的重要性。

    arXiv:2609.02941v1 Announce Type: cross  Abstract: Speech emotion recognition (SER) faces two fundamental challenges: scarcity of labeled data and inter-speaker variability, both of which hinder generalization of emotion recognition systems. While prior adversarial approaches address speaker variability, they fall short in leveraging powerful pre-trained representations. We propose SISER (Speaker-Invariant Speech Emotion Recognition), integrating wav2vec 2.0 as a feature encoder and ECAPA-TDNN as a speaker discriminator within an entropy-based adversarial training scheme. wav2vec 2.0 provides rich self-supervised representations that alleviate dependency on large labeled datasets, while ECAPA-TDNN enables suppression of speaker identity via a stronger adversarial signal than shallow classifiers. Evaluated on IEMOCAP, SISER achieves a UA of 60.63%, outperforming the baseline (51.15%) and wav2vec 2.0 without speaker suppression (56.46%), with ablation emphasizing that the choice of speak
    
[^101]: 倾听潜在表示：通过隐藏状态交互实现大型音频语言模型中的自纠正语音识别

    Listen to the Latents: Self-Correcting Speech Recognition in Large Audio Language Models Through Hidden-State Interactions

    [https://arxiv.org/abs/2609.02940](https://arxiv.org/abs/2609.02940)

    该论文提出Hybrid Search纠错策略，利用基于LLM的ASR隐藏状态与基础LLM隐藏状态之间的交互特征来识别语义依赖程度高的词元并进行选择性精炼，从而显著提升LoRA适配的热启动初始化语音识别模型性能，超越重打分等全局纠错方法。

    

    近期的自动语音识别（ASR）系统越来越多地集成大型语言模型（LLM）以利用其语义知识，方式包括通过logit融合进行外部集成，或通过热启动初始化进行内部集成。然而，如何有效结合这两种策略仍未得到充分探索。在本工作中，我们通过利用模型自身预适配阶段的基础LLM来改进基于热启动初始化LLM的ASR模型，重点关注保留基础LLM的LoRA适配设置。为实现这一目标，我们提出了Hybrid Search，一种由两点观察启发的针对性纠错策略。第一，刻画基于LLM的ASR隐藏状态与基础LLM隐藏状态之间关系的交互特征，能够为词元的语义依赖程度提供有信息量的信号。第二，选择性地对语义依赖程度高的目标词元进行精炼，其ASR性能提升远超朴素的全局LLM纠错方法，包括重打分（rescoring）等方法。

    arXiv:2609.02940v1 Announce Type: cross  Abstract: Recent automatic speech recognition (ASR) systems increasingly integrate large language models (LLMs) to leverage their semantic knowledge, either externally through logit fusion or internally through warm initialization. However, how to effectively combine these two strategies remains underexplored. In this work, we refine warm-initialized LLM-based ASR models by leveraging their own pre-adaptation base LLMs, focusing on LoRA-adapted settings where the base LLM is preserved. To achieve this, we propose Hybrid Search, a targeted correction strategy motivated by two observations. First, interaction features that characterize the relationship between LLM-based ASR hidden states and base-LLM hidden states provide informative signals about a token's degree of semantic dependence. Second, selectively refining targeted tokens with high semantic dependence improves ASR performance far beyond naive global LLM-correction methods including resco
    
[^102]: 复线性映射的谱相位可容许性证书

    A Spectral Phase Admissibility Certificate for Complex Linear Maps

    [https://arxiv.org/abs/2609.02911](https://arxiv.org/abs/2609.02911)

    该论文将量子引力中的Kontsevich-Segal-Witten判据引入机器学习，提出三种可微证书来约束复线性映射的谱集体相位，并通过Schur参数化实现可微执行，同时指出该约束因损害特征向量条件数而不适用于深度线性传播。

    

    该论文将量子引力领域中的Kontsevich-Segal-Witten判据引入机器学习，用于评估复线性映射。传统技术分析幅值或正定性，而该方法专门限制谱的集体相位。研究人员构建了三种不同的可微证书，包括行列式扇区、子集乘积包络和完整判据。子集包络防止所有外幂特征值触及负实轴，这一约束精确匹配指数子式枚举的接受或拒绝选择，同时大幅降低处理开销。团队通过Schur参数化提供了可微执行方案。论文还指明了该系统适用的关键边界：该约束无法平衡深度线性传播，因为限制相位预算会损害特征向量的条件数。

    arXiv:2609.02911v1 Announce Type: cross  Abstract: The paper imports the Kontsevich Segal Witten criterion from quantum gravity into machine learning to evaluate complex linear maps Standard techniques analyze magnitude or positive definiteness whereas this method exclusively limits the collective phase of a spectrum The researchers create three distinct differentiable certificates comprising a determinant sector a subset product envelope and the full criterion The subset envelope prevents all exterior power eigenvalues from touching the negative real axis This constraint precisely matches the accept or reject choices of an exponential minor enumeration while reducing processing expenses drastically The team provides a differentiable enforcement application via a Schur parameterization The document also identifies crucial boundaries regarding where this system works The constraint cannot balance deep linear propagation since restricting the phase budget damages eigenvector conditioning
    
[^103]: RL-ADA：面向对抗性鲁棒企业对话代理的世界反馈框架

    RL-ADA: A World-Feedback Framework for Adversarially Robust Enterprise Dialogue Agents

    [https://arxiv.org/abs/2609.02902](https://arxiv.org/abs/2609.02902)

    该论文提出RL-ADA框架，用基于可衡量交互结果的世界反馈取代人工标注，让30亿参数的客户支持代理与70亿参数的对抗性客户代理在自动裁判引导下共同进化，从而训练出对抗鲁棒的企业对话代理。

    

    在企业客户支持中部署任务导向型对话代理面临一个持续的标注瓶颈：稳健的训练需要大规模的标注交互数据，然而企业对话日志涉及隐私敏感且标注成本高昂，同时用户行为的演变速度远快于标注流程的跟进速度。我们提出了 RL-ADA（基于对抗性对话代理的强化学习），这是一个共同进化的训练框架，通过用“世界反馈”取代人工标注来消除这一瓶颈：世界反馈是直接从可衡量的交互结果中得出的、基于后果的奖励信号。一个客户支持代理（DA，30亿参数）和一个对抗性客户代理（CA，70亿参数）在固定自动裁判的引导下，于对抗竞技场中共同进化：当 DA 正确处理多轮客户对话并成功解决问题时获得奖励，而 CA 则因产生逼真的、意图一致的（对抗性输入）而获得奖励。

    arXiv:2609.02902v1 Announce Type: new  Abstract: Deploying task-oriented dialogue agents in enterprise customer support faces a persistent annotation bottleneck: robust training requires labelled interaction data at scale, yet enterprise conversational logs are privacy-sensitive and expensive to annotate, while user behaviour evolves faster than labelling pipelines can keep pace. We present RL-ADA (Reinforcement Learning with Adversarial Dialogue Agents), a co-evolutionary training framework that eliminates this bottleneck by replacing human labels with \emph{world feedback}: consequence-based reward signals derived directly from measurable interaction outcomes. A Customer Support Agent (DA, 3B parameters) and an Adversarial Customer Agent (CA, 7B parameters) co-evolve in an adversarial arena guided by a fixed automated judge: the DA is rewarded for correctly handling multi-turn customer conversations to successful resolution, while the CA is rewarded for producing realistic, intent-co
    
[^104]: 双形式语音识别：面向中文语音识别的语义感知逆文本正则化

    Dual-Form ASR: Semantics-Aware Inverse Text Normalization for Chinese Speech Recognition

    [https://arxiv.org/abs/2609.02901](https://arxiv.org/abs/2609.02901)

    提出双形式语音识别框架DF-ASR，利用LLM驱动的成对口语-书面形式监督数据构建及ITN-MWER序列级训练目标，将口语形式语音识别能力扩展到语义感知的书面形式逆文本正则化，统一了两种转录形式的建模。

    

    现代自动语音识别（ASR）场景既需要用于忠实转录的口语形式文本，也需要经过逆文本正则化（ITN）处理、可读性强的书面形式文本。然而，这两种形式通常由级联模块生成，即口语形式的ASR输出由独立的ITN组件进行改写，这使得书面形式的ASR-ITN容易受到识别错误的影响，并将正则化过程与声学上下文建模解耦，对于语义高度依赖的数字表达式而言问题尤为突出。本文提出双形式语音识别，这是一个通过成对的口语形式与书面形式监督，将口语形式ASR能力扩展至语义感知的书面形式ITN的框架，同时保留了在提示层面灵活选择输出转录形式的能力。双形式监督数据通过大型语言模型（LLM）驱动的“生成-评判”工作流构建，并通过ITN-MWER这一序列级目标函数进一步增强训练。

    arXiv:2609.02901v1 Announce Type: new  Abstract: Modern automatic speech recognition (ASR) scenarios require both spoken-form transcripts for faithful transcription and readable written-form transcripts with inverse text normalization (ITN). However, these forms are typically produced by cascaded modules, where a spoken-form ASR output is rewritten by a separate ITN component, making written-form ASR-ITN vulnerable to recognition errors and decoupling normalization from acoustic-contextual modeling, especially for semantically dependent numeric expressions. In this paper, we propose Dual-Form ASR (DF-ASR), a framework that extends spoken-form ASR capability to semantics-aware written-form ITN through paired spoken-form and written-form supervision while retaining prompt-level selection between transcript forms. The dual-form supervision is constructed via a large language model (LLM)-driven generate-and-judge workflow, and training is further enhanced by ITN-MWER, a sequence-level obje
    
[^105]: DisclosureBeta：基于大语言模型读取风险披露的状态条件贝塔测量信道理论

    DisclosureBeta: A Measurement-Channel Theory for Regime-Conditioned Betas from LLM-Read Risk Disclosures

    [https://arxiv.org/abs/2609.02900](https://arxiv.org/abs/2609.02900)

    该论文将大语言模型建模为读取企业风险披露的含噪测量信道，在分段平稳的五因子框架下首次为状态条件贝塔建立了完整的可识别性、一致性与匹配误差下界理论，从而为价格历史过短的公司（如IPO企业）提供带误差预算的可靠贝塔估计。

    

    本文要解决的问题是：当一家公司的价格历史过短而不可信时——例如提交S-1上市申请的公司、新近上市的公司，或刚刚经历市场状态切换的公司——交易台所需的贝塔值从何而来。现有最先进的方法退化为基于可比公司的同伴贝塔，且没有误差预算；而近期基于文本的竞争方法Breitung（2025）虽然报告了较强的IPO实证准确率，但缺乏识别理论、没有误差预算，也没有下界。我们填补了这一空白。我们将大语言模型建模为对企业潜在风险特征的一个含噪测量信道，并将其信道噪声纳入资产定价的误差预算。在分段平稳的Fama-French五因子模型中，因子载荷是潜在风险特征与推断出的市场状态的函数。我们在对信道、检测器以及状态内采样的显式假设下，证明了状态条件载荷函数的可识别性与一致性，并给出了匹配的下界，表明披露……（原文摘要至此截断）

    arXiv:2609.02900v1 Announce Type: cross  Abstract: The problem is the beta a desk needs when a firm's price history is too short to trust: an S-1 filer, a recent listing, or a name just past a regime break. The state of the art collapses to a comparable-firm peer beta with no error budget, and the recent text-based competitor Breitung (2025) reports strong empirical IPO accuracy but no identification theory, no error budget, and no lower bound. We fill that gap. We model a large language model as a noisy measurement channel on a firm's latent risk characteristics and write its channel noise into the asset-pricing error budget. In a piecewise-stationary Fama-French five-factor model the loadings are a function of latent risk characteristics and an inferred regime. We prove identification and consistency of the regime-conditional loading function under explicit assumptions on the channel, the detector, and within-regime sampling, and give a matching lower bound showing that the disclosur
    
[^106]: 污染会抬高分数，但很少改变大语言模型排行榜的排名

    Contamination Inflates Scores but Rarely Reorders Large Language Model Leaderboards

    [https://arxiv.org/abs/2609.02899](https://arxiv.org/abs/2609.02899)

    该论文提出一种基于题目内改写对比的污染度量方法，发现基准污染虽然会抬高模型的绝对分数，但很少改变大语言模型排行榜的相对排名。

    

    基准污染，即测试题目泄漏到训练数据中，被广泛描述为威胁大语言模型（LLM）排行榜可靠性的因素。我们认为这一担忧混淆了两个不同的问题：污染是否会抬高绝对分数，以及污染是否会改变模型之间的排名。我们将污染重新界定为对锚定题目不变性的违反，并通过原始题目与语义等价的改写题目之间的差异表现来度量污染——这种题目内的对比方法固定了所测量的能力，从而将记忆效应与真实能力分离开来。基于47个公开发布的模型和74个以已知污染剂量进行微调的模型的逐实例响应，涵盖四个基准测试（ARC、GSM8K、HellaSwag、MMLU），我们首先针对真实情况对该度量方法进行了校准：它能够随注入污染的剂量成比例地检测出污染（对于测试集泄漏，校正后的效应为+0.187个准确率百分点……

    arXiv:2609.02899v1 Announce Type: new  Abstract: Benchmark contamination, the leakage of test items into training data, is widely described as a threat to the reliability of large language model (LLM) leaderboards. We argue that this concern conflates two distinct questions: whether contamination inflates absolute scores, and whether it reorders the ranking of models. We recast contamination as a violation of anchor-item invariance and measure it through the differential functioning of original versus semantically equivalent paraphrased items, a within-item contrast that holds the measured skill fixed and isolates memorization from capability. Using per-instance responses from 47 publicly released models and 74 models finetuned with a known dose of contamination, across four benchmarks (ARC, GSM8K, HellaSwag, MMLU), we first calibrate the measure against ground truth: it recovers injected contamination dose-responsively (a corrected effect of +0.187 accuracy points for test-set leakage
    
[^107]: 蒸馏快速嵌入迁移（DRET）：通过基于优先级的嵌入迁移实现参数高效的生物医学领域自适应

    Distilled Rapid Embedding Transfer (DRET): Parameter-Efficient Biomedical Domain Adaptation via Priority-Based Embedding Transfer

    [https://arxiv.org/abs/2609.02898](https://arxiv.org/abs/2609.02898)

    DRET提出了一种无需在专业语料上重新训练的参数高效领域自适应方法，通过基于优先级的嵌入迁移机制，将BioBERT等大型生物医学专用模型的领域知识注入DistilBERT等轻量级通用模型中。

    

    BioBERT和ClinicalBERT等大型领域专用语言模型在生物医学自然语言处理任务上表现出色，但其计算需求使其在许多实际部署场景中难以应用。而DistilBERT等通用参数高效模型虽然轻量，却缺乏PICO（人群、干预、对照、结果）分类等专业任务所需的领域知识。我们提出了蒸馏快速嵌入迁移（DRET），这是一种知识迁移范式，能够将大型专用模型中的生物医学领域知识注入到较小的通用模型中，而无需在原始专业语料库上重新训练。DRET发展为一个迭代式的策略家族：统一的分词器合并策略（DRET 1.x）、混合嵌入平均（DRET 2.0），以及基于优先级的嵌入迁移机制（DRET 3.x），该机制能够从最权威的源模型中分层选择嵌入……

    arXiv:2609.02898v1 Announce Type: new  Abstract: Large domain-specific language models such as BioBERT and ClinicalBERT achieve strong performance on biomedical NLP tasks, but their computational demands make them impractical for many real-world deployments. General-purpose, parameter-efficient models such as DistilBERT are lightweight yet lack the domain knowledge required for specialized tasks such as PICO (Population, Intervention, Comparison, Outcome) classification. We introduce Distilled Rapid Embedding Transfer (DRET), a knowledge-transfer paradigm that injects biomedical domain knowledge from large specialized models into a smaller general-purpose model without retraining on the original specialized corpora. DRET is developed as an iterative family of strategies: a unified tokenizer-merge strategy (DRET 1.x), hybrid embedding averaging (DRET 2.0), and a priority-based embedding-transfer mechanism (DRET 3.x) that hierarchically selects embeddings from the most authoritative sour
    
[^108]: 边际而非窗口：无需训练的逐步有损推测解码

    Margins, Not Windows: Training-Free Per-Step Lossy Speculative Decoding

    [https://arxiv.org/abs/2609.02897](https://arxiv.org/abs/2609.02897)

    AdaptiveSpec提出了一种无需训练的逐步推测解码方法，通过边际概率比规则放宽严格的token匹配验证，并动态调整草稿树的深度、宽度和节点数，从而在不受草稿长度和起草器架构限制的情况下加速LLM推理。

    

    推测解码通过起草候选token并并行验证来加速大语言模型（LLM）推理。以EAGLE-3为代表的树注意力起草器被广泛采用，但其通常固定了两个决策：（1）严格的token匹配验证规则，（2）静态的草稿树形状。先前的工作在限制性假设下分别对这两者进行放松：基于长草稿链实现无需训练的有损验证，以及在固定token预算下进行自适应树形调整。我们提出了AdaptiveSpec，一种无需训练的逐步推测解码方法，它利用解码过程中已经产生的内部信号来自适应地调整这两个决策。逐步边际规则在目标模型在草拟token上的概率与其top-1概率之比超过阈值时，接受不匹配的草稿提议token，且不依赖于草稿长度或底层起草器架构。逐步树策略则直接调整草稿树的深度、宽度和节点数。

    arXiv:2609.02897v1 Announce Type: new  Abstract: Speculative decoding accelerates LLM inference by drafting candidate tokens and verifying them in parallel. Tree-attention drafters such as EAGLE-3 are widely adopted, yet typically hold two decisions fixed: (1) a strict token-match verification rule and (2) a static draft-tree shape. Prior work relaxes each in isolation under limiting assumptions: long draft chains for training-free lossy verification, and adaptive tree shaping under a fixed token budget. We introduce AdaptiveSpec, a training-free per-step speculative decoding method that adapts both decisions from internal signals already produced during decoding. A per-step margin rule promotes a mismatched draft-proposed token when the ratio of the target's probability on the drafted token to its top-1 probability exceeds a threshold with no dependence on draft length or underlying drafter architecture. A per-step tree policy adjusts the draft tree's depth, width, and node count dire
    
[^109]: PiPMRE：一种基于语言模型的医学关系抽取流水线

    PiPMRE: A Pipeline Based on Language Model for Medical Relation Extraction

    [https://arxiv.org/abs/2609.02896](https://arxiv.org/abs/2609.02896)

    该论文提出了一种基于语言模型的新型流水线框架PiPMRE，通过关系生成器产生候选关系三元组、再由关系过滤器筛选出合格结果的方式，摆脱了传统序列标注模式的限制，从而提升了医学关系抽取的性能。

    

    医学关系抽取（MRE）通常指从医学文本中联合抽取实体及其关系，近年来受到广泛关注。以往的研究将MRE视为序列标注任务，由于医学实体之间关系的错综复杂，这要么导致标注模式设计的困难，要么无法成功抽取多种关系。在本工作中，我们从语言学的角度重新审视该任务，并提出了一种新颖的流水线框架PiPMRE，该框架基于语言模型开发，以提升医学关系抽取的性能。具体而言，PiPMRE由关系生成器和关系过滤器两部分组成。给定一段文本，生成器首先产生多个关系三元组，然后过滤器对每个三元组进行评分，仅保留通过阈值的那些作为最终结果。实现PiPMRE不需要任何标注模式；取而代之的是，我们使用一个简单的模板来重新表述输入文本……

    arXiv:2609.02896v1 Announce Type: new  Abstract: Medical relation extraction (MRE) is commonly known for extracting entities and their relations jointly from a medical text, which has attracted considerable attention in recent years. Previous studies treat MRE as a sequence tagging task, which results in either a challenging design of the tagging schema or a failed extraction of multiple relations, due to intricate relationships among medical entities. In this work, we review the task from the linguistic perspective and propose a novel pipeline framework, PiPMRE, developed on language models to enhance MRE performance. Specifically, PiPMRE consists of a relation generator and a relation filter. Given a text, the generator first yields multiple relational triplets, and then the filter scores each triplet and retains only those that pass the borderline as the final results. Implementing PiPMRE requires no tagging schema; instead, we use a simple template to reformulate the input text, en
    
[^110]: BharatGather：一个面向印度公共事件虚假信息与假新闻检测的文化感知基准数据集

    BharatGather: A Culturally-Informed Benchmark Dataset for Misinformation and Fake News Detection in Indian Public Events

    [https://arxiv.org/abs/2609.02895](https://arxiv.org/abs/2609.02895)

    本文提出了BharatGather数据集，一个专为印度大型公共活动中虚假信息二元分类设计的文化感知基准数据集，包含14,646条通过事实核查平台爬取、多媒体转录提取与大语言模型合成增强相结合的混合流水线构建的记录。

    

    大型公共活动，如宗教节庆、政治集会和文化聚会，日益容易受到虚假信息快速传播的影响，对公共安全和社会凝聚力构成重大风险。尽管自动假新闻检测在方法论上取得了显著进展，但现有基准往往无法捕捉印度背景下特有的社会文化细微差异和事件特定动态。本文介绍了BharatGather，这是一个精心策划的多源数据集，专门为印度大型集会生态系统中的二元虚假信息分类而设计。该语料库包含14,646条记录，通过混合流水线构建，包括对知名事实核查平台的系统性网络爬取、多媒体转录文本提取，以及由大语言模型（LLM）介导的合成数据增强，以确保叙事的多样性。通过提供一个针对该领域量身定制的资源（摘要原文在此处截断）

    arXiv:2609.02895v1 Announce Type: new  Abstract: Large-scale public events, such as religious festivals, political rallies, and cultural gatherings, are increasingly vulnerable to the rapid dissemination of misinformation, posing substantial risks to public safety and social cohesion. While automated fake news detection has seen significant methodological progress, existing benchmarks frequently fail to capture the socio-cultural nuances and event-specific dynamics characteristic of the Indian context. This paper introduces BharatGather, a curated, multi-source dataset specifically engineered for binary misinformation classification within the ecosystem of Indian mass gatherings. The corpus comprises 14,646 records constructed through a hybrid pipeline involving systematic web scraping of prominent fact-checking platforms, multimedia transcript extraction, and Large Language Model (LLM)-mediated synthetic augmentation to ensure narrative diversity. By providing a resource tailored to t
    
[^111]: R²Adapter：面向高效混合RAG的路由与重写适配器

    R$^{2}$Adapter: A Routing and Rewriting Adapter for Efficient Hybrid RAG

    [https://arxiv.org/abs/2609.02894](https://arxiv.org/abs/2609.02894)

    提出了轻量级即插即用的路由与重写适配器R²Adapter，可动态将查询分配给原生RAG或基于图的RAG，仅对真正受益于图推理的查询进行图检索，从而降低不必要的开销并减少对底层大模型的依赖。

    

    检索增强生成（RAG）已成为利用非参数知识增强大语言模型（LLM）的主流范式。原生RAG能够高效处理简单查询，但在关系推理或多跳推理方面表现不佳。基于图的RAG缓解了这一问题，但会带来更高的推理复杂度和延迟。在实际应用中，用户查询的复杂度差异巨大，采用固定不变的RAG策略并非最优。然而，现有的混合文本-图RAG方法通常依赖启发式方法和基于LLM的路由，导致不必要的开销，并对底层LLM有很强的依赖。为应对这些挑战，我们提出了R²Adapter，一个轻量级、即插即用的路由与重写适配器，旨在动态地将查询分配给原生RAG或基于图的RAG。通过仅对真正能从图推理中受益的查询进行路由，R²Adapter减少了不必要的图检索开销。

    arXiv:2609.02894v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) has become a prevailing paradigm for enhancing Large Language Models (LLMs) with non-parametric knowledge. Vanilla RAG efficiently handles simple queries but struggles with relational or multi-hop reasoning. Graph-based RAG alleviates this issue but incurs higher inference complexity and latency. In practice, user queries can differ significantly in their complexity, rendering a fixed RAG strategy suboptimal. However, existing hybrid text-graph RAG methods typically rely on heuristic and LLM-based routing, resulting in unnecessary overhead and strong dependence on the underlying LLM. To address these challenges, we propose R$^{2}$Adapter, a lightweight plug-in Routing and Rewriting Adapter designed to allocate queries between vanilla and graph-based RAG dynamically. By routing only the queries that genuinely benefit from graph-based reasoning, R$^{2}$Adapter reduces unnecessary graph retrieval overhea
    
[^112]: 作为子空间选择的探针泛化方法用于分布外欺骗检测

    Probe Generalization as Subspace Selection for OOD Deception Detection

    [https://arxiv.org/abs/2609.02893](https://arxiv.org/abs/2609.02893)

    该论文发现，将输入投影到训练激活分布中经LLM评判器筛选出的关键主成分子集上，可大幅提升线性探针在分布外欺骗检测任务上的跨域泛化能力，在Insider Trading Report上缩小了78%的性能差距。

    

    线性探针可用于检测语言模型激活中的行为和概念，但在迁移到分布外样本时可能会失效。在研究Llama-3.1-8B-Instruct探针在3个保留的欺骗检测数据集上的泛化性能时，我们发现将输入投影到来自激活训练分布的少量主成分（PCs）子集上，能够实现跨域迁移，其性能几乎与直接在测试分布上训练的探针相当。此外，我们发现主成分的解释分析可用于找出其中可迁移主成分的子集。通过使用LLM评判器对每个主成分进行评分，判断其最大/最小激活样本是否暗示一个可迁移的欺骗方向，然后对得分最高的主成分进行探测，我们在Insider Trading Report上缩小了基线到oracle之间差距的78%，在Sandbagging上缩小了25%。源探针高度加权所指向的方向似乎编码了……

    arXiv:2609.02893v1 Announce Type: new  Abstract: Linear probes can be used to detect behaviors and concepts inside language model activations, but may fail to transfer to out-of-distribution examples. When studying the generalization performance of Llama-3.1-8B-Instruct probes over 3 held-out deception detection datasets, we find that projecting inputs onto a small subset of principal components (PCs) from the training distribution of activations enables cross-domain transfer that nearly matches the performance of probes trained directly on the test distribution. Furthermore, we find that PC interpretations can be used to find a subset of those transferable PCs. By using an LLM judge to score each PC on whether its most/ least activating examples imply a transferable deception direction, then probing on the highest-scoring PCs, we close the baseline-to-oracle gap by 78% on Insider Trading Report and by 25% on Sandbagging. The directions a source probe weights heavily appear to encode s
    
[^113]: 反例作为智能体自我纠正的反馈

    Counterexamples as Feedback for Agent Self-Correction

    [https://arxiv.org/abs/2609.02892](https://arxiv.org/abs/2609.02892)

    本文提出 A-CEGIS 轻量级框架，将反例作为反馈来评估智能体在自然语言到正则表达式合成中的多轮自我纠正能力，在四轮预算内解决了 90% 的任务，显著优于零样本生成和通用自我纠正方法。

    

    单轮代码生成指标低估了已部署智能体的一个核心特性：它们在收到具体反馈后能否修复错误的产物。本文提出了 A-CEGIS，一个轻量级框架，它使用反例作为反馈来评估从自然语言到正则表达式合成任务中的多轮精炼能力。智能体提出一个正则表达式，确定性预言机在全匹配语义下对其进行检查，紧凑的假阳性或假阴性反例样本则引导下一轮迭代。在 30 个 NL-RX-Turk 任务上，诊断性反例反馈在四轮消融预算内解决了 90% 的任务，相比之下，零样本生成仅为 17%，通用自我纠正为 27%，仅错误反馈为 23%。在带有强化的完整诊断运行中，所有任务在最终轮次时都在隐藏测试集上得到解决，平均成功耗时为 2.7 轮，经过针对性探测后的稳健成功率为 77%。这些结果表明，A-CEGIS 衡量的是智能体在实际部署场景中利用具体反馈进行自我修复的能力。

    arXiv:2609.02892v1 Announce Type: cross  Abstract: Single-turn code-generation metrics understate a central property of deployed agents: whether they can repair a wrong artifact after receiving concrete feedback. This paper presents A-CEGIS, a lightweight framework that uses counterexamples as feedback for evaluating multi-turn refinement in natural-language-to-regex synthesis. An agent proposes a regex, a deterministic oracle checks it under full-match semantics, and compact false-positive or false-negative witnesses guide the next turn. On 30 NL-RX-Turk tasks, diagnostic counterexample feedback solves 90\% of tasks within a four-turn ablation budget, compared with 17% for zero-shot generation, 27% for generic self-correction, and 23% for error-only feedback. In a full diagnostic run with hardening, all tasks are solved on the hidden set by the final turn, with mean time-to-success of 2.7 turns and robust success of 77% after targeted probing. These results show that A-CEGIS measures 
    
[^114]: 有界人格画像在冻结智能体上于分类任务可媲美检索、但在回归任务上则不然

    Bounded Personas Match Retrieval on Classification but Not Regression for a Frozen Agent

    [https://arxiv.org/abs/2609.02890](https://arxiv.org/abs/2609.02890)

    提出免训练方法PersonaLink，将用户历史蒸馏为有界的三字段人格画像并通过递归自评估精炼，证明该蒸馏人格画像在冻结智能体的分类任务上能够媲美检索方法，但在回归任务上仍无法匹敌。

    

    个性化语言智能体必须在推理时将用户的交互历史转化为针对每个新请求的行为。目前有两种主流策略。检索方法将用户过去最相关的少量条目拉入提示词中，这种方法准确，但每次查询都需付出选择与上下文成本，且该成本会随历史记录的增长而增加。蒸馏方法则将历史一次性压缩为紧凑的自然语言人格画像，它是有界的、与具体查询无关且可解释的，但人们普遍认为它以牺牲准确性为代价。蒸馏出的人格画像能否匹配检索、以及在哪些任务上能够匹配，这一问题尚未得到清晰的刻画。我们提出 PersonaLink，这是一种免训练方法，它将用户历史蒸馏为有界的三字段人格画像，并进行递归精炼：每一轮在用户自身带标注历史的保留切片上对冻结智能体进行自我评估，根据其错误重写人格画像，且仅当结果未出现退化时才予以保留

    arXiv:2609.02890v1 Announce Type: new  Abstract: A personalized language agent must convert a user's interaction history into behavior on each new request at inference time. Two strategies dominate. Retrieval pulls a few of the user's most relevant past items into the prompt, which is accurate but pays a per-query selection and context cost that grows with the history. Distillation instead compresses the history once into a compact natural-language persona, which is bounded, query-independent, and interpretable, but is widely assumed to sacrifice accuracy. Whether, and on which tasks, a distilled persona can match retrieval has not been characterized cleanly. We introduce PersonaLink, a training-free method that distills a user's history into a bounded three-field persona and recursively refines it: each pass self-evaluates the frozen agent on a held-out slice of the user's own labeled history, rewrites the persona from its errors, and keeps the result only when it does not regress on 
    
[^115]: Harness优化的价值究竟在哪里？自进化LLM智能体中的局部化收益与预算分割陷阱

    Where Does Harness-Optimization Value Live? Localized Gains and the Budget-Splitting Trap in Self-Evolving LLM Agents

    [https://arxiv.org/abs/2609.02889](https://arxiv.org/abs/2609.02889)

    提出HARNESSEVO，将智能体harness分解为角色、任务策略、工具/格式规则和反思/控制四个可独立进化的槽位，通过等预算对照与留一归因分析揭示：整体成功率提升有限（预算分割会稀释收益），优化的价值主要局部化于特定槽位。

    

    越来越多的研究通过进化冻结的大型语言模型（LLM）智能体的harness——即模型周围的文本脚手架，包括角色设定、策略、格式规则和控制启发式方法——来提升其性能。现有的反思式提示进化方法通常将harness作为一个扁平字符串进行整体优化。我们则追问：优化的价值究竟位于何处？我们提出HARNESSEVO，它将harness分解为四个可独立进化的槽位：角色、任务策略、工具/格式规则以及反思/控制。在等预算设置下使用相同的反思式优化器，并结合留一纳入与留一排除的归因方法来衡量每个槽位的贡献。在ALFWorld上使用冻结的7B骨干模型时，HARNESSEVO在整体二元成功率上相比原始harness或扁平字符串进化均无显著提升：分别为0.657对0.642和0.642。然而，槽位级分析……（摘要原文在此处截断）

    arXiv:2609.02889v1 Announce Type: new  Abstract: A growing body of work improves frozen large language models (LLMs) as agents by evolving their harness: the textual scaffolding around the model, including persona, strategy, format rules, and control heuristics. Existing reflective prompt-evolution methods usually optimize this harness as one flat string. We instead ask where the optimization value actually resides. We introduce HARNESSEVO, which decomposes the harness into four separately evolvable slots: role, task-strategy, tool/format-rules, and reflection/control. Using the same reflective optimizer under an iso-budget setting, we pair this decomposition with leave-one-in and leave-one-out attribution to measure the contribution of each slot.   On ALFWorld with a frozen 7B backbone, HARNESSEVO does not significantly improve the overall binary success rate over either the stock harness or flat-string evolution: 0.657 versus 0.642 and 0.642, respectively. However, the slot-level ana
    
[^116]: ViSAR：面向视觉文档问答的无需训练的自适应k值检索方法

    ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering

    [https://arxiv.org/abs/2609.02486](https://arxiv.org/abs/2609.02486)

    提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。

    

    文档视觉问答通常利用检索增强生成技术，其中晚期交互编码器常被用于识别与用户查询相关的文档页面，然后由大型视觉-语言模型生成答案。现有方法通常无论查询复杂度如何都检索固定数量的前k个页面，这会增加大型视觉-语言模型的延迟，并可能降低答案的准确性。我们提出了ViSAR（视觉语义激活检索），这是一种面向晚期交互视觉文档检索的无需训练的自适应k值检索方法。ViSAR直接在嵌入空间中运行，构建以查询为条件的页面级相似度矩阵，突出与查询相关的语义，并动态确定需要检索的页面数量。在多个编码器和大型视觉-语言模型上的实验表明，ViSAR能够检索紧凑且适应查询的页面集合，将RAG延迟降低高达58.7%，同时保持或提升答案准确性。

    arXiv:2609.02486v1 Announce Type: cross  Abstract: Document Visual Question Answering (DocVQA) often leverages Retrieval-Augmented Generation (RAG), where late-interaction encoders are commonly used to identify document pages relevant to a user query, before answer generation by a Large Vision-Language Model (LVLM). Existing approaches typically retrieve a fixed top-$k$ number of pages regardless of query complexity, which increases LVLM latency and may degrade answer accuracy. We introduce ViSAR (Visual Semantic Activation Retrieval), a training-free adaptive-$k$ retrieval method for late-interaction visual document retrieval. ViSAR operates directly in the embedding space to construct a query-conditioned page-level similarity matrix that highlights query-relevant semantics and dynamically determines the number of pages to retrieve. Across multiple encoders and LVLMs, ViSAR retrieves compact, query-adapted page sets that reduce RAG latency by up to 58.7\%, while maintaining or improvi
    
[^117]: 多模态大语言模型中跨模态安全漂移的安全意识迁移

    Transfer Safety Awareness for Cross-Modal Safety Drift in Multimodal Large Language Models

    [https://arxiv.org/abs/2609.02082](https://arxiv.org/abs/2609.02082)

    针对多模态大语言模型中“跨模态安全漂移”这一新安全问题（无害文本结合图像即可传达有害意图且模型难以拒绝），提出轻量级的安全意识表示迁移方法（SRT），将文本安全信号迁移至视觉场景以有效缓解该风险。

    

    视觉模态增强了多模态大语言模型（MLLMs）的能力，但也引入了安全隐患：一个本身无害的文本查询在与视觉图像结合时可能传达有害意图。我们将这种现象称为“跨模态安全漂移”，我们的初步研究表明，此类请求的安全响应率显著低于包含明确不安全文本的请求。本文旨在系统研究这一问题。首先，我们进行了实证分析，识别出代表性的不安全响应模式。在此基础上，我们对模型表示和注意力机制进行了解释分析，揭示出视觉风险线索受到的关注有限，难以有效触发拒绝响应。受不安全文本处理中的安全信号可以迁移这一观察的启发，我们提出了安全意识表示迁移，这是一种轻量级的方向细化方法，能够缓解跨模态安全漂移并显著提升……

    arXiv:2609.02082v1 Announce Type: cross  Abstract: Visual modality enhances the capabilities of multimodal large language models (MLLMs) but also introduces a safety concern: a benign textual query may convey harmful intent when grounded in a visual image. We term this cross-modal safety drift and our pilot studies show that the safety response rate for such requests is substantially lower than that for requests containing explicitly unsafe text. This paper aims to systematically study this issue. First, we conduct an empirical analysis to identify representative unsafe response patterns. Building on these, we interpret model representations and attentions, revealing that visually risky cues receive limited attention and weakly trigger refusal. Motivated by the observation that safety signals from unsafe text processing can be transferred, we propose safety-awareness representation transfer (SRT), a lightweight direction-refinement method that mitigates cross-modal safety drift with a 
    
[^118]: SFAD：面向事实性的推测解码

    SFAD: Speculative Factuality-Aware Decoding

    [https://arxiv.org/abs/2609.00796](https://arxiv.org/abs/2609.00796)

    提出SFAD推测解码框架，通过构建细粒度扰动偏好数据集ConFide并利用DPO训练上下文忠实草稿模型，结合认知摩擦机制检测幻觉，在不增加推理开销的情况下显著增强大语言模型的上下文忠实度。

    

    作为大语言模型最关键的挑战之一，上下文忠实度直接决定了其在知识密集型应用中的可靠性。这项任务尤其具有挑战性，因为它需要在事实一致性与生成效率之间取得平衡。对比解码方法需要进行双重前向传播（分别带上下文和不带上下文）来比较模型输出，使推理计算开销翻倍；而后训练对齐则需要大量的强化学习，带来高昂的计算开销。为应对这一挑战，我们提出了SFAD，一种能够在不降低推理性能的前提下增强上下文忠实度的推测解码框架。我们首先构建了ConFide，一个包含细粒度原子级扰动的偏好数据集，用于通过直接偏好优化（DPO）训练一个上下文忠实的草稿模型。在推理过程中，认知摩擦通过量化……来检测潜在的幻觉。（注：原摘要在此处被截断）

    arXiv:2609.00796v1 Announce Type: new  Abstract: As one of the most critical challenges in large language models, contextual faithfulness directly determines their reliability in knowledge-intensive applications. This task is particularly challenging as it requires balancing factual consistency with generation efficiency. Contrastive decoding methods require dual forward passes (with and without context) to compare model outputs, doubling inference computational overhead, while post-training alignment demands extensive reinforcement learning with substantial computational overhead. To address this challenge, we present \textbf{SFAD}, a speculative decoding framework that enhances contextual faithfulness without inference degradation. We first construct \textbf{ConFide}, a preference dataset with fine-grained atomic perturbations, to train a context-faithful draft model via Direct Preference Optimization. During inference, Epistemic Friction detects potential hallucinations by quantifyi
    
[^119]: 科学智能体技能：面向科研智能体的程序性知识库

    Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents

    [https://arxiv.org/abs/2609.00065](https://arxiv.org/abs/2609.00065)

    该论文提出了一个名为“科学智能体技能”的开放库，收录了基因组学、化学信息学等16个科研实践领域共163项程序性知识，使语言模型智能体能够遵循领域规范做出站得住脚的科学分析，而非仅仅返回能运行的代码。

    

    被要求分析实验的语言模型智能体通常只会返回一段能运行的代码，但该分析是否站得住脚则是另一回事。一个站得住脚的分析取决于程序性选择：该领域接受哪种统计检验方法、哪个标识符命名空间是权威的、以及结果必须附带哪些注意事项。我们提出了“科学智能体技能”，这是一个开放的知识库，包含16个实践领域的163项此类程序，涵盖基因组学、化学信息学、医学影像、研究设计和科学传播等。每项技能都是一个目录，围绕一个版本化、人类可读的指令文件构建。智能体仅在任务需要时才加载该文件；目录中通常还包含参考资料和可运行的脚本。我们未报告任务级评估结果和宿主选择率。该库采用开放许可证，可在 https://github.com/K-Dense-AI/scientific-agent-skills 获取。

    arXiv:2609.00065v1 Announce Type: cross  Abstract: A language-model agent asked to analyse an experiment will usually return working code. Whether the analysis is defensible is a different question. A defensible analysis depends on procedural choices: which test the field accepts, which identifier namespace is authoritative, and which caveats must accompany a result. We present Scientific Agent Skills, an open library of 163 such procedures in 16 areas of practice, including genomics, cheminformatics, medical imaging, study design and scientific communication. Each skill is a directory built around a versioned, human-readable instruction file. An agent loads the file only when a task calls for it; the directory often also contains reference material and runnable scripts. We report no task-level evaluation and no host selection rate. Openly licensed and available at https://github.com/K-Dense-AI/scientific-agent-skills.
    
[^120]: 面向单GPU大语言模型推理的预算感知压缩流水线：方法、权衡与耦合效应

    Budget-Aware Compression Pipeline for Single-GPU LLM Inference: Methods, Trade-offs, and Coupling Effects

    [https://arxiv.org/abs/2608.30076](https://arxiv.org/abs/2608.30076)

    该论文提出一种预算感知的单GPU大模型压缩流水线，通过系统研究剪枝、量化与KV缓存压缩之间的耦合效应，将70B模型压缩至约33GB并在单张A40上实现约57 tokens/s的推理速度，同时将精度损失控制在5%以内。

    

    在NVIDIA GPU上单卡部署700亿参数（70B）的大语言模型受到设备显存、长上下文吞吐量和工程集成成本的制约。我们将单GPU推理建模为一个围绕这三个维度的预算感知设计问题，并研究剪枝、量化和KV缓存压缩在真实执行条件下如何相互作用。受控消融实验表明，逐层剪枝能够使权重量化更加鲁棒。KV缓存稀疏化与INT8 KV量化形成互补，能够在不损害解码速度的前提下减少显存占用，而静态向量量化器则常常与动态缓存机制发生冲突。基于这些耦合效应的研究结果和显式的预算跟踪，我们构建了一套实用的压缩流水线，将一个70B模型压缩至约33 GB，在单张A40上对10k token的提示词实现了约57 tokens/s的推理速度，并在常识与推理基准测试上将绝对精度损失控制在5%以内。我们贡献了设计准则和可复现的评估流程。

    arXiv:2608.30076v1 Announce Type: new  Abstract: Single-GPU deployment of 70B-parameter language models on an NVIDIA GPU is constrained by device memory, long-context throughput, and engineering integration cost. We cast single-GPU inference as a budget-aware design problem over these three axes and study how pruning, quantization, and KV-cache compression interact under realistic execution. Controlled ablations show that layer-wise pruning makes weight quantization more robust. KV-cache sparsification complements INT8 KV quantization by reducing memory without hurting decoding speed, while static vector quantizers often conflict with dynamic caching. Guided by these coupling results and explicit budget tracking, we assembled a practical pipeline and compressed a 70B model to about 33 GB, sustained about 57 tokens/s on 10k token prompts on a single A40, and kept absolute accuracy within 5% on common and reasoning benchmarks. We contribute design rules and a reproducible evaluation prot
    
[^121]: Puro-2B：在RTX 5090上花费5090美元训练的“穷人版”Qwen2-1.5B

    Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090

    [https://arxiv.org/abs/2608.27370](https://arxiv.org/abs/2608.27370)

    本文提出了一种开源且成本高效的预训练配方，使在消费级RTX 5090 GPU上以极低计算成本训练出接近Qwen2.5-1.5B性能的Puro-2B模型成为可能。

    

    arXiv:2608.27370v1 公告类型：新  摘要：语言模型预训练几乎已成为高昂成本的代名词，使其对学术界和开源社区的大部分人来说遥不可及。尽管已有强大的开源努力，包括开放权重模型和开源训练配方，但长期以来一直缺少一种成本高效、硬件可访问且开源预训练配方。即使在小规模下，训练Llama-3.2-3B也需花费超过150万美元，而复现SmolLM3-3B则需超过70万美元。在本报告中，我们提出了一种旨在降低这一门槛的开源预训练配方。利用该配方，我们从零开始训练了一系列Puro-2B模型，使用FP8精度，在消费级RTX 5090 GPU上处理多达1.4万亿个令牌。该系列中的模型在令牌预算和所选配方变体上有所不同。我们最好的模型以不到6900美元的计算成本进行训练，并在我们的评估协议下接近Qwen2.5-1.5B的性能。这种成本效益...

    arXiv:2608.27370v1 Announce Type: new  Abstract: Language model pretraining has become almost synonymous with prohibitive cost, placing it out of reach for much of the academic and open-source communities. Although strong open-source efforts already exist, including open-weight models and open-source training recipes, a cost-efficient, hardware-accessible, and open-source pretraining recipe has long been missing. Even at a small scale, training Llama-3.2-3B costs over \$1.5M, and reproducing SmolLM3-3B needs over \$700K. In this report, we present an open pretraining recipe designed to lower this barrier. Using this recipe, we train a collection of Puro-2B models from scratch on up to 1.4 trillion tokens with FP8 precision on consumer-grade RTX 5090 GPUs. The models in the collection differ in token budgets and selected recipe variants. Our best model is trained at a compute cost of less than \$6.9K and approaches Qwen2.5-1.5B performance under our evaluation protocol. This cost effici
    
[^122]: JIT-Agent：通过即时工具进化扩展智能体能力的规模化方法

    JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution

    [https://arxiv.org/abs/2608.25593](https://arxiv.org/abs/2608.25593)

    JIT-Agent通过训练一个能即时生成和优化任务自适应工具模型的系统，显著提升了现成智能体的性能，使工具设计从手动变为自动化。

    

    arXiv:2608.25593v1 公告类型：新 摘要：智能体的能力并非仅由模型决定。智能体工具（包括记忆管理、规划策略、动作协议以及工具/技能编排）可能主导底层基础模型的贡献。然而，工具设计仍是手动的、任务特定的，并且从根本上不可扩展。我们提出了JIT-Agent，一个经过训练的工具智能模型，能够即时为任意现成的智能体大型语言模型合成任务自适应的工具。我们将智能体工具形式化为一个可组合、可机器生成的工件，由固定的四模块协议控制，并训练JIT-Agent为给定任务定制工具，修复工具以确保稳定可靠的执行，并通过从先前工具配置的扩展存档中提取性能信号来自我进化。配备JIT-Agent作为工具助手，DeepSeek-V4-Flash在DeepSearchQA（+9.1）和OdysseyBench（+4.3）上超越了GPT-5.6。

    arXiv:2608.25593v1 Announce Type: new  Abstract: Agent capability is not determined by the model alone. The agent harness, encompassing memory management, planning strategy, action protocol, and tool/skill orchestration, can dominate the contribution of the underlying foundation model. Yet harness design remains manual, task-specific, and fundamentally unscalable. We present JIT-Agent, a harness intelligence model trained to synthesize task-adaptive agent harnesses on the fly for arbitrary off-the-shelf agentic LLMs. We formalize the agent harness as a composable, machine-generatable artifact governed by a fixed four-module protocol, and train JIT-Agent to customize harnesses for a given task at hand, repair harnesses for stable and reliable execution, and self-evolve by distilling performance signals from an expanding archive of prior harness configurations. Equipped with JIT-Agent as a harness helper, DeepSeek-V4-Flash surpasses GPT-5.6 on DeepSearchQA (+9.1) and OdysseyBench (+4.3),
    
[^123]: TRACE：一种自我进化的技能库，用于一致且具备限制意识的LLM代理

    TRACE: A Self-Evolving Skill Bank for Consistent, Limit-Aware LLM Agents

    [https://arxiv.org/abs/2608.22793](https://arxiv.org/abs/2608.22793)

    TRACE通过构建自我进化的技能库，在不修改模型权重的情况下，提升LLM代理在重复任务中的一致性和限制意识，弥合了单次成功与一致成功之间的可靠性差距。

    

    arXiv:2608.22793v1 公告类型：交叉 摘要：在面向用户的产品中可靠部署LLM代理，不仅取决于原始任务解决能力，还取决于一致性和限制意识：即在重复试验中表现相同，并识别请求何时无法或暂时无法安全完成。CAR-bench在车载助手领域揭示了这一可靠性缺口：一个由LLM模拟的用户发出不完整或模糊的请求，要求代理通过多轮对话和工具使用来解决不确定性，同时严格遵守领域政策。即使是前沿模型，在其至少能解决一次（Pass@3）和跨试验一致解决（Pass^k）之间也显示出显著差距。我们用TRACE（轨迹对比进化）弥合了这一差距，该方法在不修改模型权重的情况下，迭代改进基于技能的代理的行为知识。这些知识被组织为一个可检索的模块化技能库，每个技能编码一个自包含的行为模式。

    arXiv:2608.22793v1 Announce Type: cross  Abstract: Reliable deployment of LLM agents in user-facing products depends not on raw task-solving ability but on consistency and limit-awareness: behaving the same way across repeated trials, and recognizing when a request cannot, or cannot yet, be safely fulfilled. CAR-bench exposes this reliability gap in the domain of in-car assistants: an LLM-simulated user issues incomplete or ambiguous requests, requiring the agent to resolve uncertainty through multi-turn dialogue and tool use while strictly adhering to domain policies. Even frontier models show a substantial gap between what they can solve at least once (Pass@3) and what they solve consistently across trials (Pass^k). We bridge this gap with TRACE (TRAjectory-Contrastive Evolution), which iteratively improves a skill-based agent's behavioral knowledge without modifying model weights. This knowledge is organized as a Skill Bank of modular, retrievable skills, each encoding a self-contai
    
[^124]: K-Bench：在真实科学代理请求上衡量模型性能

    K-Bench: measuring model performance on real scientific agent requests

    [https://arxiv.org/abs/2608.21601](https://arxiv.org/abs/2608.21601)

    本论文提出K-Bench 01，一个基于真实科学请求的评估框架，发现当前前沿模型在满足领域科学家接受标准上均未达到阈值，其中gpt-5.6-sol表现最优但仍有不确定性。

    

    arXiv:2608.21601v1 公告类型：新 摘要：科学人工智能的基准测试大多是为评分而编写的：多项选择题、带参考答案的策划代理任务，或具有已知生成结构的模拟器。真实的科学请求则有所不同。它们规定不充分，携带附件，且缺乏基本事实。我们报告了K-Bench 01，一个从K-Dense Web实时用户流量中抽取的首轮请求构建的评估，并由九个前沿模型在相同沙盒环境中端到端运行，产生了1,602个完成的代理运行。三个盲审语言模型裁判根据八维评分标准对每次运行进行评分。在一个8锚点指示裁判认为领域科学家会接受该工作（仅需少量修改）的评分标准上，没有模型能在所有三位裁判下达标。gpt-5.6-sol具有最高的汇总平均值，为8.04，但其95%置信区间[7.80, 8.23]跨越了阈值，且三位裁判中有两位将claude-opus-5排在第一。

    arXiv:2608.21601v1 Announce Type: new  Abstract: Benchmarks for scientific artificial intelligence are mostly written to be scored: multiple-choice questions, curated agent tasks with reference solutions, or simulators with a known generative structure. Real scientific requests arrive differently. They are underspecified, they carry attachments, and lack ground truth. We report K-Bench 01, an evaluation built from first-turn requests sampled from live user traffic on K-Dense Web and run end to end by nine frontier models in identical sandboxes, yielding 1,602 completed agent runs. Three blinded language-model judges scored every run against an eight-dimension rubric. On a rubric whose 8-anchor instructs judges that a domain scientist would accept the work with minor edits, no model clears the line under all three judges. gpt-5.6-sol has the highest pooled mean, 8.04, but its 95% interval [7.80, 8.23] spans the threshold, and two of the three judges rank claude-opus-5 first instead. We 
    
[^125]: DeltaMomentum：一种基于键值对的各向异性动量更新方法，采用增量规则

    DeltaMomentum: A Key-Value based Anisotropic Momentum Update via Delta Rule

    [https://arxiv.org/abs/2608.19491](https://arxiv.org/abs/2608.19491)

    DeltaMomentum通过利用梯度中的键值结构，将方向感知引入动量更新规则，使每个方向以与其出现频率相关的速率被遗忘，从而无需矩阵即可实现输入侧曲率校正。

    

    arXiv:2608.19491v1 公告类型：交叉 摘要：大多数现代优化器将动量形成过去梯度的指数移动平均（EMA），以固定速率遗忘每个方向。然而，深度网络在训练过程中看到的输入可能高度各向异性，少数方向频繁被查询，而大多数方向很少出现。近期方法通过在该缓冲区外增加额外处理来应对这种各向异性，但动量更新本身保持不变。我们提出DeltaMomentum，将方向感知构建到动量更新规则中。主要观察是线性层的梯度分解为作为键的输入和作为值的输出侧误差。利用键值结构，DeltaMomentum通过标准增量规则更新动量缓冲区，使每个方向以其出现频率设定的速率被遗忘。我们证明这是一种有效的动量，它无需矩阵即可应用输入侧曲率校正。

    arXiv:2608.19491v1 Announce Type: cross  Abstract: Most modern optimizers form their momentum as an exponential moving average (EMA) of past gradients, forgetting every direction at one fixed rate. However, the inputs a deep network sees during training can be highly anisotropic, with a few directions queried frequently while most are seen rarely. Recent methods address this anisotropy by wrapping extra processing around this buffer, leaving the momentum update itself unchanged. We propose DeltaMomentum, which builds direction-awareness into the momentum update rule. The main observation is that the gradient of a linear layer splits into an input that acts as a key and an output-side error that acts as a value. Exploiting the key-value structure, DeltaMomentum updates the momentum buffer by the canonical delta rule, so each direction is forgotten at a rate set by how often it appears. We prove that it is a valid momentum, that it applies the input-side curvature correction without matr
    
[^126]: Ex-Omni-2D：具备原生视觉形象的表达性全模态对话模型

    Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence

    [https://arxiv.org/abs/2608.10720](https://arxiv.org/abs/2608.10720)

    Ex-Omni-2D 提出通过“视觉思维计划”协调文本、个性化语音与基于参考视频的生成，使全模态对话智能体在语音回答的同时拥有原生视觉形象，并可将全序列教师模型蒸馏为少步骤的流式学生模型。

    

    全模态对话模型能够理解多模态输入并合成语音回复，但语音回答仍然使智能体在视觉上“缺席”。我们提出了 Ex-Omni-2D，一个能够以协调的文本、个性化语音和基于参考条件的视频来回答多模态查询的框架。该对话模型首先为场景、情感和动作编写结构化的“视觉思维计划”（Visual Thought Plan, VTP），然后生成回复文本和多码本语音单元。这些语音单元被解码为音频并与视频帧对齐，为语音模块和虚拟形象模块提供共同的时间同步信号，同时允许二者从不同的数据源中学习。视频模块作为一个全序列的“教师”模型进行训练，其条件包括参考外观、VTP 语义以及帧对齐的语音单元。我们进一步探索将其蒸馏为一个少步骤、块因果的“流式学生”模型；其前缀流式机制携带……（原文此处截断）

    arXiv:2608.10720v2 Announce Type: replace  Abstract: Omni-modal dialogue models can understand multimodal inputs and synthesize spoken replies, but a spoken answer still leaves the agent visually absent. We introduce \textbf{Ex-Omni-2D}, a framework that answers a multimodal query with coordinated text, personalized speech, and reference-conditioned video. The dialogue model first writes a structured \textit{Visual Thought Plan} (VTP) for scene, emotion, and motion, then generates the response text and multi-codebook speech units. These speech units are decoded into audio and aligned with video frames, giving the speech and avatar modules a common timing signal while allowing them to learn from different data sources. The video module is trained as a full-sequence Teacher conditioned on reference appearance, VTP semantics, and frame-aligned speech units. We further explore to distill it into a few-step block-causal \emph{Streaming Student}; its Prefix Streaming mechanism carries the pr
    
[^127]: Gaokerena：一个小型波斯语医学语言模型家族

    Gaokerena: A Small Persian Medical Language Model Family

    [https://arxiv.org/abs/2608.00932](https://arxiv.org/abs/2608.00932)

    本文提出了Gaokerena，一个专为消费级硬件设计的小型波斯语医学语言模型家族，其中Gaokerena-V通过新构建的波斯语医学语料库训练提升了医学问答性能，Gaokerena-R则结合思维链与两个新型RLAIF框架来增强临床推理能力。

    

    人工智能融入医学问答系统的发展迅速；然而，相关研究仍主要集中在英语上，导致波斯语等低资源语言的服务严重不足。为填补这一空白，本文提出了Gaokerena，这是一个新型的小型波斯语医学语言模型家族，专为在消费级硬件上部署而优化。作为迈向本地化数字医疗的基础步骤，我们首先介绍了Gaokerena-V，它是通过在一个新构建的9000万词元波斯语医学语料库和2万个经专家审核的医生问答对上训练基线模型而开发的，其在翻译版医学MMLU基准上的性能从46.28%提升至49.31%。其次，考虑到临床推理的关键需求，我们通过将思维链方法与两个新颖的AI反馈强化学习（RLAIF）框架相结合，开发了Gaokerena-R，以优化偏好……

    arXiv:2608.00932v2 Announce Type: replace  Abstract: The integration of artificial intelligence into medical question-answering systems has advanced rapidly; however, research remains predominantly focused on English, leaving low resource languages like Persian significantly underserved. To address this gap, this paper introduces Gaokerena, a novel family of compact Persian medical language models optimized for deployment on consumer grade hardware. As a foundational step toward localized digital healthcare, we first present Gaokerena-V, developed by training a baseline model on a newly curated 90-million-token Persian medical corpus and 20,000 expert-vetted physician Q&A pairs, which improved performance on a translated medical MMLU benchmark from 46.28% to 49.31%. Second, recognizing the critical demands of clinical reasoning, we developed Gaokerena-R by integrating a Chain-of-Thought approach with two novel Reinforcement Learning with AI Feedback (RLAIF) frameworks to optimize prefe
    
[^128]: 在开放权重语言模型中读取与调控材料科学机制的表征

    Reading and Steering Representations of Materials-Science Mechanisms in an Open-Weight Language Model

    [https://arxiv.org/abs/2607.20058](https://arxiv.org/abs/2607.20058)

    该研究在开放权重语言模型中首次识别出材料科学机制内部表征的三个可实验验证的特征，并证明通过因果干预可以读取并调控模型对材料物理机制的表征。

    

    大语言模型能够回答科学问题，然而正确的输出并不能揭示模型是否真正表征或运用了其背后的物理规律。本研究使用三个开放权重的 Gemma 4 模型，识别出材料科学机制信息的三个可通过实验区分的特征：选择性概念可读性、定性本构取向的关系性编码，以及对受限工程答案的因果性、上下文相关控制。我们结合了匹配的直接词汇读出与雅可比词汇读出、无选项状态几何、包含60条定律的反事实基准以及因果干预方法。在50个保留的材料描述中，三个独立拟合的雅可比透镜重现了概念排名，且来自两种读出方法的无目标词集使得在盲测条件下能够识别10个机制族中的9个。另一个包含72个提示的独立基准产生了机制特异性的……

    arXiv:2607.20058v2 Announce Type: replace  Abstract: Large language models can answer scientific questions, yet a correct output does not reveal whether the model represents or uses the governing physics. Here, using three open-weight Gemma 4 models (google/gemma-4-E4B-it, google/gemma-4-12B-it, google/gemma-4-31B-it) we identify three experimentally separable signatures of materials-science mechanism information: selective concept readability, relational encoding of qualitative constitutive orientation, and causal, context-dependent control of constrained engineering answers. We combine matched direct and Jacobian vocabulary readouts, option-free state geometry, a 60-law counterfactual benchmark and causal interventions. In 50 held-out materials descriptions, three independently fitted Jacobian lenses reproduced concept ranks, and target-free word sets from both readouts enabled blinded identification of 9 of 10 mechanism families. A separate 72-prompt benchmark produced mechanism-spe
    
[^129]: PalmClaw：一个面向手机的原生端上智能体框架

    PalmClaw: A Native On-Device Agent Framework for Mobile Phones

    [https://arxiv.org/abs/2607.13027](https://arxiv.org/abs/2607.13027)

    PalmClaw 是一个原生运行在手机上的开源智能体框架，直接在设备端管理会话、记忆、技能与工具，并将设备能力封装为可调用的设备工具，从而突破了传统依赖 GUI 操作的移动智能体的局限。

    

    大语言模型（LLM）智能体已经超越了单纯生成回复的阶段，转向通过调用工具、观察结果并迭代决定下一步行动来执行多步骤任务。大多数智能体系统运行在桌面电脑或服务器上，以支持工具使用和任务自动化。移动设备同样是重要的智能体运行环境，因为它们普及度高、易于访问，并包含用户的数据、传感器和日常使用的应用程序。现有的移动智能体主要通过点击、滑动、输入等图形用户界面（GUI）操作来控制智能手机，这种方式往往形成冗长且依赖界面的操作序列，无法直接访问设备能力，并且使执行边界难以界定。我们提出了 PalmClaw，一个开源的智能体框架，它原生运行在手机上，直接在设备端管理会话、记忆、技能、工具和智能体循环。PalmClaw 将设备能力以设备工具的形式暴露……（原文摘要在此处截断）

    arXiv:2607.13027v2 Announce Type: replace-cross  Abstract: Large Language Model (LLM) agents have moved beyond generating responses to executing multi-step tasks by calling tools, observing the results, and iteratively deciding the next action. Most agent systems run on desktops or servers, which support tool use and task automation. Mobile devices are also important agent environments because they are widely accessible and contain users' data, sensors, and daily-use applications. Existing mobile agents mainly operate smartphones through graphical user interface (GUI) actions such as tapping, swiping, and typing, which often form long, interface-dependent sequences, cannot directly access device capabilities, and make execution boundaries difficult to define. We present PalmClaw, an open-source agent framework that runs natively on mobile phones and manages the sessions, memory, skills, tools, and agent loop directly on the device. PalmClaw exposes device capabilities as device tools w
    
[^130]: 所见即所得：面向图表到代码生成的观察对齐监督

    What You See Is What You Get: Observation-Aligned Supervision for Chart-to-Code Generation

    [https://arxiv.org/abs/2607.04726](https://arxiv.org/abs/2607.04726)

    论文揭示了图表到代码生成训练中存在的四类潜在变量与观察图像不匹配问题，并提出观察对齐监督方法，用视觉上可约束的量替换潜在变量作为监督目标。

    

    图表到代码生成通常通过对参考绘图脚本进行监督微调来训练，这隐式地将黄金代码视为完全可观察的目标。然而，许多图表程序包含无法从渲染图像中唯一恢复的潜在变量。我们在五种图表类型中识别出这种潜在变量与观察不匹配问题的四种形式：聚合导致的不匹配，即原始样本被简化为箱线图统计量或直方图分箱统计；归一化导致的不匹配，即饼图中绝对尺度被移除；投影导致的不匹配，即三维信息在二维渲染中丢失；以及水平集导致的不匹配，即标量场只能通过选定的等高线被观察。这些不匹配引入了目标歧义，并要求模型生成图像本身无法支持的信息。我们提出观察对齐监督方法，用视觉上可约束的量来替换潜在变量。

    arXiv:2607.04726v4 Announce Type: replace  Abstract: Chart-to-code generation is commonly trained through supervised fine-tuning on reference plotting scripts, implicitly treating the gold code as a fully observable target. However, many chart programs contain latent variables that cannot be uniquely recovered from the rendered image. We identify this latent-observation mismatch in four forms across five chart types: aggregation-induced mismatch, where raw samples are reduced to box statistics or histogram bin masses; normalization-induced mismatch, where absolute scale is removed in pie charts; projection-induced mismatch, where 3D information is lost through 2D rendering; and level-set-induced mismatch, where a scalar field is observable only through selected contour lines. These mismatches introduce target ambiguity and require models to generate information unsupported by the image. We propose Observation-Aligned Supervision, which replaces latent variables with visually constraine
    
[^131]: 方言能否像语言一样被操控？阿拉伯语大语言模型中的稀疏神经元与分布式方向

    Can Dialects Be Steered Like Languages? Sparse Neurons and Distributed Directions in Arabic LLMs

    [https://arxiv.org/abs/2607.03936](https://arxiv.org/abs/2607.03936)

    该研究发现阿拉伯语大语言模型中的方言特征可通过占MLP维度不足1%的稀疏神经元与分布式激活方向在推理时被探测和引导，无需微调即可将模型输出导向目标方言。

    

    与现代标准阿拉伯语（MSA）相比，方言数据十分稀缺，这导致阿拉伯语大语言模型过度生成标准阿拉伯语，难以进行方言准确度的生成。这引发了一个根本性的可解释性问题：方言特征在模型内部何处以及如何被编码，以及这些表征能否在不进行微调的情况下改进方言生成。我们研究了两种推理时方法，将其作为可解释性探测工具和控制机制。首先，神经元层面的分析识别出编码方言特定特征的稀疏神经元群体，并测试放大或抑制这些神经元能否将模型输出引导至目标方言。其次，向量引导方法提取方言特定的激活方向并在推理时进行注入，其动机源于神经元层面的特征纠缠现象。我们发现这些神经元是真实存在的，但只能提供部分解释。它们占据的MLP维度不足1%，但仅覆盖了5%到21%……

    arXiv:2607.03936v2 Announce Type: replace  Abstract: Dialectal data are scarce relative to Modern Standard Arabic (MSA), causing Arabic LLMs to overproduce MSA and struggle with dialectally accurate generation. This raises a fundamental interpretability question about where and how dialectal features are encoded within model internals and whether these representations can improve dialect generation without fine-tuning. We study two inference-time approaches as interpretability probes and control mechanisms. First, neuron-level analysis identifies sparse populations that encode dialect-specific features and tests whether amplifying or suppressing them steers model outputs toward target dialects. Second, vector steering extracts dialect-specific activation directions and injects them during inference, motivated by feature entanglement at the neuron level. We find that these neurons are real but only partially explanatory. They occupy under 1\% of MLP dimensions but span only 5\% to 21\% 
    
[^132]: KARMA：基于知识图谱的自动推理具体化与对齐

    KARMA: Knowledge graph-based Automated Reasoning Materialization and Alignment

    [https://arxiv.org/abs/2607.03166](https://arxiv.org/abs/2607.03166)

    KARMA 通过在领域知识图谱上枚举模式约束路径生成槽位对齐的对比候选样本，并利用槽位并行对齐（SPA）将偏好监督精准路由至区分性实体槽位，从而解决了基于模板的对比合成中的分辨率不匹配问题。

    

    基于模板的对比合成具有良好的可扩展性，但其候选样本往往仅在少数实体槽位上存在差异，而序列级优化会将监督信号分散到大部分共享的模板上。我们将这一问题形式化为“分辨率不匹配问题”，并提出 KARMA 方法，该方法在领域知识图谱上枚举受模式约束的路径，并将其言语化为槽位对齐的对比候选样本。随后，槽位并行对齐应用解耦的槽位级目标函数，将偏好监督精准引导至具有区分性的实体槽位，其中槽位感知的掩码注意力可作为打包评估的可选实现。在生物医学、计算机科学和化学基准测试中，KARMA 优于基础 LLM 和相同数据下的 SFT 基线，并与序列级和词元级偏好方法相比表现更优。

    arXiv:2607.03166v2 Announce Type: replace-cross  Abstract: Template-based contrastive synthesis is scalable, but its candidates often differ only in a few entity-slots while sequence-level optimization spreads supervision over mostly shared templates. We formalize this as the Resolution Mismatch Problem and propose KARMA, which enumerates schema-constrained paths over domain knowledge graphs and verbalizes them into slot-aligned contrastive candidates. Slot-Parallel Alignment (SPA) then applies a decoupled slot-level objective to route preference supervision to discriminative entity-slots, with slot-aware masked attention serving as an optional packed-evaluation implementation. Across biomedical, computer-science, and chemistry benchmarks, KARMA outperforms base LLM and same-data SFT baselines, and compares favorably with sequence- and token-level preference methods.
    
[^133]: 超越编译：评估忠实的自然语言到Lean语句形式化

    Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization

    [https://arxiv.org/abs/2606.31002](https://arxiv.org/abs/2606.31002)

    该论文提出将Lean编译与GPT-5.2和Gemini-2.5-Pro严格语义共识相结合的自动形式化评估标准，发现编译通过率会高估语义忠实度达3.0至29.0个百分点，且该标准与人类多数判断的一致率达89.7%。

    

    Lean能够验证生成的声明类型正确，但无法验证它表达了用户真正想要的语句。我们针对没有规范Lean目标的自动形式化研究两个问题：LLM评判者能否为人类语义审查提供可用的代理，以及编译在不同系统之间会在多大程度上高估忠实度。我们的评估标准将Lean编译与GPT-5.2和Gemini-2.5-Pro之间的严格语义共识相结合。在一个独立审计的随机样本上，该标准与人类多数意见在89.7%的案例中保持一致（Wilson 95%置信区间：82.1–94.3%）。在400个研究生水平语句上评估的八个系统中，每个系统都存在非零的编译-忠实度差距，其观测幅度介于3.0到29.0个百分点之间。完整的GPT-5.2工具增强代理表现出最大的差距，编译率达89.5%，而满足语义标准的仅为60.5%。人工审查、一个独立的第三族评判者以及BEq形式化交叉……

    arXiv:2606.31002v2 Announce Type: replace  Abstract: Lean verifies that a generated declaration is well typed, but not that it expresses the statement a user intended. We study two questions for autoformalization without canonical Lean targets: whether LLM judges can provide a usable proxy for human semantic review, and how much compilation overstates faithfulness across systems. Our criterion combines Lean compilation with strict semantic consensus between GPT-5.2 and Gemini-2.5-Pro. On an independently audited random sample, it agrees with human majority on 89.7\% of cases (Wilson 95\% CI: 82.1--94.3\%). Across eight systems evaluated on 400 graduate-level statements, every system has a nonzero compile--faithfulness gap, whose observed magnitude ranges from 3.0 to 29.0 percentage points. The full GPT-5.2 tool-augmented agent shows the largest gap, compiling 89.5\% while satisfying the semantic criterion on 60.5\%. Human review, an independent third-family judge, and a BEq formal cros
    
[^134]: 构造即忠实：面向多文档摘要的声明锚定归因方法

    Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization

    [https://arxiv.org/abs/2606.23989](https://arxiv.org/abs/2606.23989)

    提出CAMS框架，将声明级归因嵌入“提取—选择—改写”流程，使多文档摘要中的每句话都能锚定到经过验证、可溯源的源文本片段，从而在构造层面保证摘要的忠实性。

    

    端到端大语言模型（LLM）能够生成流畅的多文档摘要，但仍容易产生幻觉，且其提供的归因通常较为粗糙（仅指向整篇文档或段落）并属于事后生成，导致每条摘要陈述都难以验证。我们重新审视模块化的“提取—选择—改写”范式，并将其中间表示重新构建为归因的基本单元。我们提出了CAMS（Claim-Anchored Multi-document Summarization，声明锚定多文档摘要）框架，该框架：(i) 从每个源文档中提取带有词元级溯源信息的原子声明；(ii) 跨文档聚类等价声明，同时标记源间冲突；(iii) 选择一个兼顾支持度与显著性的子集；(iv) 将所选内容改写为摘要，其中每个句子都锚定到一个经过支持性检验的声明，该声明可回链至一个或多个源文本片段。由于内容在生成之前就已完成定位，整个流程从构造上即是面向归因的。

    arXiv:2606.23989v3 Announce Type: replace-cross  Abstract: End-to-end large language models (LLMs) produce fluent multi-document summaries but remain prone to hallucination, and the attributions they offer are typically coarse (whole documents or passages) and generated post hoc, leaving each summary statement hard to verify. We revisit the modular Extract--Select--Rewrite paradigm and recast its intermediate representation as the unit of attribution. We present CAMS, a Claim-Anchored Multi-document Summarization framework that (i) extracts atomic claims with token-level provenance from every source document, (ii) clusters equivalent claims across documents while flagging inter-source conflicts, (iii) selects a support-aware and salient subset, and (iv) rewrites the selection into a summary in which every sentence is anchored to a support-checked claim that links back to one or more source spans. Because content is localized before it is realized, the pipeline is attribution-oriented b
    
[^135]: 打破似然陷阱：用于大语言模型解码的方差校准调制

    Breaking the Likelihood Trap: Variance-Calibrated Modulation for Large Language Model Decoding

    [https://arxiv.org/abs/2606.22511](https://arxiv.org/abs/2606.22511)

    提出一种无需训练的解码前干预方法VCM，通过基于PMI的上下文探照灯和自适应自我去偏两种动态机制在截断前重塑概率分布，解决大语言模型生成中的重复退化和词汇贫乏问题。

    

    在开放式文本生成中，大语言模型（LLM）经常陷入“似然陷阱”，其特征是重复退化和词汇贫乏，导致机器生成的文本与人类撰写的文本之间存在差异。虽然事后尾部截断方法（如Top-p、Min-p）避免了从不可靠的尾部进行采样，但由于从未经校准的头部过度采样，可能使生成结果与人类的词汇偏好不一致；而固定的标量重复惩罚则忽略了logit分布的尺度在不同推理步骤间的变化，可能会破坏语义连贯性。为了解决这两个缺陷，我们提出了方差校准调制（VCM），这是一种无需训练的解码前干预方法。VCM通过两种动态机制在截断之前直接重塑概率分布：（1）基于PMI的上下文探照灯，能够自然地抑制全局停用词并提升由上下文唤起的词元；（2）自适应自我去偏机制……

    arXiv:2606.22511v3 Announce Type: replace  Abstract: In open-ended generation, LLMs frequently fall into the "likelihood trap", characterized by repetitive degeneration and vocabulary dullness, resulting in a discrepancy between machine-generated and human-written text. While post-hoc tail truncation (e.g., Top-p, Min-p) avoids sampling from the unreliable tail, it can misalign generation with human lexical preferences by over-sampling from the uncalibrated head; fixed scalar repetition penalties, in turn, ignore how the scale of the logit distribution varies across inference steps, which can disrupt semantic coherence. To address both shortcomings, we propose Variance-Calibrated Modulation (VCM), a training-free pre-decoding intervention. VCM directly reshapes the probability distribution prior to truncation via two dynamic mechanisms: (1) Contextual Searchlight via PMI, which naturally suppresses global stopwords and elevates context-evoked tokens, and (2) Adaptive Self-Debiasing, wh
    
[^136]: Chehre：一个以表情符号为提示的数据集，用于探索视频语言模型中的感知灵活性

    Chehre: An Emoji-Prompted Dataset to Explore Perceptual Flexibility in Video Language Models

    [https://arxiv.org/abs/2606.21657](https://arxiv.org/abs/2606.21657)

    提出了Chehre数据集，通过203名参与者录制40种表情符号并将其面部动作迁移到合成面部以保护隐私，收集了由1,242名标注者标注的2,111个视频，并据此定义了“分布式表情识别”这一新任务，以测试视频语言模型能否复现人类对面部表情感知的变异性。

    

    人们对同一面部表情的感知方式是否相同？我们是否应该期望视觉模型在感知面部表情时具有灵活性？面部表情是人类互动中使用的非语言社交信号，但现有的面部表情识别数据集通常只关注每个样本的单一确定性标注。我们推出了Chehre，这是一个以表情符号为提示的视频数据集，包含广泛的动态面部表情，用于探索感知变异性。在Chehre中，203名参与者被要求表达并录制40种面部表情符号。随后，他们的面部动作被迁移到合成面部上以保护隐私。由另一个独立群体对视频进行标注，最终得到2,111个视频，由1,242名感知者标注，每个视频约有30名标注者。Chehre使我们能够定义一个新任务：“分布式表情识别”，用于测试模型能否再现标注者响应中观察到的变异性。

    arXiv:2606.21657v2 Announce Type: replace-cross  Abstract: Do people perceive the same facial expression in the same way? Should we expect vision models to be flexible in how they perceive facial expressions? Facial expressions are nonverbal social signals used in human interaction, but facial expression recognition datasets often focus on a single deterministic annotation per sample. We introduce Chehre, an emoji-prompted video dataset with a wide range of dynamic facial expressions for exploring perceptual variation. In Chehre, 203 participants were prompted to express and record 40 facial emojis. Later, their facial motions were transferred onto synthetic faces to preserve privacy. A separate group annotated the videos, resulting in 2,111 videos annotated by 1,242 perceivers, with ~30 annotators per video. Chehre enables us to define a new task: "distributional expression recognition", which tests whether a model can reproduce the variation observed across annotator responses. We te
    
[^137]: 学会不遗忘什么：从几千字节学习中获得的长时程智能体记忆

    Learning What Not to Forget: Long-Horizon Agent Memory from a Few Kilobytes of Learning

    [https://arxiv.org/abs/2606.20954](https://arxiv.org/abs/2606.20954)

    LRE是一种千字节级、仅用CPU、无需语言模型的学习型淘汰评分器，通过逐字提取保留任务关键历史信息，在智能体任务上以极低成本恢复保留完整历史93%的准确率，并将最坏情况峰值提示削减52%。

    

    长时间运行的语言模型系统会积累超出上下文窗口的交互历史，因此必须持续进行淘汰。当淘汰策略丢弃了任务关键细节时——例如登录时发放的访问令牌或下一次调用所需的路径——操作就会失败。我们提出了LRE（Learned Relevance Eviction，学习相关性淘汰），这是一个千字节规模、仅使用CPU、无需语言模型的评分器，它通过学习判断历史中的哪些单元对任务至关重要，并通过逐字提取的方式保留它们。在匹配预算的对比实验中，没有任何基线方法在准确率-成本平面上全面优于LRE。在智能体任务上，LRE恢复了保留完整历史93%的总体准确率（41.1 vs. 44.0），并在最简单的任务上超出其27%，同时无需任何压缩器调用，并将最坏情况下的峰值提示长度削减了52%。一项受控研究轨迹显示，LRE能够完成其他方法陷入循环的任务，其中一个任务比保留完整历史少用37%的调用次数即可完成。

    arXiv:2606.20954v2 Announce Type: replace-cross  Abstract: Long-running language-model systems accumulate interaction history that outgrows the context window, so they must continually evict. When an eviction policy drops a task-critical detail, for example an access token issued at login or a path the next call needs, the action fails. We present LRE (Learned Relevance Eviction), a kilobyte-scale, CPU-only, language-model-free scorer that learns which units of history are task-critical and keeps them by verbatim extraction. Under a matched-budget comparison, in our experiment, no baseline dominates LRE on the accuracy-cost plane. On agents, LRE recovers 93% of the aggregate accuracy of keeping the entire history (41.1 vs. 44.0) and exceeds it by 27% on the simplest tasks, while requiring zero compressor calls and cutting the worst-case peak prompt by 52%. A controlled study trace shows LRE completes tasks where the others loop, finishing one such task in 37% fewer calls than keeping e
    
[^138]: 大语言吉布斯的结构化推断

    Structured Inference with Large Language Gibbs

    [https://arxiv.org/abs/2606.19264](https://arxiv.org/abs/2606.19264)

    提出了Large Language Gibbs方法，将LLM的条件分布作为MCMC的转移算子，通过在其他变量条件下迭代重采样单个变量来实现结构化概率推断，从而避免了自回归生成中的顺序依赖偏差。

    

    大语言模型（LLM）中编码的知识可以作为对描述复杂世界的变量进行结构化推理的基础，但以概率上连贯的方式获取这些知识是一个困难的推断问题。我们提出了大语言吉布斯，这是一种结构化概率推断方案，它将LLM的条件分布用作转移算子。我们不是通过单次自回归生成来采样结构化对象，而是利用LLM的下一个词元条件分布，在其他变量的条件下迭代地重采样各个变量。这种方法避免了顺序依赖的偏差，并产生一个反映所有局部条件之间折中的平稳分布。我们将该方法应用于从合成分布中采样、一致性推理任务以及贝叶斯结构学习。结果表明，在MCMC中使用LLM条件分布是一种实用的

    arXiv:2606.19264v2 Announce Type: replace-cross  Abstract: The knowledge encoded in large language models (LLMs) can serve as a substrate for structured reasoning over variables describing a complex world, but accessing this knowledge in a probabilistically coherent manner poses a difficult inference problem. We propose Large Language Gibbs, a scheme for structured probabilistic inference that uses conditional distributions of an LLM as transition operators. Rather than sampling structured objects through single-pass autoregressive generation, we iteratively resample individual variables conditioned on others using an LLM's next-token conditionals. This approach avoids order-dependent biases and produces a stationary distribution that reflects a compromise between all local conditionals. We apply this approach to sampling from synthetic distributions, consistent reasoning tasks, and Bayesian structure learning. The results suggest that the use of LLM conditionals in MCMC is a practical
    
[^139]: LLMZero：通过大语言模型智能体发现强化学习后训练的自适应训练策略

    LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents

    [https://arxiv.org/abs/2606.18388](https://arxiv.org/abs/2606.18388)

    LLMZero利用大语言模型智能体结合树搜索，通过在每个检查点诊断训练状态来自适应地优化RL后训练的多参数调度策略，在四个GRPO任务上比基础模型提升9%-140%、比网格搜索提升6%-15%，并揭示了容量参数单调累积、正则化参数震荡变化的训练规律。

    

    强化学习（RL）后训练策略依赖于数据集，并呈现出一个反复出现的经验规律：容量参数在各阶段单调累积，而正则化参数则主要随着训练动态的变化而震荡。这一区别凸显了固定训练调度的一个潜在缺陷：由于强迫所有参数沿着僵化的路径变化，固定调度无法捕捉正则化所必须追踪的动态探索-利用权衡。我们通过LLMZero揭示了这一点——LLMZero是一个通过树搜索优化训练轨迹的智能体系统，它在每个检查点诊断训练中的病理现象，并提出协调的多参数转换方案。在四个多样化的GRPO任务中，LLMZero发现的策略相比基础模型提升了9%至140%，相比网格搜索相对提升了6%至15%，在相同计算预算下，其表现始终优于随机搜索和基于技能的智能体。

    arXiv:2606.18388v2 Announce Type: replace-cross  Abstract: RL post-training strategies are dataset-dependent and reveal a recurring empirical pattern: capacity parameters accumulate monotonically across stages, while regularization parameters predominantly oscillate in response to shifting training dynamics. This distinction highlights a potential flaw in fixed training schedules: by forcing all parameters along rigid paths, they fail to capture the dynamic exploration-exploitation tradeoffs that regularization must track. We uncover this through LLMZero, an agentic system that optimizes training trajectories via tree search by diagnosing pathologies at each checkpoint and proposing coordinated multi-parameter transitions. Across four diverse GRPO tasks, LLMZero discovers strategies that improve over the base model by 9% to 140% and over grid search by 6% to 15% (relative), consistently outperforming random search and a skill-based agent under a matched compute budget. The capacity--re
    
[^140]: 不确定性并非临床视觉问答的安全网，但它能否预判模型失效？

    Uncertainty Is Not a Safety Net for Clinical VQA, but Can It Anticipate Model Failure?

    [https://arxiv.org/abs/2606.16583](https://arxiv.org/abs/2606.16583)

    该研究通过在12个临床视觉语言模型上基准测试8种不确定性估计方法，发现不确定性估计的质量会随模型准确率下降而退化、且在正确答案被隐藏时无法发出警示，但未扰动输入上的不确定性本身携带诊断信息，能够预判哪些预测将在模型失效时崩溃。

    

    临床视觉语言模型的安全部署需要可靠的不确定性估计，即一种指示何时应当信任预测结果、或何时应将决策升级给临床医生的信号。我们检验了当前的不确定性估计方法是否真正能够提供这种信号。通过在临床视觉问答任务上对12个视觉语言模型中的8种不确定性估计方法进行基准测试，我们发现不确定性估计的质量并非该方法本身的内在属性：它跟随模型准确率而变化，恰好在模型性能最弱、也最需要可靠性保证的地方发生退化。当我们通过在多选题答案中隐藏正确选项（NOTA扰动）对模型进行压力测试时，准确率急剧下降，而不确定性却几乎没有任何变化，导致模型出现系统性的校准失准。然而我们发现，未受扰动输入上的不确定性能够可靠地预判哪些预测会在NOTA扰动下崩溃，这表明当前视觉语言模型中的不确定性携带关于模型失效的诊断信息。

    arXiv:2606.16583v2 Announce Type: replace  Abstract: Safe deployment of clinical vision-language models (VLMs) requires reliable uncertainty estimation (UE): a signal indicating when predictions should be trusted or escalated to a clinician. We test whether current UE methods actually deliver this signal. Benchmarking 8 methods across 12 VLMs on clinical visual question-answering (VQA), we find that UE quality is not an intrinsic property of the UE method: it tracks model accuracy, degrading precisely where the model performance is weakest, and therefore where reliability is most needed. When we stress-test models by hiding the correct option among the multiple-choice answers (NOTA perturbations), accuracy collapses while uncertainty barely changes, leaving models systematically miscalibrated. Yet, we find that uncertainty on the unperturbed input reliably anticipates which predictions will collapse under NOTA, indicating that UE in current VLMs carries diagnostic information about mod
    
[^141]: ParaBridge：在语音语言模型中架起副语言感知与对话行为之间的桥梁

    ParaBridge: Bridging Paralinguistic Perception and Dialogue Behavior in Speech Language Models

    [https://arxiv.org/abs/2606.10581](https://arxiv.org/abs/2606.10581)

    ParaBridge提出一种在线策略自蒸馏方法，将推理时脆弱的副语言指令支架转化为语音语言模型中稳定的行为，从而弥合副语言感知与对话行为之间的差距。

    

    语音所承载的信息远不止文字本身：孩子的声音、恐惧的语气或嘈杂的背景音，都应该使一个足够胜任的语音对话助手给出不同的回复。当前的语音语言模型（SLM）虽然能够识别这类副语言线索，但在开放式对话中却常常将其忽略。我们观察到，在推理阶段使用一种简单的副语言指令支架能够缩小这种感知-行为差距，这表明相关线索其实已经潜在地存在于模型之中。然而，这种支架在多轮对话情境和相互竞争的指令下仍然十分脆弱。因此，我们提出了ParaBridge，一种在线策略（on-policy）自蒸馏方法，将脆弱的推理时支架转化为稳定的模型内在行为。在训练过程中，支架仅作为临时的特权视图：无支架的模型自行生成回复，而带支架的视图则为其提供密集的、覆盖全词表的下一词元监督信号。

    arXiv:2606.10581v2 Announce Type: replace  Abstract: Speech carries more information than just words: a child's voice, a fearful tone, or a noisy background should all lead a sufficiently competent spoken-dialogue assistant to different replies. Current Speech Language Models (SLMs) can recognize such paralinguistic cues but often ignore them in open-ended dialogue. We observe that a simple paralinguistic instruction scaffold at the inference stage narrows this perception-behavior gap, suggesting that the relevant cues are already latent in the model. Such scaffolds, however, remain brittle under multi-turn context and competing instructions. Therefore, we propose \textbf{ParaBridge}, an on-policy self-distillation method that turns a brittle inference-time scaffold into stable model behavior. During training, the scaffold serves only as a temporary privileged view; the scaffold-free model rolls out its own response, while the scaffolded view supplies dense, full-vocabulary next-token 
    
[^142]: 超越准确性：社区对机器翻译的看法

    Beyond Accuracy: Community Perspectives on Machine Translation

    [https://arxiv.org/abs/2606.09655](https://arxiv.org/abs/2606.09655)

    该论文首次大规模分析了四个利益相关社区在社交媒体上发布的79,286条关于机器翻译的帖子和评论，揭示了技术进步与真实用户需求之间的差距，强调倾听用户社区声音以引导研究方向的必要性。

    

    尽管机器翻译（MT）取得了显著进展，非AI社区对机器翻译系统的担忧却日益增加，这表明技术进步与现实世界用户需求之间存在明显差距。例如，当NLP研究人员关注基准测试性能时，最终用户则关心伦理问题、信任、可靠性、成本等。我们认为，倾听各类用户社区的声音至关重要，这样才能使研究工作指向社区所关心的问题。为此，我们首次提出了一项大规模分析，研究四个利益相关社区（AI开发者、专业译员、语言学习者和语言服务提供商）在社交媒体上发布的关于机器翻译技术的内容。为此，我们构建了一个数据集，包含2019年至2025年间来自Reddit、Facebook、Bluesky和Mastodon的79,286条帖子和评论，并分析了这些社区之间在哪些方面存在分歧。

    arXiv:2606.09655v2 Announce Type: replace  Abstract: Despite remarkable progress in machine translation (MT), non-AI communities have raised growing concerns about MT systems, suggesting a noticeable gap between technical advancement and the needs of real-world users. For instance, while NLP researchers focus on benchmark performance, end users care about ethical concerns, trust, reliability, costs, and more. We argue that listening to various user communities is essential so that research efforts would be directed towards the problems that the communities care about. To this end, we present a large-scale analysis, for the first time, that investigates what four stakeholder communities (AI developers, professional translators, language learners, and language service providers) post about MT technology on social media. To do so, we construct a dataset of 79,286 posts and comments from Reddit, Facebook, Bluesky, and Mastodon from 2019 to 2025, and analyse where these communities disagree
    
[^143]: TEVI：通过稀疏自编码器实现文本条件化的视觉表征编辑，以改进视觉-语言对齐

    TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment

    [https://arxiv.org/abs/2606.07451](https://arxiv.org/abs/2606.07451)

    TEVI框架利用稀疏自编码器解耦图像嵌入，并通过文本条件化的掩码模块只保留与文本描述相符的信息、剔除多余内容，从而改善CLIP等视觉-语言模型中图像-文本嵌入的对齐问题。

    

    像CLIP这样的视觉-语言模型由于其共享的图像-文本嵌入空间而在多种任务中非常有用。尽管如此，图像和文本嵌入之间的对齐往往不佳，从而影响下游任务的性能。近期的研究假设这可以归因于信息不平衡问题：图像所包含的信息多于其文本描述所涵盖的内容。在这项工作中，我们提出了TEVI，一个利用文本描述作为信号来决定从图像嵌入中保留哪些信息的框架。具体而言，我们使用稀疏自编码器来解耦图像嵌入，并训练一个掩码模块，根据给定的文本描述选择性地重建嵌入。在使用合成文本描述的受控实验设置中，我们展示了TEVI能够有效地保留文本描述所涉及的属性，同时丢弃其他属性。我们发现这一方法可以扩展到在自然图像上训练的CLIP模型，其中TEVI学会了进行有意义的掩码操作，并支持基于内容的检索。

    arXiv:2606.07451v2 Announce Type: replace-cross  Abstract: Vision-language models such as CLIP are highly useful for diverse tasks due to their shared image-text embedding space. Despite this, the image and text embeddings are often poorly aligned, affecting downstream performance. Recent work has hypothesized that this can be attributed to an information imbalance: images contain more information than their captions describe. In this work, we propose TEVI, a framework that uses captions as a signal for what to retain from image embeddings. Specifically, we use sparse autoencoders to disentangle image embeddings and train a masking module to selectively reconstruct the embedding based on a given caption. In a controlled setup with synthetic captions, we show that TEVI is effective at preserving caption-described attributes while discarding others. We find that this extends to CLIP models trained on natural images, where TEVI learns to mask meaningfully and allows retrieval based on con
    
[^144]: SV-Detect：基于转向向量的AI生成文本检测

    SV-Detect: AI-generated Text Detection with Steering Vectors

    [https://arxiv.org/abs/2606.07313](https://arxiv.org/abs/2606.07313)

    本文提出SV-Detect，一种利用从冻结语言模型隐藏表示中提取的转向向量构建逐层投影特征来检测AI生成文本的方法，在跨领域、跨源模型及编辑攻击等分布偏移场景下均实现了强大的检测性能。

    

    在分布偏移（如跨领域、跨源模型和编辑攻击的迁移）情况下，检测AI生成的文本尤其困难。我们提出了一种基于转向向量的AI生成文本检测器，该转向向量从冻结语言模型的隐藏表示中提取。在每一层，我们构建一个能够区分人类撰写文本与AI生成文本的方向，并通过每个输入与这些方向的逐层对齐程度来表示该输入。在这些投影特征上训练的轻量级分类器产生最终的检测分数。我们的方法在分布内和分布偏移下均取得了强大的性能，包括跨领域、跨源模型，以及诸如润色和重写等机器编辑变换。解释性分析表明，学习到的方向与可识别的风格线索相一致，同时捕捉到了超越表面特征的大量额外信号。这些结果使AI...

    arXiv:2606.07313v2 Announce Type: replace-cross  Abstract: Detecting AI-generated text is especially difficult under distribution shift, such as transfer across domains, source models, and editing attacks. We propose an AI-generated text detector based on steering vectors extracted from the hidden representations of a frozen language model. At each layer, we construct a direction that separates human-written from AI-generated text, and represent each input by its layer-wise alignment with these directions. A lightweight classifier trained on these projection features yields the final detection score. Our method achieves strong performance both in-distribution and under distribution shift, including across domains, source models, and machine-editing transformations such as polishing and rewriting. Interpretation analyses show that the learned directions align with recognizable stylistic cues while capturing substantial additional signal beyond surface features. These results position AI
    
[^145]: EDIT：面向规则忠实型LLM评分的证据诊断干预训练

    EDIT: Evidence-Diagnosed Intervention Training for Rule-Faithful LLM Grading

    [https://arxiv.org/abs/2606.06350](https://arxiv.org/abs/2606.06350)

    EDIT是一个两阶段训练框架，利用模型内部的信念信号定位并修正LLM评分器中有问题的推理步骤，并通过信念引导的强化学习奖励塑形，使LLM的评分更忠实于评分规则和学生答案证据。

    

    可靠的量表评分不仅需要准确的分数预测，每个评判都必须扎根于评分标准和来自学生答案的证据。现有的信用分配与干预方法主要针对数学推理等自包含推理任务设计，难以适用于这一场景，因为它们既无法定位评分推理在何处出错，也无法刻画模型对最终分数的信念在推理过程中如何变化。我们提出证据诊断干预训练，这是一个用于训练更忠实于评分量表的LLM评分器的两阶段框架。首先，EDIT-SFT利用模型内部信号——对最终分数的后验信念和输入扎根得分——来定位有问题的推理步骤，并在评分量表清单的辅助下仅对这些局部步骤进行修订。其次，EDIT-RL通过信念引导的奖励塑形对评分器进行校准，惩罚较大的有害信念漂移，同时仍允许有益的信念更新。

    arXiv:2606.06350v2 Announce Type: replace  Abstract: Reliable rubric grading requires more than accurate score prediction. Each judgement must be grounded in the mark scheme and evidence from the student answer. Existing credit-assignment and intervention methods, primarily designed for self-contained reasoning tasks such as mathematics reasoning, struggle in this setting because they do not identify where grading reasoning goes wrong or how the model's belief about the final mark changes during reasoning. We propose Evidence-Diagnosed Intervention Training (EDIT), a two-phase framework for training more rubric-faithful LLM graders. First, EDIT-SFT locates problematic reasoning steps using internal model signals: posterior belief over the final mark and input-grounding scores. It then revises only these local steps with help from a rubric checklist. Second, EDIT-RL calibrates the grader with belief-guided reward shaping, penalising large harmful belief drifts while still allowing helpf
    
[^146]: ArcANE：角色扮演语言代理能否在恰当的时机保持角色设定？

    ArcANE: Do Role-Playing Language Agents Stay in Character at the Right Time?

    [https://arxiv.org/abs/2606.05553](https://arxiv.org/abs/2606.05553)

    本文提出ArcANE基准，通过构建刻画角色价值观、动机和关系随故事演变的“情节弧”，评估角色扮演语言代理能否在叙事的不同阶段准确呈现角色的动态发展，弥补了现有基准将角色视为固定设定的不足。

    

    角色扮演语言代理（RPLA）在娱乐、陪伴、互动叙事和教育等应用中模拟特定的角色和人物设定。忠实的角色扮演不仅仅是产生合理的、符合角色的回复：随着叙事中角色的价值观和行为发生变化，RPLA 应当反映出角色在相应阶段的状态。然而，现有的基准测试大多将角色视为固定的人物设定，或仅测试角色在叙事某一时间点上所知道的内容。我们提出了 ArcANE（Arc-Aware Narrative Evaluation，情节弧感知叙事评估），这是一个用于评估 RPLA 是否能跟随角色在叙事中发展变化的基准。ArcANE 首先构建一个“情节弧”，刻画角色的价值观、动机或关系如何随故事发展而演变。随后，该基准评估 RPLA 的回复与情节弧相应阶段的契合程度，涵盖三种不同的场景类型：来自……

    arXiv:2606.05553v2 Announce Type: replace-cross  Abstract: Role-playing language agents (RPLAs) simulate specific characters and personas across applications such as entertainment, companionship, interactive storytelling, and education. Faithful role-play requires more than producing plausible, in-character responses: as a character's values and behavior change over a narrative, an RPLA should reflect the character's state at the relevant stage. However, existing benchmarks largely treat characters as fixed personas or test only what they know at a given point in the narrative. We introduce ArcANE (Arc-Aware Narrative Evaluation), a benchmark for evaluating whether an RPLA follows a character's development across a narrative. ArcANE first builds an Arc that maps how a character's values, motivations, or relationships change over the story. The benchmark then scores how well an RPLA's responses fit the corresponding stages of the Arc, covering three distinct scenario types: scenes from 
    
[^147]: 稀疏MoE语言模型中事实回忆的专家感知因果追踪

    Expert-Aware Causal Tracing of Factual Recall in Sparse MoE Language Models

    [https://arxiv.org/abs/2606.03780](https://arxiv.org/abs/2606.03780)

    该研究将激活修补方法细化到专家层面，首次证明在稀疏MoE语言模型（如Qwen3-30B-A3B-Base）中，事实回忆的恢复可定位于单个路由专家（L44E069），但这种专家级定位在不同模型间并不一致。

    

    激活修补方法可以识别出这样的混合专家（MoE）模块：其干净输出能够恢复被破坏的事实性预测。然而，由于模块输出结合了多个被路由专家的贡献，模块级别的恢复并不能确定这种恢复是定位于某个单个专家，还是依赖于整个被路由的专家集合。我们在单token的COUNTERFACT对比样本上研究这一问题：先破坏主语token的嵌入，恢复干净的模块输出，然后在固定路由条件下恢复“干净减去噪声”的专家更新。在Qwen3-30B-A3B-Base中，发现性扫描选定了第44层，留出集分析识别出L44E069作为一个反复出现的路由贡献者，其相对同层活跃专家具有正的特异性。该专家的效应与事实匹配，并能提升真实token的概率和排名，从而解释了部分层级别的恢复效果。而在Mixtral-8x7B-v0.1中，被选中的反复出现的单一专家并不具备特异性；匹配规模的对照组……

    arXiv:2606.03780v2 Announce Type: replace  Abstract: Activation patching can identify a mixture-of-experts (MoE) block whose clean output restores a corrupted factual prediction. However, because the block output combines contributions from multiple routed experts, block-level rescue does not establish whether the recovery localizes to an individual expert or depends on the routed expert set. We study this question on single-token COUNTERFACT contrasts by corrupting subject-token embeddings, restoring clean block outputs, and then restoring clean-minus-noised expert updates under fixed routing. In Qwen3-30B-A3B-Base, a discovery sweep selects layer 44, and held-out analysis identifies L44E069 as a recurrent routed contributor with positive specificity over same-layer active experts. Its effect is fact-matched and improves true-token probability and rank, which explains part of the layer rescue. In Mixtral-8x7B-v0.1, the selected recurrent singleton is not specific; matched-size control
    
[^148]: 大型人工智能模型在牙科医疗中的应用：从通用系统到领域专用基础模型

    Large AI Models in Dental Healthcare: From General-Purpose Systems to Domain-Specific Foundation Models

    [https://arxiv.org/abs/2606.02914](https://arxiv.org/abs/2606.02914)

    本文首次提出按架构范式和牙科专业化程度划分的二维分类框架，系统综述了97项研究，统一考察了语言生成模型、视觉基础模型和牙科专用基础模型三类大规模AI模型在牙科医疗中的关系与共同局限。

    

    背景：口腔疾病影响全球近35亿人，然而大规模AI模型在牙科领域的比较临床潜力仍知之甚少。目前出现了三种不同的模型类别：语言生成模型、判别式视觉基础模型和牙科专用基础模型，尚无统一的综述来考察它们之间的关系及其共同局限性。方法：遵循PRISMA-ScR指南，我们系统检索了四个数据库（PubMed、Google Scholar、Scopus、arXiv），并由两名评审员独立筛选。应用纳入/排除标准后，共纳入97项研究（2020-2026年）。我们提出了一个二维分类框架，按架构范式和牙科专业化程度对模型进行分类组织。结果：语言生成模型在文本类任务中表现出色（临床推理、执业考试、患者沟通），但表现不一致……

    arXiv:2606.02914v3 Announce Type: replace  Abstract: Background: Oral diseases affect nearly 3.5 billion people worldwide, yet the comparative clinical potential of large-scale AI models in dentistry remains poorly understood. Three distinct model categories have emerged: language-generative models, discriminative vision foundation models, and dental-specific foundation models, with no unified review examining their relationships and collective limitations.   Methods: Following PRISMA-ScR guidelines, we systematically searched four databases (PubMed, Google Scholar, Scopus, arXiv), screened independently by two reviewers. After applying inclusion/exclusion criteria, 97 studies (2020-2026) were included. We propose a two-dimensional classification framework organizing models by architectural paradigm and dental specialization degree.   Results: Language-generative models excel at text-based tasks (clinical reasoning, licensing exams, patient communication) but show inconsistent performa
    
[^149]: 修复 FOLIO 和 MALLS：经验证的标注与借助大语言模型聚焦人工重新标注的框架

    Fixing FOLIO and MALLS: Verified Annotations and an LLM-assisted Framework to Focus Human Relabeling

    [https://arxiv.org/abs/2606.02837](https://arxiv.org/abs/2606.02837)

    本研究系统性审查发现 FOLIO 和 MALLS 基准中约 42% 的一阶逻辑标注存在错误，并发布修正后的真值标注以及一个 LLM 辅助框架来聚焦人工重新标注工作。

    

    从自然语言到一阶逻辑（NL-to-FOL）的准确转换是神经符号 AI 系统和自然语言推理（NLI）的基础，因此 NL-to-FOL 基准数据集的质量至关重要——然而这些数据集从未经过严格审查。我们的第一个贡献是对 FOLIO 验证集和 MALLS 测试实例的一个子集进行了系统性的人工检查，发现分别约有 42.5% 和 42% 的条目包含错误的一阶逻辑形式化（即真值标签），此外还存在歧义自然语言句子（分别为 17.8% 和 51%）以及 FOLIO 中错误的 NLI 标签（8.4%）。我们的第二个贡献是为这些数据集开发并发布了修正后的真值标注，并证明标注错误会扭曲模型在参考基准任务上的评估：在测试三个最先进的大语言模型（Gemma 4 31B-it、Qwen3-30B-A3B 和 GPT-4o-mini）时……

    arXiv:2606.02837v2 Announce Type: replace-cross  Abstract: Accurate translation from Natural Language to First-Order Logic (NL-to-FOL) underpins neurosymbolic AI systems and Natural Language Inference (NLI), making the quality of NL-to-FOL benchmarks essential---yet these datasets have never been rigorously audited. Our first contribution is to present a systematic human inspection of the validation split of \textsf{FOLIO} and a subset of \textsf{MALLS} test instances, finding that approximately 42.5\% and 42\% of entries, respectively, contain incorrect FOL formalizations (i.e., ground truth labels), with additional rates of ambiguous NL sentences (17.8\% and 51\%) and incorrect NLI labels in \textsf{FOLIO} (8.4\%). Our second contribution is to develop and release corrected ground truths for such datasets, showing that annotation errors distort model evaluation on a reference benchmark task: testing three state-of-the-art LLMs (Gemma~4 31B-it, Qwen3-30B-A3B, and GPT-4o-mini) with the
    
[^150]: CoMAP：面向大语言模型智能体的世界模型与智能体策略共同演化框架

    CoMAP: Co-Evolving World Models and Agent Policies for LLM Agents

    [https://arxiv.org/abs/2606.02372](https://arxiv.org/abs/2606.02372)

    本文提出COMAP框架，通过闭环交互使文本世界模型与LLM智能体策略共同演化——世界模型为候选动作预测未来状态反馈以支持智能体的前瞻性反思，智能体产生的同策略轨迹又通过自蒸馏反哺更新世界模型，从而摆脱了对外部奖励或验证器的依赖。

    

    arXiv:2606.02372v2 公告类型：替换 摘要：为语言智能体配备世界模型，使其能够在执行之前预测环境动态并评估候选动作。然而，现有的文本世界模型通常在训练完成后即被固定，无法适应不断演化的智能体所产生的同策略状态-动作分布。与此同时，智能体改进方法往往依赖外部奖励或验证器，这限制了它们在现实交互环境中的适用性。本文提出了COMAP，一种通过闭环交互使文本世界模型与智能体策略共同演化的新框架。在每个决策步骤中，世界模型为候选动作预测未来状态反馈，智能体则通过估计该反馈的可靠性并据此优化自身动作来进行具有前瞻性的反思。随后，所产生的同策略轨迹被用于通过自蒸馏来更新世界模型，使其能够……（摘要原文在此处截断）

    arXiv:2606.02372v2 Announce Type: replace  Abstract: Equipping language agents with world models enables them to anticipate environment dynamics and evaluate candidate actions before execution. However, existing textual world models are typically fixed after training, preventing them from adapting to the on-policy state-action distributions induced by an evolving agent. Meanwhile, agent-improvement methods often rely on external rewards or verifiers, limiting their applicability in realistic interactive environments. In this paper, we propose COMAP, a novel framework that co-evolves textual world models and agent policies through closed-loop interaction. At each decision step, the world model predicts future state feedback for candidate actions, and the agent performs future-aware reflection by estimating the reliability of this feedback and refining its action accordingly. The resulting on-policy trajectories are then used to update the world model via self-distillation, allowing it t
    
[^151]: 论点坍塌：大语言模型使长篇公共辩论趋于扁平化

    Argument Collapse: LLMs Flatten Long-Form Public Debate

    [https://arxiv.org/abs/2606.01736](https://arxiv.org/abs/2606.01736)

    该论文首次提出并系统量化了“论点坍塌”现象：大语言模型生成的议论文会高度收敛到极少数相同论点（人类论点65.3%独特而模型仅3.4%），即使显式要求多样性也只能恢复约一半的人类论点，揭示了LLM可能使公共辩论同质化、扁平化的风险。

    

    随着大语言模型越来越多地被用于起草面向公众的论辩文章，它们可能通过反复引入相同的、经过润色且貌似合理的论点而使公共辩论趋于扁平化。我们研究了“论点坍塌”现象，即不同大语言模型生成的文章倾向于收敛到较小的一组主要论点、子论点和段落级结构上。我们比较了来自195场《纽约时报》(NYT)辩论的1,039条人类回复、来自61个篇幅更长的《波士顿评论》(BR)论坛的448条人类回复，以及23,381篇由大语言模型生成的文章。在NYT语料库中，65.3%的人类主要论点在同一场辩论中是独特的，而大语言模型生成的主要论点中仅有3.4%是独特的。要求大语言模型生成多样化答案虽能增加变化，但一个典型模型只能恢复约一半的人类独特主要论点，且许多新增的变化落在已观察到的人类论点空间之外。坍塌现象同样出现在子论点层面：在具有相同主要论点的文章中，41.0%的

    arXiv:2606.01736v4 Announce Type: replace-cross  Abstract: As LLMs are increasingly used to draft publicfacing arguments, they may flatten public debate by repeatedly introducing the same polished, plausible arguments. We study argument collapse, the tendency of essays generated by different LLMs to converge to a smaller set of main arguments, sub-arguments, and paragraph-level structures. We compare 1,039 human responses from 195 New York Times (NYT) debates, 448 human responses from 61 longer-form Boston Review (BR) forums, and 23,381 LLM-generated essays. In the NYT corpus, 65.3% of human main arguments are unique within a debate, compared to 3.4% of LLM main arguments. Asking LLMs to generate diverse answers adds variation, but a typical model recovers only about half of the distinct human main arguments, with much of the added variation falling outside the observed human argument space. Collapse also appears in sub-arguments, where among essays with the same main argument, 41.0% o
    
[^152]: 关注证据：面向多模态RLVR的证据锚定空间注意力监督

    Attend to Evidence: Evidence-Anchored Spatial Attention Supervision for Multimodal RLVR

    [https://arxiv.org/abs/2605.30912](https://arxiv.org/abs/2605.30912)

    EASE通过将标注证据区域转化为平滑的视觉token目标，在高奖励轨迹上监督回复到图像的空间注意力，为多模态RLVR引入视觉证据过程监督，且推理时无需任何标注。

    

    基于可验证奖励的强化学习（RLVR）通过优化由最终答案得出的结局奖励来提升视觉语言模型（VLM）。然而，这种仅基于结局的奖励无法告诉模型哪些图像区域支撑了答案。对于需要视觉定位的问题，这类奖励无法区分由相关视觉证据支持的回答与由语言先验捷径或幸运猜测产生的回答。我们提出了EASE（证据锚定空间注意力），它为多模态RLVR引入了视觉证据过程监督。EASE将标注的证据区域转换为平滑的视觉token目标，并在RL训练期间利用它引导回复到图像的注意力，但仅作用于高奖励轨迹。这些标注仅作为特权训练标签使用，推理时只需要原始图像和问题。在Qwen2.5-VL-7B、Qwen3-VL-4B和Qwen3-VL-8B上，EASE显著提升了性能。

    arXiv:2605.30912v2 Announce Type: replace-cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) improves vision-language models (VLMs) by optimizing outcome rewards derived from final answers. However, such outcome-only rewards do not tell the model which image regions justify an answer. For questions that require visual grounding, these rewards cannot distinguish responses supported by relevant visual evidence from those produced by language-prior shortcuts or lucky guesses. We introduce EASE (Evidence-Anchored Spatial Attention), which augments multimodal RLVR with visual-evidence process supervision. EASE converts annotated evidence regions into a smoothed visual-token target and uses it to guide response-to-image attention during RL training, but only on high-reward trajectories. The annotations are used solely as privileged training labels, while inference requires only the original image and question. Across Qwen2.5-VL-7B, Qwen3-VL-4B, and Qwen3-VL-8B, EASE raise
    
[^153]: 技能条件门控自蒸馏用于大语言模型推理

    Skill-Conditioned Gated Self-Distillation for LLM Reasoning

    [https://arxiv.org/abs/2605.28791](https://arxiv.org/abs/2605.28791)

    本文提出SGSD，通过技能库和门控机制将自蒸馏从无条件模仿转为教师假设验证，以应对不可靠技能并提升大语言模型推理能力。

    

    摘要：arXiv:2605.28791v2 公告类型：跨版本替换。摘要：在策略自蒸馏（SD）通过利用教师端的特权信息（PI）将稀疏的验证器结果转化为密集的令牌级监督，从而提升大语言模型的推理能力。现有方法通常假设PI是可信的，例如参考答案或成功轨迹。我们提出疑问：PI是否可以从经验衍生的技能库中获得，其中检索到的技能虽然紧凑且可复用，但也可能不相关或具有误导性。我们提出了技能条件门控自蒸馏（SGSD），该方法将基于技能的SD视为教师假设验证而非无条件模仿。SGSD检索技能-错误对，构建多教师池，并让所有技能条件的教师对相同的普通提示学生轨迹进行评分。验证器验证每位教师的极性：支持成功或抑制失败提供正向监督，而相反立场则被反转。一个稳健的门控目标随后将...

    arXiv:2605.28791v2 Announce Type: replace-cross  Abstract: On-policy self-distillation (SD) improves LLM reasoning by using teacher-side privileged information (PI) to turn sparse verifier outcomes into dense token-level supervision. Existing methods usually assume trusted PI, such as reference answers or successful traces. We ask whether PI can instead come from an experience-derived skill bank, where retrieved skills are compact and reusable but may also be irrelevant or misleading. We propose Skill-Conditioned Gated Self-Distillation (SGSD), which formulates skill-based SD as teacher hypothesis validation rather than unconditional imitation. SGSD retrieves skill-mistake pairs, constructs a multi-teacher pool, and lets all skill-conditioned teachers score the same plain-prompt student rollout. The verifier validates each teacher's polarity: supporting a success or suppressing a failure gives positive supervision, while the opposite stance is reversed. A robust gated objective then di
    
[^154]: SuperValid：面向可泛化下游扩展的能力对齐分布外验证

    SuperValid: Capability-Aligned OOD Validation for Generalizable Downstream Scaling

    [https://arxiv.org/abs/2605.28179](https://arxiv.org/abs/2605.28179)

    该论文提出SuperValid框架，通过从能力域内的基准中提炼核心概念并扩展为多样化的知识型文本，合成能力对齐的分布外验证数据，从而在能力层面实现更具泛化性的下游扩展预测。

    

    扩展定律通过将计算量与交叉熵损失相关联来指导大语言模型的训练，近期的工作进一步将其扩展用于预测下游基准性能。然而，先前的方法在泛化性方面存在两个局限：聚焦于基准层面的性能会引入特定场景的伪影，而依赖独立同分布（IID）验证损失在训练分布变化时无法追踪能力的提升。在本工作中，我们认为应当在能力层面研究下游扩展，因为能力层面能够捕捉相关任务间的共享技能因素，同时抽象掉基准特有的噪声。我们提出了SuperValid，该框架通过从能力域内的基准中提炼核心概念，并将其扩展为多样化、知识丰富的文本，从而合成分布外（OOD）且能力对齐的验证数据。大量实验覆盖了划分为6个能力领域的16个基准……

    arXiv:2605.28179v2 Announce Type: replace  Abstract: Scaling laws guide large language model training by relating compute to cross-entropy loss, and recent work further extends them to predict downstream benchmark performance. However, prior approaches face generalization limitations from two aspects: focusing on benchmark-level performance introduces scenario-specific artifacts, while relying on IID validation loss fails to track capability improvements when training distributions vary. In this work, we argue that downstream scaling should be studied at the capability level, which captures shared skill factors across related tasks while abstracting away benchmark-specific noise. We propose SuperValid, a framework that synthesizes OOD (out-of-distribution), capability-aligned validation data by distilling core concepts from benchmarks within a capability domain and expanding them into diverse, knowledge-rich texts. Extensive experiments spanning 16 benchmarks grouped into 6 capability 
    
[^155]: MIRA：一个用于医疗信息响应审计的双语基准测试

    MIRA: A Bilingual Benchmark for Medical Information Response Audit

    [https://arxiv.org/abs/2605.28025](https://arxiv.org/abs/2605.28025)

    该论文提出了首个双语医疗基准MIRA，揭示了大语言模型在应对低健康素养用户提问时会系统性遗漏关键医疗信息、减少后续行动建议的“差异化信息稀释”（DID）这一安全隐患。

    

    现有的大语言模型安全评估忽视了这样一个问题：当用户以不同措辞提出同一问题时，模型的回答是否保留了可比较的医疗信息。为解决这一问题，我们提出了医疗信息响应审计基准，这是一个双语、受控的基准测试，用于评估大语言模型在用户语言、语域和健康素养信号变化下能否提供可比较的医疗信息。MIRA包含由60个经过医学审阅的低风险健康问题构建的4,320个提示词。在五个主流大语言模型上的实验表明，模型能够回答所有医疗问题，但针对低健康素养信号的回答一致地遗漏了更多关键信息，提供了更少的后续具体步骤，对独立判断的支持也更少。我们将这一现象命名为差异化信息稀释。与300个真实世界健康查询的对比为排序效度提供了初步证据。一种知识引导的缓解方法…

    arXiv:2605.28025v2 Announce Type: replace  Abstract: Existing safety evaluations for large language models overlook whether responses preserve comparable medical information across different user phrasings of the same question. To address this, we introduce the Medical Information Response Audit (MIRA), a bilingual, controlled benchmark that assesses whether LLMs provide comparable medical information across user-side language, register, and health literacy signals. MIRA contains 4,320 prompts built from 60 medically reviewed, low-risk health questions. Across five mainstream LLMs, models answered all medical questions, but responses to low health-literacy signals consistently omitted more key information, provided fewer concrete next steps, and offered less support for independent judgment. We term this pattern Differential Information Dilution (DID). A comparison with 300 real-world health queries provides preliminary evidence of rank-order validity. A knowledge-guided mitigation pro
    
[^156]: EmoDistill：面向对抗性谈判中语言模型智能体的离线情感技能蒸馏

    EmoDistill: Offline Emotion Skill Distillation for Language Model Agents in Adversarial Negotiation

    [https://arxiv.org/abs/2605.26785](https://arxiv.org/abs/2605.26785)

    提出EmoDistill离线蒸馏框架，通过IQL情感选择器与LoRA微调的表达策略相结合，将大模型间对抗谈判中的情感技能迁移到小型语言模型智能体，使其能够抵御情感操纵并维护用户谈判目标。

    

    经过训练后优化的大语言模型通常被调整为产生有帮助、礼貌且迁就他人的回应。然而，在对抗性谈判中，这种行为可能成为一种脆弱性：带有情感色彩的语言可能会以与其用户目标相冲突的方式影响智能体的谈判决策。因此，我们提出了EmoDistill，这是一个离线框架，用于将大语言模型之间交互中的情感谈判技能蒸馏到更小的语言模型智能体中。在这里，情感谈判技能是一种状态条件行为，它决定在谈判状态下应调用哪种显式情感，以及如何将该情感转化为有效的谈判话语。EmoDistill分别学习这两个组件：一个隐式Q学习选择器学习在每个谈判状态下应表达哪种情感，而一个经LoRA适配的7B策略通过监督微调和Judge Policy学习情感条件下的表达方式

    arXiv:2605.26785v2 Announce Type: replace-cross  Abstract: Post-trained LLMs are often optimized to produce helpful, polite, and accommodating responses. In adversarial negotiation, however, such behavior can become a vulnerability: emotionally framed language may influence an agent's bargaining decisions in ways that conflict with its user's objectives. We therefore introduce EmoDistill, an offline framework for distilling emotional negotiation skills from LLM-LLM interactions into smaller language-model agents. Here, an emotional negotiation skill is a state-conditioned behavior that determines which explicit emotion to invoke in a bargaining state and how to realize that emotion as an effective negotiation utterance. EmoDistill learns these two components separately: an Implicit Q-Learning (IQL) selector learns which emotion to express in each bargaining state, while a LoRA-adapted 7B policy learns emotion-conditioned expression through Supervised Fine-Tuning (SFT) and Judge Policy 
    
[^157]: 好奇之龄遇上AI时代：大型语言模型中的儿童安全基准测试

    The Age of Curiosity Meets the Age of AI: Benchmarking Child Safety in Large Language Models

    [https://arxiv.org/abs/2605.25510](https://arxiv.org/abs/2605.25510)

    提出了基于发展心理学的儿童安全评估基准KIDBench，用于测试大语言模型对7-11岁儿童提问的安全性，并发现提供儿童身份线索能显著提升模型的安全表现。

    

    儿童越来越多地接触大型语言模型（LLM），这可能使他们面临发展不适宜的回复，或需要年龄敏感的安全、指导和界限。现有的LLM安全评估主要集中于一般有害内容的规避，并未明确针对面向儿童的安全问题。我们提出了KIDBench，一个基于发展心理学的、使用LLM作为评判标准的基准，用于评估面向7-11岁儿童的LLM安全性。KIDBench包含涵盖十个类别的真实儿童提问，包括单轮提示和多轮儿童角色模拟。我们比较了无线索提示（不含儿童背景信息）、隐式线索提示（暗示说话者是儿童）以及明确年龄指令。隐式线索比无线索提示的得分提高了8.6%-46.8%，而明确年龄指令在隐式线索的基础上又带来了9.9%-30.4%的额外提升。跨语言和文化评估显示安全表现并不均衡。

    arXiv:2605.25510v3 Announce Type: replace  Abstract: Children increasingly have access to Large Language Models (LLMs), which may expose them to responses that are developmentally inappropriate or require age-sensitive safety, guidance, and boundaries. Existing LLM safety evaluations largely focus on general harmful-content avoidance and do not explicitly target child-facing safety. We introduce KIDBench, a benchmark for evaluating child-facing LLM safety for ages 7-11 using a LLM-as-a-Judge rubric grounded in developmental-psychology. KIDBench contains realistic child queries across ten categories, with single-turn prompts and multi-turn child-actor simulations. We compare no-cues prompts with no child context, implicit-cues prompts that suggest a child speaker, and explicit age instructions. Implicit-cues improve scores by 8.6-46.8% over no-cue, while explicit age provides an additional 9.9-30.4% improvement over implicit-cues. Cross-lingual and cultural evaluations show uneven safet
    
[^158]: 电路能告诉我们多少？测量语言模型电路的一致性与特异性

    How Much Do Circuits Tell Us? Measuring the Consistency and Specificity of Language Model Circuits

    [https://arxiv.org/abs/2605.08348](https://arxiv.org/abs/2605.08348)

    本文提出用一致性和特异性两个新属性来评估机制可解释性中的电路，发现组件级电路虽高度一致且因果重要但不具任务特异性，而神经元级电路虽具任务特异性却一致性较差。

    

    arXiv:2605.08348v2 公告类型：替换。摘要：机制可解释性中的电路框架旨在识别对某种行为具有因果责任的模型组件稀疏子图，通常通过测量必要性和充分性来进行评估。但这些标准很少能说明一个电路是否一致地捕捉了模型执行任务的方式，或者该电路是否为该任务所特有。我们在六个任务和五个模型上研究了这两个属性——一致性和特异性，分别在组件层面（注意力头和MLP块）以及单个MLP神经元层面提取电路。我们发现，组件层面的电路在大多数任务上具有高度一致性和因果重要性，但它们并不具备特异性：消融某一任务的电路对另一任务性能造成的损害，与消融该任务自身电路的损害相差无几。另一方面，神经元层面的电路表现出更高的任务特异性，但其在任务内的一致性却要低得多。这可以通过电路……（原文在此处截断）

    arXiv:2605.08348v2 Announce Type: replace  Abstract: The circuits framework in mechanistic interpretability aims to identify sparse subgraphs of model components that are causally responsible for a behavior, typically evaluated by measuring necessity and sufficiency. But these criteria say little about whether a circuit consistently captures how a model performs a task, or if it is specific to that task. We study these two properties, consistency and specificity, across six tasks and five models, extracting circuits at the component level (attention heads and MLP blocks) and at the level of individual MLP neurons. We find that component-level circuits are highly consistent and causally important on most tasks, but they are not specific: ablating one task's circuit damages another task's performance about as much as that task's own circuit does. Neuron-level circuits, on the other hand, exhibit higher task-specificity but are far less consistent within tasks. This is explained by circui
    
[^159]: 超越可解码性：利用编码探针重构语言模型表示

    Beyond Decodability: Reconstructing Language Model Representations with an Encoding Probe

    [https://arxiv.org/abs/2605.00607](https://arxiv.org/abs/2605.00607)

    本文提出一种编码探针方法，通过可解释特征重构语言模型的内部表示，克服了传统解码探针无法直接比较不同特征贡献且受特征相关性干扰的两大局限。

    

    探针技术被广泛用于研究哪些特征可以从语言模型表示中解码出来。然而，常见的解码探针方法存在两个局限性：不同特征对模型表示的贡献无法直接比较，且特征之间的相关性会影响探针结果。我们提出了一种编码探针，它反转了这一方向，使用可解释的特征来重构模型的内部表示。我们在文本和语音Transformer模型上对该方法进行了评估，使用了涵盖声学、语音学、句法、词汇和说话人身份的特征集。结果表明，与说话人相关的效应在不同的训练目标和数据集之间差异很大，而句法和词汇特征则独立地对重构做出贡献。这些结果表明，编码探针为解释语言模型提供了一个互补的视角。

    arXiv:2605.00607v2 Announce Type: replace  Abstract: Probing is widely used to study which features can be decoded from language model representations. However, the common decoding probe approach has two limitations that we aim to solve with our new encoding probe approach: contributions of different features to model representations cannot be directly compared, and feature correlations can affect probing results. We present an Encoding Probe that reverses this direction and reconstructs internal representations of models using interpretable features. We evaluate this method on text and speech transformer models, using feature sets spanning acoustics, phonetics, syntax, lexicon, and speaker identity. Our results suggest that speaker-related effects vary strongly across different training objectives and datasets, while syntactic and lexical features contribute independently to reconstruction. These results show that the Encoding Probe provides a complementary perspective on interpreting
    
[^160]: 当思维链失败时，解决方案隐藏在隐藏状态之中

    When Chain-of-Thought Fails, the Solution Hides in the Hidden States

    [https://arxiv.org/abs/2604.23351](https://arxiv.org/abs/2604.23351)

    研究发现，即使思维链推理轨迹本身是错误的，其隐藏状态（尤其集中于中后层和轨迹早期）仍编码了足以恢复正确答案的任务相关信息，通过激活修补将这些隐藏状态注入直接回答过程可显著提升答题准确率。

    

    arXiv:2604.23351v2 公告类型：replace-cross。摘要：中间推理在计算上究竟是有用的，还是仅仅起解释作用，取决于思维链（CoT）标记中是否包含与任务相关的信息。我们利用激活修补（activation patching）对 GSM8K 上的思维链进行了机制层面的因果分析：将同一问题在思维链生成中得到的标记级隐藏状态转移到直接回答的运行中，然后测量其对最终答案准确率的影响。在各个模型上，修补后生成答案的准确率显著高于直接回答提示和原始思维链轨迹，这表明即使原始推理轨迹是错误的，单个思维链标记也能编码足以恢复正确答案的信息。这种与任务相关的信息在正确的思维链运行中比在错误的运行中更为普遍，并且在各标记之间分布不均：它集中于中间至较深的层，并在推理轨迹的较早位置出现。此外，修补语言标记，例如……（原文摘要在此处截断）

    arXiv:2604.23351v2 Announce Type: replace-cross  Abstract: Whether intermediate reasoning is computationally useful or merely explanatory depends on whether chain-of-thought (CoT) tokens contain task-relevant information. We present a mechanistic causal analysis of CoT on GSM8K using activation patching: transferring token-level hidden states from a CoT generation to a direct-answer run for the same question, then measuring the effect on final-answer accuracy. Across models, generating after patching yields substantially higher accuracy than both direct-answer prompting and the original CoT trace, revealing that individual CoT tokens can encode sufficient information to recover the correct answer, even when the original trace is incorrect. This task-relevant information is more prevalent in correct than incorrect CoT runs and is unevenly distributed across tokens, concentrating in mid-to-late layers and appearing earlier in the reasoning trace. Moreover, patching language tokens such a
    
[^161]: 一个模型翻译所有语言？通往末日火山的多语言模型合并之旅

    One Model to Translate Them All? A Journey to Mount Doom for Multilingual Model Merging

    [https://arxiv.org/abs/2604.02881](https://arxiv.org/abs/2604.02881)

    该论文系统研究了多语言机器翻译中的权重空间模型合并，揭示了显著的方向性不对称现象——当模型共享目标语言时合并相对有效但无法保留各语言模型的峰值性能，而当目标语言不同时性能则急剧下降。

    

    权重空间模型合并可以在不访问原始训练数据的情况下，将独立微调的模型检查点组合起来。虽然合并在多任务设置中已展现出潜力，但其在多语言生成系统中的行为仍未得到充分探索。我们通过在大规模双语语料库上对语言模型进行完全微调，并在共享源语言、共享目标语言和双向整合等设置下评估代表性的合并策略，系统性地研究了多语言机器翻译中的权重空间合并。我们的实验揭示了强烈的方向性不对称性：当模型共享目标语言时，合并相对更为有效，相比基础模型能够提升多语言覆盖范围，但仍无法保留特定语言检查点的峰值性能；相反，当目标语言不同时，性能会急剧下降，尤其是在共享源语言和双向设置中。为了解释这一……

    arXiv:2604.02881v2 Announce Type: replace-cross  Abstract: Weight-space model merging combines independently fine-tuned checkpoints without access to the original training data. While merging has shown promise in multitask settings, its behavior in multilingual generative systems remains underexplored. We systematically study weight-space merging for multilingual machine translation by fully fine-tuning language models on large-scale bilingual corpora and evaluating representative merging strategies across shared-source, shared-target, and bidirectional consolidation settings. Our experiments reveal a strong directional asymmetry. Merging is comparatively more effective when models share a target language, improving multilingual coverage over the base model, but it still fails to preserve the peak performance of language-specific checkpoints. In contrast, when target languages differ, performance degrades sharply, especially in shared-source and bidirectional settings. To explain this 
    
[^162]: 基于大语言模型的阿拉伯语形态句法标注与依存句法分析

    Arabic Morphosyntactic Tagging and Dependency Parsing with Large Language Models

    [https://arxiv.org/abs/2603.16718](https://arxiv.org/abs/2603.16718)

    本文对大语言模型在阿拉伯语形态句法标注与依存句法分析任务上进行了统一评估，发现基于检索的上下文学习能显著提升性能，最强的大语言模型已接近有监督系统的水平，但需要大量标注数据和计算资源。

    

    大语言模型在自然语言处理的各个领域表现优异，但其生成显式语法分析的能力仍不明确。阿拉伯语因其丰富的形态变化和拼写歧义，形成了强烈的形态-句法交互，因而提供了一个具有挑战性的测试平台。我们对大语言模型在阿拉伯语形态句法标注和依存句法分析上的表现进行了统一评估，涵盖预分词、原始文本和级联三种设置。我们比较了零样本提示与基于检索的上下文学习，发现相关示例能够显著提升性能。最强大的大语言模型已接近有监督标注与分析系统的水平；然而，它们需要大量标注数据用于示例检索，并需要可观的计算资源。我们已将本文使用的所有代码和数据公开。

    arXiv:2603.16718v2 Announce Type: replace  Abstract: LLMs perform strongly across NLP, but their ability to produce explicit grammatical analyses remains unclear. Arabic provides a challenging testbed due to its rich morphology and orthographic ambiguity, which create strong morphology-syntax interactions. We present a unified evaluation of LLMs on Arabic morphosyntactic tagging and dependency parsing, covering pre-tokenized, raw-text, and cascaded settings. We compare zero-shot prompting with retrieval-based in-context learning. Relevant demonstrations substantially improve performance. The strongest LLMs approach supervised tagging and parsing systems; however, they require substantial annotated data for demonstration retrieval and considerable computational resources. We make all code and data used in this paper publicly available.
    
[^163]: 想象有助于视觉推理，但在潜在空间中尚未实现

    Imagination Helps Visual Reasoning, But Not Yet in Latent Space

    [https://arxiv.org/abs/2602.22766](https://arxiv.org/abs/2602.22766)

    本文通过因果中介分析揭示当前潜在视觉推理存在输入与潜在token、潜在token与最终答案之间的两个关键因果断连，表明潜在空间的想象机制尚未真正发挥有效作用。

    

    潜在视觉推理旨在通过多模态大语言模型的隐藏状态进行沉思，以模拟人类的想象过程。虽然这被认为是视觉推理的一种有前景的范式，但驱动其有效性的底层机制仍不清楚。为了揭示其有效性的真正来源，我们使用因果中介分析研究了潜在推理的有效性。我们将该过程建模为一条因果链：输入作为处理变量，潜在token作为中介变量，最终答案作为结果变量。我们的发现揭示了两个关键的断连： 潜在断连：对输入进行剧烈扰动仅导致潜在token发生微不足道的变化，表明潜在token没有有效地关注输入序列。 答案断连：对潜在token进行扰动对最终答案的影响极小，表明潜在token的因果效应十分有限。

    arXiv:2602.22766v3 Announce Type: replace  Abstract: Latent visual reasoning aims to mimic human's imagination process by meditating through hidden states of Multimodal Large Language Models. While recognized as a promising paradigm for visual reasoning, the underlying mechanisms driving its effectiveness remain unclear. Motivated to demystify the true source of its efficacy, we investigate the validity of latent reasoning using Causal Mediation Analysis. We model the process as a causal chain: the input as the treatment, the latent tokens as the mediator, and the final answer as the outcome. Our findings uncover two critical disconnections: (a) Input-Latent Disconnect: dramatic perturbations on the input result in negligible changes to the latent tokens, suggesting that latent tokens do not effectively attend to the input sequence. (b) Latent-Answer Disconnect: perturbations on the latent tokens yield minimal impact on the final answer, indicating the limited causal effect latent toke
    
[^164]: Ex-Omni：为全模态大语言模型实现3D面部动画生成

    Ex-Omni: Enabling 3D Facial Animation Generation for Omni-modal Large Language Models

    [https://arxiv.org/abs/2602.07106](https://arxiv.org/abs/2602.07106)

    该论文提出Ex-Omni框架，通过带混合变形协同监督的语音单元生成器与非自回归混合变形解码器将语义推理与时间生成解耦，并结合令牌即查询门控融合接口和120万样本弱监督数据集InstructS2SF-1200K，首次使全模态大语言模型能够联合生成语音与3D面部动画。

    

    全模态大语言模型（OLLMs）旨在统一多模态理解与生成，然而将其扩展至联合生成语音与3D面部动画的研究仍鲜有探索。一个关键挑战在于大语言模型的离散语义推理与3D面部运动所需的密集时间动态之间的不匹配。我们提出Expressive Omni（Ex-Omni），一个为全模态大语言模型赋予伴随语音的3D面部动画能力的框架。Ex-Omni通过带有混合变形协同监督的语音单元生成器和非自回归的混合变形解码器，将语义推理与时间生成解耦，其中语音单元提供时间骨架支撑，隐藏语音表示则承载面部相关线索。我们进一步引入了令牌即查询门控融合（TQGF）接口，用于可控的语义注入，并构建了InstructS2SF-1200K，一个包含120万样本的弱监督数据集，用于伴随语音的面部动画生成。

    arXiv:2602.07106v3 Announce Type: replace-cross  Abstract: Omni-modal large language models (OLLMs) aim to unify multimodal understanding and generation, yet extending them to jointly produce speech and 3D facial animation remains largely underexplored. A key challenge is the mismatch between the discrete semantic reasoning of LLMs and the dense temporal dynamics required for 3D facial motion. We propose Expressive Omni (Ex-Omni), a framework that augments OLLMs with speech-accompanied 3D facial animation. Ex-Omni decouples semantic reasoning from temporal generation through a speech-unit generator with blendshape co-supervision and a non-autoregressive blendshape decoder, where speech units provide temporal scaffolding and hidden speech representations carry facially relevant cues. We further introduce a token-as-query gated fusion (TQGF) interface for controlled semantic injection, as well as InstructS2SF-1200K, a 1.2M-sample weakly supervised dataset for speech-accompanied facial an
    
[^165]: 深度网络从局部统计中学习解析等深度上下文无关语言

    Deep networks learn to parse uniform-depth context-free languages from local statistics

    [https://arxiv.org/abs/2602.06065](https://arxiv.org/abs/2602.06065)

    该研究引入了一类可调节歧义程度和跨尺度相关结构的概率上下文无关文法，揭示了深度网络能够仅从局部统计特征中学习解析语言的层次结构。

    

    理解如何仅从句子中学习语言的结构是认知科学与机器学习中的一个核心问题。对大型语言模型（LLMs）内部表示的研究支持了它们在预测下一个词时解析文本的能力，同时以独立于表面形式的方式表征语义概念。然而，究竟是哪些数据统计特征使这些能力成为可能，以及需要多少数据，在很大程度上仍是未知的。概率上下文无关文法（PCFGs）为研究这些问题提供了一个易于处理的测试平台。然而，先前的工作要么集中于对训练后网络所使用的类解析算法进行事后表征，要么集中于具有固定语法的PCFG的可学习性——在这种情况下解析本身是不必要的。在此，我们引入了一类可调节的PCFG，其中歧义程度和跨尺度的相关结构均可被控制……

    arXiv:2602.06065v4 Announce Type: replace-cross  Abstract: Understanding how the structure of language can be learned from sentences alone is a central question in both cognitive science and machine learning. Studies of the internal representations of Large Language Models (LLMs) support their ability to parse text when predicting the next word, while representing semantic notions independently of surface form. Yet, which data statistics make these feats possible, and how much data is required, remain largely unknown. Probabilistic context-free grammars (PCFGs) provide a tractable testbed for studying these questions. However, prior work has focused either on the post-hoc characterization of the parsing-like algorithms used by trained networks; or on the learnability of PCFGs with fixed syntax, where parsing is unnecessary. Here, we (i) introduce a tunable class of PCFGs in which both the degree of ambiguity and the correlation structure across scales can be controlled; (ii) provide a 
    
[^166]: 中文社交媒体文本机器翻译基准测试

    Benchmarking Machine Translation on Chinese Social Media Texts

    [https://arxiv.org/abs/2601.22931](https://arxiv.org/abs/2601.22931)

    该论文提出了CSM-MTBench基准，涵盖五个中外语言方向，包含“趣味帖子”和“社交片段”两个专家策划的子集，并针对中文社交媒体文本提出了定制化评估方法，以解决数据稀缺和传统评估指标难以捕捉风格保真度的双重难题。

    

    非正式用户生成文本中快速演变的俚语、新词和高度风格化的表达，尤其是在中国社交媒体上，给机器翻译（MT）基准测试带来了重大挑战。具体而言，我们识别出两个主要障碍：（1）数据稀缺，因为高质量的平行数据需要熟悉平台特定俚语以及两种语言风格特征的双语标注者；（2）评估指标的局限性，即COMET等传统评估器往往无法捕捉风格保真度和非标准表达。为了弥补这些差距，我们引入了CSM-MTBench，这是一个涵盖五个中外语言方向的基准测试，由两个专家策划的子集组成：趣味帖子，其特点是上下文丰富、俚语和新词密集的内容；以及社交片段，强调简洁、情感和风格驱动的表达。此外，我们为每个子集提出了定制的评估方法。

    arXiv:2601.22931v2 Announce Type: replace  Abstract: The prevalence of rapidly evolving slang, neologisms, and highly stylized expressions in informal user-generated text, particularly on Chinese social media, poses significant challenges for Machine Translation (MT) benchmarking. Specifically, we identify two primary obstacles: (1) data scarcity, as high-quality parallel data requires bilingual annotators familiar with platform-specific slang, and stylistic cues in both languages; and (2) metric limitations, where traditional evaluators like COMET often fail to capture stylistic fidelity and nonstandard expressions. To bridge these gaps, we introduce CSM-MTBench, a benchmark covering five Chinese-foreign language directions and consisting of two expert-curated subsets: Fun Posts, featuring context-rich, slang- and neologism-heavy content, and Social Snippets, emphasizing concise, emotion- and style- driven expressions. Furthermore, we propose tailored evaluation approaches for each su
    
[^167]: 关系线性性是幻觉的预测指标

    Relational Linearity is a Predictor of Hallucinations

    [https://arxiv.org/abs/2601.11429](https://arxiv.org/abs/2601.11429)

    该论文提出关系线性性可预测语言模型的幻觉：由于抽象表示方案，语言模型能轻松为线性关系中不存在的主体生成看似合理的客体从而导致幻觉，而在面对非线性关系时这种机制失效，幻觉更容易避免。

    

    幻觉是语言模型（LM）的一个核心失败模式。我们关注语言模型在回答诸如“Glenn Gould演奏什么乐器？”这类问题时产生的幻觉，但我们针对被设计为模型未知的合成实体来提出这些问题。我们发现像Gemma-7B-IT这样的语言模型经常产生幻觉，即它们难以识别所幻觉出的事实并不属于其自身知识。基于线性关系嵌入的思想，我们提出了以下假设：（i）由于用于表示它们的抽象方案，语言模型可以轻松地为线性关系中不存在的主体生成看似合理的客体，这可能导致幻觉。（ii）对于非线性关系，这种生成客体的机制不可用，因此幻觉更容易被避免。为了检验这一假设，我们创建了SynthHal，这是一个针对15种关系的合成未知实体基准测试。我们发现，在四个……

    arXiv:2601.11429v3 Announce Type: replace-cross  Abstract: Hallucination is a central failure mode of language models (LMs). We focus on hallucinations in response to questions like: "Which instrument did Glenn Gould play?", but we ask these questions for synthetic entities designed to be unknown to the model. We find that LMs like Gemma-7B-IT frequently hallucinate, i.e., they have difficulty recognizing that the hallucinated fact is not part of their knowledge. Based on the idea of linear relational embeddings, we put forward the following hypothesis. (i) Due to the abstract scheme that is used to represent them, LMs can easily produce plausible objects for non-existing subjects of linear relations, which can lead to hallucinations. (ii) For nonlinear relations, this mechanism for producing an object is not available and so a hallucination is easier to avoid. To test this hypothesis, we create SynthHal, a synthetic unknown-entity benchmark for 15 relations. We find that across four i
    
[^168]: HOMURA：通过强化学习驯服“沙漏”，实现时间受限的大语言模型翻译

    HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning

    [https://arxiv.org/abs/2601.10187](https://arxiv.org/abs/2601.10187)

    该论文提出了Sand-Glass音节级时长约束翻译基准和Homura强化学习框架，通过新颖的动态音节比率奖励有效解决LLM翻译的跨语言冗长偏差问题，使其适用于字幕、配音等时间受限场景。

    

    大语言模型（LLM）在多语言翻译方面取得了显著进展，但受到系统性的跨语言冗长偏差的阻碍，使其不适用于字幕制作和配音等严格时间受限的任务。当前的提示工程方法难以解决语义忠实性与严格时间可行性之间的冲突。为了弥合这一差距，我们首先介绍了Sand-Glass，一个专门设计用于在音节级时长约束下评估翻译效果的基准。此外，我们提出了Homura，一个显式优化语义保留与时间合规之间权衡的强化学习框架。通过采用包含新颖动态音节比率奖励的约束强化学习目标，Homura有效地“驯服”了输出长度。实验结果表明，Homura显著优于强大的基线方法，实现了……

    arXiv:2601.10187v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have achieved remarkable strides in multilingual translation but are hindered by a systemic cross-lingual verbosity bias, rendering them unsuitable for strict time-constrained tasks like subtitling and dubbing. Current prompt-engineering approaches struggle to resolve this conflict between semantic fidelity and rigid temporal feasibility. To bridge this gap, we first introduce Sand-Glass, a benchmark specifically designed to evaluate translation under syllable-level duration constraints. Furthermore, we propose Homura, a reinforcement learning framework that explicitly optimizes the trade-off between semantic preservation and temporal compliance. By employing a constrained reinforcement learning objective featuring a novel dynamic syllable-ratio reward, Homura effectively "tames" the output length. Experimental results demonstrate that Homura significantly outperforms strong baselines, achieving pre
    
[^169]: 先想象后规划：基于世界模型自适应前瞻的智能体学习

    Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models

    [https://arxiv.org/abs/2601.08955](https://arxiv.org/abs/2601.08955)

    提出了ITP统一框架，让智能体策略模型与世界模型交互生成多步想象轨迹，并通过权衡最终目标与任务进度的自适应前瞻机制，充分释放世界模型在复杂任务规划中的潜力。

    

    世界模型的最新进展在建模环境状态的未来动态方面展现出巨大潜力，使智能体无需访问真实环境即可进行推理与行动。然而，当前方法主要执行单步或固定时域的推演，其在复杂任务规划中的潜力尚未得到充分挖掘。我们提出了“先想象后规划”（Imagine-then-Plan, ITP），这是一个通过前瞻想象进行智能体学习的统一框架，其中智能体的策略模型与学习到的世界模型进行交互，生成多步“想象”轨迹。由于想象时域可能因任务和阶段的不同而变化，我们引入了一种新颖的自适应前瞻机制，通过权衡最终目标与任务进度来确定想象步长。由此得到的想象轨迹提供了关于未来后果的丰富信号，例如已取得的进展和潜在的冲突，这些信号与当前观测相融合，构成了部分可观测与想象的（摘要在此处截断）

    arXiv:2601.08955v3 Announce Type: replace-cross  Abstract: Recent advances in world models have shown promise for modeling future dynamics of environmental states, enabling agents to reason and act without accessing real environments. Current methods mainly perform single-step or fixed-horizon rollouts, leaving their potential for complex task planning under-exploited. We propose Imagine-then-Plan (\texttt{ITP}), a unified framework for agent learning via lookahead imagination, where an agent's policy model interacts with the learned world model, yielding multi-step ``imagined'' trajectories. Since the imagination horizon may vary by tasks and stages, we introduce a novel adaptive lookahead mechanism by trading off the ultimate goal and task progress. The resulting imagined trajectories provide rich signals about future consequences, such as achieved progress and potential conflicts, which are fused with current observations, formulating a partially \textit{observable} and \textit{imag
    
[^170]: CHisAgent：面向中国古代文化体系的事件分类体系构建多智能体框架

    CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems

    [https://arxiv.org/abs/2601.05520](https://arxiv.org/abs/2601.05520)

    该论文提出CHisAgent多智能体框架，通过归纳、扩展、充实三个角色专业化阶段，从《二十四史》等中国古代文献中自动构建历史事件分类体系，克服了LLM在中国历史语境下推理能力不足和人工分类构建成本高的问题。

    

    尽管大型语言模型（LLM）在许多任务上表现出色，但其在历史与文化推理方面的能力有限，尤其是在中国历史等非英语语境中。分类体系结构为组织历史知识、提升理解提供了一种有效机制，然而人工构建分类体系成本高昂且难以规模化。因此，我们提出了CHisAgent，一个面向中国古代语境的历史分类体系构建多智能体LLM框架。CHisAgent将分类体系构建分解为三个角色专业化的阶段：自下而上的“归纳器”从原始历史语料中推导出初始层级结构；自上而下的“扩展器”利用LLM的世界知识补充缺失的中间概念；以及证据引导的“充实器”整合外部结构化历史资源以确保忠实性。利用《二十四史》……

    arXiv:2601.05520v2 Announce Type: replace  Abstract: Despite strong performance on many tasks, large language models (LLMs) show limited ability in historical and cultural reasoning, particularly in non-English contexts such as Chinese history. Taxonomic structures offer an effective mechanism to organize historical knowledge and improve understanding. However, manual taxonomy construction is costly and difficult to scale. Therefore, we propose \textbf{CHisAgent}, a multi-agent LLM framework for historical taxonomy construction in ancient Chinese contexts. CHisAgent decomposes taxonomy construction into three role-specialized stages: a bottom-up \textit{Inducer} that derives an initial hierarchy from raw historical corpora, a top-down \textit{Expander} that introduces missing intermediate concepts using LLM world knowledge, and an evidence-guided \textit{Enricher} that integrates external structured historical resources to ensure faithfulness. Using the \textit{Twenty-Four Histories}, 
    
[^171]: 迈向多模态多轮安全：从智能体交互到战略性对齐

    Towards Multi-modal Multi-turn Safety: From Agentic Interaction to Strategic Alignment

    [https://arxiv.org/abs/2601.04736](https://arxiv.org/abs/2601.04736)

    提出MINT-Safe开源视觉多轮安全训练数据集，针对多轮交互中攻击者渐进式重构有害意图的新型安全风险，弥补现有RLHF方法无法捕捉跨轮次风险动态的不足。

    

    尽管多模态大语言模型（MLLMs）在多模态理解方面展现出卓越能力，但将其部署于开放式对话场景中会引入安全风险，而现有的对齐方法对此尚未充分解决。与简单的恶意视觉问答（VQA）对不同，多轮交互使攻击者能够在多轮对话中逐步重构有害意图，以逐轮渐进的方式绕过安全约束，而这种绕过方式在单独的任何一轮中都难以被检测。与此同时，传统的基于人类反馈的强化学习（RLHF）方法并不适用于这种情况：由于其主要为VQA任务设计，既无法捕捉跨轮次的风险动态，也难以在缺乏昂贵人工偏好标注的情况下高效扩展。为填补这一空白，我们提出了MINT-Safe，这是一个开源的视觉多轮训练数据集，包含11,270个多图像对话和500个拒答VQA对……

    arXiv:2601.04736v2 Announce Type: replace  Abstract: Despite remarkable capability in multi-modal understanding, deploying Multi-modal Large Language Models (MLLMs) in open-ended conversational scenarios introduces safety risks that remain poorly addressed by existing alignment methods. Unlike simple malicious visual question and answer (VQA) pairs , multi-turn interactions enable adversaries to incrementally reconstruct harmful intent across dialogues, progressively bypassing safety constraints in ways that are difficult to detect at any individual turn. Meanwhile, conventional reinforcement learning from human feedback (RLHF) approaches are unsuitable for this situation: designed primarily for VQA tasks, they neither capture cross-turn risk dynamics nor scale efficiently without costly manual preference annotation. To close this gap, we introduce \textbf{MINT-Safe}, an open-source visual multi-turn training dataset comprising 11,270 multi-image dialogues and 500 refusal VQA pairs, co
    
[^172]: 用户感知与代理LLM评审：LLM对隐私敏感场景响应中的隐私性与有用性

    User Perceptions vs. Proxy LLM Judges: Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios

    [https://arxiv.org/abs/2510.20721](https://arxiv.org/abs/2510.20721)

    该研究通过94人的用户实验发现，用户对LLM在隐私敏感场景下响应的评价彼此一致性较低，这表明此前以代理LLM作为评审的基准测试结果可能与真实用户的隐私和有用性感知存在显著偏差。

    

    大语言模型（LLM）正迅速被应用于起草电子邮件、总结会议记录和回答健康问题等任务。在这些场景中，用户可能需要分享私人信息（例如联系方式、健康记录）。为评估LLM识别并隐去此类信息的能力，先前的研究引入了基于真实生活场景的基准测试（如ConfAIde、PrivacyLens），并发现LLM在复杂场景中可能会泄露私人信息。然而，这些评估依赖代理LLM来判断LLM响应的有用性和隐私保护质量，而非直接测量用户的真实感知。为了解用户如何感知LLM对隐私敏感场景响应的有用性与隐私保护质量，我们使用90个PrivacyLens场景开展了一项用户研究（样本量n=94）。我们发现，用户在评估相同的LLM响应时彼此之间的一致性较低。

    arXiv:2510.20721v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) are rapidly being adopted for tasks like drafting emails, summarizing meetings, and answering health questions. In these settings, users may need to share private information (e.g., contact details, health records). To evaluate LLMs' ability to identify and redact such information, prior work introduced real-life, scenario-based benchmarks (e.g., ConfAIde, PrivacyLens) and found that LLMs can leak private information in complex scenarios. However, these evaluations relied on proxy LLMs to judge the helpfulness and privacy-preservation quality of LLM responses, rather than directly measuring users' perceptions. To understand how users perceive the helpfulness and privacy-preservation quality of LLM responses to privacy-sensitive scenarios, we conducted a user study ($n=94$) using 90 PrivacyLens scenarios. We found that users had low agreement with each other when evaluating identical LLM responses. I
    
[^173]: 在乌尔都语习语上评估大型语言模型

    Evaluating Large Language Models on Urdu Idioms

    [https://arxiv.org/abs/2510.17460](https://arxiv.org/abs/2510.17460)

    该研究构建了一个包含4000个人工验证的乌尔都语习语句对的双文字基准数据集，通过翻译、改写、习语检测和回译等多项任务及多种提示策略进行评估，发现前沿大语言模型在所有设置下均优于传统神经机器翻译系统。

    

    习语因其比喻性和植根于文化的含义而区别于字面表达，始终是自然语言处理领域的一个持续性挑战。尽管大型语言模型（LLM）的最新进展改善了多种语言对习语的处理，但乌尔都语等低资源语言在这方面的关注仍然有限。在这项工作中，我们提出了一个面向乌尔都语到英语习语翻译的综合基准，包含一个经过人工验证的数据集，该数据集由4,000个对齐的习语句对组成，同时涵盖波斯-阿拉伯文字（乌尔都语原生文字）和罗马化乌尔都语两种书写形式。我们评估了多项任务，包括翻译、改写、习语片段检测和回译，并采用了多样化的提示策略，如文化提示、习语提示和少样本学习。我们的研究结果表明，在各评估设置下，前沿LLM始终优于传统的神经机器翻译系统。

    arXiv:2510.17460v2 Announce Type: replace  Abstract: Idioms remain a persistent challenge in natural language processing due to their figurative and culturally grounded meanings, which distinguish them from literal expressions. Although recent advances in large language models (LLMs) have improved idiom handling across several languages, limited attention has been given to low resource languages such as Urdu. In this work, we present a comprehensive benchmark for Urdu to English idiomatic translation, consisting of a manually verified dataset of 4,000 aligned idiom sentence pairs in both Perso Arabic (native Urdu script) and Romanized Urdu. We evaluate multiple tasks, including translation, paraphrasing, idiom span detection, and back-translation, using diverse prompting strategies such as cultural prompting, idiomatic prompting, and few-shot learning. Our findings show that frontier LLMs consistently outperform traditional neural machine translation systems across all evaluation setti
    
[^174]: EasySteer：一个高性能且可扩展的大语言模型引导统一框架

    EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering

    [https://arxiv.org/abs/2509.25175](https://arxiv.org/abs/2509.25175)

    EasySteer是一个基于vLLM构建的高性能、可扩展的大语言模型推理时引导统一框架，相比现有框架实现了10.8-22.3倍的加速，并提供模块化可插拔接口、细粒度参数控制以及八个应用领域的预计算引导向量。

    

    大语言模型（LLM）引导已成为一种在推理时通过定向操作隐藏状态来控制模型行为的有前景的范式，为昂贵的重新训练提供了一种轻量级替代方案。然而，现有的引导框架存在关键局限性：计算效率低、可扩展性有限以及功能受限，这些都阻碍了研究进展和实际部署。我们提出了EasySteer，一个基于vLLM构建的高性能、可扩展的大语言模型引导统一框架。该系统具有模块化架构，为基于分析和基于学习的方法提供可插拔接口，支持细粒度参数控制，提供八个应用领域的预计算引导向量，并配备了交互式演示系统。通过与vLLM优化的推理引擎深度集成，EasySteer相比现有框架实现了10.8-22.3倍的加速。

    arXiv:2509.25175v3 Announce Type: replace-cross  Abstract: Large language model (LLM) steering has emerged as a promising paradigm for controlling model behavior at inference time through targeted manipulation of hidden states, offering a lightweight alternative to expensive retraining. However, existing steering frameworks suffer from critical limitations: computational inefficiency, limited extensibility, and restricted functionality that hinder both research progress and practical deployment. We present EasySteer, a unified framework for high-performance, extensible LLM steering built on vLLM. Our system features modular architecture with pluggable interfaces for both analysis-based and learning-based methods, fine-grained parameter control, pre-computed steering vectors for eight application domains, and an interactive demonstration system. Through deep integration with vLLM's optimized inference engine, EasySteer achieves 10.8-22.3$\times$ speedup over existing frameworks. Extensi
    
[^175]: 因果-反事实RAG：将因果-反事实推理融入检索增强生成

    Causal-Counterfactual RAG: The Integration of Causal-Counterfactual Reasoning into RAG

    [https://arxiv.org/abs/2509.14435](https://arxiv.org/abs/2509.14435)

    该论文提出因果-反事实RAG框架，通过将显式因果图与反事实推理融入检索增强生成过程，解决了传统RAG系统因文本分块破坏上下文完整性和过度依赖语义相似性检索而导致的回答浅显不准确的问题。

    

    大型语言模型（LLMs）通过整合大规模预训练知识，变革了自然语言处理（NLP）领域，支持了多样化的应用。然而，其静态知识限制了对信息进行动态推理的能力，尤其是在知识密集型领域。检索增强生成（RAG）通过将检索机制与生成模型相结合以提升上下文理解能力，从而应对这一挑战。传统RAG系统由于文本分块破坏了上下文完整性，并且过度依赖语义相似性进行检索，往往导致回答浅显且不够准确。我们提出了因果-反事实RAG（Causal-Counterfactual RAG），这是一个新颖的框架，它将表示因果关系的显式因果图融入检索过程，并引入基于因果结构的反事实推理。与传统方法不同，我们的框架不仅评估……

    arXiv:2509.14435v3 Announce Type: replace  Abstract: Large language models (LLMs) have transformed natural language processing (NLP), enabling diverse applications by integrating large-scale pre-trained knowledge. However, their static knowledge limits dynamic reasoning over external information, especially in knowledge-intensive domains. Retrieval-Augmented Generation (RAG) addresses this challenge by combining retrieval mechanisms with generative modeling to improve contextual understanding. Traditional RAG systems suffer from disrupted contextual integrity due to text chunking and over-reliance on semantic similarity for retrieval, often resulting in shallow and less accurate responses. We propose Causal-Counterfactual RAG, a novel framework that integrates explicit causal graphs representing cause-effect relationships into the retrieval process and incorporates counterfactual reasoning grounded on the causal structure. Unlike conventional methods, our framework evaluates not only d
    
[^176]: 人类心理测量问卷无法准确刻画大语言模型的行为

    Human Psychometric Questionnaires Mischaracterize LLM Behavior

    [https://arxiv.org/abs/2509.10078](https://arxiv.org/abs/2509.10078)

    该研究发现，人类心理测量问卷中的题目含有明显的词汇线索，会让大语言模型识别出被测构念并给出符合社会期望的回答，因此基于问卷得到的模型人格与价值观画像并不能反映其在真实日常用户交互中的实际生成行为。

    

    我们检验了人类心理测量问卷能否作为可靠工具来刻画和预测大语言模型在日常用户交互中的行为。我们分析了八个开源大语言模型，通过两种不同的方法比较其价值观与人格画像：一是基于既定问卷（PVQ-40/21 和 BFI-44/10）的利克特量表自我报告，二是对日常用户查询中带有价值倾向回答的生成概率。两种画像存在显著分歧。通常被引用为大语言模型稳定倾向证据的构念内题目一致性，在生成概率中消失了。我们发现，既定的问卷题目包含明确的词汇线索，使模型能够识别目标构念，并以与构念一致、符合社会期望的方式作答；而真实的用户查询所包含的可识别线索要少得多。此外，人口统计角色提示会使模型对人类问卷的回答发生偏移……

    arXiv:2509.10078v5 Announce Type: replace-cross  Abstract: We examine whether human psychometric questionnaires can serve as reliable tools for characterizing and predicting LLM behavior in everyday user interactions. We analyze eight open-source LLMs by comparing their value and personality profiles derived from two different methods: Likert self-reports on established questionnaires (PVQ-40/21 and BFI-44/10) and generation probabilities over value-laden responses to everyday user queries. The two profiles diverge substantially. Within-construct item consistency, often cited as evidence of stable LLM dispositions, disappears in generation probabilities. We find that established questionnaire items contain explicit lexical cues that allow models to recognize the target construct and respond in alignment-consistent, socially desirable ways, whereas realistic user queries contain far less recognizable cues. In addition, demographic persona prompts shift models' responses to human questio
    
[^177]: 大语言模型时代的医学推理：增强技术与应用的系统综述

    Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications

    [https://arxiv.org/abs/2508.00669](https://arxiv.org/abs/2508.00669)

    本文是首个针对大语言模型医学推理领域的系统综述，提出了涵盖训练时策略与测试时机制的推理增强技术分类体系，并系统分析了这些技术在多种数据模态和临床应用中的实践与评估方法。

    

    大语言模型在医学领域的蓬勃发展带来了令人印象深刻的能力，但其在执行系统性、透明性和可验证性推理方面仍存在关键差距，而这正是临床实践的基石。这一差距推动了从单步答案生成向专为医学推理设计的大语言模型发展的范式转变。本文对该新兴领域进行了首次系统综述。我们提出了一个推理增强技术的分类体系，将其分为训练时策略（如监督微调、强化学习）和测试时机制（如提示工程、多智能体系统）。我们分析了这些技术如何应用于不同的数据模态（文本、图像、代码）以及关键的医疗场景中，如诊断、教育和治疗规划。此外，我们还梳理了评估基准从简单准确率指标……（原文摘要到此截断）

    arXiv:2508.00669v2 Announce Type: replace-cross  Abstract: The proliferation of Large Language Models (LLMs) in medicine has enabled impressive capabilities, yet a critical gap remains in their ability to perform systematic, transparent, and verifiable reasoning, a cornerstone of clinical practice. This has catalyzed a shift from single-step answer generation to the development of LLMs explicitly designed for medical reasoning. This paper provides the first systematic review of this emerging field. We propose a taxonomy of reasoning enhancement techniques, categorized into training-time strategies (e.g., supervised fine-tuning, reinforcement learning) and test-time mechanisms (e.g., prompt engineering, multi-agent systems). We analyze how these techniques are applied across different data modalities (text, image, code) and in key clinical applications such as diagnosis, education, and treatment planning. Furthermore, we survey the evolution of evaluation benchmarks from simple accuracy
    
[^178]: IDRBench：理解大语言模型在跨学科研究中的能力

    IDRBench: Understanding the Capability of Large Language Models on Interdisciplinary Research

    [https://arxiv.org/abs/2507.15736](https://arxiv.org/abs/2507.15736)

    该论文提出IDRBench框架，通过论文识别、想法整合和想法推荐三项评估任务，系统评估了十个主流大语言模型在跨学科研究中整合不同领域知识的能力。

    

    创新是人类文明的关键驱动力。随着知识体系的不断增长，跨越不同学科整合知识（重大创新往往产生于此）变得越来越具有挑战性。机器学习模型，特别是大语言模型（LLMs）的最新进展，为获取广泛的知识来源提供了有效途径，并展现出令人印象深刻的推理能力，为跨学科发现带来了重大机遇。我们的研究旨在理解最先进的大语言模型在整合不同领域知识以开展跨学科研究（IDR）方面的能力。为了解决这一根本性问题，我们推出了IDRBench，这是一个开创性的框架，包含数据集和评估任务：(1) IDR论文识别，(2) IDR想法整合，以及(3) IDR想法推荐。我们对十个主流大语言模型的研究提供了全面的……（摘要在此处截断）

    arXiv:2507.15736v4 Announce Type: replace  Abstract: Innovation is a key driving force of human civilization. As the body of knowledge has grown considerably, bridging knowledge across different disciplines, where significant innovation often emerges, has become increasingly challenging. The recent advancements in machine learning models, particularly Large Language Models (LLMs), have provided effective access to extensive knowledge sources and shown impressive abilities in reasoning, rendering significant opportunities for interdisciplinary discovery. Our research aims to understand the capabilities of state-of-the-art LLMs in integrating knowledge from different fields for interdisciplinary research (IDR). To address this fundamental problem, we introduce IDRBench, a pioneering framework that includes both datasets and evaluation tasks: (1) IDR Paper Identification, (2) IDR Idea Integration, and (3) IDR Idea Recommendation. Our study on ten mainstream LLMs provides a comprehensive a
    
[^179]: LLM即GNN：面向文本属性图基础模型的图词表学习

    LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models

    [https://arxiv.org/abs/2503.03313](https://arxiv.org/abs/2503.03313)

    该论文提出PromptGFM，通过图词表学习将图节点融入语言模型词表空间，克服了现有LLM与GNN解耦架构及OOV token带来的跨图、跨任务迁移难题，构建了面向文本属性图的多功能图基础模型。

    

    文本属性图（TAGs）在现实场景中无处不在，其中每个节点都与文本描述相关联。它们通常表现出独特的结构和领域特定知识，这促使人们开发能够在多样化图和任务上进行泛化的图基础模型（GFM）。尽管人们已在为TAGs集成大型语言模型（LLMs）和图神经网络（GNNs）方面做出了大量努力，但现有方法存在解耦架构和两阶段对齐的问题，限制了它们的协同潜力。更糟糕的是，现有方法为图节点分配词表外（OOV）token，导致图特定语义、token爆炸以及与面向任务的提示模板不兼容，这阻碍了跨图和跨任务的可迁移性。为了解决这些挑战，我们提出了PromptGFM，这是一个基于图词表学习的、适用于TAGs的多功能图基础模型。PromptGFM包含两个关键组件。

    arXiv:2503.03313v4 Announce Type: replace-cross  Abstract: Text-Attributed Graphs (TAGs), where each node is associated with text descriptions, are ubiquitous in real-world scenarios. They typically exhibit distinctive structure and domain-specific knowledge, motivating the development of a Graph Foundation Model (GFM) that generalizes across diverse graphs and tasks. Despite large efforts to integrate Large Language Models (LLMs) and Graph Neural Networks (GNNs) for TAGs, existing approaches suffer from decoupled architectures with two-stage alignment, limiting their synergistic potential. Even worse, existing methods assign out-of-vocabulary (OOV) tokens to graph nodes, leading to graph-specific semantics, token explosion, and incompatibility with task-oriented prompt templates, which hinders cross-graph and cross-task transferability. To address these challenges, we propose PromptGFM, a versatile GFM for TAGs grounded in graph vocabulary learning. PromptGFM comprises two key compone
    
[^180]: AgentRM：通过奖励建模增强智能体的泛化能力

    AgentRM: Enhancing Agent Generalization with Reward Modeling

    [https://arxiv.org/abs/2502.18407](https://arxiv.org/abs/2502.18407)

    本论文提出可泛化奖励模型AgentRM，发现微调奖励模型来引导测试时搜索比直接微调策略模型更稳健，在九个智能体任务上平均提升8.8分并超越最强通用智能体4.0分。

    

    现有的基于大语言模型（LLM）的智能体在已见任务上取得了强大的性能，但它们对未见任务的泛化能力仍然较差。因此，近期的一些工作专注于使用更多样化的任务对策略模型进行微调以提升泛化能力。在这项工作中，我们发现微调一个奖励模型来引导策略模型，比直接微调策略模型更加稳健。基于这一发现，我们提出了AgentRM，一个可泛化的奖励模型，用于引导策略模型进行有效的测试时搜索。我们全面研究了构建奖励模型的三种方法，包括显式奖励建模、隐式奖励建模以及LLM作为评判者（LLM-as-a-judge）。随后，我们使用AgentRM通过Best-of-N采样和步级束搜索来引导答案生成。在四类共九个智能体任务上，AgentRM使基础策略模型平均提升了8.8分，超越了最顶尖的通用智能体4.0分。此外，它……

    arXiv:2502.18407v2 Announce Type: replace-cross  Abstract: Existing LLM-based agents have achieved strong performance on held-in tasks, but their generalizability to unseen tasks remains poor. Hence, some recent work focus on fine-tuning the policy model with more diverse tasks to improve the generalizability. In this work, we find that finetuning a reward model to guide the policy model is more robust than directly finetuning the policy model. Based on this finding, we propose AgentRM, a generalizable reward model, to guide the policy model for effective test-time search. We comprehensively investigate three approaches to construct the reward model, including explicit reward modeling, implicit reward modeling and LLM-as-a-judge. We then use AgentRM to guide the answer generation with Best-of-N sampling and step-level beam search. On four types of nine agent tasks, AgentRM enhances the base policy model by $8.8$ points on average, surpassing the top general agent by $4.0$. Moreover, it
    
[^181]: LDC：通过动态控制学习生成研究想法

    LDC: Learning to Generate Research Idea with Dynamic Control

    [https://arxiv.org/abs/2412.14626](https://arxiv.org/abs/2412.14626)

    提出首个结合监督微调与可控强化学习的两阶段框架，利用多维奖励模型和细粒度反馈动态控制研究想法生成，从而在新颖性、可行性和有效性之间实现平衡，提升大语言模型科研构思质量。

    

    arXiv:2412.14626v3 公告类型：交叉替换 摘要：大型语言模型（LLMs）的最新进展已展现出其在自动化科学研究构思方面的潜力。现有方法主要侧重于提示技术，往往产生的想法与专家标准不符——即新颖性、可行性和有效性，这些被研究界广泛认可为高质量研究想法的三个关键子维度。此外，由于这些维度之间存在固有的权衡关系，平衡这些维度仍然具有挑战性。为解决这些局限性，我们提出了首个采用两阶段方法的框架，结合监督微调（SFT）和可控强化学习（RL）来完成该任务。在SFT阶段，模型从研究论文及其对应后续想法的配对数据中学习基础模式。在RL阶段，由细粒度反馈引导的多维奖励模型在关键维度上对模型进行评估和优化。

    arXiv:2412.14626v3 Announce Type: replace-cross  Abstract: Recent advancements in large language models (LLMs) have demonstrated their potential in automating the scientific research ideation. Existing approaches primarily focus on prompting techniques, often producing ideas misaligned with expert standards - novelty, feasibility, and effectiveness, which are widely recognized by the research community as the three key subdimensions of high-quality ideas. Also, balancing these dimensions remains challenging due to their inherent trade-offs. To address these limitations, we propose the first framework that employs a two-stage approach combining Supervised Fine-Tuning (SFT) and controllable Reinforcement Learning (RL) for the task. In the SFT stage, the model learns foundational patterns from pairs of research papers and their corresponding follow-up ideas. In the RL stage, multi-dimensional reward models guided by fine-grained feedback evaluate and optimize the model across key dimensio
    
[^182]: 基于意图感知提示的对话式心理操纵检测

    Detecting Conversational Mental Manipulation with Intent-Aware Prompting

    [https://arxiv.org/abs/2412.08414](https://arxiv.org/abs/2412.08414)

    提出意图感知提示（IAP）方法，利用大语言模型捕捉对话参与者的潜在意图来检测心理操纵，在MentalManip数据集上表现优于其他提示策略，并显著减少了假阴性。

    

    心理操纵通过隐蔽且负面的方式扭曲人们的决策，严重损害心理健康。尽管自然语言处理领域对心理健康的关注日益增加，但由于检测对话中细微、隐蔽的操纵策略十分复杂，在应对操纵方面的进展仍然有限。在本文中，我们提出了意图感知提示，这是一种利用大语言模型检测心理操纵的新方法，通过捕捉对话参与者的潜在意图，从而对操纵策略提供更深入的理解。在MentalManip数据集上的实验结果表明，IAP相比其他先进的提示策略具有更优的效果。值得注意的是，我们的方法大幅减少了假阴性，有助于检测出更多的心理操纵实例，同时将对正例的误判降到最低。本文的代码已公开。

    arXiv:2412.08414v2 Announce Type: replace  Abstract: Mental manipulation severely undermines mental wellness by covertly and negatively distorting decision-making. While there is an increasing interest in mental health care within the natural language processing community, progress in tackling manipulation remains limited due to the complexity of detecting subtle, covert tactics in conversations. In this paper, we propose Intent-Aware Prompting (IAP), a novel approach for detecting mental manipulations using large language models (LLMs), providing a deeper understanding of manipulative tactics by capturing the underlying intents of participants. Experimental results on the MentalManip dataset demonstrate superior effectiveness of IAP against other advanced prompting strategies. Notably, our approach substantially reduces false negatives, helping detect more instances of mental manipulation with minimal misjudgment of positive cases. The code of this paper is available.
    
[^183]: 语法对齐解码

    Grammar-Aligned Decoding

    [https://arxiv.org/abs/2405.21047](https://arxiv.org/abs/2405.21047)

    本文揭示了语法约束解码会扭曲大语言模型的输出分布，导致生成结果虽符合语法但质量低下，并提出了一种名为ASAp的语法对齐解码算法来解决这一问题。

    

    大语言模型（LLMs）难以可靠地生成高度结构化的输出，例如程序代码、数学公式或格式良好的标记语言。约束解码方法通过在每个步骤贪心地限制LLM可以输出的token来缓解这一问题，从而保证输出符合给定的约束。具体而言，在语法约束解码（GCD）中，LLM的输出必须遵循给定的语法。在本文中，我们证明了GCD技术（以及广义上的约束解码技术）会扭曲LLM的分布，导致输出虽然符合语法，但其出现的概率与LLM本身给出的概率不成比例，因此最终质量较低。我们将采样与语法约束对齐这一问题称为语法对齐解码（GAD），并提出了一种名为ASAp（基于近似期望未来的自适应采样）的解码算法，该算法保证输出……

    arXiv:2405.21047v4 Announce Type: replace  Abstract: Large Language Models (LLMs) struggle with reliably generating highly structured outputs, such as program code, mathematical formulas, or well-formed markup. Constrained decoding approaches mitigate this problem by greedily restricting what tokens an LLM can output at each step to guarantee that the output matches a given constraint. Specifically, in grammar-constrained decoding (GCD), the LLM's output must follow a given grammar. In this paper, we demonstrate that GCD techniques (and in general constrained decoding techniques) can distort the LLM's distribution, leading to outputs that are grammatical but appear with likelihoods that are not proportional to the ones given by the LLM, and so ultimately are low-quality. We call the problem of aligning sampling with a grammar constraint, grammar-aligned decoding (GAD), and propose adaptive sampling with approximate expected futures (ASAp), a decoding algorithm that guarantees the outpu
    

