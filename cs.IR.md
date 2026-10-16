# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CARA: Causally Guided Retrieval Agent for Question Answering](https://arxiv.org/abs/2610.22017) | CARA是一个无需训练的因果引导检索智能体，通过后果性和本质性两个因果准则评估证据相关性并结合路径一致性引导知识图谱探索，克服了现有方法仅依赖文本相似性的局限。 |
| [^2] | [GACC-Bench: Benchmarking Goal-Adaptive Context Compression Ability across Large Language Models](https://arxiv.org/abs/2610.21893) | 该论文提出GACC-Bench基准，通过隐藏下游问题并仅给定高级目标与长度预算，评估大语言模型能否在未知具体问题的情况下可靠地进行目标自适应的上下文压缩。 |
| [^3] | [Aligning LLMs with Sparse Retrieval via Adaptive Sparse Supervision](https://arxiv.org/abs/2610.21877) | 该论文提出了一种基于LLM的稀疏检索框架，通过基于SPLADE累积激活质量的动态α-截断策略实现输入自适应的稀疏监督，兼顾大模型的语义表示能力与稀疏检索的效率。 |
| [^4] | [Retrieved but Not Read: Understanding Why Search-Augmented LLMs Fail to Use Available Evidence](https://arxiv.org/abs/2610.21731) | 本研究揭示了搜索增强型大语言模型的失效机制：模型常只凭搜索摘要作答而不阅读源网页，且即使检索证据已影响内部预测，最终答案仍可能偏向过时信息。 |
| [^5] | [VoteGEO-Bench: Measuring the Impact of Generative Engine Optimization on Electoral Visibility](https://arxiv.org/abs/2610.21330) | 该论文提出VoteGEO-Bench基准，覆盖美日两国选举场景，系统评估了11种生成式引擎优化（GEO）改写策略对LLM生成回答中政治信息可见性的影响，揭示了选举信息可能被低成本来源与改写策略操纵的潜在风险。 |
| [^6] | [Making every stage of medical device document assessment with large language models deterministic](https://arxiv.org/abs/2610.21300) | 研究发现十四个大语言模型商业部署在确定性采样设置下仍无法对医疗设备文档评估给出一致结果，作者据此构建了一个在每个阶段都具备确定性的评估流水线。 |
| [^7] | [TRACER: Trajectory-Aware Tool Retrieval for LLM Agents](https://arxiv.org/abs/2610.21267) | TRACER通过在字典树中索引工具调用序列，并利用经验与上下文先验的学习混合对候选工具排序，在仅暴露极少量工具的情况下，将大语言模型智能体的输入token减少高达80%，同时显著提升工具选择准确率。 |
| [^8] | [RecSoup: Efficient Model Merging Approach for Controllable Multi-Objective Recommendation](https://arxiv.org/abs/2610.21220) | 提出RecSoup，一种基于专门化模型合并的轻量级两阶段方法，通过剪枝策略缓解跨目标参数干扰，实现无需重新训练即可在推理时动态调整目标权衡的可控多目标推荐。 |
| [^9] | [SSR: Structural Signal Retriever for Scalable Agent Discovery in LLM-Orchestrated Multi-Agent Systems](https://arxiv.org/abs/2610.21113) | 提出了一种无需训练的确定性检索方法SSR，通过从智能体卡中提取结构化词法信号、结合IDF加权与倒排信号索引，实现了LLM编排多智能体系统中可扩展、低成本的智能体发现。 |
| [^10] | [ReviewPilot: A Multi-Agent System for Screening and Data Extraction in Systematic Reviews](https://arxiv.org/abs/2610.21042) | 提出对话式多智能体系统ReviewPilot，通过可对话定制的标准与全程可追溯的决策来源实现PRISMA筛选与数据提取流程自动化，在流行病学专家综述上显著优于单提示词LLM和AgentSLR等方法。 |
| [^11] | [Learning When to Think: Utility-Guided Adaptive Reasoning for Efficient Text Retrieval](https://arxiv.org/abs/2610.21030) | 论文提出AURA框架，通过直接监督中间推理状态区分相关文档与困难干扰项，并利用轻量级效用估计器在推理时自适应调整推理深度，从而在BRIGHT基准上以极少推理步骤和生成token实现高检索精度并大幅降低延迟。 |
| [^12] | [TransX Technical Report: Scaling Sequence Transduction for Recommendation](https://arxiv.org/abs/2610.20913) | TransX是一个面向工业推荐的序列转导框架与配套基础设施，通过将序列编码、行为记忆和候选条件融合解耦为可独立扩展的组件，结合训练阶段的计算摊销与服务阶段的预填充-刷新-预测解耦，实现了推荐模型的高效扩展，并揭示了编码器深度、监督信号与记忆分辨率之间的各向异性扩展规律。 |
| [^13] | [Idea2Prior: Auditable Paper Retrieval and Ranking from Research Ideas](https://arxiv.org/abs/2610.20857) | 提出了构建于PaperArena之上的可审计文献检索系统Idea2Prior，能从非正式研究想法出发，通过多源发现、身份验证、分面摘要评估、语义引用扩展与Expert联合排序来检索和排序论文，其Hit@5达53.3%，超过ChatGPT和Gemini的Deep Research。 |
| [^14] | [Information-Gain Calibrated On-Policy Distillation for Deep Search Agents](https://arxiv.org/abs/2610.20664) | 该论文揭示了标准在线策略蒸馏中因忽视检索证据而产生的“证据盲监督”失败模式及其背后的预算先验与方向先验两种隐式偏差，并提出用信息增益来校准监督信号，从而改进深度搜索智能体的蒸馏训练。 |
| [^15] | [Knowledge Graphs as Files: An Agent Harness Where Tool Creation and Tool Use Co-Evolve](https://arxiv.org/abs/2610.18616) | 提出DKGI框架，让语言模型智能体将知识图谱视为完整的节点和边文件直接交互，通过编写Python代码自行创建和修订图访问工具，实现工具创建与工具使用的协同演化，在三个领域的基准上无需训练即超越最强基线至少6.7个Hit@1点。 |
| [^16] | [When Inputs Receive No Score: Evaluating Open-Web Event Admission on the Complete Stream](https://arxiv.org/abs/2610.17814) | 该研究提出在完整数据流（包括处理流水线无法评分的输入）上评估开放网络事件条目准入，发现评分可用性会改变排序比较结果并使可实现的保留率存在上限，导致90%全流目标无法仅靠阈值调整达成。 |
| [^17] | [Self-improving data synthesis can scale embedding models across domains and tasks](https://arxiv.org/abs/2610.14730) | 提出了SYNTRA自改进检索数据合成方法，仅需无标注语料库和少量查询-文档对，通过自改进循环迭代优化合成指令，在多数任务上超越人工校准，并使嵌入模型随数据规模扩展在15个领域持续提升性能。 |
| [^18] | [KoViDoRe: Korean Visual Document Retrieval](https://arxiv.org/abs/2608.20840) | 本文提出了KoViDoRe，一个针对韩语视觉文档检索的基准数据集，通过多阶段数据整理流程解决了现有基准中韩语复杂文档覆盖不足和单页检索局限的问题。 |
| [^19] | [The MPB Corpus: A Dataset of Melody, Rhythm, Harmony, and Melody-Harmony Relationships in Brazilian Popular Music](https://arxiv.org/abs/2608.13842) | 该论文构建了首个包含500首巴西流行音乐、覆盖旋律、节奏、和声及旋律-和声关系的最全面数据集，并提供了分析工具和探索性研究，以支持计算音乐学分析。 |
| [^20] | [MemSyco-Bench: Benchmarking Sycophancy in Agent Memory](https://arxiv.org/abs/2607.01071) | 提出MemSyco-Bench基准，系统评估智能体记忆诱发的谄媚问题——即检索记忆导致智能体过度迎合用户而牺牲事实准确性与客观推理，涵盖五项任务衡量记忆何时应影响决策及如何正确使用有效记忆。 |
| [^21] | [Multi-Agent Transactive Memory](https://arxiv.org/abs/2606.19911) | 提出多智能体交互记忆（MATM）框架，通过群体级的共享存储与检索机制，让智能体能够复用彼此的运行轨迹中所蕴含的程序性知识，从而提升任务执行效率。 |
| [^22] | [GRASP: Plan-Guided Graph Retrieval with Adaptive Fusion and Reranking on Semi-Structured Knowledge Bases](https://arxiv.org/abs/2605.30237) | GRASP通过小型LLM规划器生成可执行图计划、融合图排序与文本排序，并用强化学习优化规划器以最大化融合排序增益，在半结构化知识库检索上取得全面领先的性能，平均Hit@1比最强基线提升18.6个百分点且单查询仅需2.47秒。 |
| [^23] | [What Tracks Hard-Negative Transfer? A Study of Lexical and Semantic Signals](https://arxiv.org/abs/2603.20990) | 本研究发现在训练前预测稠密检索器迁移效果时，基于IDF加权的查询词元非覆盖率指标（平均词汇残差度）能够稳定重现难负样本来源的迁移排序，而冻结编码器的各类语义统计量无论单独使用还是与词汇信号结合均无法做到这一点。 |
| [^24] | [GraphCompliance: Structure-Aware Regulatory Compliance through Policy-Context Graph Alignment](https://arxiv.org/abs/2510.26309) | 提出GraphCompliance框架，将监管文档建模为模式引导的策略图、将被评估情境建模为实体关系上下文图，并通过合规门控进行图对齐，从而实现结构感知的监管合规自动评估。 |
| [^25] | [How Do LLM-Generated Texts Impact Term-Based Retrieval Models?](https://arxiv.org/abs/2508.17715) | 本文通过语言学分析发现LLM生成的文本具有独特的词频分布、更高的词项特异性和文档多样性，并据此研究了这些特征如何影响以BM25为代表的基于词项的检索模型的表现。 |
| [^26] | [Enhancing Knowledge Density via Hierarchical Context Pruning for Retrieval-Augmented Repository-Level Code Completion](https://arxiv.org/abs/2508.10068) | 提出SaraCoder框架，在固定上下文预算下通过分层上下文剪枝提升检索上下文的知识密度，避免冗余代码浪费token并释放跨文件依赖与相似代码片段的协同作用，从而提升仓库级代码补全性能。 |

# 详细

[^1]: CARA：因果引导的问答检索智能体

    CARA: Causally Guided Retrieval Agent for Question Answering

    [https://arxiv.org/abs/2610.22017](https://arxiv.org/abs/2610.22017)

    CARA是一个无需训练的因果引导检索智能体，通过后果性和本质性两个因果准则评估证据相关性并结合路径一致性引导知识图谱探索，克服了现有方法仅依赖文本相似性的局限。

    

    在文本丰富的知识图谱上进行检索增强生成，需要识别对回答问题有贡献的证据，而不仅仅是匹配其文本内容。现有的智能体检索器虽然能够自适应地平衡全局搜索与多跳探索，但主要依赖文本相似性来引导检索。我们提出了CARA，一个无需训练的、因果引导的检索智能体，它通过两个互补的准则显式评估所检索证据的因果相关性：后果性，衡量添加证据是否能提高问题的可回答性；本质性，评估移除证据是否会削弱问题的可回答性。CARA将这些评估与路径一致性相结合以引导图谱探索，并采用倒数排名融合来聚合来自多个智能体的证据。我们在三个STaRK数据集上评估了CARA，并利用真实答案对STaRK进行了扩充，以支持下游问答任务的评估。

    arXiv:2610.22017v1 Announce Type: new  Abstract: Retrieval-augmented generation over text-rich knowledge graphs requires identifying evidence that contributes to answering a question, rather than merely matching its textual content. Existing agentic retrievers adaptively balance global search and multi-hop exploration but primarily rely on textual similarity to guide retrieval. We propose CARA, a training-free, CAusally guided Retrieval Agent that explicitly evaluates the causal relevance of retrieved evidence through two complementary criteria: consequentiality, measuring whether adding evidence improves answerability, and essentiality, assessing whether removing evidence weakens answerability. CARA integrates these assessments with path consistency to guide graph exploration and employs reciprocal rank fusion to aggregate evidence from multiple agents. We evaluate CARA on three STaRK datasets and augment STaRK with ground-truth answers to enable downstream question answering evaluati
    
[^2]: GACC-Bench：跨大语言模型的目标自适应上下文压缩能力基准测试

    GACC-Bench: Benchmarking Goal-Adaptive Context Compression Ability across Large Language Models

    [https://arxiv.org/abs/2610.21893](https://arxiv.org/abs/2610.21893)

    该论文提出GACC-Bench基准，通过隐藏下游问题并仅给定高级目标与长度预算，评估大语言模型能否在未知具体问题的情况下可靠地进行目标自适应的上下文压缩。

    

    在大语言模型（LLM）的长时间运行应用中，上下文压缩必须保留与目标相关的信息，而这往往发生在具体下游问题已知之前。我们提出了GACC-Bench，一个用于目标自适应上下文压缩的基准，用于评估大语言模型根据指定高级目标可靠地调整信息保留策略的能力。对于每个源上下文，我们通过根据共享的下游目标对基于源文本的问题进行分组，推导出一个或多个高级目标。该基准包含500个目标和2,571个问题，涵盖来自七个集合、跨越多个领域的281个文档和智能体轨迹上下文，所有实例均经过人类专家审核。压缩器接收源上下文、目标和上下文长度预算，而相关的评估问题则被隐藏。固定的阅读器使用相同的压缩上下文逐一回答这些问题。我们对十个LLM压缩器的评估揭示了实质性……

    arXiv:2610.21893v1 Announce Type: new  Abstract: Context compression in long-running applications of large language models (LLMs) must preserve goal-relevant information, often before specific downstream questions are known. We introduce GACC-Bench, a benchmark for Goal-Adaptive Context Compression that evaluates how reliably LLMs adapt information retention to specified high-level goals. For each source context, we derive one or more high-level goals by grouping source-grounded questions according to shared downstream objectives. The benchmark contains 500 goals and 2,571 questions over 281 document and agent-trace contexts from seven collections spanning diverse domains, with all instances reviewed by human experts. The compressor receives a source context, a goal, and a context-length budget, with the associated evaluation questions withheld. A fixed reader answers these questions individually using the same compressed context. Our evaluation of ten LLM compressors reveals substanti
    
[^3]: 通过自适应稀疏监督将大语言模型与稀疏检索对齐

    Aligning LLMs with Sparse Retrieval via Adaptive Sparse Supervision

    [https://arxiv.org/abs/2610.21877](https://arxiv.org/abs/2610.21877)

    该论文提出了一种基于LLM的稀疏检索框架，通过基于SPLADE累积激活质量的动态α-截断策略实现输入自适应的稀疏监督，兼顾大模型的语义表示能力与稀疏检索的效率。

    

    尽管大语言模型（LLM）具备强大的语义表示能力，但将其应用于第一阶段检索会带来显著的延迟和存储成本。诸如SPLADE之类的神经稀疏检索器虽然能提供高效检索，但依赖于表示能力有限的相对紧凑的编码器。为了弥合这一差距，我们提出了一种基于LLM的稀疏检索框架，实现输入自适应的稀疏对齐。具体而言，我们引入了一种动态α-截断策略，该策略基于SPLADE表示的累积激活质量来选择特定于输入的监督支持。该策略引导LLM在保留显著词汇信号的同时抑制信息量较低的激活，而无需依赖固定的支持预算。随后的检索微调结合了对比学习、知识蒸馏和稀疏性正则化，以提升检索效果。在MS MARCO及六个选定数据集上的实验……

    arXiv:2610.21877v1 Announce Type: new  Abstract: While large language models (LLMs) offer strong semantic representations, their application to first-stage retrieval incurs substantial latency and storage costs. Neural sparse retrievers such as SPLADE provide efficient retrieval but rely on relatively compact encoders with limited representational capacity. To bridge this gap, we propose an LLM-based sparse retrieval framework with input-adaptive sparse alignment. Specifically, we introduce a dynamic $\alpha$-truncation strategy that selects input-specific supervision supports based on the cumulative activation mass of SPLADE representations. This strategy guides the LLM to preserve salient lexical signals while suppressing less informative activations, without relying on a fixed support budget. Subsequent retrieval fine-tuning combines contrastive learning, knowledge distillation, and sparsity regularization to improve retrieval effectiveness. Experiments on MS MARCO and six selected 
    
[^4]: 已检索但未阅读：理解搜索增强型大语言模型为何无法利用可用证据

    Retrieved but Not Read: Understanding Why Search-Augmented LLMs Fail to Use Available Evidence

    [https://arxiv.org/abs/2610.21731](https://arxiv.org/abs/2610.21731)

    本研究揭示了搜索增强型大语言模型的失效机制：模型常只凭搜索摘要作答而不阅读源网页，且即使检索证据已影响内部预测，最终答案仍可能偏向过时信息。

    

    网络搜索使大语言模型（LLM）能够获取最新信息，但为什么它们在相关证据可用时有时仍会产生过时的答案，这一点尚不清楚。为了研究这个问题，我们构建了在模型训练截止日期之后其值发生变化的时间性事实。我们评估了一个部署网络搜索的专有模型，以及在受控检索条件下测试的六个开源权重模型配置。我们发现，模型经常只根据搜索摘要作答而不打开源网页，这与更高的过时答案率和不可信引用率相关。我们的受控实验和逐层分析揭示了证据可用性与有效利用之间的差距：即使检索到的证据已经影响了模型的内部预测，模型在最终答案中仍可能偏向过时的信息。我们希望这些发现有助于推进关于搜索增强型大语言模型中可靠使用检索证据的研究。

    arXiv:2610.21731v1 Announce Type: cross  Abstract: Web search allows LLMs to access up-to-date information, but it remains unclear why they sometimes produce outdated answers even when relevant evidence is available. To study this problem, we construct temporal facts whose values changed after the models' training cutoffs. We evaluate one proprietary model under deployed web search and six open-weight model configurations under controlled retrieval. We find that models often answer from search snippets without opening the source pages, which is associated with higher rates of outdated answers and unfaithful citations. Our controlled experiments and layer-wise analysis reveal a gap between the availability of evidence and its effective use. Even when retrieved evidence influences internal predictions, models may still favour outdated information in their final answers. We hope these findings will help advance research on the reliable use of retrieved evidence in search-augmented LLMs.
    
[^5]: VoteGEO-Bench：衡量生成式引擎优化对选举可见性的影响

    VoteGEO-Bench: Measuring the Impact of Generative Engine Optimization on Electoral Visibility

    [https://arxiv.org/abs/2610.21330](https://arxiv.org/abs/2610.21330)

    该论文提出VoteGEO-Bench基准，覆盖美日两国选举场景，系统评估了11种生成式引擎优化（GEO）改写策略对LLM生成回答中政治信息可见性的影响，揭示了选举信息可能被低成本来源与改写策略操纵的潜在风险。

    

    大型语言模型（LLM）日益被用于获取选举信息，但外部网页内容如何塑造其所呈现的政治信息仍鲜为人知。生成式引擎优化（GEO）通过改写网页内容来提升其在生成式回答中的可见性。为探究其在投票场景中的潜在风险，我们提出了VoteGEO-Bench，一个覆盖2024年美国总统大选和2026年日本众议院选举的基准，其中结合了选举相关查询、历史网页和选民画像。我们首先分析了生成式引擎所引用的来源，发现许多引用来自成本较低的自有来源和开放平台来源，而部分网页能够在多天内持续保持可见。随后，我们评估了4大类共11种改写策略，考察了目标指定方式、跨引擎与跨选举的有效性以及选民画像所起的作用。未指定目标的改写（摘要原文在此处截断）……

    arXiv:2610.21330v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used for election information, yet little is known about how external web content shapes the political information they present. Generative engine optimization (GEO) rewrites web content to improve its visibility in generated responses. To investigate its potential risks in voting scenarios (VoteGEO), we introduce VoteGEO-Bench, a benchmark covering the 2024 U.S. presidential election and the 2026 Japanese House of Representatives election, combining election-related queries, historical webpages, and voter personas. We first analyze sources cited by generative engines and find that many citations come from lower-cost owned and open-platform sources, while some webpages remain visible across days. We then evaluate 11 rewriting strategies in 4 categories, examining target specification, effectiveness across engines and elections, and the role of voter personas. Rewriting without specifying a 
    
[^6]: 利用大语言模型使医疗设备文档评估的每个阶段都具备确定性

    Making every stage of medical device document assessment with large language models deterministic

    [https://arxiv.org/abs/2610.21300](https://arxiv.org/abs/2610.21300)

    研究发现十四个大语言模型商业部署在确定性采样设置下仍无法对医疗设备文档评估给出一致结果，作者据此构建了一个在每个阶段都具备确定性的评估流水线。

    

    为了证明医疗设备符合法规要求，专家需要判断文档和记录是否满足每一项要求，并将判断结果记录下来供他人核查。对于专家定义的每个评估项目，大语言模型可以准备一份初步评估：诸如“满足”或“不满足”的标签、理由以及引用的证据。我们向十四个此类模型的商业部署（来自三家供应商的七个模型，通过六种途径）重复发送相同的请求，并使用各自接受的确定性采样设置；每个部署至少对一个请求返回了不同的评估结果，其中十二个部署的标签发生了变化。在接口允许的情况下减少推理并不能减少这种变化，而使用检索段落的提示仅对最小的模型消除了这种变化。同一个开放权重模型是否可复现，仅取决于其部署服务的方式。我们构建的一个在每个阶段都确定性的流水线——

    arXiv:2610.21300v1 Announce Type: new  Abstract: To show that a medical device complies with regulation, experts judge whether documents and records meet each requirement and record the judgment for others to verify. For each assessment item that experts define, large language models could prepare a preliminary assessment: a label such as Satisfied or Not satisfied, a reason and quoted evidence. We sent fourteen commercial deployments of such models (seven models from three providers, through six routes) the same request repeatedly, with the deterministic sampling settings each accepted; every deployment returned differing assessments for at least one request, and in twelve the label changed. Reducing reasoning where the interface allows it did not reduce the variation, and prompts of retrieved passages removed it only for the smallest models. The same open-weight model was reproducible or not depending only on how it was served. A pipeline we built to be deterministic at every stage -
    
[^7]: TRACER：面向大语言模型智能体的轨迹感知工具检索

    TRACER: Trajectory-Aware Tool Retrieval for LLM Agents

    [https://arxiv.org/abs/2610.21267](https://arxiv.org/abs/2610.21267)

    TRACER通过在字典树中索引工具调用序列，并利用经验与上下文先验的学习混合对候选工具排序，在仅暴露极少量工具的情况下，将大语言模型智能体的输入token减少高达80%，同时显著提升工具选择准确率。

    

    现代智能体需要从工具注册表中选择工具，但暴露整个注册表会带来高昂的token成本，且随着注册表规模的增长可能损害选择质量。我们提出了TRACER，这是一个检索前端，它在字典树（trie）中索引观察到的工具调用序列，并使用经验先验与上下文先验的学习混合方式对后续候选工具进行排序。在一个零售与电信混合的τ²环境中，TRACER将GPT-5.4的工具选择准确率提高了4.7个百分点，同时将每步输入token减少了80%，仅暴露128个工具中的5个。在自主的完整回合执行中，它以每步减少61%的输入token实现了与使用全注册表的GPT-5.4相当的任务成功率。在六个环境和四个模型的测试中，TRACER在24个模型-环境对中的21个上将工具选择准确率提升至超过全注册表选择的水平，平均提升6到10个百分点；三个例外情况最多落后2.2个百分点。我们分析了两种模式下的失败案例，以确定未来的研究方向。

    arXiv:2610.21267v1 Announce Type: cross  Abstract: Modern agents select tools from registries, but exposing entire registries incurs high token costs and can impair selection as they grow. We present TRACER, a retrieval front-end that indexes observed tool-call sequences in a trie and ranks continuations using a learned blend of empirical and contextual priors. On a mixed retail-and-telecom $\tau^2$ environment, TRACER improves GPT-5.4's tool-selection accuracy by 4.7 points while reducing input tokens per step by 80%, exposing just 5 of 128 tools. In autonomous full-turn execution, it matches GPT-5.4's full-registry task success with 61% fewer input tokens per step. Across six environments and four models, TRACER improves tool-selection accuracy over full-registry selection in 21 of 24 model-environment pairs, with average gains of 6 to 10 points; the three exceptions trail by at most 2.2. We analyze failures in both regimes to identify directions for future work.
    
[^8]: RecSoup：一种面向可控多目标推荐的高效模型合并方法

    RecSoup: Efficient Model Merging Approach for Controllable Multi-Objective Recommendation

    [https://arxiv.org/abs/2610.21220](https://arxiv.org/abs/2610.21220)

    提出RecSoup，一种基于专门化模型合并的轻量级两阶段方法，通过剪枝策略缓解跨目标参数干扰，实现无需重新训练即可在推理时动态调整目标权衡的可控多目标推荐。

    

    推荐系统通常需要平衡多个相互冲突的目标。可控多目标推荐（CMOR）能够在推理阶段动态调整目标间的权衡，而无需重新训练，从而适应不断变化的用户或系统需求。现有的CMOR方法通常依赖于输出层的重排序或参数生成模块，这些方法要么限制了可行权衡的范围，要么带来高昂的训练和推理成本。模型合并通过参数组合的方式复用多个面向特定目标的模型，提供了一种有前景的替代方案。然而，由于缺乏共享的线性区域以及跨目标的参数干扰，将其直接应用于推荐系统面临挑战。为解决这些问题，我们提出了RecSoup，一种通过专门化模型合并实现有效CMOR的轻量级两阶段方法。RecSoup首先通过剪枝策略缓解跨目标的参数干扰……

    arXiv:2610.21220v1 Announce Type: cross  Abstract: Recommender systems often need to balance multiple conflicting objectives. Controllable Multi-Objective Recommendation (CMOR) enables dynamic adjustment of objective trade-offs at inference time without retraining, adapting to shifting user or system requirements. Existing CMOR approaches typically rely on output-level reranking or parameter-generating modules, which either limit the scope of feasible trade-offs or incur substantial training and inference cost. Model merging provides a promising alternative by reusing multiple objective-specific models via parameter combination. However, its direct application to recommender systems is challenging due to the lack of a shared linear regime and cross-objective interference. To address these issues, we propose RecSoup, a lightweight two-stage approach for effective CMOR via specialized model merging. RecSoup first mitigates parameter interference across objectives through a pruning strate
    
[^9]: SSR：面向LLM编排多智能体系统中可扩展智能体发现的结构信号检索器

    SSR: Structural Signal Retriever for Scalable Agent Discovery in LLM-Orchestrated Multi-Agent Systems

    [https://arxiv.org/abs/2610.21113](https://arxiv.org/abs/2610.21113)

    提出了一种无需训练的确定性检索方法SSR，通过从智能体卡中提取结构化词法信号、结合IDF加权与倒排信号索引，实现了LLM编排多智能体系统中可扩展、低成本的智能体发现。

    

    大规模多智能体生态系统需要在不向大型语言模型（LLM）编排器暴露整个注册表的前提下发现合适智能体的机制。挑战不仅在于从不断增长的注册表中检索相关的智能体卡，还在于维护紧凑的候选上下文，因为每增加一个候选都会增大提示词规模、推理成本、延迟，以及超出模型上下文窗口的风险。我们将该问题表述为候选上下文可扩展性问题，并提出了结构信号检索器（SSR），这是一种为智能体卡发现设计的确定性、无需训练的检索方法。SSR从承载能力的智能体卡字段中提取结构化词法信号，使用逆文档频率（IDF）对其进行加权，并利用倒排信号索引（ISI）进行选择性候选评分。我们通过合成智能体基准、公开的MS MARCO改编数据集以及端到端的LLM编排对SSR进行了评估。

    arXiv:2610.21113v1 Announce Type: cross  Abstract: Large-scale multi-agent ecosystems require mechanisms for discovering suitable agents without exposing entire registries to a large language model (LLM) orchestrator. The challenge is not only to retrieve relevant Agent Cards from a growing registry, but also to maintain a compact candidate context, since each additional candidate increases prompt size, inference cost, latency, and the risk of exceeding the model context window. We formulate this problem as candidate-context scalability and propose the Structural Signal Retriever (SSR), our deterministic and training-free retrieval method designed for Agent Card discovery. SSR derives structural lexical signals from capability-bearing Agent Card fields, weights them using inverse document frequency (IDF), and uses an Inverted Signal Index (ISI) for selective candidate scoring. We evaluate SSR using a synthetic agent benchmark, a public MS MARCO adaptation, and end-to-end LLM orchestrat
    
[^10]: ReviewPilot：用于系统综述筛选与数据提取的多智能体系统

    ReviewPilot: A Multi-Agent System for Screening and Data Extraction in Systematic Reviews

    [https://arxiv.org/abs/2610.21042](https://arxiv.org/abs/2610.21042)

    提出对话式多智能体系统ReviewPilot，通过可对话定制的标准与全程可追溯的决策来源实现PRISMA筛选与数据提取流程自动化，在流行病学专家综述上显著优于单提示词LLM和AgentSLR等方法。

    

    系统性文献综述遵循PRISMA流程，即筛选学术文献并从纳入的研究中提取证据。基于大语言模型（LLM）的深度研究工具能够搜索和总结文献，但不会记录哪些研究被排除，也不会标注提取数值的来源。我们开发了一个名为ReviewPilot的对话式多智能体系统，将PRISMA工作流从论文筛选到信息提取实现自动化。研究人员可以通过对话定制筛选标准和提取模式，项目记忆会保存其配置和修订记录。每项筛选决定都附有理由，每个提取值都标明来源。在流行病学专家综述上，ReviewPilot在三种大语言模型上的表现均优于单提示词LLM和AgentSLR，筛选宏观F1最高达0.776，提取F1最高达0.788。在生物医学和人机交互领域的案例研究中，其结果与人类标注者高度一致。

    arXiv:2610.21042v1 Announce Type: cross  Abstract: Systematic literature reviews follow the PRISMA process of screening scholarly works and extracting evidence from included studies. Deep research tools based on large language models (LLMs) can search and summarize the literature but do not record which studies are excluded or where extracted values come from. We develop a conversational multi-agent system called ReviewPilot that automates the PRISMA workflow from paper screening to information extraction. Researchers can tailor criteria and schemas through dialogue, and a project memory keeps their configurations and revisions. Each screening decision comes with its reason and each extracted value with its source. On epidemiological expert reviews, ReviewPilot outperforms a single-prompt LLM and AgentSLR across three LLMs, with screening macro F1 up to 0.776 and extraction F1 up to 0.788. Case studies in biomedicine and human-computer interaction show close agreement with human annota
    
[^11]: 学习何时思考：效用引导的自适应推理以实现高效文本检索

    Learning When to Think: Utility-Guided Adaptive Reasoning for Efficient Text Retrieval

    [https://arxiv.org/abs/2610.21030](https://arxiv.org/abs/2610.21030)

    论文提出AURA框架，通过直接监督中间推理状态区分相关文档与困难干扰项，并利用轻量级效用估计器在推理时自适应调整推理深度，从而在BRIGHT基准上以极少推理步骤和生成token实现高检索精度并大幅降低延迟。

    

    推理条件化的嵌入可以通过逐步细化语义约束来提升检索效果。然而，额外的推理可能引入过度泛化或语义上有噪声的抽象，从而削弱检索的特异性。现有方法主要通过最终检索结果来监督推理，对细化过程本身的指导有限。我们提出了AURA框架，它根据中间推理状态区分相关文档与困难干扰文档的能力来直接监督这些中间状态。AURA还训练了一个基于前缀条件化嵌入的轻量级效用估计器，以便在推理时自适应调整推理深度。在推理密集型的BRIGHT基准上，AURA-4B实现了34.7的平均nDCG@10，同时每个查询仅使用2.8个推理步骤和96个生成token。在相同的部署条件下，自适应停止机制相比完整轨迹推理将延迟从4.6秒降低至0.76秒。

    arXiv:2610.21030v1 Announce Type: cross  Abstract: Reasoning-conditioned embeddings can improve retrieval by progressively refining semantic constraints. However, additional reasoning may introduce over-generalized or semantically noisy abstractions that weaken retrieval specificity. Existing methods largely supervise reasoning through final retrieval outcomes, providing limited guidance over the refinement process itself. We propose AURA, a framework that directly supervises intermediate reasoning states according to how well they separate relevant documents from hard distractors. AURA further trains a lightweight utility estimator over prefix-conditioned embeddings to adapt reasoning depth at inference time. On the reasoning-intensive BRIGHT benchmark, AURA-4B achieves 34.7 average nDCG@10 while using 2.8 reasoning steps and 96 generated tokens per query. Under matched deployment conditions, adaptive stopping reduces latency from 4.6 to 0.76 seconds relative to full-trajectory reason
    
[^12]: TransX技术报告：面向推荐的序列转导扩展

    TransX Technical Report: Scaling Sequence Transduction for Recommendation

    [https://arxiv.org/abs/2610.20913](https://arxiv.org/abs/2610.20913)

    TransX是一个面向工业推荐的序列转导框架与配套基础设施，通过将序列编码、行为记忆和候选条件融合解耦为可独立扩展的组件，结合训练阶段的计算摊销与服务阶段的预填充-刷新-预测解耦，实现了推荐模型的高效扩展，并揭示了编码器深度、监督信号与记忆分辨率之间的各向异性扩展规律。

    

    我们提出TransX，这是一个序列转导框架及与之协同设计的基础设施，用于扩展工业级推荐系统，并对模型-计算分配与执行摊销进行显式控制。TransX将序列编码、行为记忆和基于候选的融合分离为可独立扩展的组件。在训练阶段，系统在行为前缀之间摊销因果轨迹计算，并将记忆状态扇出至各个候选；在服务阶段，通过紧凑的行为记忆KV，将批量预填充、近线刷新和在线预测进行解耦。我们观察到各向异性的扩展收益：在固定记忆分辨率与融合容量下，编码器扩展可能在中等深度之后趋于饱和；随着监督信号的增加，分配前沿会向基于候选的融合方向移动；而保持质量所需的记忆分辨率随着模型容量的增长大致保持稳定。我们在点击与转化任务上训练TransX，并对其共享……（原文摘要在此处截断）

    arXiv:2610.20913v1 Announce Type: new  Abstract: We present TransX, a sequence-transduction framework and co-designed infrastructure for scaling industrial recommendation with explicit control over model-compute allocation and execution amortization. TransX separates sequence encoding, behavior memory, and candidate-conditioned fusion into independently scalable components. Training amortizes causal trajectory computation across behavioral prefixes and fans out memory states to candidates; serving disaggregates bulk prefill, nearline refresh, and online prediction through compact behavior-memory KV.   We observe anisotropic scaling returns: encoder scaling may saturate beyond moderate depth at fixed memory resolution and fusion capacity; increasing supervision shifts the allocation frontier toward candidate-conditioned fusion; and the memory resolution needed to preserve quality remains roughly stable as model capacity grows. We train TransX on click and conversion, post-train its shar
    
[^13]: Idea2Prior：从研究想法出发的可审计论文检索与排序

    Idea2Prior: Auditable Paper Retrieval and Ranking from Research Ideas

    [https://arxiv.org/abs/2610.20857](https://arxiv.org/abs/2610.20857)

    提出了构建于PaperArena之上的可审计文献检索系统Idea2Prior，能从非正式研究想法出发，通过多源发现、身份验证、分面摘要评估、语义引用扩展与Expert联合排序来检索和排序论文，其Hit@5达53.3%，超过ChatGPT和Gemini的Deep Research。

    

    从一个非正式的研究想法出发寻找相关论文，仅靠匹配主题词是不够的：相关工作可能使用不同的术语，而其独特的机制可能并未体现在摘要中。我们提出了Idea2Prior，一个构建在PaperArena之上的可审计文献检索系统。它结合了多源发现、身份验证、基于分面的摘要评估、语义引用扩展以及与Expert的联合排序。其“仅检索”模式从最多60篇的候选名单中对至多五篇论文进行排序，利用完整可用的摘要以及通过五次固定获取尝试在本地筛选出的PDF证据，而无需生成长篇报告。在涵盖EASY、MEDIUM、HARD三个难度各十个查询的历史比较中，按照所记录的统计口径，Idea2Prior的总体报告Hit@5为53.3%，ChatGPT网页版Deep Research为43.3%，Gemini网页版Deep Research为36.7%。该优势主要集中在EASY和MEDIUM等级上。

    arXiv:2610.20857v1 Announce Type: new  Abstract: Finding a paper from an informal research idea requires more than matching topic words: relevant work may use different terminology, and its distinguishing mechanism may be absent from the abstract. We present \system{}, an auditable literature-search system built on PaperArena. It combines multi-source discovery, identity verification, facet-based abstract assessment, semantic citation expansion, and joint Expert ranking. A retrieval-only mode ranks up to five papers from a shortlist of at most 60, using complete available abstracts and locally selected PDF evidence from five fixed acquisition attempts, without generating a long-form report. In a historical comparison covering ten queries from each of EASY, MEDIUM, and HARD, aggregate reported Hit@5 is 53.3\% for \system{}, 43.3\% for ChatGPT web Deep Research, and 36.7\% for Gemini web Deep Research under the documented accounting convention. The advantage is concentrated in EASY and M
    
[^14]: 面向深度搜索智能体的信息增益校准在线策略蒸馏

    Information-Gain Calibrated On-Policy Distillation for Deep Search Agents

    [https://arxiv.org/abs/2610.20664](https://arxiv.org/abs/2610.20664)

    该论文揭示了标准在线策略蒸馏中因忽视检索证据而产生的“证据盲监督”失败模式及其背后的预算先验与方向先验两种隐式偏差，并提出用信息增益来校准监督信号，从而改进深度搜索智能体的蒸馏训练。

    

    在线策略蒸馏（OPD）近来已成为一种将大型智能体能力迁移到紧凑学生模型的有前景的方法。在深度搜索场景中，智能体需要迭代地检索外部证据，而教师在检索之前对各候选查询的偏好可能无法预测哪些查询会返回有用的证据。然而，标准的OPD仅依据先前的上下文来监督学生的每个查询，由此引发了一种我们称为“证据盲监督（EBS）”的失败模式。为了理解这一问题，我们按交互轮次对OPD目标进行重新分组，并揭示了两个隐式先验：一个是按词元数量分配监督权重的“预算先验”，另一个是偏好教师所偏好词元的“方向先验”，而这两者都没有考虑当前查询所返回的证据。我们的实证分析表明，这两个先验在实践中都助长了EBS：第一轮搜索占据了51.1%的[摘要在此处截断]

    arXiv:2610.20664v1 Announce Type: new  Abstract: On-policy distillation (OPD) has recently emerged as a promising approach for transferring the capabilities of large agents to compact students. In deep search, where agents iteratively retrieve external evidence, a teacher's preference among plausible queries before retrieval may not predict which queries will return useful evidence. Standard OPD nevertheless supervises each student query from the preceding context alone, inducing a failure mode we term \textbf{\textit{evidence-blind supervision (EBS)}}. To understand this, we regroup the OPD objective by interaction turns and reveal two implicit priors: a \textit{budget prior} that allocates supervision by token volume, and a \textit{direction prior} that favors tokens preferred by the teacher, neither of which accounts for the evidence returned by the current query. Our empirical analysis shows that both priors contribute to EBS in practice: the first search turn accounts for 51.1\% o
    
[^15]: 知识图谱即文件：一种工具创建与工具使用协同演化的智能体框架

    Knowledge Graphs as Files: An Agent Harness Where Tool Creation and Tool Use Co-Evolve

    [https://arxiv.org/abs/2610.18616](https://arxiv.org/abs/2610.18616)

    提出DKGI框架，让语言模型智能体将知识图谱视为完整的节点和边文件直接交互，通过编写Python代码自行创建和修订图访问工具，实现工具创建与工具使用的协同演化，在三个领域的基准上无需训练即超越最强基线至少6.7个Hit@1点。

    

    搜索富文本知识图谱需要在遍历关系的同时审阅文本证据。预定义的图接口限制了智能体所能访问的证据范围，即使智能体编写代码来组合工具调用或处理检索到的子图也是如此。我们提出直接知识图谱交互，这是一种让语言模型智能体直接基于完整的节点和边文件创建图访问工具的框架。智能体通过编写Python代码来遍历关系、合并节点集合并筛选文本证据，将代码执行与shell搜索和文件检查交替进行。这使智能体能够批量处理大型节点集合，同时返回筛选后的文本供模型评估。执行反馈指导工具的创建与修订，而不断演化的工具又决定了智能体接下来收集哪些证据。在跨越三个领域的基准数据集上，无需训练的DKGI超过最强已发表基线至少6.7个Hit@1点。

    arXiv:2610.18616v1 Announce Type: cross  Abstract: Searching text-rich knowledge graphs requires following relations while examining textual evidence. Predefined graph interfaces constrain what evidence agents can access, even when they write code to combine tool calls or process retrieved subgraphs. We propose direct knowledge graph interaction (DKGI), a harness in which language model agents create graph access tools directly over complete node and edge files. The agent writes Python code to follow relations, combine node sets, and select textual evidence, interleaving execution with shell search and file inspection. This lets agents process large node sets in bulk while returning selected text for the model to assess. Execution feedback guides tool creation and revision, while the evolving tools determine what evidence the agent gathers next. On benchmark datasets spanning three domains, training-free DKGI exceeds the strongest published baseline by at least 6.7 Hit@1 points on ever
    
[^16]: 当输入未获得评分时：在完整数据流上评估开放网络事件准入

    When Inputs Receive No Score: Evaluating Open-Web Event Admission on the Complete Stream

    [https://arxiv.org/abs/2610.17814](https://arxiv.org/abs/2610.17814)

    该研究提出在完整数据流（包括处理流水线无法评分的输入）上评估开放网络事件条目准入，发现评分可用性会改变排序比较结果并使可实现的保留率存在上限，导致90%全流目标无法仅靠阈值调整达成。

    

    一条网络条目可以链接到正确的事件，却不贡献值得组织的信息。因此，条目级别的准入评估应从完整的输入开始，包括那些处理流水线无法评分的输入。我们研究了这种上游选择如何改变排序比较并约束运营决策。在六个指令模型骨干、三种策略和602个人工标注的评估条目上，各系统对具有不同类别构成的不同条目群体进行评分。在共享输入上比较排序，可以将这些群体差异与判别能力区分开来。评分的可用性还限制了可达到的保留率：四种配置仅对十二个正类开发条目中的十个进行了评分，使得仅通过阈值调整无法达到全流90%的目标。配对的精确率实验表明，这一约束同时取决于执行配置与模型选择。Qwen 对相同数量的评估条目进行评分，未……（原文摘要在此处截断）

    arXiv:2610.17814v2 Announce Type: replace  Abstract: A Web item can link to the correct event without contributing information worth organizing. Evaluating item-level admission therefore begins with complete inputs, including those a processing pipeline fails to score. We study how this upstream selection changes ranking comparisons and constrains operational decisions. Across six instruction-model backbones, three strategies, and 602 human-annotated evaluation items, systems score different populations with different class compositions. Comparing rankings on shared inputs separates these population differences from discrimination. Score availability also limits attainable retention: four configurations score only ten of twelve positive development items, making a 90% full-stream target unreachable by threshold adjustment. Paired precision experiments show that this constraint depends on execution configuration as well as model choice. Qwen scores the same number of evaluation items un
    
[^17]: 自改进的数据合成可跨领域、跨任务扩展嵌入模型

    Self-improving data synthesis can scale embedding models across domains and tasks

    [https://arxiv.org/abs/2610.14730](https://arxiv.org/abs/2610.14730)

    提出了SYNTRA自改进检索数据合成方法，仅需无标注语料库和少量查询-文档对，通过自改进循环迭代优化合成指令，在多数任务上超越人工校准，并使嵌入模型随数据规模扩展在15个领域持续提升性能。

    

    将嵌入模型适应到新任务需要能反映任务特定信息需求和相关性标准的训练数据。然而，人工设计数据成本高昂且难以规模化。本文提出了SYNTRA，一种新颖的检索数据合成方法，其自改进循环利用对试验输出的反馈来迭代优化可复用的合成指令。该过程仅需一个无标注语料库和少量相关的查询-文档对。在横跨15个领域的24个检索数据集上的实验证明了合成数据的有效性。这些数据还能在有无现有标注训练集的任务上提升广泛使用的强嵌入模型的性能。自改进在全部六个评估任务上都提升了检索效果，并在其中四个任务上超越了人工校准。此外，随着训练数据集规模扩展到初始大小的32倍，MS MARCO上的检索性能仍持续提升。

    arXiv:2610.14730v1 Announce Type: cross  Abstract: Adapting embedding models to new tasks requires training data that reflect task-specific information needs and relevance criteria. However, manual data design is costly and difficult to scale. Here we introduce SYNTRA, a novel retrieval data synthesis method whose self-improvement loop iteratively refines reusable synthesis instructions using feedback on trial outputs. This process requires only an unlabelled corpus and a few relevant query-document pairs. Experiments across 24 retrieval datasets spanning 15 domains demonstrate the effectiveness of the synthesized data. These data also improve widely used, strong embedding models on tasks with and without existing labelled training sets. Self-improvement enhances retrieval on all six evaluated tasks, outperforming human calibration on four. Moreover, retrieval performance continues to improve on MS MARCO as we scale the training dataset to 32 times its initial size. Overall, with minim
    
[^18]: KoViDoRe：韩语视觉文档检索

    KoViDoRe: Korean Visual Document Retrieval

    [https://arxiv.org/abs/2608.20840](https://arxiv.org/abs/2608.20840)

    本文提出了KoViDoRe，一个针对韩语视觉文档检索的基准数据集，通过多阶段数据整理流程解决了现有基准中韩语复杂文档覆盖不足和单页检索局限的问题。

    

    arXiv:2608.20840v1 公告类型：新 摘要：近期多模态检索的进展提高了从视觉丰富文档（如PDF和报告）中检索信息的能力。然而，现有基准大多以英语为中心，对具有复杂结构的韩语视觉文档覆盖有限。此外，大多数现有韩语资源主要评估单页检索，未能捕捉需要跨多页证据聚合的现实场景。为解决这些不足，我们引入了KoViDoRe，一个韩语视觉文档检索基准。该数据集基于公开可用的韩语文档构建，包含多样化的布局，如表格、图形和多栏结构。我们开发了一个多阶段数据整理流程，包括结构化文档解析、基于摘要和基于上下文的合成查询生成，以及人工验证的相关性映射。使用KoViDoRe，我们进行了广泛实验以评估现有模型性能。

    arXiv:2608.20840v1 Announce Type: new  Abstract: Recent advances in multimodal retrieval have improved the ability to retrieve information from visually rich documents such as PDFs and reports. However, existing benchmarks remain largely centered on English and provide limited coverage of Korean visual documents with complex structures. Furthermore, most existing Korean resources primarily evaluate single-page retrieval, failing to capture realistic scenarios that require evidence aggregation across multiple pages. To address these gaps, we introduce KoViDoRe, a benchmark for Korean visual document retrieval. The dataset is constructed from publicly available Korean documents with diverse layouts, including tables, figures, and multi-column structures. We develop a multi-stage data curation pipeline consisting of structured document parsing, synthetic query generation using both summary-based and context-based strategies, and relevance mapping with human verification. Using KoViDoRe, w
    
[^19]: MPB语料库：巴西流行音乐中旋律、节奏、和声及旋律-和声关系的数据集

    The MPB Corpus: A Dataset of Melody, Rhythm, Harmony, and Melody-Harmony Relationships in Brazilian Popular Music

    [https://arxiv.org/abs/2608.13842](https://arxiv.org/abs/2608.13842)

    该论文构建了首个包含500首巴西流行音乐、覆盖旋律、节奏、和声及旋律-和声关系的最全面数据集，并提供了分析工具和探索性研究，以支持计算音乐学分析。

    

    本文介绍了MPB语料库，这是一个包含500首音乐作品的数据集，涵盖了四个音乐参数：旋律轮廓、旋律节奏、和声以及旋律与和声之间的关系。它构成了迄今为止用于巴西音乐计算音乐学的最全面、最详细的数据集。为支持编码过程，我们引入了专门的分析模型，旨在精确捕捉节奏和旋律信息，并辅以定制化的可视化和指标，以总结关键音乐参数。最后，我们对数据集进行了简要的定性探索性分析，展示了其在提出和系统解决该流派音乐学问题方面的潜力。

    arXiv:2608.13842v1 Announce Type: cross  Abstract: This paper presents the MPB Corpus, a collection of 500 musical pieces encoded across four musical parameters: melodic contour, melodic rhythm, harmony, and the relationship between melody and harmony. It constitutes the most comprehensive and detailed dataset to date for computational musicology of Brazilian music. To support the encoding process, we introduce specific analytical models designed to capture rhythmic and melodic information with precision, alongside tailored visualizations and metrics summarizing key musical parameters. Finally, we provide a brief qualitative exploratory analysis of the dataset, illustrating its potential to both formulate and systematically address musicological questions concerning the genre.
    
[^20]: MemSyco-Bench：智能体记忆中谄媚行为的基准测试

    MemSyco-Bench: Benchmarking Sycophancy in Agent Memory

    [https://arxiv.org/abs/2607.01071](https://arxiv.org/abs/2607.01071)

    提出MemSyco-Bench基准，系统评估智能体记忆诱发的谄媚问题——即检索记忆导致智能体过度迎合用户而牺牲事实准确性与客观推理，涵盖五项任务衡量记忆何时应影响决策及如何正确使用有效记忆。

    

    记忆已成为现代基于LLM的智能体的基石，支持其从单轮助手向长期合作伙伴的演进。然而，记忆并非总是有益的：检索到的记忆常常引发谄媚（sycophancy）这一关键问题，导致智能体过度迎合用户，以牺牲事实准确性或客观推理为代价。尽管存在这一新兴风险，现有的记忆基准主要评估记忆是否被正确存储、检索或更新，而忽视了检索到的记忆如何影响下游的推理与决策。为弥合这一差距，我们提出了MemSyco-Bench，这是一个用于评估智能体系统中记忆诱发谄媚行为的综合基准。MemSyco-Bench衡量记忆何时应当影响决策，以及有效的记忆应如何被使用。具体而言，它涵盖五项任务，评估智能体能否拒绝将记忆作为事实证据、尊重记忆的适用范围。

    arXiv:2607.01071v3 Announce Type: replace-cross  Abstract: Memory has emerged as a cornerstone of modern LLM-based agents, supporting their evolution from single-turn assistants to long-term collaborators. However, memory is not always beneficial: retrieved memories often induce a critical issue of sycophancy, causing agents to over-align with the user at the cost of factual accuracy or objective reasoning. Despite this emerging risk, existing memory benchmarks primarily evaluate whether memories are correctly stored, retrieved, or updated, while overlooking how retrieved memories influence downstream reasoning and decision-making. To bridge this gap, we propose MemSyco-Bench, a comprehensive benchmark for evaluating memory-induced sycophancy in agent systems. MemSyco-Bench measures when memory should influence a decision and how valid memory should be used. Specifically, it covers five tasks that assess whether agents can reject memory as factual evidence, respect its applicable scope
    
[^21]: 多智能体交互记忆

    Multi-Agent Transactive Memory

    [https://arxiv.org/abs/2606.19911](https://arxiv.org/abs/2606.19911)

    提出多智能体交互记忆（MATM）框架，通过群体级的共享存储与检索机制，让智能体能够复用彼此的运行轨迹中所蕴含的程序性知识，从而提升任务执行效率。

    

    具有多样化能力的LLM智能体的去中心化部署，催生了对跨异构智能体群体知识共享基础设施的需求。正如搜索引擎为人类的问题解决而索引人类生成的产物一样，检索系统可以组织智能体生成的产物以供智能体之间复用，从而将检索增强生成从人类撰写的文档扩展到智能体生成的内容。尤其是智能体的运行轨迹中编码了可复用的程序性知识，但这些轨迹通常在使用一次后就被丢弃，或仅由产生它们的智能体所保留，迫使新实例化的智能体重新探索已有的解决方案。我们提出多智能体交互记忆（MATM），这是一个面向智能体轨迹的群体级存储与检索框架，其中生产者智能体将轨迹贡献到共享存储库中，消费者智能体则检索这些轨迹以改进任务执行。在具有长程、程序性……（原文摘要在此处截断）

    arXiv:2606.19911v2 Announce Type: replace  Abstract: Decentralized deployment of LLM agents with diverse capabilities motivates infrastructure for knowledge sharing across heterogeneous agent populations. Just as search engines index human-generated artifacts for human problem solving, retrieval systems can organize agent-generated artifacts for reuse across agents, extending retrieval-augmented generation from human-authored documents to agent-generated ones. Agent trajectories in particular encode reusable procedural knowledge, yet are typically discarded after a single use or retained only by the producing agent, forcing newly instantiated agents to rediscover existing solutions. We propose Multi-Agent Transactive Memory (MATM), a framework for population-level storage and retrieval of agent trajectories, where producer agents contribute trajectories to a shared repository and consumer agents retrieve them to improve task execution. In interactive environments with long, procedurall
    
[^22]: GRASP：面向半结构化知识库的计划引导图检索与自适应融合重排序方法

    GRASP: Plan-Guided Graph Retrieval with Adaptive Fusion and Reranking on Semi-Structured Knowledge Bases

    [https://arxiv.org/abs/2605.30237](https://arxiv.org/abs/2605.30237)

    GRASP通过小型LLM规划器生成可执行图计划、融合图排序与文本排序，并用强化学习优化规划器以最大化融合排序增益，在半结构化知识库检索上取得全面领先的性能，平均Hit@1比最强基线提升18.6个百分点且单查询仅需2.47秒。

    

    在半结构化知识库（SKB）上进行检索时，必须同时处理文本描述和关系约束。基于文本的检索器能够捕捉查询的整体语义，但难以处理多跳关系；而基于图的检索器可以精确地沿着关系路径检索，却容易忽略更广泛的整体含义。我们提出了GRASP，一种学习融合两者的混合检索器。一个小型LLM规划器通过单次调用将查询转化为可执行的图计划，其图排序结果与文本排序结果进行融合，再由微调过的重排序器对融合后的候选结果进行重新排序。不同于单独训练规划器使其自身预测准确，我们采用强化学习对规划器进行优化，以最大化其图排序为最终融合排序带来的增益。在三个STaRK基准测试上，GRASP在所有指标上都取得了最佳结果，相比最强基线平均Hit@1提升了18.6个百分点，同时每个查询仅需2.47秒。

    arXiv:2605.30237v3 Announce Type: replace-cross  Abstract: Retrieval over semi-structured knowledge bases (SKBs) must handle both textual descriptions and relational constraints. Text-based retrievers capture the overall semantics of a query but struggle with multi-hop relations, whereas graph-based retrievers follow relational paths precisely but miss the broader meaning. We propose GRASP, a hybrid retriever that learns to fuse the two. A small LLM planner translates the query into an executable graph plan in a single call, its graph ranking is fused with a text ranking, and a fine-tuned reranker reorders the fused candidates. Instead of training the planner to be accurate on its own, we optimize it with reinforcement learning to maximize the gain its graph ranking brings to the fused ranking. On three STaRK benchmarks, GRASP achieves the best results on all metrics, improving the average Hit@1 by 18.6 points over the strongest baseline, while taking only 2.47 seconds per query.
    
[^23]: 是什么在追踪难负样本的迁移效果？一项关于词汇信号与语义信号的研究

    What Tracks Hard-Negative Transfer? A Study of Lexical and Semantic Signals

    [https://arxiv.org/abs/2603.20990](https://arxiv.org/abs/2603.20990)

    本研究发现在训练前预测稠密检索器迁移效果时，基于IDF加权的查询词元非覆盖率指标（平均词汇残差度）能够稳定重现难负样本来源的迁移排序，而冻结编码器的各类语义统计量无论单独使用还是与词汇信号结合均无法做到这一点。

    

    arXiv:2603.20990v5 公告类型： replace-cross。摘要：难负样本监督的哪些属性能够在训练之前预测稠密检索器的迁移表现？我们在MS MARCO训练场景下，跨BM25、稠密检索、LLM生成和混合来源，比较了词汇非覆盖率、冻结编码器相似度统计量、难度代理指标以及这些信号的组合。评估采用三个检索器骨干网络和十一个BEIR数据集。平均词汇残差度——一种基于IDF加权的查询词元非覆盖率度量——在全部六个骨干与设置组合中均重现了聚合迁移排序。单独的语义统计量均无法恢复完整的排序，而将它们与词汇信号结合也不能提升所观察到的排序一致性。在将BM25、稠密和LLM诊断样本的词汇残差度大致匹配后，所测试的语义统计量中没有任何一个能够在三个骨干、三种匹配分辨率和100次重复实验中恢复所观察到的 LLM > Dense > BM25 迁移排序。

    arXiv:2603.20990v5 Announce Type: replace-cross  Abstract: Which properties of hard-negative supervision track dense-retriever transfer before training? We compare lexical noncoverage, frozenencoder similarity statistics, hardness proxies, and combinations of these signals across BM25, dense, LLM-generated, and hybrid sources for MS MARCO training. Evaluation uses three retriever backbones and eleven BEIR datasets. Mean lexical residuality, an IDF-weighted measure of query-token noncoverage, reproduces the aggregate transfer ordering in all six backbone and setting blocks. The individual semantic statistics recover none of the complete orderings, and combining them with the lexical signal does not improve the observed rank agreement. After approximately matching BM25, dense, and LLM diagnostic samples for lexical residuality, none of the tested semantic statistics recovers the observed LLM > Dense > BM25 transfer ordering across three backbones, three matching resolutions, and 100 repe
    
[^24]: GraphCompliance：通过策略-上下文图对齐实现的结构感知监管合规

    GraphCompliance: Structure-Aware Regulatory Compliance through Policy-Context Graph Alignment

    [https://arxiv.org/abs/2510.26309](https://arxiv.org/abs/2510.26309)

    提出GraphCompliance框架，将监管文档建模为模式引导的策略图、将被评估情境建模为实体关系上下文图，并通过合规门控进行图对齐，从而实现结构感知的监管合规自动评估。

    

    arXiv:2510.26309v2 公告类型：替换 摘要：监管合规是许多工业和行政场景中的一项重要要求。然而，合规评估仍然十分困难，因为监管要求分布在结构化文档之中，可能依赖于层级结构、定义、条件以及交叉引用，而与特定要求相关的证据可能嵌入在被评估上下文中多个相互作用的参与者、事件、对象和条件之间。我们提出了GraphCompliance，这是一个结构感知框架，它将监管文档表示为模式引导的策略图，并将上下文信息表示为由实体-关系三元组构成的上下文图。GraphCompliance通过合规门控对这两个图进行对齐，该门控识别相关的监管要求，并将其与相应的上下文证据进行评估。这种设计使监管结构显式化，同时将上下文事实组织成可比较的形式，有助（摘要原文在此处截断）

    arXiv:2510.26309v2 Announce Type: replace  Abstract: Regulatory compliance is an important requirement in many industrial and administrative settings. Yet compliance assessment remains difficult because regulatory requirements are distributed across structured documents and may depend on hierarchy, definitions, conditions, and cross-references, while the evidence relevant to a particular requirement may be embedded among multiple interacting actors, events, objects, and conditions in the assessed context. We present GraphCompliance, a structure-aware framework that represents regulatory documents as a schema-guided Policy Graph and contextual information as a Context Graph of entity-relation triples. GraphCompliance aligns the two graphs through a Compliance Gate, which identifies relevant regulatory requirements and evaluates them against the corresponding contextual evidence. This design makes regulatory structure explicit while organizing contextual facts into a comparable form, hel
    
[^25]: 大语言模型生成的文本如何影响基于词项的检索模型？

    How Do LLM-Generated Texts Impact Term-Based Retrieval Models?

    [https://arxiv.org/abs/2508.17715](https://arxiv.org/abs/2508.17715)

    本文通过语言学分析发现LLM生成的文本具有独特的词频分布、更高的词项特异性和文档多样性，并据此研究了这些特征如何影响以BM25为代表的基于词项的检索模型的表现。

    

    arXiv:2508.17715v2 公告类型：replace-cross 摘要：随着越来越多由大语言模型（LLM）生成的内容涌入互联网，信息检索（IR）系统如今面临着区分和处理人类撰写与机器生成文本混合内容的挑战。近期研究表明，神经检索器可能对LLM生成的内容表现出偏好性倾向，而像BM25这样的经典基于词项的检索器则更倾向于人类撰写的文档。本文研究了LLM生成的内容对基于词项的检索模型的影响，这类模型因其高效性和跨领域的强泛化能力而备受重视。我们的语言学分析揭示，LLM生成的文本呈现出更平缓的高频段和更陡峭的低频段Zipf斜率、更高的词项特异性以及更大的文档级多样性。这些特征与LLM被训练为通过多样且精确的表达来优化读者体验相一致。我们的研究进一步探讨了……

    arXiv:2508.17715v2 Announce Type: replace-cross  Abstract: As more content generated by large language models (LLMs) floods into the Internet, information retrieval (IR) systems now face the challenge of distinguishing and handling a blend of human-authored and machine-generated texts. Recent studies suggest that neural retrievers may exhibit a preferential inclination toward LLM-generated content, while classic term-based retrievers like BM25 tend to favor human-written documents. This paper investigates the influence of LLM-generated content on term-based retrieval models, which are valued for their efficiency and robust generalization across domains. Our linguistic analysis reveals that LLM-generated texts exhibit smoother high-frequency and steeper low-frequency Zipf slopes, higher term specificity, and greater document-level diversity. These traits are aligned with LLMs being trained to optimize reader experience through diverse and precise expressions. Our study further explores 
    
[^26]: 通过分层上下文剪枝提升知识密度以实现检索增强的仓库级代码补全

    Enhancing Knowledge Density via Hierarchical Context Pruning for Retrieval-Augmented Repository-Level Code Completion

    [https://arxiv.org/abs/2508.10068](https://arxiv.org/abs/2508.10068)

    提出SaraCoder框架，在固定上下文预算下通过分层上下文剪枝提升检索上下文的知识密度，避免冗余代码浪费token并释放跨文件依赖与相似代码片段的协同作用，从而提升仓库级代码补全性能。

    

    arXiv:2508.10068v3 公告类型：replace-cross。摘要：检索增强生成通过检索外部代码片段、API文档或类型定义来扩展模型的上下文感知能力，从而推动了仓库级代码补全的发展。然而，在上下文窗口有限的token预算下，现有的检索增强方法往往用高度相似甚至冗余的代码填满预算，这不仅浪费了token，还抑制了关键的跨文件依赖，阻碍了语义相似的参考片段与跨文件依赖定义之间的协同交互。因此，性能的进一步提升受到检索上下文中知识密度低的限制，而非模型能力的限制。我们提出了SaraCoder，一个在固定上下文预算下通过分层上下文剪枝来提升知识密度的检索增强框架。其中的分层特征优化模块分四个阶段对超完备的候选池进行剪枝……（摘要原文在此处截断）

    arXiv:2508.10068v3 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) has advanced repository-level code completion by retrieving external code snippets, API documentation, or type definitions to expand the model's contextual awareness. However, under the finite token budget of the context window, existing retrieval-augmented methods often saturate the budget with highly similar or even redundant code, which not only wastes tokens but also suppresses critical cross-file dependencies, preventing synergistic interaction between semantically similar reference snippets and cross-file dependency definitions. Therefore, further gains are limited by low knowledge density in the retrieved context rather than by model capacity. We propose SaraCoder, a retrieval-augmented framework that enhances knowledge density via hierarchical context pruning under a fixed context budget. A Hierarchical Feature Optimization module prunes an over-complete candidate pool in four stages
    

