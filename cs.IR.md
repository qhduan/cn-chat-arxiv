# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tool Recommendation for Multi-Step Agents: Finding Every Tool a Task Needs](https://arxiv.org/abs/2610.23707) | 提出了一种已部署的“工具推荐”网关系统，通过固定的三个工具接口让智能体能够访问约8,200个工具（是典型MCP服务器暴露数量的数百倍），并在ToolRet基准上将查询所需工具排进前十的成功率比BM25提高57%。 |
| [^2] | [Humanity's Last Choice: A Common Evaluation Standard for Generative Engine Optimization](https://arxiv.org/abs/2610.23541) | 该论文呼吁建立生成式引擎优化的统一评估标准，并提出一个元分类法来区分“优化算法改进”与“信息源编辑自由度”所带来的不同收益，从而实现对不同基准测试评估结果的公平比较。 |
| [^3] | [LSNR: Learned Sparse News Recommendations with Attentive User Modeling](https://arxiv.org/abs/2610.23510) | LSNR通过独立编码每篇新闻文章，并结合基于注意力的加权与用户级稠密扩展来构建稀疏用户向量，解决了现有学习稀疏推荐方法在新闻场景下用户兴趣被扁平化以及长文本导致截断的两大瓶颈，在MIND和EBNeRD数据集上显著优于STAR。 |
| [^4] | [QD-Rec: Open-Ended Evolution of Recommendations via Quality-Diversity](https://arxiv.org/abs/2610.23470) | 提出QD-Rec方法，将质量-多样性算法、上下文老虎机与基础模型相结合，为高风险决策生成多样化、高质量且附带自然语言理由的开放式推荐。 |
| [^5] | [Jev-as-a-Judge for Brazilian Portuguese Information Retrieval: Relevance Assessment and Reranking](https://arxiv.org/abs/2610.23346) | 该研究证明，作为快速结构化决策模型的Jev充当“评判者”时，虽然在单个文档层面的相关性判断与人工标注存在一定差异，但其生成的相关性判断能产生与官方标注高度一致（Kendall's tau达0.90–0.93）的检索系统排名，并能有效充当重排序器，适用于巴西葡萄牙语信息检索。 |
| [^6] | [Code Skill Server: A Large-Scale Search Engine for Code Generation Skills](https://arxiv.org/abs/2610.23186) | 提出了Code Skill Server（CSS），一个从多个异构来源增量收集并索引公开代码生成技能的大规模搜索引擎，能够自动为目标任务检索最有用的技能，解决了手动搜索技能在大规模场景下不可行的问题。 |
| [^7] | [Reasoning Reranking with Semantic IDs as the Sole Item Representation](https://arxiv.org/abs/2610.22797) | 本文构建了ReRankID受控测试平台，首次系统评估仅以语义ID作为物品表示进行推理重排序的可行性，发现此类模型常常直接复制输入顺序而非真正重排序，且引入可读元数据也无法修复这一失败。 |
| [^8] | [Improving OCR Accuracy Through Multi-Classifier Verification of Visually Similar Characters](https://arxiv.org/abs/2610.22726) | 提出字符验证模型（CVM），利用针对视觉易混淆字符对训练的专用二元分类器进行字符级后处理，无需依赖词典即可纠正OCR输出中的模糊字符，从而提升序列号、车牌等非词汇数据的识别准确率。 |
| [^9] | [NoveltyScope: A Multi-Agent Research Assistant for Evidence-Grounded Novelty Assessment of Scientific Papers](https://arxiv.org/abs/2610.22703) | 论文提出NoveltyScope，一个通过检索先前工作、请求配对反驳证据并核对引文，由控方、辩方、法官和评论家四个智能体角色对论文作者声称的贡献进行基于证据的新颖性评估并输出五级评分的多智能体研究助手，其在RINoBench上的误差优于单LLM评分和向量相似度基线。 |
| [^10] | [AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems](https://arxiv.org/abs/2610.22511) | AuditKIT是一个统一的评估框架，将基准测试、评判评估、运行时性能、工具使用和检索质量整合到单一结果表示中，解决了大语言模型与智能体系统评估中接口和指标语义不一致的问题。 |
| [^11] | [Star Light, Star Bright: Can Multimodal Models Parse an Astronomy Visual Question Answering Dataset Right?](https://arxiv.org/abs/2610.22474) | 该研究构建了一个包含2001张天文图像（等高线图、NASA SkyView真实望远镜图像和伪天空图像）及47357个问题的领域特定视觉问答数据集，测试发现低成本多模态大模型难以高准确率地回答天文领域特定问题。 |
| [^12] | [HEM-Med: A Hierarchical Agentic Memory Framework for Safe Multi-Center Medication Recommendation](https://arxiv.org/abs/2610.20191) | HEM-Med提出了一种分层智能体记忆框架，通过将可迁移的跨医院临床经验与医院特异性处方模式解耦，并纳入药物相互作用等多标签安全约束，实现安全的多中心用药推荐。 |
| [^13] | [Kwai Summary Attention V2 Technical Report](https://arxiv.org/abs/2610.18654) | 提出KSA V2架构，通过允许不同层设置不同的摘要压缩分块窗口，实现更灵活的KV缓存内存压缩，并解决全注意力层中摘要标记带来的额外计算开销问题。 |
| [^14] | [RIT-RAG: Navigating Document Corpora with Retrieval-Induced Trees](https://arxiv.org/abs/2610.11370) | RIT-RAG将内容检索与文档结构导航相结合，离线为每篇文档构建目录树，查询时先检索宽泛的文本块并利用其位置诱导出可跨多个文档的、规模可控的子树，再由LLM智能体在子树中导航，从而在保留结构感知优势的同时克服了现有方法无法扩展到大型语料库且选错文档后无法恢复的缺陷。 |
| [^15] | [UniRec: Cross-stage Multi-Task Fusion with Preference Alignment for Cascaded Recommender Systems](https://arxiv.org/abs/2609.11052) | UniRec提出了一种统一的跨阶段推荐融合模型，通过共享嵌入和单一计算图联合训练实现跨阶段梯度传播，并结合双轴偏好对齐机制，解决级联推荐系统中预排序与排序阶段的跨阶段不一致问题。 |
| [^16] | [KGVoyager: Knowledge Graph Agnostic Question Answering via Agentic Navigation](https://arxiv.org/abs/2609.01780) | KGVoyager提出了一种知识图谱无关的智能体架构，通过“思考-行动-观察”循环动态探索图谱结构与语义，仅需查询端点即可从自然语言问题生成SPARQL查询，在四个基准上将F1提升约8分，同时将成本与运行时间各降低约22%。 |
| [^17] | [EKLAV: Elicited Key CoT Learning via Answer only superVision](https://arxiv.org/abs/2607.25182) | EKLAV将教师模型的思维链轨迹作为特权输入上下文而非模仿目标，仅对最终答案进行监督训练，在重排序任务上以更低的训练计算成本和推理时间超越了传统推理蒸馏方法。 |
| [^18] | [Agent-Facing Information Design in LLM Tool Registries: A Preregistered Test of Rhetoric, Position and Structure](https://arxiv.org/abs/2605.23916) | 研究通过预注册实验发现，LLM智能体的工具选择会被工具描述中的推销性赞美语言显著影响（可提升约43个百分点选中率，甚至选到无法胜任任务的工具），列表排序和结构化字段也会强烈左右选择，因此建议注册表以结构化字段而非营销文本来呈现工具限制。 |

# 详细

[^1]: 面向多步骤智能体的工具推荐：找到任务所需的每一个工具

    Tool Recommendation for Multi-Step Agents: Finding Every Tool a Task Needs

    [https://arxiv.org/abs/2610.23707](https://arxiv.org/abs/2610.23707)

    提出了一种已部署的“工具推荐”网关系统，通过固定的三个工具接口让智能体能够访问约8,200个工具（是典型MCP服务器暴露数量的数百倍），并在ToolRet基准上将查询所需工具排进前十的成功率比BM25提高57%。

    

    当多步骤智能体任务所需的某一个工具缺失时，任务就会失败，然而智能体可能需要的工具数量庞大、难以尽数列出：在我们所研究的平台上，约8,200个工具的定义加起来超过了我们所评估的任何模型的上下文窗口。我们提出了“工具推荐”，这是一个已部署的系统，通过一个由三个工具组成的固定接口，为智能体提供所有这些工具，其数量是典型MCP服务器所暴露工具的数百倍。该系统运行在智能体与工具之间的网关处，以最终用户自己的凭证执行每一次调用，并根据用户对每项服务的授权同意以及智能体可使用的工具范围对其进行核查。我们在公开的检索基准和按计费统计的端到端会话成本上对其进行了评估。在ToolRet上，“工具推荐”将查询所需的每一个工具排进前十名的频率比BM25高出57%（BM25是商业工具搜索所采用的词法排序方法），并且与规模最大达其九倍的已发表系统相比也表现更优。

    arXiv:2610.23707v1 Announce Type: new  Abstract: A multi-step agent task fails when a single tool it needs is missing, yet the tools an agent may need are too many to list: the roughly 8,200 tools on the platform we study have definitions that together exceed the context window of any model we evaluate. We present Tool Recommendation, a deployed system that gives an agent all of them, hundreds of times more than a typical MCP server exposes, through a fixed interface of three tools. It runs at the gateway between agent and tools, executing every call under the end user's own credentials and checking it against the user's consent for each service and the tools the agent may use. We evaluate it on both a public retrieval benchmark and billed end-to-end session cost. On ToolRet, Tool Recommendation places every tool a query needs in the top ten 57% more often than BM25, a lexical ranking that commercial tool search offers, and compares favorably with published systems up to nine times its
    
[^2]: 人类的最后抉择：生成式引擎优化的统一评估标准

    Humanity's Last Choice: A Common Evaluation Standard for Generative Engine Optimization

    [https://arxiv.org/abs/2610.23541](https://arxiv.org/abs/2610.23541)

    该论文呼吁建立生成式引擎优化的统一评估标准，并提出一个元分类法来区分“优化算法改进”与“信息源编辑自由度”所带来的不同收益，从而实现对不同基准测试评估结果的公平比较。

    

    “人类的最后抉择”描述了这样一种理念：人们需要做出的最后一个选择，就是将后续决策托付给哪一个AI系统。这些系统依赖外部信息源，而内容提供者可以通过生成式引擎优化来编辑这些信息源，以提升信息源的可见度并影响推荐结果。因此，在这些系统之间进行选择时，需要可比较的证据，以衡量它们在受到GEO影响时执行用户请求的能力。现有基准测试在标准化任务、模型和指标方面已取得进展，但它们通常各自定义GEO策略空间，并使用不同的编辑约束。这些差异可能将更优优化算法带来的收益与更大信息源编辑自由度带来的收益混为一谈，同时使不同基准测试之间的评估覆盖范围难以比较。因此，我们呼吁建立统一的评估标准，并提出一个元分类法来描述信息源在哪里被编辑、如何被编辑（摘要原文在此处被截断）。

    arXiv:2610.23541v1 Announce Type: new  Abstract: Humanity's Last Choice describes the idea that the last choice people need to make is which AI system they entrust with the decisions that follow. These systems rely on external sources that content providers can edit through generative engine optimization (GEO) to increase source visibility and influence recommendations. Choosing among these systems therefore requires comparable evidence of how well they carry out user requests when exposed to GEO. Benchmarks have made progress in standardizing tasks, models, and metrics, but they often define their own spaces of GEO strategies and use different editing constraints. These differences can conflate gains from better optimization algorithms with those from greater freedom to edit sources, while making evaluation coverage difficult to compare across benchmarks. We therefore call for a common evaluation standard and propose a meta-taxonomy that describes where sources are edited, how they ar
    
[^3]: LSNR：基于注意力用户建模的学习稀疏新闻推荐

    LSNR: Learned Sparse News Recommendations with Attentive User Modeling

    [https://arxiv.org/abs/2610.23510](https://arxiv.org/abs/2610.23510)

    LSNR通过独立编码每篇新闻文章，并结合基于注意力的加权与用户级稠密扩展来构建稀疏用户向量，解决了现有学习稀疏推荐方法在新闻场景下用户兴趣被扁平化以及长文本导致截断的两大瓶颈，在MIND和EBNeRD数据集上显著优于STAR。

    

    当前推荐系统中的学习稀疏方法（如STAR）通过将用户的交互历史拼接成单个查询字符串来表示用户。在新闻场景下，这会带来两个瓶颈：拼接操作将用户各自不同的阅读兴趣压缩成一条扁平的查询；同时，由于新闻文章属于长文本，拼接多篇新闻很快就会超出作为骨干的语言模型的上下文窗口，从而被迫进行损害质量的截断。为了解决这两个瓶颈，我们提出了学习稀疏新闻推荐（LSNR）。LSNR对每篇文章进行独立编码，并通过将基于学习的注意力加权作用于每篇文章的稀疏表示，再结合用户级别的稠密投影扩展，来构建稀疏用户向量。在MIND和EBNeRD数据集上的实验表明，LSNR显著优于STAR，并且即使是在历史记录较短、需要截断的设置下，这些优势依然保持。

    arXiv:2610.23510v1 Announce Type: new  Abstract: Current learned sparse approaches for recommender systems, such as STAR, represent a user by concatenating their interaction history into a single query string. For news, this creates two bottlenecks: concatenation collapses a user's distinct reading interests into one flat query, and, because news articles are long-form text, concatenating several news articles quickly overruns the context window of the language model used as backbone and forces quality-degrading truncation.   To address both bottlenecks, we introduce learned sparse news recommendations (LSNR). LSNR encodes each article independently and constructs a sparse user vector by combining learned, attention-based weighting of the per-article sparse representations with a dense-projected expansion at the user level. Experiments on the MIND and EBNeRD datasets show that LSNR significantly outperforms STAR, with these gains holding even in short-history settings where truncation 
    
[^4]: QD-Rec：基于质量-多样性的推荐开放式演化

    QD-Rec: Open-Ended Evolution of Recommendations via Quality-Diversity

    [https://arxiv.org/abs/2610.23470](https://arxiv.org/abs/2610.23470)

    提出QD-Rec方法，将质量-多样性算法、上下文老虎机与基础模型相结合，为高风险决策生成多样化、高质量且附带自然语言理由的开放式推荐。

    

    开放式搜索与基础模型相结合，已催生了诸如更快的算法、开放数学问题的新成果以及自我改进智能体等进展。我们将这一范式扩展到用于决策制定的推荐，重点关注那些备选方案集合固定但选择理由并非固定的决策。在高风险决策中，例如收购哪家公司或瞄准哪些客户，比较多样化且有充分论证的选项可能比单一的排序列表更有用。我们将开放式推荐形式化为构建一个多样化、高质量的推荐集合，其中每个推荐由一个项目和一个假设组成，该假设是用自然语言解释该项目为何适合决策者需求的理由。我们提出QD-Rec，这是一种质量-多样性方法，它使用上下文老虎机来选择项目，并利用基础模型以存档中保留的早期假设为踏脚石来生成新的假设。

    arXiv:2610.23470v1 Announce Type: cross  Abstract: Open-ended search combined with foundation models has produced advances such as faster algorithms, new results on open mathematical problems, and self-improving agents. We extend this paradigm to recommendation for decision-making, focusing on decisions where the set of possible choices is fixed but the reasons for choosing them are not. In high-stakes decisions, e.g., which company to acquire or which customer to target, comparing diverse, well-reasoned options can be more useful than a single ranked list. We formalize open-ended recommendation as building a diverse, high-quality set of recommendations, each an item together with a hypothesis, a natural-language reason why the item fits the needs of a decision maker. We propose QD-Rec, a quality--diversity method that selects items with a contextual bandit and generates hypotheses with a foundation model from earlier ones kept in an archive as stepping stones. A new recommendation ent
    
[^5]: Jev作为评判者应用于巴西葡萄牙语信息检索：相关性评估与重排序

    Jev-as-a-Judge for Brazilian Portuguese Information Retrieval: Relevance Assessment and Reranking

    [https://arxiv.org/abs/2610.23346](https://arxiv.org/abs/2610.23346)

    该研究证明，作为快速结构化决策模型的Jev充当“评判者”时，虽然在单个文档层面的相关性判断与人工标注存在一定差异，但其生成的相关性判断能产生与官方标注高度一致（Kendall's tau达0.90–0.93）的检索系统排名，并能有效充当重排序器，适用于巴西葡萄牙语信息检索。

    

    本研究探讨了Jev——一个为快速、结构化决策而设计的System One（第一系统）模型——在信息检索中的应用。我们采用Jev-as-a-judge（Jev作为评判者）的方式，为四个涵盖不同领域的巴西葡萄牙语信息检索文档集合生成相关性判断，并将其与各集合的官方相关性判断进行比较。我们还将Jev作为重排序器进行了评估。在相关性评估方面，完全一致率介于52.9%至64.9%之间，二次加权Cohen's kappa介于0.52至0.82之间。大多数分歧都涉及相邻的相关性等级，至少94.5%的标注完全匹配或仅相差一个相关性等级。尽管在单个查询—文档层面存在差异，但Jev生成的qrels（相关性判断集合）产生了与官方标注相似的检索系统排名：对于NormasTCU和Ulysses-RFCorpus两个数据集，在两种评估指标下Kendall's tau（肯德尔等级相关系数）介于0.90至0.93之间。在重排序任务中，Jev对每个查询的BM25检索出的前1000个候选文档进行了重新排序（原文在此处截断）。

    arXiv:2610.23346v1 Announce Type: new  Abstract: This study investigates Jev, a System One model designed for fast, structured decision-making, in information retrieval. We use Jev-as-a-judge to generate relevance judgments for four Brazilian Portuguese IR collections covering different domains and compare them with the official relevance judgments of each collection. We also evaluate Jev as a reranker. For relevance assessment, exact agreement ranged from 52.9% to 64.9%, while quadratic weighted Cohen's kappa ranged from 0.52 to 0.82. Most disagreements involved adjacent relevance levels, with at least 94.5% of annotations matching exactly or differing by only one relevance level. Despite differences at the individual query--document level, Jev-generated qrels produced similar retrieval system rankings, with Kendall's tau ranging from 0.90 to 0.93 for NormasTCU and Ulysses-RFCorpus across the two evaluation metrics. For reranking, Jev reordered the top 1,000 BM25 candidates for each o
    
[^6]: 代码技能服务器：面向代码生成技能的大规模搜索引擎

    Code Skill Server: A Large-Scale Search Engine for Code Generation Skills

    [https://arxiv.org/abs/2610.23186](https://arxiv.org/abs/2610.23186)

    提出了Code Skill Server（CSS），一个从多个异构来源增量收集并索引公开代码生成技能的大规模搜索引擎，能够自动为目标任务检索最有用的技能，解决了手动搜索技能在大规模场景下不可行的问题。

    

    智能体技能已被证明是提升智能体在特定任务（包括软件工程、自然科学和数学）上能力的有效技术。在这一过程中，使用智能体技能通常需要从异构来源中手动发现和整理技能，而现有的技能检索方法主要侧重于从已有技能库中选取相关技能。然而，随着可复用智能体技能的数量和种类快速增长，在异构来源中手动搜索相关技能在大规模场景下已变得不切实际。在本文中，我们提出了代码技能服务器，这是一个大规模搜索引擎，用于为目标任务找到最有用的代码生成技能。CSS由一个技能服务器组成，该服务器从多个来源增量式地收集并索引公开可用的代码生成技能，本工作所用语料库中包含993个技能。

    arXiv:2610.23186v1 Announce Type: cross  Abstract: Agent skills have proven to be an effective technique for improving agent ability on specific tasks, including software engineering, natural science, and mathematics. During such a process, using agent skills often involves manually discovering and organizing skills from heterogeneous sources, while existing skill retrieval methods mainly focus on selecting relevant skills from established skill libraries. However, with the rapidly increasing number and variety of reusable agent skills, manually searching for relevant skills across heterogeneous sources is becoming impractical at scale. In this paper, we introduce the Code Skill Server (CSS), a large-scale search engine for finding the most useful code generation skills for a target task. CSS consists of a skill server that incrementally collects and indexes publicly available code generation skills from multiple sources, with 993 skills included in the corpus used in this work. Extens
    
[^7]: 以语义ID作为唯一物品表示的推理重排序

    Reasoning Reranking with Semantic IDs as the Sole Item Representation

    [https://arxiv.org/abs/2610.22797](https://arxiv.org/abs/2610.22797)

    本文构建了ReRankID受控测试平台，首次系统评估仅以语义ID作为物品表示进行推理重排序的可行性，发现此类模型常常直接复制输入顺序而非真正重排序，且引入可读元数据也无法修复这一失败。

    

    推理重排序器在排定候选列表之前会先生成一段推理依据，而现有系统通过读取物品元数据来实现这一过程，尽管生成式推荐系统正日益采用紧凑的语义ID（SIDs）来表示物品。仅凭SID进行排序具有两点吸引力：其一，SID提示词远短于匹配的物品文本；其二，尽管物品编码由内容固定，但骨干模型会部分地从用户-物品交互序列中学习这些token的嵌入，因此这些token可以承载物品文本所缺乏的行为信号。然而，此类训练以“下一物品生成”为目标，其能否支持仅凭SID对给定的候选列表进行排序尚未得到验证。我们构建了ReRankID——一个受控测试平台，模型仅依据以SID形式呈现的历史与候选列表进行推理和排序，并每次仅改变一个变量：物品呈现方式、历史长度或候选列表大小。在该测试平台上，此类模型往往直接复制输入顺序而不进行重排序，且可读的元数据也无法修复这一失败。

    arXiv:2610.22797v1 Announce Type: cross  Abstract: Reasoning rerankers write a rationale before ranking a candidate slate, and existing systems read item metadata to do so, even though generative recommenders increasingly represent items as compact Semantic IDs (SIDs). Ranking from SIDs alone has two appeals: SID prompts are far shorter than the matched item text, and although item codes are fixed from content, the backbone learns their token embeddings in part from user-item interaction sequences, so the tokens can carry behavioral signal that item text lacks. That training targets next-item generation, and whether it supports ranking a supplied slate from SIDs alone is untested. We build ReRankID, a controlled testbed in which models reason and rank from a history and slate shown only as SIDs, varying item rendering, history length, and slate size one at a time. On this testbed, such models often copy the input order instead of reranking, and readable metadata does not repair the fai
    
[^8]: 通过多分类器验证视觉相似字符来提高OCR准确率

    Improving OCR Accuracy Through Multi-Classifier Verification of Visually Similar Characters

    [https://arxiv.org/abs/2610.22726](https://arxiv.org/abs/2610.22726)

    提出字符验证模型（CVM），利用针对视觉易混淆字符对训练的专用二元分类器进行字符级后处理，无需依赖词典即可纠正OCR输出中的模糊字符，从而提升序列号、车牌等非词汇数据的识别准确率。

    

    区分视觉相似的字符是光学字符识别（OCR）系统面临的一个长期挑战，会导致文本数字化的关键应用中出现错误。虽然预处理技术可以改善图像质量、词级别的后处理可以纠正拼写错误，但两者都无法解决字符级别的视觉歧义问题——这是某些字符对的固有属性，即使是最先进的OCR模型也受限于此。我们提出了字符验证模型（CVM），这是一个字符级别的后处理框架，利用在视觉易混淆字符对（如'O'/'0'、'Z'/'2'和'S'/'5'）上训练的专用二元分类器来验证和纠正OCR输出中的模糊字符。CVM独立于词典验证运行，因此适用于包括序列号、字母数字代码和车牌在内的非词汇数据。在三个OCR引擎（Tesseract、EasyOCR、TrOCR）和十种字体上的实验……

    arXiv:2610.22726v1 Announce Type: cross  Abstract: Distinguishing visually similar characters is a persistent challenge for optical character recognition (OCR) systems, causing errors in critical applications of text digitization. While preprocessing techniques improve image quality and word-level post-processing corrects spelling errors, neither addresses character-level visual ambiguity, an intrinsic property of certain character pairs that limits even state-of-the-art OCR models. We present the Character Verification Model (CVM), a character-level post-processing framework that verifies and corrects ambiguous characters in OCR outputs using specialized binary classifiers trained on visually confusable pairs such as 'O'/'0', 'Z'/'2', and 'S'/'5'. CVM operates independently of dictionary validation, making it suitable for non-lexical data including serial numbers, alphanumeric codes, and license plates. Experiments on three OCR engines (Tesseract, EasyOCR, TrOCR) and ten fonts with lo
    
[^9]: NoveltyScope：一个用于基于证据的学术论文新颖性评估的多智能体研究助手

    NoveltyScope: A Multi-Agent Research Assistant for Evidence-Grounded Novelty Assessment of Scientific Papers

    [https://arxiv.org/abs/2610.22703](https://arxiv.org/abs/2610.22703)

    论文提出NoveltyScope，一个通过检索先前工作、请求配对反驳证据并核对引文，由控方、辩方、法官和评论家四个智能体角色对论文作者声称的贡献进行基于证据的新颖性评估并输出五级评分的多智能体研究助手，其在RINoBench上的误差优于单LLM评分和向量相似度基线。

    

    NoveltyScope是一个原型系统，它将一篇科学论文与检索到的先前工作进行对比，并生成结构化报告和五级新颖性评分。该系统提取论文的核心任务以及最多两个作者声称的贡献，然后使用BM25与稠密检索相结合倒数排名融合的方法，在一个封闭的本地语料库中进行搜索。贡献对比时会为潜在的反驳观点请求配对的文本段落，并且所有引文都会与缓存的源文本进行核对。四个评分角色（控方、辩方、法官和评论家）基于由此产生的证据进行评分，法官的评分最多只进行一次修订。评估使用了50个类别均衡的RINoBench样本，其中想法层面的标签被保留用于全文输入。使用Gemini时，单LLM评分和多智能体评分的平均绝对误差分别为1.48和1.26；使用Qwen3时分别为1.22和1.16。向量相似度基线的误差为1.68。HiCLIP案例研究展示了贡献对比的具体效果。

    arXiv:2610.22703v1 Announce Type: cross  Abstract: NoveltyScope is a prototype that compares a scientific paper with retrieved prior work and produces a structured report and a five-level novelty score. The system extracts the core task and at most two author-claimed contributions, then searches a closed local corpus using BM25 and dense retrieval with reciprocal rank fusion. Contribution comparisons request paired passages for potential refutations, and quotations are checked against cached source text. Four scoring roles (Prosecutor, Defender, Judge, and Critic) use the resulting evidence, with at most one revision of the Judge's score. Evaluation uses 50 class-balanced RINoBench samples whose idea-level labels are retained for full-paper input. Mean absolute error is 1.48 and 1.26 for single-LLM and multi-agent scoring with Gemini, respectively, and 1.22 and 1.16 with Qwen3. The vector-similarity baseline has an error of 1.68. A HiCLIP case study illustrates the contribution compari
    
[^10]: AuditKIT：一个用于大语言模型与智能体系统基准测试、评判与比较的统一评估主干

    AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems

    [https://arxiv.org/abs/2610.22511](https://arxiv.org/abs/2610.22511)

    AuditKIT是一个统一的评估框架，将基准测试、评判评估、运行时性能、工具使用和检索质量整合到单一结果表示中，解决了大语言模型与智能体系统评估中接口和指标语义不一致的问题。

    

    评估语言模型和智能体需要综合基准测试准确率、基于评判模型的评估、运行时性能、工具使用行为以及检索质量等多方面指标。接口、指标语义和轨迹格式的差异使这些比较变得复杂，并可能掩盖智能体工作流中的失败。我们提出了AuditKIT，一个源码可用的评估库，它将这些测量统一记录在单一的结果表示中，具备方向感知的指标契约、配置指纹以及可扩展的指标和后端。针对智能体，它提供回合感知的工具调用匹配，区分并行与顺序执行，并将经过验证的任务结果与轨迹级诊断区分开来。缺失或不可观察的证据会被标记为“未评分”，而不是被计为失败。针对检索增强生成（RAG），它结合了排序检索指标、评判式的依据性评估以及确定性的证据检查。七个案例研究涵盖模型（比较）……

    arXiv:2610.22511v1 Announce Type: cross  Abstract: Evaluating language models and agents requires combining benchmark accuracy, judge-based assessment, runtime performance, tool-use behavior, and retrieval quality. Differences in interfaces, metric semantics, and trace formats complicate these comparisons and can hide failures in agentic workflows. We present AuditKIT, a source-available evaluation library that records these measurements in one result representation, with direction-aware metric contracts, configuration fingerprints, and extensible metrics and backends. For agents, it provides turn-aware tool-call matching, separates parallel from sequential execution, and distinguishes verified task outcomes from trajectory-level diagnostics. Missing or unobservable evidence is marked unscored rather than counted as failure. For retrieval-augmented generation, it combines ranked retrieval metrics, judged grounding, and deterministic evidence checks. Seven case studies spanning model co
    
[^11]: 星光璀璨：多模态模型能正确解析天文视觉问答数据集吗？

    Star Light, Star Bright: Can Multimodal Models Parse an Astronomy Visual Question Answering Dataset Right?

    [https://arxiv.org/abs/2610.22474](https://arxiv.org/abs/2610.22474)

    该研究构建了一个包含2001张天文图像（等高线图、NASA SkyView真实望远镜图像和伪天空图像）及47357个问题的领域特定视觉问答数据集，测试发现低成本多模态大模型难以高准确率地回答天文领域特定问题。

    

    图表是科学交流的重要媒介，而视觉问答（VQA）领域旨在测试大型多模态模型（LMM）理解图表的能力。本文研究了多个低成本LMM在针对天空天文图像的领域特定VQA任务上的准确性。为实现这一目标，我们构建了一个领域特定的VQA数据集，包含2001张部分合成的图表，平均分为三类：等高线图、取自NASA SkyView的真实望远镜图像，以及“伪”天空图像（后者与真实图像共享坐标轴和角尺度，但其颜色由高斯混合模型生成），并配有47357个问题。每张图表均以原始版本发布，同时提供模拟历史天文文献页面老化效果的逼真退化版本。通过对三个低成本专有LMM进行测试，我们发现这些模型经常无法以高准确率回答领域特定问题。与1（注：原文摘要在此处截断）

    arXiv:2610.22474v1 Announce Type: cross  Abstract: Figures are essential for scientific communication, and the field of Visual Question Answering (VQA) aims to test the figure understanding capabilities of large multimodal models (LMMs). Here, we investigate the accuracy of several low-cost LMMs on domain-specific VQA tasks for astronomical images of the sky. As instrument for this task we build a domain-specific VQA dataset comprising 2001 partially-synthetic figures divided evenly between contour plots, real telescope images drawn from NASA's SkyView, and "fake" sky images which share the axes and angular scales of the real images but draw their colors from a gaussian mixture model, alongside 47357 questions. Each figure is released as pristine and with realistic page-aging degradations mimicking the historical astronomical corpus. Testing three low-cost proprietary LMMs, we find models frequently fail to answer domain-specific questions at high accuracy. An initial comparison with 1
    
[^12]: HEM-Med：一种面向安全多中心用药推荐的分层智能体记忆框架

    HEM-Med: A Hierarchical Agentic Memory Framework for Safe Multi-Center Medication Recommendation

    [https://arxiv.org/abs/2610.20191](https://arxiv.org/abs/2610.20191)

    HEM-Med提出了一种分层智能体记忆框架，通过将可迁移的跨医院临床经验与医院特异性处方模式解耦，并纳入药物相互作用等多标签安全约束，实现安全的多中心用药推荐。

    

    用药推荐（MR）是临床决策支持的关键环节。现有的用药推荐方法大多针对单中心学习设计，即仅涉及一家独立医院；而在实践中，涉及不同规模医院的多中心用药推荐需求日益增长。在多中心场景下，大型医疗中心的丰富数据可以提升低资源中心的推荐性能。尽管已有一些针对该任务的初步尝试，但仍存在两个关键缺口。缺口一：现有的多中心方法未能将可迁移的跨医院临床经验与医院特异的处方模式解耦，限制了临床可行性和医生的接受度。缺口二：这些方法普遍忽视了用药推荐的多标签安全约束，包括药物-药物相互作用风险和本地处方集可行性，而这些对于安全的真实世界部署至关重要。为解决这些缺口，我们提出了HEM-Med，一种新颖的...

    arXiv:2610.20191v1 Announce Type: cross  Abstract: Medication recommendation (MR) is critical for clinical decision support. While existing MR methods are mostly designed for single-center learning, where only one independent hospital is involved, multi-center MR with hospitals of different scales has become increasingly demanding in practice. In multi center settings, abundant data from large medical centers can improve recommendation performance for low-resource centers. Despite a few preliminary attempts on this task, two critical gaps remain. Gap 1: Existing multi-center methods fail to disentangle transferable cross-hospital clinical experience from hospital-specific prescribing patterns, limiting clinical feasibility and physician acceptance. Gap 2: They commonly overlook multi label safety constraints of MR, including drug-drug interaction risks and local formulary feasibility, which are essential for safe real-world deployment. To address these gaps, we propose HEM-Med, a novel
    
[^13]: 快手摘要注意力 V2 技术报告

    Kwai Summary Attention V2 Technical Report

    [https://arxiv.org/abs/2610.18654](https://arxiv.org/abs/2610.18654)

    提出KSA V2架构，通过允许不同层设置不同的摘要压缩分块窗口，实现更灵活的KV缓存内存压缩，并解决全注意力层中摘要标记带来的额外计算开销问题。

    

    arXiv:2610.18654v1 公告类型：cross  摘要：诸如快手摘要注意力（KSA）之类的摘要标记架构，将远距离上下文压缩为每个分块中可学习的摘要标记，混合变体则将这些层与全注意力层交错排列，以保留对原始文本的精确访问。在这种混合 KSA-Full 架构中，交错插入的摘要标记在不同层之间充当流式传输的角色。尽管有效，但它也引入了一些瓶颈：(1) 不同层的摘要压缩分块必须相同，因此难以实现灵活的 KV 缓存压缩，例如无法为不同层设置不同的摘要压缩分块窗口。(2) 在全注意力层中，摘要标记引入了额外的计算成本，但该层却忽略它们，仍然支付二次方复杂度的预填充开销，因此对计算端不友好。受此启发，我们开发了 KSA V2，它不仅实现了更灵活的 KV 缓存内存压缩……

    arXiv:2610.18654v1 Announce Type: cross  Abstract: Summary-token architectures such as Kwai Summary Attention (KSA) compress distant context into learnable Summary token per chunk, and hybrid variants interleave these layers with full attention to retain exact access to the original text. In such hybrid KSA-Full architecture, the interleaving inserted summary tokens are served as a streaming flow role across different layers. Although effective, however, it also introduce some bottlenecks: (1) the different layers summary compression chunk must be same, therefore hard to reach a flexible KV Cache compression, e.g., different layer set different summary compression chunk window. (2) in the full-attention layers, the summary token introduced additional computation cost, but such layer ignore them and still pay quadratic prefilling, therefore unfriendly to the computation side. Motivated by them, we developed the KSA V2, which not only achieves a more flexible KV cache memory compression 
    
[^14]: RIT-RAG：利用检索诱导树导航文档语料库

    RIT-RAG: Navigating Document Corpora with Retrieval-Induced Trees

    [https://arxiv.org/abs/2610.11370](https://arxiv.org/abs/2610.11370)

    RIT-RAG将内容检索与文档结构导航相结合，离线为每篇文档构建目录树，查询时先检索宽泛的文本块并利用其位置诱导出可跨多个文档的、规模可控的子树，再由LLM智能体在子树中导航，从而在保留结构感知优势的同时克服了现有方法无法扩展到大型语料库且选错文档后无法恢复的缺陷。

    

    检索增强生成（RAG）将语言模型建立在外部语料库之上。智能体式RAG（Agentic RAG）支持迭代式搜索，但使模型只能接触到缺乏文档结构的孤立文本块，难以区分真正相关的证据与仅在表面上与查询相似的文本块。诸如PageIndex之类的结构感知方法虽然能够导航文档结构，但无法扩展到大型语料库的结构，因为这些结构无法容纳于LLM的上下文中。因此，这类方法首先需要通过文档检索器确定单一文档，一旦选择错误便无法恢复。我们提出RIT-RAG（检索诱导树RAG），将内容检索与结构导航相结合。在离线阶段，RIT-RAG根据每个文档的目录或站点地图为其构建一棵树。在查询时，它检索一组宽泛的文本块，并利用这些文本块的位置来诱导出规模可控的子树，这些子树可能跨越多个文档。LLM智能体在这些子树中进行导航，有选择地……（原文摘要在此处截断）

    arXiv:2610.11370v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) grounds language models in external corpora. Agentic RAG enables iterative search, yet exposes the model to isolated chunks without document structure, making it difficult to distinguish relevant evidence from chunks that merely resemble the query. Structure-aware methods such as PageIndex navigate document structure but cannot scale to the structures of large corpora, which do not fit in the LLM context. Hence, they first commit to a single document using a document retriever and cannot recover from a wrong choice. We propose RIT-RAG (Retrieval-Induced Tree RAG), which combines content retrieval with structural navigation. Offline, RIT-RAG builds a tree for each document from its table of contents or sitemap. At query time, it retrieves a broad set of chunks and uses their positions to induce manageable sub-trees, potentially across multiple documents. An LLM agent navigates these sub-trees, selectiv
    
[^15]: UniRec：面向级联推荐系统的跨阶段多任务融合与偏好对齐

    UniRec: Cross-stage Multi-Task Fusion with Preference Alignment for Cascaded Recommender Systems

    [https://arxiv.org/abs/2609.11052](https://arxiv.org/abs/2609.11052)

    UniRec提出了一种统一的跨阶段推荐融合模型，通过共享嵌入和单一计算图联合训练实现跨阶段梯度传播，并结合双轴偏好对齐机制，解决级联推荐系统中预排序与排序阶段的跨阶段不一致问题。

    

    工业推荐系统采用具有不同目标、特征空间和延迟约束的级联阶段。分别优化预排序和排序可能导致跨阶段不一致：上游模型可能会过滤掉下游排序器所偏好的物品，而独立调优的下游融合可能会抵消上游的改进效果。现有多任务融合方法主要关注排序阶段内的多目标融合，而跨阶段方法通常只是在上游排序中添加下游分数因子。对两个阶段融合模块的联合优化在很大程度上仍未被探索。我们提出了UniRec，一个统一的跨阶段推荐融合模型。首先，两个融合代理部分共享输入嵌入，并在单一计算图中进行训练，因此来自任一阶段的梯度都会通过共享表示进行传播并影响另一个阶段。其次，我们引入了双轴偏好对齐

    arXiv:2609.11052v1 Announce Type: new  Abstract: Industrial recommender systems use cascaded stages with different objectives, feature spaces, and latency constraints. Optimizing pre-ranking and ranking separately can create cross-stage inconsistency: upstream models may filter out items preferred by downstream rankers, and independently tuned downstream fusion can offset upstream improvements. Existing multi-task fusion methods focus on multi-objective fusion within the ranking stage, and cross-stage methods typically only add a downstream score factor to upstream ranking. Joint optimization of fusion modules across both stages remains largely unexplored.   We propose UniRec, a Unified Cross-stage Recommendation Fusion model. First, the two fusion agents partially share input embeddings and are trained in a single computation graph, so gradients from either stage propagate through the shared representation and influence the other. Second, we introduce a dual-axis preference alignment 
    
[^16]: KGVoyager：基于智能体导航的知识图谱无关问答

    KGVoyager: Knowledge Graph Agnostic Question Answering via Agentic Navigation

    [https://arxiv.org/abs/2609.01780](https://arxiv.org/abs/2609.01780)

    KGVoyager提出了一种知识图谱无关的智能体架构，通过“思考-行动-观察”循环动态探索图谱结构与语义，仅需查询端点即可从自然语言问题生成SPARQL查询，在四个基准上将F1提升约8分，同时将成本与运行时间各降低约22%。

    

    在特定领域场景下，基于RDF图谱的知识图谱问答（KGQA）仍然充满挑战，因为形式化本体和人工整理的文本-SPARQL配对数据通常不可获得。我们提出了KGVoyager，这是一种知识图谱无关的智能体架构，能够从自然语言问题生成SPARQL查询，其方式是动态发现图谱的结构与语义，且仅需要底层图谱的一个查询端点。通过采用配备搜索、探索和执行工具的“思考-行动-观察”循环，KGVoyager能够将术语映射到图谱的IRI、发现图谱结构，并通过执行反馈不断改进查询——所有这些都不需要预先存在的本体或示例。与之前的最先进方法不同，KGVoyager仅需要一个轻量级的类索引，这使其能够应用于更多的真实世界端点。在四个基准测试中，KGVoyager将F1分数提升了约8个百分点，同时将成本和运行时间各降低了约22%。

    arXiv:2609.01780v1 Announce Type: new  Abstract: Knowledge Graph Question Answering (KGQA) over RDF graphs remains challenging in domain-specific settings, where formal ontologies and curated text-SPARQL pairs are often unavailable. We present KGVoyager, a KG-agnostic agentic architecture that generates SPARQL queries from natural language questions by dynamically discovering graph structure and semantics, requiring only a query endpoint of the underlying graph. Using a think-act-observe loop with search, exploration, and execution tools, KGVoyager maps terms to graph IRIs, uncovers structure, and refines queries through execution feedback - all without pre-existing ontologies or examples. Unlike the prior state of the art, KGVoyager requires only a lightweight class index which renders it applicable for far more real-world endpoints. Across four benchmarks, KGVoyager improves F1 by ~8 points while cutting cost and runtime by ~22% each.
    
[^17]: EKLAV：通过仅答案监督引出关键思维链学习

    EKLAV: Elicited Key CoT Learning via Answer only superVision

    [https://arxiv.org/abs/2607.25182](https://arxiv.org/abs/2607.25182)

    EKLAV将教师模型的思维链轨迹作为特权输入上下文而非模仿目标，仅对最终答案进行监督训练，在重排序任务上以更低的训练计算成本和推理时间超越了传统推理蒸馏方法。

    

    推理蒸馏训练学生模型复现教师模型的思维链轨迹及其最终答案。我们探讨是否将思维链轨迹用作特权训练上下文比作为模仿目标更为有效。EKLAV将教师模型的推理保留在输入序列中，但仅对最终任务答案进行监督，并采用任务相关的放置策略，将部分或全部思维链轨迹移入提示中。在BRIGHT段落重排序任务上，EKLAV在两个模型家族的所有四个测试检查点上都提升了平均nDCG@10，在48个模型-领域对比中赢得41个，同时将训练计算量减少高达31.6%，推理时间减少高达32.7%。在四个留出的表格重排序基准上，它在四个条件中的三个中表现最佳，平均超过完整思维链蒸馏方法3.0分，超过无推理的仅答案训练5.5分。对生成轨迹的分析表明，EKLAV学生模型会延迟其首次口头表达的（摘要在此处截断）

    arXiv:2607.25182v2 Announce Type: replace-cross  Abstract: Reasoning distillation trains a student to reproduce both the teacher's chain-of-thought trace and its final answer. We ask whether the trace is better used as privileged training context than as an imitation target. EKLAV keeps the teacher's reasoning in the input sequence but supervises only the final task answer, applying a task-dependent placement that moves part or all of the trace into the prompt. On BRIGHT passage reranking, EKLAV improves average nDCG@10 at all four tested checkpoints across two model families, winning 41 of 48 model-domain comparisons while reducing training compute by up to 31.6% and inference time by up to 32.7%. On four held-out table-reranking benchmarks, it is the best of four conditions on three, exceeding full-trace distillation by 3.0 points and answer-only training without reasoning by 5.5 points on average. Analysis of generated traces reveals that EKLAV students defer their first verbalized 
    
[^18]: LLM工具注册表中的面向智能体的信息设计：对修辞、位置与结构的预注册检验

    Agent-Facing Information Design in LLM Tool Registries: A Preregistered Test of Rhetoric, Position and Structure

    [https://arxiv.org/abs/2605.23916](https://arxiv.org/abs/2605.23916)

    研究通过预注册实验发现，LLM智能体的工具选择会被工具描述中的推销性赞美语言显著影响（可提升约43个百分点选中率，甚至选到无法胜任任务的工具），列表排序和结构化字段也会强烈左右选择，因此建议注册表以结构化字段而非营销文本来呈现工具限制。

    

    AI智能体通常从工具注册表中挑选工具，而注册表中每个工具的描述由其提供者撰写。我们探究这些描述中的推销语言是否会改变智能体所选择的工具。我们构建了仅在某一受控因素上存在差异的工具列表对（添加赞美之词、可验证的规范说明、或列表顺序），并让两个OpenAI模型调用其中一个工具。在一项预注册研究中，堆叠使用赞美之词（四种类型组合）使工具的选中率提高了约43个百分点，其效果与可验证规范相当甚至更强。赞美之词还会使部分选择偏向无法完成任务的工具，但很少使选择偏向索要不必要数据访问权限的工具。当列表内容完全相同时，排在首位的工具被选中的概率高出约72个百分点。在包含数值限制的任务中，结构化字段有助于智能体选择具备所需能力的工具；但在字段旁附上提供者的推销文字会削弱甚至消除这一收益。注册表可以将工具限制以字段形式列出，并隐藏推销性文字……（原文截断）

    arXiv:2605.23916v2 Announce Type: replace-cross  Abstract: AI agents often pick tools from registries, where each tool's provider writes its description. We ask whether sales language in those descriptions changes which tool an agent picks. We built pairs of listings differing in one controlled way (added praise, a verifiable specification, or list order) and asked two OpenAI models to call one tool. In a preregistered study, stacked praise (four kinds combined) raised a tool's pick rate by about 43 percentage points, matching or beating a verifiable specification. Praise also pulled some picks toward tools that could not do the task, but rarely toward tools asking for unneeded data access. With identical listings, the first-listed tool was picked about 72 points more often. On tasks with numeric limits, structured fields helped agents pick the capable tool; adding the provider's sales text beside the fields reduced or erased that gain. Registries could list limits as fields, hide sale
    

