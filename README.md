# 利用ChatGPT实现中文Arxiv论文的快速阅读

可以直接看每个子项目的markdown文件（例如人工智能领域是cs.AI.md）

也可以通过RSS阅读器订阅每个子项目的xml文件（例如人工智能领域是cs.AI.xml）

项目会基于GitHub Action自动更新（如果不出问题的话）

> 上游 arXiv 的 RSS 偶尔会返回空的 channel（例如停止 announcement、周末、或者 CDN 缓存尚未构建）。
> 脚本在拿到空 feed 时会重试，仍为空则跳过本次更新，**不会清空已有的 md/xml**。
> `latest_updated.txt` 只在内容真的变化时才刷新，所以它停在某一天通常意味着 arXiv 没有新论文。

## 补全残缺的摘要字段

`papers/` 里保存每篇论文的 JSON 缓存。历史上 LLM 偶尔少输出一行，会留下缺
`translated_title` / `translated_abstract` / `tldr` 的记录，这些记录渲染不进 md/xml。

现在抓取流程会**先把论文元数据落盘，再调用 LLM**：即使摘要失败，论文信息也已经保存，
之后只需补字段，不用重新抓取。

```
python backfill.py --dry-run    # 统计还缺多少条、缺什么字段
python backfill.py              # 补全一批（默认 200 条，可用 BACKFILL_LIMIT 调整）
```

也可以直接运行 GitHub Action：

- `Backfill incomplete papers`（手动触发，可指定本次条数）— `.github/workflows/backfill.yml`
- 每天 04:17 UTC 自动跑一次小批量，直到补完为止

订阅地址如：

https://github.com/qhduan/cn-chat-arxiv/raw/master/cs.AI.xml

https://github.com/qhduan/cn-chat-arxiv/raw/master/ + 具体文件

## 领域列表

来自arxiv的rss导出

http://export.arxiv.org/rss/{channel}

valid archive names are:

- astro-ph, cond-mat, cs, econ, eess, gr-qc, hep-ex, hep-lat, hep-ph, hep-th
- math, math-ph, nlin, nucl-ex, nucl-th, physics, q-bio, q-fin, quant-ph, stat.

details:

- astro-ph: Astrophysics, including cosmology, astrophysics of galaxies, stars, the interstellar medium, and other related topics.
- cond-mat: Condensed Matter Physics, including materials science, soft condensed matter, and other related topics.
- cs: Computer Science, including artificial intelligence, databases, programming languages, and other related topics.
- econ: Economics, including economic theory, econometrics, finance, and other related topics.
- eess: Electrical Engineering and Systems Science, including signal processing, control theory, communication systems, and other related topics.
- gr-qc: General Relativity and Quantum Cosmology, including theoretical and experimental aspects of gravitational physics and cosmology.
- hep-ex: High Energy Physics - Experiment, including experimental studies of particle physics.
- hep-lat: High Energy Physics - Lattice, including lattice field theory, lattice gauge theory, and other related topics.
- hep-ph: High Energy Physics - Phenomenology, including theoretical and computational studies of particle physics.
- hep-th: High Energy Physics - Theory, including theoretical studies of particle physics, quantum field theory, and other related topics.
- math: Mathematics, including algebra, geometry, analysis, number theory, and other related topics.
- math-ph: Mathematical Physics, including mathematical aspects of classical and quantum physics, statistical mechanics, and other related topics.
- nlin: Nonlinear Sciences, including chaos theory, fluid dynamics, nonlinear optics, and other related topics.
- nucl-ex: Nuclear Experiment, including experimental studies of nuclear physics.
- nucl-th: Nuclear Theory, including theoretical studies of nuclear physics.
- physics: General Physics, including topics that do not fit into any of the other categories.
- q-bio: Quantitative Biology, including biological physics, computational biology, and other related topics.
- q-fin: Quantitative Finance, including mathematical models and computational methods for finance and economics.
- quant-ph: Quantum Physics, including theoretical and experimental studies of quantum mechanics and quantum information.

### CS的分类

http://export.arxiv.org/rss/cs.{classes}

e.g. http://export.arxiv.org/rss/cs.AI

valid arxiv cs classes are:

- AI, AR, CC, CE, CG, CL, CR, CV, CY, DB,
- DC, DL, DM, DS, ET, FL, GL, GR, GT, HC,
- IR, IT, LG, LO, MA, MM, MS, NA, NE, NI,
- OH, OS, PF, PL, RO, SC, SD, SE, SI, SY

- AI: Artificial Intelligence
- AR: Hardware Architecture
- CC: Computational Complexity
- CE: Computational Engineering, Finance, and Science
- CG: Computational Geometry
- CL: Computation and Language
- CR: Cryptography and Security
- CV: Computer Vision and Pattern Recognition
- CY: Control Theory
- DB: Databases
- DC: Distributed, Parallel, and Cluster Computing
- DL: Digital Libraries
- DM: Discrete Mathematics
- DS: Data Structures and Algorithms
- ET: Emerging Technologies
- FL: Formal Languages and Automata Theory
- GL: General Literature
- GR: Graphics
- GT: Computer Science and Game Theory
- HC: Human-Computer Interaction
- IR: Information Retrieval
- IT: Information Theory
- LG: Machine Learning
- LO: Logic in Computer Science
- MA: Multiagent Systems
- MM: Multimedia
- MS: Mathematical Software
- NA: Numerical Analysis
- NE: Neural and Evolutionary Computing
- NI: Networking and Internet Architecture
- OH: Other Computer Science
- OS: Operating Systems
- PF: Performance
- PL: Programming Languages
- RO: Robotics
- SC: Symbolic Computation
- SD: Sound
- SE: Software Engineering
- SI: Social and Information Networks
- SY: Systems and Control
