# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collusion-Resilience in Transaction Fee Mechanism Design](https://arxiv.org/abs/2402.09321) | 本论文研究了交易手续费机制设计中的防勾结性问题，讨论了多个要求和属性，并指出在存在交易竞争时，任何TFM都无法同时满足所有要求和属性。 |
| [^2] | [Teacher bias or measurement error?.](http://arxiv.org/abs/2401.04200) | 本研究发现，在分配学生到中学阶段时，教师的学术建议对低社会经济地位家庭的学生有偏见。但是，这个偏见可能是由测试成绩的测量误差导致的，测量误差解释了35%到43%的条件SES差距。 |
| [^3] | [Health Impacts of Public Pawnshops in Industrializing Tokyo.](http://arxiv.org/abs/2305.09352) | 该研究发现公共当铺的贷款贡献了历史性的婴儿和胎儿死亡率下降，私人当铺没有这样的影响。 |

# 详细

[^1]: 交易手续费机制设计中的防勾结性

    Collusion-Resilience in Transaction Fee Mechanism Design

    [https://arxiv.org/abs/2402.09321](https://arxiv.org/abs/2402.09321)

    本论文研究了交易手续费机制设计中的防勾结性问题，讨论了多个要求和属性，并指出在存在交易竞争时，任何TFM都无法同时满足所有要求和属性。

    

    在区块链协议中，用户通过交易手续费机制（TFM）进行竞标，以便将其交易包含并获得确认。Roughgarden（EC'21）对TFM进行了正式的处理，并提出了三个要求：用户激励兼容性（UIC），矿工激励兼容性（MIC）以及一种称为OCA-proofness的防勾结性形式。当没有交易之间的竞争时，Ethereum的EIP-1559机制同时满足这三个属性，但当有过多的符合条件的交易无法放入单个区块时，失去了UIC属性。Chung和Shi（SODA'23）考虑了一种替代的防勾结性概念，称为c-side-construct-proofness(c-SCP)，并证明了当交易之间存在竞争时，任何TFM都不能满足UIC、MIC和至少为1的任何c的c-SCP。OCA-proofness断言用户和矿工不应该能够从协议中“偷取”，并且在直觉上比UIC、MIC更弱。

    arXiv:2402.09321v1 Announce Type: cross Abstract: Users bid in a transaction fee mechanism (TFM) to get their transactions included and confirmed by a blockchain protocol. Roughgarden (EC'21) initiated the formal treatment of TFMs and proposed three requirements: user incentive compatibility (UIC), miner incentive compatibility (MIC), and a form of collusion-resilience called OCA-proofness. Ethereum's EIP-1559 mechanism satisfies all three properties simultaneously when there is no contention between transactions, but loses the UIC property when there are too many eligible transactions to fit in a single block. Chung and Shi (SODA'23) considered an alternative notion of collusion-resilience, called c-side-constract-proofness (c-SCP), and showed that, when there is contention between transactions, no TFM can satisfy UIC, MIC, and c-SCP for any c at least 1. OCA-proofness asserts that the users and a miner should not be able to "steal from the protocol" and is intuitively weaker than the
    
[^2]: 教师偏见还是测量误差？

    Teacher bias or measurement error?. (arXiv:2401.04200v1 [econ.EM])

    [http://arxiv.org/abs/2401.04200](http://arxiv.org/abs/2401.04200)

    本研究发现，在分配学生到中学阶段时，教师的学术建议对低社会经济地位家庭的学生有偏见。但是，这个偏见可能是由测试成绩的测量误差导致的，测量误差解释了35%到43%的条件SES差距。

    

    在许多国家，教师的学术建议用于将学生分配到不同的中学阶段。先前的研究表明，低社会经济地位（SES）家庭的学生在标准化考试成绩相同的情况下，与高SES家庭的同龄人相比，他们得到的学术建议较低。通常认为这可能是教师的偏见。然而，如果存在测试成绩的测量误差，这个论断是无效的。本文讨论了测试成绩的测量误差如何导致条件SES差距的偏误，并考虑了三种实证策略来解决这种偏误。使用荷兰的行政数据，我们发现测量误差解释了学术建议中条件SES差距的35%到43%。

    In many countries, teachers' track recommendations are used to allocate students to secondary school tracks. Previous studies have shown that students from families with low socioeconomic status (SES) receive lower track recommendations than their peers from high SES families, conditional on standardized test scores. It is often argued this indicates teacher bias. However, this claim is invalid in the presence of measurement error in test scores. We discuss how measurement error in test scores generates a biased coefficient of the conditional SES gap, and consider three empirical strategies to address this bias. Using administrative data from the Netherlands, we find that measurement error explains 35 to 43% of the conditional SES gap in track recommendations.
    
[^3]: 东京工业化时期公共当铺对健康的影响

    Health Impacts of Public Pawnshops in Industrializing Tokyo. (arXiv:2305.09352v1 [econ.GN])

    [http://arxiv.org/abs/2305.09352](http://arxiv.org/abs/2305.09352)

    该研究发现公共当铺的贷款贡献了历史性的婴儿和胎儿死亡率下降，私人当铺没有这样的影响。

    

    本研究是首次调查收入低下人口财务机构是否促进了历史性的死亡率下降。我们使用战前东京市的区级面板数据发现，公共当铺贷款与婴儿和胎儿死亡率的降低有关，可能是通过改善营养和卫生措施实现的。简单计算表明，从1927年到1935年推广公共当铺导致婴儿死亡率和胎儿死亡率分别下降了6%和8%。相反，私人当铺没有与健康改善的显着关联。我们的发现丰富了不断扩大的人口统计和金融历史文献。

    This study is the first to investigate whether financial institutions for low-income populations have contributed to the historical decline in mortality rates. Using ward-level panel data from prewar Tokyo City, we found that public pawn loans were associated with reductions in infant and fetal death rates, potentially through improved nutrition and hygiene measures. Simple calculations suggest that popularizing public pawnshops led to a 6% and 8% decrease in infant mortality and fetal death rates, respectively, from 1927 to 1935. Contrarily, private pawnshops showed no significant association with health improvements. Our findings enrich the expanding literature on demographics and financial histories.
    

