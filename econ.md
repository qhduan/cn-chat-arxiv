# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extract Mechanisms from Heterogeneous Effects: Identification Strategy for Mediation Analysis](https://arxiv.org/abs/2403.04131) | 该论文开发了一种新的识别策略，通过将总处理效应进行分解，将复杂的中介问题转化为简单的线性回归问题，实现了因果效应和中介效应的同时估计，建立了新的因果中介和因果调节之间的联系。 |
| [^2] | [require: Package dependencies for reproducible research.](http://arxiv.org/abs/2309.11058) | require 命令是一个用于确保 Stata 软件包依赖兼容的工具，它可以验证、检查和安装软件包，以实现可复现的研究。 |

# 详细

[^1]: 从异质效应中提取机制：中介分析的识别策略

    Extract Mechanisms from Heterogeneous Effects: Identification Strategy for Mediation Analysis

    [https://arxiv.org/abs/2403.04131](https://arxiv.org/abs/2403.04131)

    该论文开发了一种新的识别策略，通过将总处理效应进行分解，将复杂的中介问题转化为简单的线性回归问题，实现了因果效应和中介效应的同时估计，建立了新的因果中介和因果调节之间的联系。

    

    理解因果机制对于解释和概括经验现象至关重要。因果中介分析提供了量化中介效应的统计技术。然而，现有方法通常需要强大的识别假设或复杂的研究设计。我们开发了一种新的识别策略，简化了这些假设，实现了因果效应和中介效应的同时估计。该策略基于总处理效应的新型分解，将具有挑战性的中介问题转化为简单的线性回归问题。新方法建立了因果中介和因果调节之间的新联系。我们讨论了几种研究设计和估计器，以增加我们的识别策略在各种实证研究中的可用性。我们通过在实验中估计因果中介效应来演示我们方法的应用。

    arXiv:2403.04131v1 Announce Type: cross  Abstract: Understanding causal mechanisms is essential for explaining and generalizing empirical phenomena. Causal mediation analysis offers statistical techniques to quantify mediation effects. However, existing methods typically require strong identification assumptions or sophisticated research designs. We develop a new identification strategy that simplifies these assumptions, enabling the simultaneous estimation of causal and mediation effects. The strategy is based on a novel decomposition of total treatment effects, which transforms the challenging mediation problem into a simple linear regression problem. The new method establishes a new link between causal mediation and causal moderation. We discuss several research designs and estimators to increase the usability of our identification strategy for a variety of empirical studies. We demonstrate the application of our method by estimating the causal mediation effect in experiments concer
    
[^2]: 需求：可复现研究的软件包依赖 (arXiv:2309.11058v1 [econ.EM])

    require: Package dependencies for reproducible research. (arXiv:2309.11058v1 [econ.EM])

    [http://arxiv.org/abs/2309.11058](http://arxiv.org/abs/2309.11058)

    require 命令是一个用于确保 Stata 软件包依赖兼容的工具，它可以验证、检查和安装软件包，以实现可复现的研究。

    

    在 Stata 中进行可复现研究的能力常常受到用户提交的软件包缺乏版本控制的限制。本文章介绍了 require 命令，这是一个旨在确保 Stata 软件包依赖在不同用户和计算机系统上兼容的工具。给定一个 Stata 软件包列表，require 验证每个包是否安装，检查最低或准确版本或软件包发布日期，并在研究人员提示时可选择安装该软件包。

    The ability to conduct reproducible research in Stata is often limited by the lack of version control for user-submitted packages. This article introduces the require command, a tool designed to ensure Stata package dependencies are compatible across users and computer systems. Given a list of Stata packages, require verifies that each package is installed, checks for a minimum or exact version or package release date, and optionally installs the package if prompted by the researcher.
    

