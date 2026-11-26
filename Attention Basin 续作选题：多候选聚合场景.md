# Attention Basin 续作选题：多候选聚合场景

Nov 22, 2026 · @Gregory PorterMi

## 结论

推荐把 Attention Basin 的续作放在**测试时扩展里的多候选聚合**：模型先并行采样 N 个候选解（或 N 条并行推理分支、N 个子 agent 的输出），再由 LLM 一次读完全部候选，选出一个或综合出最终答案。GenSelect、self-aggregation、parallel thinking 都属于这一类。

这个场景能拆成两篇不重叠的论文：ICML 做有标准答案的推理任务上的机制分析和去偏方法，ACL 做开放式的多视角综合。动手前先花一周做 go/no-go 小实验。

## 为什么选这个场景

### 1. 和 Attention Basin 的发现结构上完全对口

原文的关键结论是：偏置发生在“结构块”这一层，去掉分隔符 U 形就消失。候选解本来就是编了号、互相独立、顺序无所谓的块，重新排序不花任何代价。候选数 N 本身就是扩展变量，可以研究 basin 随 N 变大怎么变形，这是 ICML 喜欢的 scaling 叙事。

原文只用 5 篇文档，AttnRank 比最强基线 Ascending 只高 0.2–1.2 个点（HotpotQA 44.72 vs 43.55，2WikiMultiHopQA 34.72 vs 34.50），这是续作最容易被审稿人抓的弱点。候选一多，位置效应大得多：[Overwhelmed by Choice](https://arxiv.org/abs/2609.32809) 在 N=160 时，用排列或分组方法把准确率提高了约 20 个百分点。

### 2. 有一个刚提出、还没人解释的问题可以接

[On the Limits of Test-Time Compute for Self-Evaluation](https://arxiv.org/abs/2611.00350)（2026 年 11 月）发现，候选越多，选择准确率和 oracle 的差距越大，聚合类方法甚至掉点。他们把排序误差拆成 bias 和 variance：高算力下 bias 占约 91%–99%，而且不随算力下降。但他们没有做位置或注意力分析，只把候选顺序敏感性列为可能来源。

[ParaThinker](https://arxiv.org/abs/2509.04475) 也观察到，多条推理路径依次铺开编码时，靠前的路径对最终答案贡献更小。续作要回答的问题是：多加算力也消不掉的那部分偏差里，有多少来自 attention basin？

### 3. 有一篇结论相反的工作：既是切入点，也是最大风险

[Overwhelmed by Choice](https://arxiv.org/abs/2609.32809)（2026 年 9 月）发现 N 越大，模型的预测越偏向靠前的候选，但认为注意力分配不能直接解释预测落在哪个位置。这和“注意力越高越容易答对”的命题直接冲突。从他们的论文里看不出分析的是哪几层的注意力。

原文“浅层由位置主导、深层由内容主导”的分层结论，可能正好能化解这个矛盾。如果能证明按块统计的浅层注意力能预测模型选哪个，或说清楚 basin 在大 N 下何时变成“只看前面”，就是一篇 ICML 机制论文。如果验证失败，这条线就站不住。

### 4. 能做出新方法，而不只是换数据集

RAG 里有检索相似度，直接把高相关文档放到高注意力位置就行。聚合时事先不知道哪个候选是对的，AttnRank 没法直接用。可做的方向：

- 用便宜的先验（同一答案出现的次数、模型自信度）决定哪个候选放在高注意力位置
- 用浅层的位置先验，在一次前向里做校准

要比的基线都需要多次调用：SCORPIO（来自 On the Limits）把候选随机分成 K 个一组联合打分；Overwhelmed by Choice 的排列法每个样本 5.76 次调用，分层法 42 次。“调用一次就达到多次调用的效果”是很清楚的卖点。

## 一个场景拆成 ICML + ACL 两篇

同一个场景拆成两篇：ACL 先截稿，ICML 后截稿，两篇的任务、指标和贡献必须明显分开。

| 论文 | 任务 | 主要贡献 |
| --- | --- | --- |
| ICML | 有标准答案的推理任务（数学、代码）上的生成式选择与聚合 | 机制分析 + 随 N 变化的 scaling 分析 + 只调用一次的去偏方法 |
| ACL | 开放式多视角综合：汇总多个 agent 的报告、多文档观点摘要、deep research 式多来源综合 | 对上 [ACL 2027](https://2027.aclweb.org/calls/main/) 主题赛道“Homogenization and knowledge collapse in LLMs”：排在中间的观点被 basin 吞掉，综合结果变得同质化；既然偏差来自结构，方法上从输入结构入手 |

**双投风险：**[ICML 往年 CFP](https://icml.cc/Conferences/2026/CallForPapers) 规定，不能投与同时在别处在审的稿件相同或实质相似的论文。ACL 那篇 1 月 4 日投出后，到 ICML 截稿时仍在审。

**时间**：从今天算，ACL 还剩约 6 周，ICML 全文约 9 周。

&#91;embedded content: 截稿时间线 · 1 个小实验，3 个截止\]

ACL 截稿最早，所以小实验的结论要在动笔 ACL 之前出来，它决定两篇各自讲什么故事。

## 看过但没选的场景

下面四个场景，要么已经有很多人做，要么块太少、U 形展不开，要么算力门槛高，都不如多候选聚合。

| 场景 | 判断 |
| --- | --- |
| Agent 工具选择（MCP 大工具列表） | 贴近工业应用，但已有不少工作：[BiasBusters](https://arxiv.org/abs/2510.00307) 已发在 ICLR 2026；[Tool Preferences in Agentic LLMs are Unreliable](https://arxiv.org/abs/2505.18135) 发现两个功能相同的工具里，排在前面的被调用得更多。结论还不一致：[Tool Calling is Linearly Readable and Steerable](https://arxiv.org/abs/2605.07990) 在 15 个工具的列表上没发现对首尾的稳定偏好。适合当 ACL 那篇的补充实验 |
| 两两比较的 LLM 评审、选择题 | 研究已经很多，而且只有 2–4 个块，U 形展不开 |
| 继续做 RAG / few-shot | 原文已经做过，再做会被看成小改进 |
| 改并行思考的位置编码（[ParaThinker](https://arxiv.org/abs/2509.04475)、[LaneRoPE](https://arxiv.org/abs/2605.27570)） | 需要训练，算力要求高；放在论文里作对比和讨论 |

## 一周 go/no-go 小实验

先用一周验证两件事：位置是否明显改变聚合结果，浅层注意力能否预测这种改变。

1. 在 AIME、HMMT、MATH500 上，用 Qwen3-4B/8B 为每道题采 N 个候选并存下来。只保留候选里有对有错的题。
2. 用同一个模型做聚合，把正确候选依次放到每个位置，N 取 4、8、16、32。
3. 记录三条曲线：
   - 选对的准确率随位置怎么变
   - 浅层（第 1–2 层）的块级注意力随位置怎么变
   - 去掉“Solution k”这类分隔符后，曲线会不会变平

**判断标准**

- 仅因位置不同，准确率就差 5 个点以上，而且浅层注意力能预测这种差别：ICML 这条线成立。
- 看到的是“只偏向前面”而不是 U 形：正面回应 Overwhelmed by Choice，论文改讲“N 变大时 basin 怎么变成只看前面”。

注意力只需取浅层、query 位置的那几行，原文用的 3090 级别算力就够。

## 参考文献

- [Attention Basin: Why Contextual Position Matters in Large Language Models](https://aclanthology.org/2026.acl-long.1198/)（ACL 2026，原文）
- [On the Limits of Test-Time Compute for Self-Evaluation](https://arxiv.org/abs/2611.00350)（含 SCORPIO）
- [Overwhelmed by Choice: Studying LLM Decision Making at Scale](https://arxiv.org/abs/2609.32809)
- [ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-time Compute](https://arxiv.org/abs/2509.04475)
- [LaneRoPE: Positional Encoding for Collaborative Parallel Reasoning and Generation](https://arxiv.org/abs/2605.27570)
- [BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](https://arxiv.org/abs/2510.00307)
- [Tool Preferences in Agentic LLMs are Unreliable](https://arxiv.org/abs/2505.18135)
- [Tool Calling is Linearly Readable and Steerable in Language Models](https://arxiv.org/abs/2605.07990)
- [ICML 2027 Dates](https://icml.cc/Conferences/2027/Dates)
- [ACL 2027 Call for Main Conference Papers](https://2027.aclweb.org/calls/main/)
- [ICML 2026 Call for Papers](https://icml.cc/Conferences/2026/CallForPapers)（双投政策）
