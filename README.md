# TAT3
This repository contains the source code for my master's thesis Large Language Model-Driven Data Enrichment-


## Abstract
This thesis introduces the **T**able **A**ugmentation via **T**ext-to-**T**ext **T**ransfer (TAT3) framework that leverages foundational pretrained LLMs to perform subject suggestion. TAT3 suggests missing entities of an incomplete table that consists of rows representing entities. The system is inspired by recent advances in AI Assistants, specifically GitHub Copilot. With TAT3, we address the limitations of previous works that focus on subject suggestion having strict assumptions about the source of knowledge as well as the input and output format. TAT3 achieves similar results to established baselines on a standard subject suggestion benchmark. On the benchmark, a correct first suggestion is found for 73.0% of the queries, which is comparable to our best employed baseline from previous work that has stricter assumptions. For 89.1% of the queries, a correct suggestion is found within the first five suggestions. On average, 63.1% of the first five suggestions are correct. An offline optimization strategy increased TAT3's probability of finding a correct entity within the first suggestion by 6.05% relative to the unoptimized approach.  Ensembling, used to address possible hallucinations of LLMs, increased the probability of finding a correct entity within the first suggestion by 3.95% relative to choosing the best LLM. Additionally, we demonstrate that TAT3 can adapt its output format of the results to the rows already present in a table.
