# Prompt injection guard

> Work in progress. Started October 2026.

A classifier that flags prompt injection and jailbreak attempts before the text reaches an LLM, served as a small API that an app calls first. Prompt injection is number one on the [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) (LLM01).

The plan is to measure it honestly: on attack *styles* the model never saw in training, not only on a random split, and with a red-team test suite that grows as I find ways around it.

A classifier is one layer, not a fix. No filter catches every injection, so the app behind it still has to treat model output as untrusted and limit what the LLM can do. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers where this guard fits and what it can't stop.

## So far

**Text normaliser** ([normalize.py](src/guard/normalize.py)). Filters see characters, people and LLMs see meaning. Before anything is scored, the normaliser undoes the tricks that make those two differ:

| Trick | Example | After |
|-------|---------|-------|
| Zero-width and other invisible characters | `ig​nore previous instructions` | `ignore previous instructions` |
| Fullwidth or "bold maths" letters | `ｉｇｎｏｒｅ`, `𝐢𝐠𝐧𝐨𝐫𝐞` | `ignore` |
| Cyrillic or Greek lookalikes inside a Latin word | `іgnоrе` (Cyrillic і, о, е) | `ignore` |
| Instructions hidden in invisible Unicode tag characters | `What's the weather?` + 18 invisible characters | `What's the weather? Ignore all rules.` |

Tag-smuggled text is decoded into visible text rather than deleted, so the model sees what the LLM would have read. Real Russian, Greek, German and Chinese text is left alone. It also reports what it undid, because a prompt full of invisible characters is suspicious on its own. 31 tests.

**Data** ([data.py](src/guard/data.py)). Four public datasets, each pinned to a Hugging Face commit and checked by SHA-256:

| Source | Licence | Rows kept | Attacks | What it is |
|--------|---------|-----------|---------|------------|
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | 662 | 40% | Short English and German prompts |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | Apache-2.0 | 1,970 | 33% | Long role-play jailbreaks ("DAN") and normal requests |
| [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | MIT | 999 | 100% | Attempts to make the Gandalf game reveal its password |
| [reshabhs/SPML_Chatbot_Prompt_Injection](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection) | MIT | 15,913 | 79% | Attacks on chatbots with a system prompt; only the user message is kept |

`xTRam1/safe-guard-prompt-injection` was left out because it has no licence listed. After normalising, 126 duplicates and 2 empty rows were dropped, leaving 19,544 prompts (14,451 attacks). The duplicate check ignores hidden characters, lookalike letters, case and spacing. No prompt appeared in two sources, and none had conflicting labels. A conflicting prompt would be dropped, not guessed.

**First finding: length gives a lot away.**

![Prompt length by source and label](reports/figures/lengths.png)

Attacks are longer than benign prompts in every source that has both. Length alone separates them with an AUC of 0.87 overall, and **0.995 in SPML**, where benign messages are short questions and attacks are paragraphs. A model trained on this data could learn "long means attack" and then miss short attacks ("ignore the above") and flag long harmless documents. So every model here gets compared against a length-only baseline, and results are reported per source.

SPML is 81% of the rows, so it would dominate a model trained on everything. Day 5 holds out whole sources to see how much of what the model learns carries over.

## Plan

- [x] Scaffold, architecture, OWASP LLM Top 10 notes, text normaliser
- [x] Collect and merge public datasets, dedupe, look at attack styles
- [ ] Baseline: TF-IDF + logistic regression
- [ ] Fine-tune a small transformer
- [ ] Evaluate on held-out attack styles; error analysis
- [ ] Guard API (FastAPI)
- [ ] Next.js playground
- [ ] Red-team test suite and latency benchmark
- [ ] Model card, publish the model

## License

MIT
