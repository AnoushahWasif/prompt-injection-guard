# Prompt injection guard

> Work in progress. Started October 2026.

A classifier that flags prompt injection and jailbreak attempts before the text reaches an LLM, served as a small API that an app calls first. Prompt injection is number one on the [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) (LLM01).

The plan is to measure it honestly: on attack *styles* the model never saw in training, not only on a random split, and with a red-team test suite that grows as I find ways around it.

A classifier is one layer, not a fix. No filter catches every injection, so the app behind it still has to treat model output as untrusted and limit what the LLM can do. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers where this guard fits and what it can't stop.

## Plan

- [x] Scaffold, architecture, OWASP LLM Top 10 notes, text normaliser
- [ ] Collect and merge public datasets, dedupe, look at attack styles
- [ ] Baseline: TF-IDF + logistic regression
- [ ] Fine-tune a small transformer
- [ ] Evaluate on held-out attack styles; error analysis
- [ ] Guard API (FastAPI)
- [ ] Next.js playground
- [ ] Red-team test suite and latency benchmark
- [ ] Model card, publish the model

## License

MIT
