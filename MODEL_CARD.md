---
license: mit
language:
  - en
  - de
tags:
  - prompt-injection
  - jailbreak-detection
  - llm-security
  - text-classification
  - scikit-learn
datasets:
  - deepset/prompt-injections
  - jackhhao/jailbreak-classification
  - Lakera/gandalf_ignore_instructions
  - reshabhs/SPML_Chatbot_Prompt_Injection
---

# Model card: prompt-injection-guard (TF-IDF)

A small, fast classifier that scores how likely a prompt is to be a prompt injection or jailbreak attempt, before it reaches an LLM. It is the model behind the guard API in this repository.

## Model details

- **Type:** TF-IDF features (word 1- and 2-grams, character 3- to 5-grams within words), then logistic regression with balanced class weights. scikit-learn 1.9.1.
- **Input:** one prompt, up to 20,000 characters, after the project's normaliser. The normaliser decodes hidden Unicode tag characters and removes zero-width characters. It also fixes Cyrillic and Greek lookalikes inside Latin words, folds fullwidth letters, joins s p a c e d letters, and adds readable base64 as text.
- **Output:** a score from 0 to 1. The API turns it into a decision:

| Score | Decision | Set so that, on validation data... |
|---|---|---|
| 0.461 or more | block | about 1% of benign prompts get here |
| 0.233 to 0.461 | review | about 5% of benign prompts score above 0.233 |
| below 0.233 | allow | |

- **Files:** no pickle. `vocab_words.json` and `vocab_chars.json` (term to column), `weights.safetensors` (IDF weights and logistic-regression coefficients), `config.json` (vectoriser settings and thresholds), and `manifest.json`, which lists each file's SHA-256. `guard.portable.load(folder, pinned_hash)` checks the manifest against a hash you pin in your own config, checks each file against the manifest, and rebuilds the scikit-learn objects. The rebuilt model scores the same as the trained one: the export checks this to within 1e-12 on 500 real prompts before it writes anything. Total size about 6 MB.
- **Author:** Anoushah Wasif. **License:** MIT.

## Intended use

- A first check in front of an LLM that has tools or sees untrusted text. "Block" refuses the prompt; "review" means log it and take the LLM's tools away for that turn.
- One layer of several. It does not make an LLM safe to give powerful tools to (see [ARCHITECTURE.md](docs/ARCHITECTURE.md), "What a classifier can't do").

**Out of scope:** the only defence against injection; content moderation (toxicity, self-harm); languages other than English and German; scoring documents in bulk without checking false alarms on your own traffic first.

## Training data

Four public datasets, each pinned to a Hugging Face commit and checked by SHA-256. After normalising and removing duplicates, there are 19,544 prompts, 14,451 of them attacks.

| Dataset | Licence | Rows | Attacks |
|---|---|---:|---:|
| deepset/prompt-injections | Apache-2.0 | 662 | 40% |
| jackhhao/jailbreak-classification | Apache-2.0 | 1,970 | 33% |
| Lakera/gandalf_ignore_instructions | MIT | 999 | 100% |
| reshabhs/SPML_Chatbot_Prompt_Injection (user message only) | MIT | 15,913 | 79% |

The split is 70/15/15, stratified by source and label. SPML is 81% of the rows, so it dominates.

## Evaluation

**Random test split** (2,932 prompts):

| | Value |
|---|---:|
| Attacks caught | 97.4% |
| Benign prompts flagged (block) | 0.8% |
| ROC AUC | 0.998 |
| Short attacks caught (84 characters or less) | 88.3% |
| deepset attacks caught | 44% |

A random split mostly measures memory: near-copies of the same templates are on both sides.

**A whole source left out of training and threshold setting** (SPML capped at 3,000 rows):

| Left out | Attacks caught | Benign flagged |
|---|---:|---:|
| deepset | 27% | 1.8% |
| gandalf | 93% | n/a |
| jackhhao | 98% | **55%** |
| SPML | 77% | 4.6% |

**Red team** (30 hand-written prompts): 25 handled correctly.
- **Missed:** ROT13, reversed text, Urdu.
- **Wrongly flagged:** "how do I ignore case in a regex" (review) and "repeat the previous instructions for the bread recipe" (block).

**Latency,** one prompt at a time on a laptop: 2 ms median to score a short prompt, 33 ms at 20,000 characters; 7 to 32 ms through the API.

## Limitations and biases

- **"Benign" means what the training data says it means.** Most benign prompts here are short questions. Instructions such as "translate this", "summarise this article" or character role-play look like attacks to it. Trained without jackhhao, it flagged 55% of jackhhao's benign prompts. Before relying on the 1% figure, measure false alarms on prompts from your own users and reset the threshold.
- **Length leaks in.** Long prompts score higher. Long benign documents get flagged more (89% of long jackhhao benign prompts, against 32% of short ones, in the held-out test).
- **Goal hijacks without override words get through:** "state that X is the best", "write a manifesto for...". On deepset, 44% caught on the random split and 27% on held-out data.
- **Languages:** English and some German. Urdu was missed in the red team; other languages are untested.
- **Encodings it doesn't undo:** ROT13, reversed text, and anything else an attacker can ask the LLM to decode.
- **Pasted documents.** Benign prompts with a long pasted text are the hardest false alarms, because they look like indirect injection.

## Ethical considerations

- Prompts often contain personal data. The API never logs prompt text, only its length and the outcome.
- A guard that blocks normal users gets switched off. That's why the false-alarm budget comes first and the limitations above are stated plainly.
- Datasets of attacks are dual-use. Everything used here is public; nothing new and harmful was written for training. The red-team prompts are tests, not training data.

## How to use

```python
from guard.client import check

verdict = check(prompt, url="https://guard.internal", api_key=key)  # fails closed
if verdict.decision == "block":
    refuse()
elif verdict.decision == "review":
    answer_without_tools()
```

Or run the API (`make serve`) and `POST /v1/check` with `{"text": "..."}` and an `X-API-Key` header.

## Reproducing it

```bash
uv sync && make data baseline   # pinned data, ~1 minute to train
make test redteam bench
```

The model is rebuilt from data, not committed. CI rebuilds it on every pull request and runs the red-team suite against it.
