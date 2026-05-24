# MARS v0.1 — Multi-Agent Research System

**Цель:** обогнать опубликованные baseline'ы крупных LLM на **Auto-Bench**
(causal-graph discovery via interactive intervention) и на **LLM-tractable
subset of AIRS-Bench** при том же inner-LLM (gpt-4o / gpt-4o-mini).

**Pre-registration:** этот документ фиксирует архитектуру **до** прогона
платных свипов. Коммит-хэш будет проставлен при заморозке v0.1.

---

## 1. Why a different system than OLS v0.2

Контр-факты из нашей собственной ablation работы по OLS:

| Что узнали | Дизайн-урок для MARS |
|---|---|
| ClaimStore-as-text-dump **вредит** mini (+0.24 RPS при удалении) | **MemorySelector**: показывать модели лишь top-k релевантных claims, не всю историю |
| AgendaController-priority-queue **вредит** mini (+0.21 RPS при удалении) | **Дроп AgendaController**. Sub-goal partition приходит из adapter'а, без меты-приоритезации |
| Claim-gate сам по себе даёт +0.004 (~ноль) | Force claim emission через **dialogue** (Reflector требует evidence), не через flag |
| FutilityDetector — единственная directionally-consistent ось | **Сохранить как есть** |
| На gpt-4o overlay marginally **вредит** (контекст не bottleneck) | Все доп-структуры должны быть **opt-in**, не append-by-default |
| 4-stage DB sub-goal decomposition + R1-R5 prompt = source of +0.637 gain | **Адаптеры** отвечают за task decomposition; sys prompts остаются directive |

---

## 2. High-level architecture

```
       ┌─────────────────────────────────────────────┐
       │              CO O R D I N AT O R            │
       │   schedules turns, budgets, terminates       │
       └─┬─────────┬─────────┬─────────┬──────────────┘
         │         │         │         │
         ▼         ▼         ▼         ▼
   ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────────┐
   │Generator│ │Reflect │ │Memory  │ │Futility    │
   │         │ │ -or    │ │Selector│ │Detector    │
   │ propose │ │critique│ │ top-k  │ │abandon when│
   │ actions │ │ + ask  │ │ claims │ │ budget>θ   │
   │ + claims│ │evidence│ │ for    │ │ no claims  │
   │         │ │        │ │context │ │            │
   └────┬────┘ └───┬────┘ └────┬───┘ └─────┬──────┘
        │          │           │           │
        └──────────┴──── ClaimStore ───────┘
                          ▲ ▼
                    ┌─────────────┐
                    │   ADAPTER   │  (AutoBench / AIRS / LMW)
                    │ observe()   │
                    │ intervene() │
                    │ submit()    │
                    │ budget_left │
                    └─────────────┘
```

**Четыре агента** + три не-LLM substrate-модуля (ClaimStore,
FutilityDetector, Adapter).

---

## 3. Agents (LLM-driven, narrow contracts)

### 3.1 Generator
**Роль:** на ходу предлагает 1-2 действия (`observe`/`intervene`/`submit`)
и/или утверждение (`assert claim`). Узкий контекст: текущий sub-goal +
top-k claims (от MemorySelector) + history of THIS sub-goal.

**JSON-контракт:**
```json
{"rationale": "...",
 "actions": [{"action": "...", "args": {...}}],
 "claims": [{"op": "assert", "statement": "...", "confidence": 0.0-1.0}],
 "advance_subgoal": false}
```

### 3.2 Reflector
**Роль:** критикует Generator'а **post-hoc**: видит последний ход Generator
(action + claim) и result. Решает одно из:
- `accept` — двинуть дальше
- `require_evidence` — заставить Generator провести ещё одну верификационную интервенцию
- `retract` — попросить отозвать claim как недостаточно обоснованный
- `revise` — переформулировать claim точнее

**JSON-контракт:**
```json
{"verdict": "accept|require_evidence|retract|revise",
 "target_claim": "<statement>",
 "reason": "...",
 "suggested_action": {"action": "...", "args": {...}}  // optional
}
```

Это аналог Reflection Agent в Google co-scientist + scientific-debate.
Force-emission через **dialogue**, не через scaffold-gate.

### 3.3 MemorySelector
**Роль:** перед каждым ходом Generator-а решает, какие top-k claims из
ClaimStore показать ему. Реализация v0.1: **CMI-inspired causal-intervention
selection** — для каждого claim прикинуть, **меняет ли его включение** ответ
Generator-а на тестовый вопрос текущего sub-goal'а. Дроп тех, что либо
не влияют (irrelevant), либо ухудшают (harmful / unstable).

v0.1 упрощение (не полная CMI): **scoring claims по 2 критериям**:
- *recency*: log(1+budget_spent_at_claim) — старее = ниже вес
- *relevance*: cosine между embedding(claim.statement) и embedding(sub_goal.question)
- top-k (k=5) после взвешенной суммы; полный CMI оставить на v0.2

**JSON-контракт:** на вход — Generator's incoming sub-goal + all active claims;
на выход — отсортированный список k claim-id для подкачки в context.

### 3.4 Coordinator (не LLM, scheduler)
**Роль:** routing turns между Generator → adapter execute → Reflector →
[optionally Generator again if verdict ≠ accept]. Управляет budget'ом,
terminating conditions, и счётчиком turns per sub-goal.

Псевдокод:
```
while adapter.budget_left() > 0 and not done:
    sg = next_subgoal_from_adapter()
    if sg is None: break
    for turn in range(MAX_TURNS_PER_SG):
        ctx = MemorySelector.pick(sg, claims=ClaimStore.active())
        resp = Generator.propose(ctx)
        for a in resp.actions:
            adapter.execute(a)
        for c in resp.claims:
            ClaimStore.assert(c, sub_goal=sg)
        if resp.claims or resp.actions:
            ref = Reflector.review(resp, last_result=...)
            if ref.verdict == "retract":
                ClaimStore.retract(ref.target_claim)
            elif ref.verdict == "require_evidence":
                # next turn: Generator gets a hint via history
                continue
            elif ref.verdict == "revise":
                ClaimStore.retract(...); continue
            elif ref.verdict == "accept":
                break
        FutilityDetector.check(...)
```

**Termination:** budget ≤ 0, или MAX_TURNS hit, или Reflector accepted final
`submit_hypothesis`.

---

## 4. Substrate (reused from OLS, домен-агностично)

- `mars/core/claim_store.py` — re-export `ols.core.ClaimStore` (с
  CMI-friendly API для `assert(c, sub_goal=…)` и `top_k_for(query, k)`).
- `mars/core/abandon.py` — re-export `ols.FutilityDetector`.
- `mars/adapters/base.py` — re-export `ols.adapters.base.ResearchEnvAdapter`.

Что **выкидываем** из OLS:
- ❌ `AgendaController` (priority queue) — показал вред.
- ❌ `require_claim_before_advance` gate — заменён на dialogue с Reflector'ом.

---

## 5. Adapters (revised after benchmarks-landscape recon)

**Решение после landscape recon** (`BENCHMARKS_LANDSCAPE.md`):
Auto-Bench (нет public code → re-implementation = слабее claim) и AIRS-Bench
(нужен ~$10-20k GPU compute) **дропнуты**. Новый портфель — 4 бенча с public
code, все API-tractable в $500 budget.

### 5.1 `mars/adapters/lmw_adapter.py` — dev-loop (бесплатный)
Re-use `ols.adapters.lmw_adapter`. **Не публикуется как результат**, только
для smoke-tests и regression-checks.

### 5.2 `mars/adapters/mlagentbench_adapter.py` — **Tier-1**
Wrap [snap-stanford/MLAgentBench](https://github.com/snap-stanford/MLAgentBench)
harness. 13 ML-experimentation задач (CIFAR-10 → BabyLM). Baseline для
побития: **Claude 3 Opus 37.5%** average success rate (Huang et al. 2023).
Bug бюджет: ~$100-200.

### 5.3 `mars/adapters/scienceagentbench_adapter.py` — **Tier-1**
Wrap [OSU-NLP-Group/ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench).
ICLR'25, 102 задачи из 44 peer-reviewed папир в 4 дисциплинах (биоинф/хим/гео/др).
Containerized harness — 30 минут на 8 потоках для всех 102 задач.
Baseline для побития: GPT-4o ~30%+ (точная цифра в статье).
Бюджет: ~$100-150. Стартуем с subset 30 задач, расширяем по результатам.

### 5.4 `mars/adapters/discoverybench_adapter.py` — **Tier-2 / convergent-validity**
Re-use наш existing OLS-DiscoveryBench adapter из `ols/adapters/discoverybench_adapter.py`,
расширяем с 25 train-задач до полного labeled set. Convergent-validity control:
если MARS улучшает там же, где OLS дал null — внутренняя консистентность.
Бюджет: ~$30.

### 5.5 `mars/adapters/mlrcbench_adapter.py` — **Tier-2 / easy-win**
Wrap [MLRC-Bench HF space](https://huggingface.co/spaces/launch/MLRC_Bench).
7 ML-research-competition задач. Текущий SoTA — `gemini-exp-1206` всего **9.3% gap
closure**. Низкая планка → быстрая «headline» победа в paper.
Бюджет: ~$50-100.

**Optional v0.2 additions:** HeurekaBench (биология, Jan 2026), LMR-Bench
(NLP-research reproduction, нужен sandbox), CORE-Bench (Princeton, paper-repro).

---

## 6. Pre-registered hypotheses (revised for new bench portfolio)

**H1 (MLAgentBench primary):** MARS / gpt-4o обгонит published Claude 3 Opus
baseline 37.5% average success rate на 13 MLAgentBench задачах, с 95% CI
excluding 37.5%, при N ≥ 3 сидов на задачу.

**H2 (ScienceAgentBench coverage):** MARS / gpt-4o-mini улучшает успех на
ScienceAgentBench относительно published GPT-4o ReAct baseline на subset из
30+ задач (стартовая выборка), с paired sign-test p < 0.05.

**H3 (Component):** Reflector — наиболее load-bearing компонент (>50% gain);
MemorySelector — secondary; FutilityDetector — marginal. Проверяется ablation:
MARS-full vs MARS-no-reflect vs MARS-no-mem-select vs MARS-no-futility.

**H4 (MLRC-Bench easy-win / sanity):** MARS gap-closure > 9.3% (текущий SoTA
gemini-exp-1206) на ≥4 из 7 задач MLRC-Bench. Это «sanity check» — если
не побьём такой низкий baseline, вся архитектура под вопросом.

**H5 (DiscoveryBench convergent validity):** MARS показывает позитивный Δ
HMS относительно нашего OLS-v0.2 null на тех же DB-Real-train задачах
(N=25), на тех же inner-LLM (gpt-4o-mini). Это внутренняя проверка: если
новая архитектура реально улучшает то, где OLS дал null, мы воспроизвели
своё собственное негативное-в-позитивное.

Все H1-H5 могут опровергнуться — это и есть pre-registration.

---

## 7. Cost & timeline

| Phase | Time | API cost | Compute |
|---|---|--:|---|
| 0 — recon (done) | — | $0 | — |
| 1 — spec + impl | 5-7 days | $5 (dev smoke) | — |
| 2 — adapter clone/reimpl | 3-5 days | $20 | — |
| 3 — pre-reg small sweeps | 5 days | $50 | — |
| 4 — power sweeps + ablation | 10-14 days | $300-400 | — |
| 5 — paper writeup | 7-10 days | — | — |
| **Total** | **~6 weeks** | **~$400** | API only |

---

## 8. Open questions / risks

1. **Auto-Bench re-implementation accuracy**: без авторского кода наша версия
   может разойтись по edge cases. Митигация — после Phase 2 прогнать GPT-4o
   на нашей версии и сверить с Table 1 в статье; если расходимость > 10
   процентных пунктов, дебажить.

2. **CMI selection cost**: для каждого ClaimStore.assert делать
   counterfactual call к LLM — дорого. v0.1 fallback: embedding+recency
   scoring; полный CMI отложить.

3. **Reflector loop blow-up**: Reflector может бесконечно требовать evidence.
   Hard cap MAX_TURNS_PER_SG = 4 и Coordinator принудительно принимает
   последнее предложение, помеченное как `provisional`.

4. **AIRS-Bench harness совместимость**: их MLGym/AIRA-dojo формат может
   потребовать существенной обвязки. Если Phase 2.3 займёт >5 дней — drop
   AIRS subset до 3-4 простейших задач.

---

## 9. Что есть **не делаем** в v0.1

- Не делаем multi-LLM scientific-debate self-play (как в Google
  co-scientist). Слишком дорого; v0.2 candidate.
- Не делаем Bayesian belief updating (Bayes-Entropy paper). v0.2.
- Не делаем MCTS-style hypothesis trees (Iterative Nash). v0.2.
- Не делаем cross-task learning / persistence beyond episode. v0.2.
