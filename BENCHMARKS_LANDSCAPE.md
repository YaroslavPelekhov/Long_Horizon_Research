# Бенчмарки для AI-research / autonomous-science агентов — полный обзор

**Дата:** 2026-05-21
**Цель:** выбрать бенчи для MARS-системы с учётом (а) наличия публичного кода
для воспроизводимости, (б) API-only бюджета ~$500, (в) baseline-числа,
которые осмысленно бить.

---

## ✅ Бенчи с публичным кодом и API-tractable (наши кандидаты)

| Бенч | Год · паб. | Задачи | SoTA baseline | Код | Compute | Подходит? |
|---|---|---|---|---|---|---|
| **MLAgentBench** | 2023, Stanford | 13 ML-experimentation задач | **Claude 3 Opus 37.5%** | [snap-stanford/MLAgentBench](https://github.com/snap-stanford/MLAgentBench) | API + light compute | **✅ Sweet spot** |
| **ScienceAgentBench** | ICLR'25, OSU | 102 задачи из 44 peer-reviewed папир (биоинф/хим/гео) | GPT-4o ~30%+ | [OSU-NLP-Group/ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench), containerized harness (30 min, 8 threads) | API + light compute | **✅ Сильный вариант** |
| **DiscoveryBench** | NeurIPS'24, AI2 | 264 задачи hypothesis-from-data (real + synthetic) | Best 25% HMS | [allenai/discoverybench](https://github.com/allenai/discoverybench) | API-only | **✅ Уже работали** |
| **HeurekaBench** | Jan 2026, EPFL mlbio | AI co-scientist eval, biology/genomics | Claude 4 / Sonnet 4.5 / GPT-4 / Qwen 3 / Gemini tested | [mlbio-epfl/HeurekaBench](https://github.com/mlbio-epfl/HeurekaBench) | API + biology data | **✅ Recent, hot** |
| **MLRC-Bench** | Apr 2025 | 7 ML research-competition задач | gemini-exp-1206 9.3% gap closure | [HuggingFace space](https://huggingface.co/spaces/launch/MLRC_Bench) | API + light | **✅ Low bar = easy win** |
| **CORE-Bench** | Sep 2024, Princeton | 270 задач paper reproduction (CS/SS/Med, py/R) | n/a в нашем поиске | Inspect AI repo, [arxiv 2409.11363](https://arxiv.org/abs/2409.11363) | API + code-exec | ⚠️ нужен code-exec |
| **LMR-Bench** | Jun 2025 | 28 задач LM-research reproduction (23 папир, 9 категорий) | "advanced models still limited" | Не показан в abstract | API + code-exec | ⚠️ нужен code-exec |
| **SciReplicate-Bench** | Apr 2025 | агентная репродукция алгоритмов из папир | n/a | [arxiv 2504.00255](https://arxiv.org/abs/2504.00255) | API + code-exec | ⚠️ |
| **ResearchCodeBench** | Jun 2025 | LLM → код из 2024-2025 ML папир | n/a | n/a | API + code-exec | ⚠️ |
| **LLM-SRBench** | ICML'25 Oral | 239 задач scientific equation discovery | n/a | [deep-symbolic-mathematics/llm-srbench](https://github.com/deep-symbolic-mathematics/llm-srbench) | API-only | **✅ Уравнения, niche** |
| **ResearchBench** | Mar 2025 | LLM в scientific discovery via inspiration-task decomposition | n/a | n/a verified | API | ✅ |
| **IdeaBench** | KDD'25 | research idea generation, biomedical | n/a | n/a | API | ✅ |
| **AI Idea Bench 2025** | Apr 2025 | quantitative idea eval, 3,495 AI папир + inspired works | n/a | n/a | API | ✅ |
| **HypoSpace** | Oct 2025 | set-valued hypothesis generation under underdetermination (causal/voxel/genetic) | mode collapse | repo упомянут | API | ✅ |

## ⛔ Бенчи **без публичного кода** или **слишком тяжёлый compute**

| Бенч | Проблема |
|---|---|
| **Auto-Bench** (Chen et al., Feb 2025) | **Код не выпущен.** Ни на сайтах соавторов (Vedant Shah / Goyal), ни в репо Mila / NUS. Только paper. Re-implementation = weakened claim. **DROP** |
| **AIRS-Bench** (Meta, Feb 2026) | Код есть [facebookresearch/airs-bench](https://github.com/facebookresearch/airs-bench), но **24h H-200 GPU × 10 сидов × 20 задач** — ~$10-20k compute. **DROP** для API-only бюджета |
| **MLE-bench** (OpenAI, Oct 2024) | Код [openai/mle-bench](https://github.com/openai/mle-bench) есть, но требует **36 vCPU + 440GB RAM + 24GB A10 GPU**, 158GB-3.3TB данных. SoTA Famou-Agent 2.0 = 64.44%. **Не API-only** |
| **MLAgentBench** (Stanford 2023) | После клонирования: 13 задач — CIFAR-10, ogbn-arxiv, babylm, fathomnet и т.д. → **реально тренируют модели**, GPU необходим (Claude 3 Opus 37.5% — на GPU-setup). **DROP** для API-only |
| **MLRC-Bench** (Apr 2025) | После клонирования: **построен поверх MLAgentBench**, launch.sh принимает `${GPU_ID}`, задачи (llm-merging Llama-3-8B, machine_unlearning torchvision, weather_forecast и т.д.) → CUDA-12.1 environments, GPU необходим. **DROP** |
| **Scientist-Bench** (часть AI-Researcher) | Код в [HKUDS/AI-Researcher](https://github.com/HKUDS/AI-Researcher) — но это framework вместе с бенчем; нужно изолировать только бенч-часть |

## 🎯 Окончательная картина API-only viable бенчей

Из всех ~15 проверенных, **API-only viable** только те, где задача — это **анализ данных / hypothesis generation / code-gen на маленьких датасетах**, а не реальное model training:

| Бенч | Тип | Adapter status |
|---|---|---|
| **ScienceAgentBench** | code-gen, eval через docker/LLM-judge | ✅ MARS adapter v0.1 готов (LLM-judge proxy) |
| **DiscoveryBench** | hypothesis-from-data | ✅ адаптер из OLS-времён, нужно расширить |
| **HypoSpace** | set-valued hypothesis gen | ⚠️ нужен adapter |
| **IdeaBench** | biomedical idea generation | ⚠️ нужен adapter |
| **HeurekaBench** | AI co-scientist (biology) | ⚠️ нужен adapter |

**Все «ML research engineering»-бенчи (MLAgentBench / MLRC / MLE / AIRS) дроп для API-only.**
Если будет доступ к GPU — можно вернуться к ним отдельной фазой.

## 📚 Связанные / косвенно релевантные

- **GAIA** (Mialon et al., 2023) — general AI assistants, multi-step real-world questions
- **CUB** — common-use benchmark
- **AgentBench** — general agent eval suite
- **TableBench** (2025) — таблица-анализ
- **DS-1000** (Lai 2023) — data-science code-gen
- **EXPBench / HyperBench / WritingBench** — упомянуты в survey 2510.23045, репо/детали не подтверждены
- **CausalGraph2LLM** (NAACL'25, [ivaxi0s/CausalGraph2LLM](https://github.com/ivaxi0s/CausalGraph2LLM)) — causal queries eval, **не интерактивный**, но релевантный

## 🔍 Master-индекс

Самый полный публичный список — survey paper [HKUST-KnowComp/Awesome-LLM-Scientific-Discovery](https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery) (EMNLP 2025 *From Automation to Autonomy*). Стоит проверить там же — обновляется.

---

## 🎯 Рекомендуемый новый target set для MARS

Дроп Auto-Bench. Новый портфель из **3-4 публичных бенчей**, выбранных по
(а) public code, (б) API-only, (в) defensible baseline:

### Tier-1 (основные таргеты, paper-grade)
1. **MLAgentBench** (13 задач, Claude Opus baseline 37.5%) — заметный бенч, реальная ML-инженерия, бить Claude Opus = весомый claim.
2. **ScienceAgentBench** (102 задачи) — широкое покрытие, fast containerized harness, ICLR'25 — престиж.

### Tier-2 (control / диагностика)
3. **DiscoveryBench** — уже работаем с ним; hypothesis-from-data; используем как convergent validity test.
4. **MLRC-Bench** (7 задач, gemini SoTA 9.3% gap closure) — низкая планка = быстрая победа для demonstration.

### Optional
5. **HeurekaBench** — если хотим AI co-scientist territory (biology).

**Бюджет под этот сет:**
- MLAgentBench: ~$100-200 (13 задач × ~5 моделей-сидов × $1-3/run)
- ScienceAgentBench: ~$100-150 (subset 30-50 задач из 102, container быстрый)
- DiscoveryBench: ~$30 (расширить с 25 до 100 задач)
- MLRC-Bench: ~$50-100 (7 задач, dense sweep)

Итого: **~$300-500**, реалистично в обещанном бюджете.

---

## Что важно для MARS-архитектуры в свете этого

MARS spec (`MARS_SPEC_v0.md`) остаётся валиден, но **адаптеры меняются**:
- Удалить запланированный `autobench_adapter.py` (re-implementation)
- Удалить `airsbench_text_adapter.py` (compute проблема)
- Добавить:
  - `mlagentbench_adapter.py`
  - `scienceagentbench_adapter.py`
  - `mlrcbench_adapter.py`
  - Обновить существующий `discoverybench_adapter.py` (расширение)
- Все они уже имеют public code → наша задача — обёртка-адаптер вокруг их harness'а, не реимплементация.
