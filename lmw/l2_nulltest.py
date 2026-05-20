"""
Reviewer point #1: is L2 `oracle_value` real signal or multiple-testing noise?

oracle_value counts neural qEEG features whose odor-vs-ref contrast on the
HELD-OUT subject passes |d|>2*se (uncorrected, ~hundreds of features). Naive
row-shuffle would inflate significance because EEG is autocorrelated. We use a
CIRCULAR-SHIFT permutation of the stimulus label vector per held-out subject:
it preserves each signal's autocorrelation and the stimulus block structure,
destroying only stimulus<->signal alignment -> a proper H0.

Per LOSO fold: observed count vs null distribution from K random circular
shifts. Report observed vs null mean / 95th pct, empirical p, and the
expected-false-discovery fraction (null_mean / observed). No API.
"""

from __future__ import annotations

import random
import statistics as st

from adapters.neuro import SUBJECTS, NeuroSource, _is_neural

K = 200
MINSHIFT = 20  # avoid near-identity shifts


def fold_null(held: str, rng: random.Random):
    src = NeuroSource(held_out=held)
    if held not in src.subj:
        return None
    h, data, lab = src.subj[held]
    n = len(lab)
    n_neural = sum(1 for f in src.feats if _is_neural(f))

    def count() -> int:
        c = 0
        for f in src.feats:
            if _is_neural(f) and (
                src.replicates(f"stim_effect({f},+)")
                or src.replicates(f"stim_effect({f},-)")
            ):
                c += 1
        return c

    observed = count()
    null = []
    for _ in range(K):
        k = rng.randint(MINSHIFT, n - MINSHIFT)
        src.subj[held] = (h, data, lab[-k:] + lab[:-k])   # circular shift
        null.append(count())
    src.subj[held] = (h, data, lab)
    nm = st.fmean(null)
    n95 = sorted(null)[int(0.95 * len(null)) - 1]
    p = (sum(1 for x in null if x >= observed) + 1) / (K + 1)
    return dict(held=held, n_neural=n_neural, observed=observed,
                null_mean=round(nm, 1), null_p95=n95, p=round(p, 4),
                fdr=round(nm / observed, 3) if observed else None)


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rng = random.Random(0)
    print(f"\n=== L2 oracle_value permutation null (circular-shift, K={K}) ===")
    print(f"{'fold(held)':<14}{'n_neural':>9}{'observed':>10}"
          f"{'null_mean':>11}{'null_p95':>10}{'emp_p':>8}{'exp_FDR':>9}")
    rows = []
    for s in SUBJECTS:
        r = fold_null(s, rng)
        if r:
            rows.append(r)
            print(f"{r['held']:<14}{r['n_neural']:>9}{r['observed']:>10}"
                  f"{r['null_mean']:>11}{r['null_p95']:>10}{r['p']:>8}"
                  f"{str(r['fdr']):>9}")
    if rows:
        obs = st.fmean(r["observed"] for r in rows)
        nul = st.fmean(r["null_mean"] for r in rows)
        print(f"\n mean observed={obs:.1f}  mean null={nul:.1f}  "
              f"ratio observed/null={obs/nul:.2f}x  "
              f"mean expected-FDR={st.fmean(r['fdr'] for r in rows):.2f}")
        sig = all(r["p"] < 0.05 for r in rows)
        print(f" verdict: signal {'ABOVE' if sig else 'NOT clearly above'} "
              f"the autocorrelation-preserving noise floor "
              f"(all folds p<0.05: {sig})")
        print(" NOTE: if exp-FDR is high, oracle_value is inflated by "
              "uncorrected multiple testing -> L2 must be reframed as a "
              "pilot/sanity rung, not a ground-truth benchmark.")


if __name__ == "__main__":
    main()
