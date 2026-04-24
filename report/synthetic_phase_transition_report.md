# Synthetic Phase-Transition Report

## Source families
- IID categorical (Zipf/random-softmax, entropy-controllable)
- First-order Markov (sparsity/mixing/regime control)
- HMM latent-state source (forward likelihood)

## Watermark
KGW-style red/green token partition with seeded hash PRF, green fraction `gamma`, bias `delta`.

## Detectors
- Optimal LLR: `log q(x) - log p(x)`
- Green-token z-score detector

## Attackers
- Information-theoretic exact attacker (tractable small settings)
- Greedy local attacker
- Beam-search attacker
- MCMC random-walk attacker

## Main findings
- hmm_latent: max observed LLR AUC = 1.000.
- iid_zipf: max observed LLR AUC = 1.000.
- markov_medium: max observed LLR AUC = 1.000.

## Statistical-computational gap (empirical proxy)

## Sanity checks
- `delta=0` approximately collapsed to random detection in low-n settings.
- Detectability generally improved with larger `n` and `delta`.
- Lower quality tolerance reduced removal success.
- Exact attacker upper bound was never worse than bounded attackers in tested tiny regimes.

## Figures
- Figure 1 detectability diagrams: `results/figures/figure1_detectability_llr.png`, `..._z.png`
- Figure 2 removability diagrams: `results/figures/removability_*`
- Figure 3 gap region: `results/figures/gap_region_*_greedy.png`
- Figure 4 scaling curves: `results/figures/scaling_*`
- Figure 5 detector comparison: `results/figures/figure5_detector_comparison.png`
- Figure 6 quality-removal frontier: `results/figures/figure6_quality_removal_frontier.png`