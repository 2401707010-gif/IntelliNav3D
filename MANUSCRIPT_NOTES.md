# Notes on the manuscript (version 31-10-2026) and how the code handles each point

These are places where the manuscript does not fully specify the method, contradicts itself, or reports a result that
no described experiment could produce. The code makes a documented choice in each case. Values chosen by the code are
marked `not in paper` in `configs/default.yaml`.

## A. Results the code cannot reproduce as printed

1. **Table 12 (significance tests against SemLayoutDiff and CHOrD).** The text of Section 4.2 says the baseline values
   were copied from the cited papers and not recomputed. A paired test needs per-seed results of both methods on the
   same test rooms, so it cannot be computed from published numbers. The printed mean differences also disagree with
   Table 3: for example, navigation success 96.7 vs 96.0 differs by +0.7, not +8.31. The code computes paired tests
   only against methods it actually evaluated: the ablation variants, plus any baseline you re-run and pass through
   `scripts/evaluate_external.py`.
2. **Table 15.** The "independent dataset" and "physics-simulator execution" rows have no protocol in the paper, and
   Section 4.11 states that neither was done. The code supports a held-out room type (`--stage heldout`) and an
   independent dataset if you supply its rooms in this repository's JSON format (`--stage external --external_dir`).
   It contains no physics simulator, so that row is not produced.
3. **Table 2 vs Section 4.1.1.** Table 2 lists 3,800 test scenes; Section 4.1.1 lists 15,200 / 1,000 / 2,800. The code
   uses the latter, and Table 2 is filled from the split that was actually used.
4. **Table 13, "before PPO" row.** Its navigation success (89.6%) is identical to the "Without DDPM" row of Table 8. In
   the code, "before PPO" is the diffusion candidate that was refined into each final layout, and its success rate is
   measured separately.

## B. Definitions that are ambiguous; the choice made in the code

5. **Human Comfort Score (Eq. 16).** As written (free floor at least d_c = 0.60 m from every furniture footprint, in the
   largest connected region), HC is low in small furnished rooms: about 25% on the procedural test rooms. A mean of
   95.8% is hard to reach with this definition. If your earlier experiments used a different definition or d_c, change
   `monff.comfort_distance` or `navigation.human_comfort`, and make the paper match the code.
6. **Accessibility.** Section 3.7 calls it "free floor area / total floor area"; Algorithm 2 calls it
   |Reach(s1)| / |F| over free (inflated) cells. The code follows Algorithm 2: the share of navigable cells reachable
   from the first start. This is also the quantity compared with "walkable space" in Tables 4 and 5.
7. **Reference clearance d_max for C_s** is not given. The code uses 0.5 m.
8. **Spatial consistency SC.** "One minus the normalised mean deviation of pairwise distances from training
   statistics". The code uses |d − μ|/μ per category pair, clipped to 1.
9. **Displacement penalty λ_d (Section 3.8)** has no value. The code uses 0.05 per metre.
10. **"MONFF stops improving"** is implemented as 10 steps without improvement (`ppo.patience`).

## C. Method details the paper leaves open

11. **GAT training objective.** Not stated. The code pre-trains the GAT to predict relation type and pairwise distance
    of every object pair, with half of the input distances replaced by category-pair priors.
12. **G\* during generation.** Algorithm 1 lists object coordinates as node features, but the coordinates of a layout
    being generated are unknown. The code conditions the DDPM on a template graph built from the object list. Its
    edges are the functional and support relations plus pairs whose training-set mean distance is below the threshold,
    and its edge distances are those means. The PPO state uses the real graph of the current layout.
13. **What the DDPM generates.** Section 3.3 says category, position, size and orientation. The code generates the
    position and orientation of every object in the room's object list; categories and sizes come from that list.
    This is the re-arrangement setting of Fig. 10.
14. **"Navigation-conditioned".** The code conditions on the (ANS, HC) of each training room, with classifier-free
    guidance towards `ddpm.nav_target` at sampling.
15. **Eq. (6)** writes the noise term as α_t z. The standard DDPM sampler, used in the code, adds σ_t z with the
    posterior variance.
16. **"Without DDPM" ablation.** The paper does not say what replaces the generator. The code places each object at
    the normalised position of a random training instance of the same category and room type, then applies PPO.
17. **"Without MONFF" ablation.** The PPO reward and the final selection use ANS alone.
18. **Unconstrained PPO (Table 13).** The constraints of Section 3.8 and the displacement penalty are switched off.
19. **FID renderer.** Section 4.8 mentions Blender 4.3 / Cycles. The default code path renders semantic-colour
    top-down views with PIL. `scripts/render_blender.py` gives a textured Blender alternative. Use the same renderer
    for real and generated rooms, and state in the paper which one you used.
20. **Supported objects** (e.g. a lamp on a nightstand) move with their parent, are linked in the graph only to their
    parent, and are not floor obstacles. Ceiling lamps are dropped during pre-processing.
