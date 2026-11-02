# kcert-experiments: certified, label-free auditing of perception with the rules of an expert system

Code, notebook and full theory for the paper

> **Auditing the perception layer of rule-based expert systems without concept labels: Certified error bounds, drift alarms and shortcut repair.** Satabarto Sarkar and Prithwineel Paul. Submitted to *Expert Systems with Applications*.

The installable library lives in its own repository, [`kcert`](https://github.com/researchworksofsomeone/kcert) (`pip install kcert`). This repository holds everything needed to reproduce the paper: one notebook containing every experiment we ran, the pre-registrations, the result files, and this README, which states and proves every result we use.

---

## Contents

1. [The idea in one paragraph](#1-the-idea-in-one-paragraph)
2. [Repository layout](#2-repository-layout)
3. [Setting and notation](#3-setting-and-notation)
4. [Theory](#4-theory)
   - [4.1 The fibre principle: sharp bounds](#41-the-fibre-principle-sharp-bounds)
   - [4.2 What rules add over tables: the coupling gain](#42-what-rules-add-over-tables-the-coupling-gain)
   - [4.3 Minimax model selection and the certified top set](#43-minimax-model-selection-and-the-certified-top-set)
   - [4.4 Shortcut indistinguishability: where label-free auditing must stop](#44-shortcut-indistinguishability-where-label-free-auditing-must-stop)
   - [4.5 Projection: refuting and repairing any estimator](#45-projection-refuting-and-repairing-any-estimator)
   - [4.6 Select by any estimator, certify by the rules](#46-select-by-any-estimator-certify-by-the-rules)
   - [4.7 Certified drift alarms](#47-certified-drift-alarms)
   - [4.8 Reasoning shortcuts of additive rules: exact count and minimal repair](#48-reasoning-shortcuts-of-additive-rules-exact-count-and-minimal-repair)
   - [4.9 Imperfect rules: k-robust bounds](#49-imperfect-rules-k-robust-bounds)
   - [4.10 Certifying k from a few labels](#410-certifying-k-from-a-few-labels)
   - [4.11 Knowledge-stratified labelling (KSE)](#411-knowledge-stratified-labelling-kse)
   - [4.12 Instance-level versus population-level identification](#412-instance-level-versus-population-level-identification)
   - [4.13 The assumption ladder](#413-the-assumption-ladder)
   - [4.14 What is not claimed](#414-what-is-not-claimed)
5. [Theorem numbering: paper, notebook, library](#5-theorem-numbering-paper-notebook-library)
6. [Experiments](#6-experiments)
7. [Reproducing the results](#7-reproducing-the-results)
8. [Using the library on your own system](#8-using-the-library-on-your-own-system)
9. [Citation and licence](#9-citation-and-licence)

---

## 1. The idea in one paragraph

Many expert systems take their facts from a neural perception model and then apply rules: a driver-assistance system reads the shape, colour and symbol of a sign and then applies the traffic law. The rules state which fact combinations are **legal**, and for each decision which of them are **consistent**. So, if the world obeys the rules, every input's unknown true fact vector lies in a known finite set (its *fibre*). A model's error is then a sum, over inputs, of a quantity that depends on an unknown point of a known set, and its range can be computed exactly. This is partial identification (Manski, 2003) applied to perception. It gives guarantees rather than estimates, with no concept labels, and its only assumption is about the domain, not about the data distribution or the models.

---

## 2. Repository layout

| path | what it is |
|---|---|
| `notebooks/kcert_experiments.py` | the full study as one [marimo](https://marimo.io) notebook: every experiment, every pre-registration, every verdict |
| `notebooks/kcert_experiments.ipynb` | the same notebook exported to Jupyter format |
| `results/` | the result files of the paper run (`paper_numbers.txt`, figures, `runtime.json`), added after the final run |
| `docs/` | extra material (pre-registration log, figures) |
| `CITATION.cff`, `LICENSE` | how to cite, MIT licence |

The notebook is cumulative. Sections 1–11 are the setup and the earlier versions of the theory (kept so that every pre-registration can be checked against the code that scored it), and sections 12–23 are the experiments reported in the paper.

---

## 3. Setting and notation

- **Inputs:** $`i = 1,\dots,n`$ (images, video frames, records).
- **Facts:** the unknown true fact vector of input $`i`$ is $`c_i \in \{0,1\}^K`$ (shape, colour, digit value in one-hot form, ...).
- **Decisions:** the observed decision $`\ell_i`$ (the system's logged action, a task label, an outcome).
- **Rules:** a finite set $`\mathcal V \subseteq \{0,1\}^K`$ of legal fact vectors (*worlds*) and, for each decision $`\ell`$, the consistent subset $`\mathcal V(\ell) \subseteq \mathcal V`$. We call $`\mathcal V(\ell_i)`$ the *fibre* of input $`i`$.
  - A logical program gives arbitrary fibres (traffic law, sums of digits, driving rules).
  - A class-to-attribute table gives *box* fibres $`\mathcal V(\ell) = \prod_j \mathcal V_j(\ell)`$ (CUB birds, a sign catalogue).
- **Models:** perception models $`r = 1,\dots,R`$ output $`\hat c_{ri} \in \{0,1\}^K`$.
- **Error:** $`d(a, v)`$ is the Hamming distance divided by $`K`$, and $`\mathrm{err}_r = \frac1n\sum_i d(\hat c_{ri}, c_i)`$.
- **Forced facts:** fact $`j`$ is *forced* on input $`i`$ when all worlds of the fibre agree on it, $`v_j = f_{ij}`$ for every $`v \in \mathcal V(\ell_i)`$. The *forced fraction* is the share of forced (input, fact) slots; it is computable before any training.

**Assumption (K):** $`c_i \in \mathcal V(\ell_i)`$ for every input.

This is the only assumption of every guarantee below (Section 4.9 relaxes it to "all but $`k`$ inputs"). It is a statement about the world (the traffic law holds, the sum is the sum), not about the data distribution, the shift or the models.

---

## 4. Theory

Every statement below is proved here. The library tests check each one numerically, including brute-force checks of sharpness.

### 4.1 The fibre principle: sharp bounds

**Theorem 1 (fibre principle):** Let $`\Phi = \frac1n\sum_i g_i(c_i)`$ be any additive functional, where $`g_i`$ may depend on everything observed about input $`i`$. Under (K), the set of values $`\Phi`$ can take is exactly

```math
\mathcal I(\Phi) = \Big\{\tfrac1n\textstyle\sum_i g_i(v_i) : v_i \in \mathcal V(\ell_i)\Big\},
```

with minimum $`\frac1n\sum_i \min_{v\in\mathcal V(\ell_i)} g_i(v)`$ and maximum $`\frac1n\sum_i \max_{v\in\mathcal V(\ell_i)} g_i(v)`$, both attained.

**Proof:** (K) constrains each $`c_i`$ separately and nothing else, so the feasible truths are exactly the product $`\prod_i \mathcal V(\ell_i)`$. An additive functional is optimised input by input, and the per-input optimum is attained at a world of the fibre. ∎

**Remark (vector functionals):** for $`g_i \in \mathbb R^d`$ the identified set is the Minkowski sum of the finite sets $`g_i(\mathcal V(\ell_i))/n`$. By the Shapley–Folkman lemma its convex hull is within $`\sqrt d \cdot \max_i \mathrm{diam}`$, that is $`O(d/n)`$, of the set itself, so the convex hull is the sharp set up to $`O(d/n)`$.

Three choices of $`g_i`$ give the tools of the paper.

**Bracket:** $`g_i(v) = d(\hat c_{ri}, v)`$ gives $`[\mathrm{LO}_r, \mathrm{HI}_r] \ni \mathrm{err}_r`$, the tightest interval any method can certify from the same information.

**Pairwise certificate:** $`g_i(v) = d(\hat c_{ri}, v) - d(\hat c_{si}, v)`$ gives

```math
\mathrm{UP}(r,s) = \tfrac1n\textstyle\sum_i \max_{v\in\mathcal V(\ell_i)} \big[d(\hat c_{ri}, v) - d(\hat c_{si}, v)\big] \ \ge\ \mathrm{err}_r - \mathrm{err}_s,
```

with equality for some truth allowed by (K). $`\mathrm{UP}(r,s) < 0`$ **certifies** that $`r`$ is better than $`s`$ with no concept labels. Per fact, this is $`\Delta_F(r,s) + D_U(r,s)`$: the difference of errors on forced slots (known exactly) plus the disagreement on unforced slots (worst case).

**Floor with nothing logged:** replacing $`\mathcal V(\ell_i)`$ by $`\mathcal V`$ gives $`\mathrm{LO}^{\mathrm{free}}_r \le \mathrm{LO}_r`$, valid when no decisions are logged at all.

### 4.2 What rules add over tables: the coupling gain

**Theorem 2 (coupling gain):** For a functional that sums over facts, $`g_i(v) = \sum_j g_{ij}(v_j)`$, the joint bound is $`\sum_i \max_{v\in\mathcal V(\ell_i)} \sum_j g_{ij}(v_j)`$ and the per-fact bound is $`\sum_i \sum_j \max_{v\in\mathcal V(\ell_i)} g_{ij}(v_j)`$. The gain

```math
\Gamma = \textstyle\sum_i \Big[\sum_j \max_{\mathcal V(\ell_i)} g_{ij} - \max_{\mathcal V(\ell_i)} \sum_j g_{ij}\Big] \ \ge\ 0
```

is zero if and only if, on every input, the per-fact maximisers can be chosen jointly inside the fibre. For box fibres (tables) $`\Gamma = 0`$ always; rules that tie facts together are strictly more informative than any table with the same forced values whenever the per-fact worst cases cannot be chosen jointly.

**Proof:** a sum of maxima is at least the maximum of the sum, with equality if and only if a common maximiser exists. In a box the coordinates are free, so one always exists. ∎

### 4.3 Minimax model selection and the certified top set

**Theorem 3 (minimax selection and its lower bound):** Let $`\mathrm{CR}(r) = \max_s [\mathrm{UP}(r,s)]_+`$. Then $`\mathrm{CR}(r)`$ is the exact worst-case regret of deploying $`r`$ over all truths satisfying (K). For any label-free selector $`\sigma`$ (any function of what is observed),

```math
\sup_{\text{truths}\,\models\,K} \mathrm{regret}(\sigma) = \mathrm{CR}(\sigma) \ \ge\ \min_r \mathrm{CR}(r) = \mathrm{CR}(r^\dagger).
```

So $`r^\dagger = \arg\min_r \mathrm{CR}(r)`$ is minimax optimal, and no method that uses the same information can certify a smaller regret.

**Proof:** $`\mathrm{regret}(r) = \max_s(\mathrm{err}_r - \mathrm{err}_s)`$. Exchanging the supremum over truths with the maximum over $`s`$ gives $`\max_s \sup_{\text{truth}}(\mathrm{err}_r - \mathrm{err}_s) = \max_s \mathrm{UP}(r,s)`$ by Theorem 1, and the regret is at least 0 because $`s = r`$ is allowed. A selector is a function of the observations, so it picks the same model for every consistent truth, and its worst case is the worst case of that model. ∎

**Corollary (certified top set):** $`T = \{r : \text{no } s \text{ has } \mathrm{UP}(s,r) < 0\}`$ always contains the best model. Its size measures how much the rules decide on their own.

**Proposition (label consistency is the wrong selector):** Selecting a model for fact $`j`$ by $`P(\hat c_{ri} \in \mathcal V(\ell_i))`$, the usual neuro-symbolic "consistency" score, depends on the model's errors on the other facts. Take model A perfect on $`j`$ and wrong on fact $`k`$ wherever $`k`$ is forced, and B the reverse. Consistency prefers B for fact $`j`$ whenever $`P(F_k) > P(F_j)`$, while the certificate proves A better on $`j`$; B's regret on $`j`$ is then $`P(F_j)`$, although the minimax regret is $`\mathrm{CR}(A) = 0`$. ∎

### 4.4 Shortcut indistinguishability: where label-free auditing must stop

**Theorem 4 (shortcut indistinguishability):** If on input $`i`$ both $`\hat c_{ri}`$ and $`\hat c_{si}`$ lie in $`\mathcal V(\ell_i)`$, then

```math
\max_{v\in\mathcal V(\ell_i)} \big[d(\hat c_{ri}, v) - d(\hat c_{si}, v)\big] = d(\hat c_{ri}, \hat c_{si}), \qquad \min_{v\in\mathcal V(\ell_i)} [\cdots] = -d(\hat c_{ri}, \hat c_{si}).
```

**Proof:** the triangle inequality gives $`d(\hat c_{ri}, v) - d(\hat c_{si}, v) \le d(\hat c_{ri}, \hat c_{si})`$, with equality at $`v = \hat c_{si}`$, which is in the fibre. The minimum is symmetric, attained at $`v = \hat c_{ri}`$. ∎

**Consequences:**
- A model whose outputs always obey the rules (a reasoning shortcut is exactly such a model on the training support) can **never** be certified worse than any other model by any label-free method: its inputs contribute $`\ge 0`$ to every $`\mathrm{UP}(\cdot, \text{it})`$.
- Among rule-consistent models, label consistency ties at 1, and the label-free minimax regret equals the Hamming radius $`\min_r \max_s \bar d(r,s)`$.
- The only things that separate a shortcut from the correct model are (a) inputs where the rules force the facts and the shortcut's outputs break them, which is why the audit data should cover decisions outside the training support, or (b) a label on an input where they disagree (stratum C of Section 4.11).

### 4.5 Projection: refuting and repairing any estimator

**Theorem 5 (projection never hurts):** Let $`\hat e \in \mathbb R^R`$ be any estimate of the models' errors (ATC, DoC, average confidence, Dawid–Skene, a few labels). Let $`\mathcal S`$ be any closed convex set that contains the true error vector $`e^\ast`$ (the box $`\prod_r[\mathrm{LO}_r, \mathrm{HI}_r]`$, or the $`k`$-robust version of Section 4.9). Then

```math
\|\Pi_{\mathcal S}(\hat e) - e^\ast\|_2 \le \|\hat e - e^\ast\|_2,
```

strictly when $`\hat e \notin \mathcal S`$. For the box, the projection is coordinate-wise clipping and the inequality holds for every coordinate: $`|\mathrm{clip}(\hat e_r, \mathrm{LO}_r, \mathrm{HI}_r) - \mathrm{err}_r| \le |\hat e_r - \mathrm{err}_r|`$. Any $`\hat e_r`$ outside $`[\mathrm{LO}_r, \mathrm{HI}_r]`$ is **provably impossible**: no truth allowed by the rules produces it, so the estimator's assumption is refuted for that model.

**Proof:** Euclidean projection onto a closed convex set is firmly non-expansive and $`e^\ast \in \mathcal S`$ is a fixed point; strictness follows from the obtuse-angle property. For an interval, clipping moves $`\hat e_r`$ towards every point of the interval, including $`\mathrm{err}_r`$. ∎

### 4.6 Select by any estimator, certify by the rules

**Proposition 6:** For any estimate $`\hat e \in \mathbb R^R`$, let $`\tilde e = \mathrm{clip}(\hat e, \mathrm{LO}, \mathrm{HI})`$ and $`\hat r = \arg\min \tilde e`$, optionally restricted to the certified top set $`T`$. Then
(a) $`\mathrm{regret}(\hat r) \le \mathrm{CR}(\hat r)`$ for every truth satisfying (K);
(b) $`|\tilde e_r - \mathrm{err}_r| \le |\hat e_r - \mathrm{err}_r|`$ for every $`r`$;
(c) any $`\hat e_r < \mathrm{LO}_r`$ (or $`> \mathrm{HI}_r`$) is provably impossible.

**Proof:** (a) is Theorem 3 applied to the fixed index $`\hat r`$: the certificate holds for every model, whatever chose it. (b) and (c) are Theorem 5. ∎

**Why it matters in practice:** practitioners keep their favourite estimator to pick a model, and the rule base certifies the pick. When the estimator's assumption happens to hold, the typical regret is the estimator's; when it fails, the certificate still bounds the damage. In our traffic-sign study, estimator-plus-veto picks had regret $`10^{-4}`$ with certified bounds below $`10^{-3}`$, while selecting by the certificate alone had regret 0.080: the certificate is best used to vouch for an estimator's pick, not to replace it.

### 4.7 Certified drift alarms

On a deployment domain where nothing is logged, only the legal set $`\mathcal V`$ is available; on the source, decisions are logged.

**Theorem 7 (certified drift):** For an input with prediction $`a`$, let $`\mathrm{LO}^{\mathrm{free}} = \min_{v\in\mathcal V} d(a,v)`$, $`\mathrm{LO} = \min_{v\in\mathcal V(\ell)} d(a,v)`$ and $`\mathrm{HI} = \max_{v\in\mathcal V(\ell)} d(a,v)`$. Under (K), $`\mathrm{LO}^{\mathrm{free}} \le \mathrm{LO} \le \mathrm{err} \le \mathrm{HI}`$ holds input by input, hence on every average. Therefore:
1. **Certified alarm:** if $`\mathrm{LO}^{\mathrm{free}}_r(\mathcal T) > \mathrm{HI}_r(\mathcal S)`$ on the unlabelled target $`\mathcal T`$ and the source $`\mathcal S`$, then $`\mathrm{err}_r(\mathcal T) > \mathrm{err}_r(\mathcal S)`$. A false alarm is impossible, whatever the shift.
2. **Population form:** per-input bounds lie in $`[0,1]`$, so with Hoeffding margins $`\varepsilon_n = \sqrt{\log(2/\delta)/2n}`$ on both sides, $`\mathrm{LO}^{\mathrm{free}}_r(\mathcal T) - \varepsilon_{n_T} > \mathrm{HI}_r(\mathcal S) + \varepsilon_{n_S}`$ certifies that the population error rose, with probability at least $`1 - 2\delta`$.
3. **Certified rank reversal:** if the source certifies $`\mathrm{err}_r < \mathrm{err}_s`$ ($`\mathrm{UP}_{\mathcal S}(r,s) < 0`$) and the target certifies $`\mathrm{err}_s < \mathrm{err}_r`$ ($`\mathrm{UP}^{\mathrm{free}}_{\mathcal T}(s,r) < 0`$), the shift provably reversed their order, and model selection on the source is provably wrong for the target.
4. **Repaired estimators under shift:** Theorem 5 applies on the target with the bracket $`[\mathrm{LO}^{\mathrm{free}}, \mathrm{HI}^{\mathrm{free}}]`$.

When the target decisions are logged, $`\mathcal V(\ell_i)`$ replaces $`\mathcal V`$ and the alarm becomes much more sensitive (in our study: 39% of real degradations caught with logged decisions, against 1 of 66 with nothing logged, and zero false alarms in both cases by proof).

**Proof:** under (K) the truth lies in $`\mathcal V(\ell) \subseteq \mathcal V`$, so its distance to the prediction lies between the minimum and the maximum over each set; averaging preserves the order, giving 1 and 4. For 2, apply Hoeffding to the two means and take a union bound. 3 is Theorem 1 on each domain. ∎

**The difference from a conventional alarm:** an estimator-based alarm (ATC, DoC, disagreement) can be lucky on a given benchmark; ours cannot be unlucky. On our corruptions ATC also raised no false alarm (89 alarms, 0 false, recall 56%), which is an observation on one benchmark, while ours is a theorem.

### 4.8 Reasoning shortcuts of additive rules: exact count and minimal repair

Two facts with values in $`\{0,\dots,9\}`$ are read by the same model and the decision is their sum. Training covers a support $`S`$ of value pairs. A deterministic reasoning shortcut is a map $`\alpha \ne \mathrm{id}`$ with $`\alpha(a) + \alpha(b) = a + b`$ for every $`(a,b) \in S`$. On $`S`$ it produces exactly the true decisions, so by Theorem 4 no label-free method can tell it apart there.

**Theorem 8 (exact count and minimal repair):** Let $`G`$ be the graph on the values with an edge for each pair of $`S`$, with connected components $`G_1,\dots,G_m`$.
1. Along every edge, $`\alpha(b) = a + b - \alpha(a)`$, so the value of $`\alpha`$ at one root determines the whole component. Each component admits $`N_j \le 10`$ maps, found by trying the 10 root values and propagating; the number of maps is $`\prod_j N_j`$, computed exactly in $`O(10\,|S|)`$ time. An isolated value is unconstrained ($`N_j = 10`$).
2. Adding sum-labelled pairs can only add constraints, so it can only remove maps. The minimal repair is the smallest set $`E`$ of extra pairs with $`\prod_j N_j(S \cup E) = 1`$.
3. For MNIST-EvenOdd, $`S = \{0{+}6, 2{+}8, 4{+}6, 4{+}8, 1{+}5, 3{+}7, 1{+}9, 3{+}9\}`$. The even and the odd values form two components with 6 maps each, so there are 36 maps and 35 shortcuts (confirmed by brute force over all assignments). The minimal repair has size **one**: exactly 12 single extra pairs (for example $`0{+}1`$) join the two components and pin $`\alpha`$ to the identity.

**Proof:** 1. Within a component every value is reached from the root along a path, and each edge determines the next value, so a root value either propagates to a consistent map or hits a contradiction or an out-of-range value; trying the 10 root values enumerates all maps of the component. Components are independent, so counts multiply. 2. A map satisfying the constraints of $`S \cup E`$ satisfies those of $`S`$. 3. For the even component, $`\alpha(0) = x`$ forces $`\alpha(6) = 6 - x`$, $`\alpha(4) = 4 + x`$, $`\alpha(8) = 8 - x`$, $`\alpha(2) = 2 + x`$, all within $`\{0,\dots,9\}`$ exactly for $`x \in \{0,\dots,5\}`$: 6 maps. The odd component similarly gives 6. An extra pair $`a + b`$ with $`a`$ even and $`b`$ odd joins the components; the joint propagation then has a single consistent root value. The 12 count is checked by enumeration. ∎

**For a data engineer:** shortcut removal is a data design problem with an exact, label-free answer. Collect decision labels for one more kind of case and the shortcut becomes impossible. In our study, adding sum labels for the single pair type $`0{+}1`$ (10% more training pairs) cut the digit error of sum-only models from 0.45–0.53 to 0.06–0.13.

### 4.9 Imperfect rules: k-robust bounds

Real rule bases are wrong on some inputs (on CUB, 18% of reliable per-image attribute labels differ from the class table).

**Theorem 9 (k-robust sharp bound):** Suppose $`c_i \in \mathcal V(\ell_i)`$ except on at most $`k`$ unknown inputs, where the truth can be anything in a universe $`\mathcal U \supseteq \mathcal V(\ell_i)`$ (the legal set, or the whole cube). For an additive $`g`$, the sharp upper bound is

```math
\tfrac1n\Big[\textstyle\sum_i \max_{\mathcal V(\ell_i)} g_i + \mathrm{top}_k\{h_i\}\Big], \qquad h_i = \max_{\mathcal U} g_i - \max_{\mathcal V(\ell_i)} g_i \ \ge 0,
```

and symmetrically for the lower bound. For the pairwise certificate, per fact, $`h_i \in \{0, 2/K\}`$ and $`h_i > 0`$ exactly when the fact is forced, the models disagree and $`r`$ agrees with the forced value: the certificate widens by at most $`2k/n`$, and only through forced disagreements.

**Proof:** the adversary chooses a violation set $`B`$ with $`|B| \le k`$. The value is $`\sum_{i\notin B}\max_{\mathcal V(\ell_i)} g_i + \sum_{i\in B}\max_{\mathcal U} g_i = \text{base} + \sum_{i\in B} h_i`$, maximised by the $`k`$ largest $`h_i`$. For the pairwise term, $`d(a,v) - d(b,v) \le d(a,b)`$ over the cube, with equality at $`v = b`$, so $`h_i \le d(a,b) - \max_{\mathcal V(\ell_i)}[\cdot] \le 2\,d(a,b)`$ on forced disagreements and 0 elsewhere. ∎

### 4.10 Certifying k from a few labels

**Theorem 10 (certifying the rules):** Label $`m`$ inputs drawn uniformly without replacement from the $`n`$ audited inputs and count the $`x`$ that violate (K). Let $`k^+`$ be the largest $`K`$ for which the hypergeometric probability $`P(X \le x \mid N = n, K, m) > \delta`$. Then, with probability at least $`1 - \delta`$, at most $`k^+`$ inputs violate the rules, and **every** bound of Theorem 9 with $`k = k^+`$ holds simultaneously, for all models and all pairs. (With replacement, use the one-sided Clopper–Pearson bound $`\varepsilon^+`$ and $`k^+ = \lceil \varepsilon^+ n \rceil`$.)

**Proof:** the hypergeometric statement is an exact finite-population confidence bound on the number of violations. Given it, Theorem 9 applies to all pairs at once, because the adversary's budget is shared and each bound is a supremum over it. ∎

**Fast rate:** when the rules are exact ($`x = 0`$), $`\varepsilon^+ = 1 - \delta^{1/m} \approx \log(1/\delta)/m`$, so the certificate's slack decays at rate $`1/m`$. Any certificate built from labels alone decays at $`\Omega(\sqrt{D/m})`$ (Le Cam's two-point bound), where $`D`$ is the disagreement rate.

**Sharpened version (used in the notebook):** by Theorem 9, violations can move a pairwise certificate only on $`B = \{i : \text{fact forced on } i \text{ and some pair of models disagrees}\}`$. Certifying the rules on $`B`$ alone (draw the $`m`$ labels from $`B`$) is valid and much tighter, because $`|B| \ll n`$. On real CUB this cut the minimax certificate from 0.273 to 0.210 with validity 1.000.

### 4.11 Knowledge-stratified labelling (KSE)

When some fact labels can be bought, split the slots of each fact, without labels, into three strata:
- **A:** all models agree. These slots add the same amount to every model's error, so they never change $`\mathrm{err}_r - \mathrm{err}_s`$ and never need labels.
- **B:** the rules force the fact and the models disagree. The rules supply the truth (up to the $`k^+`$ violations of Theorem 10).
- **C:** the rules are silent and the models disagree. Only here is the truth unknown.

**Theorem 11 (KSE):** Sample $`m_C`$ inputs uniformly without replacement from C and label them. The estimate

```math
\hat\Delta_{rs} = \tfrac1n\Big[\textstyle\sum_{i\in B} t_{rs,i}(f_i) + \frac{|C|}{m_C}\sum_{\text{labelled } i\in C} t_{rs,i}(c_i)\Big], \qquad t_{rs,i}(v) = d(\hat c_{ri},v) - d(\hat c_{si},v),
```

is unbiased for $`\mathrm{err}_r - \mathrm{err}_s`$ up to the rule bias (at most $`2k^+/n`$). With probability at least $`1-\delta`$ it is within $`\frac{|C|}{n}\sqrt{2\log(2R^2/\delta)/m_C}\,\sqrt{1 - (m_C-1)/|C|}`$ of the truth for all pairs at once. The labels needed for precision $`\eta`$ scale as $`(|C|/n)^2/\eta^2`$ instead of $`D/\eta^2`$, a saving of at least $`n/|C|`$; with $`m_C = |C|`$ the differences are exact.

**Proof:** stratum A cancels identically in every difference; stratum B is deterministic given the rules; in stratum C each term lies in $`[-1,1]`$ and sampling without replacement from a finite population gives Serfling's inequality, with a union bound over the $`R^2`$ pairs. ∎

Active testing (Kossen et al., 2021) labels all of $`B \cup C`$; the rules remove $`B`$, which is often most of the disagreement. In our study KSE halved the labels on CUB (50 KSE labels beat 100 random labels) and cut selection regret 7.4 to 9.4 times on traffic signs.

### 4.12 Instance-level versus population-level identification

The bracket of Theorem 1 is an *instance-level* guarantee. Weak rules can leave it wide even when the model's error is identified at the *population* level under an extra assumption. The notebook studies this boundary.

**Proposition 12 (sums barely force digits):** for the sum of two digits, a digit is forced if and only if the sum is 0 or 18, a fraction of about 2/100 under uniform digits. So the bracket of any imperfect digit reader has width at least $`1 - O(1/50)`$, although under class-conditional noise (CCN) the digit confusion is identified from the sums.

**Theorem 13 (identification index):** under CCN, the observable law of the decisions and predictions is $`P_\theta = G^\top M_\ell G`$ with $`G = \mathrm{diag}(\pi)\,C`$ ($`C`$ the confusion matrix, $`\pi`$ the value frequencies) and $`M_\ell`$ the fibre mask of decision $`\ell`$. Let $`J`$ be its Jacobian in the tangent space of $`\theta = (C, \pi)`$ and $`g`$ the gradient of $`\mathrm{err} = 1 - \sum_a \pi_a C_{aa}`$. If $`g`$ has a component in $`\mathrm{null}(J)`$ at a regular point, the error is not locally identified; if $`g \perp \mathrm{null}(J)`$ throughout a neighbourhood, it is (Rothenberg, 1971). The index $`\rho = \|P_{\mathrm{null}(J)}\, g\| / \|g\|`$ is computable from the rule and the model's outputs alone: $`\rho = 0`$ means the error is point-identified, and $`\rho > 0`$ gives an exact direction along which the observable law is flat while the error moves. The criterion is not obvious: the carry rule leaves 28 directions of $`C`$ unidentified, yet $`\rho = 0`$, so its error is identified; parity leaves 72, with $`\rho \approx 0.94`$. In our study $`\rho`$ separated identifying rules (worst estimation error 0.0031) from the others (best 0.020), with Spearman 0.93 against the error.

**Theorem 14 (spread is a certified lower bound):** for any estimates $`\hat e_p`$ of the same quantity $`e`$, $`\max_p |\hat e_p - e| \ge (\max_p \hat e_p - \min_p \hat e_p)/2`$. So the spread between rules that each claim to identify the error is a label-free proof that some estimate is off by at least half of it.

**Proof:** if every estimate were within $`s/2`$ of $`e`$, the two extreme estimates would be within $`s`$ of each other with strict inequality for at least one, contradiction; formally $`\hat e_{\max} - \hat e_{\min} \le |\hat e_{\max} - e| + |e - \hat e_{\min}| \le 2\max_p|\hat e_p - e|`$. ∎

**Theorem 15 (over-identification test):** if two or more rules identify the confusion under CCN, their estimates converge to the same value; a spread larger than sampling error refutes CCN without labels (the knowledge analogue of the Sargan–Hansen J test).

### 4.13 The assumption ladder

$`(K) \subset (K) + \text{difference-locality} \subset (K) + \text{locality} \subset (K) + \text{conditional independence (Dawid–Skene)}`$.

Each rung shrinks the identified set (an interval for differences, then a point for levels, then a point for the whole confusion) and each rung is testable against the rung below it by Theorem 5. On real CLIP models, locality was refuted in 64–73% of cells, difference-locality less often, and (K) never. The paper's guarantees use the bottom rung only.

### 4.14 What is not claimed

- Exact shortcut counting and minimal repair are proved for additive rules only; general logical programs are open.
- Calibration error and AUROC are not additive; their identified sets need a fractional or pairwise program and are not computed here.
- Choosing which rule to add next with a label-free proxy is open (refining fibres can only shrink every set).
- The computations enumerate the fibre of each input; rule bases with more than millions of legal combinations need a compiled representation.

---

## 5. Theorem numbering: paper, notebook, library

| result | main paper | supplement | this README | notebook | `kcert` function |
|---|---|---|---|---|---|
| sharp bounds, pairwise certificate | Theorem 1 | Theorem S1, Lemma S2 | Theorem 1 | T1, Theorem C (v8) | `brackets`, `pairwise_certificate` |
| knowledge budget (forced fraction) | Proposition 1 | Proposition S4 | — | — | — |
| coupling gain | Proposition 2 | Proposition S3 | Theorem 2 | T3 | — |
| minimax selection, top set | Theorem 2, Corollary 1 | Theorem S2, Corollary S1 | Theorem 3 | T2 | `certified_regret`, `minimax_select`, `top_set` |
| rule consistency is not a valid selector | Proposition 3 | Proposition S5 | — | Prop. 2b | — |
| shortcut indistinguishability | Theorem 3 | Theorem S3 | Theorem 4 | T9 | `shortcut_indistinguishable` |
| select by any estimator, projection | Proposition 4 | Proposition S6 | Theorem 5, Proposition 6 | T7, Prop. 12 | `project`, `impossible`, `select`, `audit` |
| k-robust bounds | Theorem 4 | Theorem S4 | Theorem 9 | T4 | `brackets(k=...)`, `pairwise_certificate(k=...)` |
| certifying k | Theorem 5 | Theorem S5 | Theorem 10 | T5, T5′ | `certify_k` |
| knowledge-stratified labelling | Theorem 6 | Theorem S6 | Theorem 11 | T6 | `kse_sample`, `kse_estimate` |
| certified drift | Theorem 7 | Theorem S7 | Theorem 7 | T13 | `drift_alarm` |
| shortcut count and repair | Theorem 8 | Theorems S8, S9 | Theorem 8 | T14 | `count_shortcuts`, `min_repair` |
| identification index, spread, over-identification | Section 4.9 | Theorems S10, S11 | Theorems 13–15 | T10, T11, T8 | notebook only |

---

## 6. Experiments

All experiments use a zoo of 10 CLIP-family image encoders with frozen features. On top of each encoder we train several heads (zero-shot, a probe trained on a few concept-labelled images, and neuro-symbolic heads trained only from decisions through the exact rule likelihood), giving 30–90 models per experiment.

| notebook section | experiment | data | rules | headline result |
|---|---|---|---|---|
| 14 | E-A: natural, noisy knowledge | CUB-200-2011, all 11,788 images, 86 attributes | class-attribute table | selection regret 0.0074, below every label-free baseline and below 100 concept labels (0.0109); KSE halves the labels |
| 15–16 | E-B: exact rules | MNIST pairs | 8 programs (sum, product, max, difference, carry, parity, box control, three-digit sum) | every guarantee held for all 30 models and programs; $`\rho`$ separates identifying rules (Spearman 0.93) |
| 17 | E-D: traffic law | GTSRB official test set | 12 visual facts, 8 legal actions, 13 legal designs; forced fraction 94.7% | certificate valid for all 40 models, 44.9% of pairs certified with zero wrong; KSE regret 7.4–9.4 times lower |
| 18 | E-E: certified drift | GTSRB under 7 real corruptions | traffic law | 320 model–domain pairs, all bounds held, 0 false alarms by proof, 26 of 66 degradations caught with logged decisions |
| 18e | selection under shift | same | same | estimator + veto: regret $`10^{-4}`$ with certificate below $`10^{-3}`$ |
| 19 | E-F: reasoning shortcuts | MNIST-EvenOdd (rsbench protocol) | sum | 32 shortcut models; floor flagged all 32, no false flag (AUROC 0.91); one extra pair type cut digit error from ~0.5 to 0.06–0.13 |
| 21 | head-to-head with COCOCO | MNIST-EvenOdd, 90 models | sum | COCOCO (1,200 concept labels) coverage fell below target for 41 models under shift (min 0.175); our bracket stayed valid for all 90; AUROC 0.909 vs 0.829; Spearman 0.987 with rsbench concept F1 |
| 20 | confirmatory run | fresh seed and run name | — | every guarantee passed; 11 of 14 scorable pre-registered predictions confirmed, the same 11 as on the first seed |
| 22–23 | paper numbers and figures | — | — | writes `paper_numbers.txt`, all figures and `runtime.json` |
| 16c–16e, 24–25 | E-C: real dashcam frames | BDD-OIA (rsbench release), 21 reason concepts, 4 actions | forward/stop, left, right blocks | section 24 fetches the data; 25 adds k-robust bounds with k certified from 100 labelled frames, and estimator selection certified by the rules |
| 26 | final bundle | — | — | `paper_final.zip`: every figure (incl. `fig_bdd`), `paper_numbers_final.txt` |
| 27–28 (revision) | label-budget spread; equal-budget shortcut controls | saved results of 14c, 15c, 16d, 17c; MNIST-EvenOdd | sum | `notebooks/revision/`: paired 95% intervals and win rates of KSE vs random labels (Table S2, Fig. 6); controls +support 0.485, +0+5 0.595, +0+2 0.192, repair 0+1 0.087 vs 0.499 base, map purity 0.93–0.97 (Table S3). Printed outputs in `results/revision1_outputs.txt`. Paste each file as a new cell at the end of the notebook (it uses the notebook's helpers). |

**Pre-registration:** every guarantee (G) and every expected finding (X, P) was written in the notebook before the run that scored it. The three predictions that failed on both seeds are reported in the paper (no shortcut on traffic signs, because the law forces 94.7% of facts; selection on traffic signs was already easy; ATC happened to raise no false alarms). Five predictions concern BDD-OIA (pre-registration 9); section 25 scores them.

**Concept labels** are used only to score the methods, never by the certificates.

---

## 7. Reproducing the results

The notebook was run on [molab](https://molab.marimo.io) (hosted marimo) with one RTX 6000-class GPU.

```bash
pip install marimo kcert
marimo edit notebooks/kcert_experiments.py        # run all cells
# or headless:
python notebooks/kcert_experiments.py
# quick plumbing check on synthetic stand-ins (minutes, CPU):
SMOKE=1 python notebooks/kcert_experiments.py
```

| variable | meaning |
|---|---|
| `FAG_RUN` | run name (checkpoints go to `<work dir>/<run>`) |
| `FAG_SEED` | random seed (the confirmatory run used seed 1) |
| `FAG_WORK` | work directory |
| `SMOKE=1` | synthetic stand-ins with the same formats, to test the code only |
| `HF_TOKEN_NESY`, `HF_BUCKET_NESY`, `HF_SYNC=1` | optional: mirror checkpoints to your own Hugging Face bucket so a new session resumes |
| `BDDOIA_ZIP` | path to a local BDD-OIA zip (optional; section 24 downloads it from the rsbench Zenodo record, DOI 10.5281/zenodo.11612556) |

Every stage writes atomic checkpoints and resumes after a restart. Datasets are downloaded from their public sources (CUB from Caltech, MNIST, GTSRB and its corruptions from the Hugging Face Hub). Budgets: under 30 GB RAM and 70 GB VRAM.

**Checkpoints of the paper run:** the trained heads, cached encoder features and every intermediate checkpoint of the run reported in the paper are stored on Hugging Face under [`APassionateResearchers`](https://huggingface.co/APassionateResearchers) (`APassionateResearchers/NeSyCVPR`), and the BDD-OIA frames in the dataset `APassionateResearchers/bdd-oia`. They will be made public together with the paper, so every number can be reproduced without retraining.

---

## 8. Using the library on your own system

```python
import numpy as np, kcert

# 1. Your rule base: legal fact vectors and the action each one triggers
kn = kcert.Knowledge.from_rules(worlds, decide, concept_names=FACTS)

# 2. Your models' fact predictions (R, n, K) and the logged actions (n,)
report = kcert.audit(A, kn, actions, names=model_names, estimate=atc_estimates)
print(report.summary())

# 3. Monitoring: does today's batch prove a degradation?
alarm = kcert.drift_alarm(A_source, kn.consistent(actions_source), A_today, kn)
```

See the [`kcert` repository](https://github.com/researchworksofsomeone/kcert) for the full API and runnable examples.

---

## 9. Citation and licence

Please cite the paper (see `CITATION.cff`). Code under the MIT licence. The datasets keep their own licences.
