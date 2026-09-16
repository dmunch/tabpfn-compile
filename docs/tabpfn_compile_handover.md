# TabPFN Compile --- Hackathon Handover

## 1. Goal

Build a compiler that turns a labelled numerical dataset into the
**smallest synthetic in-context dataset that preserves the predictive
behaviour of full-context TabPFN**.

``` text
large labelled dataset D
        │
        ├──► frozen TabPFN ──► teacher predictor T_D
        │
        ▼
  TabPFN Compile
        │
        ▼
small synthetic context C*
        │
        └──► same frozen TabPFN ──► T_C* ≈ T_D
```

The output is ordinary synthetic `(X, y)` rows usable directly as TabPFN
context. TabPFN weights are never changed.

**Hackathon claim:** context size is not a user-chosen constant such as
32. The compiler discovers how much context the task actually needs.

------------------------------------------------------------------------

## 2. Scope

### MVP

-   Classification first.
-   Numerical features only.
-   Frozen TabPFN-3.5.
-   Synthetic context features **and labels** are optimizable.
-   Teacher/student distillation objective.
-   Fixed-budget solver for arbitrary `m`.
-   Automatic search for the minimum sufficient `m`.
-   Baselines: random and stratified real-row subsets.
-   Report fidelity, downstream predictive performance, compression
    ratio, latency and memory.

### Explicit non-goals

No categorical synthesis, TabICLv2 transfer, new runtime, web UI, formal
convergence proof, MILP, optimal transport, or exhaustive solver zoo
unless the MVP is already complete.

------------------------------------------------------------------------

## 3. Mathematical object

Let the original labelled dataset be

\[ D={(x_i,y_i)}\_{i=1}\^{N},
`\qquad `{=tex}x_i`\in`{=tex}`\mathbb `{=tex}R\^d. \]

Freeze TabPFN parameters (`\theta`{=tex}). A context (C) induces a
predictor

\[ T_C(x) := f\_`\theta`{=tex}(C,x). \]

The full dataset therefore induces the teacher

\[ T_D(x)=f\_`\theta`{=tex}(D,x). \]

A synthetic context with (m) atoms is

\[ C_m={(`\tilde `{=tex}x_j,`\tilde `{=tex}y_j)}\_{j=1}\^{m}. \]

Equivalently, view it as an atomic empirical measure

\[ `\mu`{=tex}*{C_m} = `\frac{1}{m}`{=tex}`\sum`{=tex}*{j=1}\^{m}
`\delta`{=tex}\_{(`\tilde `{=tex}x_j,`\tilde `{=tex}y_j)}. \]

The project is an approximation problem over contexts:

\[ E_m(D) = `\inf`{=tex}\_{\|C\|`\le `{=tex}m} `\mathcal `{=tex}E_D(C).
\]

------------------------------------------------------------------------

## 4. Fidelity functional

Use a held-out **compile set** (Q={q_k}\_{k=1}\^{M}), disjoint from the
final test set.

For classification, cache the full-context teacher distribution

\[ p_D(`\cdot`{=tex}`\mid `{=tex}q_k)=T_D(q_k). \]

The primary fidelity functional is

\[ `\mathcal `{=tex}E_D(C) = `\frac`{=tex}1M `\sum`{=tex}*{k=1}\^{M}
D*{`\mathrm{KL}`{=tex}} `\left`{=tex}(
p_D(`\cdot`{=tex}`\mid `{=tex}q_k) ;`\Vert`{=tex};
p_C(`\cdot`{=tex}`\mid `{=tex}q_k) `\right`{=tex}). \]

In implementation, clamp probabilities by a small (`\delta`{=tex})
before taking logs.

A symmetric alternative worth reporting, but not required for
optimization, is Jensen--Shannon divergence.

**Important:** optimize fidelity to the teacher distribution, not
accuracy. Accuracy/AUC/log-loss against ground truth are independent
evaluation metrics.

------------------------------------------------------------------------

## 5. The central quantity: context complexity

For a chosen fidelity tolerance (`\varepsilon`{=tex}), define

\[ `\boxed{
\mathcal C_\varepsilon(D)
=
\min
\left\{
m :
E_m(D)\le\varepsilon
\right\}.
}`{=tex} \]

This is the **TabPFN context complexity** of dataset (D) at tolerance
(`\varepsilon`{=tex}).

The compiler's primary task is therefore

\[ `\boxed{
\text{find } C^\star
\text{ with }
|C^\star|=\mathcal C_\varepsilon(D).
}`{=tex} \]

Equivalent penalized formulation:

\[ C\_`\lambda`{=tex}\^`\star`{=tex} = `\arg`{=tex}`\min`{=tex}\_C
`\left[
\mathcal E_D(C)+\lambda |C|
\right]`{=tex}. \]

For the hackathon implementation, prefer the constrained
(`\varepsilon`{=tex})-form because its user-facing meaning is clearer.

------------------------------------------------------------------------

## 6. Numerical optimization

TabPFN-3.5 already exposes the required primitive: with differentiable
input enabled, gradients flow through `fit_with_differentiable_input`
into prompt/context tensors while model weights remain frozen.

For fixed (m), parameterize

\[ Z_m= (`\tilde `{=tex}X_m,`\tilde `{=tex}y_m) \]

and solve

\[ Z_m\^`\star`{=tex} `\approx`{=tex} `\arg`{=tex}`\min`{=tex}\_{Z_m}
`\mathcal `{=tex}E_D(C(Z_m)) \]

using Adam as the baseline optimizer.

### Initialization

Implement at least:

1.  stratified real-row initialization;
2.  random real-row initialization.

Run multiple seeds where affordable and retain the best compile-set
solution.

### Constraints / regularization

Synthetic feature values should not escape to absurd numerical scales.
Work in standardized feature coordinates and either clamp to a generous
observed range or add a soft domain penalty

\[ R_X(C) = `\sum`{=tex}\_{j,r} `\left[
\max(0,\tilde x_{jr}-u_r)^2
+
\max(0,l_r-\tilde x_{jr})^2
\right]`{=tex}. \]

Then optimize

\[ J(C)=`\mathcal `{=tex}E_D(C)+`\beta `{=tex}R_X(C). \]

For classification, `prompt_y` is differentiable in TabPFN's current
prompt-tuning path. Do not prematurely force it to integer labels during
optimization. Preserve the learned values in the compiled artifact if
the inference path accepts them; otherwise investigate a final
projection/rounding step and quantify its cost.

------------------------------------------------------------------------

## 7. Finding the context size

Do **not** optimize every integer (m).

### Stage A --- bracket the solution

Choose

\[ m`\in`{=tex}{1,2,4,8,16,32,64,128,`\ldots`{=tex}} \]

until the optimized context satisfies

\[ `\mathcal `{=tex}E_D(C_m)`\le`{=tex}`\varepsilon`{=tex}. \]

This gives a bracket

\[ m\_{`\mathrm{fail}`{=tex}} \<
`\mathcal `{=tex}C\_`\varepsilon`{=tex}(D)
`\le `{=tex}m\_{`\mathrm{pass}`{=tex}}. \]

### Stage B --- search inside the bracket

Use integer binary search, warm-starting where practical, until the
smallest passing (m) is found.

Because the numerical optimizer only approximates (E_m), enforce
robustness:

-   multiple seeds near the boundary;
-   require the tolerance to hold on a separate validation set;
-   optionally use a margin
    (`\varepsilon`{=tex}*{`\mathrm{compile}`{=tex}}\<`\varepsilon`{=tex}*{`\mathrm{accept}`{=tex}}).

The result is an **empirical** estimate

\[ `\widehat{\mathcal C}`{=tex}\_`\varepsilon`{=tex}(D), \]

not a proof of the global optimum.

That distinction must be explicit in the README.

------------------------------------------------------------------------

## 8. Stretch solver: growing atomic context

The mathematically interesting extension is to construct a nested
sequence

\[ C_1`\subset `{=tex}C_2`\subset`{=tex}`\cdots`{=tex} \]

instead of repeatedly solving unrelated fixed-(m) problems.

Starting from (C_0=`\varnothing`{=tex}):

1.  propose a new synthetic atom (z=(x,y));
2.  choose it to maximally reduce the fidelity functional;
3.  add it to the context;
4.  jointly re-optimize all atoms;
5.  stop as soon as the acceptance criterion holds.

In measure notation,

\[ `\mu`{=tex}\_{m+1} = `\mu`{=tex}\_m+w,`\delta`{=tex}\_z. \]

The relevant first-variation intuition is

\[ `\Phi`{=tex}\_{`\mu`{=tex}}(z) = `\left`{=tex}.
`\frac{d}{d\alpha}`{=tex} J(`\mu`{=tex}+`\alpha`{=tex}`\delta`{=tex}*z)
`\right`{=tex}\|*{`\alpha=0`{=tex}}. \]

Choose approximately

\[ z\^`\star`{=tex} = `\arg`{=tex}`\min`{=tex}*z
`\Phi`{=tex}*`\mu`{=tex}(z). \]

This is the bridge to sparse measure optimization / conditional-gradient
thinking. It is a stretch goal, not required for the first working
compiler.

------------------------------------------------------------------------

## 9. Approximation-theory interpretation

The fixed-budget problem defines the approximation curve

\[ m`\longmapsto `{=tex}E_m(D). \]

Measure it empirically.

Questions worth reporting:

\[ E_m(D)`\stackrel{?}{\approx}`{=tex}A
m\^{-`\alpha`{=tex}}+E\_`\infty`{=tex}, \]

whether different datasets exhibit substantially different decay rates,
and whether (`\mathcal `{=tex}C\_`\varepsilon`{=tex}(D)) correlates with
obvious task properties such as noise, intrinsic dimensionality,
redundancy or class-boundary complexity.

Do **not** claim theoretical rates from empirical fits. The curve itself
is already a useful result.

------------------------------------------------------------------------

## 10. Functional / variational interpretation

The dataset can be represented by

\[ `\mu`{=tex}\_D =
`\frac`{=tex}1N`\sum`{=tex}*i`\delta`{=tex}*{(x_i,y_i)}. \]

TabPFN induces a nonlinear map

\[ T:`\mu`{=tex}`\mapsto `{=tex}T\_`\mu`{=tex}. \]

Compilation seeks a sparse atomic measure satisfying

\[ T\_{`\mu`{=tex}*C}`\approx `{=tex}T*{`\mu`{=tex}\_D}. \]

This defines the useful conceptual equivalence relation

\[ `\mu`{=tex}`\sim`{=tex}`\nu`{=tex} `\iff`{=tex}
T\_`\mu`{=tex}=T\_`\nu`{=tex}. \]

The compiler is therefore searching for a low-complexity representative
of the predictive equivalence class of (`\mu`{=tex}\_D).

This is the mathematical motivation; it does **not** need to become an
elaborate theoretical section in the hackathon demo.

------------------------------------------------------------------------

## 11. Data splitting

Avoid leakage between compilation and evaluation.

Recommended split:

``` text
D
├── teacher/context set     used to define full-context T_D
├── compile/query set       differentiable optimization objective
├── validation set          context-size stopping/search
└── test set                final metrics only
```

A simpler three-way split is acceptable if compute/data are limited, but
the final test set must never influence context optimization or the
choice of (m).

Cache teacher probabilities for compile/validation/test queries so
repeated optimization does not rerun full-context teacher inference
unnecessarily.

------------------------------------------------------------------------

## 12. Baselines

For each discovered context size (m\^`\star`{=tex}), compare the
compiled context against:

-   random real-row subset of size (m\^`\star`{=tex});
-   stratified real-row subset of size (m\^`\star`{=tex});
-   full original context.

Optional if trivial: k-means/medoid-style prototype selection.

The key comparison is:

\[ `\text{optimized synthetic }`{=tex}m\^`\star`{=tex}
`\quad`{=tex}`\text{vs}`{=tex}`\quad`{=tex}
`\text{ordinary }`{=tex}m\^`\star`{=tex}`\text{-row context}`{=tex}. \]

If that gap is small, the project loses much of its point.

------------------------------------------------------------------------

## 13. Metrics

Report two separate families.

### Fidelity to full-context TabPFN

\[ `\mathrm{KL}`{=tex}(T_D`\Vert `{=tex}T_C), `\qquad`{=tex}
`\mathrm{JS}`{=tex}(T_D,T_C), \]

plus prediction agreement if useful.

### Predictive utility against ground truth

Classification:

-   ROC-AUC where applicable;
-   log-loss;
-   accuracy;
-   calibration metric if time permits.

Systems metrics:

-   original rows (N);
-   compiled rows (m\^`\star`{=tex});
-   compression ratio (N/m\^`\star`{=tex});
-   inference latency;
-   peak memory / VRAM where measurable.

Never conflate teacher fidelity with ground-truth quality.

------------------------------------------------------------------------

## 14. Acceptance criterion

The user-facing mode should be something like

``` bash
tabpfn-compile data.csv \
  --target subscribed \
  --fidelity 0.99 \
  --output compiled.csv
```

Internally, define precisely what `0.99` means. Prefer exposing a
normalized fidelity score derived from divergence rather than pretending
that "99% accuracy retained" is the optimization target.

For the research/demo CLI, expose the mathematically unambiguous form
too:

``` bash
tabpfn-compile data.csv \
  --target subscribed \
  --max-kl 0.01
```

------------------------------------------------------------------------

## 15. Compiled artifact

The artifact should contain:

``` text
compiled.csv
metadata.json
```

`metadata.json` should include at least:

-   TabPFN checkpoint/version;
-   source dataset hash;
-   feature names/order;
-   standardization parameters;
-   target encoding;
-   learned context size;
-   fidelity tolerance;
-   achieved compile/validation fidelity;
-   optimizer/configuration;
-   random seed;
-   source row count;
-   creation timestamp.

The compiled context is model-specific. Do not imply that it is a
general-purpose compressed version of the source dataset.

------------------------------------------------------------------------

## 16. Killer visualization

The main figure is the approximation frontier:

``` text
teacher divergence
▲
│ ●
│  \
│   ●
│     \
│       ●
│          ●
│              ●────●
│              ─────── ε
│              ↑
│             m*
└────────────────────────► context rows
```

Overlay random/stratified subset baselines.

The demo headline then becomes concrete:

``` text
Bank Marketing
41,188 original rows
        ↓
47 synthetic rows
876× smaller context
≤ ε teacher divergence
X% of full-context AUC retained
Y× inference speedup
```

All numbers must come from held-out evaluation.

------------------------------------------------------------------------

## 17. Three-week execution plan

### Days 1--2 --- feasibility gate

Modify Prior Labs' prompt-tuning classifier example to distill
full-context teacher probabilities.

Run one clean numerical classification dataset at fixed budgets:

\[ m=8,32,128. \]

**Kill criterion:** optimized contexts must materially outperform
equal-sized random/stratified contexts.

### Days 3--7 --- fixed-budget compiler

Build reusable dataset splitting, teacher caching, optimization,
evaluation, artifact export and baselines. Run 3--5 datasets.

### Days 8--12 --- automatic context complexity

Implement exponential bracketing + boundary search for

\[ `\widehat{\mathcal C}`{=tex}\_`\varepsilon`{=tex}(D). \]

Add repeated seeds and validation acceptance.

### Days 13--16 --- benchmark

Run a small, varied benchmark suite. Produce approximation curves and
systems measurements. Investigate failure cases rather than hiding them.

### Days 17--19 --- stretch mathematics

If the core works, implement growing-atom / greedy initialization or
another clearly motivated solver improvement.

### Days 20--21 --- submission

Polish README, reproducible CLI/notebook, figures, concise video and
hackathon narrative.

------------------------------------------------------------------------

## 18. Success criteria

The project is compelling if all of the following hold:

1.  synthetic contexts consistently beat equal-size sampling baselines;
2.  useful datasets admit substantial compression;
3.  automatic context-size discovery is stable enough to reproduce;
4.  smaller contexts produce a measurable inference/memory benefit;
5.  the resulting approximation curves differ meaningfully across tasks.

The strongest possible result is not a particular compression ratio. It
is evidence that

\[ `\boxed{
\mathcal C_\varepsilon(D)
}`{=tex} \]

is a measurable, useful notion of **task complexity as seen by TabPFN**.

------------------------------------------------------------------------

## 19. One-sentence pitch

> **TabPFN Compile treats an in-context dataset as an optimizable
> mathematical object and automatically finds the smallest synthetic
> context that makes frozen TabPFN behave like it had seen the full
> dataset.**

## 20. Immediate first experiment

Start from Prior Labs' `examples/prompt_tuning_classifier.py`.

Replace its ground-truth NLL objective with distillation against cached
full-context TabPFN probabilities, then compare optimized vs stratified
contexts at `m = 8, 32, 128`.

Do nothing else until that experiment answers whether the central
phenomenon is real.
