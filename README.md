## Disclaimer! 
Both the project and the README file are under development and should be updated regularly. 

## Monte Carlo Simulation

Monte Carlo Call & Put Pricer with Variance Reduction, Greeks and OpenMP Parallelism

## Short description

This C++ project implements a Monte Carlo pricer for call and put options. It includes:

* Antithetic variates for path-pairing
* A control variate using the discounted terminal stock price (risk-neutral expectation)
* Likelihood-ratio estimators for Greeks (Delta, Gamma, Vega) and control-variate adjustments for those Greeks
* Single-threaded and OpenMP-parallel implementations and a small benchmarking harness that reports speedup & efficiency
* Black–Scholes closed-form benchmark and error reporting

## Key features / highlights

* **Antithetic variates** to reduce variance by pairing `Z` and `-Z` for each normal draw.
* **Control variate (CV)** using (e^{-rT}S_T) with known risk-neutral expectation (S_0) to reduce estimator variance.
* **Likelihood-ratio Greeks** (Delta, Gamma, Vega) computed inside the same simulation loop.
* **Control-variate adjusted Greeks** to reduce noise in sensitivities.
* **Parallel version** using OpenMP with per-thread deterministic seeding via `std::seed_seq`.
* **Comprehensive diagnostics**: variances, standard errors, 95% confidence intervals, variance reduction percentage, Black–Scholes comparison, and timings.

## Requirements

* A standards-compliant C++ compiler (GCC/Clang) with C++17 support
* OpenMP support (for the parallel build)
* POSIX `math` (M_PI etc.) — the code defines `_USE_MATH_DEFINES` for portability

## Build & Run

Save the code to a file (e.g. `monte_carlo.cpp`). Build with optimization and OpenMP enabled:

```bash
# Recommended compile (GCC / Clang)
g++ -O3 -std=c++17 -fopenmp monte_carlo.cpp -o monte_carlo

# If you need to be explicit about math library (rarely necessary on modern toolchains):
# g++ -O3 -std=c++17 -fopenmp monte_carlo.cpp -o monte_carlo -lm
```

Run the program (defaults are in the `main()` function):

```bash
./monte_carlo
```

### Parameters you can tweak (inside `main()`)

* `num_sims` (default: `10_000_000`) — total Monte Carlo samples (note: the code uses antithetic pairs so the loop iterates `num_sims/2` times).
* `S`, `K`, `r`, `sigma`, `T` — initial stock price, strike, risk-free rate, volatility, maturity (years).
* `omp_set_num_threads(omp_get_num_procs())` picks as many threads as logical processors by default; you can override using `OMP_NUM_THREADS` environment variable.

## What the program prints (explanation)

The program prints (annotated):

* `Call price` / `Puts price`: Monte Carlo estimate (discounted payoff mean)
* `Call Var` / `Puts Var`: estimator variance (discounted)
* `Call SE` / `Puts SE`: Monte Carlo standard error
* `Call CI`: 95% confidence interval (uses z = 1.96)
* `CV Call` / `CV Puts`: price estimates after applying the control variate
* `CV Call Var` / `CV Puts Var`: CV-adjusted variances
* `Var Red`: variance reduction fraction (1 - var_CV / var_MC)
* `Call Parallel` / `Puts Parallel`: prices from the OpenMP-parallel function
* `Speedup`, `Efficiency`: wall-clock speedup and parallel efficiency
* `Black-Scholes call/puts`: closed-form benchmarks and absolute/relative errors
* `Greeks`: Beta (CV coefficient), Delta, Gamma, Vega and their CV-adjusted counterparts

## Design notes & maths (brief)

* **Risk-neutral drift decomposition:** The terminal stock price is simulated via

[ S_T = S_0 \exp\left((r - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}Z\right) ]

with antithetic pair using `Z` and `-Z`.

* **Control variate:** use the discounted terminal stock (Y = e^{-rT} S_T) whose expectation under the risk-neutral measure is (\mathbb{E}[Y] = S_0). For a payoff estimator (X) (discounted payoff), the CV estimator is

[ X_{CV} = X + \beta (\mathbb{E}[Y] - Y) ]

where (\beta = \frac{\mathrm{Cov}(X,Y)}{\mathrm{Var}(Y)}). The code computes (\beta) from the sample covariances and applies it to both calls and puts; CV variance is computed analytically from sample moments.

* **Greeks by likelihood ratio:** Delta, Gamma and Vega are estimated using likelihood-ratio weights derived from differentiation under the integral (implemented inside the loop). The code also computes CV-adjusted versions for reduced variance.

* **Antithetic variates**: pairing reduces variance for payoffs that are convex or smooth in the Gaussian variable.

## Implementation details & gotchas

* The code performs *antithetic pairing* inside a `half_sims` loop; `num_sims` should be even to avoid off-by-one asymmetry.
* The control-variate derivation assumes non-degenerate `Var(Y)`. Defensive checks avoid division by near-zero variance (`1e-12` threshold).
* Deterministic, per-thread seeding for parallel builds uses `std::seed_seq{1234, omp_get_thread_num()}` — this yields reproducible parallel runs across machines with identical thread counts, but if true reproducibility across different thread counts is required, use a single-generator strategy with careful subsequence allocation instead.
* The program uses the z-value `1.96` for a 95% CI; for small-sample or heavy-tailed simulations you may prefer studentized bootstrapping or other robust CI procedures.

## Suggested extensions / improvements

* Add antithetic + control-variate together for Greeks more carefully (the code already mixes both but can be modularized).
* Implement quasi-Monte Carlo (Sobol/halton) paths and check low-discrepancy effects on variance.
* Studentize the confidence intervals using the sample standard deviation of the estimator rather than a z-approximation when `num_sims` is small.
* Provide command-line flags (e.g. via `argparse`-style parsing) to configure `num_sims`, `S`, `K`, etc., without recompiling.
* Save outputs (JSON / CSV) for downstream plotting or batch experiment automation.

## License
??
