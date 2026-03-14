## Monte Carlo Simulation Engine

A high-performance C++ engine for pricing options and estimating Greeks via Monte Carlo simulation. Implements antithetic variates, control variates with analytically optimal β
and Likelihood Ratio Method for sensitivity estimation. 
All validated against Black-Scholes closed-form solutions.

---

## Results

Convergence vs Black-Scholes(Call options:S=100,K=100,r=0.05,σ=0.2,T=1.0)

| Paths     | MC Price  | BS Price  | Rel. Error  | 95% CI Width  |
|-----------|-----------|-----------|-------------|---------------|
|10,000     | 10.5167   | 10.4506   | 0.751%      | ±0.2901       |
|100,000    | 10.4427   | 10.4506   | 0.074%      | ±0.091        |
|1,000,000  | 10.4530   | 10.4506   | 0.023%      | ±0.0289       |
|10,000,000 | 10.4493   | 10.4506   | 0.012%      | ±0.009        |

>At 10M paths the MC price converges to within 0.012% of the analytical Black-Scholes price.
>Error shrinks at the theoretical 1/√N rate as paths increase.

## Variance Reduction (Control Variates)

| Method           | Call price | Variance  | SE          | CI Width  |
|------------------|------------|-----------|-------------|-----------|
| Standard MC      | 10.4493    | 216.67    | 0.004655    |  ±0.009   |
| Control Variates | 10.4493    | 31.49     | 0.001775    |  ±0.003   |
| **Reduction**    |            | **85.5%** |             |           |

Optimal β coefficient computed analyticaly: 
β = Cov(discounted payoff, discounted spot) / Var(discounted spot)= **0.6736**

## Greeks vs Black-Scholes (10M paths)
 
| Greek       | MC Value  | BS Analytical | Error   |
|-------------|-----------|---------------|---------|
| Call Delta  | 0.6369    | ~0.6368       | < 0.02% |
| Put Delta   | −0.3631   | ~−0.3632      | < 0.03% |
| Gamma       | 0.018766  | 0.018762      | 0.02%   |
| Call Vega   | 37.53     | ~37.52        | < 0.05% |
 
All sensitivities estimated in a single simulation pass via the Likelihood Ratio Method.

### Performance (10M paths, at-the-money call)
 
| Configuration            | Time   | Speedup | Efficiency |
|--------------------------|--------|---------|------------|
| Sequential (1 thread)    | 1.93   | 1.0×    | 100%       |
| OpenMP (8 threads)       | 0.45   | 4.28×   | 53.6%      |
 
> Near-linear speedup is expected up to ~4 threads; efficiency falls at 8 due to memory bandwidth saturation on most consumer CPUs.
 
---

## Key Features
 
- **Antithetic Variates** — paths simulated in (Z, −Z) pairs; negative correlation between paired payoffs cancels noise without additional simulations
- **Control Variates with optimal β** — uses discounted spot price E[e^(−rT)·S_T] = S₀ as control; β minimises residual variance analytically, achieving **85.5% variance reduction** at 10M paths
- **Likelihood Ratio Method for Greeks** — Delta, Gamma, and Vega estimated in a single pass by differentiating the sampling density, not the payoff. Finite-difference alternatives require re-running the simulation per Greek; LRM does not
- **OpenMP parallelism** — thread-safe per-thread RNG with deterministic seeding (`seed_seq{1234, thread_id, rd()}`); all accumulators reduced via `#pragma omp reduction`
- **Input validation** — parameter bounds enforced before simulation begins
- **95% Confidence Interval reporting** — automated on every run
 
---

## Techical Notes 

### Why Likelihood Ratio Method for Greeks?

Finite-difference estimation of Delta requires two simulation runs (at S and S+ε). For N Greeks that is N+1 runs, each at full cost. LRM differentiates the log-density of the sampling distribution with respect to the parameter of interest and multiplies by the payoff. This produces an unbiased estimator of each sensitivity within the same simulation loop — runtime cost is zero.

The Gamma weight under LRM is :

'''
w_Γ = (Z² − 1) / (S²σ²T) − Z / (S²σ√T)
'''

This is the second-order score function of the log-normal density with respect to S. At 10M paths it converges to within 0.02% of Black-Scholes Gamma (0.018766 vs 0.018762).

### Why Control Variates reduce variance by 85.5%
 
The control variate is Y = e^(−rT)·S_T. Under the risk-neutral measure, E[Y] = S₀ exactly. The adjusted estimator is:
 
```
X̂ = X + β(S₀ − Ȳ)
```
 
where β = Cov(X, Y) / Var(Y) is the analytically optimal coefficient. Because discounted stock price and option payoff are highly correlated (both driven by the same Brownian path), β captures most of the noise, reducing variance from 216.67 to 31.49.
 
### Parallel seeding strategy
 
Each thread receives a `seed_seq{1234, thread_id, rd()}`. The fixed base seed (1234) makes results reproducible given the same thread count. The thread id differentiates streams to prevent correlated draws across threads. Results from parallel and sequential runs at 10M paths agree to within 0.005% (10.4493 vs 10.4488), confirming thread safety.
 
---
 
## Build Instructions
 
### Requirements
- C++17 compiler (GCC 11+ or MSVC 2022)
- CMake 3.20+
- OpenMP
- Boost (random)
 
### Build
```bash
git clone https://github.com/onesilverspoon/MonteCarloSimulation
cd MonteCarloSimulation
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```
 
### Run
```bash
./montecarlo
```
 
### Run tests
```bash
ctest --output-on-failure
```
 
---

## Project Status
 
- [x] Core simulation engine (GBM paths, options call/put pricing)
- [x] Antithetic Variates
- [x] Control Variates with optimal β
- [x] Likelihood Ratio Method (Delta, Gamma, Vega)
- [x] 95% Confidence Interval reporting
- [x] Variance Reduction Ratio reporting
- [x] OpenMP multi-threading with thread-safe RNG
- [x] Black-Scholes analytical benchmark (call, put, gamma)
- [x] Input validation
- [x] CMake build system
- [x] Google Test unit tests
- [ ] CUDA GPU back-end (in progress)
 
---
 
## Dataset / Parameters
 
Default parameters used throughout: S=100, K=100, r=0.05, σ=0.2, T=1.0 (at-the-money option). These are standard benchmark parameters in options pricing literature, chosen to allow direct comparison against the Black-Scholes closed-form solution.
 
---
## License
MIT — see [LICENSE](LICENSE)
