# Mathematical Methodology

## Financial Options Pricing Engine

## 1. Purpose

This project implements and compares three methods for financial option valuation:

1. Black-Scholes analytical pricing.
2. Cox-Ross-Rubinstein binomial trees.
3. Risk-neutral Monte Carlo simulation.

The project values European call and put options, American call and put options, and calculates the principal Black-Scholes option Greeks.

The objective is to connect financial mathematics, probability, numerical methods, risk analysis, and software engineering in a reproducible Python project.

---

## 2. Financial Option Fundamentals

An option is a financial derivative whose value depends on an underlying asset.

A **call option** gives its holder the right, but not the obligation, to buy the underlying asset at a predetermined strike price.

A **put option** gives its holder the right, but not the obligation, to sell the underlying asset at a predetermined strike price.

### European options

European options can only be exercised at maturity.

### American options

American options can be exercised at any time up to and including maturity.

---

## 3. Model Parameters

The pricing models use the following parameters:

| Parameter | Symbol | Description |
|---|---:|---|
| Spot price | $S_0$ | Current price of the underlying asset |
| Strike price | $K$ | Contractual exercise price |
| Time to maturity | $T$ | Remaining lifetime of the option in years |
| Risk-free rate | $r$ | Continuously compounded annual interest rate |
| Volatility | $\sigma$ | Annual standard deviation of asset returns |
| Dividend yield | $q$ | Continuously compounded annual dividend yield |

The baseline analysis uses:

| Parameter | Value |
|---|---:|
| Spot price | \$100 |
| Strike price | \$100 |
| Time to maturity | 1 year |
| Risk-free rate | 5% |
| Volatility | 20% |
| Dividend yield | 0% |

---

## 4. Terminal Payoffs

The terminal payoff of a European call option is:

$$
C_T = \max(S_T-K,0)
$$

The terminal payoff of a European put option is:

$$
P_T = \max(K-S_T,0)
$$

An option payoff cannot be negative because the holder has a right, not an obligation, to exercise.

The payoff does not include the premium originally paid for the option. Therefore, payoff and profit are different concepts.

---

## 5. Black-Scholes Model

The Black-Scholes model assumes that the underlying asset follows Geometric Brownian Motion:

$$
dS_t = (\mu-q)S_t\,dt+\sigma S_t\,dW_t
$$

where:

- $\mu$ is the expected asset return.
- $q$ is the continuous dividend yield.
- $\sigma$ is volatility.
- $W_t$ is a standard Wiener process.

Under the risk-neutral probability measure, the expected asset return is replaced by the risk-free rate:

$$
dS_t = (r-q)S_t\,dt+\sigma S_t\,dW_t
$$

The risk-neutral terminal asset price is:

$$
S_T =
S_0
\exp
\left[
\left(
r-q-\frac{\sigma^2}{2}
\right)T
+
\sigma\sqrt{T}Z
\right]
$$

where:

$$
Z \sim N(0,1)
$$

---

## 6. Black-Scholes Terms

The Black-Scholes model uses:

$$
d_1 =
\frac{
\ln(S_0/K)
+
\left(
r-q+\frac{\sigma^2}{2}
\right)T
}{
\sigma\sqrt{T}
}
$$

and:

$$
d_2 = d_1-\sigma\sqrt{T}
$$

The function $N(x)$ represents the cumulative distribution function of the standard normal distribution.

The function $\phi(x)$ represents the probability density function of the standard normal distribution.

---

## 7. European Call Price

The Black-Scholes price of a European call with continuous dividends is:

$$
C =
S_0e^{-qT}N(d_1)
-
Ke^{-rT}N(d_2)
$$

The first term represents the dividend-adjusted expected asset component.

The second term represents the present value of the strike payment weighted by the risk-neutral exercise probability.

For the baseline parameters:

$$
C \approx \$10.4506
$$

---

## 8. European Put Price

The Black-Scholes price of a European put with continuous dividends is:

$$
P =
Ke^{-rT}N(-d_2)
-
S_0e^{-qT}N(-d_1)
$$

For the baseline parameters:

$$
P \approx \$5.5735
$$

---

## 9. Put-Call Parity

European call and put prices with the same strike and maturity satisfy:

$$
C-P =
S_0e^{-qT}
-
Ke^{-rT}
$$

Equivalently:

$$
C + Ke^{-rT}
=
P + S_0e^{-qT}
$$

Put-call parity represents a no-arbitrage relationship.

The project calculates the numerical difference between both sides. A value close to zero indicates that parity is satisfied.

For the baseline scenario:

$$
\text{Parity difference}
\approx 0
$$

---

## 10. Option Greeks

Option Greeks quantify the sensitivity of the option price to changes in market variables.

### 10.1 Delta

Delta measures the approximate change in option price produced by a one-unit change in the underlying asset.

Call Delta:

$$
\Delta_C =
e^{-qT}N(d_1)
$$

Put Delta:

$$
\Delta_P =
e^{-qT}
\left[
N(d_1)-1
\right]
$$

For the baseline parameters:

$$
\Delta_C \approx 0.636831
$$

$$
\Delta_P \approx -0.363169
$$

The call gains value when the underlying price increases, while the put generally loses value.

---

### 10.2 Gamma

Gamma measures the change in Delta produced by a one-unit change in the underlying price:

$$
\Gamma =
\frac{
e^{-qT}\phi(d_1)
}{
S_0\sigma\sqrt{T}
}
$$

Call and put options with identical parameters have the same Gamma.

For the baseline parameters:

$$
\Gamma \approx 0.018762
$$

---

### 10.3 Vega

Vega measures sensitivity to volatility:

$$
\text{Vega} =
S_0e^{-qT}\phi(d_1)\sqrt{T}
$$

The implementation divides this value by 100:

$$
\text{Vega}_{1\%}
=
\frac{
S_0e^{-qT}\phi(d_1)\sqrt{T}
}{
100
}
$$

Therefore, the reported Vega estimates the option price change caused by a one-percentage-point change in annual volatility.

For the baseline parameters:

$$
\text{Vega}_{1\%}
\approx 0.375240
$$

Call and put options have the same Vega under Black-Scholes.

---

### 10.4 Theta

Theta measures sensitivity to the passage of time.

Annual call Theta is:

$$
\Theta_C =
-
\frac{
S_0e^{-qT}\phi(d_1)\sigma
}{
2\sqrt{T}
}
-
rKe^{-rT}N(d_2)
+
qS_0e^{-qT}N(d_1)
$$

Annual put Theta is:

$$
\Theta_P =
-
\frac{
S_0e^{-qT}\phi(d_1)\sigma
}{
2\sqrt{T}
}
+
rKe^{-rT}N(-d_2)
-
qS_0e^{-qT}N(-d_1)
$$

The implementation reports daily Theta:

$$
\Theta_{\text{daily}}
=
\frac{
\Theta_{\text{annual}}
}{
365
}
$$

For the baseline parameters:

$$
\Theta_C \approx -0.017573
$$

$$
\Theta_P \approx -0.004542
$$

Negative Theta represents time-value decay.

---

### 10.5 Rho

Rho measures sensitivity to the risk-free interest rate.

Call Rho is:

$$
\rho_C =
KTe^{-rT}N(d_2)
$$

Put Rho is:

$$
\rho_P =
-KTe^{-rT}N(-d_2)
$$

The implementation divides these values by 100 to report the effect of a one-percentage-point change:

$$
\rho_{1\%}
=
\frac{\rho}{100}
$$

For the baseline parameters:

$$
\rho_C \approx 0.532325
$$

$$
\rho_P \approx -0.418905
$$

---

## 11. Cox-Ross-Rubinstein Binomial Model

The binomial model divides the option lifetime into $N$ discrete time steps:

$$
\Delta t = \frac{T}{N}
$$

During each step, the asset price moves upward by factor:

$$
u=e^{\sigma\sqrt{\Delta t}}
$$

or downward by factor:

$$
d=\frac{1}{u}
$$

The risk-neutral probability of an upward movement is:

$$
p =
\frac{
e^{(r-q)\Delta t}-d
}{
u-d
}
$$

The risk-neutral probability of a downward movement is:

$$
1-p
$$

The no-arbitrage condition requires:

$$
0 \leq p \leq 1
$$

---

## 12. Binomial Backward Induction

At maturity, the option values equal their intrinsic values.

Call:

$$
V_N =
\max(S_N-K,0)
$$

Put:

$$
V_N =
\max(K-S_N,0)
$$

The value at each earlier node is calculated through backward induction:

$$
V_t =
e^{-r\Delta t}
\left[
pV_{t+\Delta t}^{up}
+
(1-p)V_{t+\Delta t}^{down}
\right]
$$

This expression is the discounted risk-neutral expected option value.

---

## 13. American Exercise

For an American option, each node compares the continuation value with immediate exercise.

The American option value is:

$$
V_t^{American}
=
\max
\left(
V_t^{continuation},
V_t^{intrinsic}
\right)
$$

For a call:

$$
V_t^{intrinsic}
=
\max(S_t-K,0)
$$

For a put:

$$
V_t^{intrinsic}
=
\max(K-S_t,0)
$$

An American option cannot be worth less than an otherwise identical European option.

For a non-dividend-paying asset, early exercise of an American call is generally not optimal. Consequently, its value is approximately equal to the European call value.

American puts may benefit from early exercise and can therefore be more valuable than European puts.

---

## 14. Binomial Convergence

As the number of steps increases, the European binomial price converges toward the Black-Scholes analytical price:

$$
\lim_{N\to\infty}
V_N^{Binomial}
=
V^{Black-Scholes}
$$

Using 500 steps, the project produces approximately:

| Option | Binomial value |
|---|---:|
| European call | \$10.4466 |
| European put | \$5.5695 |
| American call | \$10.4466 |
| American put | \$6.0888 |

The small difference between European binomial and Black-Scholes prices is numerical discretization error.

---

## 15. Risk-Neutral Monte Carlo Simulation

Monte Carlo valuation simulates terminal asset prices under the risk-neutral probability measure:

$$
S_T^{(i)}
=
S_0
\exp
\left[
\left(
r-q-\frac{\sigma^2}{2}
\right)T
+
\sigma\sqrt{T}Z_i
\right]
$$

where:

$$
Z_i \sim N(0,1)
$$

for:

$$
i=1,2,\ldots,M
$$

and $M$ is the number of simulations.

---

## 16. Monte Carlo Option Prices

The European call price estimator is:

$$
\widehat{C}
=
e^{-rT}
\frac{1}{M}
\sum_{i=1}^{M}
\max
\left(
S_T^{(i)}-K,
0
\right)
$$

The European put price estimator is:

$$
\widehat{P}
=
e^{-rT}
\frac{1}{M}
\sum_{i=1}^{M}
\max
\left(
K-S_T^{(i)},
0
\right)
$$

Using 500,000 simulations and a fixed seed, the project produces approximately:

| Option | Monte Carlo estimate |
|---|---:|
| European call | \$10.4557 |
| European put | \$5.5738 |

These results are close to the Black-Scholes analytical values.

---

## 17. Antithetic Variates

The project uses antithetic variates as a variance-reduction technique.

For each standard normal shock $Z_i$, the simulation also uses:

$$
-Z_i
$$

The paired shocks are:

$$
Z_i,\ -Z_i
$$

This creates a more balanced sample and can improve numerical stability without changing the theoretical distribution.

Antithetic variates do not eliminate simulation uncertainty, but they can reduce unnecessary random imbalance.

---

## 18. Monte Carlo Standard Error

Let the discounted simulated payoffs be:

$$
X_1,X_2,\ldots,X_M
$$

The estimated option price is:

$$
\overline{X}
=
\frac{1}{M}
\sum_{i=1}^{M}X_i
$$

The estimated standard error is:

$$
SE(\overline{X})
=
\frac{s_X}{\sqrt{M}}
$$

where $s_X$ is the sample standard deviation of discounted payoffs.

As the number of simulations increases, the standard error generally decreases at the approximate rate:

$$
SE \propto \frac{1}{\sqrt{M}}
$$

Reducing the standard error by half generally requires approximately four times as many simulations.

---

## 19. Confidence Interval

A two-sided confidence interval is calculated as:

$$
\overline{X}
\pm
z_{\alpha/2}
SE(\overline{X})
$$

For a 95% confidence level:

$$
z_{\alpha/2}
\approx 1.96
$$

Therefore:

$$
CI_{95\%}
=
\left[
\overline{X}
-
1.96SE,
\overline{X}
+
1.96SE
\right]
$$

The confidence interval quantifies statistical sampling uncertainty. It does not capture model risk or parameter-estimation risk.

---

## 20. Method Comparison

| Method | Primary advantage | Primary limitation |
|---|---|---|
| Black-Scholes | Fast analytical solution | Restrictive assumptions and European exercise |
| Binomial tree | Supports American exercise | Discretization and computational cost |
| Monte Carlo | Flexible for complex stochastic payoffs | Sampling error and computational cost |

Black-Scholes is used as the analytical benchmark for European options.

The binomial model demonstrates numerical convergence and supports early exercise.

Monte Carlo provides a flexible probabilistic framework and reports statistical uncertainty.

---

## 21. Baseline Results

| Metric | Result |
|---|---:|
| Black-Scholes call | \$10.4506 |
| Black-Scholes put | \$5.5735 |
| Binomial European call | \$10.4466 |
| Binomial European put | \$5.5695 |
| Binomial American call | \$10.4466 |
| Binomial American put | \$6.0888 |
| Monte Carlo call | \$10.4557 |
| Monte Carlo put | \$5.5738 |
| Call Delta | 0.636831 |
| Put Delta | -0.363169 |
| Gamma | 0.018762 |
| Vega per 1% | 0.375240 |
| Call Theta per day | -0.017573 |
| Put Theta per day | -0.004542 |
| Call Rho per 1% | 0.532325 |
| Put Rho per 1% | -0.418905 |

The numerical results depend on the selected parameters, binomial steps, Monte Carlo sample size, and random seed.

---

## 22. Model Assumptions

The implemented models use simplifying assumptions:

1. The underlying asset price is lognormally distributed.
2. Volatility remains constant.
3. The risk-free rate remains constant.
4. Dividend yield remains constant.
5. Trading is continuous.
6. Markets are frictionless.
7. There are no transaction costs or taxes.
8. Assets are perfectly divisible.
9. Short selling is permitted.
10. Markets are sufficiently liquid.
11. European options can only be exercised at maturity.
12. American options may be exercised at any tree node.

---

## 23. Model Limitations

Real financial markets may exhibit:

- Volatility smiles and skews.
- Volatility clustering.
- Sudden price jumps.
- Changing interest rates.
- Discrete dividends.
- Transaction costs.
- Bid-ask spreads.
- Liquidity restrictions.
- Market-impact costs.
- Heavy-tailed return distributions.

The reported confidence intervals measure Monte Carlo sampling uncertainty only. They do not represent total financial risk.

---

## 24. Automated Validation

The project uses automated tests to verify:

- Known Black-Scholes call and put values.
- Calculation of $d_1$ and $d_2$.
- Put-call parity.
- Continuous-dividend pricing.
- Known Greek values.
- Relationships between call and put Greeks.
- Binomial prices.
- Convergence toward Black-Scholes.
- American early-exercise relationships.
- Monte Carlo reproducibility.
- Monte Carlo confidence intervals.
- Payoff calculations.
- Parameter validation.

The current implementation contains 66 passing tests.

---

## 25. Reproducibility

The Monte Carlo implementation uses a fixed random seed:

```python
seed = 42