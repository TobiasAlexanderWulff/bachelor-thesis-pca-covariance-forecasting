# Current Status

Last updated: 2026-07-29

## Project phase

The codebase is being rebuilt from the ground up to reflect the current methodological direction and to make every implementation decision explicit and understandable.

The previous repository, `bachelor-thesis-volatility-forecasting`, remains unchanged as a reference for earlier implementations, tests, experiments, and decisions. Components from it are not transferred automatically.

The model-independent data pipeline, the PCA-based approximation pipeline, residual-basis diagnostics, covariance-forecast evaluation utilities, and the primary end-to-end forecasting comparison have been implemented. This includes the look-ahead-safe training reference covariance and basis, construction of single-indicator covariance approximations, residual eigendecompositions and diagnostics, timestamp-level forecast losses, positive-semidefiniteness diagnostics, dependence-adjusted loss-difference inference, and circular block-bootstrap intervals.

Commit `d418fc6` introduced the repository-reproducible primary forecasting analysis in `scripts/analyze_covariance_forecasts.py`. It estimates the dominant-indicator ARFIMA model using training data only and compares it against the direct naive covariance forecast and the naive dominant-indicator forecast on the fixed holdout.

Forecasts of the complete residual eigendecomposition, residual-correction experiments, loss-concentration diagnostics, and additional inference sensitivity checks remain exploratory unless explicitly identified below. They must not be presented as repository-reproducible thesis evidence yet.

## Covariance-matrix construction

The one-minute log returns are partitioned chronologically into consecutive, non-overlapping blocks of \(m=30\) observations. For block \(i\), the sample covariance matrix is estimated as

$$
\Sigma_i
=
\frac{1}{m-1}
\sum_{j=1}^{m}
\left(r_{m(i-1)+j}-\bar{r}_i\right)
\left(r_{m(i-1)+j}-\bar{r}_i\right)^\top,
\qquad m=30,
$$

where

$$
\bar{r}_i
=
\frac{1}{m}
\sum_{j=1}^{m} r_{m(i-1)+j}.
$$

Thus, $\Sigma_1$ uses returns 1–30, $\Sigma_2$ uses returns 31–60, and so forth. Only complete blocks are retained. Each matrix is timestamped with the final return observation in its block. No temporal scaling or annualization is applied.

## Fixed PCA reference basis

The 4,367 covariance matrices are split chronologically into 2,183
training matrices and 2,184 test matrices. The fixed reference
covariance matrix is estimated exclusively from the training set as

$$
\widetilde{\Sigma}_{\mathrm{train}}
=
\frac{1}{N_{\mathrm{train}}}
\sum_{i=1}^{N_{\mathrm{train}}}\Sigma_i,
\qquad
N_{\mathrm{train}}=2183.
$$

Its eigendecomposition is

$$
\widetilde{\Sigma}_{\mathrm{train}}
=
\widetilde{B}
\widetilde{\Lambda}
\widetilde{B}^{\top},
$$

where the eigenvalues and corresponding eigenvectors are ordered by
decreasing eigenvalue. The resulting reference basis
$\widetilde{B}$ remains fixed for both the training and test periods.

Each covariance matrix is transformed into this reference basis as

$$
\widetilde{\Lambda}_i
=
\widetilde{B}^{\top}\Sigma_i\widetilde{B}.
$$

The first reference component explains approximately $75.87\%$ of the total variance represented by the training reference covariance matrix, followed by $17.53\%$ and $6.60\%$ for the second and third components.

The implementation was verified by checking the orthonormality of
$\widetilde{B}$, reconstruction of the reference covariance matrix,
and reconstruction of all 4,367 individual covariance matrices. The
mean transformed training matrix is diagonal up to floating-point
rounding and its diagonal equals the ordered reference eigenvalues.
Individual transformed covariance matrices are generally not
diagonal.

## Established methodological direction

Given a time series of covariance matrices

$$
\Sigma_1, \Sigma_2, \ldots, \Sigma_n,
$$

a historical reference covariance matrix is decomposed as

$$
\widetilde{\Sigma}
=
\widetilde{B}
\widetilde{\Lambda}
\widetilde{B}^{\top}.
$$

Each covariance matrix is represented in the fixed reference basis:

$$
\widetilde{\Lambda}_i
=
\widetilde{B}^{\top}
\Sigma_i
\widetilde{B}.
$$

The dominant diagonal element

$$
\widetilde{\lambda}_{i,11}
$$

is the central scalar indicator investigated in the thesis.

The main research question is whether this single indicator contains enough information to obtain useful covariance forecasts.

## Approximation and residual

A covariance approximation is reconstructed using the dominant indicator and the historical reference decomposition. The remaining approximation error is

$$
E_i
=
\Sigma_i-\widehat{\Sigma}_i.
$$

The residual matrices must be examined to determine whether their information is small enough to be neglected in forecasting.

The currently discussed residual decomposition is

$$
E_i
=
B_{E,i}\Theta_i B_{E,i}^{\top},
$$

where $B_{E,i}$ contains eigenvectors of $E_i$ and $\Theta_i$ contains its eigenvalues.

This replaces the representation in the initial PCA document that used the reference basis $\widetilde{B}$ for the residual decomposition. That earlier representation is not treated as the current specification.

## Temporal matching of residual eigencomponents

The eigendecomposition of each residual matrix is initially computed independently. Eigenvector signs and, particularly when eigenvalues change order, component positions are not temporally identifiable from these independent decompositions alone.

For each transition from $i-1$ to $i$, the current eigenvectors are therefore assigned to the previous eigenvectors by maximizing the total absolute similarity

$$
\sum_k
\left|
b_{E,k,i-1}^{\top}b_{E,\pi(k),i}
\right|
$$

over all component permutations $\pi$. The eigenvectors and their corresponding eigenvalues are reordered jointly. Eigenvector signs are subsequently aligned with the matched vectors from the previous time step.

The first residual eigendecomposition provides the component anchor. After matching, an error-component label describes the temporally tracked continuation of its initial direction rather than the eigenvalue rank at every time step. Consequently, the matched eigenvalues are not necessarily ordered by decreasing value.

Permutation and sign alignment do not alter the represented residual matrix:

$$
E_i
=
B_{E,i}\Theta_iB_{E,i}^{\top}.
$$

Near-equal eigenvalues remain a methodological complication because individual eigenvectors may then be poorly identifiable even when their joint eigenspace is stable.

## Open methodological questions

- The parametric profile Whittle estimator, admissible range of $d$, fractional forecast recursion, and associated diagnostics still require a precise literature-based presentation in the thesis.
- Forecasting the complete residual eigendecomposition remains open. The exploratory identity-basis diagonal corrections do not forecast the changing residual eigenvector basis and are not equivalent to a complete residual forecast.
- The earlier proposal to forecast additional series with autoregressive models belongs to an extended specification. It is not part of the committed primary one-indicator model and requires an explicit scope decision before implementation.
- The exploratory concentration diagnostic must be reproduced from committed code before it is used as formal thesis evidence.
- Evidence from one Q1 2024 holdout and one three-asset portfolio must not be generalized to other periods, frequencies, portfolios, or markets without further evidence.

## Forecasting specification

The committed primary candidate model forecasts only the dominant series

$$
\widetilde{\lambda}_{t,11}
$$

using ARFIMA$(0,d,0)$. Its mean, fractional differencing parameter $d$, and innovation variance are estimated exclusively from the outer training set using a parametric profile Whittle estimator. These parameter estimates remain fixed throughout the holdout. Each one-step-ahead forecast uses only observations strictly preceding its target timestamp.

The second and third diagonal values in the fixed reference basis remain equal to their training-reference eigenvalues, while the reference-basis off-diagonal values remain zero. The primary candidate therefore uses no residual correction and deliberately represents the one-indicator research design.

The direct naive covariance forecast

$$
\widehat{\Sigma}_t=\Sigma_{t-1}
$$

is the primary benchmark. A naive dominant-indicator forecast is retained as an additional baseline.

## Reproducible primary forecasting result and exploratory supplements

The primary residual-free forecasting comparison and its designated primary loss-difference inference are reproduced by the committed script `scripts/analyze_covariance_forecasts.py` introduced in commit `d418fc6`. The underlying evaluation functions are covered by the repository test suite.

The experiment uses all 4,367 covariance matrices, with the first 2,183 matrices as outer training data and the remaining 2,184 matrices as a fixed outer test set. The PCA reference basis and all ARFIMA parameters are estimated only from the outer training set and then held fixed throughout the outer test. Every forecast uses only observations strictly before its target timestamp.

Residual-correction results, inference sensitivity checks beyond the committed primary configuration, and the concentration analysis remain exploratory. The empirical model preference is a working project decision and has not been separately confirmed by the supervisor.

### Main covariance-forecast comparison

| Method | Overall test RMSE | RMSE change versus direct naive matrix | Non-PSD forecasts |
| --- | ---: | ---: | ---: |
| Direct naive covariance, $\widehat\Sigma_t=\Sigma_{t-1}$ | $2.623096\times10^{-6}$ | $0.00\%$ | $0$ |
| Naive dominant indicator, no residual correction | $2.495972\times10^{-6}$ | $-4.85\%$ | $0$ |
| ARFIMA dominant indicator, no residual correction | $2.221592\times10^{-6}$ | $-15.31\%$ | $0$ |

Relative to the direct naive covariance forecast, the ARFIMA dominant-indicator forecast without residual correction reduced RMSE by $15.31\%$. Its diagonal RMSE decreased by $16.95\%$ and its off-diagonal RMSE by $13.33\%$. Equivalently, its mean timestamp-level squared Frobenius loss was $28.27\%$ lower:

$
1-
\frac{4.441923\times10^{-11}}
     {6.192570\times10^{-11}}
=
0.282701.
$

Relative to the naive dominant-indicator forecast without residual correction, ARFIMA reduced overall RMSE by $10.99\%$.

### Exploratory diagonal residual corrections

The supplementary residual forecasts used the three diagonal coefficients of the residual in the identity basis,

$
d_{a,t}=E_{t,aa},
$

and therefore changed only the covariance diagonals. They did not forecast the complete residual eigendecomposition.

AR forecasting of these coefficients improved the complete ARFIMA-indicator model by only $0.29\%$ in overall RMSE and produced 52 non-PSD matrices. Separate ARFIMA$(0,d_a,0)$ forecasts improved the complete model by $0.52\%$ and produced 65 non-PSD matrices. A naive residual carry-forward worsened overall RMSE by $4.44\%$ and produced 925 non-PSD matrices.

The exploratory residual ARFIMA estimates were

$
\widehat d_{\mathrm{BTC}}=0.046322,\qquad
\widehat d_{\mathrm{ETH}}=0.103209,\qquad
\widehat d_{\mathrm{BNB}}=0.150444.
$

ARFIMA reduced the isolated residual-coefficient RMSE by $2.73\%$ relative to a zero correction, but the incremental benefit in the complete covariance model remained small. These results do not justify accepting invalid covariance forecasts merely because their unconstrained RMSE is slightly lower.

### Primary loss-difference inference

The pre-specified primary comparison was the ARFIMA dominant-indicator forecast without residual correction against the direct naive covariance forecast, using timestamp-level squared Frobenius loss.

A two-sided HAC test using a Bartlett/Newey-West long-run variance estimate with automatic truncation lag 7 produced

$
p_{\mathrm{HAC}}=0.1903.
$

The supplementary 95% circular block-bootstrap percentile interval, using block length 13, 100,000 replications, and random seed 20260729, was

$$
[-1.1632\times10^{-13},\ 4.6911\times10^{-11}],
$$

which includes zero.

The null hypothesis of equal expected loss was therefore not rejected at the 5% level. This does not establish equal predictive accuracy; it means that the analysis did not provide sufficient evidence of a nonzero expected loss difference under the applied inference procedures.

For the supplementary off-diagonal loss, the bootstrap lower bound was only marginally above zero, while the corresponding HAC test produced $p=0.1690$. This disagreement does not support a robust supplementary significance claim.

### Exploratory concentration of the aggregate advantage

The ARFIMA dominant-indicator model had the smaller timestamp-level loss for 793 of 2,184 test observations, or $36.31\%$. The direct naive covariance forecast won the remaining $63.69\%$.

The 22 largest positive loss differences, corresponding to approximately 1% of the test observations, accounted for $94.05\%$ of all positive gross gains and $157.98\%$ of the final net advantage. A net share above 100% is possible because losses at other timestamps offset part of these gains.

The largest single gain occurred at 2024-02-28 18:30 UTC and was approximately equal to the entire final net advantage. The second-largest occurred at 2024-03-05 20:30 UTC. Large candidate losses occurred one 30-minute interval before both events. A plausible interpretation is that ARFIMA initially misses abrupt covariance changes but avoids carrying short-lived extreme matrices forward as strongly as the direct naive benchmark. This is a diagnostic interpretation, not an established causal explanation.

### Current working model decision

Further residual tuning is frozen. The preferred structurally valid exploratory model is

$
\boxed{\text{ARFIMA dominant indicator without residual correction}}.
$

This choice does not simply select the numerically lowest unconstrained RMSE. It prioritizes the central one-indicator research design, a material aggregate error reduction, and positive-semidefinite forecasts. The result must be described with two central caveats:

1. statistical superiority over the direct naive covariance forecast was not established at the 5% level;
2. the aggregate advantage was highly concentrated in a few large-error events rather than being present at most timestamps.

The model is now repository-reproducible. It nevertheless remains a working project decision until the final scope and interpretation have been confirmed with the supervisor.

### Oracle interpretation

An oracle variant that inserts observed diagonals while retaining approximated off-diagonals is not necessarily positive semidefinite. It is only an unconstrained Frobenius-error representation bound and must not be described as a valid perfect covariance forecast.

### Statistical references

- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics, 13*(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599
- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica, 55*(3), 703–708. https://doi.org/10.2307/1913610
- Künsch, H. R. (1989). The Jackknife and the Bootstrap for General Stationary Observations. *The Annals of Statistics, 17*(3), 1217–1241. https://doi.org/10.1214/aos/1176347265

## Current implementation status

The repository currently contains:

- a Python 3.12.13 environment managed with `uv`,
- an installable `src` package,
- `numpy` and `pandas` as runtime dependencies,
- a downloader for the fixed Q1 2024 Binance Spot dataset,
- a central validator for the downloaded raw data,
- a loader that combines the validated Q1 2024 closing-price series,
- a function that calculates one-minute log returns,
- a function that estimates sample covariance matrices from consecutive, non-overlapping blocks of 30 one-minute returns,
- construction and eigendecomposition of the training reference covariance matrix,
- transformation of covariance matrices into and from the fixed reference basis,
- construction of covariance approximations using the dominant transformed component,
- calculation and independent eigendecomposition of the residual matrices,
- sequential matching and sign alignment of residual eigencomponents,
- reconstruction of residual matrices from matched eigendecompositions,
- measurement of consecutive matched residual-basis similarities,
- residual approximation using the previous matched basis and current eigenvalues,
- Frobenius norms, relative Frobenius errors, and aggregated relative Frobenius errors,
- timestamp-level overall, diagonal, and off-diagonal squared Frobenius losses,
- overall, diagonal, and off-diagonal covariance RMSE,
- positive-semidefiniteness diagnostics,
- Bartlett-HAC equal-accuracy tests with an automatically selected truncation lag,
- circular block-bootstrap intervals with a deterministic random seed,
- a committed end-to-end analysis comparing direct naive covariance, naive dominant-indicator, and ARFIMA dominant-indicator forecasts,
- 22 passing unit tests covering the current package-level functionality,
- an initial README,
- this knowledge base.

The fixed raw dataset comprises one-minute Spot klines for BTCUSDT, ETHUSDT, and BNBUSDT from January through March 2024.

The validator checks:

- the official SHA-256 checksums,
- the expected ZIP and CSV structure,
- twelve Kline columns per observation,
- complete monthly coverage,
- exact one-minute timestamp continuity,
- numeric, finite, and strictly positive OHLC prices.

After this central validation succeeds, subsequent pipeline stages may treat the raw input data as structurally valid. This validation establishes file integrity and suitability for the pipeline; it does not independently verify the economic accuracy of Binance market data.

For the complete Q1 2024 dataset, the covariance calculation produces 4,367 matrices of dimension $3 \times 3$. The final 29 returns form an incomplete block and are therefore discarded. The resulting matrices were verified to contain only finite values, to be symmetric and positive semidefinite, and the first matrix was compared strictly against a direct calculation from the corresponding return block.

The next production implementation steps are:

1. reproduce the committed primary analysis from a fresh checkout and clean `uv` environment;
2. add targeted regression tests for the ARFIMA estimator, fractional forecast recursion, and strict one-step-ahead alignment in the end-to-end analysis;
3. decide with the supervisor whether forecasting the complete residual structure remains within scope;
4. integrate the concentration diagnostic only if it will be used as formal thesis evidence;
5. transfer the committed methodology and primary results into thesis-ready tables, figures, and written interpretation.

No additional residual-model tuning should be performed before the residual-forecasting scope has been explicitly decided.