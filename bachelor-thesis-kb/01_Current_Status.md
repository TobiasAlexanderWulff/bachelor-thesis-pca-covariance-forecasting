# Current Status

Last updated: 2026-07-28

## Project phase

The codebase is being rebuilt from the ground up to reflect the current methodological direction and to make every implementation decision explicit and understandable.

The previous repository, `bachelor-thesis-volatility-forecasting`, remains unchanged as a reference for earlier implementations, tests, experiments, and decisions. Components from it are not transferred automatically.

The model-independent data pipeline and the main PCA-based approximation pipeline have been implemented. This includes the look-ahead-safe training reference covariance and basis, transformation into and from the reference basis, construction of covariance approximations, calculation and eigendecomposition of the residual matrices, and sequential matching of the residual eigencomponents. No forecasting model has been implemented yet.
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

- The exact construction of the historical reference covariance matrix and reference basis must be specified without look-ahead bias.
- It remains to be decided whether the residuals are negligible or must be forecast explicitly.
- If residuals are forecast, the treatment or estimation of their changing eigenvector bases remains open.
- Because $E_i$ is not necessarily positive semidefinite, its decomposition must be interpreted as an eigendecomposition; its eigenvalues are not automatically variances.
- The exact FARIMA specification remains open.

## Forecasting clarification

The dominant series $\widetilde{\lambda}_{i,11}$ is to be forecast using FARIMA$(0,d,0)$, where $d$ is estimated from the data. The remaining series are initially to be forecast using autoregressive models. If this treatment is not satisfactory, FARIMA$(0,d,0)$ may also be applied to the remaining series.

A naive one-step-ahead forecast remains the forecasting baseline. The estimator and implementation used for the fractional differencing parameter $d$, as well as the exact autoregressive specification for the remaining series, still need to be selected and justified.

## Current implementation status

The repository currently contains:

- a Python 3.12.13 environment managed with `uv`,
- an installable `src` package,
- `pandas` as the first runtime dependency,
- a downloader for the fixed Q1 2024 Binance Spot dataset,
- a central validator for the downloaded raw data,
- a loader that combines the validated Q1 2024 closing-price series,
- a function that calculates one-minute log returns,
- a function that estimates sample covariance matrices from consecutive, non-overlapping blocks of 30 one-minute returns,
- construction and eigendecomposition of the training reference covariance matrix,
- transformation of covariance matrices into and from the fixed reference basis,
- construction of covariance approximations using the dominant transformed component,
- calculation and independent eigendecomposition of the residual matrices,
- sequential matching and sign alignment of residual eigencomponents.
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

The next implementation step is to reconstruct the residual matrices from their matched eigenvectors and eigenvalues and verify that temporal matching preserves every original residual matrix. Afterwards, the temporal stability of the residual bases and the suitability of the previous basis as a naive one-step estimate can be investigated.