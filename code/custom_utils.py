import numpy as np
import pandas as pd
import lightkurve as lk
from patsy import dmatrix
from tqdm import tqdm
import celerite
from scipy.optimize import minimize
import astropy.units as u
#import logging

def get_spline_dm(x, n_knots=20, degree=3, name='spline',
                   include_intercept=False):
    """Returns a `.DesignMatrix` which models splines using `patsy.dmatrix`.

    Parameters
    ----------
    x : np.ndarray
        vector to spline
    n_knots: int
        Number of knots (default: 20).
    degree: int
        Polynomial degree
    name: string
        Name to pass to `.DesignMatrix` (default: 'spline').
    include_intercept: bool
        Whether to include row of ones to find intercept. Default False.

    Returns
    -------
    dm: `.DesignMatrix`
        Design matrix object with shape (len(x), n_knots*degree).
    """
    dm_formula = "bs(x, df={}, degree={}, include_intercept={}) - 1" \
                 "".format(n_knots, degree, include_intercept)
    spline_dm = np.asarray(dmatrix(dm_formula, {"x": x}))
    df = pd.DataFrame(spline_dm, columns=['knot{}'.format(idx + 1)
                                          for idx in range(n_knots)])
    return lk.correctors.DesignMatrix(df, name=name)


def compute_BIC(lc, model_flux, nparams):
    """Compute the BIC given a residual"""
    resid = lc.flux - model_flux
    npoints = len(resid)
    penalty_term = nparams * np.log(npoints)
    # The sigma term can be dropped since it is constant
    #  we only care about changes to BIC
    sigma = np.median(np.abs(np.diff(lc.flux) / np.sqrt(2))) * 1.48
    likelihood_term = npoints * np.log(2 * np.pi * sigma**2) + np.sum(resid**2) / sigma**2

    BIC = likelihood_term + penalty_term
    return BIC


def spline_model_comparison_BIC(lc, cadence_mask=None):
    """Iteratively fits for spline knots with BIC for model comparison"""
    # Initialize regression corrector once
    regc = lk.correctors.RegressionCorrector(lc)

    # Set up the model comparison
    n_model_comparisons = 20
    # Breakpoint spacing will yield units of days, logarithmically spaced.
    bkspaces = np.logspace(np.log10(0.9), np.log10(20), num=n_model_comparisons)
    total_time = lc.time.max() - lc.time.min()
    all_n_knots = np.ceil(total_time / bkspaces + 1).astype(np.int)
    assert np.min(all_n_knots) >=3

    BICs = np.zeros(len(all_n_knots))
    model_fits = {}
    for i, n_knots in enumerate(all_n_knots):
        dm = get_spline_dm(lc.time, n_knots=n_knots).append_constant()
        # It's possible this will fail if too many cadences are masked...
        _ = regc.correct(dm, sigma=3, niters=5, cadence_mask=cadence_mask)
        model_flux = np.dot(dm.values, regc.coefficients)
        model_fits[i] = model_flux
        n_params = n_knots+3-1
        BICs[i] = compute_BIC(lc, model_flux, n_params)

    best = np.argmin(BICs)
    return (model_fits[best], bkspaces[best])


def make_cadence_mask(lc_raw, this_kepid_df):
    """Make the mask of TCEs"""
    n_tces = len(this_kepid_df)
    tce_mask = np.zeros(len(lc_raw), dtype=bool)
    duration_bloom = 1.5 # mask 50% more of the duration, since TTVs could exist
    for i in range(n_tces):
        this_fold = lc_raw.fold(this_kepid_df.tce_period.values[i],
                               t0=this_kepid_df.tce_time0bk.values[i])
        fractional_duration = (this_kepid_df.tce_duration.values[i] / 24.0) / this_kepid_df.tce_period.values[i]
        phase_mask = np.abs(this_fold.phase) < (fractional_duration * duration_bloom)
        this_mask = np.in1d(lc_raw.time, this_fold.time_original[phase_mask])
        tce_mask = tce_mask | this_mask
    return tce_mask


def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]


def estimate_gp_mean_model(lc, t=None):
    """return a MAP GP mean model for input lightcurve"""
    pg = lc.normalize(unit='unscaled').to_periodogram(normalization='psd', freq_unit=1/u.day)
    guess_period = pg.period_at_max_power.to(u.day).value
    variance = np.var(lc.flux)

    Q_guess = 15.0
    w0_guess = 2.0*np.pi / guess_period
    S0_guess = variance /10
    bounds1 = dict(log_S0=(np.log(variance/1000), np.log(variance*1000)),
               log_Q=(np.log(3), np.log(200)),
               log_omega0=(np.log(w0_guess*0.8),np.log(w0_guess*1.2)))

    kernel_sho = celerite.terms.SHOTerm(log_S0=np.log(S0_guess), log_Q=np.log(Q_guess),
                       log_omega0=np.log(w0_guess), bounds=bounds1)

    net_kernel = kernel_sho
    gp = celerite.GP(net_kernel, fit_mean=False, mean=lc.flux.mean())
    gp.compute(lc.time, yerr=lc.flux_err)

    initial_params = gp.get_parameter_vector()

    bounds = gp.get_parameter_bounds()
    soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                    method="L-BFGS-B", bounds=bounds, args=(lc.flux, gp))
    gp.set_parameter_vector(soln.x)
    out = gp.predict(lc.flux, return_cov=False)
    if t is not None:
        out = gp.predict(lc.flux, t=t, return_cov=False)
    return out

def batch_flatten_collection(lcs, kepid_df, method='GP', return_trend=False):
    """Batch flatten a LightCurveCollection, grouping by quarter

    Parameters:
    -----------
    lcs: LightCurveCollection
        A collection of lightcurves for a single planet host star system

    kepid_df: pandas DataFrame
        A pandas DataFrame containing the periods and t0 of each TCE in the
        system.  Uses the same column names as the DR24 catalog

    method: string
        Either "GP" or "spline"

    return_trend: boolean
        Whether or not to return the inferred trend.  Default: False
    """
    # Flatten all the quarters first using splines
    lcs_flattened = lk.LightCurveCollection([])
    lcs_trends = lk.LightCurveCollection([])
    for lc_per_quarter in lcs:
        lc_per_quarter = lc_per_quarter.remove_nans()
        tce_mask = make_cadence_mask(lc_per_quarter, kepid_df)
        if method=='GP':
            input_lc = lc_per_quarter[~tce_mask].remove_outliers()
            trend = estimate_gp_mean_model(input_lc, t=lc_per_quarter.time)
        if method=='spline':
            trend, knot_spacing = spline_model_comparison_BIC(lc_per_quarter, ~tce_mask)

        flattened = lc_per_quarter / trend
        lcs_flattened.append(flattened)
        lc_trend = lc_per_quarter.copy()
        lc_trend.flux = trend
        lcs_trends.append(lc_trend)
    if return_trend:
        return lcs_flattened, lcs_trends
    else:
        return lcs_flattened
