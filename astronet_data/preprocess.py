# Modified from astronet for the PyTorch Summer Hackathon
# This version uses lightkurve to read in the Kepler/K2/TESS data

"""Functions for reading and preprocessing light curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from light_curve_util import kepler_io
from light_curve_util import median_filter
from light_curve_util import util
from third_party.kepler_spline import kepler_spline

import lightkurve as lk

def read_and_process_light_curve(kepid, kepler_data_dir, campaign, max_gap_width=0.75):
  """Reads a light curve, fits a B-spline and divides the curve by the spline.

  Args:
    kepid: Kepler id of the target star.
    kepler_data_dir: Base directory containing Kepler data. See
        kepler_io.kepler_filenames().
    campaign: K2 campaign where data was taken.
    max_gap_width: Gap size (in days) above which the light curve is split for
        the fitting of B-splines.

  Returns:
    time: 1D NumPy array; the time values of the light curve.
    flux: 1D NumPy array; the normalized flux values of the light curve.

  Raises:
    IOError: If the light curve files for this Kepler ID cannot be found.
    ValueError: If the spline could not be fit.
  """
  # Read the Kepler light curve.
  lcf = lk.search_lightcurvefile(kepid, campaign=campaign).download()

  lcf.SAP_FLUX

  #TODO: do some data cleaning/ data munging quality assurance here

  return lcf.SAP_FLUX.time, lcf.SAP_FLUX.flux


def phase_fold_and_sort_light_curve(lc, period, t0):
  """Phase folds a light curve and sorts by ascending time.

  Args:
    time: 1D NumPy array of time values.
    flux: 1D NumPy array of flux values.
    period: A positive real scalar; the period to fold over.
    t0: The center of the resulting folded vector; this value is mapped to 0.

  Returns:
    folded_time: 1D NumPy array of phase folded time values in
        [-period / 2, period / 2), where 0 corresponds to t0 in the original
        time array. Values are sorted in ascending order.
    folded_flux: 1D NumPy array. Values are the same as the original input
        array, but sorted by folded_time.
  """
  # Phase fold time.
  folded_lc = lc.fold(period, t0=t0)

  return folded_lc.time, folded_lc.flux


def generate_view(lc, num_bins, bin_width, t_min, t_max,
                  normalize=True):
  """Generates a view of a phase-folded light curve using a median filter.

  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    num_bins: The number of intervals to divide the time axis into.
    bin_width: The width of each bin on the time axis.
    t_min: The inclusive leftmost value to consider on the time axis.
    t_max: The exclusive rightmost value to consider on the time axis.
    normalize: Whether to center the median at 0 and minimum value at -1.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  if bin_width is not None:
      num_bins = len(lc.flux)/bin_width

  binned_lc = lc.bin(num_bins)
  view = median_filter.median_filter(time, flux, num_bins, bin_width, t_min,
                                     t_max)



  return view


def global_view(time, flux, period, num_bins=701, bin_width_factor=1 / 701):
  """Generates a 'global view' of a phase folded light curve.

  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of period.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=period * bin_width_factor,
      t_min=-period / 2,
      t_max=period / 2)


def local_view(time,
               flux,
               period,
               duration,
               num_bins=51,
               bin_width_factor=8/51,
               num_durations=4):
  """Generates a 'local view' of a phase folded light curve.

  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    duration: The duration of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of duration.
    num_durations: The number of durations to consider on either side of 0 (the
        event is assumed to be centered at 0).

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=duration * bin_width_factor,
      t_min=max(-period / 2, -duration * num_durations),
      t_max=min(period / 2, duration * num_durations))
