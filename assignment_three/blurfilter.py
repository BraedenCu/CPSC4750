import numpy as np

def blurfilter(sigma, n=None)
	"""BLURFILTER Creates a Gaussian blur filter.
   FILT = BLURFILTER(SIGMA, N) creates a Gaussian blur filter with
   bandwidth parameter SIGMA, and size N x N. If N isn't specified, it
   defaults to ceil(6*sigma) + 1.

   The output of this function is the filter FILT."""

	if n == None:
	 	n = int(np.ceil(6*sigma)) + 1


	rad = (n - 1)/2.
	xs, ys = np.meshgrid(np.linspace(-rad, rad, n), np.linspace(rad, -rad, n));

	# Your code here -- (make sure you understand what xs and ys represent!)
