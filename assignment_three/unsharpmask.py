import numpy as np
from skimage.filters import convolve

def blurfilter(sigma, n=None):
	"""BLURFILTER Creates a Gaussian blur filter.
	FILT = BLURFILTER(SIGMA, N) creates a Gaussian blur filter with
	bandwidth parameter SIGMA, and size N x N. If N isn't specified, it
	defaults to ceil(6*sigma) + 1.

	The output of this function is the filter FILT."""

	if n == None:
		n = int(np.ceil(6*sigma)) + 1
			
	rad = (n - 1)/2
	xs, ys = np.meshgrid(np.linspace(-rad, rad, n), np.linspace(rad, -rad, n))
		
	# Your code here -- (make sure you understand what xs and ys represent!)
	# xs = [-rad, ..., rad] repeated n times as rows
	# ys = [rad, ..., -rad] repeated n times as cols
	# rad = radius of filter (in pixels)
	filter = np.exp(-(xs**2 + ys**2) / (2 * sigma**2))

	filter = filter / np.sum(filter) 
	return filter


def unsharpmask(img, sigma, amount):
   """UNSHARPMASK Perform unsharp masking on an image.
   FILTERED = UNSHARPMASK(IMG, SIGMA, AMOUNT) performs unsharp masking on
   the image IMG, using a blur parameter SIGMA and a strength AMOUNT.
   Unsharp masking consists of:

      1. Blurring the input image;
      2. Forming the "mask" by substracting the blurred image from the
         input image;
      3. Adding the mask, scaled by some amount, back to the input image.

   The output of this function is the filtered image FILTERED."""

   # Your code here
   filter = blurfilter(sigma=sigma)
   blurred = convolve(img, filter, mode="reflect")
   mask = img - blurred
   filtered = img + amount * mask
   return filtered