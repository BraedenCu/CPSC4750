import numpy as np

def gaborfilter(theta, wavelength, phase, sigma, aspect, ksize=None):

	"""
	GB = GABORFILTER(THETA, WAVELENGTH, PHASE, SIGMA, ASPECT, KSIZE)
	creates a Gabor filter GB with orientation THETA (in radians),
	wavelength WAVELENGTH (in pixels), phase offset PHASE (in radians),
	envelope standard deviation SIGMA, aspect ratio ASPECT, and dimensions
	KSIZE x KSIZE. KSIZE is an optional parameter, and if omitted default
	dimensions are selected.
	 """

	if ksize is None:
		ksize = 8*sigma*aspect

	if type(ksize) == int or len(ksize) == 1:
		ksize = [ksize, ksize]

	xmax = np.floor(ksize[1]/2.)
	xmin = -xmax
	ymax = np.floor(ksize[0]/2.)
	ymin = -ymax

	xs, ys = np.meshgrid(np.arange(xmin,xmax+1), np.arange(ymax,ymin-1,-1))

	# Your code here

	# note: my gabor filter generates all 16 at once, and returns them
	# as a list. Got permission from a ULA that this approach was okay.
	orientations_edge = np.linspace(0, 7*np.pi/4.0, num=8, endpoint=True)
	orientations_line = np.linspace(0, 3*np.pi/4.0, num=4, endpoint=True)
	filters = []

	for theta in orientations_edge:
		x_prime = xs * np.cos(theta) + ys * np.sin(theta)
		y_prime = -xs * np.sin(theta) + ys * np.cos(theta)

		g_ij = (np.sin(2 * np.pi * y_prime / wavelength) *
				np.exp(-(x_prime**2 / (2 * (sigma * aspect)**2) +
						y_prime**2 / (2 * sigma**2))))

		g_ij = g_ij - np.mean(g_ij)
		g_ij = g_ij / np.sqrt(np.sum(g_ij**2))
		filters.append(g_ij)

	for theta in orientations_line:
		x_prime = xs * np.cos(theta) + ys * np.sin(theta)
		y_prime = -xs * np.sin(theta) + ys * np.cos(theta)

		g_ij = (np.cos(2 * np.pi * y_prime / wavelength) *
				np.exp(-(x_prime**2 / (2 * (sigma * aspect)**2) +
						y_prime**2 / (2 * sigma**2))))

		g_ij = g_ij - np.mean(g_ij)
		g_ij = g_ij / np.sqrt(np.sum(g_ij**2))
		filters.append(g_ij)

	for theta in orientations_line:
		x_prime = xs * np.cos(theta) + ys * np.sin(theta)
		y_prime = -xs * np.sin(theta) + ys * np.cos(theta)

		g_ij = (-np.cos(2 * np.pi * y_prime / wavelength) *
				np.exp(-(x_prime**2 / (2 * (sigma * aspect)**2) +
						y_prime**2 / (2 * sigma**2))))

		g_ij = g_ij - np.mean(g_ij)
		g_ij = g_ij / np.sqrt(np.sum(g_ij**2))
		filters.append(g_ij)

	return filters