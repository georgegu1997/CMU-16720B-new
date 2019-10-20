import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
	# Input:
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    p = np.zeros(2)

    # Get the interpolation
    th, tw = It.shape
    ih, iw = It1.shape
    interp_It = RectBivariateSpline(np.arange(th), np.arange(tw), It)
    interp_It1 = RectBivariateSpline(np.arange(ih), np.arange(iw), It1)

    # Create the initial rectangle
    n_x = round(rect[2,0] - rect[0,0] + 1)
    n_y = round(rect[3,0] - rect[1,0] + 1)
    X = np.linspace(rect[0, 0], rect[2, 0], n_x)
    Y = np.linspace(rect[1, 0], rect[3, 0], n_y)

    while True:
        # Get the image gradient
        It1_y_grad = interp_It1(Y+p[1], X+p[0], dx=1)
        It1_x_grad = interp_It1(Y+p[1], X+p[0], dy=1)

        # Construct A and b
        A = np.hstack([It1_x_grad.reshape(-1, 1), It1_y_grad.reshape(-1, 1)])
        b = (interp_It(Y, X) - interp_It1(Y+p[1], X+p[0])).reshape(-1, 1)

        # Construct I-BB^T
        K = bases.shape[2]
        B = bases.reshape((-1, K))
        IBBT = np.eye(B.shape[0]) - B.dot(B.T)

        # Solve the least square problem
        dp, _, _, _ = np.linalg.lstsq(IBBT.dot(A), IBBT.dot(b), rcond=None)

        # p <- p + dp and the compare
        old_p = p.copy()
        p[0] += dp[0, 0]
        p[1] += dp[1, 0]

        change = np.linalg.norm(p - old_p)**2
        if change < 0.01:
            break

    return p

class LucasKanadeTrackerBasis():
    def __init__(self, rect, frame=None, bases=None):
        self.init_rect = rect
        self.init_frame = frame
        self.rect = self.init_rect.copy()
        self.bases = bases

    def update(self, f1, f2):
        dp = LucasKanadeBasis(f1, f2, self.rect, self.bases)
        self.rect[0] += dp[0]
        self.rect[2] += dp[0]
        self.rect[1] += dp[1]
        self.rect[3] += dp[1]

    def get_rect(self):
        return self.rect.copy()
