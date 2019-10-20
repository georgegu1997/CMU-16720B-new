import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
    # Input:
    #    It: template image
    #    It1: Current image
    #    rect: Current position of the car
    #    (top left, bot right coordinates)
    #    p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #    p: movement vector [dp_x, dp_y]

    # Put your implementation here
    p = p0.copy()

    # Get the interpolation
    th, tw = It.shape
    ih, iw = It1.shape
    interp_It = RectBivariateSpline(np.arange(th), np.arange(tw), It)
    interp_It1 = RectBivariateSpline(np.arange(ih), np.arange(iw), It1)

    # Create the initial rectangle
    X = np.arange(rect[0, 0], rect[2, 0]+1, 1)
    Y = np.arange(rect[1, 0], rect[3, 0]+1, 1)

    while True:
        # Get the image gradient, Y first because Y indicates row number
        It1_y_grad = interp_It1(Y+p[1], X+p[0], dx=1) # dx=1 means first derivative along the first input Y
        It1_x_grad = interp_It1(Y+p[1], X+p[0], dy=1)

        # Construct A and b
        A = np.hstack([It1_x_grad.reshape(-1, 1), It1_y_grad.reshape(-1, 1)])
        b = (interp_It(Y, X) - interp_It1(Y+p[1], X+p[0])).reshape(-1, 1)

        # Solve the least square problem
        dp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # p <- p + dp and the compare
        old_p = p.copy()
        p[0] += dp[0, 0]
        p[1] += dp[1, 0]

        change = np.linalg.norm(dp)**2
        if change < 0.01:
            break

    return p

class LucasKanadeTracker():
    def __init__(self, rect, frame=None):
        self.init_rect = rect
        self.init_frame = frame
        self.rect = self.init_rect.copy()

    def update(self, f1, f2):
        dp = LucasKanade(f1, f2, self.rect)
        self.rect[0] += dp[0]
        self.rect[2] += dp[0]
        self.rect[1] += dp[1]
        self.rect[3] += dp[1]

    def get_rect(self):
        return self.rect.copy()

def getTemplate(frame, rect):
    h, w = frame.shape
    interp = RectBivariateSpline(np.arange(h), np.arange(w), frame)
    X = np.arange(rect[0, 0], rect[2, 0]+1, 1)
    Y = np.arange(rect[1, 0], rect[3, 0]+1, 1)
    template = interp(Y, X)
    return template

class LucasKanadeTrackerWithTemplateCorrection():
    def __init__(self, rect, frame):
        self.init_rect = rect
        self.init_frame = frame

        # Get the initial template
        self.init_template = getTemplate(self.init_frame, self.init_rect)
        # Here, we explicit separate the template and keep it in a standalone variable
        self.template = self.init_template.copy()
        self.template_rect = np.array([[0], [0], [self.init_rect[2] - self.init_rect[0]], [self.init_rect[3]-self.init_rect[1]]], dtype=float)

        self.p0 = self.init_rect[:2, 0].reshape(-1)
        self.p = self.p0.copy()

        self.rect = None
        self.rect_star = None

    def update(self, f1, f2):
        # Ordinary tracking the template in the last frame to the next frame
        pn = LucasKanade(self.template, f2, self.template_rect, self.p) # from p_{n-1}

        rect = self.template_rect.copy()
        rect[0] += pn[0]
        rect[2] += pn[0]
        rect[1] += pn[1]
        rect[3] += pn[1]
        self.rect = rect

        # Try to recover the initial template
        # By track the initial template from pn to the next frame
        pnstar = LucasKanade(self.init_template, f2, self.template_rect, pn) # from p_n

        rect_star = self.template_rect.copy()
        rect_star[0] += pnstar[0]
        rect_star[2] += pnstar[0]
        rect_star[1] += pnstar[1]
        rect_star[3] += pnstar[1]
        self.rect_star = rect_star

        # If small change, then get the template from the pnstar
        # Else, keep the template unchanged
        if np.linalg.norm(pn-pnstar, ord=1) <= 0.1:
            self.template = getTemplate(f2, rect_star)
        else:
            pass

        self.p = pn

    def get_rect(self):
        return self.rect_star
