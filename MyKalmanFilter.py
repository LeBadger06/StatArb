import numpy as np
class MyKalmanFilter:

    def __init__(self, delta=1e-4, R=1e-3):
        # measurement noise variance
        self.R = R

        # co-variance of process noise(2 dimensions)
        self.Q = delta / (1 - delta) * np.eye(2)

        # state (slope, intercept) will be (2 x n), we will initialize with just one column at first
        self.x = np.zeros((2, 1))

        # state covariance
        self.P = np.zeros((2, 2))
        self.i = 0

    def step_forward(self, y1, y2):
        self.i += 1
        print(self.i)
        # Before entering the equations, let's define H as (1, 2) matrix
        H = np.array([y2, 1])[None]
        # and define z
        z = np.array(y1)

        ## TIME UPDATE ##
        # first thing is to predict new state as the previous one (2x1)
        x_hat = self.x[:, -1][..., None]

        # then, the uncertainty or covariance prediction
        P_hat = self.P + self.Q

        ## MEASUREMENT UPDATE ##
        # calc the Kalman gain
        K = P_hat.dot(H.T) / (H.dot(P_hat.dot(H.T)) + self.R)
        # print(K)

        # state update part 1 (measurement estimation)
        z_hat = H.dot(x_hat)
        # print(z_hat)
        # print(z)
        # print(z-z_hat)
        # state update part 2
        x = x_hat + K.dot(z - z_hat)

        # uncertainty update
        P = (np.eye(2) - K.dot(H)).dot(P_hat)
        # self.P=P

        # append the new state to the vector
        self.x = np.concatenate([self.x, x], axis=1)

        return x, P, K, z_hat