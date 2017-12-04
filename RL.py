import numpy as np


class LogisticRegression:

    def __init__(self, input, n_in, n_out):

        self.W = np.zeros((n_in, n_out))
        self.b = np.zeros((n_out,))
        self.p_y_given_x = self.softmax(input.dot(self.W) + self.b)

    def softmax(self, f):
        f -= np.max(f)
        p = np.exp(f) / np.sum(np.exp(f))
        return p


def main():
    LogisticRegression()


if __name__ == '__main__':
    main()
