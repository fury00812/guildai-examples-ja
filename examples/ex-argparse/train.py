import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--x', type=float, default=0.1)
parser.add_argument('--noise', type=float, default=0.1)
opt = parser.parse_args()


def main():
    x = opt.x
    noise = opt.noise
    loss = (np.sin(5 * x) * (1 - np.tanh(x ** 2)) + np.random.randn() * noise)
    print("x: %f" % x)
    print("noise: %f" % noise)
    print("loss: %f" % loss)


if __name__ == "__main__":
    main()
