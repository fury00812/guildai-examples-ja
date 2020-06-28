import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="data_default")
opt = parser.parse_args()


def main():
    print("function: preprocess.main")
    data = opt.data
    print("data: %s" % data)


if __name__ == "__main__":
    main()
