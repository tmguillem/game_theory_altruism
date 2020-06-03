import argparse


def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--population', type=int, default=10, help='Number of individuals in the population')

    parser.add_argument('--iterations', type=int, default=10, help='Number of evolution iterations')

    parser.add_argument('--k', type=float, default=0.5, help='Externality parameter')

    parser.add_argument('--m', type=float, default=0.5, help='Self-utility parameter')

    parser.add_argument('--mu', type=float, default=0.1, help='Standard deviation of mutation normal distribution.')

    args = parser.parse_args()
    return args
