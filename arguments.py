import argparse


def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--population', type=int, default=10, help='Number of individuals in the population')

    parser.add_argument('--iterations', type=int, default=100, help='Number of evolution iterations')

    parser.add_argument('--k', type=float, default=0.5, help='Externality parameter')

    parser.add_argument('--m', type=float, default=0.5, help='Self-utility parameter')

    parser.add_argument('--mu', type=float, default=0.1, help='Standard deviation of mutation normal distribution.')

    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha (altruism) parameter. alpha=1 is an egoistic '
                                                                 'player. alpha=0.5 cares as much for others as for '
                                                                 'himself.')
    parser.add_argument('--rational', type=boolean, default = false, help = 'whether agents are rational or not')
    parser.add_argument('--reprod_method', type=str, default = "utility", help = 'Reproduction method')

    parser.add_argument('--alpha_mutable', dest='alpha_mutable',
                        help='If true, alpha will be subject to evolution mutations.')
    parser.set_defaults(alpha_mutable=False)

    parser.add_argument('--x_mutable', dest='x_mutable',
                        help='If true, alpha will be subject to evolution mutations.')
    parser.set_defaults(x_mutable=False)

    parser.add_argument('--random_x', dest='random_x',
                        help='If true, all x are initialized at random')
    parser.set_defaults(random_x=False)

    parser.add_argument('--random_alpha', dest='random_alpha',
                        help='If true, all alphas are initialized at random')
    parser.set_defaults(random_alpha=False)

    args = parser.parse_args()
    return args
