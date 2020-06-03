import numpy as np
from arguments import get_argument_parser
from genetic_algorithm import GA


def main():
    args = get_argument_parser()

    genetic_algo = GA(n_population=args.population, m_iterations=args.iterations,
                      k=args.k, m=args.m,
                      mu=args.mu)

    genetic_algo.run()

    print('End of program')


if __name__ == '__main__':
    main()
