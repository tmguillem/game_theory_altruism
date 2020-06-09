from arguments import get_argument_parser
from genetic_algorithm import GA


def main():
    args = get_argument_parser()

    if not args.random_alpha:
        # Use pre-defined alpha in arguments
        alpha = args.alpha
    else:
        # Will result in randomly generated alphas for each individual.
        alpha = None

    mutable_params = []
    if args.alpha_mutable:
        mutable_params += ['alpha']
    if args.x_mutable:
        mutable_params += ['x']

    genetic_algo = GA(n_population=args.population, m_iterations=args.iterations,
                      k=args.k, m=args.m,
                      mu=args.mu, alpha=alpha,
                      mutable_parameters=mutable_params)

    genetic_algo.run()

    print('End of program')


if __name__ == '__main__':
    main()
