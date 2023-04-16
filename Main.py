import argparse
from Parameters import Parameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='System Parameters')

    """ ================ Command Line Options ================= """
    parser.add_argument('--algo', dest='algo', default='Baseline')

    args = parser.parse_args()

    """ ================ Parameters ================ """

    params = Parameters()

    if args.algo == 'DQN':
        import DQN
        DQN.train(params)
        params = Parameters()
        DQN.evaluate(params)
    elif args.algo == 'Baseline':
        import Baseline
        Baseline.evaluate(params)
    else:
        raise Exception('Algo not found.')
