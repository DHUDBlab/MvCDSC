import argparse


def parse_args():
    """
    Parses the arguments.
    """
    parser = argparse.ArgumentParser(description="MvCDSC")
    parser.add_argument('--dataset', type=str, default='citeseer', help='Input dataset')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate. Default is 1e-3')
    parser.add_argument('--pre_lr', type=float, default=2e-5, help='Pretraining Learning rate. Default is 1e-5')
    parser.add_argument('--seed', type=int, default=0, help='Seed for fixing the results')
    parser.add_argument('--n-epochs', default=300, type=int, help='Number of epochs')
    parser.add_argument('--hidden_dims1', type=list, nargs='+', default=[1024, 512], help='Number of dimensions1')
    parser.add_argument('--hidden_dims2', type=list, nargs='+', default=[1024, 512], help='Number of dimensions2')
    parser.add_argument('--hidden_dims3', type=list, nargs='+', default=[512], help='Number of the common dimensions')
    parser.add_argument('--embedding', type=int, default=512, help='The dimension of hidden layer')
    parser.add_argument('--lambda_1', default=0.5, type=float, help='Edge reconstruction loss function')
    parser.add_argument('--lambda_2', default=10.0, type=float, help='C_Regular loss function')
    parser.add_argument('--lambda_3', default=5.0, type=float, help='Contrastive loss function')
    parser.add_argument('--lambda_4', default=0.08, type=float, help='Cpq loss function')
    parser.add_argument('--lambda_5', default=55, type=float, help='Consistent loss function')
    parser.add_argument('--beta_1', default=50, type=float, help='Combination coefficient')
    parser.add_argument('--beta_2', default=1, type=float, help='Combination coefficient')
    parser.add_argument('--beta_3', default=1, type=float, help='Combination coefficient')
    parser.add_argument('--cluster', default=6, type=float, help='The number of clusters')
    parser.add_argument('--n_sample', type=int, default=3327, help='The number of the samples')
    parser.add_argument('--alpha', type=float, default=1.0, help='Self supervised clustering parameter')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout')
    parser.add_argument('--init', default=72, type=int, help='Fix initial centroids')
    parser.add_argument('--gradient_clipping', default=5.0, type=float, help='Gradient clipping')

    return parser.parse_args()