import argparse
import multiprocessing

def config():
    p = argparse.ArgumentParser()
    p.add_argument('--n_latent', type=int, default=120,
                   help='Latent variable dimension.')
    p.add_argument('--log_dir', type=str, default='logs',
                   help='Tensorboard directory name.')
    p.add_argument('--result_dir', type=str, default='result',
                   help='Network models saving directory name.')
    p.add_argument('--beta1', type=float, default=0)
    p.add_argument('--beta2', type=float, default=0.99)
    p.add_argument('--eps', type=float, default=1e-8)
    p.add_argument('--gpu', nargs='*', type=int, required=True, help='GPU number.')
    p.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help='Number of CPU worker.')
    p.add_argument('--n_img', type=int, default=15000)
    p.add_argument('-n', '--n_gen_img', type=int, default=10,
                   help='Number of generation images by Generater when test time.')
    p.add_argument('-l', '--loss', type=str, default='wgan-gp')
    p.add_argument('--gen', type=str, default='gen')
    p.add_argument('--dis', type=str, default='dis')
    p.add_argument('--depth', type=int, default=9)
    p.add_argument('--data_path', type=str, default='/root/work/Datasets/CelebA_1024')
    p.add_argument('--txt_path', type=str, default='/root/work/Datasets/CelebA-HQ/img_attributes_list.txt')
    p.add_argument('--n_channel', type=int, default=64)
    p.add_argument('--n_ch_g', type=int, default=512)
    p.add_argument('--n_ch_d', type=int, default=16)
    p.add_argument('--last_img_size', type=int, default=1024)
    p.add_argument('--switch_timing', type=int, default=4)
    p.add_argument('--n_style', type=int, default=512)
    p.add_argument('--n_mapping_network', type=int, default=8)
    p.add_argument('--alpha', type=float, default=0.0)
    p.add_argument('--lambda_gp', type=int, default=10)
    p.add_argument('--lambda_r1', type=int, default=10)
    p.add_argument('--delta', type=float, default=5e-5)
    p.add_argument('--yaml_path', type=str, default='/root/work/mprg_lectures/StyleGAN/params.yml')
    p.add_argument('--lambda_drift', type=float, default=1e-3)
    p.add_argument('--G_opt', type=str, default='gen_opt')
    p.add_argument('--D_opt', type=str, default='dis_opt')
    p.add_argument('--use_pretrain', action='store_true')
    return p.parse_args()