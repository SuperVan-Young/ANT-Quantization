import os
import subprocess
import time
from multiprocessing import Pool
from itertools import product

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, 'log')
os.makedirs(LOG_DIR, exist_ok=True)

NUM_GPUS = 4

def run_experiment(args):
    """ Run single experiment and record log.
    """

    # configure command line
    cmd = f"python -u -m torch.distributed.launch --nproc_per_node=1 --master_port {os.environ['pytorch_dist_master_port']} main.py"
    for arg_name, arg_val in args.items():
        cmd += f" --{arg_name}={arg_val}" if arg_val else f" --{arg_name}"
    print(f"Running command: {cmd}")

    # configure log file
    def get_log_filename(args):
        model = args['model']
        mode = args['mode']

        w_opt_target = args['w_opt_target']
        a_opt_target = args['a_opt_target']
        w_opt_metric = args['w_opt_metric']
        a_opt_metric = args['a_opt_metric']
        
        opt =  f"W-{w_opt_target}-{w_opt_metric}-A-{a_opt_target}-{a_opt_metric}"

        log_filename = f"M-{model}-D-{mode}-{opt}.log"

        return log_filename

    log_filename = get_log_filename(args)
    log_filepath = os.path.join(LOG_DIR, log_filename)
    
    # run experiment
    start_time = time.time()
    
    with open(log_filepath, 'w') as log_file:
        subprocess.run(cmd.split(), stdout=log_file, stderr=log_file)

    end_time = time.time()
    
    with open(log_filepath, 'a') as log_file:
        log_file.write(f"\nTotal running time: {end_time - start_time} seconds\n")
    
    print(f"Finished experiment {log_filename}")

def run_all_experiments(args_list):
    """ Run all experiments with process pool.
    """
    
    def pool_initializer():
        """ Make sure each worker has an exclusive GPU.
        """
        pid = os.getpid()  # workers are spawned with consecutive pids
        gpu_id = pid % NUM_GPUS
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpu_id}"
        os.environ['pytorch_dist_master_port'] = f"{46666 + gpu_id}"

    with Pool(processes=NUM_GPUS, initializer=pool_initializer) as pool:
        pool.map(run_experiment, args_list)
        pool.close()
        pool.join()

def create_args_list():
    arch_list = [
        'resnet18',
        'resnet50',
        'vgg16',
        'inception_v3',
        # 'vit_b_16',
    ]
    datatype_list = [
        'int',
        # 'ant-int-pot',
        # 'ant-int-pot-float',
        # 'ant-int-pot-flint',
        # 'ant-int-pot-float-flint',
    ]
    opts = {
        ('tensor', 'mse', 'tensor', 'mse'),
        ('output', 'mse', 'output', 'mse'),
        ('output', 'fisher_diag', 'output', 'fisher_diag'),
        ('activated_output', 'mse', 'activated_output', 'mse'),
    }

    default_args = {
        'ptq': None,
        'dataset': 'imagenet',
        'calib_size': 1024,
        'wbit': 4,
        'abit': 4,
        'w_low': 1,
        'a_low': 1,
        'w_up': 105,
        'a_up': 105,
    }

    args_list = []

    for model, mode, opt in product(arch_list, datatype_list, opts):
        args = {
            'model': model,
            'mode': mode,
            'w_opt_target': opt[0],
            'w_opt_metric': opt[1],
            'a_opt_target': opt[2],
            'a_opt_metric': opt[3],
        }
        args.update(default_args)

        args_list.append(args)
    
    return args_list

def main():
    run_all_experiments(create_args_list())

if __name__ == '__main__':
    main()