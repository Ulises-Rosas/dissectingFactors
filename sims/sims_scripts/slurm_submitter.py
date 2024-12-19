import os
import argparse

def set_bash_script(i, offset, window, params_file, iqtree = "iqtree", 
                    random_len = 0, random_prunning = 0, twohypos = 0):
    
    mstr = """#!/bin/sh

#SBATCH --time=2-00:00:00
#SBATCH --mem=10G
#SBATCH --job-name={i}_sims_prota_ggi
#SBATCH --output={i}_sims_prota_ggi.out
#SBATCH --error={i}_sims_prota_ggi.err
#SBATCH --ntasks=15
#SBATCH --partition=normal

module load Miniconda3
conda activate py3.7

./pipe_sim.sh {offset} {window} {params_file} {iqtree} {random_len} {random_prunning} {twohypos}

""".format(i = i, offset = offset, window = window, params_file = params_file,
            iqtree = iqtree, random_len = random_len, random_prunning = random_prunning, 
            twohypos = twohypos)
    return mstr


def args_prota():
    parser = argparse.ArgumentParser(description='Process slurm files')
    parser.add_argument('--n', type=int, help='number of iterations')
    parser.add_argument('--window', type=int, help='window size')
    parser.add_argument('--params', type=str, help='params file')
    parser.add_argument('--iqtree', type=str, help='path to iqtree')
    parser.add_argument('--random_len', type=str, help='random length')
    parser.add_argument('--random_prunning', type=str, help='random prunning')
    parser.add_argument('--twohypos', type=str, help='twohypos')
    args = parser.parse_args()
    return args

def main():
    args = args_prota()
    n = args.n
    # window = 30
    for i in range(0, n+1):
        with open(f"./sim_iter_ggi{i}.sh", "w") as f:
            f.write(set_bash_script(
                i, 
                offset=i*args.window, 
                window=args.window,
                params_file     = os.path.abspath(args.params),
                iqtree          = os.path.abspath(args.iqtree),
                random_len      = os.path.abspath(args.random_len),
                random_prunning = os.path.abspath(args.random_prunning),
                twohypos        = os.path.abspath(args.twohypos)
            ))
        print(f"sbatch sim_iter_ggi{i}.sh")
    
if __name__ == "__main__":
    main()
