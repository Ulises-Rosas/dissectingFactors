#!/usr/bin/env python3

import os
import argparse
from collections import deque

# pattern='/Users/ulisesrosas/Desktop/dissectingFactors/sims/sim_0_progress/*.fa.iqtree'
# files = glob.glob(pattern)
# file = files[0]

def process_files(files, out_file):

    out = deque()
    for file in files:
        file_lines = deque()
        with open(file, 'r') as f:
            k = 0 
            for i in f.readlines():
                line = i.strip()

                if line.endswith('  p-AU'):
                    # print(line)
                    # file_lines.append(line)
                    k += 1
                    continue

                if 0 < k < 3:
                    if line.startswith('-'):
                        continue   
                    # print(line)                
                    file_lines.append(line)
                    k += 1
        if k == 0:
            print(f"File {file} has no p-values")
            continue

        file_name = os.path.basename(file)
        p_vals = [file_name]*3
        r = 1
        while file_lines:
            line = file_lines.popleft()
            # line = "1 -8249.059793       0       1 +      1 +      1 +         1 +        1 +"
            line = line.split()
            p_vals[r] = round(float(line[len(line)-2]), 4)
            r += 1
        
        out.append(p_vals)

    with open(out_file, 'w') as f:

        while out:
            i = out.popleft()
            f.write('\t'.join(map(str, i)) + '\n')

def args_prota():
    parser = argparse.ArgumentParser(description='Process iqtree files.')
    parser.add_argument('files', type=str, nargs='+', help='iqtree files to process')
    parser.add_argument('-o','--output', type=str, help='output file', default=None, required=True)
    args = parser.parse_args()
    return args


def main():
    args = args_prota()
    process_files(args.files, args.output)


if __name__ == "__main__":
    main()
