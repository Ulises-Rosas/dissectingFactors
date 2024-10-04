#!/bin/bash

iqtree2="/Users/ulisesrosas/Desktop/iqtree-2.3.6-macOS/bin/iqtree2"
tree="/Users/ulisesrosas/Desktop/dissectingFactors/data/trees_alignments/flatfishes/h1_flat_mod_test.tree"

n_alns=100
ave_seq_len=461
lengths_file="seq_len_flat.txt"

python3 -c "import numpy as np; a = np.random.normal($ave_seq_len, $ave_seq_len/4, $n_alns); np.savetxt(\"$lengths_file\", a, fmt='%d')"
for i in {1..$n_alns}; do
    seq_len=$(sed -n "${i}p" $lengths_file)
    
    $iqtree2 --alisim "HKY_h1_"$i -t $tree -m JC --num-alignments 1 --out-format fasta --length $seq_len
    
done

# if [ $seq_len -gt 0 ]; then# fi