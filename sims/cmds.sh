#!/bin/bash

# iqtree2="/Users/ulises/Desktop/GOL/software/ext_modules/iqtree2"
tree_h1="/Users/ulises/Desktop/ABL/software/dissectingFactors/data/trees_alignments/flatfishes/h1_flat_mod_test.tree"
tree_h2="/Users/ulises/Desktop/ABL/software/dissectingFactors/data/trees_alignments/flatfishes/ml_tree_inter_taxa_properly_rooted.tree"

n_alns=2
ave_seq_len=461
lengths_file="seq_len_flat.txt"

python3 -c "import numpy as np; a = np.random.normal($ave_seq_len, $ave_seq_len/4, $n_alns); np.savetxt(\"$lengths_file\", a, fmt='%d')"

for ((i=1; i<=$n_alns; i++)); do

    seq_len=$(sed -n "${i}p" $lengths_file)
    iqtree2 --alisim "HKY_h1_"$i\
        -t $tree_h1 -m JC\
        --num-alignments 1\
        --out-format fasta\
        --length $seq_len

    iqtree2 --alisim "HKY_h2_"$i\
        -t $tree_h2 -m JC\
        --num-alignments 1\
        --out-format fasta\
        --length $seq_len
    # echo
    # echo "Reconstructing the tree for alignment $i"
    # echo
    iqtree2 -s "HKY_h1_"$i".fa" -m JC  --fast
    iqtree2 -s "HKY_h2_"$i".fa" -m JC  --fast
done

ggpy ggi HKY_h*1.fa -H myboth_hypos_woHKY.txt -e -E JC69
