#!/bin/bash

# path=""

# iqtree2="/Users/ulisesrosas/Desktop/iqtree-2.3.6-macOS/bin/iqtree2"
# random_len="/Users/ulisesrosas/Desktop/dissectingFactors/sims/sim_files/random_len.py"
# random_prunning="/Users/ulisesrosas/Desktop/dissectingFactors/sims/sim_files/random_prunning.py"


# twohypos="/Users/ulisesrosas/Desktop/dissectingFactors/sims/sim_files/prota_myboth_hypos_v2.txt"
# param_file="/Users/ulisesrosas/Desktop/dissectingFactors/sims/sim_files/prota_params.txt"

from=$1
n_alns=$2
param_file=$3


iqtree2=$4
random_len=$5
random_prunning=$6
twohypos=$7


mkdir -p "sim_${from}"
cd "sim_${from}"


ave_seq_len=461
var_seq_len=10

prob=1
prop=1
M=MG{2.0}
seed=12038


k=0
for ((i=1; i<=$n_alns; i++)); do
    # i=1
    boot=$(($i + $from))
    seed=$(($seed + $boot))

    pruned_i="pruned_${boot}"
    echo 'Seed:' $seed

    python3.7 $random_prunning $twohypos --prop $prop\
            --prob $prob --outname $pruned_i\
            --seed $seed

    if [ $k == 0 ]; then
        sed -n '1p' $pruned_i > 'h_'$i
        param_i=$(sed -n '1p' $param_file)
        k=$(($k + 1))
    else
        sed -n '2p' $pruned_i > 'h_'$i
        param_i=$(sed -n '2p' $param_file)
        k=0
    fi

    aln_i="indel_${boot}"
    len_i=$(python3.7 $random_len --n 1 --mean $ave_seq_len --var $var_seq_len --seed $seed)

    # aln_i
    $iqtree2 --alisim $aln_i\
             -t 'h_'$i -m $M\
             --num-alignments 1\
             --out-format fasta\
             --length $len_i\
             --indel $param_i\
             --seed $seed

    # tree_i
    $iqtree2 -s $aln_i".fa" -st CODON -m $M -T 15 --ufboot 1000\
             --prefix $aln_i"_gt" --seed $seed
    echo "Unconstrained tree for aln $i done"

    # constrained trees
    for((j=1; j<=2; j++)); do
        sed -n $j'p' $pruned_i > "tmp_p_${j}"
        # delete later the -n 0
        $iqtree2 -s $aln_i".fa" -st CODON -m $M -T 15\
                 -g "tmp_p_${j}" --prefix "${aln_i}.cons_${j}"\
                 --seed $seed
    done 

    all_cons="cons_${i}_trees.txt"    
    cat "${aln_i}.cons_"*".treefile" > $all_cons 
    echo "Constrained trees for aln $i done"

    $iqtree2 -s $aln_i'.fa' -st CODON -m $M -z $all_cons -n 0\
     -zb 10000 -au -T 15 --prefix "${aln_i}_au" --seed $seed

    rm -f *.ckp.gz
    rm -f *.bionj
    rm -f *.mldist
    rm -f *.log
    rm -f *.uniqueseq.phy
    rm -f *.reduced
    # rm -f *.iqtree
    rm -f *.best_model.nex
    rm -f *.splits.nex
    rm -f *.parstree
    rm -f *.contree
    rm -f *.unaligned.fa

    echo "Done with boot $boot"
done


