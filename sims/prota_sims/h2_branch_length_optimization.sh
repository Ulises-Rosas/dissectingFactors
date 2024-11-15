aln='/Users/ulisesrosas/Desktop/dissectingFactors/sims/prota_sims/mysupermatrix.txt'
partitions='/Users/ulisesrosas/Desktop/dissectingFactors/sims/prota_sims/mypartitions.txt'
twohypos='/Users/ulisesrosas/Desktop/dissectingFactors/sims/prota_sims/prota_myboth_hypos.txt'
iqtree2='/Users/ulisesrosas/Desktop/iqtree-2.3.6-macOS/bin/iqtree2'


sed -n 2p $twohypos > tree_h2.txt

# error at 9558
$iqtree2 -s $aln -p $partitions -g tree_h2.txt --prefix h2_constrained -T 15

