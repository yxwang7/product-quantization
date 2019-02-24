# Ground truth
# python run_ground_truth.py  --dataset netflix --topk 20 --metric euclid_norm

# Run PQ
# python3 run_pq.py --dataset netflix --topk 20 --metric euclid_norm --num_codebook 2 --Ks 256

# Run MPQ
python3 run_mpq.py --dataset netflix --topk 20 --metric euclid_norm --num_codebook 2 --Ks 256 --num_table 128