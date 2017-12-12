#MDL='intel_noatt'
#MDL='intel_att'
MDL='intel_glove'
#export CUDA_VISIBLE_DEVICES=3 

# Training sequence-to-sequence model
python -m mt.model \
    --src=src --tgt=tgt \
    --vocab_prefix=$DATA/vocab  \
    --train_prefix=$DATA/train \
    --dev_prefix=$DATA/dev  \
    --test_prefix="$DATA/test" \
    --out_dir=../results/$MDL \
    --num_train_steps=1000 \
    --steps_per_stats=10 \
    --num_layers=1 \
    --num_units=200 \
    --dropout=0.4 \
	--share_vocab \
    --embed_prefix="/mnt/brain2/scratch/llajan/word_vectors/glove.840B.300d.txt" \
    --metrics=bleu
#	--attention="" \


# Inference using sequence-to-sequence model
python -m mt.model \
    --vocab_prefix=$DATA/vocab  \
    --num_layers=1 \
    --num_units=200 \
    --dropout=0.4 \
	--share_vocab \
	--beam_width=0 \
	--random_seed=20 \
    --out_dir=/home/llajan/errata/errata_analysis_eecs573/results/intel_noatt \
    --inference_input_file=/home/llajan/errata/nmt_data/test.src \
	--length_penalty_weight=5.0 \
    --inference_output_file=/home/llajan/errata/errata_analysis_eecs573/results/intel_noatt/test2

