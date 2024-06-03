DATA=koran
DATA_PATH=data-bin/${DATA}
MODEL_PATH=models/cmlmc_checkpoint.pt
SAVE_DIR=$SAVE_ROOT_DIR/cmlmc_${DATA}
DATASTORE_LOAD_PATH=datastore/vanilla/${DATA}_renew
python fairseq_cli/train.py $DATA_PATH \
    --task translation_lev \
    --train-subset train --valid-subset valid \
    --best-checkpoint-metric "loss" \
    --finetune-from-model ${MODEL_PATH} \
    --insertCausalSelfAttn \
    --concatPE \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --share-all-embeddings \
    --no-scale-embedding \
    --source-lang de --target-lang en \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 0.1 \
    --lr $LR --lr-scheduler reduce_lr_on_plateau \
    --lr-patience 5 --lr-shrink 0.5 --patience ${PATIENCE} \
    --min-lr 1e-5 --label-smoothing 0.001 \
    --dropout 0.1 \
    --max-epoch 5000 --max-update 15000 --validate-after-updates 100 \
    --criterion label_smoothed_cross_entropy_for_cmlmc \
    --log-interval 10 \
    --save-interval-updates 100 \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --tensorboard-logdir $SAVE_DIR/log \
    --save-dir $SAVE_DIR \
    --batch-size 12 \
    --user-dir knnbox/models \
    --arch "vanilla_knn_mt@cmlmc_knn_wmt_en_de" \
    --knn-mode "train_metak" \
    --knn-datastore-path $DATASTORE_LOAD_PATH \
    --knn-max-k 32 \
    --knn-combiner-path $SAVE_DIR \
    --robust-wp-hidden-size 512 \
    --robust-dc-hidden-size 64 \
    --mask-mode random-mask \
    --num-workers 0 ;