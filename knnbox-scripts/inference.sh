DATA=koran
DATA_PATH=data-bin/${DATA}
MODEL_PATH=models/cmlmc_checkpoint.pt
SAVE_DIR=$SAVE_ROOT_DIR/cmlmc_${DATA}
DATASTORE_LOAD_PATH=datastore/vanilla/${DATA}_renew

python knnbox-scripts/common/generate.py $DATA_PATH \
    --task translation_lev \
    --path ${MODEL_PATH} \
    --dataset-impl mmap \
    --source-lang de --target-lang en \
    --batch-size 2 \
    --gen-subset test \
    --scoring sacrebleu \
    --tokenizer moses --remove-bpe \
    --arch vanilla_knn_mt@cmlmc_knn_wmt_en_de \
    --user-dir knnbox/models \
    --knn-mode inference \
    --knn-datastore-path  $DATASTORE_LOAD_PATH \
    --knn-max-k 32 \
    --knn-combiner-path $SAVE_DIR \
    --iter-decode-max-iter 10 \
    --iter-decode-eos-penalty 0 \
    --iter-decode-with-beam 5 ;