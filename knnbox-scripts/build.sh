DATA=koran
DATA_PATH=data-bin/${DATA}
MODEL_PATH=models/cmlmc_checkpoint.pt
DATASTORE_PATH=datastore/vanilla/${DATA}
python knnbox-scripts/common/validate.py $DATA_PATH \
    --task translation_lev \
    --path $MODEL_PATH \
    --valid-subset train \
    --batch-size 8 \
    --bpe fastbpe \
    --bpe-codes data-bin/ende30k.fastbpe.code \
    --user-dir knnbox/models \
    --arch vanilla_knn_mt@cmlmc_knn_wmt_en_de \
    --knn-mode build_datastore \
    --knn-datastore-path $DATASTORE_PATH \
    --mask-mode  random-mask-with-iteration-all \
    --random-datastore-iteration-times 5 ;