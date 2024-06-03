KNN_NUMBER=16       # for koran
lam=0.7
TEMP=100
DATA=koran
DATA_PATH=data-bin/${DATA}
MODEL_PATH=models/cmlmc_checkpoint.pt
DATASTORE_PATH=datastore/vanilla/${DATA}

python knnbox-scripts/common/validate.py $DATA_PATH \
    --path $MODEL_PATH \
    --task translation_lev \
    --valid-subset train \
    --skip-invalid-size-inputs-valid-test \
    --batch-size 8 \
    --bpe fastbpe \
    --user-dir knnbox/models \
    --arch "vanilla_knn_mt@cmlmc_knn_wmt_en_de" \
    --knn-mode "renew_datastore" \
    --knn-datastore-path $DATASTORE_PATH \
    --new-datastore $NEW_DATASTORE \
    --mask-mode random-mask \
    --num-workers 0 \
    --knn-k $KNN_NUMBER \
    --knn-lambda ${lam} \
    --knn-temperature ${TEMP} \
    --del-wrong True ;