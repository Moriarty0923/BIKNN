# Biknn
Code for Efficient Domain Adaptation for Non-Autoregressive Machine Translation

## Requirements and Installation
* python >= 3.7
* pytorch >= 1.10.0
* faiss-gpu >= 1.7.3
* sacremoses == 0.0.41
* sacrebleu == 1.5.1
* fastBPE == 0.1.0
* scikit-learn >= 1.0.2
* seaborn >= 0.12.1
* editdistance >= 0.8.1
* elasticsearch >= 8.13.1


You can install this toolkit by
```shell
cd Biknn
pip install --editable ./
```

Note: Installing faiss with pip is not suggested. For stability, we recommand you to install faiss with conda

```bash
CPU version only:
conda install faiss-cpu -c pytorch

GPU version:
conda install faiss-gpu -c pytorch # For CUDA
```

## Data
The data we used in the paper can be found here [multi-domain de-en dataset](https://github.com/roeeaharoni/unsupervised-domain-clusters)

WMT19 data can be found [wmt19](https://github.com/facebookresearch/fairseq/blob/main/examples/wmt19/README.md)


## Get Datastore and Combiner
You can download the cached datastore and trained combiner at:

[HuggingFace](https://huggingface.co/Moriarty0923/Biknn)
```bash
# inference 
bash knnbox-scripts/inference.sh
```
You can download the datastore and pre-trained combiner and put them in the according dir, change the path in the script to your own path.

To generate the code, using the following command:
```bash
cd knnbox-scripts
# step 1. build datastore
bash build.sh
# step 2. renew datastore
bash renew.sh
# step 3. train metanetwork
bash train_metanetwork
# step 4. inference 
bash inference.sh
```


The code is primarily implemented through [knn-mt](https://github.com/urvashik/knnmt) and [knnbox](https://github.com/NJUNLP/knn-box).
We will release a cleaner version in future;
