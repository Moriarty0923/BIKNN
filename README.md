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

The data we used in the paper can be found here (https://github.com/roeeaharoni/unsupervised-domain-clusters)



To facilitate the reproducing the result, you can download the cached datastore and generate code snippets by:
```bash
cd knnbox-scripts/knntranx_adaptive
# step 1. download datastores
bash get_datastore.sh
# step 2. inference
bash inference.sh
```

To generate the code, using the following command:
```bash
cd knnbox-scripts/knntranx_adaptive
# step 1. build datastore
bash build_datastore.sh
# step 2. train meta-k network
bash train_metak.sh
# step 3. inference
bash inference.sh
```

This code is used for producing all results in the paper. We will release a cleaner version of the code soon;