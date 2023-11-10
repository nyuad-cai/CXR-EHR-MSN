Pretraining with Masked Siamese Network Improves Robustness in Chest X-ray Image Classification
====================================

<!--ts-->
  * [Overview](#Overview)
  * [Environment setup](#Environment-setup)
  * [Dataset](#Dataset)
  * [CheXMSN training](#Model-training)
  * [Downstream evaluation](#Model-evaluation)
  * [Citation](#Citation)
   
 Overview
====================================
Transferability of deep neural networks for medical image classification in real-world set-
tings must account for robustness to missing information that may affect the clinical in-
terpretation of the underlying pathology, for example due to locally missing information
or image-level noise perturbations. Here, we perform the first generalizability and ro-
bustness analysis of self-supervised pretraining with Masked Siamese Network (MSN) for
transformer-based chest X-ray classification, denoted as CheXMSN. We develop and eval-
uate CheXMSN and other baselines using the publicly available dataset MIMIC-CXR. At
inference, we find that masking 70% of the input image leads to a 5.9% decrease in the
area under the receiver operating characteristic (AUROC) curve of CheXMSN, compared
to 8.5% with the masked auto-encoder, and 32.9% with SimSiam. Poisson noise and high
levels of Gaussian blur lead to 4.9% and 18.3% decrease in AUROC with CheXMSN and
SimSiam, respectively, while speckle and impulse noise lead to significant performance
degradation across all self-supervised methods. Overall, the work highlights the promise
of masked modeling and presents opportunities for future work to overcome shortcomings
of existing methods towards perturbations that may be more prevalent in other medical
modalities, such as ultrasound imaging or time-series data.

![image](assets/MMSN.png)

Environment-setup
-------------
Simply run the command below to install the necessary packages in your conda environment:
```
conda env create -f environment.yml
```

Dataset
-------------
We use [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) for all the experiments. For dataset processing, please refer to the [Medfuse](https://github.com/nyuad-cai/MedFuse) repository which explains how the chest X-ray dataset is processed and split into training, validation, and test sets. The MIMIC-CXR dataset is a restricted-access resource, which requires the user to undergo training and sign a data use agreement.

 CheXMSN training
 -------------
All experiment parameters are specified in argparse command-line-arguments.

Our implementation of pretraining of CheXMSN starts from the [ssl-trainer.py](ssl-trainer.py), which is called from the [ssl-trainer.py](slurm/job-scripts/ssl-trainer.sh). For example, use the command:
```
python ssl-trainer.py -m msn -d $DATA_DIR  -b 128 --pm True -w 24  --mr 0.30 --pn 1024 --lr 0.0001 --wd 0.001 --ngpus 1 --acc 'gpu' -e 100 -p 16 --esd 0.0001
```

Downstream evaluation
-------------
```
python downstream-trainer.py --tm 'transformer' -d $DATA_DIR --ckpt $CKPT_DIR --dp 1.0 -b 128 --pm True -w 24 --mr 0.15 -c 14 --lr 0.0001 --wd 0.001 -f False --ngpus 1 --acc 'gpu' -e 100 -p 32 --esd 0.0001
```


Evaluation against patch random dropping is in the [inference.ipynb](notebooks/inference.ipynb)


Citation
-------------
TODO

If you find this repository useful in your research, please consider giving a star :star: and a citation
