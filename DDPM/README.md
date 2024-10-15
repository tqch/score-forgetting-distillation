# Score Forgetting Distillation for Denoising Diffusion Probabilistic Models (DDPM)
This is the official repository for Score Forgetting Distillation (SFD) applied to DDPM. The code structure builds upon the following repositories:
- [DDIM](https://github.com/ermongroup/ddim)
- [SA](https://github.com/clear-nus/selective-amnesia/tree/a7a27ab573ba3be77af9e7aae4a3095da9b136ac/ddpm)
- [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency)

## Environment setup
Install the dependencies into a `conda` environment:
```
conda create --name sfd-ddpm python=3.8
conda activate sfd-ddpm
pip install -r requirements.txt
```

## Class forgetting with Score Forgetting Distillation

1. Train a conditional DDPM on all 10 CIFAR-10 or STL-10 classes.

   We provide an example to run SFD on CIFAR-10, which can be easily adapted to run STL-10 experiments by replacing config and dataset flags accordingly. For instance, the following command will prepare a pretrained conditional DDPM on CIFAR-10 using four GPUs with IDs of 0,1,2,3:
   ```bash
   # Pass visible GPUs to the training process through the environment variable, `CUDA_VISIBLE_DEVICES`.
   CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 --standalone --rdzv_backend c10d train.py --config cifar10_train.yml --mode train
   ```
   After finishing training, a checkpoint will appear at location `results/cifar10/yyyy_mm_dd_hhmmss/ckpts`. 

2. (Optional) Distill the conditional DDPM trained on all 10 classes

   This step is optional, and should only be included to reproduce the "SFD-Two Stage" results in the paper.
   ```bash
   CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 --standalone --rdzv_backend c10d train.py --config cifar10_sid.yml --init_ckpt results/cifar10/yyyy_mm_dd_hhmmss/ckpts/ckpt.pth --label_to_forget -1 --mode sfd --sample_type one_step --cond_scale 0
   ```
   During training, checkpoints will be saved to `results/cifar10/YYYY_MM_DD_HHMMSS/ckpts`.

3. Forget specified class with SFD

   Perform class-forgetting using SFD, finetuning either the pretrained model by Step 1 or the distilled model by Step 2. By default, we set the class to forget as 0 and the class to override as 1. 
   ```bash
   # SFD CIFAR-10
   CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 --standalone --rdzv_backend c10d train.py --config cifar10_sfd.yml --init_ckpt "results/cifar10/yyyy_mm_dd_hhmmss/ckpts/ckpt.pth" --label_to_forget 0 --mode sfd --sample_type one_step
   
   # SFD-Two Stage
   CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 --standalone --rdzv_backend c10d train.py --config cifar10_sfd_2stage.yml --init_ckpt "results/cifar10/YYYY_MM_DD_HHMMSS/ckpts/ckpt-49999.pth" --label_to_forget 0 --mode sfd --sample_type one_step
   ```

# Evaluation
1. Save images of classes to remember 
   ```bash
   python save_base_dataset.py --dataset cifar10 --label_to_forget -1 --num_samples_per_class 5000
   ```

2. Sample all classes using the finetuned checkpoint
   ```bash
   CUDA_VISIBLE_DEVICES="0,1" python sample.py --config results/cifar10/yyyy_mm_dd_hhmmss/logs/config.yaml --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss/ckpts/ckpt-49999.pth --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --cond_scale 0.0 --sample_type one_step
   ```

3. Run evaluator on generated images and real images
   ```bash
   LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py cifar10_without_label_0 ./results/cifar10/yyyy_mm_dd_hhmmss/fid_samples_guidance_0.0 "000_\d{5}\.png"
   # Example outputs:
   # Inception Score: 9.534753799438477
   # FID: 5.324725926494693
   # sFID: 7.696774859268089
   # Precision: 0.6587111111111111
   # Recall: 0.5470888888888888
   ```

4. Classifier Evaluation

    First fine-tune a pretrained ResNet34 classifier for CIFAR10
    ```bash
    CUDA_VISIBLE_DEVICES="0" python train_classifier.py --dataset cifar10 
    ```
    The classifier checkpoint will be saved as `cifar10_resnet34.pth`. 

   Then, evaluate the forgetting class with the trained classifier
    ```bash
   CUDA_VISIBLE_DEVICES="0,1" python ua_imagefolder.py --image_folder="results/cifar10/yyyy_mm_dd_hhmmss/fid_samples_guidance_0.0" --clf_path="cifar10_resnet34.pth" --config "results/cifar10/yyyy_mm_dd_hhmmss/logs/config.yaml"
   # Example outputs:
   # Average entropy: 0.15093368443194777
   # Average prob of forgotten class: 0.005658201234837179
   # Average accuracy of forgotten class: 0.0036000000000000008
   ```
