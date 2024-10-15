# Score Forgetting Distillation for Stable Diffusion (SD)

This is the official repository for Score Forgetting Distillation (SFD) applied to DDPM. The code structure builds upon [SiD-LSG](https://github.com/mingyuanzhou/SiD-LSG).

## Environment setup
Install the dependencies into a `conda` environment:
```bash
conda env create -f environment.yml
conda activate sfd-sd
```

## Experiments

### 1. Prepare the Datasets

To finetune from a pretrained model and perform score forgetting distillation, you must supply the text prompts serving to preserve the original generative capability. By default, we use Aesthetic6+, but you have the option to choose Aesthetic6.25+, Aesthetic6.5+, or any list of prompts that does not include COCO captions. To obtain the Aesthetic6+ prompts from Hugging Face, follow their provided guidelines. Once you have prepared the Aesthetic6+ prompts, place them in 'data/aesthetics_6_plus/aesthetics_6_plus.txt'.

### 2. Download pretrained distilled checkpoint 

The distilled SDv1.5 checkpoint `batch512_cfg4.54.54.5_t625_8380_v2.pkl` can be found in [huggingface/UT-Austin-PML/SiD-LSG](https://huggingface.co/UT-Austin-PML/SiD-LSG/tree/main).

### 3. Training
   **Example 1: Celebrity Forgetting (e.g., Brad Pitt)**
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 BATCH_SIZE=8 BATCH_SIZE_PER_GPU=2 \
   SG_LEARNING_RATE=0.000003 G_LEARNING_RATE=0.000001 \
   FORGET_DATA_PROMPT_TEXT="prompts/brad_pitt_prompts.txt" \
   FORGET_DATA_PROMPT_TEXT_VAL="prompts/brad_pitt_prompts_val.txt" \
   CONCEPT_TO_FORGET="brad pitt" \
   CONCEPT_TO_OVERRIDE="a middle aged man" \
   bash run_df2_stage2.sh
   ```

   **Example 2: Nudity Forgetting**
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 BATCH_SIZE=8 BATCH_SIZE_PER_GPU=1 \
   SG_LEARNING_RATE=0.000003 G_LEARNING_RATE=0.000001 \
   FORGET_DATA_PROMPT_TEXT="prompts/12_subjects.txt" \
   FORGET_DATA_PROMPT_TEXT_VAL="prompts/nudity_prompts_val.txt" \
   CONCEPT_TO_FORGET="" \
   CONCEPT_TO_OVERRIDE="" \
   OVERRIDE_DATA_PROMPT_TEXT="" \
   POS_DATA_PROMPT_TEXT="prompts/nsfw_keywords.txt" \
   NEG_DATA_PROMPT_TEXT="prompts/nsfw_prompt.txt" \
   USE_NEG=1,1,1 \
   bash run_df2_stage2.sh
   ```

### Evaluations
1. Generate images for evaluation
    ```bash
    # generate 1,000 images from 50 brad pitt eval prompts
    torchrun \
      --standalone \
      --nproc_per_node 5 \
      eval_prompts.py \
        --network 'image_experiment/df2-stage2-train-runs/00000-aesthetics-text_cond-glr1e-06-lr3e-06-initsigma625-gpus8-alpha1.0-batch8-tmax980-fp16/network-snapshot-1.000000-000100.pkl' \
        --outdir 'image_experiment/df2-stage2-eval-runs/00000-aesthetics-text_cond-glr1e-06-lr3e-06-initsigma625-gpus8-alpha1.0-batch8-tmax980-fp16/giphy_eval_000100' \
        --seeds '0-19' \
        --batch 4 \
        --forget_data_prompt_text 'prompts/brad_pitt_prompts.txt'
    ```
2. Follow the corresponding evaluation protocols detailed in [selective-amnesia](https://github.com/clear-nus/selective-amnesia), [erasing](https://github.com/rohitgandikota/erasing), and [i2p](https://github.com/ml-research/i2p) repositories.

## License

This project uses the following license: Apache-2.0 license.
