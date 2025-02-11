# Text-to-Image Synthesis using DF-GAN

This repository contains the implementation of **DF-GAN (Deep Fusion Generative Adversarial Network)** for **text-to-image synthesis**, where textual descriptions are transformed into **high-resolution and semantically relevant images**. The model is trained on the **CUB-200-2011** dataset and achieves **state-of-the-art** performance in realism and diversity. DF-GAN eliminates the need for stacked generators by using a **one-stage text-to-image backbone** with components like **Deep Fusion Blocks (DFBlock), Target-Aware Discriminator, and Matching-Aware Gradient Penalty (MA-GP)** for efficient and consistent image synthesis.

---

## Requirements
- Python 3.8
- PyTorch 1.9
- At least **1x12GB NVIDIA GPU**

## Installation
Unzip the zip file into a folder named **DF-GAN**. Once that is done, run the following commands to install the required libraries:
```bash
pip install -r requirements.txt
cd DF-GAN/code/
```

## Preparation

### Datasets
1. Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) and extract them to `data/`
2. Download the [CUB-200-2011 bird dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and extract them to `data/birds/`

---

## Training

### Train the DF-GAN model
To train the **DF-GAN** model on the **birds** dataset, run the following command:
```bash
bash scripts/train.sh ./cfg/bird.yml
```

---

## Evaluation

### Evaluate DF-GAN models
To evaluate the model, synthesize **3W images from test descriptions** and compute the **FID score** between the **synthesized images** and **test images**. 

1. Locate the trained DF-GAN model in the latest directory inside `/saved_models/`
2. Update the `'checkpoint'` field in `./cfg/bird.yml` to the path of the trained model
3. Run the following command:
```bash
bash scripts/calc_FID.sh ./cfg/bird.yml
```
For **Inception Score (IS)** computation, use [StackGAN-inception-model](https://github.com/hanzhanggit/StackGAN-inception-model).

---

## Sampling

### Synthesize images from example captions
Run the following command to generate images from the provided example captions:
```bash
bash scripts/sample.sh ./cfg/bird.yml
```

### Synthesize images from custom text descriptions
If you want to generate images based on your own text descriptions:
1. Replace your text descriptions in `./code/example_captions/dataset_name.txt`
2. Run the same command as above:
```bash
bash scripts/sample.sh ./cfg/bird.yml
```
The synthesized images will be saved in `./code/samples/`

---

---

## Results & Insights

- **Competitive Performance with Fewer Parameters:**  
  DF-GAN achieves **state-of-the-art** results while using **fewer parameters (NoP: 19M)** compared to other advanced models.  

- **Inception Score (IS) & Frechet Inception Distance (FID) Improvements:**  
  - **Compared to AttnGAN:** IS improved from **4.36 → 5.10**, and FID reduced from **23.98 → 14.90**.  
  - **Compared to DM-GAN:** IS increased from **4.75 → 5.0**, and FID decreased from **16.09 → 14.90**.  

- **Outperforms Models with Extra Supervision:**  
  DF-GAN competes with **DAE-GAN, CPGAN, XMC-GAN, and SD-GAN**, despite not relying on extra pre-trained models like **YOLO-V3, VGG-19, or BERT**.  

### 📊 **Performance Comparison on the CUB Dataset**
| Model        | IS ↑  | FID ↓  | NoP ↓  |
|-------------|------|------|------|
| StackGAN    | 3.70  | -    | -    |
| StackGAN++  | 3.84  | -    | -    |
| AttnGAN     | 4.36  | 23.98 | 230M  |
| DM-GAN      | 4.75  | 16.09 | 46M   |
| DAE-GAN     | 4.42  | 15.19 | 98M   |
| TIME        | 4.91  | 14.30 | 120M  |
| **DF-GAN**  | **~5.00**  | **~14.90**  | **19M**  |

- **Simplicity & Efficiency:**  
  Unlike stacked architectures, DF-GAN directly generates **256×256** images **without feature entanglements**, making it more efficient and effective.

---

### 📌 **Contributors**
Feel free to contribute to this repository by opening issues or submitting pull requests.
