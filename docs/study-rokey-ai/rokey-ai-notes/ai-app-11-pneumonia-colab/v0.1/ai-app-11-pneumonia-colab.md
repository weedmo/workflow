# 11장 의료 영상 AI

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/intermediate/ai-app-11-pneumonia-colab.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, intermediate, study  
> source: raw/2026-09-26-rokey-ai/intermediate/강의_3기_AI응용_11차시__Pneumonia__colab.ipynb  
> source: https://github.com/weedmo/Rokey-AI  
- Chest X-RAY 폐렴 분류


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive


## Introduction
In this notebook we will preprocess the data for our classification task.<br />
We will train a classifier to predict whether an X-Ray of a patient shows signs of pneumonia or not based on the RSNA Pneumonia Detection Challenge (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).

At first we download the data from kaggle (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data), by clicking on "Download All" and extract it afterwards.

Acknowledgements:
Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR 2017, http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf

Original Source: https://nihcc.app.box.com/v/ChestXray-NIHCC

## Imports
* pathlib for convenient path handling
* pydicom for reading dicom files
* numpy for storing the actual images
* cv2 for directly resizing the images
* pandas to read the provided labels
* matplotlib for visualizing some images
* tqdm for nice progress bar


```python
!pip install -Uqq pydicom ipdb torchmetrics pytorch_lightning
```

    [?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/2.4 MB[0m [31m?[0m eta [36m-:--:--[0m
[2K   [91m━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.4/2.4 MB[0m [31m10.9 MB/s[0m eta [36m0:00:01[0m
[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m [32m2.4/2.4 MB[0m [31m37.8 MB/s[0m eta [36m0:00:01[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m29.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m927.3/927.3 kB[0m [31m51.8 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m819.3/819.3 kB[0m [31m52.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m363.4/363.4 MB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.8/13.8 MB[0m [31m113.6 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.6/24.6 MB[0m [31m89.6 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m883.7/883.7 kB[0m [31m54.7 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m664.8/664.8 MB[0m [31m2.1 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m211.5/211.5 MB[0m [31m4.7 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.3/56.3 MB[0m [31m42.5 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m127.9/127.9 MB[0m [31m19.1 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m207.5/207.5 MB[0m [31m4.9 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.1/21.1 MB[0m [31m97.2 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m78.8 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import os
import glob
import ipdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import pydicom
import cv2
from tqdm.notebook import tqdm
```

## Preprocessing

### Read the csv file containing the labels


```python
root = "/content/drive/MyDrive/AI_application/pn_data_n1000/"
# root = "./pn_data_n1000/"

labels = pd.read_csv(root + "stage_2_train_labels.csv")
print("shape = ", labels.shape)
print(labels.head(6))
```

    shape =  (30227, 6)
                                  patientId      x      y  width  height  Target
    0  0004cfab-14fd-4e49-80ba-63a80b6bddd6    NaN    NaN    NaN     NaN       0
    1  00313ee0-9eaa-42f4-b0ab-c148ed3241cd    NaN    NaN    NaN     NaN       0
    2  00322d4d-1c29-4943-afc9-b6754be640eb    NaN    NaN    NaN     NaN       0
    3  003d8fa0-6bf1-40ed-b54c-ac657f8495c5    NaN    NaN    NaN     NaN       0
    4  00436515-870c-4b36-a041-de91049b9ab4  264.0  152.0  213.0   379.0       1
    5  00436515-870c-4b36-a041-de91049b9ab4  562.0  152.0  256.0   453.0       1


Note that subjects may occur multiple times in the dataset because different pneumonia spots are handled indivually. For our classification task, we can remove those duplicates as we are only interested in the binary label.


```python
# Remove duplicate entries
labels = labels.drop_duplicates("patientId")
print("shape = ", labels.shape)
labels.head()

```

    shape =  (26684, 6)






  <div id="df-5a83f2c4-efee-46cc-901a-0acbd856a240" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patientId</th>
      <th>x</th>
      <th>y</th>
      <th>width</th>
      <th>height</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0004cfab-14fd-4e49-80ba-63a80b6bddd6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00313ee0-9eaa-42f4-b0ab-c148ed3241cd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00322d4d-1c29-4943-afc9-b6754be640eb</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>003d8fa0-6bf1-40ed-b54c-ac657f8495c5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00436515-870c-4b36-a041-de91049b9ab4</td>
      <td>264.0</td>
      <td>152.0</td>
      <td>213.0</td>
      <td>379.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5a83f2c4-efee-46cc-901a-0acbd856a240')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5a83f2c4-efee-46cc-901a-0acbd856a240 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5a83f2c4-efee-46cc-901a-0acbd856a240');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f47fa1cd-12c6-4720-a990-e2b666c52b7f">
  <button class="colab-df-quickchart" onclick="quickchart('df-f47fa1cd-12c6-4720-a990-e2b666c52b7f')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f47fa1cd-12c6-4720-a990-e2b666c52b7f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Let's define the path to the dicom files and also the path were we want to store our processed npy files


```python
ROOT_PATH = Path(root + "stage_2_train_images")
SAVE_PATH = Path(root + "Processed")
print(ROOT_PATH)
print(SAVE_PATH)
```

    /content/drive/MyDrive/AI_application/pn_data_n1000/stage_2_train_images
    /content/drive/MyDrive/AI_application/pn_data_n1000/Processed


### Chest X-ray images


```python

c = 0
image_list = []
target_list = []
while len(image_list) <= 8:
    patient_id = labels.patientId.iloc[c]
    dcm_path = ROOT_PATH/patient_id
    dcm_path = dcm_path.with_suffix(".dcm")
    try:
        dcm = pydicom.dcmread(dcm_path).pixel_array
        image_list.append(dcm_path)
        target_list.append(labels.Target.iloc[c])
        c += 1
    except:
        c +=1
print("image list = \n", image_list)
print("target_list = \n", target_list)


fig, axis = plt.subplots(3, 3, figsize=(9, 9))
for i in range(9):
    row, col = divmod(i, 3)
    dcm = pydicom.dcmread(image_list[i]).pixel_array

    label = target_list[i]

    axis[row, col].imshow(dcm, cmap="bone")
    axis[row, col].set_title(label)

```

    image list = 
     [PosixPath('/content/drive/MyDrive/AI_application/pn_data_n1000/stage_2_train_images/00a85be6-6eb0-421d-8acf-ff2dc0007e8a.dcm'), PosixPath('/content/drive/MyDrive/AI_application/pn_data_n1000/stage_2_train_images/00aecb01-a116-45a2-956c-08d2fa55433f.dcm'), PosixPath('/content/drive/MyDrive/AI_application/pn_data_n1000/stage_2_train_images/00c0b293-48e7-4e16-ac76-9269ba535a62.dcm'), PosixPath('/content/drive/MyDrive/AI_application/pn_data_n1000/stage_2_train_images/00d7c36e-3cdf-4df6-ac03-6c30cdc8e85b.dcm'), PosixPath('/content/drive/MyDrive/AI_application/pn_data_n1000/stage_2_train_images/00f08de1-517e-4652-a04f-d1dc9ee48593.dcm'), PosixPath('/content/drive/MyDrive/AI_application/pn_data_n1000/stage_2_train_images/00f87de5-5fe0-4921-93ea-914d7e683266.dcm'), PosixPath('/content/drive/MyDrive/AI_application/pn_data_n1000/stage_2_train_images/01a4059c-22f7-4f51-8a27-50aff0b3aeb3.dcm'), PosixPath('/content/drive/MyDrive/AI_application/pn_data_n1000/stage_2_train_images/01a5594f-e5d4-4f7a-b79d-3f57559fe37b.dcm'), PosixPath('/content/drive/MyDrive/AI_application/pn_data_n1000/stage_2_train_images/01a6eaa6-222f-4ea8-9874-bbd89dc1a1ce.dcm')]
    target_list = 
     [0, 1, 1, 0, 1, 0, 0, 0, 1]



    
*(이미지 없음 / image not preserved: output_14_1.png)*
    


### Dicom Reading & Effective storage

In order to efficiently handle our data in the Dataloader, we convert the X-Ray images stored in the DICOM format to numpy arrays. Afterwards we compute the overall mean and standard deviation of the pixels of the whole dataset, for the purpose of normalization.
Then the created numpy images are stored in two separate folders according to their binary label:
* 0: All X-Rays which do not show signs of pneumonia
* 1: All X-Rays which show signs of pneumonia

To do so, we iterate over the patient ids and concat the patient ID with the ROOT_PATH.

We then directly save the standardized and resized files into the corresponding directory (0 for healthy, 1 for pneumonia).
This allows to take advantage of the ready-to-use torchvision **DatasetFolder** for simple file reading


We standardize all images by the maximum pixel value in the provided dataset, 255.
All images are resized to 224x224.

To compute dataset mean and standard deviation, we compute the sum of the pixel values as well as the sum of the squared pixel values for each subject.
This allows to compute the overall mean and standard deviation without keeping the whole dataset in memory.



```python
ROOT_PATH
```




    PosixPath('/content/drive/MyDrive/AI_application/pn_data_n1000/stage_2_train_images')




```python
lentrain = len(os.listdir(ROOT_PATH)) # ROOT_PATH = pn_data_n1000\stage_2_train_images
print("train number = ", lentrain)

lentest = len(os.listdir(root + "/stage_2_test_images/"))
print("test number = ", lentest)
```

    train number =  1000
    test number =  100



```python
lentrain = 800
```


```python
import os
sums = 0
sums_squared = 0
count = 0
# ipdb.set_trace()
for c, patient_id in enumerate(tqdm(labels.patientId)):
    dcm_path = ROOT_PATH/patient_id  # Create the path to the dcm file
    dcm_path = dcm_path.with_suffix(".dcm")  # And add the .dcm suffix

    # if not os.path.isdir(dcm_path):
    #   print("I do not have this file")
    # Read the dicom file with pydicom and standardize the array
    if os.path.exists(dcm_path):
      dcm = pydicom.dcmread(dcm_path).pixel_array / 255

      # Resize the image as 1024x1024 is way to large to be handeled by Deep Learning models at the moment
      # Let's use a shape of 224x224
      # In order to use less space when storing the image we convert it to float16
      dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)

      # Retrieve the corresponding label
      label = labels.Target.iloc[c]

      # 4/5 train split, 1/5 val split
      train_or_val = "train" if count < lentrain else "val"

      current_save_path = SAVE_PATH/train_or_val/str(label) # Define save path and create if necessary
      current_save_path.mkdir(parents=True, exist_ok=True)
      np.save(current_save_path/patient_id, dcm_array)  # Save the array in the corresponding directory

      normalizer = dcm_array.shape[0] * dcm_array.shape[1]  # Normalize sum of image
      # if train_or_val == "train":  # Only use train data to compute dataset statistics
      #     sums += np.sum(dcm_array) / normalizer
      #     sums_squared += (np.power(dcm_array, 2).sum()) / normalizer
      count += 1

```


      0%|          | 0/26684 [00:00<?, ?it/s]



```python
print(len(os.listdir(root + "Processed/train/0")))
print(len(os.listdir(root + "Processed/train/1")))

print(len(os.listdir(root + "Processed/val/0")))
print(len(os.listdir(root + "Processed/val/1")))
```

    527
    273
    142
    58



```python
SAVE_PATH
```




    PosixPath('/content/drive/MyDrive/AI_application/pn_data_n1000/Processed')




```python
for root, dirs, files in os.walk(os.path.join(SAVE_PATH,"train")):
  print(root, dirs, len(files))
print()

for root, dirs, files in os.walk(os.path.join(SAVE_PATH,"val")):
  print(root, dirs, len(files))
```

    /content/drive/MyDrive/AI_application/pn_data_n1000/Processed/train ['0', '1'] 0
    /content/drive/MyDrive/AI_application/pn_data_n1000/Processed/train/0 [] 527
    /content/drive/MyDrive/AI_application/pn_data_n1000/Processed/train/1 [] 273
    
    /content/drive/MyDrive/AI_application/pn_data_n1000/Processed/val ['0', '1'] 0
    /content/drive/MyDrive/AI_application/pn_data_n1000/Processed/val/0 [] 142
    /content/drive/MyDrive/AI_application/pn_data_n1000/Processed/val/1 [] 58


### 평균 (Mean) 과 표준 편차 (Std)


```python
os.path.join(SAVE_PATH, "train/**/*.npy")
len(glob.glob(os.path.join(SAVE_PATH, "train/**/*.npy")))
```




    800




```python
sums = 0
sums_squared = 0

for files in tqdm(glob.glob(os.path.join(SAVE_PATH, "train/**/*.npy"))):
  dcm_array = np.load(files)
  # Create the path to the dcm file
  normalizer = dcm_array.shape[0] * dcm_array.shape[1]  # Normalize sum of image
  sums += np.sum(dcm_array) / normalizer
  sums_squared += (np.power(dcm_array, 2).sum()) / normalizer
```


      0%|          | 0/1000 [00:00<?, ?it/s]



```python
mean = sums / lentrain
std = np.sqrt(sums_squared / lentrain - (mean**2))

print(f"Mean of Dataset: {mean}, STD: {std}")
```

    Mean of Dataset: 0.49042354910714325, STD: 0.24845305988425231


## Training


```python
import torch
from torch import nn, optim
import torch.nn.functional as F
# import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
print(torch.__version__)
```

    cuda:0
    2.5.1+cu124


First we create our dataset.
We can leverage the **DatasetFolder** from torchvision: It allows to simply pass a root directory and return return a dataset object with access to all files within the directory and the directory name as class label. <br />
We only need to define a loader function, **load_file**, which defines how the files shall be loaded.
This is very comfortable as we only have to load our previously stored numpy files.
Additionally, we need to define a list of file extensions (just "npy" in our case).

Finally we can pass a transformation sequence for Data Augmentation and Normalization.

We use:
* RandomResizedCrops which applies a random crop of the image and resizes it to the original image size (224x224)
* Random Rotations between -5 and 5 degrees
* Random Translation (max 5%)
* Random Scaling (0.9-1.1 of original image size)


```python
train_transforms = transforms.Compose([
                            transforms.ToTensor(),  # Convert numpy array to tensor
                            transforms.Normalize(0.097, 0.241),  # Use mean and std from preprocessing notebook
                            transforms.RandomAffine( # Data Augmentation
                                degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
                            transforms.RandomResizedCrop((224, 224), scale=(0.35, 1))
                            ])

val_transforms = transforms.Compose([
                            transforms.ToTensor(),  # Convert numpy array to tensor
                            transforms.Normalize(0.097, 0.241),  # Use mean and std from preprocessing notebook
                            ])
```

Finally, we create the train and val dataset and the corresponding data loaders.

Please adapt batch size and num_workers according to your hardware ressources.


```python
os.path.join(SAVE_PATH)
```




    '/content/drive/MyDrive/AI_application/pn_data_n1000/Processed'




```python
def load_file(path):
    return np.load(path).astype(np.float32)

# PyTorch dataset class that expects a folder structure where each subfolder represents a class.
# More generic than ImageFolder
# Can be used for any dataset type, not just images (e.g., audio, text, numpy files).
# Requires a custom loader function to specify how the data should be read.
# Labels are still inferred from folder names, similar to ImageFolder
os.path.join(SAVE_PATH)
train_dataset = datasets.DatasetFolder(root = os.path.join(SAVE_PATH, "train"),
                                        loader=load_file,
                                        extensions="npy",
                                        transform=train_transforms)

val_dataset = datasets.DatasetFolder(root = os.path.join(SAVE_PATH, "val"),
                                       loader=load_file,
                                       extensions="npy",
                                       transform=val_transforms)

```

###  Augmented train images


```python
train_dataset
```




    Dataset DatasetFolder
        Number of datapoints: 1000
        Root location: /content/drive/MyDrive/AI_application/pn_data_n1000/Processed/train
        StandardTransform
    Transform: Compose(
                   ToTensor()
                   Normalize(mean=0.097, std=0.241)
                   RandomAffine(degrees=[-5.0, 5.0], translate=(0, 0.05), scale=(0.9, 1.1))
                   RandomResizedCrop(size=(224, 224), scale=(0.35, 1), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=True)
               )




```python
fig, axis = plt.subplots(2, 2, figsize=(9, 9))
for i in range(2):
    for j in range(2):
        random_index = np.random.randint(0, 800)
        x_ray, label = train_dataset[random_index]
        axis[i][j].imshow(x_ray[0], cmap="bone")
        axis[i][j].set_title(f"Label:{label}")
```


    
*(이미지 없음 / image not preserved: output_37_0.png)*
    



```python
batch_size = 64 #TODO
# num_workers = 4#TODO

train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        # num_workers=num_workers,
                        shuffle=True)
val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        # num_workers=num_workers,
                        # persistent_workers=True,
                        shuffle=False)

print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images")
print(f"There are {len(train_loader)} train batches and {len(val_loader)} val batches")

```

    There are 1000 train images and 200 val images
    There are 16 train batches and 4 val batches



```python
a, b = next(iter(train_loader))
print(b, ":", b.shape)
print(a.dtype, ":", a.shape)
```

    tensor([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
            0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1]) : torch.Size([64])
    torch.float32 : torch.Size([64, 1, 224, 224])


The classes are imbalanced: There are more images without signs of pneumonia than with pneumonia.
There are multiple ways to deal with imbalanced datasets:
* Weighted Loss
* Oversampling
* Doing nothing :)

In this example, we will simply do nothing as this often yields the best results.
Buf feel free to play around with a weighted loss. A template to define a customized weighted loss function is provided below.

Oversampling will be shown in a later lecture.


```python
print(np.unique(train_dataset.targets, return_counts=True))
print(np.unique(val_dataset.targets, return_counts=True))
train_dataset[0][0].shape
```

    (array([0, 1]), array([669, 331]))
    (array([0, 1]), array([142,  58]))





    torch.Size([1, 224, 224])



### Model Creation in pytorch lightning

Each pytorch lightning model is defined by at least an initialization method, a **forward** function which defines the forward pass/prediction, a **training_step** which yields the loss and **configure_optimizers** to specify the optimization algorithm.

Additionally, we can use a **training_epoch_end** callback to compute overall dataset statistics and metrics such as accuracy.

Subsequently, we define the **validation_step**. The validation step performs more or less the same steps as the training step, however, on the validation data. In this case, pytorch lightning doesn't update the weights.
Again, we can use **validation_epoch_end** to compute overall dataset metrics.

No loops or manual weight updates are needed!<br />
Additionally, pl also handles device management.  Just pass the number of GPUS when creating the trainer.

**Now it is time to create the model** - We will use the ResNet18 network architecture.

As most of the torchvision models, the original ResNet expects a three channel input in **conv1**. <br />
However, our X-Ray image data has only one channel.
Thus we need to change the in_channel parameter from 3 to 1.

Additionally, we will change the last fully connected layer to have only one output as we have a binary class label.

### Model import


```python
## check pretrained model

weigths = models.ResNet18_Weights.IMAGENET1K_V1
resnet18 = models.resnet18(weights = weigths)
resnet18
```

    Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
    100%|██████████| 44.7M/44.7M [00:00<00:00, 175MB/s]





    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )




```python
torch.manual_seed(123)
torch.cuda.manual_seed(123)
class PneumoniaModel(nn.Module):
    def __init__(self, weight=1):
        super().__init__()

        weigths = models.ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights = weigths)
        for param in self.model.parameters():
            param.requires_grad = False
        # change conv1 from 3 to 1 input channels
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # change out_feature of the last fully connected layer (called fc in resnet18) from 1000 to 1
        self.model.fc = nn.Linear(in_features=512, out_features=1)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))


    def forward(self, data): #needs to compute output from the resnet and return prediction
        pred = self.model(data)
        return pred

model = PneumoniaModel().to(device)  # Instanciate the model
# model.model.conv1.weight
```

### Optimizer and Loss
We use the **Adam** Optimizer with a learning rate of 0.0001 and the **BinaryCrossEntropy** Loss function.<br />
(In fact we use **BCEWithLogitsLoss** which directly accepts the raw unprocessed predicted values and computes the sigmoid activation function before applying Cross Entropy).
Feel free to pass a weight different from 1 to the Pneumonia model in order to use the weighted loss function.


```python
# 학습률
lr = 1e-4
weight = 2
# 손실 함수 정의
# handling class imbalance in binary classification tasks.
# pos_weight를 device에 맞게 이동
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight])).to(device)

# 파라미터 수정 대상을 최종 노드로 제한
optimizer = optim.Adam(model.parameters(), lr=lr)
# step_size = 3
# gamma = 0.3
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
```


```python
a, b = next(iter(train_loader))
a = a.to(device)
a.shape
# output_a = model(a)
# output_a.shape
b.shape
```




    torch.Size([64])




```python
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments
# torch.cuda.empty_cache()
# 반복 횟수

num_epochs = 10

# 평가 결과 기록
history = []
acc = 0.0
for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0

    model.train()
    print("training")
    for X, Y in tqdm(train_loader):

        data, target = X.to(device), Y.to(device) # X shape = torch.Size([64, 1, 224, 224]), target shape = torch.Size([64])
        output = model(data)
        # output shape = torch.Size([64, 1])
        loss = criterion(output.squeeze(), target.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.where(output.squeeze() < 0.0, 0, 1)
        train_loss += loss.item()
        train_acc += (predicted == target).float().mean().item()

    # scheduler.step()
    avg_loss = train_loss/len(train_loader)
    avg_acc = train_acc/len(train_loader)
    print(f"epoch = {epoch}, avg_train_loss = {avg_loss}, avg_train_acc = {avg_acc}")


    model.eval()
    print("validating")
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for X_, Y_ in tqdm(val_loader):
            data_, target_ = X_.to(device), Y_.to(device)
            output_ = model(data_)
            val_loss = criterion(output_.squeeze(), target_.float())
            predicted_ = torch.where(output_.squeeze() < 0.0, 0, 1)
            val_loss += val_loss.item()
            val_acc += (predicted_ == target_).float().mean().item()

    avg_loss_ = val_loss/len(val_loader)
    avg_acc_ = val_acc/len(val_loader)
    print(f"avg_val_loss = {avg_loss_}, avg_val_acc = {avg_acc_}")

    if acc < avg_acc_:
        acc = avg_acc_
        torch.save(model.state_dict(), f"chest_x_epoch_{num_epochs}.pth") # model.load_state_dict(torch.load('chest_x.pth'))
        print("model saved!")

```

## Class Acvitation Maps (CAM)

The key idea of CAM is to multiply the output of the last convolutional layer (BasicBlock 1 of layer 4) $A_k$ (consisting of k channels) with the parameters $w$ of the subsequent fully connected layer to compute an activation map $M$:
$$ M = \sum_k w_kA_k$$

To do so, we need to access this particular output of the trained resnet18.<br />
Let's recap the resnet architecture:


```python
## check feature map
weights = models.ResNet18_Weights.IMAGENET1K_V1
resnet = models.resnet18(weights = weights)
list(resnet.children())[:-2]
```




    [Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
     BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
     ReLU(inplace=True),
     MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
     Sequential(
       (0): BasicBlock(
         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       )
       (1): BasicBlock(
         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       )
     ),
     Sequential(
       (0): BasicBlock(
         (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (downsample): Sequential(
           (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
           (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         )
       )
       (1): BasicBlock(
         (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       )
     ),
     Sequential(
       (0): BasicBlock(
         (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (downsample): Sequential(
           (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
           (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         )
       )
       (1): BasicBlock(
         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       )
     ),
     Sequential(
       (0): BasicBlock(
         (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (downsample): Sequential(
           (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
           (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         )
       )
       (1): BasicBlock(
         (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       )
     )]




```python
class PneumoniaModelCam(nn.Module):
    def __init__(self, weight=1):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights = weights)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # change conv1 from 3 to 1 input channels
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # change out_feature of the last fully connected layer (called fc in resnet18) from 1000 to 1
        self.model.fc = nn.Linear(in_features=512, out_features=1)

        # Extract the feature map
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])


    def forward(self, data): #needs to compute output from the resnet and return prediction

        # Compute feature map
        feature_map = self.feature_map(data)
        # Use Adaptive Average Pooling as in the original model
        avg_pool_output = F.adaptive_avg_pool2d(input=feature_map, output_size=(1, 1))
        print("avg_pool shape = ", avg_pool_output.shape)
        # Flatten the output into a 512 element vector
        avg_pool_output_flattened = torch.flatten(avg_pool_output)
        print("flatten shape = ", avg_pool_output_flattened.shape)
        # Compute prediction
        pred = self.model.fc(avg_pool_output_flattened)
        return pred, feature_map

```


```python
modelcam = PneumoniaModelCam().to(device)
modelcam.load_state_dict(torch.load('chest_x_epoch_100.pth'), strict=False)
modelcam.eval()
```

    <ipython-input-57-cb4b463a8f4a>:2: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      modelcam.load_state_dict(torch.load('chest_x_epoch_100.pth'), strict=False)





    PneumoniaModelCam(
      (model): ResNet(
        (conv1): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (layer1): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (layer2): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (layer3): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (layer4): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        (fc): Linear(in_features=512, out_features=1, bias=True)
      )
      (feature_map): Sequential(
        (0): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (4): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (6): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (7): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
    )




```python
val_dataset[-6][0].shape
```




    torch.Size([1, 224, 224])




```python
img = val_dataset[-6][0].to(device) # torch.Size([1, 224, 224])
pred, feature = modelcam(img.unsqueeze(0))
print("Pred = ", pred)
print('feature shape = ', feature.shape)
```

    avg_pool shape =  torch.Size([1, 512, 1, 1])
    flatten shape =  torch.Size([512])
    Pred =  tensor([-0.8030], device='cuda:0', grad_fn=<ViewBackward0>)
    feature shape =  torch.Size([1, 512, 7, 7])



```python
dict(model.model.fc.named_parameters())
```




    {'weight': Parameter containing:
     tensor([[ 4.4837e-03, -2.9426e-02, -2.2707e-02, -3.4825e-02,  6.7029e-03,
               1.2620e-02,  1.9409e-02, -1.4616e-02, -1.2789e-02, -3.5215e-02,
              -3.9603e-02, -2.4320e-02,  2.4804e-02, -1.3401e-02, -2.4492e-02,
              -2.9951e-02, -3.9652e-02,  8.5152e-03,  4.1891e-02,  1.9988e-02,
               2.6466e-02, -1.1007e-02, -5.3456e-03,  2.3835e-02, -3.1685e-02,
               2.6306e-03,  8.4966e-03, -3.7089e-03, -3.0864e-02,  4.4492e-02,
              -2.7702e-02, -2.8961e-02, -3.8384e-02, -1.8901e-02, -2.1286e-02,
               2.5557e-02,  1.5611e-02, -1.9796e-02,  3.1271e-02, -3.4391e-02,
               2.7614e-02, -2.9202e-02, -8.7393e-03,  2.1729e-02, -2.6405e-02,
              -3.3928e-02,  1.1580e-03, -9.7882e-03,  1.3733e-02, -3.1444e-02,
               3.0391e-02, -3.2474e-02, -2.6669e-02,  1.2753e-02,  2.6817e-02,
              -3.9953e-02,  3.1175e-02, -3.2446e-02, -4.1963e-02, -4.6603e-03,
              -3.2204e-02,  3.5213e-02, -2.5521e-02, -1.0758e-02,  1.4334e-02,
              -1.1336e-02, -2.4301e-02, -2.2196e-02,  4.8337e-03,  2.1057e-03,
              -4.3737e-02, -3.0265e-02,  4.2882e-02,  9.6468e-04, -2.7220e-02,
              -3.2731e-03,  3.3219e-02,  1.6968e-02,  3.0762e-02, -2.1792e-02,
              -9.1683e-03,  2.6800e-02,  1.4621e-02, -4.4345e-02, -1.6669e-02,
               1.0906e-02,  1.5351e-02, -1.6852e-02,  4.1689e-02, -3.7136e-02,
               4.3461e-03, -4.4889e-02,  3.8577e-02,  5.2315e-03,  1.0580e-02,
              -1.7068e-02,  1.1604e-02,  3.3422e-02,  8.8073e-03,  2.5628e-02,
               2.5104e-02, -4.1619e-02, -6.9177e-03,  4.5590e-02,  1.1161e-02,
               2.1309e-02, -4.1536e-03,  4.8795e-03, -3.4450e-02,  3.5939e-02,
               2.5745e-02, -9.6238e-03, -1.5370e-02,  1.5858e-02, -4.4127e-02,
               4.0935e-02,  1.2036e-02, -3.4339e-03,  2.5998e-02, -2.3569e-02,
              -3.1198e-02,  2.6030e-02, -2.8680e-02,  3.9746e-02, -6.7387e-03,
              -1.9180e-02,  1.3141e-02, -2.1830e-02,  1.9632e-03,  2.3311e-02,
               5.2505e-03, -1.9511e-02, -3.5149e-02,  2.9225e-02, -3.8404e-02,
               1.9871e-02, -5.4025e-03,  4.0007e-02,  2.5855e-02, -4.7596e-02,
               1.1247e-02,  7.6232e-04,  7.5117e-03, -1.7188e-02,  3.8284e-02,
              -3.9439e-02,  5.2593e-04,  2.5728e-02, -8.4607e-03,  3.2562e-02,
              -3.1045e-05, -3.5377e-02, -7.0924e-03,  2.1085e-03,  7.1730e-03,
               4.5055e-04,  1.5169e-02, -3.7730e-03, -4.3073e-03, -4.0440e-02,
              -1.0004e-02,  4.0593e-02,  1.6517e-02, -3.5212e-02,  2.0970e-02,
               3.6095e-02, -6.7599e-03, -8.5133e-03,  1.2882e-02,  1.3752e-02,
               2.5993e-02, -1.1882e-02, -1.8143e-02, -2.9318e-02,  1.4239e-02,
              -2.5879e-02, -4.0010e-02, -8.7290e-03, -2.2759e-03,  4.4731e-02,
              -4.1757e-02,  3.8309e-02, -1.3020e-02,  3.6918e-02, -9.5870e-03,
              -8.3810e-03,  1.5180e-02,  4.0428e-02,  3.5096e-02, -2.8873e-02,
              -3.5614e-02,  6.2930e-03,  3.2467e-02, -4.6826e-03, -4.3270e-02,
              -1.1834e-02,  1.4647e-02,  3.5313e-02,  1.6642e-03, -2.7663e-02,
               2.5754e-02, -6.9747e-03,  3.0550e-02, -3.1728e-02,  2.4688e-02,
              -4.0744e-02,  3.8282e-03,  4.1327e-02, -5.6688e-03,  4.3080e-02,
              -1.0477e-02,  1.4219e-02,  1.8678e-02, -2.0500e-02,  5.9493e-03,
               4.1107e-02, -3.7892e-02,  2.7756e-02, -1.9315e-02,  2.0952e-02,
              -2.8009e-02,  3.8936e-03, -1.4093e-02, -4.6712e-03, -1.9819e-02,
              -1.2852e-02,  4.4376e-02,  1.4799e-02, -2.4135e-02,  4.4003e-03,
              -2.8784e-02, -1.7416e-03,  4.2421e-02, -1.1165e-02, -9.6734e-03,
              -4.3825e-02,  3.8849e-02,  8.7949e-03, -2.2777e-02,  8.3222e-03,
              -5.0719e-03,  1.2983e-02, -1.0972e-05, -2.3511e-04,  3.1520e-04,
              -4.4017e-02,  8.8746e-03, -2.7652e-02,  1.1316e-02,  1.2743e-03,
              -4.0350e-03, -1.1532e-02,  1.6216e-02,  3.2067e-02,  3.8915e-02,
               3.7305e-02, -1.9653e-02,  4.1584e-02, -3.5471e-02,  3.7601e-02,
              -4.2470e-02, -3.0593e-02, -2.0477e-02,  4.7870e-03,  3.2055e-02,
               2.9212e-02, -1.6392e-02, -1.4549e-02, -2.8559e-02,  8.6178e-03,
              -3.9819e-02,  3.8249e-02, -1.8188e-02, -2.9748e-02,  4.2645e-02,
               3.5564e-02, -2.4974e-02, -9.4164e-03, -1.3783e-03,  1.9673e-03,
              -4.6057e-02,  1.7382e-02, -1.7852e-02,  3.8356e-02,  4.4211e-03,
              -2.7774e-02, -3.1426e-02,  3.9334e-02,  2.4808e-02, -1.7353e-02,
              -3.5931e-02, -1.5957e-02, -9.9747e-03, -3.4218e-02, -5.5544e-03,
              -1.6510e-02,  2.3113e-03, -4.5738e-02, -3.4350e-03,  2.8116e-03,
              -1.7029e-02, -1.4785e-02,  5.5162e-03,  3.9966e-02,  3.4080e-02,
              -9.4782e-03,  4.0492e-03,  7.3805e-04,  3.2176e-02,  5.4201e-04,
               2.8000e-02,  1.9747e-02,  1.2282e-02, -3.3732e-03,  1.6869e-03,
               1.0282e-02, -2.7003e-02, -5.6750e-03, -3.3441e-02,  3.3387e-03,
              -3.8749e-02,  2.8121e-02,  8.4170e-03, -3.8681e-02,  2.6013e-03,
               1.7253e-02,  3.0605e-02, -2.5683e-02,  8.1453e-03,  6.3724e-03,
               3.8821e-02, -2.1262e-02, -2.6260e-03,  1.2808e-02,  3.8530e-02,
              -3.1688e-02,  2.7188e-02,  2.3628e-02, -5.3052e-03,  1.8840e-02,
               2.8629e-02,  3.9443e-02, -2.3486e-02, -2.8353e-02,  4.1440e-02,
              -4.1207e-02,  4.0289e-02, -2.2603e-02,  3.2866e-02, -3.3771e-02,
              -1.0108e-03, -3.0773e-02,  2.5017e-02,  4.6277e-03, -2.2369e-03,
              -2.0259e-02,  9.9187e-03, -4.4707e-02, -2.5830e-03, -1.0369e-02,
               1.5601e-02,  2.7646e-02, -1.9365e-02,  4.3594e-02,  4.2947e-02,
               3.7891e-03,  2.2037e-02, -2.3685e-02, -2.2248e-02,  4.0354e-02,
               2.2568e-02,  2.5339e-02, -3.9379e-02,  3.6035e-02,  5.4586e-03,
               3.4198e-02,  3.6501e-02, -8.3062e-03, -2.3200e-02,  8.6123e-03,
               4.5316e-02,  6.1646e-03,  3.4129e-02,  1.2116e-02, -1.9044e-02,
              -1.9437e-02, -1.6777e-02, -1.6821e-03, -2.0052e-02,  2.0456e-02,
               3.4149e-02,  2.0829e-02, -4.0876e-02, -1.5561e-03, -5.3483e-03,
              -2.7087e-02, -1.1526e-02, -1.2869e-02,  3.0620e-02,  3.2182e-02,
              -1.1112e-02, -1.9727e-02,  2.0537e-02,  4.2681e-02,  3.9479e-02,
               2.5992e-02, -2.4351e-02, -4.7910e-03, -2.6562e-02, -2.6966e-02,
              -1.4962e-02, -1.8355e-02,  3.3818e-04, -8.5953e-04,  9.3356e-03,
              -2.1392e-02, -4.6986e-02,  2.3361e-02, -2.4421e-02,  3.5858e-02,
              -2.6182e-02,  4.4403e-02,  3.9767e-02, -2.1602e-02, -9.0311e-04,
               3.3430e-02, -3.7945e-03,  8.3900e-03, -2.5353e-02, -7.7217e-03,
               3.2036e-03,  2.3067e-02, -2.6664e-02,  1.0320e-02,  3.0638e-02,
              -3.9085e-02, -5.2819e-03, -2.2597e-02, -3.3811e-04,  1.7338e-02,
               3.3368e-02,  3.8475e-03,  1.9426e-02,  3.0402e-02, -1.7961e-02,
               1.9587e-02, -2.7189e-03,  8.4402e-03,  1.2235e-02,  2.2238e-02,
              -3.3480e-02, -1.4274e-03,  4.6438e-02, -9.0158e-03,  4.2387e-02,
              -3.0621e-02, -1.4190e-02, -3.9433e-02,  1.5915e-02, -2.7501e-02,
              -3.8073e-02, -1.7789e-02, -2.2284e-02,  1.1835e-02, -2.8318e-02,
              -8.9280e-03,  4.0069e-02, -3.4960e-02, -2.9226e-02, -3.8003e-02,
               2.3278e-02,  1.3586e-02,  3.8724e-02, -2.6453e-02,  3.3572e-03,
               5.8360e-03, -3.6453e-02, -1.3299e-03, -5.0975e-03, -1.6501e-02,
              -2.1061e-02,  1.4884e-02,  2.2573e-02,  4.3982e-02, -2.2044e-02,
              -6.5082e-03,  8.3677e-03,  1.3108e-02, -3.3834e-02,  1.4092e-02,
               2.3025e-02, -8.7456e-03, -4.1529e-02, -2.5860e-02,  1.4938e-02,
               3.0774e-04, -4.6779e-03,  1.2506e-02, -4.2616e-04,  3.7720e-02,
              -2.9546e-02, -1.4640e-02, -1.4034e-02,  1.6368e-03, -4.0602e-02,
              -1.7172e-03, -4.0597e-02,  1.6055e-02,  4.1927e-02, -1.5109e-02,
               2.0825e-02,  3.2754e-02]], device='cuda:0', requires_grad=True),
     'bias': Parameter containing:
     tensor([-0.0407], device='cuda:0', requires_grad=True)}




```python
weight_params = list(model.model.fc.parameters())[0]
weight_params
weight = weight_params[0].detach()
weight.shape
# torch.matmul(weight, feature.reshape((512, 49)))
```




    torch.Size([512])




```python
def cam(model, img):
    with torch.no_grad():
        pred, features = model(img.unsqueeze(0))
    features = features.reshape((512, 49))
    weight_params = list(model.model.fc.parameters())[0]
    weight = weight_params[0].detach()

    cam = torch.matmul(weight, features)
    cam_img = cam.reshape(7, 7).cpu()
    return cam_img, torch.sigmoid(pred)

```


```python
def cam(model, img):
    """
    Compute class activation map according to cam algorithm
    """
    with torch.no_grad():
        pred, features = model(img.unsqueeze(0))
    b, c, h, w = features.shape

    # We reshape the 512x7x7 feature tensor into a 512x49 tensor in order to simplify the multiplication
    features = features.reshape((c, h*w))

    # Get only the weights, not the bias
    weight_params = list(model.model.fc.parameters())[0]

    # Remove gradient information from weight parameters to enable numpy conversion
    weight = weight_params[0].detach()
    print(weight.shape)
    # Compute multiplication between weight and features with the formula from above.
    # We use matmul because it directly multiplies each filter with the weights
    # and then computes the sum. This yields a vector of 49 (7x7 elements)
    cam = torch.matmul(weight, features)
    print(features.shape)

    ### The following loop performs the same operations in a less optimized way
    #cam = torch.zeros((7 * 7))
    #for i in range(len(cam)):
    #    cam[i] = torch.sum(weight*features[:,i])
    ##################################################################

    # Normalize and standardize the class activation map (Not always necessary, thus not shown in the lecture)
    cam = cam - torch.min(cam)
    cam_img = cam / torch.max(cam)
    # Reshape the class activation map to 512x7x7 and move the tensor back to CPU
    cam_img = cam_img.reshape(h, w).cpu()

    return cam_img, torch.sigmoid(pred)

def visualize(img, heatmap, pred):
    """
    Visualization function for class activation maps
    """
    img = img[0]
    # Resize the activation map of size 7x7 to the original image size (224x224)
    heatmap = transforms.functional.resize(heatmap.unsqueeze(0), (img.shape[0], img.shape[1]))[0]

    # Create a figure
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img, cmap="bone")
    # Overlay the original image with the upscaled class activation map
    axis[1].imshow(img, cmap="bone")
    axis[1].imshow(heatmap, alpha=0.5, cmap="jet")
    plt.title(f"Pneumonia: {(pred > 0.5).item()}")
```


```python
idx = np.random.randint(0, len(val_dataset), (1,))

img = val_dataset[idx[0]][0].to(device)  # Select a subject

activation_map, pred = cam(modelcam, img)
visualize(img.cpu(), activation_map, pred)
```

    avg_pool shape =  torch.Size([1, 512, 1, 1])
    flatten shape =  torch.Size([512])
    torch.Size([512])
    torch.Size([512, 49])



    
*(이미지 없음 / image not preserved: output_62_1.png)*
    

