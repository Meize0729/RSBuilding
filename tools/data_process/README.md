# Data process

## Step 1
Next, go to the dataset you want to use and proceed with further processing. 

**I believe that these preprocessing steps and code will satisfy you and won't consume a lot of your time. Be sure to pay attention to the following tips, especially the third one.**
```shell script
# must choose one
cd WHU_Building_Dataset / Inria_Building_Dataset / Levir_CD_Dataset / S2Looking_Dataset / BANDON_Dataset
```
**Notice:** 
1. You should replace **data_dir** with your specific path or specific the exact location of **data_dir** in the terminal beforehand.
2. All labels are saved in uint8 PNG format, where pixel value **255** represents buildings and pixel value **0** represents non-buildings.
3. Each dataset will eventually form a txt file called ***data_list***, and the data format saved in it will be as follows:
```
# building extraction
image, **, **, label, ** 
# change detection
image_a, image_b, label_cd, **, **
# both
image_a, image_b, label_cd, label_a, label_b
```

## Step 2

1. Now, please double-check if the data_list for your required datasets have been generated and if the quantities match those described in the instructions.

2. Next, you should fill in the absolute path of the generated data_list in the corresponding locations [data_list](../../data_list/) below:
```
RSbuilding
├── ···
├── ···
├── data_list
│   ├── whu
│   │   ├── train.txt (need fill)
│   │   ├── test.txt (need fill)
│   ├── inria
│   │   ├── train.txt (need fill)
│   │   ├── test.txt (need fill)
│   ├── levircd
│   │   ├── train.txt (need fill)
│   │   ├── test.txt (need fill)
│   ├── s2looking
│   │   ├── train.txt (need fill)
│   │   ├── test.txt (need fill)
│   ├── bandon
│   │   ├── train.txt (need fill)
│   │   ├── test.txt (need fill)
│   ├── bandon
│   │   ├── train.txt (need fill)
│   │   ├── test.txt (need fill)
```

3. Additionally, you have the flexibility to mix and match the training and testing data as desired. For example, you can easily merge the 'levir-cd' and 'whu' datasets by simply combining the data_lists of both datasets, as shown like [here](../../data_list/pretrain/train.txt).

