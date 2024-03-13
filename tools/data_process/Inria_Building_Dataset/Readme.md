# ***Inria Building Dataset***
Go to this dir, let us begin~~~
```shell script
cd Inria_Building_Dataset
```
First, you need to download the dataset from [this website](https://project.inria.fr/aerialimagelabeling/) and unzip it to **data_dir**. You will obtain a data folder named "AerialImageDataset" following the format outlined below.
```none
$data_dir
├── AerialImageDataset
│   ├── train
│   │   ├── gt(*.tif)(0,255)
│   │   ├── images(*.tif)
│   ├── test
│   │   ├── images(*.tif)
```

I hope you can specify **data_dir** by this way, and then you can use following commands without any change:
```
export data_dir='Your specific path'
```
**Secondly**, since the testing set of this dataset lacks annotations, consistent with [Buildformer](https://github.com/WangLibo1995/BuildFormer/tree/main), use IDs 1-5 of each city as the testing set. You can use the following command to split train and test data:
```shell script
python split_train_val.py \
--input-img-dir="$data_dir/AerialImageDataset/train/images" \
--input-mask-dir="$data_dir/AerialImageDataset/train/gt" \
--output-train-img-dir="$data_dir/AerialImageDataset/train_images" \
--output-train-mask-dir="$data_dir/AerialImageDataset/train_masks" \
--output-val-img-dir="$data_dir/AerialImageDataset/val_images" \
--output-val-mask-dir="$data_dir/AerialImageDataset/val_masks" 
```

**And then**, divide them into non-overlapping images of size 512×512. Buildings with a pixel occupancy of less than 5% in the training set will be excluded. You can use the following command to perform the above steps.
```shell script
# for train
python stride_crop.py \
--input-img-dir="$data_dir/AerialImageDataset/train_images" \
--input-mask-dir="$data_dir/AerialImageDataset/train_masks" \
--output-img-dir="$data_dir/AerialImageDataset/train_processed/images" \
--output-mask-dir="$data_dir/AerialImageDataset/train_processed/gt" \
--mode="train"

# for test
python stride_crop.py \
--input-img-dir="$data_dir/AerialImageDataset/val_images" \
--input-mask-dir="$data_dir/AerialImageDataset/val_masks" \
--output-img-dir="$data_dir/AerialImageDataset/val_processed/images" \
--output-mask-dir="$data_dir/AerialImageDataset/val_processed/gt" \
--mode="val"
```
**Thirdly**, generate the data_list for train and test according to the following command.
```shell script
# for train
python generate_data_list.py \
--input-img-dir="$data_dir/AerialImageDataset/train_processed/images" \
--input-mask-dir="$data_dir/AerialImageDataset/train_processed/gt" \
--output-txt-dir="$data_dir/AerialImageDataset/train.txt"

# for test
python generate_data_list.py \
--input-img-dir="$data_dir/AerialImageDataset/val_processed/images" \
--input-mask-dir="$data_dir/AerialImageDataset/val_processed/gt" \
--output-txt-dir="$data_dir/AerialImageDataset/val.txt"
```

Finally, the following folder structure will be obtained.
```none
$data_dir
├── AerialImageDataset
│   ├── train (origin, after unzip)
│   │   ├── gt
│   │   ├── images
│   ├── test (origin, after unzip)
│   │   ├── images
**********************************************************************
│   ├── train_images (tmp, after split split_train_val.py)
│   ├── train_masks (tmp, after split split_train_val.py)
│   ├── val_images (tmp, after split split_train_val.py)
│   ├── val_masks (tmp, after split split_train_val.py)
**********************************************************************
│   ├── train_processed (terminal, after stride_crop.py)(9737 pictures)
│   │   ├── gt
│   │   ├── images
│   ├── val_processed (terminal, after stride_crop.py)(1942 pictures)
│   │   ├── gt
│   │   ├── images
│   ├── train.txt (terminal, after generate_data_list.py)
│   ├── val.txt (terminal, after generate_data_list.py)
```