# ***BANDON Building Dataset***
This dataset may be a bit tricky, but it will also be quick. Don't worry~

Go to this dir, let us begin~~~
```shell script
cd BADNON_Dataset
```
First, you need to download the dataset from [this website](https://github.com/fitzpchao/BANDON) and unzip it to **data_dir**. Also, change the folder name to "BANDON". You will obtain a data folder following the format outlined below.
```none
$data_dir
├── BANDON
│   ├── train/val/test (following is the subdir of 'train' for example)
│   │   ├── imgs
│   │   |   ├── bj
│   │   │   |   ├── t1 (*.jpg)
│   │   │   |   ├── t2 (*.jpg)
│   │   │   |   ├── t3 (*.jpg)
│   │   |   ├── sh
│   │   │   |   ├── t1 (*.jpg)
│   │   │   |   ├── t2 (*.jpg)
│   │   │   |   ├── t3 (*.jpg)
│   │   ├── building_labels
│   │   |   ├── bj
│   │   │   |   ├── t1 (*.png)
│   │   │   |   ├── t2 (*.png)
│   │   │   |   ├── t3 (*.png)
│   │   |   ├── sh
│   │   │   |   ├── t1 (*.png)
│   │   │   |   ├── t2 (*.png)
│   │   │   |   ├── t3 (*.png)
│   │   ├── labels_unch0ch1ig255 (labels when val)
│   │   |   ├── bj
│   │   │   |   ├── t1VSt2 (*.png)
│   │   │   |   ├── t1VSt3 (*.png)
│   │   │   |   ├── t2VSt3 (*.png)
│   │   |   ├── sh
│   │   │   |   ├── t1VSt2 (*.png)
│   │   │   |   ├── t1VSt3 (*.png)
│   │   │   |   ├── t2VSt3 (*.png)
│   │   ├── flow_bt_resize256
│   │   ├── offset_st_resize256
```
I hope you can specify **data_dir** by this way, and then you can use following commands without any change:
```
export data_dir='Your specific path'
```
**Secondly**, we divide test dataset into non-overlapping images of size 512×512. You can use the following command to stride crop:
```shell script
# stride crop images
python stride_crop.py \
--input-dir="$data_dir/BANDON/test" \
--output-dir="$data_dir/BANDON/test_512_nooverlap" \
--mode="img"

# !!! stride crop masks and do mask[mask != 0] = 255 !!!
python stride_crop.py \
--input-dir="$data_dir/BANDON/test" \
--output-dir="$data_dir/BANDON/test_512_nooverlap" \
--mode="mask" 
```

**Thirdly**, generate the data_list for train and test according to the following command.
```shell script
# for train
python generate_data_list.py \
--input-dir="$data_dir/BANDON/train" \
--output-dir="$data_dir/BANDON/train.txt" \
--split="train"

# for test
python generate_data_list.py \
--input-dir="$data_dir/BANDON/test_512_nooverlap" \
--output-dir="$data_dir/BANDON/test.txt" \
--split="test"
```

For convenience, we did another conversion here, splitting the labels containing both building and change detection for testing into two separate txt files, one containing only building labels and the other containing only change detection labels.
```shell script
python all2bxcd.py \
--input-all-dir="$data_dir/BANDON/test.txt" \
--output-bx-dir="$data_dir/BANDON/test_bx.txt" \
--output-cd-dir="$data_dir/BANDON/test_cd.txt" 
```
You will change the txt file(test.txt) where each line is in this format
```
img_a, img_b, label_cd, label_a, label_b
```
to
```
img_a, **, **, label_a, **     # test_bx.txt
img_a, img_b, label_cd, **, ** # test_cd.txt
```

Finally, the following folder structure will be obtained.
```none
$data_dir
├── BANDON
│   ├── train (subdir is omitted)
│   ├── val   (subdir is omitted)
│   ├── test  (subdir is omitted)
│   ├── train.txt    (1689 lines)
│   ├── test.txt     (3312 lines)
│   ├── test_bx.txt  (3312 lines)
│   ├── test_cd.txt  (3312 lines)
```