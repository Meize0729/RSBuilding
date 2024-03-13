# ***WHU Building Dataset***
Go to this dir, let us begin~~~
```shell script
cd WHU_Building_Dataset
```
**Firstly**, we recommend that you download the dataset from [this website](http://gpcv.whu.edu.cn/data/building_dataset.html) and unzip it to **data_dir**. Also, change the folder name to "whubuilding". You will obtain a data folder  following the format outlined below.
```none
$data_dir
├── whubuilding
│   ├── train
│   │   ├── image (*.tif)
│   │   ├── label (*.tif)(0,255)
│   ├── val
│   │   ├── image (*.tif)
│   │   ├── label (*.tif)(0,255)
│   ├── test
│   │   ├── image (*.tif)
│   │   ├── label (*.tif)(0,255)
```
I hope you can specify **data_dir** by this way, and then you can use following commands without any change:
```
export data_dir='Your specific path'
```

**And then**, generate the data_list for train and test according to the following command.
```shell script
# for train
python generate_data_list.py \
--input-img-dir="$data_dir/whubuilding/train/image" \
--input-mask-dir="$data_dir/whubuilding/train/label" \
--output-txt-dir="$data_dir/whubuilding/train.txt"

# for val
python generate_data_list.py \
--input-img-dir="$data_dir/whubuilding/val/image" \
--input-mask-dir="$data_dir/whubuilding/val/label" \
--output-txt-dir="$data_dir/whubuilding/val.txt"

# for test
python generate_data_list.py \
--input-img-dir="$data_dir/whubuilding/test/image" \
--input-mask-dir="$data_dir/whubuilding/test/label" \
--output-txt-dir="$data_dir/whubuilding/test.txt"
```

Finally, the following folder structure will be obtained.
```none
$data_dir
├── whubuilding
│   ├── train (4736 pictures)
│   │   ├── image (*.tif)
│   │   ├── label (*.tif)(0,255)
│   ├── val (1036 pictures)
│   │   ├── image (*.tif)
│   │   ├── label (*.tif)(0,255)
│   ├── test (2416 pictures)
│   │   ├── image (*.tif)
│   │   ├── label (*.tif)(0,255)
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
```