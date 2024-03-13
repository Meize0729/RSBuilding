# ***S2Looking Building Dataset***
Go to this dir, let us begin~~~
```shell script
cd S2Looking_Dataset
```
First, you need to download the dataset from [this website](https://chenhao.in/LEVIR/) and unzip it to **data_dir**. Also, change the folder name to "S2Looking". You will obtain a data folder following the format outlined below.
```none
$data_dir
├── S2Looking
│   ├── train
│   │   ├── Image1 (*.png)
│   │   ├── Image2 (*.png)
│   │   ├── label (*.png)(0,255)
│   ├── val
│   │   ├── Image1 (*.png)
│   │   ├── Image2 (*.png)
│   │   ├── label (*.png)(0,255)
│   ├── test
│   │   ├── Image1 (*.png)
│   │   ├── Image2 (*.png)
│   │   ├── label (*.png)(0,255)
```
I hope you can specify **data_dir** by this way, and then you can use following commands without any change:
```
export data_dir='Your specific path'
```
**Secondly**, we divide test dataset into non-overlapping images of size 512×512. You can use the following command to stride crop:
```shell script
python stride_crop.py \
--input-dir="$data_dir/S2Looking/test/Image1" \
--output-dir="$data_dir/S2Looking/test/Image1_512_nooverlap" 
python stride_crop.py \
--input-dir="$data_dir/S2Looking/test/Image2" \
--output-dir="$data_dir/S2Looking/test/Image2_512_nooverlap" 
python stride_crop.py \
--input-dir="$data_dir/S2Looking/test/label" \
--output-dir="$data_dir/S2Looking/test/label_512_nooverlap" 
```

**Thirdly**, generate the data_list for train and test according to the following command.
```shell script
# for train
python generate_data_list.py \
--input-img-A-dir="$data_dir/S2Looking/train/Image1" \
--input-img-B-dir="$data_dir/S2Looking/train/Image2" \
--input-mask-dir="$data_dir/S2Looking/train/label" \
--output-txt-dir="$data_dir/S2Looking/train.txt"

# for test
python generate_data_list.py \
--input-img-A-dir="$data_dir/S2Looking/test/Image1_512_nooverlap" \
--input-img-B-dir="$data_dir/S2Looking/test/Image2_512_nooverlap" \
--input-mask-dir="$data_dir/S2Looking/test/label_512_nooverlap" \
--output-txt-dir="$data_dir/S2Looking/test.txt"
```

Finally, the following folder structure will be obtained.
```none
$data_dir
├── S2Looking
│   ├── train (3500 pictures)(1024 x 1024)
│   │   ├── Image1 (*.png)
│   │   ├── Image2 (*.png)
│   │   ├── label (*.png)(0,255)
│   ├── val (500 pictures)(1024 x 1024)
│   │   ├── Image1 (*.png)
│   │   ├── Image2 (*.png)
│   │   ├── label (*.png)(0,255)
│   ├── test 
│   │   ├── Image1 (*.png) (1000 pictures)(1024 x 1024)
│   │   ├── Image2 (*.png) (1000 pictures)(1024 x 1024)
│   │   ├── label (*.png)(0,255) (1000 pictures)(1024 x 1024)
│   │   ├── Image1_512_nooverlap (*.png) (4000 pictures)(512 x 512)
│   │   ├── Image2_512_nooverlap (*.png) (4000 pictures)(512 x 512)
│   │   ├── label_512_nooverlap (*.png) (4000 pictures)(512 x 512)
│   ├── train.txt
│   ├── test.txt
```