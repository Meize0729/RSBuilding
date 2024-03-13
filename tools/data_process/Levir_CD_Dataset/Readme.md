# ***Levir-CD Building Dataset***
Go to this dir, let us begin~~~
```shell script
cd Levir_CD_Dataset
```
First, you need to download the dataset from [this website](https://chenhao.in/LEVIR/) and unzip it to **data_dir**. Also, change the folder name to "Levir-CD". You will obtain a data folder following the format outlined below.
```none
$data_dir
├── Levir-CD
│   ├── train
│   │   ├── A (*.png)
│   │   ├── B (*.png)
│   │   ├── label (*.png)(0,255)
│   ├── val
│   │   ├── A (*.png)
│   │   ├── B (*.png)
│   │   ├── label (*.png)(0,255)
│   ├── test
│   │   ├── A (*.png)
│   │   ├── B (*.png)
│   │   ├── label (*.png)(0,255)
```
I hope you can specify **data_dir** by this way, and then you can use following commands without any change:
```
export data_dir='Your specific path'
```
**Secondly**, we divide test dataset into non-overlapping images of size 512×512. You can use the following command to stride crop:
```shell script
python stride_crop.py \
--input-dir="$data_dir/Levir-CD/test/A" \
--output-dir="$data_dir/Levir-CD/test/A_512_nooverlap" 
python stride_crop.py \
--input-dir="$data_dir/Levir-CD/test/B" \
--output-dir="$data_dir/Levir-CD/test/B_512_nooverlap" 
python stride_crop.py \
--input-dir="$data_dir/Levir-CD/test/label" \
--output-dir="$data_dir/Levir-CD/test/label_512_nooverlap" 
```

**Thirdly**, generate the data_list for train and test according to the following command.
```shell script
# for train
python generate_data_list.py \
--input-img-A-dir="$data_dir/Levir-CD/train/A" \
--input-img-B-dir="$data_dir/Levir-CD/train/B" \
--input-mask-dir="$data_dir/Levir-CD/train/label" \
--output-txt-dir="$data_dir/Levir-CD/train.txt"

# for test
python generate_data_list.py \
--input-img-A-dir="$data_dir/Levir-CD/test/A_512_nooverlap" \
--input-img-B-dir="$data_dir/Levir-CD/test/B_512_nooverlap" \
--input-mask-dir="$data_dir/Levir-CD/test/label_512_nooverlap" \
--output-txt-dir="$data_dir/Levir-CD/test.txt"
```

Finally, the following folder structure will be obtained.
```none
$data_dir
├── Levir-CD
│   ├── train (445 pictures)(1024 x 1024)
│   │   ├── A (*.png)
│   │   ├── B (*.png)
│   │   ├── label (*.png)(0,255)
│   ├── val (64 pictures)(1024 x 1024)
│   │   ├── A (*.png)
│   │   ├── B (*.png)
│   │   ├── label (*.png)(0,255)
│   ├── test 
│   │   ├── A (*.png) (128 pictures)(1024 x 1024)
│   │   ├── B (*.png) (128 pictures)(1024 x 1024)
│   │   ├── label (*.png)(0,255) (128 pictures)(1024 x 1024)
│   │   ├── A_512_nooverlap (*.png) (512 pictures)(512 x 512)
│   │   ├── B_512_nooverlap (*.png) (512 pictures)(512 x 512)
│   │   ├── label_512_nooverlap (*.png) (512 pictures)(512 x 512)
│   ├── train.txt
│   ├── test.txt
```