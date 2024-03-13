import argparse
import os
import shutil
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Distribute images and masks into train and val sets based on city ID.")
    parser.add_argument("--input-img-dir",         default='data_dir/AerialImageDataset/train/images', help="Directory containing input images.")
    parser.add_argument("--input-mask-dir",        default='data_dir/AerialImageDataset/train/gt', help="Directory containing input masks.")
    parser.add_argument("--output-train-img-dir",  default='data_dir/AerialImageDataset/train_images', help="Output directory for train images.")
    parser.add_argument("--output-train-mask-dir", default='data_dir/AerialImageDataset/train_masks', help="Output directory for train masks.")
    parser.add_argument("--output-val-img-dir",    default='data_dir/AerialImageDataset/val_images', help="Output directory for val images.")
    parser.add_argument("--output-val-mask-dir",   default='data_dir/AerialImageDataset/val_masks', help="Output directory for val masks.")
    return parser.parse_args()

def distribute_files(args):
    # Create output directories if they do not exist
    os.makedirs(args.output_train_img_dir, exist_ok=True)
    os.makedirs(args.output_train_mask_dir, exist_ok=True)
    os.makedirs(args.output_val_img_dir, exist_ok=True)
    os.makedirs(args.output_val_mask_dir, exist_ok=True)

    # Iterate through images in the input image directory
    for img_filename in os.listdir(args.input_img_dir):

        city_id = int(re.findall(r'\d+', img_filename)[0])  # Assuming city ID is the last character before the file extension
        src_img_path = os.path.join(args.input_img_dir, img_filename)
        src_mask_path = os.path.join(args.input_mask_dir, img_filename)

        # Determine if the file should go to train or val based on city ID
        if city_id in [1, 2, 3, 4, 5]:  # For val
            dst_img_path = os.path.join(args.output_val_img_dir, img_filename)
            dst_mask_path = os.path.join(args.output_val_mask_dir, img_filename)
        else:  # For train
            dst_img_path = os.path.join(args.output_train_img_dir, img_filename)
            dst_mask_path = os.path.join(args.output_train_mask_dir, img_filename)

        # Copy the files to the appropriate destination
        shutil.copy2(src_img_path, dst_img_path)
        shutil.copy2(src_mask_path, dst_mask_path)
        print(f"Copied {src_img_path} to {dst_img_path}")
        print(f"Copied {src_mask_path} to {dst_mask_path}")

if __name__ == "__main__":
    args = parse_args()
    distribute_files(args)