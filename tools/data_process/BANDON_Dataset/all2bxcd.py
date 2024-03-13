import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-all-dir",  default="/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_tmp.txt")
    parser.add_argument("--output-bx-dir",  default="/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_bx_tmp.txt")
    parser.add_argument("--output-cd-dir",  default="/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_cd_tmp.txt")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    all_path = args.input_all_dir
    bx_path = args.output_bx_dir
    cd_path = args.output_cd_dir
    bx = []
    cd_list = []

    with open(bx_path, 'w') as fbx:
        with open(cd_path, 'w') as fcd:
            with open(all_path, 'r') as fr:
                for line in fr.readlines():
                    a, b, cd, label_a, label_b = line.strip().split('\t')
                    if cd not in cd_list:
                        fcd.write(f'{a}\t{b}\t{cd}\t**\t**\n')
                    if a not in bx:
                        bx.append(a)
                        fbx.write(f'{a}\t**\t**\t{label_a}\t**\n')
                    if b not in bx:
                        bx.append(b)
                        fbx.write(f'{b}\t**\t**\t{label_b}\t**\n')