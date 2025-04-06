def reformat_lidar_file(input_path, output_path, block_size=362):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        values = f_in.read().replace('\n', '').split('\t')
        values = [v.strip() for v in values if v.strip() != '']

        for i in range(0, len(values), block_size):
            block = values[i:i + block_size]
            if len(block) == block_size:
                f_out.write('\t'.join(block) + '\n')

if __name__ == '__main__':
    reformat_lidar_file("dataset/LASER.txt", "dataset/LASER_processed.txt")