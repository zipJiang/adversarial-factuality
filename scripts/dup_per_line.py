'''
Duplicate each line to N times in a text file
'''

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input file path')
    parser.add_argument('output_file', type=str, help='Output file path')
    parser.add_argument('N', type=int, help='Number of times to duplicate each line')
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        lines = f.readlines()

    with open(args.output_file, 'w') as f:
        for line in lines:
            for _ in range(args.N):
                f.write(line)
