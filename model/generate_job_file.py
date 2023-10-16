import configparser
import argparse
import pathlib
import os

# Set up argparse to parse the configuration filename
parser = argparse.ArgumentParser(description="Read configuration and generate job file.")
parser.add_argument("config", type=str, help="Path to the configuration .ini file")
args = parser.parse_args()

# Read from the specified configuration file
config = configparser.ConfigParser()
config.read(args.config)

# Extract values from the config file
start = int(config['SETTINGS']['start'])
num = int(config['SETTINGS']['num'])
total_n = int(config['SETTINGS']['total_n'])
tau_ratio = int(config['SETTINGS']['tau_ratio'])
if_generate_cloud = config['SETTINGS']['if_generate_cloud'].lower() == 'true'
noise = float(config['SETTINGS']['noise'])
file_path = config['SETTINGS']['file_path']
mfd_type = config['SETTINGS']['mfd_type']
a = float(config['SETTINGS']['a'])
b = float(config['SETTINGS']['b'])
c = float(config['SETTINGS']['c'])
r = float(config['SETTINGS']['r'])
R = float(config['SETTINGS']['R_outer'])
seed = int(config['SETTINGS']['seed'])
epsilon_PCA = float(config['SETTINGS']['epsilon_PCA'])
job_file = config['SETTINGS']['job_file']
wd = config['SETTINGS'].get('wd', )
pythond = config['SETTINGS'].get('pythond', '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/.conda_envs/py39/bin/python')
wd = os.getcwd()
# generate the job file
content = ""
for i in range(num):
    end = start + int(total_n/num)
    content += f"cd {wd}; {pythond} main.py --eval_id_start {start} --eval_id_end {end} --tau_ratio {tau_ratio} --if_generate_cloud {if_generate_cloud}  --num_points {total_n} --file_path {file_path} --manifold_type {mfd_type} --a {a} --b {b} --c {c} --r {r} --R {R} --seed {seed} --epsilon_PCA {epsilon_PCA} --tau_ratio {tau_ratio} --noise {noise}\n"
    start = end

with open(f'{job_file}.txt', 'w') as f:
    # Write the string to the file
    f.write(content)

content_init = f"cd {wd}; {pythond} main.py --eval_id_start {start} --eval_id_end {end} --tau_ratio {tau_ratio} --if_generate_cloud {if_generate_cloud}  --num_points {total_n} --file_path {file_path} --manifold_type {mfd_type} --a {a} --b {b} --c {c} --r {r} --R {R} --seed {seed} --epsilon_PCA {epsilon_PCA} --tau_ratio {tau_ratio} --init True --noise {noise}\n"

with open(f'{job_file}_init.sh', 'w') as f:
    # Write the string to the file
    f.write(content_init)