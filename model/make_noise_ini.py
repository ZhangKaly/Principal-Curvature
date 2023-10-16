import pathlib
import numpy as np
ini_folder = "torus_noise_configs"
pathlib.Path(ini_folder).mkdir(parents=True, exist_ok=True)
noises = np.round(np.arange(10) * 0.1, 2)
bash_str = ""
init_str = ""
dsq_str = ""
for noise in noises:
    ini_str = f"""[SETTINGS]
start = 0
num = 10
total_n = 5000
tau_ratio = 4
if_generate_cloud = False
noise = {noise}
file_path = torus_noise_{noise}
mfd_type = torus
a = 1.5
b = 0.9
c = 0.9
r = 0.375
R_outer = 1
seed = 42
epsilon_PCA = 0.2
job_file = job_torus_{noise}
"""
    bash_str += f"python generate_job_file.py {ini_folder}/torus_{noise}.ini\n"
    dsq_str += f"bash dsq_submit.sh job_torus_{noise}\n"
    init_str += f"bash /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/sheaf-neural-network/model/job_torus_{noise}_init.sh\n"
    with open(f'{ini_folder}/torus_{noise}.ini', 'w') as f:
        # Write the string to the file
        f.write(ini_str)
with open(f'torus_make_jobs.sh', 'w') as f:
    # Write the string to the file
    f.write(bash_str)
with open(f'torus_submit_jobs.sh', 'w') as f:
    # Write the string to the file
    f.write(dsq_str)
with open(f'torus_init_all.txt', 'w') as f:
    # Write the string to the file
    f.write(init_str)