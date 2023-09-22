start = 0
num = 10
tau_ratio = 4

content = ""
for i in range(num):
    end = start + int(5000/num)
    content += f"cd /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/sheaf-neural-network/src; /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/.conda_envs/py39/bin/python torus_new.py --eval_id_start {start} --eval_id_end {end} --tau_ratio {tau_ratio} --if_generate_torus_cloud False  --num_points 5000\n"
    start = end

with open('job_torus_rerun.txt', 'w') as f:
    # Write the string to the file
    f.write(content)