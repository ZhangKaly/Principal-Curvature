start = 0
num = 10
content = ""
for i in range(num):
    end = start + int(5000/num)
    content += f"cd /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/sheaf-neural-network/src; /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/.conda_envs/py39/bin/python ellipsoid.py --eval_id_start {start} --eval_id_end {end} --tau_ratio 4 --if_generate_ellipsoid_cloud False  --num_points 5000 --file_path ellipsoid --seed 42 --epsilon_PCA 0.2 --tau_ratio 4 --a 1.5 --b 0.9 --c 0.9\n"
    start = end

with open('job_ellipsoid.txt', 'w') as f:
    # Write the string to the file
    f.write(content)