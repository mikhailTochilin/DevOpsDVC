import os
import yaml

with open('conf/config.yaml', 'r') as file:
    prime_service = yaml.safe_load(file)

executable_model = prime_service["exp_dir"]

if not os.path.exists("enhanced"):
    os.makedirs("enhanced")

save_dir = os.path.join("enhanced", os.path.basename(executable_model))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# print(os.path.basename(executable_model))

checkpoint_file = os.path.join(executable_model, "best.th")
noisy_dir = prime_service["dset"]["noisy_dir"]
os.system(f"python3.8 -m enhancer_toch.enhance --model_path={checkpoint_file} --noisy_dir={noisy_dir} --out_dir={save_dir}")