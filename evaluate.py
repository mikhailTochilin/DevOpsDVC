import os
import yaml

with open('conf/config.yaml', 'r') as file:
    prime_service = yaml.safe_load(file)

executable_model = prime_service["exp_dir"]
checkpoint_file = os.path.join(executable_model, "best.th")
jsons_dir = prime_service["dset"]["test"]



os.system(f"python3.8 -m enhancer_toch.evaluate --model_path={checkpoint_file} --data_dir={jsons_dir}")
