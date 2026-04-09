import os, sys
import shutil

def move_files(input_obj_path, output_obj_path):
    os.makedirs(output_obj_path, exist_ok=True)

    for file in os.listdir(input_obj_path):
        if file.endswith('.json'):
            shutil.copy2(os.path.join(input_obj_path, file), os.path.join(output_obj_path, file))
        



if __name__ == '__main__':
    input_path = '/Users/xinyuesheng/Documents/astro_projects/data/image_sets_jsons'
    output_path = '/Users/xinyuesheng/Documents/astro_projects/data/image_sets_v3'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in os.listdir(input_path):
        if file.startswith('ZTF'):
            input_obj_path = os.path.join(input_path, file)
            output_obj_path = os.path.join(output_path, file)
            move_files(input_obj_path, output_obj_path)