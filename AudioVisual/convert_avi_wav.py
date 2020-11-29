import os
import subprocess

orig_dir = r'..\datasets\enterface/original/'
wav_dir = r'..\datasets\enterface/wav/'

# print(os.system("ls " + orig_dir))

for path, dirs, files in os.walk(orig_dir):
    if files:
        for file in files:
            name, ext = file.split('.')
            if ext == 'avi':

                new_dir = os.path.join(wav_dir, path[len(orig_dir):])
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)



                orig_file = os.path.join(path, file)
                wav_file = os.path.join(new_dir, name + '.wav')
                # print(orig_file)
                # print(wav_file)
                command = "ffmpeg -i " + '"' + orig_file + '" "' + wav_file + '"'
                # print(command)

                os.system(command)

