from definitions import LOGS_DIR
import os

def log_config(config):
    filename = 'README.txt'
    type = config['type']
    name = config['name']
    dir_to_create = LOGS_DIR + '/' + type + '/' + name
    result = ''
    for key in config:
        result += "key: " + str(key) + "\tvalue: " + str(config[key]) + "\n"

    if not os.path.exists(dir_to_create):
        raise FileNotFoundError('Can not write because no directory is created')

    with open(dir_to_create+'/'+filename, "w+") as f:
        f.write(result)



