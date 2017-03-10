class Writer:
    def write(self, config):
        if 'logfile' in config:
            for key in config:
                print("key: ", key, " value: ", config[key])


