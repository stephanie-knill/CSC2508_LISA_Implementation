import osOPTIONS = {}"""Parameters for Learned Index Structure for Spatial Data (LISA)Args:    keys_init:          Keys in the initial dataset    page_size:          Predefined page size"""OPTIONS['keys_init'] = 0OPTIONS['page_size'] = 0# Datasettiny_imagenet_dataset = {'name' : 'tiny_imagenet',                         'data_folder' : os.path.join('.', 'data', 'tiny_imagenet'),                         'num_channels' : 3}imis_3months_dataset = {'name' : 'imis_3months',                        'data_folder' : os.path.join('.', 'data', 'imis_3months'),                        'file_name' : 'original.txt'}imis_3months_dataset_subset1000 = {'name' : 'imis_3months_subset1000',                                    'data_folder' : os.path.join('.', 'data', 'imis_3months'),                                    'file_name' : 'subset1000.txt'}                                    imis_3months_dataset_subset10000 = {'name' : 'imis_3months_subset10000',                                    'data_folder' : os.path.join('.', 'data', 'imis_3months'),                                    'file_name' : 'subset10000.txt'}OPTIONS['datasets'] = imis_3months_dataset_subset1000