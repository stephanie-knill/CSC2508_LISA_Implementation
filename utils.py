import os
import math
import numpy as np
import pandas

# Modules
import parameters


def load_dataset():
    """
    Load locally stored dataset as specified in parameters.OPTIONS dictionary.
    
    Returns:
        initial_dataset:        Randomly select 50% of points
        (extra_dataset:          Randomly select other 50% of points -- Removed)

    """
    
    try:
        df = load_imis_3months()
    except:
        ValueError("Invalid dataset name %s specified in parameters.py" %parameters.OPTIONS['datasets']['name'])
    
    # TODO: uncomment if wish to also implement insertion/deletion
    # Select 50% of point as initial_dataset I
    # n = len(df)
    # shuffled = df.sample(n)
    # initial_dataset = shuffled.head(math.floor(n/2))
        
    # # Select 50% of points as extra_dataset E
    # extra_dataset = shuffled.tail(math.floor(n/2))
    
    # # Reorder dataset + reset index
    # initial_dataset = initial_dataset.sort_values(by=parameters.OPTIONS['datasets']['labels'][0]).reset_index(drop=True)
    # extra_dataset = extra_dataset.sort_values(by=parameters.OPTIONS['datasets']['labels'][0]).reset_index(drop=True)

    initial_dataset = df.sort_values(by=parameters.OPTIONS['datasets']['labels'][0]).reset_index(drop=True)

    return initial_dataset#, extra_dataset
    

def load_imis_3months():
    """
    Load Imis-3months dataset (http://chorochronos.datastories.org/?q=content/imis-3months).
    
    Returns:
        df:     Panda dataframe of imis_3months dataset for lon and lat
    
    """
    # Load Textfile
    file_path = os.path.join(parameters.OPTIONS['datasets']['data_folder'], 
                             parameters.OPTIONS['datasets']['file_name'])
    try:  
        df = pandas.read_table(file_path, delim_whitespace=True, names=('lon', 'lat'))
    except NameError:
        print("Filepath %s specified in parameters.py invalid" %file_path)
    
    return df
    
"""
=================== Data Preprocessing ===================
"""
def remove_duplicates_imis():
    """
    Load Imis-3months dataset (http://chorochronos.datastories.org/?q=content/imis-3months).
    Each record is in the form "t, obj_id, lon, lat" in a large .txt file.
    
    Extract only the coordinates (lon, lat) and remove duplicates.
    Saves this locally as 'imis_3months.txt'
    
    Rounded to 9 decimal places (https://gis.stackexchange.com/questions/8650/measuring-accuracy-of-latitude-and-longitude)
    
    Returns:
        df:     Panda dataframe of imis_3months dataset for lon and lat

    """
    # Load Textfile
    file_path = os.path.join(parameters.OPTIONS['datasets']['data_folder'], 
                             'original.txt')    
    df = pandas.read_table(file_path, delim_whitespace=True, names=('t', 'obj_id','lon', 'lat'))
    
    # Remove Duplicates: remove t, obj_id
    df.drop(['t', 'obj_id'], axis = 1, inplace=True)
    df.round(9)
    df.drop_duplicates(inplace=True)
    
    # Save dataset locally
    file_name = os.path.join(parameters.OPTIONS['datasets']['data_folder'], 'imis_3months.txt')
    np.savetxt(file_name, df.values, fmt='%.9f')
    
    return df
    
    

def export_subset_imis(n):
    """
    Create a subset of the imis_3months dataset (100+ million unique points) and 
    save this locally as 'imis_3months_subsetN.txt'
     
    Args:
        n:      Size of subset
    """
    
    # Load Full Dataset
    file_path = os.path.join(parameters.OPTIONS['datasets']['data_folder'], 'imis_3months.txt')
    df = pandas.read_table(file_path, delim_whitespace=True, names=('lon', 'lat'))
    
    # Create Subset of size n
    subset = df.sample(n)
    
    # Save subset locally
    subset_file_name = os.path.join(parameters.OPTIONS['datasets']['data_folder'], 'imis_3months_'+'subset'+str(n)+'.txt')
    np.savetxt(subset_file_name, subset.values, fmt='%.9f')