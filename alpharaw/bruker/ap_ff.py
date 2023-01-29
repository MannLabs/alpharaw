import pandas as pd
import numpy as np
import sqlalchemy as db

import subprocess
import os
import platform

from tqdm import tqdm


def extract_bruker(file:str, base_dir:str = "ext/bruker/FF", config:str = "proteomics_4d.config"):
    """Call Bruker Feautre Finder via subprocess.

    Args:
        file (str): Filename for feature finding.
        base_dir (str, optional): Base dir where the feature finder is stored.. Defaults to "ext/bruker/FF".
        config (str, optional): Config file for feature finder. Defaults to "proteomics_4d.config".

    Raises:
        NotImplementedError: Unsupported operating system.
        FileNotFoundError: Feature finder not found.
        FileNotFoundError: Config file not found.
        FileNotFoundError: Feature file not found.
    """    
    feature_path = file + '/'+ os.path.split(file)[-1] + '.features'

    base_dir = os.path.join(os.path.dirname(__file__), base_dir)

    operating_system = platform.system()

    if operating_system == 'Linux':
        ff_dir = os.path.join(base_dir, 'linux64','uff-cmdline2')
        print('Using Linux FF')
    elif operating_system == 'Windows':
        ff_dir = os.path.join(base_dir, 'win64','uff-cmdline2.exe')
        print('Using Windows FF')
    else:
        raise NotImplementedError(f"System {operating_system} not supported.")

    if os.path.exists(feature_path):
        return feature_path
    else:
        if not os.path.isfile(ff_dir):
            raise FileNotFoundError(f'Bruker feature finder cmd not found here {ff_dir}.')

        config_path = base_dir + '/'+ config

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f'Config file not found here {config_path}.')

        if operating_system == 'Windows':
            FF_parameters = [ff_dir,'--ff 4d',f'--readconfig "{config_path}"', f'--analysisDirectory "{file}"']

            process = subprocess.Popen(' '.join(FF_parameters), stdout=subprocess.PIPE)
            for line in iter(process.stdout.readline, b''):
                logtxt = line.decode('utf8')
                print(logtxt[48:].rstrip()) #Remove logging info from FF
        elif operating_system == 'Linux':
            FF_parameters = [
                ff_dir,
                '--ff',
                '4d',
                '--readconfig',
                config_path,
                '--analysisDirectory',
                file
            ]
            process = subprocess.run(FF_parameters, stdout=subprocess.PIPE)

        if os.path.exists(feature_path):
            return feature_path
        else:
            raise FileNotFoundError(f"Feature file {feature_path} does not exist.")

def convert_bruker(feature_path:str)->pd.DataFrame:
    """Reads feature table and converts to feature table to be used with AlphaPept.

    Args:
        feature_path (str): Path to the feature file from Bruker FF (.features-file).

    Returns:
        pd.DataFrame: DataFrame containing features information.
    """
    engine_featurefile = db.create_engine('sqlite:///{}'.format(feature_path))
    feature_table = pd.read_sql_table('LcTimsMsFeature', engine_featurefile)
    feature_cluster_mapping = pd.read_sql_table('FeatureClusterMapping', engine_featurefile)

    # feature_table['Mass'] = feature_table['MZ'].values * feature_table['Charge'].values - feature_table['Charge'].values*M_PROTON
    feature_table = feature_table.rename(columns={
        "MZ": "mz","Mass": "mass", "RT": "rt_apex", 
        "RT_lower":"rt_start", "RT_upper":"rt_end", 
        "Mobility": "mobility", "Mobility_lower": "mobility_lower", 
        "Mobility_upper": "mobility_upper", "Charge":"charge",
        "Intensity":'ms1_int_sum_apex',"ClusterCount":'n_isotopes'
    })
    feature_table['rt_apex'] = feature_table['rt_apex']/60
    feature_table['rt_start'] = feature_table['rt_start']/60
    feature_table['rt_end'] = feature_table['rt_end']/60
    
    feature_cluster_mapping = feature_cluster_mapping.rename(columns={
        "FeatureId": "feature_id", "ClusterId": "cluster_id", 
        "Monoisotopic": "monoisotopic", "Intensity": "ms1_int_sum_apex"
    })
    
    return feature_table, feature_cluster_mapping


def map_bruker(feature_path:str, feature_table:pd.DataFrame, query_data:dict)->pd.DataFrame:
    """Map Ms1 to Ms2 via Table FeaturePrecursorMapping from Bruker FF.

    Args:
        feature_path (str): Path to the feature file from Bruker FF (.features-file).
        feature_table (pd.DataFrame): Pandas DataFrame containing the features.
        query_data (dict): Data structure containing the query data.

    Returns:
        pd.DataFrame: DataFrame containing features information.
    """
    engine_featurefile = db.create_engine('sqlite:///{}'.format(feature_path))

    mapping = pd.read_sql_table('FeaturePrecursorMapping', engine_featurefile)
    mapping = mapping.set_index('PrecursorId')
    feature_table= feature_table.set_index('Id')


    query_prec_id = query_data['prec_id']

    #Now look up the feature for each precursor

    mass_matched = []
    mz_matched = []
    rt_matched = []
    query_idx = []
    f_idx = []

    for idx, prec_id in tqdm(enumerate(query_prec_id)):
        try:
            f_id = mapping.loc[prec_id]['FeatureId']
            all_matches = feature_table.loc[f_id]
            if type(f_id) == np.int64:
                match = all_matches
                mz_matched.append(match['mz'])
                rt_matched.append(match['rt_apex'])
                mass_matched.append(match['mass'])
                query_idx.append(idx)
                f_idx.append(match['FeatureId'])

            else:
                for k in range(len(all_matches)):
                    match = all_matches.iloc[k]
                    mz_matched.append(match['mz'])
                    rt_matched.append(match['rt_apex'])
                    mass_matched.append(match['mass'])
                    query_idx.append(idx)
                    f_idx.append(match['FeatureId'])

        except KeyError:
            pass

    features = pd.DataFrame(np.array([mass_matched, mz_matched, rt_matched, query_idx, f_idx]).T, columns = ['mass_matched', 'mz_matched', 'rt_matched', 'query_idx', 'feature_idx'])

    features['query_idx'] = features['query_idx'].astype('int')

    return features