import pandas as pd
import numpy as np
import yaml
import pickle

def load_config(path):
    # load the config file
    with open(path, 'r') as file:
          return yaml.safe_load(file)['feature_engineering']
 
def apply_feature_engineering(df, config):
 # apply temporal and fourier transform features
    df['date'] = pd.to_datetime(df[config['date_column']])
    processed_df = pd.DataFrame()

    for site_id in df[config['grouping_column']].unique():
        site_df = df[df[config['grouping_column']] == site_id].copy()

        # Extract temporal features
        for feature in config['temporal_features']:
            site_df[feature] = getattr(site_df['date'].dt, feature)

        # Apply fourier transforms
        for ft in config['fourier_features']:
            period = config['output_period']
            if 'sin' in ft:
                site_df[ft] = np.sin(2 * np.pi * np.arange(len(site_df)) / period)
            elif 'cos' in ft:
                site_df[ft] = np.cos(2 * np.pi * np.arange(len(site_df)) / period)



        processed_df = pd.concat([processed_df, site_df], ignore_index=True)

    return processed_df

def process_file(input_path, output_path, config):
    df = pd.read_csv(input_path)
    processed_df = apply_feature_engineering(df, config)
    processed_df.to_csv(output_path, index=False)
    # save processed data as a pickle file separate from the csv file
    processed_df.to_pickle(output_path.replace('.csv', '.pkl'))
    


if __name__ == '__main__':
    config_path = 'configs/feature_engineering.yml'
    config = load_config(config_path)

    # process the training data
    process_file(
        config['data_paths']['input_test'],
        config['data_paths']['output_test'],
        config
    )
    
    # Process the test data
    process_file(
        config['data_paths']['input_train'],
        config['data_paths']['output_train'],
        config
    )



