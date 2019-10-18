""" """
# Standard dist imports
import logging

# Third party imports
import pandas as pd


# Project level imports

# Module level constants

class SPCDataTransformer():
    """Helper class"""

    def __init__(self, data, logger=None):
        assert isinstance(data, pd.DataFrame)
        # Initialize Logging
        self.logger = logger if logger else logging.getLogger('spcdata')

        self.data = data

        # TODO accept this as parameter initialization. Need to keep constant
        # record of accepted classes for training and prediction
        self.classes = {'Non-Prorocentrum': 0,
                        'Unidentified': 0,
                        'Prorocentrum': 1}

    def transform(self, image_dir=None, clean=False):
        """Transform for training/prediction data"""

        # Clean labels
        data = self.data
        if clean:
            self.logger.info('Cleaning labels...')
            data = self._clean_labels(data=data)

        # Extract timestamps
        self.logger.info('Extracting timestamps...')
        timed_df = self._extract_timestamp(data=data)

        # Create abs paths
        if image_dir:
            self.logger.info('Creating abs paths for images...')
            appd_df = self._append_img_dir(data=timed_df, image_dir=image_dir)

        return appd_df

    def _clean_labels(self, data, label_col='user_labels'):
        """Clean up labels given tag assignment
        - specially configured for prorocentrum data atm
        """

        def clean_up(lbl, tag):
            if not lbl:
                if not tag:
                    return 'Unidentified'
                else:
                    return 'Non-Prorocentrum'
            elif 'False Prorocentrum' in lbl or \
                    'Prorocentrum_false_positiveal' in lbl:
                return 'Non-Prorocentrum'
            elif lbl[0] in ['Prorocentrum', 'False Non-Prorocentrum']:
                return lbl[0]
            else:
                return 'Non-Prorocentrum'

        df = data.copy()
        df[label_col] = df.apply(lambda x: clean_up(x[label_col],
                                                    x['tags']), axis=1)
        df['label'] = df[label_col].map(self.classes)
        return df

    def _extract_timestamp(self, data, time_col='image_timestamp', drop=True):
        """Extracts date and time from image timestamps"""

        def parse_timestamp(x):
            """Parses example fmt: 'Sat Dec 23 10:01:24 2017 PST' """
            return ' '.join(x.split(' ')[1:-1])

        df = data.copy()
        pre = 'tmp-'
        df[pre + time_col] = df[time_col].apply(parse_timestamp)
        df[pre + time_col] = pd.to_datetime(df[pre + time_col], infer_datetime_format=True)
        df['image_date'] = df[pre + time_col].dt.date
        df['image_time'] = df[pre + time_col].dt.time
        if drop:
            df = df.drop(columns=[pre + time_col])

        return df

    def _append_img_dir(self, data, image_dir, image_col='images'):
        """Creates absolute path to each image"""
        import os
        df = data.copy()
        df[image_col] = df['image_url'].apply(lambda x: os.path.join(
            image_dir, os.path.basename(x) + '.jpg'))
        return df
