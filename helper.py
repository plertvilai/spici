""" """
# Standard dist imports
import logging

# Third party imports
import pandas as pd


# Project level imports

# Module level constants

class SPCDataTransformer():
    """Helper class"""

    def __init__(self, data, csv_fname=None, classes=None, logger=None):
        # Initialize Logging
        self.logger = logger if logger else logging.getLogger('spcdata')

        if csv_fname:
            self.csv_fname = csv_fname
            self.dataset = pd.read_csv(csv_fname)

        if classes:
            self.cls2idx, self.idx2cls = self.get_classes(classes)

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

    def get_predictions(self, pred_col='pred', verbose=False, decode=False, write=False):
        """Get the prediction distribution"""
        if self.idx2cls and decode:
            self.dataset[pred_col] = self.dataset[pred_col].map(self.idx2cls)

        if verbose:
            print(self.dataset[pred_col].value_counts())
        self.pred = self.dataset[pred_col].value_counts().to_dict()

        if write:
            # writes it in SPC formatting
            with open('data/predictions.txt', 'w') as f:
                for idx, row in self.dataset[['image_id', 'pred']].iterrows():
                    f.write(row['image_id'] + ',' + str(row['pred']) + '\n')
            f.close()
            print('Finished writing')

        return self.pred

    def _export_labels(self):
        pass

    def get_classes(self, filename):
        """Set class2idx, idx2class encoding/decoding dictionaries"""
        class_list = SPCDataTransformer._parse_classes(filename)
        cls2idx = {i: idx for idx, i in enumerate(sorted(class_list))}
        idx2cls = {idx: i for idx, i in enumerate(sorted(class_list))}
        return cls2idx, idx2cls

    @staticmethod
    def _parse_classes(filename):
        """Parse MODE_data.info file"""
        lbs_all_classes = []
        with open(filename, 'r') as f:
            label_counts = f.readlines()
        label_counts = label_counts[:-1]
        for i in label_counts:
            class_counts = i.strip()
            class_counts = class_counts.split()
            class_name = ''
            for j in class_counts:
                if not j.isdigit():
                    class_name += (' ' + j)
            class_name = class_name.strip()
            lbs_all_classes.append(class_name)
        return lbs_all_classes
