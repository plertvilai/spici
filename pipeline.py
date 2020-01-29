"""Pipeline Initialization for HAB-ML on Pier Deployment"""
# Standard dist imports
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve()
sys.path.insert(0, PROJECT_DIR.parents[0])
sys.path.insert(0, PROJECT_DIR.parents[1])
sys.path.insert(0, PROJECT_DIR.parents[2])

# Third party imports

# Project level imports
from config.config import opt
from constants.genericconstants import SPCConstants as SPCCONST
from spcserver import SPCServer
from helper import SPCDataTransformer
from hab_ml.data.prepare_db import DatasetGenerator
from hab_ml.utils.logger import Logger
from hab_ml.utils.constants import Constants as MLCONST

# Module level constants

class Pipeline():
    """Pipeline Instance for HAB-ML

    Operations consist of the following:
        - pull images from the spc website
        - running predictions on the images
        - push the predicted labels to the spc website

    """

    def __init__(self):
        self.pre = 'hab_in_situ_'
        self.data_file = None
        self.output_meta_fname = None
        self.predictions_csv = None
        self.classes_txt = None
        self.machine_name = None
        self.spc = SPCServer(daylight_savings=True)

    def run_app(self, textfile, data_dir, download=False):
        # download images
        test_date = 'summer2019'
        textfile = 'DB/meta/{}/time_period.txt'.format(test_date)
        # RUN_APP (DEPLOYMENT) --> steps 1, 4, 5, 6
        # PULL_AND_PREPARE --> steps 1 OR 2 || hab_ml.main.py (fine_tuning)
        # PUSH --> steps 4, 5, 6
        self.pull(textfile=textfile, data_dir=opt.data_dir.format(test_date), download=download)
        self.predict()
        self.push()


    def predict(self):
        """Run model prediction"""
        gpu = 4
        hab_ml_main = opt.hab_ml_main.format('main.py')
        self.machine_name = os.path.basename(opt.model_dir)
        cmd = "CUDA_VISIBLE_DEVICES={} python {} --mode deploy --batch_size 128 --deploy_data {} --model_dir {}"
        cmd = cmd.format(gpu, hab_ml_main, self.data_file, opt.model_dir)
        os.system(cmd)

        # create prediction txt file and label txt file for uploading
        self.predictions_csv = self.data_file.strip('.csv') + '-predictions.csv'
        self.classes_txt = os.path.join(opt.model_dir, 'train_data.info')
        print('SUCCESS: Images predicted')

    def pull(self, textfile, data_dir, download=False):
        """Pull data from the spc website"""
        # initialize
        output_meta_fname = os.path.join(opt.meta_dir, self.pre +
                                         os.path.basename(data_dir) + '.csv')

        # download images
        self.spc.retrieve(textfile=textfile,
                          data_dir=data_dir,
                          output_csv_filename=output_meta_fname,
                          download=download)

        # initialize data file for model deployment
        self.data_file = output_meta_fname
        print('SUCCESS: Images pulled\n')

    def filtered_pull(self, data, hab_eval=True):
        fname = self.pre + data
        log_fname = os.path.join(opt.meta_dir, fname + '.log')
        Logger(log_fname, logging.INFO, False)
        logger = logging.getLogger('pull_data')
        Logger.section_break(title='SPICI')

        converted_date = datetime.strptime(data, '%Y%m%d').strftime('%Y-%m-%d')
        gen = DatasetGenerator(csv_file=opt.summer2019_csv, data_dir=opt.meta_dir, labels=None, dates=None)
        df = gen.data[gen.data['image_date'] == converted_date].reset_index(drop=True)

        logger.info('SUCCESS: Meta file generated')
        logger.info(f'Dates pulled: {converted_date}')
        logger.info(f'Dataset size: {df.shape[0]}')
        logger.info(f'Label Distribution\n{"-" * 30}\n{df.label.value_counts()}')

        if hab_eval:
            df = Pipeline()._reformat_lab_data(df)

        csv_fname = os.path.join(opt.meta_dir, fname + '.csv')
        df.to_csv(csv_fname, index=False)
        logger.info(f'Dataset saved as {csv_fname}')

    @staticmethod
    def _reformat_lab_data(data):
        df = data.copy()
        logger = logging.getLogger('_reformat_lab_data')
        logger.debug('Reformatting dataset...\n')

        hab_species = open('/data6/phytoplankton-db/hab.txt', 'r').read().splitlines()[
                      :-1]

        def label_mapping(x):
            if x in hab_species:
                return x
            else:
                return 'Other'

        df[MLCONST.LABEL] = df[MLCONST.LABEL].apply(label_mapping)
        logger.info(f'HAB Lbl Distribution\n{"-" * 30}\n'
                    f'{df[MLCONST.LABEL].value_counts()}')
        return df


    def push(self):
        """Push data up to the spc website"""
        self.spc.submit_dict[SPCCONST.LBL_SET_NAME] = opt.label_instance_name
        self.spc.submit_dict[SPCCONST.TAG] = opt.tag
        self.spc.submit_dict[SPCCONST.IS_MCHN] = opt.is_machine
        self.spc.submit_dict[SPCCONST.MCHN_NAME] = self.machine_name
        # todo incorporate this into the dataset_generator
        SPCDataTransformer(data=None, csv_fname=self.predictions_csv,
                           classes=self.classes_txt).get_predictions(write=True)
        #todo create export classes
        labels = 'data/labels.txt'
        predictions = 'data/predictions.txt'

        self.spc.upload(login_url=opt.login_url,
                        account_info=opt.account_info,
                        textfile=predictions,
                        label_file=labels)
        print('SUCCESS: Images pushed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPICI Pipeline')
    parser.add_argument('--run_app', action='store_true', help='Run entire pipeline')
    parser.add_argument('--pull', action='store_true', help='Pull SPC images')
    parser.add_argument('--filtered_pull', action='store_true', help='Pull filtered pre-downloaded SPC images')
    parser.add_argument('--push', action='store_true', help='Push predicted SPC images')
    parser.add_argument('--data', default=None, help='Dataset name to pull/push from. This is represented by the date')
    parser.add_argument('--download', '-d', action='store_true', help='Download SPC images if pulling')
    arg = parser.parse_args()

    # Initialize pipeline
    pip = Pipeline()
    # Run the entire pipeline
    if arg.run_app and arg.data:
        dataset_name = arg.data
        time_period_txtfile = 'DB/meta/{}/time_period.txt'.format(dataset_name)
        pip.run_app(textfile=time_period_txtfile, data_dir=opt.data_dir.format(dataset_name), download=arg.download)
    # Determine if pulling data, given the dataset_name (e.g. 20190530, 20190610, ...)
    elif arg.pull and arg.data:
        dataset_name = arg.data
        time_period_txtfile = 'DB/meta/{}/time_period.txt'.format(dataset_name)
        pip.pull(textfile=time_period_txtfile, data_dir=opt.data_dir.format(dataset_name), download=arg.download)
    elif arg.filtered_pull and arg.data:
        pip.filtered_pull(data=arg.data)
    # Push data to the spc website
    # elif arg.push and arg.data:
    #     dataset_name = arg.data
    #     pip.push()
