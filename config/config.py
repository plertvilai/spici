""" configuration file to store constants
"""
from __future__ import absolute_import

# Standard dist imports
import os
from pathlib import Path

# Project level imports
from constants.genericconstants import GenericConstants as CONST

# Module level constants
DEFAULT_ENV = CONST.DEV_ENV  # SVCL local environment
# DEFAULT_ENV = CONST.PROD_ENV  # SPC Lab machine
PROJECT_DIR = Path(__file__).resolve().parents[1]


class Environment():
    """Sets up Environment Variables, given the DEFAULT ENV
    
    Set up all directory-related variables under here
    """

    def __init__(self, env_type=None):
        """Initializes Environment()
        
        Given the environment type (dev or prod), it sets up the model, data, and db direcotry related stuff. All variables need to be initialized with a string formatting so these are all RELATIVE PATHS
        """
        if env_type == CONST.DEV_ENV:
            # Model and database items
            root = '/data6/phytoplankton-db'
            self.model_dir = '/data6/lekevin/hab-master/hab_rnd/hab-ml/experiments/hab_model_v1:20191023/'
            self.data_dir = os.path.join(root, 'hab_in_situ/images', '{}')
            self.meta_dir = os.path.join(root, 'csv')
            self.hab_ml_main = os.path.join(PROJECT_DIR, 'hab_ml', '{}')
            # SPC items
            self.login_url = 'http://spc.ucsd.edu/data/admin/?next=/data/admin'

        elif env_type == CONST.PROD_ENV:
            self.model_dir = '/data6/plankton_test_db_new/model/20191023/00:51:01/'
            self.data_dir = os.path.join(PROJECT_DIR, 'images', '{}')
            self.hab_ml_main = os.path.join(PROJECT_DIR, 'hab_ml', '{}')


class Config(Environment):
    """Default Configs for training and inference
    After initializing instance of Config, user can import configurations as a
    state dictionary into other files. User can also add additional
    configuration items by initializing them below.
    Example for importing and using `opt`:
    config.py
        >> opt = Config()
    main.py
        >> from config import opt
        >> lr = opt.lr
        
    NOTE: all path related configurations should be set up in the Environment() class above to avoid issues with developing on a person vs production environment.

    """
    # SPC Submission dictionary
    account_info = {'username': 'kevin',
                    'password': 'ceratium'}
    label_instance_name = 'hab_24'
    tag = 'hab_24'
    is_machine = True

    def __init__(self, env_type):
        super().__init__(env_type)

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        # print('======user config========')
        # pprint(self._state_dict())
        # print('==========end============')

    def _state_dict(self):
        """Return current configuration state
        Allows user to view current state of the configurations
        Example:
        >>  from config import opt
        >> print(opt._state_dict())
        """
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


def set_config(**kwargs):
    """ Set configuration to train/test model
    Able to set configurations dynamically without changing fixed value
    within Config initialization. Keyword arguments in here will overwrite
    preset configurations under `Config()`.
    Example:
    Below is an example for changing the print frequency of the loss and
    accuracy logs.
    >> opt = set_config(print_freq=50) # Default print_freq=10
    >> ...
    >> model, meter = train(trainer=music_trainer, data_loader=data_loader,
                            print_freq=opt.print_freq) # PASSED HERE
    """
    opt._parse(kwargs)
    return opt


opt = Config(DEFAULT_ENV)
