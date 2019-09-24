""" Script to download and upload images to spc.ucsd.edu

With this script you can download images from the spc.ucsd.edu website provided by
the configuration settings set in the text file. You could also upload images with
machine labeled annotations.

Author: Kevin Le
contact: kevin.le@gmail.com

"""
from __future__ import print_function, division

import json
import os
import numpy as np
import datetime
from lxml import html
import urllib.request as urllib2
import http.cookiejar
import pandas as pd
import asyncio
import aiohttp
import aiofiles
from spc.spctransformer import SPCDataTransformer

CAMERAS = ['SPC2' , 'SPCP2', 'SPC-BIG']
IMG_PARAM = ['image_filename', 'image_id', 'user_labels', 'image_timestamp', 'tags']
IMGS_PER_PAGE = 500
page_sem = asyncio.Semaphore(20)
sem = asyncio.Semaphore(20)
file_sem = asyncio.Semaphore(512)

class SPCServer(object):
    """ Represents a wrapper class for accessing the spc.ucsd.edu pipeline

    A 'SPCServer' can be used to represent as an input/output pipeline,
    where it could be a collection of elements as inputs or a collection
    of images as outputs from the website.

    """

    def __init__(self):
        """ Creates a new SPCServer.

        """

        # Dates initialized for uploading purposes
        date = datetime.datetime.now().strftime('%s')
        self.date_submitted = str(int(date)*1000)
        self.date_started = str((int(date)*1000)-500)

        # Data for uploading images
        self.submit_dict = {"label": "",                            # 'str' DENOTING LABEL ASSOCIATED WITH IMAGES
                            "tag": "",                              # 'str' DENOTING TAG FOR FILTERING
                            "image_ids": [],                        # 'list' REPRESENTING COLLECTION OF IMAGE IDS
                            "name": "",                             # 'str' DENOTING NAME OF LABELING INSTANCE
                            "machine_name":"",                      # 'str' DENOTING MACHINE NAME
                            "started":self.date_started,            # PREFILLED START DATE
                            "submitted": self.date_submitted,       # PREFILLED SUBMISSION DATE
                            "is_machine": False,                    # 'bool' FLAGGING MACHINE OR HUMAN
                            "query_path": "",                       # MISC.
                            "notes": ""                             # MISC.
        }

        # Url links stored into list from _build_url()
        self.url_list = []

        # Camera resolution configuration
        self.cam_res = [7.38 / 1000, 0.738 / 1000]

        # Set to true if synchronous image writing is needed (i.e. on faster hard drives like SSDs)
        self.sync_write = False



    def upload(self, login_url, account_info, textfile, label_file):
        ''' Uploads submit dictionary to initialized url from prep_for_upload()

        Args:
            login_url: A 'str', representing the administrative link to spc.ucsd.edu
            account_info: A 'dict', containing the 'username' and 'password' to access the spc pipeline
            textfile: A 'str' representing the path to a textfile of images and ground truth/predicted labels (machine or human)

        Usage:
            ==> account_info = {'username':'kevin', 'password': 'plankt0n'}
            ==> login_url = 'http://planktivore.ucsd.edu/caymans_data/admin/?next=/data/admin'
            ==> predictions = 'examples/prorocentrum/predictions.txt'
            ==> label_file = 'examples/prorocentrum/labels.txt'
            ==> spc.upload(login_url=login_url,
                   account_info=account_info,
                   textfile=predictions,
                   label_file=label_file)
        :return:
        '''
        # Access server pipeline using account credentials
        self.label_file = label_file

        self._access_server(login_url=login_url, account_info=account_info)

        if not os.path.exists(textfile):
            raise ValueError("{} does not exist".format(textfile))

        # Gather images and labels
        grouped_labels = self._read_text_file(textfile=textfile)

        # Check for valid configuration for uploading
        self._assert_submission_dict()

        # Image limit for uploading images in a session
        maximum_images = 15000

        # Loop over number of labels
        total_labels = 0
        for label in self.labels:

            # Submit label
            self.submit_dict['label'] = label

            # Group images based on labels
            image_ids = grouped_labels.get_group(label)['image'].tolist()

            # Number of images uploaded
            image_size = len(image_ids)

            # Check for necessary batching to avoid upload limit
            if image_size > maximum_images:
                for i in range(0, image_size, maximum_images):
                    batch = image_ids[i:i+maximum_images]
                    self.submit_dict['image_ids'] = batch
                    self._push_submission_dict(label=label)

            else:
                self.submit_dict['image_ids'] = image_ids
                self._push_submission_dict(label=label)

            total_labels += image_size
            print("Uploaded {} {} images".format(image_size, label))
        print("Uploaded {} total labels for {}".format(total_labels, self.labels))

    async def extract_img_list(self, session, server, url, download):
        """ Retrieve raw img urls from paginated responses """
        async with sem:
            loop = asyncio.get_event_loop()
            async with session.get(url) as resp:
                json_data = await resp.json()
                img_dicts = json_data['image_data']['results']
                img_urls = [i['image_url'] for i in img_dicts]
                print("Number of images: {}".format(len(img_urls)))
                subsubtasks = []
                df = pd.DataFrame(img_dicts)
                if download:
                    for i in img_urls:
                        srcpath = self.imgurl.format(server, i)
                        destpath = self.output_dir.format(os.path.basename(i))
                        subsubtasks.append(loop.create_task(self.fetch(session, srcpath, destpath)))
                    await asyncio.gather(*subsubtasks)
                return df

    async def get_paginated_list(self, session, server, url, download):
        """ Retrieve paginated lists -> raw img urls from main urls """
        async with page_sem:
            loop = asyncio.get_event_loop()
            url = url.format(IMGS_PER_PAGE)
            # create a shorter url with one result per page just to extract the image count
            short_url = url.format(1)
            async with session.get(short_url) as resp:
                json_data = await resp.json()
                num_results = json_data['image_data']['count']
            print("Number of results: {}".format(num_results))
            # dumb ceiling function
            num_pgs = int((num_results - 1) / IMGS_PER_PAGE) + 1
            subtasks = []
            for i in range(1, num_pgs + 1):
                url_paginated = url + '?page={}'.format(i)
                subtasks.append(loop.create_task(self.extract_img_list(session, server, url_paginated, download)))
            dfs = await asyncio.gather(*subtasks)
            return pd.concat(dfs, ignore_index=True)

    async def async_e2e_dl(self, urls, download):
        """ End-to-end asynchronous retrieval of images """
        loop = asyncio.get_event_loop()
        async with aiohttp.ClientSession() as session:
            tasks = []
            print("Number of urls: {}".format(len(urls)))
            for i in urls:
                if 'planktivore.ucsd.edu' in i:
                    server = 'planktivore'
                else:
                    server = 'spc'
                tasks.append(loop.create_task(self.get_paginated_list(session, server, i, download)))
            dfs = await asyncio.gather(*tasks)
            return pd.concat(dfs, ignore_index=True)

    def retrieve(self, textfile, output_dir, output_csv_filename, download=False):
        """Retrieves images from url and outputs images and meta data to desired output dir and filename respectively

        Usage:
        ==> spc = SPCServer()
        ==> spc.retrieve(textfile='examples/prorocentrum/time_period.txt',
                        output_dir='examples/proroentrum/images',
                        output_csv_filename='examples/prorocentrum/meta_data.csv',
                        download=True)

        :param textfile: 'str' representing path to text file for parsing download configurations
        :param output_dir: 'str' representing path to desired output directory for downloaded images
        :param output_csv_filename: 'str' representing where to output meta csv file
        :param download: 'bool' to flag downloading option
        :return:
        """
        time_init = datetime.datetime.now()

        # Output directory
        if output_dir is not None:
            self.output_dir = output_dir
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Desired output filename
            self.output_dir = os.path.join (self.output_dir, '{!s}.jpg')

        # Read text file and output dir
        self._prep_for_retrieval(textfile=textfile)

        # Source url
        self.inurl = 'http://{}.ucsd.edu{!s}'

        # Image url
        self.imgurl = 'http://{}.ucsd.edu{!s}.jpg'

        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(self.async_e2e_dl(self.data['url'], download))
        df = SPCDataTransformer(data=df).transform(image_dir=output_dir)
        df.to_csv(output_csv_filename, index=False)

        time_done = datetime.datetime.now()
        time_delta = time_done - time_init
        num_imgs = df.shape[0]
        if download:
            print('Downloaded {} images in {} ({} images/sec)'.format(num_imgs, time_delta, int(num_imgs / time_delta.total_seconds())))
        else:
            print('Downloaded data for {} images in {} ({} images/sec)'.format(num_imgs, time_delta, int(num_imgs / time_delta.total_seconds())))


    @staticmethod
    def map_labels(dataframe, label_file, mapped_column):
        """ Map enumerated labels into class names

        :param dataframe: 'pandas' dataframe containing image filenames and labels
        :param label_file: 'str' representing path to label text file
        :param mapped_column: 'str' representing column of dataframe to perform mapping
        :return: dataframe with new column of the mapped class names
        """
        with open(label_file, "r") as f:
            # Map labels to class names
            mapped_labels = {int(k): v for line in f for (k, v) in (line.strip ().split (None, 1),)}

        # Store into dataframe
        dataframe['class'] = dataframe[mapped_column].map(mapped_labels)

        return dataframe



    def _access_server(self, login_url, account_info):
        ''' Authorizes access to the server

        Usage:
            ==> account_info = {'username':'kevin', 'password': 'plankt0n'}
            ==> login_url = 'http://planktivore.ucsd.edu/caymans_data/admin/?next=/data/admin'
            ==> prep_for_upload(login_url=login_url)

        :return:
        '''
        assert isinstance(account_info, dict)
        assert isinstance(login_url, str)

        cj = http.cookiejar.CookieJar()
        self.opener = urllib2.build_opener(
            urllib2.HTTPCookieProcessor(cj),
            urllib2.HTTPHandler(debuglevel=1)
        )

        if login_url == None:
            login_url = 'http://planktivore.ucsd.edu/caymans_data/admin/?next=/data/admin'

        self.parsed_url = '/'.join(login_url.split('/')[:4])
        login_form = self.opener.open (login_url).read ()

        self.csrf_token = html.fromstring(login_form).xpath(
            '//input[@name="csrfmiddlewaretoken"]/@value')[0]

        params = json.dumps(account_info).encode('utf8')
        req = urllib2.Request('{}/rois/login_user'.format(self.parsed_url),
                               params, headers={'X-CSRFToken': str (self.csrf_token),
                                                'X-Requested-With': 'XMLHttpRequest',
                                                'User-agent': 'Mozilla/5.0',
                                                'Content-type': 'application/json'
                                                }
                               )
        self.resp = self.opener.open(req)
        print('Successfully logged in {}'.format(self.resp.read()))



    def _assert_submission_dict(self):
        """ Validate submission dictionary

        :return:
        """

        # Check for valid types and filled variables
        if not isinstance(self.submit_dict['image_ids'], list):
            raise TypeError("'image_ids' of 'submission_dict' must be a 'list'")
        if not isinstance(self.submit_dict['label'], str):
            raise TypeError("'label' of 'submit_dict' must be 'str'")
        if self.submit_dict['name'] == "":
            raise ValueError("'name' of 'submit_dict' must not be left empty")

        # Check for correct image id and correct if wrong
        if any(item.endswith("jpg") for item in self.submit_dict['image_ids']):
            self.submit_dict['image_ids'] = [item.replace('jpg', 'tif') for item in self.submit_dict['image_ids']]



    def _push_submission_dict(self, label, save=True, output_dir='labels'):
        """ Pushes data up to spc.ucsd.edu pipeline

        :param label: 'str' representing organism label
        :return:
        """

        #TODO Log errors with label
        try:
            if save:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                json_fname = os.path.join(output_dir, f'{label.replace(" ", "_")}.json')
                with open(json_fname, 'w', encoding='utf-8') as json_file:
                    self.submit_json = json.dump(self.submit_dict, json_file, indent=4)
                json_file.close()
                print(f'Saved as {json_fname}')

            self.submit_json = json.dumps(self.submit_dict).encode('utf-8')
            self.req1 = urllib2.Request('{}/rois/label_images'.format(self.parsed_url),
                           self.submit_json, headers={'X-CSRFToken': str(self.csrf_token),
                                                 'X-Requested-With': 'XMLHttpRequest',
                                                 'User-agent': 'Mozilla/5.0',
                                                 'Content-type': 'application/json'
                                                 }
                           )
            self.resp1 = self.opener.open(self.req1)
        except:
            print('{} labels written to error log'.format(label))
            error_log = open ('error_log.txt', 'a')
            error_log.write('{}\n'.format(label))



    def _read_text_file(self, textfile):
        """ Parse text file containing image file names and respective labels for uploading

        :param textfile: 'str' representing path to text file of images and machine/human labels
        :return: 'pandas' group object containing images organized by their labels
        """

        try:
            # Read text file
            df = pd.read_csv(textfile, sep=',', names=['image', 'label'])
        except:
            raise IndexError("{} could not be parsed correctly. Please check formatting".format(textfile))

        # Map enumerated labels into class names
        df = self.map_labels(dataframe=df, label_file=self.label_file, mapped_column='label')

        # Labels
        self.labels = sorted(df['class'].unique())

        return df.groupby(df['class'])



    def _prep_for_retrieval(self, textfile):
        ''' Parses desired text file for url configurations and builds the url

        :param textfile: 'str' representing path to text filename to parse from.
                         Expecting items to be separated by ', ' and ordered in such fashion:
                         ['start_time', 'end_time', 'min_len', 'max_len', 'cam']
        :return:
        '''

        try:
            # Read textfile
            self.data = pd.read_csv(textfile, sep=',', names=['start_time', 'end_time', 'min_len', 'max_len', 'cam'])
        except:
            print('{} could not be parsed correctly. Check formatting.'.format(os.path.basename(textfile)))

        #TODO ensure that min len and max len are numbers

        if not self.data.cam.isin(CAMERAS).all():
            raise ValueError('Camera specification(s) in ./{} not listed in camera choices. Options: {}'.
                             format(os.path.basename(textfile), CAMERAS))

        # Convert all at once and build url as new column
        self._build_url()



    def _build_url(self):
        """ Builds url for accessing spc.ucsd.edu pipeline for retrieving images and meta data

        :return:
        """
        def convert_date(date):
            """ Converts dates to Epoch Unix Time for Pacific West time zone

            :param date:
            :return:
            """
            import calendar
            import datetime as datetime
            import pytz

            def is_dst(dt=None, timezone="UTC"):
                """Check if date is daylight savings time enforced"""
                if dt is None:
                    dt = datetime.utcnow()
                timezone = pytz.timezone(timezone)
                timezone_aware_date = timezone.localize(dt, is_dst=None)
                return timezone_aware_date.tzinfo._dst.seconds != 0
            # Parse date
            dt = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

            # Check if daylight savings time
            dst = is_dst(dt, timezone="America/Los_Angeles")
            utc_date = calendar.timegm(pytz.timezone(
                'America/Los_Angeles').localize(dt).utctimetuple())

            # Returns Epoch Unix time
            if dst:
                return (utc_date+3600)*1000
            else:
                return utc_date*1000

        # Convert date & time to UTC & daylight savings
        self.data.start_time = self.data.start_time.apply(convert_date)
        self.data.end_time = self.data.end_time.apply(convert_date)

        # Convert camera resolution
        min_len = self.data.min_len.astype(float)
        max_len = self.data.max_len.astype(float)

        # Initialize min/max len based off camera
        if self.data.cam.any() == 'SPC2':
            self.data.min_len = np.floor(min_len / self.cam_res[0])
            self.data.max_len = np.ceil (max_len/ self.cam_res[0])
        elif self.data.cam.any() == 'SPCP2':
            self.data.min_len = np.floor (min_len / self.cam_res[1])
            self.data.max_len = np.ceil (max_len / self.cam_res[1])
        elif self.data.cam.any() == 'SPC-BIG':
            self.data['min_len'] = 1485936000000
            self.data['max_len'] = 1488441599000

        #TODO include option for parsing labels and type of annotator

        # Build url
        #TODO specify actual constants within url ??? = 'Any'
        # double bracket IMGS_PER_PAGE to prevent loading here
        pattern = "http://{}.ucsd.edu/data/rois/images/{}/{!s}/{!s}/0/24/{{!s}}/{!s}/{!s}/0.05/1/noexclude/ordered/skip/Any/anytype/Any/Any/"
        self.data['url'] = self.data.apply(
            lambda row: pattern.format('spc' if row.start_time < 1501488000000 else 'planktivore',
                                                row.cam, row.start_time, row.end_time,
                                                int(row.min_len), int(row.max_len)), axis=1)

    async def fetch(self, session, srcpath, destpath):
        async with session.get(srcpath) as resp:
            img = await resp.read()
            if self.sync_write:
                with open(destpath, "wb") as f:
                    f.write(img)
            else:
                async with file_sem:
                    async with aiofiles.open(destpath, "wb") as f:
                        await f.write(img)
