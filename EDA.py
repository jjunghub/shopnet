# -*- coding: utf-8 -*-
# Copyright 2019 jjunghub, Kakao Arena contest.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# eda.py
# written by jjung 2018.12.10
# EDA code for kakao arena dataset
"""

import os
import json
import h5py
import traceback
import sys
import pickle
import re
import numpy as np

re_sc = re.compile('[\W\d_]')

from misc import get_logger, Option
from multiprocessing import Pool 
from tqdm import tqdm

os.environ['OPT_NUM_THREADS'] = '1'
opt = Option('./config.json')


def get_count(data, cat_type = 'b') :
    try:
        data_path, div = data
        h = h5py.File(data_path, 'r')[div]
        
        cate_types = ['b', 'm', 's', 'd']
        count = {'b':{}, 'm':{}, 's':{}, 'd':{}, 'dummy' : {}}
        
        # read file as batch not by one by one to make faster
        m = h['pid'].shape[0]
        batch_size = 50000
        n_batch = int((m-1)/batch_size) + 1
        start = 0
        h_batch = dict()

        for i in range(n_batch) :
            # read
            for cat_type in cate_types : 
                h_batch[cat_type] = h['%scateid'%cat_type][start:min(start+batch_size, m)]

            # count => Count모듈사용하면 더 빠르게 할 수 있음.
            for j in range(len(h_batch['b'])) :
                cat_names = []
                for cat_type in cate_types :
                    cat_name = h_batch[cat_type][j]

                    cat_names.append(str(cat_name))
                    count[cat_type][cat_name] = (count[cat_type][cat_name]+1) if count[cat_type].get(cat_name) != None else 1 

                dummy_name = '>'.join(cat_names)
                count['dummy'][dummy_name] = (count['dummy'][dummy_name]+1) if count['dummy'].get(dummy_name) != None else 1 

            start = start + batch_size

        print('processed ', data_path, '(', m,') unique number of b,m,s,d', [len(count[cate_type].keys()) for cate_type in cate_types])
        return (data_path.split('.')[-1], count)
        
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))



class EDA:
    def __init__(self, data_path) :
        self.logger = get_logger('EDA')
        self.write_path = './EDA/'
        self.data_path = data_path

        self.cate1 = self.load_catefile(self.data_path + '/' + opt.cate_filename)


    def load_catefile(self, path) :
        return json.loads(open(path, 'rb').read().decode('utf-8'))

    def load_datafile(self, path) :
        return h5py.File(path, 'r')

    def catename(self, kind, value) :
        """
        get categoryname by index number.
        """
        for k,v in self.cate1[kind].items() :
            if v == value:
                return (k,v)

    def show_one(self, h, random) :
        for colname in ['pid', 'product', 'brand', 'model', 'maker', 'price', 'updttm', 'img_feat', '-', 'bcateid', 'mcateid', 'scateid', 'dcateid'] :
            if colname == '-' :
                print("-"*50)
                continue
            val = h['train'][colname][random]
            if (val.dtype != np.int32) and (val.dtype != np.float32): 
                val = val.decode('utf8')
            elif colname=='bcateid' :
                val = self.catename('b', val)
            elif colname=='mcateid' :
                val = self.catename('m', val)
            elif colname=='scateid' :
                val = self.catename('s', val)
            elif colname=='dcateid' :
                val = self.catename('d', val)
                
            if colname in ['product', 'brand', 'model', 'maker'] :
                print(colname, val, '\n if split : ', re_sc.sub(' ', val).strip().split())
            else :
                print(colname, val)
        
    def show_byvalue(self, value, bywhich='pid', maxshow=100) :
        data_lists = [os.path.join(self.data_path,path) for path in opt.train_data_list]
        cnt = 0
        for path in data_lists :
            h = h5py.File(path, 'r')
            for i in range(0,h['train'][bywhich].shape[0], 10000) :
                data = h['train'][bywhich][i:min(h['train'][bywhich].shape[0], i+10000)]
                for j,pid in enumerate(data):
                    if pid.decode('utf8') == value :
                        cnt += 1
                        self.show_one(h, i+j)
                        print(path)
                        print('\n')

                        if cnt > maxshow : return



    def check_y_proportion(self, datalist_name, div) :
        """
        Arguments:
        datalist_name - datalist name in config to check.
        div 
        """
        self.logger.info('Checking label proportion in %s, with %d core.' % 
                         (datalist_name, opt.num_workers))
        
        pool = Pool(opt.num_workers)
        try :
            counts = pool.map_async(get_count, 
                                   [(data_path, div)
                                    for data_path in opt[datalist_name]]).get(99999999)
            pool.close()
            pool.join()  
            
            # save 
            if not os.path.isdir(self.write_path):
                os.makedirs(self.write_path)
            for file_num, count in counts :
                pickle.dump(count, open(self.write_path+file_num, 'wb'))
                self.logger.info('Save label counting as picke file {}'.format(self.write_path+file_num))

            return counts
                
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise


