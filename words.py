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
# words.py
# get important words list to be embedded.
"""

import os
os.environ['OPT_NUM_THREADS'] = '1'
from misc import get_logger, Option
opt = Option('./config.json')
from multiprocessing import Pool 
from tqdm import tqdm
import h5py
import traceback
import sys
import pickle
from collections import Counter
import fire

import re

re_sc = re.compile('[\W\d_]')

# for brand parsing
re_brand_prefix = re.compile('^브랜드\)')
re_sc_brand = re.compile('[()\[\],]')
re_space = re.compile('[\s]')


def parse(raw, is_brand=False) :
    # split by re_sc + strip + upper
    if type(raw) != str :
        raw = raw.decode('utf8')

    if not is_brand : 
        parse_words = re_sc.sub(' ', raw).strip().split()
        parse_words = [w.strip().upper() for w in parse_words]
    else :
        raw = re_brand_prefix.sub('',raw) 
        parse_words = re_sc.sub(' ', raw).strip().split()
        parse_words = [w.strip().upper() for w in parse_words]

        # raw = re_brand_prefix.sub('',raw)
        # raw = re_space.sub('', raw)    
        # parse_words = re_sc_brand.sub(' ', raw).strip().split()
        # parse_words = [w.strip().upper() for w in parse_words]

    return parse_words

def _get_meta_words(data) :
    # 어떤 우선순위로 임베딩할 단어를 정할 것인가
    try:
        data_path, div = data
        print(data_path, div)
        h = h5py.File(data_path, 'r')[div]
        m = h['pid'].shape[0]
        # 일단 all
        to_parse = ['product', 'brand', 'model', 'maker'] 
        # to_parse = ['product'] 

        # words = set()
        batch_size = 50000
        n_batch = int((m-1)/batch_size) + 1
        words = []
        start = 0
        for i in tqdm(range(n_batch), mininterval=1) :
            for col in to_parse :
                batch =  h[col][start:min(start+batch_size, m)]
                s = h['scateid'][start:min(start+batch_size, m)]
                d = h['dcateid'][start:min(start+batch_size, m)]
                for each in range(batch.shape[0]) : 
                    # if (s[each]!=-1 or d[each]!=-1):
                    if True:

                        if col == 'brand' :
                            parse_words = parse(batch[each], is_brand=True)
                        else : 
                            parse_words = parse(batch[each])
                            # 최소 갯수 2~
                            parse_words = [w for w in parse_words
                                            if len(w) >= opt.min_word_length and len(w) < opt.max_word_length]
                                            
                        words.extend(parse_words)
            start = start + batch_size

        print('{} unique words in {}'.format(len(set(words)), data_path))
        return Counter(words)
        
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))



class Words:
    def __init__(self) :
        self.logger = get_logger('words')
        self.ignore = ['기타', '참조', '상품상세설명', '주', '청구할인', '상세설명참조', '없음', '상세정보참조', 
                        '해당없음', '품명', '상품상세정보', '상세설명', '상세정보별도표시', '알수', '상세페이지', 
                        '상세참조', 'ETC', '상세내용참조', '기타상세참조', '상세정보', '별도표기', 
                        '상세페이지참조', '알수없음', '상품상세설명참조'] + [chr(asc) for asc in range(65,91)]

    def get_word_idx(self, datalist_name='train_data_list', write_path=opt.word_path) :
        """
        Select {opt.max_embd_words} words to be embedded and make word_idx dict.
        """
        words_category = self.get_category_words()
        words_meta = self.get_meta_words(datalist_name=datalist_name)

        for ig in self.ignore :
            try:
                words_meta.pop(ig)
            except:
                pass
        # n_remain = max(0, opt.max_embd_words - len(words_category))
        n_remain = max(0, opt.max_embd_words)
        common_words = words_meta.most_common(n_remain)

        word_to_embd = list(set(words_category + [word for word,_ in common_words]))
        self.logger.info("Total {:,} words to be embedded are selected. \n {:,} words from category-name and {:,} from meta-info(least common word : {})".format(len(word_to_embd), len(words_category), n_remain, str(common_words[-1])))

        word_to_idx = {word:idx+1 for idx, word in enumerate(word_to_embd)}
        
        # save word_to_idx pickle
        write_dir = os.path.dirname(write_path)
        if not os.path.isdir(write_dir):
            os.makedirs(write_dir)

        pickle.dump(word_to_idx, open(write_path, 'wb'))
        pickle.dump({'category':words_category, 'meta':words_meta}, open(write_path+'_detail', 'wb'))

        
        self.logger.info("dict write on {}".format(write_path))

        # return word_to_idx
        

    def get_category_words(self) :
        import json

        catefile = opt.dataset_location + opt.cate_filename
        cate1 = json.loads(open(catefile, 'rb').read().decode('utf-8'))

        words = []
        for kind in ['b', 'm', 's', 'd'] :
            for each in cate1[kind].keys() :
                words.extend(parse(each))

        self.logger.info('{} unique words in {}'.format(len(set(words)), opt.cate_filename))

        return list(set(words))


    def get_meta_words(self, datalist_name='train_data_list', div='train') :   
        pool = Pool(opt.num_workers)
        try :
            g_words = pool.map_async(_get_meta_words, 
                                   [(data_path, div)
                                    for data_path in opt[datalist_name]]).get(999999)
            pool.close()
            pool.join()  

            words = Counter()
            for w in g_words :
                words = words + w
            

            self.logger.info('total {} words'.format(len(words)))
            return words
                
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise


if __name__ == '__main__':
    words = Words()
    fire.Fire({'make-dict': words.get_word_idx})

