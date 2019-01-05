Environment :

- python ver.
- tensorflow ver.

## 1. generate word idx

```
words.py make-dict
```

set store path on config.json -> word_path

all
[INFO ] 2019-01-05 15:23:36 [words.py][get_meta_words:164] total 2437602 words
[INFO ] 2019-01-05 15:23:40 [words.py][get_word_idx:116] Total **500,408** words to be embedded are selected.
4,202 words from category-name and 500,000 from meta-info(least common word : ('방수방진기능', 5))

havesd
[INFO ] 2019-01-05 15:29:01 [words.py][get_meta_words:164] total 2044802 words
[INFO ] 2019-01-05 15:29:04 [words.py][get_word_idx:116] Total **500,424** words to be embedded are selected.
4,202 words from category-name and 500,000 from meta-info(least common word : ('산초분말', 4))

## 2. prepare train/dev data with text indexing

```
# check config's word_path, *_data_list, data_root, max_len
# for trainset
python3 datashopnet.py make-db train --train-ratio=0.85 --sequence=True

# for validation set
python3 datashopnet.py make-db dev --train-ratio=0 --sequence=True

# for test set
python3 datashopnet.py make-db test --train-ratio=0 --sequence=True

```

## 3. train

```
python3 shopnet_ensemble.py train --case='image' --load=False
```
