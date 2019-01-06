Environment :

- python ver.3.6.5
- tensorflow ver.1.12.0

## 모델 설명

ResNet 끝단에서 추출된 상품이미지벡터(2048)를 활용한 분류기와 상품 텍스트정보(product, brand, model, maker)를 활용한 분류기를 따로 학습시킨 후,
두 분류기를 통합한 새로운 앙상블모델을 학습.

## Start from raw dataset

### STEP 1. generate word idx (깃허브에 생성된 `word_to_idx`가 포함되어있으므로 건너뛰어도 됨)

카테고리명과 train dataset의 상품정보 텍스트에서 빈도수를 기준으로 임베딩될 단어 목록을 뽑고, index를 부여.
특히, 파싱시에 대문자로 전환하고, 브랜드의 경우 앞에 '브랜드)'가 붙은 경우를 없애주는 작업을 추가로 수행.

set storing path at word_path in `config.json`

```
$ python3 words.py make-dict
```

all
[INFO ] 2019-01-05 15:23:36 [words.py][get_meta_words:164] total 2437602 words
[INFO ] 2019-01-05 15:23:40 [words.py][get_word_idx:116] Total **500,408** words to be embedded are selected.
4,202 words from category-name and 500,000 from meta-info(least common word : ('방수방진기능', 5))

havesd
[INFO ] 2019-01-05 15:29:01 [words.py][get_meta_words:164] total 2044802 words
[INFO ] 2019-01-05 15:29:04 [words.py][get_word_idx:116] Total **500,424** words to be embedded are selected.
4,202 words from category-name and 500,000 from meta-info(least common word : ('산초분말', 4))

### STEP 2. prepare train/dev data with text indexing

```
# check word_path, *_data_list, data_root, max_len in `config.json`
# for trainset
python3 datashopnet.py make-db train --train-ratio=0.95 --sequence=False

# for validation set
python3 datashopnet.py make-db dev --train-ratio=0 --sequence=False

# for test set
python3 datashopnet.py make-db test --train-ratio=0 --sequence=False
```

### STEP 3. train

```
python3 shopnet.py train --case='image' --load=False

python3 shopnet.py train --case='text' --load=False

python3 shopnet.py train --case='ensemble' --load=False
```

## 제출 모델로 부터 결과 재현

1. 제출모델을 [다운로드]() 합니다.
2. `python3 datashopnet.py make-db test --train-ratio=0 --sequence=Fasle` 를 수행하여 전처리된 데이터셋을 만들어줍니다.
3. `inference.py` 코드의 SUBMIT_MODEL : [다운로드한 모델의 경로] , PROCESSED DATA : [전처리된 데이터 셋 경로] 로 설정 후,

```
python3 inference.py
```

## 모델 재현

1. `python3 datashopnet.py make-db train --train-ratio=0.95 --sequence=Fasle` 를 수행하여 학습에 사용할 전처리된 데이터셋을 준비합니다.
2. `python3 train.py` 를 수행하여 학습을 진행합니다.
