"""
test 예측 결과 재현을 위한 코드
"""
import shopnet as sn

SUBMIT_MODEL = './ensemble/model_1024_02_relu.h5'
PROCESSED_DATA = './processed_data/dev/data.h5py'
PREDICT = './predict_result.tsv'


if __name__ == '__main__':
	shopnet = sn.ShopNet()
	shopnet.predict(model_path = SUBMIT_MODEL, datafile= PROCESSED_DATA, datakind="dev", writefile = PREDICT)