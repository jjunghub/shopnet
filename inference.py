"""
test 예측 결과 재현을 위한 코드
"""
import shopnet as sn

SUBMIT_MODEL = './model_final.h5'
PROCESSED_DATA = './processed_data/test/data.h5py'
PREDICT = './predict_result.tsv'


if __name__ == '__main__':
	shopnet = sn.ShopNet()
	shopnet.predict(model_path = SUBMIT_MODEL, datafile= PROCESSED_DATA, datakind="test", writefile = PREDICT)