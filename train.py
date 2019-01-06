"""
모델 생성 재현을 위한 코드
최종모델 저장경로 : {config.json:model_root}/ensemble/model.h5
"""
import shopnet as sn

if __name__ == '__main__':
	shopnet = sn.ShopNet()

	# train image classifier
	shopnet.train(case='image', load=False, lr=0.001, num_epochs=3)
	shopnet.train(case='image', load=True, lr=0.0001, num_epochs=10)

	# train text classifier
	shopnet.train(case='text', load=False, lr=0.001, num_epochs=3)
	shopnet.train(case='text', load=True, lr=0.0001, num_epochs=10)

	# train ensemble classifier
	shopnet.train(case='ensemble', load=False, lr=0.001, num_epochs=10)

