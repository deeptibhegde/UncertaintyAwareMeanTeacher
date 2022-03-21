import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns


e = [1,30]

n = int(input("Enter num samples: "))


for epoch in e:

	path = "/media/vishwa/hd3/code/detection3d/ST3D_unc_roi/output/kitti_models/secondiou_oracle_ros/unc_waymo_out_0p999_it5_positive/var_epoch%d/"%epoch
	

	it = os.listdir(path)

	n = n//4

	m = []
	for i in range(n):
		m.append(it[i])

	l = []

	for i in range(n):

		mat = np.load(path + m[i])
		l.append(mat)



	corr = []

	incorr = []



	for mat in l:



		rpn_cls_teacher_score, rpn_cls_label, rpn_teacher_pred_var = np.squeeze(np.around(mat['a']).astype(int)), mat['b'], np.squeeze(mat['c'])


		# print(rpn_cls_teacher_score.shape,rpn_cls_label.shape)


		for i in range(len(rpn_cls_label)):
			if rpn_cls_teacher_score[i] == rpn_cls_label[i]:
				corr.append(rpn_teacher_pred_var[i])
			else:
				incorr.append(rpn_teacher_pred_var[i])
			


	print(rpn_cls_teacher_score,rpn_cls_label,rpn_teacher_pred_var)

	plt.figure()


	if epoch == 1:
		sns.distplot(corr,kde=True,label='Correct Predictions, first epoch')
		sns.distplot(incorr, kde=True,label='Incorrect Predictions, first epoch')
		pyplot.legend(loc='upper right')
		pyplot.xlim(-10,20)
	else:
		sns.distplot(corr,kde=True,label='Correct Predictions, last epoch')
		sns.distplot(incorr, kde=True,label='Incorrect Predictions, last epoch')
		pyplot.legend(loc='upper right')
		pyplot.xlim(-10,20)






plt.show()