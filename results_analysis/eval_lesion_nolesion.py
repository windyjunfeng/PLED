# coding=utf-8
import os
import pingouin as pg
import numpy as np
import pandas as pd


evaluation_rootpath=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/evaluation/efield_calculation'
lesion_rootpath=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
filename_order_path=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/evaluation/val_filename_order.npy'
filenames=np.load(filename_order_path)
full_paths = [os.path.join(lesion_rootpath, 'm2m_'+filename.split('_')[0], 'local_lesion',filename) for filename in filenames]
vectorized_exists = np.vectorize(os.path.exists)
lesion_flag = vectorized_exists(full_paths)
nolesion_flag =~lesion_flag
eval_names=os.listdir(evaluation_rootpath)
for eval_name in eval_names:
    eval_path=os.path.join(evaluation_rootpath,eval_name,'total')
    filenames=os.listdir(eval_path)
    with open(os.path.join(eval_path,'lesion_nolesion_total.txt'), 'w') as f:
        for filename in filenames:
            if filename.endswith('.npy'):
                eval_data=np.load(os.path.join(eval_path,filename))
                if len(eval_data)!=len(lesion_flag):  # 防止评价指标有的为none，没有保存，这样索引就对不上了
                    continue
                eval_lesion=np.sum(eval_data[lesion_flag])/np.sum(lesion_flag)
                eval_nolesion = np.sum(eval_data[nolesion_flag]) / np.sum(nolesion_flag)
                ttest_results = pg.ttest(eval_data[lesion_flag], eval_data[nolesion_flag], paired=False)
                print(filename.split('_total.npy')[0],':',file=f)
                print('lesion: {:.4f}'.format(eval_lesion),file=f)
                print('nolesion: {:.4f}'.format(eval_nolesion), file=f)
                print('p-value: ',ttest_results['p-val'][0],file=f)
                print('\n',file=f)