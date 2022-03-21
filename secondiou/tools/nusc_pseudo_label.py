import pickle 


with open('/media/HDD/vishwa/detection3d/ST3D/data/nuscenes/nuscenes_infos_10sweeps_train.pkl', 'rb') as ff:
    data_nusc = pickle.load(ff)


with open('/media/HDD/vishwa/detection3d/ST3D/data/nuscenes/ps/ps_label_e0.pkl', 'rb') as f:
    data_pseudo = pickle.load(f)



data_pseudo_keys = list(data_pseudo.keys())

# print(data_pseudo[data_pseudo_keys[0]]['cls_scores'])


for sample in data_pseudo_keys:
    for item in data_nusc:
        if sample in item['lidar_path']:
            item['gt_boxes'] = data_pseudo[sample]['gt_boxes']
            # print(item['gt_names'])
            item['gt_names'] = ['car']*len(data_pseudo[sample]['gt_boxes'])
            # print(item['gt_names'])

            print(sample)


dbfile = open('/media/HDD/vishwa/detection3d/ST3D/data/nuscenes/nuscenes_infos_10sweeps_train_pseudo.pkl', 'ab')
      
# source, destination
pickle.dump(data_nusc, dbfile)                     
dbfile.close()