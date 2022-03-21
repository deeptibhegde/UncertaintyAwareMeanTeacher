import torch
from .vfe_template import VFETemplate
from torch.autograd import Variable

class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def gaussian(self, ins, is_training, mean, stddev):
        if is_training:
            # import pdb; pdb.set_trace()
            
            noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
            return ins + noise
        return ins

    def forward(self, batch_dict,is_training=False,is_teacher=False,**kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        # import pdb; pdb.set_trace()
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']

        #add gaussian noise to input
        

        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        if is_training and is_teacher:
            points_mean = self.gaussian(points_mean,is_training,0,0.1)
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict
