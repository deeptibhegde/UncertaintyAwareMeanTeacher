import torch
from .detector3d_template import Detector3DTemplate
from ..model_utils.model_nms_utils import class_agnostic_nms
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from collections import OrderedDict
from ...utils import common_utils, loss_utils
import numpy as np

class SECONDNetIoU(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()


    def _update_teacher_models(self,keep_rate=0.999):

        #backbone_3d (VoxelBackBone8x)
        student_model_dict = self.module_list[1].state_dict()
        new_teacher_dict = OrderedDict()

        for key, value in self.module_list[2].state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))
        if student_model_dict.keys:
            self.module_list[2].load_state_dict(new_teacher_dict)



        #backbone_2d (BaseBEVBackbone)
        student_model_dict = self.module_list[4].state_dict()
        new_teacher_dict = OrderedDict()

        for key, value in self.module_list[5].state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))
        if student_model_dict.keys:
            self.module_list[5].load_state_dict(new_teacher_dict)




        #dense_head (AnchorHeadSingle)
        student_model_dict = self.module_list[6].state_dict()
        new_teacher_dict = OrderedDict()

        for key, value in self.module_list[7].state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))
        if student_model_dict.keys:
            self.module_list[7].load_state_dict(new_teacher_dict)


        #roi_head (SecondHead)
        student_model_dict = self.roi_head_stu.state_dict()
        new_teacher_dict = OrderedDict()

        for key, value in self.roi_head_tea.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))
        if student_model_dict.keys:
            self.roi_head_tea.load_state_dict(new_teacher_dict)


    def forward(self, batch_dict):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg

        batch_dict_stu = {}
        batch_dict_tea = {}

        it_vfe = []
        it_backbone_3d = []
        it_map_bev = []
        it_backbone_2d = []
        it_dense_head = []
        it_roi_head = []

        it_batch_dict_tea = []

        n = 5

        for i in range(n):
            it_batch_dict_tea.append(OrderedDict())
            for k in batch_dict.keys():
                it_batch_dict_tea[i][k] = batch_dict[k]


        for k in batch_dict.keys():
            batch_dict_stu[k] = batch_dict[k]
            # batch_dict_tea[k] = batch_dict[k]




        # import pdb; pdb.set_trace()
        # for cur_module in self.module_list:
        #     batch_dict = cur_module(batch_dict)

        #vfe (MeanVFE)
        batch_dict_stu = self.vfe(batch_dict_stu)
        for i in range(n):
            # it_vfe.append(self.vfe(batch_dict_tea,is_training=self.training,is_teacher=True))
            it_batch_dict_tea[i] = self.vfe(it_batch_dict_tea[i],is_training=self.training,is_teacher=True)

        #backbone_3d (VoxelBackBone8x)
        batch_dict_stu = self.backbone_3d_stu(batch_dict_stu)

        if self.training:
            for i in range(n):
                # it_backbone_3d.append(self.backbone_3d_tea(it_vfe[i]))
                it_batch_dict_tea[i] = self.backbone_3d_tea(it_batch_dict_tea[i])


        #map_to_bev (HeightCompression)
        batch_dict_stu = self.map_to_bev_module(batch_dict_stu)
        if self.training:
            for i in range(n):
                # it_map_bev.append(self.map_to_bev_module(it_backbone_3d[i])) 
                it_batch_dict_tea[i] = self.map_to_bev_module(it_batch_dict_tea[i])
        
        #backbone_2d (BaseBEVBackbone)
        batch_dict_stu = self.backbone_2d_stu(batch_dict_stu)

        if self.training:
            for i in range(n):
                # it_backbone_2d.append(self.backbone_2d_tea(it_map_bev[i]))
                it_batch_dict_tea[i] = self.backbone_2d_tea(it_batch_dict_tea[i])
                

        #dense_head (AnchorHeadSingle)
        batch_dict_stu = self.dense_head_stu(batch_dict_stu,is_student=True)

        if self.training:
            for i in range(n):
                # it_dense_head.append(self.dense_head_tea(it_backbone_2d[i])) 
                it_batch_dict_tea[i] = self.dense_head_tea(it_batch_dict_tea[i])


        batch_dict_stu = self.roi_head_stu(batch_dict_stu)
        if self.training:
            for i in range(n):
                # it_roi_head.append(self.roi_head_tea(it_dense_head[i],is_teacher=True))
                it_batch_dict_tea[i] = self.roi_head_tea(it_batch_dict_tea[i])
        
        
        if self.training:
            self._update_teacher_models()

        # import pdb; pdb.set_trace()

        if self.training:
            loss,cons_loss, tb_dict, disp_dict = self.get_training_loss(batch_dict_stu,it_batch_dict_tea)

            ret_dict = {
                'loss': loss,
                'cons_loss': cons_loss

            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict_stu)
            return pred_dicts, recall_dicts

    def get_consistency_loss(self,batch_dict_stu,it_roi_head):

        eps = 10e-6

        cls_pred_stu = batch_dict_stu['rcnn_iou'].view(-1)
        cls_pred_tea_it = []
        for i in range(len(it_roi_head)):
            cls_pred_tea_it.append(it_roi_head[i]['rcnn_iou'].view(-1).detach())

        cls_pred_tea_it = torch.stack(cls_pred_tea_it)

        cls_pred_tea_mean = torch.mean(cls_pred_tea_it,0)
        cls_pred_tea_var = torch.var(cls_pred_tea_it,0)

        # cls_pred_tea_var = cls_pred_tea_var/cls_pred_tea_var.max()

        c = (1.0/(cls_pred_tea_var+eps))

        c = torch.clamp(c,10e-5,1)

        # c = (cls_pred_tea_var<0.1).float()

        cls_pred_tea_score = torch.sigmoid(cls_pred_tea_mean).view(-1)
        cls_pred_stu_score = torch.sigmoid(cls_pred_stu).view(-1)

        roi_cls_labels = batch_dict_stu['rcnn_cls_labels'].view(-1)

        roi_valid_mask = (roi_cls_labels >= 0).float()


        # batch_loss_iou = (c*(torch.nn.functional.l1_loss(cls_pred_stu,cls_pred_tea_mean,reduce=False))).mean()
        # batch_loss_iou = (c*(loss_utils.sigmoid_focal_cls_loss(cls_pred_stu,cls_pred_tea_mean))).mean()


        batch_loss_iou = torch.nn.functional.binary_cross_entropy_with_logits(cls_pred_stu_score.float(),cls_pred_tea_score.float(), reduction='none')

        rcnn_loss_iou = (batch_loss_iou * roi_valid_mask*c).sum() / torch.clamp(roi_valid_mask.sum(), min=1.0)


        # import pdb; pdb.set_trace()

        ind = batch_dict_stu['frame_id']

        # np.savez_compressed('/media/vishwa/hd3/code/detection3d/ST3D_unc_roi/output/kitti_models/secondiou_oracle_ros/unc_waymo_out_0p999_it5_positive/var_epoch1/%s.npz'%ind[0],a=cls_pred_tea_score.cpu(),b=roi_cls_labels.cpu(),c=cls_pred_tea_var.cpu())


        return rcnn_loss_iou,c    

    def get_box_reg_cons_loss(self,batch_dict_stu,it_roi_head,tb_dict):
        box_preds_stu = batch_dict_stu['box_preds']

        reg_loss_func = loss_utils.WeightedSmoothL1Loss(code_weights=self.model_cfg.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])

        it_box_preds_tea = []
        for i in range(len(it_roi_head)):
            it_box_preds_tea.append(it_roi_head[i]['box_preds'].detach())

        it_box_preds_tea = torch.stack(it_box_preds_tea)

        box_preds_tea = torch.mean(it_box_preds_tea,0)


        box_cls_labels = batch_dict_stu['box_cls_labels']
        batch_size = int(box_preds_stu.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        box_preds_stu = box_preds_stu.view(batch_size, -1,
                                   box_preds_stu.shape[-1] // 2)
        box_preds_tea = box_preds_tea.view(batch_size, -1,
                                   box_preds_tea.shape[-1] // 2)
        # sin(a - b) = sinacosb-cosasinb
        
        # import pdb; pdb.set_trace()

        box_preds_stu_sin, box_preds_tea_sin = self.add_sin_difference(box_preds_stu, box_preds_tea)
        loc_loss_src = reg_loss_func(box_preds_stu_sin, box_preds_tea_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']


 

        tb_dict['rpn_reg_cons_loss'] = loc_loss.item()

        return loc_loss, tb_dict

    def add_sin_difference(self,boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2


    def get_training_loss(self,batch_dict_stu,it_roi_head):
        disp_dict = {}

        cons_loss,unc_weight = self.get_consistency_loss(batch_dict_stu,it_roi_head)

        

        loss_rpn, tb_dict = self.dense_head_stu.get_loss()
        loss_rcnn, tb_dict = self.roi_head_stu.get_loss(unc_weight,tb_dict)

        # reg_cons_loss,tb_dict = self.get_box_reg_cons_loss(batch_dict_stu,it_roi_head,tb_dict)

        # cons_loss += reg_cons_loss

        loss = loss_rpn + loss_rcnn 
        return loss, cons_loss, tb_dict, disp_dict

    @staticmethod
    def cal_scores_by_npoints(cls_scores, iou_scores, num_points_in_gt, cls_thresh=10, iou_thresh=100):
        """
        Args:
            cls_scores: (N)
            iou_scores: (N)
            num_points_in_gt: (N, 7+c)
            cls_thresh: scalar
            iou_thresh: scalar
        """
        assert iou_thresh >= cls_thresh
        alpha = torch.zeros(cls_scores.shape, dtype=torch.float32).cuda()
        alpha[num_points_in_gt <= cls_thresh] = 0
        alpha[num_points_in_gt >= iou_thresh] = 1
        
        mask = ((num_points_in_gt > cls_thresh) & (num_points_in_gt < iou_thresh))
        alpha[mask] = (num_points_in_gt[mask] - 10) / (iou_thresh - cls_thresh)
        
        scores = (1 - alpha) * cls_scores + alpha * iou_scores

        return scores

    def set_nms_score_by_class(self, iou_preds, cls_preds, label_preds, score_by_class):
        n_classes = torch.unique(label_preds).shape[0]
        nms_scores = torch.zeros(iou_preds.shape, dtype=torch.float32).cuda()
        for i in range(n_classes):
            mask = label_preds == (i + 1)
            class_name = self.class_names[i]
            score_type = score_by_class[class_name]
            if score_type == 'iou':
                nms_scores[mask] = iou_preds[mask]
            elif score_type == 'cls':
                nms_scores[mask] = cls_preds[mask]
            else:
                raise NotImplementedError

        return nms_scores

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            iou_preds = batch_dict['batch_cls_preds'][batch_mask]
            cls_preds = batch_dict['roi_scores'][batch_mask]

            src_iou_preds = iou_preds
            src_box_preds = box_preds
            src_cls_preds = cls_preds
            assert iou_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                iou_preds = torch.sigmoid(iou_preds)
                cls_preds = torch.sigmoid(cls_preds)

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                iou_preds, label_preds = torch.max(iou_preds, dim=-1)
                label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels', False) else label_preds + 1

                if post_process_cfg.NMS_CONFIG.get('SCORE_BY_CLASS', None) and \
                        post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'score_by_class':
                    nms_scores = self.set_nms_score_by_class(
                        iou_preds, cls_preds, label_preds, post_process_cfg.NMS_CONFIG.SCORE_BY_CLASS
                    )
                elif post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) == 'iou' or \
                        post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) is None:
                    nms_scores = iou_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'cls':
                    nms_scores = cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'weighted_iou_cls':
                    nms_scores = post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.iou * iou_preds + \
                                 post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.cls * cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'num_pts_iou_cls':
                    point_mask = (batch_dict['points'][:, 0] == batch_mask)
                    batch_points = batch_dict['points'][point_mask][:, 1:4]

                    num_pts_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                        batch_points.cpu(), box_preds[:, 0:7].cpu()
                    ).sum(dim=1).float().cuda()
                    
                    score_thresh_cfg = post_process_cfg.NMS_CONFIG.SCORE_THRESH
                    nms_scores = self.cal_scores_by_npoints(
                        cls_preds, iou_preds, num_pts_in_gt, 
                        score_thresh_cfg.cls, score_thresh_cfg.iou
                    )
                else:
                    raise NotImplementedError

                selected, selected_scores = class_agnostic_nms(
                    box_scores=nms_scores, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    raise NotImplementedError

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
                'pred_cls_scores': cls_preds[selected],
                'pred_iou_scores': iou_preds[selected]
            }

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict
