import math

from lib.models.ortrack import build_ortrackst
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

# for MATU
import random
# for supervision
import supervision as sv
import numpy as np
from typing import List, Tuple

class DMFTrack_ST(BaseTracker):
    def __init__(self, params, dataset_name):
        super(DMFTrack_ST, self).__init__(params)
        network = build_dmftrackst(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        # self.network = network.cpu()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.feat_template_sz = self.cfg.TEST.TEMPLATE_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        # self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cpu()

        # for debug
        self.debug = params.debug
        # self.use_visdom = params.debug
        self.use_visdom = 0
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                # self.save_dir = "debug"
                self.save_dir = os.path.join("debug", dataset_name, "result")
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                self.debug_update_dir = os.path.join("debug", dataset_name, "update")
                os.makedirs(self.debug_update_dir, exist_ok=True)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.is_distill = params.cfg.MODEL['IS_DISTILL']
        
        # === MATU ===
        self.template_update_interval = params.cfg.MATU['UPDATE_INTERVAL']  
        self.feature_alpha = params.cfg.MATU['FEATURE_ALPHA']
        self.update_thresh = params.cfg.MATU.get('UPDATE_THRESH', 0.9)  

        # === gallery buffer ===
        self.template_pool = []   
        self.max_templates = 3   

        # === Supervision ===
        self.box_annotator = sv.BoxAnnotator(thickness=2,
            color=sv.Color(r=255, g=0, b=0)  
        )
        
        self.trace_annotator = sv.TraceAnnotator(thickness=2,
            trace_length=30, 
            position=sv.Position.CENTER 
        )
        
        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_scale=0.5, text_thickness=1, text_padding=5)
        
        self.trace_points: List[Tuple[float, float]] = []
        self.max_trace_length = 30
        
        self.previous_center = None
        self.previous_time = None
        self.current_speed = 0.0
    
    def _update_template_pool(self, template_nestedtensor, conf_score):
        if conf_score < self.update_thresh:
            return
        # store (NestedTensor, conf)
        self.template_pool.append((template_nestedtensor, conf_score))
        # keep top-k by conf, but self.template_pool[0] is always the initial template.

        self.template_pool.sort(key=lambda x: x[1], reverse=True)

        if len(self.template_pool) > self.max_templates:
            self.template_pool = self.template_pool[:self.max_templates+1]  # +1 for initial template

    def initialize(self, image, info: dict):
       # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)


        self.template_pool.append((template, 1.1))  
        
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)


        self.trace_points = []
        self.previous_center = None
        self.previous_time = None
        self.current_speed = 0.0

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, is_distill=self.is_distill)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()


        if self.frame_id % self.template_update_interval == 0: #  and conf_score > self.update_thresh

            new_z_patch_arr, _, new_z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                output_sz=self.params.template_size)
            new_template = self.preprocessor.process(new_z_patch_arr, new_z_amask_arr)
            
            self._update_template_pool(new_template, conf_score) 
            
            if len(self.template_pool) >= self.max_templates:
            # if self.template_pool is not None:
                rand_template, _ = random.choice(self.template_pool)

                self.z_dict1.tensors = (self.feature_alpha * self.z_dict1.tensors + (1 - self.feature_alpha) * rand_template.tensors)
            
            if self.debug:
                if not self.use_visdom:
                    print(f"conf_score={conf_score},Frame {self.frame_id}: Template updated")
                    template_path = os.path.join(self.debug_update_dir, f"{self.frame_id:04d}_update.jpg")
                    cv2.imwrite(template_path, cv2.cvtColor(new_z_patch_arr, cv2.COLOR_RGB2BGR))
                    # save the current search as well
                    search_path = os.path.join(self.debug_update_dir, f"{self.frame_id:04d}_search.jpg")
                    cv2.imwrite(search_path, cv2.cvtColor(x_patch_arr, cv2.COLOR_RGB2BGR))
                else:
                    self.visdom.register(torch.from_numpy(new_z_patch_arr).permute(2, 0, 1), 'image', 1, f'new_template_{self.frame_id}')
                    # self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'initial_template')
                    # self.visdom.register(torch.from_numpy(rand_template.tensors.squeeze(0).cpu().numpy()).permute(2, 0, 1), 'image', 1, f'random_template_{self.frame_id}')
                    # self.visdom.register(torch.from_numpy(new_template.tensors.squeeze(0).cpu().numpy()).permute(2, 0, 1), 'image', 1, f'new_template_{self.frame_id}')
                    # self.visdom.register(torch.from_numpy(self.z_dict1.tensors.squeeze(0).cpu().numpy()).permute(2, 0, 1), 'image', 1, f'updated_template_{self.frame_id}')
                
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state

                current_center = (x1 + w/2, y1 + h/2)

                current_time = self.frame_id 
                if self.previous_center is not None and self.previous_time is not None:

                    dx = current_center[0] - self.previous_center[0]
                    dy = current_center[1] - self.previous_center[1]
                    displacement = math.sqrt(dx**2 + dy**2)
        
                    time_diff = current_time - self.previous_time
                    if time_diff > 0:
                        self.current_speed = displacement / time_diff

                self.previous_center = current_center
                self.previous_time = current_time

                self.trace_points.append(current_center)
                if len(self.trace_points) > self.max_trace_length:
                    self.trace_points.pop(0)
    
                if self.debug:
                    if not self.use_visdom:
       
                        xyxy = [x1, y1, x1 + w, y1 + h]
                        detections = sv.Detections(
                            xyxy=np.array([xyxy]),
                            confidence=np.array([conf_score]),
                            class_id=np.array([0]), 
                            tracker_id=np.array([0]) 
                        )
            
                      
                        annotated_image = image.copy()

                        annotated_image = self.box_annotator.annotate(
                            scene=annotated_image,
                            detections=detections
                        )

                annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, annotated_image_bgr)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score}
        else:
            return {"target_bbox": self.state,
                    "conf_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return DMFTrack_ST