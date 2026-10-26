from . import DMFTrackActor


class DMFTrackSTActor(DMFTrackActor):
    """ Actor for training DMFTrack_ST models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective, loss_weight, settings)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)

        if len(template_list) == 1:
            template_list = template_list[0]

        # ====distillation====
        if self.net.is_distill_training:
            with torch.no_grad():
                out_dict_teacher = self.net_teacher(template=template_list,
                                                    search=search_img, is_distill=True)
        # ==========================

        out_dict = self.net(template=template_list,
                            search=search_img, is_distill=False)

        if self.net.is_distill_training:
            feat_teacher = out_dict_teacher['backbone_feat']
            feat_student = out_dict['backbone_feat']
            distill_loss = torch.tensor([torch.nn.functional.mse_loss(feat_teacher[i], feat_student[i]) for i in range(feat_student.shape[0])]).cuda()
            out_dict['distill_loss'] = distill_loss

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # compute cls loss
        labels = gt_dict['label'].view(-1) # # (batch, ) 0 or 1, [32]
        # labels = labels.float() # long -> float
        cls_loss = self.objective['cls'](pred_dict["pred_logits"].view(-1), labels) # (32*1),[batch](32)
        
        loss = self.loss_weight['cls'] * cls_loss   # add cls_loss
        
        if return_status:
            # status for log
            status = {"Loss/cls": cls_loss.item()}
            return loss, status
        else:
            return loss
