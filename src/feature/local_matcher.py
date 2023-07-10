from feature.local_extractor import Local_extractor
from third_party.local_feature.LightGlue.lightglue.utils import load_image, match_pair
import torch
import numpy as np

class Local_matcher():
    device='cuda' if torch.cuda.is_available() else "cpu"
    def __init__(self,db_name, hfile_local, registered_feature,**feature_configs):
        local_feature = Local_extractor(feature_configs['local'])
        self.local_feature_matcher = local_feature.matcher().to(self.device)
        self.db_name=db_name
        self.hfile_local=hfile_local
        self.registered_feature=registered_feature

    def match_filter(self, matches, feats0, feats1, feature_dict, index_list):
        pts0 = np.empty((len(matches), 2))
        pts1 = np.empty((len(matches), 2))
        lms = np.empty((len(matches),3))
        counter = 0
        for n, m in enumerate(matches):
            if m != -1 and m in index_list:
                pts0[counter] = feats0['keypoints'][n]
                pts1[counter] = feats1['keypoints'][m]
                lms[counter] = feature_dict[m]
                counter += 1
        pts0 = pts0[:counter]
        pts1 = pts1[:counter]
        lms = lms[:counter]
        return pts0, pts1, lms

    # def lightglue(self, topk, feats0):

    #     feats1 = self.hfile_local[self.db_name[i]]

    #     # data = {'image0': image0[None].cuda(), 'image1': image1[None].cuda()}
    #     # img0, img1 = data['image0'], data['image1']

    #     # feats0, feats1 = extractor({'image': img0}), extractor({'image': img1})

    #     pred = {**{k+'0': torch.tensor(v).unsqueeze(0).to(self.device) for k, v in feats0.items()},
    #             **{k+'1': torch.tensor(v).unsqueeze(0).to(self.device) for k, v in feats1.items()},}
    #             # **data}
    #     pred['keypoint_scores0']=pred.pop('scores0')
    #     pred['keypoint_scores1']=pred.pop('scores1')

    #     pred=self.local_feature_matcher(pred)
    #     matches = pred['matches0'][0].detach().cpu().short().numpy()
    #     registered_feature = self.registered_feature[i]
    #     index_list = set(registered_feature[0])
    #     feature_dict = {key: value for key, value in zip(registered_feature[0], registered_feature[1])}
    #     pts0 = np.empty((len(matches), 2))
    #     pts1 = np.empty((len(matches), 2))
    #     lms = np.empty((len(matches),3))
    #     counter = 0
    #     for n, m in enumerate(matches):
    #         if m != -1 and m in index_list:
    #             pts0[counter] = feats0['keypoints'][n]
    #             pts1[counter] = feats1['keypoints'][m]
    #             lms[counter] = feature_dict[m]
    #             counter += 1
    #     pts0 = pts0[:counter]
    #     pts1 = pts1[:counter]
    #     lms = lms[:counter]
    #     del pred
    #     # torch.cuda.empty_cache()
    #     return [pts0,pts1,lms]

    def lightglue(self, i, feats0):
        feats1 = self.hfile_local[self.db_name[i]]

        # Batch data transfer to GPU
        pred = {
            **{k+'0': torch.tensor(np.array(v)).unsqueeze(0) for k, v in feats0.items()},
            **{k+'1': torch.tensor(np.array(v)).unsqueeze(0) for k, v in feats1.items()},
        }
        pred = {k: v.to(self.device) for k, v in pred.items()}
        pred['keypoint_scores0'] = pred.pop('scores0')
        pred['keypoint_scores1'] = pred.pop('scores1')

        with torch.inference_mode():
            pred = self.local_feature_matcher(pred)
            # prof.step()

        matches = pred['matches0'][0].detach().cpu().short().numpy()
        registered_feature = self.registered_feature[i]
        index_list = set(registered_feature[0])
        feature_dict = {key: value for key, value in zip(registered_feature[0], registered_feature[1])}

        # Reuse memory allocation
        pts0, pts1, lms = self.match_filter(matches, feats0, feats1, feature_dict, index_list)

        return [pts0, pts1, lms]
    
    def superglue(self, i, feats0):
        feats1 = self.hfile_local[self.db_name[i]]
        data = {}
        for k in feats0.keys():
            data[k + '0'] = feats0[k]
        for k in feats0.keys():
            data[k + '1'] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(self.device)
                for k, v in data.items()}
        data['image0'] = torch.empty((1, 1,) + tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1,) + tuple(feats1['image_size'])[::-1])

        with torch.inference_mode():
            pred = self.local_feature_matcher(data)
        matches = pred['matches0'][0].detach().cpu().short().numpy()
        registered_feature = self.registered_feature[i]
        index_list = set(registered_feature[0])
        feature_dict = {key: value for key, value in zip(registered_feature[0], registered_feature[1])}
        pts0 = np.empty((len(matches), 2))
        pts1 = np.empty((len(matches), 2))
        lms = np.empty((len(matches),3))
        counter = 0
        for n, m in enumerate(matches):
            if m != -1 and m in index_list:
                pts0[counter] = feats0['keypoints'][n]
                pts1[counter] = feats1['keypoints'][m]
                lms[counter] = feature_dict[m]
                counter += 1
        pts0 = pts0[:counter]
        pts1 = pts1[:counter]
        lms = lms[:counter]
        return [pts0,pts1,lms]