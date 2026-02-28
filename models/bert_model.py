# import timm
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torch import nn
import torch.nn.functional as F
from .torchcrf import CRF
# from TorchCRF import CRF
from torch._C._nn import pad_sequence
from torchvision.transforms import transforms

from transformers import ViTModel, AlbertModel, CLIPVisionModel
# from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import RobertaModel, RobertaConfig, XLMRobertaModel, AlbertModel,BertModel,DistilBertPreTrainedModel


from .modeling_bert import BertModel
from .modeling_bert1 import BertModel1
from .modeling_bert2 import BertModel2
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50

from .modeling_bert3 import BertModel3
from .modeling_bert4 import BertModel4
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Image classification model
class ImageModel(nn.Module):
    """
    Image feature extractor using CLIP Vision Transformer
    Generates visual prompt embeddings for multimodal fusion
    """

    def __init__(self):
        super().__init__()
        # Load pre-trained CLIP vision model (local path for efficiency)
        self.clip = CLIPVisionModel.from_pretrained('D:/lyy/pycharm/project/HVPNeT-main/clip')

    def forward(self, images, aux_imgs=True):
        # Extract visual prompts from main images (13 layers of hidden states: 13x(bsz,50,768))
        prompt_guids = self.get_vision_prompt(images)

        # Process auxiliary images if provided (3 auxiliary image sets)
        if aux_imgs is not None:
            aux_prompt_guids = []
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])  # Reshape: 3xbszx3x224x224

            # Extract prompts for each auxiliary image set
            for i in range(len(aux_imgs)):
                aux_prompt_guid = self.get_vision_prompt(aux_imgs[i])
                aux_prompt_guids.append(aux_prompt_guid)
            return prompt_guids, aux_prompt_guids
        return prompt_guids, None

    def get_vision_prompt(self, x):
        """Extract all hidden states from CLIP as visual prompts"""
        prompt_guids = list(self.clip(x, output_hidden_states=True).hidden_states)
        return prompt_guids  # Output shape: 13x(bsz, seq_len, hidden_size)

# NER Core Model
class HMNeTREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(HMNeTREModel, self).__init__()
        self.bert1 = BertModel1.from_pretrained(args.bert_name, ignore_mismatched_sizes=True)
        self.bert2 = BertModel2.from_pretrained(args.bert_name, ignore_mismatched_sizes=True)
        self.bert3 = BertModel3.from_pretrained(args.bert_name, ignore_mismatched_sizes=True)
        self.bert4 = BertModel4.from_pretrained(args.bert_name, ignore_mismatched_sizes=True)
        self.bert = BertModel.from_pretrained('D:/lyy/pycharm/project/HVPNeT-main/bert-base-uncased')

        # self.object = DetrForObjectDetection.from_pretrained("D:\lyy\pycharm\project\HVPNeT-main\object", revision="no_timm")
        self.albert = AlbertModel.from_pretrained('D:/lyy/pycharm/project/HVPNeT-main/albert')
        self.vit = ViTModel.from_pretrained('D:/lyy/pycharm/project/HVPNeT-main/dino-vit')
        self.visual_projection = nn.Linear(768, 768)  # visual_dim = 768, text_dim = 768
        self.text_projection = nn.Linear(768, 768)  # text_dim = 768
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

        if args.use_prompt:
            self.image_model = ImageModel()  # bsz, 6, 56, 56
            # self.encoder_conv = nn.Sequential(
            #     nn.Linear(in_features=6400, out_features=1000),
            #     nn.Tanh(),
            #     nn.Linear(in_features=1000, out_features=8*768)
            # )
            self.encoder_conv = nn.Sequential(
                nn.Linear(in_features=41600, out_features=1000),
                nn.Tanh(),
                nn.Linear(in_features=1000, out_features=12 * 2 * 768)  # 6*2*768
            )

            self.gates2 = nn.ModuleList([nn.Linear(24 * 768 * 2, 4) for i in range(12)])
            self.dim1 = nn.Linear(197, 192)
            self.gate1 = nn.Linear(768, 1)
        if self.args.use_contrastive:
            self.temp = nn.Parameter(torch.ones([]) * 0.179)  # temp: 0.179

            self.vision_proj = nn.Sequential(
                nn.Linear(self.bert1.config.hidden_size, 4 * self.bert1.config.hidden_size),
                nn.Tanh(),
                nn.Linear(4 * self.bert1.config.hidden_size, self.args.embed_dim)
            )
            self.text_proj = nn.Sequential(
                nn.Linear(self.bert1.config.hidden_size, 4 * self.bert1.config.hidden_size),
                nn.Tanh(),
                nn.Linear(4 * self.bert1.config.hidden_size, self.args.embed_dim)
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        images=None,
        aux_imgs=None,
    ):

        bsz = attention_mask.size(0)
        with torch.no_grad():
            image, aux_images = self.image_model(images, aux_imgs)  # prompt_guids:13x(bsz,50,768) aux_prompt_guids:3(num)x13x(bsz,50,768)
        guids, image_guids, aux_image_guids = self.get_visual_prompt(image,aux_images)  # prompt_guids：6x(key,value), image_guids:(13,bsz,50,768) aux_image_guids:(3,13,bsz,50,768)
        image_atts = torch.ones((bsz, guids[0][0].shape[2])).to(self.args.device)
        prompt_attention_mask = torch.cat((image_atts, attention_mask), dim=1)
        if self.args.use_prompt:
            prompt_g = self.get_visual_aux_prompt(input_ids, images, aux_imgs)
            # print(prompt_guids[2].shape)
            prompt_g_length = prompt_g[0][0].shape[2]
            bsz = attention_mask.size(0)
            prompt_g_mask = torch.ones((bsz, prompt_g_length)).to(self.args.device)
            prompt_mask = torch.cat((prompt_g_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None
        text_features = self.bert(input_ids=input_ids)
        text_last_hidden_state = text_features.last_hidden_state
        bert_output1 = self.bert1(input_ids=input_ids,
                                  attention_mask=image_atts,
                                  token_type_ids=token_type_ids,
                                  past_key_values=guids,
                                  return_dict=True)
        sequence_output1 = bert_output1['last_hidden_state']  # bsz, len, hidden
        bert_output2 = self.bert2(input_ids=input_ids,
                                  attention_mask=prompt_mask,
                                  token_type_ids=token_type_ids,
                                  past_key_values=prompt_g,
                                  return_dict=True)

        sequence_output2 = bert_output2['last_hidden_state']  # 8,196,768
        bert_output3 = self.bert3(input_ids=input_ids,
                                  attention_mask=prompt_attention_mask,
                                  token_type_ids=token_type_ids,
                                  past_key_values=None,
                                  return_dict=True)
        sequence_output3 = bert_output3['last_hidden_state']
        bert_output4 = self.bert4(input_ids=input_ids,
                                  attention_mask=prompt_mask,
                                  token_type_ids=token_type_ids,
                                  past_key_values=prompt_g,
                                  return_dict=True)
        sequence_output4 = bert_output4['last_hidden_state']  # 8,196,768
        gate1 = torch.sigmoid(self.gate1(sequence_output1).squeeze(-1))
        gate2 = torch.sigmoid(self.gate1(sequence_output2).squeeze(-1))
        gate3 = torch.sigmoid(self.gate1(sequence_output3).squeeze(-1))
        # gate4 = torch.sigmoid(self.gate1(sequence_output4).squeeze(-1))
        gates = torch.stack([gate1, gate2, gate3], dim=-1)
        gates = F.softmax(gates, dim=-1)
        sequence_output = gates[..., 0:1] * sequence_output1 + gates[..., 1:2] * sequence_output2 + gates[..., 2:3] * sequence_output3 + 0.005 * sequence_output4

        bsz, seq_len, hidden_size =sequence_output.shape
        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = sequence_output[i, head_idx, :].squeeze()
            tail_hidden = sequence_output[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            text_feats = self.text_proj(text_last_hidden_state[:, 0, :])  # [CLS]
            image_feats = self.vision_proj(image[12][:, 0, :])  # [CLS] (bsz,768)
            cl_loss = self.get_contrastive_loss(text_feats, image_feats)
            main_loss = loss_fn(logits, labels.view(-1))
            loss = 0.6 * main_loss  + 0.4 * cl_loss  # 84.57
            return loss,logits
        return logits
    def get_contrastive_loss(self,image_feat, text_feat):
        logits = text_feat @ image_feat.t() / self.temp
        bsz = text_feat.shape[0]
        labels = torch.arange(bsz, device=self.args.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = loss_i2t + loss_t2i
        return loss
    def get_visual_prompt(self, images=None,aux_images=None):  # images:13x(bsz,50,768) aux_images:3(num)x13x(bsz,50,768)
        bsz = images[0].size(0)
        image_guids = torch.stack(images)  # image_guids:(13,bsz,50,768)
        aux_image_guids = torch.stack([torch.stack(aux_image) for aux_image in aux_images])  # aux_image_guids:(3，13，bsz,50,768)

        prompt_guids = torch.cat(images, dim=1).view(bsz, self.args.prompt_len, -1)  # (bsz,12, 41600)
        # prompt_guids = self.dropout(prompt_guids)
        prompt_guids = self.encoder_conv(prompt_guids)  # (bsz,12,12*2*768)
        split_prompt_guids = prompt_guids.split(768 * 2, dim=-1)  # 12x(bsz,12,768*2)

        aux_prompt_guids = [torch.cat(aux_image, dim=1).view(bsz, self.args.prompt_len, -1) for aux_image in
                            aux_images]  # 3x(13,bsz, 12, 41600)
        # aux_prompt_guids = [self.dropout(aux_prompt_guid) for aux_prompt_guid in
        #                     aux_prompt_guids]
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in
                            aux_prompt_guids]  # 3x(bsz, 12, 12*2*768)
        split_aux_prompt_guids = [aux_prompt_guid.split(768 * 2, dim=-1) for aux_prompt_guid in
                                  aux_prompt_guids]  # 3x(12x(bsz,12,768*2))

        result = []
        for idx in range(12):  # 6
            aux_key_vals = []  # 3 x [bsz, 12, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                aux_key_vals.append(split_aux_prompt_guid[idx])
            key_val = [split_prompt_guids[idx]] + aux_key_vals  # 4x(bsz,12,768*2))
            key_val = torch.cat(key_val, dim=1)  # (bsz,12*4,768*2)
            key_val = key_val.split(768, dim=-1)  # 2*(bsz,6*4,768)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1,64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result, image_guids, aux_image_guids  # image_guids:(13,bsz,50,768) aux_image_guids:#(3，13，bsz,50,768)
    def process_imgs(self, images, aux_imgs):
        # print(aux_imgs.shape)
        aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])  # 3(nums) x bsz x 3 x 224 x 224
        img_set1 = aux_imgs[0]  # 形状为 [8, 3, 224, 224]
        img_set2 = aux_imgs[1]  # 形状为 [8, 3, 224, 224]
        img_set3 = aux_imgs[2]  # 形状为 [8, 3, 224, 224]
        outputs1 = self.vit(img_set1)
        outputs2 = self.vit(img_set2)
        outputs3 = self.vit(img_set3)
        outputs = self.vit(images)
        last_hidden_state1 = outputs1.last_hidden_state
        last_hidden_state2 = outputs2.last_hidden_state
        last_hidden_state3 = outputs3.last_hidden_state
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state1, last_hidden_state2, last_hidden_state3, last_hidden_state
    def DynamicFilterNetwork(self, input_ids, images, aux_imgs):
        # 提取文本和视觉特征
        text_features = self.bert(input_ids=input_ids)
        last_hidden_state1, last_hidden_state2, last_hidden_state3, last_hidden_state = self.process_imgs(images,aux_imgs)
        visual_features = last_hidden_state
        visual_feature1 = last_hidden_state1
        visual_feature2 = last_hidden_state2
        visual_feature3 = last_hidden_state3

        # 投影视觉和文本特征
        visual_proj = self.visual_projection(visual_features)
        visual_proj1 = self.visual_projection(visual_feature1)
        visual_proj2 = self.visual_projection(visual_feature2)
        visual_proj3 = self.visual_projection(visual_feature3)
        text_proj = self.text_projection(text_features.last_hidden_state)
        # 通过自适应池化对视觉特征进行降维
        visual_proj_pooled = F.adaptive_avg_pool1d(visual_proj.transpose(1, 2), 192).transpose(1, 2)
        visual_proj_pooled1 = F.adaptive_avg_pool1d(visual_proj1.transpose(1, 2), 192).transpose(1, 2)
        visual_proj_pooled2 = F.adaptive_avg_pool1d(visual_proj2.transpose(1, 2), 192).transpose(1, 2)
        visual_proj_pooled3 = F.adaptive_avg_pool1d(visual_proj3.transpose(1, 2), 192).transpose(1, 2)
        # 计算相关性得分
        relevance_score = self.cosine_similarity(visual_proj_pooled, text_proj)
        relevance_score1 = self.cosine_similarity(visual_proj_pooled1, text_proj)
        relevance_score2 = self.cosine_similarity(visual_proj_pooled2, text_proj)
        relevance_score3 = self.cosine_similarity(visual_proj_pooled3, text_proj)

        # 将相关性得分低于 0 的部分置为 0
        relevance_score = torch.clamp(relevance_score, min=0)
        relevance_score1 = torch.clamp(relevance_score1, min=0)
        relevance_score2 = torch.clamp(relevance_score2, min=0)
        relevance_score3 = torch.clamp(relevance_score3, min=0)

        # 生成视觉掩码矩阵
        visual_mask = relevance_score.unsqueeze(-1)
        visual_mask1 = relevance_score1.unsqueeze(-1)
        visual_mask2 = relevance_score2.unsqueeze(-1)
        visual_mask3 = relevance_score3.unsqueeze(-1)
        # 使用视觉掩码过滤视觉特征
        filtered_visual = visual_mask * visual_proj_pooled
        filtered_visual1 = visual_mask1 * visual_proj_pooled1
        filtered_visual2 = visual_mask2 * visual_proj_pooled2
        filtered_visual3 = visual_mask3 * visual_proj_pooled3

        return filtered_visual, filtered_visual1, filtered_visual2, filtered_visual3

    def get_visual_aux_prompt(self,input_ids, images, aux_imgs):
        last_hidden_state,last_hidden_state1, last_hidden_state2, last_hidden_state3 = self.DynamicFilterNetwork(input_ids, images, aux_imgs)
        bsz = images.size(0)
        prompt_guids = last_hidden_state.view(bsz, 24, -1)

        aux_prompt_guid1 = last_hidden_state1.view(bsz, 24, -1)
        aux_prompt_guid2 = last_hidden_state2.view(bsz, 24, -1)
        aux_prompt_guid3 = last_hidden_state3.view(bsz, 24, -1)
        aux_prompt_guids = [aux_prompt_guid1, aux_prompt_guid2, aux_prompt_guid3]
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]
        prompt_guids1 = last_hidden_state.view(bsz, 48, -1)
        aux_prompt_guid11 = last_hidden_state1.view(bsz, 48, -1)
        aux_prompt_guid22 = last_hidden_state2.view(bsz, 48, -1)
        aux_prompt_guid33= last_hidden_state3.view(bsz,  48, -1)
        aux_prompt_guids_q = [aux_prompt_guid11, aux_prompt_guid22, aux_prompt_guid33]
        split_prompt_guids_q = prompt_guids1.split(768 , dim=-1)  # 4 x [bsz, 4, 768]
        split_aux_prompt_guids_q = [aux_prompt_guid.split(768 , dim=-1) for aux_prompt_guid in aux_prompt_guids_q]  # 3x [4 x [bsz, 4, 768*2]]

        result = []
        for idx in range(12):  # 12
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4
            # sum_prompt_guids = self.dropout(sum_prompt_guids)
            prompt_gate = F.softmax(F.hardswish(self.gates2[idx](sum_prompt_guids)), dim=-1)
            # prompt_gate = F.softmax(F.selu(self.gates2[idx](sum_prompt_guids)), dim=-1)
            k_v= torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768
            for i in range(4):
                k_v = k_v + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            aux_k_vs = []   # 3 x [bsz, 8, 768]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 8, 768
                # sum_aux_prompt_guids = self.dropout(sum_aux_prompt_guids)
                aux_prompt_gate = F.softmax(F.hardswish(self.gates2[idx](sum_aux_prompt_guids)), dim=-1)
                aux_k_v = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768
                for i in range(4):
                    aux_k_v = aux_k_v + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_k_vs.append(aux_k_v)
            key_val = [k_v] + aux_k_vs
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1,64).contiguous()

            sum_prompt_guids1 = torch.stack(split_prompt_guids_q).sum(0).view(bsz, -1) / 4
            prompt_gate1 = F.softmax(F.leaky_relu(self.gates2[idx](sum_prompt_guids1)), dim=-1)

            q = torch.zeros_like(split_prompt_guids_q[0]).to(self.args.device)  # bsz, 4, 768
            for i in range(4):
                q = q + torch.einsum('bg,blh->blh', prompt_gate1[:, i].view(-1, 1), split_prompt_guids_q[i])

            aux_qs = []  # 3 x [bsz, 8, 768]
            for split_aux_prompt_guid in split_aux_prompt_guids_q:
                sum_aux_prompt_guids1 = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4  # bsz, 8, 768
                aux_prompt_gate1 = F.softmax(F.leaky_relu(self.gates2[idx](sum_aux_prompt_guids1)), dim=-1)
                aux_q = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768
                for i in range(4):
                    aux_q = aux_q + torch.einsum('bg,blh->blh', aux_prompt_gate1[:, i].view(-1, 1),split_aux_prompt_guid[i])
                aux_qs.append(aux_q)
            q= [q] + aux_qs
            q = torch.cat(q, dim=1)
            query = q.reshape(bsz, 12, -1, 64).contiguous()
            temp_dict = (query,key, value)
            result.append(temp_dict)
        return result


# --------------------------- NER Core Model: HMNeTNERModel ---------------------------
class HMNeTNERModel(nn.Module):
    """
    Multimodal Named Entity Recognition (NER) Model
    Core Functionality: Fuses text (Albert) + vision (ViT/CLIP) features, uses CRF for sequence labeling
    Key Features: Dynamic visual prompt learning, contrastive alignment, and entity-aware feature fusion
    """

    def __init__(self, label_list, args):
        super(HMNeTNERModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len

        # --------------------------- Text Encoder (Albert: Optimized for NER) ---------------------------
        # Albert is lighter than BERT, better for sequence labeling tasks (e.g., NER)
        self.bert1 = BertModel1.from_pretrained(args.bert_name)
        self.bert2 = BertModel2.from_pretrained(args.bert_name)
        self.bert3 = BertModel3.from_pretrained(args.bert_name)
        self.bert4 = BertModel4.from_pretrained(args.bert_name)
        self.bert = BertModel.from_pretrained('D:/lyy/pycharm/project/HVPNeT-main/bert-base-uncased')

        # --------------------------- Vision Encoders (ViT/CLIP: Multimodal Fusion) ---------------------------
        self.albert = AlbertModel.from_pretrained('D:/lyy/pycharm/project/HVPNeT-main/albert')  # Text encoder for NER
        self.vit = ViTModel.from_pretrained(
            'D:/lyy/pycharm/project/HVPNeT-main/dino-vit')  # Vision encoder for visual features
        self.visual_projection = nn.Linear(768, 768)  # Align vision feature dimensions (768→768)
        self.text_projection = nn.Linear(768, 768)  # Align text feature dimensions
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)  # Compute text-vision relevance

        self.bert_config = self.bert1.config

        # --------------------------- Visual Prompt Learning (NER-Specific) ---------------------------
        if args.use_prompt:
            self.image_model = ImageModel()  # CLIP-based visual prompt extractor
            # Linear encoder to project visual prompts to Bert-compatible dimensions
            self.encoder_conv = nn.Sequential(
                nn.Linear(in_features=41600, out_features=1000),
                nn.Tanh(),
                nn.Linear(in_features=1000, out_features=12 * 2 * 768)  # 12 layers × 2 heads × 768 dim
            )
            # Dynamic gating layers for adaptive visual prompt fusion
            self.gates2 = nn.ModuleList([nn.Linear(24 * 768 * 2, 4) for i in range(12)])
            self.dim1 = nn.Linear(197, 192)
            self.gate1 = nn.Linear(768, 1)  # Gating for text-vision feature weighting

        # --------------------------- Contrastive Learning (NER Performance Boost) ---------------------------
        if self.args.use_contrastive:
            self.temp = nn.Parameter(
                torch.ones([]) * 0.179)  # Temperature for contrastive loss (critical for alignment)
            # Projection heads for text-vision contrastive learning
            self.vision_proj = nn.Sequential(
                nn.Linear(self.bert1.config.hidden_size, 4 * self.bert1.config.hidden_size),
                nn.Tanh(),
                nn.Linear(4 * self.bert1.config.hidden_size, self.args.embed_dim)
            )
            self.text_proj = nn.Sequential(
                nn.Linear(self.bert1.config.hidden_size, 4 * self.bert1.config.hidden_size),
                nn.Tanh(),
                nn.Linear(4 * self.bert1.config.hidden_size, self.args.embed_dim)
            )

        # --------------------------- NER-Specific Components ---------------------------
        self.num_labels = len(label_list)  # Number of NER labels (e.g., PER, LOC, ORG)
        print(f"[NER] Number of labels: {self.num_labels}")
        self.crf = CRF(self.num_labels,
                       batch_first=True)  # Critical for NER sequence labeling (handles label dependencies)
        self.fc = nn.Linear(768, self.num_labels)  # Linear layer to generate emission scores for CRF
        self.dropout = nn.Dropout(0.1)  # Dropout for regularization (prevents overfitting)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None,
                aux_imgs=None):
        """
        Forward Pass for NER Task:
        1. Extract visual prompts (no gradient for CLIP features → stable training)
        2. Fuse text/vision features via multi-path Bert encoders
        3. Generate emission scores for CRF sequence labeling
        4. Compute loss (CRF negative log-likelihood + contrastive loss)
        """
        bsz = attention_mask.size(0)  # Batch size (critical for tensor operations)

        # --------------------------- Step 1: Extract Visual Prompts (Multimodal Input) ---------------------------
        # No gradient for CLIP features → avoid visual feature contamination during text training
        with torch.no_grad():
            image, aux_images = self.image_model(images, aux_imgs)

        # Generate visual prompt key/value pairs (QKV) for Bert attention layers
        guids, image_guids, aux_image_guids = self.get_visual_prompt(image, aux_images)

        # Build attention mask for visual prompts + text (align dimensions for fusion)
        image_atts = torch.ones((bsz, guids[0][0].shape[2])).to(self.args.device)
        prompt_attention_mask = torch.cat((image_atts, attention_mask), dim=1)

        # --------------------------- Step 2: Generate Auxiliary Visual Prompts (if enabled) ---------------------------
        if self.args.use_prompt:
            prompt_g = self.get_visual_aux_prompt(input_ids, images, aux_imgs)
            prompt_g_length = prompt_g[0][0].shape[2]
            prompt_g_mask = torch.ones((bsz, prompt_g_length)).to(self.args.device)
            prompt_mask = torch.cat((prompt_g_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None

        # --------------------------- Step 3: Extract Text Features (Albert: NER-optimized) ---------------------------
        text_features = self.albert(input_ids=input_ids)  # Albert is lighter than BERT for sequence tasks
        text_last_hidden_state = text_features.last_hidden_state  # [CLS] token + sequence features

        # --------------------------- Step 4: Multi-Path Bert Encoding (Fusion Text + Vision) ---------------------------
        # Multi-path Bert: Each encoder handles a different aspect of text-vision fusion
        bert_output1 = self.bert1(input_ids=input_ids, attention_mask=image_atts, token_type_ids=token_type_ids,
                                  past_key_values=guids, return_dict=True)
        bert_output2 = self.bert2(input_ids=input_ids, attention_mask=prompt_mask, token_type_ids=token_type_ids,
                                  past_key_values=prompt_g, return_dict=True)
        bert_output3 = self.bert3(input_ids=input_ids, attention_mask=prompt_attention_mask,
                                  token_type_ids=token_type_ids, past_key_values=guids, return_dict=True)
        bert_output4 = self.bert4(input_ids=input_ids, attention_mask=prompt_mask, token_type_ids=token_type_ids,
                                  past_key_values=prompt_g, return_dict=True)

        # Extract sequence outputs from each Bert path
        sequence_output1 = bert_output1['last_hidden_state']
        sequence_output2 = bert_output2['last_hidden_state']
        sequence_output3 = bert_output3['last_hidden_state']
        sequence_output4 = bert_output4['last_hidden_state']

        # --------------------------- Step 5: Dynamic Gating (Fusion of Multi-Path Features) ---------------------------
        # Gating: Dynamically weight text-vision features to improve NER accuracy
        gate1 = torch.sigmoid(self.gate1(sequence_output1).squeeze(-1))
        gate2 = torch.sigmoid(self.gate1(sequence_output2).squeeze(-1))
        gate3 = torch.sigmoid(self.gate1(sequence_output3).squeeze(-1))
        gates = torch.stack([gate1, gate2, gate3], dim=-1)
        gates = F.softmax(gates, dim=-1)  # Softmax for feature weighting
        sequence_output = gates[..., 0:1] * sequence_output1 + gates[..., 1:2] * sequence_output2 + gates[...,
                                                                                                    2:3] * sequence_output3 + 0.005 * sequence_output4

        # --------------------------- Step 6: Generate Emission Scores (CRF Input) ---------------------------
        # Dropout + FC layer → generate emission scores for CRF (sequence labeling)
        sequence_output2 = self.dropout(sequence_output)
        emissions2 = self.fc(sequence_output2)  # Shape: (bsz, seq_len, num_labels)

        # --------------------------- Step 7: CRF Decoding (NER Inference) ---------------------------
        # CRF: Decode the best sequence of labels (handles label dependencies)
        logits = self.crf.decode(emissions2, attention_mask.byte())  # Decode the optimal label sequence

        # --------------------------- Step 8: Compute Loss (Training Phase) ---------------------------
        loss = None
        if labels is not None:
            # --------------------------- Text-Visual Contrastive Loss ---------------------------
            # Align text and visual features (improves NER performance by linking text context to visual cues)
            text_feats = self.text_proj(text_last_hidden_state[:, 0, :])  # [CLS] token (text feature)
            image_feats = self.vision_proj(image[12][:, 0, :])  # [CLS] token (visual feature)
            cl_loss = self.get_contrastive_loss(text_feats, image_feats)

            # --------------------------- CRF Loss (NER Core Loss) ---------------------------
            # Negative log-likelihood (NLL) → measures the quality of the label sequence
            main_loss = -1 * self.crf(emissions2, labels, mask=attention_mask.byte(), reduction='mean')

            # --------------------------- Weighted Loss Combination (Tuned for NER) ---------------------------
            # 0.8 for CRF loss (NER core) + 0.2 for contrastive loss (alignment)
            loss = 0.8 * main_loss + 0.2 * cl_loss

        # Return standard HuggingFace output format (critical for integration with training pipelines)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    # --------------------------- Utility Methods (NER-Specific) ---------------------------
    def get_contrastive_loss(self, image_feat, text_feat):
        """
        Compute contrastive loss for text-vision alignment (NER performance boost)
        Implements symmetric InfoNCE loss (image→text + text→image)
        """
        logits = text_feat @ image_feat.t() / self.temp  # Temperature-scaled similarity
        bsz = text_feat.shape[0]
        labels = torch.arange(bsz, device=self.args.device)  # Self-supervised labels (no need for manual labels)
        loss_i2t = F.cross_entropy(logits, labels)  # Image to text loss
        loss_t2i = F.cross_entropy(logits.t(), labels)  # Text to image loss
        loss = loss_i2t + loss_t2i
        return loss

    def get_visual_prompt(self, images=None, aux_images=None):
        """
        Generate visual prompt key/value pairs (QKV) for Bert attention layers
        Key: For NER, this is used to align visual features with text features
        """
        bsz = images[0].size(0)
        image_guids = torch.stack(images)
        aux_image_guids = torch.stack([torch.stack(aux_image) for aux_image in aux_images])

        # Reshape and project visual features to prompt dimensions
        prompt_guids = torch.cat(images, dim=1).view(bsz, self.args.prompt_len, -1)
        prompt_guids = self.encoder_conv(prompt_guids)
        split_prompt_guids = prompt_guids.split(768 * 2, dim=-1)

        # Process auxiliary image prompts
        aux_prompt_guids = [torch.cat(aux_image, dim=1).view(bsz, self.args.prompt_len, -1) for aux_image in aux_images]
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids]
        split_aux_prompt_guids = [aux_prompt_guid.split(768 * 2, dim=-1) for aux_prompt_guid in aux_prompt_guids]

        # Build QKV for each Bert layer (12 layers)
        result = []
        for idx in range(12):
            aux_key_vals = [split_aux_prompt_guid[idx] for split_aux_prompt_guid in split_aux_prompt_guids]
            key_val = [split_prompt_guids[idx]] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            # Reshape to match Bert's attention head dimensions (64 per head)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1,
                                                                                              64).contiguous()
            result.append((key, value))
        return result, image_guids, aux_image_guids

    def process_imgs(self, images, aux_imgs):
        """Process main and auxiliary images with ViT to extract visual features (NER input)"""
        aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])  # Reshape to (3 sets) x bsz x 3 x 224 x 224
        img_set1, img_set2, img_set3 = aux_imgs[0], aux_imgs[1], aux_imgs[2]

        # Extract ViT features for all image sets
        outputs1, outputs2, outputs3 = self.vit(img_set1), self.vit(img_set2), self.vit(img_set3)
        outputs = self.vit(images)

        return outputs1.last_hidden_state, outputs2.last_hidden_state, outputs3.last_hidden_state, outputs.last_hidden_state

    def DynamicFilterNetwork(self, input_ids, images, aux_imgs):
        """
        Dynamic Filter Network (DFN) for NER:
        Filters visual features based on text-vision relevance (improves NER accuracy)
        Key: Only keeps visual features that are relevant to the text (e.g., entity names)
        """
        # Extract text and vision features
        text_features = self.bert(input_ids=input_ids)
        last_hidden_state1, last_hidden_state2, last_hidden_state3, last_hidden_state = self.process_imgs(images,
                                                                                                          aux_imgs)

        # Project features to common dimension
        visual_proj = self.visual_projection(last_hidden_state)
        visual_proj1 = self.visual_projection(last_hidden_state1)
        visual_proj2 = self.visual_projection(last_hidden_state2)
        visual_proj3 = self.visual_projection(last_hidden_state3)
        text_proj = self.text_projection(text_features.last_hidden_state)

        # Adaptive pooling to align sequence length (192)
        visual_proj_pooled = F.adaptive_avg_pool1d(visual_proj.transpose(1, 2), 192).transpose(1, 2)
        visual_proj_pooled1 = F.adaptive_avg_pool1d(visual_proj1.transpose(1, 2), 192).transpose(1, 2)
        visual_proj_pooled2 = F.adaptive_avg_pool1d(visual_proj2.transpose(1, 2), 192).transpose(1, 2)
        visual_proj_pooled3 = F.adaptive_avg_pool1d(visual_proj3.transpose(1, 2), 192).transpose(1, 2)

        # Compute cosine similarity (text-vision relevance)
        relevance_score = self.cosine_similarity(visual_proj_pooled, text_proj)
        relevance_score1 = self.cosine_similarity(visual_proj_pooled1, text_proj)
        relevance_score2 = self.cosine_similarity(visual_proj_pooled2, text_proj)
        relevance_score3 = self.cosine_similarity(visual_proj_pooled3, text_proj)

        # Clamp negative scores (filter out irrelevant visual features)
        relevance_score = torch.clamp(relevance_score, min=0)
        relevance_score1 = torch.clamp(relevance_score1, min=0)
        relevance_score2 = torch.clamp(relevance_score2, min=0)
        relevance_score3 = torch.clamp(relevance_score3, min=0)

        # Generate masks and apply to visual features
        visual_mask = relevance_score.unsqueeze(-1)
        visual_mask1 = relevance_score1.unsqueeze(-1)
        visual_mask2 = relevance_score2.unsqueeze(-1)
        visual_mask3 = relevance_score3.unsqueeze(-1)

        filtered_visual = visual_mask * visual_proj_pooled
        filtered_visual1 = visual_mask1 * visual_proj_pooled1
        filtered_visual2 = visual_mask2 * visual_proj_pooled2
        filtered_visual3 = visual_mask3 * visual_proj_pooled3

        return filtered_visual, filtered_visual1, filtered_visual2, filtered_visual3

    def get_visual_aux_prompt(self, input_ids, images, aux_imgs):
        """
        Generate auxiliary visual prompts with QKV for Bert attention
        Key: For NER, this is used to align visual features with text features
        """
        # Get filtered visual features from DFN
        last_hidden_state, last_hidden_state1, last_hidden_state2, last_hidden_state3 = self.DynamicFilterNetwork(
            input_ids, images, aux_imgs)
        bsz = images.size(0)

        # Reshape visual features for prompt generation
        prompt_guids = last_hidden_state.view(bsz, 24, -1)
        aux_prompt_guid1 = last_hidden_state1.view(bsz, 24, -1)
        aux_prompt_guid2 = last_hidden_state2.view(bsz, 24, -1)
        aux_prompt_guid3 = last_hidden_state3.view(bsz, 24, -1)
        aux_prompt_guids = [aux_prompt_guid1, aux_prompt_guid2, aux_prompt_guid3]

        # Split features into Q/K/V components
        split_prompt_guids = prompt_guids.split(768 * 2, dim=-1)
        split_aux_prompt_guids = [aux_prompt_guid.split(768 * 2, dim=-1) for aux_prompt_guid in aux_prompt_guids]

        prompt_guids1 = last_hidden_state.view(bsz, 48, -1)
        aux_prompt_guid11 = last_hidden_state1.view(bsz, 48, -1)
        aux_prompt_guid22 = last_hidden_state2.view(bsz, 48, -1)
        aux_prompt_guid33 = last_hidden_state3.view(bsz, 48, -1)
        aux_prompt_guids_q = [aux_prompt_guid11, aux_prompt_guid22, aux_prompt_guid33]

        split_prompt_guids_q = prompt_guids1.split(768, dim=-1)
        split_aux_prompt_guids_q = [aux_prompt_guid.split(768, dim=-1) for aux_prompt_guid in aux_prompt_guids_q]

        # Generate QKV prompts for each Bert layer (12 layers)
        result = []
        for idx in range(12):
            # Compute gate weights for key/value fusion
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4
            prompt_gate = F.softmax(F.hardswish(self.gates2[idx](sum_prompt_guids)), dim=-1)

            # Weighted fusion of key/value prompts
            k_v = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)
            for i in range(4):
                k_v = k_v + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            # Process auxiliary key/value prompts
            aux_k_vs = []
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4
                aux_prompt_gate = F.softmax(F.hardswish(self.gates2[idx](sum_aux_prompt_guids)), dim=-1)
                aux_k_v = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)
                for i in range(4):
                    aux_k_v = aux_k_v + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1),
                                                     split_aux_prompt_guid[i])
                aux_k_vs.append(aux_k_v)

            # Combine main + auxiliary K/V and reshape
            key_val = [k_v] + aux_k_vs
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1,
                                                                                              64).contiguous()

            # Compute gate weights for query fusion
            sum_prompt_guids1 = torch.stack(split_prompt_guids_q).sum(0).view(bsz, -1) / 4
            prompt_gate1 = F.softmax(F.leaky_relu(self.gates2[idx](sum_prompt_guids1)), dim=-1)

            # Weighted fusion of query prompts
            q = torch.zeros_like(split_prompt_guids_q[0]).to(self.args.device)
            for i in range(4):
                q = q + torch.einsum('bg,blh->blh', prompt_gate1[:, i].view(-1, 1), split_prompt_guids_q[i])

            # Process auxiliary query prompts
            aux_qs = []
            for split_aux_prompt_guid in split_aux_prompt_guids_q:
                sum_aux_prompt_guids1 = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4
                aux_prompt_gate1 = F.softmax(F.leaky_relu(self.gates2[idx](sum_aux_prompt_guids1)), dim=-1)
                aux_q = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)
                for i in range(4):
                    aux_q = aux_q + torch.einsum('bg,blh->blh', aux_prompt_gate1[:, i].view(-1, 1),
                                                 split_aux_prompt_guid[i])
                aux_qs.append(aux_q)

            # Combine main + auxiliary query and reshape
            q = [q] + aux_qs
            q = torch.cat(q, dim=1)
            query = q.reshape(bsz, 12, -1, 64).contiguous()

            # Store QKV tuple for current layer
            result.append((query, key, value))
        return result
