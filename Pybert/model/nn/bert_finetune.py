#encoding:utf-8
import torch.nn as nn
import torch
from pytorch_pretrained_bert.modeling import BertModel,BertPreTrainedModel
from Config.BasicConfig import ner_configs as config

class Bert_Finetune(BertPreTrainedModel):
    def __init__(self,bertConfig,num_classes):
        '''
        本质n分类器
        :param bertConfig:
        :param num_classes:
        '''
        super(Bert_Finetune,self).__init__(bertConfig)
        self.bert = BertModel(bertConfig)
        self.dropout = nn.Dropout(bertConfig.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=bertConfig.hidden_size,out_features=num_classes)
        self.apply(self.init_bert_weights)
        #train all parameters
        self.unfreeze_bert_encoder()

    def unfreeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = True

    def forward(self,input_ids,attention_mask,token_type_ids=None,label_ids =None,output_all_encodedLayers = False):
        #if token_type_ids is None:
            #token_type_ids = torch.zeros_like(input_ids)
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        encoded_layers,pooled_output = self.bert(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,
                                    output_all_encodedLayers = output_all_encodedLayers)
        pooled_output = self.dropout(pooled_output)
        logits  = self.classifier(pooled_output)
        return logits

class Bert_Ner_Finetune(BertPreTrainedModel):
    def __init__(self,bertConfig,num_classes):
        '''
        本质n分类器
        :param bertConfig:
        :param num_classes:
        '''
        super(Bert_Ner_Finetune,self).__init__(bertConfig)
        self.bert = BertModel(bertConfig)
        self.dropout = nn.Dropout(bertConfig.hidden_dropout_prob)

        self.apply(self.init_bert_weights)
        self.fc = nn.Linear(768, len(config['Ner']))
        #train all parameters
        self.unfreeze_bert_encoder()

    def unfreeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = True

    def forward(self,input_ids,token_type_ids=None,attention_mask=None,y =None,output_all_encodedLayers = False):
        #if token_type_ids is None:
        #    token_type_ids = torch.zeros_like(input_ids)
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        #Debug: Don't use 'input_ids=' , send to __call__ as **kwargs
        encoded_layer,pooled_output = self.bert(input_ids,token_type_ids,attention_mask,
                                                output_all_encoded_layers = False
                                                )
        logits = self.fc(encoded_layer)
        y_hat = logits.argmax(-1)
        return logits

class Bert_Mnli_Finetune(BertPreTrainedModel):
    def __init__(self,bertConfig,num_classes):
        '''
        本质n分类器
        :param bertConfig:
        :param num_classes:
        '''
        super(Bert_Mnli_Finetune,self).__init__(bertConfig)
        self.bert = BertModel(bertConfig)
        self.dropout = nn.Dropout(bertConfig.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=bertConfig.hidden_size,out_features=num_classes)
        self.apply(self.init_bert_weights)
        self.unfreeze_bert_encoder()

    def unfreeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = True

    def forward(self,input_ids,token_type_ids=None,attention_mask=None,output_all_encodedLayers = False):
        encoded_layers,pooled_output = self.bert(input_ids,token_type_ids,attention_mask,output_all_encoded_layers = False)
        pooled_output = self.dropout(pooled_output)
        logits  = self.classifier(pooled_output )
        y_hat = logits.argmax(-1)
        return logits

class Bert_BIO_Finetune(BertPreTrainedModel):
    def __init__(self,bertConfig):

        super(Bert_BIO_Finetune,self).__init__(bertConfig)
        self.bert = BertModel(bertConfig)
        '''
        self.output_v = nn.Linear(768, 1)
        self.output_arg0 = nn.Linear(768, 2)
        self.output_arg1 = nn.Linear(768, 2)
        '''
        self.output = nn.Linear(768, 5)
        self.apply(self.init_bert_weights)
        self.unfreeze_bert_encoder()

    def unfreeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = True

    def forward(self,input_ids,token_type_ids=None,attention_mask=None,y =None,output_all_encodedLayers = False):
        #if token_type_ids is None:
        #    token_type_ids = torch.zeros_like(input_ids)
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        #Debug: Don't use 'input_ids=' , send to __call__ as **kwargs
        encoded_layer,pooled_output = self.bert(input_ids,token_type_ids,attention_mask,
                                                output_all_encoded_layers = False
                                                )
        logits = self.output(encoded_layer)
        logits= tuple(logit.squeeze(-1) for logit in logits.split(1, dim=-1))
        v_logits,arg0_start_logits,arg0_end_logits,arg1_start_logits,arg1_end_logits=logits
        return [v_logits,arg0_start_logits,arg0_end_logits,arg1_start_logits, arg1_end_logits]
