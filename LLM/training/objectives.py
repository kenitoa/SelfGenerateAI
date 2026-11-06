"""Explicit opt-in objectives. All inputs still require approved snapshots."""
import json
from LLM.core import require

def prepare(tokenizer,row,maximum,objective,encode_sft):
    if objective=='dpo':
        try:pair=json.loads(row['output'])
        except (ValueError,TypeError):pair={}
        require(isinstance(pair,dict) and all(isinstance(pair.get(k),str) and pair[k].strip() for k in ('chosen','rejected')) and pair['chosen']!=pair['rejected'],'INVALID_PREFERENCE_EXAMPLE')
        return {k:encode_sft(tokenizer,{**row,'output':pair[k]},maximum) for k in ('chosen','rejected')}
    if objective=='cpt':
        values=tokenizer.encode(row['output'],add_special_tokens=False)
        require(2<=len(values)<=maximum,'INPUT_TOO_LARGE_FOR_TRAINING')
        return {'input_ids':values,'labels':list(values)}
    return encode_sft(tokenizer,row,maximum)

def tensors(torch,rows,pad,device):
    width=max(len(r['input_ids']) for r in rows)
    return {'input_ids':torch.tensor([r['input_ids']+[pad]*(width-len(r['input_ids'])) for r in rows],device=device),
        'labels':torch.tensor([r['labels']+[-100]*(width-len(r['labels'])) for r in rows],device=device),
        'attention_mask':torch.tensor([[1]*len(r['input_ids'])+[0]*(width-len(r['input_ids'])) for r in rows],device=device)}

def loss(torch,model,rows,pad,device,objective,beta=.1):
    if objective!='dpo':return model(**tensors(torch,rows,pad,device)).loss
    import torch.nn.functional as functional
    chosen=tensors(torch,[r['chosen'] for r in rows],pad,device)
    rejected=tensors(torch,[r['rejected'] for r in rows],pad,device)
    def log_probability(batch):
        logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits[:,:-1,:].float()
        labels=batch['labels'][:,1:];mask=labels!=-100
        selected=labels.masked_fill(~mask,0)
        return (functional.log_softmax(logits,-1).gather(-1,selected.unsqueeze(-1)).squeeze(-1)*mask).sum(-1)
    was_training=model.training
    model.eval()
    with torch.no_grad(),model.disable_adapter():reference_chosen=log_probability(chosen);reference_rejected=log_probability(rejected)
    model.train(was_training)
    policy_chosen=log_probability(chosen);policy_rejected=log_probability(rejected)
    return -functional.logsigmoid(beta*((policy_chosen-policy_rejected)-(reference_chosen-reference_rejected))).mean()
