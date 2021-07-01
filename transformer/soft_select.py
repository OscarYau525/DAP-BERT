##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, torch
import torch.nn as nn

# logits = [4, 3, 312] / [4, 1, 312]
# input must be 2-dim logits
def selectNwithP(logits, tau, num, just_prob=False, eps=1e-7):
  if tau <= 0:
    new_logits = logits
    probs = nn.functional.softmax(new_logits, dim=1)
  else       :
    while True: # a trick to avoid the gumbels bug
      gumbels = -torch.empty_like(logits).exponential_().log()
      new_logits = (logits.log_softmax(dim=1) + gumbels) / tau
      probs = nn.functional.softmax(new_logits, dim=1)
      if (not torch.isinf(gumbels).any()) and (not torch.isinf(probs).any()) and (not torch.isnan(probs).any()): break

  if just_prob: return probs

  #with torch.no_grad(): # add eps for unexpected torch error
  #  probs = nn.functional.softmax(new_logits, dim=1)
  #  selected_index = torch.multinomial(probs + eps, 2, False)
  # with torch.no_grad(): # add eps for unexpected torch error
  # probs          = probs.cpu()
  selected_index = torch.multinomial(probs + eps, num, False).to(logits.device)
  selected_logit = torch.gather(new_logits, 1, selected_index)
  selcted_probs  = nn.functional.softmax(selected_logit, dim=1)
  return selected_index, selcted_probs

def selectNwithP_1dimLogit(logits, tau, num, just_prob=False, eps=1e-7):
  if tau <= 0:
    new_logits = logits
    probs = nn.functional.softmax(new_logits, dim=0)
  else       :
    while True: # a trick to avoid the gumbels bug
      gumbels = -torch.empty_like(logits).exponential_().log()
      new_logits = (logits.log_softmax(dim=0) + gumbels) / tau
      probs = nn.functional.softmax(new_logits, dim=0)
      if (not torch.isinf(gumbels).any()) and (not torch.isinf(probs).any()) and (not torch.isnan(probs).any()): break

  if just_prob: return probs

  #with torch.no_grad(): # add eps for unexpected torch error
  #  probs = nn.functional.softmax(new_logits, dim=1)
  #  selected_index = torch.multinomial(probs + eps, 2, False)
  # with torch.no_grad(): # add eps for unexpected torch error
    # probs          = probs.cpu()
  selected_index = torch.multinomial(probs + eps, num, False).to(logits.device)
  selected_logit = torch.gather(new_logits, 0, selected_index)
  selcted_probs  = nn.functional.softmax(selected_logit, dim=0)
  return selected_index, selcted_probs

def selectNMax(logits, num):
  probs = nn.functional.softmax(logits, dim=-1)
  _, top_index = torch.topk(probs, num, dim=-1)  
  return top_index
