import torch
class SearchDatasetIterator():
    def __init__(self, output_mode, features, batch_size):
        self.len = len(features)
        self.all_seq_length = torch.tensor([f.seq_length for f in features], dtype=torch.long)
        self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        self.all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            self.all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            self.all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        self.batch_size = batch_size
        self.index = 0
    
    def __next__(self):
        if self.index + self.batch_size > self.len:
            tail = self.len - self.index
            batch_seq_length = torch.cat([self.all_seq_length[self.index:], self.all_seq_length[0:self.batch_size-tail]], dim=0)
            batch_input_ids = torch.cat([self.all_input_ids[self.index:], self.all_input_ids[0:self.batch_size-tail]], dim=0)
            batch_input_mask = torch.cat([self.all_input_mask[self.index:], self.all_input_mask[0:self.batch_size-tail]], dim=0)
            batch_segment_ids = torch.cat([self.all_segment_ids[self.index:], self.all_segment_ids[0:self.batch_size-tail]], dim=0)
            batch_label_ids = torch.cat([self.all_label_ids[self.index:], self.all_label_ids[0:self.batch_size-tail]], dim=0)
            self.index = (self.index + self.batch_size) % self.len
            return batch_seq_length, batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids
        else:  
            batch_seq_length = self.all_seq_length[self.index:self.index + self.batch_size]
            batch_input_ids = self.all_input_ids[self.index:self.index + self.batch_size]
            batch_input_mask = self.all_input_mask[self.index:self.index + self.batch_size]
            batch_segment_ids = self.all_segment_ids[self.index:self.index + self.batch_size]
            batch_label_ids = self.all_label_ids[self.index:self.index + self.batch_size]
            self.index += self.batch_size
            return batch_seq_length, batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids

    def __len__(self):
        return self.len
