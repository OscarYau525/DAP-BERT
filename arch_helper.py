import numpy as np
from src.flop_benchmark import get_model_infos
import json

def show_arch_summary(arch, text, num_attention_heads=12):
    if 'qkv' in text:
        print('================%s===================='%(text))
        for layer_arch in arch:
            for head_arch in layer_arch:
                one_head = [len(qkv_arch)for qkv_arch in head_arch]
                print(one_head, end=' | ')
            print()
        print('====================================================')
    elif 'emb' in text:
        print('================%s===================='%(text))
        for layer_arch in arch:
            one_head = [len(qkv_arch)for qkv_arch in layer_arch]
            print(one_head)
        print('====================================================')
    elif 'ff' in text or 'skip' in text:
        print('================%s===================='%(text))
        one_layer = [len(layer_arch) for layer_arch in arch]
        print(one_layer, end=' | ')
        print()
        print('====================================================')
    elif 'multihead' in text:
        print('================%s===================='%(text))
        for lay in arch:
            status = np.zeros(num_attention_heads, dtype=np.int)
            status[lay] = 1
            print(status)
        print('====================================================')
    else:
        raise ValueError('incorrect text of arch %s'%text)

def count_arch_size(archs):
    def summingemb(arch):
        ret_sum = 0
        for layer_arch in arch:
            ret_sum += sum([len(qkv_arch)for qkv_arch in layer_arch])
        return ret_sum
    def summing(arch):
        ret_sum = 0
        for layer_arch in arch:
            for head_arch in layer_arch:
                ret_sum += sum([len(qkv_arch)for qkv_arch in head_arch])
        return ret_sum
    
    embedding_arch, qkv_head_arch, intermediate_arch, multihead_arch, sc_arch = archs
    emb_sum = 0
    qkv_head_sum = 0
    inter_sum = 0
    head_sum = 0
    sc_sum = 0
    if embedding_arch != None:
        emb_sum = summingemb(embedding_arch)
    if qkv_head_arch != None:
        qkv_head_sum = summing(qkv_head_arch)
    if intermediate_arch != None:
        inter_sum = sum([len(layer_arch)for layer_arch in intermediate_arch])
    if multihead_arch != None:
        head_sum = sum([len(layer_arch)for layer_arch in multihead_arch])
    if sc_arch != None:
        sc_sum = sum([len(layer_arch)for layer_arch in sc_arch])
    return emb_sum, qkv_head_sum, inter_sum, head_sum, sc_sum


def store_arch_summary(archs, f, searchable_size, model=None, one_input=None, total_flops=None, total_params=None, original_flops=None, num_attention_heads=12):
    def wrapper(sentence, writer):
        print(sentence)
        writer.write(sentence+'\n')

    def printer(arch, writer, name):
        if 'qkv' in name:
            wrapper('================%s===================='%(name), writer)
            for layer_arch in arch:
                for head_arch in layer_arch:
                    one_head = [len(qkv_arch)for qkv_arch in head_arch]
                    print(one_head, end=' | ')
                    writer.write(str(one_head))
                    writer.write(' | ')
                wrapper('', writer)
            wrapper('====================================================', writer)
        elif 'emb' in name:
            wrapper('================%s===================='%(name), writer)
            for layer_arch in arch:
                one_qkv = [len(qkv_arch)for qkv_arch in layer_arch]
                print(one_qkv, end=' | ')
                writer.write(str(one_qkv))
                writer.write(' | ')
            wrapper('', writer)
            wrapper('====================================================', writer)
        elif 'ff' in name or 'skip' in name:
            wrapper('================%s===================='%(name), writer)
            one_layer = [len(layer_arch)for layer_arch in arch]
            print(one_layer, end=' | ')
            writer.write(str(one_layer))
            writer.write(' | ')
            wrapper('', writer)
            wrapper('====================================================', writer)
        elif 'multihead' in name:
            wrapper('================%s===================='%(name), writer)
            for lay in arch:
                status = np.zeros(num_attention_heads, dtype=np.int)
                status[lay] = 1
                print(status)
                writer.write(str(status))
                writer.write('\n')
            wrapper('====================================================', writer)
        else:
            raise ValueError('incorrect name of arch %s'%name)
    
    embedding_arch, qkv_head_arch, intermediate_arch, multihead_arch, sc_arch = archs  
    emb_size, qkv_head_size, inter_size, multihead_size, sc_size = count_arch_size(archs)
    if model != None and one_input != None:
        flops, param = get_arch_flop(model, one_input)
    with open(f, 'w') as writer:
        if embedding_arch != None:
            printer(embedding_arch, writer, 'embedding length')
            wrapper('searched size: %d'%(emb_size), writer)
            wrapper('searched ratio: %.3f'%(emb_size / searchable_size['emb']), writer)
        if qkv_head_arch != None:
            printer(qkv_head_arch, writer, 'qkv head length')
            wrapper('searched size: %d'%(qkv_head_size), writer)
            wrapper('searched ratio: %.3f'%(qkv_head_size / searchable_size['qkv_hidden']), writer)
        if intermediate_arch != None:
            printer(intermediate_arch, writer, 'ff intermediate length')
            wrapper('searched size: %d'%(inter_size), writer)
            wrapper('searched ratio: %.3f'%(inter_size / searchable_size['ff']), writer)
        if multihead_arch != None:
            printer(multihead_arch, writer, 'multihead status')
            wrapper('searched size: %d'%(multihead_size), writer)
            wrapper('searched ratio: %.3f'%(multihead_size / searchable_size['multihead']), writer)
        if sc_arch != None:
            printer(sc_arch, writer, 'skip connection length')
            wrapper('searched size: %d'%(sc_size), writer)
            wrapper('searched ratio: %.3f'%(sc_size / searchable_size['sc']), writer)
        if one_input != None:
            wrapper("flops by TinyBERT: " + str(flops*1e6), writer)
            wrapper("param: " + str(param), writer)
        if total_flops != None:
            wrapper("flops by Searcher: " + str(total_flops), writer)
        if original_flops != None:
            wrapper("original flops of TinyBERT: " + str(original_flops), writer)
        if total_params != None:
            wrapper("total num of params: " + str(total_params), writer)
            
            

def get_arch_flop(model, one_input):
    flops, params = get_model_infos(model, one_input)
    return flops, params

def write_to_final_arch_txt(student_model, f_path):
    archs, _, _ = student_model.get_archs()
    with open(f_path, "w") as f:
        if archs[0] != None:
            # input embedding arch
            input_archs = []
            for layer in archs[0]:
                for a in layer:
                    input_archs.append(','.join([str(i) for i in a]))
            f.write('input_embedding:' + '|'.join(input_archs))
            f.write('\n')

        if archs[1] != None:
            # qkv arch
            qkv_archs = []
            for layer in archs[1]:
                for heads in layer:
                    for qk_v in heads:
                        qkv_archs.append(','.join([str(i) for i in qk_v]))
            f.write('qkv:' + '|'.join(qkv_archs))
            f.write('\n')
        
        if archs[2] != None:
            # ff arch
            ff_archs = []
            for layer in archs[2]:
                ff_archs.append(','.join([str(i) for i in layer]))
            f.write('ff:' + '|'.join(ff_archs))
            f.write('\n')
        
        if archs[3] != None:
            # multihead arch
            head_archs = []
            for layer in archs[3]:
                head_archs.append(','.join([str(i) for i in layer]))
            f.write('multihead:' + '|'.join(head_archs))
            f.write('\n')

        if archs[4] != None:
            # sc arch
            sc_archs = []
            for layer in archs[4]:
                sc_archs.append(','.join([str(i) for i in layer]))
            f.write('sc:' + '|'.join(sc_archs))
            f.write('\n')

def write_to_final_arch_json(student_model, f_path):
    archs, _, _ = student_model.get_archs()
    out_arch = {}
    if archs[2] != None:
        # ff arch
        out_arch["ff"] = archs[2]
    
    if archs[3] != None:
        # multihead arch
        out_arch["multihead"] = archs[3]
    print("******Ready to json dump")
    with open(f_path, "w") as f:
        json.dump(out_arch, f)
    