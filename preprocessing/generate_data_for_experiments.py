import argparse
import json
import os
from functools import reduce
import random

def get_parameter():
    parser = argparse.ArgumentParser(description='Generate dataset for training the model')

    parser.add_argument('-input_path', action="store", default="../raw_data/")
    parser.add_argument('-out_path', action="store", default="../data/")
    parser.add_argument('-dataset', action="store", default="snips")

    args = parser.parse_args()
    return args

def get_service(service):
    return service.split("_")[0] 

def get_slots_sgd(path, intents):
    slot2desc = {}
    data_type = ["train", "dev", "test"]
    for data in data_type:
        with open(path+data+"/schema.json") as f:
            data = json.load(f)
            for item in data:
                
                domain = get_service(item["service_name"])

                if domain not in intents:
                    domain = "others"
                
                if domain in slot2desc:
                    domain_slots = slot2desc[domain]
                else:
                    domain_slots = {}
        
                for slot in item["slots"]:
                    domain_slots[slot['name']] = slot['description']

                slot2desc[domain] = domain_slots 
    
    return slot2desc 

def get_slots_for_domain(domain, slots):
    domain_slots = slots[domain]
    other_slots = {}
    for k,v in slots.items():
        for k1,v1 in v.items():
            other_slots[k1] = v1

    return domain_slots, other_slots

def get_domains(path):
    data_type = ["train", "dev", "test"]
    intent2domain = {}
    for data in data_type:
        with open(path+data+"/schema.json") as f:
            data = json.load(f)
            for item in data:
                domain = get_service(item["service_name"])
            
                for intent in item['intents']:
                    intent2domain[intent['name']] = domain
                
    return intent2domain


def all_intents(path):
    all_intents = [x[0].replace(path, "") for x in os.walk(path) if len(x[0].replace(path, "") ) > 0 
           and x[0].replace(path, "")[0]!="."]
    return all_intents

def get_slot2desc(path):
    slots2desc = {}
    with open(path+"schema.json") as f:
        data = json.load(f)
        for item in data:
            all_slots = item["slots"]
            
            for slot in all_slots:
                slots2desc[slot["name"]] = slot["description"]
        
    return slots2desc

def get_slots(line):
    label = line.split("\t")[1]
    slot_type = [token[2:] for token in label.split() 
                       if token[0] != "O" and token[0] != "I"]
    return slot_type

def get_all_slots(lines):
    all_slots = [get_slots(line) for line in lines]
    all_slots = reduce(lambda x,y: x+y, all_slots)
    return list(set(all_slots))

def gen_positive(sentence, label, slots, slot2desc, others, step1_labels = False):
    if step1_labels:
        iob = " ".join([token[0] for token in label.split()])
    pos = []
    for slot in slots:
        new_label = ["B" if token[0] == "B" and token[2:] == slot 
         else ("I" if token[0] == "I" and token[2:] == slot else "O") 
         for token in label.split()]
        if step1_labels:
            if slot2desc is not None:
                if slot in slot2desc:
                    pos.append(sentence+"\t"+iob+"\t"+slot2desc[slot]+"\t"+" ".join(new_label))
                elif others is not None and slot in others:
                    pos.append(sentence+"\t"+iob+"\t"+others[slot]+"\t"+" ".join(new_label))
            else:
                pos.append(sentence+"\t"+iob+"\t"+slot.replace("_", " ").replace(".", " ")+"\t"+" ".join(new_label))
        else:
            if slot2desc is not None:
                if slot in slot2desc:
                    pos.append(sentence+"\t"+slot2desc[slot]+"\t"+" ".join(new_label))
                elif others is not None and slot in others:
                    pos.append(sentence+"\t"+others[slot]+"\t"+" ".join(new_label))
            else:
                pos.append(sentence+"\t"+slot.replace("_", " ").replace(".", " ")+"\t"+" ".join(new_label))
    return pos        

def gen_neg(sentence, true_label, slots, neg_ratio, slot2desc, others, step1_labels = False):
    if step1_labels:
        iob = " ".join([token[0] for token in true_label.split()])
        
    label = " ".join(["O"] * len(sentence.split()))
    neg_slots = random.sample(set(slots), neg_ratio) if len(slots) > neg_ratio else slots
    neg = []
    for slot in neg_slots:
        if step1_labels:
            if slot2desc is not None:
                if slot in slot2desc:
                    neg.append(sentence+"\t"+iob+"\t"+slot2desc[slot]+"\t"+label)
                elif others is not None and slot in others:
                    neg.append(sentence+"\t"+iob+"\t"+others[slot]+"\t"+label)
            else:
                neg.append(sentence+"\t"+iob+"\t"+slot.replace("_", " ").replace(".", " ")+"\t"+label)
        else:
            if slot2desc is not None:
                if slot in slot2desc:
                    neg.append(sentence+"\t"+slot2desc[slot]+"\t"+label)
                elif others is not None and slot in others:
                    neg.append(sentence+"\t"+others[slot]+"\t"+label)
            else:
                neg.append(sentence+"\t"+slot.replace("_", " ").replace(".", " ")+"\t"+label)
    return neg

def get_example(line, slots, slot2desc, others = None, neg_ratio=2):
    ex_slots = get_slots(line)
    neg_slots = list(set(slots) - set(ex_slots))
    
    sen, label = line.split("\t")
    sen = sen.strip()
    if "B" in label:
        pos_ex = gen_positive(sen, label, ex_slots, slot2desc, others, True)
        neg_ex = gen_neg(sen, label, neg_slots, neg_ratio, slot2desc, others, True)
    
        return pos_ex + neg_ex
    else:
        neg = random.sample(set(slots), 1)[0]
        return [sen+"\t"+label+"\t"+neg.replace("-", " ")+"\t"+label]

def read_data(path):
    in_lines = open(path, "r").readlines()
    return [line.strip() for line in in_lines if len(line) > 0 ]
    
def write_out(path, all_examples):
    f = open(path, "w")
    for line in all_examples:
        f.write(line+"\n")
    f.close()

if __name__ == '__main__':
    args = get_parameter()
    input_path = args.input_path
    out_path = args.out_path
    dataset = args.dataset

    intents = all_intents(out_path+dataset+"/")

    if dataset == "sgd":
        domain_slot2desc = get_slots_sgd(input_path+dataset+"/", intents)
        
    elif dataset == "multiwoz":
        slot2desc = get_slot2desc(input_path+dataset+"/")
        rest = None

    elif dataset == "snips" or dataset == "atis":
        slot2desc, rest = None, None

    else:
        print(dataset,"not supported yet")


    for domain in intents:
        in_lines = read_data(out_path+dataset+"/"+domain+"/"+domain+".txt")

        all_slots = get_all_slots(in_lines)

        if dataset == "sgd":
            slot2desc, rest = get_slots_for_domain(domain, domain_slot2desc)

        all_examples = [get_example(line, all_slots, slot2desc, rest) for line in in_lines]
        all_examples = reduce(lambda x,y: x+y, all_examples)
        
        f_name = out_path+dataset+"/"+domain+"/"+domain+"_neg.txt"
        write_out(f_name, all_examples)
        print("file written to:", f_name)


    