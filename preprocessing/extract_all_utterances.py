import argparse
import json
import os

def get_parameter():
    parser = argparse.ArgumentParser(description='Extract all utterances from the dataset')

    parser.add_argument('-input_path', action="store", default="../raw_data/")
    parser.add_argument('-out_path', action="store", default="../data/")
    parser.add_argument('-dataset', action="store", default="snips")

    args = parser.parse_args()
    return args

def get_service(service):
    return service.split("_")[0] 

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

def get_starting_index(utterance, slot):
    tokens = utterance.lower().split()
    index = utterance.lower().find(slot.lower())
    
    pre_slot = utterance[:index]
    token_index = len(pre_slot.split())

    if token_index < len(tokens):
        token = tokens[token_index].replace(",", "").replace(".", "").replace("?", "").replace("!", "")
        if token == slot.split()[0]:
            return token_index
        else:
            return -1
    else:
        return -1


def get_starting_index_sgd(utterance, slot, index):
    if index == 0:
        return 0
    req_index = 0
    for my_index, word in enumerate(utterance.split()):
        req_index += len(list(word) )+1
        if req_index >=index:
            return my_index+1 if my_index+1 < len(utterance.split()) else len(utterance.split())-1
    

def update_iob(iob, index, entity, length):
    iob[index] = "B-"+entity
    for n_index in range(index+1, index+length):
        iob[n_index] = "I-"+entity
    return iob

def update_iob_sgd(iob, index, entity, val):
    iob[index] = "B-"+entity
    length = len(val.split())
    last_index = index+length if (index+length) < len(iob) else len(iob)
    for n_index in range(index+1, last_index):
        iob[n_index] = "I-"+entity
    return iob

def get_utterances(path):
    data_types = ["train", "valid", "test"] 

    in_lines, out_lines, intents = [], [], []
    for data in data_types:
        lines = open(path+data+"/seq.in", "r").readlines()
        in_lines.extend([line.strip() for line in lines if len(line) > 0 ])

        lines = open(path+data+"/seq.out", "r").readlines()
        out_lines.extend([line.strip() for line in lines if len(line) > 0 ])

        lines = open(path+data+"/label", "r").readlines()
        intents.extend([line.strip() for line in lines if len(line) > 0 ])

    all_lines = {}
    for index in range(len(in_lines)):
        if intents[index] not in all_lines:
            all_lines[intents[index]] = list()
        
        all_lines[intents[index]].append(in_lines[index]+"\t"+out_lines[index])

    return all_lines 

def get_utterances_multiwoz(path, hack_map):
    data_type = ["train", "dev", "test"]
    sen_intent_iob = []

    for train in data_type:
        for file_index in range(1, 18):
            f_name = '{:03d}'.format(file_index)
            file_path = path+train+"/dialogues_"+f_name+".json"
            
            if not os.path.exists(file_path):
                continue
            
            with open(file_path) as f:
                data = json.load(f)
                for item in data:
                    if "turns" in item:
                        turns = item["turns"]

                        for turn in turns:

                            if "speaker" in turn  and turn["speaker"] == "USER" and "utterance" in turn:
                                utterance = turn["utterance"]

                                if "frames" in turn:
                                    frames = turn["frames"]
                                    for frame in frames:
                                        #intent, slot_values
                                        if "state" in frame and "active_intent" in frame["state"] \
                                        and frame["state"]["active_intent"] != "NONE" and "slot_values" in frame["state"] \
                                        and len(frame["state"]["slot_values"]) > 0:
                                            active_intent = frame["state"]["active_intent"]
                                            
                                            iob = ["O"] * len(utterance.split())
                                            slot_values = frame["state"]["slot_values"]

                                            # slot_values
                                            if "slots" in frame and len(frame["slots"]) > 0:
                                                slots = frame["slots"]
                                                for slot in slots:
                                                    k = slot["slot"]
                                                    v = slot["value"]
                                                    if isinstance(v, list):
                                                        v = v[0]

                                                    if v.lower()!= "yes": 
                                                        index = utterance.lower().find(v.lower())
                                                        if index == -1 and v in hack_map:
                                                            index = utterance.lower().find(hack_map[v])
                                                            if index != -1:
                                                                v = hack_map[v]
                                                        

                                                        if  index != -1:
                                                            
                                                            start_index = get_starting_index(utterance,v.lower())

                                                            if start_index!= -1:
                                                                iob = update_iob(iob, start_index, k, len(v.split()))

                                            # more slot_values
                                            for k, v in slot_values.items():
                                                value = v[0]
                                                if value.lower()!= "yes": 
                                                    index = utterance.lower().find(value.lower())
                                                    
                                                    if index == -1 and value in hack_map:
                                                        index = utterance.lower().find(hack_map[value])
                                                        if index != -1:
                                                            value = hack_map[value]

                                                    if  index != -1:
                                                        
                                                        start_index = get_starting_index(utterance,value.lower())

                                                        if start_index!= -1:
                                                            iob = update_iob(iob, start_index, k, len(value.split()))

                                            f_sen = utterance+"\t"+active_intent+"\t"+" ".join(iob)
                                            
                                            sen_intent_iob.append(f_sen)  
                                            
                                        else:
                                            continue



                            else: # system
                                if "speaker" in turn  and turn["speaker"] == "SYSTEM" and "utterance" in turn:
                                    
                                    if "frames" in turn:
                                        utterance = turn["utterance"]
                                        
                                        iob = ["O"] * len(utterance.split())
                                        
                                        frames = turn["frames"]
                                        for frame in frames:
                                            # slot_values
                                            if "slots" in frame and len(frame["slots"]) > 0:
                                                slots = frame["slots"]
                                                for slot in slots:
                                                    k = slot["slot"]
                                                    v = slot["value"]
                                                    
                                                    if isinstance(v, list):
                                                        v = v[0]

                                                    if v.lower()!= "yes": 
                                                        index = utterance.lower().find(v.lower())
                                                        if index == -1 and v in hack_map:
                                                            index = utterance.lower().find(hack_map[v])
                                                            if index != -1:
                                                                v = hack_map[v]
                                                        

                                                        if  index != -1:
                                                            
                                                            start_index = get_starting_index(utterance,v.lower())

                                                            if start_index!= -1:
                                                                iob = update_iob(iob, start_index, k, len(v.split()))
                                                                
                                            f_sen = utterance+"\t"+active_intent+"\t"+" ".join(iob)
                                            
                                            sen_intent_iob.append(f_sen)  
                                            
                                        else:
                                            continue
                                
    all_lines = {}
    for line in sen_intent_iob:
        in_txt, intent, out_txt = line.split("\t")
        if intent not in all_lines:
            all_lines[intent] = list()
        
        all_lines[intent].append(in_txt+"\t"+out_txt)

    return all_lines                         


def get_utterances_sgd(path):  
    data_type = ["train", "dev", "test"]

    sen_intent_iob = []

    for train in data_type:
        for file_index in range(1, 128):
            f_name = '{:03d}'.format(file_index)
            file_path = path+train+"/dialogues_"+f_name+".json"
            
            if not os.path.exists(file_path):
                continue
                
            with open(file_path) as f:
                data = json.load(f)
                for item in data:
                    if "turns" in item:
                        turns = item["turns"]
                        #all_slots = {}

                        for turn in turns:

                            #utterance (user)
                            if "speaker" in turn and "utterance" in turn:
                                utterance = turn["utterance"].replace('\n', '').replace(r'\n', ' ').replace(r'\r', '').replace('  ', ' ')
                                iob = ["O"] * len(utterance.split())

                                if "frames" in turn:
                                    frames = turn["frames"]
                                    for frame in frames:
                                        #intent, slot_values
                                        if "slots" in frame and len(frame["slots"]) > 0 : # turn["speaker"] == "USER"  
                                            if "state" in frame and "active_intent" in frame["state"]:
                                                active_intent = frame["state"]["active_intent"]
                                            for slot in frame["slots"]:
                                                
                                                entity = slot["slot"]
                                                start = slot["start"]
                                                end = slot["exclusive_end"]
                                                value = utterance[start:end]
                                                start_index = get_starting_index_sgd(utterance, value, start)
                                                
                                                iob = update_iob_sgd(iob, start_index, entity, value)
                                                
                                            sentence = utterance+"\t"+active_intent+"\t"+" ".join(iob)
                                            sen_intent_iob.append(sentence)
    domains = get_domains(path)
    all_lines = {}

    for line in sen_intent_iob:
        in_txt, intent, out_txt = line.split("\t")
        if domains[intent] not in all_lines:
            all_lines[domains[intent]] = list()
        
        all_lines[domains[intent]].append(in_txt+"\t"+out_txt)

    return all_lines                                     
            

def write_lines(path, lines, limit = None):
    others, to_pop = [], []
    for k,v in lines.items():
        if len(v) < limit:
            others.extend(v)
            to_pop.append(k)
    
    if len(others) > 0:    
        lines["others"] = others

    for item in to_pop:
        lines.pop(item)

    

    for k,v in lines.items():
        if not os.path.exists(path+k):
            os.makedirs(path+k)
    
        f = open(path+k+"/"+k+".txt", "w")
        for item in v:
            f.write(item+"\n")
        f.close()

if __name__ == '__main__':
    args = get_parameter()
    input_path = args.input_path
    out_path = args.out_path
    dataset = args.dataset

    hack_map = {"1": "one", "2":"two", "3":"three", "4": "four", "5":"five",
           "6":"six", "7":"seven", "8":"eight", "9":"nine", "10": "ten",
           "11":"eleven", "12":"twelve", "13":"thirteen", "14" : "fourteen", "15": "fifteen",
           "20": "twenty"}

    if dataset == "sgd":
        all_sentences = get_utterances_sgd(input_path+dataset+"/")
        limit = 1850
    elif dataset == "multiwoz":
        all_sentences = get_utterances_multiwoz(input_path+dataset+"/", hack_map)
        limit = 650
    elif dataset == "snips" or dataset == "atis":
        all_sentences = get_utterances(input_path+dataset+"/")
        limit = 100
    else:
        print(dataset,"not supported yet")

    write_lines(out_path+dataset+"/", all_sentences, limit)
    print("results written to:", out_path+dataset)
    