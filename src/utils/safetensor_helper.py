

def load_x_from_safetensor(checkpoint, key, skip_key1=None, skip_key2=None):
    x_generator = {}
    for k,v in checkpoint.items():
        if skip_key1 is not None and skip_key1 in k:
            continue
        if skip_key2 is not None and skip_key2 in k:
            continue

        if key in k:
            x_generator[k.replace(key+'.', '')] = v
    return x_generator