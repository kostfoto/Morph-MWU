import torch.optim as optim

def make_optimizer(params, config):
    name = config['name']
    if name == 'adam':
        return optim.Adam(
            params, 
            lr=config['lr'], 
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif name == 'adamw':
        return optim.AdamW(
            params, 
            lr=config['lr'], 
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif name == "sgd":
        return optim.SGD(
            params,
            lr=config["lr"],
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 0.0),
        )

    raise ValueError(f"Unknown optimizer: {name}")