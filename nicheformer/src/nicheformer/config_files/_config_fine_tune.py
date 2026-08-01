sweep_config = {
    'checkpoint_path': "/media/rokny/DATA3/Sally/Nicheformer/nicheformer/nicheformer.ckpt",
    'freeze': False,
    'extract_layers': [11],  # Which layers to extract features from
    'function_layers': 'mean',  # Architecture of prediction head
    'reinit_layers': None,
    'extractor': False,
    'batch_size': 9,
    'lr': 1e-4,
    'warmup': 1,
    'max_epochs': 1,
    'pool': 'mean', # relevant when 'extractor': True
    'n_classes': 17, #for classification tasks
    'dim_prediction': 1, 
    'supervised_task': 'density_regression',
    'regress_distribution': False,
    'predict_density': True,
    'ignore_zeros': False, # for classification tasks
    'baseline': False,
    'organ': 'xenium_lung',
    'label': 'density_0',
    }
