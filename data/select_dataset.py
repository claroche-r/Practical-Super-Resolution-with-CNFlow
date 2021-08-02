def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    if dataset_type == 'cnflow':
        from data.dataset_cnflow import DatasetCNFlow as D

    elif dataset_type == 'sr' or dataset_type == 'super-resolution':
        from data.dataset_sr import DatasetSR as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
