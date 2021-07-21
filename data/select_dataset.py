

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    if dataset_type in ['cnflow']:
        from data.dataset_cnflow import DatasetCNFlow as D

    if dataset_type in ['sr', 'super-resolution']:
        from data.dataset_sr import DatasetSR as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
