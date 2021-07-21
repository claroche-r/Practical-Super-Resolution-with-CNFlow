
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'cnflow':
        from models.model_cnflow import CNFlowModel as M

    elif model == 'sr':
        from models.model_sr import ModelSR as M

    elif model == 'sr-gan':
        from models.model_sr_gan import ModelGAN as M

    elif model == 'kernel-pred':
        from models.model_kernel_pred import ModelPrediction as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
