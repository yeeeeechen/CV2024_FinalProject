# Experiment Settings
exp_name   = 'doorDetection' # name of experiment

model_type = 'fastrcnn'

# Learning Options
epochs     = 4          # train how many epochs
batch_size = 1            # batch size for dataloader 
# use_adam   = False        # Adam or SGD optimizer
lr         = 1e-2         # learning rate
milestones = [16, 32, 45] # reduce learning rate at 'milestones' epochs
