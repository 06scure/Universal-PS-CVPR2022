from torch.utils.tensorboard import SummaryWriter
import os, sys, glob
import swanlab


class logger():
    def __init__(self, args, mode, suffix = None):
        self.outdir = f'{args.outdir}/{args.session_name}'
        logdir = f'{self.outdir}/log'
        for file in glob.glob(f'{logdir}/*'):
            if os.path.isfile(file):
                os.remove(file)
        self.writer = SummaryWriter(log_dir=f'{logdir}', flush_secs=1)
        
        swanlab.init(
            project="Universal-PS",
            experiment_name=args.session_name,
            config=dict(vars(args))
        )


    def add(self, tag, item, step, itemtype):

        if itemtype == 'Image':
            if len(item.shape) == 3:
                self.writer.add_image(tag, item, global_step=step, dataformats='CHW')
                swanlab.log({tag: swanlab.Image(item)}, step=step)
            elif len(item.shape) == 4:
                self.writer.add_images(tag, item, global_step=step, dataformats='NCHW')
                swanlab.log({tag: swanlab.Image(item)}, step=step)
            else:
                raise Exception("item.shape must be 2 or 3 (%d)" % len(item.shape))
        elif itemtype == 'Scalar':
            self.writer.add_scalar(tag, item, global_step=step)
            swanlab.log({tag: item}, step=step)
        else:
            print('itemtype is not in "Image, Scalar"', file = sys.stderr)
