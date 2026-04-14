import glob
import os
from typing import Optional

import torch.utils.data as data

from .dataloader import adobenpi
from .dataloader import realdata


class dataio(data.Dataset):
    def __init__(self, mode, args, conf, outdir):
        self.mode = mode

        if mode == 'Train':
            data_root = [args.training_dir]
            extension = conf.train_suffix
            self.numberOfImageBuffer = conf.train_maxNumberOfImages
            self.datatype = conf.train_datatype
            self.prefix = conf.train_prefix
            self.outdir = outdir
            self.max_objects = None
        elif mode == 'Test':
            data_root = [args.test_dir]
            extension = conf.test_suffix
            self.numberOfImageBuffer = conf.test_maxNumberOfImages
            self.datatype = conf.test_datatype
            self.prefix = conf.test_prefix
            self.outdir = outdir
            self.max_objects = getattr(args, 'max_test_objects', None)
        else:
            raise ValueError('mode must be one of [Train, Test]')

        self.data_name = []
        self.set_id = []
        self.valid = []
        self.sample_id = []
        self.dataCount = 0
        self.dataLength = -1
        self.loader_imgsize: Optional[tuple[int, int]] = None

        self.objlist = []
        for root in data_root:
            print(f'Initialize {root}')

            if not os.path.exists(root):
                raise FileNotFoundError(f'Data root does not exist: {root}')
            if not os.path.isdir(root):
                raise NotADirectoryError(f'Data root is not a directory: {root}')

            contents = os.listdir(root)
            print(f'Contents of {root}: {len(contents)} items')
            if contents:
                print(f'First few items: {contents[:min(5, len(contents))]}')

            search_pattern = root + '/**/*%s' % extension
            print(f'Searching recursively for directories with pattern: {search_pattern}')
            objlist = sorted(
                path for path in glob.glob(search_pattern, recursive=True)
                if os.path.isdir(path)
            )

            if mode == 'Test' and self.max_objects is not None and len(objlist) > self.max_objects:
                print(f'Limiting test objects from {len(objlist)} to {self.max_objects}')
                objlist = objlist[:self.max_objects]

            self.objlist.extend(objlist)

        print(f'Number of {mode} set is {len(self.objlist)}')
        if len(self.objlist) == 0:
            raise RuntimeError(
                f'No data found for {mode} mode. Expected directory extension: {extension}. '
                f'Searched recursively in: {data_root}'
            )

        if self.datatype == 'AdobeNPI':
            self.data = adobenpi.dataloader(self.numberOfImageBuffer)
        elif self.datatype == 'RealData':
            self.data = realdata.dataloader(self.numberOfImageBuffer, self.outdir)
        else:
            raise ValueError('datatype must be one of [AdobeNPI, RealData]')

    def __getitem__(self, index_):
        objid = index_
        if self.datatype == 'AdobeNPI':
            self.data.load(self.objlist, objid, prefix=self.prefix)
        elif self.datatype == 'RealData':
            self.data.load(self.objlist, objid, prefix=self.prefix, loader_imgsize=self.loader_imgsize[0])
        else:
            raise ValueError('datatype must be one of [AdobeNPI, RealData]')

        img = self.data.I.transpose(2, 0, 1, 3)
        nml = self.data.N.transpose(2, 0, 1)
        mask = self.data.mask.transpose(2, 0, 1)
        return img, nml, mask

    def __len__(self):
        return len(self.objlist)
