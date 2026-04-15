import glob
import os,sys
from typing import Optional
import torch.utils.data as data
from .dataloader import adobenpi
from .dataloader import realdata
from .utils import *


class dataio(data.Dataset):
    def __init__(self, mode, args, conf, outdir):

        self.mode = mode  # 保存模式信息

        if mode == 'Train':
            data_root = [args.training_dir]
            extension = conf.train_suffix
            self.numberOfImageBuffer = conf.train_maxNumberOfImages
            self.datatype = conf.train_datatype
            self.prefix= conf.train_prefix
            self.outdir = outdir

        elif mode == 'Test':
            data_root = [args.test_dir]
            extension = conf.test_suffix
            self.numberOfImageBuffer = conf.test_maxNumberOfImages
            self.datatype = conf.test_datatype
            self.prefix= conf.test_prefix
            self.outdir = outdir
        else:
            print("mode must be from [Train, Test]", file=sys.stderr)
            sys.exit(1)

        self.data_name = []
        self.set_id = []
        self.valid = []
        self.sample_id = []
        self.dataCount = 0
        self.dataLength = -1
        self.mode = mode
        self.loader_imgsize:Optional[tuple[int,int]] = conf.image_size, conf.image_size


        self.objlist = []
        for i in range(len(data_root)):
            print('Initialize %s' % (data_root[i]))

            # Check if data root exists
            if not os.path.exists(data_root[i]):
                print(f"ERROR: Data root does not exist: {data_root[i]}", file=sys.stderr)
                print(f"Please check your --{mode.lower()}_dir parameter!", file=sys.stderr)
                sys.exit(1)

            if not os.path.isdir(data_root[i]):
                print(f"ERROR: Data root is not a directory: {data_root[i]}", file=sys.stderr)
                sys.exit(1)

            # List contents of the directory for debugging
            contents = os.listdir(data_root[i])
            print(f"Contents of {data_root[i]}: {len(contents)} items")
            if len(contents) > 0:
                print(f"First few items: {contents[:min(5, len(contents))]}")

            # Search recursively for directories with the specified extension
            search_pattern = data_root[i] + '/**/*%s' % extension
            print(f"Searching recursively for directories with pattern: {search_pattern}")

            objlist = []
            for p in glob.glob(search_pattern, recursive=True):
                if os.path.isdir(p):
                    objlist.append(p)

            objlist = sorted(objlist)

            # 限制推理数量（仅在Test模式下生效）
            if mode == 'Test' and hasattr(args, 'min_nimg') and args.min_nimg is not None:
                if len(objlist) > args.min_nimg:
                    print(f"Limiting test objects from {len(objlist)} to {args.min_nimg}")
                    objlist = objlist[:args.min_nimg]

            self.objlist = self.objlist + objlist

        print(f"Number of {mode} set is {len(self.objlist)}")

        # Check if we have any data
        if len(self.objlist) == 0:
            print(f"ERROR: No data found for {mode} mode!", file=sys.stderr)
            print(f"Please check your data directory structure and extension settings.", file=sys.stderr)
            print(f"Expected directory extension: {extension}", file=sys.stderr)
            print(f"Searched recursively in: {data_root}", file=sys.stderr)
            sys.exit(1)


        if self.datatype == 'AdobeNPI':
            self.data = adobenpi.dataloader(self.numberOfImageBuffer)
        elif self.datatype == 'RealData':
            self.data = realdata.dataloader(self.numberOfImageBuffer, self.outdir)
        else:
            raise Exception(' "datatype" != in "Cycles, Adobe, DiLiGenT"')


    def __getitem__(self, index_):
        """
        Args:
            index_ (int): 数据索引
        Returns:
            img (torch.Tensor): 图像数据，形状为 (C, H, W, N)
            nml (torch.Tensor): 法线数据，形状为 (3, H, W)
            mask (torch.Tensor): 前景掩码，形状为 (1, H, W)
        """
        objid = index_
        if self.datatype == 'AdobeNPI':
            self.data.load(self.objlist, objid, prefix = self.prefix)
        elif self.datatype == 'RealData':
            self.data.load(self.objlist, objid, prefix = self.prefix, loader_imgsize = self.loader_imgsize[0])

        else:
            raise Exception(' "datatype" != in "Cycles, Adobe, DiLiGenT"')

        img = self.data.I.transpose(2,0,1,3) # c, h, w, N
        nml = self.data.N.transpose(2,0,1) # 3, h, w
        mask = self.data.mask.transpose(2,0,1) # 1, h, w
        return img, nml, mask


    def __len__(self):
        return len(self.objlist)
