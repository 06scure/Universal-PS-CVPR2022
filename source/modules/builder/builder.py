import torch
import numpy as np
from tqdm import tqdm
from modules.model import model
import datetime
import cv2



class builder():
    def __init__(self, args, conf, device):
        self.img_channels = conf.img_channels
        self.device = device
        self.net = model.Net(args, device)
        if args.pretrained is not None:
            self.net.load_models(args.pretrained)
            print(f'Pretrained Model Loaded! initial LR is {self.net.print_lr()}')

    def run(self, mode, epoch = 0, writer = None, steps_per_test = 200,
            traindata = None, train_batch_size = 3, train_shuffle = True, train_loader_imgsize = None, train_encoder_imgsize = None, train_decoder_imgsize = None,
            testdata = None, test_batch_size = 3, test_shuffle = False, test_loader_imgsize = None, test_encoder_imgsize = None, test_decoder_imgsize = None):

        if mode == 'Train' or mode == 'TrainAndTest':
            self.net.set_mode('Train')
            traindata.loader_imgsize = train_loader_imgsize
            print(f'Train Batch Size is {train_batch_size}')
            train_data_loader = torch.utils.data.DataLoader(traindata, batch_size = train_batch_size, shuffle=train_shuffle, num_workers=4, pin_memory=True)

            # Only set up test data loader if test data is available
            test_data_loader = None
            if testdata is not None:
                testdata.loader_imgsize = test_loader_imgsize
                test_data_loader = torch.utils.data.DataLoader(testdata, batch_size = test_batch_size, shuffle=test_shuffle, num_workers=4, pin_memory=True)
            else:
                print("WARNING: No test data available. Skipping testing during training.")

            losses = 0
            cnt = 0
            for batch in tqdm(train_data_loader, leave=False):
                global_step = epoch * len(train_data_loader) + cnt

                """ test every steps_per_test """
                if np.mod(global_step, steps_per_test) == 0 and mode == 'TrainAndTest' and cnt > 0 and test_data_loader is not None:
                    self.net.set_mode('Test')
                    for batch_test in test_data_loader:
                        _, output, input = self.net.step(batch_test, decoder_imgsize=test_decoder_imgsize, encoder_imgsize=test_encoder_imgsize) # output = [B, 3, h, w]
                        # 保存输入和完整输出（左右拼接）
                        cv2.imwrite(f'{testdata.data.data_workspace}/input.png', 255 * input[0,:,:,:].transpose(1,2,0)[:,:,::-1])
                        cv2.imwrite(f'{testdata.data.data_workspace}/normal.png', 255 * output[0,:,:,:].transpose(1,2,0)[:,:,::-1])

                        # 单独保存右侧的高分辨率normal图（主要输出）
                        h, w = output.shape[2], output.shape[3]
                        output_high = output[0,:,:,w//2:].transpose(1,2,0)

                        # 使用原始mask裁剪背景（从testdata.data中获取）
                        if hasattr(testdata.data, 'mask'):
                            mask_img = testdata.data.mask

                            # 处理mask的维度（可能是(h,w)或(h,w,1)或(1,h,w)）
                            if len(mask_img.shape) == 3:
                                if mask_img.shape[0] == 1:
                                    mask_img = mask_img[0]  # (1, h, w)
                                elif mask_img.shape[2] == 1:
                                    mask_img = mask_img[:, :, 0]  # (h, w, 1)

                            # 确保mask尺寸与输出一致
                            mask_high = cv2.resize(mask_img, (output_high.shape[1], output_high.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST)

                            output_high = output_high * mask_high[:, :, np.newaxis]

                        cv2.imwrite(f'{testdata.data.data_workspace}/normal_highres.png', 255 * output_high[:,:,::-1])
                    self.net.set_mode('Train')
                    savedir = writer.outdir + '/checkpoint/current'
                    self.net.save_models(savedir)

                # TRAIN STEP
                loss, output, input  = self.net.step(batch, decoder_imgsize=train_decoder_imgsize, encoder_imgsize=train_encoder_imgsize) # output = [B, 3, h, w]
                losses += loss
                cnt += 1

            writer.add('Train Loss', losses/cnt, epoch, 'Scalar')
            writer.add('Learning Rate', self.net.print_lr(), epoch, 'Scalar')
            savedir = writer.outdir + '/checkpoint/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.net.save_models(savedir)
            self.net.scheduler_step()
            return losses/cnt

        if mode == 'Test':
            cnt = 0
            global_step = epoch
            self.net.set_mode('Test')
            testdata.loader_imgsize = test_loader_imgsize
            test_data_loader = torch.utils.data.DataLoader(testdata, batch_size = test_batch_size, shuffle=test_shuffle, num_workers=0, pin_memory=True)
            for i, batch in enumerate(test_data_loader):
                global_step = epoch * len(test_data_loader) + cnt
                mask = batch[2]  # 获取原始mask (B, 1, H, W)
                _, output, input = self.net.step(batch, decoder_imgsize=test_decoder_imgsize, encoder_imgsize=test_encoder_imgsize) # output = [B, 3, h, w]

                # 保存输入和完整输出（左右拼接）
                cv2.imwrite(f'{testdata.data.data_workspace}/input.png', 255 * input[0,:,:,:].transpose(1,2,0)[:,:,::-1])
                cv2.imwrite(f'{testdata.data.data_workspace}/normal.png', 255 * output[0,:,:,:].transpose(1,2,0)[:,:,::-1])

                # 单独保存右侧的高分辨率normal图（主要输出）
                h, w = output.shape[2], output.shape[3]
                output_high = output[0,:,:,w//2:].transpose(1,2,0)

                # 使用原始mask裁剪背景（从testdata.data中获取）
                if hasattr(testdata.data, 'mask'):
                    mask_img = testdata.data.mask

                    # 处理mask的维度（可能是(h,w)或(h,w,1)或(1,h,w)）
                    if len(mask_img.shape) == 3:
                        if mask_img.shape[0] == 1:
                            mask_img = mask_img[0]  # (1, h, w)
                        elif mask_img.shape[2] == 1:
                            mask_img = mask_img[:, :, 0]  # (h, w, 1)

                    # 确保mask尺寸与输出一致
                    mask_high = cv2.resize(mask_img, (output_high.shape[1], output_high.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                    output_high = output_high * mask_high[:, :, np.newaxis]

                cv2.imwrite(f'{testdata.data.data_workspace}/normal_highres.png', 255 * output_high[:,:,::-1])
                cnt +=1
