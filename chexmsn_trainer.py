import os
import torch
import argparse
import pytorch_lightning as pl




from src.models.losses import MSNLoss

from torch.utils.data import DataLoader
from src.data.datsets import ChexMSNDataset
from pytorch_lightning.loggers import  CSVLogger
from src.models.chexmsn import MSN1
from src.data.utils import MSNTransform


from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
torch.set_float32_matmul_precision('high')

# pl.seed_everything(24)

# csv_logger = CSVLogger(save_dir=os.path.join('./','models'),
#                        version = '-'.join([os.getenv('SLURM_JOBID'), os.getenv('RUN_NAME')]),
#                        flush_logs_every_n_steps=1)

# ckpt_save_dir = os.path.join('./','models','lightning_logs',
#                              '-'.join([os.getenv('SLURM_JOBID'), os.getenv('RUN_NAME')]))


parser = argparse.ArgumentParser(description='SSL training command line interface')



# backbone options
parser.add_argument('--image-size', '--is',type=int, default=224)
parser.add_argument('--patch-size', '--ps',type=int, default=16)
parser.add_argument('--num-layers', '--nl',type=int, default=12)
parser.add_argument('--num-heads', '--nh',type=int, default=6)
parser.add_argument('--hidden-dim', '--hd',type=int, default=192)
parser.add_argument('--mlp-dim', '--md',type=int, default=192*4)
parser.add_argument('--embed-dropout', '--ed',type=float, default=0.0)
parser.add_argument('--attent-dropout', '--ad',type=float, default=0.0)
#parser.add_argument('--num-cls', '--cls',type=int, default=2)

# projection head options
parser.add_argument('--projection-in', '--pi',type=int, default=192)
parser.add_argument('--projection-hidden', '--ph',type=int, default=192*4)
parser.add_argument('--projection-out', '--po',type=int, default=19)

# chexmsn options
parser.add_argument('--mask-ratio','--mr', type=float, default=0.15)
parser.add_argument('--exponent-average', '--ema', type=float, default=0.996)
parser.add_argument('--focal-views', '--fv', type=bool, default=True)

#chexmsn loss options
parser.add_argument('--temprature-ratio','--tr', type=float, default=0.1)
parser.add_argument('--sinkhorn-iterations', '--si', type=int, default=3)
# parser.add_argument('--sim-weight','--sw', type=float, default=sw)
# parser.add_argument('--age-weight','--aw', type=float, default=aw)
#parser.add_argument('--gender-weight','--gw', type=float, default=1.0)
parser.add_argument('--reg-weight','--rw', type=float, default=1)


# dataloader options
parser.add_argument('--data-dir', '-dd',type=str, default=os.path.join('./','data','meta.csv'))
parser.add_argument('--same-age', '--sa', type=bool, default=True)
parser.add_argument('-b','--batch-size', type=int, default=64)
parser.add_argument('-w', '--num-workers', type=int, default=24)
parser.add_argument('--pin-memory', '--pm', type=bool, default=True)


#model options
parser.add_argument('--num-prototypes', '--np', type=int, default=1024)
parser.add_argument('--learning-rate','--lr', type=float, default=0.0001)
parser.add_argument('--weight-decay','--wd', type=float, default=0.001)
parser.add_argument('--max-epochs','--me', type=int, default=100)
parser.add_argument('--mixed-precision', '--mp', type=str, default='16-mixed')


# callbacks options
parser.add_argument('--monitor-quantity','--mq', type=str, default='train_loss')
parser.add_argument('--monitor-mode','--mm', type=str, default='min')
parser.add_argument('--es-delta', '--esd', type=float, default=0.00000000001)
parser.add_argument('--es-patience','--esp', type=int, default=15)



args = parser.parse_args()



transforms = MSNTransform()
dataset = ChexMSNDataset(data_dir=os.getenv('DATA_DIR'),
                         transforms= transforms)

dataloader = DataLoader(dataset=dataset,
                              batch_size=64,
                              num_workers=24,
                              pin_memory=True,
                              shuffle=True
                              )

model = MSN1()

checkpoint_callback = ModelCheckpoint(monitor=args.monitor_quantity, 
                                      mode=args.monitor_mode,
                                      every_n_epochs=1,
                                      save_top_k=1,
                                      )

early_stop = EarlyStopping(monitor=args.monitor_quantity, 
                           min_delta=args.es_delta,
                           mode=args.monitor_mode, 
                           patience=args.es_patience)




trainer = pl.Trainer(accelerator='auto', 
                     devices='auto',
                     strategy='auto',
                     #logger=csv_logger, 
                     log_every_n_steps=1,
                     max_epochs=100,
                     precision='16-mixed', 
                     callbacks=[checkpoint_callback],
                     default_root_dir='./models/'
                     )



trainer.fit(model=model, train_dataloaders=dataloader)