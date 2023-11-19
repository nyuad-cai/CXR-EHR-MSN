import os
import glob
import argparse
import pytorch_lightning as pl



from src.models.visiontransformer import VisionTransformer
from src.models.chexmsn import ProjectionHead, ChexMSN, ChexMSNModel 
from src.models.losses import MSNLoss
from src.models.utils import ConvStemConfig
from torch.utils.data import DataLoader
from src.data.datsets import ChexMSNDataset
from pytorch_lightning.loggers import  CSVLogger
from src.models.chexmsn import ChexMSN
from src.data.utils import MSNTransform


from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


pl.seed_everything(24)

parser = argparse.ArgumentParser(description='SSL training command line interface')

csv_logger = CSVLogger(save_dir=os.path.join('./','models'),
                       version = '-'.join([os.getenv('SLURM_JOBID'), os.getenv('RUN_NAME')]),
                       flush_logs_every_n_steps=1)

ckpt_save_dir = os.path.join('./','models',
                             '-'.join([os.getenv('SLURM_JOBID'), os.getenv('RUN_NAME')]))



# backbone options
parser.add_argument('--image-size', '--is',type=int, default=224)
parser.add_argument('--patch-size', '--ps',type=int, default=16)
parser.add_argument('--num-layers', '--nl',type=int, default=12)
parser.add_argument('--num-heads', '--nh',type=int, default=3)
parser.add_argument('--hidden-dim', '--hd',type=int, default=192)
parser.add_argument('--mlp-dim', '--md',type=int, default=792)
parser.add_argument('--embed-dropout', '--ed',type=float, default=0.0)
parser.add_argument('--attent-dropout', '--ad',type=float, default=0.0)
parser.add_argument('--num-cls', '--cls',type=int, default=3)

# projection head options
parser.add_argument('--projection-in', '--pi',type=int, default=192)
parser.add_argument('--projection-hidden', '--ph',type=int, default=768)
parser.add_argument('--projection-out', '--po',type=int, default=192)

# chexmsn options
parser.add_argument('--mask-ratio','--mr', type=float, default=0.15)
parser.add_argument('--exponet-average', '--ema', type=float, default=0.996)
parser.add_argument('--focal-views', '--fv', type=bool, default=True)

#chexmsn loss options
parser.add_argument('--temprature-ratio','--tr', type=float, default=0.1)
parser.add_argument('--sinkhorn-iterations', '--si', type=int, default=3)
parser.add_argument('--sim-weight','--sw', type=float, default=1.0)
parser.add_argument('--age-weight','--aw', type=float, default=1.0)
parser.add_argument('--gender-weight','--gw', type=float, default=1.0)
parser.add_argument('--reg-weight','--rw', type=float, default=1.0)


# dataloader options
parser.add_argument('--data-dir', '-dd',type=str, default=os.path.join('./','data','meta.csv'))
parser.add_argument('--same-age', '--sa', type=bool, default=True)
parser.add_argument('-b','--batch-size', type=int, default=64)
parser.add_argument('-w', '--num-workers', type=int, default=8)
parser.add_argument('--pin-memory', '--pm', type=bool, default=True)


#model options
parser.add_argument('--num-prototypes', '--np', type=int, default=3)
parser.add_argument('--learning-rate','--lr', type=float, default=0.001)
parser.add_argument('--weight-decay','--wd', type=float, default=0.0001)
parser.add_argument('--max-epochs','--me', type=int, default=100)
parser.add_argument('--mixed-precision', '--mp', type=int, default=16)


# callbacks options
parser.add_argument('--monitor-quantity','--mq', type=str, default='train_loss')
parser.add_argument('--monitor-mode','--mm', type=str, default='min')
parser.add_argument('--es-delta', '--esd', type=float, default=0.001)
parser.add_argument('--es-patience','--esp', type=int, default=5)



args = parser.parse_args()


stemconfig = [ConvStemConfig(out_channels= 64, kernel_size = 3 , stride = 2) for i in range(4)]

backbone = VisionTransformer(image_size=args.image_size,
                             patch_size=args.patch_size,
                             num_layers=args.num_layers,
                             num_heads=args.num_heads,
                             hidden_dim=args.hidden_dim,
                             mlp_dim=args.mlp_dim,
                             dropout=args.embed_dropout,
                             attention_dropout=args.attent_dropout,
                             num_cls_tokens=args.num_cls,
                             conv_stem_configs=stemconfig)

projection_head = ProjectionHead(in_features=args.projection_in,
                                 hidden_features=args.projection_hidden,
                                 out_features=args.projection_out)


chexmsn = ChexMSN(backbone=backbone,
                  projection_head=projection_head,
                  masking_ratio=args.mask_ratio,
                  ema_p=args.exponent_average,
                  focal=args.focal_views)


criterion = MSNLoss(temperature=args.temprature_ratio,
                    sinkhorn_iterations=args.sinkhorn_iterations,
                    similarity_weight=args.sim_weight,
                    age_weight=args.age_weight,
                    gender_weight=args.gender_weight,
                    regularization_weight=args.reg_weight)

transforms = MSNTransform()
dataset = ChexMSNDataset(data_dir=args.data_dir,
                         transforms= transforms,
                         same=args.same_age)

dataloader = DataLoader(dataset=dataset,
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       pin_memory=args.pin_memory)

model = ChexMSNModel(model=chexmsn,
                     criterion=criterion,
                     num_prototypes=args.num_prototypes,
                     learning_rate=args.learning_rate,
                     weight_decay=args.weight_decay,
                     max_epochs=args.max_epochs)

checkpoint_callback = ModelCheckpoint(monitor=args.monitor_quantity, 
                                      mode=args.monitor_mode,
                                      every_n_epochs=1,
                                      dirpath=ckpt_save_dir,
                                      save_top_k=-1)

early_stop = EarlyStopping(monitor=args.monitor_quantity, 
                           min_delta=args.es_delta,
                           mode=args.monitor_mode, 
                           patience=args.es_patience)

lr_logger = LearningRateMonitor(logging_interval='epoch')



trainer = pl.Trainer(accelerator='auto', 
                     devices='auto',
                     strategy='auto',
                     logger=csv_logger, 
                     max_epochs=args.max_epochs,
                     precision=args.mixed_precision, 
                     callbacks=[checkpoint_callback, lr_logger, early_stop],
                     )



trainer.fit(model=model, train_dataloaders=dataloader)