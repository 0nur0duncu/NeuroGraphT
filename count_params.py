import torch
from models.epigraphnet import EpiGraphNet
from models.baselines import BaselineCNNTransformer, Baseline1DCNNTransformer

model = EpiGraphNet(
    in_channels=1,
    conv_channels=[32, 64, 128],
    transformer_dim=128,
    num_heads=8,
    transformer_layers=4,
    d_ff=512,
    num_nodes=16,
    sparsity=25.0,
    thresholding='value',
    gcn_hidden=128,
    gcn_layers=3,
    num_classes=5,
    dropout=0.1
)

print('\n--- CNN Encoder ---')
cnn_total = 0
for name, param in model.cnn_transformer.cnn.named_parameters():
    print(f'  {name}: {param.numel():,} | shape={list(param.shape)}')
    cnn_total += param.numel()
print(f'  CNN Toplam: {cnn_total:,}')

print('\n--- Input Projection ---')
proj_total = 0
for name, param in model.cnn_transformer.input_projection.named_parameters():
    print(f'  {name}: {param.numel():,} | shape={list(param.shape)}')
    proj_total += param.numel()
print(f'  Projection Toplam: {proj_total:,}')

print('\n--- Transformer Encoder ---')
trans_total = 0
for name, param in model.cnn_transformer.transformer.named_parameters():
    print(f'  {name}: {param.numel():,} | shape={list(param.shape)}')
    trans_total += param.numel()
print(f'  Transformer Toplam: {trans_total:,}')

print('\n--- GCN Encoder ---')
gcn_total = 0
for name, param in model.gcn.named_parameters():
    print(f'  {name}: {param.numel():,} | shape={list(param.shape)}')
    gcn_total += param.numel()
print(f'  GCN Toplam: {gcn_total:,}')

print('\n--- Classifier ---')
cls_total = 0
for name, param in model.classifier.named_parameters():
    print(f'  {name}: {param.numel():,} | shape={list(param.shape)}')
    cls_total += param.numel()
print(f'  Classifier Toplam: {cls_total:,}')

print('\n' + '='*70)
grand_total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'CNN Encoder:         {cnn_total:>10,} ({cnn_total/grand_total*100:5.1f}%)')
print(f'Input Projection:    {proj_total:>10,} ({proj_total/grand_total*100:5.1f}%)')
print(f'Transformer Encoder: {trans_total:>10,} ({trans_total/grand_total*100:5.1f}%)')
print(f'GCN Encoder:         {gcn_total:>10,} ({gcn_total/grand_total*100:5.1f}%)')
print(f'Classifier:          {cls_total:>10,} ({cls_total/grand_total*100:5.1f}%)')
print('-'*70)
print(f'TOPLAM PARAMETRE:    {grand_total:>10,}')
print(f'EĞİTİLEBİLİR:        {trainable:>10,}')
print(f'Yaklaşık:            {grand_total/1e6:.4f}M ({grand_total/1e6:.2f} milyon)')
print('='*70)
