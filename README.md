# CKBC_Model

### Dataset
The ConceptNet datasets are stored in the data folder, and the training, test, and validation sets are train.txt, test.txt, and dev.txt, respectively. the fine-tuned trained BERT model weights from the paper are stored in the [link](https://pan.baidu.com/s/19hYHzU3J336DHCdlvZ8QUQ)(password: bs45), and the folder in which the link is downloaded should be placed in the ConceptNet folder.

### Training

**Parameters:**

`--epochs_gat`: Number of epochs for gat training.

`--epochs_conv`: Number of epochs for convolution training.

`--lr`: Initial learning rate.

`--weight_decay_gat`: L2 reglarization for gat.

`--weight_decay_conv`: L2 reglarization for conv.

`--get_2hop`: Get a pickle object of 2 hop neighbors.

`--use_2hop`: Use 2 hop neighbors for training.  

`--partial_2hop`: Use only 1 2-hop neighbor per node for training.

`--output_folder`: Path of output folder for saving models.

`--batch_size_gat`: Batch size for gat model.

`--valid_invalid_ratio_gat`: Ratio of valid to invalid triples for GAT training.

`--drop_gat`: Dropout probability for attention layer.

`--alpha`: LeakyRelu alphas for attention layer.

`--nhead_GAT`: Number of heads for multihead attention.

`--margin`: Margin used in hinge loss.

`--batch_size_conv`: Batch size for convolution model.

`--alpha_conv`: LeakyRelu alphas for conv layer.

`--valid_invalid_ratio_conv`: Ratio of valid to invalid triples for conv training.

`--out_channels`: Number of output channels in conv layer.

`--drop_conv`: Dropout probability for conv layer.


The specific value settings for all parameters are included in the code

### Reproducing results

To reproduce the results published in the paper:      

        $ python code/SIM_BERT_RGAT_ConvKB.py
        
