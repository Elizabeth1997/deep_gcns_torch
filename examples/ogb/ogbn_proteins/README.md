# ogbn-proteins

We simply apply a random partition to generate batches for both mini-batch training and test. We set the number of partitions to be 10 for training and 5 for test, and we set the batch size to 1 subgraph.  We initialize the features of nodes through aggregating the features of their connected edges by a Sum (Add) aggregation. We also provide an option to ultilize the species information provided in the dataset by converting them into 8-dim one-hot-encoding which is exactly the same with the node features provided in the dataset before the [first release of OGB](https://github.com/snap-stanford/ogb/releases/tag/1.1.0). In short, in all our experiments, the input node features of the first layer of graph convolution layer is obtained by `MLP(MLP(extracted node features);MLP(one-hot-encoding))` where ; represents concatenation operation.
## Default 
	--use_gpu False 
    --cluster_number 10 
    --valid_cluster_number 5 
    --aggr add 	#options: [mean, max, add]
    --use_one_hot_encoding False
    --block plain 	#options: [plain, res, res+]
    --conv gen
    --gcn_aggr max 	#options: [max, mean, add, softmax_sg, softmax, power]
    --num_layers 3
    --conv_encode_edge False
	--mlp_layers 2
    --norm layer
    --hidden_channels 64
    --epochs 1000
    --lr 0.01
    --dropout 0.0
    --num_evals 1

## DyResGEN-112

### Train the model that performs best
	python main.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 112 --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.1 
### Test (use pre-trained model, [download](https://drive.google.com/file/d/1LjsgXZo02WgzpIJe-SQHrbrwEuQl8VQk/view?usp=sharing) from Google Drive)
	python test.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 112 --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.1
### Test by multiple evaluations (e.g. 5 times)

    python test.py --use_gpu --use_one_hot_encoding --num_evals 5 --conv_encode_edge --num_layers 112 --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.1 
    
## Train ResGCN-112
	python main.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 112 --block res --gcn_aggr max

#### Train with different GCN models with 28 layers on GPU 

SoftMax aggregator with learnable t (initialized as 1.0)

    python main.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr softmax --t 1.0 --learn_t

PowerMean aggregator with learnable p (initialized as 1.0)

    python main.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr power --p 1.0 --learn_p

Apply MsgNorm (message normalization) layer (e.g. SoftMax aggregator with fixed t (e.g. 0.1))

**Not learn parameter s (message scale)**

    python main.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr softmax_sg --t 0.1 --msg_norm
**Learn parameter s (message scale)**

    python main.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr softmax_sg --t 0.1 --msg_norm --learn_msg_scale
    
## ResGEN
SoftMax aggregator with fixed t (e.g. 0.001)

    python main.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr softmax_sg --t 0.001
    
PowerMean aggregator with fixed p (e.g. 5.0)
  
    python main.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr power --p 5.0
## ResGCN+
	python main.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 28 --block res+ --gcn_aggr mean
## ResGCN 
	python main.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 28 --block res --gcn_aggr mean
## PlainGCN 
	python main.py --use_gpu --use_one_hot_encoding --conv_encode_edge --num_layers 28 --gcn_aggr mean



    
