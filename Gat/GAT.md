
## <a name="_30uoxdq7qsdg"></a>**Graph Attention Network (GAT) Documentation**



In the dataset loading process, we use a masking technique to isolate nodes belonging to each individual graph




edge\_index: Shape (2, Edges)

node\_features: Shape (Nodes, 50)

node\_labels: Shape (Nodes, 121)


## <a name="_sswph8z6jwb3"></a>GAT Structure
import torch.nn as nn

from torch.optim import Adam


class GAT(torch.nn.Module):



`    `def \_\_init\_\_(self, num\_of\_layers, num\_heads\_per\_layer, num\_features\_per\_layer, add\_skip\_connection=True, bias=True,

`                 `dropout=0.6, log\_attention\_weights=False):

`        `super().\_\_init\_\_()

`        `assert num\_of\_layers == len(num\_heads\_per\_layer) == len(num\_features\_per\_layer) - 1, f'Enter valid  params.'

`        `num\_heads\_per\_layer = [1] + num\_heads\_per\_layer  #first layer of the GAT model, the input data typically does not have any hierarchical representations or higher-level features that require multiple attention heads to capture.

`        `gat\_layers = []  

`        `for i in range(num\_of\_layers):

`            `layer = GATLayer(

`                `num\_in\_features=num\_features\_per\_layer[i] \* num\_heads\_per\_layer[i],  

`                `num\_out\_features=num\_features\_per\_layer[i+1],

`                `num\_of\_heads=num\_heads\_per\_layer[i+1],

`                `concat=True if i < num\_of\_layers - 1 else False,  # last GAT layer does mean avg, the others do concat

`                `activation=nn.ELU() if i < num\_of\_layers - 1 else None,  # last layer just outputs raw scores

`                `dropout\_prob=dropout,

`                `add\_skip\_connection=add\_skip\_connection,

`                `bias=bias,

`                `log\_attention\_weights=log\_attention\_weights

`            `)

`            `gat\_layers.append(layer)

`        `self.gat\_net = nn.Sequential(

`            `\*gat\_layers,

`        `)



`    `def forward(self, data):

`        `return self.gat\_net(data)



The number of heads for the first layer is set to 1 because the input data usually doesn't require multiple attention heads.It iterates through the specified number of layers and creates GATLayer instances, appending them to a list.


ELU (Exponential Linear Unit) is chosen over ReLU (Rectified Linear Unit) in this because ELU provides smoothness, non-zero gradients for negative inputs, and robustness to negative values, which can help mitigate issues like the dying ReLU problem and vanishing gradients, particularly in deeper networks

### <a name="_t32kke2yncui"></a>**1. Linear Transformation**
The GAT model begins by linearly transforming node features to generate initial node embeddings. This step is essential for capturing relevant information from the input features and preparing them for the attention mechanism.
#### <a name="_twz6e3t349"></a>Implementation Details:
- Code Segments:
  - Initialization of the GAT class with parameters for linear transformation, such as the number of input and output features per layer.
  - The \_\_init\_\_ method initializes weight matrices (linear\_proj) using Xavier uniform initialization to ensure stable training.
  - Weight matrices are defined as learnable parameters within the model.
### <a name="_ma7h8pk3h22t"></a>**2. Attention Mechanism**
The attention mechanism computes attention scores between each node and its neighbors, allowing the model to focus on relevant information during aggregation. It enables nodes to attend to different neighbors with varying degrees of importance.
#### <a name="_o8twsptigxkn"></a>Implementation Details:
- Code Segments:
  - Computation of attention scores (scores\_source and scores\_target) using a shared attention mechanism based on node feature representations.
  - LeakyReLU activation function is applied to the attention scores to introduce non-linearity and handle negative values.
  - Scoring function parameters (scoring\_fun\_source and scoring\_fun\_target) are learned during training to adapt to the task.
### <a name="_dgdj5axydtwg"></a>**3. Attention Coefficients**
Attention coefficients are obtained by applying the softmax function to the computed attention scores. These coefficients determine the importance of each neighbor for a given node and are crucial for weighted aggregation of neighbor embeddings.
#### <a name="_s340w9achdna"></a>Implementation Details:
- Code Segments:
  - Calculation of attention coefficients (attentions\_per\_edge) using the softmax function in the neighborhood\_aware\_softmax method.
  - The softmax operation ensures that attention coefficients sum up to 1, making them interpretable as probabilities.
### <a name="_13oftacbt4xi"></a>**4. Aggregation**
Node embeddings are aggregated by weighted summation of neighbor embeddings using attention coefficients. This step combines information from neighboring nodes according to their importance weights, enhancing the representation of each node.
#### <a name="_d6gxcnmdtkeb"></a>Implementation Details:
- Code Segments:
  - Aggregation of node embeddings (out\_nodes\_features) by performing a weighted summation of neighbor embeddings in the aggregate\_neighbors method.
  - The aggregation process is weighted by attention coefficients, ensuring that more attention is given to relevant neighbors.
### <a name="_2tjt8ur31rb5"></a>**5. Training Function and Main Loop**
The training function orchestrates the training procedure, including data loading, model instantiation, loss computation, optimization, and logging of training/validation metrics. The main training loop iterates over epochs and batches, updating model parameters based on computed gradients.
#### <a name="_p87ut1d7fojy"></a>Implementation Details:
- Code Segments:
  - Implementation of the training procedure in the train\_gat function, handling data loading, model setup, optimization, and logging.
  - Definition of the main training loop in the get\_main\_loop function, responsible for iterating over training and validation phases, computing loss and metrics, and handling early stopping based on validation performance.
### <a name="_8pgrg1h3v6pa"></a>**6. Training Configuration**
The training configuration specifies hyperparameters and settings required for training the GAT model. It includes parameters such as the number of epochs, learning rate, batch size, and whether to enable logging or use specific datasets.
#### <a name="_ovh6yqi0viat"></a>Implementation Details:
- Code Segments:
  - Definition of training configuration parameters in the get\_training\_args function, allowing customization of hyperparameters and settings.
  - The training configuration encompasses aspects crucial for model training, such as optimization parameters, logging frequency, and dataset selection
#### <a name="_eusj3gflkkit"></a>**Accuracy**	(after 200 epochs)	
**Test micro-F1 = 0.9785256000421215**

The accuracy in PAPAer is GAT 0.973 ± 0.002

`   `gat\_config = {

`        `"num\_of\_layers": 3,  

`        `"num\_heads\_per\_layer": [4, 4, 6],  

`        `"num\_features\_per\_layer": [PPI\_NUM\_INPUT\_FEATURES, 64, 64, PPI\_NUM\_CLASSES], 

`        `"add\_skip\_connection": True, 

`        `"bias": True,  #

`        `"dropout": 0.0,  

`    `}




