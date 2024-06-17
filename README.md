# LightbulbModels
Lightbulb Partners Models

# Language 1B
Objective: Next Token Prediction

Data: FineWeb_Edu: HuggingFaceFW/fineweb-edu
Models:
1) GPT2: just an implementation of Karpathy's GPT2 from: https://youtu.be/l8pRSuU81PU?si=KvSqpJhMT5FCMY5e

2) multi_model:
This small language model utilises a custom 'Expert' Method. The Language Expert utilises Sparse Flash2 Attention for processing the inputs, and feeds the attention distribution to the Switch Router. the Switch Router routes the sequence to a sub-expert:

Sub-Experts:
1) TransformerRAG: Influences the attention matrix with context embeddings retrieved through Dense Passage Retrieval. (https://arxiv.org/pdf/2312.10997)
2) TransformerDPO: Transformer Fine tuned using Direct Preference Optimisation. (https://arxiv.org/pdf/2305.18290) // TODO: thinking of just making this a regular Transformer Model and runnign DPO on the whole Expert.
3) Mamba: As they say, if we can get the same or better result with linear complexity rather than quadratic, lets do it! Mamba is the selective state space model. (https://arxiv.org/pdf/2312.00752)

The output of the sub-expert is fed into a switch transformer decoder, which utilises Beam Search or TopK Sampling for improving the decoded output.

# Vision 1B
Objective: Multi Object Classification and Multi Object Bounding Box Regression
Data:
COCO: detection-datasets/coco
1) simple vision model:
Use a resnet for feature extraction, use those features and the input features to train a ViT for multi object classification and multi object bounding box detection.

2) multi_vision_model
We utilise a DETR processor to preprocess the input images, and then feed these processed tensors into the RESNET for creating feature maps.

We route sequences of feature maps to sub experts:
1) Detection Transformer:Global understanding of Local Features from the resnet backbone used in the detection transformer. (https://arxiv.org/pdf/2005.12872)
2) Vision Transformer: The ViT is there to capture global relationships from the images. (https://arxiv.org/pdf/2010.11929)
3) Vision Mamba: Linear complexity (https://arxiv.org/pdf/2401.09417)

The output of the sub-expert gets fed into a Switch Transformer with LORY MoE Layers, and BEam Search or TopK Sampling Decoding of final object detections and bounding box regressions.

# MultiModal 1B
Papers this will be based on:
1) https://arxiv.org/pdf/2405.09818
Utilise a Transformer Architecture, "deviate from the Llama architecture by using query-key normalization (QK-Norm). QK-Norm directly controls the norm growth of input to the softmax by applying layer norm to the query and key vectors within the attention." (Chameleon Team, 2024).

"h = x + attention_norm(attention(x))
output = h + ffn_norm(feed_forward(h))

we apply z-loss regularization. Specifically, we regularize the partition function Z of the Softmax Function.

The data is split into Text, Code, Visual Chat, Image Gen, Interleaved Text/Image Gen, Safety." (Chameleon)

Image Size 512x512

Need to utilise data balancing across modalities for high quality alignment.

2) https://arxiv.org/pdf/2405.17247


3) https://arxiv.org/pdf/2403.09611
MM1:
– Image Encoder: A ViT-L/14 [27] model trained with a CLIP loss [91] on
DFN-5B [31] and VeCap-300M [57]; images of size 336×336.
– Vision-Language Connector: C-Abstractor [12] with 144 image tokens.
– Pre-training Data: A mix of captioned images (45%), interleaved imagetext documents (45%), and text-only (10%) data.
– Language Model: A 1.2B transformer decoder-only language model.
To evaluate the different design decisions, we use zero-shot and few-shot (4-
and 8-shot) performance on a variety of captioning and VQA tasks: COCO Captioning [18], NoCaps [2], TextCaps [103], VQAv2 [38], TextVQA [104], VizWiz [39],
GQA [46], and OK-VQA [82].

# Agent 1B
papers:
1) GATO Agent: https://arxiv.org/pdf/2205.06175
2) World Models: https://arxiv.org/pdf/1803.10122
3) Reward free cirricula: https://arxiv.org/pdf/2306.09205v2
The idea here is to use the chameleon multi-modal model as the GATO agent backbone, and train a world model (2) that the gato agent (1) can interact with and engage in a 2 player zero sum minimax game with. The World Model is trying to maximise the regret while the agent is trying to minimize regret (3). Using this architecture, we can model the latent space, which can be useful for robustness of our agents.
# MultiAgent 1B

Multiple Agents create a Multi-Agent System (MAS). A shared network is created over all the agents in the system, and the shared network utilises feedback from all the agents in the system to update an overall MAS policy or value function, that the individual agents utilise in their given environments on their tasks. 

You could have both agents: 'News Summariser' and 'Sentiment Analysis', and the MAS network has a shared objective of Writing Articles based on the summariser and sentiment agents. The individual agents will get feedback from the Article Writer MAS network which will make them interact with their individual environments differently, hopefully more optimally in trying to achieve both their individual objectives and their shared objective.
