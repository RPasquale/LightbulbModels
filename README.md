# LightbulbModels
Lightbulb Partners Models

# Language 1B
Objective: Next Token Prediction

Data: FineWeb_Edu: HuggingFaceFW/fineweb-edu

This small language model utilises a custom 'Expert' Method. The Language Expert utilises Sparse Flash2 Attention for processing the inputs, and feeds the attention distribution to the Switch Router. the Switch Router routes the sequence to a sub-expert:

Sub-Experts:
1) TransformerRAG: Influences the attention matrix with context embeddings retrieved through Dense Passage Retrieval.
2) TransformerDPO: Transformer Fine tuned using Direct Preference Optimisation // TODO: thinking of just making this a regular Transformer Model and runnign DPO on the whole Expert.
3) Mamba: As they say, if we can get the same or better result with linear complexity rather than quadratic, lets do it! Mamba is the selective state space model.

The output of the sub-expert is fed into a switch transformer decoder, which utilises Beam Search or TopK Sampling for improving the decoded output.

# Vision 1B
Objective: Multi Object Classification and Multi Object Bounding Box Regression
Data:
1) COCO: detection-datasets/coco

We utilise a DETR processor to preprocess the input images, and then feed these processed tensors into the RESNET for creating feature maps.

We route sequences of feature maps to sub experts:
1) Detection Transformer:Global understanding of Local Features from the resnet backbone used in the detection transformer.
2) Vision Transformer: The ViT is there to capture global relationships from the images.
3) Vision Mamba: Linear complexity

The output of the sub-expert gets fed into a Switch Transformer with LORY MoE Layers, and BEam Search or TopK Sampling Decoding of final object detections and bounding box regressions.

# MultiModal 1B
// TODO: MULTI MODAL COMING SOON
Papers this will be based on:
1) https://arxiv.org/pdf/2405.09818
Utilise a Transformer Architecture, "deviate from the Llama architecture by using query-key normalization (QK-Norm). QK-Norm directly controls the norm growth of input to the softmax by applying layer norm to the query and key vectors within the attention." (Chameleon Team, 2024).

h = x + attention_norm(attention(x))
output = h + ffn_norm(feed_forward(h))

we apply z-loss regularization. Specifically, we regularize the partition function Z of the Softmax Function.

The data is split into Text, Code, Visual Chat, Image Gen, Interleaved Text/Image Gen, Safety.

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
1) https://arxiv.org/pdf/2205.06175

An Agent performs a certain task in a certain environment. Any of the Models can be used as an Agent Backbone. The Agent has sub-agents that are activated depending on the given task:
1) Deep Actor Critic
2) Deep Q Agent

Tasks could range from 'News Summariser' to 'Sentiment Analysis', or anything. Each of these tasks requires its own environment and own separate objective. The Agent, and sub agent architecture , whilst using the Model backbone, operates in a task specific environment with a unique objective. the Agent trains parameters to learn about achieving this new objective in the new environment. 

# MultiAgent 1B

Multiple Agents create a Multi-Agent System (MAS). A shared network is created over all the agents in the system, and the shared network utilises feedback from all the agents in the system to update an overall MAS policy or value function, that the individual agents utilise in their given environments on their tasks. 

You could have both agents: 'News Summariser' and 'Sentiment Analysis', and the MAS network has a shared objective of Writing Articles based on the summariser and sentiment agents. The individual agents will get feedback from the Article Writer MAS network which will make them interact with their individual environments differently, hopefully more optimally in trying to achieve both their individual objectives and their shared objective.
