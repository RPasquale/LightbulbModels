# LightbulbModels
Lightbulb Partners Models

# Language 1B
Objective: Next Token Prediction
Data: wikipedia
This small language model utilises a custom 'Expert' Method. The Language Expert utilises Sparse Flash2 Attention for processing the inputs, and feeds the attention distribution to the Switch Router. the Switch Router routes the sequence to a sub-expert:

Sub-Experts:
1) TransformerRAG: Influences the attention matrix with context embeddings retrieved through Dense Passage Retrieval.
2) TransformerDPO: Transformer Fine tuned using Direct Preference Optimisation // TODO: thinking of just making this a regular Transformer Model and runnign DPO on the whole Expert.
3) Mamba: As they say, if we can get the same or better result with linear complexity rather than quadratic, lets do it! Mamba is the selective state space model.

The output of the sub-expert is fed into a switch transformer decoder, which utilises Beam Search or TopK Sampling for improving the decoded output.

# Vision 1B
Objective: Multi Object Classification and Multi Object Bounding Box Regression
Data:
1) detection-datasets/coco
We utilise a DETR processor to preprocess the input images, and then feed these processed tensors into the RESNET for creating feature maps.

We route sequences of feature maps to sub experts:
1) Detection Transformer:Global understanding of Local Features from the resnet backbone used in the detection transformer.
2) Vision Transformer: The ViT is there to capture global relationships from the images.
3) Vision Mamba: Linear complexity

The output of the sub-expert gets fed into a Switch Transformer with LORY MoE Layers, and BEam Search or TopK Sampling Decoding of final object detections and bounding box regressions.

# MultiModal 1B
// TODO: MULTI MODAL COMING SOON

# Agent 1B
An Agent performs a certain task in a certain environment. Any of the Models can be used as an Agent Backbone. The Agent has sub-agents that are activated depending on the given task:
1) Deep Actor Critic
2) Deep Q Agent

Tasks could range from 'News Summariser' to 'Sentiment Analysis', or anything. Each of these tasks requires its own environment and own separate objective. The Agent, and sub agent architecture , whilst using the Model backbone, operates in a task specific environment with a unique objective. the Agent trains parameters to learn about achieving this new objective in the new environment. 

# MultiAgent 1B

Multiple Agents create a Multi-Agent System (MAS). A shared network is created over all the agents in the system, and the shared network utilises feedback from all the agents in the system to update an overall MAS policy or value function, that the individual agents utilise in their given environments on their tasks. 

You could have both agents: 'News Summariser' and 'Sentiment Analysis', and the MAS network has a shared objective of Writing Articles based on the summariser and sentiment agents. The individual agents will get feedback from the Article Writer MAS network which will make them interact with their individual environments differently, hopefully more optimally in trying to achieve both their individual objectives and their shared objective.
