# Transformer architecture explained

## Traditional Architectures for Sequence Modelling
RNNs and LSTMs were the commonly and famously used architectures for the purposes of sequence modelling. However they posed a major challenge of efficency due to their sequential processing. While LSTM and RNNs were a gamechanger in the field of sequential modelling, they were inefficent in terms of utilizing the full processing power of a system. Specifically, parallelism through LSTMs or RNNs could not be achieved. These architectures processed one element at each timestep.
The transformer architecture explored the idea of achieving parallelism by processing the sequence as a function of the entire sentence as a whole.

## The transformer architecture

<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" width="400" height="550" />

### Encoder and Decoder Stacks
- Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. A residual connection around each of the two sub-layers is employed, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.

- Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder,residual connections around each of the sub-layers is employed, followed by layer normalization. The self-attention is also modified for the sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

### Multi-Head Attention and Self-Attention
- Self Attention:
<br> 
<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*47UCxMjpfJ2yo48fctNv-g.png" width="400" height="550" />
<br>
    - Self-attention in the Encoder — the input sequence pays attention to itself
    - Self-attention in the Decoder — the target sequence pays attention to itself
    - Encoder-Decoder-attention in the Decoder — the target sequence pays attention to the input sequence

    The input sequence is converted to an encoded representation that encodes the semantics of the sequence as well as the positional information using the input embeddings and postional embeddings. An encoded representation for each word is then fed to the self-attention sub-layer which then calculates the attention score for each word by taking the dot product of the vectors "Query, key and values". The idea of "Q, K and V" vectors can be mapped to the idea of searching a video on youtube. The Query is the text that we search on youtube, key is the titles and description of videos, and values are the videos themselves. 
    These vectors are linear projections of the input embedding. 
    Consider the following sequence:
    `The cat sat on the mat`

This is converted to an input embedding:
```
Example: [Embedding for 'The', Embedding for 'cat', Embedding for 'sat', 
Embedding for 'on', Embedding for 'the', Embedding for 'mat', 
Embedding for '.']
```

The Query, key and value vectors are generated:
```
Queries: [Query for 'The', Query for 'cat', Query for 'sat', 
          Query for 'on', Query for 'the', Query for 'mat', Query for '.'] 

Keys: [Key for 'The', Key for 'cat', Key for 'sat', Key for 'on',
       Key for 'the', Key for 'mat', Key for '.'] 

Values: [Value for 'The', Value for 'cat', Value for 'sat', 
         Value for 'on', Value for 'the', Value for 'mat', Value for '.']

Random data:

Queries: [[0.23,0.4,.67,....],[0.4,0.6,.67,....],[0.2,0.2,.67,....], 
          [0.5,0.3,.8,....], [0.1,0.4,.67,....], [0.2,0.4,.67,....], 
          [0.7,0.4,.6,....]] 

Keys: [[0.1,0.4,.5,....],[0.2,0.4,.67,....],[0.3,0.4,.67,....], 
          [0.4,0.4,.67,....], [0.5,0.4,.67,....], [0.6,0.7,.8,....], 
          [0.6,0.4,.8,....]]

Values: [[0.4,0.5,.67,....],[0.23,0.4,.5,....],[0.23,0.4,.8,....], 
          [0.23,0.4,.45,....], [0.23,0.4,.9,....], [0.23,0.4,.6,....], 
          [0.23,0.4,.10,....]]
```
    
Attention scores are then calculated by taking the dot product of these vectors. These scores represent the importance or relevance of each word to the current word being processed. 

```
Randoms Scores
Example: Attention scores for 'The': [0.9,0.7,0.5, 0.4,0.45,0.56,0.23]
Attention scores for 'cat': [0.6,0.5,0.7, 0.23,0.44,0.58,0.23]
.....
Attention scores for '.': [0.3,0.5,0.9, 0.4,0.45,0.56,0.23]
```

Self-attention describes the relevance of each word in a sequence to other words in the same sequence. 

- Multi-Attention Head: In the Transformer, the Attention module repeats its computations multiple times in parallel. Each of these is called an Attention Head. The Attention module splits its Query, Key, and Value parameters N-ways and passes each split independently through a separate Head. All of these similar Attention calculations are then combined together to produce a final Attention score. This is called Multi-head attention and gives the Transformer greater power to encode multiple relationships and nuances for each word.
 


### Understanding the transformer architecture visually

Let's understand how the transformer architecture works for a translation problem:
Consider an example input sequence:
`You are welcome`
and the target sequence:
`De Nada` which is the spanish translation of the input sequence

Let's see how the input flows through the architecture of transformer:
*An interesting point to note is that the training of the transformer architecture is not similar to the inference of the architecture. We'll understand how data flows through both  the architectures.

- Training Architecture
<br> 
<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*0g4qdq7Rt6QvDalFFAkL5g.png" width="400" height="550" />
<br>

The Transformer processes the data like this:

    - The input sequence is converted into Embeddings (with Position Encoding) and fed to the Encoder.
    - The stack of Encoders processes this and produces an encoded representation of the input sequence.
    - The target sequence is prepended with a start-of-sentence token, converted into Embeddings (with Position Encoding), and fed to the Decoder.
    - The stack of Decoders processes this along with the Encoder stack’s encoded representation to produce an encoded representation of the target sequence.
    - The Output layer converts it into word probabilities and the final output sequence.
    - The Transformer’s Loss function compares this output sequence with the target sequence from the training data. This loss is used to generate gradients to train the Transformer during back-propagation.

- Inference
During Inference, we have only the input sequence and don’t have the target sequence to pass as input to the Decoder. The goal of the Transformer is to produce the target sequence from the input sequence alone.
So, like in a Seq2Seq model, we generate the output in a loop and feed the output sequence from the previous timestep to the Decoder in the next timestep until we come across an end-of-sentence token.
The difference from the Seq2Seq model is that, at each timestep, we re-feed the entire output sequence generated thus far, rather than just the last word.

<br> 
<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*-uvybwr8xULd3ug9ZwcSaQ.png" width="400" height="550" />
<br>

The flow of data during Inference is:

    - The input sequence is converted into Embeddings (with Position Encoding) and fed to the Encoder.
    - The stack of Encoders processes this and produces an encoded representation of the input sequence.
    Instead of the target sequence, we use an empty sequence with only a start-of-sentence token. This is converted into Embeddings (with Position Encoding) and fed to the Decoder.
    - The stack of Decoders processes this along with the Encoder stack’s encoded representation to produce an encoded representation of the target sequence.
    - The Output layer converts it into word probabilities and produces an output sequence.
    - We take the last word of the output sequence as the predicted word. That word is now filled into the second position of our Decoder input sequence, which now contains a start-of-sentence token and the first word.
    - Go back to step #3. As before, feed the new Decoder sequence into the model. Then take the second word of the output and append it to the Decoder sequence. Repeat this until it predicts an end-of-sentence token. Note that since the Encoder sequence does not change for each iteration, we do not have to repeat steps #1 and #2 each time (Thanks to Michal Kučírka for pointing this out).