import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderModel(nn.Module):
    def __init__(self, vocab_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_dec=2, dropout=0.2, max_length=43, p_tags=5,ignore_index = 1, sos_index = 101):
        super(DecoderModel, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.device = device
        self.p_tags = p_tags
        self.pad_idx=ignore_index
        self.sos_idx = sos_index

        self.word_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.personality_layer = nn.Sequential(
            nn.Linear(p_tags, hidden_dim),
            nn.ReLU()
        )

        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout, batch_first=True)

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers_dec)

        self.final_layer = nn.Linear(hidden_dim, vocab_size)

        self.apply(self.init_weights)

    def init_weights(self, m):
        ## using a similar weight initialization from resnet 
        # https://discuss.pytorch.org/t/initialising-weights-in-nn-sequential/76553/6 using this for understanding how to initialize the weights
        # using kaiming initilization as it is good for relu
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_text, personality = None, attention_mask = None):
        batch_size, input_len = input_text.size()

        
        word_emd = self.word_embedding(input_text)
        

        position = torch.arange(0, input_len, device=self.device).unsqueeze(0)
        # print(position.shape)

        position_emd = self.position_embedding(position)

        # if personality is given then train the model according to the personality
        if personality!=None:
            personality_embed = self.personality_layer(personality)
            position_emd = personality_embed + position_emd


        embeddings = position_emd + word_emd

        # tgt_mask = torch.triu(torch.ones(input_len, input_len, device=self.device), diagonal=1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(input_len).to(self.device)
        
        if attention_mask!=None:
            attention_mask = attention_mask == 0
            out = self.decoder(embeddings, memory = embeddings, tgt_mask = tgt_mask, tgt_key_padding_mask = attention_mask)

        else:
            out = self.decoder(embeddings, memory = embeddings, tgt_mask = tgt_mask)

        out = self.final_layer(out)

        return out
    
    @torch.no_grad()
    def generate(self, input_text, personality, tokenizer, temperature = 0.7, top_k = 10, top_p = 0.9,
                 repeation_penalty = 1.2, repeat_array_len = 4):
        self.eval()
        
        input_text = f"Question: {input_text} \nAnswer:"
        input_index = tokenizer(input_text, add_special_tokens = True, return_tensors = "pt")['input_ids']
        # input_index = torch.tensor(input_index, dtype=torch.long, device=self.device)

        generated = input_index.to(device = self.device)
        # output_text = torch.ones((1,1), device=self.device, dtype = torch.long)
        # output_text[:] = tokenizer.bos_token_id

        recent_tokens = []

        for _ in range(self.max_length):
            # take the sequence of tokens from the max length incase the sequence gets too long
            model_input = generated[:,-self.max_length:]

            if personality != None:
                out = self.forward(model_input, personality)

            else:
                out = self.forward(model_input)

            # take all logits for the most recent generated token for the entire sequence
            next_token_logits = out[:,-1,:]

            # skip checking if the recenet tokens is empty
            if recent_tokens:
                recent_tokens_tensor = torch.tensor(recent_tokens, dtype = torch.long, device = self.device)
                penalty_tensor = torch.ones_like(next_token_logits)
                penalty_tensor.index_fill_(1, recent_tokens_tensor, repeation_penalty)

                next_token_logits /= penalty_tensor

            next_token_logits /= temperature

            if top_k>0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits.scatter_(1, top_k_indices, top_k_logits)

                next_token_logits = filtered_logits
            
            if top_p<1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                sorted_prob = F.softmax(sorted_logits, dim = -1)

                cumm_prob = torch.cumsum(sorted_prob, dim=-1)
                # remove the indices where the prob>=top_p
                remove_indices = cumm_prob > top_p
                remove_indices[:,0] = False
                sorted_logits[remove_indices] = float("-inf")

                full_arry = torch.full_like(next_token_logits, float('-inf'), device=self.device)
                full_arry.scatter_(1, sorted_indices, sorted_logits)
                next_token_logits = full_arry
            
            probs = F.softmax(next_token_logits, dim = -1)
            # print(_)
            # print(probs)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim = 1)
            # output_text = torch.cat((output_text, next_token), dim = 1)

            recent_tokens.append(next_token.item())
            if len(recent_tokens)>repeat_array_len:
                recent_tokens.pop(0)

        output_text = tokenizer.decode(generated.squeeze(0).tolist())

        self.train()
        return output_text
