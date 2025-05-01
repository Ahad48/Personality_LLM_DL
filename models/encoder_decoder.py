import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, p_tags=5,ignore_index = 1, sos_index = 101):
        super(EncoderDecoder, self).__init__()

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

        self.transformer_layer = nn.Transformer(hidden_dim, num_heads, num_layers_enc, num_layers_dec, dim_feedforward, dropout, batch_first=True)
        
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.personality_layer = nn.Linear(p_tags, hidden_dim)

        self.final_linear_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_text, output_text, input_personality):

        batch_size,input_len = input_text.shape
        _, output_len = output_text.shape
        position_input = torch.arange(input_len).repeat(batch_size,1).to(device = self.device)
        position_output = torch.arange(output_len).repeat(batch_size,1).to(device = self.device)


        input_emb = self.position_embedding(position_input)
        output_emb = self.position_embedding(position_output)

        input_emb += self.word_embedding(input_text) # B,len, hidden
        output_emb += self.word_embedding(output_text)

        personality = F.softmax(input_personality, dim = -1) # B,hidden
        personality = self.personality_layer(personality).unsqueeze(1) #B,1,hidden

        input_emb += personality # B, len, hidden

        input_padding_mask = input_text == self.pad_idx
        output_padding_mask = output_text == self.pad_idx

        output_mask = self.transformer_layer.generate_square_subsequent_mask(output_len,device=self.device)

        out = self.transformer_layer(input_emb, output_emb, tgt_mask = output_mask,src_key_padding_mask=input_padding_mask , tgt_key_padding_mask = output_padding_mask)

        out = self.final_linear_layer(out)

        return out
    
    @torch.no_grad()
    def generate(self, encoder_input ,personality, tokenizer, temperature = 0.7, top_k = 100, repeation_penalty = 1.7, repeat_array_len = 10, top_p = 0.8):
        self.eval()

        encoder_input = tokenizer(encoder_input, max_length = 150, padding = "max_length", add_special_tokens = True, return_tensors = "pt", truncation = True)
        encoder_input = encoder_input['input_ids'].to(self.device)
        decoder_input = torch.tensor(self.pad_idx, device=self.device).repeat(encoder_input.shape[0], self.max_length)          #used as an temporary variable to keep track of predicted tokens
        decoder_input[:,0] = self.sos_idx
        recent_tokens = []
        # print(decoder_input.shape)
        for t in range(self.max_length-1):
            output = self.forward(encoder_input, decoder_input, personality)
            next_token_logits = output[:,t,:] # B, 1, vocab_size

            if recent_tokens:
                recent_tokens_tensor = torch.tensor(recent_tokens, dtype = torch.long, device = self.device)
                penalty_tensor = torch.ones_like(next_token_logits)
                penalty_tensor.index_fill_(1, recent_tokens_tensor, repeation_penalty)

                next_token_logits /= penalty_tensor

            next_token_logits /= temperature
            # print(torch.nansum(nex_token_logits))

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
            # print(outputs.shape)
            # print(outputs.argmax(-1).shape)

            if(t<self.max_length-1):
                # decoder_input[:,t+1] = output.argmax(-1)
                decoder_input[:,t+1] = next_token
                # print(new_token)

            recent_tokens.append(next_token.item())
            if len(recent_tokens)>repeat_array_len:
                recent_tokens.pop(0)
 

        decoder_input = tokenizer.decode(decoder_input.tolist()[0])
        return decoder_input
            
            
