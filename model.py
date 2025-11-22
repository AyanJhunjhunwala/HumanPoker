import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SharedEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dims=[256, 256], use_lstm=False, lstm_hidden=128):
        super().__init__()
        self.use_lstm = use_lstm
        
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        
        if use_lstm:
            self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True)
            self.encoder_dim = lstm_hidden
        else:
            self.encoder_dim = input_dim
    
    def forward(self, x, hidden=None):
        if self.use_lstm:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            x = self.mlp(x)
            lstm_out, hidden = self.lstm(x, hidden)
            return lstm_out[:, -1, :], hidden
        else:
            return self.mlp(x), None


class BehaviorCloningModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], use_lstm=False):
        super().__init__()
        self.encoder = SharedEncoder(state_dim, hidden_dims, use_lstm)
        self.policy_head = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state, hidden=None):
        encoded, hidden = self.encoder(state, hidden)
        logits = self.policy_head(encoded)
        return F.softmax(logits, dim=-1), hidden
    
    def get_log_probs(self, state, action, hidden=None):
        encoded, hidden = self.encoder(state, hidden)
        logits = self.policy_head(encoded)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, action.unsqueeze(1)).squeeze(1), hidden


class DuelingDQN(nn.Module):
    """Dueling Double DQN architecture"""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], use_lstm=False):
        super().__init__()
        self.encoder = SharedEncoder(state_dim, hidden_dims, use_lstm)
        encoder_dim = self.encoder.encoder_dim
        
        self.value_stream = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state, hidden=None):
        encoded, hidden = self.encoder(state, hidden)
        value = self.value_stream(encoded)
        advantage = self.advantage_stream(encoded)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values, hidden


class TraitHead(nn.Module):
    def __init__(self, encoder_dim, num_traits=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_traits)
        )
    
    def forward(self, encoded_state):
        return F.softmax(self.head(encoded_state), dim=-1)


class CombinedDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], 
                 use_lstm=False, use_trait_head=False, num_traits=4):
        super().__init__()
        self.dqn = DuelingDQN(state_dim, action_dim, hidden_dims, use_lstm)
        self.use_trait_head = use_trait_head
        if use_trait_head:
            self.trait_head = TraitHead(self.dqn.encoder.encoder_dim, num_traits)
    
    def forward(self, state, hidden=None, return_traits=False):
        q_values, hidden = self.dqn(state, hidden)
        if self.use_trait_head and return_traits:
            encoded, _ = self.dqn.encoder(state, hidden)
            traits = self.trait_head(encoded)
            return q_values, traits, hidden
        return q_values, hidden

