import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import os
from model import BehaviorCloningModel, CombinedDQN

@dataclass
class Config:
    state_dim: int = 128
    action_dim: int = 3
    hidden_dims: List[int] = None
    use_lstm: bool = False
    use_trait_head: bool = False
    
    # Training
    lr: float = 1e-4
    batch_size: int = 64
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 1000
    alpha_human: float = 0.1
    
    buffer_size: int = 100000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 0.0001
    
    bc_epochs: int = 50
    bc_batch_size: int = 128
    
    train_steps: int = 100000
    eval_freq: int = 5000
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


class PokerEnvironment:
    def __init__(self, num_players=2, initial_stack=1000, small_blind=5, big_blind=10):
        self.num_players = num_players
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.suits = ['♠', '♥', '♦', '♣']
        self.rank_to_val = {r: i for i, r in enumerate(self.ranks)}
        
        self.reset()
    
    def reset(self):    
        self.deck = self._create_deck()
        self.stacks = [self.initial_stack] * self.num_players
        # >>> CHANGED: track starting stacks for net-change rewards
        self.starting_stacks = self.stacks.copy()
        # <<<
        self.pot = 0
        self.community_cards = []
        self.player_cards = [[] for _ in range(self.num_players)]
        self.current_bet = 0
        self.bets_this_round = [0] * self.num_players
        self.street = 'preflop'
        self.dealer = random.randint(0, self.num_players - 1)
        self.action_history = []
        self.hand_over = False
        
        for _ in range(2):
            for p in range(self.num_players):
                self.player_cards[p].append(self.deck.pop())
        
        sb_idx = (self.dealer + 1) % self.num_players
        bb_idx = (self.dealer + 2) % self.num_players
        self.stacks[sb_idx] -= self.small_blind
        self.stacks[bb_idx] -= self.big_blind
        self.bets_this_round[sb_idx] = self.small_blind
        self.bets_this_round[bb_idx] = self.big_blind
        self.current_bet = self.big_blind
        self.pot = self.small_blind + self.big_blind
        
        return self.get_state(0)
    
    def _create_deck(self):
        deck = [(r, s) for r in self.ranks for s in self.suits]
        random.shuffle(deck)
        return deck
    
    def get_state(self, player_idx):
        features = []
        
        hand_features = self._get_hand_features(self.player_cards[player_idx], self.community_cards)
        features.extend(hand_features)
        
        features.append(self.pot / 1000.0)  # Normalized
        features.append(self.stacks[player_idx] / 1000.0)
        features.append(self.current_bet / 1000.0)
        features.append(sum(self.bets_this_round) / 1000.0)
        
        position = (player_idx - self.dealer) % self.num_players
        features.append(position / self.num_players)
        
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        features.append(street_map[self.street] / 3.0)
        
        for i in range(self.num_players):
            if i != player_idx:
                features.append(self.stacks[i] / 1000.0)
                features.extend([0.3, 0.2, 2.0, 0.5])
        
        history_features = [0.0] * 15  # 5 actions * 3 features
        for i, (p, a, amt) in enumerate(self.action_history[-5:]):
            history_features[i*3] = 1.0 if p == player_idx else 0.0
            history_features[i*3+1] = a / 2.0
            history_features[i*3+2] = min(amt / 100.0, 1.0)
        features.extend(history_features)
        
        target_size = 128
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _get_hand_features(self, hole_cards, community_cards):
        all_cards = hole_cards + community_cards
        if len(all_cards) < 2:
            return [0.0] * 10
        
        rank_counts = defaultdict(int)
        suit_counts = defaultdict(int)
        for r, s in all_cards:
            rank_counts[r] += 1
            suit_counts[s] += 1
        
        hole_ranks = [self.rank_to_val.get(r, 0) for r, _ in hole_cards]
        high_card = max(hole_ranks) / 12.0 if hole_ranks else 0.0
        
        pairs = sum(1 for count in rank_counts.values() if count >= 2)
        trips = sum(1 for count in rank_counts.values() if count >= 3)
        quads = sum(1 for count in rank_counts.values() if count >= 4)
        flush = max(suit_counts.values()) >= 5 if len(all_cards) >= 5 else False
        
        hand_strength = high_card
        if pairs > 0:
            hand_strength += 0.2
        if trips > 0:
            hand_strength += 0.3
        if quads > 0:
            hand_strength += 0.4
        if flush:
            hand_strength += 0.3
        
        return [
            high_card,
            min(pairs / 3.0, 1.0),
            min(trips / 2.0, 1.0),
            float(quads > 0),
            float(flush),
            hand_strength,
            len(community_cards) / 5.0,
            sum(hole_ranks) / 24.0,
            float(hole_cards[0][0] == hole_cards[1][0] if len(hole_cards) == 2 else False),  # Pocket pair
            float(len(set(s for _, s in hole_cards)) == 1) if len(hole_cards) == 2 else 0.0  # Suited
        ]
    
    def get_valid_actions(self, player_idx):
        stack = self.stacks[player_idx]
        to_call = self.current_bet - self.bets_this_round[player_idx]
        
        actions = []
        actions.append(('fold', 0))
        if to_call <= stack:
            actions.append(('call', to_call))
        if stack > to_call:
            min_raise = max(self.big_blind, self.current_bet * 2 - sum(self.bets_this_round))
            max_raise = stack
            if min_raise <= max_raise:
                raise_amounts = [min_raise, (min_raise + max_raise) // 2, max_raise]
                for amt in raise_amounts:
                    if amt <= stack:
                        actions.append(('raise', amt))
        
        return actions
    
    def step(self, player_idx, action_type, amount=0):
        if self.hand_over:
            return self.get_state(player_idx), 0.0, True, {'pot': self.pot, 'stack': self.stacks[player_idx]}
        
        valid_actions = self.get_valid_actions(player_idx)
        action_map = {a[0]: a for a in valid_actions}
        
        if action_type not in action_map:
            action_type = valid_actions[0][0]
        
        action_type, amount = action_map[action_type]
        
        done = False
        
        if action_type == 'fold':
            # Fold = end of hand, opponent wins pot
            opponent = (player_idx + 1) % self.num_players
            self.stacks[opponent] += self.pot
            self.pot = 0
            self.hand_over = True
            done = True
        elif action_type == 'call':
            call_amount = min(amount, self.stacks[player_idx])
            self.stacks[player_idx] -= call_amount
            self.bets_this_round[player_idx] += call_amount
            self.pot += call_amount
            self.current_bet = max(self.current_bet, self.bets_this_round[player_idx])
        elif action_type == 'raise':
            raise_amount = min(amount, self.stacks[player_idx])
            self.stacks[player_idx] -= raise_amount
            self.bets_this_round[player_idx] += raise_amount
            self.pot += raise_amount
            self.current_bet = self.bets_this_round[player_idx]
        
        self.action_history.append(
            (player_idx, 0 if action_type == 'fold' else (1 if action_type == 'call' else 2), amount)
        )
        
        # Check if someone is all-in - go straight to showdown
        all_in_players = [i for i in range(self.num_players) if self.stacks[i] == 0]
        if len(all_in_players) > 0 and not self.hand_over:
            # Run out remaining community cards
            while len(self.community_cards) < 5:
                self.community_cards.append(self.deck.pop())
            # Showdown - determine winner
            winner = self._determine_winner()
            self.stacks[winner] += self.pot
            self.pot = 0
            self.hand_over = True
            done = True
        elif not self.hand_over:
            # Simulate opponent action (simple strategy)
            self._simulate_opponent_action(player_idx)
            
            # Check for street transitions after both players have acted
            if self._betting_round_complete():
                if self.street == 'preflop':
                    self.street = 'flop'
                    self.community_cards = [self.deck.pop() for _ in range(3)]
                    self.bets_this_round = [0] * self.num_players
                    self.current_bet = 0
                elif self.street == 'flop':
                    self.street = 'turn'
                    self.community_cards.append(self.deck.pop())
                    self.bets_this_round = [0] * self.num_players
                    self.current_bet = 0
                elif self.street == 'turn':
                    self.street = 'river'
                    self.community_cards.append(self.deck.pop())
                    self.bets_this_round = [0] * self.num_players
                    self.current_bet = 0
                else:  # river complete
                    winner = self._determine_winner()
                    self.stacks[winner] += self.pot
                    self.pot = 0
                    self.hand_over = True
                    done = True
        
        # Reward is net chip change vs starting stack, only when hand ends
        if self.hand_over:
            reward = float(self.stacks[player_idx] - self.starting_stacks[player_idx])
        else:
            reward = 0.0
        
        next_state = self.get_state(player_idx)
        info = {'pot': self.pot, 'stack': self.stacks[player_idx]}
        
        return next_state, reward, done, info
    
    def _simulate_opponent_action(self, hero_idx):
        """Simulate a simple opponent strategy."""
        opp_idx = (hero_idx + 1) % self.num_players
        if self.stacks[opp_idx] == 0:
            return  # Opponent is all-in
        
        to_call = self.current_bet - self.bets_this_round[opp_idx]
        
        # Simple opponent strategy based on pot odds and randomness
        if to_call == 0:
            # Check or small bet
            if random.random() < 0.3 and self.stacks[opp_idx] > self.big_blind:
                bet_amt = min(self.big_blind * 2, self.stacks[opp_idx])
                self.stacks[opp_idx] -= bet_amt
                self.bets_this_round[opp_idx] += bet_amt
                self.pot += bet_amt
                self.current_bet = self.bets_this_round[opp_idx]
            # else check (do nothing)
        elif to_call <= self.stacks[opp_idx]:
            # Decide to call, raise, or fold
            pot_odds = to_call / (self.pot + to_call) if self.pot > 0 else 0.5
            
            if pot_odds > 0.4 and random.random() < 0.3:
                # Fold against big bets sometimes
                self.stacks[hero_idx] += self.pot
                self.pot = 0
                self.hand_over = True
            else:
                # Call
                call_amt = min(to_call, self.stacks[opp_idx])
                self.stacks[opp_idx] -= call_amt
                self.bets_this_round[opp_idx] += call_amt
                self.pot += call_amt
        else:
            # Can't afford to call - fold
            self.stacks[hero_idx] += self.pot
            self.pot = 0
            self.hand_over = True
    
    def _betting_round_complete(self):
        """Check if the betting round is complete."""
        if self.hand_over:
            return False
        # All players have matched the current bet or are all-in
        for i in range(self.num_players):
            if self.stacks[i] > 0 and self.bets_this_round[i] < self.current_bet:
                return False
        # Need at least one action in the round (not just blinds)
        return len(self.action_history) > 0
    
    def _determine_winner(self):
        """Determine the winner based on hand strength (simplified)."""
        best_player = 0
        best_strength = -1
        
        for p in range(self.num_players):
            strength = self._evaluate_hand(self.player_cards[p], self.community_cards)
            if strength > best_strength:
                best_strength = strength
                best_player = p
        
        return best_player
    
    def _evaluate_hand(self, hole_cards, community_cards):
        """Evaluate hand strength (simplified scoring)."""
        all_cards = hole_cards + community_cards
        if len(all_cards) < 5:
            return random.random()  # Not enough cards, random
        
        rank_counts = defaultdict(int)
        suit_counts = defaultdict(int)
        ranks = []
        
        for r, s in all_cards:
            rank_counts[r] += 1
            suit_counts[s] += 1
            ranks.append(self.rank_to_val.get(r, 0))
        
        # Count hand types
        pairs = sum(1 for c in rank_counts.values() if c == 2)
        trips = sum(1 for c in rank_counts.values() if c == 3)
        quads = sum(1 for c in rank_counts.values() if c == 4)
        flush = max(suit_counts.values()) >= 5
        
        # Check for straight
        unique_ranks = sorted(set(ranks))
        straight = False
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i+4] - unique_ranks[i] == 4:
                straight = True
                break
        # Wheel (A-2-3-4-5)
        if set([0, 1, 2, 3, 12]).issubset(set(ranks)):
            straight = True
        
        # Score the hand
        high_card = max(ranks)
        score = high_card / 100.0  # Base score from high card
        
        if quads:
            score += 7.0
        elif trips and pairs:  # Full house
            score += 6.0
        elif flush:
            score += 5.0
        elif straight:
            score += 4.0
        elif trips:
            score += 3.0
        elif pairs >= 2:
            score += 2.0
        elif pairs == 1:
            score += 1.0
        
        return score


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.0001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done, human_log_prob=None):
        experience = (state, action, reward, next_state, done, human_log_prob)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = list(zip(*experiences))
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(batch[1])
        rewards = torch.FloatTensor(batch[2])
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.BoolTensor(batch[4])
        human_log_probs = torch.FloatTensor(batch[5]) if batch[5][0] is not None else None
        
        return (states, actions, rewards, next_states, dones, human_log_probs), indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class PokerTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.bc_model = BehaviorCloningModel(
            config.state_dim, config.action_dim, 
            config.hidden_dims, config.use_lstm
        ).to(self.device)
        
        self.dqn_model = CombinedDQN(
            config.state_dim, config.action_dim,
            config.hidden_dims, config.use_lstm, config.use_trait_head
        ).to(self.device)
        
        self.target_model = CombinedDQN(
            config.state_dim, config.action_dim,
            config.hidden_dims, config.use_lstm, config.use_trait_head
        ).to(self.device)
        self.target_model.load_state_dict(self.dqn_model.state_dict())
        
        # Optimizers
        self.bc_optimizer = optim.Adam(self.bc_model.parameters(), lr=config.lr)
        self.dqn_optimizer = optim.Adam(self.dqn_model.parameters(), lr=config.lr)
        
        self.env = PokerEnvironment()
        self.replay_buffer = PrioritizedReplayBuffer(
            config.buffer_size, config.per_alpha, 
            config.per_beta, config.per_beta_increment
        )

        self.epsilon = config.epsilon_start
        self.step_count = 0
        self.human_policy = None
    
    def train_behavior_cloning(self, human_data: List[Tuple]):
        print("Training Behavior Cloning Model...")
        
        states = torch.FloatTensor(np.array([s for s, _, _ in human_data])).to(self.device)
        actions = torch.LongTensor(np.array([a for _, a, _ in human_data])).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(states, actions)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.bc_batch_size, shuffle=True
        )
        
        for epoch in range(self.config.bc_epochs):
            total_loss = 0
            for batch_states, batch_actions in dataloader:
                self.bc_optimizer.zero_grad()
                log_probs, _ = self.bc_model.get_log_probs(batch_states, batch_actions)
                loss = -log_probs.mean()
                loss.backward()
                self.bc_optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"BC Epoch {epoch+1}/{self.config.bc_epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.human_policy = self.bc_model
        self.human_policy.eval()
        print("Behavior Cloning Complete!")
    
    def get_action(self, state, training=True):
        valid_actions = self.env.get_valid_actions(0)
        action_mask = torch.zeros(self.config.action_dim).to(self.device)
        
        action_map = {}
        for i, (action_type, amt) in enumerate(valid_actions):
            if action_type == 'fold':
                action_map[0] = (action_type, 0)
                action_mask[0] = 1
            elif action_type == 'call':
                action_map[1] = (action_type, amt)
                action_mask[1] = 1
            elif action_type == 'raise':
                action_map[2] = (action_type, amt)
                action_mask[2] = 1
        
        if training and random.random() < self.epsilon:
            valid_indices = [i for i, m in enumerate(action_mask.cpu().numpy()) if m > 0]
            action_idx = random.choice(valid_indices) if valid_indices else 0
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values, _ = self.dqn_model(state_tensor)
                q_values = q_values * action_mask - (1 - action_mask) * 1e9
                action_idx = q_values.argmax().item()
        
        if training:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )
        
        return action_idx, action_map.get(action_idx, ('fold', 0))
    
    def compute_reward(self, chips_reward, state, action):
        if self.human_policy is None:
            return chips_reward
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.LongTensor([action]).to(self.device)
            log_prob, _ = self.human_policy.get_log_probs(state_tensor, action_tensor)
            human_bonus = self.config.alpha_human * log_prob.item()
        
        return chips_reward + human_bonus
    
    def train_dqn_step(self):
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        (states, actions, rewards, next_states, dones, human_log_probs), indices, weights = \
            self.replay_buffer.sample(self.config.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        q_values, _ = self.dqn_model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values, _ = self.dqn_model(next_states)
            next_actions = next_q_values.argmax(1)
            next_q_values_target, _ = self.target_model(next_states)
            next_q_value = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones.float()) * self.config.gamma * next_q_value
        
        td_errors = target_q - q_value
        loss = (weights * td_errors.pow(2)).mean()
        
        self.dqn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn_model.parameters(), 1.0)
        self.dqn_optimizer.step()
        
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        if self.step_count % self.config.target_update_freq == 0:
            self.target_model.load_state_dict(self.dqn_model.state_dict())
        
        self.step_count += 1
        return loss.item()
    
    def train(self, num_steps=None):
        num_steps = num_steps or self.config.train_steps
        
        print("Generating synthetic human data for BC...")
        human_data = self._generate_synthetic_human_data(1000)
        self.train_behavior_cloning(human_data)
        
        print("Starting RL Training...")
        state = self.env.reset()
        episode_rewards = []
        episode_reward = 0
        episode_steps = 0
        max_episode_steps = 100
        
        for step in range(num_steps):
            action_idx, (action_type, amount) = self.get_action(state, training=True)

            next_state, chips_reward, done, info = self.env.step(0, action_type, amount)
            
            reward = self.compute_reward(chips_reward, state, action_idx)
            
            human_log_prob = None
            if self.human_policy is not None:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_tensor = torch.LongTensor([action_idx]).to(self.device)
                    log_prob, _ = self.human_policy.get_log_probs(state_tensor, action_tensor)
                    human_log_prob = log_prob.item()
            
            self.replay_buffer.add(state, action_idx, reward, next_state, done, human_log_prob)
            
            episode_reward += chips_reward
            state = next_state
            episode_steps += 1
            
            loss = self.train_dqn_step()
            
            if done or episode_steps >= max_episode_steps:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                episode_steps = 0
                state = self.env.reset()
            
            if (step + 1) % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                loss_val = loss if loss else 0.0
                print(f"Step {step+1}/{num_steps}, Epsilon: {self.epsilon:.3f}, "
                      f"Avg Reward: {avg_reward:.2f}, Loss: {loss_val:.4f}")
            
            if (step + 1) % self.config.eval_freq == 0:
                self.evaluate()
    
    def _generate_synthetic_human_data(self, num_samples):
        data = []
        for _ in range(num_samples):
            state = self.env.reset()
            hand_strength = state[0]
            if hand_strength > 0.7:
                action = 2
            elif hand_strength > 0.4:
                action = 1
            else:
                action = 0
            data.append((state, action, None))
        return data
    
    def compute_kl_divergence(self, states):
        if self.human_policy is None:
            return None
        
        self.dqn_model.eval()
        self.human_policy.eval()
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            q_values, _ = self.dqn_model(states_tensor)
            dqn_probs = F.softmax(q_values, dim=-1)
            dqn_probs = dqn_probs.cpu().numpy()
            human_probs, _ = self.human_policy(states_tensor)
            human_probs = human_probs.cpu().numpy()
            kl = (human_probs * (np.log(human_probs + 1e-8) - np.log(dqn_probs + 1e-8))).sum(axis=-1)
            avg_kl = kl.mean()
        
        self.dqn_model.train()
        self.human_policy.eval()
        return avg_kl
    
    def compute_stats(self, states, actions):
        if len(actions) == 0:
            return {}
        
        actions = np.array(actions)
        total_hands = len(actions)
        
        vpip = np.mean(actions > 0) * 100
        
        pfr = np.mean(actions == 2) * 100
        
        raises = np.sum(actions == 2)
        calls = np.sum(actions == 1)
        af = raises / (calls + 1e-6)
        
        fold_pct = np.mean(actions == 0) * 100
        
        return {
            'VPIP': vpip,
            'PFR': pfr,
            'AF': af,
            'Fold%': fold_pct,
            'Total Hands': total_hands
        }
    
    def evaluate(self):
        self.dqn_model.eval()
        wins = 0
        total_hands = 20
        eval_states = []
        eval_actions = []
        eval_rewards = []
        
        for _ in range(total_hands):
            state = self.env.reset()
            done = False
            hand_actions = []
            hand_reward = 0.0
            max_steps = 50
            step_count = 0
            while not done and step_count < max_steps:
                eval_states.append(state.copy())
                action_idx, (action_type, amount) = self.get_action(state, training=False)
                hand_actions.append(action_idx)
                state, reward, done, _ = self.env.step(0, action_type, amount)
                hand_reward += reward
                step_count += 1
            
            # >>> CHANGED: win if net hand profit > 0
            if hand_reward > 0:
                wins += 1
            # <<<
            
            eval_actions.extend(hand_actions)
            eval_rewards.append(hand_reward)
        
        winrate = wins / total_hands if total_hands > 0 else 0.0
        
        agent_stats = self.compute_stats(eval_states[:len(eval_actions)], eval_actions)
        
        kl_div = self.compute_kl_divergence(eval_states[:100])  # Sample for efficiency
        
        avg_reward = np.mean(eval_rewards) if eval_rewards else 0.0
        # >>> CHANGED: report BB/100 as well
        bb_per_100 = (avg_reward / self.env.big_blind) * 100.0 if self.env.big_blind > 0 else 0.0
        # <<<
        
        print(f"\n=== Evaluation Results ===")
        print(f"Winrate: {winrate:.2%}")
        print(f"Avg Reward (chips/hand): {avg_reward:.2f}")
        print(f"BB/100: {bb_per_100:.2f}")
        if kl_div is not None:
            print(f"KL Divergence (human || agent): {kl_div:.4f}")
        print(f"Agent Stats:")
        for key, value in agent_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 30 + "\n")
        
        self.dqn_model.train()
        return {
            'winrate': winrate,
            'avg_reward': avg_reward,
            'bb_per_100': bb_per_100,
            'kl_divergence': kl_div,
            'stats': agent_stats
        }
    
    def inference(self, state):
        self.dqn_model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values, _ = self.dqn_model(state_tensor)
            
            valid_actions = self.env.get_valid_actions(0)
            action_mask = torch.zeros(self.config.action_dim).to(self.device)
            action_map = {}
            
            for i, (action_type, amt) in enumerate(valid_actions):
                if action_type == 'fold':
                    action_map[0] = (action_type, 0)
                    action_mask[0] = 1
                elif action_type == 'call':
                    action_map[1] = (action_type, amt)
                    action_mask[1] = 1
                elif action_type == 'raise':
                    action_map[2] = (action_type, amt)
                    action_mask[2] = 1
            
            q_values = q_values * action_mask - (1 - action_mask) * 1e9
            action_idx = q_values.argmax().item()
            
            probs = F.softmax(q_values, dim=-1).cpu().numpy()[0]
        
        self.dqn_model.train()
        return action_idx, action_map.get(action_idx, ('fold', 0)), probs
    
    def save(self, path):
        torch.save({
            'dqn_model': self.dqn_model.state_dict(),
            'bc_model': self.bc_model.state_dict(),
            'config': self.config,
            'step_count': self.step_count
        }, path)
        print(f"Models saved to {path}")
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.dqn_model.load_state_dict(checkpoint['dqn_model'])
        self.bc_model.load_state_dict(checkpoint['bc_model'])
        self.step_count = checkpoint.get('step_count', 0)
        print(f"Models loaded from {path}")


if __name__ == "__main__":
    config = Config()
    trainer = PokerTrainer(config)
    trainer.train()
    trainer.save("poker_dqn_checkpoint.pt")
