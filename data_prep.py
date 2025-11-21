import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class PokerState:
    hole_cards: List[Tuple[str, str]]
    community_cards: List[Tuple[str, str]]
    pot: float
    position: int
    stack: float
    opponent_stacks: List[float]
    current_bet: float
    to_call: float
    street: int
    opponent_vpip: float
    opponent_pfr: float
    opponent_af: float
    opponent_wwsf: float
    recent_wins: int
    recent_losses: int
    streak: int
    action_history: List[Tuple[str, float]]


class HandHistoryParser:
    def __init__(self):
        self.stats_tracker = OpponentStatsTracker()
    
    def parse(self, history: Dict) -> List[PokerState]:
        states = []
        return states


class OpponentStatsTracker:
    def __init__(self):
        self.hands_played = defaultdict(int)
        self.voluntary_puts = defaultdict(int)
        self.preflop_raises = defaultdict(int)
        self.aggressive_actions = defaultdict(int)
        self.passive_actions = defaultdict(int)
        self.showdowns = defaultdict(int)
        self.showdown_wins = defaultdict(int)
    
    def update(self, player_id: str, action: str, street: int, is_showdown: bool = False, won: bool = False):
        self.hands_played[player_id] += 1
        
        if action in ['call', 'raise', 'bet'] and street == 0:
            self.voluntary_puts[player_id] += 1
            if action in ['raise', 'bet']:
                self.preflop_raises[player_id] += 1
        
        if action in ['raise', 'bet']:
            self.aggressive_actions[player_id] += 1
        elif action in ['call', 'check']:
            self.passive_actions[player_id] += 1
        
        if is_showdown:
            self.showdowns[player_id] += 1
            if won:
                self.showdown_wins[player_id] += 1
    
    def get_vpip(self, player_id: str) -> float:
        if self.hands_played[player_id] == 0:
            return 0.0
        return self.voluntary_puts[player_id] / self.hands_played[player_id]
    
    def get_pfr(self, player_id: str) -> float:
        if self.hands_played[player_id] == 0:
            return 0.0
        return self.preflop_raises[player_id] / self.hands_played[player_id]
    
    def get_af(self, player_id: str) -> float:
        passive = self.passive_actions[player_id]
        if passive == 0:
            return float('inf') if self.aggressive_actions[player_id] > 0 else 0.0
        return self.aggressive_actions[player_id] / passive
    
    def get_wwsf(self, player_id: str) -> float:
        if self.hands_played[player_id] == 0:
            return 0.0
        return self.showdowns[player_id] / self.hands_played[player_id]
    
    def get_all_stats(self, player_id: str) -> Dict[str, float]:
        return {
            'vpip': self.get_vpip(player_id),
            'pfr': self.get_pfr(player_id),
            'af': self.get_af(player_id),
            'wwsf': self.get_wwsf(player_id)
        }


class FeatureExtractor:
    def __init__(self, max_players: int = 6):
        self.max_players = max_players
    
    def extract(self, state: PokerState) -> np.ndarray:
        features = []
        
        features.extend(self._encode_cards(state.hole_cards, max_cards=2))
        features.extend(self._encode_cards(state.community_cards, max_cards=5))
        
        features.extend([
            state.pot / 1000.0,
            state.position / 9.0,
            state.stack / 1000.0,
            state.current_bet / 1000.0,
            state.to_call / 1000.0,
            state.street / 3.0,
        ])
        
        for i in range(self.max_players - 1):
            if i < len(state.opponent_stacks):
                features.append(state.opponent_stacks[i] / 1000.0)
            else:
                features.append(0.0)
        
        features.extend([
            state.opponent_vpip,
            state.opponent_pfr,
            min(state.opponent_af, 10.0) / 10.0,
            state.opponent_wwsf,
        ])
        
        features.extend([
            state.recent_wins / 10.0,
            state.recent_losses / 10.0,
            np.tanh(state.streak / 5.0),
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _encode_cards(self, cards: List[Tuple[str, str]], max_cards: int) -> List[float]:
        encoded = []
        ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                 '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suits = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
        
        for i in range(max_cards):
            if i < len(cards):
                rank, suit = cards[i]
                encoded.extend([
                    ranks.get(rank, 0) / 12.0,
                    suits.get(suit, 0) / 3.0
                ])
            else:
                encoded.extend([0.0, 0.0])
        
        return encoded
    
    def get_feature_dim(self) -> int:
        return 14 + 6 + (self.max_players - 1) + 4 + 3


class TiltTracker:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.recent_results = defaultdict(list)
    
    def update(self, player_id: str, result: float):
        self.recent_results[player_id].append(result)
        if len(self.recent_results[player_id]) > self.window_size:
            self.recent_results[player_id].pop(0)
    
    def get_features(self, player_id: str) -> Tuple[int, int, int]:
        results = self.recent_results[player_id]
        wins = sum(1 for r in results if r > 0)
        losses = sum(1 for r in results if r < 0)
        streak = 0
        for r in reversed(results):
            if r > 0:
                streak += 1
            elif r < 0:
                streak -= 1
            else:
                break
        return wins, losses, streak


if __name__ == "__main__":
    extractor = FeatureExtractor()
    print(f"Feature dimension: {extractor.get_feature_dim()}")
    
    state = PokerState(
        hole_cards=[('A', '♠'), ('K', '♠')],
        community_cards=[('Q', '♠'), ('J', '♠'), ('10', '♠')],
        pot=100.0,
        position=2,
        stack=1000.0,
        opponent_stacks=[800.0, 1200.0],
        current_bet=50.0,
        to_call=50.0,
        street=1,
        opponent_vpip=0.25,
        opponent_pfr=0.15,
        opponent_af=2.5,
        opponent_wwsf=0.20,
        recent_wins=3,
        recent_losses=1,
        streak=2,
        action_history=[]
    )
    
    features = extractor.extract(state)
    print(f"Extracted features shape: {features.shape}")

