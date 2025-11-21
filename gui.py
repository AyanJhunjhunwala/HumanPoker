import tkinter as tk
from tkinter import ttk
import traceback

# --- Configuration & Colors (Dark Modern Theme) ---
COLORS = {
    "bg": "#1E1E24",           # Deep dark background
    "surface": "#292930",      # Slightly lighter for panels
    "text": "#EAEAEA",         # Off-white text
    "accent_fold": "#FF4B4B",  # Red
    "accent_check": "#4CAF50", # Green
    "accent_raise": "#3D91FF", # Blue
    "card_bg": "#F0F0F0",      # White-ish for cards
    "card_border": "#CCCCCC",
    "success": "#00E676"       # Bright Green for wins
}

FONTS = {
    "header": ("Segoe UI", 24, "bold"),
    "sub": ("Segoe UI", 14),
    "label": ("Segoe UI", 10, "bold"),
    "card": ("Times New Roman", 18, "bold")
}

# --- UI Components ---

class ModernButton(tk.Button):
    """A custom button class that supports flat design and hover effects."""
    def __init__(self, master, text, color, command=None, **kwargs):
        super().__init__(master, text=text, command=command, **kwargs)
        self.default_bg = color
        self.hover_bg = self._adjust_color_lightness(color, 1.2)
        self.disabled_bg = "#555555"
        
        self.configure(
            bg=self.default_bg,
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            bd=0,
            padx=20,
            pady=10,
            cursor="hand2",
            activebackground=self.hover_bg,
            activeforeground="white"
        )
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        if self['state'] != 'disabled':
            self['bg'] = self.hover_bg

    def on_leave(self, e):
        if self['state'] != 'disabled':
            self['bg'] = self.default_bg

    def _adjust_color_lightness(self, color_hex, factor):
        try:
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            r = min(255, int(r * factor))
            g = min(255, int(g * factor))
            b = min(255, int(b * factor))
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return color_hex

class CardWidget(tk.Frame):
    """Visual representation of a Poker Card."""
    def __init__(self, master, rank, suit):
        super().__init__(master, bg=COLORS["card_bg"], width=60, height=85)
        self.pack_propagate(False)
        
        color = "#D32F2F" if suit in ['♥', '♦'] else "#212121"
        
        # Top Left
        tk.Label(self, text=f"{rank}\n{suit}", fg=color, bg=COLORS["card_bg"], 
                 font=("Arial", 10, "bold"), justify="left").place(x=2, y=2)
        
        # Center
        tk.Label(self, text=suit, fg=color, bg=COLORS["card_bg"], 
                 font=("Arial", 24)).place(relx=0.5, rely=0.5, anchor="center")

        # Bottom Right (Inverted)
        tk.Label(self, text=f"{rank}\n{suit}", fg=color, bg=COLORS["card_bg"], 
                 font=("Arial", 10, "bold"), justify="right").place(relx=1.0, rely=1.0, x=-2, y=-2, anchor="se")

        self.config(highlightbackground=COLORS["card_border"], highlightthickness=1)

# --- Main Interface Class ---

class PokerGUI(tk.Frame):
    """
    A pure View class. It displays cards and buttons, and calls 
    `on_action_callback(action, amount)` when the user interacts.
    """
    def __init__(self, master, on_action_callback=None):
        super().__init__(master, bg=COLORS["bg"])
        self.pack(fill="both", expand=True)
        
        self.callback = on_action_callback
        
        self._setup_layout()
        self._setup_ui_components()

    def _setup_layout(self):
        self.top_panel = tk.Frame(self, bg=COLORS["bg"], pady=20)
        self.top_panel.pack(fill="x")
        
        self.mid_panel = tk.Frame(self, bg=COLORS["bg"], pady=20)
        self.mid_panel.pack(fill="both", expand=True)
        
        self.btm_panel = tk.Frame(self, bg=COLORS["surface"], pady=30, padx=30)
        self.btm_panel.pack(fill="x", side="bottom")

    def _setup_ui_components(self):
        # Pot & Status
        self.lbl_pot = tk.Label(self.top_panel, text="POT: $0", 
                                fg="#FFD700", bg=COLORS["bg"], font=FONTS["header"])
        self.lbl_pot.pack()
        
        self.lbl_status = tk.Label(self.top_panel, text="Waiting for game...", 
                 fg=COLORS["text"], bg=COLORS["bg"], font=FONTS["sub"])
        self.lbl_status.pack(pady=(0, 20))

        # Community Cards
        tk.Label(self.top_panel, text="COMMUNITY CARDS", fg="#888", bg=COLORS["bg"], 
                 font=("Segoe UI", 8, "bold")).pack()
        self.community_frame = tk.Frame(self.top_panel, bg=COLORS["bg"])
        self.community_frame.pack(pady=5)

        # Player Hand
        tk.Label(self.mid_panel, text="YOUR HAND", fg="#888", bg=COLORS["bg"], 
                 font=("Segoe UI", 8, "bold")).pack()
        self.hand_frame = tk.Frame(self.mid_panel, bg=COLORS["bg"])
        self.hand_frame.pack(pady=5)

        # Balance
        self.lbl_balance = tk.Label(self.mid_panel, text="Balance: $0", 
                                    fg=COLORS["success"], bg=COLORS["bg"], font=FONTS["sub"])
        self.lbl_balance.pack(pady=10)

        # --- Controls ---
        slider_frame = tk.Frame(self.btm_panel, bg=COLORS["surface"])
        slider_frame.pack(fill="x", pady=(0, 20))
        
        self.lbl_raise_amt = tk.Label(slider_frame, text="Raise To: $0", 
                                      fg=COLORS["text"], bg=COLORS["surface"], font=FONTS["label"])
        self.lbl_raise_amt.pack(side="left")
        
        # Slider styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Horizontal.TScale", background=COLORS["surface"], 
                        troughcolor="#444", borderwidth=0)
        
        self.slider = ttk.Scale(slider_frame, from_=0, to=100, 
                                orient="horizontal", command=self._update_slider_label)
        self.slider.pack(side="right", fill="x", expand=True, padx=(15, 0))

        # Buttons
        btn_frame = tk.Frame(self.btm_panel, bg=COLORS["surface"])
        btn_frame.pack(fill="x")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        btn_frame.grid_columnconfigure(2, weight=1)

        self.btn_fold = ModernButton(btn_frame, "FOLD", COLORS["accent_fold"], 
                                command=lambda: self._send_action("fold"))
        self.btn_fold.grid(row=0, column=0, sticky="ew", padx=5)

        self.btn_check_call = ModernButton(btn_frame, "CHECK", COLORS["accent_check"], 
                                command=lambda: self._send_action("call"))
        self.btn_check_call.grid(row=0, column=1, sticky="ew", padx=5)

        self.btn_raise = ModernButton(btn_frame, "RAISE", COLORS["accent_raise"], 
                                      command=lambda: self._send_action("raise"))
        self.btn_raise.grid(row=0, column=2, sticky="ew", padx=5)

    # --- API Methods (Call these from your external script) ---

    def set_status(self, text, color=None):
        """Updates the status text label."""
        fg = color if color else COLORS["text"]
        self.lbl_status.config(text=text, fg=fg)

    def set_pot(self, amount):
        """Updates the displayed pot amount."""
        self.lbl_pot.config(text=f"POT: ${amount}")

    def set_balance(self, amount):
        """Updates the displayed player balance."""
        self.lbl_balance.config(text=f"Balance: ${amount}")

    def set_player_cards(self, cards):
        """
        Displays the player's cards.
        :param cards: List of tuples e.g. [('A', '♠'), ('10', '♥')]
        """
        for widget in self.hand_frame.winfo_children():
            widget.destroy()
        
        for rank, suit in cards:
            CardWidget(self.hand_frame, rank, suit).pack(side="left", padx=5)

    def set_community_cards(self, cards):
        """
        Displays community cards.
        :param cards: List of tuples e.g. [('A', '♠'), ('10', '♥'), ('2', '♦')]
        """
        for widget in self.community_frame.winfo_children():
            widget.destroy()
            
        for rank, suit in cards:
            CardWidget(self.community_frame, rank, suit).pack(side="left", padx=5)

    def update_controls(self, min_raise, max_raise, call_amount):
        """
        Updates button text and slider limits based on game state.
        """
        # Update Check/Call button text
        if call_amount == 0:
            self.btn_check_call.config(text="CHECK")
        else:
            self.btn_check_call.config(text=f"CALL ${call_amount}")

        # Update Slider
        self.slider.config(from_=min_raise, to=max_raise)
        # Reset slider position
        self.slider.set(min_raise)
        self.lbl_raise_amt.config(text=f"Raise To: ${min_raise}")
        self.btn_raise.config(text=f"RAISE ${min_raise}")

    def enable_controls(self, enable=True):
        state = "normal" if enable else "disabled"
        self.btn_fold.config(state=state)
        self.btn_check_call.config(state=state)
        self.btn_raise.config(state=state)

    # --- Internal Helpers ---

    def _update_slider_label(self, val):
        amount = int(float(val))
        self.lbl_raise_amt.config(text=f"Raise To: ${amount}")
        self.btn_raise.config(text=f"RAISE ${amount}")

    def _send_action(self, action_type):
        if self.callback:
            amount = 0
            if action_type == "raise":
                amount = int(self.slider.get())
            elif action_type == "call":
                # We don't calculate call amount here, the controller should know it
                # based on game state. We just signal intent.
                pass
            
            self.callback(action_type, amount)


# --- Demo Controller (How to use the class) ---

import random

class DemoController:
    def __init__(self, root):
        # 1. Initialize the GUI
        self.gui = PokerGUI(root, on_action_callback=self.handle_user_action)
        
        # 2. Initialize Game State
        self.deck = []
        self.pot = 0
        self.balance = 1000
        self.current_bet = 0
        self.community_cards = []
        
        # 3. Start
        self.start_round()

    def create_deck(self):
        suits = ['♠', '♥', '♦', '♣']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [(r, s) for s in suits for r in ranks]
        random.shuffle(deck)
        return deck

    def start_round(self):
        self.deck = self.create_deck()
        self.pot = 50
        self.current_bet = 0
        self.community_cards = []
        
        # Update GUI with initial state
        self.gui.set_status("New Round Started")
        self.gui.set_pot(self.pot)
        self.gui.set_balance(self.balance)
        
        # Deal Player Cards
        player_cards = [self.deck.pop(), self.deck.pop()]
        self.gui.set_player_cards(player_cards)
        self.gui.set_community_cards([]) # Clear community
        
        # Configure Buttons (Min raise 10, Max raise balance, Call 0)
        self.gui.update_controls(min_raise=10, max_raise=self.balance, call_amount=0)
        self.gui.enable_controls(True)

    def handle_user_action(self, action, amount):
        """
        This function is called by the GUI when a user clicks a button.
        """
        print(f"User clicked: {action} with amount {amount}")
        
        if action == "fold":
            self.gui.set_status("Folded! Next hand in 2s...", "#FF4B4B")
            self.gui.enable_controls(False)
            self.gui.after(2000, self.start_round)
            
        elif action == "call":
            # Simulate calling logic
            self.gui.set_status("Check/Call", "#4CAF50")
            self.deal_community_card()
            
        elif action == "raise":
            self.pot += amount
            self.balance -= amount
            self.gui.set_pot(self.pot)
            self.gui.set_balance(self.balance)
            self.gui.set_status(f"Raised to {amount}", "#3D91FF")
            self.deal_community_card()

    def deal_community_card(self):
        # Simple logic to add cards one by one for demo
        if len(self.community_cards) < 5:
            self.community_cards.append(self.deck.pop())
            self.gui.set_community_cards(self.community_cards)
        else:
            self.gui.set_status("Showdown! Restarting...", "#00E676")
            self.gui.enable_controls(False)
            self.gui.after(3000, self.start_round)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.title("Poker Client")
        root.geometry("600x700")
        
        # Center window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_cords = int((screen_width/2) - (600/2))
        y_cords = int((screen_height/2) - (700/2))
        root.geometry(f"600x700+{x_cords}+{y_cords}")

        # Initialize Controller
        app = DemoController(root)
        
        root.mainloop()
        
    except Exception:
        traceback.print_exc()
        input("Error detected. Press Enter to exit...")