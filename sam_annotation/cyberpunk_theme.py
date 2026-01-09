#!/usr/bin/env python3
"""
Cyberpunk Theme for SAM Annotator
Sick aesthetic inspired by classic antivirus software and hacker terminals
"""

import tkinter as tk
from tkinter import ttk
import time
import threading
from typing import Dict, Any

class CyberpunkTheme:
    """Cyberpunk/Hacker aesthetic theme manager"""

    # Base resolution for scaling (the original design target)
    BASE_WIDTH = 1920
    BASE_HEIGHT = 1080

    # Base font sizes (smaller for better fit)
    BASE_FONTS = {
        'title': 7,
        'header': 7,
        'button': 7,
        'label': 7,
        'info': 6,
        'small': 6,
        'large': 8,
        'xlarge': 10,
        'xxlarge': 14,
    }

    # Color palette - neon cyberpunk vibes
    COLORS = {
        # Dark background tones
        'bg_primary': '#0a0a0f',      # Deep dark blue-black
        'bg_secondary': '#141420',     # Slightly lighter dark
        'bg_panel': '#1a1a2e',        # Panel background
        'bg_accent': '#16213e',       # Accent panels
        
        # Neon accent colors
        'neon_cyan': '#00ffff',       # Bright cyan
        'neon_green': '#39ff14',      # Electric green
        'neon_pink': '#ff00ff',       # Hot pink
        'neon_orange': '#ff6600',     # Neon orange
        'neon_blue': '#0080ff',       # Electric blue
        'neon_purple': '#8a2be2',     # Neon purple
        
        # Status colors
        'success': '#00ff41',         # Matrix green
        'warning': '#ffaa00',         # Warning orange
        'error': '#ff0040',           # Error red
        'info': '#00aaff',            # Info blue
        
        # Text colors
        'text_primary': '#ffffff',    # White text
        'text_secondary': '#b0b0b0',  # Light gray
        'text_accent': '#00ffff',     # Cyan text
        'text_dim': '#666666',        # Dim gray
        
        # UI Elements
        'border_glow': '#00ffff',     # Glowing borders
        'button_bg': '#1f2937',       # Button background
        'button_hover': '#374151',    # Button hover
        'canvas_bg': '#000000',       # Canvas background
    }
    
    # ASCII Art and Unicode symbols
    SYMBOLS = {
        'scanner': '◢◣◤◥',
        'processing': '⟐⟑⟒⟓',
        'neural': '⧨⧩⧪⧫',
        'data': '▓▒░',
        'arrow_right': '▶',
        'arrow_left': '◀',
        'diamond': '◆',
        'square': '■',
        'triangle': '▲',
        'warning': '⚠',
        'skull': '☠',
        'lightning': '⚡',
        'atom': '⚛',
        'crosshair': '⊕',
        'target': '◎'
    }
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.animation_active = True
        self.animation_frame = 0

        # Calculate UI scale factor based on screen resolution
        self.scale_factor = self._calculate_scale_factor()
        self._cache_scaled_fonts()

        self.setup_styles()
        self.start_animations()

    def _calculate_scale_factor(self) -> float:
        """Calculate scale factor based on screen resolution"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate scale based on the smaller ratio to fit everything
        width_scale = screen_width / self.BASE_WIDTH
        height_scale = screen_height / self.BASE_HEIGHT

        # Use the smaller scale to ensure everything fits
        scale = min(width_scale, height_scale)

        # Clamp scale between reasonable bounds (0.7 to 2.0)
        # Allows scaling up on larger screens
        scale = max(0.7, min(2.0, scale))

        print(f"[UI Scale] Screen: {screen_width}x{screen_height}, Scale factor: {scale:.2f}")

        return scale

    def _cache_scaled_fonts(self):
        """Pre-calculate scaled font sizes"""
        self.fonts = {}
        for name, base_size in self.BASE_FONTS.items():
            self.fonts[name] = max(7, int(base_size * self.scale_factor))

    def get_font(self, size_name: str, bold: bool = False) -> tuple:
        """Get a scaled font tuple for the given size name"""
        size = self.fonts.get(size_name, self.fonts['label'])
        if bold:
            return ('Consolas', size, 'bold')
        return ('Consolas', size)

    def get_scaled_size(self, base_size: int) -> int:
        """Scale any pixel dimension based on screen resolution"""
        return max(1, int(base_size * self.scale_factor))

    def get_scaled_font_size(self, base_size: int) -> int:
        """Scale a specific font size"""
        return max(7, int(base_size * self.scale_factor))
    
    def setup_styles(self):
        """Setup custom TTK styles for cyberpunk theme"""
        style = ttk.Style()
        
        # Configure main theme
        style.theme_use('clam')  # Base theme to modify
        
        # Configure colors for all widgets
        style.configure('.',
            background=self.COLORS['bg_primary'],
            foreground=self.COLORS['text_primary'],
            fieldbackground=self.COLORS['bg_secondary'],
            bordercolor=self.COLORS['border_glow'],
            focuscolor=self.COLORS['neon_cyan'],
            selectbackground=self.COLORS['neon_cyan'],
            selectforeground=self.COLORS['bg_primary']
        )
        
        # Custom frame styles
        style.configure('Cyber.TFrame',
            background=self.COLORS['bg_panel'],
            relief='flat',
            borderwidth=1
        )
        
        style.configure('Panel.TFrame',
            background=self.COLORS['bg_accent'],
            relief='solid',
            borderwidth=2
        )
        
        # Custom label frame styles
        style.configure('Cyber.TLabelframe',
            background=self.COLORS['bg_panel'],
            foreground=self.COLORS['neon_cyan'],
            bordercolor=self.COLORS['border_glow'],
            relief='groove',
            borderwidth=2
        )
        
        style.configure('Cyber.TLabelframe.Label',
            background=self.COLORS['bg_panel'],
            foreground=self.COLORS['neon_cyan'],
            font=self.get_font('header', bold=True)
        )
        
        # Custom button styles
        style.configure('Cyber.TButton',
            background=self.COLORS['button_bg'],
            foreground=self.COLORS['text_primary'],
            bordercolor=self.COLORS['border_glow'],
            focuscolor=self.COLORS['neon_cyan'],
            font=self.get_font('button', bold=True),
            relief='raised',
            borderwidth=2
        )
        
        style.map('Cyber.TButton',
            background=[('active', self.COLORS['button_hover']),
                       ('pressed', self.COLORS['neon_cyan'])],
            foreground=[('pressed', self.COLORS['bg_primary'])]
        )
        
        # Navigation button styles
        style.configure('Nav.TButton',
            background=self.COLORS['bg_accent'],
            foreground=self.COLORS['neon_green'],
            bordercolor=self.COLORS['neon_green'],
            font=self.get_font('header', bold=True),
            relief='solid',
            borderwidth=2
        )

        # Action button styles
        style.configure('Action.TButton',
            background=self.COLORS['button_bg'],
            foreground=self.COLORS['neon_orange'],
            bordercolor=self.COLORS['neon_orange'],
            font=self.get_font('button', bold=True),
            relief='raised',
            borderwidth=2
        )
        
        # Custom scale styles
        style.configure('Cyber.TScale',
            background=self.COLORS['bg_panel'],
            troughcolor=self.COLORS['bg_secondary'],
            bordercolor=self.COLORS['border_glow'],
            lightcolor=self.COLORS['neon_cyan'],
            darkcolor=self.COLORS['neon_cyan']
        )
        
        # Custom label styles
        style.configure('Title.TLabel',
            background=self.COLORS['bg_primary'],
            foreground=self.COLORS['neon_cyan'],
            font=self.get_font('title', bold=True)
        )

        style.configure('Status.TLabel',
            background=self.COLORS['bg_panel'],
            foreground=self.COLORS['neon_green'],
            font=self.get_font('label')
        )

        style.configure('Info.TLabel',
            background=self.COLORS['bg_panel'],
            foreground=self.COLORS['text_secondary'],
            font=self.get_font('info')
        )
    
    def create_glowing_frame(self, parent, text="", width=300, height=200) -> tk.Frame:
        """Create a frame with glowing border effect"""
        # Scale dimensions based on screen resolution
        scaled_width = self.get_scaled_size(width)
        scaled_height = self.get_scaled_size(height)

        outer_frame = tk.Frame(parent,
            bg=self.COLORS['border_glow'],
            relief='solid',
            bd=1,
            width=scaled_width,
            height=scaled_height
        )

        inner_frame = tk.Frame(outer_frame,
            bg=self.COLORS['bg_panel'],
            relief='flat',
            bd=2
        )
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Prevent frame from shrinking/expanding
        outer_frame.pack_propagate(False)

        if text:
            title_label = tk.Label(inner_frame,
                text=f"▰▰▰ {text.upper()} ▰▰▰",
                bg=self.COLORS['bg_panel'],
                fg=self.COLORS['neon_cyan'],
                font=self.get_font('header', bold=True)
            )
            title_label.pack(pady=(5, 10))

        return inner_frame
    
    def create_cyber_button(self, parent, text, command=None, style_color='neon_cyan') -> tk.Button:
        """Create a cyberpunk-styled button"""
        return tk.Button(parent,
            text=f"▶ {text.upper()}",
            bg=self.COLORS['button_bg'],
            fg=self.COLORS[style_color],
            activebackground=self.COLORS[style_color],
            activeforeground=self.COLORS['bg_primary'],
            relief='raised',
            bd=2,
            font=self.get_font('button', bold=True),
            command=command,
            cursor='hand2'
        )
    
    def create_status_display(self, parent) -> tk.Text:
        """Create a terminal-style status display"""
        # Create outer frame for glow effect
        status_outer = tk.Frame(parent,
            bg=self.COLORS['neon_green'],
            relief='solid',
            bd=1
        )

        # Header
        header = tk.Label(status_outer,
            text="◢◣ NEURAL NETWORK STATUS ◤◥",
            bg=self.COLORS['neon_green'],
            fg=self.COLORS['bg_primary'],
            font=self.get_font('header', bold=True)
        )
        header.pack(fill=tk.X)

        # Status text widget
        status_text = tk.Text(status_outer,
            bg=self.COLORS['bg_primary'],
            fg=self.COLORS['neon_green'],
            insertbackground=self.COLORS['neon_green'],
            selectbackground=self.COLORS['neon_cyan'],
            selectforeground=self.COLORS['bg_primary'],
            font=self.get_font('label'),
            relief='flat',
            wrap=tk.WORD,
            state=tk.NORMAL
        )
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(status_outer, 
            command=status_text.yview,
            bg=self.COLORS['bg_secondary'],
            activebackground=self.COLORS['neon_green'],
            troughcolor=self.COLORS['bg_primary']
        )
        status_text.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        status_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        return status_text, status_outer
    
    def create_stats_panel(self, parent) -> tk.Frame:
        """Create a cyberpunk stats panel"""
        stats_frame = self.create_glowing_frame(parent, "SYSTEM METRICS")
        
        # Create sections
        sections = [
            ("⚡ NEURAL PROCESSING", 'neon_orange'),
            ("◎ TARGET ACQUISITION", 'neon_cyan'), 
            ("▓ MEMORY ALLOCATION", 'neon_purple'),
            ("⚛ QUANTUM STATE", 'neon_pink')
        ]
        
        section_frames = {}
        for title, color in sections:
            section = tk.Frame(stats_frame, bg=self.COLORS['bg_panel'])
            section.pack(fill=tk.X, pady=2)
            
            # Section header
            header = tk.Label(section,
                text=title,
                bg=self.COLORS['bg_panel'],
                fg=self.COLORS[color],
                font=self.get_font('button', bold=True),
                anchor='w'
            )
            header.pack(fill=tk.X)
            
            section_frames[title] = section
        
        return stats_frame, section_frames
    
    def animate_title(self, label, base_text):
        """Animate title with glowing effect"""
        def update_title():
            while self.animation_active:
                frames = [
                    f"◢◣ {base_text} ◤◥",
                    f"▰▰ {base_text} ▰▰", 
                    f"▓▓ {base_text} ▓▓",
                    f"▒▒ {base_text} ▒▒",
                    f"░░ {base_text} ░░",
                    f"▒▒ {base_text} ▒▒",
                    f"▓▓ {base_text} ▓▓",
                    f"▰▰ {base_text} ▰▰"
                ]
                
                for frame in frames:
                    if not self.animation_active:
                        break
                    try:
                        label.config(text=frame)
                        time.sleep(0.5)
                    except:
                        break
        
        thread = threading.Thread(target=update_title, daemon=True)
        thread.start()
    
    def start_animations(self):
        """Start background animations"""
        self.animation_active = True
    
    def stop_animations(self):
        """Stop all animations"""
        self.animation_active = False
    
    def apply_theme_to_widget(self, widget, widget_type='default'):
        """Apply cyberpunk theme to any widget"""
        theme_configs = {
            'frame': {
                'bg': self.COLORS['bg_panel'],
                'relief': 'flat'
            },
            'label': {
                'bg': self.COLORS['bg_panel'],
                'fg': self.COLORS['text_primary'],
                'font': self.get_font('label')
            },
            'button': {
                'bg': self.COLORS['button_bg'],
                'fg': self.COLORS['neon_cyan'],
                'activebackground': self.COLORS['neon_cyan'],
                'activeforeground': self.COLORS['bg_primary'],
                'font': self.get_font('button', bold=True),
                'relief': 'raised',
                'bd': 2
            },
            'canvas': {
                'bg': self.COLORS['canvas_bg'],
                'highlightbackground': self.COLORS['border_glow'],
                'highlightcolor': self.COLORS['neon_cyan'],
                'selectbackground': self.COLORS['neon_cyan']
            }
        }
        
        if widget_type in theme_configs:
            try:
                widget.config(**theme_configs[widget_type])
            except:
                pass  # Some widgets might not support all options
    
    def get_cyber_colors(self):
        """Get the color palette for external use"""
        return self.COLORS
    
    def get_cyber_symbols(self):
        """Get the symbol set for external use"""
        return self.SYMBOLS