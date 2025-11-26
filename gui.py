import pygame
import chess
import sys
import threading
import queue
import time
import textwrap
from datetime import datetime

# Importaciones locales
from ai_core import ChessEngine, NeuralBrain

class ChessGUI:
    def __init__(self, brain, engine):
        self.brain = brain
        self.engine = engine
        self.board = chess.Board()
        
        pygame.init()
        
        # Default window size (Resizable)
        self.WIDTH = 1200
        self.HEIGHT = 800
        
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("PCNN Chess - AlphaZero Lite")
        
        self.WHITE = (240, 217, 181)
        self.BLACK = (181, 136, 99)
        self.HIGHLIGHT = (186, 202, 68)
        self.SIDEBAR_BG = (40, 44, 52)
        self.SIDEBAR_TEXT = (171, 178, 191)
        
        self.player_name = "Player"
        self.player_color = chess.WHITE
        self.logs = [] # Raw logs
        self.wrapped_logs = [] # (text, color) tuples for rendering
        self.scroll_offset = 0
        self.loading_text = "Iniciando..."
        self.loading_progress = 0.0
        
        self.COMMENTARY_MODE = True
        
        # Threading for AI
        self.ai_thinking = False
        self.move_queue = queue.Queue()
        self.ai_thread = None
        
        # Initial layout calculation
        self.recalculate_layout(self.WIDTH, self.HEIGHT)
        
        # Connect logger callback
        self.engine.logger.gui_callback = self.log_message

    def recalculate_layout(self, w, h):
        self.WIDTH = w
        self.HEIGHT = h
        
        # Sidebar takes 35% of width, but at least 350px
        self.SIDEBAR_WIDTH = max(350, int(self.WIDTH * 0.35))
        
        # Commentary Panel Height
        self.COMMENTARY_HEIGHT = 120
        
        # Available area for board (Right side, Top part)
        self.BOARD_AREA_WIDTH = self.WIDTH - self.SIDEBAR_WIDTH
        self.BOARD_AREA_HEIGHT = self.HEIGHT - self.COMMENTARY_HEIGHT
        
        # Board size
        board_size = min(self.BOARD_AREA_WIDTH, self.BOARD_AREA_HEIGHT) - 20
        self.SQ_SIZE = max(10, board_size // 8)
        
        # Centering the board
        self.BOARD_OFFSET_X = self.SIDEBAR_WIDTH + (self.BOARD_AREA_WIDTH - (self.SQ_SIZE * 8)) // 2
        self.BOARD_OFFSET_Y = (self.BOARD_AREA_HEIGHT - (self.SQ_SIZE * 8)) // 2
        
        # Update fonts
        self.font = pygame.font.SysFont("segoe ui symbol", int(self.SQ_SIZE * 0.8))
        self.ui_font = pygame.font.SysFont("arial", max(16, int(self.HEIGHT * 0.025)))
        self.log_font = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("arial", max(30, int(self.HEIGHT * 0.06)), bold=True)
        
        self.rewrap_logs()

    def rewrap_logs(self):
        self.wrapped_logs = []
        for log in self.logs:
            self._wrap_and_append(log)
            
    def _wrap_and_append(self, log):
        words = log.split(' ')
        current_line = []
        
        color = self.SIDEBAR_TEXT
        if "RAZONAMIENTO" in log: color = (255, 200, 100)
        elif "PREDICTIVE CODING" in log or "Sorpresa" in log: color = (255, 100, 100)
        elif "DECISIÓN FINAL" in log: color = (100, 255, 100)
        elif "ANÁLISIS" in log or "analyzing" in log.lower(): color = (100, 200, 255)
        
        max_width = self.SIDEBAR_WIDTH - 50
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if self.log_font.size(test_line)[0] < max_width:
                current_line.append(word)
            else:
                self.wrapped_logs.append(( ' '.join(current_line), color ))
                current_line = [word]
        self.wrapped_logs.append(( ' '.join(current_line), color ))

    def log_message(self, message):
        self.logs.append(message)
        if len(self.logs) > 500: 
            self.logs.pop(0)
            self.rewrap_logs()
        else:
            self._wrap_and_append(message)
            
        line_height = self.log_font.get_linesize()
        total_height = len(self.wrapped_logs) * line_height
        log_area_height = self.HEIGHT - 160
        
        if total_height > log_area_height:
            self.scroll_offset = total_height - log_area_height
        
        # Force redraw if running (thread safe check needed? Pygame isn't thread safe)
        # We rely on main loop to redraw

    def update_loading(self, text, progress):
        self.loading_text = text
        self.loading_progress = min(1.0, max(0.0, progress))

    def draw_loading_screen(self):
        self.screen.fill(self.SIDEBAR_BG)
        title = self.title_font.render("PCNN Chess", True, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(self.WIDTH//2, self.HEIGHT//3)))
        txt = self.ui_font.render(self.loading_text, True, (200, 200, 200))
        self.screen.blit(txt, txt.get_rect(center=(self.WIDTH//2, self.HEIGHT//2)))
        
        bar_w, bar_h = 400, 20
        bar_x, bar_y = (self.WIDTH - bar_w)//2, self.HEIGHT//2 + 40
        pygame.draw.rect(self.screen, (30, 30, 30), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, (97, 175, 239), (bar_x, bar_y, int(bar_w * self.loading_progress), bar_h))
        pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), 2)
        pygame.display.flip()

    def draw_sidebar(self):
        pygame.draw.rect(self.screen, self.SIDEBAR_BG, (0, 0, self.SIDEBAR_WIDTH, self.HEIGHT))
        
        title_text = "PCNN Chess"
        title_surf = self.title_font.render(title_text, True, (255, 255, 255))
        if title_surf.get_width() > self.SIDEBAR_WIDTH - 40:
            scale = (self.SIDEBAR_WIDTH - 40) / title_surf.get_width()
            new_size = (int(title_surf.get_width() * scale), int(title_surf.get_height() * scale))
            title_surf = pygame.transform.smoothscale(title_surf, new_size)
        self.screen.blit(title_surf, (20, 20))
        
        subtitle = self.ui_font.render("AlphaZero Lite - XAI Mode", True, (100, 200, 255))
        self.screen.blit(subtitle, (20, 80))
        
        pygame.draw.line(self.screen, (100, 100, 100), (20, 120), (self.SIDEBAR_WIDTH - 20, 120), 2)
        
        # Botón de Rendirse
        resign_btn = pygame.Rect(20, 130, 100, 30)
        mouse_pos = pygame.mouse.get_pos()
        btn_color = (180, 60, 60) if resign_btn.collidepoint(mouse_pos) else (150, 50, 50)
        pygame.draw.rect(self.screen, btn_color, resign_btn, border_radius=5)
        
        resign_txt = self.ui_font.render("Rendirse", True, (255, 255, 255))
        if resign_txt.get_width() > 90:
            scale = 90 / resign_txt.get_width()
            resign_txt = pygame.transform.smoothscale(resign_txt, (int(resign_txt.get_width()*scale), int(resign_txt.get_height()*scale)))
        self.screen.blit(resign_txt, resign_txt.get_rect(center=resign_btn.center))
        
        log_start_y = 170
        log_area_height = self.HEIGHT - log_start_y - 20
        line_height = self.log_font.get_linesize()
        
        total_lines = len(self.wrapped_logs)
        total_height = total_lines * line_height
        
        max_scroll = max(0, total_height - log_area_height)
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))
        
        start_index = int(self.scroll_offset // line_height)
        end_index = start_index + int(log_area_height // line_height) + 2
        
        visible_lines = self.wrapped_logs[start_index:end_index]
        y = log_start_y - (self.scroll_offset % line_height)
        
        old_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(0, log_start_y, self.SIDEBAR_WIDTH, log_area_height))
        
        for line_text, line_color in visible_lines:
            surf = self.log_font.render(line_text, True, line_color)
            self.screen.blit(surf, (20, y))
            y += line_height
            
        self.screen.set_clip(old_clip)
        
        if total_height > log_area_height:
            bar_bg_rect = pygame.Rect(self.SIDEBAR_WIDTH - 15, log_start_y, 10, log_area_height)
            pygame.draw.rect(self.screen, (30, 30, 30), bar_bg_rect, border_radius=5)
            
            scroll_ratio = log_area_height / total_height
            thumb_height = max(20, int(log_area_height * scroll_ratio))
            thumb_pos_ratio = self.scroll_offset / max_scroll
            thumb_y = log_start_y + int(thumb_pos_ratio * (log_area_height - thumb_height))
            
            thumb_rect = pygame.Rect(self.SIDEBAR_WIDTH - 15, thumb_y, 10, thumb_height)
            pygame.draw.rect(self.screen, (100, 100, 100), thumb_rect, border_radius=5)

    def draw_board(self):
        # Clear board area
        pygame.draw.rect(self.screen, (30, 30, 30), (self.SIDEBAR_WIDTH, 0, self.BOARD_AREA_WIDTH, self.BOARD_AREA_HEIGHT))
        
        king_sq = None
        if self.board.is_check(): king_sq = self.board.king(self.board.turn)
        
        is_flipped = (self.player_color == chess.BLACK)
        
        for r in range(8):
            for c in range(8):
                color = self.WHITE if (r + c) % 2 == 0 else self.BLACK
                
                if is_flipped:
                    file_idx = 7 - c
                    rank_idx = r
                else:
                    file_idx = c
                    rank_idx = 7 - r
                sq = chess.square(file_idx, rank_idx)
                
                if self.selected_square == sq:
                    color = self.HIGHLIGHT
                if king_sq == sq:
                    color = (255, 80, 80)
                    
                pygame.draw.rect(self.screen, color, (self.BOARD_OFFSET_X + c * self.SQ_SIZE, self.BOARD_OFFSET_Y + r * self.SQ_SIZE, self.SQ_SIZE, self.SQ_SIZE))

    def draw_pieces(self):
        pieces = {'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚', 'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'}
        is_flipped = (self.player_color == chess.BLACK)
        
        for sq in chess.SQUARES:
            p = self.board.piece_at(sq)
            if p:
                s = pieces[p.symbol()]
                c = (255, 255, 255) if p.color == chess.WHITE else (0, 0, 0)
                txt = self.font.render(s, True, c)
                
                f = chess.square_file(sq)
                rk = chess.square_rank(sq)
                
                if is_flipped:
                    col_idx = 7 - f
                    row_idx = rk
                else:
                    col_idx = f
                    row_idx = 7 - rk
                
                rect = txt.get_rect(center=(self.BOARD_OFFSET_X + col_idx*self.SQ_SIZE + self.SQ_SIZE//2, self.BOARD_OFFSET_Y + row_idx*self.SQ_SIZE + self.SQ_SIZE//2))
                self.screen.blit(txt, rect)

    def handle_click(self, pos):
        if pos[0] < self.SIDEBAR_WIDTH: return False
        
        board_rect = pygame.Rect(self.BOARD_OFFSET_X, self.BOARD_OFFSET_Y, self.SQ_SIZE*8, self.SQ_SIZE*8)
        if not board_rect.collidepoint(pos): return False
        
        col = (pos[0] - self.BOARD_OFFSET_X) // self.SQ_SIZE
        row = (pos[1] - self.BOARD_OFFSET_Y) // self.SQ_SIZE
        
        is_flipped = (self.player_color == chess.BLACK)
        if is_flipped:
            file_idx = 7 - col
            rank_idx = row
        else:
            file_idx = col
            rank_idx = 7 - row
            
        sq = chess.square(file_idx, rank_idx)
        
        if self.selected_square is None:
            p = self.board.piece_at(sq)
            if p and p.color == self.board.turn:
                self.selected_square = sq
                return True
        else:
            move = chess.Move(self.selected_square, sq)
            # Promotion check (auto-queen)
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                if (self.board.turn == chess.WHITE and chess.square_rank(sq) == 7) or \
                   (self.board.turn == chess.BLACK and chess.square_rank(sq) == 0):
                    move = chess.Move(self.selected_square, sq, promotion=chess.QUEEN)
            
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                return True
            else:
                # Deselect or select other piece
                p = self.board.piece_at(sq)
                if p and p.color == self.board.turn:
                    self.selected_square = sq
                    return True
                self.selected_square = None
        return False

    def input_name_screen(self):
        input_box = pygame.Rect(self.WIDTH//2 - 100, self.HEIGHT//2, 200, 32)
        color_inactive = pygame.Color('lightskyblue3')
        color_active = pygame.Color('dodgerblue2')
        color = color_inactive
        active = False
        text = self.player_name
        done = False
        
        selected_color = chess.WHITE
        white_btn = pygame.Rect(self.WIDTH//2 - 110, self.HEIGHT//2 + 50, 100, 40)
        black_btn = pygame.Rect(self.WIDTH//2 + 10, self.HEIGHT//2 + 50, 100, 40)
        start_btn = pygame.Rect(self.WIDTH//2 - 50, self.HEIGHT//2 + 110, 100, 40)
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if input_box.collidepoint(event.pos): active = not active
                    else: active = False
                    color = color_active if active else color_inactive
                    
                    if white_btn.collidepoint(event.pos): selected_color = chess.WHITE
                    if black_btn.collidepoint(event.pos): selected_color = chess.BLACK
                    if start_btn.collidepoint(event.pos):
                        if text.strip():
                            self.player_name = text
                            self.player_color = selected_color
                            done = True
                if event.type == pygame.KEYDOWN:
                    if active:
                        if event.key == pygame.K_RETURN:
                            if text.strip():
                                self.player_name = text
                                self.player_color = selected_color
                                done = True
                        elif event.key == pygame.K_BACKSPACE: text = text[:-1]
                        else: text += event.unicode

            self.screen.fill((30, 30, 30))
            title = self.title_font.render("PCNN Chess", True, (255, 255, 255))
            self.screen.blit(title, title.get_rect(center=(self.WIDTH//2, self.HEIGHT//3)))
            
            instr = self.ui_font.render("Nombre y Color:", True, (200, 200, 200))
            self.screen.blit(instr, instr.get_rect(center=(self.WIDTH//2, self.HEIGHT//2 - 50)))
            
            txt_surf = self.ui_font.render(text, True, color)
            input_box.w = max(200, txt_surf.get_width()+10)
            input_box.centerx = self.WIDTH // 2
            self.screen.blit(txt_surf, (input_box.x+5, input_box.y+5))
            pygame.draw.rect(self.screen, color, input_box, 2)
            
            c_white = (100, 200, 100) if selected_color == chess.WHITE else (100, 100, 100)
            c_black = (100, 200, 100) if selected_color == chess.BLACK else (100, 100, 100)
            
            pygame.draw.rect(self.screen, c_white, white_btn, border_radius=5)
            pygame.draw.rect(self.screen, c_black, black_btn, border_radius=5)
            
            w_txt = self.ui_font.render("Blancas", True, (255, 255, 255))
            b_txt = self.ui_font.render("Negras", True, (255, 255, 255))
            self.screen.blit(w_txt, w_txt.get_rect(center=white_btn.center))
            self.screen.blit(b_txt, b_txt.get_rect(center=black_btn.center))
            
            btn_color = (0, 200, 0) if (text.strip() and selected_color is not None) else (50, 50, 50)
            pygame.draw.rect(self.screen, btn_color, start_btn, border_radius=5)
            btn_txt = self.ui_font.render("Iniciar", True, (255, 255, 255))
            self.screen.blit(btn_txt, btn_txt.get_rect(center=start_btn.center))
            
            pygame.display.flip()

    def show_game_over(self, result):
        winner = None
        if result == "1-0": winner = chess.WHITE
        elif result == "0-1": winner = chess.BLACK
        
        msg = "Tablas / Empate"
        if winner is not None:
            if winner == self.player_color:
                msg = f"¡Increíble {self.player_name}! Tu cerebro orgánico ha vencido."
            else:
                msg = f"¡Jaque Mate {self.player_name}! La IA ha dominado esta partida."
        
        # Metrics
        metrics = self.brain.learn_from_game(self.game_history, 1.0 if winner == chess.WHITE else -1.0 if winner == chess.BLACK else 0.0)
        
        logs_saved = False
        
        # Snapshot of the current screen (board + logs)
        background_snapshot = self.screen.copy()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Buttons
                    mx, my = event.pos
                    cx, cy = self.WIDTH//2, self.HEIGHT//2
                    if pygame.Rect(cx-100, cy+50, 200, 40).collidepoint((mx, my)):
                        return "restart"
                    if pygame.Rect(cx-100, cy+100, 200, 40).collidepoint((mx, my)):
                        logs_saved = self.save_logs()

            # Restore background
            self.screen.blit(background_snapshot, (0, 0))

            # Draw overlay
            s = pygame.Surface((self.WIDTH, self.HEIGHT))
            s.set_alpha(200)
            s.fill((0,0,0))
            self.screen.blit(s, (0,0))
            
            game_over_font = pygame.font.SysFont("arial", 40, bold=True)
            lines = textwrap.wrap(msg, width=40)
            y_off = self.HEIGHT//2 - 100
            for line in lines:
                base_color = (100, 255, 100) if winner == self.player_color else (255, 100, 100)
                if winner is None: base_color = (200, 200, 255)
                txt = game_over_font.render(line, True, base_color)
                self.screen.blit(txt, txt.get_rect(center=(self.WIDTH//2, y_off)))
                y_off += 50
                
            # Buttons
            cx, cy = self.WIDTH//2, self.HEIGHT//2
            pygame.draw.rect(self.screen, (50, 150, 50), (cx-100, cy+50, 200, 40), border_radius=5)
            rtxt = self.ui_font.render("Jugar de Nuevo", True, (255, 255, 255))
            self.screen.blit(rtxt, rtxt.get_rect(center=(cx, cy+70)))
            
            pygame.draw.rect(self.screen, (50, 50, 150), (cx-100, cy+100, 200, 40), border_radius=5)
            ltxt = self.ui_font.render("Guardar Logs", True, (255, 255, 255))
            self.screen.blit(ltxt, ltxt.get_rect(center=(cx, cy+120)))
            
            if logs_saved:
                logs_saved_msg = f"Logs guardados en logs/"
                saved_txt = self.ui_font.render(logs_saved_msg, True, (100, 255, 100))
                self.screen.blit(saved_txt, saved_txt.get_rect(center=(cx, cy+160)))
            
            pygame.display.flip()

    def save_logs(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"logs/game_logs_{ts}.txt"
        try:
            with open(fn, "w", encoding="utf-8") as f:
                for l in self.logs: f.write(l + "\n")
            return True
        except: return False

    def draw_commentary_overlay(self, text):
        if not text or not self.COMMENTARY_MODE:
            return
            
        panel_x = self.SIDEBAR_WIDTH
        panel_y = self.HEIGHT - self.COMMENTARY_HEIGHT
        panel_w = self.WIDTH - self.SIDEBAR_WIDTH
        panel_h = self.COMMENTARY_HEIGHT
        
        pygame.draw.rect(self.screen, (25, 25, 30), (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.line(self.screen, (60, 60, 70), (panel_x, panel_y), (self.WIDTH, panel_y), 2)
        
        title_font = pygame.font.SysFont("arial", 14, bold=True)
        title = title_font.render("COMENTARIOS GM", True, (150, 150, 160))
        self.screen.blit(title, (panel_x + 20, panel_y + 10))
        
        font = pygame.font.SysFont("georgia", 18, italic=True)
        wrapped = textwrap.fill(text, width=int(panel_w / 10))
        lines = wrapped.splitlines()
        
        y_offset = panel_y + 35
        for line in lines[:3]:
            txt = font.render(line, True, (230, 230, 210))
            self.screen.blit(txt, (panel_x + 20, y_offset))
            y_offset += 24

    def run(self):
        training_thread = threading.Thread(target=self.brain.train_from_kaggle, args=(self.update_loading,))
        training_thread.start()
        while training_thread.is_alive():
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            self.draw_loading_screen(); pygame.time.wait(50)

        while True:
            self.input_name_screen()
            self.board = chess.Board()
            self.game_history = []
            self.selected_square = None
            self.running = True
            self.logs = []
            self.wrapped_logs = []
            self.log_message("Motor AlphaZero Lite Iniciado.")
            
            clock = pygame.time.Clock()
            restart = False
            
            while self.running:
                if self.board.is_game_over():
                    res_val = 0.0
                    if self.board.result() == "1-0": res_val = 1.0
                    elif self.board.result() == "0-1": res_val = -1.0
                    
                    self.log_message("--- Aprendiendo (TD-Lambda) ---")
                    # Note: Learning is done in show_game_over now to avoid freeze before screen
                    
                    if self.show_game_over(self.board.result()) == "restart":
                        restart = True; break
                
                # Event Handling
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT: self.running = False; pygame.quit(); sys.exit()
                    elif event.type == pygame.VIDEORESIZE:
                        self.recalculate_layout(event.w, event.h)
                    elif event.type == pygame.MOUSEWHEEL:
                        if pygame.mouse.get_pos()[0] < self.SIDEBAR_WIDTH:
                            self.scroll_offset -= event.y * 30
                            self.draw_sidebar()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.running = False; pygame.quit(); sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # Check Resign Button (Global)
                        if pygame.Rect(20, 130, 100, 30).collidepoint(event.pos):
                            self.engine.logger.log("DECISIÓN FINAL", "Jugador se rinde.")
                            
                            # Stop AI if thinking
                            if self.ai_thinking:
                                self.engine.stop_search = True
                                # Wait briefly for thread? No, just proceed.
                                # The thread will finish and put something in queue, or we ignore it.
                            
                            res = "0-1" if self.player_color == chess.WHITE else "1-0"
                            if self.show_game_over(res) == "restart":
                                restart = True
                            self.running = False

                # 2. MEJORA: Multithreading para la IA
                if self.board.turn != self.player_color and not self.board.is_game_over():
                    if not self.ai_thinking:
                        self.ai_thinking = True
                        # Start thread
                        self.ai_thread = threading.Thread(target=self._ai_worker, args=(self.board.copy(),))
                        self.ai_thread.start()
                    
                    # Check for result
                    try:
                        best_move = self.move_queue.get_nowait()
                        if best_move:
                            self.board.push(best_move)
                            self.game_history.append(self.board.copy())
                        else:
                            self.engine.logger.log("DECISIÓN FINAL", "IA se rinde.")
                            res = "1-0" if self.player_color == chess.WHITE else "0-1"
                            if self.show_game_over(res) == "restart":
                                restart = True
                            self.running = False
                        self.ai_thinking = False
                    except queue.Empty:
                        pass # Still thinking
                
                # Human Turn
                elif self.board.turn == self.player_color:
                    for event in events:
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if self.handle_click(event.pos):
                                self.game_history.append(self.board.copy())
                                self.draw_sidebar(); self.draw_board(); self.draw_pieces()
                                pygame.display.flip()

                self.draw_sidebar()
                self.draw_board()
                self.draw_pieces()
                self.draw_commentary_overlay(self.engine.latest_commentary)
                
                # Draw thinking indicator
                if self.ai_thinking:
                    # Small spinner or text
                    txt = self.ui_font.render("Pensando...", True, (100, 255, 100))
                    self.screen.blit(txt, (self.WIDTH - 120, 20))
                
                pygame.display.flip()
                clock.tick(30)

    def _ai_worker(self, board_copy):
        """Worker thread for AI calculation"""
        move = self.engine.get_best_move(board_copy)
        self.move_queue.put(move)
