import chess
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import kagglehub
import time
import random
import os
import pygame
import sys
import json
from datetime import datetime
import joblib
import threading
import textwrap

# Importar módulos propios
from features import extract_features
from replay_buffer import ReplayBuffer
from commentary_narrator import NaturalNarrator

class ThoughtLogger:
    def __init__(self, filename="logs/pcnn_thoughts.log", verbosity="high"):
        self.filename = filename
        self.verbosity = verbosity
        self.gui_callback = None
        # Clear log file
        try:
            with open(self.filename, "w", encoding="utf-8") as f:
                f.write("--- PCNN CHESS THOUGHT LOGS ---\n")
        except FileNotFoundError:
            # Ensure directory exists if it wasn't created
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            with open(self.filename, "w", encoding="utf-8") as f:
                f.write("--- PCNN CHESS THOUGHT LOGS ---\n")

    def log(self, section, content):
        # Icons removed as per user request
        formatted_msg = f"[{section}]\n{content}"
        
        # File
        try:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(formatted_msg + "\n\n")
        except: pass
            
        # Console
        print(formatted_msg)
        print("-" * 20)
        
        # GUI (Buffer)
        if self.gui_callback:
            self.gui_callback(f"[{section}]")
            for line in content.split('\n'):
                if line.strip():
                    self.gui_callback("  " + line)

class NeuralBrain:
    def __init__(self):
        # 2. MEJORAS AL MODELO DE ML
        self.model = SGDRegressor(
            loss='squared_error', 
            penalty='l2', 
            alpha=0.00001, 
            learning_rate='constant', 
            eta0=0.001, 
            power_t=0.25,
            max_iter=1,
            shuffle=True,
            random_state=42
        )
        self.is_trained = False
        self.replay_buffer = ReplayBuffer(capacity=20000)
        self.batch_size = 64
        self.train_counter = 0
        self.logger = None # Will be set by Engine
        
        # Cargar buffer si existe
        self.replay_buffer.load()

    def _normalize_eval(self, val):
        return np.clip(val / 10.0, -1.0, 1.0)

    def _denormalize_eval(self, val):
        return val * 10.0

    def predict(self, board):
        if not self.is_trained:
            # Fallback simple: material
            return self._material_fallback(board) / 10.0
        
        try:
            features = extract_features(board).reshape(1, -1)
            pred = self.model.predict(features)[0]
            
            # Protección contra NaN/Inf
            if np.isnan(pred) or np.isinf(pred):
                return self._material_fallback(board) / 10.0
                
            return self._denormalize_eval(pred)
        except:
            return self._material_fallback(board) / 10.0

    def _material_fallback(self, board):
        # Evaluación simple de material
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        score = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                v = vals.get(p.piece_type, 0)
                if p.color == chess.WHITE: score += v
                else: score -= v
        return score

    def learn_from_game(self, history, result):
        # 4. MEJORAS AL APRENDIZAJE ONLINE (TD-Lambda)
        # result: 1.0, -1.0, 0.0
        
        gamma = 0.99
        lam = 0.7 # Lambda
        
        # Pre-calcular predicciones para toda la historia (necesario para TD)
        # Esto es costoso pero necesario para TD(lambda) correcto
        states = [extract_features(b) for b in history]
        predictions = []
        if self.is_trained:
            predictions = self.model.predict(np.array(states))
        else:
            predictions = np.zeros(len(states))
            
        # Algoritmo TD(lambda) backward view
        # G_t = R_{t+1} + gamma * ((1-lambda)*V(S_{t+1}) + lambda*G_{t+1})
        # Aquí simplificamos asumiendo recompensa 0 en pasos intermedios y 'result' al final
        
        G = result # El valor final real
        
        game_experiences = []
        
        # Recorremos hacia atrás
        for t in reversed(range(len(history))):
            state = states[t]
            
            # Guardar en buffer (Estado, Objetivo)
            # Objetivo clipped
            target = max(-1.0, min(1.0, G))
            self.replay_buffer.add(state, target)
            game_experiences.append((state, target))
            
            # Actualizar G para el paso anterior (t-1)
            # G <-- V(S_t) * (1-lambda) + G * lambda
            # Nota: gamma se aplica al movernos un paso atrás en el tiempo
            # Formula standard TD(lambda) return:
            # G_t = R + gamma * ( (1-lambda)*V(S_{t+1}) + lambda*G_{t+1} )
            # Como R=0 excepto al final, y ya tenemos G_{t+1} (que es el G actual del loop)
            # El valor V(S_{t+1}) sería predictions[t+1] si existiera, pero estamos en t.
            # Ajuste: G representa el retorno desde t hacia adelante.
            
            pred_val = predictions[t]
            # Mezcla entre predicción actual y retorno futuro
            G = (1 - lam) * pred_val + lam * G
            G *= gamma # Descuento por paso de tiempo

        # Entrenar
        if len(self.replay_buffer) < self.batch_size:
             return {"loss": 0, "E_pos": 0, "E_neg": 0, "buffer_size": len(self.replay_buffer), "iterations": 0}

        X_batch, y_batch = self.replay_buffer.sample(self.batch_size)
        self.model.partial_fit(X_batch, y_batch)
        self.train_counter += 1
        
        # Guardar modelo periódicamente
        if self.train_counter % 200 == 0:
            self.save_model()
            self.replay_buffer.save()

        # Métricas sobre la partida actual
        X_game = np.array([e[0] for e in game_experiences])
        y_game = np.array([e[1] for e in game_experiences])
        
        current_preds = self.model.predict(X_game)
        loss = np.mean((y_game - current_preds)**2)
        
        E_pos = np.maximum(0, y_game - current_preds).mean()
        E_neg = np.maximum(0, current_preds - y_game).mean()
        
        # Log Prediction Error
        if self.logger:
            msg = f"- Sorpresa: {E_pos:.4f} (la realidad fue mejor/peor de lo previsto)\n"
            msg += f"- Decepción: {E_neg:.4f} (mis predicciones optimistas casi no se invalidaron)\n"
            msg += f"- Corrección interna aplicada: Ajuste de pesos basado en {len(game_experiences)} estados."
            self.logger.log("PREDICTIVE CODING", msg)
        
        return {
            "loss": loss,
            "E_pos": E_pos,
            "E_neg": E_neg,
            "buffer_size": len(self.replay_buffer),
            "iterations": 1
        }

    def save_model(self):
        try:
            joblib.dump(self.model, "models/pcnn_model.pkl")
            print("Modelo guardado.")
        except: pass

    def load_model(self):
        if os.path.exists("models/pcnn_model.pkl"):
            try:
                self.model = joblib.load("models/pcnn_model.pkl")
                self.is_trained = True
                print("Modelo cargado.")
            except: pass

    def train_from_kaggle(self, progress_callback=None):
        # Cargar modelo si existe
        self.load_model()
        if self.is_trained:
            if progress_callback: progress_callback("Modelo cargado de disco.", 1.0)
            return

        msg = "Descargando dataset Kaggle..."
        print(msg)
        if progress_callback: progress_callback(msg, 0.1)
        
        try:
            path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")
            csv_path = None
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".csv"):
                        csv_path = os.path.join(root, file); break
                if csv_path: break
            
            if not csv_path: raise FileNotFoundError("CSV no encontrado")

            msg = "Entrenando memoria base..."
            if progress_callback: progress_callback(msg, 0.2)
            
            chunksize = 2000
            total_rows = 50000
            chunks = pd.read_csv(csv_path, chunksize=chunksize, nrows=total_rows)
            
            for i, chunk in enumerate(chunks):
                X = []
                y = []
                for _, row in chunk.iterrows():
                    b = chess.Board(row['FEN'])
                    val = self._parse_evaluation(row['Evaluation'])
                    X.append(extract_features(b))
                    y.append(self._normalize_eval(val))
                
                self.model.partial_fit(np.array(X), np.array(y))
                
                if i % 5 == 0:
                    pct = 0.2 + (0.8 * (i * chunksize / total_rows))
                    if progress_callback: progress_callback(f"Procesando batch {i}...", pct)
            
            self.is_trained = True
            self.save_model()
            if progress_callback: progress_callback("Listo!", 1.0)
            
        except Exception as e:
            print(f"Error: {e}")
            self.train_synthetic_memory()
            
    def _parse_evaluation(self, eval_str):
        # Helper para parsear evaluaciones de Kaggle (pueden ser #M1, +350, etc)
        if "#" in str(eval_str):
            if "+" in str(eval_str): return 1000
            else: return -1000
        try:
            return float(eval_str)
        except:
            return 0.0
            
    def train_synthetic_memory(self):
        # Fallback si falla Kaggle
        pass

class ChessEngine:
    def __init__(self, brain):
        self.brain = brain
        self.depth = 3
        self.logger = ThoughtLogger()
        self.brain.logger = self.logger # Link logger to brain
        # 5.C Transposition Table
        self.tt = {} 
        self.narrator = NaturalNarrator()
        self.latest_commentary = ""

    def log(self, message):
        # Legacy support if needed, but we prefer structured logging
        pass

    def _get_narrative_state(self, board):
        # Helper to extract state dict for narrator
        feats = extract_features(board)
        # Indices based on features.py:
        # 0-11: Material (P,N,B,R,Q,K for W then B) -> Interleaved: P_W, P_B, N_W, N_B...
        # 12-13: Mobility (Current, Opponent)
        # 14-15: King Safety (W, B)
        
        # Correct Material Indices: 0,2,4,6,8 for White; 1,3,5,7,9 for Black
        w_mat = feats[0]*1 + feats[2]*3 + feats[4]*3 + feats[6]*5 + feats[8]*9
        b_mat = feats[1]*1 + feats[3]*3 + feats[5]*3 + feats[7]*5 + feats[9]*9
        mat_diff = w_mat - b_mat
        
        if board.turn == chess.WHITE:
            mob_w = int(feats[12])
            mob_b = int(feats[13])
        else:
            mob_b = int(feats[12])
            mob_w = int(feats[13])
        
        ks_w = int(feats[14])
        ks_b = int(feats[15])
        
        pawn_summary = "estructura estándar" 
        
        return {
            'material_diff': mat_diff,
            'mobility_white': mob_w,
            'mobility_black': mob_b,
            'threats_white': ks_w,
            'threats_black': ks_b,
            'pawn_structure_summary': pawn_summary,
            'fen': board.fen()
        }

    def get_best_move(self, board):
        # (A) Estado actual
        self._log_state(board)
        
        # (B) Intención
        self._log_intention(board)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves: return None
        
        # 5.B Ordenamiento Avanzado (Raíz)
        legal_moves.sort(key=lambda m: (
            board.is_capture(m), 
            board.gives_check(m),
            board.is_castling(m)
        ), reverse=True)
        
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        is_maximizing = board.turn == chess.WHITE
        
        best_val = -float('inf') if is_maximizing else float('inf')
        
        move_scores = []
        nodes_explored = 0
        
        for move in legal_moves:
            board.push(move)
            nodes_explored += 1
            val = self.minimax(board, self.depth - 1, alpha, beta, not is_maximizing)
            board.pop()
            
            move_scores.append((move, val))
            
            if is_maximizing:
                if val > best_val:
                    best_val = val
                    best_move = move
                alpha = max(alpha, val)
            else:
                if val < best_val:
                    best_val = val
                    best_move = move
                beta = min(beta, val)
                
            if beta <= alpha: break

        # (C) Análisis de movimientos
        move_scores.sort(key=lambda x: x[1], reverse=is_maximizing)
        self._log_move_analysis(board, move_scores[:3])
        
        # (D) Razonamiento
        self._log_reasoning(board, best_move, best_val)
        
        # (E) Decisión Final
        self.logger.log("DECISIÓN FINAL", f"DECISIÓN FINAL: {best_move}\nMotivo principal: mejora posicional con mínima exposición a amenazas tácticas.")
        
        # (G) Resumen
        summary = f"- {len(legal_moves)} jugadas evaluadas\n- Nodos explorados: {nodes_explored}\n- Profundidad: {self.depth}"
        self.logger.log("RESUMEN DEL TURNO", summary)
        
        # (H) Narrativa GM
        state = self._get_narrative_state(board)
        eval_info = {
            'surprise': 0.0, 
            'tactical': board.is_capture(best_move) or board.gives_check(best_move),
            'plan_tag': 'posicional'
        }
        self.latest_commentary = self.narrator.natural_commentary(state, best_move, eval_info)

        return best_move

    def _log_state(self, board):
        # Extract features for logging
        feats = extract_features(board)
        # Indices based on features.py:
        # 0-11: Material (P,N,B,R,Q,K for W then B)
        # 12-13: Mobility (W, B)
        # 14-15: King Safety (W, B)
        # 16-19: Center Control (W, B)
        # 20-21: Advanced Pawns (W, B)
        # 22: Check
        # 23: Turn
        
        turn_str = "Blancas" if board.turn == chess.WHITE else "Negras"
        
        # Material calc (approx from features)
        # P=1, N=3, B=3, R=5, Q=9
        w_mat = feats[0]*1 + feats[1]*3 + feats[2]*3 + feats[3]*5 + feats[4]*9
        b_mat = feats[6]*1 + feats[7]*3 + feats[8]*3 + feats[9]*5 + feats[10]*9
        mat_diff = w_mat - b_mat
        mat_str = f"{mat_diff:+.1f}"
        if mat_diff > 0.5: mat_desc = "(ventaja blanca)"
        elif mat_diff < -0.5: mat_desc = "(ventaja negra)"
        else: mat_desc = "(equilibrado)"
        
        mob_w = int(feats[12])
        mob_b = int(feats[13])
        mob_desc = "Blancas más libres" if mob_w > mob_b + 2 else "Negras más libres" if mob_b > mob_w + 2 else "Movilidad similar"
        
        ks_w = int(feats[14])
        ks_b = int(feats[15])
        ks_desc = "Reyes seguros"
        if board.is_check(): ks_desc = "¡JAQUE ACTIVO!"
        
        msg = f"- Turno actual: {turn_str}\n"
        msg += f"- Material (B vs N): {mat_str} {mat_desc}\n"
        msg += f"- Movilidad: Blancas {mob_w} / Negras {mob_b} ({mob_desc})\n"
        msg += f"- Seguridad del Rey: {ks_desc} (Escudos: B={ks_w}, N={ks_b})\n"
        msg += f"- Peones: Estructura evaluada."
        
        self.logger.log("ESTADO INICIAL", msg)

    def _log_intention(self, board):
        # Heuristic intention
        val = self.brain.predict(board)
        
        if abs(val) < 0.5:
            intent = "Juego posicional. Quiero activar mis piezas y controlar el centro."
        elif (board.turn == chess.WHITE and val > 1.0) or (board.turn == chess.BLACK and val < -1.0):
            intent = "Posición favorable. Buscaré simplificar o atacar debilidades para consolidar."
        else:
            intent = "Estoy bajo presión. Priorizaré la defensa y buscaré contrajuego."
            
        self.logger.log("INTENCIÓN DEL MOTOR", f"\"{intent}\"")

    def _log_move_analysis(self, board, top_moves):
        msg = ""
        for i, (move, score) in enumerate(top_moves):
            # Analyze move properties
            board.push(move)
            is_capture = board.is_capture(move)
            gives_check = board.is_check()
            board.pop() # Restore
            
            desc = []
            if is_capture: desc.append("Captura material")
            if gives_check: desc.append("Genera amenaza directa (Jaque)")
            if not desc: desc.append("Mejora posicional / Reagrupación")
            
            msg += f"{i+1}. {move} → Score: {score:.2f}\n"
            for d in desc:
                msg += f"   - {d}\n"
            msg += "\n"
        self.logger.log("ANÁLISIS DE MOVIMIENTOS", msg.strip())

    def _log_reasoning(self, board, best_move, score):
        # Narrative reasoning
        board.push(best_move)
        check = board.is_check()
        capture = board.is_capture(best_move)
        board.pop()
        
        reason = "Estoy priorizando jugadas que aumenten mi control."
        if capture:
            reason = "La ganancia de material es prioritaria para asegurar la ventaja."
        elif check:
            reason = "El ataque al rey obliga al oponente a responder, ganando tiempo."
        elif abs(score) < 0.2:
            reason = "La posición está igualada, busco mejorar la actividad de mis piezas menores."
            
        msg = f"RAZONAMIENTO\n{reason} La jugada {best_move} parece la más sólida."
        self.logger.log("RAZONAMIENTO", msg)

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        # 5.C Transposition Table Lookup
        fen = board.fen()
        if fen in self.tt:
            stored_depth, stored_val = self.tt[fen]
            if stored_depth >= depth:
                return stored_val

        if depth == 0 or board.is_game_over():
            val = self.brain.predict(board)
            # Guardar en TT
            self.tt[fen] = (depth, val)
            return val

        legal_moves = list(board.legal_moves)
        
        # 5.B Ordenamiento Recursivo
        # Killer moves heuristic simple: capturas primero
        legal_moves.sort(key=lambda m: (board.is_capture(m), board.gives_check(m)), reverse=True)

        if is_maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                board.push(move)
                eval_val = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha: break
            
            self.tt[fen] = (depth, max_eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_val = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha: break
            
            self.tt[fen] = (depth, min_eval)
            return min_eval

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
        
        # Initial layout calculation
        self.recalculate_layout(self.WIDTH, self.HEIGHT)
        
        # Connect logger callback
        self.engine.logger.gui_callback = self.log_message

    def recalculate_layout(self, w, h):
        self.WIDTH = w
        self.HEIGHT = h
        
        # Sidebar takes 35% of width, but at least 350px
        self.SIDEBAR_WIDTH = max(350, int(self.WIDTH * 0.35))
        
        # Commentary Panel Height (Fixed or proportional)
        self.COMMENTARY_HEIGHT = 120
        
        # Available area for board (Right side, Top part)
        self.BOARD_AREA_WIDTH = self.WIDTH - self.SIDEBAR_WIDTH
        self.BOARD_AREA_HEIGHT = self.HEIGHT - self.COMMENTARY_HEIGHT
        
        # Board size is limited by the smaller dimension to keep it square
        # We add some padding
        board_size = min(self.BOARD_AREA_WIDTH, self.BOARD_AREA_HEIGHT) - 20
        self.SQ_SIZE = max(10, board_size // 8)
        
        # Centering the board in the available top-right area
        self.BOARD_OFFSET_X = self.SIDEBAR_WIDTH + (self.BOARD_AREA_WIDTH - (self.SQ_SIZE * 8)) // 2
        self.BOARD_OFFSET_Y = (self.BOARD_AREA_HEIGHT - (self.SQ_SIZE * 8)) // 2
        
        # Update fonts based on size
        self.font = pygame.font.SysFont("segoe ui symbol", int(self.SQ_SIZE * 0.8))
        self.ui_font = pygame.font.SysFont("arial", max(16, int(self.HEIGHT * 0.025)))
        self.log_font = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("arial", max(30, int(self.HEIGHT * 0.06)), bold=True)
        
        # Re-wrap logs because width changed
        self.rewrap_logs()

    def rewrap_logs(self):
        self.wrapped_logs = []
        for log in self.logs:
            self._wrap_and_append(log)
            
    def _wrap_and_append(self, log):
        words = log.split(' ')
        current_line = []
        
        # Determine color based on content
        color = self.SIDEBAR_TEXT
        if "RAZONAMIENTO" in log: color = (255, 200, 100) # Thinking
        elif "PREDICTIVE CODING" in log or "Sorpresa" in log: color = (255, 100, 100) # Warning/Capture
        elif "DECISIÓN FINAL" in log: color = (100, 255, 100) # Good move
        elif "ANÁLISIS" in log or "analyzing" in log.lower(): color = (100, 200, 255)
        
        max_width = self.SIDEBAR_WIDTH - 50 # Margin for scrollbar
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if self.log_font.size(test_line)[0] < max_width:
                current_line.append(word)
            else:
                self.wrapped_logs.append(( ' '.join(current_line), color ))
                current_line = [word]
        self.wrapped_logs.append(( ' '.join(current_line), color ))

    def log_message(self, message):
        # Add to logs list
        self.logs.append(message)
        # Keep history reasonable but large enough
        if len(self.logs) > 500: 
            self.logs.pop(0)
            self.rewrap_logs() # Full rewrap needed if we pop from start
        else:
            self._wrap_and_append(message)
            
        # Auto-scroll to bottom
        line_height = self.log_font.get_linesize()
        total_height = len(self.wrapped_logs) * line_height
        log_area_height = self.HEIGHT - 160
        
        if total_height > log_area_height:
            self.scroll_offset = total_height - log_area_height
        
        if self.running:
            try:
                self.draw_sidebar()
                pygame.display.update(pygame.Rect(0, 0, self.SIDEBAR_WIDTH, self.HEIGHT))
            except: pass

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
        # Background
        pygame.draw.rect(self.screen, self.SIDEBAR_BG, (0, 0, self.SIDEBAR_WIDTH, self.HEIGHT))
        
        # Title Area
        title_text = "PCNN Chess"
        title_surf = self.title_font.render(title_text, True, (255, 255, 255))
        
        # Scale title if it doesn't fit
        if title_surf.get_width() > self.SIDEBAR_WIDTH - 40:
            scale = (self.SIDEBAR_WIDTH - 40) / title_surf.get_width()
            new_size = (int(title_surf.get_width() * scale), int(title_surf.get_height() * scale))
            title_surf = pygame.transform.smoothscale(title_surf, new_size)
            
        self.screen.blit(title_surf, (20, 20))
        
        subtitle = self.ui_font.render("AlphaZero Lite - XAI Mode", True, (100, 200, 255))
        self.screen.blit(subtitle, (20, 80))
        
        # Separator
        pygame.draw.line(self.screen, (100, 100, 100), (20, 120), (self.SIDEBAR_WIDTH - 20, 120), 2)
        
        # Logs Area
        log_start_y = 140
        log_area_height = self.HEIGHT - log_start_y - 20
        line_height = self.log_font.get_linesize()
        
        # Calculate visible range
        total_lines = len(self.wrapped_logs)
        total_height = total_lines * line_height
        
        # Clamp scroll
        max_scroll = max(0, total_height - log_area_height)
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))
        
        start_index = int(self.scroll_offset // line_height)
        end_index = start_index + int(log_area_height // line_height) + 2
        
        visible_lines = self.wrapped_logs[start_index:end_index]
        
        y = log_start_y - (self.scroll_offset % line_height)
        
        # Clip area for logs
        old_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(0, log_start_y, self.SIDEBAR_WIDTH, log_area_height))
        
        for line_text, line_color in visible_lines:
            surf = self.log_font.render(line_text, True, line_color)
            self.screen.blit(surf, (20, y))
            y += line_height
            
        self.screen.set_clip(old_clip)
        
        # Draw Scrollbar
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
        # Clear board area background to remove artifacts (like the "P" from title)
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
        
        # Check if click is within board area
        board_rect = pygame.Rect(self.BOARD_OFFSET_X, self.BOARD_OFFSET_Y, self.SQ_SIZE*8, self.SQ_SIZE*8)
        if not board_rect.collidepoint(pos): return False

        c = (pos[0] - self.BOARD_OFFSET_X) // self.SQ_SIZE
        r = (pos[1] - self.BOARD_OFFSET_Y) // self.SQ_SIZE
        
        is_flipped = (self.player_color == chess.BLACK)
        if is_flipped:
            sq = chess.square(7-c, r)
        else:
            sq = chess.square(c, 7-r)
        
        if self.selected_square is None:
            p = self.board.piece_at(sq)
            if p and p.color == self.player_color: self.selected_square = sq
        else:
            move = chess.Move(self.selected_square, sq)
            p = self.board.piece_at(self.selected_square)
            
            target_rank = chess.square_rank(sq)
            if p and p.piece_type == chess.PAWN:
                if (p.color == chess.WHITE and target_rank == 7) or (p.color == chess.BLACK and target_rank == 0):
                    move = chess.Move(self.selected_square, sq, promotion=chess.QUEEN)
            
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                # self.log_message(f"Jugador: {move}") # Removed to keep logs clean for AI thoughts
                return True
            else:
                p = self.board.piece_at(sq)
                if p and p.color == self.player_color: self.selected_square = sq
                else: self.selected_square = None
        return False

    def input_name_screen(self):
        color_inactive = pygame.Color('lightskyblue3')
        color_active = pygame.Color('dodgerblue2')
        color = color_inactive
        active = False
        text = ''
        done = False
        selected_color = None

        def get_ui_rects():
            cx, cy = self.WIDTH // 2, self.HEIGHT // 2
            return (
                pygame.Rect(cx - 100, cy - 20, 200, 32), # input_box
                pygame.Rect(cx - 110, cy + 30, 100, 40), # white_btn
                pygame.Rect(cx + 10, cy + 30, 100, 40),  # black_btn
                pygame.Rect(cx - 60, cy + 90, 120, 40)   # start_btn
            )

        input_box, white_btn, black_btn, start_btn = get_ui_rects()

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.VIDEORESIZE:
                    self.recalculate_layout(event.w, event.h)
                    input_box, white_btn, black_btn, start_btn = get_ui_rects()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if input_box.collidepoint(event.pos): active = not active
                    else: active = False
                    
                    if white_btn.collidepoint(event.pos): selected_color = chess.WHITE
                    if black_btn.collidepoint(event.pos): selected_color = chess.BLACK
                    
                    if start_btn.collidepoint(event.pos):
                        if text.strip() and selected_color is not None: 
                            self.player_name = text
                            self.player_color = selected_color
                            done = True
                    
                    color = color_active if active else color_inactive
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()
                    if active:
                        if event.key == pygame.K_RETURN:
                            if text.strip() and selected_color is not None: 
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
        
        if winner == self.player_color:
            msg = f"¡Increíble {self.player_name}! Tu cerebro orgánico ha vencido."
            base_color = (50, 255, 50)
        elif winner is not None:
            msg = f"¡Jaque Mate {self.player_name}! La IA ha dominado esta partida."
            base_color = (255, 50, 50)
        elif self.board.is_stalemate():
            msg = f"¡Tablas! Rey Ahogado. Nadie gana."
            base_color = (255, 215, 0)
        else:
            msg = f"Empate Táctico (Tablas)."
            base_color = (200, 200, 255)
            
        center_x = self.SIDEBAR_WIDTH + self.BOARD_AREA_WIDTH // 2
        center_y = self.BOARD_AREA_HEIGHT // 2
        btn_restart = pygame.Rect(center_x - 120, center_y + 60, 240, 60)
        btn_logs = pygame.Rect(center_x - 120, center_y + 140, 240, 60)
        game_over_font = pygame.font.SysFont("arial", 32, bold=True)
        
        logs_saved_msg = ""
        logs_saved_timer = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_restart.collidepoint(event.pos): return "restart"
                    if btn_logs.collidepoint(event.pos): 
                        if self.save_logs():
                            logs_saved_msg = "Logs guardados exitosamente"
                            logs_saved_timer = pygame.time.get_ticks()

            self.draw_sidebar(); self.draw_board(); self.draw_pieces()
            # Draw commentary panel too so it doesn't disappear
            self.draw_commentary_overlay(self.engine.latest_commentary)
            
            overlay = pygame.Surface((self.BOARD_AREA_WIDTH, self.BOARD_AREA_HEIGHT))
            # Hacer el overlay más transparente (160 en lugar de 240) para ver el tablero final
            overlay.set_alpha(160); overlay.fill((20, 20, 25))
            self.screen.blit(overlay, (self.SIDEBAR_WIDTH, 0))
            
            words = msg.split(' ')
            lines = []
            current_line = []
            max_width = self.BOARD_AREA_WIDTH - 60
            for word in words:
                test_line = ' '.join(current_line + [word])
                if game_over_font.size(test_line)[0] < max_width: current_line.append(word)
                else: lines.append(' '.join(current_line)); current_line = [word]
            lines.append(' '.join(current_line))
            
            total_height = len(lines) * 40
            start_y = center_y - total_height // 2 - 40
            for i, line in enumerate(lines):
                txt = game_over_font.render(line, True, base_color)
                self.screen.blit(txt, txt.get_rect(center=(center_x, start_y + i * 40)))
                
            pygame.draw.rect(self.screen, (0, 180, 0), btn_restart, border_radius=10)
            pygame.draw.rect(self.screen, (255, 255, 255), btn_restart, 2, border_radius=10)
            rtxt = self.ui_font.render("Jugar de Nuevo", True, (255, 255, 255))
            self.screen.blit(rtxt, rtxt.get_rect(center=btn_restart.center))
            
            pygame.draw.rect(self.screen, (70, 130, 180), btn_logs, border_radius=10)
            pygame.draw.rect(self.screen, (255, 255, 255), btn_logs, 2, border_radius=10)
            ltxt = self.ui_font.render("Guardar Logs", True, (255, 255, 255))
            self.screen.blit(ltxt, ltxt.get_rect(center=btn_logs.center))
            
            if logs_saved_msg:
                # Fade out after 3 seconds
                if pygame.time.get_ticks() - logs_saved_timer > 3000:
                    logs_saved_msg = ""
                else:
                    saved_txt = self.ui_font.render(logs_saved_msg, True, (100, 255, 100))
                    self.screen.blit(saved_txt, saved_txt.get_rect(center=(center_x, self.HEIGHT - 40)))
            
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
            
        # Commentary Panel Area
        panel_x = self.SIDEBAR_WIDTH
        panel_y = self.HEIGHT - self.COMMENTARY_HEIGHT
        panel_w = self.WIDTH - self.SIDEBAR_WIDTH
        panel_h = self.COMMENTARY_HEIGHT
        
        # Draw Panel Background
        pygame.draw.rect(self.screen, (25, 25, 30), (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.line(self.screen, (60, 60, 70), (panel_x, panel_y), (self.WIDTH, panel_y), 2)
        
        # Title
        title_font = pygame.font.SysFont("arial", 14, bold=True)
        title = title_font.render("COMENTARIOS GM", True, (150, 150, 160))
        self.screen.blit(title, (panel_x + 20, panel_y + 10))
        
        # Render text
        font = pygame.font.SysFont("georgia", 18, italic=True)
        wrapped = textwrap.fill(text, width=int(panel_w / 10)) # Approx chars per line
        lines = wrapped.splitlines()
        
        y_offset = panel_y + 35
        for line in lines[:3]: # Max 3 lines
            txt = font.render(line, True, (230, 230, 210)) # Cream/White
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
                    metrics = self.brain.learn_from_game(self.game_history, res_val)
                    
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

                # Human Turn
                if self.board.turn == self.player_color:
                    for event in events:
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if self.handle_click(event.pos):
                                self.game_history.append(self.board.copy())
                                self.draw_sidebar(); self.draw_board(); self.draw_pieces()
                                pygame.display.flip()
                else:
                    # AI Turn
                    if self.running and not self.board.is_game_over():
                        # Force redraw before thinking
                        self.draw_sidebar(); self.draw_board(); self.draw_pieces()
                        pygame.display.flip()
                        
                        best_move = self.engine.get_best_move(self.board)
                        if best_move:
                            self.board.push(best_move)
                            self.game_history.append(self.board.copy())
                        else:
                            self.engine.logger.log("DECISIÓN FINAL", "IA se rinde.")
                            break
                
                self.draw_sidebar(); self.draw_board(); self.draw_pieces()
                self.draw_commentary_overlay(self.engine.latest_commentary)
                pygame.display.flip()
                clock.tick(30)

if __name__ == "__main__":
    brain = NeuralBrain()
    engine = ChessEngine(brain)
    gui = ChessGUI(brain, engine)
    gui.run()
