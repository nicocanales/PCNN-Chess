import chess
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import kagglehub
import joblib
import os
import time
import threading

# Importaciones locales
from features import extract_features
from replay_buffer import ReplayBuffer
from commentary_narrator import NaturalNarrator

class ThoughtLogger:
    """Sistema de logging estructurado para el pensamiento de la IA."""
    def __init__(self, filename="logs/pcnn_thoughts.log"):
        self.filename = filename
        self.gui_callback = None
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("--- PCNN CHESS THOUGHT LOGS ---\n")

    def log(self, section, content):
        formatted_msg = f"[{section}]\n{content}"
        try:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(formatted_msg + "\n\n")
        except: pass
        
        print(formatted_msg)
        print("-" * 20)
        
        if self.gui_callback:
            self.gui_callback(f"[{section}]")
            for line in content.split('\n'):
                if line.strip():
                    self.gui_callback("  " + line)

class NeuralBrain:
    """
    Cerebro basado en Red Neuronal (MLPRegressor).
    Maneja la evaluación de posiciones y el aprendizaje incremental.
    """
    def __init__(self):
        # 5. MEJORA: Reemplazo de SGD por MLPRegressor (Red Neuronal Real)
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate_init=0.001,
            max_iter=1,  # Para partial_fit simulado (warm_start)
            warm_start=True,
            random_state=42
        )
        self.is_trained = False
        self.replay_buffer = ReplayBuffer(capacity=20000)
        self.batch_size = 64
        self.train_counter = 0
        self.logger = None
        
        self.load_model()

    def _normalize_eval(self, val):
        return np.clip(val / 10.0, -1.0, 1.0)

    def _denormalize_eval(self, val):
        return val * 10.0

    def predict(self, board):
        if not self.is_trained:
            return self._material_fallback(board) / 10.0
        
        try:
            features = extract_features(board).reshape(1, -1)
            pred = self.model.predict(features)[0]
            if np.isnan(pred) or np.isinf(pred):
                return self._material_fallback(board) / 10.0
            return self._denormalize_eval(pred)
        except:
            return self._material_fallback(board) / 10.0

    def _material_fallback(self, board):
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
        """Aprendizaje TD-Lambda Offline al final de la partida."""
        gamma = 0.99
        lam = 0.7
        
        states = [extract_features(b) for b in history]
        predictions = []
        if self.is_trained:
            predictions = self.model.predict(np.array(states))
        else:
            predictions = np.zeros(len(states))
            
        G = result
        game_experiences = []
        
        for t in reversed(range(len(history))):
            state = states[t]
            target = max(-1.0, min(1.0, G))
            self.replay_buffer.add(state, target)
            game_experiences.append((state, target))
            
            pred_val = predictions[t]
            G = (1 - lam) * pred_val + lam * G
            G *= gamma

        if len(self.replay_buffer) < self.batch_size:
             return {"loss": 0, "E_pos": 0, "E_neg": 0}

        X_batch, y_batch = self.replay_buffer.sample(self.batch_size)
        self.model.partial_fit(X_batch, y_batch) # MLP soporta partial_fit
        self.train_counter += 1
        
        if self.train_counter % 200 == 0:
            self.save_model()
            self.replay_buffer.save()

        # Métricas
        X_game = np.array([e[0] for e in game_experiences])
        y_game = np.array([e[1] for e in game_experiences])
        current_preds = self.model.predict(X_game)
        loss = np.mean((y_game - current_preds)**2)
        E_pos = np.maximum(0, y_game - current_preds).mean()
        E_neg = np.maximum(0, current_preds - y_game).mean()
        
        if self.logger:
            msg = f"- Sorpresa: {E_pos:.4f}\n- Decepción: {E_neg:.4f}\n- Loss: {loss:.4f}"
            self.logger.log("PREDICTIVE CODING", msg)
        
        return {"loss": loss, "E_pos": E_pos, "E_neg": E_neg}

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
        self.load_model()
        if self.is_trained:
            if progress_callback: progress_callback("Modelo cargado.", 1.0)
            return

        if progress_callback: progress_callback("Descargando dataset...", 0.1)
        try:
            path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")
            csv_path = None
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".csv"):
                        csv_path = os.path.join(root, file); break
                if csv_path: break
            
            if not csv_path: raise FileNotFoundError("CSV no encontrado")

            if progress_callback: progress_callback("Entrenando...", 0.2)
            
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
                    if progress_callback: progress_callback(f"Batch {i}...", pct)
            
            self.is_trained = True
            self.save_model()
            if progress_callback: progress_callback("Listo!", 1.0)
            
        except Exception as e:
            print(f"Error entrenamiento: {e}")

    def _parse_evaluation(self, eval_str):
        if "#" in str(eval_str):
            return 1000 if "+" in str(eval_str) else -1000
        try: return float(eval_str)
        except: return 0.0

class ChessEngine:
    """
    Motor de búsqueda con Minimax, Alpha-Beta, Quiescence Search e Iterative Deepening.
    """
    def __init__(self, brain):
        self.brain = brain
        self.logger = ThoughtLogger()
        self.brain.logger = self.logger
        self.tt = {} 
        self.narrator = NaturalNarrator()
        self.latest_commentary = ""
        self.stop_search = False
        self.start_time = 0
        self.time_limit = 3.0 # Segundos por jugada

    def get_best_move(self, board):
        """
        Método principal para obtener la mejor jugada.
        Usa Iterative Deepening.
        """
        self.stop_search = False
        self.start_time = time.time()
        
        self._log_state(board)
        self._log_intention(board)
        
        best_move = None
        best_score = 0
        
        # 4. MEJORA: Iterative Deepening
        # Profundidad 1 a 4 (o más si el tiempo lo permite)
        max_depth = 4
        
        for depth in range(1, max_depth + 1):
            if self.stop_search or (time.time() - self.start_time > self.time_limit):
                break
                
            score, move = self.minimax_root(board, depth)
            
            if not self.stop_search:
                best_move = move
                best_score = score
                # Log parcial
                print(f"Depth {depth}: Move {move}, Score {score:.2f}")
        
        # Generar narrativa final
        self._log_reasoning(board, best_move, best_score)
        
        if best_move is None:
            return None

        state = self._get_narrative_state(board)
        eval_info = {
            'surprise': 0.0, 
            'tactical': board.is_capture(best_move) or board.gives_check(best_move),
            'plan_tag': 'posicional'
        }
        self.latest_commentary = self.narrator.natural_commentary(state, best_move, eval_info)
        
        return best_move

    def minimax_root(self, board, depth):
        legal_moves = list(board.legal_moves)
        if not legal_moves: return 0, None
        
        # Ordenamiento
        legal_moves.sort(key=lambda m: (board.is_capture(m), board.gives_check(m)), reverse=True)
        
        best_move = legal_moves[0]
        alpha = -float('inf')
        beta = float('inf')
        is_maximizing = board.turn == chess.WHITE
        best_val = -float('inf') if is_maximizing else float('inf')
        
        move_scores = []
        
        for move in legal_moves:
            if time.time() - self.start_time > self.time_limit:
                self.stop_search = True
                break
                
            board.push(move)
            val = self.minimax(board, depth - 1, alpha, beta, not is_maximizing)
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
        
        # Log top moves for depth
        move_scores.sort(key=lambda x: x[1], reverse=is_maximizing)
        if depth >= 2: # Solo loguear en profundidades decentes
            self._log_move_analysis(board, move_scores[:3])
            
        return best_val, best_move

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        # Check time
        if self.stop_search or (time.time() - self.start_time > self.time_limit):
            self.stop_search = True
            return 0 # Valor dummy, se descartará
            
        # Transposition Table
        fen = board.fen()
        if fen in self.tt:
            stored_depth, stored_val = self.tt[fen]
            if stored_depth >= depth:
                return stored_val

        if board.is_game_over():
            return self.brain.predict(board)

        # 3. MEJORA: Quiescence Search en profundidad 0
        if depth == 0:
            return self.quiescence(board, alpha, beta)

        legal_moves = list(board.legal_moves)
        # Ordenamiento simple
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

    def quiescence(self, board, alpha, beta):
        """
        Búsqueda en reposo para evitar el efecto horizonte.
        Solo evalúa capturas y jaques.
        """
        # Check time
        if self.stop_search or (time.time() - self.start_time > self.time_limit):
            self.stop_search = True
            return self.brain.predict(board)

        # RE-IMPLEMENTATION OF QUIESCENCE (Correct Minimax Style)
        is_maximizing = board.turn == chess.WHITE
        
        stand_pat = self.brain.predict(board)
        
        if is_maximizing:
            if stand_pat >= beta: return beta
            if stand_pat > alpha: alpha = stand_pat
        else:
            if stand_pat <= alpha: return alpha
            if stand_pat < beta: beta = stand_pat
            
        tactical_moves = [m for m in board.legal_moves if board.is_capture(m)]
        
        if is_maximizing:
            for move in tactical_moves:
                if self.stop_search: break
                board.push(move)
                score = self.quiescence(board, alpha, beta)
                board.pop()
                
                if score >= beta: return beta
                if score > alpha: alpha = score
            return alpha
        else:
            for move in tactical_moves:
                if self.stop_search: break
                board.push(move)
                score = self.quiescence(board, alpha, beta)
                board.pop()
                
                if score <= alpha: return alpha
                if score < beta: beta = score
            return beta

    def _get_narrative_state(self, board):
        feats = extract_features(board)
        # Indices based on features.py (updated):
        # 0-11: Material (P,N,B,R,Q,K for W then B) -> Interleaved? No, features.py appends W then B for each type.
        # P_W, P_B, N_W, N_B...
        
        w_mat = feats[0]*1 + feats[2]*3 + feats[4]*3 + feats[6]*5 + feats[8]*9
        b_mat = feats[1]*1 + feats[3]*3 + feats[5]*3 + feats[7]*5 + feats[9]*9
        mat_diff = w_mat - b_mat
        
        # Mobility is at index 12, 13
        if board.turn == chess.WHITE:
            mob_w = int(feats[12])
            mob_b = int(feats[13])
        else:
            mob_b = int(feats[12])
            mob_w = int(feats[13])
            
        # King Safety 14, 15
        ks_w = int(feats[14])
        ks_b = int(feats[15])
        
        return {
            'material_diff': mat_diff,
            'mobility_white': mob_w,
            'mobility_black': mob_b,
            'threats_white': ks_w,
            'threats_black': ks_b,
            'pawn_structure_summary': "estándar",
            'fen': board.fen()
        }

    def _log_state(self, board):
        self.logger.log("ESTADO", f"FEN: {board.fen()}")

    def _log_intention(self, board):
        val = self.brain.predict(board)
        self.logger.log("INTENCIÓN", f"Eval: {val:.2f}")

    def _log_move_analysis(self, board, top_moves):
        msg = "\n".join([f"{m}: {s:.2f}" for m, s in top_moves])
        self.logger.log("ANÁLISIS", msg)

    def _log_reasoning(self, board, best_move, score):
        self.logger.log("RAZONAMIENTO", f"Elegida {best_move} con score {score:.2f}")
