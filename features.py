import chess
import numpy as np

def extract_features(board):
    """
    Extracción de características optimizada usando operaciones de bits (Bitboards).
    Vector de ~30 dimensiones.
    """
    features = []
    
    # 1. Material (12)
    # Usamos bit_count() que es mucho más rápido que len(board.pieces())
    for piece_type in chess.PIECE_TYPES:
        features.append(board.pieces_mask(piece_type, chess.WHITE).bit_count())
        features.append(board.pieces_mask(piece_type, chess.BLACK).bit_count())
    
    # 2. Movilidad (2)
    # legal_moves.count() es costoso, pero necesario para precisión.
    if board.turn == chess.WHITE:
        features.append(board.legal_moves.count())
        board.turn = chess.BLACK
        features.append(board.legal_moves.count())
        board.turn = chess.WHITE
    else:
        features.append(board.legal_moves.count())
        board.turn = chess.WHITE
        features.append(board.legal_moves.count())
        board.turn = chess.BLACK

    # 3. Seguridad del Rey (2)
    # Usamos máscaras de bits para verificar peones alrededor del rey
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is not None:
            # Máscara de los alrededores del rey
            king_mask = chess.BB_KING_ATTACKS[king_sq]
            
            # Máscara de peones propios
            pawns_mask = board.pieces_mask(chess.PAWN, color)
            
            # Intersección: Peones propios que protegen al rey
            shield = (king_mask & pawns_mask).bit_count()
            features.append(shield)
        else:
            features.append(0)

    # 4. Estructura de Peones (4)
    # Peones centrales y avanzados usando máscaras
    white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
    
    # Máscara central (d4, d5, e4, e5)
    center_mask = 0x0000001818000000 
    features.append((white_pawns & center_mask).bit_count())
    features.append((black_pawns & center_mask).bit_count())
    
    # Peones avanzados
    white_advanced_mask = 0x00FFFFFFFF000000 # Ranks 5, 6, 7, 8
    black_advanced_mask = 0x00000000FFFFFFFF # Ranks 4, 3, 2, 1
    
    features.append((white_pawns & white_advanced_mask).bit_count())
    features.append((black_pawns & black_advanced_mask).bit_count())

    # 5. Jaque (1)
    features.append(1 if board.is_check() else 0)
    
    # 6. Turno (1)
    features.append(1 if board.turn == chess.WHITE else -1)

    return np.array(features, dtype=np.float32)
