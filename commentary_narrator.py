# commentary_narrator.py
# Módulo: Natural Narrator (Gran Maestro)
# Propósito: convertir señales del motor en comentarios estilo Gran Maestro
# Diseño: sistema mixto
# - Generador programático de plantillas (permite producir ~1500 frases únicas)
# - Lógica narrativa robusta, con estilos, intensidad, reglas de no-tecnicismo
# - API sencilla: natural_commentary(state, move, eval_info, style='gm_classic')

import random
import textwrap
import math
from typing import Dict, Optional, List

# ---------------------------
# Configurable parameters
# ---------------------------
MAX_COMMENT_CHARS = 200
MAX_COMMENT_LINES = 3

# Styles available: 'gm_classic', 'kasparov', 'karpov', 'polgar', 'nepo'
DEFAULT_STYLE = 'gm_classic'
STYLE_LIST = ['gm_classic', 'kasparov', 'karpov', 'polgar', 'nepo']

# ---------------------------
# Utility functions
# ---------------------------
def clamp_text_to_lines(text: str, max_lines: int = MAX_COMMENT_LINES, max_chars: int = MAX_COMMENT_CHARS) -> str:
    """Ensure output respects max lines and chars (break gracefully)."""
    # cut by chars first
    if len(text) > max_chars:
        text = text[:max_chars-1] + "…"
    # then ensure lines
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    # join first max_lines and truncate last line gracefully
    kept = lines[:max_lines]
    kept[-1] = kept[-1][:max(0, max_chars - sum(len(l)+1 for l in kept[:-1]) - 3)] + "…"
    return "\n".join(kept)

def safe_str(x):
    try:
        return str(x)
    except:
        return ""

# ---------------------------
# Fragments: building blocks for templates
# These are small well-crafted phrases that will be combined
# ---------------------------

# Intent fragments: describe intention/plan
INTENT_FRAGMENTS = [
    "buscar{target}",
    "presionar{target}",
    "mantener{target}",
    "simplificar la posición",
    "aprovechar la iniciativa",
    "mejorar la coordinación de piezas",
    "crear tensiones en{target}",
    "abrir líneas hacia el rey enemigo",
    "consolidar su ventaja material",
    "restringir la movilidad rival",
    "provocar cambios favorables en la estructura de peones",
    "desplegar piezas hacia el flanco rey",
    "forzar concesiones posicionales",
    "abrir la columna{file}",
    "crear peones pasados en el flanco{side}",
    "activarse en la séptima fila",
    "ocupar casillas claves en el centro",
    "inducir un cambio de estructura que favorece los finales",
    "liberar la presión tras un intercambio oportuno",
    "buscar contrajuego en el flanco dama",
]

# Target phrases for intent placeholders
TARGETS = [
    " el centro",
    " la columna abierta",
    " las casillas oscuras",
    " las casillas claras",
    " la casilla e4",
    " la casilla d5",
    " la casilla c3",
    " el flanco rey",
    " el flanco dama",
    " la diagonal larga",
    " las rutas hacia el rey",
    " la casilla crítica",
    ""
]

# File / side placeholders
FILES = [" 'd'", " 'e'"]
SIDES = [" dama", " rey", " de rey", " de dama"]

# Motivational fragments: why a move is good/bad
MOTIVATION_FRAGMENTS = [
    "mejora la actividad de las piezas",
    "reduces la presión enemiga",
    "amplía el control central",
    "prepara maniobras de infiltración",
    "evita debilidades inmediatas",
    "crea amenazas tácticas claras",
    "es profiláctica y rotunda",
    "genera contrajuego inmediato",
    "limita el acceso a casillas crítica",
    "simplifica hacia un final favorables",
    "aumenta la coordinación lateral",
    "sirve a un plan a largo plazo",
    "es sutil pero efectivo",
    "ahorra tiempos y mejora la estructura",
]

# Emotion/intensity phrases
EMOTION_POS = [
    "Una idea precisa.",
    "Juego convincente.",
    "Magnífica ejecución.",
    "Una decisión inspirada.",
    "Una jugada con intención clara.",
    "Muy instructiva.",
]
EMOTION_NEU = [
    "Juego posicional correcto.",
    "Movimiento sobrio y cuerdo.",
    "Una jugada rutinaria, pero sólida.",
    "Progresión lógica.",
]
EMOTION_NEG = [
    "Una imprecisión evidente.",
    "Un paso que ofrece recursos al rival.",
    "Juego peligroso que exige cuidado.",
    "Demasiado optimista en la posición actual.",
    "Exposición innecesaria.",
]

# Transition phrases
TRANSITIONS = [
    "En la práctica esto significa",
    "En términos sencillos",
    "De forma práctica",
    "En esencia",
    "Por tanto",
    "Como resultado",
    "En consecuencia",
]

# Tactical cues
TACTICAL = [
    "doble amenaza",
    "clavada decisiva",
    "ataque a la base del enroque",
    "infiltración en la séptima",
    "peón pasado avanzado",
    "sacrificio por iniciativa",
    "gambito posicional",
    "ganancia de tiempos",
    "desvío táctico",
    "amenaza de mate",
]

# Closure patterns (end of phrase)
CLOSURES = [
    "y la posición mejora.",
    "sin perder la claridad del plan.",
    "con poco contrajuego al rival.",
    "abriendo opciones de ataque.",
    "y el plan queda claro.",
    "con riesgos controlados.",
    "pero requiere precisión.",
    "y obliga a defender con cuidado.",
]

# Style overlays: how each style transforms / prefers phrases
STYLE_TONE = {
    'gm_classic': {
        'intro_presence': 0.6,
        'prefer_positional': 0.5,
        'prefer_tactical': 0.5,
        'emotion_mix': (0.6, 0.25, 0.15)  # pos, neutral, neg
    },
    'kasparov': {
        'intro_presence': 0.8,
        'prefer_positional': 0.3,
        'prefer_tactical': 0.9,
        'emotion_mix': (0.7, 0.1, 0.2)
    },
    'karpov': {
        'intro_presence': 0.4,
        'prefer_positional': 0.95,
        'prefer_tactical': 0.2,
        'emotion_mix': (0.5, 0.4, 0.1)
    },
    'polgar': {
        'intro_presence': 0.7,
        'prefer_positional': 0.6,
        'prefer_tactical': 0.6,
        'emotion_mix': (0.6, 0.3, 0.1)
    },
    'nepo': {
        'intro_presence': 0.9,
        'prefer_positional': 0.4,
        'prefer_tactical': 0.7,
        'emotion_mix': (0.5, 0.2, 0.3)
    }
}

# ---------------------------
# Programmatic template generator
# ---------------------------

def generate_templates_for_style(style: str, n_desired: int = 1500) -> List[str]:
    """
    Programmatically build a large set of templates for a style by combining fragments.
    This avoids manual listing of 1500 lines while producing varied, high-quality sentences.
    """
    random.seed(42 + hash(style))  # deterministic-ish per style
    templates = set()
    iterations = 0
    max_iter = n_desired * 10
    while len(templates) < n_desired and iterations < max_iter:
        iterations += 1
        # choose structure pattern
        pattern_type = random.choices(
            ['intent_then_motiv', 'motiv_then_closure', 'short_emotion', 'transition_explain', 'tactical_alert'],
            weights=[30, 30, 15, 15, 10],
            k=1
        )[0]

        # pick intensity and emotion biases from style
        tone = STYLE_TONE.get(style, STYLE_TONE['gm_classic'])
        emotion_probs = tone['emotion_mix']

        # compose fragments
        if pattern_type == 'intent_then_motiv':
            intent = random.choice(INTENT_FRAGMENTS)
            target = random.choice(TARGETS)
            intent_text = intent.format(target=target, file=random.choice(FILES), side=random.choice(SIDES))
            motiv = random.choice(MOTIVATION_FRAGMENTS)
            closure = random.choice(CLOSURES)
            sentence = f"{intent_text.capitalize()}: {motiv} {closure}"
        elif pattern_type == 'motiv_then_closure':
            motiv = random.choice(MOTIVATION_FRAGMENTS)
            closure = random.choice(CLOSURES)
            sentence = f"{motiv.capitalize()}, {closure}"
        elif pattern_type == 'short_emotion':
            # pick positive/neutral/neg based on probabilities
            r = random.random()
            if r < emotion_probs[0]:
                sentence = random.choice(EMOTION_POS)
            elif r < emotion_probs[0] + emotion_probs[1]:
                sentence = random.choice(EMOTION_NEU)
            else:
                sentence = random.choice(EMOTION_NEG)
        elif pattern_type == 'transition_explain':
            tr = random.choice(TRANSITIONS)
            intent = random.choice(INTENT_FRAGMENTS).format(target=random.choice(TARGETS), file=random.choice(FILES), side=random.choice(SIDES))
            sentence = f"{tr}, {intent}."
        else:  # tactical_alert
            tact = random.choice(TACTICAL)
            motiv = random.choice(MOTIVATION_FRAGMENTS)
            sentence = f"Atención táctica: {tact}. {motiv.capitalize()}."
        # stylistic tweaks
        if style == 'kasparov' and random.random() < 0.25:
            sentence = sentence.replace(".", "!")  # more exclamatory
        if style == 'karpov' and random.random() < 0.35:
            # longer, calmer phrasing
            sentence = sentence.replace(":", ", una idea que").replace("!", ".").replace("  ", " ")
            sentence = sentence.replace("  ", " ")
        # ensure punctuation and capitalization
        sentence = sentence.strip()
        if not sentence.endswith(".") and not sentence.endswith("!") and not sentence.endswith("…"):
            sentence = sentence + "."
        # constrain length roughly
        if len(sentence) > MAX_COMMENT_CHARS - 10:
            sentence = sentence[:MAX_COMMENT_CHARS-12] + "…"
        templates.add(sentence)
    # convert to list and shuffle
    templates_list = list(templates)
    random.shuffle(templates_list)
    return templates_list[:n_desired]

# ---------------------------
# Narrative logic: selection, context-awareness, coherence
# ---------------------------

class NaturalNarrator:
    """
    Clase principal que expone la API natural_commentary().
    Internamente mantiene plantillas generadas por estilo y lógica para seleccionar
    la frase más adecuada según el estado y el movimiento.
    """
    def __init__(self, pregen_templates: Optional[Dict[str, List[str]]] = None):
        # Load or generate templates
        self.templates = {}
        for s in STYLE_LIST:
            if pregen_templates and s in pregen_templates:
                self.templates[s] = pregen_templates[s]
            else:
                # generate templates on demand; keep light for startup (generate 300 per style)
                self.templates[s] = generate_templates_for_style(s, n_desired=300)
        # custom manual phrases (higher priority)
        self.custom_phrases = {s: [] for s in STYLE_LIST}
        # short cache to preserve coherence between consecutive moves
        self.recent_history = []  # list of (fen, last_comment)
        self.max_history = 8
        # verbosity / mode
        self.commentary_mode = True
        # intensity mapping function
        self.intensity_map = lambda surprise: min(1.0, max(0.0, surprise / 5.0))  # surprise~[0,5+]
        # style mixing probabilities for fallback
        self.fallback_mix = {s: 0.2 for s in STYLE_LIST}
        # minimal keywords mapping for contextual selection
        self.keyword_map = {
            'tactical': ['ataque', 'amenaza', 'doble', 'mate', 'clavada', 'sacrificio', 'infiltr'],
            'positional': ['control', 'central', 'estructura', 'peón', 'columna', 'movilidad'],
            'king_safety': ['rey', 'enroque', 'séptima', 'ataque al rey', 'defensa'],
            'endgame': ['final', 'peones', 'coronación', 'reedición']
        }

    # ---------- Core API ----------
    def natural_commentary(self, state: Dict, move, eval_info: Optional[Dict] = None, style: str = DEFAULT_STYLE) -> str:
        """
        Produce un comentario corto y humano a partir del estado, jugada y metadatos.
        - state: dictionary con keys importantes
        - move: chess.Move or string
        - eval_info: optional dict with keys: 'reason', 'surprise', 'plan_tag', 'tactical_flag'
        - style: style string
        """
        if not self.commentary_mode:
            return ""

        if style not in STYLE_LIST:
            style = DEFAULT_STYLE

        # Defensive defaults
        eval_info = eval_info or {}
        # Extract signal values (fallback defaults)
        material_diff = float(state.get('material_diff', 0.0))
        mobility_w = int(state.get('mobility_white', 0))
        mobility_b = int(state.get('mobility_black', 0))
        mobility_diff = mobility_w - mobility_b
        king_threats_w = int(state.get('threats_white', 0))
        king_threats_b = int(state.get('threats_black', 0))
        pawn_summary = safe_str(state.get('pawn_structure_summary', 'estructura sin cambios'))
        plan_tag = safe_str(eval_info.get('plan_tag', ''))
        surprise = float(eval_info.get('surprise', 0.0))
        tactical_flag = eval_info.get('tactical', False) or eval_info.get('tactical_flag', False)

        # compute intensity from surprise and tactical flag
        intensity_base = self.intensity_map(surprise)
        if tactical_flag:
            intensity_base = max(intensity_base, 0.6)

        # Determine contextual keywords (quick heuristic)
        context_keys = []
        if abs(material_diff) >= 2:
            context_keys.append('material')
        if abs(mobility_diff) >= 4:
            context_keys.append('mobility')
        if king_threats_b > 0 or king_threats_w > 0:
            context_keys.append('king_safety')
        if 'final' in pawn_summary.lower() or 'end' in pawn_summary.lower():
            context_keys.append('endgame')
        if tactical_flag:
            context_keys.append('tactical')

        # Compose a small internal "meaning" map for template selection
        meaning = {
            'material_diff': material_diff,
            'mobility_diff': mobility_diff,
            'king_threats_w': king_threats_w,
            'king_threats_b': king_threats_b,
            'pawn_summary': pawn_summary,
            'plan_tag': plan_tag,
            'intensity': intensity_base,
            'context_keys': context_keys,
            'move_str': safe_str(move)
        }

        # First try high-priority custom phrase
        custom_list = self.custom_phrases.get(style, [])
        if custom_list and random.random() < 0.05:
            phrase = random.choice(custom_list)
            out = clamp_text_to_lines(phrase)
            self._update_history(safe_str(state.get('fen')), out)
            return out

        # Build selection candidate pool
        pool = []
        # give higher weight to templates that match context heuristics (simple scoring)
        templates = self.templates.get(style, []) + self.templates.get('gm_classic', [])
        random.shuffle(templates)
        # simple scoring heuristic
        for t in templates:
            score = 0.0
            low = t.lower()
            # presence of tactical keywords
            if any(k in low for k in self.keyword_map['tactical']) and 'tactical' in context_keys:
                score += 2.0
            if any(k in low for k in self.keyword_map['positional']) and 'positional' in context_keys:
                score += 1.5
            if any(k in low for k in self.keyword_map['king_safety']) and 'king_safety' in context_keys:
                score += 2.0
            # favor longer calm sentences for Karpov
            if style == 'karpov' and len(t.split()) > 7:
                score += 0.5
            # Kasparov likes exclamations / shorter high-intensity lines
            if style == 'kasparov' and ('!' in t or len(t.split()) <= 6):
                score += 0.5 * (1 + intensity_base)
            # increase chance if template mentions common words of plan_tag
            if plan_tag and plan_tag.lower() in low:
                score += 1.2
            # add base random factor for variety
            score += random.random() * 0.5
            pool.append((score, t))

        # pick top N candidates by score
        pool.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [p[1] for p in pool[:12]] if pool else []
        # If nothing matched, fallback to short neutral lines
        if not top_candidates:
            top_candidates = generate_templates_for_style(style, n_desired=20)[:8]

        # choose final candidate with small stochasticity influenced by intensity
        if intensity_base > 0.7:
            # prefer more forceful wording
            chosen = top_candidates[0] if top_candidates else random.choice(self.templates[style])
        elif intensity_base > 0.3:
            chosen = random.choice(top_candidates[:4])
        else:
            chosen = random.choice(top_candidates)

        # Now contextualize: insert small, specific clauses based on the state
        chosen = self._contextualize_phrase(chosen, meaning, style)

        # Apply small style postprocessing (avoid numeric mentions, motor jargon)
        chosen = self._postprocess_text(chosen, style)

        # Ensure length constraints and update history for coherence
        final = clamp_text_to_lines(chosen)
        self._update_history(safe_str(state.get('fen')), final)
        return final

    # ---------- Helper internal methods ----------
    def _contextualize_phrase(self, phrase: str, meaning: Dict, style: str) -> str:
        """
        Given a base phrase template, we add small contextual clauses:
        - mention central squares if center control is present
        - mention king danger if threats > 0
        - mention material if big imbalance
        - incorporate the move notation softly if useful (e.g., 'con ...')
        """
        ph = phrase
        # If phrase contains placeholders (rare) we try mild replacement
        ph = ph.replace("{file}", " la columna").replace("{side}", " el flanco dama")
        # Add small clauses
        clauses = []
        md = meaning.get('material_diff', 0.0)
        if abs(md) >= 2.0:
            if md > 0:
                clauses.append("con clara superioridad material")
            else:
                clauses.append("con desventaja material que exige precisión")
        # mobility
        mob = meaning.get('mobility_diff', 0)
        if abs(mob) >= 4:
            if mob > 0:
                clauses.append("aprovechando mayor libertad de movimiento")
            else:
                clauses.append("buscando reducir la movilidad rival")
        # king threats
        if meaning.get('king_threats_b', 0) > 0:
            clauses.append("presionando el enroque enemigo")
        if meaning.get('king_threats_w', 0) > 0:
            clauses.append("defendiendo la seguridad del propio rey")
        # tactical hints
        if 'tactical' in meaning.get('context_keys', []):
            clauses.append("hay motivos tácticos inmediatos")
        # attach move mention softly, avoid engine-speak
        mv = meaning.get('move_str', None)
        if mv:
            mv = mv.replace(" ", "")
            if random.random() < 0.4:
                clauses.append(f"tras la jugada {mv}")
        # combine
        if clauses:
            # prefer to append one or two clauses
            selected = clauses[:2] if len(clauses) > 1 else clauses
            conj = ", ".join(selected)
            # Decide joiner style based on style
            if style == 'kasparov':
                ph = ph.rstrip(".!") + f"; {conj}."
            else:
                ph = ph.rstrip(".!") + f", {conj}."
        return ph

    def _postprocess_text(self, text: str, style: str) -> str:
        """
        Clean text: remove motor jargon, prevent numeric evaluations,
        normalize punctuation, ensure human tone.
        """
        # Remove patterns like 'eval', 'depth', 'minimax', 'SGD', 'surprise'
        bad_tokens = ['eval', 'minimax', 'SGD', 'depth', 'loss', 'surprise', 'precision', 'replay buffer']
        low = text.lower()
        for b in bad_tokens:
            if b in low:
                # replace token with human substitute
                text = text.replace(b, "")
        # trim extra whitespace
        text = ' '.join(text.split())
        # ensure first char uppercase
        if len(text) > 0:
            text = text[0].upper() + text[1:]
        # some style-specific flourish
        if style == 'kasparov' and random.random() < 0.15:
            if not text.endswith("!"):
                text = text.rstrip(".") + "!"
        return text

    def _update_history(self, fen: str, comment: str):
        if fen is None:
            fen = "no_fen"
        self.recent_history.append((fen, comment))
        if len(self.recent_history) > self.max_history:
            self.recent_history.pop(0)

    # ---------- Utility for manual tweaking ----------
    def add_custom_phrase(self, style: str, phrase: str):
        if style not in STYLE_LIST:
            style = DEFAULT_STYLE
        self.custom_phrases[style].append(phrase)

    def set_commentary_mode(self, enabled: bool):
        self.commentary_mode = bool(enabled)

    def seed_templates(self, style: str, templates: List[str]):
        """replace templates for a style (useful to inject handcrafted phrases)"""
        if style not in STYLE_LIST:
            return
        self.templates[style] = templates
